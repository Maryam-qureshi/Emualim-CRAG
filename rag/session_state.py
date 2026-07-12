"""
rag/session_state.py
====================
Shared store for behavior-module engagement scores.

Redis-backed when available, falls back to an in-memory dict otherwise.
The public API is identical to the original — callers (monitoring_routes.py
and graph.py) do not need to change.

Redis layout
------------
  engagement:{session_id}:readings  — Redis list, JSON strings, capped at 5
  engagement:{session_id}:meta      — Redis hash  {student_name, course}
  Both keys share a 24-hour TTL, refreshed on every write.

Rolling window
--------------
  RPUSH + LTRIM(-5, -1) is atomic in Redis, so the "last 5 readings" window
  is maintained without any application-level locking.

Fallback
--------
  If Redis is not available (init_redis() was not called or failed),
  every method silently falls back to the in-memory dict so the server
  still works in single-process development mode.
"""

import json
import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_WINDOW      = 5        # rolling window size
_SESSION_TTL = 86_400   # 24 hours in seconds


# ── Data classes (kept for type-hinting and in-memory fallback) ────────────────

@dataclass
class EngagementRecord:
    """One reading from the behavior module."""
    engagement: float   # 0.0 – 1.0
    emotion:    str     # happy | neutral | sad | confused | angry | fear
    timestamp:  float   # time.time()


@dataclass
class SessionData:
    """All behavior data for one active session (in-memory fallback)."""
    session_id:   str
    student_name: str
    course:       str
    readings:     deque = field(default_factory=lambda: deque(maxlen=_WINDOW))

    def avg_engagement(self) -> float:
        if not self.readings:
            return 1.0
        return sum(r.engagement for r in self.readings) / len(self.readings)

    def latest_emotion(self) -> str:
        return self.readings[-1].emotion if self.readings else "neutral"

    def latest_engagement(self) -> float:
        return self.readings[-1].engagement if self.readings else 1.0


# ── Store ──────────────────────────────────────────────────────────────────────

class SessionStateStore:
    """
    Thread-safe engagement store. Redis-backed with in-memory fallback.
    Import the module-level singleton ``session_state`` — do not instantiate
    this class directly.
    """

    def __init__(self):
        self._sessions: dict[str, SessionData] = {}   # in-memory fallback
        self._lock = threading.Lock()

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rk(session_id: str) -> str:
        """Redis key for the readings list."""
        return f"engagement:{session_id}:readings"

    @staticmethod
    def _mk(session_id: str) -> str:
        """Redis key for the meta hash."""
        return f"engagement:{session_id}:meta"

    @staticmethod
    def _get_r():
        """Return the Redis client or None."""
        try:
            from core.redis_client import _redis   # noqa: PLC0415
            return _redis
        except ImportError:
            return None

    # ── public API ────────────────────────────────────────────────────────────

    def create(
        self,
        session_id:   str,
        student_name: str = "Student",
        course:       str = "introduction_to_python",
    ) -> None:
        """Register a new session. Safe to call multiple times."""
        r = self._get_r()
        if r:
            try:
                mk = self._mk(session_id)
                if not r.exists(mk):
                    r.hset(mk, mapping={
                        "student_name": student_name,
                        "course":       course,
                    })
                    r.expire(mk, _SESSION_TTL)
                return
            except Exception as e:
                logger.warning("session_state.create Redis error: %s", e)

        # Memory fallback
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData(
                    session_id=session_id,
                    student_name=student_name,
                    course=course,
                )

    def update(
        self,
        session_id: str,
        engagement: float,
        emotion:    str = "neutral",
    ) -> None:
        """
        Called by the behavior loop every ~2 s with a new reading.
        Creates the session entry if it does not exist yet.
        """
        record = {
            "engagement": max(0.0, min(1.0, engagement)),
            "emotion":    emotion,
            "timestamp":  time.time(),
        }
        r = self._get_r()
        if r:
            try:
                rk = self._rk(session_id)
                mk = self._mk(session_id)
                # Ensure meta exists
                if not r.exists(mk):
                    r.hset(mk, mapping={
                        "student_name": "Student",
                        "course":       "introduction_to_python",
                    })
                    r.expire(mk, _SESSION_TTL)
                # Atomic push + trim to last _WINDOW items
                r.rpush(rk, json.dumps(record))
                r.ltrim(rk, -_WINDOW, -1)
                r.expire(rk, _SESSION_TTL)
                return
            except Exception as e:
                logger.warning("session_state.update Redis error: %s", e)

        # Memory fallback
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData(
                    session_id=session_id,
                    student_name="Student",
                    course="introduction_to_python",
                )
            self._sessions[session_id].readings.append(
                EngagementRecord(**record)
            )

    def get_avg_engagement(self, session_id: str) -> float:
        """
        Returns rolling average engagement.
        Returns 1.0 (high) if session not found — safe default so
        LangGraph never simplifies without a real signal.
        """
        r = self._get_r()
        if r:
            try:
                items = r.lrange(self._rk(session_id), 0, -1)
                if not items:
                    return 1.0
                readings = [json.loads(i) for i in items]
                return sum(x["engagement"] for x in readings) / len(readings)
            except Exception as e:
                logger.warning("session_state.get_avg_engagement Redis error: %s", e)

        with self._lock:
            session = self._sessions.get(session_id)
            return session.avg_engagement() if session else 1.0

    def get_latest_emotion(self, session_id: str) -> str:
        """Returns the most recently detected emotion."""
        r = self._get_r()
        if r:
            try:
                item = r.lindex(self._rk(session_id), -1)
                if item:
                    return json.loads(item).get("emotion", "neutral")
                return "neutral"
            except Exception as e:
                logger.warning("session_state.get_latest_emotion Redis error: %s", e)

        with self._lock:
            session = self._sessions.get(session_id)
            return session.latest_emotion() if session else "neutral"

    def get_snapshot(self, session_id: str) -> Optional[dict]:
        """Full snapshot for debugging. Returns None if session not found."""
        r = self._get_r()
        if r:
            try:
                rk = self._rk(session_id)
                mk = self._mk(session_id)
                items = r.lrange(rk, 0, -1)
                meta  = r.hgetall(mk)
                if not meta:
                    return None
                readings = [json.loads(i) for i in items]
                avg     = (sum(x["engagement"] for x in readings) / len(readings)
                           if readings else 1.0)
                emotion = readings[-1]["emotion"] if readings else "neutral"
                return {
                    "session_id":     session_id,
                    "student_name":   meta.get("student_name", "Student"),
                    "course":         meta.get("course", ""),
                    "avg_engagement": avg,
                    "latest_emotion": emotion,
                    "reading_count":  len(readings),
                    "readings":       readings,
                }
            except Exception as e:
                logger.warning("session_state.get_snapshot Redis error: %s", e)

        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            return {
                "session_id":     session.session_id,
                "student_name":   session.student_name,
                "course":         session.course,
                "avg_engagement": session.avg_engagement(),
                "latest_emotion": session.latest_emotion(),
                "reading_count":  len(session.readings),
                "readings": [
                    {
                        "engagement": rec.engagement,
                        "emotion":    rec.emotion,
                        "timestamp":  rec.timestamp,
                    }
                    for rec in session.readings
                ],
            }

    # ── Conversation history ──────────────────────────────────────────────────

    _HISTORY_WINDOW = 5   # rolling window: keep last 5 turns

    @staticmethod
    def _hk(session_id: str) -> str:
        return f"conv:history:{session_id}"

    @staticmethod
    def _topic_key(session_id: str) -> str:
        return f"topic:{session_id}"

    @staticmethod
    def _notes_key(session_id: str) -> str:
        return f"notes:{session_id}"

    def add_history_turn(self, session_id: str, human_msg: str, ai_msg: str) -> None:
        """Append a (human, ai) exchange to the rolling conversation history."""
        turn = json.dumps({"human": human_msg[:500], "ai": ai_msg[:500]})
        r = self._get_r()
        if r:
            try:
                key = self._hk(session_id)
                r.rpush(key, turn)
                r.ltrim(key, -self._HISTORY_WINDOW, -1)
                r.expire(key, _SESSION_TTL)
                return
            except Exception as e:
                logger.warning("add_history_turn Redis error: %s", e)
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                if not hasattr(s, '_history'):
                    s._history = []
                s._history.append({"human": human_msg[:500], "ai": ai_msg[:500]})
                s._history = s._history[-self._HISTORY_WINDOW:]

    def get_history(self, session_id: str) -> list:
        """Return last N turns as list of {human, ai} dicts."""
        r = self._get_r()
        if r:
            try:
                items = r.lrange(self._hk(session_id), 0, -1)
                return [json.loads(i) for i in items]
            except Exception as e:
                logger.warning("get_history Redis error: %s", e)
        with self._lock:
            s = self._sessions.get(session_id)
            return list(getattr(s, '_history', [])) if s else []

    # ── Topic tracking ────────────────────────────────────────────────────────

    def set_current_topic(self, session_id: str, topic: str) -> None:
        """Store the main topic of the last substantive exchange."""
        r = self._get_r()
        if r:
            try:
                r.set(self._topic_key(session_id), topic[:200], ex=_SESSION_TTL)
                return
            except Exception as e:
                logger.warning("set_current_topic Redis error: %s", e)
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s._current_topic = topic

    def get_current_topic(self, session_id: str) -> str:
        """Return the last tracked topic, or '' if none."""
        r = self._get_r()
        if r:
            try:
                val = r.get(self._topic_key(session_id))
                return val if val else ""
            except Exception as e:
                logger.warning("get_current_topic Redis error: %s", e)
        with self._lock:
            s = self._sessions.get(session_id)
            return getattr(s, '_current_topic', '') if s else ''

    # ── Student notes context ─────────────────────────────────────────────────

    def set_student_notes(self, session_id: str, notes: list) -> None:
        """Cache the student's course notes (list of {title, content})."""
        r = self._get_r()
        if r:
            try:
                r.set(self._notes_key(session_id), json.dumps(notes), ex=_SESSION_TTL)
                return
            except Exception as e:
                logger.warning("set_student_notes Redis error: %s", e)
        with self._lock:
            s = self._sessions.setdefault(session_id, SessionData(
                session_id=session_id,
                student_name="Student",
                course="introduction_to_python",
            ))
            s._student_notes = notes

    def get_student_notes(self, session_id: str) -> list:
        """Return the student's cached notes, or [] if none."""
        r = self._get_r()
        if r:
            try:
                val = r.get(self._notes_key(session_id))
                return json.loads(val) if val else []
            except Exception as e:
                logger.warning("get_student_notes Redis error: %s", e)
        with self._lock:
            s = self._sessions.get(session_id)
            return list(getattr(s, '_student_notes', [])) if s else []

    # ── Focus alert helpers ───────────────────────────────────────────────────

    @staticmethod
    def _streak_key(session_id: str) -> str:
        return f"focus:streak:{session_id}"

    @staticmethod
    def _alert_key(session_id: str) -> str:
        return f"focus:alert:{session_id}"

    def increment_distraction_streak(self, session_id: str) -> int:
        """
        Increment NOT_FOCUSED streak counter.
        Returns the new streak value.
        """
        r = self._get_r()
        if r:
            try:
                val = r.incr(self._streak_key(session_id))
                r.expire(self._streak_key(session_id), _SESSION_TTL)
                return int(val)
            except Exception as e:
                logger.warning("increment_distraction_streak Redis error: %s", e)

        with self._lock:
            s = self._sessions.setdefault(session_id, SessionData(
                session_id=session_id, student_name="Student",
                course="introduction_to_python"
            ))
            if not hasattr(s, '_distraction_streak'):
                s._distraction_streak = 0
            s._distraction_streak += 1
            return s._distraction_streak

    def reset_distraction_streak(self, session_id: str) -> None:
        """Reset streak to 0 (called on REFOCUSED or after alert fires)."""
        r = self._get_r()
        if r:
            try:
                r.delete(self._streak_key(session_id))
                return
            except Exception as e:
                logger.warning("reset_distraction_streak Redis error: %s", e)

        with self._lock:
            s = self._sessions.get(session_id)
            if s and hasattr(s, '_distraction_streak'):
                s._distraction_streak = 0

    def set_focus_alert(self, session_id: str, message: str) -> None:
        """
        Store a focus reminder for the LiveKit agent to speak.
        The agent's background watcher picks this up and calls generate_reply().
        TTL of 120s — if the agent doesn't pick it up quickly, discard it.
        """
        r = self._get_r()
        if r:
            try:
                r.set(self._alert_key(session_id), message, ex=120)
                return
            except Exception as e:
                logger.warning("set_focus_alert Redis error: %s", e)

        with self._lock:
            s = self._sessions.setdefault(session_id, SessionData(
                session_id=session_id, student_name="Student",
                course="introduction_to_python"
            ))
            s._focus_alert = message

    def get_and_clear_focus_alert(self, session_id: str) -> Optional[str]:
        """
        Atomically get + delete the pending focus alert.
        Returns None if no alert is pending.
        Called by the LiveKit agent background watcher every 5 s.
        """
        r = self._get_r()
        if r:
            try:
                key = self._alert_key(session_id)
                msg = r.get(key)
                if msg:
                    r.delete(key)
                return msg if msg else None
            except Exception as e:
                logger.warning("get_and_clear_focus_alert Redis error: %s", e)

        with self._lock:
            s = self._sessions.get(session_id)
            if s and hasattr(s, '_focus_alert') and s._focus_alert:
                msg = s._focus_alert
                s._focus_alert = ""
                return msg
        return None

    # ── Session event log ─────────────────────────────────────────────────────

    _LOG_WINDOW = 200   # keep last 200 entries per session

    @staticmethod
    def _logs_key(session_id: str) -> str:
        return f"session_logs:{session_id}"

    def log_event(self, session_id: str, event_type: str, data: dict) -> None:
        """Append a structured log entry to the per-session event list."""
        entry = json.dumps({
            "event":     event_type,
            "timestamp": time.time(),
            **{k: v for k, v in data.items() if v is not None},
        })
        r = self._get_r()
        if r:
            try:
                key = self._logs_key(session_id)
                r.rpush(key, entry)
                r.ltrim(key, -self._LOG_WINDOW, -1)
                r.expire(key, _SESSION_TTL)
                return
            except Exception as e:
                logger.warning("log_event Redis error: %s", e)

        with self._lock:
            s = self._sessions.setdefault(session_id, SessionData(
                session_id=session_id, student_name="Student",
                course="introduction_to_python",
            ))
            if not hasattr(s, '_event_log'):
                s._event_log = []
            s._event_log.append(json.loads(entry))
            s._event_log = s._event_log[-self._LOG_WINDOW:]

    def get_session_logs(self, session_id: str) -> list:
        """Return all stored log entries for the session, newest last."""
        r = self._get_r()
        if r:
            try:
                items = r.lrange(self._logs_key(session_id), 0, -1)
                return [json.loads(i) for i in items]
            except Exception as e:
                logger.warning("get_session_logs Redis error: %s", e)

        with self._lock:
            s = self._sessions.get(session_id)
            return list(getattr(s, '_event_log', [])) if s else []

    def close(self, session_id: str) -> None:
        """Remove a session. Call from stop_monitoring."""
        r = self._get_r()
        if r:
            try:
                r.delete(self._rk(session_id), self._mk(session_id))
                return
            except Exception as e:
                logger.warning("session_state.close Redis error: %s", e)

        with self._lock:
            self._sessions.pop(session_id, None)

    def active_sessions(self) -> list[str]:
        """Returns list of active session IDs. For debugging."""
        r = self._get_r()
        if r:
            try:
                keys = r.keys("engagement:*:meta")
                return [k.split(":")[1] for k in keys]
            except Exception as e:
                logger.warning("session_state.active_sessions Redis error: %s", e)

        with self._lock:
            return list(self._sessions.keys())


# ── Singleton ──────────────────────────────────────────────────────────────────

# Import this everywhere — do not instantiate SessionStateStore directly.
session_state = SessionStateStore()
