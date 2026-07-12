"""
rag/tutor_llm.py
================
LiveKit-compatible LLM wrapper that routes all student messages
through the LangGraph pipeline instead of calling Groq directly.

Written for livekit-agents==1.3.12.

Key API for this version:
  - LLMStream.__init__(llm, *, chat_ctx, tools: list[Tool], conn_options)
  - LLM.chat(*, chat_ctx, tools, conn_options, ...)
  - _event_ch.send_nowait(ChatChunk(...))
  - ChatChunk(id=str, delta=ChoiceDelta(role=..., content=str))
"""

import asyncio
import json
import logging
from typing import Any, Optional

from livekit.agents import utils as lk_utils
from livekit.agents.llm import (
    LLM,
    LLMStream,
    ChatChunk,
    ChatContext,
    ChoiceDelta,
)
from livekit.agents.llm.llm import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils.aio import ChanClosed

from rag.graph import graph, fast_generate, _classify_by_rules, DUAL_TRACK_INTENTS
from rag.session_state import session_state

logger = logging.getLogger(__name__)

STREAM_CHUNK_SIZE = 20


class TutorLLMStream(LLMStream):

    def __init__(
        self,
        llm: "TutorLLM",
        *,
        chat_ctx: ChatContext,
        tools: list,
        conn_options: APIConnectOptions,
        course: str,
        session_id: str,
        student_name: str,
        tutor_name: str,
        room=None,
    ) -> None:
        super().__init__(
            llm,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._course       = course
        self._session_id   = session_id
        self._student_name = student_name
        self._tutor_name   = tutor_name
        self._room         = room

    async def _publish_code(self, code: str) -> None:
        """Send a code_inject data message to all room participants."""
        if not code or not self._room:
            return
        try:
            payload = json.dumps({"type": "code_inject", "code": code})
            await self._room.local_participant.publish_data(
                payload, topic="code_inject", reliable=True
            )
            logger.info(f"[TutorLLMStream] code_inject published ({len(code)} chars)")
        except Exception as e:
            logger.warning(f"[TutorLLMStream] code_inject publish failed: {e}")

    async def _send_chunks(self, text: str) -> bool:
        """
        Send text to LiveKit TTS in small chunks.
        Returns False if the channel was closed (user interrupted), True otherwise.
        """
        chunk_id = lk_utils.shortuuid("tutor_")
        parts = [text[i: i + STREAM_CHUNK_SIZE]
                 for i in range(0, len(text), STREAM_CHUNK_SIZE)]
        for i, part in enumerate(parts):
            try:
                self._event_ch.send_nowait(
                    ChatChunk(
                        id=chunk_id,
                        delta=ChoiceDelta(role="assistant", content=part),
                    )
                )
            except ChanClosed:
                logger.info("[TutorLLMStream] channel closed — user interrupted")
                return False
            if i < len(parts) - 1:
                await asyncio.sleep(0)
        return True

    async def _run(self) -> None:
        # ── Extract last user message
        message = ""
        for item in reversed(self._chat_ctx.items):
            if item.type == "message" and item.role == "user":
                message = item.text_content or ""
                break
        if not message:
            message = "Hello"

        logger.info(f"[TutorLLMStream] message='{message[:60]}'")

        # ── Load session context from Redis (history, topic, notes)
        history       = session_state.get_history(self._session_id)
        current_topic = session_state.get_current_topic(self._session_id)
        student_notes = session_state.get_student_notes(self._session_id)

        initial_state = {
            "message":              message,
            "course":               self._course,
            "session_id":           self._session_id,
            "student_name":         self._student_name,
            "tutor_name":           self._tutor_name,
            # Context
            "conversation_history": history,
            "current_topic":        current_topic,
            "student_notes":        student_notes,
            # Pipeline defaults
            "intent":            "",
            "chunks":            [],
            "graded_chunks":     [],
            "needs_fallback":    False,
            "response":          "",
            "is_grounded":       True,
            "retry_count":       0,
            "trigger_flashcard": False,
            "simplify":          False,
            "code_snippet":      "",
        }

        # ── Detect intent with rule-based classifier (no LLM, ~0 ms)
        quick_intent = _classify_by_rules(message)
        use_dual_track = quick_intent in DUAL_TRACK_INTENTS

        if use_dual_track:
            # ── DUAL-TRACK: fire 8b and 70b graph concurrently
            logger.info(
                f"[TutorLLMStream] dual-track ON — intent={quick_intent}"
            )
            graph_task = asyncio.create_task(graph.ainvoke(initial_state))
            quick_task = asyncio.create_task(
                fast_generate(
                    message,
                    self._student_name,
                    self._tutor_name,
                    quick_intent,
                )
            )

            # Phase 1 — quick 8b response (~300-500 ms) → TTS starts immediately
            quick_response = await quick_task
            if quick_response:
                logger.info(
                    f"[TutorLLMStream] quick response ready ({len(quick_response)} chars)"
                )
                ok = await self._send_chunks(quick_response)
                if not ok:
                    graph_task.cancel()
                    return
                # Small natural pause between intro and elaboration
                await self._send_chunks(" ")

            # Phase 2 — full 70b grounded response
            try:
                result = await graph_task
            except Exception as e:
                logger.error(f"[TutorLLMStream] graph failed: {e}")
                if not quick_response:
                    await self._send_chunks(
                        "I'm having a moment — could you ask me that again?"
                    )
                return

            full_response = result.get("response", "").strip()
            trigger_flashcard = result.get("trigger_flashcard", False)
            code_snippet = result.get("code_snippet", "")

            if trigger_flashcard:
                logger.info(f"[TutorLLMStream] flashcard — session={self._session_id}")

            # Publish code block to frontend via data channel (before TTS)
            if code_snippet:
                await self._publish_code(code_snippet)

            if full_response:
                logger.info(
                    f"[TutorLLMStream] full response ready ({len(full_response)} chars)"
                )
                await self._send_chunks(full_response)
            elif not quick_response:
                await self._send_chunks(
                    "Could you rephrase that? I want to give you the right answer."
                )

            # ── Persist turn to conversation history
            stored_response = full_response or quick_response
            if stored_response and self._session_id:
                session_state.add_history_turn(
                    self._session_id, message, stored_response
                )

        else:
            # ── SINGLE-TRACK: conversational / flashcard — graph is fast enough
            logger.info(
                f"[TutorLLMStream] single-track — intent={quick_intent or 'unknown'}"
            )
            try:
                result = await graph.ainvoke(initial_state)
            except Exception as e:
                logger.error(f"[TutorLLMStream] graph failed: {e}")
                result = {
                    "response":          "I'm having a moment — could you ask me that again?",
                    "trigger_flashcard": False,
                }

            response_text     = result.get("response", "").strip()
            trigger_flashcard = result.get("trigger_flashcard", False)
            code_snippet      = result.get("code_snippet", "")

            if trigger_flashcard:
                logger.info(f"[TutorLLMStream] flashcard — session={self._session_id}")

            # Publish code block to frontend via data channel (before TTS)
            if code_snippet:
                await self._publish_code(code_snippet)

            if not response_text:
                response_text = "Could you rephrase that? I want to give you the right answer."

            await self._send_chunks(response_text)

            # ── Persist turn to conversation history
            if self._session_id:
                session_state.add_history_turn(
                    self._session_id, message, response_text
                )


class TutorLLM(LLM):

    def __init__(
        self,
        course:       str = "introduction_to_python",
        session_id:   str = "",
        student_name: str = "Student",
        tutor_name:   str = "Tutor",
        room=None,
    ) -> None:
        super().__init__()
        self._course       = course
        self._session_id   = session_id
        self._student_name = student_name
        self._tutor_name   = tutor_name
        self._room         = room

    @property
    def model(self) -> str:
        return "langgraph-tutor"

    @property
    def provider(self) -> str:
        return "emualim"

    def update_session(self, session_id: str, student_name: str) -> None:
        self._session_id   = session_id
        self._student_name = student_name
        session_state.create(
            session_id=session_id,
            student_name=student_name,
            course=self._course,
        )

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: Any = None,
        tool_choice: Any = None,
        extra_kwargs: Any = None,
    ) -> TutorLLMStream:
        return TutorLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            course=self._course,
            session_id=self._session_id,
            student_name=self._student_name,
            tutor_name=self._tutor_name,
            room=self._room,
        )