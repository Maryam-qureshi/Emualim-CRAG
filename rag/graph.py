"""
rag/graph.py
============
E-Mualim LangGraph pipeline — 6 nodes, full RAG + CRAG + hallucination check.

Node order:
    1. intent_classifier  → rule-based first, Groq 8b fallback
    2. rag_retriever      → calls retriever.py based on intent
    3. crag_grader        → scores each chunk, discards below threshold
    4. generator          → Groq 70b, grounded prompt
    5. hallucination_checker → Groq 8b, YES/NO check
    6. response_router    → trim for voice, check engagement, flashcard signal

Models:
    GROQ_API_KEY_LIGHT → llama3-groq-8b-8192  (classifier, grader, checker)
    GROQ_API_KEY_MAIN  → llama-3.3-70b-versatile (generator)

Usage:
    from rag.graph import graph

    result = await graph.ainvoke({
        "message":      "What is a function?",
        "course":       "introduction_to_python",
        "session_id":   "session_20260316_maryam",
        "student_name": "Maryam",
    })
    print(result["response"])
    print(result["trigger_flashcard"])
"""

import os
import re
import logging
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from rag.retriever import (
    retrieve_concept, retrieve_howto, retrieve_mistake,
    retrieve_practice, RetrievedChunk, format_context, build_grounded_prompt,
)
from rag.session_state import session_state

load_dotenv()

logger = logging.getLogger(__name__)

# ── MODELS ────────────────────────────────────────────────────────────────────

def _light_llm(temperature: float = 0.0) -> ChatGroq:
    """Groq 8b — classifier, grader, hallucination checker."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY_LIGHT"),
        model="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=512,
    )


def _main_llm(temperature: float = 0.3) -> ChatGroq:
    """Groq 70b — generator."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY_MAIN"),
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=600,   # keep responses short for voice
    )


# ── STATE ─────────────────────────────────────────────────────────────────────

class TutorState(TypedDict):
    # ── Input (provided before graph starts)
    message:          str
    course:           str
    session_id:       str
    student_name:     str
    tutor_name:       str

    # ── Context injected before graph (from session_state)
    conversation_history: list   # [{human, ai}, ...] last N turns
    current_topic:        str    # main topic of the previous exchange
    student_notes:        list   # [{title, content}, ...] from Postgres

    # ── Node 1 output
    intent:           str      # concept | howto | mistake | practice | flashcard | conversational

    # ── Node 2 output
    chunks:           list     # list of RetrievedChunk

    # ── Node 3 output
    graded_chunks:    list     # filtered RetrievedChunk list
    needs_fallback:   bool     # True = no chunks passed grading

    # ── Node 4 output
    response:         str      # full grounded answer (70b)

    # ── Node 5 output
    is_grounded:      bool
    retry_count:      int      # generator retry counter (max 2)

    # ── Node 6 output
    trigger_flashcard: bool
    simplify:          bool    # True = low engagement detected
    code_snippet:      str     # first code block extracted before TTS trim, or ""


# ── INTENT KEYWORDS ───────────────────────────────────────────────────────────

_INTENT_RULES = {
    # ── Conversational / social — checked FIRST so greetings never hit RAG
    "conversational": [
        "hi ", "hi!", "hi,", "hello", "hey ", "hey!", "hey,",
        "good morning", "good afternoon", "good evening", "good night",
        "how are you", "how r you", "what's up", "whats up", "sup ",
        "nice to meet", "who are you", "what's your name", "whats your name",
        "thank you", "thanks", "thx", "ty ", "ty!", "cheers",
        "got it", "okay", "ok ", "ok!", "alright", "sounds good",
        "i'm ready", "im ready", "let's start", "lets start",
        "great!", "awesome", "cool!", "nice!", "wow",
        "bye", "goodbye", "see you", "talk later",
    ],
    "concept": [
        "what is", "what are", "define", "definition", "explain",
        "tell me about", "describe", "meaning of", "what does",
    ],
    "howto": [
        "how do", "how to", "how can", "show me", "show how",
        "give me an example", "example of", "demonstrate", "walk me through",
        "write a", "write the", "code for",
    ],
    "mistake": [
        "error", "wrong", "not working", "why isn't", "why is it",
        "bug", "traceback", "exception", "fails", "incorrect",
        "what's wrong", "fix", "debug",
    ],
    "practice": [
        "practice", "problem", "exercise", "question", "quiz",
        "test me", "give me a", "challenge", "task",
    ],
    "flashcard": [
        "flashcard", "flash card", "remember this", "save this",
        "create a card", "make a card", "make a flashcard",
    ],
}


def _classify_by_rules(message: str) -> str | None:
    """
    Try to classify intent using keyword rules.
    Returns intent string if matched, None if no match.
    """
    lower = message.lower()
    for intent, keywords in _INTENT_RULES.items():
        if any(kw in lower for kw in keywords):
            return intent
    return None


# ── Vague message patterns — trigger topic expansion when current_topic known
_VAGUE_PATTERNS = [
    "give me an example", "give an example", "show me an example",
    "show me again", "show that again", "explain that", "explain more",
    "tell me more", "more details", "what about that", "how about that",
    "can you elaborate", "elaborate", "go deeper", "go on",
    "continue", "and then what", "more on that", "another example",
    "example of that", "like what", "such as",
]


def _is_vague(message: str) -> bool:
    """Return True if the message is too vague to retrieve without context."""
    lower = message.lower().strip()
    return any(p in lower for p in _VAGUE_PATTERNS)


# ════════════════════════════════════════════════════════════════════════════════
# NODE 1 — INTENT CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════════

def intent_classifier(state: TutorState) -> TutorState:
    """
    Classify the student's message into one of five intents.
    Uses keyword rules first (no LLM call).
    Falls back to Groq 8b if no rule matches.
    """
    message = state["message"]
    logger.info(f"[intent_classifier] message='{message[:60]}'")

    # ── Step 0: expand vague messages using the last known topic
    current_topic = state.get("current_topic", "")
    if current_topic and _is_vague(message):
        expanded = f"{message} about {current_topic}"
        logger.info(f"[intent_classifier] expanded vague → '{expanded[:80]}'")
        state = {**state, "message": expanded}
        message = expanded

    # ── Step 1: keyword rules
    intent = _classify_by_rules(message)
    if intent:
        logger.info(f"[intent_classifier] rule match → {intent}")
        return {**state, "intent": intent}

    # ── Step 2: Groq fallback
    logger.info("[intent_classifier] no rule match → calling Groq 8b")
    llm = _light_llm(temperature=0.0)
    prompt = (
        "Classify the following student message into exactly one of these "
        "categories: conversational, concept, howto, mistake, practice, flashcard.\n\n"
        "Definitions:\n"
        "  conversational = greeting, thanks, small talk, social message, or anything "
        "not related to a specific programming topic\n"
        "  concept   = asking what something is or means\n"
        "  howto     = asking how to do something or for an example\n"
        "  mistake   = reporting an error or asking why something is wrong\n"
        "  practice  = asking for a practice problem or exercise\n"
        "  flashcard = asking to create a flashcard or save something\n\n"
        "When in doubt between conversational and a topic category, prefer conversational.\n\n"
        f"Student message: {message}\n\n"
        "Reply with ONE word only — the category name."
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip().lower()
        # Extract first word in case model adds extra text
        intent = raw.split()[0] if raw else "conversational"
        if intent not in _INTENT_RULES:
            intent = "concept"   # safe default
    except Exception as e:
        logger.warning(f"[intent_classifier] Groq call failed ({e}), defaulting to concept")
        intent = "concept"

    logger.info(f"[intent_classifier] Groq → {intent}")
    return {**state, "intent": intent}


# ════════════════════════════════════════════════════════════════════════════════
# NODE 2 — RAG RETRIEVER
# ════════════════════════════════════════════════════════════════════════════════

def rag_retriever(state: TutorState) -> TutorState:
    """
    Retrieve relevant chunks from MongoDB based on intent.
    Calls retriever.py convenience wrappers.
    """
    intent  = state["intent"]
    course  = state["course"]
    message = state["message"]
    logger.info(f"[rag_retriever] intent={intent}, course={course}")

    try:
        if intent == "concept":
            chunks = retrieve_concept(message, course, top_k=5)
        elif intent == "howto":
            chunks = retrieve_howto(message, course, top_k=5)
        elif intent == "mistake":
            chunks = retrieve_mistake(message, course, top_k=5)
        elif intent == "practice":
            chunks = retrieve_practice(message, course, top_k=3)
        elif intent == "flashcard":
            # flashcard uses concept retrieval — we need content to make the card
            chunks = retrieve_concept(message, course, top_k=3)
        else:
            chunks = retrieve_concept(message, course, top_k=5)
    except Exception as e:
        logger.error(f"[rag_retriever] retrieval failed: {e}")
        chunks = []

    logger.info(f"[rag_retriever] retrieved {len(chunks)} chunks")
    return {**state, "chunks": chunks}


# ════════════════════════════════════════════════════════════════════════════════
# NODE 3 — CRAG GRADER
# ════════════════════════════════════════════════════════════════════════════════

def crag_grader(state: TutorState) -> TutorState:
    """
    Score all retrieved chunks for relevance in a SINGLE LLM call.
    Previously made one call per chunk (up to 5 calls); now batched into one.
    Discard chunks scoring below 6/10.
    Sets needs_fallback=True if all chunks fail.
    """
    chunks  = state.get("chunks", [])
    message = state["message"]
    intent  = state.get("intent", "concept")
    logger.info(f"[crag_grader] grading {len(chunks)} chunks (batched)")

    # Conversational messages don't need RAG — skip grading entirely
    if intent == "conversational":
        logger.info("[crag_grader] conversational intent → skipping RAG grading")
        return {**state, "graded_chunks": [], "needs_fallback": False}

    if not chunks:
        logger.info("[crag_grader] no chunks to grade → fallback")
        return {**state, "graded_chunks": [], "needs_fallback": True}

    # Build a single prompt listing all excerpts numbered 1..N
    excerpts = "\n\n".join(
        f"[{i+1}] {chunk.text[:400].replace(chr(10), ' ')}"
        for i, chunk in enumerate(chunks)
    )
    prompt = (
        f"Rate the relevance of each text excerpt below to the student question "
        f"on a scale of 0-10.\n\n"
        f"Student question: {message}\n\n"
        f"{excerpts}\n\n"
        f"Reply with ONLY the scores as a comma-separated list in order, e.g.: 8,3,7,5,9"
    )

    scores = [0] * len(chunks)   # default all to 0
    try:
        llm = _light_llm(temperature=0.0)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        # Parse comma-separated integers; tolerate extra text by extracting all numbers
        numbers = [max(0, min(10, int(n))) for n in re.findall(r'\d+', raw)]
        for i, score in enumerate(numbers[:len(chunks)]):
            scores[i] = score
    except Exception as e:
        logger.warning(f"[crag_grader] batch scoring failed ({e}), all scores=0")

    graded = []
    for chunk, score in zip(chunks, scores):
        logger.debug(f"[crag_grader] score={score} topic={chunk.topic}")
        if score >= 6:
            graded.append(chunk)

    needs_fallback = len(graded) == 0
    logger.info(
        f"[crag_grader] {len(graded)}/{len(chunks)} chunks passed "
        f"scores={scores} (needs_fallback={needs_fallback})"
    )
    return {**state, "graded_chunks": graded, "needs_fallback": needs_fallback}


# ════════════════════════════════════════════════════════════════════════════════
# NODE 4 — GENERATOR
# ════════════════════════════════════════════════════════════════════════════════

def generator(state: TutorState) -> TutorState:
    """
    Generate Danish's response using Groq 70b.
    Uses grounded prompt with retrieved chunks.
    Adjusts temperature and prompt based on intent and simplify flag.
    """
    intent               = state["intent"]
    needs_fallback       = state.get("needs_fallback", False)
    simplify             = state.get("simplify", False)
    retry_count          = state.get("retry_count", 0)
    student_name         = state.get("student_name", "Student")
    tutor_name           = state.get("tutor_name", "Tutor")
    graded_chunks        = state.get("graded_chunks", [])
    message              = state["message"]
    conversation_history = state.get("conversation_history", [])
    student_notes        = state.get("student_notes", [])

    logger.info(
        f"[generator] intent={intent} fallback={needs_fallback} "
        f"simplify={simplify} retry={retry_count}"
    )

    # ── Format conversation history (last 3 turns for brevity)
    history_section = ""
    if conversation_history:
        recent = conversation_history[-3:]
        lines  = "\n".join(
            f"Student: {t['human']}\nTutor: {t['ai']}" for t in recent
        )
        history_section = f"\n\nRECENT CONVERSATION (for context only):\n{lines}\n"

    # ── Format student notes
    notes_section = ""
    if student_notes:
        note_lines = "\n".join(
            f"- {n['title']}: {n['content'][:200]}" for n in student_notes[:5]
        )
        notes_section = (
            f"\n\nSTUDENT'S SAVED NOTES (use these to calibrate tone and depth):\n"
            f"{note_lines}\n"
        )

    # ── Temperature by intent
    temp_map = {
        "concept":   0.3,
        "howto":     0.3,
        "mistake":   0.2,
        "practice":  0.7,
        "flashcard": 0.4,
    }
    temperature = temp_map.get(intent, 0.3)
    llm = _main_llm(temperature=temperature)

    # ── Build prompt
    if intent == "conversational":
        # Greetings, thanks, small talk — respond warmly without RAG
        prompt = (
            f"You are {tutor_name}, a warm and experienced Python programming tutor. "
            f"You have a natural, human way of talking — never stiff or robotic. "
            f"The student {student_name} just said: '{message}'."
            f"{history_section}"
            f"\nRespond the way a real teacher would in casual conversation — "
            f"naturally and briefly (1-2 sentences). "
            f"If it's a greeting, welcome them by name and leave the door open for their first question. "
            f"If it's thanks or acknowledgment, accept it warmly and offer to keep going. "
            f"If it's small talk, match their energy and gently steer back to learning. "
            f"Never say 'Certainly!', 'Of course!', or 'Great question!' — just be human."
        )
    elif needs_fallback:
        # No relevant chunks found — answer Python basics from LLM knowledge,
        # only redirect if the topic is genuinely outside Python entirely.
        prompt = (
            f"You are {tutor_name}, an experienced and enthusiastic Python programming tutor "
            f"who loves making concepts click for beginners. "
            f"The student {student_name} asked: '{message}'."
            f"{history_section}"
            f"\nThe course retrieval system didn't surface specific material for this question, "
            f"but you know Python fundamentals deeply and should answer directly.\n\n"
            f"RULES:\n"
            f"- If the question touches Python in any way — variables, data types, operators, "
            f"control flow, loops, functions, lists, tuples, dictionaries, sets, strings, "
            f"file I/O, error handling, modules, OOP, or any Python concept — answer it "
            f"confidently and helpfully. Use a clear explanation, a relatable analogy, "
            f"and a short code snippet if it helps. Do NOT say 'this isn't in the material'.\n"
            f"- Only if the question is entirely unrelated to programming should you "
            f"warmly redirect the student and invite a Python question.\n"
            f"Tone: conversational, encouraging, clear. 3-5 sentences."
        )
    else:
        # Normal grounded generation
        base_prompt = build_grounded_prompt(
            question=message,
            chunks=graded_chunks,
            tutor_name=tutor_name,
        )

        # Append history and notes after the base prompt
        context_addons = history_section + notes_section

        # Add simplify instruction if engagement is low
        simplify_instruction = ""
        if simplify:
            simplify_instruction = (
                "\n\nIMPORTANT: The student's engagement appears low. "
                "Use a much simpler explanation. Add a real-world analogy. "
                "Maximum 3 short sentences. Avoid technical jargon."
            )

        # Add strictness instruction on retry
        retry_instruction = ""
        if retry_count > 0:
            retry_instruction = (
                "\n\nIMPORTANT: Stay strictly within the provided course material. "
                "Do not add any information not present in the sources above."
            )

        prompt = base_prompt + context_addons + simplify_instruction + retry_instruction

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
    except Exception as e:
        logger.error(f"[generator] Groq call failed: {e}")
        response_text = (
            "I'm having a moment — could you repeat that? "
            "I want to make sure I give you the right answer."
        )

    logger.info(f"[generator] response length={len(response_text)} chars")
    return {**state, "response": response_text, "retry_count": retry_count}


# ════════════════════════════════════════════════════════════════════════════════
# NODE 5 — HALLUCINATION CHECKER
# ════════════════════════════════════════════════════════════════════════════════

def hallucination_checker(state: TutorState) -> TutorState:
    """
    Check if the generated response stays within the source material.
    Uses Groq 8b — YES/NO answer only.
    Skipped if needs_fallback=True (no source material to check against).
    """
    needs_fallback = state.get("needs_fallback", False)
    retry_count    = state.get("retry_count", 0)
    graded_chunks  = state.get("graded_chunks", [])
    response       = state.get("response", "")
    intent         = state.get("intent", "concept")

    # Skip check if conversational, no source material, or already retried twice
    if intent == "conversational" or needs_fallback or retry_count >= 2:
        logger.info(
            f"[hallucination_checker] skipped "
            f"(fallback={needs_fallback}, retry={retry_count})"
        )
        return {**state, "is_grounded": True}

    logger.info("[hallucination_checker] checking groundedness")
    llm = _light_llm(temperature=0.0)

    # Format source material for comparison
    context = format_context(graded_chunks, max_chars=2000)

    prompt = (
        "You are checking if an AI tutor's answer stays within provided course material.\n\n"
        f"COURSE MATERIAL:\n{context}\n\n"
        f"TUTOR ANSWER:\n{response}\n\n"
        "Does the tutor answer contain information that contradicts or goes significantly "
        "beyond the provided course material?\n"
        "Reply with YES (if it goes beyond) or NO (if it stays within material). "
        "One word only."
    )

    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        verdict = result.content.strip().upper()
        goes_beyond = verdict.startswith("YES")
    except Exception as e:
        logger.warning(f"[hallucination_checker] check failed ({e}), assuming grounded")
        goes_beyond = False

    is_grounded = not goes_beyond
    logger.info(f"[hallucination_checker] is_grounded={is_grounded}")

    if not is_grounded:
        # Increment retry counter for Generator to use on next pass
        return {**state, "is_grounded": False, "retry_count": retry_count + 1}

    return {**state, "is_grounded": True}


# ════════════════════════════════════════════════════════════════════════════════
# NODE 5b — RESPONSE EXPANDER
# ════════════════════════════════════════════════════════════════════════════════

# Intents where citation-based expansion adds value
_EXPANDABLE_INTENTS = {"concept", "howto", "mistake", "practice"}

# Expansion prompt per intent — tells the 8b what kind of enrichment to pull
_EXPANSION_INSTRUCTIONS: dict[str, str] = {
    "concept": (
        "Enrich the answer by naturally weaving in (from the source material where available):\n"
        "- A concrete, relatable analogy or a short code snippet that makes the concept tangible\n"
        "- One sentence on *why* this matters in real Python code — give it stakes\n"
        "- One common beginner trap or misconception to gently warn the student about\n"
        "Write this as a natural continuation of the answer above, not as a list."
    ),
    "howto": (
        "Enrich the answer by naturally weaving in (from the source material where available):\n"
        "- The clearest code example available — then walk through what each key line does in plain English\n"
        "- One edge case, gotcha, or subtle behaviour the student should know about\n"
        "Write this as a natural continuation of the answer above, not as a list."
    ),
    "mistake": (
        "Enrich the answer by naturally weaving in (from the source material where available):\n"
        "- The corrected version of the code with a one-line explanation of the fix\n"
        "- Why this error happens (the mental model that was wrong) and one tip to avoid it next time\n"
        "Write this as a natural continuation of the answer above, not as a list."
    ),
    "practice": (
        "Enrich the answer by naturally weaving in (from the source material where available):\n"
        "- A gentle hint or starting point — enough to unstick the student without giving it away\n"
        "- Which concept this problem is really testing, and one common misstep to watch for\n"
        "Write this as a natural continuation of the answer above, not as a list."
    ),
}


def response_expander(state: TutorState) -> TutorState:
    """
    Citation-driven elaboration pass (Node 5b).

    Takes the grounded generator response and the CRAG-graded source chunks,
    then asks the 8b model to extract and append:
      • A concrete code example or analogy from the source
      • A 'why it matters' sentence
      • A common beginner mistake

    Skipped when:
      - intent is conversational or flashcard (no source material to expand from)
      - needs_fallback is True (no graded chunks)
      - retry_count > 0 (simplify/hallucination retry loop — keep latency low)
    """
    intent        = state.get("intent", "concept")
    needs_fallback = state.get("needs_fallback", False)
    retry_count   = state.get("retry_count", 0)
    graded_chunks = state.get("graded_chunks", [])
    response      = state.get("response", "")
    student_name  = state.get("student_name", "Student")
    tutor_name    = state.get("tutor_name", "Tutor")
    message       = state["message"]

    skip = (
        intent not in _EXPANDABLE_INTENTS
        or needs_fallback
        or retry_count > 0
        or not graded_chunks
        or not response
    )
    if skip:
        logger.info(
            f"[response_expander] skipped "
            f"(intent={intent}, fallback={needs_fallback}, retry={retry_count})"
        )
        return state

    logger.info(f"[response_expander] expanding response for intent={intent}")

    # Use full chunk text for richer extraction (not the 400-char excerpt used in grader)
    context = format_context(graded_chunks, max_chars=2000)
    instructions = _EXPANSION_INSTRUCTIONS.get(intent, _EXPANSION_INSTRUCTIONS["concept"])

    prompt = (
        f"You are {tutor_name}, an experienced Python tutor mid-conversation with {student_name}.\n\n"
        f"You just answered this question: '{message}'\n\n"
        f"YOUR EXISTING ANSWER:\n{response}\n\n"
        f"SOURCE MATERIAL:\n{context}\n\n"
        f"Now write one natural follow-on paragraph that enriches the answer above.\n"
        f"{instructions}\n\n"
        f"STRICT RULES:\n"
        f"- Draw from the SOURCE MATERIAL; supplement with Python knowledge only where the "
        f"material is thin — never invent facts\n"
        f"- Do NOT repeat anything from the existing answer\n"
        f"- Do NOT open with 'Additionally', 'Furthermore', 'Also', or 'Moreover' — "
        f"find a more natural transition that flows from the answer above "
        f"(e.g. 'Here's where it gets interesting...', 'The thing to watch out for...', "
        f"'A good way to picture this...')\n"
        f"- Match the warm, conversational tutor voice of the existing answer\n"
        f"- Flowing prose only — no bullet points, no headers\n\n"
        f"ENRICHMENT PARAGRAPH:"
    )

    llm = _light_llm(temperature=0.3)   # 8b — fast, targeted task
    try:
        result   = llm.invoke([HumanMessage(content=prompt)])
        enrichment = result.content.strip()

        if enrichment:
            expanded = response.rstrip(". ") + ". " + enrichment
            logger.info(
                f"[response_expander] added {len(enrichment)} chars of enrichment"
            )
            return {**state, "response": expanded}

    except Exception as exc:
        logger.warning(f"[response_expander] 8b call failed: {exc}")

    return state   # fall through to router with original response unchanged


# ════════════════════════════════════════════════════════════════════════════════
# NODE 6 — RESPONSE ROUTER
# ════════════════════════════════════════════════════════════════════════════════

def response_router(state: TutorState) -> TutorState:
    """
    Final processing before response goes to LiveKit TTS.

    1. Flashcard signal — set trigger_flashcard if intent == flashcard
    2. Voice length trim — cap at 400 chars for natural TTS output
    3. Engagement check — read from SessionState, set simplify if low
       (simplify triggers a loop back to Generator — one loop max)
    """
    intent        = state["intent"]
    response      = state.get("response", "")
    session_id    = state.get("session_id", "")
    simplify      = state.get("simplify", False)
    graded_chunks = state.get("graded_chunks", [])
    message       = state.get("message", "")

    logger.info(f"[response_router] intent={intent} len={len(response)}")

    # ── Step 0: Extract code block BEFORE any trimming
    _CODE_RE = re.compile(r'```(?:\w+)?\n?([\s\S]*?)```')
    _code_match = _CODE_RE.search(response)
    code_snippet = _code_match.group(1).strip() if _code_match else ""
    if code_snippet:
        logger.info(f"[response_router] code_snippet extracted ({len(code_snippet)} chars)")

    # ── Step 1: Flashcard signal
    trigger_flashcard = (intent == "flashcard")
    if trigger_flashcard:
        logger.info("[response_router] flashcard signal set")

    # ── Step 2: Voice length trim
    # Expandable intents (concept/howto/mistake/practice) get 900 chars so the
    # response_expander's enrichment paragraph (~200-300 chars) survives TTS trim.
    # Conversational/flashcard keep the tighter 400-char limit for natural brevity.
    tts_limit = 900 if intent in _EXPANDABLE_INTENTS else 400

    if len(response) > tts_limit:
        trimmed = response[:tts_limit]
        last_period = max(
            trimmed.rfind(". "),
            trimmed.rfind("! "),
            trimmed.rfind("? "),
        )
        if last_period > 100:   # don't trim to nothing
            response = trimmed[:last_period + 1]
        else:
            response = trimmed.rstrip() + "..."
        logger.info(f"[response_router] trimmed to {len(response)} chars")

    # ── Step 3: Engagement check
    # Only trigger simplify if not already simplified (prevents infinite loop)
    should_simplify = False
    if not simplify and session_id:
        avg_engagement = session_state.get_avg_engagement(session_id)
        emotion        = session_state.get_latest_emotion(session_id)
        logger.info(
            f"[response_router] avg_engagement={avg_engagement:.2f} "
            f"emotion={emotion}"
        )
        if avg_engagement < 0.5:
            logger.info(
                "[response_router] low engagement detected → triggering simplify"
            )
            should_simplify = True

    # ── Step 4: Topic tracking — store the topic of this exchange for
    # vague follow-up rewriting ("give me an example" → "… of for loops")
    if intent not in ("conversational", "flashcard") and session_id:
        if graded_chunks:
            # Best signal: topic field of the highest-ranked graded chunk
            topic = graded_chunks[0].topic or message[:80]
        else:
            topic = message[:80]
        session_state.set_current_topic(session_id, topic)

    return {
        **state,
        "response":          response,
        "trigger_flashcard": trigger_flashcard,
        "simplify":          should_simplify,
        "code_snippet":      code_snippet,
    }


# ════════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def route_after_checker(state: TutorState) -> str:
    """
    After hallucination checker:
    - If not grounded and retries remain → back to generator
    - If grounded (or max retries reached) → response_expander
    """
    if not state.get("is_grounded", True) and state.get("retry_count", 0) < 2:
        return "generator"
    return "response_expander"


def route_after_router(state: TutorState) -> str:
    """
    After response router:
    - If simplify was just triggered → back to generator
    - Otherwise → END
    """
    if state.get("simplify", False) and state.get("retry_count", 0) == 0:
        return "generator"
    return END


# ════════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ════════════════════════════════════════════════════════════════════════════════

def _build_graph():
    builder = StateGraph(TutorState)

    # ── Add nodes
    builder.add_node("intent_classifier",    intent_classifier)
    builder.add_node("rag_retriever",        rag_retriever)
    builder.add_node("crag_grader",          crag_grader)
    builder.add_node("generator",            generator)
    builder.add_node("hallucination_checker", hallucination_checker)
    builder.add_node("response_expander",    response_expander)
    builder.add_node("response_router",      response_router)

    # ── Entry point
    builder.set_entry_point("intent_classifier")

    # ── Linear edges
    builder.add_edge("intent_classifier",    "rag_retriever")
    builder.add_edge("rag_retriever",        "crag_grader")
    builder.add_edge("crag_grader",          "generator")
    builder.add_edge("generator",            "hallucination_checker")
    builder.add_edge("response_expander",    "response_router")

    # ── Conditional: after hallucination checker
    builder.add_conditional_edges(
        "hallucination_checker",
        route_after_checker,
        {
            "response_expander": "response_expander",
            "generator":         "generator",
        },
    )

    # ── Conditional: after response router
    builder.add_conditional_edges(
        "response_router",
        route_after_router,
        {
            "generator": "generator",
            END:         END,
        },
    )

    return builder.compile()


# Compile once at import time — reused for all invocations
graph = _build_graph()


# ════════════════════════════════════════════════════════════════════════════════
# FAST GENERATOR — 8b direct call, no RAG, fired in parallel with the graph
# Returns a 1-sentence hook so TTS can start immediately (~300-500 ms)
# ════════════════════════════════════════════════════════════════════════════════

# Intents that benefit from dual-track (skip for conversational / flashcard)
DUAL_TRACK_INTENTS: frozenset[str] = frozenset({"concept", "howto", "mistake", "practice"})

_INTENT_HINTS: dict[str, str] = {
    "concept":  "Open with one clear, plain-English sentence that captures the essence of the concept — like the first thing a great teacher would say before diving deeper.",
    "howto":    "Open with one confident sentence that frames the approach at a high level — set the student up for the detailed explanation that's coming.",
    "mistake":  "Open with one direct sentence that names the most likely cause of the problem — give the student the 'aha' moment right away.",
    "practice": "Open with one engaging sentence that frames what the practice problem is about and why it's a useful thing to try.",
}


async def fast_generate(
    message:      str,
    student_name: str,
    tutor_name:   str,
    intent:       str | None = None,
) -> str:
    """
    Fire an 8b LLM call directly — no RAG, no graph overhead.
    Returns a 1-sentence intro answer so TTS can start within ~300-500 ms
    while the full 70b grounded response is still being generated.

    Called in parallel with graph.ainvoke() from TutorLLMStream._run().
    """
    hint = _INTENT_HINTS.get(intent or "", "Answer in 1 clear, direct sentence.")
    prompt = (
        f"You are {tutor_name}, a warm and knowledgeable Python tutor. "
        f"The student {student_name} asked: '{message}'.\n\n"
        f"{hint} "
        f"A fuller explanation is coming right after yours, so this is purely the opening hook — "
        f"make it count. Do NOT say 'let me explain', 'great question', 'certainly', or anything "
        f"like that. No filler. Just a crisp, confident, human opening sentence."
    )
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY_LIGHT"),
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=80,
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as exc:
        logger.warning(f"[fast_generate] 8b call failed: {exc}")
        return ""   # caller falls back to full response only


# ════════════════════════════════════════════════════════════════════════════════
# QUICK TEST (run directly: python rag/graph.py)
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    async def test():
        test_cases = [
            ("Explain inheritance to me?",           "concept"),
            ("Show me how to use a for loop",        "howto"),
            ("I keep getting a TypeError, help",     "mistake"),
            ("Give me a practice problem on lists",  "practice"),
            ("Make a flashcard of what we just covered", "flashcard"),
        ]

        for message, expected_intent in test_cases:
            print(f"\n{'='*60}")
            print(f"  Message : {message}")
            print(f"  Expected: {expected_intent}")

            result = await graph.ainvoke({
                "message":      message,
                "course":       "introduction_to_python",
                "session_id":   "test_session_001",
                "student_name": "Maryam",
                "tutor_name":   "Tutor",
                # Context
                "conversation_history": [],
                "current_topic":        "",
                "student_notes":        [],
                # Pipeline defaults
                "intent":          "",
                "chunks":          [],
                "graded_chunks":   [],
                "needs_fallback":  False,
                "response":        "",
                "is_grounded":     True,
                "retry_count":     0,
                "trigger_flashcard": False,
                "simplify":        False,
            })

            print(f"  Intent  : {result.get('intent')}")
            print(f"  Chunks  : {len(result.get('graded_chunks', []))} passed grading")
            print(f"  Fallback: {result.get('needs_fallback')}")
            print(f"  Flashcard signal: {result.get('trigger_flashcard')}")
            print(f"  Response:\n    {result.get('response', '')}")

    asyncio.run(test())