"""
rag/retriever.py
================
RAG retrieval module. Called by LangGraph nodes — not exposed
as an API endpoint directly.

Usage (from a LangGraph node):

    from rag.retriever import retrieve, RetrievedChunk

    chunks = retrieve(
        question   = "What is the difference between a parameter and an argument?",
        course     = "introduction_to_python",
        chunk_types = ["concept", "analogy"],
        top_k      = 5,
    )
    for c in chunks:
        print(c.score, c.type, c.text[:80])

The Atlas Vector Search index must exist before calling retrieve().
Create it in Atlas UI after running ingest.py.
"""

import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
from pymongo import MongoClient

from rag.embedder import embed_query

load_dotenv()

logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────

DB_NAME          = "emualim"
COLLECTION       = "chunks"
VECTOR_INDEX     = "vector_index"       # name given to index in Atlas UI
MIN_SCORE        = 0.65                 # discard chunks below this cosine similarity
DEFAULT_TOP_K    = 5
CANDIDATE_FACTOR = 10                   # numCandidates = top_k * CANDIDATE_FACTOR


# ── DATA CLASS ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A single retrieved chunk returned by retrieve().
    LangGraph nodes consume a list of these.
    """
    text:       str
    score:      float          # cosine similarity 0-1
    topic:      str
    type:       str            # concept | code_example | common_mistake | etc.
    difficulty: str
    chapter:    str
    section:    str
    has_code:   bool
    source_pdf: str


# ── CONNECTION ────────────────────────────────────────────────────────────────

_client     = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise EnvironmentError("MONGODB_URI not found in environment.")
        _client     = MongoClient(uri, serverSelectionTimeoutMS=8000)
        _collection = _client[DB_NAME][COLLECTION]
        logger.info("MongoDB connection established (retriever)")
    return _collection


# ── FILTER BUILDER ────────────────────────────────────────────────────────────

def _build_filter(
    course:      str,
    chunk_types: list[str] | None = None,
    difficulty:  str | None       = None,
    chapter:     str | None       = None,
) -> dict:
    """
    Build the MongoDB filter dict for $vectorSearch.
    Only fields declared as 'filter' type in the Atlas index can be used here.
    """
    f = {"metadata.course": {"$eq": course}}

    if chunk_types:
        if len(chunk_types) == 1:
            f["metadata.type"] = {"$eq": chunk_types[0]}
        else:
            f["metadata.type"] = {"$in": chunk_types}

    if difficulty:
        f["metadata.difficulty"] = {"$eq": difficulty}

    if chapter:
        f["metadata.chapter"] = {"$eq": chapter}

    return f


# ── CORE RETRIEVAL ────────────────────────────────────────────────────────────

def retrieve(
    question:    str,
    course:      str                = "introduction_to_python",
    chunk_types: list[str] | None   = None,
    difficulty:  str | None         = None,
    chapter:     str | None         = None,
    top_k:       int                = DEFAULT_TOP_K,
    min_score:   float              = MIN_SCORE,
) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for a question.

    Parameters
    ----------
    question    : The student's question or intent description.
    course      : Course identifier — must match metadata.course in MongoDB.
    chunk_types : Optional list of chunk types to restrict retrieval to.
                  e.g. ["concept", "analogy"] for explanations
                       ["code_example"] for how-to questions
                       ["common_mistake"] for error correction
                       ["practice_problem"] for exercises
    difficulty  : Optional filter — "beginner", "intermediate", "advanced".
    chapter     : Optional filter — "1", "2", etc.
    top_k       : Maximum number of chunks to return.
    min_score   : Minimum cosine similarity — chunks below this are discarded.

    Returns
    -------
    List of RetrievedChunk sorted by score descending.
    Returns empty list if Atlas index is missing or no results pass min_score.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    collection = _get_collection()

    # ── 1. Embed the query
    try:
        query_vector = embed_query(question)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return []

    # ── 2. Build $vectorSearch pipeline
    search_filter = _build_filter(course, chunk_types, difficulty, chapter)

    pipeline = [
        {
            "$vectorSearch": {
                "index":        VECTOR_INDEX,
                "path":         "embedding",
                "queryVector":  query_vector,
                "numCandidates": top_k * CANDIDATE_FACTOR,
                "limit":        top_k,
                "filter":       search_filter,
            }
        },
        {
            "$project": {
                "text":       1,
                "metadata":   1,
                "source_pdf": 1,
                "score": {"$meta": "vectorSearchScore"},
                # Note: embedding field is simply not included here
                # MongoDB does not allow mixing inclusion and exclusion
                # in the same $project — omitting a field excludes it
            }
        },
    ]

    # ── 3. Execute
    try:
        results = list(collection.aggregate(pipeline))
    except Exception as e:
        # Common cause: vector search index does not exist yet
        if "index" in str(e).lower() or "vectorSearch" in str(e):
            logger.error(
                "Atlas Vector Search index not found. "
                "Create it in Atlas UI → Search → Create Index. "
                "See the JSON printed at the end of ingest.py."
            )
        else:
            logger.error(f"Vector search failed: {e}")
        return []

    # ── 4. Filter by minimum score and convert to dataclass
    chunks = []
    for doc in results:
        score = doc.get("score", 0.0)
        if score < min_score:
            continue

        meta = doc.get("metadata", {})
        chunks.append(RetrievedChunk(
            text       = doc.get("text", ""),
            score      = round(score, 4),
            topic      = meta.get("topic", ""),
            type       = meta.get("type", ""),
            difficulty = meta.get("difficulty", ""),
            chapter    = meta.get("chapter", ""),
            section    = meta.get("section", ""),
            has_code   = meta.get("has_code", "false") == "true",
            source_pdf = doc.get("source_pdf", ""),
        ))

    # Already sorted by Atlas (highest score first)
    logger.info(
        f"retrieve() → {len(chunks)}/{len(results)} chunks passed "
        f"min_score={min_score} for question: '{question[:60]}...'"
    )
    return chunks


# ── CONVENIENCE WRAPPERS ──────────────────────────────────────────────────────
# These are what LangGraph nodes will call directly —
# one function per intent type so nodes don't need to know chunk_types.

def retrieve_concept(question: str, course: str, **kwargs) -> list[RetrievedChunk]:
    """For 'what is X' / definition questions."""
    return retrieve(
        question,
        course,
        chunk_types=["concept", "analogy"],
        **kwargs
    )


def retrieve_howto(question: str, course: str, **kwargs) -> list[RetrievedChunk]:
    """For 'how do I' / 'show me how to' questions."""
    return retrieve(
        question,
        course,
        chunk_types=["code_example", "concept"],
        **kwargs
    )


def retrieve_mistake(question: str, course: str, **kwargs) -> list[RetrievedChunk]:
    """For error correction — student made a mistake."""
    return retrieve(
        question,
        course,
        chunk_types=["common_mistake"],
        **kwargs
    )


def retrieve_practice(question: str, course: str, **kwargs) -> list[RetrievedChunk]:
    """For practice problem requests."""
    return retrieve(
        question,
        course,
        chunk_types=["practice_problem"],
        **kwargs
    )


def retrieve_glossary(term: str, course: str, **kwargs) -> list[RetrievedChunk]:
    """For quick definition lookups."""
    return retrieve(
        term,
        course,
        chunk_types=["glossary"],
        top_k=3,
        **kwargs
    )


# ── FORMAT HELPERS ────────────────────────────────────────────────────────────

def format_context(chunks: list[RetrievedChunk], max_chars: int = 6000) -> str:
    """
    Format retrieved chunks into a single context string
    ready to be injected into the LLM prompt.

    Respects max_chars to stay within Gemini's context window.
    Each chunk is labelled with its type and topic for the LLM.
    """
    parts = []
    total = 0

    for i, chunk in enumerate(chunks, 1):
        label  = f"[Source {i} | {chunk.type} | {chunk.topic}]"
        block  = f"{label}\n{chunk.text}"
        length = len(block)

        if total + length > max_chars:
            logger.debug(
                f"format_context: stopped at chunk {i} "
                f"(would exceed {max_chars} chars)"
            )
            break

        parts.append(block)
        total += length

    return "\n\n---\n\n".join(parts)


def build_grounded_prompt(
    question: str,
    chunks:   list[RetrievedChunk],
    tutor_name: str = "Danish",
) -> str:
    """
    Build the final LLM prompt that grounds the answer in retrieved chunks.
    Used by the LangGraph generation node.
    """
    context = format_context(chunks)

    if not context:
        return (
            f"You are {tutor_name}, an experienced and friendly Python programming tutor. "
            f"The student asked: '{question}'. "
            "No specific course material was retrieved for this question, but answer it "
            "helpfully from your Python knowledge if it falls within Python fundamentals. "
            "Be warm, clear, and encouraging."
        )

    return f"""You are {tutor_name}, an experienced Python programming tutor who genuinely loves teaching. \
You have a gift for making complex ideas click — you use everyday analogies, well-chosen examples, \
and a warm conversational tone that makes students feel supported, not lectured at.

COURSE MATERIAL (your primary reference for this answer):
{context}

STUDENT QUESTION: {question}

HOW TO ANSWER — think of how a great human tutor would respond:

1. OPEN NATURALLY — start with a direct, friendly sentence that addresses the question head-on. \
No "Great question!", no "Certainly!". Just dive in like a confident teacher would.

2. EXPLAIN THE CONCEPT — use the course material as your backbone. \
Explain the *why*, not just the *what*. If the material has a code example, use it and walk through \
what each part does in plain English. If the material is thin on examples, add a short intuitive one.

3. USE ANALOGIES — whenever possible, ground abstract ideas in something the student already understands \
from daily life. ("Think of a variable like a labelled jar — the label is the name, the jar holds the value.")

4. KEEP IT CONVERSATIONAL — write like you're talking to the student, not writing a textbook. \
Short sentences. Active voice. Occasional rhetorical nudges ("Notice how...", "Here's the key thing...", \
"The trick is...").

5. CLOSE WARMLY — end with a brief encouraging line or a natural follow-up hook \
("Once you're comfortable with this, loops become really satisfying to write." / \
"Let me know if you want to see another example!").

STYLE RULES:
- Beginner-friendly: no jargon without explanation
- Vary your sentence rhythm — mix short punchy sentences with slightly longer explanatory ones
- Include code in ```python blocks``` when it adds clarity
- Never use bullet points or headers in your response — flowing prose only
- Aim for 4-8 sentences total; more only if a code example genuinely needs walking through

ANSWER:"""