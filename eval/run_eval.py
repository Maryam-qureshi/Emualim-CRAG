"""
eval/run_eval.py
================
Run the E-Mualim pipeline in multiple configurations on the test set.
Produces per-config JSON results ready for RAGAS scoring.

Usage:
    python -m eval.run_eval                           # all 4 configs
    python -m eval.run_eval --config naive_rag        # single config
    python -m eval.run_eval --test-set data/test_sets/test_set_v1.csv

Configurations:
    naive_rag    — no CRAG grader, no hallucination checker
    crag_only    — CRAG grader ON, hallucination checker OFF
    hall_only    — CRAG grader OFF, hallucination checker ON
    full_emualim — both ON (production configuration)
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.graph import (
    intent_classifier,
    rag_retriever,
    crag_grader,
    generator,
    hallucination_checker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Evaluation configurations ────────────────────────────────────────────────

CONFIGS = {
    "naive_rag": {
        "crag": False,
        "hallucination": False,
        "description": "Baseline: all chunks passed directly to generator",
    },
    "crag_only": {
        "crag": True,
        "hallucination": False,
        "description": "CRAG grader ON, hallucination checker OFF",
    },
    "hall_only": {
        "crag": False,
        "hallucination": True,
        "description": "CRAG grader OFF, hallucination checker ON",
    },
    "full_emualim": {
        "crag": True,
        "hallucination": True,
        "description": "Full pipeline: both corrective stages active",
    },
}


# ── Pipeline runner ──────────────────────────────────────────────────────────

def build_initial_state(question: str) -> dict:
    """Build a clean initial state for one pipeline run."""
    return {
        "message":              question,
        "course":               "introduction_to_python",
        "session_id":           "eval_session",
        "student_name":         "Student",
        "tutor_name":           "Tutor",
        "conversation_history": [],
        "current_topic":        "",
        "student_notes":        [],
        "intent":               "",
        "chunks":               [],
        "graded_chunks":        [],
        "needs_fallback":       False,
        "response":             "",
        "is_grounded":          True,
        "retry_count":          0,
        "trigger_flashcard":    False,
        "simplify":             False,
        "code_snippet":         "",
    }


def run_pipeline(question: str, config: dict) -> dict:
    """
    Run one question through the pipeline with the given config.
    Returns dict with answer, contexts, and diagnostic metadata.
    """
    state = build_initial_state(question)

    # Node 1: Intent classifier
    state = intent_classifier(state)

    # Node 2: RAG retriever
    state = rag_retriever(state)
    raw_chunk_count = len(state.get("chunks", []))

    # Node 3: CRAG grader
    if config["crag"]:
        state = crag_grader(state)
    else:
        # Bypass: pass all chunks through unfiltered
        state = {
            **state,
            "graded_chunks": list(state.get("chunks", [])),
            "needs_fallback": len(state.get("chunks", [])) == 0,
        }

    graded_chunk_count = len(state.get("graded_chunks", []))

    # Node 4: Generator
    state = generator(state)

    # Node 5: Hallucination checker with retry loop
    retry_count = 0
    if config["hallucination"]:
        state = hallucination_checker(state)
        while not state.get("is_grounded", True) and state.get("retry_count", 0) < 2:
            logger.info(f"  Retry {state['retry_count']} for: {question[:50]}...")
            state = generator(state)
            state = hallucination_checker(state)
            retry_count = state.get("retry_count", 0)
    else:
        state = {**state, "is_grounded": True}

    # Extract results
    graded = state.get("graded_chunks", [])
    contexts = [c.text for c in graded] if graded else []

    return {
        "answer":             state.get("response", ""),
        "contexts":           contexts,
        "intent":             state.get("intent", ""),
        "retry_count":        retry_count,
        "needs_fallback":     state.get("needs_fallback", False),
        "is_grounded":        state.get("is_grounded", True),
        "raw_chunk_count":    raw_chunk_count,
        "graded_chunk_count": graded_chunk_count,
    }


# ── Main evaluation loop ────────────────────────────────────────────────────

def run_eval(
    test_file: str,
    output_dir: str = "eval/results",
    configs_to_run: list[str] | None = None,
    delay: float = 1.5,
):
    """
    Run evaluation across configurations and save results.

    Parameters
    ----------
    test_file     : Path to CSV with columns: question, ground_truth, category
    output_dir    : Where to save per-config JSON results
    configs_to_run: List of config names, or None for all
    delay         : Seconds between API calls (respect Groq rate limits)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load test set
    with open(test_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_set = list(reader)

    logger.info(f"Loaded {len(test_set)} questions from {test_file}")

    if configs_to_run is None:
        configs_to_run = list(CONFIGS.keys())

    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {config_name} — {config['description']}")
        logger.info(f"{'='*60}")

        results = []
        errors = []

        for i, row in enumerate(test_set):
            question = row["question"]
            logger.info(f"  [{i+1}/{len(test_set)}] {question[:60]}...")

            try:
                result = run_pipeline(question, config)
                results.append({
                    "question":           question,
                    "ground_truth":       row["ground_truth"],
                    "category":           row.get("category", ""),
                    "expected_intent":    row.get("expected_intent", ""),
                    "answer":             result["answer"],
                    "contexts":           result["contexts"],
                    "intent":             result["intent"],
                    "retry_count":        result["retry_count"],
                    "needs_fallback":     result["needs_fallback"],
                    "is_grounded":        result["is_grounded"],
                    "raw_chunk_count":    result["raw_chunk_count"],
                    "graded_chunk_count": result["graded_chunk_count"],
                })
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                errors.append({"question": question, "error": str(e)})
                results.append({
                    "question":       question,
                    "ground_truth":   row["ground_truth"],
                    "category":       row.get("category", ""),
                    "answer":         f"[ERROR: {e}]",
                    "contexts":       [],
                    "intent":         "",
                    "retry_count":    0,
                    "needs_fallback": True,
                    "is_grounded":    False,
                })

            # Rate limit delay
            if delay > 0 and i < len(test_set) - 1:
                time.sleep(delay)

        # Save results
        out_path = os.path.join(output_dir, f"{config_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved {len(results)} results → {out_path}")

        if errors:
            err_path = os.path.join(output_dir, f"{config_name}_errors.json")
            with open(err_path, "w") as f:
                json.dump(errors, f, indent=2)
            logger.warning(f"  {len(errors)} errors → {err_path}")

        # Print quick summary
        fallbacks = sum(1 for r in results if r["needs_fallback"])
        retries = sum(1 for r in results if r["retry_count"] > 0)
        logger.info(f"  Summary: {len(results)} total, "
                     f"{fallbacks} fallbacks, {retries} retried")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E-Mualim evaluation")
    parser.add_argument(
        "--test-set",
        default="data/test_sets/test_set_v1.csv",
        help="Path to test set CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results",
        help="Directory for result JSON files",
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        default=None,
        help="Run a single config (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between API calls (default: 1.5)",
    )
    args = parser.parse_args()

    configs = [args.config] if args.config else None
    run_eval(args.test_set, args.output_dir, configs, args.delay)
