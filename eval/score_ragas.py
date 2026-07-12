"""
eval/score_ragas.py
===================
Compute RAGAS metrics from run_eval.py output files.
Produces a comparison table across all configurations.

Usage:
    python -m eval.score_ragas                              # all configs
    python -m eval.score_ragas --config full_emualim        # single config
    python -m eval.score_ragas --evaluator groq             # use Groq as evaluator

Prerequisites:
    pip install ragas datasets langchain-groq langchain-openai
"""

import argparse
import json
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

CONFIGS = ["naive_rag", "crag_only", "hall_only", "full_emualim"]


def get_evaluator_llm(evaluator: str):
    """
    Return a LangChain-compatible LLM for RAGAS evaluation.
    RAGAS uses this to judge faithfulness, relevancy, etc.
    """
    if evaluator == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY_MAIN"),
            temperature=0.0,
        )
    elif evaluator == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
        )
    else:
        raise ValueError(f"Unknown evaluator: {evaluator}. Use 'groq' or 'openai'.")


def load_results(results_dir: str, config_name: str) -> list[dict]:
    """Load a config's results JSON."""
    path = os.path.join(results_dir, f"{config_name}.json")
    if not os.path.exists(path):
        logger.warning(f"Results file not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_diagnostics(data: list[dict]) -> dict:
    """Compute non-RAGAS diagnostic metrics."""
    total = len(data)
    if total == 0:
        return {}

    retried = sum(1 for d in data if d.get("retry_count", 0) > 0)
    fallbacks = sum(1 for d in data if d.get("needs_fallback", False))

    # Intent accuracy (if expected_intent is provided)
    intent_correct = sum(
        1 for d in data
        if d.get("expected_intent") and d["intent"] == d["expected_intent"]
    )
    intent_total = sum(1 for d in data if d.get("expected_intent"))

    # Average chunks retained
    avg_graded = sum(d.get("graded_chunk_count", 0) for d in data) / total
    avg_raw = sum(d.get("raw_chunk_count", 0) for d in data) / total

    return {
        "retry_rate_pct":         round(retried / total * 100, 1),
        "fallback_rate_pct":      round(fallbacks / total * 100, 1),
        "intent_accuracy_pct":    round(intent_correct / intent_total * 100, 1) if intent_total > 0 else None,
        "avg_chunks_retrieved":   round(avg_raw, 1),
        "avg_chunks_after_crag":  round(avg_graded, 1),
    }


def score_config(
    config_name: str,
    results_dir: str,
    evaluator_llm=None,
) -> dict | None:
    """Score one config with RAGAS and return metrics dict."""
    data = load_results(results_dir, config_name)
    if not data:
        return None

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question":     [d["question"] for d in data],
        "answer":       [d["answer"] for d in data],
        "contexts":     [d["contexts"] for d in data],
        "ground_truth": [d["ground_truth"] for d in data],
    })

    # Run RAGAS
    eval_kwargs = {
        "dataset": dataset,
        "metrics": [faithfulness, answer_relevancy, context_precision, context_recall],
    }
    if evaluator_llm:
        eval_kwargs["llm"] = evaluator_llm

    result = evaluate(**eval_kwargs)

    # Combine with diagnostics
    diagnostics = compute_diagnostics(data)

    return {
        "faithfulness":       round(result["faithfulness"], 4),
        "answer_relevancy":   round(result["answer_relevancy"], 4),
        "context_precision":  round(result["context_precision"], 4),
        "context_recall":     round(result["context_recall"], 4),
        **diagnostics,
    }


def print_comparison_table(all_scores: dict[str, dict]):
    """Print a formatted comparison table for the paper."""
    print("\n" + "=" * 90)
    print("  RAGAS EVALUATION RESULTS — Copy these into your LaTeX \\todo{} cells")
    print("=" * 90)

    header = f"{'Configuration':<20} {'Faith.':>8} {'Ans.Rel.':>8} {'Ctx.Prec.':>9} {'Ctx.Rec.':>8} {'Retry%':>7}"
    print(header)
    print("-" * 70)

    for config_name, scores in all_scores.items():
        if scores is None:
            print(f"{config_name:<20} {'(no results)':>40}")
            continue

        print(
            f"{config_name:<20} "
            f"{scores['faithfulness']:>8.4f} "
            f"{scores['answer_relevancy']:>8.4f} "
            f"{scores['context_precision']:>9.4f} "
            f"{scores['context_recall']:>8.4f} "
            f"{scores.get('retry_rate_pct', 0):>6.1f}%"
        )

    # Delta row
    if "naive_rag" in all_scores and "full_emualim" in all_scores:
        nr = all_scores["naive_rag"]
        fe = all_scores["full_emualim"]
        if nr and fe:
            print("-" * 70)
            print(
                f"{'Δ (Full vs Naive)':<20} "
                f"{fe['faithfulness'] - nr['faithfulness']:>+8.4f} "
                f"{fe['answer_relevancy'] - nr['answer_relevancy']:>+8.4f} "
                f"{fe['context_precision'] - nr['context_precision']:>+9.4f} "
                f"{fe['context_recall'] - nr['context_recall']:>+8.4f} "
                f"{fe.get('retry_rate_pct', 0) - nr.get('retry_rate_pct', 0):>+6.1f}%"
            )

    print()

    # Diagnostics
    print("DIAGNOSTICS")
    print("-" * 70)
    for config_name, scores in all_scores.items():
        if scores is None:
            continue
        intent_acc = scores.get("intent_accuracy_pct")
        print(
            f"  {config_name}: "
            f"fallback={scores.get('fallback_rate_pct', 0):.1f}%, "
            f"avg_chunks={scores.get('avg_chunks_after_crag', 0):.1f}/"
            f"{scores.get('avg_chunks_retrieved', 0):.1f}"
            + (f", intent_acc={intent_acc:.1f}%" if intent_acc else "")
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Score E-Mualim results with RAGAS")
    parser.add_argument("--results-dir", default="eval/results")
    parser.add_argument("--config", choices=CONFIGS, default=None)
    parser.add_argument(
        "--evaluator",
        choices=["groq", "openai"],
        default="groq",
        help="LLM backend for RAGAS evaluation (default: groq)",
    )
    parser.add_argument(
        "--save",
        default="eval/results/ragas_scores.json",
        help="Save scores to JSON",
    )
    args = parser.parse_args()

    evaluator_llm = get_evaluator_llm(args.evaluator)
    configs_to_score = [args.config] if args.config else CONFIGS

    all_scores = {}
    for config_name in configs_to_score:
        logger.info(f"Scoring {config_name}...")
        scores = score_config(config_name, args.results_dir, evaluator_llm)
        all_scores[config_name] = scores

    # Print comparison table
    print_comparison_table(all_scores)

    # Save to JSON
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(all_scores, f, indent=2)
        logger.info(f"Scores saved → {args.save}")


if __name__ == "__main__":
    main()
