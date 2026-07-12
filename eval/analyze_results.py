"""
eval/analyze_results.py
=======================
Analyze evaluation results by question category, generate
LaTeX-ready tables and diagnostic summaries.

Usage:
    python -m eval.analyze_results
    python -m eval.analyze_results --results-dir eval/results
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIGS = ["naive_rag", "crag_only", "hall_only", "full_emualim"]
CATEGORIES = ["concept", "howto", "mistake", "complex", "out_of_scope"]


def load_results(results_dir: str, config_name: str) -> list[dict]:
    path = os.path.join(results_dir, f"{config_name}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def per_category_breakdown(results_dir: str):
    """Show per-category diagnostics for each configuration."""
    print("\n" + "=" * 80)
    print("  PER-CATEGORY BREAKDOWN")
    print("=" * 80)

    for config_name in CONFIGS:
        data = load_results(results_dir, config_name)
        if not data:
            continue

        print(f"\n  {config_name}")
        print(f"  {'-'*60}")

        by_cat = defaultdict(list)
        for d in data:
            cat = d.get("category", "unknown")
            by_cat[cat].append(d)

        header = f"    {'Category':<15} {'Count':>5} {'Fallback':>8} {'Retried':>8} {'Avg Chunks':>10}"
        print(header)
        print(f"    {'-'*50}")

        for cat in CATEGORIES:
            items = by_cat.get(cat, [])
            if not items:
                continue
            n = len(items)
            fb = sum(1 for d in items if d.get("needs_fallback"))
            rt = sum(1 for d in items if d.get("retry_count", 0) > 0)
            avg_c = sum(d.get("graded_chunk_count", 0) for d in items) / n
            print(f"    {cat:<15} {n:>5} {fb:>8} {rt:>8} {avg_c:>10.1f}")


def intent_accuracy_report(results_dir: str):
    """Check if intent classifier matched expected intents."""
    print("\n" + "=" * 80)
    print("  INTENT CLASSIFICATION ACCURACY")
    print("=" * 80)

    # Use full_emualim (all configs share the same intent classifier)
    data = load_results(results_dir, "full_emualim")
    if not data:
        data = load_results(results_dir, "naive_rag")
    if not data:
        print("  No results found.")
        return

    correct = 0
    total = 0
    mismatches = []

    for d in data:
        expected = d.get("expected_intent", "")
        actual = d.get("intent", "")
        if not expected:
            continue
        total += 1
        if actual == expected:
            correct += 1
        else:
            mismatches.append({
                "question": d["question"][:60],
                "expected": expected,
                "actual": actual,
            })

    if total == 0:
        print("  No expected_intent annotations in test set.")
        return

    print(f"\n  Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)}):")
        for m in mismatches[:10]:
            print(f"    '{m['question']}...'")
            print(f"      expected={m['expected']}, got={m['actual']}")


def qualitative_examples(results_dir: str, n: int = 3):
    """
    Find examples where naive RAG and full E-Mualim diverge most,
    useful for Table III in the paper.
    """
    print("\n" + "=" * 80)
    print("  QUALITATIVE COMPARISON EXAMPLES (for Table III)")
    print("=" * 80)

    naive = load_results(results_dir, "naive_rag")
    full = load_results(results_dir, "full_emualim")

    if not naive or not full:
        print("  Need both naive_rag and full_emualim results.")
        return

    # Pair by question
    naive_map = {d["question"]: d for d in naive}
    candidates = []

    for d in full:
        q = d["question"]
        nd = naive_map.get(q)
        if not nd:
            continue

        # Score: naive had fallback or retry but full didn't
        divergence = 0
        if nd.get("needs_fallback") and not d.get("needs_fallback"):
            divergence += 2
        if nd.get("retry_count", 0) > 0 and d.get("retry_count", 0) == 0:
            divergence += 1
        if nd.get("graded_chunk_count", 5) > d.get("graded_chunk_count", 0):
            divergence += 1

        if divergence > 0:
            candidates.append((divergence, q, nd, d))

    candidates.sort(reverse=True, key=lambda x: x[0])

    for i, (score, q, nd, fd) in enumerate(candidates[:n]):
        print(f"\n  Example {i+1} (divergence={score}):")
        print(f"  Question: {q}")
        print(f"  Category: {fd.get('category', '?')}")
        print(f"\n  Naive RAG answer (first 200 chars):")
        print(f"    {nd['answer'][:200]}...")
        print(f"  Naive: chunks={nd.get('graded_chunk_count','?')}, "
              f"fallback={nd.get('needs_fallback')}, retry={nd.get('retry_count',0)}")
        print(f"\n  E-Mualim answer (first 200 chars):")
        print(f"    {fd['answer'][:200]}...")
        print(f"  E-Mualim: chunks={fd.get('graded_chunk_count','?')}, "
              f"fallback={fd.get('needs_fallback')}, retry={fd.get('retry_count',0)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze E-Mualim eval results")
    parser.add_argument("--results-dir", default="eval/results")
    args = parser.parse_args()

    per_category_breakdown(args.results_dir)
    intent_accuracy_report(args.results_dir)
    qualitative_examples(args.results_dir)


if __name__ == "__main__":
    main()
