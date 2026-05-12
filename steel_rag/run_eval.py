"""
run_eval.py — Run all 10 ground-truth questions and save results.
Usage: python run_eval.py
"""

import json
import time
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag import rag_query

GROUND_TRUTH = Path(__file__).parent / "eval" / "ground_truth.json"
OUTPUT = Path(__file__).parent / "eval" / "baseline_v1.json"


def run_eval():
    with open(GROUND_TRUTH) as f:
        qa_pairs = json.load(f)

    results = []
    print("=" * 60)
    print("STEEL RAG - Baseline Eval (10 Q&A pairs)")
    print("=" * 60)
    print("Loading model and index (first run ~30s)...\n")

    for i, qa in enumerate(qa_pairs, 1):
        q = qa["question"]
        expected = qa["answer"]
        print(f"[{i}/10] {q[:70]}...")

        t0 = time.time()
        result = rag_query(q)
        latency_ms = int((time.time() - t0) * 1000)

        answered = "cannot answer" not in result["answer"].lower()
        top_source = result["sources"][0]["file_name"] if result["sources"] else "none"

        results.append({
            "id": qa["id"],
            "question": q,
            "expected_answer": expected,
            "actual_answer": result["answer"],
            "answered": answered,
            "top_source": top_source,
            "sources": [s["file_name"] for s in result["sources"]],
            "latency_ms": latency_ms,
            "type": qa["type"],
        })

        status = "[ANSWERED]" if answered else "[REFUSED - no context]"
        print(f"  {status} ({latency_ms}ms) | Top source: {top_source}")
        print(f"  Answer: {result['answer'][:120]}...")
        print()

        time.sleep(0.5)  # avoid Groq rate limit

    # Summary
    answered_count = sum(1 for r in results if r["answered"])
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Answered: {answered_count}/10")
    print(f"  Refused (no context): {10 - answered_count}/10")
    print(f"  Avg latency: {avg_latency:.0f}ms")
    print()

    by_type = {}
    for r in results:
        t = r["type"]
        by_type.setdefault(t, {"answered": 0, "total": 0})
        by_type[t]["total"] += 1
        if r["answered"]:
            by_type[t]["answered"] += 1
    print("  By question type:")
    for qtype, counts in by_type.items():
        print(f"    {qtype}: {counts['answered']}/{counts['total']} answered")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({"summary": {
            "answered": answered_count,
            "refused": 10 - answered_count,
            "avg_latency_ms": round(avg_latency),
        }, "results": results}, f, indent=2)

    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    run_eval()
