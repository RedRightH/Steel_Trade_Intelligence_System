"""
run_eval.py - Evaluate the RAG pipeline against ground_truth.json.

Usage:
    python eval/run_eval.py                      # saves baseline_v1b.json + .csv
    python eval/run_eval.py --tag v1c            # custom run tag
    python eval/run_eval.py --compare            # compare v1 vs v1b side-by-side

Metrics captured per question:
  - answered        : bool (did RAG give an answer vs refuse?)
  - top_source      : first retrieved document
  - sources         : all retrieved document names
  - latency_ms      : total RAG latency
  - source_hit      : bool (was the expected source doc in top-3?)

Summary metrics:
  - answered_rate   : % questions answered (vs refused)
  - source_hit_rate : % questions where expected doc was in top-3
  - avg_latency_ms  : average latency across all questions
"""

import json
import time
import sys
import csv
import argparse
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_DIR    = Path(__file__).parent
GT_PATH     = EVAL_DIR / "ground_truth.json"
BASELINE_V1 = EVAL_DIR / "baseline_v1.json"

sys.path.insert(0, str(EVAL_DIR.parent))
from rag import rag_query

# ── Load ground truth ─────────────────────────────────────────────────────────

def load_ground_truth():
    with open(GT_PATH) as f:
        return json.load(f)


# ── Single question eval ──────────────────────────────────────────────────────

def eval_question(item: dict) -> dict:
    question = item["question"]
    expected_src = item.get("source_doc", "")

    t = time.time()
    result = rag_query(question)
    latency_ms = int((time.time() - t) * 1000)

    answer   = result["answer"]
    sources  = [s["file_name"] for s in result["sources"]]
    answered = "cannot answer" not in answer.lower()

    # Source hit: is the expected doc keyword in any retrieved filename?
    src_keywords = [w.lower() for w in expected_src.split() if len(w) > 4]
    source_hit = any(
        any(kw in s.lower() for kw in src_keywords)
        for s in sources
    ) if src_keywords else None

    return {
        "id":          item["id"],
        "question":    question,
        "type":        item.get("type", ""),
        "expected_src": expected_src,
        "actual_answer": answer,
        "answered":    answered,
        "top_source":  sources[0] if sources else "",
        "sources":     sources,
        "source_hit":  source_hit,
        "latency_ms":  latency_ms,
    }


# ── Run full eval ─────────────────────────────────────────────────────────────

def run_eval(tag: str = "v1b") -> dict:
    gt = load_ground_truth()
    results = []

    print(f"\n{'='*60}")
    print(f"EVAL RUN — tag={tag}  ({len(gt)} questions)")
    print(f"{'='*60}\n")

    for item in gt:
        print(f"  Q{item['id']}: {item['question'][:65]}...")
        r = eval_question(item)
        results.append(r)
        status     = "✓ ANSWERED" if r["answered"] else "✗ REFUSED"
        hit_str    = f"  src={'HIT' if r['source_hit'] else 'MISS'}" if r["source_hit"] is not None else ""
        print(f"         {status}  [{r['latency_ms']}ms]{hit_str}")
        print(f"         Top doc: {r['top_source']}")
        if r["answered"]:
            print(f"         Answer : {r['actual_answer'][:120]}...")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    answered      = [r for r in results if r["answered"]]
    refused       = [r for r in results if not r["answered"]]
    hits          = [r for r in results if r["source_hit"] is True]
    hit_eligible  = [r for r in results if r["source_hit"] is not None]
    avg_latency   = int(sum(r["latency_ms"] for r in results) / len(results))

    summary = {
        "tag":              tag,
        "total":            len(results),
        "answered":         len(answered),
        "refused":          len(refused),
        "answered_rate":    round(len(answered) / len(results), 3),
        "source_hit":       len(hits),
        "source_hit_rate":  round(len(hits) / len(hit_eligible), 3) if hit_eligible else None,
        "avg_latency_ms":   avg_latency,
    }

    output = {"summary": summary, "results": results}

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = EVAL_DIR / f"baseline_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {json_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = EVAL_DIR / f"baseline_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id","question","type","answered","source_hit",
            "top_source","latency_ms","actual_answer"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "id":           r["id"],
                "question":     r["question"],
                "type":         r["type"],
                "answered":     r["answered"],
                "source_hit":   r["source_hit"],
                "top_source":   r["top_source"],
                "latency_ms":   r["latency_ms"],
                "actual_answer": r["actual_answer"][:300],
            })
    print(f"Saved → {csv_path}")

    return output


# ── Compare two runs ──────────────────────────────────────────────────────────

def compare_runs(path_a: Path, path_b: Path):
    with open(path_a) as f: run_a = json.load(f)
    with open(path_b) as f: run_b = json.load(f)

    sa = run_a["summary"]
    sb = run_b["summary"]
    ra = {r["id"]: r for r in run_a["results"]}
    rb = {r["id"]: r for r in run_b["results"]}

    tag_a = sa.get("tag", path_a.stem)
    tag_b = sb.get("tag", path_b.stem)

    # Normalise: old baseline_v1.json may lack 'total', 'answered_rate', 'source_hit_rate'
    total_a = sa.get("total", sa.get("answered", 0) + sa.get("refused", 0))
    total_b = sb.get("total", sb.get("answered", 0) + sb.get("refused", 0))
    rate_a  = sa.get("answered_rate", sa["answered"] / total_a if total_a else 0)
    rate_b  = sb.get("answered_rate", sb["answered"] / total_b if total_b else 0)

    print(f"\n{'='*65}")
    print(f"COMPARISON: {tag_a}  →  {tag_b}")
    print(f"{'='*65}")
    print(f"  {'Metric':<22} {tag_a:>10} {tag_b:>10}  {'Δ':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Answered':<22} {sa['answered']:>7}/{total_a} {sb['answered']:>7}/{total_b}  Δ={sb['answered']-sa['answered']:+d}")
    print(f"  {'Refused':<22} {sa['refused']:>7}/{total_a} {sb['refused']:>7}/{total_b}  Δ={sb['refused']-sa['refused']:+d}")
    print(f"  {'Answered rate':<22} {rate_a:>10.1%} {rate_b:>10.1%}  Δ={rate_b-rate_a:>+.1%}")
    if sb.get("source_hit_rate") is not None:
        sh_a = sa.get("source_hit_rate", "—")
        sh_b = sb["source_hit_rate"]
        if isinstance(sh_a, float):
            print(f"  {'Source hit rate':<22} {sh_a:>10.1%} {sh_b:>10.1%}  Δ={sh_b-sh_a:>+.1%}")
        else:
            print(f"  {'Source hit rate':<22} {'—':>10} {sh_b:>10.1%}  Δ=—")
    lat_a = sa.get("avg_latency_ms", 3627)
    lat_b = sb.get("avg_latency_ms", 0)
    print(f"  {'Avg latency (ms)':<22} {lat_a:>10,} {lat_b:>10,}  Δ={lat_b-lat_a:>+,}")

    print(f"\n  Question-level changes:")
    for qid in sorted(ra.keys()):
        a = ra[qid]; b = rb.get(qid)
        if b is None: continue
        a_ok = a["answered"]; b_ok = b["answered"]
        if a_ok != b_ok:
            change = "REFUSED → ANSWERED ✅" if b_ok else "ANSWERED → REFUSED ⚠️"
            print(f"    Q{qid} [{a['type']}]: {change}")
            print(f"          {a['question'][:70]}")
        else:
            # Same outcome — check if source hit changed
            if a.get("source_hit") != b.get("source_hit"):
                sh = "MISS → HIT" if b.get("source_hit") else "HIT → MISS"
                print(f"    Q{qid} [{a['type']}]: source_hit {sh}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",     default="v1b", help="Run tag (default: v1b)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare baseline_v1.json vs baseline_<tag>.json")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only compare existing files, do not run eval")
    args = parser.parse_args()

    if args.compare_only:
        b_path = EVAL_DIR / f"baseline_{args.tag}.json"
        compare_runs(BASELINE_V1, b_path)
    else:
        run_eval(tag=args.tag)
        if args.compare or True:   # always compare after a run
            b_path = EVAL_DIR / f"baseline_{args.tag}.json"
            if BASELINE_V1.exists() and b_path.exists():
                compare_runs(BASELINE_V1, b_path)
