"""
run_eval.py - Evaluate the RAG pipeline against a ground truth file.

Usage:
    python eval/run_eval.py                          # v1b: 10-Q RAG eval
    python eval/run_eval.py --tag v2 --gt v2         # v2:  25-Q full eval
    python eval/run_eval.py --tag v2 --gt v2 --judge # v2 + LLM-as-judge scoring
    python eval/run_eval.py --compare-only --tag v2  # compare v1b vs v2

Metrics (base):
  answered, source_hit, latency_ms, question_type routing accuracy

Metrics (with --judge):
  faithfulness  0-1  is the answer grounded in retrieved context?
  relevance     0-1  does the answer address the question?
  completeness  0-1  does it cover key facts from the expected answer?
  judge_score   0-1  average of the three
"""

import json
import os
import re
import time
import sys
import csv
import argparse
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_DIR    = Path(__file__).parent
BASELINE_V1B = EVAL_DIR / "baseline_v1b.json"

sys.path.insert(0, str(EVAL_DIR.parent))

from dotenv import load_dotenv
load_dotenv(EVAL_DIR.parent.parent / ".env")


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_ground_truth(name: str = "v1") -> list[dict]:
    """Load ground_truth.json (name='v1') or ground_truth_v2.json (name='v2')."""
    fname = "ground_truth.json" if name == "v1" else f"ground_truth_{name}.json"
    path  = EVAL_DIR / fname
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── RAG evaluator ─────────────────────────────────────────────────────────────

def eval_question_rag(item: dict) -> dict:
    """Run RAG on one question, return base metrics."""
    from rag import rag_query

    question     = item["question"]
    expected_src = item.get("source_doc", "")
    expected_ans = item.get("answer", "")

    t          = time.time()
    result     = rag_query(question)
    latency_ms = int((time.time() - t) * 1000)

    answer  = result["answer"]
    sources = [s["file_name"] for s in result["sources"]]
    context = result.get("context_used", "")
    answered = "cannot answer" not in answer.lower()

    # Source hit
    src_keywords = [w.lower() for w in expected_src.split() if len(w) > 4]
    non_data_src = expected_src not in ("TRADESTAT_XLSX", "WTO_WITS_MFN_CSV")
    source_hit = (
        any(any(kw in s.lower() for kw in src_keywords) for s in sources)
        if (src_keywords and non_data_src) else None
    )

    return {
        "id":            item["id"],
        "question":      question,
        "question_type": item.get("question_type", ""),
        "type":          item.get("type", ""),
        "expected_src":  expected_src,
        "expected_ans":  expected_ans,
        "actual_answer": answer,
        "answered":      answered,
        "top_source":    sources[0] if sources else "",
        "sources":       sources,
        "context":       context[:1500],
        "source_hit":    source_hit,
        "latency_ms":    latency_ms,
        # judge fields filled in later
        "faithfulness":  None,
        "relevance":     None,
        "completeness":  None,
        "judge_score":   None,
        "judge_reason":  "",
    }


# ── Router evaluator ──────────────────────────────────────────────────────────

def eval_question_router(item: dict) -> dict:
    """Run full router on one question (used for DATA_ANALYSIS / TARIFF_ANALYSIS)."""
    from router import route_query

    question     = item["question"]
    expected_ans = item.get("answer", "")
    expected_qt  = item.get("question_type", "")

    t  = time.time()
    ro = route_query(question)
    latency_ms = int((time.time() - t) * 1000)

    answer   = ro.result.answer_text
    answered = "cannot answer" not in answer.lower()
    routing_correct = (ro.question_type == expected_qt) if expected_qt else None

    return {
        "id":              item["id"],
        "question":        question,
        "question_type":   item.get("question_type", ""),
        "type":            item.get("type", ""),
        "expected_src":    item.get("source_doc", ""),
        "expected_ans":    expected_ans,
        "actual_answer":   answer,
        "answered":        answered,
        "top_source":      ro.agent_used,
        "sources":         [ro.agent_used],
        "context":         "",
        "source_hit":      None,
        "routing_correct": routing_correct,
        "routed_to":       ro.question_type,
        "latency_ms":      latency_ms,
        "faithfulness":    None,
        "relevance":       None,
        "completeness":    None,
        "judge_score":     None,
        "judge_reason":    "",
    }


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator for a steel trade intelligence RAG system.

You will be given:
- QUESTION: the user's question
- EXPECTED ANSWER: the ground truth answer
- ACTUAL ANSWER: what the system produced
- CONTEXT: the retrieved document excerpts used to generate the answer

Score the actual answer on three dimensions (each 0.0 to 1.0):

faithfulness  — Is every claim in the actual answer supported by the context?
                0.0 = answer is hallucinated / contradicts context
                0.5 = partially grounded, some unsupported claims
                1.0 = fully grounded in retrieved context

relevance     — Does the actual answer directly address the question?
                0.0 = completely off-topic
                0.5 = partially addresses the question
                1.0 = directly and completely addresses the question

completeness  — Does the actual answer cover the key facts in the expected answer?
                0.0 = misses all key facts
                0.5 = covers some key facts
                1.0 = covers all key facts

Special cases:
- If the actual answer is "I cannot answer this from the provided documents" score:
  faithfulness=1.0, relevance=0.5, completeness=0.0
- If expected answer is marked as DATA_ANALYSIS or TARIFF_ANALYSIS with live data,
  focus on whether numbers are in the right ballpark and reasoning is correct.

Return ONLY valid JSON — no markdown:
{
  "faithfulness": <0.0-1.0>,
  "relevance": <0.0-1.0>,
  "completeness": <0.0-1.0>,
  "reasoning": "<one sentence explaining the scores>"
}"""


def judge_answer(r: dict) -> dict:
    """Score one eval result with the LLM judge. Returns updated dict."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    user_msg = (
        f"QUESTION: {r['question']}\n\n"
        f"EXPECTED ANSWER: {r['expected_ans']}\n\n"
        f"ACTUAL ANSWER: {r['actual_answer']}\n\n"
        f"CONTEXT (retrieved docs): {r.get('context','')[:1200]}"
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw   = resp.choices[0].message.content.strip()
        start = raw.find("{"); end = raw.rfind("}") + 1
        j     = json.loads(raw[start:end])
        r["faithfulness"]  = round(float(j.get("faithfulness",  0.5)), 2)
        r["relevance"]     = round(float(j.get("relevance",     0.5)), 2)
        r["completeness"]  = round(float(j.get("completeness",  0.5)), 2)
        r["judge_score"]   = round(
            (r["faithfulness"] + r["relevance"] + r["completeness"]) / 3, 2
        )
        r["judge_reason"]  = j.get("reasoning", "")
    except Exception as e:
        r["judge_reason"] = f"Judge error: {e}"
    return r


# ── Run full eval ─────────────────────────────────────────────────────────────

def run_eval(tag: str = "v2", gt_name: str = "v2", use_judge: bool = False) -> dict:
    gt      = load_ground_truth(gt_name)
    results = []

    print(f"\n{'='*65}")
    print(f"EVAL RUN — tag={tag}  gt={gt_name}  judge={'ON' if use_judge else 'OFF'}")
    print(f"Questions: {len(gt)}  |  LLM judge: {'Groq llama-3.3-70b' if use_judge else 'disabled'}")
    print(f"{'='*65}\n")

    DATA_TYPES = {"DATA_ANALYSIS", "TARIFF_ANALYSIS"}

    for item in gt:
        print(f"  Q{item['id']:>2} [{item.get('question_type','?'):20}] {item['question'][:55]}…")

        qt = item.get("question_type", "")
        if qt in DATA_TYPES:
            r = eval_question_router(item)
        else:
            r = eval_question_rag(item)

        if use_judge and r["answered"]:
            r = judge_answer(r)

        results.append(r)

        status  = "✓" if r["answered"] else "✗"
        j_str   = f"  judge={r['judge_score']:.2f}" if r["judge_score"] is not None else ""
        rt_str  = (f"  route={'✓' if r.get('routing_correct') else '✗'}({r.get('routed_to','')})"
                   if "routing_correct" in r else "")
        src_str = (f"  src={'HIT' if r['source_hit'] else 'MISS'}"
                   if r["source_hit"] is not None else "")
        print(f"       {status} [{r['latency_ms']:>5}ms]{j_str}{rt_str}{src_str}")
        if r["judge_reason"]:
            print(f"         {r['judge_reason'][:90]}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    n         = len(results)
    answered  = [r for r in results if r["answered"]]
    refused   = [r for r in results if not r["answered"]]
    hits      = [r for r in results if r.get("source_hit") is True]
    hit_elig  = [r for r in results if r.get("source_hit") is not None]
    judged    = [r for r in results if r["judge_score"] is not None]
    routed    = [r for r in results if r.get("routing_correct") is not None]
    avg_lat   = int(sum(r["latency_ms"] for r in results) / n)

    summary = {
        "tag":              tag,
        "gt":               gt_name,
        "total":            n,
        "answered":         len(answered),
        "refused":          len(refused),
        "answered_rate":    round(len(answered) / n, 3),
        "source_hit":       len(hits),
        "source_hit_rate":  round(len(hits) / len(hit_elig), 3) if hit_elig else None,
        "routing_accuracy": round(sum(1 for r in routed if r["routing_correct"]) / len(routed), 3) if routed else None,
        "avg_faithfulness": round(sum(r["faithfulness"]  for r in judged) / len(judged), 3) if judged else None,
        "avg_relevance":    round(sum(r["relevance"]     for r in judged) / len(judged), 3) if judged else None,
        "avg_completeness": round(sum(r["completeness"]  for r in judged) / len(judged), 3) if judged else None,
        "avg_judge_score":  round(sum(r["judge_score"]   for r in judged) / len(judged), 3) if judged else None,
        "avg_latency_ms":   avg_lat,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"SUMMARY — {tag}")
    print(f"{'='*65}")
    print(f"  Answered:         {summary['answered']}/{n}  ({summary['answered_rate']:.0%})")
    print(f"  Refused:          {summary['refused']}/{n}")
    if summary["source_hit_rate"] is not None:
        print(f"  Source hit rate:  {summary['source_hit_rate']:.0%}")
    if summary["routing_accuracy"] is not None:
        print(f"  Routing accuracy: {summary['routing_accuracy']:.0%}")
    if summary["avg_judge_score"] is not None:
        print(f"  Avg judge score:  {summary['avg_judge_score']:.2f}")
        print(f"    faithfulness:   {summary['avg_faithfulness']:.2f}")
        print(f"    relevance:      {summary['avg_relevance']:.2f}")
        print(f"    completeness:   {summary['avg_completeness']:.2f}")
    print(f"  Avg latency:      {avg_lat:,}ms")

    output = {"summary": summary, "results": results}

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = EVAL_DIR / f"baseline_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved → {json_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = EVAL_DIR / f"baseline_{tag}.csv"
    fields = ["id","question","question_type","type","answered","source_hit",
              "routing_correct","routed_to","faithfulness","relevance",
              "completeness","judge_score","judge_reason","top_source","latency_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved → {csv_path}")

    return output


# ── Compare two runs ──────────────────────────────────────────────────────────

def compare_runs(path_a: Path, path_b: Path):
    with open(path_a, encoding="utf-8") as f: run_a = json.load(f)
    with open(path_b, encoding="utf-8") as f: run_b = json.load(f)

    sa = run_a["summary"]; sb = run_b["summary"]
    ra = {r["id"]: r for r in run_a["results"]}
    rb = {r["id"]: r for r in run_b["results"]}
    tag_a = sa.get("tag", path_a.stem)
    tag_b = sb.get("tag", path_b.stem)

    total_a = sa.get("total", sa.get("answered",0) + sa.get("refused",0))
    total_b = sb.get("total", sb.get("answered",0) + sb.get("refused",0))
    rate_a  = sa.get("answered_rate", sa["answered"]/total_a if total_a else 0)
    rate_b  = sb.get("answered_rate", sb["answered"]/total_b if total_b else 0)

    def _fmt(val, prev=None, pct=False):
        if val is None: return "—"
        s = f"{val:.0%}" if pct else f"{val:.2f}"
        if prev is not None and prev is not None:
            d = val - prev
            s += f" ({d:+.0%})" if pct else f" ({d:+.2f})"
        return s

    print(f"\n{'='*65}")
    print(f"COMPARISON: {tag_a} ({total_a}Q)  →  {tag_b} ({total_b}Q)")
    print(f"{'='*65}")
    print(f"  {'Metric':<26} {tag_a:>12} {tag_b:>12}")
    print(f"  {'-'*52}")
    print(f"  {'Answered':<26} {sa['answered']:>6}/{total_a}    {sb['answered']:>6}/{total_b}")
    print(f"  {'Answered rate':<26} {rate_a:>11.0%} {rate_b:>11.0%}  Δ={rate_b-rate_a:+.0%}")

    sh_a = sa.get("source_hit_rate"); sh_b = sb.get("source_hit_rate")
    if sh_b:
        d = f"Δ={sh_b-sh_a:+.0%}" if sh_a else ""
        print(f"  {'Source hit rate':<26} {(f'{sh_a:.0%}' if sh_a else '—'):>12} {sh_b:>11.0%}  {d}")

    ra_acc = sa.get("routing_accuracy"); rb_acc = sb.get("routing_accuracy")
    if rb_acc:
        d = f"Δ={rb_acc-ra_acc:+.0%}" if ra_acc else ""
        print(f"  {'Routing accuracy':<26} {(f'{ra_acc:.0%}' if ra_acc else '—'):>12} {rb_acc:>11.0%}  {d}")

    ajs_a = sa.get("avg_judge_score"); ajs_b = sb.get("avg_judge_score")
    if ajs_b:
        d = f"Δ={ajs_b-ajs_a:+.2f}" if ajs_a else ""
        print(f"  {'Avg judge score':<26} {(f'{ajs_a:.2f}' if ajs_a else '—'):>12} {ajs_b:>11.2f}  {d}")
        for dim in ("avg_faithfulness","avg_relevance","avg_completeness"):
            va = sa.get(dim); vb = sb.get(dim)
            if vb:
                print(f"    {dim[4:]:<24} {(f'{va:.2f}' if va else '—'):>12} {vb:>11.2f}")

    lat_a = sa.get("avg_latency_ms", 0); lat_b = sb.get("avg_latency_ms", 0)
    print(f"  {'Avg latency (ms)':<26} {lat_a:>12,} {lat_b:>12,}  Δ={lat_b-lat_a:+,}")

    # Question-level changes (only for IDs in both)
    common = sorted(set(ra) & set(rb))
    changes = [(qid, ra[qid], rb[qid]) for qid in common
               if ra[qid]["answered"] != rb[qid]["answered"]]
    if changes:
        print(f"\n  Question-level outcome changes ({len(changes)}):")
        for qid, a, b in changes:
            arrow = "REFUSED → ANSWERED ✅" if b["answered"] else "ANSWERED → REFUSED ⚠️"
            print(f"    Q{qid} [{a.get('type','')}]: {arrow}")
            print(f"          {a['question'][:70]}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",          default="v2",   help="Output tag (default: v2)")
    parser.add_argument("--gt",           default="v2",   help="Ground truth file: v1 or v2 (default: v2)")
    parser.add_argument("--judge",        action="store_true", help="Enable LLM-as-judge scoring")
    parser.add_argument("--compare-only", action="store_true", help="Only compare, do not run eval")
    parser.add_argument("--base",         default=None,   help="Baseline file to compare against (default: baseline_v1b.json)")
    args = parser.parse_args()

    base_path = Path(args.base) if args.base else BASELINE_V1B

    if args.compare_only:
        b_path = EVAL_DIR / f"baseline_{args.tag}.json"
        compare_runs(base_path, b_path)
    else:
        run_eval(tag=args.tag, gt_name=args.gt, use_judge=args.judge)
        b_path = EVAL_DIR / f"baseline_{args.tag}.json"
        if base_path.exists() and b_path.exists():
            compare_runs(base_path, b_path)
