"""
eval_harness.py - Faithfulness evaluation for the Steel RAG system.

Two complementary scorers:
  1. NLI scorer  - cross-encoder/nli-deberta-v3-small (local, CPU)
  2. LLM judge   - Groq llama-3.3-70b-versatile

score_triple(question, context, answer) -> dict is the main export.

Usage:
    python eval_harness.py          # runs full baseline eval on ground_truth.json
    from eval_harness import score_triple
"""

import sys
import json
import time
import csv
import os
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROUND_TRUTH = Path(__file__).parent / "eval" / "ground_truth.json"
OUTPUT_CSV   = Path(__file__).parent / "eval" / "baseline_v1.csv"
OUTPUT_JSON  = Path(__file__).parent / "eval" / "baseline_v1_scored.json"
# ─────────────────────────────────────────────────────────────────────────────

_nli_model = None
_groq_client = None


# ── NLI Scorer ────────────────────────────────────────────────────────────────

def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading NLI model: {NLI_MODEL} ...")
        _nli_model = CrossEncoder(NLI_MODEL)
        print("NLI model loaded.")
    return _nli_model


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter on . ! ? boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def nli_faithfulness(context: str, answer: str) -> dict:
    """
    For each sentence in the answer, check if context ENTAILS or CONTRADICTS it.
    Score = non-contradicted sentences / total sentences.
    NLI labels from deberta: 0=contradiction, 1=entailment, 2=neutral
    Refusal answers ("I cannot answer...") score 1.0 — no hallucination.
    """
    # Refusal = no hallucination = perfect faithfulness
    if "cannot answer" in answer.lower():
        return {"score": 1.0, "total": 0, "contradicted": 0, "entailed": 0,
                "latency_ms": 0, "details": [{"sentence": answer[:80], "label": "refusal", "confidence": 1.0}]}

    model = _get_nli_model()
    sentences = _split_sentences(answer)

    if not sentences:
        return {"score": 1.0, "total": 0, "contradicted": 0, "entailed": 0, "details": []}

    pairs = [(context, s) for s in sentences]
    t0 = time.time()
    scores = model.predict(pairs, apply_softmax=True)
    latency_ms = int((time.time() - t0) * 1000)

    details = []
    contradicted = 0
    entailed = 0

    for sentence, score_arr in zip(sentences, scores):
        label_idx = int(score_arr.argmax())
        label = ["contradiction", "entailment", "neutral"][label_idx]
        conf = float(score_arr[label_idx])

        if label == "contradiction":
            contradicted += 1
        elif label == "entailment":
            entailed += 1

        details.append({
            "sentence": sentence[:100],
            "label": label,
            "confidence": round(conf, 3),
        })

    faith_score = round((len(sentences) - contradicted) / len(sentences), 3)

    return {
        "score": faith_score,
        "total": len(sentences),
        "contradicted": contradicted,
        "entailed": entailed,
        "latency_ms": latency_ms,
        "details": details,
    }


# ── LLM Judge ─────────────────────────────────────────────────────────────────

def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


LLM_JUDGE_SYSTEM = """You are a faithfulness evaluator for a RAG system about Indian steel trade policy.

Given a CONTEXT (retrieved document chunks) and an ANSWER generated from that context,
rate how faithful the answer is to the context.

Faithfulness means: every claim in the answer is supported by the context.
Penalise: hallucinated figures, invented country names, unsupported claims.
Do NOT penalise: refusing to answer, saying "I cannot answer from the documents."

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{"faithfulness": <float 0.0-1.0>, "reason": "<one sentence>"}

Scoring guide:
1.0 = every claim directly supported by context
0.7-0.9 = mostly supported, minor gaps
0.4-0.6 = some claims unsupported or vague
0.0-0.3 = significant hallucination or contradiction"""


def llm_faithfulness(question: str, context: str, answer: str) -> dict:
    """Use Groq LLM as a faithfulness judge. Returns score + reason."""
    if "cannot answer" in answer.lower():
        return {"score": 1.0, "reason": "System correctly refused to answer (no hallucination)."}

    client = _get_groq()
    user_msg = f"CONTEXT:\n{context[:3000]}\n\nANSWER:\n{answer}"

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        score = float(parsed["faithfulness"])
        reason = str(parsed["reason"])
    except json.JSONDecodeError:
        score = 0.5
        reason = f"Parse error — raw: {raw[:80]}"
    except Exception as e:
        score = 0.0
        reason = f"Error: {e}"

    latency_ms = int((time.time() - t0) * 1000)
    return {"score": round(score, 3), "reason": reason, "latency_ms": latency_ms}


# ── score_triple — main export ─────────────────────────────────────────────────

def score_triple(question: str, context: str, answer: str) -> dict:
    """
    Score a (question, context, answer) triple with both NLI and LLM judge.

    Returns:
        nli_faithfulness     float  0-1
        llm_faithfulness     float  0-1
        llm_reason           str
        agreement            bool   (scores within 0.2 of each other)
        nli_details          list
        latency_ms           int    (combined)
    """
    nli = nli_faithfulness(context, answer)
    llm = llm_faithfulness(question, context, answer)

    agreement = abs(nli["score"] - llm["score"]) <= 0.2

    return {
        "nli_faithfulness":  nli["score"],
        "llm_faithfulness":  llm["score"],
        "llm_reason":        llm["reason"],
        "agreement":         agreement,
        "nli_contradicted":  nli["contradicted"],
        "nli_entailed":      nli["entailed"],
        "nli_total":         nli["total"],
        "nli_details":       nli["details"],
        "latency_ms":        nli.get("latency_ms", 0) + llm.get("latency_ms", 0),
    }


# ── Full baseline run ──────────────────────────────────────────────────────────

def run_baseline():
    from rag import rag_query

    with open(GROUND_TRUTH) as f:
        qa_pairs = json.load(f)

    print("=" * 60)
    print("STEEL RAG - Day 2 Eval Harness (NLI + LLM Judge)")
    print("=" * 60)
    print()

    rows = []
    scored = []

    for i, qa in enumerate(qa_pairs, 1):
        q = qa["question"]
        print(f"[{i}/10] {q[:65]}...")

        # Get RAG answer
        t0 = time.time()
        result = rag_query(q)
        rag_ms = int((time.time() - t0) * 1000)

        context = result["context_used"]
        answer  = result["answer"]

        # Score it
        scores = score_triple(q, context, answer)

        nli   = scores["nli_faithfulness"]
        llm   = scores["llm_faithfulness"]
        agree = "YES" if scores["agreement"] else "NO"

        print(f"  NLI={nli:.2f}  LLM={llm:.2f}  Agree={agree}")
        print(f"  Reason: {scores['llm_reason'][:80]}")
        print()

        rows.append({
            "id":               qa["id"],
            "question":         q,
            "type":             qa["type"],
            "answer":           answer[:120],
            "nli_faithfulness": nli,
            "llm_faithfulness": llm,
            "agreement":        scores["agreement"],
            "nli_contradicted": scores["nli_contradicted"],
            "llm_reason":       scores["llm_reason"],
            "rag_latency_ms":   rag_ms,
            "eval_latency_ms":  scores["latency_ms"],
        })

        scored.append({**qa, "scores": scores, "actual_answer": answer})
        time.sleep(0.3)

    # Summary stats
    avg_nli = sum(r["nli_faithfulness"] for r in rows) / len(rows)
    avg_llm = sum(r["llm_faithfulness"] for r in rows) / len(rows)
    agree_count = sum(1 for r in rows if r["agreement"])

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Avg NLI faithfulness : {avg_nli:.3f}")
    print(f"  Avg LLM faithfulness : {avg_llm:.3f}")
    print(f"  Agreement (within 0.2): {agree_count}/10")
    print()

    gate_nli   = avg_nli >= 0.65
    gate_agree = agree_count >= 8
    print(f"  Gate test - NLI avg >= 0.65  : {'PASS' if gate_nli else 'FAIL'} ({avg_nli:.3f})")
    print(f"  Gate test - Agreement >= 8/10: {'PASS' if gate_agree else 'FAIL'} ({agree_count}/10)")
    print()

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved  -> {OUTPUT_CSV}")

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": {
            "avg_nli": round(avg_nli, 3),
            "avg_llm": round(avg_llm, 3),
            "agreement_count": agree_count,
            "gate_nli_pass": gate_nli,
            "gate_agree_pass": gate_agree,
        }, "results": scored}, f, indent=2, ensure_ascii=False)
    print(f"JSON saved -> {OUTPUT_JSON}")


if __name__ == "__main__":
    run_baseline()
