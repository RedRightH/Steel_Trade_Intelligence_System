"""
eval/router_accuracy_test.py — Router classification accuracy gate test.

Tests the question classifier on 10 questions covering all 7 categories.
Gate requirement: >= 8/10 correct (>= 80% accuracy).

Only tests classification (Groq call), not full agent execution — fast to run.

Run:  python eval/router_accuracy_test.py
"""
import os, sys, json, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "steel_rag"))
os.chdir(ROOT / "steel_rag")

from router import classify_question

# 10 questions — 2 per class (7 classes, last 3 double-covered)
ROUTER_TEST_QUESTIONS = [
    # ANTI_DUMPING (2)
    ("What anti-dumping duty has India imposed on seamless steel tubes from China?",
     "ANTI_DUMPING"),
    ("What is the dumping margin for hot-rolled coil imports from Korea under the DGTR investigation?",
     "ANTI_DUMPING"),

    # SAFEGUARD (1)
    ("Which flat steel products are covered by India's current safeguard duty investigation?",
     "SAFEGUARD"),

    # POLICY_OPPORTUNITY (1)
    ("How does the India-UAE CEPA benefit Indian steel exporters?",
     "POLICY_OPPORTUNITY"),

    # RAW_MATERIAL (2)
    ("What is the impact of Australian port disruptions on India's coking coal imports?",
     "RAW_MATERIAL"),
    ("How dependent is India's steel sector on imported iron ore?",
     "RAW_MATERIAL"),

    # CBAM_COMPLIANCE (1)
    ("What are the EU CBAM carbon border reporting obligations for Indian steel exporters?",
     "CBAM_COMPLIANCE"),

    # DATA_ANALYSIS (2)
    ("Which 5 countries receive the most Indian steel exports by value in the last 6 months?",
     "DATA_ANALYSIS"),
    ("What is the year-on-year growth trend for India's flat steel exports to Vietnam?",
     "DATA_ANALYSIS"),

    # TARIFF_ANALYSIS (1)
    ("What is India's MFN tariff rate on hot-rolled steel coils HS 7208 and how has it changed since 2015?",
     "TARIFF_ANALYSIS"),
]


def run_router_accuracy() -> dict:
    """Classify all 10 questions and measure accuracy."""
    print("\n" + "=" * 65)
    print("ROUTER ACCURACY GATE TEST — 10 Questions (7 categories)")
    print("=" * 65)

    results = []
    correct = 0

    for q, expected in ROUTER_TEST_QUESTIONS:
        t0 = time.time()
        predicted, conf, reason = classify_question(q)
        latency_ms = int((time.time() - t0) * 1000)

        ok = predicted == expected
        if ok:
            correct += 1

        status = "PASS" if ok else "FAIL"
        print(f"\n  [{status}] {q[:70]}...")
        print(f"         Expected : {expected}")
        print(f"         Got      : {predicted}  conf={conf:.2f}  [{latency_ms}ms]")
        if not ok:
            print(f"         Reason   : {reason}")

        results.append({
            "question":    q,
            "expected":    expected,
            "predicted":   predicted,
            "confidence":  conf,
            "correct":     ok,
            "latency_ms":  latency_ms,
        })

    accuracy = correct / len(ROUTER_TEST_QUESTIONS)
    gate_pass = accuracy >= 0.80

    print("\n" + "=" * 65)
    print(f"Accuracy    : {correct}/{len(ROUTER_TEST_QUESTIONS)} = {accuracy:.0%}")
    print(f"Gate target : >= 80%")
    print(f"Gate status : {'PASS' if gate_pass else 'FAIL'}")
    print("=" * 65)

    out_path = Path(__file__).parent / "router_accuracy.json"
    summary = {
        "accuracy":       accuracy,
        "correct":        correct,
        "total":          len(ROUTER_TEST_QUESTIONS),
        "gate_pass":      gate_pass,
        "gate_threshold": 0.80,
        "questions":      results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")
    return summary


if __name__ == "__main__":
    run_router_accuracy()
