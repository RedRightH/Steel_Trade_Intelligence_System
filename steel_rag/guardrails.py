"""
guardrails.py — Domain guardrails for the India Steel Trade Intelligence Platform.

Two functions (as specified in the Day 5 plan):
  is_in_domain(question) -> bool
      Returns False for questions unrelated to steel trade.
      Uses keyword allowlist + NLI zero-shot as fallback.

  is_answer_grounded(answer, context_chunks) -> bool
      Returns False if any sentence in the answer is contradicted by context.
      Wraps the NLI faithfulness check from eval_harness.py.

Red-team test (20 adversarial questions) can be run via:
  python guardrails.py --redteam
"""

import re
import sys
from pathlib import Path

# ── Domain keyword sets ───────────────────────────────────────────────────────

# Any of these → almost certainly in domain
_STEEL_KEYWORDS = {
    "steel", "iron", "coil", "hr coil", "cr coil", "flat rolled",
    "hot rolled", "cold rolled", "galvanized", "galvanised", "tinplate",
    "seamless tube", "pipe", "billet", "slab", "ingot", "rebar", "wire rod",
    "hs 72", "hs 73", "chapter 72", "chapter 73",
    "anti-dumping", "anti dumping", "dumping margin", "dumping duty",
    "safeguard", "countervailing", "cvd", "dgtr", "dgft",
    "ad duty", "provisional duty", "definitive duty",
    "wto", "dsc", "panel report", "appellate body",
    "bis", "is 2062", "qco", "quality control order",
    "cbam", "carbon border", "emissions trading",
    "fta", "cepa", "ecta", "free trade agreement", "aifta",
    "pli", "production linked incentive",
    "tariff", "customs duty", "basic customs", "igst", "bcd",
    "tradestat", "exports", "imports", "trade flow", "trade balance",
    "coking coal", "iron ore", "scrap", "pig iron", "dri", "sponge iron",
    "blast furnace", "bof", "electric arc", "eaf",
    "sail", "tata steel", "jsw", "jspl", "essar",
    "india steel", "indian steel", "ministry of steel",
    "eu cbam", "section 232", "us tariff", "gfsec",
    "gravity model", "bilateral trade", "hs code",
}

# Any of these → definitely out of domain (shortcut to False)
_OUT_OF_DOMAIN_KEYWORDS = {
    "capital of", "president of", "prime minister of",
    "weather", "recipe", "cook", "football", "cricket", "movie",
    "poem", "write a story", "write a poem",
    "bitcoin", "cryptocurrency", "stock market" ,
    "medical", "diagnosis", "symptom", "disease", "treatment",
    "legal advice", "lawsuit", "court case",
    "chemistry", "biology", "physics equation",
}

_NLI_MODEL = None   # lazy-loaded


def _get_nli():
    global _NLI_MODEL
    if _NLI_MODEL is None:
        from sentence_transformers import CrossEncoder
        _NLI_MODEL = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _NLI_MODEL


# ── Public API ────────────────────────────────────────────────────────────────

def is_in_domain(question: str) -> bool:
    """
    Returns True if the question is about steel trade, False otherwise.

    Logic (fast-path first, NLI fallback):
    1. If any out-of-domain keyword matches → False immediately.
    2. If any steel keyword matches → True immediately.
    3. NLI zero-shot: score against a steel-trade hypothesis.
       If entailment probability ≥ 0.4 → True, else False.
    """
    q = question.lower().strip()

    # Fast reject
    for kw in _OUT_OF_DOMAIN_KEYWORDS:
        if kw in q:
            return False

    # Fast accept
    for kw in _STEEL_KEYWORDS:
        if kw in q:
            return True

    # NLI zero-shot fallback
    nli = _get_nli()
    hypothesis = "This question is about steel trade, tariffs, or Indian trade policy."
    scores = nli.predict([(question, hypothesis)], apply_softmax=True)[0]
    # scores: [contradiction, neutral, entailment]
    entailment_prob = float(scores[2])
    return entailment_prob >= 0.4


def is_answer_grounded(answer: str, context_chunks: list) -> bool:
    """
    Returns True if the answer is grounded in context_chunks (no contradictions).
    Returns False if any sentence in the answer is contradicted by the context.

    Args:
        answer: The LLM-generated answer text.
        context_chunks: List of strings or dicts with a 'text' key.
    """
    if not answer or not context_chunks:
        return True  # nothing to check

    # Normalise chunks to strings
    texts = []
    for c in context_chunks:
        if isinstance(c, dict):
            texts.append(c.get("text", c.get("page_content", str(c))))
        else:
            texts.append(str(c))
    context = " ".join(texts)[:3000]  # cap to avoid OOM on reranker

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if len(s.strip()) > 15]
    if not sentences:
        return True

    nli  = _get_nli()
    pairs = [(context, s) for s in sentences]
    scores = nli.predict(pairs, apply_softmax=True)

    contradicted = [s for s, sc in zip(sentences, scores) if sc.argmax() == 0]
    if contradicted:
        print(f"[guardrails] {len(contradicted)}/{len(sentences)} sentences contradicted:")
        for s in contradicted:
            print(f"  ✗ {s[:100]}")
    return len(contradicted) == 0


# ── Red-team test ─────────────────────────────────────────────────────────────

RED_TEAM_QUESTIONS = [
    # OUT-OF-DOMAIN (expect False) — 10 questions
    ("What is the capital of France?",                                      False),
    ("Who won the FIFA World Cup in 2022?",                                 False),
    ("Write a poem about the ocean.",                                       False),
    ("What is the boiling point of water?",                                 False),
    ("Give me a chocolate cake recipe.",                                    False),
    ("Who is the prime minister of the UK?",                                False),
    ("What are Bitcoin's price predictions for 2027?",                      False),
    ("Explain quantum entanglement.",                                       False),
    ("What medications treat hypertension?",                                False),
    ("Who wrote Pride and Prejudice?",                                      False),

    # IN-DOMAIN (expect True) — 10 questions
    ("What are India's anti-dumping duties on HR coil from China?",         True),
    ("Which countries face AD investigation on electrogalvanized steel?",   True),
    ("How does EU CBAM affect Indian flat steel exporters?",                True),
    ("What is IS 2062 and which products does it cover?",                   True),
    ("What triggers a safeguard investigation in India?",                   True),
    ("What is the dumping margin for seamless tubes from China?",           True),
    ("How does the PLI scheme for specialty steel work?",                   True),
    ("What is PCN in the context of anti-dumping proceedings?",             True),
    ("What is the gravity model prediction for India-UAE steel trade?",     True),
    ("What is India's steel export volume under the CEPA with UAE?",        True),
]


def run_redteam(verbose: bool = True) -> dict:
    """Run all 20 red-team questions and report pass/fail counts."""
    correct = 0
    false_positives = []   # out-of-domain accepted
    false_negatives = []   # in-domain rejected

    print("=" * 60)
    print("Red-team test — 20 adversarial questions")
    print("=" * 60)

    for question, expected in RED_TEAM_QUESTIONS:
        result = is_in_domain(question)
        passed = result == expected
        correct += int(passed)
        status = "PASS" if passed else "FAIL"
        if not passed:
            if expected is False:
                false_positives.append(question)
            else:
                false_negatives.append(question)
        if verbose:
            tag = "IN" if result else "OUT"
            exp = "IN" if expected else "OUT"
            print(f"  {status} [{tag}|exp={exp}] {question[:70]}")

    total = len(RED_TEAM_QUESTIONS)
    accuracy = correct / total
    print(f"\nResult: {correct}/{total} correct  ({accuracy:.0%})")
    print(f"False positives (OOD accepted):  {len(false_positives)}")
    print(f"False negatives (in-domain rejected): {len(false_negatives)}")

    gate = len(false_positives) == 0 and len(false_negatives) == 0
    print(f"\nGate test: {'PASS' if gate else 'FAIL'}")
    print("  Target: 0 false positives (no OOD answers), 0 false negatives")

    return {
        "correct": correct,
        "total":   total,
        "accuracy": accuracy,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "gate_pass": gate,
    }


if __name__ == "__main__":
    if "--redteam" in sys.argv:
        run_redteam()
    else:
        # Quick sanity check
        tests = [
            ("What are India's anti-dumping duties on HR coil from China?", True),
            ("What is the capital of France?", False),
            ("How does CBAM affect Indian steel exports to the EU?", True),
            ("Who won the cricket World Cup?", False),
        ]
        print("Guardrails quick test:")
        for q, expected in tests:
            result = is_in_domain(q)
            status = "✓" if result == expected else "✗"
            print(f"  {status} is_in_domain({repr(q)[:50]}) = {result} (expected {expected})")
