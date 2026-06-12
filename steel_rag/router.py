"""
router.py - Question router for the India Steel Trade Intelligence Platform.

Classifies each question, picks the right agent, and returns a structured
Pydantic output. Traces each agent call to Langfuse when keys are configured.

Question types → Agents:
  ANTI_DUMPING        → PolicyAnalystAgent  → PolicyAnalystOutput
  SAFEGUARD           → PolicyAnalystAgent  → PolicyAnalystOutput
  POLICY_OPPORTUNITY  → FTAPolicyAgent      → PolicyAnalystOutput
  RAW_MATERIAL        → SupplyChainRiskAgent → SupplyChainRiskOutput
  CBAM_COMPLIANCE     → SupplyChainRiskAgent → SupplyChainRiskOutput
  DATA_ANALYSIS       → DataAnalysisAgent   → DataAnalysisOutput
  TARIFF_ANALYSIS     → TariffAnalysisAgent → TariffAnalysisOutput

Usage:
    from router import route_query
    result = route_query("Which countries are dumping steel into India?")
    print(result.answer_text)

Or run directly: python router.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Literal

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from pydantic import BaseModel, Field
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

QuestionType = Literal[
    "ANTI_DUMPING", "SAFEGUARD", "POLICY_OPPORTUNITY",
    "RAW_MATERIAL", "CBAM_COMPLIANCE", "DATA_ANALYSIS",
    "TARIFF_ANALYSIS"
]

_groq_client = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Pydantic Output Schemas ───────────────────────────────────────────────────

class PolicyAnalystOutput(BaseModel):
    """Structured output for AD, safeguard, and FTA policy questions."""
    question_type:    QuestionType
    duty_type:        str  = Field(description="e.g. 'Anti-Dumping Duty' | 'Safeguard Duty' | 'FTA Benefit'")
    product:          str  = Field(description="Product under consideration")
    countries:        list[str] = Field(description="Countries subject to the measure")
    duty_rate:        str  = Field(description="Duty rate or measure level, e.g. '18.5%' or 'Not specified'")
    effective_date:   str  = Field(description="Date measure was imposed or investigated, or 'Not specified'")
    source_docs:      list[str] = Field(description="Source document filenames cited")
    confidence:       float = Field(ge=0.0, le=1.0, description="RAG answer confidence 0-1")
    answer_text:      str  = Field(description="Full human-readable answer with citations")
    gravity_scenario: str | None = Field(
        default=None,
        description="Gravity model trade flow prediction for relevant countries, if applicable",
    )


class SupplyChainRiskOutput(BaseModel):
    """Structured output for raw material and CBAM/green steel questions."""
    question_type:        QuestionType
    risk_type:            str  = Field(description="'RAW_MATERIAL' | 'CBAM_COMPLIANCE' | 'EMISSIONS'")
    commodity:            str  = Field(description="e.g. 'coking coal', 'iron ore', 'carbon credits'")
    risk_level:           Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
    key_facts:            list[str] = Field(description="Bullet facts extracted from context")
    recommended_action:   str  = Field(description="Suggested response for Indian steel industry")
    source_docs:          list[str] = Field(description="Source document filenames cited")
    answer_text:          str  = Field(description="Full human-readable answer with citations")


class DataAnalysisOutput(BaseModel):
    """Structured output for trade statistics and trend questions."""
    question_type:  QuestionType = "DATA_ANALYSIS"
    analysis_focus: str  = Field(description="e.g. 'country trend', 'regional breakdown', 'market growth'")
    period:         str  = Field(description="Time period covered, e.g. 'Jan 2024 - Feb 2026'")
    key_numbers:    list[str] = Field(description="Top 3-5 headline numbers from the analysis")
    chart_path:     str | None = Field(description="Path to generated chart PNG, or None")
    answer_text:    str  = Field(description="Full narrative answer with numbers")


class TariffAnalysisOutput(BaseModel):
    """Structured output for MFN tariff rate questions."""
    question_type:  QuestionType = "TARIFF_ANALYSIS"
    hs_codes:       list[str]  = Field(description="HS-6 codes identified in the question")
    tariff_rates:   list[str]  = Field(description="Key rates found, e.g. ['HS 7304: 10.0% (2023)']")
    trend:          str        = Field(description="Direction of tariff change: 'rising', 'falling', 'stable', or 'N/A'")
    period:         str        = Field(description="Years covered, e.g. '2010-2023'")
    chart_path:     str | None = Field(description="Path to generated chart PNG, or None")
    answer_text:    str        = Field(description="Full narrative answer with specific rates")


class RouterOutput(BaseModel):
    """Wrapper returned by route_query()."""
    question:      str
    question_type: QuestionType
    agent_used:    str
    latency_ms:    int
    result:        PolicyAnalystOutput | SupplyChainRiskOutput | DataAnalysisOutput | TariffAnalysisOutput


# ── Question Classifier ───────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """You are a question classifier for an India steel trade intelligence system.

Classify the question into EXACTLY one of these categories:

ANTI_DUMPING      - Questions about AD duties, dumping margins, DGTR investigations,
                    anti-dumping measures on specific products or countries.
SAFEGUARD         - Questions about safeguard duties, import surges, serious injury
                    to domestic industry, provisional/final safeguard measures.
POLICY_OPPORTUNITY - Questions about FTAs, PLI scheme, export promotion, trade
                    agreements, market access, RCEP, CEPA, DGFT policy.
RAW_MATERIAL      - Questions about coking coal, iron ore, scrap steel, input costs,
                    supply disruptions, raw material availability.
CBAM_COMPLIANCE   - Questions about EU CBAM, carbon border tax, green steel,
                    emissions reporting, decarbonisation.
DATA_ANALYSIS     - Questions about export/import numbers, trade statistics, trends,
                    growth rates, country rankings, market comparisons, charts,
                    monthly data, YoY growth, top destinations.
TARIFF_ANALYSIS   - Questions about MFN (Most Favoured Nation) tariff rates, applied
                    duties, HS code tariffs, WTO tariff schedules, how tariff rates
                    have changed over years, highest/lowest duty products.

Return ONLY a JSON object: {"type": "<CATEGORY>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}
No markdown, no explanation outside the JSON.
"""


def classify_question(question: str, memory=None) -> tuple[QuestionType, float, str]:
    """
    Classify a question into one of 7 categories.
    Injects conversation history when memory is provided so follow-ups
    (e.g. 'what about Vietnam?') resolve correctly.
    Returns (question_type, confidence, reason).
    """
    user_content = question
    if memory and not memory.is_empty:
        ctx = memory.classifier_context(n=3)
        user_content = f"{ctx}\n\nNew question: {question}"
    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        raw = resp.choices[0].message.content.strip()
        start = raw.find("{"); end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        qtype  = parsed["type"].upper()
        conf   = float(parsed.get("confidence", 0.8))
        reason = parsed.get("reason", "")
        if qtype not in QuestionType.__args__:
            qtype = "TARIFF_ANALYSIS" if "tariff" in question.lower() or "mfn" in question.lower() else "DATA_ANALYSIS"
        return qtype, conf, reason
    except Exception as e:
        return "DATA_ANALYSIS", 0.5, f"Classifier error: {e}"


# ── Answer synthesizer (fallback for short/refused RAG answers) ───────────────

_SYNTH_SYSTEM = (
    "You are an India steel trade intelligence expert. "
    "Answer the question using the provided knowledge-base context. "
    "If context is thin, answer from general steel trade policy knowledge with a caveat. "
    "Be concise but complete — at least 2 full sentences."
)


def _synthesize_answer(question: str, context: str, answer: str) -> str:
    """Return a longer answer if RAG answer was a refusal or too short."""
    if len(answer.strip()) >= 100:
        return answer
    prompt = (
        f"Context from knowledge base:\n{context[:3000]}\n\n"
        f"Question: {question}\n\n"
        "Provide a comprehensive 2-3 sentence answer."
    )
    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYNTH_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        synth = resp.choices[0].message.content.strip()
        return synth if len(synth) > len(answer) else answer
    except Exception:
        return answer


# ── Agent: Policy Analyst ─────────────────────────────────────────────────────

POLICY_EXTRACTION_SYSTEM = """You are a structured data extractor for Indian steel trade policy.

Given a RAG answer and its source context, extract structured fields.
Return ONLY valid JSON matching this schema — no markdown:

{
  "duty_type":      "<Anti-Dumping Duty | Safeguard Duty | FTA Benefit | Other>",
  "product":        "<product name from context>",
  "countries":      ["<country1>", "<country2>"],
  "duty_rate":      "<rate or 'Not specified'>",
  "effective_date": "<date or 'Not specified'>",
  "source_docs":    ["<filename1>"],
  "confidence":     <0.0-1.0>
}

Rules:
- Extract ONLY information present in the answer/context — do not invent.
- If a field is not mentioned, use "Not specified" or [].
- confidence = 1.0 if answer is specific and cited, 0.5 if partial, 0.0 if refused.
"""


def _run_policy_agent(question: str, question_type: QuestionType, memory=None) -> PolicyAnalystOutput:
    from rag import rag_query
    # Enrich retrieval query with memory context so follow-ups retrieve correctly
    retrieval_q = question
    if memory and not memory.is_empty:
        retrieval_q = f"{memory.agent_context(n=2)}\nCurrent question: {question}"
    rag_result = rag_query(retrieval_q)
    answer     = rag_result["answer"]
    context    = rag_result["context_used"]
    sources    = [s["file_name"] for s in rag_result["sources"]]

    # Synthesize a fuller answer if RAG returned a refusal or short text
    answer = _synthesize_answer(question, context, answer)

    # Extract structured fields
    mem_ctx = (memory.agent_context(n=2) + "\n\n") if (memory and not memory.is_empty) else ""
    extraction_prompt = (
        f"{mem_ctx}RAG ANSWER:\n{answer}\n\n"
        f"CONTEXT (first 2000 chars):\n{context[:2000]}"
    )
    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": POLICY_EXTRACTION_SYSTEM},
                {"role": "user",   "content": extraction_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        start = raw.find("{"); end = raw.rfind("}") + 1
        fields = json.loads(raw[start:end])
    except Exception:
        fields = {
            "duty_type": "Other", "product": "Steel",
            "countries": [], "duty_rate": "Not specified",
            "effective_date": "Not specified",
            "source_docs": sources, "confidence": 0.5,
        }

    # Apply sensible defaults when extraction found nothing
    if not fields.get("duty_type") or fields["duty_type"] in ("Not specified", ""):
        _type_defaults = {
            "ANTI_DUMPING":       "Anti-Dumping Duty",
            "SAFEGUARD":          "Safeguard Duty",
            "POLICY_OPPORTUNITY": "FTA Benefit",
        }
        fields["duty_type"] = _type_defaults.get(question_type, "Other")

    # Gravity model scenario — run for FTA/CBAM questions where trade-flow impact is relevant
    gravity_text = None
    if question_type in ("POLICY_OPPORTUNITY", "CBAM_COMPLIANCE"):
        countries_detected = fields.get("countries", [])
        if not countries_detected:
            # Try to extract from question keywords (UAE, EU, etc.)
            kw_map = {
                "uae": "U ARAB EMTS", "u.a.e": "U ARAB EMTS",
                "eu": None,           "europe": None,
                "usa": "U S A",       "us ": "U S A",
                "china": "CHINA P RP",
                "vietnam": "VIETNAM SOC REP",
            }
            ql = question.lower()
            countries_detected = [v for k, v in kw_map.items() if k in ql and v]

        if countries_detected:
            try:
                from gravity_model import predict_trade_flow, ensure_model_ready
                ensure_model_ready()
                gravity_parts = []
                for c in countries_detected[:3]:  # max 3 countries
                    res = predict_trade_flow(c, model_type="xgb")
                    if res.get("status") != "no_data":
                        chg = res.get("change_pct", 0)
                        gravity_parts.append(
                            f"{c}: baseline ${res.get('baseline_usd', 0):.1f}M/yr, "
                            f"scenario {chg:+.1f}% ({res.get('scenario', 'baseline')})"
                        )
                if gravity_parts:
                    gravity_text = "Gravity model predictions — " + " | ".join(gravity_parts)
            except Exception:
                pass

    return PolicyAnalystOutput(
        question_type    = question_type,
        duty_type        = fields.get("duty_type", "Other"),
        product          = fields.get("product", "Steel"),
        countries        = fields.get("countries", []),
        duty_rate        = fields.get("duty_rate", "Not specified"),
        effective_date   = fields.get("effective_date", "Not specified"),
        source_docs      = fields.get("source_docs") or sources,
        confidence       = float(fields.get("confidence", 0.5)),
        answer_text      = answer,
        gravity_scenario = gravity_text,
    )


# ── Agent: Supply Chain Risk ──────────────────────────────────────────────────

RISK_EXTRACTION_SYSTEM = """You are a structured data extractor for steel supply chain risk analysis.

Given a RAG answer about raw materials or CBAM compliance, extract structured fields.
Return ONLY valid JSON — no markdown:

{
  "risk_type":          "<RAW_MATERIAL | CBAM_COMPLIANCE | EMISSIONS>",
  "commodity":          "<coking coal | iron ore | scrap | carbon credits | other>",
  "risk_level":         "<HIGH | MEDIUM | LOW | UNKNOWN>",
  "key_facts":          ["<fact1>", "<fact2>", "<fact3>"],
  "recommended_action": "<one sentence recommendation for Indian steel industry>",
  "source_docs":        ["<filename1>"]
}

Rules:
- risk_level HIGH = immediate disruption or >20% cost impact mentioned.
- risk_level MEDIUM = moderate concern, supply tightness, 10-20% impact.
- risk_level LOW = minor or long-term concern.
- Extract ONLY from the provided answer/context.
"""


def _run_supply_chain_agent(question: str, question_type: QuestionType, memory=None) -> SupplyChainRiskOutput:
    from rag import rag_query
    retrieval_q = question
    if memory and not memory.is_empty:
        retrieval_q = f"{memory.agent_context(n=2)}\nCurrent question: {question}"
    rag_result = rag_query(retrieval_q)
    answer     = rag_result["answer"]
    context    = rag_result["context_used"]
    sources    = [s["file_name"] for s in rag_result["sources"]]

    # Synthesize a fuller answer if RAG returned a refusal or short text
    answer = _synthesize_answer(question, context, answer)

    mem_ctx = (memory.agent_context(n=2) + "\n\n") if (memory and not memory.is_empty) else ""
    extraction_prompt = (
        f"{mem_ctx}RAG ANSWER:\n{answer}\n\n"
        f"CONTEXT (first 2000 chars):\n{context[:2000]}"
    )
    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": RISK_EXTRACTION_SYSTEM},
                {"role": "user",   "content": extraction_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        start = raw.find("{"); end = raw.rfind("}") + 1
        fields = json.loads(raw[start:end])
    except Exception:
        fields = {
            "risk_type": question_type, "commodity": "steel inputs",
            "risk_level": "UNKNOWN", "key_facts": [answer[:200]],
            "recommended_action": "Monitor situation closely.",
            "source_docs": sources,
        }

    return SupplyChainRiskOutput(
        question_type      = question_type,
        risk_type          = fields.get("risk_type", question_type),
        commodity          = fields.get("commodity", "steel inputs"),
        risk_level         = fields.get("risk_level", "UNKNOWN"),
        key_facts          = fields.get("key_facts", []),
        recommended_action = fields.get("recommended_action", ""),
        source_docs        = fields.get("source_docs") or sources,
        answer_text        = answer,
    )


# ── Agent: Data Analysis ──────────────────────────────────────────────────────

DATA_FOCUS_SYSTEM = """You are a query analyser for a steel trade data system.

Given a question about India's steel export statistics, identify:
1. What KIND of analysis is needed.
2. What the key numbers/facts from the answer are.

Return ONLY valid JSON:
{
  "analysis_focus": "<country trend | regional breakdown | market growth | top destinations | yoy comparison | other>",
  "period":         "<time period mentioned or 'latest available'>",
  "key_numbers":    ["<number + context>", ...]
}
"""


def _run_data_agent(question: str, memory=None) -> DataAnalysisOutput:
    from data_agent import query_export_data

    enriched_q = question
    if memory and not memory.is_empty:
        enriched_q = f"{memory.agent_context(n=2)}\nCurrent question: {question}"
    result = query_export_data(enriched_q)
    answer = result["answer"]

    # Extract focus metadata
    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": DATA_FOCUS_SYSTEM},
                {"role": "user",
                 "content": f"QUESTION: {question}\nANSWER: {answer}"},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        start = raw.find("{"); end = raw.rfind("}") + 1
        meta = json.loads(raw[start:end])
    except Exception:
        meta = {
            "analysis_focus": "trade statistics",
            "period": "latest available",
            "key_numbers": [answer[:200]],
        }

    return DataAnalysisOutput(
        question_type  = "DATA_ANALYSIS",
        analysis_focus = meta.get("analysis_focus", "trade statistics"),
        period         = meta.get("period", "latest available"),
        key_numbers    = meta.get("key_numbers", []),
        chart_path     = result.get("chart_path"),
        answer_text    = answer,
    )


# ── Agent: Tariff Analysis ───────────────────────────────────────────────────

TARIFF_META_SYSTEM = """You are a query analyser for a steel MFN tariff system.

Given a question and a tariff answer (which may be a data table or narrative), extract:
1. Which HS codes were looked up — 4-6 digit codes like 7208, 720810, 7304.
2. Key rates found — combine hs_code + rate + year, e.g. "HS 7208: 7.5% (2023)".
3. Whether the tariff trend is rising, falling, or stable (compare earliest to latest year shown).
4. The time period covered (first_year to last_year).

Return ONLY valid JSON — no markdown:
{
  "hs_codes":     ["720810", "720811"],
  "tariff_rates": ["HS 7208: 7.5% (2023)", "HS 7208: 5.0% (2015)"],
  "trend":        "rising",
  "period":       "2015-2023"
}

If no rates found, use empty lists and "N/A" for trend.
"""


def _run_tariff_agent(question: str, memory=None) -> TariffAnalysisOutput:
    from tariff_agent import query_tariff

    enriched_q = question
    if memory and not memory.is_empty:
        enriched_q = f"{memory.agent_context(n=2)}\nCurrent question: {question}"
    result     = query_tariff(enriched_q)
    answer     = result["answer"]
    chart_path = result.get("chart_path")

    # Extract metadata via LLM
    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": TARIFF_META_SYSTEM},
                {"role": "user",
                 "content": f"QUESTION: {question}\nANSWER: {answer}"},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw   = resp.choices[0].message.content.strip()
        start = raw.find("{"); end = raw.rfind("}") + 1
        meta  = json.loads(raw[start:end])
    except Exception:
        meta = {
            "hs_codes":     [],
            "tariff_rates": [],
            "trend":        "N/A",
            "period":       "2010-2023",
        }

    return TariffAnalysisOutput(
        question_type = "TARIFF_ANALYSIS",
        hs_codes      = meta.get("hs_codes", []),
        tariff_rates  = meta.get("tariff_rates", []),
        trend         = meta.get("trend", "N/A"),
        period        = meta.get("period", "2010-2023"),
        chart_path    = chart_path,
        answer_text   = answer,
    )


# ── Main Router ───────────────────────────────────────────────────────────────

def route_query(question: str, verbose: bool = False, memory=None) -> RouterOutput:
    """
    Classify question → pick agent → return structured output.

    Args:
        question: Natural language question about India steel trade.
        verbose:  Print classification and timing info.
        memory:   Optional ConversationMemory for multi-turn context.

    Returns:
        RouterOutput containing the structured agent result.
    """
    t_start = time.time()

    # Step 1: Classify (with conversation context if available)
    qtype, conf, reason = classify_question(question, memory=memory)

    if verbose:
        mem_info = f"  memory={memory.turn_count} turns" if memory else ""
        print(f"  [Router] Type={qtype}  conf={conf:.2f}  reason={reason}{mem_info}")

    # Step 2: Route to agent (pass memory for context injection)
    if qtype == "DATA_ANALYSIS":
        agent_name = "DataAnalysisAgent"
        result = _run_data_agent(question, memory=memory)

    elif qtype == "TARIFF_ANALYSIS":
        agent_name = "TariffAnalysisAgent"
        result = _run_tariff_agent(question, memory=memory)

    elif qtype in ("ANTI_DUMPING", "SAFEGUARD", "POLICY_OPPORTUNITY"):
        agent_name = "PolicyAnalystAgent"
        result = _run_policy_agent(question, qtype, memory=memory)

    else:  # RAW_MATERIAL, CBAM_COMPLIANCE
        agent_name = "SupplyChainRiskAgent"
        result = _run_supply_chain_agent(question, qtype, memory=memory)

    latency_ms = int((time.time() - t_start) * 1000)

    return RouterOutput(
        question      = question,
        question_type = qtype,
        agent_used    = agent_name,
        latency_ms    = latency_ms,
        result        = result,
    )


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_result(ro: RouterOutput):
    print("\n" + "=" * 65)
    print(f"Q: {ro.question}")
    print(f"   Type={ro.question_type}  Agent={ro.agent_used}  [{ro.latency_ms}ms]")
    print("=" * 65)

    r = ro.result

    if isinstance(r, PolicyAnalystOutput):
        print(f"  Duty type      : {r.duty_type}")
        print(f"  Product        : {r.product}")
        print(f"  Countries      : {', '.join(r.countries) or 'N/A'}")
        print(f"  Duty rate      : {r.duty_rate}")
        print(f"  Effective date : {r.effective_date}")
        print(f"  Confidence     : {r.confidence:.0%}")
        print(f"  Sources        : {', '.join(r.source_docs[:2])}")
        print(f"\n  Answer: {r.answer_text[:300]}...")

    elif isinstance(r, SupplyChainRiskOutput):
        print(f"  Risk type      : {r.risk_type}")
        print(f"  Commodity      : {r.commodity}")
        print(f"  Risk level     : {r.risk_level}")
        print(f"  Key facts      :")
        for f in r.key_facts[:4]:
            print(f"    - {f}")
        print(f"  Recommendation : {r.recommended_action}")
        print(f"  Sources        : {', '.join(r.source_docs[:2])}")
        print(f"\n  Answer: {r.answer_text[:300]}...")

    elif isinstance(r, DataAnalysisOutput):
        print(f"  Focus          : {r.analysis_focus}")
        print(f"  Period         : {r.period}")
        print(f"  Key numbers    :")
        for n in r.key_numbers[:5]:
            print(f"    - {n}")
        if r.chart_path:
            print(f"  Chart saved    : {r.chart_path}")
        print(f"\n  Answer: {r.answer_text[:300]}...")

    elif isinstance(r, TariffAnalysisOutput):
        print(f"  HS codes       : {', '.join(r.hs_codes) or 'N/A'}")
        print(f"  Trend          : {r.trend}  |  Period: {r.period}")
        print(f"  Tariff rates   :")
        for rate in r.tariff_rates[:5]:
            print(f"    - {rate}")
        if r.chart_path:
            print(f"  Chart saved    : {r.chart_path}")
        print(f"\n  Answer: {r.answer_text[:300]}...")


# ── Gate test ─────────────────────────────────────────────────────────────────

GATE_QUESTIONS = [
    # (question, expected_type, expected_agent)
    ("What anti-dumping duty was imposed on seamless tubes from China?",
     "ANTI_DUMPING", "PolicyAnalystAgent"),

    ("What products are covered by India's safeguard investigation on steel flat products?",
     "SAFEGUARD", "PolicyAnalystAgent"),

    ("Which 5 countries receive the most Indian steel exports by value?",
     "DATA_ANALYSIS", "DataAnalysisAgent"),

    ("What are the growing and shrinking markets for Indian steel exports in the last 6 months?",
     "DATA_ANALYSIS", "DataAnalysisAgent"),

    ("How does the EU CBAM affect Indian steel exporters?",
     "CBAM_COMPLIANCE", "SupplyChainRiskAgent"),

    ("What is India's MFN tariff rate on hot-rolled steel coils HS 7208 and how has it changed since 2015?",
     "TARIFF_ANALYSIS", "TariffAnalysisAgent"),
]


def run_gate_test():
    n_questions = len(GATE_QUESTIONS)
    print("\n" + "=" * 65)
    print(f"DAY 4 GATE TEST — Router ({n_questions} questions)")
    print("=" * 65)

    passed = 0
    for question, expected_type, expected_agent in GATE_QUESTIONS:
        ro = route_query(question, verbose=False)
        type_ok  = ro.question_type == expected_type
        agent_ok = ro.agent_used    == expected_agent
        ok       = type_ok and agent_ok
        if ok:
            passed += 1

        status = "PASS" if ok else "FAIL"
        print(f"\n  [{status}] {question[:60]}...")
        print(f"         Expected : {expected_type} → {expected_agent}")
        print(f"         Got      : {ro.question_type} → {ro.agent_used}  [{ro.latency_ms}ms]")

        # Show key structured field
        r = ro.result
        if isinstance(r, PolicyAnalystOutput):
            print(f"         Product  : {r.product}  |  Rate: {r.duty_rate}  |  Conf: {r.confidence:.0%}")
        elif isinstance(r, SupplyChainRiskOutput):
            print(f"         Risk     : {r.risk_level}  |  Commodity: {r.commodity}")
        elif isinstance(r, DataAnalysisOutput):
            print(f"         Focus    : {r.analysis_focus}  |  Chart: {'yes' if r.chart_path else 'no'}")
        elif isinstance(r, TariffAnalysisOutput):
            print(f"         HS codes : {', '.join(r.hs_codes[:3]) or 'N/A'}  |  Trend: {r.trend}")

    print(f"\nGate result: {passed}/{n_questions} passed")
    print(f"Gate status: {'PASS' if passed == n_questions else 'FAIL'}")
    return passed == n_questions


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true", help="Run gate test only")
    parser.add_argument("--question", "-q", type=str, help="Route a single question")
    args = parser.parse_args()

    if args.question:
        ro = route_query(args.question, verbose=True)
        print_result(ro)

    elif args.gate:
        run_gate_test()

    else:
        # Demo mode: run gate test + a few extra examples
        run_gate_test()

        extras = [
            "Compare Vietnam and Italy steel export trends over the last year",
            "What is India's Foreign Trade Policy 2023 aiming to achieve?",
        ]
        print("\n" + "=" * 65)
        print("ADDITIONAL EXAMPLES")
        print("=" * 65)
        for q in extras:
            ro = route_query(q, verbose=True)
            print_result(ro)
            print()
