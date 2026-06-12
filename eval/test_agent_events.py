"""
eval/test_agent_events.py — 5 Agent Event Gate Tests (Replan Day 9)

Tests PolicyAnalystAgent and SupplyChainRiskAgent on 5 real steel trade events.
Gate: all 5 must return structured JSON with >= 1 source citation and correct agent routing.

Run:  python eval/test_agent_events.py
"""
import os, sys, json, time
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "steel_rag"))
os.chdir(ROOT / "steel_rag")

from router import route_query, PolicyAnalystOutput, SupplyChainRiskOutput

# ── 5 gate events (Replan, Day 9) ────────────────────────────────────────────
GATE_EVENTS = [
    {
        "id": 1,
        "name": "India AD on Chinese HR coil",
        "question": (
            "India has imposed a 12% anti-dumping duty on hot-rolled steel coils "
            "imported from China. What is the policy rationale and trade impact?"
        ),
        "expected_agent": "PolicyAnalystAgent",
        "expected_qtype": ["ANTI_DUMPING"],
        "duty_type_keywords": ["anti-dumping", "dumping"],
        "min_citations": 1,
    },
    {
        "id": 2,
        "name": "Australian port strike — coking coal delay",
        "question": (
            "A major port strike in Australia is delaying coking coal cargo shipments "
            "to India. What is the supply chain risk for Indian steel producers?"
        ),
        "expected_agent": "SupplyChainRiskAgent",
        "expected_qtype": ["RAW_MATERIAL"],
        "risk_level_ok": ["HIGH", "MEDIUM"],
        "commodity_keywords": ["coal", "coking"],
        "min_citations": 1,
    },
    {
        "id": 3,
        "name": "India-UAE CEPA steel concessions",
        "question": (
            "India-UAE Comprehensive Economic Partnership Agreement CEPA steel concessions "
            "have taken effect. Which Indian steel products benefit and what is the export impact?"
        ),
        "expected_agent": "PolicyAnalystAgent",
        "expected_qtype": ["POLICY_OPPORTUNITY", "ANTI_DUMPING", "SAFEGUARD"],
        "duty_type_keywords": ["fta", "cepa", "benefit", "concession", "agreement", "other"],
        "min_citations": 1,
    },
    {
        "id": 4,
        "name": "China restricts scrap steel exports",
        "question": (
            "China has announced restrictions on scrap steel exports. "
            "What is the impact on India's raw material supply chain and steel prices?"
        ),
        "expected_agent": "SupplyChainRiskAgent",
        "expected_qtype": ["RAW_MATERIAL"],
        "commodity_keywords": ["scrap", "steel", "iron"],
        "min_citations": 1,
    },
    {
        "id": 5,
        "name": "EU CBAM reporting for Indian steel firms",
        "question": (
            "EU Carbon Border Adjustment Mechanism CBAM mandatory reporting has begun for "
            "Indian steel exporters. What compliance obligations apply and which grades are affected?"
        ),
        "expected_agent": "SupplyChainRiskAgent",
        "expected_qtype": ["CBAM_COMPLIANCE"],
        "min_citations": 1,
    },
]


def _check_policy(result: PolicyAnalystOutput, event: dict) -> tuple[bool, list[str]]:
    """Validate a PolicyAnalystOutput against gate requirements."""
    issues = []

    # Must have source citations
    if len(result.source_docs) < event.get("min_citations", 1):
        issues.append(f"Only {len(result.source_docs)} citations (need >= {event['min_citations']})")

    # duty_type keyword check (relaxed — model may phrase differently)
    if "duty_type_keywords" in event:
        dt_lower = result.duty_type.lower()
        if not any(kw in dt_lower for kw in event["duty_type_keywords"]):
            issues.append(f"duty_type='{result.duty_type}' not in expected keywords {event['duty_type_keywords']}")

    # answer must be non-empty
    if len(result.answer_text.strip()) < 50:
        issues.append("answer_text too short (< 50 chars)")

    return len(issues) == 0, issues


def _check_supply_chain(result: SupplyChainRiskOutput, event: dict) -> tuple[bool, list[str]]:
    """Validate a SupplyChainRiskOutput against gate requirements."""
    issues = []

    # Must have source citations
    if len(result.source_docs) < event.get("min_citations", 1):
        issues.append(f"Only {len(result.source_docs)} citations (need >= {event['min_citations']})")

    # risk_level check
    if "risk_level_ok" in event:
        if result.risk_level not in event["risk_level_ok"]:
            issues.append(f"risk_level='{result.risk_level}' not in {event['risk_level_ok']}")

    # commodity keyword check
    if "commodity_keywords" in event:
        c_lower = result.commodity.lower()
        if not any(kw in c_lower for kw in event["commodity_keywords"]):
            issues.append(f"commodity='{result.commodity}' not in expected keywords {event['commodity_keywords']}")

    # answer must be non-empty
    if len(result.answer_text.strip()) < 50:
        issues.append("answer_text too short (< 50 chars)")

    return len(issues) == 0, issues


def run_event_gate() -> dict:
    """Run all 5 gate events and return summary."""
    print("\n" + "=" * 70)
    print("AGENT EVENT GATE TEST — 5 Real Steel Trade Events (Replan Day 9)")
    print("=" * 70)

    results = []
    passed = 0

    for ev in GATE_EVENTS:
        print(f"\n[Event {ev['id']}] {ev['name']}")
        print(f"  Q: {ev['question'][:90]}...")
        t0 = time.time()

        try:
            ro = route_query(ev["question"], verbose=False)
            latency_ms = int((time.time() - t0) * 1000)

            # Agent routing check
            agent_ok = ro.agent_used == ev["expected_agent"]
            qtype_ok = ro.question_type in ev["expected_qtype"]

            # Structured output validation
            if isinstance(ro.result, PolicyAnalystOutput):
                struct_ok, issues = _check_policy(ro.result, ev)
                structured_fields = {
                    "duty_type":      ro.result.duty_type,
                    "product":        ro.result.product,
                    "countries":      ro.result.countries,
                    "duty_rate":      ro.result.duty_rate,
                    "source_docs":    ro.result.source_docs,
                    "confidence":     ro.result.confidence,
                    "gravity_scenario": getattr(ro.result, "gravity_scenario", None),
                }
            elif isinstance(ro.result, SupplyChainRiskOutput):
                struct_ok, issues = _check_supply_chain(ro.result, ev)
                structured_fields = {
                    "risk_type":          ro.result.risk_type,
                    "commodity":          ro.result.commodity,
                    "risk_level":         ro.result.risk_level,
                    "key_facts":          ro.result.key_facts[:2],
                    "recommended_action": ro.result.recommended_action,
                    "source_docs":        ro.result.source_docs,
                }
            else:
                struct_ok, issues = False, ["Unexpected result type"]
                structured_fields = {}

            event_pass = agent_ok and qtype_ok and struct_ok

        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            event_pass = False
            agent_ok = qtype_ok = struct_ok = False
            issues = [f"Exception: {e}"]
            structured_fields = {}
            ro = None

        status = "PASS" if event_pass else "FAIL"
        if event_pass:
            passed += 1

        print(f"  Agent  : {ro.agent_used if ro else 'N/A'} (expected {ev['expected_agent']}) {'OK' if agent_ok else 'MISMATCH'}")
        print(f"  QType  : {ro.question_type if ro else 'N/A'} (expected {ev['expected_qtype']}) {'OK' if qtype_ok else 'MISMATCH'}")
        print(f"  Struct : {'OK' if struct_ok else 'ISSUES: ' + '; '.join(issues)}")
        print(f"  Key fields: {structured_fields}")
        print(f"  Latency: {latency_ms}ms  |  Result: [{status}]")

        results.append({
            "event_id":          ev["id"],
            "event_name":        ev["name"],
            "question":          ev["question"],
            "expected_agent":    ev["expected_agent"],
            "actual_agent":      ro.agent_used if ro else None,
            "expected_qtype":    ev["expected_qtype"],
            "actual_qtype":      ro.question_type if ro else None,
            "agent_ok":          agent_ok,
            "qtype_ok":          qtype_ok,
            "struct_ok":         struct_ok,
            "struct_issues":     issues if not struct_ok else [],
            "structured_fields": structured_fields,
            "latency_ms":        latency_ms,
            "pass":              event_pass,
        })

    print("\n" + "=" * 70)
    print(f"GATE RESULT: {passed}/{len(GATE_EVENTS)} events PASS")
    gate_pass = passed == len(GATE_EVENTS)
    print(f"GATE STATUS: {'PASS' if gate_pass else 'FAIL'}")
    print("=" * 70)

    out_path = Path(__file__).parent / "agent_event_results.json"
    summary = {
        "gate_pass":    gate_pass,
        "passed":       passed,
        "total":        len(GATE_EVENTS),
        "events":       results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")
    return summary


if __name__ == "__main__":
    run_event_gate()
