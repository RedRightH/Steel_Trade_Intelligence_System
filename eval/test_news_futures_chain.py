"""
eval/test_news_futures_chain.py — Gate test for the news → futures → markets chain.

Gates:
  1. GPR-conditioned forecast : forecast_price(gpr_df=...) returns gpr_used=True
                                and a fitted regressor coefficient
  2. Calibrated news impact   : analyze_news_impact applies the event-study
                                calibration factor and returns non-zero impacts
  3. Market opportunity ranker: >= 10 ranked markets with finite scores
  4. Event → market mapping   : markets_affected_by_event resolves event countries
                                into the ranking with event flags

Run:  python eval/test_news_futures_chain.py   (default Python, from steel_rag/)
"""
import os, sys, json, math
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "steel_rag"))
os.chdir(ROOT / "steel_rag")

results = {}


def gate(name, ok, detail):
    results[name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


print("=" * 72)
print("NEWS → FUTURES → MARKETS CHAIN — Gate Test")
print("=" * 72)

# ── Gate 1: GPR-conditioned forecast ─────────────────────────────────────────
print("\n[1] GPR-conditioned Prophet forecast")
from steel_futures import fetch_ticker, compute_technicals, forecast_price, load_steel_gpr_index

df  = compute_technicals(fetch_ticker("HRC=F", period="2y"))
gpr = load_steel_gpr_index()
fc  = forecast_price(df, days=60, gpr_df=gpr)
gate("gpr_regressor_active",
     fc.get("gpr_used") is True and fc.get("gpr_coef") is not None,
     f"gpr_used={fc.get('gpr_used')}, beta={fc.get('gpr_coef')}, "
     f"target_30={fc.get('target_30'):.1f} ({fc.get('change_pct_30'):+.1f}%)")

# ── Gate 2: calibrated news impact ───────────────────────────────────────────
print("\n[2] Calibrated news impact analysis")
from steel_futures import analyze_news_impact, CALIBRATION_PATH

ann = ("India has imposed a 12% anti-dumping duty on hot-rolled steel coil "
       "imports from China following a DGTR investigation.")
impact = analyze_news_impact(ann)
calib_exists = CALIBRATION_PATH.exists()
gate("calibration_applied",
     calib_exists and impact.get("calibration_factor") is not None
     and impact.get("futures_impact_pct") != 0,
     f"calib_file={calib_exists}, factor={impact.get('calibration_factor')}, "
     f"futures={impact.get('futures_impact_pct')}%, "
     f"trade_flow={impact.get('trade_flow_impact_pct')}%, "
     f"event_type={impact.get('event_type')}")

# ── Gate 3: market opportunity ranker ────────────────────────────────────────
print("\n[3] Market opportunity ranker")
from market_opportunity import rank_market_opportunities

top = rank_market_opportunities(top_n=15)
scores_finite = all(math.isfinite(s) for s in top["opportunity_score"])
gate("ranker_output",
     len(top) >= 10 and scores_finite,
     f"{len(top)} markets ranked, top: "
     + ", ".join(f"{r.country}({r.opportunity_score})" for r in top.head(3).itertuples()))

# ── Gate 4: event → market mapping ───────────────────────────────────────────
print("\n[4] Event-to-market mapping")
from market_opportunity import markets_affected_by_event

mapped = markets_affected_by_event(impact, top_n=15)
has_table = len(mapped["opportunity_markets"]) >= 10
gate("event_market_mapping",
     has_table,
     f"resolved={mapped['affected_named']}, diversion={mapped['trade_diversion']}, "
     f"flags={[m['country'] for m in mapped['opportunity_markets'] if m['event_flag']]}")

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for r in results.values() if r["pass"])
total  = len(results)
print("\n" + "=" * 72)
print(f"GATE RESULT: {passed}/{total} PASS")
print(f"GATE STATUS: {'PASS' if passed == total else 'FAIL'}")
print("=" * 72)

out = ROOT / "eval" / "news_futures_chain_results.json"
out.write_text(json.dumps({"passed": passed, "total": total,
                           "gate_pass": passed == total, "gates": results}, indent=2))
print(f"\nResults saved to {out}")
sys.exit(0 if passed == total else 1)
