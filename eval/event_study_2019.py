"""
eval/event_study_2019.py — Extended event study: 2019–2025, 22 events.

Extends eval/event_study_calibration.py back to 2019:
  - 12 additional documented steel trade events (2019–2023)
  - scored in ONE batched Groq call (risk_score, event_type, persistence)
    to conserve API quota; predicted impact computed locally with the same
    formula analyze_news_impact() uses (uncalibrated — we are re-deriving
    the calibration, so the raw model prediction is what gets compared)
  - the original 10 events (2024–2025) reuse their stored scores from
    eval/event_study_results.json — no recompute
  - actual returns measured on the full 2018-09 → today HRC=F series
    (fetched direct from yfinance; the shared futures cache is not touched)
  - drift baseline recomputed over the full sample and applied to ALL events

Outputs:
  eval/event_study_results_2019.json
  steel_rag/futures_cache/impact_calibration.json   (overwritten with the
                                                     re-derived factors)

Run:  python eval/event_study_2019.py
"""
import os, sys, json, time, statistics
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "steel_rag"))
os.chdir(ROOT / "steel_rag")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

PRE_DAYS, POST_DAYS = 1, 5

# ── 12 additional documented events, 2019–2023 ────────────────────────────────
EVENTS_2019 = [
    {"id": "US_CN_25PCT_2019", "date": "2019-05-10",
     "name": "US raises tariffs on $200B Chinese goods to 25%",
     "text": ("The United States raised tariffs on $200 billion of Chinese goods from "
              "10% to 25%, escalating the trade war with direct effects on steel-"
              "intensive manufactured goods and global steel demand.")},
    {"id": "US_CN_300B_2019", "date": "2019-08-01",
     "name": "US announces 10% tariff on remaining $300B Chinese imports",
     "text": ("The US President announced a 10% tariff on the remaining $300 billion "
              "of Chinese imports effective September 1, deepening the trade war and "
              "pressuring global manufacturing and steel demand.")},
    {"id": "PHASE_ONE_2020", "date": "2020-01-15",
     "name": "US-China Phase One trade deal signed",
     "text": ("The United States and China signed the Phase One trade agreement, "
              "pausing tariff escalation and committing China to increased purchases "
              "of US goods, easing trade tensions affecting steel markets.")},
    {"id": "COVID_CRASH_2020", "date": "2020-03-11",
     "name": "WHO declares COVID-19 pandemic, steel demand collapses",
     "text": ("The WHO declared COVID-19 a global pandemic. Automotive and "
              "construction shutdowns worldwide collapsed steel demand, with mills "
              "idling blast furnaces across Europe, North America and India.")},
    {"id": "CN_REBATE_1_2021", "date": "2021-04-28",
     "name": "China removes VAT export rebate on 146 steel products",
     "text": ("China announced removal of the 13% VAT export rebate on 146 steel "
              "products effective May 1, 2021, discouraging steel exports to keep "
              "supply at home and tightening global steel availability.")},
    {"id": "CN_REBATE_2_2021", "date": "2021-07-29",
     "name": "China cancels more export rebates, raises export duties",
     "text": ("China cancelled export rebates on a further 23 steel products and "
              "raised export duties on ferrochrome and high-purity pig iron effective "
              "August 1, 2021, further restricting steel exports.")},
    {"id": "US_EU_TRQ_2021", "date": "2021-10-30",
     "name": "US-EU deal replaces Section 232 tariffs with quotas",
     "text": ("The United States and European Union agreed to replace the 25% "
              "Section 232 steel tariff on EU imports with a tariff-rate quota "
              "system, de-escalating the transatlantic steel trade dispute.")},
    {"id": "RU_UA_WAR_2022", "date": "2022-02-24",
     "name": "Russia invades Ukraine, steel supply shock",
     "text": ("Russia launched a full-scale invasion of Ukraine, halting steel "
              "production at major Ukrainian mills including Azovstal and disrupting "
              "Black Sea exports of steel, iron ore and pig iron to Europe.")},
    {"id": "EU_RU_SANCTIONS_2022", "date": "2022-03-15",
     "name": "EU bans Russian steel imports",
     "text": ("The European Union adopted a fourth sanctions package banning imports "
              "of Russian steel products, cutting roughly 3.3 million tonnes of "
              "annual supply to the EU market.")},
    {"id": "IN_EXPORT_DUTY_2022", "date": "2022-05-21",
     "name": "India imposes 15% export duty on steel",
     "text": ("India imposed a 15% export duty on eight steel products including "
              "hot-rolled and cold-rolled coil to cool domestic prices, sharply "
              "curtailing Indian steel exports.")},
    {"id": "IN_DUTY_REMOVED_2022", "date": "2022-11-19",
     "name": "India removes steel export duty",
     "text": ("India removed the 15% export duty on steel products imposed in May, "
              "restoring export competitiveness for Indian mills amid weakening "
              "domestic demand.")},
    {"id": "EU_CBAM_2023", "date": "2023-10-01",
     "name": "EU CBAM transitional reporting begins",
     "text": ("The EU Carbon Border Adjustment Mechanism entered its transitional "
              "phase: importers of steel into the EU must now report embedded carbon "
              "emissions, the first step toward carbon levies on steel imports.")},
]

IMPACT_MULTIPLIERS = None   # loaded from steel_futures
PERSIST_SCALE = {"structural": 1.5, "cyclical": 1.0, "one_off": 0.7}

BATCH_PROMPT = """You are a steel trade risk scorer. For EACH event below, return:
- risk_score: 0.0-1.0 steel trade risk intensity (continuous; 0.8+ = critical structural event)
- event_type: one of [TARIFF_INCREASE, TARIFF_DECREASE, ANTIDUMPING_LEVY, SAFEGUARD_DUTY,
  SUPPLY_DISRUPTION, DEMAND_SURGE, DEMAND_SLOWDOWN, CAPACITY_EXPANSION, TRADE_AGREEMENT,
  SANCTIONS, GENERAL_POLICY]
- persistence: structural | cyclical | one_off

Return ONLY a JSON array, one object per event, in the same order:
[{"id": "...", "risk_score": 0.0, "event_type": "...", "persistence": "..."}]
"""


def score_events_batched(events: list[dict]) -> list[dict]:
    """One Groq call scoring all events; retries on rate limit."""
    from groq import Groq, RateLimitError
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    lines = [f'{i+1}. id={e["id"]} ({e["date"]}): {e["text"]}'
             for i, e in enumerate(events)]
    user = "\n\n".join(lines)

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": BATCH_PROMPT},
                          {"role": "user", "content": user}],
                temperature=0, max_tokens=900,
            )
            raw = resp.choices[0].message.content.strip()
            start, end = raw.find("["), raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except RateLimitError as e:
            wait = 90 * (attempt + 1)
            print(f"  [rate limit] attempt {attempt+1}/4 — waiting {wait}s …")
            time.sleep(wait)
    raise RuntimeError("Groq rate limit persisted after 4 attempts")


def predicted_futures_pct(risk: float, etype: str, persistence: str) -> float:
    """Replicates analyze_news_impact() raw (uncalibrated) futures impact."""
    mult = IMPACT_MULTIPLIERS.get(etype, IMPACT_MULTIPLIERS["GENERAL_POLICY"])
    return mult["futures"] * (risk * 2.5) * PERSIST_SCALE.get(persistence, 1.0)


def get_prices():
    import yfinance as yf
    out = {}
    for tk in ("HRC=F", "SLX"):
        df = yf.Ticker(tk).history(start="2018-09-01", auto_adjust=True)
        s = df["Close"].dropna()
        s.index = s.index.tz_localize(None) if s.index.tz else s.index
        out[tk] = s
    return out


def window_return(close, event_date, pre=PRE_DAYS, post=POST_DAYS):
    import pandas as pd
    ts = pd.Timestamp(event_date)
    before = close[close.index < ts]
    after  = close[close.index >= ts]
    if before.empty or len(after) < post:
        return None
    return (float(after.iloc[post - 1]) - float(before.iloc[-1])) / float(before.iloc[-1]) * 100


def main():
    global IMPACT_MULTIPLIERS
    from steel_futures import IMPACT_MULTIPLIERS as IM
    IMPACT_MULTIPLIERS = IM

    print("=" * 72)
    print("EXTENDED EVENT STUDY — 2019-2025, 22 events vs HRC=F abnormal returns")
    print("=" * 72)

    prices = get_prices()
    hrc, slx = prices["HRC=F"], prices["SLX"]
    drift_hrc = float((hrc.pct_change(PRE_DAYS + POST_DAYS).dropna() * 100).median())
    drift_slx = float((slx.pct_change(PRE_DAYS + POST_DAYS).dropna() * 100).median())
    print(f"\nHRC=F: {hrc.index.min().date()} -> {hrc.index.max().date()} "
          f"({len(hrc)} sessions) | drift {drift_hrc:+.2f}%")

    # ── Score the 12 new events (one batched call) ────────────────────────────
    print(f"\nScoring {len(EVENTS_2019)} historical events (single batched call) …")
    scored = score_events_batched(EVENTS_2019)
    score_map = {s["id"]: s for s in scored}

    rows = []
    for ev in EVENTS_2019:
        s = score_map.get(ev["id"])
        if not s:
            print(f"  [WARN] {ev['id']} missing from batch response — skipped")
            continue
        risk  = max(0.0, min(1.0, float(s["risk_score"])))
        etype = s["event_type"]
        pers  = s["persistence"]
        pred  = predicted_futures_pct(risk, etype, pers)

        raw_hrc = window_return(hrc, ev["date"])
        raw_slx = window_return(slx, ev["date"])
        if raw_hrc is None:
            print(f"  [SKIP] {ev['id']} — outside price window")
            continue
        abn_hrc = raw_hrc - drift_hrc
        abn_slx = (raw_slx - drift_slx) if raw_slx is not None else None
        ratio   = abn_hrc / pred if pred else None
        sign_ok = (pred > 0) == (abn_hrc > 0) if pred else None

        print(f"  [{ev['id']}] {ev['date']}  {etype}/{pers}  risk={risk:.2f}  "
              f"pred {pred:+.2f}%  actual {abn_hrc:+.2f}%  "
              f"{'AGREE' if sign_ok else 'DISAGREE' if sign_ok is not None else 'n/a'}")

        rows.append({
            "id": ev["id"], "date": ev["date"], "name": ev["name"],
            "model_type": etype, "persistence": pers, "risk_score": risk,
            "predicted_pct": round(pred, 2),
            "actual_raw_hrc_pct": round(raw_hrc, 2),
            "actual_abnormal_hrc_pct": round(abn_hrc, 2),
            "actual_abnormal_slx_pct": round(abn_slx, 2) if abn_slx is not None else None,
            "ratio_actual_over_pred": round(ratio, 3) if ratio is not None else None,
            "sign_agreement": sign_ok,
            "era": "2019-2023",
        })

    # ── Reuse the original 10 events (recompute abnormal with the new drift) ──
    prior = json.loads((ROOT / "eval" / "event_study_results.json").read_text())
    print(f"\nMerging {len(prior['events'])} prior events (2024-2025, stored scores) …")
    for r in prior["events"]:
        abn = r["actual_raw_hrc_pct"] - drift_hrc
        pred = r["predicted_pct"]
        ratio = abn / pred if pred else None
        sign_ok = (pred > 0) == (abn > 0) if pred else None
        rows.append({**{k: r[k] for k in
                        ("id", "date", "name", "model_type", "risk_score",
                         "predicted_pct", "actual_raw_hrc_pct")},
                     "persistence": r.get("persistence"),
                     "actual_abnormal_hrc_pct": round(abn, 2),
                     "actual_abnormal_slx_pct": r.get("actual_abnormal_slx_pct"),
                     "ratio_actual_over_pred": round(ratio, 3) if ratio is not None else None,
                     "sign_agreement": sign_ok,
                     "era": "2024-2025"})

    rows.sort(key=lambda r: r["date"])

    # ── Summary ───────────────────────────────────────────────────────────────
    preds = [r["predicted_pct"] for r in rows]
    acts  = [r["actual_abnormal_hrc_pct"] for r in rows]
    n = len(rows)
    corr = None
    if n >= 3 and statistics.pstdev(preds) > 0 and statistics.pstdev(acts) > 0:
        mp, ma = statistics.mean(preds), statistics.mean(acts)
        cov = sum((p - mp) * (a - ma) for p, a in zip(preds, acts)) / n
        corr = cov / (statistics.pstdev(preds) * statistics.pstdev(acts))

    signs = [r["sign_agreement"] for r in rows if r["sign_agreement"] is not None]
    sign_rate = sum(signs) / len(signs) if signs else None
    ratios = [r["ratio_actual_over_pred"] for r in rows
              if r["ratio_actual_over_pred"] is not None and r["sign_agreement"]]
    k = round(statistics.median(ratios), 3) if ratios else 1.0

    by_type = {}
    for r in rows:
        if r["ratio_actual_over_pred"] is not None:
            by_type.setdefault(r["model_type"], []).append(r["ratio_actual_over_pred"])
    type_ratios = {t: {"ratio": round(statistics.median(v), 3), "n": len(v)}
                   for t, v in by_type.items()}

    print("\n" + "=" * 72)
    print("EXTENDED CALIBRATION SUMMARY (2019-2025)")
    print("=" * 72)
    print(f"  Events analysed       : {n}")
    print(f"  Sign agreement        : {sum(signs)}/{len(signs)} = {sign_rate:.0%}")
    print(f"  Pearson corr          : {corr:.3f}" if corr is not None else "  corr n/a")
    print(f"  Global factor k       : {k}")
    print(f"  Per-type (ratio, n)   :")
    for t, v in sorted(type_ratios.items()):
        print(f"    {t:<20} {v['ratio']:>7.3f}  (n={v['n']})")

    results = {
        "methodology": {
            "window": f"close[t-{PRE_DAYS}] to close[t+{POST_DAYS}]",
            "drift": round(drift_hrc, 3),
            "price_series": "HRC=F 2018-09 onward (yfinance direct)",
            "n_events": n,
            "note": ("2019-2023 events scored in one batched Groq call; "
                     "2024-2025 events reuse stored scores; predicted impact is "
                     "the raw uncalibrated model output in all cases"),
        },
        "events": rows,
        "summary": {
            "sign_agreement_rate": round(sign_rate, 3),
            "pearson_correlation": round(corr, 3) if corr is not None else None,
            "global_calibration_factor_k": k,
            "per_type_ratios": {t: v["ratio"] for t, v in type_ratios.items()},
            "per_type_n": {t: v["n"] for t, v in type_ratios.items()},
        },
    }
    out = ROOT / "eval" / "event_study_results_2019.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")

    calib = {
        "calibration_factor": k,
        "source": "eval/event_study_2019.py (22 events, 2019-2025)",
        "n_events": n,
        "sign_agreement_rate": round(sign_rate, 3),
        "per_type_ratios": {t: v["ratio"] for t, v in type_ratios.items()},
        "per_type_n": {t: v["n"] for t, v in type_ratios.items()},
    }
    calib_path = ROOT / "steel_rag" / "futures_cache" / "impact_calibration.json"
    calib_path.write_text(json.dumps(calib, indent=2))
    print(f"Runtime calibration updated -> {calib_path}")


if __name__ == "__main__":
    main()
