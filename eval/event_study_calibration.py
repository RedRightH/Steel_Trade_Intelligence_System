"""
eval/event_study_calibration.py — Event study to calibrate news → futures impact multipliers.

Methodology
───────────
For each documented steel trade event (2024–2025, within the cached 2-year HRC=F window):
  1. PREDICTED : run analyze_news_impact() on the announcement text
                 → futures_impact_pct from IMPACT_MULTIPLIERS
  2. ACTUAL    : HRC=F abnormal return over the event window
                 raw  = (close[t+5] − close[t−1]) / close[t−1] × 100
                 abn  = raw − median 6-session return over the full sample (de-trend)
  3. CALIBRATE : per-event ratio actual/predicted, sign agreement,
                 Pearson correlation, global scaling factor k

Outputs
───────
  eval/event_study_results.json            — full per-event table + summary
  steel_rag/futures_cache/impact_calibration.json — global factor consumed by
                                             analyze_news_impact() at runtime

Run:  python eval/event_study_calibration.py        (default Python — needs groq)
"""
import os, sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "steel_rag"))
os.chdir(ROOT / "steel_rag")

# ── Documented steel trade events (announcement date, text) ──────────────────
# Dates are the public announcement dates; texts paraphrase the announcements.
# Sources: project corpus (DGTR/NCV documents, Ministry of Steel reports) and
# widely reported trade policy actions within the HRC=F price window.
EVENTS = [
    {
        "id": "S301_CN_2024",
        "date": "2024-09-13",
        "name": "US finalizes Section 301 tariff hike on Chinese steel",
        "text": ("The United States has finalized Section 301 tariff increases on imports "
                 "from China, raising duties on Chinese steel and aluminum products to 25%, "
                 "effective September 27, 2024."),
        "expected_type": "TARIFF_INCREASE",
    },
    {
        "id": "CN_STIMULUS_2024",
        "date": "2024-09-24",
        "name": "China announces broad monetary stimulus package",
        "text": ("China's central bank announced a sweeping stimulus package including rate "
                 "cuts and property market support, boosting expectations for steel demand "
                 "recovery in the world's largest steel consumer."),
        "expected_type": "DEMAND_SURGE",
    },
    {
        "id": "IN_SG_INIT_2024",
        "date": "2024-12-19",
        "name": "India initiates safeguard investigation on flat steel",
        "text": ("India's DGTR has initiated a safeguard investigation concerning imports of "
                 "non-alloy and alloy steel flat products into India, citing a surge in "
                 "imports causing serious injury to the domestic industry."),
        "expected_type": "SAFEGUARD_DUTY",
    },
    {
        "id": "S232_RESTORE_2025",
        "date": "2025-02-10",
        "name": "US restores 25% Section 232 steel tariffs, ends exemptions",
        "text": ("The US President signed a proclamation restoring the full 25% Section 232 "
                 "tariff on all steel imports and revoking all country exemptions and quota "
                 "arrangements, effective March 12, 2025."),
        "expected_type": "TARIFF_INCREASE",
    },
    {
        "id": "VN_AD_CN_2025",
        "date": "2025-02-21",
        "name": "Vietnam imposes provisional AD duty on Chinese HRC",
        "text": ("Vietnam announced provisional anti-dumping duties of up to 27.83% on "
                 "hot-rolled coil imports from China following its dumping investigation, "
                 "effective March 2025."),
        "expected_type": "ANTIDUMPING_LEVY",
    },
    {
        "id": "S232_EFFECT_2025",
        "date": "2025-03-12",
        "name": "Section 232 25% steel tariffs take effect globally",
        "text": ("The expanded US Section 232 tariffs of 25% on all steel imports took effect "
                 "today, with no country exemptions, covering all steel mill products and "
                 "derivative steel articles."),
        "expected_type": "TARIFF_INCREASE",
    },
    {
        "id": "EU_STEEL_PLAN_2025",
        "date": "2025-03-19",
        "name": "EU launches Steel and Metals Action Plan",
        "text": ("The European Commission presented its Steel and Metals Action Plan, "
                 "tightening steel import safeguard measures, cutting quota volumes, and "
                 "announcing a review of trade defence instruments to protect EU producers."),
        "expected_type": "SAFEGUARD_DUTY",
    },
    {
        "id": "IN_SG_DUTY_2025",
        "date": "2025-04-21",
        "name": "India imposes 12% provisional safeguard duty on flat steel",
        "text": ("India has imposed a 12% provisional safeguard duty on imports of non-alloy "
                 "and alloy steel flat products for 200 days, following the DGTR's "
                 "preliminary finding of serious injury from an import surge."),
        "expected_type": "SAFEGUARD_DUTY",
    },
    {
        "id": "US_CN_TRUCE_2025",
        "date": "2025-05-12",
        "name": "US-China Geneva trade truce de-escalates tariffs",
        "text": ("The United States and China agreed in Geneva to a 90-day truce "
                 "substantially lowering reciprocal tariffs, easing trade tensions and "
                 "reducing near-term pressure on global steel trade flows."),
        "expected_type": "TARIFF_DECREASE",
    },
    {
        "id": "S232_50PCT_2025",
        "date": "2025-06-03",
        "name": "US doubles Section 232 steel tariffs to 50%",
        "text": ("The United States has increased Section 232 duties on steel imports from "
                 "25% to 50% and revoked remaining country exemptions, a major escalation "
                 "of steel trade protection."),
        "expected_type": "TARIFF_INCREASE",
    },
]

PRE_DAYS, POST_DAYS = 1, 5   # window: close[t−1] → close[t+5]


def get_price_series():
    """HRC=F daily closes (2y cache) as a tz-naive Series, plus SLX for robustness."""
    from steel_futures import fetch_ticker
    out = {}
    for tk in ("HRC=F", "SLX"):
        df = fetch_ticker(tk, period="2y")
        s = df["Close"].dropna()
        s.index = s.index.tz_localize(None) if s.index.tz else s.index
        out[tk] = s
    return out


def window_return(close, event_date, pre=PRE_DAYS, post=POST_DAYS):
    """Return % move from last close before event to `post` sessions after, or None."""
    import pandas as pd
    ts = pd.Timestamp(event_date)
    before = close[close.index < ts]
    after  = close[close.index >= ts]
    if before.empty or len(after) < post:
        return None
    p0 = float(before.iloc[-1])
    p1 = float(after.iloc[post - 1])
    return (p1 - p0) / p0 * 100


def median_window_return(close, span=PRE_DAYS + POST_DAYS):
    """Median rolling `span`-session return across the sample (drift baseline)."""
    r = close.pct_change(span).dropna() * 100
    return float(r.median())


def main():
    from steel_futures import analyze_news_impact

    print("=" * 72)
    print("EVENT STUDY — News impact multiplier calibration vs actual HRC=F moves")
    print(f"Window: close[t-{PRE_DAYS}] -> close[t+{POST_DAYS}]  |  {len(EVENTS)} events")
    print("=" * 72)

    prices = get_price_series()
    hrc, slx = prices["HRC=F"], prices["SLX"]
    drift_hrc = median_window_return(hrc)
    drift_slx = median_window_return(slx)
    print(f"\nSample drift (median {PRE_DAYS+POST_DAYS}-session return): "
          f"HRC {drift_hrc:+.2f}%  SLX {drift_slx:+.2f}%")
    print(f"HRC data: {hrc.index.min().date()} -> {hrc.index.max().date()}  ({len(hrc)} sessions)")

    rows = []
    for ev in EVENTS:
        print(f"\n[{ev['id']}] {ev['name']}  ({ev['date']})")

        raw_hrc = window_return(hrc, ev["date"])
        raw_slx = window_return(slx, ev["date"])
        if raw_hrc is None:
            print("  SKIP — event date outside available price window")
            continue
        abn_hrc = raw_hrc - drift_hrc
        abn_slx = (raw_slx - drift_slx) if raw_slx is not None else None

        impact = analyze_news_impact(ev["text"])
        pred = impact.get("futures_impact_pct", 0.0)
        etype = impact.get("event_type", "?")
        risk  = impact.get("risk_score", 0.0)

        ratio = abn_hrc / pred if pred not in (0, 0.0) else None
        sign_ok = (pred > 0) == (abn_hrc > 0) if pred != 0 else None

        print(f"  Model   : type={etype} (expected {ev['expected_type']})  "
              f"risk={risk:.2f}  predicted {pred:+.2f}%")
        print(f"  Actual  : HRC raw {raw_hrc:+.2f}%  abnormal {abn_hrc:+.2f}%"
              + (f"  |  SLX abnormal {abn_slx:+.2f}%" if abn_slx is not None else ""))
        print(f"  Sign    : {'AGREE' if sign_ok else 'DISAGREE' if sign_ok is not None else 'n/a'}"
              + (f"  |  ratio actual/pred = {ratio:.2f}" if ratio is not None else ""))

        rows.append({
            "id": ev["id"], "date": ev["date"], "name": ev["name"],
            "expected_type": ev["expected_type"], "model_type": etype,
            "risk_score": risk,
            "predicted_pct": round(pred, 2),
            "actual_raw_hrc_pct": round(raw_hrc, 2),
            "actual_abnormal_hrc_pct": round(abn_hrc, 2),
            "actual_abnormal_slx_pct": round(abn_slx, 2) if abn_slx is not None else None,
            "ratio_actual_over_pred": round(ratio, 3) if ratio is not None else None,
            "sign_agreement": sign_ok,
        })

    # ── Summary statistics ────────────────────────────────────────────────────
    import statistics
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
    # Global scaling factor from sign-agreeing events only (median is robust to outliers)
    k = round(statistics.median(ratios), 3) if ratios else 1.0

    # Per-event-type calibration
    by_type = {}
    for r in rows:
        if r["ratio_actual_over_pred"] is not None:
            by_type.setdefault(r["model_type"], []).append(r["ratio_actual_over_pred"])
    type_ratios = {t: round(statistics.median(v), 3) for t, v in by_type.items()}

    print("\n" + "=" * 72)
    print("CALIBRATION SUMMARY")
    print("=" * 72)
    print(f"  Events analysed       : {n}")
    print(f"  Sign agreement        : {sum(signs)}/{len(signs)}"
          + (f" = {sign_rate:.0%}" if sign_rate is not None else ""))
    print(f"  Pearson corr (p, a)   : {corr:.3f}" if corr is not None else "  Correlation: n/a")
    print(f"  Median |ratio| (agree): k = {k}")
    print(f"  Per-type ratios       : {type_ratios}")
    print(f"\n  Interpretation: multiply IMPACT_MULTIPLIERS futures coefficients by k={k}")
    print(f"  to align model predictions with observed {PRE_DAYS+POST_DAYS}-session abnormal returns.")
    print(f"  Caveat: n={n} events, single price series — directional validation, not")
    print(f"  statistical proof. Expand the event list as the news pipeline accumulates.")

    results = {
        "methodology": {
            "window": f"close[t-{PRE_DAYS}] to close[t+{POST_DAYS}]",
            "abnormal_return": "raw window return minus sample median window return (drift adjustment)",
            "price_series": "HRC=F (primary), SLX (robustness)",
            "n_events": n,
        },
        "events": rows,
        "summary": {
            "sign_agreement_rate": round(sign_rate, 3) if sign_rate is not None else None,
            "pearson_correlation": round(corr, 3) if corr is not None else None,
            "global_calibration_factor_k": k,
            "per_type_ratios": type_ratios,
        },
    }
    out = ROOT / "eval" / "event_study_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")

    # Write runtime calibration file consumed by analyze_news_impact()
    calib = {
        "calibration_factor": k,
        "source": "eval/event_study_calibration.py",
        "n_events": n,
        "sign_agreement_rate": round(sign_rate, 3) if sign_rate is not None else None,
        "per_type_ratios": type_ratios,
    }
    calib_path = ROOT / "steel_rag" / "futures_cache" / "impact_calibration.json"
    calib_path.write_text(json.dumps(calib, indent=2))
    print(f"Runtime calibration written to {calib_path}")


if __name__ == "__main__":
    main()
