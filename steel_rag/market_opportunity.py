"""
market_opportunity.py — Ranked new-market identification for Indian steel exports.

Combines three independent signals into one opportunity score per destination:

  1. Gravity gap   : XGBoost gravity model predicted vs actual exports (latest FY).
                     Predicted >> actual ⇒ India under-serves a market its size,
                     distance and trade ties say it should reach.  (signal = −residual)
  2. Momentum      : last-6-month export growth vs the prior 6 months (TRADESTAT).
  3. Market size   : gravity-predicted potential (ln USD) — bigger pies score higher.
  4. FTA tailwind  : RTA/FTA flag from the gravity dataset (CEPA, SAFTA, ASEAN…).

  opportunity = 0.40·z(gravity_gap) + 0.30·z(momentum) + 0.20·z(ln_size) + 0.10·rta

Also maps a news-impact result (steel_futures.analyze_news_impact output) onto the
ranking — flagging which ranked markets a tariff/AD event helps or hurts.

Usage:
    from market_opportunity import rank_market_opportunities, markets_affected_by_event
    top = rank_market_opportunities(top_n=15)

Or run directly:  python steel_rag/market_opportunity.py
"""

import sys
import json
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# Score weights
W_GAP, W_MOMENTUM, W_SIZE, W_RTA = 0.40, 0.30, 0.20, 0.10
MOMENTUM_WINDOW_M = 6     # last 6 months vs prior 6
MIN_BASE_USD_M    = 1.0   # ignore markets below $1M/yr — too small to rank

# Common-name aliases → TRADESTAT country names (used for event mapping)
COUNTRY_ALIASES = {
    "uae": "U ARAB EMTS", "united arab emirates": "U ARAB EMTS",
    "usa": "U S A", "united states": "U S A", "us": "U S A", "america": "U S A",
    "china": "CHINA P RP", "vietnam": "VIETNAM SOC REP", "south korea": "KOREA RP",
    "korea": "KOREA RP", "uk": "U K", "united kingdom": "U K",
    "saudi arabia": "SAUDI ARAB", "bangladesh": "BANGLADESH PR",
    "sri lanka": "SRI LANKA DSR", "european union": None, "eu": None,
}


def _z(series: pd.Series) -> pd.Series:
    """Z-score, safe for zero variance."""
    sd = series.std()
    if sd == 0 or np.isnan(sd):
        return series * 0.0
    return (series - series.mean()) / sd


def _momentum_table() -> pd.DataFrame:
    """Per-country export momentum: last 6 calendar months vs prior 6."""
    from data_agent import load_export_data
    df = load_export_data()
    df = df.dropna(subset=["monthly_curr_usd"]).copy()
    df["ym"] = df["report_year"] * 100 + df["report_month_num"]

    months = sorted(df["ym"].unique())
    if len(months) < 2 * MOMENTUM_WINDOW_M:
        recent_m, prior_m = months[len(months)//2:], months[:len(months)//2]
    else:
        recent_m = months[-MOMENTUM_WINDOW_M:]
        prior_m  = months[-2*MOMENTUM_WINDOW_M:-MOMENTUM_WINDOW_M]

    recent = (df[df["ym"].isin(recent_m)]
              .groupby("country")["monthly_curr_usd"].sum().rename("recent_6m_usd"))
    prior  = (df[df["ym"].isin(prior_m)]
              .groupby("country")["monthly_curr_usd"].sum().rename("prior_6m_usd"))

    mom = pd.concat([recent, prior], axis=1).fillna(0.0)
    mom["momentum_pct"] = np.where(
        mom["prior_6m_usd"] > 0,
        (mom["recent_6m_usd"] - mom["prior_6m_usd"]) / mom["prior_6m_usd"] * 100,
        0.0,
    )
    # Winsorise: tiny bases produce 1000%+ swings that would dominate the z-score
    mom["momentum_pct"] = mom["momentum_pct"].clip(-100, 200)
    return mom.reset_index()


def rank_market_opportunities(top_n: int = 15, min_base_usd_m: float = MIN_BASE_USD_M) -> pd.DataFrame:
    """
    Build the ranked market opportunity table.

    Returns DataFrame (descending opportunity_score):
      country, actual_usd_m, predicted_usd_m, gravity_gap_pct, momentum_pct,
      rta, opportunity_score, signal_breakdown
    """
    from gravity_model import ensure_model_ready, get_actual_vs_predicted

    mdl = ensure_model_ready()
    avp = get_actual_vs_predicted()

    # Latest FY per country
    latest_fy = avp["fy"].max()
    g = avp[avp["fy"] == latest_fy].copy()

    # RTA flag from gravity dataset
    rta_map = (mdl["df"][mdl["df"]["fy"] == latest_fy]
               .set_index("country")["rta"].to_dict())
    g["rta"] = g["country"].map(rta_map).fillna(0).astype(int)

    # Gravity gap: model says India should export more than it does ⇒ opportunity.
    # residual = actual_ln − predicted_ln, so gap = −residual (in log points → ≈ %)
    g["gravity_gap_ln"]  = -g["residual"]
    g["gravity_gap_pct"] = (np.exp(g["gravity_gap_ln"]) - 1) * 100

    # Momentum
    mom = _momentum_table()
    g = g.merge(mom[["country", "momentum_pct", "recent_6m_usd"]], on="country", how="left")
    g["momentum_pct"] = g["momentum_pct"].fillna(0.0)

    # Floor: skip negligible markets
    g = g[g["predicted_usd"] >= min_base_usd_m].copy()

    # Composite score
    g["z_gap"]  = _z(g["gravity_gap_ln"])
    g["z_mom"]  = _z(g["momentum_pct"])
    g["z_size"] = _z(np.log(g["predicted_usd"]))
    g["opportunity_score"] = (
        W_GAP * g["z_gap"] + W_MOMENTUM * g["z_mom"]
        + W_SIZE * g["z_size"] + W_RTA * g["rta"]
    ).round(3)

    g = g.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    g["rank"] = g.index + 1

    out = g[["rank", "country", "actual_usd", "predicted_usd",
             "gravity_gap_pct", "momentum_pct", "rta", "opportunity_score"]].copy()
    out = out.rename(columns={"actual_usd": "actual_usd_m", "predicted_usd": "predicted_usd_m"})
    for col in ("actual_usd_m", "predicted_usd_m", "gravity_gap_pct", "momentum_pct"):
        out[col] = out[col].round(2)

    return out.head(top_n) if top_n else out


def _resolve_country(name: str, known: set[str]) -> str | None:
    """Map a free-text country name to a TRADESTAT country name."""
    n = name.strip().lower()
    if n in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[n]
    for k in known:
        if n == k.lower() or n in k.lower():
            return k
    return None


def markets_affected_by_event(impact: dict, top_n: int = 15) -> dict:
    """
    Map a news-impact analysis (steel_futures.analyze_news_impact output) onto
    the market opportunity ranking.

    Returns:
      {
        "event_type", "trade_flow_impact_pct",
        "opportunity_markets": ranked table (list of dicts) with "event_flag"
                               column: "boosted" | "at_risk" | "",
        "affected_named": [resolved TRADESTAT names from the event analysis],
      }
    """
    ranked = rank_market_opportunities(top_n=top_n)
    known = set(ranked["country"])

    # Names the 3-layer analysis surfaced
    spill = impact.get("india_spillover", {}) or {}
    named = (impact.get("respondent_countries", [])
             + impact.get("initiator_countries", [])
             + spill.get("india_export_markets_affected", []))
    resolved = {r for n in named if (r := _resolve_country(n, known))}

    diversion = spill.get("trade_diversion_direction", "none")
    flow_pct  = impact.get("trade_flow_impact_pct", 0.0)

    def _flag(country: str) -> str:
        if country not in resolved:
            return ""
        if diversion == "positive" or flow_pct > 0:
            return "boosted"
        if diversion == "negative" or flow_pct < 0:
            return "at_risk"
        return "affected"

    ranked = ranked.copy()
    ranked["event_flag"] = ranked["country"].map(_flag)

    return {
        "event_type":            impact.get("event_type"),
        "trade_flow_impact_pct": flow_pct,
        "trade_diversion":       diversion,
        "affected_named":        sorted(resolved),
        "opportunity_markets":   ranked.to_dict(orient="records"),
    }


# ── CLI smoke test ────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("MARKET OPPORTUNITY RANKER — under-served Indian steel export markets")
    print(f"score = {W_GAP}·z(gravity gap) + {W_MOMENTUM}·z(6m momentum) "
          f"+ {W_SIZE}·z(ln size) + {W_RTA}·FTA")
    print("=" * 72)

    top = rank_market_opportunities(top_n=15)
    print(f"\n{'#':>3} {'Country':<20} {'Actual $M':>10} {'Gravity $M':>11} "
          f"{'Gap %':>8} {'Mom %':>7} {'FTA':>4} {'Score':>7}")
    for _, r in top.iterrows():
        print(f"{r['rank']:>3} {r['country']:<20} {r['actual_usd_m']:>10.1f} "
              f"{r['predicted_usd_m']:>11.1f} {r['gravity_gap_pct']:>8.1f} "
              f"{r['momentum_pct']:>7.1f} {r['rta']:>4} {r['opportunity_score']:>7.3f}")

    out_path = Path(__file__).parent.parent / "eval" / "market_opportunities.json"
    full = rank_market_opportunities(top_n=0)
    out_path.write_text(json.dumps(full.to_dict(orient="records"), indent=2))
    print(f"\nFull ranking ({len(full)} markets) saved to {out_path}")


if __name__ == "__main__":
    main()
