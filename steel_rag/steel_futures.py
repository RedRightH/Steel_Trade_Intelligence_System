"""
steel_futures.py — Steel futures data, price forecasting, and news-impact analysis.

Capabilities
────────────
1. fetch_futures_data()        : Pull HRC futures + Indian/global steel proxies via yfinance
2. compute_technicals()        : MA20/50, RSI-14, Bollinger Bands
3. forecast_price()            : 30/60/90-day Prophet forecast
4. analyze_news_impact()       : LLM-powered announcement → quantified price + trade impact
5. get_futures_snapshot()      : One-call summary dict for the dashboard

Tickers tracked
────────────────
  HRC=F        US HRC Steel Futures (CME) — USD / short ton
  SLX          VanEck Steel ETF
  TATASTEEL.NS Tata Steel Ltd (NSE, INR)
  SAIL.NS      Steel Authority of India (NSE, INR)
  JSWSTEEL.NS  JSW Steel Ltd (NSE, INR)
  MT           ArcelorMittal (NYSE, USD)
  NUE          Nucor Corporation (NYSE, USD)

Usage
─────
  python steel_rag/steel_futures.py
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR    = Path(__file__).parent / "futures_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_H  = 4          # re-fetch after 4 hours
FORECAST_DAYS = 60        # default Prophet horizon

TICKERS = {
    "HRC=F":        {"name": "HRC Steel Futures",     "currency": "USD", "unit": "/short ton"},
    "SLX":          {"name": "VanEck Steel ETF",       "currency": "USD", "unit": ""},
    "TATASTEEL.NS": {"name": "Tata Steel (NSE)",       "currency": "INR", "unit": ""},
    "SAIL.NS":      {"name": "SAIL (NSE)",             "currency": "INR", "unit": ""},
    "JSWSTEEL.NS":  {"name": "JSW Steel (NSE)",        "currency": "INR", "unit": ""},
    "MT":           {"name": "ArcelorMittal (NYSE)",   "currency": "USD", "unit": ""},
    "NUE":          {"name": "Nucor Corp (NYSE)",      "currency": "USD", "unit": ""},
}

# Impact multipliers by event type (% price change per event unit)
# Calibrated from academic literature on commodity price responses
IMPACT_MULTIPLIERS = {
    "TARIFF_INCREASE":    {"futures": -0.8,  "trade_flow": -1.5,  "direction": -1},
    "TARIFF_DECREASE":    {"futures": +0.6,  "trade_flow": +1.2,  "direction": +1},
    "ANTIDUMPING_LEVY":   {"futures": +1.2,  "trade_flow": -2.0,  "direction": -1},
    "SAFEGUARD_DUTY":     {"futures": +1.5,  "trade_flow": -2.5,  "direction": -1},
    "SUPPLY_DISRUPTION":  {"futures": +2.0,  "trade_flow": -1.0,  "direction": -1},
    "DEMAND_SURGE":       {"futures": +1.8,  "trade_flow": +2.0,  "direction": +1},
    "DEMAND_SLOWDOWN":    {"futures": -1.5,  "trade_flow": -1.5,  "direction": -1},
    "CAPACITY_EXPANSION": {"futures": -0.5,  "trade_flow": -0.8,  "direction": -1},
    "TRADE_AGREEMENT":    {"futures": +0.3,  "trade_flow": +3.0,  "direction": +1},
    "SANCTIONS":          {"futures": +1.0,  "trade_flow": -3.5,  "direction": -1},
    "GENERAL_POLICY":     {"futures": +0.2,  "trade_flow": +0.5,  "direction": +1},
}

_price_cache: dict = {}   # in-memory cache

# Event-study calibration (written by eval/event_study_calibration.py).
# Scales the literature-based futures multipliers to observed HRC=F abnormal returns.
CALIBRATION_PATH = CACHE_DIR / "impact_calibration.json"


def _load_calibration() -> dict:
    if CALIBRATION_PATH.exists():
        try:
            return json.loads(CALIBRATION_PATH.read_text())
        except Exception:
            pass
    return {}


# ── Data fetching ─────────────────────────────────────────────────────────────

def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('=','_').replace('.','_')}.json"


def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return age_h > CACHE_TTL_H


def fetch_ticker(ticker: str, period: str = "2y") -> "pd.DataFrame":
    """Fetch OHLCV for one ticker; returns DataFrame with DatetimeIndex."""
    import pandas as pd
    import yfinance as yf

    cache_p = _cache_path(ticker)
    if not _is_stale(cache_p):
        raw = json.loads(cache_p.read_text())
        df = pd.DataFrame(raw)
        df.index = pd.to_datetime(df.index)
        return df

    print(f"  [yfinance] Fetching {ticker} ({period}) …")
    t = yf.Ticker(ticker)
    df = t.history(period=period, auto_adjust=True)
    if df.empty:
        return df

    # Persist
    cache_p.write_text(df.to_json(date_format="iso"))
    return df


def fetch_futures_data(period: str = "2y") -> dict:
    """
    Returns dict: ticker → DataFrame (OHLCV, DatetimeIndex).
    Skips tickers that return no data.
    """
    result = {}
    for ticker in TICKERS:
        try:
            df = fetch_ticker(ticker, period=period)
            if not df.empty:
                result[ticker] = df
        except Exception as e:
            print(f"  [WARN] {ticker}: {e}")
    return result


# ── Technical indicators ──────────────────────────────────────────────────────

def compute_technicals(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add MA20, MA50, RSI14, BB_upper, BB_lower to a price DataFrame."""
    import pandas as pd

    df = df.copy()
    close = df["Close"]

    # Moving averages
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("nan"))
    df["RSI14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-period, ±2σ)
    std20         = close.rolling(20).std()
    df["BB_upper"] = df["MA20"] + 2 * std20
    df["BB_lower"] = df["MA20"] - 2 * std20

    return df


# ── Prophet price forecast ────────────────────────────────────────────────────

def forecast_price(df: "pd.DataFrame", days: int = FORECAST_DAYS,
                   gpr_df: "pd.DataFrame | None" = None) -> dict:
    """
    Run Prophet on Close prices, optionally conditioned on the Steel-GPR index.

    gpr_df: optional DataFrame with columns [date, steel_gpr_index] (from
    load_steel_gpr_index()). When provided, the index is merged as an external
    regressor so news-driven geopolitical risk shifts the price forecast.
    Days with no scored articles are filled with the baseline value 100;
    future days hold the mean of the last 5 observed index values.

    Returns:
      {
        "forecast":  DataFrame with ds, yhat, yhat_lower, yhat_upper,
        "current":   float,
        "target_30": float,
        "target_60": float,
        "trend":     "bullish" | "bearish" | "neutral",
        "change_pct_30": float,
        "change_pct_60": float,
        "gpr_used":  bool,
        "gpr_coef":  float | None,   # regressor beta (price units per index point)
        "gpr_future_level": float | None,
      }
    """
    from prophet import Prophet
    import pandas as pd

    close = df["Close"].dropna()
    if len(close) < 30:
        return {}

    prophet_df = pd.DataFrame({
        "ds": close.index.tz_localize(None) if close.index.tz else close.index,
        "y":  close.values,
    })

    # ── Steel-GPR regressor (news risk → price forecast link) ────────────────
    gpr_used, gpr_future_level = False, None
    if gpr_df is not None and not gpr_df.empty and "steel_gpr_index" in gpr_df.columns:
        g = gpr_df[["date", "steel_gpr_index"]].copy()
        g["date"] = pd.to_datetime(g["date"]).dt.tz_localize(None)
        g = g.rename(columns={"date": "ds", "steel_gpr_index": "gpr"})
        prophet_df = prophet_df.merge(g, on="ds", how="left")
        # Baseline=100 on days with no scored articles (index is normalised to 100)
        prophet_df["gpr"] = prophet_df["gpr"].fillna(100.0)
        gpr_future_level = float(g["gpr"].tail(5).mean())
        gpr_used = True

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
        interval_width=0.80,
    )
    if gpr_used:
        model.add_regressor("gpr", mode="additive")

    # Suppress Stan output
    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=days, freq="B")  # business days
    if gpr_used:
        # Historical days: actual index (baseline-filled); future days: hold recent level
        hist_gpr = prophet_df.set_index("ds")["gpr"]
        future["gpr"] = future["ds"].map(hist_gpr).fillna(gpr_future_level)
    forecast = model.predict(future)

    current     = float(close.iloc[-1])
    fc_tail     = forecast[forecast["ds"] > prophet_df["ds"].max()]

    def _pick(n_days):
        subset = fc_tail.head(n_days)
        return float(subset["yhat"].iloc[-1]) if not subset.empty else current

    t30 = _pick(30)
    t60 = _pick(60)
    chg30 = (t30 - current) / current * 100
    chg60 = (t60 - current) / current * 100

    if chg30 > 2:    trend = "bullish"
    elif chg30 < -2: trend = "bearish"
    else:            trend = "neutral"

    # Extract the fitted GPR regressor coefficient (price units per index point)
    gpr_coef = None
    if gpr_used:
        try:
            from prophet.utilities import regressor_coefficients
            rc = regressor_coefficients(model)
            row = rc[rc["regressor"] == "gpr"]
            if not row.empty:
                gpr_coef = float(row["coef"].iloc[0])
        except Exception:
            pass

    return {
        "forecast":      forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        "current":       current,
        "target_30":     t30,
        "target_60":     t60,
        "trend":         trend,
        "change_pct_30": chg30,
        "change_pct_60": chg60,
        "gpr_used":      gpr_used,
        "gpr_coef":      gpr_coef,
        "gpr_future_level": gpr_future_level,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AI-GPR–Inspired 3-Layer News Impact Architecture
# Based on Iacoviello & Tong (2026) "The AI-GPR Index"
#
# Layer 1 — Continuous Steel Trade Risk Score (0.0–1.0)
#   Replaces binary keyword matching with semantic intensity scoring.
#   All articles scored; only those > 0.3 proceed to Layer 2.
#
# Layer 2 — Domain + Actor + Persistence Classification
#   Identifies event type, initiator/respondent/spillover countries, and whether
#   the shock is structural (persistent) vs one-off.  Mirrors the paper's
#   "second LLM classification layer" (oil supply disruption analogue → here,
#   steel trade domain decomposition).
#
# Layer 3 — India Bilateral Spillover Analysis
#   Directed bilateral analysis: who acts on whom, and what trade-diversion
#   opportunity or threat does that create for India.  Implements the paper's
#   bilateral GPR concept (Section 5.3): initiator → respondent → spillover.
#
# Quantification uses persistence-adjusted multipliers:
#   structural  → 1.5× base (paper: persistent component has ~2× effect on asset prices)
#   cyclical    → 1.0× base
#   one_off     → 0.7× base (immediate spike, fades quickly)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Layer 1 prompt: continuous 0-1 Steel Trade Risk Score ────────────────────
_L1_PROMPT = """
You will be given a news article or trade announcement. Score its steel trade risk intensity
on a continuous scale from 0.0 to 1.0 based only on what the text states or strongly implies.

Steel trade risk = events that materially affect the pricing, volume, or policy framework
of international steel trade: tariffs, antidumping/safeguard duties, supply disruptions,
demand shocks, trade agreements, sanctions, geopolitical events with direct steel implications.

Score scale:
  0.0-0.2  No material steel trade impact. General news, no direct trade relevance.
  0.2-0.4  Minor background relevance. Limited or indirect impact on steel trade flows.
  0.4-0.6  Significant development. Moderate impact on specific products or trade routes.
  0.6-0.8  Major policy action or supply/demand shock. Substantial market impact expected.
  0.8-1.0  Critical event. Structural disruption, landmark tariff action, or severe supply crisis.

Return JSON only (no markdown): {"risk_score": float, "steel_relevant": true|false}
Temperature is set to 0 for reproducibility.
""".strip()

# ── Layer 2 prompt: domain + actors + persistence ────────────────────────────
_L2_PROMPT = """
This steel trade announcement has a risk score of {score:.2f}/1.0. Classify it further.

Return JSON only (no markdown, no explanation):
{{
  "event_type": one of [TARIFF_INCREASE, TARIFF_DECREASE, ANTIDUMPING_LEVY, SAFEGUARD_DUTY,
                         SUPPLY_DISRUPTION, DEMAND_SURGE, DEMAND_SLOWDOWN, CAPACITY_EXPANSION,
                         TRADE_AGREEMENT, SANCTIONS, GEOPOLITICAL_SPILLOVER, OTHER],
  "initiator_countries": ["country"],   // country taking the action (e.g. imposing tariff)
  "respondent_countries": ["country"],  // country directly targeted
  "affected_products": ["HRC", "CRC", "rebar", "plates", "pipes", "billets"],
  "persistence": "structural"|"cyclical"|"one_off",
  "persistence_reason": "brief reason why this is structural/cyclical/one-off",
  "summary": "one sentence: what this means for India's steel trade"
}}

persistence definitions:
  structural = long-lasting regime change (new tariff law, permanent trade agreement, multi-year capacity investment)
  cyclical   = medium-term pattern (quarterly demand shift, seasonal supply factor, temporary duty review)
  one_off    = single discrete event (port closure, one announcement, short-term emergency measure)
""".strip()

# ── Layer 3 prompt: bilateral India spillover analysis ────────────────────────
_L3_PROMPT = """
Given this steel trade event (initiator: {initiator}, respondent: {respondent},
event: {event_type}, persistence: {persistence}):

Analyse India's position as a SPILLOVER country. In Iacoviello & Tong (2026)'s framework,
spillover countries are not directly involved but are significantly affected through trade
diversion, price spillovers, or competitive dynamics.

Return JSON only (no markdown):
{{
  "india_role": "beneficiary"|"victim"|"neutral",
  "trade_diversion_direction": "positive"|"negative"|"none",
  "trade_diversion_reason": "brief explanation of the trade diversion mechanism",
  "india_export_markets_affected": ["market1", "market2"],
  "india_import_competition_change": "increases"|"decreases"|"unchanged",
  "india_spillover_score": float 0.0-1.0,
  "india_spillover_summary": "one sentence on India's net position"
}}
""".strip()


def _llm_call(client, system: str, user: str, max_tokens: int = 300) -> dict:
    """Single LLM call with JSON extraction and error handling. Temperature=0 per AI-GPR paper."""
    import re
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user[:3000]}],  # 3000 chars per paper's Layer 2 window
        temperature=0,   # deterministic — per AI-GPR paper Section 2.2
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {}


def _get_current_hrc() -> float | None:
    """Return latest HRC=F close price (cached)."""
    try:
        df = fetch_ticker("HRC=F", period="5d")
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


def score_article(text: str, client=None) -> dict:
    """
    Layer 1: score a single article/announcement on the 0-1 Steel Trade Risk scale.
    Returns {"risk_score": float, "steel_relevant": bool}.
    Used by both the news impact analyzer and the Steel-GPR index builder.
    """
    if client is None:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    result = _llm_call(client, _L1_PROMPT, text[:2000], max_tokens=60)
    score = float(result.get("risk_score", 0.0))
    score = max(0.0, min(1.0, score))
    return {"risk_score": score, "steel_relevant": bool(result.get("steel_relevant", score > 0.2))}


def analyze_news_impact(announcement: str, rag_context: str = "") -> dict:
    """
    3-layer AI-GPR–inspired analysis pipeline.

    Layer 1 → continuous risk score (0-1)
    Layer 2 → domain, actors, persistence  [only if score > 0.3]
    Layer 3 → India bilateral spillover     [only if score > 0.5]

    Quantification uses persistence-adjusted multipliers:
      structural → 1.5×  (per AI-GPR finding: persistent GPR has ~2× price impact vs shock)
      cyclical   → 1.0×
      one_off    → 0.7×  (immediate spike, fades quickly)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY not set"}

    from groq import Groq
    client = Groq(api_key=api_key)

    user_text = announcement
    if rag_context:
        user_text = f"ANNOUNCEMENT:\n{announcement}\n\nHISTORICAL ANALOGUES (from document corpus):\n{rag_context}"

    # ── Layer 1: Steel Trade Risk Score ───────────────────────────────────────
    l1 = score_article(user_text[:2000], client=client)
    risk_score = l1["risk_score"]

    if risk_score < 0.2:
        return {
            "risk_score": risk_score, "steel_relevant": False,
            "summary": "No material steel trade impact detected.",
            "futures_impact_pct": 0.0, "trade_flow_impact_pct": 0.0,
            "layer": 1,
        }

    # ── Layer 2: Domain + Actors + Persistence ────────────────────────────────
    l2 = _llm_call(client, _L2_PROMPT.format(score=risk_score), user_text[:3000], max_tokens=350)

    event_type   = l2.get("event_type", "OTHER")
    initiators   = l2.get("initiator_countries", [])
    respondents  = l2.get("respondent_countries", [])
    persistence  = l2.get("persistence", "one_off")
    products     = l2.get("affected_products", [])
    summary      = l2.get("summary", "")

    # Persistence multiplier — inspired by AI-GPR finding that persistent GPR
    # has ~2× the effect on asset prices vs transient shocks
    persist_scale = {"structural": 1.5, "cyclical": 1.0, "one_off": 0.7}.get(persistence, 1.0)

    # Risk-score scale: maps 0-1 score to 0-2.5 multiplier (continuous, replaces 1/2/3 buckets)
    risk_scale = risk_score * 2.5

    mult = IMPACT_MULTIPLIERS.get(event_type, IMPACT_MULTIPLIERS["GENERAL_POLICY"])
    futures_impact_pct    = mult["futures"]    * risk_scale * persist_scale
    trade_flow_impact_pct = mult["trade_flow"] * risk_scale * persist_scale

    # Event-study calibration: scale futures impact to observed HRC=F abnormal
    # returns (per-type ratio when available, else global factor k)
    calib = _load_calibration()
    calib_factor = None
    if calib:
        calib_factor = (calib.get("per_type_ratios", {}).get(event_type)
                        or calib.get("calibration_factor"))
        if calib_factor:
            # Clip: per-type ratios come from few events each; cap leverage
            calib_factor = max(-2.0, min(2.5, float(calib_factor)))
            futures_impact_pct *= calib_factor

    # Tariff events: override trade flow with gravity-model elasticity (ε = -1.5)
    if event_type in ("TARIFF_INCREASE", "TARIFF_DECREASE", "ANTIDUMPING_LEVY", "SAFEGUARD_DUTY"):
        # Extract tariff % from text if mentioned; otherwise use risk_score as proxy
        import re as _re
        pct_matches = _re.findall(r"(\d+(?:\.\d+)?)\s*%", announcement)
        tariff_pct  = float(pct_matches[0]) if pct_matches else risk_score * 30
        direction   = mult["direction"]
        # Gravity: Δln(trade) ≈ ε × Δln(1+t) ≈ -1.5 × (Δt/100)
        trade_flow_impact_pct = -1.5 * tariff_pct * direction * persist_scale

    # ── Layer 3: India Bilateral Spillover (only for high-impact events) ──────
    l3 = {}
    india_spillover = {}
    if risk_score >= 0.5 and (initiators or respondents):
        l3 = _llm_call(
            client,
            _L3_PROMPT.format(
                initiator   = ", ".join(initiators) or "unknown",
                respondent  = ", ".join(respondents) or "unknown",
                event_type  = event_type,
                persistence = persistence,
            ),
            user_text[:3000],
            max_tokens=300,
        )
        india_spillover = {
            "india_role":                    l3.get("india_role", "neutral"),
            "trade_diversion_direction":     l3.get("trade_diversion_direction", "none"),
            "trade_diversion_reason":        l3.get("trade_diversion_reason", ""),
            "india_export_markets_affected": l3.get("india_export_markets_affected", []),
            "india_import_competition_change": l3.get("india_import_competition_change", "unchanged"),
            "india_spillover_score":         l3.get("india_spillover_score", 0.0),
            "india_spillover_summary":       l3.get("india_spillover_summary", ""),
        }
        # Adjust trade flow based on India's spillover role
        spillover_adj = {"beneficiary": +0.5, "victim": -0.5, "neutral": 0.0}
        trade_flow_impact_pct += spillover_adj.get(l3.get("india_role", "neutral"), 0.0) * risk_scale

    # ── Scenario table (bull / base / bear) ───────────────────────────────────
    current_hrc = _get_current_hrc()
    scenarios = []
    for label, adj in [("Base case", 1.0), ("Bull case", 1.5), ("Bear case", 0.5)]:
        hrc_new = current_hrc * (1 + futures_impact_pct * adj / 100) if current_hrc else None
        scenarios.append({
            "scenario":           label,
            "futures_chg_pct":    round(futures_impact_pct * adj, 2),
            "hrc_est":            round(hrc_new, 1) if hrc_new else None,
            "trade_flow_chg_pct": round(trade_flow_impact_pct * adj, 2),
        })

    return {
        # Layer 1
        "risk_score":             round(risk_score, 3),
        "steel_relevant":         l1["steel_relevant"],
        # Layer 2
        "event_type":             event_type,
        "persistence":            persistence,
        "persistence_reason":     l2.get("persistence_reason", ""),
        "initiator_countries":    initiators,
        "respondent_countries":   respondents,
        "affected_products":      products,
        "summary":                summary,
        # Layer 3
        "india_spillover":        india_spillover,
        # Quantified impact
        "futures_impact_pct":     round(futures_impact_pct, 2),
        "trade_flow_impact_pct":  round(trade_flow_impact_pct, 2),
        "calibration_factor":     calib_factor,
        "current_hrc_usd":        current_hrc,
        "scenarios":              scenarios,
        "layer":                  3 if l3 else (2 if l2 else 1),
        "model_note": (
            f"Risk score {risk_score:.2f}/1.0 (Layer 1, continuous 0-1 scale per AI-GPR methodology). "
            f"Persistence: {persistence} → {persist_scale}x multiplier "
            f"(AI-GPR finding: structural/persistent shocks have ~2x price impact vs one-off events). "
            f"Trade flow uses gravity elasticity ε=-1.5 for tariff events, literature multipliers otherwise."
        ),
    }


# ── RAG-augmented news impact ─────────────────────────────────────────────────

def analyze_news_impact_with_rag(announcement: str) -> dict:
    """
    Full pipeline: RAG retrieval for historical analogues → 3-layer AI-GPR impact analysis.
    RAG context is injected into Layers 1-3 prompts for richer semantic scoring.
    """
    rag_context = ""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from rag import rag_query
        result = rag_query(f"historical trade impact precedent: {announcement[:300]}", top_k=3)
        rag_context = result.get("answer", "")
    except Exception:
        pass
    return analyze_news_impact(announcement, rag_context=rag_context)


# ── Steel-GPR Index builder (from RSS feed) ───────────────────────────────────

STEEL_GPR_INDEX_PATH = CACHE_DIR / "steel_gpr_index.json"


def build_steel_gpr_index(articles: list[dict], save: bool = True) -> "pd.DataFrame":
    """
    Build a daily Steel-GPR index from RSS feed articles, following the AI-GPR methodology:

      Steel-GPR_t = (1/A_t) × Σ_i S_it × (1/S̄)

    where S_it = Layer-1 risk score for article i on day t,
          A_t  = total articles published on day t (normalises for volume),
          S̄   = mean score over the baseline window (normalised to 100).

    articles: list of {"title": str, "text": str, "published": "YYYY-MM-DD", ...}
    Returns DataFrame with columns: date, article_count, mean_score, steel_gpr_index.
    """
    import pandas as pd
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return pd.DataFrame()

    client = Groq(api_key=api_key)

    records = []
    print(f"[Steel-GPR] Scoring {len(articles)} articles …")
    for art in articles:
        text = f"{art.get('title','')} {art.get('text','')}"[:2000]
        date = art.get("published", datetime.utcnow().strftime("%Y-%m-%d"))[:10]
        try:
            scored = score_article(text, client=client)
            records.append({"date": date, "score": scored["risk_score"],
                            "steel_relevant": scored["steel_relevant"]})
        except Exception:
            records.append({"date": date, "score": 0.0, "steel_relevant": False})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate daily: mean score / article count → raw index
    daily = (df.groupby("date")
               .agg(article_count=("score", "count"),
                    mean_score=("score", "mean"),
                    steel_relevant_count=("steel_relevant", "sum"))
               .reset_index())

    # Normalise to 100 over baseline period (use available window if short)
    baseline_mean = daily["mean_score"].mean()
    norm = 100.0 / baseline_mean if baseline_mean > 0 else 100.0
    daily["steel_gpr_index"] = (daily["mean_score"] * norm).round(2)

    if save:
        STEEL_GPR_INDEX_PATH.write_text(
            daily.to_json(orient="records", date_format="iso")
        )
        print(f"[Steel-GPR] Saved index → {STEEL_GPR_INDEX_PATH}")

    return daily


def load_steel_gpr_index() -> "pd.DataFrame":
    """Load the saved Steel-GPR index (built from RSS feed)."""
    import pandas as pd
    if not STEEL_GPR_INDEX_PATH.exists():
        return pd.DataFrame()
    df = pd.read_json(STEEL_GPR_INDEX_PATH)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ── Snapshot for dashboard ────────────────────────────────────────────────────

def get_futures_snapshot() -> dict:
    """
    One-call summary used by the dashboard tab.
    Returns:
      {
        "prices":    {ticker: {name, currency, last, chg_pct_1d, chg_pct_5d}},
        "hrc_forecast": forecast dict for HRC=F,
        "last_updated": ISO timestamp,
      }
    """
    import pandas as pd

    data   = fetch_futures_data(period="1y")
    prices = {}

    for ticker, df in data.items():
        if df.empty or len(df) < 2:
            continue
        close   = df["Close"].dropna()
        last    = float(close.iloc[-1])
        prev1   = float(close.iloc[-2])
        prev5   = float(close.iloc[-6]) if len(close) >= 6 else prev1
        chg1    = (last - prev1) / prev1 * 100
        chg5    = (last - prev5) / prev5 * 100
        prices[ticker] = {
            "name":        TICKERS[ticker]["name"],
            "currency":    TICKERS[ticker]["currency"],
            "unit":        TICKERS[ticker]["unit"],
            "last":        round(last, 2),
            "chg_pct_1d":  round(chg1, 2),
            "chg_pct_5d":  round(chg5, 2),
        }

    # Prophet forecast for HRC futures, conditioned on the Steel-GPR news index
    hrc_forecast = {}
    if "HRC=F" in data and not data["HRC=F"].empty:
        try:
            df_hrc = compute_technicals(data["HRC=F"])
            gpr_df = load_steel_gpr_index()
            hrc_forecast = forecast_price(df_hrc, days=FORECAST_DAYS, gpr_df=gpr_df)
        except Exception as e:
            print(f"[WARN] HRC forecast failed: {e}")

    return {
        "prices":       prices,
        "hrc_forecast": hrc_forecast,
        "last_updated": datetime.utcnow().isoformat(),
    }


# ── CLI smoke test ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Steel Futures — Smoke Test")
    print("=" * 60)

    # 1. Snapshot
    print("\n[1] Fetching price snapshot …")
    snap = get_futures_snapshot()
    print(f"Last updated: {snap['last_updated']}")
    for sym, p in snap["prices"].items():
        arrow = "▲" if p["chg_pct_1d"] >= 0 else "▼"
        print(f"  {sym:<18} {p['last']:>9.2f} {p['currency']}  "
              f"{arrow} {p['chg_pct_1d']:+.2f}% 1d  |  {p['chg_pct_5d']:+.2f}% 5d   {p['name']}")

    # 2. HRC Forecast
    fc = snap.get("hrc_forecast", {})
    if fc:
        print(f"\n[2] HRC Futures Forecast (Prophet, {FORECAST_DAYS}-day horizon)")
        print(f"  Current : {fc['current']:.1f} USD/short ton")
        print(f"  30-day  : {fc['target_30']:.1f}  ({fc['change_pct_30']:+.1f}%)")
        print(f"  60-day  : {fc['target_60']:.1f}  ({fc['change_pct_60']:+.1f}%)")
        print(f"  Trend   : {fc['trend'].upper()}")

    # 3. 3-Layer AI-GPR Impact Test
    test_cases = [
        ("Structural/high-impact",
         "India imposes 25% safeguard duty on hot-rolled steel coil imports "
         "for 200 days following surge in cheap Chinese and Korean steel."),
        ("Geopolitical spillover",
         "US raises Section 232 steel tariffs to 50% on all imports. "
         "China retaliates with 35% tariff on US agricultural goods."),
        ("Low relevance",
         "Federal Reserve keeps interest rates unchanged at 4.25% following FOMC meeting."),
    ]

    for label, ann in test_cases:
        print(f"\n[3] {label}")
        print(f"  Text: {ann[:90]}…")
        impact = analyze_news_impact(ann)
        if "error" in impact:
            print(f"  Error: {impact['error']}")
            continue

        print(f"  Layer      : {impact.get('layer', '?')}/3")
        print(f"  Risk score : {impact['risk_score']:.2f}/1.0  "
              f"(AI-GPR continuous scale, temp=0)")
        if impact.get("event_type"):
            print(f"  Event type : {impact['event_type']}  "
                  f"| Persistence: {impact.get('persistence','?')}  "
                  f"({impact.get('persistence_reason','')})")
        if impact.get("initiator_countries"):
            print(f"  Initiator  : {impact['initiator_countries']}  →  "
                  f"Respondent: {impact.get('respondent_countries', [])}")
        sp = impact.get("india_spillover", {})
        if sp:
            print(f"  India role : {sp.get('india_role','?').upper()}  "
                  f"| Diversion: {sp.get('trade_diversion_direction','?')}  "
                  f"| Spillover score: {sp.get('india_spillover_score', 0):.2f}")
            print(f"  India note : {sp.get('india_spillover_summary','')}")
        print(f"  Summary    : {impact.get('summary','')}")
        print(f"  Futures Δ  : {impact['futures_impact_pct']:+.1f}%  "
              f"| Trade Δ: {impact['trade_flow_impact_pct']:+.1f}%")
        if impact.get("scenarios"):
            print("  Scenarios  :")
            for sc in impact["scenarios"]:
                print(f"    {sc['scenario']:<12} futures {sc['futures_chg_pct']:+.1f}%  "
                      f"HRC ~{sc['hrc_est']} USD  trade {sc['trade_flow_chg_pct']:+.1f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
