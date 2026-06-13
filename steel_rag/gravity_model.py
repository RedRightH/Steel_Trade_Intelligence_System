"""
gravity_model.py — Gravity model for India steel export flows.

Equation (PPML-style log-linear, estimated via OLS on logs):
  ln(exports_ij) = α + β₁·ln(GDP_j) + β₂·ln(dist_ij)
                 + β₃·contiguous + β₄·common_language + β₅·rta
                 + γ_t (year FE) + ε_ij

Models:  OLS (statsmodels) + XGBoost
Data:    TRADESTAT annual exports (FY 2018–2026)
         World Bank GDP (cached locally)
         Haversine distances from embedded capital coordinates

Usage:
    from gravity_model import predict_trade_flow, get_gravity_insights, ensure_model_ready
    ensure_model_ready()   # trains + caches on first call (~5s)
    result = predict_trade_flow("U ARAB EMTS", gdp_growth_pct=5.0)
"""

import json
import math
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT           = Path(__file__).parent
CACHE_DIR       = _ROOT / "gravity_cache"
GDP_CACHE_FILE  = CACHE_DIR / "wb_gdp.json"
MODEL_PKL       = CACHE_DIR / "gravity_models.pkl"
CACHE_DIR.mkdir(exist_ok=True)

# India — New Delhi
INDIA_LAT, INDIA_LON = 28.63, 77.22

# ── Country reference table ───────────────────────────────────────────────────
# TRADESTAT name → (iso2, capital_lat, capital_lon, contiguous, english_official, rta_india)
# rta_india = 1 if covered by SAFTA / ASEAN FTA / CEPA / CECA / bilateral PTA
COUNTRY_META: dict[str, tuple] = {
    # South Asia — SAFTA members (+ bilateral neighbours)
    "NEPAL":            ("NP",  27.70,  85.32, 1, 0, 1),
    "BANGLADESH PR":    ("BD",  23.72,  90.41, 1, 0, 1),
    "SRI LANKA DSR":    ("LK",   6.93,  79.84, 0, 0, 1),
    "PAKISTAN IR":      ("PK",  33.72,  73.04, 1, 0, 1),
    "BHUTAN":           ("BT",  27.47,  89.64, 1, 0, 1),
    "MALDIVES":         ("MV",   4.18,  73.51, 0, 0, 1),
    "AFGHANISTAN":      ("AF",  34.52,  69.18, 0, 0, 1),
    # East Asia
    "CHINA P RP":       ("CN",  39.91, 116.38, 1, 0, 0),
    "JAPAN":            ("JP",  35.69, 139.69, 0, 0, 1),
    "KOREA RP":         ("KR",  37.57, 126.98, 0, 0, 1),
    "TAIWAN":           ("TW",  25.05, 121.55, 0, 0, 0),
    "HONG KONG":        ("HK",  22.32, 114.18, 0, 1, 0),
    "MACAO":            ("MO",  22.20, 113.55, 0, 0, 0),
    "MONGOLIA":         ("MN",  47.91, 106.92, 0, 0, 0),
    # Southeast Asia — ASEAN FTA members
    "VIETNAM SOC REP":  ("VN",  21.03, 105.85, 0, 0, 1),
    "THAILAND":         ("TH",  13.75, 100.52, 0, 0, 1),
    "MALAYSIA":         ("MY",   3.15, 101.69, 0, 1, 1),
    "INDONESIA":        ("ID",  -6.21, 106.85, 0, 0, 1),
    "PHILIPPINES":      ("PH",  14.60, 120.98, 0, 0, 1),
    "SINGAPORE":        ("SG",   1.35, 103.82, 0, 1, 1),
    "MYANMAR":          ("MM",  19.76,  96.08, 1, 0, 1),
    "CAMBODIA":         ("KH",  11.56, 104.92, 0, 0, 1),
    "LAO PD RP":        ("LA",  17.97, 102.60, 0, 0, 1),
    "BRUNEI":           ("BN",   4.94, 114.95, 0, 1, 1),
    "TIMOR LESTE":      ("TL",  -8.56, 125.58, 0, 0, 0),
    # West Asia / Middle East
    "U ARAB EMTS":      ("AE",  24.47,  54.37, 0, 0, 1),  # CEPA 2022
    "SAUDI ARAB":       ("SA",  24.69,  46.72, 0, 0, 0),
    "TURKEY":           ("TR",  39.93,  32.86, 0, 0, 0),
    "IRAN":             ("IR",  35.69,  51.42, 0, 0, 0),
    "IRAQ":             ("IQ",  33.34,  44.40, 0, 0, 0),
    "ISRAEL":           ("IL",  31.77,  35.22, 0, 0, 0),
    "JORDAN":           ("JO",  31.95,  35.93, 0, 0, 0),
    "KUWAIT":           ("KW",  29.37,  47.98, 0, 0, 0),
    "OMAN":             ("OM",  23.61,  58.59, 0, 0, 0),
    "QATAR":            ("QA",  25.28,  51.52, 0, 0, 0),
    "BAHARAIN IS":      ("BH",  26.21,  50.59, 0, 0, 0),
    "YEMEN REPUBLC":    ("YE",  15.55,  44.21, 0, 0, 0),
    "SYRIA":            ("SY",  33.51,  36.29, 0, 0, 0),
    "LEBANON":          ("LB",  33.89,  35.50, 0, 0, 0),
    # Europe
    "ITALY":            ("IT",  41.90,  12.48, 0, 0, 0),
    "GERMANY":          ("DE",  52.52,  13.40, 0, 0, 0),
    "SPAIN":            ("ES",  40.42,  -3.70, 0, 0, 0),
    "FRANCE":           ("FR",  48.86,   2.35, 0, 0, 0),
    "UNITED KINGDOM":   ("GB",  51.51,  -0.13, 0, 1, 0),
    "U K":              ("GB",  51.51,  -0.13, 0, 1, 0),  # TRADESTAT alias
    "NETHERLANDS":      ("NL",  52.37,   4.90, 0, 0, 0),
    "NETHERLAND":       ("NL",  52.37,   4.90, 0, 0, 0),  # TRADESTAT alias
    "BELGIUM":          ("BE",  50.85,   4.35, 0, 0, 0),
    "GREECE":           ("GR",  37.98,  23.73, 0, 0, 0),
    "POLAND":           ("PL",  52.23,  21.01, 0, 0, 0),
    "SWEDEN":           ("SE",  59.33,  18.07, 0, 0, 0),
    "DENMARK":          ("DK",  55.68,  12.57, 0, 0, 0),
    "FINLAND":          ("FI",  60.17,  24.94, 0, 0, 0),
    "NORWAY":           ("NO",  59.91,  10.75, 0, 0, 0),
    "PORTUGAL":         ("PT",  38.72,  -9.14, 0, 0, 0),
    "CZECHIA":          ("CZ",  50.08,  14.44, 0, 0, 0),
    "CZECH REP":        ("CZ",  50.08,  14.44, 0, 0, 0),
    "AUSTRIA":          ("AT",  48.21,  16.37, 0, 0, 0),
    "HUNGARY":          ("HU",  47.50,  19.04, 0, 0, 0),
    "ROMANIA":          ("RO",  44.43,  26.10, 0, 0, 0),
    "UKRAINE":          ("UA",  50.45,  30.52, 0, 0, 0),
    "RUSSIA":           ("RU",  55.75,  37.62, 0, 0, 0),
    "CROATIA":          ("HR",  45.81,  15.98, 0, 0, 0),
    "BULGARIA":         ("BG",  42.70,  23.32, 0, 0, 0),
    "SLOVAKIA":         ("SK",  48.15,  17.11, 0, 0, 0),
    "SLOVENIA":         ("SI",  46.05,  14.51, 0, 0, 0),
    "SERBIA":           ("RS",  44.80,  20.46, 0, 0, 0),
    "SWITZERLAND":      ("CH",  46.95,   7.45, 0, 0, 0),
    "IRELAND":          ("IE",  53.33,  -6.25, 0, 1, 0),
    "LUXEMBOURG":       ("LU",  49.61,   6.13, 0, 0, 0),
    # Americas
    "USA":              ("US",  38.90, -77.04, 0, 1, 0),
    "U S A":            ("US",  38.90, -77.04, 0, 1, 0),  # TRADESTAT alias
    "CANADA":           ("CA",  45.42, -75.70, 0, 1, 0),
    "BRAZIL":           ("BR", -15.78, -47.93, 0, 0, 0),
    "MEXICO":           ("MX",  19.43, -99.13, 0, 0, 0),
    "ARGENTINA":        ("AR", -34.60, -58.38, 0, 0, 0),
    "CHILE":            ("CL", -33.46, -70.65, 0, 0, 1),  # PTA
    "COLOMBIA":         ("CO",   4.71, -74.07, 0, 0, 0),
    "PERU":             ("PE", -12.05, -77.05, 0, 0, 0),
    "VENEZUELA":        ("VE",  10.49, -66.88, 0, 0, 0),
    "CUBA":             ("CU",  23.13, -82.38, 0, 0, 0),
    "COSTA RICA":       ("CR",   9.93, -84.09, 0, 0, 0),
    "ECUADOR":          ("EC",  -0.23, -78.52, 0, 0, 0),
    "TRINIDAD TBG":     ("TT",  10.65, -61.52, 0, 1, 0),
    "DOMINICAN REP":    ("DO",  18.48, -69.90, 0, 0, 0),
    # Africa
    "SOUTH AFRICA":     ("ZA", -25.75,  28.19, 0, 1, 0),
    "EGYPT A REP":      ("EG",  30.06,  31.25, 0, 0, 0),
    "EGYPT A RP":       ("EG",  30.06,  31.25, 0, 0, 0),  # TRADESTAT alias
    "NIGERIA":          ("NG",   9.07,   7.40, 0, 1, 0),
    "KENYA":            ("KE",  -1.29,  36.82, 0, 1, 0),
    "ETHIOPIA":         ("ET",   9.02,  38.75, 0, 0, 0),
    "TANZANIA REP":     ("TZ",  -6.17,  35.74, 0, 1, 0),
    "GHANA":            ("GH",   5.55,  -0.20, 0, 1, 0),
    "MOZAMBIQUE":       ("MZ", -25.97,  32.59, 0, 0, 0),
    "MAURITIUS":        ("MU", -20.16,  57.50, 0, 1, 1),  # CECPA
    "UGANDA":           ("UG",   0.32,  32.58, 0, 1, 0),
    "ANGOLA":           ("AO",  -8.84,  13.23, 0, 0, 0),
    "ZAMBIA":           ("ZM", -15.42,  28.28, 0, 1, 0),
    "ZIMBABWE":         ("ZW", -17.83,  31.05, 0, 1, 0),
    "ALGERIA":          ("DZ",  36.75,   3.04, 0, 0, 0),
    "MOROCCO":          ("MA",  34.02,  -6.83, 0, 0, 0),
    "TUNISIA":          ("TN",  36.82,  10.17, 0, 0, 0),
    "SENEGAL":          ("SN",  14.72, -17.47, 0, 0, 0),
    "CAMEROON":         ("CM",   3.87,  11.52, 0, 0, 0),
    "SEYCHELLES":       ("SC",  -4.62,  55.45, 0, 1, 0),
    "MADAGASCAR":       ("MG", -18.91,  47.54, 0, 0, 0),
    "SUDAN":            ("SD",  15.55,  32.53, 0, 0, 0),
    "DJIBOUTI":         ("DJ",  11.59,  43.15, 0, 0, 0),
    # Oceania
    "AUSTRALIA":        ("AU", -35.28, 149.13, 0, 1, 0),
    "NEW ZEALAND":      ("NZ", -41.29, 174.78, 0, 1, 0),
    "FIJI IS":          ("FJ", -18.14, 178.44, 0, 1, 0),
}


# ── Distance ──────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _distance_to_india(country: str) -> float | None:
    """Return distance in km from New Delhi to country capital. None if unknown."""
    meta = COUNTRY_META.get(country)
    if meta is None:
        return None
    _, lat, lon, *_ = meta
    return _haversine_km(INDIA_LAT, INDIA_LON, lat, lon)


# ── World Bank GDP ─────────────────────────────────────────────────────────────

def _fetch_wb_gdp(iso2_list: list[str], start: int = 2017, end: int = 2025) -> dict:
    """
    Fetch GDP (current USD) from World Bank API.
    Returns {iso2_upper: {year_int: gdp_float}}.
    """
    codes = ";".join(sorted(set(iso2_list)))
    url   = (
        f"https://api.worldbank.org/v2/country/{codes}/indicator/NY.GDP.MKTP.CD"
        f"?format=json&per_page=1000&date={start}:{end}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if len(data) < 2 or not data[1]:
            return {}
        result: dict = {}
        for rec in data[1]:
            # World Bank returns country.id = ISO2, countryiso3code = ISO3
            # We index by ISO2 so lookups match COUNTRY_META
            iso2 = ((rec.get("country") or {}).get("id") or "").upper()
            year = int(rec.get("date", 0))
            val  = rec.get("value")
            if iso2 and year and val:
                result.setdefault(iso2, {})[year] = float(val)
        return result
    except Exception as e:
        print(f"[gravity] World Bank GDP fetch failed: {e}")
        return {}


def _load_gdp_cache() -> dict:
    if GDP_CACHE_FILE.exists():
        try:
            with open(GDP_CACHE_FILE) as f:
                payload = json.load(f)
            # Refresh if cache is older than 30 days
            if time.time() - payload.get("_ts", 0) < 86400 * 30:
                raw = payload.get("data", {})
                # JSON converts int year keys → str; normalise back to int
                return {
                    iso: {int(yr): val for yr, val in years.items()}
                    for iso, years in raw.items()
                }
        except Exception:
            pass
    return {}


def _save_gdp_cache(data: dict):
    with open(GDP_CACHE_FILE, "w") as f:
        json.dump({"_ts": time.time(), "data": data}, f)


def get_wb_gdp(iso2_list: list[str]) -> dict:
    """
    Return cached World Bank GDP data.
    iso2_list: list of 2-letter ISO codes (will be mapped to WB iso3).
    Returns {iso2: {year: gdp}}.
    """
    cached = _load_gdp_cache()
    needed = [c for c in iso2_list if c not in cached]
    if needed:
        print(f"[gravity] Fetching GDP for {len(needed)} countries from World Bank...")
        fresh = _fetch_wb_gdp(needed)
        cached.update(fresh)
        _save_gdp_cache(cached)
    return cached


# ── Build dataset ─────────────────────────────────────────────────────────────

def build_gravity_dataset() -> pd.DataFrame:
    """
    Merge TRADESTAT annual exports + World Bank GDP + distances → gravity panel.

    Returns DataFrame with columns:
        country, fy, fy_start (int year), exports_usd,
        ln_exports, ln_gdp_partner, ln_distance,
        contiguous, common_language, rta
    """
    # 1. Load export data
    sys.path.insert(0, str(_ROOT))
    from data_agent import load_export_data

    df_raw = load_export_data()

    # 2. Aggregate to FY totals (sum monthly_curr_usd per country per FY)
    annual = (
        df_raw
        .groupby(["country", "fiscal_year"], as_index=False)["monthly_curr_usd"]
        .sum()
        .rename(columns={"monthly_curr_usd": "exports_usd", "fiscal_year": "fy"})
    )

    # FY start year: "FY2018-19" → 2018
    annual["fy_start"] = annual["fy"].str.extract(r"FY(\d{4})").astype(int)

    # Drop zero/tiny exports (< $1k) — structural zeros distort log model
    annual = annual[annual["exports_usd"] > 0.001].copy()

    # 3. Add country metadata
    def _meta(row):
        m = COUNTRY_META.get(row["country"])
        if m is None:
            return pd.Series({"iso2": None, "dist_km": None,
                              "contiguous": 0, "common_language": 0, "rta": 0})
        iso2, lat, lon, contig, eng, rta = m
        dist = _haversine_km(INDIA_LAT, INDIA_LON, lat, lon)
        return pd.Series({"iso2": iso2, "dist_km": dist,
                          "contiguous": contig, "common_language": eng, "rta": rta})

    meta_df = annual.apply(_meta, axis=1)
    annual  = pd.concat([annual, meta_df], axis=1)
    annual  = annual.dropna(subset=["iso2", "dist_km"])  # drop unmapped countries

    # 4. Fetch GDP
    iso2_list = annual["iso2"].dropna().unique().tolist()
    gdp_data  = get_wb_gdp(iso2_list)

    def _gdp(row):
        iso  = row["iso2"]
        year = row["fy_start"]
        country_gdp = gdp_data.get(iso, {})
        # Try requested year, then ±1
        for y in [year, year - 1, year + 1]:
            if y in country_gdp:
                return country_gdp[y]
        return np.nan

    annual["gdp_partner"] = annual.apply(_gdp, axis=1)
    annual = annual.dropna(subset=["gdp_partner"])
    annual = annual[annual["gdp_partner"] > 0]

    # 5. Log-transform
    annual["ln_exports"]     = np.log(annual["exports_usd"])
    annual["ln_gdp_partner"] = np.log(annual["gdp_partner"])
    annual["ln_distance"]    = np.log(annual["dist_km"])

    return annual.reset_index(drop=True)


# ── Train models ──────────────────────────────────────────────────────────────

FEATURES = ["ln_gdp_partner", "ln_distance", "contiguous", "common_language", "rta", "fy_start"]
TARGET   = "ln_exports"


def train_gravity_models(df: pd.DataFrame) -> dict:
    """
    Train OLS + XGBoost gravity models.
    Returns dict with model objects, metrics, and coefficient tables.
    """
    import statsmodels.formula.api as smf
    from sklearn.ensemble import GradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split

    df = df.copy()

    # ── OLS ───────────────────────────────────────────────────────────────────
    formula = (
        "ln_exports ~ ln_gdp_partner + ln_distance + "
        "contiguous + common_language + rta + fy_start"
    )
    ols_result = smf.ols(formula, data=df).fit()

    ols_pred = ols_result.fittedvalues
    ols_r2   = r2_score(df[TARGET], ols_pred)
    ols_mae  = mean_absolute_error(df[TARGET], ols_pred)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    X = df[FEATURES].values
    y = df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    xgb.fit(X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False)

    xgb_pred_all = xgb.predict(X)
    xgb_r2  = r2_score(y, xgb_pred_all)
    xgb_mae = mean_absolute_error(y, xgb_pred_all)

    # ── Feature importance ───────────────────────────────────────────────────
    fi = dict(zip(FEATURES, xgb.feature_importances_))

    # ── OLS coefficient table ────────────────────────────────────────────────
    coef_df = pd.DataFrame({
        "coefficient": ols_result.params,
        "std_err":     ols_result.bse,
        "p_value":     ols_result.pvalues,
        "significant": ols_result.pvalues < 0.05,
    }).drop(index="Intercept", errors="ignore")

    print(f"[gravity] OLS  R²={ols_r2:.3f}  MAE(ln)={ols_mae:.3f}  n={len(df)}")
    print(f"[gravity] XGB  R²={xgb_r2:.3f}  MAE(ln)={xgb_mae:.3f}")

    return {
        "ols":           ols_result,
        "xgb":           xgb,
        "features":      FEATURES,
        "coef_df":       coef_df,
        "feature_imp":   fi,
        "metrics": {
            "ols_r2": round(ols_r2, 4), "ols_mae": round(ols_mae, 4),
            "xgb_r2": round(xgb_r2, 4), "xgb_mae": round(xgb_mae, 4),
            "n_obs": len(df),
        },
        "df": df,   # keep dataset for baseline lookups
    }


# ── Load / cache ──────────────────────────────────────────────────────────────

_model_cache: dict | None = None


def ensure_model_ready(force_retrain: bool = False) -> dict:
    """
    Load gravity model from cache, or build + train from scratch.
    Returns the model dict.
    """
    global _model_cache

    if _model_cache is not None and not force_retrain:
        return _model_cache

    if MODEL_PKL.exists() and not force_retrain:
        try:
            with open(MODEL_PKL, "rb") as f:
                _model_cache = pickle.load(f)
            print(f"[gravity] Model loaded from cache ({MODEL_PKL})")
            return _model_cache
        except Exception as e:
            print(f"[gravity] Cache load failed ({e}), retraining...")

    print("[gravity] Building dataset and training models...")
    t0  = time.time()
    df  = build_gravity_dataset()
    mdl = train_gravity_models(df)

    with open(MODEL_PKL, "wb") as f:
        pickle.dump(mdl, f)

    _model_cache = mdl
    print(f"[gravity] Ready in {time.time()-t0:.1f}s  "
          f"({len(df)} obs, {df['country'].nunique()} countries, "
          f"{df['fy'].nunique()} FYs)")
    return _model_cache


# ── Prediction ────────────────────────────────────────────────────────────────

# Steel-specific trade elasticities (literature range: –0.5 to –2.0)
TARIFF_ELASTICITY = -1.5   # ln(exports) change per 1-unit change in ln(1 + tariff/100)

# ── Macro scenario assumptions ──────────────────────────────────────────────
# Per-market 2026 real-GDP growth (%). Source: IMF World Economic Outlook,
# Oct 2025 (rounded). A single elasticity model gives an IDENTICAL % response to
# a uniform GDP shock across markets, which makes cross-market scenarios
# meaningless; differentiating the *assumption* per market is what produces a
# genuine market-specific scenario. Unlisted markets fall back to GDP_DEFAULT.
GDP_OUTLOOK_SOURCE = "IMF World Economic Outlook, Oct 2025 (2026 real-GDP growth)"
GDP_DEFAULT = 2.5
GDP_OUTLOOK_2026: dict[str, float] = {
    # South Asia
    "NEPAL": 5.0, "BANGLADESH PR": 6.0, "SRI LANKA DSR": 3.5, "PAKISTAN IR": 3.0,
    "BHUTAN": 4.5, "MALDIVES": 4.5, "AFGHANISTAN": 2.5,
    # East Asia
    "CHINA P RP": 4.2, "JAPAN": 0.6, "KOREA RP": 2.0, "TAIWAN": 2.5,
    "HONG KONG": 2.4, "MONGOLIA": 5.0,
    # Southeast Asia
    "VIETNAM SOC REP": 6.0, "THAILAND": 2.8, "MALAYSIA": 4.5, "INDONESIA": 5.1,
    "PHILIPPINES": 6.1, "SINGAPORE": 2.2, "MYANMAR": 2.6, "CAMBODIA": 5.8,
    "LAO PD RP": 3.5, "BRUNEI": 2.5,
    # West Asia
    "U ARAB EMTS": 4.0, "SAUDI ARAB": 3.5, "TURKEY": 2.7, "IRAN": 2.0, "IRAQ": 3.0,
    "ISRAEL": 3.0, "JORDAN": 2.7, "KUWAIT": 2.5, "OMAN": 2.5, "QATAR": 2.4,
    "BAHARAIN IS": 3.0, "YEMEN REPUBLC": 2.0, "SYRIA": 2.0, "LEBANON": 1.0,
    # Europe
    "ITALY": 0.8, "GERMANY": 0.9, "SPAIN": 1.8, "FRANCE": 1.1, "UNITED KINGDOM": 1.3,
    "U K": 1.3, "NETHERLANDS": 1.3, "NETHERLAND": 1.3, "BELGIUM": 1.0, "GREECE": 2.0,
    "POLAND": 3.0, "SWEDEN": 1.8, "DENMARK": 1.5, "FINLAND": 1.2, "NORWAY": 1.4,
    "PORTUGAL": 2.0, "CZECHIA": 2.0, "CZECH REP": 2.0, "AUSTRIA": 1.1, "HUNGARY": 2.5,
    "ROMANIA": 3.0, "UKRAINE": 2.0, "RUSSIA": 1.0, "CROATIA": 2.5, "BULGARIA": 2.5,
    "SLOVAKIA": 2.0, "SLOVENIA": 2.0, "SERBIA": 3.5, "SWITZERLAND": 1.4,
    "IRELAND": 3.5, "LUXEMBOURG": 2.0,
    # Americas
    "USA": 1.8, "U S A": 1.8, "CANADA": 1.9, "BRAZIL": 2.2, "MEXICO": 1.5,
    "ARGENTINA": 3.0, "CHILE": 2.3, "COLOMBIA": 2.8, "PERU": 2.8, "VENEZUELA": 3.0,
    "CUBA": 1.0, "COSTA RICA": 3.5, "ECUADOR": 1.8, "TRINIDAD TBG": 2.0,
    "DOMINICAN REP": 4.5,
    # Africa
    "SOUTH AFRICA": 1.5, "EGYPT A REP": 4.0, "EGYPT A RP": 4.0, "NIGERIA": 3.2,
    "KENYA": 5.0, "ETHIOPIA": 6.5, "TANZANIA REP": 6.0, "GHANA": 4.5,
    "MOZAMBIQUE": 5.0, "MAURITIUS": 4.0, "UGANDA": 6.0, "ANGOLA": 3.0, "ZAMBIA": 5.0,
    "ZIMBABWE": 3.5, "ALGERIA": 3.0, "MOROCCO": 3.5, "TUNISIA": 2.0, "SENEGAL": 8.0,
    "CAMEROON": 4.5, "SEYCHELLES": 3.5, "MADAGASCAR": 4.5, "SUDAN": 2.0,
    "DJIBOUTI": 5.0,
    # Oceania
    "AUSTRALIA": 2.2, "NEW ZEALAND": 2.5, "FIJI IS": 3.0,
}

# Advanced markets with active steel import protection (Section 232, EU
# safeguard/CBAM, etc.) — carry heavier bear-case tariff risk.
PROTECTION_PRONE: set[str] = {
    "USA", "U S A", "CANADA", "UNITED KINGDOM", "U K",
    "GERMANY", "ITALY", "FRANCE", "SPAIN", "NETHERLANDS", "NETHERLAND", "BELGIUM",
    "AUSTRIA", "POLAND", "SWEDEN", "DENMARK", "FINLAND", "PORTUGAL", "GREECE",
    "CZECHIA", "CZECH REP", "HUNGARY", "ROMANIA", "CROATIA", "BULGARIA",
    "SLOVAKIA", "SLOVENIA", "IRELAND", "LUXEMBOURG",
}


def country_gdp_outlook(country: str) -> float:
    """2026 real-GDP growth assumption for a market (IMF WEO; default if unlisted)."""
    return GDP_OUTLOOK_2026.get(country, GDP_DEFAULT)


def predict_trade_flow(
    country: str,
    gdp_growth_pct: float = 0.0,
    tariff_change_pct: float = 0.0,
    model_type: str = "ols",        # "ols" (default — interpretable) | "xgb"
) -> dict:
    """
    Predict India steel exports to a country under a scenario.

    Parameters
    ----------
    country          : TRADESTAT country name (e.g. "U ARAB EMTS", "USA")
    gdp_growth_pct   : % change in partner GDP vs latest year (e.g. 5.0 = +5 %)
    tariff_change_pct: % change in effective tariff rate (e.g. -10 = tariff cut)
    model_type       : "xgb" (default) or "ols"

    Returns
    -------
    dict with baseline_usd, scenario_usd, change_pct, explanation, sources
    """
    mdl = ensure_model_ready()
    df  = mdl["df"]

    # Latest data for this country
    country_data = df[df["country"] == country].sort_values("fy_start")
    if country_data.empty:
        return {
            "country": country, "status": "no_data",
            "message": f"No gravity data for '{country}'. "
                        "Check TRADESTAT country name or run ensure_model_ready(force_retrain=True).",
        }

    latest = country_data.iloc[-1]

    # Baseline features
    base_features = {f: latest[f] for f in FEATURES}

    # Scenario: adjust ln_gdp_partner for GDP growth
    scenario_features = base_features.copy()
    scenario_features["ln_gdp_partner"] += math.log(1 + gdp_growth_pct / 100) if gdp_growth_pct != 0 else 0

    # Tariff effect: Δln(exports) = elasticity × ln(1 + Δtariff/100)
    tariff_effect = 0.0
    if tariff_change_pct != 0:
        tariff_effect = TARIFF_ELASTICITY * math.log(1 + tariff_change_pct / 100)

    X_feat = FEATURES

    if model_type == "ols":
        ols = mdl["ols"]
        import pandas as _pd
        base_row = _pd.DataFrame([base_features])
        scen_row = _pd.DataFrame([scenario_features])
        ln_base  = float(ols.predict(base_row).iloc[0])
        ln_scen  = float(ols.predict(scen_row).iloc[0]) + tariff_effect
    else:
        xgb = mdl["xgb"]
        X_base = np.array([[base_features[f] for f in X_feat]])
        X_scen = np.array([[scenario_features[f] for f in X_feat]])
        ln_base = float(xgb.predict(X_base)[0])
        ln_scen = float(xgb.predict(X_scen)[0]) + tariff_effect

    baseline_usd = math.exp(ln_base)
    scenario_usd = math.exp(ln_scen)
    change_pct   = (scenario_usd / baseline_usd - 1) * 100

    # Build explanation
    parts = []
    if gdp_growth_pct != 0:
        parts.append(f"GDP growth of {gdp_growth_pct:+.1f}%")
    if tariff_change_pct != 0:
        parts.append(f"tariff change of {tariff_change_pct:+.1f}%")
    scenario_desc = " + ".join(parts) if parts else "no change (baseline)"

    return {
        "country":       country,
        "fy_base":       latest["fy"],
        "baseline_usd":  round(baseline_usd, 4),    # USD million
        "scenario_usd":  round(scenario_usd, 4),
        "change_pct":    round(change_pct, 2),
        "scenario_desc": scenario_desc,
        "model_used":    model_type.upper(),
        "gdp_growth_pct":    gdp_growth_pct,
        "tariff_change_pct": tariff_change_pct,
        "tariff_elasticity": TARIFF_ELASTICITY,
    }


def predict_top_scenarios(countries: list[str] | None = None,
                          gdp_growth_pct: float | None = None,
                          model_type: str = "ols",
                          use_outlook: bool = True) -> pd.DataFrame:
    """
    Run a scenario for a list of countries and return a ranked DataFrame.

    By default (use_outlook=True, gdp_growth_pct=None) each market uses its OWN
    IMF 2026 GDP-growth outlook, so the % responses differ across markets. Pass
    an explicit gdp_growth_pct to apply one uniform shock to every market (the
    old behaviour) — note a single-elasticity model then returns identical %
    changes everywhere.
    """
    mdl = ensure_model_ready()
    if countries is None:
        countries = mdl["df"]["country"].unique().tolist()

    rows = []
    for c in countries:
        try:
            g = (gdp_growth_pct if gdp_growth_pct is not None
                 else (country_gdp_outlook(c) if use_outlook else 0.0))
            r = predict_trade_flow(c, gdp_growth_pct=g, model_type=model_type)
            if r.get("status") != "no_data":
                rows.append(r)
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("scenario_usd", ascending=False).reset_index(drop=True)


def run_scenario_matrix(countries: list[str] | None = None,
                        top_n: int = 15,
                        model_type: str = "ols") -> pd.DataFrame:
    """
    Bull / base / bear export scenarios with MARKET-SPECIFIC assumptions.

    For each market: baseline (no change), a bull case (GDP +1.5pp above the
    IMF 2026 outlook, tariff easing — deeper for non-FTA markets), and a bear
    case (global slowdown −3pp, protection — heaviest for protection-prone
    advanced markets). Because the GDP outlook and tariff path differ by market,
    the resulting bull/bear % changes are genuinely differentiated rather than a
    flat uniform-shock response.

    Returns DataFrame (descending baseline_usd_m):
      country, gdp_outlook_pct, fta, baseline_usd_m,
      bull_usd_m, bull_change_pct, bear_usd_m, bear_change_pct
    """
    mdl = ensure_model_ready()
    latest_fy = mdl["df"]["fy"].max()
    rta_map = (mdl["df"][mdl["df"]["fy"] == latest_fy]
               .set_index("country")["rta"].to_dict())

    if countries is None:
        countries = mdl["df"]["country"].unique().tolist()

    rows = []
    for c in countries:
        base = predict_trade_flow(c, model_type=model_type)
        if base.get("status") == "no_data":
            continue
        g = country_gdp_outlook(c)
        is_fta = int(rta_map.get(c, 0)) == 1

        bull = predict_trade_flow(c, gdp_growth_pct=g + 1.5,
                                  tariff_change_pct=(-5.0 if is_fta else -12.0),
                                  model_type=model_type)
        bear_tariff = (22.0 if c in PROTECTION_PRONE
                       else (8.0 if is_fta else 15.0))
        bear = predict_trade_flow(c, gdp_growth_pct=g - 3.0,
                                  tariff_change_pct=bear_tariff,
                                  model_type=model_type)
        rows.append({
            "country": c, "gdp_outlook_pct": g, "fta": is_fta,
            "baseline_usd_m": round(base["baseline_usd"], 1),
            "bull_usd_m": round(bull["scenario_usd"], 1),
            "bull_change_pct": round(bull["change_pct"], 1),
            "bear_usd_m": round(bear["scenario_usd"], 1),
            "bear_change_pct": round(bear["change_pct"], 1),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("baseline_usd_m", ascending=False).reset_index(drop=True)
    return df.head(top_n) if top_n else df


# ── Insights ──────────────────────────────────────────────────────────────────

def get_gravity_insights() -> dict:
    """
    Return model summary: OLS coefficients, XGBoost feature importance, metrics.

    Useful for dashboard display and policy brief.
    """
    mdl = ensure_model_ready()
    return {
        "metrics":      mdl["metrics"],
        "coef_df":      mdl["coef_df"],
        "feature_imp":  mdl["feature_imp"],
        "n_countries":  mdl["df"]["country"].nunique(),
        "n_obs":        len(mdl["df"]),
        "fy_range":     (mdl["df"]["fy"].min(), mdl["df"]["fy"].max()),
    }


def get_actual_vs_predicted() -> pd.DataFrame:
    """Return DataFrame with actual and predicted ln_exports for model fit plot."""
    mdl = ensure_model_ready()
    df  = mdl["df"].copy()
    xgb = mdl["xgb"]
    X   = df[FEATURES].values
    df["predicted_ln"] = xgb.predict(X)
    df["actual_ln"]    = df[TARGET]
    df["predicted_usd"] = np.exp(df["predicted_ln"])
    df["actual_usd"]    = np.exp(df["actual_ln"])
    df["residual"]      = df["actual_ln"] - df["predicted_ln"]
    return df[["country", "fy", "actual_usd", "predicted_usd",
               "actual_ln", "predicted_ln", "residual"]]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("Steel RAG — Gravity Model")
    print("=" * 64)

    mdl = ensure_model_ready()
    ins = get_gravity_insights()

    print(f"\nDataset: {ins['n_obs']} obs  |  {ins['n_countries']} countries  |  {ins['fy_range'][0]}–{ins['fy_range'][1]}")
    print(f"OLS  R²={ins['metrics']['ols_r2']}  MAE(ln)={ins['metrics']['ols_mae']}")
    print(f"XGB  R²={ins['metrics']['xgb_r2']}  MAE(ln)={ins['metrics']['xgb_mae']}")

    print("\n── OLS Coefficients ────────────────────────────────────────")
    print(ins["coef_df"].to_string())

    print("\n── XGBoost Feature Importance ──────────────────────────────")
    for feat, imp in sorted(ins["feature_imp"].items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<20} {imp:.4f}  {bar}")

    print("\n── Scenario: UAE +5% GDP growth (OLS) ──────────────────────")
    r = predict_trade_flow("U ARAB EMTS", gdp_growth_pct=5.0, model_type="ols")
    print(f"  Baseline:   ${r['baseline_usd']:.1f}M  →  Scenario: ${r['scenario_usd']:.1f}M  ({r['change_pct']:+.1f}%)")

    print("\n── Scenario: U S A –10% tariff cut (OLS) ───────────────────")
    r = predict_trade_flow("U S A", tariff_change_pct=-10.0, model_type="ols")
    if r.get("status") == "no_data":
        print(f"  {r['message']}")
    else:
        print(f"  Baseline:   ${r['baseline_usd']:.1f}M  →  Scenario: ${r['scenario_usd']:.1f}M  ({r['change_pct']:+.1f}%)")

    print("\n── Bull/Bear scenario matrix (market-specific assumptions) ──")
    print(f"  GDP outlook source: {GDP_OUTLOOK_SOURCE}")
    mat = run_scenario_matrix(top_n=10, model_type="ols")
    print(f"  {'Country':<22} {'GDP26':>6} {'FTA':>4} {'Base $M':>9} "
          f"{'Bull':>7} {'Bear':>7}")
    for _, row in mat.iterrows():
        print(f"  {row['country']:<22} {row['gdp_outlook_pct']:>5.1f}% "
              f"{('Y' if row['fta'] else '-'):>4} {row['baseline_usd_m']:>9.1f} "
              f"{row['bull_change_pct']:>+6.1f}% {row['bear_change_pct']:>+6.1f}%")

    print("\n" + "=" * 64)
