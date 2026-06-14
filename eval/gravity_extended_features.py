"""
eval/gravity_extended_features.py — Do economically-motivated regressors lift the
gravity model's HONEST (leave-country-out) skill?

Adds three features to the base gravity panel and measures whether out-of-sample
R² (leave-country-out CV) actually rises — the only metric that distinguishes real
structural signal from in-sample curve-fitting:

  1. ln_gdp_pc      : partner GDP per capita (World Bank NY.GDP.PCAP.CD), year-matched.
  2. dest_tariff    : destination's applied tariff on MANUFACTURED imports
                      (World Bank TM.TAX.MANF.WM.AR.ZS, country mean) — an honest
                      proxy for import openness (NOT steel-specific; the project's
                      WITS data only covers India as reporter). + missing flag.
  3. ln_steel_prod  : destination crude-steel production (worldsteel 2023, static),
                      a self-sufficiency / substitution signal. 0 for non-producers.

Reports in-sample AND leave-country-out R² for OLS and XGBoost, base vs extended,
plus the OLS coefficient table for the extended model. Promote to production only
if LOCO improves.

Output: eval/gravity_extended_features.json
Run:    python eval/gravity_extended_features.py
"""
import os, sys, json, time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "steel_rag"))
os.chdir(ROOT / "steel_rag")

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from gravity_model import build_gravity_dataset

CACHE = ROOT / "steel_rag" / "gravity_cache" / "wb_extended.json"
XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)

# ── worldsteel 2023 crude-steel production (Mt), major producers ──────────────
# Source: World Steel Association, "World Steel in Figures 2024" (Top producers).
# Keyed by TRADESTAT/gravity country name. Non-listed destinations → 0.
STEEL_PROD_MT = {
    "CHINA P RP": 1019.1, "JAPAN": 87.0, "U S A": 81.4, "RUSSIA": 75.8,
    "KOREA RP": 66.7, "GERMANY": 35.4, "TURKEY": 33.7, "BRAZIL": 31.9,
    "IRAN": 31.1, "ITALY": 21.1, "TAIWAN": 20.7, "VIETNAM SOC REP": 19.0,
    "MEXICO": 16.9, "INDONESIA": 16.0, "FRANCE": 10.7, "SPAIN": 11.4,
    "CANADA": 11.9, "POLAND": 7.9, "UKRAINE": 6.2, "AUSTRIA": 7.2,
    "BELGIUM": 6.9, "EGYPT A RP": 9.8, "SAUDI ARAB": 9.5, "U ARAB EMTS": 3.4,
    "NETHERLAND": 6.0, "MALAYSIA": 6.7, "THAILAND": 4.4, "U K": 5.6,
    "SOUTH AFRICA": 4.9, "SWEDEN": 4.2, "AUSTRALIA": 5.5, "SLOVAKIA": 4.0,
    "ARGENTINA": 4.0, "BANGLADESH PR": 8.0, "PHILIPPINES": 1.5, "GREECE": 1.3,
    "PAKISTAN IR": 4.0, "CZECH REP": 4.5, "FINLAND": 3.5, "PORTUGAL": 2.0,
    "QATAR": 1.2, "OMAN": 2.5, "SWITZERLAND": 1.3,
}


def _fetch_wb(indicator: str, iso2_list, start=2010, end=2023) -> dict:
    """Return {iso2: {year: value}} for a World Bank indicator."""
    codes = ";".join(sorted(set(iso2_list)))
    url = (f"https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}"
           f"?format=json&per_page=20000&date={start}:{end}")
    out: dict = {}
    try:
        data = requests.get(url, timeout=60).json()
        if len(data) < 2 or not data[1]:
            return out
        for rec in data[1]:
            iso2 = ((rec.get("country") or {}).get("id") or "").upper()
            yr, val = int(rec.get("date", 0)), rec.get("value")
            if iso2 and yr and val is not None:
                out.setdefault(iso2, {})[yr] = float(val)
    except Exception as e:
        print(f"[WB] {indicator} fetch failed: {e}")
    return out


def get_extended_wb(iso2_list):
    if CACHE.exists() and time.time() - json.loads(CACHE.read_text()).get("_ts", 0) < 86400 * 30:
        p = json.loads(CACHE.read_text())["data"]
        return ({k: {int(y): v for y, v in d.items()} for k, d in p["gdp_pc"].items()},
                {k: {int(y): v for y, v in d.items()} for k, d in p["tariff"].items()})
    print("[WB] fetching GDP per capita + manufactured tariff …")
    gdp_pc = _fetch_wb("NY.GDP.PCAP.CD", iso2_list)
    tariff = _fetch_wb("TM.TAX.MANF.WM.AR.ZS", iso2_list)
    CACHE.write_text(json.dumps({"_ts": time.time(), "data": {"gdp_pc": gdp_pc, "tariff": tariff}}))
    return gdp_pc, tariff


def build_extended():
    df = build_gravity_dataset().copy()
    iso2 = df["iso2"].dropna().unique().tolist()
    gdp_pc, tariff = get_extended_wb(iso2)

    def _pc(r):
        d = gdp_pc.get(r["iso2"], {})
        for y in (r["fy_start"], r["fy_start"] - 1, r["fy_start"] + 1):
            if y in d:
                return d[y]
        return np.nan
    df["gdp_pc"] = df.apply(_pc, axis=1)
    df["gdp_pc"] = df["gdp_pc"].fillna(df["gdp_pc"].median())
    df["ln_gdp_pc"] = np.log(df["gdp_pc"])

    # Destination tariff: country mean over available years (slow-moving), + flag
    tar_mean = {k: np.mean(list(v.values())) for k, v in tariff.items() if v}
    df["dest_tariff"] = df["iso2"].map(tar_mean)
    df["tariff_missing"] = df["dest_tariff"].isna().astype(int)
    df["dest_tariff"] = df["dest_tariff"].fillna(np.nanmedian(list(tar_mean.values())))

    # Destination steel production (static worldsteel), 0 for non-producers
    df["steel_prod_mt"] = df["country"].map(STEEL_PROD_MT).fillna(0.0)
    df["ln_steel_prod"] = np.log1p(df["steel_prod_mt"])

    cov_t = 100 * (1 - df["tariff_missing"].mean())
    cov_s = 100 * (df["steel_prod_mt"] > 0).mean()
    print(f"Coverage — GDP/capita: ~full · dest tariff: {cov_t:.0f}% real (rest imputed) · "
          f"steel producers: {cov_s:.0f}% of rows")
    return df


BASE = ["ln_gdp_partner", "ln_distance", "contiguous", "common_language", "rta", "fy_start"]
NEW  = ["ln_gdp_pc", "dest_tariff", "tariff_missing", "ln_steel_prod"]
TARGET = "ln_exports"


def evaluate(df, feats, label):
    X, y, grp = df[feats].values, df[TARGET].values, df["country"].values
    gkf = GroupKFold(n_splits=5)
    res = {}
    for name, mk in [("OLS", lambda: LinearRegression()),
                     ("XGB", lambda: XGBRegressor(**XGB_PARAMS))]:
        insample = mk().fit(X, y).predict(X)
        loco = cross_val_predict(mk(), X, y, cv=gkf, groups=grp)
        res[name] = {"r2_insample": round(r2_score(y, insample), 3),
                     "r2_loco": round(r2_score(y, loco), 3)}
    print(f"\n{label}  (p={len(feats)} features, n={len(df)})")
    for k, v in res.items():
        print(f"   {k:<4} in-sample R²={v['r2_insample']:.3f}   leave-country-out R²={v['r2_loco']:.3f}")
    return res


def main():
    print("=" * 72)
    print("GRAVITY — EXTENDED REGRESSORS · honest (leave-country-out) evaluation")
    print("=" * 72)
    df = build_extended()

    base = evaluate(df, BASE, "BASELINE features")
    ext  = evaluate(df, BASE + NEW, "EXTENDED (+ gdp/capita, dest tariff, steel prod)")

    # OLS coefficients for the extended model (significance of new features)
    formula = "ln_exports ~ " + " + ".join(BASE + NEW)
    ols = smf.ols(formula, data=df).fit()
    print("\nExtended OLS — new-feature coefficients:")
    coef_rows = []
    for v in NEW:
        c, p = ols.params.get(v, np.nan), ols.pvalues.get(v, np.nan)
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"   {v:<16} coef={c:+.3f}  p={p:.3f} {sig}")
        coef_rows.append({"variable": v, "coef": round(float(c), 4),
                          "p_value": round(float(p), 4), "sig": bool(p < 0.05)})

    d_ols  = ext["OLS"]["r2_loco"] - base["OLS"]["r2_loco"]
    d_xgb  = ext["XGB"]["r2_loco"] - base["XGB"]["r2_loco"]
    print("\n" + "=" * 72)
    print(f"Leave-country-out lift:  OLS {base['OLS']['r2_loco']:.3f} → {ext['OLS']['r2_loco']:.3f} "
          f"({d_ols:+.3f})   |   XGB {base['XGB']['r2_loco']:.3f} → {ext['XGB']['r2_loco']:.3f} ({d_xgb:+.3f})")
    best = max([("OLS", ext["OLS"]["r2_loco"]), ("XGB", ext["XGB"]["r2_loco"])], key=lambda t: t[1])
    print(f"Best honest out-of-sample model (extended): {best[0]} at R²={best[1]:.3f}")
    print("=" * 72)

    out = ROOT / "eval" / "gravity_extended_features.json"
    out.write_text(json.dumps({
        "n_obs": len(df), "features_base": BASE, "features_new": NEW,
        "baseline": base, "extended": ext,
        "loco_lift": {"ols": round(d_ols, 3), "xgb": round(d_xgb, 3)},
        "extended_new_coefficients": coef_rows,
        "data_sources": {
            "gdp_pc": "World Bank NY.GDP.PCAP.CD",
            "dest_tariff": "World Bank TM.TAX.MANF.WM.AR.ZS (manufactured-goods proxy, not steel-specific)",
            "steel_prod": "worldsteel, World Steel in Figures 2024 (static 2023, major producers)",
        },
    }, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
