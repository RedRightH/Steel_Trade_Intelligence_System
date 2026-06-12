"""
eval/backtest_futures.py — Walk-forward backtest of the HRC futures forecast model.

Design
──────
Rolling-origin evaluation on 2 years of HRC=F daily closes:
  - Cutoffs every ~25 business days, starting once 250 sessions of history exist.
  - At each cutoff: fit Prophet on closes <= cutoff, forecast 30 business days,
    score against the actual closes that materialised.

Three configurations compared:
  A. baseline      — production Prophet params (weekly+yearly seasonality,
                     multiplicative, changepoint_prior=0.05)
  B. event_gpr     — baseline + news-risk regressor built from the 10 documented
                     steel trade events (risk scores reused from
                     eval/event_study_results.json — no API calls).
                     NO LOOK-AHEAD: future regressor values only continue the
                     decay of events that occurred before the cutoff.
  C. tuned         — additive seasonality, no yearly term (2y of data is too
                     short to learn a stable yearly cycle), changepoint=0.1,
                     + the same event-GPR regressor.

Metrics per config: MAE, MAPE, RMSE on the 30-day path, and directional
accuracy (sign of predicted 30-day move vs actual).

Outputs:
  eval/futures_backtest_results.json
  steel_rag/futures_cache/event_gpr_history.json  (historical news-risk series
                                                   consumed by steel_futures)

Run:  python eval/backtest_futures.py
"""
import os, sys, json, math
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "steel_rag"))
os.chdir(ROOT / "steel_rag")

import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import numpy as np
import pandas as pd

HORIZON_BDAYS   = 30
MIN_TRAIN       = 250    # sessions before first cutoff
CUTOFF_STEP     = 25     # business days between cutoffs
DECAY_HALFLIFE  = 3.0    # business days — event risk decays by half every 3 sessions
EVENT_SCALE     = 100.0  # index points added per unit of risk score on event day


def load_prices() -> pd.Series:
    from steel_futures import fetch_ticker
    df = fetch_ticker("HRC=F", period="2y")
    s = df["Close"].dropna()
    s.index = s.index.tz_localize(None) if s.index.tz else s.index
    return s


def build_event_gpr_series(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Historical news-risk index over the price calendar:
    baseline 100; each documented event adds risk_score×EVENT_SCALE on its date,
    decaying exponentially (half-life DECAY_HALFLIFE sessions). Events overlap
    additively.
    """
    ev_path = ROOT / "eval" / "event_study_results.json"
    events = json.loads(ev_path.read_text())["events"]

    idx = pd.Series(100.0, index=dates)
    pos = {d: i for i, d in enumerate(dates)}
    lam = math.log(2) / DECAY_HALFLIFE

    for ev in events:
        ts = pd.Timestamp(ev["date"])
        after = dates[dates >= ts]
        if after.empty:
            continue
        start_i = pos[after[0]]
        bump = float(ev["risk_score"]) * EVENT_SCALE
        for i in range(start_i, min(start_i + 15, len(dates))):  # ~5 half-lives
            idx.iloc[i] += bump * math.exp(-lam * (i - start_i))
    return idx


def fit_and_forecast(train: pd.Series, horizon_dates: pd.DatetimeIndex,
                     config: str, gpr: pd.Series | None) -> np.ndarray:
    """Fit one Prophet config on `train`, return yhat aligned to horizon_dates."""
    from prophet import Prophet

    pdf = pd.DataFrame({"ds": train.index, "y": train.values})

    if config == "baseline":
        m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, changepoint_prior_scale=0.05,
                    seasonality_mode="multiplicative", interval_width=0.80)
    elif config == "event_gpr":
        m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, changepoint_prior_scale=0.05,
                    seasonality_mode="multiplicative", interval_width=0.80)
    else:  # tuned
        m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=False, changepoint_prior_scale=0.1,
                    seasonality_mode="additive", interval_width=0.80)

    use_gpr = config in ("event_gpr", "tuned") and gpr is not None
    if use_gpr:
        pdf["gpr"] = gpr.reindex(train.index).fillna(100.0).values
        m.add_regressor("gpr", mode="additive")

    m.fit(pdf)

    future = pd.DataFrame({"ds": horizon_dates})
    if use_gpr:
        # No look-ahead: continue decay of pre-cutoff events only.
        # Decay from the last observed pre-cutoff level toward baseline 100.
        last_level = float(pdf["gpr"].iloc[-1])
        lam = math.log(2) / DECAY_HALFLIFE
        future["gpr"] = [100.0 + (last_level - 100.0) * math.exp(-lam * (k + 1))
                         for k in range(len(horizon_dates))]

    fc = m.predict(future)
    return fc["yhat"].values


def main():
    print("=" * 72)
    print("FUTURES FORECAST BACKTEST — walk-forward on HRC=F")
    print(f"Horizon {HORIZON_BDAYS} sessions · cutoff every {CUTOFF_STEP} · "
          f"min train {MIN_TRAIN}")
    print("=" * 72)

    close = load_prices()
    print(f"\nHRC=F: {close.index.min().date()} -> {close.index.max().date()} "
          f"({len(close)} sessions)")

    gpr_hist = build_event_gpr_series(close.index)
    n_event_days = int((gpr_hist > 100.5).sum())
    print(f"Event-GPR series: {n_event_days} elevated sessions "
          f"(peak {gpr_hist.max():.0f})")

    # Persist historical series for production use (merged with live RSS index)
    hist_out = ROOT / "steel_rag" / "futures_cache" / "event_gpr_history.json"
    hist_out.write_text(json.dumps(
        [{"date": d.strftime("%Y-%m-%d"), "steel_gpr_index": round(v, 2)}
         for d, v in gpr_hist.items()], indent=1))
    print(f"Historical GPR series saved -> {hist_out}")

    cutoff_idxs = list(range(MIN_TRAIN, len(close) - HORIZON_BDAYS, CUTOFF_STEP))
    print(f"Cutoffs: {len(cutoff_idxs)}")

    configs = ["baseline", "event_gpr", "tuned"]
    per_cfg = {c: {"mae": [], "mape": [], "rmse": [], "dir_ok": []} for c in configs}
    fold_rows = []

    for ci, cut in enumerate(cutoff_idxs):
        train = close.iloc[:cut]
        actual = close.iloc[cut:cut + HORIZON_BDAYS]
        hdates = actual.index
        current = float(train.iloc[-1])
        actual_move = float(actual.iloc[-1]) - current

        row = {"cutoff": str(train.index[-1].date()), "n_train": cut,
               "actual_30d_move_pct": round(actual_move / current * 100, 2)}

        for cfg in configs:
            try:
                yhat = fit_and_forecast(train, hdates, cfg, gpr_hist)
                err  = yhat - actual.values
                mae  = float(np.mean(np.abs(err)))
                mape = float(np.mean(np.abs(err) / actual.values)) * 100
                rmse = float(np.sqrt(np.mean(err ** 2)))
                pred_move = float(yhat[-1]) - current
                dir_ok = (pred_move > 0) == (actual_move > 0)
                per_cfg[cfg]["mae"].append(mae)
                per_cfg[cfg]["mape"].append(mape)
                per_cfg[cfg]["rmse"].append(rmse)
                per_cfg[cfg]["dir_ok"].append(dir_ok)
                row[cfg] = {"mae": round(mae, 1), "mape": round(mape, 2),
                            "pred_move_pct": round(pred_move / current * 100, 2),
                            "dir_ok": dir_ok}
            except Exception as e:
                row[cfg] = {"error": str(e)[:120]}

        fold_rows.append(row)
        parts = []
        for cfg in configs:
            r = row.get(cfg, {})
            if "mape" in r:
                parts.append(f"{cfg}: MAPE {r['mape']:.1f}% "
                             f"{'OK' if r['dir_ok'] else 'X'}")
        print(f"  [{ci+1}/{len(cutoff_idxs)}] cutoff {row['cutoff']} "
              f"(actual {row['actual_30d_move_pct']:+.1f}%)  |  " + "  ".join(parts))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BACKTEST SUMMARY")
    print("=" * 72)
    print(f"{'Config':<12} {'MAE $':>8} {'MAPE %':>8} {'RMSE $':>8} {'Direction':>12}")
    summary = {}
    for cfg in configs:
        d = per_cfg[cfg]
        if not d["mae"]:
            continue
        dir_rate = sum(d["dir_ok"]) / len(d["dir_ok"])
        summary[cfg] = {
            "mae":  round(float(np.mean(d["mae"])), 2),
            "mape": round(float(np.mean(d["mape"])), 2),
            "rmse": round(float(np.mean(d["rmse"])), 2),
            "directional_accuracy": round(dir_rate, 3),
            "n_folds": len(d["mae"]),
        }
        print(f"{cfg:<12} {summary[cfg]['mae']:>8.1f} {summary[cfg]['mape']:>8.2f} "
              f"{summary[cfg]['rmse']:>8.1f} "
              f"{sum(d['dir_ok'])}/{len(d['dir_ok'])} = {dir_rate:>5.0%}")

    # Winner: lowest MAPE, tie-break on directional accuracy
    winner = min(summary, key=lambda c: (summary[c]["mape"],
                                         -summary[c]["directional_accuracy"]))
    print(f"\nWinner: {winner}")

    results = {
        "methodology": {
            "series": "HRC=F daily close, 2y",
            "horizon_bdays": HORIZON_BDAYS,
            "cutoff_step": CUTOFF_STEP,
            "min_train": MIN_TRAIN,
            "gpr_regressor": (f"event-based, decay half-life {DECAY_HALFLIFE} sessions, "
                              "no look-ahead in forecast window"),
            "n_folds": len(cutoff_idxs),
        },
        "summary": summary,
        "winner": winner,
        "folds": fold_rows,
    }
    out = ROOT / "eval" / "futures_backtest_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
