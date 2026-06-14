# Gravity model — the ≥0.5 R² gate, honestly accounted

**Status: documented exception (narrowly missed), not a pass.**

## The original gate

From the build plan (Week-3 gate), verbatim:

> *"Gravity model **OLS R-squared >= 0.5 on holdout set**, scenario function works"*

Three terms matter: it specifies **OLS** (not XGBoost), **≥ 0.5**, and **on a holdout set** (out-of-sample, not in-sample fit).

## What was previously reported vs. what is true

The project had been reporting **"XGBoost R² = 0.922"** as clearing this gate. That number is **in-sample and leakage-inflated** — it was computed by predicting on the full panel (including the rows the model trained on), and four of six gravity features (distance, contiguity, common language, FTA) are **time-invariant per country**, so a random split leaks country identity. It is neither OLS, nor a holdout, nor an honest skill estimate, so it does not bear on this gate.

## Honest numbers

Measured on the FY2018–26 panel (859 obs, 100 countries; `gravity_model.train_gravity_models`):

| Model / specification | In-sample R² | Random-holdout R² | Leave-country-out CV R² |
|---|---|---|---|
| **OLS — baseline (6 features)** | 0.431 | **0.420** | 0.337 |
| OLS + continent fixed effects | 0.445 | 0.429 | 0.283 |
| OLS + sub-region fixed effects | 0.502 | **0.476** | 0.246 |
| XGBoost — baseline | 0.936 | 0.836 | 0.267 |

- **The gate metric (OLS, holdout): 0.42 baseline → 0.48 with sub-region fixed effects. Narrowly missed.**
- In-sample, sub-region fixed effects just clear 0.50 (0.502) — but the gate specifies *holdout*, so that does not count.
- The honest skill at predicting a *never-seen* market (leave-country-out) is ~0.34 — modest.

## Why 0.5 on a holdout was optimistic for this problem

Aggregate gravity models do reach R² 0.8–0.9 — but only **in-sample and with importer + exporter fixed effects**. This model is **single-product** (steel only) and **single-exporter** (India only), evaluated **out-of-sample**. That strips out exactly the country-pair heterogeneity (entrepôt hubs, captive neighbours, buyer relationships, policy shocks) that gives gravity its high R². For single-commodity bilateral gravity evaluated out-of-sample, a holdout R² near 0.45–0.50 is a sound result; ~0.34 is the structural ceiling on a genuinely new market.

There is also a real tension: **sub-region fixed effects raise the holdout fit (0.42 → 0.48) but reduce new-market skill (LOCO 0.34 → 0.25)**, because regional dummies do not transfer to a held-out country. The opportunity ranker's job *is* new-market extrapolation, so production deliberately keeps the **baseline** OLS specification (best LOCO) rather than chasing the gate number with fixed effects.

## Richer regressors were tested and rejected

Three economically-motivated features were added and evaluated on leave-country-out CV (`eval/gravity_extended_features.py`):

1. Destination GDP per capita (World Bank `NY.GDP.PCAP.CD`)
2. Destination import tariff — manufactured-goods proxy (World Bank `TM.TAX.MANF.WM.AR.ZS`; the project holds no steel-specific partner tariffs)
3. Destination crude-steel production (worldsteel 2023, static)

| Feature set (OLS) | Leave-country-out R² |
|---|---|
| Baseline | 0.337 |
| + GDP/capita | 0.326 |
| + steel production | 0.332 |
| + destination tariff (+ missing flag) | 0.269 |
| All three | 0.229 |

Every addition raised **in-sample** R² (OLS 0.43 → 0.45, XGB 0.94 → 0.96) while leaving honest out-of-sample skill flat-to-worse — the classic overfitting signature that leave-country-out is designed to catch. The destination-tariff proxy is the main culprit (its missingness flag encodes a spurious "data-poor country" signal). **None were adopted.** A genuine lift would require steel-specific bilateral data (WITS partner-as-reporter tariff lines, time-varying capacity) the project does not hold.

## Decision

- **Do not claim the gate is passed.** It is a documented, justified exception: best honest OLS holdout ≈ 0.48 vs. a 0.50 target.
- **Keep the baseline OLS specification in production** for the opportunity ranking (best leave-country-out skill); use OLS coefficients for interpretable elasticities (+GDP 0.85, −distance 0.64, +FTA 0.44 — all textbook).
- **Treat gravity-gap rankings as an indicative candidate-screen**, then apply the geopolitical viability overlay (report §4.1).

Artifacts: `eval/gravity_extended_features.json`, `steel_rag/gravity_model.py` (`train_gravity_models` honest metrics), report §12.2–12.3.
