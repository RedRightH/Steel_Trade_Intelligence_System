# Steel RAG Project Status Report

**Author:** Suchit Paul Santosh | B.Tech ECE, IIIT Kottayam
**Last updated:** June 12, 2026
**Reference:** SteelRAG_Replan_SuchitPaulSantosh.pdf (4-Week Build Guide)

---

## Executive summary

### Overall progress: **All plan items complete** (Weeks 1–4 + post-plan extensions)

Every gate from the replan has been formally measured and passes (or carries a
documented, justified exception). The platform now also includes a market
intelligence layer well beyond the original plan scope: an integrated
news → futures → markets dependency chain with event-study calibration and a
7-year walk-forward forecast backtest.

---

## Gate status (all weeks)

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| V1 NLI faithfulness | ≥ 0.65 | 0.883 | ✅ |
| V3 faithfulness Δ vs V1 | ≥ +0.10 | +0.117 (1.00 judge score) | ✅ |
| Corpus size | ≥ 1,000 chunks | 5,640 vectors | ✅ |
| Router accuracy | ≥ 80% | **10/10 = 100%** | ✅ |
| 5 agent event tests | 5/5 structured + cited | **5/5** | ✅ |
| Guardrails red-team | 0 FP / 0 FN | 20/20 | ✅ |
| Classifier gate | 5/5 | 5/5 (avg conf 0.976) | ✅ |
| DPO training documented | result table | eval_loss 0.100, reward acc 100% | ✅ |
| DPO faithfulness | ≥ Groq − 0.05 | 0.938 vs 1.000 (−0.062) | ❌ → keep Groq (valid null result per plan) |
| Gravity OLS R² | ≥ 0.5 | OLS 0.431 / **XGB 0.922** | ⚠️ OLS below, XGB well above — documented |
| Gravity wired to agent + dashboard | structured field + slider | `gravity_scenario` + Tab 6 | ✅ |
| Multimodal extraction | 3 test cases | 3/3 parsed | ✅ |
| Semantic cache hit rate | 20–35% | **60%** | ✅ |
| vLLM throughput | documented | Groq 562ms / 133 tok/s measured; vLLM Linux-only noted | ✅ |
| Policy brief | 1,500 words | `policy_brief.md` | ✅ |
| 3-version benchmark | report | `eval/benchmark_v1_vs_v2_vs_v3.md` | ✅ |

---

## Post-plan extensions (June 2026)

### Integrated news → futures → markets chain
The three previously independent pipelines (news scoring, futures forecasting,
gravity trade flow) are now one dependency chain — gate test 4/4
(`eval/test_news_futures_chain.py`):

1. **GPR-conditioned forecast** — the Steel-GPR daily news-risk index feeds
   Prophet as an external regressor (1,954 sessions of history: 22 documented
   events + live RSS scoring).
2. **Event-study calibration** — news impact multipliers calibrated against
   actual HRC=F abnormal returns around 22 documented events (2019–2025):
   sign agreement 59%, Pearson r = 0.469. Per-type factors applied at runtime,
   clipped to [−2.0, 2.5].
3. **Market opportunity ranker** — 83 export markets scored by gravity gap +
   6-month momentum + size + FTA status; news events map onto the ranking
   with boosted / at-risk flags.

### 7-year forecast backtest (the most important finding)
Walk-forward validation on HRC=F 2019–2026 (34 folds, 30-session horizon):

| Config | MAPE | Direction |
|--------|------|-----------|
| **Production (yearly + multiplicative + GPR)** | 11.2% | 62% |
| No-yearly variant (won the earlier 2-year backtest) | 13.6% | 47% |

**Lesson encoded in the repo:** the config that won a 2-year backtest collapsed
over 7 years — config selection on short windows is itself overfitting. The
production config was reverted accordingly, and ~11% MAPE / 62% direction is
quoted as the honest performance envelope for this asset class.

---

## What works well

1. **Retrieval** — hybrid Pinecone + BM25 + RRF + BGE rerank; faithfulness 1.00 on the eval set
2. **Routing** — 100% classification accuracy across 7 categories
3. **Agents** — 5/5 event gates with structured Pydantic output and citations; refusal-synthesizer fallback fixed the short-answer failure mode
4. **Live news** — RSS daemon classifies and upserts every 4h; articles immediately queryable; Steel-GPR index built daily
5. **Quantified news impact** — 3-layer AI-GPR (Iacoviello & Tong 2026) with measured calibration, not just literature coefficients
6. **Evaluation culture** — every claim in the docs traces to a JSON artifact in `eval/`

## Known limitations

1. HRC futures forecasting carries irreducible regime risk (25–35% MAPE in break periods)
2. Event-study sample is 22 events; several per-type factors rest on n=1 (clipped at runtime)
3. GPR regressor is backtest-neutral; its value is live conditioning
4. OLS gravity R² (0.43) below the 0.5 gate — XGBoost (0.92) used for predictions, OLS retained for interpretability
5. DPO model loses to Groq 70B (expected with 70 pairs); checkpoint retained for serving experiments
6. Local-only deployment (by choice); vLLM requires Linux/WSL2

## Deployment state

- **Streamlit dashboard** — local, port 8501, 6 tabs
- **FastAPI backend** — local, 7 routes, semantic cache
- **Decision:** keep local (user choice); Vercel/Render configs exist but unused

---

## Conclusion

The platform is feature-complete against the replan with every gate measured,
and extended with a calibrated, backtested market intelligence layer. The
remaining work is operational rather than developmental: let the RSS daemon
accumulate scored articles, re-run `eval/event_study_2019.py` and
`eval/backtest_futures.py` periodically, and keep the honest numbers honest.
