# India Steel Trade Intelligence Platform

A multi-agent RAG system for analysing Indian steel trade policy, export trends, MFN tariffs, steel futures, and news-driven market impact — built as an AI engineering capstone.

## Architecture

```
                            ┌─ Live RSS daemon ─ classify ─ upsert ─┐
Data sources ─ ingest ──────┤                                       ├─ Pinecone (5,640 vectors) + BM25
(PDF / XLSX / CSV / RSS)    └─ Steel-GPR daily risk index ──────────┘
                                                                        │
Question                                                                ▼
   │                                                          Hybrid RAG (dense + BM25
   ▼                                                          → RRF → BGE rerank → Groq 70B)
Router (classify → 7 types)                                             │
   ├── ANTI_DUMPING / SAFEGUARD / POLICY_OPPORTUNITY → PolicyAnalystAgent ──┤ (+ gravity scenario)
   ├── RAW_MATERIAL / CBAM_COMPLIANCE                → SupplyChainRiskAgent ┤
   ├── DATA_ANALYSIS                                 → DataAnalysisAgent ── TRADESTAT (99 months)
   └── TARIFF_ANALYSIS                               → TariffAnalysisAgent ─ WTO WITS (2010–23)

News event → 3-layer AI-GPR scoring → calibrated futures impact (event study, 22 events)
           → GPR-conditioned Prophet forecast → market opportunity ranker (gravity gap + momentum)
```

## Features

| Module | What it does |
|--------|-------------|
| `steel_rag/rag.py` | Hybrid retrieval: Pinecone dense + BM25 + RRF fusion + BGE cross-encoder rerank → Groq LLaMA-3.3-70b |
| `steel_rag/router.py` | 7-category classifier (100% gate accuracy), 4 agents, Pydantic outputs, answer synthesizer fallback |
| `steel_rag/data_agent.py` | 99 months of TRADESTAT export data (FY2018–2026), trend analysis, LLM chart generation |
| `steel_rag/tariff_agent.py` | WTO WITS MFN tariffs (HS 72 & 73, 2010–2023), trends, rankings, natural-language lookup |
| `steel_rag/gravity_model.py` | Trade gravity model (OLS R²=0.43, XGBoost R²=0.92), `predict_trade_flow()` scenarios |
| `steel_rag/steel_futures.py` | HRC futures + steel equities, Prophet forecast with Steel-GPR news regressor, 3-layer AI-GPR news impact (Iacoviello & Tong 2026) |
| `steel_rag/market_opportunity.py` | Ranks 83 export markets by gravity gap + momentum + size + FTA; maps news events to boosted/at-risk markets |
| `steel_rag/classifier_pipeline.py` | Live RSS daemon: fetch → classify → embed → Pinecone upsert + Steel-GPR index build |
| `steel_rag/semantic_cache.py` | SQLite semantic cache (cosine ≥ 0.92, 24h TTL, 60% measured hit rate) |
| `steel_rag/guardrails.py` | Domain guardrails: 20/20 red-team gate (0 false accepts/rejects) |
| `steel_rag/dashboard.py` | Streamlit app — 6 tabs: Query · Trade Flows · Tariffs · Eval · Futures & Impact · Gravity Scenarios |
| `api.py` | FastAPI backend (7 routes) with semantic cache |
| `multimodal/qwen_vl_extract.py` | Qwen2-VL-2B document image extraction (3/3 test cases pass) |
| `dpo/` | DPO fine-tune of Qwen2.5-1.5B (experimental; Groq stays in production) |

## Measured results

| Gate | Result |
|------|--------|
| Router accuracy | 10/10 = 100% (target ≥ 80%) |
| Agent event tests | 5/5 pass with structured output + citations |
| v3 RAG faithfulness | 1.00 (LLM judge), +0.117 vs v1 (target +0.10) |
| Guardrails red-team | 20/20 |
| Multimodal extraction | 3/3 test cases |
| Forecast backtest (7y, 34 folds) | MAPE 11.2%, directional accuracy 62% on 30-session HRC horizon |
| News-impact event study (22 events, 2019–2025) | sign agreement 59%, Pearson r = 0.469 |
| Semantic cache hit rate | 60% (target 20–35%) |

Full details: [eval/benchmark_v1_vs_v2_vs_v3.md](eval/benchmark_v1_vs_v2_vs_v3.md) and [eval/week3_summary.md](eval/week3_summary.md).

## Data sources

| Dataset | Description | Format |
|---------|-------------|--------|
| DGTR notifications | Anti-dumping & safeguard investigation PDFs | PDF → Pinecone |
| Ministry of Steel Annual Reports | Policy context, production statistics | PDF → Pinecone |
| Foreign Trade Policy 2023 | FTP chapters | PDF → Pinecone |
| BIS quality orders | IS 2062, QCO 2024 | PDF → Pinecone |
| TRADESTAT monthly exports | India steel exports by country, 99 months | XLSX |
| WTO WITS MFN tariffs | India applied MFN rates HS 72 & 73, 2010–2023 | CSV (14 files) |
| Live RSS news | 4 feeds, classified and upserted every 4h | RSS → Pinecone |
| Market data | HRC=F, SLX, Tata Steel, SAIL, JSW, MT, NUE | yfinance |
| World Bank GDP | Partner-country GDP for the gravity model | API (cached) |

> Source documents (`Base documents/`) and vector indexes are not committed due to size. Rebuild with `python steel_rag/ingest_pinecone.py` after adding documents.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY + PINECONE_API_KEY
python steel_rag/ingest_pinecone.py
python steel_rag/build_bm25_corpus.py
streamlit run steel_rag/dashboard.py   # http://localhost:8501
```

Optional services:

```bash
python api.py                                  # FastAPI backend on :8000
python steel_rag/classifier_pipeline.py --daemon   # live news pipeline (4h cycle)
```

## Usage

```python
from steel_rag.router import route_query
result = route_query("What anti-dumping duty was imposed on seamless tubes from China?")
print(result.result.answer_text)       # grounded answer with citations
print(result.result.gravity_scenario)  # trade-flow prediction when relevant
```

```python
from steel_rag.steel_futures import analyze_news_impact
impact = analyze_news_impact("US doubles Section 232 steel tariffs to 50%")
print(impact["futures_impact_pct"], impact["trade_flow_impact_pct"])
```

```python
from steel_rag.market_opportunity import rank_market_opportunities
print(rank_market_opportunities(top_n=10))   # under-served export markets, scored
```

## Evaluation & backtests

```bash
python eval/router_accuracy_test.py        # router gate (≥ 80%)
python eval/test_agent_events.py           # 5-event agent gate
python eval/test_news_futures_chain.py     # news→futures→markets chain gate
python eval/event_study_2019.py            # recalibrate news multipliers (22 events)
python eval/backtest_futures.py            # 7-year walk-forward forecast backtest
```

## Tech stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Pinecone serverless (+ FAISS local fallback) |
| Sparse + rerank | rank-bm25, `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq `llama-3.3-70b-versatile` (562ms avg, 133 tok/s measured) |
| Forecasting | Meta Prophet (+ Steel-GPR external regressor) |
| Gravity / ML | statsmodels OLS, XGBoost |
| Fine-tuning | TRL DPO, PEFT LoRA, bitsandbytes 4-bit (RTX 4090) |
| Multimodal | Qwen2-VL-2B-Instruct (4-bit) |
| Validation | Pydantic v2 |
| Observability | Langfuse |
| UI / API | Streamlit, FastAPI |

## Project structure

```
AI_Trade_Capstone/
├── steel_rag/
│   ├── rag.py · router.py · memory.py · guardrails.py · semantic_cache.py
│   ├── data_agent.py · tariff_agent.py · gravity_model.py
│   ├── steel_futures.py · market_opportunity.py
│   ├── classifier.py · classifier_pipeline.py
│   ├── ingest.py · ingest_pinecone.py · build_bm25_corpus.py
│   ├── eval_harness.py · run_eval.py
│   └── dashboard.py
├── eval/            # gate tests, benchmarks, event studies, backtests
├── dpo/             # DPO preference pairs + checkpoint
├── multimodal/      # Qwen2-VL extraction
├── vllm/            # serving setup + throughput benchmark
├── api.py · app.py · policy_brief.md
└── requirements.txt
```

## Honest limitations

- HRC futures are regime-switching: ~11% MAPE / 62% direction on a 30-session horizon is the realistic envelope; regime-break periods hit 25–35% error regardless of config.
- News-impact calibration rests on 22 documented events; per-type factors with n=1 are clipped at runtime.
- The GPR forecast regressor is metric-neutral in backtest — its value is live news-conditioning, not historical accuracy lift.
- Local-only deployment by design (Streamlit + FastAPI on localhost); vLLM serving requires Linux/WSL2.
