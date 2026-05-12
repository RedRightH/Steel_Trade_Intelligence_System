# India Steel Trade Intelligence Platform

A multi-agent RAG system for analysing Indian steel trade policy, export trends, and MFN tariff rates — built as a capstone project for IIFT Personal Interview preparation.

## Architecture

```
Question
   │
   ▼
Router (classify → 7 types)
   ├── ANTI_DUMPING / SAFEGUARD / POLICY_OPPORTUNITY  → PolicyAnalystAgent   → RAG (FAISS + Groq)
   ├── RAW_MATERIAL / CBAM_COMPLIANCE                 → SupplyChainRiskAgent → RAG (FAISS + Groq)
   ├── DATA_ANALYSIS                                  → DataAnalysisAgent    → TRADESTAT XLSX + LLM code-gen
   └── TARIFF_ANALYSIS                                → TariffAnalysisAgent  → WTO WITS MFN CSVs + LLM code-gen
```

## Features

| Module | What it does |
|--------|-------------|
| `rag.py` | FAISS vector store + Groq LLM, Langfuse v4 tracing |
| `ingest.py` | PDF/document ingestion → chunking → FAISS index |
| `router.py` | 7-category classifier, Pydantic output schemas, multi-agent dispatch |
| `data_agent.py` | 26-month TRADESTAT export data, trend analysis, regional breakdown, LLM code-gen charts |
| `tariff_agent.py` | WTO WITS MFN tariff data (HS 72 & 73, 2010-2023), trend charts, heatmaps |
| `dashboard.py` | Streamlit app — 4 tabs: Query · Export Trends · Tariff Lookup · Eval Report |
| `eval/run_eval.py` | Evaluation harness, baseline comparison |

## Data Sources

| Dataset | Description | Format |
|---------|-------------|--------|
| DGTR notifications | Anti-dumping & safeguard investigation PDFs | PDF → FAISS |
| Ministry of Steel Annual Reports | Policy context, production statistics | PDF → FAISS |
| Foreign Trade Policy 2023 | FTP chapters 1-9 | PDF → FAISS |
| TRADESTAT monthly exports | India's steel exports by country, 26 months | XLSX (26 files) |
| WTO WITS MFN tariffs | India's applied MFN tariff rates HS 72 & 73, 2010-2023 | CSV (14 files) |
| BIS product manuals | IS 2062 and other steel standards | PDF → FAISS |

> **Note:** Source documents (`Base documents/`) and the FAISS index (`steel_rag/faiss_index/`) are not included due to size. Rebuild the index with `python steel_rag/ingest.py` after adding your documents.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/india-steel-trade-intelligence.git
cd india-steel-trade-intelligence
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)
```

### 3. Add source documents

Place your PDFs and data files into `Base documents/`:

```
Base documents/
├── Anti-Dumping/          # DGTR AD investigation PDFs
├── Safeguard/             # Safeguard investigation PDFs
├── FTP/                   # Foreign Trade Policy chapters
├── Ministry of Steel/     # Annual Report PDFs
├── India_Steel_exports/   # TRADESTAT monthly XLSX files
└── MFN_India/             # WTO WITS MFN CSV files (one per year)
```

### 4. Build the FAISS index

```bash
python steel_rag/ingest.py
# First run: ~5-10 min depending on corpus size
```

### 5. Launch the dashboard

```bash
streamlit run steel_rag/dashboard.py
# Opens at http://localhost:8501
```

## Usage

### Dashboard tabs

- **Intelligence Query** — ask any question; the router classifies it and dispatches to the right agent
- **Export Trends** — top destinations, growing/shrinking markets, regional breakdown, country comparison
- **Tariff Lookup** — MFN rate by HS code, trend charts, heatmaps, LLM-powered tariff questions
- **Eval Report** — RAG evaluation baseline comparison

### Direct API

```python
from steel_rag.router import route_query

result = route_query("What anti-dumping duty was imposed on seamless tubes from China?")
print(result.result.answer_text)
print(result.question_type)   # "ANTI_DUMPING"
print(result.agent_used)      # "PolicyAnalystAgent"
```

```python
from steel_rag.data_agent import get_market_trends, get_latest_top_destinations

trends = get_market_trends(lookback_months=6, min_avg_usd=2.0, n=10)
print(trends["growing"])

top = get_latest_top_destinations(n=10)
```

```python
from steel_rag.tariff_agent import get_tariff_trend, query_tariff

trend = get_tariff_trend("720810")

result = query_tariff("Which steel products face the highest MFN tariff in India?")
print(result["answer"])
```

## Evaluation

```bash
# Run full eval (10 questions)
python steel_rag/eval/run_eval.py --tag v1b

# Compare two runs
python steel_rag/eval/run_eval.py --compare-only --tag v1b
```

| Run | Corpus | Answered | Source Hit Rate | Avg Latency |
|-----|--------|----------|-----------------|-------------|
| v1  | 3,086 chunks (55 docs) | 7/10 (70%) | - | 3,627ms |
| v1b | 5,625 chunks (expanded) | 7/10 (70%) | 70% | 6,063ms |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (CPU) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Validation | Pydantic v2 |
| Observability | Langfuse v4 |
| Dashboard | Streamlit 1.35+ |
| Data | pandas, numpy, matplotlib |

## Project Structure

```
AI_Trade_Capstone/
├── .env.example
├── requirements.txt
├── README.md
├── steel_rag/
│   ├── ingest.py             # Document ingestion → FAISS
│   ├── rag.py                # Core RAG query function
│   ├── router.py             # Multi-agent router + Pydantic schemas
│   ├── data_agent.py         # Export trend analysis (TRADESTAT XLSX)
│   ├── tariff_agent.py       # MFN tariff analysis (WTO WITS CSV)
│   ├── dashboard.py          # Streamlit dashboard
│   └── eval/
│       ├── run_eval.py       # Evaluation harness
│       ├── ground_truth.json
│       ├── baseline_v1.json
│       └── baseline_v1b.json
```

## Roadmap

- [ ] Live RSS auto-classification pipeline into FAISS
- [ ] Expand eval set from 10 to 25 questions
- [ ] Deploy dashboard to Streamlit Cloud
