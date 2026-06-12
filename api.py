"""
api.py — FastAPI backend for the India Steel Trade Intelligence Platform.

Routes:
  GET  /health                  → system health check
  POST /query                   → main RAG query (uses semantic cache)
  POST /query/impact            → news announcement impact analysis
  GET  /cache/stats             → semantic cache statistics
  POST /cache/clear             → clear the cache (admin)
  GET  /gravity/insights        → gravity model coefficients + metrics
  POST /gravity/scenario        → predict trade flow for a scenario

Start locally:
  cd C:\\Users\\suchi\\AIMasterClass\\AI_Trade_Capstone
  uvicorn api:app --reload --port 8000

Vercel deployment:
  Add vercel.json (see below). All LLM calls go through Groq — no GPU needed.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Add steel_rag to path
sys.path.insert(0, str(Path(__file__).parent / "steel_rag"))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="India Steel Trade Intelligence Platform",
    description="RAG-powered steel trade policy analyst — grounded in DGTR, WTO, BIS, and Ministry of Steel documents.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_warmup():
    """Pre-load embedder, Pinecone, BM25, reranker at boot — not on first query."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    def _warm():
        try:
            from rag import warmup
            warmup()
        except Exception as e:
            print(f"[startup] Warmup failed (non-fatal): {e}")
    await loop.run_in_executor(ThreadPoolExecutor(max_workers=1), _warm)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    use_cache: bool = True

class QueryResponse(BaseModel):
    question:       str
    answer:         str
    question_type:  str
    agent_used:     str
    sources:        list[dict]
    cache_hit:      bool
    latency_ms:     int

class ImpactRequest(BaseModel):
    announcement:   str
    use_rag:        bool = True

class GravityScenarioRequest(BaseModel):
    country:            str
    gdp_growth_pct:     float = 0.0
    tariff_change_pct:  float = 0.0
    model_type:         str   = "xgb"


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version":   "2.0.0",
        "services": {
            "groq":     bool(os.getenv("GROQ_API_KEY")),
            "pinecone": bool(os.getenv("PINECONE_API_KEY")),
        }
    }


# ── Main RAG query ─────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Route a steel trade question through the multi-agent RAG pipeline.
    Returns a structured answer grounded in retrieved documents.
    Uses semantic cache by default (cosine ≥ 0.92, 24h TTL).
    """
    t0 = time.time()
    try:
        if req.use_cache:
            from semantic_cache import cached_rag_query
            raw = cached_rag_query(req.question)
            cache_hit = raw.get("cache_hit", False)
            # Wrap in router-style output
            answer  = raw.get("answer", "")
            sources = raw.get("sources", [])
            qtype   = "RAG"
            agent   = "rag"
        else:
            from router import route_query
            ro      = route_query(req.question)
            answer  = ro.result.answer_text
            sources = [{"file_name": c.get("file_name", ""), "text": c.get("text", "")[:200]}
                       for c in (ro.result.source_citations or [])]
            qtype   = ro.question_type
            agent   = ro.agent_used
            cache_hit = False

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=req.question,
        answer=answer,
        question_type=qtype,
        agent_used=agent,
        sources=sources,
        cache_hit=cache_hit,
        latency_ms=int((time.time() - t0) * 1000),
    )


# ── News impact ───────────────────────────────────────────────────────────────

@app.post("/query/impact")
def query_impact(req: ImpactRequest):
    """
    Analyse a steel trade announcement through the 3-layer AI-GPR pipeline.
    Returns risk score, event classification, persistence, India spillover,
    and projected futures + trade flow impact.
    """
    try:
        if req.use_rag:
            from steel_futures import analyze_news_impact_with_rag
            result = analyze_news_impact_with_rag(req.announcement)
        else:
            from steel_futures import analyze_news_impact
            result = analyze_news_impact(req.announcement)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Cache management ──────────────────────────────────────────────────────────

@app.get("/cache/stats")
def cache_stats():
    from semantic_cache import get_cache_stats
    return get_cache_stats()

@app.post("/cache/clear")
def cache_clear():
    from semantic_cache import clear_cache
    clear_cache()
    return {"status": "cleared"}


# ── Gravity model ─────────────────────────────────────────────────────────────

@app.get("/gravity/insights")
def gravity_insights():
    """Return gravity model coefficients, R², and feature importances."""
    try:
        from gravity_model import get_gravity_insights
        ins = get_gravity_insights()
        # Convert DataFrames to JSON-serialisable dicts
        ins["coef_df"]     = ins["coef_df"].to_dict(orient="records")
        ins["feature_imp"] = ins["feature_imp"]
        ins["fy_range"]    = list(ins["fy_range"])
        return ins
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gravity/scenario")
def gravity_scenario(req: GravityScenarioRequest):
    """Predict trade flow change for a country under a given scenario."""
    try:
        from gravity_model import predict_trade_flow
        result = predict_trade_flow(
            req.country,
            gdp_growth_pct=req.gdp_growth_pct,
            tariff_change_pct=req.tariff_change_pct,
            model_type=req.model_type,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run locally ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
