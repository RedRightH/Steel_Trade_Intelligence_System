"""
semantic_cache.py — GPTCache-backed semantic caching for the Steel RAG pipeline.

Cache key  = MiniLM embedding of the query (384-dim cosine)
Hit threshold = cosine similarity ≥ 0.92
Storage    = SQLite (gptcache.db) in the steel_rag/ directory
TTL        = 24 hours (queries answered in the last day are cached)

Usage:
    from semantic_cache import cached_rag_query, get_cache_stats, clear_cache
    result = cached_rag_query("What are India's AD duties on HR coil?")

Or as a drop-in replacement for rag_query:
    from semantic_cache import cached_rag_query as rag_query
"""

import os
import time
import json
import hashlib
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DB    = Path(__file__).parent / "gptcache.db"
SIMILARITY_THRESHOLD = 0.92
CACHE_TTL_HOURS      = 24
EMBED_MODEL          = "sentence-transformers/all-MiniLM-L6-v2"

_embedder = None
_cache_hits   = 0
_cache_misses = 0


# ── Embedding ─────────────────────────────────────────────────────────────────

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _embed(text: str) -> list[float]:
    model = _get_embedder()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    # Vectors are L2-normalised so dot product = cosine similarity
    return min(1.0, max(-1.0, dot))


# ── SQLite cache store ─────────────────────────────────────────────────────────

def _init_db():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text  TEXT    NOT NULL,
            embedding   BLOB    NOT NULL,
            result_json TEXT    NOT NULL,
            created_at  REAL    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _load_entries() -> list[dict]:
    """Load all non-expired cache entries."""
    if not CACHE_DB.exists():
        return []
    cutoff = time.time() - CACHE_TTL_HOURS * 3600
    conn = sqlite3.connect(CACHE_DB)
    rows = conn.execute(
        "SELECT id, query_text, embedding, result_json, created_at FROM cache WHERE created_at > ?",
        (cutoff,)
    ).fetchall()
    conn.close()
    entries = []
    for row_id, query_text, emb_blob, result_json, created_at in rows:
        entries.append({
            "id":          row_id,
            "query_text":  query_text,
            "embedding":   pickle.loads(emb_blob),
            "result_json": result_json,
            "created_at":  created_at,
        })
    return entries


def _save_entry(query_text: str, embedding: list[float], result: dict):
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        "INSERT INTO cache (query_text, embedding, result_json, created_at) VALUES (?, ?, ?, ?)",
        (query_text, pickle.dumps(embedding), json.dumps(result, default=str), time.time())
    )
    conn.commit()
    conn.close()


def _purge_expired():
    """Delete entries older than TTL."""
    if not CACHE_DB.exists():
        return 0
    cutoff = time.time() - CACHE_TTL_HOURS * 3600
    conn = sqlite3.connect(CACHE_DB)
    deleted = conn.execute("DELETE FROM cache WHERE created_at <= ?", (cutoff,)).rowcount
    conn.commit()
    conn.close()
    return deleted


# ── Main cached query ─────────────────────────────────────────────────────────

def cached_rag_query(question: str) -> dict:
    """
    Semantic cache wrapper around rag_query().

    If a semantically similar question (cosine ≥ 0.92) was answered in the last
    24 hours, return the cached result. Otherwise, call rag_query(), cache the
    result, and return it.

    The returned dict has an extra field:
        cache_hit  bool   True if served from cache
        cache_sim  float  similarity to matched query (1.0 if exact, None if miss)
    """
    global _cache_hits, _cache_misses

    _init_db()
    q_embedding = _embed(question)
    entries = _load_entries()

    # Find best match above threshold
    best_sim  = 0.0
    best_entry = None
    for entry in entries:
        sim = _cosine(q_embedding, entry["embedding"])
        if sim > best_sim:
            best_sim  = sim
            best_entry = entry

    if best_sim >= SIMILARITY_THRESHOLD and best_entry is not None:
        _cache_hits += 1
        result = json.loads(best_entry["result_json"])
        result["cache_hit"] = True
        result["cache_sim"] = round(best_sim, 4)
        result["cache_matched_query"] = best_entry["query_text"]
        return result

    # Cache miss — call real RAG
    _cache_misses += 1
    from rag import rag_query
    t0 = time.time()
    result = rag_query(question)
    result["latency_ms"] = int((time.time() - t0) * 1000)
    result["cache_hit"]  = False
    result["cache_sim"]  = None

    # Store in cache
    _save_entry(question, q_embedding, result)
    return result


# ── Stats & management ────────────────────────────────────────────────────────

def get_cache_stats() -> dict:
    """Return hit/miss counts and DB size."""
    _init_db()
    entries = _load_entries()
    total_entries = entries.__len__()
    total_requests = _cache_hits + _cache_misses
    db_size_kb = round(CACHE_DB.stat().st_size / 1024, 1) if CACHE_DB.exists() else 0

    return {
        "cache_hits":       _cache_hits,
        "cache_misses":     _cache_misses,
        "hit_rate":         round(_cache_hits / total_requests, 3) if total_requests else 0.0,
        "active_entries":   total_entries,
        "ttl_hours":        CACHE_TTL_HOURS,
        "threshold":        SIMILARITY_THRESHOLD,
        "db_size_kb":       db_size_kb,
    }


def clear_cache():
    """Delete all cache entries."""
    if CACHE_DB.exists():
        CACHE_DB.unlink()
    print("[cache] Cleared all entries.")


def warm_cache(questions: list[str]):
    """Pre-populate cache with a set of questions (e.g., FAQ)."""
    print(f"[cache] Warming with {len(questions)} questions …")
    for i, q in enumerate(questions, 1):
        result = cached_rag_query(q)
        hit = "HIT" if result["cache_hit"] else "MISS"
        print(f"  [{i}/{len(questions)}] {hit}: {q[:60]}")


# ── Benchmark: measure hit rate on test set ───────────────────────────────────

def benchmark_cache(test_queries: list[str] | None = None) -> dict:
    """
    Runs test_queries through the cache (warm + repeat pass) and measures hit rate.
    If no queries provided, uses a built-in 20-query steel trade FAQ set.
    """
    if test_queries is None:
        test_queries = [
            "What are India's anti-dumping duties on Chinese HR coil?",
            "Which countries are subject to AD investigation on electrogalvanized steel?",
            "What is the safeguard duty on steel flat products?",
            "How does EU CBAM affect Indian steel exporters?",
            "What is IS 2062 standard for structural steel?",
            "What are the products covered under the seamless tube AD investigation?",
            "What triggers a safeguard investigation in India?",
            "What is PCN in anti-dumping investigations?",
            "What is India's PLI scheme for specialty steel?",
            "How does India-UAE CEPA benefit steel exports?",
            # Paraphrases — should hit cache
            "India anti-dumping duty on hot rolled coil from China",
            "Countries under anti-dumping investigation for galvanised steel India",
            "safeguard measure flat steel products India",
            "CBAM carbon border adjustment Indian steel",
            "IS2062 BIS structural steel standard",
            "seamless pipe tube anti-dumping China India",
            "what initiates safeguard probe India",
            "product control number dumping investigation",
            "PLI specialty steel India scheme details",
            "UAE CEPA India steel export benefit",
        ]

    # Pass 1: warm the cache
    print("[cache benchmark] Pass 1 — warming cache …")
    for q in test_queries[:10]:
        cached_rag_query(q)

    # Pass 2: run all queries
    print("[cache benchmark] Pass 2 — measuring hit rate …")
    hits = 0
    for q in test_queries:
        r = cached_rag_query(q)
        if r["cache_hit"]:
            hits += 1

    stats = get_cache_stats()
    hit_rate = hits / len(test_queries)
    print(f"\nCache hit rate: {hits}/{len(test_queries)} = {hit_rate:.0%}")
    print(f"Threshold: {SIMILARITY_THRESHOLD}  |  DB size: {stats['db_size_kb']} KB")
    return {"hit_rate": hit_rate, "hits": hits, "total": len(test_queries), **stats}


if __name__ == "__main__":
    import sys
    if "--benchmark" in sys.argv:
        benchmark_cache()
    elif "--clear" in sys.argv:
        clear_cache()
    elif "--stats" in sys.argv:
        print(json.dumps(get_cache_stats(), indent=2))
    else:
        # Quick test
        q = "What are India's anti-dumping duties on HR coil from China?"
        print(f"Query: {q}")
        r = cached_rag_query(q)
        print(f"Cache hit: {r['cache_hit']}  |  Sim: {r['cache_sim']}")
        print(f"Answer: {r['answer'][:200]}")
        print(f"\nStats: {get_cache_stats()}")
