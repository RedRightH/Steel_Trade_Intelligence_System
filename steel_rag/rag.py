"""
rag.py - Core RAG query function with hybrid vector store support.

Retrieval pipeline (in order):
  1. Dense search   — Pinecone (cloud) or FAISS (local), cosine similarity
  2. BM25 search    — sparse keyword search over bm25_corpus.pkl (if present)
  3. RRF merge      — Reciprocal Rank Fusion over dense + BM25 candidates
  4. BGE reranker   — cross-encoder/ms-marco-MiniLM-L-6-v2 (if available)

Vector store priority:
  1. Pinecone  — when PINECONE_API_KEY is set (Streamlit Cloud / production)
  2. FAISS     — local fallback for development

Uses native Pinecone SDK (v5+) — no langchain-pinecone dependency needed.
Langfuse tracing is enabled when LANGFUSE_PUBLIC_KEY is configured.

Usage:
    from rag import rag_query
    result = rag_query("What are India's anti-dumping duties on seamless tubes?")
"""

import os
import sys
import time
import pickle
from pathlib import Path
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load .env FIRST — must happen before Langfuse reads env vars
load_dotenv(Path(__file__).parent.parent / ".env")

# Pull secrets from Streamlit when running inside Streamlit Cloud
try:
    import streamlit as st
    for _k in ("GROQ_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
               "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass  # Not running inside Streamlit

from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
FAISS_INDEX_DIR  = Path(__file__).parent / "faiss_index"
BM25_CORPUS_PATH = Path(__file__).parent / "bm25_corpus.pkl"
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL       = "llama-3.3-70b-versatile"
TOP_K            = 3        # final results returned to LLM
HYBRID_FETCH_K   = 10       # candidates fetched from each source before reranking
PINECONE_INDEX   = os.getenv("PINECONE_INDEX_NAME", "steel-rag")

# Auto-detect which backend to use
_USE_PINECONE = bool(os.getenv("PINECONE_API_KEY", "").strip())

SYSTEM_PROMPT = (
    "You are a steel trade policy analyst specializing in Indian trade law, "
    "anti-dumping investigations, safeguard measures, and foreign trade policy. "
    "Answer questions using ONLY the context provided below. "
    "If the answer is not in the context, say exactly: "
    "'I cannot answer this from the provided documents.' "
    "Always cite the source document name and relevant passage in your answer. "
    "Be specific -- include duty rates, dates, product descriptions, and country names "
    "wherever the context supports it."
)

_LANGFUSE_ENABLED = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    and not os.getenv("LANGFUSE_PUBLIC_KEY", "").startswith("your_")
)
# ─────────────────────────────────────────────────────────────────────────────

_embeddings   = None
_vectorstore  = None
_groq_client  = None
_bm25_index   = None   # rank_bm25.BM25Okapi instance
_bm25_docs    = None   # list of corpus dicts {text, file_name, category, page}
_reranker     = None   # sentence_transformers.CrossEncoder instance


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── Pinecone adapter — same interface as FAISS ────────────────────────────────

class _PineconeStore:
    """
    Thin wrapper around Pinecone v5+ SDK that exposes the same
    similarity_search_with_score() interface as LangChain FAISS,
    so the rest of rag.py works without any changes.
    """
    def __init__(self, index):
        self._index = index

    def similarity_search_with_score(self, query: str, k: int = TOP_K):
        emb_model = _get_embeddings()
        query_vec = emb_model.embed_query(query)

        resp = self._index.query(
            vector=query_vec,
            top_k=k,
            include_metadata=True,
        )

        results = []
        for match in resp.matches:
            meta = match.metadata or {}
            text = meta.get("text", "")
            # Build a minimal LangChain-compatible Document
            from langchain_core.documents import Document
            doc = Document(
                page_content=text,
                metadata={
                    "file_name": meta.get("file_name", "Unknown"),
                    "category":  meta.get("category", ""),
                    "page":      meta.get("page", "?"),
                },
            )
            # Pinecone returns cosine similarity (0–1); convert to a "score"
            score = float(match.score)
            results.append((doc, score))

        return results


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if _USE_PINECONE:
        _vectorstore = _load_pinecone()
    else:
        _vectorstore = _load_faiss()

    return _vectorstore


def _load_pinecone():
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set")

    pc    = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX)
    print(f"[rag] Connected to Pinecone index '{PINECONE_INDEX}'")
    return _PineconeStore(index)


def _load_faiss():
    from langchain_community.vectorstores import FAISS

    if not FAISS_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_DIR}. "
            "Run ingest.py first, or set PINECONE_API_KEY for cloud mode."
        )
    vs = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    print(f"[rag] Loaded FAISS index from {FAISS_INDEX_DIR}")
    return vs


def _get_groq():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env or Streamlit secrets")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── BM25 sparse retrieval ─────────────────────────────────────────────────────

def _get_bm25():
    """Load BM25 index from bm25_corpus.pkl (lazy, cached). Returns (None, None) if missing."""
    global _bm25_index, _bm25_docs
    if _bm25_index is not None:
        return _bm25_index, _bm25_docs
    if not BM25_CORPUS_PATH.exists():
        return None, None
    try:
        from rank_bm25 import BM25Okapi
        with open(BM25_CORPUS_PATH, "rb") as f:
            corpus = pickle.load(f)
        _bm25_docs = corpus
        tokenized  = [doc["text"].lower().split() for doc in corpus]
        _bm25_index = BM25Okapi(tokenized)
        print(f"[rag] BM25 index loaded: {len(corpus)} chunks")
    except Exception as e:
        print(f"[rag] BM25 load failed ({e}), falling back to dense-only")
        return None, None
    return _bm25_index, _bm25_docs


def _bm25_search(query: str, k: int) -> list:
    """Return top-k (Document, bm25_score) pairs. Empty list if corpus unavailable."""
    from langchain_core.documents import Document
    bm25, docs = _get_bm25()
    if bm25 is None:
        return []
    tokens = query.lower().split()
    scores  = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = []
    for idx in top_idx:
        d = docs[idx]
        doc = Document(
            page_content=d["text"],
            metadata={
                "file_name": d.get("file_name", "Unknown"),
                "category":  d.get("category", ""),
                "page":      d.get("page", "?"),
            },
        )
        results.append((doc, float(scores[idx])))
    return results


# ── RRF merge ─────────────────────────────────────────────────────────────────

def _rrf_merge(dense: list, sparse: list, k: int = 60, top_n: int = TOP_K) -> list:
    """
    Reciprocal Rank Fusion — combine dense and sparse ranked lists.
    Each result's RRF score = Σ 1/(k + rank).
    Deduplication is by first 200 chars of page_content.
    """
    scores: dict  = {}
    docs_map: dict = {}

    for rank, (doc, _) in enumerate(dense):
        key = doc.page_content[:200]
        scores[key]   = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        docs_map[key] = doc

    for rank, (doc, _) in enumerate(sparse):
        key = doc.page_content[:200]
        scores[key]   = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in docs_map:
            docs_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]
    return [(docs_map[key], scores[key]) for key in sorted_keys]


# ── BGE cross-encoder reranker ────────────────────────────────────────────────

def _get_reranker():
    """Load cross-encoder reranker (lazy, cached). Returns None on failure."""
    global _reranker
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
        print(f"[rag] Reranker loaded: {RERANKER_MODEL}")
    except Exception as e:
        print(f"[rag] Reranker load failed ({e}), skipping rerank step")
        _reranker = False   # sentinel so we don't retry
    return _reranker if _reranker else None


def _rerank(query: str, candidates: list, top_n: int = TOP_K) -> list:
    """Rerank candidates with cross-encoder. Falls back to input order on error."""
    reranker = _get_reranker()
    if reranker is None or not candidates:
        return candidates[:top_n]
    try:
        pairs  = [(query, doc.page_content) for doc, _ in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(item[0], float(s)) for item, s in ranked[:top_n]]
    except Exception as e:
        print(f"[rag] Rerank failed ({e}), using RRF order")
        return candidates[:top_n]


def _build_sources(raw_results):
    sources, context_parts = [], []
    for i, (doc, score) in enumerate(raw_results):
        file_name = doc.metadata.get("file_name", "Unknown")
        category  = doc.metadata.get("category", "")
        page      = doc.metadata.get("page", "?")
        excerpt   = doc.page_content.strip()
        context_parts.append(f"[Source {i+1}] {file_name} (page {page})\n{excerpt}")
        sources.append({
            "file_name":        file_name,
            "category":         category,
            "page":             page,
            "excerpt":          excerpt[:200],
            "similarity_score": round(float(score), 4),
        })
    return sources, context_parts


# ── Core query ────────────────────────────────────────────────────────────────

def rag_query(question: str, top_k: int = TOP_K) -> dict:
    """
    Query the RAG system. Uses Pinecone or FAISS based on env config.
    Traces to Langfuse when keys are configured.

    Returns:
        question, answer, sources, context_used, latency, token_counts, vector_backend
    """
    if _LANGFUSE_ENABLED:
        return _rag_query_traced(question, top_k)
    return _rag_query_core(question, top_k)


def _rag_query_core(question: str, top_k: int) -> dict:
    t_total = time.time()

    # ── Retrieval ──────────────────────────────────────────────────────────────
    t_ret      = time.time()
    fetch_k    = max(HYBRID_FETCH_K, top_k * 3)  # wider net for reranking

    # 1. Dense (Pinecone / FAISS)
    dense = _get_vectorstore().similarity_search_with_score(question, k=fetch_k)

    # 2. BM25 sparse (if corpus available)
    sparse = _bm25_search(question, k=fetch_k)

    # 3. Merge
    if sparse:
        merged = _rrf_merge(dense, sparse, top_n=fetch_k)
        retrieval_method = "hybrid_rrf"
    else:
        merged = dense[:fetch_k]
        retrieval_method = "dense_only"

    # 4. BGE reranker
    raw = _rerank(question, merged, top_n=top_k)
    if sparse:
        retrieval_method += "+rerank"

    chunk_retrieval_ms = int((time.time() - t_ret) * 1000)

    sources, context_parts = _build_sources(raw)
    context  = "\n\n---\n\n".join(context_parts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    t_llm    = time.time()
    response = _get_groq().chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.1, max_tokens=1024,
    )
    llm_call_ms = int((time.time() - t_llm) * 1000)
    answer      = response.choices[0].message.content.strip()

    return {
        "question": question, "answer": answer,
        "sources": sources, "context_used": context,
        "latency": {
            "chunk_retrieval_ms": chunk_retrieval_ms,
            "llm_call_ms":        llm_call_ms,
            "total_ms":           int((time.time() - t_total) * 1000),
        },
        "token_counts": {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        },
        "vector_backend":    "pinecone" if _USE_PINECONE else "faiss",
        "retrieval_method":  retrieval_method,
    }


def _rag_query_traced(question: str, top_k: int) -> dict:
    """RAG query with full Langfuse v4 trace tree."""
    from langfuse import Langfuse, observe

    lf = Langfuse()

    @observe(name="rag_query")
    def _inner(q, k):
        t_total = time.time()
        lf.set_current_trace_io(input={"question": q, "top_k": k})

        t_ret   = time.time()
        fetch_k = max(HYBRID_FETCH_K, k * 3)
        with lf.start_as_current_observation(name="retrieval", as_type="retriever"):
            dense  = _get_vectorstore().similarity_search_with_score(q, k=fetch_k)
            sparse = _bm25_search(q, k=fetch_k)
            if sparse:
                merged           = _rrf_merge(dense, sparse, top_n=fetch_k)
                retrieval_method = "hybrid_rrf+rerank"
            else:
                merged           = dense[:fetch_k]
                retrieval_method = "dense_only"
            raw = _rerank(q, merged, top_n=k)
            chunk_retrieval_ms = int((time.time() - t_ret) * 1000)
            sources, context_parts = _build_sources(raw)
            lf.update_current_span(
                input={"question": q, "top_k": k},
                output={"chunks": [
                    {"rank": i + 1, "file": s["file_name"],
                     "page": s["page"], "score": s["similarity_score"],
                     "excerpt": s["excerpt"][:120]}
                    for i, s in enumerate(sources)
                ]},
                metadata={"chunk_retrieval_ms": chunk_retrieval_ms,
                          "backend": "pinecone" if _USE_PINECONE else "faiss",
                          "retrieval_method": retrieval_method},
            )

        context  = "\n\n---\n\n".join(context_parts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {q}"},
        ]

        t_llm = time.time()
        with lf.start_as_current_observation(name="llm_call", as_type="generation"):
            response = _get_groq().chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                temperature=0.1, max_tokens=1024,
            )
            llm_call_ms = int((time.time() - t_llm) * 1000)
            answer      = response.choices[0].message.content.strip()
            pt, ct, tt  = (response.usage.prompt_tokens,
                           response.usage.completion_tokens,
                           response.usage.total_tokens)
            lf.update_current_generation(
                model=GROQ_MODEL, input=messages, output=answer,
                usage_details={"input": pt, "output": ct, "total": tt},
                metadata={"llm_call_ms": llm_call_ms, "temperature": 0.1},
            )

        total_ms = int((time.time() - t_total) * 1000)
        lf.set_current_trace_io(output={
            "answer":       answer[:300],
            "answered":     "cannot answer" not in answer.lower(),
            "sources_used": [s["file_name"] for s in sources],
        })
        lf.update_current_span(metadata={
            "chunk_retrieval_ms": chunk_retrieval_ms,
            "llm_call_ms": llm_call_ms, "total_ms": total_ms,
            "prompt_tokens": pt, "completion_tokens": ct,
            "embed_model": EMBED_MODEL, "groq_model": GROQ_MODEL,
            "vector_backend": "pinecone" if _USE_PINECONE else "faiss",
            "retrieval_method": retrieval_method,
        })

        return {
            "question": q, "answer": answer,
            "sources": sources, "context_used": context,
            "latency": {
                "chunk_retrieval_ms": chunk_retrieval_ms,
                "llm_call_ms":        llm_call_ms,
                "total_ms":           total_ms,
            },
            "token_counts": {
                "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt,
            },
            "vector_backend":   "pinecone" if _USE_PINECONE else "faiss",
            "retrieval_method": retrieval_method,
        }

    result = _inner(question, top_k)
    lf.flush()
    return result


def _print_result(result: dict):
    backend  = result.get("vector_backend", "?")
    ret_mode = result.get("retrieval_method", "?")
    print("\n" + "=" * 60)
    print(f"Q: {result['question']}")
    print(f"Backend: {backend}  |  Retrieval: {ret_mode}")
    print("=" * 60)
    print(f"\nA: {result['answer']}")
    lat = result.get("latency", {})
    tok = result.get("token_counts", {})
    print(f"\n[retrieval={lat.get('chunk_retrieval_ms')}ms  "
          f"llm={lat.get('llm_call_ms')}ms  "
          f"total={lat.get('total_ms')}ms  "
          f"tokens={tok.get('total_tokens')}]")
    print("\n--- Sources ---")
    for s in result["sources"]:
        print(f"  {s['file_name']} (page {s['page']}, score={s['similarity_score']:.3f})")
        print(f"  {s['excerpt'][:100]}...")


if __name__ == "__main__":
    backend = "Pinecone" if _USE_PINECONE else "FAISS"
    print(f"Steel RAG — Vector backend: {backend}")
    print(f"Langfuse: {'enabled' if _LANGFUSE_ENABLED else 'disabled'}")
    print("Loading index and model (first run ~30s)...\n")

    test_questions = [
        "What anti-dumping duty was imposed on seamless tubes from China?",
        "Which countries are subject to anti-dumping on electrogalvanized steel?",
    ]
    for q in test_questions:
        result = rag_query(q)
        _print_result(result)
        print()
