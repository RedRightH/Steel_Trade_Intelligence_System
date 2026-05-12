"""
rag.py - Core RAG query function with hybrid vector store support.

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
FAISS_INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
TOP_K           = 3
PINECONE_INDEX  = os.getenv("PINECONE_INDEX_NAME", "steel-rag")

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

    t_ret = time.time()
    raw   = _get_vectorstore().similarity_search_with_score(question, k=top_k)
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
        "vector_backend": "pinecone" if _USE_PINECONE else "faiss",
    }


def _rag_query_traced(question: str, top_k: int) -> dict:
    """RAG query with full Langfuse v4 trace tree."""
    from langfuse import Langfuse, observe

    lf = Langfuse()

    @observe(name="rag_query")
    def _inner(q, k):
        t_total = time.time()
        lf.set_current_trace_io(input={"question": q, "top_k": k})

        t_ret = time.time()
        with lf.start_as_current_observation(name="retrieval", as_type="retriever"):
            raw = _get_vectorstore().similarity_search_with_score(q, k=k)
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
                          "backend": "pinecone" if _USE_PINECONE else "faiss"},
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
            "vector_backend": "pinecone" if _USE_PINECONE else "faiss",
        }

    result = _inner(question, top_k)
    lf.flush()
    return result


def _print_result(result: dict):
    backend = result.get("vector_backend", "?")
    print("\n" + "=" * 60)
    print(f"Q: {result['question']}")
    print(f"Backend: {backend}")
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
