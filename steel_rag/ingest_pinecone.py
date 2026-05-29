"""
ingest_pinecone.py — Embed all PDFs and upsert into a Pinecone serverless index.

Run this ONCE locally after signing up for Pinecone free tier:
    pip install pinecone langchain-pinecone
    python steel_rag/ingest_pinecone.py

Requirements:
  - PINECONE_API_KEY in .env  (or env var)
  - PINECONE_INDEX_NAME in .env  (default: "steel-rag")
  - Pinecone free tier: 100k vectors, us-east-1, serverless

The script is idempotent: rerunning it re-upserts the same vectors (no duplication
because Pinecone upsert is an upsert by vector ID).
"""

import os
import sys
import time
import glob
import pickle
import hashlib
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DOCS_DIR      = Path(__file__).parent.parent / "Base documents"
BM25_CORPUS_PATH   = Path(__file__).parent / "bm25_corpus.pkl"
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM          = 384          # all-MiniLM-L6-v2 output dimension
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 100
BATCH_SIZE         = 100          # vectors per upsert batch (Pinecone limit ~1000)
PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME", "steel-rag")
PINECONE_REGION    = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_CLOUD     = os.getenv("PINECONE_CLOUD", "aws")
# ─────────────────────────────────────────────────────────────────────────────


def _chunk_id(chunk_text: str, meta: dict) -> str:
    """Stable ID for deduplication: hash of file+page+content."""
    raw = f"{meta.get('file_name','')}|{meta.get('page','')}|{chunk_text[:200]}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_pdf_markitdown(path: str) -> list:
    """Load a PDF via MarkItDown (better table/layout extraction) → list of Document."""
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(path)
        text = result.text_content or ""
        if not text.strip():
            return []
        # MarkItDown returns full-doc text; wrap as single Document
        return [Document(
            page_content=text,
            metadata={
                "category":  Path(path).parent.name,
                "file_name": Path(path).name,
                "page":      0,
            }
        )]
    except Exception as e:
        raise RuntimeError(f"MarkItDown failed: {e}") from e


def load_and_chunk() -> list:
    pdf_paths = list(set(
        glob.glob(str(BASE_DOCS_DIR / "**" / "*.pdf"), recursive=True)
        + glob.glob(str(BASE_DOCS_DIR / "*.pdf"))
    ))
    print(f"Found {len(pdf_paths)} PDFs")

    docs, failed = [], []
    for path in sorted(pdf_paths):
        rel = os.path.relpath(path, BASE_DOCS_DIR)
        try:
            # ── Primary: MarkItDown (preserves tables as markdown) ────────────
            pages = _load_pdf_markitdown(path)
            loader = "markitdown"
        except Exception as md_err:
            # ── Fallback: PyPDFLoader ─────────────────────────────────────────
            try:
                pages = PyPDFLoader(path).load()
                for p in pages:
                    p.metadata["category"]  = Path(path).parent.name
                    p.metadata["file_name"] = Path(path).name
                loader = "pypdf"
            except Exception as e:
                failed.append(rel)
                print(f"  [FAIL] {rel} — {e}")
                continue

        docs.extend(pages)
        print(f"  [OK/{loader}] {rel}  ({len(pages)} doc(s))")

    if failed:
        print(f"\nFailed to load: {failed}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"\n{len(docs)} documents → {len(chunks)} chunks")
    return chunks


def embed_chunks(chunks: list) -> tuple[list, list[str]]:
    """Return (embeddings_list, ids_list)."""
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    texts = [c.page_content for c in chunks]
    print(f"Embedding {len(texts)} chunks …")
    t0 = time.time()
    embeddings = model.embed_documents(texts)
    print(f"Embedding done in {time.time()-t0:.1f}s")
    ids = [_chunk_id(c.page_content, c.metadata) for c in chunks]
    return embeddings, ids


def upsert_to_pinecone(chunks, embeddings, ids):
    from pinecone import Pinecone, ServerlessSpec

    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in .env")

    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        print(f"\nCreating Pinecone index '{PINECONE_INDEX}' …")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            print("  … waiting for index to be ready")
            time.sleep(2)
        print(f"Index '{PINECONE_INDEX}' created and ready.")
    else:
        print(f"\nUsing existing Pinecone index '{PINECONE_INDEX}'")

    index = pc.Index(PINECONE_INDEX)

    # Build vector records
    records = []
    for i, (chunk, emb, vid) in enumerate(zip(chunks, embeddings, ids)):
        meta = {
            "text":      chunk.page_content[:1000],   # Pinecone metadata value limit
            "file_name": chunk.metadata.get("file_name", ""),
            "category":  chunk.metadata.get("category", ""),
            "page":      str(chunk.metadata.get("page", "")),
        }
        records.append({"id": vid, "values": emb, "metadata": meta})

    # Upsert in batches
    total = len(records)
    print(f"\nUpserting {total} vectors to Pinecone in batches of {BATCH_SIZE} …")
    t0 = time.time()
    for start in range(0, total, BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"  Upserted {min(start+BATCH_SIZE, total)}/{total}")

    elapsed = time.time() - t0
    stats = index.describe_index_stats()
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Pinecone index stats: {stats.total_vector_count} vectors")


def save_bm25_corpus(chunks: list):
    """Save chunk texts to bm25_corpus.pkl for hybrid search in rag.py."""
    corpus = [
        {
            "id":        _chunk_id(c.page_content, c.metadata),
            "text":      c.page_content[:1000],
            "file_name": c.metadata.get("file_name", "Unknown"),
            "category":  c.metadata.get("category", ""),
            "page":      str(c.metadata.get("page", "?")),
        }
        for c in chunks
    ]
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(corpus, f)
    size_mb = BM25_CORPUS_PATH.stat().st_size / 1024 / 1024
    print(f"BM25 corpus saved: {len(corpus)} chunks → {BM25_CORPUS_PATH} ({size_mb:.1f} MB)")


def main():
    print("=" * 60)
    print("STEEL RAG — Pinecone Ingestion Pipeline")
    print("=" * 60)

    t_start = time.time()
    chunks = load_and_chunk()
    if not chunks:
        raise RuntimeError(f"No documents loaded from {BASE_DOCS_DIR}")

    embeddings, ids = embed_chunks(chunks)
    upsert_to_pinecone(chunks, embeddings, ids)
    save_bm25_corpus(chunks)   # keep BM25 corpus in sync

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print("=" * 60)
    print("Re-run anytime to upsert new or updated documents.")
    print("=" * 60)


if __name__ == "__main__":
    main()
