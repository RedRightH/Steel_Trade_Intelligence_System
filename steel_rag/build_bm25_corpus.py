"""
build_bm25_corpus.py — Fetch all chunks from Pinecone and save a local BM25 corpus.

Run ONCE after ingest_pinecone.py to enable hybrid search:
    python steel_rag/build_bm25_corpus.py

Output: steel_rag/bm25_corpus.pkl
  - List of {id, text, file_name, category, page} dicts
  - Loaded by rag.py at startup for BM25 sparse retrieval

Commit bm25_corpus.pkl to git so Streamlit Cloud can also use hybrid search.
Re-run whenever you add new documents via ingest_pinecone.py.
"""

import os
import sys
import time
import pickle
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

OUTPUT_PATH = Path(__file__).parent / "bm25_corpus.pkl"


def fetch_corpus_from_pinecone() -> list:
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in .env")

    index_name = os.getenv("PINECONE_INDEX_NAME", "steel-rag")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    stats = index.describe_index_stats()
    total = stats.total_vector_count
    print(f"Index '{index_name}': {total} vectors")

    # List all IDs (list() yields batches of ListItem objects; extract .id)
    print("Listing all vector IDs ...")
    all_ids = []
    for id_batch in index.list():
        all_ids.extend([item.id for item in id_batch])
    print(f"  Found {len(all_ids)} IDs")

    # Fetch metadata in batches (Pinecone fetch limit: 1000 per call)
    print("Fetching metadata in batches of 200 ...")
    corpus = []
    batch_size = 200
    t0 = time.time()

    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i : i + batch_size]
        response = index.fetch(ids=batch)
        for vid, vec in response.vectors.items():
            meta = vec.metadata or {}
            text = meta.get("text", "").strip()
            if not text:
                continue
            corpus.append({
                "id":        vid,
                "text":      text,
                "file_name": meta.get("file_name", "Unknown"),
                "category":  meta.get("category", ""),
                "page":      meta.get("page", "?"),
            })
        done = min(i + batch_size, len(all_ids))
        print(f"  {done}/{len(all_ids)} fetched  ({len(corpus)} chunks so far)")

    elapsed = time.time() - t0
    print(f"Corpus ready: {len(corpus)} chunks in {elapsed:.1f}s")
    return corpus


def main():
    print("=" * 60)
    print("Steel RAG — Build BM25 Corpus from Pinecone")
    print("=" * 60)

    corpus = fetch_corpus_from_pinecone()

    if not corpus:
        print("ERROR: No chunks fetched. Check Pinecone connection and index name.")
        sys.exit(1)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(corpus, f)

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(corpus)} chunks → {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print("\nNext steps:")
    print("  1. Commit bm25_corpus.pkl to git (enables hybrid search on Streamlit Cloud)")
    print("  2. Run 'python steel_rag/rag.py' to verify hybrid search is active")
    print("  3. Re-run this script whenever you add new documents")
    print("=" * 60)


if __name__ == "__main__":
    main()
