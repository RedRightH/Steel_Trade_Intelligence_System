"""
ingest.py — Load all PDFs from Base documents/, chunk, embed, and store in FAISS.
Run this once (or whenever you add new documents) to rebuild the index.

Usage: python ingest.py
"""

import os
import sys
import time
import glob
from pathlib import Path

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DOCS_DIR = Path(__file__).parent.parent / "Base documents"
FAISS_INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
# ─────────────────────────────────────────────────────────────────────────────


def load_pdfs(base_dir: Path) -> list:
    """Recursively find and load all PDFs under base_dir."""
    pdf_paths = glob.glob(str(base_dir / "**" / "*.pdf"), recursive=True)
    # also pick up PDFs directly in base_dir
    pdf_paths += glob.glob(str(base_dir / "*.pdf"))
    pdf_paths = list(set(pdf_paths))  # deduplicate

    print(f"Found {len(pdf_paths)} PDF files")
    docs = []
    failed = []

    for path in sorted(pdf_paths):
        rel = os.path.relpath(path, base_dir)
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            # tag each page with its source folder category
            category = Path(path).parent.name
            for page in pages:
                page.metadata["category"] = category
                page.metadata["file_name"] = Path(path).name
            docs.extend(pages)
            print(f"  [OK] {rel}  ({len(pages)} pages)")
        except Exception as e:
            failed.append(rel)
            print(f"  [FAIL] {rel}  -- {e}")

    if failed:
        print(f"\nFailed to load {len(failed)} file(s): {failed}")

    return docs


def chunk_documents(docs: list) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"\nChunked into {len(chunks)} chunks  "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # show a sample
    if chunks:
        sample = chunks[len(chunks) // 2]
        print(f"\nSample chunk ({sample.metadata.get('file_name', '?')}):")
        print("-" * 60)
        print(sample.page_content[:300])
        print("-" * 60)

    return chunks


def build_faiss_index(chunks: list, index_dir: Path) -> FAISS:
    """Embed chunks and store in a local FAISS index."""
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Building FAISS index (this takes ~1-3 min for 60+ docs)...")
    t0 = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - t0
    print(f"Index built in {elapsed:.1f}s")

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"Index saved to {index_dir}/")

    return vectorstore


def main():
    print("=" * 60)
    print("STEEL RAG - Ingestion Pipeline")
    print("=" * 60)

    t_start = time.time()

    docs = load_pdfs(BASE_DOCS_DIR)
    if not docs:
        raise RuntimeError(f"No documents loaded from {BASE_DOCS_DIR}. "
                           "Check the path exists and contains PDFs.")

    print(f"\nTotal pages loaded: {len(docs)}")

    chunks = chunk_documents(docs)
    build_faiss_index(chunks, FAISS_INDEX_DIR)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done. {len(docs)} pages -> {len(chunks)} chunks -> FAISS index")
    print(f"Total time: {total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
