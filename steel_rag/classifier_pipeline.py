"""
classifier_pipeline.py — Live RSS News Pipeline for India Steel Trade Intelligence

Fetches articles from steel trade RSS feeds, classifies each one,
chunks the text, and upserts into the same Pinecone index the RAG system uses.
New articles are immediately queryable via the dashboard.

Deduplication: each article's URL is hashed into a stable vector ID prefix;
Pinecone upsert is idempotent, so re-running never creates duplicates.

Modes:
    python classifier_pipeline.py --once           # fetch all feeds once, exit
    python classifier_pipeline.py --daemon         # run every 4 hours (APScheduler)
    python classifier_pipeline.py --daemon --interval 2   # every 2 hours
    python classifier_pipeline.py --status         # show feed health + article count

Requirements (added to requirements.txt):
    feedparser>=6.0.0
    apscheduler>=3.10.0
    beautifulsoup4>=4.12.0
    requests>=2.31.0
"""

import os
import sys
import json
import time
import hashlib
import textwrap
import traceback
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

# ── Setup ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import feedparser
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_INDEX   = os.getenv("PINECONE_INDEX_NAME", "steel-rag")
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL       = "llama-3.3-70b-versatile"
CHUNK_SIZE       = 600
CHUNK_OVERLAP    = 80
FETCH_TIMEOUT    = 15          # seconds per article HTTP request
MAX_ARTICLE_CHARS = 8_000     # cap article body before chunking
LOG_FILE         = Path(__file__).parent / "pipeline_log.json"

# ── RSS Feed Catalogue ────────────────────────────────────────────────────────
# Note: Most Indian steel-specific sites don't expose RSS.
# We use broader industry/business feeds and filter by steel keywords below.
FEEDS = [
    {
        "name":     "Livemint — Industry",
        "url":      "https://www.livemint.com/rss/industry",
        "category": "steel_news",
    },
    {
        "name":     "Hindu BusinessLine — Markets/Commodities",
        "url":      "https://www.thehindubusinessline.com/markets/commodities/feeder/default.rss",
        "category": "steel_news",
    },
    {
        "name":     "Hindu BusinessLine — Economy & Policy",
        "url":      "https://www.thehindubusinessline.com/economy/feeder/default.rss",
        "category": "policy_news",
    },
    {
        "name":     "Times of India — Business",
        "url":      "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
        "category": "steel_news",
    },
]

# ── Steel relevance filter ────────────────────────────────────────────────────
# Articles must contain at least one of these keywords (case-insensitive)
# to be processed. This filters out off-topic articles from general feeds.
STEEL_KEYWORDS = {
    # Products (specific — avoid single-word false positives)
    "steel", "stainless", "galvanized", "galvanised",
    "seamless tube", "seamless pipe", "steel coil", "steel plate",
    "hr coil", "cr coil", "hot rolled", "cold rolled", "alloy steel",
    "rebar", "steel billet", "steel slab", "structural steel",
    "flat product", "long product", "steel sheet",
    # Companies
    "tata steel", "jsw steel", "sail", "rinl", "jspl", "jindal steel",
    "posco", "arcelormittal", "essar steel", "vizag steel", "nmdc",
    "steel authority",
    # Trade policy
    "anti-dumping", "anti dumping", "safeguard measure", "dgtr",
    "qco steel", "bis standard", "trade remedy", "steel import",
    "steel export", "import duty on steel",
    # Raw materials
    "iron ore", "coking coal", "metallurgical coal", "scrap steel",
    "blast furnace", "electric arc furnace", "steel pellet",
    # Carbon / CBAM
    "cbam", "carbon border adjustment", "green steel", "decarboni",
    # General steel sector
    "steel producer", "steel maker", "steel industry", "steel sector",
    "steel demand", "steel price", "steel output", "steel capacity",
}

# ── Classification labels ─────────────────────────────────────────────────────
LABELS = [
    "ANTI_DUMPING",
    "SAFEGUARD",
    "POLICY_OPPORTUNITY",
    "RAW_MATERIAL",
    "CBAM_COMPLIANCE",
    "DATA_ANALYSIS",
    "TARIFF_ANALYSIS",
    "GENERAL_STEEL_NEWS",   # catch-all for articles that don't fit other categories
]

CLASSIFY_SYSTEM = """You are a steel trade policy classifier.
Given a news article about the steel industry, classify it into exactly ONE of these categories:

ANTI_DUMPING       — anti-dumping investigations, duties, margins, ADD orders
SAFEGUARD          — safeguard investigations, provisional/final safeguard measures
POLICY_OPPORTUNITY — trade policy, FTP, BIS/QCO standards, export promotion
RAW_MATERIAL       — iron ore, coking coal, scrap, raw material prices/supply
CBAM_COMPLIANCE    — EU Carbon Border Adjustment Mechanism, carbon pricing, decarbonisation
DATA_ANALYSIS      — trade statistics, export/import volumes, market share data
TARIFF_ANALYSIS    — MFN tariffs, FTA preferential rates, HS code duties
GENERAL_STEEL_NEWS — production, capacity, company news, prices (none of the above)

Reply with ONLY the category name, nothing else."""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _article_id(url: str) -> str:
    """Stable 16-char hex ID derived from URL."""
    return hashlib.md5(url.encode()).hexdigest()[:16]


def _chunk_id(article_id: str, chunk_idx: int) -> str:
    return f"news_{article_id}_{chunk_idx:04d}"


def _fetch_article_text(url: str) -> str:
    """
    Fetch and extract main article text from a URL.
    Falls back to empty string on any error.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SteelRAGBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove boilerplate elements
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "button", "iframe"]):
            tag.decompose()

        # Try common article containers first
        for selector in ["article", "div.article-body", "div.story-body",
                         "div.content", "div.post-content", "main"]:
            container = soup.select_one(selector)
            if container:
                text = container.get_text(separator=" ", strip=True)
                if len(text) > 200:
                    return text[:MAX_ARTICLE_CHARS]

        # Fallback: all paragraph text
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        return text[:MAX_ARTICLE_CHARS]

    except Exception as e:
        log.debug(f"Failed to fetch {url}: {e}")
        return ""


def _classify_article(title: str, summary: str, groq_client: Groq) -> str:
    """Classify article into one of LABELS using LLM."""
    snippet = f"Title: {title}\n\nSummary: {summary[:800]}"
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user",   "content": snippet},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        label = resp.choices[0].message.content.strip().upper()
        return label if label in LABELS else "GENERAL_STEEL_NEWS"
    except Exception as e:
        log.warning(f"Classification failed: {e}")
        return "GENERAL_STEEL_NEWS"


def _load_log() -> dict:
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"seen_ids": [], "runs": []}


def _save_log(data: dict):
    LOG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def fetch_new_articles(seen_ids: set) -> list[dict]:
    """
    Parse all feeds and return new articles not in seen_ids.

    Each article dict:
        id, url, title, summary, full_text, published, feed_name, category
    """
    new_articles = []

    for feed_cfg in FEEDS:
        log.info(f"Fetching: {feed_cfg['name']}")
        try:
            parsed = feedparser.parse(feed_cfg["url"])
            entries = parsed.get("entries", [])
            log.info(f"  {len(entries)} entries found")

            for entry in entries:
                url   = entry.get("link", "")
                title = entry.get("title", "").strip()
                if not url or not title:
                    continue

                aid = _article_id(url)
                if aid in seen_ids:
                    continue

                # Steel relevance filter — skip off-topic articles
                searchable = (title + " " + entry.get("summary", "")).lower()
                if not any(kw in searchable for kw in STEEL_KEYWORDS):
                    continue

                # Get summary from feed
                summary = (entry.get("summary", "")
                           or entry.get("description", "")
                           or "")
                # Strip HTML from summary
                summary = BeautifulSoup(summary, "html.parser").get_text(strip=True)
                summary = summary[:1000]

                # Published date
                published_struct = entry.get("published_parsed")
                if published_struct:
                    published = datetime(*published_struct[:6],
                                        tzinfo=timezone.utc).isoformat()
                else:
                    published = datetime.now(timezone.utc).isoformat()

                new_articles.append({
                    "id":        aid,
                    "url":       url,
                    "title":     title,
                    "summary":   summary,
                    "full_text": "",    # fetched separately
                    "published": published,
                    "feed_name": feed_cfg["name"],
                    "category":  feed_cfg["category"],
                })

        except Exception as e:
            log.warning(f"  Feed error for {feed_cfg['name']}: {e}")

    log.info(f"New articles to process: {len(new_articles)}")
    return new_articles


def process_articles(articles: list[dict]) -> int:
    """
    For each article: fetch full text, classify, chunk, embed, upsert to Pinecone.
    Returns count of successfully upserted articles.
    """
    if not articles:
        return 0

    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key:
        log.error("PINECONE_API_KEY not set — cannot upsert")
        return 0

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        log.error("GROQ_API_KEY not set")
        return 0

    # Lazy-load heavy dependencies
    from pinecone import Pinecone

    log.info("Loading embedding model…")
    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    groq_client = Groq(api_key=groq_key)
    pc          = Pinecone(api_key=api_key)
    index       = pc.Index(PINECONE_INDEX)

    upserted_count = 0

    for i, art in enumerate(articles, 1):
        log.info(f"[{i}/{len(articles)}] {art['title'][:70]}")

        # 1. Fetch full text
        log.debug(f"  Fetching article text from {art['url']}")
        art["full_text"] = _fetch_article_text(art["url"])

        # 2. Build document text: title + summary + body
        doc_text = f"{art['title']}\n\n{art['summary']}"
        if art["full_text"] and len(art["full_text"]) > len(art["summary"]) + 100:
            doc_text += f"\n\n{art['full_text']}"
        doc_text = doc_text.strip()

        if len(doc_text) < 50:
            log.warning("  Too short — skipping")
            continue

        # 3. Classify
        label = _classify_article(art["title"], art["summary"], groq_client)
        log.info(f"  Classified as: {label}")

        # 3b. AI-GPR Layer-1 Steel Trade Risk Score (0-1, temp=0)
        try:
            from steel_futures import score_article as _score_article
            scored = _score_article(f"{art['title']} {art['summary'][:500]}", client=groq_client)
            art["risk_score"]     = scored["risk_score"]
            art["steel_relevant"] = scored["steel_relevant"]
            log.info(f"  Risk score: {scored['risk_score']:.2f}")
        except Exception as _e:
            art["risk_score"]     = 0.0
            art["steel_relevant"] = False
            log.debug(f"  Risk score failed: {_e}")

        # 4. Chunk
        chunks = splitter.split_text(doc_text)
        if not chunks:
            continue

        # 5. Embed
        try:
            embeddings = embedder.embed_documents(chunks)
        except Exception as e:
            log.warning(f"  Embedding failed: {e}")
            continue

        # 6. Build Pinecone records
        records = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vid = _chunk_id(art["id"], idx)
            records.append({
                "id":     vid,
                "values": emb,
                "metadata": {
                    "text":        chunk[:1000],
                    "file_name":   f"[NEWS] {art['title'][:60]}",
                    "category":    label,
                    "source_type": "rss_news",
                    "feed_name":   art["feed_name"],
                    "url":         art["url"],
                    "published":   art["published"],
                    "page":        str(idx),
                },
            })

        # 7. Upsert
        try:
            index.upsert(vectors=records)
            log.info(f"  Upserted {len(records)} chunks to Pinecone")
            upserted_count += 1
        except Exception as e:
            log.warning(f"  Pinecone upsert failed: {e}")

        # Brief pause to avoid hammering APIs
        time.sleep(0.5)

    return upserted_count


def run_pipeline() -> dict:
    """
    Full pipeline run: fetch → classify → embed → upsert.
    Returns a run summary dict.
    """
    log.info("=" * 55)
    log.info("Steel RAG — News Pipeline Run")
    log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 55)

    t0 = time.time()

    # Load dedup log
    log_data  = _load_log()
    seen_ids  = set(log_data.get("seen_ids", []))
    log.info(f"Previously seen articles: {len(seen_ids)}")

    # Fetch new articles
    articles = fetch_new_articles(seen_ids)

    # Process them
    upserted = process_articles(articles)

    # Update log
    new_ids = [a["id"] for a in articles]
    log_data["seen_ids"] = list(seen_ids | set(new_ids))

    run_record = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "new_found":    len(articles),
        "upserted":     upserted,
        "elapsed_s":    round(time.time() - t0, 1),
        "articles": [
            {
                "title":         a["title"],
                "feed":          a["feed_name"],
                "label":         a.get("label", ""),
                "url":           a["url"],
                "published":     a.get("published", ""),
                "risk_score":    round(a.get("risk_score", 0.0), 3),
                "steel_relevant": a.get("steel_relevant", False),
                "summary":       a.get("summary", "")[:300],
            }
            for a in articles[:50]   # keep last 50 in log for dashboard
        ],
    }

    # Build / update Steel-GPR index from all scored articles this run
    try:
        from steel_futures import build_steel_gpr_index
        scored_arts = [
            {"title": a["title"], "text": a.get("summary", ""),
             "published": a.get("published", "")[:10]}
            for a in articles if a.get("risk_score") is not None
        ]
        if scored_arts:
            build_steel_gpr_index(scored_arts, save=True)
            log.info(f"Steel-GPR index updated with {len(scored_arts)} articles")
    except Exception as _e:
        log.warning(f"Steel-GPR index build failed: {_e}")
    log_data.setdefault("runs", []).append(run_record)
    # Keep only last 50 runs in log
    log_data["runs"] = log_data["runs"][-50:]
    _save_log(log_data)

    elapsed = time.time() - t0
    log.info(f"Done. {upserted}/{len(articles)} articles upserted in {elapsed:.1f}s")
    return run_record


def show_status():
    """Print current pipeline status."""
    log_data = _load_log()
    runs     = log_data.get("runs", [])
    seen     = log_data.get("seen_ids", [])

    print(f"\n{'='*50}")
    print("Steel RAG — Pipeline Status")
    print(f"{'='*50}")
    print(f"Total seen articles : {len(seen)}")
    print(f"Total runs          : {len(runs)}")

    if runs:
        last = runs[-1]
        print(f"Last run            : {last['timestamp'][:19]}")
        print(f"  New found         : {last['new_found']}")
        print(f"  Upserted          : {last['upserted']}")
        print(f"  Elapsed           : {last['elapsed_s']}s")

    print(f"\nFeed catalogue ({len(FEEDS)} feeds):")
    for f in FEEDS:
        print(f"  - {f['name']}")
        print(f"    {f['url']}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="India Steel Trade — Live RSS News Pipeline"
    )
    parser.add_argument("--once",     action="store_true",
                        help="Run the pipeline once and exit")
    parser.add_argument("--daemon",   action="store_true",
                        help="Run on a schedule (default: every 4 hours)")
    parser.add_argument("--interval", type=int, default=4,
                        help="Interval in hours for daemon mode (default: 4)")
    parser.add_argument("--status",   action="store_true",
                        help="Show pipeline status and exit")
    parser.add_argument("--debug",    action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.status:
        show_status()
        return

    if args.once:
        run_pipeline()
        return

    if args.daemon:
        from apscheduler.schedulers.blocking import BlockingScheduler

        log.info(f"Starting daemon — will run every {args.interval} hour(s)")
        log.info("Press Ctrl+C to stop.")

        # Run once immediately on startup
        run_pipeline()

        scheduler = BlockingScheduler()
        scheduler.add_job(
            run_pipeline,
            trigger="interval",
            hours=args.interval,
            id="news_pipeline",
            max_instances=1,
            misfire_grace_time=300,
        )
        try:
            scheduler.start()
        except KeyboardInterrupt:
            log.info("Daemon stopped by user.")
        return

    # No args — print help
    parser.print_help()


if __name__ == "__main__":
    main()
