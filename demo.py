#!/usr/bin/env python3
"""
demo.py — FinChat SEC QA demo

Usage:
    python demo.py AAPL
    python demo.py MSFT --form 10-Q
    python demo.py TSLA --year 2023

Fetches the latest SEC 10-K filing for a ticker, builds a TF-IDF index,
answers 5 standard financial questions with citations, and extracts
key entities (executives, products, companies, metrics).
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

# Ensure src/ is on the path when run from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: beautifulsoup4 not installed. Run: pip install beautifulsoup4")
    sys.exit(1)

from finchat_sec_qa.edgar_client import EdgarClient
from finchat_sec_qa.qa_engine import FinancialQAEngine
from finchat_sec_qa.entity_linker import extract_entities, summarize_entity_map, get_top_entities
from finchat_sec_qa.citation import Citation

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

USER_AGENT = "Daniel Schmidt danschmidt88@gmail.com"

STANDARD_QUESTIONS = [
    "What was total revenue?",
    "What are the main risk factors?",
    "Who are the executives?",
    "What are the key products?",
    "What is the net income?",
]


def fetch_and_parse_filing(ticker: str, form_type: str = "10-K") -> tuple[str, dict]:
    """
    Fetch the most recent filing for the ticker, download it, and parse text.
    Returns (plain_text, filing_metadata_dict).
    """
    client = EdgarClient(USER_AGENT)
    print(f"[1/4] Looking up {ticker} on SEC EDGAR...")
    filings = client.get_recent_filings(ticker, form_type=form_type, limit=5)

    if not filings:
        print(f"ERROR: No {form_type} filings found for {ticker}")
        sys.exit(1)

    filing = filings[0]
    print(f"      Found: {form_type} filed {filing.filing_date}")
    print(f"      URL: {filing.document_url}")

    print(f"[2/4] Downloading filing (may be cached)...")
    path = client.download_filing(filing)
    size_kb = path.stat().st_size // 1024
    print(f"      {path.name} ({size_kb} KB)")

    print(f"[3/4] Parsing HTML...")
    html = path.read_bytes().decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    char_count = len(text)
    print(f"      Extracted {char_count:,} characters of text")

    meta = {
        "ticker": ticker,
        "form_type": filing.form_type,
        "filing_date": str(filing.filing_date),
        "document_url": filing.document_url,
        "cik": filing.cik,
        "accession_no": filing.accession_no,
    }
    return text, meta


def build_qa_index(text: str, doc_id: str) -> FinancialQAEngine:
    """Build a TF-IDF index from filing text."""
    engine = FinancialQAEngine()  # in-memory, no persistence
    engine.add_document(doc_id, text)
    print(f"      TF-IDF index: {len(engine.chunks)} chunks")
    return engine


def answer_questions(engine: FinancialQAEngine, doc_id: str) -> list[dict]:
    """Answer all standard questions and return structured results."""
    results = []
    for question in STANDARD_QUESTIONS:
        answer, citations = engine.answer_with_citations(question, top_k=2)
        results.append({
            "question": question,
            "answer": answer,
            "citations": citations,
        })
    return results


def extract_and_summarize_entities(text: str, doc_id: str) -> dict:
    """Extract named entities from filing text."""
    entity_map = extract_entities(text, doc_id)
    return summarize_entity_map(entity_map)


def print_results(ticker: str, meta: dict, qa_results: list[dict], entity_summary: dict) -> None:
    """Pretty-print QA results and entities."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  FinChat SEC QA — {ticker} {meta['form_type']} ({meta['filing_date']})")
    print(f"{bar}\n")

    print("📋 Q&A WITH CITATIONS")
    print("-" * 70)
    for i, r in enumerate(qa_results, 1):
        print(f"\n[Q{i}] {r['question']}")
        # Wrap answer to 80 chars
        snippet = r["answer"][:400].replace("\n", " ").strip()
        wrapped = textwrap.fill(snippet, width=78, initial_indent="    ", subsequent_indent="    ")
        print(wrapped)
        if r["citations"]:
            c = r["citations"][0]
            print(f"     ↳ Citation: {c.doc_id} | chars {c.start}–{c.end}")

    print(f"\n\n🔗 ENTITY MAP")
    print("-" * 70)

    type_icons = {
        "PERSON": "👤",
        "COMPANY": "🏢",
        "PRODUCT": "📱",
        "METRIC": "📊",
    }

    for etype, items in entity_summary.items():
        icon = type_icons.get(etype, "•")
        print(f"\n{icon} {etype} ({len(items)} unique)")
        for item in items[:6]:
            print(f"   • {item['name']} ({item['mentions']} mention{'s' if item['mentions'] != 1 else ''})")

    print(f"\n{bar}")
    print(f"  Source: {meta['document_url']}")
    print(f"  Accession: {meta['accession_no']}")
    print(bar)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch SEC filing and answer financial questions with citations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python demo.py AAPL
              python demo.py MSFT --form 10-Q
              python demo.py TSLA --form 10-K
        """)
    )
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--form", type=str, default="10-K",
        choices=["10-K", "10-Q"],
        help="SEC form type (default: 10-K)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ticker = args.ticker.upper()

    # Step 1-3: Fetch, download, parse
    text, meta = fetch_and_parse_filing(ticker, form_type=args.form)
    doc_id = f"{ticker}-{args.form}-{meta['filing_date']}"

    # Step 4: Build QA index
    print(f"[4/4] Building TF-IDF index and answering questions...")
    engine = build_qa_index(text, doc_id)

    # Step 5: Answer questions
    qa_results = answer_questions(engine, doc_id)

    # Step 6: Extract entities
    print(f"      Extracting entities...")
    entity_summary = extract_and_summarize_entities(text, doc_id)

    # Step 7: Display results
    print_results(ticker, meta, qa_results, entity_summary)


if __name__ == "__main__":
    main()
