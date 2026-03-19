# FinChat-SEC-QA

RAG agent that answers questions about 10-K/10-Q filings and outputs citation-anchored responses for financial analysis.

## Features

- **EDGAR Integration**: Automated fetcher for SEC filings (free public API, no auth required)
- **Financial QA Engine**: TF-IDF based RAG pipeline for 10-K/10-Q document analysis
- **Entity Linking**: Extracts executives, products, companies, and financial metrics with graph-based entity map
- **Risk Intelligence**: Sentiment analysis and risk-flag tagging of financial passages
- **Citation Tracking**: Precise source attribution with character-level offsets
- **Voice Interface**: Optional text-to-speech output using `pyttsx3`
- **Multi-Company Analysis**: Compare answers to the same question across multiple filings

## Quick Demo

```bash
# Fetch AAPL's latest 10-K and answer 5 standard questions with citations
python demo.py AAPL

# Try other tickers
python demo.py MSFT
python demo.py TSLA --form 10-Q
```

## Getting Started

### Prerequisites
- Python 3.9+
- SEC EDGAR API (public and free, no registration needed)

### Installation

```bash
git clone https://github.com/danieleschmidt/finchat-sec-qa.git
cd finchat-sec-qa
pip install -e .
pip install beautifulsoup4   # required for HTML parsing
# Optional dev tools
pip install -e '.[dev]'
# Optional voice dependencies
pip install '.[voice]'
```

No API keys needed — SEC EDGAR is a free public API.

### Configuration

Filings are cached at `~/.cache/finchat_sec_qa/`. No configuration required for basic use.

```bash
# Set User-Agent (recommended per SEC guidelines)
export EDGAR_USER_AGENT="Your Name your@email.com"
```

### Usage

```bash
# Download latest filings for AAPL
python -m finchat_sec_qa.cli ingest AAPL --dest filings/

# Answer a question from text files
python -m finchat_sec_qa.cli query "What risks?" filings/aapl.html

# Speak the answer aloud
python -m finchat_sec_qa.cli query --voice "Summarize liquidity" filings/aapl.html

# Assess risk in a text document
python -m finchat_sec_qa.cli risk filings/aapl.html

# Enable debug logging
python -m finchat_sec_qa.cli query --log-level DEBUG "key risks" filings/aapl.html

# Start REST API server
uvicorn finchat_sec_qa.server:app --reload

# The server persists its index under ~/.cache/finchat_sec_qa using
# a joblib file. Any legacy .pkl files will be migrated automatically.

# Start Flask web app (requires FINCHAT_TOKEN)
python -m flask --app finchat_sec_qa.webapp run
```

## Example Queries

- "What were Apple's main risk factors in their latest 10-K?"
- "Compare revenue growth between MSFT and GOOGL over the last 3 quarters"
- "Summarize management's discussion on supply chain challenges"

### Multi-Company Example

```python
from finchat_sec_qa import compare_question_across_filings

docs = {
    "AAPL": open("aapl.txt").read(),
    "MSFT": open("msft.txt").read(),
}

for ans in compare_question_across_filings("revenue guidance", docs):
    print(ans.doc_id, ans.answer)
```

## API Reference

### Core Pipeline

```python
from finchat_sec_qa.edgar_client import EdgarClient
from finchat_sec_qa.qa_engine import FinancialQAEngine
from finchat_sec_qa.entity_linker import extract_entities, summarize_entity_map
from bs4 import BeautifulSoup

# 1. Fetch filing
client = EdgarClient("MyApp/1.0 myemail@example.com")
filings = client.get_recent_filings("AAPL", form_type="10-K", limit=1)
path = client.download_filing(filings[0])

# 2. Parse text
html = path.read_bytes().decode("utf-8", errors="replace")
text = BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)

# 3. Build TF-IDF index
engine = FinancialQAEngine()
engine.add_document("AAPL-10K", text)

# 4. Answer questions with citations
answer, citations = engine.answer_with_citations("What was total revenue?", top_k=3)
print(answer[:300])
for c in citations:
    print(f"  → {c.doc_id} chars {c.start}-{c.end}")

# 5. Extract entities
entities = extract_entities(text, "AAPL-10K")
summary = summarize_entity_map(entities)
for etype, items in summary.items():
    print(f"{etype}: {[i['name'] for i in items[:3]]}")
```

### Entity Linker

The `entity_linker` module extracts named entities from filing text and builds
a dict-based entity map suitable for downstream graph analysis or display:

```python
from finchat_sec_qa.entity_linker import extract_entities, get_top_entities

entities = extract_entities(filing_text, doc_id="AAPL-10K-2024")

# Get top products by mention count
top_products = get_top_entities(entities, "PRODUCT", top_k=5)
for p in top_products:
    print(f"{p.name}: {p.mention_count} mentions")

# Entity types: PERSON, COMPANY, PRODUCT, METRIC
```

For additional examples and advanced usage, see
[API_USAGE_GUIDE.md](docs/API_USAGE_GUIDE.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of releases and notable changes.

## Architecture

- **Data Layer**: EDGAR API connector + vector database
- **Processing Layer**: Financial text preprocessing + embedding generation  
- **Agent Layer**: RAG orchestration + citation tracking
- **Interface Layer**: CLI, Web UI, and REST API

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and code style guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Disclaimer

This tool provides information extracted from public SEC filings. It is not financial advice and should not be used as the sole basis for investment decisions.
