# FinChat-SEC-QA

RAG agent that answers questions about 10-K/10-Q filings and outputs citation-anchored responses for financial analysis.

## Features

- **EDGAR Integration**: Automated scraper for SEC filings with real-time updates
- **Financial QA Engine**: Specialized RAG pipeline for 10-K/10-Q document analysis
- **Risk Intelligence**: Sentiment analysis and risk-flag tagging of financial passages
- **Citation Tracking**: Precise source attribution with section and page references
- **Voice Interface**: Optional text-to-speech output using `pyttsx3`
- **Multi-Company Analysis**: Compare answers to the same question across multiple filings

## Getting Started

### Prerequisites
- Python 3.9+
- SEC EDGAR API access (free registration required)
- OpenAI API key

### Installation

```bash
git clone https://github.com/yourusername/finchat-sec-qa.git
cd finchat-sec-qa
pip install -r requirements.txt
# Install optional voice dependencies to use the --voice flag
pip install '.[voice]'
```

### Configuration

See `docs/setup.md` for detailed EDGAR API keys and OpenAI configuration.

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env

# Cache directory is ~/.cache/finchat_sec_qa
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

# The server persists its index under ~/.cache/finchat_sec_qa

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

```python
from finchat import FinChatAgent

agent = FinChatAgent("my-app/1.0")
result = agent.query(
    question="What is the debt-to-equity ratio trend?",
    ticker="TSLA",
    filing_type="10-Q"
)
print(result.answer)
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
