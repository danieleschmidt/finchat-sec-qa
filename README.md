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
```

### Usage

```bash
# Download and index filings for a company
python ingest_filings.py --ticker AAPL --years 2023,2024

# Start interactive chat
python chat.py

# Launch web interface
streamlit run web_app.py

# Answer a question from local text files
python -m finchat_sec_qa.cli "What risks are highlighted?" aapl.txt

# Speak the answer aloud
python -m finchat_sec_qa.cli --voice "Summarize liquidity discussion" aapl.txt
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

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Run `ruff check . --fix` and `bandit -r src -q` to ensure quality
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Disclaimer

This tool provides information extracted from public SEC filings. It is not financial advice and should not be used as the sole basis for investment decisions.
