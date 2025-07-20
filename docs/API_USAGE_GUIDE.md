# API Usage Guide

This document provides basic examples for using the main modules of the `finchat_sec_qa` package.

## EDGAR Client

```python
from finchat_sec_qa import EdgarClient

client = EdgarClient("your-app/1.0")
filings = client.get_recent_filings("AAPL")
for f in filings:
    print(f.filing_date, f.form_type, f.document_url)
```

## Downloading Filings

```python
from pathlib import Path
from datetime import date
from finchat_sec_qa import EdgarClient, FilingMetadata

client = EdgarClient("my-app")
filing = FilingMetadata(
    cik="0000320193",
    accession_no="0000320193-24-000050",
    form_type="10-K",
    filing_date=date(2024, 10, 30),
    document_url="https://example.com/aapl-20241030.htm",
)
path = client.download_filing(filing, Path("data"))
print(path)
```

## Question Answering

### Basic Usage

```python
from finchat_sec_qa import FinancialQAEngine

engine = FinancialQAEngine()
engine.add_document("aapl", open("aapl.txt").read())
answer, citations = engine.answer_with_citations("What risks are mentioned?")
print(answer)
for c in citations:
    print(c.doc_id, c.text)
```

### Bulk Document Addition (Optimized)

For adding multiple documents efficiently, use bulk operations to avoid rebuilding the index after each document:

```python
from finchat_sec_qa import FinancialQAEngine

engine = FinancialQAEngine()

# Option 1: Use the bulk context manager
with engine.bulk_operation():
    engine.add_document("aapl", open("aapl.txt").read())
    engine.add_document("msft", open("msft.txt").read())
    engine.add_document("googl", open("googl.txt").read())
# Index is rebuilt only once when the context exits

# Option 2: Use the convenience method
documents = [
    ("aapl", open("aapl.txt").read()),
    ("msft", open("msft.txt").read()),
    ("googl", open("googl.txt").read()),
]
engine.add_documents(documents)  # Automatically uses bulk operations
```

**Performance Note**: Bulk operations provide significant performance improvements when adding multiple documents, as the vector index is rebuilt only once instead of after each document addition.

## Multi-Company Analysis

```python
from finchat_sec_qa import compare_question_across_filings

docs = {"AAPL": open("aapl.txt").read(), "MSFT": open("msft.txt").read()}
for result in compare_question_across_filings("revenue", docs):
    print(result.doc_id, result.answer)
```

## Authentication and Security

### Token-Based Authentication

The Flask webapp supports optional token-based authentication via the `FINCHAT_TOKEN` environment variable.

#### Basic Authentication

```bash
# Set a strong authentication token
export FINCHAT_TOKEN="MyStrongToken123!@#"

# Start the webapp
python -m flask --app finchat_sec_qa.webapp run
```

#### Using the API with Authentication

```python
import requests

# Option 1: Authorization header (recommended)
headers = {"Authorization": "Bearer MyStrongToken123!@#"}
response = requests.post("http://localhost:5000/query", 
                        json={"question": "revenue", "ticker": "AAPL"},
                        headers=headers)

# Option 2: Query parameter
response = requests.post("http://localhost:5000/query?token=MyStrongToken123!@#",
                        json={"question": "revenue", "ticker": "AAPL"})
```

### Security Features

The webapp includes several security enhancements:

- **Rate Limiting**: 100 requests per hour per IP address
- **Brute Force Protection**: Exponential backoff after failed authentication attempts
- **Timing Attack Prevention**: Constant-time token comparison
- **Security Headers**: Added to all responses (HSTS, CSP, X-Frame-Options, etc.)
- **Token Strength Validation**: Warns about weak tokens on startup

#### Strong Token Requirements

For optimal security, tokens should:
- Be at least 16 characters long
- Include a mix of uppercase, lowercase, numbers, and special characters
- Use 3 of the 4 character types minimum

Example strong token: `MyApp2024!SecureAuth@`

### Rate Limiting

The API enforces rate limits to prevent abuse:

- **Default**: 100 requests per hour per IP
- **Response**: HTTP 429 when exceeded
- **Reset**: Rolling window, not fixed intervals

### Brute Force Protection

Failed authentication attempts trigger protection:

- **Threshold**: 3 failed attempts
- **Backoff**: Exponential (2^attempts minutes)
- **Reset**: Successful authentication clears history

