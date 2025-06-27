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

```python
from finchat_sec_qa import FinancialQAEngine

engine = FinancialQAEngine()
engine.add_document("aapl", open("aapl.txt").read())
answer, citations = engine.answer_with_citations("What risks are mentioned?")
print(answer)
for c in citations:
    print(c.doc_id, c.text)
```

## Multi-Company Analysis

```python
from finchat_sec_qa import compare_question_across_filings

docs = {"AAPL": open("aapl.txt").read(), "MSFT": open("msft.txt").read()}
for result in compare_question_across_filings("revenue", docs):
    print(result.doc_id, result.answer)
```

