# FinChat SEC QA SDK Usage Guide

The FinChat SEC QA SDK provides a clean, typed Python interface for interacting with the FinChat SEC QA service. It supports both synchronous and asynchronous operations with comprehensive error handling and type safety.

## Installation

Install the SDK with the optional dependencies:

```bash
pip install finchat-sec-qa[sdk]
```

This installs the package with the `httpx` dependency required for the SDK client.

## Quick Start

### Synchronous Client

```python
from finchat_sec_qa.sdk import FinChatClient

# Basic usage
client = FinChatClient(base_url="https://api.finchat.example.com")

# Query SEC filings
result = client.query(
    question="What was Apple's revenue in fiscal 2022?",
    ticker="AAPL",
    form_type="10-K"
)

print(f"Answer: {result.answer}")
for citation in result.citations:
    print(f"Citation: {citation.text} (from {citation.source})")

# Analyze risk sentiment
risk_result = client.analyze_risk(
    "The company faces increased competition and supply chain challenges."
)
print(f"Sentiment: {risk_result.sentiment}")
print(f"Risk flags: {risk_result.flags}")

# Check service health
health = client.health_check()
print(f"Service status: {health.status}")
print(f"Version: {health.version}")
```

### Asynchronous Client

```python
import asyncio
from finchat_sec_qa.sdk import AsyncFinChatClient

async def main():
    async with AsyncFinChatClient(base_url="https://api.finchat.example.com") as client:
        # Query SEC filings asynchronously
        result = await client.query(
            question="What are Microsoft's key risk factors?",
            ticker="MSFT",
            form_type="10-K"
        )
        
        print(f"Answer: {result.answer}")
        
        # Perform risk analysis
        risk_result = await client.analyze_risk(
            "Strong financial performance with growing market share."
        )
        print(f"Sentiment: {risk_result.sentiment}")

# Run the async function
asyncio.run(main())
```

### Context Manager Usage

Both clients support context manager protocol for automatic resource cleanup:

```python
from finchat_sec_qa.sdk import FinChatClient, AsyncFinChatClient

# Synchronous context manager
with FinChatClient(base_url="https://api.finchat.example.com") as client:
    result = client.query("What is the company's revenue?", "AAPL", "10-K")
    print(result.answer)

# Asynchronous context manager
async def query_with_context():
    async with AsyncFinChatClient(base_url="https://api.finchat.example.com") as client:
        result = await client.query("What is the company's revenue?", "AAPL", "10-K")
        return result
```

## Configuration

### Client Configuration Options

```python
from finchat_sec_qa.sdk import FinChatClient

client = FinChatClient(
    base_url="https://api.finchat.example.com",  # API endpoint
    timeout=60,                                   # Request timeout in seconds
    api_key="your-api-key",                      # Optional API key
    max_retries=3,                               # Max retries for failed requests
    retry_delay=1.0,                             # Delay between retries
    user_agent="MyApp/1.0"                       # Custom user agent
)
```

### Environment Variables

You can also configure the client using environment variables:

```bash
export FINCHAT_BASE_URL="https://api.finchat.example.com"
export FINCHAT_API_KEY="your-api-key"
export FINCHAT_TIMEOUT="60"
```

```python
import os
from finchat_sec_qa.sdk import FinChatClient

client = FinChatClient(
    base_url=os.getenv("FINCHAT_BASE_URL", "http://localhost:8000"),
    api_key=os.getenv("FINCHAT_API_KEY"),
    timeout=int(os.getenv("FINCHAT_TIMEOUT", "30"))
)
```

## API Methods

### Query Method

Query SEC filings for specific information about companies:

```python
response = client.query(
    question="What was the company's total revenue?",
    ticker="AAPL",           # Stock ticker symbol
    form_type="10-K",        # SEC form type (10-K, 10-Q, etc.)
    limit=1                  # Max number of documents to search
)

# Response structure
print(response.answer)       # Generated answer
for citation in response.citations:
    print(citation.text)     # Citation text
    print(citation.source)   # Source document
    print(citation.page)     # Page number (if available)
    print(citation.start_pos) # Start position in document
    print(citation.end_pos)   # End position in document
```

### Risk Analysis Method

Analyze sentiment and risk factors in financial text:

```python
risk_response = client.analyze_risk(
    "The company reported strong quarterly earnings with increased market share."
)

print(risk_response.sentiment)  # "positive", "negative", or "neutral"
print(risk_response.flags)      # List of detected risk/opportunity flags
```

### Health Check Method

Check the health and status of the FinChat service:

```python
health_response = client.health_check()

print(health_response.status)     # "healthy", "degraded", or "unhealthy"
print(health_response.version)    # Service version
print(health_response.timestamp)  # Health check timestamp
print(health_response.services)   # Individual service statuses

# Check if service is healthy
if health_response.is_healthy:
    print("Service is running normally")
```

## Error Handling

The SDK provides specific exception types for different error conditions:

```python
from finchat_sec_qa.sdk import (
    FinChatClient,
    FinChatValidationError,
    FinChatNotFoundError,
    FinChatTimeoutError,
    FinChatConnectionError,
    FinChatAPIError,
)

client = FinChatClient()

try:
    result = client.query("What is revenue?", "AAPL", "10-K")
    print(result.answer)
    
except FinChatValidationError as e:
    # Input validation failed (HTTP 400)
    print(f"Validation error: {e.message}")
    
except FinChatNotFoundError as e:
    # Requested resource not found (HTTP 404)
    print(f"Not found: {e.message}")
    
except FinChatTimeoutError as e:
    # Request timeout
    print(f"Request timed out: {e.message}")
    
except FinChatConnectionError as e:
    # Connection/network error
    print(f"Connection failed: {e.message}")
    
except FinChatAPIError as e:
    # Generic API error
    print(f"API error {e.status_code}: {e.message}")
```

## Type Safety and IDE Support

The SDK is fully typed with comprehensive type annotations:

```python
from finchat_sec_qa.sdk import FinChatClient, QueryResponse, Citation
from typing import List

def analyze_company(client: FinChatClient, ticker: str) -> QueryResponse:
    """Type-safe function using the SDK."""
    response: QueryResponse = client.query(
        question="What are the main business segments?",
        ticker=ticker,
        form_type="10-K"
    )
    
    # Citations are properly typed
    citations: List[Citation] = response.citations
    
    return response

# IDE will provide full autocompletion and type checking
client = FinChatClient()
result = analyze_company(client, "AAPL")
```

## Advanced Usage

### Batch Processing

Process multiple queries efficiently:

```python
import asyncio
from finchat_sec_qa.sdk import AsyncFinChatClient

async def batch_analysis(tickers: list[str]) -> dict[str, str]:
    """Analyze multiple companies concurrently."""
    results = {}
    
    async with AsyncFinChatClient() as client:
        tasks = []
        
        for ticker in tickers:
            task = client.query(
                question="What is the company's primary revenue source?",
                ticker=ticker,
                form_type="10-K"
            )
            tasks.append((ticker, task))
        
        # Execute all queries concurrently
        for ticker, task in tasks:
            try:
                result = await task
                results[ticker] = result.answer
            except Exception as e:
                results[ticker] = f"Error: {str(e)}"
    
    return results

# Usage
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
results = asyncio.run(batch_analysis(tickers))
for ticker, answer in results.items():
    print(f"{ticker}: {answer}")
```

### Custom Retry Logic

Implement custom retry logic for specific use cases:

```python
import time
from finchat_sec_qa.sdk import FinChatClient, FinChatConnectionError

def query_with_custom_retry(client: FinChatClient, question: str, ticker: str, max_attempts: int = 5):
    """Custom retry logic with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return client.query(question, ticker, "10-K")
        except FinChatConnectionError:
            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
    
# Usage
client = FinChatClient(max_retries=0)  # Disable built-in retries
result = query_with_custom_retry(client, "What is the revenue?", "AAPL")
```

### Integration with Data Analysis

Integrate with pandas and other data analysis tools:

```python
import pandas as pd
from finchat_sec_qa.sdk import FinChatClient

def create_financial_summary(tickers: list[str]) -> pd.DataFrame:
    """Create a DataFrame with financial summaries."""
    client = FinChatClient()
    
    data = []
    questions = [
        "What is the total revenue?",
        "What is the net income?",
        "What are the main risk factors?"
    ]
    
    for ticker in tickers:
        row = {"Ticker": ticker}
        for question in questions:
            try:
                result = client.query(question, ticker, "10-K")
                row[question] = result.answer
            except Exception as e:
                row[question] = f"Error: {str(e)}"
        data.append(row)
    
    return pd.DataFrame(data)

# Usage
df = create_financial_summary(["AAPL", "MSFT", "GOOGL"])
print(df)
```

## Testing

The SDK includes comprehensive test utilities for testing your applications:

```python
import pytest
from unittest.mock import Mock, patch
from finchat_sec_qa.sdk import FinChatClient, QueryResponse, Citation

def test_my_analysis_function():
    """Test function that uses the FinChat SDK."""
    
    # Mock the client
    with patch('finchat_sec_qa.sdk.FinChatClient') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        # Mock response
        mock_response = QueryResponse(
            answer="Apple Inc. reported revenue of $394.3 billion.",
            citations=[
                Citation(
                    text="Revenue was $394.3 billion",
                    source="AAPL 10-K 2022",
                    page=1,
                    start_pos=1000,
                    end_pos=1025
                )
            ]
        )
        mock_client.query.return_value = mock_response
        
        # Test your function
        result = my_analysis_function("AAPL")
        
        # Verify calls
        mock_client.query.assert_called_once_with(
            question="What is the revenue?",
            ticker="AAPL", 
            form_type="10-K"
        )
        
        assert "394.3 billion" in result
```

## Performance Considerations

- **Connection Pooling**: The SDK automatically manages HTTP connection pooling
- **Async Operations**: Use the async client for better concurrency
- **Retry Logic**: Built-in retry logic handles temporary failures
- **Timeout Configuration**: Set appropriate timeouts for your use case
- **Resource Cleanup**: Always use context managers or call `close()` explicitly

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure to install with SDK dependencies:
   ```bash
   pip install finchat-sec-qa[sdk]
   ```

2. **Connection Errors**: Check the base URL and network connectivity:
   ```python
   # Test connection
   try:
       health = client.health_check()
       print(f"Connected successfully: {health.status}")
   except FinChatConnectionError:
       print("Failed to connect to FinChat API")
   ```

3. **Authentication Issues**: Verify API key configuration:
   ```python
   client = FinChatClient(
       api_key="your-api-key",
       base_url="https://api.finchat.example.com"
   )
   ```

4. **Timeout Issues**: Increase timeout for complex queries:
   ```python
   client = FinChatClient(timeout=120)  # 2 minute timeout
   ```

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
import httpx

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# The SDK will log HTTP requests and responses
client = FinChatClient()
result = client.query("What is revenue?", "AAPL", "10-K")
```

## Support

- **Documentation**: Check the main FinChat documentation
- **Examples**: See the `examples/` directory for more use cases
- **Issues**: Report bugs and feature requests on the project repository
- **API Reference**: Full API documentation is available in the docstrings

---

The FinChat SEC QA SDK provides a robust, type-safe way to integrate SEC filing analysis into your applications. With comprehensive error handling, async support, and extensive documentation, it's designed to make financial data analysis accessible and reliable.