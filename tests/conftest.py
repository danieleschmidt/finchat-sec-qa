"""
Pytest configuration and shared fixtures for FinChat-SEC-QA tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# Configure pytest-asyncio to use session scope
pytest_asyncio.asyncio_mode = "auto"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_filing_path(test_data_dir: Path) -> Path:
    """Path to sample SEC filing for testing."""
    sample_file = test_data_dir / "sample_filings" / "sample_10k.txt"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not sample_file.exists():
        # Create a minimal sample filing for testing
        sample_file.write_text("""
        UNITED STATES SECURITIES AND EXCHANGE COMMISSION
        Washington, D.C. 20549
        
        FORM 10-K
        
        ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
        
        For the fiscal year ended December 31, 2023
        
        Commission File Number: 001-12345
        
        SAMPLE COMPANY INC.
        
        Risk Factors:
        
        Our business is subject to various risks including:
        - Market volatility may impact our revenue
        - Regulatory changes could affect our operations
        - Competition in the technology sector is intense
        
        Management's Discussion and Analysis:
        
        Our revenue increased by 15% year-over-year due to strong product adoption.
        We continue to invest in research and development to maintain our competitive position.
        """)
    
    return sample_file


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1] * 1536)
    ]
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(
            message=MagicMock(
                content="This is a sample answer with relevant financial information."
            )
        )
    ]
    return mock_client


@pytest.fixture
def mock_edgar_response():
    """Mock EDGAR API response."""
    return {
        "filings": [
            {
                "accessionNumber": "0001234567-23-000001",
                "filingDate": "2023-12-31",
                "reportDate": "2023-12-31",
                "acceptanceDateTime": "2023-12-31T16:30:00.000Z",
                "act": "34",
                "form": "10-K",
                "fileNumber": "001-12345",
                "filmNumber": "231234567",
                "items": "1.01",
                "size": 1234567,
                "isXBRL": 1,
                "isInlineXBRL": 0,
                "primaryDocument": "sample-10k.htm",
                "primaryDocDescription": "FORM 10-K"
            }
        ]
    }


@pytest.fixture
def test_config() -> Dict:
    """Test configuration dictionary."""
    return {
        "edgar": {
            "user_agent": "Test Company test@example.com",
            "rate_limit": 10,
            "timeout": 30
        },
        "openai": {
            "api_key": "test-api-key",
            "model": "gpt-3.5-turbo",
            "max_tokens": 1500,
            "temperature": 0.1
        },
        "cache": {
            "ttl": 3600,
            "max_size": 1000
        },
        "server": {
            "host": "localhost",
            "port": 8000,
            "debug": True
        }
    }


@pytest.fixture
def api_client():
    """FastAPI test client."""
    from finchat_sec_qa.server import app
    return TestClient(app)


@pytest.fixture
async def async_api_client():
    """Async FastAPI test client."""
    from httpx import AsyncClient
    from finchat_sec_qa.server import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_query() -> str:
    """Sample query for testing."""
    return "What are the main risk factors for this company?"


@pytest.fixture
def sample_expected_response() -> Dict:
    """Sample expected response structure."""
    return {
        "answer": "The main risk factors include market volatility, regulatory changes, and intense competition.",
        "citations": [
            {
                "text": "Market volatility may impact our revenue",
                "source": "Risk Factors",
                "page": 1,
                "confidence": 0.95
            }
        ],
        "confidence": 0.87,
        "processing_time": 2.34
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EDGAR_USER_AGENT", "Test Agent test@example.com")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CACHE_TTL", "300")


@pytest.fixture
def performance_thresholds() -> Dict:
    """Performance thresholds for testing."""
    return {
        "query_response_time": 5.0,  # seconds
        "ingestion_time_per_mb": 10.0,  # seconds
        "memory_usage_mb": 500,  # MB
        "concurrent_users": 100
    }


# Pytest markers for test organization
pytest.register_assert_rewrite("tests.helpers.assertions")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "external: Tests requiring external services")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)


# Test data factories
@pytest.fixture
def filing_factory():
    """Factory for creating test filing data."""
    def _create_filing(
        ticker: str = "AAPL",
        form_type: str = "10-K",
        filing_date: str = "2023-12-31",
        content: str = None
    ):
        if content is None:
            content = f"Sample {form_type} filing for {ticker} dated {filing_date}"
        
        return {
            "ticker": ticker,
            "form_type": form_type,
            "filing_date": filing_date,
            "content": content,
            "accession_number": f"0001234567-23-{ticker[:4].upper()}",
            "url": f"https://sec.gov/filings/{ticker.lower()}-{form_type.lower()}.htm"
        }
    
    return _create_filing


@pytest.fixture
def citation_factory():
    """Factory for creating test citation data."""
    def _create_citation(
        text: str = "Sample citation text",
        source: str = "Risk Factors",
        page: int = 1,
        confidence: float = 0.9
    ):
        return {
            "text": text,
            "source": source,
            "page": page,
            "confidence": confidence,
            "start_char": 100,
            "end_char": 100 + len(text)
        }
    
    return _create_citation


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_cache_files():
    """Clean up any cache files created during testing."""
    yield
    # Cleanup logic would go here
    import glob
    for cache_file in glob.glob("*.joblib") + glob.glob("*.pkl"):
        try:
            os.unlink(cache_file)
        except FileNotFoundError:
            pass


# Database fixtures (if needed in the future)
@pytest.fixture(scope="session")
def database_url():
    """Database URL for testing (SQLite in-memory)."""
    return "sqlite:///:memory:"


# Mock external services
@pytest.fixture
def mock_sec_edgar():
    """Mock SEC EDGAR service responses."""
    class MockEDGARService:
        def __init__(self):
            self.call_count = 0
        
        def fetch_filing(self, ticker, form_type):
            self.call_count += 1
            return {
                "content": f"Mock filing content for {ticker} {form_type}",
                "filing_date": "2023-12-31",
                "accession_number": f"mock-{ticker}-{form_type}"
            }
    
    return MockEDGARService()