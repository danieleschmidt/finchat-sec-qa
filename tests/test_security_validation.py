"""Security validation tests for input sanitization and validation."""

import pytest
from pydantic import ValidationError

from finchat_sec_qa.server import QueryRequest, RiskRequest
from finchat_sec_qa.edgar_client import EdgarClient


class TestQueryRequestValidation:
    """Test security validation for QueryRequest model."""

    def test_valid_ticker_formats(self):
        """Test that valid ticker formats are accepted."""
        valid_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'A', 'BRK']
        
        for ticker in valid_tickers:
            req = QueryRequest(question="What is revenue?", ticker=ticker)
            assert req.ticker == ticker.upper()

    def test_invalid_ticker_formats_rejected(self):
        """Test that invalid ticker formats are rejected."""
        invalid_tickers = [
            '',           # Empty string
            '123',        # Numbers only
            'AAPL1',      # Contains number
            'GOOGLE',     # Too long (6 chars)
            'AA-PL',      # Contains hyphen
            'AA.PL',      # Contains dot
            'AA PL',      # Contains space
            'AAPL<',      # Contains special char
            'AAPL>',      # Contains special char
            'AAPL&',      # Contains ampersand
            'AAPL"',      # Contains quote
            "AAPL'",      # Contains single quote
            'AAPL/',      # Contains slash
            'AAPL\\',     # Contains backslash
            'AAPL%',      # Contains percent
            None,         # None value
        ]
        
        for ticker in invalid_tickers:
            with pytest.raises(ValidationError):
                QueryRequest(question="What is revenue?", ticker=ticker)

    def test_valid_form_types(self):
        """Test that valid form types are accepted."""
        valid_forms = ['10-K', '10-Q', '8-K', 'DEF-14A', 'S-1']
        
        for form_type in valid_forms:
            req = QueryRequest(question="What is revenue?", ticker="AAPL", form_type=form_type)
            assert req.form_type == form_type

    def test_invalid_form_types_rejected(self):
        """Test that invalid form types are rejected."""
        invalid_forms = [
            '',                    # Empty string
            '10-K-INVALID',       # Too long
            '10K',                # Missing hyphen (depends on validation)
            '10-K<script>',       # Script injection attempt
            'javascript:alert',   # Script injection
            None,                 # None value
        ]
        
        for form_type in invalid_forms:
            with pytest.raises(ValidationError):
                QueryRequest(question="What is revenue?", ticker="AAPL", form_type=form_type)

    def test_question_xss_protection(self):
        """Test that XSS attempts in questions are rejected."""
        dangerous_questions = [
            '<script>alert("xss")</script>',
            'javascript:alert("xss")',
            'vbscript:msgbox("xss")',
            'What is revenue? <script>',
            'Revenue onload=alert("xss")',
            'Revenue onerror=alert("xss")',
            'Revenue onclick=alert("xss")',
        ]
        
        for question in dangerous_questions:
            with pytest.raises(ValidationError):
                QueryRequest(question=question, ticker="AAPL")

    def test_question_length_limits(self):
        """Test question length validation."""
        # Test empty question
        with pytest.raises(ValidationError):
            QueryRequest(question="", ticker="AAPL")
        
        # Test extremely long question (over 1000 chars)
        long_question = "A" * 1001
        with pytest.raises(ValidationError):
            QueryRequest(question=long_question, ticker="AAPL")
        
        # Test valid length question
        valid_question = "A" * 500
        req = QueryRequest(question=valid_question, ticker="AAPL")
        assert len(req.question) == 500


class TestRiskRequestValidation:
    """Test security validation for RiskRequest model."""

    def test_text_xss_protection(self):
        """Test that XSS attempts in text are rejected."""
        dangerous_texts = [
            '<script>alert("xss")</script>',
            'javascript:alert("xss")',
            'vbscript:msgbox("xss")',
            'Company performance <script>',
            'Revenue onload=alert("xss")',
            'Earnings onerror=alert("xss")',
            'Growth onclick=alert("xss")',
        ]
        
        for text in dangerous_texts:
            with pytest.raises(ValidationError):
                RiskRequest(text=text)

    def test_text_length_limits(self):
        """Test text length validation."""
        # Test empty text
        with pytest.raises(ValidationError):
            RiskRequest(text="")
        
        # Test extremely long text (over 50000 chars)
        long_text = "A" * 50001
        with pytest.raises(ValidationError):
            RiskRequest(text=long_text)
        
        # Test valid length text
        valid_text = "Company shows strong financial performance." * 100  # Around 4300 chars
        req = RiskRequest(text=valid_text)
        assert req.text == valid_text.strip()

    def test_valid_text_content(self):
        """Test that valid text content is accepted."""
        valid_texts = [
            "The company shows strong revenue growth.",
            "Quarterly earnings exceeded expectations by 15%.",
            "Risk factors include market volatility and competition.",
            "Management discussed supply chain challenges.",
        ]
        
        for text in valid_texts:
            req = RiskRequest(text=text)
            assert req.text == text


class TestEdgarClientValidation:
    """Test security validation in EdgarClient."""

    def test_ticker_validation(self):
        """Test ticker validation in EdgarClient."""
        client = EdgarClient("TestAgent/1.0")
        
        # Test valid tickers
        valid_tickers = ['AAPL', 'aapl', '  MSFT  ']  # Should handle case and whitespace
        for ticker in valid_tickers:
            validated = client._validate_ticker(ticker)
            assert validated == ticker.strip().upper()
        
        # Test invalid tickers
        invalid_tickers = [
            '',
            None,
            123,
            'AAPL1',
            'GOOGLE',  # Too long
            'AA-PL',
            'AA<script>',
        ]
        
        for ticker in invalid_tickers:
            with pytest.raises(ValueError):
                client._validate_ticker(ticker)

    def test_cik_validation(self):
        """Test CIK validation in EdgarClient."""
        client = EdgarClient("TestAgent/1.0")
        
        # Test valid CIKs
        valid_ciks = ['123456789', '0000123456', '1234567890']
        for cik in valid_ciks:
            validated = client._validate_cik(cik)
            assert len(validated) == 10
            assert validated.isdigit()
        
        # Test invalid CIKs
        invalid_ciks = [
            '',
            None,
            'abc123',
            '123<script>',
            '123-456-789',
        ]
        
        for cik in invalid_ciks:
            with pytest.raises(ValueError):
                client._validate_cik(cik)

    def test_accession_number_validation(self):
        """Test accession number validation in EdgarClient."""
        client = EdgarClient("TestAgent/1.0")
        
        # Test valid accession numbers
        valid_accessions = ['1234567890-12-123456', '0000123456-23-654321']
        for accession in valid_accessions:
            validated = client._validate_accession_number(accession)
            assert validated == accession
        
        # Test invalid accession numbers
        invalid_accessions = [
            '',
            None,
            '123456789-12-123456',  # First part too short
            '12345678901-12-123456',  # First part too long
            '1234567890-1-123456',   # Middle part too short
            '1234567890-123-123456', # Middle part too long
            '1234567890-12-12345',   # Last part too short
            '1234567890-12-1234567', # Last part too long
            '1234567890-ab-123456',  # Non-numeric
            '1234567890<script>',    # Script injection
        ]
        
        for accession in invalid_accessions:
            with pytest.raises(ValueError):
                client._validate_accession_number(accession)

    def test_get_recent_filings_input_validation(self, monkeypatch):
        """Test input validation in get_recent_filings method."""
        client = EdgarClient("TestAgent/1.0")
        
        # Mock the ticker_to_cik method to avoid actual API calls
        def mock_ticker_to_cik(ticker):
            return "1234567890"
        
        def mock_get(url):
            class MockResponse:
                def json(self):
                    return {"filings": {"recent": {}}}
            return MockResponse()
        
        monkeypatch.setattr(client, 'ticker_to_cik', mock_ticker_to_cik)
        monkeypatch.setattr(client, '_get', mock_get)
        
        # Test invalid form_type
        with pytest.raises(ValueError):
            client.get_recent_filings("AAPL", form_type="<script>alert('xss')</script>")
        
        # Test invalid limit values
        with pytest.raises(ValueError):
            client.get_recent_filings("AAPL", limit=0)
        
        with pytest.raises(ValueError):
            client.get_recent_filings("AAPL", limit=101)
        
        with pytest.raises(ValueError):
            client.get_recent_filings("AAPL", limit="invalid")


class TestSecurityIntegration:
    """Integration tests for security measures."""

    def test_end_to_end_validation_flow(self):
        """Test that validation works across the entire request flow."""
        # Test that a malicious request gets blocked at validation stage
        with pytest.raises(ValidationError):
            QueryRequest(
                question='<script>alert("xss")</script>',
                ticker='<script>',
                form_type='<script>'
            )
        
        # Test that a valid request passes validation
        valid_request = QueryRequest(
            question="What is the company's revenue growth?",
            ticker="AAPL",
            form_type="10-K"
        )
        assert valid_request.ticker == "AAPL"
        assert valid_request.form_type == "10-K"
        assert "revenue growth" in valid_request.question

    def test_input_sanitization_preserves_functionality(self):
        """Test that security measures don't break legitimate use cases."""
        # Test common legitimate inputs
        legitimate_requests = [
            {
                "question": "What are the main risk factors?",
                "ticker": "AAPL",
                "form_type": "10-K"
            },
            {
                "question": "How has revenue changed year-over-year?",
                "ticker": "MSFT",
                "form_type": "10-Q"
            },
            {
                "question": "What is management's outlook for Q4?",
                "ticker": "GOOGL",
                "form_type": "8-K"
            }
        ]
        
        for req_data in legitimate_requests:
            req = QueryRequest(**req_data)
            assert req.ticker == req_data["ticker"]
            assert req.form_type == req_data["form_type"]
            assert req.question == req_data["question"]