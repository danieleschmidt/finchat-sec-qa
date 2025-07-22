"""Tests for shared EDGAR validation utilities."""

import pytest

from finchat_sec_qa.edgar_validation import validate_ticker, validate_cik, validate_accession_number


class TestTickerValidation:
    """Test ticker validation utility."""
    
    def test_valid_tickers(self):
        """Test valid ticker symbols."""
        assert validate_ticker("AAPL") == "AAPL"
        assert validate_ticker("MSFT") == "MSFT"
        assert validate_ticker("GOOGL") == "GOOGL"
        assert validate_ticker("T") == "T"  # Single letter
        
    def test_ticker_normalization(self):
        """Test ticker normalization (whitespace, case)."""
        assert validate_ticker("  aapl  ") == "AAPL"
        assert validate_ticker("msft") == "MSFT"
        assert validate_ticker("Googl") == "GOOGL"
        
    def test_invalid_tickers(self):
        """Test invalid ticker symbols."""
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            validate_ticker("")
            
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            validate_ticker(None)
            
        with pytest.raises(ValueError, match="Invalid ticker format"):
            validate_ticker("123ABC")
            
        with pytest.raises(ValueError, match="Invalid ticker format"):
            validate_ticker("TOOLONG")  # More than 5 characters
            
        with pytest.raises(ValueError, match="Invalid ticker format"):
            validate_ticker("ABC-D")  # Special characters
            
        with pytest.raises(ValueError, match="Invalid ticker format"):
            validate_ticker("BRK.A")  # Dots not allowed
            
    def test_ticker_type_validation(self):
        """Test ticker type validation."""
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            validate_ticker(123)
            
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            validate_ticker([])


class TestCikValidation:
    """Test CIK validation utility."""
    
    def test_valid_ciks(self):
        """Test valid CIK values."""
        assert validate_cik("320193") == "0000320193"
        assert validate_cik("0000320193") == "0000320193"
        assert validate_cik("789019") == "0000789019"
        
    def test_cik_normalization(self):
        """Test CIK normalization (padding, cleaning)."""
        assert validate_cik("320193") == "0000320193"
        assert validate_cik("  320193  ") == "0000320193"
        assert validate_cik("CIK-320193") == "0000320193"  # Remove non-digits
        
    def test_invalid_ciks(self):
        """Test invalid CIK values."""
        with pytest.raises(ValueError, match="CIK must be a non-empty string"):
            validate_cik("")
            
        with pytest.raises(ValueError, match="CIK must be a non-empty string"):
            validate_cik(None)
            
        with pytest.raises(ValueError, match="Invalid CIK format"):
            validate_cik("INVALID")
            
        with pytest.raises(ValueError, match="Invalid CIK format"):
            validate_cik("ABC-DEF")
            
    def test_cik_type_validation(self):
        """Test CIK type validation."""
        with pytest.raises(ValueError, match="CIK must be a non-empty string"):
            validate_cik(123)


class TestAccessionNumberValidation:
    """Test accession number validation utility."""
    
    def test_valid_accession_numbers(self):
        """Test valid accession numbers."""
        assert validate_accession_number("0000320193-23-000007") == "0000320193-23-000007"
        assert validate_accession_number("0000789019-22-000118") == "0000789019-22-000118"
        
    def test_accession_number_normalization(self):
        """Test accession number normalization."""
        assert validate_accession_number("  0000320193-23-000007  ") == "0000320193-23-000007"
        
    def test_invalid_accession_numbers(self):
        """Test invalid accession numbers."""
        with pytest.raises(ValueError, match="Accession number must be a non-empty string"):
            validate_accession_number("")
            
        with pytest.raises(ValueError, match="Accession number must be a non-empty string"):
            validate_accession_number(None)
            
        with pytest.raises(ValueError, match="Invalid accession number format"):
            validate_accession_number("invalid-format")
            
        with pytest.raises(ValueError, match="Invalid accession number format"):
            validate_accession_number("123-45-678")  # Too short
            
        with pytest.raises(ValueError, match="Invalid accession number format"):
            validate_accession_number("0000320193-23-0000078")  # Too many digits
            
    def test_accession_number_type_validation(self):
        """Test accession number type validation.""" 
        with pytest.raises(ValueError, match="Accession number must be a non-empty string"):
            validate_accession_number(123)