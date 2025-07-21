"""Tests for the shared validation module."""
import pytest
from unittest.mock import patch, Mock


class TestValidationModule:
    """Test cases for shared validation utilities."""

    def test_validate_text_safety_valid_text(self):
        """Test text validation with safe input."""
        from finchat_sec_qa.validation import validate_text_safety
        
        safe_text = "This is a safe question about financial performance."
        result = validate_text_safety(safe_text)
        
        assert result == safe_text

    def test_validate_text_safety_strips_whitespace(self):
        """Test text validation strips leading/trailing whitespace."""
        from finchat_sec_qa.validation import validate_text_safety
        
        text_with_spaces = "  Safe question  "
        result = validate_text_safety(text_with_spaces)
        
        assert result == "Safe question"

    def test_validate_text_safety_empty_string(self):
        """Test text validation rejects empty strings."""
        from finchat_sec_qa.validation import validate_text_safety
        
        with pytest.raises(ValueError, match="text cannot be empty"):
            validate_text_safety("")

    def test_validate_text_safety_whitespace_only(self):
        """Test text validation rejects whitespace-only strings."""
        from finchat_sec_qa.validation import validate_text_safety
        
        with pytest.raises(ValueError, match="text cannot be empty"):
            validate_text_safety("   ")

    def test_validate_text_safety_none_input(self):
        """Test text validation rejects None input."""
        from finchat_sec_qa.validation import validate_text_safety
        
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            validate_text_safety(None)

    def test_validate_text_safety_non_string_input(self):
        """Test text validation rejects non-string input."""
        from finchat_sec_qa.validation import validate_text_safety
        
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            validate_text_safety(123)

    def test_validate_text_safety_custom_field_name(self):
        """Test text validation uses custom field name in error messages."""
        from finchat_sec_qa.validation import validate_text_safety
        
        with pytest.raises(ValueError, match="question must be a non-empty string"):
            validate_text_safety(None, field_name="question")

    @pytest.mark.parametrize("dangerous_input", [
        "This contains <script>alert('xss')</script> code",
        "Click here javascript:alert('hack')",
        "Dangerous vbscript:alert('bad')",
        "Image with onload=alert('xss')",
        "Input with onerror=alert('hack')",
        "Button with onclick=alert('bad')",
        "Mixed case <SCRIPT>alert('xss')</SCRIPT>",
        "Uppercase JAVASCRIPT:alert('hack')"
    ])
    def test_validate_text_safety_dangerous_patterns(self, dangerous_input):
        """Test text validation rejects dangerous XSS patterns."""
        from finchat_sec_qa.validation import validate_text_safety
        
        with pytest.raises(ValueError, match="contains potentially dangerous content"):
            validate_text_safety(dangerous_input)

    def test_validate_text_safety_case_insensitive_detection(self):
        """Test that dangerous pattern detection is case-insensitive."""
        from finchat_sec_qa.validation import validate_text_safety
        
        dangerous_variations = [
            "Test <Script>alert('xss')</Script>",
            "Test <SCRIPT>alert('xss')</SCRIPT>",
            "Test JAVASCRIPT:alert('hack')",
            "Test JavaScript:alert('hack')"
        ]
        
        for variation in dangerous_variations:
            with pytest.raises(ValueError, match="contains potentially dangerous content"):
                validate_text_safety(variation)

    def test_validate_ticker_valid_ticker(self):
        """Test ticker validation with valid input."""
        from finchat_sec_qa.validation import validate_ticker
        
        with patch('finchat_sec_qa.validation.get_config') as mock_config:
            mock_config.return_value.MAX_TICKER_LENGTH = 5
            
            result = validate_ticker("aapl")
            assert result == "AAPL"

    def test_validate_ticker_converts_to_uppercase(self):
        """Test ticker validation converts to uppercase."""
        from finchat_sec_qa.validation import validate_ticker
        
        with patch('finchat_sec_qa.validation.get_config') as mock_config:
            mock_config.return_value.MAX_TICKER_LENGTH = 5
            
            result = validate_ticker("msft")
            assert result == "MSFT"

    def test_validate_ticker_strips_whitespace(self):
        """Test ticker validation strips whitespace."""
        from finchat_sec_qa.validation import validate_ticker
        
        with patch('finchat_sec_qa.validation.get_config') as mock_config:
            mock_config.return_value.MAX_TICKER_LENGTH = 5
            
            result = validate_ticker("  googl  ")
            assert result == "GOOGL"

    def test_validate_ticker_empty_string(self):
        """Test ticker validation rejects empty strings."""
        from finchat_sec_qa.validation import validate_ticker
        
        with pytest.raises(ValueError, match="ticker must be a non-empty string"):
            validate_ticker("")

    def test_validate_ticker_none_input(self):
        """Test ticker validation rejects None input."""
        from finchat_sec_qa.validation import validate_ticker
        
        with pytest.raises(ValueError, match="ticker must be a non-empty string"):
            validate_ticker(None)

    def test_validate_ticker_non_string_input(self):
        """Test ticker validation rejects non-string input."""
        from finchat_sec_qa.validation import validate_ticker
        
        with pytest.raises(ValueError, match="ticker must be a non-empty string"):
            validate_ticker(123)

    def test_validate_ticker_too_long(self):
        """Test ticker validation rejects tickers that are too long."""
        from finchat_sec_qa.validation import validate_ticker
        
        with patch('finchat_sec_qa.validation.get_config') as mock_config:
            mock_config.return_value.MAX_TICKER_LENGTH = 5
            
            with pytest.raises(ValueError, match="ticker must be 1-5 uppercase letters only"):
                validate_ticker("TOOLONG")

    def test_validate_ticker_invalid_characters(self):
        """Test ticker validation rejects non-alphabetic characters."""
        from finchat_sec_qa.validation import validate_ticker
        
        with patch('finchat_sec_qa.validation.get_config') as mock_config:
            mock_config.return_value.MAX_TICKER_LENGTH = 10
            
            invalid_tickers = ["AAPL123", "AA-PL", "AA.PL", "AA PL"]
            
            for ticker in invalid_tickers:
                with pytest.raises(ValueError, match="ticker must be 1-10 uppercase letters only"):
                    validate_ticker(ticker)

    def test_validate_ticker_injection_characters(self):
        """Test ticker validation rejects potential injection characters."""
        from finchat_sec_qa.validation import validate_ticker
        
        with patch('finchat_sec_qa.validation.get_config') as mock_config:
            mock_config.return_value.MAX_TICKER_LENGTH = 10
            
            dangerous_chars = ['<', '>', '&', '"', "'", '/', '\\', '%']
            
            for char in dangerous_chars:
                ticker_with_char = f"AA{char}PL"
                with pytest.raises(ValueError, match="ticker contains invalid characters"):
                    validate_ticker(ticker_with_char)

    def test_validate_ticker_minimum_length(self):
        """Test ticker validation enforces minimum length."""
        from finchat_sec_qa.validation import validate_ticker
        
        with patch('finchat_sec_qa.validation.get_config') as mock_config:
            mock_config.return_value.MAX_TICKER_LENGTH = 5
            
            # Empty after stripping should fail
            with pytest.raises(ValueError, match="ticker must be a non-empty string"):
                validate_ticker("   ")

    def test_validate_ticker_boundary_cases(self):
        """Test ticker validation boundary cases."""
        from finchat_sec_qa.validation import validate_ticker
        
        with patch('finchat_sec_qa.validation.get_config') as mock_config:
            mock_config.return_value.MAX_TICKER_LENGTH = 5
            
            # Single character should be valid
            assert validate_ticker("A") == "A"
            
            # Max length should be valid
            assert validate_ticker("ABCDE") == "ABCDE"

    def test_dangerous_patterns_constant(self):
        """Test that DANGEROUS_PATTERNS constant contains expected patterns."""
        from finchat_sec_qa.validation import DANGEROUS_PATTERNS
        
        expected_patterns = [
            '<script', 'javascript:', 'vbscript:', 
            'onload=', 'onerror=', 'onclick='
        ]
        
        assert DANGEROUS_PATTERNS == expected_patterns

    def test_validate_text_safety_integration_with_config(self):
        """Test text validation works with actual config values."""
        from finchat_sec_qa.validation import validate_text_safety
        
        # This should work with any reasonable text
        safe_text = "What is the company's revenue for Q4 2023?"
        result = validate_text_safety(safe_text, field_name="question")
        
        assert result == safe_text

    def test_validate_ticker_integration_with_config(self):
        """Test ticker validation works with actual config values."""
        from finchat_sec_qa.validation import validate_ticker
        from finchat_sec_qa.config import get_config
        
        # Use actual config
        config = get_config()
        max_length = config.MAX_TICKER_LENGTH
        
        # Test with common ticker symbols
        valid_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        for ticker in valid_tickers:
            if len(ticker) <= max_length:
                result = validate_ticker(ticker.lower())
                assert result == ticker.upper()