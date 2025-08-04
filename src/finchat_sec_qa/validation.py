"""Shared validation utilities for input sanitization and security."""
from __future__ import annotations

import re

from .config import get_config

# Shared dangerous patterns for XSS protection
DANGEROUS_PATTERNS = [
    '<script', 'javascript:', 'vbscript:',
    'onload=', 'onerror=', 'onclick='
]


def validate_text_safety(text: str, field_name: str = "text") -> str:
    """Validate text input for XSS and other security issues.
    
    Args:
        text: The text to validate
        field_name: Name of the field for error messages
        
    Returns:
        The validated and cleaned text
        
    Raises:
        ValueError: If text is invalid or contains dangerous patterns
    """
    if not isinstance(text, str):
        raise ValueError(f'{field_name} must be a non-empty string')

    # Handle empty string case specifically
    if not text:
        raise ValueError(f'{field_name} cannot be empty')

    text = text.strip()
    if not text:
        raise ValueError(f'{field_name} cannot be empty')

    # Check for dangerous patterns (case-insensitive)
    text_lower = text.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern in text_lower:
            raise ValueError(f'{field_name} contains potentially dangerous content')

    return text


def validate_ticker(ticker: str) -> str:
    """Validate and normalize ticker symbol.
    
    Args:
        ticker: The ticker symbol to validate
        
    Returns:
        The validated and normalized ticker symbol (uppercase, stripped)
        
    Raises:
        ValueError: If ticker is invalid, too long, or contains invalid characters
    """
    config = get_config()

    if not isinstance(ticker, str):
        raise ValueError('ticker must be a non-empty string')

    if not ticker:
        raise ValueError('ticker must be a non-empty string')

    ticker = ticker.strip().upper()

    # Check if empty after stripping
    if not ticker:
        raise ValueError('ticker must be a non-empty string')

    # Check for injection characters first
    if any(char in ticker for char in ['<', '>', '&', '"', "'", '/', '\\', '%']):
        raise ValueError('ticker contains invalid characters')

    # Strict validation: 1-MAX_TICKER_LENGTH uppercase letters only
    pattern = f'^[A-Z]{{1,{config.MAX_TICKER_LENGTH}}}$'
    if not re.match(pattern, ticker):
        raise ValueError(
            f'ticker must be 1-{config.MAX_TICKER_LENGTH} uppercase letters only (A-Z)'
        )

    return ticker
