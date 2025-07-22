"""Shared validation utilities for SEC EDGAR data."""

import re
from typing import Union


def validate_ticker(ticker: Union[str, None]) -> str:
    """Validate and sanitize ticker symbol.
    
    Args:
        ticker: Raw ticker symbol input
        
    Returns:
        Validated and normalized ticker symbol
        
    Raises:
        ValueError: If ticker is invalid or malformed
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")
    
    # Normalize: strip whitespace, convert to uppercase  
    ticker = ticker.strip().upper()
    
    # Validate ticker format: 1-5 uppercase letters only
    if not re.match(r'^[A-Z]{1,5}$', ticker):
        raise ValueError(f"Invalid ticker format: {ticker}. Must be 1-5 uppercase letters only")
    
    return ticker


def validate_cik(cik: Union[str, None]) -> str:
    """Validate and sanitize CIK (Central Index Key).
    
    Args:
        cik: Raw CIK input
        
    Returns:
        Validated and normalized CIK (zero-padded to 10 digits)
        
    Raises:
        ValueError: If CIK is invalid or malformed
    """
    if not cik or not isinstance(cik, str):
        raise ValueError("CIK must be a non-empty string")
    
    # Remove any non-digit characters and validate
    cik_digits = re.sub(r'\D', '', cik.strip())
    if not cik_digits or not cik_digits.isdigit():
        raise ValueError(f"Invalid CIK format: {cik}. Must contain only digits")
    
    # Zero-pad to 10 digits
    return cik_digits.zfill(10)


def validate_accession_number(accession: Union[str, None]) -> str:
    """Validate and sanitize accession number.
    
    Args:
        accession: Raw accession number input
        
    Returns:
        Validated and normalized accession number
        
    Raises:
        ValueError: If accession number is invalid or malformed
    """
    if not accession or not isinstance(accession, str):
        raise ValueError("Accession number must be a non-empty string")
    
    # Normalize whitespace
    accession = accession.strip()
    
    # Validate accession number format: 10 digits, 2 digits, 6 digits with hyphens
    if not re.match(r'^\d{10}-\d{2}-\d{6}$', accession):
        raise ValueError(f"Invalid accession number format: {accession}")
    
    return accession