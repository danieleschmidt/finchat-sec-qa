"""Exceptions for the FinChat SEC QA SDK.

This module defines custom exceptions for different error conditions that can occur
when using the FinChat SDK.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class FinChatAPIError(Exception):
    """Base exception for FinChat API errors.
    
    Args:
        message: Error message
        status_code: HTTP status code (if applicable)
        response_data: Response data from the API (if available)
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}

    def __str__(self) -> str:
        """String representation of the error."""
        if self.status_code:
            return f"FinChat API Error {self.status_code}: {self.message}"
        return f"FinChat API Error: {self.message}"

    @classmethod
    def from_response(cls, response: httpx.Response) -> FinChatAPIError:
        """Create exception from HTTP response."""
        try:
            response_data = response.json()
            message = response_data.get("detail", f"HTTP {response.status_code}")
        except (ValueError, KeyError):
            message = f"HTTP {response.status_code}"
            response_data = {}

        # Return specific exception types based on status code
        if response.status_code == 400:
            return FinChatValidationError(message, response.status_code, response_data)
        elif response.status_code == 404:
            return FinChatNotFoundError(message, response.status_code, response_data)
        elif response.status_code == 408 or response.status_code == 504:
            return FinChatTimeoutError(message, response.status_code, response_data)
        else:
            return cls(message, response.status_code, response_data)


class FinChatValidationError(FinChatAPIError):
    """Exception raised for validation errors (HTTP 400).
    
    This is typically raised when the request parameters are invalid,
    such as malformed ticker symbols or questions that are too long.
    """
    pass


class FinChatNotFoundError(FinChatAPIError):
    """Exception raised when requested resource is not found (HTTP 404).
    
    This is typically raised when a requested SEC filing or company 
    cannot be found.
    """
    pass


class FinChatTimeoutError(FinChatAPIError):
    """Exception raised when a request times out.
    
    This can occur due to network timeouts or server-side processing timeouts.
    """
    pass


class FinChatConnectionError(FinChatAPIError):
    """Exception raised when unable to connect to the FinChat API.
    
    This typically indicates network connectivity issues or that the
    FinChat service is unavailable.
    """

    def __init__(self, message: str, original_exception: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_exception = original_exception

    @classmethod
    def from_httpx_error(cls, error: httpx.RequestError) -> FinChatConnectionError:
        """Create connection error from httpx RequestError."""
        return cls(
            f"Failed to connect to FinChat API: {str(error)}",
            original_exception=error
        )


class FinChatConfigurationError(FinChatAPIError):
    """Exception raised for SDK configuration errors.
    
    This is raised when the SDK is misconfigured, such as invalid
    base URLs or missing required configuration.
    """
    pass
