"""FinChat SEC QA SDK client implementations.

This module provides synchronous and asynchronous clients for interacting with
the FinChat SEC QA service API.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx

from .exceptions import (
    FinChatAPIError,
    FinChatConfigurationError,
    FinChatConnectionError,
)
from .models import (
    ClientConfig,
    HealthCheckResponse,
    QueryResponse,
    RiskAnalysisResponse,
)


class FinChatClient:
    """Synchronous client for the FinChat SEC QA API.
    
    This client provides a simple interface for querying SEC filings and
    performing risk analysis using the FinChat service.
    
    Example:
        >>> client = FinChatClient(base_url="https://api.finchat.example.com")
        >>> result = client.query("What was Apple's revenue?", "AAPL", "10-K")
        >>> print(result.answer)
        >>> 
        >>> # Using context manager
        >>> with FinChatClient() as client:
        ...     result = client.query("What was Apple's revenue?", "AAPL", "10-K")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        user_agent: str = "FinChat-SDK/1.0",
    ) -> None:
        """Initialize the FinChat client.
        
        Args:
            base_url: Base URL of the FinChat API
            timeout: Request timeout in seconds
            api_key: API key for authentication (optional)
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            user_agent: User agent string for requests
        """
        self.config = ClientConfig(
            base_url=base_url,
            timeout=timeout,
            api_key=api_key,
            max_retries=max_retries,
            retry_delay=retry_delay,
            user_agent=user_agent,
        )
        self._client: Optional[httpx.Client] = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate client configuration."""
        if not self.config.base_url:
            raise FinChatConfigurationError("base_url is required")

        if not self.config.base_url.startswith(("http://", "https://")):
            raise FinChatConfigurationError("base_url must start with http:// or https://")

        if self.config.timeout <= 0:
            raise FinChatConfigurationError("timeout must be positive")

    @property
    def base_url(self) -> str:
        """Get the base URL."""
        return self.config.base_url

    @property
    def timeout(self) -> int:
        """Get the timeout."""
        return self.config.timeout

    @property
    def api_key(self) -> Optional[str]:
        """Get the API key."""
        return self.config.api_key

    def __enter__(self) -> FinChatClient:
        """Context manager entry."""
        self._client = httpx.Client(
            timeout=self.config.timeout,
            headers=self.config.get_headers(),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self._client:
            self._client.close()
            self._client = None

    def _get_client(self) -> httpx.Client:
        """Get HTTP client, creating one if needed."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.config.timeout,
                headers=self.config.get_headers(),
            )
        return self._client

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with retries and error handling."""
        url = urljoin(self.config.base_url, endpoint)
        client = self._get_client()

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = client.get(url, params=params)
                elif method.upper() == "POST":
                    response = client.post(url, json=data, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Handle HTTP errors
                if response.status_code >= 400:
                    raise FinChatAPIError.from_response(response)

                return response.json()

            except httpx.RequestError as e:
                last_exception = FinChatConnectionError.from_httpx_error(e)
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    raise last_exception

            except FinChatAPIError:
                # Don't retry API errors
                raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise FinChatConnectionError("Failed to make request after retries")

    def query(
        self,
        question: str,
        ticker: str,
        form_type: str = "10-K",
        limit: int = 1,
    ) -> QueryResponse:
        """Query SEC filings for information about a company.
        
        Args:
            question: The question to ask about the company
            ticker: Company ticker symbol (e.g., "AAPL")
            form_type: Type of SEC form to query (e.g., "10-K", "10-Q")
            limit: Maximum number of documents to search
            
        Returns:
            QueryResponse containing the answer and citations
            
        Raises:
            FinChatValidationError: If the input parameters are invalid
            FinChatNotFoundError: If the requested filing is not found
            FinChatAPIError: For other API errors
            FinChatConnectionError: If unable to connect to the service
        """
        data = {
            "question": question,
            "ticker": ticker,
            "form_type": form_type,
            "limit": limit,
        }

        response_data = self._make_request("POST", "/query", data=data)
        return QueryResponse.from_dict(response_data)

    def analyze_risk(self, text: str) -> RiskAnalysisResponse:
        """Analyze the risk/sentiment of financial text.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            RiskAnalysisResponse containing sentiment and risk flags
            
        Raises:
            FinChatValidationError: If the text is invalid
            FinChatAPIError: For other API errors
            FinChatConnectionError: If unable to connect to the service
        """
        data = {"text": text}

        response_data = self._make_request("POST", "/risk", data=data)
        return RiskAnalysisResponse.from_dict(response_data)

    def health_check(self) -> HealthCheckResponse:
        """Check the health status of the FinChat service.
        
        Returns:
            HealthCheckResponse containing service status information
            
        Raises:
            FinChatConnectionError: If unable to connect to the service
        """
        response_data = self._make_request("GET", "/health")
        return HealthCheckResponse.from_dict(response_data)

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


class AsyncFinChatClient:
    """Asynchronous client for the FinChat SEC QA API.
    
    This client provides an async interface for querying SEC filings and
    performing risk analysis using the FinChat service.
    
    Example:
        >>> async with AsyncFinChatClient(base_url="https://api.finchat.example.com") as client:
        ...     result = await client.query("What was Apple's revenue?", "AAPL", "10-K")
        ...     print(result.answer)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        user_agent: str = "FinChat-SDK/1.0",
    ) -> None:
        """Initialize the async FinChat client.
        
        Args:
            base_url: Base URL of the FinChat API
            timeout: Request timeout in seconds
            api_key: API key for authentication (optional)
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            user_agent: User agent string for requests
        """
        self.config = ClientConfig(
            base_url=base_url,
            timeout=timeout,
            api_key=api_key,
            max_retries=max_retries,
            retry_delay=retry_delay,
            user_agent=user_agent,
        )
        self._client: Optional[httpx.AsyncClient] = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate client configuration."""
        if not self.config.base_url:
            raise FinChatConfigurationError("base_url is required")

        if not self.config.base_url.startswith(("http://", "https://")):
            raise FinChatConfigurationError("base_url must start with http:// or https://")

        if self.config.timeout <= 0:
            raise FinChatConfigurationError("timeout must be positive")

    @property
    def base_url(self) -> str:
        """Get the base URL."""
        return self.config.base_url

    @property
    def timeout(self) -> int:
        """Get the timeout."""
        return self.config.timeout

    @property
    def api_key(self) -> Optional[str]:
        """Get the API key."""
        return self.config.api_key

    async def __aenter__(self) -> AsyncFinChatClient:
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers=self.config.get_headers(),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client, creating one if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout,
                headers=self.config.get_headers(),
            )
        return self._client

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make async HTTP request with retries and error handling."""
        url = urljoin(self.config.base_url, endpoint)
        client = self._get_client()

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = await client.get(url, params=params)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Handle HTTP errors
                if response.status_code >= 400:
                    raise FinChatAPIError.from_response(response)

                return response.json()

            except httpx.RequestError as e:
                last_exception = FinChatConnectionError.from_httpx_error(e)
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                else:
                    raise last_exception

            except FinChatAPIError:
                # Don't retry API errors
                raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise FinChatConnectionError("Failed to make request after retries")

    async def query(
        self,
        question: str,
        ticker: str,
        form_type: str = "10-K",
        limit: int = 1,
    ) -> QueryResponse:
        """Query SEC filings for information about a company.
        
        Args:
            question: The question to ask about the company
            ticker: Company ticker symbol (e.g., "AAPL")
            form_type: Type of SEC form to query (e.g., "10-K", "10-Q")
            limit: Maximum number of documents to search
            
        Returns:
            QueryResponse containing the answer and citations
            
        Raises:
            FinChatValidationError: If the input parameters are invalid
            FinChatNotFoundError: If the requested filing is not found
            FinChatAPIError: For other API errors
            FinChatConnectionError: If unable to connect to the service
        """
        data = {
            "question": question,
            "ticker": ticker,
            "form_type": form_type,
            "limit": limit,
        }

        response_data = await self._make_request("POST", "/query", data=data)
        return QueryResponse.from_dict(response_data)

    async def analyze_risk(self, text: str) -> RiskAnalysisResponse:
        """Analyze the risk/sentiment of financial text.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            RiskAnalysisResponse containing sentiment and risk flags
            
        Raises:
            FinChatValidationError: If the text is invalid
            FinChatAPIError: For other API errors
            FinChatConnectionError: If unable to connect to the service
        """
        data = {"text": text}

        response_data = await self._make_request("POST", "/risk", data=data)
        return RiskAnalysisResponse.from_dict(response_data)

    async def health_check(self) -> HealthCheckResponse:
        """Check the health status of the FinChat service.
        
        Returns:
            HealthCheckResponse containing service status information
            
        Raises:
            FinChatConnectionError: If unable to connect to the service
        """
        response_data = await self._make_request("GET", "/health")
        return HealthCheckResponse.from_dict(response_data)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
