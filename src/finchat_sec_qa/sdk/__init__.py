"""FinChat SEC QA SDK for external developers.

This module provides a clean, typed interface for interacting with the FinChat SEC QA service.
It includes both synchronous and asynchronous clients for flexibility.

Example usage:

    # Synchronous client
    from finchat_sec_qa.sdk import FinChatClient
    
    with FinChatClient(base_url="https://api.finchat.example.com") as client:
        result = client.query(
            question="What was Apple's revenue in 2022?",
            ticker="AAPL",
            form_type="10-K"
        )
        print(f"Answer: {result.answer}")
        for citation in result.citations:
            print(f"Citation: {citation.text} from {citation.source}")

    # Asynchronous client
    from finchat_sec_qa.sdk import AsyncFinChatClient
    
    async with AsyncFinChatClient(base_url="https://api.finchat.example.com") as client:
        result = await client.query(
            question="What was Apple's revenue in 2022?",
            ticker="AAPL",
            form_type="10-K"
        )
        print(f"Answer: {result.answer}")
"""

from .client import FinChatClient, AsyncFinChatClient
from .models import (
    QueryResponse,
    RiskAnalysisResponse, 
    HealthCheckResponse,
    Citation,
)
from .exceptions import (
    FinChatAPIError,
    FinChatValidationError,
    FinChatNotFoundError,
    FinChatTimeoutError,
)

__all__ = [
    # Clients
    "FinChatClient",
    "AsyncFinChatClient",
    
    # Response models
    "QueryResponse",
    "RiskAnalysisResponse",
    "HealthCheckResponse",
    "Citation",
    
    # Exceptions
    "FinChatAPIError",
    "FinChatValidationError", 
    "FinChatNotFoundError",
    "FinChatTimeoutError",
]