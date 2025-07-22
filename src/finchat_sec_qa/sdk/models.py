"""Data models for the FinChat SEC QA SDK.

This module defines the typed data structures used by the SDK for API responses
and internal data representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import json


@dataclass
class Citation:
    """Represents a citation from a SEC filing.
    
    Args:
        text: The cited text from the document
        source: The source document (e.g., "AAPL 10-K 2022")
        page: Page number in the document (optional)
        start_pos: Starting character position in the document
        end_pos: Ending character position in the document
    """
    text: str
    source: str
    page: Optional[int] = None
    start_pos: int = 0
    end_pos: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary representation."""
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Citation:
        """Create Citation from dictionary data."""
        return cls(
            text=data["text"],
            source=data["source"],
            page=data.get("page"),
            start_pos=data.get("start_pos", 0),
            end_pos=data.get("end_pos", 0),
        )


@dataclass
class QueryResponse:
    """Response from a query to the FinChat SEC QA service.
    
    Args:
        answer: The generated answer to the question
        citations: List of citations supporting the answer
    """
    answer: str
    citations: List[Citation]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary representation."""
        return {
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryResponse:
        """Create QueryResponse from dictionary data."""
        citations = [
            Citation.from_dict(citation_data) 
            for citation_data in data.get("citations", [])
        ]
        return cls(
            answer=data["answer"],
            citations=citations,
        )
    
    def __str__(self) -> str:
        """String representation showing answer and citation count."""
        citation_count = len(self.citations)
        return f"QueryResponse(answer='{self.answer[:100]}...', citations={citation_count})"


@dataclass
class RiskAnalysisResponse:
    """Response from risk analysis of financial text.
    
    Args:
        sentiment: Overall sentiment (positive, negative, neutral)
        flags: List of risk/opportunity flags identified
    """
    sentiment: str
    flags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary representation."""
        return {
            "sentiment": self.sentiment,
            "flags": self.flags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RiskAnalysisResponse:
        """Create RiskAnalysisResponse from dictionary data."""
        return cls(
            sentiment=data["sentiment"],
            flags=data.get("flags", []),
        )


@dataclass
class HealthCheckResponse:
    """Response from service health check.
    
    Args:
        status: Overall service status (healthy, degraded, unhealthy)
        version: Service version
        timestamp: Timestamp of the health check (optional)
        services: Status of individual services (optional)
    """
    status: str
    version: str
    timestamp: Optional[str] = None
    services: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary representation."""
        result = {
            "status": self.status,
            "version": self.version,
        }
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.services:
            result["services"] = self.services
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HealthCheckResponse:
        """Create HealthCheckResponse from dictionary data."""
        return cls(
            status=data["status"],
            version=data["version"],
            timestamp=data.get("timestamp"),
            services=data.get("services"),
        )
    
    @property
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self.status == "healthy"


@dataclass
class ClientConfig:
    """Configuration for FinChat SDK clients.
    
    Args:
        base_url: Base URL of the FinChat API
        timeout: Request timeout in seconds
        api_key: API key for authentication (optional)
        max_retries: Maximum number of retries for failed requests
        retry_delay: Delay between retries in seconds
        user_agent: User agent string for requests
    """
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    api_key: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    user_agent: str = "FinChat-SDK/1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary representation."""
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "api_key": self.api_key,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "user_agent": self.user_agent,
        }
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for requests."""
        headers = {
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers