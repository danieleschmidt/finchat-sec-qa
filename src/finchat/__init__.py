"""FinChat SEC QA package."""

from __future__ import annotations

from .agent import FinChatAgent, QueryResult

__all__ = ["echo", "FinChatAgent", "QueryResult"]


def echo(text: str | None) -> str:
    """Return the provided text or an empty string if None."""
    return text or ""
