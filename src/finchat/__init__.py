"""FinChat SEC QA package."""

from __future__ import annotations

__all__ = ["echo"]


def echo(text: str | None) -> str:
    """Return the provided text or an empty string if None."""
    return text or ""
