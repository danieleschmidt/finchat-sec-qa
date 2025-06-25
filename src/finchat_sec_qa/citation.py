from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import re


@dataclass
class Citation:
    """Reference to a text span within a document."""

    doc_id: str
    text: str
    start: int
    end: int
    section: Optional[str] = None
    page: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must be provided")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Invalid text offsets")
        if self.page is not None and self.page <= 0:
            raise ValueError("page must be positive")


def extract_citation_anchors(doc_id: str, html: str) -> List[Citation]:
    """Parse HTML for span tags with section/page attributes."""
    if not doc_id:
        raise ValueError("doc_id must be provided")
    if not html:
        raise ValueError("html must be provided")

    pattern = re.compile(
        r'<span[^>]*data-section="(?P<section>[^"]+)"[^>]*data-page="(?P<page>\d+)"[^>]*>(?P<text>.*?)</span>',
        re.IGNORECASE | re.DOTALL,
    )
    citations: List[Citation] = []
    for match in pattern.finditer(html):
        section = match.group("section")
        page = int(match.group("page"))
        text = match.group("text")
        citations.append(
            Citation(
                doc_id=doc_id,
                text=text,
                start=match.start(),
                end=match.end(),
                section=section,
                page=page,
            )
        )
    return citations
