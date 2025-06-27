"""Utilities for analyzing questions across multiple filings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .qa_engine import FinancialQAEngine


@dataclass
class CompanyAnswer:
    """Answer extracted from a company's filing."""

    doc_id: str
    answer: str


def compare_question_across_filings(
    question: str, documents: Dict[str, str]
) -> List[CompanyAnswer]:
    """Return answers for each filing given a question."""
    if not question:
        raise ValueError("question must be provided")
    if not documents:
        raise ValueError("documents must be provided")

    results: List[CompanyAnswer] = []
    for doc_id, text in documents.items():
        engine = FinancialQAEngine()
        engine.add_document(doc_id, text)
        answer, _ = engine.answer_with_citations(question)
        results.append(CompanyAnswer(doc_id=doc_id, answer=answer))
    return results
