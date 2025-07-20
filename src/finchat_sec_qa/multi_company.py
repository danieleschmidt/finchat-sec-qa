"""Utilities for analyzing questions across multiple filings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .qa_engine import FinancialQAEngine
import logging

logger = logging.getLogger(__name__)


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

    logger.debug("Processing %d documents with single QA engine", len(documents))
    
    # Create single QA engine instance for better performance
    engine = FinancialQAEngine()
    
    # Add all documents at once using bulk operations
    documents_list = [(doc_id, text) for doc_id, text in documents.items()]
    engine.add_documents(documents_list)
    
    # Query each document to get answers
    results: List[CompanyAnswer] = []
    for doc_id in documents.keys():
        logger.debug("Querying document %s", doc_id)
        answer, _ = engine.answer_with_citations(question)
        results.append(CompanyAnswer(doc_id=doc_id, answer=answer))
    
    return results
