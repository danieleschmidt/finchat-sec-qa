"""Utilities for analyzing questions across multiple filings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .qa_engine import FinancialQAEngine
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompanyAnswer:
    """Answer extracted from a company's filing."""

    doc_id: str
    answer: str


def _process_single_document(doc_id: str, document_text: str, question: str) -> CompanyAnswer:
    """Process a single document with its own QA engine instance."""
    logger.debug("Processing document %s", doc_id)
    
    # Create dedicated QA engine for this document
    engine = FinancialQAEngine()
    engine.add_document(doc_id, document_text)
    
    # Get answer for this specific document
    answer, _ = engine.answer_with_citations(question)
    return CompanyAnswer(doc_id=doc_id, answer=answer)


def compare_question_across_filings(
    question: str, documents: Dict[str, str], max_workers: int = None
) -> List[CompanyAnswer]:
    """Return answers for each filing given a question using parallel processing.
    
    Args:
        question: The question to ask each document
        documents: Dictionary mapping document IDs to document text
        max_workers: Maximum number of parallel workers (default: min(4, len(documents)))
    
    Returns:
        List of CompanyAnswer objects with results from each document
    """
    if not question:
        raise ValueError("question must be provided")
    if not documents:
        raise ValueError("documents must be provided")

    # Auto-configure workers based on document count if not specified
    if max_workers is None:
        max_workers = min(4, len(documents))

    logger.info("Processing %d documents in parallel with %d workers", len(documents), max_workers)
    
    results: List[CompanyAnswer] = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all document processing tasks
        future_to_doc = {
            executor.submit(_process_single_document, doc_id, doc_text, question): doc_id
            for doc_id, doc_text in documents.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_doc):
            doc_id = future_to_doc[future]
            try:
                result = future.result()
                results.append(result)
                logger.debug("Completed processing for document %s", doc_id)
            except Exception as e:
                logger.error("Error processing document %s: %s", doc_id, e)
                # Add error result so we don't lose track of failed documents
                results.append(CompanyAnswer(doc_id=doc_id, answer=f"Error: {str(e)}"))
    
    logger.info("Completed parallel processing of %d documents", len(results))
    return results


def compare_question_across_filings_sequential(
    question: str, documents: Dict[str, str]
) -> List[CompanyAnswer]:
    """Sequential version for comparison/fallback - DEPRECATED.
    
    This is the original sequential implementation kept for backwards compatibility
    and performance comparison. Use compare_question_across_filings() instead.
    """
    if not question:
        raise ValueError("question must be provided")
    if not documents:
        raise ValueError("documents must be provided")

    logger.debug("Processing %d documents sequentially", len(documents))
    
    # Create single QA engine instance
    engine = FinancialQAEngine()
    
    # Add all documents at once using bulk operations
    documents_list = [(doc_id, text) for doc_id, text in documents.items()]
    engine.add_documents(documents_list)
    
    # Query each document sequentially
    results: List[CompanyAnswer] = []
    for doc_id in documents.keys():
        logger.debug("Querying document %s", doc_id)
        answer, _ = engine.answer_with_citations(question)
        results.append(CompanyAnswer(doc_id=doc_id, answer=answer))
    
    return results
