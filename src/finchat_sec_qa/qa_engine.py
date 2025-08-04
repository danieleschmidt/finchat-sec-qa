from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - imported for type hints
    from .citation import Citation
    from .photonic_bridge import PhotonicBridge, PhotonicEnhancedResult

import logging
import re

from joblib import dump, load
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import get_config
from .validation import validate_text_safety


@dataclass
class DocumentChunk:
    """A slice of a document stored for retrieval."""

    doc_id: str
    text: str
    start_pos: int = 0
    end_pos: int = 0

    def __post_init__(self) -> None:
        """Validate and set default positions if not provided."""
        if self.end_pos == 0:
            self.end_pos = self.start_pos + len(self.text)


class FinancialQAEngine:
    """Simple QA engine using TF-IDF retrieval over filing texts.

    The index is persisted via joblib to avoid pickle security issues.
    Old ``.pkl`` indexes are migrated automatically on startup.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, storage_path: Path | None = None, enable_quantum: bool = True) -> None:
        self.storage_path = storage_path
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunks: List[DocumentChunk] = []
        self.tfidf_matrix: np.ndarray | None = None
        self.valid_chunk_indices: List[int] = []
        self._bulk_mode = False
        self.enable_quantum = enable_quantum
        self._photonic_bridge: Optional[PhotonicBridge] = None
        
        if storage_path and storage_path.exists():
            self.load(storage_path)
        elif storage_path and storage_path.with_suffix('.pkl').exists():
            # migrate old pickle-based index
            self.load(storage_path.with_suffix('.pkl'))
            storage_path.with_suffix('.pkl').unlink()
            self.storage_path = storage_path
            self.save()

    def _chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into overlapping chunks with position tracking.
        
        Returns list of (chunk_text, start_pos, end_pos) tuples.
        Tries to split at sentence boundaries when possible.
        """
        if self._is_single_chunk(text):
            return [(text, 0, len(text))]

        return self._create_overlapping_chunks(text)

    def _is_single_chunk(self, text: str) -> bool:
        """Check if text is small enough to be a single chunk."""
        config = get_config()
        return len(text) <= config.CHUNK_SIZE

    def _create_overlapping_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        """Create overlapping chunks from text, preferring sentence boundaries."""
        config = get_config()
        chunks = []
        start = 0

        while start < len(text):
            chunk = self._create_next_chunk(text, start)
            chunks.append(chunk)

            # Move start position for next chunk with overlap
            if chunk[2] >= len(text):  # end_pos >= text length (final chunk)
                break
            start = chunk[2] - config.CHUNK_OVERLAP

        return chunks

    def _create_next_chunk(self, text: str, start: int) -> Tuple[str, int, int]:
        """Create the next chunk starting at the given position."""
        config = get_config()
        end = start + config.CHUNK_SIZE

        if end >= len(text):
            # Final chunk - include all remaining text
            return self._create_chunk_at_position(text, start, len(text))

        # Try to find a sentence boundary for a clean break
        boundary_end = self._find_sentence_boundary(text, start, end)
        if boundary_end is not None:
            return self._create_chunk_at_boundary(text, start, boundary_end)

        # No sentence boundary found, split at target position
        return self._create_chunk_at_position(text, start, end)

    def _find_sentence_boundary(self, text: str, start: int, target_end: int) -> int | None:
        """Find the best sentence boundary near the target end position."""
        config = get_config()

        # Define search window for sentence boundaries
        search_start = max(start + config.CHUNK_SIZE - config.CHUNK_OVERLAP, start + config.CHUNK_SIZE // 2)
        search_end = min(target_end + 100, len(text))  # Look a bit beyond target
        search_text = text[search_start:search_end]

        # Find sentence boundaries using regex
        sentence_pattern = r'[.!?]\s+'
        matches = list(re.finditer(sentence_pattern, search_text))

        if not matches:
            return None

        # Find the best match within our flexibility range
        flexibility = 50  # Allow some flexibility beyond target
        best_match = None

        for match in matches:
            abs_pos = search_start + match.end()
            if abs_pos <= target_end + flexibility:
                best_match = match
            else:
                break

        if best_match:
            return search_start + best_match.end()

        return None

    def _create_chunk_at_boundary(self, text: str, start: int, end: int) -> Tuple[str, int, int]:
        """Create a chunk ending at a sentence boundary."""
        return (text[start:end], start, end)

    def _create_chunk_at_position(self, text: str, start: int, end: int) -> Tuple[str, int, int]:
        """Create a chunk ending at a specific position."""
        return (text[start:end], start, end)

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index, splitting into chunks if necessary."""
        self.logger.debug("Adding document %s", doc_id)

        # Split text into chunks with position tracking
        text_chunks = self._chunk_text(text)

        for chunk_text, start_pos, end_pos in text_chunks:
            chunk = DocumentChunk(
                doc_id=doc_id,
                text=chunk_text,
                start_pos=start_pos,
                end_pos=end_pos
            )
            self.chunks.append(chunk)

        self.logger.debug("Document %s split into %d chunks", doc_id, len(text_chunks))

        if not self._bulk_mode:
            self._rebuild_index()

    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """Add multiple documents efficiently using bulk operations.
        
        Validates all documents before processing to ensure data integrity.
        If any document fails validation, no documents are added.
        
        Args:
            documents: List of (doc_id, text) tuples to add
            
        Raises:
            ValueError: If any document has invalid doc_id or text
        """
        # Validate all documents before processing any
        for doc_id, text in documents:
            self._validate_document(doc_id, text)

        # All documents validated, now process them
        with self.bulk_operation():
            for doc_id, text in documents:
                self.add_document(doc_id, text)

    def _validate_document(self, doc_id: str, text: str) -> None:
        """Validate a single document's doc_id and text.
        
        Args:
            doc_id: Document identifier to validate
            text: Document text to validate
            
        Raises:
            ValueError: If doc_id or text is invalid
        """
        self._validate_doc_id(doc_id)
        validate_text_safety(text, "text")

    def _validate_doc_id(self, doc_id: str) -> None:
        """Validate document ID format and content.
        
        Args:
            doc_id: Document identifier to validate
            
        Raises:
            ValueError: If doc_id is invalid
        """
        if not isinstance(doc_id, str):
            raise ValueError('doc_id must be a non-empty string')

        if not doc_id or not doc_id.strip():
            raise ValueError('doc_id cannot be empty')

    @contextmanager
    def bulk_operation(self):
        """Context manager for bulk operations that rebuilds index only once."""
        if self._bulk_mode:
            raise RuntimeError("already in bulk operation mode")

        self._bulk_mode = True
        try:
            yield
        finally:
            self._bulk_mode = False
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        texts = [c.text for c in self.chunks]
        self.logger.debug("Rebuilding index with %d documents", len(texts))

        # Filter out empty or whitespace-only texts
        # Keep track of indices to maintain alignment with chunks
        self.valid_chunk_indices = []
        valid_texts = []

        for i, text in enumerate(texts):
            if text.strip():  # Non-empty text
                valid_texts.append(text)
                self.valid_chunk_indices.append(i)

        if valid_texts:
            try:
                self.tfidf_matrix = self.vectorizer.fit_transform(valid_texts)
            except ValueError as e:
                # Handle case where all documents are stop words
                self.logger.warning("Failed to build TF-IDF index: %s", e)
                self.tfidf_matrix = None
                self.valid_chunk_indices = []
        else:
            self.tfidf_matrix = None
            self.valid_chunk_indices = []
        self.save()

    def query(self, question: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """Return top document chunks most relevant to the question."""
        if not self.chunks or self.tfidf_matrix is None or not hasattr(self, 'valid_chunk_indices'):
            return []
        self.logger.debug("Querying index: %s", question)
        q_vec = self.vectorizer.transform([question])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().ravel()
        idxs = scores.argsort()[::-1][:top_k]
        # Map back to original chunk indices
        return [(self.chunks[self.valid_chunk_indices[i]].doc_id, self.chunks[self.valid_chunk_indices[i]].text) for i in idxs]

    def query_with_chunks(self, question: str, top_k: int = 3) -> List[DocumentChunk]:
        """Return top document chunks most relevant to the question."""
        if not self.chunks or self.tfidf_matrix is None or not hasattr(self, 'valid_chunk_indices'):
            return []
        self.logger.debug("Querying index: %s", question)
        q_vec = self.vectorizer.transform([question])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().ravel()
        idxs = scores.argsort()[::-1][:top_k]
        # Map back to original chunk indices
        return [self.chunks[self.valid_chunk_indices[i]] for i in idxs]

    def answer_with_citations(
        self, question: str, top_k: int = 3
    ) -> Tuple[str, List[Citation]]:
        """Return concatenated answers with citations for top documents."""
        from .citation import Citation

        if not question:
            raise ValueError("question must be provided")
        self.logger.debug("Answering with citations: %s", question)

        # Use the new query method that returns full chunk objects
        chunk_results = self.query_with_chunks(question, top_k=top_k)
        answer_parts: List[str] = []
        citations: List[Citation] = []

        for chunk in chunk_results:
            answer_parts.append(chunk.text)
            # Use actual chunk positions instead of hardcoded values
            citations.append(Citation(
                doc_id=chunk.doc_id,
                text=chunk.text,
                start=chunk.start_pos,
                end=chunk.end_pos
            ))

        answer = "\n".join(answer_parts)
        return answer, citations
    
    @property
    def photonic_bridge(self) -> PhotonicBridge:
        """Lazy initialization of photonic bridge."""
        if self._photonic_bridge is None:
            from .photonic_bridge import PhotonicBridge
            self._photonic_bridge = PhotonicBridge(qa_engine=self)
        return self._photonic_bridge
    
    def quantum_enhanced_query(
        self,
        question: str,
        document_path: str,
        top_k: int = 3,
        quantum_threshold: float = 0.7
    ) -> PhotonicEnhancedResult:
        """
        Perform quantum-enhanced query processing.
        
        Args:
            question: The financial question to ask
            document_path: Path to the financial document
            top_k: Number of top results to return
            quantum_threshold: Threshold for applying quantum enhancement
            
        Returns:
            PhotonicEnhancedResult with combined classical and quantum analysis
        """
        if not self.enable_quantum:
            # Fall back to classical processing
            answer, citations = self.answer_with_citations(question, top_k)
            from .photonic_mlir import FinancialQueryType
            from .photonic_bridge import PhotonicEnhancedResult
            
            # Create mock quantum result for disabled quantum processing
            mock_quantum_result = self.photonic_bridge._create_mock_quantum_result(
                FinancialQueryType.RISK_ASSESSMENT
            )
            
            return PhotonicEnhancedResult(
                classical_answer=answer,
                quantum_result=mock_quantum_result,
                citations=citations,
                confidence_score=0.8,
                processing_metadata={
                    "quantum_enabled": False,
                    "quantum_applied": False,
                    "fallback_reason": "quantum_disabled"
                }
            )
        
        return self.photonic_bridge.process_enhanced_query(
            query=question,
            document_path=document_path,
            enable_quantum=self.enable_quantum,
            quantum_threshold=quantum_threshold
        )
    
    async def quantum_enhanced_query_async(
        self,
        question: str,
        document_path: str,
        top_k: int = 3,
        quantum_threshold: float = 0.7
    ) -> PhotonicEnhancedResult:
        """
        Async version of quantum-enhanced query processing.
        
        Args:
            question: The financial question to ask
            document_path: Path to the financial document
            top_k: Number of top results to return
            quantum_threshold: Threshold for applying quantum enhancement
            
        Returns:
            PhotonicEnhancedResult with combined classical and quantum analysis
        """
        if not self.enable_quantum:
            # Use synchronous version for disabled quantum
            return self.quantum_enhanced_query(question, document_path, top_k, quantum_threshold)
        
        return await self.photonic_bridge.process_enhanced_query_async(
            query=question,
            document_path=document_path,
            enable_quantum=self.enable_quantum,
            quantum_threshold=quantum_threshold
        )
    
    def get_quantum_capabilities(self) -> Dict[str, Any]:
        """Get information about quantum capabilities."""
        if not self.enable_quantum:
            return {"quantum_enabled": False, "reason": "quantum_disabled"}
        
        return {
            "quantum_enabled": True,
            **self.photonic_bridge.get_quantum_capabilities()
        }
    
    def benchmark_quantum_performance(
        self,
        test_queries: List[str],
        test_documents: List[str]
    ) -> Dict[str, Any]:
        """
        Benchmark quantum vs classical performance.
        
        Args:
            test_queries: List of test financial queries
            test_documents: List of test document paths
            
        Returns:
            Benchmark results comparing quantum and classical performance
        """
        if not self.enable_quantum:
            return {
                "error": "Quantum benchmarking not available - quantum processing disabled"
            }
        
        return self.photonic_bridge.benchmark_quantum_advantage(
            queries=test_queries,
            document_paths=test_documents
        )

    def save(self, path: Path | None = None) -> None:
        """Persist the vector store to disk."""
        target = path or self.storage_path
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        dump(
            {
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "chunks": self.chunks,
                "valid_chunk_indices": self.valid_chunk_indices,
            },
            target,
        )

    def load(self, path: Path) -> None:
        """Load a persisted vector store."""
        data = load(path)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.chunks = data["chunks"]

        # Rebuild valid_chunk_indices for compatibility with new chunking system
        if hasattr(data, 'get') and 'valid_chunk_indices' in data:
            self.valid_chunk_indices = data["valid_chunk_indices"]
        else:
            # For older saved data or when not available, rebuild indices
            self.valid_chunk_indices = []
            if self.chunks:
                for i, chunk in enumerate(self.chunks):
                    if chunk.text.strip():  # Non-empty chunk
                        self.valid_chunk_indices.append(i)
