from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple
from contextlib import contextmanager

if TYPE_CHECKING:  # pragma: no cover - imported for type hints
    from .citation import Citation

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging
from joblib import dump, load
import re

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

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunks: List[DocumentChunk] = []
        self.tfidf_matrix: np.ndarray | None = None
        self.valid_chunk_indices: List[int] = []
        self._bulk_mode = False
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
        config = get_config()
        
        if len(text) <= config.CHUNK_SIZE:
            # Document is small enough to be a single chunk
            return [(text, 0, len(text))]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position for this chunk
            end = start + config.CHUNK_SIZE
            
            if end >= len(text):
                # This is the final chunk
                chunks.append((text[start:], start, len(text)))
                break
            
            # Try to find a good breaking point near the end of the chunk
            # Look for sentence boundaries (. ! ?) followed by whitespace
            search_start = max(start + config.CHUNK_SIZE - config.CHUNK_OVERLAP, start + config.CHUNK_SIZE // 2)
            search_text = text[search_start:end + 100]  # Look a bit beyond the target end
            
            # Find sentence boundaries
            sentence_pattern = r'[.!?]\s+'
            matches = list(re.finditer(sentence_pattern, search_text))
            
            if matches:
                # Use the last sentence boundary before or near our target end
                best_match = None
                for match in matches:
                    abs_pos = search_start + match.end()
                    if abs_pos <= end + 50:  # Allow some flexibility beyond target
                        best_match = match
                    else:
                        break
                
                if best_match:
                    chunk_end = search_start + best_match.end()
                    chunks.append((text[start:chunk_end], start, chunk_end))
                    start = chunk_end - config.CHUNK_OVERLAP
                    continue
            
            # No good sentence boundary found, just split at the target position
            chunks.append((text[start:end], start, end))
            start = end - config.CHUNK_OVERLAP
        
        return chunks

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
