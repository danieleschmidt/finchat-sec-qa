from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:  # pragma: no cover - imported for type hints
    from .citation import Citation

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging
from joblib import dump, load


@dataclass
class DocumentChunk:
    """A slice of a document stored for retrieval."""

    doc_id: str
    text: str


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
        if storage_path and storage_path.exists():
            self.load(storage_path)
        elif storage_path and storage_path.with_suffix('.pkl').exists():
            # migrate old pickle-based index
            self.load(storage_path.with_suffix('.pkl'))
            storage_path.with_suffix('.pkl').unlink()
            self.storage_path = storage_path
            self.save()

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index."""
        self.logger.debug("Adding document %s", doc_id)
        self.chunks.append(DocumentChunk(doc_id, text))
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        texts = [c.text for c in self.chunks]
        self.logger.debug("Rebuilding index with %d documents", len(texts))
        if texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.tfidf_matrix = None
        self.save()

    def query(self, question: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """Return top document chunks most relevant to the question."""
        if not self.chunks or self.tfidf_matrix is None:
            return []
        self.logger.debug("Querying index: %s", question)
        q_vec = self.vectorizer.transform([question])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().ravel()
        idxs = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i].doc_id, self.chunks[i].text) for i in idxs]

    def answer_with_citations(
        self, question: str, top_k: int = 3
    ) -> Tuple[str, List[Citation]]:
        """Return concatenated answers with citations for top documents."""
        from .citation import Citation

        if not question:
            raise ValueError("question must be provided")
        self.logger.debug("Answering with citations: %s", question)

        results = self.query(question, top_k=top_k)
        answer_parts: List[str] = []
        citations: List[Citation] = []
        for doc_id, text in results:
            answer_parts.append(text)
            citations.append(Citation(doc_id=doc_id, text=text, start=0, end=len(text)))
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
            },
            target,
        )

    def load(self, path: Path) -> None:
        """Load a persisted vector store."""
        data = load(path)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.chunks = data["chunks"]
