from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:  # pragma: no cover - imported for type hints
    from .citation import Citation

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


@dataclass
class DocumentChunk:
    """A slice of a document stored for retrieval."""

    doc_id: str
    text: str


class FinancialQAEngine:
    """Simple QA engine using TF-IDF retrieval over filing texts."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunks: List[DocumentChunk] = []
        self.tfidf_matrix: np.ndarray | None = None

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index."""
        self.chunks.append(DocumentChunk(doc_id, text))
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        texts = [c.text for c in self.chunks]
        if texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.tfidf_matrix = None

    def query(self, question: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """Return top document chunks most relevant to the question."""
        if not self.chunks or self.tfidf_matrix is None:
            return []
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

        results = self.query(question, top_k=top_k)
        answer_parts: List[str] = []
        citations: List[Citation] = []
        for doc_id, text in results:
            answer_parts.append(text)
            citations.append(Citation(doc_id=doc_id, text=text, start=0, end=len(text)))
        answer = "\n".join(answer_parts)
        return answer, citations
