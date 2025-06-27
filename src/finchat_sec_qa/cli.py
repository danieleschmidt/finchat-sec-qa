import argparse
from pathlib import Path
from typing import List

from .qa_engine import FinancialQAEngine
from .voice_interface import speak


def build_engine(docs: List[Path]) -> FinancialQAEngine:
    engine = FinancialQAEngine()
    for path in docs:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        engine.add_document(path.name, path.read_text())
    return engine


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Answer questions with citations from text files"
    )
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("documents", nargs="+", help="Paths to text documents")
    parser.add_argument(
        "-v",
        "--voice",
        action="store_true",
        help="Speak the answer aloud",
    )
    args = parser.parse_args(argv)

    engine = build_engine([Path(p) for p in args.documents])
    answer, citations = engine.answer_with_citations(args.question)
    print(answer)
    if args.voice:
        speak(answer)
    for c in citations:
        print(f"[{c.doc_id}] {c.text}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
