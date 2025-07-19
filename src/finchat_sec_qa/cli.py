import argparse
from pathlib import Path
from typing import List

from .qa_engine import FinancialQAEngine
from .voice_interface import speak
from .logging_utils import configure_logging
from .edgar_client import EdgarClient
from .risk_intelligence import RiskAnalyzer

import logging

logger = logging.getLogger(__name__)


def build_engine(docs: List[Path]) -> FinancialQAEngine:
    engine = FinancialQAEngine()
    for path in docs:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        engine.add_document(path.name, path.read_text())
    return engine


def _cmd_query(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    logger.info("Starting query command with %d documents", len(args.documents))
    engine = build_engine([Path(p) for p in args.documents])
    logger.info("Answering question: %s", args.question)
    answer, citations = engine.answer_with_citations(args.question)
    logger.info("Generated answer with %d citations", len(citations))
    print(answer)
    if args.voice:
        logger.debug("Speaking answer aloud")
        speak(answer)
    for c in citations:
        print(f"[{c.doc_id}] {c.text}")
    logger.info("Query command completed successfully")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    logger.info("Starting ingest command for ticker: %s", args.ticker)
    client = EdgarClient("FinChatCLI")
    filings = client.get_recent_filings(
        args.ticker, form_type=args.form_type, limit=args.limit
    )
    logger.info("Found %d filings to download", len(filings))
    for filing in filings:
        logger.debug("Downloading filing: %s", filing.document_url)
        client.download_filing(filing, args.dest)
    logger.info("Ingest command completed successfully")
    return 0


def _cmd_risk(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    logger.info("Starting risk analysis for file: %s", args.file)
    analyzer = RiskAnalyzer()
    text = Path(args.file).read_text()
    logger.debug("Loaded text with %d characters", len(text))
    assessment = analyzer.assess(text)
    logger.info("Risk assessment completed with %d flags", len(assessment.flags))
    print(assessment.sentiment)
    for flag in assessment.flags:
        print(flag)
    logger.info("Risk command completed successfully")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FinChat-SEC-QA command line interface"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("query", help="Answer a question from documents")
    q.add_argument("question")
    q.add_argument("documents", nargs="+")
    q.add_argument("-v", "--voice", action="store_true", help="Speak answer")
    q.add_argument("-l", "--log-level", default="INFO", help="Logging level")
    q.set_defaults(func=_cmd_query)

    ing = sub.add_parser("ingest", help="Download filings for a ticker")
    ing.add_argument("ticker")
    ing.add_argument("--form-type", default="10-K")
    ing.add_argument("--limit", type=int, default=1)
    ing.add_argument("--dest", type=Path, default=Path.cwd())
    ing.add_argument("-l", "--log-level", default="INFO", help="Logging level")
    ing.set_defaults(func=_cmd_ingest)

    r = sub.add_parser("risk", help="Assess risk from a text file")
    r.add_argument("file")
    r.add_argument("-l", "--log-level", default="INFO", help="Logging level")
    r.set_defaults(func=_cmd_risk)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
