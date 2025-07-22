"""FinChat-SEC-QA package."""

__version__ = "1.4.1"

from .edgar_client import EdgarClient, AsyncEdgarClient, FilingMetadata
from .qa_engine import DocumentChunk, FinancialQAEngine
from .risk_intelligence import RiskAnalyzer, RiskAssessment
from .citation import Citation, extract_citation_anchors
from .cli import main as cli_main
from .voice_interface import speak
from .multi_company import CompanyAnswer, compare_question_across_filings
from .logging_utils import configure_logging
from .query_handler import QueryHandler, AsyncQueryHandler

__all__ = [
    "EdgarClient",
    "AsyncEdgarClient",
    "FilingMetadata",
    "DocumentChunk",
    "FinancialQAEngine",
    "RiskAnalyzer",
    "RiskAssessment",
    "Citation",
    "extract_citation_anchors",
    "cli_main",
    "speak",
    "CompanyAnswer",
    "compare_question_across_filings",
    "configure_logging",
    "QueryHandler",
    "AsyncQueryHandler",
]
