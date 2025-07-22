"""FinChat-SEC-QA package."""

__version__ = "1.4.7"

from .edgar_client import EdgarClient, AsyncEdgarClient, FilingMetadata
from .qa_engine import DocumentChunk, FinancialQAEngine
from .risk_intelligence import RiskAnalyzer, RiskAssessment
from .citation import Citation, extract_citation_anchors
from .cli import main as cli_main
from .voice_interface import speak
from .multi_company import CompanyAnswer, compare_question_across_filings
from .logging_utils import configure_logging
from .query_handler import QueryHandler, AsyncQueryHandler
from .edgar_validation import validate_ticker, validate_cik, validate_accession_number

# SDK is an optional import that requires httpx
try:
    from . import sdk
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

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
    "validate_ticker",
    "validate_cik",
    "validate_accession_number",
]

# Add SDK to __all__ if available
if _SDK_AVAILABLE:
    __all__.append("sdk")
