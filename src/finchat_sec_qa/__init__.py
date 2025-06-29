"""FinChat-SEC-QA package."""

__version__ = "1.3.2"

from .edgar_client import EdgarClient, FilingMetadata
from .qa_engine import DocumentChunk, FinancialQAEngine
from .risk_intelligence import RiskAnalyzer, RiskAssessment
from .citation import Citation, extract_citation_anchors
from .cli import main as cli_main
from .voice_interface import speak
from .multi_company import CompanyAnswer, compare_question_across_filings
from .logging_utils import configure_logging

__all__ = [
    "EdgarClient",
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
]
