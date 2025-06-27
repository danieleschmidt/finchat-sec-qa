"""FinChat-SEC-QA package."""

from .edgar_client import EdgarClient, FilingMetadata
from .qa_engine import DocumentChunk, FinancialQAEngine
from .risk_intelligence import RiskAnalyzer, RiskAssessment
from .citation import Citation, extract_citation_anchors
from .cli import main as cli_main
from .voice_interface import speak
from .multi_company import CompanyAnswer, compare_question_across_filings

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
]
