"""FinChat-SEC-QA package with Photonic Quantum Computing capabilities."""

__version__ = "2.0.0"

from .citation import Citation, extract_citation_anchors
from .cli import main as cli_main
from .edgar_client import AsyncEdgarClient, EdgarClient, FilingMetadata
from .edgar_validation import validate_accession_number, validate_cik, validate_ticker
from .logging_utils import configure_logging
from .multi_company import CompanyAnswer, compare_question_across_filings
from .qa_engine import DocumentChunk, FinancialQAEngine
from .query_handler import AsyncQueryHandler, QueryHandler
from .risk_intelligence import RiskAnalyzer, RiskAssessment
from .voice_interface import speak

# Photonic Quantum Computing components
try:
    from .photonic_mlir import (
        PhotonicMLIRSynthesizer,
        QuantumFinancialProcessor,
        FinancialQueryType,
        QuantumGateType,
        PhotonicCircuit,
        QuantumFinancialResult
    )
    from .photonic_bridge import PhotonicBridge, PhotonicEnhancedResult
    from .photonic_cache import (
        QuantumCircuitCache,
        QuantumOptimizer,
        ParallelQuantumProcessor,
        PerformanceProfiler
    )
    _QUANTUM_AVAILABLE = True
except ImportError:
    _QUANTUM_AVAILABLE = False

# SDK is an optional import that requires httpx
try:
    from . import sdk
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

__all__ = [
    # Core components
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

# Add Quantum components to __all__ if available
if _QUANTUM_AVAILABLE:
    __all__.extend([
        "PhotonicMLIRSynthesizer",
        "QuantumFinancialProcessor", 
        "FinancialQueryType",
        "QuantumGateType",
        "PhotonicCircuit",
        "QuantumFinancialResult",
        "PhotonicBridge",
        "PhotonicEnhancedResult",
        "QuantumCircuitCache",
        "QuantumOptimizer",
        "ParallelQuantumProcessor",
        "PerformanceProfiler"
    ])

# Add SDK to __all__ if available
if _SDK_AVAILABLE:
    __all__.append("sdk")

# Feature availability flags
FEATURES = {
    "quantum_computing": _QUANTUM_AVAILABLE,
    "sdk": _SDK_AVAILABLE,
    "core": True
}
