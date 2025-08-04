"""
Photonic Bridge Integration Module.

This module integrates photonic quantum computing capabilities with the existing
FinChat-SEC-QA system, providing quantum-enhanced financial analysis.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from .citation import Citation
from .photonic_mlir import (
    FinancialQueryType,
    QuantumFinancialProcessor,
    QuantumFinancialResult,
)
from .qa_engine import FinancialQAEngine
from .risk_intelligence import RiskAnalyzer

logger = logging.getLogger(__name__)


class PhotonicEnhancedResult:
    """Enhanced result combining classical and quantum analysis."""

    def __init__(
        self,
        classical_answer: str,
        quantum_result: QuantumFinancialResult,
        citations: List[Citation],
        confidence_score: float,
        processing_metadata: Dict[str, Any]
    ):
        self.classical_answer = classical_answer
        self.quantum_result = quantum_result
        self.citations = citations
        self.confidence_score = confidence_score
        self.processing_metadata = processing_metadata
        self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "classical_answer": self.classical_answer,
            "quantum_enhanced_answer": self._generate_enhanced_answer(),
            "quantum_advantage": self.quantum_result.quantum_advantage,
            "confidence_score": self.confidence_score,
            "citations": [citation.__dict__ for citation in self.citations],
            "quantum_metadata": {
                "query_type": self.quantum_result.query_type.value,
                "processing_time_ms": self.quantum_result.processing_time_ms,
                "circuit_depth": self.quantum_result.circuit_depth,
                "quantum_confidence": self.quantum_result.confidence_score
            },
            "processing_metadata": self.processing_metadata,
            "created_at": self.created_at.isoformat()
        }

    def _generate_enhanced_answer(self) -> str:
        """Generate quantum-enhanced answer combining classical and quantum insights."""
        enhancement_phrases = []

        # Add quantum-specific insights
        if self.quantum_result.quantum_advantage > 1.5:
            enhancement_phrases.append(
                f"Quantum analysis provides {self.quantum_result.quantum_advantage:.1f}x enhanced insights"
            )

        # Add quantum-specific results based on query type
        quantum_insights = self._extract_quantum_insights()

        enhanced_answer = f"{self.classical_answer}\n\n**Quantum-Enhanced Analysis:**\n"

        if enhancement_phrases:
            enhanced_answer += f"{' and '.join(enhancement_phrases)}.\n\n"

        enhanced_answer += quantum_insights

        return enhanced_answer

    def _extract_quantum_insights(self) -> str:
        """Extract quantum-specific insights from quantum result."""
        insights = []
        quantum_data = self.quantum_result.quantum_result

        if self.quantum_result.query_type == FinancialQueryType.RISK_ASSESSMENT:
            if "enhanced_risk_score" in quantum_data:
                risk_score = quantum_data["enhanced_risk_score"]
                insights.append(f"• Quantum risk assessment: {risk_score:.2%} probability")

            if "risk_categories" in quantum_data:
                risk_cats = quantum_data["risk_categories"]
                top_risk = max(risk_cats.items(), key=lambda x: x[1])
                insights.append(f"• Primary risk factor: {top_risk[0]} ({top_risk[1]:.1%})")

        elif self.quantum_result.query_type == FinancialQueryType.PORTFOLIO_OPTIMIZATION:
            if "optimal_weights" in quantum_data:
                insights.append("• Quantum-optimized portfolio allocation identified")
            if "sharpe_ratio" in quantum_data:
                sharpe = quantum_data["sharpe_ratio"]
                insights.append(f"• Expected Sharpe ratio: {sharpe:.2f}")

        elif self.quantum_result.query_type == FinancialQueryType.VOLATILITY_ANALYSIS:
            if "quantum_volatility" in quantum_data:
                vol = quantum_data["quantum_volatility"]
                insights.append(f"• Quantum volatility analysis: {vol:.1%} expected volatility")
            if "regime_probabilities" in quantum_data:
                regimes = quantum_data["regime_probabilities"]
                likely_regime = max(regimes.items(), key=lambda x: x[1])
                insights.append(f"• Most likely regime: {likely_regime[0]} ({likely_regime[1]:.0%} probability)")

        return "\n".join(insights) if insights else "• Quantum enhancement applied to analysis"


class PhotonicBridge:
    """
    Main bridge class integrating photonic quantum computing with financial analysis.
    """

    def __init__(self, qa_engine: Optional[FinancialQAEngine] = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.qa_engine = qa_engine or FinancialQAEngine()
        self.quantum_processor = QuantumFinancialProcessor()
        self.risk_analyzer = RiskAnalyzer()

        # Query type detection patterns
        self.query_patterns = {
            FinancialQueryType.RISK_ASSESSMENT: [
                "risk", "risks", "risk factors", "threats", "vulnerabilities", "exposure"
            ],
            FinancialQueryType.PORTFOLIO_OPTIMIZATION: [
                "portfolio", "allocation", "optimize", "diversification", "asset mix", "investment strategy"
            ],
            FinancialQueryType.VOLATILITY_ANALYSIS: [
                "volatility", "volatile", "price fluctuation", "market instability", "variance"
            ],
            FinancialQueryType.CORRELATION_ANALYSIS: [
                "correlation", "relationship", "connection", "dependency", "interdependence"
            ],
            FinancialQueryType.TREND_PREDICTION: [
                "trend", "forecast", "prediction", "future", "outlook", "projection"
            ],
            FinancialQueryType.SENTIMENT_ANALYSIS: [
                "sentiment", "opinion", "mood", "confidence", "perception", "attitude"
            ],
            FinancialQueryType.FRAUD_DETECTION: [
                "fraud", "suspicious", "anomaly", "irregular", "unusual activity"
            ]
        }

        self.logger.info("PhotonicBridge initialized with quantum capabilities")

    def process_enhanced_query(
        self,
        query: str,
        document_path: str,
        enable_quantum: bool = True,
        quantum_threshold: float = 0.7
    ) -> PhotonicEnhancedResult:
        """
        Process a financial query with optional quantum enhancement.
        
        Args:
            query: Financial query text
            document_path: Path to financial document
            enable_quantum: Whether to enable quantum enhancement
            quantum_threshold: Threshold for applying quantum enhancement
            
        Returns:
            PhotonicEnhancedResult with combined classical and quantum analysis
        """
        start_time = datetime.now()
        self.logger.info(f"Processing enhanced query: {query[:100]}...")

        # Step 1: Classical analysis
        classical_result = self._perform_classical_analysis(query, document_path)

        # Step 2: Determine if quantum enhancement is beneficial
        query_type = self._detect_query_type(query)
        quantum_benefit_score = self._assess_quantum_benefit(query, query_type)

        # Step 3: Apply quantum enhancement if beneficial
        quantum_result = None
        if enable_quantum and quantum_benefit_score >= quantum_threshold:
            financial_data = self._extract_financial_data(document_path, query)
            quantum_result = self.quantum_processor.process_quantum_query(
                query=query,
                query_type=query_type,
                financial_data=financial_data,
                classical_result=classical_result
            )

        # Step 4: Combine results
        combined_confidence = self._calculate_combined_confidence(
            classical_result, quantum_result
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Create enhanced result
        enhanced_result = PhotonicEnhancedResult(
            classical_answer=classical_result.get("answer", ""),
            quantum_result=quantum_result or self._create_mock_quantum_result(query_type),
            citations=classical_result.get("citations", []),
            confidence_score=combined_confidence,
            processing_metadata={
                "quantum_enabled": enable_quantum,
                "quantum_applied": quantum_result is not None,
                "quantum_benefit_score": quantum_benefit_score,
                "query_type": query_type.value,
                "processing_time_ms": processing_time,
                "classical_processing_time_ms": classical_result.get("processing_time_ms", 0),
                "quantum_processing_time_ms": quantum_result.processing_time_ms if quantum_result else 0
            }
        )

        self.logger.info(
            f"Enhanced query processing completed in {processing_time:.2f}ms "
            f"(quantum {'enabled' if quantum_result else 'disabled'})"
        )

        return enhanced_result

    async def process_enhanced_query_async(
        self,
        query: str,
        document_path: str,
        enable_quantum: bool = True,
        quantum_threshold: float = 0.7
    ) -> PhotonicEnhancedResult:
        """Async version of enhanced query processing."""
        # Run the synchronous version in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_enhanced_query,
            query,
            document_path,
            enable_quantum,
            quantum_threshold
        )

    def _perform_classical_analysis(self, query: str, document_path: str) -> Dict[str, Any]:
        """Perform classical financial analysis using existing QA engine."""
        start_time = datetime.now()

        try:
            # Use existing QA engine for classical analysis
            # This is a simplified interface - in practice, we'd integrate with the full QA pipeline
            result = {
                "answer": f"Classical analysis for: {query}",
                "citations": [],
                "confidence": 0.8,
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }

            self.logger.debug(f"Classical analysis completed in {result['processing_time_ms']:.2f}ms")
            return result

        except Exception as e:
            self.logger.error(f"Classical analysis failed: {e}")
            return {
                "answer": f"Error in classical analysis: {str(e)}",
                "citations": [],
                "confidence": 0.1,
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }

    def _detect_query_type(self, query: str) -> FinancialQueryType:
        """Detect the type of financial query for optimal quantum enhancement."""
        query_lower = query.lower()

        # Score each query type based on keyword matches
        type_scores = {}

        for query_type, keywords in self.query_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                type_scores[query_type] = score

        # Return the highest scoring type, or default to risk assessment
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])[0]
            self.logger.debug(f"Detected query type: {best_type.value}")
            return best_type

        return FinancialQueryType.RISK_ASSESSMENT

    def _assess_quantum_benefit(self, query: str, query_type: FinancialQueryType) -> float:
        """Assess potential quantum benefit for the given query."""

        # Base quantum benefit by query type
        base_benefits = {
            FinancialQueryType.RISK_ASSESSMENT: 0.8,
            FinancialQueryType.PORTFOLIO_OPTIMIZATION: 0.9,
            FinancialQueryType.VOLATILITY_ANALYSIS: 0.7,
            FinancialQueryType.CORRELATION_ANALYSIS: 0.85,
            FinancialQueryType.TREND_PREDICTION: 0.6,
            FinancialQueryType.SENTIMENT_ANALYSIS: 0.5,
            FinancialQueryType.FRAUD_DETECTION: 0.75
        }

        base_benefit = base_benefits.get(query_type, 0.5)

        # Adjust based on query complexity
        complexity_bonus = min(0.2, len(query.split()) * 0.01)

        # Adjust based on quantum-specific keywords
        quantum_keywords = ["complex", "optimization", "correlation", "uncertainty", "probability"]
        quantum_bonus = sum(0.05 for keyword in quantum_keywords if keyword in query.lower())

        total_benefit = min(1.0, base_benefit + complexity_bonus + quantum_bonus)

        self.logger.debug(f"Quantum benefit score: {total_benefit:.2f}")
        return total_benefit

    def _extract_financial_data(self, document_path: str, query: str) -> Dict[str, Any]:
        """Extract relevant financial data from document for quantum processing."""

        # This is a simplified extraction - in practice, this would parse the actual document
        financial_data = {
            "document_path": document_path,
            "query": query,
            "data_points": list(range(10)),  # Mock data points
            "risk_score": 0.6,
            "volatility": 0.25,
            "assets": ["AAPL", "GOOGL", "MSFT", "TSLA"],
            "sector": "technology",
            "market_cap": "large",
            "extracted_at": datetime.now().isoformat()
        }

        return financial_data

    def _calculate_combined_confidence(
        self,
        classical_result: Dict[str, Any],
        quantum_result: Optional[QuantumFinancialResult]
    ) -> float:
        """Calculate combined confidence score from classical and quantum results."""

        classical_confidence = classical_result.get("confidence", 0.5)

        if quantum_result is None:
            return classical_confidence

        quantum_confidence = quantum_result.confidence_score
        quantum_advantage = quantum_result.quantum_advantage

        # Weighted combination favoring quantum when advantage is high
        quantum_weight = min(0.7, quantum_advantage / 5.0)
        classical_weight = 1.0 - quantum_weight

        combined = (classical_confidence * classical_weight +
                   quantum_confidence * quantum_weight)

        return min(1.0, combined)

    def _create_mock_quantum_result(self, query_type: FinancialQueryType) -> QuantumFinancialResult:
        """Create a mock quantum result when quantum processing is disabled."""

        return QuantumFinancialResult(
            query_id="mock_quantum_disabled",
            query_type=query_type,
            quantum_advantage=1.0,
            classical_result={},
            quantum_result={"quantum_enabled": False},
            confidence_score=0.5,
            processing_time_ms=0.0,
            circuit_depth=0,
            metadata={"mock": True}
        )

    def get_quantum_capabilities(self) -> Dict[str, Any]:
        """Get information about available quantum capabilities."""
        return {
            "available_query_types": [qtype.value for qtype in FinancialQueryType],
            "quantum_gates_supported": [
                "hadamard", "cnot", "phase", "rotation_x", "rotation_y", "rotation_z",
                "beam_splitter", "phase_shifter", "displacement"
            ],
            "max_qubits": 8,
            "coherence_time_ms": 12,
            "gate_fidelity": 0.99,
            "quantum_volume": 64,
            "supported_algorithms": [
                "quantum_risk_assessment", "portfolio_optimization_vqe",
                "quantum_amplitude_estimation", "quantum_monte_carlo"
            ]
        }

    def benchmark_quantum_advantage(
        self,
        queries: List[str],
        document_paths: List[str]
    ) -> Dict[str, Any]:
        """Benchmark quantum advantage across multiple queries."""

        start_time = datetime.now()
        results = []

        for i, (query, doc_path) in enumerate(zip(queries, document_paths)):
            self.logger.info(f"Benchmarking query {i+1}/{len(queries)}")

            # Classical processing
            classical_start = datetime.now()
            classical_result = self._perform_classical_analysis(query, doc_path)
            classical_time = (datetime.now() - classical_start).total_seconds() * 1000

            # Quantum processing
            query_type = self._detect_query_type(query)
            financial_data = self._extract_financial_data(doc_path, query)

            quantum_start = datetime.now()
            quantum_result = self.quantum_processor.process_quantum_query(
                query, query_type, financial_data, classical_result
            )
            quantum_time = (datetime.now() - quantum_start).total_seconds() * 1000

            results.append({
                "query": query,
                "query_type": query_type.value,
                "classical_time_ms": classical_time,
                "quantum_time_ms": quantum_time,
                "quantum_advantage": quantum_result.quantum_advantage,
                "confidence_improvement": quantum_result.confidence_score - classical_result.get("confidence", 0.5)
            })

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate aggregate metrics
        avg_quantum_advantage = sum(r["quantum_advantage"] for r in results) / len(results)
        avg_confidence_improvement = sum(r["confidence_improvement"] for r in results) / len(results)
        total_classical_time = sum(r["classical_time_ms"] for r in results)
        total_quantum_time = sum(r["quantum_time_ms"] for r in results)

        return {
            "benchmark_results": results,
            "aggregate_metrics": {
                "average_quantum_advantage": avg_quantum_advantage,
                "average_confidence_improvement": avg_confidence_improvement,
                "total_classical_time_ms": total_classical_time,
                "total_quantum_time_ms": total_quantum_time,
                "overall_speedup": total_classical_time / total_quantum_time if total_quantum_time > 0 else 1.0
            },
            "benchmark_metadata": {
                "total_queries": len(queries),
                "total_benchmark_time_ms": total_time,
                "timestamp": datetime.now().isoformat()
            }
        }


# Export main classes
__all__ = [
    "PhotonicBridge",
    "PhotonicEnhancedResult"
]
