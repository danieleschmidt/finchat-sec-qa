"""
Tests for the Photonic MLIR Synthesis Bridge module.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from finchat_sec_qa.photonic_mlir import (
    PhotonicMLIRSynthesizer,
    QuantumFinancialProcessor,
    PhotonicCircuit,
    QuantumFinancialResult,
    FinancialQueryType,
    QuantumGateType,
    MLIRQuantumOperation
)


class TestMLIRQuantumOperation:
    """Test MLIR quantum operation functionality."""
    
    def test_mlir_operation_creation(self):
        """Test creating MLIR quantum operations."""
        operation = MLIRQuantumOperation(
            operation_type="test_op",
            gate_type=QuantumGateType.HADAMARD,
            qubits=[0, 1],
            parameters={"angle": 0.5},
            metadata={"purpose": "test"}
        )
        
        assert operation.operation_type == "test_op"
        assert operation.gate_type == QuantumGateType.HADAMARD
        assert operation.qubits == [0, 1]
        assert operation.parameters == {"angle": 0.5}
        assert operation.metadata == {"purpose": "test"}
    
    def test_to_mlir_conversion(self):
        """Test MLIR representation generation."""
        operation = MLIRQuantumOperation(
            operation_type="hadamard_op",
            gate_type=QuantumGateType.HADAMARD,
            qubits=[0],
            parameters={},
            metadata={}
        )
        
        mlir_repr = operation.to_mlir()
        assert "quantum.hadamard" in mlir_repr
        assert "%q0" in mlir_repr
        assert "!quantum.qubit" in mlir_repr


class TestPhotonicCircuit:
    """Test photonic quantum circuit functionality."""
    
    @pytest.fixture
    def sample_operations(self):
        """Create sample quantum operations."""
        return [
            MLIRQuantumOperation(
                operation_type="prep",
                gate_type=QuantumGateType.HADAMARD,
                qubits=[0],
                parameters={},
                metadata={"purpose": "superposition"}
            ),
            MLIRQuantumOperation(
                operation_type="entangle",
                gate_type=QuantumGateType.CNOT,
                qubits=[0, 1],
                parameters={},
                metadata={"purpose": "entanglement"}
            )
        ]
    
    def test_circuit_creation(self, sample_operations):
        """Test creating photonic quantum circuits."""
        circuit = PhotonicCircuit(
            circuit_id="test_circuit",
            operations=sample_operations,
            input_qubits=2,
            output_qubits=2,
            financial_context={"query": "test", "sector": "tech"},
            created_at=datetime.now()
        )
        
        assert circuit.circuit_id == "test_circuit"
        assert len(circuit.operations) == 2
        assert circuit.input_qubits == 2
        assert circuit.output_qubits == 2
        assert "query" in circuit.financial_context
    
    def test_mlir_module_generation(self, sample_operations):
        """Test MLIR module generation from circuit."""
        circuit = PhotonicCircuit(
            circuit_id="test_123",
            operations=sample_operations,
            input_qubits=2,
            output_qubits=2,
            financial_context={},
            created_at=datetime.now()
        )
        
        mlir_module = circuit.to_mlir_module()
        assert "module @financial_photonic_circuit_test_123" in mlir_module
        assert "func.func @quantum_financial_analysis" in mlir_module
        assert "!quantum.state<2>" in mlir_module


class TestPhotonicMLIRSynthesizer:
    """Test photonic MLIR synthesizer functionality."""
    
    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer instance."""
        return PhotonicMLIRSynthesizer()
    
    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial data."""
        return {
            "risk_score": 0.6,
            "volatility": 0.25,
            "assets": ["AAPL", "GOOGL", "MSFT"],
            "sector": "technology",
            "data_points": list(range(10))
        }
    
    def test_synthesizer_initialization(self, synthesizer):
        """Test synthesizer initialization."""
        assert synthesizer.circuit_registry == {}
        assert synthesizer.optimization_cache == {}
        assert hasattr(synthesizer, 'logger')
    
    def test_risk_assessment_synthesis(self, synthesizer, sample_financial_data):
        """Test synthesizing risk assessment quantum circuits."""
        circuit = synthesizer.synthesize_financial_query(
            query="What are the risk factors?",
            query_type=FinancialQueryType.RISK_ASSESSMENT,
            financial_data=sample_financial_data
        )
        
        assert circuit.circuit_id.startswith("fin_risk_assessment_")
        assert len(circuit.operations) > 0
        assert circuit.input_qubits >= 2
        assert circuit.output_qubits >= 2
        assert circuit.financial_context["query_type"] == "risk_assessment"
    
    def test_portfolio_optimization_synthesis(self, synthesizer, sample_financial_data):
        """Test synthesizing portfolio optimization quantum circuits."""
        circuit = synthesizer.synthesize_financial_query(
            query="Optimize my portfolio allocation",
            query_type=FinancialQueryType.PORTFOLIO_OPTIMIZATION,
            financial_data=sample_financial_data
        )
        
        assert circuit.circuit_id.startswith("fin_portfolio_optimization_")
        assert len(circuit.operations) > 0
        # Should have operations for each asset
        asset_ops = [op for op in circuit.operations if "asset" in op.operation_type]
        assert len(asset_ops) > 0
    
    def test_volatility_analysis_synthesis(self, synthesizer, sample_financial_data):
        """Test synthesizing volatility analysis quantum circuits."""
        circuit = synthesizer.synthesize_financial_query(
            query="Analyze market volatility",
            query_type=FinancialQueryType.VOLATILITY_ANALYSIS,
            financial_data=sample_financial_data
        )
        
        assert circuit.circuit_id.startswith("fin_volatility_analysis_")
        assert len(circuit.operations) > 0
        # Should have operations for volatility modeling
        volatility_ops = [op for op in circuit.operations if "volatility" in op.operation_type]
        assert len(volatility_ops) > 0
    
    def test_circuit_optimization(self, synthesizer):
        """Test quantum circuit optimization."""
        # Create a circuit with redundant operations
        operations = [
            MLIRQuantumOperation(
                operation_type="rotation1",
                gate_type=QuantumGateType.ROTATION_X,
                qubits=[0],
                parameters={"theta": 0.1},
                metadata={}
            ),
            MLIRQuantumOperation(
                operation_type="rotation2",
                gate_type=QuantumGateType.ROTATION_X,
                qubits=[0],
                parameters={"theta": 0.2},
                metadata={}
            ),
            MLIRQuantumOperation(
                operation_type="zero_rotation",
                gate_type=QuantumGateType.ROTATION_Y,
                qubits=[1],
                parameters={"theta": 0.0},  # This should be removed
                metadata={}
            )
        ]
        
        circuit = PhotonicCircuit(
            circuit_id="test_optimization",
            operations=operations,
            input_qubits=2,
            output_qubits=2,
            financial_context={},
            created_at=datetime.now()
        )
        
        optimized = synthesizer.optimize_circuit(circuit)
        
        # Should have fewer operations after optimization
        assert len(optimized.operations) < len(circuit.operations)
        assert optimized.circuit_id.endswith("_optimized")
    
    def test_input_qubit_calculation(self, synthesizer):
        """Test calculation of optimal input qubits."""
        # Small data should use minimum qubits
        small_data = {"data_points": [1, 2, 3]}
        qubits = synthesizer._calculate_input_qubits(small_data)
        assert qubits >= 2
        
        # Large data should use more qubits (but capped)
        large_data = {"data_points": list(range(1000))}
        qubits = synthesizer._calculate_input_qubits(large_data)
        assert qubits <= 8
    
    def test_output_qubit_calculation(self, synthesizer):
        """Test calculation of output qubits based on query type."""
        risk_qubits = synthesizer._calculate_output_qubits(FinancialQueryType.RISK_ASSESSMENT)
        portfolio_qubits = synthesizer._calculate_output_qubits(FinancialQueryType.PORTFOLIO_OPTIMIZATION)
        
        assert risk_qubits >= 2
        assert portfolio_qubits >= 2
        # Portfolio optimization typically needs more qubits
        assert portfolio_qubits >= risk_qubits


class TestQuantumFinancialProcessor:
    """Test quantum financial processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return QuantumFinancialProcessor()
    
    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial data."""
        return {
            "risk_score": 0.6,
            "volatility": 0.25,
            "assets": ["AAPL", "GOOGL"],
            "sector": "technology"
        }
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert hasattr(processor, 'synthesizer')
        assert processor.execution_cache == {}
    
    def test_quantum_query_processing(self, processor, sample_financial_data):
        """Test quantum query processing."""
        result = processor.process_quantum_query(
            query="What are the main risks?",
            query_type=FinancialQueryType.RISK_ASSESSMENT,
            financial_data=sample_financial_data
        )
        
        assert isinstance(result, QuantumFinancialResult)
        assert result.query_type == FinancialQueryType.RISK_ASSESSMENT
        assert result.quantum_advantage >= 1.0
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.processing_time_ms > 0
        assert result.circuit_depth > 0
    
    def test_risk_assessment_simulation(self, processor, sample_financial_data):
        """Test risk assessment quantum simulation."""
        result = processor.process_quantum_query(
            query="Assess financial risks",
            query_type=FinancialQueryType.RISK_ASSESSMENT,
            financial_data=sample_financial_data
        )
        
        quantum_result = result.quantum_result
        assert "enhanced_risk_score" in quantum_result
        assert "risk_categories" in quantum_result
        assert "uncertainty_bounds" in quantum_result
        assert 0.0 <= quantum_result["enhanced_risk_score"] <= 1.0
    
    def test_portfolio_optimization_simulation(self, processor, sample_financial_data):
        """Test portfolio optimization quantum simulation."""
        result = processor.process_quantum_query(
            query="Optimize portfolio allocation",
            query_type=FinancialQueryType.PORTFOLIO_OPTIMIZATION,
            financial_data=sample_financial_data
        )
        
        quantum_result = result.quantum_result
        assert "optimal_weights" in quantum_result
        assert "expected_return" in quantum_result
        assert "sharpe_ratio" in quantum_result
        
        # Weights should sum to approximately 1
        weights = quantum_result["optimal_weights"]
        assert abs(sum(weights) - 1.0) < 0.01
    
    def test_volatility_analysis_simulation(self, processor, sample_financial_data):
        """Test volatility analysis quantum simulation."""
        result = processor.process_quantum_query(
            query="Analyze market volatility",
            query_type=FinancialQueryType.VOLATILITY_ANALYSIS,
            financial_data=sample_financial_data
        )
        
        quantum_result = result.quantum_result
        assert "quantum_volatility" in quantum_result
        assert "volatility_confidence" in quantum_result
        assert "regime_probabilities" in quantum_result
        
        # Check regime probabilities sum to 1
        regimes = quantum_result["regime_probabilities"]
        assert abs(sum(regimes.values()) - 1.0) < 0.01
    
    def test_quantum_advantage_calculation(self, processor):
        """Test quantum advantage calculation."""
        quantum_result = {"quantum_speedup": "2.5x", "enhanced": True}
        classical_result = {"basic": True}
        
        advantage = processor._calculate_quantum_advantage(quantum_result, classical_result)
        assert advantage >= 2.5
        assert advantage <= 10.0  # Capped maximum
    
    def test_confidence_score_calculation(self, processor):
        """Test confidence score calculation."""
        # High coherence should give high confidence
        high_coherence_result = {"quantum_coherence": 0.95, "extra_data": True}
        confidence = processor._calculate_confidence_score(high_coherence_result)
        assert confidence > 0.9
        
        # Low coherence should give lower confidence
        low_coherence_result = {"quantum_coherence": 0.3}
        confidence = processor._calculate_confidence_score(low_coherence_result)
        assert confidence < 0.5
    
    def test_execution_caching(self, processor, sample_financial_data):
        """Test that results are cached properly."""
        result1 = processor.process_quantum_query(
            query="Test query",
            query_type=FinancialQueryType.RISK_ASSESSMENT,
            financial_data=sample_financial_data
        )
        
        # Check result is cached
        assert result1.query_id in processor.execution_cache
        cached_result = processor.execution_cache[result1.query_id]
        assert cached_result.query_id == result1.query_id


class TestQuantumFinancialResult:
    """Test quantum financial result data structure."""
    
    @pytest.fixture
    def sample_result(self):
        """Create sample quantum result."""
        return QuantumFinancialResult(
            query_id="test_123",
            query_type=FinancialQueryType.RISK_ASSESSMENT,
            quantum_advantage=2.5,
            classical_result={"answer": "classical"},
            quantum_result={"enhanced_answer": "quantum", "risk_score": 0.7},
            confidence_score=0.9,
            processing_time_ms=150.5,
            circuit_depth=10,
            metadata={"test": True}
        )
    
    def test_result_structure(self, sample_result):
        """Test quantum result structure."""
        assert sample_result.query_id == "test_123"
        assert sample_result.query_type == FinancialQueryType.RISK_ASSESSMENT
        assert sample_result.quantum_advantage == 2.5
        assert sample_result.confidence_score == 0.9
        assert sample_result.processing_time_ms == 150.5
        assert sample_result.circuit_depth == 10
    
    def test_result_data_integrity(self, sample_result):
        """Test that result data maintains integrity."""
        assert "answer" in sample_result.classical_result
        assert "enhanced_answer" in sample_result.quantum_result
        assert sample_result.metadata["test"] is True


class TestFinancialQueryType:
    """Test financial query type enumeration."""
    
    def test_query_type_values(self):
        """Test query type enumeration values."""
        assert FinancialQueryType.RISK_ASSESSMENT.value == "risk_assessment"
        assert FinancialQueryType.PORTFOLIO_OPTIMIZATION.value == "portfolio_optimization"
        assert FinancialQueryType.VOLATILITY_ANALYSIS.value == "volatility_analysis"
        assert FinancialQueryType.CORRELATION_ANALYSIS.value == "correlation_analysis"
        assert FinancialQueryType.TREND_PREDICTION.value == "trend_prediction"
        assert FinancialQueryType.SENTIMENT_ANALYSIS.value == "sentiment_analysis"
        assert FinancialQueryType.FRAUD_DETECTION.value == "fraud_detection"
    
    def test_query_type_completeness(self):
        """Test that all query types are defined."""
        query_types = list(FinancialQueryType)
        assert len(query_types) == 7  # Update if adding more types


class TestQuantumGateType:
    """Test quantum gate type enumeration."""
    
    def test_gate_type_values(self):
        """Test gate type enumeration values."""
        assert QuantumGateType.HADAMARD.value == "hadamard"
        assert QuantumGateType.CNOT.value == "cnot"
        assert QuantumGateType.BEAM_SPLITTER.value == "beam_splitter"
        assert QuantumGateType.PHASE_SHIFTER.value == "phase_shifter"
    
    def test_photonic_gates_present(self):
        """Test that photonic-specific gates are included."""
        photonic_gates = [
            QuantumGateType.BEAM_SPLITTER,
            QuantumGateType.PHASE_SHIFTER,
            QuantumGateType.DISPLACEMENT
        ]
        
        for gate in photonic_gates:
            assert gate in QuantumGateType


# Integration tests
class TestPhotonicMLIRIntegration:
    """Integration tests for the complete photonic MLIR system."""
    
    @pytest.fixture
    def full_system(self):
        """Create complete system with synthesizer and processor."""
        return {
            "synthesizer": PhotonicMLIRSynthesizer(),
            "processor": QuantumFinancialProcessor()
        }
    
    @pytest.fixture
    def complex_financial_data(self):
        """Create complex financial dataset."""
        return {
            "risk_score": 0.65,
            "volatility": 0.28,
            "assets": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
            "sector": "technology",
            "market_cap": "large",
            "data_points": list(range(50)),
            "correlations": np.random.rand(5, 5).tolist(),
            "returns": np.random.normal(0.1, 0.2, 50).tolist(),
            "fundamentals": {
                "pe_ratio": 25.3,
                "debt_to_equity": 0.4,
                "roa": 0.15
            }
        }
    
    def test_end_to_end_risk_assessment(self, full_system, complex_financial_data):
        """Test complete end-to-end risk assessment workflow."""
        synthesizer = full_system["synthesizer"]
        processor = full_system["processor"]
        
        # Generate circuit
        circuit = synthesizer.synthesize_financial_query(
            query="Comprehensive risk assessment with multi-factor analysis",
            query_type=FinancialQueryType.RISK_ASSESSMENT,
            financial_data=complex_financial_data
        )
        
        # Optimize circuit
        optimized_circuit = synthesizer.optimize_circuit(circuit)
        
        # Process quantum query
        result = processor.process_quantum_query(
            query="Comprehensive risk assessment with multi-factor analysis",
            query_type=FinancialQueryType.RISK_ASSESSMENT,
            financial_data=complex_financial_data
        )
        
        # Validate end-to-end result
        assert result.quantum_advantage > 1.0
        assert result.confidence_score > 0.0
        assert len(optimized_circuit.operations) <= len(circuit.operations)
        assert "enhanced_risk_score" in result.quantum_result
    
    def test_multiple_query_types_processing(self, full_system, complex_financial_data):
        """Test processing multiple query types in sequence."""
        processor = full_system["processor"]
        
        query_types = [
            FinancialQueryType.RISK_ASSESSMENT,
            FinancialQueryType.PORTFOLIO_OPTIMIZATION,
            FinancialQueryType.VOLATILITY_ANALYSIS
        ]
        
        results = []
        for query_type in query_types:
            result = processor.process_quantum_query(
                query=f"Test {query_type.value}",
                query_type=query_type,
                financial_data=complex_financial_data
            )
            results.append(result)
        
        # All results should be valid
        for result in results:
            assert result.quantum_advantage >= 1.0
            assert 0.0 <= result.confidence_score <= 1.0
            assert result.processing_time_ms > 0
        
        # Results should be cached
        assert len(processor.execution_cache) == len(query_types)
    
    def test_performance_scaling(self, full_system):
        """Test system performance with varying data sizes."""
        processor = full_system["processor"]
        
        data_sizes = [10, 50, 100]
        processing_times = []
        
        for size in data_sizes:
            financial_data = {
                "data_points": list(range(size)),
                "risk_score": 0.5,
                "assets": [f"STOCK_{i}" for i in range(min(5, size // 10))]
            }
            
            result = processor.process_quantum_query(
                query="Performance scaling test",
                query_type=FinancialQueryType.RISK_ASSESSMENT,
                financial_data=financial_data
            )
            
            processing_times.append(result.processing_time_ms)
        
        # Processing time should scale reasonably (not exponentially)
        assert all(t > 0 for t in processing_times)
        # Larger datasets may take more time, but not excessively more
        assert processing_times[-1] / processing_times[0] < 10  # Less than 10x slower