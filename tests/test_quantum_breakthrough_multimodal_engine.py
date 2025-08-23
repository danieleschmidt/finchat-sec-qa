"""
Comprehensive Test Suite for Quantum Breakthrough Multimodal Engine
Research-Grade Testing Framework with Statistical Validation

Tests Cover:
- Quantum circuit initialization and adaptation
- Multimodal feature extraction and fusion
- Market regime detection accuracy
- Statistical significance validation
- Comparative study reproducibility
- Quantum advantage measurement
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import json

from finchat_sec_qa.quantum_breakthrough_multimodal_engine import (
    QuantumBreakthroughMultimodalEngine,
    QuantumModalityType,
    MarketRegimeQuantum,
    QuantumMultimodalFeature,
    MultimodalAnalysisResult,
    create_quantum_breakthrough_engine
)


class TestQuantumBreakthroughEngine:
    """Comprehensive test suite for quantum breakthrough capabilities."""

    @pytest.fixture
    async def engine(self):
        """Create test engine instance."""
        return await create_quantum_breakthrough_engine(
            quantum_depth=4,  # Smaller for testing
            multimodal_dims=64
        )

    @pytest.fixture
    def sample_documents(self):
        """Sample financial documents for testing."""
        return [
            "The company demonstrates excellent growth with strong fundamentals and positive outlook",
            "Significant risks identified including regulatory challenges and declining market position",
            "Mixed signals with uncertain revenue trends but good management practices",
            "Outstanding performance metrics with record profits and expanding market share",
            "Volatile conditions with potential downside risks and uncertain regulatory environment"
        ]

    @pytest.fixture
    def sample_financial_data(self):
        """Sample financial data for testing."""
        return [
            {'revenue_growth': 0.15, 'debt_ratio': 0.25, 'volatility': 0.15, 'profit_margin': 0.12},
            {'revenue_growth': -0.08, 'debt_ratio': 0.65, 'volatility': 0.45, 'profit_margin': -0.03},
            {'revenue_growth': 0.03, 'debt_ratio': 0.45, 'volatility': 0.35, 'profit_margin': 0.05},
            {'revenue_growth': 0.22, 'debt_ratio': 0.20, 'volatility': 0.12, 'profit_margin': 0.18},
            {'revenue_growth': -0.12, 'debt_ratio': 0.75, 'volatility': 0.55, 'profit_margin': -0.08}
        ]

    @pytest.mark.asyncio
    async def test_quantum_circuit_initialization(self, engine):
        """Test quantum circuit initialization for all modalities."""
        assert len(engine.quantum_circuits) == len(QuantumModalityType)
        
        for modality in QuantumModalityType:
            circuit = engine.quantum_circuits[modality.value]
            assert 'depth' in circuit
            assert 'qubits' in circuit
            assert 'gates' in circuit
            assert 'entanglement_pattern' in circuit
            assert circuit['depth'] > 0
            assert circuit['qubits'] > 0
            assert len(circuit['gates']) > 0

    @pytest.mark.asyncio
    async def test_market_regime_detection(self, engine, sample_financial_data):
        """Test quantum market regime detection accuracy."""
        regimes = []
        for data in sample_financial_data:
            regime = await engine.detect_market_regime(data)
            regimes.append(regime)
            assert isinstance(regime, MarketRegimeQuantum)

        # Should detect different regimes for different market conditions
        unique_regimes = set(regimes)
        assert len(unique_regimes) >= 2, "Should detect multiple market regimes"

    @pytest.mark.asyncio
    async def test_multimodal_feature_extraction(self, engine, sample_documents, sample_financial_data):
        """Test multimodal feature extraction for all modalities."""
        for doc, data in zip(sample_documents, sample_financial_data):
            features = await engine.extract_multimodal_features(doc, data)
            
            assert len(features) >= 4, "Should extract features from multiple modalities"
            
            # Check each feature
            for feature in features:
                assert isinstance(feature, QuantumMultimodalFeature)
                assert len(feature.quantum_state_vector) > 0
                assert len(feature.classical_features) > 0
                assert 0 <= feature.entanglement_score <= 1
                assert 0 <= feature.coherence_measure <= 1
                assert len(feature.uncertainty_bounds) == 2

    @pytest.mark.asyncio
    async def test_quantum_text_processing(self, engine):
        """Test quantum-enhanced text processing."""
        test_texts = [
            "Excellent financial performance and strong growth outlook",
            "Significant losses and declining market position",
            "Neutral position with mixed financial indicators"
        ]

        for text in test_texts:
            features = await engine.extract_multimodal_features(text, {})
            text_feature = next(f for f in features if f.modality_type == QuantumModalityType.TEXT_SEMANTIC)
            
            assert len(text_feature.quantum_state_vector) > 0
            assert np.all(text_feature.quantum_state_vector >= 0)  # Probabilities should be positive
            assert np.isfinite(text_feature.quantum_state_vector).all()

    @pytest.mark.asyncio
    async def test_quantum_numerical_processing(self, engine):
        """Test quantum numerical feature processing."""
        test_data = [
            {'revenue': 100, 'profit': 20, 'debt': 30},
            {'revenue': 80, 'profit': -5, 'debt': 60},
            {'revenue': 120, 'profit': 25, 'debt': 20}
        ]

        for data in test_data:
            features = await engine.extract_multimodal_features("", data)
            numerical_feature = next(f for f in features if f.modality_type == QuantumModalityType.NUMERICAL_FEATURES)
            
            assert len(numerical_feature.quantum_state_vector) > 0
            assert np.all(numerical_feature.quantum_state_vector >= 0)
            assert np.isfinite(numerical_feature.quantum_state_vector).all()

    @pytest.mark.asyncio
    async def test_sentiment_pattern_processing(self, engine):
        """Test quantum sentiment pattern analysis."""
        sentiment_texts = [
            "Excellent growth and positive outlook with strong fundamentals",
            "Poor performance and negative trends with significant risks",
            "Moderate performance with some positive and negative aspects"
        ]

        sentiment_scores = []
        for text in sentiment_texts:
            features = await engine.extract_multimodal_features(text, {})
            sentiment_feature = next(f for f in features if f.modality_type == QuantumModalityType.SENTIMENT_PATTERNS)
            sentiment_scores.append(np.sum(sentiment_feature.quantum_state_vector))

        # Should show variation in sentiment processing
        assert len(set([round(s, 2) for s in sentiment_scores])) > 1, "Should detect sentiment variations"

    @pytest.mark.asyncio
    async def test_multimodal_fusion(self, engine, sample_documents, sample_financial_data):
        """Test quantum multimodal feature fusion."""
        for doc, data in zip(sample_documents, sample_financial_data):
            regime = await engine.detect_market_regime(data)
            features = await engine.extract_multimodal_features(doc, data)
            
            fused_features, fusion_weights = await engine.fuse_multimodal_features(features, regime)
            
            assert len(fused_features) == engine.multimodal_dims
            assert np.isfinite(fused_features).all()
            assert len(fusion_weights) > 0
            assert abs(sum(fusion_weights.values()) - 1.0) < 0.01  # Weights should sum to 1

    @pytest.mark.asyncio
    async def test_quantum_prediction_with_uncertainty(self, engine, sample_documents, sample_financial_data):
        """Test quantum prediction with uncertainty quantification."""
        for doc, data in zip(sample_documents[:3], sample_financial_data[:3]):  # Test subset
            regime = await engine.detect_market_regime(data)
            features = await engine.extract_multimodal_features(doc, data)
            fused_features, _ = await engine.fuse_multimodal_features(features, regime)
            
            prediction, uncertainty = await engine.predict_with_uncertainty(fused_features, regime)
            
            assert 0 <= prediction <= 1, "Prediction should be probability"
            assert 'prediction_mean' in uncertainty
            assert 'prediction_std' in uncertainty
            assert 'confidence_interval_95' in uncertainty
            assert uncertainty['prediction_std'] >= 0

    @pytest.mark.asyncio
    async def test_statistical_significance_validation(self, engine):
        """Test statistical significance validation framework."""
        # Generate test data with known differences
        quantum_results = np.random.normal(0.7, 0.1, 50)  # Higher performance
        classical_results = np.random.normal(0.6, 0.1, 50)  # Lower performance
        
        p_value, is_significant = await engine.validate_statistical_significance(
            quantum_results.tolist(), classical_results.tolist()
        )
        
        assert 0 <= p_value <= 1
        assert isinstance(is_significant, bool)

    @pytest.mark.asyncio
    async def test_comparative_study_framework(self, engine, sample_documents, sample_financial_data):
        """Test comprehensive comparative study framework."""
        # Use subset for faster testing
        docs = sample_documents[:3]
        data = sample_financial_data[:3]
        
        results = await engine.run_comparative_study(docs, data)
        
        # Validate result structure
        assert 'experiment_timestamp' in results
        assert 'samples_processed' in results
        assert 'quantum_mean_accuracy' in results
        assert 'classical_mean_accuracy' in results
        assert 'improvement_percentage' in results
        assert 'statistical_significance' in results
        assert 'quantum_advantage_score' in results
        assert 'reproducibility_hash' in results
        
        # Validate statistical significance structure
        stat_sig = results['statistical_significance']
        assert 'p_value' in stat_sig
        assert 'is_significant' in stat_sig
        assert 'significance_threshold' in stat_sig

    @pytest.mark.asyncio
    async def test_adaptive_fusion_weights(self, engine, sample_documents, sample_financial_data):
        """Test adaptive fusion weights based on market regime."""
        regimes = [
            MarketRegimeQuantum.BULL_QUANTUM_STATE,
            MarketRegimeQuantum.BEAR_QUANTUM_STATE,
            MarketRegimeQuantum.VOLATILITY_SUPERPOSITION,
            MarketRegimeQuantum.UNCERTAINTY_ENTANGLED
        ]
        
        doc = sample_documents[0]
        data = sample_financial_data[0]
        features = await engine.extract_multimodal_features(doc, data)
        
        fusion_weights_by_regime = {}
        for regime in regimes:
            _, fusion_weights = await engine.fuse_multimodal_features(features, regime)
            fusion_weights_by_regime[regime.value] = fusion_weights
        
        # Weights should differ by regime
        assert len(set(str(fw) for fw in fusion_weights_by_regime.values())) > 1

    @pytest.mark.asyncio
    async def test_classical_baseline_comparison(self, engine, sample_documents, sample_financial_data):
        """Test classical baseline provides reasonable comparison."""
        classical_predictions = []
        for doc, data in zip(sample_documents, sample_financial_data):
            pred = engine._classical_baseline_prediction(doc, data)
            classical_predictions.append(pred)
            assert 0 <= pred <= 1, "Classical prediction should be valid probability"
        
        # Should show variation across different inputs
        assert len(set([round(p, 2) for p in classical_predictions])) > 1

    @pytest.mark.asyncio
    async def test_reproducibility_framework(self, engine):
        """Test reproducibility and research framework."""
        hash1 = engine._generate_reproducibility_hash()
        hash2 = engine._generate_reproducibility_hash()
        
        # Hashes should be consistent for same parameters (within time window)
        assert len(hash1) == len(hash2) == 32  # MD5 hash length

    @pytest.mark.asyncio
    async def test_research_results_saving(self, engine):
        """Test research results saving for publication."""
        test_results = {
            'quantum_mean_accuracy': 0.85,
            'classical_mean_accuracy': 0.78,
            'improvement_percentage': 8.97,
            'statistical_significance': {'p_value': 0.001, 'is_significant': True}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            await engine.save_research_results(test_results, output_path)
            
            # Verify file was saved correctly
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            
            assert 'title' in saved_data
            assert 'methodology' in saved_data
            assert 'results' in saved_data
            assert 'reproducibility' in saved_data
            
        finally:
            output_path.unlink()  # Cleanup

    def test_quantum_modality_types(self):
        """Test quantum modality type enumeration."""
        expected_modalities = {
            'text_semantic', 'numerical_features', 'sentiment_patterns', 
            'risk_indicators', 'temporal_sequences', 'cross_modal_correlations'
        }
        actual_modalities = {modality.value for modality in QuantumModalityType}
        assert actual_modalities == expected_modalities

    def test_market_regime_types(self):
        """Test market regime quantum state enumeration."""
        expected_regimes = {
            'bull_quantum', 'bear_quantum', 'volatility_superposition',
            'uncertainty_entangled', 'transition_coherent'
        }
        actual_regimes = {regime.value for regime in MarketRegimeQuantum}
        assert actual_regimes == expected_regimes

    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, engine):
        """Test performance metrics tracking and retrieval."""
        metrics = engine.get_performance_metrics()
        
        assert 'quantum_advantage_scores' in metrics
        assert 'performance_history' in metrics
        assert 'active_circuits' in metrics
        assert 'total_experiments' in metrics
        assert metrics['active_circuits'] > 0

    @pytest.mark.asyncio
    async def test_engine_factory_function(self):
        """Test engine factory function."""
        engine = await create_quantum_breakthrough_engine(quantum_depth=6, multimodal_dims=128)
        
        assert engine.quantum_depth == 6
        assert engine.multimodal_dims == 128
        assert len(engine.quantum_circuits) > 0

    @pytest.mark.asyncio
    async def test_quantum_advantage_measurement(self, engine, sample_documents, sample_financial_data):
        """Test quantum advantage measurement methodology."""
        # Run mini comparative study
        docs = sample_documents[:2]
        data = sample_financial_data[:2]
        
        results = await engine.run_comparative_study(docs, data)
        
        quantum_advantage = results['quantum_advantage_score']
        assert isinstance(quantum_advantage, float)
        assert np.isfinite(quantum_advantage)

    @pytest.mark.asyncio
    async def test_error_handling_robustness(self, engine):
        """Test error handling in quantum processing."""
        # Test with invalid inputs
        empty_features = await engine.extract_multimodal_features("", {})
        assert len(empty_features) > 0  # Should handle empty inputs gracefully
        
        # Test regime detection with minimal data
        regime = await engine.detect_market_regime({})
        assert isinstance(regime, MarketRegimeQuantum)

    @pytest.mark.performance
    async def test_performance_benchmarks(self, engine, sample_documents, sample_financial_data):
        """Test performance benchmarks for research validation."""
        import time
        
        start_time = time.time()
        
        # Process multiple documents
        for doc, data in zip(sample_documents, sample_financial_data):
            regime = await engine.detect_market_regime(data)
            features = await engine.extract_multimodal_features(doc, data)
            fused_features, _ = await engine.fuse_multimodal_features(features, regime)
            _, _ = await engine.predict_with_uncertainty(fused_features, regime)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process reasonably fast for real-time applications
        avg_time_per_doc = processing_time / len(sample_documents)
        assert avg_time_per_doc < 2.0, f"Processing too slow: {avg_time_per_doc:.2f}s per document"


@pytest.mark.integration
class TestQuantumIntegration:
    """Integration tests for quantum breakthrough engine."""

    @pytest.mark.asyncio
    async def test_end_to_end_breakthrough_pipeline(self):
        """Test complete end-to-end quantum breakthrough pipeline."""
        engine = await create_quantum_breakthrough_engine()
        
        # Sample research data
        documents = [
            "Company shows exceptional growth with strong market position and excellent financials",
            "Significant challenges including regulatory risks and declining revenue trends"
        ]
        
        financial_data = [
            {'revenue_growth': 0.18, 'debt_ratio': 0.22, 'volatility': 0.15},
            {'revenue_growth': -0.12, 'debt_ratio': 0.68, 'volatility': 0.52}
        ]
        
        # Run complete analysis
        results = await engine.run_comparative_study(documents, financial_data)
        
        # Validate complete pipeline
        assert results['samples_processed'] == 2
        assert 'quantum_mean_accuracy' in results
        assert 'statistical_significance' in results
        assert 'reproducibility_hash' in results
        
        print(f"✅ Integration test completed: {results['improvement_percentage']:.2f}% improvement")


if __name__ == "__main__":
    # Run demo test
    pytest.main([__file__ + "::TestQuantumBreakthroughEngine::test_quantum_circuit_initialization", "-v"])