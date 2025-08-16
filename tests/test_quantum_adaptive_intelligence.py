"""
Comprehensive test suite for Quantum Adaptive Intelligence Engine.

Tests cover:
1. Core functionality and API contracts
2. Statistical significance validation
3. Quantum circuit adaptation logic
4. Research reproducibility
5. Performance benchmarks
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

from finchat_sec_qa.quantum_adaptive_intelligence import (
    QuantumAdaptiveIntelligence,
    QuantumVariationalAutoencoder,
    AdaptiveQuantumConfig,
    MarketRegime,
    QuantumAdaptationResult,
    create_research_experiment
)


class TestQuantumVariationalAutoencoder:
    """Test suite for Quantum Variational Autoencoder."""
    
    def test_initialization(self):
        """Test QVAE initialization."""
        qvae = QuantumVariationalAutoencoder(n_qubits=4, n_latent=2)
        
        assert qvae.n_qubits == 4
        assert qvae.n_latent == 2
        assert qvae.circuit_depth == 6  # default
        assert qvae.encoder_params is not None
        assert qvae.decoder_params is not None
        assert qvae.latent_params is not None
    
    def test_encode_decode_cycle(self):
        """Test encode-decode cycle maintains data structure."""
        qvae = QuantumVariationalAutoencoder(n_qubits=4, n_latent=2)
        
        # Test data
        data = np.random.rand(10, 4)
        
        # Encode
        latent_features = qvae.encode(data)
        assert latent_features.shape == (10, 2)
        
        # Decode
        reconstructed = qvae.decode(latent_features)
        assert reconstructed.shape == (10, 4)
    
    def test_parameter_properties(self):
        """Test quantum circuit parameter properties."""
        from finchat_sec_qa.quantum_adaptive_intelligence import QuantumCircuitParameters
        
        theta = np.array([0.1, 0.2, 0.3])
        phi = np.array([0.4, 0.5])
        entanglement = [(0, 1), (1, 2)]
        measurement = ["Z", "X"]
        
        params = QuantumCircuitParameters(
            theta=theta,
            phi=phi,
            entanglement_structure=entanglement,
            measurement_basis=measurement
        )
        
        assert params.parameter_count == 5  # 3 + 2
        assert len(params.entanglement_structure) == 2
        assert len(params.measurement_basis) == 2


class TestAdaptiveQuantumConfig:
    """Test suite for adaptive configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AdaptiveQuantumConfig()
        
        assert config.base_qubits == 8
        assert config.max_qubits == 16
        assert config.adaptation_threshold == 0.1
        assert config.statistical_significance_level == 0.05
        assert config.ensemble_size == 5
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AdaptiveQuantumConfig(
            base_qubits=6,
            max_qubits=12,
            adaptation_threshold=0.05,
            statistical_significance_level=0.01
        )
        
        assert config.base_qubits == 6
        assert config.max_qubits == 12
        assert config.adaptation_threshold == 0.05
        assert config.statistical_significance_level == 0.01


class TestQuantumAdaptiveIntelligence:
    """Test suite for Quantum Adaptive Intelligence Engine."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample financial data for testing."""
        np.random.seed(42)
        
        # Create sample financial data
        n_samples = 100
        n_features = 6
        
        dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
        
        data = {
            "returns": np.random.normal(0.001, 0.02, n_samples),
            "volatility": np.random.uniform(0.1, 0.3, n_samples),
            "volume": np.random.uniform(1000, 5000, n_samples),
            "ma_ratio": np.random.uniform(0.9, 1.1, n_samples),
            "rsi": np.random.uniform(20, 80, n_samples),
            "macd": np.random.normal(0, 0.1, n_samples)
        }
        
        df = pd.DataFrame(data, index=dates)
        target = np.random.choice([0, 1], size=n_samples)
        
        return df, target
    
    @pytest.fixture
    def quantum_intelligence(self):
        """Create quantum intelligence instance for testing."""
        config = AdaptiveQuantumConfig(
            base_qubits=4,  # Smaller for testing
            max_qubits=6,
            adaptation_threshold=0.1
        )
        return QuantumAdaptiveIntelligence(config)
    
    def test_initialization(self, quantum_intelligence):
        """Test quantum intelligence initialization."""
        qi = quantum_intelligence
        
        assert qi.config.base_qubits == 4
        assert qi.qvae.n_qubits == 4
        assert qi.qvae.n_latent == 2
        assert qi.performance_baseline is None
        assert qi.current_regime is None
        assert len(qi.adaptation_history) == 0
    
    def test_market_regime_detection(self, quantum_intelligence, sample_data):
        """Test market regime detection."""
        qi = quantum_intelligence
        financial_data, _ = sample_data
        
        regime = qi._detect_market_regime(financial_data)
        assert isinstance(regime, MarketRegime)
        
        # Test with insufficient data
        small_data = financial_data.head(10)
        regime_small = qi._detect_market_regime(small_data)
        assert regime_small == MarketRegime.SIDEWAYS_MARKET
    
    def test_regime_feature_extraction(self, quantum_intelligence, sample_data):
        """Test regime feature extraction."""
        qi = quantum_intelligence
        financial_data, _ = sample_data
        
        regime_features = qi._extract_regime_features(financial_data)
        assert regime_features.shape[0] == len(financial_data)
        assert regime_features.shape[1] > 0
    
    def test_fit_method(self, quantum_intelligence, sample_data):
        """Test model fitting."""
        qi = quantum_intelligence
        financial_data, target = sample_data
        
        # Fit the model
        trained_qi = qi.fit(financial_data, target)
        
        assert trained_qi is qi  # Returns self
        assert qi.performance_baseline is not None
        assert qi.current_regime is not None
        assert qi.scaler is not None
    
    def test_predict_method(self, quantum_intelligence, sample_data):
        """Test prediction method."""
        qi = quantum_intelligence
        financial_data, target = sample_data
        
        # First fit the model
        qi.fit(financial_data, target)
        
        # Test predictions
        test_data = financial_data.tail(20)
        predictions, metadata = qi.predict(test_data)
        
        assert len(predictions) == len(test_data)
        assert "regime" in metadata
        assert "confidence" in metadata
        assert "quantum_features_used" in metadata
        assert "timestamp" in metadata
    
    def test_statistical_significance_calculation(self, quantum_intelligence):
        """Test statistical significance calculation."""
        qi = quantum_intelligence
        
        baseline = 0.6
        new_performance = 0.65
        n_samples = 100
        
        p_value = qi._calculate_statistical_significance(baseline, new_performance, n_samples)
        
        assert 0 <= p_value <= 1
        assert isinstance(p_value, float)
    
    def test_confidence_interval_calculation(self, quantum_intelligence):
        """Test confidence interval calculation."""
        qi = quantum_intelligence
        
        performance = 0.75
        ci = qi._calculate_confidence_interval(performance)
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < performance < ci[1]
    
    def test_adaptation_summary(self, quantum_intelligence, sample_data):
        """Test adaptation summary generation."""
        qi = quantum_intelligence
        financial_data, target = sample_data
        
        # Test with no adaptations
        summary = qi.get_adaptation_summary()
        assert "message" in summary
        
        # Fit model to generate adaptations
        qi.fit(financial_data, target)
        summary = qi.get_adaptation_summary()
        
        expected_keys = [
            "total_adaptations", "significant_adaptations", 
            "total_performance_improvement", "average_improvement",
            "regimes_encountered", "baseline_performance"
        ]
        
        for key in expected_keys:
            assert key in summary
    
    def test_regime_adaptation(self, quantum_intelligence, sample_data):
        """Test adaptation to new market regime."""
        qi = quantum_intelligence
        financial_data, target = sample_data
        
        qi.fit(financial_data, target)
        
        # Test adaptation to different regime
        original_qubits = qi.qvae.n_qubits
        qi._adapt_to_regime(MarketRegime.HIGH_VOLATILITY, financial_data.values)
        
        # Should have adapted configuration
        assert qi.qvae.n_qubits >= original_qubits  # Might increase for high volatility


class TestResearchExperiment:
    """Test suite for research experiment functionality."""
    
    def test_create_research_experiment(self):
        """Test research experiment creation."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 50  # Small for testing
        
        financial_data = pd.DataFrame({
            "returns": np.random.normal(0, 0.01, n_samples),
            "volatility": np.random.uniform(0.1, 0.2, n_samples),
            "volume": np.random.uniform(1000, 2000, n_samples)
        })
        
        target = np.random.choice([0, 1], size=n_samples)
        
        # Create experiment
        model, results = create_research_experiment(
            financial_data, target, "test_experiment"
        )
        
        assert isinstance(model, QuantumAdaptiveIntelligence)
        assert isinstance(results, dict)
        
        # Check required result keys
        expected_keys = [
            "experiment_name", "training_time", "model_summary",
            "configuration", "data_shape", "timestamp", "research_contribution"
        ]
        
        for key in expected_keys:
            assert key in results
        
        # Check research contribution structure
        contribution = results["research_contribution"]
        assert "novel_algorithms" in contribution
        assert "performance_metrics" in contribution
        assert isinstance(contribution["novel_algorithms"], list)


class TestQuantumAdaptationResult:
    """Test adaptation result data structure."""
    
    def test_adaptation_result_creation(self):
        """Test adaptation result creation."""
        result = QuantumAdaptationResult(
            regime=MarketRegime.BULL_MARKET,
            optimal_qubits=8,
            optimal_depth=6,
            performance_improvement=0.05,
            statistical_significance=0.03,
            confidence_interval=(0.65, 0.75),
            adaptation_time=1234567890.0
        )
        
        assert result.regime == MarketRegime.BULL_MARKET
        assert result.optimal_qubits == 8
        assert result.optimal_depth == 6
        assert result.performance_improvement == 0.05
        assert result.statistical_significance == 0.03
        assert result.confidence_interval == (0.65, 0.75)
        assert result.adaptation_time == 1234567890.0


class TestPerformanceAndScalability:
    """Test performance and scalability characteristics."""
    
    def test_encoding_performance(self):
        """Test quantum encoding performance with different data sizes."""
        qvae = QuantumVariationalAutoencoder(n_qubits=6, n_latent=3)
        
        # Test different batch sizes
        for batch_size in [1, 10, 50]:
            data = np.random.rand(batch_size, 6)
            
            start_time = datetime.now()
            latent_features = qvae.encode(data)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            assert latent_features.shape == (batch_size, 3)
            assert duration < 10.0  # Should complete within 10 seconds
    
    def test_memory_usage(self):
        """Test memory usage with larger quantum circuits."""
        # Test with maximum practical size for simulation
        qvae = QuantumVariationalAutoencoder(n_qubits=8, n_latent=4)
        
        data = np.random.rand(100, 8)
        latent_features = qvae.encode(data)
        
        assert latent_features.shape == (100, 4)
        assert not np.any(np.isnan(latent_features))
        assert not np.any(np.isinf(latent_features))


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_qubit_configuration(self):
        """Test handling of invalid qubit configurations."""
        with pytest.raises((ValueError, AssertionError)):
            QuantumVariationalAutoencoder(n_qubits=0, n_latent=2)
        
        with pytest.raises((ValueError, AssertionError)):
            QuantumVariationalAutoencoder(n_qubits=4, n_latent=0)
    
    def test_mismatched_data_dimensions(self):
        """Test handling of mismatched data dimensions."""
        qvae = QuantumVariationalAutoencoder(n_qubits=4, n_latent=2)
        
        # Data with wrong number of features
        wrong_data = np.random.rand(10, 6)  # Should be 4 features
        
        # Should handle gracefully or raise appropriate error
        try:
            latent_features = qvae.encode(wrong_data)
            # If it doesn't raise an error, should handle dimension mismatch
            assert latent_features.shape[0] == 10
            assert latent_features.shape[1] == 2
        except (ValueError, IndexError):
            # Acceptable to raise error for dimension mismatch
            pass
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        qvae = QuantumVariationalAutoencoder(n_qubits=4, n_latent=2)
        
        empty_data = np.array([]).reshape(0, 4)
        latent_features = qvae.encode(empty_data)
        
        assert latent_features.shape == (0, 2)


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Generate realistic financial data
        np.random.seed(42)
        n_samples = 200
        
        # Create time series with market patterns
        dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
        
        returns = []
        volatility = []
        
        # Create bull/bear market cycles
        for i in range(n_samples):
            cycle_position = (i % 60) / 60  # 60-day cycles
            
            if cycle_position < 0.5:  # Bull phase
                ret = 0.002 + np.random.normal(0, 0.01)
                vol = 0.15 + np.random.normal(0, 0.02)
            else:  # Bear phase
                ret = -0.001 + np.random.normal(0, 0.015)
                vol = 0.25 + np.random.normal(0, 0.03)
            
            returns.append(ret)
            volatility.append(max(0.05, vol))  # Minimum volatility
        
        financial_data = pd.DataFrame({
            "returns": returns,
            "volatility": volatility,
            "volume": np.random.uniform(1000, 5000, n_samples),
            "rsi": np.random.uniform(20, 80, n_samples)
        }, index=dates)
        
        # Create meaningful target
        target = (np.array(returns) > np.median(returns)).astype(int)
        
        # Run complete experiment
        model, results = create_research_experiment(
            financial_data, target, "integration_test"
        )
        
        # Verify results
        assert model.performance_baseline is not None
        assert model.current_regime is not None
        
        # Test prediction on new data
        test_data = financial_data.tail(30)
        predictions, metadata = model.predict(test_data)
        
        assert len(predictions) == 30
        assert all(pred in [0, 1] for pred in predictions)
        assert 0.0 <= metadata["confidence"] <= 1.0
    
    def test_regime_change_adaptation(self):
        """Test adaptation across multiple regime changes."""
        config = AdaptiveQuantumConfig(
            base_qubits=4,
            max_qubits=6,
            regime_detection_window=20
        )
        
        qi = QuantumAdaptiveIntelligence(config)
        
        # Create data with clear regime changes
        n_samples = 120
        
        # Create 3 distinct regimes
        regime1_data = pd.DataFrame({
            "returns": np.random.normal(0.01, 0.005, 40),  # Bull market
            "volatility": np.random.uniform(0.1, 0.15, 40)
        })
        
        regime2_data = pd.DataFrame({
            "returns": np.random.normal(-0.005, 0.02, 40),  # Bear market
            "volatility": np.random.uniform(0.25, 0.35, 40)
        })
        
        regime3_data = pd.DataFrame({
            "returns": np.random.normal(0.001, 0.008, 40),  # Sideways
            "volatility": np.random.uniform(0.12, 0.18, 40)
        })
        
        full_data = pd.concat([regime1_data, regime2_data, regime3_data], ignore_index=True)
        full_data.index = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
        
        target = np.random.choice([0, 1], size=n_samples)
        
        # Train on full data
        qi.fit(full_data, target)
        
        # Test predictions across regimes
        for regime_start in [0, 40, 80]:
            regime_data = full_data.iloc[regime_start:regime_start+20]
            predictions, metadata = qi.predict(regime_data)
            
            assert len(predictions) == 20
            assert "regime" in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])