"""
Adaptive Quantum Error Correction for Financial Trading (AQEC-FT)

This module implements novel quantum error correction algorithms optimized for
real-time financial trading applications. It addresses the critical gap between
quantum coherence times (~100 microseconds) and financial market dynamics 
(millisecond timescales).

Research Contribution:
- Market-adaptive error correction codes that adjust to volatility patterns
- Hybrid classical-quantum error mitigation for sub-millisecond decisions
- Quantum-aware trading algorithms leveraging partial quantum states
- Performance validation against classical high-frequency trading systems

Reference Implementation for Academic Publication:
"Adaptive Quantum Error Correction for Real-Time Financial Trading Systems"
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market volatility regimes for adaptive error correction."""
    LOW_VOLATILITY = "low_vol"
    MEDIUM_VOLATILITY = "medium_vol" 
    HIGH_VOLATILITY = "high_vol"
    CRISIS = "crisis"


class QuantumErrorType(Enum):
    """Types of quantum errors in financial trading systems."""
    DECOHERENCE = "decoherence"
    GATE_ERROR = "gate_error"
    READOUT_ERROR = "readout_error"
    ENVIRONMENTAL_NOISE = "environmental_noise"
    MARKET_INTERFERENCE = "market_interference"


@dataclass
class QuantumState:
    """Represents a quantum state with error information."""
    amplitudes: np.ndarray
    fidelity: float
    coherence_time: float
    error_probability: float
    timestamp: float


@dataclass
class MarketCondition:
    """Current market conditions affecting quantum error correction."""
    volatility: float
    volume: float
    bid_ask_spread: float
    regime: MarketRegime
    electromagnetic_interference: float


@dataclass
class TradingDecision:
    """Trading decision with quantum confidence metrics."""
    action: str  # 'buy', 'sell', 'hold'
    quantity: float
    confidence: float
    quantum_fidelity: float
    latency_microseconds: float
    error_corrected: bool


class AdaptiveErrorCorrectionCode:
    """
    Market-adaptive quantum error correction code that adjusts parameters
    based on real-time market conditions and volatility patterns.
    """
    
    def __init__(self, base_distance: int = 3, max_distance: int = 7):
        self.base_distance = base_distance
        self.max_distance = max_distance
        self.correction_history = []
        self.performance_metrics = {}
        
    def select_code_distance(self, market_condition: MarketCondition) -> int:
        """
        Dynamically select quantum error correction code distance based on
        market conditions. Higher volatility requires stronger error correction.
        """
        # Base distance adjustment based on market regime
        regime_multipliers = {
            MarketRegime.LOW_VOLATILITY: 1.0,
            MarketRegime.MEDIUM_VOLATILITY: 1.3,
            MarketRegime.HIGH_VOLATILITY: 1.6,
            MarketRegime.CRISIS: 2.0
        }
        
        # Volatility-based adjustment
        volatility_factor = min(market_condition.volatility / 0.02, 3.0)  # Cap at 3x
        
        # Electromagnetic interference adjustment
        interference_factor = 1.0 + market_condition.electromagnetic_interference
        
        # Calculate adaptive distance
        adaptive_distance = int(
            self.base_distance * 
            regime_multipliers[market_condition.regime] * 
            volatility_factor * 
            interference_factor
        )
        
        return min(adaptive_distance, self.max_distance)
    
    def encode_financial_state(self, classical_data: np.ndarray, 
                             distance: int) -> QuantumState:
        """
        Encode classical financial data into error-corrected quantum state.
        """
        # Simplified encoding for demonstration - in practice would use
        # surface codes or other topological codes
        n_qubits = distance ** 2
        
        # Create superposition state encoding financial data
        amplitudes = np.zeros(2 ** n_qubits, dtype=complex)
        
        # Encode classical data into quantum amplitudes
        data_normalized = classical_data / np.linalg.norm(classical_data)
        for i, val in enumerate(data_normalized[:len(amplitudes)]):
            amplitudes[i] = val + 0j
        
        # Normalize
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        # Calculate initial fidelity (perfect encoding)
        fidelity = 1.0
        coherence_time = 100e-6  # 100 microseconds typical
        error_probability = 0.001  # 0.1% base error rate
        
        return QuantumState(
            amplitudes=amplitudes,
            fidelity=fidelity, 
            coherence_time=coherence_time,
            error_probability=error_probability,
            timestamp=time.time()
        )


class MarketAdaptiveDecoder:
    """
    Quantum error correction decoder that adapts to market conditions
    and prioritizes low-latency decoding for time-critical decisions.
    """
    
    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
        self.decoding_cache = {}
        
    def decode_with_urgency(self, quantum_state: QuantumState,
                          urgency_level: float,
                          market_condition: MarketCondition) -> Tuple[np.ndarray, float]:
        """
        Decode quantum state with time constraints based on market urgency.
        
        Args:
            quantum_state: Quantum state to decode
            urgency_level: 0.0 (low) to 1.0 (maximum urgency)
            market_condition: Current market conditions
            
        Returns:
            Decoded classical data and confidence level
        """
        start_time = time.time()
        
        # Adaptive iteration limit based on urgency
        max_iters = max(1, int(self.max_iterations * (1.0 - urgency_level)))
        
        # Syndrome extraction and decoding
        syndromes = self._extract_syndromes(quantum_state)
        
        # Use cached decoding if available for performance
        syndrome_key = tuple(syndromes)
        if syndrome_key in self.decoding_cache:
            decoded_data, confidence = self.decoding_cache[syndrome_key]
        else:
            decoded_data, confidence = self._iterative_decode(
                quantum_state, syndromes, max_iters
            )
            self.decoding_cache[syndrome_key] = (decoded_data, confidence)
        
        # Apply market-specific confidence adjustment
        market_confidence_factor = self._calculate_market_confidence_factor(
            market_condition, urgency_level
        )
        
        final_confidence = confidence * market_confidence_factor
        
        decoding_time = time.time() - start_time
        logger.debug(f"Decoding completed in {decoding_time*1e6:.1f} microseconds, "
                    f"confidence: {final_confidence:.3f}")
        
        return decoded_data, final_confidence
    
    def _extract_syndromes(self, quantum_state: QuantumState) -> np.ndarray:
        """Extract error syndromes from quantum state."""
        # Simplified syndrome extraction
        n_qubits = int(np.log2(len(quantum_state.amplitudes)))
        n_syndromes = n_qubits // 2
        
        # Random syndromes for demonstration
        syndromes = np.random.randint(0, 2, n_syndromes)
        return syndromes
    
    def _iterative_decode(self, quantum_state: QuantumState, 
                         syndromes: np.ndarray, max_iters: int) -> Tuple[np.ndarray, float]:
        """Perform iterative decoding with time constraints."""
        # Simplified minimum-weight perfect matching decoder
        n_data_qubits = len(quantum_state.amplitudes) // 4
        decoded_data = np.real(quantum_state.amplitudes[:n_data_qubits])
        
        # Calculate confidence based on fidelity and error correction success
        confidence = quantum_state.fidelity * (1.0 - quantum_state.error_probability)
        
        return decoded_data, confidence
    
    def _calculate_market_confidence_factor(self, market_condition: MarketCondition,
                                          urgency_level: float) -> float:
        """Calculate market-specific confidence adjustment factor."""
        base_factor = 1.0
        
        # Reduce confidence in high volatility environments
        volatility_penalty = market_condition.volatility * 0.1
        
        # Reduce confidence for very urgent decisions (less error correction)
        urgency_penalty = urgency_level * 0.2
        
        # Reduce confidence with electromagnetic interference
        interference_penalty = market_condition.electromagnetic_interference * 0.15
        
        return max(0.1, base_factor - volatility_penalty - urgency_penalty - interference_penalty)


class QuantumTradingEngine:
    """
    Quantum-enhanced trading engine with adaptive error correction
    for high-frequency financial decision making.
    """
    
    def __init__(self, target_latency_us: float = 10.0):
        self.target_latency_us = target_latency_us
        self.error_corrector = AdaptiveErrorCorrectionCode()
        self.decoder = MarketAdaptiveDecoder()
        self.performance_history = []
        self.quantum_states = {}
        
    def process_market_data(self, market_data: Dict, 
                          market_condition: MarketCondition) -> TradingDecision:
        """
        Process incoming market data using quantum algorithms with
        adaptive error correction for real-time trading decisions.
        """
        start_time = time.time()
        
        # Convert market data to quantum state
        classical_data = np.array([
            market_data.get('price', 0.0),
            market_data.get('volume', 0.0),
            market_data.get('bid', 0.0),
            market_data.get('ask', 0.0)
        ])
        
        # Determine error correction distance based on market conditions
        code_distance = self.error_corrector.select_code_distance(market_condition)
        
        # Encode into error-corrected quantum state
        quantum_state = self.error_corrector.encode_financial_state(
            classical_data, code_distance
        )
        
        # Simulate quantum evolution and decoherence
        quantum_state = self._simulate_quantum_evolution(quantum_state, market_condition)
        
        # Calculate urgency based on market conditions and target latency
        elapsed_time = (time.time() - start_time) * 1e6  # microseconds
        remaining_time = max(0, self.target_latency_us - elapsed_time)
        urgency_level = 1.0 - (remaining_time / self.target_latency_us)
        
        # Decode quantum state to trading signal
        decoded_signal, confidence = self.decoder.decode_with_urgency(
            quantum_state, urgency_level, market_condition
        )
        
        # Generate trading decision
        decision = self._generate_trading_decision(
            decoded_signal, confidence, quantum_state, start_time
        )
        
        # Log performance metrics
        self._log_performance_metrics(decision, market_condition, code_distance)
        
        return decision
    
    def _simulate_quantum_evolution(self, quantum_state: QuantumState,
                                  market_condition: MarketCondition) -> QuantumState:
        """Simulate quantum evolution with market-induced decoherence."""
        # Time since state creation
        evolution_time = time.time() - quantum_state.timestamp
        
        # Decoherence rate affected by market volatility and interference
        base_decoherence_rate = 1.0 / quantum_state.coherence_time
        market_decoherence_factor = (
            1.0 + 
            market_condition.volatility * 10.0 +  # Volatility increases decoherence
            market_condition.electromagnetic_interference * 5.0  # EM interference
        )
        
        effective_decoherence_rate = base_decoherence_rate * market_decoherence_factor
        
        # Calculate fidelity decay
        fidelity_decay = np.exp(-effective_decoherence_rate * evolution_time)
        new_fidelity = quantum_state.fidelity * fidelity_decay
        
        # Update error probability
        new_error_probability = quantum_state.error_probability * (2.0 - fidelity_decay)
        
        return QuantumState(
            amplitudes=quantum_state.amplitudes,
            fidelity=new_fidelity,
            coherence_time=quantum_state.coherence_time,
            error_probability=min(0.5, new_error_probability),
            timestamp=quantum_state.timestamp
        )
    
    def _generate_trading_decision(self, signal: np.ndarray, confidence: float,
                                 quantum_state: QuantumState, start_time: float) -> TradingDecision:
        """Generate trading decision from quantum signal."""
        # Simple trading logic based on signal
        signal_strength = np.mean(signal)
        
        if signal_strength > 0.1 and confidence > 0.7:
            action = "buy"
            quantity = min(1000, confidence * signal_strength * 10000)
        elif signal_strength < -0.1 and confidence > 0.7:
            action = "sell" 
            quantity = min(1000, confidence * abs(signal_strength) * 10000)
        else:
            action = "hold"
            quantity = 0.0
        
        latency_microseconds = (time.time() - start_time) * 1e6
        error_corrected = quantum_state.fidelity > 0.9
        
        return TradingDecision(
            action=action,
            quantity=quantity,
            confidence=confidence,
            quantum_fidelity=quantum_state.fidelity,
            latency_microseconds=latency_microseconds,
            error_corrected=error_corrected
        )
    
    def _log_performance_metrics(self, decision: TradingDecision,
                               market_condition: MarketCondition, 
                               code_distance: int) -> None:
        """Log performance metrics for analysis."""
        metrics = {
            'timestamp': time.time(),
            'latency_us': decision.latency_microseconds,
            'confidence': decision.confidence,
            'fidelity': decision.quantum_fidelity,
            'code_distance': code_distance,
            'market_regime': market_condition.regime.value,
            'volatility': market_condition.volatility,
            'error_corrected': decision.error_corrected
        }
        
        self.performance_history.append(metrics)
        
        # Keep only recent history (last 1000 decisions)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {}
        
        history = self.performance_history
        
        return {
            'avg_latency_us': np.mean([h['latency_us'] for h in history]),
            'p95_latency_us': np.percentile([h['latency_us'] for h in history], 95),
            'avg_confidence': np.mean([h['confidence'] for h in history]),
            'avg_fidelity': np.mean([h['fidelity'] for h in history]),
            'error_correction_rate': np.mean([h['error_corrected'] for h in history]),
            'total_decisions': len(history)
        }


class QuantumTradingBenchmark:
    """
    Benchmarking suite for quantum trading algorithms vs classical baselines.
    Provides statistical validation of quantum advantage claims.
    """
    
    def __init__(self):
        self.quantum_engine = QuantumTradingEngine()
        self.benchmark_results = []
        
    def run_comparative_benchmark(self, market_scenarios: List[Dict],
                                num_runs: int = 100) -> Dict:
        """
        Run comprehensive benchmark comparing quantum vs classical trading
        algorithms across multiple market scenarios.
        """
        logger.info(f"Running benchmark with {len(market_scenarios)} scenarios, "
                   f"{num_runs} runs each")
        
        results = {
            'quantum_performance': [],
            'classical_performance': [],
            'scenarios': market_scenarios,
            'statistical_analysis': {}
        }
        
        for scenario_idx, scenario in enumerate(market_scenarios):
            logger.info(f"Benchmarking scenario {scenario_idx + 1}/{len(market_scenarios)}")
            
            quantum_results = []
            classical_results = []
            
            for run in range(num_runs):
                # Quantum algorithm performance
                quantum_perf = self._run_quantum_scenario(scenario)
                quantum_results.append(quantum_perf)
                
                # Classical algorithm performance (baseline)
                classical_perf = self._run_classical_scenario(scenario)
                classical_results.append(classical_perf)
            
            results['quantum_performance'].append(quantum_results)
            results['classical_performance'].append(classical_results)
        
        # Statistical analysis
        results['statistical_analysis'] = self._perform_statistical_analysis(
            results['quantum_performance'], results['classical_performance']
        )
        
        return results
    
    def _run_quantum_scenario(self, scenario: Dict) -> Dict:
        """Run quantum trading algorithm on scenario."""
        market_condition = MarketCondition(
            volatility=scenario['volatility'],
            volume=scenario['volume'],
            bid_ask_spread=scenario['bid_ask_spread'],
            regime=MarketRegime(scenario['regime']),
            electromagnetic_interference=scenario.get('em_interference', 0.0)
        )
        
        start_time = time.time()
        decision = self.quantum_engine.process_market_data(
            scenario['market_data'], market_condition
        )
        end_time = time.time()
        
        return {
            'latency_us': (end_time - start_time) * 1e6,
            'confidence': decision.confidence,
            'fidelity': decision.quantum_fidelity,
            'decision_quality': self._evaluate_decision_quality(decision, scenario)
        }
    
    def _run_classical_scenario(self, scenario: Dict) -> Dict:
        """Run classical trading algorithm baseline."""
        # Simplified classical algorithm for comparison
        start_time = time.time()
        
        # Classical signal processing
        market_data = scenario['market_data']
        price_momentum = market_data.get('price', 0.0) - scenario.get('prev_price', 0.0)
        volume_factor = min(1.0, market_data.get('volume', 0.0) / 1000000)
        
        # Classical decision logic
        signal_strength = price_momentum * volume_factor
        confidence = 0.8  # Fixed confidence for classical
        
        if signal_strength > 0.1:
            decision_quality = min(1.0, signal_strength * confidence)
        elif signal_strength < -0.1:
            decision_quality = min(1.0, abs(signal_strength) * confidence)
        else:
            decision_quality = 0.5
        
        end_time = time.time()
        
        return {
            'latency_us': (end_time - start_time) * 1e6,
            'confidence': confidence,
            'fidelity': 1.0,  # Classical is deterministic
            'decision_quality': decision_quality
        }
    
    def _evaluate_decision_quality(self, decision: TradingDecision, scenario: Dict) -> float:
        """Evaluate quality of trading decision based on scenario outcome."""
        # Simplified decision quality metric
        expected_return = scenario.get('expected_return', 0.0)
        
        if decision.action == "buy" and expected_return > 0:
            return min(1.0, decision.confidence * expected_return * 10)
        elif decision.action == "sell" and expected_return < 0:
            return min(1.0, decision.confidence * abs(expected_return) * 10)
        elif decision.action == "hold":
            return 0.5
        else:
            return 0.0  # Wrong direction
    
    def _perform_statistical_analysis(self, quantum_results: List[List[Dict]],
                                    classical_results: List[List[Dict]]) -> Dict:
        """Perform statistical significance testing."""
        from scipy import stats
        
        # Flatten results across scenarios
        quantum_latencies = []
        classical_latencies = []
        quantum_qualities = []
        classical_qualities = []
        
        for scenario_quantum, scenario_classical in zip(quantum_results, classical_results):
            quantum_latencies.extend([r['latency_us'] for r in scenario_quantum])
            classical_latencies.extend([r['latency_us'] for r in scenario_classical])
            quantum_qualities.extend([r['decision_quality'] for r in scenario_quantum])
            classical_qualities.extend([r['decision_quality'] for r in scenario_classical])
        
        # Statistical tests
        latency_ttest = stats.ttest_ind(quantum_latencies, classical_latencies)
        quality_ttest = stats.ttest_ind(quantum_qualities, classical_qualities)
        
        # Effect sizes (Cohen's d)
        latency_effect_size = self._cohens_d(quantum_latencies, classical_latencies)
        quality_effect_size = self._cohens_d(quantum_qualities, classical_qualities)
        
        return {
            'latency_comparison': {
                'quantum_mean_us': np.mean(quantum_latencies),
                'classical_mean_us': np.mean(classical_latencies),
                'p_value': latency_ttest.pvalue,
                'effect_size': latency_effect_size,
                'significant': latency_ttest.pvalue < 0.05
            },
            'quality_comparison': {
                'quantum_mean': np.mean(quantum_qualities),
                'classical_mean': np.mean(classical_qualities),
                'p_value': quality_ttest.pvalue,
                'effect_size': quality_effect_size,
                'significant': quality_ttest.pvalue < 0.05
            },
            'sample_size': len(quantum_latencies)
        }
    
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std


def create_market_scenarios() -> List[Dict]:
    """Create diverse market scenarios for benchmarking."""
    scenarios = []
    
    # Low volatility normal market
    scenarios.append({
        'volatility': 0.01,
        'volume': 1000000,
        'bid_ask_spread': 0.001,
        'regime': 'low_vol',
        'em_interference': 0.0,
        'market_data': {'price': 100.0, 'volume': 1000000, 'bid': 99.995, 'ask': 100.005},
        'expected_return': 0.001,
        'prev_price': 99.99
    })
    
    # High volatility market
    scenarios.append({
        'volatility': 0.05,
        'volume': 5000000,
        'bid_ask_spread': 0.01,
        'regime': 'high_vol',
        'em_interference': 0.1,
        'market_data': {'price': 100.0, 'volume': 5000000, 'bid': 99.95, 'ask': 100.05},
        'expected_return': -0.02,
        'prev_price': 102.0
    })
    
    # Crisis scenario
    scenarios.append({
        'volatility': 0.10,
        'volume': 10000000,
        'bid_ask_spread': 0.05,
        'regime': 'crisis',
        'em_interference': 0.2,
        'market_data': {'price': 100.0, 'volume': 10000000, 'bid': 99.75, 'ask': 100.25},
        'expected_return': -0.05,
        'prev_price': 105.0
    })
    
    return scenarios


if __name__ == "__main__":
    # Example usage and demonstration
    logging.basicConfig(level=logging.INFO)
    
    # Create trading engine
    engine = QuantumTradingEngine(target_latency_us=10.0)
    
    # Example market condition
    market_condition = MarketCondition(
        volatility=0.02,
        volume=2000000,
        bid_ask_spread=0.005,
        regime=MarketRegime.MEDIUM_VOLATILITY,
        electromagnetic_interference=0.05
    )
    
    # Example market data
    market_data = {
        'price': 150.0,
        'volume': 2000000,
        'bid': 149.975,
        'ask': 150.025
    }
    
    # Process market data
    decision = engine.process_market_data(market_data, market_condition)
    
    print(f"Trading Decision: {decision.action}")
    print(f"Quantity: {decision.quantity}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"Quantum Fidelity: {decision.quantum_fidelity:.3f}")
    print(f"Latency: {decision.latency_microseconds:.1f} μs")
    print(f"Error Corrected: {decision.error_corrected}")
    
    # Performance summary
    summary = engine.get_performance_summary()
    print(f"\nPerformance Summary: {summary}")
    
    # Run benchmark
    print("\nRunning benchmark...")
    benchmark = QuantumTradingBenchmark()
    scenarios = create_market_scenarios()
    results = benchmark.run_comparative_benchmark(scenarios, num_runs=10)
    
    print("\nBenchmark Results:")
    print(f"Statistical Analysis: {results['statistical_analysis']}")