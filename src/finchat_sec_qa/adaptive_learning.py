"""
Adaptive Learning and Continuous Improvement Engine.

This module implements self-improving patterns that learn from usage and 
automatically evolve the system's capabilities for better performance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import threading
from collections import defaultdict, deque

import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from .config import get_config
from .logging_utils import configure_logging

logger = logging.getLogger(__name__)


class LearningPattern(Enum):
    """Types of learning patterns the system can adapt to."""
    QUERY_OPTIMIZATION = "query_optimization"
    CACHE_STRATEGY = "cache_strategy"
    RESOURCE_ALLOCATION = "resource_allocation"
    ERROR_PREVENTION = "error_prevention"
    USER_BEHAVIOR = "user_behavior"
    PERFORMANCE_TUNING = "performance_tuning"


@dataclass
class AdaptationEvent:
    """Represents an adaptation made by the learning system."""
    timestamp: datetime
    pattern_type: LearningPattern
    trigger: str
    adaptation_made: str
    impact_metrics: Dict[str, float] = field(default_factory=dict)
    validation_score: float = 0.0


@dataclass
class PerformanceMetrics:
    """Performance metrics tracked for learning."""
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_rates: deque = field(default_factory=lambda: deque(maxlen=1000))
    throughput: deque = field(default_factory=lambda: deque(maxlen=1000))
    resource_usage: deque = field(default_factory=lambda: deque(maxlen=1000))
    user_satisfaction: deque = field(default_factory=lambda: deque(maxlen=1000))


class AdaptiveLearningEngine:
    """
    Adaptive learning engine that continuously improves system performance
    through ML-driven analysis and automated optimizations.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".cache" / "finchat_sec_qa" / "adaptive_learning"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics = PerformanceMetrics()
        self.adaptations: List[AdaptationEvent] = []
        
        # ML models for different learning patterns
        self.performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Adaptive parameters that get tuned
        self.adaptive_params = {
            'cache_ttl': 3600,  # seconds
            'batch_size': 32,
            'timeout_threshold': 30.0,  # seconds
            'error_retry_count': 3,
            'resource_allocation_factor': 1.0
        }
        
        # Learning state
        self.feature_history: List[List[float]] = []
        self.target_history: List[float] = []
        self.anomaly_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Background learning
        self._running = False
        self._learning_thread: Optional[threading.Thread] = None
        
        self._load_learning_state()
        configure_logging()

    def start_adaptive_learning(self) -> None:
        """Start the adaptive learning process."""
        if self._running:
            return
            
        self._running = True
        self._learning_thread = threading.Thread(
            target=self._learning_loop, 
            daemon=True
        )
        self._learning_thread.start()
        logger.info("Adaptive learning engine started")

    def stop_adaptive_learning(self) -> None:
        """Stop the adaptive learning process."""
        self._running = False
        if self._learning_thread:
            self._learning_thread.join(timeout=5)
        logger.info("Adaptive learning engine stopped")

    def _learning_loop(self) -> None:
        """Main learning loop running in background thread."""
        while self._running:
            try:
                # Collect and analyze metrics
                self._collect_metrics()
                
                # Detect performance patterns
                self._analyze_patterns()
                
                # Make adaptive improvements
                self._make_adaptations()
                
                # Validate adaptations
                self._validate_adaptations()
                
                # Train predictive models
                self._train_models()
                
                # Save learning state
                self._save_learning_state()
                
                # Sleep for 2 minutes
                time.sleep(120)
                
            except Exception as e:
                logger.error(f"Error in adaptive learning loop: {e}")
                time.sleep(30)  # Back off on error

    def _collect_metrics(self) -> None:
        """Collect current system metrics for learning."""
        try:
            current_time = time.time()
            
            # Simulate metric collection (in practice, integrate with monitoring)
            response_time = self._get_current_response_time()
            error_rate = self._get_current_error_rate()
            throughput = self._get_current_throughput()
            resource_usage = self._get_current_resource_usage()
            
            self.metrics.response_times.append(response_time)
            self.metrics.error_rates.append(error_rate)
            self.metrics.throughput.append(throughput)
            self.metrics.resource_usage.append(resource_usage)
            
            # Create feature vector for ML
            features = [response_time, error_rate, throughput, resource_usage, current_time % 86400]
            self.feature_history.append(features)
            
            # Performance score as target (higher is better)
            performance_score = self._calculate_performance_score(
                response_time, error_rate, throughput, resource_usage
            )
            self.target_history.append(performance_score)
            
            # Keep history bounded
            if len(self.feature_history) > 10000:
                self.feature_history = self.feature_history[-5000:]
                self.target_history = self.target_history[-5000:]
                
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    def _analyze_patterns(self) -> None:
        """Analyze performance patterns to identify optimization opportunities."""
        if len(self.feature_history) < 50:
            return
            
        try:
            # Recent performance analysis
            recent_features = np.array(self.feature_history[-100:])
            recent_targets = np.array(self.target_history[-100:])
            
            # Detect anomalies in performance
            if len(recent_features) >= 10:
                anomalies = self.anomaly_detector.fit_predict(recent_features)
                anomaly_indices = np.where(anomalies == -1)[0]
                
                if len(anomaly_indices) > 0:
                    logger.info(f"Detected {len(anomaly_indices)} performance anomalies")
                    self._handle_performance_anomalies(recent_features[anomaly_indices])
            
            # Analyze trends
            if len(recent_targets) >= 20:
                trend = np.polyfit(range(len(recent_targets)), recent_targets, 1)[0]
                if trend < -0.01:  # Declining performance
                    logger.info("Detected declining performance trend, triggering adaptations")
                    self._trigger_performance_adaptation()
                    
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")

    def _make_adaptations(self) -> None:
        """Make adaptive improvements based on learned patterns."""
        try:
            # Query optimization adaptations
            self._adapt_query_optimization()
            
            # Cache strategy adaptations
            self._adapt_cache_strategy()
            
            # Resource allocation adaptations
            self._adapt_resource_allocation()
            
            # Error prevention adaptations
            self._adapt_error_prevention()
            
        except Exception as e:
            logger.error(f"Error making adaptations: {e}")

    def _adapt_query_optimization(self) -> None:
        """Adapt query optimization parameters."""
        if len(self.metrics.response_times) < 10:
            return
            
        recent_response_times = list(self.metrics.response_times)[-50:]
        avg_response_time = np.mean(recent_response_times)
        
        # If response times are high, adjust batch size
        if avg_response_time > 1.0 and self.adaptive_params['batch_size'] > 16:
            old_batch_size = self.adaptive_params['batch_size']
            self.adaptive_params['batch_size'] = max(16, int(old_batch_size * 0.8))
            
            self.adaptations.append(AdaptationEvent(
                timestamp=datetime.now(),
                pattern_type=LearningPattern.QUERY_OPTIMIZATION,
                trigger=f"High response time: {avg_response_time:.2f}s",
                adaptation_made=f"Reduced batch size from {old_batch_size} to {self.adaptive_params['batch_size']}",
                impact_metrics={'response_time_before': avg_response_time}
            ))
            
            logger.info(f"Adapted batch size: {old_batch_size} -> {self.adaptive_params['batch_size']}")

    def _adapt_cache_strategy(self) -> None:
        """Adapt caching parameters based on usage patterns."""
        if len(self.metrics.response_times) < 20:
            return
            
        # Analyze cache effectiveness
        recent_response_times = list(self.metrics.response_times)[-100:]
        cache_hit_improvement = self._estimate_cache_effectiveness()
        
        if cache_hit_improvement > 0.2:  # 20% improvement potential
            old_ttl = self.adaptive_params['cache_ttl']
            self.adaptive_params['cache_ttl'] = min(7200, int(old_ttl * 1.2))  # Increase TTL
            
            self.adaptations.append(AdaptationEvent(
                timestamp=datetime.now(),
                pattern_type=LearningPattern.CACHE_STRATEGY,
                trigger=f"Cache effectiveness improvement potential: {cache_hit_improvement:.1%}",
                adaptation_made=f"Increased cache TTL from {old_ttl} to {self.adaptive_params['cache_ttl']}",
                impact_metrics={'cache_effectiveness': cache_hit_improvement}
            ))
            
            logger.info(f"Adapted cache TTL: {old_ttl} -> {self.adaptive_params['cache_ttl']}")

    def _adapt_resource_allocation(self) -> None:
        """Adapt resource allocation based on usage patterns."""
        if len(self.metrics.resource_usage) < 20:
            return
            
        recent_usage = list(self.metrics.resource_usage)[-50:]
        avg_usage = np.mean(recent_usage)
        
        # Adjust resource allocation factor
        if avg_usage > 0.8:  # High usage
            old_factor = self.adaptive_params['resource_allocation_factor']
            self.adaptive_params['resource_allocation_factor'] = min(2.0, old_factor * 1.1)
            
            self.adaptations.append(AdaptationEvent(
                timestamp=datetime.now(),
                pattern_type=LearningPattern.RESOURCE_ALLOCATION,
                trigger=f"High resource usage: {avg_usage:.1%}",
                adaptation_made=f"Increased allocation factor from {old_factor:.2f} to {self.adaptive_params['resource_allocation_factor']:.2f}",
                impact_metrics={'resource_usage': avg_usage}
            ))
            
        elif avg_usage < 0.3:  # Low usage
            old_factor = self.adaptive_params['resource_allocation_factor']
            self.adaptive_params['resource_allocation_factor'] = max(0.5, old_factor * 0.9)
            
            self.adaptations.append(AdaptationEvent(
                timestamp=datetime.now(),
                pattern_type=LearningPattern.RESOURCE_ALLOCATION,
                trigger=f"Low resource usage: {avg_usage:.1%}",
                adaptation_made=f"Decreased allocation factor from {old_factor:.2f} to {self.adaptive_params['resource_allocation_factor']:.2f}",
                impact_metrics={'resource_usage': avg_usage}
            ))

    def _adapt_error_prevention(self) -> None:
        """Adapt error prevention mechanisms."""
        if len(self.metrics.error_rates) < 10:
            return
            
        recent_errors = list(self.metrics.error_rates)[-50:]
        avg_error_rate = np.mean(recent_errors)
        
        # If error rate is high, increase retry count
        if avg_error_rate > 0.05 and self.adaptive_params['error_retry_count'] < 5:
            old_retry_count = self.adaptive_params['error_retry_count']
            self.adaptive_params['error_retry_count'] = min(5, old_retry_count + 1)
            
            self.adaptations.append(AdaptationEvent(
                timestamp=datetime.now(),
                pattern_type=LearningPattern.ERROR_PREVENTION,
                trigger=f"High error rate: {avg_error_rate:.1%}",
                adaptation_made=f"Increased retry count from {old_retry_count} to {self.adaptive_params['error_retry_count']}",
                impact_metrics={'error_rate': avg_error_rate}
            ))
            
            logger.info(f"Adapted retry count: {old_retry_count} -> {self.adaptive_params['error_retry_count']}")

    def _validate_adaptations(self) -> None:
        """Validate the effectiveness of recent adaptations."""
        recent_adaptations = [
            a for a in self.adaptations 
            if (datetime.now() - a.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        for adaptation in recent_adaptations:
            if adaptation.validation_score == 0.0:  # Not yet validated
                validation_score = self._calculate_adaptation_impact(adaptation)
                adaptation.validation_score = validation_score
                
                if validation_score > 0.1:  # 10% improvement
                    logger.info(f"Successful adaptation: {adaptation.adaptation_made} "
                              f"(improvement: {validation_score:.1%})")
                elif validation_score < -0.05:  # 5% degradation
                    logger.warning(f"Adaptation caused degradation: {adaptation.adaptation_made} "
                                 f"(impact: {validation_score:.1%})")
                    self._revert_adaptation(adaptation)

    def _train_models(self) -> None:
        """Train predictive models with collected data."""
        if len(self.feature_history) < 100:
            return
            
        try:
            # Prepare training data
            X = np.array(self.feature_history[-1000:])  # Recent 1000 samples
            y = np.array(self.target_history[-1000:])
            
            if len(X) >= 50:
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train performance predictor
                cv_scores = cross_val_score(self.performance_predictor, X_scaled, y, cv=5)
                if np.mean(cv_scores) > 0.5:  # Reasonable performance
                    self.performance_predictor.fit(X_scaled, y)
                    logger.debug(f"Performance predictor trained with CV score: {np.mean(cv_scores):.3f}")
                
                # Update anomaly detector
                self.anomaly_detector.fit(X_scaled)
                
        except Exception as e:
            logger.error(f"Error training models: {e}")

    def _handle_performance_anomalies(self, anomaly_features: np.ndarray) -> None:
        """Handle detected performance anomalies."""
        for features in anomaly_features:
            response_time, error_rate, throughput, resource_usage, time_of_day = features
            
            # Log anomaly details
            logger.warning(f"Performance anomaly detected - "
                         f"Response time: {response_time:.2f}s, "
                         f"Error rate: {error_rate:.1%}, "
                         f"Throughput: {throughput:.1f}")
            
            # Store anomaly pattern for learning
            anomaly_key = f"{int(time_of_day // 3600)}"  # Hour of day
            self.anomaly_patterns[anomaly_key].extend(features.tolist())

    def _trigger_performance_adaptation(self) -> None:
        """Trigger immediate performance adaptations."""
        logger.info("Triggering emergency performance adaptations")
        
        # Reduce batch size for faster response
        if self.adaptive_params['batch_size'] > 8:
            self.adaptive_params['batch_size'] = max(8, self.adaptive_params['batch_size'] // 2)
        
        # Reduce cache TTL to ensure fresher data
        if self.adaptive_params['cache_ttl'] > 600:
            self.adaptive_params['cache_ttl'] = max(600, self.adaptive_params['cache_ttl'] // 2)

    def _calculate_performance_score(
        self, response_time: float, error_rate: float, 
        throughput: float, resource_usage: float
    ) -> float:
        """Calculate overall performance score (higher is better)."""
        # Normalize and weight different metrics
        response_score = max(0, 1 - (response_time / 5.0))  # Penalty for slow response
        error_score = max(0, 1 - (error_rate * 10))  # Heavy penalty for errors
        throughput_score = min(1, throughput / 100)  # Normalized throughput
        resource_score = 1 - abs(resource_usage - 0.6)  # Optimal around 60% usage
        
        return (response_score * 0.3 + error_score * 0.4 + 
                throughput_score * 0.2 + resource_score * 0.1)

    def _estimate_cache_effectiveness(self) -> float:
        """Estimate potential cache effectiveness improvement."""
        # Simplified estimation - in practice would analyze cache hit rates
        if len(self.metrics.response_times) < 10:
            return 0.0
        
        response_times = list(self.metrics.response_times)[-20:]
        variance = np.var(response_times)
        
        # Higher variance suggests more cache misses
        return min(0.5, variance / 10)

    def _calculate_adaptation_impact(self, adaptation: AdaptationEvent) -> float:
        """Calculate the impact of a specific adaptation."""
        # Compare performance before and after adaptation
        adaptation_time = adaptation.timestamp.timestamp()
        recent_metrics = [
            self.target_history[i] for i, t in enumerate(self.feature_history)
            if len(t) > 4 and abs(t[4] - (adaptation_time % 86400)) < 1800  # Within 30 minutes
        ]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        # Simple before/after comparison
        baseline = adaptation.impact_metrics.get('performance_baseline', 0.5)
        current = np.mean(recent_metrics[-5:]) if len(recent_metrics) >= 5 else recent_metrics[-1]
        
        return (current - baseline) / baseline if baseline > 0 else 0.0

    def _revert_adaptation(self, adaptation: AdaptationEvent) -> None:
        """Revert a harmful adaptation."""
        logger.info(f"Reverting adaptation: {adaptation.adaptation_made}")
        
        # This would contain logic to revert specific parameter changes
        # For now, just log the reversion
        adaptation.adaptation_made += " [REVERTED]"

    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current adaptive parameters."""
        return self.adaptive_params.copy()

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of adaptations made."""
        return [
            {
                'timestamp': a.timestamp.isoformat(),
                'pattern_type': a.pattern_type.value,
                'trigger': a.trigger,
                'adaptation_made': a.adaptation_made,
                'validation_score': a.validation_score,
                'impact_metrics': a.impact_metrics
            }
            for a in self.adaptations[-20:]  # Recent 20 adaptations
        ]

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process."""
        if len(self.target_history) < 10:
            return {'status': 'insufficient_data'}
        
        recent_performance = np.mean(self.target_history[-50:])
        historical_performance = np.mean(self.target_history[-200:-50]) if len(self.target_history) >= 200 else recent_performance
        
        improvement = (recent_performance - historical_performance) / historical_performance if historical_performance > 0 else 0
        
        return {
            'recent_performance_score': recent_performance,
            'performance_improvement': improvement,
            'total_adaptations': len(self.adaptations),
            'successful_adaptations': len([a for a in self.adaptations if a.validation_score > 0.05]),
            'current_parameters': self.adaptive_params,
            'anomaly_patterns_detected': len(self.anomaly_patterns),
            'model_training_samples': len(self.feature_history)
        }

    # Simulated metric collection methods (in practice, integrate with real monitoring)
    def _get_current_response_time(self) -> float:
        return 0.3 + np.random.normal(0, 0.1)
    
    def _get_current_error_rate(self) -> float:
        return max(0, min(0.1, 0.02 + np.random.normal(0, 0.01)))
    
    def _get_current_throughput(self) -> float:
        return max(0, 50 + np.random.normal(0, 10))
    
    def _get_current_resource_usage(self) -> float:
        return max(0, min(1, 0.6 + np.random.normal(0, 0.2)))

    def _save_learning_state(self) -> None:
        """Save learning state to disk."""
        try:
            state_file = self.storage_path / "learning_state.json"
            state = {
                'adaptive_params': self.adaptive_params,
                'adaptations': [
                    {
                        'timestamp': a.timestamp.isoformat(),
                        'pattern_type': a.pattern_type.value,
                        'trigger': a.trigger,
                        'adaptation_made': a.adaptation_made,
                        'validation_score': a.validation_score,
                        'impact_metrics': a.impact_metrics
                    }
                    for a in self.adaptations[-100:]  # Keep recent 100
                ],
                'performance_history': {
                    'features': self.feature_history[-500:],  # Keep recent 500
                    'targets': self.target_history[-500:]
                },
                'anomaly_patterns': dict(self.anomaly_patterns)
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")

    def _load_learning_state(self) -> None:
        """Load learning state from disk."""
        try:
            state_file = self.storage_path / "learning_state.json"
            if not state_file.exists():
                return
                
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore adaptive parameters
            self.adaptive_params.update(state.get('adaptive_params', {}))
            
            # Restore adaptations
            self.adaptations = [
                AdaptationEvent(
                    timestamp=datetime.fromisoformat(a['timestamp']),
                    pattern_type=LearningPattern(a['pattern_type']),
                    trigger=a['trigger'],
                    adaptation_made=a['adaptation_made'],
                    validation_score=a['validation_score'],
                    impact_metrics=a['impact_metrics']
                )
                for a in state.get('adaptations', [])
            ]
            
            # Restore performance history
            perf_history = state.get('performance_history', {})
            self.feature_history = perf_history.get('features', [])
            self.target_history = perf_history.get('targets', [])
            
            # Restore anomaly patterns
            anomaly_data = state.get('anomaly_patterns', {})
            for key, values in anomaly_data.items():
                self.anomaly_patterns[key] = values
            
            logger.info(f"Loaded learning state with {len(self.adaptations)} adaptations")
            
        except Exception as e:
            logger.error(f"Error loading learning state: {e}")


# Global instance for adaptive learning
_global_learning_engine: Optional[AdaptiveLearningEngine] = None


def get_learning_engine() -> AdaptiveLearningEngine:
    """Get the global adaptive learning engine instance."""
    global _global_learning_engine
    if _global_learning_engine is None:
        _global_learning_engine = AdaptiveLearningEngine()
        _global_learning_engine.start_adaptive_learning()
    return _global_learning_engine


def get_adaptive_parameters() -> Dict[str, Any]:
    """Get current adaptive parameters."""
    engine = get_learning_engine()
    return engine.get_current_parameters()


def get_learning_insights() -> Dict[str, Any]:
    """Get learning insights and performance improvements."""
    engine = get_learning_engine()
    return engine.get_learning_insights()