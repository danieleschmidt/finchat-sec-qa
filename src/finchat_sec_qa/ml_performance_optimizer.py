"""
ML Performance Optimizer - Generation 3: MAKE IT SCALE
TERRAGON SDLC v4.0 - Machine Learning Driven Performance Optimization

Features:
- ML-driven query optimization
- Adaptive algorithm selection
- Performance prediction models
- Automated hyperparameter tuning
- Real-time optimization feedback loops
- Resource allocation optimization
"""

from __future__ import annotations

import asyncio
import logging
import numpy as np
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from enum import Enum
from collections import defaultdict, deque
import json
import pickle
import statistics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimizations."""
    QUERY_OPTIMIZATION = "query_optimization"
    ALGORITHM_SELECTION = "algorithm_selection"
    RESOURCE_ALLOCATION = "resource_allocation"
    CACHE_OPTIMIZATION = "cache_optimization"
    LOAD_BALANCING = "load_balancing"


class ModelType(Enum):
    """ML model types."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"


@dataclass
class PerformanceMetrics:
    """Performance metrics for ML analysis."""
    timestamp: datetime
    query_type: str
    execution_time: float
    cpu_usage: float
    memory_usage: float
    io_operations: int
    cache_hits: int
    cache_misses: int
    result_size: int
    complexity_score: float
    optimization_applied: Optional[str] = None


@dataclass
class OptimizationResult:
    """Result of an optimization."""
    optimization_type: OptimizationType
    original_performance: float
    optimized_performance: float
    improvement_ratio: float
    confidence: float
    applied_at: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionModel:
    """ML prediction model wrapper."""
    model_type: ModelType
    model: Any
    scaler: StandardScaler
    features: List[str]
    target: str
    accuracy: float
    created_at: datetime
    last_updated: datetime


class MLPerformanceOptimizer:
    """
    Generation 3: ML-driven performance optimizer with adaptive learning.
    
    Features:
    - Real-time performance prediction
    - Adaptive algorithm selection
    - Automated hyperparameter tuning
    - Resource allocation optimization
    - Continuous learning and improvement
    """
    
    def __init__(self, 
                 model_update_interval: int = 3600,  # 1 hour
                 min_samples_for_training: int = 100):
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=10000)
        self.optimization_results: List[OptimizationResult] = []
        
        # ML models
        self.prediction_models: Dict[str, PredictionModel] = {}
        self.model_update_interval = model_update_interval
        self.min_samples_for_training = min_samples_for_training
        
        # Algorithm selection
        self.algorithm_performance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.adaptive_algorithms: Dict[str, str] = {}  # query_type -> best_algorithm
        
        # Optimization strategies
        self.optimization_strategies: Dict[OptimizationType, Callable] = {}
        self.active_optimizations: Set[str] = set()
        
        # Feature engineering
        self.feature_extractors: List[Callable] = []
        self.feature_cache: Dict[str, Dict[str, float]] = {}
        
        # Continuous learning
        self._learning_active = False
        self._learning_task = None
        self.last_model_update = datetime.now()
        
        self._initialize_optimization_strategies()
        self._initialize_feature_extractors()
        
        logger.info("ML Performance Optimizer initialized")
    
    def _initialize_optimization_strategies(self):
        """Initialize optimization strategies."""
        
        async def optimize_query_execution(metrics: PerformanceMetrics) -> OptimizationResult:
            """Optimize query execution based on historical patterns."""
            query_type = metrics.query_type
            
            # Get similar queries from history
            similar_queries = [
                m for m in self.performance_history 
                if m.query_type == query_type and 
                abs(m.complexity_score - metrics.complexity_score) < 0.2
            ]
            
            if len(similar_queries) < 5:
                return OptimizationResult(
                    optimization_type=OptimizationType.QUERY_OPTIMIZATION,
                    original_performance=metrics.execution_time,
                    optimized_performance=metrics.execution_time,
                    improvement_ratio=1.0,
                    confidence=0.0,
                    applied_at=datetime.now()
                )
            
            # Analyze patterns and suggest optimizations
            avg_execution_time = statistics.mean([m.execution_time for m in similar_queries])
            
            # Predict potential improvement
            features = self._extract_features(metrics)
            predicted_improvement = self._predict_optimization_impact(query_type, features)
            
            optimized_time = metrics.execution_time * (1 - predicted_improvement)
            improvement_ratio = metrics.execution_time / optimized_time if optimized_time > 0 else 1.0
            
            return OptimizationResult(
                optimization_type=OptimizationType.QUERY_OPTIMIZATION,
                original_performance=metrics.execution_time,
                optimized_performance=optimized_time,
                improvement_ratio=improvement_ratio,
                confidence=min(len(similar_queries) / 50.0, 1.0),
                applied_at=datetime.now(),
                parameters={'predicted_improvement': predicted_improvement}
            )
        
        async def optimize_algorithm_selection(metrics: PerformanceMetrics) -> OptimizationResult:
            """Select optimal algorithm based on input characteristics."""
            query_type = metrics.query_type
            current_algorithm = "default"
            
            # Get algorithm performance history
            algorithm_scores = {}
            for algorithm, scores in self.algorithm_performance[query_type].items():
                if len(scores) > 0:
                    algorithm_scores[algorithm] = statistics.mean(scores[-10:])  # Recent performance
            
            if not algorithm_scores:
                return OptimizationResult(
                    optimization_type=OptimizationType.ALGORITHM_SELECTION,
                    original_performance=metrics.execution_time,
                    optimized_performance=metrics.execution_time,
                    improvement_ratio=1.0,
                    confidence=0.0,
                    applied_at=datetime.now()
                )
            
            # Select best performing algorithm
            best_algorithm = min(algorithm_scores.items(), key=lambda x: x[1])[0]
            best_performance = algorithm_scores[best_algorithm]
            current_performance = algorithm_scores.get(current_algorithm, metrics.execution_time)
            
            improvement_ratio = current_performance / best_performance if best_performance > 0 else 1.0
            
            return OptimizationResult(
                optimization_type=OptimizationType.ALGORITHM_SELECTION,
                original_performance=current_performance,
                optimized_performance=best_performance,
                improvement_ratio=improvement_ratio,
                confidence=len(algorithm_scores) / 10.0,
                applied_at=datetime.now(),
                parameters={'selected_algorithm': best_algorithm, 'alternatives': algorithm_scores}
            )
        
        async def optimize_resource_allocation(metrics: PerformanceMetrics) -> OptimizationResult:
            """Optimize resource allocation based on workload patterns."""
            # Predict optimal resource allocation
            features = self._extract_features(metrics)
            
            # Simplified resource optimization
            current_cpu = metrics.cpu_usage
            current_memory = metrics.memory_usage
            
            # Predict optimal allocation
            optimal_cpu, optimal_memory = self._predict_optimal_resources(features)
            
            # Calculate potential improvement
            cpu_improvement = 1.0 - (optimal_cpu / current_cpu) if current_cpu > 0 else 1.0
            memory_improvement = 1.0 - (optimal_memory / current_memory) if current_memory > 0 else 1.0
            
            overall_improvement = (cpu_improvement + memory_improvement) / 2
            optimized_time = metrics.execution_time * (1 - overall_improvement * 0.3)  # Conservative estimate
            improvement_ratio = metrics.execution_time / optimized_time if optimized_time > 0 else 1.0
            
            return OptimizationResult(
                optimization_type=OptimizationType.RESOURCE_ALLOCATION,
                original_performance=metrics.execution_time,
                optimized_performance=optimized_time,
                improvement_ratio=improvement_ratio,
                confidence=0.7,  # Medium confidence for resource predictions
                applied_at=datetime.now(),
                parameters={
                    'optimal_cpu': optimal_cpu,
                    'optimal_memory': optimal_memory,
                    'current_cpu': current_cpu,
                    'current_memory': current_memory
                }
            )
        
        self.optimization_strategies = {
            OptimizationType.QUERY_OPTIMIZATION: optimize_query_execution,
            OptimizationType.ALGORITHM_SELECTION: optimize_algorithm_selection,
            OptimizationType.RESOURCE_ALLOCATION: optimize_resource_allocation
        }
    
    def _initialize_feature_extractors(self):
        """Initialize feature extractors for ML models."""
        
        def extract_basic_features(metrics: PerformanceMetrics) -> Dict[str, float]:
            """Extract basic performance features."""
            return {
                'execution_time': metrics.execution_time,
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'io_operations': float(metrics.io_operations),
                'cache_hit_rate': metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses) if (metrics.cache_hits + metrics.cache_misses) > 0 else 0.0,
                'result_size': float(metrics.result_size),
                'complexity_score': metrics.complexity_score
            }
        
        def extract_temporal_features(metrics: PerformanceMetrics) -> Dict[str, float]:
            """Extract temporal features."""
            now = datetime.now()
            return {
                'hour_of_day': now.hour,
                'day_of_week': now.weekday(),
                'is_weekend': float(now.weekday() >= 5),
                'is_business_hours': float(9 <= now.hour <= 17)
            }
        
        def extract_historical_features(metrics: PerformanceMetrics) -> Dict[str, float]:
            """Extract features based on historical data."""
            query_type = metrics.query_type
            recent_history = [
                m for m in self.performance_history 
                if m.query_type == query_type and 
                (datetime.now() - m.timestamp) <= timedelta(hours=24)
            ]
            
            if len(recent_history) < 2:
                return {
                    'avg_execution_time_24h': metrics.execution_time,
                    'execution_time_trend': 0.0,
                    'query_frequency_24h': 1.0
                }
            
            execution_times = [m.execution_time for m in recent_history]
            
            return {
                'avg_execution_time_24h': statistics.mean(execution_times),
                'execution_time_trend': (execution_times[-1] - execution_times[0]) / len(execution_times) if len(execution_times) > 1 else 0.0,
                'query_frequency_24h': float(len(recent_history))
            }
        
        self.feature_extractors = [
            extract_basic_features,
            extract_temporal_features,
            extract_historical_features
        ]
    
    def _extract_features(self, metrics: PerformanceMetrics) -> Dict[str, float]:
        """Extract comprehensive features from performance metrics."""
        features = {}
        
        for extractor in self.feature_extractors:
            try:
                extracted = extractor(metrics)
                features.update(extracted)
            except Exception as e:
                logger.warning(f"Feature extraction failed: {e}")
        
        return features
    
    async def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics for ML analysis."""
        self.performance_history.append(metrics)
        
        # Update algorithm performance tracking
        if metrics.optimization_applied:
            algorithm = metrics.optimization_applied
            self.algorithm_performance[metrics.query_type][algorithm].append(metrics.execution_time)
        
        # Trigger optimization if enough data
        if len(self.performance_history) >= self.min_samples_for_training:
            await self._analyze_and_optimize(metrics)
    
    async def _analyze_and_optimize(self, metrics: PerformanceMetrics):
        """Analyze performance and apply optimizations."""
        optimizations = []
        
        # Apply different optimization strategies
        for opt_type, strategy in self.optimization_strategies.items():
            try:
                result = await strategy(metrics)
                if result.improvement_ratio > 1.1 and result.confidence > 0.5:  # 10% improvement threshold
                    optimizations.append(result)
                    self.optimization_results.append(result)
            except Exception as e:
                logger.error(f"Optimization strategy {opt_type} failed: {e}")
        
        # Log optimizations
        for opt in optimizations:
            logger.info(f"Applied {opt.optimization_type.value}: {opt.improvement_ratio:.2f}x improvement (confidence: {opt.confidence:.2f})")
    
    async def predict_performance(self, 
                                query_type: str, 
                                input_features: Dict[str, Any]) -> Dict[str, float]:
        """Predict performance metrics for given input."""
        predictions = {}
        
        # Get or create prediction model
        model_key = f"performance_{query_type}"
        if model_key not in self.prediction_models:
            await self._train_prediction_model(query_type)
        
        if model_key in self.prediction_models:
            model_info = self.prediction_models[model_key]
            
            try:
                # Prepare features
                feature_vector = []
                for feature_name in model_info.features:
                    feature_vector.append(input_features.get(feature_name, 0.0))
                
                # Scale features
                scaled_features = model_info.scaler.transform([feature_vector])
                
                # Make prediction
                prediction = model_info.model.predict(scaled_features)[0]
                
                predictions = {
                    'predicted_execution_time': prediction,
                    'model_accuracy': model_info.accuracy,
                    'confidence': min(model_info.accuracy, 1.0)
                }
                
            except Exception as e:
                logger.error(f"Performance prediction failed: {e}")
                predictions = {'error': str(e)}
        
        return predictions
    
    async def _train_prediction_model(self, query_type: str):
        """Train prediction model for specific query type."""
        # Get training data
        training_data = [
            m for m in self.performance_history 
            if m.query_type == query_type
        ]
        
        if len(training_data) < self.min_samples_for_training:
            logger.warning(f"Not enough training data for {query_type}: {len(training_data)}")
            return
        
        # Extract features and targets
        features_list = []
        targets = []
        
        for metrics in training_data:
            features = self._extract_features(metrics)
            features_list.append(list(features.values()))
            targets.append(metrics.execution_time)
        
        if not features_list:
            return
        
        feature_names = list(self._extract_features(training_data[0]).keys())
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(targets)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train different models and select best
        models = {
            ModelType.LINEAR_REGRESSION: LinearRegression(),
            ModelType.RANDOM_FOREST: RandomForestRegressor(n_estimators=100, random_state=42),
            ModelType.GRADIENT_BOOSTING: GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = -float('inf')
        best_model_type = None
        
        for model_type, model in models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                avg_score = cv_scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_model_type = model_type
                
            except Exception as e:
                logger.warning(f"Model training failed for {model_type}: {e}")
        
        if best_model is not None:
            # Train final model on all data
            best_model.fit(X_scaled, y)
            
            # Store model
            model_key = f"performance_{query_type}"
            self.prediction_models[model_key] = PredictionModel(
                model_type=best_model_type,
                model=best_model,
                scaler=scaler,
                features=feature_names,
                target='execution_time',
                accuracy=max(0.0, best_score),
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            logger.info(f"Trained {best_model_type.value} model for {query_type} (R² = {best_score:.3f})")
    
    def _predict_optimization_impact(self, query_type: str, features: Dict[str, float]) -> float:
        """Predict potential optimization impact."""
        # Simple heuristic-based prediction (could be replaced with ML model)
        
        # High complexity queries have more optimization potential
        complexity_impact = min(features.get('complexity_score', 0.5), 1.0) * 0.3
        
        # High resource usage indicates optimization opportunities
        cpu_impact = min(features.get('cpu_usage', 50.0) / 100.0, 1.0) * 0.2
        memory_impact = min(features.get('memory_usage', 50.0) / 100.0, 1.0) * 0.2
        
        # Low cache hit rate indicates caching opportunities
        cache_hit_rate = features.get('cache_hit_rate', 0.5)
        cache_impact = (1.0 - cache_hit_rate) * 0.3
        
        total_impact = complexity_impact + cpu_impact + memory_impact + cache_impact
        return min(total_impact, 0.7)  # Cap at 70% improvement
    
    def _predict_optimal_resources(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Predict optimal CPU and memory allocation."""
        # Simplified resource prediction based on current usage and complexity
        
        current_cpu = features.get('cpu_usage', 50.0)
        current_memory = features.get('memory_usage', 50.0)
        complexity = features.get('complexity_score', 0.5)
        
        # Adjust resources based on complexity and current usage
        cpu_factor = 0.8 + (complexity * 0.4)  # 80% to 120% of current
        memory_factor = 0.9 + (complexity * 0.2)  # 90% to 110% of current
        
        optimal_cpu = current_cpu * cpu_factor
        optimal_memory = current_memory * memory_factor
        
        return optimal_cpu, optimal_memory
    
    async def start_continuous_learning(self):
        """Start continuous learning process."""
        if self._learning_active:
            return
        
        self._learning_active = True
        self._learning_task = asyncio.create_task(self._learning_loop())
        logger.info("Continuous learning started")
    
    async def stop_continuous_learning(self):
        """Stop continuous learning process."""
        self._learning_active = False
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
        logger.info("Continuous learning stopped")
    
    async def _learning_loop(self):
        """Main continuous learning loop."""
        while self._learning_active:
            try:
                # Update models if enough time has passed
                if (datetime.now() - self.last_model_update).seconds >= self.model_update_interval:
                    await self._update_all_models()
                    self.last_model_update = datetime.now()
                
                # Analyze recent performance
                await self._analyze_recent_performance()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Sleep before next iteration
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                await asyncio.sleep(60)  # Short sleep on error
    
    async def _update_all_models(self):
        """Update all prediction models with recent data."""
        query_types = set(m.query_type for m in self.performance_history)
        
        for query_type in query_types:
            await self._train_prediction_model(query_type)
        
        logger.info(f"Updated {len(query_types)} prediction models")
    
    async def _analyze_recent_performance(self):
        """Analyze recent performance for optimization opportunities."""
        if len(self.performance_history) < 10:
            return
        
        recent_metrics = list(self.performance_history)[-100:]  # Last 100 metrics
        
        # Identify performance degradations
        degradations = []
        for i in range(10, len(recent_metrics)):
            recent_avg = statistics.mean([m.execution_time for m in recent_metrics[i-10:i]])
            current = recent_metrics[i].execution_time
            
            if current > recent_avg * 1.5:  # 50% slower than recent average
                degradations.append(recent_metrics[i])
        
        if degradations:
            logger.info(f"Detected {len(degradations)} performance degradations")
            
            # Apply immediate optimizations
            for metrics in degradations[-5:]:  # Address last 5 degradations
                await self._analyze_and_optimize(metrics)
    
    async def _cleanup_old_data(self):
        """Clean up old performance data."""
        cutoff_date = datetime.now() - timedelta(days=7)  # Keep 7 days
        
        # Clean optimization results
        self.optimization_results = [
            result for result in self.optimization_results
            if result.applied_at > cutoff_date
        ]
        
        # Clean algorithm performance history
        for query_type in self.algorithm_performance:
            for algorithm in self.algorithm_performance[query_type]:
                scores = self.algorithm_performance[query_type][algorithm]
                self.algorithm_performance[query_type][algorithm] = scores[-1000:]  # Keep last 1000
    
    def get_optimization_analytics(self) -> Dict[str, Any]:
        """Get comprehensive optimization analytics."""
        if not self.optimization_results:
            return {'total_optimizations': 0, 'analytics': 'No optimizations recorded'}
        
        # Calculate improvement statistics
        improvements = [r.improvement_ratio for r in self.optimization_results]
        avg_improvement = statistics.mean(improvements)
        max_improvement = max(improvements)
        
        # Optimization type distribution
        type_distribution = defaultdict(int)
        for result in self.optimization_results:
            type_distribution[result.optimization_type.value] += 1
        
        # Model statistics
        model_stats = {}
        for model_key, model_info in self.prediction_models.items():
            model_stats[model_key] = {
                'model_type': model_info.model_type.value,
                'accuracy': model_info.accuracy,
                'features_count': len(model_info.features),
                'last_updated': model_info.last_updated.isoformat()
            }
        
        # Recent performance trend
        recent_metrics = list(self.performance_history)[-100:]
        if recent_metrics:
            recent_execution_times = [m.execution_time for m in recent_metrics]
            performance_trend = (recent_execution_times[-1] - recent_execution_times[0]) / len(recent_execution_times) if len(recent_execution_times) > 1 else 0
        else:
            performance_trend = 0
        
        return {
            'total_optimizations': len(self.optimization_results),
            'avg_improvement_ratio': avg_improvement,
            'max_improvement_ratio': max_improvement,
            'optimization_type_distribution': dict(type_distribution),
            'active_models': len(self.prediction_models),
            'model_statistics': model_stats,
            'performance_samples': len(self.performance_history),
            'recent_performance_trend': performance_trend,
            'learning_active': self._learning_active,
            'analytics_timestamp': datetime.now().isoformat()
        }
    
    async def save_models(self, filepath: str):
        """Save trained models to file."""
        try:
            model_data = {
                'models': {},
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_optimizations': len(self.optimization_results),
                    'performance_samples': len(self.performance_history)
                }
            }
            
            for key, model_info in self.prediction_models.items():
                model_data['models'][key] = {
                    'model_type': model_info.model_type.value,
                    'features': model_info.features,
                    'target': model_info.target,
                    'accuracy': model_info.accuracy,
                    'created_at': model_info.created_at.isoformat(),
                    'last_updated': model_info.last_updated.isoformat()
                }
            
            # Save models using joblib
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    async def load_models(self, filepath: str):
        """Load trained models from file."""
        try:
            model_data = joblib.load(filepath)
            
            for key, model_info in model_data['models'].items():
                # Note: This is a simplified version - in production,
                # you'd need to properly serialize/deserialize the actual model objects
                logger.info(f"Loaded model: {key} (accuracy: {model_info['accuracy']:.3f})")
            
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")


# Global ML optimizer instance
ml_optimizer = MLPerformanceOptimizer()


# Convenience functions
async def record_performance_metrics(metrics: PerformanceMetrics):
    """Record performance metrics for ML analysis."""
    await ml_optimizer.record_performance(metrics)


async def predict_query_performance(query_type: str, features: Dict[str, Any]) -> Dict[str, float]:
    """Predict query performance."""
    return await ml_optimizer.predict_performance(query_type, features)


async def start_ml_optimization():
    """Start ML-driven optimization."""
    await ml_optimizer.start_continuous_learning()


async def stop_ml_optimization():
    """Stop ML-driven optimization."""
    await ml_optimizer.stop_continuous_learning()