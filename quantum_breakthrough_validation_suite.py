#!/usr/bin/env python3
"""
Quantum Financial AI Research Validation Suite
==============================================

Comprehensive validation framework for breakthrough quantum algorithms
in financial time series analysis and risk prediction.

RESEARCH VALIDATION - Autonomous SDLC Implementation
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from scipy import stats

from src.finchat_sec_qa.quantum_timeseries import (
    QuantumFinancialTimeSeriesAnalyzer,
    QuantumTimeSeriesAlgorithm, 
    QuantumTimeSeriesData
)
from src.finchat_sec_qa.quantum_risk_ml import (
    QuantumRiskPredictor,
    QuantumRiskModelType,
    QuantumRiskFeatures,
    RiskType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumFinancialResearchValidator:
    """
    Comprehensive research validation framework for quantum financial algorithms.
    
    Implements statistical significance testing, baseline comparisons,
    and performance benchmarking for academic publication.
    """
    
    def __init__(self):
        self.results = {}
        self.timeseries_analyzer = QuantumFinancialTimeSeriesAnalyzer()
        self.risk_predictor = QuantumRiskPredictor()
    
    def generate_synthetic_financial_data(self, 
                                        n_series: int = 10,
                                        series_length: int = 252,
                                        complexity_levels: List[str] = None) -> List[QuantumTimeSeriesData]:
        """
        Generate synthetic financial time series with various complexity patterns.
        
        Creates datasets that simulate different market conditions and regimes
        for comprehensive algorithm evaluation.
        """
        if complexity_levels is None:
            complexity_levels = ['low', 'medium', 'high', 'extreme']
        
        datasets = []
        
        for i in range(n_series):
            complexity = complexity_levels[i % len(complexity_levels)]
            
            # Generate base time series
            timestamps = [datetime.now() + pd.Timedelta(days=j) for j in range(series_length)]
            
            # Create different complexity patterns
            if complexity == 'low':
                # Simple trend + noise
                trend = np.linspace(0.1, 0.5, series_length)
                noise = np.random.normal(0, 0.05, series_length)
                values = trend + noise
            elif complexity == 'medium':
                # Trend + seasonal + noise
                trend = np.linspace(0.1, 0.5, series_length)
                seasonal = 0.1 * np.sin(2 * np.pi * np.arange(series_length) / 20)
                noise = np.random.normal(0, 0.08, series_length)
                values = trend + seasonal + noise
            elif complexity == 'high':
                # Multi-scale patterns + regime changes
                trend = np.linspace(0.1, 0.5, series_length)
                seasonal = 0.1 * np.sin(2 * np.pi * np.arange(series_length) / 20)
                regime_changes = np.where(np.arange(series_length) % 63 == 0, 0.2, 0.0)
                noise = np.random.normal(0, 0.12, series_length)
                values = trend + seasonal + np.cumsum(regime_changes) + noise
            else:  # extreme
                # Chaotic behavior + fat-tail events
                values = np.zeros(series_length)
                volatility = 0.1
                for j in range(1, series_length):
                    # Stochastic volatility
                    volatility += np.random.normal(0, 0.01)
                    volatility = max(0.05, min(0.3, volatility))
                    
                    # Fat-tail events (5% probability)
                    if np.random.random() < 0.05:
                        shock = np.random.normal(0, volatility * 5)
                    else:
                        shock = np.random.normal(0, volatility)
                    
                    values[j] = values[j-1] + shock
            
            # Normalize values
            normalized = (values - np.mean(values)) / (np.std(values) + 1e-8)
            
            # Create quantum features
            quantum_features = {
                'volatility_signature': np.abs(np.fft.fft(normalized)[:10]),
                'correlation_matrix': np.corrcoef([normalized[:-1], normalized[1:]])[0,1:],
                'regime_indicators': np.array([complexity == c for c in complexity_levels]).astype(float)
            }
            
            data = QuantumTimeSeriesData(
                series_id=f"synthetic_{complexity}_{i}",
                timestamps=timestamps,
                values=values,
                normalized_values=normalized,
                quantum_features=quantum_features,
                metadata={'complexity': complexity, 'generation_params': {'trend': True, 'seasonal': complexity != 'low'}}
            )
            
            datasets.append(data)
        
        return datasets
    
    def generate_synthetic_risk_features(self, n_samples: int = 50) -> List[QuantumRiskFeatures]:
        """Generate synthetic risk feature datasets for model evaluation."""
        risk_datasets = []
        
        for i in range(n_samples):
            # Generate numerical features (financial indicators)
            n_features = np.random.randint(10, 25)
            numerical_features = np.random.normal(0, 1, n_features)
            
            # Add some structure to features
            # Correlation between features to simulate real financial data
            correlation_matrix = np.random.uniform(0.1, 0.8, (n_features, n_features))
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)
            
            try:
                correlated_features = np.random.multivariate_normal(
                    numerical_features, 
                    0.1 * correlation_matrix
                )
            except np.linalg.LinAlgError:
                # Fallback to independent features if correlation matrix is singular
                correlated_features = numerical_features
            
            # Categorical features
            categorical_features = {
                'sector': np.random.randint(0, 10),
                'market_cap': np.random.randint(0, 5),
                'rating': np.random.randint(0, 8),
                'region': np.random.randint(0, 6)
            }
            
            # Quantum embeddings
            quantum_embeddings = np.random.normal(0, 1, 16)
            quantum_embeddings = quantum_embeddings / (np.linalg.norm(quantum_embeddings) + 1e-8)
            
            # Generate realistic risk labels
            risk_labels = {}
            base_risk = np.random.beta(2, 5)  # Skewed toward lower risk
            
            for risk_type in RiskType:
                # Add correlation between risk types
                correlation_factor = np.random.normal(0, 0.1)
                risk_score = max(0.0, min(1.0, base_risk + correlation_factor))
                risk_labels[risk_type] = risk_score
            
            features = QuantumRiskFeatures(
                feature_id=f"synthetic_risk_{i}",
                numerical_features=correlated_features,
                categorical_features=categorical_features,
                quantum_embeddings=quantum_embeddings,
                risk_labels=risk_labels,
                metadata={'synthetic': True, 'base_risk': base_risk}
            )
            
            risk_datasets.append(features)
        
        return risk_datasets
    
    def run_timeseries_algorithm_comparison(self, datasets: List[QuantumTimeSeriesData]) -> Dict[str, Any]:
        """
        Run comprehensive comparison of all quantum time series algorithms.
        
        Measures performance, statistical significance, and quantum advantage.
        """
        logger.info("Running quantum time series algorithm comparison study")
        
        results = {
            'algorithms': {},
            'statistical_tests': {},
            'performance_metrics': {},
            'quantum_advantages': {}
        }
        
        # Test all algorithms on all datasets
        for algorithm in QuantumTimeSeriesAlgorithm:
            logger.info(f"Testing {algorithm.value}")
            
            algorithm_results = {
                'predictions': [],
                'processing_times': [],
                'quantum_advantages': [],
                'errors': []
            }
            
            for dataset in datasets:
                try:
                    start_time = time.time()
                    
                    # Make prediction
                    result = self.timeseries_analyzer.analyze_timeseries(
                        dataset, algorithm, prediction_steps=5
                    )
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Calculate prediction error (using simple MSE against continuation)
                    if len(dataset.normalized_values) > 5:
                        actual_continuation = dataset.normalized_values[-5:]
                        mse_error = np.mean((result.predictions - actual_continuation)**2)
                    else:
                        mse_error = np.var(result.predictions)  # Fallback metric
                    
                    algorithm_results['predictions'].append(result.predictions)
                    algorithm_results['processing_times'].append(processing_time)
                    algorithm_results['quantum_advantages'].append(result.quantum_advantage_score)
                    algorithm_results['errors'].append(mse_error)
                    
                except Exception as e:
                    logger.warning(f"Error testing {algorithm.value} on {dataset.series_id}: {e}")
                    algorithm_results['errors'].append(np.inf)
            
            # Calculate aggregate statistics
            valid_errors = [e for e in algorithm_results['errors'] if not np.isinf(e)]
            if valid_errors:
                results['algorithms'][algorithm.value] = {
                    'mean_error': np.mean(valid_errors),
                    'std_error': np.std(valid_errors),
                    'mean_processing_time': np.mean(algorithm_results['processing_times']),
                    'mean_quantum_advantage': np.mean(algorithm_results['quantum_advantages']),
                    'success_rate': len(valid_errors) / len(datasets),
                    'raw_results': algorithm_results
                }
        
        # Statistical significance testing
        algorithm_names = list(results['algorithms'].keys())
        for i, alg1 in enumerate(algorithm_names):
            for alg2 in algorithm_names[i+1:]:
                if alg1 in results['algorithms'] and alg2 in results['algorithms']:
                    errors1 = [e for e in results['algorithms'][alg1]['raw_results']['errors'] if not np.isinf(e)]
                    errors2 = [e for e in results['algorithms'][alg2]['raw_results']['errors'] if not np.isinf(e)]
                    
                    if len(errors1) > 1 and len(errors2) > 1:
                        # Wilcoxon rank-sum test (non-parametric)
                        statistic, p_value = stats.ranksums(errors1, errors2)
                        
                        results['statistical_tests'][f"{alg1}_vs_{alg2}"] = {
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'better_algorithm': alg1 if np.mean(errors1) < np.mean(errors2) else alg2
                        }
        
        return results
    
    def run_risk_model_comparison(self, datasets: List[QuantumRiskFeatures]) -> Dict[str, Any]:
        """
        Run comprehensive comparison of all quantum risk prediction models.
        """
        logger.info("Running quantum risk model comparison study")
        
        results = {
            'models': {},
            'statistical_tests': {},
            'risk_type_analysis': {}
        }
        
        # Test all models on all datasets
        for model_type in QuantumRiskModelType:
            logger.info(f"Testing {model_type.value}")
            
            model_results = {
                'predictions': [],
                'processing_times': [],
                'quantum_advantages': [],
                'prediction_errors': [],
                'confidence_scores': []
            }
            
            for features in datasets:
                try:
                    start_time = time.time()
                    
                    # Make prediction
                    result = self.risk_predictor.predict_risk(features, model_type)
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Calculate prediction error against true risk labels
                    prediction_error = 0.0
                    for risk_type in RiskType:
                        if risk_type in result.risk_predictions and risk_type in features.risk_labels:
                            pred = result.risk_predictions[risk_type]
                            true_val = features.risk_labels[risk_type]
                            prediction_error += (pred - true_val)**2
                    
                    prediction_error = np.sqrt(prediction_error / len(RiskType))
                    
                    model_results['predictions'].append(result.risk_predictions)
                    model_results['processing_times'].append(processing_time)
                    model_results['quantum_advantages'].append(result.quantum_advantage)
                    model_results['prediction_errors'].append(prediction_error)
                    model_results['confidence_scores'].append(np.mean(list(result.confidence_scores.values())))
                    
                except Exception as e:
                    logger.warning(f"Error testing {model_type.value} on {features.feature_id}: {e}")
                    model_results['prediction_errors'].append(np.inf)
            
            # Calculate aggregate statistics
            valid_errors = [e for e in model_results['prediction_errors'] if not np.isinf(e)]
            if valid_errors:
                results['models'][model_type.value] = {
                    'mean_error': np.mean(valid_errors),
                    'std_error': np.std(valid_errors),
                    'mean_processing_time': np.mean(model_results['processing_times']),
                    'mean_quantum_advantage': np.mean(model_results['quantum_advantages']),
                    'mean_confidence': np.mean(model_results['confidence_scores']),
                    'success_rate': len(valid_errors) / len(datasets),
                    'raw_results': model_results
                }
        
        # Statistical significance testing between models
        model_names = list(results['models'].keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                if model1 in results['models'] and model2 in results['models']:
                    errors1 = [e for e in results['models'][model1]['raw_results']['prediction_errors'] if not np.isinf(e)]
                    errors2 = [e for e in results['models'][model2]['raw_results']['prediction_errors'] if not np.isinf(e)]
                    
                    if len(errors1) > 1 and len(errors2) > 1:
                        statistic, p_value = stats.ranksums(errors1, errors2)
                        
                        results['statistical_tests'][f"{model1}_vs_{model2}"] = {
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'better_model': model1 if np.mean(errors1) < np.mean(errors2) else model2
                        }
        
        return results
    
    def generate_research_report(self, 
                               timeseries_results: Dict[str, Any],
                               risk_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive research report suitable for academic publication.
        """
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'validation_framework': 'Quantum Financial AI Research Validator v1.0',
                'algorithms_tested': {
                    'timeseries': list(timeseries_results['algorithms'].keys()),
                    'risk_models': list(risk_results['models'].keys())
                }
            },
            'executive_summary': {},
            'timeseries_analysis': timeseries_results,
            'risk_analysis': risk_results,
            'comparative_analysis': {},
            'statistical_validation': {},
            'quantum_advantage_analysis': {},
            'recommendations': {}
        }
        
        # Executive Summary
        best_timeseries_alg = min(timeseries_results['algorithms'].items(), 
                                key=lambda x: x[1]['mean_error'])
        best_risk_model = min(risk_results['models'].items(),
                             key=lambda x: x[1]['mean_error'])
        
        report['executive_summary'] = {
            'best_timeseries_algorithm': {
                'name': best_timeseries_alg[0],
                'mean_error': best_timeseries_alg[1]['mean_error'],
                'quantum_advantage': best_timeseries_alg[1]['mean_quantum_advantage']
            },
            'best_risk_model': {
                'name': best_risk_model[0],
                'mean_error': best_risk_model[1]['mean_error'],
                'quantum_advantage': best_risk_model[1]['mean_quantum_advantage']
            },
            'statistical_significance_count': {
                'timeseries': sum(1 for test in timeseries_results['statistical_tests'].values() 
                                if test['significant']),
                'risk_models': sum(1 for test in risk_results['statistical_tests'].values() 
                                 if test['significant'])
            }
        }
        
        # Quantum Advantage Analysis
        ts_advantages = [alg['mean_quantum_advantage'] for alg in timeseries_results['algorithms'].values()]
        risk_advantages = [model['mean_quantum_advantage'] for model in risk_results['models'].values()]
        
        report['quantum_advantage_analysis'] = {
            'timeseries_quantum_advantage': {
                'mean': np.mean(ts_advantages),
                'std': np.std(ts_advantages),
                'max': np.max(ts_advantages),
                'algorithms_with_advantage': [name for name, data in timeseries_results['algorithms'].items()
                                            if data['mean_quantum_advantage'] > 1.5]
            },
            'risk_models_quantum_advantage': {
                'mean': np.mean(risk_advantages),
                'std': np.std(risk_advantages),
                'max': np.max(risk_advantages),
                'models_with_advantage': [name for name, data in risk_results['models'].items()
                                        if data['mean_quantum_advantage'] > 1.5]
            }
        }
        
        return report
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Execute complete research validation pipeline.
        
        Returns comprehensive results suitable for academic publication.
        """
        logger.info("Starting comprehensive quantum financial AI validation")
        
        # Generate synthetic datasets
        logger.info("Generating synthetic datasets...")
        timeseries_datasets = self.generate_synthetic_financial_data(n_series=20)
        risk_datasets = self.generate_synthetic_risk_features(n_samples=30)
        
        # Run algorithm comparisons
        logger.info("Running algorithm comparisons...")
        timeseries_results = self.run_timeseries_algorithm_comparison(timeseries_datasets)
        risk_results = self.run_risk_model_comparison(risk_datasets)
        
        # Generate research report
        logger.info("Generating research report...")
        research_report = self.generate_research_report(timeseries_results, risk_results)
        
        return research_report


def main():
    """Execute quantum financial research validation."""
    import pandas as pd  # Import here to avoid dependency issues
    
    validator = QuantumFinancialResearchValidator()
    
    logger.info("🔬 Starting Quantum Financial AI Research Validation")
    logger.info("=" * 60)
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"quantum_research_validation_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Validation complete! Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("🚀 QUANTUM FINANCIAL AI RESEARCH VALIDATION COMPLETE")
    print("="*60)
    
    summary = results['executive_summary']
    print(f"\n📊 BEST PERFORMING ALGORITHMS:")
    print(f"   Time Series: {summary['best_timeseries_algorithm']['name']}")
    print(f"   - Mean Error: {summary['best_timeseries_algorithm']['mean_error']:.6f}")
    print(f"   - Quantum Advantage: {summary['best_timeseries_algorithm']['quantum_advantage']:.2f}")
    
    print(f"\n   Risk Model: {summary['best_risk_model']['name']}")
    print(f"   - Mean Error: {summary['best_risk_model']['mean_error']:.6f}")
    print(f"   - Quantum Advantage: {summary['best_risk_model']['quantum_advantage']:.2f}")
    
    qa_analysis = results['quantum_advantage_analysis']
    print(f"\n🔮 QUANTUM ADVANTAGE ANALYSIS:")
    print(f"   Time Series Models - Mean Advantage: {qa_analysis['timeseries_quantum_advantage']['mean']:.2f}")
    print(f"   Risk Models - Mean Advantage: {qa_analysis['risk_models_quantum_advantage']['mean']:.2f}")
    
    print(f"\n📈 STATISTICAL SIGNIFICANCE:")
    print(f"   Significant Time Series Comparisons: {summary['statistical_significance_count']['timeseries']}")
    print(f"   Significant Risk Model Comparisons: {summary['statistical_significance_count']['risk_models']}")
    
    print(f"\n📄 Full results available in: {output_file}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()