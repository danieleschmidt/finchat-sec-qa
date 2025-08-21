"""
Hypothesis-Driven Development Framework v4.0
Implements autonomous hypothesis-driven development with A/B testing and metrics.
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import random

logger = logging.getLogger(__name__)


class HypothesisStatus(Enum):
    """Hypothesis testing status"""
    FORMULATED = "formulated"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class ExperimentType(Enum):
    """Types of experiments"""
    AB_TEST = "ab_test"
    MULTIVARIATE = "multivariate"
    PERFORMANCE = "performance"
    ALGORITHM_COMPARISON = "algorithm_comparison"
    USER_BEHAVIOR = "user_behavior"


@dataclass
class SuccessCriteria:
    """Success criteria for hypothesis validation"""
    metric_name: str
    target_value: float
    comparison_operator: str  # ">=", "<=", ">", "<", "=="
    confidence_level: float = 0.95
    statistical_power: float = 0.80


@dataclass
class ExperimentMetrics:
    """Metrics collected during experiment"""
    metric_name: str
    values: List[float] = field(default_factory=list)
    mean: Optional[float] = None
    std_dev: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    sample_size: int = 0
    
    def add_measurement(self, value: float):
        """Add a measurement to the metrics"""
        self.values.append(value)
        self.sample_size = len(self.values)
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate statistical measures"""
        if len(self.values) > 0:
            self.mean = statistics.mean(self.values)
            if len(self.values) > 1:
                self.std_dev = statistics.stdev(self.values)
                # Simple confidence interval (assumes normal distribution)
                margin_of_error = 1.96 * (self.std_dev / (len(self.values) ** 0.5))
                self.confidence_interval = (
                    self.mean - margin_of_error,
                    self.mean + margin_of_error
                )


@dataclass
class Hypothesis:
    """Research hypothesis with testable predictions"""
    id: str
    name: str
    description: str
    prediction: str
    success_criteria: List[SuccessCriteria]
    status: HypothesisStatus = HypothesisStatus.FORMULATED
    confidence_score: float = 0.0
    statistical_significance: Optional[float] = None
    effect_size: Optional[float] = None
    experiment_results: List[Dict[str, Any]] = field(default_factory=list)
    validation_date: Optional[str] = None


class ExperimentFramework(ABC):
    """Abstract base class for experiment implementations"""
    
    def __init__(self, name: str, hypothesis: Hypothesis):
        self.name = name
        self.hypothesis = hypothesis
        self.experiment_type: ExperimentType = ExperimentType.AB_TEST
        self.baseline_metrics: Dict[str, ExperimentMetrics] = {}
        self.treatment_metrics: Dict[str, ExperimentMetrics] = {}
        self.is_running = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    @abstractmethod
    async def setup_experiment(self) -> Dict[str, Any]:
        """Setup the experiment environment"""
        pass
    
    @abstractmethod
    async def run_baseline(self) -> Dict[str, Any]:
        """Run baseline implementation"""
        pass
    
    @abstractmethod
    async def run_treatment(self) -> Dict[str, Any]:
        """Run treatment implementation"""
        pass
    
    @abstractmethod
    async def collect_metrics(self, variant: str) -> Dict[str, float]:
        """Collect metrics for a variant"""
        pass
    
    async def run_experiment(self, sample_size: int = 100) -> Dict[str, Any]:
        """Run complete experiment"""
        logger.info(f"🧪 Starting experiment: {self.name}")
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            # Setup experiment
            setup_result = await self.setup_experiment()
            
            # Initialize metrics
            for criteria in self.hypothesis.success_criteria:
                self.baseline_metrics[criteria.metric_name] = ExperimentMetrics(criteria.metric_name)
                self.treatment_metrics[criteria.metric_name] = ExperimentMetrics(criteria.metric_name)
            
            # Run experiment iterations
            for i in range(sample_size):
                # Randomly assign to baseline or treatment (50/50 split)
                if random.random() < 0.5:
                    await self.run_baseline()
                    metrics = await self.collect_metrics("baseline")
                    for metric_name, value in metrics.items():
                        if metric_name in self.baseline_metrics:
                            self.baseline_metrics[metric_name].add_measurement(value)
                else:
                    await self.run_treatment()
                    metrics = await self.collect_metrics("treatment")
                    for metric_name, value in metrics.items():
                        if metric_name in self.treatment_metrics:
                            self.treatment_metrics[metric_name].add_measurement(value)
            
            self.end_time = time.time()
            self.is_running = False
            
            # Analyze results
            results = self._analyze_results()
            
            logger.info(f"✅ Experiment completed: {self.name}")
            logger.info(f"   Duration: {self.end_time - self.start_time:.2f}s")
            logger.info(f"   Statistical Significance: {results.get('statistical_significance', 'N/A')}")
            
            return results
            
        except Exception as e:
            self.is_running = False
            logger.error(f"❌ Experiment failed: {self.name} - {str(e)}")
            raise
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results"""
        results = {
            "experiment_name": self.name,
            "hypothesis_id": self.hypothesis.id,
            "duration": self.end_time - self.start_time if self.end_time and self.start_time else 0,
            "baseline_metrics": {},
            "treatment_metrics": {},
            "statistical_tests": {},
            "hypothesis_validation": {}
        }
        
        # Compile metrics
        for metric_name, baseline_metric in self.baseline_metrics.items():
            treatment_metric = self.treatment_metrics.get(metric_name)
            
            if treatment_metric and baseline_metric.sample_size > 0 and treatment_metric.sample_size > 0:
                results["baseline_metrics"][metric_name] = {
                    "mean": baseline_metric.mean,
                    "std_dev": baseline_metric.std_dev,
                    "sample_size": baseline_metric.sample_size,
                    "confidence_interval": baseline_metric.confidence_interval
                }
                
                results["treatment_metrics"][metric_name] = {
                    "mean": treatment_metric.mean,
                    "std_dev": treatment_metric.std_dev,
                    "sample_size": treatment_metric.sample_size,
                    "confidence_interval": treatment_metric.confidence_interval
                }
                
                # Simple statistical test (t-test approximation)
                if baseline_metric.std_dev and treatment_metric.std_dev:
                    effect_size = abs(treatment_metric.mean - baseline_metric.mean)
                    pooled_std = ((baseline_metric.std_dev ** 2 + treatment_metric.std_dev ** 2) / 2) ** 0.5
                    cohen_d = effect_size / pooled_std if pooled_std > 0 else 0
                    
                    # Simple p-value approximation (this is simplified)
                    z_score = effect_size / (pooled_std * ((1/baseline_metric.sample_size + 1/treatment_metric.sample_size) ** 0.5))
                    p_value = 2 * (1 - self._normal_cdf(abs(z_score))) if pooled_std > 0 else 1.0
                    
                    results["statistical_tests"][metric_name] = {
                        "effect_size": effect_size,
                        "cohen_d": cohen_d,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
        
        # Validate hypothesis
        validation_results = self._validate_hypothesis(results)
        results["hypothesis_validation"] = validation_results
        
        return results
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal cumulative distribution function"""
        # Simple approximation using error function
        return 0.5 * (1 + self._erf(x / (2 ** 0.5)))
    
    def _erf(self, x: float) -> float:
        """Approximate error function"""
        # Abramowitz and Stegun approximation
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (2.718281828 ** (-x * x))
        return sign * y
    
    def _validate_hypothesis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hypothesis against success criteria"""
        validation = {
            "criteria_met": 0,
            "total_criteria": len(self.hypothesis.success_criteria),
            "overall_validation": False,
            "criteria_details": {}
        }
        
        for criteria in self.hypothesis.success_criteria:
            metric_name = criteria.metric_name
            treatment_result = results["treatment_metrics"].get(metric_name)
            
            if treatment_result:
                actual_value = treatment_result["mean"]
                target_value = criteria.target_value
                operator = criteria.comparison_operator
                
                # Evaluate criteria
                criteria_met = False
                if operator == ">=":
                    criteria_met = actual_value >= target_value
                elif operator == "<=":
                    criteria_met = actual_value <= target_value
                elif operator == ">":
                    criteria_met = actual_value > target_value
                elif operator == "<":
                    criteria_met = actual_value < target_value
                elif operator == "==":
                    criteria_met = abs(actual_value - target_value) < 0.01  # Small tolerance
                
                if criteria_met:
                    validation["criteria_met"] += 1
                
                validation["criteria_details"][metric_name] = {
                    "criteria_met": criteria_met,
                    "actual_value": actual_value,
                    "target_value": target_value,
                    "operator": operator,
                    "statistical_significance": results["statistical_tests"].get(metric_name, {}).get("significant", False)
                }
        
        # Overall validation requires all criteria to be met
        validation["overall_validation"] = validation["criteria_met"] == validation["total_criteria"]
        
        return validation


class PerformanceHypothesisExperiment(ExperimentFramework):
    """Performance improvement hypothesis experiment"""
    
    def __init__(self, hypothesis: Hypothesis):
        super().__init__("Performance Improvement", hypothesis)
        self.experiment_type = ExperimentType.PERFORMANCE
        self.baseline_function: Optional[Callable] = None
        self.treatment_function: Optional[Callable] = None
    
    def set_implementations(self, baseline: Callable, treatment: Callable):
        """Set baseline and treatment implementations"""
        self.baseline_function = baseline
        self.treatment_function = treatment
    
    async def setup_experiment(self) -> Dict[str, Any]:
        """Setup performance experiment"""
        return {
            "experiment_type": "performance",
            "baseline_ready": self.baseline_function is not None,
            "treatment_ready": self.treatment_function is not None
        }
    
    async def run_baseline(self) -> Dict[str, Any]:
        """Run baseline implementation"""
        if not self.baseline_function:
            raise ValueError("Baseline function not set")
        
        start_time = time.time()
        result = await self.baseline_function() if asyncio.iscoroutinefunction(self.baseline_function) else self.baseline_function()
        execution_time = time.time() - start_time
        
        return {
            "execution_time": execution_time,
            "result": result
        }
    
    async def run_treatment(self) -> Dict[str, Any]:
        """Run treatment implementation"""
        if not self.treatment_function:
            raise ValueError("Treatment function not set")
        
        start_time = time.time()
        result = await self.treatment_function() if asyncio.iscoroutinefunction(self.treatment_function) else self.treatment_function()
        execution_time = time.time() - start_time
        
        return {
            "execution_time": execution_time,
            "result": result
        }
    
    async def collect_metrics(self, variant: str) -> Dict[str, float]:
        """Collect performance metrics"""
        if variant == "baseline":
            result = await self.run_baseline()
        else:
            result = await self.run_treatment()
        
        return {
            "response_time": result["execution_time"] * 1000,  # Convert to milliseconds
            "memory_usage": random.uniform(10, 100),  # Simulated memory usage
            "cpu_usage": random.uniform(5, 50)  # Simulated CPU usage
        }


class AlgorithmComparisonExperiment(ExperimentFramework):
    """Algorithm comparison experiment"""
    
    def __init__(self, hypothesis: Hypothesis):
        super().__init__("Algorithm Comparison", hypothesis)
        self.experiment_type = ExperimentType.ALGORITHM_COMPARISON
        self.algorithms: Dict[str, Callable] = {}
        self.test_data: List[Any] = []
    
    def add_algorithm(self, name: str, implementation: Callable):
        """Add algorithm implementation"""
        self.algorithms[name] = implementation
    
    def set_test_data(self, data: List[Any]):
        """Set test data for algorithms"""
        self.test_data = data
    
    async def setup_experiment(self) -> Dict[str, Any]:
        """Setup algorithm comparison"""
        return {
            "experiment_type": "algorithm_comparison",
            "algorithms_count": len(self.algorithms),
            "test_data_size": len(self.test_data)
        }
    
    async def run_baseline(self) -> Dict[str, Any]:
        """Run baseline algorithm"""
        baseline_name = "baseline"
        if baseline_name in self.algorithms:
            algorithm = self.algorithms[baseline_name]
            return await self._run_algorithm(algorithm)
        else:
            # Use first algorithm as baseline
            first_algo = next(iter(self.algorithms.values()))
            return await self._run_algorithm(first_algo)
    
    async def run_treatment(self) -> Dict[str, Any]:
        """Run treatment algorithm"""
        treatment_name = "treatment"
        if treatment_name in self.algorithms:
            algorithm = self.algorithms[treatment_name]
            return await self._run_algorithm(algorithm)
        else:
            # Use second algorithm as treatment
            algos = list(self.algorithms.values())
            if len(algos) > 1:
                return await self._run_algorithm(algos[1])
            else:
                return await self._run_algorithm(algos[0])
    
    async def _run_algorithm(self, algorithm: Callable) -> Dict[str, Any]:
        """Run algorithm with test data"""
        start_time = time.time()
        
        results = []
        for data_point in self.test_data[:10]:  # Limit to first 10 for performance
            if asyncio.iscoroutinefunction(algorithm):
                result = await algorithm(data_point)
            else:
                result = algorithm(data_point)
            results.append(result)
        
        execution_time = time.time() - start_time
        
        return {
            "execution_time": execution_time,
            "results": results,
            "accuracy": random.uniform(0.7, 0.95)  # Simulated accuracy
        }
    
    async def collect_metrics(self, variant: str) -> Dict[str, float]:
        """Collect algorithm metrics"""
        if variant == "baseline":
            result = await self.run_baseline()
        else:
            result = await self.run_treatment()
        
        return {
            "execution_time": result["execution_time"] * 1000,
            "accuracy": result["accuracy"] * 100,
            "throughput": len(result["results"]) / result["execution_time"]
        }


class HypothesisDrivenDevelopment:
    """
    Hypothesis-Driven Development Framework
    Manages hypothesis formulation, testing, and validation
    """
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.experiments: Dict[str, ExperimentFramework] = {}
        self.results_history: List[Dict[str, Any]] = []
    
    def formulate_hypothesis(
        self,
        hypothesis_id: str,
        name: str,
        description: str,
        prediction: str,
        success_criteria: List[SuccessCriteria]
    ) -> Hypothesis:
        """Formulate a new hypothesis"""
        hypothesis = Hypothesis(
            id=hypothesis_id,
            name=name,
            description=description,
            prediction=prediction,
            success_criteria=success_criteria
        )
        
        self.hypotheses[hypothesis_id] = hypothesis
        logger.info(f"📋 Hypothesis formulated: {name}")
        
        return hypothesis
    
    def create_performance_experiment(self, hypothesis_id: str, baseline: Callable, treatment: Callable) -> PerformanceHypothesisExperiment:
        """Create performance experiment"""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis not found: {hypothesis_id}")
        
        experiment = PerformanceHypothesisExperiment(self.hypotheses[hypothesis_id])
        experiment.set_implementations(baseline, treatment)
        self.experiments[hypothesis_id] = experiment
        
        return experiment
    
    def create_algorithm_experiment(self, hypothesis_id: str) -> AlgorithmComparisonExperiment:
        """Create algorithm comparison experiment"""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis not found: {hypothesis_id}")
        
        experiment = AlgorithmComparisonExperiment(self.hypotheses[hypothesis_id])
        self.experiments[hypothesis_id] = experiment
        
        return experiment
    
    async def test_hypothesis(self, hypothesis_id: str, sample_size: int = 100) -> Dict[str, Any]:
        """Test a specific hypothesis"""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis not found: {hypothesis_id}")
        
        if hypothesis_id not in self.experiments:
            raise ValueError(f"Experiment not found for hypothesis: {hypothesis_id}")
        
        hypothesis = self.hypotheses[hypothesis_id]
        experiment = self.experiments[hypothesis_id]
        
        hypothesis.status = HypothesisStatus.TESTING
        logger.info(f"🧪 Testing hypothesis: {hypothesis.name}")
        
        try:
            results = await experiment.run_experiment(sample_size)
            
            # Update hypothesis based on results
            validation = results["hypothesis_validation"]
            if validation["overall_validation"]:
                hypothesis.status = HypothesisStatus.VALIDATED
                hypothesis.confidence_score = 95.0
            else:
                hypothesis.status = HypothesisStatus.REJECTED
                hypothesis.confidence_score = 20.0
            
            hypothesis.experiment_results.append(results)
            hypothesis.validation_date = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Store results
            self.results_history.append({
                "hypothesis_id": hypothesis_id,
                "timestamp": time.time(),
                "results": results
            })
            
            logger.info(f"✅ Hypothesis testing complete: {hypothesis.name}")
            logger.info(f"   Status: {hypothesis.status.value}")
            logger.info(f"   Confidence: {hypothesis.confidence_score:.1f}%")
            
            return results
            
        except Exception as e:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE
            logger.error(f"❌ Hypothesis testing failed: {hypothesis.name} - {str(e)}")
            raise
    
    async def test_all_hypotheses(self, sample_size: int = 100) -> Dict[str, Any]:
        """Test all formulated hypotheses"""
        logger.info(f"🎯 Testing all hypotheses for {self.project_name}")
        
        all_results = {
            "project_name": self.project_name,
            "total_hypotheses": len(self.hypotheses),
            "hypothesis_results": {},
            "summary": {}
        }
        
        validated_count = 0
        rejected_count = 0
        inconclusive_count = 0
        
        for hypothesis_id in self.hypotheses.keys():
            if hypothesis_id in self.experiments:
                try:
                    results = await self.test_hypothesis(hypothesis_id, sample_size)
                    all_results["hypothesis_results"][hypothesis_id] = results
                    
                    status = self.hypotheses[hypothesis_id].status
                    if status == HypothesisStatus.VALIDATED:
                        validated_count += 1
                    elif status == HypothesisStatus.REJECTED:
                        rejected_count += 1
                    else:
                        inconclusive_count += 1
                        
                except Exception as e:
                    all_results["hypothesis_results"][hypothesis_id] = {"error": str(e)}
                    inconclusive_count += 1
        
        all_results["summary"] = {
            "validated": validated_count,
            "rejected": rejected_count,
            "inconclusive": inconclusive_count,
            "success_rate": f"{(validated_count / len(self.hypotheses) * 100):.1f}%" if self.hypotheses else "0%"
        }
        
        logger.info(f"🎉 Hypothesis testing complete:")
        logger.info(f"   Validated: {validated_count}")
        logger.info(f"   Rejected: {rejected_count}")
        logger.info(f"   Inconclusive: {inconclusive_count}")
        logger.info(f"   Success Rate: {all_results['summary']['success_rate']}")
        
        return all_results
    
    def get_hypothesis_summary(self) -> Dict[str, Any]:
        """Get summary of all hypotheses"""
        summary = {
            "total_hypotheses": len(self.hypotheses),
            "status_breakdown": {},
            "hypotheses": {}
        }
        
        status_counts = {}
        for hypothesis in self.hypotheses.values():
            status = hypothesis.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            summary["hypotheses"][hypothesis.id] = asdict(hypothesis)
        
        summary["status_breakdown"] = status_counts
        return summary
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save results to file"""
        if not filename:
            timestamp = int(time.time())
            filename = f"hypothesis_driven_results_{timestamp}.json"
        
        results = {
            "project_name": self.project_name,
            "hypotheses": self.get_hypothesis_summary(),
            "results_history": self.results_history,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Results saved: {filename}")
        return filename


# Factory functions
def create_hypothesis_driven_development(project_name: str) -> HypothesisDrivenDevelopment:
    """Create hypothesis-driven development framework"""
    return HypothesisDrivenDevelopment(project_name)


async def demonstrate_hypothesis_testing():
    """Demonstrate hypothesis-driven development"""
    hdd = create_hypothesis_driven_development("FinChat-SEC-QA")
    
    # Formulate hypothesis
    criteria = [
        SuccessCriteria("response_time", 100.0, "<="),  # Response time <= 100ms
        SuccessCriteria("accuracy", 90.0, ">=")         # Accuracy >= 90%
    ]
    
    hypothesis = hdd.formulate_hypothesis(
        "perf_001",
        "Caching Improves Performance",
        "Adding intelligent caching will improve response times by at least 50%",
        "Cached responses will be ≤100ms with ≥90% accuracy",
        criteria
    )
    
    # Create experiment with mock implementations
    async def baseline_impl():
        await asyncio.sleep(0.15)  # 150ms response time
        return {"accuracy": 92.0}
    
    async def treatment_impl():
        await asyncio.sleep(0.08)  # 80ms response time with caching
        return {"accuracy": 93.0}
    
    experiment = hdd.create_performance_experiment("perf_001", baseline_impl, treatment_impl)
    
    # Test hypothesis
    results = await hdd.test_hypothesis("perf_001", sample_size=50)
    
    return results


if __name__ == "__main__":
    # Example usage
    async def main():
        results = await demonstrate_hypothesis_testing()
        print(f"🧪 Hypothesis testing results: {results['hypothesis_validation']}")
    
    asyncio.run(main())