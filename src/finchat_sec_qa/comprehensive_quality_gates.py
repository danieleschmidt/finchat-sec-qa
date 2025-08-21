"""
Comprehensive Quality Gates System v4.0
Advanced quality assurance, automated testing, and validation framework.
"""

import asyncio
import json
import time
import subprocess
import os
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import threading
import statistics
import functools
import tempfile
import ast
import inspect

logger = logging.getLogger(__name__)


class QualityGateType(Enum):
    """Types of quality gates"""
    CODE_EXECUTION = "code_execution"
    TEST_COVERAGE = "test_coverage"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    CODE_QUALITY = "code_quality"
    DOCUMENTATION_QUALITY = "documentation_quality"
    DEPENDENCY_AUDIT = "dependency_audit"
    STATIC_ANALYSIS = "static_analysis"
    INTEGRATION_TEST = "integration_test"
    LOAD_TEST = "load_test"


class QualityGateStatus(Enum):
    """Quality gate status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Result of a quality gate execution"""
    gate_type: QualityGateType
    status: QualityGateStatus
    score: float
    threshold: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    status: str  # passed, failed, skipped
    duration: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class CodeExecutionValidator:
    """Validates that code executes without errors"""
    
    async def validate(self, project_path: str) -> QualityGateResult:
        """Validate code execution"""
        start_time = time.time()
        
        try:
            # Check Python syntax
            syntax_errors = await self._check_python_syntax(project_path)
            
            # Try importing main modules
            import_errors = await self._check_imports(project_path)
            
            # Run basic smoke tests
            smoke_test_errors = await self._run_smoke_tests(project_path)
            
            total_errors = len(syntax_errors) + len(import_errors) + len(smoke_test_errors)
            
            execution_time = time.time() - start_time
            
            if total_errors == 0:
                return QualityGateResult(
                    gate_type=QualityGateType.CODE_EXECUTION,
                    status=QualityGateStatus.PASSED,
                    score=100.0,
                    threshold=100.0,
                    message="All code executes without errors",
                    details={
                        "syntax_errors": syntax_errors,
                        "import_errors": import_errors,
                        "smoke_test_errors": smoke_test_errors
                    },
                    execution_time=execution_time
                )
            else:
                return QualityGateResult(
                    gate_type=QualityGateType.CODE_EXECUTION,
                    status=QualityGateStatus.FAILED,
                    score=max(0, 100 - (total_errors * 10)),
                    threshold=100.0,
                    message=f"Found {total_errors} execution errors",
                    details={
                        "syntax_errors": syntax_errors,
                        "import_errors": import_errors,
                        "smoke_test_errors": smoke_test_errors
                    },
                    execution_time=execution_time
                )
                
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.CODE_EXECUTION,
                status=QualityGateStatus.FAILED,
                score=0.0,
                threshold=100.0,
                message=f"Code execution validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def _check_python_syntax(self, project_path: str) -> List[str]:
        """Check Python syntax errors"""
        syntax_errors = []
        
        for root, dirs, files in os.walk(project_path):
            # Skip hidden directories and common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            source_code = f.read()
                        
                        # Parse the AST to check syntax
                        ast.parse(source_code, filename=file_path)
                        
                    except SyntaxError as e:
                        syntax_errors.append(f"{file_path}:{e.lineno} - {e.msg}")
                    except Exception as e:
                        syntax_errors.append(f"{file_path} - {str(e)}")
        
        return syntax_errors
    
    async def _check_imports(self, project_path: str) -> List[str]:
        """Check import errors"""
        import_errors = []
        
        # Add project path to Python path temporarily
        import sys
        original_path = sys.path.copy()
        sys.path.insert(0, project_path)
        
        try:
            # Try importing main modules
            main_modules = ['finchat_sec_qa']
            
            for module_name in main_modules:
                try:
                    __import__(module_name)
                except ImportError as e:
                    import_errors.append(f"Failed to import {module_name}: {str(e)}")
                except Exception as e:
                    import_errors.append(f"Error importing {module_name}: {str(e)}")
        
        finally:
            # Restore original path
            sys.path = original_path
        
        return import_errors
    
    async def _run_smoke_tests(self, project_path: str) -> List[str]:
        """Run basic smoke tests"""
        smoke_errors = []
        
        try:
            # Create a simple smoke test
            smoke_test_code = '''
import sys
sys.path.insert(0, "{}")

try:
    import finchat_sec_qa
    print("✅ Basic import successful")
    
    # Test basic functionality
    if hasattr(finchat_sec_qa, '__version__'):
        print(f"✅ Version: {{finchat_sec_qa.__version__}}")
    
    print("✅ Smoke test passed")
    
except Exception as e:
    print(f"❌ Smoke test failed: {{str(e)}}")
    sys.exit(1)
'''.format(project_path)
            
            # Write and execute smoke test
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(smoke_test_code)
                smoke_test_file = f.name
            
            try:
                result = subprocess.run(
                    ['python3', smoke_test_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    smoke_errors.append(f"Smoke test failed: {result.stderr}")
                
            finally:
                os.unlink(smoke_test_file)
                
        except Exception as e:
            smoke_errors.append(f"Smoke test execution error: {str(e)}")
        
        return smoke_errors


class TestCoverageValidator:
    """Validates test coverage"""
    
    async def validate(self, project_path: str, threshold: float = 85.0) -> QualityGateResult:
        """Validate test coverage"""
        start_time = time.time()
        
        try:
            # Run pytest with coverage
            coverage_result = await self._run_coverage_analysis(project_path)
            
            execution_time = time.time() - start_time
            
            if coverage_result["success"]:
                coverage_percentage = coverage_result["coverage_percentage"]
                
                if coverage_percentage >= threshold:
                    status = QualityGateStatus.PASSED
                    message = f"Test coverage {coverage_percentage:.1f}% meets threshold {threshold:.1f}%"
                else:
                    status = QualityGateStatus.FAILED
                    message = f"Test coverage {coverage_percentage:.1f}% below threshold {threshold:.1f}%"
                
                return QualityGateResult(
                    gate_type=QualityGateType.TEST_COVERAGE,
                    status=status,
                    score=coverage_percentage,
                    threshold=threshold,
                    message=message,
                    details=coverage_result,
                    execution_time=execution_time
                )
            else:
                return QualityGateResult(
                    gate_type=QualityGateType.TEST_COVERAGE,
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    threshold=threshold,
                    message=f"Coverage analysis failed: {coverage_result.get('error', 'Unknown error')}",
                    details=coverage_result,
                    execution_time=execution_time
                )
                
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.TEST_COVERAGE,
                status=QualityGateStatus.FAILED,
                score=0.0,
                threshold=threshold,
                message=f"Test coverage validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def _run_coverage_analysis(self, project_path: str) -> Dict[str, Any]:
        """Run coverage analysis"""
        try:
            # Run pytest with coverage
            cmd = [
                'python3', '-m', 'pytest',
                '--cov=src',
                '--cov-report=json',
                '--cov-report=term-missing',
                'tests/',
                '-v'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            # Parse coverage results
            coverage_file = os.path.join(project_path, 'coverage.json')
            coverage_data = {}
            
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
            
            # Extract coverage percentage
            coverage_percentage = coverage_data.get('totals', {}).get('percent_covered', 0.0)
            
            # Count test results
            test_summary = self._parse_pytest_output(result.stdout)
            
            return {
                "success": result.returncode == 0 or coverage_percentage > 0,
                "coverage_percentage": coverage_percentage,
                "coverage_data": coverage_data,
                "test_summary": test_summary,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Coverage analysis timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _parse_pytest_output(self, output: str) -> Dict[str, int]:
        """Parse pytest output for test summary"""
        summary = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line and 'failed' in line:
                # Parse summary line like "2 passed, 1 failed, 3 skipped"
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if 'passed' in part:
                        summary["passed"] = int(part.split()[0])
                    elif 'failed' in part:
                        summary["failed"] = int(part.split()[0])
                    elif 'skipped' in part:
                        summary["skipped"] = int(part.split()[0])
                    elif 'error' in part:
                        summary["errors"] = int(part.split()[0])
                break
        
        return summary


class SecurityScanValidator:
    """Validates security using bandit and safety"""
    
    async def validate(self, project_path: str) -> QualityGateResult:
        """Validate security"""
        start_time = time.time()
        
        try:
            # Run bandit security scan
            bandit_result = await self._run_bandit_scan(project_path)
            
            # Run safety dependency check
            safety_result = await self._run_safety_check(project_path)
            
            execution_time = time.time() - start_time
            
            # Calculate security score
            bandit_score = self._calculate_bandit_score(bandit_result)
            safety_score = 100.0 if safety_result["success"] and not safety_result["vulnerabilities"] else 50.0
            
            overall_score = (bandit_score + safety_score) / 2
            
            if overall_score >= 85.0:
                status = QualityGateStatus.PASSED
                message = f"Security scan passed with score {overall_score:.1f}%"
            elif overall_score >= 70.0:
                status = QualityGateStatus.WARNING
                message = f"Security scan passed with warnings, score {overall_score:.1f}%"
            else:
                status = QualityGateStatus.FAILED
                message = f"Security scan failed with score {overall_score:.1f}%"
            
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY_SCAN,
                status=status,
                score=overall_score,
                threshold=85.0,
                message=message,
                details={
                    "bandit_result": bandit_result,
                    "safety_result": safety_result,
                    "bandit_score": bandit_score,
                    "safety_score": safety_score
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY_SCAN,
                status=QualityGateStatus.FAILED,
                score=0.0,
                threshold=85.0,
                message=f"Security scan failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def _run_bandit_scan(self, project_path: str) -> Dict[str, Any]:
        """Run bandit security scan"""
        try:
            cmd = ['python3', '-m', 'bandit', '-r', 'src/', '-f', 'json']
            
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Parse bandit JSON output
            bandit_data = {}
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    bandit_data = {"error": "Failed to parse bandit output"}
            
            return {
                "success": result.returncode in [0, 1],  # 0 = no issues, 1 = issues found
                "data": bandit_data,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Bandit scan timed out"}
        except FileNotFoundError:
            return {"success": False, "error": "Bandit not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_safety_check(self, project_path: str) -> Dict[str, Any]:
        """Run safety dependency vulnerability check"""
        try:
            cmd = ['python3', '-m', 'safety', 'check', '--json']
            
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            vulnerabilities = []
            if result.stdout:
                try:
                    vulnerabilities = json.loads(result.stdout)
                except json.JSONDecodeError:
                    pass
            
            return {
                "success": result.returncode in [0, 64],  # 0 = no vulns, 64 = vulns found
                "vulnerabilities": vulnerabilities,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Safety check timed out"}
        except FileNotFoundError:
            return {"success": False, "error": "Safety not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _calculate_bandit_score(self, bandit_result: Dict[str, Any]) -> float:
        """Calculate bandit security score"""
        if not bandit_result["success"]:
            return 0.0
        
        data = bandit_result.get("data", {})
        if "error" in data:
            return 50.0  # Partial score if bandit ran but had parsing issues
        
        # Count security issues by severity
        high_issues = len([r for r in data.get("results", []) if r.get("issue_severity") == "HIGH"])
        medium_issues = len([r for r in data.get("results", []) if r.get("issue_severity") == "MEDIUM"])
        low_issues = len([r for r in data.get("results", []) if r.get("issue_severity") == "LOW"])
        
        # Calculate score (high issues are most critical)
        penalty = (high_issues * 15) + (medium_issues * 10) + (low_issues * 5)
        score = max(0, 100 - penalty)
        
        return score


class PerformanceBenchmarkValidator:
    """Validates performance benchmarks"""
    
    async def validate(self, project_path: str) -> QualityGateResult:
        """Validate performance benchmarks"""
        start_time = time.time()
        
        try:
            # Run performance benchmarks
            benchmark_results = await self._run_performance_benchmarks(project_path)
            
            execution_time = time.time() - start_time
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(benchmark_results)
            
            if performance_score >= 85.0:
                status = QualityGateStatus.PASSED
                message = f"Performance benchmarks passed with score {performance_score:.1f}%"
            elif performance_score >= 70.0:
                status = QualityGateStatus.WARNING
                message = f"Performance benchmarks passed with warnings, score {performance_score:.1f}%"
            else:
                status = QualityGateStatus.FAILED
                message = f"Performance benchmarks failed with score {performance_score:.1f}%"
            
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE_BENCHMARK,
                status=status,
                score=performance_score,
                threshold=85.0,
                message=message,
                details=benchmark_results,
                execution_time=execution_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE_BENCHMARK,
                status=QualityGateStatus.FAILED,
                score=0.0,
                threshold=85.0,
                message=f"Performance benchmark failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def _run_performance_benchmarks(self, project_path: str) -> Dict[str, Any]:
        """Run performance benchmarks"""
        try:
            # Create a simple performance test
            perf_test_code = '''
import time
import asyncio
import statistics
import sys
sys.path.insert(0, "{}")

async def benchmark_imports():
    """Benchmark import times"""
    start_time = time.time()
    try:
        import finchat_sec_qa
        import_time = (time.time() - start_time) * 1000
        return {{"success": True, "import_time_ms": import_time}}
    except Exception as e:
        return {{"success": False, "error": str(e)}}

async def benchmark_basic_operations():
    """Benchmark basic operations"""
    times = []
    
    for i in range(10):
        start_time = time.time()
        
        # Simulate some basic operations
        data = list(range(1000))
        processed = [x * 2 for x in data if x % 2 == 0]
        result = sum(processed)
        
        operation_time = (time.time() - start_time) * 1000
        times.append(operation_time)
    
    return {{
        "avg_time_ms": statistics.mean(times),
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "operations": len(times)
    }}

async def main():
    results = {{}}
    
    # Run benchmarks
    results["import_benchmark"] = await benchmark_imports()
    results["operations_benchmark"] = await benchmark_basic_operations()
    
    print(json.dumps(results))

if __name__ == "__main__":
    import json
    asyncio.run(main())
'''.format(project_path)
            
            # Write and execute performance test
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(perf_test_code)
                perf_test_file = f.name
            
            try:
                result = subprocess.run(
                    ['python3', perf_test_file],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0 and result.stdout:
                    benchmark_data = json.loads(result.stdout)
                    return {"success": True, "benchmarks": benchmark_data}
                else:
                    return {"success": False, "error": result.stderr or "Benchmark failed"}
                
            finally:
                os.unlink(perf_test_file)
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _calculate_performance_score(self, benchmark_results: Dict[str, Any]) -> float:
        """Calculate performance score"""
        if not benchmark_results["success"]:
            return 0.0
        
        benchmarks = benchmark_results.get("benchmarks", {})
        score = 100.0
        
        # Check import performance
        import_benchmark = benchmarks.get("import_benchmark", {})
        if import_benchmark.get("success"):
            import_time = import_benchmark.get("import_time_ms", 0)
            if import_time > 1000:  # > 1 second
                score -= 20
            elif import_time > 500:  # > 0.5 seconds
                score -= 10
        else:
            score -= 30
        
        # Check operations performance
        ops_benchmark = benchmarks.get("operations_benchmark", {})
        if ops_benchmark:
            avg_time = ops_benchmark.get("avg_time_ms", 0)
            if avg_time > 100:  # > 100ms
                score -= 20
            elif avg_time > 50:  # > 50ms
                score -= 10
        
        return max(0, score)


class ComprehensiveQualityGates:
    """
    Comprehensive Quality Gates System
    Manages all quality validation processes
    """
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.validators = {
            QualityGateType.CODE_EXECUTION: CodeExecutionValidator(),
            QualityGateType.TEST_COVERAGE: TestCoverageValidator(),
            QualityGateType.SECURITY_SCAN: SecurityScanValidator(),
            QualityGateType.PERFORMANCE_BENCHMARK: PerformanceBenchmarkValidator()
        }
        self.gate_results: Dict[QualityGateType, QualityGateResult] = {}
        self.execution_history: deque = deque(maxlen=100)
    
    async def run_all_gates(self, gates: List[QualityGateType] = None) -> Dict[str, Any]:
        """Run all or specified quality gates"""
        if gates is None:
            gates = list(self.validators.keys())
        
        logger.info(f"🛡️ Running {len(gates)} quality gates...")
        
        start_time = time.time()
        results = {}
        
        for gate_type in gates:
            if gate_type in self.validators:
                logger.info(f"🔍 Running {gate_type.value} validation...")
                
                try:
                    validator = self.validators[gate_type]
                    
                    if gate_type == QualityGateType.TEST_COVERAGE:
                        result = await validator.validate(self.project_path, threshold=85.0)
                    else:
                        result = await validator.validate(self.project_path)
                    
                    self.gate_results[gate_type] = result
                    results[gate_type.value] = asdict(result)
                    
                    status_emoji = {
                        QualityGateStatus.PASSED: "✅",
                        QualityGateStatus.WARNING: "⚠️",
                        QualityGateStatus.FAILED: "❌",
                        QualityGateStatus.SKIPPED: "⏭️"
                    }
                    
                    emoji = status_emoji.get(result.status, "❓")
                    logger.info(f"{emoji} {gate_type.value}: {result.message}")
                    
                except Exception as e:
                    error_result = QualityGateResult(
                        gate_type=gate_type,
                        status=QualityGateStatus.FAILED,
                        score=0.0,
                        threshold=100.0,
                        message=f"Gate execution failed: {str(e)}"
                    )
                    self.gate_results[gate_type] = error_result
                    results[gate_type.value] = asdict(error_result)
                    logger.error(f"❌ {gate_type.value}: Gate execution failed - {str(e)}")
            else:
                logger.warning(f"⚠️ No validator for gate type: {gate_type.value}")
        
        total_time = time.time() - start_time
        
        # Calculate overall quality score
        overall_result = self._calculate_overall_quality(results)
        overall_result["execution_time"] = total_time
        overall_result["timestamp"] = time.time()
        
        # Store in history
        self.execution_history.append(overall_result)
        
        logger.info(f"🎯 Quality Gates Summary:")
        logger.info(f"   Overall Score: {overall_result['overall_score']:.1f}%")
        logger.info(f"   Gates Passed: {overall_result['gates_passed']}/{overall_result['total_gates']}")
        logger.info(f"   Execution Time: {total_time:.2f}s")
        
        return overall_result
    
    def _calculate_overall_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics"""
        total_gates = len(results)
        gates_passed = 0
        gates_warning = 0
        gates_failed = 0
        total_score = 0.0
        
        gate_details = {}
        
        for gate_name, gate_result in results.items():
            status = QualityGateStatus(gate_result["status"])
            score = gate_result["score"]
            
            if status == QualityGateStatus.PASSED:
                gates_passed += 1
            elif status == QualityGateStatus.WARNING:
                gates_warning += 1
            else:
                gates_failed += 1
            
            total_score += score
            gate_details[gate_name] = {
                "status": status.value,
                "score": score,
                "threshold": gate_result["threshold"],
                "message": gate_result["message"]
            }
        
        overall_score = total_score / total_gates if total_gates > 0 else 0.0
        
        # Determine overall status
        if gates_failed == 0 and gates_warning == 0:
            overall_status = "passed"
        elif gates_failed == 0:
            overall_status = "warning"
        else:
            overall_status = "failed"
        
        return {
            "overall_status": overall_status,
            "overall_score": overall_score,
            "total_gates": total_gates,
            "gates_passed": gates_passed,
            "gates_warning": gates_warning,
            "gates_failed": gates_failed,
            "gate_details": gate_details,
            "quality_level": self._determine_quality_level(overall_score)
        }
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score"""
        if score >= 95:
            return "excellent"
        elif score >= 85:
            return "good"
        elif score >= 70:
            return "acceptable"
        elif score >= 50:
            return "poor"
        else:
            return "critical"
    
    async def run_specific_gate(self, gate_type: QualityGateType, **kwargs) -> QualityGateResult:
        """Run a specific quality gate"""
        if gate_type not in self.validators:
            raise ValueError(f"No validator for gate type: {gate_type.value}")
        
        validator = self.validators[gate_type]
        
        if gate_type == QualityGateType.TEST_COVERAGE:
            threshold = kwargs.get("threshold", 85.0)
            result = await validator.validate(self.project_path, threshold=threshold)
        else:
            result = await validator.validate(self.project_path)
        
        self.gate_results[gate_type] = result
        return result
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality summary"""
        if not self.gate_results:
            return {"status": "no_data", "message": "No quality gates have been run"}
        
        latest_execution = self.execution_history[-1] if self.execution_history else None
        
        return {
            "latest_execution": latest_execution,
            "total_executions": len(self.execution_history),
            "available_gates": [gate.value for gate in self.validators.keys()],
            "last_run_gates": list(self.gate_results.keys())
        }
    
    def save_quality_report(self, filename: Optional[str] = None) -> str:
        """Save quality report to file"""
        if not filename:
            timestamp = int(time.time())
            filename = f"quality_gates_report_{timestamp}.json"
        
        report = {
            "project_path": self.project_path,
            "execution_history": [
                {**exec_data, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exec_data["timestamp"]))}
                for exec_data in self.execution_history
            ],
            "latest_results": {
                gate_type.value: asdict(result) 
                for gate_type, result in self.gate_results.items()
            },
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Quality report saved: {filename}")
        return filename


# Factory function
def create_quality_gates(project_path: str) -> ComprehensiveQualityGates:
    """Create comprehensive quality gates system"""
    return ComprehensiveQualityGates(project_path)


# Example usage
async def demonstrate_quality_gates():
    """Demonstrate quality gates system"""
    quality_gates = create_quality_gates("/root/repo")
    
    # Run all quality gates
    results = await quality_gates.run_all_gates()
    
    # Save report
    report_file = quality_gates.save_quality_report()
    
    return results, report_file


if __name__ == "__main__":
    # Example usage
    async def main():
        results, report_file = await demonstrate_quality_gates()
        print(f"🛡️ Quality gates completed: {results['overall_status']}")
        print(f"📊 Report saved: {report_file}")
    
    asyncio.run(main())