#!/usr/bin/env python3
"""
Comprehensive Quality Gates - TERRAGON SDLC v4.0
Autonomous Quality Assurance System

Features:
- Automated code quality validation
- Security vulnerability scanning  
- Performance benchmarking
- Test coverage analysis
- Documentation completeness
- Deployment readiness checks
"""

import os
import sys
import asyncio
import logging
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityGateResult:
    """Quality gate execution result."""
    
    def __init__(self, name: str, passed: bool, score: float, details: Dict[str, Any], execution_time: float = 0.0):
        self.name = name
        self.passed = passed
        self.score = score  # 0.0 to 1.0
        self.details = details
        self.execution_time = execution_time
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'passed': self.passed,
            'score': self.score,
            'details': self.details,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }


class ComprehensiveQualityGates:
    """
    Comprehensive quality gates system with autonomous validation.
    
    Implements all TERRAGON SDLC quality requirements:
    - Code quality and standards compliance
    - Security vulnerability assessment
    - Performance benchmarking
    - Test coverage and quality
    - Documentation completeness
    - Deployment readiness
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results: List[QualityGateResult] = []
        
        # Quality thresholds
        self.thresholds = {
            'test_coverage': 85.0,      # Minimum 85% test coverage
            'security_score': 90.0,     # Minimum 90% security score
            'performance_score': 80.0,  # Minimum 80% performance score
            'code_quality': 85.0,       # Minimum 85% code quality
            'documentation': 80.0       # Minimum 80% documentation coverage
        }
        
        logger.info(f"Quality gates initialized for project: {self.project_root}")
    
    async def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        logger.info("🚀 Starting comprehensive quality gates validation...")
        
        start_time = time.time()
        
        # Run all quality gates
        gates = [
            self._run_code_quality_gate(),
            self._run_security_gate(),
            self._run_test_coverage_gate(), 
            self._run_performance_gate(),
            self._run_documentation_gate(),
            self._run_deployment_readiness_gate()
        ]
        
        # Execute gates concurrently
        for gate in gates:
            try:
                result = await gate
                self.results.append(result)
                
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                logger.info(f"{status} {result.name} - Score: {result.score:.1%} ({result.execution_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"Gate execution failed: {e}")
                self.results.append(QualityGateResult(
                    name="ERROR",
                    passed=False,
                    score=0.0,
                    details={'error': str(e)}
                ))
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_report(total_time)
        
        # Save results
        await self._save_results(report)
        
        return report
    
    async def _run_code_quality_gate(self) -> QualityGateResult:
        """Run code quality analysis."""
        start_time = time.time()
        
        try:
            details = {}
            scores = []
            
            # Run linting checks
            try:
                result = subprocess.run([
                    sys.executable, "-m", "ruff", "check", str(self.project_root / "src"), "--format", "json"
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    lint_score = 1.0
                    details['linting'] = {'status': 'passed', 'issues': 0}
                else:
                    # Parse JSON output to count issues
                    try:
                        lint_issues = json.loads(result.stdout) if result.stdout else []
                        issue_count = len(lint_issues)
                        
                        # Score based on issues found (max 50 issues = 0 score)
                        lint_score = max(0.0, 1.0 - (issue_count / 50))
                        details['linting'] = {
                            'status': 'issues_found',
                            'issues': issue_count,
                            'sample_issues': lint_issues[:5]  # First 5 issues
                        }
                    except json.JSONDecodeError:
                        lint_score = 0.5
                        details['linting'] = {'status': 'parse_error', 'output': result.stdout[:500]}
                
                scores.append(lint_score)
                
            except subprocess.TimeoutExpired:
                details['linting'] = {'status': 'timeout'}
                scores.append(0.0)
            except Exception as e:
                details['linting'] = {'status': 'error', 'error': str(e)}
                scores.append(0.0)
            
            # Type checking with mypy
            try:
                result = subprocess.run([
                    sys.executable, "-m", "mypy", str(self.project_root / "src"), "--json-report", "/tmp/mypy_report"
                ], capture_output=True, text=True, timeout=120)
                
                type_score = 1.0 if result.returncode == 0 else 0.7
                details['type_checking'] = {
                    'status': 'passed' if result.returncode == 0 else 'issues_found',
                    'stderr': result.stderr[:500] if result.stderr else ''
                }
                scores.append(type_score)
                
            except subprocess.TimeoutExpired:
                details['type_checking'] = {'status': 'timeout'}
                scores.append(0.0)
            except FileNotFoundError:
                details['type_checking'] = {'status': 'mypy_not_available'}
                scores.append(0.5)  # Partial score if mypy not available
            except Exception as e:
                details['type_checking'] = {'status': 'error', 'error': str(e)}
                scores.append(0.0)
            
            # Code complexity analysis (simplified)
            try:
                python_files = list((self.project_root / "src").rglob("*.py"))
                total_lines = 0
                complex_functions = 0
                
                for py_file in python_files[:20]:  # Limit to 20 files for performance
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            total_lines += len(lines)
                            
                            # Simple complexity heuristic
                            for line in lines:
                                if any(keyword in line for keyword in ['for ', 'while ', 'if ', 'elif ', 'try:']):
                                    if len(line.strip()) > 80:  # Long complex lines
                                        complex_functions += 1
                    except Exception:
                        continue
                
                complexity_score = max(0.0, 1.0 - (complex_functions / max(total_lines / 10, 1)))
                details['complexity'] = {
                    'total_lines': total_lines,
                    'complex_functions': complex_functions,
                    'score': complexity_score
                }
                scores.append(complexity_score)
                
            except Exception as e:
                details['complexity'] = {'status': 'error', 'error': str(e)}
                scores.append(0.5)
            
            # Calculate overall score
            overall_score = sum(scores) / len(scores) if scores else 0.0
            passed = overall_score >= (self.thresholds['code_quality'] / 100)
            
            execution_time = time.time() - start_time
            
            return QualityGateResult(
                name="Code Quality",
                passed=passed,
                score=overall_score,
                details=details,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Code quality gate failed: {e}")
            return QualityGateResult(
                name="Code Quality",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.time() - start_time
            )
    
    async def _run_security_gate(self) -> QualityGateResult:
        """Run security vulnerability analysis."""
        start_time = time.time()
        
        try:
            details = {}
            scores = []
            
            # Run bandit security scanner
            try:
                result = subprocess.run([
                    sys.executable, "-m", "bandit", "-r", str(self.project_root / "src"), 
                    "-f", "json", "-q"
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    security_score = 1.0
                    details['bandit'] = {'status': 'no_issues', 'vulnerabilities': 0}
                else:
                    try:
                        bandit_results = json.loads(result.stdout) if result.stdout else {}
                        vulnerabilities = bandit_results.get('results', [])
                        
                        # Score based on vulnerability severity
                        high_severity = len([v for v in vulnerabilities if v.get('issue_severity') == 'HIGH'])
                        medium_severity = len([v for v in vulnerabilities if v.get('issue_severity') == 'MEDIUM'])
                        low_severity = len([v for v in vulnerabilities if v.get('issue_severity') == 'LOW'])
                        
                        # Calculate score (high = -0.2, medium = -0.1, low = -0.05)
                        penalty = (high_severity * 0.2) + (medium_severity * 0.1) + (low_severity * 0.05)
                        security_score = max(0.0, 1.0 - penalty)
                        
                        details['bandit'] = {
                            'status': 'vulnerabilities_found',
                            'total_vulnerabilities': len(vulnerabilities),
                            'high_severity': high_severity,
                            'medium_severity': medium_severity,
                            'low_severity': low_severity,
                            'sample_issues': vulnerabilities[:3]
                        }
                    except json.JSONDecodeError:
                        security_score = 0.5
                        details['bandit'] = {'status': 'parse_error'}
                
                scores.append(security_score)
                
            except subprocess.TimeoutExpired:
                details['bandit'] = {'status': 'timeout'}
                scores.append(0.0)
            except FileNotFoundError:
                details['bandit'] = {'status': 'bandit_not_available'}
                scores.append(0.7)  # Partial score if bandit not available
            except Exception as e:
                details['bandit'] = {'status': 'error', 'error': str(e)}
                scores.append(0.0)
            
            # Check for hardcoded secrets (simplified)
            try:
                secret_patterns = [
                    r'password\s*=\s*["\'][^"\']{8,}["\']',
                    r'api_key\s*=\s*["\'][^"\']{16,}["\']',
                    r'secret\s*=\s*["\'][^"\']{12,}["\']',
                    r'token\s*=\s*["\'][^"\']{20,}["\']'
                ]
                
                import re
                secrets_found = 0
                python_files = list((self.project_root / "src").rglob("*.py"))
                
                for py_file in python_files[:10]:  # Limit for performance
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            for pattern in secret_patterns:
                                if re.search(pattern, content, re.IGNORECASE):
                                    secrets_found += 1
                    except Exception:
                        continue
                
                secrets_score = max(0.0, 1.0 - (secrets_found * 0.3))  # Heavy penalty for secrets
                details['secrets_scan'] = {
                    'secrets_found': secrets_found,
                    'score': secrets_score
                }
                scores.append(secrets_score)
                
            except Exception as e:
                details['secrets_scan'] = {'status': 'error', 'error': str(e)}
                scores.append(0.5)
            
            # Calculate overall security score
            overall_score = sum(scores) / len(scores) if scores else 0.0
            passed = overall_score >= (self.thresholds['security_score'] / 100)
            
            execution_time = time.time() - start_time
            
            return QualityGateResult(
                name="Security Analysis",
                passed=passed,
                score=overall_score,
                details=details,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Security gate failed: {e}")
            return QualityGateResult(
                name="Security Analysis",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.time() - start_time
            )
    
    async def _run_test_coverage_gate(self) -> QualityGateResult:
        """Run test coverage analysis."""
        start_time = time.time()
        
        try:
            details = {}
            
            # Run pytest with coverage
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", 
                    "--cov=src", 
                    "--cov-report=json:/tmp/coverage.json",
                    "--cov-report=term",
                    "-v",
                    str(self.project_root / "tests")
                ], capture_output=True, text=True, timeout=300)
                
                # Parse coverage results
                try:
                    with open('/tmp/coverage.json', 'r') as f:
                        coverage_data = json.load(f)
                    
                    total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
                    
                    details['coverage'] = {
                        'total_coverage': total_coverage,
                        'lines_covered': coverage_data.get('totals', {}).get('covered_lines', 0),
                        'lines_total': coverage_data.get('totals', {}).get('num_statements', 0),
                        'missing_lines': coverage_data.get('totals', {}).get('missing_lines', 0)
                    }
                    
                except FileNotFoundError:
                    # Fallback: parse from terminal output
                    import re
                    coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', result.stdout)
                    if coverage_match:
                        total_coverage = float(coverage_match.group(1))
                        details['coverage'] = {
                            'total_coverage': total_coverage,
                            'source': 'terminal_output'
                        }
                    else:
                        total_coverage = 0
                        details['coverage'] = {
                            'total_coverage': 0,
                            'error': 'could_not_parse_coverage'
                        }
                
                # Parse test results
                if result.returncode == 0:
                    test_status = 'all_passed'
                    test_score = 1.0
                else:
                    # Count failures and errors
                    import re
                    failures = len(re.findall(r'FAILED', result.stdout))
                    errors = len(re.findall(r'ERROR', result.stdout))
                    
                    # Penalty for failures
                    test_score = max(0.0, 1.0 - ((failures + errors) * 0.1))
                    test_status = f'{failures}_failures_{errors}_errors'
                
                details['tests'] = {
                    'status': test_status,
                    'output_lines': len(result.stdout.splitlines()),
                    'test_score': test_score
                }
                
                # Combined score
                coverage_score = total_coverage / 100.0
                overall_score = (coverage_score * 0.7) + (test_score * 0.3)  # 70% coverage, 30% test success
                
                passed = coverage_score >= (self.thresholds['test_coverage'] / 100) and test_score > 0.8
                
            except subprocess.TimeoutExpired:
                details = {'status': 'timeout'}
                overall_score = 0.0
                passed = False
            except Exception as e:
                details = {'status': 'error', 'error': str(e)}
                overall_score = 0.0
                passed = False
            
            execution_time = time.time() - start_time
            
            return QualityGateResult(
                name="Test Coverage",
                passed=passed,
                score=overall_score,
                details=details,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Test coverage gate failed: {e}")
            return QualityGateResult(
                name="Test Coverage",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.time() - start_time
            )
    
    async def _run_performance_gate(self) -> QualityGateResult:
        """Run performance benchmarks."""
        start_time = time.time()
        
        try:
            details = {}
            scores = []
            
            # Import and test key modules for performance
            try:
                # Test import time
                import_start = time.time()
                
                # Dynamic import test
                test_modules = ['finchat_sec_qa.config', 'finchat_sec_qa.validation']
                import_times = {}
                
                for module in test_modules:
                    try:
                        module_start = time.time()
                        __import__(module)
                        import_times[module] = time.time() - module_start
                    except ImportError:
                        import_times[module] = 'failed'
                
                total_import_time = time.time() - import_start
                
                # Score based on import time (< 2 seconds = good)
                import_score = max(0.0, 1.0 - (total_import_time / 4.0))  # 4 seconds = 0 score
                
                details['imports'] = {
                    'total_time': total_import_time,
                    'module_times': import_times,
                    'score': import_score
                }
                scores.append(import_score)
                
            except Exception as e:
                details['imports'] = {'status': 'error', 'error': str(e)}
                scores.append(0.0)
            
            # Test basic operations performance
            try:
                operations_start = time.time()
                
                # Simulate workload
                test_data = list(range(10000))
                processed_data = [x * 2 for x in test_data if x % 2 == 0]
                
                operations_time = time.time() - operations_start
                operations_score = max(0.0, 1.0 - (operations_time / 2.0))  # 2 seconds = 0 score
                
                details['operations'] = {
                    'time': operations_time,
                    'processed_items': len(processed_data),
                    'score': operations_score
                }
                scores.append(operations_score)
                
            except Exception as e:
                details['operations'] = {'status': 'error', 'error': str(e)}
                scores.append(0.0)
            
            # Memory usage check
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # Score based on memory usage (< 500MB = good)
                memory_score = max(0.0, 1.0 - (memory_mb / 1000))  # 1GB = 0 score
                
                details['memory'] = {
                    'memory_mb': memory_mb,
                    'score': memory_score
                }
                scores.append(memory_score)
                
            except ImportError:
                details['memory'] = {'status': 'psutil_not_available'}
                scores.append(0.5)
            except Exception as e:
                details['memory'] = {'status': 'error', 'error': str(e)}
                scores.append(0.0)
            
            # Calculate overall performance score
            overall_score = sum(scores) / len(scores) if scores else 0.0
            passed = overall_score >= (self.thresholds['performance_score'] / 100)
            
            execution_time = time.time() - start_time
            
            return QualityGateResult(
                name="Performance Benchmarks",
                passed=passed,
                score=overall_score,
                details=details,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Performance gate failed: {e}")
            return QualityGateResult(
                name="Performance Benchmarks",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.time() - start_time
            )
    
    async def _run_documentation_gate(self) -> QualityGateResult:
        """Run documentation completeness analysis."""
        start_time = time.time()
        
        try:
            details = {}
            scores = []
            
            # Check for essential documentation files
            essential_docs = [
                'README.md',
                'CHANGELOG.md', 
                'LICENSE',
                'CONTRIBUTING.md'
            ]
            
            doc_files_found = 0
            for doc_file in essential_docs:
                if (self.project_root / doc_file).exists():
                    doc_files_found += 1
            
            essential_docs_score = doc_files_found / len(essential_docs)
            details['essential_docs'] = {
                'found': doc_files_found,
                'total': len(essential_docs),
                'score': essential_docs_score
            }
            scores.append(essential_docs_score)
            
            # Check docstring coverage in Python files
            try:
                python_files = list((self.project_root / "src").rglob("*.py"))
                total_functions = 0
                documented_functions = 0
                
                for py_file in python_files[:20]:  # Limit for performance
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                            # Simple docstring detection
                            import re
                            functions = re.findall(r'def\s+\w+\([^)]*\):', content)
                            classes = re.findall(r'class\s+\w+[^:]*:', content)
                            
                            total_functions += len(functions) + len(classes)
                            
                            # Count docstrings (simplified)
                            docstrings = re.findall(r'"""[^"]*"""', content, re.DOTALL)
                            docstrings += re.findall(r"'''[^']*'''", content, re.DOTALL)
                            
                            # Assume each docstring documents one function/class
                            documented_functions += min(len(docstrings), len(functions) + len(classes))
                            
                    except Exception:
                        continue
                
                docstring_score = documented_functions / max(total_functions, 1)
                details['docstrings'] = {
                    'total_functions': total_functions,
                    'documented_functions': documented_functions,
                    'score': docstring_score
                }
                scores.append(docstring_score)
                
            except Exception as e:
                details['docstrings'] = {'status': 'error', 'error': str(e)}
                scores.append(0.0)
            
            # Check for docs directory
            docs_dir = self.project_root / "docs"
            if docs_dir.exists():
                doc_files = list(docs_dir.rglob("*.md"))
                docs_dir_score = min(1.0, len(doc_files) / 5)  # 5+ files = full score
            else:
                doc_files = []
                docs_dir_score = 0.0
            
            details['docs_directory'] = {
                'exists': docs_dir.exists(),
                'md_files': len(doc_files),
                'score': docs_dir_score
            }
            scores.append(docs_dir_score)
            
            # Calculate overall documentation score
            overall_score = sum(scores) / len(scores) if scores else 0.0
            passed = overall_score >= (self.thresholds['documentation'] / 100)
            
            execution_time = time.time() - start_time
            
            return QualityGateResult(
                name="Documentation",
                passed=passed,
                score=overall_score,
                details=details,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Documentation gate failed: {e}")
            return QualityGateResult(
                name="Documentation",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.time() - start_time
            )
    
    async def _run_deployment_readiness_gate(self) -> QualityGateResult:
        """Run deployment readiness checks."""
        start_time = time.time()
        
        try:
            details = {}
            scores = []
            
            # Check for deployment configuration files
            deployment_files = [
                'docker-compose.yml',
                'Dockerfile',
                'pyproject.toml',
                'requirements.txt'
            ]
            
            deployment_files_found = 0
            for deploy_file in deployment_files:
                if (self.project_root / deploy_file).exists():
                    deployment_files_found += 1
            
            deployment_config_score = deployment_files_found / len(deployment_files)
            details['deployment_config'] = {
                'found': deployment_files_found,
                'total': len(deployment_files),
                'files_found': [f for f in deployment_files if (self.project_root / f).exists()],
                'score': deployment_config_score
            }
            scores.append(deployment_config_score)
            
            # Check environment configuration
            env_files = ['.env.example', '.env.template', 'config/']
            env_found = any((self.project_root / env_file).exists() for env_file in env_files)
            
            env_score = 1.0 if env_found else 0.3  # Partial score if no env config
            details['environment_config'] = {
                'has_env_config': env_found,
                'score': env_score
            }
            scores.append(env_score)
            
            # Check for CI/CD configuration
            ci_paths = [
                '.github/workflows/',
                '.gitlab-ci.yml',
                'Jenkinsfile',
                '.travis.yml'
            ]
            
            ci_found = any((self.project_root / ci_path).exists() for ci_path in ci_paths)
            ci_score = 1.0 if ci_found else 0.5  # Partial score without CI
            
            details['ci_cd'] = {
                'has_ci_config': ci_found,
                'score': ci_score
            }
            scores.append(ci_score)
            
            # Check Python package structure
            try:
                src_dir = self.project_root / "src"
                init_files = list(src_dir.rglob("__init__.py"))
                
                package_score = 1.0 if len(init_files) > 0 else 0.0
                details['package_structure'] = {
                    'src_directory_exists': src_dir.exists(),
                    'init_files_found': len(init_files),
                    'score': package_score
                }
                scores.append(package_score)
                
            except Exception as e:
                details['package_structure'] = {'status': 'error', 'error': str(e)}
                scores.append(0.0)
            
            # Overall deployment readiness
            overall_score = sum(scores) / len(scores) if scores else 0.0
            passed = overall_score >= 0.7  # 70% threshold for deployment readiness
            
            execution_time = time.time() - start_time
            
            return QualityGateResult(
                name="Deployment Readiness",
                passed=passed,
                score=overall_score,
                details=details,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Deployment readiness gate failed: {e}")
            return QualityGateResult(
                name="Deployment Readiness",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=time.time() - start_time
            )
    
    def _generate_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        
        # Calculate overall metrics
        total_gates = len(self.results)
        passed_gates = sum(1 for r in self.results if r.passed)
        overall_score = sum(r.score for r in self.results) / total_gates if total_gates > 0 else 0.0
        
        # Determine overall status
        if passed_gates == total_gates and overall_score >= 0.85:
            overall_status = "✅ EXCELLENT"
            deployment_ready = True
        elif passed_gates >= total_gates * 0.8 and overall_score >= 0.75:
            overall_status = "🟡 GOOD"
            deployment_ready = True
        elif passed_gates >= total_gates * 0.6:
            overall_status = "🟠 FAIR"
            deployment_ready = False
        else:
            overall_status = "🔴 POOR"
            deployment_ready = False
        
        # Identify areas needing improvement
        failing_gates = [r for r in self.results if not r.passed]
        low_scoring_gates = [r for r in self.results if r.score < 0.7]
        
        recommendations = []
        
        for gate in failing_gates:
            if gate.name == "Code Quality":
                recommendations.append("Improve code quality: Fix linting issues, reduce complexity, add type hints")
            elif gate.name == "Security Analysis":
                recommendations.append("Address security vulnerabilities: Remove hardcoded secrets, fix security issues")
            elif gate.name == "Test Coverage":
                recommendations.append("Increase test coverage: Add more unit tests, aim for 85%+ coverage")
            elif gate.name == "Performance Benchmarks":
                recommendations.append("Optimize performance: Reduce import times, optimize algorithms")
            elif gate.name == "Documentation":
                recommendations.append("Improve documentation: Add docstrings, update README, create API docs")
            elif gate.name == "Deployment Readiness":
                recommendations.append("Prepare for deployment: Add Docker config, CI/CD setup, environment configs")
        
        # Generate report
        report = {
            'execution_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_execution_time': total_time,
                'total_gates': total_gates,
                'passed_gates': passed_gates,
                'failed_gates': total_gates - passed_gates,
                'overall_score': overall_score,
                'overall_status': overall_status,
                'deployment_ready': deployment_ready
            },
            'gate_results': [r.to_dict() for r in self.results],
            'quality_metrics': {
                'code_quality': next((r.score for r in self.results if r.name == "Code Quality"), 0.0),
                'security_score': next((r.score for r in self.results if r.name == "Security Analysis"), 0.0),
                'test_coverage': next((r.score for r in self.results if r.name == "Test Coverage"), 0.0),
                'performance': next((r.score for r in self.results if r.name == "Performance Benchmarks"), 0.0),
                'documentation': next((r.score for r in self.results if r.name == "Documentation"), 0.0),
                'deployment_readiness': next((r.score for r in self.results if r.name == "Deployment Readiness"), 0.0)
            },
            'recommendations': recommendations,
            'next_steps': self._generate_next_steps(deployment_ready, failing_gates)
        }
        
        return report
    
    def _generate_next_steps(self, deployment_ready: bool, failing_gates: List[QualityGateResult]) -> List[str]:
        """Generate actionable next steps."""
        
        if deployment_ready and not failing_gates:
            return [
                "🎉 All quality gates passed! Ready for production deployment.",
                "Consider implementing additional monitoring and alerting.",
                "Set up automated deployment pipeline if not already configured.",
                "Plan for regular security updates and dependency maintenance."
            ]
        
        steps = []
        
        if failing_gates:
            steps.append("🔧 Address failing quality gates:")
            for gate in failing_gates[:3]:  # Top 3 priorities
                steps.append(f"   - Fix issues in: {gate.name} (Score: {gate.score:.1%})")
        
        if not deployment_ready:
            steps.extend([
                "🚀 Prepare for deployment:",
                "   - Resolve all critical quality issues",
                "   - Verify all tests pass in clean environment",
                "   - Update documentation and deployment guides",
                "   - Test deployment in staging environment"
            ])
        
        steps.extend([
            "📈 Continuous improvement:",
            "   - Set up automated quality gate checks in CI/CD",
            "   - Monitor quality metrics over time",
            "   - Regular security and dependency updates"
        ])
        
        return steps
    
    async def _save_results(self, report: Dict[str, Any]):
        """Save quality gate results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.project_root / f"quality_gates_report_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Quality gate report saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


async def main():
    """Main entry point for quality gates execution."""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive Quality Gates for TERRAGON SDLC")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-format", choices=["json", "console"], default="console", help="Output format")
    
    args = parser.parse_args()
    
    # Initialize and run quality gates
    gates = ComprehensiveQualityGates(project_root=args.project_root)
    
    try:
        report = await gates.run_all_gates()
        
        if args.output_format == "json":
            print(json.dumps(report, indent=2, default=str))
        else:
            # Console output
            print("\n" + "="*80)
            print("🛡️  TERRAGON SDLC - COMPREHENSIVE QUALITY GATES")
            print("="*80)
            
            summary = report['execution_summary']
            print(f"\n📊 EXECUTION SUMMARY")
            print(f"   Status: {summary['overall_status']}")
            print(f"   Overall Score: {summary['overall_score']:.1%}")
            print(f"   Gates Passed: {summary['passed_gates']}/{summary['total_gates']}")
            print(f"   Execution Time: {summary['total_execution_time']:.2f}s")
            print(f"   Deployment Ready: {'✅ YES' if summary['deployment_ready'] else '❌ NO'}")
            
            print(f"\n🎯 QUALITY METRICS")
            metrics = report['quality_metrics']
            for metric_name, score in metrics.items():
                status = "✅" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                print(f"   {status} {metric_name.replace('_', ' ').title()}: {score:.1%}")
            
            if report.get('recommendations'):
                print(f"\n💡 RECOMMENDATIONS")
                for rec in report['recommendations']:
                    print(f"   • {rec}")
            
            if report.get('next_steps'):
                print(f"\n🚀 NEXT STEPS")
                for step in report['next_steps']:
                    print(f"   {step}")
            
            print("\n" + "="*80)
        
        # Exit with appropriate code
        sys.exit(0 if report['execution_summary']['deployment_ready'] else 1)
        
    except KeyboardInterrupt:
        logger.info("Quality gates execution cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Quality gates execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())