#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation Script for FinChat-SEC-QA
Validates all quality gates including security, performance, and reliability.
"""

import ast
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.src_dir = repo_root / "src"
        self.results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'pending',
            'gates': {},
            'summary': {},
            'recommendations': []
        }
    
    def validate_all_gates(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        logger.info("🚀 Starting comprehensive quality gate validation...")
        
        # Core quality gates
        self.validate_code_syntax()
        self.validate_security_standards()
        self.validate_performance_requirements()
        self.validate_reliability_features()
        self.validate_scalability_design()
        self.validate_monitoring_capabilities()
        self.validate_documentation_quality()
        
        # Calculate overall status
        self._calculate_overall_status()
        
        # Generate summary
        self._generate_summary()
        
        logger.info(f"✅ Quality gate validation completed: {self.results['overall_status']}")
        return self.results
    
    def validate_code_syntax(self) -> Dict[str, Any]:
        """Validate code syntax and basic structure."""
        logger.info("🔍 Validating code syntax and structure...")
        
        gate_results = {
            'status': 'passed',
            'checks': [],
            'errors': []
        }
        
        # Check Python syntax for all .py files
        py_files = list(self.src_dir.rglob("*.py"))
        syntax_errors = []
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
                gate_results['checks'].append(f"✅ Syntax valid: {py_file.relative_to(self.repo_root)}")
            except SyntaxError as e:
                syntax_errors.append(f"❌ Syntax error in {py_file}: {e}")
                gate_results['errors'].append(str(e))
        
        # Check for basic code quality indicators
        self._validate_import_structure(gate_results)
        self._validate_function_complexity(gate_results)
        
        if syntax_errors:
            gate_results['status'] = 'failed'
            gate_results['errors'].extend(syntax_errors)
        
        self.results['gates']['code_syntax'] = gate_results
        return gate_results
    
    def validate_security_standards(self) -> Dict[str, Any]:
        """Validate security implementation standards."""
        logger.info("🔒 Validating security standards...")
        
        gate_results = {
            'status': 'passed',
            'checks': [],
            'errors': [],
            'security_features': []
        }
        
        # Check for security features implementation
        security_patterns = {
            'input_validation': [
                'validate_ticker', 'validate_text_safety', 'validate_accession_number'
            ],
            'authentication': [
                'SECRET_TOKEN', 'authentication_attempts_total', 'MIN_TOKEN_LENGTH'
            ],
            'encryption': [
                'cryptography', 'secrets_manager', 'encryption_key'
            ],
            'security_headers': [
                'SecurityHeadersMiddleware', 'X-Content-Type-Options', 'HSTS'
            ],
            'rate_limiting': [
                'rate_limiting', 'RATE_LIMIT_MAX_REQUESTS', 'distributed_rate_limiting'
            ]
        }
        
        for category, patterns in security_patterns.items():
            found_patterns = []
            for pattern in patterns:
                if self._search_pattern_in_codebase(pattern):
                    found_patterns.append(pattern)
            
            if found_patterns:
                gate_results['checks'].append(f"✅ {category}: {len(found_patterns)} implementations found")
                gate_results['security_features'].extend(found_patterns)
            else:
                gate_results['errors'].append(f"❌ No {category} implementations found")
        
        # Check for security vulnerabilities patterns
        vulnerability_patterns = [
            r'eval\s*\(',  # eval usage
            r'exec\s*\(',  # exec usage
            r'__import__\s*\(',  # dynamic imports
            r'pickle\.loads\s*\(',  # unsafe pickle
            r'subprocess\.call.*shell=True',  # shell injection
            r'open\s*\([^r].*["\']w["\']',  # file write without validation
        ]
        
        vulnerabilities_found = []
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern in vulnerability_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            vulnerabilities_found.append(
                                f"Potential vulnerability in {py_file}:{line_num} - {match.group()}"
                            )
            except Exception as e:
                gate_results['errors'].append(f"Error scanning {py_file}: {e}")
        
        if vulnerabilities_found:
            gate_results['status'] = 'failed'
            gate_results['errors'].extend(vulnerabilities_found)
        else:
            gate_results['checks'].append("✅ No obvious security vulnerabilities detected")
        
        # Validate secrets management
        if self._search_pattern_in_codebase('SecretsManager'):
            gate_results['checks'].append("✅ Secrets management system implemented")
        else:
            gate_results['errors'].append("❌ Secrets management system not found")
        
        if gate_results['errors']:
            gate_results['status'] = 'failed'
        
        self.results['gates']['security_standards'] = gate_results
        return gate_results
    
    def validate_performance_requirements(self) -> Dict[str, Any]:
        """Validate performance optimization implementations."""
        logger.info("⚡ Validating performance requirements...")
        
        gate_results = {
            'status': 'passed',
            'checks': [],
            'errors': [],
            'performance_features': []
        }
        
        # Check for performance features
        performance_patterns = {
            'caching': [
                'IntelligentCache', 'CachePolicy', 'cached', 'cache_operations_total'
            ],
            'async_operations': [
                'async def', 'asyncio', 'await', 'AsyncQueryHandler'
            ],
            'connection_pooling': [
                'ConnectionPool', 'httpx.AsyncClient', 'pool_max_connections'
            ],
            'metrics_collection': [
                'prometheus_client', 'Histogram', 'Counter', 'Gauge'
            ],
            'optimization': [
                'ThreadPoolExecutor', 'ProcessPoolExecutor', 'concurrent.futures'
            ]
        }
        
        for category, patterns in performance_patterns.items():
            found_patterns = []
            for pattern in patterns:
                if self._search_pattern_in_codebase(pattern):
                    found_patterns.append(pattern)
            
            if found_patterns:
                gate_results['checks'].append(f"✅ {category}: {len(found_patterns)} implementations")
                gate_results['performance_features'].extend(found_patterns)
            else:
                gate_results['errors'].append(f"⚠️ Limited {category} implementations")
        
        # Check for response time targets
        response_time_patterns = [
            'response_time', 'duration_seconds', 'timeout', 'slow_request_threshold'
        ]
        
        response_time_features = []
        for pattern in response_time_patterns:
            if self._search_pattern_in_codebase(pattern):
                response_time_features.append(pattern)
        
        if response_time_features:
            gate_results['checks'].append(f"✅ Response time monitoring: {len(response_time_features)} features")
        else:
            gate_results['errors'].append("❌ Response time monitoring not implemented")
        
        # Validate auto-scaling capabilities
        if self._search_pattern_in_codebase('AutoScaler') and self._search_pattern_in_codebase('ScalingRule'):
            gate_results['checks'].append("✅ Auto-scaling system implemented")
        else:
            gate_results['errors'].append("❌ Auto-scaling system not found")
        
        if len(gate_results['errors']) > len(gate_results['checks']) / 2:
            gate_results['status'] = 'failed'
        
        self.results['gates']['performance_requirements'] = gate_results
        return gate_results
    
    def validate_reliability_features(self) -> Dict[str, Any]:
        """Validate reliability and fault tolerance features."""
        logger.info("🛡️ Validating reliability features...")
        
        gate_results = {
            'status': 'passed',
            'checks': [],
            'errors': [],
            'reliability_features': []
        }
        
        # Check for reliability patterns
        reliability_patterns = {
            'circuit_breaker': [
                'CircuitBreaker', 'CircuitState', 'failure_threshold'
            ],
            'retry_logic': [
                'retry', 'backoff', 'exponential', 'max_retries'
            ],
            'health_checks': [
                'HealthChecker', 'health_check', 'readiness_probe', 'liveness_probe'
            ],
            'error_handling': [
                'try:', 'except', 'HTTPException', 'logging.error'
            ],
            'graceful_shutdown': [
                'graceful', 'shutdown', 'cleanup', 'finally'
            ],
            'timeout_handling': [
                'timeout', 'asyncio.wait_for', 'TimeoutError'
            ]
        }
        
        for category, patterns in reliability_patterns.items():
            found_patterns = []
            for pattern in patterns:
                if self._search_pattern_in_codebase(pattern):
                    found_patterns.append(pattern)
            
            if found_patterns:
                gate_results['checks'].append(f"✅ {category}: {len(found_patterns)} implementations")
                gate_results['reliability_features'].extend(found_patterns)
            else:
                gate_results['errors'].append(f"❌ {category} not implemented")
        
        # Validate comprehensive error handling
        error_handling_score = 0
        py_files = list(self.src_dir.rglob("*.py"))
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'try:' in content and 'except' in content:
                        error_handling_score += 1
            except Exception:
                continue
        
        error_handling_ratio = error_handling_score / max(len(py_files), 1)
        if error_handling_ratio > 0.7:
            gate_results['checks'].append(f"✅ Error handling coverage: {error_handling_ratio:.1%}")
        else:
            gate_results['errors'].append(f"❌ Low error handling coverage: {error_handling_ratio:.1%}")
        
        if gate_results['errors']:
            gate_results['status'] = 'failed'
        
        self.results['gates']['reliability_features'] = gate_results
        return gate_results
    
    def validate_scalability_design(self) -> Dict[str, Any]:
        """Validate scalability design patterns."""
        logger.info("📈 Validating scalability design...")
        
        gate_results = {
            'status': 'passed',
            'checks': [],
            'errors': [],
            'scalability_features': []
        }
        
        # Check scalability patterns
        scalability_patterns = {
            'horizontal_scaling': [
                'load_balancing', 'worker_processes', 'distributed'
            ],
            'resource_management': [
                'ResourcePool', 'ThreadPoolExecutor', 'ProcessPoolExecutor'
            ],
            'caching_layers': [
                'cache_hit_ratio', 'cache_size', 'eviction'
            ],
            'async_processing': [
                'asyncio', 'concurrent.futures', 'background_tasks'
            ],
            'monitoring': [
                'prometheus', 'metrics', 'observability'
            ]
        }
        
        for category, patterns in scalability_patterns.items():
            found_patterns = []
            for pattern in patterns:
                if self._search_pattern_in_codebase(pattern):
                    found_patterns.append(pattern)
            
            if found_patterns:
                gate_results['checks'].append(f"✅ {category}: {len(found_patterns)} implementations")
                gate_results['scalability_features'].extend(found_patterns)
            else:
                gate_results['errors'].append(f"⚠️ Limited {category} support")
        
        # Check for auto-scaling implementation
        auto_scaling_features = [
            'auto_scaling', 'ScalingRule', 'AdaptiveMetricsCollector'
        ]
        
        auto_scaling_count = sum(1 for pattern in auto_scaling_features 
                               if self._search_pattern_in_codebase(pattern))
        
        if auto_scaling_count >= 2:
            gate_results['checks'].append("✅ Auto-scaling capabilities implemented")
        else:
            gate_results['errors'].append("❌ Auto-scaling capabilities incomplete")
        
        # Validate resource optimization
        optimization_patterns = [
            'memory_limit', 'cpu_limit', 'resource_utilization'
        ]
        
        optimization_count = sum(1 for pattern in optimization_patterns 
                               if self._search_pattern_in_codebase(pattern))
        
        if optimization_count > 0:
            gate_results['checks'].append(f"✅ Resource optimization: {optimization_count} features")
        else:
            gate_results['errors'].append("❌ Resource optimization features missing")
        
        if len(gate_results['errors']) > len(gate_results['checks']):
            gate_results['status'] = 'failed'
        
        self.results['gates']['scalability_design'] = gate_results
        return gate_results
    
    def validate_monitoring_capabilities(self) -> Dict[str, Any]:
        """Validate monitoring and observability capabilities."""
        logger.info("📊 Validating monitoring capabilities...")
        
        gate_results = {
            'status': 'passed',
            'checks': [],
            'errors': [],
            'monitoring_features': []
        }
        
        # Check monitoring patterns
        monitoring_patterns = {
            'metrics_collection': [
                'Counter', 'Histogram', 'Gauge', 'prometheus_client'
            ],
            'health_monitoring': [
                'health_check', 'HealthChecker', 'system_health'
            ],
            'logging': [
                'logging', 'logger', 'log_level'
            ],
            'tracing': [
                'trace', 'span', 'correlation_id'
            ],
            'alerting': [
                'alert', 'threshold', 'notification'
            ]
        }
        
        for category, patterns in monitoring_patterns.items():
            found_patterns = []
            for pattern in patterns:
                if self._search_pattern_in_codebase(pattern):
                    found_patterns.append(pattern)
            
            if found_patterns:
                gate_results['checks'].append(f"✅ {category}: {len(found_patterns)} implementations")
                gate_results['monitoring_features'].extend(found_patterns)
            else:
                gate_results['errors'].append(f"⚠️ {category} implementation incomplete")
        
        # Validate metrics endpoints
        metrics_endpoints = [
            '/metrics', '/health', '/ready', '/live'
        ]
        
        endpoint_count = sum(1 for endpoint in metrics_endpoints 
                           if self._search_pattern_in_codebase(endpoint))
        
        if endpoint_count >= 3:
            gate_results['checks'].append(f"✅ Monitoring endpoints: {endpoint_count}/4 implemented")
        else:
            gate_results['errors'].append(f"❌ Insufficient monitoring endpoints: {endpoint_count}/4")
        
        # Check for business metrics
        business_metrics = [
            'qa_queries_total', 'risk_analyses_total', 'business_value'
        ]
        
        business_metrics_count = sum(1 for metric in business_metrics 
                                   if self._search_pattern_in_codebase(metric))
        
        if business_metrics_count > 0:
            gate_results['checks'].append(f"✅ Business metrics: {business_metrics_count} implemented")
        else:
            gate_results['errors'].append("❌ Business metrics not implemented")
        
        self.results['gates']['monitoring_capabilities'] = gate_results
        return gate_results
    
    def validate_documentation_quality(self) -> Dict[str, Any]:
        """Validate documentation quality and completeness."""
        logger.info("📚 Validating documentation quality...")
        
        gate_results = {
            'status': 'passed',
            'checks': [],
            'errors': [],
            'documentation_score': 0
        }
        
        # Check for required documentation files
        required_docs = [
            'README.md', 'CONTRIBUTING.md', 'CHANGELOG.md', 'LICENSE'
        ]
        
        missing_docs = []
        for doc in required_docs:
            if (self.repo_root / doc).exists():
                gate_results['checks'].append(f"✅ {doc} exists")
            else:
                missing_docs.append(doc)
        
        if missing_docs:
            gate_results['errors'].extend([f"❌ Missing: {doc}" for doc in missing_docs])
        
        # Check documentation in docs/ directory
        docs_dir = self.repo_root / "docs"
        if docs_dir.exists():
            doc_files = list(docs_dir.rglob("*.md"))
            gate_results['checks'].append(f"✅ Documentation files: {len(doc_files)} found")
            gate_results['documentation_score'] += len(doc_files)
        else:
            gate_results['errors'].append("❌ No docs/ directory found")
        
        # Check function docstrings
        docstring_coverage = self._calculate_docstring_coverage()
        if docstring_coverage > 0.7:
            gate_results['checks'].append(f"✅ Docstring coverage: {docstring_coverage:.1%}")
        else:
            gate_results['errors'].append(f"❌ Low docstring coverage: {docstring_coverage:.1%}")
        
        # Check for API documentation
        api_docs = ['API_USAGE_GUIDE.md', 'SDK_USAGE_GUIDE.md']
        api_doc_count = sum(1 for doc in api_docs 
                           if (self.repo_root / "docs" / doc).exists())
        
        if api_doc_count > 0:
            gate_results['checks'].append(f"✅ API documentation: {api_doc_count} guides")
        else:
            gate_results['errors'].append("❌ API documentation missing")
        
        if gate_results['errors']:
            gate_results['status'] = 'failed'
        
        self.results['gates']['documentation_quality'] = gate_results
        return gate_results
    
    def _search_pattern_in_codebase(self, pattern: str) -> bool:
        """Search for a pattern in the codebase."""
        try:
            for py_file in self.src_dir.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    if pattern in f.read():
                        return True
            return False
        except Exception:
            return False
    
    def _validate_import_structure(self, gate_results: Dict[str, Any]):
        """Validate import structure and dependencies."""
        try:
            init_file = self.src_dir / "finchat_sec_qa" / "__init__.py"
            if init_file.exists():
                with open(init_file, 'r') as f:
                    content = f.read()
                    if '__version__' in content and '__all__' in content:
                        gate_results['checks'].append("✅ Proper package initialization")
                    else:
                        gate_results['errors'].append("❌ Incomplete package initialization")
        except Exception as e:
            gate_results['errors'].append(f"❌ Import structure validation failed: {e}")
    
    def _validate_function_complexity(self, gate_results: Dict[str, Any]):
        """Validate function complexity metrics."""
        complex_functions = []
        
        try:
            for py_file in self.src_dir.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                # Simple complexity metric: count nested structures
                                complexity = self._calculate_complexity(node)
                                if complexity > 10:  # Threshold for high complexity
                                    complex_functions.append(
                                        f"{py_file.name}:{node.name} (complexity: {complexity})"
                                    )
                    except SyntaxError:
                        continue
        except Exception as e:
            gate_results['errors'].append(f"❌ Function complexity analysis failed: {e}")
        
        if complex_functions:
            gate_results['errors'].extend([
                f"⚠️ High complexity function: {func}" for func in complex_functions[:5]
            ])
        else:
            gate_results['checks'].append("✅ Function complexity within acceptable limits")
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try,
                                ast.ExceptHandler, ast.With, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_docstring_coverage(self) -> float:
        """Calculate docstring coverage percentage."""
        total_functions = 0
        documented_functions = 0
        
        try:
            for py_file in self.src_dir.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                                total_functions += 1
                                if (ast.get_docstring(node)):
                                    documented_functions += 1
                    except SyntaxError:
                        continue
        except Exception:
            return 0.0
        
        return documented_functions / max(total_functions, 1)
    
    def _calculate_overall_status(self):
        """Calculate overall validation status."""
        gates = self.results['gates']
        total_gates = len(gates)
        passed_gates = sum(1 for gate in gates.values() if gate['status'] == 'passed')
        
        success_rate = passed_gates / max(total_gates, 1)
        
        if success_rate >= 0.9:
            self.results['overall_status'] = 'passed'
        elif success_rate >= 0.7:
            self.results['overall_status'] = 'warning'
        else:
            self.results['overall_status'] = 'failed'
    
    def _generate_summary(self):
        """Generate validation summary and recommendations."""
        gates = self.results['gates']
        
        # Summary statistics
        total_checks = sum(len(gate.get('checks', [])) for gate in gates.values())
        total_errors = sum(len(gate.get('errors', [])) for gate in gates.values())
        
        self.results['summary'] = {
            'total_gates': len(gates),
            'passed_gates': sum(1 for gate in gates.values() if gate['status'] == 'passed'),
            'failed_gates': sum(1 for gate in gates.values() if gate['status'] == 'failed'),
            'total_checks_passed': total_checks,
            'total_issues_found': total_errors,
            'success_rate': sum(1 for gate in gates.values() if gate['status'] == 'passed') / max(len(gates), 1)
        }
        
        # Generate recommendations
        recommendations = []
        
        for gate_name, gate_data in gates.items():
            if gate_data['status'] == 'failed':
                error_count = len(gate_data.get('errors', []))
                recommendations.append(
                    f"🔧 {gate_name}: Address {error_count} critical issues"
                )
        
        # Add specific recommendations based on patterns
        if self.results['summary']['success_rate'] < 0.8:
            recommendations.append("📈 Focus on improving overall code quality and test coverage")
        
        if any('security' in errors for gate in gates.values() 
               for errors in gate.get('errors', [])):
            recommendations.append("🔒 Prioritize security vulnerability fixes")
        
        self.results['recommendations'] = recommendations
    
    def save_results(self, output_file: Path):
        """Save validation results to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"✅ Validation results saved to {output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    def print_report(self):
        """Print comprehensive validation report."""
        print("\n" + "="*80)
        print("🚀 FINCHAT-SEC-QA QUALITY GATES VALIDATION REPORT")
        print("="*80)
        
        status_emoji = {
            'passed': '✅',
            'warning': '⚠️',
            'failed': '❌'
        }
        
        overall_status = self.results['overall_status']
        print(f"\n📊 OVERALL STATUS: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
        
        summary = self.results['summary']
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   • Total Gates: {summary['total_gates']}")
        print(f"   • Passed: {summary['passed_gates']}")
        print(f"   • Failed: {summary['failed_gates']}")
        print(f"   • Success Rate: {summary['success_rate']:.1%}")
        print(f"   • Checks Passed: {summary['total_checks_passed']}")
        print(f"   • Issues Found: {summary['total_issues_found']}")
        
        print(f"\n🔍 GATE-BY-GATE RESULTS:")
        for gate_name, gate_data in self.results['gates'].items():
            status = gate_data['status']
            emoji = status_emoji.get(status, '❓')
            print(f"\n   {emoji} {gate_name.replace('_', ' ').title()}")
            
            # Show checks
            for check in gate_data.get('checks', [])[:3]:  # Show first 3 checks
                print(f"     {check}")
            
            # Show errors
            for error in gate_data.get('errors', [])[:3]:  # Show first 3 errors
                print(f"     {error}")
            
            if len(gate_data.get('errors', [])) > 3:
                remaining = len(gate_data.get('errors', [])) - 3
                print(f"     ... and {remaining} more issues")
        
        if self.results.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in self.results['recommendations']:
                print(f"   • {rec}")
        
        print("\n" + "="*80)
        print(f"🎯 Validation completed at {self.results['timestamp']}")
        print("="*80)


def main():
    """Main validation script execution."""
    repo_root = Path(__file__).parent.parent
    
    # Initialize validator
    validator = QualityGateValidator(repo_root)
    
    # Run all validations
    results = validator.validate_all_gates()
    
    # Save results
    output_file = repo_root / f"quality_validation_report_{int(time.time())}.json"
    validator.save_results(output_file)
    
    # Print report
    validator.print_report()
    
    # Exit with appropriate code
    if results['overall_status'] == 'passed':
        sys.exit(0)
    elif results['overall_status'] == 'warning':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()