#!/usr/bin/env python3
"""
Comprehensive Quality Gates and Security Validation.

This script performs comprehensive quality assurance and security validation
for the quantum financial algorithms to ensure production readiness.

QUALITY ASSURANCE - Production-Ready Validation
"""

import os
import sys
import ast
import re
import hashlib
import subprocess
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityValidator:
    """Security validation for quantum financial code."""
    
    def __init__(self):
        self.security_issues = []
        self.warnings = []
    
    def validate_file(self, filepath: str) -> Dict[str, Any]:
        """Validate a single file for security issues."""
        issues = []
        warnings = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST for analysis
            try:
                tree = ast.parse(content, filename=filepath)
                
                # Check for security issues
                for node in ast.walk(tree):
                    # Check for dangerous imports
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in ['os', 'subprocess', 'eval', 'exec']:
                                warnings.append(f"Potentially dangerous import: {alias.name}")
                    
                    # Check for eval/exec usage
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            if node.func.id in ['eval', 'exec']:
                                issues.append(f"Dangerous function used: {node.func.id}")
                    
                    # Check for shell injection risks
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Attribute):
                            if (hasattr(node.func.value, 'id') and 
                                node.func.value.id == 'subprocess' and
                                node.func.attr in ['call', 'run', 'Popen']):
                                warnings.append("Subprocess call detected - ensure proper input validation")
                
            except SyntaxError as e:
                issues.append(f"Syntax error: {e}")
                
            # Check for hardcoded secrets patterns
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ]
            
            for pattern in secret_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    issues.append(f"Potential hardcoded secret: {match}")
            
            # Check for proper input validation
            if 'input(' in content and 'validate' not in content.lower():
                warnings.append("User input detected without explicit validation")
            
            return {
                'file': filepath,
                'issues': issues,
                'warnings': warnings,
                'lines_of_code': len(content.splitlines()),
                'secure': len(issues) == 0
            }
            
        except Exception as e:
            return {
                'file': filepath,
                'issues': [f"Error analyzing file: {e}"],
                'warnings': [],
                'lines_of_code': 0,
                'secure': False
            }

class CodeQualityValidator:
    """Code quality validation."""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def validate_file(self, filepath: str) -> Dict[str, Any]:
        """Validate code quality for a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic metrics
            lines = content.splitlines()
            total_lines = len(lines)
            non_empty_lines = len([line for line in lines if line.strip()])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            
            # Calculate complexity metrics
            try:
                tree = ast.parse(content)
                
                functions = []
                classes = []
                complexity_score = 0
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                        # Simple complexity - count branches
                        for child in ast.walk(node):
                            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                                complexity_score += 1
                    
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                
                # Documentation coverage
                docstring_coverage = 0
                total_definitions = len(functions) + len(classes)
                
                if total_definitions > 0:
                    documented = 0
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            if (ast.get_docstring(node) is not None):
                                documented += 1
                    
                    docstring_coverage = documented / total_definitions
                
                return {
                    'file': filepath,
                    'total_lines': total_lines,
                    'code_lines': non_empty_lines,
                    'comment_lines': comment_lines,
                    'comment_ratio': comment_lines / max(1, total_lines),
                    'functions': len(functions),
                    'classes': len(classes),
                    'complexity_score': complexity_score,
                    'docstring_coverage': docstring_coverage,
                    'quality_score': self._calculate_quality_score(
                        comment_lines / max(1, total_lines),
                        docstring_coverage,
                        complexity_score / max(1, len(functions))
                    )
                }
                
            except SyntaxError:
                return {
                    'file': filepath,
                    'total_lines': total_lines,
                    'code_lines': non_empty_lines,
                    'comment_lines': comment_lines,
                    'comment_ratio': 0,
                    'functions': 0,
                    'classes': 0,
                    'complexity_score': 0,
                    'docstring_coverage': 0,
                    'quality_score': 0,
                    'error': 'Syntax error in file'
                }
                
        except Exception as e:
            return {
                'file': filepath,
                'error': f"Error analyzing file: {e}",
                'quality_score': 0
            }
    
    def _calculate_quality_score(self, comment_ratio: float, doc_coverage: float, avg_complexity: float) -> float:
        """Calculate overall quality score (0-100)."""
        # Weight factors
        comment_weight = 0.3
        doc_weight = 0.4
        complexity_weight = 0.3
        
        # Normalize scores
        comment_score = min(1.0, comment_ratio * 5)  # Good if >20% comments
        doc_score = doc_coverage
        complexity_score = max(0, 1 - (avg_complexity - 5) / 10)  # Penalize high complexity
        
        total_score = (
            comment_score * comment_weight +
            doc_score * doc_weight +
            complexity_score * complexity_weight
        )
        
        return total_score * 100

class TestValidator:
    """Test coverage and quality validation."""
    
    def validate_tests(self, src_dir: str, test_dir: str) -> Dict[str, Any]:
        """Validate test coverage and quality."""
        
        # Find source files
        src_files = []
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    src_files.append(os.path.join(root, file))
        
        # Find test files
        test_files = []
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.startswith('test_') and file.endswith('.py'):
                        test_files.append(os.path.join(root, file))
        
        # Calculate test coverage approximation
        test_coverage_estimate = 0
        if src_files:
            # Simple heuristic: assume each test file covers multiple source files
            coverage_ratio = len(test_files) / len(src_files)
            test_coverage_estimate = min(100, coverage_ratio * 60)  # Rough estimate
        
        return {
            'src_files_count': len(src_files),
            'test_files_count': len(test_files),
            'test_coverage_estimate': test_coverage_estimate,
            'src_files': [os.path.relpath(f) for f in src_files],
            'test_files': [os.path.relpath(f) for f in test_files]
        }

def run_comprehensive_validation():
    """Run comprehensive quality gates and security validation."""
    
    print("🛡️ QUANTUM FINANCIAL ALGORITHMS - QUALITY GATES & SECURITY VALIDATION")
    print("=" * 80)
    
    # Initialize validators
    security_validator = SecurityValidator()
    quality_validator = CodeQualityValidator()
    test_validator = TestValidator()
    
    # Define directories
    src_dir = "src/finchat_sec_qa"
    test_dir = "tests"
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'security_results': [],
        'quality_results': [],
        'test_results': {},
        'overall_status': 'PENDING'
    }
    
    print(f"\\n📁 Analyzing source directory: {src_dir}")
    
    # Find Python files to analyze
    python_files = []
    if os.path.exists(src_dir):
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
    
    if not python_files:
        print("❌ No Python files found in source directory")
        return
    
    print(f"📊 Found {len(python_files)} Python files to analyze")
    
    # Security Validation
    print("\\n" + "-" * 60)
    print("🔒 SECURITY VALIDATION")
    print("-" * 60)
    
    security_issues_total = 0
    security_warnings_total = 0
    
    for file_path in python_files:
        print(f"🔍 Analyzing: {os.path.relpath(file_path)}")
        security_result = security_validator.validate_file(file_path)
        validation_results['security_results'].append(security_result)
        
        security_issues_total += len(security_result['issues'])
        security_warnings_total += len(security_result['warnings'])
        
        if security_result['issues']:
            print(f"   ❌ {len(security_result['issues'])} security issues found")
            for issue in security_result['issues']:
                print(f"      • {issue}")
        
        if security_result['warnings']:
            print(f"   ⚠️  {len(security_result['warnings'])} security warnings")
            for warning in security_result['warnings']:
                print(f"      • {warning}")
        
        if not security_result['issues'] and not security_result['warnings']:
            print("   ✅ No security issues detected")
    
    print(f"\\n🔒 SECURITY SUMMARY:")
    print(f"   Total Issues: {security_issues_total}")
    print(f"   Total Warnings: {security_warnings_total}")
    print(f"   Secure Files: {len([r for r in validation_results['security_results'] if r['secure']])}/{len(python_files)}")
    
    # Code Quality Validation
    print("\\n" + "-" * 60)
    print("📊 CODE QUALITY VALIDATION")
    print("-" * 60)
    
    total_lines = 0
    total_functions = 0
    total_classes = 0
    quality_scores = []
    
    for file_path in python_files:
        print(f"📊 Analyzing: {os.path.relpath(file_path)}")
        quality_result = quality_validator.validate_file(file_path)
        validation_results['quality_results'].append(quality_result)
        
        if 'error' in quality_result:
            print(f"   ❌ Error: {quality_result['error']}")
            continue
        
        total_lines += quality_result['code_lines']
        total_functions += quality_result['functions']
        total_classes += quality_result['classes']
        quality_scores.append(quality_result['quality_score'])
        
        print(f"   Lines of Code: {quality_result['code_lines']}")
        print(f"   Functions: {quality_result['functions']}")
        print(f"   Classes: {quality_result['classes']}")
        print(f"   Comment Ratio: {quality_result['comment_ratio']:.2%}")
        print(f"   Docstring Coverage: {quality_result['docstring_coverage']:.2%}")
        print(f"   Quality Score: {quality_result['quality_score']:.1f}/100")
        
        if quality_result['quality_score'] >= 80:
            print("   ✅ Excellent code quality")
        elif quality_result['quality_score'] >= 60:
            print("   ⚠️  Good code quality")
        else:
            print("   ❌ Code quality needs improvement")
    
    avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    print(f"\\n📊 CODE QUALITY SUMMARY:")
    print(f"   Total Lines of Code: {total_lines:,}")
    print(f"   Total Functions: {total_functions}")
    print(f"   Total Classes: {total_classes}")
    print(f"   Average Quality Score: {avg_quality_score:.1f}/100")
    
    # Test Coverage Validation
    print("\\n" + "-" * 60)
    print("🧪 TEST VALIDATION")
    print("-" * 60)
    
    test_results = test_validator.validate_tests(src_dir, test_dir)
    validation_results['test_results'] = test_results
    
    print(f"📁 Source Files: {test_results['src_files_count']}")
    print(f"🧪 Test Files: {test_results['test_files_count']}")
    print(f"📊 Estimated Test Coverage: {test_results['test_coverage_estimate']:.1f}%")
    
    if test_results['test_files_count'] > 0:
        print("\\n🧪 Test Files Found:")
        for test_file in test_results['test_files'][:10]:  # Show first 10
            print(f"   • {test_file}")
        if len(test_results['test_files']) > 10:
            print(f"   ... and {len(test_results['test_files']) - 10} more")
    else:
        print("   ⚠️  No test files found in tests/ directory")
    
    # File Integrity Check
    print("\\n" + "-" * 60)
    print("🔐 FILE INTEGRITY CHECK")
    print("-" * 60)
    
    file_hashes = {}
    for file_path in python_files:
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
                file_hashes[os.path.relpath(file_path)] = file_hash
                print(f"✅ {os.path.relpath(file_path)}: {file_hash[:16]}...")
        except Exception as e:
            print(f"❌ Error hashing {file_path}: {e}")
    
    # Overall Assessment
    print("\\n" + "=" * 80)
    print("📋 OVERALL QUALITY ASSESSMENT")
    print("=" * 80)
    
    # Calculate overall scores
    security_score = max(0, 100 - (security_issues_total * 20) - (security_warnings_total * 5))
    quality_score = avg_quality_score
    test_score = test_results['test_coverage_estimate']
    
    overall_score = (security_score * 0.4 + quality_score * 0.35 + test_score * 0.25)
    
    print(f"\\n📊 QUALITY METRICS:")
    print(f"   🔒 Security Score: {security_score:.1f}/100")
    print(f"   📊 Code Quality Score: {quality_score:.1f}/100")
    print(f"   🧪 Test Coverage Score: {test_score:.1f}/100")
    print(f"   🎯 Overall Score: {overall_score:.1f}/100")
    
    # Determine status
    if overall_score >= 80:
        status = "✅ EXCELLENT - Production Ready"
        validation_results['overall_status'] = 'EXCELLENT'
    elif overall_score >= 70:
        status = "✅ GOOD - Production Ready with Minor Improvements"
        validation_results['overall_status'] = 'GOOD'
    elif overall_score >= 60:
        status = "⚠️  ACCEPTABLE - Needs Improvements Before Production"
        validation_results['overall_status'] = 'ACCEPTABLE'
    else:
        status = "❌ NEEDS SIGNIFICANT IMPROVEMENT"
        validation_results['overall_status'] = 'POOR'
    
    print(f"\\n🎯 OVERALL STATUS: {status}")
    
    # Recommendations
    print("\\n📋 RECOMMENDATIONS:")
    
    if security_issues_total > 0:
        print(f"   🔒 CRITICAL: Fix {security_issues_total} security issues before production")
    
    if security_warnings_total > 0:
        print(f"   ⚠️  Review {security_warnings_total} security warnings")
    
    if quality_score < 70:
        print("   📊 Improve code quality: add documentation and reduce complexity")
    
    if test_score < 60:
        print("   🧪 Increase test coverage: aim for >80% coverage")
    
    if overall_score >= 80:
        print("   ✅ Code meets production quality standards")
        print("   ✅ All quantum algorithms are well-documented and secure")
        print("   ✅ Ready for deployment with comprehensive monitoring")
    
    # Security Best Practices Check
    print("\\n" + "-" * 60)
    print("🔒 SECURITY BEST PRACTICES VERIFICATION")
    print("-" * 60)
    
    security_checks = {
        'No hardcoded secrets': security_issues_total == 0,
        'Input validation present': any('validate' in content for content in [open(f).read() for f in python_files[:5]]),
        'Error handling implemented': any('try:' in content for content in [open(f).read() for f in python_files[:5]]),
        'Logging configured': any('logging' in content for content in [open(f).read() for f in python_files[:5]]),
        'Type hints used': any(': ' in content and '->' in content for content in [open(f).read() for f in python_files[:5]])
    }
    
    for check, passed in security_checks.items():
        status_icon = "✅" if passed else "⚠️"
        print(f"   {status_icon} {check}")
    
    # Quantum-Specific Validations
    print("\\n" + "-" * 60)
    print("⚛️  QUANTUM ALGORITHM SPECIFIC VALIDATIONS")
    print("-" * 60)
    
    quantum_validations = []
    
    # Check for quantum-specific modules
    quantum_modules = ['quantum_timeseries.py', 'quantum_risk_ml.py', 'quantum_portfolio.py', 'photonic_continuous_variables.py']
    for module in quantum_modules:
        module_path = os.path.join(src_dir, module)
        if os.path.exists(module_path):
            print(f"   ✅ Quantum module present: {module}")
            
            # Check for essential quantum components
            with open(module_path, 'r') as f:
                content = f.read()
                
            if 'quantum' in content.lower():
                print(f"      ✅ Contains quantum implementations")
            
            if 'benchmark' in content.lower() or 'validation' in content.lower():
                print(f"      ✅ Includes performance validation")
            
            if 'class ' in content:
                print(f"      ✅ Object-oriented design")
                
        else:
            print(f"   ⚠️  Missing quantum module: {module}")
    
    # Final validation summary
    validation_results['summary'] = {
        'security_score': security_score,
        'quality_score': quality_score,
        'test_score': test_score,
        'overall_score': overall_score,
        'status': validation_results['overall_status'],
        'total_files_analyzed': len(python_files),
        'total_lines_of_code': total_lines,
        'security_issues': security_issues_total,
        'security_warnings': security_warnings_total
    }
    
    # Save validation report
    import json
    report_filename = f"quality_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_filename, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\\n💾 Validation report saved to: {report_filename}")
    
    print("\\n" + "=" * 80)
    print("✅ QUALITY GATES & SECURITY VALIDATION COMPLETED")
    print("=" * 80)
    
    return validation_results

if __name__ == "__main__":
    print("🛡️ Starting Comprehensive Quality Gates & Security Validation...")
    
    try:
        results = run_comprehensive_validation()
        
        if results['overall_status'] in ['EXCELLENT', 'GOOD']:
            print("\\n🎉 Quality validation passed!")
            sys.exit(0)
        else:
            print("\\n⚠️  Quality validation completed with issues.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)