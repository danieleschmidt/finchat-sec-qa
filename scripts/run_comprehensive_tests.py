"""
Comprehensive Test Runner and Quality Gates Validation.

This script runs all tests, validates quality gates, and generates
comprehensive test reports for the autonomous SDLC implementation.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_command(command: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out after 5 minutes"
    except Exception as e:
        return 1, "", str(e)


def run_pytest_tests(test_dir: Path, markers: Optional[str] = None) -> Dict[str, Any]:
    """Run pytest tests and return results."""
    print(f"Running tests in {test_dir}...")
    
    command = ["python", "-m", "pytest", str(test_dir), "-v", "--tb=short"]
    
    if markers:
        command.extend(["-m", markers])
    
    # Add coverage reporting
    command.extend([
        "--cov=finchat_sec_qa",
        "--cov-report=term-missing",
        "--cov-report=json",
        "--cov-fail-under=85"
    ])
    
    start_time = time.time()
    exit_code, stdout, stderr = run_command(command)
    duration = time.time() - start_time
    
    return {
        "exit_code": exit_code,
        "duration": duration,
        "stdout": stdout,
        "stderr": stderr,
        "passed": exit_code == 0
    }


def run_linting_checks() -> Dict[str, Any]:
    """Run code linting checks."""
    print("Running linting checks...")
    
    results = {}
    
    # Black formatting check
    print("  Checking code formatting with black...")
    exit_code, stdout, stderr = run_command([
        "python", "-m", "black", "--check", "--diff", "src/", "tests/"
    ])
    results["black"] = {
        "passed": exit_code == 0,
        "output": stdout + stderr
    }
    
    # isort import sorting check
    print("  Checking import sorting with isort...")
    exit_code, stdout, stderr = run_command([
        "python", "-m", "isort", "--check-only", "--diff", "src/", "tests/"
    ])
    results["isort"] = {
        "passed": exit_code == 0,
        "output": stdout + stderr
    }
    
    # Pylint code analysis
    print("  Running pylint analysis...")
    exit_code, stdout, stderr = run_command([
        "python", "-m", "pylint", "src/finchat_sec_qa/", "--score=yes", "--fail-under=8.0"
    ])
    results["pylint"] = {
        "passed": exit_code == 0,
        "output": stdout + stderr
    }
    
    # MyPy type checking
    print("  Running mypy type checking...")
    exit_code, stdout, stderr = run_command([
        "python", "-m", "mypy", "src/finchat_sec_qa/", "--ignore-missing-imports"
    ])
    results["mypy"] = {
        "passed": exit_code == 0,
        "output": stdout + stderr
    }
    
    return results


def run_security_checks() -> Dict[str, Any]:
    """Run security checks."""
    print("Running security checks...")
    
    results = {}
    
    # Bandit security analysis
    print("  Running bandit security analysis...")
    exit_code, stdout, stderr = run_command([
        "python", "-m", "bandit", "-r", "src/", "-f", "json"
    ])
    results["bandit"] = {
        "passed": exit_code == 0,
        "output": stdout + stderr
    }
    
    # Safety dependency vulnerability check
    print("  Checking dependencies for vulnerabilities...")
    exit_code, stdout, stderr = run_command([
        "python", "-m", "safety", "check", "--json"
    ])
    results["safety"] = {
        "passed": exit_code == 0,
        "output": stdout + stderr
    }
    
    return results


def run_performance_tests() -> Dict[str, Any]:
    """Run performance benchmarks."""
    print("Running performance tests...")
    
    # Run performance-marked tests
    result = run_pytest_tests(Path("tests/"), "performance")
    
    return {
        "performance_tests": result
    }


def validate_quality_gates(test_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate quality gates based on test results."""
    print("Validating quality gates...")
    
    gates = {
        "test_coverage": {
            "threshold": 85.0,
            "passed": False,
            "actual": 0.0
        },
        "unit_tests": {
            "threshold": 100.0,
            "passed": False,
            "actual": 0.0
        },
        "integration_tests": {
            "threshold": 100.0,
            "passed": False,
            "actual": 0.0
        },
        "code_quality": {
            "threshold": 8.0,
            "passed": False,
            "actual": 0.0
        },
        "security_issues": {
            "threshold": 0,
            "passed": False,
            "actual": 0
        },
        "performance_benchmarks": {
            "threshold": 100.0,
            "passed": False,
            "actual": 0.0
        }
    }
    
    # Parse coverage data
    try:
        with open("coverage.json") as f:
            coverage_data = json.load(f)
            coverage_percent = coverage_data["totals"]["percent_covered"]
            gates["test_coverage"]["actual"] = coverage_percent
            gates["test_coverage"]["passed"] = coverage_percent >= gates["test_coverage"]["threshold"]
    except (FileNotFoundError, KeyError):
        print("  Warning: Could not parse coverage data")
    
    # Validate unit tests
    if "unit_tests" in test_results:
        gates["unit_tests"]["passed"] = test_results["unit_tests"]["passed"]
        gates["unit_tests"]["actual"] = 100.0 if test_results["unit_tests"]["passed"] else 0.0
    
    # Validate integration tests
    if "integration_tests" in test_results:
        gates["integration_tests"]["passed"] = test_results["integration_tests"]["passed"]
        gates["integration_tests"]["actual"] = 100.0 if test_results["integration_tests"]["passed"] else 0.0
    
    # Validate code quality (from pylint)
    if "linting" in test_results and "pylint" in test_results["linting"]:
        pylint_output = test_results["linting"]["pylint"]["output"]
        # Parse pylint score
        for line in pylint_output.split("\n"):
            if "Your code has been rated at" in line:
                try:
                    score = float(line.split()[6].split("/")[0])
                    gates["code_quality"]["actual"] = score
                    gates["code_quality"]["passed"] = score >= gates["code_quality"]["threshold"]
                except (IndexError, ValueError):
                    pass
    
    # Validate security issues
    if "security" in test_results and "bandit" in test_results["security"]:
        gates["security_issues"]["passed"] = test_results["security"]["bandit"]["passed"]
        # Count issues from bandit output
        bandit_output = test_results["security"]["bandit"]["output"]
        try:
            bandit_data = json.loads(bandit_output)
            issue_count = len(bandit_data.get("results", []))
            gates["security_issues"]["actual"] = issue_count
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Validate performance benchmarks
    if "performance" in test_results:
        gates["performance_benchmarks"]["passed"] = test_results["performance"]["performance_tests"]["passed"]
        gates["performance_benchmarks"]["actual"] = 100.0 if gates["performance_benchmarks"]["passed"] else 0.0
    
    # Overall gate status
    all_passed = all(gate["passed"] for gate in gates.values())
    
    return {
        "overall_passed": all_passed,
        "gates": gates
    }


def generate_test_report(test_results: Dict[str, Any], quality_gates: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive test report."""
    print("Generating test report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "overall_status": "PASSED" if quality_gates["overall_passed"] else "FAILED",
            "total_duration": 0.0,
            "test_suites_run": 0,
            "tests_passed": 0,
            "tests_failed": 0
        },
        "test_results": test_results,
        "quality_gates": quality_gates,
        "recommendations": []
    }
    
    # Calculate summary statistics
    total_duration = 0.0
    suites_run = 0
    
    for suite_name, suite_results in test_results.items():
        if isinstance(suite_results, dict) and "duration" in suite_results:
            total_duration += suite_results["duration"]
            suites_run += 1
    
    report["summary"]["total_duration"] = total_duration
    report["summary"]["test_suites_run"] = suites_run
    
    # Generate recommendations
    recommendations = []
    
    if not quality_gates["gates"]["test_coverage"]["passed"]:
        recommendations.append(
            f"Increase test coverage from {quality_gates['gates']['test_coverage']['actual']:.1f}% "
            f"to at least {quality_gates['gates']['test_coverage']['threshold']:.1f}%"
        )
    
    if not quality_gates["gates"]["code_quality"]["passed"]:
        recommendations.append(
            f"Improve code quality score from {quality_gates['gates']['code_quality']['actual']:.1f} "
            f"to at least {quality_gates['gates']['code_quality']['threshold']:.1f}"
        )
    
    if quality_gates["gates"]["security_issues"]["actual"] > 0:
        recommendations.append(
            f"Fix {quality_gates['gates']['security_issues']['actual']} security issues identified by bandit"
        )
    
    if not test_results.get("linting", {}).get("black", {}).get("passed", True):
        recommendations.append("Run 'black src/ tests/' to fix code formatting issues")
    
    if not test_results.get("linting", {}).get("isort", {}).get("passed", True):
        recommendations.append("Run 'isort src/ tests/' to fix import sorting issues")
    
    report["recommendations"] = recommendations
    
    return report


def save_report(report: Dict[str, Any], output_file: Path) -> None:
    """Save test report to file."""
    print(f"Saving report to {output_file}...")
    
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to {output_file}")


def print_report_summary(report: Dict[str, Any]) -> None:
    """Print a summary of the test report."""
    print("\n" + "="*60)
    print("TEST REPORT SUMMARY")
    print("="*60)
    
    summary = report["summary"]
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Total Duration: {summary['total_duration']:.2f} seconds")
    print(f"Test Suites Run: {summary['test_suites_run']}")
    
    print("\nQuality Gates:")
    for gate_name, gate_info in report["quality_gates"]["gates"].items():
        status = "✓ PASS" if gate_info["passed"] else "✗ FAIL"
        actual = gate_info["actual"]
        threshold = gate_info["threshold"]
        print(f"  {gate_name}: {status} ({actual} vs {threshold} threshold)")
    
    if report["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    print("="*60)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Comprehensive test runner and quality gates validation")
    parser.add_argument("--output", "-o", type=Path, default=Path("test_report.json"),
                       help="Output file for test report")
    parser.add_argument("--unit-only", action="store_true",
                       help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true",
                       help="Run only integration tests")
    parser.add_argument("--performance-only", action="store_true",
                       help="Run only performance tests")
    parser.add_argument("--skip-linting", action="store_true",
                       help="Skip linting checks")
    parser.add_argument("--skip-security", action="store_true",
                       help="Skip security checks")
    parser.add_argument("--fail-fast", action="store_true",
                       help="Stop on first failure")
    
    args = parser.parse_args()
    
    print("Starting comprehensive test suite...")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    test_results = {}
    
    # Determine which tests to run
    if args.unit_only:
        test_results["unit_tests"] = run_pytest_tests(Path("tests/"), "unit")
    elif args.integration_only:
        test_results["integration_tests"] = run_pytest_tests(Path("tests/"), "integration")
    elif args.performance_only:
        test_results["performance"] = run_performance_tests()
    else:
        # Run all tests
        print("\n1. Running unit tests...")
        test_results["unit_tests"] = run_pytest_tests(Path("tests/"), "unit")
        
        if args.fail_fast and not test_results["unit_tests"]["passed"]:
            print("Unit tests failed. Stopping due to --fail-fast.")
            sys.exit(1)
        
        print("\n2. Running integration tests...")
        test_results["integration_tests"] = run_pytest_tests(Path("tests/"), "integration")
        
        if args.fail_fast and not test_results["integration_tests"]["passed"]:
            print("Integration tests failed. Stopping due to --fail-fast.")
            sys.exit(1)
        
        print("\n3. Running all tests (including new autonomous modules)...")
        test_results["all_tests"] = run_pytest_tests(Path("tests/"))
        
        if args.fail_fast and not test_results["all_tests"]["passed"]:
            print("Tests failed. Stopping due to --fail-fast.")
            sys.exit(1)
    
    # Run linting checks
    if not args.skip_linting and not args.performance_only:
        print("\n4. Running linting checks...")
        test_results["linting"] = run_linting_checks()
        
        if args.fail_fast and not all(result["passed"] for result in test_results["linting"].values()):
            print("Linting checks failed. Stopping due to --fail-fast.")
            sys.exit(1)
    
    # Run security checks
    if not args.skip_security and not args.performance_only:
        print("\n5. Running security checks...")
        test_results["security"] = run_security_checks()
        
        if args.fail_fast and not all(result["passed"] for result in test_results["security"].values()):
            print("Security checks failed. Stopping due to --fail-fast.")
            sys.exit(1)
    
    # Run performance tests
    if not args.unit_only and not args.integration_only:
        print("\n6. Running performance tests...")
        test_results["performance"] = run_performance_tests()
        
        if args.fail_fast and not test_results["performance"]["performance_tests"]["passed"]:
            print("Performance tests failed. Stopping due to --fail-fast.")
            sys.exit(1)
    
    # Validate quality gates
    print("\n7. Validating quality gates...")
    quality_gates = validate_quality_gates(test_results)
    
    # Generate comprehensive report
    print("\n8. Generating test report...")
    report = generate_test_report(test_results, quality_gates)
    
    # Save report
    save_report(report, args.output)
    
    # Print summary
    print_report_summary(report)
    
    # Exit with appropriate code
    if quality_gates["overall_passed"]:
        print("\n✓ All quality gates passed!")
        sys.exit(0)
    else:
        print("\n✗ Some quality gates failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()