#!/usr/bin/env python3
"""
Automated maintenance tasks for FinChat-SEC-QA.

This script performs routine maintenance tasks to keep the project
healthy, secure, and up-to-date.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MaintenanceTask:
    """Base class for maintenance tasks."""
    
    def __init__(self, name: str, description: str, priority: str = "medium"):
        self.name = name
        self.description = description
        self.priority = priority
        self.start_time = None
        self.end_time = None
        self.success = False
        self.error_message = None
    
    def execute(self) -> bool:
        """Execute the maintenance task."""
        self.start_time = time.time()
        logger.info(f"Starting task: {self.name}")
        
        try:
            self.success = self._run()
            if self.success:
                logger.info(f"✅ Task completed: {self.name}")
            else:
                logger.warning(f"⚠️ Task completed with warnings: {self.name}")
        except Exception as e:
            self.success = False
            self.error_message = str(e)
            logger.error(f"❌ Task failed: {self.name} - {e}")
        finally:
            self.end_time = time.time()
        
        return self.success
    
    def _run(self) -> bool:
        """Override this method to implement the task."""
        raise NotImplementedError
    
    def get_duration(self) -> float:
        """Get task execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class DependencyUpdateTask(MaintenanceTask):
    """Update and audit dependencies."""
    
    def __init__(self):
        super().__init__(
            "dependency-update",
            "Update dependencies and check for security vulnerabilities",
            "high"
        )
    
    def _run(self) -> bool:
        success = True
        
        # Check for outdated dependencies
        logger.info("Checking for outdated dependencies...")
        result = subprocess.run([
            "pip", "list", "--outdated", "--format=json"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout:
            outdated = json.loads(result.stdout)
            if outdated:
                logger.info(f"Found {len(outdated)} outdated dependencies")
                for dep in outdated[:5]:  # Show first 5
                    logger.info(f"  {dep['name']}: {dep['version']} -> {dep['latest_version']}")
            else:
                logger.info("All dependencies are up to date")
        
        # Run security audit
        logger.info("Running security audit...")
        result = subprocess.run([
            "safety", "check", "--json"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning("Security vulnerabilities found - see safety output")
            success = False
        else:
            logger.info("No security vulnerabilities found")
        
        return success


class CodeQualityTask(MaintenanceTask):
    """Run code quality checks and fixes."""
    
    def __init__(self):
        super().__init__(
            "code-quality",
            "Run linting, formatting, and type checking",
            "medium"
        )
    
    def _run(self) -> bool:
        success = True
        
        # Format code
        logger.info("Formatting code...")
        result = subprocess.run([
            "ruff", "format", "src/", "tests/"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning("Code formatting issues found")
            success = False
        
        # Run linting
        logger.info("Running linting...")
        result = subprocess.run([
            "ruff", "check", "src/", "tests/", "--fix"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning("Linting issues found")
            # Don't fail on linting issues, just warn
        
        # Type checking
        logger.info("Running type checking...")
        result = subprocess.run([
            "mypy", "src/"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning("Type checking issues found")
            # Don't fail on type issues, just warn
        
        return success


class TestMaintenanceTask(MaintenanceTask):
    """Run tests and update coverage reports."""
    
    def __init__(self):
        super().__init__(
            "test-maintenance",
            "Run tests and update coverage reports",
            "medium"
        )
    
    def _run(self) -> bool:
        # Run tests with coverage
        logger.info("Running test suite with coverage...")
        result = subprocess.run([
            "pytest", "tests/", "-v", "--cov=src", 
            "--cov-report=html", "--cov-report=term"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("All tests passed")
            return True
        else:
            logger.error("Some tests failed")
            return False


class SecurityMaintenanceTask(MaintenanceTask):
    """Run security scans and audits."""
    
    def __init__(self):
        super().__init__(
            "security-maintenance",
            "Run comprehensive security scans",
            "high"
        )
    
    def _run(self) -> bool:
        success = True
        
        # Bandit security scan
        logger.info("Running Bandit security scan...")
        result = subprocess.run([
            "bandit", "-r", "src/", "-f", "json", "-o", "security-report.json"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning("Security issues found - see security-report.json")
            success = False
        
        # Secrets scan
        logger.info("Scanning for secrets...")
        result = subprocess.run([
            "detect-secrets", "scan", "--baseline", ".secrets.baseline"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning("Potential secrets found")
            success = False
        
        return success


class CacheCleanupTask(MaintenanceTask):
    """Clean up various cache directories."""
    
    def __init__(self):
        super().__init__(
            "cache-cleanup",
            "Clean up cache directories and temporary files",
            "low"
        )
    
    def _run(self) -> bool:
        cache_dirs = [
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "__pycache__",
            ".tox",
            "htmlcov"
        ]
        
        cleaned = 0
        for cache_dir in cache_dirs:
            try:
                result = subprocess.run([
                    "find", ".", "-name", cache_dir, "-type", "d", "-exec", "rm", "-rf", "{}", "+"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Could not clean {cache_dir}: {e}")
        
        logger.info(f"Cleaned {cleaned} cache directories")
        return True


class DocumentationTask(MaintenanceTask):
    """Update and validate documentation."""
    
    def __init__(self):
        super().__init__(
            "documentation",
            "Update and validate documentation",
            "medium"
        )
    
    def _run(self) -> bool:
        # Check for broken links in markdown files
        logger.info("Checking documentation...")
        
        # Find all markdown files
        result = subprocess.run([
            "find", ".", "-name", "*.md", "-not", "-path", "./.venv/*"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            md_files = result.stdout.strip().split('\n')
            logger.info(f"Found {len(md_files)} markdown files")
        
        # TODO: Add link checking logic
        return True


class MetricsCollectionTask(MaintenanceTask):
    """Collect and update project metrics."""
    
    def __init__(self):
        super().__init__(
            "metrics-collection",
            "Collect and update project metrics",
            "medium"
        )
    
    def _run(self) -> bool:
        # Run metrics collection script
        logger.info("Collecting project metrics...")
        
        metrics_script = Path(__file__).parent / "metrics_automation.py"
        if metrics_script.exists():
            result = subprocess.run([
                sys.executable, str(metrics_script)
            ], capture_output=True, text=True)
            
            return result.returncode == 0
        else:
            logger.warning("Metrics collection script not found")
            return False


class BackupTask(MaintenanceTask):
    """Create backups of important files."""
    
    def __init__(self):
        super().__init__(
            "backup",
            "Create backups of important configuration files",
            "low"
        )
    
    def _run(self) -> bool:
        import shutil
        
        backup_dir = Path("backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        important_files = [
            "pyproject.toml",
            ".env.example",
            "docker-compose.yml",
            ".github/project-metrics.json"
        ]
        
        backed_up = 0
        for file_path in important_files:
            src = Path(file_path)
            if src.exists():
                try:
                    dst = backup_dir / src.name
                    shutil.copy2(src, dst)
                    backed_up += 1
                except Exception as e:
                    logger.warning(f"Could not backup {file_path}: {e}")
        
        logger.info(f"Backed up {backed_up} files to {backup_dir}")
        return True


class PerformanceOptimizationTask(MaintenanceTask):
    """Run performance optimization tasks."""
    
    def __init__(self):
        super().__init__(
            "performance-optimization",
            "Run performance optimization and benchmarks",
            "low"
        )
    
    def _run(self) -> bool:
        # Run performance benchmarks
        logger.info("Running performance benchmarks...")
        
        benchmark_script = Path("scripts/benchmark.py")
        if benchmark_script.exists():
            result = subprocess.run([
                sys.executable, str(benchmark_script)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Performance benchmarks completed")
                return True
            else:
                logger.warning("Performance benchmarks failed")
                return False
        else:
            logger.info("No benchmark script found, skipping")
            return True


class MaintenanceRunner:
    """Main maintenance task runner."""
    
    def __init__(self):
        self.tasks = []
        self.results = {}
    
    def add_task(self, task: MaintenanceTask):
        """Add a task to the runner."""
        self.tasks.append(task)
    
    def run_all(self, task_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all maintenance tasks."""
        start_time = time.time()
        
        # Filter tasks if specified
        tasks_to_run = self.tasks
        if task_filter:
            tasks_to_run = [t for t in self.tasks if t.name in task_filter]
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        tasks_to_run.sort(key=lambda t: priority_order.get(t.priority, 1))
        
        logger.info(f"Running {len(tasks_to_run)} maintenance tasks...")
        
        results = {
            "start_time": start_time,
            "tasks": [],
            "summary": {"total": len(tasks_to_run), "passed": 0, "failed": 0, "warnings": 0}
        }
        
        for task in tasks_to_run:
            success = task.execute()
            
            task_result = {
                "name": task.name,
                "description": task.description,
                "priority": task.priority,
                "success": success,
                "duration": task.get_duration(),
                "error_message": task.error_message
            }
            
            results["tasks"].append(task_result)
            
            if success:
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1
        
        results["end_time"] = time.time()
        results["total_duration"] = results["end_time"] - results["start_time"]
        
        self.results = results
        return results
    
    def generate_report(self) -> str:
        """Generate a maintenance report."""
        if not self.results:
            return "No maintenance tasks have been run."
        
        report = []
        report.append("# Maintenance Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Summary
        summary = self.results["summary"]
        report.append("## Summary")
        report.append(f"- Total tasks: {summary['total']}")
        report.append(f"- Passed: {summary['passed']}")
        report.append(f"- Failed: {summary['failed']}")
        report.append(f"- Total duration: {self.results['total_duration']:.2f} seconds")
        report.append("")
        
        # Task details
        report.append("## Task Details")
        for task in self.results["tasks"]:
            status = "✅" if task["success"] else "❌"
            report.append(f"### {status} {task['name']} ({task['priority']} priority)")
            report.append(f"Description: {task['description']}")
            report.append(f"Duration: {task['duration']:.2f} seconds")
            if task["error_message"]:
                report.append(f"Error: {task['error_message']}")
            report.append("")
        
        return "\n".join(report)
    
    def save_report(self, output_file: Path = None):
        """Save maintenance report to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"maintenance_report_{timestamp}.md")
        
        report = self.generate_report()
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Maintenance report saved to {output_file}")


def create_default_tasks() -> List[MaintenanceTask]:
    """Create the default set of maintenance tasks."""
    return [
        DependencyUpdateTask(),
        SecurityMaintenanceTask(),
        CodeQualityTask(),
        TestMaintenanceTask(),
        CacheCleanupTask(),
        DocumentationTask(),
        MetricsCollectionTask(),
        BackupTask(),
        PerformanceOptimizationTask()
    ]


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run automated maintenance tasks")
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific tasks to run (default: all)"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Output file for maintenance report"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    # Create maintenance runner
    runner = MaintenanceRunner()
    
    # Add default tasks
    for task in create_default_tasks():
        runner.add_task(task)
    
    # List tasks if requested
    if args.list_tasks:
        print("Available maintenance tasks:")
        for task in runner.tasks:
            print(f"  {task.name} ({task.priority}) - {task.description}")
        return 0
    
    # Dry run
    if args.dry_run:
        tasks_to_run = runner.tasks
        if args.tasks:
            tasks_to_run = [t for t in runner.tasks if t.name in args.tasks]
        
        print("Would run the following tasks:")
        for task in tasks_to_run:
            print(f"  {task.name} ({task.priority}) - {task.description}")
        return 0
    
    # Run maintenance tasks
    try:
        logger.info("🔧 Starting automated maintenance...")
        
        results = runner.run_all(args.tasks)
        
        # Generate and display report
        report = runner.generate_report()
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        # Save report if requested
        if args.report:
            runner.save_report(Path(args.report))
        
        # Return appropriate exit code
        if results["summary"]["failed"] > 0:
            logger.error("Some maintenance tasks failed")
            return 1
        else:
            logger.info("✅ All maintenance tasks completed successfully")
            return 0
    
    except KeyboardInterrupt:
        logger.info("Maintenance interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Maintenance failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())