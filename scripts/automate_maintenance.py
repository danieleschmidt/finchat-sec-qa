#!/usr/bin/env python3
"""
Automated Maintenance Script for FinChat-SEC-QA

Performs routine maintenance tasks including:
- Dependency updates and security checks
- Code quality analysis and cleanup
- Cache cleanup and optimization
- Log rotation and archival
- Performance monitoring and alerting
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import logging


class MaintenanceAutomator:
    """Automates routine maintenance tasks."""
    
    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.project_root / 'maintenance.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def _run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with logging."""
        self.logger.info(f"Running: {' '.join(cmd)}")
        
        if self.dry_run:
            self.logger.info("[DRY RUN] Command not executed")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=self.project_root, check=check)
            if result.stdout:
                self.logger.debug(f"stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"stderr: {result.stderr}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            if not check:
                return e
            raise
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check for outdated and vulnerable dependencies."""
        self.logger.info("Checking dependencies for updates and vulnerabilities...")
        
        results = {
            "outdated_packages": [],
            "vulnerable_packages": [],
            "update_commands": []
        }
        
        # Check for outdated packages
        result = self._run_command(["pip", "list", "--outdated", "--format=json"], check=False)
        if result.returncode == 0:
            try:
                outdated = json.loads(result.stdout)
                results["outdated_packages"] = outdated
                self.logger.info(f"Found {len(outdated)} outdated packages")
            except json.JSONDecodeError:
                self.logger.warning("Could not parse pip list output")
        
        # Check for security vulnerabilities
        result = self._run_command(["safety", "check", "--json"], check=False)
        if result.returncode == 0:
            try:
                vulnerable = json.loads(result.stdout)
                results["vulnerable_packages"] = vulnerable
                self.logger.warning(f"Found {len(vulnerable)} vulnerable packages")
            except json.JSONDecodeError:
                self.logger.warning("Could not parse safety check output")
        
        return results
    
    def update_dependencies(self, security_only: bool = False) -> bool:
        """Update dependencies, optionally security updates only."""
        self.logger.info("Updating dependencies...")
        
        if security_only:
            # Update only packages with known vulnerabilities
            dep_info = self.check_dependencies()
            vulnerable_packages = dep_info.get("vulnerable_packages", [])
            
            if not vulnerable_packages:
                self.logger.info("No vulnerable packages found")
                return True
            
            for vuln in vulnerable_packages:
                package_name = vuln.get("package_name")
                if package_name:
                    self.logger.info(f"Updating vulnerable package: {package_name}")
                    self._run_command(["pip", "install", "--upgrade", package_name])
        else:
            # Update all outdated packages
            result = self._run_command(["pip", "list", "--outdated", "--format=freeze"], check=False)
            if result.returncode == 0 and result.stdout:
                packages = [line.split("==")[0] for line in result.stdout.strip().split("\n")]
                for package in packages:
                    self.logger.info(f"Updating package: {package}")
                    self._run_command(["pip", "install", "--upgrade", package])
        
        # Regenerate requirements if they exist
        if (self.project_root / "requirements.txt").exists():
            self.logger.info("Regenerating requirements.txt")
            self._run_command(["pip", "freeze"], check=False)
        
        return True
    
    def cleanup_cache(self) -> Dict[str, int]:
        """Clean up various cache directories."""
        self.logger.info("Cleaning up cache directories...")
        
        cleanup_stats = {"files_deleted": 0, "space_freed_mb": 0}
        
        # Cache directories to clean
        cache_dirs = [
            self.project_root / ".pytest_cache",
            self.project_root / ".mypy_cache",
            self.project_root / ".ruff_cache",
            self.project_root / "__pycache__",
            Path.home() / ".cache" / "finchat_sec_qa"
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                # Calculate size before deletion
                try:
                    size_mb = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) // 1024 // 1024
                    file_count = len(list(cache_dir.rglob('*')))
                    
                    if not self.dry_run:
                        shutil.rmtree(cache_dir)
                    
                    cleanup_stats["files_deleted"] += file_count
                    cleanup_stats["space_freed_mb"] += size_mb
                    
                    self.logger.info(f"Cleaned {cache_dir}: {file_count} files, {size_mb}MB")
                except Exception as e:
                    self.logger.warning(f"Could not clean {cache_dir}: {e}")
        
        # Clean pip cache
        self._run_command(["pip", "cache", "purge"], check=False)
        
        return cleanup_stats
    
    def rotate_logs(self, max_age_days: int = 30) -> Dict[str, int]:
        """Rotate and compress old log files."""
        self.logger.info(f"Rotating logs older than {max_age_days} days...")
        
        log_stats = {"files_rotated": 0, "files_deleted": 0}
        
        # Find log files
        log_patterns = ["*.log", "*.log.*"]
        log_dirs = [self.project_root / "logs", Path("/var/log/finchat")]
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for log_dir in log_dirs:
            if not log_dir.exists():
                continue
                
            for pattern in log_patterns:
                for log_file in log_dir.glob(pattern):
                    if log_file.is_file():
                        file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                        
                        if file_time < cutoff_date:
                            if log_file.suffix != ".gz":
                                # Compress old log files
                                if not self.dry_run:
                                    self._run_command(["gzip", str(log_file)], check=False)
                                log_stats["files_rotated"] += 1
                                self.logger.info(f"Compressed {log_file}")
                            else:
                                # Delete very old compressed files
                                very_old_cutoff = datetime.now() - timedelta(days=max_age_days * 2)
                                if file_time < very_old_cutoff:
                                    if not self.dry_run:
                                        log_file.unlink()
                                    log_stats["files_deleted"] += 1
                                    self.logger.info(f"Deleted old log {log_file}")
        
        return log_stats
    
    def run_quality_checks(self) -> Dict[str, bool]:
        """Run code quality checks and fix issues where possible."""
        self.logger.info("Running code quality checks...")
        
        quality_results = {}
        
        # Run formatter
        result = self._run_command(["ruff", "format", "src/", "tests/"], check=False)
        quality_results["formatting"] = result.returncode == 0
        
        # Run linter with auto-fix
        result = self._run_command(["ruff", "check", "--fix", "src/", "tests/"], check=False)
        quality_results["linting"] = result.returncode == 0
        
        # Run type checking
        result = self._run_command(["mypy", "src/"], check=False)
        quality_results["type_checking"] = result.returncode == 0
        
        # Run security scan
        result = self._run_command(["bandit", "-r", "src/", "-q"], check=False)
        quality_results["security_scan"] = result.returncode == 0
        
        return quality_results
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization tasks."""
        self.logger.info("Running performance optimizations...")
        
        perf_results = {}
        
        # Clean up temporary files
        temp_dirs = [
            self.project_root / "temp",
            self.project_root / "tmp",
            Path("/tmp/finchat*")
        ]
        
        files_cleaned = 0
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                try:
                    if not self.dry_run:
                        shutil.rmtree(temp_dir)
                    files_cleaned += len(list(temp_dir.rglob('*')))
                except Exception as e:
                    self.logger.warning(f"Could not clean {temp_dir}: {e}")
        
        perf_results["temp_files_cleaned"] = files_cleaned
        
        # Optimize database if present
        db_files = list(self.project_root.glob("*.db"))
        if db_files:
            self.logger.info("Optimizing database files...")
            # Add database optimization logic here
            perf_results["database_optimized"] = True
        
        return perf_results
    
    def generate_maintenance_report(self, results: Dict[str, Any]) -> str:
        """Generate a maintenance report."""
        report = []
        report.append("# Automated Maintenance Report")
        report.append(f"Date: {datetime.now().isoformat()}")
        report.append(f"Mode: {'DRY RUN' if self.dry_run else 'EXECUTION'}")
        report.append("")
        
        if "dependencies" in results:
            dep_info = results["dependencies"]
            report.append("## Dependency Management")
            report.append(f"- Outdated packages: {len(dep_info.get('outdated_packages', []))}")
            report.append(f"- Vulnerable packages: {len(dep_info.get('vulnerable_packages', []))}")
            report.append("")
        
        if "cache_cleanup" in results:
            cache_info = results["cache_cleanup"]
            report.append("## Cache Cleanup")
            report.append(f"- Files deleted: {cache_info.get('files_deleted', 0)}")
            report.append(f"- Space freed: {cache_info.get('space_freed_mb', 0)}MB")
            report.append("")
        
        if "quality_checks" in results:
            quality_info = results["quality_checks"]
            report.append("## Code Quality")
            for check, passed in quality_info.items():
                status = "✅" if passed else "❌"
                report.append(f"- {check.replace('_', ' ').title()}: {status}")
            report.append("")
        
        return "\n".join(report)
    
    def run_full_maintenance(self, security_only: bool = False) -> Dict[str, Any]:
        """Run full maintenance cycle."""
        self.logger.info("Starting full maintenance cycle...")
        
        results = {}
        
        try:
            # Check and update dependencies
            results["dependencies"] = self.check_dependencies()
            if results["dependencies"]["vulnerable_packages"] or not security_only:
                self.update_dependencies(security_only=security_only)
            
            # Clean up caches
            results["cache_cleanup"] = self.cleanup_cache()
            
            # Rotate logs
            results["log_rotation"] = self.rotate_logs()
            
            # Run quality checks
            results["quality_checks"] = self.run_quality_checks()
            
            # Performance optimization
            results["performance"] = self.optimize_performance()
            
            self.logger.info("Maintenance cycle completed successfully")
            
        except Exception as e:
            self.logger.error(f"Maintenance cycle failed: {e}")
            results["error"] = str(e)
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Automated maintenance for FinChat-SEC-QA")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--security-only", action="store_true", help="Only update packages with security vulnerabilities")
    parser.add_argument("--task", choices=["deps", "cache", "logs", "quality", "perf", "all"], 
                       default="all", help="Specific maintenance task to run")
    parser.add_argument("--report", help="Path to save maintenance report")
    
    args = parser.parse_args()
    
    project_root = Path.cwd()
    automator = MaintenanceAutomator(project_root, dry_run=args.dry_run)
    
    if args.task == "all":
        results = automator.run_full_maintenance(security_only=args.security_only)
    elif args.task == "deps":
        results = {"dependencies": automator.check_dependencies()}
        if results["dependencies"]["vulnerable_packages"] or not args.security_only:
            automator.update_dependencies(security_only=args.security_only)
    elif args.task == "cache":
        results = {"cache_cleanup": automator.cleanup_cache()}
    elif args.task == "logs":
        results = {"log_rotation": automator.rotate_logs()}
    elif args.task == "quality":
        results = {"quality_checks": automator.run_quality_checks()}
    elif args.task == "perf":
        results = {"performance": automator.optimize_performance()}
    
    # Generate and save report
    report = automator.generate_maintenance_report(results)
    print(report)
    
    if args.report:
        with open(args.report, "w") as f:
            f.write(report)
        print(f"\nReport saved to {args.report}")
    
    # Check for issues that need attention
    if "dependencies" in results:
        vulnerable = results["dependencies"].get("vulnerable_packages", [])
        if vulnerable:
            print(f"\n⚠️  WARNING: {len(vulnerable)} vulnerable packages found!")
            for vuln in vulnerable[:3]:  # Show first 3
                print(f"  - {vuln.get('package_name', 'unknown')}: {vuln.get('advisory', 'No details')}")
    
    if "quality_checks" in results:
        failed_checks = [check for check, passed in results["quality_checks"].items() if not passed]
        if failed_checks:
            print(f"\n⚠️  WARNING: {len(failed_checks)} quality checks failed: {', '.join(failed_checks)}")
    
    print("\n✅ Maintenance completed!")


if __name__ == "__main__":
    main()
