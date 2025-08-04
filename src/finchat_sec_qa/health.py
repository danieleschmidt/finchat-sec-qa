"""
Health check endpoints and monitoring utilities for FinChat-SEC-QA.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import time
from typing import Any, Dict, Optional

import httpx
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import psutil

# Metrics
HEALTH_CHECK_COUNTER = Counter('health_checks_total', 'Total health checks', ['endpoint', 'status'])
RESPONSE_TIME_HISTOGRAM = Histogram('health_check_duration_seconds', 'Health check response time')
SYSTEM_MEMORY_GAUGE = Gauge('system_memory_usage_bytes', 'System memory usage')
SYSTEM_CPU_GAUGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
CACHE_SIZE_GAUGE = Gauge('cache_size_items', 'Number of items in cache')


@dataclass
class HealthStatus:
    """Health status data structure."""
    service: str
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    version: str
    uptime_seconds: float
    checks: Dict[str, Any]
    metrics: Dict[str, Any]


class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self, app_start_time: Optional[float] = None):
        self.app_start_time = app_start_time or time.time()
        self.external_services = {
            'openai': 'https://api.openai.com/v1/models',
            'sec_edgar': 'https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json'
        }

    async def get_health_status(self, include_external: bool = True) -> HealthStatus:
        """Get comprehensive health status."""
        start_time = time.time()

        try:
            # Core health checks
            checks = {
                'database': await self._check_database(),
                'cache': await self._check_cache(),
                'filesystem': self._check_filesystem(),
                'memory': self._check_memory(),
                'cpu': self._check_cpu()
            }

            # External service checks (optional)
            if include_external:
                checks['external_services'] = await self._check_external_services()

            # System metrics
            metrics = self._collect_system_metrics()

            # Determine overall status
            overall_status = self._determine_overall_status(checks)

            # Record metrics
            HEALTH_CHECK_COUNTER.labels(endpoint='health', status=overall_status).inc()
            RESPONSE_TIME_HISTOGRAM.observe(time.time() - start_time)

            return HealthStatus(
                service="finchat-sec-qa",
                status=overall_status,
                timestamp=datetime.now(timezone.utc).isoformat(),
                version=self._get_version(),
                uptime_seconds=time.time() - self.app_start_time,
                checks=checks,
                metrics=metrics
            )

        except Exception as e:
            HEALTH_CHECK_COUNTER.labels(endpoint='health', status='error').inc()
            return HealthStatus(
                service="finchat-sec-qa",
                status="unhealthy",
                timestamp=datetime.now(timezone.utc).isoformat(),
                version=self._get_version(),
                uptime_seconds=time.time() - self.app_start_time,
                checks={"error": str(e)},
                metrics={}
            )

    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            # For file-based storage, check if cache directory is accessible
            cache_dir = os.path.expanduser("~/.cache/finchat_sec_qa")

            start_time = time.time()

            # Test write/read operation
            test_file = os.path.join(cache_dir, "health_check.tmp")
            test_data = {"timestamp": time.time()}

            os.makedirs(cache_dir, exist_ok=True)

            with open(test_file, 'w') as f:
                json.dump(test_data, f)

            with open(test_file) as f:
                read_data = json.load(f)

            os.unlink(test_file)

            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "storage_path": cache_dir,
                "read_write_test": "passed"
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def _check_cache(self) -> Dict[str, Any]:
        """Check cache system health."""
        try:
            # Check cache directory size and accessibility
            cache_dir = os.path.expanduser("~/.cache/finchat_sec_qa")

            if not os.path.exists(cache_dir):
                return {
                    "status": "degraded",
                    "message": "Cache directory does not exist"
                }

            # Count cache files
            cache_files = []
            total_size = 0

            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        cache_files.append({
                            "name": file,
                            "size_bytes": file_size,
                            "modified": os.path.getmtime(file_path)
                        })
                        total_size += file_size
                    except OSError:
                        continue

            CACHE_SIZE_GAUGE.set(len(cache_files))

            return {
                "status": "healthy",
                "file_count": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def _check_filesystem(self) -> Dict[str, Any]:
        """Check filesystem health and disk space."""
        try:
            # Check disk usage
            disk_usage = psutil.disk_usage('/')
            free_space_gb = disk_usage.free / (1024**3)
            total_space_gb = disk_usage.total / (1024**3)
            usage_percent = (disk_usage.used / disk_usage.total) * 100

            status = "healthy"
            if usage_percent > 90:
                status = "unhealthy"
            elif usage_percent > 80:
                status = "degraded"

            return {
                "status": status,
                "disk_usage_percent": round(usage_percent, 2),
                "free_space_gb": round(free_space_gb, 2),
                "total_space_gb": round(total_space_gb, 2)
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()

            SYSTEM_MEMORY_GAUGE.set(memory.used)

            status = "healthy"
            if memory.percent > 90:
                status = "unhealthy"
            elif memory.percent > 80:
                status = "degraded"

            return {
                "status": status,
                "usage_percent": round(memory.percent, 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2)
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            # Get CPU usage over a short interval
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            SYSTEM_CPU_GAUGE.set(cpu_percent)

            status = "healthy"
            if cpu_percent > 90:
                status = "unhealthy"
            elif cpu_percent > 80:
                status = "degraded"

            return {
                "status": status,
                "usage_percent": round(cpu_percent, 2),
                "cpu_count": cpu_count,
                "load_average": list(os.getloadavg()) if hasattr(os, 'getloadavg') else None
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def _check_external_services(self) -> Dict[str, Any]:
        """Check external service connectivity."""
        results = {}

        async with httpx.AsyncClient(timeout=10.0) as client:
            for service_name, url in self.external_services.items():
                try:
                    start_time = time.time()

                    # Skip actual API calls to avoid rate limiting in health checks
                    # Instead, do a basic connectivity test
                    if service_name == 'openai':
                        # Just check if we can resolve the domain
                        response = await client.get('https://openai.com', follow_redirects=True)
                        status = "healthy" if response.status_code < 500 else "degraded"
                    elif service_name == 'sec_edgar':
                        # Test SEC EDGAR with a minimal request
                        response = await client.head('https://www.sec.gov')
                        status = "healthy" if response.status_code < 500 else "degraded"
                    else:
                        status = "unknown"

                    response_time = time.time() - start_time

                    results[service_name] = {
                        "status": status,
                        "response_time_ms": round(response_time * 1000, 2)
                    }

                except Exception as e:
                    results[service_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }

        return results

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        try:
            process = psutil.Process()

            return {
                "process_memory_mb": round(process.memory_info().rss / (1024**2), 2),
                "process_cpu_percent": round(process.cpu_percent(), 2),
                "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0,
                "threads": process.num_threads(),
                "connections": len(process.connections()) if hasattr(process, 'connections') else 0
            }

        except Exception:
            return {}

    def _determine_overall_status(self, checks: Dict[str, Any]) -> str:
        """Determine overall health status from individual checks."""
        statuses = []

        for check_name, check_result in checks.items():
            if isinstance(check_result, dict) and 'status' in check_result:
                statuses.append(check_result['status'])
            elif isinstance(check_result, dict):
                # For external services, check individual service statuses
                for service_status in check_result.values():
                    if isinstance(service_status, dict) and 'status' in service_status:
                        statuses.append(service_status['status'])

        # Priority: unhealthy > degraded > healthy
        if 'unhealthy' in statuses:
            return 'unhealthy'
        elif 'degraded' in statuses:
            return 'degraded'
        else:
            return 'healthy'

    def _get_version(self) -> str:
        """Get application version."""
        try:
            from finchat_sec_qa import __version__
            return __version__
        except ImportError:
            return "unknown"


# Ready and liveness probe functions
async def readiness_probe() -> Dict[str, Any]:
    """Kubernetes readiness probe - checks if app is ready to serve traffic."""
    checker = HealthChecker()

    # For readiness, we only check essential services
    checks = {
        'database': await checker._check_database(),
        'cache': await checker._check_cache(),
    }

    overall_status = checker._determine_overall_status(checks)

    return {
        "status": overall_status,
        "ready": overall_status in ["healthy", "degraded"],
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


async def liveness_probe() -> Dict[str, Any]:
    """Kubernetes liveness probe - checks if app is alive."""
    return {
        "status": "healthy",
        "alive": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": time.time() - HealthChecker().app_start_time
    }


def get_metrics() -> str:
    """Get Prometheus metrics."""
    return generate_latest().decode('utf-8')
