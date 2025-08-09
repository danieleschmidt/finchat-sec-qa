"""Enhanced Prometheus metrics collection for FinChat SEC QA service with adaptive monitoring."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Dict, Optional, Set

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    generate_latest,
)

logger = logging.getLogger(__name__)

# Enhanced HTTP request metrics
http_requests_total = Counter(
    'finchat_http_requests_total',
    'Total HTTP requests processed',
    ['method', 'endpoint', 'status_code', 'client_type']
)

http_request_duration_seconds = Histogram(
    'finchat_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, float("inf"))
)

http_request_size_bytes = Histogram(
    'finchat_http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

http_response_size_bytes = Histogram(
    'finchat_http_response_size_bytes', 
    'HTTP response size in bytes',
    ['method', 'endpoint', 'status_code']
)

# Business metrics with enhanced dimensions
qa_queries_total = Counter(
    'finchat_qa_queries_total',
    'Total QA queries processed',
    ['ticker', 'form_type', 'status', 'query_type', 'source']
)

qa_query_duration_seconds = Histogram(
    'finchat_qa_query_duration_seconds',
    'QA query processing duration in seconds',
    ['ticker', 'form_type', 'query_type'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, float("inf"))
)

qa_query_tokens = Histogram(
    'finchat_qa_query_tokens_total',
    'Total tokens processed in QA queries',
    ['ticker', 'form_type', 'token_type'],
    buckets=(100, 500, 1000, 2000, 5000, 10000, 20000, 50000, float("inf"))
)

risk_analyses_total = Counter(
    'finchat_risk_analyses_total',
    'Total risk analyses performed',
    ['status', 'risk_level', 'analysis_type']
)

risk_score_distribution = Histogram(
    'finchat_risk_score_distribution',
    'Distribution of risk scores',
    ['ticker', 'analysis_type'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# System performance metrics
system_memory_usage_bytes = Gauge(
    'finchat_system_memory_usage_bytes',
    'Current system memory usage in bytes'
)

system_cpu_usage_percent = Gauge(
    'finchat_system_cpu_usage_percent',
    'Current system CPU usage percentage'
)

system_disk_usage_bytes = Gauge(
    'finchat_system_disk_usage_bytes',
    'Current system disk usage in bytes',
    ['mount_point']
)

# Cache metrics
cache_operations_total = Counter(
    'finchat_cache_operations_total',
    'Total cache operations',
    ['operation', 'cache_type', 'status']
)

cache_hit_ratio = Gauge(
    'finchat_cache_hit_ratio',
    'Cache hit ratio (0-1)',
    ['cache_type']
)

cache_size_bytes = Gauge(
    'finchat_cache_size_bytes',
    'Current cache size in bytes',
    ['cache_type']
)

cache_entries_total = Gauge(
    'finchat_cache_entries_total',
    'Total number of cache entries',
    ['cache_type']
)

# Database/Storage metrics
storage_operations_total = Counter(
    'finchat_storage_operations_total',
    'Total storage operations',
    ['operation', 'storage_type', 'status']
)

storage_operation_duration_seconds = Histogram(
    'finchat_storage_operation_duration_seconds',
    'Storage operation duration in seconds',
    ['operation', 'storage_type']
)

# Service health and availability metrics
service_health = Gauge(
    'finchat_service_health',
    'Service health status (1=healthy, 0.5=degraded, 0=unhealthy)',
    ['service', 'check_type']
)

service_availability = Gauge(
    'finchat_service_availability',
    'Service availability percentage over time window',
    ['service', 'time_window']
)

error_rate = Gauge(
    'finchat_error_rate',
    'Current error rate (errors per minute)',
    ['service', 'error_type']
)

# External service metrics
external_service_requests_total = Counter(
    'finchat_external_service_requests_total',
    'Total requests to external services',
    ['service', 'endpoint', 'status_code']
)

external_service_duration_seconds = Histogram(
    'finchat_external_service_duration_seconds',
    'External service request duration in seconds',
    ['service', 'endpoint']
)

# Rate limiting metrics
rate_limit_requests_total = Counter(
    'finchat_rate_limit_requests_total',
    'Total requests subject to rate limiting',
    ['client_id', 'limit_type', 'status']
)

rate_limit_current_usage = Gauge(
    'finchat_rate_limit_current_usage',
    'Current rate limit usage',
    ['client_id', 'limit_type']
)

# Security metrics
security_events_total = Counter(
    'finchat_security_events_total',
    'Total security events',
    ['event_type', 'severity', 'action_taken']
)

authentication_attempts_total = Counter(
    'finchat_authentication_attempts_total',
    'Total authentication attempts',
    ['auth_type', 'status', 'client_type']
)

# Quantum computing metrics (if available)
quantum_operations_total = Counter(
    'finchat_quantum_operations_total',
    'Total quantum computing operations',
    ['operation_type', 'quantum_backend', 'status']
)

quantum_circuit_depth = Histogram(
    'finchat_quantum_circuit_depth',
    'Quantum circuit depth distribution',
    ['algorithm_type'],
    buckets=(10, 20, 50, 100, 200, 500, 1000, 2000, 5000, float("inf"))
)

quantum_execution_time_seconds = Histogram(
    'finchat_quantum_execution_time_seconds',
    'Quantum algorithm execution time in seconds',
    ['algorithm_type', 'quantum_backend']
)

# Application metadata
application_info = Info(
    'finchat_application_info',
    'Application metadata and build information'
)

# Custom alerts and SLO tracking
slo_request_duration_seconds = Summary(
    'finchat_slo_request_duration_seconds',
    'Request duration for SLO tracking',
    ['service', 'endpoint']
)

alert_threshold_breaches_total = Counter(
    'finchat_alert_threshold_breaches_total',
    'Total alert threshold breaches',
    ['alert_name', 'severity', 'service']
)

# Business value metrics
user_satisfaction_score = Gauge(
    'finchat_user_satisfaction_score',
    'User satisfaction score (1-5)',
    ['feature', 'time_window']
)

business_value_generated = Counter(
    'finchat_business_value_generated_total',
    'Total business value generated (in USD equivalent)',
    ['value_type', 'feature']
)


class MetricsMiddleware:
    """Enhanced FastAPI middleware to collect comprehensive HTTP metrics."""

    def __init__(self, app):
        self.app = app
        self.request_count = 0
        self.active_requests = set()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        method = request.method
        path = request.url.path
        
        # Generate request ID for tracking
        self.request_count += 1
        request_id = f"{method}_{self.request_count}_{int(time.time() * 1000)}"
        self.active_requests.add(request_id)

        # Normalize endpoint paths to avoid high cardinality
        endpoint = self._normalize_endpoint(path)
        
        # Determine client type from headers
        client_type = self._get_client_type(request)

        start_time = time.time()
        request_size = 0

        # Calculate request size
        if hasattr(request, 'headers'):
            content_length = request.headers.get('content-length')
            if content_length:
                try:
                    request_size = int(content_length)
                except ValueError:
                    pass

        async def receive_wrapper():
            message = await receive()
            nonlocal request_size
            if message.get("type") == "http.request" and "body" in message:
                request_size += len(message["body"])
            return message

        response_size = 0
        status_code = None

        async def send_wrapper(message):
            nonlocal response_size, status_code
            
            if message["type"] == "http.response.start":
                status_code = str(message["status"])
            elif message["type"] == "http.response.body":
                if "body" in message:
                    response_size += len(message["body"])
                
                # Record metrics when response is complete
                if not message.get("more_body", False):
                    duration = time.time() - start_time
                    
                    # Record all metrics
                    http_requests_total.labels(
                        method=method,
                        endpoint=endpoint,
                        status_code=status_code,
                        client_type=client_type
                    ).inc()

                    http_request_duration_seconds.labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(duration)
                    
                    if request_size > 0:
                        http_request_size_bytes.labels(
                            method=method,
                            endpoint=endpoint
                        ).observe(request_size)
                    
                    if response_size > 0:
                        http_response_size_bytes.labels(
                            method=method,
                            endpoint=endpoint,
                            status_code=status_code
                        ).observe(response_size)
                    
                    # SLO tracking
                    slo_request_duration_seconds.labels(
                        service='api',
                        endpoint=endpoint
                    ).observe(duration)
                    
                    # Remove from active requests
                    self.active_requests.discard(request_id)

            await send(message)

        await self.app(scope, receive_wrapper, send_wrapper)

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint paths to avoid high cardinality metrics."""
        import re
        
        # Map known endpoints
        endpoint_patterns = {
            r'^/query/?$': '/query',
            r'^/risk/?$': '/risk',
            r'^/health/?$': '/health',
            r'^/metrics/?$': '/metrics',
            r'^/ready/?$': '/ready',
            r'^/live/?$': '/live',
            r'^/docs/?$': '/docs',
            r'^/openapi\.json$': '/openapi.json',
            r'^/static/.*': '/static/*',
            r'^/api/v\d+/.*': '/api/v*/*'
        }
        
        for pattern, normalized in endpoint_patterns.items():
            if re.match(pattern, path):
                return normalized
        
        # For unknown paths, categorize by pattern
        if path.startswith('/api/'):
            return '/api/*'
        elif '.' in path.split('/')[-1]:  # Likely a file
            return '/static/*'
        else:
            return '/unknown'

    def _get_client_type(self, request: Request) -> str:
        """Determine client type from request headers."""
        user_agent = request.headers.get('user-agent', '').lower()
        
        if 'python' in user_agent or 'requests' in user_agent or 'httpx' in user_agent:
            return 'api_client'
        elif 'curl' in user_agent:
            return 'curl'
        elif 'postman' in user_agent:
            return 'postman'
        elif 'swagger' in user_agent:
            return 'swagger_ui'
        elif any(browser in user_agent for browser in ['mozilla', 'chrome', 'safari', 'firefox']):
            return 'browser'
        else:
            return 'unknown'

    def get_active_requests_count(self) -> int:
        """Get current number of active requests."""
        return len(self.active_requests)


class AdaptiveMetricsCollector:
    """Adaptive metrics collector that adjusts collection frequency based on load."""
    
    def __init__(self):
        self.collection_interval = 60  # Default 1 minute
        self.min_interval = 10  # Minimum 10 seconds
        self.max_interval = 300  # Maximum 5 minutes
        self.load_threshold_high = 80  # High load threshold
        self.load_threshold_low = 20   # Low load threshold
        self.last_collection = 0
        self.metrics_history = []
        self.max_history = 100
        
    async def collect_system_metrics(self):
        """Collect system metrics with adaptive frequency."""
        current_time = time.time()
        
        if current_time - self.last_collection < self.collection_interval:
            return
        
        try:
            # Collect system metrics
            import psutil
            
            # Memory metrics
            memory = psutil.virtual_memory()
            system_memory_usage_bytes.set(memory.used)
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            system_cpu_usage_percent.set(cpu_percent)
            
            # Disk metrics
            for mount_point in ['/', '/tmp', '/var']:
                if os.path.exists(mount_point):
                    try:
                        disk_usage = psutil.disk_usage(mount_point)
                        system_disk_usage_bytes.labels(mount_point=mount_point).set(disk_usage.used)
                    except (PermissionError, FileNotFoundError):
                        continue
            
            # Adjust collection frequency based on system load
            avg_load = (memory.percent + cpu_percent) / 2
            
            if avg_load > self.load_threshold_high:
                self.collection_interval = max(self.min_interval, self.collection_interval * 0.8)
            elif avg_load < self.load_threshold_low:
                self.collection_interval = min(self.max_interval, self.collection_interval * 1.2)
            
            # Store metrics history
            self.metrics_history.append({
                'timestamp': current_time,
                'memory_percent': memory.percent,
                'cpu_percent': cpu_percent,
                'collection_interval': self.collection_interval
            })
            
            # Trim history
            if len(self.metrics_history) > self.max_history:
                self.metrics_history = self.metrics_history[-self.max_history:]
            
            self.last_collection = current_time
            
        except ImportError:
            logger.warning("psutil not available for system metrics collection")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class BusinessMetricsTracker:
    """Track business-specific metrics and KPIs."""
    
    def __init__(self):
        self.query_cache = {}
        self.performance_baseline = {
            'avg_query_time': 2.0,
            'success_rate_threshold': 0.95,
            'cache_hit_threshold': 0.8
        }
    
    def track_qa_query(self, ticker: str, form_type: str, query_type: str, 
                      duration: float, status: str, source: str = 'api',
                      tokens_processed: Optional[int] = None):
        """Track QA query with comprehensive metrics."""
        # Sanitize inputs
        safe_ticker = (ticker or 'unknown')[:10].upper()
        safe_form_type = (form_type or 'unknown')[:20]
        safe_query_type = (query_type or 'unknown')[:50]
        
        # Record basic metrics
        qa_queries_total.labels(
            ticker=safe_ticker,
            form_type=safe_form_type,
            status=status,
            query_type=safe_query_type,
            source=source
        ).inc()
        
        if status == 'success':
            qa_query_duration_seconds.labels(
                ticker=safe_ticker,
                form_type=safe_form_type,
                query_type=safe_query_type
            ).observe(duration)
            
            if tokens_processed:
                qa_query_tokens.labels(
                    ticker=safe_ticker,
                    form_type=safe_form_type,
                    token_type='processed'
                ).observe(tokens_processed)
        
        # Track performance against baseline
        if duration > self.performance_baseline['avg_query_time'] * 2:
            alert_threshold_breaches_total.labels(
                alert_name='slow_query',
                severity='warning',
                service='qa_engine'
            ).inc()
    
    def track_risk_analysis(self, ticker: str, risk_score: float, 
                           analysis_type: str = 'comprehensive', 
                           risk_level: str = 'unknown', status: str = 'success'):
        """Track risk analysis metrics."""
        safe_ticker = (ticker or 'unknown')[:10].upper()
        
        risk_analyses_total.labels(
            status=status,
            risk_level=risk_level,
            analysis_type=analysis_type
        ).inc()
        
        if status == 'success' and 0 <= risk_score <= 1:
            risk_score_distribution.labels(
                ticker=safe_ticker,
                analysis_type=analysis_type
            ).observe(risk_score)
    
    def track_cache_operation(self, cache_type: str, operation: str, 
                             status: str, hit_ratio: Optional[float] = None):
        """Track cache operations and performance."""
        cache_operations_total.labels(
            operation=operation,
            cache_type=cache_type,
            status=status
        ).inc()
        
        if hit_ratio is not None:
            cache_hit_ratio.labels(cache_type=cache_type).set(hit_ratio)
            
            # Alert on low cache hit ratio
            if hit_ratio < self.performance_baseline['cache_hit_threshold']:
                alert_threshold_breaches_total.labels(
                    alert_name='low_cache_hit_ratio',
                    severity='warning',
                    service='cache'
                ).inc()
    
    def track_external_service(self, service: str, endpoint: str, 
                              duration: float, status_code: int):
        """Track external service interactions."""
        external_service_requests_total.labels(
            service=service,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        external_service_duration_seconds.labels(
            service=service,
            endpoint=endpoint
        ).observe(duration)
    
    def track_quantum_operation(self, operation_type: str, quantum_backend: str,
                               duration: Optional[float] = None, 
                               circuit_depth: Optional[int] = None,
                               algorithm_type: Optional[str] = None,
                               status: str = 'success'):
        """Track quantum computing operations."""
        quantum_operations_total.labels(
            operation_type=operation_type,
            quantum_backend=quantum_backend,
            status=status
        ).inc()
        
        if circuit_depth is not None and algorithm_type:
            quantum_circuit_depth.labels(
                algorithm_type=algorithm_type
            ).observe(circuit_depth)
        
        if duration is not None and algorithm_type:
            quantum_execution_time_seconds.labels(
                algorithm_type=algorithm_type,
                quantum_backend=quantum_backend
            ).observe(duration)


# Global instances
_adaptive_collector = AdaptiveMetricsCollector()
_business_tracker = BusinessMetricsTracker()


@asynccontextmanager
async def metrics_context(operation_name: str):
    """Context manager for operation timing and error tracking."""
    start_time = time.time()
    try:
        yield
    except Exception as e:
        logger.error(f"Operation {operation_name} failed: {e}")
        raise
    finally:
        duration = time.time() - start_time
        logger.debug(f"Operation {operation_name} completed in {duration:.3f}s")


def timed_operation(operation_type: str):
    """Decorator for timing operations."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                # Record successful operation
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{operation_type} failed after {duration:.3f}s: {e}")
                raise
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{operation_type} failed after {duration:.3f}s: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


async def start_metrics_collection():
    """Start background metrics collection."""
    while True:
        try:
            await _adaptive_collector.collect_system_metrics()
            await asyncio.sleep(_adaptive_collector.collection_interval)
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying


def initialize_application_info():
    """Initialize application metadata metrics."""
    try:
        import platform
        from finchat_sec_qa import __version__
        
        application_info.info({
            'version': __version__,
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'build_date': os.getenv('BUILD_DATE', 'unknown'),
            'git_commit': os.getenv('GIT_COMMIT', 'unknown'),
            'environment': os.getenv('ENVIRONMENT', 'development')
        })
    except Exception as e:
        logger.warning(f"Could not initialize application info: {e}")


# Initialize on import
initialize_application_info()


# Convenience functions for backward compatibility
def record_qa_query(ticker: str, form_type: str, duration: float, status: str = 'success'):
    """Record QA query metrics (backward compatibility)."""
    _business_tracker.track_qa_query(ticker, form_type, 'general', duration, status)


def record_risk_analysis(status: str = 'success'):
    """Record risk analysis metrics (backward compatibility)."""
    _business_tracker.track_risk_analysis('unknown', 0.5, 'general', 'unknown', status)


def update_service_health(services: Dict[str, str]) -> None:
    """Update service health metrics based on health check."""
    for service, status in services.items():
        health_value = 1.0 if status == 'healthy' else 0.5 if status == 'degraded' else 0.0
        service_health.labels(service=service, check_type='general').set(health_value)


def get_metrics() -> Response:
    """Return Prometheus metrics in the expected format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Export convenience functions
def get_business_tracker() -> BusinessMetricsTracker:
    """Get the global business metrics tracker."""
    return _business_tracker


def get_adaptive_collector() -> AdaptiveMetricsCollector:
    """Get the global adaptive metrics collector."""
    return _adaptive_collector
