"""Prometheus metrics collection for FinChat SEC QA service."""
from __future__ import annotations

import time
from typing import Dict

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# HTTP request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# Business metrics
qa_queries_total = Counter(
    'qa_queries_total',
    'Total QA queries processed',
    ['ticker', 'form_type', 'status']
)

qa_query_duration_seconds = Histogram(
    'qa_query_duration_seconds',
    'QA query processing duration in seconds',
    ['ticker', 'form_type']
)

risk_analyses_total = Counter(
    'risk_analyses_total',
    'Total risk analyses performed',
    ['status']
)

# Service health metrics
service_health = Gauge(
    'service_health',
    'Service health status (1=healthy, 0=unhealthy)',
    ['service']
)


class MetricsMiddleware:
    """FastAPI middleware to collect HTTP metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        method = request.method
        path = request.url.path

        # Normalize endpoint paths to avoid high cardinality
        endpoint = self._normalize_endpoint(path)

        start_time = time.time()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = str(message["status"])
                duration = time.time() - start_time

                # Record metrics
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code
                ).inc()

                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)

            await send(message)

        await self.app(scope, receive, send_wrapper)

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint paths to avoid high cardinality metrics."""
        # Map known endpoints
        endpoint_map = {
            '/query': '/query',
            '/risk': '/risk',
            '/health': '/health',
            '/metrics': '/metrics'
        }

        return endpoint_map.get(path, 'unknown')


def record_qa_query(ticker: str, form_type: str, duration: float, status: str = 'success'):
    """Record QA query metrics."""
    # Sanitize inputs for metrics labels
    safe_ticker = (ticker or 'unknown')[:10]  # Limit length and handle None
    safe_form_type = form_type or 'unknown'

    qa_queries_total.labels(
        ticker=safe_ticker,
        form_type=safe_form_type,
        status=status
    ).inc()

    if status == 'success':  # Only record duration for successful queries
        qa_query_duration_seconds.labels(
            ticker=safe_ticker,
            form_type=safe_form_type
        ).observe(duration)


def record_risk_analysis(status: str = 'success'):
    """Record risk analysis metrics."""
    risk_analyses_total.labels(status=status).inc()


def update_service_health(services: Dict[str, str]) -> None:
    """Update service health metrics based on health check."""
    for service, status in services.items():
        health_value = 1.0 if status == 'ready' else 0.0
        service_health.labels(service=service).set(health_value)


def get_metrics() -> Response:
    """Return Prometheus metrics in the expected format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
