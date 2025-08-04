from __future__ import annotations

from contextlib import asynccontextmanager
import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_config
from .edgar_client import AsyncEdgarClient
from .logging_utils import configure_logging
from .metrics import (
    MetricsMiddleware,
    get_metrics,
    record_qa_query,
    record_risk_analysis,
    update_service_health,
)
from .qa_engine import FinancialQAEngine
from .query_handler import AsyncQueryHandler
from .risk_intelligence import RiskAnalyzer
from .validation import validate_text_safety, validate_ticker


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"

        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request size limits."""

    def __init__(self, app, max_size_bytes: int):
        super().__init__(app)
        self.max_size_bytes = max_size_bytes

    async def dispatch(self, request: Request, call_next):
        # Check Content-Length header
        content_length = request.headers.get('content-length')
        if content_length:
            if int(content_length) > self.max_size_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request too large. Maximum size is {self.max_size_bytes} bytes"
                )

        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources."""
    configure_logging("INFO")
    cache = Path.home() / ".cache" / "finchat_sec_qa"
    cache.mkdir(parents=True, exist_ok=True)
    app.state.client = AsyncEdgarClient("FinChatBot")
    app.state.engine = FinancialQAEngine(storage_path=cache / "index.joblib")
    app.state.query_handler = AsyncQueryHandler(app.state.client, app.state.engine)
    try:
        yield
    finally:
        engine = app.state.engine
        if engine is not None:
            engine.save()
        # Close async client
        if app.state.client is not None:
            await app.state.client.session.aclose()
        app.state.client = None
        app.state.engine = None
        app.state.query_handler = None


app = FastAPI(lifespan=lifespan)

# Security configuration
config = get_config()

# Add security middleware (order matters - most specific first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimitMiddleware, max_size_bytes=config.MAX_REQUEST_SIZE_MB * 1024 * 1024)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['Content-Type', 'Authorization', 'X-CSRF-Token'],  # Allow CSRF token header
    max_age=config.CORS_MAX_AGE,
)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

risk = RiskAnalyzer()


class QueryRequest(BaseModel):
    """Query request with basic validation - detailed validation happens in the endpoint."""
    question: str
    ticker: str
    form_type: str = "10-K"
    limit: int = 1


class RiskRequest(BaseModel):
    """Risk analysis request with basic validation - detailed validation happens in the endpoint."""
    text: str





@app.post("/query")
async def query(req: QueryRequest) -> Dict[str, Any]:
    import time
    start_time = time.time()

    query_handler: AsyncQueryHandler | None = app.state.query_handler
    if query_handler is None:  # pragma: no cover - startup sets this
        raise HTTPException(status_code=500, detail="Server not ready")

    # Validate input using shared validation
    try:
        ticker = validate_ticker(req.ticker)
        question = validate_text_safety(req.question, "question")

        # Validate limit
        if req.limit < 1:
            raise ValueError("limit must be a positive integer")

    except ValueError as e:
        record_qa_query(req.ticker, req.form_type, time.time() - start_time, 'validation_error')
        raise HTTPException(status_code=400, detail=str(e))

    # Process query using shared async handler
    try:
        answer, citations = await query_handler.process_query(
            ticker, question, req.form_type, req.limit
        )

        # Record successful query metrics
        duration = time.time() - start_time
        record_qa_query(ticker, req.form_type, duration, 'success')

        return {
            "answer": answer,
            "citations": query_handler.serialize_citations(citations)
        }
    except ValueError as e:
        record_qa_query(ticker, req.form_type, time.time() - start_time, 'not_found')
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError:
        record_qa_query(ticker, req.form_type, time.time() - start_time, 'file_error')
        raise HTTPException(status_code=500, detail="Error processing filing")
    except Exception:
        record_qa_query(ticker, req.form_type, time.time() - start_time, 'internal_error')
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/risk")
def analyze_risk(req: RiskRequest) -> Dict[str, Any]:
    # Validate input using shared validation
    try:
        text = validate_text_safety(req.text, "text")
    except ValueError as e:
        record_risk_analysis('validation_error')
        raise HTTPException(status_code=400, detail=str(e))

    # Perform risk assessment
    try:
        assessment = risk.assess(text)
        record_risk_analysis('success')
        return {"sentiment": assessment.sentiment, "flags": assessment.flags}
    except Exception:
        record_risk_analysis('internal_error')
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint for container orchestration and monitoring.
    
    Returns:
        Health status with service availability and version information.
    """
    from . import __version__

    # Check service availability
    services = {}

    # Check Edgar client availability
    if hasattr(app.state, 'client') and app.state.client is not None:
        services['edgar_client'] = 'ready'
    else:
        services['edgar_client'] = 'unavailable'

    # Check QA engine availability
    if hasattr(app.state, 'engine') and app.state.engine is not None:
        services['qa_engine'] = 'ready'
    else:
        services['qa_engine'] = 'unavailable'

    # Check query handler availability
    if hasattr(app.state, 'query_handler') and app.state.query_handler is not None:
        services['query_handler'] = 'ready'
    else:
        services['query_handler'] = 'unavailable'

    # Overall status - healthy if core services are ready
    core_services_ready = (
        services.get('edgar_client') == 'ready' and
        services.get('qa_engine') == 'ready' and
        services.get('query_handler') == 'ready'
    )

    status = 'healthy' if core_services_ready else 'degraded'

    # Update service health metrics
    update_service_health(services)

    return {
        'status': status,
        'version': __version__,
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'services': services
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint.
    
    Returns:
        Metrics in Prometheus format for scraping.
    """
    return get_metrics()
