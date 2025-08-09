from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import datetime
import logging
from pathlib import Path
from typing import Any, Dict
import time

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
    get_business_tracker
)
from .qa_engine import FinancialQAEngine
from .query_handler import AsyncQueryHandler
from .risk_intelligence import RiskAnalyzer
from .validation import validate_text_safety, validate_ticker
from .circuit_breaker import circuit_breaker_context, CircuitBreakerConfig, CircuitBreakerOpenError
from .intelligent_cache import CacheType
from .i18n import get_globalization_service, localize_for_request

logger = logging.getLogger(__name__)


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
    """Initialize shared resources with robust error handling and enhanced features."""
    import asyncio
    from .circuit_breaker import get_circuit_breaker_manager, CircuitBreakerConfig
    from .auto_scaling import get_auto_scaler, start_auto_scaling, create_managed_thread_pool
    from .intelligent_cache import create_intelligent_cache, CachePolicy, CacheType
    from .metrics import start_metrics_collection, get_business_tracker
    from .health import HealthChecker
    
    configure_logging("INFO")
    cache = Path.home() / ".cache" / "finchat_sec_qa"
    cache.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize circuit breakers with custom configurations
        cb_manager = get_circuit_breaker_manager()
        
        # Configure circuit breakers for critical services
        edgar_cb_config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30.0,
            slow_request_threshold=15.0,
            expected_error_types=(HTTPException,)
        )
        
        # Initialize intelligent caches
        app.state.query_cache = await create_intelligent_cache(
            "query_results", 
            max_size=5000,
            max_memory_mb=512,
            policy=CachePolicy.ADAPTIVE
        )
        
        app.state.document_cache = await create_intelligent_cache(
            "document_chunks",
            max_size=10000, 
            max_memory_mb=256,
            policy=CachePolicy.LRU
        )
        
        app.state.risk_cache = await create_intelligent_cache(
            "risk_analysis",
            max_size=2000,
            max_memory_mb=128,
            policy=CachePolicy.TTL
        )
        
        # Initialize auto-scaling resources
        auto_scaler = get_auto_scaler()
        app.state.thread_pool = create_managed_thread_pool(
            "api_processing", 
            initial_size=4,
            min_size=2, 
            max_size=16
        )
        
        # Initialize core services with enhanced configurations
        app.state.client = AsyncEdgarClient("FinChatBot/2.0")
        app.state.engine = FinancialQAEngine(storage_path=cache / "index.joblib")
        app.state.query_handler = AsyncQueryHandler(app.state.client, app.state.engine)
        app.state.health_checker = HealthChecker()
        
        # Start background services
        app.state.background_tasks = []
        
        # Start metrics collection
        metrics_task = asyncio.create_task(start_metrics_collection())
        app.state.background_tasks.append(metrics_task)
        
        # Start auto-scaling
        scaling_task = asyncio.create_task(start_auto_scaling())
        app.state.background_tasks.append(scaling_task)
        
        logger.info("FinChat server initialized with enhanced robustness features")
        logger.info(f"- Circuit breakers: Enabled for critical services")
        logger.info(f"- Intelligent caching: 3 adaptive cache pools")
        logger.info(f"- Auto-scaling: Thread pool management enabled")
        logger.info(f"- Background tasks: {len(app.state.background_tasks)} monitoring services")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise
    finally:
        # Graceful shutdown
        logger.info("Starting graceful shutdown...")
        
        # Cancel background tasks
        for task in getattr(app.state, 'background_tasks', []):
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        
        # Save engine state
        engine = getattr(app.state, 'engine', None)
        if engine is not None:
            try:
                engine.save()
                logger.info("Saved engine state")
            except Exception as e:
                logger.error(f"Failed to save engine state: {e}")
        
        # Close async client
        client = getattr(app.state, 'client', None)
        if client is not None:
            try:
                if hasattr(client, 'session') and client.session:
                    await client.session.aclose()
                logger.info("Closed EDGAR client")
            except Exception as e:
                logger.error(f"Failed to close EDGAR client: {e}")
        
        # Stop auto-scaling
        try:
            from .auto_scaling import stop_auto_scaling
            await stop_auto_scaling()
            logger.info("Stopped auto-scaling")
        except Exception as e:
            logger.error(f"Failed to stop auto-scaling: {e}")
        
        # Close cache systems
        for cache_name in ['query_cache', 'document_cache', 'risk_cache']:
            cache = getattr(app.state, cache_name, None)
            if cache:
                try:
                    await cache.stop_maintenance()
                    logger.info(f"Stopped {cache_name} maintenance")
                except Exception as e:
                    logger.error(f"Failed to stop {cache_name}: {e}")
        
        # Clean up state
        for attr in ['client', 'engine', 'query_handler', 'health_checker', 
                    'query_cache', 'document_cache', 'risk_cache', 'thread_pool']:
            setattr(app.state, attr, None)
        
        logger.info("Graceful shutdown completed")


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
    """Enhanced query endpoint with circuit breaker protection and intelligent caching."""
    start_time = time.time()
    
    # Check if services are ready
    query_handler = app.state.query_handler
    query_cache = app.state.query_cache
    if query_handler is None or query_cache is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    # Validate input using shared validation
    try:
        ticker = validate_ticker(req.ticker)
        question = validate_text_safety(req.question, "question")

        # Validate limit
        if req.limit < 1 or req.limit > 10:
            raise ValueError("limit must be between 1 and 10")

    except ValueError as e:
        get_business_tracker().track_qa_query(
            req.ticker, req.form_type, "validation", 
            time.time() - start_time, 'validation_error'
        )
        raise HTTPException(status_code=400, detail=str(e))

    # Generate cache key
    cache_key = f"{ticker}:{req.form_type}:{hash(question)}:{req.limit}"
    
    try:
        # Try to get from cache first
        cached_result = await query_cache.get(cache_key, CacheType.QUERY_RESULT)
        if cached_result is not None:
            logger.debug(f"Cache hit for query: {ticker} - {question[:50]}...")
            
            # Record cache hit metrics
            duration = time.time() - start_time
            get_business_tracker().track_qa_query(
                ticker, req.form_type, "cached_query",
                duration, 'cache_hit'
            )
            
            return cached_result

        # Process query with circuit breaker protection
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30.0,
            slow_request_threshold=20.0
        )
        
        async with circuit_breaker_context("query_processing", circuit_breaker_config):
            # Execute query in managed thread pool
            answer, citations = await query_handler.process_query(
                ticker, question, req.form_type, req.limit
            )
            
            # Prepare response
            result = {
                "answer": answer,
                "citations": query_handler.serialize_citations(citations),
                "metadata": {
                    "ticker": ticker,
                    "form_type": req.form_type,
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                    "source": "live",
                    "cache_key": cache_key
                }
            }
            
            # Cache the result with intelligent scoring
            computation_cost = time.time() - start_time
            importance_score = min(0.9, 0.5 + (len(citations) * 0.1))  # More citations = higher importance
            
            await query_cache.set(
                cache_key, 
                result,
                cache_type=CacheType.QUERY_RESULT,
                ttl_seconds=3600,  # 1 hour TTL
                importance_score=importance_score,
                computation_cost=computation_cost,
                metadata={
                    "ticker": ticker,
                    "form_type": req.form_type,
                    "question_length": len(question),
                    "citation_count": len(citations)
                }
            )
            
            # Record successful query metrics
            duration = time.time() - start_time
            get_business_tracker().track_qa_query(
                ticker, req.form_type, "live_query",
                duration, 'success',
                tokens_processed=len(question) + len(answer)
            )
            
            logger.info(f"Processed query for {ticker} in {duration:.3f}s, cached for reuse")
            
            # Apply globalization and localization
            try:
                from starlette.requests import Request as StarletteRequest
                client_ip = None
                if isinstance(req, StarletteRequest):
                    client_ip = req.client.host if req.client else None
                
                localized_result = localize_for_request(
                    result, 
                    dict(req.headers) if hasattr(req, 'headers') else {},
                    client_ip
                )
                return localized_result
            except Exception as e:
                logger.warning(f"Localization failed, returning original result: {e}")
                return result
            
    except CircuitBreakerOpenError as e:
        # Circuit breaker is open
        logger.warning(f"Query circuit breaker open: {e}")
        get_business_tracker().track_qa_query(
            ticker, req.form_type, "circuit_breaker",
            time.time() - start_time, 'circuit_breaker_open'
        )
        raise HTTPException(
            status_code=503, 
            detail="Service temporarily unavailable due to high error rate. Please try again later."
        )
        
    except ValueError as e:
        # Data not found or invalid
        duration = time.time() - start_time
        get_business_tracker().track_qa_query(
            ticker, req.form_type, "not_found",
            duration, 'not_found'
        )
        raise HTTPException(status_code=404, detail=str(e))
        
    except FileNotFoundError as e:
        # File processing error
        duration = time.time() - start_time
        get_business_tracker().track_qa_query(
            ticker, req.form_type, "file_error",
            duration, 'file_error'
        )
        raise HTTPException(status_code=500, detail="Error accessing filing data")
        
    except asyncio.TimeoutError:
        # Query timeout
        duration = time.time() - start_time
        get_business_tracker().track_qa_query(
            ticker, req.form_type, "timeout",
            duration, 'timeout'
        )
        raise HTTPException(status_code=504, detail="Query processing timeout")
        
    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error processing query: {e}", exc_info=True)
        duration = time.time() - start_time
        get_business_tracker().track_qa_query(
            ticker, req.form_type, "internal_error",
            duration, 'internal_error'
        )
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


@app.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Enhanced health check with comprehensive system monitoring and diagnostics."""
    from . import __version__
    from .circuit_breaker import get_all_circuit_breaker_status
    from .auto_scaling import get_scaling_status
    from .intelligent_cache import get_cache_manager
    
    try:
        # Get comprehensive health status using HealthChecker
        health_checker = getattr(app.state, 'health_checker', None)
        if health_checker:
            comprehensive_health = await health_checker.check_system_health()
        else:
            comprehensive_health = {"status": "unknown", "message": "Health checker not available"}
        
        # Get cache manager statistics
        try:
            cache_manager = await get_cache_manager()
            cache_stats = await cache_manager.get_global_statistics()
        except Exception as e:
            cache_stats = {"error": f"Cache statistics unavailable: {e}"}
        
        # Get circuit breaker details
        try:
            circuit_breaker_status = await get_all_circuit_breaker_status()
        except Exception as e:
            circuit_breaker_status = {"error": f"Circuit breaker status unavailable: {e}"}
        
        # Get scaling details
        try:
            scaling_status = await get_scaling_status()
        except Exception as e:
            scaling_status = {"error": f"Auto-scaling status unavailable: {e}"}
        
        return {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'version': __version__,
            'detailed_system_health': comprehensive_health,
            'cache_statistics': cache_stats,
            'circuit_breaker_details': circuit_breaker_status,
            'auto_scaling_details': scaling_status,
            'background_tasks': {
                'count': len(getattr(app.state, 'background_tasks', [])),
                'running': len([t for t in getattr(app.state, 'background_tasks', []) if not t.done()])
            },
            'intelligent_features': {
                'circuit_breakers': 'enabled',
                'adaptive_caching': 'enabled', 
                'auto_scaling': 'enabled',
                'comprehensive_monitoring': 'enabled'
            }
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.datetime.utcnow().isoformat()
        }


@app.get("/ready")
async def readiness_probe() -> Dict[str, Any]:
    """Kubernetes readiness probe - checks if service is ready to accept traffic."""
    # Check core services required for serving requests
    core_services = ['client', 'engine', 'query_handler', 'query_cache']
    
    ready = True
    services_status = {}
    
    for service in core_services:
        if hasattr(app.state, service) and getattr(app.state, service) is not None:
            services_status[service] = True
        else:
            services_status[service] = False
            ready = False
    
    return {
        'ready': ready,
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'services': services_status
    }


@app.get("/live")
async def liveness_probe() -> Dict[str, Any]:
    """Kubernetes liveness probe - checks if service is alive."""
    from . import __version__
    return {
        'alive': True,
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'version': __version__,
        'uptime_seconds': time.time() - getattr(app.state, 'start_time', time.time())
    }


@app.get("/locales")
async def get_supported_locales() -> Dict[str, Any]:
    """Get list of supported languages and regions for internationalization."""
    try:
        globalization_service = get_globalization_service()
        locales = globalization_service.get_supported_locales()
        
        return {
            "supported_locales": locales[:20],  # Return first 20 to avoid huge response
            "total_count": len(locales),
            "default_locale": "en_US",
            "features": {
                "multi_language_ui": True,
                "localized_financial_terms": True,
                "regional_compliance": True,
                "currency_formatting": True,
                "date_time_formatting": True
            }
        }
    except Exception as e:
        logger.error(f"Error fetching supported locales: {e}")
        raise HTTPException(status_code=500, detail="Unable to fetch locale information")


@app.get("/privacy-notice")
async def get_privacy_notice(language: str = "en", region: str = "us") -> Dict[str, Any]:
    """Get privacy notice for specific language and region."""
    try:
        from .i18n import get_privacy_notice
        
        privacy_notice = get_privacy_notice(language, region)
        
        return {
            "privacy_notice": privacy_notice,
            "language": language,
            "region": region,
            "last_updated": "2025-08-09",
            "version": "2.0"
        }
    except Exception as e:
        logger.error(f"Error generating privacy notice: {e}")
        raise HTTPException(status_code=500, detail="Unable to generate privacy notice")


@app.get("/compliance/{region}")
async def get_compliance_info(region: str) -> Dict[str, Any]:
    """Get compliance information for a specific region."""
    try:
        from .i18n import SupportedRegion, get_globalization_service
        
        try:
            region_enum = SupportedRegion(region.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported region: {region}")
        
        globalization_service = get_globalization_service()
        compliance_requirements = globalization_service.compliance_manager.get_compliance_requirements(region_enum)
        
        return {
            "region": region,
            "compliance_framework": compliance_requirements,
            "data_protection_summary": {
                "encryption_required": True,
                "audit_logging": True,
                "data_minimization": True,
                "purpose_limitation": True,
                "user_rights": True
            },
            "applicable_regulations": {
                "us": ["SOX", "SEC", "CCPA"],
                "eu": ["GDPR", "MiFID II", "ESMA"],
                "uk": ["UK GDPR", "FCA Rules"],
                "ca": ["PIPEDA", "CSA Rules"],
                "sg": ["PDPA", "MAS Regulations"],
                "jp": ["APPI", "JFSA Rules"]
            }.get(region.lower(), ["General data protection"])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching compliance info: {e}")
        raise HTTPException(status_code=500, detail="Unable to fetch compliance information")


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint.
    
    Returns:
        Metrics in Prometheus format for scraping.
    """
    return get_metrics()
