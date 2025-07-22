from __future__ import annotations

import datetime
from pathlib import Path
from typing import Type, Dict, Any, List

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .edgar_client import AsyncEdgarClient
from .qa_engine import FinancialQAEngine
from .risk_intelligence import RiskAnalyzer
from .logging_utils import configure_logging
from .config import get_config
from .query_handler import AsyncQueryHandler
from .validation import validate_text_safety, validate_ticker


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
        raise HTTPException(status_code=400, detail=str(e))
    
    # Process query using shared async handler
    try:
        answer, citations = await query_handler.process_query(
            ticker, question, req.form_type, req.limit
        )
        return {
            "answer": answer, 
            "citations": query_handler.serialize_citations(citations)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="Error processing filing")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/risk")
def analyze_risk(req: RiskRequest) -> Dict[str, Any]:
    # Validate input using shared validation
    try:
        text = validate_text_safety(req.text, "text")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Perform risk assessment
    try:
        assessment = risk.assess(text)
        return {"sentiment": assessment.sentiment, "flags": assessment.flags}
    except Exception as e:
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
    
    return {
        'status': status,
        'version': __version__,
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'services': services
    }
