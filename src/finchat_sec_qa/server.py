from __future__ import annotations

from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr, validator

from .edgar_client import EdgarClient
from .qa_engine import FinancialQAEngine
from .risk_intelligence import RiskAnalyzer
from .logging_utils import configure_logging
from .config import get_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources."""
    configure_logging("INFO")
    cache = Path.home() / ".cache" / "finchat_sec_qa"
    cache.mkdir(parents=True, exist_ok=True)
    app.state.client = EdgarClient("FinChatBot")
    app.state.engine = FinancialQAEngine(storage_path=cache / "index.joblib")
    try:
        yield
    finally:
        engine = app.state.engine
        if engine is not None:
            engine.save()
        app.state.client = None
        app.state.engine = None


app = FastAPI(lifespan=lifespan)

risk = RiskAnalyzer()


def _get_query_request_model():
    """Create QueryRequest model with config-based constraints."""
    config = get_config()
    
    class QueryRequest(BaseModel):
        question: constr(min_length=1, max_length=config.MAX_QUESTION_LENGTH)
        ticker: constr(min_length=1, max_length=config.MAX_TICKER_LENGTH)
        form_type: str = "10-K"
        
        @validator('ticker')
        def ticker_must_be_valid(cls, v):
            import re
            if not v or not isinstance(v, str):
                raise ValueError('ticker must be a non-empty string')
            
            v = v.strip().upper()
            
            # Strict validation: 1-MAX_TICKER_LENGTH uppercase letters only, no special characters
            pattern = f'^[A-Z]{{1,{config.MAX_TICKER_LENGTH}}}$'
            if not re.match(pattern, v):
                raise ValueError(f'ticker must be 1-{config.MAX_TICKER_LENGTH} uppercase letters only (A-Z)')
            
            # Additional check against known ticker patterns to prevent injection
            if any(char in v for char in ['<', '>', '&', '"', "'", '/', '\\', '%']):
                raise ValueError('ticker contains invalid characters')
            
            return v
        
        @validator('form_type')
        def form_type_must_be_valid(cls, v):
            import re
            if not v or not isinstance(v, str):
                raise ValueError('form_type must be a non-empty string')
            
            v = v.strip().upper()
            
            # Allow alphanumeric and hyphens, reasonable length limit
            pattern = f'^[A-Z0-9-]{{1,{config.MAX_FORM_TYPE_LENGTH}}}$'
            if not re.match(pattern, v):
                raise ValueError(f'form_type must be 1-{config.MAX_FORM_TYPE_LENGTH} characters, alphanumeric and hyphens only')
            
            return v
        
        @validator('question')
        def question_must_be_safe(cls, v):
            if not v or not isinstance(v, str):
                raise ValueError('question must be a non-empty string')
            
            v = v.strip()
            
            # Basic XSS protection - reject obvious script injection attempts
            dangerous_patterns = ['<script', 'javascript:', 'vbscript:', 'onload=', 'onerror=', 'onclick=']
            v_lower = v.lower()
            
            for pattern in dangerous_patterns:
                if pattern in v_lower:
                    raise ValueError('question contains potentially dangerous content')
            
            return v
    
    return QueryRequest


def _get_risk_request_model():
    """Create RiskRequest model with config-based constraints."""
    config = get_config()
    
    class RiskRequest(BaseModel):
        text: constr(min_length=1, max_length=config.MAX_TEXT_INPUT_LENGTH)
    
    return RiskRequest


# Create models using current config
QueryRequest = _get_query_request_model()
RiskRequest = _get_risk_request_model()





@app.post("/query")
def query(req: QueryRequest):
    client: EdgarClient | None = app.state.client
    engine: FinancialQAEngine | None = app.state.engine
    if client is None or engine is None:  # pragma: no cover - startup sets these
        raise HTTPException(status_code=500, detail="Server not ready")
    filings = client.get_recent_filings(
        req.ticker, form_type=req.form_type, limit=1
    )
    if not filings:
        raise HTTPException(status_code=404, detail="Filing not found")
    path = client.download_filing(filings[0])
    text = Path(path).read_text()
    engine.add_document(filings[0].accession_no, text)
    answer, cites = engine.answer_with_citations(req.question)
    return {"answer": answer, "citations": [c.__dict__ for c in cites]}


class RiskRequest(BaseModel):
    text: constr(min_length=1, max_length=50000)  # Set reasonable upper limit
    
    @validator('text')
    def text_must_be_safe(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('text must be a non-empty string')
        
        v = v.strip()
        
        # Basic XSS protection - reject obvious script injection attempts
        dangerous_patterns = ['<script', 'javascript:', 'vbscript:', 'onload=', 'onerror=', 'onclick=']
        v_lower = v.lower()
        
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError('text contains potentially dangerous content')
        
        return v


@app.post("/risk")
def analyze_risk(req: RiskRequest):
    assessment = risk.assess(req.text)
    return {"sentiment": assessment.sentiment, "flags": assessment.flags}
