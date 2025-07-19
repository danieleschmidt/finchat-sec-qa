from __future__ import annotations

from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr, validator

from .edgar_client import EdgarClient
from .qa_engine import FinancialQAEngine
from .risk_intelligence import RiskAnalyzer
from .logging_utils import configure_logging


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


class QueryRequest(BaseModel):
    question: constr(min_length=1)
    ticker: constr(min_length=1, max_length=5)
    form_type: str = "10-K"
    
    @validator('ticker')
    def ticker_must_be_alpha(cls, v):
        if not v.isalpha():
            raise ValueError('ticker must contain only letters')
        return v.upper()





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
    text: constr(min_length=1)


@app.post("/risk")
def analyze_risk(req: RiskRequest):
    assessment = risk.assess(req.text)
    return {"sentiment": assessment.sentiment, "flags": assessment.flags}
