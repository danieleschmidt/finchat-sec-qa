from __future__ import annotations

import os
import atexit
import logging
from pathlib import Path
from flask import Flask, request, abort, jsonify
from pydantic import ValidationError
from .server import QueryRequest, RiskRequest

from .edgar_client import EdgarClient
from .qa_engine import FinancialQAEngine
from .risk_intelligence import RiskAnalyzer
from .logging_utils import configure_logging

app = Flask(__name__)
configure_logging("INFO")
logger = logging.getLogger(__name__)

SECRET_TOKEN = os.getenv("FINCHAT_TOKEN")
client = EdgarClient("FinChatWeb")
engine = FinancialQAEngine(
    storage_path=Path(os.path.expanduser("~/.cache/finchat_sec_qa/index.joblib"))
)
risk = RiskAnalyzer()

atexit.register(engine.save)


def _auth() -> None:
    if SECRET_TOKEN and request.args.get("token") != SECRET_TOKEN:
        abort(401)


@app.route("/query", methods=["POST"])
def query() -> object:
    _auth()
    data = request.json or {}
    
    # Validate request data
    try:
        req = QueryRequest(**data)
    except ValidationError as e:
        logger.warning("Invalid query request data: %s", e)
        abort(400, description=f"Invalid request data: {e}")
    except Exception as e:
        logger.error("Unexpected error validating query request: %s", e)
        abort(500, description="Internal server error")
    
    logger.info("Processing query for ticker: %s, question: %s", req.ticker, req.question)
    
    # Fetch SEC filings
    try:
        filings = client.get_recent_filings(req.ticker, limit=1)
        if not filings:
            logger.warning("No filings found for ticker: %s", req.ticker)
            abort(404, description=f"No filings found for ticker {req.ticker}")
    except Exception as e:
        logger.error("Error fetching filings for ticker %s: %s", req.ticker, e)
        abort(500, description="Error fetching SEC filings")
    
    # Download and process filing
    try:
        path = client.download_filing(filings[0])
        text = Path(path).read_text()
        engine.add_document(filings[0].accession_no, text)
        answer, cites = engine.answer_with_citations(req.question)
        logger.info("Query completed successfully with %d citations", len(cites))
        return jsonify({"answer": answer, "citations": [c.__dict__ for c in cites]})
    except FileNotFoundError as e:
        logger.error("Filing file not found: %s", e)
        abort(500, description="Error processing filing")
    except Exception as e:
        logger.error("Error processing query: %s", e)
        abort(500, description="Error processing query")


@app.route("/risk", methods=["POST"])
def risk_endpoint() -> object:
    _auth()
    data = request.json or {}
    
    # Validate request data
    try:
        req = RiskRequest(**data)
    except ValidationError as e:
        logger.warning("Invalid risk request data: %s", e)
        abort(400, description=f"Invalid request data: {e}")
    except Exception as e:
        logger.error("Unexpected error validating risk request: %s", e)
        abort(500, description="Internal server error")
    
    logger.info("Processing risk analysis for text of %d characters", len(req.text))
    
    # Perform risk assessment
    try:
        assessment = risk.assess(req.text)
        logger.info("Risk analysis completed with %d flags", len(assessment.flags))
        return jsonify({"sentiment": assessment.sentiment, "flags": assessment.flags})
    except Exception as e:
        logger.error("Error performing risk assessment: %s", e)
        abort(500, description="Error performing risk assessment")
