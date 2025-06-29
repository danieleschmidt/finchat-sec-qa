from __future__ import annotations

import os
import atexit
from pathlib import Path
from flask import Flask, request, abort, jsonify
from .server import QueryRequest, RiskRequest

from .edgar_client import EdgarClient
from .qa_engine import FinancialQAEngine
from .risk_intelligence import RiskAnalyzer
from .logging_utils import configure_logging

app = Flask(__name__)
configure_logging("INFO")

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
    try:
        req = QueryRequest(**data)
    except Exception:
        abort(400)
    filings = client.get_recent_filings(req.ticker, limit=1)
    if not filings:
        abort(404)
    path = client.download_filing(filings[0])
    text = Path(path).read_text()
    engine.add_document(filings[0].accession_no, text)
    answer, cites = engine.answer_with_citations(req.question)
    return jsonify({"answer": answer, "citations": [c.__dict__ for c in cites]})


@app.route("/risk", methods=["POST"])
def risk_endpoint() -> object:
    _auth()
    data = request.json or {}
    try:
        req = RiskRequest(**data)
    except Exception:
        abort(400)
    assessment = risk.assess(req.text)
    return jsonify({"sentiment": assessment.sentiment, "flags": assessment.flags})
