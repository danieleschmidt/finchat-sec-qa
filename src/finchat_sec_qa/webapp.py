from __future__ import annotations

import os
import atexit
import logging
import time
import hmac
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from flask import Flask, request, abort, jsonify, g
from pydantic import ValidationError
from .server import QueryRequest, RiskRequest

from .edgar_client import EdgarClient
from .qa_engine import FinancialQAEngine
from .risk_intelligence import RiskAnalyzer
from .logging_utils import configure_logging

class RateLimiter:
    """Rate limiting with sliding window."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()
        
        # Clean old requests outside the window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
        
        return False


class BruteForceProtection:
    """Brute force protection with exponential backoff."""
    
    def __init__(self):
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.success_attempts: Dict[str, float] = {}
    
    def record_failed_attempt(self, client_id: str) -> None:
        """Record a failed authentication attempt."""
        now = time.time()
        self.failed_attempts[client_id].append(now)
        
        # Keep only recent attempts (last hour)
        self.failed_attempts[client_id] = [
            attempt for attempt in self.failed_attempts[client_id]
            if now - attempt < 3600
        ]
    
    def record_successful_attempt(self, client_id: str) -> None:
        """Record a successful authentication - resets failures."""
        self.success_attempts[client_id] = time.time()
        if client_id in self.failed_attempts:
            del self.failed_attempts[client_id]
    
    def is_blocked(self, client_id: str) -> bool:
        """Check if client is temporarily blocked due to brute force."""
        now = time.time()
        failed_count = len(self.failed_attempts[client_id])
        
        if failed_count < 3:
            return False
        
        # Exponential backoff: 2^(attempts-3) minutes
        last_attempt = max(self.failed_attempts[client_id]) if self.failed_attempts[client_id] else 0
        backoff_seconds = (2 ** (failed_count - 3)) * 60
        
        return now - last_attempt < backoff_seconds


app = Flask(__name__)
configure_logging("INFO")
logger = logging.getLogger(__name__)

SECRET_TOKEN = os.getenv("FINCHAT_TOKEN")
client = EdgarClient("FinChatWeb")
engine = FinancialQAEngine(
    storage_path=Path(os.path.expanduser("~/.cache/finchat_sec_qa/index.joblib"))
)
risk = RiskAnalyzer()

# Security components
rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)  # 100 requests per hour
brute_force_protection = BruteForceProtection()

atexit.register(engine.save)


def validate_token_strength(token: str) -> bool:
    """Validate that a token meets security requirements."""
    if not token or not isinstance(token, str):
        return False
    
    # Minimum length
    if len(token) < 16:
        return False
    
    # Must contain mix of characters
    has_upper = bool(re.search(r'[A-Z]', token))
    has_lower = bool(re.search(r'[a-z]', token))
    has_digit = bool(re.search(r'\d', token))
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', token))
    
    # At least 3 of 4 character types
    return sum([has_upper, has_lower, has_digit, has_special]) >= 3


def _validate_token_constant_time(provided_token: str, expected_token: str) -> bool:
    """Compare tokens in constant time to prevent timing attacks."""
    if not provided_token or not expected_token:
        return False
    
    # Use HMAC for constant-time comparison
    return hmac.compare_digest(provided_token.encode(), expected_token.encode())


def _get_client_ip() -> str:
    """Get client IP address, considering proxy headers."""
    # Check for forwarded IP (common in production behind load balancers)
    forwarded_ip = request.headers.get('X-Forwarded-For')
    if forwarded_ip:
        # Take the first IP in case of multiple
        return forwarded_ip.split(',')[0].strip()
    
    return request.environ.get('REMOTE_ADDR', 'unknown')


def _auth() -> None:
    """Enhanced authentication with rate limiting and brute force protection."""
    client_ip = _get_client_ip()
    
    # Skip auth if no token is configured
    if not SECRET_TOKEN:
        return
    
    # Validate token strength on startup (log warning if weak)
    if not validate_token_strength(SECRET_TOKEN):
        logger.warning("FINCHAT_TOKEN is weak. Use a strong token with 16+ characters, "
                      "mixing uppercase, lowercase, numbers, and special characters.")
    
    # Check rate limiting
    if not rate_limiter.is_allowed(client_ip):
        logger.warning("Rate limit exceeded for IP: %s", client_ip)
        abort(429, description="Rate limit exceeded. Try again later.")
    
    # Check brute force protection
    if brute_force_protection.is_blocked(client_ip):
        logger.warning("IP blocked due to brute force attempts: %s", client_ip)
        abort(429, description="Too many failed attempts. Try again later.")
    
    # Extract token from header or query parameter
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        provided_token = auth_header[7:]  # Remove 'Bearer ' prefix
    else:
        provided_token = request.args.get("token")
    
    # Validate token
    if not provided_token or not _validate_token_constant_time(provided_token, SECRET_TOKEN):
        brute_force_protection.record_failed_attempt(client_ip)
        logger.warning("Authentication failed for IP: %s", client_ip)
        abort(401, description="Invalid authentication token")
    
    # Record successful authentication
    brute_force_protection.record_successful_attempt(client_ip)
    g.authenticated_ip = client_ip


@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response


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
