from __future__ import annotations

import os
import logging
import time
import hmac
import hashlib
import re
import secrets
import base64
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from flask import Flask, request, abort, jsonify, g, Response

from .edgar_client import EdgarClient
from .qa_engine import FinancialQAEngine
from .risk_intelligence import RiskAnalyzer
from .logging_utils import configure_logging
from .config import get_config
from .query_handler import QueryHandler
from .validation import validate_text_safety, validate_ticker
from .rate_limiting import DistributedRateLimiter
from .utils import TimeBoundedCache

# Backward compatibility - RateLimiter is now an alias for DistributedRateLimiter
RateLimiter = DistributedRateLimiter


class CSRFProtection:
    """CSRF token management and validation with bounded memory usage."""
    
    def __init__(self) -> None:
        config = get_config()
        self.max_cache_size = config.CSRF_MAX_CACHE_SIZE
        self.tokens = TimeBoundedCache[str, float](
            max_size=self.max_cache_size,
            ttl_seconds=config.CSRF_TOKEN_EXPIRY_SECONDS
        )
        self.secret_key = os.urandom(32)  # Session-specific secret
    
    def generate_token(self) -> str:
        """Generate a new CSRF token with automatic expiration."""
        token = secrets.token_urlsafe(32)
        expiry_time = time.time() + self.tokens.ttl_seconds
        self.tokens.set(token, expiry_time)
        
        # Cleanup expired tokens periodically
        self.tokens.cleanup_expired()
        
        return token
    
    def validate_token(self, token: str) -> bool:
        """Validate a CSRF token (automatically handles expiration)."""
        if not token:
            return False
        
        # TimeBoundedCache automatically handles expiration
        expiry_time = self.tokens.get(token)
        return expiry_time is not None
    
    def _cleanup_expired_tokens(self) -> None:
        """Remove expired tokens from memory (handled automatically by TimeBoundedCache)."""
        self.tokens.cleanup_expired()


class BruteForceProtection:
    """Brute force protection with exponential backoff."""
    
    def __init__(self) -> None:
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.success_attempts: Dict[str, float] = {}
    
    def record_failed_attempt(self, client_id: str) -> None:
        """Record a failed authentication attempt."""
        now = time.time()
        self.failed_attempts[client_id].append(now)
        
        # Keep only recent attempts within lockout duration
        config = get_config()
        self.failed_attempts[client_id] = [
            attempt for attempt in self.failed_attempts[client_id]
            if now - attempt < config.LOCKOUT_DURATION_SECONDS
        ]
    
    def record_successful_attempt(self, client_id: str) -> None:
        """Record a successful authentication - resets failures."""
        self.success_attempts[client_id] = time.time()
        if client_id in self.failed_attempts:
            del self.failed_attempts[client_id]
    
    def is_blocked(self, client_id: str) -> bool:
        """Check if client is temporarily blocked due to brute force."""
        config = get_config()
        now = time.time()
        failed_count = len(self.failed_attempts[client_id])
        
        if failed_count < config.FAILED_ATTEMPTS_LOCKOUT_THRESHOLD:
            return False
        
        # Exponential backoff: base^(attempts-threshold) * unit_seconds
        last_attempt = max(self.failed_attempts[client_id]) if self.failed_attempts[client_id] else 0
        backoff_seconds = (config.EXPONENTIAL_BACKOFF_BASE ** (failed_count - config.FAILED_ATTEMPTS_LOCKOUT_THRESHOLD)) * config.EXPONENTIAL_BACKOFF_UNIT_SECONDS
        
        return now - last_attempt < backoff_seconds


app = Flask(__name__)

# Configure Flask security settings
config = get_config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_REQUEST_SIZE_MB * 1024 * 1024  # Convert MB to bytes

configure_logging("INFO")
logger = logging.getLogger(__name__)

SECRET_TOKEN = config.SECRET_TOKEN
client = EdgarClient("FinChatWeb")
engine = FinancialQAEngine(
    storage_path=Path(os.path.expanduser("~/.cache/finchat_sec_qa/index.joblib"))
)
risk = RiskAnalyzer()
query_handler = QueryHandler(client, engine)

# Security components
rate_limiter = RateLimiter()  # Uses config defaults
brute_force_protection = BruteForceProtection()
csrf_protection = CSRFProtection()

# Resource cleanup using Flask teardown handlers instead of atexit
@app.teardown_appcontext
def cleanup_resources(exception: Optional[Exception]) -> None:
    """Clean up resources when application context tears down."""
    try:
        logger.debug("Cleaning up resources")
        if engine is not None:
            engine.save()
            logger.debug("QA engine saved successfully")
    except Exception as e:
        logger.error("Error during resource cleanup: %s", e)


def validate_token_strength(token: str) -> bool:
    """Validate that a token meets security requirements."""
    if not token or not isinstance(token, str):
        return False
    
    # Minimum length
    if len(token) < config.MIN_TOKEN_LENGTH:
        return False
    
    # Must contain mix of characters
    has_upper = bool(re.search(r'[A-Z]', token))
    has_lower = bool(re.search(r'[a-z]', token))
    has_digit = bool(re.search(r'\d', token))
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', token))
    
    # At least MIN_PASSWORD_CRITERIA of 4 character types
    return sum([has_upper, has_lower, has_digit, has_special]) >= config.MIN_PASSWORD_CRITERIA


def _validate_token_constant_time(provided_token: str, expected_token: str) -> bool:
    """Compare tokens in constant time to prevent timing attacks."""
    if not provided_token or not expected_token:
        return False
    
    # Use HMAC for constant-time comparison
    return hmac.compare_digest(provided_token.encode(), expected_token.encode())


def _get_client_ip() -> str:
    """Get client IP address with secure proxy header validation."""
    import ipaddress
    
    direct_ip = request.environ.get('REMOTE_ADDR', 'unknown')
    
    # Only trust proxy headers if they come from trusted networks
    try:
        direct_addr = ipaddress.ip_address(direct_ip)
        
        # Define trusted proxy networks (private networks commonly used for load balancers)
        trusted_networks = [
            ipaddress.ip_network('10.0.0.0/8'),      # Private network
            ipaddress.ip_network('172.16.0.0/12'),   # Private network  
            ipaddress.ip_network('192.168.0.0/16'),  # Private network
            ipaddress.ip_network('127.0.0.0/8'),     # Localhost
        ]
        
        # Check if direct IP is from a trusted proxy
        is_trusted_proxy = any(direct_addr in network for network in trusted_networks)
        
        if is_trusted_proxy:
            # Check X-Forwarded-For first
            forwarded_ip = request.headers.get('X-Forwarded-For')
            if forwarded_ip:
                # Take the first IP and validate it
                client_ip = forwarded_ip.split(',')[0].strip()
                try:
                    # Validate that it's a valid IP address
                    ipaddress.ip_address(client_ip)
                    return client_ip
                except ValueError:
                    # Invalid IP in header, fall back to direct IP
                    pass
            
            # Check X-Real-IP as secondary option
            real_ip = request.headers.get('X-Real-IP')
            if real_ip:
                try:
                    ipaddress.ip_address(real_ip.strip())
                    return real_ip.strip()
                except ValueError:
                    pass
    
    except ValueError:
        # Invalid direct IP, this shouldn't happen in normal operations
        pass
    
    # Return direct IP if no trusted proxy headers or validation failed
    return direct_ip


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
    
    # Extract token from Authorization header only (security: no query parameter tokens)
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        provided_token = auth_header[7:]  # Remove 'Bearer ' prefix
    else:
        provided_token = None  # No fallback to query parameters for security
    
    # Validate token
    if not provided_token or not _validate_token_constant_time(provided_token, SECRET_TOKEN):
        brute_force_protection.record_failed_attempt(client_ip)
        logger.warning("Authentication failed for IP: %s", client_ip)
        abort(401, description="Invalid authentication token")
    
    # Record successful authentication
    brute_force_protection.record_successful_attempt(client_ip)
    g.authenticated_ip = client_ip


@app.after_request
def add_security_headers(response: Response) -> Response:
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = config.XSS_PROTECTION_MODE
    response.headers['Strict-Transport-Security'] = f'max-age={config.HSTS_MAX_AGE}; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    
    # Add CORS headers
    _add_cors_headers(response)
    
    return response


def _add_cors_headers(response: Response) -> None:
    """Add CORS headers to response if origin is allowed."""
    origin = request.headers.get('Origin')
    
    if origin and origin in config.CORS_ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = str(config.CORS_MAX_AGE)
        
        if config.CORS_ALLOW_CREDENTIALS:
            response.headers['Access-Control-Allow-Credentials'] = 'true'


@app.before_request
def add_security_headers() -> None:
    """Add security headers to all responses."""
    # Skip security headers for OPTIONS requests
    if request.method == 'OPTIONS':
        return
    
    @app.after_request
    def apply_security_headers(response: Response) -> Response:
        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
        return response


def _validate_csrf_token() -> None:
    """Validate CSRF token for state-changing operations."""
    if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
        # Exempt certain endpoints from CSRF protection
        exempt_paths = ['/csrf-token']  # CSRF token generation itself
        if request.path in exempt_paths:
            return
            
        csrf_token = request.headers.get('X-CSRF-Token')
        if not csrf_token or not csrf_protection.validate_token(csrf_token):
            logger.warning("CSRF token validation failed for IP: %s", _get_client_ip())
            abort(403, description="CSRF token missing or invalid")


@app.route('/csrf-token', methods=['GET'])
def get_csrf_token() -> Response:
    """Generate and return a CSRF token."""
    _auth()  # Require authentication for CSRF token
    
    token = csrf_protection.generate_token()
    response = jsonify({'csrf_token': token})
    _add_cors_headers(response)
    return response


@app.route('/<path:path>', methods=['OPTIONS'])
@app.route('/', methods=['OPTIONS'])
def handle_options(path: str = '') -> Response:
    """Handle CORS preflight requests."""
    response = jsonify({'status': 'ok'})
    _add_cors_headers(response)
    return response


@app.route("/query", methods=["POST"])
def query() -> Response:
    _auth()
    _validate_csrf_token()
    data = request.json or {}
    
    # Validate request data using shared validation
    try:
        ticker = validate_ticker(data.get("ticker", ""))
        question = validate_text_safety(data.get("question", ""), "question")
        form_type = data.get("form_type", "10-K")
        limit = data.get("limit", 1)
        
        # Additional validation for limit
        if not isinstance(limit, int) or limit < 1:
            abort(400, description="limit must be a positive integer")
            
    except ValueError as e:
        logger.warning("Invalid query request data: %s", e)
        abort(400, description=str(e))
    except Exception as e:
        logger.error("Unexpected error validating query request: %s", e)
        abort(500, description="Internal server error")
    
    logger.info("Processing query for ticker: %s, question: %s", ticker, question[:50])
    
    # Process query using shared handler
    try:
        answer, citations = query_handler.process_query(ticker, question, form_type, limit)
        logger.info("Query completed successfully with %d citations", len(citations))
        return jsonify({
            "answer": answer, 
            "citations": query_handler.serialize_citations(citations)
        })
    except ValueError as e:
        logger.warning("Query processing failed: %s", e)
        abort(404, description=str(e))
    except FileNotFoundError as e:
        logger.error("Filing file not found: %s", e)
        abort(500, description="Error processing filing")
    except Exception as e:
        logger.error("Error processing query: %s", e)
        abort(500, description="Error processing query")


@app.route("/risk", methods=["POST"])
def risk_endpoint() -> Response:
    _auth()
    _validate_csrf_token()
    data = request.json or {}
    
    # Validate request data using shared validation
    try:
        text = validate_text_safety(data.get("text", ""), "text")
    except ValueError as e:
        logger.warning("Invalid risk request data: %s", e)
        abort(400, description=str(e))
    except Exception as e:
        logger.error("Unexpected error validating risk request: %s", e)
        abort(500, description="Internal server error")
    
    logger.info("Processing risk analysis for text of %d characters", len(text))
    
    # Perform risk assessment
    try:
        assessment = risk.assess(text)
        logger.info("Risk analysis completed with %d flags", len(assessment.flags))
        return jsonify({"sentiment": assessment.sentiment, "flags": assessment.flags})
    except Exception as e:
        logger.error("Error performing risk assessment: %s", e)
        abort(500, description="Error performing risk assessment")
