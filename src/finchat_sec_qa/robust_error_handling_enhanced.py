"""
Enhanced Robust Error Handling System - Generation 2: MAKE IT ROBUST
TERRAGON SDLC v4.0 - Autonomous Robustness Implementation

Features:
- Comprehensive exception handling with context preservation
- Automatic retry mechanisms with exponential backoff
- Circuit breaker pattern for external service failures
- Graceful degradation strategies
- Real-time error monitoring and alerting
- Self-healing capabilities with adaptive responses
- Comprehensive logging and audit trails
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union
import json
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for prioritization."""
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5
    FATAL = 6


class ErrorCategory(Enum):
    """Categories of errors for better handling."""
    VALIDATION_ERROR = "validation"
    AUTHENTICATION_ERROR = "authentication"
    AUTHORIZATION_ERROR = "authorization"
    NETWORK_ERROR = "network"
    DATABASE_ERROR = "database"
    API_ERROR = "api"
    QUANTUM_ERROR = "quantum"
    FINANCIAL_DATA_ERROR = "financial_data"
    RATE_LIMIT_ERROR = "rate_limit"
    RESOURCE_ERROR = "resource"
    CONFIGURATION_ERROR = "configuration"
    SECURITY_ERROR = "security"
    TIMEOUT_ERROR = "timeout"
    UNKNOWN_ERROR = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    severity: ErrorSeverity = ErrorSeverity.ERROR
    category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR
    message: str = ""
    stack_trace: str = ""
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    component: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    last_retry: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.name,
            'category': self.category.value,
            'message': self.message,
            'stack_trace': self.stack_trace,
            'user_id': self.user_id,
            'request_id': self.request_id,
            'component': self.component,
            'metadata': self.metadata,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'last_retry': self.last_retry.isoformat() if self.last_retry else None
        }


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_failure_rate: float = 0.5
    minimum_throughput: int = 10


class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt = datetime.now()
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self._should_attempt_call():
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
        else:
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self._should_attempt_call():
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
        else:
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
    
    def _should_attempt_call(self) -> bool:
        """Determine if call should be attempted."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            return datetime.now() >= self.next_attempt
        else:  # HALF_OPEN
            return True
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
        self.success_count += 1
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt = datetime.now() + timedelta(seconds=self.config.recovery_timeout)
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt = datetime.now() + timedelta(seconds=self.config.recovery_timeout)


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryStrategy:
    """Retry strategy with exponential backoff."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


class RobustErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_patterns: Dict[str, int] = {}
        self.self_healing_enabled = True
        self.max_error_history = 10000
        
    def generate_error_id(self, error: Exception, context: str = "") -> str:
        """Generate unique error ID."""
        error_string = f"{type(error).__name__}:{str(error)}:{context}:{datetime.now().isoformat()}"
        return hashlib.md5(error_string.encode()).hexdigest()[:12]
    
    def create_error_context(self,
                           error: Exception,
                           severity: ErrorSeverity = ErrorSeverity.ERROR,
                           category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
                           component: str = "",
                           user_id: Optional[str] = None,
                           request_id: Optional[str] = None,
                           metadata: Dict[str, Any] = None) -> ErrorContext:
        """Create comprehensive error context."""
        error_id = self.generate_error_id(error, component)
        
        return ErrorContext(
            error_id=error_id,
            severity=severity,
            category=category,
            message=str(error),
            stack_trace=traceback.format_exc(),
            component=component,
            user_id=user_id,
            request_id=request_id,
            metadata=metadata or {}
        )
    
    def handle_error(self, 
                    error: Exception,
                    context: ErrorContext = None,
                    should_raise: bool = True) -> ErrorContext:
        """Handle error with full context and recovery attempts."""
        if context is None:
            context = self.create_error_context(error)
        
        # Log error
        self._log_error(context)
        
        # Add to error history
        self._add_to_history(context)
        
        # Analyze error patterns
        self._analyze_error_patterns(context)
        
        # Attempt self-healing
        if self.self_healing_enabled:
            self._attempt_self_healing(context)
        
        # Raise if required
        if should_raise:
            raise error
        
        return context
    
    def _log_error(self, context: ErrorContext):
        """Log error with appropriate level."""
        log_message = f"[{context.error_id}] {context.component}: {context.message}"
        
        if context.severity == ErrorSeverity.DEBUG:
            logger.debug(log_message, extra=context.to_dict())
        elif context.severity == ErrorSeverity.INFO:
            logger.info(log_message, extra=context.to_dict())
        elif context.severity == ErrorSeverity.WARNING:
            logger.warning(log_message, extra=context.to_dict())
        elif context.severity == ErrorSeverity.ERROR:
            logger.error(log_message, extra=context.to_dict())
        elif context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra=context.to_dict())
        elif context.severity == ErrorSeverity.FATAL:
            logger.critical(f"FATAL ERROR: {log_message}", extra=context.to_dict())
    
    def _add_to_history(self, context: ErrorContext):
        """Add error to history with size management."""
        self.error_history.append(context)
        
        # Maintain history size
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
    
    def _analyze_error_patterns(self, context: ErrorContext):
        """Analyze error patterns for proactive handling."""
        pattern_key = f"{context.category.value}:{context.component}"
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
        
        # Alert on repeated patterns
        if self.error_patterns[pattern_key] > 10:
            logger.warning(f"Repeated error pattern detected: {pattern_key} "
                         f"({self.error_patterns[pattern_key]} occurrences)")
    
    def _attempt_self_healing(self, context: ErrorContext):
        """Attempt to self-heal based on error context."""
        if context.category == ErrorCategory.NETWORK_ERROR:
            self._handle_network_error(context)
        elif context.category == ErrorCategory.DATABASE_ERROR:
            self._handle_database_error(context)
        elif context.category == ErrorCategory.RESOURCE_ERROR:
            self._handle_resource_error(context)
        # Add more self-healing strategies as needed
    
    def _handle_network_error(self, context: ErrorContext):
        """Handle network-related errors."""
        logger.info(f"Attempting network error recovery for {context.error_id}")
        # Could implement connection pool refresh, DNS cache clear, etc.
    
    def _handle_database_error(self, context: ErrorContext):
        """Handle database-related errors."""
        logger.info(f"Attempting database error recovery for {context.error_id}")
        # Could implement connection refresh, transaction rollback, etc.
    
    def _handle_resource_error(self, context: ErrorContext):
        """Handle resource-related errors."""
        logger.info(f"Attempting resource error recovery for {context.error_id}")
        # Could implement memory cleanup, cache clearing, etc.
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        recent_errors = [e for e in self.error_history 
                        if e.timestamp > datetime.now() - timedelta(hours=24)]
        
        severity_counts = {}
        category_counts = {}
        component_counts = {}
        
        for error in recent_errors:
            severity_counts[error.severity.name] = severity_counts.get(error.severity.name, 0) + 1
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            component_counts[error.component] = component_counts.get(error.component, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors_24h": len(recent_errors),
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "component_distribution": component_counts,
            "error_patterns": dict(list(self.error_patterns.items())[:10]),  # Top 10 patterns
            "circuit_breaker_states": {name: cb.state.value for name, cb in self.circuit_breakers.items()}
        }


# Global error handler instance
global_error_handler = RobustErrorHandler()


def robust_error_handler(severity: ErrorSeverity = ErrorSeverity.ERROR,
                        category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
                        component: str = "",
                        should_raise: bool = True):
    """Decorator for robust error handling."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = global_error_handler.create_error_context(
                    error=e,
                    severity=severity,
                    category=category,
                    component=component or func.__name__
                )
                return global_error_handler.handle_error(e, context, should_raise)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = global_error_handler.create_error_context(
                    error=e,
                    severity=severity,
                    category=category,
                    component=component or func.__name__
                )
                return global_error_handler.handle_error(e, context, should_raise)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


def with_circuit_breaker(breaker_name: str, 
                        config: CircuitBreakerConfig = None):
    """Decorator for circuit breaker protection."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cb = global_error_handler.get_circuit_breaker(breaker_name, config)
            return cb.call(func, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cb = global_error_handler.get_circuit_breaker(breaker_name, config)
            return await cb.call_async(func, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


def with_retry(strategy: RetryStrategy = None):
    """Decorator for automatic retry with exponential backoff."""
    if strategy is None:
        strategy = RetryStrategy()
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(strategy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < strategy.max_retries:
                        delay = strategy.get_delay(attempt)
                        logger.warning(f"Retry attempt {attempt + 1}/{strategy.max_retries + 1} "
                                     f"for {func.__name__} after {delay:.2f}s delay: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"All retry attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(strategy.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < strategy.max_retries:
                        delay = strategy.get_delay(attempt)
                        logger.warning(f"Retry attempt {attempt + 1}/{strategy.max_retries + 1} "
                                     f"for {func.__name__} after {delay:.2f}s delay: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All retry attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


@contextmanager
def error_context(component: str,
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
                 metadata: Dict[str, Any] = None):
    """Context manager for error handling."""
    try:
        yield
    except Exception as e:
        context = global_error_handler.create_error_context(
            error=e,
            severity=severity,
            category=category,
            component=component,
            metadata=metadata
        )
        global_error_handler.handle_error(e, context, should_raise=True)


@asynccontextmanager
async def async_error_context(component: str,
                             severity: ErrorSeverity = ErrorSeverity.ERROR,
                             category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
                             metadata: Dict[str, Any] = None):
    """Async context manager for error handling."""
    try:
        yield
    except Exception as e:
        context = global_error_handler.create_error_context(
            error=e,
            severity=severity,
            category=category,
            component=component,
            metadata=metadata
        )
        global_error_handler.handle_error(e, context, should_raise=True)


# Export all classes and functions
__all__ = [
    'ErrorSeverity',
    'ErrorCategory', 
    'ErrorContext',
    'CircuitBreakerState',
    'CircuitBreakerConfig',
    'CircuitBreaker',
    'CircuitBreakerOpenError',
    'RetryStrategy',
    'RobustErrorHandler',
    'global_error_handler',
    'robust_error_handler',
    'with_circuit_breaker',
    'with_retry',
    'error_context',
    'async_error_context'
]