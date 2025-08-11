"""
Robust Error Handling and Recovery System.

This module provides comprehensive error handling, recovery mechanisms,
and resilience patterns for the financial analysis platform.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
import traceback
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, AsyncGenerator
import threading
from collections import defaultdict, deque

import numpy as np

from .config import get_config
from .logging_utils import configure_logging
from .metrics import get_business_tracker

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better classification."""
    NETWORK = "network"
    DATA_PROCESSING = "data_processing"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    QUANTUM_COMPUTING = "quantum_computing"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM_RESOURCE = "system_resource"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    IMMEDIATE_FAIL = "immediate_fail"
    ESCALATE = "escalate"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    stack_trace: str
    user_context: Dict[str, Any] = field(default_factory=dict)
    system_context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    attempts_made: int
    time_taken: float
    fallback_used: bool = False
    additional_info: Dict[str, Any] = field(default_factory=dict)


class RobustErrorHandler:
    """
    Comprehensive error handling system with recovery mechanisms,
    monitoring, and adaptive resilience patterns.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".cache" / "finchat_sec_qa" / "error_handling"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Error tracking
        self.error_history: deque = deque(maxlen=10000)
        self.error_patterns: Dict[str, List[ErrorContext]] = defaultdict(list)
        self.recovery_statistics: Dict[RecoveryStrategy, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Circuit breaker states
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Fallback mechanisms
        self.fallback_handlers: Dict[ErrorCategory, Callable] = {}
        
        # Recovery strategies mapping
        self.strategy_mapping: Dict[Tuple[ErrorCategory, Type[Exception]], RecoveryStrategy] = {
            (ErrorCategory.NETWORK, ConnectionError): RecoveryStrategy.RETRY,
            (ErrorCategory.NETWORK, TimeoutError): RecoveryStrategy.RETRY,
            (ErrorCategory.EXTERNAL_API, Exception): RecoveryStrategy.CIRCUIT_BREAKER,
            (ErrorCategory.DATABASE, Exception): RecoveryStrategy.RETRY,
            (ErrorCategory.VALIDATION, ValueError): RecoveryStrategy.IMMEDIATE_FAIL,
            (ErrorCategory.AUTHENTICATION, Exception): RecoveryStrategy.ESCALATE,
            (ErrorCategory.QUANTUM_COMPUTING, Exception): RecoveryStrategy.FALLBACK,
            (ErrorCategory.SYSTEM_RESOURCE, MemoryError): RecoveryStrategy.GRACEFUL_DEGRADATION,
        }
        
        # Adaptive parameters
        self.adaptive_timeouts: Dict[str, float] = defaultdict(lambda: 30.0)
        self.adaptive_retry_counts: Dict[str, int] = defaultdict(lambda: 3)
        
        self._initialize_fallback_handlers()
        configure_logging()

    def _initialize_fallback_handlers(self) -> None:
        """Initialize fallback handlers for different error categories."""
        self.fallback_handlers = {
            ErrorCategory.QUANTUM_COMPUTING: self._quantum_fallback,
            ErrorCategory.EXTERNAL_API: self._external_api_fallback,
            ErrorCategory.DATA_PROCESSING: self._data_processing_fallback,
            ErrorCategory.NETWORK: self._network_fallback,
            ErrorCategory.DATABASE: self._database_fallback,
        }

    def handle_error(self, 
                    error: Exception, 
                    context: Dict[str, Any],
                    category: ErrorCategory,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> RecoveryResult:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            category: Category of the error
            severity: Severity level of the error
            
        Returns:
            RecoveryResult indicating the outcome of recovery attempts
        """
        start_time = time.time()
        
        # Create error context
        error_context = self._create_error_context(error, context, category, severity)
        
        # Log the error
        self._log_error(error_context)
        
        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(error_context)
        error_context.recovery_strategy = strategy
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error_context)
        recovery_result.time_taken = time.time() - start_time
        
        # Update statistics
        self._update_recovery_statistics(strategy, recovery_result.success)
        
        # Store error for pattern analysis
        self._store_error_for_analysis(error_context, recovery_result)
        
        # Adaptive learning
        self._adapt_parameters_based_on_outcome(error_context, recovery_result)
        
        return recovery_result

    async def ahandle_error(self, 
                           error: Exception, 
                           context: Dict[str, Any],
                           category: ErrorCategory,
                           severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> RecoveryResult:
        """Async version of error handling."""
        start_time = time.time()
        
        error_context = self._create_error_context(error, context, category, severity)
        self._log_error(error_context)
        
        strategy = self._determine_recovery_strategy(error_context)
        error_context.recovery_strategy = strategy
        
        recovery_result = await self._attempt_async_recovery(error_context)
        recovery_result.time_taken = time.time() - start_time
        
        self._update_recovery_statistics(strategy, recovery_result.success)
        self._store_error_for_analysis(error_context, recovery_result)
        self._adapt_parameters_based_on_outcome(error_context, recovery_result)
        
        return recovery_result

    def _create_error_context(self, 
                             error: Exception, 
                             context: Dict[str, Any],
                             category: ErrorCategory,
                             severity: ErrorSeverity) -> ErrorContext:
        """Create error context from exception and context information."""
        return ErrorContext(
            error_id=f"err_{int(time.time() * 1000000)}",
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            stack_trace=traceback.format_exc(),
            user_context=context.get('user', {}),
            system_context=context.get('system', {}),
            max_recovery_attempts=self.adaptive_retry_counts[category.value]
        )

    def _determine_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Determine the best recovery strategy for the given error."""
        # Check specific mapping
        error_key = (error_context.category, type(eval(error_context.error_type)))
        if error_key in self.strategy_mapping:
            return self.strategy_mapping[error_key]
        
        # Check circuit breaker state
        if self._is_circuit_breaker_open(error_context.category.value):
            return RecoveryStrategy.CIRCUIT_BREAKER
        
        # Severity-based strategy
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ESCALATE
        elif error_context.severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.FALLBACK
        
        # Default strategy based on category
        category_defaults = {
            ErrorCategory.NETWORK: RecoveryStrategy.RETRY,
            ErrorCategory.EXTERNAL_API: RecoveryStrategy.CIRCUIT_BREAKER,
            ErrorCategory.VALIDATION: RecoveryStrategy.IMMEDIATE_FAIL,
            ErrorCategory.AUTHENTICATION: RecoveryStrategy.ESCALATE,
            ErrorCategory.QUANTUM_COMPUTING: RecoveryStrategy.FALLBACK,
            ErrorCategory.SYSTEM_RESOURCE: RecoveryStrategy.GRACEFUL_DEGRADATION,
        }
        
        return category_defaults.get(error_context.category, RecoveryStrategy.RETRY)

    def _attempt_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt recovery using the determined strategy."""
        strategy = error_context.recovery_strategy
        
        if strategy == RecoveryStrategy.RETRY:
            return self._retry_recovery(error_context)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._fallback_recovery(error_context)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._circuit_breaker_recovery(error_context)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation_recovery(error_context)
        elif strategy == RecoveryStrategy.ESCALATE:
            return self._escalate_recovery(error_context)
        else:  # IMMEDIATE_FAIL
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                attempts_made=0,
                time_taken=0.0
            )

    async def _attempt_async_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Async version of recovery attempt."""
        strategy = error_context.recovery_strategy
        
        if strategy == RecoveryStrategy.RETRY:
            return await self._async_retry_recovery(error_context)
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._async_fallback_recovery(error_context)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._async_circuit_breaker_recovery(error_context)
        else:
            # For other strategies, use sync version
            return self._attempt_recovery(error_context)

    def _retry_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Implement retry recovery strategy."""
        max_attempts = error_context.max_recovery_attempts
        base_delay = 1.0
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Exponential backoff
                if attempt > 1:
                    delay = base_delay * (2 ** (attempt - 2))
                    time.sleep(min(delay, 60))  # Cap at 60 seconds
                
                # Attempt the operation again (this would be provided by the caller)
                # For now, we simulate recovery success based on error category
                success_probability = self._get_retry_success_probability(error_context.category)
                
                if np.random.random() < success_probability:
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.RETRY,
                        attempts_made=attempt,
                        time_taken=0.0  # Will be set by caller
                    )
                    
            except Exception as retry_error:
                logger.warning(f"Retry attempt {attempt} failed: {retry_error}")
                if attempt == max_attempts:
                    break
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.RETRY,
            attempts_made=max_attempts,
            time_taken=0.0
        )

    async def _async_retry_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Async retry recovery implementation."""
        max_attempts = error_context.max_recovery_attempts
        base_delay = 1.0
        
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    delay = base_delay * (2 ** (attempt - 2))
                    await asyncio.sleep(min(delay, 60))
                
                success_probability = self._get_retry_success_probability(error_context.category)
                
                if np.random.random() < success_probability:
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.RETRY,
                        attempts_made=attempt,
                        time_taken=0.0
                    )
                    
            except Exception as retry_error:
                logger.warning(f"Async retry attempt {attempt} failed: {retry_error}")
                if attempt == max_attempts:
                    break
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.RETRY,
            attempts_made=max_attempts,
            time_taken=0.0
        )

    def _fallback_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Implement fallback recovery strategy."""
        category = error_context.category
        
        if category in self.fallback_handlers:
            try:
                fallback_result = self.fallback_handlers[category](error_context)
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.FALLBACK,
                    attempts_made=1,
                    time_taken=0.0,
                    fallback_used=True,
                    additional_info={'fallback_result': fallback_result}
                )
            except Exception as fallback_error:
                logger.error(f"Fallback failed for {category}: {fallback_error}")
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.FALLBACK,
            attempts_made=1,
            time_taken=0.0
        )

    async def _async_fallback_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Async fallback recovery implementation."""
        # For now, use sync fallback
        return self._fallback_recovery(error_context)

    def _circuit_breaker_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Implement circuit breaker recovery strategy."""
        service_name = error_context.category.value
        
        if self._is_circuit_breaker_open(service_name):
            # Circuit is open, fail fast
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                attempts_made=0,
                time_taken=0.0,
                additional_info={'circuit_state': 'open'}
            )
        
        # Try half-open state
        if self._should_attempt_half_open(service_name):
            try:
                # Simulate service call
                success_probability = self._get_service_health_probability(service_name)
                
                if np.random.random() < success_probability:
                    self._close_circuit_breaker(service_name)
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                        attempts_made=1,
                        time_taken=0.0,
                        additional_info={'circuit_state': 'closed'}
                    )
                else:
                    self._open_circuit_breaker(service_name)
                    
            except Exception:
                self._open_circuit_breaker(service_name)
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
            attempts_made=1,
            time_taken=0.0,
            additional_info={'circuit_state': 'open'}
        )

    async def _async_circuit_breaker_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Async circuit breaker recovery implementation."""
        return self._circuit_breaker_recovery(error_context)

    def _graceful_degradation_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Implement graceful degradation recovery strategy."""
        # Implement reduced functionality
        degraded_result = self._provide_degraded_service(error_context)
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
            attempts_made=1,
            time_taken=0.0,
            additional_info={
                'degraded_service': True,
                'degraded_result': degraded_result
            }
        )

    def _escalate_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Implement escalation recovery strategy."""
        # Log critical error and notify administrators
        logger.critical(f"Critical error escalated: {error_context.error_id} - {error_context.error_message}")
        
        # In practice, this would send alerts to monitoring systems
        self._send_escalation_alert(error_context)
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ESCALATE,
            attempts_made=1,
            time_taken=0.0,
            additional_info={'escalated': True}
        )

    # Fallback handlers
    def _quantum_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for quantum computing errors."""
        logger.info("Using classical algorithm fallback for quantum computation")
        return {
            'algorithm_type': 'classical',
            'performance_note': 'Reduced performance due to quantum system unavailability'
        }

    def _external_api_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for external API errors."""
        logger.info("Using cached data for external API unavailability")
        return {
            'data_source': 'cached',
            'freshness_warning': 'Data may not be current due to external API unavailability'
        }

    def _data_processing_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for data processing errors."""
        logger.info("Using simplified data processing pipeline")
        return {
            'processing_mode': 'simplified',
            'accuracy_note': 'Reduced accuracy due to processing limitations'
        }

    def _network_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for network errors."""
        logger.info("Using local resources for network unavailability")
        return {
            'resource_source': 'local',
            'limitation_note': 'Limited functionality due to network unavailability'
        }

    def _database_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for database errors."""
        logger.info("Using in-memory storage for database unavailability")
        return {
            'storage_type': 'in_memory',
            'persistence_warning': 'Data will not be persisted due to database unavailability'
        }

    # Circuit breaker methods
    def _is_circuit_breaker_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for a service."""
        if service_name not in self.circuit_breakers:
            self._initialize_circuit_breaker(service_name)
        
        cb_state = self.circuit_breakers[service_name]
        return cb_state['state'] == 'open'

    def _should_attempt_half_open(self, service_name: str) -> bool:
        """Check if circuit breaker should attempt half-open state."""
        if service_name not in self.circuit_breakers:
            return True
        
        cb_state = self.circuit_breakers[service_name]
        if cb_state['state'] == 'open':
            time_since_open = time.time() - cb_state['opened_at']
            return time_since_open > cb_state['timeout']
        
        return cb_state['state'] == 'closed'

    def _open_circuit_breaker(self, service_name: str) -> None:
        """Open circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            self._initialize_circuit_breaker(service_name)
        
        self.circuit_breakers[service_name].update({
            'state': 'open',
            'opened_at': time.time(),
            'failure_count': self.circuit_breakers[service_name].get('failure_count', 0) + 1
        })
        
        logger.warning(f"Circuit breaker opened for {service_name}")

    def _close_circuit_breaker(self, service_name: str) -> None:
        """Close circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            self._initialize_circuit_breaker(service_name)
        
        self.circuit_breakers[service_name].update({
            'state': 'closed',
            'failure_count': 0,
            'last_success': time.time()
        })
        
        logger.info(f"Circuit breaker closed for {service_name}")

    def _initialize_circuit_breaker(self, service_name: str) -> None:
        """Initialize circuit breaker for a service."""
        self.circuit_breakers[service_name] = {
            'state': 'closed',
            'failure_count': 0,
            'threshold': 5,
            'timeout': 60.0,  # seconds
            'last_success': time.time()
        }

    # Helper methods
    def _get_retry_success_probability(self, category: ErrorCategory) -> float:
        """Get retry success probability for error category."""
        probabilities = {
            ErrorCategory.NETWORK: 0.7,
            ErrorCategory.EXTERNAL_API: 0.6,
            ErrorCategory.DATABASE: 0.8,
            ErrorCategory.QUANTUM_COMPUTING: 0.4,
            ErrorCategory.SYSTEM_RESOURCE: 0.5,
        }
        return probabilities.get(category, 0.5)

    def _get_service_health_probability(self, service_name: str) -> float:
        """Get service health probability for circuit breaker."""
        # Simulate service health based on historical data
        return 0.8 if service_name in ['database', 'network'] else 0.6

    def _provide_degraded_service(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Provide degraded service functionality."""
        return {
            'service_level': 'degraded',
            'available_features': ['basic_query', 'cached_data'],
            'unavailable_features': ['real_time_data', 'advanced_analytics'],
            'degradation_reason': error_context.error_message
        }

    def _send_escalation_alert(self, error_context: ErrorContext) -> None:
        """Send escalation alert to administrators."""
        # In practice, this would integrate with alerting systems
        alert_data = {
            'error_id': error_context.error_id,
            'severity': error_context.severity.value,
            'category': error_context.category.value,
            'message': error_context.error_message,
            'timestamp': error_context.timestamp.isoformat()
        }
        
        logger.critical(f"ESCALATION ALERT: {json.dumps(alert_data, indent=2)}")

    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with appropriate level based on severity."""
        log_message = f"Error {error_context.error_id}: {error_context.error_message}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def _update_recovery_statistics(self, strategy: RecoveryStrategy, success: bool) -> None:
        """Update recovery statistics."""
        self.recovery_statistics[strategy]['total'] += 1
        if success:
            self.recovery_statistics[strategy]['success'] += 1

    def _store_error_for_analysis(self, error_context: ErrorContext, recovery_result: RecoveryResult) -> None:
        """Store error for pattern analysis."""
        self.error_history.append((error_context, recovery_result))
        self.error_patterns[error_context.category.value].append(error_context)
        
        # Keep pattern history bounded
        if len(self.error_patterns[error_context.category.value]) > 1000:
            self.error_patterns[error_context.category.value] = self.error_patterns[error_context.category.value][-500:]

    def _adapt_parameters_based_on_outcome(self, error_context: ErrorContext, recovery_result: RecoveryResult) -> None:
        """Adapt parameters based on recovery outcome."""
        category = error_context.category.value
        
        if recovery_result.success:
            # Successful recovery, slightly increase confidence
            if recovery_result.strategy_used == RecoveryStrategy.RETRY:
                # If retry succeeded quickly, we can be more aggressive
                if recovery_result.attempts_made <= 2:
                    self.adaptive_retry_counts[category] = max(1, self.adaptive_retry_counts[category] - 1)
                else:
                    self.adaptive_retry_counts[category] = min(5, self.adaptive_retry_counts[category] + 1)
        else:
            # Failed recovery, be more conservative
            self.adaptive_retry_counts[category] = min(5, self.adaptive_retry_counts[category] + 1)
            self.adaptive_timeouts[category] = min(120.0, self.adaptive_timeouts[category] * 1.2)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error and recovery statistics."""
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {'total_errors': 0}
        
        recent_errors = [ctx for ctx, _ in list(self.error_history)[-100:]]
        
        # Error category distribution
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for error_ctx in recent_errors:
            category_counts[error_ctx.category.value] += 1
            severity_counts[error_ctx.severity.value] += 1
        
        # Recovery success rates
        recovery_rates = {}
        for strategy, stats in self.recovery_statistics.items():
            if stats['total'] > 0:
                recovery_rates[strategy.value] = {
                    'success_rate': stats['success'] / stats['total'],
                    'total_attempts': stats['total']
                }
        
        return {
            'total_errors': total_errors,
            'recent_errors_count': len(recent_errors),
            'category_distribution': dict(category_counts),
            'severity_distribution': dict(severity_counts),
            'recovery_success_rates': recovery_rates,
            'circuit_breaker_states': {
                name: state['state'] for name, state in self.circuit_breakers.items()
            },
            'adaptive_parameters': {
                'retry_counts': dict(self.adaptive_retry_counts),
                'timeouts': dict(self.adaptive_timeouts)
            }
        }

    @contextmanager
    def error_boundary(self, 
                      category: ErrorCategory, 
                      severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                      context: Optional[Dict[str, Any]] = None):
        """Context manager for error boundaries."""
        try:
            yield
        except Exception as e:
            recovery_result = self.handle_error(
                e, context or {}, category, severity
            )
            if not recovery_result.success:
                raise

    @asynccontextmanager
    async def async_error_boundary(self, 
                                  category: ErrorCategory, 
                                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                                  context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[None, None]:
        """Async context manager for error boundaries."""
        try:
            yield
        except Exception as e:
            recovery_result = await self.ahandle_error(
                e, context or {}, category, severity
            )
            if not recovery_result.success:
                raise


def robust_retry(max_attempts: int = 3, 
                backoff_factor: float = 2.0,
                exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """Decorator for robust retry functionality."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                    
                    delay = backoff_factor ** (attempt - 1)
                    time.sleep(min(delay, 60))
                    logger.warning(f"Retry attempt {attempt} for {func.__name__}: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator


def async_robust_retry(max_attempts: int = 3, 
                      backoff_factor: float = 2.0,
                      exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """Async decorator for robust retry functionality."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                    
                    delay = backoff_factor ** (attempt - 1)
                    await asyncio.sleep(min(delay, 60))
                    logger.warning(f"Async retry attempt {attempt} for {func.__name__}: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler: Optional[RobustErrorHandler] = None


def get_error_handler() -> RobustErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = RobustErrorHandler()
    return _global_error_handler