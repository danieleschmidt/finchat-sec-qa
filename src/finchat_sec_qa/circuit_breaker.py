"""Advanced circuit breaker implementation for fault tolerance and self-healing."""
import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union
from dataclasses import dataclass, field
from functools import wraps
from contextlib import asynccontextmanager

from .metrics import (
    alert_threshold_breaches_total,
    service_health,
    error_rate
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5              # Failures before opening
    success_threshold: int = 3              # Successes to close from half-open
    timeout_seconds: float = 60.0           # Time before trying half-open
    reset_timeout_seconds: float = 300.0    # Time to reset failure count
    max_concurrent_requests: int = 10        # Max concurrent requests in half-open
    slow_request_threshold: float = 10.0     # Seconds to consider request slow
    slow_request_rate_threshold: float = 0.5 # Ratio of slow requests to trigger
    expected_error_types: tuple = ()         # Error types that don't count as failures


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics and state tracking."""
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    last_state_change: float = field(default_factory=time.time)
    concurrent_requests: int = 0
    slow_requests: int = 0
    recent_response_times: list = field(default_factory=list)


class CircuitBreaker:
    """Advanced circuit breaker with adaptive behavior and self-healing."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        
        # Adaptive thresholds based on historical performance
        self._baseline_response_time = 1.0
        self._adaptive_threshold_multiplier = 1.0
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._check_and_update_state()
        
        if self.state == CircuitState.OPEN:
            self._record_rejected_request()
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        if self.state == CircuitState.HALF_OPEN:
            async with self._lock:
                if self.metrics.concurrent_requests >= self.config.max_concurrent_requests:
                    self._record_rejected_request()
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' HALF_OPEN: max concurrent requests reached"
                    )
                self.metrics.concurrent_requests += 1
        
        self.metrics.total_requests += 1
        self._request_start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        response_time = time.time() - self._request_start_time
        
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.metrics.concurrent_requests -= 1
            
            # Update response time metrics
            self.metrics.recent_response_times.append(response_time)
            if len(self.metrics.recent_response_times) > 50:  # Keep last 50 requests
                self.metrics.recent_response_times.pop(0)
            
            if exc_type is None:
                await self._record_success(response_time)
            elif exc_type in self.config.expected_error_types:
                # Expected errors don't count as failures
                await self._record_expected_error(exc_type, response_time)
            else:
                await self._record_failure(exc_type, response_time)
    
    async def _check_and_update_state(self):
        """Check and update circuit breaker state based on current conditions."""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # Check if we should transition to HALF_OPEN
            if current_time - self.metrics.last_failure_time >= self.config.timeout_seconds:
                await self._transition_to_half_open()
        
        elif self.state == CircuitState.HALF_OPEN:
            # Check if we should transition to CLOSED or back to OPEN
            if self.metrics.consecutive_successes >= self.config.success_threshold:
                await self._transition_to_closed()
            elif self.metrics.consecutive_failures > 0:
                await self._transition_to_open()
        
        elif self.state == CircuitState.CLOSED:
            # Check if we should open due to failures
            if self.metrics.consecutive_failures >= self.config.failure_threshold:
                await self._transition_to_open()
            # Check slow request rate
            elif self._should_open_due_to_slow_requests():
                await self._transition_to_open("slow_requests")
            # Reset failure count after reset timeout
            elif (current_time - self.metrics.last_failure_time >= self.config.reset_timeout_seconds and 
                  self.metrics.failure_count > 0):
                self.metrics.failure_count = 0
                self.metrics.consecutive_failures = 0
                logger.info(f"Circuit breaker '{self.name}' reset failure count after timeout")
    
    def _should_open_due_to_slow_requests(self) -> bool:
        """Check if circuit should open due to high rate of slow requests."""
        if len(self.metrics.recent_response_times) < 10:  # Need sufficient samples
            return False
        
        recent_times = self.metrics.recent_response_times[-20:]  # Last 20 requests
        slow_requests = sum(1 for t in recent_times if t > self.config.slow_request_threshold)
        slow_rate = slow_requests / len(recent_times)
        
        return slow_rate >= self.config.slow_request_rate_threshold
    
    async def _record_success(self, response_time: float):
        """Record a successful request."""
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0
        self.metrics.last_success_time = time.time()
        
        # Update adaptive baseline
        if response_time < self._baseline_response_time * 2:
            self._baseline_response_time = (self._baseline_response_time * 0.9 + response_time * 0.1)
        
        # Update service health metric
        service_health.labels(service=self.name, check_type='circuit_breaker').set(1.0)
        
        logger.debug(f"Circuit breaker '{self.name}' recorded success (response_time: {response_time:.3f}s)")
    
    async def _record_failure(self, exc_type: type, response_time: float):
        """Record a failed request."""
        self.metrics.failure_count += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.last_failure_time = time.time()
        
        # Update error rate metric
        error_rate.labels(service=self.name, error_type=exc_type.__name__ if exc_type else 'unknown').set(
            self.metrics.failure_count / max(self.metrics.total_requests, 1)
        )
        
        # Update service health
        health_value = max(0.0, 1.0 - (self.metrics.consecutive_failures / self.config.failure_threshold))
        service_health.labels(service=self.name, check_type='circuit_breaker').set(health_value)
        
        logger.warning(f"Circuit breaker '{self.name}' recorded failure: {exc_type.__name__ if exc_type else 'Unknown'} "
                      f"(consecutive: {self.metrics.consecutive_failures}, total: {self.metrics.failure_count})")
    
    async def _record_expected_error(self, exc_type: type, response_time: float):
        """Record an expected error (doesn't count as failure)."""
        logger.debug(f"Circuit breaker '{self.name}' recorded expected error: {exc_type.__name__}")
    
    def _record_rejected_request(self):
        """Record a request that was rejected due to circuit breaker state."""
        alert_threshold_breaches_total.labels(
            alert_name='circuit_breaker_rejection',
            severity='warning',
            service=self.name
        ).inc()
        
        logger.warning(f"Circuit breaker '{self.name}' rejected request (state: {self.state.value})")
    
    async def _transition_to_open(self, reason: str = "failures"):
        """Transition circuit breaker to OPEN state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.metrics.last_state_change = time.time()
        
        # Update service health
        service_health.labels(service=self.name, check_type='circuit_breaker').set(0.0)
        
        # Record alert
        alert_threshold_breaches_total.labels(
            alert_name='circuit_breaker_opened',
            severity='critical',
            service=self.name
        ).inc()
        
        logger.error(f"Circuit breaker '{self.name}' OPENED (reason: {reason}, "
                    f"failures: {self.metrics.consecutive_failures})")
    
    async def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.metrics.last_state_change = time.time()
        self.metrics.concurrent_requests = 0
        
        # Reset consecutive counters for fresh evaluation
        self.metrics.consecutive_successes = 0
        self.metrics.consecutive_failures = 0
        
        service_health.labels(service=self.name, check_type='circuit_breaker').set(0.5)
        
        logger.warning(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
    
    async def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.metrics.last_state_change = time.time()
        
        # Reset metrics for fresh start
        self.metrics.consecutive_failures = 0
        self.metrics.consecutive_successes = 0
        
        service_health.labels(service=self.name, check_type='circuit_breaker').set(1.0)
        
        logger.info(f"Circuit breaker '{self.name}' CLOSED (recovered)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status and metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.metrics.failure_count,
            "success_count": self.metrics.consecutive_successes,
            "total_requests": self.metrics.total_requests,
            "consecutive_failures": self.metrics.consecutive_failures,
            "last_failure_time": self.metrics.last_failure_time,
            "last_success_time": self.metrics.last_success_time,
            "state_duration": time.time() - self.metrics.last_state_change,
            "concurrent_requests": self.metrics.concurrent_requests,
            "average_response_time": (
                sum(self.metrics.recent_response_times) / len(self.metrics.recent_response_times)
                if self.metrics.recent_response_times else 0
            ),
            "baseline_response_time": self._baseline_response_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "slow_request_threshold": self.config.slow_request_threshold
            }
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """Manages multiple circuit breakers with global policies."""
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._global_config = CircuitBreakerConfig()
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker for a service."""
        if name not in self._circuit_breakers:
            effective_config = config or self._global_config
            self._circuit_breakers[name] = CircuitBreaker(name, effective_config)
            logger.info(f"Created new circuit breaker for service: {name}")
        
        return self._circuit_breakers[name]
    
    def set_global_config(self, config: CircuitBreakerConfig):
        """Set global configuration for new circuit breakers."""
        self._global_config = config
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all circuit breakers."""
        return {name: cb.get_status() for name, cb in self._circuit_breakers.items()}
    
    async def reset_circuit_breaker(self, name: str) -> bool:
        """Manually reset a circuit breaker."""
        if name in self._circuit_breakers:
            cb = self._circuit_breakers[name]
            async with cb._lock:
                await cb._transition_to_closed()
                cb.metrics.failure_count = 0
                cb.metrics.consecutive_failures = 0
                logger.info(f"Manually reset circuit breaker: {name}")
                return True
        return False
    
    def remove_circuit_breaker(self, name: str) -> bool:
        """Remove a circuit breaker."""
        if name in self._circuit_breakers:
            del self._circuit_breakers[name]
            logger.info(f"Removed circuit breaker: {name}")
            return True
        return False


# Global circuit breaker manager
_circuit_breaker_manager = CircuitBreakerManager()


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to apply circuit breaker to a function."""
    def decorator(func):
        cb = _circuit_breaker_manager.get_circuit_breaker(name, config)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with cb:
                return await func(*args, **kwargs)
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to use asyncio
            async def async_func():
                async with cb:
                    return func(*args, **kwargs)
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an event loop, create a task
                    future = asyncio.create_task(async_func())
                    return loop.run_until_complete(future)
                else:
                    return loop.run_until_complete(async_func())
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(async_func())
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@asynccontextmanager
async def circuit_breaker_context(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Context manager for circuit breaker protection."""
    cb = _circuit_breaker_manager.get_circuit_breaker(name, config)
    async with cb:
        yield cb


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager."""
    return _circuit_breaker_manager


# Convenience functions
async def get_all_circuit_breaker_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers."""
    return await _circuit_breaker_manager.health_check_all()


async def reset_circuit_breaker(name: str) -> bool:
    """Reset a specific circuit breaker."""
    return await _circuit_breaker_manager.reset_circuit_breaker(name)