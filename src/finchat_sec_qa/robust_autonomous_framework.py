"""
Robust Autonomous Framework v4.0
Comprehensive error handling, monitoring, and reliability for autonomous SDLC execution.
"""

import asyncio
import json
import time
import traceback
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
import functools
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Retry strategies"""
    NONE = "none"
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"


@dataclass
class ErrorContext:
    """Context information for errors"""
    error_id: str
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    traceback_info: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    resolved: bool = False


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 30
    success_threshold: int = 3
    timeout: int = 10


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = threading.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Success
            with self.lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.config.success_threshold:
                        self.state = CircuitBreakerState.CLOSED
                        self.failure_count = 0
                elif self.state == CircuitBreakerState.CLOSED:
                    self.failure_count = max(0, self.failure_count - 1)
            
            return result
            
        except Exception as e:
            # Failure
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
            
            raise


class HealthCheck:
    """Health check for system components"""
    
    def __init__(self, name: str, check_func: Callable, interval: int = 30):
        self.name = name
        self.check_func = check_func
        self.interval = interval
        self.last_check = 0
        self.status = "unknown"
        self.last_error = None
        self.consecutive_failures = 0
    
    async def check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(self.check_func):
                result = await self.check_func()
            else:
                result = self.check_func()
            
            check_duration = time.time() - start_time
            
            self.status = "healthy"
            self.last_check = time.time()
            self.last_error = None
            self.consecutive_failures = 0
            
            return {
                "name": self.name,
                "status": "healthy",
                "response_time": check_duration,
                "details": result if isinstance(result, dict) else {"result": result}
            }
            
        except Exception as e:
            self.status = "unhealthy"
            self.last_check = time.time()
            self.last_error = str(e)
            self.consecutive_failures += 1
            
            return {
                "name": self.name,
                "status": "unhealthy",
                "error": str(e),
                "consecutive_failures": self.consecutive_failures
            }


class RobustErrorHandler:
    """Comprehensive error handling system"""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.error_history: deque = deque(maxlen=1000)
        self.error_stats = defaultdict(int)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_strategies: Dict[str, RetryStrategy] = {}
    
    def register_circuit_breaker(self, component: str, config: CircuitBreakerConfig):
        """Register circuit breaker for component"""
        self.circuit_breakers[component] = CircuitBreaker(config)
        logger.info(f"🔌 Circuit breaker registered: {component}")
    
    def set_retry_strategy(self, component: str, strategy: RetryStrategy):
        """Set retry strategy for component"""
        self.retry_strategies[component] = strategy
    
    async def handle_with_protection(
        self,
        func: Callable,
        component: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with comprehensive error protection"""
        
        circuit_breaker = self.circuit_breakers.get(component)
        retry_strategy = self.retry_strategies.get(component, RetryStrategy.EXPONENTIAL_BACKOFF)
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if circuit_breaker:
                    return await circuit_breaker.call(func, *args, **kwargs)
                else:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                error_context = self._create_error_context(e, component, attempt)
                self._log_error(error_context)
                
                if attempt < self.max_retries:
                    delay = self._calculate_retry_delay(retry_strategy, attempt)
                    logger.warning(f"⚠️ Retrying {component} in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ {component} failed after {self.max_retries} retries")
                    break
        
        # All retries exhausted
        if last_exception:
            raise last_exception
    
    def _create_error_context(self, error: Exception, component: str, retry_count: int) -> ErrorContext:
        """Create error context for tracking"""
        error_type = type(error).__name__
        
        # Determine severity based on error type
        severity = ErrorSeverity.MEDIUM
        if error_type in ["ConnectionError", "TimeoutError", "DatabaseError"]:
            severity = ErrorSeverity.HIGH
        elif error_type in ["MemoryError", "SystemError"]:
            severity = ErrorSeverity.CRITICAL
        elif error_type in ["ValueError", "TypeError"]:
            severity = ErrorSeverity.LOW
        
        return ErrorContext(
            error_id=f"{component}_{int(time.time())}_{retry_count}",
            timestamp=time.time(),
            error_type=error_type,
            error_message=str(error),
            severity=severity,
            component=component,
            traceback_info=traceback.format_exc(),
            retry_count=retry_count
        )
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with context"""
        self.error_history.append(error_context)
        self.error_stats[error_context.error_type] += 1
        
        log_msg = f"Error in {error_context.component}: {error_context.error_message}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_msg)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def _calculate_retry_delay(self, strategy: RetryStrategy, attempt: int) -> float:
        """Calculate retry delay based on strategy"""
        if strategy == RetryStrategy.NONE:
            return 0
        elif strategy == RetryStrategy.FIXED_DELAY:
            return 1.0
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            return attempt + 1
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return min(2 ** attempt, 30)  # Cap at 30 seconds
        else:
            return 1.0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_types": dict(self.error_stats),
            "circuit_breaker_states": {
                name: cb.state.value for name, cb in self.circuit_breakers.items()
            }
        }


class HealthMonitor:
    """System health monitoring"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_checks: Dict[str, HealthCheck] = {}
        self.monitoring_active = False
        self.health_history: deque = deque(maxlen=1000)
    
    def register_health_check(self, name: str, check_func: Callable, interval: int = None):
        """Register a health check"""
        if interval is None:
            interval = self.check_interval
        
        health_check = HealthCheck(name, check_func, interval)
        self.health_checks[name] = health_check
        logger.info(f"💓 Health check registered: {name}")
    
    async def start_monitoring(self):
        """Start health monitoring"""
        self.monitoring_active = True
        logger.info("💓 Health monitoring started")
        
        while self.monitoring_active:
            await self.check_all_health()
            await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        logger.info("💓 Health monitoring stopped")
    
    async def check_all_health(self) -> Dict[str, Any]:
        """Check health of all registered components"""
        health_results = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        unhealthy_count = 0
        
        for name, health_check in self.health_checks.items():
            # Check if it's time to run this health check
            if time.time() - health_check.last_check >= health_check.interval:
                try:
                    result = await health_check.check()
                    health_results["checks"][name] = result
                    
                    if result["status"] != "healthy":
                        unhealthy_count += 1
                        
                except Exception as e:
                    health_results["checks"][name] = {
                        "name": name,
                        "status": "error",
                        "error": str(e)
                    }
                    unhealthy_count += 1
            else:
                # Use cached result
                health_results["checks"][name] = {
                    "name": name,
                    "status": health_check.status,
                    "last_check": health_check.last_check
                }
                
                if health_check.status != "healthy":
                    unhealthy_count += 1
        
        # Determine overall health
        if unhealthy_count == 0:
            health_results["overall_status"] = "healthy"
        elif unhealthy_count <= len(self.health_checks) // 2:
            health_results["overall_status"] = "degraded"
        else:
            health_results["overall_status"] = "unhealthy"
        
        health_results["unhealthy_count"] = unhealthy_count
        health_results["total_checks"] = len(self.health_checks)
        
        self.health_history.append(health_results)
        
        return health_results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        if not self.health_history:
            return {"status": "no_data", "message": "No health data available"}
        
        latest = self.health_history[-1]
        recent_checks = [h for h in self.health_history if time.time() - h["timestamp"] < 3600]
        
        # Calculate uptime percentage
        healthy_checks = sum(1 for h in recent_checks if h["overall_status"] == "healthy")
        uptime_percentage = (healthy_checks / len(recent_checks)) * 100 if recent_checks else 0
        
        return {
            "current_status": latest["overall_status"],
            "uptime_percentage": uptime_percentage,
            "total_components": latest["total_checks"],
            "unhealthy_components": latest["unhealthy_count"],
            "last_check": latest["timestamp"],
            "recent_checks_count": len(recent_checks)
        }


class RobustAutonomousFramework:
    """
    Robust Autonomous Framework
    Combines error handling, monitoring, and reliability features
    """
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.error_handler = RobustErrorHandler()
        self.health_monitor = HealthMonitor()
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Setup default circuit breakers
        self._setup_default_circuit_breakers()
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def _setup_default_circuit_breakers(self):
        """Setup default circuit breakers"""
        components = [
            "autonomous_engine",
            "enhancement_system", 
            "hypothesis_testing",
            "quality_gates",
            "documentation_generator"
        ]
        
        for component in components:
            config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                success_threshold=2,
                timeout=60
            )
            self.error_handler.register_circuit_breaker(component, config)
            self.error_handler.set_retry_strategy(component, RetryStrategy.EXPONENTIAL_BACKOFF)
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        
        async def check_memory():
            """Check memory usage"""
            import psutil
            memory = psutil.virtual_memory()
            return {
                "memory_usage_percent": memory.percent,
                "available_mb": memory.available // (1024 * 1024)
            }
        
        async def check_disk():
            """Check disk usage"""
            import psutil
            disk = psutil.disk_usage('/')
            return {
                "disk_usage_percent": (disk.used / disk.total) * 100,
                "free_gb": disk.free // (1024 ** 3)
            }
        
        def check_python_version():
            """Check Python version"""
            import sys
            return {
                "python_version": sys.version,
                "version_info": list(sys.version_info[:3])
            }
        
        # Register health checks
        try:
            self.health_monitor.register_health_check("memory", check_memory, 60)
            self.health_monitor.register_health_check("disk", check_disk, 300)
        except ImportError:
            logger.warning("psutil not available, skipping system health checks")
        
        self.health_monitor.register_health_check("python", check_python_version, 3600)
    
    async def start(self):
        """Start the robust framework"""
        logger.info(f"🛡️ Starting Robust Autonomous Framework for {self.project_name}")
        
        # Start health monitoring
        self.monitoring_task = asyncio.create_task(self.health_monitor.start_monitoring())
        
        logger.info("🛡️ Robust framework started successfully")
    
    async def stop(self):
        """Stop the robust framework"""
        logger.info("🛡️ Stopping Robust Autonomous Framework")
        
        # Stop health monitoring
        self.health_monitor.stop_monitoring()
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛡️ Robust framework stopped")
    
    async def execute_with_protection(
        self,
        func: Callable,
        component: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with full protection"""
        return await self.error_handler.handle_with_protection(func, component, *args, **kwargs)
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get complete framework status"""
        return {
            "project_name": self.project_name,
            "error_summary": self.error_handler.get_error_summary(),
            "health_summary": self.health_monitor.get_health_summary(),
            "monitoring_active": self.health_monitor.monitoring_active,
            "circuit_breakers": list(self.error_handler.circuit_breakers.keys()),
            "health_checks": list(self.health_monitor.health_checks.keys())
        }
    
    def register_custom_health_check(self, name: str, check_func: Callable, interval: int = None):
        """Register custom health check"""
        self.health_monitor.register_health_check(name, check_func, interval)
    
    def register_custom_circuit_breaker(self, component: str, config: CircuitBreakerConfig):
        """Register custom circuit breaker"""
        self.error_handler.register_circuit_breaker(component, config)
    
    @asynccontextmanager
    async def protected_operation(self, component: str):
        """Context manager for protected operations"""
        start_time = time.time()
        try:
            logger.debug(f"🔒 Starting protected operation: {component}")
            yield self
        except Exception as e:
            logger.error(f"❌ Protected operation failed: {component} - {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            logger.debug(f"🔓 Protected operation completed: {component} ({duration:.2f}s)")


# Decorator for automatic error handling
def robust_operation(component: str, max_retries: int = 3):
    """Decorator for automatic error handling"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create framework instance
            framework = getattr(wrapper, '_framework', None)
            if framework is None:
                framework = RobustAutonomousFramework("default")
                wrapper._framework = framework
            
            return await framework.execute_with_protection(func, component, *args, **kwargs)
        
        return wrapper
    return decorator


# Factory function
def create_robust_framework(project_name: str) -> RobustAutonomousFramework:
    """Create robust autonomous framework"""
    return RobustAutonomousFramework(project_name)


# Example usage and testing
async def demonstrate_robust_framework():
    """Demonstrate robust framework capabilities"""
    framework = create_robust_framework("FinChat-SEC-QA")
    
    try:
        await framework.start()
        
        # Example protected operation
        @robust_operation("test_component")
        async def risky_operation():
            import random
            if random.random() < 0.3:  # 30% chance of failure
                raise Exception("Simulated failure")
            return {"status": "success", "data": "operation completed"}
        
        # Test the operation multiple times
        for i in range(5):
            try:
                result = await risky_operation()
                logger.info(f"✅ Operation {i+1} succeeded: {result}")
            except Exception as e:
                logger.error(f"❌ Operation {i+1} failed: {str(e)}")
        
        # Get framework status
        status = framework.get_framework_status()
        logger.info(f"📊 Framework status: {status}")
        
        # Check health
        health = await framework.health_monitor.check_all_health()
        logger.info(f"💓 Health check results: {health}")
        
    finally:
        await framework.stop()


if __name__ == "__main__":
    # Example usage
    async def main():
        await demonstrate_robust_framework()
    
    asyncio.run(main())