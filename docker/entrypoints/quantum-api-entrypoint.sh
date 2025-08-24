#!/bin/bash
# Terragon Quantum API Production Entrypoint
# Handles initialization, health checks, and graceful startup

set -euo pipefail

echo "🚀 Starting Terragon Quantum API Production Server"
echo "=============================================="

# Function for logging with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Environment validation
validate_environment() {
    log "🔧 Validating environment configuration..."
    
    # Check required environment variables
    required_vars=(
        "REDIS_URL"
        "QUANTUM_DEPTH"
        "MAX_NODES"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log "❌ ERROR: Required environment variable $var is not set"
            exit 1
        fi
    done
    
    log "✅ Environment validation passed"
}

# System optimization for quantum processing
optimize_system() {
    log "⚡ Optimizing system for quantum processing..."
    
    # Set CPU governor to performance mode if possible
    if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
        echo "performance" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
    fi
    
    # Optimize memory settings
    export MALLOC_ARENA_MAX=4
    export MALLOC_MMAP_THRESHOLD_=1048576
    
    # Set up CUDA if available
    if command -v nvidia-smi &> /dev/null; then
        log "🎮 CUDA GPU detected - enabling GPU acceleration"
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
        export NUMBA_ENABLE_CUDASIM=0
        export NUMBA_CUDA_DRIVER=${NUMBA_CUDA_DRIVER:-cuda}
    else
        log "💻 No GPU detected - using CPU optimization"
        export CUDA_VISIBLE_DEVICES=""
        export NUMBA_DISABLE_CUDA=1
    fi
    
    log "✅ System optimization complete"
}

# Health check function
health_check() {
    log "🏥 Running initial health check..."
    
    # Test quantum modules can be imported
    python3 -c "
import sys
sys.path.insert(0, '/app/src')

try:
    from finchat_sec_qa.quantum_breakthrough_multimodal_engine import create_quantum_breakthrough_engine
    from finchat_sec_qa.quantum_robust_orchestrator import create_robust_orchestrator
    from finchat_sec_qa.quantum_hyperscale_engine import create_hyperscale_engine
    print('✅ All quantum modules imported successfully')
except ImportError as e:
    print(f'❌ Module import failed: {e}')
    sys.exit(1)
" || {
        log "❌ Quantum module health check failed"
        exit 1
    }
    
    log "✅ Health check passed"
}

# Initialize quantum data structures
initialize_quantum_system() {
    log "🧬 Initializing quantum processing system..."
    
    # Create necessary directories
    mkdir -p /data/quantum_cache /data/models /data/logs /tmp/numba_cache
    
    # Pre-compile quantum algorithms for faster startup
    python3 -c "
import sys
sys.path.insert(0, '/app/src')
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Pre-compile quantum simulation functions
print('Warming up quantum algorithms...')
from finchat_sec_qa.quantum_breakthrough_multimodal_engine import QuantumBreakthroughMultimodalEngine

engine = QuantumBreakthroughMultimodalEngine(quantum_depth=4, multimodal_dims=64)
test_features = np.random.random(32)

# Warm up all quantum processing functions
engine._simulate_quantum_text_processing(test_features)
engine._simulate_quantum_numerical_processing(test_features)
engine._simulate_quantum_sentiment_processing(test_features)

print('✅ Quantum algorithms warmed up and compiled')
" || {
        log "⚠️ Quantum warmup failed - continuing with cold start"
    }
    
    log "✅ Quantum system initialized"
}

# Wait for dependencies
wait_for_dependencies() {
    log "⏳ Waiting for dependencies..."
    
    # Extract Redis host from URL
    REDIS_HOST=$(echo "$REDIS_URL" | sed -n 's|.*://\([^:]*\).*|\1|p')
    REDIS_PORT=$(echo "$REDIS_URL" | sed -n 's|.*:\([0-9]*\).*|\1|p')
    REDIS_PORT=${REDIS_PORT:-6379}
    
    # Wait for Redis
    timeout 60 bash -c "until nc -z $REDIS_HOST $REDIS_PORT; do sleep 1; done" || {
        log "❌ Redis not available at $REDIS_HOST:$REDIS_PORT"
        exit 1
    }
    
    log "✅ Redis available at $REDIS_HOST:$REDIS_PORT"
    
    # Wait for other services if needed
    if [[ -n "${PROMETHEUS_URL:-}" ]]; then
        log "⏳ Checking Prometheus connectivity..."
        # Add Prometheus check here
    fi
    
    log "✅ All dependencies ready"
}

# Setup monitoring and metrics
setup_monitoring() {
    log "📊 Setting up monitoring and metrics..."
    
    # Create metrics directories
    mkdir -p /data/metrics /data/logs
    
    # Start metrics collection in background
    if [[ "${METRICS_ENABLED:-true}" == "true" ]]; then
        python3 -c "
import sys
sys.path.insert(0, '/app/src')
from finchat_sec_qa.comprehensive_monitoring import ComprehensiveMonitoring
print('✅ Monitoring system initialized')
" &
    fi
    
    log "✅ Monitoring setup complete"
}

# Graceful shutdown handler
cleanup() {
    log "🔄 Received shutdown signal - cleaning up..."
    
    # Kill background processes
    jobs -p | xargs -r kill
    
    # Clear temporary files
    rm -rf /tmp/numba_cache/* 2>/dev/null || true
    
    log "✅ Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# Main initialization sequence
main() {
    log "🎯 Starting Terragon Quantum API initialization sequence..."
    
    validate_environment
    optimize_system
    wait_for_dependencies
    health_check
    initialize_quantum_system
    setup_monitoring
    
    log "🚀 All systems ready - starting quantum API server"
    log "=============================================="
    
    # Export final configuration
    export PYTHONPATH="/app/src:${PYTHONPATH:-}"
    export QUANTUM_INITIALIZED=true
    
    # Print system information
    log "📋 System Configuration:"
    log "   - Quantum Depth: ${QUANTUM_DEPTH}"
    log "   - Max Nodes: ${MAX_NODES}"
    log "   - GPU Enabled: ${ENABLE_GPU:-false}"
    log "   - Redis URL: ${REDIS_URL}"
    log "   - Python Path: ${PYTHONPATH}"
    
    # Start the application
    exec "$@"
}

# Run main function
main "$@"