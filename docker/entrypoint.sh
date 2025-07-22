#!/bin/bash
# FastAPI Server Entrypoint Script
# Handles initialization and configuration for containerized deployment

set -e

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ENTRYPOINT: $*"
}

log "Starting FinChat SEC QA FastAPI Server..."

# Set default values for environment variables if not provided
export FINCHAT_HOST=${FINCHAT_HOST:-0.0.0.0}
export FINCHAT_PORT=${FINCHAT_PORT:-8000}
export FINCHAT_LOG_LEVEL=${FINCHAT_LOG_LEVEL:-INFO}

# Create cache directory if it doesn't exist
mkdir -p /app/.cache/finchat_sec_qa

# Log configuration
log "Configuration:"
log "  Host: $FINCHAT_HOST"
log "  Port: $FINCHAT_PORT"
log "  Log Level: $FINCHAT_LOG_LEVEL"
log "  Cache Directory: /app/.cache/finchat_sec_qa"

# Add src to Python path
export PYTHONPATH="/app/src:$PYTHONPATH"

# Wait for any dependencies if needed (can be extended for database connections, etc.)
log "Checking dependencies..."

# Verify FinChat module can be imported
python3 -c "import finchat_sec_qa; print('FinChat SEC QA module loaded successfully')" || {
    log "ERROR: Failed to import finchat_sec_qa module"
    exit 1
}

log "All dependencies verified successfully"

# If no command is provided, show help
if [ $# -eq 0 ]; then
    log "No command provided, using default: uvicorn server"
    exec uvicorn finchat_sec_qa.server:app --host "$FINCHAT_HOST" --port "$FINCHAT_PORT" --log-level "${FINCHAT_LOG_LEVEL,,}"
fi

# Execute the provided command
log "Executing command: $*"
exec "$@"