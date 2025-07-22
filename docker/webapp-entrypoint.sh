#!/bin/bash
# Flask WebApp Entrypoint Script
# Handles initialization and configuration for containerized webapp deployment

set -e

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WEBAPP-ENTRYPOINT: $*"
}

log "Starting FinChat SEC QA Flask WebApp..."

# Set default values for environment variables if not provided
export FINCHAT_WEBAPP_HOST=${FINCHAT_WEBAPP_HOST:-0.0.0.0}
export FINCHAT_WEBAPP_PORT=${FINCHAT_WEBAPP_PORT:-5000}
export FINCHAT_LOG_LEVEL=${FINCHAT_LOG_LEVEL:-INFO}
export FLASK_ENV=${FLASK_ENV:-production}

# Create cache directory if it doesn't exist
mkdir -p /app/.cache/finchat_sec_qa

# Log configuration
log "Configuration:"
log "  Host: $FINCHAT_WEBAPP_HOST"
log "  Port: $FINCHAT_WEBAPP_PORT"
log "  Log Level: $FINCHAT_LOG_LEVEL"
log "  Flask Environment: $FLASK_ENV"
log "  Cache Directory: /app/.cache/finchat_sec_qa"

# Add src to Python path
export PYTHONPATH="/app/src:$PYTHONPATH"

# Wait for any dependencies if needed
log "Checking dependencies..."

# Verify FinChat module can be imported
python3 -c "import finchat_sec_qa; print('FinChat SEC QA module loaded successfully')" || {
    log "ERROR: Failed to import finchat_sec_qa module"
    exit 1
}

log "All dependencies verified successfully"

# If no command is provided, show help
if [ $# -eq 0 ]; then
    log "No command provided, using default: gunicorn webapp"
    exec gunicorn --bind "$FINCHAT_WEBAPP_HOST:$FINCHAT_WEBAPP_PORT" \
                  --workers 4 \
                  --timeout 120 \
                  --log-level "${FINCHAT_LOG_LEVEL,,}" \
                  finchat_sec_qa.webapp:app
fi

# Execute the provided command
log "Executing command: $*"
exec "$@"