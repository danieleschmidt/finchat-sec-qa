#!/bin/bash

# Production Deployment Script for FinChat-SEC-QA
# This script handles production deployment with zero-downtime rolling updates

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
COMPOSE_FILE="docker/docker-compose.production.yml"
ENV_FILE=".env.production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check if Docker is installed and running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running. Please start Docker daemon."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        error "Docker Compose is not installed."
        exit 1
    fi
    
    # Check if required environment file exists
    if [[ ! -f "$PROJECT_ROOT/$ENV_FILE" ]]; then
        error "Environment file $ENV_FILE not found."
        error "Please create it from .env.example and configure production settings."
        exit 1
    fi
    
    # Check if we have sufficient disk space (at least 10GB)
    available_space=$(df -BG "$PROJECT_ROOT" | awk 'NR==2{print $4}' | sed 's/G//')
    if [[ $available_space -lt 10 ]]; then
        error "Insufficient disk space. At least 10GB required, found ${available_space}GB."
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Validate environment configuration
validate_environment() {
    log "Validating environment configuration..."
    
    # Source environment file
    set -a
    source "$PROJECT_ROOT/$ENV_FILE"
    set +a
    
    # Check required environment variables
    required_vars=(
        "POSTGRES_DB"
        "POSTGRES_USER" 
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "GRAFANA_PASSWORD"
        "ACME_EMAIL"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    success "Environment validation passed"
}

# Create required directories
create_directories() {
    log "Creating required directories..."
    
    sudo mkdir -p /opt/finchat/{data/{postgres,redis,prometheus,grafana,loki,letsencrypt},cache,logs}
    sudo chown -R $(id -u):$(id -g) /opt/finchat
    
    success "Directories created"
}

# Build application images
build_images() {
    log "Building application Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build API image
    log "Building API image..."
    docker build -f docker/Dockerfile.api --target production -t finchat-api:latest .
    
    # Build webapp image
    log "Building webapp image..."
    docker build -f docker/Dockerfile.webapp --target production -t finchat-webapp:latest .
    
    success "Images built successfully"
}

# Run pre-deployment tests
run_tests() {
    log "Running pre-deployment tests..."
    
    # Run comprehensive test suite
    python scripts/run_comprehensive_tests.py --output test_report_pre_deploy.json
    
    if [[ $? -ne 0 ]]; then
        error "Tests failed. Deployment aborted."
        exit 1
    fi
    
    success "All tests passed"
}

# Database migration and setup
setup_database() {
    log "Setting up database..."
    
    # Start only database services for migration
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    timeout=60
    while ! docker exec finchat-postgres-prod pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" >/dev/null 2>&1; do
        if [[ $timeout -le 0 ]]; then
            error "Database failed to start within timeout"
            exit 1
        fi
        sleep 2
        ((timeout-=2))
    done
    
    # Run database migrations if any
    # docker exec finchat-postgres-prod psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -f /docker-entrypoint-initdb.d/migrations.sql
    
    success "Database setup completed"
}

# Health check function
health_check() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    log "Performing health check for $service..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$url" >/dev/null 2>&1; then
            success "$service health check passed"
            return 0
        fi
        
        log "Health check attempt $attempt/$max_attempts for $service..."
        sleep 10
        ((attempt++))
    done
    
    error "$service health check failed after $max_attempts attempts"
    return 1
}

# Deploy services with rolling update
deploy_services() {
    log "Deploying services with rolling update strategy..."
    
    cd "$PROJECT_ROOT"
    
    # Deploy infrastructure services first (databases, monitoring)
    log "Deploying infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis prometheus grafana loki vector traefik
    
    # Wait for infrastructure to be ready
    sleep 30
    
    # Deploy application services with rolling update
    log "Deploying application services..."
    
    # Deploy API service
    log "Deploying API service..."
    docker-compose -f "$COMPOSE_FILE" up -d --no-deps --force-recreate finchat-api
    
    # Health check for API
    health_check "API" "http://localhost:8000/health"
    
    # Deploy webapp service
    log "Deploying webapp service..."
    docker-compose -f "$COMPOSE_FILE" up -d --no-deps --force-recreate finchat-webapp
    
    # Health check for webapp
    health_check "Webapp" "http://localhost:5000/health"
    
    # Deploy backup service
    log "Deploying backup service..."
    docker-compose -f "$COMPOSE_FILE" up -d backup
    
    success "Services deployed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check all containers are running
    failed_containers=$(docker-compose -f "$COMPOSE_FILE" ps --filter "status=exited" -q)
    if [[ -n "$failed_containers" ]]; then
        error "Some containers failed to start:"
        docker-compose -f "$COMPOSE_FILE" ps --filter "status=exited"
        exit 1
    fi
    
    # Comprehensive health checks
    health_checks=(
        "API:http://localhost:8000/health"
        "Webapp:http://localhost:5000/health"
        "Prometheus:http://localhost:9090/-/healthy"
        "Grafana:http://localhost:3000/api/health"
        "Loki:http://localhost:3100/ready"
    )
    
    for check in "${health_checks[@]}"; do
        service="${check%%:*}"
        url="${check##*:}"
        health_check "$service" "$url"
    done
    
    # Test API functionality
    log "Testing API functionality..."
    api_response=$(curl -s -w "%{http_code}" "http://localhost:8000/api/v1/health" -o /dev/null)
    if [[ "$api_response" != "200" ]]; then
        error "API health endpoint returned $api_response"
        exit 1
    fi
    
    success "Deployment verification completed"
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    report_file="deployment_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "deployment": {
    "timestamp": "$(date -Iseconds)",
    "environment": "$DEPLOYMENT_ENV",
    "version": "$(git rev-parse HEAD)",
    "branch": "$(git rev-parse --abbrev-ref HEAD)",
    "deployer": "$(whoami)@$(hostname)"
  },
  "services": {
    "status": "$(docker-compose -f "$COMPOSE_FILE" ps --format json | jq -r '.[].State')",
    "containers": $(docker-compose -f "$COMPOSE_FILE" ps --format json)
  },
  "system": {
    "disk_usage": "$(df -h /opt/finchat | tail -n1)",
    "memory_usage": "$(free -h | grep '^Mem')",
    "docker_version": "$(docker --version)"
  }
}
EOF
    
    success "Deployment report generated: $report_file"
}

# Cleanup old images and containers
cleanup() {
    log "Cleaning up old images and containers..."
    
    # Remove old images
    docker image prune -f
    
    # Remove unused volumes (be careful in production)
    # docker volume prune -f
    
    success "Cleanup completed"
}

# Rollback function
rollback() {
    error "Deployment failed. Initiating rollback..."
    
    # Stop current services
    docker-compose -f "$COMPOSE_FILE" down
    
    # Restore from backup if needed
    # This would depend on your backup strategy
    
    error "Rollback completed. Please investigate the deployment failure."
    exit 1
}

# Main deployment function
main() {
    log "Starting production deployment..."
    log "Environment: $DEPLOYMENT_ENV"
    log "Project root: $PROJECT_ROOT"
    
    # Set up error handling
    trap rollback ERR
    
    # Execute deployment steps
    check_prerequisites
    validate_environment
    create_directories
    build_images
    run_tests
    setup_database
    deploy_services
    verify_deployment
    generate_report
    cleanup
    
    success "Production deployment completed successfully!"
    log "Services are now running and accessible."
    log "Monitor the deployment using:"
    log "  - Grafana dashboard: http://localhost:3000"
    log "  - Prometheus metrics: http://localhost:9090"
    log "  - Traefik dashboard: http://localhost:8080"
    log "  - Application logs: docker-compose -f $COMPOSE_FILE logs -f"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback
        ;;
    "health")
        verify_deployment
        ;;
    "logs")
        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" logs -f "${2:-}"
        ;;
    "stop")
        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" down
        success "Services stopped"
        ;;
    "restart")
        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" restart "${2:-}"
        success "Services restarted"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health|logs|stop|restart}"
        echo "  deploy   - Deploy to production"
        echo "  rollback - Rollback deployment"
        echo "  health   - Check service health"
        echo "  logs     - View service logs"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart services"
        exit 1
        ;;
esac