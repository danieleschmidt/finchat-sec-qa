#!/bin/bash
# FinChat-SEC-QA Build Script
# Comprehensive build automation for development and CI/CD

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"
DOCKER_TAG_PREFIX="finchat-sec-qa"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
FinChat-SEC-QA Build Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    clean           Clean build artifacts
    install         Install dependencies
    test            Run test suite
    lint            Run linting and formatting
    security        Run security scans
    build           Build Python packages
    docker          Build Docker images
    docs            Build documentation
    release         Prepare release artifacts
    all             Run complete build pipeline
    help            Show this help message

Options:
    --dev           Include development dependencies
    --no-cache      Disable Docker build cache
    --verbose       Enable verbose output
    --skip-tests    Skip test execution
    --tag TAG       Docker image tag (default: latest)

Examples:
    $0 all                  # Full build pipeline
    $0 docker --tag v1.0.0  # Build Docker with specific tag
    $0 test --verbose       # Run tests with verbose output
    $0 build --dev          # Build with dev dependencies

EOF
}

# Parse command line arguments
COMMAND=""
DEV_MODE=false
NO_CACHE=false
VERBOSE=false
SKIP_TESTS=false
DOCKER_TAG="latest"

while [[ $# -gt 0 ]]; do
    case $1 in
        clean|install|test|lint|security|build|docker|docs|release|all|help)
            COMMAND="$1"
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --tag)
            DOCKER_TAG="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ -z "$COMMAND" ]]; then
    log_error "No command specified"
    show_help
    exit 1
fi

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Change to project root
cd "$PROJECT_ROOT"

# Build functions
clean_build() {
    log_info "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR" "$DIST_DIR"
    rm -rf .pytest_cache .mypy_cache .ruff_cache
    rm -rf htmlcov .coverage
    rm -rf src/*.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    log_success "Build artifacts cleaned"
}

install_dependencies() {
    log_info "Installing dependencies..."
    if [[ "$DEV_MODE" == "true" ]]; then
        pip install -e .[dev,testing,security,docs,voice,performance,sdk]
    else
        pip install -e .
    fi
    log_success "Dependencies installed"
}

run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests as requested"
        return 0
    fi
    
    log_info "Running test suite..."
    pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
    log_success "Tests completed"
}

run_linting() {
    log_info "Running linting and formatting..."
    ruff check src/ tests/
    ruff format --check src/ tests/
    mypy src/
    log_success "Linting completed"
}

run_security() {
    log_info "Running security scans..."
    bandit -r src/ -f json -o security_report.json || true
    safety check --json --output safety_report.json || true
    log_success "Security scans completed"
}

build_packages() {
    log_info "Building Python packages..."
    python -m build
    log_success "Python packages built"
}

build_docker() {
    log_info "Building Docker images..."
    
    local cache_args=()
    if [[ "$NO_CACHE" == "true" ]]; then
        cache_args+=("--no-cache")
    fi
    
    # Build API image
    docker build "${cache_args[@]}" \
        -f docker/Dockerfile.api \
        -t "${DOCKER_TAG_PREFIX}-api:${DOCKER_TAG}" \
        .
    
    # Build WebApp image
    docker build "${cache_args[@]}" \
        -f docker/Dockerfile.webapp \
        -t "${DOCKER_TAG_PREFIX}-webapp:${DOCKER_TAG}" \
        .
    
    log_success "Docker images built with tag: ${DOCKER_TAG}"
}

build_docs() {
    log_info "Building documentation..."
    if [[ -d "docs" ]]; then
        mkdocs build
        log_success "Documentation built"
    else
        log_warning "No docs directory found, skipping documentation build"
    fi
}

prepare_release() {
    log_info "Preparing release artifacts..."
    
    # Generate SBOM
    if command -v cyclonedx-py &> /dev/null; then
        cyclonedx-py -o SBOM.json
        log_info "SBOM generated"
    else
        log_warning "cyclonedx-py not found, skipping SBOM generation"
    fi
    
    # Create release archive
    local version
    version=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
    tar -czf "finchat-sec-qa-${version}.tar.gz" \
        --exclude-from=.dockerignore \
        --exclude=.git \
        .
    
    log_success "Release artifacts prepared"
}

# Execute command
case "$COMMAND" in
    clean)
        clean_build
        ;;
    install)
        install_dependencies
        ;;
    test)
        run_tests
        ;;
    lint)
        run_linting
        ;;
    security)
        run_security
        ;;
    build)
        build_packages
        ;;
    docker)
        build_docker
        ;;
    docs)
        build_docs
        ;;
    release)
        prepare_release
        ;;
    all)
        log_info "Running complete build pipeline..."
        clean_build
        install_dependencies
        run_linting
        run_security
        run_tests
        build_packages
        build_docker
        build_docs
        prepare_release
        log_success "Complete build pipeline finished"
        ;;
    help)
        show_help
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

log_success "Build script completed successfully"
