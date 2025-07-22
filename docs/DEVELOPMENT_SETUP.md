# Development Setup Guide

This guide helps you set up a development environment for the FinChat SEC QA project. Follow these steps to get started with contributing to the project.

## Prerequisites

Before starting, ensure you have the following installed on your system:

### Required Software

- **Python 3.8+**: The project requires Python 3.8 or higher
- **Git**: For version control and cloning the repository
- **Docker** (optional but recommended): For containerized development
- **Docker Compose** (optional but recommended): For orchestrating services

### Operating System Support

The project supports development on:
- **Linux** (Ubuntu 20.04+, CentOS 8+)
- **macOS** (10.15+)  
- **Windows** (with WSL2 recommended)

## Local Development

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/finchat-sec-qa.git
cd finchat-sec-qa
```

### 2. Set Up Python Environment

We strongly recommend using a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

Install the project in development mode with all optional dependencies:

```bash
# Install base package in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev,performance,sdk]

# Install pre-commit hooks (recommended)
pip install pre-commit
pre-commit install
```

### 4. Configure Environment

Copy the environment template and customize it:

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your preferred settings
# Key settings for development:
# FINCHAT_LOG_LEVEL=DEBUG
# FINCHAT_HOST=localhost
# FINCHAT_PORT=8000
```

### 5. Verify Installation

Run a quick verification to ensure everything is working:

```bash
# Run basic tests
python -m pytest tests/test_foundational.py -v

# Check that imports work
python -c "import finchat_sec_qa; print('✅ Package imported successfully')"

# Check SDK availability (if installed with [sdk])
python -c "from finchat_sec_qa.sdk import FinChatClient; print('✅ SDK available')"
```

## Docker Development

For a consistent development environment, you can use Docker:

### 1. Docker-Only Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Development with Hot Reload

```bash
# Start with development configuration
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# This mounts your source code for hot reloading
# Changes to Python files will be reflected immediately
```

### 3. Docker Development Tools

```bash
# Start dev tools container for testing/debugging
docker-compose --profile dev-tools up -d dev-tools

# Run commands inside the container
docker-compose exec dev-tools python -m pytest
docker-compose exec dev-tools python scripts/load_test.py
```

## Testing

### Running Tests

The project uses pytest for testing with comprehensive coverage:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=finchat_sec_qa --cov-report=html

# Run specific test file
python -m pytest tests/test_server.py -v

# Run tests matching a pattern
python -m pytest -k "test_query" -v

# Run with debugging output
python -m pytest tests/test_server.py::test_health_check -vvs
```

### Test Categories

- **Unit Tests**: Fast tests of individual components
- **Integration Tests**: Tests of component interactions
- **End-to-End Tests**: Full workflow tests
- **Performance Tests**: Load and benchmark tests (require `[performance]` extra)

### Coverage Requirements

- Maintain **>85% test coverage** for all new code
- All public APIs must have comprehensive tests
- Critical paths (security, data processing) require **>95% coverage**

## Code Quality

### Pre-commit Hooks

The project uses pre-commit hooks for code quality:

```bash
# Install hooks (one-time setup)
pre-commit install

# Run hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
```

### Code Style

- **Python Style**: Follow PEP 8 with line length of 100 characters
- **Type Hints**: Required for all public APIs and recommended for internal code
- **Docstrings**: Required for all public functions and classes
- **Import Sorting**: Handled automatically by ruff

### Security Scanning

```bash
# Run security scan (included in pre-commit)
bandit -r src/

# Check for known vulnerabilities in dependencies
pip-audit
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write tests first (TDD approach recommended)
- Implement the feature
- Ensure tests pass
- Update documentation if needed

### 3. Quality Checks

```bash
# Run all tests
python -m pytest

# Check code coverage
python -m pytest --cov=finchat_sec_qa --cov-report=term-missing

# Run pre-commit hooks
pre-commit run --all-files

# Type checking (if mypy is configured)
mypy src/finchat_sec_qa
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat(module): add new feature with comprehensive tests

- Implement feature X with Y capability
- Add comprehensive test coverage
- Update documentation

Fixes #123"
```

## Working with Specific Components

### API Server Development

```bash
# Start FastAPI server in development mode
uvicorn finchat_sec_qa.server:app --reload --host 0.0.0.0 --port 8000

# Test API endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question":"What is revenue?","ticker":"AAPL","form_type":"10-K"}'
```

### WebApp Development

```bash
# Start Flask webapp
python -m flask --app finchat_sec_qa.webapp:app run --debug --host 0.0.0.0 --port 5000

# Access webapp
open http://localhost:5000
```

### SDK Development

```bash
# Test SDK locally
python examples/basic_usage.py

# Test async SDK
python examples/async_usage.py

# Run SDK-specific tests
python -m pytest tests/test_sdk_client.py -v
```

### Performance Testing

```bash
# Install performance dependencies
pip install -e .[performance]

# Run load tests
python scripts/load_test.py http://localhost:8000

# Run benchmarks
python scripts/benchmark.py 1.4.6 http://localhost:8000 "Local development test"

# View performance reports
cat performance_report_*.md
```

## Environment Variables

### Core Configuration

```bash
# Server Configuration
FINCHAT_HOST=localhost                    # Server host
FINCHAT_PORT=8000                        # Server port
FINCHAT_LOG_LEVEL=DEBUG                  # Logging level

# API Configuration  
FINCHAT_MAX_QUESTION_LENGTH=1000         # Max question length
FINCHAT_MAX_TICKER_LENGTH=5              # Max ticker length
FINCHAT_HTTP_TIMEOUT_SECONDS=30          # HTTP timeout

# Rate Limiting
FINCHAT_RATE_LIMIT_MAX_REQUESTS=100      # Max requests per window
FINCHAT_RATE_LIMIT_WINDOW_SECONDS=3600   # Rate limit window

# Security
FINCHAT_MIN_TOKEN_LENGTH=16              # Min API token length
FINCHAT_FAILED_ATTEMPTS_LOCKOUT_THRESHOLD=3  # Lockout threshold
```

### Development-Specific Settings

```bash
# Development toggles
FINCHAT_DEBUG=true                       # Enable debug mode
FINCHAT_VERBOSE_LOGGING=true             # Enable verbose logs

# Testing
FINCHAT_TEST_MODE=true                   # Enable test mode
FINCHAT_MOCK_EXTERNAL_APIS=true          # Mock external APIs

# Performance
FINCHAT_ENABLE_METRICS=true              # Enable Prometheus metrics
FINCHAT_CACHE_DIR=/tmp/finchat_cache     # Cache directory
```

## IDE Setup

### VS Code

Recommended extensions and settings:

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "editor.formatOnSave": true,
    "python.formatting.provider": "black"
}
```

### PyCharm

1. Open project in PyCharm
2. Configure Python interpreter to use `./venv/bin/python`
3. Enable pytest as test runner
4. Configure code style to follow PEP 8

## Common Issues and Solutions

### Import Errors

```bash
# Issue: Cannot import finchat_sec_qa
# Solution: Install in development mode
pip install -e .

# Issue: SDK not available
# Solution: Install with SDK extras
pip install -e .[sdk]
```

### Permission Errors

```bash
# Issue: Permission denied when running scripts
# Solution: Make scripts executable
chmod +x scripts/*.py

# Issue: Docker permission denied
# Solution: Add user to docker group (Linux)
sudo usermod -aG docker $USER
```

### Port Conflicts

```bash
# Issue: Port 8000 already in use
# Solution: Use different port
FINCHAT_PORT=8080 uvicorn finchat_sec_qa.server:app --reload --port 8080

# Issue: Docker ports in use
# Solution: Stop conflicting services
docker-compose down
```

### Test Failures

```bash
# Issue: Tests fail due to missing dependencies
# Solution: Install test dependencies
pip install -e .[dev,performance,sdk]

# Issue: Tests fail due to environment
# Solution: Set test environment variables
export FINCHAT_TEST_MODE=true
export FINCHAT_LOG_LEVEL=DEBUG
```

## Getting Help

- **Documentation**: Check the `docs/` directory for comprehensive guides
- **Examples**: See `examples/` for usage patterns
- **Issues**: Search existing issues on GitHub before creating new ones
- **Discussions**: Use GitHub Discussions for questions and ideas

## Next Steps

1. **Explore the Codebase**: Start with `src/finchat_sec_qa/__init__.py`
2. **Run Tests**: Ensure everything works with `pytest`
3. **Try Examples**: Run examples to understand the API
4. **Read Documentation**: Check other docs for specific areas
5. **Make Your First Contribution**: Start with documentation or tests

For more specific guidance, see:
- [API Usage Guide](API_USAGE_GUIDE.md) - Using the REST API
- [SDK Usage Guide](SDK_USAGE_GUIDE.md) - Using the Python SDK  
- [Docker Deployment](DOCKER_DEPLOYMENT.md) - Container deployment
- [Performance Testing](PERFORMANCE_TESTING.md) - Load testing and benchmarking

Happy coding! 🚀