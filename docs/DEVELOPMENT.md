# Development Guide

## Quick Setup

### Prerequisites
- Python 3.8+ ([Installation Guide](https://python.org/downloads/))
- Docker & Docker Compose ([Installation Guide](https://docs.docker.com/get-docker/))
- Git ([Installation Guide](https://git-scm.com/downloads))

### Local Development
```bash
# Clone and setup
git clone <repository-url>
cd finchat-sec-qa
python -m venv venv && source venv/bin/activate
pip install -e .[dev,performance,sdk]

# Install pre-commit hooks
pre-commit install

# Verify setup
python -m pytest tests/test_foundational.py -v
```

### Docker Development
```bash
# Start services
docker-compose up -d

# Run tests in container
docker-compose exec api python -m pytest
```

## Essential Commands
- **Tests**: `python -m pytest --cov=finchat_sec_qa`
- **Type Check**: `mypy src/finchat_sec_qa`
- **Security Scan**: `bandit -r src/`
- **Format Code**: `pre-commit run --all-files`

## Architecture Overview
- **API**: FastAPI-based REST service
- **SDK**: Python client library with sync/async support
- **WebApp**: Flask-based web interface
- **Data**: EDGAR SEC filing integration

For complete setup instructions, see [DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md).