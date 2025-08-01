# FinChat-SEC-QA Makefile
# Standardized development and deployment commands

.PHONY: help install install-dev clean test test-unit test-integration test-security test-performance
.PHONY: lint format type-check security-scan coverage build docker-build docker-run
.PHONY: serve serve-dev docs deploy release pre-commit-install pre-commit-run
.PHONY: terragon-discovery terragon-continuous terragon-report terragon-setup terragon-full-cycle terragon-backlog

# Default target
help: ## Show this help message
	@echo "FinChat-SEC-QA Development Commands"
	@echo "=================================="
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e .[dev,testing,security,docs,voice,performance,sdk]
	pre-commit install
	pre-commit install --hook-type commit-msg

# Cleaning targets
clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Testing targets
test: ## Run all tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term --cov-fail-under=85

test-unit: ## Run unit tests only
	pytest tests/ -v -m "unit" --cov=src --cov-report=term

test-integration: ## Run integration tests only
	pytest tests/ -v -m "integration"

test-security: ## Run security tests only
	pytest tests/ -v -m "security"

test-performance: ## Run performance tests only
	pytest tests/ -v -m "performance" --benchmark-only

test-watch: ## Run tests in watch mode
	pytest-watch tests/ -- -v --cov=src

# Code quality targets
lint: ## Run linting (ruff)
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Format code (ruff)
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check: ## Run type checking (mypy)
	mypy src/

security-scan: ## Run security scanning
	bandit -r src/ -c pyproject.toml
	safety check
	semgrep --config=auto src/

coverage: ## Generate coverage report
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Pre-commit targets
pre-commit-install: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# Build targets
build: ## Build package
	python -m build

docker-build: ## Build Docker images
	docker build -f docker/Dockerfile.api -t finchat-sec-qa:api .
	docker build -f docker/Dockerfile.webapp -t finchat-sec-qa:webapp .

docker-run: ## Run Docker containers
	docker-compose up -d

docker-stop: ## Stop Docker containers
	docker-compose down

# Development servers
serve: ## Start FastAPI server
	uvicorn finchat_sec_qa.server:app --host 0.0.0.0 --port 8000

serve-dev: ## Start FastAPI server with auto-reload
	uvicorn finchat_sec_qa.server:app --reload --host 0.0.0.0 --port 8000

serve-webapp: ## Start Flask web application
	flask --app finchat_sec_qa.webapp run --debug

# Documentation
docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

# Release targets
release-check: ## Check if ready for release
	@echo "Running pre-release checks..."
	$(MAKE) clean
	$(MAKE) install-dev
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-scan
	@echo "✅ All checks passed! Ready for release."

release-patch: ## Release patch version
	semantic-release version --patch

release-minor: ## Release minor version
	semantic-release version --minor

release-major: ## Release major version
	semantic-release version --major

# Deployment targets
deploy-staging: ## Deploy to staging
	@echo "Deploying to staging..."
	$(MAKE) docker-build
	docker tag finchat-sec-qa:api staging.example.com/finchat-sec-qa:api
	docker tag finchat-sec-qa:webapp staging.example.com/finchat-sec-qa:webapp
	docker push staging.example.com/finchat-sec-qa:api
	docker push staging.example.com/finchat-sec-qa:webapp

deploy-prod: ## Deploy to production
	@echo "Deploying to production..."
	$(MAKE) release-check
	$(MAKE) docker-build
	docker tag finchat-sec-qa:api prod.example.com/finchat-sec-qa:api
	docker tag finchat-sec-qa:webapp prod.example.com/finchat-sec-qa:webapp
	docker push prod.example.com/finchat-sec-qa:api
	docker push prod.example.com/finchat-sec-qa:webapp

# Utility targets
benchmark: ## Run performance benchmarks
	python scripts/benchmark.py

load-test: ## Run load testing
	python scripts/load_test.py

update-deps: ## Update dependencies
	pip-compile requirements.in
	pip-compile requirements-dev.in
	pre-commit autoupdate

init-secrets: ## Initialize secrets baseline
	detect-secrets scan --baseline .secrets.baseline

# Database targets (if applicable)
db-migrate: ## Run database migrations
	@echo "No database migrations configured yet"

db-seed: ## Seed database with test data
	@echo "No database seeding configured yet"

# Monitoring targets
metrics: ## Show application metrics
	curl -s http://localhost:8000/metrics | grep -E '^(finchat|http)'

health-check: ## Check application health
	curl -f http://localhost:8000/health || exit 1
	curl -f http://localhost:5000/health || exit 1

# IDE targets
vscode-setup: ## Setup VSCode workspace
	@echo "VSCode settings already configured in .vscode/"
	@echo "Install recommended extensions for the best experience"

# Terragon Autonomous SDLC Enhancement targets
terragon-discovery: ## Run Terragon value discovery
	@echo "🔍 Running Terragon autonomous value discovery..."
	./.terragon/autonomous-execution.sh discovery

terragon-continuous: ## Run continuous discovery cycles
	@echo "🔄 Running continuous Terragon discovery..."
	./.terragon/autonomous-execution.sh continuous 3

terragon-report: ## Generate Terragon execution report
	@echo "📊 Generating Terragon execution report..."
	./.terragon/autonomous-execution.sh report

terragon-setup: ## Setup Terragon scheduled execution
	@echo "⏰ Setting up Terragon scheduled execution..."
	./.terragon/autonomous-execution.sh setup-cron
	@echo "Review .terragon/cron-setup.txt for scheduling options"

terragon-full-cycle: ## Run complete Terragon cycle (discovery + execution + learning)
	@echo "🚀 Running complete Terragon SDLC enhancement cycle..."
	./.terragon/autonomous-execution.sh full-cycle

terragon-backlog: ## View current Terragon value backlog
	@echo "📋 Current Terragon Value Backlog:"
	@echo "================================="
	@head -30 TERRAGON_VALUE_BACKLOG.md