#!/bin/bash
set -e

echo "🚀 Setting up FinChat-SEC-QA development environment..."

# Update package lists
sudo apt-get update -qq

# Install additional system dependencies
sudo apt-get install -y -qq \
    build-essential \
    curl \
    wget \
    jq \
    tree \
    htop \
    git-lfs \
    espeak \
    portaudio19-dev \
    python3-pyaudio

# Install Python dependencies
echo "📦 Installing Python dependencies..."
python -m pip install --upgrade pip setuptools wheel
pip install -e .[dev,voice,performance,sdk,testing,security,docs]

# Install additional development tools
pip install \
    jupyterlab \
    ipython \
    pre-commit \
    commitizen \
    semantic-release \
    tox

# Setup git hooks
echo "🔧 Setting up git hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p \
    ~/.cache/finchat_sec_qa \
    logs \
    data/filings \
    data/cache \
    reports \
    benchmarks

# Setup environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please update .env with your API keys and configuration"
fi

# Install git-lfs hooks
git lfs install

# Setup Jupyter kernel
python -m ipykernel install --user --name finchat-sec-qa --display-name "FinChat SEC QA"

# Create useful aliases
echo "🔗 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# FinChat-SEC-QA Development Aliases
alias finchat-test='python -m pytest tests/ -v'
alias finchat-coverage='python -m pytest tests/ --cov=src --cov-report=html'
alias finchat-lint='ruff check src tests'
alias finchat-format='black src tests && isort src tests'
alias finchat-security='bandit -r src -q'
alias finchat-typecheck='mypy src'
alias finchat-server='uvicorn finchat_sec_qa.server:app --reload --host 0.0.0.0 --port 8000'
alias finchat-webapp='python -m flask --app finchat_sec_qa.webapp run --host 0.0.0.0 --port 5000'
alias finchat-docs='mkdocs serve --dev-addr 0.0.0.0:8080'
alias finchat-clean='find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true'
EOF

# Download sample test data (if available)
echo "📄 Setting up test data..."
if [ -d "tests/fixtures/sample_filings" ]; then
    echo "Test fixtures already exist"
else
    mkdir -p tests/fixtures/sample_filings
    echo "Test fixture directory created - add sample SEC filings here for testing"
fi

# Run initial code quality checks
echo "🔍 Running initial code quality checks..."
ruff check src tests --fix || echo "⚠️  Linting issues found - run 'finchat-lint' to see details"
black --check src tests || echo "⚠️  Formatting issues found - run 'finchat-format' to fix"

# Setup VS Code workspace settings
echo "⚙️  Configuring VS Code workspace..."
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "python.defaultInterpreterPath": "/usr/local/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests", "-v"],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".mypy_cache": true,
    ".ruff_cache": true,
    "*.egg-info": true,
    ".coverage": true,
    "htmlcov": true,
    ".tox": true
  },
  "editor.rulers": [88],
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "terminal.integrated.env.linux": {
    "PYTHONPATH": "${workspaceFolder}/src"
  }
}
EOF

# Create launch configuration for debugging
cat > .vscode/launch.json << 'EOF'
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FinChat CLI",
      "type": "python",
      "request": "launch",
      "module": "finchat_sec_qa.cli",
      "args": ["query", "What are the main risks?", "tests/fixtures/sample_filings/sample.txt"],
      "console": "integratedTerminal",
      "justMyCode": true,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    },
    {
      "name": "FastAPI Server",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["finchat_sec_qa.server:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
      "console": "integratedTerminal",
      "justMyCode": true,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    },
    {
      "name": "Flask Web App",
      "type": "python",
      "request": "launch",
      "module": "flask",
      "args": ["--app", "finchat_sec_qa.webapp", "run", "--host", "0.0.0.0", "--port", "5000"],
      "console": "integratedTerminal",
      "justMyCode": true,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src",
        "FLASK_ENV": "development"
      }
    },
    {
      "name": "Debug Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v", "-s"],
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    }
  ]
}
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🛠️  Available commands:"
echo "  finchat-test      - Run test suite"
echo "  finchat-coverage  - Run tests with coverage"
echo "  finchat-lint      - Run linting"
echo "  finchat-format    - Format code"
echo "  finchat-security  - Security scan"
echo "  finchat-server    - Start FastAPI server"
echo "  finchat-webapp    - Start Flask web app"
echo "  finchat-docs      - Start documentation server"
echo ""
echo "🔧 Next steps:"
echo "  1. Update .env with your API keys"
echo "  2. Run 'finchat-test' to verify setup"
echo "  3. Start developing! 🚀"