# GitHub Workflows to Add Manually

Due to GitHub App permissions, the following workflow files need to be added manually to the `.github/workflows/` directory:

## 1. Comprehensive CI/CD Pipeline

Create `.github/workflows/comprehensive-ci.yml`:

```yaml
name: Comprehensive CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run nightly builds at 2 AM UTC
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

jobs:
  # ============================================================================
  # CODE QUALITY & SECURITY
  # ============================================================================
  
  code-quality:
    name: Code Quality & Security
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for better analysis
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -e .[dev,security,testing]
      
      - name: Code Formatting (Ruff)
        run: |
          ruff format --check src tests
          ruff check src tests
      
      - name: Type Checking (MyPy)
        run: mypy src/
      
      - name: Security Scan (Bandit)
        run: bandit -r src/ -c pyproject.toml
      
      - name: Dependency Security (Safety)
        run: safety check --json
        continue-on-error: true
      
      - name: SAST Analysis (Semgrep)
        uses: returntocorp/semgrep-action@v1
        with:
          config: auto
        env:
          SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}
        continue-on-error: true

  # ============================================================================
  # TESTING MATRIX
  # ============================================================================
  
  test-matrix:
    name: Test Suite
    runs-on: ${{ matrix.os }}
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11', '3.12']
        exclude:
          # Reduce matrix size - test fewer combinations on non-Linux
          - os: windows-latest
            python-version: '3.9'
          - os: windows-latest
            python-version: '3.10'
          - os: macos-latest
            python-version: '3.9'
          - os: macos-latest
            python-version: '3.10'
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -e .[testing,performance]
      
      - name: Unit Tests
        run: |
          pytest tests/unit/ -v \
            --cov=src \
            --cov-report=xml \
            --cov-report=term \
            --cov-fail-under=80 \
            --junit-xml=test-results-unit.xml
      
      - name: Integration Tests
        run: |
          pytest tests/integration/ -v \
            --junit-xml=test-results-integration.xml
        env:
          TEST_TIMEOUT: 60
      
      - name: Upload Coverage to Codecov
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

  # ============================================================================
  # BUILD & PACKAGE
  # ============================================================================
  
  build-and-package:
    name: Build & Package
    runs-on: ubuntu-latest
    timeout-minutes: 20
    needs: [test-matrix]
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install Build Dependencies
        run: |
          python -m pip install --upgrade pip build twine
      
      - name: Build Package
        run: python -m build
      
      - name: Check Package
        run: twine check dist/*
      
      - name: Upload Build Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: build-artifacts
          path: dist/
```

## 2. Dependency Updates

Create `.github/workflows/dependency-updates.yml`:

```yaml
name: Dependency Updates

on:
  schedule:
    # Run weekly on Mondays at 9 AM UTC
    - cron: '0 9 * * 1'
  workflow_dispatch:

permissions:
  contents: write
  pull-requests: write

jobs:
  update-dependencies:
    name: Update Python Dependencies
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip pip-tools
          pip install -e .[dev]
      
      - name: Update Dependencies
        run: |
          # Update pre-commit hooks
          pre-commit autoupdate
      
      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: "chore: update dependencies"
          title: "🔄 Weekly Dependency Updates"
          body: |
            ## Automated Dependency Updates
            
            This PR contains the weekly automated dependency updates.
            
            ### Changes:
            - Updated pre-commit hooks to latest versions
            
            ### Verification:
            - [ ] All tests pass
            - [ ] No breaking changes detected
            - [ ] Security vulnerabilities addressed
          branch: dependency-updates
          delete-branch: true
          labels: |
            dependencies
            automation
            maintenance
```

## Instructions:

1. **Copy the workflows above** into the respective files in `.github/workflows/`
2. **Commit these workflow files** to enable the CI/CD automation
3. **Configure secrets** in your GitHub repository settings:
   - `SEMGREP_APP_TOKEN` (optional - for advanced SAST analysis)
   - Any other API keys needed for your specific integrations

These workflows will provide comprehensive CI/CD automation with security scanning, testing, and dependency management once added to your repository.