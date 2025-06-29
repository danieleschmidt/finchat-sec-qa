# Contributing

Thanks for your interest in FinChat-SEC-QA!

## Workflow
1. Fork the repo and create a feature branch.
2. Install dependencies with `pip install -e .` and `pip install -r requirements-dev.txt` if available.
3. Run `ruff check src tests --fix` and `bandit -r src -q`.
4. Run `coverage run -m pytest -q && coverage report -m` and ensure >95% coverage.
5. Submit a Pull Request describing your change.

## Code Style
- Follow PEP8 and use type hints.
- Keep functions short and add docstrings for public APIs.
- Include unit tests for new features.
