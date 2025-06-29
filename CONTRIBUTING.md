# Contributing

Thanks for your interest in FinChat-SEC-QA!

## Workflow
1. Fork the repo and create a feature branch.
2. Install dependencies with `pip install -e .` and `pip install -r requirements-dev.txt` if available.
3. Install pre-commit hooks with `pre-commit install` (requires `pip install pre-commit`).
4. Run `pre-commit run --files <changed files>` before pushing.
   Bandit only scans the `src` directory to avoid false positives in tests.
5. Run `coverage run -m pytest -q && coverage report -m` and ensure >95% coverage.
6. Submit a Pull Request describing your change.

## Code Style
- Follow PEP8 and use type hints.
- Keep functions short and add docstrings for public APIs.
- Include unit tests for new features.
