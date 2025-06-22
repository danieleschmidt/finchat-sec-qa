# Code Review

## Engineer Review
- `ruff check .` reported no issues.
- `bandit -r src -q` reported no security issues.
- No performance concerns found; code is minimal and has no heavy loops.

## Product Manager Review
- Acceptance criteria in `tests/sprint_acceptance_criteria.json` expect a basic echo helper to handle normal and null input.
- Tests in `tests/test_foundational.py` cover these cases and pass after installing the package (`pip install -e .`).
- Functionality matches the outlined requirements.

All checks passed.
