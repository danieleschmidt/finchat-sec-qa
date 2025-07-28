# Test Fixtures

This directory contains test data and fixtures used across the test suite.

## Directory Structure

```
fixtures/
├── sec_filings/          # Sample SEC filing documents
├── api_responses/        # Mock API response data
├── config/              # Test configuration files
├── data/                # Test datasets
└── schemas/             # JSON schemas for validation
```

## Usage

Fixtures are automatically loaded by pytest through the conftest.py configuration:

```python
# In your test file
def test_edgar_parsing(sec_filing_fixture):
    result = parse_filing(sec_filing_fixture)
    assert result.company_name == "Test Company Inc."
```

## Guidelines

1. **Keep fixtures small**: Use minimal data necessary for testing
2. **Use realistic data**: Base fixtures on real SEC filing structures
3. **Version control safe**: No sensitive or proprietary information
4. **Organized by domain**: Group related fixtures in subdirectories
5. **Document purpose**: Include README files for complex fixture sets

## Adding New Fixtures

1. Create appropriate subdirectory if needed
2. Add fixture files with descriptive names
3. Update conftest.py if automatic loading is needed
4. Document the fixture purpose and usage

## Security Note

All fixture data should be:
- Anonymized or synthetic
- Free of real API keys or credentials
- Safe for public repositories
- Compliant with SEC data usage policies