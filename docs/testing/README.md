# Testing Guide for FinChat-SEC-QA

This guide provides comprehensive information about the testing strategy, tools, and practices for the FinChat-SEC-QA project.

## Testing Philosophy

Our testing approach follows the testing pyramid with emphasis on:
- **Fast feedback**: Quick unit tests provide immediate feedback
- **Confidence**: Integration tests ensure components work together
- **Reality**: E2E tests validate real user scenarios
- **Performance**: Load tests ensure system scalability
- **Security**: Security tests protect against vulnerabilities

## Test Types and Structure

### 1. Unit Tests (`tests/unit/`)
- **Purpose**: Test individual functions and classes in isolation
- **Coverage**: >90% code coverage target
- **Speed**: <100ms per test
- **Tools**: pytest, unittest.mock
- **Run**: `pytest tests/unit/`

### 2. Integration Tests (`tests/`)
- **Purpose**: Test component interactions and external integrations
- **Coverage**: Critical user flows and API endpoints
- **Tools**: pytest, TestClient, httpx
- **Run**: `pytest tests/ -k "not unit and not e2e"`

### 3. Contract Tests (`tests/contract/`)
- **Purpose**: Ensure API compatibility and service contracts
- **Tools**: jsonschema, pact-python
- **Run**: `pytest tests/contract/`

### 4. Performance Tests (`tests/performance/`)
- **Purpose**: Validate system performance under load
- **Tools**: k6, pytest-benchmark
- **Run**: `k6 run tests/performance/k6/load-test.js`

### 5. Security Tests
- **Purpose**: Identify security vulnerabilities
- **Tools**: bandit, safety, semgrep
- **Run**: `bandit -r src/` or `pytest tests/ -m security`

### 6. Mutation Tests (`tests/mutation/`)
- **Purpose**: Test the quality of our test suite
- **Tools**: mutmut, cosmic-ray
- **Run**: `mutmut run`

## Running Tests

### Quick Test Commands

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m performance   # Performance tests only
pytest -m security      # Security tests only

# Run tests in parallel
pytest -n auto

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_qa_engine.py

# Run specific test
pytest tests/test_qa_engine.py::test_question_answering
```

### Environment-Specific Testing

```bash
# Local development
export ENVIRONMENT=test
pytest

# CI/CD pipeline
pytest --junitxml=test-results.xml --cov-report=xml

# Docker testing
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## Test Configuration

### pytest Configuration (`pyproject.toml`)

Our pytest configuration includes:
- Test discovery patterns
- Coverage settings
- Marker definitions
- Plugin configurations

### Test Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests  
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.security`: Security tests
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.external`: Tests requiring external services

### Fixtures (`tests/conftest.py`)

Common test fixtures provide:
- Mock external services (OpenAI, SEC EDGAR)
- Test data factories
- Temporary directories
- Database setup/teardown
- API clients

## Writing Good Tests

### Unit Test Best Practices

```python
def test_qa_engine_processes_question():
    # Arrange
    engine = FinancialQAEngine()
    question = "What are the risk factors?"
    mock_context = ["Risk factor 1", "Risk factor 2"]
    
    # Act
    result = engine.answer_question(question, mock_context)
    
    # Assert
    assert result.answer is not None
    assert len(result.citations) > 0
    assert result.confidence > 0.5
```

### Integration Test Best Practices

```python
async def test_api_qa_endpoint(async_api_client):
    # Test realistic API usage
    response = await async_api_client.post(
        "/qa",
        json={
            "question": "What are the main revenue streams?",
            "ticker": "AAPL",
            "form_type": "10-K"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
```

### Test Data Management

```python
# Use factories for test data
@pytest.fixture
def sample_filing(filing_factory):
    return filing_factory(
        ticker="MSFT",
        form_type="10-Q",
        content="Sample quarterly report content..."
    )

# Use realistic but anonymized data
def test_citation_extraction(sample_filing):
    citations = extract_citations(sample_filing.content)
    assert len(citations) > 0
```

## Performance Testing

### K6 Load Testing

```bash
# Run smoke test (basic functionality)
k6 run tests/performance/k6/smoke-test.js

# Run load test (normal expected load)
k6 run tests/performance/k6/load-test.js

# Run stress test (find breaking points)
k6 run tests/performance/k6/stress-test.js
```

### Performance Targets

- **API Response Time**: p95 < 500ms under normal load
- **Throughput**: Handle 100 concurrent users
- **Error Rate**: < 1% under normal load, < 5% under stress
- **Memory Usage**: < 500MB per worker process

### Benchmark Testing

```python
def test_qa_engine_performance(benchmark):
    engine = FinancialQAEngine()
    question = "What are the key metrics?"
    context = ["Revenue: $100M", "Profit: $20M"]
    
    result = benchmark(engine.answer_question, question, context)
    assert benchmark.stats.mean < 1.0  # Less than 1 second
```

## Security Testing

### Static Analysis

```bash
# Security vulnerability scanning
bandit -r src/ -f json -o security-report.json

# Dependency vulnerability checking
safety check

# Code quality and security
semgrep --config=auto src/
```

### Dynamic Security Testing

```python
@pytest.mark.security
def test_sql_injection_protection(api_client):
    # Test SQL injection attempts
    malicious_input = "'; DROP TABLE users; --"
    response = api_client.post("/qa", json={"question": malicious_input})
    
    # Should not cause server error
    assert response.status_code in [200, 400]
    assert "error" not in response.json().get("answer", "").lower()
```

## Continuous Integration

### GitHub Actions Integration

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e .[dev,testing]
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Test Automation

- **Pre-commit hooks**: Run basic tests before commits
- **PR validation**: Full test suite on pull requests
- **Nightly runs**: Comprehensive testing including performance
- **Release testing**: Full regression suite before releases

## Coverage Requirements

### Coverage Targets

- **Overall**: >85% line coverage
- **Critical paths**: >95% coverage
- **Security functions**: 100% coverage
- **Public APIs**: 100% coverage

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Generate terminal coverage report
pytest --cov=src --cov-report=term-missing

# Check coverage thresholds
pytest --cov=src --cov-fail-under=85
```

## Debugging Tests

### Common Debugging Techniques

```bash
# Run with detailed output
pytest -vvv

# Stop on first failure
pytest -x

# Enter debugger on failure
pytest --pdb

# Run specific test with print statements
pytest -s tests/test_specific.py::test_function

# Show local variables in traceback
pytest -l
```

### Test Isolation

- Use fixtures for setup/teardown
- Mock external dependencies
- Clean up resources after tests
- Use temporary directories for file operations

## Test Maintenance

### Regular Maintenance Tasks

1. **Update test dependencies**: Keep testing tools current
2. **Review test coverage**: Identify gaps in coverage
3. **Performance baseline updates**: Update performance targets
4. **Flaky test identification**: Fix or remove unreliable tests
5. **Test data refresh**: Update fixtures with realistic data

### Test Quality Metrics

- **Test execution time**: Monitor and optimize slow tests
- **Test reliability**: Track and fix flaky tests
- **Coverage trends**: Ensure coverage doesn't decrease
- **Test maintainability**: Refactor complex or duplicated tests

## Contributing to Tests

### Adding New Tests

1. **Choose appropriate test type**: Unit, integration, or E2E
2. **Follow naming conventions**: `test_feature_scenario`
3. **Use descriptive test names**: What behavior is being tested
4. **Include edge cases**: Test boundary conditions and error cases
5. **Mock external dependencies**: Keep tests fast and reliable

### Test Review Checklist

- [ ] Test names clearly describe what is being tested
- [ ] Tests are isolated and don't depend on each other
- [ ] External dependencies are properly mocked
- [ ] Edge cases and error conditions are covered
- [ ] Performance impact is considered
- [ ] Security implications are tested
- [ ] Documentation is updated if needed

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [K6 performance testing](https://k6.io/docs/)
- [Testing best practices](https://testing.googleblog.com/)
- [Python testing tools](https://github.com/vintasoftware/python-testing)
- [Security testing guide](https://owasp.org/www-project-web-security-testing-guide/)