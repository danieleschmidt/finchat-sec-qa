# Contract Testing

This directory contains contract tests to ensure API compatibility and integration reliability.

## Overview

Contract testing verifies that:
- API endpoints maintain expected request/response schemas
- External service integrations work correctly
- Breaking changes are detected before deployment
- Consumer-provider contracts are maintained

## Test Types

### API Contract Tests
- REST API endpoint validation
- Request/response schema verification
- Error handling contract validation
- Authentication contract testing

### Service Integration Contracts
- SEC EDGAR API integration contracts
- External dependency contracts
- Database schema contracts
- Cache layer contracts

## Tools and Frameworks

### Pact (Consumer-Driven Contracts)
```bash
# Install Pact
pip install pact-python

# Run contract tests
pytest tests/contract/
```

### JSON Schema Validation
```bash
# Install jsonschema
pip install jsonschema

# Validate API responses
pytest tests/contract/test_api_schemas.py
```

## Directory Structure

```
contract/
├── api/                  # API contract tests
│   ├── test_qa_api.py
│   ├── test_search_api.py
│   └── test_metrics_api.py
├── external/            # External service contracts
│   ├── test_edgar_contract.py
│   └── test_cache_contract.py
├── schemas/             # JSON schemas
│   ├── api_request.json
│   ├── api_response.json
│   └── edgar_response.json
└── pacts/               # Generated Pact files
    └── consumer-provider.json
```

## Writing Contract Tests

### API Schema Validation
```python
import pytest
import json
from jsonschema import validate
from fastapi.testclient import TestClient

def test_qa_response_schema(client: TestClient):
    response = client.post("/qa", json={"question": "test"})
    with open("schemas/qa_response.json") as f:
        schema = json.load(f)
    validate(response.json(), schema)
```

### Pact Consumer Test
```python
import pytest
from pact import Consumer, Provider

pact = Consumer('finchat-client').has_pact_with(Provider('finchat-api'))

@pytest.fixture(scope='session')
def pact_mock():
    pact.start_service()
    yield pact
    pact.stop_service()
```

## CI/CD Integration

Contract tests should run:
1. **Before deployment**: Verify no breaking changes
2. **After integration**: Confirm external services work
3. **Scheduled runs**: Detect external service changes

```yaml
# GitHub Actions example
- name: Run Contract Tests
  run: |
    pytest tests/contract/ --junitxml=contract-results.xml
    
- name: Publish Pacts
  run: |
    pact-broker publish pacts/ --consumer-app-version=$GITHUB_SHA
```

## Best Practices

1. **Version schemas**: Track API version compatibility
2. **Isolate tests**: Contract tests should not depend on external state
3. **Mock external services**: Use contract definitions, not live services
4. **Validate both ways**: Test request and response contracts
5. **Document breaking changes**: Clear migration paths for contract changes

## Monitoring

- Track contract test success rates
- Monitor external service contract stability
- Alert on schema validation failures
- Review contract coverage regularly

## Contributing

When adding new contract tests:
1. Define clear contract expectations
2. Include both positive and negative test cases
3. Document any external service dependencies
4. Update schemas when API changes occur
5. Ensure tests are deterministic and isolated