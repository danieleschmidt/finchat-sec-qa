# Contributing

Thanks for your interest in contributing to FinChat-SEC-QA! This guide will help you get started with development and outline our contribution process.

## Quick Start

1. **Fork and Clone**: Fork the repository and clone your fork locally
2. **Setup Environment**: Follow our [Development Setup Guide](docs/DEVELOPMENT_SETUP.md)
3. **Create Branch**: Create a feature branch for your work
4. **Make Changes**: Implement your changes following our code standards
5. **Test**: Ensure all tests pass and coverage is maintained
6. **Submit PR**: Create a pull request with a clear description

## Development Setup

For detailed development setup instructions, please see our comprehensive [Development Setup Guide](docs/DEVELOPMENT_SETUP.md), which includes:

- Prerequisites and system requirements
- Local development environment setup
- Docker development workflow
- Testing procedures
- Code quality tools
- IDE configuration

### Quick Setup Summary

```bash
# Clone repository
git clone https://github.com/your-org/finchat-sec-qa.git
cd finchat-sec-qa

# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extras
pip install -e .[dev,performance,sdk]

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Verify installation
python -m pytest tests/test_foundational.py -v
```

## Testing

We maintain high test coverage standards to ensure code quality and reliability.

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage reporting
python -m pytest --cov=finchat_sec_qa --cov-report=html --cov-report=term

# Run specific test categories
python -m pytest tests/test_server.py -v
python -m pytest -k "test_query" -v
```

### Coverage Requirements

- **Minimum Coverage**: 85% for all new code
- **Critical Components**: 95% coverage for security and data processing
- **Public APIs**: 100% coverage for all public interfaces

### Test Categories

- **Unit Tests**: Fast tests of individual components
- **Integration Tests**: Tests of component interactions  
- **End-to-End Tests**: Full workflow validation
- **Performance Tests**: Load testing and benchmarks
- **Security Tests**: Vulnerability and penetration testing

## Pull Request Process

### Before Submitting

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Follow TDD Approach** (Recommended)
   - Write tests first
   - Implement feature to make tests pass
   - Refactor and optimize

3. **Quality Checks**
   ```bash
   # Run all tests
   python -m pytest
   
   # Check coverage
   python -m pytest --cov=finchat_sec_qa --cov-report=term-missing
   
   # Run pre-commit hooks
   pre-commit run --all-files
   
   # Type checking (if configured)
   mypy src/finchat_sec_qa
   ```

4. **Security Review**
   ```bash
   # Security scan
   bandit -r src/
   
   # Dependency vulnerability check
   pip-audit
   ```

### PR Submission

1. **Create Pull Request** using our [PR template](.github/pull_request_template.md)
2. **Link Related Issues** using "Fixes #123" syntax
3. **Provide Clear Description** of changes and reasoning
4. **Include Testing Evidence** showing all tests pass
5. **Add Screenshots** for UI/UX changes

### Review Process

- **Automated Checks**: All CI checks must pass
- **Code Review**: At least one maintainer review required
- **Testing**: Verify test coverage and functionality
- **Documentation**: Ensure docs are updated if needed

## Code Style

We follow strict code quality standards to maintain a consistent and secure codebase.

### Python Style Guidelines

- **PEP 8 Compliance**: Follow Python PEP 8 with 100-character line length
- **Type Hints**: Required for all public APIs, recommended for internal code
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Import Organization**: Managed automatically by ruff

### Example Code Style

```python
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class QueryHandler:
    """Handles financial question-answering queries.
    
    This class manages the complete query lifecycle from validation
    through response generation, including citation attachment.
    
    Args:
        qa_engine: The question-answering engine instance
        config: Configuration settings for query handling
    """
    
    def __init__(self, qa_engine: QAEngine, config: Config) -> None:
        self._qa_engine = qa_engine
        self._config = config
    
    async def process_query(
        self, 
        question: str, 
        ticker: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """Process a financial query and return response with citations.
        
        Args:
            question: The financial question to answer
            ticker: Company ticker symbol (e.g., 'AAPL')
            context: Optional additional context for the query
            
        Returns:
            QueryResponse containing answer and citations
            
        Raises:
            ValidationError: If question or ticker is invalid
            QueryProcessingError: If query processing fails
        """
        logger.info(f"Processing query for {ticker}: {question[:50]}...")
        
        # Implementation here
        pass
```

### Security Guidelines

- **Input Validation**: Validate and sanitize all user inputs
- **Secret Management**: Never commit secrets; use environment variables
- **Error Handling**: Don't expose sensitive information in error messages
- **Dependency Management**: Keep dependencies updated and scanned

## Workflow Guidelines

### Branch Naming

- `feature/description-of-feature`
- `bugfix/description-of-fix` 
- `security/description-of-security-fix`
- `docs/description-of-docs-change`

### Commit Messages

Follow conventional commit format:

```
type(scope): description

- Detailed explanation if needed
- References to issues: Fixes #123

Examples:
feat(sdk): add async client support for concurrent requests
fix(auth): resolve token validation timing attack vulnerability  
docs(api): update endpoint documentation with new parameters
test(qa): add comprehensive citation accuracy tests
```

### Development Workflow

1. **Issue First**: Create or reference an existing issue
2. **Branch Creation**: Create feature branch from main
3. **Development**: Follow TDD approach when possible
4. **Testing**: Ensure comprehensive test coverage
5. **Documentation**: Update relevant documentation
6. **Review**: Submit PR for code review
7. **Integration**: Merge after approval and CI success

## Component-Specific Guidelines

### API Development

- **FastAPI Standards**: Use FastAPI best practices for async endpoints
- **Validation**: Use Pydantic models for request/response validation
- **Error Handling**: Implement consistent error responses
- **Documentation**: OpenAPI documentation auto-generated

### SDK Development

- **Type Safety**: Full type hint coverage for public APIs
- **Error Handling**: Comprehensive exception hierarchy
- **Async Support**: Provide both sync and async interfaces
- **Examples**: Include usage examples for new features

### WebApp Development

- **Flask Standards**: Follow Flask application factory pattern
- **Security**: Implement CSRF protection and secure headers
- **Testing**: Include integration tests for user workflows
- **Accessibility**: Follow web accessibility guidelines

### Docker Development

- **Security**: Use non-root users and minimal base images
- **Optimization**: Multi-stage builds for smaller images
- **Health Checks**: Include proper health check endpoints
- **Documentation**: Update deployment documentation

## Issue Reporting

### Bug Reports

Use our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.yml) and include:

- **Version Information**: Exact version and environment details
- **Reproduction Steps**: Clear steps to reproduce the issue
- **Expected vs Actual**: What should happen vs what actually happens
- **Logs**: Relevant error messages and logs
- **Environment**: OS, Python version, installation method

### Feature Requests

Use our [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.yml) and include:

- **Use Case**: Specific scenario where feature would help
- **Problem Statement**: What problem does this solve
- **Proposed Solution**: Your suggested implementation approach
- **Alternatives**: Other approaches you've considered

## Security

### Reporting Security Issues

- **Private Disclosure**: Email security issues to maintainers privately
- **No Public Issues**: Don't create public issues for security vulnerabilities
- **Response Time**: We aim to respond within 48 hours

### Security Best Practices

- **Code Review**: All changes undergo security-focused review
- **Dependency Scanning**: Regular vulnerability scanning
- **Static Analysis**: Bandit security linting on all commits
- **Penetration Testing**: Regular security testing of deployed services

## Performance

### Performance Standards

- **API Response Time**: <500ms for cached queries, <2s for new queries
- **Test Suite Speed**: Full test suite should complete in <5 minutes
- **Memory Usage**: Monitor and optimize memory consumption
- **Load Testing**: Regular load testing to identify bottlenecks

### Benchmarking

```bash
# Run performance benchmarks
python scripts/benchmark.py 1.4.6 http://localhost:8000 "Development test"

# Load testing
python scripts/load_test.py http://localhost:8000

# View performance reports
cat performance_report_*.md
```

## Getting Help

- **Documentation**: Check the `docs/` directory first
- **Examples**: See `examples/` for common usage patterns
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Code Review**: Feel free to ask questions during review process

## Recognition

We value all contributions and recognize contributors in:

- **CHANGELOG.md**: Major features and fixes
- **README.md**: Significant contributors
- **Release Notes**: Notable contributions in releases

Thank you for contributing to FinChat-SEC-QA! 🚀

## Related Documentation

- [Development Setup Guide](docs/DEVELOPMENT_SETUP.md) - Complete setup instructions
- [API Usage Guide](docs/API_USAGE_GUIDE.md) - REST API documentation
- [SDK Usage Guide](docs/SDK_USAGE_GUIDE.md) - Python SDK documentation
- [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md) - Container deployment
- [Performance Testing Guide](docs/PERFORMANCE_TESTING.md) - Load testing and benchmarking
