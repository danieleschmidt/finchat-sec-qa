# Changelog

All notable changes to this project will be documented in this file.

## [1.4.7] - 2025-07-22
### Added
- Comprehensive contributor documentation for improved developer onboarding
- Detailed development setup guide with prerequisites, environment configuration, and troubleshooting (docs/DEVELOPMENT_SETUP.md)
- GitHub issue templates for structured bug reports and feature requests
- Pull request template with security, testing, and quality checklists
- Enhanced CONTRIBUTING.md with workflow guidelines, code standards, and component-specific guidance
- Comprehensive test suite for documentation completeness and quality validation
### Developer Experience
- Step-by-step setup instructions for local development and Docker environments
- Clear guidelines for testing procedures, code quality standards, and performance requirements
- Structured templates for consistent issue reporting and pull request submissions
- Cross-referenced documentation for improved navigation and discoverability
- Security-focused guidelines for safe development practices

## [1.4.6] - 2025-07-22
### Added
- Complete Python SDK with typed interfaces for external developer integration
- Synchronous and asynchronous client classes (FinChatClient, AsyncFinChatClient)
- Comprehensive data models with type safety (QueryResponse, Citation, RiskAnalysisResponse)
- Robust error handling with specific exception types for different error conditions
- Optional SDK dependencies via pip install finchat-sec-qa[sdk]
- Extensive SDK documentation with usage examples (docs/SDK_USAGE_GUIDE.md)
### SDK Features
- Type-safe interfaces with full IDE support and autocompletion
- Context manager support for automatic resource cleanup
- Built-in retry logic with exponential backoff for resilient operation
- Comprehensive error categorization (validation, not found, timeout, connection)
- Authentication support with API key management
- Flexible configuration with environment variable support
### Developer Experience
- Complete usage examples for both sync and async patterns (examples/)
- Batch processing examples for high-performance scenarios
- Error handling demonstrations with best practices
- Integration examples with data analysis libraries
- Comprehensive docstrings and type annotations for excellent IDE experience

## [1.4.5] - 2025-07-22
### Added
- Complete Docker containerization for local development and production deployment
- Multi-stage Dockerfiles for FastAPI server (Dockerfile.api) and Flask webapp (Dockerfile.webapp)
- docker-compose.yml for orchestrated local development environment
- Development-specific docker-compose.dev.yml with hot reload support
- Production-ready container configuration with health checks and security hardening
- Comprehensive Docker deployment documentation (docs/DOCKER_DEPLOYMENT.md)
### Container Features
- Non-root user execution for enhanced security
- Multi-stage builds for optimized image sizes
- Persistent volume management for caches and data
- Environment-based configuration following Twelve-Factor App principles
- Integrated health checks for container orchestration
- Prometheus monitoring integration (optional profile)
### Developer Experience
- Hot reload support for development workflow
- Separate dev-tools container for testing and debugging
- Complete .env.example with all configuration options
- .dockerignore optimization for faster builds
- Entrypoint scripts with proper initialization and logging

## [1.4.4] - 2025-07-22
### Added
- Comprehensive performance testing and benchmarking suite
- Load testing script (scripts/load_test.py) for concurrent API testing
- Benchmark tracking system (scripts/benchmark.py) for performance monitoring over time
- Performance documentation with optimization guidelines and expected targets
- Test coverage for all API endpoints with configurable load levels
- Historical performance comparison and regression detection
### Performance Tools
- Async load testing with configurable concurrency and request patterns
- Performance metrics collection (RPS, latency percentiles, error rates)
- Baseline comparison for tracking performance improvements
- Automated report generation with historical trends
- Support for different testing scenarios (health checks, metrics scraping, business logic)
### Developer Experience
- Optional performance testing dependencies via pip install -e .[performance]
- Comprehensive documentation in docs/PERFORMANCE_TESTING.md
- Integration-ready for CI/CD performance regression testing

## [1.4.3] - 2025-07-22
### Added
- Prometheus metrics endpoint at `/metrics` for production monitoring
- HTTP request metrics (counts, duration) with method/endpoint/status labels
- Business metrics for QA queries and risk analyses with detailed status tracking
- Service health metrics integration with existing health endpoint
- MetricsMiddleware for automatic HTTP request tracking
- Comprehensive test coverage for metrics functionality
### Production Ready
- Prometheus scraping support for observability
- Request/response metrics collection for performance monitoring
- Error rate tracking for reliability insights

## [1.4.2] - 2025-07-22  
### Changed
- Refactored validation method duplication in EdgarClient classes
- Created shared edgar_validation.py module for consistent validation logic
- Reduced code duplication by extracting common ticker, CIK, and accession number validation
- Improved maintainability and consistency across sync and async EDGAR clients
### Added
- Comprehensive test coverage for validation utilities
- New validation functions: validate_ticker, validate_cik, validate_accession_number

## [1.4.1] - 2025-07-22
### Added
- Health check endpoint at `/health` for production monitoring
- Service status reporting in health endpoint (edgar_client, qa_engine, query_handler)
- Version and timestamp information in health response
- Comprehensive test coverage for health endpoint
### Production Ready
- Container orchestration support with health checks
- Load balancer integration capabilities

## [1.4.0] - 2025-07-22
### Added
- AsyncEdgarClient for concurrent network requests using httpx
- AsyncQueryHandler for async query processing workflows  
- Async FastAPI endpoints for improved API performance
- Support for concurrent SEC filing downloads and processing
- Comprehensive test suite for async functionality
### Changed
- FastAPI server now uses async operations throughout
- Added httpx dependency for async HTTP operations
- Updated server resource management for async client cleanup
### Performance
- Significant performance improvements for concurrent API requests
- Async I/O enables better resource utilization and responsiveness

## [1.3.3] - 2025-07-21
### Added
- Accurate citation positioning with text chunking
- Configurable chunk size and overlap parameters (FINCHAT_CHUNK_SIZE, FINCHAT_CHUNK_OVERLAP)
- Intelligent text chunking that respects sentence boundaries
- Comprehensive test coverage for citation position accuracy
- Complete type annotations across all modules
- MyPy type checking integration with proper configuration
### Changed
- DocumentChunk now includes start_pos and end_pos for position tracking
- Citations now reference actual text positions instead of hardcoded values
- Improved TF-IDF indexing to handle empty documents gracefully
- Enhanced Pydantic model validation with proper string constraints
- Updated constr usage for compatibility with newer Pydantic versions

## [1.3.2] - 2025-07-27
### Added
- Engine state now persisted on shutdown for REST server and Flask webapp
- Test coverage for server persistence

## [1.3.0] - 2025-07-13
### Added
- CONTRIBUTING guidelines and CODEOWNERS file
- CLI subcommands `ingest`, `query`, and `risk`
- Simple Flask web app with token-based auth

## [1.3.1] - 2025-07-20
### Changed
- FastAPI server now stores resources on `app.state` for cleaner lifecycle
- Removed unused `httpx` dependency
- Web app reads downloaded filings via `Path.read_text`

## [1.2.0] - 2025-07-06
### Added
- Persistent vector store with save/load support
- Caching of downloaded filings via default cache directory
- REST API endpoints `/query` and `/risk`
### Changed
- FinChatAgent uses cache directory by default

## [1.1.0] - 2025-06-29
### Added
- HTTP timeout and retry logic for `EdgarClient`
- `configure_logging` helper and integrated logging
- CLI flag `--log-level` to control verbosity
- GitHub Actions workflow for lint, security scan and tests
### Changed
- Network and query operations now emit logs

## [1.0.0] - 2025-06-27
### Added
- Voice interface with optional pyttsx3 dependency
- Multi-company analysis utilities
- API usage guide with code examples

### Changed
- README updated with CLI and API instructions

