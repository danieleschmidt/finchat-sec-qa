# Changelog

All notable changes to this project will be documented in this file.

## [1.3.3] - 2025-07-21
### Added
- Accurate citation positioning with text chunking
- Configurable chunk size and overlap parameters (FINCHAT_CHUNK_SIZE, FINCHAT_CHUNK_OVERLAP)
- Intelligent text chunking that respects sentence boundaries
- Comprehensive test coverage for citation position accuracy
### Changed
- DocumentChunk now includes start_pos and end_pos for position tracking
- Citations now reference actual text positions instead of hardcoded values
- Improved TF-IDF indexing to handle empty documents gracefully

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

