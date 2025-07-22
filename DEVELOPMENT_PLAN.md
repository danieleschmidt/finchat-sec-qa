# 🔍 Project Vision

> FinChat-SEC-QA helps financial analysts query SEC filings quickly. It fetches and indexes 10-K/10-Q documents and returns cited answers via CLI, API and web interface.

---

# 📅 12-Week Roadmap

## I1 - Security & Reliability
- **Themes**: Security, Testing
- **Goals / Epics**
  - Harden persistence: replace pickle with safer format
  - Add request validation for server endpoints
  - Set up pre-commit with ruff and bandit
- **Definition of Done**
  - No Bandit high/medium findings
  - API rejects invalid input with 400s
  - CI includes security lint and passes consistently

## I2 - Performance & Observability
- **Themes**: Performance, Observability
- **Goals / Epics**
  - Optimize QA engine index rebuilds
  - Introduce async I/O for network calls
  - Add structured logging and basic metrics
- **Definition of Done**
  - Query latency < 500ms for cached filings
  - Logs include request IDs and durations
  - Grafana dashboard shows basic metrics

## I3 - Developer UX & Expansion
- **Themes**: Developer Experience, Features
- **Goals / Epics**
  - Provide Python SDK with typed interfaces
  - Containerize web services and add Docker compose
  - Document contribution workflow and setup scripts
- **Definition of Done**
  - `pip install finchat-sec-qa[sdk]` exposes high level APIs
  - Dockerized stack starts with one command
  - CONTRIBUTING updated with onboarding steps

---

# ✅ Epic & Task Checklist

### 🔒 Increment 1: Security & Reliability
- [x] [EPIC] Eliminate pickle usage
  - [x] Replace with JSON or joblib serialization
  - [x] Migrate existing index files
- [x] [EPIC] Input validation on REST and Flask endpoints
  - [x] Define Pydantic schemas with constraints
  - [x] Add tests for invalid payloads
- [x] [EPIC] Pre-commit security enforcement
  - [x] Configure ruff + bandit hooks
  - [x] Document local setup

### ⚡️ Increment 2: Performance & Observability
- [x] [EPIC] Optimize vector index updates
  - [x] Batch add documents without refitting each time
- [x] [EPIC] Async network operations
  - [x] Use httpx/asyncio for SEC requests
  - [x] Implement AsyncEdgarClient and AsyncQueryHandler
  - [x] Update FastAPI server to use async operations
  - [ ] Measure throughput under load
- [x] [EPIC] Structured logging and metrics
  - [x] Emit request/response metrics
  - [x] Export Prometheus endpoint

### 💻 Increment 3: Developer UX & Expansion
- [ ] [EPIC] Python SDK packaging
  - [ ] Publish typed client class
  - [ ] Example notebooks
- [ ] [EPIC] Containerized deployment
  - [ ] Dockerfiles for API and webapp
  - [ ] docker-compose for local dev
- [ ] [EPIC] Improved contributor docs
  - [ ] Step-by-step setup
  - [ ] Issue templates and PR guidelines

---

# ⚠️ Risks & Mitigation
- Large SEC downloads may exceed rate limits → add exponential backoff and caching
- Pickle persistence can be exploited → migrate to safer serialization and restrict file permissions
- Long-running synchronous calls may block requests → move network I/O to async and add timeouts
- Dependency drift could introduce CVEs → enable Dependabot or weekly `pip-audit`
- Voice interface depends on system TTS → gate behind optional extra and document requirements

---

# 📊 KPIs & Metrics
- [x] >85% test coverage
- [ ] <15 min CI pipeline time
- [ ] <5% error rate on core service
- [ ] 100% secrets loaded from vault/env

---

# 👥 Ownership & Roles (Optional)
- **DevOps**: CI/CD pipeline, Docker, monitoring
- **Backend**: QA engine, API, security
- **ML/Research**: Risk analysis models
- **QA**: Test suites, acceptance criteria

