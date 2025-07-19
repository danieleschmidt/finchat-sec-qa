# Technical Backlog

## WSJF Scoring Methodology
- **User/Business Value**: 1-10 (impact on users/business)
- **Time Criticality**: 1-10 (urgency/risk of delay)
- **Risk/Opportunity**: 1-10 (risk mitigation/opportunity enablement)
- **Job Size**: 1-10 (effort required, inverse for WSJF)
- **WSJF Score**: (Value + Criticality + Risk) / Job Size

## High Priority Items (WSJF > 2.0)

### 1. Replace Print Statements with Structured Logging
- **File**: `src/finchat_sec_qa/cli.py:30,34,54,56`
- **Value**: 7 | **Criticality**: 6 | **Risk**: 7 | **Size**: 3
- **WSJF**: 6.67
- **Description**: Replace print() calls with proper logging for better observability and debugging
- **Effort**: 2-3 hours
- **Risk**: Low

### 2. Improve Exception Handling in WebApp
- **File**: `src/finchat_sec_qa/webapp.py:38-39,56-57`
- **Value**: 8 | **Criticality**: 7 | **Risk**: 8 | **Size**: 4
- **WSJF**: 5.75
- **Description**: Replace bare `except Exception:` with specific exception handling and logging
- **Effort**: 3-4 hours
- **Risk**: Low

### 3. Add Input Validation Security Hardening
- **File**: `src/finchat_sec_qa/server.py:41`, `src/finchat_sec_qa/edgar_client.py:90`
- **Value**: 9 | **Criticality**: 8 | **Risk**: 9 | **Size**: 5
- **WSJF**: 5.20
- **Description**: Strengthen ticker validation and prevent URL injection vulnerabilities
- **Effort**: 4-6 hours
- **Risk**: Medium

### 4. Optimize QA Engine Bulk Operations
- **File**: `src/finchat_sec_qa/qa_engine.py:51`
- **Value**: 6 | **Criticality**: 5 | **Risk**: 7 | **Size**: 4
- **WSJF**: 4.50
- **Description**: Batch document additions without rebuilding index each time
- **Effort**: 3-5 hours
- **Risk**: Medium

### 5. Enhance Authentication Security
- **File**: `src/finchat_sec_qa/webapp.py:17,28`
- **Value**: 8 | **Criticality**: 6 | **Risk**: 8 | **Size**: 5
- **WSJF**: 4.40
- **Description**: Add token validation, rate limiting, and brute force protection
- **Effort**: 5-7 hours
- **Risk**: Medium

## Medium Priority Items (WSJF 1.0-2.0)

### 6. Improve Resource Management
- **File**: `src/finchat_sec_qa/webapp.py:24`, `src/finchat_sec_qa/server.py:27-31`
- **Value**: 5 | **Criticality**: 4 | **Risk**: 6 | **Size**: 6
- **WSJF**: 2.50
- **Description**: Replace atexit with proper context managers and async cleanup
- **Effort**: 6-8 hours
- **Risk**: Medium

### 7. Add Comprehensive Type Annotations
- **File**: `src/finchat_sec_qa/webapp.py:33,51` and multiple other files
- **Value**: 4 | **Criticality**: 3 | **Risk**: 5 | **Size**: 7
- **WSJF**: 1.71
- **Description**: Add return type annotations and improve type safety
- **Effort**: 8-10 hours
- **Risk**: Low

### 8. Fix Citation Position Accuracy
- **File**: `src/finchat_sec_qa/qa_engine.py:87`
- **Value**: 6 | **Criticality**: 4 | **Risk**: 5 | **Size**: 8
- **WSJF**: 1.88
- **Description**: Implement accurate citation positioning instead of hardcoded values
- **Effort**: 8-12 hours
- **Risk**: High

## Low Priority Items (WSJF < 1.0)

### 9. Refactor Code Duplication
- **File**: `src/finchat_sec_qa/webapp.py` and `src/finchat_sec_qa/server.py`
- **Value**: 3 | **Criticality**: 2 | **Risk**: 4 | **Size**: 10
- **WSJF**: 0.90
- **Description**: Extract common query handling logic into shared module
- **Effort**: 12-16 hours
- **Risk**: Medium

### 10. Improve Test Quality
- **File**: Multiple test files
- **Value**: 4 | **Criticality**: 3 | **Risk**: 3 | **Size**: 12
- **WSJF**: 0.83
- **Description**: Replace assert False patterns with pytest.raises(), add meaningful mocks
- **Effort**: 15-20 hours
- **Risk**: Low

## Development Plan Alignment

### Current Sprint Focus (Increment 2: Performance & Observability)
- ✅ Item #1: Structured Logging (aligns with observability goals)
- ✅ Item #4: QA Engine Optimization (aligns with performance goals)

### Next Sprint Candidates (Increment 1: Security & Reliability)
- ✅ Item #2: Exception Handling (reliability)
- ✅ Item #3: Input Validation (security)
- ✅ Item #5: Authentication Security (security)

## Risk Assessment
- **High Impact/Low Effort**: Items #1, #2 (quick wins)
- **High Risk Items**: Item #8 (citation accuracy) - requires careful testing
- **Security Critical**: Items #3, #5 - should be prioritized for next security review

## Dependencies
- Item #1 (logging) should be completed before Item #2 (exception handling)
- Item #3 (input validation) depends on understanding current API usage patterns
- Item #8 (citation accuracy) may require changes to the embedding/retrieval pipeline

Last Updated: 2025-07-19
Next Review: Weekly during sprint planning