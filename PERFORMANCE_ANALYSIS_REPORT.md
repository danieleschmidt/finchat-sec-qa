# Performance Analysis Report - FinChat SEC QA

## Executive Summary

This report identifies performance bottlenecks and optimization opportunities in the FinChat SEC QA codebase. The analysis reveals several high-impact areas for performance improvement, with WSJF (Weighted Shortest Job First) estimations provided for prioritization.

## Critical Performance Issues Found

### 1. **N+1 Query Pattern in Company Ticker Lookup** [HIGH IMPACT]
**Location**: `/src/finchat_sec_qa/edgar_client.py` (lines 79-89)

**Issue**: The `ticker_to_cik()` method downloads the entire company tickers JSON file (several MB) and performs a linear search through all entries every time it's called.

```python
def ticker_to_cik(self, ticker: str) -> str:
    mapping_url = urljoin(self.BASE_URL, "/files/company_tickers.json")
    data = self._get(mapping_url).json()  # Downloads entire file
    for entry in data.values():  # Linear search O(n)
        if entry["ticker"].upper() == ticker:
            cik = str(entry["cik_str"])
            return self._validate_cik(cik)
```

**Performance Impact**: 
- Downloads ~5MB JSON file on every ticker lookup
- Linear search through ~10,000+ companies
- Network latency + parsing time on every API call

**Optimization**: Implement caching with TTL:
```python
class EdgarClient:
    def __init__(self, ...):
        self._ticker_cache = {}
        self._ticker_cache_expiry = None
        self._ticker_cache_ttl = 3600  # 1 hour
```

**WSJF Score**: 8/10 (High value, low effort)

### 2. **Missing Database Indexes for QA Engine** [HIGH IMPACT]
**Location**: `/src/finchat_sec_qa/qa_engine.py`

**Issue**: The QA engine rebuilds the entire TF-IDF matrix on every document addition when not in bulk mode:
```python
def add_document(self, doc_id: str, text: str) -> None:
    # ... add chunks ...
    if not self._bulk_mode:
        self._rebuild_index()  # Rebuilds entire index!
```

**Performance Impact**:
- O(n²) complexity when adding multiple documents
- CPU-intensive matrix operations repeated unnecessarily
- No incremental index updates

**Optimization**: Implement incremental indexing or batch operations by default

**WSJF Score**: 9/10 (Very high value, moderate effort)

### 3. **Synchronous File I/O in Hot Path** [MEDIUM IMPACT]
**Location**: `/src/finchat_sec_qa/query_handler.py` (line 131)

**Issue**: File reading is synchronous even in async handler:
```python
def _download_and_read_filing(self, filing: FilingMetadata) -> str:
    path = self.client.download_filing(filing)
    filing_text = safe_read_file(path)  # Blocking I/O
```

**Performance Impact**:
- Blocks event loop in async contexts
- No parallel file operations
- Thread pool overhead in async version

**Optimization**: Use `aiofiles` for truly async file I/O

**WSJF Score**: 6/10 (Medium value, low effort)

### 4. **Inefficient JSON Parsing in Multiple Locations** [MEDIUM IMPACT]
**Locations**: Multiple files

**Issue**: JSON responses are parsed multiple times:
- Edgar client: `response.json()` called repeatedly
- No streaming JSON parsing for large responses
- Full document loaded into memory

**Performance Impact**:
- Memory spikes for large filings
- CPU overhead from repeated parsing
- No incremental processing

**WSJF Score**: 5/10 (Medium value, medium effort)

### 5. **Missing Connection Pooling for Redis** [LOW-MEDIUM IMPACT]
**Location**: `/src/finchat_sec_qa/rate_limiting.py`

**Issue**: Redis client creates new connections without pooling:
```python
self.redis_client = redis.Redis.from_url(
    self.redis_url,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
    retry_on_timeout=True
)
```

**Performance Impact**:
- Connection overhead on each rate limit check
- No connection reuse
- Latency in high-traffic scenarios

**Optimization**: Use connection pooling:
```python
pool = redis.ConnectionPool.from_url(self.redis_url, max_connections=50)
self.redis_client = redis.Redis(connection_pool=pool)
```

**WSJF Score**: 4/10 (Low value, very low effort)

### 6. **Unbounded In-Memory Caches** [MEDIUM IMPACT]
**Locations**: Multiple components

**Issues Found**:
- CSRF tokens accumulate without bounds (`webapp.py`)
- Rate limiter fallback storage grows indefinitely
- No LRU eviction or size limits

**Performance Impact**:
- Memory leaks in long-running processes
- Potential OOM errors
- GC pressure

**WSJF Score**: 7/10 (High value for stability, low effort)

### 7. **Missing Async Operations in Multi-Company Analysis** [HIGH IMPACT]
**Location**: `/src/finchat_sec_qa/multi_company.py`

**Issue**: Sequential processing of companies:
```python
for doc_id in documents.keys():
    logger.debug("Querying document %s", doc_id)
    answer, _ = engine.answer_with_citations(question)
    results.append(CompanyAnswer(doc_id=doc_id, answer=answer))
```

**Performance Impact**:
- No parallelization
- Linear time complexity O(n)
- Could be O(1) with parallel processing

**Optimization**: Use `asyncio.gather()` for parallel processing

**WSJF Score**: 8/10 (High value, low effort)

## Recommended Optimization Priority

1. **Implement Ticker Caching** (WSJF: 8/10)
   - Add in-memory LRU cache with TTL
   - Reduce API calls by 95%+
   - Estimated improvement: 100-500ms per request

2. **Fix QA Engine Indexing** (WSJF: 9/10)
   - Batch document additions by default
   - Implement incremental index updates
   - Estimated improvement: 10x faster for multi-document scenarios

3. **Add Bounded Caches** (WSJF: 7/10)
   - Implement LRU eviction for all caches
   - Add memory limits
   - Prevents memory leaks

4. **Parallelize Multi-Company Analysis** (WSJF: 8/10)
   - Use async/await properly
   - Process companies in parallel
   - Estimated improvement: N-fold speedup where N = number of companies

5. **Implement Connection Pooling** (WSJF: 4/10)
   - Add Redis connection pool
   - HTTP connection pooling for Edgar client
   - Estimated improvement: 10-50ms per request

## Quick Wins (Can implement immediately)

1. **Add Redis Connection Pooling** - 1 hour effort, immediate latency reduction
2. **Cache company tickers JSON** - 2 hours effort, 95% reduction in API calls
3. **Default bulk mode for QA engine** - 30 minutes effort, prevents accidental O(n²) behavior
4. **Add cache size limits** - 2 hours effort, prevents memory leaks

## Monitoring Recommendations

1. Add performance metrics for:
   - Cache hit rates
   - Index rebuild frequency
   - API response times by endpoint
   - Memory usage over time

2. Set up alerts for:
   - Memory usage > 80%
   - API latency > 1s
   - Cache miss rate > 50%

## Conclusion

The codebase has several performance bottlenecks that can be addressed with relatively low effort. The highest impact optimizations involve caching frequently accessed data and fixing algorithmic inefficiencies in the QA engine. Implementing these changes could result in 5-10x performance improvements for common operations.