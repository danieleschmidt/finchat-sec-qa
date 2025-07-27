# ADR-0002: Use joblib for Persistence Instead of Pickle

## Status
Accepted

## Context
The original implementation used Python's pickle module for persisting vector indices and cached data. However, pickle has security vulnerabilities as it can execute arbitrary code during deserialization, making it unsuitable for production systems.

## Decision
Replace pickle with joblib for all persistence operations:
- Vector index serialization
- Cache data storage
- Model persistence

## Consequences
### Positive
- Enhanced security (joblib is safer than pickle)
- Better performance for numpy arrays
- Automatic compression
- More reliable cross-platform compatibility

### Negative
- Migration effort for existing pickle files
- Slightly larger file sizes for some data types
- Additional dependency (though joblib is already included via scikit-learn)

## Implementation Notes
- Added automatic migration from `.pkl` to `.joblib` files
- Updated all save/load operations in `qa_engine.py`
- Maintained backward compatibility during transition period