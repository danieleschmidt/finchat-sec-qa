# Mutation Testing

This directory contains mutation testing configuration and results for FinChat-SEC-QA.

## Overview

Mutation testing evaluates the quality of test suites by introducing small code changes (mutations) and checking if tests catch these defects. It helps identify:
- Weak test coverage areas
- Tests that don't actually validate behavior
- Missing edge case testing
- Redundant or ineffective tests

## Tools

### mutmut (Primary Tool)
```bash
# Install mutmut
pip install mutmut

# Run mutation testing
mutmut run

# View results
mutmut show
mutmut html
```

### Alternative: cosmic-ray
```bash
# Install cosmic-ray
pip install cosmic-ray

# Initialize configuration
cosmic-ray init cosmic-ray.toml src/finchat_sec_qa tests/

# Run mutation testing
cosmic-ray exec cosmic-ray.toml
```

## Configuration

### mutmut Configuration (pyproject.toml)
```toml
[tool.mutmut]
paths_to_mutate = "src/finchat_sec_qa/"
tests_dir = "tests/"
runner = "python -m pytest"
exclude = [
    "__init__.py",
    "test_*.py",
    "*_test.py"
]
```

### Exclusion Patterns
Exclude from mutation testing:
- Configuration files
- Test files themselves
- Generated code
- Third-party integrations (mocked in tests)
- Logging statements
- Type annotations

## Target Metrics

### Mutation Score Targets
- **Critical paths**: 85%+ mutation score
- **Core business logic**: 80%+ mutation score
- **Utility functions**: 75%+ mutation score
- **Overall project**: 70%+ mutation score

### Quality Indicators
- High mutation score with good line coverage
- Fast test execution time
- Clear test failure reasons
- Minimal false positives

## Running Mutation Tests

### Full Suite
```bash
# Complete mutation testing run
mutmut run --paths-to-mutate src/finchat_sec_qa/

# Generate HTML report
mutmut html
open html/index.html
```

### Targeted Testing
```bash
# Test specific module
mutmut run --paths-to-mutate src/finchat_sec_qa/qa_engine.py

# Test specific function
mutmut run --paths-to-mutate src/finchat_sec_qa/qa_engine.py::process_query
```

### CI/CD Integration
```bash
# Fast mutation testing for CI
mutmut run --use-coverage --timeout-factor 2.0

# Generate machine-readable results
mutmut run --CI
```

## Results Analysis

### Interpreting Results
1. **Killed mutations**: Tests caught the defect (good)
2. **Survived mutations**: Tests missed the defect (needs attention)
3. **Timeout mutations**: Tests took too long (investigate)
4. **Suspicious mutations**: Potential false positives

### Common Survival Patterns
- **Boundary conditions**: Off-by-one errors
- **Error handling**: Exception paths not tested
- **Edge cases**: Null/empty inputs
- **Side effects**: External state changes
- **Logging**: Debug statements without assertions

## Improving Test Quality

### Based on Mutation Results
1. **Add missing assertions**: Test actual behavior, not just execution
2. **Test edge cases**: Boundary values, empty inputs, error conditions
3. **Verify side effects**: Database changes, file operations, API calls
4. **Test error paths**: Exception handling and error recovery
5. **Remove redundant tests**: Tests that don't add value

### Example Improvements
```python
# Before: Weak test
def test_process_query():
    result = process_query("test question")
    assert result is not None  # Survives many mutations

# After: Strong test
def test_process_query():
    result = process_query("What is revenue?")
    assert result.answer.startswith("Revenue")
    assert len(result.citations) > 0
    assert result.confidence > 0.5
```

## Performance Considerations

### Optimization Strategies
- Use `--use-coverage` to focus on tested code
- Set reasonable timeouts with `--timeout-factor`
- Run on dedicated CI resources
- Cache dependencies and test data
- Use parallel execution when possible

### Time Management
```bash
# Quick check (subset of mutations)
mutmut run --use-coverage --percentage 25

# Full run with timeout
mutmut run --timeout-factor 3.0
```

## Integration with Quality Gates

### GitHub Actions Example
```yaml
- name: Mutation Testing
  run: |
    mutmut run --use-coverage --CI
    mutation_score=$(mutmut show | grep "Mutation score" | cut -d' ' -f3)
    if (( $(echo "$mutation_score < 70" | bc -l) )); then
      echo "Mutation score $mutation_score below threshold"
      exit 1
    fi
```

### Quality Metrics Dashboard
- Track mutation scores over time
- Monitor test execution performance
- Identify declining test quality areas
- Compare mutation scores across modules

## Best Practices

1. **Regular execution**: Run mutation tests weekly or before major releases
2. **Incremental improvement**: Focus on one module at a time
3. **Test quality over quantity**: Better tests, not more tests
4. **Document findings**: Keep track of common mutation patterns
5. **Team education**: Share mutation testing insights with the team

## Troubleshooting

### Common Issues
- **Long execution times**: Use coverage-based mutations
- **False positives**: Review and exclude problematic patterns
- **Test flakiness**: Fix non-deterministic tests first
- **Memory issues**: Run smaller batches or increase resources

### Debug Mode
```bash
# Verbose output for debugging
mutmut run --debug

# Show specific mutation details
mutmut show 42
```

## Contributing

When working with mutation testing:
1. Review mutation results before committing
2. Add tests for survived mutations when appropriate
3. Update exclusion patterns for generated or external code
4. Document any mutation testing configuration changes
5. Share insights about effective testing patterns