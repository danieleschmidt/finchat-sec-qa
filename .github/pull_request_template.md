# Pull Request

## Summary

Brief description of what this PR accomplishes and why it's needed.

**Related Issue(s)**: Fixes #(issue number)

## Changes Made

- [ ] **Feature**: Describe new functionality added
- [ ] **Bug Fix**: Describe the bug fixed and solution
- [ ] **Refactoring**: Describe code improvements without functional changes
- [ ] **Documentation**: Describe documentation updates
- [ ] **Performance**: Describe performance improvements
- [ ] **Security**: Describe security improvements
- [ ] **Testing**: Describe test additions/improvements

### Specific Changes:

1. Change 1 - Brief description
2. Change 2 - Brief description
3. Change 3 - Brief description

## Testing

### Test Coverage
- [ ] Unit tests added/updated and passing
- [ ] Integration tests added/updated and passing
- [ ] End-to-end tests added/updated and passing (if applicable)
- [ ] Test coverage maintained above 85%

### Manual Testing
- [ ] Tested locally with development setup
- [ ] Tested with Docker containerized environment
- [ ] Tested SDK functionality (if applicable)
- [ ] Tested API endpoints (if applicable)
- [ ] Tested WebApp functionality (if applicable)

### Performance Testing
- [ ] Load testing performed (if performance-critical changes)
- [ ] Benchmarks run and documented (if applicable)
- [ ] No performance regressions introduced

## Security Review

- [ ] No sensitive data exposed in logs or responses
- [ ] Input validation implemented for new endpoints
- [ ] Authentication/authorization requirements met
- [ ] Security scan (bandit) passes
- [ ] Dependencies scanned for vulnerabilities

## Documentation

- [ ] README updated (if applicable)
- [ ] API documentation updated (if applicable)
- [ ] Code comments added for complex logic
- [ ] CHANGELOG.md updated
- [ ] Examples updated (if applicable)

## Code Quality

- [ ] Pre-commit hooks pass
- [ ] Type hints added for new code
- [ ] Code follows project style guidelines
- [ ] No code duplication introduced
- [ ] Error handling implemented appropriately

## Deployment Considerations

- [ ] Database migrations required: **Yes / No**
- [ ] Environment variables added/changed: **Yes / No**
- [ ] Breaking changes: **Yes / No**
- [ ] Backwards compatibility maintained: **Yes / No**

### Breaking Changes (if any):
Describe any breaking changes and migration path for users.

### New Environment Variables (if any):
List any new environment variables and their purpose.

## Rollback Plan

Describe how to rollback this change if issues are discovered in production:

1. Step 1
2. Step 2
3. Step 3

## Screenshots (if applicable)

Add screenshots here to show visual changes.

## Checklist

- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Additional Notes

Any additional information that reviewers should know about this PR.