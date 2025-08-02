# GitHub Actions Workflow Setup Guide

This guide provides step-by-step instructions for setting up GitHub Actions workflows for FinChat-SEC-QA.

## Prerequisites

- Repository admin access
- GitHub Actions enabled
- Required secrets configured
- Understanding of GitHub Actions concepts

## Setup Overview

### Phase 1: Core CI/CD (Essential)
1. Continuous Integration workflow
2. Security scanning workflow
3. Basic deployment workflow

### Phase 2: Advanced Automation (Recommended)
1. Dependency management
2. Release automation
3. Performance monitoring

### Phase 3: Enterprise Features (Optional)
1. Multi-environment deployments
2. Advanced security compliance
3. Comprehensive monitoring

## Step-by-Step Setup

### Step 1: Repository Configuration

#### Enable GitHub Actions
1. Go to repository Settings → Actions → General
2. Set Actions permissions to "Allow all actions and reusable workflows"
3. Set Workflow permissions to "Read and write permissions"
4. Enable "Allow GitHub Actions to create and approve pull requests"

#### Configure Branch Protection
1. Go to Settings → Branches
2. Add rule for `main` branch:
   ```
   - Require a pull request before merging
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
   - Include administrators
   ```

#### Required Status Checks
Add these status checks once workflows are created:
- `CI Status`
- `Security & Compliance`
- `Code Quality`
- `Test Suite`

### Step 2: Secrets Configuration

#### Navigate to Repository Secrets
Go to Settings → Secrets and variables → Actions

#### Add Required Secrets

##### Essential Secrets
```bash
# Container Registry
DOCKER_USERNAME=your-docker-username
DOCKER_PASSWORD=your-docker-token

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# API Keys (if using external services)
SONAR_TOKEN=your-sonarcloud-token
CODECOV_TOKEN=your-codecov-token
```

##### Deployment Secrets (if using AWS)
```bash
# Staging Environment
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Production Environment  
AWS_ACCESS_KEY_ID_PROD=AKIA...
AWS_SECRET_ACCESS_KEY_PROD=...

# Application Secrets
STAGING_TEST_TOKEN=your-staging-test-token
PRODUCTION_TEST_TOKEN=your-production-test-token
PRODUCTION_TARGET_GROUP_ARN=arn:aws:elasticloadbalancing:...
```

##### Security Scanning Secrets
```bash
# For advanced security scanning
SNYK_TOKEN=your-snyk-token
WHITESOURCE_API_KEY=your-whitesource-key
```

### Step 3: Workflow File Creation

#### Create Workflow Directory
```bash
mkdir -p .github/workflows
```

#### Copy Template Files

##### 1. Essential: Continuous Integration
```bash
cp docs/workflows/templates/ci.yml .github/workflows/ci.yml
```

##### 2. Essential: Security Scanning  
```bash
cp docs/workflows/templates/security-scan.yml .github/workflows/security-scan.yml
```

##### 3. Recommended: Continuous Deployment
```bash
cp docs/workflows/templates/cd.yml .github/workflows/cd.yml
```

##### 4. Recommended: Dependency Updates
```bash
cp docs/workflows/templates/dependency-update.yml .github/workflows/dependency-update.yml
```

##### 5. Optional: Release Automation
```bash
cp docs/workflows/templates/release.yml .github/workflows/release.yml
```

### Step 4: Workflow Customization

#### Environment-Specific Configuration

##### Update Deployment Targets
Edit `.github/workflows/cd.yml`:
```yaml
# Replace placeholder URLs with your actual deployment targets
environment:
  name: staging
  url: https://your-staging-domain.com  # Update this

environment:
  name: production  
  url: https://your-production-domain.com  # Update this
```

##### Configure Container Registry
Edit workflow files to use your container registry:
```yaml
env:
  REGISTRY: ghcr.io  # or your registry
  IMAGE_NAME: ${{ github.repository }}
```

##### Update Notification Channels
Configure Slack channels in workflow files:
```yaml
# Update these channel names
channel: '#your-alerts-channel'
channel: '#your-deployments-channel'
channel: '#your-security-alerts'
```

#### Application-Specific Configuration

##### Test Configuration
Update test commands in `ci.yml`:
```yaml
# Customize test execution
- name: Run tests
  run: |
    pytest tests/ -v \
      --cov=src \
      --cov-report=xml \
      --cov-fail-under=85  # Adjust coverage threshold
```

##### Security Scanning Configuration
Customize security scanning in workflows:
```yaml
# Adjust security scan parameters
- name: Run Bandit
  run: |
    bandit -r src/ \
      -f json \
      -o bandit-report.json \
      --severity-level medium  # Adjust severity
```

### Step 5: Environment Setup

#### Staging Environment Configuration

##### Create Environment
1. Go to Settings → Environments
2. Click "New environment"
3. Name: `staging`
4. Configure protection rules:
   - Required reviewers: 1
   - Wait timer: 0 minutes
   - Deployment branches: main only

##### Add Environment Secrets
Add staging-specific secrets to the environment:
```bash
STAGING_DATABASE_URL=postgresql://...
STAGING_API_KEY=staging-api-key
STAGING_DOMAIN=staging.yourdomain.com
```

#### Production Environment Configuration

##### Create Environment
1. Go to Settings → Environments
2. Click "New environment"
3. Name: `production`
4. Configure protection rules:
   - Required reviewers: 2
   - Wait timer: 5 minutes
   - Deployment branches: main only

##### Add Environment Secrets
Add production-specific secrets:
```bash
PRODUCTION_DATABASE_URL=postgresql://...
PRODUCTION_API_KEY=production-api-key
PRODUCTION_DOMAIN=yourdomain.com
```

### Step 6: Testing and Validation

#### Initial Validation

##### Test CI Workflow
1. Create a feature branch
2. Make a small change
3. Create pull request
4. Verify CI workflow runs successfully

##### Test Security Scanning
1. Check workflow run logs
2. Verify security reports are generated
3. Confirm no critical vulnerabilities

##### Test Build Process
1. Verify Docker images build successfully
2. Check image security scanning
3. Confirm artifacts are uploaded

#### Troubleshooting Common Issues

##### Permission Errors
```bash
# If you see permission errors:
Error: Resource not accessible by integration

# Solution: Check workflow permissions in repository settings
# Go to Settings → Actions → General → Workflow permissions
# Select "Read and write permissions"
```

##### Secret Access Issues
```bash
# If secrets are not accessible:
Error: Secret MY_SECRET not found

# Solution: Verify secret name exactly matches
# Check secret is added at correct level (repo/environment)
# Confirm secret name in workflow file
```

##### Test Failures
```bash
# If tests fail unexpectedly:
# 1. Check test dependencies are installed
# 2. Verify environment variables are set
# 3. Review test configuration
# 4. Check for environment-specific issues
```

### Step 7: Monitoring and Maintenance

#### Set Up Workflow Monitoring

##### GitHub Actions Insights
1. Go to repository Insights → Actions
2. Monitor workflow success rates
3. Track workflow execution times
4. Identify frequently failing workflows

##### Notification Configuration
Configure alerts for workflow failures:
```yaml
# Add to workflow files
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

#### Regular Maintenance Tasks

##### Weekly Tasks
- Review workflow execution reports
- Check for failed workflows
- Update dependencies in workflows
- Review security scan results

##### Monthly Tasks
- Update GitHub Actions to latest versions
- Review and optimize workflow performance
- Update documentation
- Audit secrets and permissions

##### Quarterly Tasks
- Comprehensive security review
- Performance optimization
- Workflow architecture review
- Team training on new features

## Advanced Configuration

### Matrix Testing Strategy

#### Python Version Matrix
```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    os: [ubuntu-latest, windows-latest, macos-latest]
    exclude:
      # Optimize matrix for efficiency
      - os: windows-latest
        python-version: '3.8'
```

#### Dependency Matrix Testing
```yaml
strategy:
  matrix:
    dependency-version:
      - 'fastapi==0.95.*'
      - 'fastapi==0.96.*'
      - 'fastapi==latest'
```

### Performance Optimization

#### Caching Strategy
```yaml
# Comprehensive caching
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/pip
      ~/.cache/pre-commit
      ~/.docker
    key: ${{ runner.os }}-deps-${{ hashFiles('**/pyproject.toml') }}
```

#### Parallel Job Execution
```yaml
# Optimize job dependencies
jobs:
  lint:
    # Fast jobs first
  test:
    needs: lint  # Only after lint passes
  deploy:
    needs: [lint, test]  # Only after all pass
```

### Security Hardening

#### Workflow Security Best Practices
```yaml
# Use specific action versions
- uses: actions/checkout@v4  # Not @main or @v4

# Minimal permissions
permissions:
  contents: read
  security-events: write

# Secure secret handling
env:
  SECRET_VALUE: ${{ secrets.MY_SECRET }}
  # Never: SECRET_VALUE: my-secret-value
```

#### Supply Chain Security
```yaml
# SLSA provenance generation
- name: Generate provenance
  uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.7.0
```

## Troubleshooting Guide

### Common Workflow Issues

#### Workflow Not Triggering
**Symptoms:** Workflow doesn't run on expected events

**Solutions:**
1. Check workflow syntax with GitHub's validator
2. Verify trigger conditions are met
3. Check if workflow is disabled
4. Review branch protection settings

#### Permission Denied Errors
**Symptoms:** `Error: Resource not accessible by integration`

**Solutions:**
1. Check workflow permissions in repository settings
2. Verify GITHUB_TOKEN has required permissions
3. Review environment protection rules
4. Confirm user has necessary repository access

#### Secret Not Found Errors
**Symptoms:** `Error: Secret 'SECRET_NAME' not found`

**Solutions:**
1. Verify secret exists at correct level
2. Check secret name spelling and case
3. Confirm environment name matches
4. Review environment access permissions

#### Test Failures in CI
**Symptoms:** Tests pass locally but fail in CI

**Solutions:**
1. Check environment differences
2. Review dependency versions
3. Verify environment variables
4. Check file system permissions
5. Review service dependencies

#### Container Build Failures
**Symptoms:** Docker build fails in workflow

**Solutions:**
1. Check Dockerfile syntax
2. Verify base image availability
3. Review build context and .dockerignore
4. Check for network/registry issues
5. Verify build args and secrets

### Performance Issues

#### Slow Workflow Execution
**Solutions:**
1. Implement caching strategies
2. Optimize job parallelization
3. Reduce matrix size
4. Use faster runner types
5. Optimize Docker builds

#### High Resource Usage
**Solutions:**
1. Monitor resource consumption
2. Optimize test execution
3. Reduce artifact sizes
4. Clean up old artifacts
5. Use resource limits

### Getting Help

#### Internal Resources
1. **Team Documentation:** Check team wiki/confluence
2. **Platform Team:** Contact for infrastructure issues
3. **Security Team:** For security-related questions
4. **DevOps Team:** For deployment issues

#### External Resources
1. **GitHub Actions Documentation:** https://docs.github.com/en/actions
2. **GitHub Community:** https://github.community/
3. **GitHub Support:** For platform issues
4. **Action Marketplace:** https://github.com/marketplace?type=actions

#### Emergency Contacts
- **Critical Production Issues:** @platform-oncall
- **Security Incidents:** @security-team
- **Infrastructure Issues:** @devops-team

## Best Practices Summary

### Security
- ✅ Pin action versions to specific commits/tags
- ✅ Use least privilege permissions
- ✅ Store secrets securely
- ✅ Regular security scanning
- ✅ Monitor for vulnerabilities

### Performance
- ✅ Use caching effectively
- ✅ Parallelize independent jobs
- ✅ Optimize Docker builds
- ✅ Clean up artifacts
- ✅ Monitor execution times

### Reliability
- ✅ Implement proper error handling
- ✅ Use retries for flaky operations
- ✅ Set appropriate timeouts
- ✅ Monitor success rates
- ✅ Have rollback procedures

### Maintainability
- ✅ Document workflow purposes
- ✅ Use clear naming conventions
- ✅ Regular updates and reviews
- ✅ Version control for changes
- ✅ Team knowledge sharing

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-27  
**Next Review:** 2025-04-27