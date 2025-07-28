# Manual Setup Requirements

## Overview

Due to GitHub App permission limitations during the automated SDLC implementation, certain configurations require manual setup by repository maintainers. This document provides step-by-step instructions for completing the SDLC automation.

## Priority: HIGH - Required for Complete SDLC Automation

### 1. GitHub Actions Workflows

#### Required Actions
1. **Copy workflow templates** from `docs/workflows/templates/` to `.github/workflows/`
2. **Configure repository secrets** for external service integrations
3. **Set up branch protection rules** to require workflow success
4. **Enable GitHub security features** (Dependabot, CodeQL, etc.)

#### Specific Workflows to Create

##### Enhanced CI Pipeline
```bash
# Copy the enhanced CI workflow
cp docs/workflows/templates/ci.yml .github/workflows/ci.yml
```

**Required Secrets:**
- `CODECOV_TOKEN` - For code coverage reporting
- `SLACK_WEBHOOK_URL` - For build notifications
- `SONAR_TOKEN` - For SonarCloud integration (optional)

##### Security Scanning Pipeline
```bash
# Copy the security scanning workflow
cp docs/workflows/templates/security-scan.yml .github/workflows/security-scan.yml
```

**Required Secrets:**
- `SEMGREP_APP_TOKEN` - For Semgrep SAST scanning
- `SECURITY_SLACK_WEBHOOK_URL` - For security alert notifications
- `GITLEAKS_LICENSE` - For GitLeaks secret scanning (optional)

### 2. Branch Protection Rules

#### Configure for `main` branch:
1. Navigate to Repository Settings > Branches
2. Add rule for `main` branch
3. Enable:
   - [x] Require a pull request before merging
   - [x] Require status checks to pass before merging
   - [x] Require branches to be up to date before merging
   - [x] Include administrators
   - [x] Allow force pushes (for emergency fixes only)

#### Required Status Checks:
- CI / security-checks
- CI / code-quality
- CI / test (ubuntu-latest, 3.11)
- CI / docker
- CI / documentation
- CI / supply-chain

### 3. GitHub Security Features

#### Enable in Repository Settings > Security:
1. **Dependabot alerts** - Automatic dependency vulnerability alerts
2. **Dependabot security updates** - Automatic security patch PRs
3. **CodeQL analysis** - Advanced security scanning
4. **Secret scanning** - Detect committed secrets
5. **Private vulnerability reporting** - Secure disclosure process

#### Configure Dependabot
Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    reviewers:
      - "platform-team"
    labels:
      - "dependencies"
      - "security"
```

## Priority: MEDIUM - Enhanced Automation

### 4. Continuous Deployment Pipeline

#### Create deployment workflow:
```bash
# Create CD workflow (to be implemented)
cp docs/workflows/examples/cd.yml .github/workflows/cd.yml
```

**Required Secrets:**
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub access token
- `AWS_ACCESS_KEY_ID` - AWS deployment credentials (if using AWS)
- `AWS_SECRET_ACCESS_KEY` - AWS deployment credentials
- `STAGING_DEPLOY_KEY` - SSH key for staging deployment
- `PRODUCTION_DEPLOY_KEY` - SSH key for production deployment

### 5. External Service Integrations

#### Code Quality Tools
1. **SonarCloud** (recommended)
   - Sign up at https://sonarcloud.io
   - Connect GitHub repository
   - Add `SONAR_TOKEN` to repository secrets

2. **Codecov** (recommended)
   - Sign up at https://codecov.io
   - Connect GitHub repository
   - Add `CODECOV_TOKEN` to repository secrets

#### Security Tools
1. **Semgrep** (recommended)
   - Sign up at https://semgrep.dev
   - Add repository
   - Add `SEMGREP_APP_TOKEN` to repository secrets

2. **Snyk** (optional)
   - Sign up at https://snyk.io
   - Connect GitHub repository
   - Add `SNYK_TOKEN` to repository secrets

### 6. Monitoring and Alerting

#### Slack Integration
1. Create Slack app in your workspace
2. Add incoming webhook
3. Add webhook URLs to repository secrets:
   - `SLACK_WEBHOOK_URL` - General notifications
   - `SECURITY_SLACK_WEBHOOK_URL` - Security alerts

#### External Monitoring (Optional)
1. **Datadog** - Application performance monitoring
2. **New Relic** - Full-stack observability
3. **Sentry** - Error tracking and performance

## Priority: LOW - Advanced Features

### 7. Release Automation

#### Semantic Release Setup
1. Install semantic-release dependencies:
```bash
npm install -g semantic-release @semantic-release/changelog @semantic-release/git
```

2. Configure package.json (if using npm):
```json
{
  "devDependencies": {
    "semantic-release": "^19.0.0",
    "@semantic-release/changelog": "^6.0.0",
    "@semantic-release/git": "^10.0.0"
  }
}
```

3. Create release workflow:
```bash
cp docs/workflows/examples/release.yml .github/workflows/release.yml
```

### 8. Advanced Security Configuration

#### GitHub Advanced Security (Enterprise only)
1. Enable code scanning with custom queries
2. Set up secret scanning for custom patterns
3. Configure private vulnerability reporting

#### External Security Tools
1. **GitLab Ultimate** - Advanced SAST/DAST scanning
2. **Veracode** - Static and dynamic analysis
3. **Checkmarx** - SAST scanning

## Validation Checklist

After completing the manual setup, verify the following:

### GitHub Repository Configuration
- [ ] All workflow files are in `.github/workflows/`
- [ ] Required secrets are configured
- [ ] Branch protection rules are active
- [ ] GitHub security features are enabled
- [ ] Dependabot is configured and active

### CI/CD Pipeline
- [ ] CI workflow runs successfully on PR creation
- [ ] Security scans execute without errors
- [ ] Test coverage reports are generated
- [ ] Docker images build successfully
- [ ] Deployment workflow is ready (if implemented)

### Security Configuration
- [ ] Secret scanning is active
- [ ] Dependency alerts are enabled
- [ ] Code scanning finds and reports issues
- [ ] Security notifications reach the team

### Integration Testing
1. **Create a test PR** with a small change
2. **Verify all checks pass** in the CI pipeline
3. **Check notifications** are sent to Slack/email
4. **Confirm security scans** complete successfully
5. **Test branch protection** prevents direct pushes to main

## Troubleshooting

### Common Issues

#### Workflow Permissions
```yaml
# Add to workflow files if needed
permissions:
  contents: read
  security-events: write
  actions: read
```

#### Secret Configuration
```bash
# Verify secrets are available (in workflow)
echo "Checking required secrets..."
if [ -z "${{ secrets.CODECOV_TOKEN }}" ]; then
  echo "⚠️ CODECOV_TOKEN not configured"
fi
```

#### Branch Protection Issues
- Ensure administrators are included in branch protection
- Verify status check names match workflow job names exactly
- Check that required checks are not case-sensitive

### Getting Help

1. **Documentation**: Review GitHub Actions and security documentation
2. **Community**: Ask questions in GitHub Community forum
3. **Support**: Contact GitHub Support for enterprise features
4. **Team**: Reach out to platform team for internal configurations

## Maintenance Schedule

### Weekly
- Review Dependabot PRs and merge security updates
- Check workflow success rates and investigate failures
- Monitor security scan results for new vulnerabilities

### Monthly
- Review and update workflow configurations
- Audit repository permissions and access
- Update external service integrations
- Review security policies and thresholds

### Quarterly
- Comprehensive security audit
- Performance review of CI/CD pipeline
- Update tool versions and configurations
- Review and update this documentation

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-27  
**Next Review**: 2025-04-27  
**Owner**: Platform Team

**Important**: This setup is required to complete the SDLC automation initiated by the Terragon checkpoint strategy. Repository maintainers should prioritize HIGH priority items for immediate implementation.