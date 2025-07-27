# Security Policy

## Supported Versions

We actively support and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.4.x   | :white_check_mark: |
| 1.3.x   | :white_check_mark: |
| 1.2.x   | :x:                |
| < 1.2   | :x:                |

## Reporting a Vulnerability

We take the security of FinChat-SEC-QA seriously. If you have discovered a security vulnerability, please follow these guidelines:

### How to Report

1. **Do NOT create a public issue** for security vulnerabilities
2. **Email us directly** at security@terragonlabs.com
3. **Use our GitHub Security Advisory** (preferred method)
   - Go to the [Security tab](https://github.com/danieleschmidt/finchat-sec-qa/security)
   - Click "Report a vulnerability"
   - Fill out the security advisory form

### What to Include

Please include the following information in your report:

- **Description** of the vulnerability
- **Steps to reproduce** the issue
- **Potential impact** assessment
- **Suggested fix** (if you have one)
- **Your contact information** for follow-up

### Security Advisory Template

```
**Summary:** Brief description of the vulnerability

**Impact:** What could an attacker do with this vulnerability?

**Reproduction Steps:**
1. Step one
2. Step two
3. Step three

**Affected Versions:** Which versions are affected?

**Mitigation:** Any temporary workarounds?

**Additional Context:** Any other relevant information
```

## Our Commitment

When you report a security vulnerability, we commit to:

- **Acknowledge** your report within 24 hours
- **Provide an initial assessment** within 72 hours
- **Keep you informed** of our progress
- **Credit you** in our security advisory (unless you prefer anonymity)

## Security Response Timeline

| Action | Timeline |
|--------|----------|
| Initial acknowledgment | 24 hours |
| Initial assessment | 72 hours |
| Regular updates | Weekly |
| Security patch | 30 days (critical), 90 days (high) |

## Security Best Practices

### For Users

1. **Keep dependencies updated** - Regularly update to the latest version
2. **Secure API keys** - Never commit API keys to version control
3. **Use environment variables** - Store sensitive configuration securely
4. **Enable logging** - Monitor for suspicious activity
5. **Network security** - Use HTTPS and proper firewall configurations

### For Developers

1. **Dependency scanning** - Use `safety check` and Dependabot
2. **Static analysis** - Use bandit for Python security issues
3. **Input validation** - Validate all user inputs
4. **Secure coding** - Follow OWASP guidelines
5. **Regular updates** - Keep all dependencies current

## Known Security Considerations

### API Key Management
- OpenAI API keys must be stored securely
- Use environment variables, not hardcoded values
- Implement key rotation procedures
- Monitor API key usage for anomalies

### SEC EDGAR API Compliance
- Respect rate limiting to avoid IP blocking
- Use proper User-Agent headers with contact information
- Don't attempt to overwhelm SEC servers

### Financial Data Handling
- Treat all SEC filing data as potentially sensitive
- Implement proper access controls
- Log all data access for audit purposes
- Ensure data retention compliance

### Web Application Security
- Implement proper authentication
- Use HTTPS in production
- Validate all inputs to prevent injection attacks
- Implement rate limiting to prevent abuse

## Security Tools in Use

We use the following tools to maintain security:

- **Bandit** - Python security linter
- **Safety** - Dependency vulnerability scanner
- **Dependabot** - Automated dependency updates
- **CodeQL** - Semantic code analysis
- **Trivy** - Container vulnerability scanning
- **Semgrep** - Static analysis for security patterns

## Incident Response Process

In the event of a security incident:

1. **Immediate Response** (0-4 hours)
   - Assess the scope and impact
   - Contain the threat
   - Notify stakeholders

2. **Investigation** (4-24 hours)
   - Detailed forensic analysis
   - Identify root cause
   - Document timeline

3. **Resolution** (24-72 hours)
   - Implement fixes
   - Test thoroughly
   - Deploy patches

4. **Recovery** (72+ hours)
   - Monitor for recurrence
   - Update security measures
   - Post-incident review

## Security Audit History

| Date | Type | Scope | Findings | Status |
|------|------|-------|----------|--------|
| 2024-01-15 | Internal | Full codebase | 3 medium, 7 low | Resolved |
| 2023-10-30 | Dependency | All packages | 1 high, 2 medium | Resolved |

## Contact Information

- **Security Email:** security@terragonlabs.com
- **Security Team:** @terragon-security
- **Emergency Contact:** +1-555-SEC-HELP

## Bug Bounty Program

We currently do not have a formal bug bounty program, but we appreciate responsible disclosure and will:

- Acknowledge your contribution publicly (with your permission)
- Provide swag or small tokens of appreciation
- Consider you for our future bug bounty program when established

## Legal

This security policy is subject to our [Terms of Service](LICENSE) and applicable law. By reporting vulnerabilities, you agree to:

- Act in good faith
- Not access or modify data beyond what's necessary to demonstrate the vulnerability
- Not perform any attacks that could harm our users or infrastructure
- Give us reasonable time to address issues before public disclosure

Thank you for helping keep FinChat-SEC-QA secure!