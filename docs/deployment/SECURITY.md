# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

Please report security vulnerabilities to **security@pynomaly.io**.

### What to Include

Please include the following information:

1. Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
2. Full paths of source file(s) related to the manifestation of the issue
3. The location of the affected source code (tag/branch/commit or direct URL)
4. Any special configuration required to reproduce the issue
5. Step-by-step instructions to reproduce the issue
6. Proof-of-concept or exploit code (if possible)
7. Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 5 business days
- **Resolution Timeline**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 90 days

## Security Best Practices

When using Pynomaly in production:

### API Security

1. **Authentication**: Always enable authentication in production
   ```python
   settings = Settings(
       auth_enabled=True,
       jwt_secret_key=os.getenv("JWT_SECRET_KEY")
   )
   ```

2. **HTTPS**: Always use HTTPS in production
   ```nginx
   server {
       listen 443 ssl http2;
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
   }
   ```

3. **Rate Limiting**: Configure appropriate rate limits
   ```python
   settings = Settings(
       rate_limit_requests=100,
       rate_limit_period=60  # per minute
   )
   ```

### Data Security

1. **Input Validation**: All inputs are validated using Pydantic
2. **SQL Injection**: Use parameterized queries (built-in with SQLAlchemy)
3. **File Uploads**: Validate file types and sizes
   ```python
   settings = Settings(
       max_dataset_size_mb=1000,
       allowed_file_types=[".csv", ".parquet"]
   )
   ```

### Secrets Management

1. **Environment Variables**: Never commit secrets
   ```bash
   # .env file (gitignored)
   SECRET_KEY=your-secret-key
   DATABASE_URL=postgresql://user:pass@host/db
   ```

2. **Key Rotation**: Regularly rotate secrets
3. **Encryption**: Sensitive data should be encrypted at rest

### Container Security

1. **Non-root User**: Containers run as non-root user
2. **Minimal Base Image**: Using slim Python images
3. **Security Scanning**: Regular vulnerability scanning

```dockerfile
# Run as non-root user
USER pynomaly
```

### Network Security

1. **Firewall Rules**: Restrict access to necessary ports only
2. **VPC/Private Networks**: Deploy in isolated networks
3. **Service Mesh**: Consider using Istio for microservices

## Security Headers

Recommended security headers for production:

```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'" always;
```

## Dependencies

We regularly update dependencies to patch known vulnerabilities:

```bash
# Check for vulnerabilities
poetry run safety check
poetry run pip-audit

# Update dependencies
poetry update
```

## Disclosure Policy

- Security issues are disclosed after a fix is available
- Users are notified via GitHub Security Advisories
- CVE IDs are requested for significant vulnerabilities

## Acknowledgments

We thank the following researchers for responsibly disclosing security issues:

- (List will be updated as reports are received)

## Contact

- Security Team: security@pynomaly.io
- General Support: support@pynomaly.io
- Bug Reports: https://github.com/pynomaly/pynomaly/issues