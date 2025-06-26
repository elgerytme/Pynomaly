# Security Setup & Best Practices

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 📄 Security

---


This guide covers security configuration and best practices for deploying Pynomaly in production environments.

## 🔒 **Quick Security Checklist**

- [ ] Configure authentication and authorization
- [ ] Enable HTTPS/TLS encryption
- [ ] Set up input validation and sanitization
- [ ] Configure secure headers and CORS
- [ ] Enable audit logging and monitoring
- [ ] Secure database connections
- [ ] Implement rate limiting
- [ ] Set up secret management

---

## 🛡️ **Authentication & Authorization**

### **JWT Authentication**
```python
# Environment variables
JWT_SECRET_KEY=your-256-bit-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30
```

### **Role-Based Access Control (RBAC)**
```yaml
roles:
  admin:
    - manage_users
    - manage_system
    - view_all_data
  analyst:
    - run_detection
    - view_own_data
    - export_results
  viewer:
    - view_dashboards
    - view_reports
```

### **API Key Authentication**
```bash
# Generate API keys
pynomaly auth create-key --user analyst@company.com --role analyst

# Use in requests
curl -H "Authorization: Bearer API_KEY" https://api.example.com/detect
```

---

## 🔐 **Network Security**

### **HTTPS/TLS Configuration**
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
}
```

### **CORS Configuration**
```python
# FastAPI CORS settings
CORS_ORIGINS = [
    "https://pynomaly.example.com",
    "https://dashboard.example.com"
]
CORS_METHODS = ["GET", "POST", "PUT", "DELETE"]
CORS_HEADERS = ["Content-Type", "Authorization"]
```

### **Security Headers**
```python
# Security middleware
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'"
}
```

---

## 🛡️ **Input Validation & Sanitization**

### **Data Validation**
```python
from pydantic import BaseModel, validator
from typing import List, Optional

class DetectionRequest(BaseModel):
    data: List[dict]
    algorithm: str = "isolation_forest"
    contamination: float = 0.1
    
    @validator('contamination')
    def validate_contamination(cls, v):
        if not 0.0 <= v <= 0.5:
            raise ValueError('contamination must be between 0.0 and 0.5')
        return v
    
    @validator('data')
    def validate_data_size(cls, v):
        if len(v) > 10000:
            raise ValueError('data size exceeds maximum limit')
        return v
```

### **SQL Injection Prevention**
```python
# Always use parameterized queries
cursor.execute(
    "SELECT * FROM experiments WHERE user_id = %s AND status = %s",
    (user_id, status)
)
```

### **File Upload Security**
```python
ALLOWED_EXTENSIONS = {'.csv', '.json', '.parquet'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def validate_file_upload(file):
    # Check file extension
    if not any(file.filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError("Invalid file type")
    
    # Check file size
    if file.size > MAX_FILE_SIZE:
        raise ValueError("File too large")
    
    # Scan for malicious content
    if scan_file_for_threats(file):
        raise ValueError("File contains threats")
```

---

## 🔍 **Monitoring & Auditing**

### **Audit Logging**
```python
import structlog

audit_logger = structlog.get_logger("audit")

# Log security events
audit_logger.info(
    "authentication_success",
    user_id=user.id,
    ip_address=request.client.host,
    user_agent=request.headers.get("user-agent"),
    timestamp=datetime.utcnow()
)

audit_logger.warning(
    "authentication_failure",
    username=username,
    ip_address=request.client.host,
    reason="invalid_credentials",
    timestamp=datetime.utcnow()
)
```

### **Security Monitoring**
```yaml
# Prometheus metrics
security_metrics:
  - authentication_attempts_total
  - authentication_failures_total
  - rate_limit_violations_total
  - suspicious_activity_total
  - data_access_events_total
```

### **Intrusion Detection**
```python
# Rate limiting and anomaly detection
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/detect")
@limiter.limit("10/minute")
async def detect_anomalies(request: Request):
    # Monitor for suspicious patterns
    if detect_suspicious_activity(request):
        audit_logger.warning("suspicious_activity_detected")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
```

---

## 🗄️ **Database Security**

### **Connection Security**
```python
# Encrypted database connections
DATABASE_URL = (
    "postgresql://user:password@host:5432/pynomaly"
    "?sslmode=require&sslcert=client-cert.pem&sslkey=client-key.pem"
)
```

### **Data Encryption**
```python
from cryptography.fernet import Fernet

# Encrypt sensitive data at rest
encryption_key = Fernet.generate_key()
fernet = Fernet(encryption_key)

def encrypt_sensitive_data(data: str) -> str:
    return fernet.encrypt(data.encode()).decode()

def decrypt_sensitive_data(encrypted_data: str) -> str:
    return fernet.decrypt(encrypted_data.encode()).decode()
```

### **Database Permissions**
```sql
-- Create restricted database user
CREATE USER pynomaly_api WITH PASSWORD 'secure_password';

-- Grant minimal required permissions
GRANT SELECT, INSERT, UPDATE ON experiments TO pynomaly_api;
GRANT SELECT, INSERT ON audit_logs TO pynomaly_api;

-- Revoke dangerous permissions
REVOKE CREATE, DROP, ALTER ON SCHEMA public FROM pynomaly_api;
```

---

## 🔑 **Secret Management**

### **Environment Variables**
```bash
# .env.production
DATABASE_PASSWORD=secure_db_password
JWT_SECRET_KEY=your-256-bit-secret-key
API_ENCRYPTION_KEY=encryption-key-for-api-data
THIRD_PARTY_API_KEY=external-service-key
```

### **Docker Secrets**
```yaml
# docker-compose.yml
version: '3.8'
services:
  pynomaly:
    image: pynomaly:latest
    secrets:
      - db_password
      - jwt_secret
    environment:
      - DATABASE_PASSWORD_FILE=/run/secrets/db_password
      - JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret

secrets:
  db_password:
    file: ./secrets/db_password.txt
  jwt_secret:
    file: ./secrets/jwt_secret.txt
```

### **Kubernetes Secrets**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: pynomaly-secrets
type: Opaque
data:
  database-password: <base64-encoded-password>
  jwt-secret-key: <base64-encoded-secret>
```

---

## 🚨 **Incident Response**

### **Security Incident Workflow**
1. **Detection**: Monitor logs and alerts
2. **Assessment**: Determine scope and impact
3. **Containment**: Isolate affected systems
4. **Investigation**: Analyze attack vectors
5. **Recovery**: Restore normal operations
6. **Lessons Learned**: Update security measures

### **Emergency Contacts**
```yaml
security_team:
  - name: "Security Team Lead"
    email: "security@company.com"
    phone: "+1-555-SECURITY"
  - name: "DevOps Engineer"
    email: "devops@company.com"
    phone: "+1-555-DEVOPS"
```

### **Incident Response Scripts**
```bash
#!/bin/bash
# emergency_lockdown.sh

# Disable API access
kubectl scale deployment pynomaly-api --replicas=0

# Enable maintenance mode
kubectl apply -f maintenance-mode.yaml

# Alert security team
curl -X POST "https://alerts.company.com/security" \
  -d '{"event": "emergency_lockdown", "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'
```

---

## 🔧 **Security Hardening**

### **Container Security**
```dockerfile
# Use non-root user
FROM python:3.11-slim
RUN adduser --disabled-password --gecos '' pynomaly
USER pynomaly

# Remove unnecessary packages
RUN apt-get remove --purge -y wget curl && apt-get autoremove -y

# Set secure file permissions
COPY --chown=pynomaly:pynomaly . /app
RUN chmod -R 755 /app
```

### **Kubernetes Security**
```yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: pynomaly
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

### **Network Policies**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pynomaly-network-policy
spec:
  podSelector:
    matchLabels:
      app: pynomaly
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
```

---

## 📋 **Security Compliance**

### **Compliance Frameworks**
- **SOC 2 Type II**: System and Organization Controls
- **ISO 27001**: Information Security Management
- **GDPR**: General Data Protection Regulation
- **CCPA**: California Consumer Privacy Act
- **HIPAA**: Health Insurance Portability and Accountability Act

### **Regular Security Tasks**
```yaml
security_schedule:
  daily:
    - Review security logs
    - Check for failed authentication attempts
    - Monitor system resource usage
  weekly:
    - Update security patches
    - Review access permissions
    - Backup security configurations
  monthly:
    - Conduct security assessments
    - Review and update policies
    - Test incident response procedures
  quarterly:
    - Penetration testing
    - Security training for team
    - Compliance audits
```

---

## 🔗 **Related Documentation**

- **[Production Deployment Guide](PRODUCTION_DEPLOYMENT_GUIDE.md)** - Complete production setup
- **[Docker Deployment](DOCKER_DEPLOYMENT_GUIDE.md)** - Container security
- **[Kubernetes Deployment](kubernetes.md)** - Orchestration security
- **[Security Best Practices](../security/security-best-practices.md)** - Comprehensive security guide
- **[Monitoring Guide](../user-guides/basic-usage/monitoring.md)** - System monitoring and alerting

---

## 🆘 **Getting Help**

### **Security Issues**
For security vulnerabilities, please email: security@pynomaly.org

### **Documentation**
- **[Security Best Practices](../security/security-best-practices.md)**
- **[Troubleshooting Guide](../user-guides/troubleshooting/troubleshooting.md)**
- **[GitHub Security Advisories](https://github.com/your-org/pynomaly/security/advisories)**

### **Community**
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)**
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/pynomaly)**