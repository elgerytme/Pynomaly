# Comprehensive Production Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Production Checklist](#production-checklist)
3. [Monitoring Setup](#monitoring-setup)
4. [Security Hardening Guide](#security-hardening-guide)
5. [Backup and Recovery Procedures](#backup-and-recovery-procedures)
6. [Performance Tuning Guide](#performance-tuning-guide)
7. [Troubleshooting Procedures](#troubleshooting-procedures)
8. [Infrastructure as Code Examples](#infrastructure-as-code-examples)

---

## Prerequisites

### System Requirements

**Minimum Requirements:**

- **CPU**: 4 cores, 2.4GHz
- **Memory**: 8GB RAM
- **Storage**: 100GB SSD
- **Network**: 100Mbps bandwidth

**Recommended Production Requirements:**

- **CPU**: 8+ cores, 3.0GHz+
- **Memory**: 32GB+ RAM
- **Storage**: 500GB+ NVMe SSD
- **Network**: 1Gbps+ bandwidth

### Software Dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| Docker | 24.0+ | Container runtime |
| Docker Compose | 2.20+ | Container orchestration |
| Kubernetes | 1.28+ | Container orchestration (optional) |
| PostgreSQL | 15+ | Primary database |
| Redis | 7.0+ | Caching and session storage |
| NGINX | 1.24+ | Load balancer and reverse proxy |
| Python | 3.11+ | Application runtime |

### Cloud Provider Support

**AWS:**

- EC2 instances (t3.xlarge or larger)
- RDS PostgreSQL
- ElastiCache Redis
- Application Load Balancer
- S3 for storage
- CloudWatch for monitoring

**Google Cloud:**

- Compute Engine (n2-standard-4 or larger)
- Cloud SQL PostgreSQL
- Memorystore Redis
- Cloud Load Balancing
- Cloud Storage
- Cloud Monitoring

**Azure:**

- Virtual Machines (Standard_D4s_v3 or larger)
- Azure Database for PostgreSQL
- Azure Cache for Redis
- Application Gateway
- Blob Storage
- Azure Monitor

---

## Production Checklist

### ✅ Pre-Deployment Checklist

#### Infrastructure Setup

- [ ] **Server Provisioning**: Servers provisioned and accessible
- [ ] **Network Configuration**: VPC, subnets, and security groups configured
- [ ] **Domain Setup**: DNS records configured and SSL certificates obtained
- [ ] **Load Balancer**: Load balancer configured with health checks
- [ ] **Storage**: Persistent storage volumes attached and mounted
- [ ] **Backup Storage**: Backup storage configured (S3, Azure Blob, etc.)

#### Security Configuration

- [ ] **Firewall Rules**: Only necessary ports exposed (80, 443, 22)
- [ ] **SSL/TLS**: Valid certificates installed and configured
- [ ] **Access Control**: SSH key-based access configured
- [ ] **Secrets Management**: Environment variables and secrets secured
- [ ] **Security Scanning**: Infrastructure security scan completed
- [ ] **Compliance**: Regulatory compliance requirements verified

#### Application Setup

- [ ] **Environment Variables**: Production environment variables configured
- [ ] **Database**: Database server configured with production settings
- [ ] **Cache**: Redis configured with persistence and clustering
- [ ] **Monitoring**: Monitoring stack deployed and configured
- [ ] **Logging**: Centralized logging configured
- [ ] **Alerting**: Alert rules configured with proper notification channels

### ✅ Deployment Verification Checklist

#### Core Application

- [ ] **Health Endpoints**: `/health` and `/metrics` endpoints responding
- [ ] **Authentication**: JWT authentication working correctly
- [ ] **API Endpoints**: All REST API endpoints functional
- [ ] **WebSocket**: Real-time connections established successfully
- [ ] **Database Connectivity**: Application can connect to database
- [ ] **Cache Connectivity**: Application can connect to Redis

#### Machine Learning Components

- [ ] **Model Loading**: ML models load successfully
- [ ] **Algorithm Adapters**: All algorithm adapters functional
- [ ] **Training Pipeline**: Model training completes successfully
- [ ] **Prediction Pipeline**: Anomaly detection working correctly
- [ ] **AutoML**: Automated ML features operational
- [ ] **Model Versioning**: Model registry and versioning working

#### Web Interface

- [ ] **Frontend Loading**: Web UI loads without errors
- [ ] **Dashboard**: Monitoring dashboard displays correctly
- [ ] **Visualizations**: Charts and graphs render properly
- [ ] **Real-time Updates**: Live data updates working
- [ ] **Mobile Compatibility**: Interface works on mobile devices
- [ ] **Performance**: Page load times under 3 seconds

### ✅ Post-Deployment Checklist

#### Performance Validation

- [ ] **Response Times**: API response times under 500ms (95th percentile)
- [ ] **Throughput**: System handles expected load
- [ ] **Resource Usage**: CPU and memory usage within limits
- [ ] **Database Performance**: Query performance optimized
- [ ] **Network Performance**: Network latency acceptable

#### Security Validation

- [ ] **Authentication**: All authentication mechanisms tested
- [ ] **Authorization**: Role-based access control verified
- [ ] **Data Encryption**: Data encrypted at rest and in transit
- [ ] **Audit Logging**: Security events being logged
- [ ] **Vulnerability Scan**: No critical vulnerabilities found

#### Monitoring Validation

- [ ] **Metrics Collection**: All metrics being collected
- [ ] **Dashboards**: Monitoring dashboards accessible
- [ ] **Alerting**: Alert rules triggering correctly
- [ ] **Log Aggregation**: Logs being collected and searchable
- [ ] **Backup Verification**: Automated backups completing successfully

---

## Monitoring Setup

### Prometheus Configuration

Create `/etc/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'pynomaly-production'
    environment: 'production'

rule_files:
  - "rules/*.yml"

scrape_configs:
  # Pynomaly Application
  - job_name: 'pynomaly-api'
    static_configs:
      - targets: ['pynomaly-api-1:8000', 'pynomaly-api-2:8000', 'pynomaly-api-3:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s

  # System Metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node1:9100', 'node2:9100', 'node3:9100']

  # Database Metrics
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # Redis Metrics
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  # NGINX Metrics
  - job_name: 'nginx-exporter'
    static_configs:
      - targets: ['nginx-exporter:9113']

  # Kubernetes Metrics (if using K8s)
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names:
          - default
          - kube-system
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard Configuration

Create comprehensive monitoring dashboards:

```json
{
  "dashboard": {
    "id": null,
    "title": "Pynomaly Production Monitoring",
    "tags": ["pynomaly", "production"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"pynomaly-api\"}[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time Distribution",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{job=\"pynomaly-api\"}[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"pynomaly-api\"}[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{job=\"pynomaly-api\"}[5m]))",
            "legendFormat": "99th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"pynomaly-api\",status=~\"5..\"}[5m]) / rate(http_requests_total{job=\"pynomaly-api\"}[5m]) * 100"
          }
        ],
        "thresholds": [
          {
            "value": 1,
            "colorMode": "critical"
          },
          {
            "value": 0.5,
            "colorMode": "warning"
          }
        ]
      },
      {
        "id": 4,
        "title": "Active Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pynomaly_active_connections",
            "legendFormat": "Active Connections"
          }
        ]
      },
      {
        "id": 5,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes{job=\"pynomaly-api\"}",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "id": 6,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(process_cpu_seconds_total{job=\"pynomaly-api\"}[5m]) * 100",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "id": 7,
        "title": "Anomaly Detection Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(pynomaly_detections_total[5m])",
            "legendFormat": "Detections/sec"
          }
        ]
      },
      {
        "id": 8,
        "title": "Model Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "pynomaly_model_accuracy",
            "legendFormat": "{{model_name}}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### Alert Rules Configuration

Create `/etc/prometheus/rules/pynomaly.yml`:

```yaml
groups:
  - name: pynomaly.rules
    rules:
      # Application Health Alerts
      - alert: PynomályServiceDown
        expr: up{job="pynomaly-api"} == 0
        for: 1m
        labels:
          severity: critical
          service: pynomaly
        annotations:
          summary: "Pynomaly service instance is down"
          description: "Pynomaly service instance {{ $labels.instance }} has been down for more than 1 minute"
          runbook_url: "https://docs.pynomaly.com/runbooks/service-down"

      - alert: HighErrorRate
        expr: rate(http_requests_total{job="pynomaly-api",status=~"5.."}[5m]) / rate(http_requests_total{job="pynomaly-api"}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
          service: pynomaly
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.instance }}"
          runbook_url: "https://docs.pynomaly.com/runbooks/high-error-rate"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="pynomaly-api"}[5m])) > 1
        for: 5m
        labels:
          severity: warning
          service: pynomaly
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s for {{ $labels.instance }}"

      # Resource Usage Alerts
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes{job="pynomaly-api"} / 1024 / 1024 / 1024 > 4
        for: 10m
        labels:
          severity: warning
          service: pynomaly
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanize }}GB for {{ $labels.instance }}"

      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total{job="pynomaly-api"}[5m]) * 100 > 80
        for: 10m
        labels:
          severity: warning
          service: pynomaly
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value | humanizePercentage }} for {{ $labels.instance }}"

      # Database Alerts
      - alert: DatabaseConnectionFailure
        expr: up{job="postgres-exporter"} == 0
        for: 1m
        labels:
          severity: critical
          service: database
        annotations:
          summary: "Database connection failed"
          description: "PostgreSQL database is not reachable"
          runbook_url: "https://docs.pynomaly.com/runbooks/database-down"

      - alert: DatabaseSlowQueries
        expr: pg_stat_activity_max_tx_duration{datname="pynomaly"} > 30
        for: 5m
        labels:
          severity: warning
          service: database
        annotations:
          summary: "Slow database queries detected"
          description: "Maximum transaction duration is {{ $value }}s"

      # Redis Alerts
      - alert: RedisConnectionFailure
        expr: up{job="redis-exporter"} == 0
        for: 1m
        labels:
          severity: critical
          service: redis
        annotations:
          summary: "Redis connection failed"
          description: "Redis instance is not reachable"

      - alert: RedisHighMemoryUsage
        expr: redis_memory_used_bytes / redis_memory_max_bytes * 100 > 90
        for: 5m
        labels:
          severity: warning
          service: redis
        annotations:
          summary: "Redis memory usage high"
          description: "Redis memory usage is {{ $value | humanizePercentage }}"

      # Business Logic Alerts
      - alert: LowDetectionRate
        expr: rate(pynomaly_detections_total[1h]) < 10
        for: 15m
        labels:
          severity: warning
          service: pynomaly
        annotations:
          summary: "Anomaly detection rate is low"
          description: "Detection rate has been {{ $value }}/hour for the last 15 minutes"

      - alert: ModelAccuracyDrop
        expr: pynomaly_model_accuracy < 0.85
        for: 10m
        labels:
          severity: warning
          service: pynomaly
        annotations:
          summary: "Model accuracy has dropped"
          description: "Model {{ $labels.model_name }} accuracy is {{ $value | humanizePercentage }}"
```

### Alertmanager Configuration

Create `/etc/alertmanager/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@pynomaly.com'
  smtp_auth_username: 'alerts@pynomaly.com'
  smtp_auth_password: 'your-email-password'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default-receiver'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 10s
      repeat_interval: 10m
    - match:
        severity: warning
      receiver: 'warning-alerts'
      repeat_interval: 30m

receivers:
  - name: 'default-receiver'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#pynomaly-alerts'
        title: 'Pynomaly Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@company.com'
        subject: 'CRITICAL: Pynomaly Production Alert'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ if .Annotations.runbook_url }}Runbook: {{ .Annotations.runbook_url }}{{ end }}
          {{ end }}
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#pynomaly-critical'
        title: 'CRITICAL ALERT - Pynomaly Production'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        color: 'danger'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'warning-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#pynomaly-alerts'
        title: 'Warning - Pynomaly Production'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        color: 'warning'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
```

---

## Security Hardening Guide

### System-Level Security

#### 1. Operating System Hardening

```bash
#!/bin/bash
# OS Security Hardening Script

# Update system packages
apt update && apt upgrade -y

# Install security tools
apt install -y fail2ban ufw aide rkhunter chkrootkit

# Configure automatic security updates
echo 'Unattended-Upgrade::Automatic-Reboot "false";' >> /etc/apt/apt.conf.d/50unattended-upgrades
systemctl enable unattended-upgrades

# Disable unnecessary services
systemctl disable bluetooth
systemctl disable cups
systemctl disable avahi-daemon

# Configure SSH security
sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/#MaxAuthTries 6/MaxAuthTries 3/' /etc/ssh/sshd_config
systemctl restart sshd

# Configure firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw --force enable

# Set up fail2ban
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
backend = systemd

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-dos]
enabled = true
filter = nginx-dos
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 10
findtime = 60
bantime = 600
EOF

systemctl enable fail2ban
systemctl start fail2ban
```

#### 2. Container Security

```dockerfile
# Dockerfile security best practices
FROM python:3.11-slim AS builder

# Create non-root user
RUN groupadd -r pynomaly && useradd -r -g pynomaly pynomaly

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Create non-root user
RUN groupadd -r pynomaly && useradd -r -g pynomaly pynomaly

# Install runtime dependencies and security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copy dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=pynomaly:pynomaly src/ ./src/
COPY --chown=pynomaly:pynomaly scripts/ ./scripts/

# Set proper permissions
RUN chmod -R 755 /app && \
    chmod -R 644 /app/src/ && \
    chmod +x /app/scripts/*.py

# Switch to non-root user
USER pynomaly

# Security: Remove setuid/setgid permissions
RUN find /app -type f \( -perm -4000 -o -perm -2000 \) -exec chmod -s {} \;

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "src.pynomaly.presentation.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 3. Application Security Configuration

```python
# src/pynomaly/infrastructure/config/security.py
from typing import List, Dict, Any
import os

class SecurityConfig:
    """Production security configuration"""
    
    # Authentication
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production")
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # Password policy
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_NUMBERS = True
    PASSWORD_REQUIRE_SPECIAL = True
    PASSWORD_MAX_AGE_DAYS = 90
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE = 60
    RATE_LIMIT_PER_HOUR = 1000
    RATE_LIMIT_PER_DAY = 10000
    
    # CORS settings
    CORS_ORIGINS = [
        "https://pynomaly.com",
        "https://app.pynomaly.com",
        "https://admin.pynomaly.com"
    ]
    CORS_ALLOW_CREDENTIALS = True
    CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS = ["*"]
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    }
    
    # Input validation
    MAX_REQUEST_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_FILE_SIZE = 100 * 1024 * 1024    # 100MB
    ALLOWED_FILE_EXTENSIONS = [".csv", ".json", ".parquet", ".xlsx"]
    
    # Encryption
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
    DATABASE_ENCRYPTION_ENABLED = True
    
    # Audit logging
    AUDIT_LOG_ENABLED = True
    AUDIT_LOG_SENSITIVE_FIELDS = ["password", "token", "api_key"]
    
    # IP filtering
    ALLOWED_IP_RANGES = [
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16"
    ]
    
    @classmethod
    def validate_configuration(cls) -> List[str]:
        """Validate security configuration"""
        issues = []
        
        if cls.JWT_SECRET_KEY == "change-this-in-production":
            issues.append("JWT_SECRET_KEY must be changed in production")
            
        if not cls.ENCRYPTION_KEY:
            issues.append("ENCRYPTION_KEY must be set")
            
        if len(cls.JWT_SECRET_KEY) < 32:
            issues.append("JWT_SECRET_KEY should be at least 32 characters")
            
        return issues
```

#### 4. NGINX Security Configuration

```nginx
# /etc/nginx/sites-available/pynomaly-production
server {
    listen 80;
    server_name pynomaly.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name pynomaly.example.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/pynomaly.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/pynomaly.example.com/privkey.pem;
    
    # SSL Security
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'" always;
    
    # Hide server information
    server_tokens off;
    
    # Rate limiting zones
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/m;
    limit_req_zone $binary_remote_addr zone=general:10m rate=100r/m;
    
    # Request size limits
    client_max_body_size 50M;
    client_body_buffer_size 1M;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    
    # Timeout settings
    client_body_timeout 12;
    client_header_timeout 12;
    keepalive_timeout 15;
    send_timeout 10;
    
    # Hide upstream errors
    proxy_intercept_errors on;
    
    # Upstream configuration
    upstream pynomaly_backend {
        least_conn;
        server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
        server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
        server 127.0.0.1:8003 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    # Block common attack patterns
    location ~* (\.php|\.asp|\.aspx|\.jsp)$ {
        return 444;
    }
    
    # Block file access
    location ~* \.(htaccess|htpasswd|ini|log|sh|sql|conf)$ {
        deny all;
    }
    
    # API endpoints with rate limiting
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        limit_req_status 429;
        
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }
    
    # Stricter rate limiting for authentication
    location /api/v1/auth/ {
        limit_req zone=login burst=3 nodelay;
        limit_req_status 429;
        
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Health check (no rate limiting)
    location /health {
        access_log off;
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
    }
    
    # Static files with caching
    location /static/ {
        alias /var/www/pynomaly/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        add_header X-Content-Type-Options "nosniff";
        
        # Security for static files
        location ~* \.(js|css)$ {
            add_header Content-Security-Policy "default-src 'self'";
        }
    }
    
    # Default rate limiting for other requests
    location / {
        limit_req zone=general burst=100 nodelay;
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Backup and Recovery Procedures

### Database Backup Strategy

#### 1. Automated PostgreSQL Backup Script

```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup_database.sh

set -euo pipefail

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-pynomaly}"
DB_USER="${DB_USER:-pynomaly_user}"
BACKUP_DIR="${BACKUP_DIR:-/opt/backups/postgres}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
S3_BUCKET="${S3_BUCKET:-pynomaly-backups}"
NOTIFICATION_EMAIL="${NOTIFICATION_EMAIL:-admin@pynomaly.com}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="pynomaly_backup_${TIMESTAMP}.sql"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_DIR/backup.log"
}

# Error handling
handle_error() {
    log "ERROR: Backup failed at line $1"
    echo "Database backup failed at $(date)" | mail -s "Pynomaly Backup Failed" "$NOTIFICATION_EMAIL"
    exit 1
}

trap 'handle_error $LINENO' ERR

log "Starting database backup for $DB_NAME"

# Check database connectivity
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"; then
    log "ERROR: Cannot connect to database"
    exit 1
fi

# Create backup
log "Creating backup: $BACKUP_FILE"
pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
    --verbose \
    --no-password \
    --format=custom \
    --compress=9 \
    --no-owner \
    --no-privileges \
    > "$BACKUP_PATH"

# Verify backup
if [ ! -f "$BACKUP_PATH" ] || [ ! -s "$BACKUP_PATH" ]; then
    log "ERROR: Backup file is empty or does not exist"
    exit 1
fi

# Calculate checksums
md5sum "$BACKUP_PATH" > "${BACKUP_PATH}.md5"
sha256sum "$BACKUP_PATH" > "${BACKUP_PATH}.sha256"

BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
log "Backup completed successfully. Size: $BACKUP_SIZE"

# Compress backup
log "Compressing backup"
gzip "$BACKUP_PATH"
COMPRESSED_BACKUP="${BACKUP_PATH}.gz"

# Upload to S3 (if configured)
if [ -n "$S3_BUCKET" ]; then
    log "Uploading backup to S3: s3://$S3_BUCKET/postgres/"
    aws s3 cp "$COMPRESSED_BACKUP" "s3://$S3_BUCKET/postgres/"
    aws s3 cp "${BACKUP_PATH}.md5" "s3://$S3_BUCKET/postgres/"
    aws s3 cp "${BACKUP_PATH}.sha256" "s3://$S3_BUCKET/postgres/"
    log "Upload to S3 completed"
fi

# Clean up old backups
log "Cleaning up old backups (older than $RETENTION_DAYS days)"
find "$BACKUP_DIR" -name "pynomaly_backup_*.sql.gz" -mtime +"$RETENTION_DAYS" -delete
find "$BACKUP_DIR" -name "pynomaly_backup_*.md5" -mtime +"$RETENTION_DAYS" -delete
find "$BACKUP_DIR" -name "pynomaly_backup_*.sha256" -mtime +"$RETENTION_DAYS" -delete

# Clean up S3 old backups
if [ -n "$S3_BUCKET" ]; then
    aws s3api list-objects-v2 --bucket "$S3_BUCKET" --prefix "postgres/" \
        --query "Contents[?LastModified<='$(date -d "$RETENTION_DAYS days ago" --iso-8601)'].Key" \
        --output text | xargs -r -n1 aws s3 rm "s3://$S3_BUCKET/"
fi

log "Database backup completed successfully"

# Send success notification
echo "Database backup completed successfully at $(date). Size: $BACKUP_SIZE" | \
    mail -s "Pynomaly Backup Success" "$NOTIFICATION_EMAIL"
```

#### 2. Application Data Backup Script

```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup_application.sh

set -euo pipefail

# Configuration
APP_DIR="${APP_DIR:-/opt/pynomaly}"
BACKUP_DIR="${BACKUP_DIR:-/opt/backups/application}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
S3_BUCKET="${S3_BUCKET:-pynomaly-backups}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="pynomaly_app_${TIMESTAMP}.tar.gz"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_DIR/backup.log"
}

log "Starting application backup"

# Create backup excluding unnecessary files
tar -czf "$BACKUP_PATH" \
    -C "$(dirname "$APP_DIR")" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='*.log' \
    --exclude='tmp' \
    --exclude='cache' \
    "$(basename "$APP_DIR")"

# Verify backup
if [ ! -f "$BACKUP_PATH" ] || [ ! -s "$BACKUP_PATH" ]; then
    log "ERROR: Backup file is empty or does not exist"
    exit 1
fi

BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
log "Application backup completed. Size: $BACKUP_SIZE"

# Upload to S3
if [ -n "$S3_BUCKET" ]; then
    log "Uploading to S3: s3://$S3_BUCKET/application/"
    aws s3 cp "$BACKUP_PATH" "s3://$S3_BUCKET/application/"
fi

# Clean up old backups
find "$BACKUP_DIR" -name "pynomaly_app_*.tar.gz" -mtime +"$RETENTION_DAYS" -delete

log "Application backup completed successfully"
```

#### 3. Redis Backup Script

```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup_redis.sh

set -euo pipefail

# Configuration
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
BACKUP_DIR="${BACKUP_DIR:-/opt/backups/redis}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="redis_backup_${TIMESTAMP}.rdb"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_DIR/backup.log"
}

log "Starting Redis backup"

# Create Redis backup
if [ -n "$REDIS_PASSWORD" ]; then
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --rdb "$BACKUP_PATH"
else
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --rdb "$BACKUP_PATH"
fi

# Verify backup
if [ ! -f "$BACKUP_PATH" ] || [ ! -s "$BACKUP_PATH" ]; then
    log "ERROR: Redis backup file is empty or does not exist"
    exit 1
fi

BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
log "Redis backup completed. Size: $BACKUP_SIZE"

# Compress backup
gzip "$BACKUP_PATH"

# Clean up old backups
find "$BACKUP_DIR" -name "redis_backup_*.rdb.gz" -mtime +"$RETENTION_DAYS" -delete

log "Redis backup completed successfully"
```

### Recovery Procedures

#### 1. Database Recovery Script

```bash
#!/bin/bash
# /opt/pynomaly/scripts/restore_database.sh

set -euo pipefail

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-pynomaly}"
DB_USER="${DB_USER:-pynomaly_user}"
BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    echo "Example: $0 /opt/backups/postgres/pynomaly_backup_20240101_120000.sql.gz"
    exit 1
fi

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting database restore from: $BACKUP_FILE"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    log "ERROR: Backup file does not exist: $BACKUP_FILE"
    exit 1
fi

# Check if backup file is compressed
if [[ "$BACKUP_FILE" == *.gz ]]; then
    log "Decompressing backup file"
    DECOMPRESSED_FILE="${BACKUP_FILE%.gz}"
    gunzip -c "$BACKUP_FILE" > "$DECOMPRESSED_FILE"
    RESTORE_FILE="$DECOMPRESSED_FILE"
else
    RESTORE_FILE="$BACKUP_FILE"
fi

# Verify checksums if available
if [ -f "${BACKUP_FILE%.gz}.md5" ]; then
    log "Verifying MD5 checksum"
    if ! md5sum -c "${BACKUP_FILE%.gz}.md5"; then
        log "ERROR: MD5 checksum verification failed"
        exit 1
    fi
fi

# Stop application services
log "Stopping application services"
systemctl stop pynomaly-api || true
docker-compose -f /opt/pynomaly/docker-compose.production.yml stop pynomaly-api || true

# Create backup of current database
log "Creating backup of current database"
CURRENT_BACKUP="/tmp/pynomaly_current_backup_$(date +%Y%m%d_%H%M%S).sql"
pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" > "$CURRENT_BACKUP"

# Drop and recreate database
log "Dropping and recreating database"
dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"

# Restore database
log "Restoring database from backup"
if [[ "$RESTORE_FILE" == *.sql ]]; then
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESTORE_FILE"
else
    pg_restore -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" "$RESTORE_FILE"
fi

# Run database migrations
log "Running database migrations"
cd /opt/pynomaly
python -m alembic upgrade head

# Start application services
log "Starting application services"
systemctl start pynomaly-api || true
docker-compose -f /opt/pynomaly/docker-compose.production.yml start pynomaly-api || true

# Wait for services to start
sleep 30

# Verify restoration
log "Verifying database restoration"
if curl -f http://localhost:8000/health; then
    log "Database restoration completed successfully"
else
    log "ERROR: Health check failed after restoration"
    log "Current database backup saved at: $CURRENT_BACKUP"
    exit 1
fi

# Clean up temporary files
rm -f "$DECOMPRESSED_FILE" 2>/dev/null || true

log "Database restore completed successfully"
```

#### 2. Disaster Recovery Runbook

```markdown
# Disaster Recovery Runbook

## Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)

- **RTO**: 4 hours for complete system recovery
- **RPO**: 6 hours maximum data loss
- **Critical services RTO**: 1 hour
- **Database RPO**: 1 hour (with streaming replication)

## Disaster Scenarios and Procedures

### Scenario 1: Complete Data Center Failure

1. **Immediate Response (0-15 minutes)**
   - Activate disaster recovery team
   - Assess scope of failure
   - Redirect traffic to backup data center
   - Notify stakeholders

2. **Recovery Actions (15 minutes - 2 hours)**
   ```bash
   # Spin up backup infrastructure
   terraform apply -var-file="disaster-recovery.tfvars"
   
   # Restore latest database backup
   ./scripts/restore_database.sh s3://pynomaly-backups/postgres/latest.sql.gz
   
   # Deploy application to new infrastructure
   ./scripts/deploy_disaster_recovery.sh
   
   # Update DNS to point to new infrastructure
   ./scripts/update_dns_failover.sh
   ```

3. **Verification (2-4 hours)**
   - Verify all services are operational
   - Test critical user journeys
   - Monitor system performance
   - Validate data integrity

### Scenario 2: Database Corruption

1. **Immediate Response**

   ```bash
   # Stop all write operations
   systemctl stop pynomaly-api
   
   # Assess corruption extent
   pg_dump --schema-only pynomaly > schema_check.sql
   ```

2. **Recovery Actions**

   ```bash
   # Restore from latest backup
   ./scripts/restore_database.sh /opt/backups/postgres/latest.sql.gz
   
   # Apply any missing transactions from WAL
   pg_waldump /var/lib/postgresql/14/main/pg_wal/000000010000000000000001
   ```

### Scenario 3: Application Server Failure

1. **Immediate Response**

   ```bash
   # Remove failed server from load balancer
   nginx -s reload
   
   # Check remaining servers
   docker-compose ps
   ```

2. **Recovery Actions**

   ```bash
   # Scale up remaining servers
   docker-compose up -d --scale pynomaly-api=4
   
   # Deploy to new server
   ./scripts/deploy_single_server.sh new-server-ip
   ```

## Communication Plan

### Stakeholder Notification Matrix

| Incident Level | Stakeholders | Method | Timeline |
|----------------|--------------|--------|----------|
| Critical | All | Email, Slack, Phone | 15 minutes |
| High | Technical team, Management | Email, Slack | 30 minutes |
| Medium | Technical team | Slack | 1 hour |

### Communication Templates

**Critical Incident Notification:**

```
Subject: CRITICAL - Pynomaly Production Outage

We are experiencing a production outage affecting all Pynomaly services.

- Incident Start Time: [TIME]
- Affected Services: [SERVICES]
- Estimated Recovery Time: [TIME]
- Current Status: [STATUS]

Our team is actively working on resolution. Updates will be provided every 30 minutes.

Contact: [CONTACT INFO]
```

```

---

## Performance Tuning Guide

### Database Performance Optimization

#### 1. PostgreSQL Configuration Tuning

```sql
-- /etc/postgresql/15/main/postgresql.conf
-- Memory settings (for 32GB RAM server)
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 256MB
maintenance_work_mem = 2GB
wal_buffers = 64MB

-- Connection settings
max_connections = 200
max_prepared_transactions = 200

-- Checkpoint settings
checkpoint_completion_target = 0.9
checkpoint_timeout = 15min
max_wal_size = 4GB
min_wal_size = 1GB

-- Query planner settings
default_statistics_target = 100
random_page_cost = 1.1  # For SSD storage
effective_io_concurrency = 200

-- Background writer settings
bgwriter_delay = 200ms
bgwriter_lru_maxpages = 100
bgwriter_lru_multiplier = 2.0

-- WAL settings for better performance
wal_level = replica
max_wal_senders = 3
wal_keep_size = 64MB
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'

-- Vacuum settings
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 1min
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.2
autovacuum_analyze_scale_factor = 0.1

-- Lock settings
deadlock_timeout = 1s
lock_timeout = 5s
statement_timeout = 30s

-- Logging for performance monitoring
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_statement = 'ddl'
```

#### 2. Database Index Optimization

```sql
-- Performance indexes for Pynomaly tables

-- Users table indexes
CREATE INDEX CONCURRENTLY idx_users_email_active 
ON users(email, is_active) WHERE is_active = true;

CREATE INDEX CONCURRENTLY idx_users_created_at 
ON users(created_at DESC);

CREATE INDEX CONCURRENTLY idx_users_last_login 
ON users(last_login DESC) WHERE last_login IS NOT NULL;

-- API keys table indexes
CREATE INDEX CONCURRENTLY idx_api_keys_user_id_active 
ON api_keys(user_id, is_active) WHERE is_active = true;

CREATE INDEX CONCURRENTLY idx_api_keys_key_hash 
ON api_keys USING hash(key_hash);

CREATE INDEX CONCURRENTLY idx_api_keys_expires_at 
ON api_keys(expires_at) WHERE expires_at IS NOT NULL;

-- Audit logs table indexes
CREATE INDEX CONCURRENTLY idx_audit_logs_user_timestamp 
ON audit_logs(user_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_audit_logs_resource_type 
ON audit_logs(resource_type, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_audit_logs_action_status 
ON audit_logs(action, status, timestamp DESC);

-- Partial index for failed operations
CREATE INDEX CONCURRENTLY idx_audit_logs_failures 
ON audit_logs(timestamp DESC) WHERE status = 'FAILURE';

-- Detection results table indexes
CREATE INDEX CONCURRENTLY idx_detections_model_timestamp 
ON detection_results(model_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_detections_user_timestamp 
ON detection_results(user_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_detections_anomaly_score 
ON detection_results(anomaly_score DESC) WHERE is_anomaly = true;

-- Composite index for common queries
CREATE INDEX CONCURRENTLY idx_detections_user_model_time 
ON detection_results(user_id, model_id, created_at DESC);

-- Model performance metrics indexes
CREATE INDEX CONCURRENTLY idx_model_metrics_model_timestamp 
ON model_metrics(model_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_model_metrics_accuracy 
ON model_metrics(accuracy DESC, timestamp DESC);

-- Session management indexes
CREATE INDEX CONCURRENTLY idx_sessions_user_expires 
ON user_sessions(user_id, expires_at) WHERE expires_at > NOW();

CREATE INDEX CONCURRENTLY idx_sessions_token_hash 
ON user_sessions USING hash(token_hash);
```

#### 3. Application Performance Configuration

```python
# src/pynomaly/infrastructure/config/performance.py
from typing import Dict, Any
import os

class PerformanceConfig:
    """Production performance configuration"""
    
    # Database connection pool settings
    DATABASE_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
    DATABASE_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "30"))
    DATABASE_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DATABASE_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    DATABASE_POOL_PRE_PING = True
    
    # Redis connection pool settings
    REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
    REDIS_SOCKET_KEEPALIVE = True
    REDIS_SOCKET_KEEPALIVE_OPTIONS = {
        "TCP_KEEPINTVL": 1,
        "TCP_KEEPCNT": 3,
        "TCP_KEEPIDLE": 1
    }
    REDIS_CONNECTION_TIMEOUT = 30
    REDIS_SOCKET_TIMEOUT = 30
    
    # Caching settings
    CACHE_DEFAULT_TIMEOUT = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
    CACHE_REDIS_DB = 1
    CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "10000"))
    
    # API rate limiting
    RATE_LIMIT_STORAGE_URL = os.getenv("REDIS_URL", "redis://localhost:6379/2")
    RATE_LIMIT_ENABLED = True
    
    # Background task settings
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/3")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/4")
    CELERY_WORKER_CONCURRENCY = int(os.getenv("CELERY_WORKERS", "4"))
    CELERY_TASK_SOFT_TIME_LIMIT = 300  # 5 minutes
    CELERY_TASK_TIME_LIMIT = 600       # 10 minutes
    
    # Machine Learning performance settings
    ML_MODEL_CACHE_SIZE = int(os.getenv("ML_CACHE_SIZE", "5"))
    ML_BATCH_SIZE = int(os.getenv("ML_BATCH_SIZE", "1000"))
    ML_WORKER_PROCESSES = int(os.getenv("ML_WORKERS", "2"))
    ML_MEMORY_LIMIT_GB = int(os.getenv("ML_MEMORY_LIMIT", "4"))
    
    # File processing settings
    FILE_UPLOAD_MAX_SIZE = 100 * 1024 * 1024  # 100MB
    FILE_CHUNK_SIZE = 1024 * 1024              # 1MB chunks
    FILE_PROCESSING_TIMEOUT = 300              # 5 minutes
    
    # API response optimization
    RESPONSE_COMPRESSION_ENABLED = True
    RESPONSE_COMPRESSION_LEVEL = 6
    RESPONSE_CACHE_HEADERS = {
        "Cache-Control": "public, max-age=300",
        "Vary": "Accept-Encoding"
    }
    
    # Async settings
    ASYNC_POOL_SIZE = int(os.getenv("ASYNC_POOL_SIZE", "100"))
    ASYNC_TIMEOUT = int(os.getenv("ASYNC_TIMEOUT", "30"))
    
    @classmethod
    def get_database_url_with_pool(cls) -> str:
        """Get database URL with connection pool parameters"""
        base_url = os.getenv("DATABASE_URL", "postgresql://localhost/pynomaly")
        pool_params = (
            f"?pool_size={cls.DATABASE_POOL_SIZE}"
            f"&max_overflow={cls.DATABASE_MAX_OVERFLOW}"
            f"&pool_timeout={cls.DATABASE_POOL_TIMEOUT}"
            f"&pool_recycle={cls.DATABASE_POOL_RECYCLE}"
            f"&pool_pre_ping={'true' if cls.DATABASE_POOL_PRE_PING else 'false'}"
        )
        return f"{base_url}{pool_params}"
```

#### 4. NGINX Performance Tuning

```nginx
# /etc/nginx/nginx.conf - Performance optimized configuration

user www-data;
worker_processes auto;
worker_cpu_affinity auto;
worker_rlimit_nofile 65535;
pid /run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
    accept_mutex off;
}

http {
    # Basic settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 1000;
    types_hash_max_size 2048;
    server_tokens off;
    
    # Buffer settings
    client_body_buffer_size 1M;
    client_max_body_size 50M;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    
    # Timeout settings
    client_body_timeout 12;
    client_header_timeout 12;
    send_timeout 10;
    
    # Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # Brotli compression (if module available)
    brotli on;
    brotli_comp_level 6;
    brotli_types
        text/plain
        text/css
        application/json
        application/javascript
        text/xml
        application/xml
        application/xml+rss
        text/javascript;
    
    # Open file cache
    open_file_cache max=10000 inactive=20s;
    open_file_cache_valid 30s;
    open_file_cache_min_uses 2;
    open_file_cache_errors on;
    
    # Proxy cache settings
    proxy_cache_path /var/cache/nginx/pynomaly 
                     levels=1:2 
                     keys_zone=pynomaly_cache:10m 
                     max_size=1g 
                     inactive=60m 
                     use_temp_path=off;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/m;
    limit_req_zone $binary_remote_addr zone=general:10m rate=100r/m;
    
    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;
    limit_conn_zone $server_name zone=conn_limit_per_server:10m;
    
    # Upstream configuration with load balancing
    upstream pynomaly_backend {
        least_conn;
        keepalive 32;
        keepalive_requests 1000;
        keepalive_timeout 60s;
        
        server 127.0.0.1:8001 max_fails=3 fail_timeout=30s weight=1;
        server 127.0.0.1:8002 max_fails=3 fail_timeout=30s weight=1;
        server 127.0.0.1:8003 max_fails=3 fail_timeout=30s weight=1;
        server 127.0.0.1:8004 max_fails=3 fail_timeout=30s weight=1;
    }
    
    # Main server block
    server {
        listen 443 ssl http2 reuseport;
        server_name pynomaly.example.com;
        
        # Connection limits
        limit_conn conn_limit_per_ip 10;
        limit_conn conn_limit_per_server 1000;
        
        # SSL optimization
        ssl_session_cache shared:SSL:50m;
        ssl_session_timeout 24h;
        ssl_buffer_size 4k;
        ssl_stapling on;
        ssl_stapling_verify on;
        
        # API endpoints with caching
        location /api/v1/ {
            limit_req zone=api burst=20 nodelay;
            
            # Proxy settings
            proxy_pass http://pynomaly_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout settings
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            
            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
            proxy_busy_buffers_size 8k;
            proxy_temp_file_write_size 8k;
            
            # Cache for GET requests (specific endpoints only)
            location ~* ^/api/v1/(models|algorithms|health)$ {
                proxy_cache pynomaly_cache;
                proxy_cache_valid 200 5m;
                proxy_cache_valid 404 1m;
                proxy_cache_use_stale error timeout invalid_header updating;
                proxy_cache_lock on;
                add_header X-Cache-Status $upstream_cache_status;
                
                proxy_pass http://pynomaly_backend;
            }
        }
        
        # Static files with aggressive caching
        location /static/ {
            alias /var/www/pynomaly/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
            add_header Vary "Accept-Encoding";
            
            # Pre-compressed files
            location ~* \.(js|css)$ {
                gzip_static on;
                brotli_static on;
            }
        }
        
        # WebSocket proxying for real-time features
        location /ws/ {
            proxy_pass http://pynomaly_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket specific timeouts
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;
        }
    }
}
```

#### 5. Redis Performance Optimization

```ini
# /etc/redis/redis.conf - Performance optimized configuration

# Network settings
port 6379
tcp-backlog 511
timeout 0
tcp-keepalive 300
bind 127.0.0.1 ::1

# Memory management
maxmemory 4gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistence settings (optimized for performance)
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis

# AOF settings
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
aof-use-rdb-preamble yes

# Client connection settings
maxclients 10000

# Advanced settings
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
hll-sparse-max-bytes 3000

# Threading (Redis 6+)
io-threads 4
io-threads-do-reads yes

# Latency settings
latency-monitor-threshold 100

# Security
requirepass your_secure_redis_password_here

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log
```

---

## Troubleshooting Procedures

### Common Issues and Solutions

#### 1. API Response Time Issues

**Symptoms:**

- API response times > 1 second
- High CPU usage on application servers
- Database connection pool exhaustion

**Diagnosis Steps:**

```bash
# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost/api/v1/health"

# Monitor database connections
sudo -u postgres psql -c "SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active';"

# Check application metrics
curl http://localhost:8000/metrics | grep http_request_duration

# Monitor system resources
htop
iotop
```

**Solutions:**

```python
# Optimize database queries
from sqlalchemy import text

# Use indexed queries
query = text("""
    SELECT * FROM detection_results 
    WHERE user_id = :user_id 
    AND created_at >= :start_date 
    ORDER BY created_at DESC 
    LIMIT :limit
""")

# Implement query result caching
from functools import lru_cache

@lru_cache(maxsize=128)
def get_user_models(user_id: int):
    return db.query(Model).filter(Model.user_id == user_id).all()

# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

#### 2. Memory Leaks

**Symptoms:**

- Gradually increasing memory usage
- Application crashes with OOM errors
- Slow garbage collection

**Diagnosis Steps:**

```bash
# Monitor memory usage over time
while true; do
    echo "$(date): $(docker stats --no-stream pynomaly-api | tail -n +2 | awk '{print $3}')"
    sleep 60
done

# Check Python memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# Profile memory usage
pip install memory_profiler
python -m memory_profiler your_script.py
```

**Solutions:**

```python
# Implement proper resource cleanup
import gc
from contextlib import contextmanager

@contextmanager
def ml_model_context(model_name: str):
    model = load_model(model_name)
    try:
        yield model
    finally:
        del model
        gc.collect()

# Use generators for large datasets
def process_large_dataset(file_path: str):
    with open(file_path, 'r') as f:
        for line in f:
            yield process_line(line)

# Implement memory limits
import resource

def set_memory_limit(limit_gb: int):
    limit_bytes = limit_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
```

#### 3. Database Performance Issues

**Symptoms:**

- Slow query execution
- Database connection timeouts
- High disk I/O usage

**Diagnosis Steps:**

```sql
-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check database connections
SELECT count(*) as connections, state
FROM pg_stat_activity 
GROUP BY state;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats 
WHERE tablename = 'detection_results';

-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

**Solutions:**

```sql
-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_detection_results_user_created 
ON detection_results(user_id, created_at DESC);

-- Optimize queries with proper WHERE clauses
EXPLAIN ANALYZE SELECT * FROM detection_results 
WHERE user_id = 123 
AND created_at >= '2024-01-01'::date;

-- Implement table partitioning for large tables
CREATE TABLE detection_results_2024_01 PARTITION OF detection_results
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Regular maintenance
VACUUM ANALYZE detection_results;
REINDEX INDEX CONCURRENTLY idx_detection_results_user_created;
```

#### 4. Redis Connection Issues

**Symptoms:**

- Redis connection timeouts
- Session data loss
- Cache misses

**Diagnosis Steps:**

```bash
# Check Redis status
redis-cli ping

# Monitor Redis performance
redis-cli --latency
redis-cli --latency-history -i 1

# Check Redis memory usage
redis-cli info memory

# Monitor Redis connections
redis-cli info clients
```

**Solutions:**

```python
# Implement connection retry logic
import redis
from redis.connection import ConnectionPool
from redis.retry import Retry
from redis.backoff import ExponentialBackoff

pool = ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=50,
    retry=Retry(ExponentialBackoff(), 3),
    health_check_interval=30
)

redis_client = redis.Redis(connection_pool=pool)

# Implement circuit breaker pattern
class RedisCircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            raise e
```

### Incident Response Procedures

#### 1. Incident Classification

| Severity | Description | Response Time | Escalation |
|----------|-------------|---------------|------------|
| P0 - Critical | Complete service outage | 15 minutes | Immediate |
| P1 - High | Major feature degradation | 1 hour | 30 minutes |
| P2 - Medium | Minor feature issues | 4 hours | 2 hours |
| P3 - Low | Cosmetic issues | 24 hours | Next business day |

#### 2. Incident Response Runbook

```bash
#!/bin/bash
# /opt/pynomaly/scripts/incident_response.sh

set -euo pipefail

INCIDENT_LEVEL="$1"
INCIDENT_DESCRIPTION="$2"

case "$INCIDENT_LEVEL" in
    "P0"|"critical")
        echo "🚨 CRITICAL INCIDENT DETECTED"
        
        # Immediate actions
        ./scripts/health_check_all.sh
        ./scripts/notify_oncall.sh "CRITICAL" "$INCIDENT_DESCRIPTION"
        ./scripts/enable_maintenance_mode.sh
        
        # Scale up resources
        docker-compose -f docker-compose.production.yml up -d --scale pynomaly-api=6
        
        # Collect diagnostics
        ./scripts/collect_diagnostics.sh
        ;;
        
    "P1"|"high")
        echo "⚠️ HIGH PRIORITY INCIDENT"
        
        # Check system health
        ./scripts/health_check_all.sh
        ./scripts/notify_team.sh "HIGH" "$INCIDENT_DESCRIPTION"
        
        # Collect logs
        ./scripts/collect_logs.sh
        ;;
        
    "P2"|"medium")
        echo "📋 MEDIUM PRIORITY INCIDENT"
        
        ./scripts/notify_team.sh "MEDIUM" "$INCIDENT_DESCRIPTION"
        ./scripts/collect_logs.sh
        ;;
        
    *)
        echo "Invalid incident level: $INCIDENT_LEVEL"
        exit 1
        ;;
esac
```

#### 3. Diagnostic Data Collection

```bash
#!/bin/bash
# /opt/pynomaly/scripts/collect_diagnostics.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DIAG_DIR="/tmp/pynomaly_diagnostics_$TIMESTAMP"
mkdir -p "$DIAG_DIR"

echo "Collecting diagnostic data..."

# System information
uname -a > "$DIAG_DIR/system_info.txt"
df -h > "$DIAG_DIR/disk_usage.txt"
free -h > "$DIAG_DIR/memory_usage.txt"
ps aux --sort=-%cpu | head -20 > "$DIAG_DIR/top_processes.txt"

# Docker information
docker ps > "$DIAG_DIR/docker_containers.txt"
docker stats --no-stream > "$DIAG_DIR/docker_stats.txt"

# Application logs
docker-compose -f /opt/pynomaly/docker-compose.production.yml logs --tail=1000 pynomaly-api > "$DIAG_DIR/app_logs.txt"

# Database status
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;" > "$DIAG_DIR/db_connections.txt"
sudo -u postgres psql -c "SELECT * FROM pg_stat_database;" > "$DIAG_DIR/db_stats.txt"

# Redis status
redis-cli info > "$DIAG_DIR/redis_info.txt"

# Network status
ss -tulpn > "$DIAG_DIR/network_connections.txt"
iptables -L > "$DIAG_DIR/firewall_rules.txt"

# Create archive
tar -czf "/tmp/pynomaly_diagnostics_$TIMESTAMP.tar.gz" -C /tmp "pynomaly_diagnostics_$TIMESTAMP"
rm -rf "$DIAG_DIR"

echo "Diagnostic data collected: /tmp/pynomaly_diagnostics_$TIMESTAMP.tar.gz"
```

---

## Infrastructure as Code Examples

### Terraform Configuration

#### 1. AWS Infrastructure

```hcl
# terraform/aws/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC Configuration
resource "aws_vpc" "pynomaly" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name        = "pynomaly-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "pynomaly" {
  vpc_id = aws_vpc.pynomaly.id
  
  tags = {
    Name        = "pynomaly-igw"
    Environment = var.environment
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.pynomaly.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone = var.availability_zones[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {
    Name        = "pynomaly-public-${count.index + 1}"
    Environment = var.environment
    Type        = "public"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.pynomaly.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = var.availability_zones[count.index]
  
  tags = {
    Name        = "pynomaly-private-${count.index + 1}"
    Environment = var.environment
    Type        = "private"
  }
}

# NAT Gateways
resource "aws_eip" "nat" {
  count  = length(var.availability_zones)
  domain = "vpc"
  
  tags = {
    Name        = "pynomaly-nat-eip-${count.index + 1}"
    Environment = var.environment
  }
}

resource "aws_nat_gateway" "pynomaly" {
  count         = length(var.availability_zones)
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = {
    Name        = "pynomaly-nat-${count.index + 1}"
    Environment = var.environment
  }
  
  depends_on = [aws_internet_gateway.pynomaly]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.pynomaly.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.pynomaly.id
  }
  
  tags = {
    Name        = "pynomaly-public-rt"
    Environment = var.environment
  }
}

resource "aws_route_table" "private" {
  count  = length(var.availability_zones)
  vpc_id = aws_vpc.pynomaly.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.pynomaly[count.index].id
  }
  
  tags = {
    Name        = "pynomaly-private-rt-${count.index + 1}"
    Environment = var.environment
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Security Groups
resource "aws_security_group" "alb" {
  name_prefix = "pynomaly-alb-"
  vpc_id      = aws_vpc.pynomaly.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "pynomaly-alb-sg"
    Environment = var.environment
  }
}

resource "aws_security_group" "app" {
  name_prefix = "pynomaly-app-"
  vpc_id      = aws_vpc.pynomaly.id
  
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "pynomaly-app-sg"
    Environment = var.environment
  }
}

resource "aws_security_group" "database" {
  name_prefix = "pynomaly-db-"
  vpc_id      = aws_vpc.pynomaly.id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }
  
  tags = {
    Name        = "pynomaly-db-sg"
    Environment = var.environment
  }
}

# RDS Subnet Group
resource "aws_db_subnet_group" "pynomaly" {
  name       = "pynomaly-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id
  
  tags = {
    Name        = "pynomaly-db-subnet-group"
    Environment = var.environment
  }
}

# RDS Instance
resource "aws_db_instance" "pynomaly" {
  identifier             = "pynomaly-db"
  allocated_storage      = var.db_allocated_storage
  max_allocated_storage  = var.db_max_allocated_storage
  storage_type           = "gp3"
  storage_encrypted      = true
  engine                 = "postgres"
  engine_version         = var.db_engine_version
  instance_class         = var.db_instance_class
  db_name                = var.db_name
  username               = var.db_username
  password               = var.db_password
  port                   = 5432
  
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.pynomaly.name
  
  backup_retention_period = var.db_backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "pynomaly-db-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn
  
  tags = {
    Name        = "pynomaly-db"
    Environment = var.environment
  }
}

# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "pynomaly" {
  name       = "pynomaly-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
  
  tags = {
    Name        = "pynomaly-cache-subnet-group"
    Environment = var.environment
  }
}

# ElastiCache Redis Cluster
resource "aws_elasticache_replication_group" "pynomaly" {
  replication_group_id       = "pynomaly-redis"
  description                = "Redis cluster for Pynomaly"
  
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = var.redis_num_cache_nodes
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.pynomaly.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token
  
  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"
  
  tags = {
    Name        = "pynomaly-redis"
    Environment = var.environment
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "pynomaly-redis-"
  vpc_id      = aws_vpc.pynomaly.id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }
  
  tags = {
    Name        = "pynomaly-redis-sg"
    Environment = var.environment
  }
}

# Application Load Balancer
resource "aws_lb" "pynomaly" {
  name               = "pynomaly-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = aws_subnet.public[*].id
  
  enable_deletion_protection = var.environment == "production"
  
  tags = {
    Name        = "pynomaly-alb"
    Environment = var.environment
  }
}

# Target Group
resource "aws_lb_target_group" "pynomaly" {
  name     = "pynomaly-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.pynomaly.id
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
    port                = "traffic-port"
    protocol            = "HTTP"
  }
  
  tags = {
    Name        = "pynomaly-tg"
    Environment = var.environment
  }
}

# Launch Template
resource "aws_launch_template" "pynomaly" {
  name_prefix   = "pynomaly-"
  image_id      = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_pair_name
  
  vpc_security_group_ids = [aws_security_group.app.id]
  
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    db_host     = aws_db_instance.pynomaly.endpoint
    redis_host  = aws_elasticache_replication_group.pynomaly.primary_endpoint_address
    environment = var.environment
  }))
  
  iam_instance_profile {
    name = aws_iam_instance_profile.pynomaly.name
  }
  
  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "pynomaly-app"
      Environment = var.environment
    }
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "pynomaly" {
  name                = "pynomaly-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.pynomaly.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300
  
  min_size         = var.asg_min_size
  max_size         = var.asg_max_size
  desired_capacity = var.asg_desired_capacity
  
  launch_template {
    id      = aws_launch_template.pynomaly.id
    version = "$Latest"
  }
  
  tag {
    key                 = "Name"
    value               = "pynomaly-asg"
    propagate_at_launch = false
  }
  
  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
}

# Auto Scaling Policies
resource "aws_autoscaling_policy" "scale_up" {
  name                   = "pynomaly-scale-up"
  scaling_adjustment     = 2
  adjustment_type        = "ChangeInCapacity"
  cooldown              = 300
  autoscaling_group_name = aws_autoscaling_group.pynomaly.name
}

resource "aws_autoscaling_policy" "scale_down" {
  name                   = "pynomaly-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown              = 300
  autoscaling_group_name = aws_autoscaling_group.pynomaly.name
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "pynomaly-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "120"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.scale_up.arn]
  
  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.pynomaly.name
  }
}

resource "aws_cloudwatch_metric_alarm" "low_cpu" {
  alarm_name          = "pynomaly-low-cpu"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "120"
  statistic           = "Average"
  threshold           = "10"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.scale_down.arn]
  
  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.pynomaly.name
  }
}

# IAM Role for EC2 instances
resource "aws_iam_role" "pynomaly_instance" {
  name = "pynomaly-instance-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "pynomaly_instance" {
  name = "pynomaly-instance-policy"
  role = aws_iam_role.pynomaly_instance.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.pynomaly.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "pynomaly" {
  name = "pynomaly-instance-profile"
  role = aws_iam_role.pynomaly_instance.name
}

# S3 Bucket for application data
resource "aws_s3_bucket" "pynomaly" {
  bucket = "pynomaly-${var.environment}-${random_string.bucket_suffix.result}"
  
  tags = {
    Name        = "pynomaly-bucket"
    Environment = var.environment
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_versioning" "pynomaly" {
  bucket = aws_s3_bucket.pynomaly.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "pynomaly" {
  bucket = aws_s3_bucket.pynomaly.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "pynomaly" {
  bucket = aws_s3_bucket.pynomaly.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

#### 2. Kubernetes Deployment

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pynomaly-production
  labels:
    name: pynomaly-production
    environment: production

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pynomaly-config
  namespace: pynomaly-production
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  REDIS_URL: "redis://redis-service:6379/0"
  DATABASE_URL: "postgresql://pynomaly:$(DB_PASSWORD)@postgres-service:5432/pynomaly"
  PROMETHEUS_METRICS_ENABLED: "true"

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: pynomaly-secrets
  namespace: pynomaly-production
type: Opaque
data:
  DB_PASSWORD: <base64-encoded-password>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  ENCRYPTION_KEY: <base64-encoded-encryption-key>
  REDIS_PASSWORD: <base64-encoded-redis-password>

---
# k8s/postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: pynomaly-production
  labels:
    app: postgres
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: pynomaly
        - name: POSTGRES_USER
          value: pynomaly
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: DB_PASSWORD
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - pynomaly
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - pynomaly
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
# k8s/postgres-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: pynomaly-production
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 100Gi

---
# k8s/postgres-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: pynomaly-production
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP

---
# k8s/redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: pynomaly-production
  labels:
    app: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command:
        - redis-server
        - --requirepass
        - $(REDIS_PASSWORD)
        - --appendonly
        - "yes"
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: REDIS_PASSWORD
        volumeMounts:
        - name: redis-storage
          mountPath: /data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc

---
# k8s/redis-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: pynomaly-production
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 20Gi

---
# k8s/redis-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: pynomaly-production
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP

---
# k8s/pynomaly-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-api
  namespace: pynomaly-production
  labels:
    app: pynomaly-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: pynomaly-api
  template:
    metadata:
      labels:
        app: pynomaly-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: pynomaly-serviceaccount
      containers:
      - name: pynomaly-api
        image: pynomaly:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: pynomaly-config
              key: ENVIRONMENT
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: pynomaly-config
              key: LOG_LEVEL
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: pynomaly-config
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: pynomaly-config
              key: REDIS_URL
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: DB_PASSWORD
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: JWT_SECRET_KEY
        - name: ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: ENCRYPTION_KEY
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
        volumeMounts:
        - name: tmp-volume
          mountPath: /tmp
        - name: cache-volume
          mountPath: /app/cache
      volumes:
      - name: tmp-volume
        emptyDir: {}
      - name: cache-volume
        emptyDir: {}
      imagePullSecrets:
      - name: registry-secret

---
# k8s/pynomaly-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: pynomaly-api-service
  namespace: pynomaly-production
  labels:
    app: pynomaly-api
spec:
  selector:
    app: pynomaly-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP

---
# k8s/pynomaly-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-api-hpa
  namespace: pynomaly-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pynomaly-ingress
  namespace: pynomaly-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "30"
spec:
  tls:
  - hosts:
    - pynomaly.example.com
    secretName: pynomaly-tls
  rules:
  - host: pynomaly.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pynomaly-api-service
            port:
              number: 80

---
# k8s/serviceaccount.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pynomaly-serviceaccount
  namespace: pynomaly-production

---
# k8s/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: pynomaly-production
  name: pynomaly-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pynomaly-rolebinding
  namespace: pynomaly-production
subjects:
- kind: ServiceAccount
  name: pynomaly-serviceaccount
  namespace: pynomaly-production
roleRef:
  kind: Role
  name: pynomaly-role
  apiGroup: rbac.authorization.k8s.io

---
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pynomaly-network-policy
  namespace: pynomaly-production
spec:
  podSelector:
    matchLabels:
      app: pynomaly-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to: []
    ports:
    - protocol: TCP
      port: 443
```

This comprehensive production deployment guide addresses all the acceptance criteria from Issue #108:

- ✅ **Comprehensive deployment guide**: Complete step-by-step instructions
- ✅ **Production checklist**: Detailed pre/post deployment checklists
- ✅ **Monitoring setup**: Prometheus, Grafana, and alerting configuration
- ✅ **Security hardening guide**: System, application, and network security
- ✅ **Backup and recovery procedures**: Automated scripts and disaster recovery
- ✅ **Performance tuning guide**: Database, application, and infrastructure optimization
- ✅ **Troubleshooting procedures**: Common issues, diagnostics, and incident response
- ✅ **Infrastructure as code examples**: Terraform and Kubernetes configurations

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Fix CLI Help Formatting Issues (Issue #118)", "status": "completed", "priority": "high", "id": "24"}, {"content": "Fix Documentation Navigation and Broken Links (Issue #111)", "status": "completed", "priority": "high", "id": "25"}, {"content": "Update Documentation to Reflect Actual Implementation (Issue #107)", "status": "completed", "priority": "medium", "id": "26"}, {"content": "Create Production Deployment Guide (Issue #108)", "status": "completed", "priority": "medium", "id": "27"}, {"content": "Remove Placeholder and Stub Implementations (Issue #105)", "status": "completed", "priority": "medium", "id": "28"}, {"content": "Add AutoML Feature Flag Support (Issue #116)", "status": "pending", "priority": "medium", "id": "29"}, {"content": "Run link checker to identify broken links", "status": "completed", "priority": "high", "id": "30"}, {"content": "Fix broken internal documentation links", "status": "completed", "priority": "high", "id": "31"}, {"content": "Create missing core documentation directories", "status": "completed", "priority": "high", "id": "34"}, {"content": "Create quickstart.md in main docs directory", "status": "completed", "priority": "high", "id": "35"}, {"content": "Integrate orphaned documents into navigation", "status": "completed", "priority": "high", "id": "32"}, {"content": "Create automated link checking CI workflow", "status": "completed", "priority": "high", "id": "33"}, {"content": "Create accurate feature implementation status guide", "status": "completed", "priority": "high", "id": "36"}, {"content": "Update README with implementation reality", "status": "completed", "priority": "high", "id": "37"}, {"content": "Create feature flag documentation", "status": "pending", "priority": "medium", "id": "38"}, {"content": "Fix monitoring system placeholders in alerts.py", "status": "completed", "priority": "high", "id": "39"}, {"content": "Remove MLOps service placeholders in mlops_service.py", "status": "completed", "priority": "high", "id": "40"}, {"content": "Implement active learning placeholders in manage_active_learning.py", "status": "completed", "priority": "high", "id": "41"}, {"content": "Fix enterprise service resource tracking placeholders", "status": "completed", "priority": "medium", "id": "42"}, {"content": "Remove automated retraining placeholders", "status": "completed", "priority": "medium", "id": "43"}, {"content": "Fix synthetic data generation method implementations", "status": "completed", "priority": "medium", "id": "44"}]
