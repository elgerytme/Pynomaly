# Configuration Guide

Complete guide to configuring Pynomaly Web UI for your specific environment and requirements.

## 📋 Configuration Overview

Pynomaly Web UI can be configured through multiple methods:

- **Configuration Files** (YAML/JSON)
- **Environment Variables**
- **Command Line Arguments**
- **Web Interface Settings**
- **Database Configuration**

Configuration precedence (highest to lowest):

1. Command line arguments
2. Environment variables
3. Configuration files
4. Default values

## 📁 Configuration Files

### Main Configuration File

Create `~/.pynomaly/config.yaml` or `config/pynomaly.yaml`:

```yaml
# Application Settings
app:
  name: "Pynomaly Web UI"
  version: "1.0.0"
  debug: false
  testing: false
  
# Server Configuration
server:
  host: "localhost"
  port: 8000
  workers: 4
  reload: false
  access_log: true
  
# Database Configuration
database:
  url: "sqlite:///pynomaly.db"
  echo: false
  pool_size: 5
  max_overflow: 10
  pool_timeout: 30
  pool_recycle: 3600
  
# Cache Configuration
cache:
  type: "redis"  # redis, memory, or database
  url: "redis://localhost:6379/0"
  default_timeout: 300
  key_prefix: "pynomaly:"
  
# Security Settings
security:
  secret_key: "your-secret-key-here"
  algorithm: "HS256"
  access_token_expire_minutes: 30
  session_timeout: 3600
  csrf_protection: true
  
# Authentication
auth:
  enabled: true
  providers:
    local:
      enabled: true
    oauth:
      enabled: false
      providers: []
    ldap:
      enabled: false
      server: ""
      
# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/pynomaly.log"
  max_size: "100MB"
  backup_count: 5
  
# Feature Flags
features:
  automl: true
  ensemble: true
  explainability: true
  monitoring: true
  collaboration: true
  
# Performance Settings
performance:
  max_workers: 8
  chunk_size: 1000
  memory_limit: "4GB"
  cpu_limit: "80%"
  
# Monitoring & Alerts
monitoring:
  enabled: true
  metrics_retention: "30d"
  alert_rules:
    high_cpu: 85
    high_memory: 90
    error_rate: 5
    
# Data Processing
data:
  max_file_size: "500MB"
  allowed_formats: ["csv", "xlsx", "json", "parquet"]
  auto_type_detection: true
  preprocessing:
    handle_missing: "auto"
    scale_features: true
    encode_categorical: true
```

### Environment-Specific Configurations

Create separate files for different environments:

#### Development (`config/development.yaml`)

```yaml
app:
  debug: true
  testing: false
  
server:
  reload: true
  workers: 1
  
database:
  url: "sqlite:///dev_pynomaly.db"
  echo: true
  
logging:
  level: "DEBUG"
```

#### Production (`config/production.yaml`)

```yaml
app:
  debug: false
  testing: false
  
server:
  host: "0.0.0.0"
  workers: 8
  
database:
  url: "${DATABASE_URL}"
  
security:
  secret_key: "${SECRET_KEY}"
  
cache:
  url: "${REDIS_URL}"
  
logging:
  level: "WARNING"
  file: "/var/log/pynomaly/app.log"
```

#### Testing (`config/testing.yaml`)

```yaml
app:
  testing: true
  
database:
  url: "sqlite:///:memory:"
  
cache:
  type: "memory"
  
logging:
  level: "ERROR"
```

## 🌍 Environment Variables

### Core Environment Variables

```bash
# Application
export PYNOMALY_ENV="production"
export PYNOMALY_DEBUG="false"
export PYNOMALY_SECRET_KEY="your-super-secret-key"

# Server
export PYNOMALY_HOST="0.0.0.0"
export PYNOMALY_PORT="8000"
export PYNOMALY_WORKERS="4"

# Database
export PYNOMALY_DATABASE_URL="postgresql://user:pass@localhost/pynomaly"

# Cache
export PYNOMALY_REDIS_URL="redis://localhost:6379/0"

# Security
export PYNOMALY_CSRF_SECRET="csrf-secret-key"
export PYNOMALY_JWT_SECRET="jwt-secret-key"

# Features
export PYNOMALY_ENABLE_AUTOML="true"
export PYNOMALY_ENABLE_MONITORING="true"

# Logging
export PYNOMALY_LOG_LEVEL="INFO"
export PYNOMALY_LOG_FILE="/var/log/pynomaly.log"
```

### Docker Environment Variables

For Docker deployments:

```bash
# Docker-specific settings
export PYNOMALY_DOCKER="true"
export PYNOMALY_CONTAINER_NAME="pynomaly-web"

# Health checks
export PYNOMALY_HEALTH_CHECK_PATH="/health"
export PYNOMALY_HEALTH_CHECK_INTERVAL="30s"

# Resource limits
export PYNOMALY_MEMORY_LIMIT="2g"
export PYNOMALY_CPU_LIMIT="1.0"
```

## 🗄️ Database Configuration

### SQLite (Development)

```yaml
database:
  url: "sqlite:///pynomaly.db"
  echo: false
```

### PostgreSQL (Production)

```yaml
database:
  url: "postgresql://username:password@localhost:5432/pynomaly"
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
  echo: false
```

### MySQL

```yaml
database:
  url: "mysql+pymysql://username:password@localhost:3306/pynomaly"
  pool_size: 5
  max_overflow: 10
  pool_timeout: 30
```

### Database Migration Settings

```yaml
database:
  migrations:
    auto_upgrade: true
    backup_before_upgrade: true
    timeout: 300
```

## 🚀 Performance Configuration

### Worker Configuration

```yaml
server:
  workers: 4  # Number of worker processes
  worker_class: "uvicorn.workers.UvicornWorker"
  worker_connections: 1000
  max_requests: 1000
  max_requests_jitter: 100
  preload_app: true
```

### Memory Management

```yaml
performance:
  memory:
    limit: "4GB"
    warning_threshold: "80%"
    cleanup_threshold: "90%"
    gc_threshold: [700, 10, 10]
```

### Caching Configuration

```yaml
cache:
  type: "redis"
  url: "redis://localhost:6379/0"
  default_timeout: 300
  options:
    max_connections: 50
    retry_on_timeout: true
    socket_connect_timeout: 5
    socket_timeout: 5
```

### Background Tasks

```yaml
celery:
  broker_url: "redis://localhost:6379/1"
  result_backend: "redis://localhost:6379/2"
  task_routes:
    'pynomaly.tasks.training': {'queue': 'training'}
    'pynomaly.tasks.detection': {'queue': 'detection'}
  worker_concurrency: 4
```

## 🔒 Security Configuration

### Authentication Settings

```yaml
auth:
  enabled: true
  providers:
    local:
      enabled: true
      password_policy:
        min_length: 8
        require_uppercase: true
        require_lowercase: true
        require_numbers: true
        require_symbols: true
    oauth:
      enabled: true
      providers:
        google:
          client_id: "${GOOGLE_CLIENT_ID}"
          client_secret: "${GOOGLE_CLIENT_SECRET}"
        github:
          client_id: "${GITHUB_CLIENT_ID}"
          client_secret: "${GITHUB_CLIENT_SECRET}"
    ldap:
      enabled: false
      server: "ldap://ldap.example.com"
      bind_dn: "cn=admin,dc=example,dc=com"
      bind_password: "${LDAP_PASSWORD}"
      user_search_base: "ou=users,dc=example,dc=com"
```

### Security Headers

```yaml
security:
  headers:
    x_frame_options: "DENY"
    x_content_type_options: "nosniff"
    x_xss_protection: "1; mode=block"
    strict_transport_security: "max-age=31536000; includeSubDomains"
    content_security_policy: "default-src 'self'"
```

### Rate Limiting

```yaml
security:
  rate_limiting:
    enabled: true
    default_limits:
      per_minute: 60
      per_hour: 1000
      per_day: 10000
    endpoint_limits:
      "/api/auth/login":
        per_minute: 5
        per_hour: 20
      "/api/detection/run":
        per_minute: 10
        per_hour: 100
```

### WAF Configuration

```yaml
security:
  waf:
    enabled: true
    sql_injection_protection: true
    xss_protection: true
    path_traversal_protection: true
    command_injection_protection: true
    blocked_user_agents:
      - "sqlmap"
      - "nikto"
      - "nmap"
```

## 📊 Monitoring Configuration

### Metrics Collection

```yaml
monitoring:
  enabled: true
  metrics:
    system: true
    application: true
    custom: true
  exporters:
    prometheus:
      enabled: true
      port: 9090
      path: "/metrics"
    statsd:
      enabled: false
      host: "localhost"
      port: 8125
```

### Health Checks

```yaml
monitoring:
  health_checks:
    enabled: true
    checks:
      database: true
      cache: true
      disk_space: true
      memory: true
    intervals:
      fast: 30  # seconds
      slow: 300  # seconds
```

### Alerting

```yaml
monitoring:
  alerts:
    enabled: true
    channels:
      email:
        enabled: true
        smtp_server: "smtp.example.com"
        smtp_port: 587
        username: "${SMTP_USERNAME}"
        password: "${SMTP_PASSWORD}"
      slack:
        enabled: false
        webhook_url: "${SLACK_WEBHOOK_URL}"
    rules:
      high_cpu:
        threshold: 85
        duration: 300
        severity: "warning"
      high_memory:
        threshold: 90
        duration: 180
        severity: "critical"
```

## 🎨 UI Configuration

### Theme Settings

```yaml
ui:
  theme:
    default: "light"
    dark_mode_enabled: true
    custom_colors:
      primary: "#007bff"
      secondary: "#6c757d"
      success: "#28a745"
      warning: "#ffc107"
      danger: "#dc3545"
```

### Feature Toggles

```yaml
ui:
  features:
    dashboard_widgets:
      system_health: true
      recent_activity: true
      quick_actions: true
    advanced_features:
      automl: true
      ensemble: true
      explainability: true
    experimental_features:
      real_time_detection: false
      collaborative_editing: false
```

### Layout Options

```yaml
ui:
  layout:
    sidebar_width: 250
    max_content_width: 1200
    pagination_size: 25
    chart_default_height: 400
```

## 🔧 Advanced Configuration

### Custom Algorithms

```yaml
algorithms:
  custom:
    enabled: true
    discovery_paths:
      - "~/.pynomaly/algorithms"
      - "/opt/pynomaly/algorithms"
  builtin:
    isolation_forest:
      default_params:
        contamination: 0.1
        n_estimators: 100
    local_outlier_factor:
      default_params:
        contamination: 0.1
        n_neighbors: 20
```

### Data Connectors

```yaml
data:
  connectors:
    database:
      enabled: true
      timeout: 30
      pool_size: 5
    file:
      enabled: true
      max_size: "500MB"
      allowed_types: ["csv", "xlsx", "json"]
    api:
      enabled: true
      timeout: 60
      max_retries: 3
```

### Export Configuration

```yaml
export:
  formats:
    csv: true
    excel: true
    json: true
    pdf: true
  limits:
    max_rows: 1000000
    max_file_size: "1GB"
  templates:
    report: "templates/report.html"
    summary: "templates/summary.html"
```

## 🐳 Docker Configuration

### Dockerfile Configuration

```dockerfile
FROM python:3.11-slim

ENV PYNOMALY_CONFIG_PATH=/app/config/production.yaml
ENV PYNOMALY_LOG_LEVEL=INFO
ENV PYNOMALY_WORKERS=4

COPY config/production.yaml /app/config/
COPY . /app/

WORKDIR /app
RUN pip install -e .[web]

EXPOSE 8000
CMD ["pynomaly", "web", "start", "--host", "0.0.0.0"]
```

### Docker Compose Configuration

```yaml
version: '3.8'
services:
  pynomaly:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYNOMALY_DATABASE_URL=postgresql://postgres:password@db:5432/pynomaly
      - PYNOMALY_REDIS_URL=redis://redis:6379/0
      - PYNOMALY_SECRET_KEY=${SECRET_KEY}
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      
  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=pynomaly
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
      
volumes:
  postgres_data:
  redis_data:
```

## ⚙️ Configuration Validation

### Validation Commands

```bash
# Validate configuration
pynomaly config validate

# Show current configuration
pynomaly config show

# Test database connection
pynomaly config test-db

# Test cache connection
pynomaly config test-cache
```

### Configuration Schema

The configuration is validated against a JSON schema. Invalid configurations will show descriptive error messages.

## 🔄 Configuration Management

### Configuration Updates

```bash
# Reload configuration without restart
pynomaly config reload

# Apply new configuration
pynomaly config apply config/new-config.yaml

# Backup current configuration
pynomaly config backup
```

### Version Control

Store configuration files in version control, but exclude sensitive values:

```gitignore
# .gitignore
config/local.yaml
config/secrets.yaml
*.key
*.pem
```

Use separate files for secrets:

```yaml
# config/secrets.yaml (not in git)
secret_key: "actual-secret-key"
database_password: "actual-password"
```

## 📚 Configuration Examples

### Small Development Setup

```yaml
app:
  debug: true
server:
  workers: 1
database:
  url: "sqlite:///dev.db"
cache:
  type: "memory"
```

### Medium Production Setup

```yaml
server:
  workers: 4
  host: "0.0.0.0"
database:
  url: "postgresql://user:pass@localhost/pynomaly"
  pool_size: 10
cache:
  url: "redis://localhost:6379/0"
monitoring:
  enabled: true
```

### Large Enterprise Setup

```yaml
server:
  workers: 16
  host: "0.0.0.0"
database:
  url: "postgresql://user:pass@db-cluster/pynomaly"
  pool_size: 50
cache:
  url: "redis://redis-cluster:6379/0"
monitoring:
  enabled: true
  exporters:
    prometheus:
      enabled: true
auth:
  providers:
    ldap:
      enabled: true
    oauth:
      enabled: true
security:
  waf:
    enabled: true
  rate_limiting:
    enabled: true
```

---

**Next:** Learn about [Security & Authentication](./security.md) to secure your Pynomaly deployment.
