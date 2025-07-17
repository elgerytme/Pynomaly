# Production Deployment Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../index.md) > 🚀 Production Deployment

---

## 📋 Table of Contents

1. [Prerequisites and System Requirements](#prerequisites-and-system-requirements)
2. [Installation Methods](#installation-methods)
3. [Configuration Management](#configuration-management)
4. [Database Setup and Migrations](#database-setup-and-migrations)
5. [Security Considerations](#security-considerations)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Scaling Strategies](#scaling-strategies)
9. [Troubleshooting Common Deployment Issues](#troubleshooting-common-deployment-issues)
10. [Best Practices and Recommendations](#best-practices-and-recommendations)

---

## 🎯 Overview

This comprehensive guide covers deploying Pynomaly to production environments with enterprise-grade reliability, security, and performance. Pynomaly is built on **FastAPI** with **SQLAlchemy** for database operations, **Redis** for caching, and supports multiple deployment architectures.

### Architecture Stack

- **Backend Framework**: FastAPI with Uvicorn/Gunicorn
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Caching**: Redis with connection pooling
- **Authentication**: JWT with bcrypt password hashing
- **Message Queue**: Celery with Redis/RabbitMQ
- **Monitoring**: Prometheus, Grafana, Structlog
- **Security**: CORS, CSRF protection, rate limiting

---

## 🔧 Prerequisites and System Requirements

### Minimum System Requirements

#### Server Specifications

- **CPU**: 4 cores (Intel Xeon or AMD EPYC recommended)
- **Memory**: 16GB RAM minimum, 32GB+ recommended
- **Storage**: 100GB SSD minimum, 500GB+ recommended
- **Network**: 1Gbps connection with low latency

#### Operating System Support

- **Linux**: Ubuntu 20.04 LTS+, CentOS 8+, RHEL 8+, Debian 11+
- **Container**: Docker 20.10+, Kubernetes 1.20+
- **Cloud**: AWS, GCP, Azure, DigitalOcean

### Software Dependencies

#### Core Dependencies

```bash
# System packages
sudo apt update && sudo apt install -y \
    python3.11 python3.11-dev python3.11-venv \
    postgresql-14 postgresql-client-14 \
    redis-server \
    nginx \
    curl wget git \
    build-essential \
    libpq-dev libffi-dev libssl-dev
```

#### Python Version Requirements

- **Primary**: Python 3.11+ (recommended)
- **Supported**: Python 3.9, 3.10, 3.11, 3.12
- **Package Manager**: pip 23.0+ or poetry 1.4+

#### Database Requirements

- **PostgreSQL**: 13+ (14+ recommended for performance)
- **Redis**: 6.0+ (7.0+ recommended)
- **Connection Pooling**: asyncpg for async operations

### Network and Security Requirements

#### Firewall Configuration

```bash
# Essential ports
Port 22    - SSH (administrative access)
Port 80    - HTTP (redirect to HTTPS)
Port 443   - HTTPS (application access)
Port 8000  - Application server (internal)
Port 5432  - PostgreSQL (internal)
Port 6379  - Redis (internal)
Port 9090  - Prometheus (monitoring)
Port 3000  - Grafana (monitoring dashboard)
```

#### SSL/TLS Requirements

- SSL certificate (Let's Encrypt or commercial)
- TLS 1.2+ support
- Perfect Forward Secrecy (PFS)
- HSTS headers enabled

---

## 📦 Installation Methods

### Method 1: Docker Deployment (Recommended)

Docker provides the most reliable and consistent deployment experience across environments.

#### Quick Start with Docker Compose

```bash
# Clone repository
git clone https://github.com/your-org/pynomaly.git
cd pynomaly

# Copy production configuration
cp .env.production.example .env.production
# Edit configuration (see Configuration Management section)
nano .env.production

# Build and start services
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
curl -f http://localhost/health
```

#### Production Docker Compose Configuration

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  pynomaly-app:
    image: pynomaly:production
    build:
      context: .
      dockerfile: Dockerfile.production
      target: production
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    environment:
      - DATABASE_URL=postgresql+asyncpg://pynomaly:${DB_PASSWORD}@postgres:5432/pynomaly_prod
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - pynomaly-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: postgres:14-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: pynomaly_prod
      POSTGRES_USER: pynomaly
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - pynomaly-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pynomaly -d pynomaly_prod"]
      interval: 10s
      timeout: 5s
      retries: 5
    shm_size: 256mb
    command: [
      "postgres",
      "-c", "shared_buffers=256MB",
      "-c", "max_connections=200",
      "-c", "effective_cache_size=1GB",
      "-c", "work_mem=8MB",
      "-c", "maintenance_work_mem=128MB",
      "-c", "checkpoint_completion_target=0.9",
      "-c", "wal_buffers=16MB",
      "-c", "default_statistics_target=100"
    ]

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - pynomaly-network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./config/nginx/ssl:/etc/nginx/ssl:ro
      - static_files:/var/www/static:ro
    depends_on:
      - pynomaly-app
    networks:
      - pynomaly-network
    healthcheck:
      test: ["CMD", "nginx", "-t"]
      interval: 30s
      timeout: 10s
      retries: 3

  celery-worker:
    image: pynomaly:production
    restart: unless-stopped
    command: celery -A src.pynomaly.infrastructure.tasks.celery_app worker -l info --concurrency=4
    environment:
      - DATABASE_URL=postgresql+asyncpg://pynomaly:${DB_PASSWORD}@postgres:5432/pynomaly_prod
      - REDIS_URL=redis://redis:6379/1
      - SECRET_KEY=${SECRET_KEY}
      - ENVIRONMENT=production
    depends_on:
      - postgres
      - redis
    networks:
      - pynomaly-network
    deploy:
      replicas: 2

  celery-beat:
    image: pynomaly:production
    restart: unless-stopped
    command: celery -A src.pynomaly.infrastructure.tasks.celery_app beat -l info
    environment:
      - DATABASE_URL=postgresql+asyncpg://pynomaly:${DB_PASSWORD}@postgres:5432/pynomaly_prod
      - REDIS_URL=redis://redis:6379/1
      - SECRET_KEY=${SECRET_KEY}
      - ENVIRONMENT=production
    depends_on:
      - postgres
      - redis
    networks:
      - pynomaly-network

volumes:
  postgres_data:
  redis_data:
  static_files:

networks:
  pynomaly-network:
    driver: bridge
```

### Method 2: Direct Installation

For environments where Docker is not available or preferred.

#### System Setup

```bash
# Create application user
sudo useradd -m -s /bin/bash pynomaly
sudo usermod -aG sudo pynomaly

# Create application directory
sudo mkdir -p /opt/pynomaly
sudo chown pynomaly:pynomaly /opt/pynomaly

# Switch to application user
sudo su - pynomaly
cd /opt/pynomaly
```

#### Python Environment Setup

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip wheel setuptools
pip install -r requirements-prod.txt

# Install Pynomaly
pip install -e .
```

#### Application Setup

```bash
# Copy source code
git clone https://github.com/your-org/pynomaly.git .

# Copy configuration
cp config/production.yml.example config/production.yml
# Edit configuration as needed

# Create necessary directories
mkdir -p logs data/input data/output storage/models

# Set permissions
chmod 750 /opt/pynomaly
chmod 600 config/production.yml
```

### Method 3: Package Installation

Install Pynomaly as a system package.

#### Using pip

```bash
# Install from PyPI (when available)
pip install pynomaly[production]

# Or install from source
pip install git+https://github.com/your-org/pynomaly.git
```

#### Using Poetry

```bash
# Add to existing project
poetry add pynomaly

# Or create new project
poetry new my-pynomaly-deployment
cd my-pynomaly-deployment
poetry add pynomaly[production]
```

---

## ⚙️ Configuration Management

### Environment Variables

Create a `.env.production` file with the following configuration:

```bash
# Application Configuration
APP_NAME="Pynomaly Production"
APP_VERSION="1.0.0"
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security Configuration
SECRET_KEY="your-super-secret-key-change-this-in-production"
JWT_SECRET_KEY="your-jwt-secret-key-change-this-too"
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Encryption
ENCRYPTION_KEY="your-encryption-key-for-sensitive-data"
PASSWORD_HASH_ALGORITHM=bcrypt
PASSWORD_HASH_ROUNDS=12

# Database Configuration
DATABASE_URL="postgresql+asyncpg://pynomaly:your_db_password@localhost:5432/pynomaly_prod"
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Redis Configuration
REDIS_URL="redis://localhost:6379/0"
REDIS_PASSWORD="your_redis_password"
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_KEEPALIVE=true

# Cache Configuration
CACHE_ENABLED=true
CACHE_DEFAULT_TIMEOUT=300
CACHE_REDIS_DB=1

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST=10
RATE_LIMIT_STORAGE_URL="redis://localhost:6379/2"

# CORS Configuration
CORS_ENABLED=true
CORS_ORIGINS="https://your-domain.com,https://api.your-domain.com"
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS="GET,POST,PUT,DELETE,OPTIONS"
CORS_ALLOW_HEADERS="*"

# Monitoring and Observability
MONITORING_ENABLED=true
PROMETHEUS_ENABLED=true
METRICS_ENDPOINT="/metrics"
HEALTH_CHECK_ENDPOINT="/health"
TRACING_ENABLED=true
JAEGER_ENDPOINT="http://localhost:14268/api/traces"

# Logging Configuration
LOG_FORMAT=json
LOG_FILE="/var/log/pynomaly/app.log"
LOG_ROTATION=true
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=10

# Email Configuration
SMTP_HOST="smtp.gmail.com"
SMTP_PORT=587
SMTP_USERNAME="your-email@gmail.com"
SMTP_PASSWORD="your-app-password"
SMTP_USE_TLS=true
EMAIL_FROM="noreply@your-domain.com"

# File Storage
STORAGE_TYPE=local  # or s3, gcs, azure
STORAGE_PATH="/opt/pynomaly/storage"
MAX_UPLOAD_SIZE=100MB

# AWS S3 Configuration (if using S3)
AWS_ACCESS_KEY_ID="your-access-key"
AWS_SECRET_ACCESS_KEY="your-secret-key"
AWS_REGION="us-east-1"
AWS_S3_BUCKET="pynomaly-storage"

# Feature Flags
AUTONOMOUS_MODE_ENABLED=true
AUTOML_ENABLED=true
EXPLAINABILITY_ENABLED=true
STREAMING_ENABLED=true
ENSEMBLE_ENABLED=true

# Performance Configuration
WORKER_PROCESSES=4
WORKER_CONNECTIONS=1000
WORKER_TIMEOUT=30
WORKER_KEEPALIVE=2
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=50

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
BACKUP_STORAGE_PATH="/opt/backups"
```

### Configuration Validation

Pynomaly includes built-in configuration validation:

```python
# Validate configuration on startup
from pynomaly.infrastructure.config import validate_configuration

try:
    validate_configuration()
    print("✅ Configuration valid")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    exit(1)
```

### Dynamic Configuration Updates

For configuration changes that don't require restart:

```bash
# Reload configuration
curl -X POST http://localhost:8000/api/v1/admin/reload-config \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Update feature flags
curl -X PUT http://localhost:8000/api/v1/admin/feature-flags \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"autonomous_mode_enabled": false}'
```

---

## 🗄️ Database Setup and Migrations

### PostgreSQL Installation and Configuration

#### Installation

```bash
# Ubuntu/Debian
sudo apt install postgresql-14 postgresql-client-14 postgresql-contrib-14

# CentOS/RHEL
sudo dnf install postgresql14-server postgresql14-contrib

# Initialize database (CentOS/RHEL only)
sudo postgresql-14-setup initdb
sudo systemctl enable postgresql-14
sudo systemctl start postgresql-14
```

#### Database Creation

```bash
# Switch to postgres user
sudo -u postgres psql

-- Create database and user
CREATE DATABASE pynomaly_prod;
CREATE USER pynomaly WITH ENCRYPTED PASSWORD 'your_secure_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE pynomaly_prod TO pynomaly;
GRANT CONNECT ON DATABASE pynomaly_prod TO pynomaly;

-- Connect to database and grant schema permissions
\c pynomaly_prod;
GRANT ALL ON SCHEMA public TO pynomaly;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pynomaly;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO pynomaly;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO pynomaly;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO pynomaly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO pynomaly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO pynomaly;
```

#### Performance Optimization

```bash
# Edit postgresql.conf
sudo nano /etc/postgresql/14/main/postgresql.conf
```

```ini
# Memory Configuration
shared_buffers = 4GB                    # 25% of total RAM
effective_cache_size = 12GB             # 75% of total RAM
work_mem = 64MB                         # Per connection
maintenance_work_mem = 512MB            # For maintenance operations

# Connection Configuration
max_connections = 200                   # Adjust based on load
max_prepared_transactions = 200         # For distributed transactions

# Checkpoint Configuration
checkpoint_completion_target = 0.9      # Spread checkpoint I/O
wal_buffers = 64MB                      # WAL buffer size
checkpoint_segments = 64                # Number of WAL segments

# Query Planning
default_statistics_target = 100         # Statistics detail level
random_page_cost = 1.1                  # SSD optimization
effective_io_concurrency = 200          # SSD concurrent I/O

# Logging Configuration
log_min_duration_statement = 1000       # Log slow queries (1 second)
log_checkpoints = on                    # Log checkpoint information
log_connections = on                    # Log connections
log_disconnections = on                 # Log disconnections
log_lock_waits = on                     # Log lock waits
log_temp_files = 10MB                   # Log large temp files

# Replication (for high availability)
wal_level = replica                     # Enable replication
max_wal_senders = 3                     # Number of replication connections
wal_keep_size = 1GB                     # Keep WAL files for replication
hot_standby = on                        # Enable read-only queries on standby
```

### Database Migrations with Alembic

Pynomaly uses Alembic for database schema management.

#### Initialize Migrations

```bash
# Navigate to application directory
cd /opt/pynomaly

# Initialize Alembic (if not already done)
alembic init alembic

# Generate initial migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

#### Migration Commands

```bash
# Check current migration status
alembic current

# Show migration history
alembic history

# Create new migration
alembic revision --autogenerate -m "Add new feature"

# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade revision_id

# Downgrade (use with caution in production)
alembic downgrade -1
```

#### Production Migration Script

```bash
#!/bin/bash
# scripts/migrate_production.sh

set -e

echo "Starting database migration..."

# Backup database before migration
echo "Creating backup..."
pg_dump -h localhost -U pynomaly pynomaly_prod | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Run migrations
echo "Running migrations..."
alembic upgrade head

# Verify migration
echo "Verifying migration..."
python -c "
from pynomaly.infrastructure.database import get_database_engine
from sqlalchemy import text
engine = get_database_engine()
with engine.connect() as conn:
    result = conn.execute(text('SELECT version_num FROM alembic_version'))
    print(f'Current migration version: {result.scalar()}')
"

echo "Migration completed successfully!"
```

### Database Indexes for Performance

Create essential indexes for optimal performance:

```sql
-- User management indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_created_at ON users(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active ON users(id) WHERE is_active = true;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email_active ON users(email, is_active);

-- API key indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_active ON api_keys(id) WHERE is_active = true;

-- Audit log indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_user_timestamp ON audit_logs(user_id, timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);

-- Dataset indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_datasets_user_id ON datasets(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_datasets_created_at ON datasets(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_datasets_name ON datasets(name);

-- Analysis results indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_results_dataset_id ON analysis_results(dataset_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_results_user_id ON analysis_results(user_id);

-- Performance monitoring indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_endpoint ON performance_metrics(endpoint);
```

---

## 🔒 Security Considerations

### Authentication and Authorization

#### JWT Configuration

```python
# Secure JWT settings
JWT_SECRET_KEY = "your-very-long-and-random-secret-key"
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
JWT_ISSUER = "pynomaly.io"
JWT_AUDIENCE = "pynomaly-api"
```

#### Password Security

```python
# Bcrypt configuration for password hashing
from passlib.context import CryptContext

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,  # Adjust based on security requirements vs performance
)
```

#### Role-Based Access Control (RBAC)

```python
# User roles and permissions
ROLES = {
    "admin": ["read", "write", "delete", "admin"],
    "analyst": ["read", "write"],
    "viewer": ["read"],
    "api_user": ["read", "write_limited"]
}

# Endpoint permissions
ENDPOINT_PERMISSIONS = {
    "/api/v1/admin/*": ["admin"],
    "/api/v1/users/*": ["admin", "analyst"],
    "/api/v1/analysis/*": ["admin", "analyst", "api_user"],
    "/api/v1/datasets/*": ["admin", "analyst", "api_user"],
}
```

### Rate Limiting and DDoS Protection

#### FastAPI Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply rate limits to endpoints
@app.get("/api/v1/analysis/predict")
@limiter.limit("10/minute")
async def predict_anomalies(request: Request, ...):
    pass

@app.post("/api/v1/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, ...):
    pass
```

#### Nginx Rate Limiting

```nginx
# Add to nginx.conf
http {
    # Rate limiting zones
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=10r/m;
    
    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=conn:10m;
    
    server {
        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            limit_conn conn 10;
            proxy_pass http://pynomaly_backend;
        }
        
        # Login endpoint
        location /api/v1/auth/login {
            limit_req zone=login burst=3 nodelay;
            proxy_pass http://pynomaly_backend;
        }
        
        # Upload endpoints
        location /api/v1/datasets/upload {
            limit_req zone=upload burst=5 nodelay;
            client_max_body_size 100M;
            proxy_pass http://pynomaly_backend;
        }
    }
}
```

### Data Protection and Encryption

#### Encryption at Rest

```python
# Database encryption configuration
from cryptography.fernet import Fernet

class EncryptionService:
    def __init__(self, key: str):
        self.fernet = Fernet(key.encode())
    
    def encrypt(self, data: str) -> str:
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        return self.fernet.decrypt(encrypted_data.encode()).decode()

# Use for sensitive fields
class User(Base):
    email = Column(String, nullable=False)  # Not encrypted
    phone = Column(String)  # Encrypted in application layer
    
    def set_phone(self, phone: str):
        encryption_service = get_encryption_service()
        self.phone = encryption_service.encrypt(phone)
```

#### Encryption in Transit

```nginx
# SSL/TLS configuration
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # SSL protocols and ciphers
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305;
    ssl_prefer_server_ciphers off;
    
    # SSL optimizations
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'";
}
```

### Security Monitoring and Logging

#### Audit Logging

```python
from pynomaly.infrastructure.logging import get_audit_logger

audit_logger = get_audit_logger()

async def log_security_event(
    user_id: Optional[int],
    action: str,
    resource: str,
    ip_address: str,
    user_agent: str,
    success: bool,
    additional_data: dict = None
):
    audit_logger.info(
        "Security event",
        extra={
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "additional_data": additional_data or {}
        }
    )

# Usage in endpoints
@app.post("/api/v1/auth/login")
async def login(request: Request, credentials: LoginRequest):
    try:
        user = await authenticate_user(credentials.username, credentials.password)
        await log_security_event(
            user_id=user.id,
            action="login",
            resource="auth",
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent", ""),
            success=True
        )
        return create_access_token(user)
    except AuthenticationError:
        await log_security_event(
            user_id=None,
            action="login",
            resource="auth",
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent", ""),
            success=False,
            additional_data={"username": credentials.username}
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")
```

---

## ⚡ Performance Optimization

### Application Performance

#### FastAPI Optimization

```python
# Optimized FastAPI configuration
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI(
    title="Pynomaly API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    # Optimize OpenAPI generation
    generate_unique_id_function=lambda route: f"{route.tags[0]}-{route.name}",
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["your-domain.com", "*.your-domain.com"]
)

# Optimize response models
from pydantic import BaseModel, Field
from typing import List, Optional

class OptimizedResponse(BaseModel):
    """Optimized response with field validation."""
    
    class Config:
        # Optimize serialization
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
        # Use enum values
        use_enum_values = True
        # Validate assignment
        validate_assignment = True
```

#### Database Optimization

```python
# SQLAlchemy optimization
from sqlalchemy.orm import sessionmaker, selectinload, joinedload
from sqlalchemy.pool import QueuePool

# Optimized engine configuration
engine = create_async_engine(
    DATABASE_URL,
    # Connection pool settings
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    # Query optimization
    echo=False,  # Disable in production
    future=True,
)

# Optimized session configuration
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Efficient queries with eager loading
async def get_user_with_datasets(user_id: int):
    async with async_session() as session:
        result = await session.execute(
            select(User)
            .options(
                selectinload(User.datasets).selectinload(Dataset.results),
                joinedload(User.api_keys)
            )
            .where(User.id == user_id)
        )
        return result.scalar_one_or_none()

# Bulk operations
async def bulk_insert_analysis_results(results: List[AnalysisResult]):
    async with async_session() as session:
        session.add_all(results)
        await session.commit()
```

#### Caching Strategy

```python
from redis.asyncio import Redis
from functools import wraps
import pickle
import hashlib

# Redis cache implementation
class CacheService:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.default_ttl = 300  # 5 minutes
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            data = await self.redis.get(key)
            return pickle.loads(data) if data else None
        except Exception:
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        try:
            ttl = ttl or self.default_ttl
            serialized = pickle.dumps(value)
            return await self.redis.setex(key, ttl, serialized)
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            return await self.redis.delete(key) > 0
        except Exception:
            return False

# Caching decorator
def cache_result(ttl: int = 300, key_prefix: str = ""):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{key_prefix}:{func.__name__}:{args}:{kwargs}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            cache_service = get_cache_service()
            result = await cache_service.get(cache_key)
            
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_service.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator

# Usage
@cache_result(ttl=600, key_prefix="datasets")
async def get_user_datasets(user_id: int) -> List[Dataset]:
    async with async_session() as session:
        result = await session.execute(
            select(Dataset).where(Dataset.user_id == user_id)
        )
        return result.scalars().all()
```

### Infrastructure Performance

#### Gunicorn Configuration

```python
# gunicorn.conf.py
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Performance
preload_app = True
timeout = 30
keepalive = 2

# Logging
accesslog = "/var/log/pynomaly/access.log"
errorlog = "/var/log/pynomaly/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# SSL (if terminating SSL at application level)
# keyfile = "/path/to/private.key"
# certfile = "/path/to/certificate.crt"
```

#### Nginx Optimization

```nginx
# Optimized nginx.conf
user www-data;
worker_processes auto;
worker_rlimit_nofile 65535;
pid /run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
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
    
    # Buffer sizes
    client_body_buffer_size 128k;
    client_max_body_size 100m;
    client_header_buffer_size 3m;
    large_client_header_buffers 4 256k;
    
    # Timeouts
    client_body_timeout 12;
    client_header_timeout 12;
    send_timeout 10;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_min_length 1000;
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
    
    # Upstream configuration
    upstream pynomaly_backend {
        least_conn;
        keepalive 32;
        server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
        server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
        server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
    }
    
    # Caching
    proxy_cache_path /var/cache/nginx/pynomaly levels=1:2 keys_zone=pynomaly:10m max_size=100m inactive=60m use_temp_path=off;
    
    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        # SSL configuration...
        
        # Static file caching
        location /static/ {
            alias /var/www/pynomaly/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
            gzip_static on;
        }
        
        # API caching for GET requests
        location /api/v1/datasets {
            proxy_cache pynomaly;
            proxy_cache_valid 200 302 10m;
            proxy_cache_valid 404 1m;
            proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
            proxy_cache_lock on;
            
            proxy_pass http://pynomaly_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
            proxy_http_version 1.1;
        }
        
        # API endpoints (no caching for POST/PUT/DELETE)
        location /api/ {
            proxy_pass http://pynomaly_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
            proxy_http_version 1.1;
            
            # Proxy timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
    }
}
```

---

## 📊 Monitoring and Logging

### Application Monitoring

#### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Request
import time

# Metrics definitions
REQUEST_COUNT = Counter(
    'pynomaly_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'pynomaly_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'pynomaly_active_connections',
    'Number of active connections'
)

ANALYSIS_OPERATIONS = Counter(
    'pynomaly_analyses_total',
    'Total number of pattern analyses',
    ['algorithm', 'status']
)

DATABASE_CONNECTIONS = Gauge(
    'pynomaly_database_connections',
    'Number of database connections'
)

# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Increment active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        return response
    
    finally:
        # Decrement active connections
        ACTIVE_CONNECTIONS.dec()

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
```

#### Health Checks

```python
from fastapi import HTTPException
import asyncio
import asyncpg
import redis.asyncio as redis

@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
    # Database health check
    try:
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
        health_status["checks"]["database"] = {"status": "healthy"}
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Redis health check
    try:
        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()
        await redis_client.close()
        health_status["checks"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # File system health check
    try:
        import os
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(b"health check")
            tmp.flush()
        health_status["checks"]["filesystem"] = {"status": "healthy"}
    except Exception as e:
        health_status["checks"]["filesystem"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    # Check if application is ready to serve requests
    try:
        # Basic functionality test
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        raise HTTPException(status_code=503, detail={"status": "not ready"})

@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    # Check if application is alive
    return {"status": "alive"}
```

### Structured Logging

#### Logging Configuration

```python
import structlog
import logging
from datetime import datetime

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.processors.add_logger_name,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

# Get logger
logger = structlog.get_logger("pynomaly")

# Usage in application
async def detect_anomalies(dataset_id: int, algorithm: str):
    logger.info(
        "Starting pattern analysis",
        dataset_id=dataset_id,
        algorithm=algorithm,
        user_id=current_user.id
    )
    
    try:
        start_time = time.time()
        results = await run_analysis(dataset_id, algorithm)
        duration = time.time() - start_time
        
        logger.info(
            "Pattern analysis completed",
            dataset_id=dataset_id,
            algorithm=algorithm,
            user_id=current_user.id,
            duration=duration,
            anomalies_found=len(results),
            success=True
        )
        
        return results
        
    except Exception as e:
        logger.error(
            "Pattern analysis failed",
            dataset_id=dataset_id,
            algorithm=algorithm,
            user_id=current_user.id,
            error=str(e),
            error_type=type(e).__name__,
            success=False
        )
        raise
```

#### Log Aggregation with ELK Stack

##### Docker Compose for ELK

```yaml
# elk-stack.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - elk-network

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./config/logstash/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
      - /var/log/pynomaly:/var/log/pynomaly:ro
    ports:
      - "5044:5044"
    networks:
      - elk-network
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    networks:
      - elk-network
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:

networks:
  elk-network:
    driver: bridge
```

##### Logstash Configuration

```ruby
# config/logstash/logstash.conf
input {
  file {
    path => "/var/log/pynomaly/*.log"
    start_position => "beginning"
    codec => json
  }
}

filter {
  if [level] {
    mutate {
      uppercase => [ "level" ]
    }
  }
  
  date {
    match => [ "timestamp", "ISO8601" ]
  }
  
  if [user_id] {
    mutate {
      convert => { "user_id" => "integer" }
    }
  }
  
  if [duration] {
    mutate {
      convert => { "duration" => "float" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "pynomaly-%{+YYYY.MM.dd}"
  }
  
  stdout {
    codec => rubydebug
  }
}
```

### Grafana Dashboards

#### Pynomaly Overview Dashboard

```json
{
  "dashboard": {
    "title": "Pynomaly Production Overview",
    "tags": ["pynomaly", "production"],
    "panels": [
      {
        "title": "Request Rate (req/min)",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(pynomaly_requests_total[1m])) * 60",
            "legendFormat": "Requests/min"
          }
        ]
      },
      {
        "title": "Response Time Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(pynomaly_request_duration_seconds_bucket[5m])",
            "legendFormat": "{{le}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(pynomaly_requests_total{status_code=~\"5..\"}[5m])) / sum(rate(pynomaly_requests_total[5m])) * 100",
            "legendFormat": "Error Rate %"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pynomaly_database_connections",
            "legendFormat": "Active Connections"
          }
        ]
      },
      {
        "title": "Pattern Analyses by Algorithm",
        "type": "graph",
        "targets": [
          {
            "expr": "sum by (algorithm) (rate(pynomaly_analyses_total[5m]))",
            "legendFormat": "{{algorithm}}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes",
            "legendFormat": "Memory Usage"
          }
        ]
      }
    ]
  }
}
```

---

## 📈 Scaling Strategies

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
# Enhanced load balancer configuration
upstream pynomaly_backend {
    # Load balancing method
    least_conn;  # or ip_hash, hash, random
    
    # Backend servers
    server 10.0.1.10:8000 max_fails=3 fail_timeout=30s weight=1;
    server 10.0.1.11:8000 max_fails=3 fail_timeout=30s weight=1;
    server 10.0.1.12:8000 max_fails=3 fail_timeout=30s weight=1;
    server 10.0.1.13:8000 max_fails=3 fail_timeout=30s weight=1 backup;
    
    # Keep-alive connections
    keepalive 32;
    keepalive_requests 1000;
    keepalive_timeout 60s;
}

# Health check configuration
location /health {
    access_log off;
    proxy_pass http://pynomaly_backend;
    proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
    proxy_connect_timeout 5s;
    proxy_send_timeout 5s;
    proxy_read_timeout 5s;
}
```

#### Docker Swarm Scaling

```yaml
# docker-compose.swarm.yml
version: '3.8'

services:
  pynomaly-app:
    image: pynomaly:production
    deploy:
      replicas: 6
      update_config:
        parallelism: 2
        order: start-first
        failure_action: rollback
        delay: 30s
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      placement:
        constraints:
          - node.role == worker
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    networks:
      - pynomaly-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  pynomaly-network:
    driver: overlay
    attachable: true
```

#### Kubernetes Deployment

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-app
  labels:
    app: pynomaly
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  selector:
    matchLabels:
      app: pynomaly
  template:
    metadata:
      labels:
        app: pynomaly
    spec:
      containers:
      - name: pynomaly
        image: pynomaly:production
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: redis-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

---
apiVersion: v1
kind: Service
metadata:
  name: pynomaly-service
spec:
  selector:
    app: pynomaly
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-app
  minReplicas: 3
  maxReplicas: 20
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
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
```

### Database Scaling

#### PostgreSQL Master-Slave Replication

```bash
# Master server configuration
# postgresql.conf
wal_level = replica
max_wal_senders = 3
wal_keep_size = 1GB
hot_standby = on

# pg_hba.conf
host replication replicator 10.0.1.0/24 md5

# Create replication user
sudo -u postgres psql
CREATE USER replicator REPLICATION LOGIN ENCRYPTED PASSWORD 'replica_password';
```

#### Read Replica Configuration

```python
# Database connection routing
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

class DatabaseManager:
    def __init__(self):
        # Master (write) connection
        self.master_engine = create_async_engine(
            "postgresql+asyncpg://user:pass@master:5432/pynomaly_prod",
            pool_size=20,
            max_overflow=30
        )
        
        # Read replicas
        self.read_engines = [
            create_async_engine(
                "postgresql+asyncpg://user:pass@replica1:5432/pynomaly_prod",
                pool_size=10,
                max_overflow=20
            ),
            create_async_engine(
                "postgresql+asyncpg://user:pass@replica2:5432/pynomaly_prod",
                pool_size=10,
                max_overflow=20
            )
        ]
        self._read_index = 0
    
    def get_write_engine(self):
        return self.master_engine
    
    def get_read_engine(self):
        # Round-robin load balancing
        engine = self.read_engines[self._read_index]
        self._read_index = (self._read_index + 1) % len(self.read_engines)
        return engine

# Usage in services
async def get_user_data(user_id: int):
    # Use read replica for queries
    engine = db_manager.get_read_engine()
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

async def update_user_data(user_id: int, data: dict):
    # Use master for writes
    engine = db_manager.get_write_engine()
    async with AsyncSession(engine) as session:
        user = await session.get(User, user_id)
        for key, value in data.items():
            setattr(user, key, value)
        await session.commit()
```

### Caching Strategies

#### Multi-Level Caching

```python
from typing import Optional, Any
import asyncio

class MultiLevelCache:
    def __init__(self, redis_client, local_cache_size=1000):
        self.redis = redis_client
        self.local_cache = {}
        self.local_cache_size = local_cache_size
        self.access_times = {}
    
    async def get(self, key: str) -> Optional[Any]:
        # Level 1: Local cache (fastest)
        if key in self.local_cache:
            self.access_times[key] = time.time()
            return self.local_cache[key]
        
        # Level 2: Redis cache
        try:
            data = await self.redis.get(key)
            if data:
                value = pickle.loads(data)
                self._set_local(key, value)
                return value
        except Exception:
            pass
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        # Set in both caches
        self._set_local(key, value)
        
        try:
            serialized = pickle.dumps(value)
            await self.redis.setex(key, ttl, serialized)
        except Exception:
            pass
    
    def _set_local(self, key: str, value: Any):
        # Implement LRU eviction
        if len(self.local_cache) >= self.local_cache_size:
            # Remove least recently used item
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.local_cache[lru_key]
            del self.access_times[lru_key]
        
        self.local_cache[key] = value
        self.access_times[key] = time.time()

# Cache warming strategy
class CacheWarmer:
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
    
    async def warm_user_data(self, user_ids: List[int]):
        """Pre-load frequently accessed user data."""
        tasks = [self._warm_user(user_id) for user_id in user_ids]
        await asyncio.gather(*tasks)
    
    async def _warm_user(self, user_id: int):
        # Load user data
        user_data = await get_user_data(user_id)
        if user_data:
            await self.cache.set(f"user:{user_id}", user_data, ttl=3600)
        
        # Load user datasets
        datasets = await get_user_datasets(user_id)
        if datasets:
            await self.cache.set(f"datasets:{user_id}", datasets, ttl=1800)
```

---

## 🔧 Troubleshooting Common Deployment Issues

### Application Issues

#### High Memory Usage

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -20

# Check Python memory usage
python -m memory_profiler your_script.py

# Docker memory usage
docker stats pynomaly-app

# Solutions:
# 1. Increase server memory
# 2. Optimize database queries
# 3. Implement connection pooling
# 4. Add memory limits to containers
# 5. Enable garbage collection tuning
```

#### Database Connection Issues

```bash
# Check PostgreSQL connections
sudo -u postgres psql -c "
SELECT 
    count(*) as total_connections,
    count(*) FILTER (WHERE state = 'active') as active,
    count(*) FILTER (WHERE state = 'idle') as idle,
    count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
FROM pg_stat_activity 
WHERE datname = 'pynomaly_prod';
"

# Check connection limits
sudo -u postgres psql -c "SHOW max_connections;"

# Monitor slow queries
sudo -u postgres psql -d pynomaly_prod -c "
SELECT 
    query,
    mean_time,
    calls,
    total_time
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"

# Solutions:
# 1. Increase max_connections in postgresql.conf
# 2. Implement connection pooling (PgBouncer)
# 3. Optimize slow queries
# 4. Close unused connections
```

#### Redis Connection Issues

```bash
# Check Redis status
redis-cli ping

# Monitor Redis connections
redis-cli client list

# Check memory usage
redis-cli info memory

# Monitor commands
redis-cli monitor

# Solutions:
# 1. Increase maxclients in redis.conf
# 2. Implement connection pooling
# 3. Clear unused keys
# 4. Optimize data structures
```

### Performance Issues

#### Slow API Responses

```bash
# Analyze request patterns
tail -f /var/log/pynomaly/access.log | grep -E "HTTP/[0-9\.]+ 200" | awk '{print $10}' | sort -n

# Profile specific endpoints
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/api/v1/endpoint"

# Where curl-format.txt contains:
#      time_namelookup:  %{time_namelookup}s\n
#         time_connect:  %{time_connect}s\n
#      time_appconnect:  %{time_appconnect}s\n
#     time_pretransfer:  %{time_pretransfer}s\n
#        time_redirect:  %{time_redirect}s\n
#   time_starttransfer:  %{time_starttransfer}s\n
#                      ----------\n
#           time_total:  %{time_total}s\n

# Solutions:
# 1. Add database indexes
# 2. Implement caching
# 3. Optimize queries
# 4. Add CDN for static content
# 5. Enable compression
```

#### High CPU Usage

```bash
# Check CPU usage
htop
top -p $(pgrep -f pynomaly)

# Profile Python application
python -m cProfile -o profile.stats your_script.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Check for CPU-intensive operations
# 1. Machine learning model training
# 2. Large dataset processing
# 3. Unoptimized algorithms
# 4. Memory leaks causing garbage collection overhead

# Solutions:
# 1. Optimize algorithms
# 2. Use async/await for I/O operations
# 3. Implement job queues for heavy tasks
# 4. Scale horizontally
# 5. Use compiled extensions (Cython, NumPy)
```

### Deployment Issues

#### Container Startup Issues

```bash
# Check container logs
docker logs pynomaly-app --tail 100 -f

# Check resource constraints
docker stats pynomaly-app

# Check health status
docker inspect pynomaly-app | jq '.[0].State.Health'

# Common issues and solutions:
# 1. Insufficient memory/CPU - increase resource limits
# 2. Database not ready - add proper health checks and dependencies
# 3. Missing environment variables - check configuration
# 4. Port conflicts - ensure ports are available
```

#### SSL/TLS Issues

```bash
# Test SSL certificate
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# Check certificate expiry
openssl x509 -in /etc/letsencrypt/live/your-domain.com/cert.pem -text -noout | grep "Not After"

# Test SSL configuration
curl -I https://your-domain.com

# Common issues:
# 1. Expired certificates - renew with certbot
# 2. Wrong certificate chain - check intermediate certificates
# 3. Cipher suite issues - update SSL configuration
# 4. Mixed content - ensure all resources use HTTPS
```

#### Load Balancer Issues

```bash
# Check nginx status
sudo systemctl status nginx
sudo nginx -t

# Check upstream servers
curl -H "Host: your-domain.com" http://backend-server:8000/health

# Monitor nginx access logs
tail -f /var/log/nginx/access.log

# Common issues:
# 1. Backend servers down - check health endpoints
# 2. Configuration errors - validate nginx.conf
# 3. SSL termination issues - check certificate paths
# 4. Rate limiting too aggressive - adjust limits
```

### Monitoring and Alerting Issues

#### Missing Metrics

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check metrics endpoint
curl http://localhost:8000/metrics

# Verify Grafana data sources
curl -H "Authorization: Bearer $GRAFANA_TOKEN" http://localhost:3000/api/datasources

# Solutions:
# 1. Ensure metrics endpoint is accessible
# 2. Check Prometheus configuration
# 3. Verify network connectivity
# 4. Check authentication settings
```

#### Log Collection Issues

```bash
# Check log file permissions
ls -la /var/log/pynomaly/

# Test log shipping
tail -f /var/log/pynomaly/app.log | grep ERROR

# Check ELK stack status
curl http://localhost:9200/_cluster/health
curl http://localhost:5601/api/status

# Solutions:
# 1. Fix file permissions
# 2. Check log rotation settings
# 3. Verify ELK connectivity
# 4. Check disk space for logs
```

---

## 🏆 Best Practices and Recommendations

### Security Best Practices

#### Infrastructure Security

1. **Principle of Least Privilege**
   - Create dedicated service accounts with minimal permissions
   - Use role-based access control (RBAC)
   - Regularly audit and rotate credentials

2. **Network Security**
   - Use VPC/private networks for internal communication
   - Implement proper firewall rules
   - Enable network segmentation
   - Use VPN for administrative access

3. **Data Protection**
   - Encrypt data at rest and in transit
   - Implement proper backup encryption
   - Use secure key management
   - Regular security audits

#### Application Security

```python
# Security headers middleware
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

# Input validation
from pydantic import BaseModel, validator, Field

class DatasetUpload(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$')
    description: str = Field(None, max_length=1000)
    
    @validator('name')
    def validate_name(cls, v):
        if '..' in v or '/' in v:
            raise ValueError('Invalid characters in name')
        return v
```

### Performance Best Practices

#### Database Optimization

1. **Query Optimization**

   ```sql
   -- Use indexes effectively
   CREATE INDEX CONCURRENTLY idx_datasets_user_created 
   ON datasets(user_id, created_at);
   
   -- Use partial indexes
   CREATE INDEX CONCURRENTLY idx_users_active 
   ON users(id) WHERE is_active = true;
   
   -- Analyze query performance
   EXPLAIN (ANALYZE, BUFFERS) 
   SELECT * FROM datasets WHERE user_id = 123;
   ```

2. **Connection Management**

   ```python
   # Use connection pooling
   engine = create_async_engine(
       DATABASE_URL,
       pool_size=20,
       max_overflow=30,
       pool_timeout=30,
       pool_recycle=3600,
       pool_pre_ping=True
   )
   
   # Use async context managers
   async def get_user_data(user_id: int):
       async with async_session() as session:
           # Session automatically closed
           result = await session.execute(...)
           return result.scalar_one_or_none()
   ```

#### Application Performance

1. **Async Programming**

   ```python
   # Use async/await for I/O operations
   import asyncio
   
   async def process_multiple_datasets(dataset_ids: List[int]):
       tasks = [process_dataset(id) for id in dataset_ids]
       results = await asyncio.gather(*tasks)
       return results
   
   # Use async database operations
   async def bulk_insert(items: List[Model]):
       async with async_session() as session:
           session.add_all(items)
           await session.commit()
   ```

2. **Caching Strategy**

   ```python
   # Implement intelligent caching
   @cache_result(ttl=3600, key_prefix="expensive_computation")
   async def expensive_computation(params: dict):
       # Heavy computation here
       return result
   
   # Cache invalidation
   async def update_dataset(dataset_id: int, data: dict):
       # Update data
       await update_database(dataset_id, data)
       
       # Invalidate related caches
       cache_keys = [
           f"dataset:{dataset_id}",
           f"user_datasets:{data['user_id']}",
           f"recent_datasets"
       ]
       for key in cache_keys:
           await cache.delete(key)
   ```

### Operational Best Practices

#### Deployment Strategy

1. **Blue-Green Deployment**

   ```bash
   #!/bin/bash
   # Blue-green deployment script
   
   CURRENT_ENV=$(docker ps --format "table {{.Names}}" | grep pynomaly | head -1 | cut -d'-' -f2)
   NEW_ENV=$([[ "$CURRENT_ENV" == "blue" ]] && echo "green" || echo "blue")
   
   echo "Current environment: $CURRENT_ENV"
   echo "Deploying to: $NEW_ENV"
   
   # Deploy to new environment
   docker-compose -f docker-compose.$NEW_ENV.yml up -d
   
   # Health check
   sleep 30
   if curl -f http://localhost:8001/health; then
       # Switch load balancer
       nginx -s reload
       
       # Stop old environment
       docker-compose -f docker-compose.$CURRENT_ENV.yml down
       echo "Deployment successful"
   else
       # Rollback
       docker-compose -f docker-compose.$NEW_ENV.yml down
       echo "Deployment failed, rolled back"
       exit 1
   fi
   ```

2. **Database Migrations**

   ```python
   # Safe migration practices
   from alembic import command
   from alembic.config import Config
   
   def safe_migrate():
       # Create backup before migration
       backup_database()
       
       try:
           # Run migration
           alembic_cfg = Config("alembic.ini")
           command.upgrade(alembic_cfg, "head")
           
           # Verify migration
           verify_migration()
           
       except Exception as e:
           # Rollback on failure
           rollback_migration()
           raise e
   ```

#### Monitoring and Alerting

1. **Comprehensive Monitoring**

   ```yaml
   # Prometheus alert rules
   groups:
   - name: pynomaly-alerts
     rules:
     - alert: HighErrorRate
       expr: rate(pynomaly_requests_total{status_code=~"5.."}[5m]) > 0.1
       for: 5m
       labels:
         severity: critical
       annotations:
         summary: "High error rate detected"
         description: "Error rate is {{ $value }} requests/second"
     
     - alert: HighResponseTime
       expr: histogram_quantile(0.95, rate(pynomaly_request_duration_seconds_bucket[5m])) > 2
       for: 5m
       labels:
         severity: warning
       annotations:
         summary: "High response time detected"
   ```

2. **Log Management**

   ```python
   # Structured logging for operations
   import structlog
   
   logger = structlog.get_logger("pynomaly.operations")
   
   async def deploy_model(model_id: str):
       logger.info(
           "Model deployment started",
           model_id=model_id,
           user_id=current_user.id,
           deployment_id=deployment_id
       )
       
       try:
           result = await deploy_model_to_production(model_id)
           logger.info(
               "Model deployment completed",
               model_id=model_id,
               deployment_id=deployment_id,
               success=True,
               deployment_time=result.deployment_time
           )
       except Exception as e:
           logger.error(
               "Model deployment failed",
               model_id=model_id,
               deployment_id=deployment_id,
               error=str(e),
               success=False
           )
           raise
   ```

### Maintenance Best Practices

#### Regular Maintenance Tasks

```bash
#!/bin/bash
# Daily maintenance script

# Check system health
check_disk_space() {
    USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ $USAGE -gt 80 ]; then
        echo "WARNING: Disk usage is ${USAGE}%"
        # Clean up logs
        find /var/log -name "*.log.*" -mtime +7 -delete
    fi
}

# Database maintenance
maintain_database() {
    # Update statistics
    psql -d pynomaly_prod -c "ANALYZE;"
    
    # Check for long-running queries
    psql -d pynomaly_prod -c "
    SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
    FROM pg_stat_activity 
    WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
    "
}

# Application health check
check_application_health() {
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "ERROR: Application health check failed"
        # Restart application
        docker-compose restart pynomaly-app
    fi
}

# Run maintenance tasks
check_disk_space
maintain_database
check_application_health
```

#### Backup Strategy

```bash
#!/bin/bash
# Comprehensive backup script

BACKUP_DIR="/opt/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Database backup
backup_database() {
    pg_dump -h localhost -U pynomaly pynomaly_prod | gzip > \
        "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"
    
    # Verify backup
    if [ $? -eq 0 ]; then
        echo "Database backup completed: db_backup_$TIMESTAMP.sql.gz"
    else
        echo "ERROR: Database backup failed"
        exit 1
    fi
}

# Application backup
backup_application() {
    tar -czf "$BACKUP_DIR/app_backup_$TIMESTAMP.tar.gz" \
        /opt/pynomaly \
        --exclude=/opt/pynomaly/logs \
        --exclude=/opt/pynomaly/data/temp
}

# Upload to cloud storage
upload_to_cloud() {
    # AWS S3
    aws s3 cp "$BACKUP_DIR/" s3://pynomaly-backups/ --recursive
    
    # Clean old local backups (keep 7 days)
    find "$BACKUP_DIR" -name "*backup*" -mtime +7 -delete
}

# Execute backup
backup_database
backup_application
upload_to_cloud
```

---

## 🎯 Conclusion

This production deployment guide provides a comprehensive framework for deploying Pynomaly in enterprise environments. Key takeaways:

### Critical Success Factors

1. **Security First**: Implement proper authentication, encryption, and security monitoring
2. **Performance Optimization**: Use caching, database optimization, and horizontal scaling
3. **Monitoring and Observability**: Comprehensive metrics, logging, and alerting
4. **Operational Excellence**: Automated deployments, backups, and maintenance

### Deployment Checklist

- [ ] System requirements met
- [ ] Database properly configured and tuned
- [ ] Security measures implemented
- [ ] Monitoring and logging configured
- [ ] Backup and disaster recovery tested
- [ ] Performance optimization applied
- [ ] CI/CD pipeline established
- [ ] Documentation updated

### Next Steps

1. **Start Small**: Begin with a single-server deployment
2. **Monitor Closely**: Establish baseline metrics and alerting
3. **Scale Gradually**: Add resources and complexity as needed
4. **Maintain Regularly**: Keep system updated and optimized

For additional support, refer to:

- [Troubleshooting Guide](../troubleshooting/troubleshooting-guide.md)
- [Performance Optimization](../advanced-features/performance-tuning.md)
- [Security Best Practices](../../security/security-best-practices.md)
- [Developer Guides](../../developer-guides/)

---

**Last Updated**: 2025-01-10  
**Next Review**: 2025-02-10  
**Version**: 1.0.0
