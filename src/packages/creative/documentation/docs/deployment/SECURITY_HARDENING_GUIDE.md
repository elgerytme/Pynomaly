# Pynomaly Security Hardening Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 🔒 Security Hardening

This comprehensive guide covers security hardening best practices for Pynomaly production deployments, including system-level security, application security, network security, and compliance requirements.

## 📋 Table of Contents

- [Security Architecture](#security-architecture)
- [System Hardening](#system-hardening)
- [Application Security](#application-security)
- [Network Security](#network-security)
- [Database Security](#database-security)
- [Container Security](#container-security)
- [Authentication & Authorization](#authentication--authorization)
- [Data Protection](#data-protection)
- [Monitoring & Incident Response](#monitoring--incident-response)
- [Compliance](#compliance)

## 🏗️ Security Architecture

### Defense in Depth Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Perimeter Security                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Network Security                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │            Host Security                    │   │   │
│  │  │  ┌─────────────────────────────────────┐   │   │   │
│  │  │  │        Application Security         │   │   │   │
│  │  │  │  ┌─────────────────────────────┐   │   │   │   │
│  │  │  │  │      Data Security          │   │   │   │   │
│  │  │  │  └─────────────────────────────┘   │   │   │   │
│  │  │  └─────────────────────────────────────┘   │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Security Principles

1. **Principle of Least Privilege**: Grant minimum necessary permissions
2. **Defense in Depth**: Multiple layers of security controls
3. **Fail Secure**: Default to secure state on failure
4. **Zero Trust**: Never trust, always verify
5. **Security by Design**: Build security into the architecture

## 🖥️ System Hardening

### Operating System Hardening

#### Ubuntu/Debian Hardening

```bash
#!/bin/bash
# system-hardening.sh

set -e

echo "🔒 Starting system hardening..."

# Update system packages
apt update && apt upgrade -y

# Install security tools
apt install -y \
    fail2ban \
    ufw \
    aide \
    rkhunter \
    chkrootkit \
    unattended-upgrades \
    auditd

# Configure automatic security updates
dpkg-reconfigure -plow unattended-upgrades

# Kernel hardening
cat >> /etc/sysctl.conf << 'EOF'
# Network security
net.ipv4.ip_forward = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.tcp_syncookies = 1

# IPv6 security
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Memory protection
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
kernel.yama.ptrace_scope = 1

# File system security
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
fs.suid_dumpable = 0
EOF

# Apply kernel parameters
sysctl -p

# Configure firewall
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Configure fail2ban
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

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10
EOF

# Restart fail2ban
systemctl restart fail2ban

# Configure SSH hardening
sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/X11Forwarding yes/X11Forwarding no/' /etc/ssh/sshd_config
echo "AllowUsers pynomaly" >> /etc/ssh/sshd_config
echo "Protocol 2" >> /etc/ssh/sshd_config
echo "ClientAliveInterval 300" >> /etc/ssh/sshd_config
echo "ClientAliveCountMax 2" >> /etc/ssh/sshd_config

# Restart SSH
systemctl restart sshd

# Configure audit logging
cat > /etc/audit/rules.d/pynomaly.rules << 'EOF'
# Monitor system calls
-a always,exit -F arch=b64 -S adjtimex -S settimeofday -k time-change
-a always,exit -F arch=b32 -S adjtimex -S settimeofday -S stime -k time-change
-a always,exit -F arch=b64 -S clock_settime -k time-change
-a always,exit -F arch=b32 -S clock_settime -k time-change

# Monitor user/group modifications
-w /etc/group -p wa -k identity
-w /etc/passwd -p wa -k identity
-w /etc/gshadow -p wa -k identity
-w /etc/shadow -p wa -k identity

# Monitor sudo usage
-w /var/log/sudo.log -p wa -k actions

# Monitor critical files
-w /etc/ssh/sshd_config -p wa -k ssh-config
-w /etc/nginx/nginx.conf -p wa -k nginx-config
-w /etc/crontab -p wa -k cron

# Monitor network configuration
-a always,exit -F arch=b64 -S sethostname -S setdomainname -k system-locale
-a always,exit -F arch=b32 -S sethostname -S setdomainname -k system-locale
-w /etc/issue -p wa -k system-locale
-w /etc/issue.net -p wa -k system-locale
-w /etc/hosts -p wa -k system-locale
-w /etc/network/ -p wa -k system-locale
EOF

# Restart auditd
systemctl restart auditd

echo "✅ System hardening completed!"
```

### File System Security

```bash
#!/bin/bash
# filesystem-hardening.sh

# Set secure permissions on critical files
chmod 600 /etc/shadow
chmod 600 /etc/gshadow
chmod 644 /etc/passwd
chmod 644 /etc/group
chmod 600 /boot/grub/grub.cfg

# Secure temporary directories
mount -o remount,nodev,nosuid,noexec /tmp
mount -o remount,nodev,nosuid,noexec /var/tmp
mount -o remount,nodev,nosuid /dev/shm

# Configure secure mount points in /etc/fstab
cat >> /etc/fstab << 'EOF'
tmpfs /tmp tmpfs defaults,nodev,nosuid,noexec 0 0
tmpfs /var/tmp tmpfs defaults,nodev,nosuid,noexec 0 0
tmpfs /dev/shm tmpfs defaults,nodev,nosuid,noexec 0 0
EOF

# Set up AIDE (file integrity monitoring)
aide --init
mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Schedule regular AIDE checks
cat > /etc/cron.daily/aide-check << 'EOF'
#!/bin/bash
/usr/bin/aide --check | mail -s "AIDE Report $(hostname)" admin@pynomaly.com
EOF
chmod +x /etc/cron.daily/aide-check
```

## 🛡️ Application Security

### Secure Configuration

```python
# config/security.py
import os
from cryptography.fernet import Fernet
from passlib.context import CryptContext

class SecurityConfig:
    """Security configuration for Pynomaly."""
    
    # Encryption settings
    SECRET_KEY = os.getenv("SECRET_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
    
    if not all([SECRET_KEY, JWT_SECRET_KEY, ENCRYPTION_KEY]):
        raise ValueError("Missing required security environment variables")
    
    # Password hashing
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # JWT settings
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_DELTA = 3600  # 1 hour
    JWT_REFRESH_DELTA = 604800   # 7 days
    
    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE = 100
    RATE_LIMIT_BURST = 20
    
    # Input validation
    MAX_REQUEST_SIZE_MB = 50
    MAX_FILE_SIZE_MB = 100
    ALLOWED_FILE_TYPES = {'.csv', '.json', '.parquet', '.pkl'}
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }
    
    # CORS settings
    CORS_ORIGINS = [
        "https://pynomaly.com",
        "https://app.pynomaly.com"
    ]
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE"]
    CORS_HEADERS = ["Authorization", "Content-Type"]
    
    # Session security
    SESSION_SECURE = True
    SESSION_HTTPONLY = True
    SESSION_SAMESITE = "strict"
    SESSION_TIMEOUT = 1800  # 30 minutes
    
    @classmethod
    def get_fernet_cipher(cls):
        """Get Fernet cipher for encryption/decryption."""
        return Fernet(cls.ENCRYPTION_KEY.encode())
    
    @classmethod
    def hash_password(cls, password: str) -> str:
        """Hash password using bcrypt."""
        return cls.pwd_context.hash(password)
    
    @classmethod
    def verify_password(cls, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return cls.pwd_context.verify(password, hashed)

# Input validation
from pydantic import BaseModel, validator
import re

class SecureInput(BaseModel):
    """Base class for secure input validation."""
    
    @validator('*', pre=True)
    def sanitize_input(cls, v):
        """Sanitize input to prevent injection attacks."""
        if isinstance(v, str):
            # Remove potentially dangerous characters
            v = re.sub(r'[<>"\';()&+]', '', v)
            # Limit string length
            if len(v) > 1000:
                raise ValueError("Input too long")
        return v

class SecureFileUpload(BaseModel):
    """Secure file upload validation."""
    
    filename: str
    content_type: str
    size: int
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename for security."""
        # Check for path traversal
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError("Invalid filename")
        
        # Check file extension
        ext = os.path.splitext(v)[1].lower()
        if ext not in SecurityConfig.ALLOWED_FILE_TYPES:
            raise ValueError(f"File type {ext} not allowed")
        
        return v
    
    @validator('size')
    def validate_size(cls, v):
        """Validate file size."""
        max_size = SecurityConfig.MAX_FILE_SIZE_MB * 1024 * 1024
        if v > max_size:
            raise ValueError(f"File too large (max {SecurityConfig.MAX_FILE_SIZE_MB}MB)")
        return v
```

### Security Middleware

```python
# middleware/security.py
from fastapi import Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import HTTPException
import time
import jwt
from collections import defaultdict
import asyncio

class SecurityMiddleware:
    """Security middleware for FastAPI."""
    
    def __init__(self, app):
        self.app = app
        self.rate_limiter = RateLimiter()
        self.security_headers = SecurityConfig.SECURITY_HEADERS
    
    async def __call__(self, scope, receive, send):
        """Process request through security middleware."""
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Rate limiting
            if not await self.rate_limiter.is_allowed(request.client.host):
                response = Response(
                    content="Rate limit exceeded",
                    status_code=429,
                    headers={"Retry-After": "60"}
                )
                await response(scope, receive, send)
                return
            
            # Input validation
            if not await self.validate_request(request):
                response = Response(
                    content="Invalid request",
                    status_code=400
                )
                await response(scope, receive, send)
                return
            
            # Process request
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Add security headers
                    headers = list(message.get("headers", []))
                    for name, value in self.security_headers.items():
                        headers.append([name.encode(), value.encode()])
                    message["headers"] = headers
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)
    
    async def validate_request(self, request: Request) -> bool:
        """Validate request for security."""
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length:
            max_size = SecurityConfig.MAX_REQUEST_SIZE_MB * 1024 * 1024
            if int(content_length) > max_size:
                return False
        
        # Check for SQL injection patterns
        query_string = str(request.url.query)
        sql_patterns = [
            "union", "select", "insert", "update", "delete",
            "drop", "create", "alter", "exec", "execute"
        ]
        for pattern in sql_patterns:
            if pattern in query_string.lower():
                return False
        
        return True

class RateLimiter:
    """Rate limiter implementation."""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.cleanup_task = asyncio.create_task(self.cleanup_old_requests())
    
    async def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed based on rate limit."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests for this IP
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= SecurityConfig.RATE_LIMIT_REQUESTS_PER_MINUTE:
            return False
        
        # Add current request
        self.requests[client_ip].append(now)
        return True
    
    async def cleanup_old_requests(self):
        """Cleanup old requests periodically."""
        while True:
            now = time.time()
            minute_ago = now - 60
            
            for ip in list(self.requests.keys()):
                self.requests[ip] = [
                    req_time for req_time in self.requests[ip]
                    if req_time > minute_ago
                ]
                if not self.requests[ip]:
                    del self.requests[ip]
            
            await asyncio.sleep(60)  # Cleanup every minute

class JWTBearer(HTTPBearer):
    """JWT Bearer token authentication."""
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=403,
                    detail="Invalid authentication scheme"
                )
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(
                    status_code=403,
                    detail="Invalid token or expired token"
                )
            return credentials.credentials
        else:
            raise HTTPException(
                status_code=403,
                detail="Invalid authorization code"
            )
    
    def verify_jwt(self, token: str) -> bool:
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                token,
                SecurityConfig.JWT_SECRET_KEY,
                algorithms=[SecurityConfig.JWT_ALGORITHM]
            )
            return payload is not None
        except jwt.PyJWTError:
            return False
```

## 🌐 Network Security

### Firewall Configuration

```bash
#!/bin/bash
# firewall-config.sh

# Advanced UFW configuration
ufw --force reset

# Default policies
ufw default deny incoming
ufw default allow outgoing

# SSH (limit connection rate)
ufw limit ssh

# HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Application ports (internal only)
ufw allow from 10.0.0.0/8 to any port 8000  # API
ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL
ufw allow from 10.0.0.0/8 to any port 6379  # Redis

# Monitoring ports (internal only)
ufw allow from 10.0.0.0/8 to any port 9090  # Prometheus
ufw allow from 10.0.0.0/8 to any port 3000  # Grafana
ufw allow from 10.0.0.0/8 to any port 9093  # AlertManager

# Enable firewall
ufw --force enable

# Configure iptables for additional security
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT

# Save iptables rules
iptables-save > /etc/iptables/rules.v4
```

### SSL/TLS Configuration

```nginx
# nginx-ssl.conf
server {
    listen 80;
    server_name api.pynomaly.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.pynomaly.com;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/api.pynomaly.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.pynomaly.com/privkey.pem;

    # SSL security settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/letsencrypt/live/api.pynomaly.com/chain.pem;
    resolver 8.8.8.8 8.8.4.4 valid=300s;
    resolver_timeout 5s;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none';" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Security headers for proxied requests
        proxy_hide_header X-Powered-By;
        proxy_hide_header Server;
    }

    location /auth/login {
        limit_req zone=login burst=5 nodelay;
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Block common attack patterns
    location ~* \.(php|jsp|asp|aspx|cgi)$ {
        return 444;
    }

    location ~* /(\.|wp-|admin|phpmyadmin) {
        return 444;
    }

    # Hide server information
    server_tokens off;
    more_clear_headers Server;
    more_clear_headers X-Powered-By;
}
```

## 🗄️ Database Security

### PostgreSQL Hardening

```sql
-- postgresql-security.sql

-- Enable SSL
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/etc/ssl/certs/postgresql.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/ssl/private/postgresql.key';

-- Configure authentication
ALTER SYSTEM SET password_encryption = 'scram-sha-256';
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_failed_login_attempts = on;

-- Limit connections
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET superuser_reserved_connections = 3;

-- Configure logging for security events
ALTER SYSTEM SET log_statement = 'ddl';
ALTER SYSTEM SET log_min_duration_statement = 1000;
ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';

-- Enable row level security
ALTER DATABASE pynomaly SET row_security = on;

-- Create secure database user
CREATE ROLE pynomaly_app LOGIN PASSWORD 'secure_random_password';
GRANT CONNECT ON DATABASE pynomaly TO pynomaly_app;
GRANT USAGE ON SCHEMA public TO pynomaly_app;

-- Grant specific permissions only
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE detectors TO pynomaly_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE datasets TO pynomaly_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE detection_results TO pynomaly_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO pynomaly_app;

-- Create read-only user for monitoring
CREATE ROLE pynomaly_monitor LOGIN PASSWORD 'monitor_password';
GRANT CONNECT ON DATABASE pynomaly TO pynomaly_monitor;
GRANT USAGE ON SCHEMA public TO pynomaly_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO pynomaly_monitor;

-- Row Level Security policies
CREATE POLICY user_data_policy ON detectors
  FOR ALL TO pynomaly_app
  USING (user_id = current_setting('app.current_user_id')::uuid);

CREATE POLICY user_results_policy ON detection_results
  FOR ALL TO pynomaly_app
  USING (user_id = current_setting('app.current_user_id')::uuid);

-- Enable RLS on tables
ALTER TABLE detectors ENABLE ROW LEVEL SECURITY;
ALTER TABLE datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE detection_results ENABLE ROW LEVEL SECURITY;

-- Configure pg_hba.conf for secure authentication
-- hostssl pynomaly pynomaly_app 10.0.0.0/8 scram-sha-256
-- hostssl pynomaly pynomaly_monitor 10.0.0.0/8 scram-sha-256
-- local all postgres peer
-- local all all md5

-- Reload configuration
SELECT pg_reload_conf();
```

### Database Connection Security

```python
# database/secure_connection.py
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import ssl

class SecureDatabaseManager:
    """Secure database connection manager."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = self._create_secure_engine()
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def _create_secure_engine(self):
        """Create secure database engine."""
        # SSL context for database connection
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # Configure connection pool
        engine = create_engine(
            self.database_url,
            pool_class=QueuePool,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "sslmode": "require",
                "sslcert": "/etc/ssl/certs/client.crt",
                "sslkey": "/etc/ssl/private/client.key",
                "sslrootcert": "/etc/ssl/certs/ca.crt",
                "application_name": "pynomaly_api",
                "connect_timeout": 10,
                "command_timeout": 30
            }
        )
        
        # Add connection event listeners
        @event.listens_for(engine, "connect")
        def set_security_settings(dbapi_connection, connection_record):
            """Set security settings on new connections."""
            with dbapi_connection.cursor() as cursor:
                # Set security-related session variables
                cursor.execute("SET session_timeout = '30min'")
                cursor.execute("SET statement_timeout = '10min'")
                cursor.execute("SET idle_in_transaction_session_timeout = '5min'")
                cursor.execute("SET log_statement = 'all'")
        
        return engine
    
    async def get_session(self):
        """Get database session with security context."""
        session = self.SessionLocal()
        try:
            # Set user context for RLS
            user_id = get_current_user_id()  # Get from JWT token
            await session.execute(
                f"SET app.current_user_id = '{user_id}'"
            )
            yield session
        finally:
            session.close()

# SQL injection prevention
from sqlalchemy import text
import bleach

class SecureQueryBuilder:
    """Secure query builder to prevent SQL injection."""
    
    @staticmethod
    def sanitize_input(value: str) -> str:
        """Sanitize input to prevent SQL injection."""
        # Remove dangerous characters
        sanitized = bleach.clean(value, tags=[], strip=True)
        
        # Additional SQL-specific sanitization
        dangerous_patterns = [
            '--', ';', '/*', '*/', 'xp_', 'sp_', 'exec', 'execute',
            'union', 'select', 'insert', 'update', 'delete', 'drop'
        ]
        
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, '')
        
        return sanitized
    
    @staticmethod
    def build_safe_query(template: str, **params):
        """Build safe parameterized query."""
        # Sanitize all parameters
        safe_params = {
            key: SecureQueryBuilder.sanitize_input(str(value))
            for key, value in params.items()
        }
        
        return text(template), safe_params
```

## 📦 Container Security

### Secure Dockerfile

```dockerfile
# Dockerfile.secure
FROM python:3.11-slim-bullseye

# Create non-root user
RUN groupadd -r pynomaly && useradd -r -g pynomaly pynomaly

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set secure working directory
WORKDIR /app
RUN chown pynomaly:pynomaly /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=pynomaly:pynomaly . .

# Remove sensitive files
RUN rm -rf .git .env* tests/ docs/ *.md

# Set secure permissions
RUN chmod -R 755 /app && \
    chmod -R 640 /app/config/ && \
    chmod 644 /app/requirements.txt

# Switch to non-root user
USER pynomaly

# Remove shell access
RUN rm /bin/sh && ln -s /bin/false /bin/sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Security labels
LABEL security.no-new-privileges=true
LABEL security.user=pynomaly

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "pynomaly.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Security Configuration

```yaml
# docker-compose.secure.yml
version: '3.8'

services:
  pynomaly-api:
    build:
      context: .
      dockerfile: Dockerfile.secure
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
      - /var/run:noexec,nosuid,size=50m
    volumes:
      - ./logs:/app/logs:rw,noexec,nosuid
      - ./data:/app/data:rw,noexec,nosuid
    environment:
      - PYTHONPATH=/app
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
    user: "1001:1001"  # pynomaly user
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M

  postgres:
    image: postgres:15-alpine
    security_opt:
      - no-new-privileges:true
    environment:
      - POSTGRES_DB=pynomaly
      - POSTGRES_USER=pynomaly
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data:rw,noexec,nosuid
      - ./postgresql.conf:/etc/postgresql/postgresql.conf:ro
      - ./pg_hba.conf:/etc/postgresql/pg_hba.conf:ro
    secrets:
      - db_password
    command: >
      postgres
      -c config_file=/etc/postgresql/postgresql.conf
      -c hba_file=/etc/postgresql/pg_hba.conf
    user: "999:999"  # postgres user
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    security_opt:
      - no-new-privileges:true
    command: >
      redis-server
      --requirepass "${REDIS_PASSWORD}"
      --appendonly yes
      --appendfsync everysec
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data:rw,noexec,nosuid
    user: "999:999"  # redis user
    restart: unless-stopped

secrets:
  db_password:
    file: ./secrets/db_password.txt

volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind,noexec,nosuid
      device: /opt/pynomaly/data/postgres
  
  redis_data:
    driver: local
    driver_opts:
      type: none
      o: bind,noexec,nosuid
      device: /opt/pynomaly/data/redis

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 🔐 Authentication & Authorization

### JWT Implementation

```python
# auth/jwt_manager.py
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from passlib.context import CryptContext

class JWTManager:
    """JWT token manager for authentication."""
    
    def __init__(self):
        self.secret_key = SecurityConfig.JWT_SECRET_KEY
        self.algorithm = SecurityConfig.JWT_ALGORITHM
        self.access_token_expire_minutes = SecurityConfig.JWT_EXPIRATION_DELTA // 60
        self.refresh_token_expire_days = SecurityConfig.JWT_REFRESH_DELTA // 86400
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token."""
        payload = self.verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Create new access token
        new_data = {
            "sub": payload.get("sub"),
            "user_id": payload.get("user_id"),
            "roles": payload.get("roles")
        }
        
        return self.create_access_token(new_data)

# Role-based access control
from enum import Enum
from functools import wraps

class UserRole(Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_USER = "api_user"

class Permission(Enum):
    """System permissions."""
    READ_DETECTORS = "read_detectors"
    WRITE_DETECTORS = "write_detectors"
    DELETE_DETECTORS = "delete_detectors"
    READ_DATASETS = "read_datasets"
    WRITE_DATASETS = "write_datasets"
    DELETE_DATASETS = "delete_datasets"
    ADMIN_USERS = "admin_users"
    VIEW_METRICS = "view_metrics"

# Role-permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [p for p in Permission],
    UserRole.USER: [
        Permission.READ_DETECTORS,
        Permission.WRITE_DETECTORS,
        Permission.READ_DATASETS,
        Permission.WRITE_DATASETS,
        Permission.VIEW_METRICS
    ],
    UserRole.VIEWER: [
        Permission.READ_DETECTORS,
        Permission.READ_DATASETS,
        Permission.VIEW_METRICS
    ],
    UserRole.API_USER: [
        Permission.READ_DETECTORS,
        Permission.WRITE_DETECTORS,
        Permission.READ_DATASETS
    ]
}

def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from JWT token
            current_user = get_current_user()
            user_roles = [UserRole(role) for role in current_user.get("roles", [])]
            
            # Check if user has required permission
            has_permission = any(
                permission in ROLE_PERMISSIONS.get(role, [])
                for role in user_roles
            )
            
            if not has_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(role: UserRole):
    """Decorator to require specific role."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = get_current_user()
            user_roles = [UserRole(r) for r in current_user.get("roles", [])]
            
            if role not in user_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role {role.value} required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

This security hardening guide provides comprehensive protection for Pynomaly production deployments across all layers of the technology stack.

---

## 🔗 Related Documentation

- **[Production Checklist](PRODUCTION_CHECKLIST.md)** - Deployment validation checklist
- **[Monitoring Setup](MONITORING_SETUP_GUIDE.md)** - Monitoring and alerting configuration
- **[Backup & Recovery](BACKUP_RECOVERY_GUIDE.md)** - Data protection procedures
- **[Troubleshooting](TROUBLESHOOTING_GUIDE.md)** - Security incident response

---

*Last Updated: 2024-01-15*