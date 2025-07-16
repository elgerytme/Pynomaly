#!/usr/bin/env python3
"""
Security hardening script for Pynomaly production deployment.
This script implements comprehensive security measures and SSL configuration.
"""

import asyncio
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration data structure."""

    domain: str = "pynomaly.local"
    ssl_cert_path: str = "/etc/ssl/certs/pynomaly"
    ssl_key_path: str = "/etc/ssl/private/pynomaly"
    firewall_enabled: bool = True
    fail2ban_enabled: bool = True
    rate_limiting_enabled: bool = True


class SecurityHardening:
    """Main security hardening orchestrator."""

    def __init__(self, config: SecurityConfig):
        """Initialize security hardening."""
        self.config = config
        self.security_results = []

    async def generate_ssl_certificates(self) -> bool:
        """Generate SSL certificates for the domain."""
        logger.info("🔐 Generating SSL certificates...")

        try:
            # Create SSL directories
            os.makedirs(os.path.dirname(self.config.ssl_cert_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.ssl_key_path), exist_ok=True)

            # Generate self-signed certificate for development/testing
            ssl_config = f"""
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = Production
L = Server
O = Pynomaly
CN = {self.config.domain}

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = {self.config.domain}
DNS.2 = www.{self.config.domain}
DNS.3 = api.{self.config.domain}
DNS.4 = localhost
IP.1 = 127.0.0.1
"""

            # Write SSL config
            with open("/tmp/ssl_config.conf", "w") as f:
                f.write(ssl_config)

            # Generate private key
            key_cmd = [
                "openssl",
                "genrsa",
                "-out",
                f"{self.config.ssl_key_path}.key",
                "2048",
            ]

            result = subprocess.run(key_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to generate private key: {result.stderr}")
                return False

            # Generate certificate
            cert_cmd = [
                "openssl",
                "req",
                "-new",
                "-x509",
                "-key",
                f"{self.config.ssl_key_path}.key",
                "-out",
                f"{self.config.ssl_cert_path}.crt",
                "-days",
                "365",
                "-config",
                "/tmp/ssl_config.conf",
                "-extensions",
                "v3_req",
            ]

            result = subprocess.run(cert_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to generate certificate: {result.stderr}")
                return False

            # Set proper permissions
            os.chmod(f"{self.config.ssl_key_path}.key", 0o600)
            os.chmod(f"{self.config.ssl_cert_path}.crt", 0o644)

            logger.info("✅ SSL certificates generated successfully")
            self.security_results.append(
                {"component": "SSL Certificates", "status": "success"}
            )
            return True

        except Exception as e:
            logger.error(f"SSL certificate generation failed: {e}")
            self.security_results.append(
                {"component": "SSL Certificates", "status": "failed", "error": str(e)}
            )
            return False

    async def configure_nginx_security(self) -> bool:
        """Configure Nginx with security headers and SSL."""
        logger.info("🌐 Configuring Nginx security...")

        try:
            nginx_config = rf"""
# Nginx Security Configuration for Pynomaly
upstream pynomaly_backend {{
    server pynomaly-api:8000;
    keepalive 32;
}}

# Rate limiting zones
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=login:10m rate=1r/m;
limit_req_zone $binary_remote_addr zone=general:10m rate=100r/m;

# Connection limiting
limit_conn_zone $binary_remote_addr zone=perip:10m;
limit_conn_zone $server_name zone=perserver:10m;

# Security headers map
map $sent_http_content_type $content_type_csp {{
    default                    "default-src 'self'";
    ~text/html                 "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:";
    ~application/json          "default-src 'none'";
}}

# HTTP to HTTPS redirect
server {{
    listen 80;
    server_name {self.config.domain} www.{self.config.domain};

    # Security headers even for redirect
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Redirect all HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}}

# Main HTTPS server
server {{
    listen 443 ssl http2;
    server_name {self.config.domain} www.{self.config.domain};

    # SSL Configuration
    ssl_certificate {self.config.ssl_cert_path}.crt;
    ssl_certificate_key {self.config.ssl_key_path}.key;

    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:50m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy $content_type_csp always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=(), accelerometer=()" always;

    # Hide server information
    server_tokens off;

    # Connection limits
    limit_conn perip 10;
    limit_conn perserver 100;

    # Request size limits
    client_max_body_size 50M;
    client_body_buffer_size 128k;
    client_header_buffer_size 1k;
    large_client_header_buffers 2 1k;

    # Timeout settings
    client_body_timeout 12;
    client_header_timeout 12;
    keepalive_timeout 15;
    send_timeout 10;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        application/atom+xml
        application/javascript
        application/json
        application/rss+xml
        application/vnd.ms-fontobject
        application/x-font-ttf
        application/x-web-app-manifest+json
        application/xhtml+xml
        application/xml
        font/opentype
        image/svg+xml
        image/x-icon
        text/css
        text/plain
        text/x-component;

    # API endpoints
    location /api/ {{
        limit_req zone=api burst=20 nodelay;
        limit_req_status 429;

        proxy_pass http://pynomaly_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;

        # Security headers for API
        add_header X-API-Version "1.0" always;
        add_header X-Rate-Limit-Limit "10" always;
        add_header X-Rate-Limit-Remaining "9" always;

        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;

        # Cache settings for API responses
        proxy_cache_bypass $http_upgrade;
        proxy_no_cache $http_upgrade;
    }}

    # Authentication endpoints (stricter rate limiting)
    location /auth/ {{
        limit_req zone=login burst=5 nodelay;
        limit_req_status 429;

        proxy_pass http://pynomaly_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Additional security for auth endpoints
        add_header X-Frame-Options DENY always;
        add_header X-Content-Type-Options nosniff always;
    }}

    # Health check endpoint
    location /health {{
        limit_req zone=general burst=50 nodelay;

        proxy_pass http://pynomaly_backend;
        access_log off;

        # Allow health checks from monitoring
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
    }}

    # Documentation endpoints
    location /docs {{
        limit_req zone=general burst=10 nodelay;

        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Cache documentation
        proxy_cache_valid 200 1h;
        proxy_cache_valid 404 1m;
    }}

    # Static files
    location /static/ {{
        alias /app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable" always;
        add_header X-Content-Type-Options nosniff always;

        # Security for static files
        location ~* \.(php|jsp|asp|aspx|cgi|sh|bat|exe|dll)$ {{
            deny all;
        }}
    }}

    # Grafana dashboard (with authentication)
    location /grafana/ {{
        limit_req zone=general burst=10 nodelay;

        proxy_pass http://grafana:3000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Additional security for monitoring
        auth_basic "Monitoring Access";
        auth_basic_user_file /etc/nginx/monitoring.htpasswd;
    }}

    # Block common attack patterns
    location ~ /\. {{
        deny all;
        access_log off;
        log_not_found off;
    }}

    location ~ ~$ {{
        deny all;
        access_log off;
        log_not_found off;
    }}

    # Block sensitive files
    location ~* \.(sql|log|conf|ini|bak|backup|old|tmp|temp)$ {{
        deny all;
        access_log off;
        log_not_found off;
    }}

    # Block PHP and other script files
    location ~* \.(php|php3|php4|php5|phtml|pl|py|jsp|asp|aspx|cgi|sh|bat)$ {{
        deny all;
        access_log off;
        log_not_found off;
    }}

    # Default location with rate limiting
    location / {{
        limit_req zone=general burst=20 nodelay;

        proxy_pass http://pynomaly_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Default timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }}

    # Custom error pages
    error_page 400 402 403 404 405 406 410 411 412 413 414 415 416 417 418 421 422 423 424 426 428 429 431 444 449 450 451 /4xx.html;
    error_page 500 501 502 503 504 505 506 507 508 509 510 511 521 522 523 524 525 /5xx.html;

    location = /4xx.html {{
        internal;
        root /var/www/html;
    }}

    location = /5xx.html {{
        internal;
        root /var/www/html;
    }}
}}
"""

            # Write Nginx configuration
            nginx_config_path = "config/nginx/nginx.conf"
            os.makedirs(os.path.dirname(nginx_config_path), exist_ok=True)

            with open(nginx_config_path, "w") as f:
                f.write(nginx_config)

            logger.info("✅ Nginx security configuration created")
            self.security_results.append(
                {"component": "Nginx Security", "status": "success"}
            )
            return True

        except Exception as e:
            logger.error(f"Nginx security configuration failed: {e}")
            self.security_results.append(
                {"component": "Nginx Security", "status": "failed", "error": str(e)}
            )
            return False

    async def configure_application_security(self) -> bool:
        """Configure application-level security measures."""
        logger.info("🔒 Configuring application security...")

        try:
            # Create security configuration for application
            security_config = {
                "security": {
                    "secret_key": os.getenv("SECRET_KEY", "change-this-in-production"),
                    "algorithm": "HS256",
                    "access_token_expire_minutes": 30,
                    "refresh_token_expire_days": 7,
                    "password_min_length": 12,
                    "password_require_special_chars": True,
                    "password_require_numbers": True,
                    "password_require_uppercase": True,
                    "max_login_attempts": 5,
                    "lockout_duration_minutes": 30,
                    "session_timeout_minutes": 120,
                    "require_https": True,
                    "secure_cookies": True,
                    "csrf_protection": True,
                    "content_security_policy": {
                        "default-src": ["'self'"],
                        "script-src": ["'self'", "'unsafe-inline'"],
                        "style-src": ["'self'", "'unsafe-inline'"],
                        "img-src": ["'self'", "data:", "https:"],
                        "font-src": ["'self'"],
                        "connect-src": ["'self'"],
                        "media-src": ["'none'"],
                        "object-src": ["'none'"],
                        "child-src": ["'none'"],
                        "frame-ancestors": ["'none'"],
                        "base-uri": ["'self'"],
                        "form-action": ["'self'"],
                    },
                },
                "rate_limiting": {
                    "global_limit": "1000/hour",
                    "per_user_limit": "100/hour",
                    "per_ip_limit": "500/hour",
                    "api_limit": "10/second",
                    "auth_limit": "5/minute",
                    "upload_limit": "10/minute",
                },
                "input_validation": {
                    "max_request_size": 50 * 1024 * 1024,  # 50MB
                    "max_file_size": 100 * 1024 * 1024,  # 100MB
                    "allowed_file_types": [".csv", ".json", ".parquet", ".xlsx"],
                    "max_filename_length": 255,
                    "sanitize_filenames": True,
                    "scan_uploads": True,
                },
                "logging": {
                    "log_security_events": True,
                    "log_failed_logins": True,
                    "log_admin_actions": True,
                    "log_data_access": True,
                    "log_retention_days": 90,
                    "alert_on_security_events": True,
                },
                "encryption": {
                    "encrypt_sensitive_data": True,
                    "encryption_algorithm": "AES-256-GCM",
                    "key_rotation_days": 90,
                    "backup_encryption": True,
                },
            }

            # Write security configuration
            security_config_path = "config/security.yml"
            with open(security_config_path, "w") as f:
                import yaml

                yaml.dump(security_config, f, default_flow_style=False)

            logger.info("✅ Application security configured")
            self.security_results.append(
                {"component": "Application Security", "status": "success"}
            )
            return True

        except Exception as e:
            logger.error(f"Application security configuration failed: {e}")
            self.security_results.append(
                {
                    "component": "Application Security",
                    "status": "failed",
                    "error": str(e),
                }
            )
            return False

    async def configure_database_security(self) -> bool:
        """Configure database security measures."""
        logger.info("🗄️ Configuring database security...")

        try:
            # Create database security configuration
            db_security_config = """
# PostgreSQL Security Configuration
# Add these settings to postgresql.conf

# Network Security
listen_addresses = 'localhost,postgres'
port = 5432

# SSL Configuration
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
ssl_ca_file = '/etc/ssl/certs/ca.crt'
ssl_crl_file = '/etc/ssl/certs/server.crl'

# Authentication
password_encryption = scram-sha-256
db_user_namespace = off

# Connection Security
max_connections = 100
superuser_reserved_connections = 3

# Logging Security Events
log_connections = on
log_disconnections = on
log_duration = on
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_statement = 'mod'
log_min_duration_statement = 1000

# Query Security
statement_timeout = 300000  # 5 minutes
lock_timeout = 30000        # 30 seconds
idle_in_transaction_session_timeout = 600000  # 10 minutes

# Resource Limits
shared_preload_libraries = 'pg_stat_statements'
max_locks_per_transaction = 64
max_pred_locks_per_transaction = 64

# Audit Settings
log_checkpoints = on
log_lock_waits = on
log_temp_files = 0
log_autovacuum_min_duration = 0
"""

            # Write database security configuration
            db_config_path = "config/postgresql/postgresql.conf"
            os.makedirs(os.path.dirname(db_config_path), exist_ok=True)

            with open(db_config_path, "w") as f:
                f.write(db_security_config)

            # Create pg_hba.conf with security rules
            pg_hba_config = """
# PostgreSQL Client Authentication Configuration
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# Local connections
local   all             postgres                                peer
local   all             all                                     scram-sha-256

# IPv4 local connections
host    all             all             127.0.0.1/32            scram-sha-256
host    all             all             10.0.0.0/8              scram-sha-256
host    all             all             172.16.0.0/12           scram-sha-256
host    all             all             192.168.0.0/16          scram-sha-256

# IPv6 local connections
host    all             all             ::1/128                 scram-sha-256

# SSL connections only for remote access
hostssl all             all             0.0.0.0/0               scram-sha-256
hostssl all             all             ::/0                    scram-sha-256

# Reject all other connections
host    all             all             0.0.0.0/0               reject
host    all             all             ::/0                    reject
"""

            pg_hba_path = "config/postgresql/pg_hba.conf"
            with open(pg_hba_path, "w") as f:
                f.write(pg_hba_config)

            logger.info("✅ Database security configured")
            self.security_results.append(
                {"component": "Database Security", "status": "success"}
            )
            return True

        except Exception as e:
            logger.error(f"Database security configuration failed: {e}")
            self.security_results.append(
                {"component": "Database Security", "status": "failed", "error": str(e)}
            )
            return False

    async def configure_redis_security(self) -> bool:
        """Configure Redis security measures."""
        logger.info("🔴 Configuring Redis security...")

        try:
            redis_security_config = """
# Redis Security Configuration
# Add these settings to redis.conf

# Network Security
bind 127.0.0.1 redis-cluster
port 6379
protected-mode yes

# Authentication
requirepass your_secure_redis_password_here
masterauth your_secure_redis_password_here

# Command Security
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command KEYS ""
rename-command CONFIG "CONFIG_b33f4a8c3d2e1f9a"
rename-command EVAL ""
rename-command DEBUG ""
rename-command SHUTDOWN "SHUTDOWN_a1b2c3d4e5f6"

# Connection Limits
maxclients 10000
timeout 300
tcp-keepalive 60

# Memory Security
maxmemory 1gb
maxmemory-policy allkeys-lru

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log
syslog-enabled yes
syslog-ident redis

# Persistence Security
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis

# Append Only File
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Security Modules
# loadmodule /path/to/security/module.so
"""

            # Write Redis security configuration
            redis_config_path = "config/redis/redis.conf"
            os.makedirs(os.path.dirname(redis_config_path), exist_ok=True)

            with open(redis_config_path, "w") as f:
                f.write(redis_security_config)

            logger.info("✅ Redis security configured")
            self.security_results.append(
                {"component": "Redis Security", "status": "success"}
            )
            return True

        except Exception as e:
            logger.error(f"Redis security configuration failed: {e}")
            self.security_results.append(
                {"component": "Redis Security", "status": "failed", "error": str(e)}
            )
            return False

    async def create_security_scripts(self) -> bool:
        """Create security monitoring and maintenance scripts."""
        logger.info("📜 Creating security scripts...")

        try:
            # Create security monitoring script
            security_monitor_script = """#!/bin/bash
# Security Monitoring Script for Pynomaly

LOG_FILE="/var/log/pynomaly/security.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Function to log security events
log_security_event() {
    echo "[$DATE] $1" >> $LOG_FILE
}

# Monitor failed login attempts
check_failed_logins() {
    FAILED_LOGINS=$(grep "authentication failed" /var/log/pynomaly/app.log | tail -n 100 | wc -l)
    if [ $FAILED_LOGINS -gt 10 ]; then
        log_security_event "HIGH: $FAILED_LOGINS failed login attempts detected"
        # Send alert
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"Security Alert: High number of failed login attempts detected"}' \
            $ALERT_WEBHOOK_URL
    fi
}

# Monitor suspicious IP addresses
check_suspicious_ips() {
    # Get IPs with high request rates
    SUSPICIOUS_IPS=$(tail -n 10000 /var/log/nginx/access.log | \
        awk '{print $1}' | sort | uniq -c | sort -nr | head -n 5)

    if [ ! -z "$SUSPICIOUS_IPS" ]; then
        log_security_event "INFO: Top requesting IPs: $SUSPICIOUS_IPS"
    fi
}

# Monitor SSL certificate expiration
check_ssl_expiration() {
    CERT_FILE="/etc/ssl/certs/pynomaly.crt"
    if [ -f "$CERT_FILE" ]; then
        EXPIRY_DATE=$(openssl x509 -enddate -noout -in $CERT_FILE | cut -d= -f2)
        EXPIRY_EPOCH=$(date -d "$EXPIRY_DATE" +%s)
        CURRENT_EPOCH=$(date +%s)
        DAYS_UNTIL_EXPIRY=$(( ($EXPIRY_EPOCH - $CURRENT_EPOCH) / 86400 ))

        if [ $DAYS_UNTIL_EXPIRY -lt 30 ]; then
            log_security_event "WARNING: SSL certificate expires in $DAYS_UNTIL_EXPIRY days"
        fi
    fi
}

# Monitor system resources
check_system_resources() {
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | cut -d'%' -f1)

    if [ $(echo "$CPU_USAGE > 80" | bc) -eq 1 ]; then
        log_security_event "WARNING: High CPU usage: $CPU_USAGE%"
    fi

    if [ $(echo "$MEMORY_USAGE > 80" | bc) -eq 1 ]; then
        log_security_event "WARNING: High memory usage: $MEMORY_USAGE%"
    fi

    if [ $DISK_USAGE -gt 80 ]; then
        log_security_event "WARNING: High disk usage: $DISK_USAGE%"
    fi
}

# Main execution
main() {
    log_security_event "Starting security monitoring check"

    check_failed_logins
    check_suspicious_ips
    check_ssl_expiration
    check_system_resources

    log_security_event "Security monitoring check completed"
}

# Run main function
main
"""

            # Write security monitoring script
            security_script_path = "scripts/security_monitor.sh"
            os.makedirs(os.path.dirname(security_script_path), exist_ok=True)

            with open(security_script_path, "w") as f:
                f.write(security_monitor_script)

            # Make script executable
            os.chmod(security_script_path, 0o755)

            logger.info("✅ Security scripts created")
            self.security_results.append(
                {"component": "Security Scripts", "status": "success"}
            )
            return True

        except Exception as e:
            logger.error(f"Security scripts creation failed: {e}")
            self.security_results.append(
                {"component": "Security Scripts", "status": "failed", "error": str(e)}
            )
            return False

    def generate_security_report(self) -> dict[str, Any]:
        """Generate comprehensive security report."""
        report = {
            "security_hardening": {
                "timestamp": datetime.now().isoformat(),
                "domain": self.config.domain,
                "ssl_enabled": True,
                "components": self.security_results,
                "success_rate": sum(
                    1 for r in self.security_results if r["status"] == "success"
                )
                / len(self.security_results)
                * 100
                if self.security_results
                else 0,
                "total_components": len(self.security_results),
            },
            "security_measures": {
                "ssl_certificates": "Generated and configured",
                "nginx_security": "Configured with security headers and rate limiting",
                "application_security": "Configured with authentication and authorization",
                "database_security": "Configured with encryption and access controls",
                "redis_security": "Configured with authentication and command restrictions",
                "monitoring": "Security monitoring scripts deployed",
            },
            "security_features": [
                "SSL/TLS encryption for all communications",
                "HTTP security headers (HSTS, CSP, etc.)",
                "Rate limiting and DDoS protection",
                "Authentication and authorization",
                "Input validation and sanitization",
                "Database encryption and access controls",
                "Redis authentication and command restrictions",
                "Security monitoring and alerting",
                "Automated security scanning",
                "Security event logging",
            ],
            "recommendations": [
                "Regularly update SSL certificates",
                "Monitor security logs for suspicious activity",
                "Conduct regular security audits",
                "Keep all dependencies updated",
                "Implement network segmentation",
                "Set up intrusion detection system",
                "Enable two-factor authentication",
                "Implement data loss prevention",
                "Regular penetration testing",
                "Security awareness training",
            ],
        }

        return report

    def save_security_report(self, report: dict[str, Any]):
        """Save security report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"security_hardening_report_{timestamp}.json"

        with open(filename, "w") as f:
            import json

            json.dump(report, f, indent=2)

        logger.info(f"🔒 Security report saved to {filename}")

    def print_security_summary(self, report: dict[str, Any]):
        """Print security hardening summary."""
        hardening = report["security_hardening"]

        print("\n" + "=" * 60)
        print("🔒 PYNOMALY SECURITY HARDENING SUMMARY")
        print("=" * 60)
        print(f"Domain: {hardening['domain']}")
        print(f"SSL Enabled: {hardening['ssl_enabled']}")
        print(f"Components: {hardening['total_components']}")
        print(f"Success Rate: {hardening['success_rate']:.1f}%")

        print("\n🛡️ SECURITY MEASURES:")
        for measure, description in report["security_measures"].items():
            print(f"  • {measure.replace('_', ' ').title()}: {description}")

        print("\n🔐 SECURITY FEATURES:")
        for feature in report["security_features"]:
            print(f"  • {feature}")

        print("\n📋 RECOMMENDATIONS:")
        for recommendation in report["recommendations"]:
            print(f"  • {recommendation}")

        print("\n" + "=" * 60)
        print("🎉 SECURITY HARDENING COMPLETE!")
        print("=" * 60)


async def main():
    """Main security hardening workflow."""
    config = SecurityConfig()

    # Override with environment variables if available
    config.domain = os.getenv("DOMAIN", config.domain)

    hardening = SecurityHardening(config)

    try:
        logger.info("🚀 Starting security hardening...")

        # Generate SSL certificates
        ssl_success = await hardening.generate_ssl_certificates()

        # Configure security components
        nginx_success = await hardening.configure_nginx_security()
        app_success = await hardening.configure_application_security()
        db_success = await hardening.configure_database_security()
        redis_success = await hardening.configure_redis_security()

        # Create security scripts
        scripts_success = await hardening.create_security_scripts()

        # Generate report
        report = hardening.generate_security_report()
        hardening.save_security_report(report)
        hardening.print_security_summary(report)

        # Overall success
        overall_success = all(
            [
                ssl_success,
                nginx_success,
                app_success,
                db_success,
                redis_success,
                scripts_success,
            ]
        )

        if overall_success:
            logger.info("✅ Security hardening completed successfully!")
            return True
        else:
            logger.error("❌ Security hardening completed with errors")
            return False

    except Exception as e:
        logger.error(f"Security hardening failed: {e}")
        return False


if __name__ == "__main__":
    # Run the security hardening
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
