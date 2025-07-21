#!/usr/bin/env python3
"""
Production Configuration Generator for anomaly_detection

This script generates production-ready configuration files for anomaly_detection deployment.
"""

import os
import secrets
import string
from pathlib import Path
from typing import Any

import yaml


def generate_secret_key(length: int = 64) -> str:
    """Generate a secure random secret key."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


def generate_database_url(
    host: str = "localhost",
    port: int = 5432,
    database: str = "anomaly_detection_prod",
    username: str = "anomaly_detection_user",
    password: str = None,
) -> str:
    """Generate database URL."""
    if password is None:
        password = generate_secret_key(16)
    return f"postgresql://{username}:{password}@{host}:{port}/{database}"


def generate_environment_config() -> dict[str, Any]:
    """Generate environment configuration."""
    return {
        # Application
        "ANOMALY_DETECTION_ENV": "production",
        "ANOMALY_DETECTION_DEBUG": "false",
        "ANOMALY_DETECTION_LOG_LEVEL": "INFO",
        "ANOMALY_DETECTION_VERSION": "0.1.1",
        # Security
        "SECRET_KEY": generate_secret_key(),
        "API_KEY_SALT": generate_secret_key(32),
        "JWT_SECRET": generate_secret_key(64),
        "API_KEY": generate_api_key(),
        # Database
        "DATABASE_URL": generate_database_url(),
        "DATABASE_POOL_SIZE": "10",
        "DATABASE_MAX_OVERFLOW": "20",
        # Redis
        "REDIS_URL": "redis://localhost:6379/0",
        "REDIS_MAX_CONNECTIONS": "100",
        # Performance
        "MAX_WORKERS": "8",
        "WORKER_TIMEOUT": "300",
        "MAX_REQUESTS_PER_WORKER": "1000",
        "WORKER_CLASS": "uvicorn.workers.UvicornWorker",
        # Monitoring
        "PROMETHEUS_PORT": "9090",
        "GRAFANA_PORT": "3000",
        "ENABLE_METRICS": "true",
        "METRICS_ENDPOINT": "/metrics",
        # Storage
        "STORAGE_PATH": "/app/data",
        "BACKUP_PATH": "/app/backups",
        "LOG_PATH": "/app/logs",
        # Security
        "CORS_ORIGINS": "https://your-frontend-domain.com",
        "RATE_LIMIT_REQUESTS": "1000",
        "RATE_LIMIT_WINDOW": "60",
        "ENABLE_RATE_LIMITING": "true",
        # Features
        "ENABLE_AUTOML": "true",
        "ENABLE_EXPLAINABILITY": "true",
        "ENABLE_ADVANCED_MONITORING": "true",
        "ENABLE_DRIFT_DETECTION": "true",
        # Deployment
        "DEPLOYMENT_TYPE": "docker",
        "CONTAINER_NAME": "anomaly_detection-prod",
        "HEALTHCHECK_INTERVAL": "30",
        "RESTART_POLICY": "unless-stopped",
    }


def generate_docker_compose() -> dict[str, Any]:
    """Generate Docker Compose configuration."""
    return {
        "version": "3.8",
        "services": {
            "anomaly_detection-api": {
                "build": ".",
                "ports": ["8000:8000"],
                "environment": [
                    "ANOMALY_DETECTION_ENV=production",
                    "DATABASE_URL=${DATABASE_URL}",
                    "REDIS_URL=${REDIS_URL}",
                    "SECRET_KEY=${SECRET_KEY}",
                ],
                "depends_on": ["postgres", "redis"],
                "volumes": [
                    "./data:/app/data",
                    "./logs:/app/logs",
                    "./backups:/app/backups",
                ],
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                    "start_period": "60s",
                },
            },
            "anomaly_detection-worker": {
                "build": ".",
                "command": "celery -A anomaly_detection.infrastructure.celery worker --loglevel=info",
                "depends_on": ["postgres", "redis"],
                "volumes": [
                    "./data:/app/data",
                    "./logs:/app/logs",
                ],
                "restart": "unless-stopped",
                "environment": [
                    "ANOMALY_DETECTION_ENV=production",
                    "DATABASE_URL=${DATABASE_URL}",
                    "REDIS_URL=${REDIS_URL}",
                ],
            },
            "postgres": {
                "image": "postgres:13",
                "environment": [
                    "POSTGRES_DB=anomaly_detection_prod",
                    "POSTGRES_USER=anomaly_detection_user",
                    "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}",
                ],
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
                "ports": ["5432:5432"],
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": [
                        "CMD-SHELL",
                        "pg_isready -U anomaly_detection_user -d anomaly_detection_prod",
                    ],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 5,
                },
            },
            "redis": {
                "image": "redis:6-alpine",
                "command": "redis-server --appendonly yes",
                "volumes": ["redis_data:/data"],
                "ports": ["6379:6379"],
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": ["CMD", "redis-cli", "ping"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                },
            },
            "prometheus": {
                "image": "prom/prometheus:latest",
                "ports": ["9090:9090"],
                "volumes": [
                    "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml",
                    "prometheus_data:/prometheus",
                ],
                "restart": "unless-stopped",
                "command": [
                    "--config.file=/etc/prometheus/prometheus.yml",
                    "--storage.tsdb.path=/prometheus",
                    "--web.console.libraries=/etc/prometheus/console_libraries",
                    "--web.console.templates=/etc/prometheus/consoles",
                    "--web.enable-lifecycle",
                ],
            },
            "grafana": {
                "image": "grafana/grafana:latest",
                "ports": ["3000:3000"],
                "environment": [
                    "GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}",
                    "GF_USERS_ALLOW_SIGN_UP=false",
                ],
                "volumes": [
                    "grafana_data:/var/lib/grafana",
                    "./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards",
                    "./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources",
                ],
                "restart": "unless-stopped",
            },
        },
        "volumes": {
            "postgres_data": {},
            "redis_data": {},
            "prometheus_data": {},
            "grafana_data": {},
        },
        "networks": {
            "default": {
                "name": "anomaly_detection-network",
            },
        },
    }


def generate_nginx_config() -> str:
    """Generate Nginx configuration."""
    return """
# anomaly_detection Production Nginx Configuration

upstream anomaly_detection_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    listen 80;
    server_name your-domain.com;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL configuration
    ssl_certificate /etc/ssl/certs/anomaly_detection.crt;
    ssl_certificate_key /etc/ssl/private/anomaly_detection.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Logging
    access_log /var/log/nginx/anomaly_detection.access.log;
    error_log /var/log/nginx/anomaly_detection.error.log;

    # API endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;

        proxy_pass http://anomaly_detection_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_http_version 1.1;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    # WebSocket support
    location /ws/ {
        proxy_pass http://anomaly_detection_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Health check
    location /health {
        access_log off;
        proxy_pass http://anomaly_detection_backend;
        proxy_set_header Host $host;
    }

    # Metrics (protected)
    location /metrics {
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        deny all;

        proxy_pass http://anomaly_detection_backend;
        proxy_set_header Host $host;
    }

    # Static files
    location /static/ {
        alias /app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        gzip on;
        gzip_types text/css application/javascript application/json;
    }

    # Deny access to sensitive files
    location ~ /\\.ht {
        deny all;
    }

    location ~ /\\.(git|svn) {
        deny all;
    }
}
"""


def generate_prometheus_config() -> dict[str, Any]:
    """Generate Prometheus configuration."""
    return {
        "global": {
            "scrape_interval": "15s",
            "evaluation_interval": "15s",
        },
        "scrape_configs": [
            {
                "job_name": "anomaly_detection",
                "static_configs": [{"targets": ["localhost:8000"]}],
                "metrics_path": "/metrics",
                "scrape_interval": "5s",
            },
            {
                "job_name": "node",
                "static_configs": [{"targets": ["localhost:9100"]}],
            },
            {
                "job_name": "postgres",
                "static_configs": [{"targets": ["localhost:9187"]}],
            },
            {
                "job_name": "redis",
                "static_configs": [{"targets": ["localhost:9121"]}],
            },
        ],
        "rule_files": ["alert_rules.yml"],
        "alerting": {
            "alertmanagers": [
                {
                    "static_configs": [{"targets": ["localhost:9093"]}],
                }
            ]
        },
    }


def generate_grafana_datasource() -> dict[str, Any]:
    """Generate Grafana datasource configuration."""
    return {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": "http://prometheus:9090",
                "isDefault": True,
            }
        ],
    }


def generate_systemd_service() -> str:
    """Generate systemd service file."""
    return """
[Unit]
Description=anomaly_detection Production Service
After=network.target postgresql.service redis.service

[Service]
Type=forking
User=anomaly_detection
Group=anomaly_detection
WorkingDirectory=/app
Environment=ANOMALY_DETECTION_ENV=production
ExecStart=/usr/local/bin/gunicorn -c gunicorn.conf.py anomaly_detection.presentation.api.app:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure

[Install]
WantedBy=multi-user.target
"""


def generate_backup_script() -> str:
    """Generate backup script."""
    return """#!/bin/bash
# anomaly_detection Production Backup Script

set -e

# Configuration
BACKUP_DIR="/app/backups"
DB_NAME="anomaly_detection_prod"
DB_USER="anomaly_detection_user"
DATA_DIR="/app/data"
RETENTION_DAYS=7

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Database backup
echo "Starting database backup..."
pg_dump -h localhost -U "$DB_USER" -d "$DB_NAME" | gzip > "$BACKUP_DIR/db_$TIMESTAMP.sql.gz"

# Data backup
echo "Starting data backup..."
tar czf "$BACKUP_DIR/data_$TIMESTAMP.tar.gz" -C "$DATA_DIR" .

# Upload to S3 (optional)
if [[ -n "$AWS_S3_BUCKET" ]]; then
    echo "Uploading to S3..."
    aws s3 cp "$BACKUP_DIR/db_$TIMESTAMP.sql.gz" "s3://$AWS_S3_BUCKET/backups/"
    aws s3 cp "$BACKUP_DIR/data_$TIMESTAMP.tar.gz" "s3://$AWS_S3_BUCKET/backups/"
fi

# Cleanup old backups
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -name "*.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed successfully!"
"""


def generate_monitoring_script() -> str:
    """Generate monitoring script."""
    return """#!/bin/bash
# anomaly_detection Production Monitoring Script

set -e

# Configuration
API_URL="http://localhost:8000"
ALERT_EMAIL="admin@your-domain.com"
LOG_FILE="/var/log/anomaly_detection/monitoring.log"

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to send alerts
send_alert() {
    local subject="$1"
    local message="$2"

    echo "$message" | mail -s "$subject" "$ALERT_EMAIL"
    log "ALERT: $subject"
}

# Health check
health_check() {
    local response
    response=$(curl -s -w "%{http_code}" -o /dev/null "$API_URL/health" || echo "000")

    if [[ "$response" != "200" ]]; then
        send_alert "anomaly_detection Health Check Failed" "Health check returned status: $response"
        return 1
    fi

    log "Health check passed"
    return 0
}

# Performance check
performance_check() {
    local response_time
    response_time=$(curl -s -w "%{time_total}" -o /dev/null "$API_URL/health")

    if (( $(echo "$response_time > 5.0" | bc -l) )); then
        send_alert "anomaly_detection Performance Degradation" "Response time: ${response_time}s"
        return 1
    fi

    log "Performance check passed (${response_time}s)"
    return 0
}

# Memory check
memory_check() {
    local memory_usage
    memory_usage=$(docker stats --no-stream --format "table {{.MemUsage}}" | grep -v "MEM" | head -1)

    log "Memory usage: $memory_usage"
    # Add logic to parse and check memory usage
}

# Disk space check
disk_check() {
    local disk_usage
    disk_usage=$(df -h /app | awk 'NR==2{print $5}' | sed 's/%//')

    if [[ "$disk_usage" -gt 80 ]]; then
        send_alert "anomaly_detection Disk Space Warning" "Disk usage: ${disk_usage}%"
        return 1
    fi

    log "Disk usage: ${disk_usage}%"
    return 0
}

# Main monitoring loop
main() {
    log "Starting monitoring checks..."

    health_check || exit 1
    performance_check || exit 1
    memory_check || exit 1
    disk_check || exit 1

    log "All checks passed"
}

# Run main function
main "$@"
"""


def main():
    """Main function to generate all configuration files."""
    # Create directories
    base_dir = Path("production_config")
    base_dir.mkdir(exist_ok=True)

    monitoring_dir = base_dir / "monitoring"
    monitoring_dir.mkdir(exist_ok=True)

    grafana_dir = monitoring_dir / "grafana"
    grafana_dir.mkdir(exist_ok=True)

    scripts_dir = base_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # Generate environment configuration
    env_config = generate_environment_config()
    with open(base_dir / ".env.production", "w") as f:
        for key, value in env_config.items():
            f.write(f"{key}={value}\n")

    # Generate Docker Compose
    docker_config = generate_docker_compose()
    with open(base_dir / "docker-compose.prod.yml", "w") as f:
        yaml.dump(docker_config, f, default_flow_style=False, sort_keys=False)

    # Generate Nginx configuration
    with open(base_dir / "nginx.conf", "w") as f:
        f.write(generate_nginx_config())

    # Generate Prometheus configuration
    prometheus_config = generate_prometheus_config()
    with open(monitoring_dir / "prometheus.yml", "w") as f:
        yaml.dump(prometheus_config, f, default_flow_style=False)

    # Generate Grafana datasource
    grafana_datasource = generate_grafana_datasource()
    with open(grafana_dir / "datasource.yml", "w") as f:
        yaml.dump(grafana_datasource, f, default_flow_style=False)

    # Generate systemd service
    with open(base_dir / "anomaly_detection.service", "w") as f:
        f.write(generate_systemd_service())

    # Generate scripts
    with open(scripts_dir / "backup.sh", "w") as f:
        f.write(generate_backup_script())

    with open(scripts_dir / "monitoring.sh", "w") as f:
        f.write(generate_monitoring_script())

    # Make scripts executable
    os.chmod(scripts_dir / "backup.sh", 0o755)
    os.chmod(scripts_dir / "monitoring.sh", 0o755)

    # Generate summary
    summary = {
        "generated_files": [
            ".env.production",
            "docker-compose.prod.yml",
            "nginx.conf",
            "anomaly_detection.service",
            "monitoring/prometheus.yml",
            "monitoring/grafana/datasource.yml",
            "scripts/backup.sh",
            "scripts/monitoring.sh",
        ],
        "next_steps": [
            "Review and customize the generated configurations",
            "Set up SSL certificates",
            "Configure your domain name",
            "Set up monitoring alerts",
            "Test the deployment in staging",
            "Run security scans",
            "Set up backup schedules",
        ],
        "security_notes": [
            "Change all default passwords",
            "Review and restrict network access",
            "Set up firewall rules",
            "Enable SSL/TLS encryption",
            "Configure rate limiting",
            "Set up intrusion detection",
        ],
    }

    with open(base_dir / "README.md", "w") as f:
        f.write("# anomaly_detection Production Configuration\n\n")
        f.write("This directory contains generated production configuration files.\n\n")
        f.write("## Generated Files\n\n")
        for file in summary["generated_files"]:
            f.write(f"- {file}\n")
        f.write("\n## Next Steps\n\n")
        for step in summary["next_steps"]:
            f.write(f"1. {step}\n")
        f.write("\n## Security Notes\n\n")
        for note in summary["security_notes"]:
            f.write(f"- {note}\n")

    print("✅ Production configuration generated successfully!")
    print(f"📁 Files created in: {base_dir}")
    print("\n🔐 Security reminders:")
    print("  - Change all default passwords")
    print("  - Review network access rules")
    print("  - Set up SSL certificates")
    print("  - Configure monitoring alerts")
    print("\n📖 See README.md for complete setup instructions")


if __name__ == "__main__":
    main()
