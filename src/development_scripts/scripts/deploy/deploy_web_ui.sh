#!/bin/bash

# anomaly_detection Web UI Deployment Script
# This script deploys the web UI infrastructure to production

set -e

# Configuration
PROJECT_NAME="anomaly_detection"
DEPLOY_USER="anomaly_detection"
DEPLOY_HOST="your-server.com"
DEPLOY_DIR="/opt/anomaly_detection"
BACKUP_DIR="/opt/anomaly_detection/backups"
LOG_DIR="/var/log/anomaly_detection"
SERVICE_NAME="anomaly_detection-web"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Create backup of current deployment
create_backup() {
    log_info "Creating backup of current deployment..."

    if [ -d "$DEPLOY_DIR" ]; then
        BACKUP_NAME="anomaly_detection-backup-$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        tar -czf "$BACKUP_DIR/$BACKUP_NAME.tar.gz" -C "$DEPLOY_DIR" .
        log_info "Backup created: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
    else
        log_info "No existing deployment found, skipping backup"
    fi
}

# Set up directories
setup_directories() {
    log_info "Setting up deployment directories..."

    mkdir -p "$DEPLOY_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "/etc/ssl/certs"
    mkdir -p "/etc/ssl/private"

    # Set permissions
    chown -R "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_DIR"
    chown -R "$DEPLOY_USER:$DEPLOY_USER" "$LOG_DIR"
    chmod 750 "$DEPLOY_DIR"
    chmod 750 "$LOG_DIR"
}

# Deploy application files
deploy_files() {
    log_info "Deploying application files..."

    # Copy application files (adjust source path as needed)
    cp -r ./src "$DEPLOY_DIR/"
    cp -r ./scripts "$DEPLOY_DIR/"
    cp ./requirements.txt "$DEPLOY_DIR/"
    cp ./pyproject.toml "$DEPLOY_DIR/"

    # Copy production environment file
    cp ./.env.production "$DEPLOY_DIR/.env"

    # Set permissions
    chown -R "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_DIR"
    chmod 644 "$DEPLOY_DIR/.env"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."

    cd "$DEPLOY_DIR"

    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install dependencies
    pip install -r requirements.txt

    # Install production dependencies
    pip install gunicorn uvicorn[standard] psycopg2-binary redis

    log_info "Dependencies installed successfully"
}

# Configure systemd service
configure_systemd() {
    log_info "Configuring systemd service..."

    cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=anomaly_detection Web UI
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=$DEPLOY_USER
Group=$DEPLOY_USER
WorkingDirectory=$DEPLOY_DIR
Environment=PATH=$DEPLOY_DIR/venv/bin
ExecStart=$DEPLOY_DIR/venv/bin/python scripts/run/run_web_app.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DEPLOY_DIR $LOG_DIR

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"

    log_info "Systemd service configured"
}

# Configure nginx reverse proxy
configure_nginx() {
    log_info "Configuring nginx reverse proxy..."

    cat > "/etc/nginx/sites-available/$PROJECT_NAME" << EOF
server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://\$server_name\$request_uri;
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

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Static files
    location /static/ {
        alias $DEPLOY_DIR/src/anomaly_detection/presentation/web/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API and web requests
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

    # Enable the site
    ln -sf "/etc/nginx/sites-available/$PROJECT_NAME" "/etc/nginx/sites-enabled/$PROJECT_NAME"

    # Test nginx configuration
    nginx -t

    log_info "Nginx configuration complete"
}

# Set up log rotation
setup_log_rotation() {
    log_info "Setting up log rotation..."

    cat > "/etc/logrotate.d/$PROJECT_NAME" << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $DEPLOY_USER $DEPLOY_USER
    postrotate
        systemctl reload $SERVICE_NAME
    endscript
}
EOF

    log_info "Log rotation configured"
}

# Set up monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."

    # Create monitoring script
    cat > "$DEPLOY_DIR/scripts/monitor_health.sh" << EOF
#!/bin/bash

# Health check script
SERVICE_NAME="$SERVICE_NAME"
LOG_FILE="$LOG_DIR/health_check.log"

check_service() {
    if systemctl is-active --quiet \$SERVICE_NAME; then
        echo "\$(date): Service \$SERVICE_NAME is running" >> \$LOG_FILE
        return 0
    else
        echo "\$(date): Service \$SERVICE_NAME is not running" >> \$LOG_FILE
        return 1
    fi
}

check_http() {
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo "\$(date): HTTP health check passed" >> \$LOG_FILE
        return 0
    else
        echo "\$(date): HTTP health check failed" >> \$LOG_FILE
        return 1
    fi
}

# Run checks
if check_service && check_http; then
    exit 0
else
    echo "\$(date): Health check failed, attempting restart" >> \$LOG_FILE
    systemctl restart \$SERVICE_NAME
    exit 1
fi
EOF

    chmod +x "$DEPLOY_DIR/scripts/monitor_health.sh"

    # Add to crontab
    (crontab -l 2>/dev/null; echo "*/5 * * * * $DEPLOY_DIR/scripts/monitor_health.sh") | crontab -

    log_info "Monitoring configured"
}

# Start services
start_services() {
    log_info "Starting services..."

    # Start and enable services
    systemctl start "$SERVICE_NAME"
    systemctl restart nginx

    # Check service status
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "anomaly_detection web service started successfully"
    else
        log_error "Failed to start anomaly_detection web service"
        exit 1
    fi

    if systemctl is-active --quiet nginx; then
        log_info "Nginx service started successfully"
    else
        log_error "Failed to start Nginx service"
        exit 1
    fi
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."

    sleep 5

    # Check if application is responding
    if curl -sf "http://localhost:8000/health" > /dev/null; then
        log_info "✓ Application health check passed"
    else
        log_error "✗ Application health check failed"
        exit 1
    fi

    # Check API endpoints
    if curl -sf "http://localhost:8000/api/ui/health" > /dev/null; then
        log_info "✓ API health check passed"
    else
        log_error "✗ API health check failed"
        exit 1
    fi

    log_info "All health checks passed"
}

# Main deployment function
main() {
    log_info "Starting anomaly_detection Web UI deployment..."

    check_root
    create_backup
    setup_directories
    deploy_files
    install_dependencies
    configure_systemd
    configure_nginx
    setup_log_rotation
    setup_monitoring
    start_services
    run_health_checks

    log_info "Deployment completed successfully!"
    log_info "Web UI is available at: https://your-domain.com"
    log_info "API documentation: https://your-domain.com/api/v1/docs"
    log_info "Logs location: $LOG_DIR"
    log_info "Service management: systemctl [start|stop|restart|status] $SERVICE_NAME"
}

# Run deployment
main "$@"
