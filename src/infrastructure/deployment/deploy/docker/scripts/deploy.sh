#!/bin/bash

# Production Deployment Script for Pynomaly
# This script handles the complete deployment of Pynomaly in production

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/deploy/docker"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.production.yml"
ENV_FILE="$PROJECT_ROOT/.env.production"
DATA_DIR="${DATA_PATH:-/opt/pynomaly/data}"
LOG_DIR="${LOG_PATH:-/opt/pynomaly/logs}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root for production deployment"
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed"
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error_exit "Docker Compose is not installed"
    fi

    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        error_exit "Environment file not found: $ENV_FILE"
    fi

    # Check if critical environment variables are set
    source "$ENV_FILE"

    if [[ "$JWT_SECRET_KEY" == "CHANGE_THIS_IN_PRODUCTION" ]]; then
        error_exit "JWT_SECRET_KEY must be changed from default value"
    fi

    if [[ "$POSTGRES_PASSWORD" == "CHANGE_THIS_IN_PRODUCTION" ]]; then
        error_exit "POSTGRES_PASSWORD must be changed from default value"
    fi

    if [[ "$REDIS_PASSWORD" == "CHANGE_THIS_IN_PRODUCTION" ]]; then
        error_exit "REDIS_PASSWORD must be changed from default value"
    fi

    log_success "Prerequisites check passed"
}

# Create directories
create_directories() {
    log_info "Creating required directories..."

    mkdir -p "$DATA_DIR"/{postgres,redis,kafka,zookeeper,zookeeper-logs,prometheus,grafana,elasticsearch,nginx-logs}
    mkdir -p "$DATA_DIR"/{pynomaly-storage,pynomaly-logs,pynomaly-temp,pynomaly-config}
    mkdir -p "$LOG_DIR"

    # Set permissions
    chmod 700 "$DATA_DIR"
    chmod 755 "$LOG_DIR"

    log_success "Directories created successfully"
}

# Generate SSL certificates if needed
generate_ssl_certificates() {
    log_info "Checking SSL certificates..."

    local cert_dir="/etc/ssl/certs"
    local key_dir="/etc/ssl/private"

    if [[ ! -f "$cert_dir/pynomaly.crt" ]] || [[ ! -f "$key_dir/pynomaly.key" ]]; then
        log_warn "SSL certificates not found, generating self-signed certificates..."

        mkdir -p "$cert_dir" "$key_dir"

        # Generate self-signed certificate (for development/testing)
        openssl req -x509 -newkey rsa:4096 -keyout "$key_dir/pynomaly.key" -out "$cert_dir/pynomaly.crt" \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=pynomaly.local"

        chmod 600 "$key_dir/pynomaly.key"
        chmod 644 "$cert_dir/pynomaly.crt"

        log_warn "Self-signed certificates generated. Replace with proper certificates in production."
    fi

    log_success "SSL certificates ready"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."

    cd "$PROJECT_ROOT"

    # Set build args
    export BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    export VERSION=${VERSION:-"1.0.0"}
    export VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

    # Build production image
    docker build \
        -f "$DOCKER_DIR/Dockerfile.production" \
        --target runtime \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$VCS_REF" \
        -t "pynomaly:production-$VERSION" \
        .

    log_success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."

    cd "$PROJECT_ROOT"

    # Pull external images
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull postgres redis-cluster kafka zookeeper prometheus grafana elasticsearch kibana nginx

    # Start infrastructure services first
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d postgres redis-cluster

    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    sleep 30

    # Run database migrations
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" run --rm pynomaly-api python -m alembic upgrade head

    # Start remaining services
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d

    log_success "Services deployed successfully"
}

# Health check
health_check() {
    log_info "Performing health checks..."

    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s http://localhost:8000/api/health/ready > /dev/null; then
            log_success "Health check passed"
            return 0
        fi

        log_info "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
        ((attempt++))
    done

    error_exit "Health check failed after $max_attempts attempts"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."

    # Check if Prometheus is accessible
    if curl -f -s http://localhost:9090/-/healthy > /dev/null; then
        log_success "Prometheus is running"
    else
        log_warn "Prometheus is not accessible"
    fi

    # Check if Grafana is accessible
    if curl -f -s http://localhost:3000/api/health > /dev/null; then
        log_success "Grafana is running"
    else
        log_warn "Grafana is not accessible"
    fi

    log_success "Monitoring setup completed"
}

# Cleanup old resources
cleanup() {
    log_info "Cleaning up old resources..."

    # Remove old containers
    docker container prune -f

    # Remove old images
    docker image prune -f

    # Remove old volumes (commented out for safety)
    # docker volume prune -f

    log_success "Cleanup completed"
}

# Backup before deployment
backup_before_deployment() {
    log_info "Creating backup before deployment..."

    local backup_dir="/opt/pynomaly/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    # Backup database
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T postgres pg_dump -U pynomaly pynomaly_prod > "$backup_dir/database_backup.sql"

    # Backup volumes
    tar -czf "$backup_dir/volumes_backup.tar.gz" -C "$DATA_DIR" .

    log_success "Backup created at $backup_dir"
}

# Main deployment function
main() {
    log_info "Starting Pynomaly production deployment..."

    # Check if this is an update
    local is_update=false
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps | grep -q "pynomaly-api"; then
        is_update=true
        log_info "Detected existing deployment, performing update..."
    fi

    # Backup if updating
    if [[ "$is_update" == true ]]; then
        backup_before_deployment
    fi

    # Execute deployment steps
    check_root
    check_prerequisites
    create_directories
    generate_ssl_certificates
    build_images
    deploy_services
    health_check
    setup_monitoring
    cleanup

    log_success "Pynomaly production deployment completed successfully!"
    log_info "Services accessible at:"
    log_info "  - API: http://localhost:8000"
    log_info "  - Web UI: http://localhost:80"
    log_info "  - Grafana: http://localhost:3000"
    log_info "  - Prometheus: http://localhost:9090"
    log_info "  - Kibana: http://localhost:5601"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
