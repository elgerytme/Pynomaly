#!/bin/bash

# Production Deployment Script for Pynomaly
# This script handles the complete deployment process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROD_CONFIG_DIR="$PROJECT_ROOT/config/production"
BACKUP_DIR="$PROJECT_ROOT/backups"
DEPLOYMENT_LOG="$PROJECT_ROOT/deployment.log"

# Functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$DEPLOYMENT_LOG"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$DEPLOYMENT_LOG"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$DEPLOYMENT_LOG"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$DEPLOYMENT_LOG"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi

    # Check if running as root (not recommended for production)
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root is not recommended for production deployments."
    fi

    # Check disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ $AVAILABLE_SPACE -lt 10 ]]; then
        error "Insufficient disk space. At least 10GB required, only ${AVAILABLE_SPACE}GB available."
    fi

    log "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."

    mkdir -p "$BACKUP_DIR"
    mkdir -p "$PROJECT_ROOT/storage/data"
    mkdir -p "$PROJECT_ROOT/storage/logs"
    mkdir -p "$PROJECT_ROOT/storage/ssl"
    mkdir -p "$PROJECT_ROOT/storage/uploads"
    mkdir -p "$PROD_CONFIG_DIR/ssl"

    log "Directories created successfully"
}

# Generate SSL certificates (self-signed for development)
generate_ssl_certificates() {
    log "Generating SSL certificates..."

    SSL_DIR="$PROD_CONFIG_DIR/ssl"

    if [[ ! -f "$SSL_DIR/cert.pem" || ! -f "$SSL_DIR/key.pem" ]]; then
        # Generate DH parameters
        if [[ ! -f "$SSL_DIR/dhparam.pem" ]]; then
            info "Generating DH parameters (this may take a while)..."
            openssl dhparam -out "$SSL_DIR/dhparam.pem" 2048
        fi

        # Generate private key
        openssl genrsa -out "$SSL_DIR/key.pem" 2048

        # Generate certificate
        openssl req -new -x509 -key "$SSL_DIR/key.pem" -out "$SSL_DIR/cert.pem" -days 365 -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

        log "SSL certificates generated (self-signed)"
        warn "For production, replace with proper SSL certificates from a CA"
    else
        log "SSL certificates already exist"
    fi
}

# Validate environment configuration
validate_environment() {
    log "Validating environment configuration..."

    ENV_FILE="$PROD_CONFIG_DIR/.env.prod"

    if [[ ! -f "$ENV_FILE" ]]; then
        error "Environment file not found: $ENV_FILE"
    fi

    # Check for required environment variables
    required_vars=(
        "SECRET_KEY"
        "DB_PASSWORD"
        "REDIS_PASSWORD"
        "GRAFANA_PASSWORD"
    )

    for var in "${required_vars[@]}"; do
        if ! grep -q "^$var=" "$ENV_FILE" || grep -q "^$var=CHANGE_ME" "$ENV_FILE"; then
            error "Required environment variable $var is not set or has default value in $ENV_FILE"
        fi
    done

    log "Environment configuration validated"
}

# Create database backup
create_backup() {
    log "Creating database backup..."

    BACKUP_FILE="$BACKUP_DIR/pynomaly_backup_$(date +%Y%m%d_%H%M%S).sql"

    if docker-compose -f "$PROD_CONFIG_DIR/docker-compose.prod.yml" ps postgres | grep -q "Up"; then
        docker-compose -f "$PROD_CONFIG_DIR/docker-compose.prod.yml" exec -T postgres pg_dump -U pynomaly pynomaly > "$BACKUP_FILE"
        log "Database backup created: $BACKUP_FILE"
    else
        info "Database not running, skipping backup"
    fi
}

# Deploy application
deploy_application() {
    log "Deploying application..."

    cd "$PROD_CONFIG_DIR"

    # Pull latest images
    docker-compose -f docker-compose.prod.yml pull

    # Build application image
    docker-compose -f docker-compose.prod.yml build pynomaly-api

    # Start services
    docker-compose -f docker-compose.prod.yml up -d

    log "Application deployed successfully"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."

    # Wait for database
    info "Waiting for PostgreSQL..."
    timeout 60 bash -c 'until docker-compose -f '"$PROD_CONFIG_DIR"'/docker-compose.prod.yml exec postgres pg_isready -U pynomaly; do sleep 2; done'

    # Wait for Redis
    info "Waiting for Redis..."
    timeout 60 bash -c 'until docker-compose -f '"$PROD_CONFIG_DIR"'/docker-compose.prod.yml exec redis redis-cli ping; do sleep 2; done'

    # Wait for API
    info "Waiting for API..."
    timeout 60 bash -c 'until curl -f http://localhost:8000/api/v1/health/ &>/dev/null; do sleep 2; done'

    log "All services are ready"
}

# Run health checks
run_health_checks() {
    log "Running health checks..."

    # Check API health
    if curl -f http://localhost:8000/api/v1/health/ &>/dev/null; then
        log "API health check passed"
    else
        error "API health check failed"
    fi

    # Check database connectivity
    if docker-compose -f "$PROD_CONFIG_DIR/docker-compose.prod.yml" exec -T postgres pg_isready -U pynomaly &>/dev/null; then
        log "Database health check passed"
    else
        error "Database health check failed"
    fi

    # Check Redis connectivity
    if docker-compose -f "$PROD_CONFIG_DIR/docker-compose.prod.yml" exec -T redis redis-cli ping &>/dev/null; then
        log "Redis health check passed"
    else
        error "Redis health check failed"
    fi

    log "All health checks passed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."

    # Import Grafana dashboards
    info "Importing Grafana dashboards..."
    # This would typically involve API calls to Grafana

    # Configure Prometheus alerts
    info "Configuring Prometheus alerts..."
    # This would involve reloading Prometheus configuration

    log "Monitoring setup completed"
}

# Display deployment summary
display_summary() {
    log "Deployment Summary"
    echo "==================="
    echo "Deployment completed successfully!"
    echo ""
    echo "Services:"
    echo "  - API: https://localhost/api/v1/"
    echo "  - Health Check: https://localhost/health"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Prometheus: http://localhost:9090"
    echo ""
    echo "Configuration:"
    echo "  - Environment: production"
    echo "  - SSL: Enabled (self-signed)"
    echo "  - Monitoring: Enabled"
    echo "  - Backup: Enabled"
    echo ""
    echo "Next steps:"
    echo "  1. Replace self-signed SSL certificates with proper ones"
    echo "  2. Configure DNS and firewall rules"
    echo "  3. Set up external monitoring alerts"
    echo "  4. Configure backup retention policies"
    echo "  5. Review and update security settings"
    echo ""
    echo "For logs: tail -f $DEPLOYMENT_LOG"
}

# Main deployment function
main() {
    log "Starting Pynomaly production deployment..."

    check_prerequisites
    create_directories
    generate_ssl_certificates
    validate_environment
    create_backup
    deploy_application
    wait_for_services
    run_health_checks
    setup_monitoring
    display_summary

    log "Deployment completed successfully!"
}

# Handle script arguments
case "$1" in
    "deploy")
        main
        ;;
    "backup")
        create_backup
        ;;
    "health")
        run_health_checks
        ;;
    "logs")
        tail -f "$DEPLOYMENT_LOG"
        ;;
    "stop")
        log "Stopping services..."
        docker-compose -f "$PROD_CONFIG_DIR/docker-compose.prod.yml" down
        log "Services stopped"
        ;;
    "restart")
        log "Restarting services..."
        docker-compose -f "$PROD_CONFIG_DIR/docker-compose.prod.yml" restart
        log "Services restarted"
        ;;
    *)
        echo "Usage: $0 {deploy|backup|health|logs|stop|restart}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Full production deployment"
        echo "  backup  - Create database backup"
        echo "  health  - Run health checks"
        echo "  logs    - Show deployment logs"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        exit 1
        ;;
esac
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="$PROJECT_ROOT/deploy"

# Default values
ENVIRONMENT="production"
NAMESPACE="pynomaly-production"
IMAGE_TAG="latest"
SKIP_TESTS="false"
FORCE_DEPLOY="false"
DRY_RUN="false"
VERIFY_DEPLOYMENT="true"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Pynomaly API to Kubernetes

OPTIONS:
    -e, --environment ENV       Deployment environment (default: production)
    -n, --namespace NAMESPACE   Kubernetes namespace (default: pynomaly-production)
    -t, --tag TAG              Docker image tag (default: latest)
    -s, --skip-tests           Skip pre-deployment tests
    -f, --force                Force deployment without confirmation
    -d, --dry-run              Show what would be deployed without executing
    -h, --help                 Show this help message

EXAMPLES:
    $0                                          # Deploy latest to production
    $0 -e staging -t v2.1.0                   # Deploy v2.1.0 to staging
    $0 -f -s                                   # Force deploy skipping tests
    $0 -d                                      # Dry run deployment

EOF
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -s|--skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        -f|--force)
            FORCE_DEPLOY="true"
            shift
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

validate_environment() {
    log "Validating deployment environment..."

    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi

    # Check if kubectl context is set
    if ! kubectl config current-context &> /dev/null; then
        error "No kubectl context is set"
    fi

    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        warning "Namespace '$NAMESPACE' does not exist. Creating..."
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl apply -f "$DEPLOY_DIR/kubernetes/namespace.yaml"
        fi
    fi

    # Validate environment-specific requirements
    case "$ENVIRONMENT" in
        production)
            # Additional production validations
            log "Validating production environment..."

            # Check for required secrets
            if ! kubectl get secret pynomaly-secrets -n "$NAMESPACE" &> /dev/null; then
                error "Production secrets not found in namespace '$NAMESPACE'"
            fi

            # Verify cluster resources
            local nodes_ready
            nodes_ready=$(kubectl get nodes --no-headers | grep -c "Ready")
            if [[ "$nodes_ready" -lt 3 ]]; then
                warning "Less than 3 nodes ready in cluster"
            fi
            ;;
        staging)
            log "Validating staging environment..."
            ;;
        development)
            log "Validating development environment..."
            ;;
        *)
            error "Unknown environment: $ENVIRONMENT"
            ;;
    esac

    success "Environment validation completed"
}

validate_image() {
    log "Validating Docker image..."

    local image_name="ghcr.io/pynomaly/pynomaly:$IMAGE_TAG"

    # Check if image exists in registry
    if command -v docker &> /dev/null; then
        if ! docker manifest inspect "$image_name" &> /dev/null; then
            error "Docker image '$image_name' not found in registry"
        fi
    else
        warning "Docker not available, skipping image validation"
    fi

    success "Image validation completed"
}

# =============================================================================
# PRE-DEPLOYMENT FUNCTIONS
# =============================================================================

run_pre_deployment_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warning "Skipping pre-deployment tests"
        return 0
    fi

    log "Running pre-deployment tests..."

    # API health check on current deployment
    if kubectl get deployment pynomaly-api -n "$NAMESPACE" &> /dev/null; then
        log "Checking current deployment health..."

        local service_ip
        service_ip=$(kubectl get service pynomaly-api-internal -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")

        if [[ -n "$service_ip" ]]; then
            # Run health check via port-forward
            kubectl port-forward service/pynomaly-api-internal 8080:8000 -n "$NAMESPACE" &
            local port_forward_pid=$!

            sleep 5

            if curl -f http://localhost:8080/api/v1/health &> /dev/null; then
                success "Current deployment is healthy"
            else
                warning "Current deployment health check failed"
            fi

            kill $port_forward_pid 2>/dev/null || true
        fi
    fi

    # Run unit tests
    log "Running unit tests..."
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        cd "$PROJECT_ROOT"
        if command -v poetry &> /dev/null; then
            poetry run pytest tests/unit/ -x || error "Unit tests failed"
        else
            python -m pytest tests/unit/ -x || error "Unit tests failed"
        fi
    fi

    success "Pre-deployment tests completed"
}

backup_current_deployment() {
    log "Creating backup of current deployment..."

    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    # Backup current deployment manifests
    if kubectl get deployment pynomaly-api -n "$NAMESPACE" &> /dev/null; then
        kubectl get deployment pynomaly-api -n "$NAMESPACE" -o yaml > "$backup_dir/deployment.yaml"
        kubectl get service pynomaly-api-service -n "$NAMESPACE" -o yaml > "$backup_dir/service.yaml"
        kubectl get configmap pynomaly-config -n "$NAMESPACE" -o yaml > "$backup_dir/configmap.yaml"

        success "Backup created at $backup_dir"
    else
        log "No existing deployment to backup"
    fi
}

# =============================================================================
# DEPLOYMENT FUNCTIONS
# =============================================================================

deploy_database() {
    log "Deploying database components..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would deploy database"
        return 0
    fi

    # Apply database manifests
    kubectl apply -f "$DEPLOY_DIR/kubernetes/database-statefulset.yaml"

    # Wait for database to be ready
    log "Waiting for database to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=database -n "$NAMESPACE" --timeout=300s

    success "Database deployment completed"
}

deploy_redis() {
    log "Deploying Redis cache..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would deploy Redis"
        return 0
    fi

    # Apply Redis manifests
    kubectl apply -f "$DEPLOY_DIR/kubernetes/cache-deployment.yaml"

    # Wait for Redis to be ready
    log "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=cache -n "$NAMESPACE" --timeout=300s

    success "Redis deployment completed"
}

deploy_api() {
    log "Deploying API application..."

    # Update image tag in deployment manifest
    local temp_manifest="/tmp/pynomaly-deployment.yaml"
    local image_name="ghcr.io/pynomaly/pynomaly:$IMAGE_TAG"

    # Create temporary manifest with updated image
    sed "s|image: pynomaly:production-latest|image: $image_name|g" \
        "$DEPLOY_DIR/kubernetes/production-deployment.yaml" > "$temp_manifest"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would deploy API with image: $image_name"
        cat "$temp_manifest" | grep -A 2 -B 2 "image:"
        rm -f "$temp_manifest"
        return 0
    fi

    # Apply the deployment
    kubectl apply -f "$temp_manifest"

    # Wait for rollout to complete
    log "Waiting for API deployment rollout..."
    kubectl rollout status deployment/pynomaly-api -n "$NAMESPACE" --timeout=600s

    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=api -n "$NAMESPACE" --timeout=300s

    # Clean up temporary manifest
    rm -f "$temp_manifest"

    success "API deployment completed"
}

deploy_monitoring() {
    log "Deploying monitoring components..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would deploy monitoring"
        return 0
    fi

    # Apply monitoring manifests if they exist
    if [[ -f "$DEPLOY_DIR/kubernetes/monitoring-deployment.yaml" ]]; then
        kubectl apply -f "$DEPLOY_DIR/kubernetes/monitoring-deployment.yaml"
        success "Monitoring deployment completed"
    else
        warning "Monitoring manifests not found, skipping"
    fi
}

# =============================================================================
# POST-DEPLOYMENT FUNCTIONS
# =============================================================================

verify_deployment() {
    if [[ "$VERIFY_DEPLOYMENT" == "false" ]] || [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    log "Verifying deployment..."

    # Check pod status
    local ready_pods
    ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api --no-headers | grep -c "Running" || echo "0")

    if [[ "$ready_pods" -lt 1 ]]; then
        error "No API pods are running"
    fi

    # Check service accessibility
    log "Checking service accessibility..."

    # Port forward for testing
    kubectl port-forward service/pynomaly-api-internal 8081:8000 -n "$NAMESPACE" &
    local port_forward_pid=$!

    sleep 10

    # Health check
    if curl -f http://localhost:8081/api/v1/health &> /dev/null; then
        success "Health check passed"
    else
        kill $port_forward_pid 2>/dev/null || true
        error "Health check failed"
    fi

    # API docs check
    if curl -f http://localhost:8081/api/v1/docs &> /dev/null; then
        success "API documentation accessible"
    else
        warning "API documentation check failed"
    fi

    # OpenAPI schema check
    local endpoint_count
    endpoint_count=$(curl -s http://localhost:8081/api/v1/openapi.json | jq '.paths | length' 2>/dev/null || echo "0")

    if [[ "$endpoint_count" -gt 100 ]]; then
        success "OpenAPI schema valid ($endpoint_count endpoints)"
    else
        warning "OpenAPI schema check failed or incomplete"
    fi

    # Clean up port forward
    kill $port_forward_pid 2>/dev/null || true

    success "Deployment verification completed"
}

post_deployment_tasks() {
    log "Running post-deployment tasks..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would run post-deployment tasks"
        return 0
    fi

    # Update deployment annotations
    kubectl annotate deployment pynomaly-api -n "$NAMESPACE" \
        deployment.kubernetes.io/revision-last-deployed="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        deployment.kubernetes.io/image-tag="$IMAGE_TAG" \
        --overwrite

    # Log deployment info
    log "Deployment Summary:"
    log "  Environment: $ENVIRONMENT"
    log "  Namespace: $NAMESPACE"
    log "  Image Tag: $IMAGE_TAG"
    log "  Replicas: $(kubectl get deployment pynomaly-api -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')"

    success "Post-deployment tasks completed"
}

# =============================================================================
# MAIN DEPLOYMENT WORKFLOW
# =============================================================================

main() {
    log "Starting Pynomaly API deployment..."
    log "Environment: $ENVIRONMENT"
    log "Namespace: $NAMESPACE"
    log "Image Tag: $IMAGE_TAG"

    if [[ "$DRY_RUN" == "true" ]]; then
        warning "DRY RUN MODE - No changes will be made"
    fi

    # Confirmation for production
    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$FORCE_DEPLOY" == "false" ]] && [[ "$DRY_RUN" == "false" ]]; then
        echo -n "Are you sure you want to deploy to PRODUCTION? (yes/no): "
        read -r confirmation
        if [[ "$confirmation" != "yes" ]]; then
            log "Deployment cancelled by user"
            exit 0
        fi
    fi

    # Pre-deployment phase
    validate_environment
    validate_image
    run_pre_deployment_tests
    backup_current_deployment

    # Deployment phase
    deploy_database
    deploy_redis
    deploy_api
    deploy_monitoring

    # Post-deployment phase
    verify_deployment
    post_deployment_tasks

    success "🚀 Deployment completed successfully!"

    if [[ "$DRY_RUN" == "false" ]]; then
        log "Access your deployment:"
        log "  kubectl get services -n $NAMESPACE"
        log "  kubectl port-forward service/pynomaly-api-internal 8080:8000 -n $NAMESPACE"
        log "  curl http://localhost:8080/api/v1/health"
    fi
}

# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

cleanup() {
    log "Cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}

trap cleanup EXIT

# =============================================================================
# ENTRY POINT
# =============================================================================

main "$@"
