#!/bin/bash

# Deploy Pynomaly to Staging Environment
# This script deploys the complete anomaly detection application to the staging environment

set -euo pipefail

# Configuration
NAMESPACE="pynomaly-staging"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
K8S_STAGING_DIR="${PROJECT_ROOT}/k8s/staging"
TIMEOUT=600
VERBOSE=false
DRY_RUN=false

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

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Pynomaly to staging environment

OPTIONS:
    -h, --help           Show this help message
    -v, --verbose        Enable verbose output
    -d, --dry-run        Show what would be deployed without actually deploying
    -t, --timeout SECS   Timeout for deployment operations (default: 600)
    -n, --namespace NAME Override namespace (default: pynomaly-staging)
    --skip-build         Skip Docker image build
    --skip-tests         Skip pre-deployment tests
    --force              Force deployment even if validation fails

EXAMPLES:
    $0                   # Deploy with default settings
    $0 --verbose         # Deploy with verbose output
    $0 --dry-run         # Show what would be deployed
    $0 --timeout 300     # Set custom timeout
    $0 --skip-build      # Skip Docker build step
EOF
}

# Parse command line arguments
SKIP_BUILD=false
SKIP_TESTS=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Verbose logging
verbose_log() {
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "$1"
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi

    # Check Docker (if not skipping build)
    if [[ "$SKIP_BUILD" == "false" ]] && ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check Kubernetes cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check if staging manifests exist
    if [[ ! -d "$K8S_STAGING_DIR" ]]; then
        log_error "Staging manifests directory not found: $K8S_STAGING_DIR"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Build Docker image for staging
build_staging_image() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Skipping Docker build step"
        return 0
    fi

    log_info "Building staging Docker image..."

    cd "$PROJECT_ROOT"

    # Build staging image
    docker build -t pynomaly:staging \
        --build-arg ENV=staging \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
        --build-arg VERSION="$(git describe --tags --always --dirty)" \
        --label "environment=staging" \
        --label "build.date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --label "build.vcs-ref=$(git rev-parse --short HEAD)" \
        --label "build.version=$(git describe --tags --always --dirty)" \
        -f Dockerfile.staging .

    log_success "Docker image built successfully"
}

# Run pre-deployment tests
run_pre_deployment_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_info "Skipping pre-deployment tests"
        return 0
    fi

    log_info "Running pre-deployment tests..."

    cd "$PROJECT_ROOT"

    # Run fast tests only for staging deployment
    if [[ -f "pytest.ini" ]]; then
        verbose_log "Running pytest with staging markers..."
        python -m pytest tests/ -m "fast and not slow" --tb=short --maxfail=5 -q
    else
        log_warning "No pytest.ini found, skipping tests"
    fi

    # Validate Kubernetes manifests
    log_info "Validating Kubernetes manifests..."
    for manifest in "$K8S_STAGING_DIR"/*.yaml; do
        if [[ -f "$manifest" ]]; then
            verbose_log "Validating $manifest"
            kubectl apply --dry-run=client -f "$manifest" > /dev/null
        fi
    done

    log_success "Pre-deployment tests passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"

    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would create namespace: $NAMESPACE"
        else
            kubectl create namespace "$NAMESPACE"
            log_success "Created namespace: $NAMESPACE"
        fi
    fi
}

# Apply Kubernetes manifests
apply_manifests() {
    log_info "Applying Kubernetes manifests..."

    # Order of deployment matters for dependencies
    local manifests=(
        "namespace.yaml"
        "configmap.yaml"
        "secrets.yaml"
        "databases.yaml"
        "pynomaly-staging.yaml"
        "monitoring.yaml"
        "ingress.yaml"
    )

    for manifest in "${manifests[@]}"; do
        local manifest_path="$K8S_STAGING_DIR/$manifest"

        if [[ -f "$manifest_path" ]]; then
            log_info "Applying $manifest"

            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "[DRY RUN] Would apply: $manifest"
                kubectl apply --dry-run=client -f "$manifest_path"
            else
                kubectl apply -f "$manifest_path"
                verbose_log "Applied $manifest successfully"
            fi
        else
            log_warning "Manifest not found: $manifest_path"
        fi
    done

    log_success "Manifests applied successfully"
}

# Wait for deployments to be ready
wait_for_deployments() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would wait for deployments to be ready"
        return 0
    fi

    log_info "Waiting for deployments to be ready..."

    local deployments=(
        "pynomaly-staging-app"
        "postgres-staging"
        "redis-staging"
        "mongodb-staging"
        "prometheus-staging"
        "grafana-staging"
    )

    for deployment in "${deployments[@]}"; do
        log_info "Waiting for $deployment to be ready..."

        if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
            kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${TIMEOUT}s"
            verbose_log "$deployment is ready"
        elif kubectl get statefulset "$deployment" -n "$NAMESPACE" &> /dev/null; then
            kubectl rollout status statefulset/"$deployment" -n "$NAMESPACE" --timeout="${TIMEOUT}s"
            verbose_log "$deployment is ready"
        else
            log_warning "Deployment/StatefulSet $deployment not found"
        fi
    done

    log_success "All deployments are ready"
}

# Run post-deployment health checks
run_health_checks() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run health checks"
        return 0
    fi

    log_info "Running post-deployment health checks..."

    # Check if main application is healthy
    local app_pod=$(kubectl get pods -n "$NAMESPACE" -l app=pynomaly,component=app -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [[ -n "$app_pod" ]]; then
        log_info "Testing application health endpoint..."

        if kubectl exec -n "$NAMESPACE" "$app_pod" -- curl -f http://localhost:8000/health &> /dev/null; then
            log_success "Application health check passed"
        else
            log_error "Application health check failed"
            if [[ "$FORCE" == "false" ]]; then
                exit 1
            fi
        fi
    else
        log_warning "No application pod found for health check"
    fi

    # Check database connectivity
    local postgres_pod=$(kubectl get pods -n "$NAMESPACE" -l app=pynomaly,component=postgres -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [[ -n "$postgres_pod" ]]; then
        log_info "Testing PostgreSQL connectivity..."

        if kubectl exec -n "$NAMESPACE" "$postgres_pod" -- pg_isready -U postgres &> /dev/null; then
            log_success "PostgreSQL health check passed"
        else
            log_error "PostgreSQL health check failed"
            if [[ "$FORCE" == "false" ]]; then
                exit 1
            fi
        fi
    fi

    # Check Redis connectivity
    local redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=pynomaly,component=redis -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [[ -n "$redis_pod" ]]; then
        log_info "Testing Redis connectivity..."

        if kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli ping &> /dev/null; then
            log_success "Redis health check passed"
        else
            log_error "Redis health check failed"
            if [[ "$FORCE" == "false" ]]; then
                exit 1
            fi
        fi
    fi

    log_success "Health checks completed"
}

# Display deployment status
show_deployment_status() {
    log_info "Deployment Status:"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would show deployment status"
        return 0
    fi

    echo
    echo "Namespace: $NAMESPACE"
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide

    echo
    echo "Services:"
    kubectl get services -n "$NAMESPACE" -o wide

    echo
    echo "Ingress:"
    kubectl get ingress -n "$NAMESPACE" -o wide

    echo
    echo "Persistent Volumes:"
    kubectl get pvc -n "$NAMESPACE" -o wide

    echo
    echo "Recent Events:"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

# Display access information
show_access_info() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would show access information"
        return 0
    fi

    log_info "Access Information:"

    # Get ingress external IP
    local ingress_ip=$(kubectl get ingress pynomaly-staging-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

    echo
    echo "Application URLs (add to /etc/hosts if using local cluster):"
    echo "  Main Application: https://staging.pynomaly.com"
    echo "  API Endpoint: https://api-staging.pynomaly.com"
    echo "  Grafana Dashboard: https://grafana-staging.pynomaly.com"
    echo "  Prometheus Metrics: https://prometheus-staging.pynomaly.com"
    echo
    echo "Ingress IP: $ingress_ip"
    echo
    echo "Port Forward Commands (for local development):"
    echo "  kubectl port-forward -n $NAMESPACE svc/pynomaly-staging-service 8000:8000"
    echo "  kubectl port-forward -n $NAMESPACE svc/grafana-staging-service 3000:3000"
    echo "  kubectl port-forward -n $NAMESPACE svc/prometheus-staging-service 9090:9090"
    echo
    echo "Useful Commands:"
    echo "  kubectl get pods -n $NAMESPACE"
    echo "  kubectl logs -n $NAMESPACE -l app=pynomaly,component=app -f"
    echo "  kubectl exec -n $NAMESPACE -it <pod-name> -- /bin/bash"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"

        if [[ "$DRY_RUN" == "false" ]]; then
            log_info "Recent logs from failed deployment:"
            kubectl logs -n "$NAMESPACE" -l app=pynomaly,component=app --tail=50 2>/dev/null || true
        fi
    fi
    exit $exit_code
}

# Main deployment function
main() {
    log_info "Starting Pynomaly staging deployment..."

    # Set up error handling
    trap cleanup EXIT

    # Run deployment steps
    check_prerequisites
    build_staging_image
    run_pre_deployment_tests
    create_namespace
    apply_manifests
    wait_for_deployments
    run_health_checks
    show_deployment_status
    show_access_info

    log_success "Staging deployment completed successfully!"

    if [[ "$DRY_RUN" == "false" ]]; then
        log_info "Next steps:"
        echo "1. Run comprehensive load testing"
        echo "2. Execute security testing"
        echo "3. Validate all application features"
        echo "4. Monitor metrics and logs"
        echo "5. Proceed with production deployment when ready"
    fi
}

# Run main function
main "$@"
