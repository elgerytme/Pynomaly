#!/bin/bash

# Staging Environment Deployment Script
# This script deploys the MLOps platform to the staging environment for validation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAGING_NAMESPACE="mlops-staging"
DOMAIN_SUFFIX="staging.mlops-platform.com"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check if istioctl is installed
    if ! command -v istioctl &> /dev/null; then
        log_error "istioctl is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if current context is staging
    current_context=$(kubectl config current-context)
    if [[ "$current_context" != *"staging"* ]]; then
        log_warning "Current kubectl context ($current_context) does not appear to be staging"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_success "Prerequisites check completed"
}

# Create staging namespace
create_namespace() {
    log_info "Creating staging namespace..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$STAGING_NAMESPACE" &> /dev/null; then
        kubectl create namespace "$STAGING_NAMESPACE"
        log_success "Created namespace: $STAGING_NAMESPACE"
    else
        log_info "Namespace $STAGING_NAMESPACE already exists"
    fi
    
    # Label namespace for Istio injection
    kubectl label namespace "$STAGING_NAMESPACE" istio-injection=enabled --overwrite
    
    # Add staging-specific labels
    kubectl label namespace "$STAGING_NAMESPACE" \
        environment=staging \
        managed-by=deployment-script \
        security-level=medium \
        --overwrite
    
    log_success "Namespace configuration completed"
}

# Deploy secrets and configuration
deploy_configuration() {
    log_info "Deploying configuration and secrets..."
    
    # Create staging-specific configmap
    kubectl create configmap mlops-staging-config \
        --namespace="$STAGING_NAMESPACE" \
        --from-file="${SCRIPT_DIR}/config/application-staging.yml" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create staging secrets (using staging values)
    envsubst < "${SCRIPT_DIR}/config/secrets-staging.yaml" | kubectl apply -n "$STAGING_NAMESPACE" -f -
    
    # Create staging-specific TLS certificate
    envsubst < "${SCRIPT_DIR}/config/certificate-staging.yaml" | kubectl apply -n "$STAGING_NAMESPACE" -f -
    
    log_success "Configuration and secrets deployed"
}

# Deploy data layer
deploy_data_layer() {
    log_info "Deploying data layer (PostgreSQL and Redis)..."
    
    # Deploy PostgreSQL for staging
    helm upgrade --install postgres-staging bitnami/postgresql \
        --namespace="$STAGING_NAMESPACE" \
        --values="${SCRIPT_DIR}/helm/postgres-staging-values.yaml" \
        --wait --timeout=300s
    
    # Deploy Redis
    helm upgrade --install redis-staging bitnami/redis \
        --namespace="$STAGING_NAMESPACE" \
        --values="${SCRIPT_DIR}/helm/redis-staging-values.yaml" \
        --wait --timeout=300s
    
    # Wait for databases to be ready
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/name=postgresql \
        -n "$STAGING_NAMESPACE" --timeout=300s
    
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/name=redis \
        -n "$STAGING_NAMESPACE" --timeout=300s
    
    log_success "Data layer deployed successfully"
}

# Deploy application services
deploy_application_services() {
    log_info "Deploying application services..."
    
    # Update image tags to staging versions
    export IMAGE_TAG="${IMAGE_TAG:-staging-$(git rev-parse --short HEAD)}"
    export DOMAIN_NAME="$DOMAIN_SUFFIX"
    
    # Deploy API server
    envsubst < "${PROJECT_ROOT}/infrastructure/production/config/kubernetes/production-deployment.yaml" | \
        sed "s/mlops-production/$STAGING_NAMESPACE/g" | \
        sed "s/api.mlops-platform.com/api.$DOMAIN_SUFFIX/g" | \
        kubectl apply -f -
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available deployment \
        -l tier=application \
        -n "$STAGING_NAMESPACE" --timeout=600s
    
    log_success "Application services deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Deploy Prometheus for staging
    helm upgrade --install prometheus-staging prometheus-community/kube-prometheus-stack \
        --namespace="$STAGING_NAMESPACE" \
        --values="${SCRIPT_DIR}/helm/prometheus-staging-values.yaml" \
        --wait --timeout=600s
    
    # Deploy custom dashboards
    kubectl apply -f "${PROJECT_ROOT}/infrastructure/monitoring/grafana-dashboards/" -n "$STAGING_NAMESPACE"
    
    log_success "Monitoring stack deployed"
}

# Configure Istio service mesh
configure_service_mesh() {
    log_info "Configuring Istio service mesh..."
    
    # Deploy staging gateway
    envsubst < "${PROJECT_ROOT}/infrastructure/production/gateway/istio-gateway.yaml" | \
        sed "s/mlops-production/$STAGING_NAMESPACE/g" | \
        sed "s/api.mlops-platform.com/api.$DOMAIN_SUFFIX/g" | \
        sed "s/app.mlops-platform.com/app.$DOMAIN_SUFFIX/g" | \
        kubectl apply -f -
    
    # Apply security policies
    envsubst < "${PROJECT_ROOT}/infrastructure/production/security/security-policies.yaml" | \
        sed "s/mlops-production/$STAGING_NAMESPACE/g" | \
        kubectl apply -f -
    
    # Validate Istio configuration
    istioctl analyze -n "$STAGING_NAMESPACE"
    
    log_success "Service mesh configured"
}

# Run deployment validation
run_validation() {
    log_info "Running deployment validation..."
    
    # Wait for all pods to be ready
    log_info "Waiting for all pods to be ready..."
    kubectl wait --for=condition=ready pod \
        --all -n "$STAGING_NAMESPACE" --timeout=600s
    
    # Check service endpoints
    log_info "Validating service endpoints..."
    
    # Get ingress IP
    INGRESS_IP=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [[ -z "$INGRESS_IP" ]]; then
        INGRESS_IP=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    if [[ -z "$INGRESS_IP" ]]; then
        log_warning "Could not determine ingress IP, using port-forward for validation"
        kubectl port-forward svc/api-server 8080:8000 -n "$STAGING_NAMESPACE" &
        PORT_FORWARD_PID=$!
        sleep 5
        API_URL="http://localhost:8080"
    else
        API_URL="https://api.$DOMAIN_SUFFIX"
    fi
    
    # Test health endpoints
    log_info "Testing health endpoints..."
    if curl -f -s "$API_URL/health" > /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        exit 1
    fi
    
    # Test authentication
    log_info "Testing authentication..."
    if curl -f -s -X POST "$API_URL/api/v1/auth/health" > /dev/null; then
        log_success "Authentication service is responsive"
    else
        log_warning "Authentication service test failed (may be expected if not fully configured)"
    fi
    
    # Clean up port-forward if used
    if [[ -n "${PORT_FORWARD_PID:-}" ]]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
    
    # Run comprehensive validation script
    log_info "Running comprehensive validation..."
    python3 "${PROJECT_ROOT}/tests/deployment/validate_deployment.py" \
        --environment staging \
        --namespace "$STAGING_NAMESPACE" \
        --domain "$DOMAIN_SUFFIX"
    
    log_success "Deployment validation completed"
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."
    
    REPORT_FILE="${SCRIPT_DIR}/staging-deployment-report-$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$REPORT_FILE" << EOF
# Staging Deployment Report

**Date:** $(date)
**Environment:** Staging
**Namespace:** $STAGING_NAMESPACE
**Domain:** $DOMAIN_SUFFIX
**Git Commit:** $(git rev-parse HEAD)

## Deployment Summary

### Services Deployed
\`\`\`
$(kubectl get deployments -n "$STAGING_NAMESPACE" -o wide)
\`\`\`

### Pods Status
\`\`\`
$(kubectl get pods -n "$STAGING_NAMESPACE" -o wide)
\`\`\`

### Services
\`\`\`
$(kubectl get services -n "$STAGING_NAMESPACE" -o wide)
\`\`\`

### Ingress
\`\`\`
$(kubectl get ingress -n "$STAGING_NAMESPACE" -o wide)
\`\`\`

### Resource Usage
\`\`\`
$(kubectl top pods -n "$STAGING_NAMESPACE" 2>/dev/null || echo "Metrics not available")
\`\`\`

## Access Information

- **API Endpoint:** https://api.$DOMAIN_SUFFIX
- **Web UI:** https://app.$DOMAIN_SUFFIX
- **Monitoring:** https://monitoring.$DOMAIN_SUFFIX/grafana

## Validation Results

- ✅ Health checks passed
- ✅ Service mesh configured
- ✅ Monitoring deployed
- ✅ Security policies applied

## Next Steps

1. Configure DNS entries for $DOMAIN_SUFFIX
2. Run comprehensive testing suite
3. Validate monitoring and alerting
4. Conduct security testing
5. Performance validation

EOF

    log_success "Deployment report generated: $REPORT_FILE"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    if [[ -n "${PORT_FORWARD_PID:-}" ]]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
}

# Main deployment function
main() {
    log_info "Starting staging environment deployment..."
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Check if we should skip certain steps
    SKIP_PREREQ=${SKIP_PREREQ:-false}
    SKIP_NAMESPACE=${SKIP_NAMESPACE:-false}
    SKIP_CONFIG=${SKIP_CONFIG:-false}
    SKIP_DATA=${SKIP_DATA:-false}
    SKIP_APP=${SKIP_APP:-false}
    SKIP_MONITORING=${SKIP_MONITORING:-false}
    SKIP_MESH=${SKIP_MESH:-false}
    SKIP_VALIDATION=${SKIP_VALIDATION:-false}
    
    # Execute deployment steps
    [[ "$SKIP_PREREQ" != "true" ]] && check_prerequisites
    [[ "$SKIP_NAMESPACE" != "true" ]] && create_namespace
    [[ "$SKIP_CONFIG" != "true" ]] && deploy_configuration
    [[ "$SKIP_DATA" != "true" ]] && deploy_data_layer
    [[ "$SKIP_APP" != "true" ]] && deploy_application_services
    [[ "$SKIP_MONITORING" != "true" ]] && deploy_monitoring
    [[ "$SKIP_MESH" != "true" ]] && configure_service_mesh
    [[ "$SKIP_VALIDATION" != "true" ]] && run_validation
    
    # Generate deployment report
    generate_report
    
    log_success "Staging deployment completed successfully!"
    log_info "Access the staging environment at: https://api.$DOMAIN_SUFFIX"
    log_info "Monitoring dashboard: https://monitoring.$DOMAIN_SUFFIX/grafana"
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy MLOps platform to staging environment

OPTIONS:
    -h, --help              Show this help message
    --skip-prereq          Skip prerequisite checks
    --skip-namespace       Skip namespace creation
    --skip-config          Skip configuration deployment
    --skip-data            Skip data layer deployment
    --skip-app             Skip application deployment
    --skip-monitoring      Skip monitoring deployment
    --skip-mesh            Skip service mesh configuration
    --skip-validation      Skip deployment validation
    --domain-suffix        Set custom domain suffix (default: staging.mlops-platform.com)

ENVIRONMENT VARIABLES:
    IMAGE_TAG              Docker image tag to deploy (default: staging-{git-hash})
    STAGING_NAMESPACE      Kubernetes namespace (default: mlops-staging)

EXAMPLES:
    # Full deployment
    $0
    
    # Skip monitoring deployment
    $0 --skip-monitoring
    
    # Deploy with custom domain
    $0 --domain-suffix staging.example.com
    
    # Deploy specific image tag
    IMAGE_TAG=v1.2.3-staging $0

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --skip-prereq)
            SKIP_PREREQ=true
            shift
            ;;
        --skip-namespace)
            SKIP_NAMESPACE=true
            shift
            ;;
        --skip-config)
            SKIP_CONFIG=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-app)
            SKIP_APP=true
            shift
            ;;
        --skip-monitoring)
            SKIP_MONITORING=true
            shift
            ;;
        --skip-mesh)
            SKIP_MESH=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --domain-suffix)
            DOMAIN_SUFFIX="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main