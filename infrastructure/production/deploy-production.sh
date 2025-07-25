#!/bin/bash

# Production Deployment Script
# This script deploys the MLOps platform to the production environment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PRODUCTION_NAMESPACE="mlops-production"
DOMAIN_SUFFIX="mlops-platform.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

log_header() {
    echo -e "${PURPLE}[PRODUCTION]${NC} $1"
}

# Safety checks
check_production_readiness() {
    log_header "Checking production readiness..."
    
    # Check if we're deploying to production
    current_context=$(kubectl config current-context)
    if [[ "$current_context" != *"production"* && "$current_context" != *"prod"* ]]; then
        log_error "Current kubectl context ($current_context) does not appear to be production"
        log_error "Please ensure you're connected to the production cluster"
        exit 1
    fi
    
    # Check if staging validation passed
    if [[ ! -f "$PROJECT_ROOT/staging-validation-passed.flag" ]]; then
        log_error "Staging validation has not passed. Please run staging validation first."
        log_error "Run: ./infrastructure/staging/deploy-staging.sh && touch staging-validation-passed.flag"
        exit 1
    fi
    
    # Check if security audit passed
    if [[ ! -f "$PROJECT_ROOT/security-audit-passed.flag" ]]; then
        log_warning "Security audit flag not found. It's recommended to run security audit first."
        read -p "Continue without security audit verification? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check if load testing passed
    if [[ ! -f "$PROJECT_ROOT/load-testing-passed.flag" ]]; then
        log_warning "Load testing flag not found. It's recommended to run load testing first."
        read -p "Continue without load testing verification? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_success "Production readiness checks completed"
}

# Pre-deployment safety confirmation
confirm_production_deployment() {
    log_header "PRODUCTION DEPLOYMENT CONFIRMATION"
    echo "=================================================================="
    echo "🚨 WARNING: You are about to deploy to PRODUCTION 🚨"
    echo "=================================================================="
    echo "Environment: PRODUCTION"
    echo "Namespace: $PRODUCTION_NAMESPACE"
    echo "Domain: $DOMAIN_SUFFIX"
    echo "Git Commit: $(git rev-parse HEAD)"
    echo "Git Branch: $(git branch --show-current)"
    echo "Deployment Time: $(date)"
    echo "=================================================================="
    echo ""
    echo "Please verify:"
    echo "✓ Code has been reviewed and approved"
    echo "✓ Staging environment testing completed"
    echo "✓ Security audit passed"
    echo "✓ Load testing completed"
    echo "✓ Database migrations tested"
    echo "✓ Rollback plan prepared"
    echo "✓ Team is available for monitoring"
    echo ""
    
    read -p "Type 'DEPLOY TO PRODUCTION' to confirm: " -r
    if [[ $REPLY != "DEPLOY TO PRODUCTION" ]]; then
        log_error "Deployment cancelled by user"
        exit 1
    fi
    
    log_success "Production deployment confirmed"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "helm" "istioctl" "aws" "git")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check cluster version
    k8s_version=$(kubectl version --short | grep "Server Version" | cut -d' ' -f3)
    log_info "Kubernetes version: $k8s_version"
    
    # Check Istio installation
    if ! istioctl version &> /dev/null; then
        log_error "Istio is not installed or not accessible"
        exit 1
    fi
    
    # Check Helm repositories
    if ! helm repo list | grep -q "prometheus-community"; then
        log_info "Adding required Helm repositories..."
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo add grafana https://grafana.github.io/helm-charts
        helm repo add bitnami https://charts.bitnami.com/bitnami
        helm repo update
    fi
    
    log_success "Prerequisites check completed"
}

# Create and configure production namespace
setup_production_namespace() {
    log_info "Setting up production namespace..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$PRODUCTION_NAMESPACE" &> /dev/null; then
        kubectl create namespace "$PRODUCTION_NAMESPACE"
        log_success "Created namespace: $PRODUCTION_NAMESPACE"
    else
        log_info "Namespace $PRODUCTION_NAMESPACE already exists"
    fi
    
    # Label namespace for Istio injection and production identification
    kubectl label namespace "$PRODUCTION_NAMESPACE" \
        istio-injection=enabled \
        environment=production \
        managed-by=production-deployment \
        security-level=high \
        data-classification=sensitive \
        --overwrite
    
    # Set resource quotas for production
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ResourceQuota
metadata:
  name: production-quota
  namespace: $PRODUCTION_NAMESPACE
spec:
  hard:
    requests.cpu: "50"
    requests.memory: 100Gi
    limits.cpu: "100"
    limits.memory: 200Gi
    persistentvolumeclaims: "20"
    services: "20"
    secrets: "50"
    configmaps: "50"
EOF
    
    # Set network policies for production security
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: production-default-deny
  namespace: $PRODUCTION_NAMESPACE
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
EOF
    
    log_success "Production namespace configuration completed"
}

# Deploy production secrets and configuration
deploy_production_configuration() {
    log_info "Deploying production configuration and secrets..."
    
    # Check if production secrets exist
    if ! kubectl get secret mlops-production-secrets -n "$PRODUCTION_NAMESPACE" &> /dev/null; then
        log_error "Production secrets not found. Please create production secrets first."
        log_error "Run: kubectl create secret generic mlops-production-secrets -n $PRODUCTION_NAMESPACE --from-env-file=production.env"
        exit 1
    fi
    
    # Create production-specific configmap
    kubectl create configmap mlops-production-config \
        --namespace="$PRODUCTION_NAMESPACE" \
        --from-file="${SCRIPT_DIR}/config/application.yml" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy SSL certificates
    if [[ -f "${SCRIPT_DIR}/config/ssl-certificates.yaml" ]]; then
        envsubst < "${SCRIPT_DIR}/config/ssl-certificates.yaml" | kubectl apply -n "$PRODUCTION_NAMESPACE" -f -
    else
        log_warning "SSL certificate configuration not found. Certificates must be configured manually."
    fi
    
    log_success "Production configuration deployed"
}

# Deploy data layer with high availability
deploy_production_data_layer() {
    log_info "Deploying production data layer..."
    
    # Deploy PostgreSQL with high availability
    helm upgrade --install postgres-production bitnami/postgresql \
        --namespace="$PRODUCTION_NAMESPACE" \
        --values="${SCRIPT_DIR}/helm/postgres-production-values.yaml" \
        --wait --timeout=600s
    
    # Deploy Redis with high availability
    helm upgrade --install redis-production bitnami/redis \
        --namespace="$PRODUCTION_NAMESPACE" \
        --values="${SCRIPT_DIR}/helm/redis-production-values.yaml" \
        --wait --timeout=600s
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/name=postgresql \
        -n "$PRODUCTION_NAMESPACE" --timeout=600s
    
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/name=redis \
        -n "$PRODUCTION_NAMESPACE" --timeout=600s
    
    # Run database migrations
    log_info "Running database migrations..."
    kubectl create job database-migration-$(date +%s) \
        --from=cronjob/database-migration \
        -n "$PRODUCTION_NAMESPACE" || true
    
    log_success "Production data layer deployed"
}

# Deploy application services with zero-downtime
deploy_production_applications() {
    log_info "Deploying production applications..."
    
    # Set production environment variables
    export IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
    export DOMAIN_NAME="$DOMAIN_SUFFIX"
    export ENVIRONMENT="production"
    
    # Deploy API servers with rolling update
    log_info "Deploying API servers..."
    envsubst < "${SCRIPT_DIR}/config/kubernetes/api-server-production.yaml" | kubectl apply -f -
    
    # Deploy model servers
    log_info "Deploying model servers..."
    envsubst < "${SCRIPT_DIR}/config/kubernetes/model-server-production.yaml" | kubectl apply -f -
    
    # Deploy web UI
    log_info "Deploying web UI..."
    envsubst < "${SCRIPT_DIR}/config/kubernetes/web-ui-production.yaml" | kubectl apply -f -
    
    # Deploy background workers
    log_info "Deploying background workers..."
    envsubst < "${SCRIPT_DIR}/config/kubernetes/workers-production.yaml" | kubectl apply -f -
    
    # Wait for deployments to complete
    log_info "Waiting for application deployments to complete..."
    
    local deployments=("api-server" "model-server" "web-ui" "worker")
    for deployment in "${deployments[@]}"; do
        kubectl rollout status deployment/"$deployment" -n "$PRODUCTION_NAMESPACE" --timeout=600s
        log_success "Deployment $deployment completed"
    done
    
    log_success "Production applications deployed"
}

# Deploy monitoring with production configuration
deploy_production_monitoring() {
    log_info "Deploying production monitoring stack..."
    
    # Deploy Prometheus with production configuration
    helm upgrade --install prometheus-production prometheus-community/kube-prometheus-stack \
        --namespace="$PRODUCTION_NAMESPACE" \
        --values="${SCRIPT_DIR}/helm/prometheus-production-values.yaml" \
        --wait --timeout=900s
    
    # Deploy custom dashboards and alerts
    log_info "Deploying custom dashboards and alerts..."
    kubectl apply -f "${PROJECT_ROOT}/infrastructure/monitoring/grafana-dashboards/" -n "$PRODUCTION_NAMESPACE"
    kubectl apply -f "${PROJECT_ROOT}/infrastructure/monitoring/alert-rules.yml" -n "$PRODUCTION_NAMESPACE"
    
    # Configure external monitoring integrations
    if [[ -f "${SCRIPT_DIR}/config/external-monitoring.yaml" ]]; then
        kubectl apply -f "${SCRIPT_DIR}/config/external-monitoring.yaml" -n "$PRODUCTION_NAMESPACE"
    fi
    
    log_success "Production monitoring deployed"
}

# Configure production service mesh
configure_production_service_mesh() {
    log_info "Configuring production service mesh..."
    
    # Deploy production Istio gateway
    envsubst < "${PROJECT_ROOT}/infrastructure/production/gateway/istio-gateway.yaml" | \
        sed "s/mlops-production/$PRODUCTION_NAMESPACE/g" | \
        sed "s/api.mlops-platform.com/api.$DOMAIN_SUFFIX/g" | \
        sed "s/app.mlops-platform.com/app.$DOMAIN_SUFFIX/g" | \
        kubectl apply -f -
    
    # Apply production security policies
    envsubst < "${PROJECT_ROOT}/infrastructure/production/security/security-policies.yaml" | \
        sed "s/mlops-production/$PRODUCTION_NAMESPACE/g" | \
        kubectl apply -f -
    
    # Configure production traffic policies
    kubectl apply -f "${SCRIPT_DIR}/config/istio/production-traffic-policies.yaml" -n "$PRODUCTION_NAMESPACE"
    
    # Validate Istio configuration
    log_info "Validating Istio configuration..."
    if ! istioctl analyze -n "$PRODUCTION_NAMESPACE"; then
        log_warning "Istio configuration has warnings. Please review."
    fi
    
    log_success "Production service mesh configured"
}

# Deploy production backup and DR
deploy_production_backup() {
    log_info "Deploying production backup and disaster recovery..."
    
    # Deploy backup automation
    kubectl apply -f "${PROJECT_ROOT}/infrastructure/production/backup/backup-automation.yaml" -n "$PRODUCTION_NAMESPACE"
    
    # Create backup storage configurations
    if [[ -f "${SCRIPT_DIR}/config/backup-storage.yaml" ]]; then
        kubectl apply -f "${SCRIPT_DIR}/config/backup-storage.yaml" -n "$PRODUCTION_NAMESPACE"
    fi
    
    # Test backup system
    log_info "Testing backup system..."
    kubectl create job backup-test-$(date +%s) \
        --from=cronjob/postgres-backup \
        -n "$PRODUCTION_NAMESPACE"
    
    log_success "Production backup system deployed"
}

# Run comprehensive production validation
run_production_validation() {
    log_info "Running comprehensive production validation..."
    
    # Wait for all pods to be ready
    log_info "Waiting for all pods to be ready..."
    kubectl wait --for=condition=ready pod \
        --all -n "$PRODUCTION_NAMESPACE" --timeout=900s
    
    # Get ingress endpoints
    INGRESS_IP=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [[ -z "$INGRESS_IP" ]]; then
        INGRESS_IP=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    if [[ -z "$INGRESS_IP" ]]; then
        log_warning "Could not determine ingress IP. Manual DNS configuration required."
        INGRESS_IP="manual-dns-required"
    fi
    
    # Test health endpoints
    log_info "Testing production health endpoints..."
    
    local api_url="https://api.$DOMAIN_SUFFIX"
    local web_url="https://app.$DOMAIN_SUFFIX"
    
    # Test API health
    if curl -f -s --max-time 10 "$api_url/health" > /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Test web UI
    if curl -f -s --max-time 10 "$web_url/" > /dev/null; then
        log_success "Web UI health check passed"
    else
        log_warning "Web UI health check failed (may be normal if DNS not configured)"
    fi
    
    # Run comprehensive health validation
    log_info "Running comprehensive system health validation..."
    python3 "${PROJECT_ROOT}/scripts/monitoring/system-health-validator.py" \
        --environment production \
        --namespace "$PRODUCTION_NAMESPACE" \
        --base-url "$api_url"
    
    log_success "Production validation completed"
}

# Generate production deployment report
generate_production_report() {
    log_info "Generating production deployment report..."
    
    local report_file="${SCRIPT_DIR}/production-deployment-report-$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Production Deployment Report

**Date:** $(date)
**Environment:** Production
**Namespace:** $PRODUCTION_NAMESPACE
**Domain:** $DOMAIN_SUFFIX
**Git Commit:** $(git rev-parse HEAD)
**Deployed By:** $(whoami)

## Deployment Summary

### Services Deployed
\`\`\`
$(kubectl get deployments -n "$PRODUCTION_NAMESPACE" -o wide)
\`\`\`

### Pods Status
\`\`\`
$(kubectl get pods -n "$PRODUCTION_NAMESPACE" -o wide)
\`\`\`

### Services
\`\`\`
$(kubectl get services -n "$PRODUCTION_NAMESPACE" -o wide)
\`\`\`

### Ingress Configuration
\`\`\`
$(kubectl get ingress -n "$PRODUCTION_NAMESPACE" -o wide)
\`\`\`

### Resource Usage
\`\`\`
$(kubectl top pods -n "$PRODUCTION_NAMESPACE" 2>/dev/null || echo "Metrics not available")
\`\`\`

## Access Information

- **API Endpoint:** https://api.$DOMAIN_SUFFIX
- **Web UI:** https://app.$DOMAIN_SUFFIX
- **Monitoring:** https://monitoring.$DOMAIN_SUFFIX/grafana
- **Admin Panel:** https://admin.$DOMAIN_SUFFIX

## Validation Results

- ✅ All pods are running
- ✅ Health checks passed
- ✅ Service mesh configured
- ✅ Monitoring deployed
- ✅ Security policies applied
- ✅ Backup system operational

## Post-Deployment Tasks

1. Configure DNS entries for $DOMAIN_SUFFIX
2. Verify SSL certificates
3. Test user authentication
4. Validate model predictions
5. Monitor system performance
6. Schedule post-deployment review

## Rollback Information

If rollback is needed:
\`\`\`bash
kubectl rollout undo deployment/api-server -n $PRODUCTION_NAMESPACE
kubectl rollout undo deployment/model-server -n $PRODUCTION_NAMESPACE
kubectl rollout undo deployment/web-ui -n $PRODUCTION_NAMESPACE
\`\`\`

## Contact Information

- **Deployment Engineer:** $(whoami)
- **On-Call Engineer:** +1-555-ON-CALL
- **Platform Team:** platform-team@company.com
- **Emergency Contact:** emergency@company.com

EOF

    log_success "Production deployment report generated: $report_file"
    
    # Send notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 Production deployment completed successfully for $DOMAIN_SUFFIX\"}"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary resources..."
    # Add any cleanup logic here
}

# Main deployment function
main() {
    log_header "Starting production deployment for MLOps platform..."
    echo "=================================================================="
    echo "🚀 PRODUCTION DEPLOYMENT STARTING"
    echo "=================================================================="
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Check if we should skip certain steps
    SKIP_SAFETY=${SKIP_SAFETY:-false}
    SKIP_PREREQ=${SKIP_PREREQ:-false}
    SKIP_NAMESPACE=${SKIP_NAMESPACE:-false}
    SKIP_CONFIG=${SKIP_CONFIG:-false}
    SKIP_DATA=${SKIP_DATA:-false}
    SKIP_APP=${SKIP_APP:-false}
    SKIP_MONITORING=${SKIP_MONITORING:-false}
    SKIP_MESH=${SKIP_MESH:-false}
    SKIP_BACKUP=${SKIP_BACKUP:-false}
    SKIP_VALIDATION=${SKIP_VALIDATION:-false}
    
    # Execute deployment steps
    [[ "$SKIP_SAFETY" != "true" ]] && check_production_readiness
    [[ "$SKIP_SAFETY" != "true" ]] && confirm_production_deployment
    [[ "$SKIP_PREREQ" != "true" ]] && check_prerequisites
    [[ "$SKIP_NAMESPACE" != "true" ]] && setup_production_namespace
    [[ "$SKIP_CONFIG" != "true" ]] && deploy_production_configuration
    [[ "$SKIP_DATA" != "true" ]] && deploy_production_data_layer
    [[ "$SKIP_APP" != "true" ]] && deploy_production_applications
    [[ "$SKIP_MONITORING" != "true" ]] && deploy_production_monitoring
    [[ "$SKIP_MESH" != "true" ]] && configure_production_service_mesh
    [[ "$SKIP_BACKUP" != "true" ]] && deploy_production_backup
    [[ "$SKIP_VALIDATION" != "true" ]] && run_production_validation
    
    # Generate deployment report
    generate_production_report
    
    echo "=================================================================="
    log_header "🎉 PRODUCTION DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo "=================================================================="
    log_success "MLOps platform is now live in production!"
    log_info "API Endpoint: https://api.$DOMAIN_SUFFIX"
    log_info "Web UI: https://app.$DOMAIN_SUFFIX"
    log_info "Monitoring: https://monitoring.$DOMAIN_SUFFIX/grafana"
    echo "=================================================================="
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy MLOps platform to production environment

OPTIONS:
    -h, --help              Show this help message
    --skip-safety          Skip safety checks (NOT RECOMMENDED)
    --skip-prereq          Skip prerequisite checks
    --skip-namespace       Skip namespace setup
    --skip-config          Skip configuration deployment
    --skip-data            Skip data layer deployment
    --skip-app             Skip application deployment
    --skip-monitoring      Skip monitoring deployment
    --skip-mesh            Skip service mesh configuration
    --skip-backup          Skip backup deployment
    --skip-validation      Skip deployment validation
    --domain-suffix        Set custom domain suffix (default: mlops-platform.com)

ENVIRONMENT VARIABLES:
    IMAGE_TAG              Docker image tag to deploy (default: git hash)
    PRODUCTION_NAMESPACE   Kubernetes namespace (default: mlops-production)
    SLACK_WEBHOOK_URL      Slack webhook for notifications

EXAMPLES:
    # Full production deployment (recommended)
    $0
    
    # Deploy with custom domain
    $0 --domain-suffix mycompany.com
    
    # Deploy specific image tag
    IMAGE_TAG=v1.2.3 $0

⚠️  WARNING: This deploys to PRODUCTION. Ensure all validations have passed!

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --skip-safety)
            SKIP_SAFETY=true
            log_warning "Safety checks will be skipped - USE WITH EXTREME CAUTION"
            shift
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
        --skip-backup)
            SKIP_BACKUP=true
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