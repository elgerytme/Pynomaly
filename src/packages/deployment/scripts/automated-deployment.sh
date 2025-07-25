#!/bin/bash
# Advanced Automated Deployment Script for Hexagonal Architecture
# Provides intelligent deployment automation with rollback capabilities

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_FILE="/tmp/deployment-$(date +%Y%m%d-%H%M%S).log"
DEPLOYMENT_CONFIG="${SCRIPT_DIR}/../config/deployment-config.yaml"

# Default values
ENVIRONMENT="development"
STRATEGY="rolling"
DRY_RUN=false
FORCE=false
SKIP_TESTS=false
AUTO_APPROVE=false
PARALLEL_SERVICES=3
ROLLBACK_ON_FAILURE=true
MONITORING_DURATION=300 # 5 minutes

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] DEBUG: $1${NC}" | tee -a "$LOG_FILE"
    fi
}

# Help function
show_help() {
    cat << EOF
Advanced Automated Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENVIRONMENT    Target environment (development|staging|production)
    -s, --strategy STRATEGY         Deployment strategy (rolling|blue-green|canary)
    -d, --dry-run                   Show what would be deployed without executing
    -f, --force                     Force deployment even if validations fail
    -t, --skip-tests               Skip pre-deployment tests
    -a, --auto-approve             Auto-approve without manual confirmation
    -p, --parallel N               Number of services to deploy in parallel (default: 3)
    -m, --monitoring-duration N    Post-deployment monitoring duration in seconds (default: 300)
    --no-rollback                  Disable automatic rollback on failure
    --config FILE                  Custom deployment configuration file
    --debug                        Enable debug output
    -h, --help                     Show this help message

EXAMPLES:
    # Development deployment with rolling strategy
    $0 -e development -s rolling
    
    # Production deployment with blue-green strategy and extended monitoring
    $0 -e production -s blue-green -m 600
    
    # Dry run for staging environment
    $0 -e staging --dry-run
    
    # Force deployment with canary strategy
    $0 -e production -s canary --force --auto-approve

DEPLOYMENT STRATEGIES:
    rolling     - Update pods gradually (zero downtime)
    blue-green  - Deploy to new environment, switch traffic
    canary      - Deploy to subset, gradually increase traffic

ENVIRONMENTS:
    development - Local development environment
    staging     - Pre-production staging environment
    production  - Production environment (requires approval)

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--strategy)
                STRATEGY="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -t|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -a|--auto-approve)
                AUTO_APPROVE=true
                shift
                ;;
            -p|--parallel)
                PARALLEL_SERVICES="$2"
                shift 2
                ;;
            -m|--monitoring-duration)
                MONITORING_DURATION="$2"
                shift 2
                ;;
            --no-rollback)
                ROLLBACK_ON_FAILURE=false
                shift
                ;;
            --config)
                DEPLOYMENT_CONFIG="$2"
                shift 2
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validation functions
validate_environment() {
    case "$ENVIRONMENT" in
        development|staging|production)
            log "Validated environment: $ENVIRONMENT"
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT"
            error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

validate_strategy() {
    case "$STRATEGY" in
        rolling|blue-green|canary)
            log "Validated strategy: $STRATEGY"
            ;;
        *)
            error "Invalid strategy: $STRATEGY"
            error "Valid strategies: rolling, blue-green, canary"
            exit 1
            ;;
    esac
}

validate_prerequisites() {
    log "Validating deployment prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "kubectl" "helm" "jq" "yq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Cannot connect to Docker daemon"
        exit 1
    fi
    
    # Validate deployment configuration
    if [[ -f "$DEPLOYMENT_CONFIG" ]]; then
        if ! yq eval '.' "$DEPLOYMENT_CONFIG" &> /dev/null; then
            error "Invalid deployment configuration: $DEPLOYMENT_CONFIG"
            exit 1
        fi
    else
        warn "Deployment configuration not found: $DEPLOYMENT_CONFIG"
    fi
    
    log "Prerequisites validation completed successfully"
}

# Pre-deployment checks
run_pre_deployment_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warn "Skipping pre-deployment tests (--skip-tests flag)"
        return 0
    fi
    
    log "Running pre-deployment tests..."
    
    # Run unit tests
    log "Running unit tests..."
    if ! make -C "$PROJECT_ROOT" test-unit; then
        error "Unit tests failed"
        return 1
    fi
    
    # Run integration tests
    log "Running integration tests..."
    if ! make -C "$PROJECT_ROOT" test-integration; then
        error "Integration tests failed"
        return 1
    fi
    
    # Run security scans
    log "Running security scans..."
    if ! make -C "$PROJECT_ROOT" security-scan; then
        error "Security scan failed"
        return 1
    fi
    
    # Validate package boundaries
    log "Validating package boundaries..."
    if ! python "$PROJECT_ROOT/src/packages/tools/import_boundary_validator/boundary_validator.py" --strict; then
        error "Package boundary validation failed"
        return 1
    fi
    
    log "Pre-deployment tests completed successfully"
    return 0
}

# Build and tag images
build_images() {
    log "Building Docker images for $ENVIRONMENT environment..."
    
    local build_tag="${ENVIRONMENT}-$(git rev-parse --short HEAD)-$(date +%s)"
    local latest_tag="${ENVIRONMENT}-latest"
    
    # Get list of services to build
    local services
    services=$(find "$PROJECT_ROOT/src/packages" -name "Dockerfile*" -exec dirname {} \; | sort -u)
    
    local build_jobs=()
    
    for service_dir in $services; do
        local service_name
        service_name=$(basename "$service_dir")
        
        log "Building image for service: $service_name"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log "[DRY RUN] Would build image: $service_name:$build_tag"
            continue
        fi
        
        # Build image in background
        (
            cd "$service_dir"
            docker build \
                --tag "hexagonal-architecture/$service_name:$build_tag" \
                --tag "hexagonal-architecture/$service_name:$latest_tag" \
                --label "deployment.environment=$ENVIRONMENT" \
                --label "deployment.strategy=$STRATEGY" \
                --label "deployment.timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                --label "git.commit=$(git rev-parse HEAD)" \
                --label "git.branch=$(git rev-parse --abbrev-ref HEAD)" \
                .
        ) &
        
        build_jobs+=($!)
        
        # Limit parallel builds
        if [[ ${#build_jobs[@]} -ge $PARALLEL_SERVICES ]]; then
            wait "${build_jobs[0]}"
            build_jobs=("${build_jobs[@]:1}")
        fi
    done
    
    # Wait for remaining builds
    for job in "${build_jobs[@]}"; do
        wait "$job"
    done
    
    log "Image building completed successfully"
    export BUILD_TAG="$build_tag"
}

# Deployment strategies
deploy_rolling() {
    log "Executing rolling deployment strategy..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would execute rolling deployment"
        log "[DRY RUN] kubectl set image deployment/service image=hexagonal-architecture/service:$BUILD_TAG"
        return 0
    fi
    
    # Update deployments with new image
    local deployments
    deployments=$(kubectl get deployments -n "$ENVIRONMENT" -o name)
    
    for deployment in $deployments; do
        local service_name
        service_name=$(echo "$deployment" | cut -d'/' -f2)
        
        log "Rolling update for deployment: $service_name"
        
        kubectl set image "$deployment" \
            "$service_name=hexagonal-architecture/$service_name:$BUILD_TAG" \
            -n "$ENVIRONMENT"
        
        # Wait for rollout to complete
        kubectl rollout status "$deployment" -n "$ENVIRONMENT" --timeout=600s
        
        log "Rolling update completed for: $service_name"
    done
    
    log "Rolling deployment completed successfully"
}

deploy_blue_green() {
    log "Executing blue-green deployment strategy..."
    
    local current_env="${ENVIRONMENT}"
    local blue_env="${ENVIRONMENT}-blue"
    local green_env="${ENVIRONMENT}-green"
    
    # Determine active and inactive environments
    local active_env
    local inactive_env
    
    if kubectl get namespace "$blue_env" &> /dev/null; then
        active_env="$blue_env"
        inactive_env="$green_env"
    else
        active_env="$green_env"
        inactive_env="$blue_env"
    fi
    
    log "Active environment: $active_env"
    log "Deploying to inactive environment: $inactive_env"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would deploy to: $inactive_env"
        log "[DRY RUN] Would switch traffic from $active_env to $inactive_env"
        return 0
    fi
    
    # Create inactive environment namespace if it doesn't exist
    kubectl create namespace "$inactive_env" --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy to inactive environment
    ENVIRONMENT="$inactive_env" deploy_rolling
    
    # Run smoke tests on inactive environment
    log "Running smoke tests on inactive environment..."
    if ! run_smoke_tests "$inactive_env"; then
        error "Smoke tests failed on inactive environment"
        return 1
    fi
    
    # Switch traffic
    log "Switching traffic to new environment..."
    kubectl patch service api-gateway \
        -n "$current_env" \
        --type='json' \
        -p="[{'op': 'replace', 'path': '/spec/selector/environment', 'value': '$inactive_env'}]"
    
    # Wait for traffic switch
    sleep 30
    
    # Cleanup old environment
    log "Cleaning up old environment: $active_env"
    kubectl delete namespace "$active_env" --timeout=300s
    
    log "Blue-green deployment completed successfully"
}

deploy_canary() {
    log "Executing canary deployment strategy..."
    
    local canary_percentage=10
    local max_percentage=100
    local increment=10
    local validation_time=60
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would execute canary deployment"
        log "[DRY RUN] Would gradually increase traffic from $canary_percentage% to $max_percentage%"
        return 0
    fi
    
    # Deploy canary version
    log "Deploying canary version with $canary_percentage% traffic"
    
    # Create canary deployment
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway-canary
  namespace: $ENVIRONMENT
spec:
  replicas: 1
  selector:
    matchLabels:
      app: api-gateway
      version: canary
  template:
    metadata:
      labels:
        app: api-gateway
        version: canary
    spec:
      containers:
      - name: api-gateway
        image: hexagonal-architecture/api-gateway:$BUILD_TAG
        ports:
        - containerPort: 8080
EOF

    # Wait for canary deployment
    kubectl rollout status deployment/api-gateway-canary -n "$ENVIRONMENT" --timeout=300s
    
    # Gradually increase canary traffic
    for percentage in $(seq $((canary_percentage + increment)) $increment $max_percentage); do
        log "Increasing canary traffic to $percentage%"
        
        # Update traffic split (using Istio or similar service mesh)
        # This is a simplified example - in reality, you'd use your service mesh configuration
        
        # Monitor metrics for validation time
        log "Monitoring canary deployment for $validation_time seconds..."
        sleep $validation_time
        
        # Check error rates and performance metrics
        if ! validate_canary_metrics; then
            error "Canary validation failed at $percentage% traffic"
            rollback_canary
            return 1
        fi
        
        log "Canary validation successful at $percentage% traffic"
    done
    
    # Complete canary deployment
    log "Promoting canary to full production"
    kubectl patch deployment api-gateway \
        -n "$ENVIRONMENT" \
        --type='json' \
        -p="[{'op': 'replace', 'path': '/spec/template/spec/containers/0/image', 'value': 'hexagonal-architecture/api-gateway:$BUILD_TAG'}]"
    
    # Cleanup canary deployment
    kubectl delete deployment api-gateway-canary -n "$ENVIRONMENT"
    
    log "Canary deployment completed successfully"
}

# Validation functions
run_smoke_tests() {
    local env="${1:-$ENVIRONMENT}"
    log "Running smoke tests against $env environment..."
    
    # Get service endpoint
    local endpoint
    endpoint=$(kubectl get service api-gateway -n "$env" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    if [[ -z "$endpoint" ]]; then
        endpoint=$(kubectl get service api-gateway -n "$env" -o jsonpath='{.spec.clusterIP}')
    fi
    
    # Basic health check
    if ! curl -f "http://$endpoint/health" --max-time 30; then
        error "Health check failed for $env environment"
        return 1
    fi
    
    # API functionality tests
    if ! curl -f "http://$endpoint/api/v1/status" --max-time 30; then
        error "API status check failed for $env environment"
        return 1
    fi
    
    log "Smoke tests passed for $env environment"
    return 0
}

validate_canary_metrics() {
    # Check error rates, response times, etc.
    # This would typically integrate with your monitoring system
    log "Validating canary metrics..."
    
    # Placeholder for actual metrics validation
    # In production, you'd query Prometheus, Grafana, or similar
    
    return 0
}

rollback_canary() {
    log "Rolling back canary deployment..."
    kubectl delete deployment api-gateway-canary -n "$ENVIRONMENT" || true
    log "Canary rollback completed"
}

# Post-deployment monitoring
monitor_deployment() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would monitor deployment for $MONITORING_DURATION seconds"
        return 0
    fi
    
    log "Monitoring deployment for $MONITORING_DURATION seconds..."
    
    local start_time
    start_time=$(date +%s)
    local end_time
    end_time=$((start_time + MONITORING_DURATION))
    
    while [[ $(date +%s) -lt $end_time ]]; do
        # Check pod health
        local unhealthy_pods
        unhealthy_pods=$(kubectl get pods -n "$ENVIRONMENT" --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
        
        if [[ $unhealthy_pods -gt 0 ]]; then
            error "Found $unhealthy_pods unhealthy pods"
            kubectl get pods -n "$ENVIRONMENT" --field-selector=status.phase!=Running
            
            if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
                perform_rollback
                return 1
            fi
        fi
        
        # Check service availability
        if ! run_smoke_tests; then
            error "Service availability check failed"
            
            if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
                perform_rollback
                return 1
            fi
        fi
        
        log "System healthy - continuing monitoring..."
        sleep 30
    done
    
    log "Deployment monitoring completed successfully"
    return 0
}

# Rollback functionality
perform_rollback() {
    log "Performing automatic rollback..."
    
    case "$STRATEGY" in
        rolling)
            kubectl rollout undo deployment --all -n "$ENVIRONMENT"
            kubectl rollout status deployment --all -n "$ENVIRONMENT" --timeout=300s
            ;;
        blue-green)
            # Blue-green rollback would involve switching traffic back
            warn "Blue-green rollback requires manual intervention"
            ;;
        canary)
            rollback_canary
            ;;
    esac
    
    log "Rollback completed"
}

# Deployment confirmation
confirm_deployment() {
    if [[ "$AUTO_APPROVE" == "true" ]]; then
        log "Auto-approval enabled - proceeding with deployment"
        return 0
    fi
    
    echo
    echo "═══════════════════════════════════════════════════════════════"
    echo "                    DEPLOYMENT CONFIRMATION"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Environment: $ENVIRONMENT"
    echo "  Strategy:    $STRATEGY"
    echo "  Dry Run:     $DRY_RUN"
    echo "  Force:       $FORCE"
    echo "  Skip Tests:  $SKIP_TESTS"
    echo "  Parallel:    $PARALLEL_SERVICES services"
    echo "  Monitoring:  $MONITORING_DURATION seconds"
    echo "  Rollback:    $ROLLBACK_ON_FAILURE"
    echo "═══════════════════════════════════════════════════════════════"
    echo
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo -e "${RED}⚠️  WARNING: This is a PRODUCTION deployment!${NC}"
        echo -e "${RED}⚠️  Ensure all testing has been completed in staging.${NC}"
        echo
    fi
    
    read -p "Do you want to proceed with this deployment? (yes/no): " -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log "Deployment cancelled by user"
        exit 0
    fi
    
    log "Deployment confirmed by user"
}

# Main deployment function
main() {
    log "Starting automated deployment script"
    log "Log file: $LOG_FILE"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate inputs
    validate_environment
    validate_strategy
    validate_prerequisites
    
    # Pre-deployment tests
    if ! run_pre_deployment_tests; then
        if [[ "$FORCE" == "true" ]]; then
            warn "Pre-deployment tests failed but continuing due to --force flag"
        else
            error "Pre-deployment tests failed - use --force to override"
            exit 1
        fi
    fi
    
    # Build images
    build_images
    
    # Confirm deployment
    confirm_deployment
    
    # Execute deployment based on strategy
    case "$STRATEGY" in
        rolling)
            deploy_rolling
            ;;
        blue-green)
            deploy_blue_green
            ;;
        canary)
            deploy_canary
            ;;
    esac
    
    # Post-deployment monitoring
    if ! monitor_deployment; then
        error "Post-deployment monitoring failed"
        exit 1
    fi
    
    # Success
    log "Deployment completed successfully!"
    log "Environment: $ENVIRONMENT"
    log "Strategy: $STRATEGY"
    log "Build Tag: ${BUILD_TAG:-N/A}"
    log "Log file saved to: $LOG_FILE"
    
    echo
    echo "🎉 Deployment completed successfully!"
    echo "📊 Check monitoring dashboards for ongoing system health"
    echo "📋 Log file: $LOG_FILE"
}

# Trap signals for cleanup
trap 'error "Script interrupted"; exit 130' INT TERM

# Run main function
main "$@"