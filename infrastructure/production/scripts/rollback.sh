#!/bin/bash

# Production Rollback Script for MLOps Platform
# This script provides safe rollback capabilities for production deployments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/infrastructure/production/terraform"
KUBERNETES_DIR="$PROJECT_ROOT/infrastructure/production/kubernetes"
ENVIRONMENT="${ENVIRONMENT:-production}"
AWS_REGION="${AWS_REGION:-us-west-2}"

# Default rollback strategy
ROLLBACK_STRATEGY="${ROLLBACK_STRATEGY:-blue-green}"  # blue-green, rolling, immediate

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

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Production Rollback Script for MLOps Platform

OPTIONS:
    -s, --strategy STRATEGY    Rollback strategy (blue-green|rolling|immediate)
    -t, --target TARGET        Rollback target (previous|version:X.X.X|commit:SHA)
    -c, --component COMPONENT  Rollback specific component only
    -d, --dry-run             Show what would be done without executing
    -f, --force               Force rollback without confirmation
    -h, --help                Show this help message

EXAMPLES:
    $0 --strategy blue-green --target previous
    $0 --strategy rolling --target version:1.2.3 --component model-server
    $0 --dry-run --target commit:abc123
    $0 --force --strategy immediate

STRATEGIES:
    blue-green    Safe rollback using blue-green deployment
    rolling       Rolling update rollback with zero downtime
    immediate     Immediate rollback (fastest but may cause brief downtime)

TARGETS:
    previous      Rollback to previous stable version
    version:X.X.X Rollback to specific version
    commit:SHA    Rollback to specific git commit

COMPONENTS:
    all                    Rollback entire platform (default)
    model-server          Model serving component
    feature-store         Feature store component
    inference-engine      Inference engine component
    ab-testing-service    A/B testing service
    model-governance      Model governance service
    automl-service        AutoML service
    explainability-service Explainability service
    api-gateway           API gateway component

EOF
}

# Parse command line arguments
parse_args() {
    ROLLBACK_TARGET="${ROLLBACK_TARGET:-previous}"
    ROLLBACK_COMPONENT="${ROLLBACK_COMPONENT:-all}"
    DRY_RUN=false
    FORCE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--strategy)
                ROLLBACK_STRATEGY="$2"
                shift 2
                ;;
            -t|--target)
                ROLLBACK_TARGET="$2"
                shift 2
                ;;
            -c|--component)
                ROLLBACK_COMPONENT="$2"
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
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Validate strategy
    case $ROLLBACK_STRATEGY in
        blue-green|rolling|immediate)
            ;;
        *)
            log_error "Invalid rollback strategy: $ROLLBACK_STRATEGY"
            exit 1
            ;;
    esac
    
    # Validate component
    local valid_components=("all" "model-server" "feature-store" "inference-engine" "ab-testing-service" "model-governance" "automl-service" "explainability-service" "api-gateway")
    if [[ ! " ${valid_components[*]} " =~ " ${ROLLBACK_COMPONENT} " ]]; then
        log_error "Invalid component: $ROLLBACK_COMPONENT"
        log_error "Valid components: ${valid_components[*]}"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites for rollback..."
    
    # Check required tools
    for tool in kubectl aws helm; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Verify we're in the correct cluster
    local current_context
    current_context=$(kubectl config current-context)
    log_info "Current Kubernetes context: $current_context"
    
    if [[ ! "$current_context" =~ $ENVIRONMENT ]]; then
        log_warning "Current context doesn't appear to be production environment"
        if [ "$FORCE" != true ]; then
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Rollback cancelled"
                exit 0
            fi
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Get current deployment state
get_current_state() {
    log_info "Getting current deployment state..."
    
    # Create state backup directory
    local state_dir="/tmp/mlops-rollback-state-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$state_dir"
    export STATE_DIR="$state_dir"
    
    # Backup current deployment state
    log_info "Backing up current state to $state_dir"
    
    # Get current deployments
    kubectl get deployments -n mlops-production -o yaml > "$state_dir/deployments.yaml"
    
    # Get current services
    kubectl get services -n mlops-production -o yaml > "$state_dir/services.yaml"
    
    # Get current HPAs
    kubectl get hpa -n mlops-production -o yaml > "$state_dir/hpa.yaml"
    
    # Get current configmaps and secrets
    kubectl get configmaps -n mlops-production -o yaml > "$state_dir/configmaps.yaml"
    kubectl get secrets -n mlops-production -o yaml > "$state_dir/secrets.yaml"
    
    # Get current image versions
    log_info "Current image versions:"
    kubectl get deployments -n mlops-production -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.template.spec.containers[0].image}{"\n"}{end}' | column -t
    
    log_success "Current state backed up to $state_dir"
}

# Resolve rollback target
resolve_target() {
    log_info "Resolving rollback target: $ROLLBACK_TARGET"
    
    case $ROLLBACK_TARGET in
        previous)
            # Get previous version from deployment annotations or labels
            local previous_version
            previous_version=$(kubectl get deployment model-server -n mlops-production -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}' 2>/dev/null || echo "")
            
            if [ -n "$previous_version" ] && [ "$previous_version" != "1" ]; then
                RESOLVED_TARGET="revision:$((previous_version - 1))"
                log_info "Resolved to revision: $((previous_version - 1))"
            else
                log_error "Cannot determine previous version"
                exit 1
            fi
            ;;
        version:*)
            RESOLVED_TARGET="$ROLLBACK_TARGET"
            local version="${ROLLBACK_TARGET#version:}"
            log_info "Resolved to version: $version"
            ;;
        commit:*)
            RESOLVED_TARGET="$ROLLBACK_TARGET"
            local commit="${ROLLBACK_TARGET#commit:}"
            log_info "Resolved to commit: $commit"
            ;;
        revision:*)
            RESOLVED_TARGET="$ROLLBACK_TARGET"
            local revision="${ROLLBACK_TARGET#revision:}"
            log_info "Resolved to revision: $revision"
            ;;
        *)
            log_error "Invalid rollback target: $ROLLBACK_TARGET"
            exit 1
            ;;
    esac
    
    export RESOLVED_TARGET
}

# Validate rollback safety
validate_rollback() {
    log_info "Validating rollback safety..."
    
    # Check if target version exists
    if [[ "$RESOLVED_TARGET" =~ ^revision: ]]; then
        local revision="${RESOLVED_TARGET#revision:}"
        
        # Check if revision exists for the deployment
        local deployment_name="model-server"
        if [ "$ROLLBACK_COMPONENT" != "all" ]; then
            deployment_name="$ROLLBACK_COMPONENT"
        fi
        
        if ! kubectl rollout history deployment/"$deployment_name" -n mlops-production --revision="$revision" &> /dev/null; then
            log_error "Revision $revision not found for deployment $deployment_name"
            exit 1
        fi
    fi
    
    # Check cluster health
    log_info "Checking cluster health..."
    
    local unhealthy_nodes
    unhealthy_nodes=$(kubectl get nodes --no-headers | grep -v " Ready " | wc -l)
    
    if [ "$unhealthy_nodes" -gt 0 ]; then
        log_warning "$unhealthy_nodes unhealthy nodes detected"
        if [ "$FORCE" != true ]; then
            read -p "Continue with rollback anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Rollback cancelled due to unhealthy nodes"
                exit 0
            fi
        fi
    fi
    
    # Check ongoing operations
    local ongoing_rollouts
    ongoing_rollouts=$(kubectl get deployments -n mlops-production -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.conditions[?(@.type=="Progressing")].status}{"\n"}{end}' | grep "True" | wc -l)
    
    if [ "$ongoing_rollouts" -gt 0 ]; then
        log_warning "Ongoing rollouts detected"
        if [ "$FORCE" != true ]; then
            read -p "Continue with rollback anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Rollback cancelled due to ongoing rollouts"
                exit 0
            fi
        fi
    fi
    
    log_success "Rollback safety validation passed"
}

# Create rollback plan
create_rollback_plan() {
    log_info "Creating rollback plan..."
    
    local plan_file="$STATE_DIR/rollback-plan.txt"
    
    cat > "$plan_file" << EOF
MLOps Platform Rollback Plan
============================

Date: $(date)
Environment: $ENVIRONMENT
Strategy: $ROLLBACK_STRATEGY
Target: $ROLLBACK_TARGET -> $RESOLVED_TARGET
Component: $ROLLBACK_COMPONENT
Dry Run: $DRY_RUN

Current State:
$(kubectl get deployments -n mlops-production -o custom-columns="NAME:.metadata.name,READY:.status.readyReplicas,AVAILABLE:.status.availableReplicas,IMAGE:.spec.template.spec.containers[0].image" --no-headers)

Rollback Steps:
EOF
    
    case $ROLLBACK_STRATEGY in
        blue-green)
            cat >> "$plan_file" << EOF
1. Create green environment with target version
2. Update load balancer to point to green environment
3. Validate green environment health
4. Gradually shift traffic from blue to green
5. Monitor metrics and error rates
6. Complete traffic cutover
7. Decommission blue environment
EOF
            ;;
        rolling)
            cat >> "$plan_file" << EOF
1. Update deployment with target version
2. Perform rolling update with zero downtime
3. Monitor pod health during rollout
4. Validate service availability
5. Check application metrics
6. Confirm rollback completion
EOF
            ;;
        immediate)
            cat >> "$plan_file" << EOF
1. Scale down current deployment
2. Update deployment with target version
3. Scale up new deployment
4. Validate service availability
5. Check application health
EOF
            ;;
    esac
    
    cat >> "$plan_file" << EOF

Risk Assessment:
- Service downtime: $([ "$ROLLBACK_STRATEGY" = "immediate" ] && echo "High" || echo "Low")
- Data loss risk: Low (stateless services)
- Recovery time: $([ "$ROLLBACK_STRATEGY" = "immediate" ] && echo "Fast" || echo "Gradual")

Rollback Validation Checks:
- Pod readiness probes
- Service health endpoints
- Application metrics
- Error rate monitoring
- Performance baseline comparison

Emergency Procedures:
- Emergency stop: kubectl delete deployment <name> -n mlops-production
- Traffic drain: kubectl scale deployment <name> --replicas=0 -n mlops-production
- Restore from backup: Use state backup in $STATE_DIR
EOF
    
    log_info "Rollback plan created: $plan_file"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN MODE - Showing rollback plan:"
        cat "$plan_file"
        return
    fi
    
    # Show plan and get confirmation
    if [ "$FORCE" != true ]; then
        echo ""
        log_warning "ROLLBACK PLAN SUMMARY:"
        echo "Strategy: $ROLLBACK_STRATEGY"
        echo "Target: $RESOLVED_TARGET"
        echo "Component: $ROLLBACK_COMPONENT"
        echo ""
        read -p "Do you want to proceed with this rollback? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Rollback cancelled by user"
            exit 0
        fi
    fi
}

# Execute blue-green rollback
execute_blue_green_rollback() {
    log_info "Executing blue-green rollback..."
    
    local components=()
    if [ "$ROLLBACK_COMPONENT" = "all" ]; then
        components=("model-server" "feature-store" "inference-engine" "ab-testing-service" "model-governance" "automl-service" "explainability-service" "api-gateway")
    else
        components=("$ROLLBACK_COMPONENT")
    fi
    
    for component in "${components[@]}"; do
        log_info "Rolling back $component using blue-green strategy..."
        
        # Create green deployment with target version
        local green_name="$component-green"
        
        # Get current deployment YAML and modify for green
        kubectl get deployment "$component" -n mlops-production -o yaml | \
            sed "s/name: $component/name: $green_name/" | \
            sed "s/app: $component/app: $green_name/" > "/tmp/$green_name.yaml"
        
        # Update image to target version if specified
        if [[ "$RESOLVED_TARGET" =~ ^version: ]]; then
            local target_version="${RESOLVED_TARGET#version:}"
            sed -i "s/:v[0-9.]*/:$target_version/" "/tmp/$green_name.yaml"
        fi
        
        # Deploy green environment
        kubectl apply -f "/tmp/$green_name.yaml"
        
        # Wait for green deployment to be ready
        log_info "Waiting for green deployment to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/"$green_name" -n mlops-production
        
        # Validate green deployment health
        log_info "Validating green deployment health..."
        local green_pods
        green_pods=$(kubectl get pods -n mlops-production -l app="$green_name" --field-selector=status.phase=Running --no-headers | wc -l)
        
        if [ "$green_pods" -eq 0 ]; then
            log_error "Green deployment for $component failed to start"
            kubectl delete deployment "$green_name" -n mlops-production
            continue
        fi
        
        # Update service to point to green deployment
        log_info "Switching traffic to green deployment..."
        kubectl patch service "$component" -n mlops-production -p '{"spec":{"selector":{"app":"'$green_name'"}}}'
        
        # Monitor for 30 seconds
        log_info "Monitoring green deployment for 30 seconds..."
        sleep 30
        
        # Check if green deployment is still healthy
        local healthy_pods
        healthy_pods=$(kubectl get pods -n mlops-production -l app="$green_name" --field-selector=status.phase=Running --no-headers | wc -l)
        
        if [ "$healthy_pods" -eq 0 ]; then
            log_error "Green deployment for $component became unhealthy"
            log_info "Rolling back to blue deployment..."
            kubectl patch service "$component" -n mlops-production -p '{"spec":{"selector":{"app":"'$component'"}}}'
            kubectl delete deployment "$green_name" -n mlops-production
            continue
        fi
        
        # Success - remove blue deployment and rename green to original
        log_info "Rollback successful for $component - cleaning up..."
        kubectl delete deployment "$component" -n mlops-production
        kubectl patch deployment "$green_name" -n mlops-production -p '{"metadata":{"name":"'$component'"}}'
        kubectl patch service "$component" -n mlops-production -p '{"spec":{"selector":{"app":"'$component'"}}}'
        
        log_success "Blue-green rollback completed for $component"
    done
}

# Execute rolling rollback
execute_rolling_rollback() {
    log_info "Executing rolling rollback..."
    
    local components=()
    if [ "$ROLLBACK_COMPONENT" = "all" ]; then
        components=("model-server" "feature-store" "inference-engine" "ab-testing-service" "model-governance" "automl-service" "explainability-service" "api-gateway")
    else
        components=("$ROLLBACK_COMPONENT")
    fi
    
    for component in "${components[@]}"; do
        log_info "Rolling back $component..."
        
        if [[ "$RESOLVED_TARGET" =~ ^revision: ]]; then
            local revision="${RESOLVED_TARGET#revision:}"
            
            # Rollback to specific revision
            kubectl rollout undo deployment/"$component" -n mlops-production --to-revision="$revision"
        else
            # Rollback to previous revision
            kubectl rollout undo deployment/"$component" -n mlops-production
        fi
        
        # Wait for rollout to complete
        log_info "Waiting for rollout to complete..."
        kubectl rollout status deployment/"$component" -n mlops-production --timeout=300s
        
        # Validate deployment health
        local ready_replicas
        ready_replicas=$(kubectl get deployment "$component" -n mlops-production -o jsonpath='{.status.readyReplicas}')
        local desired_replicas
        desired_replicas=$(kubectl get deployment "$component" -n mlops-production -o jsonpath='{.spec.replicas}')
        
        if [ "$ready_replicas" = "$desired_replicas" ]; then
            log_success "Rolling rollback completed for $component"
        else
            log_error "Rolling rollback failed for $component - not all replicas ready"
        fi
    done
}

# Execute immediate rollback
execute_immediate_rollback() {
    log_warning "Executing immediate rollback - this may cause brief service disruption..."
    
    local components=()
    if [ "$ROLLBACK_COMPONENT" = "all" ]; then
        components=("model-server" "feature-store" "inference-engine" "ab-testing-service" "model-governance" "automl-service" "explainability-service" "api-gateway")
    else
        components=("$ROLLBACK_COMPONENT")
    fi
    
    for component in "${components[@]}"; do
        log_info "Immediately rolling back $component..."
        
        # Scale down to 0
        kubectl scale deployment "$component" -n mlops-production --replicas=0
        
        # Wait for pods to terminate
        kubectl wait --for=delete pod -n mlops-production -l app="$component" --timeout=60s
        
        # Update to target version if specified
        if [[ "$RESOLVED_TARGET" =~ ^version: ]]; then
            local target_version="${RESOLVED_TARGET#version:}"
            kubectl set image deployment/"$component" -n mlops-production "$component"="mlops/$component:$target_version"
        elif [[ "$RESOLVED_TARGET" =~ ^revision: ]]; then
            local revision="${RESOLVED_TARGET#revision:}"
            kubectl rollout undo deployment/"$component" -n mlops-production --to-revision="$revision"
        fi
        
        # Scale back up
        local desired_replicas
        desired_replicas=$(kubectl get deployment "$component" -n mlops-production -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}' | wc -l)
        if [ "$desired_replicas" -eq 0 ]; then
            desired_replicas=3  # Default fallback
        fi
        
        kubectl scale deployment "$component" -n mlops-production --replicas="$desired_replicas"
        
        # Wait for deployment to be ready
        kubectl wait --for=condition=available --timeout=300s deployment/"$component" -n mlops-production
        
        log_success "Immediate rollback completed for $component"
    done
}

# Validate rollback success
validate_rollback_success() {
    log_info "Validating rollback success..."
    
    # Check deployment status
    local unhealthy_deployments
    unhealthy_deployments=$(kubectl get deployments -n mlops-production --no-headers | grep -v " Available " | wc -l)
    
    if [ "$unhealthy_deployments" -gt 0 ]; then
        log_error "Some deployments are not healthy after rollback"
        kubectl get deployments -n mlops-production
        return 1
    fi
    
    # Check pod status
    local unhealthy_pods
    unhealthy_pods=$(kubectl get pods -n mlops-production --no-headers | grep -v " Running " | grep -v " Completed " | wc -l)
    
    if [ "$unhealthy_pods" -gt 0 ]; then
        log_error "Some pods are not healthy after rollback"
        kubectl get pods -n mlops-production
        return 1
    fi
    
    # Test service endpoints if available
    local api_gateway_service
    api_gateway_service=$(kubectl get service api-gateway -n mlops-production -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
    
    if [ -n "$api_gateway_service" ]; then
        log_info "Testing API Gateway health endpoint..."
        if kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -- curl -f "$api_gateway_service:8000/health" &> /dev/null; then
            log_success "API Gateway health check passed"
        else
            log_warning "API Gateway health check failed"
        fi
    fi
    
    log_success "Rollback validation completed"
}

# Generate rollback report
generate_report() {
    log_info "Generating rollback report..."
    
    local report_file="$STATE_DIR/rollback-report.txt"
    
    cat > "$report_file" << EOF
MLOps Platform Rollback Report
==============================

Date: $(date)
Environment: $ENVIRONMENT
Strategy: $ROLLBACK_STRATEGY
Target: $ROLLBACK_TARGET -> $RESOLVED_TARGET
Component: $ROLLBACK_COMPONENT
Success: $([ $? -eq 0 ] && echo "Yes" || echo "No")

Final State:
$(kubectl get deployments -n mlops-production -o custom-columns="NAME:.metadata.name,READY:.status.readyReplicas,AVAILABLE:.status.availableReplicas,IMAGE:.spec.template.spec.containers[0].image" --no-headers)

Pod Status:
$(kubectl get pods -n mlops-production --no-headers)

Service Status:
$(kubectl get services -n mlops-production --no-headers)

HPA Status:
$(kubectl get hpa -n mlops-production --no-headers)

Rollback Duration: $(( $(date +%s) - START_TIME )) seconds

State Backup Location: $STATE_DIR

Notes:
- All deployment configurations backed up before rollback
- Monitoring should be checked for any anomalies
- Consider running performance tests to validate rollback success
- Review logs for any error patterns
EOF
    
    log_success "Rollback report generated: $report_file"
    
    # Display summary
    echo ""
    log_info "ROLLBACK SUMMARY:"
    echo "Strategy: $ROLLBACK_STRATEGY"
    echo "Target: $RESOLVED_TARGET"
    echo "Component: $ROLLBACK_COMPONENT"
    echo "Duration: $(( $(date +%s) - START_TIME )) seconds"
    echo "Report: $report_file"
    echo ""
}

# Main rollback function
main() {
    export START_TIME=$(date +%s)
    
    log_info "🚨 MLOps Platform Production Rollback Started"
    
    # Parse arguments
    parse_args "$@"
    
    # Show configuration
    log_info "Rollback Configuration:"
    echo "  Strategy: $ROLLBACK_STRATEGY"
    echo "  Target: $ROLLBACK_TARGET"
    echo "  Component: $ROLLBACK_COMPONENT"
    echo "  Dry Run: $DRY_RUN"
    echo "  Force: $FORCE"
    echo ""
    
    # Execute rollback steps
    check_prerequisites
    get_current_state
    resolve_target
    validate_rollback
    create_rollback_plan
    
    if [ "$DRY_RUN" = true ]; then
        log_success "Dry run completed - no changes made"
        exit 0
    fi
    
    # Execute rollback based on strategy
    case $ROLLBACK_STRATEGY in
        blue-green)
            execute_blue_green_rollback
            ;;
        rolling)
            execute_rolling_rollback
            ;;
        immediate)
            execute_immediate_rollback
            ;;
    esac
    
    # Validate and report
    validate_rollback_success
    generate_report
    
    log_success "🎯 MLOps Platform Rollback Completed Successfully!"
    log_info "Please monitor the system for the next 15-30 minutes to ensure stability"
}

# Handle script interruption
cleanup() {
    log_warning "Rollback interrupted - check system state manually"
    if [ -n "${STATE_DIR:-}" ]; then
        log_info "State backup available at: $STATE_DIR"
    fi
    exit 1
}

trap cleanup INT TERM

# Run main function
main "$@"