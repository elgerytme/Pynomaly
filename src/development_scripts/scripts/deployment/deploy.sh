#!/bin/bash

# anomaly_detection Deployment Automation Script
# Comprehensive deployment automation for all environments and monorepos

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_CONFIG="${PROJECT_ROOT}/deploy/config/deployment.yaml"
LOG_FILE="/tmp/anomaly_detection_deploy_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "${LOG_FILE}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_FILE}"
}

# Print banner
print_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
    ____                                _       
   |  _ \ _   _ _ __   ___  _ __ ___   __ _| |_   _ 
   | |_) | | | | '_ \ / _ \| '_ ` _ \ / _` | | | | |
   |  __/| |_| | | | | (_) | | | | | | (_| | | |_| |
   |_|    \__, |_| |_|\___/|_| |_| |_|\__,_|_|\__, |
          |___/                              |___/ 
          
    Deployment Automation Platform
    
EOF
    echo -e "${NC}"
}

# Help function
show_help() {
    cat << EOF
anomaly_detection Deployment Automation Script

USAGE:
    $0 [OPTIONS] COMMAND

COMMANDS:
    deploy          Deploy to specified environment
    status          Check deployment status
    rollback        Rollback to previous version
    health          Check application health
    logs            View deployment logs
    cleanup         Clean up old deployments

OPTIONS:
    -e, --environment   Target environment (development|staging|production)
    -v, --version       Version to deploy (default: latest)
    -p, --platform      Platform (docker_compose|kubernetes|helm)
    -s, --strategy      Deployment strategy (rolling_update|blue_green|canary)
    -f, --force         Force deployment without approval
    -d, --dry-run       Dry run without actual deployment
    -c, --config        Custom configuration file
    -h, --help          Show this help message
    -q, --quiet         Quiet mode (less verbose output)
    -v, --verbose       Verbose mode (more detailed output)

EXAMPLES:
    # Deploy to development environment
    $0 deploy -e development

    # Deploy specific version to staging with rolling update
    $0 deploy -e staging -v 1.2.3 -s rolling_update

    # Production deployment with blue-green strategy
    $0 deploy -e production -v 1.2.3 -s blue_green

    # Dry run production deployment
    $0 deploy -e production -v 1.2.3 --dry-run

    # Check deployment status
    $0 status -e production

    # Rollback production deployment
    $0 rollback -e production

    # Check application health
    $0 health -e production

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY         Docker registry URL
    KUBECONFIG             Kubernetes configuration file
    SLACK_WEBHOOK_URL      Slack webhook for notifications
    SMTP_USERNAME          SMTP username for email notifications
    SMTP_PASSWORD          SMTP password for email notifications
    PAGERDUTY_API_KEY      PagerDuty API key for alerts

EOF
}

# Parse command line arguments
parse_args() {
    ENVIRONMENT=""
    VERSION="latest"
    PLATFORM=""
    STRATEGY=""
    FORCE=false
    DRY_RUN=false
    CONFIG=""
    QUIET=false
    VERBOSE=false
    COMMAND=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -p|--platform)
                PLATFORM="$2"
                shift 2
                ;;
            -s|--strategy)
                STRATEGY="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -c|--config)
                CONFIG="$2"
                shift 2
                ;;
            -q|--quiet)
                QUIET=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            deploy|status|rollback|health|logs|cleanup)
                COMMAND="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$COMMAND" ]]; then
        log_error "Command is required"
        show_help
        exit 1
    fi

    if [[ "$COMMAND" != "logs" && "$COMMAND" != "cleanup" && -z "$ENVIRONMENT" ]]; then
        log_error "Environment is required for $COMMAND command"
        show_help
        exit 1
    fi
}

# Validate environment
validate_environment() {
    case "$ENVIRONMENT" in
        development|staging|production)
            log_info "Target environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if running in project root
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log_error "Must be run from project root directory"
        exit 1
    fi

    # Check required tools
    local required_tools=("docker" "git" "python3")
    
    if [[ "$PLATFORM" == "kubernetes" || "$PLATFORM" == "helm" ]]; then
        required_tools+=("kubectl")
    fi
    
    if [[ "$PLATFORM" == "helm" ]]; then
        required_tools+=("helm")
    fi

    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    # Check Kubernetes connectivity (if needed)
    if [[ "$PLATFORM" == "kubernetes" || "$PLATFORM" == "helm" ]]; then
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Cannot connect to Kubernetes cluster"
            exit 1
        fi
    fi

    log_success "Prerequisites check passed"
}

# Determine monorepo from environment if not specified
determine_platform() {
    if [[ -z "$PLATFORM" ]]; then
        case "$ENVIRONMENT" in
            development)
                PLATFORM="docker_compose"
                ;;
            staging|production)
                PLATFORM="kubernetes"
                ;;
        esac
    fi
    log_info "Using monorepo: $PLATFORM"
}

# Execute deployment
execute_deployment() {
    log_info "Starting deployment to $ENVIRONMENT environment..."
    log_info "Version: $VERSION"
    log_info "Platform: $PLATFORM"
    log_info "Strategy: ${STRATEGY:-default}"

    # Prepare deployment command
    local deploy_cmd="python3 ${SCRIPT_DIR}/deploy_manager.py deploy"
    deploy_cmd+=" --environment $ENVIRONMENT"
    deploy_cmd+=" --platform $PLATFORM"
    
    if [[ -n "$STRATEGY" ]]; then
        deploy_cmd+=" --strategy $STRATEGY"
    fi
    
    if [[ "$FORCE" == true ]]; then
        deploy_cmd+=" --force"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        deploy_cmd+=" --dry-run"
    fi
    
    deploy_cmd+=" --project-root $PROJECT_ROOT"

    # Set environment variables
    export VERSION="$VERSION"
    export ANOMALY_DETECTION_ENVIRONMENT="$ENVIRONMENT"

    # Execute deployment
    log_info "Executing: $deploy_cmd"
    
    if $deploy_cmd; then
        log_success "Deployment completed successfully!"
        return 0
    else
        log_error "Deployment failed!"
        return 1
    fi
}

# Check deployment status
check_deployment_status() {
    log_info "Checking deployment status for $ENVIRONMENT..."

    local status_cmd="python3 ${SCRIPT_DIR}/deploy_manager.py status"
    status_cmd+=" --environment $ENVIRONMENT"
    status_cmd+=" --platform $PLATFORM"
    status_cmd+=" --project-root $PROJECT_ROOT"

    if $status_cmd; then
        log_success "Status check completed"
        return 0
    else
        log_error "Status check failed"
        return 1
    fi
}

# Execute rollback
execute_rollback() {
    log_warn "Starting rollback for $ENVIRONMENT environment..."

    # Use automated deployer for rollback
    local rollback_cmd="python3 ${SCRIPT_DIR}/automated_deployer.py"
    rollback_cmd+=" --environment $ENVIRONMENT"
    rollback_cmd+=" --version previous"  # This would need to be implemented
    rollback_cmd+=" --force"

    if [[ -n "$CONFIG" ]]; then
        rollback_cmd+=" --config $CONFIG"
    fi

    log_info "Executing: $rollback_cmd"
    
    if $rollback_cmd; then
        log_success "Rollback completed successfully!"
        return 0
    else
        log_error "Rollback failed!"
        return 1
    fi
}

# Check application health
check_health() {
    log_info "Checking application health for $ENVIRONMENT..."

    local base_url
    case "$ENVIRONMENT" in
        production)
            base_url="https://api.anomaly_detection.io"
            ;;
        staging)
            base_url="https://staging-api.anomaly_detection.io"
            ;;
        development)
            base_url="http://localhost:8000"
            ;;
    esac

    local endpoints=(
        "/api/health"
        "/api/health/ready"
        "/api/health/live"
    )

    local all_healthy=true

    for endpoint in "${endpoints[@]}"; do
        local url="${base_url}${endpoint}"
        log_info "Checking: $url"
        
        if curl -sf "$url" &> /dev/null; then
            log_success "✓ $endpoint is healthy"
        else
            log_error "✗ $endpoint is unhealthy"
            all_healthy=false
        fi
    done

    if [[ "$all_healthy" == true ]]; then
        log_success "All health checks passed!"
        return 0
    else
        log_error "Some health checks failed!"
        return 1
    fi
}

# View deployment logs
view_logs() {
    log_info "Viewing deployment logs..."
    
    if [[ -f "$LOG_FILE" ]]; then
        tail -f "$LOG_FILE"
    else
        log_warn "No log file found"
    fi
}

# Cleanup old deployments
cleanup_deployments() {
    log_info "Cleaning up old deployments..."

    # Docker cleanup
    log_info "Cleaning up Docker images and containers..."
    docker system prune -f &> /dev/null || true
    
    # Remove old deployment logs (older than 30 days)
    find /tmp -name "anomaly_detection_deploy_*.log" -mtime +30 -delete 2>/dev/null || true
    
    # Kubernetes cleanup (if applicable)
    if command -v kubectl &> /dev/null; then
        log_info "Cleaning up old Kubernetes resources..."
        
        # Remove completed jobs older than 7 days
        kubectl get jobs --all-namespaces -o go-template \
            --template='{{range .items}}{{if gt (len .status.conditions) 0}}{{range .status.conditions}}{{if eq .type "Complete"}}{{if .status}}{{printf "%s %s\n" $.metadata.namespace $.metadata.name}}{{end}}{{end}}{{end}}{{end}}{{end}}' \
            | while read namespace job; do
                kubectl delete job "$job" -n "$namespace" 2>/dev/null || true
            done
    fi

    log_success "Cleanup completed"
}

# Send notification
send_notification() {
    local event_type="$1"
    local message="$2"
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local color
        case "$event_type" in
            success) color="good" ;;
            error) color="danger" ;;
            warning) color="warning" ;;
            *) color="#439FE0" ;;
        esac
        
        local payload="{
            \"text\": \"anomaly_detection Deployment $event_type\",
            \"attachments\": [{
                \"color\": \"$color\",
                \"fields\": [{
                    \"title\": \"Environment\",
                    \"value\": \"$ENVIRONMENT\",
                    \"short\": true
                }, {
                    \"title\": \"Version\",
                    \"value\": \"$VERSION\",
                    \"short\": true
                }, {
                    \"title\": \"Message\",
                    \"value\": \"$message\",
                    \"short\": false
                }]
            }]
        }"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "$payload" \
            "$SLACK_WEBHOOK_URL" &> /dev/null || true
    fi
}

# Main execution function
main() {
    print_banner
    
    log "Starting anomaly_detection deployment automation..."
    log "Log file: $LOG_FILE"
    
    parse_args "$@"
    
    if [[ "$COMMAND" != "logs" && "$COMMAND" != "cleanup" ]]; then
        validate_environment
        determine_platform
        check_prerequisites
    fi

    case "$COMMAND" in
        deploy)
            if execute_deployment; then
                send_notification "success" "Deployment to $ENVIRONMENT completed successfully"
                exit 0
            else
                send_notification "error" "Deployment to $ENVIRONMENT failed"
                exit 1
            fi
            ;;
        status)
            check_deployment_status
            ;;
        rollback)
            if execute_rollback; then
                send_notification "warning" "Rollback for $ENVIRONMENT completed"
                exit 0
            else
                send_notification "error" "Rollback for $ENVIRONMENT failed"
                exit 1
            fi
            ;;
        health)
            check_health
            ;;
        logs)
            view_logs
            ;;
        cleanup)
            cleanup_deployments
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Trap to ensure cleanup on exit
trap 'log "Deployment script finished"' EXIT

# Execute main function
main "$@"