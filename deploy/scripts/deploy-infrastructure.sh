#!/bin/bash

# Infrastructure Deployment Script
# Automated deployment of the anomaly detection platform infrastructure

set -euo pipefail

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

# Default values
ENVIRONMENT=""
DRY_RUN=false
SKIP_VALIDATION=false
AUTO_APPROVE=false

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy infrastructure for the anomaly detection platform.

OPTIONS:
    -e, --environment ENVIRONMENT    Target environment (staging|production)
    -d, --dry-run                   Show what would be deployed without making changes
    -s, --skip-validation          Skip pre-deployment validation
    -y, --auto-approve             Auto-approve Terraform changes
    -h, --help                     Show this help message

EXAMPLES:
    $0 -e staging                  Deploy to staging
    $0 -e production -y            Deploy to production with auto-approve
    $0 -e staging -d               Dry run for staging

ENVIRONMENT VARIABLES:
    TF_VAR_kubernetes_cluster_endpoint      Kubernetes API endpoint
    TF_VAR_kubernetes_cluster_ca_certificate Kubernetes CA certificate
    TF_VAR_kubernetes_token                 Kubernetes authentication token
    AWS_REGION                              AWS region for state backend
    AWS_ACCESS_KEY_ID                       AWS access key
    AWS_SECRET_ACCESS_KEY                   AWS secret key

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -s|--skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        -y|--auto-approve)
            AUTO_APPROVE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown parameter: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment parameter
if [[ -z "$ENVIRONMENT" ]]; then
    log_error "Environment parameter is required"
    usage
    exit 1
fi

if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    log_error "Environment must be 'staging' or 'production'"
    exit 1
fi

# Check required tools
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required commands
    for cmd in terraform kubectl helm; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_tools+=("$cmd")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and try again"
        exit 1
    fi
    
    # Check Terraform version
    local tf_version
    tf_version=$(terraform version -json | jq -r '.terraform_version')
    if [[ "$(printf '%s\n' "1.0.0" "$tf_version" | sort -V | head -n1)" != "1.0.0" ]]; then
        log_error "Terraform version 1.0+ is required, found: $tf_version"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Validate environment variables
validate_environment() {
    log_info "Validating environment configuration..."
    
    local required_vars=(
        "TF_VAR_kubernetes_cluster_endpoint"
        "TF_VAR_kubernetes_cluster_ca_certificate"
        "TF_VAR_kubernetes_token"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        printf '  %s\n' "${missing_vars[@]}"
        log_info "Please set the missing variables and try again"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Test Kubernetes connectivity
test_kubernetes_connectivity() {
    log_info "Testing Kubernetes connectivity..."
    
    # Configure kubectl temporarily
    local temp_kubeconfig
    temp_kubeconfig=$(mktemp)
    
    # Create kubeconfig
    cat > "$temp_kubeconfig" << EOF
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority-data: ${TF_VAR_kubernetes_cluster_ca_certificate}
    server: ${TF_VAR_kubernetes_cluster_endpoint}
  name: ${ENVIRONMENT}-cluster
contexts:
- context:
    cluster: ${ENVIRONMENT}-cluster
    user: ${ENVIRONMENT}-user
  name: ${ENVIRONMENT}-context
current-context: ${ENVIRONMENT}-context
users:
- name: ${ENVIRONMENT}-user
  user:
    token: ${TF_VAR_kubernetes_token}
EOF
    
    # Test connection
    if KUBECONFIG="$temp_kubeconfig" kubectl cluster-info >/dev/null 2>&1; then
        log_success "Kubernetes connectivity test passed"
    else
        log_error "Kubernetes connectivity test failed"
        rm -f "$temp_kubeconfig"
        exit 1
    fi
    
    rm -f "$temp_kubeconfig"
}

# Initialize Terraform
initialize_terraform() {
    log_info "Initializing Terraform..."
    
    cd "$(dirname "$0")/../terraform"
    
    # Initialize Terraform with backend configuration
    terraform init \
        -backend-config="bucket=anomaly-detection-terraform-state" \
        -backend-config="key=infrastructure/${ENVIRONMENT}/terraform.tfstate" \
        -backend-config="region=${AWS_REGION:-us-east-1}"
    
    log_success "Terraform initialization completed"
}

# Plan infrastructure changes
plan_infrastructure() {
    log_info "Planning infrastructure changes..."
    
    local plan_args=(
        "-var-file=environments/${ENVIRONMENT}.tfvars"
        "-out=${ENVIRONMENT}.tfplan"
    )
    
    if [[ "$DRY_RUN" == "true" ]]; then
        plan_args+=("-detailed-exitcode")
    fi
    
    terraform plan "${plan_args[@]}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run completed. No changes were applied."
        exit 0
    fi
    
    log_success "Infrastructure planning completed"
}

# Apply infrastructure changes
apply_infrastructure() {
    log_info "Applying infrastructure changes..."
    
    local apply_args=("${ENVIRONMENT}.tfplan")
    
    if [[ "$AUTO_APPROVE" == "true" ]]; then
        terraform apply "${apply_args[@]}"
    else
        log_warning "Review the plan above. Do you want to apply these changes?"
        read -p "Enter 'yes' to continue: " -r
        if [[ $REPLY == "yes" ]]; then
            terraform apply "${apply_args[@]}"
        else
            log_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    log_success "Infrastructure deployment completed"
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    # Get Terraform outputs
    APPLICATION_URL=$(terraform output -raw application_url)
    NAMESPACE=$(terraform output -raw namespace)
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check if pods are running
    local temp_kubeconfig
    temp_kubeconfig=$(mktemp)
    
    cat > "$temp_kubeconfig" << EOF
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority-data: ${TF_VAR_kubernetes_cluster_ca_certificate}
    server: ${TF_VAR_kubernetes_cluster_endpoint}
  name: ${ENVIRONMENT}-cluster
contexts:
- context:
    cluster: ${ENVIRONMENT}-cluster
    user: ${ENVIRONMENT}-user
  name: ${ENVIRONMENT}-context
current-context: ${ENVIRONMENT}-context
users:
- name: ${ENVIRONMENT}-user
  user:
    token: ${TF_VAR_kubernetes_token}
EOF
    
    # Check pod status
    local pods_ready=0
    for i in {1..30}; do
        if KUBECONFIG="$temp_kubeconfig" kubectl get pods -n "$NAMESPACE" | grep -E "(Running|Completed)" >/dev/null 2>&1; then
            pods_ready=1
            break
        fi
        log_info "Waiting for pods to be ready... (attempt $i/30)"
        sleep 10
    done
    
    if [[ $pods_ready -eq 0 ]]; then
        log_error "Pods are not ready after 5 minutes"
        rm -f "$temp_kubeconfig"
        exit 1
    fi
    
    # Test application health (if accessible)
    if [[ "$APPLICATION_URL" != *"localhost"* ]]; then
        for i in {1..10}; do
            if curl -f -s "${APPLICATION_URL}/api/health/ready" >/dev/null 2>&1; then
                log_success "Application health check passed"
                break
            elif [[ $i -eq 10 ]]; then
                log_warning "Application health check failed (may need DNS/ingress setup)"
            else
                log_info "Waiting for application to be ready... (attempt $i/10)"
                sleep 15
            fi
        done
    fi
    
    rm -f "$temp_kubeconfig"
    log_success "Deployment validation completed"
}

# Generate deployment report
generate_deployment_report() {
    log_info "Generating deployment report..."
    
    local report_file="deployment-report-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# 🚀 Infrastructure Deployment Report

**Environment:** ${ENVIRONMENT}  
**Deployment Date:** $(date)  
**Terraform Version:** $(terraform version -json | jq -r '.terraform_version')  

## 📊 Deployment Summary

$(terraform output -json | jq -r 'to_entries[] | "- **\(.key):** \(.value.value)"')

## 🔗 Access URLs

- **Application:** $(terraform output -raw application_url)
$(if terraform output grafana_url >/dev/null 2>&1; then
    echo "- **Grafana:** $(terraform output -raw grafana_url)"
fi)

## ✅ Next Steps

1. Configure DNS records to point to the application URL
2. Set up SSL certificates for HTTPS
3. Configure monitoring alerts and notifications
4. Run comprehensive security scans
5. Perform load testing
6. Set up backup and disaster recovery procedures

## 🛠️ Manual Configuration Required

1. **DNS Configuration:**
   - Point $(terraform output -raw application_url | sed 's|https://||') to your load balancer
   $(if terraform output grafana_url >/dev/null 2>&1; then
       echo "   - Point $(terraform output -raw grafana_url | sed 's|https://||') to your monitoring ingress"
   fi)

2. **SSL Certificates:**
   - Install SSL certificates for your domains
   - Configure cert-manager for automatic certificate renewal

3. **Monitoring Setup:**
   - Access Grafana with admin credentials
   - Import custom dashboards from deploy/monitoring/grafana/dashboards/
   - Configure alert notification channels

4. **Security Configuration:**
   - Review and update security group rules
   - Configure network policies
   - Set up vulnerability scanning schedules

## 📞 Support

If you encounter issues:
1. Check the deployment logs above
2. Validate Kubernetes cluster health
3. Verify network connectivity and DNS resolution
4. Review Terraform state for any inconsistencies

EOF

    log_success "Deployment report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting infrastructure deployment for environment: $ENVIRONMENT"
    
    # Run pre-deployment checks
    check_prerequisites
    
    if [[ "$SKIP_VALIDATION" != "true" ]]; then
        validate_environment
        test_kubernetes_connectivity
    fi
    
    # Deploy infrastructure
    initialize_terraform
    plan_infrastructure
    
    if [[ "$DRY_RUN" != "true" ]]; then
        apply_infrastructure
        validate_deployment
        generate_deployment_report
        
        log_success "🎉 Infrastructure deployment completed successfully!"
        log_info "📋 Check the deployment report for next steps and access information"
    fi
}

# Execute main function
main "$@"