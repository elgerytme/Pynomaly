#!/bin/bash

# Multi-Cloud Deployment Script for MLOps Platform
# Deploys MLOps platform across AWS, GCP, and Azure

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENVIRONMENT="${ENVIRONMENT:-prod}"
PROJECT_NAME="${PROJECT_NAME:-mlops-platform}"

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
    echo -e "${PURPLE}[MULTI-CLOUD]${NC} $1"
}

# Cloud provider flags
ENABLE_AWS="${ENABLE_AWS:-true}"
ENABLE_GCP="${ENABLE_GCP:-true}"
ENABLE_AZURE="${ENABLE_AZURE:-false}"

# Check prerequisites
check_prerequisites() {
    log_header "Checking prerequisites for multi-cloud deployment..."
    
    local required_tools=("terraform" "kubectl" "helm" "istioctl")
    
    # Check AWS tools if AWS is enabled
    if [[ "$ENABLE_AWS" == "true" ]]; then
        required_tools+=("aws")
    fi
    
    # Check GCP tools if GCP is enabled
    if [[ "$ENABLE_GCP" == "true" ]]; then
        required_tools+=("gcloud")
    fi
    
    # Check Azure tools if Azure is enabled
    if [[ "$ENABLE_AZURE" == "true" ]]; then
        required_tools+=("az")
    fi
    
    # Verify all required tools are installed
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Verify cloud authentication
    if [[ "$ENABLE_AWS" == "true" ]]; then
        if ! aws sts get-caller-identity &> /dev/null; then
            log_error "AWS credentials not configured. Run 'aws configure' first."
            exit 1
        fi
        log_success "AWS authentication verified"
    fi
    
    if [[ "$ENABLE_GCP" == "true" ]]; then
        if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 &> /dev/null; then
            log_error "GCP credentials not configured. Run 'gcloud auth login' first."
            exit 1
        fi
        log_success "GCP authentication verified"
    fi
    
    if [[ "$ENABLE_AZURE" == "true" ]]; then
        if ! az account show &> /dev/null; then
            log_error "Azure credentials not configured. Run 'az login' first."
            exit 1
        fi
        log_success "Azure authentication verified"
    fi
    
    log_success "Prerequisites check completed"
}

# Initialize Terraform
initialize_terraform() {
    log_header "Initializing Terraform..."
    
    cd "${SCRIPT_DIR}/terraform"
    
    # Initialize Terraform with backend configuration
    terraform init \
        -backend-config="bucket=${PROJECT_NAME}-terraform-state" \
        -backend-config="key=multi-cloud/${ENVIRONMENT}/terraform.tfstate" \
        -backend-config="region=us-east-1"
    
    # Validate Terraform configuration
    terraform validate
    
    log_success "Terraform initialized successfully"
}

# Plan infrastructure deployment
plan_infrastructure() {
    log_header "Planning infrastructure deployment..."
    
    cd "${SCRIPT_DIR}/terraform"
    
    # Create terraform variables file
    cat > terraform.tfvars << EOF
environment = "${ENVIRONMENT}"
project_name = "${PROJECT_NAME}"
enable_aws = ${ENABLE_AWS}
enable_gcp = ${ENABLE_GCP}
enable_azure = ${ENABLE_AZURE}

# Instance configurations
node_count = 3
node_instance_type = {
  aws   = "t3.large"
  gcp   = "e2-standard-4"
  azure = "Standard_D4s_v3"
}

# Regional configurations
aws_region = "us-east-1"
gcp_region = "us-central1"
azure_location = "East US"
EOF
    
    # Generate Terraform plan
    terraform plan \
        -var-file="terraform.tfvars" \
        -out="tfplan" \
        -detailed-exitcode
    
    local plan_exitcode=$?
    
    if [[ $plan_exitcode -eq 0 ]]; then
        log_info "No infrastructure changes required"
    elif [[ $plan_exitcode -eq 2 ]]; then
        log_info "Infrastructure changes planned successfully"
    else
        log_error "Terraform plan failed"
        exit 1
    fi
    
    log_success "Infrastructure planning completed"
}

# Deploy infrastructure
deploy_infrastructure() {
    log_header "Deploying infrastructure..."
    
    cd "${SCRIPT_DIR}/terraform"
    
    # Apply Terraform plan
    terraform apply "tfplan"
    
    # Save outputs for later use
    terraform output -json > outputs.json
    
    log_success "Infrastructure deployment completed"
}

# Configure kubectl contexts
configure_kubectl_contexts() {
    log_header "Configuring kubectl contexts..."
    
    cd "${SCRIPT_DIR}/terraform"
    
    # Read Terraform outputs
    local outputs=$(cat outputs.json)
    
    # Configure AWS EKS context
    if [[ "$ENABLE_AWS" == "true" ]]; then
        local aws_cluster_name=$(echo "$outputs" | jq -r '.aws_cluster_endpoint.value // empty' | sed 's/.*\///')
        if [[ -n "$aws_cluster_name" ]]; then
            aws eks update-kubeconfig --region us-east-1 --name "${PROJECT_NAME}-${ENVIRONMENT}-aws"
            kubectl config rename-context "arn:aws:eks:us-east-1:$(aws sts get-caller-identity --query Account --output text):cluster/${PROJECT_NAME}-${ENVIRONMENT}-aws" "${PROJECT_NAME}-aws"
            log_success "AWS EKS context configured"
        fi
    fi
    
    # Configure GCP GKE context
    if [[ "$ENABLE_GCP" == "true" ]]; then
        local gcp_project=$(gcloud config get-value project)
        gcloud container clusters get-credentials "${PROJECT_NAME}-${ENVIRONMENT}-gcp" --region us-central1 --project "$gcp_project"
        kubectl config rename-context "gke_${gcp_project}_us-central1_${PROJECT_NAME}-${ENVIRONMENT}-gcp" "${PROJECT_NAME}-gcp"
        log_success "GCP GKE context configured"
    fi
    
    # Configure Azure AKS context
    if [[ "$ENABLE_AZURE" == "true" ]]; then
        az aks get-credentials --resource-group "${PROJECT_NAME}-${ENVIRONMENT}" --name "${PROJECT_NAME}-${ENVIRONMENT}-azure"
        kubectl config rename-context "${PROJECT_NAME}-${ENVIRONMENT}-azure" "${PROJECT_NAME}-azure"
        log_success "Azure AKS context configured"
    fi
    
    # List available contexts
    log_info "Available kubectl contexts:"
    kubectl config get-contexts
}

# Deploy Istio service mesh for multi-cluster
deploy_istio_multicluster() {
    log_header "Deploying Istio multi-cluster service mesh..."
    
    # Install Istio on each cluster
    local clusters=()
    [[ "$ENABLE_AWS" == "true" ]] && clusters+=("${PROJECT_NAME}-aws")
    [[ "$ENABLE_GCP" == "true" ]] && clusters+=("${PROJECT_NAME}-gcp")
    [[ "$ENABLE_AZURE" == "true" ]] && clusters+=("${PROJECT_NAME}-azure")
    
    for cluster in "${clusters[@]}"; do
        log_info "Installing Istio on cluster: $cluster"
        
        # Switch to cluster context
        kubectl config use-context "$cluster"
        
        # Install Istio
        istioctl install --set values.pilot.env.EXTERNAL_ISTIOD=false -y
        
        # Label namespace for Istio injection
        kubectl label namespace default istio-injection=enabled --overwrite
        
        # Create network for multi-cluster
        kubectl label namespace istio-system topology.istio.io/network="network-${cluster#*-}" --overwrite
        
        log_success "Istio installed on cluster: $cluster"
    done
    
    # Configure cross-cluster discovery
    if [[ ${#clusters[@]} -gt 1 ]]; then
        log_info "Configuring cross-cluster service discovery..."
        
        # Create secrets for cross-cluster access
        for primary_cluster in "${clusters[@]}"; do
            kubectl config use-context "$primary_cluster"
            
            # Create cluster secret for other clusters
            for remote_cluster in "${clusters[@]}"; do
                if [[ "$remote_cluster" != "$primary_cluster" ]]; then
                    # Get remote cluster credentials
                    kubectl config use-context "$remote_cluster"
                    kubectl get secret -n istio-system istio-reader-service-account -o yaml > "/tmp/${remote_cluster}-secret.yaml"
                    
                    # Apply to primary cluster
                    kubectl config use-context "$primary_cluster"
                    sed "s/istio-reader-service-account/${remote_cluster}-secret/g" "/tmp/${remote_cluster}-secret.yaml" | \
                    kubectl apply -n istio-system -f -
                    
                    # Label the secret
                    kubectl label secret "${remote_cluster}-secret" -n istio-system istio/cluster-name="$remote_cluster"
                    
                    # Annotate with endpoint
                    local remote_endpoint=$(kubectl --context="$remote_cluster" get svc istio-pilot -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
                    if [[ -z "$remote_endpoint" ]]; then
                        remote_endpoint=$(kubectl --context="$remote_cluster" get svc istio-pilot -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
                    fi
                    kubectl annotate secret "${remote_cluster}-secret" -n istio-system istio/cluster-endpoint="$remote_endpoint"
                fi
            done
        done
        
        log_success "Cross-cluster service discovery configured"
    fi
}

# Deploy MLOps applications
deploy_mlops_applications() {
    log_header "Deploying MLOps applications to clusters..."
    
    local clusters=()
    [[ "$ENABLE_AWS" == "true" ]] && clusters+=("${PROJECT_NAME}-aws")
    [[ "$ENABLE_GCP" == "true" ]] && clusters+=("${PROJECT_NAME}-gcp")
    [[ "$ENABLE_AZURE" == "true" ]] && clusters+=("${PROJECT_NAME}-azure")
    
    for cluster in "${clusters[@]}"; do
        log_info "Deploying applications to cluster: $cluster"
        
        # Switch to cluster context
        kubectl config use-context "$cluster"
        
        # Create namespace
        kubectl create namespace mlops-production --dry-run=client -o yaml | kubectl apply -f -
        kubectl label namespace mlops-production istio-injection=enabled --overwrite
        
        # Deploy applications using Helm
        helm repo add mlops-charts "${PROJECT_ROOT}/charts"
        helm repo update
        
        # Deploy with cluster-specific values
        helm upgrade --install mlops-platform mlops-charts/mlops-platform \
            --namespace mlops-production \
            --values "${SCRIPT_DIR}/helm-values/${cluster}-values.yaml" \
            --set global.environment="$ENVIRONMENT" \
            --set global.cluster="$cluster" \
            --wait --timeout=10m
        
        log_success "Applications deployed to cluster: $cluster"
    done
}

# Configure global load balancing
configure_global_load_balancing() {
    log_header "Configuring global load balancing..."
    
    # Configure DNS-based load balancing using Route53 (if AWS is enabled)
    if [[ "$ENABLE_AWS" == "true" ]]; then
        log_info "Setting up Route53 DNS-based load balancing..."
        
        # Get load balancer endpoints
        local aws_lb_dns=""
        local gcp_lb_ip=""
        
        if [[ "$ENABLE_AWS" == "true" ]]; then
            kubectl config use-context "${PROJECT_NAME}-aws"
            aws_lb_dns=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        fi
        
        if [[ "$ENABLE_GCP" == "true" ]]; then
            kubectl config use-context "${PROJECT_NAME}-gcp"
            gcp_lb_ip=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        fi
        
        # Create Route53 records with weighted routing
        if [[ -n "$aws_lb_dns" ]]; then
            aws route53 change-resource-record-sets \
                --hosted-zone-id "$(aws route53 list-hosted-zones-by-name --dns-name mlops-platform.com --query 'HostedZones[0].Id' --output text | sed 's|/hostedzone/||')" \
                --change-batch "{
                    \"Changes\": [{
                        \"Action\": \"UPSERT\",
                        \"ResourceRecordSet\": {
                            \"Name\": \"api.mlops-platform.com\",
                            \"Type\": \"CNAME\",
                            \"SetIdentifier\": \"aws-region\",
                            \"Weight\": 100,
                            \"TTL\": 60,
                            \"ResourceRecords\": [{\"Value\": \"$aws_lb_dns\"}]
                        }
                    }]
                }"
        fi
        
        if [[ -n "$gcp_lb_ip" ]]; then
            aws route53 change-resource-record-sets \
                --hosted-zone-id "$(aws route53 list-hosted-zones-by-name --dns-name mlops-platform.com --query 'HostedZones[0].Id' --output text | sed 's|/hostedzone/||')" \
                --change-batch "{
                    \"Changes\": [{
                        \"Action\": \"UPSERT\",
                        \"ResourceRecordSet\": {
                            \"Name\": \"api.mlops-platform.com\",
                            \"Type\": \"A\",
                            \"SetIdentifier\": \"gcp-region\",
                            \"Weight\": 100,
                            \"TTL\": 60,
                            \"ResourceRecords\": [{\"Value\": \"$gcp_lb_ip\"}]
                        }
                    }]
                }"
        fi
        
        log_success "Global load balancing configured"
    fi
}

# Validate deployment
validate_deployment() {
    log_header "Validating multi-cloud deployment..."
    
    local clusters=()
    [[ "$ENABLE_AWS" == "true" ]] && clusters+=("${PROJECT_NAME}-aws")
    [[ "$ENABLE_GCP" == "true" ]] && clusters+=("${PROJECT_NAME}-gcp")
    [[ "$ENABLE_AZURE" == "true" ]] && clusters+=("${PROJECT_NAME}-azure")
    
    for cluster in "${clusters[@]}"; do
        log_info "Validating cluster: $cluster"
        
        # Switch to cluster context
        kubectl config use-context "$cluster"
        
        # Check pod status
        log_info "Checking pod status..."
        kubectl get pods -n mlops-production
        
        # Check if all pods are running
        local not_running=$(kubectl get pods -n mlops-production --field-selector=status.phase!=Running --no-headers | wc -l)
        if [[ $not_running -gt 0 ]]; then
            log_warning "$not_running pods are not running in cluster $cluster"
        else
            log_success "All pods are running in cluster $cluster"
        fi
        
        # Test service connectivity
        log_info "Testing service connectivity..."
        local api_endpoint=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [[ -z "$api_endpoint" ]]; then
            api_endpoint=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        fi
        
        if [[ -n "$api_endpoint" ]]; then
            if curl -f -s --max-time 10 "http://$api_endpoint/health" > /dev/null; then
                log_success "API health check passed for cluster $cluster"
            else
                log_warning "API health check failed for cluster $cluster"
            fi
        fi
    done
    
    log_success "Multi-cloud deployment validation completed"
}

# Generate deployment report
generate_deployment_report() {
    log_header "Generating deployment report..."
    
    local report_file="${SCRIPT_DIR}/multi-cloud-deployment-report-$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Multi-Cloud Deployment Report

**Date:** $(date)
**Environment:** $ENVIRONMENT
**Project:** $PROJECT_NAME

## Deployment Summary

### Enabled Cloud Providers
- AWS: $ENABLE_AWS
- GCP: $ENABLE_GCP
- Azure: $ENABLE_AZURE

### Cluster Information

EOF
    
    if [[ "$ENABLE_AWS" == "true" ]]; then
        kubectl config use-context "${PROJECT_NAME}-aws"
        echo "#### AWS EKS Cluster" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
        kubectl cluster-info >> "$report_file" 2>/dev/null || echo "Cluster info not available" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    if [[ "$ENABLE_GCP" == "true" ]]; then
        kubectl config use-context "${PROJECT_NAME}-gcp"
        echo "#### GCP GKE Cluster" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
        kubectl cluster-info >> "$report_file" 2>/dev/null || echo "Cluster info not available" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    if [[ "$ENABLE_AZURE" == "true" ]]; then
        kubectl config use-context "${PROJECT_NAME}-azure"
        echo "#### Azure AKS Cluster" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
        kubectl cluster-info >> "$report_file" 2>/dev/null || echo "Cluster info not available" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

### Service Endpoints

- Global API: https://api.mlops-platform.com
- Monitoring: https://monitoring.mlops-platform.com

### Next Steps

1. Configure DNS entries for production domains
2. Set up SSL certificates for all endpoints
3. Configure monitoring and alerting across all clusters
4. Test cross-cluster service communication
5. Implement disaster recovery procedures

### Management Commands

\`\`\`bash
# Switch between clusters
kubectl config use-context ${PROJECT_NAME}-aws
kubectl config use-context ${PROJECT_NAME}-gcp
kubectl config use-context ${PROJECT_NAME}-azure

# View all contexts
kubectl config get-contexts

# Deploy updates to all clusters
${SCRIPT_DIR}/deploy-multi-cloud.sh
\`\`\`

EOF
    
    log_success "Deployment report generated: $report_file"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/*-secret.yaml
}

# Main deployment function
main() {
    log_header "Starting multi-cloud deployment for MLOps platform..."
    echo "=================================================================="
    echo "🌐 MULTI-CLOUD DEPLOYMENT STARTING"
    echo "=================================================================="
    echo "Environment: $ENVIRONMENT"
    echo "AWS: $ENABLE_AWS"
    echo "GCP: $ENABLE_GCP"
    echo "Azure: $ENABLE_AZURE"
    echo "=================================================================="
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Check if we should skip certain steps
    SKIP_INFRA=${SKIP_INFRA:-false}
    SKIP_APPS=${SKIP_APPS:-false}
    SKIP_VALIDATION=${SKIP_VALIDATION:-false}
    
    # Execute deployment steps
    check_prerequisites
    
    if [[ "$SKIP_INFRA" != "true" ]]; then
        initialize_terraform
        plan_infrastructure
        deploy_infrastructure
        configure_kubectl_contexts
        deploy_istio_multicluster
    fi
    
    if [[ "$SKIP_APPS" != "true" ]]; then
        deploy_mlops_applications
        configure_global_load_balancing
    fi
    
    if [[ "$SKIP_VALIDATION" != "true" ]]; then
        validate_deployment
    fi
    
    generate_deployment_report
    
    echo "=================================================================="
    log_header "🎉 MULTI-CLOUD DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo "=================================================================="
    log_success "MLOps platform is now deployed across multiple cloud providers!"
    echo "=================================================================="
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy MLOps platform across multiple cloud providers

OPTIONS:
    -h, --help              Show this help message
    --skip-infra           Skip infrastructure deployment
    --skip-apps            Skip application deployment
    --skip-validation      Skip deployment validation
    --aws-only             Deploy only to AWS
    --gcp-only             Deploy only to GCP
    --azure-only           Deploy only to Azure

ENVIRONMENT VARIABLES:
    ENVIRONMENT            Environment name (default: prod)
    PROJECT_NAME           Project name (default: mlops-platform)
    ENABLE_AWS             Enable AWS deployment (default: true)
    ENABLE_GCP             Enable GCP deployment (default: true)
    ENABLE_AZURE           Enable Azure deployment (default: false)

EXAMPLES:
    # Full multi-cloud deployment
    $0
    
    # AWS only deployment
    $0 --aws-only
    
    # Skip infrastructure deployment
    $0 --skip-infra

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --skip-infra)
            SKIP_INFRA=true
            shift
            ;;
        --skip-apps)
            SKIP_APPS=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --aws-only)
            ENABLE_AWS=true
            ENABLE_GCP=false
            ENABLE_AZURE=false
            shift
            ;;
        --gcp-only)
            ENABLE_AWS=false
            ENABLE_GCP=true
            ENABLE_AZURE=false
            shift
            ;;
        --azure-only)
            ENABLE_AWS=false
            ENABLE_GCP=false
            ENABLE_AZURE=true
            shift
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