#!/bin/bash

# Production Deployment Script for MLOps Platform
# This script automates the complete deployment process

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/infrastructure/production/terraform"
KUBERNETES_DIR="$PROJECT_ROOT/infrastructure/production/kubernetes"
ENVIRONMENT="${ENVIRONMENT:-production}"
AWS_REGION="${AWS_REGION:-us-west-2}"

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
    
    local missing_tools=()
    
    # Check required tools
    for tool in terraform kubectl aws helm; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install all required tools before running this script"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        exit 1
    fi
    
    # Check Terraform version
    local terraform_version
    terraform_version=$(terraform version -json | jq -r '.terraform_version')
    log_info "Using Terraform version: $terraform_version"
    
    # Check Kubernetes version
    local kubectl_version
    kubectl_version=$(kubectl version --client -o json | jq -r '.clientVersion.gitVersion')
    log_info "Using kubectl version: $kubectl_version"
    
    log_success "All prerequisites satisfied"
}

# Initialize Terraform backend
init_terraform() {
    log_info "Initializing Terraform..."
    
    cd "$TERRAFORM_DIR"
    
    # Create terraform.tfvars if it doesn't exist
    if [ ! -f "terraform.tfvars" ]; then
        log_warning "terraform.tfvars not found, creating template..."
        cat > terraform.tfvars << EOF
# Production Environment Configuration
environment = "production"
aws_region = "$AWS_REGION"

# VPC Configuration
vpc_cidr = "10.0.0.0/16"
private_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
public_subnet_cidrs = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

# Database Configuration
postgres_password = "CHANGE_ME_SECURE_PASSWORD"

# Node Group Configuration
cpu_node_desired_size = 5
memory_node_desired_size = 3
gpu_node_desired_size = 2

# Security Configuration
allowed_cidr_blocks = ["0.0.0.0/0"]  # RESTRICT THIS IN PRODUCTION

# Add more configuration as needed
EOF
        log_warning "Please edit terraform.tfvars with your specific configuration before continuing"
        log_warning "Especially update the postgres_password and allowed_cidr_blocks"
        read -p "Press enter to continue after updating terraform.tfvars..."
    fi
    
    # Initialize Terraform
    terraform init
    
    log_success "Terraform initialized"
}

# Plan Terraform deployment
plan_terraform() {
    log_info "Planning Terraform deployment..."
    
    cd "$TERRAFORM_DIR"
    
    # Run terraform plan
    terraform plan -out=tfplan
    
    log_success "Terraform plan completed"
    log_info "Review the plan above before proceeding"
    read -p "Do you want to continue with the deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deployment cancelled by user"
        exit 0
    fi
}

# Apply Terraform configuration
apply_terraform() {
    log_info "Applying Terraform configuration..."
    
    cd "$TERRAFORM_DIR"
    
    # Apply the plan
    terraform apply tfplan
    
    # Output important values
    log_info "Extracting deployment outputs..."
    
    # Get cluster name for kubectl configuration
    CLUSTER_NAME=$(terraform output -raw cluster_name)
    export CLUSTER_NAME
    
    # Configure kubectl
    aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"
    
    log_success "Terraform deployment completed"
    log_info "EKS cluster: $CLUSTER_NAME"
}

# Wait for cluster to be ready
wait_for_cluster() {
    log_info "Waiting for EKS cluster to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if kubectl get nodes &> /dev/null; then
            log_success "EKS cluster is ready"
            break
        else
            log_info "Attempt $attempt/$max_attempts - Waiting for cluster..."
            sleep 30
            ((attempt++))
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "Cluster failed to become ready within expected time"
        exit 1
    fi
    
    # Show node status
    log_info "Current node status:"
    kubectl get nodes -o wide
}

# Install cluster addons
install_addons() {
    log_info "Installing cluster addons..."
    
    # Install AWS Load Balancer Controller
    log_info "Installing AWS Load Balancer Controller..."
    helm repo add eks https://aws.github.io/eks-charts
    helm repo update
    
    # Create IAM role for AWS Load Balancer Controller (simplified - should use IRSA in production)
    helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName="$CLUSTER_NAME" \
        --set serviceAccount.create=true \
        --set serviceAccount.name=aws-load-balancer-controller \
        --wait
    
    # Install Metrics Server
    log_info "Installing Metrics Server..."
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    
    # Install Cluster Autoscaler
    log_info "Installing Cluster Autoscaler..."
    helm repo add autoscaler https://kubernetes.github.io/autoscaler
    helm repo update
    
    helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler \
        -n kube-system \
        --set autoDiscovery.clusterName="$CLUSTER_NAME" \
        --set awsRegion="$AWS_REGION" \
        --set rbac.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/cluster-autoscaler" \
        --wait
    
    # Install NVIDIA GPU Operator (if GPU nodes are present)
    if kubectl get nodes -l node-type=gpu-enabled &> /dev/null; then
        log_info "Installing NVIDIA GPU Operator..."
        helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
        helm repo update
        
        helm upgrade --install gpu-operator nvidia/gpu-operator \
            -n gpu-operator \
            --create-namespace \
            --wait
    fi
    
    log_success "Cluster addons installed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Install Prometheus using kube-prometheus-stack
    log_info "Installing Prometheus and Grafana..."
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        -n monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=7d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=10Gi \
        --set alertmanager.persistence.enabled=true \
        --set alertmanager.persistence.size=5Gi \
        --wait
    
    # Apply custom monitoring configuration
    if [ -f "$PROJECT_ROOT/infrastructure/production/monitoring/advanced-metrics.yml" ]; then
        log_info "Applying custom Prometheus configuration..."
        kubectl apply -f "$PROJECT_ROOT/infrastructure/production/monitoring/advanced-metrics.yml"
    fi
    
    if [ -f "$PROJECT_ROOT/infrastructure/production/monitoring/production-alert-rules.yml" ]; then
        log_info "Applying custom alert rules..."
        kubectl apply -f "$PROJECT_ROOT/infrastructure/production/monitoring/production-alert-rules.yml"
    fi
    
    log_success "Monitoring stack deployed"
}

# Update secrets with actual values
update_secrets() {
    log_info "Updating Kubernetes secrets with actual infrastructure values..."
    
    cd "$TERRAFORM_DIR"
    
    # Get database endpoint and credentials
    DB_HOST=$(terraform output -raw postgres_endpoint)
    DB_USERNAME=$(terraform output -raw postgres_username)
    
    # Get Redis endpoint
    REDIS_ENDPOINT=$(terraform output -raw redis_endpoint)
    REDIS_PORT=$(terraform output -raw redis_port)
    
    # Get S3 bucket names
    S3_DATA_BUCKET=$(terraform output -json s3_buckets | jq -r '.data_bucket.id')
    S3_MODELS_BUCKET=$(terraform output -json s3_buckets | jq -r '.models_bucket.id')
    S3_ARTIFACTS_BUCKET=$(terraform output -json s3_buckets | jq -r '.artifacts_bucket.id')
    
    # Update ConfigMap
    kubectl patch configmap app-config -n mlops-production --patch "{\
        \"data\": {\
            \"s3-data-bucket\": \"$S3_DATA_BUCKET\",\
            \"s3-models-bucket\": \"$S3_MODELS_BUCKET\",\
            \"s3-artifacts-bucket\": \"$S3_ARTIFACTS_BUCKET\"\
        }\
    }"
    
    # Update database secret (password should be set separately for security)
    kubectl patch secret database-credentials -n mlops-production --patch "{\
        \"data\": {\
            \"host\": \"$(echo -n "$DB_HOST" | base64 -w 0)\",\
            \"username\": \"$(echo -n "$DB_USERNAME" | base64 -w 0)\"\
        }\
    }"
    
    # Update Redis secret
    REDIS_URL="redis://$REDIS_ENDPOINT:$REDIS_PORT"
    kubectl patch secret redis-credentials -n mlops-production --patch "{\
        \"data\": {\
            \"url\": \"$(echo -n "$REDIS_URL" | base64 -w 0)\"\
        }\
    }"
    
    log_success "Secrets updated with infrastructure values"
    log_warning "Remember to update the database password secret manually for security"
}

# Deploy MLOps platform
deploy_platform() {
    log_info "Deploying MLOps platform..."
    
    # Apply Kubernetes manifests
    log_info "Applying Kubernetes deployments..."
    kubectl apply -f "$KUBERNETES_DIR/deployments.yaml"
    
    log_info "Applying Kubernetes services..."
    kubectl apply -f "$KUBERNETES_DIR/services.yaml"
    
    log_info "Applying HPA configurations..."
    kubectl apply -f "$KUBERNETES_DIR/hpa.yaml"
    
    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."
    
    local deployments=(
        "model-server"
        "feature-store"
        "inference-engine"
        "ab-testing-service"
        "model-governance"
        "automl-service"
        "explainability-service"
        "api-gateway"
    )
    
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for $deployment to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/"$deployment" -n mlops-production
    done
    
    log_success "MLOps platform deployed successfully"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    log_info "Pod status:"
    kubectl get pods -n mlops-production -o wide
    
    # Check service status
    log_info "Service status:"
    kubectl get services -n mlops-production
    
    # Check HPA status
    log_info "HPA status:"
    kubectl get hpa -n mlops-production
    
    # Get external endpoints
    log_info "Getting external endpoints..."
    
    # Get load balancer endpoint
    local lb_hostname
    lb_hostname=$(kubectl get service api-gateway-external -n mlops-production -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "pending")
    
    if [ "$lb_hostname" != "pending" ] && [ -n "$lb_hostname" ]; then
        log_success "API Gateway available at: http://$lb_hostname"
        log_info "Testing API Gateway health endpoint..."
        
        # Wait a bit for the load balancer to be fully ready
        sleep 30
        
        if curl -f -s "http://$lb_hostname/health" > /dev/null; then
            log_success "API Gateway health check passed"
        else
            log_warning "API Gateway health check failed - may need more time to initialize"
        fi
    else
        log_warning "Load balancer endpoint not yet available"
    fi
    
    # Get Grafana endpoint
    local grafana_port
    grafana_port=$(kubectl get service kube-prometheus-stack-grafana -n monitoring -o jsonpath='{.spec.ports[0].port}')
    log_info "Grafana available via port-forward: kubectl port-forward svc/kube-prometheus-stack-grafana -n monitoring $grafana_port:80"
    log_info "Default Grafana credentials: admin / prom-operator"
    
    log_success "Deployment verification completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    cd "$TERRAFORM_DIR"
    rm -f tfplan
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting MLOps Platform Production Deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "AWS Region: $AWS_REGION"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    init_terraform
    plan_terraform
    apply_terraform
    wait_for_cluster
    install_addons
    deploy_monitoring
    update_secrets
    deploy_platform
    verify_deployment
    
    log_success "🎉 MLOps Platform Production Deployment Completed Successfully!"
    log_info "Next steps:"
    log_info "1. Update database password secret manually"
    log_info "2. Configure DNS for your domain (if applicable)"
    log_info "3. Set up SSL certificates"
    log_info "4. Configure backup and disaster recovery"
    log_info "5. Set up CI/CD pipelines"
    
    # Output important information
    cd "$TERRAFORM_DIR"
    echo ""
    log_info "Important deployment information:"
    echo "Cluster Name: $(terraform output -raw cluster_name)"
    echo "VPC ID: $(terraform output -raw vpc_id)"
    echo "Database Endpoint: $(terraform output -raw postgres_endpoint)"
    echo "Redis Endpoint: $(terraform output -raw redis_endpoint)"
    echo ""
    log_info "Run 'kubectl get all -n mlops-production' to see all deployed resources"
}

# Run main function
main "$@"