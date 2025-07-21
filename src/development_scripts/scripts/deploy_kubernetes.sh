#!/bin/bash

# anomaly_detection Kubernetes Deployment Script
# This script deploys anomaly_detection to a Kubernetes cluster

set -euo pipefail

# Configuration
NAMESPACE="anomaly_detection-production"
DOCKER_IMAGE="anomaly_detection:latest"
DOCKER_REGISTRY="your-registry.com"
KUBECTL_CONTEXT="production-cluster"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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
}

# Check if kubectl is installed and configured
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        error "kubectl is not configured or cluster is not accessible."
        exit 1
    fi

    success "kubectl is configured and cluster is accessible"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    success "Docker is installed"
}

# Build Docker image
build_image() {
    log "Building Docker image..."

    # Build production image
    docker build -t ${DOCKER_IMAGE} -f Dockerfile.production .

    # Tag for registry
    docker tag ${DOCKER_IMAGE} ${DOCKER_REGISTRY}/${DOCKER_IMAGE}

    success "Docker image built successfully"
}

# Push Docker image to registry
push_image() {
    log "Pushing Docker image to registry..."

    # Login to registry (assumes credentials are configured)
    docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE}

    success "Docker image pushed to registry"
}

# Create namespace
create_namespace() {
    log "Creating namespace..."

    kubectl apply -f k8s/namespace.yaml

    success "Namespace created"
}

# Deploy secrets
deploy_secrets() {
    log "Deploying secrets..."

    # Check if secrets exist
    if kubectl get secret anomaly_detection-secrets -n ${NAMESPACE} &> /dev/null; then
        warning "Secrets already exist. Skipping secret creation."
        return
    fi

    # Create database password secret
    kubectl create secret generic anomaly_detection-secrets \
        --from-literal=SECRET_KEY="$(openssl rand -base64 32)" \
        --from-literal=JWT_SECRET_KEY="$(openssl rand -base64 32)" \
        --from-literal=POSTGRES_PASSWORD="$(openssl rand -base64 16)" \
        --from-literal=POSTGRES_USER="anomaly_detection_user" \
        --from-literal=REDIS_PASSWORD="$(openssl rand -base64 16)" \
        --from-literal=MONGODB_PASSWORD="$(openssl rand -base64 16)" \
        --from-literal=MONGODB_USER="anomaly_detection_user" \
        -n ${NAMESPACE}

    success "Secrets deployed"
}

# Deploy ConfigMaps
deploy_configmaps() {
    log "Deploying ConfigMaps..."

    kubectl apply -f k8s/configmap.yaml

    success "ConfigMaps deployed"
}

# Deploy databases
deploy_databases() {
    log "Deploying databases..."

    # Deploy PostgreSQL
    kubectl apply -f k8s/postgres.yaml

    # Deploy Redis
    kubectl apply -f k8s/redis.yaml

    # Deploy MongoDB
    kubectl apply -f k8s/mongodb.yaml

    success "Databases deployed"
}

# Wait for databases to be ready
wait_for_databases() {
    log "Waiting for databases to be ready..."

    # Wait for PostgreSQL
    kubectl wait --for=condition=ready pod -l component=postgres -n ${NAMESPACE} --timeout=300s
    success "PostgreSQL is ready"

    # Wait for Redis
    kubectl wait --for=condition=ready pod -l component=redis -n ${NAMESPACE} --timeout=300s
    success "Redis is ready"

    # Wait for MongoDB
    kubectl wait --for=condition=ready pod -l component=mongodb -n ${NAMESPACE} --timeout=300s
    success "MongoDB is ready"
}

# Deploy application
deploy_application() {
    log "Deploying application..."

    # Update image in deployment
    sed -i "s|anomaly_detection:latest|${DOCKER_REGISTRY}/${DOCKER_IMAGE}|g" k8s/anomaly_detection-app.yaml

    kubectl apply -f k8s/anomaly_detection-app.yaml

    success "Application deployed"
}

# Deploy monitoring
deploy_monitoring() {
    log "Deploying monitoring..."

    kubectl apply -f k8s/monitoring.yaml

    success "Monitoring deployed"
}

# Deploy ingress
deploy_ingress() {
    log "Deploying ingress..."

    kubectl apply -f k8s/nginx-ingress.yaml

    success "Ingress deployed"
}

# Wait for application to be ready
wait_for_application() {
    log "Waiting for application to be ready..."

    kubectl wait --for=condition=available deployment/anomaly_detection-app -n ${NAMESPACE} --timeout=300s

    success "Application is ready"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."

    # Get one of the application pods
    POD=$(kubectl get pods -l component=app -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')

    # Run migrations
    kubectl exec -n ${NAMESPACE} ${POD} -- python -m alembic upgrade head

    success "Database migrations completed"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."

    # Check application health
    kubectl get pods -l component=app -n ${NAMESPACE}

    # Check services
    kubectl get services -n ${NAMESPACE}

    # Check ingress
    kubectl get ingress -n ${NAMESPACE}

    # Test health endpoint
    EXTERNAL_IP=$(kubectl get service nginx-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -n "$EXTERNAL_IP" ]; then
        log "Testing health endpoint at $EXTERNAL_IP"
        curl -f http://$EXTERNAL_IP/health || warning "Health check failed"
    else
        warning "External IP not available yet"
    fi

    success "Deployment validation completed"
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo ""
    echo "📊 Pods:"
    kubectl get pods -n ${NAMESPACE}
    echo ""
    echo "🔧 Services:"
    kubectl get services -n ${NAMESPACE}
    echo ""
    echo "🌐 Ingress:"
    kubectl get ingress -n ${NAMESPACE}
    echo ""
    echo "💾 Storage:"
    kubectl get pvc -n ${NAMESPACE}
    echo ""
    echo "🔍 HPA:"
    kubectl get hpa -n ${NAMESPACE}
    echo ""

    # Get external IP
    EXTERNAL_IP=$(kubectl get service nginx-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -n "$EXTERNAL_IP" ]; then
        echo "🚀 Access your application at:"
        echo "   API: http://$EXTERNAL_IP/api/v1/"
        echo "   Health: http://$EXTERNAL_IP/health"
        echo "   Metrics: http://$EXTERNAL_IP/metrics"
    else
        echo "⏳ External IP is being assigned..."
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up failed deployment..."

    # Delete all resources
    kubectl delete all --all -n ${NAMESPACE}
    kubectl delete pvc --all -n ${NAMESPACE}
    kubectl delete configmap --all -n ${NAMESPACE}
    kubectl delete secret --all -n ${NAMESPACE}

    error "Deployment failed. Resources cleaned up."
}

# Parse command line arguments
COMMAND=${1:-deploy}

case $COMMAND in
    build)
        log "Building Docker image only..."
        check_docker
        build_image
        success "Build completed"
        ;;
    push)
        log "Pushing Docker image to registry..."
        check_docker
        push_image
        success "Push completed"
        ;;
    deploy)
        log "Starting full deployment..."
        trap cleanup ERR

        check_kubectl
        check_docker
        build_image
        push_image
        create_namespace
        deploy_secrets
        deploy_configmaps
        deploy_databases
        wait_for_databases
        deploy_application
        wait_for_application
        run_migrations
        deploy_monitoring
        deploy_ingress
        validate_deployment
        show_status

        success "Deployment completed successfully!"
        ;;
    status)
        log "Showing deployment status..."
        show_status
        ;;
    cleanup)
        log "Cleaning up deployment..."
        cleanup
        success "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {build|push|deploy|status|cleanup}"
        echo ""
        echo "Commands:"
        echo "  build   - Build Docker image"
        echo "  push    - Push Docker image to registry"
        echo "  deploy  - Full deployment (build, push, deploy)"
        echo "  status  - Show deployment status"
        echo "  cleanup - Clean up all resources"
        exit 1
        ;;
esac
