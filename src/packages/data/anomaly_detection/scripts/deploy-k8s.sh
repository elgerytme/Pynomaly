#!/bin/bash
# Kubernetes deployment script for Anomaly Detection Service

set -e

# Configuration
ENVIRONMENT=${1:-development}
VERSION=${2:-latest}
NAMESPACE="anomaly-detection"

if [ "$ENVIRONMENT" = "development" ]; then
    NAMESPACE="anomaly-detection-dev"
fi

echo "☸️  Deploying Anomaly Detection Service to Kubernetes"
echo "=================================================="
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Namespace: $NAMESPACE"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed or not in PATH"
    exit 1
fi

# Check if kustomize is available
if ! command -v kustomize &> /dev/null; then
    echo "❌ kustomize is not installed or not in PATH"
    echo "   Install with: curl -s \"https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh\" | bash"
    exit 1
fi

# Check cluster connection
echo "🔗 Checking cluster connection..."
kubectl cluster-info --request-timeout=5s > /dev/null || {
    echo "❌ Cannot connect to Kubernetes cluster"
    exit 1
}

echo "✅ Connected to cluster: $(kubectl config current-context)"

# Create namespace if it doesn't exist
echo ""
echo "📦 Ensuring namespace exists..."
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Apply configurations based on environment
OVERLAY_PATH="k8s/overlays/$ENVIRONMENT"

if [ ! -d "$OVERLAY_PATH" ]; then
    echo "❌ Environment overlay not found: $OVERLAY_PATH"
    exit 1
fi

echo ""
echo "🚀 Deploying to $ENVIRONMENT environment..."

# Build and apply manifests
kustomize build "$OVERLAY_PATH" | kubectl apply -f -

echo ""
echo "⏳ Waiting for deployments to be ready..."

# Wait for deployments
kubectl wait --for=condition=available --timeout=600s deployment/anomaly-detection-api -n "$NAMESPACE" || {
    echo "❌ API deployment failed to become ready"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api
    kubectl describe deployment/anomaly-detection-api -n "$NAMESPACE"
    exit 1
}

kubectl wait --for=condition=available --timeout=600s deployment/anomaly-detection-worker -n "$NAMESPACE" || {
    echo "❌ Worker deployment failed to become ready"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=worker
    kubectl describe deployment/anomaly-detection-worker -n "$NAMESPACE"
    exit 1
}

echo ""
echo "✅ Deployment successful!"

# Show deployment status
echo ""
echo "📊 Deployment Status:"
kubectl get deployments -n "$NAMESPACE" -l app.kubernetes.io/name=anomaly-detection

echo ""
echo "🏃 Running Pods:"
kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=anomaly-detection

echo ""
echo "🌐 Services:"
kubectl get services -n "$NAMESPACE" -l app.kubernetes.io/name=anomaly-detection

# Show ingress if available
if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
    echo ""
    echo "🚪 Ingress:"
    kubectl get ingress -n "$NAMESPACE"
fi

# Health check
echo ""
echo "❤️  Performing health check..."
API_POD=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}')

if [ -n "$API_POD" ]; then
    kubectl exec -n "$NAMESPACE" "$API_POD" -- curl -f http://localhost:8000/health && {
        echo "✅ Health check passed!"
    } || {
        echo "⚠️  Health check failed"
    }
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📝 Useful commands:"
echo "   Monitor pods: kubectl get pods -n $NAMESPACE -w"
echo "   View logs: kubectl logs -f deployment/anomaly-detection-api -n $NAMESPACE"
echo "   Port forward: kubectl port-forward svc/anomaly-detection-api 8000:8000 -n $NAMESPACE"
echo "   Delete deployment: kubectl delete -k $OVERLAY_PATH"