#!/bin/bash

# MLOps Platform Staging Environment Deployment Script

set -e

echo "🚀 Deploying MLOps Platform Staging Environment to Kubernetes..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="mlops-staging"
KUBECTL_TIMEOUT="300s"

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl is not installed. Please install kubectl first.${NC}"
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo -e "${YELLOW}⚠️  helm is not installed. Some features may not be available.${NC}"
fi

# Check cluster connectivity
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to Kubernetes cluster. Please check your kubeconfig.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Create namespace and basic configurations
echo "📦 Creating namespace and configurations..."
kubectl apply -f kubernetes/namespace.yaml

# Wait for namespace to be ready
kubectl wait --for=condition=Ready --timeout=$KUBECTL_TIMEOUT namespace/$NAMESPACE 2>/dev/null || true

# Deploy persistent volumes for stateful services
echo "💾 Setting up persistent storage..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: gp2
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: gp2
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: minio-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: gp2
EOF

# Deploy core infrastructure services
echo "🏗️  Deploying infrastructure services..."
kubectl apply -f kubernetes/deployments.yaml

# Deploy services and networking
echo "🌐 Setting up services and networking..."
kubectl apply -f kubernetes/services.yaml

# Deploy monitoring stack (if Helm is available)
if command -v helm &> /dev/null; then
    echo "📊 Setting up monitoring stack..."
    
    # Add Prometheus Helm repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus stack
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace $NAMESPACE \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.retention=30d \
        --set grafana.adminPassword=admin_password \
        --wait --timeout=$KUBECTL_TIMEOUT
    
    echo -e "${GREEN}✅ Monitoring stack deployed${NC}"
else
    echo -e "${YELLOW}⚠️  Skipping monitoring stack (Helm not available)${NC}"
fi

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."

deployments=("postgres" "redis" "model-server" "feature-store" "inference-engine")

for deployment in "${deployments[@]}"; do
    echo -n "  Waiting for $deployment... "
    if kubectl wait --for=condition=Available --timeout=$KUBECTL_TIMEOUT deployment/$deployment -n $NAMESPACE > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Ready${NC}"
    else
        echo -e "${YELLOW}⚠️  Not ready (may still be starting)${NC}"
    fi
done

# Port forward setup for testing
echo "🔧 Setting up port forwards for testing..."

# Function to setup port forward
setup_port_forward() {
    local service=$1
    local local_port=$2
    local remote_port=$3
    
    # Kill existing port forward if running
    pkill -f "kubectl port-forward.*$service" 2>/dev/null || true
    
    # Setup new port forward in background
    kubectl port-forward -n $NAMESPACE svc/$service $local_port:$remote_port > /dev/null 2>&1 &
    echo "  📡 $service: localhost:$local_port -> $remote_port"
}

# Setup port forwards (commented out by default - uncomment for testing)
# setup_port_forward "model-server-service" 8000 8000
# setup_port_forward "feature-store-service" 8001 8001  
# setup_port_forward "inference-engine-service" 8002 8002
# setup_port_forward "prometheus-kube-prometheus-prometheus" 9090 9090
# setup_port_forward "prometheus-grafana" 3000 80

# Display deployment status
echo ""
echo "📊 Deployment Status:"
kubectl get all -n $NAMESPACE

echo ""
echo "🎉 MLOps Platform Staging Environment deployed successfully!"
echo ""
echo -e "${BLUE}📚 Access Information:${NC}"
echo "  • Namespace: $NAMESPACE"
echo "  • Context: $(kubectl config current-context)"
echo "  • Cluster: $(kubectl config view --minify -o jsonpath='{.clusters[0].name}')"
echo ""
echo -e "${BLUE}🌐 Service Access (via port-forward):${NC}"
echo "  # Model Server API"
echo "  kubectl port-forward -n $NAMESPACE svc/model-server-service 8000:8000"
echo "  # Feature Store API"  
echo "  kubectl port-forward -n $NAMESPACE svc/feature-store-service 8001:8001"
echo "  # Inference Engine API"
echo "  kubectl port-forward -n $NAMESPACE svc/inference-engine-service 8002:8002"
echo ""
if command -v helm &> /dev/null; then
echo "  # Grafana Dashboard"
echo "  kubectl port-forward -n $NAMESPACE svc/prometheus-grafana 3000:80"
echo "  # Prometheus UI"
echo "  kubectl port-forward -n $NAMESPACE svc/prometheus-kube-prometheus-prometheus 9090:9090"
echo ""
fi
echo -e "${BLUE}🛠️  Management Commands:${NC}"
echo "  • View pods:          kubectl get pods -n $NAMESPACE"
echo "  • View logs:          kubectl logs -f deployment/[name] -n $NAMESPACE"
echo "  • Scale deployment:   kubectl scale deployment [name] --replicas=N -n $NAMESPACE"
echo "  • Delete environment: kubectl delete namespace $NAMESPACE"
echo ""
echo -e "${BLUE}🔍 Troubleshooting:${NC}"
echo "  • Pod status:         kubectl describe pod [pod-name] -n $NAMESPACE"
echo "  • Service status:     kubectl describe svc [service-name] -n $NAMESPACE"
echo "  • Events:             kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'"
echo ""
echo "✨ Staging environment is ready for testing!"