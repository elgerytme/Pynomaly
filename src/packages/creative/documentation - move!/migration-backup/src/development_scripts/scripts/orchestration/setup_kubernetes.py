#!/usr/bin/env python3
"""
Container Orchestration Setup Script for Pynomaly
This script sets up Kubernetes cluster configuration and deployment automation
"""

import os
import subprocess
import sys
from pathlib import Path


def check_prerequisites():
    """Check if required tools are installed."""
    required_tools = [
        ("kubectl", "kubectl version --client"),
        ("helm", "helm version"),
        ("docker", "docker --version"),
        ("kind", "kind version"),  # For local development
    ]

    missing_tools = []

    for tool, version_cmd in required_tools:
        try:
            result = subprocess.run(
                version_cmd.split(), capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"✅ {tool}: {result.stdout.strip().split()[0]}")
            else:
                missing_tools.append(tool)
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            missing_tools.append(tool)

    if missing_tools:
        print(f"❌ Missing required tools: {', '.join(missing_tools)}")
        print("\nInstallation instructions:")
        print("- kubectl: https://kubernetes.io/docs/tasks/tools/")
        print("- helm: https://helm.sh/docs/intro/install/")
        print("- docker: https://docs.docker.com/get-docker/")
        print("- kind: https://kind.sigs.k8s.io/docs/user/quick-start/")
        return False

    return True


def create_kind_cluster(cluster_name: str = "pynomaly-local"):
    """Create a local Kind cluster for development."""
    kind_config = f"""
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: {cluster_name}
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
  - containerPort: 30000
    hostPort: 30000
    protocol: TCP
  - containerPort: 30001
    hostPort: 30001
    protocol: TCP
  - containerPort: 30002
    hostPort: 30002
    protocol: TCP
- role: worker
  extraMounts:
  - hostPath: /var/lib/docker
    containerPath: /var/lib/docker
    readOnly: true
- role: worker
  extraMounts:
  - hostPath: /var/lib/docker
    containerPath: /var/lib/docker
    readOnly: true
"""

    config_dir = Path("/tmp/pynomaly-kind")
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "kind-config.yaml"

    with open(config_file, "w") as f:
        f.write(kind_config.strip())

    print(f"Creating Kind cluster '{cluster_name}'...")

    try:
        # Check if cluster already exists
        result = subprocess.run(
            ["kind", "get", "clusters"], capture_output=True, text=True
        )

        if cluster_name in result.stdout:
            print(f"Cluster '{cluster_name}' already exists")
            return True

        # Create the cluster
        subprocess.run(
            [
                "kind",
                "create",
                "cluster",
                "--config",
                str(config_file),
                "--wait",
                "60s",
            ],
            check=True,
        )

        print(f"✅ Kind cluster '{cluster_name}' created successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create Kind cluster: {e}")
        return False


def install_ingress_controller():
    """Install NGINX Ingress Controller."""
    print("Installing NGINX Ingress Controller...")

    try:
        # Apply NGINX Ingress Controller
        subprocess.run(
            [
                "kubectl",
                "apply",
                "-f",
                "https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml",
            ],
            check=True,
        )

        # Wait for ingress controller to be ready
        print("Waiting for ingress controller to be ready...")
        subprocess.run(
            [
                "kubectl",
                "wait",
                "--namespace",
                "ingress-nginx",
                "--for=condition=ready",
                "pod",
                "--selector=app.kubernetes.io/component=controller",
                "--timeout=90s",
            ],
            check=True,
        )

        print("✅ NGINX Ingress Controller installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install ingress controller: {e}")
        return False


def install_cert_manager():
    """Install cert-manager for TLS certificate management."""
    print("Installing cert-manager...")

    try:
        # Add cert-manager Helm repository
        subprocess.run(
            ["helm", "repo", "add", "jetstack", "https://charts.jetstack.io"],
            check=True,
        )

        subprocess.run(["helm", "repo", "update"], check=True)

        # Install cert-manager
        subprocess.run(
            [
                "helm",
                "install",
                "cert-manager",
                "jetstack/cert-manager",
                "--namespace",
                "cert-manager",
                "--create-namespace",
                "--version",
                "v1.13.0",
                "--set",
                "installCRDs=true",
            ],
            check=True,
        )

        # Wait for cert-manager to be ready
        print("Waiting for cert-manager to be ready...")
        subprocess.run(
            [
                "kubectl",
                "wait",
                "--namespace",
                "cert-manager",
                "--for=condition=ready",
                "pod",
                "--selector=app.kubernetes.io/instance=cert-manager",
                "--timeout=90s",
            ],
            check=True,
        )

        print("✅ cert-manager installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install cert-manager: {e}")
        return False


def create_cluster_issuer():
    """Create ClusterIssuer for Let's Encrypt certificates."""
    cluster_issuer = """
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-staging
spec:
  acme:
    server: https://acme-staging-v02.api.letsencrypt.org/directory
    email: admin@pynomaly.local
    privateKeySecretRef:
      name: letsencrypt-staging
    solvers:
    - http01:
        ingress:
          class: nginx
---
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@pynomaly.local
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
"""

    config_dir = Path("/tmp/pynomaly-k8s")
    config_dir.mkdir(exist_ok=True)
    issuer_file = config_dir / "cluster-issuer.yaml"

    with open(issuer_file, "w") as f:
        f.write(cluster_issuer.strip())

    try:
        subprocess.run(["kubectl", "apply", "-f", str(issuer_file)], check=True)

        print("✅ ClusterIssuer created successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create ClusterIssuer: {e}")
        return False


def install_storage_provisioner():
    """Install storage provisioner for dynamic volume provisioning."""
    print("Setting up storage provisioner...")

    storage_class = """
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: rancher.io/local-path
parameters:
  nodePath: /opt/local-path-provisioner
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete
"""

    config_dir = Path("/tmp/pynomaly-k8s")
    config_dir.mkdir(exist_ok=True)
    storage_file = config_dir / "storage-class.yaml"

    with open(storage_file, "w") as f:
        f.write(storage_class.strip())

    try:
        # Install local-path-provisioner
        subprocess.run(
            [
                "kubectl",
                "apply",
                "-f",
                "https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.24/deploy/local-path-storage.yaml",
            ],
            check=True,
        )

        # Apply custom storage class
        subprocess.run(["kubectl", "apply", "-f", str(storage_file)], check=True)

        print("✅ Storage provisioner installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install storage provisioner: {e}")
        return False


def install_metrics_server():
    """Install Metrics Server for resource metrics."""
    print("Installing Metrics Server...")

    try:
        subprocess.run(
            [
                "kubectl",
                "apply",
                "-f",
                "https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml",
            ],
            check=True,
        )

        # Patch metrics server for Kind
        subprocess.run(
            [
                "kubectl",
                "patch",
                "deployment",
                "metrics-server",
                "-n",
                "kube-system",
                "--type",
                "json",
                "-p",
                '[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]',
            ],
            check=True,
        )

        print("✅ Metrics Server installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Metrics Server: {e}")
        return False


def add_helm_repositories():
    """Add required Helm repositories."""
    repositories = [
        ("bitnami", "https://charts.bitnami.com/bitnami"),
        ("prometheus-community", "https://prometheus-community.github.io/helm-charts"),
        ("grafana", "https://grafana.github.io/helm-charts"),
        ("jaegertracing", "https://jaegertracing.github.io/helm-charts"),
        ("elastic", "https://helm.elastic.co"),
        ("jetstack", "https://charts.jetstack.io"),
    ]

    print("Adding Helm repositories...")

    for name, url in repositories:
        try:
            subprocess.run(["helm", "repo", "add", name, url], check=True)
            print(f"✅ Added repository: {name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to add repository {name}: {e}")

    try:
        subprocess.run(["helm", "repo", "update"], check=True)
        print("✅ Helm repositories updated")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to update repositories: {e}")
        return False


def create_namespace():
    """Create the pynomaly-production namespace."""
    namespace_yaml = """
apiVersion: v1
kind: Namespace
metadata:
  name: pynomaly-production
  labels:
    name: pynomaly-production
    environment: production
    app.kubernetes.io/name: pynomaly
    app.kubernetes.io/instance: production
"""

    config_dir = Path("/tmp/pynomaly-k8s")
    config_dir.mkdir(exist_ok=True)
    namespace_file = config_dir / "namespace.yaml"

    with open(namespace_file, "w") as f:
        f.write(namespace_yaml.strip())

    try:
        subprocess.run(["kubectl", "apply", "-f", str(namespace_file)], check=True)

        print("✅ Namespace 'pynomaly-production' created")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create namespace: {e}")
        return False


def create_deployment_scripts():
    """Create deployment automation scripts."""
    scripts_dir = Path("/opt/pynomaly/scripts/orchestration")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Deployment script
    deploy_script = """#!/bin/bash
# Pynomaly Kubernetes Deployment Script

set -e

NAMESPACE="${NAMESPACE:-pynomaly-production}"
HELM_RELEASE="${HELM_RELEASE:-pynomaly}"
VALUES_FILE="${VALUES_FILE:-values.production.yaml}"
CHART_PATH="${CHART_PATH:-./deploy/helm/pynomaly-complete}"

echo "🚀 Deploying Pynomaly to Kubernetes..."
echo "Namespace: $NAMESPACE"
echo "Release: $HELM_RELEASE"
echo "Values: $VALUES_FILE"
echo "Chart: $CHART_PATH"

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
    echo "Creating namespace $NAMESPACE..."
    kubectl create namespace "$NAMESPACE"
fi

# Check if Helm release exists
if helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE"; then
    echo "📦 Upgrading existing release..."
    helm upgrade "$HELM_RELEASE" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --values "$VALUES_FILE" \
        --wait \
        --timeout 10m \
        --atomic
else
    echo "📦 Installing new release..."
    helm install "$HELM_RELEASE" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --values "$VALUES_FILE" \
        --wait \
        --timeout 10m \
        --create-namespace
fi

echo "✅ Deployment completed successfully!"

# Show deployment status
echo ""
echo "📊 Deployment Status:"
kubectl get pods -n "$NAMESPACE" -o wide

echo ""
echo "🌐 Services:"
kubectl get services -n "$NAMESPACE"

echo ""
echo "🔗 Ingress:"
kubectl get ingress -n "$NAMESPACE"

# Get ingress URLs
if kubectl get ingress -n "$NAMESPACE" >/dev/null 2>&1; then
    echo ""
    echo "🌍 Access URLs:"
    kubectl get ingress -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.spec.rules[*].host}{" -> "}{.spec.rules[*].http.paths[*].backend.service.name}{"\\n"}{end}'
fi
"""

    deploy_file = scripts_dir / "deploy.sh"
    with open(deploy_file, "w") as f:
        f.write(deploy_script.strip())
    os.chmod(deploy_file, 0o755)

    # Monitoring script
    monitor_script = """#!/bin/bash
# Pynomaly Kubernetes Monitoring Script

NAMESPACE="${NAMESPACE:-pynomaly-production}"

echo "📊 Pynomaly Cluster Monitoring"
echo "Namespace: $NAMESPACE"
echo "=" * 50

# Check cluster status
echo ""
echo "🏠 Cluster Status:"
kubectl cluster-info

# Check node status
echo ""
echo "🖥️  Node Status:"
kubectl get nodes -o wide

# Check pod status
echo ""
echo "🐳 Pod Status:"
kubectl get pods -n "$NAMESPACE" -o wide

# Check service status
echo ""
echo "🌐 Service Status:"
kubectl get services -n "$NAMESPACE"

# Check persistent volumes
echo ""
echo "💾 Storage Status:"
kubectl get pv,pvc -n "$NAMESPACE"

# Check ingress
echo ""
echo "🔗 Ingress Status:"
kubectl get ingress -n "$NAMESPACE"

# Check resource usage
echo ""
echo "📈 Resource Usage:"
kubectl top nodes 2>/dev/null || echo "Metrics server not available"
kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Pod metrics not available"

# Check recent events
echo ""
echo "📅 Recent Events:"
kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10

# Check application logs (sample)
echo ""
echo "📋 Sample Application Logs:"
POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=pynomaly-api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$POD_NAME" ]; then
    echo "Logs from $POD_NAME:"
    kubectl logs -n "$NAMESPACE" "$POD_NAME" --tail=10
else
    echo "No application pods found"
fi
"""

    monitor_file = scripts_dir / "monitor.sh"
    with open(monitor_file, "w") as f:
        f.write(monitor_script.strip())
    os.chmod(monitor_file, 0o755)

    # Cleanup script
    cleanup_script = """#!/bin/bash
# Pynomaly Kubernetes Cleanup Script

NAMESPACE="${NAMESPACE:-pynomaly-production}"
HELM_RELEASE="${HELM_RELEASE:-pynomaly}"

echo "🧹 Cleaning up Pynomaly deployment..."
echo "Namespace: $NAMESPACE"
echo "Release: $HELM_RELEASE"

read -p "Are you sure you want to delete the deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Uninstall Helm release
if helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE"; then
    echo "📦 Uninstalling Helm release..."
    helm uninstall "$HELM_RELEASE" -n "$NAMESPACE" --wait
else
    echo "No Helm release found"
fi

# Delete persistent volumes (optional)
read -p "Delete persistent volumes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "💾 Deleting persistent volumes..."
    kubectl delete pvc --all -n "$NAMESPACE"
fi

# Delete namespace
read -p "Delete namespace $NAMESPACE? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Deleting namespace..."
    kubectl delete namespace "$NAMESPACE"
fi

echo "✅ Cleanup completed!"
"""

    cleanup_file = scripts_dir / "cleanup.sh"
    with open(cleanup_file, "w") as f:
        f.write(cleanup_script.strip())
    os.chmod(cleanup_file, 0o755)

    # Scaling script
    scaling_script = """#!/bin/bash
# Pynomaly Kubernetes Scaling Script

NAMESPACE="${NAMESPACE:-pynomaly-production}"

usage() {
    echo "Usage: $0 <component> <replicas>"
    echo "Components: api, worker-training, worker-drift"
    echo "Example: $0 api 5"
    exit 1
}

if [ $# -ne 2 ]; then
    usage
fi

COMPONENT="$1"
REPLICAS="$2"

case $COMPONENT in
    "api")
        DEPLOYMENT="pynomaly-api"
        ;;
    "worker-training")
        DEPLOYMENT="pynomaly-worker-training"
        ;;
    "worker-drift")
        DEPLOYMENT="pynomaly-worker-drift"
        ;;
    *)
        echo "Unknown component: $COMPONENT"
        usage
        ;;
esac

echo "📊 Scaling $DEPLOYMENT to $REPLICAS replicas..."

kubectl scale deployment "$DEPLOYMENT" -n "$NAMESPACE" --replicas="$REPLICAS"

echo "Waiting for deployment to be ready..."
kubectl rollout status deployment "$DEPLOYMENT" -n "$NAMESPACE" --timeout=300s

echo "✅ Scaling completed!"
echo ""
echo "📊 Current status:"
kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" -o wide
"""

    scaling_file = scripts_dir / "scale.sh"
    with open(scaling_file, "w") as f:
        f.write(scaling_script.strip())
    os.chmod(scaling_file, 0o755)

    print(f"✅ Deployment scripts created in {scripts_dir}")
    return True


def create_development_values():
    """Create development values file for Helm."""
    dev_values = """
# Development values for Pynomaly Helm chart
# Optimized for local development with reduced resources

global:
  storageClass: "standard"

app:
  replicaCount: 1
  resources:
    requests:
      memory: "256Mi"
      cpu: "100m"
    limits:
      memory: "1Gi"
      cpu: "500m"

workers:
  training:
    replicaCount: 1
    resources:
      requests:
        memory: "512Mi"
        cpu: "250m"
      limits:
        memory: "2Gi"
        cpu: "1000m"
  drift:
    replicaCount: 1
    resources:
      requests:
        memory: "256Mi"
        cpu: "100m"
      limits:
        memory: "1Gi"
        cpu: "500m"

autoscaling:
  enabled: false

postgresql:
  auth:
    postgresPassword: "postgres_dev"
    password: "pynomaly_dev"
  primary:
    persistence:
      size: 5Gi
    resources:
      requests:
        memory: "256Mi"
        cpu: "100m"
      limits:
        memory: "512Mi"
        cpu: "250m"

redis:
  auth:
    password: "redis_dev"
  master:
    persistence:
      size: 1Gi
    resources:
      requests:
        memory: "128Mi"
        cpu: "50m"
      limits:
        memory: "256Mi"
        cpu: "100m"

monitoring:
  prometheus:
    server:
      persistentVolume:
        size: 10Gi
      resources:
        requests:
          memory: "512Mi"
          cpu: "100m"
        limits:
          memory: "1Gi"
          cpu: "500m"
  grafana:
    persistence:
      size: 2Gi
    resources:
      requests:
        memory: "128Mi"
        cpu: "50m"
      limits:
        memory: "256Mi"
        cpu: "100m"
    adminPassword: "admin"

tracing:
  jaeger:
    storage:
      badger:
        persistence:
          size: 5Gi

logging:
  elasticsearch:
    persistence:
      size: 10Gi
    resources:
      requests:
        memory: "512Mi"
        cpu: "250m"
      limits:
        memory: "1Gi"
        cpu: "500m"
  kibana:
    persistence:
      size: 1Gi

ingress:
  hosts:
    - host: api.pynomaly.localhost
      paths:
        - path: /
          pathType: Prefix
          service:
            name: pynomaly-api
            port: 8000
    - host: grafana.pynomaly.localhost
      paths:
        - path: /
          pathType: Prefix
          service:
            name: grafana
            port: 3000
    - host: jaeger.pynomaly.localhost
      paths:
        - path: /
          pathType: Prefix
          service:
            name: jaeger-query
            port: 16686
"""

    values_dir = Path("/mnt/c/Users/andre/Pynomaly/deploy/helm/pynomaly-complete")
    values_file = values_dir / "values.development.yaml"

    with open(values_file, "w") as f:
        f.write(dev_values.strip())

    print(f"✅ Development values file created: {values_file}")
    return True


def main():
    """Main orchestration setup function."""
    print("Setting up Pynomaly Container Orchestration...")
    print("=" * 60)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Setup options
    setup_local = (
        input("Setup local Kind cluster for development? (y/N): ").lower() == "y"
    )

    try:
        if setup_local:
            # Local development setup
            if not create_kind_cluster():
                sys.exit(1)

            if not install_ingress_controller():
                sys.exit(1)

            if not install_metrics_server():
                sys.exit(1)

            if not install_storage_provisioner():
                sys.exit(1)

        # Common setup for both local and remote clusters
        if not install_cert_manager():
            sys.exit(1)

        if not create_cluster_issuer():
            sys.exit(1)

        if not add_helm_repositories():
            sys.exit(1)

        if not create_namespace():
            sys.exit(1)

        if not create_deployment_scripts():
            sys.exit(1)

        if not create_development_values():
            sys.exit(1)

        print("\n" + "=" * 60)
        print("✅ Container orchestration setup completed successfully!")
        print("\nNext steps:")
        print("1. Review and customize Helm values files:")
        print("   - deploy/helm/pynomaly-complete/values.yaml (production)")
        print(
            "   - deploy/helm/pynomaly-complete/values.development.yaml (development)"
        )
        print("\n2. Deploy Pynomaly:")
        print("   cd /mnt/c/Users/andre/Pynomaly")
        print("   ./scripts/orchestration/deploy.sh")
        print("\n3. Monitor deployment:")
        print("   ./scripts/orchestration/monitor.sh")
        print("\n4. Scale components:")
        print("   ./scripts/orchestration/scale.sh api 5")
        print("\n5. Access services:")
        if setup_local:
            print("   - API: http://api.pynomaly.localhost")
            print("   - Grafana: http://grafana.pynomaly.localhost")
            print("   - Jaeger: http://jaeger.pynomaly.localhost")
        else:
            print("   - Configure DNS to point to your cluster ingress")

        print("\nKey files created:")
        print("- Kubernetes manifests: deploy/kubernetes/production-complete.yaml")
        print("- Helm chart: deploy/helm/pynomaly-complete/")
        print("- Deployment scripts: /opt/pynomaly/scripts/orchestration/")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
