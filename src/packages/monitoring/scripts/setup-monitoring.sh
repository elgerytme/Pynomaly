#!/bin/bash

# Monitoring Infrastructure Setup Script
# Usage: ./setup-monitoring.sh [--namespace NAMESPACE]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITORING_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
NAMESPACE="monitoring"
FORCE_DEPLOY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --force)
            FORCE_DEPLOY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--namespace NAMESPACE] [--force]"
            echo "  --namespace    Monitoring namespace (default: monitoring)"
            echo "  --force        Force deployment even if namespace exists"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check if Helm is available (optional but recommended)
    if command -v helm &> /dev/null; then
        log_info "Helm is available - can be used for advanced monitoring setup"
    else
        log_warning "Helm not found - using kubectl for deployment"
    fi
    
    log_success "Prerequisites check passed"
}

setup_namespace() {
    log_info "Setting up monitoring namespace..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        if [ "$FORCE_DEPLOY" = false ]; then
            log_warning "Namespace $NAMESPACE already exists. Use --force to redeploy."
            read -p "Do you want to continue and update the existing deployment? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Setup cancelled"
                exit 0
            fi
        else
            log_info "Force deployment: updating existing namespace"
        fi
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Created namespace $NAMESPACE"
    fi
}

deploy_monitoring_stack() {
    log_info "Deploying monitoring stack..."
    
    # Apply the monitoring stack
    kubectl apply -f "$MONITORING_DIR/kubernetes/monitoring-stack.yaml" -n "$NAMESPACE" || {
        log_error "Failed to deploy monitoring stack"
        exit 1
    }
    
    log_success "Monitoring stack deployed"
}

wait_for_deployment() {
    log_info "Waiting for monitoring services to be ready..."
    
    # Wait for Prometheus
    log_info "Waiting for Prometheus..."
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n "$NAMESPACE" || {
        log_error "Prometheus deployment failed"
        kubectl get pods -n "$NAMESPACE" -l app=prometheus
        exit 1
    }
    
    # Wait for Grafana
    log_info "Waiting for Grafana..."
    kubectl wait --for=condition=available --timeout=300s deployment/grafana -n "$NAMESPACE" || {
        log_error "Grafana deployment failed"
        kubectl get pods -n "$NAMESPACE" -l app=grafana
        exit 1
    }
    
    log_success "All monitoring services are ready"
}

configure_service_monitoring() {
    log_info "Configuring service monitoring annotations..."
    
    # Add monitoring annotations to service deployments
    services=("hexagonal-staging" "hexagonal-production")
    apps=("data-quality" "machine-learning" "mlops" "anomaly-detection")
    
    for namespace in "${services[@]}"; do
        if kubectl get namespace "$namespace" &> /dev/null; then
            for app in "${apps[@]}"; do
                if kubectl get deployment "${app}-staging" -n "$namespace" &> /dev/null 2>&1 || \
                   kubectl get deployment "${app}-production" -n "$namespace" &> /dev/null 2>&1; then
                    
                    # Add Prometheus scraping annotations
                    kubectl patch deployment "${app}-staging" -n "$namespace" --type='merge' -p='{
                        "spec": {
                            "template": {
                                "metadata": {
                                    "annotations": {
                                        "prometheus.io/scrape": "true",
                                        "prometheus.io/port": "8080",
                                        "prometheus.io/path": "/metrics"
                                    }
                                }
                            }
                        }
                    }' 2>/dev/null || true
                    
                    kubectl patch deployment "${app}-production" -n "$namespace" --type='merge' -p='{
                        "spec": {
                            "template": {
                                "metadata": {
                                    "annotations": {
                                        "prometheus.io/scrape": "true",
                                        "prometheus.io/port": "8080",
                                        "prometheus.io/path": "/metrics"
                                    }
                                }
                            }
                        }
                    }' 2>/dev/null || true
                    
                    log_info "Added monitoring annotations to $app in $namespace"
                fi
            done
        fi
    done
    
    log_success "Service monitoring configured"
}

setup_alerting() {
    log_info "Setting up alerting rules..."
    
    # The alert rules are already included in the monitoring stack
    # Here we could add additional alerting configuration if needed
    
    log_info "Alert rules are configured in Prometheus"
    log_warning "Remember to configure AlertManager for actual alert delivery"
}

verify_monitoring() {
    log_info "Verifying monitoring setup..."
    
    # Check pod status
    log_info "Pod status:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # Check service status
    log_info "Service status:"
    kubectl get services -n "$NAMESPACE"
    
    # Test Prometheus connectivity
    log_info "Testing Prometheus connectivity..."
    kubectl port-forward -n "$NAMESPACE" service/prometheus 9090:9090 &
    PROMETHEUS_PF_PID=$!
    sleep 3
    
    if curl -f http://localhost:9090/-/healthy &> /dev/null; then
        log_success "Prometheus is healthy"
    else
        log_warning "Prometheus health check failed"
    fi
    
    kill $PROMETHEUS_PF_PID 2>/dev/null || true
    
    # Test Grafana connectivity
    log_info "Testing Grafana connectivity..."
    kubectl port-forward -n "$NAMESPACE" service/grafana 3000:3000 &
    GRAFANA_PF_PID=$!
    sleep 3
    
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        log_success "Grafana is healthy"
    else
        log_warning "Grafana health check failed"
    fi
    
    kill $GRAFANA_PF_PID 2>/dev/null || true
    sleep 1
}

display_access_info() {
    log_info "Monitoring Setup Complete!"
    echo
    echo "📊 MONITORING SERVICES DEPLOYED:"
    echo "  • Prometheus - Metrics collection and alerting"
    echo "  • Grafana - Visualization and dashboards"
    echo
    echo "🔗 ACCESS INFORMATION:"
    echo "  • Prometheus: http://monitoring.hexagonal-arch.local/prometheus"
    echo "  • Grafana: http://monitoring.hexagonal-arch.local/grafana"
    echo
    echo "🔧 LOCAL ACCESS (port forwarding):"
    echo "  • Prometheus: kubectl port-forward -n $NAMESPACE service/prometheus 9090:9090"
    echo "  • Grafana: kubectl port-forward -n $NAMESPACE service/grafana 3000:3000"
    echo
    echo "🔐 DEFAULT CREDENTIALS:"
    echo "  • Grafana: admin / admin123"
    echo
    echo "📈 MONITORING FEATURES:"
    echo "  ✓ Service health monitoring"
    echo "  ✓ Performance metrics (response time, throughput)"
    echo "  ✓ Resource usage (CPU, memory)"
    echo "  ✓ Business metrics (data quality, ML predictions)"
    echo "  ✓ Alert rules for critical issues"
    echo "  ✓ Kubernetes cluster monitoring"
    echo
    echo "🚨 NEXT STEPS:"
    echo "  1. Configure AlertManager for alert delivery"
    echo "  2. Customize Grafana dashboards for your needs"
    echo "  3. Set up log aggregation (ELK/EFK stack)"
    echo "  4. Configure distributed tracing (Jaeger/Zipkin)"
    echo "  5. Add custom business metrics to your services"
    echo
    echo "📚 USEFUL COMMANDS:"
    echo "  • View metrics: kubectl get --raw /api/v1/nodes/NODE_NAME/proxy/metrics"
    echo "  • Check alerts: kubectl logs -n $NAMESPACE deployment/prometheus"
    echo "  • Monitor pods: kubectl get pods -n $NAMESPACE -w"
}

cleanup_on_failure() {
    if [ $? -ne 0 ]; then
        log_error "Monitoring setup failed. Cleaning up..."
        kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    fi
}

main() {
    trap cleanup_on_failure ERR
    
    log_info "Starting monitoring infrastructure setup..."
    echo "Configuration:"
    echo "  Namespace: $NAMESPACE"
    echo "  Force deploy: $FORCE_DEPLOY"
    echo
    
    check_prerequisites
    setup_namespace
    deploy_monitoring_stack
    wait_for_deployment
    configure_service_monitoring
    setup_alerting
    verify_monitoring
    display_access_info
    
    log_success "Monitoring infrastructure setup completed successfully!"
}

main "$@"