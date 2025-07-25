#!/bin/bash

# Deploy hexagonal architecture to Kubernetes
# Usage: ./deploy-kubernetes.sh [environment] [action]
# Example: ./deploy-kubernetes.sh production apply

set -e

ENVIRONMENT="${1:-development}"
ACTION="${2:-apply}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUSTOMIZE_PATH="${SCRIPT_DIR}/../kubernetes/overlays/${ENVIRONMENT}"

# Validate environment
case "${ENVIRONMENT}" in
    development|staging|production)
        echo "🚀 Deploying to ${ENVIRONMENT} environment"
        ;;
    *)
        echo "❌ Invalid environment: ${ENVIRONMENT}"
        echo "Valid environments: development, staging, production"
        exit 1
        ;;
esac

# Validate action
case "${ACTION}" in
    apply|delete|diff|dry-run)
        echo "📋 Action: ${ACTION}"
        ;;
    *)
        echo "❌ Invalid action: ${ACTION}"
        echo "Valid actions: apply, delete, diff, dry-run"
        exit 1
        ;;
esac

# Check if kustomization file exists
if [[ ! -f "${KUSTOMIZE_PATH}/kustomization.yaml" ]]; then
    echo "❌ Kustomization file not found: ${KUSTOMIZE_PATH}/kustomization.yaml"
    exit 1
fi

echo ""
echo "🔧 Configuration:"
echo "  Environment: ${ENVIRONMENT}"
echo "  Action: ${ACTION}"
echo "  Kustomize Path: ${KUSTOMIZE_PATH}"
echo ""

# Function to check kubectl connection
check_kubectl() {
    echo "🔍 Checking kubectl connection..."
    if ! kubectl cluster-info >/dev/null 2>&1; then
        echo "❌ Cannot connect to Kubernetes cluster"
        echo "Please ensure kubectl is configured and you have access to the cluster"
        exit 1
    fi
    echo "✅ Connected to Kubernetes cluster"
    
    # Show current context
    local current_context
    current_context=$(kubectl config current-context)
    echo "📍 Current context: ${current_context}"
    
    # Confirm production deployment
    if [[ "${ENVIRONMENT}" == "production" ]] && [[ "${ACTION}" == "apply" ]]; then
        echo ""
        echo "⚠️  WARNING: You are about to deploy to PRODUCTION!"
        echo "Current context: ${current_context}"
        read -p "Are you sure you want to continue? (yes/no): " confirm
        if [[ "${confirm}" != "yes" ]]; then
            echo "❌ Deployment cancelled"
            exit 1
        fi
    fi
}

# Function to perform the deployment action
perform_action() {
    echo ""
    echo "🚀 Executing ${ACTION}..."
    
    case "${ACTION}" in
        apply)
            kubectl apply -k "${KUSTOMIZE_PATH}"
            echo ""
            echo "✅ Deployment applied successfully!"
            
            # Show deployment status
            echo ""
            echo "📊 Deployment Status:"
            kubectl get pods -n "hexagonal-${ENVIRONMENT}" -l part-of=hexagonal-architecture
            ;;
        delete)
            echo "⚠️  This will delete all resources in the ${ENVIRONMENT} environment"
            read -p "Are you sure? (yes/no): " confirm
            if [[ "${confirm}" == "yes" ]]; then
                kubectl delete -k "${KUSTOMIZE_PATH}"
                echo "✅ Resources deleted successfully!"
            else
                echo "❌ Deletion cancelled"
            fi
            ;;
        diff)
            kubectl diff -k "${KUSTOMIZE_PATH}" || true
            ;;
        dry-run)
            kubectl apply -k "${KUSTOMIZE_PATH}" --dry-run=client -o yaml
            ;;
    esac
}

# Function to show post-deployment information
show_info() {
    if [[ "${ACTION}" == "apply" ]]; then
        echo ""
        echo "🔗 Access Information:"
        echo ""
        
        # Get service information
        local namespace="hexagonal-${ENVIRONMENT}"
        
        echo "📋 Services:"
        kubectl get services -n "${namespace}" -l part-of=hexagonal-architecture
        
        echo ""
        echo "📊 Pods:"
        kubectl get pods -n "${namespace}" -l part-of=hexagonal-architecture
        
        echo ""
        echo "💾 Storage:"
        kubectl get pvc -n "${namespace}"
        
        if [[ "${ENVIRONMENT}" == "development" ]]; then
            echo ""
            echo "🛠️  Development Tips:"
            echo "  - View logs: kubectl logs -f deployment/dev-data-quality-service -n ${namespace}"
            echo "  - Port forward: kubectl port-forward service/dev-data-quality-service 8000:8000 -n ${namespace}"
            echo "  - Scale service: kubectl scale deployment dev-data-quality-service --replicas=2 -n ${namespace}"
        fi
        
        if [[ "${ENVIRONMENT}" == "production" ]]; then
            echo ""
            echo "🏭 Production Monitoring:"
            echo "  - Monitor: kubectl get pods -n ${namespace} --watch"
            echo "  - Health: kubectl get pods -n ${namespace} -l part-of=hexagonal-architecture"
            echo "  - Logs: kubectl logs -l app=data-quality -n ${namespace} --tail=100"
        fi
    fi
}

# Main execution
main() {
    echo "🏗️  Hexagonal Architecture Kubernetes Deployment"
    echo "================================================"
    
    check_kubectl
    perform_action
    show_info
    
    echo ""
    echo "🎉 Deployment operation completed!"
}

main