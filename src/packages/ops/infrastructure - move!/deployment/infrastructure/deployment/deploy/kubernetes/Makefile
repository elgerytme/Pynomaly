# Kubernetes deployment Makefile for Pynomaly
# Provides automated deployment, management, and monitoring commands

.PHONY: help deploy status logs clean test validate backup restore

# Default target
.DEFAULT_GOAL := help

# Configuration
NAMESPACE ?= pynomaly
KUBECONFIG ?= ~/.kube/config
KUSTOMIZE_DIR ?= .
ENVIRONMENT ?= production
TIMEOUT ?= 600s

# Helper function to check if command exists
check_command = $(shell command -v $(1) 2> /dev/null)

#=============================================================================
# Help and Information
#=============================================================================

help: ## Show this help message
	@echo "Pynomaly Kubernetes Deployment"
	@echo "==============================="
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Environment variables:"
	@echo "  NAMESPACE     Kubernetes namespace (default: pynomaly)"
	@echo "  KUBECONFIG    Path to kubeconfig file (default: ~/.kube/config)"
	@echo "  ENVIRONMENT   Deployment environment (default: production)"
	@echo "  TIMEOUT       Deployment timeout (default: 600s)"

#=============================================================================
# Prerequisites and Validation
#=============================================================================

check-prerequisites: ## Check if required tools are installed
	@echo "Checking prerequisites..."
	@if [ -z "$(call check_command,kubectl)" ]; then \
		echo "❌ kubectl not found. Please install kubectl"; \
		exit 1; \
	fi
	@if [ -z "$(call check_command,kustomize)" ]; then \
		echo "❌ kustomize not found. Please install kustomize"; \
		exit 1; \
	fi
	@echo "✅ All prerequisites are installed"

validate: check-prerequisites ## Validate Kubernetes manifests
	@echo "🔍 Validating Kubernetes manifests..."
	@kustomize build $(KUSTOMIZE_DIR) | kubectl apply --dry-run=client -f -
	@echo "✅ Manifests are valid"

lint: check-prerequisites ## Lint Kubernetes manifests
	@echo "🔍 Linting Kubernetes manifests..."
	@if [ -n "$(call check_command,kubeval)" ]; then \
		kustomize build $(KUSTOMIZE_DIR) | kubeval; \
	else \
		echo "⚠️  kubeval not installed, skipping YAML validation"; \
	fi
	@if [ -n "$(call check_command,kube-score)" ]; then \
		kustomize build $(KUSTOMIZE_DIR) | kube-score score -; \
	else \
		echo "⚠️  kube-score not installed, skipping score analysis"; \
	fi
	@echo "✅ Linting completed"

#=============================================================================
# Deployment Operations
#=============================================================================

deploy: check-prerequisites validate ## Deploy Pynomaly to Kubernetes
	@echo "🚀 Deploying Pynomaly to Kubernetes..."
	@echo "Namespace: $(NAMESPACE)"
	@echo "Environment: $(ENVIRONMENT)"
	@echo ""

	# Create namespace if it doesn't exist
	@kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -

	# Apply manifests with kustomize
	@kustomize build $(KUSTOMIZE_DIR) | kubectl apply -f - --timeout=$(TIMEOUT)

	@echo "✅ Deployment initiated"
	@echo "Use 'make status' to check deployment progress"

upgrade: check-prerequisites validate ## Upgrade existing Pynomaly deployment
	@echo "⬆️  Upgrading Pynomaly deployment..."
	@kustomize build $(KUSTOMIZE_DIR) | kubectl apply -f - --timeout=$(TIMEOUT)
	@echo "✅ Upgrade completed"

rollback: check-prerequisites ## Rollback to previous deployment
	@echo "⏪ Rolling back Pynomaly deployment..."
	@kubectl rollout undo deployment/pynomaly-api -n $(NAMESPACE)
	@kubectl rollout undo deployment/pynomaly-worker-training -n $(NAMESPACE)
	@kubectl rollout undo deployment/pynomaly-worker-drift -n $(NAMESPACE)
	@kubectl rollout undo deployment/pynomaly-prometheus -n $(NAMESPACE)
	@kubectl rollout undo deployment/pynomaly-grafana -n $(NAMESPACE)
	@echo "✅ Rollback completed"

#=============================================================================
# Status and Monitoring
#=============================================================================

status: check-prerequisites ## Show deployment status
	@echo "📊 Pynomaly Deployment Status"
	@echo "============================="
	@echo ""
	@echo "🏠 Namespace:"
	@kubectl get namespace $(NAMESPACE) -o wide 2>/dev/null || echo "Namespace not found"
	@echo ""
	@echo "🚀 Deployments:"
	@kubectl get deployments -n $(NAMESPACE) -o wide
	@echo ""
	@echo "🗃️  StatefulSets:"
	@kubectl get statefulsets -n $(NAMESPACE) -o wide
	@echo ""
	@echo "🔧 Services:"
	@kubectl get services -n $(NAMESPACE) -o wide
	@echo ""
	@echo "📦 Pods:"
	@kubectl get pods -n $(NAMESPACE) -o wide
	@echo ""
	@echo "💾 Persistent Volume Claims:"
	@kubectl get pvc -n $(NAMESPACE) -o wide
	@echo ""
	@echo "🌐 Ingresses:"
	@kubectl get ingress -n $(NAMESPACE) -o wide

pods: check-prerequisites ## Show detailed pod information
	@echo "📦 Pod Details"
	@echo "============="
	@kubectl get pods -n $(NAMESPACE) -o custom-columns="NAME:.metadata.name,STATUS:.status.phase,RESTARTS:.status.containerStatuses[0].restartCount,AGE:.metadata.creationTimestamp,NODE:.spec.nodeName"

health: check-prerequisites ## Check application health
	@echo "🏥 Health Check"
	@echo "==============="
	@echo ""
	@echo "API Health:"
	@kubectl exec -n $(NAMESPACE) deployment/pynomaly-api -- curl -f http://localhost:8000/api/health || echo "API health check failed"
	@echo ""
	@echo "Database Health:"
	@kubectl exec -n $(NAMESPACE) statefulset/pynomaly-postgres -- pg_isready -U pynomaly || echo "Database health check failed"
	@echo ""
	@echo "Redis Health:"
	@kubectl exec -n $(NAMESPACE) statefulset/pynomaly-redis -- redis-cli ping || echo "Redis health check failed"

wait: check-prerequisites ## Wait for deployment to be ready
	@echo "⏳ Waiting for deployment to be ready..."
	@kubectl wait --for=condition=available --timeout=$(TIMEOUT) deployment/pynomaly-api -n $(NAMESPACE)
	@kubectl wait --for=condition=available --timeout=$(TIMEOUT) deployment/pynomaly-worker-training -n $(NAMESPACE)
	@kubectl wait --for=condition=available --timeout=$(TIMEOUT) deployment/pynomaly-worker-drift -n $(NAMESPACE)
	@kubectl wait --for=condition=ready --timeout=$(TIMEOUT) statefulset/pynomaly-postgres -n $(NAMESPACE)
	@kubectl wait --for=condition=ready --timeout=$(TIMEOUT) statefulset/pynomaly-redis -n $(NAMESPACE)
	@echo "✅ All deployments are ready"

#=============================================================================
# Logging and Debugging
#=============================================================================

logs: check-prerequisites ## Show application logs
	@echo "📋 Recent Application Logs"
	@echo "=========================="
	@kubectl logs -n $(NAMESPACE) -l app.kubernetes.io/component=api --tail=50 --timestamps

logs-api: check-prerequisites ## Show API logs
	@kubectl logs -n $(NAMESPACE) -l app.kubernetes.io/component=api -f --timestamps

logs-workers: check-prerequisites ## Show worker logs
	@kubectl logs -n $(NAMESPACE) -l app.kubernetes.io/component=worker-training -f --timestamps

logs-database: check-prerequisites ## Show database logs
	@kubectl logs -n $(NAMESPACE) -l app.kubernetes.io/name=postgres -f --timestamps

logs-monitoring: check-prerequisites ## Show monitoring logs
	@kubectl logs -n $(NAMESPACE) -l app.kubernetes.io/name=prometheus -f --timestamps

events: check-prerequisites ## Show recent events
	@echo "📅 Recent Events"
	@echo "==============="
	@kubectl get events -n $(NAMESPACE) --sort-by='.lastTimestamp' | tail -20

describe-failed: check-prerequisites ## Describe failed pods
	@echo "🔍 Failed Pod Details"
	@echo "====================="
	@kubectl get pods -n $(NAMESPACE) --field-selector=status.phase=Failed -o name | xargs -I {} kubectl describe -n $(NAMESPACE) {}

#=============================================================================
# Access and Port Forwarding
#=============================================================================

port-forward-api: check-prerequisites ## Forward API port to localhost
	@echo "🔗 Forwarding API port 8000 to localhost:8000"
	@kubectl port-forward -n $(NAMESPACE) service/pynomaly-api-service 8000:80

port-forward-grafana: check-prerequisites ## Forward Grafana port to localhost
	@echo "🔗 Forwarding Grafana port 3000 to localhost:3000"
	@kubectl port-forward -n $(NAMESPACE) service/pynomaly-grafana-service 3000:80

port-forward-prometheus: check-prerequisites ## Forward Prometheus port to localhost
	@echo "🔗 Forwarding Prometheus port 9090 to localhost:9090"
	@kubectl port-forward -n $(NAMESPACE) service/pynomaly-prometheus-service 9090:80

port-forward-flower: check-prerequisites ## Forward Flower port to localhost
	@echo "🔗 Forwarding Flower port 5555 to localhost:5555"
	@kubectl port-forward -n $(NAMESPACE) service/pynomaly-flower-service 5555:80

shell-api: check-prerequisites ## Open shell in API pod
	@kubectl exec -it -n $(NAMESPACE) deployment/pynomaly-api -- /bin/bash

shell-database: check-prerequisites ## Open database shell
	@kubectl exec -it -n $(NAMESPACE) statefulset/pynomaly-postgres -- psql -U pynomaly -d pynomaly

#=============================================================================
# Scaling Operations
#=============================================================================

scale-api: check-prerequisites ## Scale API deployment (usage: make scale-api REPLICAS=5)
	@echo "⚖️  Scaling API to $(REPLICAS) replicas..."
	@kubectl scale deployment/pynomaly-api -n $(NAMESPACE) --replicas=$(REPLICAS)
	@echo "✅ API scaled to $(REPLICAS) replicas"

scale-workers: check-prerequisites ## Scale worker deployments (usage: make scale-workers REPLICAS=3)
	@echo "⚖️  Scaling workers to $(REPLICAS) replicas..."
	@kubectl scale deployment/pynomaly-worker-training -n $(NAMESPACE) --replicas=$(REPLICAS)
	@kubectl scale deployment/pynomaly-worker-drift -n $(NAMESPACE) --replicas=$(REPLICAS)
	@echo "✅ Workers scaled to $(REPLICAS) replicas"

autoscale-api: check-prerequisites ## Enable autoscaling for API
	@echo "📈 Enabling autoscaling for API..."
	@kubectl autoscale deployment/pynomaly-api -n $(NAMESPACE) --min=2 --max=10 --cpu-percent=70
	@echo "✅ Autoscaling enabled for API"

#=============================================================================
# Backup and Restore
#=============================================================================

backup: check-prerequisites ## Create backup of persistent data
	@echo "💾 Creating backup..."
	@kubectl create job backup-$(shell date +%Y%m%d-%H%M%S) -n $(NAMESPACE) \
		--image=postgres:16-alpine \
		-- /bin/bash -c "pg_dump -h pynomaly-postgres-service -U pynomaly pynomaly > /backup/pynomaly-$(shell date +%Y%m%d-%H%M%S).sql"
	@echo "✅ Backup job created"

list-backups: check-prerequisites ## List available backups
	@echo "📋 Available Backups"
	@echo "==================="
	@kubectl exec -n $(NAMESPACE) deployment/pynomaly-api -- ls -la /app/storage/backups/ || echo "No backups found"

restore: check-prerequisites ## Restore from backup (usage: make restore BACKUP_FILE=backup.sql)
	@echo "♻️  Restoring from backup: $(BACKUP_FILE)"
	@kubectl exec -n $(NAMESPACE) statefulset/pynomaly-postgres -- psql -U pynomaly -d pynomaly < $(BACKUP_FILE)
	@echo "✅ Restore completed"

#=============================================================================
# Security and Maintenance
#=============================================================================

security-scan: check-prerequisites ## Run security scan on deployments
	@echo "🔒 Running security scan..."
	@if [ -n "$(call check_command,kube-hunter)" ]; then \
		kube-hunter --remote $(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}'); \
	else \
		echo "⚠️  kube-hunter not installed, skipping security scan"; \
	fi

update-secrets: check-prerequisites ## Update secrets (requires manual editing)
	@echo "🔐 Updating secrets..."
	@kubectl edit secret/pynomaly-secrets -n $(NAMESPACE)
	@echo "✅ Secrets updated"

restart-all: check-prerequisites ## Restart all deployments
	@echo "♻️  Restarting all deployments..."
	@kubectl rollout restart deployment -n $(NAMESPACE)
	@kubectl rollout restart statefulset -n $(NAMESPACE)
	@echo "✅ All deployments restarted"

#=============================================================================
# Testing
#=============================================================================

test: check-prerequisites ## Run deployment tests
	@echo "🧪 Running deployment tests..."
	@kubectl run test-pod --rm -i --tty --restart=Never \
		--image=curlimages/curl:latest \
		--namespace=$(NAMESPACE) \
		-- curl -f http://pynomaly-api-service/api/health
	@echo "✅ Tests completed"

load-test: check-prerequisites ## Run basic load test
	@echo "🚀 Running load test..."
	@kubectl run load-test --rm -i --tty --restart=Never \
		--image=curlimages/curl:latest \
		--namespace=$(NAMESPACE) \
		-- /bin/sh -c "for i in \$$(seq 1 100); do curl -f http://pynomaly-api-service/api/health; done"
	@echo "✅ Load test completed"

#=============================================================================
# Cleanup Operations
#=============================================================================

clean: check-prerequisites ## Delete deployment (keeps persistent data)
	@echo "🧹 Cleaning up Pynomaly deployment..."
	@read -p "Are you sure you want to delete the deployment? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		kustomize build $(KUSTOMIZE_DIR) | kubectl delete -f - --ignore-not-found=true; \
		echo "✅ Deployment cleaned up (persistent data preserved)"; \
	else \
		echo "❌ Cleanup cancelled"; \
	fi

clean-all: check-prerequisites ## Delete everything including persistent data
	@echo "🗑️  WARNING: This will delete ALL data including persistent volumes!"
	@read -p "Type 'DELETE' to confirm: " confirm; \
	if [ "$$confirm" = "DELETE" ]; then \
		kustomize build $(KUSTOMIZE_DIR) | kubectl delete -f - --ignore-not-found=true; \
		kubectl delete pvc --all -n $(NAMESPACE); \
		kubectl delete namespace $(NAMESPACE); \
		echo "✅ Everything deleted"; \
	else \
		echo "❌ Cleanup cancelled"; \
	fi

#=============================================================================
# Information and Documentation
#=============================================================================

urls: check-prerequisites ## Show service URLs
	@echo "🌐 Service URLs"
	@echo "==============="
	@echo "API: http://pynomaly.local"
	@echo "Grafana: http://grafana.pynomaly.local"
	@echo "Prometheus: http://prometheus.pynomaly.local"
	@echo "Flower: http://flower.pynomaly.local"
	@echo ""
	@echo "Local port forwarding:"
	@echo "make port-forward-api      # API at http://localhost:8000"
	@echo "make port-forward-grafana  # Grafana at http://localhost:3000"
	@echo "make port-forward-prometheus # Prometheus at http://localhost:9090"

config: check-prerequisites ## Show current configuration
	@echo "⚙️  Current Configuration"
	@echo "========================"
	@echo "Namespace: $(NAMESPACE)"
	@echo "Environment: $(ENVIRONMENT)"
	@echo "Kubeconfig: $(KUBECONFIG)"
	@echo "Timeout: $(TIMEOUT)"
	@echo "Kustomize dir: $(KUSTOMIZE_DIR)"
	@echo ""
	@echo "Cluster info:"
	@kubectl cluster-info
	@echo ""
	@echo "Current context:"
	@kubectl config current-context

version: ## Show version information
	@echo "Pynomaly Kubernetes Deployment"
	@echo "Version: 1.0.0"
	@echo "kubectl version: $(shell kubectl version --client --short 2>/dev/null || echo 'Not available')"
	@echo "kustomize version: $(shell kustomize version --short 2>/dev/null || echo 'Not available')"
