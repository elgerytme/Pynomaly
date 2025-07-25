# Hexagonal Architecture Deployment Makefile
# Provides convenient targets for building and deploying the entire system

.PHONY: help build push deploy-dev deploy-staging deploy-prod docker-dev docker-prod k8s-dev k8s-staging k8s-prod clean

# Default registry and tag
REGISTRY ?= hexagonal-architecture
TAG ?= latest
ENVIRONMENT ?= development

help: ## Show this help message
	@echo "Hexagonal Architecture Deployment Commands"
	@echo "========================================="
	@echo ""
	@echo "Building:"
	@echo "  build              Build all Docker images"
	@echo "  build-push         Build and push all Docker images"
	@echo ""
	@echo "Docker Compose Deployment:"
	@echo "  docker-dev         Deploy with Docker Compose (development)"
	@echo "  docker-prod        Deploy with Docker Compose (production)"
	@echo "  docker-down        Stop Docker Compose services"
	@echo "  docker-logs        View Docker Compose logs"
	@echo ""
	@echo "Kubernetes Deployment:"
	@echo "  k8s-dev            Deploy to Kubernetes (development)"
	@echo "  k8s-staging        Deploy to Kubernetes (staging)"
	@echo "  k8s-prod           Deploy to Kubernetes (production)"
	@echo "  k8s-down-dev       Remove from Kubernetes (development)"
	@echo "  k8s-down-staging   Remove from Kubernetes (staging)"
	@echo "  k8s-down-prod      Remove from Kubernetes (production)"
	@echo ""
	@echo "Testing:"
	@echo "  test-integration   Run integration tests"
	@echo "  test-e2e          Run end-to-end tests"
	@echo ""
	@echo "Utilities:"
	@echo "  clean             Clean up Docker images and containers"
	@echo "  logs              View service logs"
	@echo "  status            Show deployment status"
	@echo ""
	@echo "Examples:"
	@echo "  make build TAG=v1.0.0"
	@echo "  make docker-dev"
	@echo "  make k8s-prod"

# Building targets
build: ## Build all Docker images
	@echo "🏗️  Building Docker images..."
	./scripts/build-images.sh $(TAG)

build-push: ## Build and push all Docker images
	@echo "🏗️  Building and pushing Docker images..."
	./scripts/build-images.sh $(TAG) push

# Docker Compose targets
docker-dev: ## Deploy with Docker Compose (development)
	@echo "🚀 Deploying to development with Docker Compose..."
	./scripts/deploy-docker-compose.sh development up

docker-prod: ## Deploy with Docker Compose (production)
	@echo "🚀 Deploying to production with Docker Compose..."
	./scripts/deploy-docker-compose.sh production up

docker-down: ## Stop Docker Compose services
	@echo "🛑 Stopping Docker Compose services..."
	./scripts/deploy-docker-compose.sh $(ENVIRONMENT) down

docker-restart: ## Restart Docker Compose services
	@echo "🔄 Restarting Docker Compose services..."
	./scripts/deploy-docker-compose.sh $(ENVIRONMENT) restart

docker-logs: ## View Docker Compose logs
	@echo "📜 Viewing Docker Compose logs..."
	./scripts/deploy-docker-compose.sh $(ENVIRONMENT) logs

docker-ps: ## Show Docker Compose service status
	@echo "📊 Docker Compose service status..."
	./scripts/deploy-docker-compose.sh $(ENVIRONMENT) ps

# Kubernetes targets
k8s-dev: ## Deploy to Kubernetes (development)
	@echo "🚀 Deploying to Kubernetes development..."
	./scripts/deploy-kubernetes.sh development apply

k8s-staging: ## Deploy to Kubernetes (staging)
	@echo "🚀 Deploying to Kubernetes staging..."
	./scripts/deploy-kubernetes.sh staging apply

k8s-prod: ## Deploy to Kubernetes (production)
	@echo "🚀 Deploying to Kubernetes production..."
	./scripts/deploy-kubernetes.sh production apply

k8s-down-dev: ## Remove from Kubernetes (development)
	@echo "🛑 Removing from Kubernetes development..."
	./scripts/deploy-kubernetes.sh development delete

k8s-down-staging: ## Remove from Kubernetes (staging)
	@echo "🛑 Removing from Kubernetes staging..."
	./scripts/deploy-kubernetes.sh staging delete

k8s-down-prod: ## Remove from Kubernetes (production)
	@echo "🛑 Removing from Kubernetes production..."
	./scripts/deploy-kubernetes.sh production delete

k8s-diff-dev: ## Show Kubernetes diff (development)
	@echo "🔍 Showing Kubernetes diff for development..."
	./scripts/deploy-kubernetes.sh development diff

k8s-diff-staging: ## Show Kubernetes diff (staging)
	@echo "🔍 Showing Kubernetes diff for staging..."
	./scripts/deploy-kubernetes.sh staging diff

k8s-diff-prod: ## Show Kubernetes diff (production)
	@echo "🔍 Showing Kubernetes diff for production..."
	./scripts/deploy-kubernetes.sh production diff

# Testing targets
test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	cd ../data/data_quality && python test_hexagonal_integration.py
	# Add other package integration tests here

test-e2e: docker-dev ## Run end-to-end tests
	@echo "🧪 Running end-to-end tests..."
	sleep 30  # Wait for services to start
	# Add end-to-end test commands here
	@echo "✅ End-to-end tests completed"

# Utility targets
clean: ## Clean up Docker images and containers
	@echo "🧹 Cleaning up Docker resources..."
	docker system prune -f
	docker image prune -f
	docker volume prune -f

clean-all: ## Clean up all Docker resources (including images)
	@echo "🧹 Cleaning up all Docker resources..."
	docker system prune -a -f

logs: ## View service logs (specify ENVIRONMENT and SERVICE)
	@if [ "$(SERVICE)" = "" ]; then \
		echo "❌ Please specify SERVICE. Example: make logs SERVICE=data-quality ENVIRONMENT=development"; \
		exit 1; \
	fi
	@echo "📜 Viewing logs for $(SERVICE) in $(ENVIRONMENT)..."
	./scripts/deploy-docker-compose.sh $(ENVIRONMENT) logs

status: ## Show deployment status
	@echo "📊 Deployment Status"
	@echo "==================="
	@echo ""
	@echo "🐳 Docker Compose Services:"
	@if docker-compose -f compose/development.yml ps > /dev/null 2>&1; then \
		docker-compose -f compose/development.yml ps; \
	else \
		echo "No development services running"; \
	fi
	@echo ""
	@if docker-compose -f compose/production.yml ps > /dev/null 2>&1; then \
		docker-compose -f compose/production.yml ps; \
	else \
		echo "No production services running"; \
	fi
	@echo ""
	@echo "☸️  Kubernetes Services:"
	@kubectl get pods -A -l part-of=hexagonal-architecture 2>/dev/null || echo "No Kubernetes services found"

# Development shortcuts
dev: docker-dev ## Quick development deployment
staging: k8s-staging ## Quick staging deployment  
prod: k8s-prod ## Quick production deployment

# CI/CD targets
ci-build: ## CI: Build images for CI/CD
	@echo "🏗️  CI: Building images..."
	./scripts/build-images.sh $(TAG)

ci-test: ## CI: Run all tests
	@echo "🧪 CI: Running tests..."
	$(MAKE) test-integration

ci-deploy-staging: ## CI: Deploy to staging
	@echo "🚀 CI: Deploying to staging..."
	$(MAKE) k8s-staging

ci-deploy-prod: ## CI: Deploy to production
	@echo "🚀 CI: Deploying to production..."
	$(MAKE) k8s-prod

# Local development helpers
dev-logs: ## View development logs
	@$(MAKE) logs ENVIRONMENT=development SERVICE=data-quality

dev-restart: ## Restart development services
	@$(MAKE) docker-restart ENVIRONMENT=development

dev-clean: ## Clean and restart development
	@$(MAKE) docker-down ENVIRONMENT=development
	@$(MAKE) clean
	@$(MAKE) docker-dev