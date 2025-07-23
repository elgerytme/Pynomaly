#!/bin/bash
# CI/CD Setup Script for Anomaly Detection Service

set -e

echo "🚀 Setting up CI/CD for Anomaly Detection Service"
echo "================================================"

# Configuration
PROJECT_DIR=$(pwd)
SERVICE_NAME="anomaly-detection"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    if ! command -v python3 &> /dev/null; then
        missing_tools+=("python3")
    fi
    
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi
    
    if ! command -v git &> /dev/null; then
        missing_tools+=("git")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_error "Please install missing tools and run again"
        exit 1
    fi
    
    print_success "All prerequisites satisfied"
}

# Setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created Python virtual environment"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-dev.txt
    
    print_success "Python environment ready"
}

# Setup pre-commit hooks
setup_pre_commit() {
    print_status "Setting up pre-commit hooks..."
    
    if [ -f ".pre-commit-config.yaml" ]; then
        source venv/bin/activate
        pre-commit install
        pre-commit install --hook-type commit-msg
        pre-commit install --hook-type pre-push
        
        # Run pre-commit on all files (optional)
        print_status "Running pre-commit on all files..."
        pre-commit run --all-files || print_warning "Some pre-commit checks failed"
        
        print_success "Pre-commit hooks installed"
    else
        print_warning "No .pre-commit-config.yaml found, skipping pre-commit setup"
    fi
}

# Setup Docker environment
setup_docker() {
    print_status "Setting up Docker environment..."
    
    # Test Docker build
    if docker build -t "${SERVICE_NAME}:test" . &> /dev/null; then
        print_success "Docker build test passed"
        docker rmi "${SERVICE_NAME}:test" &> /dev/null || true
    else
        print_error "Docker build test failed"
        return 1
    fi
    
    # Setup test environment
    if [ -f "docker-compose.test.yml" ]; then
        print_status "Testing Docker Compose configuration..."
        docker-compose -f docker-compose.test.yml config &> /dev/null
        print_success "Docker Compose test configuration valid"
    fi
}

# Validate Kubernetes configurations
validate_k8s() {
    print_status "Validating Kubernetes configurations..."
    
    local k8s_valid=true
    
    # Check if kustomize is available
    if ! command -v kustomize &> /dev/null; then
        print_warning "kustomize not found, installing..."
        curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
        sudo mv kustomize /usr/local/bin/
    fi
    
    # Validate each overlay
    for overlay in k8s/overlays/*/; do
        if [ -d "$overlay" ]; then
            overlay_name=$(basename "$overlay")
            print_status "Validating $overlay_name overlay..."
            
            if kustomize build "$overlay" > /dev/null 2>&1; then
                print_success "$overlay_name overlay is valid"
            else
                print_error "$overlay_name overlay has errors"
                k8s_valid=false
            fi
        fi
    done
    
    if [ "$k8s_valid" = true ]; then
        print_success "All Kubernetes configurations are valid"
    else
        print_error "Some Kubernetes configurations have errors"
        return 1
    fi
}

# Setup GitHub Actions (if using GitHub)
setup_github_actions() {
    if [ -d ".github/workflows" ]; then
        print_status "Validating GitHub Actions workflows..."
        
        # Check for required secrets documentation
        cat << EOF > .github/REQUIRED_SECRETS.md
# Required GitHub Secrets

The following secrets need to be configured in GitHub repository settings:

## Container Registry
- \`GITHUB_TOKEN\`: Automatically provided by GitHub Actions

## AWS/EKS (if deploying to AWS)
- \`AWS_ACCESS_KEY_ID\`: AWS access key for EKS access
- \`AWS_SECRET_ACCESS_KEY\`: AWS secret key for EKS access

## Notifications
- \`SLACK_WEBHOOK_URL\`: Slack webhook URL for notifications

## Additional Secrets (environment specific)
- Database credentials
- API keys
- SSL certificates

Update these in: Repository Settings > Secrets and variables > Actions
EOF
        
        print_success "GitHub Actions setup complete"
        print_warning "Please configure required secrets (see .github/REQUIRED_SECRETS.md)"
    fi
}

# Setup GitLab CI (if using GitLab)
setup_gitlab_ci() {
    if [ -f ".gitlab-ci.yml" ]; then
        print_status "Validating GitLab CI configuration..."
        
        # Create variables documentation
        cat << EOF > GITLAB_VARIABLES.md
# Required GitLab CI/CD Variables

Configure these variables in GitLab: Project Settings > CI/CD > Variables

## Container Registry
- \`CI_REGISTRY_USER\`: GitLab container registry username
- \`CI_REGISTRY_PASSWORD\`: GitLab container registry password

## Kubernetes
- \`KUBE_CONFIG_DEV\`: Base64 encoded kubeconfig for development cluster
- \`KUBE_CONFIG_STAGING\`: Base64 encoded kubeconfig for staging cluster
- \`KUBE_CONFIG_PROD\`: Base64 encoded kubeconfig for production cluster

## Notifications
- \`SLACK_WEBHOOK_URL\`: Slack webhook URL for notifications

## Database
- \`POSTGRES_PASSWORD\`: PostgreSQL password for environments
- \`REDIS_PASSWORD\`: Redis password (if using AUTH)
EOF
        
        print_success "GitLab CI setup complete"
        print_warning "Please configure required variables (see GITLAB_VARIABLES.md)"
    fi
}

# Run tests to verify setup
run_tests() {
    print_status "Running test suite to verify setup..."
    
    source venv/bin/activate
    
    # Start test services
    docker-compose -f docker-compose.test.yml up -d
    
    # Wait for services to be ready
    sleep 10
    
    # Set test environment variables
    export ANOMALY_DETECTION_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/anomaly_detection_test"
    export ANOMALY_DETECTION_REDIS_URL="redis://localhost:6379/0"
    export ANOMALY_DETECTION_ENV="testing"
    
    # Run a subset of tests
    if python -m pytest tests/unit/ -x --tb=short; then
        print_success "Test suite passed"
    else
        print_error "Test suite failed"
        # Cleanup on failure
        docker-compose -f docker-compose.test.yml down
        return 1
    fi
    
    # Cleanup
    docker-compose -f docker-compose.test.yml down
}

# Generate CI/CD documentation
generate_docs() {
    print_status "Generating CI/CD documentation..."
    
    cat << 'EOF' > CI_CD_GUIDE.md
# CI/CD Pipeline Guide

## Overview
This project uses multiple CI/CD platforms for maximum compatibility:
- GitHub Actions (`.github/workflows/ci.yml`)
- GitLab CI (`.gitlab-ci.yml`)
- Azure DevOps (`azure-pipelines.yml`)
- Jenkins (`Jenkinsfile`)

## Pipeline Stages
Each pipeline includes the following stages:

### 1. Quality Checks
- Code formatting (Black)
- Import sorting (isort)
- Linting (flake8)
- Type checking (mypy)
- Security scanning (bandit, safety)

### 2. Testing
- Unit tests with coverage
- Integration tests
- Performance benchmarks
- End-to-end tests (staging only)

### 3. Build & Security
- Docker image build
- Container security scanning (Trivy)
- Multi-architecture builds (optional)

### 4. Deployment
- Development (develop branch)
- Staging (main branch)
- Production (tags only, manual approval)

## Local Development
```bash
# Setup environment
./scripts/ci-setup.sh

# Run quality checks
pre-commit run --all-files

# Run tests
pytest tests/unit/ --cov=anomaly_detection

# Build Docker image
./scripts/build-docker.sh

# Deploy to development
./scripts/deploy-k8s.sh development
```

## Environment Configuration
Each environment has its own configuration:
- Development: `k8s/overlays/development/`
- Staging: `k8s/overlays/staging/`  
- Production: `k8s/overlays/production/`

## Monitoring
- Pipeline status: CI/CD platform dashboards
- Application metrics: Prometheus + Grafana
- Logs: Centralized logging system
- Alerts: Slack notifications

## Troubleshooting
1. Check pipeline logs in CI/CD platform
2. Validate Kubernetes manifests: `kustomize build k8s/overlays/production/`
3. Test Docker build locally: `./scripts/build-docker.sh`
4. Check application health: `kubectl get pods -n anomaly-detection`

## Security
- All secrets stored in CI/CD platform secret management
- Container images scanned for vulnerabilities
- RBAC configured for Kubernetes deployments
- Network policies applied in production
EOF

    print_success "CI/CD documentation generated"
}

# Main execution
main() {
    echo "Starting CI/CD setup for $SERVICE_NAME..."
    echo ""
    
    check_prerequisites
    setup_python_env
    setup_pre_commit
    setup_docker
    validate_k8s
    setup_github_actions
    setup_gitlab_ci
    run_tests
    generate_docs
    
    echo ""
    print_success "🎉 CI/CD setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Configure secrets/variables in your CI/CD platform"
    echo "  2. Push changes to trigger first pipeline run"
    echo "  3. Monitor deployment in your Kubernetes cluster"
    echo "  4. Set up monitoring and alerting"
    echo ""
    echo "📚 Documentation:"
    echo "  - CI/CD Guide: CI_CD_GUIDE.md"
    echo "  - GitHub Secrets: .github/REQUIRED_SECRETS.md (if applicable)"
    echo "  - GitLab Variables: GITLAB_VARIABLES.md (if applicable)"
    echo ""
    print_status "Happy deploying! 🚀"
}

# Run main function
main "$@"