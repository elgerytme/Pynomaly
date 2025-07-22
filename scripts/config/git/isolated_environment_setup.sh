#!/bin/bash
# Isolated Environment Setup Script
# Creates isolated development environments for git branches

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="$SCRIPT_DIR/branch_isolation_config.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Parse branch name to extract isolation info
parse_branch_name() {
    local branch_name="$1"
    
    # Check if it follows isolation pattern
    if [[ $branch_name =~ ^(feature|bugfix|hotfix|experiment)/(user|agent|pkg)-([a-zA-Z0-9_-]+)/(.+)$ ]]; then
        export BRANCH_TYPE="${BASH_REMATCH[1]}"
        export SCOPE_TYPE="${BASH_REMATCH[2]}"
        export SCOPE_ID="${BASH_REMATCH[3]}"
        export DESCRIPTION="${BASH_REMATCH[4]}"
        return 0
    else
        return 1
    fi
}

# Get current branch name
get_current_branch() {
    git -C "$REPO_ROOT" branch --show-current
}

# Create isolated workspace directory
create_workspace() {
    local scope_type="$1"
    local scope_id="$2"
    
    local workspace_dir="$REPO_ROOT/workspaces/$scope_type-$scope_id"
    
    if [ ! -d "$workspace_dir" ]; then
        mkdir -p "$workspace_dir"
        log_success "Created workspace directory: $workspace_dir"
        
        # Create subdirectories for isolation
        mkdir -p "$workspace_dir/logs"
        mkdir -p "$workspace_dir/temp"
        mkdir -p "$workspace_dir/data"
        mkdir -p "$workspace_dir/config"
        
        # Create .gitignore for workspace
        cat > "$workspace_dir/.gitignore" << EOF
# Workspace temporary files
*.log
*.tmp
temp/
logs/
data/
*.pid
.env.local
EOF
        
        log_info "Created workspace subdirectories and .gitignore"
    else
        log_info "Workspace directory already exists: $workspace_dir"
    fi
    
    echo "$workspace_dir"
}

# Generate isolated environment variables
generate_env_file() {
    local scope_type="$1"
    local scope_id="$2"
    local branch_name="$3"
    local workspace_dir="$4"
    
    local env_file="$workspace_dir/.env"
    
    # Base port calculation based on scope
    local base_port
    case "$scope_type" in
        "user") base_port=8000 ;;
        "agent") base_port=9000 ;;
        "pkg") base_port=7000 ;;
        *) base_port=6000 ;;
    esac
    
    # Calculate unique port based on scope_id hash
    local port_offset=$(echo -n "$scope_id" | md5sum | cut -c1-2)
    port_offset=$((16#$port_offset % 100))
    local isolated_port=$((base_port + port_offset))
    
    cat > "$env_file" << EOF
# Isolated Environment Configuration
# Generated on $(date)
# Branch: $branch_name

# Isolation identifiers
BRANCH_ISOLATION_SCOPE=$scope_type
BRANCH_ISOLATION_ID=$scope_id
BRANCH_ISOLATION_NAME=$branch_name
BRANCH_ISOLATION_WORKSPACE=$workspace_dir

# Service ports (isolated)
API_PORT=$isolated_port
WEB_UI_PORT=$((isolated_port + 1))
DEBUG_PORT=$((isolated_port + 2))
MONITORING_PORT=$((isolated_port + 3))

# Database settings (isolated)
DATABASE_NAME=pynomaly_${scope_type}_${scope_id}
TEST_DATABASE_PREFIX=test_${scope_type}_${scope_id}_

# Logging settings (isolated)
LOG_PREFIX=${scope_type}-${scope_id}
LOG_LEVEL=DEBUG
LOG_FILE=$workspace_dir/logs/app.log

# Cache settings (isolated)
CACHE_KEY_PREFIX=${scope_type}:${scope_id}:
REDIS_DB=$((port_offset % 16))

# File paths (isolated)
TEMP_DIR=$workspace_dir/temp
DATA_DIR=$workspace_dir/data
CONFIG_DIR=$workspace_dir/config

# Testing isolation
PYTEST_CURRENT_TEST_ID=${scope_type}_${scope_id}
TEST_ISOLATION_ENABLED=true

# CI/CD isolation markers
CI_ISOLATION_SCOPE=$scope_type
CI_ISOLATION_ID=$scope_id
CI_BRANCH_NAME=$branch_name
EOF
    
    log_success "Generated environment file: $env_file"
    echo "$env_file"
}

# Create isolated Docker Compose override
create_docker_compose_override() {
    local scope_type="$1"
    local scope_id="$2"
    local workspace_dir="$3"
    local isolated_port="$4"
    
    local compose_file="$workspace_dir/docker-compose.override.yml"
    
    cat > "$compose_file" << EOF
# Docker Compose Override for Isolated Environment
# Scope: $scope_type-$scope_id

version: '3.8'

services:
  api:
    ports:
      - "$isolated_port:8000"
    environment:
      - ISOLATION_SCOPE=$scope_type
      - ISOLATION_ID=$scope_id
      - DATABASE_NAME=pynomaly_${scope_type}_${scope_id}
    volumes:
      - $workspace_dir/logs:/app/logs
      - $workspace_dir/temp:/app/temp
      - $workspace_dir/data:/app/data
    
  web:
    ports:
      - "$((isolated_port + 1)):3000"
    environment:
      - REACT_APP_API_URL=http://localhost:$isolated_port
      - REACT_APP_ISOLATION_SCOPE=$scope_type
      - REACT_APP_ISOLATION_ID=$scope_id
    
  database:
    environment:
      - POSTGRES_DB=pynomaly_${scope_type}_${scope_id}
    volumes:
      - ${scope_type}_${scope_id}_postgres_data:/var/lib/postgresql/data
    
  redis:
    command: redis-server --databases 16
    
volumes:
  ${scope_type}_${scope_id}_postgres_data:
    name: pynomaly_${scope_type}_${scope_id}_postgres_data
EOF
    
    log_success "Created Docker Compose override: $compose_file"
    echo "$compose_file"
}

# Create isolated testing configuration
create_test_config() {
    local scope_type="$1"
    local scope_id="$2"
    local workspace_dir="$3"
    
    local pytest_ini="$workspace_dir/pytest.ini"
    
    cat > "$pytest_ini" << EOF
[tool:pytest]
# Isolated pytest configuration for $scope_type-$scope_id

testpaths = 
    $REPO_ROOT/src/packages

python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# Isolation markers
markers =
    isolation: tests running in isolated environment
    ${scope_type}_${scope_id}: tests for this specific isolation scope
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests

# Test database configuration
addopts = 
    --strict-markers
    --strict-config
    --verbose
    -ra
    --cov=src
    --cov-branch
    --cov-report=term-missing
    --cov-report=html:$workspace_dir/htmlcov
    --cov-report=xml:$workspace_dir/coverage.xml
    --junit-xml=$workspace_dir/test-results.xml

# Environment variables for tests
env = 
    TESTING = true
    TEST_ISOLATION_SCOPE = $scope_type
    TEST_ISOLATION_ID = $scope_id
    DATABASE_URL = postgresql://test:test@localhost:5432/test_${scope_type}_${scope_id}
    REDIS_URL = redis://localhost:6379/$(echo -n "$scope_id" | md5sum | cut -c1-2 | xargs -I{} echo "ibase=16; {}" | bc | awk '{print $1 % 16}')
EOF
    
    log_success "Created test configuration: $pytest_ini"
    echo "$pytest_ini"
}

# Create monitoring configuration
create_monitoring_config() {
    local scope_type="$1"
    local scope_id="$2"
    local workspace_dir="$3"
    local isolated_port="$4"
    
    local monitoring_dir="$workspace_dir/monitoring"
    mkdir -p "$monitoring_dir"
    
    # Prometheus configuration
    cat > "$monitoring_dir/prometheus.yml" << EOF
# Prometheus configuration for isolated environment
# Scope: $scope_type-$scope_id

global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    isolation_scope: $scope_type
    isolation_id: $scope_id

scrape_configs:
  - job_name: 'pynomaly-api-$scope_type-$scope_id'
    static_configs:
      - targets: ['localhost:$((isolated_port + 3))']
    metrics_path: /metrics
    scrape_interval: 5s
    
  - job_name: 'pynomaly-system-$scope_type-$scope_id'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 30s
EOF
    
    # Grafana dashboard configuration
    cat > "$monitoring_dir/grafana-dashboard.json" << EOF
{
  "dashboard": {
    "title": "Pynomaly Isolated Environment - $scope_type-$scope_id",
    "tags": ["isolation", "$scope_type", "$scope_id"],
    "timezone": "browser",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{isolation_scope=\"$scope_type\",isolation_id=\"$scope_id\"}[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{isolation_scope=\"$scope_type\",isolation_id=\"$scope_id\",status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      }
    ]
  }
}
EOF
    
    log_success "Created monitoring configuration: $monitoring_dir"
    echo "$monitoring_dir"
}

# Main setup function
setup_isolated_environment() {
    local branch_name="$1"
    
    log_info "Setting up isolated environment for branch: $branch_name"
    
    # Parse branch name
    if ! parse_branch_name "$branch_name"; then
        log_error "Branch name does not follow isolation pattern"
        log_error "Expected: <type>/(user|agent|pkg)-<id>/<description>"
        return 1
    fi
    
    log_info "Parsed branch info:"
    log_info "  Type: $BRANCH_TYPE"
    log_info "  Scope: $SCOPE_TYPE-$SCOPE_ID"
    log_info "  Description: $DESCRIPTION"
    
    # Create workspace
    local workspace_dir
    workspace_dir=$(create_workspace "$SCOPE_TYPE" "$SCOPE_ID")
    
    # Generate environment file
    local env_file
    env_file=$(generate_env_file "$SCOPE_TYPE" "$SCOPE_ID" "$branch_name" "$workspace_dir")
    
    # Source the environment to get isolated port
    source "$env_file"
    
    # Create Docker Compose override
    create_docker_compose_override "$SCOPE_TYPE" "$SCOPE_ID" "$workspace_dir" "$API_PORT"
    
    # Create test configuration
    create_test_config "$SCOPE_TYPE" "$SCOPE_ID" "$workspace_dir"
    
    # Create monitoring configuration
    create_monitoring_config "$SCOPE_TYPE" "$SCOPE_ID" "$workspace_dir" "$API_PORT"
    
    # Create activation script
    cat > "$workspace_dir/activate.sh" << EOF
#!/bin/bash
# Activation script for isolated environment
# Scope: $SCOPE_TYPE-$SCOPE_ID

echo "🚀 Activating isolated environment for $branch_name"
echo "📁 Workspace: $workspace_dir"
echo "🌐 API Port: $API_PORT"
echo "🔧 Web UI Port: $WEB_UI_PORT"

# Source environment variables
if [ -f "$env_file" ]; then
    source "$env_file"
    echo "✅ Environment variables loaded"
else
    echo "❌ Environment file not found: $env_file"
    exit 1
fi

# Set up aliases for convenience
alias run-api="cd $REPO_ROOT && docker-compose -f docker-compose.yml -f $workspace_dir/docker-compose.override.yml up api"
alias run-web="cd $REPO_ROOT && docker-compose -f docker-compose.yml -f $workspace_dir/docker-compose.override.yml up web"
alias run-tests="cd $REPO_ROOT && pytest -c $workspace_dir/pytest.ini"
alias run-monitoring="cd $REPO_ROOT && docker-compose -f docker-compose.yml -f $workspace_dir/docker-compose.override.yml up prometheus grafana"

echo "📝 Available commands:"
echo "  run-api        - Start API in isolated environment"
echo "  run-web        - Start web UI in isolated environment"  
echo "  run-tests      - Run tests with isolation configuration"
echo "  run-monitoring - Start monitoring stack"

echo ""
echo "🔗 Quick links:"
echo "  API: http://localhost:$API_PORT"
echo "  Web UI: http://localhost:$WEB_UI_PORT"
echo "  Monitoring: http://localhost:$((API_PORT + 3))"
echo ""
EOF
    
    chmod +x "$workspace_dir/activate.sh"
    
    log_success "Isolated environment setup complete!"
    log_info "To activate the environment, run: source $workspace_dir/activate.sh"
    
    return 0
}

# Cleanup function
cleanup_isolated_environment() {
    local scope_type="$1"
    local scope_id="$2"
    
    local workspace_dir="$REPO_ROOT/workspaces/$scope_type-$scope_id"
    
    if [ -d "$workspace_dir" ]; then
        log_info "Cleaning up isolated environment: $scope_type-$scope_id"
        
        # Stop any running containers
        if [ -f "$workspace_dir/docker-compose.override.yml" ]; then
            docker-compose -f "$REPO_ROOT/docker-compose.yml" -f "$workspace_dir/docker-compose.override.yml" down -v 2>/dev/null || true
        fi
        
        # Remove workspace directory
        rm -rf "$workspace_dir"
        log_success "Removed workspace: $workspace_dir"
        
        # Remove Docker volumes
        docker volume rm "pynomaly_${scope_type}_${scope_id}_postgres_data" 2>/dev/null || true
        log_success "Removed Docker volumes"
    else
        log_warning "No workspace found for: $scope_type-$scope_id"
    fi
}

# Main function
main() {
    local command="${1:-setup}"
    local branch_name="${2:-}"
    
    case "$command" in
        "setup")
            if [ -z "$branch_name" ]; then
                branch_name=$(get_current_branch)
                if [ -z "$branch_name" ]; then
                    log_error "Could not determine current branch"
                    exit 1
                fi
            fi
            setup_isolated_environment "$branch_name"
            ;;
            
        "cleanup")
            if [ -z "$branch_name" ]; then
                log_error "Branch name required for cleanup"
                echo "Usage: $0 cleanup <branch_name>"
                exit 1
            fi
            
            if parse_branch_name "$branch_name"; then
                cleanup_isolated_environment "$SCOPE_TYPE" "$SCOPE_ID"
            else
                log_error "Invalid branch name format"
                exit 1
            fi
            ;;
            
        "help"|"-h"|"--help")
            echo "Isolated Environment Setup Script"
            echo ""
            echo "Usage:"
            echo "  $0 [command] [branch_name]"
            echo ""
            echo "Commands:"
            echo "  setup [branch_name]   - Setup isolated environment (default: current branch)"
            echo "  cleanup <branch_name> - Cleanup isolated environment"
            echo "  help                  - Show this help message"
            echo ""
            ;;
            
        *)
            log_error "Unknown command: $command"
            echo "Run '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"