# Scripts Directory

This directory contains automation scripts for development, build, deployment, and maintenance tasks.

## Scripts Structure

```
scripts/
├── setup/        # Environment setup and initialization scripts
├── build/        # Build and compilation scripts
├── deploy/       # Deployment and release scripts
├── test/         # Testing automation scripts
└── maintenance/  # Maintenance and utility scripts
```

## Script Categories

### Setup Scripts (`setup/`)
**Environment initialization and configuration**

- `setup.sh` / `setup.ps1` - Main setup script
- `install-dependencies.sh` - Install project dependencies
- `configure-environment.sh` - Configure development environment
- `init-database.sh` - Initialize database
- `setup-git-hooks.sh` - Configure Git hooks

**Purpose**: Prepare development environment for new contributors

### Build Scripts (`build/`)
**Compilation and build automation**

- `build.sh` / `build.ps1` - Main build script
- `compile.sh` - Compile source code
- `package.sh` - Package application
- `docker-build.sh` - Build Docker images
- `optimize.sh` - Optimize build artifacts

**Purpose**: Automate build processes and artifact creation

### Deploy Scripts (`deploy/`)
**Deployment and release automation**

- `deploy.sh` / `deploy.ps1` - Main deployment script
- `deploy-staging.sh` - Deploy to staging environment
- `deploy-production.sh` - Deploy to production environment
- `rollback.sh` - Rollback deployment
- `health-check.sh` - Post-deployment health checks

**Purpose**: Automate deployment processes and releases

### Test Scripts (`test/`)
**Testing automation and validation**

- `run-tests.sh` / `run-tests.ps1` - Run all tests
- `unit-tests.sh` - Run unit tests
- `integration-tests.sh` - Run integration tests
- `e2e-tests.sh` - Run end-to-end tests
- `performance-tests.sh` - Run performance tests
- `security-tests.sh` - Run security tests

**Purpose**: Automate testing processes and validation

### Maintenance Scripts (`maintenance/`)
**Maintenance and utility tasks**

- `cleanup.sh` / `cleanup.ps1` - Clean up temporary files
- `backup.sh` - Backup important data
- `update-dependencies.sh` - Update project dependencies
- `monitor.sh` - Monitoring and health checks
- `logs.sh` - Log management and analysis

**Purpose**: Automate maintenance tasks and utilities

## Script Standards

### Cross-Platform Compatibility
- Provide both Unix shell (`.sh`) and PowerShell (`.ps1`) versions
- Use platform-appropriate file paths and commands
- Handle environment differences gracefully
- Test on multiple operating systems

### Error Handling
```bash
#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Function for error handling
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Usage
command_that_might_fail || error_exit "Command failed"
```

### Logging and Output
```bash
#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}
```

### Configuration Management
```bash
#!/bin/bash

# Load configuration
CONFIG_FILE="${CONFIG_FILE:-config/default.conf}"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    log_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi
```

## Example Scripts

### Setup Script (`setup/setup.sh`)
```bash
#!/bin/bash
set -euo pipefail

# Setup script for development environment
echo "Setting up development environment..."

# Check prerequisites
command -v git >/dev/null 2>&1 || { echo "Git is required but not installed." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed." >&2; exit 1; }

# Install dependencies
echo "Installing dependencies..."
if [[ -f "package.json" ]]; then
    npm install
elif [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
elif [[ -f "go.mod" ]]; then
    go mod download
fi

# Setup pre-commit hooks
echo "Setting up Git hooks..."
if [[ -f ".pre-commit-config.yaml" ]]; then
    pre-commit install
fi

# Initialize database
echo "Initializing database..."
if [[ -f "scripts/setup/init-database.sh" ]]; then
    ./scripts/setup/init-database.sh
fi

# Create necessary directories
mkdir -p logs
mkdir -p tmp

echo "Development environment setup complete!"
```

### Build Script (`build/build.sh`)
```bash
#!/bin/bash
set -euo pipefail

# Build script
BUILD_VERSION="${BUILD_VERSION:-latest}"
BUILD_TARGET="${BUILD_TARGET:-production}"

echo "Building application (version: $BUILD_VERSION, target: $BUILD_TARGET)..."

# Clean previous build
echo "Cleaning previous build..."
rm -rf dist/ build/

# Run tests before building
echo "Running tests..."
npm test

# Build application
echo "Building application..."
case "$BUILD_TARGET" in
    "development")
        npm run build:dev
        ;;
    "production")
        npm run build:prod
        ;;
    *)
        echo "Unknown build target: $BUILD_TARGET"
        exit 1
        ;;
esac

# Create deployment package
echo "Creating deployment package..."
tar -czf "dist/app-$BUILD_VERSION.tar.gz" -C dist/ .

echo "Build completed successfully!"
```

### Deploy Script (`deploy/deploy.sh`)
```bash
#!/bin/bash
set -euo pipefail

# Deployment script
ENVIRONMENT="${1:-staging}"
VERSION="${2:-latest}"

echo "Deploying to $ENVIRONMENT (version: $VERSION)..."

# Validate environment
case "$ENVIRONMENT" in
    "staging"|"production")
        ;;
    *)
        echo "Invalid environment: $ENVIRONMENT"
        echo "Usage: $0 <staging|production> [version]"
        exit 1
        ;;
esac

# Pre-deployment checks
echo "Running pre-deployment checks..."
if [[ "$ENVIRONMENT" == "production" ]]; then
    # Additional checks for production
    ./scripts/test/run-tests.sh
    ./scripts/deploy/health-check.sh staging
fi

# Deploy application
echo "Deploying application..."
if command -v kubectl >/dev/null 2>&1; then
    kubectl set image deployment/myapp myapp=myapp:$VERSION
    kubectl rollout status deployment/myapp
else
    # Alternative deployment method
    docker-compose -f docker-compose.$ENVIRONMENT.yml up -d
fi

# Post-deployment verification
echo "Running post-deployment checks..."
./scripts/deploy/health-check.sh $ENVIRONMENT

echo "Deployment to $ENVIRONMENT completed successfully!"
```

### Test Script (`test/run-tests.sh`)
```bash
#!/bin/bash
set -euo pipefail

# Test runner script
TEST_TYPE="${1:-all}"
COVERAGE="${COVERAGE:-false}"

echo "Running tests (type: $TEST_TYPE, coverage: $COVERAGE)..."

# Function to run tests with coverage
run_with_coverage() {
    if [[ "$COVERAGE" == "true" ]]; then
        # Language-specific coverage commands
        if [[ -f "package.json" ]]; then
            npm run test:coverage
        elif [[ -f "requirements.txt" ]]; then
            pytest --cov=src --cov-report=html
        elif [[ -f "go.mod" ]]; then
            go test -coverprofile=coverage.out ./...
            go tool cover -html=coverage.out -o coverage.html
        fi
    else
        # Regular test commands
        if [[ -f "package.json" ]]; then
            npm test
        elif [[ -f "requirements.txt" ]]; then
            pytest
        elif [[ -f "go.mod" ]]; then
            go test ./...
        fi
    fi
}

# Run specific test types
case "$TEST_TYPE" in
    "unit")
        echo "Running unit tests..."
        run_with_coverage
        ;;
    "integration")
        echo "Running integration tests..."
        # Integration test commands
        ;;
    "e2e")
        echo "Running end-to-end tests..."
        # E2E test commands
        ;;
    "all")
        echo "Running all tests..."
        ./scripts/test/run-tests.sh unit
        ./scripts/test/run-tests.sh integration
        ./scripts/test/run-tests.sh e2e
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Usage: $0 <unit|integration|e2e|all>"
        exit 1
        ;;
esac

echo "Tests completed successfully!"
```

### Maintenance Script (`maintenance/cleanup.sh`)
```bash
#!/bin/bash
set -euo pipefail

# Cleanup script
CLEANUP_TYPE="${1:-all}"

echo "Running cleanup (type: $CLEANUP_TYPE)..."

cleanup_build() {
    echo "Cleaning build artifacts..."
    rm -rf build/ dist/ target/
    rm -f *.tar.gz *.zip
}

cleanup_cache() {
    echo "Cleaning cache..."
    rm -rf .cache/ node_modules/.cache/
    rm -rf __pycache__/ .pytest_cache/
}

cleanup_logs() {
    echo "Cleaning logs..."
    find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
}

cleanup_docker() {
    echo "Cleaning Docker resources..."
    docker system prune -f
    docker volume prune -f
}

# Run specific cleanup types
case "$CLEANUP_TYPE" in
    "build")
        cleanup_build
        ;;
    "cache")
        cleanup_cache
        ;;
    "logs")
        cleanup_logs
        ;;
    "docker")
        cleanup_docker
        ;;
    "all")
        cleanup_build
        cleanup_cache
        cleanup_logs
        ;;
    *)
        echo "Unknown cleanup type: $CLEANUP_TYPE"
        echo "Usage: $0 <build|cache|logs|docker|all>"
        exit 1
        ;;
esac

echo "Cleanup completed successfully!"
```

## PowerShell Examples

### Setup Script (`setup/setup.ps1`)
```powershell
#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Environment = "development"
)

$ErrorActionPreference = "Stop"

Write-Host "Setting up development environment..." -ForegroundColor Green

# Check prerequisites
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git is required but not installed."
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is required but not installed."
}

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
if (Test-Path "package.json") {
    npm install
} elseif (Test-Path "requirements.txt") {
    pip install -r requirements.txt
}

# Setup directories
New-Item -Path "logs" -ItemType Directory -Force
New-Item -Path "tmp" -ItemType Directory -Force

Write-Host "Development environment setup complete!" -ForegroundColor Green
```

## Script Execution

### Making Scripts Executable
```bash
# Unix/Linux/macOS
chmod +x scripts/**/*.sh

# Windows (PowerShell)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Running Scripts
```bash
# Unix/Linux/macOS
./scripts/setup/setup.sh
./scripts/build/build.sh production
./scripts/deploy/deploy.sh staging v1.2.3

# Windows (PowerShell)
.\scripts\setup\setup.ps1
.\scripts\build\build.ps1 -Target production
.\scripts\deploy\deploy.ps1 -Environment staging -Version v1.2.3
```

## Best Practices

### Script Design
1. **Idempotent**: Scripts should be safe to run multiple times
2. **Self-Documenting**: Clear usage instructions and help text
3. **Error Handling**: Proper error handling and cleanup
4. **Logging**: Comprehensive logging for debugging
5. **Configuration**: Externalize configuration where possible

### Security
1. **No Hardcoded Secrets**: Use environment variables or secret management
2. **Input Validation**: Validate all inputs and parameters
3. **Permissions**: Run with minimal required permissions
4. **Audit Trail**: Log all actions for security auditing

### Maintenance
1. **Version Control**: All scripts in version control
2. **Documentation**: Document purpose and usage
3. **Testing**: Test scripts in different environments
4. **Regular Updates**: Keep scripts current with system changes

## Common Utilities

### Utility Functions
```bash
# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Wait for service to be ready
wait_for_service() {
    local url="$1"
    local timeout="${2:-30}"
    local count=0
    
    while ! curl -s "$url" >/dev/null; do
        sleep 1
        count=$((count + 1))
        if [[ $count -gt $timeout ]]; then
            echo "Service not ready after ${timeout}s"
            return 1
        fi
    done
}

# Retry function
retry() {
    local retries="$1"
    shift
    local count=0
    
    until "$@"; do
        exit_code=$?
        count=$((count + 1))
        if [[ $count -lt $retries ]]; then
            echo "Attempt $count failed. Retrying..."
            sleep 1
        else
            echo "Failed after $retries attempts."
            return $exit_code
        fi
    done
}
```

## Integration with CI/CD

### GitHub Actions
```yaml
name: CI/CD
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run tests
      run: ./scripts/test/run-tests.sh all
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build application
      run: ./scripts/build/build.sh production
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to staging
      run: ./scripts/deploy/deploy.sh staging
```

## Troubleshooting

### Common Issues
1. **Permission Denied**: Check file permissions and execution policy
2. **Path Issues**: Use absolute paths or proper relative paths
3. **Environment Variables**: Ensure all required variables are set
4. **Dependencies**: Verify all required tools are installed

### Debugging Tips
1. **Add Debug Mode**: Use `set -x` in bash or `Set-PSDebug -Trace 1` in PowerShell
2. **Log Everything**: Comprehensive logging helps identify issues
3. **Test Incrementally**: Test scripts in small pieces
4. **Use Dry Run**: Implement dry-run mode for testing
