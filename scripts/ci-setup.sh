#!/bin/bash
# CI/CD Setup Script for Pynomaly Reproducible Test Environment
# This script demonstrates how to use the reproducible test environment in CI/CD

set -e

echo "🚀 Setting up Pynomaly Reproducible Test Environment"

# Environment variables
export PYNOMALY_ENVIRONMENT="ci"
export PYNOMALY_LOG_LEVEL="INFO"
export PYNOMALY_CACHE_ENABLED="false"
export PYNOMALY_AUTH_ENABLED="false"
export PYNOMALY_TESTING="true"
export PYTHONPATH="src"
export COVERAGE_CORE="sysmon"
export PLAYWRIGHT_BROWSERS_PATH="/tmp/browsers"
export HEADLESS="true"
export CI="true"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# Function to install Poetry
install_poetry() {
    log "Installing Poetry..."
    if ! command -v poetry &> /dev/null; then
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
        success "Poetry installed successfully"
    else
        success "Poetry already installed"
    fi
}

# Function to install dependencies
install_dependencies() {
    log "Installing dependencies from requirements.lock..."
    
    # Use requirements.lock for reproducible builds
    pip install -r requirements.lock
    pip install -r requirements-dev.lock
    
    # Install additional test dependencies
    pip install \
        tox>=4.0.0 \
        pytest>=8.0.0 \
        pytest-cov>=6.0.0 \
        pytest-asyncio>=0.24.0 \
        pytest-xdist>=3.6.0 \
        playwright>=1.40.0 \
        pytest-playwright>=0.4.3 \
        mutmut>=2.4.0
    
    success "Dependencies installed"
}

# Function to install Playwright browsers
install_browsers() {
    log "Installing Playwright browsers..."
    playwright install chromium firefox webkit
    playwright install-deps
    success "Browsers installed"
}

# Function to run linting
run_lint() {
    log "Running linting checks..."
    tox -e lint
    success "Linting completed"
}

# Function to run type checking
run_type_check() {
    log "Running type checking..."
    tox -e type
    success "Type checking completed"
}

# Function to run unit tests
run_unit_tests() {
    log "Running unit tests..."
    tox -e unit
    success "Unit tests completed"
}

# Function to run integration tests
run_integration_tests() {
    log "Running integration tests..."
    tox -e integration
    success "Integration tests completed"
}

# Function to run mutation testing
run_mutation_tests() {
    log "Running mutation tests..."
    tox -e mutation
    success "Mutation tests completed"
}

# Function to run UI tests
run_ui_tests() {
    log "Running UI/E2E tests..."
    tox -e e2e-ui
    success "UI/E2E tests completed"
}

# Function to generate coverage report
generate_coverage() {
    log "Generating coverage report..."
    tox -e coverage
    success "Coverage report generated"
}

# Function to run security checks
run_security_checks() {
    log "Running security checks..."
    tox -e security
    success "Security checks completed"
}

# Function to run performance tests
run_performance_tests() {
    log "Running performance tests..."
    tox -e performance
    success "Performance tests completed"
}

# Function to export test results
export_test_results() {
    log "Exporting test results..."
    
    # Create reports directory
    mkdir -p reports
    
    # Copy test results
    find .tox -name "*.xml" -exec cp {} reports/ \;
    find .tox -name "*.json" -exec cp {} reports/ \;
    find .tox -name "htmlcov" -exec cp -r {} reports/ \;
    
    success "Test results exported to reports/"
}

# Function to clean up
cleanup() {
    log "Cleaning up..."
    tox -e clean
    success "Cleanup completed"
}

# Main execution
main() {
    log "Starting CI/CD pipeline..."
    
    # Parse command line arguments
    STAGE="${1:-all}"
    
    case "$STAGE" in
        "setup")
            install_poetry
            install_dependencies
            install_browsers
            ;;
        "lint")
            run_lint
            ;;
        "type")
            run_type_check
            ;;
        "unit")
            run_unit_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "mutation")
            run_mutation_tests
            ;;
        "ui")
            run_ui_tests
            ;;
        "security")
            run_security_checks
            ;;
        "performance")
            run_performance_tests
            ;;
        "coverage")
            generate_coverage
            ;;
        "export")
            export_test_results
            ;;
        "clean")
            cleanup
            ;;
        "all")
            install_poetry
            install_dependencies
            install_browsers
            run_lint
            run_type_check
            run_unit_tests
            run_integration_tests
            run_security_checks
            generate_coverage
            export_test_results
            ;;
        "quick")
            install_poetry
            install_dependencies
            run_lint
            run_type_check
            run_unit_tests
            ;;
        "full")
            install_poetry
            install_dependencies
            install_browsers
            run_lint
            run_type_check
            run_unit_tests
            run_integration_tests
            run_mutation_tests
            run_ui_tests
            run_security_checks
            run_performance_tests
            generate_coverage
            export_test_results
            ;;
        *)
            echo "Usage: $0 {setup|lint|type|unit|integration|mutation|ui|security|performance|coverage|export|clean|all|quick|full}"
            echo ""
            echo "Stages:"
            echo "  setup       - Install dependencies and browsers"
            echo "  lint        - Run linting checks"
            echo "  type        - Run type checking"
            echo "  unit        - Run unit tests"
            echo "  integration - Run integration tests"
            echo "  mutation    - Run mutation tests"
            echo "  ui          - Run UI/E2E tests"
            echo "  security    - Run security checks"
            echo "  performance - Run performance tests"
            echo "  coverage    - Generate coverage report"
            echo "  export      - Export test results"
            echo "  clean       - Clean up build artifacts"
            echo "  all         - Run standard CI pipeline"
            echo "  quick       - Run quick validation (lint, type, unit)"
            echo "  full        - Run comprehensive test suite"
            exit 1
            ;;
    esac
    
    success "CI/CD pipeline completed successfully!"
}

# Error handling
trap 'error "CI/CD pipeline failed!"' ERR

# Run main function
main "$@"
