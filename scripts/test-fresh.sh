#!/bin/bash

# Test runner for fresh environment (Bash)
# Usage: ./scripts/test-fresh.sh [options]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$PROJECT_ROOT/tests"
SRC_DIR="$PROJECT_ROOT/src"
REPORTS_DIR="$PROJECT_ROOT/test-reports-fresh"
COVERAGE_DIR="$REPORTS_DIR/coverage"
VENV_DIR="$PROJECT_ROOT/.test-venv"

# Default options
VERBOSE=false
COVERAGE=true
PARALLEL=true
FAST=false
INTEGRATION=true
UNIT=true
PERFORMANCE=false
SECURITY=false
FAIL_FAST=false
MARKERS=""
PYTEST_ARGS=""
PYTHON_VERSION="3.11"
CLEAN_VENV=false
KEEP_VENV=false

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run tests in a fresh virtual environment

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -q, --quiet             Disable verbose output (opposite of -v)
    -c, --coverage          Enable coverage reporting (default: enabled)
    --no-coverage           Disable coverage reporting
    -p, --parallel          Run tests in parallel (default: enabled)
    --no-parallel           Disable parallel execution
    -f, --fast              Run only fast tests (skip slow integration tests)
    --unit-only             Run only unit tests
    --integration-only      Run only integration tests
    --performance           Include performance tests
    --security              Include security tests
    --fail-fast             Stop on first failure
    -m, --markers MARKERS   Run tests with specific pytest markers
    --pytest-args ARGS      Additional arguments to pass to pytest
    --python-version VER    Python version to use (default: 3.11)
    --clean-venv            Remove existing test environment before creating new one
    --keep-venv             Keep the test environment after tests complete
    
EXAMPLES:
    $0                      # Run all tests in fresh environment
    $0 -v --clean-venv      # Verbose mode with clean environment
    $0 --fast --keep-venv   # Run only fast tests and keep environment
    $0 --python-version 3.12 # Use Python 3.12
    $0 --unit-only          # Run only unit tests in fresh environment

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            VERBOSE=false
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        --no-parallel)
            PARALLEL=false
            shift
            ;;
        -f|--fast)
            FAST=true
            shift
            ;;
        --unit-only)
            UNIT=true
            INTEGRATION=false
            shift
            ;;
        --integration-only)
            UNIT=false
            INTEGRATION=true
            shift
            ;;
        --performance)
            PERFORMANCE=true
            shift
            ;;
        --security)
            SECURITY=true
            shift
            ;;
        --fail-fast)
            FAIL_FAST=true
            shift
            ;;
        -m|--markers)
            MARKERS="$2"
            shift 2
            ;;
        --pytest-args)
            PYTEST_ARGS="$2"
            shift 2
            ;;
        --pytest-args=*)
            PYTEST_ARGS="${1#*=}"
            shift
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --python-version=*)
            PYTHON_VERSION="${1#*=}"
            shift
            ;;
        --clean-venv)
            CLEAN_VENV=true
            shift
            ;;
        --keep-venv)
            KEEP_VENV=true
            shift
            ;;
        *)
            print_color $RED "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect Python command
detect_python() {
    local python_candidates=(
        "python$PYTHON_VERSION"
        "python3.$PYTHON_VERSION"
        "python3"
        "python"
    )
    
    for cmd in "${python_candidates[@]}"; do
        if command_exists "$cmd"; then
            # Verify version if possible
            local version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
            if [[ -n "$version" ]]; then
                print_color $GREEN "✓ Found Python $version: $cmd"
                echo "$cmd"
                return 0
            fi
        fi
    done
    
    print_color $RED "Error: Python $PYTHON_VERSION not found"
    exit 1
}

# Function to setup fresh virtual environment
setup_fresh_environment() {
    print_color $BLUE "🏗️ Setting up fresh virtual environment..."
    
    # Clean existing environment if requested
    if [[ "$CLEAN_VENV" == true && -d "$VENV_DIR" ]]; then
        print_color $YELLOW "🧹 Removing existing test environment..."
        rm -rf "$VENV_DIR"
    fi
    
    # Detect Python
    PYTHON_CMD=$(detect_python)
    
    # Create virtual environment
    if [[ ! -d "$VENV_DIR" ]]; then
        print_color $BLUE "📦 Creating virtual environment with $PYTHON_CMD..."
        $PYTHON_CMD -m venv "$VENV_DIR"
    else
        print_color $GREEN "✓ Using existing virtual environment"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_color $GREEN "✓ Virtual environment activated: $VENV_DIR"
    
    # Upgrade pip
    print_color $BLUE "⬆️ Upgrading pip..."
    python -m pip install --upgrade pip wheel setuptools
    
    # Install project dependencies
    print_color $BLUE "📚 Installing project dependencies..."
    
    # Install from pyproject.toml if available
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        # Install project in development mode with all extras
        cd "$PROJECT_ROOT"
        python -m pip install -e ".[dev,test,performance,security]"
    else
        # Fallback to requirements files
        if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
            python -m pip install -r "$PROJECT_ROOT/requirements.txt"
        fi
        if [[ -f "$PROJECT_ROOT/requirements-dev.txt" ]]; then
            python -m pip install -r "$PROJECT_ROOT/requirements-dev.txt"
        fi
        if [[ -f "$PROJECT_ROOT/requirements-test.txt" ]]; then
            python -m pip install -r "$PROJECT_ROOT/requirements-test.txt"
        fi
    fi
    
    # Install additional test dependencies
    print_color $BLUE "🧪 Installing test dependencies..."
    python -m pip install pytest pytest-cov pytest-xdist pytest-mock pytest-asyncio pytest-benchmark pytest-security hypothesis
    
    # Install performance testing tools if needed
    if [[ "$PERFORMANCE" == true ]]; then
        python -m pip install pytest-benchmark memory-profiler
    fi
    
    # Install security testing tools if needed
    if [[ "$SECURITY" == true ]]; then
        python -m pip install bandit safety
    fi
    
    print_color $GREEN "✅ Fresh environment setup complete"
    
    # Show installed packages
    if [[ "$VERBOSE" == true ]]; then
        print_color $CYAN "📋 Installed packages:"
        python -m pip list
    fi
}

# Function to build pytest command
build_pytest_command() {
    local cmd="python -m pytest"
    
    # Base test directory
    if [[ "$UNIT" == true && "$INTEGRATION" == false ]]; then
        cmd="$cmd $TEST_DIR/unit $TEST_DIR/domain $TEST_DIR/application"
    elif [[ "$INTEGRATION" == true && "$UNIT" == false ]]; then
        cmd="$cmd $TEST_DIR/integration $TEST_DIR/infrastructure"
    else
        cmd="$cmd $TEST_DIR"
    fi
    
    # Verbose output
    if [[ "$VERBOSE" == true ]]; then
        cmd="$cmd -v"
    else
        cmd="$cmd -q"
    fi
    
    # Coverage options
    if [[ "$COVERAGE" == true ]]; then
        cmd="$cmd --cov=pynomaly --cov-report=html:$COVERAGE_DIR/html --cov-report=xml:$COVERAGE_DIR/coverage.xml --cov-report=term"
    fi
    
    # Parallel execution
    if [[ "$PARALLEL" == true ]]; then
        local num_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
        cmd="$cmd -n $num_cores"
    fi
    
    # Fast mode (skip slow tests)
    if [[ "$FAST" == true ]]; then
        cmd="$cmd -m 'not slow and not integration'"
    fi
    
    # Performance tests
    if [[ "$PERFORMANCE" == true ]]; then
        cmd="$cmd -m 'performance'"
    fi
    
    # Security tests
    if [[ "$SECURITY" == true ]]; then
        cmd="$cmd -m 'security'"
    fi
    
    # Fail fast
    if [[ "$FAIL_FAST" == true ]]; then
        cmd="$cmd -x"
    fi
    
    # Custom markers
    if [[ -n "$MARKERS" ]]; then
        cmd="$cmd -m '$MARKERS'"
    fi
    
    # Additional pytest arguments
    if [[ -n "$PYTEST_ARGS" ]]; then
        cmd="$cmd $PYTEST_ARGS"
    fi
    
    # Output options
    cmd="$cmd --tb=short --strict-markers"
    
    # JUnit XML for CI
    cmd="$cmd --junitxml=$REPORTS_DIR/junit.xml"
    
    echo "$cmd"
}

# Function to run tests
run_tests() {
    print_color $BLUE "🧪 Running tests in fresh environment..."
    
    # Create reports directory
    mkdir -p "$REPORTS_DIR" "$COVERAGE_DIR"
    
    local pytest_cmd=$(build_pytest_command)
    
    print_color $YELLOW "Command: $pytest_cmd"
    echo
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run tests
    local start_time=$(date +%s)
    
    if eval "$pytest_cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_color $GREEN "✅ Tests completed successfully in ${duration}s"
        
        # Show coverage summary if enabled
        if [[ "$COVERAGE" == true ]]; then
            print_color $BLUE "📊 Coverage report generated: $COVERAGE_DIR/html/index.html"
        fi
        
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_color $RED "❌ Tests failed after ${duration}s"
        return 1
    fi
}

# Function to cleanup environment
cleanup_environment() {
    if [[ "$KEEP_VENV" == false ]]; then
        print_color $BLUE "🧹 Cleaning up test environment..."
        deactivate 2>/dev/null || true
        rm -rf "$VENV_DIR"
        print_color $GREEN "✓ Test environment cleaned up"
    else
        print_color $YELLOW "📦 Test environment preserved at: $VENV_DIR"
        print_color $CYAN "To reuse: source $VENV_DIR/bin/activate"
    fi
}

# Function to show summary
show_summary() {
    print_color $BLUE "📋 Test Summary"
    echo "=================="
    echo "Project: Pynomaly"
    echo "Environment: Fresh Virtual Environment"
    echo "Python Version: $PYTHON_VERSION"
    echo "Virtual Environment: $VENV_DIR"
    echo "Test Directory: $TEST_DIR"
    echo "Reports Directory: $REPORTS_DIR"
    
    if [[ "$COVERAGE" == true ]]; then
        echo "Coverage: Enabled"
        echo "Coverage Reports: $COVERAGE_DIR/"
    else
        echo "Coverage: Disabled"
    fi
    
    if [[ "$PARALLEL" == true ]]; then
        echo "Parallel Execution: Enabled"
    else
        echo "Parallel Execution: Disabled"
    fi
    
    echo "Fast Mode: $FAST"
    echo "Unit Tests: $UNIT"
    echo "Integration Tests: $INTEGRATION"
    echo "Performance Tests: $PERFORMANCE"
    echo "Security Tests: $SECURITY"
    echo "Clean Environment: $CLEAN_VENV"
    echo "Keep Environment: $KEEP_VENV"
    
    if [[ -n "$MARKERS" ]]; then
        echo "Custom Markers: $MARKERS"
    fi
    
    echo
}

# Main execution
main() {
    print_color $BLUE "🚀 Pynomaly Test Runner (Fresh Environment)"
    print_color $BLUE "==========================================="
    echo
    
    # Check if we're in the project root
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        print_color $RED "Error: Not in project root directory"
        exit 1
    fi
    
    show_summary
    
    setup_fresh_environment
    
    local test_result=0
    if run_tests; then
        print_color $GREEN "🎉 All tests passed in fresh environment!"
        test_result=0
    else
        print_color $RED "💥 Some tests failed in fresh environment!"
        test_result=1
    fi
    
    cleanup_environment
    
    exit $test_result
}

# Trap to ensure cleanup on exit
trap 'print_color $YELLOW "🛑 Test execution interrupted"; cleanup_environment' INT TERM

# Run main function
main "$@"