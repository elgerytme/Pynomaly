#!/bin/bash

# Test runner for current environment (Bash)
# Usage: ./scripts/test-current.sh [options]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$PROJECT_ROOT/tests"
SRC_DIR="$PROJECT_ROOT/src"
REPORTS_DIR="$PROJECT_ROOT/test-reports"
COVERAGE_DIR="$REPORTS_DIR/coverage"

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

Run tests in the current environment

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
    
EXAMPLES:
    $0                      # Run all tests with default settings
    $0 -v --no-coverage     # Verbose mode without coverage
    $0 --fast               # Run only fast tests
    $0 --unit-only          # Run only unit tests
    $0 -m "not slow"        # Run tests not marked as slow
    $0 --pytest-args="-x -s" # Pass additional pytest arguments

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
    if command_exists python3; then
        echo "python3"
    elif command_exists python; then
        echo "python"
    else
        print_color $RED "Error: Python not found"
        exit 1
    fi
}

# Function to check environment
check_environment() {
    print_color $BLUE "🔍 Checking test environment..."
    
    # Check if we're in the project root
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        print_color $RED "Error: Not in project root directory"
        exit 1
    fi
    
    # Detect Python
    PYTHON_CMD=$(detect_python)
    print_color $GREEN "✓ Python found: $PYTHON_CMD"
    
    # Check if pytest is available
    if ! $PYTHON_CMD -m pytest --version >/dev/null 2>&1; then
        print_color $RED "Error: pytest not available. Install with: pip install pytest"
        exit 1
    fi
    print_color $GREEN "✓ pytest available"
    
    # Check if we're in a virtual environment
    if [[ -n "$VIRTUAL_ENV" ]] || [[ -n "$CONDA_DEFAULT_ENV" ]] || [[ "$PYTHON_CMD" == *".venv"* ]]; then
        print_color $GREEN "✓ Virtual environment detected"
    else
        print_color $YELLOW "⚠ Warning: Not in a virtual environment"
    fi
    
    # Create reports directory
    mkdir -p "$REPORTS_DIR" "$COVERAGE_DIR"
    print_color $GREEN "✓ Test reports directory: $REPORTS_DIR"
}

# Function to build pytest command
build_pytest_command() {
    local cmd="$PYTHON_CMD -m pytest"
    
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
    if [[ "$PARALLEL" == true ]] && command_exists pytest-xdist; then
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
    print_color $BLUE "🧪 Running tests in current environment..."
    
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

# Function to show summary
show_summary() {
    print_color $BLUE "📋 Test Summary"
    echo "=================="
    echo "Project: Pynomaly"
    echo "Environment: Current"
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
    
    if [[ -n "$MARKERS" ]]; then
        echo "Custom Markers: $MARKERS"
    fi
    
    echo
}

# Main execution
main() {
    print_color $BLUE "🚀 Pynomaly Test Runner (Current Environment)"
    print_color $BLUE "=============================================="
    echo
    
    check_environment
    show_summary
    
    if run_tests; then
        print_color $GREEN "🎉 All tests passed!"
        exit 0
    else
        print_color $RED "💥 Some tests failed!"
        exit 1
    fi
}

# Trap to ensure cleanup on exit
trap 'print_color $YELLOW "🛑 Test execution interrupted"' INT TERM

# Run main function
main "$@"