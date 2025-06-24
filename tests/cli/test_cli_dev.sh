#!/bin/bash
# Pynomaly CLI Testing Script for Development Environment
# Tests CLI functionality in current development environment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DATA_DIR="$SCRIPT_DIR/test_data"
TEMP_DIR="/tmp/pynomaly_cli_dev_test_$$"
LOG_FILE="$TEMP_DIR/test_results.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Create temp directory and log file
mkdir -p "$TEMP_DIR"
mkdir -p "$TEST_DATA_DIR"
touch "$LOG_FILE"

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
    ((PASSED_TESTS++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    ((FAILED_TESTS++))
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_exit_code="${3:-0}"
    
    ((TOTAL_TESTS++))
    log "Running test: $test_name"
    
    local start_time=$(date +%s)
    local output
    local exit_code
    
    if output=$(eval "$test_command" 2>&1); then
        exit_code=0
    else
        exit_code=$?
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ "$exit_code" -eq "$expected_exit_code" ]; then
        log_success "$test_name (${duration}s)"
    else
        log_error "$test_name (${duration}s) - Exit code: $exit_code, Expected: $expected_exit_code"
        log_error "Output: $output"
    fi
}

# Generate simple test data
generate_test_data() {
    log "Generating test data..."
    
    # Small CSV file
    cat > "$TEST_DATA_DIR/small_data.csv" << 'EOF'
id,value1,value2,value3,category
1,10.5,20.1,5.0,A
2,11.2,19.8,4.9,A
3,10.8,20.5,5.1,A
4,50.0,80.0,25.0,B
5,9.9,19.5,4.8,A
EOF

    log "Test data generation complete"
}

# Test basic functionality that should work even without full installation
test_basic_module_import() {
    log "Testing basic module imports..."
    
    cd "$PROJECT_ROOT"
    
    # Test Python path and basic imports
    run_test "Python Path Setup" "export PYTHONPATH='$PROJECT_ROOT/src:\$PYTHONPATH' && python3 -c 'import sys; print(\"Python path setup successful\")'"
    
    # Test basic module import
    run_test "Basic Module Import" "export PYTHONPATH='$PROJECT_ROOT/src:\$PYTHONPATH' && python3 -c 'import pynomaly.domain.entities.anomaly'"
    
    # Test CLI module import
    run_test "CLI Module Import" "export PYTHONPATH='$PROJECT_ROOT/src:\$PYTHONPATH' && python3 -c 'from pynomaly.presentation.cli.app import app'"
}

# Test CLI app directly through Python
test_cli_direct() {
    log "Testing CLI through direct Python execution..."
    
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    
    # Test CLI help
    run_test "CLI Help Direct" "python3 -m pynomaly.presentation.cli.app --help"
    
    # Test CLI version if available
    run_test "CLI Version Direct" "python3 -c 'from pynomaly.presentation.cli.app import app; import typer; typer.main.get_command(app)([\"version\"])'" || true
}

# Test autonomous service
test_autonomous_service() {
    log "Testing autonomous service functionality..."
    
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    
    # Test autonomous service import
    run_test "Autonomous Service Import" "python3 -c 'from pynomaly.application.services.autonomous_service import AutonomousDetectionService'"
    
    # Test data loader imports
    run_test "CSV Loader Import" "python3 -c 'from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader'"
    run_test "JSON Loader Import" "python3 -c 'from pynomaly.infrastructure.data_loaders.json_loader import JSONLoader'"
}

# Test container and dependency injection
test_container() {
    log "Testing container and dependency injection..."
    
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    
    # Test container import
    run_test "Container Import" "python3 -c 'from pynomaly.presentation.cli.container import get_cli_container'"
    
    # Test container creation
    run_test "Container Creation" "python3 -c 'from pynomaly.presentation.cli.container import get_cli_container; container = get_cli_container(); print(\"Container created successfully\")'"
}

# Test with Poetry if available
test_with_poetry() {
    log "Testing with Poetry..."
    
    cd "$PROJECT_ROOT"
    
    # Try to install in development mode
    if command -v poetry &> /dev/null; then
        log "Poetry detected, attempting development installation..."
        
        # Install dependencies only (not the package itself due to pyproject.toml issues)
        run_test "Poetry Dependencies Install" "poetry install --only main" || true
        
        # Test if we can run with poetry
        run_test "Poetry Python Path" "poetry run python -c 'import sys; print(sys.path[0])'" || true
    else
        log "Poetry not available, skipping Poetry tests"
    fi
}

cleanup() {
    log "Cleaning up..."
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

main() {
    log "Starting Pynomaly CLI development environment tests"
    log "Platform: $(uname -s) $(uname -r)"
    log "Python: $(python3 --version 2>&1)"
    
    generate_test_data
    test_basic_module_import
    test_autonomous_service
    test_container
    test_cli_direct
    test_with_poetry
    
    # Summary
    echo ""
    echo "=============================================="
    echo "         CLI DEVELOPMENT TEST SUMMARY"
    echo "=============================================="
    echo "Total Tests:    $TOTAL_TESTS"
    echo "Passed:         $PASSED_TESTS"
    echo "Failed:         $FAILED_TESTS"
    if [ $TOTAL_TESTS -gt 0 ]; then
        echo "Success Rate:   $(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "N/A")%"
    fi
    echo "=============================================="
    echo ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}🎉 All development tests passed!${NC}"
        echo "Log file: $LOG_FILE"
        return 0
    else
        echo -e "${RED}❌ Some tests failed. Check the log for details.${NC}"
        echo "Log file: $LOG_FILE"
        return 1
    fi
}

trap cleanup EXIT
main "$@"