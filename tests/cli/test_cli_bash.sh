#!/bin/bash
# Pynomaly CLI Testing Script for Bash Environments
# Tests CLI functionality in fresh Linux/WSL/macOS environments

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DATA_DIR="$SCRIPT_DIR/test_data"
TEMP_DIR="/tmp/pynomaly_cli_test_$$"
VENV_DIR="$TEMP_DIR/test_venv"
LOG_FILE="$TEMP_DIR/test_results.log"
RESULTS_JSON="$TEMP_DIR/test_results.json"

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

# Test results array
declare -a TEST_RESULTS=()

# Utility functions
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

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Test execution wrapper
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_exit_code="${3:-0}"
    
    ((TOTAL_TESTS++))
    log "Running test: $test_name"
    
    local start_time=$(date +%s)
    local output
    local exit_code
    
    # Run the test command and capture output
    if output=$(eval "$test_command" 2>&1); then
        exit_code=0
    else
        exit_code=$?
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Check if test passed
    if [ "$exit_code" -eq "$expected_exit_code" ]; then
        log_success "$test_name (${duration}s)"
        TEST_RESULTS+=("{\"name\":\"$test_name\",\"status\":\"PASS\",\"duration\":$duration,\"output\":\"$(echo "$output" | sed 's/"/\\"/g')\"}")
    else
        log_error "$test_name (${duration}s) - Exit code: $exit_code, Expected: $expected_exit_code"
        log_error "Output: $output"
        TEST_RESULTS+=("{\"name\":\"$test_name\",\"status\":\"FAIL\",\"duration\":$duration,\"exit_code\":$exit_code,\"output\":\"$(echo "$output" | sed 's/"/\\"/g')\"}")
    fi
}

# Setup test environment
setup_test_environment() {
    log "Setting up test environment..."
    
    # Create temporary directory
    mkdir -p "$TEMP_DIR"
    mkdir -p "$TEST_DATA_DIR"
    
    # Create virtual environment
    log "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    log "Test environment setup complete"
}

# Generate test data
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
6,10.3,20.2,5.0,A
7,45.0,75.0,20.0,B
8,10.7,20.0,5.0,A
9,10.1,19.9,4.9,A
10,55.0,85.0,30.0,B
EOF

    # Medium CSV file with anomalies
    python3 << 'EOF' > "$TEST_DATA_DIR/medium_data.csv"
import csv
import random
import numpy as np

# Generate data with anomalies
np.random.seed(42)
data = []

# Normal data
for i in range(9000):
    x1 = np.random.normal(10, 2)
    x2 = np.random.normal(20, 3)
    x3 = np.random.normal(5, 1)
    category = random.choice(['A', 'B', 'C'])
    data.append([i+1, x1, x2, x3, category])

# Anomalous data
for i in range(1000):
    x1 = np.random.normal(50, 5)  # Clear outliers
    x2 = np.random.normal(100, 10)
    x3 = np.random.normal(25, 5)
    category = random.choice(['A', 'B', 'C'])
    data.append([9000+i+1, x1, x2, x3, category])

# Shuffle data
random.shuffle(data)

# Write to CSV
with open('medium_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'value1', 'value2', 'value3', 'category'])
    writer.writerows(data)
EOF

    # JSON file
    cat > "$TEST_DATA_DIR/sample_data.json" << 'EOF'
[
    {"id": 1, "value1": 10.5, "value2": 20.1, "value3": 5.0, "category": "A"},
    {"id": 2, "value1": 11.2, "value2": 19.8, "value3": 4.9, "category": "A"},
    {"id": 3, "value1": 10.8, "value2": 20.5, "value3": 5.1, "category": "A"},
    {"id": 4, "value1": 50.0, "value2": 80.0, "value3": 25.0, "category": "B"},
    {"id": 5, "value1": 9.9, "value2": 19.5, "value3": 4.8, "category": "A"}
]
EOF

    # Malformed CSV
    cat > "$TEST_DATA_DIR/malformed_data.csv" << 'EOF'
id,value1,value2,value3,category
1,10.5,20.1,5.0,A
2,11.2,19.8,4.9
3,10.8,20.5,5.1,A,extra_column
4,not_a_number,80.0,25.0,B
5,9.9,19.5,4.8,A
EOF

    log "Test data generation complete"
}

# Install Pynomaly in test environment
install_pynomaly() {
    log "Installing Pynomaly..."
    
    # Install from local project
    cd "$PROJECT_ROOT"
    pip install -e .
    
    # Verify installation
    if command -v pynomaly &> /dev/null; then
        log_success "Pynomaly CLI installed successfully"
    else
        log_error "Pynomaly CLI not found in PATH"
        exit 1
    fi
}

# Core CLI tests
test_basic_commands() {
    log "Testing basic CLI commands..."
    
    # Test help
    run_test "CLI Help" "pynomaly --help"
    
    # Test version
    run_test "CLI Version" "pynomaly version"
    
    # Test config show
    run_test "Config Show" "pynomaly config --show"
    
    # Test status
    run_test "System Status" "pynomaly status"
    
    # Test quickstart
    run_test "Quickstart Help" "echo 'n' | pynomaly quickstart"
}

# Dataset management tests
test_dataset_commands() {
    log "Testing dataset commands..."
    
    # Test dataset help
    run_test "Dataset Help" "pynomaly dataset --help"
    
    # Test dataset list (should be empty initially)
    run_test "Dataset List Empty" "pynomaly dataset list"
    
    # Test dataset load - CSV
    run_test "Load CSV Dataset" "pynomaly dataset load '$TEST_DATA_DIR/small_data.csv' --name test_small"
    
    # Test dataset load - JSON
    run_test "Load JSON Dataset" "pynomaly dataset load '$TEST_DATA_DIR/sample_data.json' --name test_json"
    
    # Test dataset list (should show loaded datasets)
    run_test "Dataset List After Loading" "pynomaly dataset list"
    
    # Test dataset info
    run_test "Dataset Info" "pynomaly dataset info test_small"
    
    # Test dataset validation
    run_test "Dataset Validation" "pynomaly dataset validate test_small"
    
    # Test malformed data handling
    run_test "Load Malformed CSV" "pynomaly dataset load '$TEST_DATA_DIR/malformed_data.csv' --name test_malformed" 1
}

# Detector management tests
test_detector_commands() {
    log "Testing detector commands..."
    
    # Test detector help
    run_test "Detector Help" "pynomaly detector --help"
    
    # Test detector list (should be empty initially)
    run_test "Detector List Empty" "pynomaly detector list"
    
    # Test detector create
    run_test "Create IsolationForest Detector" "pynomaly detector create --name test_detector --algorithm IsolationForest"
    
    # Test detector list (should show created detector)
    run_test "Detector List After Creation" "pynomaly detector list"
    
    # Test detector info
    run_test "Detector Info" "pynomaly detector info test_detector"
    
    # Test algorithm list
    run_test "Algorithm List" "pynomaly detector algorithms"
}

# Detection workflow tests
test_detection_commands() {
    log "Testing detection commands..."
    
    # Test detection help
    run_test "Detection Help" "pynomaly detect --help"
    
    # Test train detector
    run_test "Train Detector" "pynomaly detect train --detector test_detector --dataset test_small"
    
    # Test run detection
    run_test "Run Detection" "pynomaly detect run --detector test_detector --dataset test_small"
    
    # Test results
    run_test "View Results" "pynomaly detect results --latest"
    
    # Test results with specific detector
    run_test "View Detector Results" "pynomaly detect results --detector test_detector"
}

# Autonomous mode tests
test_autonomous_commands() {
    log "Testing autonomous mode commands..."
    
    # Test autonomous help
    run_test "Autonomous Help" "pynomaly auto --help"
    
    # Test autonomous detect
    run_test "Autonomous Detection" "pynomaly auto detect '$TEST_DATA_DIR/small_data.csv' --max-algorithms 2"
    
    # Test autonomous profile
    run_test "Autonomous Profile" "pynomaly auto profile '$TEST_DATA_DIR/small_data.csv'"
    
    # Test autonomous quick detection
    run_test "Autonomous Quick" "pynomaly auto quick '$TEST_DATA_DIR/small_data.csv' --algorithm IsolationForest"
}

# Export functionality tests
test_export_commands() {
    log "Testing export commands..."
    
    # Test export help
    run_test "Export Help" "pynomaly export --help"
    
    # Test list formats
    run_test "List Export Formats" "pynomaly export list-formats"
    
    # Create a simple results file for export testing
    cat > "$TEMP_DIR/test_results.json" << 'EOF'
{
    "anomalies": [
        {"id": 1, "score": 0.8, "is_anomaly": true},
        {"id": 2, "score": 0.2, "is_anomaly": false}
    ],
    "summary": {
        "total_samples": 2,
        "anomaly_count": 1,
        "anomaly_rate": 0.5
    }
}
EOF
    
    # Test CSV export
    run_test "Export to CSV" "pynomaly export csv '$TEMP_DIR/test_results.json' '$TEMP_DIR/exported_results.csv'"
    
    # Test Excel export (if dependencies available)
    run_test "Export to Excel" "pynomaly export excel '$TEMP_DIR/test_results.json' '$TEMP_DIR/exported_results.xlsx'" || true
}

# Performance tests
test_performance() {
    log "Testing performance with medium dataset..."
    
    # Load medium dataset
    run_test "Load Medium Dataset" "pynomaly dataset load '$TEST_DATA_DIR/medium_data.csv' --name test_medium"
    
    # Create detector for medium dataset
    run_test "Create Detector for Medium Data" "pynomaly detector create --name medium_detector --algorithm IsolationForest"
    
    # Time the training process
    local start_time=$(date +%s)
    run_test "Train on Medium Dataset" "pynomaly detect train --detector medium_detector --dataset test_medium"
    local end_time=$(date +%s)
    local training_duration=$((end_time - start_time))
    
    log "Training duration: ${training_duration}s"
    
    # Time the detection process
    start_time=$(date +%s)
    run_test "Detect on Medium Dataset" "pynomaly detect run --detector medium_detector --dataset test_medium"
    end_time=$(date +%s)
    local detection_duration=$((end_time - start_time))
    
    log "Detection duration: ${detection_duration}s"
    
    # Performance assertions
    if [ "$training_duration" -gt 120 ]; then
        log_warning "Training took longer than expected: ${training_duration}s"
    fi
    
    if [ "$detection_duration" -gt 60 ]; then
        log_warning "Detection took longer than expected: ${detection_duration}s"
    fi
}

# Error handling tests
test_error_handling() {
    log "Testing error handling..."
    
    # Test non-existent dataset
    run_test "Load Non-existent File" "pynomaly dataset load '/non/existent/file.csv' --name test_missing" 1
    
    # Test invalid algorithm
    run_test "Create Invalid Algorithm Detector" "pynomaly detector create --name bad_detector --algorithm NonExistentAlgorithm" 1
    
    # Test train with non-existent detector
    run_test "Train Non-existent Detector" "pynomaly detect train --detector non_existent --dataset test_small" 1
    
    # Test train with non-existent dataset
    run_test "Train with Non-existent Dataset" "pynomaly detect train --detector test_detector --dataset non_existent" 1
}

# Resource cleanup
cleanup() {
    log "Cleaning up test environment..."
    
    # Deactivate virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    
    # Remove temporary directory
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
    
    log "Cleanup complete"
}

# Generate test report
generate_report() {
    log "Generating test report..."
    
    # Create JSON report
    cat > "$RESULTS_JSON" << EOF
{
    "test_run": {
        "timestamp": "$(date -Iseconds)",
        "environment": "bash",
        "platform": "$(uname -s)",
        "python_version": "$(python3 --version 2>&1)",
        "total_tests": $TOTAL_TESTS,
        "passed_tests": $PASSED_TESTS,
        "failed_tests": $FAILED_TESTS,
        "success_rate": $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)
    },
    "test_results": [
        $(IFS=','; echo "${TEST_RESULTS[*]}")
    ]
}
EOF
    
    # Print summary
    echo ""
    echo "=============================================="
    echo "           PYNOMALY CLI TEST SUMMARY"
    echo "=============================================="
    echo "Total Tests:    $TOTAL_TESTS"
    echo "Passed:         $PASSED_TESTS"
    echo "Failed:         $FAILED_TESTS"
    echo "Success Rate:   $(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)%"
    echo "=============================================="
    echo ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}❌ Some tests failed. Check the log for details.${NC}"
        echo "Log file: $LOG_FILE"
        echo "Results file: $RESULTS_JSON"
        return 1
    fi
}

# Main execution
main() {
    log "Starting Pynomaly CLI tests in Bash environment"
    log "Platform: $(uname -s) $(uname -r)"
    log "Python: $(python3 --version 2>&1)"
    
    # Setup
    setup_test_environment
    generate_test_data
    install_pynomaly
    
    # Run tests
    test_basic_commands
    test_dataset_commands
    test_detector_commands
    test_detection_commands
    test_autonomous_commands
    test_export_commands
    test_performance
    test_error_handling
    
    # Report and cleanup
    generate_report
    local exit_code=$?
    
    cleanup
    exit $exit_code
}

# Trap cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"
