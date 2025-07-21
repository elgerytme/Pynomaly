#!/bin/bash

# Performance Testing Script for anomaly_detection
# This script runs comprehensive performance tests and optimization validation

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_ROOT/reports/performance"
CONFIG_FILE="$PROJECT_ROOT/config/performance_optimization.yaml"
PYTHON_ENV="${PYTHON_ENV:-python3}"

# Default values
TEST_TYPE="comprehensive"
DURATION=300
USERS=100
TARGET_URL="http://localhost:8000"
ENVIRONMENT="staging"
OPTIMIZATION_STRATEGIES=""
DRY_RUN=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
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

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run comprehensive performance tests and optimization validation for anomaly_detection.

OPTIONS:
    -t, --type TYPE              Test type (load, stress, endurance, comprehensive) [default: comprehensive]
    -d, --duration SECONDS       Test duration in seconds [default: 300]
    -u, --users NUMBER          Number of concurrent users [default: 100]
    -e, --environment ENV       Environment (development, staging, production) [default: staging]
    -o, --optimize STRATEGIES   Comma-separated optimization strategies to apply
    -c, --config FILE           Configuration file path [default: config/performance_optimization.yaml]
    --target-url URL            Target URL for testing [default: http://localhost:8000]
    --dry-run                   Run optimization in dry-run mode
    --verbose                   Verbose output
    -h, --help                  Show this help message

EXAMPLES:
    # Run comprehensive performance test
    $0 -t comprehensive -d 600 -u 200

    # Run load test with optimization
    $0 -t load -o "caching,database,memory" -u 150

    # Run stress test in dry-run mode
    $0 -t stress --dry-run --verbose

    # Run endurance test for production
    $0 -t endurance -e production -d 3600

OPTIMIZATION STRATEGIES:
    - caching: Optimize caching strategies
    - database: Optimize database performance
    - memory: Optimize memory usage
    - cpu: Optimize CPU usage
    - network: Optimize network performance
    - algorithms: Optimize algorithms and data structures
    - resources: Optimize resource allocation
    - monitoring: Optimize monitoring and observability

EOF
}

# Function to parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                TEST_TYPE="$2"
                shift 2
                ;;
            -d|--duration)
                DURATION="$2"
                shift 2
                ;;
            -u|--users)
                USERS="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -o|--optimize)
                OPTIMIZATION_STRATEGIES="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --target-url)
                TARGET_URL="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to validate dependencies
validate_dependencies() {
    print_info "Validating dependencies..."
    
    # Check Python
    if ! command -v $PYTHON_ENV &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check required Python packages
    local required_packages=(
        "requests"
        "pyyaml"
        "psutil"
        "locust"
        "prometheus_client"
    )
    
    for package in "${required_packages[@]}"; do
        if ! $PYTHON_ENV -c "import $package" &> /dev/null; then
            print_error "Python package '$package' not found. Please install it."
            exit 1
        fi
    done
    
    # Check if target URL is accessible
    if ! curl -s --head "$TARGET_URL/health" &> /dev/null; then
        print_warning "Target URL $TARGET_URL may not be accessible"
    fi
    
    print_success "Dependencies validated"
}

# Function to create reports directory
setup_reports_dir() {
    print_info "Setting up reports directory..."
    mkdir -p "$REPORTS_DIR"
    
    # Create subdirectories
    mkdir -p "$REPORTS_DIR/load_tests"
    mkdir -p "$REPORTS_DIR/stress_tests"
    mkdir -p "$REPORTS_DIR/endurance_tests"
    mkdir -p "$REPORTS_DIR/optimization"
    
    print_success "Reports directory created: $REPORTS_DIR"
}

# Function to collect baseline metrics
collect_baseline_metrics() {
    print_info "Collecting baseline metrics..."
    
    local baseline_file="$REPORTS_DIR/baseline_metrics_$(date +%Y%m%d_%H%M%S).json"
    
    $PYTHON_ENV "$PROJECT_ROOT/tests/performance/performance_validation.py" \
        --config "$CONFIG_FILE" \
        --test-type load \
        --duration 60 \
        --users 10 \
        --output "$baseline_file" \
        ${VERBOSE:+--verbose}
    
    print_success "Baseline metrics collected: $baseline_file"
    echo "$baseline_file"
}

# Function to run optimization
run_optimization() {
    local strategies="$1"
    
    if [[ -z "$strategies" ]]; then
        print_info "No optimization strategies specified, skipping optimization"
        return
    fi
    
    print_info "Running optimization strategies: $strategies"
    
    local optimization_args=(
        "--config" "$CONFIG_FILE"
        "--strategies" ${strategies//,/ }
    )
    
    if [[ "$DRY_RUN" == "true" ]]; then
        optimization_args+=("--dry-run")
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        optimization_args+=("--verbose")
    fi
    
    $PYTHON_ENV "$PROJECT_ROOT/scripts/performance_optimization.py" "${optimization_args[@]}"
    
    print_success "Optimization completed"
}

# Function to run performance tests
run_performance_tests() {
    print_info "Running performance tests..."
    
    local test_args=(
        "--config" "$CONFIG_FILE"
        "--test-type" "$TEST_TYPE"
        "--duration" "$DURATION"
        "--users" "$USERS"
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        test_args+=("--verbose")
    fi
    
    # Generate output filename
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file="$REPORTS_DIR/${TEST_TYPE}_test_${timestamp}.json"
    test_args+=("--output" "$output_file")
    
    # Run the test
    $PYTHON_ENV "$PROJECT_ROOT/tests/performance/performance_validation.py" "${test_args[@]}"
    
    print_success "Performance tests completed: $output_file"
    echo "$output_file"
}

# Function to run load tests using Locust
run_locust_load_test() {
    print_info "Running Locust load test..."
    
    local locust_file="$PROJECT_ROOT/tests/load/locustfile.py"
    local locust_args=(
        "-f" "$locust_file"
        "--host" "$TARGET_URL"
        "--users" "$USERS"
        "--spawn-rate" "10"
        "--run-time" "${DURATION}s"
        "--html" "$REPORTS_DIR/locust_report_$(date +%Y%m%d_%H%M%S).html"
        "--csv" "$REPORTS_DIR/locust_stats_$(date +%Y%m%d_%H%M%S)"
        "--headless"
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        locust "${locust_args[@]}"
    else
        locust "${locust_args[@]}" > /dev/null 2>&1
    fi
    
    print_success "Locust load test completed"
}

# Function to analyze results
analyze_results() {
    local results_file="$1"
    
    print_info "Analyzing performance results..."
    
    if [[ ! -f "$results_file" ]]; then
        print_error "Results file not found: $results_file"
        return 1
    fi
    
    # Extract key metrics using jq if available
    if command -v jq &> /dev/null; then
        local throughput=$(jq -r '.load_test.throughput_rps // .throughput_rps // "N/A"' "$results_file")
        local response_time_p95=$(jq -r '.load_test.response_time_p95 // .response_time_p95 // "N/A"' "$results_file")
        local error_rate=$(jq -r '.load_test.error_rate // .error_rate // "N/A"' "$results_file")
        
        print_info "Performance Summary:"
        echo "  Throughput: $throughput RPS"
        echo "  Response Time P95: $response_time_p95 ms"
        echo "  Error Rate: $error_rate"
        
        # Check if thresholds are met
        if command -v bc &> /dev/null; then
            local throughput_threshold=100
            local response_time_threshold=500
            local error_rate_threshold=0.01
            
            if [[ "$throughput" != "N/A" ]] && (( $(echo "$throughput > $throughput_threshold" | bc -l) )); then
                print_success "Throughput threshold met"
            else
                print_warning "Throughput threshold not met"
            fi
            
            if [[ "$response_time_p95" != "N/A" ]] && (( $(echo "$response_time_p95 < $response_time_threshold" | bc -l) )); then
                print_success "Response time threshold met"
            else
                print_warning "Response time threshold not met"
            fi
            
            if [[ "$error_rate" != "N/A" ]] && (( $(echo "$error_rate < $error_rate_threshold" | bc -l) )); then
                print_success "Error rate threshold met"
            else
                print_warning "Error rate threshold not met"
            fi
        fi
    else
        print_warning "jq not available, skipping detailed analysis"
    fi
}

# Function to generate summary report
generate_summary_report() {
    print_info "Generating summary report..."
    
    local summary_file="$REPORTS_DIR/performance_summary_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$summary_file" << EOF
anomaly_detection PERFORMANCE TEST SUMMARY
=================================

Test Configuration:
  - Test Type: $TEST_TYPE
  - Duration: $DURATION seconds
  - Concurrent Users: $USERS
  - Target URL: $TARGET_URL
  - Environment: $ENVIRONMENT
  - Optimization Strategies: ${OPTIMIZATION_STRATEGIES:-"None"}
  - Dry Run: $DRY_RUN
  - Timestamp: $(date)

Test Results:
  - Reports Directory: $REPORTS_DIR
  - Configuration File: $CONFIG_FILE

Next Steps:
  1. Review detailed test results in the reports directory
  2. Analyze performance metrics and identify bottlenecks
  3. Apply optimization strategies as needed
  4. Re-run tests to validate improvements
  5. Monitor performance in production

EOF
    
    print_success "Summary report generated: $summary_file"
}

# Function to run comprehensive performance testing
run_comprehensive_testing() {
    print_info "Starting comprehensive performance testing..."
    
    # Collect baseline metrics
    local baseline_file
    baseline_file=$(collect_baseline_metrics)
    
    # Run optimization if strategies are specified
    if [[ -n "$OPTIMIZATION_STRATEGIES" ]]; then
        run_optimization "$OPTIMIZATION_STRATEGIES"
        
        # Wait for optimizations to take effect
        print_info "Waiting for optimizations to take effect..."
        sleep 30
    fi
    
    # Run performance tests
    local results_file
    results_file=$(run_performance_tests)
    
    # Run additional Locust load test for comparison
    if [[ "$TEST_TYPE" == "comprehensive" || "$TEST_TYPE" == "load" ]]; then
        run_locust_load_test
    fi
    
    # Analyze results
    analyze_results "$results_file"
    
    # Generate summary report
    generate_summary_report
    
    print_success "Comprehensive performance testing completed"
}

# Main function
main() {
    echo "======================================"
    echo "anomaly_detection Performance Testing Script"
    echo "======================================"
    
    # Parse command line arguments
    parse_args "$@"
    
    # Validate dependencies
    validate_dependencies
    
    # Setup reports directory
    setup_reports_dir
    
    # Run comprehensive testing
    run_comprehensive_testing
    
    echo ""
    echo "======================================"
    echo "Performance Testing Complete"
    echo "======================================"
    echo ""
    echo "Results available in: $REPORTS_DIR"
    echo ""
    
    # Exit with appropriate code based on results
    if [[ -f "$REPORTS_DIR/performance_summary_"*.txt ]]; then
        print_success "Performance testing completed successfully"
        exit 0
    else
        print_error "Performance testing may have failed"
        exit 1
    fi
}

# Handle script termination
trap 'print_error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"