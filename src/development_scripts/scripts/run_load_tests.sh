#!/bin/bash

# Comprehensive Load Testing Script for anomaly_detection
# This script runs various load testing scenarios using Locust

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOAD_TEST_DIR="${PROJECT_ROOT}/tests/load"
REPORTS_DIR="${PROJECT_ROOT}/reports/load_tests"
STAGING_HOST="http://localhost:8000"
PRODUCTION_HOST="https://api.anomaly_detection.com"
DEFAULT_USERS=50
DEFAULT_SPAWN_RATE=10
DEFAULT_DURATION="5m"
DEFAULT_SCENARIO="basic"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run comprehensive load tests for anomaly_detection

OPTIONS:
    -h, --help              Show this help message
    -s, --scenario SCENARIO Test scenario (basic, comprehensive, stress, endurance)
    -u, --users NUMBER      Number of concurrent users (default: $DEFAULT_USERS)
    -r, --spawn-rate NUMBER User spawn rate per second (default: $DEFAULT_SPAWN_RATE)
    -t, --duration TIME     Test duration (e.g., 5m, 30s, 1h) (default: $DEFAULT_DURATION)
    -H, --host URL          Target host URL (default: $STAGING_HOST)
    --staging               Use staging environment
    --production            Use production environment
    --headless              Run in headless mode (no web UI)
    --distributed           Run distributed load testing
    --master                Run as master node (for distributed testing)
    --worker                Run as worker node (for distributed testing)
    --csv PREFIX            Save results as CSV with given prefix
    --html FILE             Save HTML report to file
    --log-level LEVEL       Set log level (DEBUG, INFO, WARNING, ERROR)
    --expect-workers COUNT  Number of expected worker nodes
    --master-host HOST      Master node host for distributed testing
    --master-port PORT      Master node port for distributed testing
    --tags TAGS             Only run tasks with these tags
    --exclude-tags TAGS     Exclude tasks with these tags
    --stop-timeout SECONDS  Stop timeout in seconds
    --reset-stats           Reset stats between test runs
    --autostart             Start test automatically
    --autoquit SECONDS      Quit after specified seconds
    --config FILE           Load configuration from file
    --dry-run               Show what would be executed without running

SCENARIOS:
    basic         Basic load test (50 users, 5 minutes)
    comprehensive Full feature test (100 users, 30 minutes)
    stress        Stress test (200 users, 10 minutes)
    endurance     Endurance test (25 users, 60 minutes)
    spike         Spike test (variable load)
    soak          Soak test (extended duration)
    volume        Volume test (high data throughput)
    api           API-focused test
    ui            UI-focused test

EXAMPLES:
    $0                                    # Run basic test
    $0 -s comprehensive -u 100 -t 30m     # Comprehensive test
    $0 -s stress --staging --headless     # Stress test on staging
    $0 --distributed --master             # Run as master node
    $0 --distributed --worker             # Run as worker node
    $0 --csv results --html report.html   # Save results to files
    $0 --config load_test.conf            # Load from config file
EOF
}

# Configuration variables
SCENARIO="$DEFAULT_SCENARIO"
USERS="$DEFAULT_USERS"
SPAWN_RATE="$DEFAULT_SPAWN_RATE"
DURATION="$DEFAULT_DURATION"
HOST="$STAGING_HOST"
HEADLESS=false
DISTRIBUTED=false
MASTER=false
WORKER=false
CSV_PREFIX=""
HTML_FILE=""
LOG_LEVEL="INFO"
EXPECT_WORKERS=1
MASTER_HOST="localhost"
MASTER_PORT=5557
TAGS=""
EXCLUDE_TAGS=""
STOP_TIMEOUT=10
RESET_STATS=false
AUTOSTART=false
AUTOQUIT=0
CONFIG_FILE=""
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--scenario)
            SCENARIO="$2"
            shift 2
            ;;
        -u|--users)
            USERS="$2"
            shift 2
            ;;
        -r|--spawn-rate)
            SPAWN_RATE="$2"
            shift 2
            ;;
        -t|--duration)
            DURATION="$2"
            shift 2
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        --staging)
            HOST="$STAGING_HOST"
            shift
            ;;
        --production)
            HOST="$PRODUCTION_HOST"
            shift
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --distributed)
            DISTRIBUTED=true
            shift
            ;;
        --master)
            MASTER=true
            DISTRIBUTED=true
            shift
            ;;
        --worker)
            WORKER=true
            DISTRIBUTED=true
            shift
            ;;
        --csv)
            CSV_PREFIX="$2"
            shift 2
            ;;
        --html)
            HTML_FILE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --expect-workers)
            EXPECT_WORKERS="$2"
            shift 2
            ;;
        --master-host)
            MASTER_HOST="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --tags)
            TAGS="$2"
            shift 2
            ;;
        --exclude-tags)
            EXCLUDE_TAGS="$2"
            shift 2
            ;;
        --stop-timeout)
            STOP_TIMEOUT="$2"
            shift 2
            ;;
        --reset-stats)
            RESET_STATS=true
            shift
            ;;
        --autostart)
            AUTOSTART=true
            shift
            ;;
        --autoquit)
            AUTOQUIT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Load configuration from file if specified
load_config() {
    if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
        log_info "Loading configuration from: $CONFIG_FILE"
        source "$CONFIG_FILE"
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if Locust is installed
    if ! command -v locust &> /dev/null; then
        log_error "Locust is not installed. Install with: pip install locust"
        exit 1
    fi

    # Check if locustfile exists
    if [[ ! -f "$LOAD_TEST_DIR/locustfile.py" ]]; then
        log_error "Locustfile not found: $LOAD_TEST_DIR/locustfile.py"
        exit 1
    fi

    # Create reports directory
    mkdir -p "$REPORTS_DIR"

    # Check target host connectivity
    if [[ "$DRY_RUN" == "false" ]]; then
        log_info "Testing connectivity to: $HOST"
        if ! curl -f -s --max-time 10 "$HOST/health" > /dev/null 2>&1; then
            log_warning "Cannot reach $HOST/health - target may not be ready"
        else
            log_success "Target host is reachable"
        fi
    fi

    log_success "Prerequisites check passed"
}

# Build Locust command
build_locust_command() {
    local cmd="locust"

    # Basic options
    cmd+=" -f $LOAD_TEST_DIR/locustfile.py"
    cmd+=" --host $HOST"
    cmd+=" --users $USERS"
    cmd+=" --spawn-rate $SPAWN_RATE"
    cmd+=" --run-time $DURATION"
    cmd+=" --loglevel $LOG_LEVEL"
    cmd+=" --stop-timeout $STOP_TIMEOUT"

    # Headless mode
    if [[ "$HEADLESS" == "true" ]]; then
        cmd+=" --headless"
    fi

    # Distributed testing
    if [[ "$DISTRIBUTED" == "true" ]]; then
        if [[ "$MASTER" == "true" ]]; then
            cmd+=" --master"
            cmd+=" --expect-workers $EXPECT_WORKERS"
        elif [[ "$WORKER" == "true" ]]; then
            cmd+=" --worker"
            cmd+=" --master-host $MASTER_HOST"
            cmd+=" --master-port $MASTER_PORT"
        fi
    fi

    # Output options
    if [[ -n "$CSV_PREFIX" ]]; then
        cmd+=" --csv $REPORTS_DIR/$CSV_PREFIX"
    fi

    if [[ -n "$HTML_FILE" ]]; then
        cmd+=" --html $REPORTS_DIR/$HTML_FILE"
    fi

    # Tags
    if [[ -n "$TAGS" ]]; then
        cmd+=" --tags $TAGS"
    fi

    if [[ -n "$EXCLUDE_TAGS" ]]; then
        cmd+=" --exclude-tags $EXCLUDE_TAGS"
    fi

    # Additional options
    if [[ "$RESET_STATS" == "true" ]]; then
        cmd+=" --reset-stats"
    fi

    if [[ "$AUTOSTART" == "true" ]]; then
        cmd+=" --autostart"
    fi

    if [[ "$AUTOQUIT" -gt 0 ]]; then
        cmd+=" --autoquit $AUTOQUIT"
    fi

    echo "$cmd"
}

# Run specific scenario
run_scenario() {
    local scenario="$1"

    log_info "Running scenario: $scenario"

    case "$scenario" in
        "basic")
            USERS=50
            SPAWN_RATE=10
            DURATION="5m"
            ;;
        "comprehensive")
            USERS=100
            SPAWN_RATE=20
            DURATION="30m"
            ;;
        "stress")
            USERS=200
            SPAWN_RATE=50
            DURATION="10m"
            ;;
        "endurance")
            USERS=25
            SPAWN_RATE=5
            DURATION="60m"
            ;;
        "spike")
            run_spike_test
            return
            ;;
        "soak")
            USERS=50
            SPAWN_RATE=10
            DURATION="120m"
            ;;
        "volume")
            USERS=100
            SPAWN_RATE=30
            DURATION="15m"
            ;;
        "api")
            USERS=75
            SPAWN_RATE=15
            DURATION="20m"
            TAGS="api"
            ;;
        "ui")
            USERS=30
            SPAWN_RATE=5
            DURATION="10m"
            TAGS="ui"
            ;;
        *)
            log_error "Unknown scenario: $scenario"
            exit 1
            ;;
    esac

    local cmd=$(build_locust_command)

    log_info "Command: $cmd"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: $cmd"
        return 0
    fi

    # Execute the command
    eval "$cmd"
}

# Run spike test with variable load
run_spike_test() {
    log_info "Running spike test with variable load pattern"

    local phases=(
        "25:5:2m"    # 25 users, 5 spawn rate, 2 minutes
        "100:25:3m"  # 100 users, 25 spawn rate, 3 minutes
        "50:10:2m"   # 50 users, 10 spawn rate, 2 minutes
        "200:50:1m"  # 200 users, 50 spawn rate, 1 minute
        "25:5:2m"    # 25 users, 5 spawn rate, 2 minutes
    )

    for phase in "${phases[@]}"; do
        IFS=':' read -r users spawn_rate duration <<< "$phase"

        log_info "Spike phase: $users users, $spawn_rate spawn rate, $duration duration"

        USERS="$users"
        SPAWN_RATE="$spawn_rate"
        DURATION="$duration"

        local cmd=$(build_locust_command)

        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would execute: $cmd"
        else
            eval "$cmd"
        fi

        # Brief pause between phases
        if [[ "$DRY_RUN" == "false" ]]; then
            sleep 10
        fi
    done
}

# Run multiple scenarios
run_multiple_scenarios() {
    local scenarios=("$@")

    log_info "Running multiple scenarios: ${scenarios[*]}"

    for scenario in "${scenarios[@]}"; do
        log_info "Starting scenario: $scenario"
        run_scenario "$scenario"

        if [[ "$DRY_RUN" == "false" ]]; then
            log_info "Scenario $scenario completed. Waiting 30 seconds before next scenario..."
            sleep 30
        fi
    done
}

# Generate test report
generate_report() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would generate test report"
        return 0
    fi

    log_info "Generating test report..."

    local report_file="$REPORTS_DIR/load_test_summary_$(date +%Y%m%d_%H%M%S).md"

    cat > "$report_file" << EOF
# Load Test Report

## Test Configuration
- **Scenario**: $SCENARIO
- **Users**: $USERS
- **Spawn Rate**: $SPAWN_RATE
- **Duration**: $DURATION
- **Target Host**: $HOST
- **Timestamp**: $(date)

## Test Results

### Summary
- Test execution completed successfully
- Results saved to: $REPORTS_DIR/

### Files Generated
EOF

    # List generated files
    find "$REPORTS_DIR" -name "*$(date +%Y%m%d)*" -type f >> "$report_file"

    log_success "Test report generated: $report_file"
}

# Cleanup function
cleanup() {
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        log_error "Load test failed with exit code $exit_code"
    fi

    # Kill any remaining Locust processes
    if pgrep -f "locust" > /dev/null; then
        log_info "Cleaning up Locust processes..."
        pkill -f "locust" || true
    fi

    exit $exit_code
}

# Pre-test health check
pre_test_health_check() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform pre-test health check"
        return 0
    fi

    log_info "Performing pre-test health check..."

    # Check application health
    local health_check=$(curl -f -s --max-time 10 "$HOST/health" 2>/dev/null || echo "FAILED")

    if [[ "$health_check" == "FAILED" ]]; then
        log_error "Target application health check failed"
        exit 1
    fi

    # Check system resources
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')

    log_info "System resources before test:"
    log_info "  CPU Usage: ${cpu_usage}%"
    log_info "  Memory Usage: ${memory_usage}%"

    log_success "Pre-test health check passed"
}

# Post-test analysis
post_test_analysis() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform post-test analysis"
        return 0
    fi

    log_info "Performing post-test analysis..."

    # Analyze results if CSV was generated
    if [[ -n "$CSV_PREFIX" ]]; then
        local stats_file="$REPORTS_DIR/${CSV_PREFIX}_stats.csv"

        if [[ -f "$stats_file" ]]; then
            log_info "Analyzing test results..."

            # Extract key metrics
            local total_requests=$(tail -n +2 "$stats_file" | cut -d',' -f2 | awk '{sum+=$1} END {print sum}')
            local total_failures=$(tail -n +2 "$stats_file" | cut -d',' -f3 | awk '{sum+=$1} END {print sum}')
            local avg_response_time=$(tail -n +2 "$stats_file" | cut -d',' -f4 | awk '{sum+=$1; count++} END {print sum/count}')

            log_info "Test Results Summary:"
            log_info "  Total Requests: $total_requests"
            log_info "  Total Failures: $total_failures"
            log_info "  Average Response Time: ${avg_response_time}ms"

            # Calculate failure rate
            if [[ "$total_requests" -gt 0 ]]; then
                local failure_rate=$(echo "scale=2; $total_failures * 100 / $total_requests" | bc -l)
                log_info "  Failure Rate: ${failure_rate}%"
            fi
        fi
    fi

    log_success "Post-test analysis completed"
}

# Main function
main() {
    log_info "Starting anomaly_detection load testing..."

    # Set up error handling
    trap cleanup EXIT

    # Load configuration
    load_config

    # Check prerequisites
    check_prerequisites

    # Pre-test health check
    pre_test_health_check

    # Run the test scenario
    run_scenario "$SCENARIO"

    # Post-test analysis
    post_test_analysis

    # Generate report
    generate_report

    log_success "Load testing completed successfully!"

    if [[ "$DRY_RUN" == "false" ]]; then
        log_info "Next steps:"
        echo "1. Review test results in $REPORTS_DIR"
        echo "2. Analyze performance metrics"
        echo "3. Compare with performance baselines"
        echo "4. Identify bottlenecks and optimization opportunities"
        echo "5. Run security tests if results are satisfactory"
    fi
}

# Run main function
main "$@"
