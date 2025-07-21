#!/bin/bash

# Comprehensive Test Suite Runner for anomaly_detection
# This script runs all test categories with proper reporting

set -euo pipefail

# Configuration
TEST_DIR="tests"
VENV_DIR="environments/.venv"
COVERAGE_DIR="reports/coverage"
REPORT_DIR="reports/test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
SKIPPED_TESTS=0

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Setup test environment
setup_test_environment() {
    log "Setting up test environment..."

    # Activate virtual environment
    source ${VENV_DIR}/bin/activate

    # Create report directories
    mkdir -p ${COVERAGE_DIR}
    mkdir -p ${REPORT_DIR}

    # Install test dependencies
    pip install -q pytest pytest-cov pytest-html pytest-xdist pytest-mock pytest-asyncio

    # Install the package in development mode
    pip install -e .

    success "Test environment setup completed"
}

# Run unit tests
run_unit_tests() {
    log "Running unit tests..."

    pytest tests/unit/ \
        --cov=src/anomaly_detection \
        --cov-report=html:${COVERAGE_DIR}/unit \
        --cov-report=xml:${COVERAGE_DIR}/unit_coverage.xml \
        --cov-report=term-missing \
        --html=${REPORT_DIR}/unit_test_report_${TIMESTAMP}.html \
        --junitxml=${REPORT_DIR}/unit_test_results_${TIMESTAMP}.xml \
        --tb=short \
        --disable-warnings \
        -v || {
        error "Unit tests failed"
        return 1
    }

    success "Unit tests completed"
}

# Run integration tests
run_integration_tests() {
    log "Running integration tests..."

    pytest tests/integration/ \
        --cov=src/anomaly_detection \
        --cov-append \
        --cov-report=html:${COVERAGE_DIR}/integration \
        --cov-report=xml:${COVERAGE_DIR}/integration_coverage.xml \
        --html=${REPORT_DIR}/integration_test_report_${TIMESTAMP}.html \
        --junitxml=${REPORT_DIR}/integration_test_results_${TIMESTAMP}.xml \
        --tb=short \
        --disable-warnings \
        -v || {
        warning "Integration tests failed (some may require external services)"
        return 1
    }

    success "Integration tests completed"
}

# Run API tests
run_api_tests() {
    log "Running API tests..."

    pytest tests/api/ \
        --cov=src/anomaly_detection \
        --cov-append \
        --cov-report=html:${COVERAGE_DIR}/api \
        --cov-report=xml:${COVERAGE_DIR}/api_coverage.xml \
        --html=${REPORT_DIR}/api_test_report_${TIMESTAMP}.html \
        --junitxml=${REPORT_DIR}/api_test_results_${TIMESTAMP}.xml \
        --tb=short \
        --disable-warnings \
        -v || {
        warning "API tests failed (may require running services)"
        return 1
    }

    success "API tests completed"
}

# Run CLI tests
run_cli_tests() {
    log "Running CLI tests..."

    pytest tests/cli/ \
        --cov=src/anomaly_detection \
        --cov-append \
        --cov-report=html:${COVERAGE_DIR}/cli \
        --cov-report=xml:${COVERAGE_DIR}/cli_coverage.xml \
        --html=${REPORT_DIR}/cli_test_report_${TIMESTAMP}.html \
        --junitxml=${REPORT_DIR}/cli_test_results_${TIMESTAMP}.xml \
        --tb=short \
        --disable-warnings \
        -v || {
        warning "CLI tests failed"
        return 1
    }

    success "CLI tests completed"
}

# Run performance tests
run_performance_tests() {
    log "Running performance tests..."

    pytest tests/performance/ \
        --benchmark-only \
        --benchmark-sort=mean \
        --benchmark-html=${REPORT_DIR}/performance_report_${TIMESTAMP}.html \
        --benchmark-json=${REPORT_DIR}/performance_results_${TIMESTAMP}.json \
        --tb=short \
        --disable-warnings \
        -v || {
        warning "Performance tests failed"
        return 1
    }

    success "Performance tests completed"
}

# Run security tests
run_security_tests() {
    log "Running security tests..."

    # Run Bandit security scan
    bandit -r src/anomaly_detection/ -f json -o ${REPORT_DIR}/bandit_report_${TIMESTAMP}.json || {
        warning "Bandit security scan found issues"
    }

    # Run Safety vulnerability scan
    safety check --json --output ${REPORT_DIR}/safety_report_${TIMESTAMP}.json || {
        warning "Safety vulnerability scan found issues"
    }

    success "Security tests completed"
}

# Run code quality tests
run_code_quality_tests() {
    log "Running code quality tests..."

    # Run Ruff linting
    ruff check src/anomaly_detection/ --output-format=json --output-file=${REPORT_DIR}/ruff_report_${TIMESTAMP}.json || {
        warning "Ruff linting found issues"
    }

    # Run MyPy type checking
    mypy src/anomaly_detection/ --junit-xml=${REPORT_DIR}/mypy_report_${TIMESTAMP}.xml || {
        warning "MyPy type checking found issues"
    }

    success "Code quality tests completed"
}

# Generate combined coverage report
generate_coverage_report() {
    log "Generating combined coverage report..."

    # Combine all coverage data
    coverage combine

    # Generate HTML report
    coverage html -d ${COVERAGE_DIR}/combined

    # Generate XML report
    coverage xml -o ${COVERAGE_DIR}/combined_coverage.xml

    # Generate text report
    coverage report > ${REPORT_DIR}/coverage_summary_${TIMESTAMP}.txt

    success "Combined coverage report generated"
}

# Parse test results
parse_test_results() {
    log "Parsing test results..."

    # Count results from XML files
    for xml_file in ${REPORT_DIR}/*_test_results_${TIMESTAMP}.xml; do
        if [[ -f "$xml_file" ]]; then
            local tests=$(grep -o 'tests="[0-9]*"' "$xml_file" | grep -o '[0-9]*' | head -1)
            local failures=$(grep -o 'failures="[0-9]*"' "$xml_file" | grep -o '[0-9]*' | head -1)
            local errors=$(grep -o 'errors="[0-9]*"' "$xml_file" | grep -o '[0-9]*' | head -1)
            local skipped=$(grep -o 'skipped="[0-9]*"' "$xml_file" | grep -o '[0-9]*' | head -1)

            TOTAL_TESTS=$((TOTAL_TESTS + ${tests:-0}))
            FAILED_TESTS=$((FAILED_TESTS + ${failures:-0} + ${errors:-0}))
            SKIPPED_TESTS=$((SKIPPED_TESTS + ${skipped:-0}))
        fi
    done

    PASSED_TESTS=$((TOTAL_TESTS - FAILED_TESTS - SKIPPED_TESTS))

    success "Test results parsed"
}

# Generate final report
generate_final_report() {
    log "Generating final test report..."

    local REPORT_FILE="${REPORT_DIR}/comprehensive_test_report_${TIMESTAMP}.md"

    cat > "$REPORT_FILE" << EOF
# Comprehensive Test Report

**Date:** $(date)
**Test Suite Version:** 1.0.0

## Summary

- **Total Tests:** ${TOTAL_TESTS}
- **Passed:** ${PASSED_TESTS}
- **Failed:** ${FAILED_TESTS}
- **Skipped:** ${SKIPPED_TESTS}
- **Success Rate:** $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%

## Test Categories

### Unit Tests
- **Coverage:** $(coverage report --show-missing | grep TOTAL | awk '{print $4}')
- **Files:** Unit test results in \`${REPORT_DIR}/unit_test_report_${TIMESTAMP}.html\`

### Integration Tests
- **Status:** $([ -f "${REPORT_DIR}/integration_test_results_${TIMESTAMP}.xml" ] && echo "Completed" || echo "Failed")
- **Files:** Integration test results in \`${REPORT_DIR}/integration_test_report_${TIMESTAMP}.html\`

### API Tests
- **Status:** $([ -f "${REPORT_DIR}/api_test_results_${TIMESTAMP}.xml" ] && echo "Completed" || echo "Failed")
- **Files:** API test results in \`${REPORT_DIR}/api_test_report_${TIMESTAMP}.html\`

### CLI Tests
- **Status:** $([ -f "${REPORT_DIR}/cli_test_results_${TIMESTAMP}.xml" ] && echo "Completed" || echo "Failed")
- **Files:** CLI test results in \`${REPORT_DIR}/cli_test_report_${TIMESTAMP}.html\`

### Performance Tests
- **Status:** $([ -f "${REPORT_DIR}/performance_results_${TIMESTAMP}.json" ] && echo "Completed" || echo "Failed")
- **Files:** Performance results in \`${REPORT_DIR}/performance_report_${TIMESTAMP}.html\`

### Security Tests
- **Bandit:** $([ -f "${REPORT_DIR}/bandit_report_${TIMESTAMP}.json" ] && echo "Completed" || echo "Failed")
- **Safety:** $([ -f "${REPORT_DIR}/safety_report_${TIMESTAMP}.json" ] && echo "Completed" || echo "Failed")

### Code Quality
- **Ruff:** $([ -f "${REPORT_DIR}/ruff_report_${TIMESTAMP}.json" ] && echo "Completed" || echo "Failed")
- **MyPy:** $([ -f "${REPORT_DIR}/mypy_report_${TIMESTAMP}.xml" ] && echo "Completed" || echo "Failed")

## Coverage Report

- **Combined Coverage:** Available in \`${COVERAGE_DIR}/combined/index.html\`
- **Coverage Summary:** Available in \`${REPORT_DIR}/coverage_summary_${TIMESTAMP}.txt\`

## Recommendations

EOF

    # Add recommendations based on results
    if [ $FAILED_TESTS -gt 0 ]; then
        echo "- **HIGH PRIORITY**: Fix ${FAILED_TESTS} failing tests before deployment" >> "$REPORT_FILE"
    fi

    if [ $SKIPPED_TESTS -gt 0 ]; then
        echo "- **MEDIUM PRIORITY**: Review ${SKIPPED_TESTS} skipped tests" >> "$REPORT_FILE"
    fi

    local SUCCESS_RATE=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))
    if [ $SUCCESS_RATE -lt 90 ]; then
        echo "- **HIGH PRIORITY**: Improve test success rate (currently ${SUCCESS_RATE}%)" >> "$REPORT_FILE"
    fi

    echo "- **ONGOING**: Maintain test coverage above 80%" >> "$REPORT_FILE"
    echo "- **ONGOING**: Regular security scans and code quality checks" >> "$REPORT_FILE"

    success "Final test report generated: $REPORT_FILE"
}

# Display results
display_results() {
    log "Test Suite Results:"
    echo ""
    echo "📊 Test Statistics:"
    echo "   Total Tests: ${TOTAL_TESTS}"
    echo "   Passed: ${PASSED_TESTS}"
    echo "   Failed: ${FAILED_TESTS}"
    echo "   Skipped: ${SKIPPED_TESTS}"
    echo "   Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    echo ""
    echo "📁 Reports Generated:"
    echo "   Coverage: ${COVERAGE_DIR}/combined/index.html"
    echo "   Test Results: ${REPORT_DIR}/comprehensive_test_report_${TIMESTAMP}.md"
    echo "   Performance: ${REPORT_DIR}/performance_report_${TIMESTAMP}.html"
    echo ""

    if [ $FAILED_TESTS -eq 0 ]; then
        success "All tests passed! 🎉"
    else
        error "${FAILED_TESTS} tests failed. Please review the reports."
    fi
}

# Main execution
main() {
    local TEST_TYPE=${1:-all}

    log "Starting comprehensive test suite..."

    case $TEST_TYPE in
        unit)
            setup_test_environment
            run_unit_tests
            ;;
        integration)
            setup_test_environment
            run_integration_tests
            ;;
        api)
            setup_test_environment
            run_api_tests
            ;;
        cli)
            setup_test_environment
            run_cli_tests
            ;;
        performance)
            setup_test_environment
            run_performance_tests
            ;;
        security)
            setup_test_environment
            run_security_tests
            ;;
        quality)
            setup_test_environment
            run_code_quality_tests
            ;;
        all)
            setup_test_environment

            # Run all test categories
            run_unit_tests || true
            run_integration_tests || true
            run_api_tests || true
            run_cli_tests || true
            run_performance_tests || true
            run_security_tests || true
            run_code_quality_tests || true

            # Generate reports
            generate_coverage_report || true
            parse_test_results
            generate_final_report
            display_results
            ;;
        *)
            echo "Usage: $0 {unit|integration|api|cli|performance|security|quality|all}"
            echo ""
            echo "Test Types:"
            echo "  unit        - Run unit tests only"
            echo "  integration - Run integration tests only"
            echo "  api         - Run API tests only"
            echo "  cli         - Run CLI tests only"
            echo "  performance - Run performance tests only"
            echo "  security    - Run security tests only"
            echo "  quality     - Run code quality tests only"
            echo "  all         - Run all test categories (default)"
            exit 1
            ;;
    esac

    success "Test suite completed!"
}

# Run main function
main "$@"
