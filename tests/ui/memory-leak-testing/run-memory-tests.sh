#!/bin/bash

# Memory Leak Testing Script for Pynomaly
# Comprehensive memory testing with detailed reporting

set -e

echo "🧠 Starting Memory Leak Testing Suite for Pynomaly"
echo "=================================================="

# Configuration
BASE_URL=${BASE_URL:-"http://localhost:8000"}
REPORTS_DIR="../../test_reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_RUN_ID="memory_test_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Create reports directory
mkdir -p "$REPORTS_DIR"

# 1. Pre-test environment check
log_info "Checking test environment..."

# Check if server is running
if curl -s "$BASE_URL/api/v1/health" > /dev/null; then
    log_success "Server is running at $BASE_URL"
else
    log_error "Server is not accessible at $BASE_URL"
    log_info "Please start the Pynomaly server before running memory tests"
    exit 1
fi

# Check if browser is available
if ! command -v chromium &> /dev/null && ! command -v google-chrome &> /dev/null; then
    log_warning "Chrome/Chromium not found in PATH. Playwright will download if needed."
fi

# Check available memory
AVAILABLE_MEMORY=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
log_info "Available system memory: ${AVAILABLE_MEMORY}GB"

if (( $(echo "$AVAILABLE_MEMORY < 2" | bc -l) )); then
    log_warning "Low system memory detected. Memory tests may be affected."
fi

# 2. Run memory leak detection tests
log_info "Running memory leak detection tests..."

export TEST_RUN_ID="$TEST_RUN_ID"
export BASE_URL="$BASE_URL"

# Run tests with Playwright
npx playwright test \
    --config=memory-leak.config.ts \
    --project=memory-leak-chrome \
    --reporter=html,json,junit \
    --output-dir="$REPORTS_DIR" \
    --timeout=120000 \
    --workers=1 \
    --retries=1 \
    memory-leak-detection.spec.ts

if [ $? -eq 0 ]; then
    log_success "Memory leak detection tests completed"
else
    log_error "Memory leak detection tests failed"
    MEMORY_TESTS_FAILED=true
fi

# 3. Run performance monitoring tests
log_info "Running performance monitoring tests..."

npx playwright test \
    --config=memory-leak.config.ts \
    --project=performance-monitoring \
    --reporter=html,json,junit \
    --output-dir="$REPORTS_DIR" \
    --timeout=120000 \
    --workers=1 \
    --retries=1 \
    performance-monitoring.spec.ts

if [ $? -eq 0 ]; then
    log_success "Performance monitoring tests completed"
else
    log_error "Performance monitoring tests failed"
    PERFORMANCE_TESTS_FAILED=true
fi

# 4. Generate comprehensive report
log_info "Generating comprehensive memory analysis report..."

# Check if memory leak report exists
MEMORY_REPORT_FILE="$REPORTS_DIR/memory-leak-report.json"
ANALYSIS_REPORT_FILE="$REPORTS_DIR/memory-analysis-comprehensive.json"
RECOMMENDATIONS_FILE="$REPORTS_DIR/memory-performance-recommendations.md"

if [ -f "$MEMORY_REPORT_FILE" ]; then
    # Parse memory leak report
    POTENTIAL_LEAKS=$(jq '.testsWithPotentialLeaks' "$MEMORY_REPORT_FILE" 2>/dev/null || echo "0")
    TOTAL_TESTS=$(jq '.totalTests' "$MEMORY_REPORT_FILE" 2>/dev/null || echo "0")

    log_info "Memory Leak Analysis Results:"
    echo "  - Total tests: $TOTAL_TESTS"
    echo "  - Potential leaks: $POTENTIAL_LEAKS"

    if [ "$POTENTIAL_LEAKS" -gt 0 ]; then
        log_warning "$POTENTIAL_LEAKS potential memory leaks detected!"
    else
        log_success "No memory leaks detected"
    fi
else
    log_warning "Memory leak report not found"
fi

# 5. Check analysis report
if [ -f "$ANALYSIS_REPORT_FILE" ]; then
    RISK_LEVEL=$(jq -r '.riskAssessment.riskLevel' "$ANALYSIS_REPORT_FILE" 2>/dev/null || echo "unknown")
    MEMORY_GROWTH=$(jq -r '.clientSideMemory.totalGrowthMB' "$ANALYSIS_REPORT_FILE" 2>/dev/null || echo "0")
    WARNING_COUNT=$(jq '.clientSideMemory.warnings | length' "$ANALYSIS_REPORT_FILE" 2>/dev/null || echo "0")
    ERROR_COUNT=$(jq '.clientSideMemory.errors | length' "$ANALYSIS_REPORT_FILE" 2>/dev/null || echo "0")

    log_info "Performance Analysis Results:"
    echo "  - Risk level: $RISK_LEVEL"
    echo "  - Memory growth: ${MEMORY_GROWTH}MB"
    echo "  - Warnings: $WARNING_COUNT"
    echo "  - Errors: $ERROR_COUNT"

    case "$RISK_LEVEL" in
        "high")
            log_error "High risk memory issues detected!"
            ;;
        "medium")
            log_warning "Medium risk memory concerns identified"
            ;;
        "low")
            log_success "Low risk - memory usage is within acceptable limits"
            ;;
        *)
            log_warning "Unknown risk level"
            ;;
    esac
else
    log_warning "Comprehensive analysis report not found"
fi

# 6. Generate summary report
log_info "Generating test summary..."

SUMMARY_FILE="$REPORTS_DIR/memory-testing-summary-$TIMESTAMP.json"

cat > "$SUMMARY_FILE" << EOF
{
  "testRunId": "$TEST_RUN_ID",
  "timestamp": "$(date -Iseconds)",
  "environment": {
    "baseUrl": "$BASE_URL",
    "availableMemoryGB": $AVAILABLE_MEMORY,
    "platform": "$(uname -s)",
    "architecture": "$(uname -m)"
  },
  "testResults": {
    "memoryLeakTests": {
      "status": "${MEMORY_TESTS_FAILED:-false}",
      "potentialLeaks": ${POTENTIAL_LEAKS:-0},
      "totalTests": ${TOTAL_TESTS:-0}
    },
    "performanceTests": {
      "status": "${PERFORMANCE_TESTS_FAILED:-false}",
      "riskLevel": "$RISK_LEVEL",
      "memoryGrowthMB": ${MEMORY_GROWTH:-0},
      "warningCount": ${WARNING_COUNT:-0},
      "errorCount": ${ERROR_COUNT:-0}
    }
  },
  "reports": {
    "memoryLeakReport": "$MEMORY_REPORT_FILE",
    "analysisReport": "$ANALYSIS_REPORT_FILE",
    "recommendationsFile": "$RECOMMENDATIONS_FILE",
    "summaryFile": "$SUMMARY_FILE"
  }
}
EOF

log_success "Test summary saved: $SUMMARY_FILE"

# 7. Display final results
echo ""
echo "🧠 Memory Leak Testing Complete"
echo "================================"

if [ "${MEMORY_TESTS_FAILED}" = "true" ] || [ "${PERFORMANCE_TESTS_FAILED}" = "true" ]; then
    log_error "Some tests failed. Please review the reports for details."
    exit 1
elif [ "$POTENTIAL_LEAKS" -gt 0 ] || [ "$RISK_LEVEL" = "high" ]; then
    log_warning "Memory issues detected. Review recommendations for improvements."
    exit 1
else
    log_success "All memory tests passed successfully!"
fi

# 8. Display report locations
echo ""
log_info "Reports generated:"
echo "  📋 Test Summary: $SUMMARY_FILE"
[ -f "$MEMORY_REPORT_FILE" ] && echo "  🔍 Memory Leak Report: $MEMORY_REPORT_FILE"
[ -f "$ANALYSIS_REPORT_FILE" ] && echo "  📊 Analysis Report: $ANALYSIS_REPORT_FILE"
[ -f "$RECOMMENDATIONS_FILE" ] && echo "  💡 Recommendations: $RECOMMENDATIONS_FILE"

echo ""
log_info "To view detailed HTML reports, open:"
echo "  🌐 $REPORTS_DIR/memory-leak-report/index.html"

echo ""
log_success "Memory leak testing completed successfully!"
