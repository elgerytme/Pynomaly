#!/bin/bash
# Pynomaly Load Testing Runner
# Executes comprehensive load tests using k6 and Artillery

set -e

# Configuration
BASE_URL=${BASE_URL:-"http://localhost:8000"}
OUTPUT_DIR="./test_reports/load-testing"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}      Pynomaly Load Testing Suite          ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to check if service is running
check_service() {
    echo -e "${YELLOW}Checking if Pynomaly service is running...${NC}"
    
    if curl -s "${BASE_URL}/api/ui/health" > /dev/null; then
        echo -e "${GREEN}✓ Service is running at ${BASE_URL}${NC}"
        return 0
    else
        echo -e "${RED}✗ Service is not available at ${BASE_URL}${NC}"
        echo "Please start the Pynomaly service before running load tests."
        exit 1
    fi
}

# Function to run k6 tests
run_k6_tests() {
    echo -e "${BLUE}Running k6 Load Tests...${NC}"
    
    if ! command -v k6 &> /dev/null; then
        echo -e "${YELLOW}k6 not found, installing...${NC}"
        # Installation instructions for different platforms
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install k6
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update && sudo apt-get install -y k6
        else
            echo -e "${RED}Please install k6 manually: https://k6.io/docs/getting-started/installation/${NC}"
            return 1
        fi
    fi
    
    echo "Starting k6 load test..."
    
    k6 run \
        --env BASE_URL="$BASE_URL" \
        --out json="$OUTPUT_DIR/k6-results-${TIMESTAMP}.json" \
        --out csv="$OUTPUT_DIR/k6-metrics-${TIMESTAMP}.csv" \
        --summary-export="$OUTPUT_DIR/k6-summary-${TIMESTAMP}.json" \
        ./k6-load-test.js
    
    echo -e "${GREEN}✓ k6 tests completed${NC}"
}

# Function to run Artillery tests
run_artillery_tests() {
    echo -e "${BLUE}Running Artillery Load Tests...${NC}"
    
    if ! command -v artillery &> /dev/null; then
        echo -e "${YELLOW}Artillery not found, installing...${NC}"
        npm install -g artillery
    fi
    
    echo "Starting Artillery load test..."
    
    artillery run \
        --environment local \
        --config artillery-config.yml \
        --output "$OUTPUT_DIR/artillery-results-${TIMESTAMP}.json" \
        artillery-config.yml
    
    # Generate HTML report
    artillery report \
        "$OUTPUT_DIR/artillery-results-${TIMESTAMP}.json" \
        --output "$OUTPUT_DIR/artillery-report-${TIMESTAMP}.html"
    
    echo -e "${GREEN}✓ Artillery tests completed${NC}"
}

# Function to generate combined report
generate_report() {
    echo -e "${BLUE}Generating Combined Report...${NC}"
    
    cat > "$OUTPUT_DIR/load-test-summary-${TIMESTAMP}.md" << EOF
# Pynomaly Load Test Report

**Test Date:** $(date)  
**Base URL:** $BASE_URL  
**Test Duration:** $TIMESTAMP

## Test Configuration

### k6 Test Configuration
- **Ramp-up:** 2 minutes to 10 users
- **Normal Load:** 5 minutes at 50 users  
- **Peak Load:** 3 minutes at 100 users
- **Stress Test:** 2 minutes at 150 users
- **Scale Down:** 3 minutes to 100 users
- **Cool Down:** 2 minutes to 0 users

### Artillery Test Configuration
- **Phases:** 6 phases over 17 minutes
- **Maximum Users:** 100 concurrent
- **Test Scenarios:** Page Load, API Stress, User Journey, WebSocket, Error Handling

## Test Results

### Performance Metrics
- See detailed results in:
  - k6: \`k6-results-${TIMESTAMP}.json\`
  - Artillery: \`artillery-report-${TIMESTAMP}.html\`

### Key Performance Indicators
- **Response Time P95:** < 2000ms (target)
- **Response Time P99:** < 5000ms (target)
- **Error Rate:** < 1% (target)
- **Success Rate:** > 95% (target)

## Files Generated
EOF

    # List generated files
    echo "### Generated Files:" >> "$OUTPUT_DIR/load-test-summary-${TIMESTAMP}.md"
    find "$OUTPUT_DIR" -name "*${TIMESTAMP}*" -type f | while read file; do
        echo "- $(basename "$file")" >> "$OUTPUT_DIR/load-test-summary-${TIMESTAMP}.md"
    done
    
    echo -e "${GREEN}✓ Report generated: $OUTPUT_DIR/load-test-summary-${TIMESTAMP}.md${NC}"
}

# Function to run performance baseline tests
run_baseline_tests() {
    echo -e "${BLUE}Running Performance Baseline Tests...${NC}"
    
    # Simple baseline test with minimal load
    if command -v k6 &> /dev/null; then
        k6 run \
            --vus 1 \
            --duration 30s \
            --env BASE_URL="$BASE_URL" \
            --out json="$OUTPUT_DIR/baseline-${TIMESTAMP}.json" \
            ./k6-load-test.js
        
        echo -e "${GREEN}✓ Baseline tests completed${NC}"
    fi
}

# Function to cleanup old test results
cleanup_old_results() {
    echo -e "${YELLOW}Cleaning up old test results (keeping last 10)...${NC}"
    
    # Keep only the 10 most recent test result files
    find "$OUTPUT_DIR" -name "*.json" -type f | sort -r | tail -n +11 | xargs rm -f
    find "$OUTPUT_DIR" -name "*.html" -type f | sort -r | tail -n +11 | xargs rm -f
    find "$OUTPUT_DIR" -name "*.csv" -type f | sort -r | tail -n +11 | xargs rm -f
    
    echo -e "${GREEN}✓ Cleanup completed${NC}"
}

# Main execution flow
main() {
    echo "Load Testing Configuration:"
    echo "  Base URL: $BASE_URL"
    echo "  Output Directory: $OUTPUT_DIR"
    echo "  Timestamp: $TIMESTAMP"
    echo ""
    
    # Check command line arguments
    case "${1:-all}" in
        "baseline")
            check_service
            run_baseline_tests
            ;;
        "k6")
            check_service
            run_k6_tests
            ;;
        "artillery") 
            check_service
            run_artillery_tests
            ;;
        "quick")
            check_service
            run_baseline_tests
            ;;
        "cleanup")
            cleanup_old_results
            ;;
        "all"|*)
            check_service
            run_baseline_tests
            run_k6_tests
            run_artillery_tests
            generate_report
            cleanup_old_results
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}      Load Testing Completed Successfully  ${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo "View HTML reports in a browser:"
    echo "  Artillery: $OUTPUT_DIR/artillery-report-${TIMESTAMP}.html"
    echo ""
}

# Check if we're in the right directory
if [[ ! -f "k6-load-test.js" ]]; then
    echo -e "${RED}Error: k6-load-test.js not found. Please run this script from the tests/ui/load-testing directory.${NC}"
    exit 1
fi

# Execute main function with all arguments
main "$@"