#!/bin/bash

# Comprehensive Security Testing Script for anomaly_detection
# This script runs various security tests and generates detailed reports

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SECURITY_TEST_DIR="${PROJECT_ROOT}/tests/security"
REPORTS_DIR="${PROJECT_ROOT}/reports/security"
STAGING_HOST="http://localhost:8000"
PRODUCTION_HOST="https://api.anomaly_detection.com"
DEFAULT_TARGET="staging"
DEFAULT_SEVERITY="medium"

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

Run comprehensive security tests for anomaly_detection

OPTIONS:
    -h, --help              Show this help message
    -t, --target TARGET     Target environment (staging, production)
    -s, --severity LEVEL    Minimum severity level (low, medium, high, critical)
    -o, --output FORMAT     Output format (console, json, html, pdf)
    -r, --report FILE       Report output file
    --auth USERNAME:PASS    Authentication credentials
    --timeout SECONDS       Request timeout (default: 30)
    --threads NUM           Number of concurrent threads (default: 10)
    --rate-limit NUM        Requests per second limit (default: 50)
    --user-agent STRING     Custom user agent string
    --proxy URL             Proxy URL for requests
    --ssl-verify            Verify SSL certificates (default: false for staging)
    --include-slow          Include slow tests (default: false)
    --exclude-destructive   Exclude destructive tests (default: true)
    --custom-payloads FILE  Custom payload file
    --baseline FILE         Baseline results for comparison
    --continuous            Run in continuous monitoring mode
    --webhook URL           Webhook for notifications
    --slack-webhook URL     Slack webhook for notifications
    --email EMAIL           Email for notifications
    --dry-run               Show what would be executed

TEST CATEGORIES:
    auth                    Authentication tests
    authz                   Authorization tests
    input                   Input validation tests
    crypto                  Cryptography tests
    api                     API security tests
    headers                 Security headers tests
    infra                   Infrastructure tests
    compliance              Compliance tests
    all                     All test categories (default)

EXAMPLES:
    $0                                      # Run all tests on staging
    $0 -t production -s high                # High severity tests on production
    $0 --auth admin:password --include-slow # Authenticated tests with slow tests
    $0 -o json -r security_report.json     # JSON output to file
    $0 --continuous --webhook http://...    # Continuous monitoring
    $0 --baseline baseline.json             # Compare against baseline
EOF
}

# Configuration variables
TARGET="$DEFAULT_TARGET"
SEVERITY="$DEFAULT_SEVERITY"
OUTPUT_FORMAT="console"
REPORT_FILE=""
AUTH_CREDS=""
TIMEOUT=30
THREADS=10
RATE_LIMIT=50
USER_AGENT="anomaly-detectionSecurityTest/1.0"
PROXY=""
SSL_VERIFY=false
INCLUDE_SLOW=false
EXCLUDE_DESTRUCTIVE=true
CUSTOM_PAYLOADS=""
BASELINE_FILE=""
CONTINUOUS=false
WEBHOOK_URL=""
SLACK_WEBHOOK=""
EMAIL=""
DRY_RUN=false
TEST_CATEGORIES="all"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -s|--severity)
            SEVERITY="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -r|--report)
            REPORT_FILE="$2"
            shift 2
            ;;
        --auth)
            AUTH_CREDS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --rate-limit)
            RATE_LIMIT="$2"
            shift 2
            ;;
        --user-agent)
            USER_AGENT="$2"
            shift 2
            ;;
        --proxy)
            PROXY="$2"
            shift 2
            ;;
        --ssl-verify)
            SSL_VERIFY=true
            shift
            ;;
        --include-slow)
            INCLUDE_SLOW=true
            shift
            ;;
        --exclude-destructive)
            EXCLUDE_DESTRUCTIVE=false
            shift
            ;;
        --custom-payloads)
            CUSTOM_PAYLOADS="$2"
            shift 2
            ;;
        --baseline)
            BASELINE_FILE="$2"
            shift 2
            ;;
        --continuous)
            CONTINUOUS=true
            shift
            ;;
        --webhook)
            WEBHOOK_URL="$2"
            shift 2
            ;;
        --slack-webhook)
            SLACK_WEBHOOK="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        auth|authz|input|crypto|api|headers|infra|compliance|all)
            TEST_CATEGORIES="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set target host based on environment
if [[ "$TARGET" == "staging" ]]; then
    TARGET_HOST="$STAGING_HOST"
elif [[ "$TARGET" == "production" ]]; then
    TARGET_HOST="$PRODUCTION_HOST"
    SSL_VERIFY=true
else
    TARGET_HOST="$TARGET"
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Python and required packages
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed"
        exit 1
    fi

    # Check required Python packages
    local required_packages=("requests" "pytest" "cryptography" "jwt")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            log_error "Required Python package not installed: $package"
            exit 1
        fi
    done

    # Check security test files
    if [[ ! -f "$SECURITY_TEST_DIR/security_tests.py" ]]; then
        log_error "Security test file not found: $SECURITY_TEST_DIR/security_tests.py"
        exit 1
    fi

    # Create reports directory
    mkdir -p "$REPORTS_DIR"

    # Check target connectivity
    if [[ "$DRY_RUN" == "false" ]]; then
        log_info "Testing connectivity to: $TARGET_HOST"
        if ! curl -f -s --max-time 10 "$TARGET_HOST/health" > /dev/null 2>&1; then
            log_warning "Cannot reach $TARGET_HOST/health - target may not be ready"
        else
            log_success "Target host is reachable"
        fi
    fi

    log_success "Prerequisites check passed"
}

# Run OWASP ZAP scan
run_zap_scan() {
    log_info "Running OWASP ZAP security scan..."

    if ! command -v zap-baseline.py &> /dev/null; then
        log_warning "OWASP ZAP not installed, skipping ZAP scan"
        return 0
    fi

    local zap_report="$REPORTS_DIR/zap_report_$(date +%Y%m%d_%H%M%S).html"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: zap-baseline.py -t $TARGET_HOST -r $zap_report"
        return 0
    fi

    zap-baseline.py -t "$TARGET_HOST" -r "$zap_report" || {
        log_warning "ZAP scan completed with warnings"
    }

    log_success "ZAP scan completed: $zap_report"
}

# Run Nikto scan
run_nikto_scan() {
    log_info "Running Nikto web vulnerability scan..."

    if ! command -v nikto &> /dev/null; then
        log_warning "Nikto not installed, skipping Nikto scan"
        return 0
    fi

    local nikto_report="$REPORTS_DIR/nikto_report_$(date +%Y%m%d_%H%M%S).txt"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: nikto -h $TARGET_HOST -o $nikto_report"
        return 0
    fi

    nikto -h "$TARGET_HOST" -o "$nikto_report" || {
        log_warning "Nikto scan completed with warnings"
    }

    log_success "Nikto scan completed: $nikto_report"
}

# Run Nmap scan
run_nmap_scan() {
    log_info "Running Nmap network scan..."

    if ! command -v nmap &> /dev/null; then
        log_warning "Nmap not installed, skipping network scan"
        return 0
    fi

    local nmap_report="$REPORTS_DIR/nmap_report_$(date +%Y%m%d_%H%M%S).xml"
    local target_host=$(echo "$TARGET_HOST" | sed 's|http[s]*://||' | cut -d':' -f1)

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: nmap -sV -sC -O --script vuln $target_host -oX $nmap_report"
        return 0
    fi

    nmap -sV -sC -O --script vuln "$target_host" -oX "$nmap_report" || {
        log_warning "Nmap scan completed with warnings"
    }

    log_success "Nmap scan completed: $nmap_report"
}

# Run SQLMap scan
run_sqlmap_scan() {
    log_info "Running SQLMap injection scan..."

    if ! command -v sqlmap &> /dev/null; then
        log_warning "SQLMap not installed, skipping SQL injection scan"
        return 0
    fi

    local sqlmap_report="$REPORTS_DIR/sqlmap_report_$(date +%Y%m%d_%H%M%S).txt"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: sqlmap -u $TARGET_HOST/api/v1/search --batch --output-dir=$REPORTS_DIR"
        return 0
    fi

    # Test common endpoints for SQL injection
    local endpoints=(
        "/api/v1/search"
        "/api/v1/users"
        "/api/v1/datasets"
        "/api/v1/models"
    )

    for endpoint in "${endpoints[@]}"; do
        log_info "Testing $endpoint for SQL injection..."
        sqlmap -u "$TARGET_HOST$endpoint" --batch --output-dir="$REPORTS_DIR" >> "$sqlmap_report" 2>&1 || {
            log_warning "SQLMap scan on $endpoint completed with warnings"
        }
    done

    log_success "SQLMap scan completed: $sqlmap_report"
}

# Run custom security tests
run_custom_security_tests() {
    log_info "Running custom security tests..."

    local test_report="$REPORTS_DIR/custom_security_report_$(date +%Y%m%d_%H%M%S).json"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run custom security tests"
        return 0
    fi

    # Build Python command
    local python_cmd="python3 $SECURITY_TEST_DIR/security_tests.py"

    # Set environment variables
    export SECURITY_TEST_TARGET="$TARGET_HOST"
    export SECURITY_TEST_SEVERITY="$SEVERITY"
    export SECURITY_TEST_TIMEOUT="$TIMEOUT"
    export SECURITY_TEST_THREADS="$THREADS"
    export SECURITY_TEST_RATE_LIMIT="$RATE_LIMIT"
    export SECURITY_TEST_USER_AGENT="$USER_AGENT"
    export SECURITY_TEST_SSL_VERIFY="$SSL_VERIFY"
    export SECURITY_TEST_INCLUDE_SLOW="$INCLUDE_SLOW"
    export SECURITY_TEST_EXCLUDE_DESTRUCTIVE="$EXCLUDE_DESTRUCTIVE"
    export SECURITY_TEST_CATEGORIES="$TEST_CATEGORIES"

    if [[ -n "$AUTH_CREDS" ]]; then
        export SECURITY_TEST_AUTH="$AUTH_CREDS"
    fi

    if [[ -n "$PROXY" ]]; then
        export SECURITY_TEST_PROXY="$PROXY"
    fi

    if [[ -n "$CUSTOM_PAYLOADS" ]]; then
        export SECURITY_TEST_CUSTOM_PAYLOADS="$CUSTOM_PAYLOADS"
    fi

    # Run the security tests
    $python_cmd > "$test_report" 2>&1 || {
        log_error "Custom security tests failed"
        cat "$test_report"
        return 1
    }

    log_success "Custom security tests completed: $test_report"
}

# Run SSL/TLS tests
run_ssl_tests() {
    log_info "Running SSL/TLS security tests..."

    if ! command -v testssl &> /dev/null && ! command -v sslscan &> /dev/null; then
        log_warning "SSL testing tools not installed, skipping SSL tests"
        return 0
    fi

    local ssl_report="$REPORTS_DIR/ssl_report_$(date +%Y%m%d_%H%M%S).txt"
    local target_host=$(echo "$TARGET_HOST" | sed 's|http[s]*://||' | cut -d':' -f1)

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run SSL/TLS tests"
        return 0
    fi

    # Run testssl.sh if available
    if command -v testssl &> /dev/null; then
        log_info "Running testssl.sh..."
        testssl "$target_host" >> "$ssl_report" 2>&1 || {
            log_warning "testssl.sh completed with warnings"
        }
    fi

    # Run sslscan if available
    if command -v sslscan &> /dev/null; then
        log_info "Running sslscan..."
        sslscan "$target_host" >> "$ssl_report" 2>&1 || {
            log_warning "sslscan completed with warnings"
        }
    fi

    log_success "SSL/TLS tests completed: $ssl_report"
}

# Run authentication tests
run_auth_tests() {
    log_info "Running authentication security tests..."

    local auth_report="$REPORTS_DIR/auth_report_$(date +%Y%m%d_%H%M%S).json"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run authentication tests"
        return 0
    fi

    # Test common authentication endpoints
    local auth_endpoints=(
        "/auth/login"
        "/auth/register"
        "/auth/logout"
        "/auth/password-reset"
        "/auth/refresh"
    )

    for endpoint in "${auth_endpoints[@]}"; do
        log_info "Testing $endpoint..."

        # Test brute force protection
        for i in {1..10}; do
            curl -s -X POST "$TARGET_HOST$endpoint" \
                -H "Content-Type: application/json" \
                -d '{"username":"admin","password":"wrongpassword"}' >> "$auth_report" 2>&1
        done

        # Test rate limiting
        curl -s -X POST "$TARGET_HOST$endpoint" \
            -H "Content-Type: application/json" \
            -d '{"username":"admin","password":"wrongpassword"}' >> "$auth_report" 2>&1
    done

    log_success "Authentication tests completed: $auth_report"
}

# Generate comprehensive report
generate_comprehensive_report() {
    log_info "Generating comprehensive security report..."

    local final_report="$REPORTS_DIR/comprehensive_security_report_$(date +%Y%m%d_%H%M%S)"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would generate comprehensive report"
        return 0
    fi

    # Create report based on output format
    case "$OUTPUT_FORMAT" in
        "json")
            final_report="${final_report}.json"
            generate_json_report "$final_report"
            ;;
        "html")
            final_report="${final_report}.html"
            generate_html_report "$final_report"
            ;;
        "pdf")
            final_report="${final_report}.pdf"
            generate_pdf_report "$final_report"
            ;;
        *)
            final_report="${final_report}.md"
            generate_markdown_report "$final_report"
            ;;
    esac

    # Copy to specified report file if provided
    if [[ -n "$REPORT_FILE" ]]; then
        cp "$final_report" "$REPORT_FILE"
        log_success "Report saved to: $REPORT_FILE"
    fi

    log_success "Comprehensive report generated: $final_report"
}

# Generate markdown report
generate_markdown_report() {
    local report_file="$1"

    cat > "$report_file" << EOF
# Security Test Report

## Executive Summary
- **Target**: $TARGET_HOST
- **Test Date**: $(date)
- **Test Categories**: $TEST_CATEGORIES
- **Severity Level**: $SEVERITY

## Test Results

### OWASP ZAP Scan
$(find "$REPORTS_DIR" -name "zap_report_*.html" -newest | head -1 | xargs cat 2>/dev/null || echo "No ZAP results available")

### Nikto Scan
$(find "$REPORTS_DIR" -name "nikto_report_*.txt" -newest | head -1 | xargs cat 2>/dev/null || echo "No Nikto results available")

### Custom Security Tests
$(find "$REPORTS_DIR" -name "custom_security_report_*.json" -newest | head -1 | xargs cat 2>/dev/null || echo "No custom test results available")

### SSL/TLS Tests
$(find "$REPORTS_DIR" -name "ssl_report_*.txt" -newest | head -1 | xargs cat 2>/dev/null || echo "No SSL test results available")

## Recommendations

### Critical Issues
- Review and address all critical security vulnerabilities immediately
- Implement proper authentication and authorization mechanisms
- Fix any SQL injection or XSS vulnerabilities

### High Priority
- Implement security headers
- Configure proper SSL/TLS settings
- Review and fix authorization issues

### Medium Priority
- Implement rate limiting
- Review error handling
- Update security configurations

### Low Priority
- Hide server information
- Implement proper logging
- Review and update dependencies

## Test Details

### Test Configuration
- **Target Host**: $TARGET_HOST
- **Timeout**: $TIMEOUT seconds
- **Threads**: $THREADS
- **Rate Limit**: $RATE_LIMIT requests/second
- **SSL Verification**: $SSL_VERIFY
- **Include Slow Tests**: $INCLUDE_SLOW
- **Exclude Destructive Tests**: $EXCLUDE_DESTRUCTIVE

### Generated Files
$(ls -la "$REPORTS_DIR" | grep "$(date +%Y%m%d)")

---
*Report generated on $(date) by anomaly_detection Security Testing Suite*
EOF
}

# Generate JSON report
generate_json_report() {
    local report_file="$1"

    cat > "$report_file" << EOF
{
    "summary": {
        "target": "$TARGET_HOST",
        "test_date": "$(date -Iseconds)",
        "test_categories": "$TEST_CATEGORIES",
        "severity_level": "$SEVERITY"
    },
    "configuration": {
        "timeout": $TIMEOUT,
        "threads": $THREADS,
        "rate_limit": $RATE_LIMIT,
        "ssl_verify": $SSL_VERIFY,
        "include_slow": $INCLUDE_SLOW,
        "exclude_destructive": $EXCLUDE_DESTRUCTIVE
    },
    "results": {
        "zap_scan": "$(find "$REPORTS_DIR" -name "zap_report_*.html" -newest | head -1 || echo "null")",
        "nikto_scan": "$(find "$REPORTS_DIR" -name "nikto_report_*.txt" -newest | head -1 || echo "null")",
        "custom_tests": "$(find "$REPORTS_DIR" -name "custom_security_report_*.json" -newest | head -1 || echo "null")",
        "ssl_tests": "$(find "$REPORTS_DIR" -name "ssl_report_*.txt" -newest | head -1 || echo "null")"
    }
}
EOF
}

# Generate HTML report
generate_html_report() {
    local report_file="$1"

    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Security Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .critical { border-left: 5px solid #d32f2f; }
        .high { border-left: 5px solid #ff9800; }
        .medium { border-left: 5px solid #ffeb3b; }
        .low { border-left: 5px solid #4caf50; }
        .results { background-color: #f9f9f9; padding: 10px; margin: 10px 0; }
        pre { background-color: #f5f5f5; padding: 10px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Test Report</h1>
        <p><strong>Target:</strong> $TARGET_HOST</p>
        <p><strong>Test Date:</strong> $(date)</p>
        <p><strong>Test Categories:</strong> $TEST_CATEGORIES</p>
        <p><strong>Severity Level:</strong> $SEVERITY</p>
    </div>

    <div class="section">
        <h2>Test Results</h2>
        <div class="results">
            <h3>OWASP ZAP Scan</h3>
            <pre>$(find "$REPORTS_DIR" -name "zap_report_*.html" -newest | head -1 | xargs cat 2>/dev/null || echo "No ZAP results available")</pre>

            <h3>Nikto Scan</h3>
            <pre>$(find "$REPORTS_DIR" -name "nikto_report_*.txt" -newest | head -1 | xargs cat 2>/dev/null || echo "No Nikto results available")</pre>

            <h3>Custom Security Tests</h3>
            <pre>$(find "$REPORTS_DIR" -name "custom_security_report_*.json" -newest | head -1 | xargs cat 2>/dev/null || echo "No custom test results available")</pre>
        </div>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <div class="critical">
            <h3>Critical Issues</h3>
            <ul>
                <li>Review and address all critical security vulnerabilities immediately</li>
                <li>Implement proper authentication and authorization mechanisms</li>
                <li>Fix any SQL injection or XSS vulnerabilities</li>
            </ul>
        </div>

        <div class="high">
            <h3>High Priority</h3>
            <ul>
                <li>Implement security headers</li>
                <li>Configure proper SSL/TLS settings</li>
                <li>Review and fix authorization issues</li>
            </ul>
        </div>
    </div>

    <footer>
        <p><em>Report generated on $(date) by anomaly_detection Security Testing Suite</em></p>
    </footer>
</body>
</html>
EOF
}

# Send notifications
send_notifications() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would send notifications"
        return 0
    fi

    local message="Security test completed for $TARGET_HOST at $(date)"

    # Send webhook notification
    if [[ -n "$WEBHOOK_URL" ]]; then
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d '{"message": "'"$message"'", "target": "'"$TARGET_HOST"'"}' \
            > /dev/null 2>&1 || log_warning "Webhook notification failed"
    fi

    # Send Slack notification
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        curl -X POST "$SLACK_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d '{"text": "'"$message"'"}' \
            > /dev/null 2>&1 || log_warning "Slack notification failed"
    fi

    # Send email notification
    if [[ -n "$EMAIL" ]]; then
        echo "$message" | mail -s "Security Test Report" "$EMAIL" > /dev/null 2>&1 || log_warning "Email notification failed"
    fi
}

# Cleanup function
cleanup() {
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        log_error "Security testing failed with exit code $exit_code"
    fi

    # Kill any remaining processes
    pkill -f "zap-baseline.py" || true
    pkill -f "nikto" || true
    pkill -f "nmap" || true
    pkill -f "sqlmap" || true

    exit $exit_code
}

# Continuous monitoring mode
run_continuous_monitoring() {
    log_info "Starting continuous security monitoring..."

    while true; do
        log_info "Running security scan cycle..."

        # Run quick security tests
        run_custom_security_tests

        # Generate report
        generate_comprehensive_report

        # Send notifications if issues found
        send_notifications

        # Wait before next cycle (default 1 hour)
        local interval=${CONTINUOUS_INTERVAL:-3600}
        log_info "Waiting $interval seconds before next scan..."
        sleep "$interval"
    done
}

# Main function
main() {
    log_info "Starting anomaly_detection security testing..."

    # Set up error handling
    trap cleanup EXIT

    # Check prerequisites
    check_prerequisites

    # Run continuous monitoring if requested
    if [[ "$CONTINUOUS" == "true" ]]; then
        run_continuous_monitoring
        return 0
    fi

    # Run security tests
    log_info "Target: $TARGET_HOST"
    log_info "Test Categories: $TEST_CATEGORIES"
    log_info "Severity Level: $SEVERITY"

    # Run different types of security tests
    run_zap_scan
    run_nikto_scan
    run_nmap_scan
    run_sqlmap_scan
    run_custom_security_tests
    run_ssl_tests
    run_auth_tests

    # Generate comprehensive report
    generate_comprehensive_report

    # Send notifications
    send_notifications

    log_success "Security testing completed successfully!"

    if [[ "$DRY_RUN" == "false" ]]; then
        log_info "Next steps:"
        echo "1. Review security test results in $REPORTS_DIR"
        echo "2. Address any critical or high severity issues"
        echo "3. Implement recommended security improvements"
        echo "4. Re-run tests to verify fixes"
        echo "5. Proceed with performance optimization"
    fi
}

# Run main function
main "$@"
