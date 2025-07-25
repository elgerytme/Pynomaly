#!/bin/bash
set -euo pipefail

# Production Alert Channel Integration Script
# Sets up and validates all production alert channels

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/src/packages/deployment/config"

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

# Environment variable validation
validate_environment_variables() {
    log_info "Validating required environment variables..."
    
    local missing_vars=()
    
    # Check for required environment variables
    if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
        missing_vars+=("SLACK_WEBHOOK_URL")
    fi
    
    if [[ -z "${GRAFANA_API_KEY:-}" ]]; then
        missing_vars+=("GRAFANA_API_KEY")
    fi
    
    # Optional but recommended variables
    local optional_vars=()
    if [[ -z "${PAGERDUTY_INTEGRATION_KEY:-}" ]]; then
        optional_vars+=("PAGERDUTY_INTEGRATION_KEY")
    fi
    
    if [[ -z "${SMTP_USERNAME:-}" ]]; then
        optional_vars+=("SMTP_USERNAME")
    fi
    
    if [[ -z "${SMTP_PASSWORD:-}" ]]; then
        optional_vars+=("SMTP_PASSWORD")
    fi
    
    # Report missing required variables
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        echo ""
        echo "Set them using:"
        for var in "${missing_vars[@]}"; do
            echo "  export $var='your_value_here'"
        done
        return 1
    fi
    
    # Report missing optional variables
    if [[ ${#optional_vars[@]} -gt 0 ]]; then
        log_warning "Missing optional environment variables (functionality will be limited):"
        for var in "${optional_vars[@]}"; do
            echo "  - $var"
        done
        echo ""
    fi
    
    log_success "Environment variable validation completed"
    return 0
}

# Test Slack integration
test_slack_integration() {
    log_info "Testing Slack integration..."
    
    if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
        log_warning "SLACK_WEBHOOK_URL not set - skipping Slack test"
        return 0
    fi
    
    # Test webhook with a simple payload
    local test_payload='{
        "channel": "#production-alerts",
        "username": "Production Monitor",
        "text": "🧪 Alert channel integration test",
        "attachments": [{
            "color": "good",
            "title": "Integration Test",
            "text": "This is a test message to verify Slack integration is working",
            "fields": [
                {"title": "Status", "value": "Testing", "short": true},
                {"title": "Component", "value": "Alert Integration", "short": true}
            ]
        }]
    }'
    
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        "$SLACK_WEBHOOK_URL" \
        --max-time 10)
    
    if [[ "$response_code" == "200" ]]; then
        log_success "Slack integration test passed"
        return 0
    else
        log_error "Slack integration test failed (HTTP $response_code)"
        return 1
    fi
}

# Test PagerDuty integration
test_pagerduty_integration() {
    log_info "Testing PagerDuty integration..."
    
    if [[ -z "${PAGERDUTY_INTEGRATION_KEY:-}" ]]; then
        log_warning "PAGERDUTY_INTEGRATION_KEY not set - skipping PagerDuty test"
        return 0
    fi
    
    local test_payload='{
        "routing_key": "'$PAGERDUTY_INTEGRATION_KEY'",
        "event_action": "trigger",
        "dedup_key": "test-alert-'$(date +%s)'",
        "payload": {
            "summary": "🧪 Alert Integration Test",
            "source": "production-monitoring-system",
            "severity": "info",
            "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
            "custom_details": {
                "test_type": "integration_test",
                "component": "alert_channels",
                "message": "This is a test alert to verify PagerDuty integration"
            }
        }
    }'
    
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        "https://events.pagerduty.com/v2/enqueue" \
        --max-time 10)
    
    if [[ "$response_code" == "202" ]]; then
        log_success "PagerDuty integration test passed"
        return 0
    else
        log_error "PagerDuty integration test failed (HTTP $response_code)"
        return 1
    fi
}

# Test email integration (basic SMTP connectivity)
test_email_integration() {
    log_info "Testing email integration..."
    
    if [[ -z "${SMTP_USERNAME:-}" ]] || [[ -z "${SMTP_PASSWORD:-}" ]]; then
        log_warning "Email credentials not set - skipping email test"
        return 0
    fi
    
    # Use Python to test SMTP connectivity
    python3 -c "
import smtplib
import sys
import os
from email.mime.text import MIMEText

try:
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.company.com')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    username = os.getenv('SMTP_USERNAME')
    password = os.getenv('SMTP_PASSWORD')
    
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(username, password)
    server.quit()
    
    print('Email connectivity test passed')
    sys.exit(0)
except Exception as e:
    print(f'Email connectivity test failed: {e}', file=sys.stderr)
    sys.exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        log_success "Email integration test passed"
        return 0
    else
        log_error "Email integration test failed"
        return 1
    fi
}

# Setup Grafana dashboards and alerts
setup_grafana_integration() {
    log_info "Setting up Grafana integration..."
    
    if [[ -z "${GRAFANA_API_KEY:-}" ]]; then
        log_error "GRAFANA_API_KEY not set - cannot setup Grafana integration"
        return 1
    fi
    
    # Test Grafana connectivity
    local grafana_url="${GRAFANA_URL:-http://localhost:3000}"
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $GRAFANA_API_KEY" \
        "$grafana_url/api/health" \
        --max-time 10)
    
    if [[ "$response_code" != "200" ]]; then
        log_error "Cannot connect to Grafana (HTTP $response_code)"
        return 1
    fi
    
    log_success "Grafana connectivity verified"
    
    # Run observability integration setup
    python3 "$CONFIG_DIR/../monitoring/observability-integration.py" \
        --config "$CONFIG_DIR/monitoring-config.yaml" \
        --setup-all
    
    if [[ $? -eq 0 ]]; then
        log_success "Grafana integration setup completed"
        return 0
    else
        log_error "Grafana integration setup failed"
        return 1
    fi
}

# Setup monitoring alerts
setup_monitoring_alerts() {
    log_info "Setting up monitoring alerts..."
    
    # Test all alert channels
    python3 "$CONFIG_DIR/alert-channels-setup.py" \
        --config "$CONFIG_DIR/monitoring-config.yaml" \
        --test-all
    
    local test_result=$?
    
    # Generate status report
    python3 "$CONFIG_DIR/alert-channels-setup.py" \
        --config "$CONFIG_DIR/monitoring-config.yaml" \
        --status
    
    if [[ $test_result -eq 0 ]]; then
        log_success "All alert channels configured successfully"
        return 0
    else
        log_warning "Some alert channels failed configuration - check output above"
        return 1
    fi
}

# Deploy production monitoring configuration
deploy_monitoring_config() {
    log_info "Deploying monitoring configuration to production..."
    
    # Copy configuration to appropriate locations
    if command -v kubectl &> /dev/null; then
        # Create ConfigMap for monitoring configuration
        kubectl create configmap monitoring-config \
            --from-file="$CONFIG_DIR/monitoring-config.yaml" \
            --namespace=production \
            --dry-run=client -o yaml | kubectl apply -f -
        
        if [[ $? -eq 0 ]]; then
            log_success "Monitoring configuration deployed to Kubernetes"
        else
            log_error "Failed to deploy monitoring configuration to Kubernetes"
            return 1
        fi
    else
        log_warning "kubectl not available - skipping Kubernetes deployment"
    fi
    
    return 0
}

# Validate integration
validate_integration() {
    log_info "Validating complete integration..."
    
    # Run comprehensive validation
    python3 "$CONFIG_DIR/../monitoring/observability-integration.py" \
        --config "$CONFIG_DIR/monitoring-config.yaml" \
        --validate \
        --report "/tmp/integration-validation.txt"
    
    if [[ $? -eq 0 ]]; then
        log_success "Integration validation completed"
        cat "/tmp/integration-validation.txt"
        return 0
    else
        log_error "Integration validation failed"
        return 1
    fi
}

# Send test alerts to verify end-to-end functionality
send_test_alerts() {
    log_info "Sending test alerts through all channels..."
    
    python3 "$CONFIG_DIR/alert-channels-setup.py" \
        --config "$CONFIG_DIR/monitoring-config.yaml" \
        --send-test-alert
    
    if [[ $? -eq 0 ]]; then
        log_success "Test alerts sent successfully"
        return 0
    else
        log_error "Failed to send test alerts"
        return 1
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --validate-env     Validate environment variables only"
    echo "  --test-channels    Test alert channels only"
    echo "  --setup-grafana    Setup Grafana integration only"
    echo "  --setup-alerts     Setup monitoring alerts only"
    echo "  --deploy-config    Deploy monitoring configuration only"
    echo "  --send-test        Send test alerts only"
    echo "  --full-setup       Run complete integration setup (default)"
    echo "  --help             Show this help message"
    echo ""
    echo "Environment Variables Required:"
    echo "  SLACK_WEBHOOK_URL           Slack webhook URL for alerts"
    echo "  GRAFANA_API_KEY            Grafana API key for dashboard setup"
    echo ""
    echo "Environment Variables Optional:"
    echo "  PAGERDUTY_INTEGRATION_KEY  PagerDuty integration key"
    echo "  SMTP_USERNAME              SMTP username for email alerts"
    echo "  SMTP_PASSWORD              SMTP password for email alerts"
    echo "  GRAFANA_URL                Grafana URL (default: http://localhost:3000)"
}

# Main execution
main() {
    echo "============================================"
    echo "Production Alert Channel Integration Setup"
    echo "============================================"
    echo ""
    
    cd "$PROJECT_ROOT/src/packages/deployment"
    
    case "${1:---full-setup}" in
        --validate-env)
            validate_environment_variables
            ;;
        --test-channels)
            validate_environment_variables || exit 1
            test_slack_integration
            test_pagerduty_integration  
            test_email_integration
            ;;
        --setup-grafana)
            validate_environment_variables || exit 1
            setup_grafana_integration
            ;;
        --setup-alerts)
            validate_environment_variables || exit 1
            setup_monitoring_alerts
            ;;
        --deploy-config)
            deploy_monitoring_config
            ;;
        --send-test)
            validate_environment_variables || exit 1
            send_test_alerts
            ;;
        --full-setup)
            log_info "Starting complete alert integration setup..."
            
            validate_environment_variables || exit 1
            test_slack_integration
            test_pagerduty_integration
            test_email_integration
            setup_grafana_integration
            setup_monitoring_alerts
            deploy_monitoring_config
            validate_integration
            send_test_alerts
            
            log_success "Alert integration setup completed!"
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"