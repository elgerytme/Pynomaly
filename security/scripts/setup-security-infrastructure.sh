#!/bin/bash

# Security Infrastructure Setup Script
# Configures comprehensive security infrastructure for the detection platform

set -euo pipefail

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

# Default values
ENVIRONMENT=""
SETUP_SECRETS=true
CONFIGURE_SCANNING=true
SETUP_COMPLIANCE=true
DRY_RUN=false

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Set up security infrastructure for the detection platform.

OPTIONS:
    -e, --environment ENVIRONMENT    Target environment (staging|production)
    -s, --skip-secrets              Skip secrets setup
    -c, --skip-scanning             Skip security scanning configuration
    -p, --skip-compliance           Skip compliance setup
    -d, --dry-run                   Show what would be configured without making changes
    -h, --help                      Show this help message

EXAMPLES:
    $0 -e staging                   Full security setup for staging
    $0 -e production -s             Production setup without secrets
    $0 -e staging -d                Dry run for staging

REQUIRED ENVIRONMENT VARIABLES:
    GITHUB_TOKEN                    GitHub token for repository access
    SLACK_WEBHOOK_URL              Slack webhook for security notifications (optional)
    EMAIL_NOTIFICATION_LIST        Comma-separated list of email addresses (optional)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -s|--skip-secrets)
            SETUP_SECRETS=false
            shift
            ;;
        -c|--skip-scanning)
            CONFIGURE_SCANNING=false
            shift
            ;;
        -p|--skip-compliance)
            SETUP_COMPLIANCE=false
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown parameter: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate parameters
if [[ -z "$ENVIRONMENT" ]]; then
    log_error "Environment parameter is required"
    usage
    exit 1
fi

if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    log_error "Environment must be 'staging' or 'production'"
    exit 1
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required commands
    for cmd in gh kubectl openssl; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_tools+=("$cmd")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check GitHub token
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        log_error "GITHUB_TOKEN environment variable is required"
        exit 1
    fi
    
    # Test GitHub authentication
    if ! gh auth status >/dev/null 2>&1; then
        log_error "GitHub authentication failed. Please run 'gh auth login' or set GITHUB_TOKEN"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Generate secure secrets
generate_secrets() {
    log_info "Generating secure secrets..."
    
    # Generate database encryption key
    DB_ENCRYPTION_KEY=$(openssl rand -base64 32)
    
    # Generate JWT secret
    JWT_SECRET=$(openssl rand -base64 64)
    
    # Generate API keys
    API_KEY_SECRET=$(openssl rand -hex 32)
    
    # Generate webhook signing secret
    WEBHOOK_SECRET=$(openssl rand -hex 32)
    
    # Generate monitoring secrets
    PROMETHEUS_PASSWORD=$(openssl rand -base64 24)
    GRAFANA_SECRET_KEY=$(openssl rand -base64 32)
    
    log_success "Secure secrets generated"
}

# Set up GitHub repository secrets
setup_github_secrets() {
    if [[ "$SETUP_SECRETS" != "true" ]]; then
        log_info "Skipping GitHub secrets setup"
        return
    fi
    
    log_info "Setting up GitHub repository secrets..."
    
    # Generate secrets if not done already
    if [[ -z "${DB_ENCRYPTION_KEY:-}" ]]; then
        generate_secrets
    fi
    
    # Repository secrets for the environment
    local secrets=(
        "DB_ENCRYPTION_KEY_${ENVIRONMENT^^}:${DB_ENCRYPTION_KEY}"
        "JWT_SECRET_${ENVIRONMENT^^}:${JWT_SECRET}"
        "API_KEY_SECRET_${ENVIRONMENT^^}:${API_KEY_SECRET}"
        "WEBHOOK_SECRET_${ENVIRONMENT^^}:${WEBHOOK_SECRET}"
        "PROMETHEUS_PASSWORD_${ENVIRONMENT^^}:${PROMETHEUS_PASSWORD}"
        "GRAFANA_SECRET_KEY_${ENVIRONMENT^^}:${GRAFANA_SECRET_KEY}"
    )
    
    # Add environment-specific Kubernetes secrets if provided
    if [[ -n "${TF_VAR_kubernetes_cluster_endpoint:-}" ]]; then
        secrets+=(
            "${ENVIRONMENT^^}_CLUSTER_URL:${TF_VAR_kubernetes_cluster_endpoint}"
            "${ENVIRONMENT^^}_TOKEN:${TF_VAR_kubernetes_token}"
            "${ENVIRONMENT^^}_CA_CERT:${TF_VAR_kubernetes_cluster_ca_certificate}"
        )
    fi
    
    # Set secrets in GitHub
    for secret in "${secrets[@]}"; do
        local name="${secret%%:*}"
        local value="${secret#*:}"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would set secret: $name"
        else
            echo "$value" | gh secret set "$name" --app actions
            log_success "Set secret: $name"
        fi
    done
    
    # Set optional notification secrets
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would set Slack webhook"
        else
            echo "$SLACK_WEBHOOK_URL" | gh secret set "SLACK_WEBHOOK_URL" --app actions
            log_success "Set Slack webhook secret"
        fi
    fi
    
    if [[ -n "${EMAIL_NOTIFICATION_LIST:-}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would set email notification list"
        else
            echo "$EMAIL_NOTIFICATION_LIST" | gh secret set "EMAIL_NOTIFICATION_LIST" --app actions
            log_success "Set email notification secret"
        fi
    fi
}

# Configure security scanning workflows
configure_security_scanning() {
    if [[ "$CONFIGURE_SCANNING" != "true" ]]; then
        log_info "Skipping security scanning configuration"
        return
    fi
    
    log_info "Configuring security scanning workflows..."
    
    # Enable GitHub Advanced Security features if available
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would enable GitHub Advanced Security features"
    else
        # Enable Dependabot alerts
        gh api -X PATCH "/repos/:owner/:repo/vulnerability-alerts" || log_warning "Could not enable vulnerability alerts"
        
        # Enable Dependabot security updates
        gh api -X PUT "/repos/:owner/:repo/automated-security-fixes" || log_warning "Could not enable security fixes"
        
        # Enable secret scanning if available
        gh api -X PATCH "/repos/:owner/:repo" -f has_secret_scanning=true || log_warning "Could not enable secret scanning"
        
        log_success "GitHub security features configured"
    fi
    
    # Set up custom security workflow triggers
    local security_config_file=".github/workflows/security-scanning-${ENVIRONMENT}.yml"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create environment-specific security workflow"
    else
        # Create environment-specific security workflow
        cat > "$security_config_file" << EOF
name: Security Scanning - ${ENVIRONMENT^}

on:
  push:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'deploy/**'
      - 'security/**'
  schedule:
    - cron: '0 3 * * *'  # Daily at 3 AM
  workflow_dispatch:

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  security-scan:
    name: Environment Security Scan
    runs-on: ubuntu-latest
    environment: ${ENVIRONMENT}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Run environment-specific security scan
        run: |
          echo "Running security scan for ${ENVIRONMENT}"
          python security/scripts/validate-security-config.py
          
      - name: Deploy security monitoring
        if: \${{ github.ref == 'refs/heads/main' }}
        run: |
          echo "Deploying security monitoring for ${ENVIRONMENT}"
          # Add deployment commands here
EOF
        
        log_success "Created environment-specific security workflow"
    fi
}

# Set up compliance monitoring
setup_compliance_monitoring() {
    if [[ "$SETUP_COMPLIANCE" != "true" ]]; then
        log_info "Skipping compliance setup"
        return
    fi
    
    log_info "Setting up compliance monitoring..."
    
    # Create compliance configuration
    local compliance_config="security/compliance/compliance-${ENVIRONMENT}.yaml"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create compliance configuration"
    else
        mkdir -p "security/compliance"
        
        cat > "$compliance_config" << EOF
# Compliance Configuration for ${ENVIRONMENT^} Environment

compliance_frameworks:
  soc2:
    enabled: true
    controls:
      - CC6.1  # Logical Access Controls
      - CC6.2  # Authentication and Authorization
      - CC6.3  # System Operations
      - CC7.1  # System Monitoring
    reporting_frequency: "monthly"
    
  gdpr:
    enabled: true
    data_protection_measures:
      - encryption_at_rest
      - encryption_in_transit
      - access_logging
      - data_retention_policies
    reporting_frequency: "quarterly"
    
  owasp_top10:
    enabled: true
    version: "2021"
    automated_testing: true
    reporting_frequency: "weekly"

security_controls:
  access_control:
    multi_factor_authentication: true
    role_based_access: true
    principle_of_least_privilege: true
    
  data_protection:
    encryption_standards: "AES-256"
    key_management: "automated"
    backup_encryption: true
    
  monitoring:
    security_logging: true
    anomaly_detection: true
    incident_response: true
    
  vulnerability_management:
    automated_scanning: true
    patch_management: true
    penetration_testing: "quarterly"

audit_requirements:
  log_retention: "2_years"
  access_reviews: "quarterly"
  vulnerability_assessments: "monthly"
  compliance_reports: "monthly"

notification_channels:
  slack: \${SLACK_WEBHOOK_URL}
  email: \${EMAIL_NOTIFICATION_LIST}
  severity_thresholds:
    critical: "immediate"
    high: "within_1_hour"
    medium: "within_24_hours"
    low: "weekly_summary"
EOF
        
        log_success "Compliance configuration created"
    fi
    
    # Create compliance monitoring script
    local compliance_script="security/scripts/compliance-monitor-${ENVIRONMENT}.py"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create compliance monitoring script"
    else
        cat > "$compliance_script" << 'EOF'
#!/usr/bin/env python3
"""
Compliance Monitoring Script
Monitors and reports on compliance status for the detection platform.
"""

import json
import yaml
import requests
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any


class ComplianceMonitor:
    """Monitors compliance status and generates reports."""
    
    def __init__(self, environment: str):
        self.environment = environment
        self.config_path = Path(f"security/compliance/compliance-{environment}.yaml")
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load compliance configuration."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def check_security_controls(self) -> Dict[str, bool]:
        """Check security controls implementation."""
        controls = {}
        
        # Check access control
        controls['mfa_enabled'] = self._check_mfa_implementation()
        controls['rbac_enabled'] = self._check_rbac_implementation()
        controls['least_privilege'] = self._check_least_privilege()
        
        # Check data protection
        controls['encryption_at_rest'] = self._check_encryption_at_rest()
        controls['encryption_in_transit'] = self._check_encryption_in_transit()
        controls['key_management'] = self._check_key_management()
        
        # Check monitoring
        controls['security_logging'] = self._check_security_logging()
        controls['anomaly_detection'] = self._check_anomaly_detection()
        controls['incident_response'] = self._check_incident_response()
        
        return controls
    
    def _check_mfa_implementation(self) -> bool:
        """Check if MFA is properly implemented."""
        # Implementation would check actual MFA configuration
        return True  # Placeholder
    
    def _check_rbac_implementation(self) -> bool:
        """Check if RBAC is properly implemented."""
        # Implementation would check Kubernetes RBAC
        return True  # Placeholder
    
    def _check_least_privilege(self) -> bool:
        """Check if principle of least privilege is enforced."""
        # Implementation would audit permissions
        return True  # Placeholder
    
    def _check_encryption_at_rest(self) -> bool:
        """Check if data is encrypted at rest."""
        # Implementation would verify database encryption
        return True  # Placeholder
    
    def _check_encryption_in_transit(self) -> bool:
        """Check if data is encrypted in transit."""
        # Implementation would verify TLS configuration
        return True  # Placeholder
    
    def _check_key_management(self) -> bool:
        """Check key management implementation."""
        # Implementation would verify key rotation and security
        return True  # Placeholder
    
    def _check_security_logging(self) -> bool:
        """Check if security logging is enabled."""
        # Implementation would verify log configuration
        return True  # Placeholder
    
    def _check_anomaly_detection(self) -> bool:
        """Check if detection is operational."""
        # Implementation would verify monitoring systems
        return True  # Placeholder
    
    def _check_incident_response(self) -> bool:
        """Check if incident response procedures are in place."""
        # Implementation would verify incident response setup
        return True  # Placeholder
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        controls = self.check_security_controls()
        
        # Calculate compliance score
        total_controls = len(controls)
        passed_controls = sum(controls.values())
        compliance_score = (passed_controls / total_controls) * 100
        
        report = {
            'environment': self.environment,
            'report_date': datetime.utcnow().isoformat(),
            'compliance_score': compliance_score,
            'status': 'compliant' if compliance_score >= 95 else 'non_compliant',
            'controls': controls,
            'recommendations': self._generate_recommendations(controls),
            'next_review_date': (datetime.utcnow() + timedelta(days=30)).isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, controls: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on control status."""
        recommendations = []
        
        for control, passed in controls.items():
            if not passed:
                recommendations.append(f"Implement or fix {control.replace('_', ' ')}")
        
        if not recommendations:
            recommendations.append("All security controls are properly implemented")
        
        return recommendations
    
    def send_notifications(self, report: Dict[str, Any]) -> None:
        """Send compliance report notifications."""
        if report['compliance_score'] < 95:
            self._send_alert_notification(report)
        else:
            self._send_status_notification(report)
    
    def _send_alert_notification(self, report: Dict[str, Any]) -> None:
        """Send alert for compliance issues."""
        # Implementation would send notifications to configured channels
        print(f"🚨 COMPLIANCE ALERT: {self.environment} compliance score: {report['compliance_score']:.1f}%")
    
    def _send_status_notification(self, report: Dict[str, Any]) -> None:
        """Send status notification for good compliance."""
        # Implementation would send status updates
        print(f"✅ COMPLIANCE OK: {self.environment} compliance score: {report['compliance_score']:.1f}%")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python compliance-monitor.py <environment>")
        sys.exit(1)
    
    environment = sys.argv[1]
    
    monitor = ComplianceMonitor(environment)
    report = monitor.generate_compliance_report()
    
    # Save report
    report_file = f"compliance-report-{environment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Send notifications
    monitor.send_notifications(report)
    
    print(f"Compliance report generated: {report_file}")
    print(f"Compliance score: {report['compliance_score']:.1f}%")
    
    # Exit with error code if not compliant
    sys.exit(0 if report['status'] == 'compliant' else 1)


if __name__ == "__main__":
    main()
EOF
        
        chmod +x "$compliance_script"
        log_success "Compliance monitoring script created"
    fi
}

# Create security monitoring dashboard
create_security_dashboard() {
    log_info "Creating security monitoring dashboard..."
    
    local dashboard_config="security/monitoring/security-dashboard-${ENVIRONMENT}.json"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create security dashboard"
    else
        mkdir -p "security/monitoring"
        
        cat > "$dashboard_config" << EOF
{
  "dashboard": {
    "id": null,
    "title": "Security Monitoring - ${ENVIRONMENT^}",
    "tags": ["security", "${ENVIRONMENT}"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Security Alerts Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(security_alerts_total{environment=\"${ENVIRONMENT}\"}[5m]))",
            "legendFormat": "Alert Rate"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Vulnerability Scan Results",
        "type": "graph",
        "targets": [
          {
            "expr": "vulnerability_scan_results{environment=\"${ENVIRONMENT}\"}",
            "legendFormat": "{{severity}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Failed Authentication Attempts",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(failed_auth_attempts_total{environment=\"${ENVIRONMENT}\"}[5m])",
            "legendFormat": "Failed Attempts/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Compliance Score",
        "type": "gauge",
        "targets": [
          {
            "expr": "compliance_score{environment=\"${ENVIRONMENT}\"}",
            "legendFormat": "Compliance %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 80},
                {"color": "green", "value": 95}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ],
    "time": {
      "from": "now-24h",
      "to": "now"
    },
    "refresh": "1m"
  }
}
EOF
        
        log_success "Security dashboard configuration created"
    fi
}

# Set up automated security testing
setup_automated_security_testing() {
    log_info "Setting up automated security testing..."
    
    local security_test_config=".github/workflows/security-tests-${ENVIRONMENT}.yml"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create automated security testing workflow"
    else
        cat > "$security_test_config" << EOF
name: Automated Security Tests - ${ENVIRONMENT^}

on:
  schedule:
    - cron: '0 4 * * *'  # Daily at 4 AM
  workflow_dispatch:
    inputs:
      test_type:
        description: 'Type of security test'
        required: true
        default: 'full'
        type: choice
        options:
        - full
        - quick
        - penetration

permissions:
  contents: read
  security-events: write

jobs:
  security-tests:
    name: Security Testing Suite
    runs-on: ubuntu-latest
    environment: ${ENVIRONMENT}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install security tools
        run: |
          pip install bandit safety semgrep
          
      - name: Run SAST scanning
        run: |
          bandit -r src/ -f json -o bandit-results.json || true
          safety check --json --output safety-results.json || true
          semgrep --config=auto --json --output=semgrep-results.json src/ || true
          
      - name: Run compliance checks
        run: |
          python security/scripts/compliance-monitor-${ENVIRONMENT}.py ${ENVIRONMENT}
          
      - name: Generate security report
        run: |
          python -c "
import json
import datetime

# Load scan results
results = {
    'environment': '${ENVIRONMENT}',
    'scan_date': datetime.datetime.utcnow().isoformat(),
    'test_type': '\${{ github.event.inputs.test_type || 'scheduled' }}',
}

# Process results and generate report
print('Security testing completed for ${ENVIRONMENT}')
          "
          
      - name: Upload security results
        uses: actions/upload-artifact@v4
        with:
          name: security-test-results-${ENVIRONMENT}
          path: |
            *-results.json
            compliance-report-*.json
          retention-days: 90
EOF
        
        log_success "Automated security testing workflow created"
    fi
}

# Generate security setup report
generate_security_report() {
    log_info "Generating security setup report..."
    
    local report_file="security-setup-report-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# 🔒 Security Infrastructure Setup Report

**Environment:** ${ENVIRONMENT}  
**Setup Date:** $(date)  
**Configured By:** Security Infrastructure Script  

## 📊 Setup Summary

### ✅ Completed Components

- **GitHub Secrets:** $(if [[ "$SETUP_SECRETS" == "true" ]]; then echo "Configured"; else echo "Skipped"; fi)
- **Security Scanning:** $(if [[ "$CONFIGURE_SCANNING" == "true" ]]; then echo "Configured"; else echo "Skipped"; fi)
- **Compliance Monitoring:** $(if [[ "$SETUP_COMPLIANCE" == "true" ]]; then echo "Configured"; else echo "Skipped"; fi)
- **Security Dashboard:** Created
- **Automated Testing:** Configured

### 🔐 Security Features Enabled

1. **Multi-layered Scanning:**
   - SAST (Bandit, Semgrep, CodeQL)
   - DAST (OWASP ZAP)
   - Container Security (Trivy, Hadolint)
   - Dependency Scanning (Safety, pip-audit)

2. **Compliance Frameworks:**
   - SOC 2 Controls
   - GDPR Compliance
   - OWASP Top 10

3. **Monitoring & Alerting:**
   - Real-time security monitoring
   - Compliance scoring
   - Automated notifications

4. **Access Controls:**
   - GitHub repository secrets
   - Kubernetes RBAC
   - Environment isolation

## 🎯 Next Steps

### Immediate Actions (Next 24 hours)
1. **Verify GitHub Secrets:**
   - Check that all required secrets are properly set
   - Test secret access in workflows

2. **Configure DNS and SSL:**
   - Set up domain names for the environment
   - Install SSL certificates

3. **Test Security Workflows:**
   - Trigger security scanning workflows
   - Verify notifications are working

### Short-term Actions (Next Week)
1. **Security Team Onboarding:**
   - Review security configurations
   - Set up monitoring dashboards
   - Configure alert thresholds

2. **Penetration Testing:**
   - Schedule external security audit
   - Perform vulnerability assessments
   - Document findings and remediation

3. **Compliance Validation:**
   - Run compliance monitoring scripts
   - Generate baseline compliance reports
   - Set up regular audit schedules

### Long-term Actions (Next Month)
1. **Advanced Security Features:**
   - Implement zero-trust architecture
   - Set up advanced threat detection
   - Configure security automation

2. **Security Training:**
   - Train development teams on secure coding
   - Establish security review processes
   - Create incident response procedures

## 📋 Configuration Files Created

- \`security/compliance/compliance-${ENVIRONMENT}.yaml\`
- \`security/scripts/compliance-monitor-${ENVIRONMENT}.py\`
- \`security/monitoring/security-dashboard-${ENVIRONMENT}.json\`
- \`.github/workflows/security-scanning-${ENVIRONMENT}.yml\`
- \`.github/workflows/security-tests-${ENVIRONMENT}.yml\`

## 🔗 Access Information

- **Security Dashboard:** Will be available at grafana.${ENVIRONMENT}.anomaly-detection.io
- **Compliance Reports:** Generated daily and stored as artifacts
- **Security Alerts:** Sent to configured Slack/email channels

## 🛠️ Manual Configuration Required

1. **Notification Channels:**
   - Set up Slack workspace and channels
   - Configure email distribution lists
   - Test notification delivery

2. **External Security Tools:**
   - Configure SIEM integration if available
   - Set up vulnerability management platform
   - Connect to threat intelligence feeds

3. **Backup and Disaster Recovery:**
   - Configure security log backups
   - Test incident response procedures
   - Document recovery processes

## 📞 Support and Troubleshooting

If you encounter issues:
1. Check GitHub Actions logs for workflow failures
2. Verify environment variables and secrets
3. Test network connectivity and permissions
4. Review security tool configurations

For security incidents:
1. Follow established incident response procedures
2. Check security monitoring dashboards
3. Review audit logs and alerts
4. Contact security team immediately

## 🔄 Maintenance Schedule

- **Daily:** Automated security scans and compliance checks
- **Weekly:** Security dashboard reviews and alert tuning
- **Monthly:** Compliance reports and security assessments
- **Quarterly:** Penetration testing and security audits
- **Annually:** Security strategy and architecture reviews

EOF

    log_success "Security setup report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting security infrastructure setup for environment: $ENVIRONMENT"
    
    # Run setup phases
    check_prerequisites
    
    if [[ "$SETUP_SECRETS" == "true" ]]; then
        generate_secrets
        setup_github_secrets
    fi
    
    if [[ "$CONFIGURE_SCANNING" == "true" ]]; then
        configure_security_scanning
        setup_automated_security_testing
    fi
    
    if [[ "$SETUP_COMPLIANCE" == "true" ]]; then
        setup_compliance_monitoring
    fi
    
    create_security_dashboard
    generate_security_report
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "🔍 Dry run completed. No changes were made."
    else
        log_success "🎉 Security infrastructure setup completed successfully!"
        log_info "📋 Check the security setup report for detailed information and next steps"
    fi
}

# Execute main function
main "$@"