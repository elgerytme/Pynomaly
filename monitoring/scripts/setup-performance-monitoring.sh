#!/bin/bash

# Performance & Monitoring Setup Script
# Sets up comprehensive performance monitoring and alerting for the detection platform

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
DRY_RUN=false
SKIP_VALIDATION=false

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Set up performance monitoring and alerting for the detection platform.

OPTIONS:
    -e, --environment ENVIRONMENT    Target environment (staging|production)
    -d, --dry-run                   Show what would be configured without making changes
    -s, --skip-validation          Skip pre-setup validation
    -h, --help                     Show this help message

EXAMPLES:
    $0 -e staging                  Setup monitoring for staging
    $0 -e production               Setup monitoring for production
    $0 -e staging -d               Dry run for staging

ENVIRONMENT VARIABLES:
    PROMETHEUS_URL                 Prometheus server URL (default: http://localhost:9090)
    GRAFANA_URL                   Grafana server URL (default: http://localhost:3000)
    ALERTMANAGER_URL              AlertManager URL (default: http://localhost:9093)
    SLACK_WEBHOOK_URL             Slack webhook for alerts
    EMAIL_NOTIFICATION_LIST       Comma-separated list of email addresses

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -s|--skip-validation)
            SKIP_VALIDATION=true
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

# Validate environment parameter
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
    for cmd in kubectl helm python3; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_tools+=("$cmd")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check Python packages
    if ! python3 -c "import yaml, requests, psutil" >/dev/null 2>&1; then
        log_warning "Some Python packages are missing. Installing..."
        pip3 install pyyaml requests psutil
    fi
    
    log_success "Prerequisites check passed"
}

# Set up Prometheus configuration
setup_prometheus() {
    log_info "Setting up Prometheus configuration..."
    
    local prometheus_config_dir="monitoring/prometheus/${ENVIRONMENT}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create Prometheus configuration"
        return
    fi
    
    mkdir -p "$prometheus_config_dir"
    
    # Create Prometheus configuration
    cat > "${prometheus_config_dir}/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    environment: '${ENVIRONMENT}'
    cluster: 'anomaly-detection-${ENVIRONMENT}'

rule_files:
  - "performance-rules.yml"
  - "alerting-rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Application metrics
  - job_name: 'anomaly-detection-api'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - anomaly-detection
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: anomaly-detection-api
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: \$1:\$2
        target_label: __address__
    scrape_interval: 5s
    
  - job_name: 'anomaly-detection-worker'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - anomaly-detection
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: anomaly-detection-worker
    scrape_interval: 10s
    
  # Infrastructure metrics
  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/\${1}/proxy/metrics
        
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
        
  # Database metrics
  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter.anomaly-detection.svc.cluster.local:9187']
    scrape_interval: 30s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter.anomaly-detection.svc.cluster.local:9121']
    scrape_interval: 30s
    
  # Performance optimizer metrics
  - job_name: 'performance-optimizer'
    static_configs:
      - targets: ['performance-optimizer.anomaly-detection.svc.cluster.local:8080']
    scrape_interval: 60s
EOF
    
    # Copy performance monitoring rules
    cp "monitoring/performance/performance-monitoring.yaml" "${prometheus_config_dir}/"
    
    log_success "Prometheus configuration created"
}

# Set up Grafana dashboards
setup_grafana_dashboards() {
    log_info "Setting up Grafana dashboards..."
    
    local grafana_dashboard_dir="monitoring/grafana/${ENVIRONMENT}/dashboards"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create Grafana dashboards"
        return
    fi
    
    mkdir -p "$grafana_dashboard_dir"
    
    # Copy performance dashboard
    cp "monitoring/alerting/alerting-dashboard.json" "$grafana_dashboard_dir/performance-alerting-dashboard.json"
    
    # Copy existing detection dashboard
    if [[ -f "deploy/monitoring/grafana/dashboards/anomaly-detection-overview.json" ]]; then
        cp "deploy/monitoring/grafana/dashboards/anomaly-detection-overview.json" "$grafana_dashboard_dir/"
    fi
    
    # Create datasource configuration
    cat > "${grafana_dashboard_dir}/../datasources.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus.anomaly-detection.svc.cluster.local:9090
    isDefault: true
    
  - name: Loki
    type: loki
    access: proxy
    url: http://loki.anomaly-detection.svc.cluster.local:3100
    
  - name: AlertManager
    type: alertmanager
    access: proxy
    url: http://alertmanager.anomaly-detection.svc.cluster.local:9093
EOF
    
    log_success "Grafana dashboards configuration created"
}

# Set up AlertManager
setup_alertmanager() {
    log_info "Setting up AlertManager configuration..."
    
    local alertmanager_config_dir="monitoring/alertmanager/${ENVIRONMENT}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create AlertManager configuration"
        return
    fi
    
    mkdir -p "$alertmanager_config_dir"
    
    # Create AlertManager configuration
    cat > "${alertmanager_config_dir}/alertmanager.yml" << EOF
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@anomaly-detection.io'
  
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 5s
      repeat_interval: 5m
      
    - match:
        severity: warning
      receiver: 'warning-alerts'
      repeat_interval: 30m
      
    - match_re:
        service: performance|monitoring
      receiver: 'performance-team'
      
    - match_re:
        service: database|storage
      receiver: 'database-team'

receivers:
  - name: 'default'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL:-}'
        channel: '#${ENVIRONMENT}-alerts'
        title: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Severity:* {{ .Labels.severity }}
          *Environment:* ${ENVIRONMENT}
          {{ end }}
        
  - name: 'critical-alerts'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL:-}'
        channel: '#critical-alerts'
        title: '🚨 CRITICAL: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *🚨 CRITICAL ALERT 🚨*
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Environment:* ${ENVIRONMENT}
          *Runbook:* {{ .Annotations.runbook_url }}
          {{ end }}
    email_configs:
      - to: '${EMAIL_NOTIFICATION_LIST:-oncall@company.com}'
        subject: '🚨 CRITICAL: {{ .GroupLabels.alertname }} (${ENVIRONMENT})'
        body: |
          {{ range .Alerts }}
          Critical alert in ${ENVIRONMENT} environment:
          
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Severity: {{ .Labels.severity }}
          
          Please investigate immediately.
          {{ end }}
          
  - name: 'warning-alerts'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL:-}'
        channel: '#${ENVIRONMENT}-alerts'
        title: '⚠️ WARNING: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *⚠️ Warning Alert*
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Environment:* ${ENVIRONMENT}
          {{ end }}
          
  - name: 'performance-team'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL:-}'
        channel: '#performance-alerts'
        title: 'Performance Alert: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Performance Issue Detected*
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Environment:* ${ENVIRONMENT}
          {{ end }}
          
  - name: 'database-team'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL:-}'
        channel: '#database-alerts'
        title: 'Database Alert: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Database Issue Detected*
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Environment:* ${ENVIRONMENT}
          {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
    
  - source_match:
      alertname: 'ServiceDown'
    target_match_re:
      alertname: '.*'
    equal: ['instance']
EOF
    
    log_success "AlertManager configuration created"
}

# Deploy monitoring stack to Kubernetes
deploy_monitoring_stack() {
    log_info "Deploying monitoring stack to Kubernetes..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy monitoring stack"
        return
    fi
    
    # Create namespace
    kubectl create namespace monitoring-${ENVIRONMENT} --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring-${ENVIRONMENT} \
        --values monitoring/helm/prometheus/values-${ENVIRONMENT}.yaml \
        --set-file prometheus.prometheusSpec.additionalScrapeConfigs=monitoring/prometheus/${ENVIRONMENT}/prometheus.yml
    
    # Deploy performance optimizer
    kubectl apply -f - << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: performance-optimizer
  namespace: monitoring-${ENVIRONMENT}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: performance-optimizer
  template:
    metadata:
      labels:
        app: performance-optimizer
    spec:
      containers:
      - name: performance-optimizer
        image: python:3.11-slim
        command: ["python", "/app/performance-optimization.py", "--continuous"]
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: optimizer-script
          mountPath: /app
        env:
        - name: PROMETHEUS_URL
          value: "http://prometheus-kube-prometheus-prometheus.monitoring-${ENVIRONMENT}.svc.cluster.local:9090"
        - name: ENVIRONMENT
          value: "${ENVIRONMENT}"
      volumes:
      - name: optimizer-script
        configMap:
          name: performance-optimizer-config
---
apiVersion: v1
kind: Service
metadata:
  name: performance-optimizer
  namespace: monitoring-${ENVIRONMENT}
spec:
  selector:
    app: performance-optimizer
  ports:
  - port: 8080
    targetPort: 8080
EOF
    
    # Create ConfigMap for performance optimizer
    kubectl create configmap performance-optimizer-config \
        --from-file=performance-optimization.py=monitoring/performance/performance-optimization.py \
        --from-file=optimization-config.yaml=monitoring/performance/optimization-config.yaml \
        --namespace monitoring-${ENVIRONMENT} \
        -o yaml --dry-run=client | kubectl apply -f -
    
    log_success "Monitoring stack deployed"
}

# Set up performance optimization automation
setup_performance_automation() {
    log_info "Setting up performance optimization automation..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would set up performance automation"
        return
    fi
    
    # Create optimization config
    mkdir -p "monitoring/performance"
    
    cat > "monitoring/performance/optimization-config.yaml" << EOF
prometheus_url: "http://prometheus-kube-prometheus-prometheus.monitoring-${ENVIRONMENT}.svc.cluster.local:9090"
optimization_interval: 300
performance_thresholds:
  api_response_time: 1.0
  cpu_usage: 70.0
  memory_usage: 80.0
  error_rate: 0.01
  throughput_min: 10.0
  database_connection_threshold: 70
optimization_strategies:
  auto_scaling:
    enabled: true
    cpu_threshold: 70
    memory_threshold: 80
    scale_up_factor: 1.5
    scale_down_factor: 0.8
    cooldown_period: 300
  caching:
    enabled: true
    redis_optimization: true
    application_cache_tuning: true
  database_optimization:
    enabled: true
    connection_pool_tuning: true
    query_optimization: true
    index_recommendations: true
  resource_optimization:
    enabled: true
    garbage_collection_tuning: true
    memory_optimization: true
    cpu_affinity: true
alerting:
  slack_webhook: "${SLACK_WEBHOOK_URL:-}"
  email_notifications: true
  notification_cooldown: 900
reporting:
  generate_reports: true
  report_interval: 3600
  retention_days: 30
EOF
    
    # Create automation cron job
    kubectl apply -f - << EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: performance-optimization
  namespace: monitoring-${ENVIRONMENT}
spec:
  schedule: "*/5 * * * *"  # Every 5 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: performance-optimizer
            image: python:3.11-slim
            command: ["python", "/app/performance-optimization.py"]
            volumeMounts:
            - name: optimizer-script
              mountPath: /app
            env:
            - name: PROMETHEUS_URL
              value: "http://prometheus-kube-prometheus-prometheus.monitoring-${ENVIRONMENT}.svc.cluster.local:9090"
            - name: ENVIRONMENT
              value: "${ENVIRONMENT}"
          volumes:
          - name: optimizer-script
            configMap:
              name: performance-optimizer-config
          restartPolicy: OnFailure
EOF
    
    log_success "Performance optimization automation configured"
}

# Validate monitoring setup
validate_monitoring_setup() {
    log_info "Validating monitoring setup..."
    
    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        log_info "Skipping validation as requested"
        return
    fi
    
    # Check if services are accessible
    local prometheus_url="${PROMETHEUS_URL:-http://localhost:9090}"
    local grafana_url="${GRAFANA_URL:-http://localhost:3000}"
    
    # Test Prometheus
    if curl -f -s "${prometheus_url}/api/v1/query?query=up" >/dev/null 2>&1; then
        log_success "Prometheus is accessible"
    else
        log_warning "Prometheus may not be accessible at ${prometheus_url}"
    fi
    
    # Test Grafana
    if curl -f -s "${grafana_url}/api/health" >/dev/null 2>&1; then
        log_success "Grafana is accessible"
    else
        log_warning "Grafana may not be accessible at ${grafana_url}"
    fi
    
    # Validate Kubernetes deployments
    if kubectl get namespace monitoring-${ENVIRONMENT} >/dev/null 2>&1; then
        log_success "Monitoring namespace exists"
        
        if kubectl get deployment prometheus-kube-prometheus-operator -n monitoring-${ENVIRONMENT} >/dev/null 2>&1; then
            log_success "Prometheus operator is deployed"
        else
            log_warning "Prometheus operator deployment not found"
        fi
    else
        log_warning "Monitoring namespace not found"
    fi
    
    log_success "Monitoring validation completed"
}

# Generate setup report
generate_setup_report() {
    log_info "Generating monitoring setup report..."
    
    local report_file="monitoring-setup-report-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# 📊 Performance & Monitoring Setup Report

**Environment:** ${ENVIRONMENT}  
**Setup Date:** $(date)  
**Configuration Mode:** $(if [[ "$DRY_RUN" == "true" ]]; then echo "Dry Run"; else echo "Full Deployment"; fi)  

## ✅ Components Configured

### Monitoring Stack
- **Prometheus:** Configured with custom rules and targets
- **Grafana:** Dashboards and datasources configured  
- **AlertManager:** Alert routing and notification channels set up
- **Performance Optimizer:** Automated optimization system deployed

### Performance Monitoring
- **API Performance:** Response time, throughput, error rate monitoring
- **System Resources:** CPU, memory, disk usage tracking
- **Database Performance:** Connection pooling and query performance
- **Custom Metrics:** Anomaly detection specific metrics

### Alerting Configuration
- **Critical Alerts:** Immediate notification via Slack and email
- **Warning Alerts:** Team-specific notification channels
- **Performance Alerts:** Automated optimization triggers
- **SLA Monitoring:** Availability and MTTR tracking

### Automation Features
- **Auto-scaling:** CPU and memory-based scaling
- **Cache Optimization:** Redis and application cache tuning
- **Database Optimization:** Connection pool and query optimization
- **Performance Reports:** Automated hourly performance reports

## 🔗 Access Information

- **Prometheus:** http://prometheus.${ENVIRONMENT}.anomaly-detection.io
- **Grafana:** http://grafana.${ENVIRONMENT}.anomaly-detection.io
  - Username: admin
  - Password: [Check Kubernetes secret]
- **AlertManager:** http://alertmanager.${ENVIRONMENT}.anomaly-detection.io

## 📋 Configuration Files

- \`monitoring/prometheus/${ENVIRONMENT}/prometheus.yml\`
- \`monitoring/grafana/${ENVIRONMENT}/dashboards/\`
- \`monitoring/alertmanager/${ENVIRONMENT}/alertmanager.yml\`
- \`monitoring/performance/optimization-config.yaml\`

## 🎯 Next Steps

### Immediate (Next 1 hour)
1. **Verify Access:** Test all monitoring URLs and credentials
2. **Import Dashboards:** Import any additional custom dashboards
3. **Test Alerts:** Trigger test alerts to verify notification channels
4. **Configure DNS:** Set up DNS records for monitoring services

### Short-term (Next 24 hours)
1. **Baseline Metrics:** Establish performance baselines
2. **Alert Tuning:** Adjust alert thresholds based on initial data
3. **Team Training:** Train team members on monitoring tools
4. **Documentation:** Update runbooks with monitoring procedures

### Long-term (Next week)
1. **SLA Definition:** Define and implement SLA monitoring
2. **Capacity Planning:** Set up capacity planning dashboards
3. **Integration Testing:** Test end-to-end monitoring scenarios
4. **Optimization Review:** Review and tune performance optimizations

## 🛠️ Manual Configuration Required

1. **Notification Channels:**
   - Configure Slack webhook URLs
   - Set up email SMTP settings
   - Test notification delivery

2. **External Integrations:**
   - Connect to existing monitoring systems
   - Set up log aggregation
   - Configure security scanning integration

3. **Access Control:**
   - Set up RBAC for monitoring tools
   - Configure SSO integration
   - Create team-specific access policies

## 📞 Troubleshooting

### Common Issues
1. **Metrics Not Appearing:** Check service discovery and scrape targets
2. **Alerts Not Firing:** Verify AlertManager configuration and routing
3. **Dashboard Empty:** Confirm Prometheus data source configuration
4. **Performance Optimizer Not Working:** Check deployment logs and permissions

### Support Resources
- **Documentation:** docs/monitoring/
- **Runbooks:** docs/runbooks/monitoring/
- **Team Chat:** #monitoring-support
- **Escalation:** monitoring-team@company.com

## 🔄 Maintenance Schedule

- **Daily:** Review critical alerts and performance reports
- **Weekly:** Analyze performance trends and optimization effectiveness
- **Monthly:** Review and update alert thresholds and dashboards
- **Quarterly:** Comprehensive monitoring stack review and updates

---
*This monitoring setup provides comprehensive observability and automated optimization for the detection platform.*
EOF

    log_success "Setup report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting performance & monitoring setup for environment: $ENVIRONMENT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "🔍 Running in dry-run mode - no changes will be made"
    fi
    
    # Run setup phases
    check_prerequisites
    setup_prometheus
    setup_grafana_dashboards
    setup_alertmanager
    
    if [[ "$DRY_RUN" != "true" ]]; then
        deploy_monitoring_stack
        setup_performance_automation
    fi
    
    validate_monitoring_setup
    generate_setup_report
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "🔍 Dry run completed. Review the configuration files created."
    else
        log_success "🎉 Performance & monitoring setup completed successfully!"
        log_info "📊 Access your monitoring dashboards and start optimizing performance!"
    fi
}

# Execute main function
main "$@"