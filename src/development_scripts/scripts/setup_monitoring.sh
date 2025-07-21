#!/bin/bash
# anomaly_detection Production Monitoring Setup Script

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MONITORING_DIR="$PROJECT_DIR/monitoring"
DOCKER_COMPOSE_FILE="$PROJECT_DIR/docker-compose.monitoring.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    log "Docker and Docker Compose are installed"
}

# Create monitoring directories
create_directories() {
    log "Creating monitoring directories..."

    mkdir -p "$MONITORING_DIR"/{prometheus,grafana,alertmanager,dashboards,rules}
    mkdir -p "$MONITORING_DIR/grafana"/{dashboards,datasources,provisioning}
    mkdir -p "$MONITORING_DIR/prometheus"/{rules,config}
    mkdir -p "$MONITORING_DIR/alertmanager"/{config,templates}

    log "Monitoring directories created"
}

# Generate Prometheus configuration
generate_prometheus_config() {
    log "Generating Prometheus configuration..."

    cat > "$MONITORING_DIR/prometheus/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'anomaly_detection'
    static_configs:
      - targets: ['anomaly_detection-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
    scrape_timeout: 10s

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 15s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 15s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s

  - job_name: 'grafana'
    static_configs:
      - targets: ['grafana:3000']
    scrape_interval: 15s

  - job_name: 'alertmanager'
    static_configs:
      - targets: ['alertmanager:9093']
    scrape_interval: 15s
EOF

    log "Prometheus configuration generated"
}

# Generate Alertmanager configuration
generate_alertmanager_config() {
    log "Generating Alertmanager configuration..."

    cat > "$MONITORING_DIR/alertmanager/alertmanager.yml" << EOF
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'anomaly_detection-alerts@your-domain.com'
  smtp_auth_username: 'alerts@your-domain.com'
  smtp_auth_password: 'your-email-password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://localhost:5001/webhook'
        send_resolved: true

  - name: 'critical-alerts'
    email_configs:
      - to: 'admin@your-domain.com'
        subject: 'CRITICAL: anomaly_detection Alert - {{ .GroupLabels.alertname }}'
        body: |
          Alert: {{ .GroupLabels.alertname }}
          Summary: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
          Description: {{ range .Alerts }}{{ .Annotations.description }}{{ end }}

          Details:
          {{ range .Alerts }}
          - Alert: {{ .Labels.alertname }}
          - Severity: {{ .Labels.severity }}
          - Service: {{ .Labels.service }}
          - Component: {{ .Labels.component }}
          {{ end }}
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#anomaly_detection-alerts'
        title: 'CRITICAL: anomaly_detection Alert'
        text: |
          Alert: {{ .GroupLabels.alertname }}
          {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}

  - name: 'warning-alerts'
    email_configs:
      - to: 'team@your-domain.com'
        subject: 'WARNING: anomaly_detection Alert - {{ .GroupLabels.alertname }}'
        body: |
          Alert: {{ .GroupLabels.alertname }}
          Summary: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
          Description: {{ range .Alerts }}{{ .Annotations.description }}{{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF

    log "Alertmanager configuration generated"
}

# Generate Grafana provisioning
generate_grafana_provisioning() {
    log "Generating Grafana provisioning configuration..."

    # Datasources
    cat > "$MONITORING_DIR/grafana/datasources/prometheus.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      timeInterval: 15s
EOF

    # Dashboards
    cat > "$MONITORING_DIR/grafana/dashboards/dashboard.yml" << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: 'anomaly_detection'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    log "Grafana provisioning configuration generated"
}

# Generate Docker Compose for monitoring
generate_docker_compose() {
    log "Generating Docker Compose for monitoring..."

    cat > "$DOCKER_COMPOSE_FILE" << EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: anomaly_detection-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert_rules.yml:/etc/prometheus/rules/alert_rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--storage.tsdb.retention.time=30d'
    restart: unless-stopped
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: anomaly_detection-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:latest
    container_name: anomaly_detection-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    restart: unless-stopped
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:latest
    container_name: anomaly_detection-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    restart: unless-stopped
    networks:
      - monitoring

  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    container_name: anomaly_detection-postgres-exporter
    ports:
      - "9187:9187"
    environment:
      - DATA_SOURCE_NAME=postgresql://anomaly_detection_user:your-password@postgres:5432/anomaly_detection_prod?sslmode=disable
    restart: unless-stopped
    networks:
      - monitoring

  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: anomaly_detection-redis-exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis:6379
    restart: unless-stopped
    networks:
      - monitoring

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: anomaly_detection-cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    devices:
      - /dev/kmsg
    restart: unless-stopped
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:

networks:
  monitoring:
    driver: bridge
EOF

    log "Docker Compose for monitoring generated"
}

# Display access information
display_access_info() {
    log "Monitoring setup completed successfully!"
    echo ""
    echo "Access Information:"
    echo "===================="
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000 (admin/admin123)"
    echo "Alertmanager: http://localhost:9093"
    echo "Node Exporter: http://localhost:9100"
    echo "cAdvisor: http://localhost:8080"
    echo ""
    echo "Next Steps:"
    echo "==========="
    echo "1. Change the default Grafana password"
    echo "2. Configure email/Slack notifications in Alertmanager"
    echo "3. Import additional dashboards"
    echo "4. Set up SSL certificates"
    echo "5. Configure firewall rules"
    echo ""
}

# Main function
main() {
    log "Starting anomaly_detection monitoring setup..."

    check_docker
    create_directories
    generate_prometheus_config
    generate_alertmanager_config
    generate_grafana_provisioning
    generate_docker_compose
    display_access_info

    log "Monitoring configuration generated successfully!"
    log "Run './scripts/setup_monitoring.sh start' to start the monitoring stack"
}

# Handle script arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "start")
        log "Starting monitoring services..."
        cd "$PROJECT_DIR"
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
        ;;
    "stop")
        log "Stopping monitoring services..."
        cd "$PROJECT_DIR"
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
        ;;
    "restart")
        log "Restarting monitoring services..."
        cd "$PROJECT_DIR"
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
        ;;
    "logs")
        cd "$PROJECT_DIR"
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
        ;;
    "status")
        cd "$PROJECT_DIR"
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        ;;
    "clean")
        log "Cleaning up monitoring services..."
        cd "$PROJECT_DIR"
        docker-compose -f "$DOCKER_COMPOSE_FILE" down -v
        docker volume prune -f
        ;;
    "help")
        echo "Usage: $0 {setup|start|stop|restart|logs|status|clean|help}"
        echo ""
        echo "Commands:"
        echo "  setup    - Initial setup of monitoring stack"
        echo "  start    - Start monitoring services"
        echo "  stop     - Stop monitoring services"
        echo "  restart  - Restart monitoring services"
        echo "  logs     - Show monitoring services logs"
        echo "  status   - Show monitoring services status"
        echo "  clean    - Clean up monitoring services and volumes"
        echo "  help     - Show this help message"
        ;;
    *)
        error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
