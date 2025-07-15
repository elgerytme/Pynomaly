#!/bin/bash
set -euo pipefail

# Advanced Monitoring Deployment Script
# This script deploys the comprehensive monitoring and observability stack

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment/monitoring"
CONFIG_DIR="${PROJECT_ROOT}/config"

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

# Check if required tools are installed
check_prerequisites() {
    log_info "Checking prerequisites..."

    local required_tools=("docker" "docker-compose" "curl" "jq")
    local missing_tools=()

    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and try again."
        exit 1
    fi

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker and try again."
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."

    local dirs=(
        "${PROJECT_ROOT}/logs"
        "${PROJECT_ROOT}/data/monitoring"
        "${PROJECT_ROOT}/data/anomaly_models"
        "${PROJECT_ROOT}/data/capacity_data"
        "${PROJECT_ROOT}/data/slo_data"
        "${PROJECT_ROOT}/data/dashboard_data"
        "${PROJECT_ROOT}/backup/monitoring"
        "${DEPLOYMENT_DIR}/config/logstash/pipeline"
        "${DEPLOYMENT_DIR}/config/logstash/config"
        "${DEPLOYMENT_DIR}/templates"
        "${DEPLOYMENT_DIR}/static"
    )

    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done

    log_success "Directories created successfully"
}

# Generate configuration files
generate_configurations() {
    log_info "Generating configuration files..."

    # Elasticsearch configuration
    cat > "${DEPLOYMENT_DIR}/config/elasticsearch.yml" << 'EOF'
cluster.name: pynomaly-logs
node.name: elasticsearch
network.host: 0.0.0.0
discovery.type: single-node
xpack.security.enabled: false
xpack.monitoring.collection.enabled: true
EOF

    # Logstash pipeline configuration
    cat > "${DEPLOYMENT_DIR}/config/logstash/pipeline/pynomaly.conf" << 'EOF'
input {
  beats {
    port => 5044
  }
  tcp {
    port => 5000
    codec => json_lines
  }
  udp {
    port => 5000
    codec => json_lines
  }
}

filter {
  if [fields][logtype] == "pynomaly" {
    # Parse JSON logs
    if [message] =~ /^\{.*\}$/ {
      json {
        source => "message"
      }
    }

    # Add timestamp
    if [timestamp] {
      date {
        match => [ "timestamp", "ISO8601" ]
      }
    }

    # Extract log level
    if [level] {
      mutate {
        uppercase => [ "level" ]
      }
    }

    # Parse correlation ID
    if [correlation_id] {
      mutate {
        add_field => { "trace_id" => "%{correlation_id}" }
      }
    }

    # Categorize logs
    if [logger_name] =~ /monitoring/ {
      mutate {
        add_field => { "log_category" => "monitoring" }
      }
    } else if [logger_name] =~ /api/ {
      mutate {
        add_field => { "log_category" => "api" }
      }
    } else if [logger_name] =~ /detection/ {
      mutate {
        add_field => { "log_category" => "detection" }
      }
    } else {
      mutate {
        add_field => { "log_category" => "application" }
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "pynomaly-logs-%{+YYYY.MM.dd}"
    template_name => "pynomaly-logs"
    template_pattern => "pynomaly-logs-*"
    template => {
      "index_patterns" => ["pynomaly-logs-*"],
      "settings" => {
        "number_of_shards" => 1,
        "number_of_replicas" => 0,
        "index.refresh_interval" => "5s"
      },
      "mappings" => {
        "properties" => {
          "@timestamp" => { "type" => "date" },
          "level" => { "type" => "keyword" },
          "logger_name" => { "type" => "keyword" },
          "message" => { "type" => "text" },
          "log_category" => { "type" => "keyword" },
          "trace_id" => { "type" => "keyword" },
          "correlation_id" => { "type" => "keyword" }
        }
      }
    }
  }

  stdout {
    codec => rubydebug
  }
}
EOF

    # Filebeat configuration
    cat > "${DEPLOYMENT_DIR}/config/filebeat.yml" << 'EOF'
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/pynomaly/*.log
  fields:
    logtype: pynomaly
  fields_under_root: true
  multiline.pattern: '^\d{4}-\d{2}-\d{2}'
  multiline.negate: true
  multiline.match: after

- type: docker
  enabled: true
  containers.ids:
    - "*"
  containers.path: "/var/lib/docker/containers"
  containers.stream: "all"

processors:
- add_host_metadata:
    when.not.contains.tags: forwarded
- add_docker_metadata: ~
- decode_json_fields:
    fields: ["message"]
    target: ""
    overwrite_keys: true

output.logstash:
  hosts: ["logstash:5044"]

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
EOF

    # Nginx configuration
    cat > "${DEPLOYMENT_DIR}/config/nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    gzip on;
    gzip_vary on;
    gzip_min_length 10240;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;

    server {
        listen 80;
        server_name localhost;

        location /static/ {
            alias /usr/share/nginx/html/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
EOF

    # Database initialization script
    cat > "${DEPLOYMENT_DIR}/config/init-monitoring-db.sql" << 'EOF'
-- Initialize monitoring database schema

-- SLO tracking table
CREATE TABLE IF NOT EXISTS slo_measurements (
    id SERIAL PRIMARY KEY,
    slo_name VARCHAR(255) NOT NULL,
    sli_name VARCHAR(255) NOT NULL,
    measurement_time TIMESTAMP NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    target_value DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_slo_measurements_time ON slo_measurements(measurement_time);
CREATE INDEX IF NOT EXISTS idx_slo_measurements_slo ON slo_measurements(slo_name);

-- Alert history table
CREATE TABLE IF NOT EXISTS alert_history (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(255) UNIQUE NOT NULL,
    severity VARCHAR(50) NOT NULL,
    source VARCHAR(255) NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    labels JSONB,
    metrics JSONB,
    correlation_id VARCHAR(255),
    created_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP,
    resolution_notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_alert_history_created ON alert_history(created_at);
CREATE INDEX IF NOT EXISTS idx_alert_history_severity ON alert_history(severity);
CREATE INDEX IF NOT EXISTS idx_alert_history_correlation ON alert_history(correlation_id);

-- Capacity predictions table
CREATE TABLE IF NOT EXISTS capacity_predictions (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    current_value DOUBLE PRECISION NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    recommended_action TEXT,
    threshold_breach_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_capacity_predictions_metric ON capacity_predictions(metric_name);
CREATE INDEX IF NOT EXISTS idx_capacity_predictions_date ON capacity_predictions(prediction_date);

-- Anomaly detections table
CREATE TABLE IF NOT EXISTS anomaly_detections (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    detection_time TIMESTAMP NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    algorithm VARCHAR(100) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_anomaly_detections_time ON anomaly_detections(detection_time);
CREATE INDEX IF NOT EXISTS idx_anomaly_detections_metric ON anomaly_detections(metric_name);

-- KPI tracking table
CREATE TABLE IF NOT EXISTS kpi_history (
    id SERIAL PRIMARY KEY,
    kpi_name VARCHAR(255) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    target_value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(50),
    category VARCHAR(100),
    measurement_time TIMESTAMP NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_kpi_history_name ON kpi_history(kpi_name);
CREATE INDEX IF NOT EXISTS idx_kpi_history_time ON kpi_history(measurement_time);

-- Create a view for recent SLO compliance
CREATE OR REPLACE VIEW recent_slo_compliance AS
SELECT
    slo_name,
    COUNT(*) as total_measurements,
    COUNT(*) FILTER (WHERE value >= target_value) as good_measurements,
    (COUNT(*) FILTER (WHERE value >= target_value) * 100.0 / COUNT(*)) as compliance_percentage,
    AVG(value) as average_value,
    MIN(measurement_time) as period_start,
    MAX(measurement_time) as period_end
FROM slo_measurements
WHERE measurement_time >= NOW() - INTERVAL '30 days'
GROUP BY slo_name;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pynomaly;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO pynomaly;
EOF

    log_success "Configuration files generated successfully"
}

# Create Docker network
create_network() {
    log_info "Creating Docker network..."

    if ! docker network ls | grep -q "pynomaly-network"; then
        docker network create pynomaly-network --driver bridge
        log_success "Created Docker network: pynomaly-network"
    else
        log_info "Docker network 'pynomaly-network' already exists"
    fi
}

# Build custom Docker images
build_images() {
    log_info "Building custom Docker images..."

    # Create Dockerfiles

    # Advanced Monitoring Dockerfile
    cat > "${DEPLOYMENT_DIR}/Dockerfile.advanced-monitoring" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional monitoring dependencies
RUN pip install --no-cache-dir \
    scikit-learn \
    pandas \
    numpy \
    plotly \
    fastapi \
    uvicorn \
    websockets \
    redis \
    psycopg2-binary \
    prometheus-client \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create directories
RUN mkdir -p /app/logs /app/data /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8090 8091

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Run the application
CMD ["python", "-m", "src.pynomaly.infrastructure.monitoring.advanced_monitoring"]
EOF

    # Anomaly Detector Dockerfile
    cat > "${DEPLOYMENT_DIR}/Dockerfile.anomaly-detector" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    scikit-learn \
    pandas \
    numpy \
    prometheus-client

COPY src/ ./src/
COPY config/ ./config/

RUN mkdir -p /app/models

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8092

CMD ["python", "-c", "from src.pynomaly.infrastructure.monitoring.advanced_monitoring import AnomalyDetector; import asyncio; import time; detector = AnomalyDetector(); print('Anomaly detector started'); time.sleep(3600)"]
EOF

    # Capacity Planner Dockerfile
    cat > "${DEPLOYMENT_DIR}/Dockerfile.capacity-planner" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    scikit-learn \
    pandas \
    numpy \
    prometheus-client

COPY src/ ./src/
COPY config/ ./config/

RUN mkdir -p /app/data

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8093

CMD ["python", "-c", "from src.pynomaly.infrastructure.monitoring.advanced_monitoring import CapacityPlanner; import time; planner = CapacityPlanner(); print('Capacity planner started'); time.sleep(3600)"]
EOF

    # SLO Monitor Dockerfile
    cat > "${DEPLOYMENT_DIR}/Dockerfile.slo-monitor" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    prometheus-client \
    psycopg2-binary

COPY src/ ./src/
COPY config/ ./config/

RUN mkdir -p /app/data

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8094

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8094/health || exit 1

CMD ["python", "-c", "from src.pynomaly.infrastructure.monitoring.advanced_monitoring import SLOMonitor; import time; monitor = SLOMonitor(); print('SLO monitor started'); time.sleep(3600)"]
EOF

    # BI Dashboard Dockerfile
    cat > "${DEPLOYMENT_DIR}/Dockerfile.bi-dashboard" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    jinja2 \
    plotly \
    pandas \
    numpy

COPY src/ ./src/
COPY config/ ./config/

RUN mkdir -p /app/templates /app/static /app/data

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "src.pynomaly.infrastructure.monitoring.business_intelligence_dashboard"]
EOF

    log_success "Docker images configuration created"
}

# Deploy the monitoring stack
deploy_stack() {
    log_info "Deploying advanced monitoring stack..."

    cd "${DEPLOYMENT_DIR}"

    # Pull base images
    log_info "Pulling base images..."
    docker-compose -f docker-compose.advanced-monitoring.yml pull

    # Build custom images
    log_info "Building custom images..."
    docker-compose -f docker-compose.advanced-monitoring.yml build

    # Start the stack
    log_info "Starting monitoring stack..."
    docker-compose -f docker-compose.advanced-monitoring.yml up -d

    log_success "Advanced monitoring stack deployed successfully"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."

    local services=(
        "elasticsearch:9200"
        "kibana:5601"
        "jaeger:16686"
        "bi-dashboard:8080"
    )

    for service in "${services[@]}"; do
        local host="${service%:*}"
        local port="${service#*:}"

        log_info "Waiting for $host:$port..."

        local retries=30
        while [ $retries -gt 0 ]; do
            if docker-compose -f "${DEPLOYMENT_DIR}/docker-compose.advanced-monitoring.yml" exec -T "$host" curl -f "http://localhost:$port" &> /dev/null; then
                log_success "$host:$port is ready"
                break
            fi

            retries=$((retries - 1))
            if [ $retries -eq 0 ]; then
                log_warning "$host:$port is not ready after waiting"
                break
            fi

            sleep 10
        done
    done
}

# Configure initial data and dashboards
configure_initial_setup() {
    log_info "Configuring initial setup..."

    # Wait a bit for all services to stabilize
    sleep 30

    # Create Kibana index patterns
    log_info "Creating Kibana index patterns..."
    curl -X POST "localhost:5601/api/saved_objects/index-pattern/pynomaly-logs-*" \
        -H "Content-Type: application/json" \
        -H "kbn-xsrf: true" \
        -d '{
            "attributes": {
                "title": "pynomaly-logs-*",
                "timeFieldName": "@timestamp"
            }
        }' || log_warning "Failed to create Kibana index pattern"

    log_success "Initial setup completed"
}

# Display service URLs
display_service_urls() {
    log_success "Advanced monitoring stack deployed successfully!"
    echo ""
    echo "Service URLs:"
    echo "============="
    echo "🎯 Business Intelligence Dashboard: http://localhost:8080"
    echo "📊 Kibana (Logs): http://localhost:5601"
    echo "🔍 Jaeger (Tracing): http://localhost:16686"
    echo "🌐 Traefik Dashboard: http://localhost:8081"
    echo "📈 Advanced Monitoring API: http://localhost:8090"
    echo ""
    echo "Service Status:"
    echo "==============="
    docker-compose -f "${DEPLOYMENT_DIR}/docker-compose.advanced-monitoring.yml" ps
    echo ""
    echo "To view logs: docker-compose -f ${DEPLOYMENT_DIR}/docker-compose.advanced-monitoring.yml logs -f [service_name]"
    echo "To stop stack: docker-compose -f ${DEPLOYMENT_DIR}/docker-compose.advanced-monitoring.yml down"
    echo ""
}

# Main deployment function
main() {
    log_info "Starting advanced monitoring deployment..."

    check_prerequisites
    create_directories
    generate_configurations
    create_network
    build_images
    deploy_stack
    wait_for_services
    configure_initial_setup
    display_service_urls

    log_success "Advanced monitoring deployment completed successfully!"
}

# Script options
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log_info "Stopping advanced monitoring stack..."
        cd "${DEPLOYMENT_DIR}"
        docker-compose -f docker-compose.advanced-monitoring.yml down
        log_success "Advanced monitoring stack stopped"
        ;;
    "restart")
        log_info "Restarting advanced monitoring stack..."
        cd "${DEPLOYMENT_DIR}"
        docker-compose -f docker-compose.advanced-monitoring.yml restart
        log_success "Advanced monitoring stack restarted"
        ;;
    "status")
        log_info "Advanced monitoring stack status:"
        cd "${DEPLOYMENT_DIR}"
        docker-compose -f docker-compose.advanced-monitoring.yml ps
        ;;
    "logs")
        log_info "Showing logs for advanced monitoring stack..."
        cd "${DEPLOYMENT_DIR}"
        docker-compose -f docker-compose.advanced-monitoring.yml logs -f "${2:-}"
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|status|logs [service]}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy the advanced monitoring stack"
        echo "  stop    - Stop the monitoring stack"
        echo "  restart - Restart the monitoring stack"
        echo "  status  - Show status of all services"
        echo "  logs    - Show logs (optionally for specific service)"
        exit 1
        ;;
esac
