#!/bin/bash
# Setup script for Pynomaly monitoring infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MONITORING_DIR="docker/monitoring"
COMPOSE_FILE="docker-compose.monitoring.yml"
PROJECT_NAME="pynomaly-monitoring"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating monitoring directories..."
    
    # Create directories if they don't exist
    mkdir -p "${MONITORING_DIR}/prometheus/alerts"
    mkdir -p "${MONITORING_DIR}/grafana/provisioning/datasources"
    mkdir -p "${MONITORING_DIR}/grafana/provisioning/dashboards"
    mkdir -p "${MONITORING_DIR}/grafana/dashboards"
    mkdir -p "${MONITORING_DIR}/alertmanager"
    mkdir -p "${MONITORING_DIR}/logstash/config"
    mkdir -p "${MONITORING_DIR}/logstash/pipeline"
    mkdir -p "${MONITORING_DIR}/logstash/templates"
    
    print_success "Directories created"
}

# Function to set permissions
set_permissions() {
    print_status "Setting permissions..."
    
    # Make sure directories are writable
    chmod -R 755 "${MONITORING_DIR}"
    
    # Create data directories with proper permissions
    mkdir -p data/prometheus data/grafana data/elasticsearch data/alertmanager
    chmod -R 777 data/
    
    print_success "Permissions set"
}

# Function to start monitoring stack
start_monitoring() {
    print_status "Starting monitoring stack..."
    
    cd "${MONITORING_DIR}"
    
    # Pull latest images
    print_status "Pulling Docker images..."
    docker-compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" pull
    
    # Start services
    print_status "Starting services..."
    docker-compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" up -d
    
    cd - > /dev/null
    
    print_success "Monitoring stack started"
}

# Function to check service health
check_services() {
    print_status "Checking service health..."
    
    sleep 10  # Wait for services to start
    
    # Define services and their health check URLs
    declare -A services=(
        ["Prometheus"]="http://localhost:9090/-/healthy"
        ["Grafana"]="http://localhost:3000/api/health"
        ["AlertManager"]="http://localhost:9093/-/healthy"
        ["Elasticsearch"]="http://localhost:9200/_cluster/health"
        ["Kibana"]="http://localhost:5601/api/status"
        ["Jaeger"]="http://localhost:16686/"
    )
    
    for service in "${!services[@]}"; do
        url="${services[$service]}"
        if curl -f -s "$url" > /dev/null 2>&1; then
            print_success "$service is healthy"
        else
            print_warning "$service may not be ready yet (this is normal during startup)"
        fi
    done
}

# Function to display access information
show_access_info() {
    print_success "Monitoring stack is running!"
    echo ""
    echo "Access the following services:"
    echo "  📊 Grafana:        http://localhost:3000 (admin/admin123)"
    echo "  📈 Prometheus:     http://localhost:9090"
    echo "  🚨 AlertManager:   http://localhost:9093"
    echo "  🔍 Kibana:         http://localhost:5601"
    echo "  📋 Jaeger:         http://localhost:16686"
    echo "  🔧 Elasticsearch:  http://localhost:9200"
    echo ""
    echo "To stop the monitoring stack:"
    echo "  docker-compose -f ${MONITORING_DIR}/${COMPOSE_FILE} -p ${PROJECT_NAME} down"
    echo ""
    echo "To view logs:"
    echo "  docker-compose -f ${MONITORING_DIR}/${COMPOSE_FILE} -p ${PROJECT_NAME} logs -f [service_name]"
    echo ""
    echo "For troubleshooting, see: ${MONITORING_DIR}/README.md"
}

# Function to setup Pynomaly application configuration
setup_app_config() {
    print_status "Setting up application configuration..."
    
    cat > .env.monitoring << EOF
# Monitoring configuration for Pynomaly
PROMETHEUS_ENABLED=true
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
JAEGER_ENDPOINT=http://localhost:14268/api/traces
ELASTICSEARCH_URL=http://localhost:9200

# Logging configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
ENABLE_CORRELATION=true
ENABLE_STRUCTLOG=true

# Metrics configuration
METRICS_ENABLED=true
METRICS_PORT=8000
METRICS_PATH=/metrics

# Tracing configuration
TRACING_ENABLED=true
TRACING_SERVICE_NAME=pynomaly
TRACING_ENVIRONMENT=development
EOF
    
    print_success "Application configuration created (.env.monitoring)"
    print_warning "Remember to configure your Pynomaly application to use these settings"
}

# Main execution
main() {
    echo "🚀 Pynomaly Monitoring Setup"
    echo "============================="
    echo ""
    
    check_prerequisites
    create_directories
    set_permissions
    start_monitoring
    setup_app_config
    check_services
    show_access_info
    
    print_success "Setup complete! 🎉"
}

# Handle script arguments
case "${1:-}" in
    "stop")
        print_status "Stopping monitoring stack..."
        cd "${MONITORING_DIR}"
        docker-compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" down
        cd - > /dev/null
        print_success "Monitoring stack stopped"
        ;;
    "restart")
        print_status "Restarting monitoring stack..."
        cd "${MONITORING_DIR}"
        docker-compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" restart
        cd - > /dev/null
        print_success "Monitoring stack restarted"
        ;;
    "status")
        print_status "Checking monitoring stack status..."
        cd "${MONITORING_DIR}"
        docker-compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" ps
        cd - > /dev/null
        ;;
    "logs")
        service="${2:-}"
        if [ -n "$service" ]; then
            print_status "Showing logs for $service..."
            cd "${MONITORING_DIR}"
            docker-compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" logs -f "$service"
            cd - > /dev/null
        else
            print_status "Showing logs for all services..."
            cd "${MONITORING_DIR}"
            docker-compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" logs -f
            cd - > /dev/null
        fi
        ;;
    "clean")
        print_warning "This will remove all monitoring data. Are you sure? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            print_status "Cleaning monitoring data..."
            cd "${MONITORING_DIR}"
            docker-compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" down -v
            cd - > /dev/null
            rm -rf data/
            print_success "Monitoring data cleaned"
        else
            print_status "Cancelled"
        fi
        ;;
    "help")
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  (none)    Start the monitoring stack"
        echo "  stop      Stop the monitoring stack"
        echo "  restart   Restart the monitoring stack"
        echo "  status    Show service status"
        echo "  logs      Show logs (optionally for specific service)"
        echo "  clean     Remove all monitoring data"
        echo "  help      Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                # Start monitoring"
        echo "  $0 stop          # Stop monitoring"
        echo "  $0 logs grafana  # Show Grafana logs"
        echo "  $0 clean         # Remove all data"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac