#!/bin/bash

# Deploy hexagonal architecture using Docker Compose
# Usage: ./deploy-docker-compose.sh [environment] [action]
# Example: ./deploy-docker-compose.sh production up

set -e

ENVIRONMENT="${1:-development}"
ACTION="${2:-up}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/../compose/${ENVIRONMENT}.yml"

# Validate environment
case "${ENVIRONMENT}" in
    development|production)
        echo "🚀 Deploying to ${ENVIRONMENT} environment"
        ;;
    *)
        echo "❌ Invalid environment: ${ENVIRONMENT}"
        echo "Valid environments: development, production"
        exit 1
        ;;
esac

# Validate action
case "${ACTION}" in
    up|down|restart|ps|logs|build)
        echo "📋 Action: ${ACTION}"
        ;;
    *)
        echo "❌ Invalid action: ${ACTION}"
        echo "Valid actions: up, down, restart, ps, logs, build"
        exit 1
        ;;
esac

# Check if compose file exists
if [[ ! -f "${COMPOSE_FILE}" ]]; then
    echo "❌ Compose file not found: ${COMPOSE_FILE}"
    exit 1
fi

echo ""
echo "🔧 Configuration:"
echo "  Environment: ${ENVIRONMENT}"
echo "  Action: ${ACTION}"
echo "  Compose File: ${COMPOSE_FILE}"
echo ""

# Function to check Docker
check_docker() {
    echo "🔍 Checking Docker..."
    if ! docker info >/dev/null 2>&1; then
        echo "❌ Docker is not running"
        echo "Please start Docker and try again"
        exit 1
    fi
    echo "✅ Docker is running"
    
    if ! command -v docker-compose >/dev/null 2>&1; then
        echo "❌ docker-compose not found"
        echo "Please install docker-compose and try again"
        exit 1
    fi
    echo "✅ docker-compose is available"
}

# Function to perform the deployment action
perform_action() {
    echo ""
    echo "🚀 Executing ${ACTION}..."
    
    case "${ACTION}" in
        up)
            # Build images first if in development
            if [[ "${ENVIRONMENT}" == "development" ]]; then
                echo "🔨 Building development images..."
                docker-compose -f "${COMPOSE_FILE}" build
            fi
            
            echo "🚀 Starting services..."
            docker-compose -f "${COMPOSE_FILE}" up -d
            
            echo ""
            echo "✅ Services started successfully!"
            echo ""
            echo "📊 Service Status:"
            docker-compose -f "${COMPOSE_FILE}" ps
            ;;
        down)
            echo "🛑 Stopping services..."
            docker-compose -f "${COMPOSE_FILE}" down
            echo "✅ Services stopped successfully!"
            ;;
        restart)
            echo "🔄 Restarting services..."
            docker-compose -f "${COMPOSE_FILE}" restart
            echo "✅ Services restarted successfully!"
            ;;
        ps)
            echo "📊 Service Status:"
            docker-compose -f "${COMPOSE_FILE}" ps
            ;;
        logs)
            echo "📜 Service Logs:"
            docker-compose -f "${COMPOSE_FILE}" logs --tail=100
            ;;
        build)
            echo "🔨 Building images..."
            docker-compose -f "${COMPOSE_FILE}" build
            echo "✅ Images built successfully!"
            ;;
    esac
}

# Function to show post-deployment information
show_info() {
    if [[ "${ACTION}" == "up" ]]; then
        echo ""
        echo "🔗 Access Information:"
        echo ""
        
        if [[ "${ENVIRONMENT}" == "development" ]]; then
            echo "🛠️  Development Services:"
            echo "  - Data Quality:      http://localhost:8000"
            echo "  - MLOps Experiments: http://localhost:8001"
            echo "  - MLOps Registry:    http://localhost:8002"
            echo "  - MLOps Config:      http://localhost:8003"
            echo "  - ML Training:       http://localhost:8004"
            echo "  - ML Prediction:     http://localhost:8005"
            echo "  - Anomaly Detection: http://localhost:8007"
            echo "  - Prometheus:        http://localhost:9090"
            echo "  - Grafana:           http://localhost:3000 (admin/dev123)"
        else
            echo "🏭 Production Services:"
            echo "  - Data Quality:      http://localhost:8000"
            echo "  - MLOps Experiments: http://localhost:8001"
            echo "  - MLOps Registry:    http://localhost:8002"
            echo "  - MLOps Config:      http://localhost:8003"
            echo "  - ML Training:       http://localhost:8004"
            echo "  - ML Prediction:     http://localhost:8005"
            echo "  - Anomaly Detection: http://localhost:8007"
            echo "  - Prometheus:        http://localhost:9090"
            echo "  - Grafana:           http://localhost:3000 (admin/admin123)"
        fi
        
        echo ""
        echo "🛠️  Useful Commands:"
        echo "  - View logs:    docker-compose -f ${COMPOSE_FILE} logs -f [service]"
        echo "  - Restart:      docker-compose -f ${COMPOSE_FILE} restart [service]"
        echo "  - Scale:        docker-compose -f ${COMPOSE_FILE} up -d --scale [service]=2"
        echo "  - Stop all:     docker-compose -f ${COMPOSE_FILE} down"
        echo "  - View status:  docker-compose -f ${COMPOSE_FILE} ps"
    fi
}

# Function to wait for services to be healthy
wait_for_services() {
    if [[ "${ACTION}" == "up" ]]; then
        echo ""
        echo "⏳ Waiting for services to be healthy..."
        
        local max_attempts=30
        local attempt=1
        
        while [[ ${attempt} -le ${max_attempts} ]]; do
            echo "  Attempt ${attempt}/${max_attempts}..."
            
            # Check if all services are running
            local running_services
            running_services=$(docker-compose -f "${COMPOSE_FILE}" ps --services --filter "status=running" | wc -l)
            local total_services
            total_services=$(docker-compose -f "${COMPOSE_FILE}" config --services | wc -l)
            
            if [[ ${running_services} -eq ${total_services} ]]; then
                echo "✅ All services are running!"
                break
            fi
            
            if [[ ${attempt} -eq ${max_attempts} ]]; then
                echo "⚠️  Some services may not be fully ready. Check logs for details."
                break
            fi
            
            sleep 5
            ((attempt++))
        done
    fi
}

# Main execution
main() {
    echo "🏗️  Hexagonal Architecture Docker Compose Deployment"
    echo "====================================================="
    
    check_docker
    perform_action
    
    if [[ "${ACTION}" == "up" ]]; then
        wait_for_services
    fi
    
    show_info
    
    echo ""
    echo "🎉 Deployment operation completed!"
}

main