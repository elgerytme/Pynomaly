#!/bin/bash

# Build Docker images for all hexagonal architecture services
# Usage: ./build-images.sh [tag] [push]
# Example: ./build-images.sh latest push

set -e

# Configuration
REGISTRY="${DOCKER_REGISTRY:-hexagonal-architecture}"
TAG="${1:-latest}"
PUSH="${2:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DEPLOYMENT_ROOT="${PROJECT_ROOT}/src/packages/deployment"

echo "🏗️  Building Hexagonal Architecture Docker Images"
echo "=================================================="
echo "Registry: ${REGISTRY}"
echo "Tag: ${TAG}"
echo "Project Root: ${PROJECT_ROOT}"
echo "Push to registry: ${PUSH:-false}"
echo ""

# Function to build a service
build_service() {
    local service=$1
    local context_path=$2
    local dockerfile_path=$3
    
    echo "🔨 Building ${service}..."
    
    # Build for both development and production
    for target in development production; do
        image_name="${REGISTRY}/${service}:${TAG}-${target}"
        echo "  Building ${image_name}..."
        
        docker build \
            --file "${dockerfile_path}" \
            --target "${target}" \
            --tag "${image_name}" \
            --platform linux/amd64 \
            "${context_path}"
        
        echo "  ✅ Built ${image_name}"
        
        # Push if requested
        if [[ "${PUSH}" == "push" ]]; then
            echo "  📤 Pushing ${image_name}..."
            docker push "${image_name}"
            echo "  ✅ Pushed ${image_name}"
        fi
    done
    
    # Also tag the production version as latest
    if [[ "${TAG}" == "latest" ]]; then
        docker tag "${REGISTRY}/${service}:${TAG}-production" "${REGISTRY}/${service}:${TAG}"
        if [[ "${PUSH}" == "push" ]]; then
            docker push "${REGISTRY}/${service}:${TAG}"
        fi
    fi
}

# Build each service
echo "📦 Building Data Quality Service..."
build_service "data-quality" \
    "${PROJECT_ROOT}/src/packages" \
    "${DEPLOYMENT_ROOT}/docker/data-quality/Dockerfile"

echo ""
echo "📦 Building Machine Learning Service..."
build_service "machine-learning" \
    "${PROJECT_ROOT}/src/packages" \
    "${DEPLOYMENT_ROOT}/docker/machine-learning/Dockerfile"

echo ""
echo "📦 Building Anomaly Detection Service..."
build_service "anomaly-detection" \
    "${PROJECT_ROOT}/src/packages" \
    "${DEPLOYMENT_ROOT}/docker/anomaly-detection/Dockerfile"

echo ""
echo "📦 Building MLOps Service..."
build_service "mlops" \
    "${PROJECT_ROOT}/src/packages" \
    "${DEPLOYMENT_ROOT}/docker/mlops/Dockerfile"

echo ""
echo "🎉 All images built successfully!"
echo ""

# List built images
echo "📋 Built images:"
docker images | grep "${REGISTRY}" | grep "${TAG}"

echo ""
echo "💡 Usage examples:"
echo "  Docker Compose (development): docker-compose -f deployment/compose/development.yml up"
echo "  Docker Compose (production):  docker-compose -f deployment/compose/production.yml up"
echo "  Kubernetes (development):     kubectl apply -k deployment/kubernetes/overlays/development"
echo "  Kubernetes (staging):         kubectl apply -k deployment/kubernetes/overlays/staging"
echo "  Kubernetes (production):      kubectl apply -k deployment/kubernetes/overlays/production"