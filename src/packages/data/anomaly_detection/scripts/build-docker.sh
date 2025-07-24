#!/bin/bash
# Docker build script for Anomaly Detection Service

set -e

# Configuration
PROJECT_NAME="anomaly-detection"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD)
VERSION=${1:-latest}
DOCKERFILE=${2:-Dockerfile.optimized}  # Use optimized Dockerfile by default for domain architecture

echo "🐳 Building Docker image for Anomaly Detection Service (Domain-Driven)"
echo "=================================================================="
echo "Project: $PROJECT_NAME"
echo "Version: $VERSION"
echo "Dockerfile: $DOCKERFILE"
echo "Build Date: $BUILD_DATE"
echo "VCS Ref: $VCS_REF"
echo "Architecture: Domain-Driven Design"
echo ""

# Build the image
echo "🔨 Building Docker image..."
docker build \
    --file "$DOCKERFILE" \
    --tag "$PROJECT_NAME:$VERSION" \
    --tag "$PROJECT_NAME:latest" \
    --build-arg BUILD_VERSION="$VERSION" \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg VCS_REF="$VCS_REF" \
    --progress=plain \
    .

echo ""
echo "✅ Docker image built successfully!"
echo "   Image: $PROJECT_NAME:$VERSION"
echo "   Size: $(docker images --format "table {{.Size}}" $PROJECT_NAME:$VERSION | tail -n1)"

# Show image details
echo ""
echo "📋 Image Details:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" "$PROJECT_NAME"

# Optional: Run security scan
if command -v trivy &> /dev/null; then
    echo ""
    echo "🔒 Running security scan..."
    trivy image "$PROJECT_NAME:$VERSION" --severity HIGH,CRITICAL
fi

echo ""
echo "🚀 Ready to run with:"
echo "   docker run -p 8000:8000 $PROJECT_NAME:$VERSION"
echo ""
echo "📝 Or use docker-compose:"
echo "   docker-compose up -d"