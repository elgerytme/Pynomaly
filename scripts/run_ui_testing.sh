#!/bin/bash

# UI Testing Automation Script for Pynomaly
# This script runs comprehensive UI tests using Docker and Playwright

set -e

echo "🚀 Starting Pynomaly UI Testing Pipeline"
echo "========================================"

# Configuration
COMPOSE_FILE="docker-compose.ui-testing.yml"
TEST_TIMEOUT="600s"
RETRY_COUNT=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to check if Docker is running
check_docker() {
    print_status "Checking Docker availability..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    print_success "Docker is available and running"
}

# Function to check if docker-compose is available
check_docker_compose() {
    print_status "Checking Docker Compose availability..."
    
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
    else
        print_error "Docker Compose is not available"
        exit 1
    fi
    
    print_success "Docker Compose is available: $DOCKER_COMPOSE_CMD"
}

# Function to prepare directories
prepare_directories() {
    print_status "Preparing test directories..."
    
    # Create required directories
    mkdir -p test-results
    mkdir -p screenshots
    mkdir -p reports
    mkdir -p visual-baselines
    
    # Set permissions
    chmod 755 test-results screenshots reports visual-baselines
    
    print_success "Test directories prepared"
}

# Function to build Docker images
build_images() {
    print_status "Building Docker images..."
    
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE build --parallel
    
    if [ $? -eq 0 ]; then
        print_success "Docker images built successfully"
    else
        print_error "Failed to build Docker images"
        exit 1
    fi
}

# Function to start the application
start_application() {
    print_status "Starting Pynomaly application..."
    
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE up -d pynomaly-app
    
    # Wait for application to be healthy
    print_status "Waiting for application to be ready..."
    
    local retry_count=0
    local max_retries=30
    
    while [ $retry_count -lt $max_retries ]; do
        if $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE ps pynomaly-app | grep -q "healthy"; then
            print_success "Application is ready and healthy"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        print_status "Waiting for application... ($retry_count/$max_retries)"
        sleep 10
    done
    
    print_error "Application failed to become healthy within timeout"
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE logs pynomaly-app
    exit 1
}

# Function to run UI tests
run_ui_tests() {
    print_status "Running comprehensive UI tests..."
    
    # Run main UI test suite
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE run --rm ui-tests
    
    local ui_exit_code=$?
    
    if [ $ui_exit_code -eq 0 ]; then
        print_success "UI tests completed successfully"
    else
        print_warning "Some UI tests failed (exit code: $ui_exit_code)"
    fi
    
    return $ui_exit_code
}

# Function to run visual regression tests
run_visual_tests() {
    print_status "Running visual regression tests..."
    
    # Run visual regression tests
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE run --rm visual-tests
    
    local visual_exit_code=$?
    
    if [ $visual_exit_code -eq 0 ]; then
        print_success "Visual regression tests completed successfully"
    else
        print_warning "Some visual regression tests failed (exit code: $visual_exit_code)"
    fi
    
    return $visual_exit_code
}

# Function to generate comprehensive report
generate_report() {
    print_status "Generating comprehensive UI test report..."
    
    # Run the report generator
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE run --rm --entrypoint="" ui-tests python tests/ui/run_ui_tests.py
    
    if [ $? -eq 0 ]; then
        print_success "Comprehensive report generated"
    else
        print_warning "Report generation encountered issues"
    fi
}

# Function to collect artifacts
collect_artifacts() {
    print_status "Collecting test artifacts..."
    
    # Copy artifacts from containers
    docker cp $(docker-compose -f $COMPOSE_FILE ps -q ui-tests):/app/screenshots/. ./screenshots/ 2>/dev/null || true
    docker cp $(docker-compose -f $COMPOSE_FILE ps -q ui-tests):/app/reports/. ./reports/ 2>/dev/null || true
    docker cp $(docker-compose -f $COMPOSE_FILE ps -q ui-tests):/app/test-results/. ./test-results/ 2>/dev/null || true
    
    # List generated artifacts
    if [ -d "./reports" ] && [ "$(ls -A ./reports)" ]; then
        print_success "Test reports available in ./reports/"
        ls -la ./reports/
    fi
    
    if [ -d "./screenshots" ] && [ "$(ls -A ./screenshots)" ]; then
        print_success "Screenshots available in ./screenshots/"
        echo "Screenshot count: $(ls ./screenshots/*.png 2>/dev/null | wc -l)"
    fi
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down -v
    
    # Remove dangling images if any
    docker image prune -f &>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to print summary
print_summary() {
    echo ""
    echo "========================================"
    echo "🎯 UI Testing Summary"
    echo "========================================"
    
    local ui_result=$1
    local visual_result=$2
    
    if [ $ui_result -eq 0 ] && [ $visual_result -eq 0 ]; then
        print_success "All UI tests passed! 🎉"
        echo "✅ Layout validation: PASSED"
        echo "✅ UX flows: PASSED" 
        echo "✅ Visual regression: PASSED"
        echo "✅ Accessibility: PASSED"
        echo "✅ Responsive design: PASSED"
    else
        print_warning "Some tests failed or need attention"
        
        if [ $ui_result -ne 0 ]; then
            echo "⚠️  UI tests: NEEDS ATTENTION"
        else
            echo "✅ UI tests: PASSED"
        fi
        
        if [ $visual_result -ne 0 ]; then
            echo "⚠️  Visual tests: NEEDS ATTENTION"
        else
            echo "✅ Visual tests: PASSED"
        fi
    fi
    
    echo ""
    echo "📊 Generated Reports:"
    if [ -d "./reports" ] && [ "$(ls -A ./reports)" ]; then
        for report in ./reports/*.html; do
            if [ -f "$report" ]; then
                echo "  📄 $(basename "$report")"
            fi
        done
    else
        echo "  ⚠️  No HTML reports found"
    fi
    
    echo ""
    echo "📸 Screenshots captured: $(ls ./screenshots/*.png 2>/dev/null | wc -l)"
    echo ""
    echo "🔍 Next Steps:"
    echo "  1. Review detailed reports in ./reports/"
    echo "  2. Check screenshots in ./screenshots/"
    echo "  3. Address any failing tests or visual regressions"
    echo "  4. Re-run tests after fixes"
    echo ""
}

# Main execution
main() {
    # Parse command line arguments
    local skip_build=false
    local cleanup_only=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-build)
                skip_build=true
                shift
                ;;
            --cleanup-only)
                cleanup_only=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-build    Skip Docker image building"
                echo "  --cleanup-only  Only cleanup Docker resources"
                echo "  -h, --help      Show this help message"
                echo ""
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Handle cleanup-only mode
    if [ "$cleanup_only" = true ]; then
        cleanup
        exit 0
    fi
    
    # Trap to ensure cleanup on exit
    trap cleanup EXIT
    
    # Pre-flight checks
    check_docker
    check_docker_compose
    prepare_directories
    
    # Build images if not skipped
    if [ "$skip_build" = false ]; then
        build_images
    else
        print_status "Skipping Docker image build"
    fi
    
    # Start application
    start_application
    
    # Run tests
    run_ui_tests
    local ui_exit_code=$?
    
    run_visual_tests  
    local visual_exit_code=$?
    
    # Generate comprehensive report
    generate_report
    
    # Collect artifacts
    collect_artifacts
    
    # Print summary
    print_summary $ui_exit_code $visual_exit_code
    
    # Return appropriate exit code
    if [ $ui_exit_code -eq 0 ] && [ $visual_exit_code -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# Run main function
main "$@"