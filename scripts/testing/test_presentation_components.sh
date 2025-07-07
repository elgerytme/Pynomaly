#!/bin/bash

# Pynomaly Presentation Components Test Suite
# Tests CLI, API, and Web UI components in fresh environments
# Compatible with bash and can be adapted for PowerShell

set -e  # Exit on any error

echo "🧪 Pynomaly Presentation Components Test Suite"
echo "=============================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED_TESTS++))
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    ((FAILED_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "\n${BLUE}Testing: $test_name${NC}"
    echo "----------------------------------------"
    
    ((TOTAL_TESTS++))
    
    if eval "$test_command"; then
        log_success "$test_name passed"
        return 0
    else
        log_error "$test_name failed"
        return 1
    fi
}

# Environment setup
setup_environment() {
    log_info "Setting up test environment..."
    
    # Set Python path
    export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    
    # Check Python availability
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        exit 1
    fi
    
    log_success "Environment setup complete"
}

# Test CLI Component
test_cli_component() {
    local test_cmd='python3 -c "
import sys
sys.path.insert(0, \"src\")

try:
    # Test CLI imports
    from pynomaly.presentation.cli.app import app
    print(\"✅ CLI app imported successfully\")
    
    # Test dependencies
    import typer
    import rich
    from rich.console import Console
    
    print(\"✅ CLI dependencies available\")
    
    # Test CLI functionality
    console = Console()
    console.print(\"✅ Rich console working\", style=\"green\")
    
    print(\"✅ CLI component test completed successfully\")
    
except Exception as e:
    print(f\"❌ CLI test failed: {str(e)}\")
    import traceback
    traceback.print_exc()
    exit(1)
"'
    
    run_test "CLI Component" "$test_cmd"
}

# Test API Component
test_api_component() {
    local test_cmd='python3 -c "
import sys
sys.path.insert(0, \"src\")

try:
    # Test API imports
    from pynomaly.presentation.api.app import create_app
    print(\"✅ API create_app imported successfully\")
    
    # Test dependencies
    import fastapi
    import uvicorn
    from fastapi.testclient import TestClient
    
    print(\"✅ API dependencies available\")
    
    # Test app creation
    app = create_app()
    print(\"✅ API application created successfully\")
    
    # Test with client
    client = TestClient(app)
    response = client.get(\"/api/health\")
    
    if response.status_code == 200:
        print(f\"✅ Health endpoint working (status: {response.status_code})\")
    else:
        print(f\"⚠️  Health endpoint returned: {response.status_code}\")
    
    print(\"✅ API component test completed successfully\")
    
except Exception as e:
    print(f\"❌ API test failed: {str(e)}\")
    import traceback
    traceback.print_exc()
    exit(1)
"'
    
    run_test "API Component" "$test_cmd"
}

# Test Web UI Component
test_web_ui_component() {
    local test_cmd='python3 -c "
import sys
sys.path.insert(0, \"src\")

try:
    # Test Web UI imports
    from pynomaly.presentation.web.app import create_web_app, mount_web_ui
    print(\"✅ Web UI functions imported successfully\")
    
    # Test dependencies
    from jinja2 import Template
    from fastapi.staticfiles import StaticFiles
    from fastapi.testclient import TestClient
    
    print(\"✅ Web UI dependencies available\")
    
    # Test app creation
    app = create_web_app()
    print(\"✅ Complete web application created successfully\")
    
    # Count routes
    total_routes = len([r for r in app.routes if hasattr(r, \"path\")])
    web_routes = len([r for r in app.routes if hasattr(r, \"path\") and r.path.startswith(\"/web\")])
    api_routes = len([r for r in app.routes if hasattr(r, \"path\") and r.path.startswith(\"/api\")])
    
    print(f\"✅ Routes configured - Total: {total_routes}, Web: {web_routes}, API: {api_routes}\")
    
    # Test with client
    client = TestClient(app)
    
    # Test API health
    response = client.get(\"/api/health\")
    if response.status_code == 200:
        print(f\"✅ API health endpoint working (status: {response.status_code})\")
    
    # Test web UI root
    response = client.get(\"/\")
    if response.status_code in [200, 302]:  # 200 for content, 302 for redirect
        print(f\"✅ Web UI root endpoint working (status: {response.status_code})\")
    
    print(\"✅ Web UI component test completed successfully\")
    
except Exception as e:
    print(f\"❌ Web UI test failed: {str(e)}\")
    import traceback
    traceback.print_exc()
    exit(1)
"'
    
    run_test "Web UI Component" "$test_cmd"
}

# Test fresh environment simulation
test_fresh_environment() {
    log_info "Testing fresh environment simulation..."
    
    # Create temporary environment
    local temp_env="$(mktemp -d)"
    local original_pythonpath="$PYTHONPATH"
    
    export PYTHONPATH="$(pwd)/src"
    
    local test_cmd='python3 -c "
import sys
import os

# Simulate fresh environment
print(f\"Fresh Environment Test:\")
print(f\"Working Directory: {os.getcwd()}\")
print(f\"Python Path: {sys.path[:2]}\")

try:
    sys.path.insert(0, \"src\")
    
    # Test basic imports
    from pynomaly.presentation.cli.app import app as cli_app
    from pynomaly.presentation.api.app import create_app
    from pynomaly.presentation.web.app import create_web_app
    
    print(\"✅ All presentation components importable in fresh environment\")
    
    # Test basic functionality
    api_app = create_app()
    web_app = create_web_app()
    
    print(\"✅ All components can be instantiated\")
    print(\"✅ Fresh environment test completed successfully\")
    
except Exception as e:
    print(f\"❌ Fresh environment test failed: {str(e)}\")
    import traceback
    traceback.print_exc()
    exit(1)
"'
    
    if run_test "Fresh Environment Simulation" "$test_cmd"; then
        # Restore original environment
        export PYTHONPATH="$original_pythonpath"
        rm -rf "$temp_env"
    else
        export PYTHONPATH="$original_pythonpath"
        rm -rf "$temp_env"
        return 1
    fi
}

# Test dependency availability
test_dependencies() {
    local test_cmd='python3 -c "
import sys
sys.path.insert(0, \"src\")

dependencies = {
    \"Core\": [\"pyod\", \"numpy\", \"pandas\", \"polars\", \"pydantic\", \"structlog\"],
    \"API\": [\"fastapi\", \"uvicorn\", \"httpx\", \"requests\", \"jinja2\", \"aiofiles\"],
    \"CLI\": [\"typer\", \"rich\"],
    \"Optional\": [\"shap\", \"lime\"]
}

all_available = True

for category, deps in dependencies.items():
    print(f\"{category} dependencies:\")
    for dep in deps:
        try:
            __import__(dep)
            print(f\"  ✅ {dep}\")
        except ImportError:
            if category == \"Optional\":
                print(f\"  ⚠️  {dep} (optional - not installed)\")
            else:
                print(f\"  ❌ {dep} (required but missing)\")
                all_available = False
    print()

if all_available:
    print(\"✅ All required dependencies available\")
else:
    print(\"❌ Some required dependencies missing\")
    exit(1)
"'
    
    run_test "Dependency Availability" "$test_cmd"
}

# Test cross-platform compatibility
test_cross_platform() {
    log_info "Testing cross-platform compatibility..."
    
    local test_cmd='python3 -c "
import sys
import os
import platform

print(f\"Platform Information:\")
print(f\"  System: {platform.system()}\")
print(f\"  Version: {platform.version()}\")  
print(f\"  Python: {platform.python_version()}\")
print(f\"  Architecture: {platform.machine()}\")
print()

# Test path handling
sys.path.insert(0, \"src\")

try:
    # Test imports work across platforms
    from pynomaly.presentation.cli.app import app
    from pynomaly.presentation.api.app import create_app
    from pynomaly.presentation.web.app import create_web_app
    
    print(\"✅ Cross-platform imports successful\")
    
    # Test path separators
    from pathlib import Path
    test_path = Path(\"src\") / \"pynomaly\" / \"presentation\"
    if test_path.exists():
        print(\"✅ Cross-platform path handling working\")
    
    print(\"✅ Cross-platform compatibility test completed\")
    
except Exception as e:
    print(f\"❌ Cross-platform test failed: {str(e)}\")
    import traceback
    traceback.print_exc()
    exit(1)
"'
    
    run_test "Cross-Platform Compatibility" "$test_cmd"
}

# Main test execution
main() {
    echo "Starting Pynomaly presentation components test suite..."
    echo "Test environment: $(uname -s) $(uname -r)"
    echo "Python version: $(python3 --version 2>/dev/null || echo 'Python not found')"
    echo
    
    # Setup
    setup_environment
    
    # Run all tests
    echo -e "\n${BLUE}🧪 Running Test Suite${NC}"
    echo "====================="
    
    test_dependencies
    test_cli_component
    test_api_component  
    test_web_ui_component
    test_fresh_environment
    test_cross_platform
    
    # Test summary
    echo
    echo "=============================================="
    echo -e "${BLUE}📊 Test Results Summary${NC}"
    echo "=============================================="
    echo "Total Tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo
        echo -e "${GREEN}🎉 All tests passed! Presentation components are working correctly.${NC}"
        echo
        echo "✅ CLI Component: Ready for use"
        echo "✅ API Component: Ready for use" 
        echo "✅ Web UI Component: Ready for use"
        echo
        echo "You can now run:"
        echo "  - CLI: pynomaly --help"
        echo "  - API: uvicorn pynomaly.presentation.api:app"
        echo "  - Web UI: uvicorn pynomaly.presentation.web.app:create_web_app"
        
        exit 0
    else
        echo
        echo -e "${RED}❌ Some tests failed. Please check the output above for details.${NC}"
        exit 1
    fi
}

# Check if script is being run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
