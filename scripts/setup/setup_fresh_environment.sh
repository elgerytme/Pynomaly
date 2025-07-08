#!/usr/bin/env bash
# Fresh Environment Setup Script for Pynomaly API
# This script sets up Pynomaly API in a fresh environment

set -e  # Exit on any error

echo "=== Pynomaly Fresh Environment Setup ==="
echo "Setting up Pynomaly API in a fresh environment..."

# Configuration
PROJECT_ROOT="/mnt/c/Users/andre/Pynomaly"
SRC_PATH="$PROJECT_ROOT/src"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
VENV_PATH="$PROJECT_ROOT/.fresh_venv"

# Function to print status
print_status() {
    echo "[INFO] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

print_success() {
    echo "[SUCCESS] $1"
}

# Step 1: Check Python availability
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
print_success "Python $PYTHON_VERSION found"

# Step 2: Create virtual environment (optional but recommended)
if [ "$1" = "--use-venv" ]; then
    print_status "Creating virtual environment..."
    if [ -d "$VENV_PATH" ]; then
        print_status "Removing existing virtual environment..."
        rm -rf "$VENV_PATH"
    fi

    python3 -m venv "$VENV_PATH" || {
        print_error "Failed to create virtual environment. Installing python3-venv..."
        sudo apt update && sudo apt install -y python3-venv python3-pip
        python3 -m venv "$VENV_PATH"
    }

    source "$VENV_PATH/bin/activate"
    print_success "Virtual environment activated"
fi

# Step 3: Install dependencies
print_status "Installing dependencies..."
if [ -f "$PROJECT_ROOT/requirements-server.txt" ]; then
    print_status "Installing server requirements (API + CLI functionality)..."
    pip install --break-system-packages -r "$PROJECT_ROOT/requirements-server.txt" || {
        print_error "Failed to install from requirements-server.txt, installing core dependencies..."
        # Install minimal core first
        pip install --break-system-packages \
            pyod numpy pandas polars pydantic structlog dependency-injector
        # Add server components
        pip install --break-system-packages \
            fastapi uvicorn httpx requests python-multipart jinja2 aiofiles \
            pydantic-settings typer rich scikit-learn scipy pyarrow
    }
elif [ -f "$REQUIREMENTS_FILE" ]; then
    print_status "Installing minimal requirements..."
    pip install --break-system-packages -r "$REQUIREMENTS_FILE"
    # Add server components for API testing
    pip install --break-system-packages \
        fastapi uvicorn httpx requests python-multipart jinja2 aiofiles \
        pydantic-settings typer rich scikit-learn scipy
else
    print_status "Installing core dependencies directly..."
    # Install minimal core
    pip install --break-system-packages \
        pyod numpy pandas polars pydantic structlog dependency-injector
    # Add server components for API functionality
    pip install --break-system-packages \
        fastapi uvicorn httpx requests python-multipart jinja2 aiofiles \
        pydantic-settings typer rich scikit-learn scipy pyarrow
fi

print_success "Dependencies installed"

# Step 4: Test import
print_status "Testing Pynomaly import..."
export PYTHONPATH="$SRC_PATH"
python3 -c "
try:
    from pynomaly.presentation.api import app
    print('✓ Pynomaly API imported successfully')
except ImportError as e:
    print('✗ Import failed:', e)
    exit(1)
"

print_success "Import test passed"

# Step 5: Start API server for testing
print_status "Starting API server for testing..."
TEST_PORT=8003
echo "Starting server on port $TEST_PORT..."
PYTHONPATH="$SRC_PATH" uvicorn pynomaly.presentation.api:app --host 127.0.0.1 --port $TEST_PORT &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Test endpoints
print_status "Testing API endpoints..."
if curl -s "http://127.0.0.1:$TEST_PORT/" > /dev/null; then
    print_success "Root endpoint accessible"
    ROOT_RESPONSE=$(curl -s "http://127.0.0.1:$TEST_PORT/")
    echo "  Response: $ROOT_RESPONSE"
else
    print_error "Root endpoint not accessible"
fi

if curl -s "http://127.0.0.1:$TEST_PORT/api/health/" > /dev/null; then
    print_success "Health endpoint accessible"
    HEALTH_STATUS=$(curl -s "http://127.0.0.1:$TEST_PORT/api/health/" | grep -o '"overall_status":"[^"]*"' | cut -d'"' -f4)
    echo "  Health status: $HEALTH_STATUS"
else
    print_error "Health endpoint not accessible"
fi

# Clean up
print_status "Stopping test server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

print_success "Fresh environment setup complete!"
echo ""
echo "To start the API server:"
echo "  export PYTHONPATH=$SRC_PATH"
echo "  uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000"
echo ""
echo "API will be available at:"
echo "  - Root: http://localhost:8000/"
echo "  - Health: http://localhost:8000/api/health/"
echo "  - Docs: http://localhost:8000/api/docs"
