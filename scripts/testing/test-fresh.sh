#!/usr/bin/env bash
# Test script for fresh bash environment

echo "=== Testing Pynomaly Web App in Fresh Bash Environment ==="
echo "Date: $(date)"
echo "Current directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Shell: $SHELL"
echo

# Change to project root
cd "$(dirname "$0")/.."

echo "Project root: $(pwd)"
echo

# Create temporary test environment
TEST_ENV_DIR="test_environments/fresh_bash_test"
echo "Creating fresh test environment in: $TEST_ENV_DIR"

# Clean up any existing test environment
if [ -d "$TEST_ENV_DIR" ]; then
    rm -rf "$TEST_ENV_DIR"
fi

mkdir -p "$TEST_ENV_DIR"
cd "$TEST_ENV_DIR"

echo "✓ Test environment created"
echo

# Test 1: Create virtual environment
echo "Test 1: Creating virtual environment..."
python3 -m venv fresh_venv 2>/dev/null || {
    echo "⚠️  Virtual environment creation failed, testing with system Python"
    SKIP_VENV=true
}

if [ "$SKIP_VENV" != "true" ]; then
    source fresh_venv/bin/activate
    echo "✓ Virtual environment activated"
    echo "Python path: $(which python3)"
else
    echo "⚠️  Using system Python"
fi
echo

# Test 2: Install minimal dependencies
echo "Test 2: Installing minimal dependencies..."
python3 -m pip install --quiet --no-warn-script-location \
    fastapi uvicorn pydantic dependency-injector \
    pandas numpy scikit-learn || {
    echo "❌ Dependency installation failed"
    exit 1
}
echo "✓ Dependencies installed"
echo

# Test 3: Copy source code to fresh environment
echo "Test 3: Setting up source code..."
cp -r ../../src .
cp -r ../../scripts .
echo "✓ Source code copied"
echo

# Test 4: Test imports in fresh environment
echo "Test 4: Testing imports in fresh environment..."
PYTHONPATH="$(pwd)/src" python3 -c "
try:
    from pynomaly.presentation.web.app import create_web_app
    print('✓ Import successful in fresh environment')
except Exception as e:
    print('✗ Import failed in fresh environment:', e)
    exit(1)
"
if [ $? -ne 0 ]; then
    echo "❌ Import test failed in fresh environment"
    exit 1
fi
echo

# Test 5: App creation in fresh environment
echo "Test 5: Testing app creation in fresh environment..."
PYTHONPATH="$(pwd)/src" python3 -c "
try:
    from pynomaly.presentation.web.app import create_web_app
    app = create_web_app()
    print('✓ App creation successful in fresh environment')
    print('✓ Routes count:', len(app.routes))
except Exception as e:
    print('✗ App creation failed in fresh environment:', e)
    exit(1)
"
if [ $? -ne 0 ]; then
    echo "❌ App creation test failed in fresh environment"
    exit 1
fi
echo

# Test 6: Server startup in fresh environment
echo "Test 6: Testing server startup in fresh environment..."
PYTHONPATH="$(pwd)/src" python3 scripts/run_web_app.py &
SERVER_PID=$!
echo "✓ Server started in fresh environment with PID: $SERVER_PID"

# Wait for server to start
sleep 8

# Test API endpoint
echo "Testing API endpoint in fresh environment..."
API_RESPONSE=$(curl -s http://localhost:8000/ 2>/dev/null)
if echo "$API_RESPONSE" | grep -q "Pynomaly API"; then
    echo "✓ API endpoint working in fresh environment"
else
    echo "✗ API endpoint failed in fresh environment"
    echo "Response: $API_RESPONSE"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Test Web UI endpoint
echo "Testing Web UI endpoint in fresh environment..."
WEB_RESPONSE=$(curl -s http://localhost:8000/web/ 2>/dev/null)
if echo "$WEB_RESPONSE" | grep -q "Dashboard - Pynomaly"; then
    echo "✓ Web UI endpoint working in fresh environment"
else
    echo "✗ Web UI endpoint failed in fresh environment"
    echo "Response: $WEB_RESPONSE" | head -3
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Stop server
kill $SERVER_PID 2>/dev/null
sleep 2
echo "✓ Server stopped"

# Deactivate virtual environment if used
if [ "$SKIP_VENV" != "true" ]; then
    deactivate 2>/dev/null || true
    echo "✓ Virtual environment deactivated"
fi

# Return to project root
cd ../..

# Clean up test environment
echo
echo "Cleaning up test environment..."
rm -rf "$TEST_ENV_DIR"
echo "✓ Test environment cleaned up"

echo
echo "🎉 All fresh environment tests passed! Pynomaly web app works correctly in fresh bash environment."
echo "✓ Virtual environment creation (or graceful fallback)"
echo "✓ Dependency installation"
echo "✓ Source code setup"
echo "✓ Python imports working"
echo "✓ App creation working"
echo "✓ Server startup working"
echo "✓ API endpoints working"
echo "✓ Web UI working"
echo "✓ Environment cleanup"
