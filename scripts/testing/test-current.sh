#!/usr/bin/env bash
# Test script for current bash environment

echo "=== Testing Pynomaly Web App in Current Bash Environment ==="
echo "Date: $(date)"
echo "Current directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Shell: $SHELL"
echo

# Change to project root
cd "$(dirname "$0")/../.."

echo "Project root: $(pwd)"
echo

# Test 1: Import test
echo "Test 1: Testing Python imports..."
PYTHONPATH="$(pwd)/src" python3 -c "
try:
    from pynomaly.presentation.web.app import create_web_app
    print('✓ Import successful')
except Exception as e:
    print('✗ Import failed:', e)
    exit(1)
"
if [ $? -ne 0 ]; then
    echo "❌ Import test failed"
    exit 1
fi
echo

# Test 2: App creation test
echo "Test 2: Testing app creation..."
PYTHONPATH="$(pwd)/src" python3 -c "
try:
    from pynomaly.presentation.web.app import create_web_app
    app = create_web_app()
    print('✓ App creation successful')
    print('✓ Routes count:', len(app.routes))
except Exception as e:
    print('✗ App creation failed:', e)
    exit(1)
"
if [ $? -ne 0 ]; then
    echo "❌ App creation test failed"
    exit 1
fi
echo

# Test 3: Server startup test
echo "Test 3: Testing server startup..."
PYTHONPATH="$(pwd)/src" python3 scripts/run/run_web_app.py &
SERVER_PID=$!
echo "✓ Server started with PID: $SERVER_PID"

# Wait for server to start
sleep 5

# Test API endpoint
echo "Testing API endpoint..."
API_RESPONSE=$(curl -s http://localhost:8000/ 2>/dev/null)
if echo "$API_RESPONSE" | grep -q "Pynomaly API"; then
    echo "✓ API endpoint working"
else
    echo "✗ API endpoint failed"
    echo "Response: $API_RESPONSE"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Test Web UI endpoint
echo "Testing Web UI endpoint..."
WEB_RESPONSE=$(curl -s http://localhost:8000/web/ 2>/dev/null)
if echo "$WEB_RESPONSE" | grep -q "Dashboard - Pynomaly"; then
    echo "✓ Web UI endpoint working"
else
    echo "✗ Web UI endpoint failed"
    echo "Response: $WEB_RESPONSE" | head -3
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Stop server
kill $SERVER_PID 2>/dev/null
sleep 2
echo "✓ Server stopped"

echo
echo "🎉 All tests passed! Pynomaly web app works correctly in current bash environment."
echo "✓ Python imports working"
echo "✓ App creation working"
echo "✓ Server startup working"
echo "✓ API endpoints working"
echo "✓ Web UI working"
