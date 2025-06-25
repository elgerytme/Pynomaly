#!/bin/bash
# Test script for fresh bash environment (modified for externally managed Python)

echo "=== Testing Pynomaly Web App in Fresh Bash Environment (Modified) ==="
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
TEST_ENV_DIR="test_environments/fresh_bash_test_modified"
echo "Creating fresh test environment in: $TEST_ENV_DIR"

# Clean up any existing test environment
if [ -d "$TEST_ENV_DIR" ]; then
    rm -rf "$TEST_ENV_DIR"
fi

mkdir -p "$TEST_ENV_DIR"
cd "$TEST_ENV_DIR"

echo "✓ Test environment created"
echo

# Test 1: Copy source code to fresh environment
echo "Test 1: Setting up source code in fresh environment..."
cp -r ../../src .
cp -r ../../scripts .
echo "✓ Source code copied"
echo

# Test 2: Test imports in fresh environment (assuming dependencies are available)
echo "Test 2: Testing imports in fresh environment..."
PYTHONPATH="$(pwd)/src" python3 -c "
try:
    # Test basic Python functionality first
    import sys, os
    print('✓ Basic Python imports working')
    
    # Test if required packages are available
    required_packages = ['fastapi', 'uvicorn', 'pydantic', 'dependency_injector', 'pandas', 'numpy', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f'✓ {package} available')
        except ImportError:
            missing_packages.append(package)
            print(f'✗ {package} not available')
    
    if missing_packages:
        print(f'⚠️  Missing packages: {missing_packages}')
        print('⚠️  Cannot test full functionality, but testing core imports...')
    
    # Test pynomaly imports
    from pynomaly.presentation.web.app import create_web_app
    print('✓ Pynomaly import successful in fresh environment')
    
except Exception as e:
    print('✗ Import failed in fresh environment:', e)
    import traceback
    traceback.print_exc()
    exit(1)
"
if [ $? -ne 0 ]; then
    echo "❌ Import test failed in fresh environment"
    cd ../..
    rm -rf "$TEST_ENV_DIR"
    exit 1
fi
echo

# Test 3: App creation in fresh environment
echo "Test 3: Testing app creation in fresh environment..."
PYTHONPATH="$(pwd)/src" python3 -c "
try:
    from pynomaly.presentation.web.app import create_web_app
    app = create_web_app()
    print('✓ App creation successful in fresh environment')
    print('✓ Routes count:', len(app.routes))
    
    # Test that both API and web routes are present
    api_routes = [r for r in app.routes if str(r.path).startswith('/api')]
    web_routes = [r for r in app.routes if str(r.path).startswith('/web')]
    
    print(f'✓ API routes: {len(api_routes)}')
    print(f'✓ Web routes: {len(web_routes)}')
    
    if len(api_routes) > 0 and len(web_routes) > 0:
        print('✓ Both API and Web UI routes present')
    else:
        print('✗ Missing API or Web UI routes')
        exit(1)
        
except Exception as e:
    print('✗ App creation failed in fresh environment:', e)
    import traceback
    traceback.print_exc()
    exit(1)
"
if [ $? -ne 0 ]; then
    echo "❌ App creation test failed in fresh environment"
    cd ../..
    rm -rf "$TEST_ENV_DIR"
    exit 1
fi
echo

# Test 4: Check that required files are present
echo "Test 4: Verifying file structure in fresh environment..."
REQUIRED_FILES=(
    "src/pynomaly/__init__.py"
    "src/pynomaly/presentation/api/app.py"
    "src/pynomaly/presentation/web/app.py"
    "src/pynomaly/infrastructure/config/container.py"
    "scripts/run_web_app.py"
)

ALL_FILES_PRESENT=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file present"
    else
        echo "✗ $file missing"
        ALL_FILES_PRESENT=false
    fi
done

if [ "$ALL_FILES_PRESENT" = true ]; then
    echo "✓ All required files present"
else
    echo "❌ Some required files missing"
    cd ../..
    rm -rf "$TEST_ENV_DIR"
    exit 1
fi
echo

# Test 5: Server startup test (if dependencies are available)
echo "Test 5: Testing server startup in fresh environment..."
PYTHONPATH="$(pwd)/src" timeout 15 python3 scripts/run_web_app.py &
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
    cd ../..
    rm -rf "$TEST_ENV_DIR"
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
    cd ../..
    rm -rf "$TEST_ENV_DIR"
    exit 1
fi

# Stop server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 2
echo "✓ Server stopped"

# Return to project root
cd ../..

# Clean up test environment
echo
echo "Cleaning up test environment..."
rm -rf "$TEST_ENV_DIR"
echo "✓ Test environment cleaned up"

echo
echo "🎉 All fresh environment tests passed! Pynomaly web app works correctly in fresh bash environment."
echo "✓ Source code setup in fresh location"
echo "✓ File structure verification"
echo "✓ Python imports working"
echo "✓ App creation working"
echo "✓ Server startup working"
echo "✓ API endpoints working"
echo "✓ Web UI working"
echo "✓ Environment cleanup"