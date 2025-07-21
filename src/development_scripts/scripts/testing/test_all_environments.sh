#!/bin/bash
# Comprehensive test script for anomaly detection API in different environments
# Tests: Current environment, Fresh environment, Different shells

set -e

PROJECT_ROOT="/mnt/c/Users/andre/Pynomaly"
SRC_PATH="$PROJECT_ROOT/src"

echo "=== anomaly detection API Multi-Environment Test Suite ==="
echo "Project: $PROJECT_ROOT"
echo "Testing Date: $(date)"
echo ""

# Function to test API endpoint
test_endpoint() {
    local url=$1
    local description=$2
    local timeout=${3:-10}

    echo -n "  Testing $description... "
    if timeout $timeout bash -c "curl -s '$url' > /dev/null"; then
        echo "✓ SUCCESS"
        return 0
    else
        echo "✗ FAILED"
        return 1
    fi
}

# Function to start server and test
test_server() {
    local port=$1
    local env_name=$2
    local pythonpath=$3

    echo "--- Testing $env_name Environment (Port $port) ---"

    # Start server in background
    echo "Starting server..."
    if [ -n "$pythonpath" ]; then
        PYTHONPATH="$pythonpath" uvicorn pynomaly.presentation.api:app --host 127.0.0.1 --port $port > /tmp/test_$port.log 2>&1 &
    else
        uvicorn pynomaly.presentation.api:app --host 127.0.0.1 --port $port > /tmp/test_$port.log 2>&1 &
    fi

    local server_pid=$!
    echo "Server PID: $server_pid"

    # Wait for server to start
    echo "Waiting for server to start..."
    sleep 5

    # Test endpoints
    local success=0
    test_endpoint "http://127.0.0.1:$port/" "root endpoint" && ((success++))
    test_endpoint "http://127.0.0.1:$port/api/health/" "health endpoint" && ((success++))
    test_endpoint "http://127.0.0.1:$port/api/docs" "docs endpoint" && ((success++))
    test_endpoint "http://127.0.0.1:$port/api/openapi.json" "OpenAPI schema" && ((success++))

    # Get response data
    if curl -s "http://127.0.0.1:$port/" > /dev/null 2>&1; then
        local root_response=$(curl -s "http://127.0.0.1:$port/")
        local version=$(echo "$root_response" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
        echo "  API Version: $version"
    fi

    if curl -s "http://127.0.0.1:$port/api/health/" > /dev/null 2>&1; then
        local health_response=$(curl -s "http://127.0.0.1:$port/api/health/")
        local status=$(echo "$health_response" | grep -o '"overall_status":"[^"]*"' | cut -d'"' -f4)
        local uptime=$(echo "$health_response" | grep -o '"uptime_seconds":[0-9.]*' | cut -d':' -f2)
        echo "  Health Status: $status"
        echo "  Uptime: ${uptime}s"
    fi

    # Stop server
    echo "Stopping server..."
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true

    echo "  Test Results: $success/4 endpoints successful"

    if [ $success -eq 4 ]; then
        echo "  ✅ $env_name Environment: PASSED"
        return 0
    else
        echo "  ❌ $env_name Environment: FAILED"
        echo "  Check logs: /tmp/test_$port.log"
        return 1
    fi
}

# Test 1: Current Environment
echo ""
echo "🧪 Test 1: Current Environment"
export PYTHONPATH="$SRC_PATH"
test_server 8010 "Current" "$SRC_PATH"

# Test 2: Fresh Environment Simulation
echo ""
echo "🧪 Test 2: Fresh Environment Simulation"
env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin bash -c "
export PYTHONPATH='$SRC_PATH'
$(declare -f test_endpoint)
$(declare -f test_server)
test_server 8011 'Fresh' '$SRC_PATH'
"

# Test 3: Minimal Environment
echo ""
echo "🧪 Test 3: Minimal Environment (System Python)"
env -i HOME=$HOME PATH=/usr/bin:/bin bash -c "
export PYTHONPATH='$SRC_PATH'
$(declare -f test_endpoint)
$(declare -f test_server)
test_server 8012 'Minimal' '$SRC_PATH'
"

# Test 4: Different Port Test
echo ""
echo "🧪 Test 4: Multiple Port Test"
export PYTHONPATH="$SRC_PATH"

# Start servers on different ports
echo "Starting multiple servers..."
uvicorn pynomaly.presentation.api:app --host 127.0.0.1 --port 8013 > /tmp/test_8013.log 2>&1 &
PID1=$!
uvicorn pynomaly.presentation.api:app --host 127.0.0.1 --port 8014 > /tmp/test_8014.log 2>&1 &
PID2=$!

sleep 5

# Test both
success_count=0
test_endpoint "http://127.0.0.1:8013/" "Port 8013" && ((success_count++))
test_endpoint "http://127.0.0.1:8014/" "Port 8014" && ((success_count++))

# Cleanup
kill $PID1 $PID2 2>/dev/null || true
wait $PID1 $PID2 2>/dev/null || true

if [ $success_count -eq 2 ]; then
    echo "  ✅ Multiple Port Test: PASSED"
else
    echo "  ❌ Multiple Port Test: FAILED"
fi

# Summary
echo ""
echo "=== Test Suite Summary ==="
echo "Environment tests completed at $(date)"
echo ""
echo "📋 Usage Instructions:"
echo ""
echo "🔧 Bash Environment:"
echo "  export PYTHONPATH=$SRC_PATH"
echo "  uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000"
echo ""
echo "🎛️  PowerShell Environment:"
echo "  \$env:PYTHONPATH = 'C:\\Users\\andre\\Pynomaly\\src'"
echo "  uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000"
echo ""
echo "🚀 Quick Start Scripts:"
echo "  Bash: ./scripts/start_api_bash.sh"
echo "  PowerShell: pwsh -File scripts/test_api_powershell.ps1"
echo "  Fresh Setup: ./scripts/setup_fresh_environment.sh"
echo ""
echo "📊 API Endpoints:"
echo "  Root: http://localhost:8000/"
echo "  Health: http://localhost:8000/api/health/"
echo "  Docs: http://localhost:8000/api/docs"
echo "  OpenAPI: http://localhost:8000/api/openapi.json"
