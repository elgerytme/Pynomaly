#!/bin/bash
# Bash script to start Pynomaly API
# Usage: ./start_api_bash.sh [PORT] [HOST]

set -e

# Configuration - Dynamic path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_PATH="$PROJECT_ROOT/src"
DEFAULT_PORT=8000
DEFAULT_HOST="0.0.0.0"

# Get arguments
PORT=${1:-$DEFAULT_PORT}
HOST=${2:-$DEFAULT_HOST}

echo "=== Starting Pynomaly API (Bash) ==="
echo "Project root: $PROJECT_ROOT"
echo "Source path: $SRC_PATH"
echo "Host: $HOST"
echo "Port: $PORT"

# Set environment
export PYTHONPATH="$SRC_PATH"

# Check if dependencies are available
echo "Checking dependencies..."
python3 -c "
import sys
required = ['fastapi', 'uvicorn', 'pydantic', 'structlog']
missing = []
for module in required:
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    print(f'Missing dependencies: {missing}')
    print('Run: pip install --break-system-packages fastapi uvicorn pydantic structlog dependency-injector')
    sys.exit(1)
else:
    print('✓ Dependencies available')
"

# Test import
echo "Testing Pynomaly import..."
python3 -c "
try:
    from pynomaly.presentation.api import app
    print('✓ Pynomaly API imported successfully')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    exit(1)
"

# Start server
echo "Starting server at http://$HOST:$PORT"
echo "API Documentation: http://$HOST:$PORT/api/docs"
echo "Health Check: http://$HOST:$PORT/api/health/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn pynomaly.presentation.api:app --host "$HOST" --port "$PORT" --reload
