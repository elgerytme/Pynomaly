#!/bin/bash

# Start Pynomaly Development Server in Isolation
# Configures and starts the FastAPI development server

set -euo pipefail

echo "🚀 Starting Pynomaly Development Server in Isolation..."

# Check if we're in the right directory
if [ ! -f "/workspace/src/pynomaly/main.py" ] && [ ! -f "/workspace/pyproject.toml" ]; then
    echo "❌ Error: Not in a Pynomaly workspace directory"
    echo "Expected files not found: src/pynomaly/main.py or pyproject.toml"
    exit 1
fi

# Set environment variables for development
export PYTHONPATH="/workspace/src"
export PYNOMALY_ENV="isolated"
export LOG_LEVEL="DEBUG"
export DEVELOPMENT_MODE="true"

# Database connection (use isolated database)
export DATABASE_URL="${DATABASE_URL:-postgresql://pynomaly:isolated@postgres-isolated:5432/pynomaly_isolated}"
export REDIS_URL="${REDIS_URL:-redis://redis-isolated:6379/0}"

# API configuration
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
export API_RELOAD="${API_RELOAD:-true}"

echo "📋 Development Configuration:"
echo "  - Environment: $PYNOMALY_ENV"
echo "  - Python Path: $PYTHONPATH"
echo "  - Database: $DATABASE_URL"
echo "  - Redis: $REDIS_URL"
echo "  - API: http://$API_HOST:$API_PORT"
echo "  - Hot Reload: $API_RELOAD"

# Wait for database to be ready
echo "⏳ Waiting for database connection..."
for i in {1..30}; do
    if python -c "import psycopg2; psycopg2.connect('$DATABASE_URL')" 2>/dev/null; then
        echo "✅ Database connection established"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Failed to connect to database after 30 attempts"
        echo "Database URL: $DATABASE_URL"
        exit 1
    fi
    sleep 1
done

# Wait for Redis to be ready
echo "⏳ Waiting for Redis connection..."
for i in {1..10}; do
    if python -c "import redis; r=redis.from_url('$REDIS_URL'); r.ping()" 2>/dev/null; then
        echo "✅ Redis connection established"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "⚠️  Redis not available, continuing without cache"
    fi
    sleep 1
done

# Run database migrations if needed
echo "🔄 Checking for database migrations..."
if [ -f "/workspace/alembic.ini" ]; then
    cd /workspace
    python -m alembic upgrade head 2>/dev/null || echo "⚠️  Migration check failed, continuing..."
fi

# Install any missing dependencies
echo "📦 Checking Python dependencies..."
cd /workspace
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet --disable-pip-version-check
fi
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt --quiet --disable-pip-version-check
fi

# Start the development server
echo "🌟 Starting FastAPI development server..."
echo "📡 Server will be available at: http://localhost:$API_PORT"
echo "📚 API Documentation: http://localhost:$API_PORT/docs"
echo "🔧 Alternative docs: http://localhost:$API_PORT/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"

# Change to source directory
cd /workspace

# Start the server with uvicorn
if [ -f "src/pynomaly/main.py" ]; then
    # Standard project structure
    exec uvicorn pynomaly.main:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --reload \
        --reload-dir /workspace/src \
        --app-dir /workspace/src
elif [ -f "main.py" ]; then
    # Alternative structure
    exec uvicorn main:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --reload
else
    echo "❌ Could not find main.py file"
    echo "Looked in:"
    echo "  - /workspace/src/pynomaly/main.py"
    echo "  - /workspace/main.py"
    exit 1
fi
