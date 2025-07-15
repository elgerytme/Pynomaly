#!/bin/bash

# Run Pynomaly Test Suite in Isolation
# Comprehensive testing with coverage and reporting

set -euo pipefail

echo "🧪 Running Pynomaly Test Suite in Isolation..."

# Configuration
WORKSPACE="/workspace"
TEST_RESULTS_DIR="$WORKSPACE/test-results"
COVERAGE_DIR="$WORKSPACE/coverage"

# Create directories
mkdir -p "$TEST_RESULTS_DIR" "$COVERAGE_DIR"

# Set environment variables for testing
export PYTHONPATH="$WORKSPACE/src"
export PYNOMALY_ENV="test"
export LOG_LEVEL="WARNING"  # Reduce noise during testing
export TESTING="true"

# Test database configuration
export DATABASE_URL="${DATABASE_URL:-postgresql://pynomaly:isolated@postgres-isolated:5432/pynomaly_test}"
export REDIS_URL="${REDIS_URL:-redis://redis-isolated:6379/1}"

echo "📋 Test Configuration:"
echo "  - Environment: $PYNOMALY_ENV"
echo "  - Python Path: $PYTHONPATH"
echo "  - Test Database: $DATABASE_URL"
echo "  - Redis: $REDIS_URL"
echo "  - Results: $TEST_RESULTS_DIR"

# Change to workspace directory
cd "$WORKSPACE"

# Install test dependencies
echo "📦 Installing test dependencies..."
pip install --quiet --disable-pip-version-check \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-html \
    pytest-json-report \
    pytest-mock \
    coverage[toml]

# Wait for test database
echo "⏳ Waiting for test database..."
for i in {1..30}; do
    if python -c "import psycopg2; psycopg2.connect('$DATABASE_URL')" 2>/dev/null; then
        echo "✅ Test database ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Test database not available"
        exit 1
    fi
    sleep 1
done

# Setup test database
echo "🔧 Setting up test database..."
python -c "
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Connect to default database
conn = psycopg2.connect('postgresql://pynomaly:isolated@postgres-isolated:5432/postgres')
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

# Drop and recreate test database
cur.execute('DROP DATABASE IF EXISTS pynomaly_test')
cur.execute('CREATE DATABASE pynomaly_test')
print('✅ Test database created')

cur.close()
conn.close()
" 2>/dev/null || echo "⚠️  Database setup failed, continuing..."

# Run database migrations for test database
echo "🔄 Running test database migrations..."
if [ -f "alembic.ini" ]; then
    python -m alembic upgrade head 2>/dev/null || echo "⚠️  Migration failed, continuing..."
fi

# Parse command line arguments
TEST_PATTERN=""
COVERAGE_ENABLED="true"
PARALLEL_ENABLED="true"
VERBOSE="false"
FAIL_FAST="false"
HTML_REPORT="true"
JSON_REPORT="true"

while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--keyword)
            TEST_PATTERN="$2"
            shift 2
            ;;
        --no-coverage)
            COVERAGE_ENABLED="false"
            shift
            ;;
        --no-parallel)
            PARALLEL_ENABLED="false"
            shift
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -x|--fail-fast)
            FAIL_FAST="true"
            shift
            ;;
        --no-html)
            HTML_REPORT="false"
            shift
            ;;
        --no-json)
            JSON_REPORT="false"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -k, --keyword PATTERN    Run tests matching keyword pattern"
            echo "  --no-coverage           Disable coverage reporting"
            echo "  --no-parallel           Disable parallel test execution"
            echo "  -v, --verbose           Verbose output"
            echo "  -x, --fail-fast         Stop on first failure"
            echo "  --no-html               Disable HTML report generation"
            echo "  --no-json               Disable JSON report generation"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="python -m pytest"

# Add test directory
if [ -d "tests" ]; then
    PYTEST_CMD="$PYTEST_CMD tests/"
elif [ -d "test" ]; then
    PYTEST_CMD="$PYTEST_CMD test/"
else
    echo "⚠️  No tests directory found, running discovery"
    PYTEST_CMD="$PYTEST_CMD"
fi

# Add options based on configuration
if [ "$VERBOSE" = "true" ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$FAIL_FAST" = "true" ]; then
    PYTEST_CMD="$PYTEST_CMD -x"
fi

if [ "$PARALLEL_ENABLED" = "true" ]; then
    # Use number of CPU cores for parallel execution
    CORES=$(nproc 2>/dev/null || echo "2")
    PYTEST_CMD="$PYTEST_CMD -n $CORES"
fi

if [ -n "$TEST_PATTERN" ]; then
    PYTEST_CMD="$PYTEST_CMD -k '$TEST_PATTERN'"
fi

# Add coverage if enabled
if [ "$COVERAGE_ENABLED" = "true" ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src/pynomaly --cov-report=html:$COVERAGE_DIR/html --cov-report=xml:$COVERAGE_DIR/coverage.xml --cov-report=term-missing"
fi

# Add HTML report if enabled
if [ "$HTML_REPORT" = "true" ]; then
    PYTEST_CMD="$PYTEST_CMD --html=$TEST_RESULTS_DIR/report.html --self-contained-html"
fi

# Add JSON report if enabled
if [ "$JSON_REPORT" = "true" ]; then
    PYTEST_CMD="$PYTEST_CMD --json-report --json-report-file=$TEST_RESULTS_DIR/report.json"
fi

# Add JUnit XML for CI integration
PYTEST_CMD="$PYTEST_CMD --junitxml=$TEST_RESULTS_DIR/junit.xml"

echo "🏃 Running tests..."
echo "Command: $PYTEST_CMD"
echo "========================================"

# Run the tests
if eval "$PYTEST_CMD"; then
    TEST_EXIT_CODE=0
    echo "========================================"
    echo "✅ All tests passed!"
else
    TEST_EXIT_CODE=$?
    echo "========================================"
    echo "❌ Some tests failed (exit code: $TEST_EXIT_CODE)"
fi

# Generate coverage report summary
if [ "$COVERAGE_ENABLED" = "true" ] && command -v coverage >/dev/null; then
    echo ""
    echo "📊 Coverage Summary:"
    echo "==================="
    coverage report --show-missing 2>/dev/null || echo "Coverage report generation failed"
fi

# Show where reports are located
echo ""
echo "📋 Test Reports Generated:"
echo "========================="
if [ "$HTML_REPORT" = "true" ] && [ -f "$TEST_RESULTS_DIR/report.html" ]; then
    echo "  📄 HTML Report: $TEST_RESULTS_DIR/report.html"
fi
if [ "$JSON_REPORT" = "true" ] && [ -f "$TEST_RESULTS_DIR/report.json" ]; then
    echo "  📊 JSON Report: $TEST_RESULTS_DIR/report.json"
fi
if [ -f "$TEST_RESULTS_DIR/junit.xml" ]; then
    echo "  🔬 JUnit XML: $TEST_RESULTS_DIR/junit.xml"
fi
if [ "$COVERAGE_ENABLED" = "true" ] && [ -d "$COVERAGE_DIR/html" ]; then
    echo "  📈 Coverage HTML: $COVERAGE_DIR/html/index.html"
fi
if [ "$COVERAGE_ENABLED" = "true" ] && [ -f "$COVERAGE_DIR/coverage.xml" ]; then
    echo "  📋 Coverage XML: $COVERAGE_DIR/coverage.xml"
fi

# Show quick stats
if [ -f "$TEST_RESULTS_DIR/report.json" ]; then
    echo ""
    echo "📈 Quick Stats:"
    echo "=============="
    python -c "
import json
try:
    with open('$TEST_RESULTS_DIR/report.json') as f:
        data = json.load(f)
    summary = data.get('summary', {})
    print(f\"  Tests: {summary.get('total', 0)}\")
    print(f\"  Passed: {summary.get('passed', 0)}\")
    print(f\"  Failed: {summary.get('failed', 0)}\")
    print(f\"  Skipped: {summary.get('skipped', 0)}\")
    print(f\"  Duration: {data.get('duration', 0):.2f}s\")
except Exception as e:
    print(f\"  Could not parse test results: {e}\")
"
fi

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "🎉 Testing completed successfully!"
else
    echo "💥 Testing completed with failures"
    echo "Check the reports above for details"
fi

exit $TEST_EXIT_CODE
