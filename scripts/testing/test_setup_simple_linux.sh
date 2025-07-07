#!/bin/bash
# Test script for setup_simple.py on Linux/Bash

echo "=========================================="
echo "Testing setup_simple.py on Linux"
echo "=========================================="

# Create a temporary test directory
testDir="test_setup_simple_$(date +%s)"
echo "Creating test directory: $testDir"

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up test directory: $testDir"
    cd .. 2>/dev/null
    rm -rf "$testDir" 2>/dev/null
}

# Set trap for cleanup
trap cleanup EXIT

# Create test directory and copy essential files
mkdir -p "$testDir"
cd "$testDir"

# Copy essential files from parent directory
cp "../pyproject.toml" . 2>/dev/null || echo "Warning: pyproject.toml not found"
cp "../requirements.txt" . 2>/dev/null || echo "Warning: requirements.txt not found"
cp "../setup.py" . 2>/dev/null || echo "Warning: setup.py not found"
cp "../scripts/setup_simple.py" . 2>/dev/null || {
    echo "Error: setup_simple.py not found"
    exit 1
}

# Create minimal src structure
mkdir -p "src/pynomaly/domain/entities"
echo "__version__ = '0.1.0'" > "src/pynomaly/__init__.py"
touch "src/pynomaly/domain/__init__.py"

# Create minimal domain entities
cat > "src/pynomaly/domain/entities/__init__.py" << 'EOF'
class Dataset:
    pass

class Anomaly:
    pass
EOF

echo "Test directory structure created"
echo "Files in test directory:"
find . -type f | sort

echo ""
echo "Running setup_simple.py..."
python3 setup_simple.py --clean

echo ""
echo "Test completed in directory: $testDir"
echo "Check the output above for results"

# Cleanup will happen automatically via trap