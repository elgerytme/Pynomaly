#!/bin/bash

# Quick test script for individual environment testing
ENV_NAME=${1:-"linux_bash"}
PROJECT_ROOT="/mnt/c/Users/andre/Pynomaly"
ENV_PATH="$PROJECT_ROOT/environments/test_environments/$ENV_NAME"

echo "Testing environment: $ENV_NAME"

# Create and activate virtual environment
cd "$ENV_PATH"
python3 -m venv test_env
source test_env/bin/activate

# Quick pip upgrade
pip install --upgrade pip --quiet

echo "Installing Pynomaly..."
cd "$PROJECT_ROOT"
pip install -e . --quiet

echo "Testing CLI..."
pynomaly --help > /dev/null && echo "✅ CLI works"

echo "Testing basic imports..."
python -c "
import sys
sys.path.insert(0, 'src')
from pynomaly.domain.entities import TrainingRequest
print('✅ Basic imports work')
"

echo "Testing with minimal dependencies..."
pip install -e ".[minimal]" --quiet
python -c "
import numpy as np
import sklearn
print('✅ Minimal dependencies work')
"

echo "Environment $ENV_NAME test completed!"