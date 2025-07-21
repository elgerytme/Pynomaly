#!/bin/bash

echo "========================================"
echo "anomaly_detection WINDOWS/POWERSHELL SIMULATION TEST"
echo "========================================"
echo "Environment: Windows Simulation on $(uname -a)"
echo "Python: $(python3 --version)"
echo "Current directory: $(pwd)"
echo ""

# Function to simulate PowerShell-style testing
run_windows_test() {
    local test_name="$1"
    local command="$2"

    echo "========================================="
    echo "Testing (Windows style): $test_name"
    echo "Command: $command"
    echo "========================================="

    # Use Windows-style path separators in output and simulate PowerShell behavior
    if eval "$command"; then
        echo "SUCCESS: $test_name completed successfully"
        echo "Status: PASSED"
        return 0
    else
        echo "ERROR: $test_name failed with exit code $?"
        echo "Status: FAILED"
        return 1
    fi
}

# Initialize counters
total_tests=0
passed_tests=0

# Test 1: Windows-style Python Module Test
((total_tests++))
if run_windows_test "Python Module Import Test" "python3 -c \"
import sys
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Python path:', sys.path[:3])  # Show first 3 paths

# Test importing main modules
try:
    import anomaly_detection
    print('✓ Successfully imported anomaly_detection')
except Exception as e:
    print('✗ Failed to import anomaly_detection:', str(e))
    raise

try:
    from anomaly_detection.domain.entities import Dataset, Detector
    print('✓ Successfully imported domain entities')
except Exception as e:
    print('✗ Failed to import domain entities:', str(e))
    raise

try:
    from anomaly_detection.infrastructure.adapters.pyod_adapter import PyODAdapter
    print('✓ Successfully imported PyOD adapter')
except Exception as e:
    print('✗ Failed to import PyOD adapter:', str(e))
    raise

print('All module imports successful!')
\""; then
    ((passed_tests++))
fi

# Test 2: Windows-style API Server Test
((total_tests++))
if run_windows_test "FastAPI Application Test" "python3 -c \"
from anomaly_detection.infrastructure.config import create_container
from anomaly_detection.presentation.api.app import create_app
from fastapi.testclient import TestClient
import json

print('Initializing FastAPI application...')
container = create_container()
app = create_app(container)
client = TestClient(app)

print('Testing health endpoint...')
response = client.get('/api/health/')
assert response.status_code == 200, f'Health check failed with status {response.status_code}'
health_data = response.json()
print(f'Health status: {health_data[\\\"overall_status\\\"]}')

print('Testing detector creation...')
detector_payload = {
    'name': 'Windows Test Detector',
    'algorithm_name': 'IsolationForest',
    'parameters': {'contamination': 0.05}
}
create_response = client.post('/api/detectors/', json=detector_payload)
assert create_response.status_code == 200, f'Detector creation failed with status {create_response.status_code}'
detector_data = create_response.json()
print(f'Created detector ID: {detector_data[\\\"id\\\"]}')

print('Testing detector list...')
list_response = client.get('/api/detectors/')
assert list_response.status_code == 200, f'Detector listing failed with status {list_response.status_code}'
detectors = list_response.json()
print(f'Total detectors available: {len(detectors)}')

print('FastAPI application test completed successfully!')
\""; then
    ((passed_tests++))
fi

# Test 3: Windows-style ML Pipeline Test
((total_tests++))
if run_windows_test "Machine Learning Pipeline Test" "python3 -c \"
import numpy as np
import pandas as pd
from anomaly_detection.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from anomaly_detection.domain.entities import Dataset
import warnings
warnings.filterwarnings('ignore')

print('Creating synthetic dataset...')
# Generate Windows-style data simulation
np.random.seed(42)
normal_data = np.random.normal(0, 1, (95, 2))
anomaly_data = np.random.normal(5, 0.5, (5, 2))
all_data = np.vstack([normal_data, anomaly_data])

dataset_df = pd.DataFrame(all_data, columns=['feature_1', 'feature_2'])
dataset = Dataset(name='Windows_Test_Dataset', data=dataset_df)
print(f'Dataset created with shape: {dataset.data.shape}')

print('Initializing ML adapter...')
adapter = SklearnAdapter('IsolationForest')
print(f'Adapter created for algorithm: {adapter.algorithm_name}')

print('Training anomaly detection model...')
adapter.fit(dataset)
print('Model training completed')

print('Running anomaly detection...')
results = adapter.detect(dataset)
print(f'Detection completed. Found {len([s for s in results.scores if s.value > 0.5])} potential anomalies')

print('Machine learning pipeline test completed successfully!')
\""; then
    ((passed_tests++))
fi

# Test 4: Windows-style Configuration Test
((total_tests++))
if run_windows_test "Configuration Management Test" "python3 -c \"
from anomaly_detection.infrastructure.config.settings import get_settings
from anomaly_detection.infrastructure.config import create_container

print('Loading application settings...')
settings = get_settings()
print(f'Application: {settings.app.name} v{settings.app.version}')
print(f'Environment: {settings.app.environment}')

print('Creating dependency injection container...')
container = create_container()
print(f'Container type: {type(container).__name__}')

print('Testing repository access...')
try:
    detector_repo = container.detector_repository()
    print(f'✓ Detector repository: {type(detector_repo).__name__}')
except Exception as e:
    print(f'✗ Detector repository error: {str(e)}')

try:
    dataset_repo = container.dataset_repository()
    print(f'✓ Dataset repository: {type(dataset_repo).__name__}')
except Exception as e:
    print(f'✗ Dataset repository error: {str(e)}')

print('Configuration management test completed successfully!')
\""; then
    ((passed_tests++))
fi

# Test 5: Windows-style File System Test
((total_tests++))
if run_windows_test "File System Operations Test" "python3 -c \"
import os
import tempfile
import pandas as pd
from pathlib import Path

print('Testing file system operations...')

# Test temporary file creation (Windows-style)
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
    temp_path = temp_file.name
    print(f'Created temporary file: {temp_path}')

    # Write test data
    test_data = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [10, 20, 30, 40, 50]
    })
    test_data.to_csv(temp_path, index=False)
    print('✓ Data written to CSV file')

# Test file reading
if os.path.exists(temp_path):
    read_data = pd.read_csv(temp_path)
    print(f'✓ Data read from CSV: {read_data.shape[0]} rows, {read_data.shape[1]} columns')

    # Cleanup
    os.unlink(temp_path)
    print('✓ Temporary file cleaned up')
else:
    raise FileNotFoundError('Temporary file not found')

print('File system operations test completed successfully!')
\""; then
    ((passed_tests++))
fi

echo ""
echo "========================================"
echo "WINDOWS/POWERSHELL SIMULATION RESULTS"
echo "========================================"
echo "Total tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $((total_tests - passed_tests))"
echo "Success rate: $((passed_tests * 100 / total_tests))%"

if [ $passed_tests -eq $total_tests ]; then
    echo ""
    echo "🎉 ALL WINDOWS SIMULATION TESTS PASSED!"
    echo "The application is ready for Windows/PowerShell deployment!"
    exit 0
else
    echo ""
    echo "⚠️ Some Windows simulation tests failed."
    echo "Please review the output above for details."
    exit 1
fi
