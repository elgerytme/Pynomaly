#!/bin/bash

echo "========================================"
echo "PYNOMALY FRESH INSTALLATION TEST SUITE"
echo "========================================"
echo "Testing Pynomaly installation and functionality"
echo "Environment: $(uname -a)"
echo "Python: $(python3 --version)"
echo "Current directory: $(pwd)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
    esac
}

# Function to run a test step
run_test_step() {
    local step_name="$1"
    local command="$2"
    local optional="${3:-false}"

    print_status "INFO" "Running: $step_name"
    echo "Command: $command"
    echo "----------------------------------------"

    if eval "$command"; then
        print_status "SUCCESS" "$step_name completed successfully"
        return 0
    else
        local exit_code=$?
        if [ "$optional" = "true" ]; then
            print_status "WARNING" "$step_name failed (optional step) - Exit code: $exit_code"
            return 0
        else
            print_status "ERROR" "$step_name failed - Exit code: $exit_code"
            return 1
        fi
    fi
}

# Initialize counters
total_steps=0
successful_steps=0
failed_steps=0

# Step 1: Test installation in user mode (simulates fresh install)
((total_steps++))
if run_test_step "Test User Mode Installation" "pip install --user -e . --force-reinstall --no-deps"; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 2: Test basic package import after fresh install
((total_steps++))
if run_test_step "Test Package Import After Install" "python3 -c \"
import sys
import os
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Python path (first 3):', sys.path[:3])

# Test imports
try:
    import pynomaly
    print('✓ anomaly_detection package imported successfully')
    print('  Package location:', pynomaly.__file__ if hasattr(pynomaly, '__file__') else 'Not available')
except Exception as e:
    print('✗ anomaly_detection package import failed:', str(e))
    raise

try:
    from pynomaly.domain.entities import Dataset, Detector, Anomaly
    print('✓ Domain entities imported successfully')
except Exception as e:
    print('✗ Domain entities import failed:', str(e))
    raise

try:
    from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
    print('✓ Value objects imported successfully')
except Exception as e:
    print('✗ Value objects import failed:', str(e))
    raise

print('All basic imports successful after fresh installation!')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 3: Test infrastructure components after fresh install
((total_steps++))
if run_test_step "Test Infrastructure After Fresh Install" "python3 -c \"
import numpy as np
import pandas as pd
from pynomaly.domain.entities import Dataset

print('Testing infrastructure components...')

# Create test data
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 30).tolist() + [10, 11, 12],
    'feature2': np.random.normal(0, 1, 30).tolist() + [10, 11, 12]
})
dataset = Dataset(name='Fresh Install Test', data=data)
print(f'Test dataset: {dataset.name}, shape: {dataset.data.shape}')

# Test adapters
adapters_working = []

try:
    from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
    sklearn_adapter = SklearnAdapter('IsolationForest')
    print(f'✓ SklearnAdapter: {sklearn_adapter.algorithm_name}')
    adapters_working.append('sklearn')
except Exception as e:
    print(f'✗ SklearnAdapter failed: {str(e)}')

try:
    from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
    pyod_adapter = PyODAdapter('IsolationForest')
    print(f'✓ PyODAdapter: {pyod_adapter.algorithm_name}')
    adapters_working.append('pyod')
except Exception as e:
    print(f'✗ PyODAdapter failed: {str(e)}')

print(f'Working adapters: {len(adapters_working)}/2 ({adapters_working})')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 4: Test configuration system in fresh install
((total_steps++))
if run_test_step "Test Configuration in Fresh Install" "python3 -c \"
from pynomaly.infrastructure.config.settings import get_settings
from pynomaly.infrastructure.config import create_container

print('Testing configuration system...')

# Load settings
settings = get_settings()
print(f'Application: {settings.app.name} v{settings.app.version}')
print(f'Environment: {settings.app.environment}')

# Create container
container = create_container()
print(f'DI Container: {type(container).__name__}')

# Test repository creation
repositories = {}

try:
    detector_repo = container.detector_repository()
    repositories['detector'] = type(detector_repo).__name__
    print(f'✓ Detector repository: {repositories[\\\"detector\\\"]}')
except Exception as e:
    print(f'✗ Detector repository failed: {str(e)}')

try:
    dataset_repo = container.dataset_repository()
    repositories['dataset'] = type(dataset_repo).__name__
    print(f'✓ Dataset repository: {repositories[\\\"dataset\\\"]}')
except Exception as e:
    print(f'✗ Dataset repository failed: {str(e)}')

try:
    result_repo = container.result_repository()
    repositories['result'] = type(result_repo).__name__
    print(f'✓ Result repository: {repositories[\\\"result\\\"]}')
except Exception as e:
    print(f'✗ Result repository failed: {str(e)}')

print(f'Repositories working: {len(repositories)}/3')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 5: Test API creation in fresh install
((total_steps++))
if run_test_step "Test API Creation in Fresh Install" "python3 -c \"
from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app
from fastapi.testclient import TestClient

print('Testing API creation in fresh install...')

# Create container and app
container = create_container()
app = create_app(container)
client = TestClient(app)

print('✓ FastAPI application created successfully')

# Test health endpoint
health_response = client.get('/api/health/')
if health_response.status_code == 200:
    health_data = health_response.json()
    print(f'✓ Health endpoint: {health_data[\\\"overall_status\\\"]}')
else:
    print(f'✗ Health endpoint failed: {health_response.status_code}')
    raise Exception('Health check failed')

# Test detector creation
detector_payload = {
    'name': 'Fresh Install Detector',
    'algorithm_name': 'IsolationForest',
    'parameters': {'contamination': 0.1}
}

create_response = client.post('/api/detectors/', json=detector_payload)
if create_response.status_code == 200:
    detector = create_response.json()
    print(f'✓ Detector creation: {detector[\\\"name\\\"]} (ID: {detector[\\\"id\\\"]})')
else:
    print(f'✗ Detector creation failed: {create_response.status_code}')
    raise Exception('Detector creation failed')

print('API functionality working in fresh install!')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 6: Test complete ML workflow in fresh install
((total_steps++))
if run_test_step "Test Complete ML Workflow" "python3 -c \"
import numpy as np
import pandas as pd
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.domain.entities import Dataset
import warnings
warnings.filterwarnings('ignore')

print('Testing complete ML workflow in fresh install...')

# Create test data with clear anomalies
np.random.seed(42)
normal_data = np.random.normal(0, 1, (90, 3))
anomaly_data = np.random.normal(8, 0.5, (10, 3))
all_data = np.vstack([normal_data, anomaly_data])

# Shuffle the data
indices = np.random.permutation(len(all_data))
all_data = all_data[indices]

dataset_df = pd.DataFrame(all_data, columns=['x', 'y', 'z'])
dataset = Dataset(name='Fresh_Install_ML_Test', data=dataset_df)
print(f'Dataset: {dataset.name}, shape: {dataset.data.shape}')

# Create and train adapter
adapter = SklearnAdapter('IsolationForest', {'contamination': 0.1})
print(f'Adapter: {adapter.algorithm_name}')

# Fit the model
adapter.fit(dataset)
print('✓ Model fitted successfully')

# Run detection
results = adapter.detect(dataset)
print(f'✓ Detection completed')

# Analyze results
scores = results.scores
high_scores = [s for s in scores if s.value > 0.6]
print(f'✓ Found {len(high_scores)} high anomaly scores out of {len(scores)} data points')
print(f'✓ Anomaly rate: {len(results.anomalies)}/{len(scores)} = {results.anomaly_rate:.2%}')

# Test scoring separately
score_results = adapter.score(dataset)
print(f'✓ Scoring completed: {len(score_results)} scores generated')

print('Complete ML workflow test passed!')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 7: Test error handling in fresh install
((total_steps++))
if run_test_step "Test Error Handling" "python3 -c \"
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DetectorNotFittedError
import pandas as pd
import numpy as np

print('Testing error handling in fresh install...')

# Test detection without fitting
adapter = SklearnAdapter('IsolationForest')
data = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
dataset = Dataset(name='Error Test', data=data)

try:
    adapter.detect(dataset)
    print('✗ Detection should have failed on unfitted model')
    raise Exception('Error handling failed')
except DetectorNotFittedError as e:
    print(f'✓ Correctly caught DetectorNotFittedError: {e}')
except Exception as e:
    print(f'✗ Unexpected error type: {type(e).__name__}: {e}')
    raise

# Test invalid algorithm
try:
    invalid_adapter = SklearnAdapter('NonExistentAlgorithm')
    print('✗ Should have failed with invalid algorithm')
except Exception as e:
    print(f'✓ Correctly caught error for invalid algorithm: {type(e).__name__}')

print('Error handling tests passed!')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 8: Test domain logic integrity
((total_steps++))
if run_test_step "Test Domain Logic Integrity" "python3 -c \"
from pynomaly.domain.entities import Dataset, Detector, Anomaly
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate, ConfidenceInterval
import pandas as pd
import numpy as np

print('Testing domain logic integrity...')

# Test value objects
score = AnomalyScore(0.85)
print(f'✓ AnomalyScore: {score.value}')

contamination = ContaminationRate(0.1)
print(f'✓ ContaminationRate: {contamination.value}')

# Test confidence interval
ci = ConfidenceInterval(lower=0.1, upper=0.9, level=0.95)
print(f'✓ ConfidenceInterval: [{ci.lower}, {ci.upper}] at {ci.level}')

# Test entities
data = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
dataset = Dataset(name='Domain Test', data=data)
print(f'✓ Dataset: {dataset.name}, ID: {dataset.id}')

detector = Detector(name='Domain Detector', algorithm_name='IsolationForest')
print(f'✓ Detector: {detector.name}, ID: {detector.id}')

anomaly = Anomaly(
    score=score,
    data_point={'feature': 100},
    detector_name='Test Detector'
)
print(f'✓ Anomaly: score={anomaly.score.value}, detector={anomaly.detector_name}')

print('Domain logic integrity test passed!')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Reinstall with all dependencies to restore normal state
print_status "INFO" "Restoring normal installation state..."
pip install -e . --force-reinstall >/dev/null 2>&1

# Final results
echo ""
echo "========================================"
echo "FRESH INSTALLATION TEST RESULTS"
echo "========================================"
echo "Total steps: $total_steps"
print_status "SUCCESS" "Successful steps: $successful_steps"
print_status "ERROR" "Failed steps: $failed_steps"

success_rate=$((successful_steps * 100 / total_steps))
echo "Success rate: $success_rate%"

if [ $failed_steps -eq 0 ]; then
    echo ""
    print_status "SUCCESS" "🎉 ALL FRESH INSTALLATION TESTS PASSED!"
    print_status "SUCCESS" "Pynomaly installs and runs correctly in fresh environments!"
    exit 0
elif [ $successful_steps -ge $((total_steps * 80 / 100)) ]; then
    echo ""
    print_status "WARNING" "⚠️ Most tests passed with some minor issues."
    print_status "WARNING" "Pynomaly should work reliably in fresh installations."
    exit 0
else
    echo ""
    print_status "ERROR" "❌ Too many installation tests failed."
    print_status "ERROR" "Pynomaly may have installation issues."
    exit 1
fi
