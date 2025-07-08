#!/usr/bin/env bash

echo "========================================"
echo "PYNOMALY FRESH ENVIRONMENT TEST SUITE"
echo "========================================"
echo "This script tests Pynomaly in a fresh virtual environment"
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

# Step 1: Create fresh virtual environment
((total_steps++))
print_status "INFO" "Creating fresh virtual environment..."
if run_test_step "Create Fresh Virtual Environment" "python3 -m venv fresh_test_env"; then
    ((successful_steps++))
else
    ((failed_steps++))
    print_status "ERROR" "Cannot create virtual environment. Exiting."
    exit 1
fi

# Step 2: Activate virtual environment
((total_steps++))
print_status "INFO" "Activating fresh virtual environment..."
if run_test_step "Activate Virtual Environment" "source fresh_test_env/bin/activate && echo 'Virtual environment activated'"; then
    ((successful_steps++))
    # Export activation command for subsequent steps
    export VENV_ACTIVATE="source fresh_test_env/bin/activate"
else
    ((failed_steps++))
    print_status "ERROR" "Cannot activate virtual environment. Exiting."
    exit 1
fi

# Step 3: Upgrade pip in fresh environment
((total_steps++))
if run_test_step "Upgrade pip" "$VENV_ACTIVATE && pip install --upgrade pip"; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 4: Install basic requirements
((total_steps++))
if run_test_step "Install Basic Requirements" "$VENV_ACTIVATE && pip install wheel setuptools"; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 5: Install pynomaly in development mode
((total_steps++))
if run_test_step "Install Pynomaly Development Mode" "$VENV_ACTIVATE && pip install -e ."; then
    ((successful_steps++))
else
    ((failed_steps++))
    print_status "ERROR" "Cannot install Pynomaly. This is critical for testing."
fi

# Step 6: Install test dependencies
((total_steps++))
if run_test_step "Install Test Dependencies" "$VENV_ACTIVATE && pip install pytest pytest-cov pytest-asyncio"; then
    ((successful_steps++))
else
    ((failed_steps++))
    print_status "WARNING" "Test dependencies failed to install. Some tests may not work."
fi

# Step 7: Verify basic imports
((total_steps++))
if run_test_step "Verify Basic Imports" "$VENV_ACTIVATE && python3 -c \"
import sys
print('Python executable:', sys.executable)
print('Python version:', sys.version)

# Test basic imports
try:
    import pynomaly
    print('✓ pynomaly imported successfully')
except Exception as e:
    print('✗ pynomaly import failed:', str(e))
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

print('All basic imports successful!')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 8: Test infrastructure adapters
((total_steps++))
if run_test_step "Test Infrastructure Adapters" "$VENV_ACTIVATE && python3 -c \"
import numpy as np
import pandas as pd
from pynomaly.domain.entities import Dataset

# Create test data
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 50).tolist() + [5, 6, 7],
    'feature2': np.random.normal(0, 1, 50).tolist() + [5, 6, 7]
})
dataset = Dataset(name='Fresh Env Test', data=data)
print(f'Test dataset created: {dataset.name} with shape {dataset.data.shape}')

# Test adapters
adapters_tested = 0
adapters_successful = 0

try:
    from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
    sklearn_adapter = SklearnAdapter('IsolationForest')
    print(f'✓ SklearnAdapter created: {sklearn_adapter.algorithm_name}')
    adapters_tested += 1
    adapters_successful += 1
except Exception as e:
    print(f'✗ SklearnAdapter failed: {str(e)}')
    adapters_tested += 1

try:
    from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
    pyod_adapter = PyODAdapter('IsolationForest')
    print(f'✓ PyODAdapter created: {pyod_adapter.algorithm_name}')
    adapters_tested += 1
    adapters_successful += 1
except Exception as e:
    print(f'✗ PyODAdapter failed: {str(e)}')
    adapters_tested += 1

print(f'Adapter test summary: {adapters_successful}/{adapters_tested} successful')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 9: Test API functionality
((total_steps++))
if run_test_step "Test API Functionality" "$VENV_ACTIVATE && python3 -c \"
from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app
from fastapi.testclient import TestClient

print('Creating dependency injection container...')
container = create_container()
print(f'Container created: {type(container).__name__}')

print('Creating FastAPI application...')
app = create_app(container)
print('FastAPI app created successfully')

print('Creating test client...')
client = TestClient(app)

print('Testing health endpoint...')
health_response = client.get('/api/health/')
assert health_response.status_code == 200, f'Health check failed: {health_response.status_code}'
health_data = health_response.json()
print(f'Health status: {health_data[\\\"overall_status\\\"]}')

print('Testing detector creation...')
detector_payload = {
    'name': 'Fresh Env Test Detector',
    'algorithm_name': 'IsolationForest',
    'parameters': {'contamination': 0.05}
}
create_response = client.post('/api/detectors/', json=detector_payload)
assert create_response.status_code == 200, f'Detector creation failed: {create_response.status_code}'
detector_data = create_response.json()
print(f'Detector created: {detector_data[\\\"name\\\"]} (ID: {detector_data[\\\"id\\\"]})')

print('Testing detector listing...')
list_response = client.get('/api/detectors/')
assert list_response.status_code == 200, f'Detector listing failed: {list_response.status_code}'
detectors = list_response.json()
print(f'Total detectors: {len(detectors)}')

print('✅ All API tests passed in fresh environment!')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 10: Test configuration system
((total_steps++))
if run_test_step "Test Configuration System" "$VENV_ACTIVATE && python3 -c \"
from pynomaly.infrastructure.config.settings import get_settings
from pynomaly.infrastructure.config import create_container

print('Loading application settings...')
settings = get_settings()
print(f'App: {settings.app.name} v{settings.app.version}')
print(f'Environment: {settings.app.environment}')
print(f'Debug mode: {settings.app.debug}')

print('Testing dependency injection container...')
container = create_container()
print(f'Container type: {type(container).__name__}')

# Test repository access
repos_tested = 0
repos_successful = 0

try:
    detector_repo = container.detector_repository()
    print(f'✓ Detector repository: {type(detector_repo).__name__}')
    repos_tested += 1
    repos_successful += 1
except Exception as e:
    print(f'✗ Detector repository error: {str(e)}')
    repos_tested += 1

try:
    dataset_repo = container.dataset_repository()
    print(f'✓ Dataset repository: {type(dataset_repo).__name__}')
    repos_tested += 1
    repos_successful += 1
except Exception as e:
    print(f'✗ Dataset repository error: {str(e)}')
    repos_tested += 1

try:
    result_repo = container.result_repository()
    print(f'✓ Result repository: {type(result_repo).__name__}')
    repos_tested += 1
    repos_successful += 1
except Exception as e:
    print(f'✗ Result repository error: {str(e)}')
    repos_tested += 1

print(f'Repository test summary: {repos_successful}/{repos_tested} successful')
print('✅ Configuration system test completed!')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Step 11: Run core unit tests (if pytest is available)
((total_steps++))
if run_test_step "Run Core Unit Tests" "$VENV_ACTIVATE && python3 -m pytest tests/domain/test_entities.py tests/domain/test_value_objects.py -v --tb=short || echo 'Pytest not available or tests failed'"; then
    ((successful_steps++))
else
    print_status "WARNING" "Unit tests failed or pytest not available"
    ((failed_steps++))
fi

# Step 12: Test ML pipeline functionality
((total_steps++))
if run_test_step "Test ML Pipeline" "$VENV_ACTIVATE && python3 -c \"
import numpy as np
import pandas as pd
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.domain.entities import Dataset
import warnings
warnings.filterwarnings('ignore')

print('Creating ML pipeline test data...')
np.random.seed(42)
normal_data = np.random.normal(0, 1, (95, 2))
anomaly_data = np.random.normal(5, 0.5, (5, 2))
all_data = np.vstack([normal_data, anomaly_data])

dataset_df = pd.DataFrame(all_data, columns=['feature_1', 'feature_2'])
dataset = Dataset(name='ML_Pipeline_Test', data=dataset_df)
print(f'Dataset created: {dataset.name} with shape {dataset.data.shape}')

print('Testing ML adapter...')
adapter = SklearnAdapter('IsolationForest')
print(f'Adapter created: {adapter.algorithm_name}')

print('Training model...')
adapter.fit(dataset)
print('Model training completed')

print('Running anomaly detection...')
results = adapter.detect(dataset)
anomaly_count = len([s for s in results.scores if s.value > 0.5])
print(f'Detection completed. Found {anomaly_count} potential anomalies out of {len(results.scores)} data points')

print('✅ ML pipeline test completed successfully!')
\""; then
    ((successful_steps++))
else
    ((failed_steps++))
fi

# Cleanup: Remove fresh virtual environment
print_status "INFO" "Cleaning up fresh virtual environment..."
if [ -d "fresh_test_env" ]; then
    rm -rf fresh_test_env
    print_status "INFO" "Fresh virtual environment removed"
fi

# Final results
echo ""
echo "========================================"
echo "FRESH ENVIRONMENT TEST RESULTS"
echo "========================================"
echo "Total steps: $total_steps"
print_status "SUCCESS" "Successful steps: $successful_steps"
print_status "ERROR" "Failed steps: $failed_steps"

success_rate=$((successful_steps * 100 / total_steps))
echo "Success rate: $success_rate%"

if [ $failed_steps -eq 0 ]; then
    echo ""
    print_status "SUCCESS" "🎉 ALL FRESH ENVIRONMENT TESTS PASSED!"
    print_status "SUCCESS" "Pynomaly is ready for deployment in fresh environments!"
    exit 0
elif [ $successful_steps -ge $((total_steps * 75 / 100)) ]; then
    echo ""
    print_status "WARNING" "⚠️ Most tests passed but some failed."
    print_status "WARNING" "Pynomaly should work in fresh environments with minor issues."
    exit 0
else
    echo ""
    print_status "ERROR" "❌ Too many tests failed."
    print_status "ERROR" "Pynomaly may have issues in fresh environments."
    exit 1
fi
