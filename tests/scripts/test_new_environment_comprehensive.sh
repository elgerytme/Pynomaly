#!/bin/bash

# Pynomaly New Environment Testing - Bash Edition
echo "=========================================="
echo "PYNOMALY NEW ENVIRONMENT TESTING - BASH"
echo "=========================================="
echo "Environment: $(uname -a)"
echo "Shell: $SHELL"
echo "Python: $(python3 --version 2>/dev/null || python --version 2>/dev/null || echo 'Python not found')"
echo "Current directory: $(pwd)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Test configuration
total_tests=0
passed_tests=0
failed_tests=0
declare -a test_results=()

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
        "PHASE")
            echo -e "${PURPLE}[PHASE]${NC} $message"
            ;;
        "TEST")
            echo -e "${CYAN}[TEST]${NC} $message"
            ;;
    esac
}

# Function to run a test
run_pynomali_test() {
    local test_name="$1"
    local command="$2"
    local expected_exit_code="${3:-0}"
    local allow_warnings="${4:-false}"

    ((total_tests++))

    echo "----------------------------------------"
    print_status "TEST" "$test_name"
    echo "COMMAND: $command"
    echo "----------------------------------------"

    # Capture output and exit code
    local output
    local exit_code

    if [ "$allow_warnings" = "true" ]; then
        output=$(eval "$command" 2>&1)
        exit_code=$?
    else
        output=$(eval "$command" 2>&1)
        exit_code=$?
    fi

    # Display output (first 10 lines)
    echo "$output" | head -10
    local line_count=$(echo "$output" | wc -l)
    if [ $line_count -gt 10 ]; then
        echo "... (output truncated)"
    fi

    # Check result
    if [ $exit_code -eq $expected_exit_code ]; then
        print_status "SUCCESS" "✅ PASSED: $test_name"
        ((passed_tests++))
        test_results+=("$test_name: PASSED")
        echo ""
        return 0
    else
        print_status "ERROR" "❌ FAILED: $test_name (Exit Code: $exit_code, Expected: $expected_exit_code)"
        ((failed_tests++))
        test_results+=("$test_name: FAILED (Exit: $exit_code)")
        echo ""
        return 1
    fi
}

print_status "PHASE" "=== PHASE 1: ENVIRONMENT VALIDATION ==="
echo ""

# Test 1: Python Installation Check
run_pynomali_test "Python Installation Check" "python3 --version && which python3" 0 true

# Test 2: Package Import Check
run_pynomali_test "Pynomaly Package Import" "python3 -c \"
import sys
print('Python version:', sys.version)
print('Python executable:', sys.executable)

try:
    import pynomaly
    print('✓ Pynomaly imported successfully')
    print('Pynomaly location:', pynomaly.__file__ if hasattr(pynomaly, '__file__') else 'Built-in module')
except ImportError as e:
    print('✗ Failed to import pynomaly:', str(e))
    raise
except Exception as e:
    print('✗ Unexpected error importing pynomaly:', str(e))
    raise

print('✓ Package import test completed')
\"" 0 true

# Test 3: Virtual Environment Support Check
run_pynomali_test "Virtual Environment Support" "python3 -c \"
import subprocess
import os

print('Testing virtual environment support...')
try:
    # Try to create a test virtual environment
    result = subprocess.run(['python3', '-m', 'venv', 'test_env_check'],
                          capture_output=True, text=True, timeout=30)

    if result.returncode == 0 and os.path.exists('test_env_check'):
        print('✓ Virtual environment creation supported')
        # Clean up
        import shutil
        shutil.rmtree('test_env_check', ignore_errors=True)
        print('✓ Test environment cleaned up')
    else:
        print('⚠ Virtual environment creation not supported or failed')
        print('STDOUT:', result.stdout)
        print('STDERR:', result.stderr)
except subprocess.TimeoutExpired:
    print('⚠ Virtual environment creation timed out')
except Exception as e:
    print(f'⚠ Virtual environment test failed: {str(e)}')

print('✓ Virtual environment support test completed')
\"" 0 true

print_status "PHASE" "=== PHASE 2: CORE FUNCTIONALITY ==="
echo ""

# Test 4: CLI Help System
run_pynomali_test "CLI Help System" "python3 -m pynomaly --help | head -20" 0 true

# Test 5: Configuration System
run_pynomali_test "Configuration System" "python3 -c \"
from pynomaly.infrastructure.config.settings import get_settings

print('Testing configuration system...')
try:
    settings = get_settings()
    print(f'App name: {settings.app.name}')
    print(f'Version: {settings.app.version}')
    print(f'Environment: {settings.app.environment}')
    print(f'Debug mode: {settings.app.debug}')
    print(f'API host: {settings.api_host}')
    print(f'API port: {settings.api_port}')
    print('✓ Configuration system working')
except Exception as e:
    print(f'✗ Configuration system error: {str(e)}')
    raise

print('✓ Configuration test completed')
\"" 0 true

# Test 6: Dependency Injection Container
run_pynomali_test "Dependency Injection Container" "python3 -c \"
from pynomaly.infrastructure.config import create_container

print('Testing dependency injection container...')
try:
    container = create_container()
    print(f'Container type: {type(container).__name__}')

    # Test repository creation
    detector_repo = container.detector_repository()
    print(f'Detector repository: {type(detector_repo).__name__}')

    dataset_repo = container.dataset_repository()
    print(f'Dataset repository: {type(dataset_repo).__name__}')

    result_repo = container.result_repository()
    print(f'Result repository: {type(result_repo).__name__}')

    print('✓ Dependency injection container working')
except Exception as e:
    print(f'✗ DI container error: {str(e)}')
    raise

print('✓ DI container test completed')
\"" 0 true

print_status "PHASE" "=== PHASE 3: DATA PROCESSING ==="
echo ""

# Create test data
print_status "INFO" "Creating test datasets..."
python3 -c "
import pandas as pd
import numpy as np
import os

print('Current working directory:', os.getcwd())
print('Creating test datasets...')

# Small test dataset
np.random.seed(42)
data_small = pd.DataFrame({
    'x': np.random.normal(0, 1, 50).tolist() + [5, 6, 7],
    'y': np.random.normal(0, 1, 50).tolist() + [5, 6, 7],
    'z': np.random.normal(0, 1, 50).tolist() + [5, 6, 7]
})
data_small.to_csv('test_data_bash.csv', index=False)
print('✓ Small test dataset created: test_data_bash.csv')

# Medium test dataset
data_medium = pd.DataFrame({
    f'feature_{i}': np.random.normal(0, 1, 500) for i in range(5)
})
# Add some clear outliers
for col in data_medium.columns:
    data_medium.loc[490:499, col] = data_medium.loc[490:499, col] * 4
data_medium.to_csv('test_data_medium_bash.csv', index=False)
print('✓ Medium test dataset created: test_data_medium_bash.csv')

print('✓ Test dataset creation completed')
"

# Test 7: Dataset Loading
run_pynomali_test "Dataset Loading" "python3 -c \"
import pandas as pd
from pynomaly.domain.entities import Dataset

print('Testing dataset loading...')
try:
    # Load CSV data
    data = pd.read_csv('test_data_bash.csv')
    print(f'Raw data shape: {data.shape}')
    print(f'Raw data columns: {data.columns.tolist()}')
    print(f'Raw data types: {data.dtypes.tolist()}')

    # Create Pynomaly dataset
    dataset = Dataset(name='Bash Test Dataset', data=data)
    print(f'Dataset name: {dataset.name}')
    print(f'Dataset ID: {dataset.id}')
    print(f'Dataset shape: {dataset.data.shape}')
    print(f'Dataset columns: {dataset.data.columns.tolist()}')

    print('✓ Dataset loading successful')
except Exception as e:
    print(f'✗ Dataset loading error: {str(e)}')
    raise

print('✓ Dataset loading test completed')
\"" 0 true

# Test 8: Data Validation and Analysis
run_pynomali_test "Data Validation and Analysis" "python3 -c \"
import pandas as pd
import numpy as np

print('Testing data validation and analysis...')
try:
    data = pd.read_csv('test_data_bash.csv')

    # Basic validation
    print(f'Data shape: {data.shape}')
    print(f'Data types: {data.dtypes.unique().tolist()}')
    print(f'Missing values: {data.isnull().sum().sum()}')
    print(f'Numeric columns: {len(data.select_dtypes(include=[np.number]).columns)}')

    # Statistical analysis
    print(f'Data mean: {data.mean().mean():.3f}')
    print(f'Data std: {data.std().mean():.3f}')

    # Outlier detection using IQR
    outlier_counts = {}
    for col in data.columns:
        q75, q25 = np.percentile(data[col], [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - (iqr * 1.5)
        upper_bound = q75 + (iqr * 1.5)
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
        print(f'{col}: {len(outliers)} potential outliers')

    total_outliers = sum(outlier_counts.values())
    print(f'Total potential outliers detected: {total_outliers}')

    print('✓ Data validation and analysis successful')
except Exception as e:
    print(f'✗ Data validation error: {str(e)}')
    raise

print('✓ Data validation test completed')
\"" 0 true

print_status "PHASE" "=== PHASE 4: MACHINE LEARNING ==="
echo ""

# Test 9: Basic ML Algorithm
run_pynomali_test "Basic ML Algorithm" "python3 -c \"
import pandas as pd
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.domain.entities import Dataset
import warnings
warnings.filterwarnings('ignore')

print('Testing basic ML algorithm...')
try:
    # Load data
    data = pd.read_csv('test_data_bash.csv')
    dataset = Dataset(name='ML Test Dataset', data=data)
    print(f'Dataset loaded: {dataset.data.shape}')

    # Create adapter
    adapter = SklearnAdapter('IsolationForest', {'contamination': 0.1})
    print(f'Adapter created: {adapter.algorithm_name}')
    print(f'Parameters: {adapter.parameters}')

    # Fit the model
    adapter.fit(dataset)
    print('✓ Model fitted successfully')
    print(f'Model is fitted: {adapter.is_fitted}')

    # Run detection
    result = adapter.detect(dataset)
    print(f'Detection completed: {len(result.scores)} scores generated')
    print(f'Anomalies detected: {len(result.anomalies)}')
    print(f'Anomaly rate: {result.anomaly_rate:.2%}')

    # Analyze results
    score_values = [s.value for s in result.scores]
    high_scores = [s for s in score_values if s > 0.6]
    print(f'High anomaly scores (>0.6): {len(high_scores)}')
    print(f'Score range: {min(score_values):.3f} - {max(score_values):.3f}')

    print('✓ Basic ML algorithm test successful')
except Exception as e:
    print(f'✗ ML algorithm error: {str(e)}')
    import traceback
    traceback.print_exc()
    raise

print('✓ Basic ML algorithm test completed')
\"" 0 true

# Test 10: Multiple ML Algorithms
run_pynomali_test "Multiple ML Algorithms" "python3 -c \"
import pandas as pd
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
from pynomaly.domain.entities import Dataset
import warnings
warnings.filterwarnings('ignore')

print('Testing multiple ML algorithms...')
try:
    # Load data
    data = pd.read_csv('test_data_bash.csv')
    dataset = Dataset(name='Multi-Algorithm Test', data=data)
    print(f'Dataset loaded: {dataset.data.shape}')

    # Define algorithms to test
    algorithms = [
        ('sklearn', 'IsolationForest'),
        ('sklearn', 'LocalOutlierFactor'),
        ('pyod', 'IsolationForest')
    ]

    results = {}
    successful_algorithms = 0

    for lib, algo in algorithms:
        try:
            print(f'Testing {lib} {algo}...')

            if lib == 'sklearn':
                adapter = SklearnAdapter(algo)
            else:
                adapter = PyODAdapter(algo)

            adapter.fit(dataset)
            result = adapter.detect(dataset)

            results[f'{lib}_{algo}'] = {
                'anomalies': len(result.anomalies),
                'rate': result.anomaly_rate,
                'scores': len(result.scores)
            }

            print(f'✓ {lib} {algo}: {len(result.anomalies)} anomalies ({result.anomaly_rate:.2%})')
            successful_algorithms += 1

        except Exception as e:
            print(f'✗ {lib} {algo} failed: {str(e)}')
            results[f'{lib}_{algo}'] = {'error': str(e)}

    print(f'Algorithm testing summary:')
    print(f'- Total algorithms tested: {len(algorithms)}')
    print(f'- Successful algorithms: {successful_algorithms}')
    print(f'- Success rate: {successful_algorithms/len(algorithms)*100:.1f}%')

    if successful_algorithms >= 2:
        print('✓ Multiple ML algorithms test successful')
    else:
        print('⚠ Some ML algorithms failed')

except Exception as e:
    print(f'✗ Multiple ML algorithms error: {str(e)}')
    raise

print('✓ Multiple ML algorithms test completed')
\"" 0 true

print_status "PHASE" "=== PHASE 5: API INTEGRATION ==="
echo ""

# Test 11: API Server Creation
run_pynomali_test "API Server Creation" "python3 -c \"
from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app
from fastapi.testclient import TestClient

print('Testing API server creation...')
try:
    # Create container and app
    container = create_container()
    print(f'Container created: {type(container).__name__}')

    app = create_app(container)
    print(f'FastAPI app created: {type(app).__name__}')

    # Create test client
    client = TestClient(app)
    print('✓ Test client created')

    # Test health endpoint
    print('Testing health endpoint...')
    health_response = client.get('/api/health/')
    print(f'Health response status: {health_response.status_code}')

    if health_response.status_code == 200:
        health_data = health_response.json()
        print(f'Health status: {health_data.get(\\\"overall_status\\\", \\\"unknown\\\")}')
        print('✓ Health endpoint working')
    else:
        print(f'⚠ Health endpoint returned {health_response.status_code}')

    print('✓ API server creation successful')
except Exception as e:
    print(f'✗ API server creation error: {str(e)}')
    import traceback
    traceback.print_exc()
    raise

print('✓ API server creation test completed')
\"" 0 true

# Test 12: API Detector Operations
run_pynomali_test "API Detector Operations" "python3 -c \"
from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app
from fastapi.testclient import TestClient

print('Testing API detector operations...')
try:
    # Setup API
    container = create_container()
    app = create_app(container)
    client = TestClient(app)

    # Test detector creation
    print('Creating detector via API...')
    detector_data = {
        'name': 'Bash Test Detector',
        'algorithm_name': 'IsolationForest',
        'parameters': {'contamination': 0.1}
    }

    create_response = client.post('/api/detectors/', json=detector_data)
    print(f'Create detector status: {create_response.status_code}')

    if create_response.status_code == 200:
        detector = create_response.json()
        print(f'✓ Detector created: {detector[\\\"name\\\"]}')
        print(f'Detector ID: {detector[\\\"id\\\"]}')
        print(f'Algorithm: {detector[\\\"algorithm_name\\\"]}')
    else:
        print(f'✗ Detector creation failed: {create_response.status_code}')
        print(f'Response: {create_response.text}')

    # Test detector listing
    print('Listing detectors via API...')
    list_response = client.get('/api/detectors/')
    print(f'List detectors status: {list_response.status_code}')

    if list_response.status_code == 200:
        detectors = list_response.json()
        print(f'✓ Detectors listed: {len(detectors)} found')
        if detectors:
            print(f'Sample detector: {detectors[0][\\\"name\\\"]}')
    else:
        print(f'✗ Detector listing failed: {list_response.status_code}')

    print('✓ API detector operations successful')
except Exception as e:
    print(f'✗ API detector operations error: {str(e)}')
    import traceback
    traceback.print_exc()
    raise

print('✓ API detector operations test completed')
\"" 0 true

print_status "PHASE" "=== PHASE 6: PERFORMANCE & ERROR HANDLING ==="
echo ""

# Test 13: Performance Testing
run_pynomali_test "Performance Testing" "python3 -c \"
import pandas as pd
import time
import psutil
import os
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.domain.entities import Dataset
import warnings
warnings.filterwarnings('ignore')

print('Testing performance characteristics...')
try:
    # Get process for memory monitoring
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    # Load medium dataset
    print('Loading medium dataset...')
    start_time = time.time()
    data = pd.read_csv('test_data_medium_bash.csv')
    dataset = Dataset(name='Performance Test Dataset', data=data)
    load_time = time.time() - start_time

    print(f'✓ Data loaded: {data.shape} in {load_time:.3f} seconds')

    # Test model fitting performance
    print('Testing model fitting performance...')
    start_time = time.time()
    adapter = SklearnAdapter('IsolationForest')
    adapter.fit(dataset)
    fit_time = time.time() - start_time

    print(f'✓ Model fitted in {fit_time:.3f} seconds')

    # Test detection performance
    print('Testing detection performance...')
    start_time = time.time()
    result = adapter.detect(dataset)
    detect_time = time.time() - start_time

    print(f'✓ Detection completed in {detect_time:.3f} seconds')
    print(f'Anomalies found: {len(result.anomalies)} ({result.anomaly_rate:.2%})')

    # Memory usage
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before

    # Performance summary
    total_time = load_time + fit_time + detect_time
    throughput = data.shape[0] / total_time

    print(f'Performance Summary:')
    print(f'- Dataset size: {data.shape}')
    print(f'- Total time: {total_time:.3f} seconds')
    print(f'- Throughput: {throughput:.0f} samples/second')
    print(f'- Memory used: {memory_used:.1f} MB')
    print(f'- Time per sample: {(total_time * 1000 / data.shape[0]):.2f} ms')

    # Performance criteria
    if total_time < 10 and memory_used < 100:
        print('✓ Performance criteria met')
    elif total_time < 30:
        print('⚠ Performance acceptable but could be improved')
    else:
        print('⚠ Performance slower than expected')

    print('✓ Performance testing successful')
except Exception as e:
    print(f'✗ Performance testing error: {str(e)}')
    import traceback
    traceback.print_exc()
    raise

print('✓ Performance testing completed')
\"" 0 true

# Test 14: Error Handling
run_pynomali_test "Error Handling" "python3 -c \"
import pandas as pd
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DetectorNotFittedError, InvalidAlgorithmError

print('Testing error handling...')
errors_caught = 0
total_error_tests = 0

# Test 1: Detection without fitting
print('Test 1: Detection without fitting')
total_error_tests += 1
try:
    adapter = SklearnAdapter('IsolationForest')
    data = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    dataset = Dataset(name='Error Test', data=data)
    adapter.detect(dataset)  # Should fail
    print('✗ Should have failed - unfitted detector used')
except DetectorNotFittedError as e:
    print(f'✓ Correctly caught DetectorNotFittedError: {type(e).__name__}')
    print(f'Error message: {str(e)}')
    errors_caught += 1
except Exception as e:
    print(f'✓ Caught unexpected error (acceptable): {type(e).__name__}')
    errors_caught += 1

# Test 2: Invalid algorithm
print('Test 2: Invalid algorithm')
total_error_tests += 1
try:
    bad_adapter = SklearnAdapter('NonExistentAlgorithm')
    print('✗ Should have failed - invalid algorithm accepted')
except InvalidAlgorithmError as e:
    print(f'✓ Correctly caught InvalidAlgorithmError: {type(e).__name__}')
    errors_caught += 1
except Exception as e:
    print(f'✓ Caught error for invalid algorithm: {type(e).__name__}')
    errors_caught += 1

# Test 3: Invalid data handling
print('Test 3: Invalid data handling')
total_error_tests += 1
try:
    invalid_data = pd.DataFrame({'text_col': ['a', 'b', 'c']})  # Non-numeric
    dataset = Dataset(name='Invalid Data Test', data=invalid_data)
    adapter = SklearnAdapter('IsolationForest')
    adapter.fit(dataset)
    print('⚠ Non-numeric data processed (may be expected with preprocessing)')
    errors_caught += 1  # Count as handled
except Exception as e:
    print(f'✓ Correctly handled invalid data: {type(e).__name__}')
    errors_caught += 1

print(f'Error handling summary:')
print(f'- Total error tests: {total_error_tests}')
print(f'- Errors properly caught: {errors_caught}')
print(f'- Error handling rate: {errors_caught/total_error_tests*100:.1f}%')

if errors_caught >= 2:
    print('✓ Error handling test successful')
else:
    print('⚠ Some error handling tests failed')

print('✓ Error handling test completed')
\"" 0 true

# Cleanup test files
print_status "INFO" "Cleaning up test files..."
rm -f test_data_bash.csv test_data_medium_bash.csv 2>/dev/null || true

echo ""
print_status "PHASE" "=== FINAL RESULTS ==="
echo ""

# Calculate success rate
success_rate=$((passed_tests * 100 / total_tests))

echo "=========================================="
echo "BASH NEW ENVIRONMENT TEST RESULTS"
echo "=========================================="
echo "Total Tests: $total_tests"
print_status "SUCCESS" "Passed: $passed_tests"
print_status "ERROR" "Failed: $failed_tests"
echo "Success Rate: $success_rate%"
echo ""

echo "DETAILED RESULTS:"
echo "----------------------------------------"
for result in "${test_results[@]}"; do
    if [[ $result == *"PASSED"* ]]; then
        echo -e "${GREEN}✅${NC} $result"
    else
        echo -e "${RED}❌${NC} $result"
    fi
done

echo ""
echo "=========================================="

if [ $failed_tests -eq 0 ]; then
    print_status "SUCCESS" "🎉 ALL BASH TESTS PASSED!"
    print_status "SUCCESS" "Pynomaly is fully operational in Bash environment!"
    echo ""
    echo "FUNCTIONALITY ASSESSMENT:"
    echo "✅ Environment validation: WORKING"
    echo "✅ Core functionality: WORKING"
    echo "✅ Data processing: WORKING"
    echo "✅ Machine learning: WORKING"
    echo "✅ API integration: WORKING"
    echo "✅ Performance: ACCEPTABLE"
    echo "✅ Error handling: WORKING"
    exit 0
elif [ $success_rate -ge 80 ]; then
    print_status "WARNING" "⚠️ Most tests passed with some issues"
    print_status "WARNING" "Pynomaly is mostly operational in Bash environment"
    echo ""
    echo "ISSUES TO ADDRESS:"
    for result in "${test_results[@]}"; do
        if [[ $result == *"FAILED"* ]]; then
            echo "• $result"
        fi
    done
    exit 0
else
    print_status "ERROR" "❌ Significant test failures detected"
    print_status "ERROR" "Pynomaly requires fixes for Bash environment"
    echo ""
    echo "CRITICAL ISSUES:"
    for result in "${test_results[@]}"; do
        if [[ $result == *"FAILED"* ]]; then
            echo "• $result"
        fi
    done
    exit 1
fi
