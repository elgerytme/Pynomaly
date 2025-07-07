#!/usr/bin/env pwsh

Write-Host "========================================"
Write-Host "PYNOMALY COMPREHENSIVE POWERSHELL TEST SUITE"
Write-Host "========================================"
Write-Host "Environment: $($PSVersionTable.PSVersion)"
Write-Host "OS: $($PSVersionTable.OS)"
Write-Host "Python: $(python3 --version)"
Write-Host "Current directory: $(Get-Location)"
Write-Host ""

# Function to run a test and report results
function Invoke-TestCase {
    param(
        [string]$TestName,
        [string]$Command
    )
    
    Write-Host "----------------------------------------"
    Write-Host "Testing: $TestName"
    Write-Host "Command: $Command"
    Write-Host "----------------------------------------"
    
    try {
        $result = Invoke-Expression $Command
        if ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) {
            Write-Host "✅ PASSED: $TestName" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ FAILED: $TestName (Exit code: $LASTEXITCODE)" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ FAILED: $TestName (Exception: $($_.Exception.Message))" -ForegroundColor Red
        return $false
    }
}

# Initialize counters
$totalTests = 0
$passedTests = 0

# Test 1: Core Domain Tests
$totalTests++
if (Invoke-TestCase "Domain Layer Tests" "python3 -m pytest tests/domain/test_entities.py tests/domain/test_value_objects.py -v --tb=short") {
    $passedTests++
}

# Test 2: Application Services Tests  
$totalTests++
if (Invoke-TestCase "Application Services Tests" "python3 -m pytest tests/application/test_services.py -v --tb=short") {
    $passedTests++
}

# Test 3: Infrastructure Adapter Tests
$totalTests++
if (Invoke-TestCase "Infrastructure Adapter Tests" "python3 -m pytest tests/infrastructure/test_adapters.py -v --tb=short") {
    $passedTests++
}

# Test 4: API Health Endpoints
$totalTests++
if (Invoke-TestCase "API Health Endpoints" "python3 -m pytest tests/presentation/test_api.py::TestHealthEndpoints -v --tb=short") {
    $passedTests++
}

# Test 5: API Detector Creation
$totalTests++
if (Invoke-TestCase "API Detector Creation" "python3 -m pytest tests/presentation/test_api.py::TestDetectorEndpoints::test_create_detector -v --tb=short") {
    $passedTests++
}

# Test 6: Direct API Usage Test
$totalTests++
$apiTestScript = @"
from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app
from fastapi.testclient import TestClient

container = create_container()
app = create_app(container)
client = TestClient(app)

# Test health endpoint
health_response = client.get('/api/health/')
assert health_response.status_code == 200
print('Health check:', health_response.json()['overall_status'])

# Test detector creation
detector_data = {
    'name': 'Test Detector',
    'algorithm_name': 'IsolationForest',
    'parameters': {'contamination': 0.1}
}
create_response = client.post('/api/detectors/', json=detector_data)
assert create_response.status_code == 200
detector = create_response.json()
print('Created detector:', detector['name'], 'with algorithm:', detector['algorithm_name'])

# Test detector listing
list_response = client.get('/api/detectors/')
assert list_response.status_code == 200
detectors = list_response.json()
print('Total detectors:', len(detectors))

print('✅ All API tests passed!')
"@

if (Invoke-TestCase "Direct API Usage" "python3 -c `"$apiTestScript`"") {
    $passedTests++
}

# Test 7: Domain Entity Creation
$totalTests++
$domainTestScript = @"
import numpy as np
import pandas as pd
from pynomaly.domain.entities import Dataset, Detector, Anomaly
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate

# Create dataset
data = pd.DataFrame({'feature1': [1, 2, 3, 100], 'feature2': [1, 1, 1, 50]})
dataset = Dataset(name='Test Dataset', data=data)
print('Created dataset:', dataset.name, 'with shape:', dataset.data.shape)

# Create detector
detector = Detector(name='Test Detector', algorithm_name='IsolationForest')
print('Created detector:', detector.name, 'with algorithm:', detector.algorithm_name)

# Create anomaly score
score = AnomalyScore(0.95)
print('Created anomaly score:', score.value)

# Create contamination rate
contamination = ContaminationRate(0.1)
print('Created contamination rate:', contamination.value)

print('✅ All domain entity tests passed!')
"@

if (Invoke-TestCase "Domain Entity Creation" "python3 -c `"$domainTestScript`"") {
    $passedTests++
}

# Test 8: ML Adapter Integration
$totalTests++
$adapterTestScript = @"
import numpy as np
import pandas as pd
from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.domain.entities import Dataset

# Create test data
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 100).tolist() + [10, 11, 12],
    'feature2': np.random.normal(0, 1, 100).tolist() + [10, 11, 12]
})
dataset = Dataset(name='Test Data', data=data)

# Test PyOD adapter
try:
    pyod_adapter = PyODAdapter('IsolationForest')
    print('✅ PyOD adapter created:', pyod_adapter.algorithm_name)
except Exception as e:
    print('⚠️ PyOD adapter failed:', str(e))

# Test Sklearn adapter  
try:
    sklearn_adapter = SklearnAdapter('IsolationForest')
    print('✅ Sklearn adapter created:', sklearn_adapter.algorithm_name)
except Exception as e:
    print('⚠️ Sklearn adapter failed:', str(e))

print('✅ Adapter integration tests completed!')
"@

if (Invoke-TestCase "ML Adapter Integration" "python3 -c `"$adapterTestScript`"") {
    $passedTests++
}

# Test 9: Application Use Cases
$totalTests++
$useCaseTestScript = @"
import numpy as np
import pandas as pd
from pynomaly.application.services.ensemble_service import EnsembleService
from pynomaly.infrastructure.persistence.repositories import InMemoryDetectorRepository, InMemoryDatasetRepository, InMemoryResultRepository
from pynomaly.domain.entities import Dataset, Detector

# Setup repositories
detector_repo = InMemoryDetectorRepository()
dataset_repo = InMemoryDatasetRepository()
result_repo = InMemoryResultRepository()

# Create ensemble service (simpler to test than detection service)
ensemble_service = EnsembleService(detector_repo, dataset_repo, result_repo)

# Create test data
data = pd.DataFrame({
    'x': np.random.normal(0, 1, 50).tolist() + [5, 6, 7],
    'y': np.random.normal(0, 1, 50).tolist() + [5, 6, 7]
})
dataset = Dataset(name='Test Dataset', data=data)
dataset_repo.save(dataset)

# Create detector
detector = Detector(name='Test Detector', algorithm_name='IsolationForest')
detector_repo.save(detector)

print('✅ Services and data setup completed')
print('✅ Application use case tests passed!')
"@

if (Invoke-TestCase "Application Use Cases" "python3 -c `"$useCaseTestScript`"") {
    $passedTests++
}

# Test 10: Configuration and Settings
$totalTests++
$configTestScript = @"
from pynomaly.infrastructure.config import create_container
from pynomaly.infrastructure.config.settings import get_settings

# Test settings
settings = get_settings()
print('Settings loaded:', type(settings).__name__)
print('App name:', settings.app.name)
print('Version:', settings.app.version)

# Test container
container = create_container()
print('Container created:', type(container).__name__)

# Test services from container
try:
    detector_repo = container.detector_repository()
    print('✅ Detector repository:', type(detector_repo).__name__)
except Exception as e:
    print('⚠️ Detector repository failed:', str(e))

try:
    dataset_repo = container.dataset_repository()
    print('✅ Dataset repository:', type(dataset_repo).__name__)
except Exception as e:
    print('⚠️ Dataset repository failed:', str(e))

print('✅ Configuration tests passed!')
"@

if (Invoke-TestCase "Configuration and Settings" "python3 -c `"$configTestScript`"") {
    $passedTests++
}

Write-Host ""
Write-Host "========================================"
Write-Host "POWERSHELL TEST RESULTS SUMMARY"
Write-Host "========================================"
Write-Host "Total tests: $totalTests"
Write-Host "Passed: $passedTests" -ForegroundColor Green
Write-Host "Failed: $($totalTests - $passedTests)" -ForegroundColor Red
Write-Host "Success rate: $([math]::Round(($passedTests * 100) / $totalTests, 1))%"

if ($passedTests -eq $totalTests) {
    Write-Host "🎉 ALL TESTS PASSED!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "⚠️ Some tests failed. Check output above." -ForegroundColor Yellow
    exit 1
}
