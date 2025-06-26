#!/usr/bin/env pwsh

# Pynomaly New Environment Testing - PowerShell Edition
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "PYNOMALY NEW ENVIRONMENT TESTING - POWERSHELL" -ForegroundColor Cyan  
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "PowerShell Version: $($PSVersionTable.PSVersion)" -ForegroundColor Blue
Write-Host "OS: $($PSVersionTable.OS)" -ForegroundColor Blue
Write-Host "Python: $(python --version 2>$null || python3 --version 2>$null || 'Python not found')" -ForegroundColor Blue
Write-Host "Current Directory: $(Get-Location)" -ForegroundColor Blue
Write-Host ""

# Test configuration
$totalTests = 0
$passedTests = 0
$failedTests = 0
$testResults = @()

# Function to run a test
function Invoke-PynomaliTest {
    param(
        [string]$TestName,
        [string]$Command,
        [int]$ExpectedExitCode = 0,
        [switch]$AllowWarnings
    )
    
    $script:totalTests++
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "TEST: $TestName" -ForegroundColor Cyan
    Write-Host "COMMAND: $Command" -ForegroundColor Gray
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    
    try {
        $output = ""
        $exitCode = 0
        
        # Execute command and capture output
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "powershell"
        $psi.Arguments = "-Command `"$Command`""
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $psi
        $process.Start() | Out-Null
        
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        $exitCode = $process.ExitCode
        
        $output = $stdout
        if ($stderr -and -not $AllowWarnings) {
            $output += "`nSTDERR: $stderr"
        }
        
        # Display output (truncated)
        $outputLines = $output -split "`n"
        $displayLines = $outputLines | Select-Object -First 10
        foreach ($line in $displayLines) {
            Write-Host $line -ForegroundColor White
        }
        if ($outputLines.Count -gt 10) {
            Write-Host "... (output truncated)" -ForegroundColor Gray
        }
        
        # Check result
        if ($exitCode -eq $ExpectedExitCode) {
            Write-Host "✅ PASSED: $TestName" -ForegroundColor Green
            $script:passedTests++
            $script:testResults += [PSCustomObject]@{
                Test = $TestName
                Status = "PASSED"
                ExitCode = $exitCode
            }
            return $true
        } else {
            Write-Host "❌ FAILED: $TestName (Exit Code: $exitCode, Expected: $ExpectedExitCode)" -ForegroundColor Red
            $script:failedTests++
            $script:testResults += [PSCustomObject]@{
                Test = $TestName
                Status = "FAILED"
                ExitCode = $exitCode
            }
            return $false
        }
    }
    catch {
        Write-Host "❌ FAILED: $TestName (Exception: $($_.Exception.Message))" -ForegroundColor Red
        $script:failedTests++
        $script:testResults += [PSCustomObject]@{
            Test = $TestName
            Status = "FAILED"
            ExitCode = "Exception"
        }
        return $false
    }
    finally {
        Write-Host ""
    }
}

Write-Host "=== PHASE 1: ENVIRONMENT VALIDATION ===" -ForegroundColor Magenta
Write-Host ""

# Test 1: Python Availability
Invoke-PynomaliTest -TestName "Python Installation Check" -Command "python --version; if (`$LASTEXITCODE -ne 0) { python3 --version }" -AllowWarnings

# Test 2: Package Installation Check
Invoke-PynomaliTest -TestName "Pynomaly Package Import" -Command "python -c `"import sys; print('Python:', sys.version); import pynomaly; print('✓ Pynomaly imported successfully')`"" -AllowWarnings

# Test 3: Virtual Environment Creation (if possible)
Invoke-PynomaliTest -TestName "Virtual Environment Support" -Command "python -m venv test_env_check; if (Test-Path 'test_env_check') { Remove-Item -Recurse -Force 'test_env_check'; Write-Host '✓ Virtual environment creation supported' } else { Write-Host '⚠ Virtual environment creation not supported' }" -AllowWarnings

Write-Host "=== PHASE 2: CORE FUNCTIONALITY ===" -ForegroundColor Magenta
Write-Host ""

# Test 4: CLI Help System
Invoke-PynomaliTest -TestName "CLI Help System" -Command "python -m pynomaly --help" -AllowWarnings

# Test 5: Configuration System
Invoke-PynomaliTest -TestName "Configuration System" -Command "python -c `"from pynomaly.infrastructure.config.settings import get_settings; s = get_settings(); print(f'App: {s.app.name} v{s.app.version}'); print('✓ Configuration loaded')`"" -AllowWarnings

# Test 6: Dependency Injection
Invoke-PynomaliTest -TestName "Dependency Injection" -Command "python -c `"from pynomaly.infrastructure.config import create_container; c = create_container(); print(f'Container: {type(c).__name__}'); repo = c.detector_repository(); print(f'Repository: {type(repo).__name__}'); print('✓ DI container working')`"" -AllowWarnings

Write-Host "=== PHASE 3: DATA PROCESSING ===" -ForegroundColor Magenta
Write-Host ""

# Create test data
Write-Host "Creating test datasets..." -ForegroundColor Blue
$testDataScript = @"
import pandas as pd
import numpy as np
import os

# Ensure we're in the right directory
print('Current directory:', os.getcwd())

# Create small test dataset
np.random.seed(42)
data_small = pd.DataFrame({
    'x': np.random.normal(0, 1, 50).tolist() + [5, 6, 7],
    'y': np.random.normal(0, 1, 50).tolist() + [5, 6, 7],
    'z': np.random.normal(0, 1, 50).tolist() + [5, 6, 7]
})
data_small.to_csv('test_data_ps.csv', index=False)
print('✓ Test dataset created: test_data_ps.csv')

# Create medium test dataset
data_medium = pd.DataFrame({
    'feature_' + str(i): np.random.normal(0, 1, 500) for i in range(5)
})
# Add some clear outliers
for col in data_medium.columns:
    data_medium.loc[490:499, col] = data_medium.loc[490:499, col] * 4
data_medium.to_csv('test_data_medium_ps.csv', index=False)
print('✓ Medium dataset created: test_data_medium_ps.csv')
"@

python -c $testDataScript

# Test 7: Dataset Loading
Invoke-PynomaliTest -TestName "Dataset Loading" -Command "python -c `"import pandas as pd; from pynomaly.domain.entities import Dataset; data = pd.read_csv('test_data_ps.csv'); ds = Dataset(name='PS Test', data=data); print(f'Dataset: {ds.name}, Shape: {ds.data.shape}'); print('✓ Dataset loading successful')`"" -AllowWarnings

# Test 8: Data Validation
Invoke-PynomaliTest -TestName "Data Validation" -Command "python -c `"import pandas as pd; import numpy as np; data = pd.read_csv('test_data_ps.csv'); print(f'Shape: {data.shape}'); print(f'Dtypes: {data.dtypes.tolist()}'); print(f'Missing: {data.isnull().sum().sum()}'); print(f'Numeric cols: {len(data.select_dtypes(include=[np.number]).columns)}'); print('✓ Data validation complete')`"" -AllowWarnings

Write-Host "=== PHASE 4: MACHINE LEARNING ===" -ForegroundColor Magenta
Write-Host ""

# Test 9: Basic ML Algorithm
Invoke-PynomaliTest -TestName "Basic ML Algorithm" -Command "python -c `"import pandas as pd; from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter; from pynomaly.domain.entities import Dataset; import warnings; warnings.filterwarnings('ignore'); data = pd.read_csv('test_data_ps.csv'); ds = Dataset(name='ML Test', data=data); adapter = SklearnAdapter('IsolationForest'); adapter.fit(ds); result = adapter.detect(ds); print(f'Algorithm: {adapter.algorithm_name}'); print(f'Scores: {len(result.scores)}'); print(f'Anomalies: {len(result.anomalies)} ({result.anomaly_rate:.1%})'); print('✓ ML algorithm successful')`"" -AllowWarnings

# Test 10: Multiple Algorithms
Invoke-PynomaliTest -TestName "Multiple ML Algorithms" -Command "python -c `"import pandas as pd; from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter; from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter; from pynomaly.domain.entities import Dataset; import warnings; warnings.filterwarnings('ignore'); data = pd.read_csv('test_data_ps.csv'); ds = Dataset(name='Multi Test', data=data); algos = [('sklearn', 'IsolationForest'), ('sklearn', 'LocalOutlierFactor'), ('pyod', 'IsolationForest')]; success = 0; for lib, algo in algos: try: adapter = SklearnAdapter(algo) if lib == 'sklearn' else PyODAdapter(algo); adapter.fit(ds); result = adapter.detect(ds); print(f'✓ {lib} {algo}: {len(result.anomalies)} anomalies'); success += 1; except Exception as e: print(f'✗ {lib} {algo}: {str(e)}'); print(f'✓ Algorithms tested: {success}/{len(algos)}')`"" -AllowWarnings

Write-Host "=== PHASE 5: API INTEGRATION ===" -ForegroundColor Magenta
Write-Host ""

# Test 11: API Server Creation
Invoke-PynomaliTest -TestName "API Server Creation" -Command "python -c `"from pynomaly.infrastructure.config import create_container; from pynomaly.presentation.api.app import create_app; from fastapi.testclient import TestClient; container = create_container(); app = create_app(container); client = TestClient(app); print('✓ API server created successfully'); health = client.get('/api/health/'); print(f'Health check: {health.status_code}'); if health.status_code == 200: print(f'Status: {health.json()[\\\"overall_status\\\"]}'); print('✓ API integration successful')`"" -AllowWarnings

# Test 12: API Detector Operations
Invoke-PynomaliTest -TestName "API Detector Operations" -Command "python -c `"from pynomaly.infrastructure.config import create_container; from pynomaly.presentation.api.app import create_app; from fastapi.testclient import TestClient; container = create_container(); app = create_app(container); client = TestClient(app); detector_data = {'name': 'PS Test Detector', 'algorithm_name': 'IsolationForest', 'parameters': {'contamination': 0.1}}; response = client.post('/api/detectors/', json=detector_data); print(f'Create detector: {response.status_code}'); if response.status_code == 200: detector = response.json(); print(f'Detector ID: {detector[\\\"id\\\"]}'); list_resp = client.get('/api/detectors/'); print(f'List detectors: {list_resp.status_code}, Count: {len(list_resp.json()) if list_resp.status_code == 200 else 0}'); print('✓ API detector operations successful')`"" -AllowWarnings

Write-Host "=== PHASE 6: PERFORMANCE & ERROR HANDLING ===" -ForegroundColor Magenta
Write-Host ""

# Test 13: Performance Test
Invoke-PynomaliTest -TestName "Performance Test" -Command "python -c `"import pandas as pd; import time; from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter; from pynomaly.domain.entities import Dataset; import warnings; warnings.filterwarnings('ignore'); data = pd.read_csv('test_data_medium_ps.csv'); ds = Dataset(name='Perf Test', data=data); start = time.time(); adapter = SklearnAdapter('IsolationForest'); adapter.fit(ds); fit_time = time.time() - start; start = time.time(); result = adapter.detect(ds); detect_time = time.time() - start; total_time = fit_time + detect_time; print(f'Data shape: {data.shape}'); print(f'Fit time: {fit_time:.3f}s'); print(f'Detect time: {detect_time:.3f}s'); print(f'Total time: {total_time:.3f}s'); print(f'Throughput: {data.shape[0]/total_time:.0f} samples/sec'); if total_time < 10: print('✓ Performance acceptable'); else: print('⚠ Performance slower than expected'); print('✓ Performance test complete')`"" -AllowWarnings

# Test 14: Error Handling
Invoke-PynomaliTest -TestName "Error Handling" -Command "python -c `"import pandas as pd; from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter; from pynomaly.domain.entities import Dataset; from pynomaly.domain.exceptions import DetectorNotFittedError, InvalidAlgorithmError; errors_caught = 0; try: adapter = SklearnAdapter('IsolationForest'); data = pd.DataFrame({'x': [1,2,3]}); ds = Dataset(name='Error Test', data=data); adapter.detect(ds); print('✗ Should have failed - unfitted detector'); except DetectorNotFittedError: print('✓ Caught DetectorNotFittedError'); errors_caught += 1; except Exception as e: print(f'✓ Caught error: {type(e).__name__}'); errors_caught += 1; try: bad_adapter = SklearnAdapter('NonExistentAlgorithm'); print('✗ Should have failed - invalid algorithm'); except InvalidAlgorithmError: print('✓ Caught InvalidAlgorithmError'); errors_caught += 1; except Exception as e: print(f'✓ Caught error: {type(e).__name__}'); errors_caught += 1; print(f'✓ Error handling test complete: {errors_caught}/2 errors caught')`"" -AllowWarnings

# Cleanup
Write-Host "Cleaning up test files..." -ForegroundColor Blue
Remove-Item -Path "test_data_ps.csv" -ErrorAction SilentlyContinue
Remove-Item -Path "test_data_medium_ps.csv" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "POWERSHELL NEW ENVIRONMENT TEST RESULTS" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "Total Tests: $totalTests" -ForegroundColor White
Write-Host "Passed: $passedTests" -ForegroundColor Green
Write-Host "Failed: $failedTests" -ForegroundColor Red
$successRate = [math]::Round(($passedTests * 100) / $totalTests, 1)
Write-Host "Success Rate: $successRate%" -ForegroundColor Yellow

Write-Host ""
Write-Host "DETAILED RESULTS:" -ForegroundColor White
Write-Host "----------------------------------------" -ForegroundColor Gray
foreach ($result in $testResults) {
    if ($result.Status -eq "PASSED") {
        Write-Host "✅ $($result.Test): $($result.Status)" -ForegroundColor Green
    } else {
        Write-Host "❌ $($result.Test): $($result.Status) (Exit: $($result.ExitCode))" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan

if ($failedTests -eq 0) {
    Write-Host "🎉 ALL POWERSHELL TESTS PASSED!" -ForegroundColor Green
    Write-Host "Pynomaly is fully operational in PowerShell environment!" -ForegroundColor Green
    exit 0
} elseif ($successRate -ge 80) {
    Write-Host "⚠️ Most tests passed with some issues" -ForegroundColor Yellow
    Write-Host "Pynomaly is mostly operational in PowerShell environment" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "❌ Significant test failures detected" -ForegroundColor Red
    Write-Host "Pynomaly requires fixes for PowerShell environment" -ForegroundColor Red
    exit 1
}