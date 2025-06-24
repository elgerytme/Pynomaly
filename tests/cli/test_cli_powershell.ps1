# Pynomaly CLI Testing Script for PowerShell Environments
# Tests CLI functionality in fresh Windows PowerShell environments

param(
    [string]$LogLevel = "Info",
    [string]$OutputPath = $null,
    [switch]$SkipPerformance = $false,
    [switch]$Verbose = $false
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$TestDataDir = Join-Path $ScriptDir "test_data"
$TempDir = Join-Path $env:TEMP "pynomaly_cli_test_$(Get-Random)"
$VenvDir = Join-Path $TempDir "test_venv"
$LogFile = Join-Path $TempDir "test_results.log"
$ResultsJson = Join-Path $TempDir "test_results.json"

# Test counters
$TotalTests = 0
$PassedTests = 0
$FailedTests = 0
$TestResults = @()

# Create timestamp function
function Get-Timestamp {
    return Get-Date -Format "yyyy-MM-dd HH:mm:ss"
}

# Logging functions
function Write-Log {
    param([string]$Message, [string]$Level = "Info")
    
    $timestamp = Get-Timestamp
    $logMessage = "[$timestamp] [$Level] $Message"
    
    switch ($Level) {
        "Success" { Write-Host $logMessage -ForegroundColor Green }
        "Error"   { Write-Host $logMessage -ForegroundColor Red }
        "Warning" { Write-Host $logMessage -ForegroundColor Yellow }
        "Info"    { Write-Host $logMessage -ForegroundColor Blue }
        default   { Write-Host $logMessage }
    }
    
    Add-Content -Path $LogFile -Value $logMessage
}

function Write-Success {
    param([string]$Message)
    Write-Log $Message "Success"
    $script:PassedTests++
}

function Write-Error {
    param([string]$Message)
    Write-Log $Message "Error"
    $script:FailedTests++
}

function Write-Warning {
    param([string]$Message)
    Write-Log $Message "Warning"
}

# Test execution wrapper
function Invoke-Test {
    param(
        [string]$TestName,
        [scriptblock]$TestCommand,
        [int]$ExpectedExitCode = 0,
        [switch]$CaptureOutput = $true
    )
    
    $script:TotalTests++
    Write-Log "Running test: $TestName"
    
    $startTime = Get-Date
    $testPassed = $false
    $output = ""
    $exitCode = 0
    
    try {
        if ($CaptureOutput) {
            $output = & $TestCommand 2>&1 | Out-String
            $exitCode = $LASTEXITCODE
        } else {
            & $TestCommand
            $exitCode = $LASTEXITCODE
        }
        
        if ($null -eq $exitCode) { $exitCode = 0 }
        
        if ($exitCode -eq $ExpectedExitCode) {
            $testPassed = $true
        }
    }
    catch {
        $output = $_.Exception.Message
        $exitCode = 1
    }
    
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    $testResult = @{
        name = $TestName
        status = if ($testPassed) { "PASS" } else { "FAIL" }
        duration = [math]::Round($duration, 2)
        exitCode = $exitCode
        output = $output
    }
    
    $script:TestResults += $testResult
    
    if ($testPassed) {
        Write-Success "$TestName ($($duration.ToString('F2'))s)"
    } else {
        Write-Error "$TestName ($($duration.ToString('F2'))s) - Exit code: $exitCode, Expected: $ExpectedExitCode"
        if ($Verbose -and $output) {
            Write-Log "Output: $output" "Error"
        }
    }
    
    return $testPassed
}

# Setup test environment
function Initialize-TestEnvironment {
    Write-Log "Setting up test environment..."
    
    # Create directories
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
    New-Item -ItemType Directory -Path $TestDataDir -Force | Out-Null
    
    # Check Python availability
    try {
        $pythonVersion = python --version 2>&1
        Write-Log "Python version: $pythonVersion"
    }
    catch {
        Write-Error "Python not found. Please install Python 3.11+ and ensure it's in PATH."
        exit 1
    }
    
    # Create virtual environment
    Write-Log "Creating virtual environment at $VenvDir"
    python -m venv $VenvDir
    
    # Activate virtual environment
    $activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        Write-Log "Virtual environment activated"
    } else {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    Write-Log "Test environment setup complete"
}

# Generate test data
function New-TestData {
    Write-Log "Generating test data..."
    
    # Small CSV file
    $smallCsvContent = @"
id,value1,value2,value3,category
1,10.5,20.1,5.0,A
2,11.2,19.8,4.9,A
3,10.8,20.5,5.1,A
4,50.0,80.0,25.0,B
5,9.9,19.5,4.8,A
6,10.3,20.2,5.0,A
7,45.0,75.0,20.0,B
8,10.7,20.0,5.0,A
9,10.1,19.9,4.9,A
10,55.0,85.0,30.0,B
"@
    
    $smallCsvPath = Join-Path $TestDataDir "small_data.csv"
    $smallCsvContent | Out-File -FilePath $smallCsvPath -Encoding UTF8
    
    # JSON file
    $jsonContent = @"
[
    {"id": 1, "value1": 10.5, "value2": 20.1, "value3": 5.0, "category": "A"},
    {"id": 2, "value1": 11.2, "value2": 19.8, "value3": 4.9, "category": "A"},
    {"id": 3, "value1": 10.8, "value2": 20.5, "value3": 5.1, "category": "A"},
    {"id": 4, "value1": 50.0, "value2": 80.0, "value3": 25.0, "category": "B"},
    {"id": 5, "value1": 9.9, "value2": 19.5, "value3": 4.8, "category": "A"}
]
"@
    
    $jsonPath = Join-Path $TestDataDir "sample_data.json"
    $jsonContent | Out-File -FilePath $jsonPath -Encoding UTF8
    
    # Malformed CSV
    $malformedCsvContent = @"
id,value1,value2,value3,category
1,10.5,20.1,5.0,A
2,11.2,19.8,4.9
3,10.8,20.5,5.1,A,extra_column
4,not_a_number,80.0,25.0,B
5,9.9,19.5,4.8,A
"@
    
    $malformedCsvPath = Join-Path $TestDataDir "malformed_data.csv"
    $malformedCsvContent | Out-File -FilePath $malformedCsvPath -Encoding UTF8
    
    # Generate medium CSV with Python
    $pythonScript = @"
import csv
import random
import numpy as np
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Generate data with anomalies
data = []

# Normal data (9000 samples)
for i in range(9000):
    x1 = np.random.normal(10, 2)
    x2 = np.random.normal(20, 3)
    x3 = np.random.normal(5, 1)
    category = random.choice(['A', 'B', 'C'])
    data.append([i+1, x1, x2, x3, category])

# Anomalous data (1000 samples)
for i in range(1000):
    x1 = np.random.normal(50, 5)  # Clear outliers
    x2 = np.random.normal(100, 10)
    x3 = np.random.normal(25, 5)
    category = random.choice(['A', 'B', 'C'])
    data.append([9000+i+1, x1, x2, x3, category])

# Shuffle data
random.shuffle(data)

# Write to CSV
output_path = os.path.join('$($TestDataDir.Replace('\', '\\'))', 'medium_data.csv')
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'value1', 'value2', 'value3', 'category'])
    writer.writerows(data)

print(f'Generated {len(data)} samples in {output_path}')
"@
    
    $pythonScript | python
    
    Write-Log "Test data generation complete"
}

# Install Pynomaly
function Install-Pynomaly {
    Write-Log "Installing Pynomaly..."
    
    # Install from local project
    Set-Location $ProjectRoot
    python -m pip install -e .
    
    # Verify installation
    try {
        $version = pynomaly version 2>&1
        Write-Success "Pynomaly CLI installed successfully"
        Write-Log "Version output: $version"
    }
    catch {
        Write-Error "Pynomaly CLI installation failed or not found in PATH"
        exit 1
    }
}

# Test basic commands
function Test-BasicCommands {
    Write-Log "Testing basic CLI commands..."
    
    Invoke-Test "CLI Help" { pynomaly --help }
    Invoke-Test "CLI Version" { pynomaly version }
    Invoke-Test "Config Show" { pynomaly config --show }
    Invoke-Test "System Status" { pynomaly status }
    
    # Test quickstart with automated input
    Invoke-Test "Quickstart Help" { 
        echo "n" | pynomaly quickstart 
    } -ExpectedExitCode 0
}

# Test dataset commands
function Test-DatasetCommands {
    Write-Log "Testing dataset commands..."
    
    $smallDataPath = Join-Path $TestDataDir "small_data.csv"
    $jsonDataPath = Join-Path $TestDataDir "sample_data.json"
    $malformedDataPath = Join-Path $TestDataDir "malformed_data.csv"
    
    Invoke-Test "Dataset Help" { pynomaly dataset --help }
    Invoke-Test "Dataset List Empty" { pynomaly dataset list }
    
    Invoke-Test "Load CSV Dataset" { 
        pynomaly dataset load $smallDataPath --name test_small 
    }
    
    Invoke-Test "Load JSON Dataset" { 
        pynomaly dataset load $jsonDataPath --name test_json 
    }
    
    Invoke-Test "Dataset List After Loading" { pynomaly dataset list }
    Invoke-Test "Dataset Info" { pynomaly dataset info test_small }
    Invoke-Test "Dataset Validation" { pynomaly dataset validate test_small }
    
    # Test malformed data handling (should fail)
    Invoke-Test "Load Malformed CSV" { 
        pynomaly dataset load $malformedDataPath --name test_malformed 
    } -ExpectedExitCode 1
}

# Test detector commands
function Test-DetectorCommands {
    Write-Log "Testing detector commands..."
    
    Invoke-Test "Detector Help" { pynomaly detector --help }
    Invoke-Test "Detector List Empty" { pynomaly detector list }
    
    Invoke-Test "Create IsolationForest Detector" { 
        pynomaly detector create --name test_detector --algorithm IsolationForest 
    }
    
    Invoke-Test "Detector List After Creation" { pynomaly detector list }
    Invoke-Test "Detector Info" { pynomaly detector info test_detector }
    Invoke-Test "Algorithm List" { pynomaly detector algorithms }
}

# Test detection commands
function Test-DetectionCommands {
    Write-Log "Testing detection commands..."
    
    Invoke-Test "Detection Help" { pynomaly detect --help }
    
    Invoke-Test "Train Detector" { 
        pynomaly detect train --detector test_detector --dataset test_small 
    }
    
    Invoke-Test "Run Detection" { 
        pynomaly detect run --detector test_detector --dataset test_small 
    }
    
    Invoke-Test "View Results" { pynomaly detect results --latest }
    
    Invoke-Test "View Detector Results" { 
        pynomaly detect results --detector test_detector 
    }
}

# Test autonomous commands
function Test-AutonomousCommands {
    Write-Log "Testing autonomous mode commands..."
    
    $smallDataPath = Join-Path $TestDataDir "small_data.csv"
    
    Invoke-Test "Autonomous Help" { pynomaly auto --help }
    
    Invoke-Test "Autonomous Detection" { 
        pynomaly auto detect $smallDataPath --max-algorithms 2 
    }
    
    Invoke-Test "Autonomous Profile" { 
        pynomaly auto profile $smallDataPath 
    }
    
    Invoke-Test "Autonomous Quick" { 
        pynomaly auto quick $smallDataPath --algorithm IsolationForest 
    }
}

# Test export commands
function Test-ExportCommands {
    Write-Log "Testing export commands..."
    
    Invoke-Test "Export Help" { pynomaly export --help }
    Invoke-Test "List Export Formats" { pynomaly export list-formats }
    
    # Create test results file
    $testResultsContent = @"
{
    "anomalies": [
        {"id": 1, "score": 0.8, "is_anomaly": true},
        {"id": 2, "score": 0.2, "is_anomaly": false}
    ],
    "summary": {
        "total_samples": 2,
        "anomaly_count": 1,
        "anomaly_rate": 0.5
    }
}
"@
    
    $testResultsPath = Join-Path $TempDir "test_results.json"
    $testResultsContent | Out-File -FilePath $testResultsPath -Encoding UTF8
    
    $csvExportPath = Join-Path $TempDir "exported_results.csv"
    Invoke-Test "Export to CSV" { 
        pynomaly export csv $testResultsPath $csvExportPath 
    }
    
    # Test Excel export if available
    $excelExportPath = Join-Path $TempDir "exported_results.xlsx"
    try {
        Invoke-Test "Export to Excel" { 
            pynomaly export excel $testResultsPath $excelExportPath 
        }
    }
    catch {
        Write-Warning "Excel export test skipped (dependencies may not be available)"
    }
}

# Test performance
function Test-Performance {
    if ($SkipPerformance) {
        Write-Log "Skipping performance tests (--SkipPerformance specified)"
        return
    }
    
    Write-Log "Testing performance with medium dataset..."
    
    $mediumDataPath = Join-Path $TestDataDir "medium_data.csv"
    
    if (-not (Test-Path $mediumDataPath)) {
        Write-Warning "Medium dataset not found, skipping performance tests"
        return
    }
    
    Invoke-Test "Load Medium Dataset" { 
        pynomaly dataset load $mediumDataPath --name test_medium 
    }
    
    Invoke-Test "Create Detector for Medium Data" { 
        pynomaly detector create --name medium_detector --algorithm IsolationForest 
    }
    
    $trainingStart = Get-Date
    Invoke-Test "Train on Medium Dataset" { 
        pynomaly detect train --detector medium_detector --dataset test_medium 
    }
    $trainingEnd = Get-Date
    $trainingDuration = ($trainingEnd - $trainingStart).TotalSeconds
    
    Write-Log "Training duration: $($trainingDuration.ToString('F2'))s"
    
    $detectionStart = Get-Date
    Invoke-Test "Detect on Medium Dataset" { 
        pynomaly detect run --detector medium_detector --dataset test_medium 
    }
    $detectionEnd = Get-Date
    $detectionDuration = ($detectionEnd - $detectionStart).TotalSeconds
    
    Write-Log "Detection duration: $($detectionDuration.ToString('F2'))s"
    
    # Performance assertions
    if ($trainingDuration -gt 120) {
        Write-Warning "Training took longer than expected: $($trainingDuration.ToString('F2'))s"
    }
    
    if ($detectionDuration -gt 60) {
        Write-Warning "Detection took longer than expected: $($detectionDuration.ToString('F2'))s"
    }
}

# Test error handling
function Test-ErrorHandling {
    Write-Log "Testing error handling..."
    
    # Test non-existent dataset
    Invoke-Test "Load Non-existent File" { 
        pynomaly dataset load "C:\NonExistent\file.csv" --name test_missing 
    } -ExpectedExitCode 1
    
    # Test invalid algorithm
    Invoke-Test "Create Invalid Algorithm Detector" { 
        pynomaly detector create --name bad_detector --algorithm NonExistentAlgorithm 
    } -ExpectedExitCode 1
    
    # Test train with non-existent detector
    Invoke-Test "Train Non-existent Detector" { 
        pynomaly detect train --detector non_existent --dataset test_small 
    } -ExpectedExitCode 1
    
    # Test train with non-existent dataset
    Invoke-Test "Train with Non-existent Dataset" { 
        pynomaly detect train --detector test_detector --dataset non_existent 
    } -ExpectedExitCode 1
}

# Cleanup function
function Remove-TestEnvironment {
    Write-Log "Cleaning up test environment..."
    
    try {
        # Deactivate virtual environment
        if ($env:VIRTUAL_ENV) {
            deactivate
        }
        
        # Remove temporary directory
        if (Test-Path $TempDir) {
            Remove-Item -Recurse -Force $TempDir
        }
        
        Write-Log "Cleanup complete"
    }
    catch {
        Write-Warning "Cleanup encountered errors: $($_.Exception.Message)"
    }
}

# Generate test report
function New-TestReport {
    Write-Log "Generating test report..."
    
    $successRate = if ($TotalTests -gt 0) { 
        [math]::Round(($PassedTests / $TotalTests) * 100, 2) 
    } else { 
        0 
    }
    
    $report = @{
        test_run = @{
            timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
            environment = "powershell"
            platform = "$($env:OS) - $(Get-ComputerInfo | Select-Object -ExpandProperty WindowsProductName)"
            powershell_version = $PSVersionTable.PSVersion.ToString()
            python_version = (python --version 2>&1)
            total_tests = $TotalTests
            passed_tests = $PassedTests
            failed_tests = $FailedTests
            success_rate = $successRate
        }
        test_results = $TestResults
    }
    
    $report | ConvertTo-Json -Depth 5 | Out-File -FilePath $ResultsJson -Encoding UTF8
    
    # Print summary
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "         PYNOMALY CLI TEST SUMMARY" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "Total Tests:    $TotalTests" -ForegroundColor White
    Write-Host "Passed:         $PassedTests" -ForegroundColor Green
    Write-Host "Failed:         $FailedTests" -ForegroundColor Red
    Write-Host "Success Rate:   $successRate%" -ForegroundColor Yellow
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host ""
    
    if ($FailedTests -eq 0) {
        Write-Host "🎉 All tests passed!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Some tests failed. Check the log for details." -ForegroundColor Red
        Write-Host "Log file: $LogFile" -ForegroundColor Yellow
        Write-Host "Results file: $ResultsJson" -ForegroundColor Yellow
        return $false
    }
}

# Main execution function
function Main {
    try {
        Write-Log "Starting Pynomaly CLI tests in PowerShell environment"
        Write-Log "Platform: $($env:OS) - $(Get-ComputerInfo | Select-Object -ExpandProperty WindowsProductName)"
        Write-Log "PowerShell: $($PSVersionTable.PSVersion)"
        Write-Log "Python: $(python --version 2>&1)"
        
        # Setup
        Initialize-TestEnvironment
        New-TestData
        Install-Pynomaly
        
        # Run tests
        Test-BasicCommands
        Test-DatasetCommands
        Test-DetectorCommands
        Test-DetectionCommands
        Test-AutonomousCommands
        Test-ExportCommands
        Test-Performance
        Test-ErrorHandling
        
        # Generate report
        $success = New-TestReport
        
        if ($OutputPath) {
            Copy-Item $ResultsJson $OutputPath -Force
            Write-Log "Results copied to: $OutputPath"
        }
        
        return $success
    }
    catch {
        Write-Error "Test execution failed: $($_.Exception.Message)"
        Write-Error $_.ScriptStackTrace
        return $false
    }
    finally {
        Remove-TestEnvironment
    }
}

# Execute main function
$testSuccess = Main

# Exit with appropriate code
if ($testSuccess) {
    exit 0
} else {
    exit 1
}