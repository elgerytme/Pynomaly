# Test runner for fresh environment (PowerShell)
# Usage: .\scripts\test-fresh.ps1 [options]

param(
    [switch]$Help,
    [switch]$Verbose,
    [switch]$Quiet,
    [switch]$Coverage = $true,
    [switch]$NoCoverage,
    [switch]$Parallel = $true,
    [switch]$NoParallel,
    [switch]$Fast,
    [switch]$UnitOnly,
    [switch]$IntegrationOnly,
    [switch]$Performance,
    [switch]$Security,
    [switch]$FailFast,
    [string]$Markers = "",
    [string]$PytestArgs = "",
    [string]$PythonVersion = "3.11",
    [switch]$CleanVenv,
    [switch]$KeepVenv
)

# Enable strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Configuration
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TestDir = Join-Path $ProjectRoot "tests"
$SrcDir = Join-Path $ProjectRoot "src"
$ReportsDir = Join-Path $ProjectRoot "test-reports-fresh"
$CoverageDir = Join-Path $ReportsDir "coverage"
$VenvDir = Join-Path $ProjectRoot ".test-venv"

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    Cyan = "Cyan"
    Magenta = "Magenta"
}

# Function to print colored output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Function to show usage
function Show-Usage {
    @"
Usage: .\scripts\test-fresh.ps1 [OPTIONS]

Run tests in a fresh virtual environment

OPTIONS:
    -Help                   Show this help message
    -Verbose                Enable verbose output
    -Quiet                  Disable verbose output (opposite of -Verbose)
    -Coverage               Enable coverage reporting (default: enabled)
    -NoCoverage             Disable coverage reporting
    -Parallel               Run tests in parallel (default: enabled)
    -NoParallel             Disable parallel execution
    -Fast                   Run only fast tests (skip slow integration tests)
    -UnitOnly               Run only unit tests
    -IntegrationOnly        Run only integration tests
    -Performance            Include performance tests
    -Security               Include security tests
    -FailFast               Stop on first failure
    -Markers <string>       Run tests with specific pytest markers
    -PytestArgs <string>    Additional arguments to pass to pytest
    -PythonVersion <string> Python version to use (default: 3.11)
    -CleanVenv              Remove existing test environment before creating new one
    -KeepVenv               Keep the test environment after tests complete
    
EXAMPLES:
    .\scripts\test-fresh.ps1                           # Run all tests in fresh environment
    .\scripts\test-fresh.ps1 -Verbose -CleanVenv       # Verbose mode with clean environment
    .\scripts\test-fresh.ps1 -Fast -KeepVenv           # Run only fast tests and keep environment
    .\scripts\test-fresh.ps1 -PythonVersion "3.12"     # Use Python 3.12
    .\scripts\test-fresh.ps1 -UnitOnly                 # Run only unit tests in fresh environment

"@ | Write-Host
}

# Handle help parameter
if ($Help) {
    Show-Usage
    exit 0
}

# Handle conflicting parameters
if ($NoCoverage) { $Coverage = $false }
if ($NoParallel) { $Parallel = $false }
if ($Quiet) { $Verbose = $false }

# Determine test scope
$Unit = $true
$Integration = $true

if ($UnitOnly) {
    $Unit = $true
    $Integration = $false
}
elseif ($IntegrationOnly) {
    $Unit = $false
    $Integration = $true
}

# Function to test if command exists
function Test-Command {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Function to detect Python command
function Get-PythonCommand {
    $PythonCandidates = @(
        "python$PythonVersion",
        "python3.$PythonVersion",
        "python3",
        "python",
        "py"
    )
    
    foreach ($cmd in $PythonCandidates) {
        if (Test-Command $cmd) {
            try {
                $version = & $cmd --version 2>&1 | Select-String -Pattern '\d+\.\d+' | ForEach-Object { $_.Matches[0].Value }
                if ($version) {
                    Write-ColorOutput "✓ Found Python $version : $cmd" $Colors.Green
                    return $cmd
                }
            }
            catch {
                continue
            }
        }
    }
    
    Write-ColorOutput "Error: Python $PythonVersion not found" $Colors.Red
    exit 1
}

# Function to setup fresh virtual environment
function New-FreshEnvironment {
    Write-ColorOutput "🏗️ Setting up fresh virtual environment..." $Colors.Blue
    
    # Clean existing environment if requested
    if ($CleanVenv -and (Test-Path $VenvDir)) {
        Write-ColorOutput "🧹 Removing existing test environment..." $Colors.Yellow
        Remove-Item -Recurse -Force $VenvDir
    }
    
    # Detect Python
    $PythonCmd = Get-PythonCommand
    
    # Create virtual environment
    if (-not (Test-Path $VenvDir)) {
        Write-ColorOutput "📦 Creating virtual environment with $PythonCmd..." $Colors.Blue
        & $PythonCmd -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Error: Failed to create virtual environment" $Colors.Red
            exit 1
        }
    }
    else {
        Write-ColorOutput "✓ Using existing virtual environment" $Colors.Green
    }
    
    # Activate virtual environment
    $ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (-not (Test-Path $ActivateScript)) {
        $ActivateScript = Join-Path $VenvDir "bin\activate"  # Unix-style
    }
    
    if (Test-Path $ActivateScript) {
        & $ActivateScript
        Write-ColorOutput "✓ Virtual environment activated: $VenvDir" $Colors.Green
    }
    else {
        Write-ColorOutput "Error: Could not find activation script" $Colors.Red
        exit 1
    }
    
    # Use python from virtual environment
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        $VenvPython = Join-Path $VenvDir "bin\python"  # Unix-style
    }
    
    # Upgrade pip
    Write-ColorOutput "⬆️ Upgrading pip..." $Colors.Blue
    & $VenvPython -m pip install --upgrade pip wheel setuptools
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Warning: Failed to upgrade pip" $Colors.Yellow
    }
    
    # Install project dependencies
    Write-ColorOutput "📚 Installing project dependencies..." $Colors.Blue
    
    # Change to project root
    Push-Location $ProjectRoot
    
    try {
        # Install from pyproject.toml if available
        if (Test-Path (Join-Path $ProjectRoot "pyproject.toml")) {
            # Install project in development mode with all extras
            & $VenvPython -m pip install -e ".[dev,test,performance,security]"
            if ($LASTEXITCODE -ne 0) {
                Write-ColorOutput "Warning: Failed to install with extras, trying basic install" $Colors.Yellow
                & $VenvPython -m pip install -e .
            }
        }
        else {
            # Fallback to requirements files
            $RequirementsFiles = @(
                "requirements.txt",
                "requirements-dev.txt",
                "requirements-test.txt"
            )
            
            foreach ($reqFile in $RequirementsFiles) {
                $reqPath = Join-Path $ProjectRoot $reqFile
                if (Test-Path $reqPath) {
                    Write-ColorOutput "Installing from $reqFile..." $Colors.Blue
                    & $VenvPython -m pip install -r $reqPath
                }
            }
        }
        
        # Install additional test dependencies
        Write-ColorOutput "🧪 Installing test dependencies..." $Colors.Blue
        $TestDependencies = @(
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "pytest-mock",
            "pytest-asyncio",
            "pytest-benchmark",
            "pytest-security",
            "hypothesis"
        )
        
        & $VenvPython -m pip install $TestDependencies
        
        # Install performance testing tools if needed
        if ($Performance) {
            & $VenvPython -m pip install pytest-benchmark memory-profiler
        }
        
        # Install security testing tools if needed
        if ($Security) {
            & $VenvPython -m pip install bandit safety
        }
        
        Write-ColorOutput "✅ Fresh environment setup complete" $Colors.Green
        
        # Show installed packages
        if ($Verbose) {
            Write-ColorOutput "📋 Installed packages:" $Colors.Cyan
            & $VenvPython -m pip list
        }
        
        return $VenvPython
    }
    finally {
        Pop-Location
    }
}

# Function to build pytest command
function Build-PytestCommand {
    param([string]$PythonCmd)
    
    $cmd = @($PythonCmd, "-m", "pytest")
    
    # Base test directory
    if ($Unit -and -not $Integration) {
        $cmd += Join-Path $TestDir "unit"
        $cmd += Join-Path $TestDir "domain"
        $cmd += Join-Path $TestDir "application"
    }
    elseif ($Integration -and -not $Unit) {
        $cmd += Join-Path $TestDir "integration"
        $cmd += Join-Path $TestDir "infrastructure"
    }
    else {
        $cmd += $TestDir
    }
    
    # Verbose output
    if ($Verbose) {
        $cmd += "-v"
    }
    else {
        $cmd += "-q"
    }
    
    # Coverage options
    if ($Coverage) {
        $cmd += "--cov=pynomaly"
        $cmd += "--cov-report=html:$CoverageDir\html"
        $cmd += "--cov-report=xml:$CoverageDir\coverage.xml"
        $cmd += "--cov-report=term"
    }
    
    # Parallel execution
    if ($Parallel -and (Test-Command "pytest-xdist")) {
        $NumCores = (Get-CimInstance -ClassName Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
        $cmd += "-n"
        $cmd += $NumCores.ToString()
    }
    
    # Fast mode (skip slow tests)
    if ($Fast) {
        $cmd += "-m"
        $cmd += "not slow and not integration"
    }
    
    # Performance tests
    if ($Performance) {
        $cmd += "-m"
        $cmd += "performance"
    }
    
    # Security tests
    if ($Security) {
        $cmd += "-m"
        $cmd += "security"
    }
    
    # Fail fast
    if ($FailFast) {
        $cmd += "-x"
    }
    
    # Custom markers
    if ($Markers) {
        $cmd += "-m"
        $cmd += $Markers
    }
    
    # Additional pytest arguments
    if ($PytestArgs) {
        $cmd += $PytestArgs.Split(' ')
    }
    
    # Output options
    $cmd += "--tb=short"
    $cmd += "--strict-markers"
    
    # JUnit XML for CI
    $cmd += "--junitxml=$ReportsDir\junit.xml"
    
    return $cmd
}

# Function to run tests
function Invoke-Tests {
    param([string]$PythonCmd)
    
    Write-ColorOutput "🧪 Running tests in fresh environment..." $Colors.Blue
    
    # Create reports directory
    if (-not (Test-Path $ReportsDir)) {
        New-Item -ItemType Directory -Path $ReportsDir -Force | Out-Null
    }
    if (-not (Test-Path $CoverageDir)) {
        New-Item -ItemType Directory -Path $CoverageDir -Force | Out-Null
    }
    
    $PytestCmd = Build-PytestCommand -PythonCmd $PythonCmd
    $CmdString = $PytestCmd -join " "
    
    Write-ColorOutput "Command: $CmdString" $Colors.Yellow
    Write-Host ""
    
    # Change to project root
    Push-Location $ProjectRoot
    
    try {
        # Run tests
        $StartTime = Get-Date
        
        $Process = Start-Process -FilePath $PytestCmd[0] -ArgumentList $PytestCmd[1..($PytestCmd.Length-1)] -Wait -PassThru -NoNewWindow
        
        $EndTime = Get-Date
        $Duration = ($EndTime - $StartTime).TotalSeconds
        
        if ($Process.ExitCode -eq 0) {
            Write-ColorOutput "✅ Tests completed successfully in $([math]::Round($Duration, 2))s" $Colors.Green
            
            # Show coverage summary if enabled
            if ($Coverage) {
                Write-ColorOutput "📊 Coverage report generated: $CoverageDir\html\index.html" $Colors.Blue
            }
            
            return $true
        }
        else {
            Write-ColorOutput "❌ Tests failed after $([math]::Round($Duration, 2))s" $Colors.Red
            return $false
        }
    }
    finally {
        Pop-Location
    }
}

# Function to cleanup environment
function Remove-TestEnvironment {
    if (-not $KeepVenv) {
        Write-ColorOutput "🧹 Cleaning up test environment..." $Colors.Blue
        
        # Deactivate virtual environment
        if ($env:VIRTUAL_ENV) {
            deactivate
        }
        
        # Remove virtual environment
        if (Test-Path $VenvDir) {
            Remove-Item -Recurse -Force $VenvDir
            Write-ColorOutput "✓ Test environment cleaned up" $Colors.Green
        }
    }
    else {
        Write-ColorOutput "📦 Test environment preserved at: $VenvDir" $Colors.Yellow
        $ActivateCmd = Join-Path $VenvDir "Scripts\Activate.ps1"
        Write-ColorOutput "To reuse: & `"$ActivateCmd`"" $Colors.Cyan
    }
}

# Function to show summary
function Show-Summary {
    Write-ColorOutput "📋 Test Summary" $Colors.Blue
    Write-Host "=================="
    Write-Host "Project: Pynomaly"
    Write-Host "Environment: Fresh Virtual Environment"
    Write-Host "Python Version: $PythonVersion"
    Write-Host "Virtual Environment: $VenvDir"
    Write-Host "Test Directory: $TestDir"
    Write-Host "Reports Directory: $ReportsDir"
    
    if ($Coverage) {
        Write-Host "Coverage: Enabled"
        Write-Host "Coverage Reports: $CoverageDir\"
    }
    else {
        Write-Host "Coverage: Disabled"
    }
    
    if ($Parallel) {
        Write-Host "Parallel Execution: Enabled"
    }
    else {
        Write-Host "Parallel Execution: Disabled"
    }
    
    Write-Host "Fast Mode: $Fast"
    Write-Host "Unit Tests: $Unit"
    Write-Host "Integration Tests: $Integration"
    Write-Host "Performance Tests: $Performance"
    Write-Host "Security Tests: $Security"
    Write-Host "Clean Environment: $CleanVenv"
    Write-Host "Keep Environment: $KeepVenv"
    
    if ($Markers) {
        Write-Host "Custom Markers: $Markers"
    }
    
    Write-Host ""
}

# Main execution
function Main {
    Write-ColorOutput "🚀 Pynomaly Test Runner (Fresh Environment)" $Colors.Blue
    Write-ColorOutput "===========================================" $Colors.Blue
    Write-Host ""
    
    try {
        # Check if we're in the project root
        if (-not (Test-Path (Join-Path $ProjectRoot "pyproject.toml"))) {
            Write-ColorOutput "Error: Not in project root directory" $Colors.Red
            exit 1
        }
        
        Show-Summary
        
        $PythonCmd = New-FreshEnvironment
        
        $TestResult = $false
        if (Invoke-Tests -PythonCmd $PythonCmd) {
            Write-ColorOutput "🎉 All tests passed in fresh environment!" $Colors.Green
            $TestResult = $true
        }
        else {
            Write-ColorOutput "💥 Some tests failed in fresh environment!" $Colors.Red
            $TestResult = $false
        }
        
        Remove-TestEnvironment
        
        if ($TestResult) {
            exit 0
        }
        else {
            exit 1
        }
    }
    catch {
        Write-ColorOutput "🛑 Test execution interrupted: $($_.Exception.Message)" $Colors.Yellow
        Remove-TestEnvironment
        exit 1
    }
}

# Run main function
Main