# Test runner for current environment (PowerShell)
# Usage: .\scripts\test-current.ps1 [options]

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
    [string]$PytestArgs = ""
)

# Enable strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Configuration
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TestDir = Join-Path $ProjectRoot "tests"
$SrcDir = Join-Path $ProjectRoot "src"
$ReportsDir = Join-Path $ProjectRoot "test-reports"
$CoverageDir = Join-Path $ReportsDir "coverage"

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
Usage: .\scripts\test-current.ps1 [OPTIONS]

Run tests in the current environment

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
    
EXAMPLES:
    .\scripts\test-current.ps1                          # Run all tests with default settings
    .\scripts\test-current.ps1 -Verbose -NoCoverage     # Verbose mode without coverage
    .\scripts\test-current.ps1 -Fast                    # Run only fast tests
    .\scripts\test-current.ps1 -UnitOnly                # Run only unit tests
    .\scripts\test-current.ps1 -Markers "not slow"      # Run tests not marked as slow
    .\scripts\test-current.ps1 -PytestArgs "-x -s"      # Pass additional pytest arguments

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
    if (Test-Command "python3") {
        return "python3"
    }
    elseif (Test-Command "python") {
        return "python"
    }
    elseif (Test-Command "py") {
        return "py"
    }
    else {
        Write-ColorOutput "Error: Python not found" $Colors.Red
        exit 1
    }
}

# Function to check environment
function Test-Environment {
    Write-ColorOutput "🔍 Checking test environment..." $Colors.Blue
    
    # Check if we're in the project root
    if (-not (Test-Path (Join-Path $ProjectRoot "pyproject.toml"))) {
        Write-ColorOutput "Error: Not in project root directory" $Colors.Red
        exit 1
    }
    
    # Detect Python
    $PythonCmd = Get-PythonCommand
    Write-ColorOutput "✓ Python found: $PythonCmd" $Colors.Green
    
    # Check if pytest is available
    try {
        $null = & $PythonCmd -m pytest --version 2>$null
        Write-ColorOutput "✓ pytest available" $Colors.Green
    }
    catch {
        Write-ColorOutput "Error: pytest not available. Install with: pip install pytest" $Colors.Red
        exit 1
    }
    
    # Check if we're in a virtual environment
    if ($env:VIRTUAL_ENV -or $env:CONDA_DEFAULT_ENV -or $PythonCmd -like "*\.venv*") {
        Write-ColorOutput "✓ Virtual environment detected" $Colors.Green
    }
    else {
        Write-ColorOutput "⚠ Warning: Not in a virtual environment" $Colors.Yellow
    }
    
    # Create reports directory
    if (-not (Test-Path $ReportsDir)) {
        New-Item -ItemType Directory -Path $ReportsDir -Force | Out-Null
    }
    if (-not (Test-Path $CoverageDir)) {
        New-Item -ItemType Directory -Path $CoverageDir -Force | Out-Null
    }
    Write-ColorOutput "✓ Test reports directory: $ReportsDir" $Colors.Green
    
    return $PythonCmd
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
    
    Write-ColorOutput "🧪 Running tests in current environment..." $Colors.Blue
    
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

# Function to show summary
function Show-Summary {
    Write-ColorOutput "📋 Test Summary" $Colors.Blue
    Write-Host "=================="
    Write-Host "Project: Pynomaly"
    Write-Host "Environment: Current"
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
    
    if ($Markers) {
        Write-Host "Custom Markers: $Markers"
    }
    
    Write-Host ""
}

# Main execution
function Main {
    Write-ColorOutput "🚀 Pynomaly Test Runner (Current Environment)" $Colors.Blue
    Write-ColorOutput "==============================================" $Colors.Blue
    Write-Host ""
    
    try {
        $PythonCmd = Test-Environment
        Show-Summary
        
        if (Invoke-Tests -PythonCmd $PythonCmd) {
            Write-ColorOutput "🎉 All tests passed!" $Colors.Green
            exit 0
        }
        else {
            Write-ColorOutput "💥 Some tests failed!" $Colors.Red
            exit 1
        }
    }
    catch {
        Write-ColorOutput "🛑 Test execution interrupted: $($_.Exception.Message)" $Colors.Yellow
        exit 1
    }
}

# Run main function
Main