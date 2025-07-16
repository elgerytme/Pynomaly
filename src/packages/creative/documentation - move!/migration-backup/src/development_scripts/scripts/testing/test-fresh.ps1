# Test script for fresh PowerShell environment

Write-Host "=== Testing Pynomaly Web App in Fresh PowerShell Environment ===" -ForegroundColor Cyan
Write-Host "Date: $(Get-Date)"
Write-Host "Current directory: $(Get-Location)"
Write-Host "Python version: $(python --version 2>$null)"
Write-Host "PowerShell version: $($PSVersionTable.PSVersion)"
Write-Host ""

# Change to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

Write-Host "Project root: $(Get-Location)"
Write-Host ""

# Create temporary test environment
$testEnvDir = "test_environments\fresh_powershell_test"
Write-Host "Creating fresh test environment in: $testEnvDir"

# Clean up any existing test environment
if (Test-Path $testEnvDir) {
    Remove-Item -Recurse -Force $testEnvDir
}

New-Item -ItemType Directory -Path $testEnvDir -Force | Out-Null
Set-Location $testEnvDir

Write-Host "✓ Test environment created" -ForegroundColor Green
Write-Host ""

# Test 1: Create virtual environment (if possible)
Write-Host "Test 1: Creating virtual environment..." -ForegroundColor Yellow
try {
    python -m venv fresh_venv 2>$null
    if (Test-Path "fresh_venv\Scripts\Activate.ps1") {
        & "fresh_venv\Scripts\Activate.ps1"
        Write-Host "✓ Virtual environment activated" -ForegroundColor Green
        Write-Host "Python path: $(Get-Command python | Select-Object -ExpandProperty Source)"
        $usingVenv = $true
    } else {
        Write-Host "⚠️  Virtual environment creation failed, using system Python" -ForegroundColor Yellow
        $usingVenv = $false
    }
} catch {
    Write-Host "⚠️  Virtual environment creation failed, using system Python" -ForegroundColor Yellow
    $usingVenv = $false
}
Write-Host ""

# Test 2: Install minimal dependencies (if in virtual environment)
if ($usingVenv) {
    Write-Host "Test 2: Installing minimal dependencies..." -ForegroundColor Yellow
    try {
        python -m pip install --quiet fastapi uvicorn pydantic dependency-injector pandas numpy scikit-learn
        Write-Host "✓ Dependencies installed" -ForegroundColor Green
    } catch {
        Write-Host "❌ Dependency installation failed: $_" -ForegroundColor Red
        if ($usingVenv) { deactivate }
        Set-Location $projectRoot
        Remove-Item -Recurse -Force $testEnvDir
        exit 1
    }
} else {
    Write-Host "Test 2: Skipping dependency installation (using system Python)" -ForegroundColor Yellow
}
Write-Host ""

# Test 3: Copy source code to fresh environment
Write-Host "Test 3: Setting up source code..." -ForegroundColor Yellow
Copy-Item -Recurse -Path "..\..\src" -Destination "."
Copy-Item -Recurse -Path "..\..\scripts" -Destination "."
Write-Host "✓ Source code copied" -ForegroundColor Green
Write-Host ""

# Test 4: Test imports in fresh environment
Write-Host "Test 4: Testing imports in fresh environment..." -ForegroundColor Yellow
$env:PYTHONPATH = "$(Get-Location)\src"
$importResult = python -c @"
try:
    # Test basic Python functionality first
    import sys, os
    print('✓ Basic Python imports working')

    # Test if required packages are available
    required_packages = ['fastapi', 'uvicorn', 'pydantic', 'dependency_injector', 'pandas', 'numpy', 'sklearn']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f'✓ {package} available')
        except ImportError:
            missing_packages.append(package)
            print(f'✗ {package} not available')

    if missing_packages:
        print(f'⚠️  Missing packages: {missing_packages}')
        print('⚠️  Cannot test full functionality, but testing core imports...')

    # Test pynomaly imports
    from pynomaly.presentation.web.app import create_web_app
    print('✓ Pynomaly import successful in fresh environment')

except Exception as e:
    print('✗ Import failed in fresh environment:', e)
    import traceback
    traceback.print_exc()
    exit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Import test failed in fresh environment" -ForegroundColor Red
    if ($usingVenv) { deactivate }
    Set-Location $projectRoot
    Remove-Item -Recurse -Force $testEnvDir
    exit 1
}
Write-Host ""

# Test 5: App creation in fresh environment
Write-Host "Test 5: Testing app creation in fresh environment..." -ForegroundColor Yellow
$appResult = python -c @"
try:
    from pynomaly.presentation.web.app import create_web_app
    app = create_web_app()
    print('✓ App creation successful in fresh environment')
    print('✓ Routes count:', len(app.routes))

    # Test that both API and web routes are present
    api_routes = [r for r in app.routes if str(r.path).startswith('/api')]
    web_routes = [r for r in app.routes if str(r.path).startswith('/web')]

    print(f'✓ API routes: {len(api_routes)}')
    print(f'✓ Web routes: {len(web_routes)}')

    if len(api_routes) > 0 and len(web_routes) > 0:
        print('✓ Both API and Web UI routes present')
    else:
        print('✗ Missing API or Web UI routes')
        exit(1)

except Exception as e:
    print('✗ App creation failed in fresh environment:', e)
    import traceback
    traceback.print_exc()
    exit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ App creation test failed in fresh environment" -ForegroundColor Red
    if ($usingVenv) { deactivate }
    Set-Location $projectRoot
    Remove-Item -Recurse -Force $testEnvDir
    exit 1
}
Write-Host ""

# Test 6: Check that required files are present
Write-Host "Test 6: Verifying file structure in fresh environment..." -ForegroundColor Yellow
$requiredFiles = @(
    "src\pynomaly\__init__.py",
    "src\pynomaly\presentation\api\app.py",
    "src\pynomaly\presentation\web\app.py",
    "src\pynomaly\infrastructure\config\container.py",
    "scripts\run_web_app.py"
)

$allFilesPresent = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✓ $file present" -ForegroundColor Green
    } else {
        Write-Host "✗ $file missing" -ForegroundColor Red
        $allFilesPresent = $false
    }
}

if ($allFilesPresent) {
    Write-Host "✓ All required files present" -ForegroundColor Green
} else {
    Write-Host "❌ Some required files missing" -ForegroundColor Red
    if ($usingVenv) { deactivate }
    Set-Location $projectRoot
    Remove-Item -Recurse -Force $testEnvDir
    exit 1
}
Write-Host ""

# Test 7: Server startup test
Write-Host "Test 7: Testing server startup in fresh environment..." -ForegroundColor Yellow
$serverJob = Start-Job -ScriptBlock {
    Set-Location $using:testEnvDir
    $env:PYTHONPATH = "$(Get-Location)\src"
    python scripts\run_web_app.py
}

Write-Host "✓ Server started in fresh environment with Job ID: $($serverJob.Id)" -ForegroundColor Green

# Wait for server to start
Start-Sleep -Seconds 10

# Test API endpoint
Write-Host "Testing API endpoint in fresh environment..."
try {
    $apiResponse = Invoke-RestMethod -Uri "http://localhost:8000/" -TimeoutSec 15
    if ($apiResponse.message -eq "Pynomaly API") {
        Write-Host "✓ API endpoint working in fresh environment" -ForegroundColor Green
    } else {
        Write-Host "✗ API endpoint failed - unexpected response" -ForegroundColor Red
        Write-Host "Response: $apiResponse"
        Stop-Job $serverJob
        Remove-Job $serverJob
        if ($usingVenv) { deactivate }
        Set-Location $projectRoot
        Remove-Item -Recurse -Force $testEnvDir
        exit 1
    }
} catch {
    Write-Host "✗ API endpoint failed - connection error: $_" -ForegroundColor Red
    Stop-Job $serverJob
    Remove-Job $serverJob
    if ($usingVenv) { deactivate }
    Set-Location $projectRoot
    Remove-Item -Recurse -Force $testEnvDir
    exit 1
}

# Test Web UI endpoint
Write-Host "Testing Web UI endpoint in fresh environment..."
try {
    $webResponse = Invoke-WebRequest -Uri "http://localhost:8000/web/" -TimeoutSec 15
    if ($webResponse.Content -like "*Dashboard - Pynomaly*") {
        Write-Host "✓ Web UI endpoint working in fresh environment" -ForegroundColor Green
    } else {
        Write-Host "✗ Web UI endpoint failed - unexpected content" -ForegroundColor Red
        Write-Host "Content snippet: $($webResponse.Content.Substring(0, [Math]::Min(200, $webResponse.Content.Length)))"
        Stop-Job $serverJob
        Remove-Job $serverJob
        if ($usingVenv) { deactivate }
        Set-Location $projectRoot
        Remove-Item -Recurse -Force $testEnvDir
        exit 1
    }
} catch {
    Write-Host "✗ Web UI endpoint failed - connection error: $_" -ForegroundColor Red
    Stop-Job $serverJob
    Remove-Job $serverJob
    if ($usingVenv) { deactivate }
    Set-Location $projectRoot
    Remove-Item -Recurse -Force $testEnvDir
    exit 1
}

# Stop server
Stop-Job $serverJob
Remove-Job $serverJob
Start-Sleep -Seconds 2
Write-Host "✓ Server stopped" -ForegroundColor Green

# Deactivate virtual environment if used
if ($usingVenv) {
    deactivate
    Write-Host "✓ Virtual environment deactivated" -ForegroundColor Green
}

# Return to project root
Set-Location $projectRoot

# Clean up test environment
Write-Host ""
Write-Host "Cleaning up test environment..."
Remove-Item -Recurse -Force $testEnvDir
Write-Host "✓ Test environment cleaned up" -ForegroundColor Green

Write-Host ""
Write-Host "🎉 All fresh environment tests passed! Pynomaly web app works correctly in fresh PowerShell environment." -ForegroundColor Green
Write-Host "✓ Virtual environment setup (or graceful fallback)" -ForegroundColor Green
Write-Host "✓ Dependency management" -ForegroundColor Green
Write-Host "✓ Source code setup in fresh location" -ForegroundColor Green
Write-Host "✓ File structure verification" -ForegroundColor Green
Write-Host "✓ Python imports working" -ForegroundColor Green
Write-Host "✓ App creation working" -ForegroundColor Green
Write-Host "✓ Server startup working" -ForegroundColor Green
Write-Host "✓ API endpoints working" -ForegroundColor Green
Write-Host "✓ Web UI working" -ForegroundColor Green
Write-Host "✓ Environment cleanup" -ForegroundColor Green
