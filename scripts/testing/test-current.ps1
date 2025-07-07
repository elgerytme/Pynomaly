# Test script for current PowerShell environment

Write-Host "=== Testing Pynomaly Web App in Current PowerShell Environment ===" -ForegroundColor Cyan
Write-Host "Date: $(Get-Date)"
Write-Host "Current directory: $(Get-Location)"
Write-Host "Python version: $(python3 --version 2>$null)"
Write-Host "PowerShell version: $($PSVersionTable.PSVersion)"
Write-Host ""

# Change to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

Write-Host "Project root: $(Get-Location)"
Write-Host ""

# Test 1: Import test
Write-Host "Test 1: Testing Python imports..." -ForegroundColor Yellow
$env:PYTHONPATH = "$(Get-Location)\src"
$importResult = python3 -c @"
try:
    from pynomaly.presentation.web.app import create_web_app
    print('✓ Import successful')
except Exception as e:
    print('✗ Import failed:', e)
    exit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Import test failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 2: App creation test
Write-Host "Test 2: Testing app creation..." -ForegroundColor Yellow
$appResult = python3 -c @"
try:
    from pynomaly.presentation.web.app import create_web_app
    app = create_web_app()
    print('✓ App creation successful')
    print('✓ Routes count:', len(app.routes))
except Exception as e:
    print('✗ App creation failed:', e)
    exit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ App creation test failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 3: Server startup test
Write-Host "Test 3: Testing server startup..." -ForegroundColor Yellow
$serverJob = Start-Job -ScriptBlock {
    Set-Location $using:projectRoot
    $env:PYTHONPATH = "$using:projectRoot\src"
    python3 scripts/run_web_app.py
}

Write-Host "✓ Server started with Job ID: $($serverJob.Id)"

# Wait for server to start
Start-Sleep -Seconds 8

# Test API endpoint
Write-Host "Testing API endpoint..."
try {
    $apiResponse = Invoke-RestMethod -Uri "http://localhost:8000/" -TimeoutSec 10
    if ($apiResponse.message -eq "Pynomaly API") {
        Write-Host "✓ API endpoint working" -ForegroundColor Green
    } else {
        Write-Host "✗ API endpoint failed - unexpected response" -ForegroundColor Red
        Write-Host "Response: $apiResponse"
        Stop-Job $serverJob
        Remove-Job $serverJob
        exit 1
    }
} catch {
    Write-Host "✗ API endpoint failed - connection error: $_" -ForegroundColor Red
    Stop-Job $serverJob
    Remove-Job $serverJob
    exit 1
}

# Test Web UI endpoint
Write-Host "Testing Web UI endpoint..."
try {
    $webResponse = Invoke-WebRequest -Uri "http://localhost:8000/web/" -TimeoutSec 10
    if ($webResponse.Content -like "*Dashboard - Pynomaly*") {
        Write-Host "✓ Web UI endpoint working" -ForegroundColor Green
    } else {
        Write-Host "✗ Web UI endpoint failed - unexpected content" -ForegroundColor Red
        Write-Host "Content snippet: $($webResponse.Content.Substring(0, [Math]::Min(200, $webResponse.Content.Length)))"
        Stop-Job $serverJob
        Remove-Job $serverJob
        exit 1
    }
} catch {
    Write-Host "✗ Web UI endpoint failed - connection error: $_" -ForegroundColor Red
    Stop-Job $serverJob
    Remove-Job $serverJob
    exit 1
}

# Stop server
Stop-Job $serverJob
Remove-Job $serverJob
Start-Sleep -Seconds 2
Write-Host "✓ Server stopped" -ForegroundColor Green

Write-Host ""
Write-Host "🎉 All tests passed! Pynomaly web app works correctly in current PowerShell environment." -ForegroundColor Green
Write-Host "✓ Python imports working" -ForegroundColor Green
Write-Host "✓ App creation working" -ForegroundColor Green
Write-Host "✓ Server startup working" -ForegroundColor Green
Write-Host "✓ API endpoints working" -ForegroundColor Green
Write-Host "✓ Web UI working" -ForegroundColor Green
