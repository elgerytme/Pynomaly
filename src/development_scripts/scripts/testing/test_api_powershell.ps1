# PowerShell script to test anomaly detection API
# Run this with: pwsh -File test_api_powershell.ps1

Write-Host "=== anomaly detection API PowerShell Test ===" -ForegroundColor Green

# Set environment variables
$env:PYTHONPATH = "C:\Users\andre\anomaly_detection\src"
$apiPort = 8002
$apiUrl = "http://127.0.0.1:$apiPort"

# Function to test API endpoint
function Test-ApiEndpoint {
    param(
        [string]$Url,
        [string]$Description
    )

    try {
        Write-Host "Testing $Description..." -ForegroundColor Yellow
        $response = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 10
        Write-Host "✓ $Description: SUCCESS" -ForegroundColor Green
        return $response
    } catch {
        Write-Host "✗ $Description: FAILED - $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Start the API server
Write-Host "Starting anomaly detection API server on port $apiPort..." -ForegroundColor Blue
$serverProcess = Start-Process -FilePath "uvicorn" -ArgumentList @(
    "anomaly_detection.presentation.api:app",
    "--host", "127.0.0.1",
    "--port", $apiPort,
    "--reload"
) -NoNewWindow -PassThru

# Wait for server to start
Write-Host "Waiting for server to start..." -ForegroundColor Blue
Start-Sleep -Seconds 8

try {
    # Test root endpoint
    $rootResponse = Test-ApiEndpoint -Url "$apiUrl/" -Description "Root endpoint"
    if ($rootResponse) {
        Write-Host "  Message: $($rootResponse.message)" -ForegroundColor Cyan
        Write-Host "  Version: $($rootResponse.version)" -ForegroundColor Cyan
    }

    # Test health endpoint
    $healthResponse = Test-ApiEndpoint -Url "$apiUrl/api/health/" -Description "Health endpoint"
    if ($healthResponse) {
        Write-Host "  Overall Status: $($healthResponse.overall_status)" -ForegroundColor Cyan
        Write-Host "  Uptime: $([math]::Round($healthResponse.uptime_seconds, 2)) seconds" -ForegroundColor Cyan
        Write-Host "  Healthy Checks: $($healthResponse.summary.healthy_checks)/$($healthResponse.summary.total_checks)" -ForegroundColor Cyan
    }

    # Test docs endpoint
    $docsResponse = Test-ApiEndpoint -Url "$apiUrl/api/docs" -Description "Documentation endpoint"

    # Test OpenAPI schema
    $openApiResponse = Test-ApiEndpoint -Url "$apiUrl/api/openapi.json" -Description "OpenAPI schema"
    if ($openApiResponse) {
        Write-Host "  API Title: $($openApiResponse.info.title)" -ForegroundColor Cyan
        Write-Host "  API Version: $($openApiResponse.info.version)" -ForegroundColor Cyan
        Write-Host "  Endpoints: $($openApiResponse.paths.Count)" -ForegroundColor Cyan
    }

    Write-Host "`n=== PowerShell API Test Complete ===" -ForegroundColor Green

} catch {
    Write-Host "Test failed with error: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    # Clean up - stop the server
    Write-Host "Stopping API server..." -ForegroundColor Blue
    if ($serverProcess -and !$serverProcess.HasExited) {
        $serverProcess.Kill()
        $serverProcess.WaitForExit(5000)
    }

    # Kill any remaining uvicorn processes
    Get-Process -Name "uvicorn" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

    Write-Host "Cleanup complete." -ForegroundColor Blue
}
