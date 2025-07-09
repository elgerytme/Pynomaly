# PowerShell script to test Pynomaly API endpoints
# 
# Usage: 
# 1. Start the server: python standalone_api.py
# 2. Run this script: .\test_api_endpoints.ps1

Write-Host "Testing Pynomaly REST API Endpoints" -ForegroundColor Green

# Test 1: Health Check
Write-Host "`nTesting Health Check Endpoint..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    Write-Host "Health Check Response:" -ForegroundColor Cyan
    $healthResponse | ConvertTo-Json -Depth 3
    
    # Verify CORS headers
    $headers = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method Get
    Write-Host "`nCORS Headers:" -ForegroundColor Cyan
    $headers.Headers["Access-Control-Allow-Origin"]
    $headers.Headers["Content-Type"]
}
catch {
    Write-Host "Error testing health endpoint: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: Detect Anomalies without file
Write-Host "`n`nTesting Detect Endpoint (no file)..." -ForegroundColor Yellow
try {
    $detectResponse = Invoke-RestMethod -Uri "http://localhost:8000/detect" -Method Post
    Write-Host "Detect Anomalies Response (no file):" -ForegroundColor Cyan
    $detectResponse | ConvertTo-Json -Depth 3
}
catch {
    Write-Host "Error testing detect endpoint: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: Detect Anomalies with CSV file (if sample.csv exists)
if (Test-Path "sample.csv") {
    Write-Host "`n`nTesting Detect Endpoint with CSV file..." -ForegroundColor Yellow
    try {
        # PowerShell multipart form data request
        $filePath = Resolve-Path "sample.csv"
        $boundary = [System.Guid]::NewGuid().ToString()
        $fileBytes = [System.IO.File]::ReadAllBytes($filePath)
        $fileContent = [System.Text.Encoding]::GetEncoding('iso-8859-1').GetString($fileBytes)
        
        $body = @"
--$boundary
Content-Disposition: form-data; name="file"; filename="sample.csv"
Content-Type: text/csv

$fileContent
--$boundary--
"@
        
        $headers = @{
            'Content-Type' = "multipart/form-data; boundary=$boundary"
        }
        
        $detectFileResponse = Invoke-RestMethod -Uri "http://localhost:8000/detect" -Method Post -Body $body -Headers $headers
        Write-Host "Detect Anomalies Response (with CSV):" -ForegroundColor Cyan
        $detectFileResponse | ConvertTo-Json -Depth 3
    }
    catch {
        Write-Host "Error testing detect endpoint with file: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "`n`nSkipping CSV file test - sample.csv not found" -ForegroundColor Yellow
}

# Test 4: Root endpoint
Write-Host "`n`nTesting Root Endpoint..." -ForegroundColor Yellow
try {
    $rootResponse = Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get
    Write-Host "Root Response:" -ForegroundColor Cyan
    $rootResponse | ConvertTo-Json -Depth 3
}
catch {
    Write-Host "Error testing root endpoint: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n`nAPI Testing Complete!" -ForegroundColor Green
