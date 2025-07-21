#!/usr/bin/env pwsh
# Test script for setup_simple.py on Windows PowerShell

Write-Host "=========================================="
Write-Host "Testing setup_simple.py on Windows"
Write-Host "=========================================="

# Create a temporary test directory
$testDir = "test_setup_simple_$(Get-Random)"
Write-Host "Creating test directory: $testDir"

try {
    # Create test directory and copy essential files
    New-Item -ItemType Directory -Path $testDir -Force | Out-Null
    Set-Location $testDir

    # Copy essential files from parent directory
    Copy-Item "../pyproject.toml" . -Force
    Copy-Item "../requirements.txt" . -Force
    Copy-Item "../setup.py" . -Force -ErrorAction SilentlyContinue
    Copy-Item "../scripts/setup_simple.py" . -Force

    # Create minimal src structure
    New-Item -ItemType Directory -Path "src\anomaly_detection" -Force | Out-Null
    New-Item -ItemType File -Path "src\anomaly_detection\__init__.py" -Force | Out-Null
    Set-Content -Path "src\anomaly_detection\__init__.py" -Value "__version__ = '0.1.0'"

    # Create minimal domain entities
    New-Item -ItemType Directory -Path "src\anomaly_detection\domain" -Force | Out-Null
    New-Item -ItemType Directory -Path "src\anomaly_detection\domain\entities" -Force | Out-Null
    New-Item -ItemType File -Path "src\anomaly_detection\domain\__init__.py" -Force | Out-Null
    New-Item -ItemType File -Path "src\anomaly_detection\domain\entities\__init__.py" -Force | Out-Null
    Set-Content -Path "src\anomaly_detection\domain\entities\__init__.py" -Value @"
class Dataset:
    pass

class Anomaly:
    pass
"@

    Write-Host "Test directory structure created"
    Write-Host "Files in test directory:"
    Get-ChildItem -Recurse | ForEach-Object { Write-Host "  $($_.FullName.Replace((Get-Location).Path, ''))" }

    Write-Host "`nRunning setup_simple.py..."
    python setup_simple.py --clean

    Write-Host "`nTest completed in directory: $testDir"
    Write-Host "Check the output above for results"

} catch {
    Write-Host "Error during test: $_" -ForegroundColor Red
} finally {
    # Return to parent directory
    Set-Location ..
    Write-Host "`nTest directory: $testDir (not automatically cleaned up)"
}
