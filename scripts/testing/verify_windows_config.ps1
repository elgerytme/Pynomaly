# Verify Windows Configuration Script
# This script checks if Windows-specific configurations are working correctly

function Write-Info {
    param([string]$Message)
    Write-Host "Info: $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "Success: $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "Warning: $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "Error: $Message" -ForegroundColor Red
}

$passed = 0
$failed = 0

Write-Host "===============================================" -ForegroundColor Magenta
Write-Host "  Verifying Windows Configuration" -ForegroundColor Magenta
Write-Host "===============================================" -ForegroundColor Magenta
Write-Host ""

# Test 1: Check Git line ending configuration
Write-Info "Testing Git line ending configuration..."
try {
    $autocrlf = git config --get core.autocrlf
    $eol = git config --get core.eol
    
    if ($autocrlf -eq "false" -and $eol -eq "lf") {
        Write-Success "Git line endings configured correctly (autocrlf=false, eol=lf)"
        $passed++
    } else {
        Write-Warning-Custom "Git line endings not configured optimally (autocrlf=$autocrlf, eol=$eol)"
        $failed++
    }
} catch {
    Write-Error-Custom "Failed to check Git configuration: $($_.Exception.Message)"
    $failed++
}

# Test 2: Check pre-commit installation
Write-Info "Testing pre-commit installation..."
try {
    $preCommitCmd = Get-Command pre-commit -ErrorAction SilentlyContinue
    if ($preCommitCmd) {
        Write-Success "Pre-commit is installed"
        $passed++
        
        # Check if hooks are installed
        if (Test-Path ".git/hooks/pre-commit") {
            Write-Success "Pre-commit hooks are installed"
            $passed++
        } else {
            Write-Warning-Custom "Pre-commit hooks not installed"
            $failed++
        }
    } else {
        Write-Warning-Custom "Pre-commit not found"
        $failed++
    }
} catch {
    Write-Error-Custom "Failed to check pre-commit: $($_.Exception.Message)"
    $failed++
}

# Test 3: Check shebang fixes
Write-Info "Testing shebang fixes..."
try {
    $bashScripts = Get-ChildItem -Path "scripts" -Filter "*.sh" -Recurse
    $fixedCount = 0
    
    foreach ($script in $bashScripts) {
        $firstLine = Get-Content -Path $script.FullName -Head 1
        if ($firstLine -match "^#!/usr/bin/env bash" -or $firstLine -match "^#!/usr/bin/env sh") {
            $fixedCount++
        }
    }
    
    if ($fixedCount -gt 0) {
        Write-Success "Found $fixedCount shell scripts with portable shebangs"
        $passed++
    } else {
        Write-Warning-Custom "No shell scripts with portable shebangs found"
        $failed++
    }
} catch {
    Write-Error-Custom "Failed to check shebangs: $($_.Exception.Message)"
    $failed++
}

# Test 4: Check pyproject.toml for Windows wheels extra
Write-Info "Testing Windows wheels extra in pyproject.toml..."
try {
    $content = Get-Content -Path "pyproject.toml" -Raw
    if ($content -match "windows-wheels.*=.*\[") {
        Write-Success "Windows wheels extra found in pyproject.toml"
        $passed++
    } else {
        Write-Warning-Custom "Windows wheels extra not found in pyproject.toml"
        $failed++
    }
} catch {
    Write-Error-Custom "Failed to check pyproject.toml: $($_.Exception.Message)"
    $failed++
}

# Test 5: Check Windows setup documentation
Write-Info "Testing Windows setup documentation..."
try {
    if (Test-Path "docs/windows_setup.md") {
        Write-Success "Windows setup documentation exists"
        $passed++
    } else {
        Write-Warning-Custom "Windows setup documentation not found"
        $failed++
    }
} catch {
    Write-Error-Custom "Failed to check documentation: $($_.Exception.Message)"
    $failed++
}

# Test 6: Check PowerShell configuration scripts
Write-Info "Testing PowerShell configuration scripts..."
try {
    $configScripts = @(
        "scripts/setup/configure_windows.ps1",
        "scripts/setup/fix_shebangs_windows_fixed.ps1",
        "scripts/setup/setup_windows.ps1"
    )
    
    $foundScripts = 0
    foreach ($script in $configScripts) {
        if (Test-Path $script) {
            $foundScripts++
        }
    }
    
    if ($foundScripts -gt 0) {
        Write-Success "Found $foundScripts PowerShell configuration scripts"
        $passed++
    } else {
        Write-Warning-Custom "No PowerShell configuration scripts found"
        $failed++
    }
} catch {
    Write-Error-Custom "Failed to check PowerShell scripts: $($_.Exception.Message)"
    $failed++
}

# Test 7: Test Python imports
Write-Info "Testing Python core imports..."
try {
    python -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'; print('Python version OK')"
    python -c "import numpy; print('NumPy import OK')"
    python -c "import pandas; print('Pandas import OK')"
    python -c "from pynomaly.domain.entities import Dataset; print('Pynomaly core import OK')"
    
    Write-Success "Python core imports working"
    $passed++
} catch {
    Write-Error-Custom "Python imports failed: $($_.Exception.Message)"
    $failed++
}

# Summary
Write-Host ""
Write-Host "===============================================" -ForegroundColor Magenta
Write-Host "  Verification Results" -ForegroundColor Magenta
Write-Host "===============================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Tests passed: $passed" -ForegroundColor Green
Write-Host "Tests failed: $failed" -ForegroundColor Red
Write-Host ""

if ($failed -eq 0) {
    Write-Success "All Windows configuration tests passed!"
    exit 0
} else {
    Write-Warning-Custom "Some Windows configuration tests failed. See output above for details."
    exit 1
}
