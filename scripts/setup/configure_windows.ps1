# Windows Configuration Script for Pynomaly
# This script applies all Windows-specific fixes and configurations

param(
    [switch]$SkipPreCommit = $false,
    [switch]$SkipShebangs = $false,
    [switch]$SkipGitConfig = $false,
    [switch]$Verbose = $false
)

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

Write-Host "=======================================" -ForegroundColor Magenta
Write-Host "  Windows Configuration for Pynomaly" -ForegroundColor Magenta
Write-Host "=======================================" -ForegroundColor Magenta
Write-Host ""

# Step 1: Configure Git for Windows-friendly line endings
if (-not $SkipGitConfig) {
    Write-Info "Configuring Git for Windows line endings..."
    try {
        git config core.autocrlf false
        git config core.eol lf
        Write-Success "Git configuration updated for Windows"
    }
    catch {
        Write-Error-Custom "Failed to configure Git: $($_.Exception.Message)"
    }
}

# Step 2: Fix bash shebangs in scripts
if (-not $SkipShebangs) {
    Write-Info "Fixing bash shebangs for Windows compatibility..."
    try {
        if (Test-Path "scripts/setup/fix_shebangs_windows_fixed.ps1") {
            & "scripts/setup/fix_shebangs_windows_fixed.ps1"
            Write-Success "Bash shebangs fixed successfully"
        } else {
            Write-Warning-Custom "Shebang fix script not found, skipping..."
        }
    }
    catch {
        Write-Error-Custom "Failed to fix shebangs: $($_.Exception.Message)"
    }
}

# Step 3: Install and configure pre-commit hooks
if (-not $SkipPreCommit) {
    Write-Info "Installing pre-commit hooks..."
    try {
        # Check if pre-commit is installed
        $preCommitCmd = Get-Command pre-commit -ErrorAction SilentlyContinue
        if (-not $preCommitCmd) {
            Write-Info "Installing pre-commit..."
            pip install pre-commit
        }
        
        # Install hooks
        pre-commit install
        Write-Success "Pre-commit hooks installed successfully"
    }
    catch {
        Write-Error-Custom "Failed to install pre-commit hooks: $($_.Exception.Message)"
        Write-Info "You can install manually with: pip install pre-commit && pre-commit install"
    }
}

# Step 4: Display Windows-specific package installation instructions
Write-Info "Windows-specific package installation options:"
Write-Host ""
Write-Host "For packages with native code (numpy, scikit-learn, etc.), use:" -ForegroundColor Yellow
Write-Host "  pip install -e `".[windows-wheels]`"" -ForegroundColor Cyan
Write-Host "or" -ForegroundColor Yellow
Write-Host "  pip install --only-binary=all -e `".[ml]`"" -ForegroundColor Cyan
Write-Host ""

# Step 5: Check for potential issues
Write-Info "Checking for potential Windows-specific issues..."

# Check Python version
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.1[1-9]") {
    Write-Success "Python version: $pythonVersion"
} else {
    Write-Warning-Custom "Python version may be incompatible: $pythonVersion"
}

# Check for Visual C++ Build Tools
$vcVarsPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (Test-Path $vcVarsPath) {
    Write-Success "Visual C++ Build Tools found"
} else {
    Write-Warning-Custom "Visual C++ Build Tools not found - may be needed for some packages"
}

# Check execution policy
$executionPolicy = Get-ExecutionPolicy
if ($executionPolicy -eq "Restricted") {
    Write-Warning-Custom "PowerShell execution policy is Restricted"
    Write-Info "Consider running: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
}

Write-Host ""
Write-Host "=======================================" -ForegroundColor Green
Write-Host "  Windows configuration completed!" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Install dependencies: pip install -e `".[windows-wheels,api,cli]`"" -ForegroundColor Cyan
Write-Host "2. Run tests: pytest tests/" -ForegroundColor Cyan
Write-Host "3. See docs/windows_setup.md for detailed instructions" -ForegroundColor Cyan
Write-Host ""
