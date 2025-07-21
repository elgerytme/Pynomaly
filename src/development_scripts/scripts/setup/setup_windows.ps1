# Pynomaly Windows PowerShell Setup Script
# This script sets up the project in development mode on Windows

param(
    [string]$InstallProfile = "server",
    [switch]$Force = $false,
    [switch]$Verbose = $false
)

# Color functions for better output
function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "🔧 $Message" -ForegroundColor Cyan
    Write-Host "=" * 50
}

# Check prerequisites
function Test-Prerequisites {
    Write-Step "Checking Prerequisites"

    # Check Python
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        Write-Error-Custom "Python not found. Please install Python 3.11+ and add it to PATH."
        return $false
    }

    $pythonVersion = python --version 2>&1
    Write-Success "Found Python: $pythonVersion"

    # Check pip
    $pipCmd = Get-Command pip -ErrorAction SilentlyContinue
    if (-not $pipCmd) {
        Write-Error-Custom "pip not found. Please ensure pip is installed."
        return $false
    }

    Write-Success "pip is available"

    # Check if we're in the right directory
    if (-not (Test-Path "pyproject.toml")) {
        Write-Error-Custom "pyproject.toml not found. Please run this script from the project root directory."
        return $false
    }

    Write-Success "Found pyproject.toml - correct directory"
    return $true
}

# Clean up existing installations
function Clear-ExistingInstall {
    Write-Step "Cleaning Up Existing Installations"

    Write-Info "Removing any existing pynomaly installations..."

    # Uninstall the wrong PyNomaly package if installed
    $wrongPackage = pip list | Select-String "PyNomaly"
    if ($wrongPackage) {
        Write-Warning-Custom "Found wrong 'PyNomaly' package - removing it"
        pip uninstall -y PyNomaly python-utils
    }

    # Uninstall our package if installed in development mode
    $ourPackage = pip list | Select-String "pynomaly.*editable"
    if ($ourPackage) {
        Write-Info "Uninstalling existing development installation"
        pip uninstall -y pynomaly
    }

    Write-Success "Cleanup completed"
}

# Set up virtual environment
function Setup-VirtualEnvironment {
    Write-Step "Setting Up Virtual Environment"

    if (Test-Path ".venv") {
        if ($Force) {
            Write-Warning-Custom "Removing existing .venv directory"
            Remove-Item -Recurse -Force .venv
        } else {
            Write-Info "Virtual environment already exists. Use -Force to recreate."
            return $true
        }
    }

    Write-Info "Creating virtual environment..."
    python -m venv .venv

    if (-not (Test-Path ".venv")) {
        Write-Error-Custom "Failed to create virtual environment"
        return $false
    }

    Write-Success "Virtual environment created"
    return $true
}

# Activate virtual environment
function Enable-VirtualEnvironment {
    Write-Step "Activating Virtual Environment"

    $activateScript = ".venv\Scripts\Activate.ps1"

    if (-not (Test-Path $activateScript)) {
        Write-Error-Custom "Virtual environment activation script not found"
        return $false
    }

    # Check execution policy
    $executionPolicy = Get-ExecutionPolicy
    if ($executionPolicy -eq "Restricted") {
        Write-Warning-Custom "PowerShell execution policy is Restricted. Attempting to run activation script..."
        Write-Info "You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    }

    try {
        & $activateScript
        Write-Success "Virtual environment activated"
        return $true
    }
    catch {
        Write-Warning-Custom "Could not activate virtual environment. Continuing without activation..."
        Write-Info "You may need to manually activate: .venv\Scripts\Activate.ps1"
        return $false
    }
}

# Upgrade pip
function Update-Pip {
    Write-Step "Upgrading pip"

    Write-Info "Upgrading pip to latest version..."
    python -m pip install --upgrade pip

    $pipVersion = pip --version
    Write-Success "pip updated: $pipVersion"
}

# Install dependencies based on profile
function Install-Dependencies {
    param([string]$Profile)

    Write-Step "Installing Dependencies (Profile: $Profile)"

    Write-Info "Installing project in development mode..."

    try {
        switch ($Profile.ToLower()) {
            "minimal" {
                Write-Info "Installing minimal core dependencies..."
                pip install -e .
            }
            "server" {
                Write-Info "Installing server dependencies (CLI + API + basic ML)..."
                pip install -e ".[api,cli,ml]"
            }
            "production" {
                Write-Info "Installing production dependencies..."
                pip install -e ".[production]"
            }
            "all" {
                Write-Info "Installing all dependencies..."
                pip install -e ".[all]"
            }
            default {
                Write-Info "Installing server dependencies (default)..."
                pip install -e ".[api,cli,ml]"
            }
        }

        Write-Success "Dependencies installed successfully"
        return $true
    }
    catch {
        Write-Error-Custom "Failed to install dependencies: $($_.Exception.Message)"

        # Fallback: try with requirements file
        Write-Info "Trying fallback installation with requirements file..."

        try {
            pip install -r requirements.txt
            Write-Warning-Custom "Installed minimal dependencies from requirements.txt"
            Write-Info "To get full functionality, manually install extras: pip install fastapi uvicorn typer rich"
            return $true
        }
        catch {
            Write-Error-Custom "Fallback installation also failed"
            return $false
        }
    }
}

# Verify installation
function Test-Installation {
    Write-Step "Verifying Installation"

    Write-Info "Testing core imports..."

    $testScript = @"
import sys
try:
    # Test core imports
    import pyod
    print("✅ PyOD available")

    import numpy as np
    print("✅ NumPy available")

    import pandas as pd
    print("✅ Pandas available")

    # Test project imports
    from pynomaly.domain.entities import Dataset
    print("✅ Pynomaly core imports working")

    # Test optional imports
    try:
        from pynomaly.presentation.cli.app import app
        print("✅ CLI component available")
    except ImportError:
        print("⚠️  CLI component not available (install with: pip install typer rich)")

    try:
        from pynomaly.presentation.api.app import create_app
        print("✅ API component available")
    except ImportError:
        print("⚠️  API component not available (install with: pip install fastapi uvicorn)")

    print("✅ Installation verification completed successfully")

except Exception as e:
    print(f"❌ Installation verification failed: {e}")
    sys.exit(1)
"@

    try {
        $result = python -c $testScript
        Write-Host $result
        Write-Success "Installation verification passed"
        return $true
    }
    catch {
        Write-Error-Custom "Installation verification failed"
        return $false
    }
}

# Show usage information
function Show-Usage {
    Write-Step "Setup Complete - Usage Information"

    Write-Host ""
    Write-Host "🎉 Pynomaly setup completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Next Steps:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Activate virtual environment (if not already active):"
    Write-Host "   .venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "2. Available commands:" -ForegroundColor Yellow

    # Test if CLI is available
    try {
        python -c "from pynomaly.presentation.cli.app import app" 2>$null
        Write-Host "   pynomaly --help                 # CLI help" -ForegroundColor Cyan
        Write-Host "   python -m pynomaly.presentation.cli.app --help  # Alternative CLI" -ForegroundColor Cyan
    }
    catch {
        Write-Host "   python -m pynomaly.presentation.cli.app --help  # CLI (typer needed)" -ForegroundColor Cyan
    }

    # Test if API is available
    try {
        python -c "from pynomaly.presentation.api.app import create_app" 2>$null
        Write-Host "   uvicorn pynomaly.presentation.api:app --reload  # API server" -ForegroundColor Cyan
    }
    catch {
        Write-Host "   # API server (fastapi uvicorn needed)" -ForegroundColor Cyan
    }

    # Test if Web UI is available
    try {
        python -c "from pynomaly.presentation.web.app import create_web_app" 2>$null
        Write-Host "   uvicorn pynomaly.presentation.web.app:create_web_app --reload  # Web UI" -ForegroundColor Cyan
    }
    catch {
        Write-Host "   # Web UI (fastapi uvicorn jinja2 needed)" -ForegroundColor Cyan
    }

    Write-Host ""
    Write-Host "3. Install additional features:" -ForegroundColor Yellow
    Write-Host "   pip install -e .[api]          # Web API functionality" -ForegroundColor Cyan
    Write-Host "   pip install -e .[cli]          # Enhanced CLI" -ForegroundColor Cyan
    Write-Host "   pip install -e .[ml]           # ML algorithms" -ForegroundColor Cyan
    Write-Host "   pip install -e .[all]          # Everything" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "4. Run tests:" -ForegroundColor Yellow
    Write-Host "   .\scripts\test_presentation_components.ps1" -ForegroundColor Cyan
    Write-Host ""
}

# Main execution
function Main {
    Write-Host "🚀 Pynomaly Windows Setup Script" -ForegroundColor Magenta
    Write-Host "=================================" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "Install Profile: $InstallProfile"
    Write-Host "Force reinstall: $Force"
    Write-Host ""

    # Check prerequisites
    if (-not (Test-Prerequisites)) {
        Write-Error-Custom "Prerequisites check failed. Please fix the issues above and try again."
        exit 1
    }

    # Clean existing installations
    Clear-ExistingInstall

    # Set up virtual environment
    if (-not (Setup-VirtualEnvironment)) {
        Write-Error-Custom "Virtual environment setup failed."
        exit 1
    }

    # Activate virtual environment (optional - may fail due to execution policy)
    Enable-VirtualEnvironment

    # Upgrade pip
    Update-Pip

    # Install dependencies
    if (-not (Install-Dependencies -Profile $InstallProfile)) {
        Write-Error-Custom "Dependency installation failed."
        exit 1
    }

    # Verify installation
    if (-not (Test-Installation)) {
        Write-Error-Custom "Installation verification failed."
        exit 1
    }

    # Show usage
    Show-Usage
}

# Run main function
try {
    Main
}
catch {
    Write-Error-Custom "Setup failed with error: $($_.Exception.Message)"
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Ensure Python 3.11+ is installed and in PATH"
    Write-Host "2. Run PowerShell as Administrator if needed"
    Write-Host "3. Set execution policy: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    Write-Host "4. Check that you're in the project directory"
    Write-Host ""
    exit 1
}
