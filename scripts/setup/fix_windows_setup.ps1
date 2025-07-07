# Fix Windows Setup Issues for Pynomaly
# PowerShell script to resolve common Windows installation problems

Write-Host "🔧 Fixing Pynomaly Windows Setup Issues" -ForegroundColor Cyan
Write-Host "=" * 50

# Step 1: Remove conflicting package
Write-Host "`n📌 Step 1: Removing conflicting pynomaly package" -ForegroundColor Yellow
try {
    python -m pip uninstall pynomaly -y
    Write-Host "✅ Removed existing pynomaly package" -ForegroundColor Green
} catch {
    Write-Host "⚠️  No existing package found or removal failed" -ForegroundColor Yellow
}

# Step 2: Check/create virtual environment
Write-Host "`n📌 Step 2: Setting up virtual environment" -ForegroundColor Yellow

if (Test-Path ".venv") {
    Write-Host "Virtual environment exists, checking integrity..." -ForegroundColor Blue
    
    if (!(Test-Path ".venv\Scripts\python.exe")) {
        Write-Host "❌ Virtual environment appears broken, recreating..." -ForegroundColor Red
        Remove-Item -Recurse -Force ".venv" -ErrorAction SilentlyContinue
        python -m venv .venv
    }
} else {
    Write-Host "Creating new virtual environment..." -ForegroundColor Blue
    python -m venv .venv
}

# Step 3: Activate virtual environment and install
Write-Host "`n📌 Step 3: Installing package in development mode" -ForegroundColor Yellow

# Activate virtual environment
& ".venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Blue
python -m pip install --upgrade pip

# Install package
Write-Host "Installing pynomaly in development mode..." -ForegroundColor Blue
try {
    python -m pip install -e .[server]
    Write-Host "✅ Installed with server extras" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Server extras failed, trying minimal install..." -ForegroundColor Yellow
    python -m pip install -e .
}

# Step 4: Install missing dependencies
Write-Host "`n📌 Step 4: Installing missing dependencies" -ForegroundColor Yellow

$dependencies = @(
    "prometheus-fastapi-instrumentator>=7.0.0",
    "shap>=0.46.0", 
    "lime>=0.2.0.1"
)

foreach ($dep in $dependencies) {
    try {
        python -m pip install $dep
        Write-Host "✅ Installed $dep" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to install $dep" -ForegroundColor Red
    }
}

# Step 5: Verify installation
Write-Host "`n📌 Step 5: Verifying installation" -ForegroundColor Yellow

try {
    python -c "import pynomaly; print('✅ pynomaly imports successfully')"
} catch {
    Write-Host "❌ Failed to import pynomaly" -ForegroundColor Red
}

try {
    python -c "from pynomaly.presentation.cli.app import app; print('✅ CLI module available')"
} catch {
    Write-Host "❌ CLI module import failed" -ForegroundColor Red
}

# Step 6: Test CLI access
Write-Host "`n📌 Step 6: Testing CLI access" -ForegroundColor Yellow

try {
    python -m pynomaly.presentation.cli.app --help | Select-Object -First 5
    Write-Host "✅ CLI accessible via python -m pynomaly.presentation.cli.app" -ForegroundColor Green
} catch {
    Write-Host "❌ CLI module test failed" -ForegroundColor Red
}

# Final instructions
Write-Host "`n📌 Setup Complete! Next Steps:" -ForegroundColor Cyan
Write-Host "1. CLI: python -m pynomaly.presentation.cli.app --help"
Write-Host "2. API: python scripts\run_api.py"  
Write-Host "3. Web UI: python scripts\run_web_ui.py"
Write-Host "4. For production: python -m pip install -e .[production]"
Write-Host "`nEnvironment is ready for development!" -ForegroundColor Green
