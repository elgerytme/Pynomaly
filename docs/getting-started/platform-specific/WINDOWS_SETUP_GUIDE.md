# Windows Setup Guide for Pynomaly

## Quick Start

The easiest way to set up Pynomaly on Windows is to use our automated setup script:

### Option 1: Automated Setup (Recommended)

1. **Open PowerShell as Administrator** (recommended) or regular user
2. **Navigate to the project directory**:
   ```powershell
   cd C:\Users\andre\Pynomaly
   ```
3. **Run the setup script**:
   ```powershell
   .\scripts\setup_windows.ps1
   ```

Or use the batch file:
```cmd
setup.bat
```

### Option 2: Manual Setup

If the automated setup doesn't work, follow these manual steps:

## Manual Setup Steps

### 1. Create Virtual Environment
```powershell
# Remove any existing .venv
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue

# Create new virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

### 2. Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### 3. Remove Wrong Package (if installed)
```powershell
# The wrong "PyNomaly" package may have been installed
pip uninstall -y PyNomaly python-utils
```

### 4. Install Pynomaly in Development Mode

Choose one of these installation profiles:

#### Minimal Installation (Core only)
```powershell
pip install -e .
```

#### Server Installation (Recommended - includes CLI + API)
```powershell
pip install -e ".[api,cli,ml]"
```

#### Production Installation  
```powershell
pip install -e ".[production]"
```

#### Complete Installation
```powershell
pip install -e ".[all]"
```

## Troubleshooting Common Issues

### Issue 1: PyOD Version Error
**Error**: `ERROR: Could not find a version that satisfies the requirement pyod>=2.0.6`

**Solution**: This has been fixed. The project now uses PyOD 2.0.5 which is available on PyPI.

### Issue 2: Wrong Package Installed
**Error**: `WARNING: pynomaly 0.3.4 does not provide the extra 'server'`

**Solution**: You installed the wrong package. Uninstall and install correctly:
```powershell
pip uninstall -y PyNomaly python-utils
pip install -e ".[server]"
```

### Issue 3: Command Not Found
**Error**: `pynomaly : The term 'pynomaly' is not recognized`

**Solution**: The CLI is not installed or not in PATH. Install with CLI support:
```powershell
pip install -e ".[cli]"
```

Then run commands using Python module:
```powershell
python -m pynomaly.presentation.cli.app --help
```

### Issue 4: PowerShell Execution Policy
**Error**: Cannot run script due to execution policy

**Solution**: Set execution policy for current user:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 5: Import Errors
**Error**: `ModuleNotFoundError: No module named 'pynomaly'`

**Solution**: Ensure you're in the virtual environment and installed in development mode:
```powershell
.venv\Scripts\Activate.ps1
pip install -e .
```

### Issue 6: FastAPI/Uvicorn Not Found
**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Install API dependencies:
```powershell
pip install -e ".[api]"
```

## Verification Steps

After installation, verify everything works:

### 1. Test Core Functionality
```powershell
python -c "
import pyod
import numpy as np
import pandas as pd
from pynomaly.domain.entities import Dataset
print('✅ Core functionality working')
"
```

### 2. Test CLI (if installed)
```powershell
python -m pynomaly.presentation.cli.app --help
```

### 3. Test API (if installed)
```powershell
python -c "
from pynomaly.presentation.api.app import create_app
from fastapi.testclient import TestClient
app = create_app()
client = TestClient(app)
response = client.get('/api/health')
print(f'✅ API working (status: {response.status_code})')
"
```

### 4. Test Web UI (if installed)
```powershell
python -c "
from pynomaly.presentation.web.app import create_web_app
app = create_web_app()
print('✅ Web UI working')
"
```

## Running the Application

### CLI Usage
```powershell
# Activate virtual environment first
.venv\Scripts\Activate.ps1

# Run CLI commands
python -m pynomaly.presentation.cli.app --help
python -m pynomaly.presentation.cli.app version
```

### API Server
```powershell
# Start API server
uvicorn pynomaly.presentation.api:app --reload

# Or with python module
python -c "
import uvicorn
from pynomaly.presentation.api.app import create_app
app = create_app()
uvicorn.run(app, host='127.0.0.1', port=8000)
"
```

### Web UI Server
```powershell
# Start Web UI server
uvicorn pynomaly.presentation.web.app:create_web_app --reload

# Access at: http://localhost:8000
```

## Development Workflow

### 1. Daily Development
```powershell
# Activate environment
.venv\Scripts\Activate.ps1

# Make changes to code...

# Run tests
python -m pytest

# Start development server
uvicorn pynomaly.presentation.api:app --reload
```

### 2. Adding Dependencies
```powershell
# Edit pyproject.toml
# Then reinstall
pip install -e ".[all]"
```

### 3. Running Tests
```powershell
# Run the test suite
.\scripts\test_presentation_components.ps1

# Or run pytest directly
python -m pytest tests/
```

## Environment Variables

Set these environment variables for development:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src;$env:PYTHONPATH"
$env:PYNOMALY_ENV = "development"
```

## Package Structure

After successful installation, you should have:

```
C:\Users\andre\Pynomaly\
├── .venv\                          # Virtual environment
├── src\pynomaly\                   # Source code
│   ├── domain\                     # Core business logic
│   ├── application\                # Use cases
│   ├── infrastructure\             # External integrations
│   └── presentation\               # CLI, API, Web UI
│       ├── cli\                    # Command-line interface
│       ├── api\                    # REST API
│       └── web\                    # Web UI
├── tests\                          # Test suite
├── scripts\                        # Setup and utility scripts
└── docs\                           # Documentation
```

## Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Run the automated setup script** - it handles most issues
3. **Verify prerequisites**: Python 3.11+, pip, PowerShell
4. **Check virtual environment** is activated
5. **Try manual installation steps** if automated setup fails

## Success Indicators

You know the setup worked when:

- ✅ Virtual environment is created and activated
- ✅ `pip list` shows `pynomaly` as editable install
- ✅ Core imports work: `from pynomaly.domain.entities import Dataset`
- ✅ CLI works: `python -m pynomaly.presentation.cli.app --help`
- ✅ API works: FastAPI health endpoint returns 200
- ✅ Web UI works: Can create web application

## Next Steps

After successful setup:

1. **Explore the CLI**: `python -m pynomaly.presentation.cli.app --help`
2. **Start the API**: `uvicorn pynomaly.presentation.api:app --reload`
3. **Try the Web UI**: `uvicorn pynomaly.presentation.web.app:create_web_app --reload`
4. **Read the documentation**: Check the `docs/` directory
5. **Run the tests**: `.\scripts\test_presentation_components.ps1`