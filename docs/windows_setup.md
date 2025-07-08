# Windows Setup Instructions for Pynomaly

This document provides setup instructions specifically for Windows users, including handling of native code packages and build issues.

## Pre-requisites

- Ensure Python 3.11+ is installed and added to PATH.
- Install Git for Windows to provide a bash-like environment if needed.
- Microsoft Visual C++ Build Tools (for packages with native code)
- Poetry (recommended) or pip for package management

## Windows-Specific Package Installation

### Handling Native Code Packages

Some packages like `numpy`, `scikit-learn`, `orjson`, and others contain native code that may fail to compile on Windows. Here are the recommended approaches:

#### Option 1: Use Pre-compiled Wheels (Recommended)

1. **Install with Windows-specific extras:**
   ```powershell
   pip install -e ".[windows-wheels]"
   ```

2. **Alternative PyPI-compatible installation:**
   ```powershell
   pip install --only-binary=all -e ".[ml]"
   ```

#### Option 2: Gohlke Wheels for Difficult Packages

For packages that still fail, you can use Christoph Gohlke's unofficial Windows binaries:

1. Visit https://www.lfd.uci.edu/~gohlke/pythonlibs/
2. Download the appropriate wheel for your Python version
3. Install manually:
   ```powershell
   pip install path/to/downloaded/wheel.whl
   ```

#### Option 3: Conda/Mamba for Scientific Packages

For complex scientific packages, consider using conda:

```powershell
conda install -c conda-forge numpy scipy scikit-learn
pip install -e . --no-deps
```

### Common Problematic Packages and Solutions

- **numpy**: Usually works with pip, but Gohlke wheels available if needed
- **scipy**: May need Visual C++ Build Tools or Gohlke wheels
- **scikit-learn**: Usually works with pip, dependencies may need attention
- **orjson**: Fast JSON library, may need Rust compiler or use slower alternatives
- **lxml**: XML processing, may need libxml2/libxslt or use Gohlke wheels
- **psycopg2**: PostgreSQL adapter, use `psycopg2-binary` instead
- **pyarrow**: Apache Arrow, usually has Windows wheels available

## Pre-commit Hook Configuration

### Line Ending Issues

Windows uses CRLF line endings, which can cause issues with pre-commit hooks. Configure git and pre-commit:

```powershell
# Configure git line endings
git config core.autocrlf false
git config core.eol lf

# Install pre-commit hooks
poetry run pre-commit install

# If using pip:
pip install pre-commit
pre-commit install
```

### Fix Existing Line Endings

```powershell
# Normalize line endings in repository
git add --renormalize .
git commit -m "Normalize line endings"
```

## Build Script Compatibility

### Shebang Line Fixes

Many build scripts use hard-coded `/bin/bash` shebangs that don't work on Windows. Use the provided PowerShell script to fix them:

```powershell
# Run the shebang fix script
.\scripts\setup\fix_shebangs_windows.ps1

# Or manually run with Git Bash
bash ./scripts/some_script.sh
```

### PowerShell Script Execution

If you encounter execution policy issues:

```powershell
# Allow script execution for current user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run a specific script bypassing policy
PowerShell -ExecutionPolicy Bypass -File .\scripts\setup\setup_windows.ps1
```

## Environment Setup

### Recommended Setup Process

1. **Use the Windows setup script:**
   ```powershell
   .\scripts\setup\setup_windows.ps1
   ```

2. **Manual setup steps:**
   ```powershell
   # Create virtual environment
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   
   # Upgrade pip
   python -m pip install --upgrade pip
   
   # Install with Windows-friendly options
   pip install -e ".[windows-wheels,api,cli]"
   
   # Configure pre-commit
   pre-commit install
   git config core.autocrlf false
   ```

## Testing on Windows

### PowerShell Test Scripts

Use the provided PowerShell equivalents for testing:

```powershell
# Run basic tests
.\scripts\testing\test-current.ps1

# Run comprehensive tests
.\scripts\testing\test_powershell_comprehensive.ps1

# Run API tests
.\scripts\testing\test_api_powershell.ps1
```

### Cross-Platform Testing

To test bash scripts on Windows:

```powershell
# Use Git Bash
bash .\scripts\testing\test-current.sh

# Or use WSL if available
wsl bash ./scripts/testing/test-current.sh
```

## Docker on Windows

### Docker Desktop

If using Docker Desktop on Windows:

```powershell
# Build and run development container
.\scripts\docker\dev\run-dev.ps1

# Run production container
.\scripts\docker\prod\run-prod.ps1
```

### WSL2 Backend

For better performance, use WSL2 backend with Docker Desktop.

## Troubleshooting Common Issues

### Python Path Issues

```powershell
# Check Python installation
python --version
where python

# Check pip installation
pip --version
where pip
```

### Virtual Environment Issues

```powershell
# If activation fails, try:
.venv\Scripts\Activate.bat

# Or use full path:
C:\path\to\project\.venv\Scripts\Activate.ps1
```

### Package Installation Failures

1. **Update pip and setuptools:**
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Install Visual C++ Build Tools:**
   - Download from Microsoft
   - Or install Visual Studio with C++ workload

3. **Use conda for scientific packages:**
   ```powershell
   conda install -c conda-forge numpy scipy scikit-learn
   ```

4. **Use Gohlke wheels for difficult packages:**
   - Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/
   - Install with `pip install wheel_file.whl`

### Git Line Ending Issues

```powershell
# Fix line endings globally
git config --global core.autocrlf false
git config --global core.eol lf

# Fix in current repository
git add --renormalize .
git commit -m "Fix line endings"
```

### Pre-commit Hook Failures

```powershell
# Skip hooks temporarily
git commit -m "message" --no-verify

# Update hooks
pre-commit autoupdate

# Run hooks manually
pre-commit run --all-files
```

## Development Tools

### Recommended Windows Development Setup

1. **Windows Terminal** - Better terminal experience
2. **Git for Windows** - Provides bash environment
3. **Visual Studio Code** - Excellent Python support
4. **PowerShell 7+** - Modern PowerShell with better compatibility
5. **Windows Subsystem for Linux (WSL2)** - For bash script compatibility

### VS Code Configuration

Recommended VS Code settings for Windows development:

```json
{
    "python.defaultInterpreterPath": "./.venv/Scripts/python.exe",
    "files.eol": "\n",
    "files.insertFinalNewline": true,
    "files.trimTrailingWhitespace": true,
    "python.terminal.activateEnvironment": true
}
```

## Performance Considerations

### Antivirus Exclusions

Add these to your antivirus exclusions for better performance:
- Project directory
- Python installation directory
- Virtual environment directories
- Node.js installation (if using)

### Windows Defender

```powershell
# Add exclusions (run as administrator)
Add-MpPreference -ExclusionPath "C:\path\to\project"
Add-MpPreference -ExclusionPath "C:\path\to\python"
```

## Additional Resources

- [Python on Windows FAQ](https://docs.python.org/3/faq/windows.html)
- [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
- [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- [Git for Windows](https://gitforwindows.org/)
- [Windows Terminal](https://github.com/Microsoft/Terminal)
