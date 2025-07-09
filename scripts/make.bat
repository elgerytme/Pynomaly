@echo off
REM Windows batch script for Pynomaly development commands
REM Usage: make.bat <command>

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="setup" goto setup
if "%1"=="install" goto install
if "%1"=="dev-install" goto dev-install
if "%1"=="lint" goto lint
if "%1"=="format" goto format
if "%1"=="style" goto style
if "%1"=="typing" goto typing
if "%1"=="test" goto test
if "%1"=="test-all" goto test-all
if "%1"=="test-cov" goto test-cov
if "%1"=="test-unit" goto test-unit
if "%1"=="test-integration" goto test-integration
if "%1"=="test-stability" goto test-stability
if "%1"=="build" goto build
if "%1"=="version" goto version
if "%1"=="clean" goto clean
if "%1"=="docs" goto docs
if "%1"=="docs-serve" goto docs-serve
if "%1"=="docs-validate" goto docs-validate
if "%1"=="pre-commit" goto pre-commit
if "%1"=="ci" goto ci
if "%1"=="status" goto status
if "%1"=="env-show" goto env-show
if "%1"=="env-clean" goto env-clean

echo Unknown command: %1
echo Run 'make.bat help' for available commands
goto end

:help
echo 🚀 Pynomaly Development Commands (Windows)
echo.
echo Setup ^& Installation:
echo   make.bat setup          - Initial project setup (install Hatch, create environments)
echo   make.bat install        - Install package in current environment
echo   make.bat dev-install    - Install package in development mode with all dependencies
echo.
echo Development ^& Quality:
echo   make.bat lint           - Run all code quality checks (style, type, format)
echo   make.bat format         - Auto-format code (ruff, black, isort)
echo   make.bat style          - Check code style without fixing
echo   make.bat typing         - Run type checking with mypy
echo.
echo Testing:
echo   make.bat test           - Run core tests (domain + application)
echo   make.bat test-all       - Run all tests including integration
echo   make.bat test-cov       - Run tests with coverage report
echo   make.bat test-unit      - Run only unit tests
echo   make.bat test-integration - Run only integration tests
echo   make.bat test-stability - Run stability tests for flaky test elimination
echo.
echo Build ^& Package:
echo   make.bat build          - Build wheel and source distribution
echo   make.bat version        - Show current version
echo   make.bat clean          - Clean build artifacts and cache
echo.
echo Pre-commit ^& CI:
echo   make.bat pre-commit     - Install and run pre-commit hooks
echo   make.bat ci             - Run full CI pipeline locally
echo.
echo Documentation ^& Deployment:
echo   make.bat docs           - Build documentation
echo   make.bat docs-serve     - Serve documentation locally
echo   make.bat docs-validate  - Validate documentation links
echo.
echo Utilities:
echo   make.bat status         - Show project status and environment info
echo   make.bat env-show       - Show all Hatch environments
echo   make.bat env-clean      - Clean and recreate environments
goto end

:setup
echo 🚀 Setting up Pynomaly development environment...
hatch --version >nul 2>&1
if errorlevel 1 (
    echo Installing Hatch...
    pip install hatch
)
echo ✅ Hatch installed
echo 📦 Creating Hatch environments...
hatch env create
echo 📋 Available environments:
hatch env show
echo ✅ Setup complete! Run 'make.bat dev-install' next.
goto end

:install
pip install -e .
goto end

:dev-install
echo 📦 Installing Pynomaly in development mode...
hatch env run dev:setup
echo ✅ Development installation complete!
goto end

:lint
echo 🔍 Running code quality checks...
echo 1️⃣ Style checking...
hatch env run lint:style
echo 2️⃣ Type checking...
hatch env run lint:typing
echo ✅ All quality checks passed!
goto end

:format
echo 🎨 Auto-formatting code...
hatch env run lint:fmt
echo ✅ Code formatting complete!
goto end

:style
echo 🎨 Checking code style...
hatch env run lint:style
goto end

:typing
echo 🔎 Running type checking...
hatch env run lint:typing
goto end

:test
echo 🧪 Running core tests...
hatch env run test:run tests/domain/ tests/application/ -v
goto end

:test-all
echo 🧪 Running all tests...
hatch env run test:run -v
goto end

:test-cov
echo 🧪 Running tests with coverage...
hatch env run test:run-cov
echo 📊 Coverage report generated in htmlcov/
goto end

:test-unit
echo 🧪 Running unit tests...
hatch env run test:run tests/domain/ tests/application/ -v
goto end

:test-integration
echo 🧪 Running integration tests...
hatch env run test:run tests/infrastructure/ -v
goto end

:test-stability
echo 🧪 Running stability tests...
hatch env run test:run tests/stability/ -v
echo 🔍 Running flaky test detection...
hatch env run test:run tests/stability/test_flaky_test_elimination.py -v
goto end

:build
echo 📦 Building package...
hatch build --clean
echo 📋 Build artifacts:
dir dist\ 2>nul
goto end

:version
hatch version
goto end

:clean
echo 🧹 Cleaning build artifacts...
hatch env run dev:clean
rd /s /q dist 2>nul
rd /s /q build 2>nul
rd /s /q *.egg-info 2>nul
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
for /r . %%f in (*.pyc) do @if exist "%%f" del "%%f" 2>nul
echo ✅ Cleanup complete!
goto end

:docs
echo 📖 Building documentation...
hatch env run docs:build
echo ✅ Documentation built in site/
goto end

:docs-serve
echo 📜 Serving documentation at http://localhost:8080
hatch env run docs:serve
goto end

:docs-validate
echo 🔍 Validating documentation links...
python scripts/analysis/check_documentation_links.py
goto end

:pre-commit
echo 🔗 Setting up pre-commit hooks...
pip install pre-commit
pre-commit install --install-hooks
echo 🧪 Running pre-commit on all files...
pre-commit run --all-files
echo ✅ Pre-commit setup complete!
goto end

:ci
echo 🚀 Running full CI pipeline locally...
echo 1️⃣ Version check...
hatch version
echo 2️⃣ Code quality...
hatch env run lint:style
hatch env run lint:typing
echo 3️⃣ Core tests...
hatch env run test:run tests/domain/ tests/application/ -v
echo 4️⃣ Build package...
hatch build --clean
echo ✅ Full CI pipeline completed successfully!
goto end

:status
echo 📊 Pynomaly Project Status
echo ==========================
echo Version: 
hatch version
echo Hatch: 
hatch --version
echo Python: 
python --version
echo Git branch: 
git branch --show-current 2>nul
echo.
echo 📋 Environments:
hatch env show --ascii 2>nul
echo.
echo 📦 Build artifacts:
dir dist\ 2>nul
goto end

:env-show
echo 📋 Hatch environments:
hatch env show
goto end

:env-clean
echo 🧹 Cleaning Hatch environments...
hatch env prune
hatch env create
echo ✅ Environments recreated!
goto end

:end
