@echo off
REM Pynomaly Windows Setup - Batch File Launcher
REM This launches the PowerShell setup script

echo Pynomaly Windows Setup
echo ======================
echo.

REM Check if PowerShell is available
powershell -Command "Write-Host 'PowerShell available'" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: PowerShell not found or not accessible
    echo Please ensure PowerShell is installed and accessible
    pause
    exit /b 1
)

echo Starting PowerShell setup script...
echo.

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1" -InstallProfile server

echo.
echo Setup script completed.
pause
