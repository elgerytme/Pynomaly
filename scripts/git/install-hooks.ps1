#!/usr/bin/env pwsh
# PowerShell script to install Git hooks for Pynomaly project
# Cross-platform alternative to `make git-hooks`

Write-Host "Installing Git hooks..." -ForegroundColor Green

# Set Git hooks path
Write-Host "Setting Git hooks path to scripts/git/hooks/" -ForegroundColor Yellow
git config core.hooksPath scripts/git/hooks

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Git hooks installed successfully!" -ForegroundColor Green
    Write-Host "Hooks available:" -ForegroundColor Cyan
    Write-Host "  - pre-commit  -> branch naming lint + partial linting" -ForegroundColor Cyan
    Write-Host "  - pre-push    -> run unit tests" -ForegroundColor Cyan
    Write-Host "  - post-checkout -> remind to restart long-running services" -ForegroundColor Cyan
    Write-Host "🎯 Cross-platform installation complete!" -ForegroundColor Green
} else {
    Write-Host "Failed to install Git hooks" -ForegroundColor Red
    exit 1
}
