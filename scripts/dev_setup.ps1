# Development environment setup script for Windows
# This script wraps the two main commands for setting up the development environment

Write-Host "Setting up Pynomaly development environment..." -ForegroundColor Green

# Create the dev environment
Write-Host "Creating development environment..." -ForegroundColor Yellow
hatch env create dev

# Run the development setup  
Write-Host "Installing Pynomaly in editable mode and setting up pre-commit hooks..." -ForegroundColor Yellow
hatch run dev:setup

Write-Host "Development environment setup complete!" -ForegroundColor Green
Write-Host "You can now use 'hatch run dev:' to execute commands in the development environment." -ForegroundColor Cyan
