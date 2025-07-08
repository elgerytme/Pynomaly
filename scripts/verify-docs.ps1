# Documentation verification script (PowerShell)
# This script verifies that the documentation can be built and served properly

$ErrorActionPreference = "Stop"

Write-Host "🔍 Verifying documentation setup..." -ForegroundColor Cyan

# Check if hatch is available
try {
    hatch --version | Out-Null
    Write-Host "✅ Hatch is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Hatch is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check if mkdocs.yml exists
if (-not (Test-Path "mkdocs.yml")) {
    Write-Host "⚠️  mkdocs.yml not found in root, checking config/docs/mkdocs.yml..." -ForegroundColor Yellow
    if (Test-Path "config/docs/mkdocs.yml") {
        Write-Host "📋 Copying mkdocs.yml from config/docs/" -ForegroundColor Blue
        Copy-Item "config/docs/mkdocs.yml" "mkdocs.yml"
    }
    else {
        Write-Host "❌ mkdocs.yml not found in either location" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ mkdocs.yml is available" -ForegroundColor Green

# Verify docs environment exists
Write-Host "🔧 Checking docs environment..." -ForegroundColor Cyan
try {
    $envs = hatch env find docs 2>$null
    if (-not $envs) {
        Write-Host "📦 Creating docs environment..." -ForegroundColor Blue
        hatch env create docs
    }
}
catch {
    Write-Host "📦 Creating docs environment..." -ForegroundColor Blue
    hatch env create docs
}

Write-Host "✅ Docs environment is ready" -ForegroundColor Green

# Test documentation build
Write-Host "🔨 Testing documentation build..." -ForegroundColor Cyan
try {
    hatch run docs:build
    Write-Host "✅ Documentation build successful" -ForegroundColor Green
}
catch {
    Write-Host "❌ Documentation build failed" -ForegroundColor Red
    exit 1
}

# Check if site directory exists and contains files
if ((Test-Path "site") -and (Get-ChildItem "site" | Measure-Object).Count -gt 0) {
    Write-Host "✅ Site directory created with content" -ForegroundColor Green
    Write-Host "📊 Generated files:" -ForegroundColor Blue
    Get-ChildItem -Path "site" -Filter "*.html" -Recurse | Select-Object -First 10 | ForEach-Object { Write-Host "  $($_.FullName)" }
}
else {
    Write-Host "❌ Site directory is empty or does not exist" -ForegroundColor Red
    exit 1
}

# Check for key files
$requiredFiles = @("site/index.html", "site/sitemap.xml")
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ $file exists" -ForegroundColor Green
    }
    else {
        Write-Host "❌ $file missing" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "🎉 Documentation verification complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📖 To serve the documentation locally, run:" -ForegroundColor Cyan
Write-Host "   hatch run docs:serve" -ForegroundColor White
Write-Host ""
Write-Host "🚀 The GitHub Pages deployment will automatically trigger on push to main branch" -ForegroundColor Cyan
Write-Host "   when changes are made to docs/, mkdocs.yml, or .github/workflows/deploy-docs.yml" -ForegroundColor White
