# Fix Bash Shebangs for Windows Compatibility
# This script replaces hardcoded /bin/bash shebangs with portable versions

param(
    [switch]$DryRun = $false
)

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

Write-Info "Starting shebang fix for Windows compatibility..."

# Find all shell scripts
$bashScripts = Get-ChildItem -Path "scripts" -Filter "*.sh" -Recurse

if ($bashScripts.Count -eq 0) {
    Write-Warning-Custom "No shell scripts found in scripts directory"
    exit 0
}

Write-Info "Found $($bashScripts.Count) shell scripts to process"

foreach ($script in $bashScripts) {
    try {
        $content = Get-Content -Path $script.FullName -Raw
        $originalContent = $content
        
        # Replace hardcoded bash shebangs with portable versions
        $content = $content -replace '^#!/bin/bash', '#!/usr/bin/env bash'
        $content = $content -replace '^#!/bin/sh', '#!/usr/bin/env sh'
        
        if ($content -ne $originalContent) {
            if ($DryRun) {
                Write-Info "Would fix shebang in: $($script.FullName)"
            } else {
                Set-Content -Path $script.FullName -Value $content -NoNewline
                Write-Success "Fixed shebang in: $($script.FullName)"
            }
        } else {
            Write-Info "No changes needed in: $($script.FullName)"
        }
    }
    catch {
        Write-Warning-Custom "Failed to process $($script.FullName): $($_.Exception.Message)"
    }
}

if ($DryRun) {
    Write-Info "Dry run completed. Use without -DryRun to apply changes."
} else {
    Write-Success "Shebang fixes applied successfully"
}
