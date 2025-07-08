#!/usr/bin/env pwsh
# Test script for git helper scripts

Write-Host "Testing Git Helper Scripts" -ForegroundColor Green

# Test 1: Create a new branch
Write-Host "`n1. Testing git_new_branch.ps1..." -ForegroundColor Yellow
try {
    & "$PSScriptRoot\git_new_branch.ps1" -Type "feature" -Name "test-validation"
    Write-Host "✅ Branch creation successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Branch creation failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: Switch back to main
Write-Host "`n2. Testing git_switch_safe.ps1..." -ForegroundColor Yellow
try {
    & "$PSScriptRoot\git_switch_safe.ps1" -Name "main"
    Write-Host "✅ Branch switching successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Branch switching failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: Clean up test branch
Write-Host "`n3. Cleaning up test branch..." -ForegroundColor Yellow
try {
    git branch -D feature/test-validation
    Write-Host "✅ Cleanup successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Cleanup failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: Test error handling
Write-Host "`n4. Testing error handling..." -ForegroundColor Yellow
try {
    & "$PSScriptRoot\git_new_branch.ps1" -Type "invalid" -Name "test"
    Write-Host "❌ Error handling test failed - should have rejected invalid type" -ForegroundColor Red
} catch {
    Write-Host "✅ Error handling works correctly" -ForegroundColor Green
}

Write-Host "`n✅ All tests completed!" -ForegroundColor Green
