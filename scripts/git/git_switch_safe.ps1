param(
    [Parameter(Mandatory=$true)]
    [string]$Name
)

# Check for uncommitted changes
$hasUncommittedChanges = git diff-index --quiet HEAD --; $LASTEXITCODE -ne 0
if ($hasUncommittedChanges) {
    Write-Host "Error: You have uncommitted changes." -ForegroundColor Red
    Write-Host "Please commit or stash your changes before switching branches." -ForegroundColor Yellow
    exit 1
}

# Switch to branch
Write-Host "Switching to branch '$Name'..." -ForegroundColor Blue
git switch $Name

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to switch to branch '$Name'" -ForegroundColor Red
    exit 1
}

Write-Host "Switched to branch '$Name' successfully!" -ForegroundColor Green
