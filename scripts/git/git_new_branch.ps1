param(
    [Parameter(Mandatory=$true)]
    [string]$Type,
    
    [Parameter(Mandatory=$true)]
    [string]$Name
)

# Validate branch type
$validTypes = @("feature", "bugfix", "hotfix", "release", "chore", "docs")
if ($Type -notin $validTypes) {
    Write-Host "Error: Invalid branch type '$Type'." -ForegroundColor Red
    Write-Host "Valid types: $($validTypes -join ', ')" -ForegroundColor Yellow
    exit 1
}

# Validate branch name format
if ($Name -notmatch '^[a-z0-9-]+$') {
    Write-Host "Error: Invalid branch name '$Name'." -ForegroundColor Red
    Write-Host "Branch names must contain only lowercase letters, numbers, and hyphens." -ForegroundColor Yellow
    exit 1
}

$branchName = "$Type/$Name"

# Check if branch already exists
$branchExists = git show-ref --verify --quiet "refs/heads/$branchName"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Error: Branch '$branchName' already exists." -ForegroundColor Red
    exit 1
}

Write-Host "Creating and switching to new branch '$branchName'..." -ForegroundColor Green
git checkout -b $branchName
