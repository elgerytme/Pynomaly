# PowerShell script for setting GitHub branch protection rules
# This script configures branch protection for the main branch with required status checks

param(
    [string]$Branch = "main"
)

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if GitHub CLI is installed
try {
    $null = Get-Command gh -ErrorAction Stop
    Write-Status "GitHub CLI found"
} catch {
    Write-Error "GitHub CLI (gh) is not installed. Please install it first."
    exit 1
}

# Check if user is authenticated
try {
    $authStatus = gh auth status 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Not authenticated with GitHub CLI. Please run 'gh auth login' first."
        exit 1
    }
    Write-Status "GitHub CLI authentication verified"
} catch {
    Write-Error "Failed to verify GitHub CLI authentication"
    exit 1
}

# Get repository information
try {
    $repoName = gh repo view --json nameWithOwner --jq '.nameWithOwner'
    $defaultBranch = gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name'
    
    Write-Status "Repository: $repoName"
    Write-Status "Default branch: $defaultBranch"
} catch {
    Write-Error "Failed to get repository information"
    exit 1
}

Write-Status "Configuring branch protection for '$Branch' branch..."

# Use the JSON configuration file
$jsonFile = ".github/branch_protection.json"

try {
    Write-Status "Setting branch protection rules..."
    gh api repos/$repoName/branches/$Branch/protection --method PUT --input $jsonFile

    if ($LASTEXITCODE -eq 0) {
        Write-Status "Branch protection rules successfully configured!"
    } else {
        Write-Error "Failed to configure branch protection rules"
        exit 1
    }
} catch {
    Write-Error "Failed to configure branch protection rules: $_"
    exit 1
}

# Verify the configuration
Write-Status "Verifying branch protection configuration..."
try {
    $protection = gh api repos/$repoName/branches/$Branch/protection --jq '{
        "required_status_checks": .required_status_checks.contexts,
        "required_approving_review_count": .required_pull_request_reviews.required_approving_review_count,
        "dismiss_stale_reviews": .required_pull_request_reviews.dismiss_stale_reviews,
        "allow_force_pushes": .allow_force_pushes.enabled,
        "enforce_admins": .enforce_admins.enabled
    }'
    
    Write-Host "Current branch protection settings:"
    Write-Host $protection
} catch {
    Write-Host "[WARNING] Could not retrieve current protection settings for verification"
}

Write-Status "Branch protection configuration completed successfully!"
Write-Status "Configuration details:"
Write-Host "  - Protected branch: $Branch"
Write-Host "  - Required status checks: CI summary, Quality Gate Summary, Validation Suite Summary"
Write-Host "  - Required approving reviews: 2"
Write-Host "  - Dismiss stale reviews: Yes"
Write-Host "  - Block force pushes: Yes"
Write-Host "  - Require up-to-date before merge: Yes"
Write-Host "  - Enforce for admins: Yes"
