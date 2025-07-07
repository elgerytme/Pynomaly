# Test Environment - Docker Run Scripts
# Isolated test environment with test database and services

param(
    [string]$Type = "unit",
    [switch]$Coverage,
    [switch]$Parallel,
    [switch]$Build,
    [switch]$Clean,
    [string]$Storage,
    [switch]$Help
)

function Show-Usage {
    Write-Host "Usage: .\run-test.ps1 [OPTIONS]"
    Write-Host "Options:"
    Write-Host "  -Type TYPE         Test type (unit|integration|e2e|all) (default: unit)"
    Write-Host "  -Coverage          Run with coverage reporting"
    Write-Host "  -Parallel          Run tests in parallel"
    Write-Host "  -Build             Build test image before running"
    Write-Host "  -Clean             Remove existing containers first"
    Write-Host "  -Storage TYPE      Include storage services (postgres|redis|all)"
    Write-Host "  -Help              Show this help message"
    exit 1
}

if ($Help) {
    Show-Usage
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir\..\..\..\"
$NetworkName = "pynomaly-test"
$ContainerName = "pynomaly-test"
$ImageName = "pynomaly:test"
$EnvFile = Join-Path $ProjectRoot ".env.test"

# Create network
docker network create $NetworkName 2>$null

# Clean up if requested
if ($Clean) {
    Write-Host "Cleaning up existing test containers..."
    docker rm -f $ContainerName 2>$null
    docker rm -f "pynomaly-postgres-test" 2>$null
    docker rm -f "pynomaly-redis-test" 2>$null
}

# Start test storage services if needed
if ($Storage) {
    if ($Storage -eq "postgres" -or $Storage -eq "all") {
        Write-Host "Starting test PostgreSQL..."
        docker run -d `
            --name "pynomaly-postgres-test" `
            --network $NetworkName `
            -e POSTGRES_DB=pynomaly_test `
            -e POSTGRES_USER=test_user `
            -e POSTGRES_PASSWORD=test_password `
            -p 5433:5432 `
            postgres:15-alpine
    }

    if ($Storage -eq "redis" -or $Storage -eq "all") {
        Write-Host "Starting test Redis..."
        docker run -d `
            --name "pynomaly-redis-test" `
            --network $NetworkName `
            -p 6380:6379 `
            redis:7-alpine
    }

    Write-Host "Waiting for test services to be ready..."
    Start-Sleep -Seconds 5
}

# Build test image if requested
if ($Build) {
    Write-Host "Building test image..."
    docker build -t $ImageName -f "$ProjectRoot\Dockerfile" $ProjectRoot
}

# Prepare test command
$TestCmd = "poetry run pytest"

switch ($Type) {
    "unit" { $TestCmd += " tests/unit tests/domain" }
    "integration" { $TestCmd += " tests/integration tests/infrastructure" }
    "e2e" { $TestCmd += " tests/e2e tests/presentation" }
    "all" { }
    default {
        Write-Host "Unknown test type: $Type"
        exit 1
    }
}

# Add coverage if requested
if ($Coverage) {
    $TestCmd += " --cov=pynomaly --cov-report=html --cov-report=xml --cov-report=term"
}

# Add parallel execution if requested
if ($Parallel) {
    $TestCmd += " -n auto"
}

# Add test-specific options
$TestCmd += " -v --tb=short --strict-markers"

Write-Host "Running tests: $Type"
Write-Host "Command: $TestCmd"

# Prepare environment variables
$EnvVars = @()
if ($Storage -eq "postgres" -or $Storage -eq "all") {
    $EnvVars += "-e", "DATABASE_URL=postgresql://test_user:test_password@pynomaly-postgres-test:5432/pynomaly_test"
}
if ($Storage -eq "redis" -or $Storage -eq "all") {
    $EnvVars += "-e", "REDIS_URL=redis://pynomaly-redis-test:6379/0"
}

# Run tests in container
$Args = @(
    "run", "--rm",
    "--name", $ContainerName,
    "--network", $NetworkName,
    "--env-file", $EnvFile,
    "-v", "${ProjectRoot}:/app",
    "-v", "${ProjectRoot}\htmlcov:/app/htmlcov",
    "-v", "${ProjectRoot}\coverage.xml:/app/coverage.xml",
    "-e", "PYTHONPATH=/app/src",
    "-e", "ENVIRONMENT=test",
    "-e", "DEBUG=false",
    "-e", "LOG_LEVEL=WARNING"
) + $EnvVars + @(
    $ImageName,
    "bash", "-c", $TestCmd
)

& docker @Args

Write-Host "Tests completed."

# Clean up test storage services
if ($Storage) {
    Write-Host "Cleaning up test services..."
    docker rm -f "pynomaly-postgres-test" 2>$null
    docker rm -f "pynomaly-redis-test" 2>$null
}
