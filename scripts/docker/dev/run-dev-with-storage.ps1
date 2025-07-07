# Development Environment with Storage Infrastructure
# Includes PostgreSQL, Redis, and MinIO for complete development setup

param(
    [string]$Port = "8000",
    [string]$Storage = "all",
    [switch]$Build,
    [switch]$Clean,
    [switch]$Stop,
    [switch]$Help
)

function Show-Usage {
    Write-Host "Usage: .\run-dev-with-storage.ps1 [OPTIONS]"
    Write-Host "Options:"
    Write-Host "  -Port PORT         Host port to bind (default: 8000)"
    Write-Host "  -Storage TYPE      Storage backend (postgres|redis|minio|all) (default: all)"
    Write-Host "  -Build             Build image before running"
    Write-Host "  -Clean             Remove existing containers first"
    Write-Host "  -Stop              Stop all containers"
    Write-Host "  -Help              Show this help message"
    exit 1
}

if ($Help) {
    Show-Usage
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir\..\..\..\"
$NetworkName = "pynomaly-dev"
$ContainerName = "pynomaly-dev"
$ImageName = "pynomaly:dev"
$EnvFile = Join-Path $ProjectRoot ".env.dev"

# Storage containers
$PostgresContainer = "pynomaly-postgres-dev"
$RedisContainer = "pynomaly-redis-dev"
$MinioContainer = "pynomaly-minio-dev"

function Stop-Containers {
    Write-Host "Stopping all containers..."
    docker rm -f $ContainerName 2>$null
    docker rm -f $PostgresContainer 2>$null
    docker rm -f $RedisContainer 2>$null
    docker rm -f $MinioContainer 2>$null
    Write-Host "All containers stopped."
    exit 0
}

if ($Stop) {
    Stop-Containers
}

# Create network
docker network create $NetworkName 2>$null

# Clean up if requested
if ($Clean) {
    Write-Host "Cleaning up existing containers..."
    docker rm -f $ContainerName 2>$null
    docker rm -f $PostgresContainer 2>$null
    docker rm -f $RedisContainer 2>$null
    docker rm -f $MinioContainer 2>$null
}

# Start PostgreSQL if needed
if ($Storage -eq "postgres" -or $Storage -eq "all") {
    Write-Host "Starting PostgreSQL..."
    docker run -d `
        --name $PostgresContainer `
        --network $NetworkName `
        -e POSTGRES_DB=pynomaly_dev `
        -e POSTGRES_USER=pynomaly `
        -e POSTGRES_PASSWORD=dev_password `
        -p 5432:5432 `
        -v pynomaly-postgres-dev:/var/lib/postgresql/data `
        postgres:15-alpine
}

# Start Redis if needed
if ($Storage -eq "redis" -or $Storage -eq "all") {
    Write-Host "Starting Redis..."
    docker run -d `
        --name $RedisContainer `
        --network $NetworkName `
        -p 6379:6379 `
        -v pynomaly-redis-dev:/data `
        redis:7-alpine redis-server --appendonly yes
}

# Start MinIO if needed
if ($Storage -eq "minio" -or $Storage -eq "all") {
    Write-Host "Starting MinIO..."
    docker run -d `
        --name $MinioContainer `
        --network $NetworkName `
        -e MINIO_ROOT_USER=minioadmin `
        -e MINIO_ROOT_PASSWORD=minioadmin123 `
        -p 9000:9000 `
        -p 9001:9001 `
        -v pynomaly-minio-dev:/data `
        minio/minio server /data --console-address ":9001"
}

# Wait for services
Write-Host "Waiting for services to be ready..."
Start-Sleep -Seconds 10

# Build image if requested
if ($Build) {
    Write-Host "Building development image..."
    docker build -t $ImageName -f "$ProjectRoot\Dockerfile" $ProjectRoot
}

Write-Host "Starting Pynomaly development environment with storage..."
Write-Host "Container: $ContainerName"
Write-Host "Port: $Port"
Write-Host "Storage: $Storage"

# Prepare environment variables
$EnvVars = @()
if ($Storage -eq "postgres" -or $Storage -eq "all") {
    $EnvVars += "-e", "DATABASE_URL=postgresql://pynomaly:dev_password@${PostgresContainer}:5432/pynomaly_dev"
}
if ($Storage -eq "redis" -or $Storage -eq "all") {
    $EnvVars += "-e", "REDIS_URL=redis://${RedisContainer}:6379/0"
}
if ($Storage -eq "minio" -or $Storage -eq "all") {
    $EnvVars += "-e", "MINIO_ENDPOINT=${MinioContainer}:9000"
    $EnvVars += "-e", "MINIO_ACCESS_KEY=minioadmin"
    $EnvVars += "-e", "MINIO_SECRET_KEY=minioadmin123"
}

# Run the main application container
$Args = @(
    "run", "-it", "--rm",
    "--name", $ContainerName,
    "--network", $NetworkName,
    "--env-file", $EnvFile,
    "-p", "${Port}:8000",
    "-v", "${ProjectRoot}:/app",
    "-v", "${ProjectRoot}\.venv:/app/.venv",
    "-v", "${ProjectRoot}\storage:/app/storage",
    "-e", "PYTHONPATH=/app/src",
    "-e", "ENVIRONMENT=development",
    "-e", "DEBUG=true",
    "-e", "LOG_LEVEL=DEBUG",
    "-e", "RELOAD=true"
) + $EnvVars + @(
    $ImageName,
    "poetry", "run", "uvicorn", "pynomaly.presentation.api:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--reload",
    "--reload-dir", "/app/src"
)

& docker @Args

Write-Host "Development environment stopped."
