# Development Environment - Docker Run Scripts
# Basic development setup with hot-reload and dev tools

param(
    [string]$Port = "8000",
    [string]$Name = "pynomaly-dev",
    [string]$Image = "pynomaly:dev",
    [string]$EnvFile = ".env.dev",
    [switch]$Build,
    [switch]$Clean,
    [switch]$Help
)

function Show-Usage {
    Write-Host "Usage: .\run-dev.ps1 [OPTIONS]"
    Write-Host "Options:"
    Write-Host "  -Port PORT         Host port to bind (default: 8000)"
    Write-Host "  -Name NAME         Container name (default: pynomaly-dev)"
    Write-Host "  -Image IMAGE       Docker image name (default: pynomaly:dev)"
    Write-Host "  -EnvFile FILE      Environment file path (default: .env.dev)"
    Write-Host "  -Build             Build image before running"
    Write-Host "  -Clean             Remove existing container first"
    Write-Host "  -Help              Show this help message"
    exit 1
}

if ($Help) {
    Show-Usage
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir\..\..\..\"
$NetworkName = "pynomaly-dev"
$ContainerPort = "8000"
$EnvFilePath = Join-Path $ProjectRoot $EnvFile

# Create network if it doesn't exist
docker network create $NetworkName 2>$null

# Clean up existing container if requested
if ($Clean) {
    Write-Host "Cleaning up existing container..."
    docker rm -f $Name 2>$null
}

# Build image if requested
if ($Build) {
    Write-Host "Building development image..."
    docker build -t $Image -f "$ProjectRoot\Dockerfile" $ProjectRoot
}

Write-Host "Starting Pynomaly development environment..."
Write-Host "Container: $Name"
Write-Host "Image: $Image"
Write-Host "Port: $Port -> $ContainerPort"
Write-Host "Network: $NetworkName"

# Run the container
docker run -it --rm `
    --name $Name `
    --network $NetworkName `
    --env-file $EnvFilePath `
    -p "${Port}:${ContainerPort}" `
    -v "${ProjectRoot}:/app" `
    -v "${ProjectRoot}\.venv:/app/.venv" `
    -v "${ProjectRoot}\storage:/app/storage" `
    -e PYTHONPATH=/app/src `
    -e ENVIRONMENT=development `
    -e DEBUG=true `
    -e LOG_LEVEL=DEBUG `
    -e RELOAD=true `
    $Image `
    poetry run uvicorn pynomaly.presentation.api:app `
    --host 0.0.0.0 `
    --port $ContainerPort `
    --reload `
    --reload-dir /app/src

Write-Host "Development server stopped."
