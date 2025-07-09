# Pynomaly Docker Helper Scripts for Windows PowerShell
# Cross-platform Docker development and production utilities

param(
    [Parameter(Position = 0)]
    [string]$Command = "help",
    [string]$Target = "development",
    [switch]$Force,
    [switch]$Verbose
)

# Set error handling
$ErrorActionPreference = "Stop"

# Color output functions
function Write-Info($message) {
    Write-Host "ℹ️ $message" -ForegroundColor Cyan
}

function Write-Success($message) {
    Write-Host "✅ $message" -ForegroundColor Green
}

function Write-Warning($message) {
    Write-Host "⚠️ $message" -ForegroundColor Yellow
}

function Write-Error($message) {
    Write-Host "❌ $message" -ForegroundColor Red
}

function Write-Title($message) {
    Write-Host "`n🚀 $message" -ForegroundColor Magenta
    Write-Host "=" * (4 + $message.Length) -ForegroundColor Gray
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Success "Docker found: $dockerVersion"
    }
    catch {
        Write-Error "Docker not found. Please install Docker Desktop."
        exit 1
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker-compose --version
        Write-Success "Docker Compose found: $composeVersion"
    }
    catch {
        Write-Error "Docker Compose not found. Please install Docker Compose."
        exit 1
    }
    
    # Check if Docker is running
    try {
        docker ps | Out-Null
        Write-Success "Docker daemon is running"
    }
    catch {
        Write-Error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    }
}

# Build Docker images
function Build-DockerImages {
    param(
        [string]$ImageTarget = "all"
    )
    
    Write-Title "Building Docker Images"
    Test-Prerequisites
    
    $images = @()
    
    switch ($ImageTarget) {
        "development" {
            $images = @("development")
        }
        "production" {
            $images = @("production")
        }
        "slim" {
            $images = @("slim")
        }
        "all" {
            $images = @("development", "production", "slim")
        }
        default {
            Write-Error "Invalid target: $ImageTarget. Use: development, production, slim, or all"
            exit 1
        }
    }
    
    foreach ($target in $images) {
        Write-Info "Building $target image..."
        $imageName = "pynomaly:$target"
        
        if ($target -eq "development") {
            $imageName = "pynomaly:dev"
        }
        elseif ($target -eq "production") {
            $imageName = "pynomaly:prod"
        }
        elseif ($target -eq "slim") {
            $imageName = "pynomaly:slim"
        }
        
        try {
            docker build --target $target -t $imageName -f Dockerfile.multi-stage . --progress=plain
            Write-Success "Built $imageName successfully"
        }
        catch {
            Write-Error "Failed to build $imageName"
            exit 1
        }
    }
    
    Write-Success "All Docker images built successfully!"
    
    # Show built images
    Write-Info "Built images:"
    docker images | Select-String "pynomaly"
}

# Run development environment
function Start-DevEnvironment {
    Write-Title "Starting Development Environment"
    Test-Prerequisites
    
    Write-Info "Starting development environment with hot-reload..."
    
    try {
        docker-compose -f docker-compose.local.yml up --build -d
        Write-Success "Development environment started successfully!"
        
        Write-Info "Services available:"
        Write-Host "  • API Server: http://localhost:8000" -ForegroundColor Green
        Write-Host "  • Web UI: http://localhost:8080" -ForegroundColor Green
        Write-Host "  • Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor Green
        Write-Host "  • Prometheus: http://localhost:9090" -ForegroundColor Green
        Write-Host "  • pgAdmin: http://localhost:8081 (admin@pynomaly.dev/admin)" -ForegroundColor Green
        Write-Host "  • Redis Commander: http://localhost:8082" -ForegroundColor Green
        Write-Host "  • Jaeger: http://localhost:16686" -ForegroundColor Green
        
        Write-Info "Use 'Stop-Environment' to stop all services"
        
    }
    catch {
        Write-Error "Failed to start development environment"
        exit 1
    }
}

# Run production environment
function Start-ProdEnvironment {
    Write-Title "Starting Production Environment"
    Test-Prerequisites
    
    Write-Info "Starting production environment..."
    
    try {
        docker-compose -f deploy/docker/docker-compose.yml up --build -d
        Write-Success "Production environment started successfully!"
        
        Write-Info "Production services available:"
        Write-Host "  • API Server: http://localhost:8000" -ForegroundColor Green
        Write-Host "  • Database: PostgreSQL on port 5432" -ForegroundColor Green
        Write-Host "  • Cache: Redis on port 6379" -ForegroundColor Green
        
    }
    catch {
        Write-Error "Failed to start production environment"
        exit 1
    }
}

# Stop all environments
function Stop-Environment {
    Write-Title "Stopping Docker Environments"
    
    Write-Info "Stopping all Docker containers..."
    
    try {
        docker-compose -f docker-compose.local.yml down
        docker-compose -f deploy/docker/docker-compose.yml down
        Write-Success "All containers stopped successfully!"
    }
    catch {
        Write-Warning "Some containers may not have been stopped properly"
    }
}

# Clean Docker resources
function Clean-DockerResources {
    Write-Title "Cleaning Docker Resources"
    
    if (-not $Force) {
        $confirm = Read-Host "This will remove all unused Docker resources. Continue? (y/N)"
        if ($confirm -ne "y" -and $confirm -ne "Y") {
            Write-Info "Cleanup cancelled"
            return
        }
    }
    
    Write-Info "Cleaning Docker resources..."
    
    try {
        docker system prune -f
        docker image prune -f
        Write-Success "Docker cleanup complete!"
    }
    catch {
        Write-Error "Failed to clean Docker resources"
        exit 1
    }
}

# Show Docker logs
function Show-DockerLogs {
    param(
        [string]$Service = ""
    )
    
    Write-Title "Docker Logs"
    
    try {
        if ($Service) {
            docker-compose -f docker-compose.local.yml logs -f $Service
        }
        else {
            docker-compose -f docker-compose.local.yml logs -f
        }
    }
    catch {
        Write-Error "Failed to show Docker logs"
        exit 1
    }
}

# Show status
function Show-Status {
    Write-Title "Pynomaly Docker Status"
    
    Write-Info "Docker Images:"
    docker images | Select-String "pynomaly" | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
    
    Write-Info "`nRunning Containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Select-String "pynomaly"
    
    Write-Info "`nDocker Compose Services:"
    if (Test-Path "docker-compose.local.yml") {
        docker-compose -f docker-compose.local.yml ps
    }
}

# Show help
function Show-Help {
    Write-Title "Pynomaly Docker Helper - PowerShell Edition"
    
    Write-Host @"
Usage: .\docker-helpers.ps1 [COMMAND] [OPTIONS]

Commands:
  build [target]     Build Docker images (targets: development, production, slim, all)
  start-dev         Start development environment with hot-reload
  start-prod        Start production environment
  stop              Stop all Docker environments
  clean             Clean Docker resources (use -Force to skip confirmation)
  logs [service]    Show Docker logs (optional service name)
  status            Show Docker status and running containers
  help              Show this help message

Examples:
  .\docker-helpers.ps1 build                   # Build all images
  .\docker-helpers.ps1 build development       # Build only development image
  .\docker-helpers.ps1 start-dev               # Start development environment
  .\docker-helpers.ps1 start-prod              # Start production environment
  .\docker-helpers.ps1 stop                    # Stop all environments
  .\docker-helpers.ps1 clean -Force            # Clean without confirmation
  .\docker-helpers.ps1 logs pynomaly-dev       # Show logs for specific service
  .\docker-helpers.ps1 status                  # Show status

Options:
  -Force            Skip confirmation prompts
  -Verbose          Enable verbose output
  -Target           Specify build target (for build command)

Development URLs:
  API Server:       http://localhost:8000
  Web UI:           http://localhost:8080
  Grafana:          http://localhost:3000 (admin/admin)
  Prometheus:       http://localhost:9090
  pgAdmin:          http://localhost:8081 (admin@pynomaly.dev/admin)
  Redis Commander:  http://localhost:8082
  Jaeger:           http://localhost:16686

For more information, visit: https://github.com/pynomaly/pynomaly
"@
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "build" {
        Build-DockerImages -ImageTarget $Target
    }
    "start-dev" {
        Start-DevEnvironment
    }
    "start-prod" {
        Start-ProdEnvironment
    }
    "stop" {
        Stop-Environment
    }
    "clean" {
        Clean-DockerResources
    }
    "logs" {
        Show-DockerLogs -Service $Target
    }
    "status" {
        Show-Status
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error "Unknown command: $Command"
        Show-Help
        exit 1
    }
}
