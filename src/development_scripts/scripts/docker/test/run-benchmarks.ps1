# Benchmark Testing Environment
# Performance and load testing with monitoring

param(
    [string]$Type = "performance",
    [string]$Datasets,
    [string]$Algorithms,
    [switch]$Build,
    [switch]$Clean,
    [switch]$Monitoring,
    [switch]$Help
)

function Show-Usage {
    Write-Host "Usage: .\run-benchmarks.ps1 [OPTIONS]"
    Write-Host "Options:"
    Write-Host "  -Type TYPE         Benchmark type (performance|load|stress|memory) (default: performance)"
    Write-Host "  -Datasets PATH     Custom datasets directory"
    Write-Host "  -Algorithms LIST   Comma-separated algorithm list"
    Write-Host "  -Build             Build benchmark image before running"
    Write-Host "  -Clean             Remove existing containers first"
    Write-Host "  -Monitoring        Include monitoring stack (Prometheus/Grafana)"
    Write-Host "  -Help              Show this help message"
    exit 1
}

if ($Help) {
    Show-Usage
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir\..\..\..\"
$NetworkName = "anomaly_detection-benchmark"
$ContainerName = "anomaly_detection-benchmark"
$ImageName = "anomaly_detection:benchmark"

# Create network
docker network create $NetworkName 2>$null

# Clean up if requested
if ($Clean) {
    Write-Host "Cleaning up existing benchmark containers..."
    docker rm -f $ContainerName 2>$null
    docker rm -f "anomaly_detection-prometheus" 2>$null
    docker rm -f "anomaly_detection-grafana" 2>$null
}

# Start monitoring stack if requested
if ($Monitoring) {
    Write-Host "Starting monitoring stack..."

    # Start Prometheus
    docker run -d `
        --name "anomaly_detection-prometheus" `
        --network $NetworkName `
        -p 9090:9090 `
        -v "${ProjectRoot}\docker\monitoring\prometheus.yml:/etc/prometheus/prometheus.yml" `
        prom/prometheus:latest

    # Start Grafana
    docker run -d `
        --name "anomaly_detection-grafana" `
        --network $NetworkName `
        -p 3000:3000 `
        -e GF_SECURITY_ADMIN_PASSWORD=admin `
        -v grafana-storage:/var/lib/grafana `
        grafana/grafana:latest

    Write-Host "Monitoring available at: http://localhost:3000 (admin/admin)"
}

# Build benchmark image if requested
if ($Build) {
    Write-Host "Building benchmark image..."
    docker build -t $ImageName -f "$ProjectRoot\Dockerfile" $ProjectRoot
}

# Prepare benchmark command based on type
switch ($Type) {
    "performance" { $BenchCmd = "poetry run python -m benchmarks.performance" }
    "load" { $BenchCmd = "poetry run python -m benchmarks.load_testing" }
    "stress" { $BenchCmd = "poetry run python -m benchmarks.stress_testing" }
    "memory" { $BenchCmd = "poetry run python -m benchmarks.memory_profiling" }
    default {
        Write-Host "Unknown benchmark type: $Type"
        exit 1
    }
}

# Add algorithm filter if specified
if ($Algorithms) {
    $BenchCmd += " --algorithms $Algorithms"
}

# Add custom datasets if specified
$DatasetMount = @()
if ($Datasets) {
    $DatasetMount = "-v", "${Datasets}:/app/benchmark_datasets"
    $BenchCmd += " --datasets /app/benchmark_datasets"
}

Write-Host "Running benchmarks: $Type"
Write-Host "Command: $BenchCmd"

# Run benchmarks in container
$Args = @(
    "run", "--rm",
    "--name", $ContainerName,
    "--network", $NetworkName,
    "-v", "${ProjectRoot}:/app",
    "-v", "${ProjectRoot}\benchmarks\results:/app/benchmarks/results"
) + $DatasetMount + @(
    "-e", "PYTHONPATH=/app/src",
    "-e", "ENVIRONMENT=benchmark",
    "-e", "LOG_LEVEL=INFO",
    "-e", "BENCHMARK_TYPE=$Type",
    "--cpus=0.000",
    "--memory=4g",
    $ImageName,
    "bash", "-c", $BenchCmd
)

& docker @Args

Write-Host "Benchmarks completed. Results saved to benchmarks/results/"

# Keep monitoring stack running if requested
if ($Monitoring) {
    Write-Host "Monitoring stack is still running. Use -Clean to stop."
    Write-Host "Prometheus: http://localhost:9090"
    Write-Host "Grafana: http://localhost:3000"
}
