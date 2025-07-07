# Production Environment - Docker Run Scripts
# Optimized production deployment with security and monitoring

param(
    [string]$Port = "80",
    [int]$Replicas = 1,
    [string]$Image = "pynomaly:latest",
    [switch]$SSL,
    [switch]$Monitoring,
    [switch]$Logging,
    [switch]$Build,
    [switch]$Clean,
    [switch]$Stop,
    [switch]$Help
)

function Show-Usage {
    Write-Host "Usage: .\run-prod.ps1 [OPTIONS]"
    Write-Host "Options:"
    Write-Host "  -Port PORT         Host port to bind (default: 80)"
    Write-Host "  -Replicas NUM      Number of replicas (default: 1)"
    Write-Host "  -Image IMAGE       Docker image name (default: pynomaly:latest)"
    Write-Host "  -SSL               Enable SSL/TLS (requires certificates)"
    Write-Host "  -Monitoring        Include monitoring stack"
    Write-Host "  -Logging           Include centralized logging"
    Write-Host "  -Build             Build production image before running"
    Write-Host "  -Clean             Remove existing containers first"
    Write-Host "  -Stop              Stop all production containers"
    Write-Host "  -Help              Show this help message"
    exit 1
}

if ($Help) {
    Show-Usage
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir\..\..\..\"
$NetworkName = "pynomaly-prod"
$ContainerName = "pynomaly-prod"
$EnvFile = Join-Path $ProjectRoot ".env.prod"

function Stop-ProductionContainers {
    Write-Host "Stopping all production containers..."
    for ($i = 1; $i -le $Replicas; $i++) {
        docker rm -f "${ContainerName}-${i}" 2>$null
    }
    docker rm -f "pynomaly-nginx-prod" 2>$null
    docker rm -f "pynomaly-prometheus-prod" 2>$null
    docker rm -f "pynomaly-grafana-prod" 2>$null
    docker rm -f "pynomaly-elasticsearch-prod" 2>$null
    docker rm -f "pynomaly-logstash-prod" 2>$null
    docker rm -f "pynomaly-kibana-prod" 2>$null
    Write-Host "All production containers stopped."
    exit 0
}

if ($Stop) {
    Stop-ProductionContainers
}

# Create network
docker network create $NetworkName 2>$null

# Clean up if requested
if ($Clean) {
    Write-Host "Cleaning up existing production containers..."
    Stop-ProductionContainers
}

# Build production image if requested
if ($Build) {
    Write-Host "Building production image..."
    docker build -t $Image -f "$ProjectRoot\Dockerfile.hardened" $ProjectRoot
}

# Start monitoring stack if requested
if ($Monitoring) {
    Write-Host "Starting monitoring stack..."
    
    # Start Prometheus
    docker run -d `
        --name "pynomaly-prometheus-prod" `
        --network $NetworkName `
        --restart unless-stopped `
        -p 9090:9090 `
        -v "${ProjectRoot}\docker\monitoring\prometheus-prod.yml:/etc/prometheus/prometheus.yml" `
        -v prometheus-prod-data:/prometheus `
        prom/prometheus:latest `
        --config.file=/etc/prometheus/prometheus.yml `
        --storage.tsdb.path=/prometheus `
        --web.console.libraries=/usr/share/prometheus/console_libraries `
        --web.console.templates=/usr/share/prometheus/consoles `
        --storage.tsdb.retention.time=30d `
        --web.enable-lifecycle

    # Start Grafana
    docker run -d `
        --name "pynomaly-grafana-prod" `
        --network $NetworkName `
        --restart unless-stopped `
        -p 3000:3000 `
        -e GF_SECURITY_ADMIN_PASSWORD_FILE=/run/secrets/grafana_password `
        -v grafana-prod-data:/var/lib/grafana `
        -v "${ProjectRoot}\docker\monitoring\grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml" `
        grafana/grafana:latest
}

# Start logging stack if requested
if ($Logging) {
    Write-Host "Starting logging stack..."
    
    # Start Elasticsearch
    docker run -d `
        --name "pynomaly-elasticsearch-prod" `
        --network $NetworkName `
        --restart unless-stopped `
        -p 9200:9200 `
        -e "discovery.type=single-node" `
        -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" `
        -v elasticsearch-prod-data:/usr/share/elasticsearch/data `
        elasticsearch:8.11.0

    # Start Logstash
    docker run -d `
        --name "pynomaly-logstash-prod" `
        --network $NetworkName `
        --restart unless-stopped `
        -p 5044:5044 `
        -v "${ProjectRoot}\docker\logging\logstash.conf:/usr/share/logstash/pipeline/logstash.conf" `
        logstash:8.11.0

    # Start Kibana
    docker run -d `
        --name "pynomaly-kibana-prod" `
        --network $NetworkName `
        --restart unless-stopped `
        -p 5601:5601 `
        -e ELASTICSEARCH_HOSTS=http://pynomaly-elasticsearch-prod:9200 `
        kibana:8.11.0
}

# Start Nginx reverse proxy
Write-Host "Starting Nginx reverse proxy..."
$NginxConfig = "nginx-prod.conf"
if ($SSL) {
    $NginxConfig = "nginx-prod-ssl.conf"
}

$NginxArgs = @(
    "run", "-d",
    "--name", "pynomaly-nginx-prod",
    "--network", $NetworkName,
    "--restart", "unless-stopped",
    "-p", "${Port}:80",
    "-v", "${ProjectRoot}\docker\nginx\${NginxConfig}:/etc/nginx/nginx.conf"
)

if ($SSL) {
    $NginxArgs += "-p", "443:443"
    $NginxArgs += "-v", "${ProjectRoot}\certs:/etc/nginx/certs"
}

$NginxArgs += "nginx:alpine"

& docker @NginxArgs

# Wait for services
Write-Host "Waiting for services to be ready..."
Start-Sleep -Seconds 10

# Start application replicas
Write-Host "Starting $Replicas application replicas..."
for ($i = 1; $i -le $Replicas; $i++) {
    $AppPort = 8000 + $i - 1
    Write-Host "Starting replica $i on port $AppPort..."
    
    $Args = @(
        "run", "-d",
        "--name", "${ContainerName}-${i}",
        "--network", $NetworkName,
        "--restart", "unless-stopped",
        "--env-file", $EnvFile,
        "-p", "${AppPort}:8000",
        "-v", "${ProjectRoot}\storage:/app/storage",
        "-v", "${ProjectRoot}\logs:/app/logs",
        "-e", "PYTHONPATH=/app/src",
        "-e", "ENVIRONMENT=production",
        "-e", "DEBUG=false",
        "-e", "LOG_LEVEL=INFO",
        "-e", "REPLICA_ID=$i",
        "-e", "PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc",
        "--security-opt", "no-new-privileges:true",
        "--read-only",
        "--tmpfs", "/tmp",
        "--tmpfs", "/app/logs",
        "--user", "1000:1000",
        "--memory", "2g",
        "--cpus", "1.0",
        "--health-cmd", "curl -f http://localhost:8000/health || exit 1",
        "--health-interval", "30s",
        "--health-timeout", "10s",
        "--health-retries", "3",
        $Image,
        "poetry", "run", "gunicorn", "pynomaly.presentation.api:app",
        "--bind", "0.0.0.0:8000",
        "--workers", "4",
        "--worker-class", "uvicorn.workers.UvicornWorker",
        "--worker-connections", "1000",
        "--max-requests", "1000",
        "--max-requests-jitter", "100",
        "--timeout", "30",
        "--keep-alive", "2",
        "--access-logfile", "/app/logs/access.log",
        "--error-logfile", "/app/logs/error.log",
        "--log-level", "info"
    )
    
    & docker @Args
}

Write-Host "Production deployment completed!"
Write-Host "Application: http://localhost:${Port}"
if ($SSL) {
    Write-Host "HTTPS: https://localhost:443"
}
if ($Monitoring) {
    Write-Host "Monitoring: http://localhost:3000"
    Write-Host "Metrics: http://localhost:9090"
}
if ($Logging) {
    Write-Host "Logs: http://localhost:5601"
}
Write-Host "Replicas: $Replicas"
Write-Host ""
Write-Host "Use -Stop to stop all containers"
