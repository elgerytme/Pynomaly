# Distributed Production Environment
# Multi-node deployment with load balancing and service discovery

param(
    [int]$Workers = 3,
    [int]$ApiReplicas = 2,
    [string]$Storage = "postgres",
    [switch]$Consul,
    [switch]$Vault,
    [switch]$Build,
    [switch]$Clean,
    [switch]$Stop,
    [switch]$Help
)

function Show-Usage {
    Write-Host "Usage: .\run-prod-distributed.ps1 [OPTIONS]"
    Write-Host "Options:"
    Write-Host "  -Workers NUM       Number of worker nodes (default: 3)"
    Write-Host "  -ApiReplicas NUM   Number of API replicas (default: 2)"
    Write-Host "  -Storage TYPE      Storage backend (postgres|mongodb|redis-cluster) (default: postgres)"
    Write-Host "  -Consul            Enable service discovery with Consul"
    Write-Host "  -Vault             Enable secrets management with Vault"
    Write-Host "  -Build             Build production image before running"
    Write-Host "  -Clean             Remove existing containers first"
    Write-Host "  -Stop              Stop all distributed containers"
    Write-Host "  -Help              Show this help message"
    exit 1
}

if ($Help) {
    Show-Usage
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir\..\..\..\"
$NetworkName = "anomaly_detection-distributed"
$ImageName = "anomaly_detection:latest"
$EnvFile = Join-Path $ProjectRoot ".env.prod"

function Stop-DistributedContainers {
    Write-Host "Stopping all distributed containers..."
    docker rm -f anomaly_detection-haproxy 2>$null
    docker rm -f anomaly_detection-consul 2>$null
    docker rm -f anomaly_detection-vault 2>$null

    for ($i = 1; $i -le $ApiReplicas; $i++) {
        docker rm -f "anomaly_detection-api-${i}" 2>$null
    }

    for ($i = 1; $i -le $Workers; $i++) {
        docker rm -f "anomaly_detection-worker-${i}" 2>$null
    }

    # Storage containers
    docker rm -f anomaly_detection-postgres-master 2>$null
    docker rm -f anomaly_detection-postgres-replica 2>$null
    docker rm -f anomaly_detection-mongodb-primary 2>$null
    docker rm -f anomaly_detection-mongodb-secondary1 2>$null
    docker rm -f anomaly_detection-mongodb-secondary2 2>$null
    docker rm -f anomaly_detection-redis-master 2>$null
    docker rm -f anomaly_detection-redis-replica1 2>$null
    docker rm -f anomaly_detection-redis-replica2 2>$null

    Write-Host "All distributed containers stopped."
    exit 0
}

if ($Stop) {
    Stop-DistributedContainers
}

# Create network
docker network create $NetworkName --driver bridge 2>$null

# Clean up if requested
if ($Clean) {
    Write-Host "Cleaning up existing distributed containers..."
    Stop-DistributedContainers
}

# Build production image if requested
if ($Build) {
    Write-Host "Building production image..."
    docker build -t $ImageName -f "$ProjectRoot\Dockerfile.hardened" $ProjectRoot
}

# Start service discovery with Consul
if ($Consul) {
    Write-Host "Starting Consul for service discovery..."
    docker run -d `
        --name anomaly_detection-consul `
        --network $NetworkName `
        -p 8500:8500 `
        -v consul-data:/consul/data `
        consul:latest agent -dev -client=0.0.0.0 -ui
}

# Start secrets management with Vault
if ($Vault) {
    Write-Host "Starting Vault for secrets management..."
    docker run -d `
        --name anomaly_detection-vault `
        --network $NetworkName `
        -p 8200:8200 `
        --cap-add=IPC_LOCK `
        -e VAULT_DEV_ROOT_TOKEN_ID=dev-root-token `
        -v vault-data:/vault/data `
        vault:latest
}

# Start distributed storage based on type
switch ($Storage) {
    "postgres" {
        Write-Host "Starting PostgreSQL cluster..."
        # Master
        docker run -d `
            --name anomaly_detection-postgres-master `
            --network $NetworkName `
            -e POSTGRES_DB=anomaly_detection `
            -e POSTGRES_USER=anomaly_detection `
            -e POSTGRES_PASSWORD=prod_password `
            -e POSTGRES_REPLICATION_MODE=master `
            -e POSTGRES_REPLICATION_USER=replicator `
            -e POSTGRES_REPLICATION_PASSWORD=replicator_password `
            -p 5432:5432 `
            -v postgres-master-data:/var/lib/postgresql/data `
            bitnami/postgresql:15

        # Replica
        docker run -d `
            --name anomaly_detection-postgres-replica `
            --network $NetworkName `
            -e POSTGRES_REPLICATION_MODE=slave `
            -e POSTGRES_REPLICATION_USER=replicator `
            -e POSTGRES_REPLICATION_PASSWORD=replicator_password `
            -e POSTGRES_MASTER_HOST=anomaly_detection-postgres-master `
            -e POSTGRES_MASTER_PORT_NUMBER=5432 `
            -p 5433:5432 `
            -v postgres-replica-data:/var/lib/postgresql/data `
            bitnami/postgresql:15
    }

    "mongodb" {
        Write-Host "Starting MongoDB replica set..."
        # Primary
        docker run -d `
            --name anomaly_detection-mongodb-primary `
            --network $NetworkName `
            -e MONGODB_REPLICA_SET_MODE=primary `
            -e MONGODB_REPLICA_SET_NAME=rs0 `
            -e MONGODB_ROOT_PASSWORD=prod_password `
            -p 27017:27017 `
            -v mongodb-primary-data:/bitnami/mongodb `
            bitnami/mongodb:7.0

        # Secondary nodes
        for ($i = 1; $i -le 2; $i++) {
            docker run -d `
                --name "anomaly_detection-mongodb-secondary${i}" `
                --network $NetworkName `
                -e MONGODB_REPLICA_SET_MODE=secondary `
                -e MONGODB_REPLICA_SET_NAME=rs0 `
                -e MONGODB_PRIMARY_HOST=anomaly_detection-mongodb-primary `
                -e MONGODB_PRIMARY_ROOT_PASSWORD=prod_password `
                -p "$(27017 + $i):27017" `
                -v "mongodb-secondary${i}-data:/bitnami/mongodb" `
                bitnami/mongodb:7.0
        }
    }

    "redis-cluster" {
        Write-Host "Starting Redis cluster..."
        # Master
        docker run -d `
            --name anomaly_detection-redis-master `
            --network $NetworkName `
            -p 6379:6379 `
            -v redis-master-data:/data `
            redis:7-alpine redis-server --appendonly yes

        # Replicas
        for ($i = 1; $i -le 2; $i++) {
            docker run -d `
                --name "anomaly_detection-redis-replica${i}" `
                --network $NetworkName `
                -p "$(6379 + $i):6379" `
                -v "redis-replica${i}-data:/data" `
                redis:7-alpine redis-server --appendonly yes --replicaof anomaly_detection-redis-master 6379
        }
    }
}

# Wait for storage services
Write-Host "Waiting for storage services to be ready..."
Start-Sleep -Seconds 15

# Start API replicas
Write-Host "Starting $ApiReplicas API replicas..."
for ($i = 1; $i -le $ApiReplicas; $i++) {
    $ApiPort = 8000 + $i - 1
    Write-Host "Starting API replica $i on port $ApiPort..."

    docker run -d `
        --name "anomaly_detection-api-${i}" `
        --network $NetworkName `
        --restart unless-stopped `
        --env-file $EnvFile `
        -p "${ApiPort}:8000" `
        -v "${ProjectRoot}\storage:/app/storage" `
        -v "${ProjectRoot}\logs:/app/logs" `
        -e PYTHONPATH=/app/src `
        -e ENVIRONMENT=production `
        -e DEBUG=false `
        -e LOG_LEVEL=INFO `
        -e API_REPLICA_ID=$i `
        -e STORAGE_TYPE=$Storage `
        --health-cmd "curl -f http://localhost:8000/health || exit 1" `
        --health-interval 30s `
        --health-timeout 10s `
        --health-retries 3 `
        $ImageName `
        poetry run gunicorn anomaly_detection.presentation.api:app `
        --bind 0.0.0.0:8000 `
        --workers 2 `
        --worker-class uvicorn.workers.UvicornWorker `
        --worker-connections 500 `
        --max-requests 1000 `
        --timeout 30
}

# Start worker nodes
Write-Host "Starting $Workers worker nodes..."
for ($i = 1; $i -le $Workers; $i++) {
    Write-Host "Starting worker node $i..."

    docker run -d `
        --name "anomaly_detection-worker-${i}" `
        --network $NetworkName `
        --restart unless-stopped `
        --env-file $EnvFile `
        -v "${ProjectRoot}\storage:/app/storage" `
        -v "${ProjectRoot}\logs:/app/logs" `
        -e PYTHONPATH=/app/src `
        -e ENVIRONMENT=production `
        -e DEBUG=false `
        -e LOG_LEVEL=INFO `
        -e WORKER_ID=$i `
        -e STORAGE_TYPE=$Storage `
        --memory 4g `
        --cpus 2.0 `
        $ImageName `
        poetry run python -m anomaly_detection.infrastructure.distributed.worker `
        --worker-id $i `
        --concurrency 4
}

# Start HAProxy load balancer
Write-Host "Starting HAProxy load balancer..."
$HaproxyConfig = @"
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog

frontend api_frontend
    bind *:80
    default_backend api_servers

backend api_servers
    balance roundrobin
    option httpchk GET /health
"@

for ($i = 1; $i -le $ApiReplicas; $i++) {
    $HaproxyConfig += "`n    server api${i} anomaly_detection-api-${i}:8000 check"
}

$HaproxyConfig | Out-File -FilePath "$env:TEMP\haproxy.cfg" -Encoding ascii

docker run -d `
    --name anomaly_detection-haproxy `
    --network $NetworkName `
    -p 80:80 `
    -v "${env:TEMP}\haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg" `
    haproxy:2.8-alpine

Write-Host "Distributed production deployment completed!"
Write-Host "Load Balancer: http://localhost:80"
Write-Host "API Replicas: $ApiReplicas"
Write-Host "Worker Nodes: $Workers"
Write-Host "Storage: $Storage"

if ($Consul) {
    Write-Host "Consul UI: http://localhost:8500"
}
if ($Vault) {
    Write-Host "Vault UI: http://localhost:8200"
}

Write-Host ""
Write-Host "Use -Stop to stop all containers"
