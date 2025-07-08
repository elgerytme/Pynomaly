#!/usr/bin/env bash

# MinIO High Availability Cluster
# Distributed object storage with erasure coding

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default configuration
NETWORK_NAME="pynomaly-minio"
CLUSTER_NAME="pynomaly-minio"
NODES=4
VOLUMES_PER_NODE=2

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --nodes NUM          Number of MinIO nodes (default: 4, min: 4)"
    echo "  --volumes-per-node N Number of volumes per node (default: 2)"
    echo "  --with-nginx         Include Nginx load balancer"
    echo "  --monitoring         Include MinIO monitoring"
    echo "  --backup             Set up automated backups"
    echo "  --clean              Remove existing containers first"
    echo "  --stop               Stop all MinIO containers"
    echo "  -h, --help           Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --volumes-per-node)
            VOLUMES_PER_NODE="$2"
            shift 2
            ;;
        --with-nginx)
            WITH_NGINX=true
            shift
            ;;
        --monitoring)
            MONITORING=true
            shift
            ;;
        --backup)
            BACKUP=true
            shift
            ;;
        --clean)
            CLEAN_CONTAINERS=true
            shift
            ;;
        --stop)
            STOP_CONTAINERS=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate minimum nodes
if [[ $NODES -lt 4 ]]; then
    echo "Error: MinIO distributed mode requires at least 4 nodes"
    exit 1
fi

# Function to stop all MinIO containers
stop_containers() {
    echo "Stopping all MinIO containers..."
    for i in $(seq 1 $NODES); do
        docker rm -f "${CLUSTER_NAME}-node-${i}" 2>/dev/null || true
    done
    docker rm -f "${CLUSTER_NAME}-nginx" 2>/dev/null || true
    docker rm -f "${CLUSTER_NAME}-monitor" 2>/dev/null || true
    docker rm -f "${CLUSTER_NAME}-backup" 2>/dev/null || true
    echo "All MinIO containers stopped."
    exit 0
}

if [[ "$STOP_CONTAINERS" == "true" ]]; then
    stop_containers
fi

# Create network
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Clean up if requested
if [[ "$CLEAN_CONTAINERS" == "true" ]]; then
    echo "Cleaning up existing MinIO containers..."
    stop_containers
fi

echo "Starting MinIO Distributed Cluster..."
echo "Nodes: $NODES"
echo "Volumes per node: $VOLUMES_PER_NODE"

# Generate MinIO credentials
MINIO_ROOT_USER="minioadmin"
MINIO_ROOT_PASSWORD="minioadmin123"

# Build distributed server command
MINIO_VOLUMES=""
for node in $(seq 1 $NODES); do
    for vol in $(seq 1 $VOLUMES_PER_NODE); do
        MINIO_VOLUMES="$MINIO_VOLUMES http://${CLUSTER_NAME}-node-${node}:9000/data${vol}"
    done
done

# Start MinIO nodes
for i in $(seq 1 $NODES); do
    NODE_PORT=$((9000 + i - 1))
    CONSOLE_PORT=$((9001 + i - 1))
    echo "Starting MinIO Node $i on ports $NODE_PORT (API) and $CONSOLE_PORT (Console)..."

    # Create volume mounts for this node
    VOLUME_MOUNTS=""
    for vol in $(seq 1 $VOLUMES_PER_NODE); do
        VOLUME_MOUNTS="$VOLUME_MOUNTS -v minio-node-${i}-data${vol}:/data${vol}"
    done

    docker run -d \
        --name "${CLUSTER_NAME}-node-${i}" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p "$NODE_PORT:9000" \
        -p "$CONSOLE_PORT:9001" \
        -e MINIO_ROOT_USER="$MINIO_ROOT_USER" \
        -e MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
        -e MINIO_DISTRIBUTED_MODE_ENABLED=yes \
        -e MINIO_DISTRIBUTED_NODES="$MINIO_VOLUMES" \
        $VOLUME_MOUNTS \
        --health-cmd "curl -f http://localhost:9000/minio/health/live || exit 1" \
        --health-interval 30s \
        --health-timeout 10s \
        --health-retries 3 \
        minio/minio:latest server $MINIO_VOLUMES \
        --console-address ":9001"
done

# Wait for nodes to be ready
echo "Waiting for MinIO nodes to be ready..."
sleep 30

# Create initial buckets
echo "Creating default buckets..."
docker run --rm \
    --network "$NETWORK_NAME" \
    -e MINIO_ROOT_USER="$MINIO_ROOT_USER" \
    -e MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
    minio/mc:latest sh -c "
        mc alias set minio http://${CLUSTER_NAME}-node-1:9000 \$MINIO_ROOT_USER \$MINIO_ROOT_PASSWORD
        mc mb minio/pynomaly-models --ignore-existing
        mc mb minio/pynomaly-datasets --ignore-existing
        mc mb minio/pynomaly-results --ignore-existing
        mc mb minio/pynomaly-backups --ignore-existing
        mc policy set public minio/pynomaly-results
        echo 'Buckets created successfully'
    "

# Start Nginx load balancer if requested
if [[ "$WITH_NGINX" == "true" ]]; then
    echo "Starting Nginx load balancer..."

    # Create Nginx configuration
    cat > /tmp/nginx-minio.conf << EOF
upstream minio_servers {
EOF

    for i in $(seq 1 $NODES); do
        echo "    server ${CLUSTER_NAME}-node-${i}:9000;" >> /tmp/nginx-minio.conf
    done

    cat >> /tmp/nginx-minio.conf << EOF
}

upstream minio_console {
EOF

    for i in $(seq 1 $NODES); do
        echo "    server ${CLUSTER_NAME}-node-${i}:9001;" >> /tmp/nginx-minio.conf
    done

    cat >> /tmp/nginx-minio.conf << EOF
}

server {
    listen 80;
    server_name _;

    # API requests
    location / {
        proxy_pass http://minio_servers;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # MinIO specific headers
        proxy_set_header Connection "";
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_request_buffering off;
    }
}

server {
    listen 8080;
    server_name _;

    # Console requests
    location / {
        proxy_pass http://minio_console;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        proxy_set_header Connection "";
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_request_buffering off;
    }
}
EOF

    docker run -d \
        --name "${CLUSTER_NAME}-nginx" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 9000:80 \
        -p 9001:8080 \
        -v /tmp/nginx-minio.conf:/etc/nginx/conf.d/default.conf \
        nginx:alpine
fi

# Start MinIO monitoring if requested
if [[ "$MONITORING" == "true" ]]; then
    echo "Starting MinIO monitoring..."
    docker run -d \
        --name "${CLUSTER_NAME}-monitor" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 9090:9090 \
        -e MINIO_ENDPOINT="http://${CLUSTER_NAME}-node-1:9000" \
        -e MINIO_ACCESS_KEY="$MINIO_ROOT_USER" \
        -e MINIO_SECRET_KEY="$MINIO_ROOT_PASSWORD" \
        prom/prometheus:latest \
        --config.file=/etc/prometheus/prometheus.yml \
        --storage.tsdb.path=/prometheus \
        --web.console.libraries=/usr/share/prometheus/console_libraries \
        --web.console.templates=/usr/share/prometheus/consoles \
        --web.enable-lifecycle
fi

# Set up automated backups if requested
if [[ "$BACKUP" == "true" ]]; then
    echo "Setting up automated backups..."

    # Create backup script
    cat > /tmp/minio-backup.sh << 'EOF'
#!/bin/bash
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
SOURCE_ALIAS="source"
BACKUP_ALIAS="backup"
BACKUP_BUCKET="pynomaly-backups"

# Configure MC aliases
mc alias set $SOURCE_ALIAS http://minio-node-1:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD
mc alias set $BACKUP_ALIAS http://minio-node-1:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD

# Backup each bucket
for bucket in pynomaly-models pynomaly-datasets pynomaly-results; do
    echo "Backing up bucket: $bucket"
    mc mirror --overwrite $SOURCE_ALIAS/$bucket $BACKUP_ALIAS/$BACKUP_BUCKET/$BACKUP_DATE/$bucket/
done

# Clean up old backups (keep last 30 days)
mc rm --recursive --force --older-than 30d $BACKUP_ALIAS/$BACKUP_BUCKET/

echo "Backup completed: $BACKUP_DATE"
EOF

    chmod +x /tmp/minio-backup.sh

    docker run -d \
        --name "${CLUSTER_NAME}-backup" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -e MINIO_ROOT_USER="$MINIO_ROOT_USER" \
        -e MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
        -v /tmp/minio-backup.sh:/usr/local/bin/backup.sh \
        minio/mc:latest sh -c "echo '0 3 * * * /usr/local/bin/backup.sh' | crontab - && crond -f"
fi

echo "MinIO Distributed Cluster started successfully!"
echo ""
echo "Connection Details:"

if [[ "$WITH_NGINX" == "true" ]]; then
    echo "API Endpoint (Load Balanced): http://localhost:9000"
    echo "Console (Load Balanced): http://localhost:9001"
else
    echo "API Endpoints:"
    for i in $(seq 1 $NODES); do
        NODE_PORT=$((9000 + i - 1))
        echo "  Node $i API: http://localhost:$NODE_PORT"
    done
    echo ""
    echo "Console URLs:"
    for i in $(seq 1 $NODES); do
        CONSOLE_PORT=$((9001 + i - 1))
        echo "  Node $i Console: http://localhost:$CONSOLE_PORT"
    done
fi

if [[ "$MONITORING" == "true" ]]; then
    echo ""
    echo "Monitoring: http://localhost:9090"
fi

echo ""
echo "Credentials:"
echo "Access Key: $MINIO_ROOT_USER"
echo "Secret Key: $MINIO_ROOT_PASSWORD"
echo ""
echo "Default Buckets:"
echo "  - pynomaly-models (private)"
echo "  - pynomaly-datasets (private)"
echo "  - pynomaly-results (public)"
echo "  - pynomaly-backups (private)"
echo ""
echo "Use --stop to stop all containers"
