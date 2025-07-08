#!/usr/bin/env bash

# Redis High Availability Cluster
# Master-replica setup with Sentinel for automatic failover

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default configuration
NETWORK_NAME="pynomaly-redis"
MASTER_NAME="pynomaly-redis-master"
REPLICA_PREFIX="pynomaly-redis-replica"
SENTINEL_PREFIX="pynomaly-redis-sentinel"
REPLICAS=2
SENTINELS=3

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --replicas NUM       Number of replica nodes (default: 2)"
    echo "  --sentinels NUM      Number of sentinel nodes (default: 3)"
    echo "  --cluster-mode       Use Redis Cluster mode instead of replication"
    echo "  --monitoring         Include Redis monitoring"
    echo "  --backup             Set up automated backups"
    echo "  --clean              Remove existing containers first"
    echo "  --stop               Stop all Redis containers"
    echo "  -h, --help           Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --replicas)
            REPLICAS="$2"
            shift 2
            ;;
        --sentinels)
            SENTINELS="$2"
            shift 2
            ;;
        --cluster-mode)
            CLUSTER_MODE=true
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

# Function to stop all Redis containers
stop_containers() {
    echo "Stopping all Redis containers..."
    docker rm -f "$MASTER_NAME" 2>/dev/null || true
    for i in $(seq 1 $REPLICAS); do
        docker rm -f "${REPLICA_PREFIX}-${i}" 2>/dev/null || true
    done
    for i in $(seq 1 $SENTINELS); do
        docker rm -f "${SENTINEL_PREFIX}-${i}" 2>/dev/null || true
    done
    docker rm -f "pynomaly-redis-monitor" 2>/dev/null || true
    docker rm -f "pynomaly-redis-backup" 2>/dev/null || true
    echo "All Redis containers stopped."
    exit 0
}

if [[ "$STOP_CONTAINERS" == "true" ]]; then
    stop_containers
fi

# Create network
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Clean up if requested
if [[ "$CLEAN_CONTAINERS" == "true" ]]; then
    echo "Cleaning up existing Redis containers..."
    stop_containers
fi

if [[ "$CLUSTER_MODE" == "true" ]]; then
    echo "Starting Redis Cluster..."
    CLUSTER_NODES=$((1 + REPLICAS))

    # Start Redis nodes for cluster
    for i in $(seq 0 $((CLUSTER_NODES - 1))); do
        NODE_PORT=$((7000 + i))
        NODE_NAME="pynomaly-redis-node-${i}"

        echo "Starting Redis Cluster Node $i on port $NODE_PORT..."
        docker run -d \
            --name "$NODE_NAME" \
            --network "$NETWORK_NAME" \
            --restart unless-stopped \
            -p "$NODE_PORT:6379" \
            -p "$((NODE_PORT + 10000)):16379" \
            -v "redis-cluster-node-${i}:/data" \
            redis:7-alpine redis-server \
            --port 6379 \
            --cluster-enabled yes \
            --cluster-config-file nodes.conf \
            --cluster-node-timeout 5000 \
            --appendonly yes \
            --appendfsync everysec \
            --save 900 1 \
            --save 300 10 \
            --save 60 10000
    done

    # Wait for nodes to start
    echo "Waiting for cluster nodes to start..."
    sleep 10

    # Create cluster
    echo "Creating Redis cluster..."
    CLUSTER_IPS=""
    for i in $(seq 0 $((CLUSTER_NODES - 1))); do
        NODE_NAME="pynomaly-redis-node-${i}"
        NODE_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$NODE_NAME")
        CLUSTER_IPS="$CLUSTER_IPS $NODE_IP:6379"
    done

    # Initialize cluster
    docker run --rm -it \
        --network "$NETWORK_NAME" \
        redis:7-alpine redis-cli --cluster create $CLUSTER_IPS --cluster-replicas 1 --cluster-yes

else
    echo "Starting Redis High Availability with Sentinel..."
    echo "Master: 1 node"
    echo "Replicas: $REPLICAS nodes"
    echo "Sentinels: $SENTINELS nodes"

    # Start Redis Master
    echo "Starting Redis Master..."
    docker run -d \
        --name "$MASTER_NAME" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 6379:6379 \
        -v redis-master-data:/data \
        -v redis-master-config:/usr/local/etc/redis \
        redis:7-alpine redis-server \
        --appendonly yes \
        --appendfsync everysec \
        --save 900 1 \
        --save 300 10 \
        --save 60 10000 \
        --maxmemory 1gb \
        --maxmemory-policy allkeys-lru \
        --tcp-keepalive 60 \
        --timeout 300

    # Wait for master to be ready
    echo "Waiting for master to be ready..."
    sleep 10

    # Start Redis Replicas
    for i in $(seq 1 $REPLICAS); do
        REPLICA_PORT=$((6379 + i))
        echo "Starting Redis Replica $i on port $REPLICA_PORT..."

        docker run -d \
            --name "${REPLICA_PREFIX}-${i}" \
            --network "$NETWORK_NAME" \
            --restart unless-stopped \
            -p "$REPLICA_PORT:6379" \
            -v "redis-replica-${i}-data:/data" \
            redis:7-alpine redis-server \
            --replicaof "$MASTER_NAME" 6379 \
            --appendonly yes \
            --appendfsync everysec \
            --replica-read-only yes \
            --tcp-keepalive 60 \
            --timeout 300
    done

    # Start Redis Sentinels
    for i in $(seq 1 $SENTINELS); do
        SENTINEL_PORT=$((26379 + i - 1))
        echo "Starting Redis Sentinel $i on port $SENTINEL_PORT..."

        # Create sentinel configuration
        cat > "/tmp/sentinel-${i}.conf" << EOF
port 26379
sentinel monitor mymaster $MASTER_NAME 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 10000
sentinel auth-pass mymaster ""
logfile ""
EOF

        docker run -d \
            --name "${SENTINEL_PREFIX}-${i}" \
            --network "$NETWORK_NAME" \
            --restart unless-stopped \
            -p "$SENTINEL_PORT:26379" \
            -v "/tmp/sentinel-${i}.conf:/usr/local/etc/redis/sentinel.conf" \
            redis:7-alpine redis-sentinel /usr/local/etc/redis/sentinel.conf
    done
fi

# Start Redis monitoring if requested
if [[ "$MONITORING" == "true" ]]; then
    echo "Starting Redis monitoring..."
    docker run -d \
        --name "pynomaly-redis-monitor" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 9121:9121 \
        -e REDIS_ADDR="redis://$MASTER_NAME:6379" \
        oliver006/redis_exporter:latest
fi

# Set up automated backups if requested
if [[ "$BACKUP" == "true" ]]; then
    echo "Setting up automated backups..."

    # Create backup script
    cat > /tmp/redis-backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
REDIS_HOST=${REDIS_HOST:-redis-master}
REDIS_PORT=${REDIS_PORT:-6379}

mkdir -p "$BACKUP_DIR"

# Create backup using BGSAVE
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE

# Wait for backup to complete
while [ $(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE) -eq $LAST_SAVE ]; do
    sleep 1
done

# Copy RDB file
cp /data/dump.rdb "$BACKUP_DIR/redis-backup-$DATE.rdb"

# Compress backup
gzip "$BACKUP_DIR/redis-backup-$DATE.rdb"

# Clean up old backups (keep last 30 days)
find "$BACKUP_DIR" -name "redis-backup-*.rdb.gz" -mtime +30 -delete

echo "Backup completed: redis-backup-$DATE.rdb.gz"
EOF

    chmod +x /tmp/redis-backup.sh

    docker run -d \
        --name "pynomaly-redis-backup" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -e REDIS_HOST="$MASTER_NAME" \
        -e REDIS_PORT=6379 \
        -v redis-backups:/backups \
        -v /tmp/redis-backup.sh:/usr/local/bin/backup.sh \
        -v redis-master-data:/data:ro \
        alpine/cron:latest sh -c "echo '0 2 * * * /usr/local/bin/backup.sh' | crontab - && crond -f"
fi

echo "Redis deployment completed successfully!"
echo ""

if [[ "$CLUSTER_MODE" == "true" ]]; then
    echo "Redis Cluster Mode:"
    for i in $(seq 0 $((CLUSTER_NODES - 1))); do
        NODE_PORT=$((7000 + i))
        echo "Node $i: localhost:$NODE_PORT"
    done
else
    echo "Redis High Availability Mode:"
    echo "Master: localhost:6379"
    for i in $(seq 1 $REPLICAS); do
        REPLICA_PORT=$((6379 + i))
        echo "Replica $i: localhost:$REPLICA_PORT"
    done
    echo ""
    echo "Sentinels:"
    for i in $(seq 1 $SENTINELS); do
        SENTINEL_PORT=$((26379 + i - 1))
        echo "Sentinel $i: localhost:$SENTINEL_PORT"
    done
fi

if [[ "$MONITORING" == "true" ]]; then
    echo ""
    echo "Metrics: localhost:9121"
fi

echo ""
echo "Use --stop to stop all containers"
