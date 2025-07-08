#!/usr/bin/env bash

# PostgreSQL High Availability Cluster
# Master-replica setup with automatic failover

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default configuration
NETWORK_NAME="pynomaly-postgres"
MASTER_NAME="pynomaly-postgres-master"
REPLICA_NAME="pynomaly-postgres-replica"
PGPOOL_NAME="pynomaly-pgpool"
REPLICAS=2

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --replicas NUM       Number of replica nodes (default: 2)"
    echo "  --with-pgpool        Include PgPool-II for load balancing"
    echo "  --monitoring         Include PostgreSQL monitoring"
    echo "  --backup             Set up automated backups"
    echo "  --clean              Remove existing containers first"
    echo "  --stop               Stop all PostgreSQL containers"
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
        --with-pgpool)
            WITH_PGPOOL=true
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

# Function to stop all PostgreSQL containers
stop_containers() {
    echo "Stopping all PostgreSQL containers..."
    docker rm -f "$MASTER_NAME" 2>/dev/null || true
    for i in $(seq 1 $REPLICAS); do
        docker rm -f "${REPLICA_NAME}-${i}" 2>/dev/null || true
    done
    docker rm -f "$PGPOOL_NAME" 2>/dev/null || true
    docker rm -f "pynomaly-postgres-monitor" 2>/dev/null || true
    docker rm -f "pynomaly-postgres-backup" 2>/dev/null || true
    echo "All PostgreSQL containers stopped."
    exit 0
}

if [[ "$STOP_CONTAINERS" == "true" ]]; then
    stop_containers
fi

# Create network
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Clean up if requested
if [[ "$CLEAN_CONTAINERS" == "true" ]]; then
    echo "Cleaning up existing PostgreSQL containers..."
    stop_containers
fi

echo "Starting PostgreSQL High Availability Cluster..."
echo "Master: 1 node"
echo "Replicas: $REPLICAS nodes"

# Start PostgreSQL Master
echo "Starting PostgreSQL Master..."
docker run -d \
    --name "$MASTER_NAME" \
    --network "$NETWORK_NAME" \
    --restart unless-stopped \
    -e POSTGRES_DB=pynomaly \
    -e POSTGRES_USER=pynomaly \
    -e POSTGRES_PASSWORD=prod_password \
    -e POSTGRES_REPLICATION_MODE=master \
    -e POSTGRES_REPLICATION_USER=replicator \
    -e POSTGRES_REPLICATION_PASSWORD=replicator_password \
    -e POSTGRESQL_SHARED_PRELOAD_LIBRARIES=pg_stat_statements \
    -p 5432:5432 \
    -v postgres-master-data:/var/lib/postgresql/data \
    -v postgres-master-config:/opt/bitnami/postgresql/conf \
    bitnami/postgresql:15

# Wait for master to be ready
echo "Waiting for master to be ready..."
sleep 30

# Start PostgreSQL Replicas
for i in $(seq 1 $REPLICAS); do
    REPLICA_PORT=$((5432 + i))
    echo "Starting PostgreSQL Replica $i on port $REPLICA_PORT..."

    docker run -d \
        --name "${REPLICA_NAME}-${i}" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -e POSTGRES_REPLICATION_MODE=slave \
        -e POSTGRES_REPLICATION_USER=replicator \
        -e POSTGRES_REPLICATION_PASSWORD=replicator_password \
        -e POSTGRES_MASTER_HOST="$MASTER_NAME" \
        -e POSTGRES_MASTER_PORT_NUMBER=5432 \
        -e POSTGRESQL_SHARED_PRELOAD_LIBRARIES=pg_stat_statements \
        -p "$REPLICA_PORT:5432" \
        -v "postgres-replica-${i}-data:/var/lib/postgresql/data" \
        bitnami/postgresql:15
done

# Start PgPool-II for load balancing if requested
if [[ "$WITH_PGPOOL" == "true" ]]; then
    echo "Starting PgPool-II load balancer..."

    # Create PgPool configuration
    cat > /tmp/pgpool.conf << EOF
listen_addresses = '*'
port = 5432
socket_dir = '/var/run/pgpool'
backend_hostname0 = '$MASTER_NAME'
backend_port0 = 5432
backend_weight0 = 1
backend_data_directory0 = '/var/lib/postgresql/data'
backend_flag0 = 'ALLOW_TO_FAILOVER'
EOF

    for i in $(seq 1 $REPLICAS); do
        cat >> /tmp/pgpool.conf << EOF
backend_hostname${i} = '${REPLICA_NAME}-${i}'
backend_port${i} = 5432
backend_weight${i} = 1
backend_data_directory${i} = '/var/lib/postgresql/data'
backend_flag${i} = 'ALLOW_TO_FAILOVER'
EOF
    done

    cat >> /tmp/pgpool.conf << EOF
enable_pool_hba = off
pool_passwd = ''
authentication_timeout = 60
ssl = off
num_init_children = 32
max_pool = 4
child_life_time = 300
child_max_connections = 0
connection_life_time = 0
client_idle_limit = 0
EOF

    docker run -d \
        --name "$PGPOOL_NAME" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 5430:5432 \
        -v /tmp/pgpool.conf:/opt/bitnami/pgpool/conf/pgpool.conf \
        -e PGPOOL_BACKEND_NODES="0:$MASTER_NAME:5432" \
        -e PGPOOL_SR_CHECK_USER=replicator \
        -e PGPOOL_SR_CHECK_PASSWORD=replicator_password \
        -e PGPOOL_ENABLE_LDAP=no \
        bitnami/pgpool:4
fi

# Start PostgreSQL monitoring if requested
if [[ "$MONITORING" == "true" ]]; then
    echo "Starting PostgreSQL monitoring..."
    docker run -d \
        --name "pynomaly-postgres-monitor" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 9187:9187 \
        -e DATA_SOURCE_NAME="postgresql://pynomaly:prod_password@$MASTER_NAME:5432/pynomaly?sslmode=disable" \
        prometheuscommunity/postgres-exporter:latest
fi

# Set up automated backups if requested
if [[ "$BACKUP" == "true" ]]; then
    echo "Setting up automated backups..."
    docker run -d \
        --name "pynomaly-postgres-backup" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -e POSTGRES_HOST="$MASTER_NAME" \
        -e POSTGRES_PORT=5432 \
        -e POSTGRES_DB=pynomaly \
        -e POSTGRES_USER=pynomaly \
        -e POSTGRES_PASSWORD=prod_password \
        -e BACKUP_SCHEDULE="0 2 * * *" \
        -e BACKUP_RETENTION_DAYS=30 \
        -v postgres-backups:/backups \
        kartoza/pg-backup:15-3.3
fi

echo "PostgreSQL High Availability Cluster started successfully!"
echo ""
echo "Connection Details:"
echo "Master: localhost:5432"
for i in $(seq 1 $REPLICAS); do
    REPLICA_PORT=$((5432 + i))
    echo "Replica $i: localhost:$REPLICA_PORT"
done

if [[ "$WITH_PGPOOL" == "true" ]]; then
    echo "PgPool Load Balancer: localhost:5430"
fi

if [[ "$MONITORING" == "true" ]]; then
    echo "Metrics: localhost:9187"
fi

echo ""
echo "Database: pynomaly"
echo "Username: pynomaly"
echo "Password: prod_password"
echo ""
echo "Use --stop to stop all containers"
