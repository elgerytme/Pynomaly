#!/bin/bash

# MongoDB High Availability Cluster
# Replica set with automatic failover

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default configuration
NETWORK_NAME="pynomaly-mongodb"
REPLICA_SET_NAME="pynomaly-rs"
PRIMARY_NAME="pynomaly-mongodb-primary"
SECONDARY_PREFIX="pynomaly-mongodb-secondary"
SECONDARIES=2

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --secondaries NUM    Number of secondary nodes (default: 2)"
    echo "  --with-arbiter       Include arbiter node for odd number voting"
    echo "  --sharding           Enable sharding (requires additional config servers)"
    echo "  --monitoring         Include MongoDB monitoring"
    echo "  --backup             Set up automated backups"
    echo "  --clean              Remove existing containers first"
    echo "  --stop               Stop all MongoDB containers"
    echo "  -h, --help           Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --secondaries)
            SECONDARIES="$2"
            shift 2
            ;;
        --with-arbiter)
            WITH_ARBITER=true
            shift
            ;;
        --sharding)
            SHARDING=true
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

# Function to stop all MongoDB containers
stop_containers() {
    echo "Stopping all MongoDB containers..."
    docker rm -f "$PRIMARY_NAME" 2>/dev/null || true
    for i in $(seq 1 $SECONDARIES); do
        docker rm -f "${SECONDARY_PREFIX}-${i}" 2>/dev/null || true
    done
    docker rm -f "pynomaly-mongodb-arbiter" 2>/dev/null || true

    if [[ "$SHARDING" == "true" ]]; then
        docker rm -f "pynomaly-mongos" 2>/dev/null || true
        for i in 1 2 3; do
            docker rm -f "pynomaly-configsvr-${i}" 2>/dev/null || true
        done
        for i in 1 2; do
            docker rm -f "pynomaly-shard${i}-primary" 2>/dev/null || true
            docker rm -f "pynomaly-shard${i}-secondary" 2>/dev/null || true
        done
    fi

    docker rm -f "pynomaly-mongodb-monitor" 2>/dev/null || true
    docker rm -f "pynomaly-mongodb-backup" 2>/dev/null || true
    echo "All MongoDB containers stopped."
    exit 0
}

if [[ "$STOP_CONTAINERS" == "true" ]]; then
    stop_containers
fi

# Create network
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Clean up if requested
if [[ "$CLEAN_CONTAINERS" == "true" ]]; then
    echo "Cleaning up existing MongoDB containers..."
    stop_containers
fi

# Generate MongoDB credentials
MONGODB_ROOT_PASSWORD="prod_password"
MONGODB_REPLICA_SET_KEY="replicasetkey123"

if [[ "$SHARDING" == "true" ]]; then
    echo "Starting MongoDB Sharded Cluster..."

    # Start Config Servers
    echo "Starting Config Server Replica Set..."
    for i in 1 2 3; do
        CONFIG_PORT=$((27019 + i - 1))
        echo "Starting Config Server $i on port $CONFIG_PORT..."

        docker run -d \
            --name "pynomaly-configsvr-${i}" \
            --network "$NETWORK_NAME" \
            --restart unless-stopped \
            -p "$CONFIG_PORT:27017" \
            -e MONGODB_REPLICA_SET_MODE=primary \
            -e MONGODB_REPLICA_SET_NAME=configReplSet \
            -e MONGODB_ROOT_PASSWORD="$MONGODB_ROOT_PASSWORD" \
            -e MONGODB_REPLICA_SET_KEY="$MONGODB_REPLICA_SET_KEY" \
            -v "mongodb-configsvr-${i}:/bitnami/mongodb" \
            bitnami/mongodb:7.0 --configsvr
    done

    sleep 15

    # Start Shard Replica Sets
    for shard in 1 2; do
        echo "Starting Shard $shard Replica Set..."

        # Primary
        SHARD_PORT=$((27017 + (shard - 1) * 10))
        docker run -d \
            --name "pynomaly-shard${shard}-primary" \
            --network "$NETWORK_NAME" \
            --restart unless-stopped \
            -p "$SHARD_PORT:27017" \
            -e MONGODB_REPLICA_SET_MODE=primary \
            -e MONGODB_REPLICA_SET_NAME="shard${shard}ReplSet" \
            -e MONGODB_ROOT_PASSWORD="$MONGODB_ROOT_PASSWORD" \
            -e MONGODB_REPLICA_SET_KEY="$MONGODB_REPLICA_SET_KEY" \
            -v "mongodb-shard${shard}-primary:/bitnami/mongodb" \
            bitnami/mongodb:7.0 --shardsvr

        # Secondary
        SHARD_SEC_PORT=$((27018 + (shard - 1) * 10))
        docker run -d \
            --name "pynomaly-shard${shard}-secondary" \
            --network "$NETWORK_NAME" \
            --restart unless-stopped \
            -p "$SHARD_SEC_PORT:27017" \
            -e MONGODB_REPLICA_SET_MODE=secondary \
            -e MONGODB_REPLICA_SET_NAME="shard${shard}ReplSet" \
            -e MONGODB_PRIMARY_HOST="pynomaly-shard${shard}-primary" \
            -e MONGODB_PRIMARY_ROOT_PASSWORD="$MONGODB_ROOT_PASSWORD" \
            -e MONGODB_REPLICA_SET_KEY="$MONGODB_REPLICA_SET_KEY" \
            -v "mongodb-shard${shard}-secondary:/bitnami/mongodb" \
            bitnami/mongodb:7.0 --shardsvr
    done

    sleep 30

    # Start Mongos Router
    echo "Starting Mongos Router..."
    docker run -d \
        --name "pynomaly-mongos" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 27017:27017 \
        -e MONGODB_CFG_PRIMARY_HOST=pynomaly-configsvr-1 \
        -e MONGODB_CFG_REPLICA_SET_NAME=configReplSet \
        -e MONGODB_ROOT_PASSWORD="$MONGODB_ROOT_PASSWORD" \
        -e MONGODB_REPLICA_SET_KEY="$MONGODB_REPLICA_SET_KEY" \
        bitnami/mongodb:7.0 mongos --configdb configReplSet/pynomaly-configsvr-1:27017,pynomaly-configsvr-2:27017,pynomaly-configsvr-3:27017

    sleep 15

    # Initialize sharding
    echo "Initializing shards..."
    docker exec -it pynomaly-mongos mongosh --eval "
        sh.addShard('shard1ReplSet/pynomaly-shard1-primary:27017,pynomaly-shard1-secondary:27017');
        sh.addShard('shard2ReplSet/pynomaly-shard2-primary:27017,pynomaly-shard2-secondary:27017');
        sh.enableSharding('pynomaly');
        sh.shardCollection('pynomaly.anomalies', {_id: 'hashed'});
        sh.shardCollection('pynomaly.detectors', {_id: 'hashed'});
        sh.status();
    "

else
    echo "Starting MongoDB Replica Set..."
    echo "Primary: 1 node"
    echo "Secondaries: $SECONDARIES nodes"

    # Start MongoDB Primary
    echo "Starting MongoDB Primary..."
    docker run -d \
        --name "$PRIMARY_NAME" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 27017:27017 \
        -e MONGODB_REPLICA_SET_MODE=primary \
        -e MONGODB_REPLICA_SET_NAME="$REPLICA_SET_NAME" \
        -e MONGODB_ROOT_PASSWORD="$MONGODB_ROOT_PASSWORD" \
        -e MONGODB_REPLICA_SET_KEY="$MONGODB_REPLICA_SET_KEY" \
        -v mongodb-primary-data:/bitnami/mongodb \
        bitnami/mongodb:7.0

    # Wait for primary to be ready
    echo "Waiting for primary to be ready..."
    sleep 30

    # Start MongoDB Secondaries
    for i in $(seq 1 $SECONDARIES); do
        SECONDARY_PORT=$((27017 + i))
        echo "Starting MongoDB Secondary $i on port $SECONDARY_PORT..."

        docker run -d \
            --name "${SECONDARY_PREFIX}-${i}" \
            --network "$NETWORK_NAME" \
            --restart unless-stopped \
            -p "$SECONDARY_PORT:27017" \
            -e MONGODB_REPLICA_SET_MODE=secondary \
            -e MONGODB_REPLICA_SET_NAME="$REPLICA_SET_NAME" \
            -e MONGODB_PRIMARY_HOST="$PRIMARY_NAME" \
            -e MONGODB_PRIMARY_ROOT_PASSWORD="$MONGODB_ROOT_PASSWORD" \
            -e MONGODB_REPLICA_SET_KEY="$MONGODB_REPLICA_SET_KEY" \
            -v "mongodb-secondary-${i}-data:/bitnami/mongodb" \
            bitnami/mongodb:7.0
    done

    # Start Arbiter if requested
    if [[ "$WITH_ARBITER" == "true" ]]; then
        echo "Starting MongoDB Arbiter..."
        docker run -d \
            --name "pynomaly-mongodb-arbiter" \
            --network "$NETWORK_NAME" \
            --restart unless-stopped \
            -p 27030:27017 \
            -e MONGODB_REPLICA_SET_MODE=arbiter \
            -e MONGODB_REPLICA_SET_NAME="$REPLICA_SET_NAME" \
            -e MONGODB_PRIMARY_HOST="$PRIMARY_NAME" \
            -e MONGODB_PRIMARY_ROOT_PASSWORD="$MONGODB_ROOT_PASSWORD" \
            -e MONGODB_REPLICA_SET_KEY="$MONGODB_REPLICA_SET_KEY" \
            bitnami/mongodb:7.0
    fi

    # Wait for replica set to initialize
    echo "Waiting for replica set to initialize..."
    sleep 30

    # Initialize database and collections
    echo "Initializing database..."
    docker exec -it "$PRIMARY_NAME" mongosh pynomaly --eval "
        db.createCollection('anomalies', {
            validator: {
                \$jsonSchema: {
                    bsonType: 'object',
                    required: ['timestamp', 'score', 'data'],
                    properties: {
                        timestamp: { bsonType: 'date' },
                        score: { bsonType: 'double', minimum: 0, maximum: 1 },
                        data: { bsonType: 'object' },
                        detector_id: { bsonType: 'string' },
                        metadata: { bsonType: 'object' }
                    }
                }
            }
        });

        db.createCollection('detectors', {
            validator: {
                \$jsonSchema: {
                    bsonType: 'object',
                    required: ['name', 'algorithm', 'created_at'],
                    properties: {
                        name: { bsonType: 'string' },
                        algorithm: { bsonType: 'string' },
                        created_at: { bsonType: 'date' },
                        parameters: { bsonType: 'object' },
                        metrics: { bsonType: 'object' }
                    }
                }
            }
        });

        db.anomalies.createIndex({ timestamp: -1 });
        db.anomalies.createIndex({ score: -1 });
        db.anomalies.createIndex({ detector_id: 1, timestamp: -1 });
        db.detectors.createIndex({ algorithm: 1 });
        db.detectors.createIndex({ created_at: -1 });

        print('Database initialized successfully');
    "
fi

# Start MongoDB monitoring if requested
if [[ "$MONITORING" == "true" ]]; then
    echo "Starting MongoDB monitoring..."
    docker run -d \
        --name "pynomaly-mongodb-monitor" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 9216:9216 \
        -e MONGODB_URI="mongodb://root:${MONGODB_ROOT_PASSWORD}@${PRIMARY_NAME}:27017/admin" \
        percona/mongodb_exporter:latest \
        --mongodb.uri="mongodb://root:${MONGODB_ROOT_PASSWORD}@${PRIMARY_NAME}:27017/admin" \
        --collect-all
fi

# Set up automated backups if requested
if [[ "$BACKUP" == "true" ]]; then
    echo "Setting up automated backups..."

    # Create backup script
    cat > /tmp/mongodb-backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
MONGODB_HOST=${MONGODB_HOST:-mongodb-primary}
MONGODB_PORT=${MONGODB_PORT:-27017}
MONGODB_PASSWORD=${MONGODB_PASSWORD}

mkdir -p "$BACKUP_DIR"

# Create backup using mongodump
mongodump --host "$MONGODB_HOST:$MONGODB_PORT" \
    --username root \
    --password "$MONGODB_PASSWORD" \
    --authenticationDatabase admin \
    --db pynomaly \
    --out "$BACKUP_DIR/mongodb-backup-$DATE"

# Compress backup
tar -czf "$BACKUP_DIR/mongodb-backup-$DATE.tar.gz" -C "$BACKUP_DIR" "mongodb-backup-$DATE"
rm -rf "$BACKUP_DIR/mongodb-backup-$DATE"

# Clean up old backups (keep last 30 days)
find "$BACKUP_DIR" -name "mongodb-backup-*.tar.gz" -mtime +30 -delete

echo "Backup completed: mongodb-backup-$DATE.tar.gz"
EOF

    chmod +x /tmp/mongodb-backup.sh

    docker run -d \
        --name "pynomaly-mongodb-backup" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -e MONGODB_HOST="$PRIMARY_NAME" \
        -e MONGODB_PORT=27017 \
        -e MONGODB_PASSWORD="$MONGODB_ROOT_PASSWORD" \
        -v mongodb-backups:/backups \
        -v /tmp/mongodb-backup.sh:/usr/local/bin/backup.sh \
        mongo:7.0 sh -c "echo '0 3 * * * /usr/local/bin/backup.sh' | crontab - && cron -f"
fi

echo "MongoDB deployment completed successfully!"
echo ""

if [[ "$SHARDING" == "true" ]]; then
    echo "MongoDB Sharded Cluster:"
    echo "Mongos Router: localhost:27017"
    echo "Config Servers:"
    for i in 1 2 3; do
        CONFIG_PORT=$((27019 + i - 1))
        echo "  Config Server $i: localhost:$CONFIG_PORT"
    done
    echo "Shards:"
    for shard in 1 2; do
        SHARD_PORT=$((27017 + (shard - 1) * 10))
        SHARD_SEC_PORT=$((27018 + (shard - 1) * 10))
        echo "  Shard $shard Primary: localhost:$SHARD_PORT"
        echo "  Shard $shard Secondary: localhost:$SHARD_SEC_PORT"
    done
else
    echo "MongoDB Replica Set:"
    echo "Primary: localhost:27017"
    for i in $(seq 1 $SECONDARIES); do
        SECONDARY_PORT=$((27017 + i))
        echo "Secondary $i: localhost:$SECONDARY_PORT"
    done
    if [[ "$WITH_ARBITER" == "true" ]]; then
        echo "Arbiter: localhost:27030"
    fi
fi

if [[ "$MONITORING" == "true" ]]; then
    echo ""
    echo "Metrics: localhost:9216"
fi

echo ""
echo "Database: pynomaly"
echo "Username: root"
echo "Password: $MONGODB_ROOT_PASSWORD"
echo "Replica Set: $REPLICA_SET_NAME"
echo ""
echo "Use --stop to stop all containers"
