# Docker Deployment Scripts

This directory contains comprehensive Docker deployment scripts for anomaly_detection across different environments and infrastructure configurations.

## Directory Structure

```
scripts/docker/
├── dev/                    # Development environment
│   ├── run-dev.sh         # Basic development setup
│   ├── run-dev.ps1        # Windows PowerShell version
│   ├── run-dev-with-storage.sh    # Dev with storage services
│   └── run-dev-with-storage.ps1   # Windows version
├── test/                   # Testing environments
│   ├── run-test.sh        # Test runner with isolation
│   ├── run-test.ps1       # Windows version
│   ├── run-benchmarks.sh  # Performance benchmarking
│   └── run-benchmarks.ps1 # Windows version
├── prod/                   # Production deployments
│   ├── run-prod.sh        # Single-node production
│   ├── run-prod.ps1       # Windows version
│   ├── run-prod-distributed.sh    # Multi-node distributed
│   └── run-prod-distributed.ps1   # Windows version
├── storage/                # Storage infrastructure
│   ├── run-postgres-cluster.sh    # PostgreSQL HA cluster
│   ├── run-redis-cluster.sh       # Redis HA cluster
│   ├── run-minio-cluster.sh       # MinIO distributed storage
│   └── run-mongodb-cluster.sh     # MongoDB replica set
└── README.md              # This file
```

## Quick Start

### Development Environment

**Basic Development Setup:**
```bash
# Linux/macOS
./dev/run-dev.sh

# Windows
.\dev\run-dev.ps1
```

**Development with Storage Services:**
```bash
# With all storage services (PostgreSQL, Redis, MinIO)
./dev/run-dev-with-storage.sh --storage all

# With specific storage only
./dev/run-dev-with-storage.sh --storage postgres
```

### Testing Environment

**Run Tests:**
```bash
# Unit tests only
./test/run-test.sh --type unit

# All tests with coverage
./test/run-test.sh --type all --coverage --parallel

# With storage services for integration tests
./test/run-test.sh --type integration --storage all
```

**Performance Benchmarks:**
```bash
# Basic performance benchmarks
./test/run-benchmarks.sh --type performance

# Load testing with monitoring
./test/run-benchmarks.sh --type load --monitoring
```

### Production Environment

**Single-Node Production:**
```bash
# Basic production deployment
./prod/run-prod.sh --port 80 --replicas 3

# With SSL, monitoring, and logging
./prod/run-prod.sh --ssl --monitoring --logging
```

**Distributed Production:**
```bash
# Multi-node distributed deployment
./prod/run-prod-distributed.sh --workers 3 --api-replicas 2

# With service discovery and secrets management
./prod/run-prod-distributed.sh --consul --vault --storage postgres
```

### Storage Infrastructure

**PostgreSQL High Availability:**
```bash
# Basic master-replica setup
./storage/run-postgres-cluster.sh --replicas 2

# With PgPool load balancer and monitoring
./storage/run-postgres-cluster.sh --replicas 2 --with-pgpool --monitoring
```

**Redis Cluster:**
```bash
# Sentinel-based HA
./storage/run-redis-cluster.sh --replicas 2 --sentinels 3

# Redis Cluster mode
./storage/run-redis-cluster.sh --cluster-mode --replicas 2
```

**MinIO Distributed Storage:**
```bash
# 4-node distributed setup
./storage/run-minio-cluster.sh --nodes 4 --volumes-per-node 2

# With load balancer and monitoring
./storage/run-minio-cluster.sh --nodes 4 --with-nginx --monitoring
```

**MongoDB Replica Set:**
```bash
# Basic replica set
./storage/run-mongodb-cluster.sh --secondaries 2

# Sharded cluster
./storage/run-mongodb-cluster.sh --sharding --secondaries 2
```

## Script Options

### Common Options (All Scripts)

- `--build`: Build Docker images before running
- `--clean`: Remove existing containers first
- `--stop`: Stop all related containers
- `--help`: Show usage information

### Development Scripts

**run-dev.sh / run-dev.ps1:**
- `-p, --port PORT`: Host port to bind (default: 8000)
- `-n, --name NAME`: Container name
- `-i, --image IMAGE`: Docker image name
- `--env-file FILE`: Environment file path

**run-dev-with-storage.sh / run-dev-with-storage.ps1:**
- `--storage TYPE`: Storage backend (postgres|redis|minio|all)

### Test Scripts

**run-test.sh / run-test.ps1:**
- `-t, --type TYPE`: Test type (unit|integration|e2e|all)
- `-c, --coverage`: Run with coverage reporting
- `--parallel`: Run tests in parallel
- `--storage TYPE`: Include storage services

**run-benchmarks.sh / run-benchmarks.ps1:**
- `-t, --type TYPE`: Benchmark type (performance|load|stress|memory)
- `--datasets PATH`: Custom datasets directory
- `--algorithms LIST`: Comma-separated algorithm list
- `--monitoring`: Include monitoring stack

### Production Scripts

**run-prod.sh / run-prod.ps1:**
- `-p, --port PORT`: Host port to bind (default: 80)
- `-r, --replicas NUM`: Number of replicas
- `--ssl`: Enable SSL/TLS
- `--monitoring`: Include monitoring stack
- `--logging`: Include centralized logging

**run-prod-distributed.sh / run-prod-distributed.ps1:**
- `--workers NUM`: Number of worker nodes
- `--api-replicas NUM`: Number of API replicas
- `--storage TYPE`: Storage backend
- `--consul`: Enable service discovery
- `--vault`: Enable secrets management

### Storage Scripts

**run-postgres-cluster.sh:**
- `--replicas NUM`: Number of replica nodes
- `--with-pgpool`: Include PgPool-II load balancer
- `--monitoring`: Include PostgreSQL monitoring
- `--backup`: Set up automated backups

**run-redis-cluster.sh:**
- `--replicas NUM`: Number of replica nodes
- `--sentinels NUM`: Number of sentinel nodes
- `--cluster-mode`: Use Redis Cluster mode
- `--monitoring`: Include Redis monitoring

**run-minio-cluster.sh:**
- `--nodes NUM`: Number of MinIO nodes (min: 4)
- `--volumes-per-node N`: Volumes per node
- `--with-nginx`: Include Nginx load balancer
- `--monitoring`: Include MinIO monitoring

**run-mongodb-cluster.sh:**
- `--secondaries NUM`: Number of secondary nodes
- `--with-arbiter`: Include arbiter node
- `--sharding`: Enable sharding
- `--monitoring`: Include MongoDB monitoring

## Environment Variables

### Development Environment

Create `.env.dev` file:
```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
DATABASE_URL=postgresql://anomaly_detection:dev_password@localhost:5432/anomaly_detection_dev
REDIS_URL=redis://localhost:6379/0
```

### Test Environment

Create `.env.test` file:
```env
ENVIRONMENT=test
DEBUG=false
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://test_user:test_password@localhost:5433/anomaly_detection_test
REDIS_URL=redis://localhost:6380/0
```

### Production Environment

Create `.env.prod` file:
```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://anomaly_detection:prod_password@postgres-master:5432/anomaly_detection
REDIS_URL=redis://redis-master:6379/0
SECRET_KEY=your-secret-key
JWT_SECRET=your-jwt-secret
```

## Docker Networks

Each environment uses its own Docker network:

- **Development**: `anomaly_detection-dev`
- **Test**: `anomaly_detection-test` / `anomaly_detection-benchmark`
- **Production**: `anomaly_detection-prod` / `anomaly_detection-distributed`
- **Storage**: `anomaly_detection-postgres` / `anomaly_detection-redis` / `anomaly_detection-minio` / `anomaly_detection-mongodb`

## Volumes and Data Persistence

### Development
- `anomaly_detection-postgres-dev`: PostgreSQL development data
- `anomaly_detection-redis-dev`: Redis development data
- `anomaly_detection-minio-dev`: MinIO development data

### Production
- `postgres-master-data`, `postgres-replica-N-data`: PostgreSQL data
- `redis-master-data`, `redis-replica-N-data`: Redis data
- `minio-node-N-dataN`: MinIO distributed data
- `mongodb-primary-data`, `mongodb-secondary-N-data`: MongoDB data

### Monitoring and Logging
- `prometheus-prod-data`: Prometheus metrics data
- `grafana-prod-data`: Grafana dashboards and settings
- `elasticsearch-prod-data`: Elasticsearch logs

## Security Considerations

### Production Security Features

1. **Container Security:**
   - Non-root user execution (`--user 1000:1000`)
   - Read-only root filesystem (`--read-only`)
   - No new privileges (`--security-opt no-new-privileges:true`)
   - Resource limits (`--memory`, `--cpus`)

2. **Network Security:**
   - Isolated Docker networks
   - Internal service communication
   - Minimal port exposure

3. **Secrets Management:**
   - Environment file separation
   - Vault integration (optional)
   - Encrypted passwords

4. **SSL/TLS:**
   - HTTPS termination at load balancer
   - Internal service encryption
   - Certificate management

## Monitoring and Observability

### Available Monitoring Stacks

1. **Prometheus + Grafana:**
   - Application metrics
   - Infrastructure monitoring
   - Custom dashboards

2. **ELK Stack (Elasticsearch + Logstash + Kibana):**
   - Centralized logging
   - Log analysis and search
   - Real-time monitoring

3. **Database-specific Monitoring:**
   - PostgreSQL: `postgres_exporter`
   - Redis: `redis_exporter`
   - MongoDB: `mongodb_exporter`
   - MinIO: Built-in metrics

### Accessing Monitoring

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601
- **MinIO Console**: http://localhost:9001

## Backup and Recovery

### Automated Backup Features

1. **PostgreSQL:**
   - Continuous WAL archiving
   - Point-in-time recovery
   - Automated cleanup

2. **Redis:**
   - RDB snapshots
   - AOF persistence
   - Scheduled backups

3. **MongoDB:**
   - Replica set backups
   - Incremental backups
   - Compressed archives

4. **MinIO:**
   - Cross-bucket replication
   - Versioning support
   - Lifecycle policies

### Backup Schedules

- **Daily**: 2:00 AM UTC (PostgreSQL, MongoDB)
- **Daily**: 3:00 AM UTC (MinIO, Redis)
- **Retention**: 30 days default

## Troubleshooting

### Common Issues

1. **Port Conflicts:**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000

   # Use different port
   ./dev/run-dev.sh --port 8001
   ```

2. **Container Cleanup:**
   ```bash
   # Stop and remove all containers
   ./prod/run-prod.sh --stop

   # Clean up with force
   ./prod/run-prod.sh --clean
   ```

3. **Storage Issues:**
   ```bash
   # Check volume usage
   docker volume ls | grep anomaly_detection

   # Clean up volumes (WARNING: Data loss)
   docker volume prune
   ```

4. **Network Issues:**
   ```bash
   # Check networks
   docker network ls | grep anomaly_detection

   # Recreate network
   docker network rm anomaly_detection-dev
   docker network create anomaly_detection-dev
   ```

### Health Checks

All production containers include health checks:
```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# View health check logs
docker inspect --format='{{json .State.Health}}' container-name
```

### Logs and Debugging

```bash
# View container logs
docker logs -f anomaly_detection-prod-1

# View aggregated logs
docker logs -f $(docker ps -q --filter name=anomaly_detection)

# Enter container for debugging
docker exec -it anomaly_detection-prod-1 /bin/bash
```

## Best Practices

### Development

1. Use `--build` flag when code changes
2. Regularly clean up development containers
3. Monitor resource usage during development
4. Use environment-specific configurations

### Testing

1. Always use isolated test environments
2. Run tests with coverage reporting
3. Include integration tests with storage
4. Benchmark performance regularly

### Production

1. Always use hardened Docker images
2. Implement proper secret management
3. Monitor all services continuously
4. Set up automated backups
5. Test disaster recovery procedures
6. Use SSL/TLS in production
7. Implement proper logging and alerting

### Storage

1. Use clustering for high availability
2. Monitor storage performance and capacity
3. Implement automated backups
4. Test backup restoration procedures
5. Use appropriate storage types for workloads

## Support

For issues or questions:

1. Check container logs for errors
2. Verify environment configuration
3. Ensure Docker daemon is running
4. Check network connectivity
5. Review Docker resource limits

Each script includes `--help` option for detailed usage information.
