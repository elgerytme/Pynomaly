# ADR-015: Production Database Technology Selection

🍞 **Breadcrumb:** 🏠 [Home](../../../index.md) > 👨‍💻 [Developer Guides](../../README.md) > 🏗️ [Architecture](../README.md) > 📋 [ADR](README.md) > Production Database Technology Selection

## Status

PROPOSED

## Context

### Problem Statement

The Pynomaly system requires a robust, scalable database solution for production deployments. Current development uses SQLite for simplicity, but production workloads demand higher performance, concurrent access, data integrity, and operational features. We need to select appropriate database technologies for different data patterns and operational requirements within the anomaly detection platform.

### Goals

- Select primary database technology for transactional data (jobs, configurations, metadata)
- Choose time-series database for anomaly detection metrics and monitoring data
- Ensure high availability and disaster recovery capabilities
- Support horizontal scaling for growing data volumes
- Maintain ACID compliance for critical business operations
- Enable efficient analytics and reporting queries
- Minimize operational complexity and maintenance overhead

### Constraints

- Must integrate with existing Python/FastAPI technology stack
- Must support SQLAlchemy ORM for application compatibility
- Cannot exceed budget constraints for cloud database services
- Must comply with data retention and privacy regulations
- Team has limited database administration expertise
- Must support both on-premises and cloud deployments

### Assumptions

- Data volumes will grow significantly (10x over 2 years)
- Read-heavy workloads will dominate over write operations
- Analytics queries will require complex aggregations
- High availability (99.9% uptime) is required for production
- Backup and recovery processes will be automated
- Database monitoring and alerting will be implemented

## Decision

### Chosen Solution

Implement a **multi-database architecture** with specialized databases for different data patterns:

1. **Primary Database: PostgreSQL 15+**
   - Transactional data (jobs, users, configurations)
   - JSONB support for flexible schema evolution
   - Strong ACID compliance and data integrity
   - Excellent SQLAlchemy integration

2. **Time-Series Database: TimescaleDB (PostgreSQL Extension)**
   - Anomaly detection metrics and time-series data
   - Built on PostgreSQL for operational consistency
   - Automatic data partitioning and compression
   - SQL compatibility with existing analytics tools

3. **Caching Layer: Redis 7+**
   - Session storage and application caching
   - Real-time feature flags and configuration
   - Pub/Sub for real-time notifications
   - High-performance in-memory operations

4. **Search & Analytics: PostgreSQL with Full-Text Search**
   - Log search and analysis capabilities
   - JSON document indexing for flexible queries
   - Integration with primary database for consistency

### Rationale

This approach provides:

- **Operational Simplicity**: PostgreSQL ecosystem reduces complexity
- **Performance Optimization**: Specialized databases for different workloads
- **Cost Effectiveness**: Open-source solutions with cloud-managed options
- **Team Familiarity**: PostgreSQL is well-known and widely supported
- **Scalability**: Each component can scale independently
- **Data Consistency**: ACID compliance where needed, eventual consistency for analytics

## Architecture

### System Overview

```mermaid
graph TB
    subgraph "Application Layer"
        API[FastAPI Application]
        BG[Background Workers]
        AN[Analytics Engine]
    end
    
    subgraph "Database Layer"
        PG[(PostgreSQL<br/>Primary DB)]
        TS[(TimescaleDB<br/>Time-Series)]
        RD[(Redis<br/>Cache)]
    end
    
    subgraph "Infrastructure"
        LB[Load Balancer]
        REP[Read Replicas]
        BK[Backup Storage]
    end
    
    API --> PG
    API --> RD
    BG --> PG
    BG --> TS
    AN --> TS
    AN --> REP
    
    PG --> REP
    PG --> BK
    TS --> BK
    RD --> BK
    
    LB --> API
```

### Data Flow Architecture

```mermaid
sequenceDiagram
    participant APP as Application
    participant PG as PostgreSQL
    participant TS as TimescaleDB
    participant RD as Redis
    participant REP as Read Replica
    
    Note over APP,REP: Write Operations
    APP->>PG: Create/Update Job
    APP->>TS: Store Metrics
    APP->>RD: Cache Results
    
    Note over APP,REP: Read Operations
    APP->>RD: Check Cache
    alt Cache Miss
        APP->>REP: Query Data
        REP->>APP: Return Data
        APP->>RD: Update Cache
    end
    
    Note over APP,REP: Analytics Queries
    APP->>TS: Time-series Analysis
    APP->>REP: Aggregated Reports
```

## Options Considered

### Pros and Cons Matrix

| Option | Pros | Cons | Score |
|--------|------|------|-------|
| **PostgreSQL + TimescaleDB** | ✅ ACID compliance<br/>✅ JSON support<br/>✅ Time-series optimization<br/>✅ SQL compatibility<br/>✅ Strong ecosystem | ❌ Single technology lock-in<br/>❌ Complex TimescaleDB setup | **9/10** |
| MySQL + InfluxDB | ✅ Mature MySQL<br/>✅ Optimized time-series<br/>✅ Good performance | ❌ Multi-technology complexity<br/>❌ Limited JSON support<br/>❌ InfluxDB learning curve | 7/10 |
| MongoDB + InfluxDB | ✅ Schema flexibility<br/>✅ Horizontal scaling<br/>✅ Time-series optimization | ❌ Eventual consistency<br/>❌ No ACID transactions<br/>❌ Learning curve | 6/10 |
| Cloud-Native (RDS + CloudWatch) | ✅ Managed services<br/>✅ High availability<br/>✅ Automatic backups | ❌ Vendor lock-in<br/>❌ Higher costs<br/>❌ Limited customization | 8/10 |
| All-PostgreSQL | ✅ Operational simplicity<br/>✅ Strong consistency<br/>✅ Team expertise | ❌ Not optimized for time-series<br/>❌ Performance limitations<br/>❌ Scaling challenges | 6/10 |

### Rejected Alternatives

- **MySQL + InfluxDB**: Additional operational complexity with limited benefits
- **MongoDB + InfluxDB**: Lack of ACID compliance for critical business data
- **Cloud-Native Only**: Vendor lock-in and higher costs without sufficient benefits
- **All-PostgreSQL**: Performance limitations for time-series workloads

## Implementation

### Technical Approach

#### 1. PostgreSQL Primary Database Configuration

```yaml
# PostgreSQL Configuration (postgresql.conf)
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# JSON and Performance Optimizations
log_statement = 'all'
log_duration = on
log_min_duration_statement = 1000ms
track_activity_query_size = 2048
```

#### 2. TimescaleDB Setup

```sql
-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create hypertable for metrics
CREATE TABLE anomaly_metrics (
    time TIMESTAMPTZ NOT NULL,
    job_id UUID NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    tags JSONB
);

-- Convert to hypertable
SELECT create_hypertable('anomaly_metrics', 'time');

-- Create indexes
CREATE INDEX idx_anomaly_metrics_job_id ON anomaly_metrics (job_id, time DESC);
CREATE INDEX idx_anomaly_metrics_name ON anomaly_metrics (metric_name, time DESC);
CREATE INDEX idx_anomaly_metrics_tags ON anomaly_metrics USING GIN (tags);

-- Set up compression policy
ALTER TABLE anomaly_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'job_id, metric_name'
);

SELECT add_compression_policy('anomaly_metrics', INTERVAL '7 days');
```

#### 3. Database Connection Management

```python
# Database Configuration
class DatabaseConfig:
    PRIMARY_DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/pynomaly"
    TIMESERIES_DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/pynomaly_ts"
    REDIS_URL = "redis://localhost:6379/0"
    
    # Connection Pool Settings
    DB_POOL_SIZE = 20
    DB_MAX_OVERFLOW = 30
    DB_POOL_TIMEOUT = 30
    DB_POOL_RECYCLE = 3600

# Database Factory
class DatabaseFactory:
    @staticmethod
    async def create_primary_engine():
        return create_async_engine(
            DatabaseConfig.PRIMARY_DATABASE_URL,
            pool_size=DatabaseConfig.DB_POOL_SIZE,
            max_overflow=DatabaseConfig.DB_MAX_OVERFLOW,
            pool_timeout=DatabaseConfig.DB_POOL_TIMEOUT,
            pool_recycle=DatabaseConfig.DB_POOL_RECYCLE,
            echo=False
        )
    
    @staticmethod
    async def create_timeseries_engine():
        return create_async_engine(
            DatabaseConfig.TIMESERIES_DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600
        )
```

#### 4. Repository Implementation

```python
# Multi-database Repository
class MultiDatabaseRepository:
    def __init__(self, primary_session, timeseries_session, redis_client):
        self.primary = primary_session
        self.timeseries = timeseries_session
        self.redis = redis_client
    
    async def save_job(self, job: AnomalyDetectionJob):
        # Save to primary database
        async with self.primary() as session:
            orm_job = JobORM.from_domain(job)
            session.add(orm_job)
            await session.commit()
    
    async def save_metrics(self, metrics: List[AnomalyMetric]):
        # Save to TimescaleDB
        async with self.timeseries() as session:
            for metric in metrics:
                await session.execute(
                    insert(anomaly_metrics).values(
                        time=metric.timestamp,
                        job_id=metric.job_id,
                        metric_name=metric.name,
                        metric_value=metric.value,
                        tags=metric.tags
                    )
                )
            await session.commit()
    
    async def cache_result(self, key: str, value: Any, ttl: int = 3600):
        # Cache in Redis
        await self.redis.setex(key, ttl, json.dumps(value))
```

#### 5. High Availability Setup

```yaml
# Docker Compose for HA Setup
version: '3.8'
services:
  postgresql-primary:
    image: postgres:15
    environment:
      POSTGRES_DB: pynomaly
      POSTGRES_USER: pynomaly_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgresql.conf:/etc/postgresql/postgresql.conf
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    
  postgresql-replica:
    image: postgres:15
    environment:
      PGUSER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      PGPASSWORD: ${DB_PASSWORD}
    command: |
      bash -c "
      until pg_basebackup --pgdata=/var/lib/postgresql/data --format=p --write-recovery-conf --checkpoint=fast --label=myclone --host=postgresql-primary --port=5432 --username=postgres --verbose --progress --wal-method=fetch
      do
        echo 'Waiting for primary to connect...'
        sleep 1s
      done
      echo 'Backup done, starting replica...'
      chmod 0700 /var/lib/postgresql/data
      postgres
      "
    depends_on:
      - postgresql-primary
      
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: pynomaly_ts
      POSTGRES_USER: pynomaly_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - timescale_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
```

### Migration Strategy

1. **Phase 1 (Week 1)**: Set up PostgreSQL production instance and migration scripts
2. **Phase 2 (Week 2)**: Configure TimescaleDB extension and create hypertables
3. **Phase 3 (Week 3)**: Implement Redis caching layer and session storage
4. **Phase 4 (Week 4)**: Create database connection management and repository abstractions
5. **Phase 5 (Week 5)**: Migrate existing SQLite data to PostgreSQL
6. **Phase 6 (Week 6)**: Set up read replicas and backup procedures
7. **Phase 7 (Week 7)**: Configure monitoring and alerting for database health

### Testing Strategy

- **Load Testing**: Simulate production workloads to validate performance
- **Failover Testing**: Test read replica promotion and recovery procedures
- **Backup Testing**: Validate backup and restore procedures regularly
- **Connection Pool Testing**: Test connection pool behavior under load
- **Data Integrity Testing**: Verify ACID compliance and constraint enforcement
- **Performance Benchmarking**: Compare query performance vs. current SQLite implementation

## Consequences

### Positive

- **Performance**: Optimized databases for different workload patterns
- **Scalability**: Independent scaling of transactional and analytical workloads
- **Reliability**: ACID compliance and proven database technologies
- **Operational Efficiency**: PostgreSQL ecosystem simplifies operations
- **Cost Effectiveness**: Open-source solutions with cloud deployment options
- **Future Flexibility**: Can migrate to cloud-managed services when needed

### Negative

- **Complexity**: Multiple database systems require additional operational overhead
- **Learning Curve**: Team needs to learn TimescaleDB-specific features
- **Resource Usage**: Higher memory and storage requirements for production
- **Backup Complexity**: Multiple databases require coordinated backup strategies
- **Network Overhead**: Distributed architecture increases network communication

### Neutral

- **Technology Lock-in**: PostgreSQL ecosystem lock-in vs. vendor lock-in
- **Development Overhead**: Additional abstraction layers for multi-database access
- **Monitoring Complexity**: Need to monitor multiple database systems

## Compliance

### Security Impact

- **Data Encryption**: TLS in transit and encryption at rest for all databases
- **Access Control**: Role-based access control and connection limits
- **Audit Logging**: Database activity logging for compliance requirements
- **Network Security**: VPC isolation and firewall rules for database access
- **Credential Management**: Secure credential storage and rotation

### Performance Impact

- **Query Performance**: 10-50x improvement for time-series queries with TimescaleDB
- **Connection Overhead**: Connection pooling reduces overhead for concurrent access
- **Memory Usage**: ~2-4GB baseline memory for database servers
- **Storage Performance**: SSD storage recommended for optimal I/O performance
- **Network Latency**: Sub-millisecond latency for local database connections

### Monitoring Requirements

- **Database Health**: CPU, memory, disk usage, and connection count monitoring
- **Query Performance**: Slow query logging and performance analysis
- **Replication Lag**: Monitor read replica synchronization status
- **Backup Status**: Automated backup success/failure monitoring
- **Storage Growth**: Database size growth trends and capacity planning

## Decision Log

| Date | Author | Action | Rationale |
|------|--------|--------|-----------|
| 2025-02-10 | Database Team | PROPOSED | Need for production-ready database solution |
| 2025-02-15 | Architecture Team | REVIEWED | Technical approach validated |
| 2025-02-20 | Operations Team | EVALUATED | Operational feasibility confirmed |

## References

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [Redis Documentation](https://redis.io/documentation)
- [SQLAlchemy Async Documentation](https://docs.sqlalchemy.org/en/14/orm/extensions/asyncio.html)
- [Database Reliability Engineering](https://www.oreilly.com/library/view/database-reliability-engineering/9781491925935/)
- [ADR-014: Repository & Unit-of-Work Pattern](ADR-014-repository-unit-of-work-pattern.md)

---

## 🔗 **Related Documentation**

### **Architecture**

- **[Repository & Unit-of-Work Pattern](ADR-014-repository-unit-of-work-pattern.md)** - Data access patterns
- **[Clean Architecture & DDD](ADR-013-clean-architecture-ddd-adoption.md)** - Architectural foundations
- **[ADR Index](README.md)** - All architectural decisions

### **Database**

- **[Database Schema Design](../../database/schema-design.md)** - Database design principles
- **[Migration Procedures](../../database/migration-guide.md)** - Database migration guide
- **[Performance Tuning](../../database/performance-tuning.md)** - Database optimization

### **Operations**

- **[Production Deployment](../../../deployment/README.md)** - Deployment procedures
- **[Monitoring Setup](../../../user-guides/basic-usage/monitoring.md)** - Database monitoring
- **[Backup Procedures](../../operations/backup-procedures.md)** - Data backup strategy

---

**Authors:** Database Team  
**Last Updated:** 2025-07-11  
**Next Review:** 2025-10-11
