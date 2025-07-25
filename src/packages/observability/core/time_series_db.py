"""
Time Series Database Management

Provides high-performance time series data storage with configurable
retention policies, compression, and query optimization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

import asyncpg
import aioredis
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Supported time series storage backends."""
    INFLUXDB = "influxdb"
    PROMETHEUS = "prometheus"
    POSTGRESQL = "postgresql"
    REDIS = "redis"


class AggregationFunction(Enum):
    """Supported aggregation functions."""
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STDDEV = "stddev"
    PERCENTILE = "percentile"


@dataclass
class RetentionPolicy:
    """Configuration for data retention and downsampling."""
    name: str
    duration: timedelta
    resolution: timedelta
    aggregation: AggregationFunction = AggregationFunction.MEAN
    compress: bool = True
    description: str = ""


@dataclass
class TimeSeriesQuery:
    """Time series query configuration."""
    measurement: str
    start_time: datetime
    end_time: datetime
    fields: Optional[List[str]] = None
    tags: Optional[Dict[str, str]] = None
    aggregation: Optional[AggregationFunction] = None
    group_by: Optional[List[str]] = None
    window: Optional[timedelta] = None
    limit: Optional[int] = None


@dataclass
class TimeSeriesPoint:
    """A single time series data point."""
    measurement: str
    timestamp: datetime
    fields: Dict[str, Union[int, float, str]]
    tags: Dict[str, str] = field(default_factory=dict)


class TimeSeriesDatabase:
    """
    High-performance time series database abstraction with support for
    multiple backends, retention policies, and advanced querying.
    """
    
    def __init__(
        self,
        backend: StorageBackend = StorageBackend.INFLUXDB,
        connection_config: Optional[Dict[str, Any]] = None,
        default_retention_policies: Optional[List[RetentionPolicy]] = None
    ):
        self.backend = backend
        self.connection_config = connection_config or {}
        self.retention_policies: Dict[str, RetentionPolicy] = {}
        
        # Initialize backend-specific client
        self.client = None
        self.write_api = None
        self.query_api = None
        
        # Connection pools for different backends
        self.pg_pool = None
        self.redis_pool = None
        
        # Setup default retention policies
        if default_retention_policies:
            for policy in default_retention_policies:
                self.retention_policies[policy.name] = policy
        else:
            self._setup_default_retention_policies()
    
    def _setup_default_retention_policies(self) -> None:
        """Setup default retention policies for different data types."""
        policies = [
            RetentionPolicy(
                name="high_frequency",
                duration=timedelta(hours=1),
                resolution=timedelta(seconds=1),
                description="High frequency data (1 second resolution) for 1 hour"
            ),
            RetentionPolicy(
                name="medium_frequency", 
                duration=timedelta(days=1),
                resolution=timedelta(minutes=1),
                description="Medium frequency data (1 minute resolution) for 1 day"
            ),
            RetentionPolicy(
                name="low_frequency",
                duration=timedelta(days=7),
                resolution=timedelta(minutes=5),
                description="Low frequency data (5 minute resolution) for 1 week"
            ),
            RetentionPolicy(
                name="long_term",
                duration=timedelta(days=90),
                resolution=timedelta(hours=1),
                description="Long term data (1 hour resolution) for 90 days"
            ),
            RetentionPolicy(
                name="historical",
                duration=timedelta(days=365),
                resolution=timedelta(days=1),
                description="Historical data (1 day resolution) for 1 year"
            )
        ]
        
        for policy in policies:
            self.retention_policies[policy.name] = policy
    
    async def initialize(self) -> None:
        """Initialize the time series database connection."""
        try:
            if self.backend == StorageBackend.INFLUXDB:
                await self._initialize_influxdb()
            elif self.backend == StorageBackend.POSTGRESQL:
                await self._initialize_postgresql()
            elif self.backend == StorageBackend.REDIS:
                await self._initialize_redis()
            
            # Create retention policies
            await self._create_retention_policies()
            
            logger.info(f"Time series database ({self.backend.value}) initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize time series database: {e}")
            raise
    
    async def _initialize_influxdb(self) -> None:
        """Initialize InfluxDB connection."""
        url = self.connection_config.get('url', 'http://localhost:8086')
        token = self.connection_config.get('token')
        org = self.connection_config.get('org', 'mlops')
        bucket = self.connection_config.get('bucket', 'metrics')
        
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        
        # Test connection
        health = self.client.health()
        if health.status != "pass":
            raise ConnectionError(f"InfluxDB health check failed: {health.message}")
    
    async def _initialize_postgresql(self) -> None:
        """Initialize PostgreSQL connection with TimescaleDB extension."""
        dsn = self.connection_config.get('dsn', 'postgresql://localhost/timeseries')
        
        self.pg_pool = await asyncpg.create_pool(dsn, min_size=5, max_size=20)
        
        # Create TimescaleDB extension and tables
        async with self.pg_pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # Create metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    time TIMESTAMPTZ NOT NULL,
                    measurement TEXT NOT NULL,
                    tags JSONB,
                    fields JSONB NOT NULL
                );
            """)
            
            # Convert to hypertable
            try:
                await conn.execute("SELECT create_hypertable('metrics', 'time', if_not_exists => TRUE);")
            except Exception as e:
                logger.warning(f"Failed to create hypertable: {e}")
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection with RedisTimeSeries."""
        redis_url = self.connection_config.get('url', 'redis://localhost:6379')
        
        self.redis_pool = aioredis.from_url(redis_url, decode_responses=True)
        
        # Test connection
        await self.redis_pool.ping()
    
    async def _create_retention_policies(self) -> None:
        """Create retention policies in the backend."""
        if self.backend == StorageBackend.INFLUXDB:
            await self._create_influxdb_retention_policies()
        elif self.backend == StorageBackend.POSTGRESQL:
            await self._create_postgresql_retention_policies()
    
    async def _create_influxdb_retention_policies(self) -> None:
        """Create InfluxDB retention policies."""
        try:
            buckets_api = self.client.buckets_api()
            org = self.connection_config.get('org', 'mlops')
            
            for policy_name, policy in self.retention_policies.items():
                bucket_name = f"metrics_{policy_name}"
                
                # Check if bucket exists
                try:
                    bucket = buckets_api.find_bucket_by_name(bucket_name)
                    if bucket:
                        continue
                except:
                    pass
                
                # Create bucket with retention policy
                retention_rules = [{
                    "type": "expire",
                    "everySeconds": int(policy.duration.total_seconds())
                }]
                
                buckets_api.create_bucket(
                    bucket_name=bucket_name,
                    retention_rules=retention_rules,
                    org=org,
                    description=policy.description
                )
                
                logger.info(f"Created InfluxDB bucket: {bucket_name}")
                
        except Exception as e:
            logger.error(f"Failed to create InfluxDB retention policies: {e}")
    
    async def _create_postgresql_retention_policies(self) -> None:
        """Create PostgreSQL retention policies using TimescaleDB."""
        try:
            async with self.pg_pool.acquire() as conn:
                for policy_name, policy in self.retention_policies.items():
                    # Create retention policy
                    retention_interval = f"INTERVAL '{int(policy.duration.total_seconds())} seconds'"
                    
                    await conn.execute(f"""
                        SELECT add_retention_policy('metrics', {retention_interval}, if_not_exists => TRUE);
                    """)
                    
                    # Create continuous aggregate for downsampling
                    if policy.resolution > timedelta(seconds=1):
                        view_name = f"metrics_{policy_name}_agg"
                        time_bucket = f"'{int(policy.resolution.total_seconds())} seconds'"
                        
                        await conn.execute(f"""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS {view_name}
                            WITH (timescaledb.continuous) AS
                            SELECT 
                                time_bucket({time_bucket}, time) AS bucket,
                                measurement,
                                tags,
                                jsonb_object_agg(
                                    field_key, 
                                    CASE 
                                        WHEN '{policy.aggregation.value}' = 'mean' THEN avg(field_value::numeric)
                                        WHEN '{policy.aggregation.value}' = 'sum' THEN sum(field_value::numeric)
                                        WHEN '{policy.aggregation.value}' = 'min' THEN min(field_value::numeric)
                                        WHEN '{policy.aggregation.value}' = 'max' THEN max(field_value::numeric)
                                        WHEN '{policy.aggregation.value}' = 'count' THEN count(field_value::numeric)
                                        ELSE avg(field_value::numeric)
                                    END
                                ) as fields
                            FROM (
                                SELECT time, measurement, tags, 
                                       key as field_key, value as field_value
                                FROM metrics, jsonb_each_text(fields)
                            ) expanded
                            GROUP BY bucket, measurement, tags
                            WITH NO DATA;
                        """)
                        
                        logger.info(f"Created PostgreSQL continuous aggregate: {view_name}")
                
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL retention policies: {e}")
    
    async def write_point(self, point: TimeSeriesPoint, retention_policy: str = "medium_frequency") -> None:
        """Write a single time series point."""
        try:
            if self.backend == StorageBackend.INFLUXDB:
                await self._write_influxdb_point(point, retention_policy)
            elif self.backend == StorageBackend.POSTGRESQL:
                await self._write_postgresql_point(point)
            elif self.backend == StorageBackend.REDIS:
                await self._write_redis_point(point)
                
        except Exception as e:
            logger.error(f"Failed to write time series point: {e}")
            raise
    
    async def write_points(self, points: List[TimeSeriesPoint], retention_policy: str = "medium_frequency") -> None:
        """Write multiple time series points efficiently."""
        try:
            if self.backend == StorageBackend.INFLUXDB:
                await self._write_influxdb_points(points, retention_policy)
            elif self.backend == StorageBackend.POSTGRESQL:
                await self._write_postgresql_points(points)
            elif self.backend == StorageBackend.REDIS:
                await self._write_redis_points(points)
                
        except Exception as e:
            logger.error(f"Failed to write time series points: {e}")
            raise
    
    async def _write_influxdb_point(self, point: TimeSeriesPoint, retention_policy: str) -> None:
        """Write point to InfluxDB."""
        bucket = f"metrics_{retention_policy}"
        org = self.connection_config.get('org', 'mlops')
        
        influx_point = Point(point.measurement)
        
        # Add tags
        for key, value in point.tags.items():
            influx_point = influx_point.tag(key, value)
        
        # Add fields
        for key, value in point.fields.items():
            influx_point = influx_point.field(key, value)
        
        # Set timestamp
        influx_point = influx_point.time(point.timestamp)
        
        self.write_api.write(bucket=bucket, org=org, record=influx_point)
    
    async def _write_influxdb_points(self, points: List[TimeSeriesPoint], retention_policy: str) -> None:
        """Write multiple points to InfluxDB."""
        bucket = f"metrics_{retention_policy}"
        org = self.connection_config.get('org', 'mlops')
        
        influx_points = []
        for point in points:
            influx_point = Point(point.measurement)
            
            for key, value in point.tags.items():
                influx_point = influx_point.tag(key, value)
            
            for key, value in point.fields.items():
                influx_point = influx_point.field(key, value)
            
            influx_point = influx_point.time(point.timestamp)
            influx_points.append(influx_point)
        
        self.write_api.write(bucket=bucket, org=org, record=influx_points)
    
    async def _write_postgresql_point(self, point: TimeSeriesPoint) -> None:
        """Write point to PostgreSQL."""
        async with self.pg_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO metrics (time, measurement, tags, fields) VALUES ($1, $2, $3, $4)",
                point.timestamp, point.measurement, json.dumps(point.tags), json.dumps(point.fields)
            )
    
    async def _write_postgresql_points(self, points: List[TimeSeriesPoint]) -> None:
        """Write multiple points to PostgreSQL."""
        async with self.pg_pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO metrics (time, measurement, tags, fields) VALUES ($1, $2, $3, $4)",
                [(p.timestamp, p.measurement, json.dumps(p.tags), json.dumps(p.fields)) for p in points]
            )
    
    async def _write_redis_point(self, point: TimeSeriesPoint) -> None:
        """Write point to Redis TimeSeries."""
        for field_name, field_value in point.fields.items():
            if isinstance(field_value, (int, float)):
                key = f"{point.measurement}:{field_name}"
                timestamp = int(point.timestamp.timestamp() * 1000)
                
                # Add labels/tags
                labels = {f"tag_{k}": v for k, v in point.tags.items()}
                
                await self.redis_pool.execute_command(
                    "TS.ADD", key, timestamp, field_value, "LABELS", *[f"{k} {v}" for k, v in labels.items()]
                )
    
    async def _write_redis_points(self, points: List[TimeSeriesPoint]) -> None:
        """Write multiple points to Redis TimeSeries."""
        pipeline = self.redis_pool.pipeline()
        
        for point in points:
            for field_name, field_value in point.fields.items():
                if isinstance(field_value, (int, float)):
                    key = f"{point.measurement}:{field_name}"
                    timestamp = int(point.timestamp.timestamp() * 1000)
                    labels = {f"tag_{k}": v for k, v in point.tags.items()}
                    
                    pipeline.execute_command(
                        "TS.ADD", key, timestamp, field_value, "LABELS", *[f"{k} {v}" for k, v in labels.items()]
                    )
        
        await pipeline.execute()
    
    async def query(self, query: TimeSeriesQuery) -> List[Dict[str, Any]]:
        """Execute a time series query."""
        try:
            if self.backend == StorageBackend.INFLUXDB:
                return await self._query_influxdb(query)
            elif self.backend == StorageBackend.POSTGRESQL:
                return await self._query_postgresql(query)
            elif self.backend == StorageBackend.REDIS:
                return await self._query_redis(query)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
                
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise
    
    async def _query_influxdb(self, query: TimeSeriesQuery) -> List[Dict[str, Any]]:
        """Execute InfluxDB query."""
        # Build Flux query
        flux_query = f'''
            from(bucket: "metrics_medium_frequency")
            |> range(start: {query.start_time.isoformat()}, stop: {query.end_time.isoformat()})
            |> filter(fn: (r) => r._measurement == "{query.measurement}")
        '''
        
        if query.fields:
            field_filter = " or ".join([f'r._field == "{field}"' for field in query.fields])
            flux_query += f'\n    |> filter(fn: (r) => {field_filter})'
        
        if query.tags:
            for tag_key, tag_value in query.tags.items():
                flux_query += f'\n    |> filter(fn: (r) => r.{tag_key} == "{tag_value}")'
        
        if query.aggregation and query.window:
            window_duration = f"{int(query.window.total_seconds())}s"
            if query.aggregation == AggregationFunction.MEAN:
                flux_query += f'\n    |> aggregateWindow(every: {window_duration}, fn: mean)'
            elif query.aggregation == AggregationFunction.SUM:
                flux_query += f'\n    |> aggregateWindow(every: {window_duration}, fn: sum)'
            elif query.aggregation == AggregationFunction.MIN:
                flux_query += f'\n    |> aggregateWindow(every: {window_duration}, fn: min)'
            elif query.aggregation == AggregationFunction.MAX:
                flux_query += f'\n    |> aggregateWindow(every: {window_duration}, fn: max)'
            elif query.aggregation == AggregationFunction.COUNT:
                flux_query += f'\n    |> aggregateWindow(every: {window_duration}, fn: count)'
        
        if query.limit:
            flux_query += f'\n    |> limit(n: {query.limit})'
        
        # Execute query
        result = self.query_api.query(flux_query)
        
        # Convert to standard format
        data = []
        for table in result:
            for record in table.records:
                data.append({
                    'time': record.get_time(),
                    'measurement': record.get_measurement(),
                    'field': record.get_field(),
                    'value': record.get_value(),
                    'tags': {k: v for k, v in record.values.items() if k.startswith('tag_')}
                })
        
        return data
    
    async def _query_postgresql(self, query: TimeSeriesQuery) -> List[Dict[str, Any]]:
        """Execute PostgreSQL query."""
        # Build SQL query
        select_fields = "time, measurement, tags, fields"
        conditions = ["time BETWEEN $1 AND $2", "measurement = $3"]
        params = [query.start_time, query.end_time, query.measurement]
        
        if query.tags:
            for tag_key, tag_value in query.tags.items():
                conditions.append(f"tags ->> '{tag_key}' = ${len(params) + 1}")
                params.append(tag_value)
        
        sql_query = f"""
            SELECT {select_fields}
            FROM metrics
            WHERE {' AND '.join(conditions)}
            ORDER BY time
        """
        
        if query.limit:
            sql_query += f" LIMIT {query.limit}"
        
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(sql_query, *params)
        
        # Convert to standard format
        data = []
        for row in rows:
            fields_data = json.loads(row['fields'])
            tags_data = json.loads(row['tags'])
            
            if query.fields:
                # Filter only requested fields
                fields_data = {k: v for k, v in fields_data.items() if k in query.fields}
            
            for field_name, field_value in fields_data.items():
                data.append({
                    'time': row['time'],
                    'measurement': row['measurement'],
                    'field': field_name,
                    'value': field_value,
                    'tags': tags_data
                })
        
        return data
    
    async def _query_redis(self, query: TimeSeriesQuery) -> List[Dict[str, Any]]:
        """Execute Redis TimeSeries query."""
        data = []
        
        # Query each field separately
        fields_to_query = query.fields or ["*"]
        
        for field in fields_to_query:
            key = f"{query.measurement}:{field}" if field != "*" else f"{query.measurement}:*"
            
            start_ts = int(query.start_time.timestamp() * 1000)
            end_ts = int(query.end_time.timestamp() * 1000)
            
            try:
                result = await self.redis_pool.execute_command(
                    "TS.RANGE", key, start_ts, end_ts
                )
                
                for timestamp, value in result:
                    data.append({
                        'time': datetime.fromtimestamp(timestamp / 1000),
                        'measurement': query.measurement,
                        'field': field,
                        'value': float(value),
                        'tags': {}  # Redis TimeSeries labels would need separate query
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to query Redis key {key}: {e}")
        
        return data
    
    async def get_retention_policies(self) -> Dict[str, RetentionPolicy]:
        """Get all configured retention policies."""
        return self.retention_policies.copy()
    
    async def add_retention_policy(self, policy: RetentionPolicy) -> None:
        """Add a new retention policy."""
        self.retention_policies[policy.name] = policy
        await self._create_retention_policies()
    
    async def cleanup_expired_data(self) -> None:
        """Clean up expired data based on retention policies."""
        try:
            if self.backend == StorageBackend.POSTGRESQL:
                # TimescaleDB handles this automatically with retention policies
                async with self.pg_pool.acquire() as conn:
                    await conn.execute("SELECT run_job((SELECT job_id FROM timescaledb_information.jobs WHERE proc_name = 'policy_retention'));")
            
            logger.info("Cleanup of expired data completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics and health information."""
        stats = {
            "backend": self.backend.value,
            "retention_policies": len(self.retention_policies),
            "connection_status": "connected"
        }
        
        try:
            if self.backend == StorageBackend.POSTGRESQL and self.pg_pool:
                async with self.pg_pool.acquire() as conn:
                    result = await conn.fetchrow("SELECT COUNT(*) as total_points FROM metrics")
                    stats["total_points"] = result["total_points"]
                    
                    # Get table size
                    size_result = await conn.fetchrow(
                        "SELECT pg_size_pretty(pg_total_relation_size('metrics')) as table_size"
                    )
                    stats["storage_size"] = size_result["table_size"]
            
            elif self.backend == StorageBackend.REDIS and self.redis_pool:
                info = await self.redis_pool.info()
                stats["memory_usage"] = info.get("used_memory_human", "unknown")
                stats["connected_clients"] = info.get("connected_clients", 0)
        
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            stats["error"] = str(e)
        
        return stats
    
    async def close(self) -> None:
        """Close database connections."""
        try:
            if self.client:
                self.client.close()
            
            if self.pg_pool:
                await self.pg_pool.close()
            
            if self.redis_pool:
                await self.redis_pool.close()
            
            logger.info("Time series database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Default configuration for different environments
DEFAULT_RETENTION_POLICIES = [
    RetentionPolicy(
        name="real_time",
        duration=timedelta(minutes=30),
        resolution=timedelta(seconds=1),
        description="Real-time data for immediate monitoring"
    ),
    RetentionPolicy(
        name="short_term",
        duration=timedelta(hours=6),
        resolution=timedelta(seconds=10),
        description="Short-term data for recent analysis"
    ),
    RetentionPolicy(
        name="medium_term",
        duration=timedelta(days=7),
        resolution=timedelta(minutes=1),
        description="Medium-term data for trend analysis"
    ),
    RetentionPolicy(
        name="long_term",
        duration=timedelta(days=90),
        resolution=timedelta(minutes=15),
        description="Long-term data for historical analysis"
    ),
    RetentionPolicy(
        name="archive",
        duration=timedelta(days=365),
        resolution=timedelta(hours=1),
        description="Archived data for compliance and reporting"
    )
]