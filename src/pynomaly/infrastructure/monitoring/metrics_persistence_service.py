"""Data persistence layer for historical metrics and time-series data."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import aiosqlite
from pydantic import BaseModel, Field


class MetricPoint(BaseModel):
    """Individual metric data point."""
    
    timestamp: datetime
    metric_name: str
    value: Union[float, int, str, bool]
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TimeSeriesQuery(BaseModel):
    """Time series query parameters."""
    
    metric_names: List[str]
    start_time: datetime
    end_time: datetime
    aggregation: str = "avg"  # avg, sum, min, max, count, last
    interval: Optional[str] = None  # 1m, 5m, 1h, 1d
    tags: Dict[str, str] = Field(default_factory=dict)
    limit: Optional[int] = None


class AggregationResult(BaseModel):
    """Aggregated metric result."""
    
    metric_name: str
    timestamp: datetime
    value: float
    count: int
    tags: Dict[str, str] = Field(default_factory=dict)


class RetentionPolicy(BaseModel):
    """Data retention policy configuration."""
    
    policy_name: str
    retention_period: timedelta
    aggregation_interval: Optional[timedelta] = None
    metric_patterns: List[str] = Field(default_factory=list)  # Glob patterns
    compression_enabled: bool = True


class MetricsPersistenceService:
    """Service for persisting and querying historical metrics data."""
    
    def __init__(
        self,
        database_path: Optional[str] = None,
        retention_policies: Optional[List[RetentionPolicy]] = None
    ):
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.db_path = database_path or "metrics.db"
        self.retention_policies = retention_policies or self._default_retention_policies()
        
        # Background tasks
        self._retention_task: Optional[asyncio.Task] = None
        self._aggregation_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 3600  # 1 hour
        
        self.logger.info(f"Metrics persistence service initialized with database: {self.db_path}")
    
    def _default_retention_policies(self) -> List[RetentionPolicy]:
        """Default retention policies for different metric types."""
        return [
            RetentionPolicy(
                policy_name="high_frequency",
                retention_period=timedelta(days=7),
                aggregation_interval=timedelta(minutes=1),
                metric_patterns=["cpu_*", "memory_*", "network_*"],
                compression_enabled=True
            ),
            RetentionPolicy(
                policy_name="application_metrics",
                retention_period=timedelta(days=30),
                aggregation_interval=timedelta(minutes=5),
                metric_patterns=["http_*", "model_*", "cache_*"],
                compression_enabled=True
            ),
            RetentionPolicy(
                policy_name="business_metrics",
                retention_period=timedelta(days=365),
                aggregation_interval=timedelta(hours=1),
                metric_patterns=["business_*", "revenue_*", "user_*"],
                compression_enabled=False
            ),
            RetentionPolicy(
                policy_name="default",
                retention_period=timedelta(days=90),
                aggregation_interval=timedelta(minutes=15),
                metric_patterns=["*"],
                compression_enabled=True
            )
        ]
    
    async def initialize(self):
        """Initialize the persistence service and database."""
        await self._create_database_schema()
        await self._start_background_tasks()
        self.logger.info("Metrics persistence service initialized")
    
    async def shutdown(self):
        """Shutdown the persistence service."""
        if self._retention_task:
            self._retention_task.cancel()
        if self._aggregation_task:
            self._aggregation_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self._retention_task, self._aggregation_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("Metrics persistence service shutdown")
    
    async def _create_database_schema(self):
        """Create database tables for metrics storage."""
        async with aiosqlite.connect(self.db_path) as db:
            # Raw metrics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    metric_name TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    value_numeric REAL,
                    value_text TEXT,
                    tags TEXT, -- JSON
                    metadata TEXT, -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Aggregated metrics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS aggregated_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    metric_name TEXT NOT NULL,
                    aggregation_type TEXT NOT NULL,
                    aggregation_interval TEXT NOT NULL,
                    value REAL NOT NULL,
                    count INTEGER NOT NULL,
                    tags TEXT, -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Retention policies table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS retention_policies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    policy_name TEXT UNIQUE NOT NULL,
                    retention_days INTEGER NOT NULL,
                    aggregation_minutes INTEGER,
                    metric_patterns TEXT, -- JSON
                    compression_enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON metrics(timestamp)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(metric_name, timestamp)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_aggregated_timestamp 
                ON aggregated_metrics(timestamp)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_aggregated_name_timestamp 
                ON aggregated_metrics(metric_name, timestamp)
            """)
            
            await db.commit()
        
        # Store retention policies
        await self._store_retention_policies()
    
    async def _store_retention_policies(self):
        """Store retention policies in database."""
        async with aiosqlite.connect(self.db_path) as db:
            for policy in self.retention_policies:
                await db.execute("""
                    INSERT OR REPLACE INTO retention_policies 
                    (policy_name, retention_days, aggregation_minutes, metric_patterns, compression_enabled)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    policy.policy_name,
                    policy.retention_period.days,
                    int(policy.aggregation_interval.total_seconds() / 60) if policy.aggregation_interval else None,
                    json.dumps(policy.metric_patterns),
                    policy.compression_enabled
                ))
            await db.commit()
    
    async def store_metric(
        self,
        metric_name: str,
        value: Union[float, int, str, bool],
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a single metric point."""
        
        timestamp = timestamp or datetime.utcnow()
        tags = tags or {}
        metadata = metadata or {}
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Determine value type and storage
                if isinstance(value, (int, float)):
                    value_type = "numeric"
                    value_numeric = float(value)
                    value_text = None
                else:
                    value_type = "text"
                    value_numeric = None
                    value_text = str(value)
                
                await db.execute("""
                    INSERT INTO metrics 
                    (timestamp, metric_name, value_type, value_numeric, value_text, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    metric_name,
                    value_type,
                    value_numeric,
                    value_text,
                    json.dumps(tags),
                    json.dumps(metadata)
                ))
                
                await db.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing metric {metric_name}: {e}")
            return False
    
    async def store_metrics_batch(self, metrics: List[MetricPoint]) -> int:
        """Store multiple metrics in a batch for better performance."""
        
        stored_count = 0
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for metric in metrics:
                    # Determine value type and storage
                    if isinstance(metric.value, (int, float)):
                        value_type = "numeric"
                        value_numeric = float(metric.value)
                        value_text = None
                    else:
                        value_type = "text"
                        value_numeric = None
                        value_text = str(metric.value)
                    
                    await db.execute("""
                        INSERT INTO metrics 
                        (timestamp, metric_name, value_type, value_numeric, value_text, tags, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metric.timestamp,
                        metric.metric_name,
                        value_type,
                        value_numeric,
                        value_text,
                        json.dumps(metric.tags),
                        json.dumps(metric.metadata)
                    ))
                    
                    stored_count += 1
                
                await db.commit()
            
            self.logger.debug(f"Stored {stored_count} metrics in batch")
            return stored_count
            
        except Exception as e:
            self.logger.error(f"Error storing metrics batch: {e}")
            return stored_count
    
    async def query_metrics(self, query: TimeSeriesQuery) -> List[MetricPoint]:
        """Query metrics with time range and filtering."""
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Build query
                conditions = ["timestamp BETWEEN ? AND ?"]
                params = [query.start_time, query.end_time]
                
                if query.metric_names:
                    placeholders = ",".join("?" * len(query.metric_names))
                    conditions.append(f"metric_name IN ({placeholders})")
                    params.extend(query.metric_names)
                
                # Tag filtering (simplified - in production, use proper JSON queries)
                for tag_key, tag_value in query.tags.items():
                    conditions.append("tags LIKE ?")
                    params.append(f'%"{tag_key}":"{tag_value}"%')
                
                where_clause = " AND ".join(conditions)
                
                sql = f"""
                    SELECT timestamp, metric_name, value_type, value_numeric, value_text, tags, metadata
                    FROM metrics
                    WHERE {where_clause}
                    ORDER BY timestamp ASC
                """
                
                if query.limit:
                    sql += f" LIMIT {query.limit}"
                
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                
                # Convert to MetricPoint objects
                metrics = []
                for row in rows:
                    timestamp, metric_name, value_type, value_numeric, value_text, tags_json, metadata_json = row
                    
                    # Parse value
                    if value_type == "numeric":
                        value = value_numeric
                    else:
                        value = value_text
                    
                    # Parse JSON fields
                    try:
                        tags = json.loads(tags_json) if tags_json else {}
                        metadata = json.loads(metadata_json) if metadata_json else {}
                    except json.JSONDecodeError:
                        tags = {}
                        metadata = {}
                    
                    metrics.append(MetricPoint(
                        timestamp=datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp,
                        metric_name=metric_name,
                        value=value,
                        tags=tags,
                        metadata=metadata
                    ))
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error querying metrics: {e}")
            return []
    
    async def query_aggregated_metrics(
        self,
        query: TimeSeriesQuery
    ) -> List[AggregationResult]:
        """Query pre-aggregated metrics for better performance."""
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Build query for aggregated data
                conditions = ["timestamp BETWEEN ? AND ?"]
                params = [query.start_time, query.end_time]
                
                if query.metric_names:
                    placeholders = ",".join("?" * len(query.metric_names))
                    conditions.append(f"metric_name IN ({placeholders})")
                    params.extend(query.metric_names)
                
                conditions.append("aggregation_type = ?")
                params.append(query.aggregation)
                
                if query.interval:
                    conditions.append("aggregation_interval = ?")
                    params.append(query.interval)
                
                where_clause = " AND ".join(conditions)
                
                sql = f"""
                    SELECT timestamp, metric_name, value, count, tags
                    FROM aggregated_metrics
                    WHERE {where_clause}
                    ORDER BY timestamp ASC
                """
                
                if query.limit:
                    sql += f" LIMIT {query.limit}"
                
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                
                # Convert to AggregationResult objects
                results = []
                for row in rows:
                    timestamp, metric_name, value, count, tags_json = row
                    
                    try:
                        tags = json.loads(tags_json) if tags_json else {}
                    except json.JSONDecodeError:
                        tags = {}
                    
                    results.append(AggregationResult(
                        metric_name=metric_name,
                        timestamp=datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp,
                        value=value,
                        count=count,
                        tags=tags
                    ))
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error querying aggregated metrics: {e}")
            return []
    
    async def get_metric_statistics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get statistical information about a metric."""
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT 
                        COUNT(*) as count,
                        MIN(value_numeric) as min_value,
                        MAX(value_numeric) as max_value,
                        AVG(value_numeric) as avg_value,
                        SUM(value_numeric) as sum_value
                    FROM metrics 
                    WHERE metric_name = ? 
                    AND timestamp BETWEEN ? AND ?
                    AND value_type = 'numeric'
                """, (metric_name, start_time, end_time)) as cursor:
                    row = await cursor.fetchone()
                
                if row:
                    count, min_val, max_val, avg_val, sum_val = row
                    return {
                        "metric_name": metric_name,
                        "time_range": {
                            "start": start_time.isoformat(),
                            "end": end_time.isoformat()
                        },
                        "statistics": {
                            "count": count or 0,
                            "min": min_val,
                            "max": max_val,
                            "average": avg_val,
                            "sum": sum_val
                        }
                    }
                
                return {
                    "metric_name": metric_name,
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    },
                    "statistics": {
                        "count": 0,
                        "min": None,
                        "max": None,
                        "average": None,
                        "sum": None
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error getting metric statistics: {e}")
            return {}
    
    async def _start_background_tasks(self):
        """Start background tasks for data retention and aggregation."""
        self._retention_task = asyncio.create_task(self._retention_cleanup_loop())
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        
        self.logger.info("Started background tasks for retention and aggregation")
    
    async def _retention_cleanup_loop(self):
        """Background loop for data retention cleanup."""
        while True:
            try:
                await self._apply_retention_policies()
                await asyncio.sleep(self._cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in retention cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _aggregation_loop(self):
        """Background loop for data aggregation."""
        while True:
            try:
                await self._perform_aggregations()
                await asyncio.sleep(self._cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _apply_retention_policies(self):
        """Apply retention policies to clean up old data."""
        async with aiosqlite.connect(self.db_path) as db:
            for policy in self.retention_policies:
                cutoff_time = datetime.utcnow() - policy.retention_period
                
                for pattern in policy.metric_patterns:
                    # Simple pattern matching (in production, use proper glob matching)
                    if pattern == "*":
                        condition = "timestamp < ?"
                        params = [cutoff_time]
                    else:
                        # Convert glob pattern to SQL LIKE pattern
                        like_pattern = pattern.replace("*", "%")
                        condition = "timestamp < ? AND metric_name LIKE ?"
                        params = [cutoff_time, like_pattern]
                    
                    # Delete old raw metrics
                    await db.execute(f"""
                        DELETE FROM metrics WHERE {condition}
                    """, params)
                    
                    # Delete old aggregated metrics
                    await db.execute(f"""
                        DELETE FROM aggregated_metrics WHERE {condition}
                    """, params)
            
            await db.commit()
            
            # Get cleanup statistics
            async with db.execute("SELECT COUNT(*) FROM metrics") as cursor:
                metrics_count = (await cursor.fetchone())[0]
            
            async with db.execute("SELECT COUNT(*) FROM aggregated_metrics") as cursor:
                aggregated_count = (await cursor.fetchone())[0]
            
            self.logger.debug(f"Retention cleanup completed. Remaining: {metrics_count} raw, {aggregated_count} aggregated")
    
    async def _perform_aggregations(self):
        """Perform data aggregation for better query performance."""
        async with aiosqlite.connect(self.db_path) as db:
            for policy in self.retention_policies:
                if not policy.aggregation_interval:
                    continue
                
                # Calculate aggregation window
                interval_minutes = int(policy.aggregation_interval.total_seconds() / 60)
                
                for pattern in policy.metric_patterns:
                    await self._aggregate_metric_pattern(db, pattern, interval_minutes)
    
    async def _aggregate_metric_pattern(
        self,
        db: aiosqlite.Connection,
        pattern: str,
        interval_minutes: int
    ):
        """Aggregate metrics matching a pattern."""
        try:
            # Find metrics that need aggregation
            if pattern == "*":
                condition = "value_type = 'numeric'"
                params = []
            else:
                like_pattern = pattern.replace("*", "%")
                condition = "value_type = 'numeric' AND metric_name LIKE ?"
                params = [like_pattern]
            
            # Get the latest aggregation timestamp
            async with db.execute(f"""
                SELECT MAX(timestamp) FROM aggregated_metrics 
                WHERE aggregation_interval = '{interval_minutes}m'
            """) as cursor:
                last_aggregation = await cursor.fetchone()
                last_aggregation = last_aggregation[0] if last_aggregation[0] else datetime.utcnow() - timedelta(days=1)
            
            # Aggregate data in windows
            current_time = datetime.utcnow()
            window_size = timedelta(minutes=interval_minutes)
            
            window_start = last_aggregation
            while window_start < current_time - window_size:
                window_end = window_start + window_size
                
                # Aggregate metrics in this window
                async with db.execute(f"""
                    SELECT 
                        metric_name,
                        AVG(value_numeric) as avg_value,
                        MIN(value_numeric) as min_value,
                        MAX(value_numeric) as max_value,
                        SUM(value_numeric) as sum_value,
                        COUNT(*) as count
                    FROM metrics 
                    WHERE {condition}
                    AND timestamp BETWEEN ? AND ?
                    GROUP BY metric_name
                """, params + [window_start, window_end]) as cursor:
                    aggregations = await cursor.fetchall()
                
                # Store aggregations
                for agg in aggregations:
                    metric_name, avg_val, min_val, max_val, sum_val, count = agg
                    
                    # Store different aggregation types
                    for agg_type, value in [
                        ("avg", avg_val),
                        ("min", min_val),
                        ("max", max_val),
                        ("sum", sum_val)
                    ]:
                        await db.execute("""
                            INSERT OR REPLACE INTO aggregated_metrics 
                            (timestamp, metric_name, aggregation_type, aggregation_interval, value, count, tags)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            window_end,
                            metric_name,
                            agg_type,
                            f"{interval_minutes}m",
                            value,
                            count,
                            json.dumps({})
                        ))
                
                window_start = window_end
            
            await db.commit()
            
        except Exception as e:
            self.logger.error(f"Error aggregating metrics for pattern {pattern}: {e}")
    
    async def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics and health information."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                stats = {}
                
                # Table sizes
                async with db.execute("SELECT COUNT(*) FROM metrics") as cursor:
                    stats["raw_metrics_count"] = (await cursor.fetchone())[0]
                
                async with db.execute("SELECT COUNT(*) FROM aggregated_metrics") as cursor:
                    stats["aggregated_metrics_count"] = (await cursor.fetchone())[0]
                
                # Database file size
                db_path = Path(self.db_path)
                if db_path.exists():
                    stats["database_size_mb"] = db_path.stat().st_size / (1024 * 1024)
                
                # Oldest and newest data
                async with db.execute("SELECT MIN(timestamp), MAX(timestamp) FROM metrics") as cursor:
                    min_time, max_time = await cursor.fetchone()
                    stats["data_range"] = {
                        "oldest": min_time,
                        "newest": max_time
                    }
                
                # Unique metrics
                async with db.execute("SELECT COUNT(DISTINCT metric_name) FROM metrics") as cursor:
                    stats["unique_metrics"] = (await cursor.fetchone())[0]
                
                return {
                    "database_path": self.db_path,
                    "statistics": stats,
                    "retention_policies": len(self.retention_policies),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting database statistics: {e}")
            return {}
    
    async def export_metrics(
        self,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime,
        format: str = "json"
    ) -> Optional[str]:
        """Export metrics data in specified format."""
        
        query = TimeSeriesQuery(
            metric_names=metric_names,
            start_time=start_time,
            end_time=end_time
        )
        
        metrics = await self.query_metrics(query)
        
        if format == "json":
            export_data = {
                "metrics": [
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "metric_name": metric.metric_name,
                        "value": metric.value,
                        "tags": metric.tags,
                        "metadata": metric.metadata
                    }
                    for metric in metrics
                ],
                "export_metadata": {
                    "exported_at": datetime.utcnow().isoformat(),
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    },
                    "count": len(metrics)
                }
            }
            return json.dumps(export_data, indent=2)
        
        elif format == "csv":
            lines = ["timestamp,metric_name,value,tags,metadata"]
            for metric in metrics:
                lines.append(f"{metric.timestamp.isoformat()},{metric.metric_name},{metric.value},{json.dumps(metric.tags)},{json.dumps(metric.metadata)}")
            return "\n".join(lines)
        
        else:
            self.logger.error(f"Unsupported export format: {format}")
            return None


# Convenience function for creating persistence service
def create_metrics_persistence_service(
    database_path: Optional[str] = None,
    retention_policies: Optional[List[RetentionPolicy]] = None
) -> MetricsPersistenceService:
    """Create and configure metrics persistence service."""
    return MetricsPersistenceService(database_path, retention_policies)