#!/usr/bin/env python3
"""
Metric Collector for Pynomaly Real-time Alerting System.
This module collects system and application metrics for alert processing.
"""

import asyncio
import json
import logging
import os
import psutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import structlog
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server
from sqlalchemy import create_engine, text

from .alert_manager import get_alert_manager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class MetricCollector:
    """Comprehensive metric collector for real-time alerting."""
    
    def __init__(self, database_url: str, prometheus_port: int = 8000):
        """Initialize metric collector."""
        self.database_url = database_url
        self.prometheus_port = prometheus_port
        self.alert_manager = get_alert_manager()
        
        # Collection intervals
        self.system_metrics_interval = 30  # seconds
        self.app_metrics_interval = 60    # seconds
        self.custom_metrics_interval = 120  # seconds
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Collection tasks
        self.collection_tasks: List[asyncio.Task] = []
        self.running = False
        
        # HTTP session for external metrics
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        logger.info("Metric collector initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent', 
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            ['path'],
            registry=self.registry
        )
        
        self.network_bytes_sent = Counter(
            'system_network_bytes_sent_total',
            'Total network bytes sent',
            registry=self.registry
        )
        
        self.network_bytes_recv = Counter(
            'system_network_bytes_recv_total',
            'Total network bytes received',
            registry=self.registry
        )
        
        # Application metrics
        self.app_response_time = Histogram(
            'app_response_time_seconds',
            'Application response time',
            ['endpoint', 'method'],
            registry=self.registry
        )
        
        self.app_requests_total = Counter(
            'app_requests_total',
            'Total application requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        self.app_errors_total = Counter(
            'app_errors_total',
            'Total application errors',
            ['error_type'],
            registry=self.registry
        )
        
        # Database metrics
        self.db_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.db_query_time = Histogram(
            'database_query_time_seconds',
            'Database query execution time',
            ['query_type'],
            registry=self.registry
        )
        
        # Business metrics
        self.anomaly_detection_count = Counter(
            'anomaly_detection_count_total',
            'Total anomaly detections',
            ['detector_type', 'result'],
            registry=self.registry
        )
        
        self.model_training_time = Histogram(
            'model_training_time_seconds',
            'Model training time',
            ['model_type'],
            registry=self.registry
        )
        
        self.alert_count = Counter(
            'alert_count_total',
            'Total alerts triggered',
            ['severity', 'rule_name'],
            registry=self.registry
        )
    
    async def start(self):
        """Start metric collection."""
        if self.running:
            return
        
        self.running = True
        self.http_session = aiohttp.ClientSession()
        
        # Start Prometheus metrics server
        start_http_server(self.prometheus_port, registry=self.registry)
        logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        
        # Start collection tasks
        self.collection_tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._collect_app_metrics()),
            asyncio.create_task(self._collect_custom_metrics()),
        ]
        
        await self.alert_manager.start()
        
        logger.info("Metric collection started")
    
    async def stop(self):
        """Stop metric collection."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel collection tasks
        for task in self.collection_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.collection_tasks, return_exceptions=True)
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
        
        await self.alert_manager.stop()
        
        logger.info("Metric collection stopped")
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        while self.running:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.set(cpu_percent)
                await self.alert_manager.process_metric("system.cpu.usage", cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.memory_usage.set(memory_percent)
                await self.alert_manager.process_metric("system.memory.usage", memory_percent)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.disk_usage.labels(path='/').set(disk_percent)
                await self.alert_manager.process_metric("system.disk.usage", disk_percent)
                
                # Network metrics
                network = psutil.net_io_counters()
                self.network_bytes_sent.inc(network.bytes_sent)
                self.network_bytes_recv.inc(network.bytes_recv)
                
                # Process metrics
                process_count = len(psutil.pids())
                await self.alert_manager.process_metric("system.process.count", process_count)
                
                # Load average (Unix only)
                if hasattr(os, 'getloadavg'):
                    load_avg = os.getloadavg()[0]
                    await self.alert_manager.process_metric("system.load.average", load_avg)
                
                logger.debug(f"System metrics collected: CPU={cpu_percent}%, Memory={memory_percent}%, Disk={disk_percent}%")
                
            except Exception as e:
                logger.error(f"Failed to collect system metrics: {e}")
            
            await asyncio.sleep(self.system_metrics_interval)
    
    async def _collect_app_metrics(self):
        """Collect application metrics."""
        while self.running:
            try:
                # Database metrics
                await self._collect_database_metrics()
                
                # Redis metrics (if configured)
                await self._collect_redis_metrics()
                
                # Application health metrics
                await self._collect_health_metrics()
                
                logger.debug("Application metrics collected")
                
            except Exception as e:
                logger.error(f"Failed to collect application metrics: {e}")
            
            await asyncio.sleep(self.app_metrics_interval)
    
    async def _collect_database_metrics(self):
        """Collect database metrics."""
        try:
            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                # Connection count
                if "postgresql" in self.database_url:
                    result = conn.execute(text("SELECT count(*) FROM pg_stat_activity"))
                    connection_count = result.fetchone()[0]
                elif "mysql" in self.database_url:
                    result = conn.execute(text("SHOW STATUS LIKE 'Threads_connected'"))
                    connection_count = result.fetchone()[1]
                else:
                    connection_count = 1  # SQLite has single connection
                
                self.db_connections.set(connection_count)
                await self.alert_manager.process_metric("database.connections", connection_count)
                
                # Query performance
                start_time = time.time()
                conn.execute(text("SELECT 1"))
                query_time = time.time() - start_time
                
                self.db_query_time.labels(query_type='health_check').observe(query_time)
                await self.alert_manager.process_metric("database.query.time", query_time * 1000)  # milliseconds
                
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")
            await self.alert_manager.process_metric("database.error", 1)
    
    async def _collect_redis_metrics(self):
        """Collect Redis metrics."""
        try:
            redis_url = os.getenv("REDIS_URL")
            if not redis_url:
                return
            
            import redis
            r = redis.from_url(redis_url)
            
            # Redis info
            info = r.info()
            
            # Memory usage
            memory_usage = info.get('used_memory', 0)
            await self.alert_manager.process_metric("redis.memory.usage", memory_usage)
            
            # Connection count
            connected_clients = info.get('connected_clients', 0)
            await self.alert_manager.process_metric("redis.connections", connected_clients)
            
            # Hit rate
            keyspace_hits = info.get('keyspace_hits', 0)
            keyspace_misses = info.get('keyspace_misses', 0)
            total_requests = keyspace_hits + keyspace_misses
            hit_rate = (keyspace_hits / total_requests * 100) if total_requests > 0 else 0
            await self.alert_manager.process_metric("redis.hit.rate", hit_rate)
            
        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")
    
    async def _collect_health_metrics(self):
        """Collect application health metrics."""
        try:
            # Check API health
            api_url = os.getenv("API_URL", "http://localhost:8000")
            
            start_time = time.time()
            async with self.http_session.get(f"{api_url}/health") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    await self.alert_manager.process_metric("app.health.status", 1)
                    await self.alert_manager.process_metric("app.response.time", response_time * 1000)
                else:
                    await self.alert_manager.process_metric("app.health.status", 0)
                    await self.alert_manager.process_metric("app.error.rate", 1)
                
        except Exception as e:
            logger.error(f"Failed to collect health metrics: {e}")
            await self.alert_manager.process_metric("app.health.status", 0)
            await self.alert_manager.process_metric("app.error.rate", 1)
    
    async def _collect_custom_metrics(self):
        """Collect custom business metrics."""
        while self.running:
            try:
                # Collect anomaly detection metrics
                await self._collect_anomaly_metrics()
                
                # Collect model performance metrics
                await self._collect_model_metrics()
                
                # Collect alert metrics
                await self._collect_alert_metrics()
                
                logger.debug("Custom metrics collected")
                
            except Exception as e:
                logger.error(f"Failed to collect custom metrics: {e}")
            
            await asyncio.sleep(self.custom_metrics_interval)
    
    async def _collect_anomaly_metrics(self):
        """Collect anomaly detection metrics."""
        try:
            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                # Recent anomaly detection count
                if "postgresql" in self.database_url or "mysql" in self.database_url:
                    result = conn.execute(text("""
                        SELECT COUNT(*) 
                        FROM detection_results 
                        WHERE created_at >= NOW() - INTERVAL '1 hour'
                    """))
                else:
                    result = conn.execute(text("""
                        SELECT COUNT(*) 
                        FROM detection_results 
                        WHERE created_at >= datetime('now', '-1 hour')
                    """))
                
                if result.rowcount > 0:
                    detection_count = result.fetchone()[0]
                    await self.alert_manager.process_metric("anomaly.detection.count", detection_count)
                
                # Anomaly rate
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomalies
                    FROM detection_results 
                    WHERE created_at >= datetime('now', '-1 hour')
                """))
                
                if result.rowcount > 0:
                    row = result.fetchone()
                    total, anomalies = row[0], row[1]
                    anomaly_rate = (anomalies / total * 100) if total > 0 else 0
                    await self.alert_manager.process_metric("anomaly.rate", anomaly_rate)
                
        except Exception as e:
            logger.error(f"Failed to collect anomaly metrics: {e}")
    
    async def _collect_model_metrics(self):
        """Collect model performance metrics."""
        try:
            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                # Model accuracy
                result = conn.execute(text("""
                    SELECT AVG(accuracy) 
                    FROM model_evaluations 
                    WHERE created_at >= datetime('now', '-24 hours')
                """))
                
                if result.rowcount > 0:
                    accuracy = result.fetchone()[0]
                    if accuracy is not None:
                        await self.alert_manager.process_metric("model.accuracy", accuracy)
                
                # Model drift
                result = conn.execute(text("""
                    SELECT AVG(drift_score) 
                    FROM model_drift_checks 
                    WHERE created_at >= datetime('now', '-1 hour')
                """))
                
                if result.rowcount > 0:
                    drift_score = result.fetchone()[0]
                    if drift_score is not None:
                        await self.alert_manager.process_metric("model.drift", drift_score)
                
        except Exception as e:
            logger.error(f"Failed to collect model metrics: {e}")
    
    async def _collect_alert_metrics(self):
        """Collect alert system metrics."""
        try:
            # Get active alerts count
            active_alerts = await self.alert_manager.get_active_alerts()
            alert_count = len(active_alerts)
            await self.alert_manager.process_metric("alert.active.count", alert_count)
            
            # Count by severity
            severity_counts = {}
            for alert in active_alerts:
                severity = alert.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            for severity, count in severity_counts.items():
                await self.alert_manager.process_metric(f"alert.{severity}.count", count)
            
        except Exception as e:
            logger.error(f"Failed to collect alert metrics: {e}")
    
    # Custom metric methods
    async def record_anomaly_detection(self, detector_type: str, result: str, processing_time: float):
        """Record anomaly detection metrics."""
        self.anomaly_detection_count.labels(detector_type=detector_type, result=result).inc()
        await self.alert_manager.process_metric("anomaly.detection.time", processing_time)
    
    async def record_model_training(self, model_type: str, training_time: float, accuracy: float):
        """Record model training metrics."""
        self.model_training_time.labels(model_type=model_type).observe(training_time)
        await self.alert_manager.process_metric("model.training.time", training_time)
        await self.alert_manager.process_metric("model.accuracy", accuracy)
    
    async def record_api_request(self, endpoint: str, method: str, status: int, response_time: float):
        """Record API request metrics."""
        self.app_requests_total.labels(endpoint=endpoint, method=method, status=str(status)).inc()
        self.app_response_time.labels(endpoint=endpoint, method=method).observe(response_time)
        
        if status >= 400:
            self.app_errors_total.labels(error_type=f"http_{status}").inc()
            await self.alert_manager.process_metric("app.error.rate", 1)
    
    async def record_database_query(self, query_type: str, execution_time: float):
        """Record database query metrics."""
        self.db_query_time.labels(query_type=query_type).observe(execution_time)
        await self.alert_manager.process_metric("database.query.time", execution_time * 1000)
    
    async def record_custom_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record custom metric."""
        await self.alert_manager.process_metric(metric_name, value, metadata or {})


# Global metric collector instance
metric_collector = None


def get_metric_collector() -> MetricCollector:
    """Get metric collector instance."""
    global metric_collector
    if metric_collector is None:
        database_url = os.getenv("DATABASE_URL", "sqlite:///metrics.db")
        prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8000"))
        metric_collector = MetricCollector(database_url, prometheus_port)
    return metric_collector


# Make components available for import
__all__ = [
    "MetricCollector",
    "get_metric_collector",
]