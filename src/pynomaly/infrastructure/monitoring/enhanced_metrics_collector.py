#!/usr/bin/env python3
"""Enhanced metrics collection system for real-time monitoring dashboard."""

import asyncio
import platform
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import psutil
from prometheus_client import Counter, Gauge, Histogram

from pynomaly.shared.logging import get_logger

logger = get_logger(__name__)


class MetricPoint:
    """Individual metric data point."""
    
    def __init__(self, name: str, value: float, timestamp: datetime, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.value = value
        self.timestamp = timestamp
        self.tags = tags or {}
        self.id = str(uuid4())


class MetricsAggregator:
    """Aggregates and processes metrics for real-time display."""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.current_values: Dict[str, float] = {}
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
    def add_metric(self, metric: MetricPoint):
        """Add a metric point to the aggregator."""
        self.metrics_history[metric.name].append(metric)
        self.current_values[metric.name] = metric.value
        
        # Store metadata if not present
        if metric.name not in self.metric_metadata:
            self.metric_metadata[metric.name] = {
                "tags": metric.tags,
                "first_seen": metric.timestamp,
                "data_type": type(metric.value).__name__
            }
    
    def get_recent_metrics(self, metric_name: str, time_range: timedelta) -> List[MetricPoint]:
        """Get metrics within a time range."""
        cutoff_time = datetime.utcnow() - time_range
        return [
            point for point in self.metrics_history[metric_name]
            if point.timestamp >= cutoff_time
        ]
    
    def get_aggregated_value(self, metric_name: str, aggregation: str = "avg", time_range: timedelta = None) -> Optional[float]:
        """Get aggregated metric value."""
        if time_range:
            points = self.get_recent_metrics(metric_name, time_range)
        else:
            points = list(self.metrics_history[metric_name])
        
        if not points:
            return None
        
        values = [p.value for p in points]
        
        if aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "last":
            return values[-1] if values else None
        else:
            return None


class EnhancedMetricsCollector:
    """Enhanced metrics collector with real-time capabilities."""
    
    def __init__(self, collection_interval: float = 5.0):
        self.collection_interval = collection_interval
        self.aggregator = MetricsAggregator()
        self.running = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # Prometheus metrics
        self.cpu_usage_gauge = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
        self.memory_usage_gauge = Gauge('system_memory_usage_percent', 'System memory usage percentage')
        self.disk_usage_gauge = Gauge('system_disk_usage_percent', 'System disk usage percentage')
        self.network_bytes_sent = Counter('system_network_bytes_sent_total', 'Total network bytes sent')
        self.network_bytes_recv = Counter('system_network_bytes_recv_total', 'Total network bytes received')
        
        # Application metrics
        self.http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
        self.http_request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
        self.active_sessions_gauge = Gauge('active_sessions_current', 'Current active sessions')
        self.websocket_connections_gauge = Gauge('websocket_connections_current', 'Current WebSocket connections')
        
        # ML model metrics
        self.model_predictions_total = Counter('model_predictions_total', 'Total model predictions', ['model_name'])
        self.model_accuracy_gauge = Gauge('model_accuracy', 'Model accuracy score', ['model_name'])
        self.anomaly_detection_rate = Gauge('anomaly_detection_rate', 'Anomaly detection rate')
        self.model_inference_duration = Histogram('model_inference_duration_seconds', 'Model inference duration')
        
        # Business metrics
        self.anomalies_detected_total = Counter('anomalies_detected_total', 'Total anomalies detected', ['type'])
        self.data_samples_processed = Counter('data_samples_processed_total', 'Total data samples processed')
        self.system_uptime_gauge = Gauge('system_uptime_seconds', 'System uptime in seconds')
        
        # Health metrics
        self.system_health_score = Gauge('system_health_score', 'Overall system health score (0-100)')
        self.service_availability = Gauge('service_availability', 'Service availability (0-1)', ['service'])
        
        # Initialize system start time
        self.system_start_time = time.time()
        
    async def start_collection(self):
        """Start the metrics collection process."""
        if self.running:
            return
        
        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Enhanced metrics collection started")
    
    async def stop_collection(self):
        """Stop the metrics collection process."""
        if not self.running:
            return
        
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Enhanced metrics collection stopped")
    
    async def _collection_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await self._collect_health_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(1)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        timestamp = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_usage_gauge.set(cpu_percent)
        self.aggregator.add_metric(MetricPoint("cpu_usage_percent", cpu_percent, timestamp))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.memory_usage_gauge.set(memory_percent)
        self.aggregator.add_metric(MetricPoint("memory_usage_percent", memory_percent, timestamp))
        self.aggregator.add_metric(MetricPoint("memory_available_bytes", memory.available, timestamp))
        self.aggregator.add_metric(MetricPoint("memory_used_bytes", memory.used, timestamp))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.disk_usage_gauge.set(disk_percent)
        self.aggregator.add_metric(MetricPoint("disk_usage_percent", disk_percent, timestamp))
        self.aggregator.add_metric(MetricPoint("disk_free_bytes", disk.free, timestamp))
        
        # Network metrics
        network = psutil.net_io_counters()
        self.network_bytes_sent.inc(network.bytes_sent)
        self.network_bytes_recv.inc(network.bytes_recv)
        self.aggregator.add_metric(MetricPoint("network_bytes_sent", network.bytes_sent, timestamp))
        self.aggregator.add_metric(MetricPoint("network_bytes_recv", network.bytes_recv, timestamp))
        
        # System uptime
        uptime = time.time() - self.system_start_time
        self.system_uptime_gauge.set(uptime)
        self.aggregator.add_metric(MetricPoint("system_uptime", uptime, timestamp))
        
        # Load average (Unix/Linux only)
        try:
            load_avg = psutil.getloadavg()
            self.aggregator.add_metric(MetricPoint("load_average_1m", load_avg[0], timestamp))
            self.aggregator.add_metric(MetricPoint("load_average_5m", load_avg[1], timestamp))
            self.aggregator.add_metric(MetricPoint("load_average_15m", load_avg[2], timestamp))
        except (AttributeError, OSError):
            # Not available on Windows
            pass
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        timestamp = datetime.utcnow()
        
        # Process metrics
        current_process = psutil.Process()
        
        # Process CPU and memory
        process_cpu = current_process.cpu_percent()
        process_memory = current_process.memory_info()
        
        self.aggregator.add_metric(MetricPoint("process_cpu_percent", process_cpu, timestamp))
        self.aggregator.add_metric(MetricPoint("process_memory_bytes", process_memory.rss, timestamp))
        
        # Thread count
        thread_count = current_process.num_threads()
        self.aggregator.add_metric(MetricPoint("process_threads", thread_count, timestamp))
        
        # File descriptors (Unix/Linux only)
        try:
            fd_count = current_process.num_fds()
            self.aggregator.add_metric(MetricPoint("process_file_descriptors", fd_count, timestamp))
        except (AttributeError, OSError):
            # Not available on Windows
            pass
    
    async def _collect_health_metrics(self):
        """Collect health and availability metrics."""
        timestamp = datetime.utcnow()
        
        # Calculate system health score
        health_score = await self._calculate_health_score()
        self.system_health_score.set(health_score)
        self.aggregator.add_metric(MetricPoint("system_health_score", health_score, timestamp))
    
    async def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        scores = []
        
        # CPU health (lower is better)
        cpu_usage = self.aggregator.current_values.get("cpu_usage_percent", 0)
        cpu_score = max(0, 100 - cpu_usage)
        scores.append(cpu_score)
        
        # Memory health (lower is better)
        memory_usage = self.aggregator.current_values.get("memory_usage_percent", 0)
        memory_score = max(0, 100 - memory_usage)
        scores.append(memory_score)
        
        # Disk health (lower is better)
        disk_usage = self.aggregator.current_values.get("disk_usage_percent", 0)
        disk_score = max(0, 100 - disk_usage)
        scores.append(disk_score)
        
        return sum(scores) / len(scores) if scores else 0
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.http_request_duration.observe(duration)
        
        timestamp = datetime.utcnow()
        self.aggregator.add_metric(MetricPoint("http_requests_total", 1, timestamp, {
            "method": method, "endpoint": endpoint, "status": str(status)
        }))
        self.aggregator.add_metric(MetricPoint("http_request_duration_seconds", duration, timestamp))
    
    def record_model_prediction(self, model_name: str, accuracy: float, inference_time: float):
        """Record ML model prediction metrics."""
        self.model_predictions_total.labels(model_name=model_name).inc()
        self.model_accuracy_gauge.labels(model_name=model_name).set(accuracy)
        self.model_inference_duration.observe(inference_time)
        
        timestamp = datetime.utcnow()
        self.aggregator.add_metric(MetricPoint("model_predictions_total", 1, timestamp, {"model": model_name}))
        self.aggregator.add_metric(MetricPoint("model_accuracy", accuracy, timestamp, {"model": model_name}))
        self.aggregator.add_metric(MetricPoint("model_inference_duration_seconds", inference_time, timestamp))
    
    def record_anomaly_detection(self, anomaly_type: str, detection_rate: float):
        """Record anomaly detection metrics."""
        self.anomalies_detected_total.labels(type=anomaly_type).inc()
        self.anomaly_detection_rate.set(detection_rate)
        
        timestamp = datetime.utcnow()
        self.aggregator.add_metric(MetricPoint("anomalies_detected_total", 1, timestamp, {"type": anomaly_type}))
        self.aggregator.add_metric(MetricPoint("anomaly_detection_rate", detection_rate, timestamp))
    
    def update_session_count(self, count: int):
        """Update active session count."""
        self.active_sessions_gauge.set(count)
        timestamp = datetime.utcnow()
        self.aggregator.add_metric(MetricPoint("active_sessions", count, timestamp))
    
    def update_websocket_connections(self, count: int):
        """Update WebSocket connection count."""
        self.websocket_connections_gauge.set(count)
        timestamp = datetime.utcnow()
        self.aggregator.add_metric(MetricPoint("websocket_connections", count, timestamp))
    
    def get_metric_value(self, metric_name: str, aggregation: str = "last", time_range: timedelta = None) -> Optional[float]:
        """Get a metric value with optional aggregation and time range."""
        return self.aggregator.get_aggregated_value(metric_name, aggregation, time_range)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_usage": self.aggregator.current_values.get("cpu_usage_percent", 0),
                "memory_usage": self.aggregator.current_values.get("memory_usage_percent", 0),
                "disk_usage": self.aggregator.current_values.get("disk_usage_percent", 0),
                "uptime": self.aggregator.current_values.get("system_uptime", 0),
                "health_score": self.aggregator.current_values.get("system_health_score", 0),
            },
            "application": {
                "active_sessions": self.aggregator.current_values.get("active_sessions", 0),
                "websocket_connections": self.aggregator.current_values.get("websocket_connections", 0),
                "process_cpu": self.aggregator.current_values.get("process_cpu_percent", 0),
                "process_memory": self.aggregator.current_values.get("process_memory_bytes", 0),
            },
            "ml_models": {
                "anomaly_detection_rate": self.aggregator.current_values.get("anomaly_detection_rate", 0),
            },
            "available_metrics": list(self.aggregator.current_values.keys()),
        }
        return summary
    
    def get_time_series_data(self, metric_name: str, time_range: timedelta) -> List[Dict[str, Any]]:
        """Get time series data for a metric."""
        points = self.aggregator.get_recent_metrics(metric_name, time_range)
        return [
            {
                "timestamp": point.timestamp.isoformat(),
                "value": point.value,
                "tags": point.tags
            }
            for point in points
        ]