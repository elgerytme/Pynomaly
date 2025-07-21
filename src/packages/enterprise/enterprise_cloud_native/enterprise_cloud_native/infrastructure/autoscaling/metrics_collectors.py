"""
Metrics Collectors for Auto-scaling

Provides collectors for gathering metrics from various sources
including Prometheus, Kubernetes metrics server, and custom sources.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from structlog import get_logger
import aiohttp
import pandas as pd
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime

logger = get_logger(__name__)


class MetricsCollector(ABC):
    """
    Abstract base class for metrics collectors.
    
    Provides interface for collecting metrics from various sources
    for auto-scaling decisions and predictive modeling.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logger.bind(collector=name)
    
    @abstractmethod
    async def collect_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str],
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Collect metrics for a resource."""
        pass
    
    @abstractmethod
    async def get_current_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str]
    ) -> Dict[str, float]:
        """Get current metric values."""
        pass
    
    @abstractmethod
    async def get_historical_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str],
        duration_hours: int = 24,
        resolution_minutes: int = 5
    ) -> pd.DataFrame:
        """Get historical metrics as DataFrame."""
        pass
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to metrics source."""
        try:
            # Default implementation - subclasses should override
            return {"status": "connected", "collector": self.name}
        except Exception as e:
            return {"status": "error", "error": str(e), "collector": self.name}


class PrometheusMetricsCollector(MetricsCollector):
    """
    Prometheus metrics collector.
    
    Collects metrics from Prometheus for auto-scaling decisions
    and historical analysis.
    """
    
    def __init__(
        self,
        prometheus_url: str = "http://prometheus:9090",
        disable_ssl: bool = True
    ):
        super().__init__("prometheus")
        self.prometheus_url = prometheus_url
        self.disable_ssl = disable_ssl
        self.client = None
        
        try:
            self.client = PrometheusConnect(
                url=prometheus_url,
                disable_ssl=disable_ssl
            )
            self.logger.info("PrometheusMetricsCollector initialized", url=prometheus_url)
        except Exception as e:
            self.logger.error("Failed to initialize Prometheus client", error=str(e))
    
    async def collect_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str],
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Collect metrics from Prometheus."""
        self.logger.debug("Collecting metrics", resource=resource_name, metrics=metric_names)
        
        if not self.client:
            raise RuntimeError("Prometheus client not initialized")
        
        try:
            results = {}
            
            # Set default time range if not provided
            if time_range is None:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=5)
                time_range = (start_time, end_time)
            
            start_time, end_time = time_range
            
            for metric_name in metric_names:
                # Build Prometheus query
                query = await self._build_metric_query(metric_name, resource_name, namespace)
                
                try:
                    # Query Prometheus
                    if start_time and end_time:
                        metric_data = self.client.get_metric_range_data(
                            metric_name=query,
                            start_time=start_time,
                            end_time=end_time,
                            step="1m"
                        )
                    else:
                        metric_data = self.client.get_current_metric_value(query)
                    
                    results[metric_name] = self._process_metric_data(metric_data)
                    
                except Exception as e:
                    self.logger.warning("Failed to collect metric", metric=metric_name, error=str(e))
                    results[metric_name] = {"error": str(e)}
            
            return {
                "resource": resource_name,
                "namespace": namespace,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "metrics": results
            }
            
        except Exception as e:
            self.logger.error("Failed to collect metrics", error=str(e))
            raise
    
    async def get_current_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str]
    ) -> Dict[str, float]:
        """Get current metric values from Prometheus."""
        self.logger.debug("Getting current metrics", resource=resource_name, metrics=metric_names)
        
        if not self.client:
            raise RuntimeError("Prometheus client not initialized")
        
        try:
            current_metrics = {}
            
            for metric_name in metric_names:
                # Build Prometheus query
                query = await self._build_metric_query(metric_name, resource_name, namespace)
                
                try:
                    # Query current value
                    result = self.client.get_current_metric_value(query)
                    
                    if result:
                        # Extract value from result
                        value = self._extract_current_value(result)
                        current_metrics[metric_name] = value
                    else:
                        current_metrics[metric_name] = 0.0
                        
                except Exception as e:
                    self.logger.warning("Failed to get current metric", metric=metric_name, error=str(e))
                    current_metrics[metric_name] = 0.0
            
            return current_metrics
            
        except Exception as e:
            self.logger.error("Failed to get current metrics", error=str(e))
            raise
    
    async def get_historical_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str],
        duration_hours: int = 24,
        resolution_minutes: int = 5
    ) -> pd.DataFrame:
        """Get historical metrics from Prometheus as DataFrame."""
        self.logger.debug("Getting historical metrics", resource=resource_name, duration=duration_hours)
        
        if not self.client:
            raise RuntimeError("Prometheus client not initialized")
        
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Collect all metrics
            all_data = []
            
            for metric_name in metric_names:
                # Build Prometheus query
                query = await self._build_metric_query(metric_name, resource_name, namespace)
                
                try:
                    # Query historical data
                    metric_data = self.client.get_metric_range_data(
                        metric_name=query,
                        start_time=start_time,
                        end_time=end_time,
                        step=f"{resolution_minutes}m"
                    )
                    
                    # Process data into DataFrame format
                    metric_df = self._convert_to_dataframe(metric_data, metric_name)
                    
                    if not metric_df.empty:
                        all_data.append(metric_df)
                        
                except Exception as e:
                    self.logger.warning("Failed to get historical metric", metric=metric_name, error=str(e))
            
            # Combine all metrics into single DataFrame
            if all_data:
                # Merge on timestamp
                result_df = all_data[0]
                for df in all_data[1:]:
                    result_df = pd.merge(result_df, df, on="timestamp", how="outer")
                
                # Sort by timestamp and fill missing values
                result_df = result_df.sort_values("timestamp")
                result_df = result_df.fillna(method="ffill").fillna(0)
                
                return result_df
            else:
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=["timestamp"] + metric_names)
            
        except Exception as e:
            self.logger.error("Failed to get historical metrics", error=str(e))
            raise
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Prometheus."""
        try:
            if not self.client:
                return {"status": "error", "error": "Client not initialized", "collector": self.name}
            
            # Try a simple query
            result = self.client.get_current_metric_value("up")
            
            return {
                "status": "connected" if result is not None else "error",
                "collector": self.name,
                "url": self.prometheus_url,
                "metrics_available": len(result) if result else 0
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e), "collector": self.name}
    
    # Private helper methods
    
    async def _build_metric_query(
        self,
        metric_name: str,
        resource_name: str,
        namespace: str
    ) -> str:
        """Build Prometheus query for metric."""
        
        # Common metric mappings
        metric_queries = {
            "cpu_utilization": f'rate(container_cpu_usage_seconds_total{{pod=~"{resource_name}.*",namespace="{namespace}"}}[5m]) * 100',
            "memory_utilization": f'(container_memory_usage_bytes{{pod=~"{resource_name}.*",namespace="{namespace}"}} / container_spec_memory_limit_bytes{{pod=~"{resource_name}.*",namespace="{namespace}"}}) * 100',
            "request_rate": f'rate(http_requests_total{{service="{resource_name}",namespace="{namespace}"}}[5m])',
            "response_time": f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{service="{resource_name}",namespace="{namespace}"}}[5m]))',
            "error_rate": f'rate(http_requests_total{{service="{resource_name}",namespace="{namespace}",status=~"5.."}}[5m])',
            "queue_length": f'rabbitmq_queue_messages{{queue="{resource_name}",namespace="{namespace}"}}',
            "disk_usage": f'(1 - (node_filesystem_avail_bytes{{mountpoint="/",instance=~".*{resource_name}.*"}} / node_filesystem_size_bytes{{mountpoint="/",instance=~".*{resource_name}.*"}})) * 100',
            "network_io": f'rate(container_network_receive_bytes_total{{pod=~"{resource_name}.*",namespace="{namespace}"}}[5m]) + rate(container_network_transmit_bytes_total{{pod=~"{resource_name}.*",namespace="{namespace}"}}[5m])'
        }
        
        # Use predefined query or use metric name as-is
        return metric_queries.get(metric_name, f'{metric_name}{{pod=~"{resource_name}.*",namespace="{namespace}"}}')
    
    def _process_metric_data(self, metric_data: List[Dict]) -> Dict[str, Any]:
        """Process Prometheus metric data."""
        if not metric_data:
            return {"values": [], "count": 0}
        
        processed_data = {
            "values": [],
            "timestamps": [],
            "count": 0
        }
        
        for item in metric_data:
            if "values" in item:
                # Range query data
                for timestamp, value in item["values"]:
                    processed_data["timestamps"].append(datetime.fromtimestamp(timestamp))
                    processed_data["values"].append(float(value))
                    processed_data["count"] += 1
            elif "value" in item:
                # Instant query data
                timestamp, value = item["value"]
                processed_data["timestamps"].append(datetime.fromtimestamp(timestamp))
                processed_data["values"].append(float(value))
                processed_data["count"] = 1
        
        # Calculate statistics
        if processed_data["values"]:
            processed_data["min"] = min(processed_data["values"])
            processed_data["max"] = max(processed_data["values"])
            processed_data["avg"] = sum(processed_data["values"]) / len(processed_data["values"])
        
        return processed_data
    
    def _extract_current_value(self, result: List[Dict]) -> float:
        """Extract current value from Prometheus result."""
        if not result:
            return 0.0
        
        # Take first result and extract value
        item = result[0]
        if "value" in item:
            _, value = item["value"]
            return float(value)
        
        return 0.0
    
    def _convert_to_dataframe(
        self,
        metric_data: List[Dict],
        metric_name: str
    ) -> pd.DataFrame:
        """Convert Prometheus data to pandas DataFrame."""
        if not metric_data:
            return pd.DataFrame()
        
        data_points = []
        
        for item in metric_data:
            if "values" in item:
                for timestamp, value in item["values"]:
                    data_points.append({
                        "timestamp": datetime.fromtimestamp(timestamp),
                        metric_name: float(value)
                    })
        
        if data_points:
            return pd.DataFrame(data_points)
        else:
            return pd.DataFrame(columns=["timestamp", metric_name])


class KubernetesMetricsCollector(MetricsCollector):
    """
    Kubernetes metrics collector.
    
    Collects metrics from Kubernetes metrics server and resource APIs.
    """
    
    def __init__(self):
        super().__init__("kubernetes")
        # Initialize Kubernetes client here if needed
    
    async def collect_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str],
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Collect metrics from Kubernetes."""
        # Implementation for Kubernetes metrics collection
        # This would use the Kubernetes metrics API
        return {
            "resource": resource_name,
            "namespace": namespace,
            "metrics": {name: {"values": [], "count": 0} for name in metric_names}
        }
    
    async def get_current_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str]
    ) -> Dict[str, float]:
        """Get current metrics from Kubernetes."""
        # Mock implementation - would query Kubernetes metrics server
        return {name: 0.0 for name in metric_names}
    
    async def get_historical_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str],
        duration_hours: int = 24,
        resolution_minutes: int = 5
    ) -> pd.DataFrame:
        """Get historical metrics from Kubernetes."""
        # Kubernetes metrics server typically doesn't store historical data
        # This would need to be implemented with a time-series database
        return pd.DataFrame(columns=["timestamp"] + metric_names)


class CustomMetricsCollector(MetricsCollector):
    """
    Custom metrics collector.
    
    Collects metrics from custom sources via HTTP APIs or other protocols.
    """
    
    def __init__(self, name: str, endpoint_url: str, headers: Optional[Dict[str, str]] = None):
        super().__init__(name)
        self.endpoint_url = endpoint_url
        self.headers = headers or {}
    
    async def collect_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str],
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Collect metrics from custom endpoint."""
        async with aiohttp.ClientSession() as session:
            try:
                # Build request parameters
                params = {
                    "resource": resource_name,
                    "namespace": namespace,
                    "metrics": ",".join(metric_names)
                }
                
                if time_range:
                    start_time, end_time = time_range
                    params["start"] = start_time.isoformat()
                    params["end"] = end_time.isoformat()
                
                async with session.get(
                    self.endpoint_url,
                    params=params,
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
                        
            except Exception as e:
                self.logger.error("Failed to collect custom metrics", error=str(e))
                raise
    
    async def get_current_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str]
    ) -> Dict[str, float]:
        """Get current metrics from custom endpoint."""
        # Implementation would call custom API for current values
        return {name: 0.0 for name in metric_names}
    
    async def get_historical_metrics(
        self,
        resource_name: str,
        namespace: str,
        metric_names: List[str],
        duration_hours: int = 24,
        resolution_minutes: int = 5
    ) -> pd.DataFrame:
        """Get historical metrics from custom endpoint."""
        # Implementation would call custom API for historical data
        return pd.DataFrame(columns=["timestamp"] + metric_names)