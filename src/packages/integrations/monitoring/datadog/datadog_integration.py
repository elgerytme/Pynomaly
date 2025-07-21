"""
Datadog Integration for Enterprise Monitoring

Provides comprehensive integration with Datadog for metrics collection,
alerting, and observability of ML models and infrastructure.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
import json

from structlog import get_logger
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.metrics_api import MetricsApi
from datadog_api_client.v1.api.events_api import EventsApi
from datadog_api_client.v1.api.dashboards_api import DashboardsApi
from datadog_api_client.v1.api.monitors_api import MonitorsApi
from datadog_api_client.v1.model.metrics_payload import MetricsPayload
from datadog_api_client.v1.model.series import Series
from datadog_api_client.v1.model.event_create_request import EventCreateRequest
from datadog_api_client.v1.model.dashboard import Dashboard
from datadog_api_client.v1.model.monitor import Monitor
import httpx

logger = get_logger(__name__)


class DatadogIntegration:
    """
    Datadog integration for enterprise monitoring.
    
    Provides comprehensive integration with Datadog for metrics,
    events, dashboards, and alerting for ML operations.
    """
    
    def __init__(
        self,
        api_key: str,
        app_key: str,
        site: str = "datadoghq.com",
        api_url: Optional[str] = None
    ):
        self.api_key = api_key
        self.app_key = app_key
        self.site = site
        self.api_url = api_url or f"https://api.{site}"
        
        self.configuration = None
        self.metrics_api = None
        self.events_api = None
        self.dashboards_api = None
        self.monitors_api = None
        self.logger = logger.bind(integration="datadog")
        
        self.logger.info("DatadogIntegration initialized", site=site)
    
    async def connect(self) -> bool:
        """Establish connection to Datadog."""
        self.logger.info("Connecting to Datadog")
        
        try:
            # Configure Datadog API client
            self.configuration = Configuration()
            self.configuration.api_key["apiKeyAuth"] = self.api_key
            self.configuration.api_key["appKeyAuth"] = self.app_key
            self.configuration.server_variables["site"] = self.site
            
            # Create API clients
            api_client = ApiClient(self.configuration)
            self.metrics_api = MetricsApi(api_client)
            self.events_api = EventsApi(api_client)
            self.dashboards_api = DashboardsApi(api_client)
            self.monitors_api = MonitorsApi(api_client)
            
            # Test connection
            await self._test_connection()
            
            self.logger.info("Successfully connected to Datadog")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to Datadog: {str(e)}"
            self.logger.error(error_msg)
            return False
    
    async def send_metrics(
        self,
        metrics: List[Dict[str, Any]],
        tags: Optional[List[str]] = None
    ) -> bool:
        """Send metrics to Datadog."""
        if not self.metrics_api:
            raise RuntimeError("Not connected to Datadog")
        
        self.logger.debug("Sending metrics to Datadog", count=len(metrics))
        
        try:
            series_list = []
            
            for metric in metrics:
                # Build metric series
                series = Series(
                    metric=metric["name"],
                    points=[[metric.get("timestamp", datetime.utcnow().timestamp()), metric["value"]]],
                    type=metric.get("type", "gauge"),
                    host=metric.get("host"),
                    tags=tags or []
                )
                
                # Add metric-specific tags
                if "tags" in metric:
                    series.tags.extend(metric["tags"])
                
                series_list.append(series)
            
            # Send metrics payload
            payload = MetricsPayload(series=series_list)
            response = self.metrics_api.submit_metrics(body=payload)
            
            self.logger.debug("Metrics sent successfully", status=response["status"])
            return True
            
        except Exception as e:
            self.logger.error("Failed to send metrics", error=str(e))
            return False
    
    async def send_model_performance_metrics(
        self,
        model_id: UUID,
        deployment_id: UUID,
        metrics: Dict[str, float],
        tenant_id: UUID,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Send ML model performance metrics to Datadog."""
        self.logger.debug("Sending model performance metrics", model_id=model_id)
        
        try:
            base_tags = [
                f"model_id:{model_id}",
                f"deployment_id:{deployment_id}",
                f"tenant_id:{tenant_id}",
                "service:anomaly_detection",
                "component:ml_model"
            ]
            
            if tags:
                base_tags.extend(tags)
            
            # Convert metrics to Datadog format
            datadog_metrics = []
            timestamp = datetime.utcnow().timestamp()
            
            for metric_name, value in metrics.items():
                datadog_metrics.append({
                    "name": f"anomaly_detection.model.{metric_name}",
                    "value": value,
                    "timestamp": timestamp,
                    "type": "gauge",
                    "tags": base_tags
                })
            
            return await self.send_metrics(datadog_metrics)
            
        except Exception as e:
            self.logger.error("Failed to send model performance metrics", error=str(e))
            return False
    
    async def send_anomaly_detection_metrics(
        self,
        tenant_id: UUID,
        data_source: str,
        anomaly_count: int,
        total_records: int,
        anomaly_score: float,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Send anomaly detection metrics to Datadog."""
        self.logger.debug("Sending anomaly detection metrics", data_source=data_source)
        
        try:
            base_tags = [
                f"tenant_id:{tenant_id}",
                f"data_source:{data_source}",
                "service:anomaly_detection",
                "component:anomaly_detection"
            ]
            
            if tags:
                base_tags.extend(tags)
            
            timestamp = datetime.utcnow().timestamp()
            anomaly_rate = (anomaly_count / max(total_records, 1)) * 100
            
            metrics = [
                {
                    "name": "anomaly_detection.anomaly.count",
                    "value": anomaly_count,
                    "timestamp": timestamp,
                    "type": "count",
                    "tags": base_tags
                },
                {
                    "name": "anomaly_detection.anomaly.rate",
                    "value": anomaly_rate,
                    "timestamp": timestamp,
                    "type": "gauge",
                    "tags": base_tags
                },
                {
                    "name": "anomaly_detection.anomaly.score",
                    "value": anomaly_score,
                    "timestamp": timestamp,
                    "type": "gauge",
                    "tags": base_tags
                },
                {
                    "name": "anomaly_detection.records.processed",
                    "value": total_records,
                    "timestamp": timestamp,
                    "type": "count",
                    "tags": base_tags
                }
            ]
            
            return await self.send_metrics(metrics)
            
        except Exception as e:
            self.logger.error("Failed to send anomaly detection metrics", error=str(e))
            return False
    
    async def send_event(
        self,
        title: str,
        text: str,
        alert_type: str = "info",
        tags: Optional[List[str]] = None,
        source_type_name: Optional[str] = None
    ) -> bool:
        """Send event to Datadog."""
        if not self.events_api:
            raise RuntimeError("Not connected to Datadog")
        
        self.logger.debug("Sending event to Datadog", title=title)
        
        try:
            event = EventCreateRequest(
                title=title,
                text=text,
                alert_type=alert_type,
                tags=tags or [],
                source_type_name=source_type_name or "anomaly_detection"
            )
            
            response = self.events_api.create_event(body=event)
            
            self.logger.debug("Event sent successfully", event_id=response["event"]["id"])
            return True
            
        except Exception as e:
            self.logger.error("Failed to send event", error=str(e))
            return False
    
    async def create_model_deployment_event(
        self,
        model_id: UUID,
        deployment_id: UUID,
        deployment_name: str,
        status: str,
        tenant_id: UUID,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create model deployment event in Datadog."""
        self.logger.debug("Creating model deployment event", deployment=deployment_name)
        
        try:
            title = f"Model Deployment {status.title()}: {deployment_name}"
            
            text_parts = [
                f"**Model ID:** {model_id}",
                f"**Deployment ID:** {deployment_id}",
                f"**Status:** {status}",
                f"**Tenant:** {tenant_id}"
            ]
            
            if details:
                text_parts.extend([f"**{k}:** {v}" for k, v in details.items()])
            
            text = "\n".join(text_parts)
            
            alert_type = "success" if status == "deployed" else "error" if status == "failed" else "info"
            
            tags = [
                f"model_id:{model_id}",
                f"deployment_id:{deployment_id}",
                f"tenant_id:{tenant_id}",
                f"status:{status}",
                "service:anomaly_detection",
                "component:model_deployment"
            ]
            
            return await self.send_event(title, text, alert_type, tags)
            
        except Exception as e:
            self.logger.error("Failed to create deployment event", error=str(e))
            return False
    
    async def create_dashboard(
        self,
        title: str,
        widgets: List[Dict[str, Any]],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """Create Datadog dashboard."""
        if not self.dashboards_api:
            raise RuntimeError("Not connected to Datadog")
        
        self.logger.info("Creating Datadog dashboard", title=title)
        
        try:
            # Build dashboard configuration
            dashboard_config = {
                "title": title,
                "description": description or f"anomaly_detection dashboard: {title}",
                "widgets": widgets,
                "layout_type": "ordered",
                "is_read_only": False,
                "notify_list": [],
                "tags": tags or ["service:anomaly_detection"]
            }
            
            dashboard = Dashboard(**dashboard_config)
            response = self.dashboards_api.create_dashboard(body=dashboard)
            
            dashboard_id = response["id"]
            self.logger.info("Dashboard created successfully", dashboard_id=dashboard_id)
            
            return dashboard_id
            
        except Exception as e:
            self.logger.error("Failed to create dashboard", error=str(e))
            return None
    
    async def create_model_monitoring_dashboard(
        self,
        model_id: UUID,
        model_name: str,
        tenant_id: UUID
    ) -> Optional[str]:
        """Create ML model monitoring dashboard."""
        self.logger.info("Creating model monitoring dashboard", model=model_name)
        
        try:
            # Define dashboard widgets
            widgets = [
                {
                    "definition": {
                        "title": "Model Request Rate",
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": f"sum:anomaly_detection.model.request_count{{model_id:{model_id}}}.as_rate()",
                                "display_type": "line",
                                "style": {"palette": "dog_classic"}
                            }
                        ]
                    }
                },
                {
                    "definition": {
                        "title": "Model Error Rate",
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": f"sum:anomaly_detection.model.error_rate{{model_id:{model_id}}}",
                                "display_type": "line",
                                "style": {"palette": "warm"}
                            }
                        ]
                    }
                },
                {
                    "definition": {
                        "title": "Model Latency",
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": f"avg:anomaly_detection.model.latency{{model_id:{model_id}}}",
                                "display_type": "line",
                                "style": {"palette": "cool"}
                            }
                        ]
                    }
                },
                {
                    "definition": {
                        "title": "Prediction Accuracy",
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": f"avg:anomaly_detection.model.accuracy{{model_id:{model_id}}}",
                                "display_type": "line",
                                "style": {"palette": "green"}
                            }
                        ]
                    }
                }
            ]
            
            dashboard_title = f"ML Model Monitoring - {model_name}"
            description = f"Monitoring dashboard for ML model {model_name} (ID: {model_id})"
            
            tags = [
                f"model_id:{model_id}",
                f"tenant_id:{tenant_id}",
                "service:anomaly_detection",
                "component:ml_monitoring"
            ]
            
            return await self.create_dashboard(dashboard_title, widgets, description, tags)
            
        except Exception as e:
            self.logger.error("Failed to create model monitoring dashboard", error=str(e))
            return None
    
    async def create_monitor(
        self,
        name: str,
        query: str,
        message: str,
        tags: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create Datadog monitor."""
        if not self.monitors_api:
            raise RuntimeError("Not connected to Datadog")
        
        self.logger.info("Creating Datadog monitor", name=name)
        
        try:
            monitor_config = {
                "name": name,
                "type": "metric alert",
                "query": query,
                "message": message,
                "tags": tags or [],
                "options": options or {}
            }
            
            monitor = Monitor(**monitor_config)
            response = self.monitors_api.create_monitor(body=monitor)
            
            monitor_id = str(response["id"])
            self.logger.info("Monitor created successfully", monitor_id=monitor_id)
            
            return monitor_id
            
        except Exception as e:
            self.logger.error("Failed to create monitor", error=str(e))
            return None
    
    async def create_anomaly_detection_monitor(
        self,
        tenant_id: UUID,
        data_source: str,
        anomaly_threshold: float = 5.0
    ) -> Optional[str]:
        """Create monitor for anomaly detection alerts."""
        self.logger.info("Creating anomaly detection monitor", data_source=data_source)
        
        try:
            monitor_name = f"High Anomaly Rate - {data_source}"
            
            query = f"avg(last_5m):avg:anomaly_detection.anomaly.rate{{tenant_id:{tenant_id},data_source:{data_source}}} > {anomaly_threshold}"
            
            message = f"""
            **High anomaly rate detected in {data_source}**
            
            The anomaly detection rate has exceeded {anomaly_threshold}% in the last 5 minutes.
            
            **Tenant:** {tenant_id}
            **Data Source:** {data_source}
            
            Please investigate the data source for potential issues.
            
            @pagerduty-anomaly_detection
            """
            
            tags = [
                f"tenant_id:{tenant_id}",
                f"data_source:{data_source}",
                "service:anomaly_detection",
                "component:anomaly_detection",
                "alert_type:anomaly_rate"
            ]
            
            options = {
                "thresholds": {
                    "critical": anomaly_threshold,
                    "warning": anomaly_threshold * 0.8
                },
                "notify_audit": False,
                "timeout_h": 0,
                "include_tags": True,
                "no_data_timeframe": 10,
                "require_full_window": True,
                "new_host_delay": 300,
                "notify_no_data": True,
                "renotify_interval": 60
            }
            
            return await self.create_monitor(monitor_name, query, message, tags, options)
            
        except Exception as e:
            self.logger.error("Failed to create anomaly detection monitor", error=str(e))
            return None
    
    async def query_metrics(
        self,
        query: str,
        from_timestamp: datetime,
        to_timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Query metrics from Datadog."""
        if not self.configuration:
            raise RuntimeError("Not connected to Datadog")
        
        self.logger.debug("Querying metrics from Datadog", query=query)
        
        try:
            # Use direct API call for metrics query
            async with httpx.AsyncClient() as client:
                url = f"{self.api_url}/api/v1/query"
                
                params = {
                    "query": query,
                    "from": int(from_timestamp.timestamp()),
                    "to": int(to_timestamp.timestamp())
                }
                
                headers = {
                    "DD-API-KEY": self.api_key,
                    "DD-APPLICATION-KEY": self.app_key
                }
                
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                self.logger.debug("Metrics query successful")
                
                return data
                
        except Exception as e:
            self.logger.error("Failed to query metrics", error=str(e))
            return None
    
    async def get_model_metrics_summary(
        self,
        model_id: UUID,
        hours_back: int = 24
    ) -> Optional[Dict[str, Any]]:
        """Get model metrics summary from Datadog."""
        self.logger.debug("Getting model metrics summary", model_id=model_id)
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Query various model metrics
            metrics_queries = {
                "request_rate": f"avg:anomaly_detection.model.request_count{{model_id:{model_id}}}.as_rate()",
                "error_rate": f"avg:anomaly_detection.model.error_rate{{model_id:{model_id}}}",
                "latency_p95": f"p95:anomaly_detection.model.latency{{model_id:{model_id}}}",
                "accuracy": f"avg:anomaly_detection.model.accuracy{{model_id:{model_id}}}"
            }
            
            summary = {}
            for metric_name, query in metrics_queries.items():
                result = await self.query_metrics(query, start_time, end_time)
                
                if result and result.get("series"):
                    # Get latest value
                    series = result["series"][0]
                    if series["pointlist"]:
                        latest_point = series["pointlist"][-1]
                        summary[metric_name] = latest_point[1]
                else:
                    summary[metric_name] = None
            
            summary["query_period_hours"] = hours_back
            summary["model_id"] = str(model_id)
            
            return summary
            
        except Exception as e:
            self.logger.error("Failed to get model metrics summary", error=str(e))
            return None
    
    # Private helper methods
    
    async def _test_connection(self) -> None:
        """Test Datadog connection."""
        try:
            # Send a test metric
            test_metric = [{
                "name": "anomaly_detection.test.connection",
                "value": 1.0,
                "timestamp": datetime.utcnow().timestamp(),
                "type": "gauge",
                "tags": ["service:anomaly_detection", "test:connection"]
            }]
            
            success = await self.send_metrics(test_metric)
            if not success:
                raise RuntimeError("Failed to send test metric")
            
            self.logger.info("Connection test successful")
            
        except Exception as e:
            raise RuntimeError(f"Connection test failed: {str(e)}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        # Datadog client doesn't require explicit cleanup
        pass