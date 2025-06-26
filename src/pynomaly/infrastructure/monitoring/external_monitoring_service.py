"""External monitoring integration service.

This module provides comprehensive integration with external monitoring systems:
- Grafana dashboard integration and alerting
- Datadog metrics and log shipping
- New Relic APM and infrastructure monitoring
- Custom webhook integrations
- SNMP trap generation for network monitoring
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import aiohttp
import asyncio
from urllib.parse import urljoin

# Optional monitoring integrations
try:
    import datadog
    from datadog import statsd
    DATADOG_AVAILABLE = True
except ImportError:
    DATADOG_AVAILABLE = False

try:
    import newrelic.agent
    NEWRELIC_AVAILABLE = True
except ImportError:
    NEWRELIC_AVAILABLE = False

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, push_to_gateway
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MonitoringProvider(Enum):
    """Supported external monitoring providers."""
    GRAFANA = "grafana"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    PROMETHEUS = "prometheus"
    CUSTOM_WEBHOOK = "custom_webhook"
    SNMP = "snmp"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MonitoringConfiguration:
    """Configuration for external monitoring providers."""
    
    provider: MonitoringProvider
    enabled: bool = True
    
    # Connection settings
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    organization_id: Optional[str] = None
    
    # Provider-specific settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Alert settings
    default_severity: AlertSeverity = AlertSeverity.MEDIUM
    alert_endpoints: List[str] = field(default_factory=list)
    
    # Retry and timeout
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class MetricData:
    """Represents a metric to be sent to external monitoring."""
    
    name: str
    value: Union[int, float]
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    unit: Optional[str] = None
    description: Optional[str] = None


@dataclass
class AlertData:
    """Represents an alert to be sent to external monitoring."""
    
    title: str
    message: str
    severity: AlertSeverity
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Additional context
    affected_components: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)


class ExternalMonitoringProvider(ABC):
    """Abstract base class for external monitoring providers."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> None:
        """Initialize the monitoring provider."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
    
    @abstractmethod
    async def send_metric(self, metric: MetricData) -> bool:
        """Send a metric to the external monitoring system."""
        pass
    
    @abstractmethod
    async def send_alert(self, alert: AlertData) -> bool:
        """Send an alert to the external monitoring system."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test connection to the monitoring system."""
        pass
    
    async def send_batch_metrics(self, metrics: List[MetricData]) -> List[bool]:
        """Send multiple metrics in batch."""
        results = []
        for metric in metrics:
            try:
                result = await self.send_metric(metric)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to send metric {metric.name}: {e}")
                results.append(False)
        return results


class GrafanaMonitoringProvider(ExternalMonitoringProvider):
    """Grafana monitoring integration."""
    
    async def send_metric(self, metric: MetricData) -> bool:
        """Send metric to Grafana via InfluxDB or Prometheus."""
        try:
            if not self.session:
                await self.initialize()
            
            # Format metric for Grafana (assuming InfluxDB line protocol)
            timestamp = metric.timestamp or datetime.now()
            
            tags_str = ",".join([f"{k}={v}" for k, v in metric.tags.items()])
            if tags_str:
                tags_str = "," + tags_str
            
            line = f"{metric.name}{tags_str} value={metric.value} {int(timestamp.timestamp() * 1e9)}"
            
            # Send to InfluxDB endpoint
            influx_url = urljoin(self.config.endpoint_url, "/write")
            params = {
                "db": self.config.settings.get("database", "pynomaly"),
                "precision": "ns"
            }
            
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            async with self.session.post(
                influx_url,
                params=params,
                headers=headers,
                data=line
            ) as response:
                if response.status == 204:
                    logger.debug(f"Sent metric {metric.name} to Grafana")
                    return True
                else:
                    logger.error(f"Failed to send metric to Grafana: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending metric to Grafana: {e}")
            return False
    
    async def send_alert(self, alert: AlertData) -> bool:
        """Send alert to Grafana alerting."""
        try:
            if not self.session:
                await self.initialize()
            
            # Format alert for Grafana
            alert_payload = {
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "source": alert.source,
                "tags": alert.tags,
                "timestamp": (alert.timestamp or datetime.now()).isoformat(),
                "alert_id": alert.alert_id
            }
            
            # Send to Grafana alerting webhook
            alert_url = urljoin(self.config.endpoint_url, "/api/alerts")
            
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            async with self.session.post(
                alert_url,
                headers=headers,
                json=alert_payload
            ) as response:
                if response.status in [200, 201, 202]:
                    logger.info(f"Sent alert {alert.alert_id} to Grafana")
                    return True
                else:
                    logger.error(f"Failed to send alert to Grafana: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending alert to Grafana: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Grafana connection."""
        try:
            if not self.session:
                await self.initialize()
            
            health_url = urljoin(self.config.endpoint_url, "/api/health")
            
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            async with self.session.get(health_url, headers=headers) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Grafana connection test failed: {e}")
            return False


class DatadogMonitoringProvider(ExternalMonitoringProvider):
    """Datadog monitoring integration."""
    
    def __init__(self, config: MonitoringConfiguration):
        super().__init__(config)
        if DATADOG_AVAILABLE and config.api_key:
            datadog.initialize(
                api_key=config.api_key,
                app_key=config.api_secret,
                host_name=config.settings.get("host_name", "pynomaly")
            )
    
    async def send_metric(self, metric: MetricData) -> bool:
        """Send metric to Datadog."""
        try:
            if not DATADOG_AVAILABLE:
                logger.warning("Datadog library not available")
                return False
            
            timestamp = metric.timestamp or datetime.now()
            
            # Convert tags to Datadog format
            tags = [f"{k}:{v}" for k, v in metric.tags.items()]
            
            # Send metric based on type
            if metric.metric_type == MetricType.COUNTER:
                statsd.increment(
                    metric.name,
                    value=metric.value,
                    tags=tags,
                    sample_rate=1.0
                )
            elif metric.metric_type == MetricType.GAUGE:
                statsd.gauge(
                    metric.name,
                    value=metric.value,
                    tags=tags,
                    sample_rate=1.0
                )
            elif metric.metric_type == MetricType.HISTOGRAM:
                statsd.histogram(
                    metric.name,
                    value=metric.value,
                    tags=tags,
                    sample_rate=1.0
                )
            elif metric.metric_type == MetricType.TIMER:
                statsd.timing(
                    metric.name,
                    value=metric.value,
                    tags=tags,
                    sample_rate=1.0
                )
            
            logger.debug(f"Sent metric {metric.name} to Datadog")
            return True
            
        except Exception as e:
            logger.error(f"Error sending metric to Datadog: {e}")
            return False
    
    async def send_alert(self, alert: AlertData) -> bool:
        """Send alert to Datadog."""
        try:
            if not DATADOG_AVAILABLE:
                logger.warning("Datadog library not available")
                return False
            
            # Send as Datadog event
            title = f"[{alert.severity.value.upper()}] {alert.title}"
            text = alert.message
            
            tags = [f"{k}:{v}" for k, v in alert.tags.items()]
            tags.append(f"severity:{alert.severity.value}")
            tags.append(f"source:{alert.source}")
            
            datadog.api.Event.create(
                title=title,
                text=text,
                tags=tags,
                alert_type=self._severity_to_datadog_type(alert.severity),
                source_type_name="pynomaly"
            )
            
            logger.info(f"Sent alert {alert.alert_id} to Datadog")
            return True
            
        except Exception as e:
            logger.error(f"Error sending alert to Datadog: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Datadog connection."""
        try:
            if not DATADOG_AVAILABLE:
                return False
            
            # Test by sending a test metric
            statsd.increment("pynomaly.test.connection", tags=["test:true"])
            return True
            
        except Exception as e:
            logger.error(f"Datadog connection test failed: {e}")
            return False
    
    def _severity_to_datadog_type(self, severity: AlertSeverity) -> str:
        """Convert severity to Datadog alert type."""
        mapping = {
            AlertSeverity.LOW: "info",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.HIGH: "error",
            AlertSeverity.CRITICAL: "error"
        }
        return mapping.get(severity, "info")


class NewRelicMonitoringProvider(ExternalMonitoringProvider):
    """New Relic monitoring integration."""
    
    async def send_metric(self, metric: MetricData) -> bool:
        """Send metric to New Relic."""
        try:
            if not NEWRELIC_AVAILABLE:
                logger.warning("New Relic library not available")
                return False
            
            # Record custom metric
            newrelic.agent.record_custom_metric(
                metric.name,
                metric.value,
                application=newrelic.agent.current_transaction()
            )
            
            logger.debug(f"Sent metric {metric.name} to New Relic")
            return True
            
        except Exception as e:
            logger.error(f"Error sending metric to New Relic: {e}")
            return False
    
    async def send_alert(self, alert: AlertData) -> bool:
        """Send alert to New Relic."""
        try:
            if not NEWRELIC_AVAILABLE:
                logger.warning("New Relic library not available")
                return False
            
            # Record as custom event
            newrelic.agent.record_custom_event(
                "PynormalyAlert",
                {
                    "title": alert.title,
                    "message": alert.message,
                    "severity": alert.severity.value,
                    "source": alert.source,
                    "alert_id": alert.alert_id,
                    **alert.tags
                }
            )
            
            logger.info(f"Sent alert {alert.alert_id} to New Relic")
            return True
            
        except Exception as e:
            logger.error(f"Error sending alert to New Relic: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test New Relic connection."""
        try:
            if not NEWRELIC_AVAILABLE:
                return False
            
            # Test by recording a test event
            newrelic.agent.record_custom_event(
                "PynormalyConnectionTest",
                {"test": True, "timestamp": time.time()}
            )
            return True
            
        except Exception as e:
            logger.error(f"New Relic connection test failed: {e}")
            return False


class PrometheusMonitoringProvider(ExternalMonitoringProvider):
    """Prometheus monitoring integration."""
    
    def __init__(self, config: MonitoringConfiguration):
        super().__init__(config)
        self.registry = CollectorRegistry()
        self.metrics = {}
    
    async def send_metric(self, metric: MetricData) -> bool:
        """Send metric to Prometheus pushgateway."""
        try:
            if not PROMETHEUS_AVAILABLE:
                logger.warning("Prometheus client library not available")
                return False
            
            # Create or get metric
            metric_obj = self._get_or_create_metric(metric)
            
            # Set metric value
            if metric.metric_type == MetricType.COUNTER:
                metric_obj.inc(metric.value)
            elif metric.metric_type in [MetricType.GAUGE, MetricType.TIMER]:
                label_values = list(metric.tags.values())
                if label_values:
                    metric_obj.labels(*label_values).set(metric.value)
                else:
                    metric_obj.set(metric.value)
            elif metric.metric_type == MetricType.HISTOGRAM:
                label_values = list(metric.tags.values())
                if label_values:
                    metric_obj.labels(*label_values).observe(metric.value)
                else:
                    metric_obj.observe(metric.value)
            
            # Push to gateway
            gateway_url = self.config.endpoint_url or "localhost:9091"
            job_name = self.config.settings.get("job_name", "pynomaly")
            
            push_to_gateway(gateway_url, job=job_name, registry=self.registry)
            
            logger.debug(f"Sent metric {metric.name} to Prometheus")
            return True
            
        except Exception as e:
            logger.error(f"Error sending metric to Prometheus: {e}")
            return False
    
    async def send_alert(self, alert: AlertData) -> bool:
        """Send alert to Prometheus (via Alertmanager webhook)."""
        try:
            if not self.session:
                await self.initialize()
            
            # Format alert for Alertmanager
            alert_payload = {
                "alerts": [{
                    "labels": {
                        "alertname": alert.title,
                        "severity": alert.severity.value,
                        "source": alert.source,
                        **alert.tags
                    },
                    "annotations": {
                        "summary": alert.title,
                        "description": alert.message,
                        "alert_id": alert.alert_id
                    },
                    "startsAt": (alert.timestamp or datetime.now()).isoformat(),
                    "generatorURL": f"pynomaly://alerts/{alert.alert_id}"
                }]
            }
            
            # Send to Alertmanager webhook
            webhook_url = self.config.settings.get("alertmanager_webhook", 
                                                  f"{self.config.endpoint_url}/api/v1/alerts")
            
            async with self.session.post(
                webhook_url,
                json=alert_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status in [200, 201, 202]:
                    logger.info(f"Sent alert {alert.alert_id} to Prometheus/Alertmanager")
                    return True
                else:
                    logger.error(f"Failed to send alert to Prometheus: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending alert to Prometheus: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Prometheus connection."""
        try:
            if not self.session:
                await self.initialize()
            
            # Test pushgateway connection
            gateway_url = self.config.endpoint_url or "localhost:9091"
            
            async with self.session.get(f"http://{gateway_url}/metrics") as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Prometheus connection test failed: {e}")
            return False
    
    def _get_or_create_metric(self, metric_data: MetricData):
        """Get existing metric or create new one."""
        metric_key = f"{metric_data.name}_{metric_data.metric_type.value}"
        
        if metric_key not in self.metrics:
            label_names = list(metric_data.tags.keys())
            
            if metric_data.metric_type == MetricType.COUNTER:
                self.metrics[metric_key] = Counter(
                    metric_data.name,
                    metric_data.description or f"Counter metric {metric_data.name}",
                    labelnames=label_names,
                    registry=self.registry
                )
            elif metric_data.metric_type == MetricType.GAUGE:
                self.metrics[metric_key] = Gauge(
                    metric_data.name,
                    metric_data.description or f"Gauge metric {metric_data.name}",
                    labelnames=label_names,
                    registry=self.registry
                )
            elif metric_data.metric_type == MetricType.HISTOGRAM:
                self.metrics[metric_key] = Histogram(
                    metric_data.name,
                    metric_data.description or f"Histogram metric {metric_data.name}",
                    labelnames=label_names,
                    registry=self.registry
                )
        
        return self.metrics[metric_key]


class CustomWebhookProvider(ExternalMonitoringProvider):
    """Custom webhook monitoring integration."""
    
    async def send_metric(self, metric: MetricData) -> bool:
        """Send metric via custom webhook."""
        try:
            if not self.session:
                await self.initialize()
            
            payload = {
                "type": "metric",
                "name": metric.name,
                "value": metric.value,
                "metric_type": metric.metric_type.value,
                "tags": metric.tags,
                "timestamp": (metric.timestamp or datetime.now()).isoformat(),
                "unit": metric.unit,
                "description": metric.description
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            async with self.session.post(
                self.config.endpoint_url,
                json=payload,
                headers=headers
            ) as response:
                if response.status in [200, 201, 202]:
                    logger.debug(f"Sent metric {metric.name} via webhook")
                    return True
                else:
                    logger.error(f"Webhook metric send failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending metric via webhook: {e}")
            return False
    
    async def send_alert(self, alert: AlertData) -> bool:
        """Send alert via custom webhook."""
        try:
            if not self.session:
                await self.initialize()
            
            payload = {
                "type": "alert",
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "source": alert.source,
                "tags": alert.tags,
                "timestamp": (alert.timestamp or datetime.now()).isoformat(),
                "alert_id": alert.alert_id,
                "affected_components": alert.affected_components,
                "remediation_steps": alert.remediation_steps,
                "related_metrics": alert.related_metrics
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            async with self.session.post(
                self.config.endpoint_url,
                json=payload,
                headers=headers
            ) as response:
                if response.status in [200, 201, 202]:
                    logger.info(f"Sent alert {alert.alert_id} via webhook")
                    return True
                else:
                    logger.error(f"Webhook alert send failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending alert via webhook: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test webhook connection."""
        try:
            if not self.session:
                await self.initialize()
            
            test_payload = {
                "type": "test",
                "message": "Connection test from Pynomaly",
                "timestamp": datetime.now().isoformat()
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            async with self.session.post(
                self.config.endpoint_url,
                json=test_payload,
                headers=headers
            ) as response:
                return response.status in [200, 201, 202]
                
        except Exception as e:
            logger.error(f"Webhook connection test failed: {e}")
            return False


class ExternalMonitoringService:
    """Service for managing external monitoring integrations."""
    
    def __init__(self):
        self.providers: Dict[str, ExternalMonitoringProvider] = {}
        self.configurations: Dict[str, MonitoringConfiguration] = {}
        self.metrics_buffer: List[MetricData] = []
        self.alerts_buffer: List[AlertData] = []
        self.buffer_size = 100
        self.flush_interval = 60  # seconds
        self._flush_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the monitoring service."""
        # Start buffer flush task
        self._flush_task = asyncio.create_task(self._flush_buffers_periodically())
        logger.info("External monitoring service initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the monitoring service."""
        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining data
        await self.flush_buffers()
        
        # Cleanup providers
        for provider in self.providers.values():
            await provider.cleanup()
        
        logger.info("External monitoring service shutdown")
    
    def add_provider(self, name: str, config: MonitoringConfiguration) -> None:
        """Add a monitoring provider."""
        if config.provider == MonitoringProvider.GRAFANA:
            provider = GrafanaMonitoringProvider(config)
        elif config.provider == MonitoringProvider.DATADOG:
            provider = DatadogMonitoringProvider(config)
        elif config.provider == MonitoringProvider.NEW_RELIC:
            provider = NewRelicMonitoringProvider(config)
        elif config.provider == MonitoringProvider.PROMETHEUS:
            provider = PrometheusMonitoringProvider(config)
        elif config.provider == MonitoringProvider.CUSTOM_WEBHOOK:
            provider = CustomWebhookProvider(config)
        else:
            raise ValueError(f"Unsupported monitoring provider: {config.provider}")
        
        self.providers[name] = provider
        self.configurations[name] = config
        
        logger.info(f"Added monitoring provider: {name} ({config.provider.value})")
    
    def remove_provider(self, name: str) -> None:
        """Remove a monitoring provider."""
        if name in self.providers:
            asyncio.create_task(self.providers[name].cleanup())
            del self.providers[name]
            del self.configurations[name]
            logger.info(f"Removed monitoring provider: {name}")
    
    async def send_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        providers: Optional[List[str]] = None,
        buffered: bool = True
    ) -> Dict[str, bool]:
        """Send metric to specified providers."""
        metric = MetricData(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            timestamp=datetime.now()
        )
        
        if buffered:
            self.metrics_buffer.append(metric)
            if len(self.metrics_buffer) >= self.buffer_size:
                await self.flush_metrics()
            return {"buffered": True}
        else:
            return await self._send_metric_to_providers(metric, providers)
    
    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        source: str = "pynomaly",
        tags: Optional[Dict[str, str]] = None,
        providers: Optional[List[str]] = None,
        buffered: bool = False
    ) -> Dict[str, bool]:
        """Send alert to specified providers."""
        alert = AlertData(
            title=title,
            message=message,
            severity=severity,
            source=source,
            tags=tags or {},
            timestamp=datetime.now()
        )
        
        if buffered:
            self.alerts_buffer.append(alert)
            return {"buffered": True}
        else:
            return await self._send_alert_to_providers(alert, providers)
    
    async def _send_metric_to_providers(
        self,
        metric: MetricData,
        providers: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Send metric to specified providers."""
        target_providers = providers or list(self.providers.keys())
        results = {}
        
        for provider_name in target_providers:
            if provider_name in self.providers and self.configurations[provider_name].enabled:
                try:
                    result = await self.providers[provider_name].send_metric(metric)
                    results[provider_name] = result
                except Exception as e:
                    logger.error(f"Error sending metric to {provider_name}: {e}")
                    results[provider_name] = False
            else:
                results[provider_name] = False
        
        return results
    
    async def _send_alert_to_providers(
        self,
        alert: AlertData,
        providers: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Send alert to specified providers."""
        target_providers = providers or list(self.providers.keys())
        results = {}
        
        for provider_name in target_providers:
            if provider_name in self.providers and self.configurations[provider_name].enabled:
                try:
                    result = await self.providers[provider_name].send_alert(alert)
                    results[provider_name] = result
                except Exception as e:
                    logger.error(f"Error sending alert to {provider_name}: {e}")
                    results[provider_name] = False
            else:
                results[provider_name] = False
        
        return results
    
    async def flush_buffers(self) -> None:
        """Flush all buffered metrics and alerts."""
        await asyncio.gather(
            self.flush_metrics(),
            self.flush_alerts(),
            return_exceptions=True
        )
    
    async def flush_metrics(self) -> None:
        """Flush buffered metrics."""
        if not self.metrics_buffer:
            return
        
        metrics_to_send = self.metrics_buffer.copy()
        self.metrics_buffer.clear()
        
        logger.debug(f"Flushing {len(metrics_to_send)} buffered metrics")
        
        for metric in metrics_to_send:
            await self._send_metric_to_providers(metric)
    
    async def flush_alerts(self) -> None:
        """Flush buffered alerts."""
        if not self.alerts_buffer:
            return
        
        alerts_to_send = self.alerts_buffer.copy()
        self.alerts_buffer.clear()
        
        logger.debug(f"Flushing {len(alerts_to_send)} buffered alerts")
        
        for alert in alerts_to_send:
            await self._send_alert_to_providers(alert)
    
    async def _flush_buffers_periodically(self) -> None:
        """Periodically flush buffers."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_buffers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during periodic buffer flush: {e}")
    
    async def test_all_providers(self) -> Dict[str, bool]:
        """Test connection to all providers."""
        results = {}
        
        for name, provider in self.providers.items():
            if self.configurations[name].enabled:
                try:
                    result = await provider.test_connection()
                    results[name] = result
                    
                    if result:
                        logger.info(f"Provider {name} connection test: PASS")
                    else:
                        logger.warning(f"Provider {name} connection test: FAIL")
                        
                except Exception as e:
                    logger.error(f"Provider {name} connection test error: {e}")
                    results[name] = False
            else:
                results[name] = False
        
        return results
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers."""
        status = {}
        
        for name, config in self.configurations.items():
            status[name] = {
                "provider": config.provider.value,
                "enabled": config.enabled,
                "endpoint": config.endpoint_url,
                "has_credentials": bool(config.api_key),
                "buffered_metrics": len(self.metrics_buffer),
                "buffered_alerts": len(self.alerts_buffer)
            }
        
        return status


# Convenience functions for common monitoring patterns

async def send_anomaly_detection_metrics(
    service: ExternalMonitoringService,
    detector_name: str,
    dataset_name: str,
    anomaly_count: int,
    total_samples: int,
    detection_time: float,
    accuracy_score: Optional[float] = None
) -> None:
    """Send anomaly detection metrics."""
    tags = {
        "detector": detector_name,
        "dataset": dataset_name
    }
    
    await service.send_metric("anomaly.detections.count", anomaly_count, MetricType.COUNTER, tags)
    await service.send_metric("anomaly.samples.total", total_samples, MetricType.GAUGE, tags)
    await service.send_metric("anomaly.detection.time", detection_time, MetricType.TIMER, tags)
    
    if accuracy_score is not None:
        await service.send_metric("anomaly.accuracy", accuracy_score, MetricType.GAUGE, tags)
    
    # Calculate anomaly rate
    anomaly_rate = (anomaly_count / total_samples) * 100 if total_samples > 0 else 0
    await service.send_metric("anomaly.rate.percent", anomaly_rate, MetricType.GAUGE, tags)


async def send_training_job_metrics(
    service: ExternalMonitoringService,
    job_id: str,
    algorithm: str,
    trial_count: int,
    best_score: Optional[float] = None,
    execution_time: Optional[float] = None
) -> None:
    """Send training job metrics."""
    tags = {
        "job_id": job_id[:8],  # Truncated for cardinality
        "algorithm": algorithm
    }
    
    await service.send_metric("training.trials.count", trial_count, MetricType.COUNTER, tags)
    
    if best_score is not None:
        await service.send_metric("training.best_score", best_score, MetricType.GAUGE, tags)
    
    if execution_time is not None:
        await service.send_metric("training.execution_time", execution_time, MetricType.TIMER, tags)


async def send_system_health_alert(
    service: ExternalMonitoringService,
    component: str,
    issue: str,
    severity: AlertSeverity = AlertSeverity.HIGH
) -> None:
    """Send system health alert."""
    await service.send_alert(
        title=f"System Health Issue: {component}",
        message=f"Issue detected in {component}: {issue}",
        severity=severity,
        source="pynomaly.health",
        tags={"component": component, "type": "health_check"}
    )