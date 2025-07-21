"""
New Relic Integration for Enterprise Monitoring

Provides comprehensive integration with New Relic for application monitoring,
metrics collection, alerting, and observability of ML models and infrastructure.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
import json
import traceback

from structlog import get_logger
import httpx
import newrelic.agent
from newrelic.api.application import application_instance
from newrelic.api.transaction import current_transaction, add_custom_attribute
from newrelic.api.external_trace import ExternalTrace
from newrelic.api.background_task import background_task
from newrelic.api.database_trace import database_trace
from newrelic.api.function_trace import function_trace
from newrelic_telemetry_sdk import MetricClient, EventClient

logger = get_logger(__name__)


class NewRelicIntegration:
    """
    New Relic integration for enterprise monitoring.
    
    Provides comprehensive integration with New Relic for metrics,
    events, dashboards, and alerting for ML operations.
    """
    
    def __init__(
        self,
        license_key: str,
        account_id: str,
        api_key: Optional[str] = None,
        insights_insert_key: Optional[str] = None,
        region: str = "US"
    ):
        self.license_key = license_key
        self.account_id = account_id
        self.api_key = api_key
        self.insights_insert_key = insights_insert_key
        self.region = region
        
        # Set base URLs based on region
        if region.upper() == "EU":
            self.base_url = "https://api.eu.newrelic.com"
            self.insights_url = "https://insights-collector.eu01.nr-data.net"
        else:
            self.base_url = "https://api.newrelic.com"
            self.insights_url = "https://insights-collector.nr-data.net"
        
        self.metric_client = None
        self.event_client = None
        self.http_client = None
        self.logger = logger.bind(integration="newrelic")
        
        self.logger.info("NewRelicIntegration initialized", account_id=account_id, region=region)
    
    async def connect(self) -> bool:
        """Establish connection to New Relic."""
        self.logger.info("Connecting to New Relic")
        
        try:
            # Initialize New Relic agent if not already done
            if not newrelic.agent.application():
                newrelic.agent.initialize()
            
            # Create telemetry clients
            if self.insights_insert_key:
                self.metric_client = MetricClient(self.insights_insert_key)
                self.event_client = EventClient(self.insights_insert_key)
            
            # Create HTTP client for API calls
            self.http_client = httpx.AsyncClient(
                headers={
                    "Api-Key": self.api_key,
                    "Content-Type": "application/json"
                } if self.api_key else {},
                timeout=30.0
            )
            
            # Test connection
            await self._test_connection()
            
            self.logger.info("Successfully connected to New Relic")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to New Relic: {str(e)}"
            self.logger.error(error_msg)
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from New Relic."""
        if self.http_client:
            await self.http_client.aclose()
        
        self.logger.info("Disconnected from New Relic")
    
    @background_task()
    async def send_metrics(
        self,
        metrics: List[Dict[str, Any]],
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send custom metrics to New Relic."""
        if not self.metric_client:
            self.logger.warning("Metric client not available, falling back to agent API")
            return await self._send_metrics_via_agent(metrics, tags)
        
        self.logger.debug("Sending metrics to New Relic", count=len(metrics))
        
        try:
            # Convert metrics to New Relic format
            nr_metrics = []
            
            for metric in metrics:
                # Build metric with tags
                metric_tags = dict(tags) if tags else {}
                if "tags" in metric:
                    metric_tags.update(metric["tags"])
                
                # Add default tags
                metric_tags.update({
                    "service": "pynomaly",
                    "component": "anomaly_detection"
                })
                
                nr_metric = {
                    "name": metric["name"],
                    "type": metric.get("type", "gauge"),
                    "value": metric["value"],
                    "timestamp": metric.get("timestamp", datetime.utcnow().timestamp()),
                    "attributes": metric_tags
                }
                
                nr_metrics.append(nr_metric)
            
            # Send metrics batch
            response = self.metric_client.send_batch(nr_metrics)
            
            if response.ok:
                self.logger.debug("Metrics sent successfully to New Relic")
                return True
            else:
                self.logger.error("Failed to send metrics", errors=response.errors)
                return False
            
        except Exception as e:
            self.logger.error("Failed to send metrics to New Relic", error=str(e))
            return False
    
    async def send_model_performance_metrics(
        self,
        model_id: UUID,
        deployment_id: UUID,
        metrics: Dict[str, float],
        tenant_id: UUID,
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send ML model performance metrics to New Relic."""
        self.logger.debug("Sending model performance metrics", model_id=model_id)
        
        try:
            base_tags = {
                "model_id": str(model_id),
                "deployment_id": str(deployment_id),
                "tenant_id": str(tenant_id),
                "service": "pynomaly",
                "component": "ml_model"
            }
            
            if tags:
                base_tags.update(tags)
            
            # Convert metrics to New Relic format
            nr_metrics = []
            timestamp = datetime.utcnow().timestamp()
            
            for metric_name, value in metrics.items():
                nr_metrics.append({
                    "name": f"pynomaly.model.{metric_name}",
                    "value": value,
                    "timestamp": timestamp,
                    "type": "gauge",
                    "tags": base_tags
                })
            
            return await self.send_metrics(nr_metrics)
            
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
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send anomaly detection metrics to New Relic."""
        self.logger.debug("Sending anomaly detection metrics", data_source=data_source)
        
        try:
            base_tags = {
                "tenant_id": str(tenant_id),
                "data_source": data_source,
                "service": "pynomaly",
                "component": "anomaly_detection"
            }
            
            if tags:
                base_tags.update(tags)
            
            timestamp = datetime.utcnow().timestamp()
            anomaly_rate = (anomaly_count / max(total_records, 1)) * 100
            
            metrics = [
                {
                    "name": "pynomaly.anomaly.count",
                    "value": anomaly_count,
                    "timestamp": timestamp,
                    "type": "count",
                    "tags": base_tags
                },
                {
                    "name": "pynomaly.anomaly.rate",
                    "value": anomaly_rate,
                    "timestamp": timestamp,
                    "type": "gauge",
                    "tags": base_tags
                },
                {
                    "name": "pynomaly.anomaly.score",
                    "value": anomaly_score,
                    "timestamp": timestamp,
                    "type": "gauge",
                    "tags": base_tags
                },
                {
                    "name": "pynomaly.records.processed",
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
        event_type: str,
        attributes: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Send custom event to New Relic."""
        if not self.event_client:
            self.logger.warning("Event client not available, falling back to agent API")
            return await self._send_event_via_agent(event_type, attributes, timestamp)
        
        self.logger.debug("Sending event to New Relic", event_type=event_type)
        
        try:
            # Add default attributes
            event_attributes = dict(attributes)
            event_attributes.update({
                "service": "pynomaly",
                "timestamp": (timestamp or datetime.utcnow()).timestamp()
            })
            
            # Create event
            event = {
                "eventType": event_type,
                "timestamp": event_attributes["timestamp"],
                **event_attributes
            }
            
            # Send event
            response = self.event_client.send_batch([event])
            
            if response.ok:
                self.logger.debug("Event sent successfully to New Relic")
                return True
            else:
                self.logger.error("Failed to send event", errors=response.errors)
                return False
            
        except Exception as e:
            self.logger.error("Failed to send event to New Relic", error=str(e))
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
        """Create model deployment event in New Relic."""
        self.logger.debug("Creating model deployment event", deployment=deployment_name)
        
        try:
            attributes = {
                "model_id": str(model_id),
                "deployment_id": str(deployment_id),
                "deployment_name": deployment_name,
                "status": status,
                "tenant_id": str(tenant_id),
                "component": "model_deployment"
            }
            
            if details:
                attributes.update({f"detail_{k}": v for k, v in details.items()})
            
            return await self.send_event("ModelDeployment", attributes)
            
        except Exception as e:
            self.logger.error("Failed to create deployment event", error=str(e))
            return False
    
    @function_trace(name="record_ml_model_prediction")
    async def record_model_prediction(
        self,
        model_id: UUID,
        deployment_id: UUID,
        prediction_latency_ms: float,
        prediction_confidence: Optional[float] = None,
        input_features: Optional[int] = None,
        success: bool = True
    ) -> None:
        """Record ML model prediction metrics."""
        try:
            # Add custom attributes to current transaction
            transaction = current_transaction()
            if transaction:
                add_custom_attribute("model_id", str(model_id))
                add_custom_attribute("deployment_id", str(deployment_id))
                add_custom_attribute("prediction_success", success)
                add_custom_attribute("prediction_latency_ms", prediction_latency_ms)
                
                if prediction_confidence is not None:
                    add_custom_attribute("prediction_confidence", prediction_confidence)
                
                if input_features is not None:
                    add_custom_attribute("input_features", input_features)
            
            # Record custom metrics
            newrelic.agent.record_custom_metric("Custom/Model/PredictionLatency", prediction_latency_ms)
            newrelic.agent.record_custom_metric("Custom/Model/PredictionCount", 1)
            
            if not success:
                newrelic.agent.record_custom_metric("Custom/Model/PredictionErrors", 1)
            
            if prediction_confidence is not None:
                newrelic.agent.record_custom_metric("Custom/Model/PredictionConfidence", prediction_confidence)
            
        except Exception as e:
            self.logger.error("Failed to record model prediction", error=str(e))
    
    async def create_alert_policy(
        self,
        name: str,
        incident_preference: str = "PER_CONDITION"
    ) -> Optional[str]:
        """Create New Relic alert policy."""
        if not self.api_key:
            self.logger.error("API key required for alert policy creation")
            return None
        
        self.logger.info("Creating New Relic alert policy", name=name)
        
        try:
            policy_data = {
                "policy": {
                    "name": name,
                    "incident_preference": incident_preference
                }
            }
            
            response = await self.http_client.post(
                f"{self.base_url}/v2/alerts_policies.json",
                json=policy_data
            )
            
            response.raise_for_status()
            policy = response.json()["policy"]
            
            self.logger.info("Alert policy created", policy_id=policy["id"])
            return str(policy["id"])
            
        except Exception as e:
            self.logger.error("Failed to create alert policy", error=str(e))
            return None
    
    async def create_anomaly_alert_condition(
        self,
        policy_id: str,
        condition_name: str,
        metric_name: str,
        threshold: float,
        tenant_id: Optional[UUID] = None,
        data_source: Optional[str] = None
    ) -> Optional[str]:
        """Create alert condition for anomaly detection."""
        if not self.api_key:
            self.logger.error("API key required for alert condition creation")
            return None
        
        self.logger.info("Creating anomaly alert condition", name=condition_name)
        
        try:
            # Build NRQL query
            nrql_parts = [f"SELECT average({metric_name}) FROM Metric"]
            
            where_conditions = ["service = 'pynomaly'"]
            if tenant_id:
                where_conditions.append(f"tenant_id = '{tenant_id}'")
            if data_source:
                where_conditions.append(f"data_source = '{data_source}'")
            
            if where_conditions:
                nrql_parts.append(f"WHERE {' AND '.join(where_conditions)}")
            
            nrql_query = " ".join(nrql_parts)
            
            condition_data = {
                "nrql_condition": {
                    "name": condition_name,
                    "enabled": True,
                    "type": "static",
                    "nrql": {
                        "query": nrql_query
                    },
                    "critical": {
                        "threshold": threshold,
                        "threshold_duration": 300,  # 5 minutes
                        "threshold_occurrences": "all",
                        "operator": "above"
                    },
                    "warning": {
                        "threshold": threshold * 0.8,
                        "threshold_duration": 300,
                        "threshold_occurrences": "all",
                        "operator": "above"
                    }
                }
            }
            
            response = await self.http_client.post(
                f"{self.base_url}/v2/alerts_nrql_conditions/policies/{policy_id}.json",
                json=condition_data
            )
            
            response.raise_for_status()
            condition = response.json()["nrql_condition"]
            
            self.logger.info("Alert condition created", condition_id=condition["id"])
            return str(condition["id"])
            
        except Exception as e:
            self.logger.error("Failed to create alert condition", error=str(e))
            return None
    
    async def query_metrics(
        self,
        nrql_query: str,
        timeout_seconds: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Query metrics using NRQL."""
        if not self.api_key:
            self.logger.error("API key required for metrics query")
            return None
        
        self.logger.debug("Querying New Relic metrics", query=nrql_query)
        
        try:
            query_params = {
                "nrql": nrql_query
            }
            
            response = await self.http_client.get(
                f"{self.base_url}/v1/accounts/{self.account_id}/query",
                params=query_params,
                timeout=timeout_seconds
            )
            
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
        """Get model metrics summary from New Relic."""
        self.logger.debug("Getting model metrics summary", model_id=model_id)
        
        try:
            # Build NRQL queries for different metrics
            since_clause = f"SINCE {hours_back} hours ago"
            
            queries = {
                "request_rate": f"SELECT rate(count(*), 1 minute) FROM Transaction WHERE model_id = '{model_id}' {since_clause}",
                "error_rate": f"SELECT percentage(count(*), WHERE error = true) FROM Transaction WHERE model_id = '{model_id}' {since_clause}",
                "avg_latency": f"SELECT average(duration) FROM Transaction WHERE model_id = '{model_id}' {since_clause}",
                "p95_latency": f"SELECT percentile(duration, 95) FROM Transaction WHERE model_id = '{model_id}' {since_clause}"
            }
            
            summary = {}
            for metric_name, query in queries.items():
                result = await self.query_metrics(query)
                
                if result and result.get("results") and len(result["results"]) > 0:
                    # Extract the value from the first result
                    first_result = result["results"][0]
                    if "average" in first_result:
                        summary[metric_name] = first_result["average"]
                    elif "rate" in first_result:
                        summary[metric_name] = first_result["rate"]
                    elif "percentage" in first_result:
                        summary[metric_name] = first_result["percentage"]
                    elif "percentile" in first_result:
                        summary[metric_name] = first_result["percentile"]
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
        """Test New Relic connection."""
        try:
            # Test API connection if API key available
            if self.api_key:
                response = await self.http_client.get(f"{self.base_url}/v2/applications.json")
                response.raise_for_status()
            
            # Test agent connection
            app = newrelic.agent.application()
            if not app:
                raise RuntimeError("New Relic agent not initialized")
            
            self.logger.info("Connection test successful")
            
        except Exception as e:
            raise RuntimeError(f"Connection test failed: {str(e)}")
    
    async def _send_metrics_via_agent(
        self,
        metrics: List[Dict[str, Any]],
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send metrics via New Relic agent API."""
        try:
            for metric in metrics:
                metric_name = f"Custom/{metric['name']}"
                value = metric["value"]
                
                newrelic.agent.record_custom_metric(metric_name, value)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to send metrics via agent", error=str(e))
            return False
    
    async def _send_event_via_agent(
        self,
        event_type: str,
        attributes: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Send event via New Relic agent API."""
        try:
            newrelic.agent.record_custom_event(event_type, attributes)
            return True
            
        except Exception as e:
            self.logger.error("Failed to send event via agent", error=str(e))
            return False
    
    def __del__(self):
        """Cleanup on object destruction."""
        # Close HTTP client if it exists
        if hasattr(self, 'http_client') and self.http_client:
            try:
                asyncio.create_task(self.http_client.aclose())
            except:
                pass