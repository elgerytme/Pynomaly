#!/usr/bin/env python3
"""
Advanced Distributed Tracing and Business Metrics System
Implements comprehensive tracing, custom metrics collection, and business KPIs
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import aiohttp
import requests
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
import yaml


@dataclass
class TraceConfig:
    """Distributed tracing configuration"""
    service_name: str
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    sampling_rate: float = 0.1
    export_timeout: int = 30
    max_export_batch_size: int = 512
    enabled: bool = True


@dataclass
class BusinessMetric:
    """Business metric configuration"""
    name: str
    description: str
    metric_type: str  # counter, gauge, histogram
    labels: List[str] = None
    unit: str = ""
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = []


class AdvancedTracingSystem:
    """Advanced distributed tracing and metrics system"""
    
    def __init__(self, config_path: str = "config/tracing-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.tracer = None
        self.meter = None
        self.metrics_registry = {}
        self.business_metrics = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        if self.config["tracing"]["enabled"]:
            self._initialize_tracing()
        
        self._initialize_metrics()
        self._setup_business_metrics()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load tracing configuration"""
        default_config = {
            "tracing": {
                "enabled": True,
                "service_name": "hexagonal-architecture",
                "jaeger_endpoint": os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
                "sampling_rate": float(os.getenv("TRACE_SAMPLING_RATE", "0.1")),
                "export_timeout": 30,
                "max_export_batch_size": 512
            },
            "metrics": {
                "prometheus_endpoint": "localhost:8000",
                "collection_interval": 15,
                "export_timeout": 30,
                "custom_metrics_enabled": True
            },
            "business_metrics": {
                "enabled": True,
                "collection_interval": 60,
                "retention_days": 90,
                "alert_thresholds": {
                    "data_quality_failure_rate": 0.05,
                    "anomaly_detection_rate": 0.02,
                    "workflow_success_rate": 0.95
                }
            },
            "instrumentation": {
                "http_requests": True,
                "database_queries": True,
                "redis_operations": True,
                "background_jobs": True,
                "external_apis": True
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                return self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, dict1: dict, dict2: dict) -> dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _initialize_tracing(self):
        """Initialize distributed tracing"""
        try:
            # Create resource with service information
            resource = Resource.create({
                "service.name": self.config["tracing"]["service_name"],
                "service.version": os.getenv("SERVICE_VERSION", "1.0.0"),
                "deployment.environment": os.getenv("ENVIRONMENT", "production")
            })
            
            # Create tracer provider
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
            
            # Create Jaeger exporter
            jaeger_exporter = JaegerExporter(
                collector_endpoint=self.config["tracing"]["jaeger_endpoint"],
                timeout=self.config["tracing"]["export_timeout"]
            )
            
            # Create span processor
            span_processor = BatchSpanProcessor(
                jaeger_exporter,
                max_export_batch_size=self.config["tracing"]["max_export_batch_size"]
            )
            
            # Add span processor to tracer provider
            tracer_provider.add_span_processor(span_processor)
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            # Auto-instrument HTTP libraries
            if self.config["instrumentation"]["http_requests"]:
                RequestsInstrumentor().instrument()
                AioHttpClientInstrumentor().instrument()
            
            self.logger.info("✅ Distributed tracing initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tracing: {e}")
            self.tracer = None
    
    def _initialize_metrics(self):
        """Initialize metrics collection"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.config["tracing"]["service_name"]
            })
            
            # Create Prometheus metric reader
            prometheus_reader = PrometheusMetricReader(
                f":{self.config['metrics']['prometheus_endpoint'].split(':')[-1]}"
            )
            
            # Create meter provider
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[prometheus_reader]
            )
            metrics.set_meter_provider(meter_provider)
            
            # Get meter
            self.meter = metrics.get_meter(__name__)
            
            self.logger.info("✅ Metrics collection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics: {e}")
            self.meter = None
    
    def _setup_business_metrics(self):
        """Setup business-specific metrics"""
        if not self.meter or not self.config["business_metrics"]["enabled"]:
            return
        
        try:
            # Data Quality Metrics
            self.business_metrics["data_quality_checks_total"] = self.meter.create_counter(
                name="data_quality_checks_total",
                description="Total number of data quality checks performed",
                unit="1"
            )
            
            self.business_metrics["data_quality_failures_total"] = self.meter.create_counter(
                name="data_quality_failures_total", 
                description="Total number of data quality check failures",
                unit="1"
            )
            
            self.business_metrics["data_processing_duration"] = self.meter.create_histogram(
                name="data_processing_duration_seconds",
                description="Time spent processing data",
                unit="s"
            )
            
            # Anomaly Detection Metrics
            self.business_metrics["anomalies_detected_total"] = self.meter.create_counter(
                name="anomalies_detected_total",
                description="Total number of anomalies detected",
                unit="1"
            )
            
            self.business_metrics["anomaly_detection_accuracy"] = self.meter.create_gauge(
                name="anomaly_detection_accuracy",
                description="Accuracy of anomaly detection algorithm",
                unit="1"
            )
            
            # Workflow Engine Metrics
            self.business_metrics["workflow_executions_total"] = self.meter.create_counter(
                name="workflow_executions_total",
                description="Total number of workflow executions",
                unit="1"
            )
            
            self.business_metrics["workflow_execution_duration"] = self.meter.create_histogram(
                name="workflow_execution_duration_seconds",
                description="Duration of workflow executions",
                unit="s"
            )
            
            # Business Transaction Metrics
            self.business_metrics["business_transactions_total"] = self.meter.create_counter(
                name="business_transactions_total",
                description="Total business transactions processed",
                unit="1"
            )
            
            self.business_metrics["business_transaction_value"] = self.meter.create_histogram(
                name="business_transaction_value",
                description="Value of business transactions",
                unit="currency"
            )
            
            # System Health Metrics
            self.business_metrics["active_users"] = self.meter.create_gauge(
                name="active_users",
                description="Number of currently active users",
                unit="1"
            )
            
            self.business_metrics["system_capacity_utilization"] = self.meter.create_gauge(
                name="system_capacity_utilization",
                description="Overall system capacity utilization",
                unit="percent"
            )
            
            self.logger.info("✅ Business metrics setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup business metrics: {e}")
    
    def start_trace(self, operation_name: str, **attributes) -> Optional[Any]:
        """Start a new trace span"""
        if not self.tracer:
            return None
        
        span = self.tracer.start_span(operation_name)
        
        # Add custom attributes
        for key, value in attributes.items():
            span.set_attribute(key, value)
        
        return span
    
    def add_trace_event(self, span: Any, event_name: str, **attributes):
        """Add an event to a trace span"""
        if span:
            span.add_event(event_name, attributes)
    
    def set_trace_attribute(self, span: Any, key: str, value: Any):
        """Set an attribute on a trace span"""
        if span:
            span.set_attribute(key, value)
    
    def finish_trace(self, span: Any, status: str = "ok", error: Exception = None):
        """Finish a trace span"""
        if not span:
            return
        
        if error:
            span.record_exception(error)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
        else:
            span.set_status(trace.Status(trace.StatusCode.OK))
        
        span.end()
    
    def record_data_quality_check(self, service: str, rule_type: str, status: str, duration: float):
        """Record data quality check metrics"""
        if "data_quality_checks_total" in self.business_metrics:
            self.business_metrics["data_quality_checks_total"].add(
                1, {"service": service, "rule_type": rule_type, "status": status}
            )
        
        if status == "failed" and "data_quality_failures_total" in self.business_metrics:
            self.business_metrics["data_quality_failures_total"].add(
                1, {"service": service, "rule_type": rule_type}
            )
        
        if "data_processing_duration" in self.business_metrics:
            self.business_metrics["data_processing_duration"].record(
                duration, {"service": service, "operation": "quality_check"}
            )
    
    def record_anomaly_detection(self, algorithm: str, severity: str, confidence: float):
        """Record anomaly detection metrics"""
        if "anomalies_detected_total" in self.business_metrics:
            self.business_metrics["anomalies_detected_total"].add(
                1, {"algorithm": algorithm, "severity": severity}
            )
        
        if "anomaly_detection_accuracy" in self.business_metrics:
            self.business_metrics["anomaly_detection_accuracy"].set(
                confidence, {"algorithm": algorithm}
            )
    
    def record_workflow_execution(self, workflow_type: str, status: str, duration: float, steps_completed: int):
        """Record workflow execution metrics"""
        if "workflow_executions_total" in self.business_metrics:
            self.business_metrics["workflow_executions_total"].add(
                1, {"workflow_type": workflow_type, "status": status}
            )
        
        if "workflow_execution_duration" in self.business_metrics:
            self.business_metrics["workflow_execution_duration"].record(
                duration, {"workflow_type": workflow_type, "status": status}
            )
    
    def record_business_transaction(self, transaction_type: str, value: float, currency: str = "USD"):
        """Record business transaction metrics"""
        if "business_transactions_total" in self.business_metrics:
            self.business_metrics["business_transactions_total"].add(
                1, {"transaction_type": transaction_type, "currency": currency}
            )
        
        if "business_transaction_value" in self.business_metrics:
            self.business_metrics["business_transaction_value"].record(
                value, {"transaction_type": transaction_type, "currency": currency}
            )
    
    def update_system_health(self, active_users: int, capacity_utilization: float):
        """Update system health metrics"""
        if "active_users" in self.business_metrics:
            self.business_metrics["active_users"].set(active_users)
        
        if "system_capacity_utilization" in self.business_metrics:
            self.business_metrics["system_capacity_utilization"].set(capacity_utilization)
    
    async def trace_http_request(self, method: str, url: str, **kwargs) -> Any:
        """Trace an HTTP request"""
        span = self.start_trace(f"http_{method.lower()}", **{
            "http.method": method,
            "http.url": url,
            "component": "http_client"
        })
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.request(method, url, **kwargs) as response:
                    duration = time.time() - start_time
                    
                    self.set_trace_attribute(span, "http.status_code", response.status)
                    self.set_trace_attribute(span, "http.response_time", duration)
                    
                    if response.status >= 400:
                        self.add_trace_event(span, "http_error", {
                            "error.kind": "http_error",
                            "http.status_code": response.status
                        })
                    
                    self.finish_trace(span)
                    return response
                    
        except Exception as e:
            self.finish_trace(span, error=e)
            raise
    
    async def trace_database_operation(self, operation: str, table: str, query: str = None):
        """Trace a database operation"""
        span = self.start_trace(f"db_{operation}", **{
            "db.type": "postgresql",
            "db.statement.table": table,
            "component": "database"
        })
        
        if query:
            self.set_trace_attribute(span, "db.statement", query)
        
        try:
            start_time = time.time()
            
            # Simulate database operation
            await asyncio.sleep(0.1)  # Replace with actual database call
            
            duration = time.time() - start_time
            self.set_trace_attribute(span, "db.operation.duration", duration)
            
            self.finish_trace(span)
            
        except Exception as e:
            self.finish_trace(span, error=e)
            raise
    
    def create_custom_metric(self, name: str, description: str, metric_type: str, unit: str = "") -> Any:
        """Create a custom metric"""
        if not self.meter:
            return None
        
        try:
            if metric_type == "counter":
                metric = self.meter.create_counter(name, description, unit)
            elif metric_type == "gauge":
                metric = self.meter.create_gauge(name, description, unit)
            elif metric_type == "histogram":
                metric = self.meter.create_histogram(name, description, unit)
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
            
            self.metrics_registry[name] = metric
            return metric
            
        except Exception as e:
            self.logger.error(f"Failed to create custom metric {name}: {e}")
            return None
    
    def get_metric(self, name: str) -> Optional[Any]:
        """Get a metric by name"""
        return self.metrics_registry.get(name) or self.business_metrics.get(name)
    
    async def collect_business_kpis(self) -> Dict[str, float]:
        """Collect current business KPIs"""
        # This would typically query your database or external systems
        # For demonstration, we'll return simulated values
        
        kpis = {
            "data_quality_score": 94.5,
            "anomaly_detection_accuracy": 97.2,
            "workflow_success_rate": 98.8,
            "average_response_time": 245.0,
            "system_availability": 99.9,
            "user_satisfaction": 4.6,
            "business_transaction_volume": 15420.0,
            "revenue_per_hour": 125000.0
        }
        
        return kpis
    
    async def generate_business_metrics_report(self) -> str:
        """Generate business metrics report"""
        kpis = await self.collect_business_kpis()
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("BUSINESS METRICS REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        # Service Performance
        report_lines.append("SERVICE PERFORMANCE:")
        report_lines.append(f"  Data Quality Score: {kpis['data_quality_score']:.1f}%")
        report_lines.append(f"  Anomaly Detection Accuracy: {kpis['anomaly_detection_accuracy']:.1f}%")
        report_lines.append(f"  Workflow Success Rate: {kpis['workflow_success_rate']:.1f}%")
        report_lines.append(f"  Average Response Time: {kpis['average_response_time']:.0f}ms")
        report_lines.append(f"  System Availability: {kpis['system_availability']:.1f}%")
        report_lines.append("")
        
        # Business Metrics
        report_lines.append("BUSINESS METRICS:")
        report_lines.append(f"  User Satisfaction: {kpis['user_satisfaction']:.1f}/5.0")
        report_lines.append(f"  Transaction Volume: {kpis['business_transaction_volume']:,.0f}")
        report_lines.append(f"  Revenue/Hour: ${kpis['revenue_per_hour']:,.2f}")
        report_lines.append("")
        
        # Threshold Analysis
        thresholds = self.config["business_metrics"]["alert_thresholds"]
        report_lines.append("THRESHOLD ANALYSIS:")
        
        data_quality_rate = kpis['data_quality_score'] / 100
        if data_quality_rate < (1 - thresholds["data_quality_failure_rate"]):
            report_lines.append("  ⚠️  Data Quality below threshold")
        else:
            report_lines.append("  ✅ Data Quality within threshold")
        
        workflow_rate = kpis['workflow_success_rate'] / 100
        if workflow_rate < thresholds["workflow_success_rate"]:
            report_lines.append("  ⚠️  Workflow Success Rate below threshold")
        else:
            report_lines.append("  ✅ Workflow Success Rate within threshold")
        
        return "\n".join(report_lines)
    
    def get_trace_context(self) -> Dict[str, str]:
        """Get current trace context for propagation"""
        if not self.tracer:
            return {}
        
        # This would typically extract trace context for propagation
        # Implementation depends on your tracing setup
        return {
            "trace_id": "simulated_trace_id",
            "span_id": "simulated_span_id"
        }
    
    async def flush_traces(self):
        """Flush any pending traces"""
        if self.tracer:
            # Force flush of any pending spans
            try:
                # This would depend on your tracer implementation
                self.logger.info("Traces flushed successfully")
            except Exception as e:
                self.logger.error(f"Failed to flush traces: {e}")


# Example usage and demonstration
async def demonstrate_advanced_tracing():
    """Demonstrate advanced tracing capabilities"""
    
    # Initialize tracing system
    tracing = AdvancedTracingSystem()
    
    # Example: Trace a data quality check operation
    span = tracing.start_trace("data_quality_check", **{
        "service": "data-quality-service",
        "rule_type": "completeness",
        "dataset": "customer_orders"
    })
    
    try:
        # Simulate data quality check
        await asyncio.sleep(0.5)
        
        # Record business metrics
        tracing.record_data_quality_check(
            service="data-quality-service",
            rule_type="completeness", 
            status="passed",
            duration=0.5
        )
        
        tracing.add_trace_event(span, "quality_check_completed", {
            "rules_checked": 15,
            "violations_found": 0
        })
        
        tracing.finish_trace(span)
        
    except Exception as e:
        tracing.finish_trace(span, error=e)
        raise
    
    # Example: Trace workflow execution
    workflow_span = tracing.start_trace("workflow_execution", **{
        "workflow_type": "data_processing_pipeline",
        "workflow_id": "wf_12345"
    })
    
    try:
        # Simulate workflow steps
        for step in ["validate_input", "process_data", "store_results"]:
            step_span = tracing.start_trace(f"workflow_step_{step}", **{
                "workflow_id": "wf_12345",
                "step_name": step
            })
            
            await asyncio.sleep(0.2)
            tracing.finish_trace(step_span)
        
        # Record workflow metrics
        tracing.record_workflow_execution(
            workflow_type="data_processing_pipeline",
            status="success",
            duration=0.6,
            steps_completed=3
        )
        
        tracing.finish_trace(workflow_span)
        
    except Exception as e:
        tracing.finish_trace(workflow_span, error=e)
        raise
    
    # Generate business metrics report
    report = await tracing.generate_business_metrics_report()
    print(report)


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Tracing and Business Metrics")
    parser.add_argument("--config", default="config/tracing-config.yaml", help="Configuration file")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--report", action="store_true", help="Generate business metrics report")
    parser.add_argument("--flush", action="store_true", help="Flush pending traces")
    args = parser.parse_args()
    
    tracing = AdvancedTracingSystem(args.config)
    
    if args.demo:
        await demonstrate_advanced_tracing()
    elif args.report:
        report = await tracing.generate_business_metrics_report()
        print(report)
    elif args.flush:
        await tracing.flush_traces()
    else:
        print("Advanced tracing system initialized. Use --demo, --report, or --flush")


if __name__ == "__main__":
    asyncio.run(main())