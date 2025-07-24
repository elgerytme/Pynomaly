"""Custom business metrics and SLA monitoring service."""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import asyncio
import threading
import time

from anomaly_detection.infrastructure.observability.intelligent_alerting_service import (
    IntelligentAlertingService, AlertSeverity, MetricPoint
)


class MetricAggregation(Enum):
    """Types of metric aggregations."""
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


class SLAStatus(Enum):
    """SLA status enumeration."""
    HEALTHY = "healthy"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    UNKNOWN = "unknown"


@dataclass
class BusinessMetricDefinition:
    """Definition of a business metric."""
    metric_id: str
    name: str
    description: str
    unit: str
    aggregation: MetricAggregation
    source_metrics: List[str]  # List of underlying technical metrics
    calculation_expression: str  # Expression to calculate from source metrics
    business_impact: str  # Description of business impact
    owner: str
    tags: Dict[str, str]
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "aggregation": self.aggregation.value,
            "source_metrics": self.source_metrics,
            "calculation_expression": self.calculation_expression,
            "business_impact": self.business_impact,
            "owner": self.owner,
            "tags": self.tags,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class SLADefinition:
    """Definition of a Service Level Agreement."""
    sla_id: str
    name: str
    description: str
    service_name: str
    metric_name: str
    target_value: float
    threshold_operator: str  # ">=", "<=", ">", "<", "=="
    measurement_window: str  # e.g., "1h", "1d", "7d"
    compliance_percentage: float  # e.g., 99.9 for 99.9% uptime
    consequences: List[str]  # Business consequences of SLA breach
    escalation_contacts: List[str]
    created_at: datetime = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "sla_id": self.sla_id,
            "name": self.name,
            "description": self.description,
            "service_name": self.service_name,
            "metric_name": self.metric_name,
            "target_value": self.target_value,
            "threshold_operator": self.threshold_operator,
            "measurement_window": self.measurement_window,
            "compliance_percentage": self.compliance_percentage,
            "consequences": self.consequences,
            "escalation_contacts": self.escalation_contacts,
            "created_at": self.created_at.isoformat(),
            "enabled": self.enabled
        }


@dataclass
class SLAStatus:
    """Current status of an SLA."""
    sla_id: str
    status: str  # "healthy", "at_risk", "breached"
    current_value: float
    target_value: float
    compliance_percentage: float
    time_to_breach: Optional[timedelta]
    last_breach: Optional[datetime]
    breach_count_24h: int
    error_budget_remaining: float  # Percentage of error budget remaining
    measurement_window_start: datetime
    measurement_window_end: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "sla_id": self.sla_id,
            "status": self.status,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "compliance_percentage": self.compliance_percentage,
            "time_to_breach": str(self.time_to_breach) if self.time_to_breach else None,
            "last_breach": self.last_breach.isoformat() if self.last_breach else None,
            "breach_count_24h": self.breach_count_24h,
            "error_budget_remaining": self.error_budget_remaining,
            "measurement_window_start": self.measurement_window_start.isoformat(),
            "measurement_window_end": self.measurement_window_end.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class BusinessMetricValue:
    """A calculated business metric value."""
    metric_id: str
    value: float
    timestamp: datetime
    calculation_details: Dict[str, Any]
    source_data: Dict[str, List[float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "metric_id": self.metric_id,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "calculation_details": self.calculation_details,
            "source_data": self.source_data
        }


class BusinessMetricsService:
    """Service for custom business metrics and SLA monitoring."""
    
    def __init__(self, alerting_service: IntelligentAlertingService):
        """Initialize business metrics service.
        
        Args:
            alerting_service: Intelligent alerting service for notifications
        """
        self.alerting_service = alerting_service
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self._business_metrics: Dict[str, BusinessMetricDefinition] = {}
        self._sla_definitions: Dict[str, SLADefinition] = {}
        self._metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._sla_status: Dict[str, SLAStatus] = {}
        self._sla_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Technical metrics buffer (from other systems)
        self._technical_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        
        # Background processing
        self._running = False
        self._calculation_thread: Optional[threading.Thread] = None
        self._calculation_interval = 60  # seconds
        
        # Built-in business metrics
        self._initialize_builtin_metrics()
    
    def _initialize_builtin_metrics(self):
        """Initialize built-in business metrics for anomaly detection."""
        # Revenue impact metric
        revenue_metric = BusinessMetricDefinition(
            metric_id="revenue_impact",
            name="Revenue Impact",
            description="Estimated revenue impact from anomaly detection accuracy",
            unit="USD",
            aggregation=MetricAggregation.SUM,
            source_metrics=["model_accuracy", "prediction_count", "false_positive_rate"],
            calculation_expression="(1 - false_positive_rate) * prediction_count * 10.0",
            business_impact="Direct revenue protection from preventing false alarms",
            owner="business-team",
            tags={"domain": "finance", "priority": "high"}
        )
        self._business_metrics[revenue_metric.metric_id] = revenue_metric
        
        # Customer satisfaction metric
        satisfaction_metric = BusinessMetricDefinition(
            metric_id="customer_satisfaction_score",
            name="Customer Satisfaction Score",
            description="Customer satisfaction based on service reliability",
            unit="score",
            aggregation=MetricAggregation.AVERAGE,
            source_metrics=["error_rate", "response_time_ms", "availability"],
            calculation_expression="100 - (error_rate * 50) - min(response_time_ms / 10, 30) + (availability * 30)",
            business_impact="Customer retention and brand reputation",
            owner="product-team",
            tags={"domain": "customer", "priority": "high"}
        )
        self._business_metrics[satisfaction_metric.metric_id] = satisfaction_metric
        
        # Operational efficiency metric
        efficiency_metric = BusinessMetricDefinition(
            metric_id="operational_efficiency",
            name="Operational Efficiency",
            description="Overall operational efficiency of anomaly detection system",
            unit="percentage",
            aggregation=MetricAggregation.AVERAGE,
            source_metrics=["cpu_usage", "memory_usage", "throughput", "manual_interventions"],
            calculation_expression="max(0, 100 - cpu_usage - memory_usage + (throughput / 100) - manual_interventions)",
            business_impact="Cost reduction and resource optimization",
            owner="ops-team",
            tags={"domain": "operations", "priority": "medium"}
        )
        self._business_metrics[efficiency_metric.metric_id] = efficiency_metric
        
        # Initialize default SLAs
        self._initialize_default_slas()
    
    def _initialize_default_slas(self):
        """Initialize default SLA definitions."""
        # Availability SLA
        availability_sla = SLADefinition(
            sla_id="availability_sla",
            name="Service Availability",
            description="99.9% uptime for anomaly detection service",
            service_name="anomaly_detection",
            metric_name="availability",
            target_value=99.9,
            threshold_operator=">=",
            measurement_window="24h",
            compliance_percentage=99.9,
            consequences=[
                "Customer complaints",
                "Revenue loss",
                "SLA penalties"
            ],
            escalation_contacts=["ops-team@company.com", "management@company.com"]
        )
        self._sla_definitions[availability_sla.sla_id] = availability_sla
        
        # Response time SLA
        response_time_sla = SLADefinition(
            sla_id="response_time_sla",
            name="Response Time",
            description="95% of requests processed within 500ms",
            service_name="anomaly_detection",
            metric_name="response_time_p95",
            target_value=500.0,
            threshold_operator="<=",
            measurement_window="1h",
            compliance_percentage=95.0,
            consequences=[
                "Poor user experience",
                "Timeout errors",
                "System overload"
            ],
            escalation_contacts=["performance-team@company.com"]
        )
        self._sla_definitions[response_time_sla.sla_id] = response_time_sla
        
        # Accuracy SLA
        accuracy_sla = SLADefinition(
            sla_id="accuracy_sla",
            name="Model Accuracy",
            description="Model accuracy must be above 90%",
            service_name="anomaly_detection",
            metric_name="model_accuracy",
            target_value=90.0,
            threshold_operator=">=",
            measurement_window="6h",
            compliance_percentage=95.0,
            consequences=[
                "Incorrect predictions",
                "Business impact",
                "Model retraining required"
            ],
            escalation_contacts=["ml-team@company.com", "data-science@company.com"]
        )
        self._sla_definitions[accuracy_sla.sla_id] = accuracy_sla
    
    def start(self):
        """Start the business metrics service."""
        if self._running:
            return
        
        self._running = True
        self._calculation_thread = threading.Thread(target=self._calculation_loop, daemon=True)
        self._calculation_thread.start()
        
        self.logger.info("Business metrics service started")
    
    def stop(self):
        """Stop the business metrics service."""
        self._running = False
        
        if self._calculation_thread:
            self._calculation_thread.join(timeout=5)
        
        self.logger.info("Business metrics service stopped")
    
    def ingest_technical_metric(self,
                              metric_name: str,
                              value: float,
                              timestamp: Optional[datetime] = None,
                              labels: Optional[Dict[str, str]] = None):
        """Ingest a technical metric for business metric calculations.
        
        Args:
            metric_name: Name of the technical metric
            value: Metric value
            timestamp: Optional timestamp
            labels: Optional metric labels
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_point = {
            "value": value,
            "timestamp": timestamp,
            "labels": labels or {}
        }
        
        self._technical_metrics[metric_name].append(metric_point)
        
        # Also send to alerting service for monitoring
        self.alerting_service.ingest_metric(metric_name, value, labels, timestamp)
    
    def create_business_metric(self, definition: BusinessMetricDefinition) -> str:
        """Create a new business metric definition.
        
        Args:
            definition: Business metric definition
            
        Returns:
            Metric ID
        """
        self._business_metrics[definition.metric_id] = definition
        self.logger.info(f"Created business metric: {definition.name}")
        return definition.metric_id
    
    def create_sla(self, definition: SLADefinition) -> str:
        """Create a new SLA definition.
        
        Args:
            definition: SLA definition
            
        Returns:
            SLA ID
        """
        self._sla_definitions[definition.sla_id] = definition
        self.logger.info(f"Created SLA: {definition.name}")
        return definition.sla_id
    
    def _calculation_loop(self):
        """Background loop for calculating business metrics and monitoring SLAs."""
        while self._running:
            try:
                self._calculate_business_metrics()
                self._monitor_slas()
                time.sleep(self._calculation_interval)
            except Exception as e:
                self.logger.error(f"Error in business metrics calculation loop: {e}")
                time.sleep(self._calculation_interval)
    
    def _calculate_business_metrics(self):
        """Calculate all business metrics."""
        current_time = datetime.now()
        
        for metric_def in self._business_metrics.values():
            if not metric_def.enabled:
                continue
            
            try:
                metric_value = self._calculate_single_metric(metric_def, current_time)
                if metric_value is not None:
                    self._metric_values[metric_def.metric_id].append(metric_value)
                    
                    # Send to alerting service for monitoring
                    self.alerting_service.ingest_metric(
                        f"business.{metric_def.metric_id}",
                        metric_value.value,
                        {"owner": metric_def.owner, **metric_def.tags},
                        current_time
                    )
                    
            except Exception as e:
                self.logger.error(f"Error calculating business metric {metric_def.name}: {e}")
    
    def _calculate_single_metric(self,
                               metric_def: BusinessMetricDefinition,
                               timestamp: datetime) -> Optional[BusinessMetricValue]:
        """Calculate a single business metric.
        
        Args:
            metric_def: Business metric definition
            timestamp: Calculation timestamp
            
        Returns:
            Calculated metric value or None
        """
        # Get source data
        source_data = {}
        calculation_window = timedelta(minutes=10)  # Look back 10 minutes
        start_time = timestamp - calculation_window
        
        for source_metric in metric_def.source_metrics:
            if source_metric not in self._technical_metrics:
                self.logger.warning(f"Source metric {source_metric} not found for {metric_def.name}")
                return None
            
            # Filter data within time window
            recent_points = [
                point for point in self._technical_metrics[source_metric]
                if start_time <= point["timestamp"] <= timestamp
            ]
            
            if not recent_points:
                self.logger.debug(f"No recent data for source metric {source_metric}")
                return None
            
            values = [point["value"] for point in recent_points]
            source_data[source_metric] = values
        
        # Calculate aggregated values for each source metric
        aggregated_values = {}
        for metric_name, values in source_data.items():
            if metric_def.aggregation == MetricAggregation.SUM:
                aggregated_values[metric_name] = sum(values)
            elif metric_def.aggregation == MetricAggregation.AVERAGE:
                aggregated_values[metric_name] = np.mean(values)
            elif metric_def.aggregation == MetricAggregation.COUNT:
                aggregated_values[metric_name] = len(values)
            elif metric_def.aggregation == MetricAggregation.MIN:
                aggregated_values[metric_name] = min(values)
            elif metric_def.aggregation == MetricAggregation.MAX:
                aggregated_values[metric_name] = max(values)
            elif metric_def.aggregation == MetricAggregation.PERCENTILE_50:
                aggregated_values[metric_name] = np.percentile(values, 50)
            elif metric_def.aggregation == MetricAggregation.PERCENTILE_95:
                aggregated_values[metric_name] = np.percentile(values, 95)
            elif metric_def.aggregation == MetricAggregation.PERCENTILE_99:
                aggregated_values[metric_name] = np.percentile(values, 99)
            else:
                aggregated_values[metric_name] = np.mean(values)  # Default to average
        
        # Evaluate calculation expression
        try:
            # Create safe evaluation context
            eval_context = {
                **aggregated_values,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "__builtins__": {}
            }
            
            calculated_value = eval(metric_def.calculation_expression, eval_context)
            
            return BusinessMetricValue(
                metric_id=metric_def.metric_id,
                value=float(calculated_value),
                timestamp=timestamp,
                calculation_details={
                    "expression": metric_def.calculation_expression,
                    "aggregated_values": aggregated_values,
                    "data_points_used": sum(len(values) for values in source_data.values())
                },
                source_data=source_data
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating expression for {metric_def.name}: {e}")
            return None
    
    def _monitor_slas(self):
        """Monitor all SLA definitions."""
        current_time = datetime.now()
        
        for sla_def in self._sla_definitions.values():
            if not sla_def.enabled:
                continue
            
            try:
                sla_status = self._evaluate_sla(sla_def, current_time)
                if sla_status:
                    self._sla_status[sla_def.sla_id] = sla_status
                    
                    # Record SLA status history
                    self._sla_history[sla_def.sla_id].append({
                        "timestamp": current_time.isoformat(),
                        "status": sla_status.status,
                        "value": sla_status.current_value,
                        "compliance": sla_status.compliance_percentage
                    })
                    
                    # Check for SLA breach
                    if sla_status.status == "breached":
                        self._handle_sla_breach(sla_def, sla_status)
                    
            except Exception as e:
                self.logger.error(f"Error monitoring SLA {sla_def.name}: {e}")
    
    def _evaluate_sla(self, sla_def: SLADefinition, current_time: datetime) -> Optional[SLAStatus]:
        """Evaluate an SLA definition.
        
        Args:
            sla_def: SLA definition
            current_time: Current timestamp
            
        Returns:
            SLA status or None
        """
        # Parse measurement window
        window_duration = self._parse_duration(sla_def.measurement_window)
        start_time = current_time - window_duration
        
        # Get metric data
        metric_data = self._get_metric_data_for_sla(sla_def.metric_name, start_time, current_time)
        
        if not metric_data:
            return SLAStatus(
                sla_id=sla_def.sla_id,
                status="unknown",
                current_value=0.0,
                target_value=sla_def.target_value,
                compliance_percentage=0.0,
                time_to_breach=None,
                last_breach=None,
                breach_count_24h=0,
                error_budget_remaining=100.0,
                measurement_window_start=start_time,
                measurement_window_end=current_time,
                last_updated=current_time
            )
        
        # Calculate current metric value
        values = [point["value"] for point in metric_data]
        current_value = np.mean(values)  # Could be customized per SLA
        
        # Evaluate threshold
        is_compliant = self._evaluate_threshold(
            current_value, sla_def.target_value, sla_def.threshold_operator
        )
        
        # Calculate compliance percentage
        compliant_points = sum(
            1 for point in metric_data
            if self._evaluate_threshold(point["value"], sla_def.target_value, sla_def.threshold_operator)
        )
        compliance_percentage = (compliant_points / len(metric_data)) * 100
        
        # Determine status
        if compliance_percentage >= sla_def.compliance_percentage:
            status = "healthy"
        elif compliance_percentage >= sla_def.compliance_percentage * 0.9:  # Within 10%
            status = "at_risk"
        else:
            status = "breached"
        
        # Calculate error budget
        error_budget_remaining = max(0, compliance_percentage - (100 - sla_def.compliance_percentage))
        
        # Get breach history
        last_breach = self._get_last_breach(sla_def.sla_id)
        breach_count_24h = self._get_breach_count_24h(sla_def.sla_id, current_time)
        
        return SLAStatus(
            sla_id=sla_def.sla_id,
            status=status,
            current_value=current_value,
            target_value=sla_def.target_value,
            compliance_percentage=compliance_percentage,
            time_to_breach=self._estimate_time_to_breach(sla_def, metric_data) if status == "at_risk" else None,
            last_breach=last_breach,
            breach_count_24h=breach_count_24h,
            error_budget_remaining=error_budget_remaining,
            measurement_window_start=start_time,
            measurement_window_end=current_time,
            last_updated=current_time
        )
    
    def _get_metric_data_for_sla(self,
                               metric_name: str,
                               start_time: datetime,
                               end_time: datetime) -> List[Dict[str, Any]]:
        """Get metric data for SLA evaluation.
        
        Args:
            metric_name: Name of the metric
            start_time: Start of measurement window
            end_time: End of measurement window
            
        Returns:
            List of metric data points
        """
        # Check technical metrics first
        if metric_name in self._technical_metrics:
            return [
                point for point in self._technical_metrics[metric_name]
                if start_time <= point["timestamp"] <= end_time
            ]
        
        # Check business metrics
        for business_metric_id, values in self._metric_values.items():
            if business_metric_id == metric_name or f"business.{business_metric_id}" == metric_name:
                return [
                    {"value": value.value, "timestamp": value.timestamp}
                    for value in values
                    if start_time <= value.timestamp <= end_time
                ]
        
        return []
    
    def _evaluate_threshold(self, current_value: float, target_value: float, operator: str) -> bool:
        """Evaluate threshold condition.
        
        Args:
            current_value: Current metric value
            target_value: Target value
            operator: Comparison operator
            
        Returns:
            True if threshold is met
        """
        if operator == ">=":
            return current_value >= target_value
        elif operator == "<=":
            return current_value <= target_value
        elif operator == ">":
            return current_value > target_value
        elif operator == "<":
            return current_value < target_value
        elif operator == "==":
            return abs(current_value - target_value) < 0.001
        else:
            return False
    
    def _parse_duration(self, duration_str: str) -> timedelta:
        """Parse duration string to timedelta."""
        if duration_str.endswith('s'):
            return timedelta(seconds=int(duration_str[:-1]))
        elif duration_str.endswith('m'):
            return timedelta(minutes=int(duration_str[:-1]))
        elif duration_str.endswith('h'):
            return timedelta(hours=int(duration_str[:-1]))
        elif duration_str.endswith('d'):
            return timedelta(days=int(duration_str[:-1]))
        else:
            return timedelta(hours=int(duration_str))  # Default to hours
    
    def _get_last_breach(self, sla_id: str) -> Optional[datetime]:
        """Get timestamp of last SLA breach."""
        history = self._sla_history.get(sla_id, [])
        for record in reversed(history):
            if record.get("status") == "breached":
                return datetime.fromisoformat(record["timestamp"])
        return None
    
    def _get_breach_count_24h(self, sla_id: str, current_time: datetime) -> int:
        """Get number of breaches in last 24 hours."""
        cutoff_time = current_time - timedelta(hours=24)
        history = self._sla_history.get(sla_id, [])
        
        return sum(
            1 for record in history
            if (record.get("status") == "breached" and
                datetime.fromisoformat(record["timestamp"]) >= cutoff_time)
        )
    
    def _estimate_time_to_breach(self,
                               sla_def: SLADefinition,
                               recent_data: List[Dict[str, Any]]) -> Optional[timedelta]:
        """Estimate time until SLA breach based on current trend."""
        if len(recent_data) < 2:
            return None
        
        # Simple linear trend analysis
        timestamps = [point["timestamp"] for point in recent_data]
        values = [point["value"] for point in recent_data]
        
        # Calculate trend (slope)
        time_diffs = [(t - timestamps[0]).total_seconds() for t in timestamps]
        if len(time_diffs) > 1:
            trend = np.polyfit(time_diffs, values, 1)[0]  # Linear trend
            
            current_value = values[-1]
            target_value = sla_def.target_value
            
            if trend != 0:
                # Estimate time to reach target
                time_to_target = (target_value - current_value) / trend
                if time_to_target > 0:
                    return timedelta(seconds=time_to_target)
        
        return None
    
    def _handle_sla_breach(self, sla_def: SLADefinition, sla_status: SLAStatus):
        """Handle SLA breach by creating alerts and notifications."""
        # Create high-severity alert
        alert_message = (
            f"SLA '{sla_def.name}' breached: "
            f"{sla_status.current_value:.2f} vs target {sla_def.target_value:.2f} "
            f"(compliance: {sla_status.compliance_percentage:.1f}%)"
        )
        
        # Send alert through alerting service
        self.alerting_service.ingest_metric(
            f"sla.breach.{sla_def.sla_id}",
            1.0,  # Breach indicator
            {
                "sla_name": sla_def.name,
                "service": sla_def.service_name,
                "severity": "critical",
                "compliance": str(sla_status.compliance_percentage)
            }
        )
        
        self.logger.critical(f"SLA BREACH: {alert_message}")
    
    def get_business_metric_values(self,
                                 metric_id: str,
                                 hours_back: int = 24) -> List[BusinessMetricValue]:
        """Get business metric values for a time period.
        
        Args:
            metric_id: Business metric ID
            hours_back: Hours to look back
            
        Returns:
            List of business metric values
        """
        if metric_id not in self._metric_values:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        return [
            value for value in self._metric_values[metric_id]
            if value.timestamp >= cutoff_time
        ]
    
    def get_sla_status(self, sla_id: str) -> Optional[SLAStatus]:
        """Get current SLA status.
        
        Args:
            sla_id: SLA ID
            
        Returns:
            SLA status or None
        """
        return self._sla_status.get(sla_id)
    
    def get_all_sla_statuses(self) -> Dict[str, SLAStatus]:
        """Get all SLA statuses.
        
        Returns:
            Dictionary of SLA statuses
        """
        return dict(self._sla_status)
    
    def generate_business_dashboard(self) -> Dict[str, Any]:
        """Generate business metrics dashboard data.
        
        Returns:
            Dashboard data with business metrics and SLA statuses
        """
        current_time = datetime.now()
        
        # Business metrics summary
        business_metrics_summary = {}
        for metric_id, metric_def in self._business_metrics.items():
            recent_values = self.get_business_metric_values(metric_id, hours_back=1)
            if recent_values:
                latest_value = recent_values[-1]
                business_metrics_summary[metric_id] = {
                    "name": metric_def.name,
                    "current_value": latest_value.value,
                    "unit": metric_def.unit,
                    "business_impact": metric_def.business_impact,
                    "owner": metric_def.owner,
                    "trend": self._calculate_trend(recent_values),
                    "last_updated": latest_value.timestamp.isoformat()
                }
        
        # SLA summary
        sla_summary = {}
        healthy_slas = 0
        at_risk_slas = 0
        breached_slas = 0
        
        for sla_id, sla_status in self._sla_status.items():
            sla_def = self._sla_definitions[sla_id]
            sla_summary[sla_id] = {
                "name": sla_def.name,
                "status": sla_status.status,
                "compliance_percentage": sla_status.compliance_percentage,
                "error_budget_remaining": sla_status.error_budget_remaining,
                "breach_count_24h": sla_status.breach_count_24h,
                "last_updated": sla_status.last_updated.isoformat()
            }
            
            if sla_status.status == "healthy":
                healthy_slas += 1
            elif sla_status.status == "at_risk":
                at_risk_slas += 1
            elif sla_status.status == "breached":
                breached_slas += 1
        
        return {
            "generated_at": current_time.isoformat(),
            "business_metrics": business_metrics_summary,
            "sla_summary": sla_summary,
            "sla_overview": {
                "total_slas": len(self._sla_definitions),
                "healthy": healthy_slas,
                "at_risk": at_risk_slas,
                "breached": breached_slas
            },
            "system_health": {
                "overall_status": "healthy" if breached_slas == 0 else "degraded",
                "metrics_tracked": len(self._business_metrics),
                "technical_metrics_ingested": len(self._technical_metrics)
            }
        }
    
    def _calculate_trend(self, values: List[BusinessMetricValue]) -> str:
        """Calculate trend direction for metric values.
        
        Args:
            values: List of metric values
            
        Returns:
            Trend direction: "up", "down", or "stable"
        """
        if len(values) < 2:
            return "stable"
        
        recent_values = [v.value for v in values[-10:]]  # Last 10 values
        
        if len(recent_values) >= 2:
            trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            if trend_slope > 0.1:
                return "up"
            elif trend_slope < -0.1:
                return "down"
        
        return "stable"
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for business metrics service.
        
        Returns:
            Health status information
        """
        return {
            "status": "healthy" if self._running else "stopped",
            "business_metrics_defined": len(self._business_metrics),
            "slas_defined": len(self._sla_definitions),
            "slas_monitored": len(self._sla_status),
            "technical_metrics_tracked": len(self._technical_metrics),
            "calculation_interval": self._calculation_interval
        }


# Global service instance
_business_metrics_service: Optional[BusinessMetricsService] = None


def initialize_business_metrics_service(alerting_service: IntelligentAlertingService) -> BusinessMetricsService:
    """Initialize global business metrics service.
    
    Args:
        alerting_service: Intelligent alerting service
        
    Returns:
        Initialized business metrics service
    """
    global _business_metrics_service
    _business_metrics_service = BusinessMetricsService(alerting_service)
    return _business_metrics_service


def get_business_metrics_service() -> Optional[BusinessMetricsService]:
    """Get global business metrics service instance.
    
    Returns:
        Business metrics service instance or None
    """
    return _business_metrics_service