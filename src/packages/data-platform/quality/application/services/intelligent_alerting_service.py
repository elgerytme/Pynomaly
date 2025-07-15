"""Intelligent Alerting Service.

Provides intelligent alerting capabilities with multi-level alerts, escalation workflows,
correlation and deduplication, and context-rich root cause analysis.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from abc import ABC, abstractmethod
import re
import statistics

from ...domain.entities.quality_monitoring import (
    QualityAlert, AlertSeverity, AlertStatus, EscalationLevel, 
    AlertId, ThresholdId, MonitoringJobId, StreamId, QualityThreshold
)

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    DASHBOARD = "dashboard"


class CorrelationStrategy(Enum):
    """Alert correlation strategies."""
    TEMPORAL = "temporal"
    THRESHOLD_BASED = "threshold_based"
    PATTERN_BASED = "pattern_based"
    CAUSAL = "causal"
    STATISTICAL = "statistical"


@dataclass(frozen=True)
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    conditions: List[str]
    actions: List[str]
    severity: AlertSeverity
    enabled: bool = True
    priority: int = 1
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule conditions."""
        # Simplified rule evaluation
        for condition in self.conditions:
            if not self._evaluate_condition(condition, context):
                return False
        return True
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        # This is a simplified implementation
        # In practice, you'd have a more sophisticated rule engine
        try:
            # Basic condition evaluation
            if ">" in condition:
                parts = condition.split(">")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = float(parts[1].strip())
                    return context.get(key, 0) > value
            elif "<" in condition:
                parts = condition.split("<")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = float(parts[1].strip())
                    return context.get(key, 0) < value
            elif "==" in condition:
                parts = condition.split("==")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    return str(context.get(key, "")) == value
            
            return False
        except Exception:
            return False


@dataclass(frozen=True)
class EscalationRule:
    """Escalation rule configuration."""
    rule_id: str
    name: str
    from_level: EscalationLevel
    to_level: EscalationLevel
    conditions: List[str]
    delay_minutes: int = 30
    enabled: bool = True
    
    def should_escalate(self, alert: QualityAlert, current_time: datetime) -> bool:
        """Check if alert should be escalated."""
        if not self.enabled:
            return False
        
        # Check if alert is at the correct level
        if alert.escalation_level != self.from_level:
            return False
        
        # Check delay
        if alert.escalated_at:
            time_since_escalation = current_time - alert.escalated_at
        else:
            time_since_escalation = current_time - alert.triggered_at
        
        if time_since_escalation < timedelta(minutes=self.delay_minutes):
            return False
        
        # Check conditions
        context = {
            'alert_age_minutes': (current_time - alert.triggered_at).total_seconds() / 60,
            'severity': alert.severity.value,
            'status': alert.status.value,
            'is_acknowledged': alert.status == AlertStatus.ACKNOWLEDGED,
            'affected_records': alert.affected_records
        }
        
        for condition in self.conditions:
            if not self._evaluate_condition(condition, context):
                return False
        
        return True
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate escalation condition."""
        # Similar to AlertRule evaluation
        try:
            if ">" in condition:
                parts = condition.split(">")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = float(parts[1].strip())
                    return context.get(key, 0) > value
            elif "==" in condition:
                parts = condition.split("==")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    return str(context.get(key, "")) == value
            
            return False
        except Exception:
            return False


@dataclass(frozen=True)
class AlertChannelConfig:
    """Alert channel configuration."""
    channel: AlertChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    severity_filter: List[AlertSeverity] = field(default_factory=list)
    escalation_filter: List[EscalationLevel] = field(default_factory=list)
    rate_limit_per_hour: int = 100
    
    def should_send(self, alert: QualityAlert) -> bool:
        """Check if alert should be sent to this channel."""
        if not self.enabled:
            return False
        
        # Check severity filter
        if self.severity_filter and alert.severity not in self.severity_filter:
            return False
        
        # Check escalation filter
        if self.escalation_filter and alert.escalation_level not in self.escalation_filter:
            return False
        
        return True


class AlertDeliveryService(ABC):
    """Abstract base class for alert delivery services."""
    
    @abstractmethod
    async def send_alert(self, alert: QualityAlert, config: AlertChannelConfig) -> bool:
        """Send an alert through this delivery service."""
        pass
    
    @abstractmethod
    async def send_batch_alerts(self, alerts: List[QualityAlert], config: AlertChannelConfig) -> bool:
        """Send multiple alerts in a batch."""
        pass


class EmailDeliveryService(AlertDeliveryService):
    """Email alert delivery service."""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        """Initialize email delivery service."""
        self.smtp_config = smtp_config
    
    async def send_alert(self, alert: QualityAlert, config: AlertChannelConfig) -> bool:
        """Send alert via email."""
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config.get('from_email', 'alerts@company.com')
            msg['To'] = config.config.get('to_email', 'admin@company.com')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create HTML body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            if self.smtp_config.get('use_tls', True):
                server.starttls()
            
            if self.smtp_config.get('username'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    async def send_batch_alerts(self, alerts: List[QualityAlert], config: AlertChannelConfig) -> bool:
        """Send multiple alerts in a batch email."""
        try:
            # Create batch email message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config.get('from_email', 'alerts@company.com')
            msg['To'] = config.config.get('to_email', 'admin@company.com')
            msg['Subject'] = f"Quality Alert Summary - {len(alerts)} alerts"
            
            # Create HTML body
            body = self._create_batch_email_body(alerts)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            if self.smtp_config.get('use_tls', True):
                server.starttls()
            
            if self.smtp_config.get('username'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send batch email alerts: {str(e)}")
            return False
    
    def _create_email_body(self, alert: QualityAlert) -> str:
        """Create HTML email body for alert."""
        severity_color = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.EMERGENCY: "#dc3545"
        }
        
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .alert-header {{ background-color: {severity_color.get(alert.severity, '#6c757d')}; 
                                color: white; padding: 15px; border-radius: 5px; }}
                .alert-body {{ padding: 20px; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }}
                .alert-field {{ margin-bottom: 10px; }}
                .alert-label {{ font-weight: bold; }}
                .alert-value {{ color: #333; }}
                .context-data {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>{alert.title}</h2>
                <p>Severity: {alert.severity.value.upper()}</p>
            </div>
            <div class="alert-body">
                <div class="alert-field">
                    <span class="alert-label">Description:</span>
                    <span class="alert-value">{alert.description}</span>
                </div>
                <div class="alert-field">
                    <span class="alert-label">Stream ID:</span>
                    <span class="alert-value">{alert.stream_id}</span>
                </div>
                <div class="alert-field">
                    <span class="alert-label">Metric:</span>
                    <span class="alert-value">{alert.metric_name}</span>
                </div>
                <div class="alert-field">
                    <span class="alert-label">Actual Value:</span>
                    <span class="alert-value">{alert.actual_value}</span>
                </div>
                <div class="alert-field">
                    <span class="alert-label">Threshold:</span>
                    <span class="alert-value">{alert.threshold_value}</span>
                </div>
                <div class="alert-field">
                    <span class="alert-label">Affected Records:</span>
                    <span class="alert-value">{alert.affected_records}</span>
                </div>
                <div class="alert-field">
                    <span class="alert-label">Triggered At:</span>
                    <span class="alert-value">{alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}</span>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_batch_email_body(self, alerts: List[QualityAlert]) -> str:
        """Create HTML email body for batch alerts."""
        alert_rows = ""
        for alert in alerts:
            alert_rows += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.severity.value.upper()}</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.title}</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.stream_id}</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.triggered_at.strftime('%H:%M:%S')}</td>
            </tr>
            """
        
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .summary-header {{ background-color: #007bff; color: white; padding: 15px; border-radius: 5px; }}
                .alert-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .alert-table th {{ background-color: #f8f9fa; padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6; }}
            </style>
        </head>
        <body>
            <div class="summary-header">
                <h2>Quality Alert Summary</h2>
                <p>Total Alerts: {len(alerts)}</p>
            </div>
            <table class="alert-table">
                <thead>
                    <tr>
                        <th>Severity</th>
                        <th>Title</th>
                        <th>Stream ID</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    {alert_rows}
                </tbody>
            </table>
        </body>
        </html>
        """


class WebhookDeliveryService(AlertDeliveryService):
    """Webhook alert delivery service."""
    
    def __init__(self, http_client=None):
        """Initialize webhook delivery service."""
        self.http_client = http_client
    
    async def send_alert(self, alert: QualityAlert, config: AlertChannelConfig) -> bool:
        """Send alert via webhook."""
        try:
            import aiohttp
            
            webhook_url = config.config.get('webhook_url')
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False
            
            # Create payload
            payload = {
                'alert_id': str(alert.alert_id),
                'severity': alert.severity.value,
                'title': alert.title,
                'description': alert.description,
                'stream_id': str(alert.stream_id),
                'metric_name': alert.metric_name,
                'actual_value': alert.actual_value,
                'threshold_value': alert.threshold_value,
                'affected_records': alert.affected_records,
                'triggered_at': alert.triggered_at.isoformat(),
                'context_data': alert.context_data
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.error(f"Webhook returned status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")
            return False
    
    async def send_batch_alerts(self, alerts: List[QualityAlert], config: AlertChannelConfig) -> bool:
        """Send multiple alerts via webhook."""
        try:
            import aiohttp
            
            webhook_url = config.config.get('webhook_url')
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False
            
            # Create batch payload
            payload = {
                'alert_count': len(alerts),
                'alerts': [
                    {
                        'alert_id': str(alert.alert_id),
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'stream_id': str(alert.stream_id),
                        'triggered_at': alert.triggered_at.isoformat()
                    }
                    for alert in alerts
                ]
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.error(f"Webhook returned status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send batch webhook alerts: {str(e)}")
            return False


@dataclass
class AlertCorrelationResult:
    """Result of alert correlation analysis."""
    correlated_alerts: List[QualityAlert]
    correlation_score: float
    correlation_type: CorrelationStrategy
    root_cause_analysis: Dict[str, Any]
    recommended_actions: List[str]


@dataclass
class AlertingServiceConfig:
    """Configuration for intelligent alerting service."""
    
    # Alert processing
    enable_correlation: bool = True
    enable_deduplication: bool = True
    enable_escalation: bool = True
    enable_root_cause_analysis: bool = True
    
    # Correlation settings
    correlation_window_minutes: int = 30
    correlation_threshold: float = 0.7
    max_correlated_alerts: int = 20
    
    # Deduplication settings
    deduplication_window_minutes: int = 15
    deduplication_similarity_threshold: float = 0.8
    
    # Escalation settings
    escalation_check_interval_minutes: int = 5
    max_escalation_level: EscalationLevel = EscalationLevel.EXECUTIVE
    
    # Rate limiting
    rate_limit_per_hour: int = 1000
    burst_threshold: int = 50
    burst_window_minutes: int = 5
    
    # Delivery
    max_delivery_attempts: int = 3
    delivery_retry_delay_seconds: int = 60
    batch_delivery_interval_minutes: int = 10
    max_batch_size: int = 50


class IntelligentAlertingService:
    """Intelligent alerting service with correlation, escalation, and delivery."""
    
    def __init__(self, config: AlertingServiceConfig = None):
        """Initialize intelligent alerting service."""
        self.config = config or AlertingServiceConfig()
        
        # State management
        self.active_alerts: Dict[AlertId, QualityAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.correlation_cache: Dict[str, AlertCorrelationResult] = {}
        
        # Rules and configuration
        self.alert_rules: List[AlertRule] = []
        self.escalation_rules: List[EscalationRule] = []
        self.channel_configs: Dict[AlertChannel, AlertChannelConfig] = {}
        
        # Delivery services
        self.delivery_services: Dict[AlertChannel, AlertDeliveryService] = {}
        
        # Rate limiting
        self.rate_limiter: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Metrics
        self.metrics = {
            'alerts_processed': 0,
            'alerts_correlated': 0,
            'alerts_escalated': 0,
            'alerts_delivered': 0,
            'delivery_failures': 0,
            'processing_errors': 0
        }
        
        logger.info("Intelligent alerting service initialized")
    
    async def start(self) -> None:
        """Start the alerting service."""
        logger.info("Starting intelligent alerting service")
        
        # Start background tasks
        if self.config.enable_escalation:
            task = asyncio.create_task(self._escalation_monitor())
            self.background_tasks.append(task)
        
        # Start batch delivery task
        task = asyncio.create_task(self._batch_delivery_monitor())
        self.background_tasks.append(task)
        
        # Start cleanup task
        task = asyncio.create_task(self._cleanup_monitor())
        self.background_tasks.append(task)
        
        logger.info("Intelligent alerting service started")
    
    async def stop(self) -> None:
        """Stop the alerting service."""
        logger.info("Stopping intelligent alerting service")
        
        self.shutdown_event.set()
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Intelligent alerting service stopped")
    
    async def process_alert(self, alert: QualityAlert) -> bool:
        """Process a new alert through the intelligent alerting pipeline."""
        try:
            self.metrics['alerts_processed'] += 1
            
            # Check rate limiting
            if self._is_rate_limited(alert):
                logger.warning(f"Alert rate limited: {alert.alert_id}")
                return False
            
            # Deduplication
            if self.config.enable_deduplication:
                if self._is_duplicate_alert(alert):
                    logger.info(f"Duplicate alert suppressed: {alert.alert_id}")
                    return False
            
            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Alert correlation
            if self.config.enable_correlation:
                correlation_result = await self._correlate_alert(alert)
                if correlation_result:
                    self.metrics['alerts_correlated'] += 1
                    # Process correlated alerts as a group
                    await self._process_correlated_alerts(correlation_result)
                    return True
            
            # Root cause analysis
            if self.config.enable_root_cause_analysis:
                root_cause = await self._analyze_root_cause(alert)
                # Update alert with root cause information
                alert = self._enrich_alert_with_root_cause(alert, root_cause)
            
            # Deliver alert
            await self._deliver_alert(alert)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing alert {alert.alert_id}: {str(e)}")
            self.metrics['processing_errors'] += 1
            return False
    
    async def acknowledge_alert(self, alert_id: AlertId, user: str, notes: str = "") -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.active_alerts:
            logger.warning(f"Alert {alert_id} not found")
            return False
        
        alert = self.active_alerts[alert_id]
        acknowledged_alert = alert.acknowledge(user, notes)
        self.active_alerts[alert_id] = acknowledged_alert
        
        logger.info(f"Alert {alert_id} acknowledged by {user}")
        return True
    
    async def resolve_alert(self, alert_id: AlertId, user: str, action: str, notes: str = "") -> bool:
        """Resolve an alert."""
        if alert_id not in self.active_alerts:
            logger.warning(f"Alert {alert_id} not found")
            return False
        
        alert = self.active_alerts[alert_id]
        resolved_alert = alert.resolve(user, action, notes)
        self.active_alerts[alert_id] = resolved_alert
        
        logger.info(f"Alert {alert_id} resolved by {user}")
        return True
    
    async def escalate_alert(self, alert_id: AlertId, new_level: EscalationLevel) -> bool:
        """Manually escalate an alert."""
        if alert_id not in self.active_alerts:
            logger.warning(f"Alert {alert_id} not found")
            return False
        
        alert = self.active_alerts[alert_id]
        escalated_alert = alert.escalate(new_level)
        self.active_alerts[alert_id] = escalated_alert
        
        # Deliver escalated alert
        await self._deliver_alert(escalated_alert)
        
        self.metrics['alerts_escalated'] += 1
        logger.info(f"Alert {alert_id} escalated to {new_level.value}")
        return True
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_escalation_rule(self, rule: EscalationRule) -> None:
        """Add an escalation rule."""
        self.escalation_rules.append(rule)
        logger.info(f"Added escalation rule: {rule.name}")
    
    def configure_channel(self, channel: AlertChannel, config: AlertChannelConfig) -> None:
        """Configure an alert channel."""
        self.channel_configs[channel] = config
        logger.info(f"Configured alert channel: {channel.value}")
    
    def register_delivery_service(self, channel: AlertChannel, service: AlertDeliveryService) -> None:
        """Register a delivery service for a channel."""
        self.delivery_services[channel] = service
        logger.info(f"Registered delivery service for channel: {channel.value}")
    
    def get_active_alerts(self) -> List[QualityAlert]:
        """Get all active alerts."""
        return [alert for alert in self.active_alerts.values() if alert.is_active()]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting service statistics."""
        active_alerts = self.get_active_alerts()
        
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            'total_alerts_processed': self.metrics['alerts_processed'],
            'active_alerts': len(active_alerts),
            'alerts_by_severity': dict(severity_counts),
            'alerts_correlated': self.metrics['alerts_correlated'],
            'alerts_escalated': self.metrics['alerts_escalated'],
            'alerts_delivered': self.metrics['alerts_delivered'],
            'delivery_failures': self.metrics['delivery_failures'],
            'processing_errors': self.metrics['processing_errors'],
            'correlation_cache_size': len(self.correlation_cache)
        }
    
    # Private methods
    
    def _is_rate_limited(self, alert: QualityAlert) -> bool:
        """Check if alert is rate limited."""
        key = f"{alert.stream_id}:{alert.severity.value}"
        now = datetime.now()
        
        # Clean old entries
        rate_window = self.rate_limiter[key]
        while rate_window and (now - rate_window[0]).total_seconds() > 3600:
            rate_window.popleft()
        
        # Check rate limit
        if len(rate_window) >= self.config.rate_limit_per_hour:
            return True
        
        # Check burst limit
        burst_window = now - timedelta(minutes=self.config.burst_window_minutes)
        recent_alerts = [ts for ts in rate_window if ts > burst_window]
        
        if len(recent_alerts) >= self.config.burst_threshold:
            return True
        
        # Add current alert
        rate_window.append(now)
        return False
    
    def _is_duplicate_alert(self, alert: QualityAlert) -> bool:
        """Check if alert is a duplicate."""
        deduplication_window = datetime.now() - timedelta(minutes=self.config.deduplication_window_minutes)
        
        for existing_alert in self.active_alerts.values():
            if existing_alert.triggered_at < deduplication_window:
                continue
            
            # Check similarity
            similarity = self._calculate_alert_similarity(alert, existing_alert)
            if similarity >= self.config.deduplication_similarity_threshold:
                return True
        
        return False
    
    def _calculate_alert_similarity(self, alert1: QualityAlert, alert2: QualityAlert) -> float:
        """Calculate similarity between two alerts."""
        similarity_score = 0.0
        
        # Stream similarity
        if alert1.stream_id == alert2.stream_id:
            similarity_score += 0.3
        
        # Metric similarity
        if alert1.metric_name == alert2.metric_name:
            similarity_score += 0.3
        
        # Severity similarity
        if alert1.severity == alert2.severity:
            similarity_score += 0.2
        
        # Title similarity (simple word matching)
        words1 = set(alert1.title.lower().split())
        words2 = set(alert2.title.lower().split())
        if words1 and words2:
            word_similarity = len(words1 & words2) / len(words1 | words2)
            similarity_score += word_similarity * 0.2
        
        return similarity_score
    
    async def _correlate_alert(self, alert: QualityAlert) -> Optional[AlertCorrelationResult]:
        """Correlate alert with existing alerts."""
        correlation_window = datetime.now() - timedelta(minutes=self.config.correlation_window_minutes)
        
        # Find potential correlations
        candidates = []
        for existing_alert in self.active_alerts.values():
            if (existing_alert.triggered_at >= correlation_window and 
                existing_alert.alert_id != alert.alert_id):
                candidates.append(existing_alert)
        
        if not candidates:
            return None
        
        # Apply correlation strategies
        best_correlation = None
        best_score = 0.0
        
        for strategy in CorrelationStrategy:
            correlation = await self._apply_correlation_strategy(alert, candidates, strategy)
            if correlation and correlation.correlation_score > best_score:
                best_correlation = correlation
                best_score = correlation.correlation_score
        
        if best_score >= self.config.correlation_threshold:
            return best_correlation
        
        return None
    
    async def _apply_correlation_strategy(self, 
                                        alert: QualityAlert, 
                                        candidates: List[QualityAlert], 
                                        strategy: CorrelationStrategy) -> Optional[AlertCorrelationResult]:
        """Apply a specific correlation strategy."""
        if strategy == CorrelationStrategy.TEMPORAL:
            return await self._temporal_correlation(alert, candidates)
        elif strategy == CorrelationStrategy.THRESHOLD_BASED:
            return await self._threshold_correlation(alert, candidates)
        elif strategy == CorrelationStrategy.PATTERN_BASED:
            return await self._pattern_correlation(alert, candidates)
        elif strategy == CorrelationStrategy.STATISTICAL:
            return await self._statistical_correlation(alert, candidates)
        else:
            return None
    
    async def _temporal_correlation(self, alert: QualityAlert, candidates: List[QualityAlert]) -> Optional[AlertCorrelationResult]:
        """Apply temporal correlation strategy."""
        # Group alerts by time proximity
        time_window = timedelta(minutes=5)
        correlated = []
        
        for candidate in candidates:
            if abs((alert.triggered_at - candidate.triggered_at).total_seconds()) <= time_window.total_seconds():
                correlated.append(candidate)
        
        if len(correlated) >= 2:  # At least 2 other alerts
            return AlertCorrelationResult(
                correlated_alerts=correlated,
                correlation_score=0.8,
                correlation_type=CorrelationStrategy.TEMPORAL,
                root_cause_analysis={'type': 'temporal_cascade', 'time_window': time_window.total_seconds()},
                recommended_actions=['Check upstream systems', 'Review recent changes']
            )
        
        return None
    
    async def _threshold_correlation(self, alert: QualityAlert, candidates: List[QualityAlert]) -> Optional[AlertCorrelationResult]:
        """Apply threshold-based correlation strategy."""
        # Group alerts by threshold violations
        same_threshold_alerts = []
        
        for candidate in candidates:
            if candidate.threshold_id == alert.threshold_id:
                same_threshold_alerts.append(candidate)
        
        if len(same_threshold_alerts) >= 1:
            return AlertCorrelationResult(
                correlated_alerts=same_threshold_alerts,
                correlation_score=0.9,
                correlation_type=CorrelationStrategy.THRESHOLD_BASED,
                root_cause_analysis={'type': 'threshold_violation', 'threshold_id': str(alert.threshold_id)},
                recommended_actions=['Review threshold configuration', 'Check data quality upstream']
            )
        
        return None
    
    async def _pattern_correlation(self, alert: QualityAlert, candidates: List[QualityAlert]) -> Optional[AlertCorrelationResult]:
        """Apply pattern-based correlation strategy."""
        # Simple pattern matching based on alert titles
        pattern_alerts = []
        
        for candidate in candidates:
            if self._extract_pattern(alert.title) == self._extract_pattern(candidate.title):
                pattern_alerts.append(candidate)
        
        if len(pattern_alerts) >= 2:
            return AlertCorrelationResult(
                correlated_alerts=pattern_alerts,
                correlation_score=0.7,
                correlation_type=CorrelationStrategy.PATTERN_BASED,
                root_cause_analysis={'type': 'pattern_match', 'pattern': self._extract_pattern(alert.title)},
                recommended_actions=['Investigate common root cause', 'Review system patterns']
            )
        
        return None
    
    async def _statistical_correlation(self, alert: QualityAlert, candidates: List[QualityAlert]) -> Optional[AlertCorrelationResult]:
        """Apply statistical correlation strategy."""
        # Group alerts by statistical anomalies
        statistical_alerts = []
        
        for candidate in candidates:
            if self._is_statistical_anomaly(alert, candidate):
                statistical_alerts.append(candidate)
        
        if len(statistical_alerts) >= 1:
            return AlertCorrelationResult(
                correlated_alerts=statistical_alerts,
                correlation_score=0.75,
                correlation_type=CorrelationStrategy.STATISTICAL,
                root_cause_analysis={'type': 'statistical_anomaly', 'correlation_method': 'value_comparison'},
                recommended_actions=['Perform statistical analysis', 'Check data distribution']
            )
        
        return None
    
    def _extract_pattern(self, text: str) -> str:
        """Extract pattern from alert title."""
        # Simple pattern extraction - remove numbers and specific values
        pattern = re.sub(r'\d+', 'N', text)
        pattern = re.sub(r'[0-9.]+', 'N', pattern)
        return pattern
    
    def _is_statistical_anomaly(self, alert1: QualityAlert, alert2: QualityAlert) -> bool:
        """Check if two alerts represent statistical anomalies."""
        # Simple check - if values are significantly different from threshold
        if alert1.metric_name == alert2.metric_name:
            threshold = alert1.threshold_value
            deviation1 = abs(alert1.actual_value - threshold)
            deviation2 = abs(alert2.actual_value - threshold)
            
            # Check if both deviations are significant
            return deviation1 > threshold * 0.1 and deviation2 > threshold * 0.1
        
        return False
    
    async def _process_correlated_alerts(self, correlation: AlertCorrelationResult) -> None:
        """Process correlated alerts as a group."""
        # Create a summary alert or handle the correlation
        logger.info(f"Processing {len(correlation.correlated_alerts)} correlated alerts")
        
        # For now, just deliver the primary alert with correlation info
        primary_alert = correlation.correlated_alerts[0]
        
        # Add correlation information to context
        enriched_context = primary_alert.context_data.copy()
        enriched_context.update({
            'correlation_type': correlation.correlation_type.value,
            'correlation_score': correlation.correlation_score,
            'correlated_alert_count': len(correlation.correlated_alerts),
            'root_cause_analysis': correlation.root_cause_analysis,
            'recommended_actions': correlation.recommended_actions
        })
        
        # Create enriched alert
        enriched_alert = QualityAlert(
            alert_id=primary_alert.alert_id,
            threshold_id=primary_alert.threshold_id,
            monitoring_job_id=primary_alert.monitoring_job_id,
            stream_id=primary_alert.stream_id,
            severity=primary_alert.severity,
            status=primary_alert.status,
            title=f"[CORRELATED] {primary_alert.title}",
            description=f"{primary_alert.description}\n\nCorrelated with {len(correlation.correlated_alerts)} other alerts",
            triggered_at=primary_alert.triggered_at,
            acknowledged_at=primary_alert.acknowledged_at,
            resolved_at=primary_alert.resolved_at,
            metric_name=primary_alert.metric_name,
            actual_value=primary_alert.actual_value,
            threshold_value=primary_alert.threshold_value,
            affected_records=primary_alert.affected_records,
            context_data=enriched_context,
            escalation_level=primary_alert.escalation_level,
            escalated_at=primary_alert.escalated_at,
            assigned_to=primary_alert.assigned_to,
            resolution_notes=primary_alert.resolution_notes,
            resolution_action=primary_alert.resolution_action
        )
        
        await self._deliver_alert(enriched_alert)
    
    async def _analyze_root_cause(self, alert: QualityAlert) -> Dict[str, Any]:
        """Analyze root cause of an alert."""
        root_cause = {
            'analysis_timestamp': datetime.now().isoformat(),
            'primary_factors': [],
            'contributing_factors': [],
            'confidence_score': 0.0,
            'recommended_actions': []
        }
        
        # Analyze based on alert characteristics
        if alert.severity == AlertSeverity.CRITICAL:
            root_cause['primary_factors'].append('Critical quality degradation')
            root_cause['confidence_score'] = 0.9
        
        if alert.affected_records > 1000:
            root_cause['contributing_factors'].append('Large dataset impact')
            root_cause['confidence_score'] += 0.1
        
        # Add metric-specific analysis
        if 'completeness' in alert.metric_name.lower():
            root_cause['primary_factors'].append('Missing data at source')
            root_cause['recommended_actions'].append('Check data pipeline integrity')
        elif 'accuracy' in alert.metric_name.lower():
            root_cause['primary_factors'].append('Data validation failures')
            root_cause['recommended_actions'].append('Review validation rules')
        
        return root_cause
    
    def _enrich_alert_with_root_cause(self, alert: QualityAlert, root_cause: Dict[str, Any]) -> QualityAlert:
        """Enrich alert with root cause analysis."""
        enriched_context = alert.context_data.copy()
        enriched_context['root_cause_analysis'] = root_cause
        
        return QualityAlert(
            alert_id=alert.alert_id,
            threshold_id=alert.threshold_id,
            monitoring_job_id=alert.monitoring_job_id,
            stream_id=alert.stream_id,
            severity=alert.severity,
            status=alert.status,
            title=alert.title,
            description=alert.description,
            triggered_at=alert.triggered_at,
            acknowledged_at=alert.acknowledged_at,
            resolved_at=alert.resolved_at,
            metric_name=alert.metric_name,
            actual_value=alert.actual_value,
            threshold_value=alert.threshold_value,
            affected_records=alert.affected_records,
            context_data=enriched_context,
            escalation_level=alert.escalation_level,
            escalated_at=alert.escalated_at,
            assigned_to=alert.assigned_to,
            resolution_notes=alert.resolution_notes,
            resolution_action=alert.resolution_action
        )
    
    async def _deliver_alert(self, alert: QualityAlert) -> None:
        """Deliver alert through configured channels."""
        delivery_tasks = []
        
        for channel, config in self.channel_configs.items():
            if not config.should_send(alert):
                continue
            
            service = self.delivery_services.get(channel)
            if not service:
                logger.warning(f"No delivery service configured for channel {channel.value}")
                continue
            
            # Create delivery task
            task = asyncio.create_task(self._deliver_to_channel(alert, service, config))
            delivery_tasks.append(task)
        
        # Wait for all deliveries to complete
        if delivery_tasks:
            results = await asyncio.gather(*delivery_tasks, return_exceptions=True)
            
            success_count = sum(1 for result in results if result is True)
            self.metrics['alerts_delivered'] += success_count
            self.metrics['delivery_failures'] += len(results) - success_count
    
    async def _deliver_to_channel(self, alert: QualityAlert, service: AlertDeliveryService, config: AlertChannelConfig) -> bool:
        """Deliver alert to a specific channel."""
        for attempt in range(self.config.max_delivery_attempts):
            try:
                success = await service.send_alert(alert, config)
                if success:
                    return True
                
                # Wait before retry
                if attempt < self.config.max_delivery_attempts - 1:
                    await asyncio.sleep(self.config.delivery_retry_delay_seconds)
                    
            except Exception as e:
                logger.error(f"Delivery attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.config.max_delivery_attempts - 1:
                    await asyncio.sleep(self.config.delivery_retry_delay_seconds)
        
        return False
    
    async def _escalation_monitor(self) -> None:
        """Monitor alerts for escalation."""
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                
                for alert in list(self.active_alerts.values()):
                    if not alert.is_active():
                        continue
                    
                    # Check escalation rules
                    for rule in self.escalation_rules:
                        if rule.should_escalate(alert, current_time):
                            await self.escalate_alert(alert.alert_id, rule.to_level)
                            break
                
                # Sleep before next check
                await asyncio.sleep(self.config.escalation_check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in escalation monitor: {str(e)}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _batch_delivery_monitor(self) -> None:
        """Monitor for batch delivery opportunities."""
        while not self.shutdown_event.is_set():
            try:
                # Wait for batch interval
                await asyncio.sleep(self.config.batch_delivery_interval_minutes * 60)
                
                # Check for batch delivery opportunities
                await self._process_batch_deliveries()
                
            except Exception as e:
                logger.error(f"Error in batch delivery monitor: {str(e)}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _process_batch_deliveries(self) -> None:
        """Process batch deliveries."""
        # Group alerts by channel for batch delivery
        batch_groups = defaultdict(list)
        
        for alert in self.active_alerts.values():
            if alert.is_active():
                for channel, config in self.channel_configs.items():
                    if config.should_send(alert) and len(batch_groups[channel]) < self.config.max_batch_size:
                        batch_groups[channel].append(alert)
        
        # Send batches
        for channel, alerts in batch_groups.items():
            if len(alerts) >= 2:  # Only batch if multiple alerts
                service = self.delivery_services.get(channel)
                config = self.channel_configs.get(channel)
                
                if service and config:
                    try:
                        await service.send_batch_alerts(alerts, config)
                        logger.info(f"Sent batch of {len(alerts)} alerts to {channel.value}")
                    except Exception as e:
                        logger.error(f"Failed to send batch alerts to {channel.value}: {str(e)}")
    
    async def _cleanup_monitor(self) -> None:
        """Monitor for cleanup opportunities."""
        while not self.shutdown_event.is_set():
            try:
                # Wait for cleanup interval
                await asyncio.sleep(3600)  # 1 hour
                
                # Cleanup resolved alerts
                self._cleanup_resolved_alerts()
                
                # Cleanup correlation cache
                self._cleanup_correlation_cache()
                
            except Exception as e:
                logger.error(f"Error in cleanup monitor: {str(e)}")
                await asyncio.sleep(60)  # Wait before retry
    
    def _cleanup_resolved_alerts(self) -> None:
        """Clean up resolved alerts."""
        retention_hours = 24
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        alerts_to_remove = []
        for alert_id, alert in self.active_alerts.items():
            if alert.is_resolved() and alert.resolved_at and alert.resolved_at < cutoff_time:
                alerts_to_remove.append(alert_id)
        
        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]
        
        if alerts_to_remove:
            logger.info(f"Cleaned up {len(alerts_to_remove)} resolved alerts")
    
    def _cleanup_correlation_cache(self) -> None:
        """Clean up correlation cache."""
        # Simple cache cleanup - remove old entries
        if len(self.correlation_cache) > 1000:
            # Keep only the most recent 500 entries
            items = list(self.correlation_cache.items())
            self.correlation_cache = dict(items[-500:])
            logger.info("Cleaned up correlation cache")