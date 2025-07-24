"""Comprehensive alerting system for model performance monitoring."""

from __future__ import annotations

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional, Protocol, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import aiohttp
from abc import ABC, abstractmethod

from ..logging import get_logger
from .model_performance_monitor import PerformanceAlert, AlertThreshold

logger = get_logger(__name__)


class AlertChannel(Protocol):
    """Protocol for alert delivery channels."""
    
    async def send_alert(self, alert: PerformanceAlert) -> bool:
        """Send an alert through this channel."""
        ...


@dataclass
class SlackConfig:
    """Slack webhook configuration."""
    
    webhook_url: str
    channel: Optional[str] = None
    username: Optional[str] = "Model Monitor"
    icon_emoji: Optional[str] = ":warning:"


@dataclass
class EmailConfig:
    """Email configuration."""
    
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    from_email: str
    to_emails: List[str]
    use_tls: bool = True


@dataclass
class WebhookConfig:
    """Generic webhook configuration."""
    
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30


class SlackAlertChannel:
    """Slack alert delivery channel."""
    
    def __init__(self, config: SlackConfig):
        self.config = config
    
    async def send_alert(self, alert: PerformanceAlert) -> bool:
        """Send alert to Slack."""
        
        try:
            # Create Slack message
            color = {
                "critical": "danger",
                "warning": "warning", 
                "info": "good"
            }.get(alert.severity, "warning")
            
            message = {
                "channel": self.config.channel,
                "username": self.config.username,
                "icon_emoji": self.config.icon_emoji,
                "attachments": [{
                    "color": color,
                    "title": f"Model Performance Alert - {alert.severity.upper()}",
                    "fields": [
                        {
                            "title": "Model ID",
                            "value": alert.model_id,
                            "short": True
                        },
                        {
                            "title": "Metric",
                            "value": alert.metric_name,
                            "short": True
                        },
                        {
                            "title": "Current Value",
                            "value": f"{alert.current_value:.4f}",
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": f"{alert.threshold_value:.4f}",
                            "short": True
                        },
                        {
                            "title": "Message",
                            "value": alert.message,
                            "short": False
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.isoformat(),
                            "short": True
                        }
                    ],
                    "footer": "Model Performance Monitor",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    success = response.status == 200
                    
                    if success:
                        logger.info("Alert sent to Slack",
                                   alert_id=alert.alert_id,
                                   channel=self.config.channel)
                    else:
                        logger.error("Failed to send Slack alert",
                                    alert_id=alert.alert_id,
                                    status=response.status)
                    
                    return success
        
        except Exception as e:
            logger.error("Error sending Slack alert",
                        alert_id=alert.alert_id,
                        error=str(e))
            return False


class EmailAlertChannel:
    """Email alert delivery channel."""
    
    def __init__(self, config: EmailConfig):
        self.config = config
    
    async def send_alert(self, alert: PerformanceAlert) -> bool:
        """Send alert via email."""
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config.from_email
            msg['To'] = ', '.join(self.config.to_emails)
            msg['Subject'] = f"Model Performance Alert - {alert.severity.upper()}: {alert.model_id}"
            
            # Create HTML body
            html_body = self._create_html_body(alert)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self._send_email, msg)
            
            if success:
                logger.info("Alert sent via email",
                           alert_id=alert.alert_id,
                           to_emails=self.config.to_emails)
            
            return success
        
        except Exception as e:
            logger.error("Error sending email alert",
                        alert_id=alert.alert_id,
                        error=str(e))
            return False
    
    def _send_email(self, msg: MIMEMultipart) -> bool:
        """Send email using SMTP."""
        
        try:
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            
            if self.config.use_tls:
                server.starttls()
            
            server.login(self.config.username, self.config.password)
            server.send_message(msg)
            server.quit()
            
            return True
        
        except Exception as e:
            logger.error("SMTP error", error=str(e))
            return False
    
    def _create_html_body(self, alert: PerformanceAlert) -> str:
        """Create HTML email body."""
        
        severity_color = {
            "critical": "#d73502",
            "warning": "#ff8c00",
            "info": "#0066cc"
        }.get(alert.severity, "#ff8c00")
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="border-left: 4px solid {severity_color}; padding-left: 20px;">
                <h2 style="color: {severity_color};">Model Performance Alert</h2>
                
                <table style="border-collapse: collapse; width: 100%; margin-top: 20px;">
                    <tr>
                        <td style="padding: 8px; font-weight: bold; width: 150px;">Severity:</td>
                        <td style="padding: 8px; color: {severity_color};">{alert.severity.upper()}</td>
                    </tr>
                    <tr style="background-color: #f9f9f9;">
                        <td style="padding: 8px; font-weight: bold;">Model ID:</td>
                        <td style="padding: 8px;">{alert.model_id}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Metric:</td>
                        <td style="padding: 8px;">{alert.metric_name}</td>
                    </tr>
                    <tr style="background-color: #f9f9f9;">
                        <td style="padding: 8px; font-weight: bold;">Current Value:</td>
                        <td style="padding: 8px;">{alert.current_value:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Threshold:</td>
                        <td style="padding: 8px;">{alert.threshold_value:.4f}</td>
                    </tr>
                    <tr style="background-color: #f9f9f9;">
                        <td style="padding: 8px; font-weight: bold;">Timestamp:</td>
                        <td style="padding: 8px;">{alert.timestamp.isoformat()}</td>
                    </tr>
                </table>
                
                <div style="margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 5px;">
                    <strong>Message:</strong><br>
                    {alert.message}
                </div>
                
                <p style="margin-top: 20px; font-size: 12px; color: #666;">
                    This alert was generated by the Model Performance Monitor.
                </p>
            </div>
        </body>
        </html>
        """


class WebhookAlertChannel:
    """Generic webhook alert delivery channel."""
    
    def __init__(self, config: WebhookConfig):
        self.config = config
    
    async def send_alert(self, alert: PerformanceAlert) -> bool:
        """Send alert to webhook."""
        
        try:
            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "model_id": alert.model_id,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.url,
                    json=payload,
                    headers=self.config.headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    success = 200 <= response.status < 300
                    
                    if success:
                        logger.info("Alert sent to webhook",
                                   alert_id=alert.alert_id,
                                   url=self.config.url)
                    else:
                        logger.error("Failed to send webhook alert",
                                    alert_id=alert.alert_id,
                                    url=self.config.url,
                                    status=response.status)
                    
                    return success
        
        except Exception as e:
            logger.error("Error sending webhook alert",
                        alert_id=alert.alert_id,
                        url=self.config.url,
                        error=str(e))
            return False


@dataclass
class AlertRule:
    """Alert routing and escalation rule."""
    
    name: str
    conditions: Dict[str, Any]  # Conditions for triggering this rule
    channels: List[str]  # Channel names to use
    escalation_minutes: Optional[int] = None  # Minutes before escalation
    escalation_channels: Optional[List[str]] = None  # Escalation channels
    enabled: bool = True


class AlertingSystem:
    """Comprehensive alerting system for model performance monitoring."""
    
    def __init__(self):
        self._channels: Dict[str, AlertChannel] = {}
        self._rules: List[AlertRule] = []
        self._alert_history: List[Dict[str, Any]] = []
        self._escalation_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Alerting system initialized")
    
    def add_slack_channel(self, name: str, config: SlackConfig) -> None:
        """Add a Slack alert channel."""
        
        self._channels[name] = SlackAlertChannel(config)
        logger.info("Slack channel added", name=name, channel=config.channel)
    
    def add_email_channel(self, name: str, config: EmailConfig) -> None:
        """Add an email alert channel."""
        
        self._channels[name] = EmailAlertChannel(config)
        logger.info("Email channel added", name=name, to_emails=config.to_emails)
    
    def add_webhook_channel(self, name: str, config: WebhookConfig) -> None:
        """Add a webhook alert channel."""
        
        self._channels[name] = WebhookAlertChannel(config)
        logger.info("Webhook channel added", name=name, url=config.url)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert routing rule."""
        
        self._rules.append(rule)
        logger.info("Alert rule added",
                   name=rule.name,
                   channels=rule.channels,
                   conditions=rule.conditions)
    
    async def process_alert(self, alert: PerformanceAlert) -> None:
        """Process and route an alert based on rules."""
        
        try:
            # Find matching rules
            matching_rules = self._find_matching_rules(alert)
            
            if not matching_rules:
                # Use default routing
                await self._send_to_all_channels(alert)
                return
            
            # Process each matching rule
            for rule in matching_rules:
                if not rule.enabled:
                    continue
                
                # Send to rule channels
                await self._send_to_channels(alert, rule.channels)
                
                # Setup escalation if configured
                if rule.escalation_minutes and rule.escalation_channels:
                    await self._setup_escalation(alert, rule)
            
            # Record alert
            self._record_alert_delivery(alert, matching_rules)
        
        except Exception as e:
            logger.error("Error processing alert",
                        alert_id=alert.alert_id,
                        error=str(e))
    
    def _find_matching_rules(self, alert: PerformanceAlert) -> List[AlertRule]:
        """Find alert rules that match the given alert."""
        
        matching_rules = []
        
        for rule in self._rules:
            if self._rule_matches_alert(rule, alert):
                matching_rules.append(rule)
        
        return matching_rules
    
    def _rule_matches_alert(self, rule: AlertRule, alert: PerformanceAlert) -> bool:
        """Check if a rule matches an alert."""
        
        conditions = rule.conditions
        
        # Check severity condition
        if "severity" in conditions:
            severities = conditions["severity"]
            if isinstance(severities, str):
                severities = [severities]
            if alert.severity not in severities:
                return False
        
        # Check model condition
        if "model_id" in conditions:
            model_patterns = conditions["model_id"]
            if isinstance(model_patterns, str):
                model_patterns = [model_patterns]
            
            model_match = False
            for pattern in model_patterns:
                if pattern == "*" or pattern in alert.model_id:
                    model_match = True
                    break
            
            if not model_match:
                return False
        
        # Check metric condition
        if "metric_name" in conditions:
            metrics = conditions["metric_name"]
            if isinstance(metrics, str):
                metrics = [metrics]
            if alert.metric_name not in metrics:
                return False
        
        # Check time condition
        if "time_of_day" in conditions:
            current_hour = alert.timestamp.hour
            time_range = conditions["time_of_day"]
            if not (time_range[0] <= current_hour <= time_range[1]):
                return False
        
        return True
    
    async def _send_to_channels(self, alert: PerformanceAlert, channel_names: List[str]) -> None:
        """Send alert to specified channels."""
        
        tasks = []
        for channel_name in channel_names:
            if channel_name in self._channels:
                task = self._channels[channel_name].send_alert(alert)
                tasks.append(task)
            else:
                logger.warning("Unknown alert channel", channel_name=channel_name)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if r is True)
            logger.info("Alert sent to channels",
                       alert_id=alert.alert_id,
                       channels=channel_names,
                       success_count=success_count,
                       total_channels=len(tasks))
    
    async def _send_to_all_channels(self, alert: PerformanceAlert) -> None:
        """Send alert to all configured channels."""
        
        if not self._channels:
            logger.warning("No alert channels configured", alert_id=alert.alert_id)
            return
        
        await self._send_to_channels(alert, list(self._channels.keys()))
    
    async def _setup_escalation(self, alert: PerformanceAlert, rule: AlertRule) -> None:
        """Setup alert escalation."""
        
        if not rule.escalation_channels or not rule.escalation_minutes:
            return
        
        escalation_key = f"{alert.alert_id}_{rule.name}"
        
        # Cancel existing escalation if any
        if escalation_key in self._escalation_tasks:
            self._escalation_tasks[escalation_key].cancel()
        
        # Schedule escalation
        async def escalate():
            await asyncio.sleep(rule.escalation_minutes * 60)
            
            # Check if alert is still active
            if not alert.resolved:
                logger.info("Escalating alert",
                           alert_id=alert.alert_id,
                           rule_name=rule.name,
                           escalation_channels=rule.escalation_channels)
                
                await self._send_to_channels(alert, rule.escalation_channels)
            
            # Clean up task
            if escalation_key in self._escalation_tasks:
                del self._escalation_tasks[escalation_key]
        
        self._escalation_tasks[escalation_key] = asyncio.create_task(escalate())
    
    def _record_alert_delivery(self, alert: PerformanceAlert, rules: List[AlertRule]) -> None:
        """Record alert delivery for auditing."""
        
        record = {
            "alert_id": alert.alert_id,
            "model_id": alert.model_id,
            "metric_name": alert.metric_name,
            "severity": alert.severity,
            "timestamp": alert.timestamp.isoformat(),
            "rules_matched": [rule.name for rule in rules],
            "channels_used": list(set(channel for rule in rules for channel in rule.channels))
        }
        
        self._alert_history.append(record)
        
        # Keep only recent history (last 1000 alerts)
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alerting system statistics."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_alerts = [
            record for record in self._alert_history
            if datetime.fromisoformat(record["timestamp"]) >= cutoff_time
        ]
        
        # Calculate statistics
        total_alerts = len(recent_alerts)
        
        severity_counts = {}
        model_counts = {}
        channel_counts = {}
        
        for record in recent_alerts:
            # Severity counts
            severity = record["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Model counts
            model_id = record["model_id"]
            model_counts[model_id] = model_counts.get(model_id, 0) + 1
            
            # Channel counts
            for channel in record["channels_used"]:
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
        
        return {
            "period_hours": hours,
            "total_alerts": total_alerts,
            "severity_distribution": severity_counts,
            "model_distribution": model_counts,
            "channel_usage": channel_counts,
            "configured_channels": list(self._channels.keys()),
            "configured_rules": len(self._rules),
            "active_escalations": len(self._escalation_tasks)
        }
    
    def export_configuration(self, output_path: Path) -> None:
        """Export alerting configuration."""
        
        config = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "channels": list(self._channels.keys()),
            "rules": [
                {
                    "name": rule.name,
                    "conditions": rule.conditions,
                    "channels": rule.channels,
                    "escalation_minutes": rule.escalation_minutes,
                    "escalation_channels": rule.escalation_channels,
                    "enabled": rule.enabled
                }
                for rule in self._rules
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Alerting configuration exported", output_path=str(output_path))
    
    async def test_channels(self) -> Dict[str, bool]:
        """Test all configured alert channels."""
        
        # Create test alert
        test_alert = PerformanceAlert(
            alert_id="test_alert",
            metric_name="test_metric",
            current_value=1.0,
            threshold_value=0.5,
            severity="info",
            message="This is a test alert from the Model Performance Monitor",
            timestamp=datetime.utcnow(),
            model_id="test_model"
        )
        
        # Test each channel
        results = {}
        for channel_name, channel in self._channels.items():
            try:
                success = await channel.send_alert(test_alert)
                results[channel_name] = success
                
                logger.info("Channel test completed",
                           channel_name=channel_name,
                           success=success)
            
            except Exception as e:
                results[channel_name] = False
                logger.error("Channel test failed",
                            channel_name=channel_name,
                            error=str(e))
        
        return results


# Global instance
_alerting_system: Optional[AlertingSystem] = None


def get_alerting_system() -> AlertingSystem:
    """Get the global alerting system instance."""
    
    global _alerting_system
    
    if _alerting_system is None:
        _alerting_system = AlertingSystem()
    
    return _alerting_system


def initialize_alerting() -> AlertingSystem:
    """Initialize the global alerting system."""
    
    global _alerting_system
    _alerting_system = AlertingSystem()
    return _alerting_system