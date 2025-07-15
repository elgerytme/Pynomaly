"""
Performance Alert System.

Provides intelligent alerting for performance regressions with multiple notification
channels, severity-based routing, and alert deduplication/throttling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

import requests
from jinja2 import Template

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for performance alerts."""
    
    enabled: bool = True
    channels: List[str] = field(default_factory=lambda: ['console', 'github'])
    severity_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'critical': {'response_time_multiplier': 5.0, 'error_rate_threshold': 10.0},
        'high': {'response_time_multiplier': 3.0, 'error_rate_threshold': 5.0},
        'medium': {'response_time_multiplier': 2.0, 'error_rate_threshold': 2.0},
        'low': {'response_time_multiplier': 1.5, 'error_rate_threshold': 1.0}
    })
    throttle_minutes: int = 30
    max_alerts_per_hour: int = 10
    alert_history_days: int = 7


@dataclass
class Alert:
    """Performance alert data structure."""
    
    id: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    title: str
    message: str
    metric_name: str
    current_value: float
    baseline_value: float
    deviation: float
    timestamp: datetime
    run_id: str
    environment: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'severity': self.severity,
            'title': self.title,
            'message': self.message,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'baseline_value': self.baseline_value,
            'deviation': self.deviation,
            'timestamp': self.timestamp.isoformat(),
            'run_id': self.run_id,
            'environment': self.environment,
            'tags': self.tags,
            'resolved': self.resolved
        }


class AlertChannel(ABC):
    """Abstract base class for alert notification channels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate channel configuration."""
        pass


class ConsoleAlertChannel(AlertChannel):
    """Console/logging alert channel."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to console/logs."""
        severity_icons = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '⚡',
            'low': 'ℹ️'
        }
        
        icon = severity_icons.get(alert.severity, '📊')
        
        log_message = (
            f"{icon} PERFORMANCE ALERT [{alert.severity.upper()}]\n"
            f"Title: {alert.title}\n"
            f"Metric: {alert.metric_name}\n"
            f"Current: {alert.current_value:.2f}\n"
            f"Baseline: {alert.baseline_value:.2f}\n"
            f"Deviation: {alert.deviation:.2f} std\n"
            f"Run ID: {alert.run_id}\n"
            f"Time: {alert.timestamp}\n"
            f"Message: {alert.message}"
        )
        
        if alert.severity in ['critical', 'high']:
            logger.error(log_message)
        elif alert.severity == 'medium':
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        return True
    
    def validate_config(self) -> bool:
        """Console channel always valid."""
        return True


class GitHubAlertChannel(AlertChannel):
    """GitHub issue/PR comment alert channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.token = config.get('github_token')
        self.repo = config.get('repository')  # format: 'owner/repo'
        self.pr_number = config.get('pr_number')  # for PR comments
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert as GitHub comment or issue."""
        if not self.validate_config():
            logger.error("GitHub alert channel not properly configured")
            return False
        
        try:
            if self.pr_number:
                return await self._send_pr_comment(alert)
            else:
                return await self._create_issue(alert)
        
        except Exception as e:
            logger.error(f"Failed to send GitHub alert: {e}")
            return False
    
    async def _send_pr_comment(self, alert: Alert) -> bool:
        """Send alert as PR comment."""
        comment_body = self._format_github_message(alert)
        
        url = f"https://api.github.com/repos/{self.repo}/issues/{self.pr_number}/comments"
        headers = {
            'Authorization': f'token {self.token}',
            'Content-Type': 'application/json'
        }
        data = {'body': comment_body}
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        logger.info(f"GitHub PR comment sent for alert {alert.id}")
        return True
    
    async def _create_issue(self, alert: Alert) -> bool:
        """Create GitHub issue for alert."""
        issue_title = f"Performance Alert: {alert.title}"
        issue_body = self._format_github_message(alert)
        
        url = f"https://api.github.com/repos/{self.repo}/issues"
        headers = {
            'Authorization': f'token {self.token}',
            'Content-Type': 'application/json'
        }
        data = {
            'title': issue_title,
            'body': issue_body,
            'labels': ['performance', f'severity-{alert.severity}']
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        logger.info(f"GitHub issue created for alert {alert.id}")
        return True
    
    def _format_github_message(self, alert: Alert) -> str:
        """Format alert message for GitHub."""
        severity_emoji = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '⚡',
            'low': 'ℹ️'
        }
        
        emoji = severity_emoji.get(alert.severity, '📊')
        
        return f"""
{emoji} **Performance Alert - {alert.severity.title()} Severity**

## Alert Details

| Field | Value |
|-------|-------|
| **Metric** | `{alert.metric_name}` |
| **Current Value** | {alert.current_value:.2f} |
| **Baseline Value** | {alert.baseline_value:.2f} |
| **Deviation** | {alert.deviation:.2f} standard deviations |
| **Run ID** | `{alert.run_id}` |
| **Timestamp** | {alert.timestamp} |

## Description

{alert.message}

## Environment

```json
{json.dumps(alert.environment, indent=2)}
```

---
*This alert was automatically generated by the Pynomaly performance monitoring system.*
"""
    
    def validate_config(self) -> bool:
        """Validate GitHub configuration."""
        return bool(self.token and self.repo)


class SlackAlertChannel(AlertChannel):
    """Slack webhook alert channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'Pynomaly Performance Bot')
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self.validate_config():
            logger.error("Slack alert channel not properly configured")
            return False
        
        try:
            payload = self._create_slack_payload(alert)
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for {alert.id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _create_slack_payload(self, alert: Alert) -> Dict[str, Any]:
        """Create Slack message payload."""
        severity_colors = {
            'critical': '#FF0000',
            'high': '#FF8C00',
            'medium': '#FFD700',
            'low': '#32CD32'
        }
        
        color = severity_colors.get(alert.severity, '#808080')
        
        return {
            'channel': self.channel,
            'username': self.username,
            'attachments': [{
                'color': color,
                'title': f"Performance Alert: {alert.title}",
                'text': alert.message,
                'fields': [
                    {'title': 'Severity', 'value': alert.severity.title(), 'short': True},
                    {'title': 'Metric', 'value': alert.metric_name, 'short': True},
                    {'title': 'Current Value', 'value': f"{alert.current_value:.2f}", 'short': True},
                    {'title': 'Baseline Value', 'value': f"{alert.baseline_value:.2f}", 'short': True},
                    {'title': 'Deviation', 'value': f"{alert.deviation:.2f} std", 'short': True},
                    {'title': 'Run ID', 'value': alert.run_id, 'short': True}
                ],
                'footer': 'Pynomaly Performance Monitor',
                'ts': int(alert.timestamp.timestamp())
            }]
        }
    
    def validate_config(self) -> bool:
        """Validate Slack configuration."""
        return bool(self.webhook_url)


class EmailAlertChannel(AlertChannel):
    """Email alert channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email')
        self.to_emails = config.get('to_emails', [])
        self.use_tls = config.get('use_tls', True)
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not self.validate_config():
            logger.error("Email alert channel not properly configured")
            return False
        
        try:
            msg = self._create_email_message(alert)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                
                for to_email in self.to_emails:
                    server.send_message(msg, to_addrs=[to_email])
            
            logger.info(f"Email alert sent for {alert.id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _create_email_message(self, alert: Alert) -> MIMEMultipart:
        """Create email message."""
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Performance Alert: {alert.title} [{alert.severity.upper()}]"
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        
        # Create text content
        text_content = f"""
Performance Alert - {alert.severity.title()} Severity

Alert Details:
- Metric: {alert.metric_name}
- Current Value: {alert.current_value:.2f}
- Baseline Value: {alert.baseline_value:.2f}
- Deviation: {alert.deviation:.2f} standard deviations
- Run ID: {alert.run_id}
- Timestamp: {alert.timestamp}

Description:
{alert.message}

Environment:
{json.dumps(alert.environment, indent=2)}

This alert was automatically generated by the Pynomaly performance monitoring system.
"""
        
        # Create HTML content
        html_content = f"""
<html>
<body>
<h2>Performance Alert - {alert.severity.title()} Severity</h2>

<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Metric</th><td>{alert.metric_name}</td></tr>
<tr><th>Current Value</th><td>{alert.current_value:.2f}</td></tr>
<tr><th>Baseline Value</th><td>{alert.baseline_value:.2f}</td></tr>
<tr><th>Deviation</th><td>{alert.deviation:.2f} standard deviations</td></tr>
<tr><th>Run ID</th><td>{alert.run_id}</td></tr>
<tr><th>Timestamp</th><td>{alert.timestamp}</td></tr>
</table>

<h3>Description</h3>
<p>{alert.message}</p>

<h3>Environment</h3>
<pre>{json.dumps(alert.environment, indent=2)}</pre>

<p><em>This alert was automatically generated by the Pynomaly performance monitoring system.</em></p>
</body>
</html>
"""
        
        msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))
        
        return msg
    
    def validate_config(self) -> bool:
        """Validate email configuration."""
        return bool(
            self.smtp_server and 
            self.username and 
            self.password and 
            self.from_email and 
            self.to_emails
        )


class AlertThrottler:
    """Manages alert throttling and deduplication."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.alert_history: List[Alert] = []
        self.alert_counts: Dict[str, int] = {}  # hour -> count
        self.last_alerts: Dict[str, datetime] = {}  # metric -> last alert time
    
    def should_send_alert(self, alert: Alert) -> bool:
        """Determine if alert should be sent based on throttling rules."""
        current_time = alert.timestamp
        current_hour = current_time.strftime('%Y-%m-%d %H')
        
        # Check hourly limit
        if self.alert_counts.get(current_hour, 0) >= self.config.max_alerts_per_hour:
            logger.warning(f"Alert throttled: hourly limit reached for {current_hour}")
            return False
        
        # Check metric-specific throttling
        last_alert_time = self.last_alerts.get(alert.metric_name)
        if last_alert_time:
            time_diff = (current_time - last_alert_time).total_seconds() / 60
            if time_diff < self.config.throttle_minutes:
                logger.info(f"Alert throttled: {alert.metric_name} throttled for {time_diff:.1f} minutes")
                return False
        
        # Check for duplicate alerts
        if self._is_duplicate_alert(alert):
            logger.info(f"Alert skipped: duplicate alert for {alert.metric_name}")
            return False
        
        return True
    
    def record_alert(self, alert: Alert) -> None:
        """Record that an alert was sent."""
        current_hour = alert.timestamp.strftime('%Y-%m-%d %H')
        self.alert_counts[current_hour] = self.alert_counts.get(current_hour, 0) + 1
        self.last_alerts[alert.metric_name] = alert.timestamp
        self.alert_history.append(alert)
        
        # Clean up old data
        self._cleanup_old_data()
    
    def _is_duplicate_alert(self, alert: Alert) -> bool:
        """Check if this is a duplicate of a recent alert."""
        cutoff_time = alert.timestamp - timedelta(minutes=self.config.throttle_minutes)
        
        for historical_alert in self.alert_history:
            if (historical_alert.timestamp > cutoff_time and
                historical_alert.metric_name == alert.metric_name and
                historical_alert.severity == alert.severity and
                abs(historical_alert.current_value - alert.current_value) < 0.1):
                return True
        
        return False
    
    def _cleanup_old_data(self) -> None:
        """Remove old alert history data."""
        cutoff_time = datetime.now() - timedelta(days=self.config.alert_history_days)
        
        # Clean alert history
        self.alert_history = [a for a in self.alert_history if a.timestamp > cutoff_time]
        
        # Clean alert counts (keep only recent hours)
        cutoff_hour = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H')
        self.alert_counts = {k: v for k, v in self.alert_counts.items() if k > cutoff_hour}


class PerformanceAlertManager:
    """Main alert manager for performance monitoring."""
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.channels: Dict[str, AlertChannel] = {}
        self.throttler = AlertThrottler(self.config)
        
        self._setup_channels()
    
    def _setup_channels(self) -> None:
        """Setup configured alert channels."""
        if not self.config.enabled:
            logger.info("Performance alerts are disabled")
            return
        
        # Always include console channel
        self.channels['console'] = ConsoleAlertChannel()
        
        # Setup additional channels based on configuration
        for channel_name in self.config.channels:
            if channel_name == 'console':
                continue  # Already added
            
            try:
                if channel_name == 'github':
                    github_config = {
                        'github_token': os.getenv('GITHUB_TOKEN'),
                        'repository': os.getenv('GITHUB_REPOSITORY'),
                        'pr_number': os.getenv('GITHUB_PR_NUMBER')
                    }
                    self.channels['github'] = GitHubAlertChannel(github_config)
                
                elif channel_name == 'slack':
                    slack_config = {
                        'webhook_url': os.getenv('SLACK_WEBHOOK_URL'),
                        'channel': os.getenv('SLACK_CHANNEL', '#alerts')
                    }
                    self.channels['slack'] = SlackAlertChannel(slack_config)
                
                elif channel_name == 'email':
                    email_config = {
                        'smtp_server': os.getenv('SMTP_SERVER'),
                        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                        'username': os.getenv('SMTP_USERNAME'),
                        'password': os.getenv('SMTP_PASSWORD'),
                        'from_email': os.getenv('SMTP_FROM_EMAIL'),
                        'to_emails': os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(',')
                    }
                    self.channels['email'] = EmailAlertChannel(email_config)
                
            except Exception as e:
                logger.error(f"Failed to setup {channel_name} channel: {e}")
    
    async def process_regression_results(self, regression_results: List[Dict[str, Any]], 
                                       run_id: str, environment: Dict[str, Any] = None) -> None:
        """Process regression results and send alerts."""
        alerts_sent = 0
        
        for result in regression_results:
            if not result.get('is_regression'):
                continue
            
            alert = self._create_alert_from_regression(result, run_id, environment or {})
            
            if self.throttler.should_send_alert(alert):
                success = await self._send_alert(alert)
                if success:
                    self.throttler.record_alert(alert)
                    alerts_sent += 1
        
        logger.info(f"Processed {len(regression_results)} regression results, sent {alerts_sent} alerts")
    
    def _create_alert_from_regression(self, regression_result: Dict[str, Any], 
                                    run_id: str, environment: Dict[str, Any]) -> Alert:
        """Create alert from regression result."""
        metric_name = regression_result['metric_name']
        severity = regression_result['severity']
        current_value = regression_result['current_value']
        baseline_value = regression_result['baseline_mean']
        deviation = regression_result['deviation_std']
        
        # Generate alert ID
        alert_id = f"{run_id}_{metric_name}_{int(datetime.now().timestamp())}"
        
        # Create alert title and message
        title = f"Performance regression detected in {metric_name}"
        message = (
            f"The metric '{metric_name}' has regressed significantly. "
            f"Current value ({current_value:.2f}) deviates {deviation:.2f} "
            f"standard deviations from the baseline ({baseline_value:.2f})."
        )
        
        return Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline_value,
            deviation=deviation,
            timestamp=datetime.now(),
            run_id=run_id,
            environment=environment,
            tags=['regression', 'performance'],
            resolved=False
        )
    
    async def _send_alert(self, alert: Alert) -> bool:
        """Send alert through all configured channels."""
        success_count = 0
        total_channels = len(self.channels)
        
        tasks = []
        for channel_name, channel in self.channels.items():
            if channel.enabled:
                task = asyncio.create_task(
                    self._send_to_channel(channel_name, channel, alert)
                )
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result is True)
        
        logger.info(f"Alert {alert.id} sent to {success_count}/{total_channels} channels")
        return success_count > 0
    
    async def _send_to_channel(self, channel_name: str, channel: AlertChannel, alert: Alert) -> bool:
        """Send alert to a specific channel."""
        try:
            return await channel.send_alert(alert)
        except Exception as e:
            logger.error(f"Failed to send alert to {channel_name}: {e}")
            return False
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert system status."""
        return {
            'enabled': self.config.enabled,
            'channels': list(self.channels.keys()),
            'throttle_minutes': self.config.throttle_minutes,
            'max_alerts_per_hour': self.config.max_alerts_per_hour,
            'recent_alerts': len(self.throttler.alert_history),
            'channel_status': {
                name: channel.enabled and channel.validate_config()
                for name, channel in self.channels.items()
            }
        }


# Factory function for easy setup
def create_alert_manager(channels: List[str] = None, 
                        enable_throttling: bool = True) -> PerformanceAlertManager:
    """Create alert manager with sensible defaults."""
    config = AlertConfig(
        enabled=True,
        channels=channels or ['console', 'github'],
        throttle_minutes=30 if enable_throttling else 0,
        max_alerts_per_hour=10
    )
    
    return PerformanceAlertManager(config)


# Example usage
if __name__ == "__main__":
    import os
    
    async def test_alerts():
        """Test alert system."""
        # Create test regression results
        regression_results = [
            {
                'metric_name': 'response_time_mean',
                'current_value': 250.5,
                'baseline_mean': 120.0,
                'deviation_std': 3.2,
                'is_regression': True,
                'severity': 'high'
            }
        ]
        
        # Create alert manager
        alert_manager = create_alert_manager(['console'])
        
        # Process regression results
        await alert_manager.process_regression_results(
            regression_results,
            'test_run_123',
            {'environment': 'test', 'version': '1.0.0'}
        )
        
        # Get status
        status = alert_manager.get_alert_status()
        print(f"Alert system status: {status}")
    
    asyncio.run(test_alerts())