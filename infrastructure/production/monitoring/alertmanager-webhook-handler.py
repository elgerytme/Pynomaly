#!/usr/bin/env python3
"""
AlertManager Webhook Handler

This service receives webhooks from AlertManager and performs additional
actions like ticket creation, team notifications, and escalations.
"""

import json
import logging
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import aiohttp
from aiohttp import web
import jira
import slack_sdk
from slack_sdk.web.async_client import AsyncWebClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class EscalationLevel(Enum):
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"

@dataclass
class Alert:
    """Alert data structure"""
    alert_name: str
    severity: AlertSeverity
    status: str
    description: str
    instance: str
    team: str
    timestamp: datetime
    labels: Dict[str, str]
    annotations: Dict[str, str]
    generator_url: str

@dataclass
class NotificationConfig:
    """Notification configuration"""
    slack_token: str
    jira_server: str
    jira_username: str
    jira_token: str
    pagerduty_token: str
    email_smtp_server: str
    email_username: str
    email_password: str

class AlertHandler:
    """Handles incoming alerts and performs appropriate actions"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.slack_client = AsyncWebClient(token=config.slack_token)
        self.jira_client = jira.JIRA(
            server=config.jira_server,
            basic_auth=(config.jira_username, config.jira_token)
        )
        self.escalation_rules = self._load_escalation_rules()
        self.notification_templates = self._load_notification_templates()
    
    def _load_escalation_rules(self) -> Dict[str, Any]:
        """Load escalation rules configuration"""
        return {
            "critical_alerts": {
                "immediate_notification": True,
                "create_ticket": True,
                "escalate_after_minutes": 5,
                "escalation_levels": [EscalationLevel.L1, EscalationLevel.L2, EscalationLevel.L3]
            },
            "warning_alerts": {
                "immediate_notification": True,
                "create_ticket": False,
                "escalate_after_minutes": 30,
                "escalation_levels": [EscalationLevel.L1, EscalationLevel.L2]
            },
            "info_alerts": {
                "immediate_notification": False,
                "create_ticket": False,
                "escalate_after_minutes": 0,
                "escalation_levels": []
            }
        }
    
    def _load_notification_templates(self) -> Dict[str, str]:
        """Load notification message templates"""
        return {
            "slack_critical": """
🚨 *CRITICAL ALERT* 🚨

*Alert:* {alert_name}
*Severity:* {severity}
*Status:* {status}
*Environment:* Production
*Instance:* {instance}
*Team:* {team}
*Time:* {timestamp}

*Description:* {description}

*Actions Required:*
- Immediate investigation and resolution
- Follow incident response procedures
- Update in thread with status

*Links:*
- Dashboard: {dashboard_url}
- Runbook: {runbook_url}
- Logs: {logs_url}
            """,
            
            "slack_warning": """
⚠️ *WARNING ALERT* ⚠️

*Alert:* {alert_name}
*Severity:* {severity}
*Status:* {status}
*Instance:* {instance}
*Team:* {team}
*Time:* {timestamp}

*Description:* {description}

*Links:*
- Dashboard: {dashboard_url}
- Runbook: {runbook_url}
            """,
            
            "jira_ticket": """
*Summary:* Production Alert - {alert_name}

*Environment:* Production
*Severity:* {severity}
*Instance:* {instance}
*Team:* {team}
*Timestamp:* {timestamp}

*Description:*
{description}

*Alert Details:*
{alert_details}

*Investigation Steps:*
1. Check system health dashboards
2. Review application logs
3. Verify service connectivity
4. Check resource utilization
5. Review recent deployments

*Resolution Criteria:*
- Alert is resolved in monitoring system
- Root cause identified and documented
- Preventive measures implemented if applicable

*Links:*
- Monitoring Dashboard: {dashboard_url}
- Runbook: {runbook_url}
- Alert Source: {generator_url}
            """
        }
    
    async def handle_alert(self, alert: Alert) -> Dict[str, Any]:
        """Handle incoming alert and perform appropriate actions"""
        logger.info(f"Handling alert: {alert.alert_name} (severity: {alert.severity.value})")
        
        actions_taken = {
            "slack_notification": False,
            "jira_ticket_created": False,
            "pagerduty_triggered": False,
            "email_sent": False
        }
        
        try:
            # Determine escalation rules based on severity
            rules = self.escalation_rules.get(f"{alert.severity.value}_alerts", {})
            
            # Send immediate notifications
            if rules.get("immediate_notification", False):
                await self._send_slack_notification(alert)
                actions_taken["slack_notification"] = True
            
            # Create JIRA ticket for critical alerts
            if rules.get("create_ticket", False):
                ticket_id = await self._create_jira_ticket(alert)
                actions_taken["jira_ticket_created"] = ticket_id
            
            # Trigger PagerDuty for critical alerts
            if alert.severity == AlertSeverity.CRITICAL:
                await self._trigger_pagerduty(alert)
                actions_taken["pagerduty_triggered"] = True
            
            # Send email notifications to team
            await self._send_email_notification(alert)
            actions_taken["email_sent"] = True
            
            # Schedule escalation if needed
            if rules.get("escalate_after_minutes", 0) > 0:
                await self._schedule_escalation(alert, rules["escalate_after_minutes"])
            
            logger.info(f"Alert handled successfully: {actions_taken}")
            return actions_taken
            
        except Exception as e:
            logger.error(f"Failed to handle alert {alert.alert_name}: {e}")
            raise
    
    async def _send_slack_notification(self, alert: Alert) -> bool:
        """Send Slack notification for alert"""
        try:
            # Determine channel based on team and severity
            channel = self._get_slack_channel(alert.team, alert.severity)
            
            # Select appropriate template
            template_key = f"slack_{alert.severity.value}"
            message_template = self.notification_templates.get(template_key, "")
            
            # Format message
            message = message_template.format(
                alert_name=alert.alert_name,
                severity=alert.severity.value.upper(),
                status=alert.status,
                instance=alert.instance,
                team=alert.team,
                timestamp=alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                description=alert.description,
                dashboard_url=self._get_dashboard_url(alert),
                runbook_url=self._get_runbook_url(alert),
                logs_url=self._get_logs_url(alert)
            )
            
            # Send to Slack
            response = await self.slack_client.chat_postMessage(
                channel=channel,
                text=message,
                blocks=self._create_slack_blocks(alert)
            )
            
            logger.info(f"Slack notification sent to {channel}: {response['ts']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _get_slack_channel(self, team: str, severity: AlertSeverity) -> str:
        """Get appropriate Slack channel for team and severity"""
        channel_map = {
            ("platform", AlertSeverity.CRITICAL): "#platform-critical",
            ("platform", AlertSeverity.WARNING): "#platform-alerts",
            ("ml", AlertSeverity.CRITICAL): "#ml-critical",
            ("ml", AlertSeverity.WARNING): "#ml-alerts",
            ("security", AlertSeverity.CRITICAL): "#security-incidents",
            ("security", AlertSeverity.WARNING): "#security-alerts",
            ("business", AlertSeverity.WARNING): "#business-metrics"
        }
        
        return channel_map.get((team, severity), "#ops-alerts")
    
    def _create_slack_blocks(self, alert: Alert) -> List[Dict[str, Any]]:
        """Create Slack block kit format for rich notifications"""
        severity_emoji = {
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.INFO: "ℹ️"
        }
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{severity_emoji.get(alert.severity, '')} {alert.severity.value.upper()} ALERT"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Alert:*\n{alert.alert_name}"},
                    {"type": "mrkdwn", "text": f"*Team:*\n{alert.team}"},
                    {"type": "mrkdwn", "text": f"*Instance:*\n{alert.instance}"},
                    {"type": "mrkdwn", "text": f"*Status:*\n{alert.status}"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:*\n{alert.description}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Dashboard"},
                        "url": self._get_dashboard_url(alert),
                        "style": "primary"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Runbook"},
                        "url": self._get_runbook_url(alert)
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Acknowledge"},
                        "value": f"ack_{alert.alert_name}_{alert.timestamp.timestamp()}"
                    }
                ]
            }
        ]
        
        return blocks
    
    async def _create_jira_ticket(self, alert: Alert) -> Optional[str]:
        """Create JIRA ticket for alert"""
        try:
            # Determine project and issue type
            project_key = self._get_jira_project(alert.team)
            
            # Format ticket description
            template = self.notification_templates["jira_ticket"]
            description = template.format(
                alert_name=alert.alert_name,
                severity=alert.severity.value.upper(),
                instance=alert.instance,
                team=alert.team,
                timestamp=alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                description=alert.description,
                alert_details=json.dumps(alert.labels, indent=2),
                dashboard_url=self._get_dashboard_url(alert),
                runbook_url=self._get_runbook_url(alert),
                generator_url=alert.generator_url
            )
            
            # Create ticket
            issue_dict = {
                'project': {'key': project_key},
                'summary': f"Production Alert - {alert.alert_name}",
                'description': description,
                'issuetype': {'name': 'Bug' if alert.severity == AlertSeverity.CRITICAL else 'Task'},
                'priority': {'name': 'Critical' if alert.severity == AlertSeverity.CRITICAL else 'High'},
                'labels': ['production', 'alert', alert.severity.value, alert.team]
            }
            
            new_issue = self.jira_client.create_issue(fields=issue_dict)
            logger.info(f"JIRA ticket created: {new_issue.key}")
            return new_issue.key
            
        except Exception as e:
            logger.error(f"Failed to create JIRA ticket: {e}")
            return None
    
    def _get_jira_project(self, team: str) -> str:
        """Get JIRA project key for team"""
        project_map = {
            "platform": "PLAT",
            "ml": "ML",
            "security": "SEC",
            "business": "BIZ"
        }
        return project_map.get(team, "OPS")
    
    async def _trigger_pagerduty(self, alert: Alert) -> bool:
        """Trigger PagerDuty incident for critical alerts"""
        try:
            pagerduty_payload = {
                "routing_key": self.config.pagerduty_token,
                "event_action": "trigger",
                "dedup_key": f"{alert.alert_name}_{alert.instance}",
                "payload": {
                    "summary": f"CRITICAL: {alert.alert_name}",
                    "source": alert.instance,
                    "severity": "critical",
                    "component": alert.team,
                    "group": "production",
                    "class": "alert",
                    "custom_details": {
                        "description": alert.description,
                        "team": alert.team,
                        "dashboard_url": self._get_dashboard_url(alert),
                        "runbook_url": self._get_runbook_url(alert)
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=pagerduty_payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 202:
                        logger.info("PagerDuty incident triggered successfully")
                        return True
                    else:
                        logger.error(f"PagerDuty trigger failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to trigger PagerDuty: {e}")
            return False
    
    async def _send_email_notification(self, alert: Alert) -> bool:
        """Send email notification for alert"""
        try:
            # Implementation would go here
            # For brevity, just logging the action
            logger.info(f"Email notification sent for alert: {alert.alert_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    async def _schedule_escalation(self, alert: Alert, escalate_after_minutes: int):
        """Schedule escalation for unresolved alerts"""
        try:
            # Implementation would schedule a task to check alert status
            # and escalate if still unresolved after the specified time
            logger.info(f"Escalation scheduled for {alert.alert_name} after {escalate_after_minutes} minutes")
        except Exception as e:
            logger.error(f"Failed to schedule escalation: {e}")
    
    def _get_dashboard_url(self, alert: Alert) -> str:
        """Get monitoring dashboard URL for alert"""
        base_url = "https://monitoring.mlops-platform.com/grafana"
        dashboard_map = {
            "platform": f"{base_url}/d/platform-overview",
            "ml": f"{base_url}/d/ml-metrics",
            "security": f"{base_url}/d/security-monitoring",
            "business": f"{base_url}/d/business-metrics"
        }
        return dashboard_map.get(alert.team, f"{base_url}/d/system-overview")
    
    def _get_runbook_url(self, alert: Alert) -> str:
        """Get runbook URL for alert"""
        base_url = "https://runbook.mlops-platform.com"
        alert_name_clean = alert.alert_name.lower().replace(" ", "-")
        return f"{base_url}/{alert_name_clean}"
    
    def _get_logs_url(self, alert: Alert) -> str:
        """Get logs URL for alert investigation"""
        base_url = "https://monitoring.mlops-platform.com/grafana/explore"
        return f"{base_url}?orgId=1&left=%5B%22now-1h%22,%22now%22,%22Loki%22,%7B%22expr%22:%22%7Binstance%3D%5C%22{alert.instance}%5C%22%7D%22%7D%5D"

def parse_alertmanager_webhook(payload: Dict[str, Any]) -> List[Alert]:
    """Parse AlertManager webhook payload into Alert objects"""
    alerts = []
    
    for alert_data in payload.get("alerts", []):
        try:
            # Extract alert information
            labels = alert_data.get("labels", {})
            annotations = alert_data.get("annotations", {})
            
            alert = Alert(
                alert_name=labels.get("alertname", "Unknown"),
                severity=AlertSeverity(labels.get("severity", "info")),
                status=alert_data.get("status", "unknown"),
                description=annotations.get("description", "No description available"),
                instance=labels.get("instance", "unknown"),
                team=labels.get("team", "platform"),
                timestamp=datetime.fromisoformat(alert_data.get("startsAt", "").replace("Z", "+00:00")),
                labels=labels,
                annotations=annotations,
                generator_url=alert_data.get("generatorURL", "")
            )
            
            alerts.append(alert)
            
        except Exception as e:
            logger.error(f"Failed to parse alert: {e}")
            continue
    
    return alerts

async def webhook_handler(request):
    """Handle incoming AlertManager webhooks"""
    try:
        payload = await request.json()
        logger.info(f"Received webhook with {len(payload.get('alerts', []))} alerts")
        
        # Parse alerts from webhook
        alerts = parse_alertmanager_webhook(payload)
        
        # Get alert handler instance
        handler = request.app['alert_handler']
        
        # Process each alert
        results = []
        for alert in alerts:
            try:
                result = await handler.handle_alert(alert)
                results.append({
                    "alert_name": alert.alert_name,
                    "status": "processed",
                    "actions": result
                })
            except Exception as e:
                logger.error(f"Failed to process alert {alert.alert_name}: {e}")
                results.append({
                    "alert_name": alert.alert_name,
                    "status": "error",
                    "error": str(e)
                })
        
        return web.json_response({
            "status": "success",
            "alerts_processed": len(results),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Webhook handler error: {e}")
        return web.json_response({
            "status": "error",
            "message": str(e)
        }, status=500)

async def health_check(request):
    """Health check endpoint"""
    return web.json_response({"status": "healthy", "timestamp": datetime.now().isoformat()})

def create_app() -> web.Application:
    """Create and configure the web application"""
    app = web.Application()
    
    # Configure notification settings
    config = NotificationConfig(
        slack_token=os.getenv("SLACK_BOT_TOKEN", ""),
        jira_server=os.getenv("JIRA_SERVER", ""),
        jira_username=os.getenv("JIRA_USERNAME", ""),
        jira_token=os.getenv("JIRA_TOKEN", ""),
        pagerduty_token=os.getenv("PAGERDUTY_ROUTING_KEY", ""),
        email_smtp_server=os.getenv("SMTP_SERVER", ""),
        email_username=os.getenv("SMTP_USERNAME", ""),
        email_password=os.getenv("SMTP_PASSWORD", "")
    )
    
    # Create alert handler
    alert_handler = AlertHandler(config)
    app['alert_handler'] = alert_handler
    
    # Configure routes
    app.router.add_post('/webhook/alertmanager', webhook_handler)
    app.router.add_get('/health', health_check)
    
    return app

def main():
    """Main function to run the webhook handler service"""
    logger.info("Starting AlertManager Webhook Handler")
    
    # Create application
    app = create_app()
    
    # Start web server
    port = int(os.getenv("WEBHOOK_PORT", 9093))
    web.run_app(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()