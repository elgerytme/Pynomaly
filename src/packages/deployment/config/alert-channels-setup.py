#!/usr/bin/env python3
"""
Alert Channels Configuration System
Automated setup for Slack, email, and PagerDuty integrations
"""

import json
import logging
import os
import requests
import smtplib
from dataclasses import dataclass, asdict
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class AlertChannel:
    """Alert channel configuration"""
    name: str
    type: str  # slack, email, pagerduty, webhook
    enabled: bool = True
    config: Dict[str, Any] = None
    test_passed: bool = False
    last_test: Optional[datetime] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class AlertChannelSetup:
    """Main alert channel configuration system"""
    
    def __init__(self, config_path: str = "monitoring-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.channels: Dict[str, AlertChannel] = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self._initialize_channels()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load alert configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "alerting": {
                "slack_webhook": os.getenv("SLACK_WEBHOOK_URL", ""),
                "slack_channel": "#production-alerts",
                "email_recipients": ["devops@company.com"],
                "email_smtp_server": "smtp.company.com",
                "email_smtp_port": 587,
                "pagerduty_integration_key": os.getenv("PAGERDUTY_INTEGRATION_KEY", "")
            }
        }
    
    def _initialize_channels(self):
        """Initialize alert channels from configuration"""
        alerting_config = self.config.get("alerting", {})
        
        # Slack channel
        if alerting_config.get("slack_webhook"):
            self.channels["slack"] = AlertChannel(
                name="Slack Notifications",
                type="slack",
                config={
                    "webhook_url": alerting_config["slack_webhook"],
                    "channel": alerting_config.get("slack_channel", "#production-alerts"),
                    "username": alerting_config.get("slack_username", "Production Monitor")
                }
            )
        
        # Email channel
        if alerting_config.get("email_recipients"):
            self.channels["email"] = AlertChannel(
                name="Email Notifications",
                type="email",
                config={
                    "recipients": alerting_config["email_recipients"],
                    "smtp_server": alerting_config.get("email_smtp_server", "localhost"),
                    "smtp_port": alerting_config.get("email_smtp_port", 587),
                    "from_address": alerting_config.get("email_from", "monitoring@company.com"),
                    "username": os.getenv("SMTP_USERNAME", ""),
                    "password": os.getenv("SMTP_PASSWORD", "")
                }
            )
        
        # PagerDuty channel
        if alerting_config.get("pagerduty_integration_key"):
            self.channels["pagerduty"] = AlertChannel(
                name="PagerDuty Alerts",
                type="pagerduty",
                config={
                    "integration_key": alerting_config["pagerduty_integration_key"],
                    "service_id": alerting_config.get("pagerduty_service_id", ""),
                    "api_url": "https://events.pagerduty.com/v2/enqueue"
                }
            )
        
        # Webhook channel (generic)
        if alerting_config.get("webhook_url"):
            self.channels["webhook"] = AlertChannel(
                name="Generic Webhook",
                type="webhook",
                config={
                    "url": alerting_config["webhook_url"],
                    "method": alerting_config.get("webhook_method", "POST"),
                    "headers": alerting_config.get("webhook_headers", {"Content-Type": "application/json"})
                }
            )
    
    def test_slack_channel(self, channel: AlertChannel) -> bool:
        """Test Slack webhook connectivity"""
        try:
            webhook_url = channel.config["webhook_url"]
            
            test_payload = {
                "channel": channel.config.get("channel", "#general"),
                "username": channel.config.get("username", "Test Bot"),
                "text": "🧪 Alert channel test message",
                "attachments": [{
                    "color": "good",
                    "title": "Alert Channel Test",
                    "text": f"This is a test message sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                    "fields": [
                        {"title": "Status", "value": "Testing", "short": True},
                        {"title": "Channel", "value": channel.name, "short": True}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=test_payload, timeout=10)
            success = response.status_code == 200
            
            if success:
                self.logger.info(f"✅ Slack channel test successful: {channel.name}")
            else:
                self.logger.error(f"❌ Slack channel test failed: {response.status_code} - {response.text}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Slack channel test error: {e}")
            return False
    
    def test_email_channel(self, channel: AlertChannel) -> bool:
        """Test email SMTP connectivity"""
        try:
            config = channel.config
            
            # Create test message
            msg = MIMEMultipart()
            msg['From'] = config["from_address"]
            msg['To'] = ", ".join(config["recipients"][:1])  # Test with first recipient only
            msg['Subject'] = "🧪 Alert Channel Test - Production Monitoring"
            
            body = f"""
This is a test message from the production monitoring system.

Test Details:
- Channel: {channel.name}
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
- SMTP Server: {config['smtp_server']}:{config['smtp_port']}

If you received this message, the email alert channel is working correctly.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            
            if config.get("username") and config.get("password"):
                server.login(config["username"], config["password"])
            
            # Send test email
            text = msg.as_string()
            server.sendmail(config["from_address"], config["recipients"][:1], text)
            server.quit()
            
            self.logger.info(f"✅ Email channel test successful: {channel.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Email channel test error: {e}")
            return False
    
    def test_pagerduty_channel(self, channel: AlertChannel) -> bool:
        """Test PagerDuty integration"""
        try:
            config = channel.config
            
            test_payload = {
                "routing_key": config["integration_key"],
                "event_action": "trigger",
                "payload": {
                    "summary": "🧪 Alert Channel Test - Production Monitoring",
                    "source": "production-monitoring-system",
                    "severity": "info",
                    "timestamp": datetime.now().isoformat(),
                    "custom_details": {
                        "test_type": "channel_connectivity",
                        "channel_name": channel.name,
                        "message": "This is a test alert to verify PagerDuty integration"
                    }
                }
            }
            
            response = requests.post(
                config["api_url"],
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            success = response.status_code == 202
            
            if success:
                result = response.json()
                self.logger.info(f"✅ PagerDuty channel test successful: {result.get('dedup_key', 'N/A')}")
            else:
                self.logger.error(f"❌ PagerDuty channel test failed: {response.status_code} - {response.text}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ PagerDuty channel test error: {e}")
            return False
    
    def test_webhook_channel(self, channel: AlertChannel) -> bool:
        """Test generic webhook connectivity"""
        try:
            config = channel.config
            
            test_payload = {
                "event_type": "test",
                "timestamp": datetime.now().isoformat(),
                "message": "Alert channel connectivity test",
                "channel": channel.name,
                "severity": "info",
                "details": {
                    "test_type": "webhook_connectivity",
                    "source": "production-monitoring-system"
                }
            }
            
            method = config.get("method", "POST").upper()
            headers = config.get("headers", {"Content-Type": "application/json"})
            
            if method == "POST":
                response = requests.post(config["url"], json=test_payload, headers=headers, timeout=10)
            elif method == "PUT":
                response = requests.put(config["url"], json=test_payload, headers=headers, timeout=10)
            else:
                response = requests.get(config["url"], headers=headers, timeout=10)
            
            success = 200 <= response.status_code < 300
            
            if success:
                self.logger.info(f"✅ Webhook channel test successful: {response.status_code}")
            else:
                self.logger.error(f"❌ Webhook channel test failed: {response.status_code} - {response.text}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Webhook channel test error: {e}")
            return False
    
    def test_channel(self, channel_name: str) -> bool:
        """Test specific alert channel"""
        if channel_name not in self.channels:
            self.logger.error(f"Channel not found: {channel_name}")
            return False
        
        channel = self.channels[channel_name]
        
        if not channel.enabled:
            self.logger.warning(f"Channel disabled: {channel.name}")
            return False
        
        self.logger.info(f"Testing alert channel: {channel.name}")
        
        # Dispatch to appropriate test method
        if channel.type == "slack":
            success = self.test_slack_channel(channel)
        elif channel.type == "email":
            success = self.test_email_channel(channel)
        elif channel.type == "pagerduty":
            success = self.test_pagerduty_channel(channel)
        elif channel.type == "webhook":
            success = self.test_webhook_channel(channel)
        else:
            self.logger.error(f"Unknown channel type: {channel.type}")
            success = False
        
        # Update channel test status
        channel.test_passed = success
        channel.last_test = datetime.now()
        
        return success
    
    def test_all_channels(self) -> Dict[str, bool]:
        """Test all configured alert channels"""
        self.logger.info("Testing all alert channels...")
        results = {}
        
        for channel_name in self.channels:
            results[channel_name] = self.test_channel(channel_name)
        
        # Summary
        successful_channels = sum(1 for result in results.values() if result)
        total_channels = len(results)
        
        self.logger.info(f"Channel testing completed: {successful_channels}/{total_channels} channels working")
        
        return results
    
    def send_test_alert(self, severity: str = "info", message: str = None) -> Dict[str, bool]:
        """Send test alert through all channels"""
        if message is None:
            message = f"🧪 Test alert sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        test_alert = {
            "id": f"test-{int(datetime.now().timestamp())}",
            "severity": severity,
            "title": "Test Alert",
            "description": message,
            "service": "monitoring-system",
            "metric": "test",
            "current_value": 100.0,
            "threshold": 80.0,
            "timestamp": datetime.now()
        }
        
        results = {}
        
        for channel_name, channel in self.channels.items():
            if not channel.enabled:
                continue
            
            try:
                if channel.type == "slack":
                    success = self._send_slack_alert(channel, test_alert)
                elif channel.type == "email":
                    success = self._send_email_alert(channel, test_alert)
                elif channel.type == "pagerduty":
                    success = self._send_pagerduty_alert(channel, test_alert)
                elif channel.type == "webhook":
                    success = self._send_webhook_alert(channel, test_alert)
                else:
                    success = False
                
                results[channel_name] = success
                
            except Exception as e:
                self.logger.error(f"Failed to send test alert via {channel_name}: {e}")
                results[channel_name] = False
        
        return results
    
    def _send_slack_alert(self, channel: AlertChannel, alert: Dict[str, Any]) -> bool:
        """Send alert to Slack"""
        try:
            color_map = {
                "info": "#36a64f",
                "warning": "#ffeb3b", 
                "critical": "#f44336",
                "emergency": "#9c27b0"
            }
            
            payload = {
                "channel": channel.config.get("channel", "#general"),
                "username": channel.config.get("username", "Production Monitor"),
                "attachments": [{
                    "color": color_map.get(alert["severity"], "#36a64f"),
                    "title": f"🚨 {alert['title']}",
                    "text": alert["description"],
                    "fields": [
                        {"title": "Service", "value": alert["service"], "short": True},
                        {"title": "Metric", "value": alert["metric"], "short": True},
                        {"title": "Current Value", "value": str(alert["current_value"]), "short": True},
                        {"title": "Threshold", "value": str(alert["threshold"]), "short": True}
                    ],
                    "footer": "Production Monitoring System"
                }]
            }
            
            response = requests.post(channel.config["webhook_url"], json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Slack alert failed: {e}")
            return False
    
    def _send_email_alert(self, channel: AlertChannel, alert: Dict[str, Any]) -> bool:
        """Send alert via email"""
        try:
            config = channel.config
            
            msg = MIMEMultipart()
            msg['From'] = config["from_address"]
            msg['To'] = ", ".join(config["recipients"])
            msg['Subject'] = f"🚨 {alert['severity'].upper()}: {alert['title']}"
            
            body = f"""
Production Alert Notification

Alert Details:
- ID: {alert['id']}
- Severity: {alert['severity'].upper()}
- Service: {alert['service']}
- Metric: {alert['metric']}
- Description: {alert['description']}
- Current Value: {alert['current_value']}
- Threshold: {alert['threshold']}
- Timestamp: {alert['timestamp']}

This alert was generated by the production monitoring system.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            
            if config.get("username") and config.get("password"):
                server.login(config["username"], config["password"])
            
            text = msg.as_string()
            server.sendmail(config["from_address"], config["recipients"], text)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
            return False
    
    def _send_pagerduty_alert(self, channel: AlertChannel, alert: Dict[str, Any]) -> bool:
        """Send alert to PagerDuty"""
        try:
            config = channel.config
            
            payload = {
                "routing_key": config["integration_key"],
                "event_action": "trigger",
                "dedup_key": alert["id"],
                "payload": {
                    "summary": f"{alert['severity'].upper()}: {alert['title']}",
                    "source": alert["service"],
                    "severity": alert["severity"],
                    "timestamp": alert["timestamp"].isoformat(),
                    "custom_details": {
                        "metric": alert["metric"],
                        "current_value": alert["current_value"],
                        "threshold": alert["threshold"],
                        "description": alert["description"]
                    }
                }
            }
            
            response = requests.post(
                config["api_url"],
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            return response.status_code == 202
            
        except Exception as e:
            self.logger.error(f"PagerDuty alert failed: {e}")
            return False
    
    def _send_webhook_alert(self, channel: AlertChannel, alert: Dict[str, Any]) -> bool:
        """Send alert to generic webhook"""
        try:
            config = channel.config
            
            payload = {
                "event_type": "alert",
                "alert": alert,
                "timestamp": alert["timestamp"].isoformat()
            }
            
            method = config.get("method", "POST").upper()
            headers = config.get("headers", {"Content-Type": "application/json"})
            
            if method == "POST":
                response = requests.post(config["url"], json=payload, headers=headers, timeout=10)
            else:
                response = requests.put(config["url"], json=payload, headers=headers, timeout=10)
            
            return 200 <= response.status_code < 300
            
        except Exception as e:
            self.logger.error(f"Webhook alert failed: {e}")
            return False
    
    def generate_channel_status_report(self) -> str:
        """Generate alert channel status report"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("ALERT CHANNELS STATUS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        if not self.channels:
            report_lines.append("❌ No alert channels configured")
            return "\n".join(report_lines)
        
        # Summary
        enabled_channels = sum(1 for c in self.channels.values() if c.enabled)
        tested_channels = sum(1 for c in self.channels.values() if c.last_test is not None)
        working_channels = sum(1 for c in self.channels.values() if c.test_passed)
        
        report_lines.append(f"Total Channels: {len(self.channels)}")
        report_lines.append(f"Enabled: {enabled_channels}")
        report_lines.append(f"Tested: {tested_channels}")
        report_lines.append(f"Working: {working_channels}")
        report_lines.append("")
        
        # Individual channel status
        for channel_name, channel in self.channels.items():
            status_symbol = "✅" if channel.test_passed else "❌" if channel.last_test else "⏸️"
            enabled_text = "Enabled" if channel.enabled else "Disabled"
            
            report_lines.append(f"{status_symbol} {channel.name} ({channel.type})")
            report_lines.append(f"   Status: {enabled_text}")
            
            if channel.last_test:
                report_lines.append(f"   Last Test: {channel.last_test.strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"   Test Result: {'✅ PASSED' if channel.test_passed else '❌ FAILED'}")
            else:
                report_lines.append(f"   Last Test: Never")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        if working_channels == 0:
            report_lines.append("🚨 No working alert channels - alerts will not be delivered!")
        elif working_channels < enabled_channels:
            report_lines.append("⚠️  Some alert channels are not working - check configuration")
        else:
            report_lines.append("✅ All channels are working correctly")
        
        return "\n".join(report_lines)
    
    def setup_environment_variables(self) -> Dict[str, str]:
        """Generate environment variables setup guide"""
        env_vars = {
            "SLACK_WEBHOOK_URL": "Your Slack webhook URL for alerts",
            "SMTP_USERNAME": "SMTP server username for email alerts",
            "SMTP_PASSWORD": "SMTP server password for email alerts", 
            "PAGERDUTY_INTEGRATION_KEY": "PagerDuty integration key for incident alerts",
            "GRAFANA_API_KEY": "Grafana API key for dashboard creation"
        }
        
        return env_vars


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alert Channels Setup and Testing")
    parser.add_argument("--config", default="monitoring-config.yaml", help="Configuration file")
    parser.add_argument("--test", help="Test specific channel")
    parser.add_argument("--test-all", action="store_true", help="Test all channels")
    parser.add_argument("--send-test-alert", action="store_true", help="Send test alert")
    parser.add_argument("--status", action="store_true", help="Show channel status")
    parser.add_argument("--env-vars", action="store_true", help="Show required environment variables")
    args = parser.parse_args()
    
    setup = AlertChannelSetup(args.config)
    
    if args.env_vars:
        env_vars = setup.setup_environment_variables()
        print("Required Environment Variables:")
        print("=" * 40)
        for var, description in env_vars.items():
            print(f"{var:<25} {description}")
        return
    
    if args.status:
        report = setup.generate_channel_status_report()
        print(report)
        return
    
    if args.test:
        success = setup.test_channel(args.test)
        exit(0 if success else 1)
    
    if args.test_all:
        results = setup.test_all_channels()
        failed_channels = [name for name, success in results.items() if not success]
        if failed_channels:
            print(f"❌ Failed channels: {', '.join(failed_channels)}")
            exit(1)
        else:
            print("✅ All channels tested successfully")
            exit(0)
    
    if args.send_test_alert:
        results = setup.send_test_alert()
        failed_channels = [name for name, success in results.items() if not success]
        if failed_channels:
            print(f"❌ Failed to send test alert via: {', '.join(failed_channels)}")
        else:
            print("✅ Test alert sent successfully via all channels")
    
    # Default: show status
    report = setup.generate_channel_status_report()
    print(report)


if __name__ == "__main__":
    main()