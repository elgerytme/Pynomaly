#!/usr/bin/env python3
"""
Demo script for Pynomaly Real-time Alerting System.

This script demonstrates the capabilities of the alerting system including:
- Creating alert rules
- Submitting metrics
- Triggering alerts
- Real-time notifications
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import requests
import websockets
from websockets.exceptions import ConnectionClosed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
CLIENT_ID = "demo_client_001"

class AlertingDemo:
    """Demo class for the alerting system."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Pynomaly-Alerting-Demo/1.0'
        })
    
    def create_demo_rules(self) -> Dict[str, Any]:
        """Create demo alert rules."""
        logger.info("Creating demo alert rules...")
        
        rules = [
            {
                "name": "High CPU Usage",
                "description": "Alert when CPU usage exceeds 80%",
                "metric_name": "system.cpu.usage",
                "condition": ">",
                "threshold": "80.0",
                "duration": 60,
                "severity": "high",
                "enabled": True,
                "notification_channels": ["email", "slack"],
                "cooldown_period": 300,
                "notification_template": "CPU usage is at {value}% on {host}"
            },
            {
                "name": "High Memory Usage",
                "description": "Alert when memory usage exceeds 90%",
                "metric_name": "system.memory.usage",
                "condition": ">",
                "threshold": "90.0",
                "duration": 120,
                "severity": "critical",
                "enabled": True,
                "notification_channels": ["email", "slack", "pagerduty"],
                "cooldown_period": 300,
                "notification_template": "Memory usage is at {value}% on {host}"
            },
            {
                "name": "Disk Space Low",
                "description": "Alert when disk usage exceeds 85%",
                "metric_name": "system.disk.usage",
                "condition": ">",
                "threshold": "85.0",
                "duration": 300,
                "severity": "medium",
                "enabled": True,
                "notification_channels": ["email"],
                "cooldown_period": 600,
                "notification_template": "Disk usage is at {value}% on {host}"
            },
            {
                "name": "Anomaly Detection Rate",
                "description": "Alert when anomaly detection rate is unusual",
                "metric_name": "pynomaly.anomaly_rate",
                "condition": ">",
                "threshold": "0.1",
                "duration": 180,
                "severity": "medium",
                "enabled": True,
                "notification_channels": ["email", "slack"],
                "cooldown_period": 900,
                "notification_template": "Anomaly detection rate is {value} on {detector}"
            }
        ]
        
        created_rules = []
        for rule in rules:
            try:
                response = self.session.post(
                    f"{self.base_url}/alerting/rules",
                    json=rule
                )
                
                if response.status_code == 200:
                    created_rule = response.json()
                    created_rules.append(created_rule)
                    logger.info(f"Created rule: {created_rule['name']} (ID: {created_rule['id']})")
                else:
                    logger.error(f"Failed to create rule {rule['name']}: {response.text}")
                    
            except Exception as e:
                logger.error(f"Error creating rule {rule['name']}: {e}")
        
        return {"created_rules": created_rules, "total": len(created_rules)}
    
    def submit_demo_metrics(self) -> Dict[str, Any]:
        """Submit demo metrics to trigger alerts."""
        logger.info("Submitting demo metrics...")
        
        # Generate realistic metric values
        metrics = [
            {
                "metric_name": "system.cpu.usage",
                "value": 85.5,  # Above threshold
                "metadata": {"host": "web-server-01", "datacenter": "us-east-1"}
            },
            {
                "metric_name": "system.memory.usage",
                "value": 92.3,  # Above threshold
                "metadata": {"host": "web-server-01", "datacenter": "us-east-1"}
            },
            {
                "metric_name": "system.disk.usage",
                "value": 75.2,  # Below threshold
                "metadata": {"host": "web-server-01", "datacenter": "us-east-1"}
            },
            {
                "metric_name": "pynomaly.anomaly_rate",
                "value": 0.15,  # Above threshold
                "metadata": {"detector": "isolation-forest", "dataset": "production"}
            }
        ]
        
        # Submit metrics individually
        submitted_metrics = []
        for metric in metrics:
            try:
                response = self.session.post(
                    f"{self.base_url}/alerting/metrics",
                    json=metric
                )
                
                if response.status_code == 200:
                    submitted_metrics.append(metric)
                    logger.info(f"Submitted metric: {metric['metric_name']} = {metric['value']}")
                else:
                    logger.error(f"Failed to submit metric {metric['metric_name']}: {response.text}")
                    
            except Exception as e:
                logger.error(f"Error submitting metric {metric['metric_name']}: {e}")
        
        # Submit batch of metrics
        try:
            response = self.session.post(
                f"{self.base_url}/alerting/metrics/batch",
                json=metrics
            )
            
            if response.status_code == 200:
                logger.info("Submitted batch of metrics successfully")
            else:
                logger.error(f"Failed to submit batch metrics: {response.text}")
                
        except Exception as e:
            logger.error(f"Error submitting batch metrics: {e}")
        
        return {"submitted_metrics": submitted_metrics, "total": len(submitted_metrics)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get alerting system status."""
        logger.info("Getting system status...")
        
        try:
            response = self.session.get(f"{self.base_url}/alerting/status")
            
            if response.status_code == 200:
                status = response.json()
                logger.info(f"System status: {status['status']}")
                logger.info(f"Active alerts: {status['active_alerts']}")
                logger.info(f"Total rules: {status['total_rules']}")
                logger.info(f"Enabled rules: {status['enabled_rules']}")
                return status
            else:
                logger.error(f"Failed to get system status: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    def get_active_alerts(self) -> Dict[str, Any]:
        """Get active alerts."""
        logger.info("Getting active alerts...")
        
        try:
            response = self.session.get(f"{self.base_url}/alerting/alerts")
            
            if response.status_code == 200:
                alerts = response.json()
                logger.info(f"Found {len(alerts)} active alerts")
                
                for alert in alerts:
                    logger.info(f"Alert: {alert['rule_name']} - {alert['severity']} - {alert['status']}")
                
                return {"alerts": alerts, "total": len(alerts)}
            else:
                logger.error(f"Failed to get active alerts: {response.text}")
                return {"alerts": [], "total": 0}
                
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return {"alerts": [], "total": 0}
    
    def trigger_demo_alert(self) -> Dict[str, Any]:
        """Trigger a demo alert."""
        logger.info("Triggering demo alert...")
        
        try:
            response = self.session.post(
                f"{self.base_url}/alerting/demo/trigger-alert",
                params={
                    "metric_name": "demo.cpu.usage",
                    "value": 95.0
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Demo alert triggered: {result['message']}")
                return result
            else:
                logger.error(f"Failed to trigger demo alert: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error triggering demo alert: {e}")
            return {}
    
    async def websocket_demo(self) -> None:
        """Demonstrate WebSocket real-time alerts."""
        logger.info("Starting WebSocket demo...")
        
        ws_url = f"{WS_URL}/alerting/ws/{CLIENT_ID}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"Connected to WebSocket: {ws_url}")
                
                # Listen for messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if data.get('type') == 'initial_alerts':
                            logger.info(f"Received initial alerts: {len(data.get('alerts', []))}")
                            
                        elif data.get('type') == 'alert_update':
                            logger.info(f"Received alert update: {len(data.get('alerts', []))} alerts")
                            
                        elif data.get('type') == 'new_alert':
                            alert = data.get('alert', {})
                            logger.info(f"NEW ALERT: {alert.get('rule_name')} - {alert.get('severity')}")
                            
                        # Send ping periodically
                        await websocket.send("ping")
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON message: {message}")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                        
        except ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    def run_demo(self) -> None:
        """Run the complete alerting demo."""
        logger.info("Starting Pynomaly Alerting System Demo")
        logger.info("=" * 50)
        
        try:
            # 1. Check system health
            logger.info("Step 1: Checking system health...")
            try:
                response = self.session.get(f"{self.base_url}/alerting/health")
                if response.status_code == 200:
                    health = response.json()
                    logger.info(f"System health: {health}")
                else:
                    logger.error(f"Health check failed: {response.text}")
                    return
            except Exception as e:
                logger.error(f"Could not connect to alerting service: {e}")
                logger.info("Make sure the Pynomaly server is running on http://localhost:8000")
                return
            
            # 2. Create demo rules
            logger.info("\nStep 2: Creating demo alert rules...")
            rule_result = self.create_demo_rules()
            logger.info(f"Created {rule_result['total']} alert rules")
            
            # 3. Get system status
            logger.info("\nStep 3: Getting system status...")
            status = self.get_system_status()
            
            # 4. Submit demo metrics
            logger.info("\nStep 4: Submitting demo metrics...")
            metric_result = self.submit_demo_metrics()
            logger.info(f"Submitted {metric_result['total']} metrics")
            
            # 5. Wait for alert processing
            logger.info("\nStep 5: Waiting for alert processing...")
            time.sleep(5)
            
            # 6. Check for active alerts
            logger.info("\nStep 6: Checking for active alerts...")
            alert_result = self.get_active_alerts()
            logger.info(f"Found {alert_result['total']} active alerts")
            
            # 7. Trigger demo alert
            logger.info("\nStep 7: Triggering demo alert...")
            self.trigger_demo_alert()
            
            # 8. WebSocket demo
            logger.info("\nStep 8: WebSocket demo (5 seconds)...")
            asyncio.get_event_loop().run_until_complete(
                asyncio.wait_for(self.websocket_demo(), timeout=5)
            )
            
        except asyncio.TimeoutError:
            logger.info("WebSocket demo timeout (expected)")
        except Exception as e:
            logger.error(f"Demo error: {e}")
        finally:
            logger.info("\nDemo completed!")
            logger.info("=" * 50)


def main():
    """Main entry point."""
    demo = AlertingDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()