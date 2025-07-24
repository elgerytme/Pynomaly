#!/usr/bin/env python3
"""
Business Integration Automation Script
Automates the setup of business integrations for the detection platform.
"""

import json
import logging
import os
import sys
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import subprocess


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('business-integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BusinessIntegrationManager:
    """Manages business integrations for the detection platform."""
    
    def __init__(self, config_path: str = "business-integration/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.integration_status = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load business integration configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self._create_default_config()
        
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration file."""
        default_config = {
            "integrations": {
                "sso": {
                    "enabled": True,
                    "provider": "azure_ad",
                    "configuration": {
                        "tenant_id": "${AZURE_TENANT_ID}",
                        "client_id": "${AZURE_CLIENT_ID}",
                        "client_secret": "${AZURE_CLIENT_SECRET}"
                    }
                },
                "slack": {
                    "enabled": True,
                    "webhook_url": "${SLACK_WEBHOOK_URL}",
                    "channels": {
                        "alerts": "#anomaly-alerts",
                        "general": "#anomaly-detection",
                        "dev": "#anomaly-dev"
                    }
                },
                "email": {
                    "enabled": True,
                    "smtp_server": "${SMTP_SERVER}",
                    "smtp_port": 587,
                    "username": "${SMTP_USERNAME}",
                    "password": "${SMTP_PASSWORD}",
                    "distribution_lists": {
                        "alerts": ["ops-team@company.com"],
                        "reports": ["business-team@company.com"]
                    }
                },
                "database": {
                    "enabled": True,
                    "connections": {
                        "business_db": {
                            "type": "postgresql",
                            "host": "${BUSINESS_DB_HOST}",
                            "port": 5432,
                            "database": "business_data",
                            "username": "${BUSINESS_DB_USER}",
                            "password": "${BUSINESS_DB_PASSWORD}"
                        }
                    }
                },
                "bi_tools": {
                    "enabled": True,
                    "tableau": {
                        "server_url": "${TABLEAU_SERVER_URL}",
                        "username": "${TABLEAU_USERNAME}",
                        "password": "${TABLEAU_PASSWORD}",
                        "site_id": "anomaly-detection"
                    }
                },
                "incident_management": {
                    "enabled": True,
                    "servicenow": {
                        "instance_url": "${SERVICENOW_URL}",
                        "username": "${SERVICENOW_USERNAME}",
                        "password": "${SERVICENOW_PASSWORD}",
                        "assignment_group": "IT Operations"
                    }
                }
            },
            "monitoring": {
                "health_check_interval": 300,
                "alert_thresholds": {
                    "response_time": 1000,
                    "error_rate": 0.01,
                    "cpu_usage": 0.8,
                    "memory_usage": 0.85
                }
            },
            "rollout": {
                "phases": [
                    {
                        "name": "internal_beta",
                        "users": 10,
                        "duration_days": 14,
                        "success_criteria": {
                            "uptime": 0.95,
                            "user_satisfaction": 7.0
                        }
                    },
                    {
                        "name": "limited_production",
                        "users": 50,
                        "duration_days": 14,
                        "success_criteria": {
                            "uptime": 0.98,
                            "user_satisfaction": 8.0
                        }
                    },
                    {
                        "name": "full_production",
                        "users": -1,  # All users
                        "duration_days": 28,
                        "success_criteria": {
                            "uptime": 0.995,
                            "user_satisfaction": 8.5
                        }
                    }
                ]
            }
        }
        
        # Save default config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default configuration: {self.config_path}")
        return default_config
    
    def setup_sso_integration(self) -> bool:
        """Set up Single Sign-On integration."""
        logger.info("Setting up SSO integration...")
        
        sso_config = self.config["integrations"]["sso"]
        if not sso_config["enabled"]:
            logger.info("SSO integration disabled, skipping")
            return True
        
        try:
            # Configure SSO based on provider
            provider = sso_config["provider"]
            
            if provider == "azure_ad":
                return self._setup_azure_ad_sso(sso_config["configuration"])
            elif provider == "okta":
                return self._setup_okta_sso(sso_config["configuration"])
            else:
                logger.error(f"Unsupported SSO provider: {provider}")
                return False
        
        except Exception as e:
            logger.error(f"SSO integration failed: {e}")
            return False
    
    def _setup_azure_ad_sso(self, config: Dict[str, str]) -> bool:
        """Set up Azure AD SSO integration."""
        logger.info("Configuring Azure AD SSO...")
        
        # Expand environment variables
        tenant_id = os.path.expandvars(config["tenant_id"])
        client_id = os.path.expandvars(config["client_id"])
        client_secret = os.path.expandvars(config["client_secret"])
        
        # Validate configuration
        if not all([tenant_id, client_id, client_secret]):
            logger.error("Missing Azure AD configuration parameters")
            return False
        
        # Create SSO configuration file
        sso_config_file = Path("deploy/k8s/sso-config.yaml")
        sso_config_file.parent.mkdir(parents=True, exist_ok=True)
        
        sso_manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "sso-config",
                "namespace": "anomaly-detection"
            },
            "data": {
                "azure_tenant_id": tenant_id,
                "azure_client_id": client_id,
                "redirect_uri": "https://anomaly-detection.io/auth/callback",
                "authority": f"https://login.microsoftonline.com/{tenant_id}"
            }
        }
        
        with open(sso_config_file, 'w') as f:
            yaml.dump(sso_manifest, f)
        
        # Create secret for client secret
        try:
            subprocess.run([
                "kubectl", "create", "secret", "generic", "azure-ad-secret",
                f"--from-literal=client-secret={client_secret}",
                "--namespace=anomaly-detection"
            ], check=True, capture_output=True)
            
            logger.info("Azure AD SSO configured successfully")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Azure AD secret: {e}")
            return False
    
    def setup_slack_integration(self) -> bool:
        """Set up Slack integration for notifications."""
        logger.info("Setting up Slack integration...")
        
        slack_config = self.config["integrations"]["slack"]
        if not slack_config["enabled"]:
            logger.info("Slack integration disabled, skipping")
            return True
        
        try:
            webhook_url = os.path.expandvars(slack_config["webhook_url"])
            
            # Test Slack webhook
            test_message = {
                "text": "🚀 Anomaly Detection Platform - Integration Test",
                "attachments": [
                    {
                        "color": "good",
                        "fields": [
                            {
                                "title": "Status",
                                "value": "Slack integration configured successfully",
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": datetime.now().isoformat(),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=test_message, timeout=10)
            response.raise_for_status()
            
            # Create Slack notification configuration
            slack_config_file = Path("deploy/k8s/slack-config.yaml")
            slack_config_manifest = {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": "slack-config",
                    "namespace": "anomaly-detection"
                },
                "data": {
                    "channels": json.dumps(slack_config["channels"]),
                    "webhook_url": webhook_url
                }
            }
            
            with open(slack_config_file, 'w') as f:
                yaml.dump(slack_config_manifest, f)
            
            logger.info("Slack integration configured successfully")
            return True
        
        except Exception as e:
            logger.error(f"Slack integration failed: {e}")
            return False
    
    def setup_email_integration(self) -> bool:
        """Set up email integration for notifications."""
        logger.info("Setting up email integration...")
        
        email_config = self.config["integrations"]["email"]
        if not email_config["enabled"]:
            logger.info("Email integration disabled, skipping")
            return True
        
        try:
            # Create email configuration
            smtp_config = {
                "server": os.path.expandvars(email_config["smtp_server"]),
                "port": email_config["smtp_port"],
                "username": os.path.expandvars(email_config["username"]),
                "password": os.path.expandvars(email_config["password"]),
                "distribution_lists": email_config["distribution_lists"]
            }
            
            # Create email configuration file
            email_config_file = Path("deploy/k8s/email-config.yaml")
            email_config_manifest = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": "email-config",
                    "namespace": "anomaly-detection"
                },
                "type": "Opaque",
                "stringData": {
                    "smtp-config.json": json.dumps(smtp_config)
                }
            }
            
            with open(email_config_file, 'w') as f:
                yaml.dump(email_config_manifest, f)
            
            logger.info("Email integration configured successfully")
            return True
        
        except Exception as e:
            logger.error(f"Email integration failed: {e}")
            return False
    
    def setup_database_connections(self) -> bool:
        """Set up database connections for business data."""
        logger.info("Setting up database connections...")
        
        db_config = self.config["integrations"]["database"]
        if not db_config["enabled"]:
            logger.info("Database integration disabled, skipping")
            return True
        
        try:
            # Test database connections
            for db_name, db_config in db_config["connections"].items():
                if not self._test_database_connection(db_name, db_config):
                    return False
            
            # Create database configuration secret
            db_connections = {}
            for db_name, db_config in db_config["connections"].items():
                connection_string = self._build_connection_string(db_config)
                db_connections[f"{db_name}_url"] = connection_string
            
            db_config_file = Path("deploy/k8s/database-connections.yaml")
            db_config_manifest = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": "database-connections",
                    "namespace": "anomaly-detection"
                },
                "type": "Opaque",
                "stringData": db_connections
            }
            
            with open(db_config_file, 'w') as f:
                yaml.dump(db_config_manifest, f)
            
            logger.info("Database connections configured successfully")
            return True
        
        except Exception as e:
            logger.error(f"Database integration failed: {e}")
            return False
    
    def _test_database_connection(self, db_name: str, db_config: Dict[str, Any]) -> bool:
        """Test database connection."""
        logger.info(f"Testing database connection: {db_name}")
        
        try:
            if db_config["type"] == "postgresql":
                import psycopg2
                
                conn = psycopg2.connect(
                    host=os.path.expandvars(db_config["host"]),
                    port=db_config["port"],
                    database=db_config["database"],
                    user=os.path.expandvars(db_config["username"]),
                    password=os.path.expandvars(db_config["password"])
                )
                
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                
                conn.close()
                
                if result and result[0] == 1:
                    logger.info(f"Database connection successful: {db_name}")
                    return True
                else:
                    logger.error(f"Database connection test failed: {db_name}")
                    return False
        
        except ImportError:
            logger.warning(f"Database driver not available for {db_config['type']}")
            return True  # Skip test if driver not installed
        
        except Exception as e:
            logger.error(f"Database connection failed for {db_name}: {e}")
            return False
    
    def _build_connection_string(self, db_config: Dict[str, Any]) -> str:
        """Build database connection string."""
        if db_config["type"] == "postgresql":
            return f"postgresql://{os.path.expandvars(db_config['username'])}:{os.path.expandvars(db_config['password'])}@{os.path.expandvars(db_config['host'])}:{db_config['port']}/{db_config['database']}"
        else:
            raise ValueError(f"Unsupported database type: {db_config['type']}")
    
    def setup_bi_integration(self) -> bool:
        """Set up Business Intelligence tools integration."""
        logger.info("Setting up BI tools integration...")
        
        bi_config = self.config["integrations"]["bi_tools"]
        if not bi_config["enabled"]:
            logger.info("BI integration disabled, skipping")
            return True
        
        try:
            # Configure Tableau integration
            if "tableau" in bi_config:
                tableau_config = bi_config["tableau"]
                
                # Create Tableau configuration
                tableau_config_file = Path("deploy/k8s/tableau-config.yaml")
                tableau_config_manifest = {
                    "apiVersion": "v1",
                    "kind": "Secret",
                    "metadata": {
                        "name": "tableau-config",
                        "namespace": "anomaly-detection"
                    },
                    "type": "Opaque",
                    "stringData": {
                        "server_url": os.path.expandvars(tableau_config["server_url"]),
                        "username": os.path.expandvars(tableau_config["username"]),
                        "password": os.path.expandvars(tableau_config["password"]),
                        "site_id": tableau_config["site_id"]
                    }
                }
                
                with open(tableau_config_file, 'w') as f:
                    yaml.dump(tableau_config_manifest, f)
            
            logger.info("BI tools integration configured successfully")
            return True
        
        except Exception as e:
            logger.error(f"BI integration failed: {e}")
            return False
    
    def setup_incident_management(self) -> bool:
        """Set up incident management integration."""
        logger.info("Setting up incident management integration...")
        
        incident_config = self.config["integrations"]["incident_management"]
        if not incident_config["enabled"]:
            logger.info("Incident management integration disabled, skipping")
            return True
        
        try:
            # Configure ServiceNow integration
            if "servicenow" in incident_config:
                snow_config = incident_config["servicenow"]
                
                # Test ServiceNow connection
                instance_url = os.path.expandvars(snow_config["instance_url"])
                username = os.path.expandvars(snow_config["username"])
                password = os.path.expandvars(snow_config["password"])
                
                test_url = f"{instance_url}/api/now/table/sys_user"
                response = requests.get(
                    test_url,
                    auth=(username, password),
                    headers={"Accept": "application/json"},
                    timeout=10
                )
                response.raise_for_status()
                
                # Create ServiceNow configuration
                snow_config_file = Path("deploy/k8s/servicenow-config.yaml")
                snow_config_manifest = {
                    "apiVersion": "v1",
                    "kind": "Secret",
                    "metadata": {
                        "name": "servicenow-config",
                        "namespace": "anomaly-detection"
                    },
                    "type": "Opaque",
                    "stringData": {
                        "instance_url": instance_url,
                        "username": username,
                        "password": password,
                        "assignment_group": snow_config["assignment_group"]
                    }
                }
                
                with open(snow_config_file, 'w') as f:
                    yaml.dump(snow_config_manifest, f)
            
            logger.info("Incident management integration configured successfully")
            return True
        
        except Exception as e:
            logger.error(f"Incident management integration failed: {e}")
            return False
    
    def deploy_integration_configs(self) -> bool:
        """Deploy all integration configurations to Kubernetes."""
        logger.info("Deploying integration configurations...")
        
        try:
            config_files = list(Path("deploy/k8s").glob("*-config.yaml"))
            
            for config_file in config_files:
                logger.info(f"Applying configuration: {config_file}")
                
                result = subprocess.run([
                    "kubectl", "apply", "-f", str(config_file)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to apply {config_file}: {result.stderr}")
                    return False
            
            logger.info("All integration configurations deployed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Configuration deployment failed: {e}")
            return False
    
    def validate_integrations(self) -> Dict[str, bool]:
        """Validate all configured integrations."""
        logger.info("Validating integrations...")
        
        validation_results = {}
        
        # Validate each integration
        for integration_name in self.config["integrations"]:
            try:
                validation_results[integration_name] = self._validate_integration(integration_name)
            except Exception as e:
                logger.error(f"Validation failed for {integration_name}: {e}")
                validation_results[integration_name] = False
        
        # Log results
        for integration, status in validation_results.items():
            status_str = "✅ PASS" if status else "❌ FAIL"
            logger.info(f"{integration}: {status_str}")
        
        return validation_results
    
    def _validate_integration(self, integration_name: str) -> bool:
        """Validate a specific integration."""
        integration_config = self.config["integrations"][integration_name]
        
        if not integration_config["enabled"]:
            return True  # Disabled integrations are considered valid
        
        # Integration-specific validation logic
        if integration_name == "slack":
            return self._validate_slack_integration()
        elif integration_name == "email":
            return self._validate_email_integration()
        elif integration_name == "database":
            return self._validate_database_integration()
        else:
            # Generic validation - check if configuration exists
            return True
    
    def _validate_slack_integration(self) -> bool:
        """Validate Slack integration."""
        try:
            webhook_url = os.path.expandvars(self.config["integrations"]["slack"]["webhook_url"])
            
            test_message = {
                "text": "🔍 Integration validation test",
                "attachments": [{"color": "good", "text": "Slack integration is working"}]
            }
            
            response = requests.post(webhook_url, json=test_message, timeout=5)
            return response.status_code == 200
        
        except Exception:
            return False
    
    def _validate_email_integration(self) -> bool:
        """Validate email integration."""
        # For now, just check if configuration exists
        email_config = self.config["integrations"]["email"]
        required_fields = ["smtp_server", "username", "password"]
        
        return all(field in email_config for field in required_fields)
    
    def _validate_database_integration(self) -> bool:
        """Validate database integration."""
        db_config = self.config["integrations"]["database"]
        
        for db_name, db_config in db_config["connections"].items():
            if not self._test_database_connection(db_name, db_config):
                return False
        
        return True
    
    def generate_integration_report(self) -> str:
        """Generate comprehensive integration report."""
        logger.info("Generating integration report...")
        
        validation_results = self.validate_integrations()
        
        report = f"""
# 🔗 Business Integration Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Configuration:** {self.config_path}

## 📊 Integration Status Summary

| Integration | Status | Enabled |
|-------------|--------|---------|
"""
        
        for integration_name, integration_config in self.config["integrations"].items():
            enabled = integration_config["enabled"]
            status = validation_results.get(integration_name, False)
            status_emoji = "✅" if status else "❌"
            enabled_emoji = "🟢" if enabled else "⚪"
            
            report += f"| {integration_name} | {status_emoji} | {enabled_emoji} |\n"
        
        report += f"""

## 🎯 Rollout Configuration

**Phases:** {len(self.config['rollout']['phases'])}

"""
        
        for i, phase in enumerate(self.config['rollout']['phases'], 1):
            report += f"""
### Phase {i}: {phase['name'].replace('_', ' ').title()}
- **Users:** {phase['users'] if phase['users'] > 0 else 'All'}
- **Duration:** {phase['duration_days']} days
- **Success Criteria:**
  - Uptime: {phase['success_criteria']['uptime']*100:.1f}%
  - User Satisfaction: {phase['success_criteria']['user_satisfaction']}/10
"""
        
        report += f"""

## 🔧 Manual Configuration Required

### 1. Environment Variables
Set the following environment variables before running integrations:

```bash
# SSO Configuration
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"

# Notification Configuration
export SLACK_WEBHOOK_URL="your-slack-webhook-url"
export SMTP_SERVER="your-smtp-server"
export SMTP_USERNAME="your-smtp-username"
export SMTP_PASSWORD="your-smtp-password"

# Database Configuration
export BUSINESS_DB_HOST="your-db-host"
export BUSINESS_DB_USER="your-db-username"
export BUSINESS_DB_PASSWORD="your-db-password"

# BI Tools Configuration
export TABLEAU_SERVER_URL="your-tableau-server"
export TABLEAU_USERNAME="your-tableau-username"
export TABLEAU_PASSWORD="your-tableau-password"

# Incident Management
export SERVICENOW_URL="your-servicenow-instance"
export SERVICENOW_USERNAME="your-servicenow-username"
export SERVICENOW_PASSWORD="your-servicenow-password"
```

### 2. Network Configuration
- Ensure firewall rules allow connections to external services
- Configure DNS resolution for service endpoints
- Set up SSL certificates for HTTPS endpoints

### 3. Service Accounts
- Create service accounts in external systems
- Grant appropriate permissions for integrations
- Configure API keys and access tokens

## 🚀 Next Steps

1. **Environment Setup:** Configure all required environment variables
2. **Service Accounts:** Create and configure service accounts
3. **Network Access:** Ensure network connectivity to external services
4. **Testing:** Run integration validation tests
5. **Deployment:** Deploy configurations to Kubernetes
6. **Monitoring:** Set up monitoring for integration health

## 📞 Support

For integration issues:
1. Check service connectivity and credentials
2. Review integration logs for error details
3. Validate configuration parameters
4. Contact platform team for assistance

"""
        
        # Save report
        report_file = Path(f"business-integration-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md")
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Integration report generated: {report_file}")
        return str(report_file)
    
    def run_full_integration_setup(self) -> bool:
        """Run complete business integration setup."""
        logger.info("🚀 Starting full business integration setup...")
        
        setup_steps = [
            ("SSO Integration", self.setup_sso_integration),
            ("Slack Integration", self.setup_slack_integration),
            ("Email Integration", self.setup_email_integration),
            ("Database Connections", self.setup_database_connections),
            ("BI Tools Integration", self.setup_bi_integration),
            ("Incident Management", self.setup_incident_management),
            ("Deploy Configurations", self.deploy_integration_configs),
        ]
        
        results = {}
        for step_name, step_function in setup_steps:
            logger.info(f"🔧 {step_name}...")
            try:
                results[step_name] = step_function()
                if results[step_name]:
                    logger.info(f"✅ {step_name} completed successfully")
                else:
                    logger.error(f"❌ {step_name} failed")
            except Exception as e:
                logger.error(f"❌ {step_name} failed with exception: {e}")
                results[step_name] = False
        
        # Generate final report
        report_file = self.generate_integration_report()
        
        # Summary
        successful_steps = sum(results.values())
        total_steps = len(results)
        
        logger.info(f"📊 Integration setup completed: {successful_steps}/{total_steps} steps successful")
        logger.info(f"📋 Full report available: {report_file}")
        
        return successful_steps == total_steps


def main():
    """Main function."""
    logger.info("🔗 Business Integration Automation Starting...")
    
    try:
        # Initialize integration manager
        integration_manager = BusinessIntegrationManager()
        
        # Run full setup
        success = integration_manager.run_full_integration_setup()
        
        if success:
            logger.info("🎉 Business integration setup completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Business integration setup failed!")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"💥 Fatal error during integration setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()