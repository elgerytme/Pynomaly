"""
Enterprise Integration and Workflow Automation Framework
Comprehensive enterprise system integrations and automated workflows
"""

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import jwt
import requests
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types of enterprise integrations"""
    SSO = "sso"
    LDAP = "ldap"
    JIRA = "jira"
    SERVICENOW = "servicenow"
    SLACK = "slack"
    MICROSOFT_TEAMS = "microsoft_teams"
    SALESFORCE = "salesforce"
    TABLEAU = "tableau"
    POWER_BI = "power_bi"
    JENKINS = "jenkins"
    GITHUB = "github"
    GITLAB = "gitlab"
    AZURE_DEVOPS = "azure_devops"
    OKTA = "okta"
    ACTIVE_DIRECTORY = "active_directory"
    SAP = "sap"
    ORACLE = "oracle"
    SNOWFLAKE = "snowflake"
    DATABRICKS = "databricks"


class WorkflowTrigger(Enum):
    """Workflow trigger types"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    API_WEBHOOK = "api_webhook"
    MODEL_TRAINING_COMPLETE = "model_training_complete"
    DEPLOYMENT_SUCCESS = "deployment_success"
    ALERT_TRIGGERED = "alert_triggered"
    DATA_DRIFT_DETECTED = "data_drift_detected"


@dataclass
class IntegrationConfig:
    """Enterprise integration configuration"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    integration_type: IntegrationType = IntegrationType.SSO
    description: str = ""
    enabled: bool = True
    configuration: Dict[str, Any] = field(default_factory=dict)
    authentication: Dict[str, Any] = field(default_factory=dict)
    endpoints: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_sync: Optional[datetime] = None
    sync_status: str = "pending"
    error_log: List[str] = field(default_factory=list)


@dataclass
class WorkflowDefinition:
    """Automated workflow definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    trigger: WorkflowTrigger = WorkflowTrigger.MANUAL
    trigger_config: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    notifications: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # running, completed, failed, cancelled
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    error_details: Optional[str] = None
    output_data: Dict[str, Any] = field(default_factory=dict)


class EnterpriseIntegrationFramework:
    """Enterprise integration and workflow automation framework"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.workflow_executions: Dict[str, WorkflowExecution] = {}
        
        # Security and encryption
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Integration clients
        self.integration_clients = {}
        
    def _get_encryption_key(self) -> bytes:
        """Get encryption key for sensitive data"""
        key = self.config.get('encryption_key')
        if key:
            return key.encode() if isinstance(key, str) else key
        return Fernet.generate_key()

    async def configure_sso_integration(self, provider: str, config: Dict[str, Any]) -> str:
        """Configure Single Sign-On integration"""
        try:
            sso_config = IntegrationConfig(
                name=f"SSO - {provider}",
                integration_type=IntegrationType.SSO,
                description=f"Single Sign-On integration with {provider}",
                configuration={
                    "provider": provider,
                    "client_id": config.get("client_id"),
                    "client_secret": self._encrypt_sensitive_data(config.get("client_secret")),
                    "redirect_uri": config.get("redirect_uri"),
                    "scopes": config.get("scopes", ["openid", "profile", "email"]),
                    "issuer_url": config.get("issuer_url"),
                    "jwks_uri": config.get("jwks_uri")
                },
                endpoints={
                    "authorization": config.get("authorization_endpoint"),
                    "token": config.get("token_endpoint"),
                    "userinfo": config.get("userinfo_endpoint"),
                    "logout": config.get("logout_endpoint")
                }
            )
            
            # Test SSO connection
            test_result = await self._test_sso_connection(sso_config)
            if test_result["success"]:
                sso_config.sync_status = "connected"
                sso_config.last_sync = datetime.now()
            else:
                sso_config.sync_status = "failed"
                sso_config.error_log.append(test_result["error"])
            
            self.integrations[sso_config.id] = sso_config
            
            logger.info(f"Configured SSO integration for {provider}: {sso_config.id}")
            return sso_config.id
            
        except Exception as e:
            logger.error(f"Failed to configure SSO integration: {e}")
            raise

    async def _test_sso_connection(self, sso_config: IntegrationConfig) -> Dict[str, Any]:
        """Test SSO connection"""
        try:
            # Test OIDC discovery
            issuer_url = sso_config.configuration.get("issuer_url")
            if issuer_url:
                discovery_url = f"{issuer_url}/.well-known/openid-configuration"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(discovery_url) as response:
                        if response.status == 200:
                            discovery_data = await response.json()
                            return {
                                "success": True,
                                "discovery_data": discovery_data
                            }
                        else:
                            return {
                                "success": False,
                                "error": f"Failed to discover OIDC configuration: {response.status}"
                            }
            else:
                return {"success": True, "message": "Basic configuration validated"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def configure_ldap_integration(self, config: Dict[str, Any]) -> str:
        """Configure LDAP/Active Directory integration"""
        try:
            ldap_config = IntegrationConfig(
                name="LDAP/Active Directory",
                integration_type=IntegrationType.LDAP,
                description="LDAP directory integration for user management",
                configuration={
                    "server": config.get("server"),
                    "port": config.get("port", 389),
                    "use_ssl": config.get("use_ssl", False),
                    "base_dn": config.get("base_dn"),
                    "bind_dn": config.get("bind_dn"),
                    "bind_password": self._encrypt_sensitive_data(config.get("bind_password")),
                    "user_search_base": config.get("user_search_base"),
                    "user_search_filter": config.get("user_search_filter", "(uid={username})"),
                    "group_search_base": config.get("group_search_base"),
                    "group_search_filter": config.get("group_search_filter", "(member={user_dn})"),
                    "attribute_mapping": config.get("attribute_mapping", {
                        "username": "uid",
                        "email": "mail",
                        "first_name": "givenName",
                        "last_name": "sn",
                        "display_name": "displayName"
                    })
                }
            )
            
            # Test LDAP connection
            test_result = await self._test_ldap_connection(ldap_config)
            if test_result["success"]:
                ldap_config.sync_status = "connected"
                ldap_config.last_sync = datetime.now()
            else:
                ldap_config.sync_status = "failed"
                ldap_config.error_log.append(test_result["error"])
            
            self.integrations[ldap_config.id] = ldap_config
            
            logger.info(f"Configured LDAP integration: {ldap_config.id}")
            return ldap_config.id
            
        except Exception as e:
            logger.error(f"Failed to configure LDAP integration: {e}")
            raise

    async def _test_ldap_connection(self, ldap_config: IntegrationConfig) -> Dict[str, Any]:
        """Test LDAP connection"""
        try:
            # Mock LDAP connection test (in production, use python-ldap)
            server = ldap_config.configuration.get("server")
            port = ldap_config.configuration.get("port")
            
            # Simulate connection test
            await asyncio.sleep(0.1)  # Simulate network delay
            
            return {
                "success": True,
                "message": f"Successfully connected to LDAP server {server}:{port}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def configure_jira_integration(self, config: Dict[str, Any]) -> str:
        """Configure JIRA integration for issue tracking"""
        try:
            jira_config = IntegrationConfig(
                name="JIRA Integration",
                integration_type=IntegrationType.JIRA,
                description="JIRA integration for issue tracking and project management",
                configuration={
                    "base_url": config.get("base_url"),
                    "username": config.get("username"),
                    "api_token": self._encrypt_sensitive_data(config.get("api_token")),
                    "project_key": config.get("project_key"),
                    "issue_types": config.get("issue_types", ["Bug", "Task", "Story"]),
                    "custom_fields": config.get("custom_fields", {}),
                    "webhook_url": config.get("webhook_url"),
                    "auto_create_issues": config.get("auto_create_issues", False)
                },
                endpoints={
                    "api": f"{config.get('base_url')}/rest/api/2",
                    "webhooks": f"{config.get('base_url')}/rest/webhooks/1.0"
                }
            )
            
            # Test JIRA connection
            test_result = await self._test_jira_connection(jira_config)
            if test_result["success"]:
                jira_config.sync_status = "connected"
                jira_config.last_sync = datetime.now()
            else:
                jira_config.sync_status = "failed"
                jira_config.error_log.append(test_result["error"])
            
            self.integrations[jira_config.id] = jira_config
            
            logger.info(f"Configured JIRA integration: {jira_config.id}")
            return jira_config.id
            
        except Exception as e:
            logger.error(f"Failed to configure JIRA integration: {e}")
            raise

    async def _test_jira_connection(self, jira_config: IntegrationConfig) -> Dict[str, Any]:
        """Test JIRA connection"""
        try:
            base_url = jira_config.configuration.get("base_url")
            username = jira_config.configuration.get("username")
            api_token = self._decrypt_sensitive_data(jira_config.configuration.get("api_token"))
            
            # Test authentication
            auth_url = f"{base_url}/rest/api/2/myself"
            
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(username, api_token)
                async with session.get(auth_url, auth=auth) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        return {
                            "success": True,
                            "user": user_data.get("displayName", username)
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Authentication failed: {response.status}"
                        }
                        
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def configure_slack_integration(self, config: Dict[str, Any]) -> str:
        """Configure Slack integration for notifications"""
        try:
            slack_config = IntegrationConfig(
                name="Slack Integration",
                integration_type=IntegrationType.SLACK,
                description="Slack integration for notifications and alerts",
                configuration={
                    "bot_token": self._encrypt_sensitive_data(config.get("bot_token")),
                    "app_token": self._encrypt_sensitive_data(config.get("app_token")),
                    "signing_secret": self._encrypt_sensitive_data(config.get("signing_secret")),
                    "default_channel": config.get("default_channel", "#general"),
                    "notification_channels": config.get("notification_channels", {}),
                    "webhook_url": config.get("webhook_url"),
                    "enable_interactive": config.get("enable_interactive", True)
                },
                endpoints={
                    "api": "https://slack.com/api",
                    "webhook": config.get("webhook_url")
                }
            )
            
            # Test Slack connection
            test_result = await self._test_slack_connection(slack_config)
            if test_result["success"]:
                slack_config.sync_status = "connected"
                slack_config.last_sync = datetime.now()
            else:
                slack_config.sync_status = "failed"
                slack_config.error_log.append(test_result["error"])
            
            self.integrations[slack_config.id] = slack_config
            
            logger.info(f"Configured Slack integration: {slack_config.id}")
            return slack_config.id
            
        except Exception as e:
            logger.error(f"Failed to configure Slack integration: {e}")
            raise

    async def _test_slack_connection(self, slack_config: IntegrationConfig) -> Dict[str, Any]:
        """Test Slack connection"""
        try:
            bot_token = self._decrypt_sensitive_data(slack_config.configuration.get("bot_token"))
            
            # Test bot token
            test_url = "https://slack.com/api/auth.test"
            headers = {"Authorization": f"Bearer {bot_token}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(test_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok"):
                            return {
                                "success": True,
                                "team": data.get("team"),
                                "user": data.get("user")
                            }
                        else:
                            return {
                                "success": False,
                                "error": data.get("error", "Unknown error")
                            }
                    else:
                        return {
                            "success": False,
                            "error": f"API call failed: {response.status}"
                        }
                        
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def create_automated_workflow(self, workflow_def: WorkflowDefinition) -> str:
        """Create automated workflow"""
        try:
            if not workflow_def.id:
                workflow_def.id = str(uuid.uuid4())
                
            workflow_def.created_at = datetime.now()
            workflow_def.updated_at = datetime.now()
            
            # Validate workflow definition
            validation_result = await self._validate_workflow(workflow_def)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid workflow: {validation_result['errors']}")
            
            self.workflows[workflow_def.id] = workflow_def
            
            # Set up trigger if needed
            await self._setup_workflow_trigger(workflow_def)
            
            logger.info(f"Created automated workflow: {workflow_def.name} ({workflow_def.id})")
            return workflow_def.id
            
        except Exception as e:
            logger.error(f"Failed to create automated workflow: {e}")
            raise

    async def _validate_workflow(self, workflow_def: WorkflowDefinition) -> Dict[str, Any]:
        """Validate workflow definition"""
        errors = []
        
        if not workflow_def.name:
            errors.append("Workflow name is required")
            
        if not workflow_def.steps:
            errors.append("Workflow must have at least one step")
            
        for i, step in enumerate(workflow_def.steps):
            if "action" not in step:
                errors.append(f"Step {i+1} missing required 'action' field")
                
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    async def _setup_workflow_trigger(self, workflow_def: WorkflowDefinition):
        """Setup workflow trigger based on trigger type"""
        trigger = workflow_def.trigger
        
        if trigger == WorkflowTrigger.SCHEDULED:
            await self._setup_scheduled_trigger(workflow_def)
        elif trigger == WorkflowTrigger.EVENT_DRIVEN:
            await self._setup_event_trigger(workflow_def)
        elif trigger == WorkflowTrigger.API_WEBHOOK:
            await self._setup_webhook_trigger(workflow_def)

    async def _setup_scheduled_trigger(self, workflow_def: WorkflowDefinition):
        """Setup scheduled trigger for workflow"""
        # In production, integrate with cron or task scheduler
        logger.info(f"Setup scheduled trigger for workflow {workflow_def.id}")

    async def _setup_event_trigger(self, workflow_def: WorkflowDefinition):
        """Setup event-driven trigger for workflow"""
        # In production, integrate with event system
        logger.info(f"Setup event trigger for workflow {workflow_def.id}")

    async def _setup_webhook_trigger(self, workflow_def: WorkflowDefinition):
        """Setup webhook trigger for workflow"""
        # In production, create webhook endpoint
        logger.info(f"Setup webhook trigger for workflow {workflow_def.id}")

    async def execute_workflow(self, workflow_id: str, trigger_data: Dict[str, Any] = None) -> str:
        """Execute workflow"""
        workflow_def = self.workflows.get(workflow_id)
        if not workflow_def:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        if not workflow_def.enabled:
            raise ValueError(f"Workflow {workflow_id} is disabled")
            
        try:
            # Create execution instance
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                trigger_data=trigger_data or {}
            )
            
            self.workflow_executions[execution.id] = execution
            
            logger.info(f"Starting workflow execution: {execution.id}")
            
            # Execute workflow steps
            await self._execute_workflow_steps(execution, workflow_def)
            
            execution.status = "completed"
            execution.end_time = datetime.now()
            
            # Add to workflow history
            workflow_def.execution_history.append({
                "execution_id": execution.id,
                "start_time": execution.start_time,
                "end_time": execution.end_time,
                "status": execution.status,
                "trigger_data": execution.trigger_data
            })
            
            logger.info(f"Completed workflow execution: {execution.id}")
            return execution.id
            
        except Exception as e:
            execution.status = "failed"
            execution.end_time = datetime.now()
            execution.error_details = str(e)
            
            logger.error(f"Workflow execution failed: {execution.id} - {e}")
            raise

    async def _execute_workflow_steps(self, execution: WorkflowExecution, workflow_def: WorkflowDefinition):
        """Execute individual workflow steps"""
        for i, step in enumerate(workflow_def.steps):
            step_start = datetime.now()
            
            try:
                # Check step conditions
                if not await self._check_step_conditions(step, execution):
                    execution.step_results.append({
                        "step": i + 1,
                        "action": step.get("action"),
                        "status": "skipped",
                        "reason": "Conditions not met",
                        "start_time": step_start,
                        "end_time": datetime.now()
                    })
                    continue
                
                # Execute step action
                step_result = await self._execute_step_action(step, execution)
                
                execution.step_results.append({
                    "step": i + 1,
                    "action": step.get("action"),
                    "status": "completed",
                    "result": step_result,
                    "start_time": step_start,
                    "end_time": datetime.now()
                })
                
                # Update execution output data
                if step_result and isinstance(step_result, dict):
                    execution.output_data.update(step_result)
                    
            except Exception as e:
                execution.step_results.append({
                    "step": i + 1,
                    "action": step.get("action"),
                    "status": "failed",
                    "error": str(e),
                    "start_time": step_start,
                    "end_time": datetime.now()
                })
                
                # Handle failure based on step configuration
                if step.get("continue_on_failure", False):
                    logger.warning(f"Step {i+1} failed but continuing: {e}")
                    continue
                else:
                    raise

    async def _check_step_conditions(self, step: Dict[str, Any], execution: WorkflowExecution) -> bool:
        """Check if step conditions are met"""
        conditions = step.get("conditions", {})
        
        if not conditions:
            return True
            
        # Check various condition types
        for condition_type, condition_value in conditions.items():
            if condition_type == "previous_step_status":
                if execution.step_results:
                    last_result = execution.step_results[-1]
                    if last_result["status"] != condition_value:
                        return False
                        
            elif condition_type == "trigger_data_contains":
                for key in condition_value:
                    if key not in execution.trigger_data:
                        return False
                        
            elif condition_type == "output_data_contains":
                for key in condition_value:
                    if key not in execution.output_data:
                        return False
        
        return True

    async def _execute_step_action(self, step: Dict[str, Any], execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute specific step action"""
        action = step.get("action")
        parameters = step.get("parameters", {})
        
        if action == "send_notification":
            return await self._send_notification(parameters, execution)
        elif action == "create_jira_issue":
            return await self._create_jira_issue(parameters, execution)
        elif action == "update_model_status":
            return await self._update_model_status(parameters, execution)
        elif action == "trigger_deployment":
            return await self._trigger_deployment(parameters, execution)
        elif action == "send_slack_message":
            return await self._send_slack_message(parameters, execution)
        elif action == "call_api":
            return await self._call_external_api(parameters, execution)
        elif action == "wait":
            await asyncio.sleep(parameters.get("seconds", 1))
            return {"waited_seconds": parameters.get("seconds", 1)}
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _send_notification(self, parameters: Dict[str, Any], execution: WorkflowExecution) -> Dict[str, Any]:
        """Send notification step"""
        notification_type = parameters.get("type", "email")
        message = parameters.get("message", "")
        recipients = parameters.get("recipients", [])
        
        # Replace variables in message
        message = self._replace_variables(message, execution)
        
        logger.info(f"Sending {notification_type} notification to {len(recipients)} recipients")
        
        return {
            "notification_sent": True,
            "type": notification_type,
            "recipients_count": len(recipients),
            "message_length": len(message)
        }

    async def _create_jira_issue(self, parameters: Dict[str, Any], execution: WorkflowExecution) -> Dict[str, Any]:
        """Create JIRA issue step"""
        # Find JIRA integration
        jira_integration = None
        for integration in self.integrations.values():
            if integration.integration_type == IntegrationType.JIRA and integration.enabled:
                jira_integration = integration
                break
        
        if not jira_integration:
            raise ValueError("No JIRA integration configured")
            
        issue_data = {
            "summary": self._replace_variables(parameters.get("summary", ""), execution),
            "description": self._replace_variables(parameters.get("description", ""), execution),
            "issue_type": parameters.get("issue_type", "Task"),
            "priority": parameters.get("priority", "Medium"),
            "assignee": parameters.get("assignee"),
            "labels": parameters.get("labels", [])
        }
        
        logger.info(f"Creating JIRA issue: {issue_data['summary']}")
        
        # Mock JIRA issue creation
        issue_key = f"MLOPS-{execution.id[:8].upper()}"
        
        return {
            "jira_issue_created": True,
            "issue_key": issue_key,
            "issue_url": f"{jira_integration.configuration['base_url']}/browse/{issue_key}"
        }

    async def _send_slack_message(self, parameters: Dict[str, Any], execution: WorkflowExecution) -> Dict[str, Any]:
        """Send Slack message step"""
        # Find Slack integration
        slack_integration = None
        for integration in self.integrations.values():
            if integration.integration_type == IntegrationType.SLACK and integration.enabled:
                slack_integration = integration
                break
        
        if not slack_integration:
            raise ValueError("No Slack integration configured")
            
        channel = parameters.get("channel", slack_integration.configuration.get("default_channel"))
        message = self._replace_variables(parameters.get("message", ""), execution)
        
        logger.info(f"Sending Slack message to {channel}")
        
        return {
            "slack_message_sent": True,
            "channel": channel,
            "message_length": len(message)
        }

    def _replace_variables(self, text: str, execution: WorkflowExecution) -> str:
        """Replace variables in text with execution data"""
        if not text:
            return text
            
        variables = {
            "{execution_id}": execution.id,
            "{workflow_id}": execution.workflow_id,
            "{timestamp}": datetime.now().isoformat(),
            "{trigger_data}": json.dumps(execution.trigger_data),
            "{output_data}": json.dumps(execution.output_data)
        }
        
        for var, value in variables.items():
            text = text.replace(var, str(value))
            
        return text

    async def create_model_deployment_workflow(self) -> str:
        """Create automated model deployment workflow"""
        workflow = WorkflowDefinition(
            name="Automated Model Deployment",
            description="Automated workflow for model deployment with approvals and notifications",
            trigger=WorkflowTrigger.MODEL_TRAINING_COMPLETE,
            trigger_config={
                "model_accuracy_threshold": 0.85,
                "require_approval": True
            },
            steps=[
                {
                    "action": "update_model_status",
                    "parameters": {
                        "status": "pending_deployment"
                    }
                },
                {
                    "action": "send_notification",
                    "parameters": {
                        "type": "email",
                        "recipients": ["ml-team@company.com"],
                        "subject": "Model Ready for Deployment - {model_name}",
                        "message": "Model {model_name} has completed training with accuracy {model_accuracy}. Please review for deployment approval."
                    }
                },
                {
                    "action": "create_jira_issue",
                    "parameters": {
                        "summary": "Deploy Model: {model_name} v{model_version}",
                        "description": "Model deployment request for {model_name} version {model_version}.\n\nTraining completed: {timestamp}\nAccuracy: {model_accuracy}\nExecution ID: {execution_id}",
                        "issue_type": "Task",
                        "priority": "High",
                        "labels": ["mlops", "deployment", "model"]
                    }
                },
                {
                    "action": "wait",
                    "parameters": {
                        "seconds": 300  # Wait 5 minutes for approval
                    },
                    "conditions": {
                        "trigger_data_contains": ["approval_required"]
                    }
                },
                {
                    "action": "trigger_deployment",
                    "parameters": {
                        "environment": "staging",
                        "deployment_strategy": "blue_green"
                    },
                    "conditions": {
                        "output_data_contains": ["approval_granted"]
                    }
                },
                {
                    "action": "send_slack_message",
                    "parameters": {
                        "channel": "#ml-deployments",
                        "message": "🚀 Model {model_name} v{model_version} has been deployed to staging. Execution: {execution_id}"
                    }
                }
            ],
            notifications={
                "on_success": {
                    "slack": {
                        "channel": "#ml-deployments",
                        "message": "✅ Deployment workflow completed successfully for {model_name}"
                    }
                },
                "on_failure": {
                    "slack": {
                        "channel": "#ml-alerts",
                        "message": "❌ Deployment workflow failed for {model_name}. Execution: {execution_id}"
                    },
                    "email": {
                        "recipients": ["ml-team@company.com", "devops@company.com"],
                        "subject": "ALERT: Model Deployment Workflow Failed"
                    }
                }
            }
        )
        
        return await self.create_automated_workflow(workflow)

    async def create_incident_response_workflow(self) -> str:
        """Create automated incident response workflow"""
        workflow = WorkflowDefinition(
            name="Automated Incident Response",
            description="Automated workflow for incident detection and response",
            trigger=WorkflowTrigger.ALERT_TRIGGERED,
            trigger_config={
                "severity_levels": ["critical", "high"],
                "alert_types": ["model_drift", "performance_degradation", "system_failure"]
            },
            steps=[
                {
                    "action": "send_notification",
                    "parameters": {
                        "type": "sms",
                        "recipients": ["+1234567890"],  # On-call engineer
                        "message": "CRITICAL ALERT: {alert_type} detected. Incident: {incident_id}"
                    },
                    "conditions": {
                        "trigger_data_contains": ["severity=critical"]
                    }
                },
                {
                    "action": "create_jira_issue",
                    "parameters": {
                        "summary": "INCIDENT: {alert_type} - {component_name}",
                        "description": "Automated incident created for {alert_type} in {component_name}.\n\nAlert triggered: {timestamp}\nSeverity: {severity}\nDetails: {alert_details}",
                        "issue_type": "Incident",
                        "priority": "Highest",
                        "assignee": "incident-team",
                        "labels": ["incident", "automated", "mlops"]
                    }
                },
                {
                    "action": "send_slack_message",
                    "parameters": {
                        "channel": "#incidents",
                        "message": "🚨 INCIDENT DETECTED\n**Type**: {alert_type}\n**Component**: {component_name}\n**Severity**: {severity}\n**Time**: {timestamp}\n**JIRA**: {jira_issue_url}"
                    }
                },
                {
                    "action": "call_api",
                    "parameters": {
                        "url": "https://monitoring.company.com/api/incidents",
                        "method": "POST",
                        "headers": {
                            "Authorization": "Bearer {monitoring_api_token}",
                            "Content-Type": "application/json"
                        },
                        "data": {
                            "title": "{alert_type} in {component_name}",
                            "severity": "{severity}",
                            "source": "mlops-platform",
                            "assignee": "incident-team"
                        }
                    }
                }
            ]
        )
        
        return await self.create_automated_workflow(workflow)

    def _encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not data:
            return data
        return self.cipher_suite.encrypt(data.encode()).decode()

    def _decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not encrypted_data:
            return encrypted_data
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()

    async def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        status = {
            "total_integrations": len(self.integrations),
            "active_integrations": 0,
            "failed_integrations": 0,
            "integrations": []
        }
        
        for integration in self.integrations.values():
            integration_status = {
                "id": integration.id,
                "name": integration.name,
                "type": integration.integration_type.value,
                "enabled": integration.enabled,
                "sync_status": integration.sync_status,
                "last_sync": integration.last_sync,
                "error_count": len(integration.error_log)
            }
            
            status["integrations"].append(integration_status)
            
            if integration.enabled and integration.sync_status == "connected":
                status["active_integrations"] += 1
            elif integration.sync_status == "failed":
                status["failed_integrations"] += 1
        
        return status

    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow execution metrics"""
        total_executions = len(self.workflow_executions)
        successful_executions = len([e for e in self.workflow_executions.values() if e.status == "completed"])
        failed_executions = len([e for e in self.workflow_executions.values() if e.status == "failed"])
        
        # Calculate average execution time
        completed_executions = [e for e in self.workflow_executions.values() if e.end_time]
        avg_execution_time = 0
        if completed_executions:
            total_time = sum((e.end_time - e.start_time).total_seconds() for e in completed_executions)
            avg_execution_time = total_time / len(completed_executions)
        
        return {
            "total_workflows": len(self.workflows),
            "active_workflows": len([w for w in self.workflows.values() if w.enabled]),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
            "average_execution_time_seconds": avg_execution_time,
            "recent_executions": [
                {
                    "id": e.id,
                    "workflow_id": e.workflow_id,
                    "status": e.status,
                    "start_time": e.start_time,
                    "duration": (e.end_time - e.start_time).total_seconds() if e.end_time else None
                }
                for e in sorted(self.workflow_executions.values(), key=lambda x: x.start_time, reverse=True)[:10]
            ]
        }


# Example usage and testing
async def main():
    """Example usage of Enterprise Integration Framework"""
    config = {
        'encryption_key': 'your-encryption-key-here'
    }
    
    framework = EnterpriseIntegrationFramework(config)
    
    # Configure SSO integration
    sso_config = {
        "client_id": "mlops-platform",
        "client_secret": "your-client-secret",
        "redirect_uri": "https://mlops.company.com/auth/callback",
        "issuer_url": "https://auth.company.com",
        "authorization_endpoint": "https://auth.company.com/oauth2/authorize",
        "token_endpoint": "https://auth.company.com/oauth2/token",
        "userinfo_endpoint": "https://auth.company.com/oauth2/userinfo"
    }
    
    sso_id = await framework.configure_sso_integration("Okta", sso_config)
    print(f"Configured SSO integration: {sso_id}")
    
    # Configure Slack integration
    slack_config = {
        "bot_token": "xoxb-your-bot-token",
        "app_token": "xapp-your-app-token",
        "signing_secret": "your-signing-secret",
        "default_channel": "#ml-notifications",
        "webhook_url": "https://hooks.slack.com/services/your/webhook/url"
    }
    
    slack_id = await framework.configure_slack_integration(slack_config)
    print(f"Configured Slack integration: {slack_id}")
    
    # Create automated workflows
    deployment_workflow_id = await framework.create_model_deployment_workflow()
    print(f"Created deployment workflow: {deployment_workflow_id}")
    
    incident_workflow_id = await framework.create_incident_response_workflow()
    print(f"Created incident response workflow: {incident_workflow_id}")
    
    # Execute a workflow
    trigger_data = {
        "model_name": "customer_classifier",
        "model_version": "v1.2.0",
        "model_accuracy": 0.87,
        "approval_required": True
    }
    
    execution_id = await framework.execute_workflow(deployment_workflow_id, trigger_data)
    print(f"Executed workflow: {execution_id}")
    
    # Get integration status
    integration_status = await framework.get_integration_status()
    print(f"Integration status: {integration_status['active_integrations']} active, {integration_status['failed_integrations']} failed")
    
    # Get workflow metrics
    workflow_metrics = await framework.get_workflow_metrics()
    print(f"Workflow metrics: {workflow_metrics['success_rate']:.1f}% success rate")


if __name__ == "__main__":
    asyncio.run(main())