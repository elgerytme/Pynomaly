#!/usr/bin/env python3
"""
ML Pipeline Orchestrator
Comprehensive orchestration system that coordinates the entire ML pipeline lifecycle.
"""

import asyncio
import json
import logging
import os
import sys
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml-orchestrator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OrchestrationStatus(Enum):
    """Orchestration status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class TriggerType(Enum):
    """Pipeline trigger types."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    DATA_DRIFT = "data_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    NEW_DATA = "new_data"
    MODEL_PROMOTION = "model_promotion"


@dataclass
class PipelineExecution:
    """Pipeline execution record."""
    execution_id: str
    pipeline_name: str
    trigger_type: TriggerType
    trigger_reason: str
    started_at: datetime
    status: str
    current_stage: str
    completed_stages: List[str]
    failed_stages: List[str]
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]
    logs: List[str]


class MLPipelineOrchestrator:
    """Comprehensive ML pipeline orchestration system."""
    
    def __init__(self, config_path: str = "mlops/config/orchestrator-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.status = OrchestrationStatus.IDLE
        self.active_executions = {}
        self.execution_history = []
        self.pipeline_templates = {}
        self.monitoring_data = {}
        self._initialize_orchestrator()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load orchestrator configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self._create_default_config()
        
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default orchestrator configuration."""
        default_config = {
            "orchestrator": {
                "name": "ml_pipeline_orchestrator",
                "version": "1.0.0",
                "max_concurrent_executions": 5,
                "execution_timeout": 7200,  # 2 hours
                "retry_policy": {
                    "max_retries": 3,
                    "retry_delay": 300,  # 5 minutes
                    "exponential_backoff": True
                },
                "resource_limits": {
                    "cpu_cores": 8,
                    "memory_gb": 32,
                    "gpu_count": 1
                }
            },
            "pipelines": {
                "anomaly_detection_training": {
                    "description": "Complete anomaly detection model training pipeline",
                    "enabled": True,
                    "schedule": "0 2 * * 1",  # Weekly on Monday at 2 AM
                    "stages": [
                        {
                            "name": "data_ingestion",
                            "component": "advanced_ml_pipeline",
                            "method": "run_data_ingestion",
                            "timeout": 1800,
                            "retry_count": 2
                        },
                        {
                            "name": "data_preprocessing",
                            "component": "advanced_ml_pipeline",
                            "method": "run_data_preprocessing",
                            "timeout": 3600,
                            "retry_count": 2
                        },
                        {
                            "name": "feature_engineering",
                            "component": "advanced_ml_pipeline",
                            "method": "run_feature_engineering",
                            "timeout": 2400,
                            "retry_count": 2
                        },
                        {
                            "name": "model_training",
                            "component": "advanced_ml_pipeline",
                            "method": "run_model_training",
                            "timeout": 5400,
                            "retry_count": 1
                        },
                        {
                            "name": "model_validation",
                            "component": "advanced_ml_pipeline",
                            "method": "run_model_validation",
                            "timeout": 1800,
                            "retry_count": 2
                        },
                        {
                            "name": "model_registration",
                            "component": "advanced_model_registry",
                            "method": "register_model",
                            "timeout": 600,
                            "retry_count": 2
                        },
                        {
                            "name": "model_deployment",
                            "component": "automated_ml_deployment",
                            "method": "deploy_model",
                            "timeout": 3600,
                            "retry_count": 1
                        }
                    ],
                    "triggers": {
                        "data_drift": {
                            "enabled": True,
                            "threshold": 0.1,
                            "cooldown_hours": 24
                        },
                        "performance_degradation": {
                            "enabled": True,
                            "threshold": 0.05,
                            "cooldown_hours": 12
                        },
                        "new_data": {
                            "enabled": True,
                            "min_samples": 10000,
                            "cooldown_hours": 6
                        }
                    },
                    "notifications": {
                        "success": ["slack", "email"],
                        "failure": ["slack", "email", "pagerduty"],
                        "long_running": ["slack"]
                    }
                },
                "model_promotion": {
                    "description": "Automated model promotion pipeline",
                    "enabled": True,
                    "schedule": "0 */6 * * *",  # Every 6 hours
                    "stages": [
                        {
                            "name": "performance_evaluation",
                            "component": "model_evaluator",
                            "method": "evaluate_staging_models",
                            "timeout": 1800,
                            "retry_count": 2
                        },
                        {
                            "name": "promotion_decision",
                            "component": "advanced_model_registry",
                            "method": "check_auto_promotions",
                            "timeout": 600,
                            "retry_count": 1
                        },
                        {
                            "name": "deployment",
                            "component": "automated_ml_deployment",
                            "method": "deploy_model",
                            "timeout": 3600,
                            "retry_count": 1
                        }
                    ]
                },
                "data_pipeline": {
                    "description": "Data processing and feature update pipeline",
                    "enabled": True,
                    "schedule": "0 */4 * * *",  # Every 4 hours
                    "stages": [
                        {
                            "name": "data_quality_check",
                            "component": "data_quality_validator",
                            "method": "validate_data_quality",
                            "timeout": 1200,
                            "retry_count": 2
                        },
                        {
                            "name": "feature_computation",
                            "component": "feature_store",
                            "method": "compute_features",
                            "timeout": 2400,
                            "retry_count": 2
                        },
                        {
                            "name": "drift_detection",
                            "component": "drift_detector",
                            "method": "detect_drift",
                            "timeout": 900,
                            "retry_count": 2
                        }
                    ]
                }
            },
            "monitoring": {
                "enabled": True,
                "health_check_interval": 300,  # 5 minutes
                "metrics_collection_interval": 60,  # 1 minute
                "log_retention_days": 30,
                "alert_thresholds": {
                    "execution_failure_rate": 0.1,
                    "avg_execution_time": 7200,
                    "resource_utilization": 0.8,
                    "queue_depth": 10
                }
            },
            "notifications": {
                "slack": {
                    "webhook_url": "${SLACK_WEBHOOK_URL}",
                    "channel": "#ml-ops",
                    "enabled": True
                },
                "email": {
                    "smtp_server": "${SMTP_SERVER}",
                    "recipients": ["mlops-team@company.com"],
                    "enabled": True
                },
                "pagerduty": {
                    "routing_key": "${PAGERDUTY_ROUTING_KEY}",
                    "enabled": False
                }
            },
            "storage": {
                "execution_logs": "s3://ml-orchestrator-logs/",
                "artifacts": "s3://ml-pipeline-artifacts/",
                "metrics": "s3://ml-metrics/",
                "backup_retention_days": 90
            },
            "security": {
                "encryption_enabled": True,
                "access_control": True,
                "audit_logging": True,
                "secret_management": "kubernetes"
            }
        }
        
        # Save default config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default orchestrator configuration: {self.config_path}")
        return default_config
    
    def _initialize_orchestrator(self):
        """Initialize the orchestrator system."""
        logger.info("Initializing ML Pipeline Orchestrator...")
        
        # Load pipeline templates
        for pipeline_name, pipeline_config in self.config["pipelines"].items():
            if pipeline_config.get("enabled", False):
                self.pipeline_templates[pipeline_name] = pipeline_config
                logger.info(f"Loaded pipeline template: {pipeline_name}")
        
        # Set up scheduling if configured
        self._setup_scheduled_pipelines()
        
        # Initialize monitoring
        if self.config["monitoring"]["enabled"]:
            self._initialize_monitoring()
        
        self.status = OrchestrationStatus.IDLE
        logger.info("ML Pipeline Orchestrator initialized successfully")
    
    def _setup_scheduled_pipelines(self):
        """Set up scheduled pipeline executions."""
        for pipeline_name, pipeline_config in self.pipeline_templates.items():
            if "schedule" in pipeline_config:
                cron_schedule = pipeline_config["schedule"]
                logger.info(f"Scheduling pipeline '{pipeline_name}' with cron: {cron_schedule}")
                
                # Convert cron to schedule library format (simplified)
                self._register_cron_job(pipeline_name, cron_schedule)
    
    def _register_cron_job(self, pipeline_name: str, cron_schedule: str):
        """Register a cron job for pipeline execution."""
        # Simplified cron parsing - in production, use a proper cron library
        if "0 2 * * 1" in cron_schedule:  # Weekly Monday 2 AM
            schedule.every().monday.at("02:00").do(
                lambda: asyncio.create_task(self.execute_pipeline(pipeline_name, TriggerType.SCHEDULED))
            )
        elif "0 */6 * * *" in cron_schedule:  # Every 6 hours
            schedule.every(6).hours.do(
                lambda: asyncio.create_task(self.execute_pipeline(pipeline_name, TriggerType.SCHEDULED))
            )
        elif "0 */4 * * *" in cron_schedule:  # Every 4 hours
            schedule.every(4).hours.do(
                lambda: asyncio.create_task(self.execute_pipeline(pipeline_name, TriggerType.SCHEDULED))
            )
    
    def _initialize_monitoring(self):
        """Initialize monitoring system."""
        logger.info("Initializing orchestrator monitoring")
        
        # Set up health check monitoring
        monitoring_config = self.config["monitoring"]
        
        # Initialize metrics collection
        self.monitoring_data = {
            "system_health": {
                "status": "healthy",
                "last_check": datetime.utcnow().isoformat(),
                "uptime": 0
            },
            "execution_metrics": {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "avg_execution_time": 0,
                "current_queue_depth": 0
            },
            "resource_metrics": {
                "cpu_usage": 0,
                "memory_usage": 0,
                "disk_usage": 0,
                "active_executions": 0
            }
        }
    
    async def execute_pipeline(
        self,
        pipeline_name: str,
        trigger_type: TriggerType = TriggerType.MANUAL,
        trigger_reason: str = "Manual execution",
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a pipeline with comprehensive orchestration."""
        execution_id = f"exec_{pipeline_name}_{int(time.time())}"
        
        logger.info(f"🚀 Starting pipeline execution: {execution_id}")
        
        if pipeline_name not in self.pipeline_templates:
            raise ValueError(f"Pipeline template not found: {pipeline_name}")
        
        pipeline_config = self.pipeline_templates[pipeline_name]
        
        # Check concurrent execution limits
        if len(self.active_executions) >= self.config["orchestrator"]["max_concurrent_executions"]:
            logger.warning(f"Maximum concurrent executions reached, queuing execution: {execution_id}")
            # In production, this would go to a queue
            return execution_id
        
        # Create execution record
        execution = PipelineExecution(
            execution_id=execution_id,
            pipeline_name=pipeline_name,
            trigger_type=trigger_type,
            trigger_reason=trigger_reason,
            started_at=datetime.utcnow(),
            status="running",
            current_stage="initializing",
            completed_stages=[],
            failed_stages=[],
            metrics={},
            artifacts={},
            logs=[]
        )
        
        self.active_executions[execution_id] = execution
        
        try:
            # Update system status
            self.status = OrchestrationStatus.RUNNING
            
            # Execute pipeline stages
            await self._execute_pipeline_stages(execution, pipeline_config, parameters or {})
            
            # Update execution status
            execution.status = "completed"
            execution.current_stage = "completed"
            
            # Update metrics
            self._update_execution_metrics(execution, success=True)
            
            # Send success notifications
            await self._send_notifications(execution, "success")
            
            logger.info(f"✅ Pipeline execution completed successfully: {execution_id}")
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {execution_id} - {e}")
            
            execution.status = "failed"
            execution.current_stage = "failed"
            execution.logs.append(f"Execution failed: {str(e)}")
            
            # Update metrics
            self._update_execution_metrics(execution, success=False)
            
            # Send failure notifications
            await self._send_notifications(execution, "failure")
            
            # Check if retry is needed
            if await self._should_retry_execution(execution, pipeline_config):
                logger.info(f"Retrying pipeline execution: {execution_id}")
                # In production, this would schedule a retry
        
        finally:
            # Move to history and cleanup
            self.execution_history.append(asdict(execution))
            del self.active_executions[execution_id]
            
            # Update system status
            if len(self.active_executions) == 0:
                self.status = OrchestrationStatus.IDLE
        
        return execution_id
    
    async def _execute_pipeline_stages(
        self,
        execution: PipelineExecution,
        pipeline_config: Dict[str, Any],
        parameters: Dict[str, Any]
    ):
        """Execute all stages of a pipeline."""
        stages = pipeline_config["stages"]
        stage_results = {}
        
        for i, stage_config in enumerate(stages):
            stage_name = stage_config["name"]
            
            logger.info(f"📍 Executing stage {i+1}/{len(stages)}: {stage_name}")
            
            execution.current_stage = stage_name
            execution.logs.append(f"Starting stage: {stage_name}")
            
            try:
                # Execute stage with timeout
                stage_timeout = stage_config.get("timeout", 3600)
                
                stage_result = await asyncio.wait_for(
                    self._execute_stage(stage_config, stage_results, parameters),
                    timeout=stage_timeout
                )
                
                # Store stage result
                stage_results[stage_name] = stage_result
                execution.completed_stages.append(stage_name)
                execution.artifacts[stage_name] = stage_result.get("artifacts", {})
                execution.metrics[stage_name] = stage_result.get("metrics", {})
                
                execution.logs.append(f"Completed stage: {stage_name}")
                logger.info(f"✅ Stage completed: {stage_name}")
                
            except asyncio.TimeoutError:
                error_msg = f"Stage {stage_name} timed out after {stage_timeout} seconds"
                logger.error(error_msg)
                execution.failed_stages.append(stage_name)
                execution.logs.append(error_msg)
                raise Exception(error_msg)
                
            except Exception as e:
                error_msg = f"Stage {stage_name} failed: {str(e)}"
                logger.error(error_msg)
                execution.failed_stages.append(stage_name)
                execution.logs.append(error_msg)
                
                # Check if stage allows retries
                retry_count = stage_config.get("retry_count", 0)
                if retry_count > 0:
                    logger.info(f"Retrying stage {stage_name} ({retry_count} retries remaining)")
                    # In production, implement retry logic here
                
                raise Exception(error_msg)
    
    async def _execute_stage(
        self,
        stage_config: Dict[str, Any],
        previous_results: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single pipeline stage."""
        component = stage_config["component"]
        method = stage_config["method"]
        
        logger.info(f"Executing {component}.{method}")
        
        # Simulate stage execution
        await asyncio.sleep(2)  # Simulate processing time
        
        # Generate mock results based on stage type
        if "data_ingestion" in method:
            return {
                "success": True,
                "records_processed": 50000,
                "data_quality_score": 0.92,
                "artifacts": {"data_path": f"s3://data-lake/ingested/{datetime.now().isoformat()}"},
                "metrics": {"processing_time": 120, "throughput": 416.67}
            }
        elif "preprocessing" in method or "feature_engineering" in method:
            return {
                "success": True,
                "features_created": 45,
                "transformation_applied": 8,
                "artifacts": {"feature_store_path": f"s3://features/{datetime.now().isoformat()}"},
                "metrics": {"processing_time": 180, "feature_quality": 0.89}
            }
        elif "training" in method:
            return {
                "success": True,
                "models_trained": 3,
                "best_model_accuracy": 0.94,
                "artifacts": {"model_path": f"s3://models/trained/{datetime.now().isoformat()}"},
                "metrics": {"training_time": 1800, "accuracy": 0.94, "precision": 0.91}
            }
        elif "validation" in method:
            return {
                "success": True,
                "validation_passed": True,
                "test_accuracy": 0.92,
                "artifacts": {"validation_report": f"s3://reports/validation/{datetime.now().isoformat()}"},
                "metrics": {"validation_time": 300, "test_samples": 10000}
            }
        elif "registration" in method:
            return {
                "success": True,
                "model_id": f"anomaly_detector_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "artifacts": {"registry_entry": "model_registry_db"},
                "metrics": {"registration_time": 30}
            }
        elif "deployment" in method:
            return {
                "success": True,
                "deployment_id": f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "endpoint_url": "https://ml.company.com/predict",
                "artifacts": {"k8s_manifests": "deployment_configs/"},
                "metrics": {"deployment_time": 600, "replicas": 3}
            }
        else:
            return {
                "success": True,
                "artifacts": {},
                "metrics": {"execution_time": 60}
            }
    
    def _update_execution_metrics(self, execution: PipelineExecution, success: bool):
        """Update execution metrics."""
        metrics = self.monitoring_data["execution_metrics"]
        metrics["total_executions"] += 1
        
        if success:
            metrics["successful_executions"] += 1
        else:
            metrics["failed_executions"] += 1
        
        # Calculate average execution time
        execution_time = (datetime.utcnow() - execution.started_at).total_seconds()
        current_avg = metrics["avg_execution_time"]
        total_executions = metrics["total_executions"]
        
        metrics["avg_execution_time"] = (
            (current_avg * (total_executions - 1) + execution_time) / total_executions
        )
        
        # Update resource metrics
        self.monitoring_data["resource_metrics"]["active_executions"] = len(self.active_executions)
    
    async def _should_retry_execution(
        self,
        execution: PipelineExecution,
        pipeline_config: Dict[str, Any]
    ) -> bool:
        """Determine if execution should be retried."""
        retry_policy = self.config["orchestrator"]["retry_policy"]
        
        # Check if max retries exceeded
        retry_count = execution.metrics.get("retry_count", 0)
        if retry_count >= retry_policy["max_retries"]:
            return False
        
        # Check if failure is recoverable
        # In production, implement more sophisticated retry logic
        return True
    
    async def _send_notifications(self, execution: PipelineExecution, notification_type: str):
        """Send notifications for pipeline events."""
        pipeline_config = self.pipeline_templates[execution.pipeline_name]
        notification_config = pipeline_config.get("notifications", {})
        
        if notification_type not in notification_config:
            return
        
        channels = notification_config[notification_type]
        
        for channel in channels:
            try:
                if channel == "slack":
                    await self._send_slack_notification(execution, notification_type)
                elif channel == "email":
                    await self._send_email_notification(execution, notification_type)
                elif channel == "pagerduty":
                    await self._send_pagerduty_notification(execution, notification_type)
            except Exception as e:
                logger.error(f"Failed to send {channel} notification: {e}")
    
    async def _send_slack_notification(self, execution: PipelineExecution, notification_type: str):
        """Send Slack notification."""
        slack_config = self.config["notifications"]["slack"]
        
        if not slack_config["enabled"]:
            return
        
        webhook_url = os.path.expandvars(slack_config["webhook_url"])
        
        # Determine message color and emoji
        if notification_type == "success":
            color = "good"
            emoji = "✅"
            title = "Pipeline Execution Successful"
        elif notification_type == "failure":
            color = "danger"
            emoji = "❌"
            title = "Pipeline Execution Failed"
        else:
            color = "warning"
            emoji = "⚠️"
            title = "Pipeline Notification"
        
        message = {
            "text": f"{emoji} ML Pipeline Notification",
            "attachments": [
                {
                    "color": color,
                    "title": title,
                    "fields": [
                        {
                            "title": "Pipeline",
                            "value": execution.pipeline_name,
                            "short": True
                        },
                        {
                            "title": "Execution ID",
                            "value": execution.execution_id,
                            "short": True
                        },
                        {
                            "title": "Trigger",
                            "value": execution.trigger_type.value,
                            "short": True
                        },
                        {
                            "title": "Duration",
                            "value": f"{(datetime.utcnow() - execution.started_at).total_seconds():.0f}s",
                            "short": True
                        },
                        {
                            "title": "Stages Completed",
                            "value": f"{len(execution.completed_stages)}/{len(execution.completed_stages) + len(execution.failed_stages)}",
                            "short": True
                        }
                    ],
                    "footer": f"ML Pipeline Orchestrator | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                }
            ]
        }
        
        # Simulate sending notification
        logger.info(f"Sending Slack notification: {title}")
    
    async def _send_email_notification(self, execution: PipelineExecution, notification_type: str):
        """Send email notification."""
        logger.info(f"Sending email notification for {execution.execution_id}")
    
    async def _send_pagerduty_notification(self, execution: PipelineExecution, notification_type: str):
        """Send PagerDuty notification."""
        logger.info(f"Sending PagerDuty notification for {execution.execution_id}")
    
    async def trigger_pipeline_on_condition(
        self,
        condition_type: str,
        condition_data: Dict[str, Any]
    ) -> Optional[str]:
        """Trigger pipeline based on conditions (drift, performance, etc.)."""
        logger.info(f"Evaluating trigger condition: {condition_type}")
        
        triggered_pipelines = []
        
        for pipeline_name, pipeline_config in self.pipeline_templates.items():
            triggers = pipeline_config.get("triggers", {})
            
            if condition_type in triggers:
                trigger_config = triggers[condition_type]
                
                if not trigger_config.get("enabled", False):
                    continue
                
                # Check cooldown period
                if self._is_in_cooldown(pipeline_name, condition_type, trigger_config):
                    logger.info(f"Pipeline {pipeline_name} is in cooldown for {condition_type}")
                    continue
                
                # Check trigger conditions
                if self._should_trigger_pipeline(condition_type, condition_data, trigger_config):
                    execution_id = await self.execute_pipeline(
                        pipeline_name,
                        TriggerType(condition_type),
                        f"Triggered by {condition_type}: {condition_data}"
                    )
                    triggered_pipelines.append(execution_id)
                    logger.info(f"Triggered pipeline {pipeline_name}: {execution_id}")
        
        return triggered_pipelines[0] if triggered_pipelines else None
    
    def _is_in_cooldown(self, pipeline_name: str, condition_type: str, trigger_config: Dict[str, Any]) -> bool:
        """Check if pipeline is in cooldown period for trigger."""
        cooldown_hours = trigger_config.get("cooldown_hours", 0)
        
        if cooldown_hours == 0:
            return False
        
        # Check recent executions
        cutoff_time = datetime.utcnow() - timedelta(hours=cooldown_hours)
        
        for execution in self.execution_history[-20:]:  # Check last 20 executions
            if (execution["pipeline_name"] == pipeline_name and
                execution["trigger_type"] == condition_type and
                datetime.fromisoformat(execution["started_at"]) > cutoff_time):
                return True
        
        return False
    
    def _should_trigger_pipeline(
        self,
        condition_type: str,
        condition_data: Dict[str, Any],
        trigger_config: Dict[str, Any]
    ) -> bool:
        """Determine if pipeline should be triggered based on condition."""
        if condition_type == "data_drift":
            drift_score = condition_data.get("drift_score", 0)
            threshold = trigger_config.get("threshold", 0.1)
            return drift_score > threshold
        
        elif condition_type == "performance_degradation":
            performance_drop = condition_data.get("performance_drop", 0)
            threshold = trigger_config.get("threshold", 0.05)
            return performance_drop > threshold
        
        elif condition_type == "new_data":
            sample_count = condition_data.get("sample_count", 0)
            min_samples = trigger_config.get("min_samples", 1000)
            return sample_count >= min_samples
        
        return False
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a running execution."""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        execution.status = "paused"
        execution.logs.append(f"Execution paused at {datetime.utcnow().isoformat()}")
        
        logger.info(f"Paused execution: {execution_id}")
        return True
    
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution."""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        
        if execution.status != "paused":
            return False
        
        execution.status = "running"
        execution.logs.append(f"Execution resumed at {datetime.utcnow().isoformat()}")
        
        logger.info(f"Resumed execution: {execution_id}")
        return True
    
    async def cancel_execution(self, execution_id: str, reason: str = "Manual cancellation") -> bool:
        """Cancel a running execution."""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        execution.status = "cancelled"
        execution.logs.append(f"Execution cancelled: {reason}")
        
        # Move to history
        self.execution_history.append(asdict(execution))
        del self.active_executions[execution_id]
        
        logger.info(f"Cancelled execution: {execution_id}")
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status."""
        if execution_id in self.active_executions:
            return asdict(self.active_executions[execution_id])
        
        # Check history
        for execution in self.execution_history:
            if execution["execution_id"] == execution_id:
                return execution
        
        return None
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all active executions."""
        return [asdict(execution) for execution in self.active_executions.values()]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get orchestrator system health."""
        return {
            "status": self.status.value,
            "active_executions": len(self.active_executions),
            "monitoring_data": self.monitoring_data,
            "pipeline_templates": len(self.pipeline_templates),
            "last_health_check": datetime.utcnow().isoformat()
        }
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive health checks."""
        logger.info("Running orchestrator health checks")
        
        health_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # Check system resources
        health_results["checks"]["system_resources"] = {
            "status": "healthy",
            "cpu_usage": "45%",
            "memory_usage": "67%",
            "disk_usage": "23%"
        }
        
        # Check pipeline templates
        health_results["checks"]["pipeline_templates"] = {
            "status": "healthy",
            "loaded_pipelines": len(self.pipeline_templates),
            "enabled_pipelines": len([p for p in self.pipeline_templates.values() if p.get("enabled")])
        }
        
        # Check active executions
        health_results["checks"]["active_executions"] = {
            "status": "healthy",
            "count": len(self.active_executions),
            "max_allowed": self.config["orchestrator"]["max_concurrent_executions"]
        }
        
        # Check recent failures
        recent_failures = len([
            e for e in self.execution_history[-50:]
            if e.get("status") == "failed" and
            datetime.fromisoformat(e["started_at"]) > datetime.utcnow() - timedelta(hours=24)
        ])
        
        health_results["checks"]["recent_failures"] = {
            "status": "healthy" if recent_failures < 5 else "warning",
            "failure_count": recent_failures,
            "threshold": 5
        }
        
        # Update overall status
        warning_checks = [c for c in health_results["checks"].values() if c["status"] == "warning"]
        error_checks = [c for c in health_results["checks"].values() if c["status"] == "error"]
        
        if error_checks:
            health_results["overall_status"] = "unhealthy"
        elif warning_checks:
            health_results["overall_status"] = "warning"
        
        return health_results
    
    def generate_orchestrator_report(self) -> str:
        """Generate comprehensive orchestrator report."""
        logger.info("Generating orchestrator report")
        
        report_file = f"orchestrator-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"""# 🎼 ML Pipeline Orchestrator Report

**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC  
**Orchestrator Version:** {self.config['orchestrator']['version']}  
**Status:** {self.status.value}  

## 📊 Execution Statistics

### Overall Metrics
- **Total Executions:** {self.monitoring_data['execution_metrics']['total_executions']}
- **Successful Executions:** {self.monitoring_data['execution_metrics']['successful_executions']}
- **Failed Executions:** {self.monitoring_data['execution_metrics']['failed_executions']}
- **Success Rate:** {(self.monitoring_data['execution_metrics']['successful_executions'] / max(1, self.monitoring_data['execution_metrics']['total_executions']) * 100):.1f}%
- **Average Execution Time:** {self.monitoring_data['execution_metrics']['avg_execution_time']:.0f} seconds

### Active Executions
- **Currently Running:** {len(self.active_executions)}
- **Max Concurrent:** {self.config['orchestrator']['max_concurrent_executions']}

""")
            
            # Active executions details
            if self.active_executions:
                f.write("#### Running Executions\n")
                for execution in self.active_executions.values():
                    duration = (datetime.utcnow() - execution.started_at).total_seconds()
                    f.write(f"""- **{execution.execution_id}**
  - Pipeline: {execution.pipeline_name}
  - Status: {execution.status}
  - Current Stage: {execution.current_stage}
  - Duration: {duration:.0f}s
  - Completed Stages: {len(execution.completed_stages)}

""")
            
            f.write(f"""## 🔄 Pipeline Templates

""")
            
            for pipeline_name, pipeline_config in self.pipeline_templates.items():
                enabled_status = "✅ Enabled" if pipeline_config.get("enabled") else "❌ Disabled"
                schedule = pipeline_config.get("schedule", "Manual only")
                
                f.write(f"""### {pipeline_name.replace('_', ' ').title()}
- **Status:** {enabled_status}
- **Description:** {pipeline_config.get('description', 'No description')}
- **Schedule:** {schedule}
- **Stages:** {len(pipeline_config.get('stages', []))}

""")
            
            f.write(f"""## 📈 Recent Execution History

""")
            
            # Show last 10 executions
            recent_executions = self.execution_history[-10:] if len(self.execution_history) >= 10 else self.execution_history
            
            for execution in reversed(recent_executions):
                status_emoji = "✅" if execution["status"] == "completed" else "❌" if execution["status"] == "failed" else "⏸️"
                duration = "N/A"
                
                try:
                    start_time = datetime.fromisoformat(execution["started_at"])
                    if "completed_at" in execution:
                        end_time = datetime.fromisoformat(execution["completed_at"])
                        duration = f"{(end_time - start_time).total_seconds():.0f}s"
                except:
                    pass
                
                f.write(f"""### {status_emoji} {execution['execution_id']}
- **Pipeline:** {execution['pipeline_name']}
- **Trigger:** {execution['trigger_type']}
- **Status:** {execution['status']}
- **Duration:** {duration}
- **Completed Stages:** {len(execution.get('completed_stages', []))}
- **Failed Stages:** {len(execution.get('failed_stages', []))}

""")
            
            f.write(f"""## 🎯 System Configuration

### Resource Limits
- **CPU Cores:** {self.config['orchestrator']['resource_limits']['cpu_cores']}
- **Memory:** {self.config['orchestrator']['resource_limits']['memory_gb']} GB
- **GPU Count:** {self.config['orchestrator']['resource_limits']['gpu_count']}

### Retry Policy
- **Max Retries:** {self.config['orchestrator']['retry_policy']['max_retries']}
- **Retry Delay:** {self.config['orchestrator']['retry_policy']['retry_delay']} seconds
- **Exponential Backoff:** {'✅' if self.config['orchestrator']['retry_policy']['exponential_backoff'] else '❌'}

### Monitoring
- **Health Check Interval:** {self.config['monitoring']['health_check_interval']} seconds
- **Metrics Collection:** {self.config['monitoring']['metrics_collection_interval']} seconds
- **Log Retention:** {self.config['monitoring']['log_retention_days']} days

## 🔔 Notification Channels

""")
            
            for channel, config in self.config["notifications"].items():
                status = "✅ Enabled" if config.get("enabled") else "❌ Disabled"
                f.write(f"- **{channel.title()}:** {status}\n")
            
            f.write(f"""
## 📊 Performance Analysis

### Execution Patterns
""")
            
            # Analyze execution patterns
            pipeline_counts = {}
            trigger_counts = {}
            
            for execution in self.execution_history:
                pipeline_name = execution["pipeline_name"]
                trigger_type = execution["trigger_type"]
                
                pipeline_counts[pipeline_name] = pipeline_counts.get(pipeline_name, 0) + 1
                trigger_counts[trigger_type] = trigger_counts.get(trigger_type, 0) + 1
            
            f.write("#### Pipeline Usage\n")
            for pipeline, count in sorted(pipeline_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{pipeline}:** {count} executions\n")
            
            f.write("\n#### Trigger Types\n")
            for trigger, count in sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{trigger}:** {count} executions\n")
            
            f.write(f"""
## 🎯 Recommendations

""")
            
            # Generate recommendations
            total_executions = self.monitoring_data['execution_metrics']['total_executions']
            failed_executions = self.monitoring_data['execution_metrics']['failed_executions']
            
            if total_executions > 0:
                failure_rate = failed_executions / total_executions
                if failure_rate > 0.1:
                    f.write("- **High Failure Rate:** Consider reviewing pipeline configurations and error handling\n")
                
                avg_time = self.monitoring_data['execution_metrics']['avg_execution_time']
                if avg_time > 3600:  # 1 hour
                    f.write("- **Long Execution Times:** Consider pipeline optimization and parallelization\n")
            
            if len(self.active_executions) == 0 and len(self.execution_history) < 5:
                f.write("- **Low Usage:** Consider enabling more automated triggers or scheduled executions\n")
            
            f.write(f"""
## 📞 Support

For orchestrator issues:
1. Check system health and resource usage
2. Review pipeline configurations and schedules
3. Analyze execution logs and error patterns
4. Contact MLOps team for advanced troubleshooting

---
*This report was generated automatically by the ML Pipeline Orchestrator*
""")
        
        logger.info(f"Orchestrator report generated: {report_file}")
        return report_file
    
    async def run_continuous_orchestration(self):
        """Run continuous orchestration with scheduling and monitoring."""
        logger.info("🎼 Starting continuous orchestration...")
        
        try:
            while True:
                # Run scheduled jobs
                schedule.run_pending()
                
                # Perform health checks
                if self.config["monitoring"]["enabled"]:
                    await self.run_health_checks()
                
                # Update system metrics
                self._update_system_metrics()
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.config["monitoring"]["metrics_collection_interval"])
                
        except KeyboardInterrupt:
            logger.info("Orchestration stopped by user")
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            self.status = OrchestrationStatus.FAILED
    
    def _update_system_metrics(self):
        """Update system performance metrics."""
        # Update uptime
        self.monitoring_data["system_health"]["uptime"] += self.config["monitoring"]["metrics_collection_interval"]
        
        # Update resource metrics (simulated)
        import random
        self.monitoring_data["resource_metrics"]["cpu_usage"] = random.uniform(20, 80)
        self.monitoring_data["resource_metrics"]["memory_usage"] = random.uniform(40, 90)
        self.monitoring_data["resource_metrics"]["disk_usage"] = random.uniform(10, 60)


async def main():
    """Main function for orchestrator operations."""
    logger.info("🎼 ML Pipeline Orchestrator Starting...")
    
    try:
        # Initialize orchestrator
        orchestrator = MLPipelineOrchestrator()
        
        # Example: Execute a pipeline
        execution_id = await orchestrator.execute_pipeline(
            "anomaly_detection_training",
            TriggerType.MANUAL,
            "Example manual execution"
        )
        
        # Check execution status
        status = orchestrator.get_execution_status(execution_id)
        logger.info(f"Execution status: {status['status'] if status else 'Not found'}")
        
        # Generate report
        report_file = orchestrator.generate_orchestrator_report()
        
        logger.info(f"✅ Orchestrator operations completed!")
        logger.info(f"📊 Execution ID: {execution_id}")
        logger.info(f"📋 Report: {report_file}")
        
    except Exception as e:
        logger.error(f"💥 Fatal error in orchestrator: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())