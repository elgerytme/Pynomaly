"""Advanced workflow orchestration for enterprise anomaly detection pipelines."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WorkflowEngine(str, Enum):
    """Supported workflow orchestration engines."""
    
    AIRFLOW = "airflow"
    PREFECT = "prefect"
    KUBERNETES = "kubernetes"
    ARGO_WORKFLOWS = "argo_workflows"
    TEMPORAL = "temporal"
    CELERY = "celery"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    SKIPPED = "skipped"
    UPSTREAM_FAILED = "upstream_failed"


class TaskType(str, Enum):
    """Types of workflow tasks."""
    
    DATA_INGESTION = "data_ingestion"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    ANOMALY_DETECTION = "anomaly_detection"
    ALERTING = "alerting"
    DATA_EXPORT = "data_export"
    CLEANUP = "cleanup"
    VALIDATION = "validation"


class ScheduleType(str, Enum):
    """Workflow schedule types."""
    
    CRON = "cron"
    INTERVAL = "interval"
    EVENT_DRIVEN = "event_driven"
    MANUAL = "manual"
    SENSOR = "sensor"


@dataclass
class TaskConfig:
    """Configuration for individual workflow task."""
    
    task_id: str
    task_type: TaskType
    function_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    retry_delay: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    timeout: timedelta = field(default_factory=lambda: timedelta(hours=1))
    resources: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class WorkflowConfig:
    """Configuration for entire workflow."""
    
    workflow_id: str
    workflow_name: str
    description: str
    engine: WorkflowEngine
    schedule: Optional[str] = None
    schedule_type: ScheduleType = ScheduleType.MANUAL
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    catchup: bool = False
    max_active_runs: int = 1
    default_args: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    tasks: List[TaskConfig] = field(default_factory=list)


@dataclass
class WorkflowExecution:
    """Workflow execution tracking."""
    
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    triggered_by: str = "manual"
    parameters: Dict[str, Any] = field(default_factory=dict)
    task_executions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class WorkflowOrchestrator(ABC):
    """Abstract base class for workflow orchestrators."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflows: Dict[str, WorkflowConfig] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
    
    @abstractmethod
    async def create_workflow(self, workflow_config: WorkflowConfig) -> bool:
        """Create a new workflow."""
        pass
    
    @abstractmethod
    async def execute_workflow(
        self,
        workflow_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a workflow and return execution ID."""
        pass
    
    @abstractmethod
    async def get_workflow_status(self, execution_id: str) -> WorkflowStatus:
        """Get status of workflow execution."""
        pass
    
    @abstractmethod
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel running workflow."""
        pass
    
    @abstractmethod
    async def list_workflows(self) -> List[str]:
        """List all available workflows."""
        pass


class AirflowOrchestrator(WorkflowOrchestrator):
    """Apache Airflow workflow orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.airflow_client = None
        self.dag_folder = config.get("dag_folder", "/opt/airflow/dags")
    
    async def _initialize_client(self):
        """Initialize Airflow client."""
        try:
            from airflow_client.client import ApiClient, Configuration
            from airflow_client.client.api import dag_api, dag_run_api
            
            configuration = Configuration(
                host=self.config.get("host", "http://localhost:8080/api/v1"),
                username=self.config.get("username"),
                password=self.config.get("password")
            )
            
            self.airflow_client = ApiClient(configuration)
            self.dag_api = dag_api.DAGApi(self.airflow_client)
            self.dag_run_api = dag_run_api.DAGRunApi(self.airflow_client)
            
        except ImportError:
            logger.warning("Airflow client not available, using mock implementation")
            self.airflow_client = None
    
    async def create_workflow(self, workflow_config: WorkflowConfig) -> bool:
        """Create Airflow DAG from workflow configuration."""
        try:
            dag_code = self._generate_airflow_dag(workflow_config)
            
            # Write DAG file
            dag_file_path = f"{self.dag_folder}/{workflow_config.workflow_id}.py"
            with open(dag_file_path, 'w') as f:
                f.write(dag_code)
            
            self.workflows[workflow_config.workflow_id] = workflow_config
            logger.info(f"Created Airflow DAG: {workflow_config.workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Airflow workflow: {e}")
            return False
    
    def _generate_airflow_dag(self, config: WorkflowConfig) -> str:
        """Generate Airflow DAG Python code."""
        dag_template = f"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from pynomaly.infrastructure.orchestration.task_operators import *

default_args = {{
    'owner': 'pynomaly',
    'depends_on_past': False,
    'start_date': datetime({config.start_date.year}, {config.start_date.month}, {config.start_date.day}) if {config.start_date is not None} else datetime.now(),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}}

dag = DAG(
    '{config.workflow_id}',
    default_args=default_args,
    description='{config.description}',
    schedule_interval={'None' if not config.schedule else f"'{config.schedule}'"},
    catchup={config.catchup},
    max_active_runs={config.max_active_runs},
    tags={config.tags}
)

# Task definitions
"""
        
        # Add tasks
        for task in config.tasks:
            task_code = self._generate_airflow_task(task)
            dag_template += task_code
        
        # Add dependencies
        dependency_code = "\n# Task dependencies\n"
        for task in config.tasks:
            if task.dependencies:
                deps = " >> ".join(task.dependencies)
                dependency_code += f"{deps} >> {task.task_id}\n"
        
        dag_template += dependency_code
        
        return dag_template
    
    def _generate_airflow_task(self, task: TaskConfig) -> str:
        """Generate Airflow task code."""
        if task.task_type == TaskType.DATA_INGESTION:
            return f"""
{task.task_id} = PythonOperator(
    task_id='{task.task_id}',
    python_callable=data_ingestion_task,
    op_kwargs={task.parameters},
    retries={task.retry_count},
    retry_delay={task.retry_delay},
    execution_timeout={task.timeout},
    dag=dag
)
"""
        elif task.task_type == TaskType.ANOMALY_DETECTION:
            return f"""
{task.task_id} = PythonOperator(
    task_id='{task.task_id}',
    python_callable=anomaly_detection_task,
    op_kwargs={task.parameters},
    retries={task.retry_count},
    retry_delay={task.retry_delay},
    execution_timeout={task.timeout},
    dag=dag
)
"""
        else:
            return f"""
{task.task_id} = PythonOperator(
    task_id='{task.task_id}',
    python_callable={task.function_name},
    op_kwargs={task.parameters},
    retries={task.retry_count},
    retry_delay={task.retry_delay},
    execution_timeout={task.timeout},
    dag=dag
)
"""
    
    async def execute_workflow(
        self,
        workflow_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute Airflow DAG."""
        if not self.airflow_client:
            await self._initialize_client()
        
        try:
            if self.airflow_client:
                # Use real Airflow API
                from airflow_client.client.model.dag_run import DAGRun
                
                dag_run = DAGRun(
                    dag_run_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    conf=parameters or {}
                )
                
                response = self.dag_run_api.post_dag_run(workflow_id, dag_run)
                execution_id = response.dag_run_id
            else:
                # Mock execution
                execution_id = f"mock_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                execution = WorkflowExecution(
                    execution_id=execution_id,
                    workflow_id=workflow_id,
                    status=WorkflowStatus.RUNNING,
                    start_time=datetime.now(),
                    parameters=parameters or {}
                )
                self.executions[execution_id] = execution
            
            logger.info(f"Started Airflow workflow execution: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to execute Airflow workflow: {e}")
            raise
    
    async def get_workflow_status(self, execution_id: str) -> WorkflowStatus:
        """Get Airflow DAG run status."""
        if execution_id.startswith("mock_"):
            # Mock status for testing
            execution = self.executions.get(execution_id)
            return execution.status if execution else WorkflowStatus.FAILED
        
        if not self.airflow_client:
            await self._initialize_client()
        
        try:
            if self.airflow_client:
                # Extract workflow_id from execution_id
                workflow_id = execution_id.split("_")[1] if "_" in execution_id else execution_id
                
                response = self.dag_run_api.get_dag_run(workflow_id, execution_id)
                
                status_mapping = {
                    "running": WorkflowStatus.RUNNING,
                    "success": WorkflowStatus.SUCCESS,
                    "failed": WorkflowStatus.FAILED,
                    "queued": WorkflowStatus.PENDING
                }
                
                return status_mapping.get(response.state, WorkflowStatus.FAILED)
            else:
                return WorkflowStatus.FAILED
                
        except Exception as e:
            logger.error(f"Failed to get Airflow workflow status: {e}")
            return WorkflowStatus.FAILED
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel Airflow DAG run."""
        if execution_id.startswith("mock_"):
            execution = self.executions.get(execution_id)
            if execution:
                execution.status = WorkflowStatus.CANCELLED
                return True
            return False
        
        if not self.airflow_client:
            await self._initialize_client()
        
        try:
            if self.airflow_client:
                # Extract workflow_id from execution_id
                workflow_id = execution_id.split("_")[1] if "_" in execution_id else execution_id
                
                # Airflow doesn't have direct cancel API, would need to implement
                logger.warning("Airflow workflow cancellation not implemented")
                return False
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel Airflow workflow: {e}")
            return False
    
    async def list_workflows(self) -> List[str]:
        """List Airflow DAGs."""
        if not self.airflow_client:
            await self._initialize_client()
        
        try:
            if self.airflow_client:
                response = self.dag_api.get_dags()
                return [dag.dag_id for dag in response.dags]
            else:
                return list(self.workflows.keys())
                
        except Exception as e:
            logger.error(f"Failed to list Airflow workflows: {e}")
            return []


class PrefectOrchestrator(WorkflowOrchestrator):
    """Prefect workflow orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.prefect_client = None
    
    async def _initialize_client(self):
        """Initialize Prefect client."""
        try:
            from prefect import Client
            
            self.prefect_client = Client(
                api_server=self.config.get("api_server", "http://localhost:4200")
            )
            
        except ImportError:
            logger.warning("Prefect not available, using mock implementation")
            self.prefect_client = None
    
    async def create_workflow(self, workflow_config: WorkflowConfig) -> bool:
        """Create Prefect flow from workflow configuration."""
        try:
            flow_code = self._generate_prefect_flow(workflow_config)
            
            # In a real implementation, this would register the flow with Prefect
            self.workflows[workflow_config.workflow_id] = workflow_config
            logger.info(f"Created Prefect flow: {workflow_config.workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Prefect workflow: {e}")
            return False
    
    def _generate_prefect_flow(self, config: WorkflowConfig) -> str:
        """Generate Prefect flow code."""
        flow_template = f"""
from prefect import Flow, task, Parameter
from prefect.schedules import Schedule
from prefect.schedules.clocks import CronClock, IntervalClock
from datetime import timedelta
from pynomaly.infrastructure.orchestration.task_operators import *

# Task definitions
"""
        
        # Add task definitions
        for task in config.tasks:
            task_code = self._generate_prefect_task(task)
            flow_template += task_code
        
        # Add flow definition
        flow_template += f"""
# Flow definition
with Flow('{config.workflow_id}', description='{config.description}') as flow:
"""
        
        # Add task instantiation and dependencies
        for task in config.tasks:
            if task.dependencies:
                deps = ", ".join([f"{dep}_result" for dep in task.dependencies])
                flow_template += f"    {task.task_id}_result = {task.task_id}({deps})\n"
            else:
                flow_template += f"    {task.task_id}_result = {task.task_id}()\n"
        
        return flow_template
    
    def _generate_prefect_task(self, task: TaskConfig) -> str:
        """Generate Prefect task code."""
        return f"""
@task(
    name='{task.task_id}',
    max_retries={task.retry_count},
    retry_delay=timedelta(seconds={task.retry_delay.total_seconds()})
)
def {task.task_id}(*args, **kwargs):
    return {task.function_name}({task.parameters})

"""
    
    async def execute_workflow(
        self,
        workflow_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute Prefect flow."""
        if not self.prefect_client:
            await self._initialize_client()
        
        try:
            execution_id = f"prefect_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if self.prefect_client:
                # Real Prefect execution would go here
                pass
            
            # Mock execution
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.RUNNING,
                start_time=datetime.now(),
                parameters=parameters or {}
            )
            self.executions[execution_id] = execution
            
            logger.info(f"Started Prefect workflow execution: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to execute Prefect workflow: {e}")
            raise
    
    async def get_workflow_status(self, execution_id: str) -> WorkflowStatus:
        """Get Prefect flow run status."""
        execution = self.executions.get(execution_id)
        return execution.status if execution else WorkflowStatus.FAILED
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel Prefect flow run."""
        execution = self.executions.get(execution_id)
        if execution:
            execution.status = WorkflowStatus.CANCELLED
            return True
        return False
    
    async def list_workflows(self) -> List[str]:
        """List Prefect flows."""
        return list(self.workflows.keys())


class KubernetesOrchestrator(WorkflowOrchestrator):
    """Kubernetes Jobs workflow orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.k8s_client = None
        self.namespace = config.get("namespace", "default")
    
    async def _initialize_client(self):
        """Initialize Kubernetes client."""
        try:
            from kubernetes import client, config
            
            if self.config.get("kubeconfig_path"):
                config.load_kube_config(config_file=self.config["kubeconfig_path"])
            else:
                config.load_incluster_config()
            
            self.k8s_client = client.BatchV1Api()
            
        except ImportError:
            logger.warning("Kubernetes client not available, using mock implementation")
            self.k8s_client = None
    
    async def create_workflow(self, workflow_config: WorkflowConfig) -> bool:
        """Create Kubernetes Job workflow."""
        try:
            self.workflows[workflow_config.workflow_id] = workflow_config
            logger.info(f"Created Kubernetes workflow: {workflow_config.workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Kubernetes workflow: {e}")
            return False
    
    def _generate_k8s_job(self, task: TaskConfig, workflow_id: str) -> Dict[str, Any]:
        """Generate Kubernetes Job manifest."""
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"{workflow_id}-{task.task_id}",
                "namespace": self.namespace,
                "labels": {
                    "workflow": workflow_id,
                    "task": task.task_id,
                    "app": "pynomaly"
                }
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": task.task_id,
                            "image": self.config.get("image", "pynomaly:latest"),
                            "command": ["python", "-c"],
                            "args": [f"from pynomaly.infrastructure.orchestration.task_operators import {task.function_name}; {task.function_name}({task.parameters})"],
                            "env": [{"name": k, "value": v} for k, v in task.environment.items()],
                            "resources": task.resources
                        }],
                        "restartPolicy": "Never"
                    }
                },
                "backoffLimit": task.retry_count,
                "activeDeadlineSeconds": int(task.timeout.total_seconds())
            }
        }
    
    async def execute_workflow(
        self,
        workflow_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute Kubernetes workflow."""
        if not self.k8s_client:
            await self._initialize_client()
        
        try:
            execution_id = f"k8s_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            workflow_config = self.workflows[workflow_id]
            
            # Create jobs for each task
            for task in workflow_config.tasks:
                job_manifest = self._generate_k8s_job(task, workflow_id)
                
                if self.k8s_client:
                    self.k8s_client.create_namespaced_job(
                        namespace=self.namespace,
                        body=job_manifest
                    )
            
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.RUNNING,
                start_time=datetime.now(),
                parameters=parameters or {}
            )
            self.executions[execution_id] = execution
            
            logger.info(f"Started Kubernetes workflow execution: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to execute Kubernetes workflow: {e}")
            raise
    
    async def get_workflow_status(self, execution_id: str) -> WorkflowStatus:
        """Get Kubernetes workflow status."""
        execution = self.executions.get(execution_id)
        return execution.status if execution else WorkflowStatus.FAILED
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel Kubernetes workflow."""
        execution = self.executions.get(execution_id)
        if execution:
            execution.status = WorkflowStatus.CANCELLED
            return True
        return False
    
    async def list_workflows(self) -> List[str]:
        """List Kubernetes workflows."""
        return list(self.workflows.keys())


# Factory function
def create_workflow_orchestrator(engine: WorkflowEngine, config: Dict[str, Any]) -> WorkflowOrchestrator:
    """Create workflow orchestrator based on engine type."""
    
    orchestrator_map = {
        WorkflowEngine.AIRFLOW: AirflowOrchestrator,
        WorkflowEngine.PREFECT: PrefectOrchestrator,
        WorkflowEngine.KUBERNETES: KubernetesOrchestrator,
    }
    
    orchestrator_class = orchestrator_map.get(engine)
    if not orchestrator_class:
        raise ValueError(f"Unsupported workflow engine: {engine}")
    
    return orchestrator_class(config)


# Workflow manager for multiple orchestrators
class WorkflowManager:
    """Manages multiple workflow orchestrators."""
    
    def __init__(self):
        self.orchestrators: Dict[WorkflowEngine, WorkflowOrchestrator] = {}
        self.default_engine = WorkflowEngine.AIRFLOW
    
    async def add_orchestrator(
        self,
        engine: WorkflowEngine,
        config: Dict[str, Any]
    ) -> bool:
        """Add workflow orchestrator."""
        try:
            orchestrator = create_workflow_orchestrator(engine, config)
            self.orchestrators[engine] = orchestrator
            logger.info(f"Added {engine} orchestrator")
            return True
        except Exception as e:
            logger.error(f"Failed to add {engine} orchestrator: {e}")
            return False
    
    async def create_workflow(
        self,
        workflow_config: WorkflowConfig,
        engine: Optional[WorkflowEngine] = None
    ) -> bool:
        """Create workflow using specified or default engine."""
        target_engine = engine or workflow_config.engine or self.default_engine
        
        orchestrator = self.orchestrators.get(target_engine)
        if not orchestrator:
            raise ValueError(f"Orchestrator not available: {target_engine}")
        
        return await orchestrator.create_workflow(workflow_config)
    
    async def execute_workflow(
        self,
        workflow_id: str,
        engine: WorkflowEngine,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute workflow using specified engine."""
        orchestrator = self.orchestrators.get(engine)
        if not orchestrator:
            raise ValueError(f"Orchestrator not available: {engine}")
        
        return await orchestrator.execute_workflow(workflow_id, parameters)
    
    async def get_workflow_status(
        self,
        execution_id: str,
        engine: WorkflowEngine
    ) -> WorkflowStatus:
        """Get workflow execution status."""
        orchestrator = self.orchestrators.get(engine)
        if not orchestrator:
            raise ValueError(f"Orchestrator not available: {engine}")
        
        return await orchestrator.get_workflow_status(execution_id)
    
    async def list_all_workflows(self) -> Dict[WorkflowEngine, List[str]]:
        """List workflows from all orchestrators."""
        all_workflows = {}
        
        for engine, orchestrator in self.orchestrators.items():
            try:
                workflows = await orchestrator.list_workflows()
                all_workflows[engine] = workflows
            except Exception as e:
                logger.error(f"Failed to list workflows from {engine}: {e}")
                all_workflows[engine] = []
        
        return all_workflows