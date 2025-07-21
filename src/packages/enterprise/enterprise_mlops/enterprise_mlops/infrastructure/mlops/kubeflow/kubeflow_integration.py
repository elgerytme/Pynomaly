"""
Kubeflow Integration for Enterprise MLOps

Provides comprehensive integration with Kubeflow Pipelines,
Katib for hyperparameter tuning, and KServe for model serving.
"""

import asyncio
import tempfile
import os
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
import json

from structlog import get_logger
import kfp
from kfp import dsl
from kfp.client import Client
from kubeflow.katib import ApiClient as KatibApiClient
from kubeflow.katib.api.katib_api import KatibApi
from kubeflow.katib.models import V1beta1Experiment
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import requests

from ...domain.entities.mlops import MLPipeline, MLExperiment, ModelDeployment, PipelineStatus

logger = get_logger(__name__)


class KubeflowIntegration:
    """
    Kubeflow integration for enterprise MLOps.
    
    Provides comprehensive integration with Kubeflow Pipelines,
    Katib for hyperparameter tuning, and KServe for model serving.
    """
    
    def __init__(
        self,
        host: str,
        namespace: str = "kubeflow",
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        client_id: Optional[str] = None,
        existing_token: Optional[str] = None
    ):
        self.host = host
        self.namespace = namespace
        self.username = username
        self.password = password
        self.token = token
        self.client_id = client_id
        self.existing_token = existing_token
        
        self.kfp_client = None
        self.katib_client = None
        self.k8s_client = None
        self.logger = logger.bind(integration="kubeflow")
        
        self.logger.info("KubeflowIntegration initialized", host=host, namespace=namespace)
    
    async def connect(self) -> bool:
        """Establish connection to Kubeflow."""
        self.logger.info("Connecting to Kubeflow")
        
        try:
            # Configure authentication
            auth_session = None
            if self.existing_token:
                # Use existing bearer token
                auth_session = self._create_auth_session_with_token(self.existing_token)
            elif self.username and self.password:
                # Authenticate with username/password
                auth_session = await self._authenticate_with_credentials()
            
            # Create Kubeflow Pipelines client
            self.kfp_client = Client(
                host=self.host,
                namespace=self.namespace,
                existing_token=self.existing_token,
                client_id=self.client_id
            )
            
            # Create Katib client for hyperparameter tuning
            try:
                config.load_incluster_config()
            except:
                try:
                    config.load_kube_config()
                except:
                    self.logger.warning("Could not load Kubernetes config")
            
            self.katib_client = KatibApi()
            self.k8s_client = client.ApiClient()
            
            # Test connection
            await self._test_connection()
            
            self.logger.info("Successfully connected to Kubeflow")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to Kubeflow: {str(e)}"
            self.logger.error(error_msg)
            return False
    
    async def create_pipeline(
        self,
        pipeline: MLPipeline,
        pipeline_func: callable,
        pipeline_package_path: Optional[str] = None
    ) -> str:
        """Create Kubeflow pipeline from ML pipeline definition."""
        if not self.kfp_client:
            raise RuntimeError("Not connected to Kubeflow")
        
        self.logger.info("Creating Kubeflow pipeline", name=pipeline.name)
        
        try:
            # Compile pipeline if function provided
            if pipeline_func:
                # Create temporary pipeline package
                if not pipeline_package_path:
                    temp_dir = tempfile.mkdtemp()
                    pipeline_package_path = os.path.join(temp_dir, f"{pipeline.name}.yaml")
                
                # Compile pipeline
                kfp.compiler.Compiler().compile(pipeline_func, pipeline_package_path)
            
            if not pipeline_package_path or not os.path.exists(pipeline_package_path):
                raise ValueError("Pipeline package path is required")
            
            # Upload pipeline
            kfp_pipeline = self.kfp_client.upload_pipeline(
                pipeline_package_path=pipeline_package_path,
                pipeline_name=pipeline.name,
                description=pipeline.description
            )
            
            # Update pipeline with external ID
            pipeline.external_pipeline_id = kfp_pipeline.id
            
            self.logger.info("Pipeline created successfully", 
                           pipeline_id=kfp_pipeline.id, name=pipeline.name)
            
            return kfp_pipeline.id
            
        except Exception as e:
            self.logger.error("Failed to create pipeline", error=str(e))
            raise
    
    async def run_pipeline(
        self,
        pipeline: MLPipeline,
        run_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None
    ) -> str:
        """Execute Kubeflow pipeline."""
        if not self.kfp_client:
            raise RuntimeError("Not connected to Kubeflow")
        
        if not pipeline.external_pipeline_id:
            raise RuntimeError("Pipeline not created in Kubeflow")
        
        self.logger.info("Running Kubeflow pipeline", pipeline=pipeline.name)
        
        try:
            # Create or get experiment
            if experiment_name:
                try:
                    experiment = self.kfp_client.get_experiment(experiment_name=experiment_name)
                except:
                    experiment = self.kfp_client.create_experiment(
                        name=experiment_name,
                        description=f"Pynomaly experiment for {pipeline.name}"
                    )
            else:
                experiment = None
            
            # Submit pipeline run
            run_name = run_name or f"{pipeline.name}-{uuid4().hex[:8]}"
            merged_parameters = dict(pipeline.parameters)
            if parameters:
                merged_parameters.update(parameters)
            
            run_result = self.kfp_client.run_pipeline(
                experiment_id=experiment.id if experiment else None,
                job_name=run_name,
                pipeline_id=pipeline.external_pipeline_id,
                params=merged_parameters
            )
            
            # Start execution tracking
            execution_id = pipeline.start_execution()
            
            self.logger.info("Pipeline run started", 
                           run_id=run_result.id, execution_id=execution_id)
            
            return run_result.id
            
        except Exception as e:
            self.logger.error("Failed to run pipeline", error=str(e))
            raise
    
    async def create_hyperparameter_experiment(
        self,
        experiment: MLExperiment,
        algorithm: str = "random",
        objective_metric: str = "accuracy",
        objective_type: str = "maximize",
        max_trial_count: int = 10,
        parallel_trial_count: int = 2,
        trial_template: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create Katib hyperparameter tuning experiment."""
        if not self.katib_client:
            raise RuntimeError("Not connected to Kubeflow")
        
        self.logger.info("Creating hyperparameter experiment", name=experiment.name)
        
        try:
            # Build parameter space from experiment hyperparameters
            parameters = []
            for param_name, param_config in experiment.hyperparameters.items():
                if isinstance(param_config, dict):
                    param_spec = {
                        "name": param_name,
                        "parameterType": param_config.get("type", "double"),
                        "feasibleSpace": {
                            "min": str(param_config.get("min", 0)),
                            "max": str(param_config.get("max", 1))
                        }
                    }
                    parameters.append(param_spec)
            
            # Default trial template if not provided
            if not trial_template:
                trial_template = {
                    "primaryContainerName": "training-container",
                    "trialSpec": {
                        "apiVersion": "batch/v1",
                        "kind": "Job",
                        "spec": {
                            "template": {
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "training-container",
                                            "image": "python:3.11-slim",
                                            "command": [
                                                "python",
                                                "/opt/pynomaly-training.py",
                                                "--tenant-id", str(experiment.tenant_id),
                                                "--experiment-id", str(experiment.id)
                                            ]
                                        }
                                    ],
                                    "restartPolicy": "Never"
                                }
                            }
                        }
                    }
                }
            
            # Create Katib experiment spec
            katib_experiment = V1beta1Experiment(
                api_version="kubeflow.org/v1beta1",
                kind="Experiment",
                metadata={
                    "name": f"pynomaly-{experiment.name.lower().replace('_', '-')}-{uuid4().hex[:8]}",
                    "namespace": self.namespace
                },
                spec={
                    "algorithm": {
                        "algorithmName": algorithm
                    },
                    "objective": {
                        "type": objective_type,
                        "objectiveMetricName": objective_metric
                    },
                    "parameters": parameters,
                    "trialTemplate": trial_template,
                    "parallelTrialCount": parallel_trial_count,
                    "maxTrialCount": max_trial_count,
                    "maxFailedTrialCount": max_trial_count // 2
                }
            )
            
            # Submit experiment
            response = self.katib_client.create_experiment(
                body=katib_experiment,
                namespace=self.namespace
            )
            
            katib_experiment_name = response.metadata.name
            
            self.logger.info("Hyperparameter experiment created", 
                           name=katib_experiment_name, trials=max_trial_count)
            
            return katib_experiment_name
            
        except Exception as e:
            self.logger.error("Failed to create hyperparameter experiment", error=str(e))
            raise
    
    async def deploy_model_kserve(
        self,
        deployment: ModelDeployment,
        model_uri: str,
        serving_runtime: str = "sklearn",
        resources: Optional[Dict[str, str]] = None
    ) -> str:
        """Deploy model using KServe."""
        if not self.k8s_client:
            raise RuntimeError("Not connected to Kubernetes")
        
        self.logger.info("Deploying model with KServe", deployment=deployment.name)
        
        try:
            # Build KServe InferenceService spec
            inference_service = {
                "apiVersion": "serving.kserve.io/v1beta1",
                "kind": "InferenceService",
                "metadata": {
                    "name": deployment.name.lower().replace('_', '-'),
                    "namespace": self.namespace,
                    "labels": {
                        "pynomaly.tenant_id": str(deployment.tenant_id),
                        "pynomaly.model_id": str(deployment.model_id),
                        "pynomaly.deployment_id": str(deployment.id)
                    }
                },
                "spec": {
                    "predictor": {
                        serving_runtime: {
                            "storageUri": model_uri,
                            "resources": resources or {
                                "limits": {
                                    "cpu": "1",
                                    "memory": "2Gi"
                                },
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "512Mi"
                                }
                            }
                        }
                    }
                }
            }
            
            # Add scaling configuration
            if deployment.scaling_config:
                min_replicas = deployment.scaling_config.get("min_replicas", 1)
                max_replicas = deployment.scaling_config.get("max_replicas", 3)
                
                inference_service["spec"]["predictor"][serving_runtime]["minReplicas"] = min_replicas
                inference_service["spec"]["predictor"][serving_runtime]["maxReplicas"] = max_replicas
            
            # Create InferenceService using custom resource API
            custom_objects_api = client.CustomObjectsApi(self.k8s_client)
            
            response = custom_objects_api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="inferenceservices",
                body=inference_service
            )
            
            service_name = response["metadata"]["name"]
            
            # Update deployment with endpoint information
            deployment.endpoint_url = f"http://{service_name}.{self.namespace}.svc.cluster.local"
            deployment.update_status("deployed")
            
            self.logger.info("Model deployed with KServe", 
                           service_name=service_name, 
                           endpoint=deployment.endpoint_url)
            
            return service_name
            
        except Exception as e:
            self.logger.error("Failed to deploy model with KServe", error=str(e))
            raise
    
    async def get_pipeline_runs(
        self,
        pipeline_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get pipeline runs for a given pipeline."""
        if not self.kfp_client:
            raise RuntimeError("Not connected to Kubeflow")
        
        try:
            runs = self.kfp_client.list_runs(
                pipeline_id=pipeline_id,
                page_size=limit
            )
            
            run_data = []
            for run in runs.runs or []:
                run_info = {
                    "run_id": run.id,
                    "name": run.name,
                    "status": run.status,
                    "created_at": run.created_at,
                    "scheduled_at": run.scheduled_at,
                    "finished_at": run.finished_at,
                    "pipeline_spec": {
                        "pipeline_id": run.pipeline_spec.pipeline_id if run.pipeline_spec else None,
                        "pipeline_name": run.pipeline_spec.pipeline_name if run.pipeline_spec else None,
                    },
                    "metrics": []
                }
                
                # Get run metrics if available
                try:
                    run_detail = self.kfp_client.get_run(run.id)
                    if run_detail.run.pipeline_runtime and run_detail.run.pipeline_runtime.workflow_manifest:
                        # Parse workflow for metrics
                        pass
                except:
                    pass
                
                run_data.append(run_info)
            
            return run_data
            
        except Exception as e:
            self.logger.error("Failed to get pipeline runs", error=str(e))
            raise
    
    async def get_experiment_results(
        self,
        experiment_name: str
    ) -> Dict[str, Any]:
        """Get Katib experiment results."""
        if not self.katib_client:
            raise RuntimeError("Not connected to Kubeflow")
        
        try:
            experiment = self.katib_client.get_experiment(
                name=experiment_name,
                namespace=self.namespace
            )
            
            # Get experiment trials
            trials = []
            if experiment.status and experiment.status.trials:
                for trial in experiment.status.trials:
                    trial_info = {
                        "trial_name": trial.trial_name,
                        "status": trial.condition,
                        "parameter_assignments": [],
                        "objective_value": None
                    }
                    
                    # Parse parameter assignments
                    if trial.parameter_assignments:
                        for param in trial.parameter_assignments:
                            trial_info["parameter_assignments"].append({
                                "name": param.name,
                                "value": param.value
                            })
                    
                    # Parse objective value
                    if trial.observation and trial.observation.metrics:
                        for metric in trial.observation.metrics:
                            if metric.name == experiment.spec.objective.objective_metric_name:
                                trial_info["objective_value"] = float(metric.value)
                    
                    trials.append(trial_info)
            
            # Find best trial
            best_trial = None
            if trials:
                objective_type = experiment.spec.objective.type.lower()
                if objective_type == "maximize":
                    best_trial = max(trials, key=lambda t: t["objective_value"] or float('-inf'))
                else:
                    best_trial = min(trials, key=lambda t: t["objective_value"] or float('inf'))
            
            return {
                "experiment_name": experiment.metadata.name,
                "status": experiment.status.conditions[-1].type if experiment.status and experiment.status.conditions else "Unknown",
                "algorithm": experiment.spec.algorithm.algorithm_name,
                "objective": {
                    "metric": experiment.spec.objective.objective_metric_name,
                    "type": experiment.spec.objective.type
                },
                "trials_completed": experiment.status.trials_succeeded if experiment.status else 0,
                "trials_total": len(trials),
                "best_trial": best_trial,
                "trials": trials
            }
            
        except Exception as e:
            self.logger.error("Failed to get experiment results", error=str(e))
            raise
    
    async def get_model_serving_status(
        self,
        service_name: str
    ) -> Dict[str, Any]:
        """Get KServe model serving status."""
        if not self.k8s_client:
            raise RuntimeError("Not connected to Kubernetes")
        
        try:
            custom_objects_api = client.CustomObjectsApi(self.k8s_client)
            
            # Get InferenceService
            service = custom_objects_api.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="inferenceservices",
                name=service_name
            )
            
            status = service.get("status", {})
            
            return {
                "service_name": service["metadata"]["name"],
                "ready": status.get("ready", False),
                "url": status.get("url"),
                "conditions": status.get("conditions", []),
                "replicas": status.get("components", {}).get("predictor", {}).get("replicas"),
                "traffic": status.get("components", {}).get("predictor", {}).get("traffic")
            }
            
        except Exception as e:
            self.logger.error("Failed to get model serving status", error=str(e))
            raise
    
    async def delete_pipeline(
        self,
        pipeline_id: str
    ) -> bool:
        """Delete Kubeflow pipeline."""
        if not self.kfp_client:
            raise RuntimeError("Not connected to Kubeflow")
        
        try:
            self.kfp_client.delete_pipeline(pipeline_id)
            self.logger.info("Pipeline deleted", pipeline_id=pipeline_id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to delete pipeline", error=str(e))
            return False
    
    async def delete_experiment(
        self,
        experiment_name: str
    ) -> bool:
        """Delete Katib experiment."""
        if not self.katib_client:
            raise RuntimeError("Not connected to Kubeflow")
        
        try:
            self.katib_client.delete_experiment(
                name=experiment_name,
                namespace=self.namespace
            )
            self.logger.info("Experiment deleted", experiment_name=experiment_name)
            return True
            
        except Exception as e:
            self.logger.error("Failed to delete experiment", error=str(e))
            return False
    
    async def delete_model_service(
        self,
        service_name: str
    ) -> bool:
        """Delete KServe model service."""
        if not self.k8s_client:
            raise RuntimeError("Not connected to Kubernetes")
        
        try:
            custom_objects_api = client.CustomObjectsApi(self.k8s_client)
            
            custom_objects_api.delete_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="inferenceservices",
                name=service_name
            )
            
            self.logger.info("Model service deleted", service_name=service_name)
            return True
            
        except Exception as e:
            self.logger.error("Failed to delete model service", error=str(e))
            return False
    
    # Private helper methods
    
    async def _test_connection(self) -> None:
        """Test Kubeflow connection."""
        try:
            # Test KFP connection
            pipelines = self.kfp_client.list_pipelines(page_size=1)
            self.logger.info("KFP connection test successful")
            
            # Test Katib connection
            try:
                experiments = self.katib_client.list_experiments(namespace=self.namespace)
                self.logger.info("Katib connection test successful")
            except:
                self.logger.warning("Katib connection test failed, but continuing")
            
        except Exception as e:
            raise RuntimeError(f"Connection test failed: {str(e)}")
    
    async def _authenticate_with_credentials(self) -> requests.Session:
        """Authenticate with username and password."""
        session = requests.Session()
        
        try:
            # This is a simplified authentication flow
            # In practice, you would need to handle the specific authentication
            # mechanism used by your Kubeflow deployment (Dex, OIDC, etc.)
            
            auth_url = f"{self.host}/dex/auth"
            login_data = {
                "login": self.username,
                "password": self.password
            }
            
            response = session.post(auth_url, data=login_data)
            response.raise_for_status()
            
            return session
            
        except Exception as e:
            raise RuntimeError(f"Authentication failed: {str(e)}")
    
    def _create_auth_session_with_token(self, token: str) -> requests.Session:
        """Create authenticated session with bearer token."""
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}"})
        return session
    
    def __del__(self):
        """Cleanup on object destruction."""
        # Kubeflow clients don't require explicit cleanup
        pass