#!/usr/bin/env python3
"""
MLOps Service for Pynomaly.
This module provides a unified interface for all MLOps operations including
model registry, experiment tracking, deployment, and automated retraining.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .automated_retraining import RetrainingConfig, TriggerType, retraining_pipeline
from .experiment_tracker import experiment_tracker
from .model_deployment import (
    DeploymentEnvironment,
    DeploymentStatus,
    deployment_manager,
)

# MLOps components
from .model_registry import ModelStatus, ModelType, model_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class ModelRegistrationRequest(BaseModel):
    """Model registration request."""

    name: str
    version: str
    model_type: str
    author: str
    description: str = ""
    tags: list[str] = []
    performance_metrics: dict[str, float] = {}
    hyperparameters: dict[str, Any] = {}


class ModelRegistrationResponse(BaseModel):
    """Model registration response."""

    model_id: str
    status: str
    message: str
    timestamp: datetime


class ExperimentRequest(BaseModel):
    """Experiment creation request."""

    name: str
    description: str = ""
    tags: list[str] = []


class ExperimentResponse(BaseModel):
    """Experiment response."""

    experiment_id: str
    name: str
    description: str
    status: str
    timestamp: datetime


class DeploymentRequest(BaseModel):
    """Deployment request."""

    model_id: str
    model_version: str
    environment: str
    configuration: dict[str, Any] = {}
    resources: dict[str, Any] = {}
    auto_deploy: bool = False


class DeploymentResponse(BaseModel):
    """Deployment response."""

    deployment_id: str
    status: str
    endpoint_url: str
    health_check_url: str
    timestamp: datetime


class RetrainingConfigRequest(BaseModel):
    """Retraining configuration request."""

    model_id: str
    trigger_type: str
    schedule_cron: str | None = None
    performance_threshold: float = 0.05
    data_drift_threshold: float = 0.1
    min_data_points: int = 100
    max_training_time_minutes: int = 60
    auto_deploy: bool = False
    notification_enabled: bool = True


class MLOpsService:
    """Unified MLOps service."""

    def __init__(self):
        """Initialize MLOps service."""
        self.model_registry = model_registry
        self.experiment_tracker = experiment_tracker
        self.deployment_manager = deployment_manager
        self.retraining_pipeline = retraining_pipeline

        # Service statistics
        self.stats = {
            "service_started": datetime.now(),
            "total_models": 0,
            "total_experiments": 0,
            "total_deployments": 0,
            "total_retraining_jobs": 0,
        }

        logger.info("MLOps service initialized")

    async def register_model_from_experiment(
        self,
        experiment_id: str,
        run_id: str,
        model_name: str,
        model_version: str,
        model_type: ModelType,
        author: str,
    ) -> str:
        """Register a model from an experiment run."""
        try:
            # Get experiment run
            run = self.experiment_tracker.get_run(run_id)
            if not run:
                raise ValueError(f"Run not found: {run_id}")

            # Create model (placeholder - in practice, load from artifacts)
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(contamination=0.1, random_state=42)

            # Register model
            model_id = await self.model_registry.register_model(
                model=model,
                name=model_name,
                version=model_version,
                model_type=model_type,
                author=author,
                description=f"Model from experiment {experiment_id}, run {run_id}",
                tags=["experiment", experiment_id, run_id],
                performance_metrics=run.metrics,
                hyperparameters=run.parameters,
            )

            # Add model to experiment
            await self.experiment_tracker.add_model_to_experiment(
                experiment_id, model_id
            )

            self.stats["total_models"] += 1

            logger.info(f"Model registered from experiment: {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Failed to register model from experiment: {e}")
            raise

    async def deploy_model_from_registry(
        self,
        model_id: str,
        environment: DeploymentEnvironment,
        configuration: dict[str, Any] = None,
        auto_deploy: bool = False,
    ) -> str:
        """Deploy a model from the registry."""
        try:
            # Get model metadata
            _, metadata = await self.model_registry.get_model(model_id)

            # Create deployment
            deployment_id = await self.deployment_manager.create_deployment(
                model_id=model_id,
                model_version=metadata.version,
                environment=environment,
                configuration=configuration or {},
                author=metadata.author,
                notes=f"Deployment of model {model_id}",
            )

            # Deploy if auto_deploy is True
            if auto_deploy:
                await self.deployment_manager.deploy_model(deployment_id)

            self.stats["total_deployments"] += 1

            logger.info(f"Model deployment created: {deployment_id}")
            return deployment_id

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise

    async def setup_automated_retraining(
        self,
        model_id: str,
        trigger_type: TriggerType,
        schedule_cron: str = None,
        performance_threshold: float = 0.05,
        data_drift_threshold: float = 0.1,
        auto_deploy: bool = False,
    ) -> bool:
        """Set up automated retraining for a model."""
        try:
            config = RetrainingConfig(
                model_id=model_id,
                trigger_type=trigger_type,
                schedule_cron=schedule_cron,
                performance_threshold=performance_threshold,
                data_drift_threshold=data_drift_threshold,
                min_data_points=100,
                max_training_time_minutes=60,
                auto_deploy=auto_deploy,
                validation_split=0.2,
                hyperparameter_tuning=False,
                notification_enabled=True,
                rollback_enabled=True,
            )

            self.retraining_pipeline.configure_retraining(config)

            logger.info(f"Automated retraining configured for {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to setup automated retraining: {e}")
            raise

    async def get_model_lifecycle_info(self, model_id: str) -> dict[str, Any]:
        """Get comprehensive model lifecycle information."""
        try:
            # Get model metadata
            _, metadata = await self.model_registry.get_model(model_id)

            # Get model lineage
            lineage = await self.model_registry.get_model_lineage(model_id)

            # Get deployments
            deployments = await self.deployment_manager.list_deployments(
                model_id=model_id
            )

            # Get retraining jobs
            retraining_jobs = self.retraining_pipeline.list_jobs(model_id=model_id)

            # Get experiments (find experiments containing this model)
            experiments = []
            for exp_id in self.experiment_tracker.experiment_index["experiments"]:
                try:
                    exp_path = (
                        self.experiment_tracker.tracking_path
                        / f"experiment_{exp_id}.json"
                    )
                    if exp_path.exists():
                        with open(exp_path) as f:
                            exp_data = json.load(f)
                        if model_id in exp_data.get("models", []):
                            experiments.append(
                                {
                                    "experiment_id": exp_id,
                                    "name": exp_data["name"],
                                    "description": exp_data["description"],
                                }
                            )
                except Exception as e:
                    logger.warning(f"Failed to check experiment {exp_id}: {e}")

            lifecycle_info = {
                "model_info": {
                    "model_id": model_id,
                    "name": metadata.name,
                    "version": metadata.version,
                    "type": metadata.model_type.value,
                    "status": metadata.status.value,
                    "created_at": metadata.created_at.isoformat(),
                    "updated_at": metadata.updated_at.isoformat(),
                    "author": metadata.author,
                    "description": metadata.description,
                    "tags": metadata.tags,
                    "performance_metrics": metadata.performance_metrics,
                    "size_bytes": metadata.size_bytes,
                },
                "lineage": lineage,
                "deployments": [
                    {
                        "deployment_id": d.deployment_id,
                        "environment": d.environment.value,
                        "status": d.status.value,
                        "created_at": d.created_at.isoformat(),
                        "endpoint_url": d.endpoint_url,
                    }
                    for d in deployments
                ],
                "retraining_jobs": [
                    {
                        "job_id": j.job_id,
                        "status": j.status.value,
                        "trigger_type": j.trigger_type.value,
                        "created_at": j.created_at.isoformat(),
                        "duration_seconds": j.duration_seconds,
                        "metrics": j.metrics,
                    }
                    for j in retraining_jobs
                ],
                "experiments": experiments,
                "statistics": {
                    "total_deployments": len(deployments),
                    "active_deployments": len(
                        [d for d in deployments if d.status == DeploymentStatus.ACTIVE]
                    ),
                    "total_retraining_jobs": len(retraining_jobs),
                    "successful_retraining_jobs": len(
                        [j for j in retraining_jobs if j.status.value == "completed"]
                    ),
                },
                "timestamp": datetime.now().isoformat(),
            }

            return lifecycle_info

        except Exception as e:
            logger.error(f"Failed to get model lifecycle info: {e}")
            raise

    async def get_mlops_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive MLOps dashboard data."""
        try:
            # Get model registry stats
            registry_stats = await self.model_registry.get_registry_stats()

            # Get deployment stats
            all_deployments = await self.deployment_manager.list_deployments()

            # Get retraining pipeline stats
            pipeline_stats = self.retraining_pipeline.get_pipeline_stats()

            # Get experiment stats
            all_experiments = list(
                self.experiment_tracker.experiment_index["experiments"].keys()
            )
            all_runs = list(self.experiment_tracker.experiment_index["runs"].keys())

            # Calculate service uptime
            uptime = datetime.now() - self.stats["service_started"]

            dashboard_data = {
                "service_info": {
                    "status": "healthy",
                    "uptime_seconds": uptime.total_seconds(),
                    "started_at": self.stats["service_started"].isoformat(),
                    "version": "1.0.0",
                },
                "model_registry": {
                    "total_models": registry_stats["total_models"],
                    "models_by_status": registry_stats["models_by_status"],
                    "models_by_type": registry_stats["models_by_type"],
                    "storage_usage_mb": registry_stats["storage_usage"][
                        "total_size_mb"
                    ],
                },
                "experiments": {
                    "total_experiments": len(all_experiments),
                    "total_runs": len(all_runs),
                    "active_experiments": len(
                        [
                            e
                            for e in all_experiments
                            if self.experiment_tracker.experiment_index["experiments"][
                                e
                            ]["status"]
                            == "active"
                        ]
                    ),
                },
                "deployments": {
                    "total_deployments": len(all_deployments),
                    "active_deployments": len(
                        [
                            d
                            for d in all_deployments
                            if d.status == DeploymentStatus.ACTIVE
                        ]
                    ),
                    "deployments_by_environment": {},
                    "deployments_by_status": {},
                },
                "retraining_pipeline": pipeline_stats,
                "recent_activities": await self._get_recent_activities(),
                "system_health": {
                    "model_registry": "healthy",
                    "experiment_tracker": "healthy",
                    "deployment_manager": "healthy",
                    "retraining_pipeline": "healthy",
                },
                "timestamp": datetime.now().isoformat(),
            }

            # Count deployments by environment and status
            for deployment in all_deployments:
                env_key = deployment.environment.value
                dashboard_data["deployments"]["deployments_by_environment"][env_key] = (
                    dashboard_data["deployments"]["deployments_by_environment"].get(
                        env_key, 0
                    )
                    + 1
                )

                status_key = deployment.status.value
                dashboard_data["deployments"]["deployments_by_status"][status_key] = (
                    dashboard_data["deployments"]["deployments_by_status"].get(
                        status_key, 0
                    )
                    + 1
                )

            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to get MLOps dashboard data: {e}")
            raise

    async def _get_recent_activities(self) -> list[dict[str, Any]]:
        """Get recent MLOps activities."""
        activities = []

        try:
            # Get recent models
            models = await self.model_registry.list_models()
            for model in models[:5]:  # Last 5 models
                activities.append(
                    {
                        "type": "model_registered",
                        "timestamp": model.created_at.isoformat(),
                        "description": f"Model {model.name} v{model.version} registered",
                        "details": {
                            "model_id": model.model_id,
                            "author": model.author,
                            "status": model.status.value,
                        },
                    }
                )

            # Get recent deployments
            deployments = await self.deployment_manager.list_deployments()
            for deployment in deployments[:5]:  # Last 5 deployments
                activities.append(
                    {
                        "type": "model_deployed",
                        "timestamp": deployment.created_at.isoformat(),
                        "description": f"Model deployed to {deployment.environment.value}",
                        "details": {
                            "deployment_id": deployment.deployment_id,
                            "model_id": deployment.model_id,
                            "environment": deployment.environment.value,
                            "status": deployment.status.value,
                        },
                    }
                )

            # Get recent retraining jobs
            retraining_jobs = self.retraining_pipeline.list_jobs()
            for job in retraining_jobs[:5]:  # Last 5 jobs
                activities.append(
                    {
                        "type": "retraining_job",
                        "timestamp": job.created_at.isoformat(),
                        "description": f"Retraining job {job.status.value} for {job.config.model_id}",
                        "details": {
                            "job_id": job.job_id,
                            "model_id": job.config.model_id,
                            "trigger_type": job.trigger_type.value,
                            "status": job.status.value,
                        },
                    }
                )

            # Sort by timestamp (newest first)
            activities.sort(key=lambda x: x["timestamp"], reverse=True)

            return activities[:10]  # Return last 10 activities

        except Exception as e:
            logger.error(f"Failed to get recent activities: {e}")
            return []

    async def run_health_check(self) -> dict[str, Any]:
        """Run comprehensive health check."""
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        # Check model registry
        try:
            stats = await self.model_registry.get_registry_stats()
            health_status["components"]["model_registry"] = {
                "status": "healthy",
                "total_models": stats["total_models"],
                "last_check": datetime.now().isoformat(),
            }
        except Exception as e:
            health_status["components"]["model_registry"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }
            health_status["overall_status"] = "degraded"

        # Check experiment tracker
        try:
            total_experiments = len(
                self.experiment_tracker.experiment_index["experiments"]
            )
            health_status["components"]["experiment_tracker"] = {
                "status": "healthy",
                "total_experiments": total_experiments,
                "last_check": datetime.now().isoformat(),
            }
        except Exception as e:
            health_status["components"]["experiment_tracker"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }
            health_status["overall_status"] = "degraded"

        # Check deployment manager
        try:
            deployments = await self.deployment_manager.list_deployments()
            active_deployments = len(
                [d for d in deployments if d.status == DeploymentStatus.ACTIVE]
            )
            health_status["components"]["deployment_manager"] = {
                "status": "healthy",
                "total_deployments": len(deployments),
                "active_deployments": active_deployments,
                "last_check": datetime.now().isoformat(),
            }
        except Exception as e:
            health_status["components"]["deployment_manager"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }
            health_status["overall_status"] = "degraded"

        # Check retraining pipeline
        try:
            pipeline_stats = self.retraining_pipeline.get_pipeline_stats()
            health_status["components"]["retraining_pipeline"] = {
                "status": "healthy",
                "total_jobs": pipeline_stats["total_jobs"],
                "active_jobs": pipeline_stats["active_jobs"],
                "last_check": datetime.now().isoformat(),
            }
        except Exception as e:
            health_status["components"]["retraining_pipeline"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }
            health_status["overall_status"] = "degraded"

        return health_status


# Initialize MLOps service
mlops_service = MLOpsService()

# Create FastAPI router
router = APIRouter(prefix="/mlops", tags=["mlops"])


# Model Registry endpoints
@router.post("/models/register", response_model=ModelRegistrationResponse)
async def register_model(request: ModelRegistrationRequest):
    """Register a new model."""
    try:
        # Create placeholder model (in practice, this would be uploaded)
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(contamination=0.1, random_state=42)

        model_id = await mlops_service.model_registry.register_model(
            model=model,
            name=request.name,
            version=request.version,
            model_type=ModelType(request.model_type.upper()),
            author=request.author,
            description=request.description,
            tags=request.tags,
            performance_metrics=request.performance_metrics,
            hyperparameters=request.hyperparameters,
        )

        return ModelRegistrationResponse(
            model_id=model_id,
            status="success",
            message="Model registered successfully",
            timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model information."""
    try:
        _, metadata = await mlops_service.model_registry.get_model(model_id)
        return {
            "model_id": model_id,
            "metadata": asdict(metadata),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/models")
async def list_models(
    status: str | None = None, model_type: str | None = None, author: str | None = None
):
    """List models with optional filtering."""
    try:
        models = await mlops_service.model_registry.list_models(
            status=ModelStatus(status) if status else None,
            model_type=ModelType(model_type) if model_type else None,
            author=author,
        )

        return {
            "models": [asdict(m) for m in models],
            "total": len(models),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Experiment Tracking endpoints
@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(request: ExperimentRequest):
    """Create a new experiment."""
    try:
        experiment_id = mlops_service.experiment_tracker.create_experiment(
            name=request.name, description=request.description, tags=request.tags
        )

        return ExperimentResponse(
            experiment_id=experiment_id,
            name=request.name,
            description=request.description,
            status="active",
            timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/summary")
async def get_experiment_summary(experiment_id: str):
    """Get experiment summary."""
    try:
        summary = mlops_service.experiment_tracker.get_experiment_summary(experiment_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# Deployment endpoints
@router.post("/deployments", response_model=DeploymentResponse)
async def create_deployment(request: DeploymentRequest):
    """Create a new deployment."""
    try:
        deployment_id = await mlops_service.deployment_manager.create_deployment(
            model_id=request.model_id,
            model_version=request.model_version,
            environment=DeploymentEnvironment(request.environment.upper()),
            configuration=request.configuration,
            resources=request.resources,
            author="api_user",
        )

        # Auto-deploy if requested
        if request.auto_deploy:
            await mlops_service.deployment_manager.deploy_model(deployment_id)

        deployment = await mlops_service.deployment_manager.get_deployment(
            deployment_id
        )

        return DeploymentResponse(
            deployment_id=deployment_id,
            status=deployment.status.value,
            endpoint_url=deployment.endpoint_url,
            health_check_url=deployment.health_check_url,
            timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments/{deployment_id}")
async def get_deployment(deployment_id: str):
    """Get deployment information."""
    try:
        deployment = await mlops_service.deployment_manager.get_deployment(
            deployment_id
        )
        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")

        return {
            "deployment": asdict(deployment),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deployments/{deployment_id}/deploy")
async def deploy_model(deployment_id: str):
    """Deploy a model."""
    try:
        success = await mlops_service.deployment_manager.deploy_model(deployment_id)
        return {
            "deployment_id": deployment_id,
            "success": success,
            "message": "Model deployed successfully"
            if success
            else "Deployment failed",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Retraining endpoints
@router.post("/retraining/configure")
async def configure_retraining(request: RetrainingConfigRequest):
    """Configure automated retraining."""
    try:
        success = await mlops_service.setup_automated_retraining(
            model_id=request.model_id,
            trigger_type=TriggerType(request.trigger_type.upper()),
            schedule_cron=request.schedule_cron,
            performance_threshold=request.performance_threshold,
            data_drift_threshold=request.data_drift_threshold,
            auto_deploy=request.auto_deploy,
        )

        return {
            "model_id": request.model_id,
            "success": success,
            "message": "Retraining configured successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retraining/jobs")
async def list_retraining_jobs(model_id: str | None = None, status: str | None = None):
    """List retraining jobs."""
    try:
        jobs = mlops_service.retraining_pipeline.list_jobs(
            model_id=model_id,
            status=None,  # Convert status string to enum if needed
        )

        return {
            "jobs": [asdict(job) for job in jobs],
            "total": len(jobs),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Dashboard and analytics endpoints
@router.get("/dashboard")
async def get_mlops_dashboard():
    """Get MLOps dashboard data."""
    try:
        dashboard_data = await mlops_service.get_mlops_dashboard_data()
        return dashboard_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/lifecycle")
async def get_model_lifecycle(model_id: str):
    """Get model lifecycle information."""
    try:
        lifecycle_info = await mlops_service.get_model_lifecycle_info(model_id)
        return lifecycle_info
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/health")
async def health_check():
    """Get MLOps service health status."""
    try:
        health_status = await mlops_service.run_health_check()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Make service and router available for import
__all__ = ["MLOpsService", "mlops_service", "router"]
