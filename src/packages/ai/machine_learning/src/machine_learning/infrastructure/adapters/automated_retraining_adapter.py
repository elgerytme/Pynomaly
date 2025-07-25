"""Automated model retraining adapter implementation."""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import asdict, dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RetrainingTrigger(Enum):
    """Types of retraining triggers."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    NEW_DATA_AVAILABLE = "new_data_available"

class RetrainingStatus(Enum):
    """Status of retraining job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class RetrainingJob:
    """Represents an automated retraining job."""
    job_id: str
    model_name: str
    trigger_type: RetrainingTrigger
    status: RetrainingStatus
    trigger_conditions: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    original_model_version: Optional[str] = None
    new_model_version: Optional[str] = None
    performance_improvement: Optional[float] = None
    retraining_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.retraining_config is None:
            self.retraining_config = {}

@dataclass
class RetrainingConfiguration:
    """Configuration for automated retraining."""
    model_name: str
    enabled: bool = True
    performance_threshold: float = 0.8  # Retrain if accuracy drops below this
    drift_threshold: float = 0.3  # Retrain if drift score exceeds this
    min_data_samples: int = 1000  # Minimum new data samples to trigger retraining
    schedule_cron: Optional[str] = None  # Cron expression for scheduled retraining
    max_retraining_frequency_hours: int = 24  # Minimum hours between retrainings
    auto_deploy_threshold: float = 0.95  # Auto-deploy if accuracy exceeds this
    notification_webhooks: List[str] = None
    
    def __post_init__(self):
        if self.notification_webhooks is None:
            self.notification_webhooks = []

class AutomatedRetrainingPort:
    """Port for automated model retraining operations."""
    
    async def configure_retraining(self, config: RetrainingConfiguration) -> bool:
        """Configure automated retraining for a model."""
        pass
    
    async def trigger_retraining(
        self,
        model_name: str,
        trigger_type: RetrainingTrigger,
        trigger_conditions: Dict[str, Any]
    ) -> str:
        """Trigger retraining job."""
        pass
    
    async def get_retraining_status(self, job_id: str) -> Optional[RetrainingJob]:
        """Get status of retraining job."""
        pass
    
    async def cancel_retraining(self, job_id: str) -> bool:
        """Cancel running retraining job."""
        pass
    
    async def list_retraining_jobs(self, model_name: Optional[str] = None) -> List[RetrainingJob]:
        """List retraining jobs."""
        pass

class AutomatedRetrainingAdapter(AutomatedRetrainingPort):
    """File-based automated retraining implementation."""
    
    def __init__(self, storage_root: str = "/tmp/automated_retraining"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.storage_root / "jobs"
        self.configs_dir = self.storage_root / "configurations"
        self.jobs_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)
        
        # In-memory tracking
        self.active_jobs: Dict[str, RetrainingJob] = {}
        self.monitoring_enabled = True
        
        # Start background monitoring
        asyncio.create_task(self._background_monitor())
    
    async def configure_retraining(self, config: RetrainingConfiguration) -> bool:
        """Configure automated retraining for a model."""
        try:
            config_file = self.configs_dir / f"{config.model_name}_config.json"
            
            with open(config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2, default=str)
            
            logger.info(f"Configured automated retraining for {config.model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to configure retraining for {config.model_name}: {e}")
            return False
    
    async def trigger_retraining(
        self,
        model_name: str,
        trigger_type: RetrainingTrigger,
        trigger_conditions: Dict[str, Any]
    ) -> str:
        """Trigger retraining job."""
        try:
            job_id = f"retrain_{model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Check if retraining is too frequent
            if not await self._can_retrain(model_name):
                raise ValueError("Retraining frequency limit exceeded")
            
            job = RetrainingJob(
                job_id=job_id,
                model_name=model_name,
                trigger_type=trigger_type,
                status=RetrainingStatus.PENDING,
                trigger_conditions=trigger_conditions,
                created_at=datetime.utcnow().isoformat()
            )
            
            # Save job
            await self._save_job(job)
            self.active_jobs[job_id] = job
            
            # Start retraining process
            asyncio.create_task(self._execute_retraining(job))
            
            logger.info(f"Triggered retraining job {job_id} for {model_name}")
            return job_id
        
        except Exception as e:
            logger.error(f"Failed to trigger retraining for {model_name}: {e}")
            raise
    
    async def get_retraining_status(self, job_id: str) -> Optional[RetrainingJob]:
        """Get status of retraining job."""
        try:
            # Check in-memory first
            if job_id in self.active_jobs:
                return self.active_jobs[job_id]
            
            # Load from file
            job_file = self.jobs_dir / f"{job_id}.json"
            if not job_file.exists():
                return None
            
            with open(job_file, 'r') as f:
                data = json.load(f)
            
            return RetrainingJob(**data)
        
        except Exception as e:
            logger.error(f"Failed to get retraining status for {job_id}: {e}")
            return None
    
    async def cancel_retraining(self, job_id: str) -> bool:
        """Cancel running retraining job."""
        try:
            job = await self.get_retraining_status(job_id)
            if not job:
                return False
            
            if job.status in [RetrainingStatus.COMPLETED, RetrainingStatus.FAILED, RetrainingStatus.CANCELLED]:
                return False
            
            job.status = RetrainingStatus.CANCELLED
            job.completed_at = datetime.utcnow().isoformat()
            
            await self._save_job(job)
            
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            logger.info(f"Cancelled retraining job {job_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to cancel retraining job {job_id}: {e}")
            return False
    
    async def list_retraining_jobs(self, model_name: Optional[str] = None) -> List[RetrainingJob]:
        """List retraining jobs."""
        try:
            jobs = []
            
            for job_file in self.jobs_dir.glob("*.json"):
                with open(job_file, 'r') as f:
                    data = json.load(f)
                
                job = RetrainingJob(**data)
                
                if model_name is None or job.model_name == model_name:
                    jobs.append(job)
            
            # Sort by created_at, newest first
            jobs.sort(key=lambda x: x.created_at, reverse=True)
            return jobs
        
        except Exception as e:
            logger.error(f"Failed to list retraining jobs: {e}")
            return []
    
    async def _execute_retraining(self, job: RetrainingJob) -> None:
        """Execute the retraining process."""
        try:
            job.status = RetrainingStatus.RUNNING
            job.started_at = datetime.utcnow().isoformat()
            await self._save_job(job)
            
            # Load retraining configuration
            config = await self._load_retraining_config(job.model_name)
            if not config:
                raise ValueError(f"No retraining configuration found for {job.model_name}")
            
            # Get original model performance
            original_performance = await self._get_model_performance(job.model_name)
            job.original_model_version = original_performance.get("version", "unknown")
            
            # Simulate retraining process
            logger.info(f"Starting retraining for {job.model_name}")
            
            # Step 1: Data preparation
            await self._prepare_training_data(job)
            await asyncio.sleep(2)  # Simulate data preparation time
            
            # Step 2: Model training
            new_model_metrics = await self._train_new_model(job, config)
            await asyncio.sleep(5)  # Simulate training time
            
            # Step 3: Model validation
            validation_results = await self._validate_new_model(job, new_model_metrics)
            await asyncio.sleep(1)  # Simulate validation time
            
            # Step 4: Performance comparison
            performance_improvement = await self._compare_model_performance(
                original_performance, new_model_metrics
            )
            
            job.new_model_version = new_model_metrics["version"]
            job.performance_improvement = performance_improvement
            
            # Step 5: Auto-deployment decision
            if (config.auto_deploy_threshold and 
                new_model_metrics["accuracy"] >= config.auto_deploy_threshold and
                performance_improvement > 0.01):  # At least 1% improvement
                
                await self._deploy_new_model(job, new_model_metrics)
                logger.info(f"Auto-deployed new model version {job.new_model_version}")
            
            job.status = RetrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow().isoformat()
            
            # Send notifications
            await self._send_notifications(job, config)
            
            logger.info(f"Completed retraining job {job.job_id}")
        
        except Exception as e:
            job.status = RetrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow().isoformat()
            logger.error(f"Retraining job {job.job_id} failed: {e}")
        
        finally:
            await self._save_job(job)
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    async def _can_retrain(self, model_name: str) -> bool:
        """Check if model can be retrained based on frequency limits."""
        try:
            config = await self._load_retraining_config(model_name)
            if not config:
                return True  # No restrictions if no config
            
            # Check recent retraining jobs
            recent_jobs = await self.list_retraining_jobs(model_name)
            
            if not recent_jobs:
                return True
            
            # Check most recent completed job
            for job in recent_jobs:
                if job.status == RetrainingStatus.COMPLETED:
                    completed_time = datetime.fromisoformat(job.completed_at.replace('Z', '+00:00'))
                    time_since = datetime.utcnow() - completed_time
                    
                    if time_since.total_seconds() < config.max_retraining_frequency_hours * 3600:
                        return False
                    break
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to check retraining eligibility: {e}")
            return False
    
    async def _load_retraining_config(self, model_name: str) -> Optional[RetrainingConfiguration]:
        """Load retraining configuration for model."""
        try:
            config_file = self.configs_dir / f"{model_name}_config.json"
            if not config_file.exists():
                return None
            
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            return RetrainingConfiguration(**data)
        
        except Exception as e:
            logger.error(f"Failed to load retraining config for {model_name}: {e}")
            return None
    
    async def _save_job(self, job: RetrainingJob) -> None:
        """Save retraining job to file."""
        try:
            job_file = self.jobs_dir / f"{job.job_id}.json"
            
            with open(job_file, 'w') as f:
                json.dump(asdict(job), f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to save retraining job {job.job_id}: {e}")
    
    async def _get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get current model performance metrics."""
        # Simulate getting model performance
        return {
            "version": "v1.0",
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        }
    
    async def _prepare_training_data(self, job: RetrainingJob) -> None:
        """Prepare training data for retraining."""
        logger.info(f"Preparing training data for {job.model_name}")
        # Simulate data preparation
        job.retraining_config["data_preparation"] = {
            "status": "completed",
            "samples_collected": 5000,
            "features_engineered": 25
        }
    
    async def _train_new_model(self, job: RetrainingJob, config: RetrainingConfiguration) -> Dict[str, Any]:
        """Train new model version."""
        logger.info(f"Training new model for {job.model_name}")
        
        # Simulate model training with improved performance
        import random
        base_accuracy = 0.85
        improvement = random.uniform(0.02, 0.08)  # 2-8% improvement
        
        return {
            "version": f"v{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
            "accuracy": base_accuracy + improvement,
            "precision": 0.82 + improvement,
            "recall": 0.88 + improvement * 0.8,
            "f1_score": 0.85 + improvement * 0.9,
            "training_duration": 300,  # seconds
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50
            }
        }
    
    async def _validate_new_model(self, job: RetrainingJob, model_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate newly trained model."""
        logger.info(f"Validating new model for {job.model_name}")
        
        return {
            "validation_accuracy": model_metrics["accuracy"] - 0.01,  # Slightly lower on validation
            "cross_validation_score": model_metrics["accuracy"] - 0.005,
            "validation_passed": model_metrics["accuracy"] > 0.8
        }
    
    async def _compare_model_performance(
        self,
        original_performance: Dict[str, Any],
        new_performance: Dict[str, Any]
    ) -> float:
        """Compare performance between original and new model."""
        original_accuracy = original_performance.get("accuracy", 0.0)
        new_accuracy = new_performance.get("accuracy", 0.0)
        
        improvement = new_accuracy - original_accuracy
        return improvement
    
    async def _deploy_new_model(self, job: RetrainingJob, model_metrics: Dict[str, Any]) -> None:
        """Deploy new model version."""
        logger.info(f"Deploying new model version {model_metrics['version']} for {job.model_name}")
        
        # Simulate deployment process
        job.retraining_config["deployment"] = {
            "status": "deployed",
            "deployment_time": datetime.utcnow().isoformat(),
            "deployment_target": "production"
        }
    
    async def _send_notifications(self, job: RetrainingJob, config: RetrainingConfiguration) -> None:
        """Send notifications about retraining completion."""
        if not config.notification_webhooks:
            return
        
        notification_data = {
            "job_id": job.job_id,
            "model_name": job.model_name,
            "status": job.status.value,
            "performance_improvement": job.performance_improvement,
            "new_version": job.new_model_version
        }
        
        logger.info(f"Sending notifications for job {job.job_id}: {notification_data}")
        # In real implementation, send HTTP requests to webhook URLs
    
    async def _background_monitor(self) -> None:
        """Background monitoring for automatic retraining triggers."""
        while self.monitoring_enabled:
            try:
                await self._check_performance_degradation()
                await self._check_data_drift()
                await self._check_scheduled_retraining()
                
                # Sleep for 5 minutes between checks
                await asyncio.sleep(300)
            
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                await asyncio.sleep(60)  # Shorter sleep on error
    
    async def _check_performance_degradation(self) -> None:
        """Check for model performance degradation."""
        # In real implementation, check actual model metrics
        # For simulation, randomly trigger retraining
        pass
    
    async def _check_data_drift(self) -> None:
        """Check for data drift that requires retraining."""
        # In real implementation, check data distribution changes
        pass
    
    async def _check_scheduled_retraining(self) -> None:
        """Check for scheduled retraining jobs."""
        # In real implementation, parse cron expressions and trigger jobs
        pass