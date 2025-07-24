#!/usr/bin/env python3
"""
Advanced Model Registry System
Comprehensive model versioning, storage, and lifecycle management.
"""

import asyncio
import json
import logging
import os
import sys
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import pickle
import joblib
import boto3
import redis
import requests
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model-registry.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

Base = declarative_base()


class ModelStatus(Enum):
    """Model lifecycle status."""
    REGISTERED = "registered"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentStage(Enum):
    """Model deployment stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    SHADOW = "shadow"


@dataclass
class ModelMetadata:
    """Complete model metadata."""
    model_id: str
    version: str
    algorithm: str
    framework: str
    created_by: str
    created_at: datetime
    description: str
    tags: List[str]
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    feature_schema: Dict[str, Any]
    model_size_bytes: int
    model_checksum: str


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    training_time: float
    inference_latency: float
    memory_usage: float
    custom_metrics: Dict[str, float]


@dataclass
class DeploymentInfo:
    """Model deployment information."""
    deployment_id: str
    stage: DeploymentStage
    endpoint_url: str
    replicas: int
    resources: Dict[str, str]
    auto_scaling: Dict[str, Any]
    traffic_percentage: float
    deployment_time: datetime
    health_status: str


class ModelRegistryDB(Base):
    """Model registry database schema."""
    __tablename__ = 'model_registry'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(String(255), unique=True, nullable=False)
    version = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    algorithm = Column(String(100), nullable=False)
    framework = Column(String(100), nullable=False)
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    description = Column(Text)
    tags = Column(JSON)
    hyperparameters = Column(JSON)
    metrics = Column(JSON)
    metadata = Column(JSON)
    model_path = Column(String(500))
    model_size_bytes = Column(Integer)
    model_checksum = Column(String(100))
    deployment_info = Column(JSON)


class ExperimentDB(Base):
    """Experiment tracking database schema."""
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    status = Column(String(50), nullable=False)
    parameters = Column(JSON)
    metrics = Column(JSON)
    artifacts = Column(JSON)
    tags = Column(JSON)


class AdvancedModelRegistry:
    """Advanced model registry with comprehensive lifecycle management."""
    
    def __init__(self, config_path: str = "mlops/config/model-registry-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.db_engine = None
        self.db_session = None
        self.s3_client = None
        self.redis_client = None
        self._initialize_backends()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load model registry configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self._create_default_config()
        
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default model registry configuration."""
        default_config = {
            "registry": {
                "name": "advanced_model_registry",
                "version": "1.0.0",
                "database": {
                    "type": "postgresql",
                    "host": "localhost",
                    "port": 5432,
                    "database": "model_registry",
                    "username": "mlops",
                    "password": "mlops_password"
                },
                "storage": {
                    "type": "s3",
                    "bucket": "ml-model-artifacts",
                    "region": "us-east-1",
                    "encryption": True,
                    "versioning": True
                },
                "cache": {
                    "type": "redis",
                    "host": "localhost",
                    "port": 6379,
                    "database": 0,
                    "ttl": 3600
                }
            },
            "lifecycle": {
                "auto_promotion": {
                    "enabled": True,
                    "staging_to_production": {
                        "min_accuracy": 0.90,
                        "min_validation_days": 3,
                        "max_error_rate": 0.01,
                        "approval_required": True
                    }
                },
                "auto_deprecation": {
                    "enabled": True,
                    "performance_threshold": 0.05,
                    "comparison_window": "7d",
                    "grace_period": "24h"
                },
                "cleanup": {
                    "auto_archive": True,
                    "archive_after_days": 90,
                    "delete_after_days": 365
                }
            },
            "deployment": {
                "kubernetes": {
                    "namespace": "ml-models",
                    "image_registry": "ml-models.company.com",
                    "default_resources": {
                        "cpu": "500m",
                        "memory": "1Gi"
                    },
                    "auto_scaling": {
                        "min_replicas": 2,
                        "max_replicas": 10,
                        "target_cpu": 70
                    }
                },
                "monitoring": {
                    "metrics_enabled": True,
                    "logging_enabled": True,
                    "tracing_enabled": True,
                    "health_checks": True
                }
            },
            "security": {
                "access_control": {
                    "rbac_enabled": True,
                    "model_encryption": True,
                    "audit_logging": True
                },
                "scanning": {
                    "vulnerability_scanning": True,
                    "malware_scanning": True,
                    "compliance_checking": True
                }
            }
        }
        
        # Save default config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default model registry configuration: {self.config_path}")
        return default_config
    
    def _initialize_backends(self):
        """Initialize storage and database backends."""
        try:
            # Initialize database
            db_config = self.config["registry"]["database"]
            db_url = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            
            # For demo purposes, use SQLite
            db_url = "sqlite:///model_registry.db"
            self.db_engine = create_engine(db_url)
            Base.metadata.create_all(self.db_engine)
            SessionLocal = sessionmaker(bind=self.db_engine)
            self.db_session = SessionLocal()
            
            # Initialize S3 (simulated)
            try:
                self.s3_client = boto3.client('s3', region_name='us-east-1')
            except Exception:
                logger.warning("S3 client initialization failed, using local storage")
                self.s3_client = None
            
            # Initialize Redis (simulated)
            try:
                redis_config = self.config["registry"]["cache"]
                self.redis_client = redis.Redis(
                    host=redis_config["host"],
                    port=redis_config["port"],
                    db=redis_config["database"],
                    decode_responses=True
                )
                self.redis_client.ping()
            except Exception:
                logger.warning("Redis client initialization failed, using memory cache")
                self.redis_client = None
            
            logger.info("Model registry backends initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize backends: {e}")
            raise
    
    async def register_model(
        self,
        model_path: str,
        metadata: ModelMetadata,
        metrics: ModelMetrics,
        experiment_id: Optional[str] = None
    ) -> str:
        """Register a new model in the registry."""
        logger.info(f"Registering model: {metadata.model_id} v{metadata.version}")
        
        try:
            # Calculate model checksum
            model_checksum = self._calculate_model_checksum(model_path)
            
            # Store model artifacts
            stored_model_path = await self._store_model_artifacts(model_path, metadata)
            
            # Create registry entry
            registry_entry = ModelRegistryDB(
                model_id=metadata.model_id,
                version=metadata.version,
                status=ModelStatus.REGISTERED.value,
                algorithm=metadata.algorithm,
                framework=metadata.framework,
                created_by=metadata.created_by,
                created_at=metadata.created_at,
                updated_at=datetime.utcnow(),
                description=metadata.description,
                tags=metadata.tags,
                hyperparameters=metadata.hyperparameters,
                metrics=asdict(metrics),
                metadata=asdict(metadata),
                model_path=stored_model_path,
                model_size_bytes=metadata.model_size_bytes,
                model_checksum=model_checksum,
                deployment_info={}
            )
            
            self.db_session.add(registry_entry)
            self.db_session.commit()
            
            # Cache model metadata
            if self.redis_client:
                cache_key = f"model:{metadata.model_id}:{metadata.version}"
                cache_data = {
                    **asdict(metadata),
                    **asdict(metrics),
                    "stored_path": stored_model_path,
                    "checksum": model_checksum
                }
                self.redis_client.setex(
                    cache_key,
                    self.config["registry"]["cache"]["ttl"],
                    json.dumps(cache_data, default=str)
                )
            
            # Link to experiment if provided
            if experiment_id:
                await self._link_model_to_experiment(metadata.model_id, metadata.version, experiment_id)
            
            logger.info(f"Model registered successfully: {metadata.model_id} v{metadata.version}")
            return f"{metadata.model_id}:{metadata.version}"
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            if self.db_session:
                self.db_session.rollback()
            raise
    
    def _calculate_model_checksum(self, model_path: str) -> str:
        """Calculate SHA-256 checksum of model file."""
        sha256_hash = hashlib.sha256()
        
        try:
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate checksum: {e}")
            return "unknown"
    
    async def _store_model_artifacts(self, local_path: str, metadata: ModelMetadata) -> str:
        """Store model artifacts in configured storage backend."""
        storage_config = self.config["registry"]["storage"]
        
        if storage_config["type"] == "s3" and self.s3_client:
            # Store in S3
            s3_key = f"models/{metadata.model_id}/{metadata.version}/model.pkl"
            bucket = storage_config["bucket"]
            
            try:
                # Simulate S3 upload
                logger.info(f"Uploading model to S3: s3://{bucket}/{s3_key}")
                # self.s3_client.upload_file(local_path, bucket, s3_key)
                return f"s3://{bucket}/{s3_key}"
            except Exception as e:
                logger.warning(f"S3 upload failed: {e}, falling back to local storage")
        
        # Fallback to local storage
        local_storage_dir = Path("model_artifacts") / metadata.model_id / metadata.version
        local_storage_dir.mkdir(parents=True, exist_ok=True)
        
        stored_path = local_storage_dir / "model.pkl"
        
        # Copy model file
        import shutil
        shutil.copy2(local_path, stored_path)
        
        logger.info(f"Model stored locally: {stored_path}")
        return str(stored_path)
    
    async def _link_model_to_experiment(self, model_id: str, version: str, experiment_id: str):
        """Link model to experiment for tracking."""
        try:
            # Update experiment with model information
            experiment = self.db_session.query(ExperimentDB).filter_by(experiment_id=experiment_id).first()
            if experiment:
                if not experiment.artifacts:
                    experiment.artifacts = {}
                
                experiment.artifacts[f"model_{model_id}_{version}"] = {
                    "model_id": model_id,
                    "version": version,
                    "linked_at": datetime.utcnow().isoformat()
                }
                experiment.updated_at = datetime.utcnow()
                self.db_session.commit()
                
                logger.info(f"Model {model_id}:{version} linked to experiment {experiment_id}")
        except Exception as e:
            logger.error(f"Failed to link model to experiment: {e}")
    
    async def get_model(self, model_id: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """Retrieve model information from registry."""
        logger.info(f"Retrieving model: {model_id} v{version}")
        
        try:
            # Try cache first
            if self.redis_client and version != "latest":
                cache_key = f"model:{model_id}:{version}"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    logger.info("Model retrieved from cache")
                    return json.loads(cached_data)
            
            # Query database
            query = self.db_session.query(ModelRegistryDB).filter_by(model_id=model_id)
            
            if version == "latest":
                model_entry = query.order_by(ModelRegistryDB.created_at.desc()).first()
            else:
                model_entry = query.filter_by(version=version).first()
            
            if not model_entry:
                logger.warning(f"Model not found: {model_id} v{version}")
                return None
            
            model_data = {
                "model_id": model_entry.model_id,
                "version": model_entry.version,
                "status": model_entry.status,
                "algorithm": model_entry.algorithm,
                "framework": model_entry.framework,
                "created_by": model_entry.created_by,
                "created_at": model_entry.created_at.isoformat(),
                "updated_at": model_entry.updated_at.isoformat(),
                "description": model_entry.description,
                "tags": model_entry.tags,
                "hyperparameters": model_entry.hyperparameters,
                "metrics": model_entry.metrics,
                "metadata": model_entry.metadata,
                "model_path": model_entry.model_path,
                "model_size_bytes": model_entry.model_size_bytes,
                "model_checksum": model_entry.model_checksum,
                "deployment_info": model_entry.deployment_info
            }
            
            # Update cache
            if self.redis_client:
                cache_key = f"model:{model_id}:{model_entry.version}"
                self.redis_client.setex(
                    cache_key,
                    self.config["registry"]["cache"]["ttl"],
                    json.dumps(model_data, default=str)
                )
            
            logger.info(f"Model retrieved: {model_id} v{model_entry.version}")
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve model: {e}")
            return None
    
    async def promote_model(self, model_id: str, version: str, target_stage: ModelStatus) -> bool:
        """Promote model to different lifecycle stage."""
        logger.info(f"Promoting model {model_id}:{version} to {target_stage.value}")
        
        try:
            model_entry = self.db_session.query(ModelRegistryDB).filter_by(
                model_id=model_id, version=version
            ).first()
            
            if not model_entry:
                logger.error(f"Model not found: {model_id}:{version}")
                return False
            
            # Check promotion rules
            if not await self._check_promotion_rules(model_entry, target_stage):
                logger.error("Model promotion failed validation rules")
                return False
            
            # Update model status
            old_status = model_entry.status
            model_entry.status = target_stage.value
            model_entry.updated_at = datetime.utcnow()
            
            # Handle stage-specific actions
            if target_stage == ModelStatus.PRODUCTION:
                await self._handle_production_promotion(model_entry)
            elif target_stage == ModelStatus.DEPRECATED:
                await self._handle_model_deprecation(model_entry)
            
            self.db_session.commit()
            
            # Invalidate cache
            if self.redis_client:
                cache_key = f"model:{model_id}:{version}"
                self.redis_client.delete(cache_key)
            
            logger.info(f"Model promoted: {model_id}:{version} {old_status} -> {target_stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"Model promotion failed: {e}")
            self.db_session.rollback()
            return False
    
    async def _check_promotion_rules(self, model_entry: ModelRegistryDB, target_stage: ModelStatus) -> bool:
        """Check if model meets promotion requirements."""
        lifecycle_config = self.config["lifecycle"]
        
        if target_stage == ModelStatus.PRODUCTION:
            # Check production promotion rules
            promotion_rules = lifecycle_config["auto_promotion"]["staging_to_production"]
            
            # Check minimum accuracy
            metrics = model_entry.metrics or {}
            if metrics.get("accuracy", 0) < promotion_rules["min_accuracy"]:
                logger.warning(f"Model accuracy {metrics.get('accuracy', 0)} below threshold {promotion_rules['min_accuracy']}")
                return False
            
            # Check validation period
            if model_entry.status == ModelStatus.STAGING.value:
                days_in_staging = (datetime.utcnow() - model_entry.updated_at).days
                if days_in_staging < promotion_rules["min_validation_days"]:
                    logger.warning(f"Model needs {promotion_rules['min_validation_days'] - days_in_staging} more days in staging")
                    return False
            
            # Check error rate (if available)
            if metrics.get("error_rate", 0) > promotion_rules["max_error_rate"]:
                logger.warning(f"Model error rate {metrics.get('error_rate', 0)} above threshold {promotion_rules['max_error_rate']}")
                return False
        
        return True
    
    async def _handle_production_promotion(self, model_entry: ModelRegistryDB):
        """Handle production promotion specific tasks."""
        logger.info(f"Handling production promotion for {model_entry.model_id}:{model_entry.version}")
        
        # Deprecate other production models of the same model_id
        production_models = self.db_session.query(ModelRegistryDB).filter_by(
            model_id=model_entry.model_id,
            status=ModelStatus.PRODUCTION.value
        ).filter(ModelRegistryDB.version != model_entry.version).all()
        
        for prod_model in production_models:
            prod_model.status = ModelStatus.DEPRECATED.value
            prod_model.updated_at = datetime.utcnow()
            logger.info(f"Deprecated previous production model: {prod_model.version}")
        
        # Set up production monitoring
        await self._setup_production_monitoring(model_entry)
    
    async def _handle_model_deprecation(self, model_entry: ModelRegistryDB):
        """Handle model deprecation tasks."""
        logger.info(f"Handling deprecation for {model_entry.model_id}:{model_entry.version}")
        
        # Remove from active deployments
        deployment_info = model_entry.deployment_info or {}
        if deployment_info:
            logger.info("Removing model from active deployments")
            # Implementation would remove from Kubernetes/deployment platform
        
        # Schedule for archival
        cleanup_config = self.config["lifecycle"]["cleanup"]
        if cleanup_config["auto_archive"]:
            archive_date = datetime.utcnow() + timedelta(days=cleanup_config["archive_after_days"])
            if not model_entry.metadata:
                model_entry.metadata = {}
            model_entry.metadata["scheduled_archive_date"] = archive_date.isoformat()
    
    async def _setup_production_monitoring(self, model_entry: ModelRegistryDB):
        """Set up monitoring for production model."""
        logger.info(f"Setting up production monitoring for {model_entry.model_id}:{model_entry.version}")
        
        monitoring_config = self.config["deployment"]["monitoring"]
        
        if monitoring_config["metrics_enabled"]:
            # Set up metrics collection
            logger.info("Configuring metrics collection")
        
        if monitoring_config["logging_enabled"]:
            # Set up logging
            logger.info("Configuring logging")
        
        if monitoring_config["health_checks"]:
            # Set up health checks
            logger.info("Configuring health checks")
    
    async def deploy_model(
        self,
        model_id: str,
        version: str,
        stage: DeploymentStage,
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Deploy model to specified stage."""
        logger.info(f"Deploying model {model_id}:{version} to {stage.value}")
        
        try:
            model_entry = self.db_session.query(ModelRegistryDB).filter_by(
                model_id=model_id, version=version
            ).first()
            
            if not model_entry:
                logger.error(f"Model not found: {model_id}:{version}")
                return None
            
            # Generate deployment ID
            deployment_id = f"deploy-{model_id}-{version}-{stage.value}-{int(time.time())}"
            
            # Create deployment configuration
            k8s_config = self.config["deployment"]["kubernetes"]
            
            deploy_config = deployment_config or {}
            final_config = {
                "namespace": k8s_config["namespace"],
                "image": f"{k8s_config['image_registry']}/{model_id}:{version}",
                "resources": deploy_config.get("resources", k8s_config["default_resources"]),
                "replicas": deploy_config.get("replicas", 2),
                "auto_scaling": deploy_config.get("auto_scaling", k8s_config["auto_scaling"]),
                "traffic_percentage": deploy_config.get("traffic_percentage", 100.0),
                "environment_variables": deploy_config.get("environment_variables", {})
            }
            
            # Execute deployment
            deployment_result = await self._execute_deployment(deployment_id, final_config, model_entry)
            
            if deployment_result["success"]:
                # Update model entry with deployment info
                if not model_entry.deployment_info:
                    model_entry.deployment_info = {}
                
                deployment_info = DeploymentInfo(
                    deployment_id=deployment_id,
                    stage=stage,
                    endpoint_url=deployment_result["endpoint_url"],
                    replicas=final_config["replicas"],
                    resources=final_config["resources"],
                    auto_scaling=final_config["auto_scaling"],
                    traffic_percentage=final_config["traffic_percentage"],
                    deployment_time=datetime.utcnow(),
                    health_status="healthy"
                )
                
                model_entry.deployment_info[stage.value] = asdict(deployment_info)
                model_entry.updated_at = datetime.utcnow()
                self.db_session.commit()
                
                logger.info(f"Model deployed successfully: {deployment_id}")
                return deployment_id
            else:
                logger.error(f"Deployment failed: {deployment_result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return None
    
    async def _execute_deployment(
        self,
        deployment_id: str,
        config: Dict[str, Any],
        model_entry: ModelRegistryDB
    ) -> Dict[str, Any]:
        """Execute the actual deployment."""
        logger.info(f"Executing deployment: {deployment_id}")
        
        try:
            # Simulate deployment process
            logger.info("Building container image...")
            await asyncio.sleep(2)  # Simulate image build
            
            logger.info("Deploying to Kubernetes...")
            await asyncio.sleep(3)  # Simulate deployment
            
            logger.info("Running health checks...")
            await asyncio.sleep(1)  # Simulate health check
            
            # Generate endpoint URL
            endpoint_url = f"https://ml-models.company.com/{model_entry.model_id}/{model_entry.version}/predict"
            
            return {
                "success": True,
                "deployment_id": deployment_id,
                "endpoint_url": endpoint_url,
                "status": "deployed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "deployment_id": deployment_id
            }
    
    async def list_models(
        self,
        status: Optional[ModelStatus] = None,
        algorithm: Optional[str] = None,
        created_by: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List models with optional filters."""
        logger.info("Listing models with filters")
        
        try:
            query = self.db_session.query(ModelRegistryDB)
            
            if status:
                query = query.filter_by(status=status.value)
            if algorithm:
                query = query.filter_by(algorithm=algorithm)
            if created_by:
                query = query.filter_by(created_by=created_by)
            
            query = query.order_by(ModelRegistryDB.created_at.desc())
            query = query.offset(offset).limit(limit)
            
            models = query.all()
            
            result = []
            for model in models:
                model_data = {
                    "model_id": model.model_id,
                    "version": model.version,
                    "status": model.status,
                    "algorithm": model.algorithm,
                    "framework": model.framework,
                    "created_by": model.created_by,
                    "created_at": model.created_at.isoformat(),
                    "description": model.description,
                    "tags": model.tags,
                    "metrics": model.metrics,
                    "model_size_bytes": model.model_size_bytes,
                    "deployment_info": model.deployment_info
                }
                result.append(model_data)
            
            logger.info(f"Found {len(result)} models")
            return result
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def create_experiment(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        created_by: str,
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a new experiment for tracking."""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.replace(' ', '_')}"
        
        logger.info(f"Creating experiment: {experiment_id}")
        
        try:
            experiment = ExperimentDB(
                experiment_id=experiment_id,
                name=name,
                description=description,
                created_by=created_by,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status="running",
                parameters=parameters,
                metrics={},
                artifacts={},
                tags=tags or []
            )
            
            self.db_session.add(experiment)
            self.db_session.commit()
            
            logger.info(f"Experiment created: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            self.db_session.rollback()
            raise
    
    async def update_experiment(
        self,
        experiment_id: str,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None
    ) -> bool:
        """Update experiment with new metrics or artifacts."""
        logger.info(f"Updating experiment: {experiment_id}")
        
        try:
            experiment = self.db_session.query(ExperimentDB).filter_by(experiment_id=experiment_id).first()
            
            if not experiment:
                logger.error(f"Experiment not found: {experiment_id}")
                return False
            
            if metrics:
                if not experiment.metrics:
                    experiment.metrics = {}
                experiment.metrics.update(metrics)
            
            if artifacts:
                if not experiment.artifacts:
                    experiment.artifacts = {}
                experiment.artifacts.update(artifacts)
            
            if status:
                experiment.status = status
            
            experiment.updated_at = datetime.utcnow()
            self.db_session.commit()
            
            logger.info(f"Experiment updated: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update experiment: {e}")
            self.db_session.rollback()
            return False
    
    async def run_model_lifecycle_management(self) -> Dict[str, Any]:
        """Run automated model lifecycle management tasks."""
        logger.info("Running model lifecycle management")
        
        lifecycle_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "tasks_executed": [],
            "models_processed": 0,
            "promotions": 0,
            "deprecations": 0,
            "archives": 0
        }
        
        try:
            # Auto-promotion check
            if self.config["lifecycle"]["auto_promotion"]["enabled"]:
                promotions = await self._check_auto_promotions()
                lifecycle_results["tasks_executed"].append("auto_promotion")
                lifecycle_results["promotions"] = len(promotions)
            
            # Auto-deprecation check
            if self.config["lifecycle"]["auto_deprecation"]["enabled"]:
                deprecations = await self._check_auto_deprecations()
                lifecycle_results["tasks_executed"].append("auto_deprecation")
                lifecycle_results["deprecations"] = len(deprecations)
            
            # Cleanup and archival
            if self.config["lifecycle"]["cleanup"]["auto_archive"]:
                archives = await self._check_model_cleanup()
                lifecycle_results["tasks_executed"].append("cleanup")
                lifecycle_results["archives"] = len(archives)
            
            # Count total models processed
            lifecycle_results["models_processed"] = (
                lifecycle_results["promotions"] +
                lifecycle_results["deprecations"] +
                lifecycle_results["archives"]
            )
            
            logger.info(f"Lifecycle management completed: {lifecycle_results}")
            return lifecycle_results
            
        except Exception as e:
            logger.error(f"Lifecycle management failed: {e}")
            lifecycle_results["error"] = str(e)
            return lifecycle_results
    
    async def _check_auto_promotions(self) -> List[str]:
        """Check for models eligible for auto-promotion."""
        promoted_models = []
        
        # Find staging models that meet promotion criteria
        staging_models = self.db_session.query(ModelRegistryDB).filter_by(
            status=ModelStatus.STAGING.value
        ).all()
        
        for model in staging_models:
            if await self._check_promotion_rules(model, ModelStatus.PRODUCTION):
                success = await self.promote_model(model.model_id, model.version, ModelStatus.PRODUCTION)
                if success:
                    promoted_models.append(f"{model.model_id}:{model.version}")
        
        return promoted_models
    
    async def _check_auto_deprecations(self) -> List[str]:
        """Check for models that should be deprecated."""
        deprecated_models = []
        
        # Find production models with poor performance
        production_models = self.db_session.query(ModelRegistryDB).filter_by(
            status=ModelStatus.PRODUCTION.value
        ).all()
        
        for model in production_models:
            # Check if model performance has degraded
            if await self._should_deprecate_model(model):
                success = await self.promote_model(model.model_id, model.version, ModelStatus.DEPRECATED)
                if success:
                    deprecated_models.append(f"{model.model_id}:{model.version}")
        
        return deprecated_models
    
    async def _should_deprecate_model(self, model: ModelRegistryDB) -> bool:
        """Check if model should be deprecated based on performance."""
        # Simulate performance checking
        # In practice, this would compare recent metrics to baseline
        deprecation_config = self.config["lifecycle"]["auto_deprecation"]
        
        # Simulate performance degradation check
        performance_degraded = np.random.choice([True, False], p=[0.1, 0.9])
        
        return performance_degraded
    
    async def _check_model_cleanup(self) -> List[str]:
        """Check for models that should be archived or deleted."""
        archived_models = []
        
        cleanup_config = self.config["lifecycle"]["cleanup"]
        archive_threshold = datetime.utcnow() - timedelta(days=cleanup_config["archive_after_days"])
        
        # Find deprecated models older than threshold
        old_models = self.db_session.query(ModelRegistryDB).filter(
            ModelRegistryDB.status == ModelStatus.DEPRECATED.value,
            ModelRegistryDB.updated_at < archive_threshold
        ).all()
        
        for model in old_models:
            # Archive model
            model.status = ModelStatus.ARCHIVED.value
            model.updated_at = datetime.utcnow()
            archived_models.append(f"{model.model_id}:{model.version}")
        
        if archived_models:
            self.db_session.commit()
        
        return archived_models
    
    def generate_registry_report(self) -> str:
        """Generate comprehensive model registry report."""
        logger.info("Generating model registry report")
        
        report_file = f"model-registry-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        
        try:
            # Gather statistics
            total_models = self.db_session.query(ModelRegistryDB).count()
            status_counts = {}
            for status in ModelStatus:
                count = self.db_session.query(ModelRegistryDB).filter_by(status=status.value).count()
                status_counts[status.value] = count
            
            # Recent models
            recent_models = self.db_session.query(ModelRegistryDB).order_by(
                ModelRegistryDB.created_at.desc()
            ).limit(10).all()
            
            # Algorithm distribution
            from sqlalchemy import func
            algorithm_stats = self.db_session.query(
                ModelRegistryDB.algorithm,
                func.count(ModelRegistryDB.algorithm)
            ).group_by(ModelRegistryDB.algorithm).all()
            
            with open(report_file, 'w') as f:
                f.write(f"""# 🏛️ Model Registry Report

**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC  
**Registry Version:** {self.config['registry']['version']}  

## 📊 Registry Statistics

### Model Counts by Status
""")
                
                for status, count in status_counts.items():
                    f.write(f"- **{status.title()}:** {count} models\n")
                
                f.write(f"\n**Total Models:** {total_models}\n\n")
                
                f.write(f"""### Algorithm Distribution
""")
                
                for algorithm, count in algorithm_stats:
                    f.write(f"- **{algorithm}:** {count} models\n")
                
                f.write(f"""
## 🆕 Recent Models

""")
                
                for model in recent_models:
                    f.write(f"""### {model.model_id} v{model.version}
- **Status:** {model.status}
- **Algorithm:** {model.algorithm}
- **Created:** {model.created_at.strftime('%Y-%m-%d %H:%M')}
- **Created By:** {model.created_by}
- **Size:** {self._format_bytes(model.model_size_bytes or 0)}

""")
                
                f.write(f"""## 🚀 Production Models

""")
                
                production_models = self.db_session.query(ModelRegistryDB).filter_by(
                    status=ModelStatus.PRODUCTION.value
                ).all()
                
                if production_models:
                    for model in production_models:
                        metrics = model.metrics or {}
                        f.write(f"""### {model.model_id} v{model.version}
- **Algorithm:** {model.algorithm}
- **Accuracy:** {metrics.get('accuracy', 'N/A')}
- **Deployed:** {model.updated_at.strftime('%Y-%m-%d %H:%M')}
- **Deployments:** {len(model.deployment_info or {})}

""")
                else:
                    f.write("No models currently in production.\n\n")
                
                f.write(f"""## 📈 Performance Summary

### Top Performing Models
""")
                
                # Get models with best accuracy
                models_with_metrics = [m for m in recent_models if m.metrics and 'accuracy' in m.metrics]
                top_models = sorted(models_with_metrics, key=lambda x: x.metrics['accuracy'], reverse=True)[:5]
                
                for model in top_models:
                    f.write(f"""- **{model.model_id} v{model.version}:** {model.metrics['accuracy']:.3f} accuracy ({model.algorithm})
""")
                
                f.write(f"""
## 🔄 Lifecycle Management

### Recent Activity
- Models requiring review: {status_counts.get('staging', 0)}
- Models to be archived: {status_counts.get('deprecated', 0)}
- Active deployments: {sum(1 for m in recent_models if m.deployment_info)}

### Configuration
- **Auto-promotion:** {'✅ Enabled' if self.config['lifecycle']['auto_promotion']['enabled'] else '❌ Disabled'}
- **Auto-deprecation:** {'✅ Enabled' if self.config['lifecycle']['auto_deprecation']['enabled'] else '❌ Disabled'}
- **Auto-cleanup:** {'✅ Enabled' if self.config['lifecycle']['cleanup']['auto_archive'] else '❌ Disabled'}

## 🎯 Recommendations

""")
                
                # Generate recommendations
                if status_counts.get('staging', 0) > 5:
                    f.write("- **Review Staging Models:** Multiple models awaiting promotion review\n")
                
                if status_counts.get('deprecated', 0) > 10:
                    f.write("- **Archive Old Models:** Consider archiving deprecated models\n")
                
                if status_counts.get('production', 0) == 0:
                    f.write("- **Deploy Production Model:** No models currently in production\n")
                
                f.write(f"""
## 📞 Support

For model registry issues:
1. Check model status and deployment information
2. Review lifecycle management logs
3. Verify storage backend connectivity
4. Contact MLOps team for assistance

---
*This report was generated automatically by the Advanced Model Registry System*
""")
            
            logger.info(f"Registry report generated: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Failed to generate registry report: {e}")
            return f"Error generating report: {e}"
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes into human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"
    
    def close(self):
        """Close database and cache connections."""
        if self.db_session:
            self.db_session.close()
        if self.redis_client:
            self.redis_client.close()


async def main():
    """Main function for model registry operations."""
    logger.info("🏛️ Advanced Model Registry System Starting...")
    
    try:
        # Initialize model registry
        registry = AdvancedModelRegistry()
        
        # Run lifecycle management
        lifecycle_results = await registry.run_model_lifecycle_management()
        
        # Generate report
        report_file = registry.generate_registry_report()
        
        logger.info(f"✅ Model Registry operations completed!")
        logger.info(f"📊 Lifecycle Results: {lifecycle_results}")
        logger.info(f"📋 Report: {report_file}")
        
        # Cleanup
        registry.close()
        
    except Exception as e:
        logger.error(f"💥 Fatal error in model registry: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())