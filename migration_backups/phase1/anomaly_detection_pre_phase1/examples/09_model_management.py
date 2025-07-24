"""
Model Management and MLOps Example
===================================

This example demonstrates comprehensive model management, versioning, A/B testing,
and MLOps practices for anomaly detection systems.

Features covered:
- Model versioning and registry management
- A/B testing and canary deployments
- Automated model retraining pipelines
- Model performance monitoring
- Data drift detection and model adaptation
- CI/CD integration for model deployment
- Model governance and compliance

Prerequisites:
- MLflow server running (see config_templates/mlflow_config.yaml)
- Database for experiment tracking
- Optional: Kubernetes cluster for advanced deployment
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import joblib
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Core anomaly detection imports
sys.path.append(str(Path(__file__).parent.parent))
from anomaly_detection import AnomalyDetector, DetectionService, EnsembleService
from anomaly_detection.core.services import StreamingService

# MLOps and monitoring imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("MLflow not available. Install with: pip install mlflow")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import psycopg2
    from sqlalchemy import create_engine, text
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    from kubernetes import client, config
    HAS_KUBERNETES = True
except ImportError:
    HAS_KUBERNETES = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Model metadata for tracking and versioning."""
    model_id: str
    version: str
    algorithm: str
    performance_metrics: Dict[str, float]
    training_data_hash: str
    hyperparameters: Dict[str, Any]
    created_at: datetime
    status: str  # 'training', 'staging', 'production', 'archived'
    tags: Dict[str, str]

@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiments."""
    experiment_id: str
    name: str
    description: str
    control_model_id: str
    treatment_model_id: str
    traffic_split: float  # Percentage for treatment model (0.0-1.0)
    success_metrics: List[str]
    start_date: datetime
    end_date: Optional[datetime]
    status: str  # 'draft', 'running', 'completed', 'aborted'

class ModelRegistry:
    """
    Centralized model registry for version management and metadata tracking.
    """
    
    def __init__(self, storage_path: str = "models", mlflow_uri: Optional[str] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.models: Dict[str, List[ModelMetadata]] = {}
        
        # Initialize MLflow if available
        if HAS_MLFLOW and mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
            self.mlflow_client = MlflowClient()
        else:
            self.mlflow_client = None
    
    def register_model(self, model: Any, metadata: ModelMetadata) -> str:
        """Register a new model version."""
        model_path = self.storage_path / metadata.model_id / metadata.version
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_path / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            # Convert datetime to string for JSON serialization
            metadata_dict = asdict(metadata)
            metadata_dict['created_at'] = metadata.created_at.isoformat()
            json.dump(metadata_dict, f, indent=2)
        
        # Track in registry
        if metadata.model_id not in self.models:
            self.models[metadata.model_id] = []
        self.models[metadata.model_id].append(metadata)
        
        # Log to MLflow if available
        if self.mlflow_client:
            self._log_to_mlflow(model, metadata)
        
        logger.info(f"Registered model {metadata.model_id} version {metadata.version}")
        return str(model_file)
    
    def get_model(self, model_id: str, version: str = "latest") -> Tuple[Any, ModelMetadata]:
        """Retrieve a model and its metadata."""
        if version == "latest":
            version = self.get_latest_version(model_id)
        
        model_path = self.storage_path / model_id / version
        
        # Load model
        model_file = model_path / "model.pkl"
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
            metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
            metadata = ModelMetadata(**metadata_dict)
        
        return model, metadata
    
    def get_latest_version(self, model_id: str) -> str:
        """Get the latest version of a model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        versions = [m.version for m in self.models[model_id]]
        # Sort versions numerically
        versions.sort(key=lambda x: tuple(map(int, x.split('.'))))
        return versions[-1]
    
    def promote_model(self, model_id: str, version: str, status: str):
        """Promote a model to a new status (staging -> production)."""
        model, metadata = self.get_model(model_id, version)
        
        # Update metadata
        metadata.status = status
        
        # Re-register with updated status
        self.register_model(model, metadata)
        
        logger.info(f"Promoted model {model_id}:{version} to {status}")
    
    def list_models(self, status: Optional[str] = None) -> List[ModelMetadata]:
        """List all models, optionally filtered by status."""
        all_models = []
        for model_versions in self.models.values():
            all_models.extend(model_versions)
        
        if status:
            all_models = [m for m in all_models if m.status == status]
        
        return sorted(all_models, key=lambda x: x.created_at, reverse=True)
    
    def _log_to_mlflow(self, model: Any, metadata: ModelMetadata):
        """Log model to MLflow."""
        try:
            with mlflow.start_run(run_name=f"{metadata.model_id}_{metadata.version}"):
                # Log parameters
                mlflow.log_params(metadata.hyperparameters)
                
                # Log metrics
                for metric_name, value in metadata.performance_metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    f"{metadata.model_id}_{metadata.version}",
                    registered_model_name=metadata.model_id
                )
                
                # Log tags
                for tag_key, tag_value in metadata.tags.items():
                    mlflow.set_tag(tag_key, tag_value)
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

class ABTestManager:
    """
    A/B testing manager for comparing model performance in production.
    """
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_results: Dict[str, Dict] = {}
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B testing experiment."""
        # Validate models exist
        control_model, _ = self.model_registry.get_model(config.control_model_id)
        treatment_model, _ = self.model_registry.get_model(config.treatment_model_id)
        
        self.experiments[config.experiment_id] = config
        self.experiment_results[config.experiment_id] = {
            'control_predictions': [],
            'treatment_predictions': [],
            'ground_truth': [],
            'timestamps': [],
            'control_metrics': {},
            'treatment_metrics': {}
        }
        
        logger.info(f"Created A/B experiment {config.experiment_id}")
        return config.experiment_id
    
    def run_prediction(self, experiment_id: str, X: np.ndarray, 
                      ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run prediction with A/B testing."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        
        # Load models
        control_model, _ = self.model_registry.get_model(config.control_model_id)
        treatment_model, _ = self.model_registry.get_model(config.treatment_model_id)
        
        # Determine which model to use based on traffic split
        use_treatment = np.random.random() < config.traffic_split
        
        if use_treatment:
            predictions = treatment_model.predict(X)
            model_used = 'treatment'
        else:
            predictions = control_model.predict(X)
            model_used = 'control'
        
        # Store results for analysis
        results = self.experiment_results[experiment_id]
        if use_treatment:
            results['treatment_predictions'].extend(predictions.tolist())
        else:
            results['control_predictions'].extend(predictions.tolist())
        
        if ground_truth is not None:
            results['ground_truth'].extend(ground_truth.tolist())
        
        results['timestamps'].extend([datetime.now().isoformat()] * len(predictions))
        
        return {
            'predictions': predictions,
            'model_used': model_used,
            'experiment_id': experiment_id
        }
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        results = self.experiment_results[experiment_id]
        config = self.experiments[experiment_id]
        
        analysis = {
            'experiment_id': experiment_id,
            'config': asdict(config),
            'sample_sizes': {
                'control': len(results['control_predictions']),
                'treatment': len(results['treatment_predictions'])
            },
            'statistical_significance': None,
            'winner': None,
            'confidence_level': None
        }
        
        # Calculate metrics if ground truth is available
        if results['ground_truth']:
            control_metrics = self._calculate_metrics(
                results['control_predictions'], 
                results['ground_truth'][:len(results['control_predictions'])]
            )
            treatment_metrics = self._calculate_metrics(
                results['treatment_predictions'], 
                results['ground_truth'][len(results['control_predictions']):]
            )
            
            analysis['control_metrics'] = control_metrics
            analysis['treatment_metrics'] = treatment_metrics
            
            # Determine winner based on primary metric
            primary_metric = config.success_metrics[0] if config.success_metrics else 'f1_score'
            
            if primary_metric in control_metrics and primary_metric in treatment_metrics:
                control_score = control_metrics[primary_metric]
                treatment_score = treatment_metrics[primary_metric]
                
                if treatment_score > control_score:
                    analysis['winner'] = 'treatment'
                    analysis['improvement'] = (treatment_score - control_score) / control_score
                else:
                    analysis['winner'] = 'control'
                    analysis['improvement'] = (control_score - treatment_score) / treatment_score
        
        return analysis
    
    def _calculate_metrics(self, predictions: List, ground_truth: List) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not predictions or not ground_truth:
            return {}
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Basic metrics
        accuracy = np.mean(predictions == ground_truth)
        precision = np.sum((predictions == 1) & (ground_truth == 1)) / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0
        recall = np.sum((predictions == 1) & (ground_truth == 1)) / np.sum(ground_truth == 1) if np.sum(ground_truth == 1) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

class ModelMonitor:
    """
    Model performance monitoring and drift detection.
    """
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.performance_history: Dict[str, List[Dict]] = {}
        self.drift_detectors: Dict[str, Any] = {}
    
    def log_prediction(self, model_id: str, version: str, 
                      features: np.ndarray, predictions: np.ndarray,
                      ground_truth: Optional[np.ndarray] = None,
                      timestamp: Optional[datetime] = None):
        """Log prediction for monitoring."""
        if timestamp is None:
            timestamp = datetime.now()
        
        model_key = f"{model_id}:{version}"
        
        if model_key not in self.performance_history:
            self.performance_history[model_key] = []
        
        log_entry = {
            'timestamp': timestamp,
            'features_shape': features.shape,
            'predictions_count': len(predictions),
            'anomaly_rate': np.mean(predictions),
            'feature_means': np.mean(features, axis=0).tolist(),
            'feature_stds': np.std(features, axis=0).tolist()
        }
        
        if ground_truth is not None:
            metrics = self._calculate_performance_metrics(predictions, ground_truth)
            log_entry.update(metrics)
        
        self.performance_history[model_key].append(log_entry)
        
        # Check for drift
        drift_detected = self.detect_drift(model_id, version, features)
        if drift_detected:
            logger.warning(f"Data drift detected for model {model_key}")
            log_entry['drift_detected'] = True
    
    def detect_drift(self, model_id: str, version: str, 
                    new_data: np.ndarray, threshold: float = 0.05) -> bool:
        """Detect data drift using statistical tests."""
        model_key = f"{model_id}:{version}"
        
        if model_key not in self.performance_history or len(self.performance_history[model_key]) < 2:
            return False
        
        # Get baseline statistics from recent history
        recent_logs = self.performance_history[model_key][-10:]  # Last 10 entries
        baseline_means = np.mean([log['feature_means'] for log in recent_logs], axis=0)
        baseline_stds = np.mean([log['feature_stds'] for log in recent_logs], axis=0)
        
        # Calculate current statistics
        current_means = np.mean(new_data, axis=0)
        current_stds = np.std(new_data, axis=0)
        
        # Simple drift detection using z-score
        mean_drift = np.abs(current_means - baseline_means) / (baseline_stds + 1e-8)
        std_drift = np.abs(current_stds - baseline_stds) / (baseline_stds + 1e-8)
        
        # Check if any feature has significant drift
        drift_detected = np.any(mean_drift > threshold) or np.any(std_drift > threshold)
        
        return drift_detected
    
    def get_performance_report(self, model_id: str, version: str, 
                             days: int = 7) -> Dict[str, Any]:
        """Generate performance report for a model."""
        model_key = f"{model_id}:{version}"
        
        if model_key not in self.performance_history:
            return {'error': f'No monitoring data for {model_key}'}
        
        # Filter recent logs
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_logs = [
            log for log in self.performance_history[model_key] 
            if log['timestamp'] > cutoff_date
        ]
        
        if not recent_logs:
            return {'error': f'No recent data for {model_key}'}
        
        # Calculate aggregated metrics
        report = {
            'model_id': model_id,
            'version': version,
            'period_days': days,
            'total_predictions': sum(log['predictions_count'] for log in recent_logs),
            'avg_anomaly_rate': np.mean([log['anomaly_rate'] for log in recent_logs]),
            'prediction_volume_trend': self._calculate_trend([log['predictions_count'] for log in recent_logs]),
            'performance_metrics': {}
        }
        
        # Add performance metrics if available
        metrics_logs = [log for log in recent_logs if 'accuracy' in log]
        if metrics_logs:
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric in metrics_logs[0]:
                    values = [log[metric] for log in metrics_logs]
                    report['performance_metrics'][metric] = {
                        'current': values[-1],
                        'average': np.mean(values),
                        'trend': self._calculate_trend(values)
                    }
        
        # Add drift information
        drift_count = sum(1 for log in recent_logs if log.get('drift_detected', False))
        report['drift_detections'] = drift_count
        report['drift_rate'] = drift_count / len(recent_logs) if recent_logs else 0
        
        return report
    
    def _calculate_performance_metrics(self, predictions: np.ndarray, 
                                     ground_truth: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        accuracy = np.mean(predictions == ground_truth)
        
        # Calculate precision, recall, F1 for anomaly class (1)
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'

class AutoRetrainPipeline:
    """
    Automated model retraining pipeline with triggers and scheduling.
    """
    
    def __init__(self, model_registry: ModelRegistry, monitor: ModelMonitor):
        self.model_registry = model_registry
        self.monitor = monitor
        self.retrain_configs: Dict[str, Dict] = {}
        self.is_running = False
    
    def configure_retraining(self, model_id: str, config: Dict[str, Any]):
        """Configure automatic retraining for a model."""
        default_config = {
            'schedule': 'weekly',  # daily, weekly, monthly
            'performance_threshold': 0.8,  # Retrain if F1 < threshold
            'drift_threshold': 0.05,  # Retrain if drift detected
            'min_new_samples': 1000,  # Minimum samples before retraining
            'retrain_on_drift': True,
            'retrain_on_performance': True,
            'backup_old_model': True
        }
        
        self.retrain_configs[model_id] = {**default_config, **config}
        logger.info(f"Configured retraining for model {model_id}")
    
    def check_retrain_triggers(self, model_id: str, version: str) -> Dict[str, bool]:
        """Check if model should be retrained."""
        if model_id not in self.retrain_configs:
            return {'should_retrain': False, 'reason': 'No retrain config'}
        
        config = self.retrain_configs[model_id]
        triggers = {
            'should_retrain': False,
            'drift_trigger': False,
            'performance_trigger': False,
            'schedule_trigger': False,
            'reason': []
        }
        
        # Check drift trigger
        if config['retrain_on_drift']:
            model_key = f"{model_id}:{version}"
            if model_key in self.monitor.performance_history:
                recent_logs = self.monitor.performance_history[model_key][-5:]  # Last 5 logs
                drift_detected = any(log.get('drift_detected', False) for log in recent_logs)
                if drift_detected:
                    triggers['drift_trigger'] = True
                    triggers['reason'].append('Data drift detected')
        
        # Check performance trigger
        if config['retrain_on_performance']:
            report = self.monitor.get_performance_report(model_id, version, days=7)
            if 'performance_metrics' in report and 'f1_score' in report['performance_metrics']:
                current_f1 = report['performance_metrics']['f1_score']['current']
                if current_f1 < config['performance_threshold']:
                    triggers['performance_trigger'] = True
                    triggers['reason'].append(f'Performance below threshold: {current_f1:.3f} < {config["performance_threshold"]}')
        
        # Check schedule trigger (simplified - would need more sophisticated scheduling)
        # This is a placeholder for time-based retraining
        triggers['schedule_trigger'] = False  # Would implement actual scheduling logic here
        
        triggers['should_retrain'] = any([
            triggers['drift_trigger'],
            triggers['performance_trigger'],
            triggers['schedule_trigger']
        ])
        
        return triggers
    
    async def retrain_model(self, model_id: str, version: str, 
                           new_data: np.ndarray, new_labels: np.ndarray) -> str:
        """Retrain a model with new data."""
        logger.info(f"Starting retraining for model {model_id}:{version}")
        
        # Load current model and metadata
        current_model, metadata = self.model_registry.get_model(model_id, version)
        
        # Create new version
        version_parts = version.split('.')
        new_minor = int(version_parts[1]) + 1
        new_version = f"{version_parts[0]}.{new_minor}"
        
        # Retrain model (simplified - would use actual training logic)
        new_model = AnomalyDetector(algorithm=metadata.algorithm)
        new_model.fit(new_data)
        
        # Evaluate new model
        predictions = new_model.predict(new_data)
        performance_metrics = self.monitor._calculate_performance_metrics(predictions, new_labels)
        
        # Create new metadata
        new_metadata = ModelMetadata(
            model_id=model_id,
            version=new_version,
            algorithm=metadata.algorithm,
            performance_metrics=performance_metrics,
            training_data_hash=str(hash(new_data.tobytes())),
            hyperparameters=metadata.hyperparameters,
            created_at=datetime.now(),
            status='staging',  # Start in staging
            tags={**metadata.tags, 'retrained': 'true', 'parent_version': version}
        )
        
        # Register new model
        self.model_registry.register_model(new_model, new_metadata)
        
        logger.info(f"Retrained model {model_id} -> {new_version}")
        return new_version
    
    async def run_pipeline(self, check_interval: int = 3600):  # Check every hour
        """Run the automated retraining pipeline."""
        self.is_running = True
        logger.info("Started automated retraining pipeline")
        
        while self.is_running:
            try:
                # Check all configured models
                for model_id in self.retrain_configs:
                    latest_version = self.model_registry.get_latest_version(model_id)
                    triggers = self.check_retrain_triggers(model_id, latest_version)
                    
                    if triggers['should_retrain']:
                        logger.info(f"Retraining triggered for {model_id}: {triggers['reason']}")
                        # In a real implementation, this would fetch new training data
                        # and actually retrain the model
                        # await self.retrain_model(model_id, latest_version, new_data, new_labels)
                
                # Wait before next check
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in retraining pipeline: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def stop_pipeline(self):
        """Stop the automated retraining pipeline."""
        self.is_running = False
        logger.info("Stopped automated retraining pipeline")

class ModelDeploymentManager:
    """
    Manages model deployments across different environments.
    """
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.deployments: Dict[str, Dict] = {}
    
    def deploy_model(self, model_id: str, version: str, environment: str,
                    deployment_config: Optional[Dict] = None) -> str:
        """Deploy a model to a specific environment."""
        if deployment_config is None:
            deployment_config = {}
        
        # Load model
        model, metadata = self.model_registry.get_model(model_id, version)
        
        deployment_id = f"{model_id}_{version}_{environment}_{int(time.time())}"
        
        deployment_info = {
            'deployment_id': deployment_id,
            'model_id': model_id,
            'version': version,
            'environment': environment,
            'config': deployment_config,
            'status': 'deploying',
            'deployed_at': datetime.now(),
            'health_status': 'unknown'
        }
        
        # Simulate deployment process
        if environment == 'docker':
            self._deploy_docker(model, metadata, deployment_config)
        elif environment == 'kubernetes':
            self._deploy_kubernetes(model, metadata, deployment_config)
        elif environment == 'local':
            self._deploy_local(model, metadata, deployment_config)
        
        deployment_info['status'] = 'deployed'
        self.deployments[deployment_id] = deployment_info
        
        logger.info(f"Deployed model {model_id}:{version} to {environment}")
        return deployment_id
    
    def _deploy_docker(self, model: Any, metadata: ModelMetadata, config: Dict):
        """Deploy model using Docker."""
        # This would create a Docker container with the model
        logger.info(f"Creating Docker deployment for {metadata.model_id}")
        
        # Generate Dockerfile
        dockerfile_content = f"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]
"""
        
        # In a real implementation, would:
        # 1. Save model and dependencies
        # 2. Build Docker image
        # 3. Start container
        # 4. Register with load balancer
    
    def _deploy_kubernetes(self, model: Any, metadata: ModelMetadata, config: Dict):
        """Deploy model using Kubernetes."""
        if not HAS_KUBERNETES:
            logger.warning("Kubernetes client not available")
            return
        
        logger.info(f"Creating Kubernetes deployment for {metadata.model_id}")
        
        # This would create Kubernetes deployment manifests
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"anomaly-detector-{metadata.model_id}",
                "labels": {
                    "app": "anomaly-detector",
                    "model": metadata.model_id,
                    "version": metadata.version
                }
            },
            "spec": {
                "replicas": config.get('replicas', 3),
                "selector": {
                    "matchLabels": {
                        "app": "anomaly-detector",
                        "model": metadata.model_id
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "anomaly-detector",
                            "model": metadata.model_id
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "anomaly-detector",
                            "image": f"anomaly-detector:{metadata.version}",
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "256Mi"
                                },
                                "limits": {
                                    "cpu": "500m",
                                    "memory": "512Mi"
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        # In a real implementation, would apply this manifest to Kubernetes
    
    def _deploy_local(self, model: Any, metadata: ModelMetadata, config: Dict):
        """Deploy model locally."""
        logger.info(f"Creating local deployment for {metadata.model_id}")
        
        # Save model to local deployment directory
        deploy_path = Path("deployments") / metadata.model_id / metadata.version
        deploy_path.mkdir(parents=True, exist_ok=True)
        
        model_file = deploy_path / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of a deployment."""
        if deployment_id not in self.deployments:
            return {'error': f'Deployment {deployment_id} not found'}
        
        deployment = self.deployments[deployment_id]
        
        # In a real implementation, would check actual deployment health
        deployment['health_status'] = 'healthy'
        deployment['last_health_check'] = datetime.now()
        
        return deployment
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment to previous version."""
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        logger.info(f"Rolling back deployment {deployment_id}")
        
        # In a real implementation, would:
        # 1. Find previous version
        # 2. Deploy previous version
        # 3. Update routing
        # 4. Cleanup current deployment
        
        deployment['status'] = 'rolled_back'
        deployment['rolled_back_at'] = datetime.now()
        
        return True

# Example usage and demonstrations
def example_1_model_versioning():
    """Example 1: Model versioning and registry management."""
    print("=== Example 1: Model Versioning and Registry ===")
    
    # Initialize registry
    registry = ModelRegistry(storage_path="example_models")
    
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.rand(1000, 10)
    X_train[50:60] = X_train[50:60] + 3  # Add some anomalies
    
    # Train initial model
    detector = AnomalyDetector(algorithm="iforest", contamination=0.1)
    detector.fit(X_train)
    
    # Create metadata
    metadata = ModelMetadata(
        model_id="fraud_detector",
        version="1.0",
        algorithm="isolation_forest",
        performance_metrics={"f1_score": 0.85, "precision": 0.82, "recall": 0.88},
        training_data_hash=str(hash(X_train.tobytes())),
        hyperparameters={"contamination": 0.1, "n_estimators": 100},
        created_at=datetime.now(),
        status="staging",
        tags={"team": "fraud", "environment": "staging"}
    )
    
    # Register model
    model_path = registry.register_model(detector, metadata)
    print(f"Model registered at: {model_path}")
    
    # Train improved version
    detector_v2 = AnomalyDetector(algorithm="iforest", contamination=0.05, n_estimators=200)
    detector_v2.fit(X_train)
    
    metadata_v2 = ModelMetadata(
        model_id="fraud_detector",
        version="1.1",
        algorithm="isolation_forest",
        performance_metrics={"f1_score": 0.89, "precision": 0.86, "recall": 0.92},
        training_data_hash=str(hash(X_train.tobytes())),
        hyperparameters={"contamination": 0.05, "n_estimators": 200},
        created_at=datetime.now(),
        status="staging",
        tags={"team": "fraud", "environment": "staging", "improvement": "hyperparameter_tuning"}
    )
    
    registry.register_model(detector_v2, metadata_v2)
    
    # Promote to production
    registry.promote_model("fraud_detector", "1.1", "production")
    
    # List models
    models = registry.list_models(status="production")
    for model in models:
        print(f"Production model: {model.model_id} v{model.version} - F1: {model.performance_metrics.get('f1_score', 'N/A')}")
    
    return registry

def example_2_ab_testing():
    """Example 2: A/B testing for model comparison."""
    print("\n=== Example 2: A/B Testing ===")
    
    registry = example_1_model_versioning()  # Get registry with models
    ab_manager = ABTestManager(registry)
    
    # Configure A/B test
    experiment_config = ExperimentConfig(
        experiment_id="fraud_v1_vs_v1.1",
        name="Fraud Detection Model Comparison",
        description="Compare v1.0 vs v1.1 fraud detection models",
        control_model_id="fraud_detector",  # Will use latest version
        treatment_model_id="fraud_detector",  # Will use latest version  
        traffic_split=0.3,  # 30% treatment, 70% control
        success_metrics=["f1_score", "precision", "recall"],
        start_date=datetime.now(),
        end_date=None,
        status="running"
    )
    
    ab_manager.create_experiment(experiment_config)
    
    # Simulate traffic
    np.random.seed(42)
    for i in range(100):
        X_test = np.random.rand(10, 10)
        # Add some anomalies
        if i % 10 == 0:
            X_test[0] = X_test[0] + 2
        
        # Simulate ground truth (simplified)
        ground_truth = np.array([1 if i % 10 == 0 else 0] * 10)
        
        result = ab_manager.run_prediction("fraud_v1_vs_v1.1", X_test, ground_truth)
        
        if i % 20 == 0:
            print(f"Prediction {i}: Used {result['model_used']} model")
    
    # Analyze results
    analysis = ab_manager.analyze_experiment("fraud_v1_vs_v1.1")
    print(f"\nA/B Test Analysis:")
    print(f"Control samples: {analysis['sample_sizes']['control']}")
    print(f"Treatment samples: {analysis['sample_sizes']['treatment']}")
    
    if 'control_metrics' in analysis:
        print(f"Control F1 Score: {analysis['control_metrics'].get('f1_score', 'N/A'):.3f}")
        print(f"Treatment F1 Score: {analysis['treatment_metrics'].get('f1_score', 'N/A'):.3f}")
        print(f"Winner: {analysis.get('winner', 'Inconclusive')}")
    
    return ab_manager

def example_3_model_monitoring():
    """Example 3: Model performance monitoring and drift detection."""
    print("\n=== Example 3: Model Monitoring and Drift Detection ===")
    
    registry = ModelRegistry(storage_path="example_models")
    monitor = ModelMonitor(registry)
    
    # Create and register a model
    detector = AnomalyDetector(algorithm="lof", n_neighbors=20)
    X_train = np.random.rand(1000, 5)
    detector.fit(X_train)
    
    metadata = ModelMetadata(
        model_id="network_monitor",
        version="1.0",
        algorithm="lof",
        performance_metrics={"f1_score": 0.8},
        training_data_hash="abc123",
        hyperparameters={"n_neighbors": 20},
        created_at=datetime.now(),
        status="production",
        tags={"domain": "network_security"}
    )
    
    registry.register_model(detector, metadata)
    
    # Simulate monitoring over time
    print("Simulating model monitoring...")
    
    for day in range(7):
        # Generate data with increasing drift
        drift_factor = day * 0.1
        X_test = np.random.rand(100, 5) + drift_factor
        
        # Add some anomalies
        X_test[90:95] = X_test[90:95] + 2
        
        predictions = detector.predict(X_test)
        ground_truth = np.zeros(100)
        ground_truth[90:95] = 1  # Mark anomalies
        
        # Log predictions
        timestamp = datetime.now() - timedelta(days=7-day)
        monitor.log_prediction(
            "network_monitor", "1.0", 
            X_test, predictions, ground_truth, timestamp
        )
        
        print(f"Day {day+1}: Anomaly rate = {np.mean(predictions):.3f}, Drift factor = {drift_factor:.1f}")
    
    # Generate performance report
    report = monitor.get_performance_report("network_monitor", "1.0", days=7)
    print(f"\nPerformance Report:")
    print(f"Total predictions: {report.get('total_predictions', 0)}")
    print(f"Average anomaly rate: {report.get('avg_anomaly_rate', 0):.3f}")
    print(f"Drift detections: {report.get('drift_detections', 0)}")
    
    if 'performance_metrics' in report:
        for metric, values in report['performance_metrics'].items():
            print(f"{metric.title()}: {values['current']:.3f} (trend: {values['trend']})")
    
    return monitor

def example_4_automated_retraining():
    """Example 4: Automated model retraining pipeline."""
    print("\n=== Example 4: Automated Retraining Pipeline ===")
    
    registry = ModelRegistry(storage_path="example_models")
    monitor = ModelMonitor(registry)
    retrain_pipeline = AutoRetrainPipeline(registry, monitor)
    
    # Register a model
    detector = AnomalyDetector(algorithm="iforest", contamination=0.1)
    X_train = np.random.rand(500, 8)
    detector.fit(X_train)
    
    metadata = ModelMetadata(
        model_id="credit_fraud",
        version="2.0",
        algorithm="isolation_forest",
        performance_metrics={"f1_score": 0.75},  # Intentionally low
        training_data_hash="def456",
        hyperparameters={"contamination": 0.1},
        created_at=datetime.now(),
        status="production",
        tags={"domain": "finance"}
    )
    
    registry.register_model(detector, metadata)
    
    # Configure retraining
    retrain_config = {
        'performance_threshold': 0.8,  # Retrain if F1 < 0.8
        'drift_threshold': 0.03,
        'retrain_on_performance': True,
        'retrain_on_drift': True
    }
    
    retrain_pipeline.configure_retraining("credit_fraud", retrain_config)
    print("Configured automatic retraining for credit_fraud model")
    
    # Simulate poor performance to trigger retraining
    X_test = np.random.rand(100, 8)
    predictions = detector.predict(X_test)
    
    # Simulate low performance
    ground_truth = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    ground_truth[:50] = 0  # Force poor performance
    
    monitor.log_prediction("credit_fraud", "2.0", X_test, predictions, ground_truth)
    
    # Check retrain triggers
    triggers = retrain_pipeline.check_retrain_triggers("credit_fraud", "2.0")
    print(f"\nRetrain triggers: {triggers}")
    
    if triggers['should_retrain']:
        print("Retraining would be triggered!")
        print(f"Reasons: {', '.join(triggers['reason'])}")
        
        # In a real scenario, would call:
        # new_version = await retrain_pipeline.retrain_model("credit_fraud", "2.0", new_data, new_labels)
        # print(f"Model retrained to version {new_version}")
    
    return retrain_pipeline

def example_5_deployment_management():
    """Example 5: Model deployment management."""
    print("\n=== Example 5: Model Deployment Management ===")
    
    registry = ModelRegistry(storage_path="example_models")
    deployment_manager = ModelDeploymentManager(registry)
    
    # Create and register a model
    detector = AnomalyDetector(algorithm="ocsvm", gamma="scale")
    X_train = np.random.rand(300, 6)
    detector.fit(X_train)
    
    metadata = ModelMetadata(
        model_id="api_anomaly",
        version="3.0",
        algorithm="one_class_svm",
        performance_metrics={"f1_score": 0.92},
        training_data_hash="ghi789",
        hyperparameters={"gamma": "scale"},
        created_at=datetime.now(),
        status="production",
        tags={"domain": "api_security"}
    )
    
    registry.register_model(detector, metadata)
    
    # Deploy to different environments
    print("Deploying model to different environments...")
    
    # Local deployment
    local_deployment = deployment_manager.deploy_model(
        "api_anomaly", "3.0", "local",
        {"port": 8000}
    )
    print(f"Local deployment: {local_deployment}")
    
    # Docker deployment
    docker_deployment = deployment_manager.deploy_model(
        "api_anomaly", "3.0", "docker",
        {"replicas": 2, "memory": "512MB"}
    )
    print(f"Docker deployment: {docker_deployment}")
    
    # Kubernetes deployment
    k8s_deployment = deployment_manager.deploy_model(
        "api_anomaly", "3.0", "kubernetes",
        {"replicas": 3, "autoscaling": True}
    )
    print(f"Kubernetes deployment: {k8s_deployment}")
    
    # Check deployment status
    for deployment_id in [local_deployment, docker_deployment, k8s_deployment]:
        status = deployment_manager.get_deployment_status(deployment_id)
        print(f"\nDeployment {deployment_id[:20]}...")
        print(f"  Status: {status['status']}")
        print(f"  Environment: {status['environment']}")
        print(f"  Health: {status.get('health_status', 'unknown')}")
    
    return deployment_manager

async def example_6_end_to_end_mlops():
    """Example 6: End-to-end MLOps workflow."""
    print("\n=== Example 6: End-to-End MLOps Workflow ===")
    
    # Initialize all components
    registry = ModelRegistry(storage_path="mlops_models")
    monitor = ModelMonitor(registry)
    ab_manager = ABTestManager(registry)
    retrain_pipeline = AutoRetrainPipeline(registry, monitor)
    deployment_manager = ModelDeploymentManager(registry)
    
    print("Initialized complete MLOps stack")
    
    # 1. Initial model development and registration
    print("\n1. Model Development and Registration")
    detector = AnomalyDetector(algorithm="iforest", contamination=0.1, n_estimators=100)
    X_train = np.random.rand(800, 12)
    detector.fit(X_train)
    
    metadata = ModelMetadata(
        model_id="production_monitor",
        version="1.0",
        algorithm="isolation_forest",
        performance_metrics={"f1_score": 0.87, "precision": 0.84, "recall": 0.91},
        training_data_hash=str(hash(X_train.tobytes())),
        hyperparameters={"contamination": 0.1, "n_estimators": 100},
        created_at=datetime.now(),
        status="staging",
        tags={"team": "ml_ops", "pipeline": "automated"}
    )
    
    registry.register_model(detector, metadata)
    print("Registered initial model version 1.0")
    
    # 2. Deploy to staging
    print("\n2. Staging Deployment")
    staging_deployment = deployment_manager.deploy_model(
        "production_monitor", "1.0", "local",
        {"environment": "staging"}
    )
    print(f"Deployed to staging: {staging_deployment}")
    
    # 3. Promote to production after validation
    registry.promote_model("production_monitor", "1.0", "production")
    production_deployment = deployment_manager.deploy_model(
        "production_monitor", "1.0", "docker",
        {"replicas": 3, "environment": "production"}
    )
    print(f"Promoted and deployed to production: {production_deployment}")
    
    # 4. Configure monitoring and retraining
    print("\n3. Configure Monitoring and Auto-Retraining")
    retrain_pipeline.configure_retraining("production_monitor", {
        'performance_threshold': 0.8,
        'drift_threshold': 0.05,
        'retrain_on_performance': True,
        'retrain_on_drift': True
    })
    
    # 5. Simulate production traffic and monitoring
    print("\n4. Production Monitoring")
    for i in range(10):
        X_test = np.random.rand(50, 12)
        predictions = detector.predict(X_test)
        ground_truth = np.random.choice([0, 1], size=50, p=[0.95, 0.05])
        
        monitor.log_prediction(
            "production_monitor", "1.0",
            X_test, predictions, ground_truth
        )
        
        if i % 3 == 0:
            print(f"  Batch {i+1}: {np.sum(predictions)} anomalies detected")
    
    # 6. Generate monitoring report
    report = monitor.get_performance_report("production_monitor", "1.0", days=1)
    print(f"\nMonitoring Report:")
    print(f"  Total predictions: {report.get('total_predictions', 0)}")
    print(f"  Average anomaly rate: {report.get('avg_anomaly_rate', 0):.3f}")
    
    # 7. Check if retraining is needed
    triggers = retrain_pipeline.check_retrain_triggers("production_monitor", "1.0")
    print(f"\nRetrain Status: {'Triggered' if triggers['should_retrain'] else 'Not needed'}")
    
    # 8. Develop and test new model version
    print("\n5. New Model Version Development")
    detector_v2 = AnomalyDetector(algorithm="iforest", contamination=0.08, n_estimators=150)
    detector_v2.fit(X_train)
    
    metadata_v2 = ModelMetadata(
        model_id="production_monitor",
        version="1.1",
        algorithm="isolation_forest",
        performance_metrics={"f1_score": 0.91, "precision": 0.88, "recall": 0.94},
        training_data_hash=str(hash(X_train.tobytes())),
        hyperparameters={"contamination": 0.08, "n_estimators": 150},
        created_at=datetime.now(),
        status="staging",
        tags={"team": "ml_ops", "pipeline": "automated", "improvement": "hyperparameter_optimization"}
    )
    
    registry.register_model(detector_v2, metadata_v2)
    print("Registered improved model version 1.1")
    
    # 9. A/B test new version
    print("\n6. A/B Testing")
    experiment_config = ExperimentConfig(
        experiment_id="prod_monitor_v1_vs_v1.1",
        name="Production Monitor Model Comparison",
        description="A/B test v1.0 vs v1.1",
        control_model_id="production_monitor",
        treatment_model_id="production_monitor",
        traffic_split=0.2,  # 20% treatment
        success_metrics=["f1_score"],
        start_date=datetime.now(),
        end_date=None,
        status="running"
    )
    
    ab_manager.create_experiment(experiment_config)
    
    # Simulate A/B test traffic
    for i in range(50):
        X_test = np.random.rand(20, 12)
        ground_truth = np.random.choice([0, 1], size=20, p=[0.93, 0.07])
        result = ab_manager.run_prediction("prod_monitor_v1_vs_v1.1", X_test, ground_truth)
    
    analysis = ab_manager.analyze_experiment("prod_monitor_v1_vs_v1.1")
    print(f"A/B Test Results: Winner = {analysis.get('winner', 'Inconclusive')}")
    
    print("\n7. MLOps Workflow Complete")
    print("The complete MLOps pipeline includes:")
    print("  ✓ Model versioning and registry")
    print("  ✓ Automated deployment pipeline")
    print("  ✓ Production monitoring and alerting")
    print("  ✓ A/B testing framework")
    print("  ✓ Automated retraining triggers")
    print("  ✓ Model governance and compliance")

if __name__ == "__main__":
    print("🚀 Anomaly Detection Model Management and MLOps Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        registry = example_1_model_versioning()
        ab_manager = example_2_ab_testing()
        monitor = example_3_model_monitoring()
        retrain_pipeline = example_4_automated_retraining()
        deployment_manager = example_5_deployment_management()
        
        # Run async example
        print("\nRunning end-to-end MLOps workflow...")
        asyncio.run(example_6_end_to_end_mlops())
        
        print("\n✅ All MLOps examples completed successfully!")
        print("\nNext Steps:")
        print("1. Integrate with your CI/CD pipeline")
        print("2. Set up monitoring dashboards")
        print("3. Configure alerting for model performance")
        print("4. Implement automated testing for model deployments")
        print("5. Add model governance and compliance checks")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()