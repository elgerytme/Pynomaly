"""Advanced ML operations interfaces for model versioning and A/B testing."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime

from ..entities.model_version import (
    ModelVersion, ABTestExperiment, ABTestResult, ModelMetrics,
    ModelRegistryEntry, FeatureDefinition, FeatureStore, ModelStatus
)

class ModelVersioningPort(ABC):
    """Port for model versioning operations."""
    
    @abstractmethod
    async def create_model_version(
        self, 
        model_name: str,
        version_number: str,
        model_data: bytes,
        metadata: Dict[str, Any]
    ) -> ModelVersion:
        """Create a new model version."""
        pass
    
    @abstractmethod
    async def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get specific model version."""
        pass
    
    @abstractmethod
    async def list_model_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        pass
    
    @abstractmethod
    async def promote_model_version(self, version_id: str, target_status: ModelStatus) -> bool:
        """Promote model version to different environment."""
        pass
    
    @abstractmethod
    async def rollback_model_version(self, model_name: str, target_version: str) -> bool:
        """Rollback model to previous version."""
        pass
    
    @abstractmethod
    async def compare_model_versions(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """Compare performance metrics between model versions."""
        pass

class ABTestingPort(ABC):
    """Port for A/B testing operations."""
    
    @abstractmethod
    async def create_experiment(self, experiment: ABTestExperiment) -> str:
        """Create new A/B testing experiment."""
        pass
    
    @abstractmethod
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start running A/B test experiment."""
        pass
    
    @abstractmethod
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop running A/B test experiment."""
        pass
    
    @abstractmethod
    async def get_experiment_results(self, experiment_id: str) -> Optional[ABTestResult]:
        """Get results of A/B test experiment."""
        pass
    
    @abstractmethod
    async def route_prediction(
        self, 
        experiment_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route prediction request based on A/B test configuration."""
        pass
    
    @abstractmethod
    async def record_experiment_outcome(
        self,
        experiment_id: str,
        prediction_id: str,
        outcome: Dict[str, Any]
    ) -> bool:
        """Record outcome for A/B test analysis."""
        pass

class ModelRegistryPort(ABC):
    """Port for model registry operations."""
    
    @abstractmethod
    async def register_model(self, model_entry: ModelRegistryEntry) -> str:
        """Register new model in registry."""
        pass
    
    @abstractmethod
    async def get_model(self, model_id: str) -> Optional[ModelRegistryEntry]:
        """Get model from registry."""
        pass
    
    @abstractmethod
    async def search_models(self, criteria: Dict[str, Any]) -> List[ModelRegistryEntry]:
        """Search models by criteria."""
        pass
    
    @abstractmethod
    async def update_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """Update model metadata."""
        pass
    
    @abstractmethod
    async def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get current production version of model."""
        pass
    
    @abstractmethod
    async def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get complete lineage and history of model."""
        pass

class ModelMonitoringPort(ABC):
    """Port for model performance monitoring."""
    
    @abstractmethod
    async def record_prediction_metrics(self, metrics: ModelMetrics) -> bool:
        """Record real-time prediction metrics."""
        pass
    
    @abstractmethod
    async def detect_model_drift(self, model_version: str) -> Dict[str, Any]:
        """Detect statistical drift in model performance."""
        pass
    
    @abstractmethod
    async def get_model_performance(
        self, 
        model_version: str,
        time_range: Dict[str, datetime]
    ) -> List[ModelMetrics]:
        """Get model performance metrics over time."""
        pass
    
    @abstractmethod
    async def trigger_retraining_alert(self, model_version: str, reason: str) -> bool:
        """Trigger alert for model retraining."""
        pass
    
    @abstractmethod
    async def calculate_model_health_score(self, model_version: str) -> float:
        """Calculate overall health score for model."""
        pass

class FeatureStorePort(ABC):
    """Port for feature store operations."""
    
    @abstractmethod
    async def create_feature_store(self, feature_store: FeatureStore) -> str:
        """Create new feature store."""
        pass
    
    @abstractmethod
    async def add_feature_definition(
        self, 
        store_name: str,
        feature: FeatureDefinition
    ) -> bool:
        """Add feature definition to store."""
        pass
    
    @abstractmethod
    async def get_feature_values(
        self,
        store_name: str,
        feature_names: List[str],
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Get feature values for entities."""
        pass
    
    @abstractmethod
    async def get_feature_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get lineage and dependencies for feature."""
        pass
    
    @abstractmethod
    async def validate_feature_consistency(
        self,
        store_name: str,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Validate consistency of features."""
        pass

class AutoMLPort(ABC):
    """Port for automated machine learning operations."""
    
    @abstractmethod
    async def start_automl_experiment(
        self,
        dataset: str,
        target_column: str,
        experiment_config: Dict[str, Any]
    ) -> str:
        """Start automated ML experiment."""
        pass
    
    @abstractmethod
    async def get_automl_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results from AutoML experiment."""
        pass
    
    @abstractmethod
    async def deploy_best_model(self, experiment_id: str) -> str:
        """Deploy best model from AutoML experiment."""
        pass