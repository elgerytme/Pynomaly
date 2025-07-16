"""Entity Mappers

Mappers to convert between domain entities and ORM models.
"""

from typing import Optional, Dict, Any

from pynomaly_mlops.domain.entities.model import Model, ModelStatus, ModelType
from pynomaly_mlops.domain.value_objects.semantic_version import SemanticVersion
from pynomaly_mlops.domain.value_objects.model_metrics import ModelMetrics

from .models import ModelORM


class ModelMapper:
    """Mapper for Model entity and ModelORM."""
    
    def domain_to_orm(self, model: Model) -> ModelORM:
        """Convert domain Model to ModelORM.
        
        Args:
            model: Domain model entity
            
        Returns:
            ModelORM instance
        """
        return ModelORM(
            id=model.id,
            name=model.name,
            
            # Version components
            version_major=model.version.major,
            version_minor=model.version.minor,
            version_patch=model.version.patch,
            version_prerelease=model.version.prerelease,
            version_build=model.version.build,
            
            # Model metadata
            model_type=model.model_type.value,
            status=model.status.value,
            description=model.description,
            
            # Lifecycle tracking
            created_at=model.created_at,
            updated_at=model.updated_at,
            created_by=model.created_by,
            
            # Model storage
            artifact_uri=model.artifact_uri,
            size_bytes=model.size_bytes,
            checksum=model.checksum,
            
            # Performance metrics
            metrics=self._metrics_to_dict(model.metrics) if model.metrics else None,
            validation_metrics=self._metrics_to_dict(model.validation_metrics) if model.validation_metrics else None,
            
            # Model configuration
            hyperparameters=model.hyperparameters,
            feature_schema=model.feature_schema,
            training_config=model.training_config,
            
            # Deployment tracking
            deployment_count=model.deployment_count,
            current_stage=model.current_stage,
            
            # Lineage
            parent_model_id=model.parent_model_id,
            experiment_id=model.experiment_id,
        )
    
    def orm_to_domain(self, model_orm: ModelORM) -> Model:
        """Convert ModelORM to domain Model.
        
        Args:
            model_orm: ORM model instance
            
        Returns:
            Domain model entity
        """
        # Reconstruct semantic version
        version = SemanticVersion(
            major=model_orm.version_major,
            minor=model_orm.version_minor,
            patch=model_orm.version_patch,
            prerelease=model_orm.version_prerelease,
            build=model_orm.version_build
        )
        
        # Convert metrics from dict
        metrics = self._dict_to_metrics(model_orm.metrics) if model_orm.metrics else None
        validation_metrics = self._dict_to_metrics(model_orm.validation_metrics) if model_orm.validation_metrics else None
        
        return Model(
            id=model_orm.id,
            name=model_orm.name,
            version=version,
            model_type=ModelType(model_orm.model_type),
            status=ModelStatus(model_orm.status),
            description=model_orm.description,
            
            # Lifecycle tracking
            created_at=model_orm.created_at,
            updated_at=model_orm.updated_at,
            created_by=model_orm.created_by,
            
            # Model storage
            artifact_uri=model_orm.artifact_uri,
            size_bytes=model_orm.size_bytes,
            checksum=model_orm.checksum,
            
            # Performance metrics
            metrics=metrics,
            validation_metrics=validation_metrics,
            
            # Model configuration
            hyperparameters=model_orm.hyperparameters or {},
            feature_schema=model_orm.feature_schema,
            training_config=model_orm.training_config,
            
            # Deployment tracking
            deployment_count=model_orm.deployment_count,
            current_stage=model_orm.current_stage,
            
            # Lineage
            parent_model_id=model_orm.parent_model_id,
            experiment_id=model_orm.experiment_id,
        )
    
    def update_orm_from_domain(self, model_orm: ModelORM, model: Model) -> None:
        """Update existing ModelORM with domain Model data.
        
        Args:
            model_orm: ORM model to update
            model: Domain model with new data
        """
        # Update basic fields
        model_orm.name = model.name
        model_orm.description = model.description
        model_orm.status = model.status.value
        model_orm.updated_at = model.updated_at
        
        # Update version
        model_orm.version_major = model.version.major
        model_orm.version_minor = model.version.minor
        model_orm.version_patch = model.version.patch
        model_orm.version_prerelease = model.version.prerelease
        model_orm.version_build = model.version.build
        
        # Update storage info
        model_orm.artifact_uri = model.artifact_uri
        model_orm.size_bytes = model.size_bytes
        model_orm.checksum = model.checksum
        
        # Update metrics
        model_orm.metrics = self._metrics_to_dict(model.metrics) if model.metrics else None
        model_orm.validation_metrics = self._metrics_to_dict(model.validation_metrics) if model.validation_metrics else None
        
        # Update configuration
        model_orm.hyperparameters = model.hyperparameters
        model_orm.feature_schema = model.feature_schema
        model_orm.training_config = model.training_config
        
        # Update deployment tracking
        model_orm.deployment_count = model.deployment_count
        model_orm.current_stage = model.current_stage
        
        # Update lineage
        model_orm.parent_model_id = model.parent_model_id
        model_orm.experiment_id = model.experiment_id
    
    def _metrics_to_dict(self, metrics: ModelMetrics) -> Dict[str, Any]:
        """Convert ModelMetrics to dictionary.
        
        Args:
            metrics: ModelMetrics instance
            
        Returns:
            Dictionary representation
        """
        return {
            # Classification metrics
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'auc_roc': metrics.auc_roc,
            'auc_pr': metrics.auc_pr,
            'false_positive_rate': metrics.false_positive_rate,
            'false_negative_rate': metrics.false_negative_rate,
            'true_positive_rate': metrics.true_positive_rate,
            'true_negative_rate': metrics.true_negative_rate,
            
            # Regression metrics
            'mse': metrics.mse,
            'rmse': metrics.rmse,
            'mae': metrics.mae,
            'r2_score': metrics.r2_score,
            'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error,
            
            # Anomaly detection metrics
            'contamination_rate': metrics.contamination_rate,
            'outlier_fraction': metrics.outlier_fraction,
            'anomaly_score_mean': metrics.anomaly_score_mean,
            'anomaly_score_std': metrics.anomaly_score_std,
            
            # Business metrics
            'business_value': metrics.business_value,
            'cost_savings': metrics.cost_savings,
            'revenue_impact': metrics.revenue_impact,
            
            # Other metrics
            'log_loss': metrics.log_loss,
            'matthews_correlation_coefficient': metrics.matthews_correlation_coefficient,
            'cohen_kappa': metrics.cohen_kappa,
            'balanced_accuracy': metrics.balanced_accuracy,
            
            # Custom metrics
            'custom_metrics': metrics.custom_metrics,
            
            # Statistical metrics
            'confidence_interval_lower': metrics.confidence_interval_lower,
            'confidence_interval_upper': metrics.confidence_interval_upper,
            'statistical_significance': metrics.statistical_significance,
            
            # Training metrics
            'training_time_seconds': metrics.training_time_seconds,
            'inference_time_milliseconds': metrics.inference_time_milliseconds,
            'memory_usage_mb': metrics.memory_usage_mb,
        }
    
    def _dict_to_metrics(self, metrics_dict: Dict[str, Any]) -> ModelMetrics:
        """Convert dictionary to ModelMetrics.
        
        Args:
            metrics_dict: Dictionary representation
            
        Returns:
            ModelMetrics instance
        """
        return ModelMetrics(
            # Classification metrics
            accuracy=metrics_dict.get('accuracy'),
            precision=metrics_dict.get('precision'),
            recall=metrics_dict.get('recall'),
            f1_score=metrics_dict.get('f1_score'),
            auc_roc=metrics_dict.get('auc_roc'),
            auc_pr=metrics_dict.get('auc_pr'),
            false_positive_rate=metrics_dict.get('false_positive_rate'),
            false_negative_rate=metrics_dict.get('false_negative_rate'),
            true_positive_rate=metrics_dict.get('true_positive_rate'),
            true_negative_rate=metrics_dict.get('true_negative_rate'),
            
            # Regression metrics
            mse=metrics_dict.get('mse'),
            rmse=metrics_dict.get('rmse'),
            mae=metrics_dict.get('mae'),
            r2_score=metrics_dict.get('r2_score'),
            mean_absolute_percentage_error=metrics_dict.get('mean_absolute_percentage_error'),
            
            # Anomaly detection metrics
            contamination_rate=metrics_dict.get('contamination_rate'),
            outlier_fraction=metrics_dict.get('outlier_fraction'),
            anomaly_score_mean=metrics_dict.get('anomaly_score_mean'),
            anomaly_score_std=metrics_dict.get('anomaly_score_std'),
            
            # Business metrics
            business_value=metrics_dict.get('business_value'),
            cost_savings=metrics_dict.get('cost_savings'),
            revenue_impact=metrics_dict.get('revenue_impact'),
            
            # Other metrics
            log_loss=metrics_dict.get('log_loss'),
            matthews_correlation_coefficient=metrics_dict.get('matthews_correlation_coefficient'),
            cohen_kappa=metrics_dict.get('cohen_kappa'),
            balanced_accuracy=metrics_dict.get('balanced_accuracy'),
            
            # Custom metrics
            custom_metrics=metrics_dict.get('custom_metrics', {}),
            
            # Statistical metrics
            confidence_interval_lower=metrics_dict.get('confidence_interval_lower'),
            confidence_interval_upper=metrics_dict.get('confidence_interval_upper'),
            statistical_significance=metrics_dict.get('statistical_significance'),
            
            # Training metrics
            training_time_seconds=metrics_dict.get('training_time_seconds'),
            inference_time_milliseconds=metrics_dict.get('inference_time_milliseconds'),
            memory_usage_mb=metrics_dict.get('memory_usage_mb'),
        )