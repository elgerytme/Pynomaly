"""AlgorithmConfiguration entity for machine learning algorithm configuration management."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import Field, validator

from packages.core.domain.abstractions.base_entity import BaseEntity


class AlgorithmConfiguration(BaseEntity):
    """Entity representing a machine learning algorithm configuration.
    
    This entity encapsulates all configuration settings for a machine learning
    algorithm including hyperparameters, preprocessing steps, validation settings,
    and optimization parameters.
    
    Attributes:
        config_id: Unique identifier for the configuration
        algorithm_name: Name of the machine learning algorithm
        algorithm_version: Version of the algorithm implementation
        algorithm_type: Type/category of the algorithm
        config_name: Human-readable name for this configuration
        description: Detailed description of the configuration
        
        # Core configuration
        hyperparameters: Algorithm hyperparameters
        preprocessing_config: Data preprocessing configuration
        feature_selection_config: Feature selection configuration
        validation_config: Model validation configuration
        
        # Optimization settings
        optimization_config: Hyperparameter optimization settings
        search_space: Parameter search space definitions
        optimization_objective: Optimization objective function
        optimization_direction: Minimize or maximize objective
        
        # Training configuration
        training_config: Training-specific settings
        early_stopping_config: Early stopping configuration
        checkpoint_config: Model checkpointing settings
        regularization_config: Regularization settings
        
        # Resource management
        compute_config: Computational resource settings
        memory_config: Memory management settings
        parallelization_config: Parallel processing settings
        gpu_config: GPU-specific settings
        
        # Data handling
        data_loading_config: Data loading configuration
        batch_config: Batch processing configuration
        sampling_config: Data sampling configuration
        augmentation_config: Data augmentation settings
        
        # Model architecture (for neural networks)
        architecture_config: Neural network architecture
        layer_configs: Individual layer configurations
        activation_functions: Activation function settings
        initialization_config: Weight initialization settings
        
        # Evaluation and metrics
        evaluation_config: Model evaluation settings
        metric_configs: Evaluation metric configurations
        cross_validation_config: Cross-validation settings
        holdout_config: Holdout validation configuration
        
        # Ensemble settings
        ensemble_config: Ensemble method configuration
        voting_config: Voting mechanism settings
        stacking_config: Stacking configuration
        bagging_config: Bagging configuration
        
        # Advanced settings
        interpretability_config: Model interpretability settings
        fairness_config: Fairness and bias mitigation settings
        privacy_config: Privacy preservation settings
        robustness_config: Adversarial robustness settings
        
        # Metadata and tracking
        created_by: User who created the configuration
        created_at: When the configuration was created
        updated_at: When the configuration was last updated
        version: Configuration version number
        tags: Tags for organization
        
        # Usage tracking
        usage_count: Number of times this configuration was used
        success_rate: Success rate of experiments using this config
        average_performance: Average performance metrics
        best_performance: Best performance achieved
        
        # Validation and constraints
        parameter_constraints: Parameter validation constraints
        compatibility_requirements: Compatibility requirements
        resource_requirements: Resource requirements
        
        # Comparison and optimization history
        parent_config_id: Reference to parent configuration
        optimization_history: History of optimization attempts
        performance_history: Performance history over time
        
        # Export and sharing
        is_public: Whether configuration is publicly shareable
        sharing_permissions: Sharing permissions
        export_formats: Available export formats
        
        # Deployment settings
        deployment_config: Deployment-specific settings
        serving_config: Model serving configuration
        monitoring_config: Model monitoring settings
        update_strategy: Model update strategy
    """
    
    # Core identification
    config_id: UUID = Field(default_factory=uuid4)
    algorithm_name: str = Field(..., min_length=1)
    algorithm_version: str = Field(default="1.0.0")
    algorithm_type: str = Field(..., min_length=1)
    config_name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    
    # Core configuration
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    preprocessing_config: dict[str, Any] = Field(default_factory=dict)
    feature_selection_config: dict[str, Any] = Field(default_factory=dict)
    validation_config: dict[str, Any] = Field(default_factory=dict)
    
    # Optimization settings
    optimization_config: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, dict[str, Any]] = Field(default_factory=dict)
    optimization_objective: str = Field(default="accuracy")
    optimization_direction: str = Field(default="maximize")
    
    # Training configuration
    training_config: dict[str, Any] = Field(default_factory=dict)
    early_stopping_config: dict[str, Any] = Field(default_factory=dict)
    checkpoint_config: dict[str, Any] = Field(default_factory=dict)
    regularization_config: dict[str, Any] = Field(default_factory=dict)
    
    # Resource management
    compute_config: dict[str, Any] = Field(default_factory=dict)
    memory_config: dict[str, Any] = Field(default_factory=dict)
    parallelization_config: dict[str, Any] = Field(default_factory=dict)
    gpu_config: dict[str, Any] = Field(default_factory=dict)
    
    # Data handling
    data_loading_config: dict[str, Any] = Field(default_factory=dict)
    batch_config: dict[str, Any] = Field(default_factory=dict)
    sampling_config: dict[str, Any] = Field(default_factory=dict)
    augmentation_config: dict[str, Any] = Field(default_factory=dict)
    
    # Model architecture (for neural networks)
    architecture_config: dict[str, Any] = Field(default_factory=dict)
    layer_configs: list[dict[str, Any]] = Field(default_factory=list)
    activation_functions: dict[str, str] = Field(default_factory=dict)
    initialization_config: dict[str, Any] = Field(default_factory=dict)
    
    # Evaluation and metrics
    evaluation_config: dict[str, Any] = Field(default_factory=dict)
    metric_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cross_validation_config: dict[str, Any] = Field(default_factory=dict)
    holdout_config: dict[str, Any] = Field(default_factory=dict)
    
    # Ensemble settings
    ensemble_config: dict[str, Any] = Field(default_factory=dict)
    voting_config: dict[str, Any] = Field(default_factory=dict)
    stacking_config: dict[str, Any] = Field(default_factory=dict)
    bagging_config: dict[str, Any] = Field(default_factory=dict)
    
    # Advanced settings
    interpretability_config: dict[str, Any] = Field(default_factory=dict)
    fairness_config: dict[str, Any] = Field(default_factory=dict)
    privacy_config: dict[str, Any] = Field(default_factory=dict)
    robustness_config: dict[str, Any] = Field(default_factory=dict)
    
    # Metadata and tracking
    created_by: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
    tags: list[str] = Field(default_factory=list)
    
    # Usage tracking
    usage_count: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0, le=1)
    average_performance: dict[str, float] = Field(default_factory=dict)
    best_performance: dict[str, float] = Field(default_factory=dict)
    
    # Validation and constraints
    parameter_constraints: dict[str, dict[str, Any]] = Field(default_factory=dict)
    compatibility_requirements: dict[str, Any] = Field(default_factory=dict)
    resource_requirements: dict[str, Any] = Field(default_factory=dict)
    
    # Comparison and optimization history
    parent_config_id: Optional[UUID] = None
    optimization_history: list[dict[str, Any]] = Field(default_factory=list)
    performance_history: list[dict[str, Any]] = Field(default_factory=list)
    
    # Export and sharing
    is_public: bool = Field(default=False)
    sharing_permissions: dict[str, Any] = Field(default_factory=dict)
    export_formats: list[str] = Field(default_factory=list)
    
    # Deployment settings
    deployment_config: dict[str, Any] = Field(default_factory=dict)
    serving_config: dict[str, Any] = Field(default_factory=dict)
    monitoring_config: dict[str, Any] = Field(default_factory=dict)
    update_strategy: str = Field(default="manual")
    
    @validator('algorithm_type')
    def validate_algorithm_type(cls, v: str) -> str:
        """Validate algorithm type."""
        valid_types = {
            'classification', 'regression', 'clustering', 'anomaly_detection',
            'dimensionality_reduction', 'reinforcement_learning', 'neural_network',
            'ensemble', 'time_series', 'nlp', 'computer_vision', 'recommendation'
        }
        
        if v.lower() not in valid_types:
            # Allow custom types but validate not empty
            if not v.strip():
                raise ValueError("Algorithm type cannot be empty")
        
        return v.lower()
    
    @validator('optimization_direction')
    def validate_optimization_direction(cls, v: str) -> str:
        """Validate optimization direction."""
        valid_directions = {'maximize', 'minimize'}
        
        if v.lower() not in valid_directions:
            raise ValueError(f"Optimization direction must be one of {valid_directions}")
        
        return v.lower()
    
    @validator('update_strategy')
    def validate_update_strategy(cls, v: str) -> str:
        """Validate update strategy."""
        valid_strategies = {'manual', 'automatic', 'scheduled', 'triggered'}
        
        if v.lower() not in valid_strategies:
            raise ValueError(f"Update strategy must be one of {valid_strategies}")
        
        return v.lower()
    
    @validator('updated_at')
    def validate_updated_at(cls, v: datetime, values: dict[str, Any]) -> datetime:
        """Ensure updated_at is not before created_at."""
        created_at = values.get('created_at')
        if created_at and v < created_at:
            raise ValueError("updated_at cannot be before created_at")
        return v
    
    def get_hyperparameter(self, parameter_name: str) -> Any:
        """Get a specific hyperparameter value."""
        return self.hyperparameters.get(parameter_name)
    
    def set_hyperparameter(self, parameter_name: str, value: Any) -> None:
        """Set a hyperparameter value."""
        self.hyperparameters[parameter_name] = value
        self.updated_at = datetime.utcnow()
    
    def update_hyperparameters(self, new_hyperparameters: dict[str, Any]) -> None:
        """Update multiple hyperparameters."""
        self.hyperparameters.update(new_hyperparameters)
        self.updated_at = datetime.utcnow()
    
    def validate_hyperparameters(self) -> list[str]:
        """Validate hyperparameters against constraints."""
        violations = []
        
        for param_name, value in self.hyperparameters.items():
            constraints = self.parameter_constraints.get(param_name, {})
            
            # Check type constraint
            if 'type' in constraints:
                expected_type = constraints['type']
                if not isinstance(value, eval(expected_type)):
                    violations.append(f"Parameter '{param_name}' should be {expected_type}, got {type(value)}")
            
            # Check range constraints
            if 'min' in constraints and value < constraints['min']:
                violations.append(f"Parameter '{param_name}' value {value} below minimum {constraints['min']}")
            
            if 'max' in constraints and value > constraints['max']:
                violations.append(f"Parameter '{param_name}' value {value} above maximum {constraints['max']}")
            
            # Check choices constraint
            if 'choices' in constraints and value not in constraints['choices']:
                violations.append(f"Parameter '{param_name}' value {value} not in allowed choices {constraints['choices']}")
        
        return violations
    
    def is_ensemble_algorithm(self) -> bool:
        """Check if this is an ensemble algorithm."""
        return self.algorithm_type == 'ensemble' or bool(self.ensemble_config)
    
    def is_neural_network(self) -> bool:
        """Check if this is a neural network algorithm."""
        return self.algorithm_type == 'neural_network' or bool(self.architecture_config)
    
    def requires_gpu(self) -> bool:
        """Check if algorithm requires GPU."""
        return bool(self.gpu_config) or self.compute_config.get('gpu_required', False)
    
    def get_memory_requirements(self) -> Optional[dict[str, Any]]:
        """Get memory requirements."""
        if not self.resource_requirements:
            return None
        
        return self.resource_requirements.get('memory')
    
    def get_compute_requirements(self) -> Optional[dict[str, Any]]:
        """Get compute requirements."""
        if not self.resource_requirements:
            return None
        
        return self.resource_requirements.get('compute')
    
    def add_performance_record(self, performance_metrics: dict[str, float], 
                             experiment_id: Optional[str] = None) -> None:
        """Add a performance record to history."""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': performance_metrics,
            'experiment_id': experiment_id
        }
        
        self.performance_history.append(record)
        
        # Update usage statistics
        self.usage_count += 1
        
        # Update average performance
        if self.average_performance:
            # Calculate running average
            for metric, value in performance_metrics.items():
                if metric in self.average_performance:
                    old_avg = self.average_performance[metric]
                    new_avg = (old_avg * (self.usage_count - 1) + value) / self.usage_count
                    self.average_performance[metric] = new_avg
                else:
                    self.average_performance[metric] = value
        else:
            self.average_performance = performance_metrics.copy()
        
        # Update best performance
        for metric, value in performance_metrics.items():
            current_best = self.best_performance.get(metric, float('-inf'))
            if value > current_best:
                self.best_performance[metric] = value
        
        self.updated_at = datetime.utcnow()
    
    def get_optimization_suggestion(self) -> dict[str, Any]:
        """Get optimization suggestions based on performance history."""
        if len(self.performance_history) < 2:
            return {"message": "Insufficient data for optimization suggestions"}
        
        suggestions = {
            "timestamp": datetime.utcnow().isoformat(),
            "based_on_runs": len(self.performance_history),
            "suggestions": []
        }
        
        # Analyze performance trends
        recent_performance = self.performance_history[-3:]  # Last 3 runs
        
        for metric in self.average_performance.keys():
            recent_values = [run['metrics'].get(metric, 0) for run in recent_performance]
            
            if len(recent_values) >= 2:
                trend = "improving" if recent_values[-1] > recent_values[0] else "declining"
                
                if trend == "declining":
                    suggestions["suggestions"].append({
                        "type": "performance_decline",
                        "metric": metric,
                        "recommendation": f"Consider adjusting hyperparameters as {metric} is declining"
                    })
        
        # Check if we're far from best performance
        if self.performance_history:
            latest_performance = self.performance_history[-1]['metrics']
            
            for metric, latest_value in latest_performance.items():
                best_value = self.best_performance.get(metric, latest_value)
                
                if best_value > 0 and (best_value - latest_value) / best_value > 0.1:  # 10% worse
                    suggestions["suggestions"].append({
                        "type": "suboptimal_performance",
                        "metric": metric,
                        "current": latest_value,
                        "best": best_value,
                        "recommendation": f"Current {metric} is {((best_value - latest_value) / best_value * 100):.1f}% below best"
                    })
        
        return suggestions
    
    def clone_configuration(self, new_name: str, modifications: Optional[dict[str, Any]] = None) -> AlgorithmConfiguration:
        """Create a clone of this configuration with optional modifications."""
        config_dict = self.to_dict()
        
        # Remove unique identifiers
        config_dict.pop('config_id', None)
        config_dict.pop('created_at', None)
        config_dict.pop('updated_at', None)
        config_dict.pop('usage_count', None)
        config_dict.pop('performance_history', None)
        
        # Set new name and parent reference
        config_dict['config_name'] = new_name
        config_dict['parent_config_id'] = self.config_id
        
        # Apply modifications if provided
        if modifications:
            config_dict.update(modifications)
        
        return AlgorithmConfiguration(**config_dict)
    
    def compare_with(self, other: AlgorithmConfiguration) -> dict[str, Any]:
        """Compare this configuration with another."""
        if not isinstance(other, AlgorithmConfiguration):
            raise ValueError("Can only compare with another AlgorithmConfiguration")
        
        comparison = {
            "config_1": {
                "id": str(self.config_id),
                "name": self.config_name,
                "algorithm": self.algorithm_name
            },
            "config_2": {
                "id": str(other.config_id),
                "name": other.config_name,
                "algorithm": other.algorithm_name
            },
            "comparison_timestamp": datetime.utcnow().isoformat()
        }
        
        # Compare basic attributes
        comparison["algorithm_match"] = self.algorithm_name == other.algorithm_name
        comparison["type_match"] = self.algorithm_type == other.algorithm_type
        
        # Compare hyperparameters
        common_params = set(self.hyperparameters.keys()) & set(other.hyperparameters.keys())
        different_params = []
        
        for param in common_params:
            if self.hyperparameters[param] != other.hyperparameters[param]:
                different_params.append({
                    "parameter": param,
                    "config_1_value": self.hyperparameters[param],
                    "config_2_value": other.hyperparameters[param]
                })
        
        comparison["hyperparameter_differences"] = different_params
        
        # Compare performance if available
        if self.average_performance and other.average_performance:
            performance_comparison = {}
            
            for metric in set(self.average_performance.keys()) & set(other.average_performance.keys()):
                diff = self.average_performance[metric] - other.average_performance[metric]
                performance_comparison[metric] = {
                    "config_1": self.average_performance[metric],
                    "config_2": other.average_performance[metric],
                    "difference": diff,
                    "better": "config_1" if diff > 0 else "config_2" if diff < 0 else "tie"
                }
            
            comparison["performance_comparison"] = performance_comparison
        
        return comparison
    
    def get_configuration_summary(self) -> dict[str, Any]:
        """Get comprehensive configuration summary."""
        summary = {
            "config_id": str(self.config_id),
            "config_name": self.config_name,
            "algorithm_name": self.algorithm_name,
            "algorithm_type": self.algorithm_type,
            "version": self.version,
            "created_by": self.created_by,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "is_public": self.is_public,
            "tags": self.tags,
        }
        
        # Add hyperparameter summary
        summary["hyperparameters"] = {
            "count": len(self.hyperparameters),
            "parameters": list(self.hyperparameters.keys())
        }
        
        # Add configuration complexity
        config_sections = [
            'preprocessing_config', 'feature_selection_config', 'validation_config',
            'optimization_config', 'training_config', 'ensemble_config'
        ]
        
        configured_sections = [
            section for section in config_sections 
            if getattr(self, section, {})
        ]
        
        summary["configuration_complexity"] = {
            "configured_sections": len(configured_sections),
            "sections": configured_sections,
            "is_complex": len(configured_sections) > 3
        }
        
        # Add performance summary
        if self.average_performance:
            summary["performance"] = {
                "average_metrics": self.average_performance,
                "best_metrics": self.best_performance,
                "performance_history_length": len(self.performance_history)
            }
        
        # Add resource requirements
        if self.resource_requirements:
            summary["resource_requirements"] = self.resource_requirements
        
        return summary
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary representation."""
        data = super().to_dict()
        
        # Convert UUIDs to strings for serialization
        data["config_id"] = str(self.config_id)
        if self.parent_config_id:
            data["parent_config_id"] = str(self.parent_config_id)
        
        # Convert datetime objects to ISO format
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        
        return data
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"AlgorithmConfiguration(id={self.config_id}, name='{self.config_name}', algorithm='{self.algorithm_name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"AlgorithmConfiguration(config_id={self.config_id}, "
            f"config_name='{self.config_name}', algorithm_name='{self.algorithm_name}', "
            f"algorithm_type='{self.algorithm_type}', version='{self.version}', "
            f"usage_count={self.usage_count})"
        )