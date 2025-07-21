"""Model Training Service for coordinating ML model training workflows."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

# TODO: Implement within data platform science domain - from packages.data_science.domain.entities.data_science_model import (
    DataScienceModel,
    ModelType,
    ModelStatus,
)
# TODO: Implement within data platform science domain - from packages.data_science.domain.entities.experiment import Experiment
# TODO: Implement within data platform science domain - from packages.data_science.domain.entities.dataset_profile import DatasetProfile
# TODO: Implement within data platform science domain - from packages.data_science.domain.value_objects.ml_model_metrics import MLModelMetrics


logger = logging.getLogger(__name__)


class ModelTrainingService:
    """Domain service for coordinating model training workflows.
    
    This service orchestrates the complete model training lifecycle including
    data validation, training execution, evaluation, and result tracking.
    """
    
    def __init__(self) -> None:
        """Initialize the model training service."""
        self._logger = logger
    
    def validate_training_prerequisites(
        self, 
        model: DataScienceModel,
        dataset_profile: DatasetProfile,
        experiment: Optional[Experiment] = None
    ) -> dict[str, Any]:
        """Validate prerequisites for model training.
        
        Args:
            model: Model to be trained
            dataset_profile: Profile of the training dataset
            experiment: Optional experiment context
            
        Returns:
            Validation results with readiness status and issues
        """
        validation_result = {
            "is_ready": True,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Validate model state
            if not model.is_trainable():
                validation_result["issues"].append(
                    f"Model status '{model.status}' is not trainable. Must be DRAFT or FAILED."
                )
                validation_result["is_ready"] = False
            
            # Validate feature list
            if not model.features:
                validation_result["issues"].append("Model must have features defined for training")
                validation_result["is_ready"] = False
            
            # Validate dataset quality
            if dataset_profile.quality_score < 0.7:
                validation_result["warnings"].append(
                    f"Dataset quality score is {dataset_profile.quality_score:.2f}, "
                    "which may impact model performance"
                )
                validation_result["recommendations"].append(
                    "Consider data cleaning and quality improvement before training"
                )
            
            # Check dataset size adequacy
            min_samples = self._get_minimum_samples_for_model_type(model.model_type)
            if dataset_profile.row_count < min_samples:
                validation_result["issues"].append(
                    f"Dataset has {dataset_profile.row_count} samples, "
                    f"but minimum {min_samples} required for {model.model_type}"
                )
                validation_result["is_ready"] = False
            
            # Validate feature availability in dataset
            dataset_columns = set(dataset_profile.column_names)
            missing_features = set(model.features) - dataset_columns
            if missing_features:
                validation_result["issues"].append(
                    f"Features not found in dataset: {missing_features}"
                )
                validation_result["is_ready"] = False
            
            # Check target variable for supervised learning
            if model.model_type in [ModelType.CLASSIFICATION, ModelType.REGRESSION]:
                if not model.target_variable:
                    validation_result["issues"].append(
                        "Supervised learning models must have target variable defined"
                    )
                    validation_result["is_ready"] = False
                elif model.target_variable not in dataset_columns:
                    validation_result["issues"].append(
                        f"Target variable '{model.target_variable}' not found in dataset"
                    )
                    validation_result["is_ready"] = False
            
            # Validate hyperparameters
            hyperparameter_validation = self._validate_hyperparameters(
                model.model_type, model.algorithm, model.hyperparameters
            )
            if not hyperparameter_validation["is_valid"]:
                validation_result["issues"].extend(hyperparameter_validation["errors"])
                validation_result["warnings"].extend(hyperparameter_validation["warnings"])
                if hyperparameter_validation["errors"]:
                    validation_result["is_ready"] = False
            
            # Experiment validation if provided
            if experiment:
                if experiment.status not in ['created', 'queued']:
                    validation_result["warnings"].append(
                        f"Experiment status is '{experiment.status}', training may conflict"
                    )
                
                if experiment.dataset_id != dataset_profile.dataset_id:
                    validation_result["issues"].append(
                        "Experiment dataset ID does not match provided dataset profile"
                    )
                    validation_result["is_ready"] = False
            
            self._logger.info(f"Training prerequisites validation completed for model {model.id}")
            
        except Exception as e:
            validation_result["issues"].append(f"Validation error: {str(e)}")
            validation_result["is_ready"] = False
            self._logger.error(f"Training prerequisites validation failed: {e}")
        
        return validation_result
    
    def prepare_training_configuration(
        self,
        model: DataScienceModel,
        dataset_profile: DatasetProfile,
        training_params: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Prepare comprehensive training configuration.
        
        Args:
            model: Model to be trained
            dataset_profile: Training dataset profile
            training_params: Additional training parameters
            
        Returns:
            Complete training configuration
        """
        config = {
            "model_config": {
                "model_id": str(model.id),
                "model_name": model.name,
                "model_type": model.model_type.value,
                "algorithm": model.algorithm,
                "features": model.features.copy(),
                "target_variable": model.target_variable,
                "hyperparameters": model.hyperparameters.copy()
            },
            "dataset_config": {
                "dataset_id": dataset_profile.dataset_id,
                "dataset_name": dataset_profile.name,
                "row_count": dataset_profile.row_count,
                "column_count": dataset_profile.column_count,
                "quality_score": dataset_profile.quality_score,
                "schema_hash": dataset_profile.schema_hash
            },
            "training_config": {
                "timestamp": datetime.utcnow().isoformat(),
                "random_seed": 42,  # Default seed for reproducibility
                "validation_split": 0.2,
                "test_split": 0.1,
                "cross_validation_folds": 5,
                "early_stopping": True,
                "metric_monitoring": True
            },
            "environment_config": {
                "python_version": "3.11+",
                "required_packages": self._get_required_packages(model.algorithm),
                "compute_requirements": self._estimate_compute_requirements(
                    model.model_type, dataset_profile.row_count, len(model.features)
                )
            }
        }
        
        # Merge additional training parameters
        if training_params:
            config["training_config"].update(training_params)
        
        # Add algorithm-specific configurations
        algorithm_config = self._get_algorithm_specific_config(
            model.algorithm, model.hyperparameters
        )
        config["algorithm_config"] = algorithm_config
        
        return config
    
    def estimate_training_duration(
        self,
        model: DataScienceModel,
        dataset_profile: DatasetProfile
    ) -> dict[str, Any]:
        """Estimate model training duration and resource requirements.
        
        Args:
            model: Model to be trained
            dataset_profile: Training dataset profile
            
        Returns:
            Duration and resource estimates
        """
        # Base estimation factors
        complexity_factors = {
            ModelType.STATISTICAL: 1.0,
            ModelType.MACHINE_LEARNING: 2.0,
            ModelType.DEEP_LEARNING: 10.0,
            ModelType.ENSEMBLE: 5.0,
            ModelType.TIME_SERIES: 3.0,
            ModelType.ANOMALY_DETECTION: 2.5,
            ModelType.REGRESSION: 1.5,
            ModelType.CLASSIFICATION: 2.0,
            ModelType.CLUSTERING: 1.8,
            ModelType.REINFORCEMENT_LEARNING: 20.0
        }
        
        # Algorithm-specific factors
        algorithm_factors = {
            "linear_regression": 0.5,
            "logistic_regression": 0.6,
            "random_forest": 2.0,
            "gradient_boosting": 3.0,
            "xgboost": 2.5,
            "lightgbm": 2.0,
            "svm": 4.0,
            "neural_network": 8.0,
            "deep_neural_network": 15.0,
            "cnn": 20.0,
            "rnn": 25.0,
            "lstm": 30.0,
            "transformer": 40.0,
            "isolation_forest": 1.5,
            "one_class_svm": 3.0,
            "autoencoder": 10.0
        }
        
        # Base time calculation (in minutes)
        data_factor = (dataset_profile.row_count / 10000) * (len(model.features) / 10)
        model_complexity = complexity_factors.get(model.model_type, 2.0)
        algorithm_complexity = algorithm_factors.get(model.algorithm.lower(), 2.0)
        
        base_time_minutes = max(1.0, data_factor * model_complexity * algorithm_complexity)
        
        # Apply cross-validation multiplier
        cv_folds = model.hyperparameters.get("cv_folds", 5)
        cv_time_minutes = base_time_minutes * cv_folds
        
        # Apply hyperparameter tuning multiplier if applicable
        tuning_multiplier = 1.0
        if "n_trials" in model.hyperparameters:
            tuning_multiplier = min(model.hyperparameters["n_trials"], 100)
        
        total_time_minutes = cv_time_minutes * tuning_multiplier
        
        return {
            "estimated_duration_minutes": round(total_time_minutes, 2),
            "estimated_duration_hours": round(total_time_minutes / 60, 2),
            "base_training_minutes": round(base_time_minutes, 2),
            "cross_validation_minutes": round(cv_time_minutes, 2),
            "hyperparameter_tuning_trials": tuning_multiplier,
            "confidence_level": "medium",  # Could be enhanced with historical data
            "factors": {
                "data_size_factor": round(data_factor, 2),
                "model_complexity_factor": model_complexity,
                "algorithm_complexity_factor": algorithm_complexity,
                "cv_folds": cv_folds
            },
            "resource_estimates": {
                "memory_gb": max(1, round(dataset_profile.row_count * len(model.features) / 1000000, 1)),
                "cpu_cores": min(8, max(1, int(total_time_minutes / 60))),
                "gpu_required": model.model_type in [ModelType.DEEP_LEARNING],
                "storage_gb": max(0.1, round(dataset_profile.row_count / 100000, 1))
            }
        }
    
    def validate_training_results(
        self,
        model: DataScienceModel,
        training_metrics: MLModelMetrics,
        validation_metrics: MLModelMetrics
    ) -> dict[str, Any]:
        """Validate training results and detect potential issues.
        
        Args:
            model: Trained model
            training_metrics: Metrics from training data
            validation_metrics: Metrics from validation data
            
        Returns:
            Validation results and recommendations
        """
        validation_result = {
            "is_valid": True,
            "quality_score": 0.0,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "performance_analysis": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Check for overfitting
            overfitting_analysis = self._detect_overfitting(training_metrics, validation_metrics)
            if overfitting_analysis["is_overfitting"]:
                validation_result["warnings"].append(
                    f"Potential overfitting detected: {overfitting_analysis['description']}"
                )
                validation_result["recommendations"].extend([
                    "Consider regularization techniques",
                    "Reduce model complexity",
                    "Increase training data",
                    "Use early stopping"
                ])
            
            # Check for underfitting
            underfitting_analysis = self._detect_underfitting(training_metrics, validation_metrics)
            if underfitting_analysis["is_underfitting"]:
                validation_result["warnings"].append(
                    f"Potential underfitting detected: {underfitting_analysis['description']}"
                )
                validation_result["recommendations"].extend([
                    "Increase model complexity",
                    "Add more features",
                    "Reduce regularization",
                    "Train for more epochs"
                ])
            
            # Calculate overall quality score
            quality_score = self._calculate_model_quality_score(
                training_metrics, validation_metrics, overfitting_analysis, underfitting_analysis
            )
            validation_result["quality_score"] = quality_score
            
            # Performance thresholds
            if quality_score < 0.5:
                validation_result["issues"].append(
                    f"Model quality score {quality_score:.2f} is below acceptable threshold (0.5)"
                )
                validation_result["is_valid"] = False
            elif quality_score < 0.7:
                validation_result["warnings"].append(
                    f"Model quality score {quality_score:.2f} is below recommended threshold (0.7)"
                )
            
            # Check metric consistency
            consistency_check = self._check_metric_consistency(training_metrics, validation_metrics)
            if not consistency_check["is_consistent"]:
                validation_result["warnings"].extend(consistency_check["issues"])
            
            # Performance analysis
            validation_result["performance_analysis"] = {
                "training_performance": training_metrics.get_primary_metric(),
                "validation_performance": validation_metrics.get_primary_metric(),
                "performance_gap": abs(
                    training_metrics.get_primary_metric() - validation_metrics.get_primary_metric()
                ),
                "overfitting_score": overfitting_analysis["score"],
                "underfitting_score": underfitting_analysis["score"],
                "stability_score": consistency_check["stability_score"]
            }
            
            self._logger.info(f"Training results validation completed for model {model.id}")
            
        except Exception as e:
            validation_result["issues"].append(f"Validation error: {str(e)}")
            validation_result["is_valid"] = False
            self._logger.error(f"Training results validation failed: {e}")
        
        return validation_result
    
    def _get_minimum_samples_for_model_type(self, model_type: ModelType) -> int:
        """Get minimum sample requirements for different model types."""
        minimum_samples = {
            ModelType.STATISTICAL: 30,
            ModelType.MACHINE_LEARNING: 100,
            ModelType.DEEP_LEARNING: 1000,
            ModelType.ENSEMBLE: 500,
            ModelType.TIME_SERIES: 50,
            ModelType.ANOMALY_DETECTION: 100,
            ModelType.REGRESSION: 50,
            ModelType.CLASSIFICATION: 100,
            ModelType.CLUSTERING: 50,
            ModelType.REINFORCEMENT_LEARNING: 1000
        }
        return minimum_samples.get(model_type, 100)
    
    def _validate_hyperparameters(
        self, model_type: ModelType, algorithm: str, hyperparameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate hyperparameters for the given model and algorithm."""
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Common validations
        if "random_state" in hyperparameters:
            if not isinstance(hyperparameters["random_state"], int) or hyperparameters["random_state"] < 0:
                result["errors"].append("random_state must be a non-negative integer")
                result["is_valid"] = False
        
        # Algorithm-specific validations
        if "random_forest" in algorithm.lower():
            if "n_estimators" in hyperparameters:
                if hyperparameters["n_estimators"] < 10:
                    result["warnings"].append("n_estimators < 10 may lead to underfitting")
                elif hyperparameters["n_estimators"] > 1000:
                    result["warnings"].append("n_estimators > 1000 may be computationally expensive")
        
        return result
    
    def _get_required_packages(self, algorithm: str) -> list[str]:
        """Get required packages for the algorithm."""
        package_map = {
            "sklearn": ["scikit-learn", "numpy", "scipy"],
            "xgboost": ["xgboost", "numpy", "scipy"],
            "lightgbm": ["lightgbm", "numpy", "scipy"],
            "tensorflow": ["tensorflow", "numpy"],
            "pytorch": ["torch", "numpy"],
            "statsmodels": ["statsmodels", "numpy", "scipy"]
        }
        
        # Default packages
        base_packages = ["numpy", "pandas", "scikit-learn"]
        
        for key, packages in package_map.items():
            if key.lower() in algorithm.lower():
                return list(set(base_packages + packages))
        
        return base_packages
    
    def _estimate_compute_requirements(
        self, model_type: ModelType, data_size: int, feature_count: int
    ) -> dict[str, Any]:
        """Estimate compute requirements."""
        base_memory = max(1, (data_size * feature_count) / 1000000)  # GB
        
        if model_type in [ModelType.DEEP_LEARNING]:
            return {
                "memory_gb": base_memory * 4,
                "cpu_cores": 4,
                "gpu_required": True,
                "gpu_memory_gb": 8
            }
        elif model_type in [ModelType.ENSEMBLE]:
            return {
                "memory_gb": base_memory * 2,
                "cpu_cores": 8,
                "gpu_required": False
            }
        else:
            return {
                "memory_gb": base_memory,
                "cpu_cores": 2,
                "gpu_required": False
            }
    
    def _get_algorithm_specific_config(
        self, algorithm: str, hyperparameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Get algorithm-specific configuration."""
        config = {
            "algorithm": algorithm,
            "optimization_strategy": "default",
            "early_stopping_enabled": True,
            "checkpointing_enabled": True
        }
        
        if "neural" in algorithm.lower() or "deep" in algorithm.lower():
            config.update({
                "optimization_strategy": "adaptive_learning_rate",
                "batch_size": hyperparameters.get("batch_size", 32),
                "epochs": hyperparameters.get("epochs", 100),
                "learning_rate": hyperparameters.get("learning_rate", 0.001)
            })
        
        return config
    
    def _detect_overfitting(
        self, training_metrics: MLModelMetrics, validation_metrics: MLModelMetrics
    ) -> dict[str, Any]:
        """Detect overfitting in training results."""
        train_score = training_metrics.get_primary_metric()
        val_score = validation_metrics.get_primary_metric()
        
        if train_score is None or val_score is None:
            return {"is_overfitting": False, "score": 0.0, "description": "Insufficient metrics"}
        
        # Calculate performance gap
        gap = abs(train_score - val_score)
        relative_gap = gap / max(train_score, 0.001)
        
        is_overfitting = relative_gap > 0.1  # 10% threshold
        
        return {
            "is_overfitting": is_overfitting,
            "score": relative_gap,
            "description": f"Training score: {train_score:.3f}, Validation score: {val_score:.3f}, Gap: {gap:.3f}"
        }
    
    def _detect_underfitting(
        self, training_metrics: MLModelMetrics, validation_metrics: MLModelMetrics
    ) -> dict[str, Any]:
        """Detect underfitting in training results."""
        train_score = training_metrics.get_primary_metric()
        val_score = validation_metrics.get_primary_metric()
        
        if train_score is None or val_score is None:
            return {"is_underfitting": False, "score": 0.0, "description": "Insufficient metrics"}
        
        # Both scores are low
        is_underfitting = train_score < 0.6 and val_score < 0.6
        
        avg_score = (train_score + val_score) / 2
        
        return {
            "is_underfitting": is_underfitting,
            "score": 1.0 - avg_score,
            "description": f"Average performance {avg_score:.3f} is below expected threshold"
        }
    
    def _calculate_model_quality_score(
        self, 
        training_metrics: MLModelMetrics, 
        validation_metrics: MLModelMetrics,
        overfitting_analysis: dict[str, Any],
        underfitting_analysis: dict[str, Any]
    ) -> float:
        """Calculate overall model quality score."""
        train_score = training_metrics.get_primary_metric() or 0.0
        val_score = validation_metrics.get_primary_metric() or 0.0
        
        # Base score from validation performance
        base_score = val_score
        
        # Penalty for overfitting
        overfitting_penalty = overfitting_analysis["score"] * 0.3
        
        # Penalty for underfitting
        underfitting_penalty = underfitting_analysis["score"] * 0.2
        
        # Final quality score
        quality_score = max(0.0, base_score - overfitting_penalty - underfitting_penalty)
        
        return min(1.0, quality_score)
    
    def _check_metric_consistency(
        self, training_metrics: MLModelMetrics, validation_metrics: MLModelMetrics
    ) -> dict[str, Any]:
        """Check consistency between training and validation metrics."""
        result = {
            "is_consistent": True,
            "issues": [],
            "stability_score": 1.0
        }
        
        # Check if metrics are reasonable
        train_score = training_metrics.get_primary_metric()
        val_score = validation_metrics.get_primary_metric()
        
        if train_score is not None and val_score is not None:
            if train_score < 0 or train_score > 1:
                result["issues"].append(f"Training metric {train_score} is outside expected range [0,1]")
                result["is_consistent"] = False
            
            if val_score < 0 or val_score > 1:
                result["issues"].append(f"Validation metric {val_score} is outside expected range [0,1]")
                result["is_consistent"] = False
            
            # Calculate stability score based on consistency
            if train_score > 0 and val_score > 0:
                stability = 1.0 - abs(train_score - val_score) / max(train_score, val_score)
                result["stability_score"] = max(0.0, stability)
        
        return result