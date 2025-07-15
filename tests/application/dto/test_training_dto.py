"""
Comprehensive tests for training DTOs.

This module tests all training-related Data Transfer Objects to ensure proper validation,
serialization, and behavior across all use cases including training configuration,
requests, results, and status tracking.
"""

import json
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from pynomaly.application.dto.training_dto import (
    AutoMLConfigDTO,
    ModelMetricsDTO,
    NotificationConfigDTO,
    ResourceConstraintsDTO,
    TrainingConfigDTO,
    TrainingPriorityLevel,
    TrainingRequestDTO,
    TrainingResultDTO,
    TrainingStatusDTO,
    TrainingTrigger,
    ValidationConfigDTO,
)


class TestResourceConstraintsDTO:
    """Test suite for ResourceConstraintsDTO."""

    def test_default_creation(self):
        """Test creation with default values."""
        constraints = ResourceConstraintsDTO()

        assert constraints.max_training_time_seconds is None
        assert constraints.max_time_per_trial is None
        assert constraints.max_memory_mb is None
        assert constraints.max_cpu_cores is None
        assert constraints.max_gpu_memory_mb is None
        assert constraints.enable_gpu is False
        assert constraints.n_jobs == 1
        assert constraints.max_concurrent_trials is None

    def test_complete_creation(self):
        """Test creation with all parameters."""
        constraints = ResourceConstraintsDTO(
            max_training_time_seconds=3600,
            max_time_per_trial=300,
            max_memory_mb=8192,
            max_cpu_cores=8,
            max_gpu_memory_mb=4096,
            enable_gpu=True,
            n_jobs=4,
            max_concurrent_trials=2,
        )

        assert constraints.max_training_time_seconds == 3600
        assert constraints.max_time_per_trial == 300
        assert constraints.max_memory_mb == 8192
        assert constraints.max_cpu_cores == 8
        assert constraints.max_gpu_memory_mb == 4096
        assert constraints.enable_gpu is True
        assert constraints.n_jobs == 4
        assert constraints.max_concurrent_trials == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        constraints = ResourceConstraintsDTO(
            max_training_time_seconds=1800,
            max_memory_mb=4096,
            enable_gpu=True,
            n_jobs=2,
        )

        result = constraints.to_dict()
        expected = {
            "max_training_time_seconds": 1800,
            "max_time_per_trial": None,
            "max_memory_mb": 4096,
            "max_cpu_cores": None,
            "max_gpu_memory_mb": None,
            "enable_gpu": True,
            "n_jobs": 2,
            "max_concurrent_trials": None,
        }

        assert result == expected

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "max_training_time_seconds": 7200,
            "max_memory_mb": 16384,
            "enable_gpu": False,
            "n_jobs": 8,
        }

        constraints = ResourceConstraintsDTO.from_dict(data)

        assert constraints.max_training_time_seconds == 7200
        assert constraints.max_memory_mb == 16384
        assert constraints.enable_gpu is False
        assert constraints.n_jobs == 8

    def test_roundtrip_serialization(self):
        """Test dictionary serialization roundtrip."""
        original = ResourceConstraintsDTO(
            max_training_time_seconds=900,
            max_cpu_cores=4,
            enable_gpu=True,
        )

        data = original.to_dict()
        reconstructed = ResourceConstraintsDTO.from_dict(data)

        assert (
            reconstructed.max_training_time_seconds
            == original.max_training_time_seconds
        )
        assert reconstructed.max_cpu_cores == original.max_cpu_cores
        assert reconstructed.enable_gpu == original.enable_gpu


class TestAutoMLConfigDTO:
    """Test suite for AutoMLConfigDTO."""

    def test_default_creation(self):
        """Test creation with default values."""
        config = AutoMLConfigDTO()

        assert config.enable_automl is True
        assert config.optimization_objective == "auc"
        assert config.max_algorithms == 3
        assert config.enable_ensemble is True
        assert config.ensemble_method == "voting"
        assert config.algorithm_whitelist is None
        assert config.algorithm_blacklist is None
        assert config.model_selection_strategy == "best_single_metric"
        assert config.metric_weights is None

    def test_complete_creation(self):
        """Test creation with all parameters."""
        whitelist = ["isolation_forest", "one_class_svm"]
        blacklist = ["lof", "knn"]
        metric_weights = {"auc": 0.6, "precision": 0.25, "recall": 0.15}

        config = AutoMLConfigDTO(
            enable_automl=False,
            optimization_objective="f1_score",
            max_algorithms=5,
            enable_ensemble=False,
            ensemble_method="stacking",
            algorithm_whitelist=whitelist,
            algorithm_blacklist=blacklist,
            model_selection_strategy="weighted_score",
            metric_weights=metric_weights,
        )

        assert config.enable_automl is False
        assert config.optimization_objective == "f1_score"
        assert config.max_algorithms == 5
        assert config.enable_ensemble is False
        assert config.ensemble_method == "stacking"
        assert config.algorithm_whitelist == whitelist
        assert config.algorithm_blacklist == blacklist
        assert config.model_selection_strategy == "weighted_score"
        assert config.metric_weights == metric_weights

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = AutoMLConfigDTO(
            optimization_objective="precision",
            max_algorithms=2,
            algorithm_whitelist=["isolation_forest"],
        )

        result = config.to_dict()

        assert result["optimization_objective"] == "precision"
        assert result["max_algorithms"] == 2
        assert result["algorithm_whitelist"] == ["isolation_forest"]

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "enable_automl": True,
            "optimization_objective": "recall",
            "max_algorithms": 4,
            "enable_ensemble": True,
            "ensemble_method": "blending",
        }

        config = AutoMLConfigDTO.from_dict(data)

        assert config.enable_automl is True
        assert config.optimization_objective == "recall"
        assert config.max_algorithms == 4
        assert config.ensemble_method == "blending"


class TestValidationConfigDTO:
    """Test suite for ValidationConfigDTO."""

    def test_default_creation(self):
        """Test creation with default values."""
        config = ValidationConfigDTO()

        assert config.validation_strategy == "holdout"
        assert config.validation_split == 0.2
        assert config.cv_folds == 5
        assert config.cv_strategy == "stratified"
        assert config.enable_early_stopping is True
        assert config.early_stopping_patience == 10
        assert config.early_stopping_metric == "validation_score"
        assert config.early_stopping_threshold == 0.001
        assert config.enable_data_validation is True
        assert config.min_samples_required == 100
        assert config.max_missing_ratio == 0.1

    def test_complete_creation(self):
        """Test creation with custom values."""
        config = ValidationConfigDTO(
            validation_strategy="cross_validation",
            validation_split=0.3,
            cv_folds=10,
            cv_strategy="kfold",
            enable_early_stopping=False,
            early_stopping_patience=20,
            early_stopping_metric="loss",
            early_stopping_threshold=0.0001,
            enable_data_validation=False,
            min_samples_required=500,
            max_missing_ratio=0.05,
        )

        assert config.validation_strategy == "cross_validation"
        assert config.validation_split == 0.3
        assert config.cv_folds == 10
        assert config.cv_strategy == "kfold"
        assert config.enable_early_stopping is False
        assert config.early_stopping_patience == 20
        assert config.early_stopping_metric == "loss"
        assert config.early_stopping_threshold == 0.0001
        assert config.enable_data_validation is False
        assert config.min_samples_required == 500
        assert config.max_missing_ratio == 0.05

    def test_validation_strategies(self):
        """Test different validation strategies."""
        strategies = ["holdout", "cross_validation", "time_series_split"]

        for strategy in strategies:
            config = ValidationConfigDTO(validation_strategy=strategy)
            assert config.validation_strategy == strategy

    def test_cv_strategies(self):
        """Test different cross-validation strategies."""
        cv_strategies = ["stratified", "kfold", "group", "time_series"]

        for cv_strategy in cv_strategies:
            config = ValidationConfigDTO(cv_strategy=cv_strategy)
            assert config.cv_strategy == cv_strategy

    def test_to_dict_and_from_dict(self):
        """Test dictionary serialization roundtrip."""
        original = ValidationConfigDTO(
            validation_strategy="time_series_split",
            cv_folds=3,
            early_stopping_patience=5,
        )

        data = original.to_dict()
        reconstructed = ValidationConfigDTO.from_dict(data)

        assert reconstructed.validation_strategy == original.validation_strategy
        assert reconstructed.cv_folds == original.cv_folds
        assert reconstructed.early_stopping_patience == original.early_stopping_patience


class TestNotificationConfigDTO:
    """Test suite for NotificationConfigDTO."""

    def test_default_creation(self):
        """Test creation with default values."""
        config = NotificationConfigDTO()

        assert config.enable_notifications is True
        assert config.notification_channels == ["websocket"]
        assert config.email_recipients == []
        assert config.email_on_completion is True
        assert config.email_on_failure is True
        assert config.webhook_urls == []
        assert config.webhook_events == ["completion", "failure"]
        assert config.slack_webhook_url is None
        assert config.slack_channel is None

    def test_complete_creation(self):
        """Test creation with all parameters."""
        email_recipients = ["admin@example.com", "ml-team@example.com"]
        webhook_urls = [
            "https://hooks.example.com/training",
            "https://api.example.com/notify",
        ]
        webhook_events = ["start", "completion", "failure", "progress"]

        config = NotificationConfigDTO(
            enable_notifications=False,
            notification_channels=["email", "slack", "webhook"],
            email_recipients=email_recipients,
            email_on_completion=False,
            email_on_failure=True,
            webhook_urls=webhook_urls,
            webhook_events=webhook_events,
            slack_webhook_url="https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
            slack_channel="#ml-alerts",
        )

        assert config.enable_notifications is False
        assert config.notification_channels == ["email", "slack", "webhook"]
        assert config.email_recipients == email_recipients
        assert config.email_on_completion is False
        assert config.email_on_failure is True
        assert config.webhook_urls == webhook_urls
        assert config.webhook_events == webhook_events
        assert config.slack_webhook_url.startswith("https://hooks.slack.com")
        assert config.slack_channel == "#ml-alerts"

    def test_notification_channels(self):
        """Test different notification channel combinations."""
        channel_combinations = [
            ["websocket"],
            ["email"],
            ["slack"],
            ["webhook"],
            ["email", "slack"],
            ["websocket", "email", "slack", "webhook"],
        ]

        for channels in channel_combinations:
            config = NotificationConfigDTO(notification_channels=channels)
            assert config.notification_channels == channels

    def test_to_dict_and_from_dict(self):
        """Test dictionary serialization roundtrip."""
        original = NotificationConfigDTO(
            notification_channels=["email", "webhook"],
            email_recipients=["test@example.com"],
            webhook_urls=["https://example.com/hook"],
        )

        data = original.to_dict()
        reconstructed = NotificationConfigDTO.from_dict(data)

        assert reconstructed.notification_channels == original.notification_channels
        assert reconstructed.email_recipients == original.email_recipients
        assert reconstructed.webhook_urls == original.webhook_urls


class TestTrainingConfigDTO:
    """Test suite for TrainingConfigDTO."""

    def test_default_creation(self):
        """Test creation with default values."""
        config = TrainingConfigDTO()

        assert config.experiment_name is None
        assert config.description is None
        assert config.tags == []
        assert config.dataset_id is None
        assert config.detector_id is None
        assert config.target_algorithms is None
        assert isinstance(config.automl_config, AutoMLConfigDTO)
        assert isinstance(config.validation_config, ValidationConfigDTO)
        assert config.optimization_config is None
        assert isinstance(config.resource_constraints, ResourceConstraintsDTO)
        assert isinstance(config.notification_config, NotificationConfigDTO)
        assert config.schedule_cron is None
        assert config.schedule_enabled is False
        assert config.performance_monitoring_enabled is True
        assert config.retrain_threshold == 0.05
        assert config.performance_window_days == 7
        assert config.auto_deploy_best_model is False
        assert config.model_versioning_enabled is True
        assert config.keep_model_versions == 5
        assert config.enable_model_explainability is True
        assert config.enable_drift_detection is True
        assert config.enable_feature_selection is True
        assert config.created_by is None
        assert config.created_at is None
        assert config.priority == TrainingPriorityLevel.NORMAL

    def test_complete_creation(self):
        """Test creation with all parameters."""
        detector_id = uuid4()
        created_at = datetime.utcnow()
        tags = ["anomaly_detection", "production", "v2"]
        target_algorithms = ["isolation_forest", "one_class_svm"]

        # Create component configurations
        automl_config = AutoMLConfigDTO(max_algorithms=5)
        validation_config = ValidationConfigDTO(cv_folds=10)
        resource_constraints = ResourceConstraintsDTO(max_memory_mb=8192)
        notification_config = NotificationConfigDTO(enable_notifications=False)

        config = TrainingConfigDTO(
            experiment_name="fraud_detection_v2",
            description="Enhanced fraud detection model with new features",
            tags=tags,
            dataset_id="dataset_123",
            detector_id=detector_id,
            target_algorithms=target_algorithms,
            automl_config=automl_config,
            validation_config=validation_config,
            resource_constraints=resource_constraints,
            notification_config=notification_config,
            schedule_cron="0 2 * * 0",  # Weekly at 2 AM on Sunday
            schedule_enabled=True,
            performance_monitoring_enabled=False,
            retrain_threshold=0.1,
            performance_window_days=14,
            auto_deploy_best_model=True,
            model_versioning_enabled=False,
            keep_model_versions=10,
            enable_model_explainability=False,
            enable_drift_detection=False,
            enable_feature_selection=False,
            created_by="data_scientist@example.com",
            created_at=created_at,
            priority=TrainingPriorityLevel.HIGH,
        )

        assert config.experiment_name == "fraud_detection_v2"
        assert config.description == "Enhanced fraud detection model with new features"
        assert config.tags == tags
        assert config.dataset_id == "dataset_123"
        assert config.detector_id == detector_id
        assert config.target_algorithms == target_algorithms
        assert config.automl_config.max_algorithms == 5
        assert config.validation_config.cv_folds == 10
        assert config.resource_constraints.max_memory_mb == 8192
        assert config.notification_config.enable_notifications is False
        assert config.schedule_cron == "0 2 * * 0"
        assert config.schedule_enabled is True
        assert config.performance_monitoring_enabled is False
        assert config.retrain_threshold == 0.1
        assert config.performance_window_days == 14
        assert config.auto_deploy_best_model is True
        assert config.model_versioning_enabled is False
        assert config.keep_model_versions == 10
        assert config.enable_model_explainability is False
        assert config.enable_drift_detection is False
        assert config.enable_feature_selection is False
        assert config.created_by == "data_scientist@example.com"
        assert config.created_at == created_at
        assert config.priority == TrainingPriorityLevel.HIGH

    def test_priority_levels(self):
        """Test different priority levels."""
        priorities = [
            TrainingPriorityLevel.LOW,
            TrainingPriorityLevel.NORMAL,
            TrainingPriorityLevel.HIGH,
            TrainingPriorityLevel.URGENT,
        ]

        for priority in priorities:
            config = TrainingConfigDTO(priority=priority)
            assert config.priority == priority

    def test_to_dict(self):
        """Test conversion to dictionary."""
        detector_id = uuid4()
        created_at = datetime.utcnow()

        config = TrainingConfigDTO(
            experiment_name="test_experiment",
            detector_id=detector_id,
            created_at=created_at,
            priority=TrainingPriorityLevel.HIGH,
        )

        result = config.to_dict()

        assert result["experiment_name"] == "test_experiment"
        assert result["detector_id"] == str(detector_id)
        assert result["created_at"] == created_at.isoformat()
        assert result["priority"] == "high"
        assert "automl_config" in result
        assert "validation_config" in result

    def test_from_dict(self):
        """Test creation from dictionary."""
        detector_id = uuid4()
        created_at = datetime.utcnow()

        data = {
            "experiment_name": "from_dict_test",
            "detector_id": str(detector_id),
            "created_at": created_at.isoformat(),
            "priority": "urgent",
            "automl_config": {"max_algorithms": 2},
            "validation_config": {"cv_folds": 3},
            "resource_constraints": {"n_jobs": 4},
            "notification_config": {"enable_notifications": False},
        }

        config = TrainingConfigDTO.from_dict(data)

        assert config.experiment_name == "from_dict_test"
        assert config.detector_id == detector_id
        assert config.created_at == created_at
        assert config.priority == TrainingPriorityLevel.URGENT
        assert config.automl_config.max_algorithms == 2
        assert config.validation_config.cv_folds == 3
        assert config.resource_constraints.n_jobs == 4
        assert config.notification_config.enable_notifications is False

    def test_roundtrip_serialization(self):
        """Test complete serialization roundtrip."""
        detector_id = uuid4()
        original = TrainingConfigDTO(
            experiment_name="roundtrip_test",
            detector_id=detector_id,
            tags=["test", "roundtrip"],
            priority=TrainingPriorityLevel.HIGH,
        )

        # Serialize to dict
        data = original.to_dict()

        # Deserialize from dict
        reconstructed = TrainingConfigDTO.from_dict(data)

        assert reconstructed.experiment_name == original.experiment_name
        assert reconstructed.detector_id == original.detector_id
        assert reconstructed.tags == original.tags
        assert reconstructed.priority == original.priority


class TestTrainingRequestDTO:
    """Test suite for TrainingRequestDTO."""

    def test_basic_creation(self):
        """Test basic training request creation."""
        detector_id = uuid4()
        dataset_id = "dataset_456"

        request = TrainingRequestDTO(detector_id=detector_id, dataset_id=dataset_id)

        assert request.detector_id == detector_id
        assert request.dataset_id == dataset_id
        assert request.config is None
        assert request.trigger_type == TrainingTrigger.MANUAL
        assert request.trigger_metadata == {}
        assert request.requested_by is None
        assert isinstance(request.requested_at, datetime)

    def test_complete_creation(self):
        """Test creation with all parameters."""
        detector_id = uuid4()
        dataset_id = "dataset_789"
        requested_at = datetime.utcnow()
        config = TrainingConfigDTO(experiment_name="test_config")
        trigger_metadata = {"threshold_breach": 0.15, "metric": "accuracy"}

        request = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            config=config,
            trigger_type=TrainingTrigger.PERFORMANCE_THRESHOLD,
            trigger_metadata=trigger_metadata,
            requested_by="system@example.com",
            requested_at=requested_at,
        )

        assert request.detector_id == detector_id
        assert request.dataset_id == dataset_id
        assert request.config == config
        assert request.trigger_type == TrainingTrigger.PERFORMANCE_THRESHOLD
        assert request.trigger_metadata == trigger_metadata
        assert request.requested_by == "system@example.com"
        assert request.requested_at == requested_at

    def test_trigger_types(self):
        """Test different trigger types."""
        detector_id = uuid4()
        dataset_id = "dataset_test"

        triggers = [
            TrainingTrigger.MANUAL,
            TrainingTrigger.SCHEDULED,
            TrainingTrigger.PERFORMANCE_THRESHOLD,
            TrainingTrigger.DATA_DRIFT,
            TrainingTrigger.NEW_DATA,
            TrainingTrigger.API_REQUEST,
        ]

        for trigger in triggers:
            request = TrainingRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, trigger_type=trigger
            )
            assert request.trigger_type == trigger

    def test_to_dict_and_from_dict(self):
        """Test dictionary serialization roundtrip."""
        detector_id = uuid4()
        requested_at = datetime.utcnow()

        original = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id="dataset_serialization",
            trigger_type=TrainingTrigger.DATA_DRIFT,
            requested_at=requested_at,
        )

        # Serialize to dict
        data = original.to_dict()

        # Deserialize from dict
        reconstructed = TrainingRequestDTO.from_dict(data)

        assert reconstructed.detector_id == original.detector_id
        assert reconstructed.dataset_id == original.dataset_id
        assert reconstructed.trigger_type == original.trigger_type
        assert reconstructed.requested_at == original.requested_at


class TestModelMetricsDTO:
    """Test suite for ModelMetricsDTO."""

    def test_default_creation(self):
        """Test creation with default values."""
        metrics = ModelMetricsDTO()

        assert metrics.accuracy is None
        assert metrics.precision is None
        assert metrics.recall is None
        assert metrics.f1_score is None
        assert metrics.roc_auc is None
        assert metrics.anomaly_score_mean is None
        assert metrics.anomaly_score_std is None
        assert metrics.contamination_detected is None
        assert metrics.cv_scores is None
        assert metrics.cv_mean is None
        assert metrics.cv_std is None
        assert metrics.training_time_seconds is None
        assert metrics.inference_time_ms is None
        assert metrics.model_size_mb is None
        assert metrics.confusion_matrix is None
        assert metrics.feature_importance is None
        assert metrics.additional_metrics == {}

    def test_complete_creation(self):
        """Test creation with all metrics."""
        cv_scores = [0.85, 0.87, 0.83, 0.89, 0.86]
        confusion_matrix = [[950, 50], [25, 475]]
        feature_importance = {"feature1": 0.3, "feature2": 0.25, "feature3": 0.2}
        additional_metrics = {"specificity": 0.95, "npv": 0.98}

        metrics = ModelMetricsDTO(
            accuracy=0.92,
            precision=0.89,
            recall=0.86,
            f1_score=0.875,
            roc_auc=0.94,
            anomaly_score_mean=0.15,
            anomaly_score_std=0.25,
            contamination_detected=0.05,
            cv_scores=cv_scores,
            cv_mean=0.86,
            cv_std=0.02,
            training_time_seconds=120.5,
            inference_time_ms=2.3,
            model_size_mb=15.7,
            confusion_matrix=confusion_matrix,
            feature_importance=feature_importance,
            additional_metrics=additional_metrics,
        )

        assert metrics.accuracy == 0.92
        assert metrics.precision == 0.89
        assert metrics.recall == 0.86
        assert metrics.f1_score == 0.875
        assert metrics.roc_auc == 0.94
        assert metrics.anomaly_score_mean == 0.15
        assert metrics.anomaly_score_std == 0.25
        assert metrics.contamination_detected == 0.05
        assert metrics.cv_scores == cv_scores
        assert metrics.cv_mean == 0.86
        assert metrics.cv_std == 0.02
        assert metrics.training_time_seconds == 120.5
        assert metrics.inference_time_ms == 2.3
        assert metrics.model_size_mb == 15.7
        assert metrics.confusion_matrix == confusion_matrix
        assert metrics.feature_importance == feature_importance
        assert metrics.additional_metrics == additional_metrics

    def test_to_dict_and_from_dict(self):
        """Test dictionary serialization roundtrip."""
        original = ModelMetricsDTO(
            accuracy=0.88,
            f1_score=0.85,
            cv_scores=[0.86, 0.84, 0.88],
            training_time_seconds=95.3,
        )

        data = original.to_dict()
        reconstructed = ModelMetricsDTO.from_dict(data)

        assert reconstructed.accuracy == original.accuracy
        assert reconstructed.f1_score == original.f1_score
        assert reconstructed.cv_scores == original.cv_scores
        assert reconstructed.training_time_seconds == original.training_time_seconds


class TestTrainingResultDTO:
    """Test suite for TrainingResultDTO."""

    def test_basic_creation(self):
        """Test basic training result creation."""
        detector_id = uuid4()

        result = TrainingResultDTO(
            training_id="training_123",
            detector_id=detector_id,
            dataset_id="dataset_456",
        )

        assert result.training_id == "training_123"
        assert result.detector_id == detector_id
        assert result.dataset_id == "dataset_456"
        assert result.experiment_name is None
        assert result.trigger_type == TrainingTrigger.MANUAL
        assert result.status == "completed"
        assert result.best_algorithm is None
        assert result.best_hyperparameters is None
        assert result.best_metrics is None
        assert result.model_id is None
        assert result.model_version is None
        assert result.model_path is None
        assert result.total_trials is None
        assert result.successful_trials is None
        assert result.training_duration_seconds is None
        assert result.optimization_history == []
        assert result.baseline_metrics is None
        assert result.performance_improvement is None
        assert result.peak_memory_mb is None
        assert result.total_cpu_hours is None
        assert result.started_at is None
        assert result.completed_at is None
        assert result.error_message is None
        assert result.warnings == []
        assert result.logs == []

    def test_complete_creation(self):
        """Test creation with all fields."""
        detector_id = uuid4()
        started_at = datetime.utcnow() - timedelta(hours=2)
        completed_at = datetime.utcnow()

        best_metrics = ModelMetricsDTO(accuracy=0.95, f1_score=0.92)
        baseline_metrics = ModelMetricsDTO(accuracy=0.88, f1_score=0.85)

        best_hyperparameters = {"n_estimators": 200, "contamination": 0.1}
        optimization_history = [
            {"trial": 1, "algorithm": "isolation_forest", "score": 0.85},
            {"trial": 2, "algorithm": "one_class_svm", "score": 0.87},
        ]
        warnings = ["High memory usage detected", "Consider feature selection"]
        logs = ["Training started", "Algorithm 1 completed", "Training finished"]

        result = TrainingResultDTO(
            training_id="training_complete",
            detector_id=detector_id,
            dataset_id="dataset_complete",
            experiment_name="complete_experiment",
            trigger_type=TrainingTrigger.SCHEDULED,
            status="completed",
            best_algorithm="isolation_forest",
            best_hyperparameters=best_hyperparameters,
            best_metrics=best_metrics,
            model_id="model_789",
            model_version="v2.1",
            model_path="/models/model_789_v2.1.pkl",
            total_trials=50,
            successful_trials=48,
            training_duration_seconds=7200.5,
            optimization_history=optimization_history,
            baseline_metrics=baseline_metrics,
            performance_improvement=0.07,
            peak_memory_mb=8192.5,
            total_cpu_hours=2.0,
            started_at=started_at,
            completed_at=completed_at,
            error_message=None,
            warnings=warnings,
            logs=logs,
        )

        assert result.experiment_name == "complete_experiment"
        assert result.trigger_type == TrainingTrigger.SCHEDULED
        assert result.best_algorithm == "isolation_forest"
        assert result.best_hyperparameters == best_hyperparameters
        assert result.best_metrics == best_metrics
        assert result.model_id == "model_789"
        assert result.model_version == "v2.1"
        assert result.total_trials == 50
        assert result.successful_trials == 48
        assert result.training_duration_seconds == 7200.5
        assert result.optimization_history == optimization_history
        assert result.baseline_metrics == baseline_metrics
        assert result.performance_improvement == 0.07
        assert result.peak_memory_mb == 8192.5
        assert result.total_cpu_hours == 2.0
        assert result.started_at == started_at
        assert result.completed_at == completed_at
        assert result.warnings == warnings
        assert result.logs == logs

    def test_failed_training_result(self):
        """Test failed training result scenario."""
        detector_id = uuid4()
        started_at = datetime.utcnow() - timedelta(minutes=30)

        result = TrainingResultDTO(
            training_id="training_failed",
            detector_id=detector_id,
            dataset_id="dataset_failed",
            status="failed",
            error_message="Insufficient memory for training",
            started_at=started_at,
            warnings=["Memory limit exceeded", "Training aborted"],
            logs=["Training started", "Memory error encountered", "Training stopped"],
        )

        assert result.status == "failed"
        assert result.error_message == "Insufficient memory for training"
        assert "Memory limit exceeded" in result.warnings
        assert "Training stopped" in result.logs

    def test_to_dict_and_from_dict(self):
        """Test dictionary serialization roundtrip."""
        detector_id = uuid4()
        started_at = datetime.utcnow()
        best_metrics = ModelMetricsDTO(accuracy=0.91)

        original = TrainingResultDTO(
            training_id="serialization_test",
            detector_id=detector_id,
            dataset_id="dataset_test",
            best_algorithm="isolation_forest",
            best_metrics=best_metrics,
            started_at=started_at,
        )

        # Serialize to dict
        data = original.to_dict()

        # Deserialize from dict
        reconstructed = TrainingResultDTO.from_dict(data)

        assert reconstructed.training_id == original.training_id
        assert reconstructed.detector_id == original.detector_id
        assert reconstructed.best_algorithm == original.best_algorithm
        assert reconstructed.best_metrics.accuracy == original.best_metrics.accuracy
        assert reconstructed.started_at == original.started_at


class TestTrainingStatusDTO:
    """Test suite for TrainingStatusDTO."""

    def test_basic_creation(self):
        """Test basic training status creation."""
        status = TrainingStatusDTO(
            training_id="status_test",
            status="in_progress",
            progress_percentage=45.5,
            current_step="hyperparameter_optimization",
        )

        assert status.training_id == "status_test"
        assert status.status == "in_progress"
        assert status.progress_percentage == 45.5
        assert status.current_step == "hyperparameter_optimization"
        assert status.current_algorithm is None
        assert status.current_trial is None
        assert status.total_trials is None
        assert status.current_score is None
        assert status.best_score is None
        assert status.started_at is None
        assert status.estimated_completion is None
        assert status.memory_usage_mb is None
        assert status.cpu_usage_percent is None
        assert status.current_message is None
        assert status.recent_logs == []
        assert status.warnings == []

    def test_complete_creation(self):
        """Test creation with all fields."""
        started_at = datetime.utcnow() - timedelta(minutes=30)
        estimated_completion = datetime.utcnow() + timedelta(minutes=15)
        recent_logs = ["Starting trial 5", "Evaluating model", "Trial 5 completed"]
        warnings = ["High memory usage", "Slow convergence"]

        status = TrainingStatusDTO(
            training_id="complete_status",
            status="training",
            progress_percentage=75.2,
            current_step="model_evaluation",
            current_algorithm="one_class_svm",
            current_trial=5,
            total_trials=10,
            current_score=0.87,
            best_score=0.92,
            started_at=started_at,
            estimated_completion=estimated_completion,
            memory_usage_mb=4096.5,
            cpu_usage_percent=85.3,
            current_message="Evaluating hyperparameter set 5/10",
            recent_logs=recent_logs,
            warnings=warnings,
        )

        assert status.training_id == "complete_status"
        assert status.status == "training"
        assert status.progress_percentage == 75.2
        assert status.current_step == "model_evaluation"
        assert status.current_algorithm == "one_class_svm"
        assert status.current_trial == 5
        assert status.total_trials == 10
        assert status.current_score == 0.87
        assert status.best_score == 0.92
        assert status.started_at == started_at
        assert status.estimated_completion == estimated_completion
        assert status.memory_usage_mb == 4096.5
        assert status.cpu_usage_percent == 85.3
        assert status.current_message == "Evaluating hyperparameter set 5/10"
        assert status.recent_logs == recent_logs
        assert status.warnings == warnings

    def test_progress_values(self):
        """Test different progress percentage values."""
        progress_values = [0.0, 25.5, 50.0, 75.8, 100.0]

        for progress in progress_values:
            status = TrainingStatusDTO(
                training_id="progress_test",
                status="in_progress",
                progress_percentage=progress,
                current_step="test_step",
            )
            assert status.progress_percentage == progress

    def test_to_dict(self):
        """Test conversion to dictionary."""
        started_at = datetime.utcnow()

        status = TrainingStatusDTO(
            training_id="dict_test",
            status="running",
            progress_percentage=60.0,
            current_step="validation",
            started_at=started_at,
        )

        result = status.to_dict()

        assert result["training_id"] == "dict_test"
        assert result["status"] == "running"
        assert result["progress_percentage"] == 60.0
        assert result["current_step"] == "validation"
        assert "timestamp" in result  # Added automatically

    def test_from_dict(self):
        """Test creation from dictionary."""
        started_at = datetime.utcnow()

        data = {
            "training_id": "from_dict_test",
            "status": "completed",
            "progress_percentage": 100.0,
            "current_step": "finished",
            "started_at": started_at.isoformat(),
            "timestamp": datetime.utcnow().isoformat(),  # This should be removed
        }

        status = TrainingStatusDTO.from_dict(data)

        assert status.training_id == "from_dict_test"
        assert status.status == "completed"
        assert status.progress_percentage == 100.0
        assert status.current_step == "finished"
        assert status.started_at == started_at


class TestTrainingDTOIntegration:
    """Integration tests for training DTOs."""

    def test_complete_training_workflow(self):
        """Test complete training workflow using multiple DTOs."""
        detector_id = uuid4()
        dataset_id = "workflow_dataset"

        # Step 1: Create training configuration
        config = TrainingConfigDTO(
            experiment_name="integration_test",
            automl_config=AutoMLConfigDTO(max_algorithms=3),
            validation_config=ValidationConfigDTO(cv_folds=5),
            resource_constraints=ResourceConstraintsDTO(max_memory_mb=4096),
        )

        # Step 2: Create training request
        request = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            config=config,
            trigger_type=TrainingTrigger.API_REQUEST,
        )

        # Step 3: Initial status
        initial_status = TrainingStatusDTO(
            training_id="integration_training",
            status="starting",
            progress_percentage=0.0,
            current_step="initialization",
        )

        # Step 4: Progress status
        progress_status = TrainingStatusDTO(
            training_id="integration_training",
            status="training",
            progress_percentage=50.0,
            current_step="model_training",
            current_algorithm="isolation_forest",
            current_trial=2,
            total_trials=3,
            current_score=0.85,
            best_score=0.87,
        )

        # Step 5: Training result
        best_metrics = ModelMetricsDTO(accuracy=0.92, f1_score=0.89)
        result = TrainingResultDTO(
            training_id="integration_training",
            detector_id=detector_id,
            dataset_id=dataset_id,
            experiment_name=config.experiment_name,
            trigger_type=request.trigger_type,
            status="completed",
            best_algorithm="isolation_forest",
            best_metrics=best_metrics,
            total_trials=3,
            successful_trials=3,
        )

        # Verify workflow consistency
        assert request.detector_id == detector_id
        assert request.config.experiment_name == config.experiment_name
        assert result.experiment_name == config.experiment_name
        assert result.trigger_type == request.trigger_type
        assert progress_status.total_trials == result.total_trials

    def test_serialization_roundtrip_complex(self):
        """Test serialization roundtrip for complex nested DTOs."""
        detector_id = uuid4()
        created_at = datetime.utcnow()

        # Create complex nested configuration
        original_config = TrainingConfigDTO(
            experiment_name="complex_serialization",
            detector_id=detector_id,
            created_at=created_at,
            automl_config=AutoMLConfigDTO(
                max_algorithms=5,
                algorithm_whitelist=["isolation_forest", "one_class_svm"],
            ),
            validation_config=ValidationConfigDTO(
                validation_strategy="cross_validation",
                cv_folds=10,
            ),
            resource_constraints=ResourceConstraintsDTO(
                max_memory_mb=8192,
                enable_gpu=True,
            ),
            notification_config=NotificationConfigDTO(
                notification_channels=["email", "slack"],
                email_recipients=["test@example.com"],
            ),
        )

        # Serialize to JSON-compatible dict
        data = original_config.to_dict()
        json_str = json.dumps(data)
        parsed_data = json.loads(json_str)

        # Reconstruct from parsed data
        reconstructed_config = TrainingConfigDTO.from_dict(parsed_data)

        # Verify complex nested structure
        assert reconstructed_config.experiment_name == original_config.experiment_name
        assert reconstructed_config.detector_id == original_config.detector_id
        assert reconstructed_config.created_at == original_config.created_at
        assert (
            reconstructed_config.automl_config.max_algorithms
            == original_config.automl_config.max_algorithms
        )
        assert (
            reconstructed_config.automl_config.algorithm_whitelist
            == original_config.automl_config.algorithm_whitelist
        )
        assert (
            reconstructed_config.validation_config.validation_strategy
            == original_config.validation_config.validation_strategy
        )
        assert (
            reconstructed_config.validation_config.cv_folds
            == original_config.validation_config.cv_folds
        )
        assert (
            reconstructed_config.resource_constraints.max_memory_mb
            == original_config.resource_constraints.max_memory_mb
        )
        assert (
            reconstructed_config.resource_constraints.enable_gpu
            == original_config.resource_constraints.enable_gpu
        )
        assert (
            reconstructed_config.notification_config.notification_channels
            == original_config.notification_config.notification_channels
        )
        assert (
            reconstructed_config.notification_config.email_recipients
            == original_config.notification_config.email_recipients
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
