"""Tests for Training DTOs."""

from datetime import datetime
from uuid import uuid4

import pytest

from monorepo.application.dto.training_dto import (
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


class TestTrainingTrigger:
    """Test suite for TrainingTrigger enum."""

    def test_enum_values(self):
        """Test all enum values."""
        assert TrainingTrigger.MANUAL.value == "manual"
        assert TrainingTrigger.SCHEDULED.value == "scheduled"
        assert TrainingTrigger.PERFORMANCE_THRESHOLD.value == "performance_threshold"
        assert TrainingTrigger.DATA_DRIFT.value == "data_drift"
        assert TrainingTrigger.NEW_DATA.value == "new_data"
        assert TrainingTrigger.API_REQUEST.value == "api_request"

    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert TrainingTrigger("manual") == TrainingTrigger.MANUAL
        assert TrainingTrigger("scheduled") == TrainingTrigger.SCHEDULED
        assert (
            TrainingTrigger("performance_threshold")
            == TrainingTrigger.PERFORMANCE_THRESHOLD
        )
        assert TrainingTrigger("data_drift") == TrainingTrigger.DATA_DRIFT
        assert TrainingTrigger("new_data") == TrainingTrigger.NEW_DATA
        assert TrainingTrigger("api_request") == TrainingTrigger.API_REQUEST

    def test_invalid_enum_value(self):
        """Test invalid enum value raises ValueError."""
        with pytest.raises(ValueError):
            TrainingTrigger("invalid_trigger")


class TestTrainingPriorityLevel:
    """Test suite for TrainingPriorityLevel enum."""

    def test_enum_values(self):
        """Test all enum values."""
        assert TrainingPriorityLevel.LOW.value == "low"
        assert TrainingPriorityLevel.NORMAL.value == "normal"
        assert TrainingPriorityLevel.HIGH.value == "high"
        assert TrainingPriorityLevel.URGENT.value == "urgent"

    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert TrainingPriorityLevel("low") == TrainingPriorityLevel.LOW
        assert TrainingPriorityLevel("normal") == TrainingPriorityLevel.NORMAL
        assert TrainingPriorityLevel("high") == TrainingPriorityLevel.HIGH
        assert TrainingPriorityLevel("urgent") == TrainingPriorityLevel.URGENT

    def test_invalid_enum_value(self):
        """Test invalid enum value raises ValueError."""
        with pytest.raises(ValueError):
            TrainingPriorityLevel("invalid_priority")


class TestResourceConstraintsDTO:
    """Test suite for ResourceConstraintsDTO."""

    def test_default_creation(self):
        """Test creating with default values."""
        dto = ResourceConstraintsDTO()

        assert dto.max_training_time_seconds is None
        assert dto.max_time_per_trial is None
        assert dto.max_memory_mb is None
        assert dto.max_cpu_cores is None
        assert dto.max_gpu_memory_mb is None
        assert dto.enable_gpu is False
        assert dto.n_jobs == 1
        assert dto.max_concurrent_trials is None

    def test_creation_with_values(self):
        """Test creating with specific values."""
        dto = ResourceConstraintsDTO(
            max_training_time_seconds=3600,
            max_time_per_trial=300,
            max_memory_mb=8192,
            max_cpu_cores=8,
            max_gpu_memory_mb=4096,
            enable_gpu=True,
            n_jobs=4,
            max_concurrent_trials=2,
        )

        assert dto.max_training_time_seconds == 3600
        assert dto.max_time_per_trial == 300
        assert dto.max_memory_mb == 8192
        assert dto.max_cpu_cores == 8
        assert dto.max_gpu_memory_mb == 4096
        assert dto.enable_gpu is True
        assert dto.n_jobs == 4
        assert dto.max_concurrent_trials == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        dto = ResourceConstraintsDTO(
            max_training_time_seconds=1800,
            max_memory_mb=4096,
            enable_gpu=True,
            n_jobs=2,
        )

        result = dto.to_dict()

        assert result["max_training_time_seconds"] == 1800
        assert result["max_time_per_trial"] is None
        assert result["max_memory_mb"] == 4096
        assert result["max_cpu_cores"] is None
        assert result["max_gpu_memory_mb"] is None
        assert result["enable_gpu"] is True
        assert result["n_jobs"] == 2
        assert result["max_concurrent_trials"] is None

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "max_training_time_seconds": 1800,
            "max_time_per_trial": 300,
            "max_memory_mb": 4096,
            "max_cpu_cores": 4,
            "max_gpu_memory_mb": 2048,
            "enable_gpu": True,
            "n_jobs": 2,
            "max_concurrent_trials": 3,
        }

        dto = ResourceConstraintsDTO.from_dict(data)

        assert dto.max_training_time_seconds == 1800
        assert dto.max_time_per_trial == 300
        assert dto.max_memory_mb == 4096
        assert dto.max_cpu_cores == 4
        assert dto.max_gpu_memory_mb == 2048
        assert dto.enable_gpu is True
        assert dto.n_jobs == 2
        assert dto.max_concurrent_trials == 3

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion to/from dict."""
        original = ResourceConstraintsDTO(
            max_training_time_seconds=7200,
            max_memory_mb=16384,
            enable_gpu=True,
            n_jobs=8,
        )

        data = original.to_dict()
        restored = ResourceConstraintsDTO.from_dict(data)

        assert restored.max_training_time_seconds == original.max_training_time_seconds
        assert restored.max_memory_mb == original.max_memory_mb
        assert restored.enable_gpu == original.enable_gpu
        assert restored.n_jobs == original.n_jobs


class TestAutoMLConfigDTO:
    """Test suite for AutoMLConfigDTO."""

    def test_default_creation(self):
        """Test creating with default values."""
        dto = AutoMLConfigDTO()

        assert dto.enable_automl is True
        assert dto.optimization_objective == "auc"
        assert dto.max_algorithms == 3
        assert dto.enable_ensemble is True
        assert dto.ensemble_method == "voting"
        assert dto.algorithm_whitelist is None
        assert dto.algorithm_blacklist is None
        assert dto.model_selection_strategy == "best_single_metric"
        assert dto.metric_weights is None

    def test_creation_with_values(self):
        """Test creating with specific values."""
        dto = AutoMLConfigDTO(
            enable_automl=False,
            optimization_objective="f1_score",
            max_algorithms=5,
            enable_ensemble=False,
            ensemble_method="stacking",
            algorithm_whitelist=["IsolationForest", "OneClassSVM"],
            algorithm_blacklist=["DBSCAN"],
            model_selection_strategy="weighted_score",
            metric_weights={"precision": 0.6, "recall": 0.4},
        )

        assert dto.enable_automl is False
        assert dto.optimization_objective == "f1_score"
        assert dto.max_algorithms == 5
        assert dto.enable_ensemble is False
        assert dto.ensemble_method == "stacking"
        assert dto.algorithm_whitelist == ["IsolationForest", "OneClassSVM"]
        assert dto.algorithm_blacklist == ["DBSCAN"]
        assert dto.model_selection_strategy == "weighted_score"
        assert dto.metric_weights == {"precision": 0.6, "recall": 0.4}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        dto = AutoMLConfigDTO(
            optimization_objective="precision",
            max_algorithms=2,
            algorithm_whitelist=["IsolationForest"],
        )

        result = dto.to_dict()

        assert result["enable_automl"] is True
        assert result["optimization_objective"] == "precision"
        assert result["max_algorithms"] == 2
        assert result["enable_ensemble"] is True
        assert result["ensemble_method"] == "voting"
        assert result["algorithm_whitelist"] == ["IsolationForest"]
        assert result["algorithm_blacklist"] is None
        assert result["model_selection_strategy"] == "best_single_metric"
        assert result["metric_weights"] is None

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "enable_automl": False,
            "optimization_objective": "recall",
            "max_algorithms": 4,
            "enable_ensemble": True,
            "ensemble_method": "blending",
            "algorithm_whitelist": ["IsolationForest", "LocalOutlierFactor"],
            "algorithm_blacklist": ["DBSCAN", "KMeans"],
            "model_selection_strategy": "pareto_optimal",
            "metric_weights": {"precision": 0.5, "recall": 0.3, "f1_score": 0.2},
        }

        dto = AutoMLConfigDTO.from_dict(data)

        assert dto.enable_automl is False
        assert dto.optimization_objective == "recall"
        assert dto.max_algorithms == 4
        assert dto.enable_ensemble is True
        assert dto.ensemble_method == "blending"
        assert dto.algorithm_whitelist == ["IsolationForest", "LocalOutlierFactor"]
        assert dto.algorithm_blacklist == ["DBSCAN", "KMeans"]
        assert dto.model_selection_strategy == "pareto_optimal"
        assert dto.metric_weights == {"precision": 0.5, "recall": 0.3, "f1_score": 0.2}


class TestValidationConfigDTO:
    """Test suite for ValidationConfigDTO."""

    def test_default_creation(self):
        """Test creating with default values."""
        dto = ValidationConfigDTO()

        assert dto.validation_strategy == "holdout"
        assert dto.validation_split == 0.2
        assert dto.cv_folds == 5
        assert dto.cv_strategy == "stratified"
        assert dto.enable_early_stopping is True
        assert dto.early_stopping_patience == 10
        assert dto.early_stopping_metric == "validation_score"
        assert dto.early_stopping_threshold == 0.001
        assert dto.enable_data_validation is True
        assert dto.min_samples_required == 100
        assert dto.max_missing_ratio == 0.1

    def test_creation_with_values(self):
        """Test creating with specific values."""
        dto = ValidationConfigDTO(
            validation_strategy="cross_validation",
            validation_split=0.3,
            cv_folds=10,
            cv_strategy="kfold",
            enable_early_stopping=False,
            early_stopping_patience=5,
            early_stopping_metric="loss",
            early_stopping_threshold=0.005,
            enable_data_validation=False,
            min_samples_required=50,
            max_missing_ratio=0.05,
        )

        assert dto.validation_strategy == "cross_validation"
        assert dto.validation_split == 0.3
        assert dto.cv_folds == 10
        assert dto.cv_strategy == "kfold"
        assert dto.enable_early_stopping is False
        assert dto.early_stopping_patience == 5
        assert dto.early_stopping_metric == "loss"
        assert dto.early_stopping_threshold == 0.005
        assert dto.enable_data_validation is False
        assert dto.min_samples_required == 50
        assert dto.max_missing_ratio == 0.05

    def test_to_dict_and_from_dict(self):
        """Test roundtrip conversion."""
        original = ValidationConfigDTO(
            validation_strategy="time_series_split",
            cv_folds=3,
            enable_early_stopping=True,
            early_stopping_patience=15,
        )

        data = original.to_dict()
        restored = ValidationConfigDTO.from_dict(data)

        assert restored.validation_strategy == original.validation_strategy
        assert restored.cv_folds == original.cv_folds
        assert restored.enable_early_stopping == original.enable_early_stopping
        assert restored.early_stopping_patience == original.early_stopping_patience


class TestNotificationConfigDTO:
    """Test suite for NotificationConfigDTO."""

    def test_default_creation(self):
        """Test creating with default values."""
        dto = NotificationConfigDTO()

        assert dto.enable_notifications is True
        assert dto.notification_channels == ["websocket"]
        assert dto.email_recipients == []
        assert dto.email_on_completion is True
        assert dto.email_on_failure is True
        assert dto.webhook_urls == []
        assert dto.webhook_events == ["completion", "failure"]
        assert dto.slack_webhook_url is None
        assert dto.slack_channel is None

    def test_creation_with_values(self):
        """Test creating with specific values."""
        dto = NotificationConfigDTO(
            enable_notifications=False,
            notification_channels=["email", "slack"],
            email_recipients=["user1@example.com", "user2@example.com"],
            email_on_completion=False,
            email_on_failure=True,
            webhook_urls=["https://example.com/webhook"],
            webhook_events=["start", "completion", "failure"],
            slack_webhook_url="https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
            slack_channel="#alerts",
        )

        assert dto.enable_notifications is False
        assert dto.notification_channels == ["email", "slack"]
        assert dto.email_recipients == ["user1@example.com", "user2@example.com"]
        assert dto.email_on_completion is False
        assert dto.email_on_failure is True
        assert dto.webhook_urls == ["https://example.com/webhook"]
        assert dto.webhook_events == ["start", "completion", "failure"]
        assert (
            dto.slack_webhook_url
            == "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        )
        assert dto.slack_channel == "#alerts"

    def test_to_dict_and_from_dict(self):
        """Test roundtrip conversion."""
        original = NotificationConfigDTO(
            notification_channels=["email", "webhook"],
            email_recipients=["admin@example.com"],
            webhook_urls=["https://example.com/hook1", "https://example.com/hook2"],
        )

        data = original.to_dict()
        restored = NotificationConfigDTO.from_dict(data)

        assert restored.notification_channels == original.notification_channels
        assert restored.email_recipients == original.email_recipients
        assert restored.webhook_urls == original.webhook_urls


class TestModelMetricsDTO:
    """Test suite for ModelMetricsDTO."""

    def test_default_creation(self):
        """Test creating with default values."""
        dto = ModelMetricsDTO()

        assert dto.accuracy is None
        assert dto.precision is None
        assert dto.recall is None
        assert dto.f1_score is None
        assert dto.roc_auc is None
        assert dto.anomaly_score_mean is None
        assert dto.anomaly_score_std is None
        assert dto.contamination_detected is None
        assert dto.cv_scores is None
        assert dto.cv_mean is None
        assert dto.cv_std is None
        assert dto.training_time_seconds is None
        assert dto.inference_time_ms is None
        assert dto.model_size_mb is None
        assert dto.confusion_matrix is None
        assert dto.feature_importance is None
        assert dto.additional_metrics == {}

    def test_creation_with_values(self):
        """Test creating with specific values."""
        dto = ModelMetricsDTO(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            roc_auc=0.94,
            anomaly_score_mean=0.35,
            anomaly_score_std=0.15,
            contamination_detected=0.05,
            cv_scores=[0.91, 0.93, 0.89, 0.94, 0.92],
            cv_mean=0.92,
            cv_std=0.018,
            training_time_seconds=120.5,
            inference_time_ms=2.3,
            model_size_mb=45.2,
            confusion_matrix=[[950, 50], [25, 475]],
            feature_importance={"feature1": 0.4, "feature2": 0.35, "feature3": 0.25},
            additional_metrics={"silhouette_score": 0.65, "calinski_harabasz": 1250.3},
        )

        assert dto.accuracy == 0.95
        assert dto.precision == 0.92
        assert dto.recall == 0.88
        assert dto.f1_score == 0.90
        assert dto.roc_auc == 0.94
        assert dto.anomaly_score_mean == 0.35
        assert dto.anomaly_score_std == 0.15
        assert dto.contamination_detected == 0.05
        assert dto.cv_scores == [0.91, 0.93, 0.89, 0.94, 0.92]
        assert dto.cv_mean == 0.92
        assert dto.cv_std == 0.018
        assert dto.training_time_seconds == 120.5
        assert dto.inference_time_ms == 2.3
        assert dto.model_size_mb == 45.2
        assert dto.confusion_matrix == [[950, 50], [25, 475]]
        assert dto.feature_importance == {
            "feature1": 0.4,
            "feature2": 0.35,
            "feature3": 0.25,
        }
        assert dto.additional_metrics == {
            "silhouette_score": 0.65,
            "calinski_harabasz": 1250.3,
        }

    def test_to_dict_and_from_dict(self):
        """Test roundtrip conversion."""
        original = ModelMetricsDTO(
            accuracy=0.87,
            precision=0.85,
            recall=0.89,
            f1_score=0.87,
            roc_auc=0.91,
            training_time_seconds=95.2,
            feature_importance={"feature1": 0.6, "feature2": 0.4},
        )

        data = original.to_dict()
        restored = ModelMetricsDTO.from_dict(data)

        assert restored.accuracy == original.accuracy
        assert restored.precision == original.precision
        assert restored.recall == original.recall
        assert restored.f1_score == original.f1_score
        assert restored.roc_auc == original.roc_auc
        assert restored.training_time_seconds == original.training_time_seconds
        assert restored.feature_importance == original.feature_importance


class TestTrainingConfigDTO:
    """Test suite for TrainingConfigDTO."""

    def test_default_creation(self):
        """Test creating with default values."""
        dto = TrainingConfigDTO()

        assert dto.experiment_name is None
        assert dto.description is None
        assert dto.tags == []
        assert dto.dataset_id is None
        assert dto.detector_id is None
        assert dto.target_algorithms is None
        assert isinstance(dto.automl_config, AutoMLConfigDTO)
        assert isinstance(dto.validation_config, ValidationConfigDTO)
        assert dto.optimization_config is None
        assert isinstance(dto.resource_constraints, ResourceConstraintsDTO)
        assert isinstance(dto.notification_config, NotificationConfigDTO)
        assert dto.schedule_cron is None
        assert dto.schedule_enabled is False
        assert dto.performance_monitoring_enabled is True
        assert dto.retrain_threshold == 0.05
        assert dto.performance_window_days == 7
        assert dto.auto_deploy_best_model is False
        assert dto.model_versioning_enabled is True
        assert dto.keep_model_versions == 5
        assert dto.enable_model_explainability is True
        assert dto.enable_drift_detection is True
        assert dto.enable_feature_selection is True
        assert dto.created_by is None
        assert dto.created_at is None
        assert dto.priority == TrainingPriorityLevel.NORMAL

    def test_creation_with_values(self):
        """Test creating with specific values."""
        detector_id = uuid4()
        created_at = datetime.now()
        automl_config = AutoMLConfigDTO(max_algorithms=5)
        validation_config = ValidationConfigDTO(cv_folds=10)
        resource_constraints = ResourceConstraintsDTO(n_jobs=4)
        notification_config = NotificationConfigDTO(enable_notifications=False)

        dto = TrainingConfigDTO(
            experiment_name="fraud_detection_experiment",
            description="Comprehensive fraud detection training",
            tags=["fraud", "production", "v2"],
            dataset_id="dataset_123",
            detector_id=detector_id,
            target_algorithms=["IsolationForest", "OneClassSVM"],
            automl_config=automl_config,
            validation_config=validation_config,
            resource_constraints=resource_constraints,
            notification_config=notification_config,
            schedule_cron="0 2 * * *",
            schedule_enabled=True,
            performance_monitoring_enabled=True,
            retrain_threshold=0.1,
            performance_window_days=14,
            auto_deploy_best_model=True,
            model_versioning_enabled=True,
            keep_model_versions=10,
            enable_model_explainability=True,
            enable_drift_detection=True,
            enable_feature_selection=False,
            created_by="user123",
            created_at=created_at,
            priority=TrainingPriorityLevel.HIGH,
        )

        assert dto.experiment_name == "fraud_detection_experiment"
        assert dto.description == "Comprehensive fraud detection training"
        assert dto.tags == ["fraud", "production", "v2"]
        assert dto.dataset_id == "dataset_123"
        assert dto.detector_id == detector_id
        assert dto.target_algorithms == ["IsolationForest", "OneClassSVM"]
        assert dto.automl_config == automl_config
        assert dto.validation_config == validation_config
        assert dto.resource_constraints == resource_constraints
        assert dto.notification_config == notification_config
        assert dto.schedule_cron == "0 2 * * *"
        assert dto.schedule_enabled is True
        assert dto.performance_monitoring_enabled is True
        assert dto.retrain_threshold == 0.1
        assert dto.performance_window_days == 14
        assert dto.auto_deploy_best_model is True
        assert dto.model_versioning_enabled is True
        assert dto.keep_model_versions == 10
        assert dto.enable_model_explainability is True
        assert dto.enable_drift_detection is True
        assert dto.enable_feature_selection is False
        assert dto.created_by == "user123"
        assert dto.created_at == created_at
        assert dto.priority == TrainingPriorityLevel.HIGH

    def test_to_dict_and_from_dict(self):
        """Test roundtrip conversion."""
        detector_id = uuid4()
        created_at = datetime.now()

        original = TrainingConfigDTO(
            experiment_name="test_experiment",
            detector_id=detector_id,
            created_at=created_at,
            priority=TrainingPriorityLevel.HIGH,
        )

        data = original.to_dict()
        restored = TrainingConfigDTO.from_dict(data)

        assert restored.experiment_name == original.experiment_name
        assert restored.detector_id == original.detector_id
        assert restored.created_at == original.created_at
        assert restored.priority == original.priority


class TestTrainingRequestDTO:
    """Test suite for TrainingRequestDTO."""

    def test_creation_with_required_fields(self):
        """Test creating with required fields only."""
        detector_id = uuid4()
        requested_at = datetime.now()

        dto = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id="dataset_123",
            requested_at=requested_at,
        )

        assert dto.detector_id == detector_id
        assert dto.dataset_id == "dataset_123"
        assert dto.config is None
        assert dto.trigger_type == TrainingTrigger.MANUAL
        assert dto.trigger_metadata == {}
        assert dto.requested_by is None
        assert dto.requested_at == requested_at

    def test_creation_with_all_fields(self):
        """Test creating with all fields."""
        detector_id = uuid4()
        requested_at = datetime.now()
        config = TrainingConfigDTO(experiment_name="test_experiment")

        dto = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id="dataset_456",
            config=config,
            trigger_type=TrainingTrigger.SCHEDULED,
            trigger_metadata={"schedule_id": "sched_123", "retry_count": 1},
            requested_by="user456",
            requested_at=requested_at,
        )

        assert dto.detector_id == detector_id
        assert dto.dataset_id == "dataset_456"
        assert dto.config == config
        assert dto.trigger_type == TrainingTrigger.SCHEDULED
        assert dto.trigger_metadata == {"schedule_id": "sched_123", "retry_count": 1}
        assert dto.requested_by == "user456"
        assert dto.requested_at == requested_at

    def test_to_dict_and_from_dict(self):
        """Test roundtrip conversion."""
        detector_id = uuid4()
        requested_at = datetime.now()
        config = TrainingConfigDTO(experiment_name="conversion_test")

        original = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id="dataset_789",
            config=config,
            trigger_type=TrainingTrigger.DATA_DRIFT,
            trigger_metadata={"drift_score": 0.85},
            requested_by="system",
            requested_at=requested_at,
        )

        data = original.to_dict()
        restored = TrainingRequestDTO.from_dict(data)

        assert restored.detector_id == original.detector_id
        assert restored.dataset_id == original.dataset_id
        assert restored.config.experiment_name == original.config.experiment_name
        assert restored.trigger_type == original.trigger_type
        assert restored.trigger_metadata == original.trigger_metadata
        assert restored.requested_by == original.requested_by
        assert restored.requested_at == original.requested_at


class TestTrainingResultDTO:
    """Test suite for TrainingResultDTO."""

    def test_creation_with_required_fields(self):
        """Test creating with required fields only."""
        detector_id = uuid4()

        dto = TrainingResultDTO(
            training_id="training_123",
            detector_id=detector_id,
            dataset_id="dataset_456",
        )

        assert dto.training_id == "training_123"
        assert dto.detector_id == detector_id
        assert dto.dataset_id == "dataset_456"
        assert dto.experiment_name is None
        assert dto.trigger_type == TrainingTrigger.MANUAL
        assert dto.status == "completed"
        assert dto.best_algorithm is None
        assert dto.best_hyperparameters is None
        assert dto.best_metrics is None
        assert dto.model_id is None
        assert dto.model_version is None
        assert dto.model_path is None
        assert dto.total_trials is None
        assert dto.successful_trials is None
        assert dto.training_duration_seconds is None
        assert dto.optimization_history == []
        assert dto.baseline_metrics is None
        assert dto.performance_improvement is None
        assert dto.peak_memory_mb is None
        assert dto.total_cpu_hours is None
        assert dto.started_at is None
        assert dto.completed_at is None
        assert dto.error_message is None
        assert dto.warnings == []
        assert dto.logs == []

    def test_creation_with_all_fields(self):
        """Test creating with all fields."""
        detector_id = uuid4()
        started_at = datetime.now()
        completed_at = datetime.now()
        best_metrics = ModelMetricsDTO(accuracy=0.95, precision=0.92)
        baseline_metrics = ModelMetricsDTO(accuracy=0.85, precision=0.82)

        dto = TrainingResultDTO(
            training_id="training_456",
            detector_id=detector_id,
            dataset_id="dataset_789",
            experiment_name="comprehensive_test",
            trigger_type=TrainingTrigger.PERFORMANCE_THRESHOLD,
            status="completed",
            best_algorithm="IsolationForest",
            best_hyperparameters={"n_estimators": 100, "contamination": 0.1},
            best_metrics=best_metrics,
            model_id="model_123",
            model_version="v1.2.3",
            model_path="/models/model_123_v1.2.3.pkl",
            total_trials=50,
            successful_trials=48,
            training_duration_seconds=1800.5,
            optimization_history=[
                {"trial": 1, "algorithm": "IsolationForest", "score": 0.85},
                {"trial": 2, "algorithm": "OneClassSVM", "score": 0.88},
            ],
            baseline_metrics=baseline_metrics,
            performance_improvement=0.1,
            peak_memory_mb=2048.5,
            total_cpu_hours=2.5,
            started_at=started_at,
            completed_at=completed_at,
            error_message=None,
            warnings=["Warning: High memory usage detected"],
            logs=["Training started", "Model trained successfully"],
        )

        assert dto.training_id == "training_456"
        assert dto.detector_id == detector_id
        assert dto.dataset_id == "dataset_789"
        assert dto.experiment_name == "comprehensive_test"
        assert dto.trigger_type == TrainingTrigger.PERFORMANCE_THRESHOLD
        assert dto.status == "completed"
        assert dto.best_algorithm == "IsolationForest"
        assert dto.best_hyperparameters == {"n_estimators": 100, "contamination": 0.1}
        assert dto.best_metrics == best_metrics
        assert dto.model_id == "model_123"
        assert dto.model_version == "v1.2.3"
        assert dto.model_path == "/models/model_123_v1.2.3.pkl"
        assert dto.total_trials == 50
        assert dto.successful_trials == 48
        assert dto.training_duration_seconds == 1800.5
        assert len(dto.optimization_history) == 2
        assert dto.baseline_metrics == baseline_metrics
        assert dto.performance_improvement == 0.1
        assert dto.peak_memory_mb == 2048.5
        assert dto.total_cpu_hours == 2.5
        assert dto.started_at == started_at
        assert dto.completed_at == completed_at
        assert dto.error_message is None
        assert dto.warnings == ["Warning: High memory usage detected"]
        assert dto.logs == ["Training started", "Model trained successfully"]

    def test_to_dict_and_from_dict(self):
        """Test roundtrip conversion."""
        detector_id = uuid4()
        started_at = datetime.now()
        completed_at = datetime.now()
        best_metrics = ModelMetricsDTO(accuracy=0.91, f1_score=0.89)

        original = TrainingResultDTO(
            training_id="training_conversion_test",
            detector_id=detector_id,
            dataset_id="dataset_conversion",
            experiment_name="conversion_experiment",
            trigger_type=TrainingTrigger.NEW_DATA,
            best_algorithm="LocalOutlierFactor",
            best_metrics=best_metrics,
            started_at=started_at,
            completed_at=completed_at,
            warnings=["Memory warning"],
            logs=["Log entry 1", "Log entry 2"],
        )

        data = original.to_dict()
        restored = TrainingResultDTO.from_dict(data)

        assert restored.training_id == original.training_id
        assert restored.detector_id == original.detector_id
        assert restored.dataset_id == original.dataset_id
        assert restored.experiment_name == original.experiment_name
        assert restored.trigger_type == original.trigger_type
        assert restored.best_algorithm == original.best_algorithm
        assert restored.best_metrics.accuracy == original.best_metrics.accuracy
        assert restored.best_metrics.f1_score == original.best_metrics.f1_score
        assert restored.started_at == original.started_at
        assert restored.completed_at == original.completed_at
        assert restored.warnings == original.warnings
        assert restored.logs == original.logs


class TestTrainingStatusDTO:
    """Test suite for TrainingStatusDTO."""

    def test_creation_with_required_fields(self):
        """Test creating with required fields only."""
        dto = TrainingStatusDTO(
            training_id="training_status_123",
            status="running",
            progress_percentage=45.5,
            current_step="hyperparameter_tuning",
        )

        assert dto.training_id == "training_status_123"
        assert dto.status == "running"
        assert dto.progress_percentage == 45.5
        assert dto.current_step == "hyperparameter_tuning"
        assert dto.current_algorithm is None
        assert dto.current_trial is None
        assert dto.total_trials is None
        assert dto.current_score is None
        assert dto.best_score is None
        assert dto.started_at is None
        assert dto.estimated_completion is None
        assert dto.memory_usage_mb is None
        assert dto.cpu_usage_percent is None
        assert dto.current_message is None
        assert dto.recent_logs == []
        assert dto.warnings == []

    def test_creation_with_all_fields(self):
        """Test creating with all fields."""
        started_at = datetime.now()
        estimated_completion = datetime.now()

        dto = TrainingStatusDTO(
            training_id="training_status_456",
            status="running",
            progress_percentage=67.8,
            current_step="model_evaluation",
            current_algorithm="IsolationForest",
            current_trial=15,
            total_trials=25,
            current_score=0.89,
            best_score=0.92,
            started_at=started_at,
            estimated_completion=estimated_completion,
            memory_usage_mb=1024.5,
            cpu_usage_percent=75.2,
            current_message="Evaluating model performance",
            recent_logs=["Starting trial 15", "Model training completed"],
            warnings=["High CPU usage detected"],
        )

        assert dto.training_id == "training_status_456"
        assert dto.status == "running"
        assert dto.progress_percentage == 67.8
        assert dto.current_step == "model_evaluation"
        assert dto.current_algorithm == "IsolationForest"
        assert dto.current_trial == 15
        assert dto.total_trials == 25
        assert dto.current_score == 0.89
        assert dto.best_score == 0.92
        assert dto.started_at == started_at
        assert dto.estimated_completion == estimated_completion
        assert dto.memory_usage_mb == 1024.5
        assert dto.cpu_usage_percent == 75.2
        assert dto.current_message == "Evaluating model performance"
        assert dto.recent_logs == ["Starting trial 15", "Model training completed"]
        assert dto.warnings == ["High CPU usage detected"]

    def test_to_dict_and_from_dict(self):
        """Test roundtrip conversion."""
        started_at = datetime.now()
        estimated_completion = datetime.now()

        original = TrainingStatusDTO(
            training_id="training_status_conversion",
            status="completed",
            progress_percentage=100.0,
            current_step="finished",
            current_algorithm="OneClassSVM",
            current_trial=10,
            total_trials=10,
            current_score=0.91,
            best_score=0.93,
            started_at=started_at,
            estimated_completion=estimated_completion,
            memory_usage_mb=512.0,
            cpu_usage_percent=25.0,
            current_message="Training completed successfully",
            recent_logs=["Training finished", "Model saved"],
            warnings=[],
        )

        data = original.to_dict()
        restored = TrainingStatusDTO.from_dict(data)

        assert restored.training_id == original.training_id
        assert restored.status == original.status
        assert restored.progress_percentage == original.progress_percentage
        assert restored.current_step == original.current_step
        assert restored.current_algorithm == original.current_algorithm
        assert restored.current_trial == original.current_trial
        assert restored.total_trials == original.total_trials
        assert restored.current_score == original.current_score
        assert restored.best_score == original.best_score
        assert restored.started_at == original.started_at
        assert restored.estimated_completion == original.estimated_completion
        assert restored.memory_usage_mb == original.memory_usage_mb
        assert restored.cpu_usage_percent == original.cpu_usage_percent
        assert restored.current_message == original.current_message
        assert restored.recent_logs == original.recent_logs
        assert restored.warnings == original.warnings


class TestTrainingDTOIntegration:
    """Test integration scenarios for training DTOs."""

    def test_complete_training_workflow(self):
        """Test complete training workflow with all DTOs."""
        detector_id = uuid4()
        requested_at = datetime.now()
        started_at = datetime.now()
        completed_at = datetime.now()

        # Create training configuration
        config = TrainingConfigDTO(
            experiment_name="integration_test",
            description="Complete workflow test",
            tags=["integration", "test"],
            automl_config=AutoMLConfigDTO(
                max_algorithms=3,
                enable_ensemble=True,
            ),
            validation_config=ValidationConfigDTO(
                validation_strategy="cross_validation",
                cv_folds=5,
            ),
            resource_constraints=ResourceConstraintsDTO(
                max_training_time_seconds=3600,
                n_jobs=4,
            ),
            priority=TrainingPriorityLevel.HIGH,
        )

        # Create training request
        request = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id="dataset_integration",
            config=config,
            trigger_type=TrainingTrigger.API_REQUEST,
            trigger_metadata={"api_version": "v1", "user_id": "user123"},
            requested_by="integration_test",
            requested_at=requested_at,
        )

        # Create training status updates
        status_running = TrainingStatusDTO(
            training_id="training_integration_123",
            status="running",
            progress_percentage=50.0,
            current_step="hyperparameter_tuning",
            current_algorithm="IsolationForest",
            current_trial=5,
            total_trials=10,
            current_score=0.85,
            best_score=0.87,
            started_at=started_at,
            memory_usage_mb=1024.0,
            cpu_usage_percent=80.0,
            current_message="Tuning hyperparameters",
            recent_logs=["Starting hyperparameter tuning"],
        )

        status_completed = TrainingStatusDTO(
            training_id="training_integration_123",
            status="completed",
            progress_percentage=100.0,
            current_step="finished",
            current_algorithm="IsolationForest",
            current_trial=10,
            total_trials=10,
            current_score=0.92,
            best_score=0.92,
            started_at=started_at,
            estimated_completion=completed_at,
            memory_usage_mb=512.0,
            cpu_usage_percent=10.0,
            current_message="Training completed successfully",
            recent_logs=["Training finished", "Model saved"],
        )

        # Create training result
        best_metrics = ModelMetricsDTO(
            accuracy=0.95,
            precision=0.93,
            recall=0.91,
            f1_score=0.92,
            roc_auc=0.96,
            training_time_seconds=1800.0,
            inference_time_ms=5.2,
            feature_importance={"feature1": 0.45, "feature2": 0.35, "feature3": 0.20},
        )

        result = TrainingResultDTO(
            training_id="training_integration_123",
            detector_id=detector_id,
            dataset_id="dataset_integration",
            experiment_name="integration_test",
            trigger_type=TrainingTrigger.API_REQUEST,
            status="completed",
            best_algorithm="IsolationForest",
            best_hyperparameters={"n_estimators": 100, "contamination": 0.05},
            best_metrics=best_metrics,
            model_id="model_integration_123",
            model_version="v1.0.0",
            model_path="/models/model_integration_123_v1.0.0.pkl",
            total_trials=10,
            successful_trials=10,
            training_duration_seconds=1800.0,
            optimization_history=[
                {"trial": 1, "algorithm": "IsolationForest", "score": 0.85},
                {"trial": 2, "algorithm": "OneClassSVM", "score": 0.88},
                {"trial": 3, "algorithm": "LocalOutlierFactor", "score": 0.87},
            ],
            performance_improvement=0.07,
            peak_memory_mb=1024.0,
            total_cpu_hours=0.5,
            started_at=started_at,
            completed_at=completed_at,
            warnings=["Memory usage peaked at 1GB"],
            logs=["Training started", "Hyperparameter tuning completed", "Model saved"],
        )

        # Verify integration consistency
        assert request.detector_id == result.detector_id
        assert request.dataset_id == result.dataset_id
        assert request.config.experiment_name == result.experiment_name
        assert request.trigger_type == result.trigger_type
        assert status_running.training_id == result.training_id
        assert status_completed.training_id == result.training_id
        assert status_completed.best_score == result.best_metrics.f1_score
        assert result.status == "completed"
        assert result.best_algorithm == "IsolationForest"
        assert result.total_trials == 10
        assert result.successful_trials == 10

    def test_serialization_roundtrip(self):
        """Test complete serialization roundtrip for all DTOs."""
        detector_id = uuid4()
        created_at = datetime.now()

        # Create complex configuration
        config = TrainingConfigDTO(
            experiment_name="serialization_test",
            description="Test serialization",
            tags=["serialization", "test"],
            dataset_id="dataset_serialization",
            detector_id=detector_id,
            target_algorithms=["IsolationForest", "OneClassSVM"],
            automl_config=AutoMLConfigDTO(
                enable_automl=True,
                optimization_objective="f1_score",
                max_algorithms=2,
                enable_ensemble=True,
                algorithm_whitelist=["IsolationForest", "OneClassSVM"],
            ),
            validation_config=ValidationConfigDTO(
                validation_strategy="cross_validation",
                cv_folds=3,
                enable_early_stopping=True,
            ),
            resource_constraints=ResourceConstraintsDTO(
                max_training_time_seconds=1800,
                max_memory_mb=2048,
                n_jobs=2,
            ),
            notification_config=NotificationConfigDTO(
                enable_notifications=True,
                notification_channels=["email"],
                email_recipients=["test@example.com"],
            ),
            created_by="test_user",
            created_at=created_at,
            priority=TrainingPriorityLevel.HIGH,
        )

        # Test serialization roundtrip
        data = config.to_dict()
        restored_config = TrainingConfigDTO.from_dict(data)

        assert restored_config.experiment_name == config.experiment_name
        assert restored_config.description == config.description
        assert restored_config.tags == config.tags
        assert restored_config.dataset_id == config.dataset_id
        assert restored_config.detector_id == config.detector_id
        assert restored_config.target_algorithms == config.target_algorithms
        assert (
            restored_config.automl_config.enable_automl
            == config.automl_config.enable_automl
        )
        assert (
            restored_config.validation_config.cv_folds
            == config.validation_config.cv_folds
        )
        assert (
            restored_config.resource_constraints.n_jobs
            == config.resource_constraints.n_jobs
        )
        assert (
            restored_config.notification_config.email_recipients
            == config.notification_config.email_recipients
        )
        assert restored_config.created_by == config.created_by
        assert restored_config.created_at == config.created_at
        assert restored_config.priority == config.priority

    def test_automl_training_scenario(self):
        """Test AutoML training scenario with comprehensive configuration."""
        detector_id = uuid4()

        # Create AutoML-focused configuration
        automl_config = AutoMLConfigDTO(
            enable_automl=True,
            optimization_objective="roc_auc",
            max_algorithms=5,
            enable_ensemble=True,
            ensemble_method="stacking",
            algorithm_whitelist=[
                "IsolationForest",
                "OneClassSVM",
                "LocalOutlierFactor",
            ],
            model_selection_strategy="weighted_score",
            metric_weights={"precision": 0.4, "recall": 0.3, "f1_score": 0.3},
        )

        validation_config = ValidationConfigDTO(
            validation_strategy="cross_validation",
            cv_folds=5,
            cv_strategy="stratified",
            enable_early_stopping=True,
            early_stopping_patience=5,
            early_stopping_metric="roc_auc",
        )

        resource_constraints = ResourceConstraintsDTO(
            max_training_time_seconds=7200,
            max_memory_mb=8192,
            n_jobs=-1,
            max_concurrent_trials=3,
        )

        config = TrainingConfigDTO(
            experiment_name="automl_fraud_detection",
            description="AutoML-based fraud detection training",
            tags=["automl", "fraud", "production"],
            automl_config=automl_config,
            validation_config=validation_config,
            resource_constraints=resource_constraints,
            priority=TrainingPriorityLevel.HIGH,
        )

        # Create training request
        request = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id="fraud_dataset_v2",
            config=config,
            trigger_type=TrainingTrigger.SCHEDULED,
            trigger_metadata={"schedule_name": "daily_retrain", "cron": "0 2 * * *"},
            requested_by="automl_scheduler",
        )

        # Create metrics for multiple algorithms
        isolation_forest_metrics = ModelMetricsDTO(
            accuracy=0.92,
            precision=0.89,
            recall=0.94,
            f1_score=0.91,
            roc_auc=0.95,
            training_time_seconds=450.0,
            inference_time_ms=3.2,
        )

        one_class_svm_metrics = ModelMetricsDTO(
            accuracy=0.88,
            precision=0.85,
            recall=0.91,
            f1_score=0.88,
            roc_auc=0.92,
            training_time_seconds=1200.0,
            inference_time_ms=8.5,
        )

        ensemble_metrics = ModelMetricsDTO(
            accuracy=0.95,
            precision=0.93,
            recall=0.96,
            f1_score=0.94,
            roc_auc=0.97,
            training_time_seconds=1800.0,
            inference_time_ms=12.0,
        )

        # Create training result
        result = TrainingResultDTO(
            training_id="automl_training_456",
            detector_id=detector_id,
            dataset_id="fraud_dataset_v2",
            experiment_name="automl_fraud_detection",
            trigger_type=TrainingTrigger.SCHEDULED,
            status="completed",
            best_algorithm="EnsembleModel",
            best_hyperparameters={
                "base_algorithms": ["IsolationForest", "OneClassSVM"],
                "ensemble_method": "stacking",
                "meta_learner": "LogisticRegression",
            },
            best_metrics=ensemble_metrics,
            model_id="ensemble_model_456",
            model_version="v2.1.0",
            model_path="/models/ensemble_model_456_v2.1.0.pkl",
            total_trials=15,
            successful_trials=13,
            training_duration_seconds=3600.0,
            optimization_history=[
                {
                    "trial": 1,
                    "algorithm": "IsolationForest",
                    "score": 0.91,
                    "metrics": isolation_forest_metrics.to_dict(),
                },
                {
                    "trial": 2,
                    "algorithm": "OneClassSVM",
                    "score": 0.88,
                    "metrics": one_class_svm_metrics.to_dict(),
                },
                {
                    "trial": 3,
                    "algorithm": "EnsembleModel",
                    "score": 0.94,
                    "metrics": ensemble_metrics.to_dict(),
                },
            ],
            performance_improvement=0.12,
            peak_memory_mb=6144.0,
            total_cpu_hours=2.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            warnings=["High training time for OneClassSVM"],
            logs=[
                "AutoML training started",
                "Testing IsolationForest",
                "Testing OneClassSVM",
                "Creating ensemble model",
                "Ensemble model achieved best performance",
                "Training completed successfully",
            ],
        )

        # Verify AutoML scenario
        assert request.config.automl_config.enable_automl is True
        assert request.config.automl_config.max_algorithms == 5
        assert request.config.automl_config.enable_ensemble is True
        assert result.best_algorithm == "EnsembleModel"
        assert result.total_trials == 15
        assert result.successful_trials == 13
        assert result.best_metrics.roc_auc == 0.97
        assert len(result.optimization_history) == 3
        assert result.performance_improvement == 0.12
