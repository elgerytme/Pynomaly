"""Tests for Configuration DTOs."""

from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from pynomaly.application.dto.configuration_dto import (
    AlgorithmConfigDTO,
    AlgorithmConfigurationDTO,
    ConfigurationExportRequestDTO,
    ConfigurationLevel,
    ConfigurationMetadataDTO,
    ConfigurationResponseDTO,
    ConfigurationSearchRequestDTO,
    ConfigurationSource,
    ConfigurationStatus,
    ConfigurationValidationResultDTO,
    DatasetConfigDTO,
    EnvironmentConfigDTO,
    EvaluationConfigDTO,
    ExperimentConfigurationDTO,
    ExportFormat,
    PerformanceResultsDTO,
    PreprocessingConfigDTO,
    RequestConfigurationDTO,
    ResponseConfigurationDTO,
    WebAPIContextDTO,
    create_basic_configuration,
    merge_configurations,
    validate_configuration_compatibility,
)


class TestConfigurationEnums:
    """Test suite for configuration enums."""

    def test_configuration_source_enum(self):
        """Test ConfigurationSource enum values."""
        assert ConfigurationSource.AUTOML == "automl"
        assert ConfigurationSource.AUTONOMOUS == "autonomous"
        assert ConfigurationSource.CLI == "cli"
        assert ConfigurationSource.WEB_API == "web_api"
        assert ConfigurationSource.WEB_UI == "web_ui"
        assert ConfigurationSource.TEST == "test"
        assert ConfigurationSource.MANUAL == "manual"
        assert ConfigurationSource.TEMPLATE == "template"

    def test_configuration_status_enum(self):
        """Test ConfigurationStatus enum values."""
        assert ConfigurationStatus.DRAFT == "draft"
        assert ConfigurationStatus.ACTIVE == "active"
        assert ConfigurationStatus.COMPLETED == "completed"
        assert ConfigurationStatus.FAILED == "failed"
        assert ConfigurationStatus.ARCHIVED == "archived"
        assert ConfigurationStatus.DEPRECATED == "deprecated"

    def test_export_format_enum(self):
        """Test ExportFormat enum values."""
        assert ExportFormat.YAML == "yaml"
        assert ExportFormat.JSON == "json"
        assert ExportFormat.PYTHON == "python"
        assert ExportFormat.NOTEBOOK == "notebook"
        assert ExportFormat.DOCKER == "docker"
        assert ExportFormat.CONFIG_INI == "config_ini"

    def test_configuration_level_enum(self):
        """Test ConfigurationLevel enum values."""
        assert ConfigurationLevel.BASIC == "basic"
        assert ConfigurationLevel.INTERMEDIATE == "intermediate"
        assert ConfigurationLevel.ADVANCED == "advanced"
        assert ConfigurationLevel.EXPERT == "expert"


class TestDatasetConfigDTO:
    """Test suite for DatasetConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid dataset configuration."""
        dto = DatasetConfigDTO(
            dataset_path="/path/to/dataset.csv",
            dataset_name="test_dataset",
            dataset_id=uuid4(),
            file_format="csv",
            delimiter=",",
            encoding="utf-8",
            header_row=0,
            feature_columns=["feature1", "feature2"],
            target_column="target",
            exclude_columns=["id"],
            datetime_columns=["timestamp"],
            expected_shape=(1000, 10),
            required_columns=["feature1", "feature2"],
            data_types={"feature1": "float", "feature2": "int"},
            sample_size=5000,
            sampling_method="stratified",
            random_seed=42,
        )

        assert dto.dataset_path == "/path/to/dataset.csv"
        assert dto.dataset_name == "test_dataset"
        assert isinstance(dto.dataset_id, UUID)
        assert dto.file_format == "csv"
        assert dto.delimiter == ","
        assert dto.encoding == "utf-8"
        assert dto.header_row == 0
        assert dto.feature_columns == ["feature1", "feature2"]
        assert dto.target_column == "target"
        assert dto.exclude_columns == ["id"]
        assert dto.datetime_columns == ["timestamp"]
        assert dto.expected_shape == (1000, 10)
        assert dto.required_columns == ["feature1", "feature2"]
        assert dto.data_types == {"feature1": "float", "feature2": "int"}
        assert dto.sample_size == 5000
        assert dto.sampling_method == "stratified"
        assert dto.random_seed == 42

    def test_default_values(self):
        """Test default values."""
        dto = DatasetConfigDTO()

        assert dto.dataset_path is None
        assert dto.dataset_name is None
        assert dto.dataset_id is None
        assert dto.file_format is None
        assert dto.delimiter == ","
        assert dto.encoding == "utf-8"
        assert dto.header_row == 0
        assert dto.feature_columns is None
        assert dto.target_column is None
        assert dto.exclude_columns is None
        assert dto.datetime_columns is None
        assert dto.expected_shape is None
        assert dto.required_columns is None
        assert dto.data_types is None
        assert dto.sample_size is None
        assert dto.sampling_method == "random"
        assert dto.random_seed == 42

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DatasetConfigDTO(unknown_field="value")


class TestAlgorithmConfigDTO:
    """Test suite for AlgorithmConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid algorithm configuration."""
        dto = AlgorithmConfigDTO(
            algorithm_name="IsolationForest",
            algorithm_family="isolation",
            algorithm_version="1.0",
            hyperparameters={"n_estimators": 100, "contamination": 0.1},
            parameter_source="optimized",
            optimization_method="bayesian",
            contamination=0.1,
            random_state=42,
            n_jobs=4,
            max_training_time=300.0,
            max_memory_usage=1024.0,
            is_ensemble=True,
            ensemble_method="voting",
            base_algorithms=["IsolationForest", "OneClassSVM"],
        )

        assert dto.algorithm_name == "IsolationForest"
        assert dto.algorithm_family == "isolation"
        assert dto.algorithm_version == "1.0"
        assert dto.hyperparameters == {"n_estimators": 100, "contamination": 0.1}
        assert dto.parameter_source == "optimized"
        assert dto.optimization_method == "bayesian"
        assert dto.contamination == 0.1
        assert dto.random_state == 42
        assert dto.n_jobs == 4
        assert dto.max_training_time == 300.0
        assert dto.max_memory_usage == 1024.0
        assert dto.is_ensemble is True
        assert dto.ensemble_method == "voting"
        assert dto.base_algorithms == ["IsolationForest", "OneClassSVM"]

    def test_default_values(self):
        """Test default values."""
        dto = AlgorithmConfigDTO(algorithm_name="IsolationForest")

        assert dto.algorithm_name == "IsolationForest"
        assert dto.algorithm_family is None
        assert dto.algorithm_version is None
        assert dto.hyperparameters == {}
        assert dto.parameter_source == "default"
        assert dto.optimization_method is None
        assert dto.contamination == 0.1
        assert dto.random_state == 42
        assert dto.n_jobs == 1
        assert dto.max_training_time is None
        assert dto.max_memory_usage is None
        assert dto.is_ensemble is False
        assert dto.ensemble_method is None
        assert dto.base_algorithms is None

    def test_contamination_validation(self):
        """Test contamination validation."""
        # Valid range
        dto = AlgorithmConfigDTO(algorithm_name="IsolationForest", contamination=0.2)
        assert dto.contamination == 0.2

        # Invalid: negative
        with pytest.raises(ValidationError):
            AlgorithmConfigDTO(algorithm_name="IsolationForest", contamination=-0.1)

        # Invalid: greater than 0.5
        with pytest.raises(ValidationError):
            AlgorithmConfigDTO(algorithm_name="IsolationForest", contamination=0.6)

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AlgorithmConfigDTO()


class TestPreprocessingConfigDTO:
    """Test suite for PreprocessingConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid preprocessing configuration."""
        dto = PreprocessingConfigDTO(
            missing_value_strategy="median",
            missing_value_threshold=0.3,
            outlier_detection_method="isolation_forest",
            outlier_threshold=2.0,
            outlier_treatment="clip",
            scaling_method="robust",
            scaling_robust=True,
            feature_selection_method="variance",
            max_features=100,
            feature_importance_threshold=0.01,
            categorical_encoding="onehot",
            high_cardinality_threshold=20,
            apply_pca=True,
            pca_components=50,
            remove_duplicates=False,
            normalize_data=True,
        )

        assert dto.missing_value_strategy == "median"
        assert dto.missing_value_threshold == 0.3
        assert dto.outlier_detection_method == "isolation_forest"
        assert dto.outlier_threshold == 2.0
        assert dto.outlier_treatment == "clip"
        assert dto.scaling_method == "robust"
        assert dto.scaling_robust is True
        assert dto.feature_selection_method == "variance"
        assert dto.max_features == 100
        assert dto.feature_importance_threshold == 0.01
        assert dto.categorical_encoding == "onehot"
        assert dto.high_cardinality_threshold == 20
        assert dto.apply_pca is True
        assert dto.pca_components == 50
        assert dto.remove_duplicates is False
        assert dto.normalize_data is True

    def test_default_values(self):
        """Test default values."""
        dto = PreprocessingConfigDTO()

        assert dto.missing_value_strategy == "mean"
        assert dto.missing_value_threshold == 0.5
        assert dto.outlier_detection_method == "iqr"
        assert dto.outlier_threshold == 1.5
        assert dto.outlier_treatment == "remove"
        assert dto.scaling_method == "standard"
        assert dto.scaling_robust is False
        assert dto.feature_selection_method is None
        assert dto.max_features is None
        assert dto.feature_importance_threshold is None
        assert dto.categorical_encoding == "label"
        assert dto.high_cardinality_threshold == 10
        assert dto.apply_pca is False
        assert dto.pca_components is None
        assert dto.remove_duplicates is True
        assert dto.normalize_data is False


class TestEvaluationConfigDTO:
    """Test suite for EvaluationConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid evaluation configuration."""
        dto = EvaluationConfigDTO(
            primary_metric="f1_score",
            secondary_metrics=["precision", "recall"],
            cv_folds=10,
            cv_strategy="kfold",
            cv_random_state=123,
            test_size=0.3,
            validation_size=0.25,
            min_accuracy=0.8,
            max_false_positive_rate=0.1,
            min_recall=0.7,
            calculate_feature_importance=False,
            generate_plots=False,
            save_predictions=True,
            detailed_results=False,
        )

        assert dto.primary_metric == "f1_score"
        assert dto.secondary_metrics == ["precision", "recall"]
        assert dto.cv_folds == 10
        assert dto.cv_strategy == "kfold"
        assert dto.cv_random_state == 123
        assert dto.test_size == 0.3
        assert dto.validation_size == 0.25
        assert dto.min_accuracy == 0.8
        assert dto.max_false_positive_rate == 0.1
        assert dto.min_recall == 0.7
        assert dto.calculate_feature_importance is False
        assert dto.generate_plots is False
        assert dto.save_predictions is True
        assert dto.detailed_results is False

    def test_default_values(self):
        """Test default values."""
        dto = EvaluationConfigDTO()

        assert dto.primary_metric == "roc_auc"
        assert dto.secondary_metrics == []
        assert dto.cv_folds == 5
        assert dto.cv_strategy == "stratified"
        assert dto.cv_random_state == 42
        assert dto.test_size == 0.2
        assert dto.validation_size == 0.2
        assert dto.min_accuracy is None
        assert dto.max_false_positive_rate is None
        assert dto.min_recall is None
        assert dto.calculate_feature_importance is True
        assert dto.generate_plots is True
        assert dto.save_predictions is False
        assert dto.detailed_results is True

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Valid ranges
        dto = EvaluationConfigDTO(cv_folds=3, test_size=0.1, validation_size=0.1)
        assert dto.cv_folds == 3
        assert dto.test_size == 0.1
        assert dto.validation_size == 0.1

        # Invalid: cv_folds too small
        with pytest.raises(ValidationError):
            EvaluationConfigDTO(cv_folds=1)

        # Invalid: cv_folds too large
        with pytest.raises(ValidationError):
            EvaluationConfigDTO(cv_folds=25)

        # Invalid: test_size too small
        with pytest.raises(ValidationError):
            EvaluationConfigDTO(test_size=0.05)

        # Invalid: validation_size too large
        with pytest.raises(ValidationError):
            EvaluationConfigDTO(validation_size=0.6)


class TestEnvironmentConfigDTO:
    """Test suite for EnvironmentConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid environment configuration."""
        dto = EnvironmentConfigDTO(
            python_version="3.9.0",
            dependencies={"numpy": "1.21.0", "pandas": "1.3.0"},
            cpu_count=8,
            memory_gb=16.0,
            gpu_available=True,
            gpu_model="NVIDIA RTX 3080",
            max_execution_time=7200.0,
            memory_limit_mb=8192.0,
            disk_space_gb=100.0,
            operating_system="Linux",
            architecture="x86_64",
            container_runtime="docker",
        )

        assert dto.python_version == "3.9.0"
        assert dto.dependencies == {"numpy": "1.21.0", "pandas": "1.3.0"}
        assert dto.cpu_count == 8
        assert dto.memory_gb == 16.0
        assert dto.gpu_available is True
        assert dto.gpu_model == "NVIDIA RTX 3080"
        assert dto.max_execution_time == 7200.0
        assert dto.memory_limit_mb == 8192.0
        assert dto.disk_space_gb == 100.0
        assert dto.operating_system == "Linux"
        assert dto.architecture == "x86_64"
        assert dto.container_runtime == "docker"

    def test_default_values(self):
        """Test default values."""
        dto = EnvironmentConfigDTO()

        assert dto.python_version is None
        assert dto.dependencies is None
        assert dto.cpu_count is None
        assert dto.memory_gb is None
        assert dto.gpu_available is False
        assert dto.gpu_model is None
        assert dto.max_execution_time is None
        assert dto.memory_limit_mb is None
        assert dto.disk_space_gb is None
        assert dto.operating_system is None
        assert dto.architecture is None
        assert dto.container_runtime is None


class TestConfigurationMetadataDTO:
    """Test suite for ConfigurationMetadataDTO."""

    def test_valid_creation(self):
        """Test creating a valid configuration metadata."""
        parent_id = uuid4()
        derived_from = [uuid4(), uuid4()]

        dto = ConfigurationMetadataDTO(
            created_by="test_user",
            source=ConfigurationSource.AUTOML,
            source_version="1.0.0",
            tags=["tag1", "tag2"],
            category="anomaly_detection",
            complexity_level=ConfigurationLevel.ADVANCED,
            version="2.0.0",
            parent_id=parent_id,
            derived_from=derived_from,
            usage_count=5,
            last_used=datetime.now(),
            success_rate=0.85,
            description="Test configuration",
            notes="Test notes",
            documentation_url="https://docs.example.com",
        )

        assert dto.created_by == "test_user"
        assert dto.source == ConfigurationSource.AUTOML
        assert dto.source_version == "1.0.0"
        assert dto.tags == ["tag1", "tag2"]
        assert dto.category == "anomaly_detection"
        assert dto.complexity_level == ConfigurationLevel.ADVANCED
        assert dto.version == "2.0.0"
        assert dto.parent_id == parent_id
        assert dto.derived_from == derived_from
        assert dto.usage_count == 5
        assert isinstance(dto.last_used, datetime)
        assert dto.success_rate == 0.85
        assert dto.description == "Test configuration"
        assert dto.notes == "Test notes"
        assert dto.documentation_url == "https://docs.example.com"

    def test_default_values(self):
        """Test default values."""
        dto = ConfigurationMetadataDTO(source=ConfigurationSource.MANUAL)

        assert dto.created_by is None
        assert isinstance(dto.created_at, datetime)
        assert dto.source == ConfigurationSource.MANUAL
        assert dto.source_version is None
        assert dto.tags == []
        assert dto.category is None
        assert dto.complexity_level == ConfigurationLevel.BASIC
        assert dto.version == "1.0.0"
        assert dto.parent_id is None
        assert dto.derived_from is None
        assert dto.usage_count == 0
        assert dto.last_used is None
        assert dto.success_rate is None
        assert dto.description is None
        assert dto.notes is None
        assert dto.documentation_url is None

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ConfigurationMetadataDTO()


class TestPerformanceResultsDTO:
    """Test suite for PerformanceResultsDTO."""

    def test_valid_creation(self):
        """Test creating a valid performance results."""
        dto = PerformanceResultsDTO(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            roc_auc=0.94,
            training_time_seconds=120.5,
            prediction_time_ms=15.2,
            memory_usage_mb=512.0,
            model_size_mb=10.5,
            cv_scores=[0.93, 0.95, 0.91, 0.94, 0.92],
            cv_mean=0.93,
            cv_std=0.015,
            confusion_matrix=[[85, 5], [10, 90]],
            feature_importance={"feature1": 0.6, "feature2": 0.4},
            anomaly_scores=[0.1, 0.3, 0.8, 0.2],
            cpu_usage_percent=75.5,
            gpu_usage_percent=80.0,
            disk_io_mb=50.0,
            stability_score=0.85,
            robustness_score=0.78,
            interpretability_score=0.65,
        )

        assert dto.accuracy == 0.95
        assert dto.precision == 0.92
        assert dto.recall == 0.88
        assert dto.f1_score == 0.90
        assert dto.roc_auc == 0.94
        assert dto.training_time_seconds == 120.5
        assert dto.prediction_time_ms == 15.2
        assert dto.memory_usage_mb == 512.0
        assert dto.model_size_mb == 10.5
        assert dto.cv_scores == [0.93, 0.95, 0.91, 0.94, 0.92]
        assert dto.cv_mean == 0.93
        assert dto.cv_std == 0.015
        assert dto.confusion_matrix == [[85, 5], [10, 90]]
        assert dto.feature_importance == {"feature1": 0.6, "feature2": 0.4}
        assert dto.anomaly_scores == [0.1, 0.3, 0.8, 0.2]
        assert dto.cpu_usage_percent == 75.5
        assert dto.gpu_usage_percent == 80.0
        assert dto.disk_io_mb == 50.0
        assert dto.stability_score == 0.85
        assert dto.robustness_score == 0.78
        assert dto.interpretability_score == 0.65

    def test_default_values(self):
        """Test default values."""
        dto = PerformanceResultsDTO()

        assert dto.accuracy is None
        assert dto.precision is None
        assert dto.recall is None
        assert dto.f1_score is None
        assert dto.roc_auc is None
        assert dto.training_time_seconds is None
        assert dto.prediction_time_ms is None
        assert dto.memory_usage_mb is None
        assert dto.model_size_mb is None
        assert dto.cv_scores is None
        assert dto.cv_mean is None
        assert dto.cv_std is None
        assert dto.confusion_matrix is None
        assert dto.feature_importance is None
        assert dto.anomaly_scores is None
        assert dto.cpu_usage_percent is None
        assert dto.gpu_usage_percent is None
        assert dto.disk_io_mb is None
        assert dto.stability_score is None
        assert dto.robustness_score is None
        assert dto.interpretability_score is None


class TestExperimentConfigurationDTO:
    """Test suite for ExperimentConfigurationDTO."""

    def test_valid_creation(self):
        """Test creating a valid experiment configuration."""
        dataset_config = DatasetConfigDTO(
            dataset_path="/path/to/dataset.csv", dataset_name="test_dataset"
        )

        algorithm_config = AlgorithmConfigDTO(
            algorithm_name="IsolationForest", hyperparameters={"n_estimators": 100}
        )

        evaluation_config = EvaluationConfigDTO(primary_metric="roc_auc", cv_folds=5)

        metadata = ConfigurationMetadataDTO(
            source=ConfigurationSource.MANUAL, created_by="test_user"
        )

        dto = ExperimentConfigurationDTO(
            name="test_experiment",
            status=ConfigurationStatus.ACTIVE,
            dataset_config=dataset_config,
            algorithm_config=algorithm_config,
            evaluation_config=evaluation_config,
            metadata=metadata,
            export_formats=[ExportFormat.YAML, ExportFormat.JSON],
            export_metadata={"format_version": "1.0"},
            is_valid=True,
            validation_errors=[],
            validation_warnings=["Warning message"],
            execution_count=3,
            average_execution_time=150.0,
        )

        assert isinstance(dto.id, UUID)
        assert dto.name == "test_experiment"
        assert dto.status == ConfigurationStatus.ACTIVE
        assert dto.dataset_config == dataset_config
        assert dto.algorithm_config == algorithm_config
        assert dto.evaluation_config == evaluation_config
        assert dto.metadata == metadata
        assert dto.export_formats == [ExportFormat.YAML, ExportFormat.JSON]
        assert dto.export_metadata == {"format_version": "1.0"}
        assert dto.is_valid is True
        assert dto.validation_errors == []
        assert dto.validation_warnings == ["Warning message"]
        assert dto.execution_count == 3
        assert dto.average_execution_time == 150.0

    def test_default_values(self):
        """Test default values."""
        dataset_config = DatasetConfigDTO()
        algorithm_config = AlgorithmConfigDTO(algorithm_name="IsolationForest")
        evaluation_config = EvaluationConfigDTO()
        metadata = ConfigurationMetadataDTO(source=ConfigurationSource.MANUAL)

        dto = ExperimentConfigurationDTO(
            name="test_experiment",
            dataset_config=dataset_config,
            algorithm_config=algorithm_config,
            evaluation_config=evaluation_config,
            metadata=metadata,
        )

        assert isinstance(dto.id, UUID)
        assert dto.status == ConfigurationStatus.DRAFT
        assert dto.preprocessing_config is None
        assert dto.environment_config is None
        assert dto.lineage is None
        assert dto.performance_results is None
        assert dto.export_formats == []
        assert dto.export_metadata == {}
        assert dto.is_valid is True
        assert dto.validation_errors == []
        assert dto.validation_warnings == []
        assert dto.last_executed is None
        assert dto.execution_count == 0
        assert dto.average_execution_time is None

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ExperimentConfigurationDTO(name="test")


class TestConfigurationSearchRequestDTO:
    """Test suite for ConfigurationSearchRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid configuration search request."""
        dto = ConfigurationSearchRequestDTO(
            query="anomaly detection",
            tags=["tag1", "tag2"],
            source=ConfigurationSource.AUTOML,
            algorithm="IsolationForest",
            created_after=datetime(2023, 1, 1),
            created_before=datetime(2023, 12, 31),
            min_accuracy=0.8,
            max_execution_time=300.0,
            offset=10,
            limit=20,
            sort_by="accuracy",
            sort_order="desc",
        )

        assert dto.query == "anomaly detection"
        assert dto.tags == ["tag1", "tag2"]
        assert dto.source == ConfigurationSource.AUTOML
        assert dto.algorithm == "IsolationForest"
        assert dto.created_after == datetime(2023, 1, 1)
        assert dto.created_before == datetime(2023, 12, 31)
        assert dto.min_accuracy == 0.8
        assert dto.max_execution_time == 300.0
        assert dto.offset == 10
        assert dto.limit == 20
        assert dto.sort_by == "accuracy"
        assert dto.sort_order == "desc"

    def test_default_values(self):
        """Test default values."""
        dto = ConfigurationSearchRequestDTO()

        assert dto.query is None
        assert dto.tags is None
        assert dto.source is None
        assert dto.algorithm is None
        assert dto.created_after is None
        assert dto.created_before is None
        assert dto.min_accuracy is None
        assert dto.max_execution_time is None
        assert dto.offset == 0
        assert dto.limit == 50
        assert dto.sort_by == "created_at"
        assert dto.sort_order == "desc"

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Valid ranges
        dto = ConfigurationSearchRequestDTO(offset=0, limit=1)
        assert dto.offset == 0
        assert dto.limit == 1

        # Invalid: offset negative
        with pytest.raises(ValidationError):
            ConfigurationSearchRequestDTO(offset=-1)

        # Invalid: limit too small
        with pytest.raises(ValidationError):
            ConfigurationSearchRequestDTO(limit=0)

        # Invalid: limit too large
        with pytest.raises(ValidationError):
            ConfigurationSearchRequestDTO(limit=1500)


class TestConfigurationResponseDTO:
    """Test suite for ConfigurationResponseDTO."""

    def test_valid_creation(self):
        """Test creating a valid configuration response."""
        dataset_config = DatasetConfigDTO()
        algorithm_config = AlgorithmConfigDTO(algorithm_name="IsolationForest")
        evaluation_config = EvaluationConfigDTO()
        metadata = ConfigurationMetadataDTO(source=ConfigurationSource.MANUAL)

        config = ExperimentConfigurationDTO(
            name="test_experiment",
            dataset_config=dataset_config,
            algorithm_config=algorithm_config,
            evaluation_config=evaluation_config,
            metadata=metadata,
        )

        dto = ConfigurationResponseDTO(
            success=True,
            message="Configuration retrieved successfully",
            configuration=config,
            configurations=[config],
            export_data="exported_data",
            export_files=["file1.yaml", "file2.json"],
            total_count=1,
            execution_time=0.5,
            errors=["Error message"],
            warnings=["Warning message"],
        )

        assert dto.success is True
        assert dto.message == "Configuration retrieved successfully"
        assert dto.configuration == config
        assert dto.configurations == [config]
        assert dto.export_data == "exported_data"
        assert dto.export_files == ["file1.yaml", "file2.json"]
        assert dto.total_count == 1
        assert dto.execution_time == 0.5
        assert dto.errors == ["Error message"]
        assert dto.warnings == ["Warning message"]

    def test_default_values(self):
        """Test default values."""
        dto = ConfigurationResponseDTO(success=True, message="Success")

        assert dto.configuration is None
        assert dto.configurations is None
        assert dto.export_data is None
        assert dto.export_files is None
        assert dto.total_count is None
        assert dto.execution_time is None
        assert dto.errors == []
        assert dto.warnings == []

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ConfigurationResponseDTO()


class TestConfigurationValidationResultDTO:
    """Test suite for ConfigurationValidationResultDTO."""

    def test_valid_creation(self):
        """Test creating a valid configuration validation result."""
        dto = ConfigurationValidationResultDTO(
            is_valid=False,
            validation_score=0.75,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            suggestions=["Suggestion 1"],
            dataset_validation={"valid": True},
            algorithm_validation={"valid": False},
            preprocessing_validation={"valid": True},
            compatibility_issues=["Issue 1"],
            missing_dependencies=["dependency1"],
            version_conflicts=["conflict1"],
            estimated_runtime=300.0,
            estimated_memory=1024.0,
            risk_assessment="medium",
        )

        assert dto.is_valid is False
        assert dto.validation_score == 0.75
        assert dto.errors == ["Error 1", "Error 2"]
        assert dto.warnings == ["Warning 1"]
        assert dto.suggestions == ["Suggestion 1"]
        assert dto.dataset_validation == {"valid": True}
        assert dto.algorithm_validation == {"valid": False}
        assert dto.preprocessing_validation == {"valid": True}
        assert dto.compatibility_issues == ["Issue 1"]
        assert dto.missing_dependencies == ["dependency1"]
        assert dto.version_conflicts == ["conflict1"]
        assert dto.estimated_runtime == 300.0
        assert dto.estimated_memory == 1024.0
        assert dto.risk_assessment == "medium"

    def test_default_values(self):
        """Test default values."""
        dto = ConfigurationValidationResultDTO(is_valid=True, validation_score=1.0)

        assert dto.errors == []
        assert dto.warnings == []
        assert dto.suggestions == []
        assert dto.dataset_validation == {}
        assert dto.algorithm_validation == {}
        assert dto.preprocessing_validation == {}
        assert dto.compatibility_issues == []
        assert dto.missing_dependencies == []
        assert dto.version_conflicts == []
        assert dto.estimated_runtime is None
        assert dto.estimated_memory is None
        assert dto.risk_assessment is None

    def test_validation_score_constraint(self):
        """Test validation score constraint."""
        # Valid range
        dto = ConfigurationValidationResultDTO(is_valid=True, validation_score=0.5)
        assert dto.validation_score == 0.5

        # Invalid: negative
        with pytest.raises(ValidationError):
            ConfigurationValidationResultDTO(is_valid=True, validation_score=-0.1)

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            ConfigurationValidationResultDTO(is_valid=True, validation_score=1.1)

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ConfigurationValidationResultDTO()


class TestDataclassesDTO:
    """Test suite for dataclass DTOs."""

    def test_request_configuration_dto(self):
        """Test RequestConfigurationDTO."""
        dto = RequestConfigurationDTO(
            method="POST",
            path="/api/v1/configurations",
            query_parameters={"param1": "value1"},
            headers={"Content-Type": "application/json"},
            body={"data": "test"},
            client_ip="192.168.1.1",
            user_agent="Test Agent",
            content_type="application/json",
            content_length="100",
        )

        assert dto.method == "POST"
        assert dto.path == "/api/v1/configurations"
        assert dto.query_parameters == {"param1": "value1"}
        assert dto.headers == {"Content-Type": "application/json"}
        assert dto.body == {"data": "test"}
        assert dto.client_ip == "192.168.1.1"
        assert dto.user_agent == "Test Agent"
        assert dto.content_type == "application/json"
        assert dto.content_length == "100"

    def test_response_configuration_dto(self):
        """Test ResponseConfigurationDTO."""
        dto = ResponseConfigurationDTO(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"result": "success"},
            processing_time_ms=150.5,
            content_type="application/json",
            content_length="200",
        )

        assert dto.status_code == 200
        assert dto.headers == {"Content-Type": "application/json"}
        assert dto.body == {"result": "success"}
        assert dto.processing_time_ms == 150.5
        assert dto.content_type == "application/json"
        assert dto.content_length == "200"

    def test_web_api_context_dto(self):
        """Test WebAPIContextDTO."""
        request_config = RequestConfigurationDTO(
            method="GET",
            path="/api/v1/test",
            query_parameters={},
            headers={},
            body=None,
            client_ip="127.0.0.1",
            user_agent=None,
            content_type=None,
            content_length=None,
        )

        response_config = ResponseConfigurationDTO(
            status_code=200,
            headers={},
            body=None,
            processing_time_ms=10.0,
            content_type=None,
            content_length=None,
        )

        dto = WebAPIContextDTO(
            request_config=request_config,
            response_config=response_config,
            endpoint="/api/v1/test",
            api_version="1.0",
            client_info={"browser": "Chrome"},
            session_id="session123",
        )

        assert dto.request_config == request_config
        assert dto.response_config == response_config
        assert dto.endpoint == "/api/v1/test"
        assert dto.api_version == "1.0"
        assert dto.client_info == {"browser": "Chrome"}
        assert dto.session_id == "session123"


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_create_basic_configuration(self):
        """Test create_basic_configuration function."""
        config = create_basic_configuration(
            name="test_config",
            dataset_path="/path/to/dataset.csv",
            algorithm_name="IsolationForest",
            source=ConfigurationSource.AUTOML,
        )

        assert config.name == "test_config"
        assert config.dataset_config.dataset_path == "/path/to/dataset.csv"
        assert config.algorithm_config.algorithm_name == "IsolationForest"
        assert config.metadata.source == ConfigurationSource.AUTOML
        assert isinstance(config.evaluation_config, EvaluationConfigDTO)

    def test_create_basic_configuration_default_source(self):
        """Test create_basic_configuration with default source."""
        config = create_basic_configuration(
            name="test_config",
            dataset_path="/path/to/dataset.csv",
            algorithm_name="IsolationForest",
        )

        assert config.metadata.source == ConfigurationSource.MANUAL

    def test_merge_configurations(self):
        """Test merge_configurations function."""
        base_config = create_basic_configuration(
            name="base_config",
            dataset_path="/path/to/base.csv",
            algorithm_name="IsolationForest",
        )

        override_config = create_basic_configuration(
            name="override_config",
            dataset_path="/path/to/override.csv",
            algorithm_name="OneClassSVM",
        )

        merged = merge_configurations(base_config, override_config)

        assert merged.name == "base_config"  # Base name preserved
        assert (
            merged.algorithm_config.algorithm_name == "OneClassSVM"
        )  # Override algorithm
        assert merged.metadata.parent_id == base_config.id
        assert merged.metadata.derived_from == [base_config.id, override_config.id]
        assert merged.metadata.version == "merged-1.0.0"

    def test_validate_configuration_compatibility(self):
        """Test validate_configuration_compatibility function."""
        config1 = create_basic_configuration(
            name="config1",
            dataset_path="/path/to/dataset1.csv",
            algorithm_name="IsolationForest",
        )

        config2 = create_basic_configuration(
            name="config2",
            dataset_path="/path/to/dataset2.csv",
            algorithm_name="OneClassSVM",
        )

        issues = validate_configuration_compatibility(config1, config2)

        assert "Different algorithms specified" in issues
        assert "Different datasets specified" in issues

    def test_validate_configuration_compatibility_compatible(self):
        """Test validate_configuration_compatibility with compatible configs."""
        config1 = create_basic_configuration(
            name="config1",
            dataset_path="/path/to/dataset.csv",
            algorithm_name="IsolationForest",
        )

        config2 = create_basic_configuration(
            name="config2",
            dataset_path="/path/to/dataset.csv",
            algorithm_name="IsolationForest",
        )

        issues = validate_configuration_compatibility(config1, config2)

        assert len(issues) == 0


class TestConfigurationDTOIntegration:
    """Test integration scenarios for configuration DTOs."""

    def test_complete_configuration_workflow(self):
        """Test complete configuration workflow."""
        # Create dataset configuration
        dataset_config = DatasetConfigDTO(
            dataset_path="/path/to/dataset.csv",
            dataset_name="test_dataset",
            file_format="csv",
            feature_columns=["feature1", "feature2"],
            target_column="target",
        )

        # Create algorithm configuration
        algorithm_config = AlgorithmConfigDTO(
            algorithm_name="IsolationForest",
            algorithm_family="isolation",
            hyperparameters={"n_estimators": 100, "contamination": 0.1},
            contamination=0.1,
        )

        # Create preprocessing configuration
        preprocessing_config = PreprocessingConfigDTO(
            missing_value_strategy="mean",
            scaling_method="standard",
            apply_pca=True,
            pca_components=10,
        )

        # Create evaluation configuration
        evaluation_config = EvaluationConfigDTO(
            primary_metric="roc_auc", cv_folds=5, test_size=0.2
        )

        # Create environment configuration
        environment_config = EnvironmentConfigDTO(
            python_version="3.9.0", cpu_count=4, memory_gb=8.0
        )

        # Create metadata
        metadata = ConfigurationMetadataDTO(
            source=ConfigurationSource.AUTOML,
            created_by="test_user",
            tags=["test", "anomaly_detection"],
            complexity_level=ConfigurationLevel.INTERMEDIATE,
        )

        # Create performance results
        performance_results = PerformanceResultsDTO(
            accuracy=0.95,
            roc_auc=0.94,
            training_time_seconds=120.0,
            cv_scores=[0.93, 0.95, 0.91, 0.94, 0.92],
        )

        # Create complete configuration
        config = ExperimentConfigurationDTO(
            name="complete_test_config",
            status=ConfigurationStatus.COMPLETED,
            dataset_config=dataset_config,
            algorithm_config=algorithm_config,
            preprocessing_config=preprocessing_config,
            evaluation_config=evaluation_config,
            environment_config=environment_config,
            metadata=metadata,
            performance_results=performance_results,
            export_formats=[ExportFormat.YAML, ExportFormat.JSON],
            is_valid=True,
            execution_count=1,
            average_execution_time=120.0,
        )

        # Verify all components are properly integrated
        assert config.name == "complete_test_config"
        assert config.status == ConfigurationStatus.COMPLETED
        assert config.dataset_config.dataset_name == "test_dataset"
        assert config.algorithm_config.algorithm_name == "IsolationForest"
        assert config.preprocessing_config.scaling_method == "standard"
        assert config.evaluation_config.primary_metric == "roc_auc"
        assert config.environment_config.python_version == "3.9.0"
        assert config.metadata.source == ConfigurationSource.AUTOML
        assert config.performance_results.accuracy == 0.95
        assert ExportFormat.YAML in config.export_formats
        assert config.is_valid is True

    def test_configuration_search_and_response(self):
        """Test configuration search and response workflow."""
        # Create search request
        search_request = ConfigurationSearchRequestDTO(
            query="anomaly detection",
            algorithm="IsolationForest",
            min_accuracy=0.8,
            limit=10,
            sort_by="accuracy",
            sort_order="desc",
        )

        # Create sample configuration
        config = create_basic_configuration(
            name="search_result_config",
            dataset_path="/path/to/dataset.csv",
            algorithm_name="IsolationForest",
        )

        # Create response
        response = ConfigurationResponseDTO(
            success=True,
            message="Search completed successfully",
            configurations=[config],
            total_count=1,
            execution_time=0.5,
        )

        # Verify search workflow
        assert search_request.query == "anomaly detection"
        assert search_request.algorithm == "IsolationForest"
        assert search_request.min_accuracy == 0.8
        assert response.success is True
        assert len(response.configurations) == 1
        assert response.configurations[0].name == "search_result_config"

    def test_configuration_validation_workflow(self):
        """Test configuration validation workflow."""
        # Create configuration with potential issues
        config = create_basic_configuration(
            name="validation_test_config",
            dataset_path="/nonexistent/path.csv",
            algorithm_name="UnknownAlgorithm",
        )

        # Create validation result
        validation_result = ConfigurationValidationResultDTO(
            is_valid=False,
            validation_score=0.6,
            errors=["Dataset file not found", "Unknown algorithm"],
            warnings=["Contamination rate not specified"],
            suggestions=["Verify dataset path", "Use supported algorithm"],
            dataset_validation={"file_exists": False},
            algorithm_validation={"supported": False},
            compatibility_issues=["Algorithm not compatible with dataset"],
            missing_dependencies=["scikit-learn"],
            estimated_runtime=300.0,
            risk_assessment="high",
        )

        # Verify validation workflow
        assert validation_result.is_valid is False
        assert validation_result.validation_score == 0.6
        assert len(validation_result.errors) == 2
        assert len(validation_result.warnings) == 1
        assert len(validation_result.suggestions) == 2
        assert validation_result.dataset_validation["file_exists"] is False
        assert validation_result.algorithm_validation["supported"] is False
        assert validation_result.risk_assessment == "high"

    def test_configuration_export_workflow(self):
        """Test configuration export workflow."""
        # Create configuration
        config = create_basic_configuration(
            name="export_test_config",
            dataset_path="/path/to/dataset.csv",
            algorithm_name="IsolationForest",
        )

        # Create export request
        export_request = ConfigurationExportRequestDTO(
            configuration_ids=[config.id],
            export_format=ExportFormat.YAML,
            include_metadata=True,
            include_performance=True,
            output_path="/path/to/export.yaml",
            template_name="isolation_forest_template",
            parameterize=True,
        )

        # Create export response
        export_response = ConfigurationResponseDTO(
            success=True,
            message="Export completed successfully",
            export_data="exported_yaml_content",
            export_files=["/path/to/export.yaml"],
            execution_time=1.0,
        )

        # Verify export workflow
        assert export_request.configuration_ids == [config.id]
        assert export_request.export_format == ExportFormat.YAML
        assert export_request.include_metadata is True
        assert export_request.template_name == "isolation_forest_template"
        assert export_response.success is True
        assert export_response.export_data == "exported_yaml_content"
        assert export_response.export_files == ["/path/to/export.yaml"]

    def test_dto_serialization(self):
        """Test DTO serialization and deserialization."""
        # Create a complex configuration
        config = create_basic_configuration(
            name="serialization_test",
            dataset_path="/path/to/dataset.csv",
            algorithm_name="IsolationForest",
        )

        # Serialize to dict
        config_dict = config.model_dump()

        assert config_dict["name"] == "serialization_test"
        assert config_dict["dataset_config"]["dataset_path"] == "/path/to/dataset.csv"
        assert config_dict["algorithm_config"]["algorithm_name"] == "IsolationForest"
        assert "id" in config_dict
        assert "metadata" in config_dict

        # Deserialize from dict
        config_restored = ExperimentConfigurationDTO.model_validate(config_dict)

        assert config_restored.name == config.name
        assert (
            config_restored.dataset_config.dataset_path
            == config.dataset_config.dataset_path
        )
        assert (
            config_restored.algorithm_config.algorithm_name
            == config.algorithm_config.algorithm_name
        )
        assert config_restored.id == config.id

    def test_backward_compatibility_aliases(self):
        """Test backward compatibility aliases."""
        # Test that aliases work correctly
        from pynomaly.application.dto.configuration_dto import (
            DatasetConfigurationDTO,
            EvaluationConfigurationDTO,
            PreprocessingConfigurationDTO,
        )

        # Create DTOs using aliases
        dataset_config = DatasetConfigurationDTO(dataset_path="/path/to/dataset.csv")
        algorithm_config = AlgorithmConfigurationDTO(algorithm_name="IsolationForest")
        preprocessing_config = PreprocessingConfigurationDTO()
        evaluation_config = EvaluationConfigurationDTO()

        # Verify aliases work
        assert isinstance(dataset_config, DatasetConfigDTO)
        assert isinstance(algorithm_config, AlgorithmConfigDTO)
        assert isinstance(preprocessing_config, PreprocessingConfigDTO)
        assert isinstance(evaluation_config, EvaluationConfigDTO)

        # Verify they can be used in main configuration
        metadata = ConfigurationMetadataDTO(source=ConfigurationSource.MANUAL)
        config = ExperimentConfigurationDTO(
            name="alias_test",
            dataset_config=dataset_config,
            algorithm_config=algorithm_config,
            preprocessing_config=preprocessing_config,
            evaluation_config=evaluation_config,
            metadata=metadata,
        )

        assert config.dataset_config.dataset_path == "/path/to/dataset.csv"
        assert config.algorithm_config.algorithm_name == "IsolationForest"
