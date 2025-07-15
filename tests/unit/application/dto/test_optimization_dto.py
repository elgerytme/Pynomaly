"""Tests for Optimization DTOs."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from pynomaly.application.dto.optimization_dto import (
    AutoMLRequestDTO,
    AutoMLResponseDTO,
    DatasetCharacteristicsDTO,
    EnsembleOptimizationDTO,
    LearningInsightsDTO,
    MetaLearningConfigDTO,
    OptimizationConfigDTO,
    OptimizationHistoryDTO,
    OptimizationMonitoringDTO,
    OptimizationObjectiveDTO,
    OptimizationResultDTO,
    PerformancePredictionDTO,
    ResourceConstraintsDTO,
    create_default_constraints,
    create_default_objectives,
    create_optimization_config,
)


class TestOptimizationObjectiveDTO:
    """Test suite for OptimizationObjectiveDTO."""

    def test_valid_creation(self):
        """Test creating a valid optimization objective DTO."""
        dto = OptimizationObjectiveDTO(
            name="accuracy",
            weight=0.6,
            direction="maximize",
            threshold=0.8,
            description="Model accuracy objective",
        )

        assert dto.name == "accuracy"
        assert dto.weight == 0.6
        assert dto.direction == "maximize"
        assert dto.threshold == 0.8
        assert dto.description == "Model accuracy objective"

    def test_default_values(self):
        """Test default values."""
        dto = OptimizationObjectiveDTO(
            name="precision",
            weight=0.5,
            direction="maximize",
        )

        assert dto.name == "precision"
        assert dto.weight == 0.5
        assert dto.direction == "maximize"
        assert dto.threshold is None
        assert dto.description == ""

    def test_weight_validation(self):
        """Test weight validation."""
        # Valid weight
        dto = OptimizationObjectiveDTO(
            name="recall",
            weight=0.3,
            direction="maximize",
        )
        assert dto.weight == 0.3

        # Invalid: zero weight
        with pytest.raises(ValidationError):
            OptimizationObjectiveDTO(
                name="recall",
                weight=0.0,
                direction="maximize",
            )

        # Invalid: negative weight
        with pytest.raises(ValidationError):
            OptimizationObjectiveDTO(
                name="recall",
                weight=-0.1,
                direction="maximize",
            )

        # Invalid: weight greater than 1
        with pytest.raises(ValidationError):
            OptimizationObjectiveDTO(
                name="recall",
                weight=1.1,
                direction="maximize",
            )

    def test_direction_validation(self):
        """Test direction validation."""
        # Valid directions
        dto_max = OptimizationObjectiveDTO(
            name="f1_score",
            weight=0.7,
            direction="maximize",
        )
        assert dto_max.direction == "maximize"

        dto_min = OptimizationObjectiveDTO(
            name="training_time",
            weight=0.3,
            direction="minimize",
        )
        assert dto_min.direction == "minimize"

        # Invalid direction
        with pytest.raises(ValidationError):
            OptimizationObjectiveDTO(
                name="f1_score",
                weight=0.7,
                direction="invalid_direction",
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            OptimizationObjectiveDTO(
                weight=0.5,
                direction="maximize",
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            OptimizationObjectiveDTO(
                name="accuracy",
                weight=0.5,
                direction="maximize",
                unknown_field="value",
            )


class TestResourceConstraintsDTO:
    """Test suite for ResourceConstraintsDTO."""

    def test_valid_creation(self):
        """Test creating a valid resource constraints DTO."""
        dto = ResourceConstraintsDTO(
            max_time_seconds=1800,
            max_trials=50,
            max_memory_mb=8192,
            max_cpu_cores=8,
            gpu_available=True,
            prefer_speed=True,
        )

        assert dto.max_time_seconds == 1800
        assert dto.max_trials == 50
        assert dto.max_memory_mb == 8192
        assert dto.max_cpu_cores == 8
        assert dto.gpu_available is True
        assert dto.prefer_speed is True

    def test_default_values(self):
        """Test default values."""
        dto = ResourceConstraintsDTO()

        assert dto.max_time_seconds == 3600
        assert dto.max_trials == 100
        assert dto.max_memory_mb == 4096
        assert dto.max_cpu_cores == 4
        assert dto.gpu_available is False
        assert dto.prefer_speed is False

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Valid minimum values
        dto = ResourceConstraintsDTO(
            max_time_seconds=60,
            max_trials=10,
            max_memory_mb=512,
            max_cpu_cores=1,
        )
        assert dto.max_time_seconds == 60
        assert dto.max_trials == 10
        assert dto.max_memory_mb == 512
        assert dto.max_cpu_cores == 1

        # Invalid: max_time_seconds too small
        with pytest.raises(ValidationError):
            ResourceConstraintsDTO(max_time_seconds=30)

        # Invalid: max_trials too small
        with pytest.raises(ValidationError):
            ResourceConstraintsDTO(max_trials=5)

        # Invalid: max_memory_mb too small
        with pytest.raises(ValidationError):
            ResourceConstraintsDTO(max_memory_mb=256)

        # Invalid: max_cpu_cores too small
        with pytest.raises(ValidationError):
            ResourceConstraintsDTO(max_cpu_cores=0)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ResourceConstraintsDTO(unknown_field="value")


class TestOptimizationConfigDTO:
    """Test suite for OptimizationConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid optimization config DTO."""
        objectives = [
            OptimizationObjectiveDTO(
                name="accuracy",
                weight=0.6,
                direction="maximize",
            ),
            OptimizationObjectiveDTO(
                name="speed",
                weight=0.4,
                direction="maximize",
            ),
        ]
        constraints = ResourceConstraintsDTO(
            max_time_seconds=1800,
            max_trials=50,
        )

        dto = OptimizationConfigDTO(
            algorithm_name="IsolationForest",
            objectives=objectives,
            constraints=constraints,
            enable_learning=True,
            enable_distributed=False,
            n_parallel_jobs=2,
        )

        assert dto.algorithm_name == "IsolationForest"
        assert len(dto.objectives) == 2
        assert dto.objectives[0].name == "accuracy"
        assert dto.objectives[1].name == "speed"
        assert dto.constraints.max_time_seconds == 1800
        assert dto.enable_learning is True
        assert dto.enable_distributed is False
        assert dto.n_parallel_jobs == 2

    def test_default_values(self):
        """Test default values."""
        objectives = [
            OptimizationObjectiveDTO(
                name="accuracy",
                weight=1.0,
                direction="maximize",
            )
        ]
        constraints = ResourceConstraintsDTO()

        dto = OptimizationConfigDTO(
            algorithm_name="OneClassSVM",
            objectives=objectives,
            constraints=constraints,
        )

        assert dto.enable_learning is True
        assert dto.enable_distributed is False
        assert dto.n_parallel_jobs == 1

    def test_n_parallel_jobs_validation(self):
        """Test n_parallel_jobs validation."""
        objectives = [
            OptimizationObjectiveDTO(
                name="accuracy",
                weight=1.0,
                direction="maximize",
            )
        ]
        constraints = ResourceConstraintsDTO()

        # Valid parallel jobs
        dto = OptimizationConfigDTO(
            algorithm_name="IsolationForest",
            objectives=objectives,
            constraints=constraints,
            n_parallel_jobs=4,
        )
        assert dto.n_parallel_jobs == 4

        # Invalid: zero parallel jobs
        with pytest.raises(ValidationError):
            OptimizationConfigDTO(
                algorithm_name="IsolationForest",
                objectives=objectives,
                constraints=constraints,
                n_parallel_jobs=0,
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            OptimizationConfigDTO(
                objectives=[],
                constraints=ResourceConstraintsDTO(),
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        objectives = [
            OptimizationObjectiveDTO(
                name="accuracy",
                weight=1.0,
                direction="maximize",
            )
        ]
        constraints = ResourceConstraintsDTO()

        with pytest.raises(ValidationError):
            OptimizationConfigDTO(
                algorithm_name="IsolationForest",
                objectives=objectives,
                constraints=constraints,
                unknown_field="value",
            )


class TestDatasetCharacteristicsDTO:
    """Test suite for DatasetCharacteristicsDTO."""

    def test_valid_creation(self):
        """Test creating a valid dataset characteristics DTO."""
        dto = DatasetCharacteristicsDTO(
            n_samples=10000,
            n_features=50,
            size_category="medium",
            feature_types={"numerical": 0.8, "categorical": 0.2},
            data_distribution={"mean": 0.5, "std": 0.3, "skewness": 0.1},
            sparsity=0.05,
            correlation_structure={"high_correlation": 0.15, "low_correlation": 0.85},
            outlier_characteristics={"outlier_ratio": 0.02, "anomaly_score": 0.75},
        )

        assert dto.n_samples == 10000
        assert dto.n_features == 50
        assert dto.size_category == "medium"
        assert dto.feature_types == {"numerical": 0.8, "categorical": 0.2}
        assert dto.data_distribution == {"mean": 0.5, "std": 0.3, "skewness": 0.1}
        assert dto.sparsity == 0.05
        assert dto.correlation_structure == {
            "high_correlation": 0.15,
            "low_correlation": 0.85,
        }
        assert dto.outlier_characteristics == {
            "outlier_ratio": 0.02,
            "anomaly_score": 0.75,
        }

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DatasetCharacteristicsDTO(
                n_features=50,
                size_category="medium",
                feature_types={},
                data_distribution={},
                sparsity=0.05,
                correlation_structure={},
                outlier_characteristics={},
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DatasetCharacteristicsDTO(
                n_samples=10000,
                n_features=50,
                size_category="medium",
                feature_types={},
                data_distribution={},
                sparsity=0.05,
                correlation_structure={},
                outlier_characteristics={},
                unknown_field="value",
            )


class TestOptimizationResultDTO:
    """Test suite for OptimizationResultDTO."""

    def test_valid_creation(self):
        """Test creating a valid optimization result DTO."""
        dto = OptimizationResultDTO(
            best_parameters={"n_estimators": 100, "contamination": 0.1},
            best_metrics={"accuracy": 0.95, "precision": 0.92, "recall": 0.89},
            optimization_time=1800.5,
            total_trials=50,
            successful_trials=48,
            pareto_solutions=[
                {"parameters": {"n_estimators": 100}, "metrics": {"accuracy": 0.95}},
                {"parameters": {"n_estimators": 200}, "metrics": {"accuracy": 0.94}},
            ],
        )

        assert dto.best_parameters == {"n_estimators": 100, "contamination": 0.1}
        assert dto.best_metrics == {"accuracy": 0.95, "precision": 0.92, "recall": 0.89}
        assert dto.optimization_time == 1800.5
        assert dto.total_trials == 50
        assert dto.successful_trials == 48
        assert len(dto.pareto_solutions) == 2

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            OptimizationResultDTO(
                best_metrics={"accuracy": 0.95},
                optimization_time=1800.5,
                total_trials=50,
                successful_trials=48,
                pareto_solutions=[],
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            OptimizationResultDTO(
                best_parameters={"n_estimators": 100},
                best_metrics={"accuracy": 0.95},
                optimization_time=1800.5,
                total_trials=50,
                successful_trials=48,
                pareto_solutions=[],
                unknown_field="value",
            )


class TestOptimizationHistoryDTO:
    """Test suite for OptimizationHistoryDTO."""

    def test_valid_creation(self):
        """Test creating a valid optimization history DTO."""
        dataset_characteristics = DatasetCharacteristicsDTO(
            n_samples=5000,
            n_features=20,
            size_category="small",
            feature_types={"numerical": 1.0},
            data_distribution={"mean": 0.0, "std": 1.0},
            sparsity=0.0,
            correlation_structure={"avg_correlation": 0.3},
            outlier_characteristics={"outlier_ratio": 0.05},
        )

        timestamp = datetime.now()

        dto = OptimizationHistoryDTO(
            algorithm_name="IsolationForest",
            dataset_characteristics=dataset_characteristics,
            best_parameters={"n_estimators": 100, "contamination": 0.1},
            performance_metrics={"accuracy": 0.92, "f1_score": 0.89},
            optimization_time=900.0,
            resource_usage={"memory_mb": 512, "cpu_percent": 75},
            user_feedback={"satisfaction": 4, "comments": "Good results"},
            timestamp=timestamp,
        )

        assert dto.algorithm_name == "IsolationForest"
        assert dto.dataset_characteristics.n_samples == 5000
        assert dto.best_parameters == {"n_estimators": 100, "contamination": 0.1}
        assert dto.performance_metrics == {"accuracy": 0.92, "f1_score": 0.89}
        assert dto.optimization_time == 900.0
        assert dto.resource_usage == {"memory_mb": 512, "cpu_percent": 75}
        assert dto.user_feedback == {"satisfaction": 4, "comments": "Good results"}
        assert dto.timestamp == timestamp

    def test_default_timestamp(self):
        """Test default timestamp generation."""
        dataset_characteristics = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=10,
            size_category="small",
            feature_types={"numerical": 1.0},
            data_distribution={"mean": 0.0, "std": 1.0},
            sparsity=0.0,
            correlation_structure={},
            outlier_characteristics={},
        )

        dto = OptimizationHistoryDTO(
            algorithm_name="OneClassSVM",
            dataset_characteristics=dataset_characteristics,
            best_parameters={"nu": 0.05},
            performance_metrics={"accuracy": 0.88},
            optimization_time=600.0,
            resource_usage={"memory_mb": 256},
        )

        assert isinstance(dto.timestamp, datetime)
        assert dto.user_feedback is None

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            OptimizationHistoryDTO(
                dataset_characteristics=DatasetCharacteristicsDTO(
                    n_samples=1000,
                    n_features=10,
                    size_category="small",
                    feature_types={},
                    data_distribution={},
                    sparsity=0.0,
                    correlation_structure={},
                    outlier_characteristics={},
                ),
                best_parameters={},
                performance_metrics={},
                optimization_time=600.0,
                resource_usage={},
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        dataset_characteristics = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=10,
            size_category="small",
            feature_types={},
            data_distribution={},
            sparsity=0.0,
            correlation_structure={},
            outlier_characteristics={},
        )

        with pytest.raises(ValidationError):
            OptimizationHistoryDTO(
                algorithm_name="IsolationForest",
                dataset_characteristics=dataset_characteristics,
                best_parameters={},
                performance_metrics={},
                optimization_time=600.0,
                resource_usage={},
                unknown_field="value",
            )


class TestAutoMLRequestDTO:
    """Test suite for AutoMLRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid AutoML request DTO."""
        optimization_config = OptimizationConfigDTO(
            algorithm_name="IsolationForest",
            objectives=[
                OptimizationObjectiveDTO(
                    name="accuracy",
                    weight=1.0,
                    direction="maximize",
                )
            ],
            constraints=ResourceConstraintsDTO(),
        )

        dto = AutoMLRequestDTO(
            dataset_name="fraud_dataset",
            algorithm_names=["IsolationForest", "OneClassSVM"],
            optimization_config=optimization_config,
            evaluation_mode="holdout",
            output_format="detailed",
        )

        assert dto.dataset_name == "fraud_dataset"
        assert dto.algorithm_names == ["IsolationForest", "OneClassSVM"]
        assert dto.optimization_config.algorithm_name == "IsolationForest"
        assert dto.evaluation_mode == "holdout"
        assert dto.output_format == "detailed"

    def test_default_values(self):
        """Test default values."""
        optimization_config = OptimizationConfigDTO(
            algorithm_name="IsolationForest",
            objectives=[
                OptimizationObjectiveDTO(
                    name="accuracy",
                    weight=1.0,
                    direction="maximize",
                )
            ],
            constraints=ResourceConstraintsDTO(),
        )

        dto = AutoMLRequestDTO(
            dataset_name="test_dataset",
            algorithm_names=["IsolationForest"],
            optimization_config=optimization_config,
        )

        assert dto.evaluation_mode == "cross_validation"
        assert dto.output_format == "comprehensive"

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AutoMLRequestDTO(
                algorithm_names=["IsolationForest"],
                optimization_config=OptimizationConfigDTO(
                    algorithm_name="IsolationForest",
                    objectives=[],
                    constraints=ResourceConstraintsDTO(),
                ),
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        optimization_config = OptimizationConfigDTO(
            algorithm_name="IsolationForest",
            objectives=[
                OptimizationObjectiveDTO(
                    name="accuracy",
                    weight=1.0,
                    direction="maximize",
                )
            ],
            constraints=ResourceConstraintsDTO(),
        )

        with pytest.raises(ValidationError):
            AutoMLRequestDTO(
                dataset_name="test_dataset",
                algorithm_names=["IsolationForest"],
                optimization_config=optimization_config,
                unknown_field="value",
            )


class TestAutoMLResponseDTO:
    """Test suite for AutoMLResponseDTO."""

    def test_valid_creation(self):
        """Test creating a valid AutoML response DTO."""
        result1 = OptimizationResultDTO(
            best_parameters={"n_estimators": 100},
            best_metrics={"accuracy": 0.95},
            optimization_time=600.0,
            total_trials=20,
            successful_trials=19,
            pareto_solutions=[],
        )

        result2 = OptimizationResultDTO(
            best_parameters={"nu": 0.05},
            best_metrics={"accuracy": 0.92},
            optimization_time=800.0,
            total_trials=25,
            successful_trials=24,
            pareto_solutions=[],
        )

        best_overall = OptimizationResultDTO(
            best_parameters={"n_estimators": 100},
            best_metrics={"accuracy": 0.95},
            optimization_time=600.0,
            total_trials=20,
            successful_trials=19,
            pareto_solutions=[],
        )

        timestamp = datetime.now()

        dto = AutoMLResponseDTO(
            request_id="req_123",
            status="completed",
            results=[result1, result2],
            best_overall=best_overall,
            execution_summary={
                "total_time": 1400.0,
                "algorithms_tested": 2,
                "best_algorithm": "IsolationForest",
            },
            recommendations=[
                "Use IsolationForest for best accuracy",
                "Consider ensemble methods for robustness",
            ],
            timestamp=timestamp,
        )

        assert dto.request_id == "req_123"
        assert dto.status == "completed"
        assert len(dto.results) == 2
        assert dto.best_overall.best_metrics["accuracy"] == 0.95
        assert dto.execution_summary["total_time"] == 1400.0
        assert len(dto.recommendations) == 2
        assert dto.timestamp == timestamp

    def test_default_timestamp(self):
        """Test default timestamp generation."""
        result = OptimizationResultDTO(
            best_parameters={"n_estimators": 100},
            best_metrics={"accuracy": 0.95},
            optimization_time=600.0,
            total_trials=20,
            successful_trials=19,
            pareto_solutions=[],
        )

        dto = AutoMLResponseDTO(
            request_id="req_456",
            status="completed",
            results=[result],
            best_overall=result,
            execution_summary={},
            recommendations=[],
        )

        assert isinstance(dto.timestamp, datetime)

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AutoMLResponseDTO(
                status="completed",
                results=[],
                best_overall=OptimizationResultDTO(
                    best_parameters={},
                    best_metrics={},
                    optimization_time=600.0,
                    total_trials=20,
                    successful_trials=19,
                    pareto_solutions=[],
                ),
                execution_summary={},
                recommendations=[],
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        result = OptimizationResultDTO(
            best_parameters={},
            best_metrics={},
            optimization_time=600.0,
            total_trials=20,
            successful_trials=19,
            pareto_solutions=[],
        )

        with pytest.raises(ValidationError):
            AutoMLResponseDTO(
                request_id="req_789",
                status="completed",
                results=[result],
                best_overall=result,
                execution_summary={},
                recommendations=[],
                unknown_field="value",
            )


class TestEnsembleOptimizationDTO:
    """Test suite for EnsembleOptimizationDTO."""

    def test_valid_creation(self):
        """Test creating a valid ensemble optimization DTO."""
        dto = EnsembleOptimizationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM", "LocalOutlierFactor"],
            ensemble_method="stacking",
            weight_optimization=True,
            diversity_threshold=0.4,
            max_ensemble_size=3,
        )

        assert dto.base_algorithms == [
            "IsolationForest",
            "OneClassSVM",
            "LocalOutlierFactor",
        ]
        assert dto.ensemble_method == "stacking"
        assert dto.weight_optimization is True
        assert dto.diversity_threshold == 0.4
        assert dto.max_ensemble_size == 3

    def test_default_values(self):
        """Test default values."""
        dto = EnsembleOptimizationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"],
        )

        assert dto.ensemble_method == "voting"
        assert dto.weight_optimization is True
        assert dto.diversity_threshold == 0.3
        assert dto.max_ensemble_size == 5

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            EnsembleOptimizationDTO(
                ensemble_method="voting",
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            EnsembleOptimizationDTO(
                base_algorithms=["IsolationForest"],
                unknown_field="value",
            )


class TestMetaLearningConfigDTO:
    """Test suite for MetaLearningConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid meta-learning config DTO."""
        dto = MetaLearningConfigDTO(
            enable_meta_learning=True,
            similarity_threshold=0.8,
            learning_rate=0.05,
            memory_size=500,
            adaptation_strategy="exponential",
        )

        assert dto.enable_meta_learning is True
        assert dto.similarity_threshold == 0.8
        assert dto.learning_rate == 0.05
        assert dto.memory_size == 500
        assert dto.adaptation_strategy == "exponential"

    def test_default_values(self):
        """Test default values."""
        dto = MetaLearningConfigDTO()

        assert dto.enable_meta_learning is True
        assert dto.similarity_threshold == 0.7
        assert dto.learning_rate == 0.1
        assert dto.memory_size == 1000
        assert dto.adaptation_strategy == "weighted"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            MetaLearningConfigDTO(unknown_field="value")


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_create_default_objectives(self):
        """Test creating default objectives."""
        objectives = create_default_objectives()

        assert len(objectives) == 4
        assert objectives[0].name == "accuracy"
        assert objectives[0].weight == 0.4
        assert objectives[0].direction == "maximize"
        assert objectives[1].name == "speed"
        assert objectives[1].weight == 0.3
        assert objectives[2].name == "interpretability"
        assert objectives[2].weight == 0.2
        assert objectives[3].name == "memory_efficiency"
        assert objectives[3].weight == 0.1

        # Verify all objectives are valid
        for obj in objectives:
            assert isinstance(obj, OptimizationObjectiveDTO)
            assert obj.weight > 0.0
            assert obj.weight <= 1.0
            assert obj.direction in ["maximize", "minimize"]

    def test_create_default_constraints(self):
        """Test creating default constraints."""
        constraints = create_default_constraints()

        assert isinstance(constraints, ResourceConstraintsDTO)
        assert constraints.max_time_seconds == 3600
        assert constraints.max_trials == 100
        assert constraints.max_memory_mb == 4096
        assert constraints.max_cpu_cores == 4
        assert constraints.gpu_available is False
        assert constraints.prefer_speed is False

    def test_create_optimization_config(self):
        """Test creating optimization config."""
        # With defaults
        config = create_optimization_config("IsolationForest")

        assert config.algorithm_name == "IsolationForest"
        assert len(config.objectives) == 4
        assert config.objectives[0].name == "accuracy"
        assert config.constraints.max_time_seconds == 3600
        assert config.enable_learning is True
        assert config.enable_distributed is False
        assert config.n_parallel_jobs == 1

        # With custom objectives and constraints
        custom_objectives = [
            OptimizationObjectiveDTO(
                name="custom_metric",
                weight=1.0,
                direction="maximize",
            )
        ]
        custom_constraints = ResourceConstraintsDTO(
            max_time_seconds=1800,
            max_trials=50,
        )

        config_custom = create_optimization_config(
            "OneClassSVM",
            objectives=custom_objectives,
            constraints=custom_constraints,
        )

        assert config_custom.algorithm_name == "OneClassSVM"
        assert len(config_custom.objectives) == 1
        assert config_custom.objectives[0].name == "custom_metric"
        assert config_custom.constraints.max_time_seconds == 1800
        assert config_custom.constraints.max_trials == 50


class TestOptimizationDTOIntegration:
    """Test integration scenarios for optimization DTOs."""

    def test_complete_automl_workflow(self):
        """Test complete AutoML workflow with all DTOs."""
        # Create AutoML request
        optimization_config = create_optimization_config("IsolationForest")
        request = AutoMLRequestDTO(
            dataset_name="fraud_detection_dataset",
            algorithm_names=["IsolationForest", "OneClassSVM", "LocalOutlierFactor"],
            optimization_config=optimization_config,
            evaluation_mode="cross_validation",
            output_format="comprehensive",
        )

        # Create optimization results for each algorithm
        results = []
        algorithms = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]
        scores = [0.95, 0.92, 0.88]
        times = [600.0, 800.0, 400.0]

        for algo, score, time in zip(algorithms, scores, times, strict=False):
            result = OptimizationResultDTO(
                best_parameters={"algorithm": algo, "contamination": 0.1},
                best_metrics={"accuracy": score, "f1_score": score - 0.02},
                optimization_time=time,
                total_trials=20,
                successful_trials=19,
                pareto_solutions=[
                    {
                        "parameters": {"contamination": 0.05},
                        "metrics": {"accuracy": score - 0.01},
                    },
                    {
                        "parameters": {"contamination": 0.15},
                        "metrics": {"accuracy": score - 0.02},
                    },
                ],
            )
            results.append(result)

        # Create AutoML response
        best_overall = results[0]  # IsolationForest with highest score
        response = AutoMLResponseDTO(
            request_id="automl_req_123",
            status="completed",
            results=results,
            best_overall=best_overall,
            execution_summary={
                "total_time": sum(times),
                "algorithms_tested": len(algorithms),
                "best_algorithm": "IsolationForest",
                "best_score": 0.95,
                "resource_usage": {"peak_memory_mb": 2048, "cpu_hours": 2.5},
            },
            recommendations=[
                "IsolationForest achieved the best performance with 95% accuracy",
                "Consider ensemble methods combining top 2 algorithms",
                "Further hyperparameter tuning could improve performance by 2-3%",
            ],
        )

        # Verify workflow consistency
        assert request.dataset_name == "fraud_detection_dataset"
        assert len(request.algorithm_names) == 3
        assert response.status == "completed"
        assert len(response.results) == 3
        assert response.best_overall.best_metrics["accuracy"] == 0.95
        assert response.execution_summary["algorithms_tested"] == 3
        assert response.execution_summary["best_algorithm"] == "IsolationForest"
        assert len(response.recommendations) == 3

        # Verify optimization results
        for i, result in enumerate(response.results):
            assert result.best_parameters["algorithm"] == algorithms[i]
            assert result.best_metrics["accuracy"] == scores[i]
            assert result.optimization_time == times[i]
            assert result.total_trials == 20
            assert result.successful_trials == 19
            assert len(result.pareto_solutions) == 2

    def test_ensemble_optimization_scenario(self):
        """Test ensemble optimization scenario."""
        # Create ensemble optimization configuration
        ensemble_config = EnsembleOptimizationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM", "LocalOutlierFactor"],
            ensemble_method="stacking",
            weight_optimization=True,
            diversity_threshold=0.4,
            max_ensemble_size=3,
        )

        # Create optimization configuration for ensemble
        optimization_config = OptimizationConfigDTO(
            algorithm_name="EnsembleModel",
            objectives=[
                OptimizationObjectiveDTO(
                    name="accuracy",
                    weight=0.5,
                    direction="maximize",
                ),
                OptimizationObjectiveDTO(
                    name="diversity",
                    weight=0.3,
                    direction="maximize",
                ),
                OptimizationObjectiveDTO(
                    name="speed",
                    weight=0.2,
                    direction="maximize",
                ),
            ],
            constraints=ResourceConstraintsDTO(
                max_time_seconds=7200,
                max_trials=150,
                max_memory_mb=8192,
            ),
            enable_learning=True,
            enable_distributed=True,
            n_parallel_jobs=4,
        )

        # Create ensemble optimization result
        ensemble_result = OptimizationResultDTO(
            best_parameters={
                "base_algorithms": ["IsolationForest", "OneClassSVM"],
                "ensemble_method": "stacking",
                "weights": [0.6, 0.4],
                "meta_learner": "LogisticRegression",
                "diversity_score": 0.45,
            },
            best_metrics={
                "accuracy": 0.97,
                "precision": 0.94,
                "recall": 0.96,
                "f1_score": 0.95,
                "diversity": 0.45,
                "ensemble_quality": 0.92,
            },
            optimization_time=3600.0,
            total_trials=100,
            successful_trials=95,
            pareto_solutions=[
                {
                    "parameters": {"weights": [0.7, 0.3]},
                    "metrics": {"accuracy": 0.96, "diversity": 0.48},
                },
                {
                    "parameters": {"weights": [0.5, 0.5]},
                    "metrics": {"accuracy": 0.95, "diversity": 0.52},
                },
            ],
        )

        # Create comprehensive optimization monitoring
        monitoring = OptimizationMonitoringDTO(
            optimization_id="ensemble_opt_456",
            current_trial=75,
            total_trials=100,
            elapsed_time=2700.0,
            estimated_remaining=900.0,
            current_best={
                "accuracy": 0.97,
                "diversity": 0.45,
                "ensemble_quality": 0.92,
            },
            trial_history=[
                {
                    "trial": 73,
                    "algorithm": "EnsembleModel",
                    "parameters": {"weights": [0.6, 0.4]},
                    "metrics": {"accuracy": 0.96, "diversity": 0.44},
                },
                {
                    "trial": 74,
                    "algorithm": "EnsembleModel",
                    "parameters": {"weights": [0.65, 0.35]},
                    "metrics": {"accuracy": 0.97, "diversity": 0.43},
                },
            ],
            resource_usage={
                "memory_mb": 4096,
                "cpu_percent": 85,
                "gpu_percent": 0,
            },
            status="running",
        )

        # Verify ensemble optimization
        assert ensemble_config.base_algorithms == [
            "IsolationForest",
            "OneClassSVM",
            "LocalOutlierFactor",
        ]
        assert ensemble_config.ensemble_method == "stacking"
        assert ensemble_config.weight_optimization is True
        assert ensemble_config.diversity_threshold == 0.4
        assert ensemble_config.max_ensemble_size == 3

        assert optimization_config.algorithm_name == "EnsembleModel"
        assert len(optimization_config.objectives) == 3
        assert optimization_config.objectives[0].name == "accuracy"
        assert optimization_config.objectives[1].name == "diversity"
        assert optimization_config.objectives[2].name == "speed"
        assert optimization_config.enable_distributed is True
        assert optimization_config.n_parallel_jobs == 4

        assert ensemble_result.best_metrics["accuracy"] == 0.97
        assert ensemble_result.best_metrics["diversity"] == 0.45
        assert ensemble_result.best_parameters["ensemble_method"] == "stacking"
        assert len(ensemble_result.best_parameters["weights"]) == 2
        assert ensemble_result.total_trials == 100
        assert ensemble_result.successful_trials == 95

        assert monitoring.optimization_id == "ensemble_opt_456"
        assert monitoring.current_trial == 75
        assert monitoring.total_trials == 100
        assert monitoring.current_best["accuracy"] == 0.97
        assert monitoring.status == "running"
        assert len(monitoring.trial_history) == 2

    def test_meta_learning_integration(self):
        """Test meta-learning integration with optimization history."""
        # Create meta-learning configuration
        meta_config = MetaLearningConfigDTO(
            enable_meta_learning=True,
            similarity_threshold=0.8,
            learning_rate=0.05,
            memory_size=500,
            adaptation_strategy="exponential",
        )

        # Create historical optimization entries
        dataset_chars_1 = DatasetCharacteristicsDTO(
            n_samples=10000,
            n_features=50,
            size_category="medium",
            feature_types={"numerical": 0.8, "categorical": 0.2},
            data_distribution={"mean": 0.5, "std": 0.3},
            sparsity=0.05,
            correlation_structure={"avg_correlation": 0.3},
            outlier_characteristics={"outlier_ratio": 0.02},
        )

        dataset_chars_2 = DatasetCharacteristicsDTO(
            n_samples=8000,
            n_features=45,
            size_category="medium",
            feature_types={"numerical": 0.75, "categorical": 0.25},
            data_distribution={"mean": 0.4, "std": 0.35},
            sparsity=0.03,
            correlation_structure={"avg_correlation": 0.35},
            outlier_characteristics={"outlier_ratio": 0.03},
        )

        history_entries = [
            OptimizationHistoryDTO(
                algorithm_name="IsolationForest",
                dataset_characteristics=dataset_chars_1,
                best_parameters={"n_estimators": 100, "contamination": 0.1},
                performance_metrics={"accuracy": 0.95, "f1_score": 0.92},
                optimization_time=1800.0,
                resource_usage={"memory_mb": 2048, "cpu_percent": 80},
                user_feedback={"satisfaction": 5, "would_recommend": True},
            ),
            OptimizationHistoryDTO(
                algorithm_name="OneClassSVM",
                dataset_characteristics=dataset_chars_2,
                best_parameters={"nu": 0.05, "gamma": "scale"},
                performance_metrics={"accuracy": 0.92, "f1_score": 0.89},
                optimization_time=2400.0,
                resource_usage={"memory_mb": 1024, "cpu_percent": 70},
                user_feedback={"satisfaction": 4, "would_recommend": True},
            ),
        ]

        # Create learning insights from history
        learning_insights = LearningInsightsDTO(
            algorithm_trends={
                "IsolationForest": {
                    "avg_accuracy": 0.94,
                    "avg_optimization_time": 1900.0,
                    "success_rate": 0.95,
                    "preferred_datasets": ["medium", "large"],
                },
                "OneClassSVM": {
                    "avg_accuracy": 0.91,
                    "avg_optimization_time": 2200.0,
                    "success_rate": 0.88,
                    "preferred_datasets": ["small", "medium"],
                },
            },
            total_optimizations=50,
            learning_insights=[
                "IsolationForest performs best on medium-sized datasets with numerical features",
                "OneClassSVM requires more optimization time but provides stable results",
                "Ensemble methods show 3-5% improvement over single algorithms",
            ],
            performance_improvements={
                "IsolationForest": 0.08,
                "OneClassSVM": 0.05,
                "EnsembleModel": 0.12,
            },
            parameter_preferences={
                "IsolationForest": {
                    "n_estimators": {"mean": 150, "std": 50, "range": [50, 300]},
                    "contamination": {"mean": 0.08, "std": 0.03, "range": [0.01, 0.2]},
                },
                "OneClassSVM": {
                    "nu": {"mean": 0.06, "std": 0.02, "range": [0.01, 0.15]},
                    "gamma": {"preferred": "scale", "alternatives": ["auto"]},
                },
            },
        )

        # Create performance prediction based on meta-learning
        new_dataset_chars = DatasetCharacteristicsDTO(
            n_samples=12000,
            n_features=55,
            size_category="medium",
            feature_types={"numerical": 0.85, "categorical": 0.15},
            data_distribution={"mean": 0.45, "std": 0.32},
            sparsity=0.04,
            correlation_structure={"avg_correlation": 0.28},
            outlier_characteristics={"outlier_ratio": 0.025},
        )

        performance_prediction = PerformancePredictionDTO(
            algorithm_name="IsolationForest",
            dataset_characteristics=new_dataset_chars,
            predicted_metrics={
                "accuracy": 0.96,
                "precision": 0.93,
                "recall": 0.94,
                "f1_score": 0.93,
            },
            confidence_intervals={
                "accuracy": [0.94, 0.98],
                "precision": [0.91, 0.95],
                "recall": [0.92, 0.96],
                "f1_score": [0.91, 0.95],
            },
            prediction_accuracy=0.87,
            risk_assessment="low",
        )

        # Verify meta-learning integration
        assert meta_config.enable_meta_learning is True
        assert meta_config.similarity_threshold == 0.8
        assert meta_config.learning_rate == 0.05
        assert meta_config.memory_size == 500
        assert meta_config.adaptation_strategy == "exponential"

        assert len(history_entries) == 2
        assert history_entries[0].algorithm_name == "IsolationForest"
        assert history_entries[0].performance_metrics["accuracy"] == 0.95
        assert history_entries[1].algorithm_name == "OneClassSVM"
        assert history_entries[1].performance_metrics["accuracy"] == 0.92

        assert learning_insights.total_optimizations == 50
        assert len(learning_insights.learning_insights) == 3
        assert (
            learning_insights.algorithm_trends["IsolationForest"]["avg_accuracy"]
            == 0.94
        )
        assert learning_insights.performance_improvements["EnsembleModel"] == 0.12

        assert performance_prediction.algorithm_name == "IsolationForest"
        assert performance_prediction.predicted_metrics["accuracy"] == 0.96
        assert performance_prediction.confidence_intervals["accuracy"] == [0.94, 0.98]
        assert performance_prediction.prediction_accuracy == 0.87
        assert performance_prediction.risk_assessment == "low"

    def test_serialization_compatibility(self):
        """Test serialization compatibility across all DTOs."""
        # Test complex nested structure serialization
        objectives = create_default_objectives()
        constraints = create_default_constraints()
        optimization_config = OptimizationConfigDTO(
            algorithm_name="IsolationForest",
            objectives=objectives,
            constraints=constraints,
            enable_learning=True,
            enable_distributed=True,
            n_parallel_jobs=4,
        )

        # Serialize and verify structure
        config_dict = optimization_config.model_dump()
        assert config_dict["algorithm_name"] == "IsolationForest"
        assert len(config_dict["objectives"]) == 4
        assert config_dict["constraints"]["max_time_seconds"] == 3600
        assert config_dict["enable_learning"] is True
        assert config_dict["enable_distributed"] is True
        assert config_dict["n_parallel_jobs"] == 4

        # Deserialize and verify integrity
        restored_config = OptimizationConfigDTO.model_validate(config_dict)
        assert restored_config.algorithm_name == optimization_config.algorithm_name
        assert len(restored_config.objectives) == len(optimization_config.objectives)
        assert (
            restored_config.constraints.max_time_seconds
            == optimization_config.constraints.max_time_seconds
        )
        assert restored_config.enable_learning == optimization_config.enable_learning
        assert (
            restored_config.enable_distributed == optimization_config.enable_distributed
        )
        assert restored_config.n_parallel_jobs == optimization_config.n_parallel_jobs

        # Test objective serialization
        for i, (original, restored) in enumerate(
            zip(
                optimization_config.objectives, restored_config.objectives, strict=False
            )
        ):
            assert original.name == restored.name
            assert original.weight == restored.weight
            assert original.direction == restored.direction
            assert original.threshold == restored.threshold
            assert original.description == restored.description
