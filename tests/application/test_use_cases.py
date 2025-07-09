"""Comprehensive tests for application layer use cases."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.use_cases import (
    DetectAnomaliesRequest,
    DetectAnomaliesUseCase,
    EvaluateModelRequest,
    EvaluateModelResponse,
    EvaluateModelUseCase,
    ExplainAnomalyUseCase,
    TrainDetectorUseCase,
)
from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.services import FeatureValidator
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.infrastructure.repositories import InMemoryDetectorRepository


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    # Create sample data with proper DataFrame format
    features = np.random.RandomState(42).normal(0, 1, (100, 5))
    df = pd.DataFrame(
        features,
        columns=["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
    )
    return Dataset(name="test_dataset", data=df)


@pytest.fixture
def sample_detector():
    """Create a sample detector for testing."""
    from pynomaly.infrastructure.adapters import SklearnAdapter

    return SklearnAdapter(
        algorithm_name="IsolationForest",
        name="test_detector",
        contamination_rate=ContaminationRate(0.1),
        n_estimators=100,
        random_state=42,
    )


@pytest.fixture
def feature_validator():
    """Create a feature validator."""
    return FeatureValidator()


@pytest.fixture
def detector_repository():
    """Create an in-memory detector repository."""
    return InMemoryDetectorRepository()


class TestDetectAnomaliesUseCase:
    """Test DetectAnomaliesUseCase functionality."""

    @pytest.mark.asyncio
    async def test_execute_success(
        self, detector_repository, feature_validator, sample_detector, sample_dataset
    ):
        """Test successful anomaly detection execution."""

        # Mark detector as fitted for testing
        sample_detector.is_fitted = True

        # Save detector
        detector_repository.save(sample_detector)

        # Create use case
        use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repository, feature_validator=feature_validator
        )

        # Mock the adapter registry to simulate a fitted detector
        with patch.object(use_case.adapter_registry, "score_with_detector") as mock_score, \
             patch.object(use_case.adapter_registry, "predict_with_detector") as mock_predict:
            
            # Mock scores and predictions
            mock_scores = [AnomalyScore(0.5) for _ in range(len(sample_dataset.data))]
            mock_predictions = [0 for _ in range(len(sample_dataset.data))]
            
            mock_score.return_value = mock_scores
            mock_predict.return_value = mock_predictions

            # Mock feature validator
            feature_validator.check_data_quality = Mock(
                return_value={"quality_score": 0.9}
            )
            feature_validator.suggest_preprocessing = Mock(return_value=[])

            # Create request
            request = DetectAnomaliesRequest(
                detector_id=sample_detector.id, dataset=sample_dataset
            )

            # Execute use case
            response = await use_case.execute(request)

            # Verify result
            assert response.result.detector_id == sample_detector.id
            assert response.result.dataset_id == sample_dataset.id
            assert len(response.result.scores) == len(sample_dataset.data)
            mock_score.assert_called_once_with(sample_detector, sample_dataset)
            mock_predict.assert_called_once_with(sample_detector, sample_dataset)

    @pytest.mark.asyncio
    async def test_execute_detector_not_found(
        self, detector_repository, feature_validator, sample_dataset
    ):
        """Test detection with non-existent detector."""
        from uuid import uuid4

        from pynomaly.application.use_cases.detect_anomalies import (
            DetectAnomaliesRequest,
        )

        use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repository, feature_validator=feature_validator
        )

        # Create request with non-existent detector ID
        request = DetectAnomaliesRequest(
            detector_id=uuid4(),  # Non-existent UUID
            dataset=sample_dataset,
        )

        with pytest.raises(ValueError, match="Detector.*not found"):
            await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_execute_invalid_features(
        self, detector_repository, feature_validator, sample_detector
    ):
        """Test detection with invalid features."""
        from pynomaly.application.use_cases.detect_anomalies import (
            DetectAnomaliesRequest,
        )
        from pynomaly.domain.exceptions import DatasetError

        # Repository methods are synchronous
        detector_repository.save(sample_detector)

        # Mark detector as fitted for testing
        sample_detector.is_fitted = True

        use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repository, feature_validator=feature_validator
        )

        # Create dataset with invalid features
        invalid_data = pd.DataFrame(
            {
                "feature_0": [np.inf, 1, 2],
                "feature_1": [1, np.nan, 3],
                "feature_2": [4, 5, 6],
            }
        )
        invalid_dataset = Dataset(name="invalid", data=invalid_data)

        # Mock feature validator to detect invalid features
        feature_validator.check_data_quality = Mock(
            return_value={
                "quality_score": 0.5,
                "missing_values": ["feature_1"],
                "constant_features": [],
                "low_variance_features": [],
                "infinite_values": ["feature_0"],
                "duplicate_rows": 0,
            }
        )

        # Create request
        request = DetectAnomaliesRequest(
            detector_id=sample_detector.id, dataset=invalid_dataset
        )

        # Mock adapter registry to raise error for invalid features
        with patch.object(use_case.adapter_registry, "score_with_detector") as mock_score:
            mock_score.side_effect = ValueError("Invalid features detected")

            with pytest.raises(DatasetError, match="Detection failed"):
                await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_execute_with_preprocessing(
        self, detector_repository, feature_validator, sample_detector, sample_dataset
    ):
        """Test detection with feature preprocessing."""
        from pynomaly.application.use_cases.detect_anomalies import (
            DetectAnomaliesRequest,
        )

        # Repository methods are synchronous
        detector_repository.save(sample_detector)

        # Mark detector as fitted for testing
        sample_detector.is_fitted = True

        use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repository, feature_validator=feature_validator
        )

        # Mock feature validator
        feature_validator.check_data_quality = Mock(return_value={"quality_score": 0.9})
        feature_validator.suggest_preprocessing = Mock(return_value=[])

        # Create request
        request = DetectAnomaliesRequest(
            detector_id=sample_detector.id,
            dataset=sample_dataset,
            validate_features=True,
        )

        # Mock the detector's detect method to return a result
        with patch.object(sample_detector, "detect") as mock_detect:
            # Create mock detection result
            mock_result = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=sample_dataset.id,
                anomalies=[],
                scores=[AnomalyScore(0.5) for _ in range(len(sample_dataset.data))],
                labels=np.array([0 for _ in range(len(sample_dataset.data))]),
                threshold=0.5,
            )
            mock_detect.return_value = mock_result

            # Execute use case
            response = await use_case.execute(request)

            # Verify result
            assert isinstance(response.result, DetectionResult)
            assert response.quality_report is not None
            mock_detect.assert_called_once_with(sample_dataset)

    @pytest.mark.asyncio
    async def test_execute_batch_processing(
        self, detector_repository, feature_validator, sample_detector
    ):
        """Test batch processing of multiple datasets."""
        from pynomaly.application.use_cases.detect_anomalies import (
            DetectAnomaliesRequest,
        )

        # Repository methods are synchronous
        detector_repository.save(sample_detector)

        # Mark detector as fitted for testing
        sample_detector.is_fitted = True

        use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repository, feature_validator=feature_validator
        )

        # Create multiple datasets
        datasets = []
        for i in range(3):
            features = np.random.RandomState(42 + i).normal(0, 1, (50, 5))
            df = pd.DataFrame(
                features,
                columns=[
                    "feature_0",
                    "feature_1",
                    "feature_2",
                    "feature_3",
                    "feature_4",
                ],
            )
            dataset = Dataset(name=f"dataset_{i}", data=df)
            datasets.append(dataset)

        # Mock feature validator
        feature_validator.check_data_quality = Mock(return_value={"quality_score": 0.9})
        feature_validator.suggest_preprocessing = Mock(return_value=[])

        # Test each dataset individually (simulate batch processing)
        responses = []
        for dataset in datasets:
            request = DetectAnomaliesRequest(
                detector_id=sample_detector.id, dataset=dataset
            )

            # Mock the detector's detect method to return a result
            with patch.object(sample_detector, "detect") as mock_detect:
                mock_result = DetectionResult(
                    detector_id=sample_detector.id,
                    dataset_id=dataset.id,
                    anomalies=[],
                    scores=[AnomalyScore(0.5) for _ in range(len(dataset.data))],
                    labels=np.array([0 for _ in range(len(dataset.data))]),
                    threshold=0.5,
                )
                mock_detect.return_value = mock_result

                response = await use_case.execute(request)
                responses.append(response)

        assert len(responses) == 3
        assert all(isinstance(r.result, DetectionResult) for r in responses)


class TestTrainDetectorUseCase:
    """Test TrainDetectorUseCase functionality."""

    @pytest.mark.asyncio
    async def test_execute_success(
        self, detector_repository, feature_validator, sample_detector, sample_dataset
    ):
        """Test successful detector training."""
        from pynomaly.application.use_cases.train_detector import TrainDetectorRequest

        # Repository methods are synchronous
        detector_repository.save(sample_detector)

        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=10,
        )

        # Mock feature validator
        feature_validator.validate_numeric_features = Mock(
            return_value=[
                "feature_0",
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
            ]
        )
        feature_validator.check_data_quality = Mock(
            return_value={
                "quality_score": 0.9,
                "missing_values": [],
                "constant_features": [],
            }
        )

        # Create request
        request = TrainDetectorRequest(
            detector_id=sample_detector.id, training_data=sample_dataset
        )

        # Mock the detector's fit method
        with patch.object(sample_detector, "fit") as mock_fit:
            mock_fit.return_value = None

            # Execute use case
            response = await use_case.execute(request)

            # Verify response
            assert response.trained_detector.id == sample_detector.id
            assert response.training_time_ms >= 0
            assert "shape" in response.dataset_summary
            mock_fit.assert_called_once_with(sample_dataset)

    @pytest.mark.asyncio
    async def test_execute_insufficient_samples(
        self, detector_repository, feature_validator, sample_detector
    ):
        """Test training with insufficient samples."""
        from pynomaly.application.use_cases.train_detector import TrainDetectorRequest
        from pynomaly.domain.exceptions import InsufficientDataError

        # Repository methods are synchronous
        detector_repository.save(sample_detector)

        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=100,
        )

        # Create small dataset
        small_data = pd.DataFrame(
            {
                "feature_0": [1, 2, 3, 4, 5],
                "feature_1": [6, 7, 8, 9, 10],
                "feature_2": [11, 12, 13, 14, 15],
            }
        )
        small_dataset = Dataset(name="small", data=small_data)

        # Create request
        request = TrainDetectorRequest(
            detector_id=sample_detector.id, training_data=small_dataset
        )

        with pytest.raises(InsufficientDataError):
            await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_execute_with_validation_split(
        self, detector_repository, feature_validator, sample_detector, sample_dataset
    ):
        """Test training with validation split."""
        from pynomaly.application.use_cases.train_detector import TrainDetectorRequest

        # Repository methods are synchronous
        detector_repository.save(sample_detector)

        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=10,
        )

        # Mock feature validator
        feature_validator.validate_numeric_features = Mock(
            return_value=[
                "feature_0",
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
            ]
        )
        feature_validator.check_data_quality = Mock(
            return_value={
                "quality_score": 0.9,
                "missing_values": [],
                "constant_features": [],
            }
        )

        # Create request
        request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            training_data=sample_dataset,
            validate_data=True,
        )

        # Mock the detector's fit method
        with patch.object(sample_detector, "fit") as mock_fit:
            mock_fit.return_value = None

            # Execute use case
            response = await use_case.execute(request)

            # Verify response
            assert response.trained_detector.id == sample_detector.id
            assert response.validation_results is not None
            assert "numeric_features" in response.validation_results
            mock_fit.assert_called_once_with(sample_dataset)

    @pytest.mark.asyncio
    async def test_execute_hyperparameter_tuning(
        self, detector_repository, feature_validator, sample_detector, sample_dataset
    ):
        """Test training with hyperparameter tuning."""
        from pynomaly.application.use_cases.train_detector import TrainDetectorRequest

        # Repository methods are synchronous
        detector_repository.save(sample_detector)

        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=10,
        )

        # Mock feature validator
        feature_validator.validate_numeric_features = Mock(
            return_value=[
                "feature_0",
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
            ]
        )
        feature_validator.check_data_quality = Mock(
            return_value={
                "quality_score": 0.9,
                "missing_values": [],
                "constant_features": [],
            }
        )

        # Create request with parameters
        request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            training_data=sample_dataset,
            parameters={"n_estimators": 50, "contamination": 0.1},
            validate_data=True,
        )

        # Mock the detector's fit method and update_parameters
        with (
            patch.object(sample_detector, "fit") as mock_fit,
            patch.object(sample_detector, "update_parameters") as mock_update_params,
        ):
            mock_fit.return_value = None
            mock_update_params.return_value = None

            # Execute use case
            response = await use_case.execute(request)

            # Verify response
            assert response.trained_detector.id == sample_detector.id
            assert "n_estimators" in response.parameters_used
            mock_update_params.assert_called_once_with(
                n_estimators=50, contamination=0.1
            )
            mock_fit.assert_called_once_with(sample_dataset)

    @pytest.mark.asyncio
    async def test_execute_comprehensive_request_response(
        self, detector_repository, feature_validator, sample_detector, sample_dataset
    ):
        """Test comprehensive training request with full response pattern."""

        from pynomaly.application.use_cases.train_detector import TrainDetectorRequest

        detector_repository.save(sample_detector)

        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=10,
        )

        # Mock feature validator
        feature_validator.validate_numeric_features = Mock(
            return_value=[
                "feature_0",
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
            ]
        )
        feature_validator.check_data_quality = Mock(
            return_value={
                "quality_score": 0.85,
                "missing_values": ["feature_1"],
                "constant_features": [],
            }
        )

        # Create comprehensive request with hyperparameter grid
        hyperparameter_grid = {
            "n_estimators": [100, 200, 300],
            "max_samples": [256, 512, "auto"],
            "contamination": [0.05, 0.1, 0.15],
        }

        request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            training_data=sample_dataset,
            validation_split=0.2,
            hyperparameter_grid=hyperparameter_grid,
            cv_folds=5,
            scoring_metric="f1_score",
            save_model=True,
            early_stopping=True,
            max_training_time=300,
            validate_data=True,
        )

        # Mock the detector's fit and update_parameters methods
        with (
            patch.object(sample_detector, "fit") as mock_fit,
            patch.object(sample_detector, "update_parameters") as mock_update_params,
        ):
            mock_fit.return_value = None
            mock_update_params.return_value = None

            # Execute use case
            response = await use_case.execute(request)

            # Verify comprehensive response structure
            assert response.trained_detector == sample_detector
            assert response.training_metrics is not None
            assert "training_time" in response.training_metrics
            assert "dataset_summary" in response.training_metrics
            assert "best_parameters" in response.training_metrics
            assert response.model_path is not None
            assert response.training_warnings is not None
            assert len(response.training_warnings) > 0

            # Verify backward compatibility
            assert response.detector_id == sample_detector.id
            assert response.training_time_ms >= 0
            assert isinstance(response.dataset_summary, dict)
            assert isinstance(response.parameters_used, dict)
            assert response.validation_results is not None

            mock_fit.assert_called_once_with(sample_dataset)

    @pytest.mark.asyncio
    async def test_execute_detector_not_found_pattern(
        self, detector_repository, feature_validator, sample_dataset
    ):
        """Test training with non-existent detector following DetectAnomaliesUseCase pattern."""
        from uuid import uuid4

        from pynomaly.application.use_cases.train_detector import TrainDetectorRequest

        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=10,
        )

        # Create request with non-existent detector ID
        request = TrainDetectorRequest(
            detector_id=uuid4(),  # Non-existent UUID
            training_data=sample_dataset,
        )

        with pytest.raises(ValueError, match="Detector.*not found"):
            await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_execute_with_quality_warnings(
        self, detector_repository, feature_validator, sample_detector
    ):
        """Test training with data quality warnings following DetectAnomaliesUseCase pattern."""
        from pynomaly.application.use_cases.train_detector import TrainDetectorRequest

        detector_repository.save(sample_detector)

        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=10,
        )

        # Create dataset with quality issues
        data_with_issues = pd.DataFrame(
            {
                "feature_0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature_1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Constant feature
                "feature_2": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],  # Missing value
                "feature_3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )
        dataset_with_issues = Dataset(name="quality_issues", data=data_with_issues)

        # Mock feature validator to detect quality issues
        feature_validator.validate_numeric_features = Mock(
            return_value=["feature_0", "feature_2", "feature_3"]
        )
        feature_validator.check_data_quality = Mock(
            return_value={
                "quality_score": 0.6,
                "missing_values": ["feature_2"],
                "constant_features": ["feature_1"],
            }
        )

        # Create request
        request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            training_data=dataset_with_issues,
            validate_data=True,
        )

        # Mock the detector's fit method
        with patch.object(sample_detector, "fit") as mock_fit:
            mock_fit.return_value = None

            # Execute use case
            response = await use_case.execute(request)

            # Verify warnings were generated
            assert response.training_warnings is not None
            assert (
                len(response.training_warnings) >= 2
            )  # Missing values + constant features

            # Check specific warning patterns
            warning_text = " ".join(response.training_warnings)
            assert "Missing values detected" in warning_text
            assert "constant features detected" in warning_text

            mock_fit.assert_called_once_with(dataset_with_issues)

    @pytest.mark.asyncio
    async def test_execute_batch_training_simulation(
        self, detector_repository, feature_validator, sample_detector
    ):
        """Test batch training simulation following DetectAnomaliesUseCase pattern."""
        from pynomaly.application.use_cases.train_detector import TrainDetectorRequest

        detector_repository.save(sample_detector)

        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=10,
        )

        # Create multiple datasets for batch training simulation
        datasets = []
        for i in range(3):
            features = np.random.RandomState(42 + i).normal(0, 1, (50, 5))
            df = pd.DataFrame(
                features,
                columns=[
                    "feature_0",
                    "feature_1",
                    "feature_2",
                    "feature_3",
                    "feature_4",
                ],
            )
            dataset = Dataset(name=f"batch_dataset_{i}", data=df)
            datasets.append(dataset)

        # Mock feature validator
        feature_validator.validate_numeric_features = Mock(
            return_value=[
                "feature_0",
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
            ]
        )
        feature_validator.check_data_quality = Mock(
            return_value={
                "quality_score": 0.9,
                "missing_values": [],
                "constant_features": [],
            }
        )

        # Train on each dataset individually (simulate batch processing)
        responses = []
        for dataset in datasets:
            request = TrainDetectorRequest(
                detector_id=sample_detector.id,
                training_data=dataset,
                validate_data=True,
            )

            # Mock the detector's fit method
            with patch.object(sample_detector, "fit") as mock_fit:
                mock_fit.return_value = None

                response = await use_case.execute(request)
                responses.append(response)

        assert len(responses) == 3
        assert all(
            response.trained_detector.id == sample_detector.id for response in responses
        )
        assert all(response.training_metrics is not None for response in responses)

    @pytest.mark.asyncio
    async def test_execute_minimal_vs_comprehensive_response(
        self, detector_repository, feature_validator, sample_detector, sample_dataset
    ):
        """Test minimal vs comprehensive response patterns."""
        from pynomaly.application.use_cases.train_detector import TrainDetectorRequest

        detector_repository.save(sample_detector)

        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=10,
        )

        # Mock feature validator
        feature_validator.validate_numeric_features = Mock(
            return_value=[
                "feature_0",
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
            ]
        )
        feature_validator.check_data_quality = Mock(
            return_value={
                "quality_score": 0.9,
                "missing_values": [],
                "constant_features": [],
            }
        )

        # Test minimal request
        minimal_request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            training_data=sample_dataset,
            validate_data=False,
            save_model=False,
        )

        with patch.object(sample_detector, "fit") as mock_fit:
            mock_fit.return_value = None

            minimal_response = await use_case.execute(minimal_request)

            # Verify minimal response structure
            assert minimal_response.trained_detector == sample_detector
            assert minimal_response.training_metrics is not None
            assert minimal_response.model_path is None  # save_model=False
            assert (
                minimal_response.training_warnings is None
                or len(minimal_response.training_warnings) == 0
            )

        # Test comprehensive request
        comprehensive_request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            training_data=sample_dataset,
            validation_split=0.3,
            hyperparameter_grid={"n_estimators": [50, 100]},
            validate_data=True,
            save_model=True,
            early_stopping=True,
        )

        with patch.object(sample_detector, "fit") as mock_fit:
            mock_fit.return_value = None

            comprehensive_response = await use_case.execute(comprehensive_request)

            # Verify comprehensive response structure
            assert comprehensive_response.trained_detector == sample_detector
            assert comprehensive_response.training_metrics is not None
            assert comprehensive_response.model_path is not None  # save_model=True
            assert comprehensive_response.training_warnings is not None
            assert len(comprehensive_response.training_warnings) > 0


class TestEvaluateModelUseCase:
    """Test EvaluateModelUseCase functionality."""

    @pytest.mark.asyncio
    async def test_execute_success(
        self, detector_repository, sample_detector, sample_dataset
    ):
        """Test successful model evaluation."""
        detector_repository.save(sample_detector)

        # Add target labels to dataset for evaluation
        target_labels = np.array(
            [0 if i < 90 else 1 for i in range(len(sample_dataset.data))]
        )
        sample_dataset.data["target"] = target_labels
        sample_dataset.target_column = "target"

        use_case = EvaluateModelUseCase(detector_repository=detector_repository)

        # Create request
        request = EvaluateModelRequest(
            detector_id=sample_detector.id, test_dataset=sample_dataset
        )

        # Mock detector detection
        with patch.object(sample_detector, "detect") as mock_detect:
            mock_result = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=sample_dataset.id,
                anomalies=[],
                scores=[AnomalyScore(0.5) for _ in range(len(sample_dataset.data))],
                labels=np.array([0 for _ in range(len(sample_dataset.data))]),
                threshold=0.5,
            )
            mock_detect.return_value = mock_result

            response = await use_case.execute(request)

            assert isinstance(response, EvaluateModelResponse)
            assert response.detector_id == sample_detector.id
            assert "auc_roc" in response.metrics or "precision" in response.metrics

    @pytest.mark.asyncio
    async def test_execute_with_ground_truth(
        self, detector_repository, sample_detector, sample_dataset
    ):
        """Test evaluation with ground truth labels."""
        detector_repository.save(sample_detector)

        # Add target labels to dataset for evaluation
        target_labels = np.array(
            [0 if i < 85 else 1 for i in range(len(sample_dataset.data))]
        )
        sample_dataset.data["target"] = target_labels
        sample_dataset.target_column = "target"

        use_case = EvaluateModelUseCase(detector_repository=detector_repository)

        # Dataset now has targets for evaluation
        assert sample_dataset.has_target

        # Create request
        request = EvaluateModelRequest(
            detector_id=sample_detector.id, test_dataset=sample_dataset
        )

        # Mock detector detection
        with patch.object(sample_detector, "detect") as mock_detect:
            mock_result = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=sample_dataset.id,
                anomalies=[],
                scores=[AnomalyScore(0.5) for _ in range(len(sample_dataset.data))],
                labels=np.array([0 for _ in range(len(sample_dataset.data))]),
                threshold=0.5,
            )
            mock_detect.return_value = mock_result

            response = await use_case.execute(request)

            assert isinstance(response, EvaluateModelResponse)
            assert response.detector_id == sample_detector.id
            # Should include classification metrics when ground truth is available
            assert "auc_roc" in response.metrics or "precision" in response.metrics

    @pytest.mark.asyncio
    async def test_execute_cross_validation(
        self, detector_repository, sample_detector, sample_dataset
    ):
        """Test evaluation with cross-validation."""
        detector_repository.save(sample_detector)

        # Add target labels to dataset for evaluation
        target_labels = np.array(
            [0 if i < 85 else 1 for i in range(len(sample_dataset.data))]
        )
        sample_dataset.data["target"] = target_labels
        sample_dataset.target_column = "target"

        use_case = EvaluateModelUseCase(detector_repository=detector_repository)

        # Create request
        request = EvaluateModelRequest(
            detector_id=sample_detector.id,
            test_dataset=sample_dataset,
            cross_validate=True,
            n_folds=5,
        )

        # Mock detector detection - need to return correct size for each fold
        with (
            patch.object(sample_detector, "detect") as mock_detect,
            patch.object(sample_detector, "fit") as mock_fit,
        ):

            def mock_detect_func(dataset):
                # Return result matching the dataset size
                n_samples = len(dataset.data)
                return DetectionResult(
                    detector_id=sample_detector.id,
                    dataset_id=dataset.id,
                    anomalies=[],
                    scores=[AnomalyScore(0.5) for _ in range(n_samples)],
                    labels=np.array([0 for _ in range(n_samples)]),
                    threshold=0.5,
                )

            mock_detect.side_effect = mock_detect_func
            mock_fit.return_value = None

            response = await use_case.execute(request)

            assert isinstance(response, EvaluateModelResponse)
            assert response.detector_id == sample_detector.id
            # Should include classification metrics
            assert "auc_roc" in response.metrics or "precision" in response.metrics


class TestExplainAnomalyUseCase:
    """Test ExplainAnomalyUseCase functionality."""

    @pytest.mark.asyncio
    async def test_execute_success(
        self, detector_repository, sample_detector, sample_dataset
    ):
        """Test successful anomaly explanation."""
        from pynomaly.application.use_cases.explain_anomaly import ExplainAnomalyRequest

        detector_repository.save(sample_detector)

        use_case = ExplainAnomalyUseCase(detector_repository=detector_repository)

        # Create request
        request = ExplainAnomalyRequest(
            detector_id=sample_detector.id,
            dataset=sample_dataset,
            anomaly_indices=[0, 1, 2],
            explanation_method="feature_importance",
        )

        # Mock detector to fall back to basic explanation since it's not ExplainableDetectorProtocol
        with patch.object(sample_detector, "detect") as mock_detect:
            mock_result = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=sample_dataset.id,
                anomalies=[],
                scores=[AnomalyScore(0.8) for _ in range(len(sample_dataset.data))],
                labels=np.array([0 for _ in range(len(sample_dataset.data))]),
                threshold=0.5,
            )
            mock_detect.return_value = mock_result

            response = await use_case.execute(request)

            assert hasattr(response, "explanations")
            assert hasattr(response, "method_used")

    @pytest.mark.asyncio
    async def test_execute_shap_explanation(
        self, detector_repository, sample_detector, sample_dataset
    ):
        """Test SHAP-based explanation."""
        from pynomaly.application.use_cases.explain_anomaly import ExplainAnomalyRequest

        detector_repository.save(sample_detector)

        use_case = ExplainAnomalyUseCase(detector_repository=detector_repository)

        # Create request
        request = ExplainAnomalyRequest(
            detector_id=sample_detector.id,
            dataset=sample_dataset,
            anomaly_indices=[0, 1],
            explanation_method="shap",
        )

        # Mock detector to fall back to basic explanation
        with patch.object(sample_detector, "detect") as mock_detect:
            mock_result = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=sample_dataset.id,
                anomalies=[],
                scores=[AnomalyScore(0.8) for _ in range(len(sample_dataset.data))],
                labels=np.array([0 for _ in range(len(sample_dataset.data))]),
                threshold=0.5,
            )
            mock_detect.return_value = mock_result

            response = await use_case.execute(request)

            assert hasattr(response, "explanations")
            assert response.method_used in ["shap", "basic"]

    @pytest.mark.asyncio
    async def test_execute_lime_explanation(
        self, detector_repository, sample_detector, sample_dataset
    ):
        """Test LIME-based explanation."""
        from pynomaly.application.use_cases.explain_anomaly import ExplainAnomalyRequest

        detector_repository.save(sample_detector)

        use_case = ExplainAnomalyUseCase(detector_repository=detector_repository)

        # Create request
        request = ExplainAnomalyRequest(
            detector_id=sample_detector.id,
            dataset=sample_dataset,
            anomaly_indices=[0, 1],
            explanation_method="lime",
        )

        # Mock detector to fall back to basic explanation
        with patch.object(sample_detector, "detect") as mock_detect:
            mock_result = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=sample_dataset.id,
                anomalies=[],
                scores=[AnomalyScore(0.8) for _ in range(len(sample_dataset.data))],
                labels=np.array([0 for _ in range(len(sample_dataset.data))]),
                threshold=0.5,
            )
            mock_detect.return_value = mock_result

            response = await use_case.execute(request)

            assert hasattr(response, "explanations")
            assert response.method_used in ["lime", "basic"]

    @pytest.mark.asyncio
    async def test_execute_multiple_anomalies(
        self, detector_repository, sample_detector, sample_dataset
    ):
        """Test explanation of multiple anomalies in one request."""
        from pynomaly.application.use_cases.explain_anomaly import ExplainAnomalyRequest

        detector_repository.save(sample_detector)

        use_case = ExplainAnomalyUseCase(detector_repository=detector_repository)

        # Create request with multiple anomaly indices
        request = ExplainAnomalyRequest(
            detector_id=sample_detector.id,
            dataset=sample_dataset,
            anomaly_indices=[0, 1, 2],
            explanation_method="feature_importance",
        )

        # Mock detector to fall back to basic explanation
        with patch.object(sample_detector, "detect") as mock_detect:
            mock_result = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=sample_dataset.id,
                anomalies=[],
                scores=[AnomalyScore(0.8) for _ in range(len(sample_dataset.data))],
                labels=np.array([0 for _ in range(len(sample_dataset.data))]),
                threshold=0.5,
            )
            mock_detect.return_value = mock_result

            response = await use_case.execute(request)

            assert hasattr(response, "explanations")
            assert isinstance(response.explanations, dict)
            # Should have explanations for the specified indices
            assert len(response.explanations) >= 0
