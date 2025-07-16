"""Tests for enhanced detection pipeline service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pynomaly.application.services.algorithm_recommendation_service import (
    AlgorithmRecommendation,
)
from pynomaly.application.services.detection_pipeline_service import (
    DetectionPipelineService,
)
from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.exceptions import AdapterError, AlgorithmNotFoundError
from pynomaly.domain.value_objects import AnomalyScore


class TestDetectionPipelineService:
    """Test DetectionPipelineService class."""

    def test_detection_pipeline_service_initialization(self):
        """Test DetectionPipelineService initialization."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        assert service.detector_repository == mock_detector_repo
        assert service.result_repository == mock_result_repo
        assert service.adapter_registry is not None
        assert service.logger is not None

    def test_detection_pipeline_service_with_custom_adapter_registry(self):
        """Test DetectionPipelineService with custom adapter registry."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()
        mock_adapter_registry = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo,
            adapter_registry=mock_adapter_registry
        )

        assert service.adapter_registry == mock_adapter_registry

    @pytest.mark.asyncio
    async def test_run_detection_pipeline_success(self):
        """Test successful detection pipeline run."""
        mock_detector_repo = AsyncMock()
        mock_result_repo = AsyncMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock recommendations
        recommendations = [
            AlgorithmRecommendation(
                algorithm="IsolationForest",
                confidence=0.9,
                hyperparams={"n_estimators": 100, "contamination": 0.1}
            ),
            AlgorithmRecommendation(
                algorithm="LocalOutlierFactor",
                confidence=0.8,
                hyperparams={"n_neighbors": 20, "contamination": 0.1}
            )
        ]

        # Mock execute_algorithm
        mock_result1 = DetectionResult(
            detector_id="detector1",
            dataset_name="test_dataset",
            scores=[AnomalyScore(value=0.1), AnomalyScore(value=0.9)],
            labels=[0, 1],
            threshold=0.5,
            execution_time_ms=100,
            anomalies=[],
            metadata={}
        )

        mock_result2 = DetectionResult(
            detector_id="detector2",
            dataset_name="test_dataset",
            scores=[AnomalyScore(value=0.2), AnomalyScore(value=0.8)],
            labels=[0, 1],
            threshold=0.5,
            execution_time_ms=150,
            anomalies=[],
            metadata={}
        )

        service._execute_algorithm = AsyncMock(side_effect=[mock_result1, mock_result2])

        # Run pipeline
        result = await service.run_detection_pipeline(
            dataset=dataset,
            recommendations=recommendations,
            auto_tune=False,
            save_results=False,
            verbose=False
        )

        # Verify results
        assert result["dataset_name"] == "test_dataset"
        assert len(result["algorithms_used"]) == 2
        assert "IsolationForest" in result["algorithms_used"]
        assert "LocalOutlierFactor" in result["algorithms_used"]
        assert result["best_algorithm"] is not None
        assert result["best_score"] > 0
        assert len(result["errors"]) == 0

        # Verify execute_algorithm was called correctly
        assert service._execute_algorithm.call_count == 2

    @pytest.mark.asyncio
    async def test_run_detection_pipeline_with_errors(self):
        """Test detection pipeline with algorithm errors."""
        mock_detector_repo = AsyncMock()
        mock_result_repo = AsyncMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock recommendations
        recommendations = [
            AlgorithmRecommendation(
                algorithm="IsolationForest",
                confidence=0.9,
                hyperparams={"n_estimators": 100}
            ),
            AlgorithmRecommendation(
                algorithm="LocalOutlierFactor",
                confidence=0.8,
                hyperparams={"n_neighbors": 20}
            )
        ]

        # Mock execute_algorithm - first succeeds, second fails
        mock_result = DetectionResult(
            detector_id="detector1",
            dataset_name="test_dataset",
            scores=[AnomalyScore(value=0.1), AnomalyScore(value=0.9)],
            labels=[0, 1],
            threshold=0.5,
            execution_time_ms=100,
            anomalies=[],
            metadata={}
        )

        service._execute_algorithm = AsyncMock(side_effect=[
            mock_result,
            Exception("Algorithm failed")
        ])

        # Run pipeline
        result = await service.run_detection_pipeline(
            dataset=dataset,
            recommendations=recommendations,
            auto_tune=False,
            save_results=False,
            verbose=False
        )

        # Verify results
        assert result["dataset_name"] == "test_dataset"
        assert len(result["algorithms_used"]) == 1
        assert "IsolationForest" in result["algorithms_used"]
        assert "LocalOutlierFactor" not in result["algorithms_used"]
        assert len(result["errors"]) == 1
        assert "LocalOutlierFactor" in result["errors"]

    @pytest.mark.asyncio
    async def test_run_detection_pipeline_with_ensemble(self):
        """Test detection pipeline with ensemble creation."""
        mock_detector_repo = AsyncMock()
        mock_result_repo = AsyncMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock recommendations
        recommendations = [
            AlgorithmRecommendation(
                algorithm="IsolationForest",
                confidence=0.9,
                hyperparams={"n_estimators": 100}
            ),
            AlgorithmRecommendation(
                algorithm="LocalOutlierFactor",
                confidence=0.8,
                hyperparams={"n_neighbors": 20}
            )
        ]

        # Mock execute_algorithm
        mock_result1 = DetectionResult(
            detector_id="detector1",
            dataset_name="test_dataset",
            scores=[AnomalyScore(value=0.1), AnomalyScore(value=0.9)],
            labels=[0, 1],
            threshold=0.5,
            execution_time_ms=100,
            anomalies=[],
            metadata={}
        )

        mock_result2 = DetectionResult(
            detector_id="detector2",
            dataset_name="test_dataset",
            scores=[AnomalyScore(value=0.2), AnomalyScore(value=0.8)],
            labels=[0, 1],
            threshold=0.5,
            execution_time_ms=150,
            anomalies=[],
            metadata={}
        )

        service._execute_algorithm = AsyncMock(side_effect=[mock_result1, mock_result2])

        # Mock ensemble creation
        mock_ensemble = DetectionResult(
            detector_id="ensemble",
            dataset_name="test_dataset",
            scores=[AnomalyScore(value=0.15), AnomalyScore(value=0.85)],
            labels=[0, 1],
            threshold=0.5,
            execution_time_ms=250,
            anomalies=[],
            metadata={"ensemble_type": "simple_average"}
        )

        service._create_ensemble_result = MagicMock(return_value=mock_ensemble)

        # Run pipeline
        result = await service.run_detection_pipeline(
            dataset=dataset,
            recommendations=recommendations,
            auto_tune=False,
            save_results=False,
            verbose=False
        )

        # Verify ensemble was created
        assert result["ensemble_result"] is not None
        assert result["ensemble_result"] == mock_ensemble
        service._create_ensemble_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_detection_pipeline_with_save_results(self):
        """Test detection pipeline with result saving."""
        mock_detector_repo = AsyncMock()
        mock_result_repo = AsyncMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock recommendations
        recommendations = [
            AlgorithmRecommendation(
                algorithm="IsolationForest",
                confidence=0.9,
                hyperparams={"n_estimators": 100}
            )
        ]

        # Mock execute_algorithm
        mock_result = DetectionResult(
            detector_id="detector1",
            dataset_name="test_dataset",
            scores=[AnomalyScore(value=0.1), AnomalyScore(value=0.9)],
            labels=[0, 1],
            threshold=0.5,
            execution_time_ms=100,
            anomalies=[],
            metadata={}
        )

        service._execute_algorithm = AsyncMock(return_value=mock_result)

        # Run pipeline with save_results=True
        result = await service.run_detection_pipeline(
            dataset=dataset,
            recommendations=recommendations,
            auto_tune=False,
            save_results=True,
            verbose=False
        )

        # Verify result was saved
        mock_result_repo.save.assert_called_once_with(mock_result)

    @pytest.mark.asyncio
    async def test_execute_algorithm_success(self):
        """Test successful algorithm execution."""
        mock_detector_repo = AsyncMock()
        mock_result_repo = AsyncMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock recommendation
        recommendation = AlgorithmRecommendation(
            algorithm="IsolationForest",
            confidence=0.9,
            hyperparams={"n_estimators": 100}
        )

        # Mock detector
        mock_detector = MagicMock()
        mock_detection_result = DetectionResult(
            detector_id="detector1",
            dataset_name="test_dataset",
            scores=[AnomalyScore(value=0.1), AnomalyScore(value=0.9)],
            labels=[0, 1],
            threshold=0.5,
            execution_time_ms=0,  # Will be updated
            anomalies=[],
            metadata={}
        )
        mock_detector.fit_detect.return_value = mock_detection_result

        service._create_detector = MagicMock(return_value=mock_detector)
        service._auto_tune_detector = AsyncMock(return_value=mock_detector)

        # Execute algorithm
        result = await service._execute_algorithm(
            dataset=dataset,
            recommendation=recommendation,
            auto_tune=True,
            verbose=False
        )

        # Verify results
        assert result == mock_detection_result
        assert result.execution_time_ms > 0
        service._create_detector.assert_called_once_with(recommendation)
        service._auto_tune_detector.assert_called_once_with(mock_detector, dataset, False)
        mock_detector.fit_detect.assert_called_once_with(dataset)
        mock_detector_repo.save.assert_called_once_with(mock_detector)

    @pytest.mark.asyncio
    async def test_execute_algorithm_without_auto_tune(self):
        """Test algorithm execution without auto-tuning."""
        mock_detector_repo = AsyncMock()
        mock_result_repo = AsyncMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock recommendation
        recommendation = AlgorithmRecommendation(
            algorithm="IsolationForest",
            confidence=0.9,
            hyperparams={"n_estimators": 100}
        )

        # Mock detector
        mock_detector = MagicMock()
        mock_detection_result = DetectionResult(
            detector_id="detector1",
            dataset_name="test_dataset",
            scores=[AnomalyScore(value=0.1), AnomalyScore(value=0.9)],
            labels=[0, 1],
            threshold=0.5,
            execution_time_ms=0,
            anomalies=[],
            metadata={}
        )
        mock_detector.fit_detect.return_value = mock_detection_result

        service._create_detector = MagicMock(return_value=mock_detector)
        service._auto_tune_detector = AsyncMock(return_value=mock_detector)

        # Execute algorithm without auto-tuning
        result = await service._execute_algorithm(
            dataset=dataset,
            recommendation=recommendation,
            auto_tune=False,
            verbose=False
        )

        # Verify auto-tuning was not called
        service._auto_tune_detector.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_algorithm_failure(self):
        """Test algorithm execution failure."""
        mock_detector_repo = AsyncMock()
        mock_result_repo = AsyncMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock recommendation
        recommendation = AlgorithmRecommendation(
            algorithm="IsolationForest",
            confidence=0.9,
            hyperparams={"n_estimators": 100}
        )

        # Mock detector creation failure
        service._create_detector = MagicMock(side_effect=Exception("Detector creation failed"))

        # Execute algorithm
        with pytest.raises(AdapterError, match="Failed to execute IsolationForest"):
            await service._execute_algorithm(
                dataset=dataset,
                recommendation=recommendation,
                auto_tune=False,
                verbose=False
            )

    def test_create_detector_success(self):
        """Test successful detector creation."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock recommendation
        recommendation = AlgorithmRecommendation(
            algorithm="IsolationForest",
            confidence=0.9,
            hyperparams={"n_estimators": 100, "contamination": 0.1}
        )

        # Mock adapter registry
        mock_adapter_class = MagicMock()
        mock_detector = MagicMock()
        mock_adapter_class.return_value = mock_detector

        service.adapter_registry.get_adapter = MagicMock(return_value=mock_adapter_class)

        # Create detector
        result = service._create_detector(recommendation)

        # Verify detector creation
        assert result == mock_detector
        service.adapter_registry.get_adapter.assert_called_once_with("IsolationForest")
        mock_adapter_class.assert_called_once_with(
            algorithm_name="IsolationForest",
            n_estimators=100,
            contamination=0.1
        )

    def test_create_detector_failure(self):
        """Test detector creation failure."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock recommendation
        recommendation = AlgorithmRecommendation(
            algorithm="UnknownAlgorithm",
            confidence=0.9,
            hyperparams={}
        )

        # Mock adapter registry failure
        service.adapter_registry.get_adapter = MagicMock(side_effect=Exception("Adapter not found"))

        # Create detector
        with pytest.raises(AlgorithmNotFoundError, match="Failed to create detector for UnknownAlgorithm"):
            service._create_detector(recommendation)

    @pytest.mark.asyncio
    async def test_auto_tune_detector_success(self):
        """Test successful detector auto-tuning."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.algorithm_name = "IsolationForest"
        mock_detector.get_params.return_value = {"n_estimators": 100}
        mock_detector.set_params.return_value = None

        # Mock grid search
        service._get_parameter_spaces = MagicMock(return_value={
            "n_estimators": [50, 100, 200],
            "contamination": [0.05, 0.1, 0.15]
        })
        service._grid_search_tuning = AsyncMock(return_value={"n_estimators": 200, "contamination": 0.05})

        # Auto-tune detector
        result = await service._auto_tune_detector(mock_detector, dataset, verbose=False)

        # Verify tuning
        assert result == mock_detector
        service._get_parameter_spaces.assert_called_once_with("IsolationForest")
        service._grid_search_tuning.assert_called_once()
        mock_detector.set_params.assert_called_once_with(n_estimators=200, contamination=0.05)

    @pytest.mark.asyncio
    async def test_auto_tune_detector_no_parameter_space(self):
        """Test auto-tuning when no parameter space is defined."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.algorithm_name = "UnknownAlgorithm"
        mock_detector.get_params.return_value = {}

        # Mock no parameter space
        service._get_parameter_spaces = MagicMock(return_value={})

        # Auto-tune detector
        result = await service._auto_tune_detector(mock_detector, dataset, verbose=False)

        # Verify no tuning occurred
        assert result == mock_detector
        mock_detector.set_params.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_tune_detector_failure(self):
        """Test auto-tuning failure."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.algorithm_name = "IsolationForest"
        mock_detector.get_params.side_effect = Exception("Get params failed")

        # Auto-tune detector
        result = await service._auto_tune_detector(mock_detector, dataset, verbose=False)

        # Verify original detector is returned on failure
        assert result == mock_detector

    def test_get_parameter_spaces(self):
        """Test getting parameter spaces for different algorithms."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Test known algorithms
        isolation_forest_params = service._get_parameter_spaces("IsolationForest")
        assert "n_estimators" in isolation_forest_params
        assert "contamination" in isolation_forest_params

        lof_params = service._get_parameter_spaces("LocalOutlierFactor")
        assert "n_neighbors" in lof_params
        assert "contamination" in lof_params

        svm_params = service._get_parameter_spaces("OneClassSVM")
        assert "kernel" in svm_params
        assert "gamma" in svm_params

        # Test unknown algorithm
        unknown_params = service._get_parameter_spaces("UnknownAlgorithm")
        assert unknown_params == {}

    @pytest.mark.asyncio
    async def test_grid_search_tuning(self):
        """Test grid search hyperparameter tuning."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.set_params.return_value = None

        # Mock parameter spaces
        param_spaces = {
            "n_estimators": [50, 100],
            "contamination": [0.1, 0.2]
        }

        # Mock performance evaluation
        service._evaluate_detector_performance = AsyncMock(side_effect=[0.5, 0.8, 0.6, 0.7])

        # Run grid search
        best_params = await service._grid_search_tuning(
            mock_detector, dataset, param_spaces, verbose=False
        )

        # Verify best parameters
        assert best_params == {"n_estimators": 100, "contamination": 0.1}
        assert service._evaluate_detector_performance.call_count == 4  # 2x2 combinations

    @pytest.mark.asyncio
    async def test_grid_search_tuning_with_limit(self):
        """Test grid search with combination limit."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.set_params.return_value = None

        # Mock parameter spaces with many combinations
        param_spaces = {
            "param1": [1, 2, 3, 4, 5],
            "param2": [1, 2, 3, 4, 5],
            "param3": [1, 2, 3, 4, 5]
        }  # 125 combinations

        # Mock performance evaluation
        service._evaluate_detector_performance = AsyncMock(return_value=0.5)

        # Run grid search
        with patch("random.sample") as mock_sample:
            mock_sample.return_value = [(1, 1, 1), (2, 2, 2)]  # Sample 2 combinations

            best_params = await service._grid_search_tuning(
                mock_detector, dataset, param_spaces, verbose=False
            )

            # Verify sampling was used
            mock_sample.assert_called_once()
            assert service._evaluate_detector_performance.call_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_detector_performance_with_labels(self):
        """Test detector performance evaluation with labels."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset with labels
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"],
            labels=[0, 1, 0]  # Ground truth labels
        )

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.fit.return_value = None
        mock_detector.score.return_value = [
            AnomalyScore(value=0.1),
            AnomalyScore(value=0.9),
            AnomalyScore(value=0.2)
        ]

        # Mock ROC AUC calculation
        with patch("sklearn.metrics.roc_auc_score") as mock_roc_auc:
            mock_roc_auc.return_value = 0.85

            score = await service._evaluate_detector_performance(mock_detector, dataset)

            assert score == 0.85
            mock_roc_auc.assert_called_once_with([0, 1, 0], [0.1, 0.9, 0.2])

    @pytest.mark.asyncio
    async def test_evaluate_detector_performance_without_labels(self):
        """Test detector performance evaluation without labels."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset without labels
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.fit.return_value = None
        mock_detector.score.return_value = [
            AnomalyScore(value=0.1),
            AnomalyScore(value=0.9),
            AnomalyScore(value=0.2)
        ]

        # Mock numpy variance calculation
        with patch("numpy.var") as mock_var:
            mock_var.return_value = 0.123

            score = await service._evaluate_detector_performance(mock_detector, dataset)

            assert score == 0.123
            mock_var.assert_called_once_with([0.1, 0.9, 0.2])

    @pytest.mark.asyncio
    async def test_evaluate_detector_performance_failure(self):
        """Test detector performance evaluation failure."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock detector failure
        mock_detector = MagicMock()
        mock_detector.fit.side_effect = Exception("Fit failed")

        score = await service._evaluate_detector_performance(mock_detector, dataset)

        # Should return 0.0 on failure
        assert score == 0.0

    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock detection result
        result = DetectionResult(
            detector_id="detector1",
            dataset_name="test_dataset",
            scores=[
                AnomalyScore(value=0.1),
                AnomalyScore(value=0.9),
                AnomalyScore(value=0.3),
                AnomalyScore(value=0.7)
            ],
            labels=[0, 1, 0, 1],
            threshold=0.5,
            execution_time_ms=100,
            anomalies=[],
            metadata={}
        )

        metrics = service._calculate_performance_metrics(result)

        # Verify metrics
        assert metrics["total_samples"] == 4
        assert metrics["anomaly_count"] == 2
        assert metrics["normal_count"] == 2
        assert metrics["anomaly_rate"] == 0.5
        assert metrics["mean_score"] == 0.5
        assert metrics["min_score"] == 0.1
        assert metrics["max_score"] == 0.9
        assert metrics["threshold"] == 0.5
        assert metrics["execution_time_ms"] == 100

    def test_calculate_performance_metrics_failure(self):
        """Test performance metrics calculation failure."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock detection result with invalid data
        result = MagicMock()
        result.labels = None  # Invalid labels

        metrics = service._calculate_performance_metrics(result)

        # Should return empty dict on failure
        assert metrics == {}

    def test_calculate_std(self):
        """Test standard deviation calculation."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Test with valid values
        std = service._calculate_std([1, 2, 3, 4, 5])
        assert std > 0

        # Test with single value
        std = service._calculate_std([5])
        assert std == 0.0

        # Test with empty list
        std = service._calculate_std([])
        assert std == 0.0

    def test_calculate_overall_performance_score(self):
        """Test overall performance score calculation."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock detection result
        result = DetectionResult(
            detector_id="detector1",
            dataset_name="test_dataset",
            scores=[
                AnomalyScore(value=0.1),
                AnomalyScore(value=0.9),
                AnomalyScore(value=0.3),
                AnomalyScore(value=0.7)
            ],
            labels=[0, 1, 0, 1],
            threshold=0.5,
            execution_time_ms=5000,  # 5 seconds
            anomalies=[],
            metadata={"key": "value"}
        )

        score = service._calculate_overall_performance_score(result)

        # Verify score is between 0 and 1
        assert 0 <= score <= 1
        assert score > 0  # Should be positive for valid result

    def test_calculate_overall_performance_score_failure(self):
        """Test overall performance score calculation failure."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock detection result with invalid data
        result = MagicMock()
        result.scores = None  # Invalid scores

        score = service._calculate_overall_performance_score(result)

        # Should return 0.0 on failure
        assert score == 0.0

    def test_create_ensemble_result(self):
        """Test ensemble result creation."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock results
        result1 = DetectionResult(
            detector_id="detector1",
            dataset_name="test_dataset",
            scores=[AnomalyScore(value=0.1), AnomalyScore(value=0.9)],
            labels=[0, 1],
            threshold=0.5,
            execution_time_ms=100,
            anomalies=[],
            metadata={}
        )

        result2 = DetectionResult(
            detector_id="detector2",
            dataset_name="test_dataset",
            scores=[AnomalyScore(value=0.2), AnomalyScore(value=0.8)],
            labels=[0, 1],
            threshold=0.6,
            execution_time_ms=150,
            anomalies=[],
            metadata={}
        )

        results = {"IsolationForest": result1, "LocalOutlierFactor": result2}
        recommendations = []

        ensemble_result = service._create_ensemble_result(dataset, results, recommendations)

        # Verify ensemble result
        assert ensemble_result.dataset_name == "test_dataset"
        assert len(ensemble_result.scores) == 2
        assert ensemble_result.scores[0].value == 0.15  # Average of 0.1 and 0.2
        assert ensemble_result.scores[1].value == 0.85  # Average of 0.9 and 0.8
        assert ensemble_result.threshold == 0.55  # Average of 0.5 and 0.6
        assert ensemble_result.execution_time_ms == 250  # Sum of 100 and 150
        assert "ensemble_type" in ensemble_result.metadata

    def test_create_ensemble_result_failure(self):
        """Test ensemble result creation failure."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock dataset
        dataset = Dataset(
            name="test_dataset",
            data=[[1, 2], [3, 4], [5, 6]],
            feature_names=["feature1", "feature2"]
        )

        # Mock invalid results
        results = {"algorithm": "invalid_result"}
        recommendations = []

        with pytest.raises(AdapterError, match="Ensemble creation failed"):
            service._create_ensemble_result(dataset, results, recommendations)

    def test_get_pipeline_summary(self):
        """Test pipeline summary generation."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock pipeline results
        pipeline_results = {
            "dataset_name": "test_dataset",
            "algorithms_used": ["IsolationForest", "LocalOutlierFactor"],
            "results": {
                "IsolationForest": DetectionResult(
                    detector_id="detector1",
                    dataset_name="test_dataset",
                    scores=[AnomalyScore(value=0.1), AnomalyScore(value=0.9)],
                    labels=[0, 1],
                    threshold=0.5,
                    execution_time_ms=100,
                    anomalies=[],
                    metadata={}
                )
            },
            "performance_metrics": {
                "IsolationForest": {
                    "anomaly_count": 1,
                    "execution_time_ms": 100
                }
            },
            "best_algorithm": "IsolationForest",
            "best_score": 0.85,
            "errors": {"LocalOutlierFactor": "Algorithm failed"},
            "ensemble_result": None
        }

        summary = service.get_pipeline_summary(pipeline_results)

        # Verify summary content
        assert "DETECTION PIPELINE SUMMARY" in summary
        assert "test_dataset" in summary
        assert "IsolationForest" in summary
        assert "LocalOutlierFactor" in summary
        assert "ERROR" in summary
        assert "Best Algorithm: IsolationForest" in summary

    def test_get_pipeline_summary_failure(self):
        """Test pipeline summary generation failure."""
        mock_detector_repo = MagicMock()
        mock_result_repo = MagicMock()

        service = DetectionPipelineService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo
        )

        # Mock invalid pipeline results
        pipeline_results = {"invalid": "data"}

        summary = service.get_pipeline_summary(pipeline_results)

        # Should return failure message
        assert "Failed to generate pipeline summary" in summary
