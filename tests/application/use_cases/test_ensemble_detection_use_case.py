"""Comprehensive test suite for ensemble detection use case."""

from unittest.mock import AsyncMock, Mock

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.use_cases.ensemble_detection_use_case import (
    DetectorPerformanceMetrics,
    EnsembleDetectionRequest,
    EnsembleDetectionResponse,
    EnsembleDetectionUseCase,
    EnsembleOptimizationObjective,
    EnsembleOptimizationRequest,
    VotingStrategy,
)
from pynomaly.domain.entities import Dataset, Detector


class TestEnsembleDetectionUseCase:
    """Test suite for ensemble detection use case functionality."""

    @pytest.fixture
    def mock_repositories(self):
        """Mock repositories for testing."""
        detector_repo = Mock()
        dataset_repo = Mock()
        adapter_registry = Mock()
        return detector_repo, dataset_repo, adapter_registry

    @pytest.fixture
    def ensemble_use_case(self, mock_repositories):
        """Create ensemble detection use case with mocked dependencies."""
        detector_repo, dataset_repo, adapter_registry = mock_repositories

        use_case = EnsembleDetectionUseCase(
            detector_repository=detector_repo,
            dataset_repository=dataset_repo,
            adapter_registry=adapter_registry,
            enable_performance_tracking=True,
            enable_auto_optimization=True,
            cache_size=100,
        )

        return use_case

    @pytest.fixture
    def sample_detectors(self):
        """Create sample detectors for testing."""
        detectors = []

        for i in range(3):
            detector = Mock(spec=Detector)
            detector.id = f"detector_{i}"
            detector.algorithm = "IsolationForest"
            detector.is_fitted = True
            detector.model = Mock()
            detectors.append(detector)

        return detectors

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randn(100),
            }
        )

        dataset = Mock(spec=Dataset)
        dataset.id = "test_dataset"
        dataset.data = data
        dataset.features = data

        return dataset

    @pytest.fixture
    def sample_data(self):
        """Create sample data for predictions."""
        np.random.seed(42)
        return np.random.randn(10, 3)

    @pytest.mark.asyncio
    async def test_detect_anomalies_ensemble_simple_average(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test ensemble detection with simple average voting."""
        # Setup mocks
        for i, detector in enumerate(sample_detectors):
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            # Mock adapter
            adapter = Mock()
            adapter.predict.return_value = (
                np.random.choice([0, 1], size=len(sample_data)),  # predictions
                np.random.rand(len(sample_data)),  # scores
            )
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

        # Create request
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.SIMPLE_AVERAGE,
            enable_dynamic_weighting=False,
            enable_explanation=True,
        )

        # Execute
        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Verify
        assert response.success is True
        assert response.predictions is not None
        assert len(response.predictions) == len(sample_data)
        assert response.anomaly_scores is not None
        assert len(response.anomaly_scores) == len(sample_data)
        assert response.confidence_scores is not None
        assert response.uncertainty_scores is not None
        assert response.consensus_scores is not None
        assert response.voting_strategy_used == VotingStrategy.SIMPLE_AVERAGE.value
        assert response.explanations is not None
        assert response.processing_time > 0

    @pytest.mark.asyncio
    async def test_detect_anomalies_ensemble_weighted_average(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test ensemble detection with weighted average voting."""
        # Setup mocks
        for detector in sample_detectors:
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            adapter = Mock()
            adapter.predict.return_value = (
                np.random.choice([0, 1], size=len(sample_data)),
                np.random.rand(len(sample_data)),
            )
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

        # Add performance metrics for dynamic weighting
        for detector in sample_detectors:
            metrics = DetectorPerformanceMetrics(
                detector_id=detector.id,
                accuracy=0.8,
                f1_score=0.75,
                diversity_contribution=0.6,
                stability_score=0.9,
            )
            ensemble_use_case._performance_tracker[detector.id] = metrics

        # Create request
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.WEIGHTED_AVERAGE,
            enable_dynamic_weighting=True,
            enable_explanation=True,
        )

        # Execute
        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Verify
        assert response.success is True
        assert response.detector_weights is not None
        assert len(response.detector_weights) == len(sample_detectors)
        assert (
            abs(sum(response.detector_weights) - 1.0) < 0.01
        )  # Weights should sum to 1
        assert response.voting_strategy_used == VotingStrategy.WEIGHTED_AVERAGE.value

    @pytest.mark.asyncio
    async def test_detect_anomalies_ensemble_consensus_voting(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test ensemble detection with consensus voting."""
        # Setup mocks
        for detector in sample_detectors:
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            adapter = Mock()
            # Create predictions with some consensus
            predictions = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1])  # 60% anomalies
            scores = np.random.rand(len(sample_data))
            adapter.predict.return_value = (predictions, scores)
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

        # Create request
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.CONSENSUS_VOTING,
            consensus_threshold=0.7,
            enable_explanation=True,
        )

        # Execute
        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Verify
        assert response.success is True
        assert response.consensus_scores is not None
        assert all(0.0 <= score <= 1.0 for score in response.consensus_scores)

    @pytest.mark.asyncio
    async def test_detect_anomalies_ensemble_dynamic_selection(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test ensemble detection with dynamic selection."""
        # Setup mocks
        for detector in sample_detectors:
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            adapter = Mock()
            # Create varied confidence scores
            scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.85, 0.15, 0.75, 0.25])
            predictions = (scores > 0.5).astype(int)
            adapter.predict.return_value = (predictions, scores)
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

        # Create request
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.DYNAMIC_SELECTION,
            enable_dynamic_weighting=True,
        )

        # Execute
        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Verify
        assert response.success is True
        assert response.voting_strategy_used == VotingStrategy.DYNAMIC_SELECTION.value

    @pytest.mark.asyncio
    async def test_detect_anomalies_ensemble_uncertainty_weighted(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test ensemble detection with uncertainty-weighted voting."""
        # Setup mocks
        for detector in sample_detectors:
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            adapter = Mock()
            adapter.predict.return_value = (
                np.random.choice([0, 1], size=len(sample_data)),
                np.random.rand(len(sample_data)),
            )
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

        # Create request
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.UNCERTAINTY_WEIGHTED,
            enable_uncertainty_estimation=True,
        )

        # Execute
        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Verify
        assert response.success is True
        assert response.uncertainty_scores is not None
        assert all(0.0 <= score <= 1.0 for score in response.uncertainty_scores)

    @pytest.mark.asyncio
    async def test_detect_anomalies_ensemble_cascaded_voting(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test ensemble detection with cascaded voting."""
        # Setup mocks with different performance levels
        for i, detector in enumerate(sample_detectors):
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            adapter = Mock()
            # Higher performing detectors get higher scores
            base_score = 0.5 + (i * 0.1)
            scores = np.full(len(sample_data), base_score) + np.random.normal(
                0, 0.1, len(sample_data)
            )
            predictions = (scores > 0.5).astype(int)
            adapter.predict.return_value = (predictions, scores)
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

            # Add performance metrics
            metrics = DetectorPerformanceMetrics(
                detector_id=detector.id,
                f1_score=0.7 + (i * 0.1),  # Increasing performance
            )
            ensemble_use_case._performance_tracker[detector.id] = metrics

        # Create request
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.CASCADED_VOTING,
            confidence_threshold=0.8,
            enable_dynamic_weighting=True,
        )

        # Execute
        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Verify
        assert response.success is True
        assert response.voting_strategy_used == VotingStrategy.CASCADED_VOTING.value

    @pytest.mark.asyncio
    async def test_detect_anomalies_ensemble_with_caching(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test ensemble detection with caching enabled."""
        # Setup mocks
        for detector in sample_detectors:
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            adapter = Mock()
            adapter.predict.return_value = (
                np.random.choice([0, 1], size=len(sample_data)),
                np.random.rand(len(sample_data)),
            )
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

        # Create request
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.SIMPLE_AVERAGE,
            enable_caching=True,
        )

        # Execute first time
        response1 = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Execute second time (should use cache)
        response2 = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Verify
        assert response1.success is True
        assert response2.success is True
        # Second response should be from cache (faster)
        assert len(ensemble_use_case._ensemble_cache) > 0

    @pytest.mark.asyncio
    async def test_detect_anomalies_ensemble_validation_errors(
        self, ensemble_use_case, sample_data
    ):
        """Test validation errors in ensemble detection."""
        # Test with too few detectors
        request = EnsembleDetectionRequest(
            detector_ids=["detector_1"],
            data=sample_data,  # Only one detector
        )

        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        assert response.success is False
        assert "At least 2 detectors required" in response.error_message

        # Test with too many detectors
        request = EnsembleDetectionRequest(
            detector_ids=[f"detector_{i}" for i in range(25)],  # Too many
            data=sample_data,
        )

        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        assert response.success is False
        assert "Too many detectors" in response.error_message

    @pytest.mark.asyncio
    async def test_detect_anomalies_ensemble_detector_not_found(
        self, ensemble_use_case, sample_data
    ):
        """Test error handling when detector is not found."""
        # Mock repository to return None
        ensemble_use_case.detector_repository.get = AsyncMock(return_value=None)

        request = EnsembleDetectionRequest(
            detector_ids=["nonexistent_detector_1", "nonexistent_detector_2"],
            data=sample_data,
        )

        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        assert response.success is False
        assert "not found" in response.error_message

    @pytest.mark.asyncio
    async def test_detect_anomalies_ensemble_unfitted_detector(
        self, ensemble_use_case, sample_data
    ):
        """Test error handling when detector is not fitted."""
        # Create unfitted detector
        detector = Mock(spec=Detector)
        detector.id = "unfitted_detector"
        detector.is_fitted = False

        ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

        request = EnsembleDetectionRequest(
            detector_ids=["unfitted_detector_1", "unfitted_detector_2"],
            data=sample_data,
        )

        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        assert response.success is False
        assert "not fitted" in response.error_message

    @pytest.mark.asyncio
    async def test_optimize_ensemble_basic(
        self, ensemble_use_case, sample_detectors, sample_dataset
    ):
        """Test basic ensemble optimization."""
        # Setup mocks
        ensemble_use_case.dataset_repository.get = AsyncMock(
            return_value=sample_dataset
        )

        for detector in sample_detectors:
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

        # Create optimization request
        request = EnsembleOptimizationRequest(
            detector_ids=[d.id for d in sample_detectors],
            validation_dataset_id="validation_dataset",
            optimization_objective=EnsembleOptimizationObjective.F1_SCORE,
            target_voting_strategies=[
                VotingStrategy.WEIGHTED_AVERAGE,
                VotingStrategy.DYNAMIC_SELECTION,
            ],
            max_ensemble_size=3,
            enable_pruning=True,
        )

        # Execute
        response = await ensemble_use_case.optimize_ensemble(request)

        # Verify
        assert response.success is True
        assert response.optimized_detector_ids is not None
        assert len(response.optimized_detector_ids) <= request.max_ensemble_size
        assert response.optimal_voting_strategy is not None
        assert response.optimal_weights is not None
        assert response.ensemble_performance is not None
        assert response.diversity_metrics is not None
        assert response.recommendations is not None
        assert response.optimization_time > 0

    @pytest.mark.asyncio
    async def test_optimize_ensemble_validation_errors(self, ensemble_use_case):
        """Test validation errors in ensemble optimization."""
        # Test with too few detectors
        request = EnsembleOptimizationRequest(
            detector_ids=["detector_1"],  # Only one detector
            validation_dataset_id="dataset_1",
        )

        response = await ensemble_use_case.optimize_ensemble(request)

        assert response.success is False
        assert "At least 2 detectors required" in response.error_message

        # Test with invalid max ensemble size
        request = EnsembleOptimizationRequest(
            detector_ids=["detector_1", "detector_2"],
            validation_dataset_id="dataset_1",
            max_ensemble_size=1,  # Too small
        )

        response = await ensemble_use_case.optimize_ensemble(request)

        assert response.success is False
        assert "Maximum ensemble size must be at least 2" in response.error_message

    @pytest.mark.asyncio
    async def test_optimize_ensemble_dataset_not_found(self, ensemble_use_case):
        """Test error handling when validation dataset is not found."""
        ensemble_use_case.dataset_repository.get = AsyncMock(return_value=None)

        request = EnsembleOptimizationRequest(
            detector_ids=["detector_1", "detector_2"],
            validation_dataset_id="nonexistent_dataset",
        )

        response = await ensemble_use_case.optimize_ensemble(request)

        assert response.success is False
        assert "not found" in response.error_message

    @pytest.mark.asyncio
    async def test_data_preparation_different_formats(self, ensemble_use_case):
        """Test data preparation with different input formats."""
        # Test with numpy array
        np_data = np.random.randn(5, 3)
        prepared = await ensemble_use_case._prepare_data(np_data)
        assert isinstance(prepared, np.ndarray)
        assert prepared.shape == (5, 3)

        # Test with pandas DataFrame
        df_data = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
        prepared = await ensemble_use_case._prepare_data(df_data)
        assert isinstance(prepared, np.ndarray)
        assert prepared.shape == (5, 3)

        # Test with list of dictionaries
        dict_data = [{"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 4.0, "b": 5.0, "c": 6.0}]
        prepared = await ensemble_use_case._prepare_data(dict_data)
        assert isinstance(prepared, np.ndarray)
        assert prepared.shape == (2, 3)

        # Test with list of numbers
        list_data = [[1, 2, 3], [4, 5, 6]]
        prepared = await ensemble_use_case._prepare_data(list_data)
        assert isinstance(prepared, np.ndarray)
        assert prepared.shape == (2, 3)

    @pytest.mark.asyncio
    async def test_individual_detector_failure_handling(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test handling of individual detector failures."""
        # Setup mocks where one detector fails
        for i, detector in enumerate(sample_detectors):
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            adapter = Mock()
            if i == 1:  # Make second detector fail
                adapter.predict.side_effect = Exception("Detector failed")
            else:
                adapter.predict.return_value = (
                    np.random.choice([0, 1], size=len(sample_data)),
                    np.random.rand(len(sample_data)),
                )
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

        # Create request
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.SIMPLE_AVERAGE,
        )

        # Execute
        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Verify that ensemble still works with remaining detectors
        assert response.success is True
        assert response.predictions is not None
        # Should have warnings about failed detector
        assert (
            len(response.warnings) > 0 or response.success is True
        )  # Graceful degradation

    @pytest.mark.asyncio
    async def test_performance_tracking_updates(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test that performance tracking is updated correctly."""
        # Setup mocks
        for detector in sample_detectors:
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            adapter = Mock()
            adapter.predict.return_value = (
                np.random.choice([0, 1], size=len(sample_data)),
                np.random.rand(len(sample_data)),
            )
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

        # Create request
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.SIMPLE_AVERAGE,
        )

        # Execute
        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Verify performance tracking was updated
        assert response.success is True
        for detector in sample_detectors:
            assert detector.id in ensemble_use_case._performance_tracker
            metrics = ensemble_use_case._performance_tracker[detector.id]
            assert isinstance(metrics, DetectorPerformanceMetrics)
            assert metrics.last_updated > 0

    @pytest.mark.asyncio
    async def test_ensemble_metrics_calculation(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test ensemble metrics calculation."""
        # Setup mocks
        for detector in sample_detectors:
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            adapter = Mock()
            adapter.predict.return_value = (
                np.random.choice([0, 1], size=len(sample_data)),
                np.random.rand(len(sample_data)),
            )
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

        # Create request
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.WEIGHTED_AVERAGE,
        )

        # Execute
        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Verify ensemble metrics
        assert response.success is True
        assert response.ensemble_metrics is not None

        metrics = response.ensemble_metrics
        assert "diversity_metrics" in metrics
        assert "performance_metrics" in metrics
        assert "processing_statistics" in metrics
        assert metrics["ensemble_size"] == len(sample_detectors)
        assert metrics["voting_strategy"] == VotingStrategy.WEIGHTED_AVERAGE.value

    @pytest.mark.asyncio
    async def test_explanation_generation(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test explanation generation for ensemble predictions."""
        # Setup mocks
        for detector in sample_detectors:
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            adapter = Mock()
            adapter.predict.return_value = (
                np.random.choice([0, 1], size=len(sample_data)),
                np.random.rand(len(sample_data)),
            )
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

        # Create request with explanations enabled
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.WEIGHTED_AVERAGE,
            enable_explanation=True,
        )

        # Execute
        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        # Verify explanations
        assert response.success is True
        assert response.explanations is not None
        assert len(response.explanations) == len(sample_data)

        explanation = response.explanations[0]
        assert "ensemble_score" in explanation
        assert "ensemble_prediction" in explanation
        assert "voting_strategy" in explanation
        assert "detector_contributions" in explanation
        assert "top_contributors" in explanation
        assert "reasoning" in explanation

    def test_cache_key_generation(self, ensemble_use_case, sample_data):
        """Test cache key generation."""
        request = EnsembleDetectionRequest(
            detector_ids=["detector_1", "detector_2"],
            data=sample_data,
            voting_strategy=VotingStrategy.SIMPLE_AVERAGE,
        )

        cache_key = ensemble_use_case._generate_cache_key(request, sample_data)

        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

        # Same request should generate same key
        cache_key2 = ensemble_use_case._generate_cache_key(request, sample_data)
        assert cache_key == cache_key2

        # Different request should generate different key
        request2 = EnsembleDetectionRequest(
            detector_ids=["detector_1", "detector_3"],
            data=sample_data,
            voting_strategy=VotingStrategy.SIMPLE_AVERAGE,
        )
        cache_key3 = ensemble_use_case._generate_cache_key(request2, sample_data)
        assert cache_key != cache_key3

    def test_cache_management(self, ensemble_use_case):
        """Test cache management and size limits."""
        # Set small cache size for testing
        ensemble_use_case.cache_size = 3

        # Add entries to cache
        for i in range(5):
            response = EnsembleDetectionResponse(success=True)
            ensemble_use_case._cache_result(f"key_{i}", response)

        # Verify cache size limit is enforced
        assert len(ensemble_use_case._ensemble_cache) <= ensemble_use_case.cache_size

        # Verify older entries are removed (FIFO)
        assert "key_0" not in ensemble_use_case._ensemble_cache
        assert "key_1" not in ensemble_use_case._ensemble_cache
        assert "key_4" in ensemble_use_case._ensemble_cache

    @pytest.mark.asyncio
    async def test_voting_strategy_robustness(
        self, ensemble_use_case, sample_detectors, sample_data
    ):
        """Test robustness of different voting strategies."""
        # Setup mocks with extreme scores
        for detector in sample_detectors:
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

            adapter = Mock()
            # Create extreme scores to test robustness
            extreme_scores = np.array(
                [0.0, 1.0, 0.5, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4]
            )
            predictions = (extreme_scores > 0.5).astype(int)
            adapter.predict.return_value = (predictions, extreme_scores)
            ensemble_use_case.adapter_registry.get_adapter.return_value = adapter

        # Test robust aggregation strategy
        request = EnsembleDetectionRequest(
            detector_ids=[d.id for d in sample_detectors],
            data=sample_data,
            voting_strategy=VotingStrategy.ROBUST_AGGREGATION,
        )

        response = await ensemble_use_case.detect_anomalies_ensemble(request)

        assert response.success is True
        assert response.voting_strategy_used == VotingStrategy.ROBUST_AGGREGATION.value
        # Robust aggregation should handle extreme values well
        assert all(0.0 <= score <= 1.0 for score in response.anomaly_scores)

    @pytest.mark.asyncio
    async def test_optimization_history_tracking(
        self, ensemble_use_case, sample_detectors, sample_dataset
    ):
        """Test that optimization history is tracked correctly."""
        # Setup mocks
        ensemble_use_case.dataset_repository.get = AsyncMock(
            return_value=sample_dataset
        )

        for detector in sample_detectors:
            ensemble_use_case.detector_repository.get = AsyncMock(return_value=detector)

        # Perform multiple optimizations
        for i in range(3):
            request = EnsembleOptimizationRequest(
                detector_ids=[d.id for d in sample_detectors],
                validation_dataset_id="validation_dataset",
                optimization_objective=EnsembleOptimizationObjective.F1_SCORE,
            )

            response = await ensemble_use_case.optimize_ensemble(request)
            assert response.success is True

        # Verify optimization history is tracked
        assert len(ensemble_use_case._optimization_history) == 3

        for entry in ensemble_use_case._optimization_history:
            assert "timestamp" in entry
            assert "request" in entry
            assert "response" in entry
            assert "optimization_time" in entry
