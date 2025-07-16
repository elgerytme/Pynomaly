"""
Comprehensive domain services tests.
Tests core domain services including ensemble aggregation, metrics calculation,
and processing orchestration.
"""

import asyncio
from unittest.mock import patch

import numpy as np
import pytest

from monorepo.domain.services.ensemble_aggregator import EnsembleAggregator
from monorepo.domain.services.metrics_calculator import MetricsCalculator
from monorepo.domain.services.processing_orchestrator import ProcessingOrchestrator
from monorepo.domain.value_objects import AnomalyScore


class TestEnsembleAggregator:
    """Test suite for ensemble aggregator domain service."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample anomaly scores for testing."""
        return {
            "detector1": [
                AnomalyScore(0.1, confidence_lower=0.05, confidence_upper=0.15),
                AnomalyScore(0.8, confidence_lower=0.75, confidence_upper=0.85),
                AnomalyScore(0.3, confidence_lower=0.25, confidence_upper=0.35),
            ],
            "detector2": [
                AnomalyScore(0.2, confidence_lower=0.15, confidence_upper=0.25),
                AnomalyScore(0.9, confidence_lower=0.85, confidence_upper=0.95),
                AnomalyScore(0.4, confidence_lower=0.35, confidence_upper=0.45),
            ],
            "detector3": [
                AnomalyScore(0.15, confidence_lower=0.1, confidence_upper=0.2),
                AnomalyScore(0.7, confidence_lower=0.65, confidence_upper=0.75),
                AnomalyScore(0.25, confidence_lower=0.2, confidence_upper=0.3),
            ],
        }

    def test_aggregate_scores_average(self, sample_scores):
        """Test average score aggregation."""
        aggregated = EnsembleAggregator.aggregate_scores(
            sample_scores, method="average"
        )

        assert len(aggregated) == 3

        # Check first score: (0.1 + 0.2 + 0.15) / 3 = 0.15
        assert abs(aggregated[0].value - 0.15) < 0.001

        # Check second score: (0.8 + 0.9 + 0.7) / 3 = 0.8
        assert abs(aggregated[1].value - 0.8) < 0.001

        # Check third score: (0.3 + 0.4 + 0.25) / 3 = 0.31667
        assert abs(aggregated[2].value - 0.31667) < 0.001

    def test_aggregate_scores_median(self, sample_scores):
        """Test median score aggregation."""
        aggregated = EnsembleAggregator.aggregate_scores(sample_scores, method="median")

        assert len(aggregated) == 3

        # Check first score: median of [0.1, 0.2, 0.15] = 0.15
        assert abs(aggregated[0].value - 0.15) < 0.001

        # Check second score: median of [0.8, 0.9, 0.7] = 0.8
        assert abs(aggregated[1].value - 0.8) < 0.001

        # Check third score: median of [0.3, 0.4, 0.25] = 0.3
        assert abs(aggregated[2].value - 0.3) < 0.001

    def test_aggregate_scores_max(self, sample_scores):
        """Test max score aggregation."""
        aggregated = EnsembleAggregator.aggregate_scores(sample_scores, method="max")

        assert len(aggregated) == 3

        # Check first score: max of [0.1, 0.2, 0.15] = 0.2
        assert abs(aggregated[0].value - 0.2) < 0.001

        # Check second score: max of [0.8, 0.9, 0.7] = 0.9
        assert abs(aggregated[1].value - 0.9) < 0.001

        # Check third score: max of [0.3, 0.4, 0.25] = 0.4
        assert abs(aggregated[2].value - 0.4) < 0.001

    def test_aggregate_scores_weighted(self, sample_scores):
        """Test weighted score aggregation."""
        weights = {"detector1": 0.5, "detector2": 0.3, "detector3": 0.2}
        aggregated = EnsembleAggregator.aggregate_scores(
            sample_scores, weights=weights, method="weighted"
        )

        assert len(aggregated) == 3

        # Check first score: 0.1*0.5 + 0.2*0.3 + 0.15*0.2 = 0.14
        assert abs(aggregated[0].value - 0.14) < 0.001

        # Check second score: 0.8*0.5 + 0.9*0.3 + 0.7*0.2 = 0.81
        assert abs(aggregated[1].value - 0.81) < 0.001

    def test_aggregate_scores_empty_input(self):
        """Test aggregation with empty input."""
        result = EnsembleAggregator.aggregate_scores({})
        assert result == []

    def test_aggregate_scores_mismatched_lengths(self):
        """Test aggregation with mismatched score lengths."""
        scores = {
            "detector1": [AnomalyScore(0.1), AnomalyScore(0.2)],
            "detector2": [AnomalyScore(0.3)],  # Different length
        }

        with pytest.raises(ValueError, match="expected"):
            EnsembleAggregator.aggregate_scores(scores)

    def test_aggregate_scores_invalid_method(self, sample_scores):
        """Test aggregation with invalid method."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            EnsembleAggregator.aggregate_scores(sample_scores, method="invalid")

    def test_aggregate_predictions_majority(self):
        """Test majority voting for predictions."""
        predictions = {
            "detector1": [0, 1, 0, 1],
            "detector2": [0, 1, 1, 1],
            "detector3": [1, 1, 0, 0],
        }

        result = EnsembleAggregator.aggregate_predictions(
            predictions, method="majority"
        )

        # Expected: [0, 1, 0, 1] (majority votes)
        assert result == [0, 1, 0, 1]

    def test_aggregate_predictions_unanimous(self):
        """Test unanimous voting for predictions."""
        predictions = {
            "detector1": [0, 1, 0, 1],
            "detector2": [0, 1, 1, 1],
            "detector3": [1, 1, 0, 0],
        }

        result = EnsembleAggregator.aggregate_predictions(
            predictions, method="unanimous"
        )

        # Expected: [0, 1, 0, 0] (only unanimous votes)
        assert result == [0, 1, 0, 0]

    def test_aggregate_predictions_weighted_voting(self):
        """Test weighted voting for predictions."""
        predictions = {
            "detector1": [0, 1, 0, 1],
            "detector2": [0, 1, 1, 1],
            "detector3": [1, 1, 0, 0],
        }
        weights = {"detector1": 0.5, "detector2": 0.3, "detector3": 0.2}

        result = EnsembleAggregator.aggregate_predictions(
            predictions, weights=weights, method="weighted"
        )

        # Should consider weights when making decisions
        assert len(result) == 4
        assert all(pred in [0, 1] for pred in result)


class TestMetricsCalculator:
    """Test suite for metrics calculator domain service."""

    @pytest.fixture
    def sample_binary_data(self):
        """Create sample binary classification data."""
        return {
            "y_true": np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0]),
            "y_pred": np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0]),
            "proba": np.array([0.1, 0.8, 0.9, 0.85, 0.2, 0.9, 0.1, 0.4, 0.95, 0.15]),
        }

    def test_compute_anomaly_metrics(self, sample_binary_data):
        """Test computing anomaly detection metrics."""
        metrics = MetricsCalculator.compute(
            sample_binary_data["y_true"],
            sample_binary_data["y_pred"],
            sample_binary_data["proba"],
            task_type="anomaly",
        )

        # Check that all expected metrics are present
        expected_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "precision_recall_auc",
            "confusion_matrix",
            "classification_report",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert metrics[metric] is not None

        # Check specific metric values
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_compute_classification_metrics(self, sample_binary_data):
        """Test computing general classification metrics."""
        metrics = MetricsCalculator.compute(
            sample_binary_data["y_true"],
            sample_binary_data["y_pred"],
            sample_binary_data["proba"],
            task_type="classification",
        )

        # Should include additional classification-specific metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

        # Check metric ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_compute_metrics_with_confidence_intervals(self, sample_binary_data):
        """Test computing metrics with confidence intervals."""
        metrics = MetricsCalculator.compute(
            sample_binary_data["y_true"],
            sample_binary_data["y_pred"],
            sample_binary_data["proba"],
            task_type="anomaly",
            confidence_level=0.95,
        )

        # Should include confidence intervals for key metrics
        assert "accuracy_ci" in metrics
        assert "precision_ci" in metrics
        assert "recall_ci" in metrics
        assert "f1_score_ci" in metrics

        # Check CI structure
        for ci_key in ["accuracy_ci", "precision_ci", "recall_ci", "f1_score_ci"]:
            ci = metrics[ci_key]
            assert "lower" in ci
            assert "upper" in ci
            assert ci["lower"] <= ci["upper"]

    def test_compute_multiclass_metrics(self):
        """Test computing metrics for multiclass problems."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 2, 0, 2, 1])

        metrics = MetricsCalculator.compute(y_true, y_pred, task_type="classification")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_compute_clustering_metrics(self):
        """Test computing clustering metrics."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        metrics = MetricsCalculator.compute(y_true, y_pred, task_type="clustering")

        assert "adjusted_rand_score" in metrics
        assert "normalized_mutual_info" in metrics
        assert "silhouette_score" in metrics

    def test_compute_metrics_invalid_inputs(self):
        """Test metrics computation with invalid inputs."""
        # Mismatched lengths
        with pytest.raises(ValueError):
            MetricsCalculator.compute([0, 1], [0, 1, 0], task_type="anomaly")

        # Empty arrays
        with pytest.raises(ValueError):
            MetricsCalculator.compute([], [], task_type="anomaly")

        # Invalid task type
        with pytest.raises(ValueError):
            MetricsCalculator.compute([0, 1], [0, 1], task_type="invalid")

    def test_compute_metrics_edge_cases(self):
        """Test metrics computation with edge cases."""
        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        metrics = MetricsCalculator.compute(y_true, y_pred, task_type="anomaly")

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

        # All predictions wrong
        y_pred_wrong = np.array([1, 1, 0, 0])
        metrics_wrong = MetricsCalculator.compute(
            y_true, y_pred_wrong, task_type="anomaly"
        )

        assert metrics_wrong["accuracy"] == 0.0

    def test_compute_metrics_async(self, sample_binary_data):
        """Test asynchronous metrics computation."""

        async def run_async_test():
            metrics = await MetricsCalculator.compute_async(
                sample_binary_data["y_true"],
                sample_binary_data["y_pred"],
                sample_binary_data["proba"],
                task_type="anomaly",
            )

            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics

            return metrics

        # Run async test
        result = asyncio.run(run_async_test())
        assert result is not None

    def test_bootstrap_metrics(self, sample_binary_data):
        """Test bootstrap confidence interval computation."""
        metrics = MetricsCalculator.compute_bootstrap_metrics(
            sample_binary_data["y_true"],
            sample_binary_data["y_pred"],
            n_bootstrap=100,
            confidence_level=0.95,
        )

        # Should include bootstrap statistics
        assert "bootstrap_accuracy" in metrics
        assert "bootstrap_precision" in metrics
        assert "bootstrap_recall" in metrics
        assert "bootstrap_f1_score" in metrics

        # Check bootstrap structure
        for bootstrap_key in [
            "bootstrap_accuracy",
            "bootstrap_precision",
            "bootstrap_recall",
            "bootstrap_f1_score",
        ]:
            bootstrap_stats = metrics[bootstrap_key]
            assert "mean" in bootstrap_stats
            assert "std" in bootstrap_stats
            assert "ci_lower" in bootstrap_stats
            assert "ci_upper" in bootstrap_stats


class TestProcessingOrchestrator:
    """Test suite for processing orchestrator domain service."""

    @pytest.fixture
    def sample_pipeline(self):
        """Create sample processing pipeline."""
        return {
            "preprocessing": {
                "normalize": True,
                "remove_outliers": True,
                "handle_missing": "mean",
            },
            "feature_engineering": {
                "create_interactions": False,
                "polynomial_features": {"degree": 2},
                "feature_selection": {"method": "variance", "threshold": 0.1},
            },
            "detection": {
                "algorithm": "IsolationForest",
                "hyperparameters": {"n_estimators": 100, "contamination": 0.1},
            },
            "postprocessing": {
                "apply_threshold": True,
                "smooth_predictions": {"window_size": 3},
            },
        }

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = ProcessingOrchestrator()

        assert orchestrator is not None
        assert orchestrator.pipeline_steps == []
        assert orchestrator.execution_history == []

    def test_orchestrator_add_step(self, sample_pipeline):
        """Test adding processing steps."""
        orchestrator = ProcessingOrchestrator()

        # Add preprocessing step
        orchestrator.add_step("preprocessing", sample_pipeline["preprocessing"])

        assert len(orchestrator.pipeline_steps) == 1
        assert orchestrator.pipeline_steps[0]["name"] == "preprocessing"
        assert (
            orchestrator.pipeline_steps[0]["config"] == sample_pipeline["preprocessing"]
        )

    def test_orchestrator_execute_pipeline(self, sample_pipeline):
        """Test executing complete pipeline."""
        orchestrator = ProcessingOrchestrator()

        # Add all steps
        for step_name, config in sample_pipeline.items():
            orchestrator.add_step(step_name, config)

        # Mock data
        mock_data = np.random.randn(100, 5)

        # Mock step processors
        with patch.object(orchestrator, "_execute_step") as mock_execute:
            mock_execute.return_value = mock_data

            result = orchestrator.execute_pipeline(mock_data)

            assert result is not None
            assert mock_execute.call_count == 4  # 4 steps in pipeline

    def test_orchestrator_parallel_execution(self, sample_pipeline):
        """Test parallel pipeline execution."""
        orchestrator = ProcessingOrchestrator()

        # Add steps
        for step_name, config in sample_pipeline.items():
            orchestrator.add_step(step_name, config)

        mock_data = np.random.randn(100, 5)

        # Mock parallel execution
        with patch.object(orchestrator, "_execute_parallel") as mock_parallel:
            mock_parallel.return_value = mock_data

            result = orchestrator.execute_pipeline_parallel(mock_data, n_jobs=2)

            assert result is not None
            mock_parallel.assert_called_once()

    def test_orchestrator_error_handling(self):
        """Test orchestrator error handling."""
        orchestrator = ProcessingOrchestrator()

        # Add step that will fail
        orchestrator.add_step("failing_step", {"fail": True})

        mock_data = np.random.randn(100, 5)

        with patch.object(orchestrator, "_execute_step") as mock_execute:
            mock_execute.side_effect = ValueError("Step failed")

            with pytest.raises(ValueError, match="Step failed"):
                orchestrator.execute_pipeline(mock_data)

    def test_orchestrator_step_validation(self):
        """Test step validation."""
        orchestrator = ProcessingOrchestrator()

        # Invalid step configuration
        with pytest.raises(ValueError, match="Step name cannot be empty"):
            orchestrator.add_step("", {"config": "value"})

        with pytest.raises(ValueError, match="Step config cannot be empty"):
            orchestrator.add_step("valid_name", {})

    def test_orchestrator_execution_history(self, sample_pipeline):
        """Test execution history tracking."""
        orchestrator = ProcessingOrchestrator()

        # Add steps
        for step_name, config in sample_pipeline.items():
            orchestrator.add_step(step_name, config)

        mock_data = np.random.randn(100, 5)

        with patch.object(orchestrator, "_execute_step") as mock_execute:
            mock_execute.return_value = mock_data

            orchestrator.execute_pipeline(mock_data)

            # Check execution history
            assert len(orchestrator.execution_history) == 4

            for i, history_entry in enumerate(orchestrator.execution_history):
                assert "step_name" in history_entry
                assert "execution_time" in history_entry
                assert "status" in history_entry
                assert history_entry["status"] == "completed"

    def test_orchestrator_pipeline_optimization(self, sample_pipeline):
        """Test pipeline optimization."""
        orchestrator = ProcessingOrchestrator()

        # Add steps
        for step_name, config in sample_pipeline.items():
            orchestrator.add_step(step_name, config)

        # Test optimization
        optimized_config = orchestrator.optimize_pipeline(
            optimization_target="speed", constraints={"max_memory": "1GB"}
        )

        assert optimized_config is not None
        assert "optimized_steps" in optimized_config
        assert "optimization_metrics" in optimized_config

    def test_orchestrator_pipeline_caching(self, sample_pipeline):
        """Test pipeline result caching."""
        orchestrator = ProcessingOrchestrator(enable_caching=True)

        # Add steps
        for step_name, config in sample_pipeline.items():
            orchestrator.add_step(step_name, config)

        mock_data = np.random.randn(100, 5)

        with patch.object(orchestrator, "_execute_step") as mock_execute:
            mock_execute.return_value = mock_data

            # First execution
            result1 = orchestrator.execute_pipeline(mock_data)

            # Second execution (should use cache)
            result2 = orchestrator.execute_pipeline(mock_data)

            assert np.array_equal(result1, result2)
            # Should only execute once due to caching
            assert mock_execute.call_count == 4  # 4 steps, executed once

    def test_orchestrator_async_execution(self, sample_pipeline):
        """Test asynchronous pipeline execution."""

        async def run_async_test():
            orchestrator = ProcessingOrchestrator()

            # Add steps
            for step_name, config in sample_pipeline.items():
                orchestrator.add_step(step_name, config)

            mock_data = np.random.randn(100, 5)

            with patch.object(orchestrator, "_execute_step_async") as mock_execute:
                mock_execute.return_value = mock_data

                result = await orchestrator.execute_pipeline_async(mock_data)

                assert result is not None
                assert mock_execute.call_count == 4

                return result

        # Run async test
        result = asyncio.run(run_async_test())
        assert result is not None


class TestDomainServiceIntegration:
    """Integration tests for domain services."""

    def test_ensemble_metrics_integration(self):
        """Test integration between ensemble aggregator and metrics calculator."""
        # Create ensemble predictions
        predictions = {
            "detector1": [0, 1, 0, 1, 0],
            "detector2": [0, 1, 1, 1, 0],
            "detector3": [1, 1, 0, 0, 0],
        }

        # Aggregate predictions
        aggregated = EnsembleAggregator.aggregate_predictions(
            predictions, method="majority"
        )

        # Calculate metrics
        y_true = np.array([0, 1, 0, 1, 0])
        metrics = MetricsCalculator.compute(y_true, aggregated, task_type="anomaly")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_orchestrator_with_ensemble(self, sample_pipeline):
        """Test orchestrator integration with ensemble processing."""
        orchestrator = ProcessingOrchestrator()

        # Add ensemble step
        ensemble_config = {
            "detectors": ["IsolationForest", "LOF", "OneClassSVM"],
            "aggregation_method": "weighted",
            "weights": [0.4, 0.3, 0.3],
        }

        orchestrator.add_step("ensemble_detection", ensemble_config)

        mock_data = np.random.randn(100, 5)

        with patch.object(orchestrator, "_execute_step") as mock_execute:
            mock_execute.return_value = mock_data

            result = orchestrator.execute_pipeline(mock_data)

            assert result is not None
            mock_execute.assert_called_once()

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Create orchestrator
        orchestrator = ProcessingOrchestrator()

        # Add preprocessing
        orchestrator.add_step("preprocessing", {"normalize": True})

        # Add ensemble detection
        orchestrator.add_step(
            "ensemble_detection",
            {"detectors": ["IsolationForest", "LOF"], "aggregation_method": "average"},
        )

        # Add metrics calculation
        orchestrator.add_step(
            "metrics_calculation", {"task_type": "anomaly", "confidence_level": 0.95}
        )

        # Mock data
        mock_data = np.random.randn(100, 5)
        mock_labels = np.random.randint(0, 2, 100)

        with patch.object(orchestrator, "_execute_step") as mock_execute:
            mock_execute.return_value = mock_data

            result = orchestrator.execute_pipeline(mock_data, labels=mock_labels)

            assert result is not None
            assert mock_execute.call_count == 3  # 3 steps in pipeline
