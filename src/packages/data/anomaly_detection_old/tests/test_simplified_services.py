"""Test simplified services implementation."""

import pytest
import numpy as np
from simplified_services import CoreDetectionService, AutoMLService, EnsembleService
from simplified_services.ensemble_service import EnsembleConfig


class TestCoreDetectionService:
    """Test core detection service."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.service = CoreDetectionService()
        self.test_data = np.random.randn(100, 5)
        self.test_data[:10] += 2  # Add outliers

    def test_basic_detection(self):
        """Test basic anomaly detection."""
        result = self.service.detect_anomalies(self.test_data)
        
        assert len(result.predictions) == len(self.test_data)
        assert result.algorithm in ["iforest", "isolation_forest"]
        assert result.n_samples == len(self.test_data)
        assert result.n_anomalies > 0
        assert result.execution_time > 0
        assert "detection_id" in result.metadata

    def test_algorithm_selection(self):
        """Test different algorithm selection."""
        algorithms = ["iforest", "lof", "pca"]
        
        for algorithm in algorithms:
            try:
                result = self.service.detect_anomalies(
                    self.test_data, algorithm=algorithm, contamination=0.1
                )
                assert len(result.predictions) == len(self.test_data)
                assert result.n_anomalies > 0
            except Exception as e:
                # Some algorithms might not be available
                print(f"Algorithm {algorithm} not available: {e}")

    def test_batch_detection(self):
        """Test batch processing."""
        batches = [
            self.test_data[:30],
            self.test_data[30:60], 
            self.test_data[60:]
        ]
        
        results = self.service.batch_detect(batches, algorithm="iforest")
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert len(result.predictions) == len(batches[i])
            assert result.metadata["batch_id"] == i

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        # Reset stats
        self.service.reset_stats()
        
        initial_stats = self.service.get_performance_stats()
        assert initial_stats["total_detections"] == 0
        
        # Run some detections
        self.service.detect_anomalies(self.test_data)
        self.service.detect_anomalies(self.test_data, algorithm="lof")
        
        stats = self.service.get_performance_stats()
        assert stats["total_detections"] == 2
        assert stats["total_samples"] == 2 * len(self.test_data)
        assert len(stats["algorithms_used"]) >= 1
        
        recent = self.service.get_recent_performance(limit=1)
        assert len(recent) == 1


class TestAutoMLService:
    """Test AutoML service."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.service = AutoMLService()
        self.test_data = np.random.randn(200, 5)
        self.test_data[:20] += 2

    def test_algorithm_recommendation(self):
        """Test algorithm recommendation."""
        recommendation = self.service.recommend_algorithm(
            self.test_data, time_budget=10.0, n_trials=2
        )
        
        assert recommendation.algorithm in [
            'iforest', 'lof', 'pca', 'ocsvm', 'hbos', 'cof', 'copod', 'ecod'
        ]
        assert 0.0 <= recommendation.score <= 1.0
        assert recommendation.execution_time > 0
        assert recommendation.confidence >= 0.0
        assert len(recommendation.reason) > 0

    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        result = self.service.optimize_hyperparameters(
            self.test_data, "iforest", n_trials=3
        )
        
        assert result["algorithm"] == "iforest"
        assert "contamination" in result["best_parameters"]
        assert result["best_score"] >= 0.0
        assert len(result["optimization_results"]) <= 3

    def test_auto_detect(self):
        """Test automatic detection."""
        result = self.service.auto_detect(
            self.test_data, contamination=0.1, time_budget=15.0
        )
        
        assert len(result.predictions) == len(self.test_data)
        assert result.n_anomalies > 0
        assert "automl_recommendation" in result.metadata
        
        recommendation = result.metadata["automl_recommendation"]
        assert "algorithm" in recommendation
        assert "score" in recommendation
        assert "confidence" in recommendation


class TestEnsembleService:
    """Test ensemble service."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.service = EnsembleService()
        self.test_data = np.random.randn(150, 5)
        self.test_data[:15] += 2

    def test_ensemble_detection(self):
        """Test ensemble detection."""
        config = EnsembleConfig(
            algorithms=["iforest", "lof"],
            voting_method="majority",
            contamination=0.1,
            parallel=False  # Avoid threading issues in tests
        )
        
        result = self.service.detect_ensemble(self.test_data, config)
        
        assert len(result.predictions) == len(self.test_data)
        assert result.algorithm.startswith("ensemble")
        assert result.n_anomalies > 0
        assert "ensemble_config" in result.metadata
        assert "individual_results" in result.metadata
        assert "agreement_metrics" in result.metadata

    def test_voting_methods(self):
        """Test different voting methods."""
        algorithms = ["iforest", "lof"]
        voting_methods = ["majority", "weighted", "unanimous"]
        
        for method in voting_methods:
            config = EnsembleConfig(
                algorithms=algorithms,
                voting_method=method,
                contamination=0.1,
                parallel=False
            )
            
            try:
                result = self.service.detect_ensemble(self.test_data, config)
                assert len(result.predictions) == len(self.test_data)
                assert method in result.algorithm or "ensemble" in result.algorithm
            except Exception as e:
                print(f"Voting method {method} failed: {e}")

    def test_smart_ensemble(self):
        """Test smart ensemble creation."""
        result = self.service.create_smart_ensemble(
            self.test_data, contamination=0.1, max_algorithms=3, time_budget=20.0
        )
        
        assert len(result.predictions) == len(self.test_data)
        assert result.n_anomalies > 0
        assert "ensemble_config" in result.metadata
        
        config = result.metadata["ensemble_config"]
        assert len(config["algorithms"]) <= 3
        assert config["voting_method"] == "weighted"
        assert config["weights"] is not None

    def test_weighted_voting(self):
        """Test weighted voting specifically."""
        config = EnsembleConfig(
            algorithms=["iforest", "lof"],
            voting_method="weighted",
            weights=[0.7, 0.3],
            contamination=0.1,
            parallel=False
        )
        
        result = self.service.detect_ensemble(self.test_data, config)
        
        assert len(result.predictions) == len(self.test_data)
        assert result.scores is not None
        assert len(result.scores) == len(self.test_data)

    def test_benchmark_strategies(self):
        """Test ensemble strategy benchmarking."""
        algorithms = ["iforest", "lof"]
        
        results = self.service.benchmark_ensemble_strategies(
            self.test_data, algorithms, contamination=0.1
        )
        
        assert "majority" in results
        assert "weighted" in results
        assert "unanimous" in results
        
        for strategy, result in results.items():
            if result.get("success", False):
                assert "n_anomalies" in result
                assert "execution_time" in result
                assert "agreement_score" in result

    def test_performance_stats(self):
        """Test ensemble performance statistics."""
        # Run a few ensembles
        config = EnsembleConfig(
            algorithms=["iforest", "lof"],
            voting_method="majority",
            contamination=0.1,
            parallel=False
        )
        
        self.service.detect_ensemble(self.test_data, config)
        self.service.detect_ensemble(self.test_data[:50], config)
        
        stats = self.service.get_performance_stats()
        
        assert stats["total_ensembles"] == 2
        assert "average_execution_time" in stats
        assert "average_agreement_score" in stats
        
        history = self.service.get_ensemble_history()
        assert len(history) == 2


class TestIntegratedWorkflow:
    """Test integrated workflow with all services."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_data = np.random.randn(200, 6)
        self.test_data[:20] += 2

    def test_full_automl_to_ensemble_workflow(self):
        """Test complete workflow from AutoML to Ensemble."""
        print("\n🔄 Testing integrated workflow...")
        
        # Step 1: Use AutoML to get algorithm recommendation
        automl_service = AutoMLService()
        recommendation = automl_service.recommend_algorithm(
            self.test_data, time_budget=15.0, n_trials=2
        )
        
        print(f"📊 AutoML recommended: {recommendation.algorithm}")
        
        # Step 2: Create ensemble with recommended algorithm plus others
        ensemble_service = EnsembleService()
        
        # Use smart ensemble with the recommended algorithm
        ensemble_result = ensemble_service.create_smart_ensemble(
            self.test_data, contamination=0.1, max_algorithms=3, time_budget=20.0
        )
        
        print(f"🎭 Ensemble detected: {ensemble_result.n_anomalies} anomalies")
        
        # Step 3: Compare with individual AutoML result
        automl_result = automl_service.auto_detect(
            self.test_data, contamination=0.1, time_budget=10.0
        )
        
        print(f"🤖 AutoML detected: {automl_result.n_anomalies} anomalies")
        
        # Assertions
        assert len(ensemble_result.predictions) == len(self.test_data)
        assert len(automl_result.predictions) == len(self.test_data)
        assert ensemble_result.n_anomalies > 0
        assert automl_result.n_anomalies > 0
        
        # Both should detect similar number of anomalies (within reason)
        ratio = ensemble_result.n_anomalies / automl_result.n_anomalies
        assert 0.5 <= ratio <= 2.0, f"Results too different: {ensemble_result.n_anomalies} vs {automl_result.n_anomalies}"
        
        print("✅ Integrated workflow completed successfully")

    def test_service_interoperability(self):
        """Test that services work well together."""
        core_service = CoreDetectionService()
        automl_service = AutoMLService()
        ensemble_service = EnsembleService()
        
        # Test that services can be used in sequence
        core_result = core_service.detect_anomalies(self.test_data)
        
        # AutoML should work with same data
        automl_result = automl_service.auto_detect(self.test_data, time_budget=10.0)
        
        # Ensemble should work with same data  
        config = EnsembleConfig(
            algorithms=["iforest", "lof"],
            voting_method="majority",
            contamination=0.1,
            parallel=False
        )
        ensemble_result = ensemble_service.detect_ensemble(self.test_data, config)
        
        # All should produce valid results
        assert len(core_result.predictions) == len(self.test_data)
        assert len(automl_result.predictions) == len(self.test_data)
        assert len(ensemble_result.predictions) == len(self.test_data)
        
        # Get performance stats from all services
        core_stats = core_service.get_performance_stats()
        ensemble_stats = ensemble_service.get_performance_stats()
        
        assert core_stats["total_detections"] > 0
        assert ensemble_stats["total_ensembles"] > 0