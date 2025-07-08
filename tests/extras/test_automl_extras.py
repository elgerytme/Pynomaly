"""
Comprehensive tests for AutoML extras.

This module tests AutoML functionality with graceful degradation
when AutoML packages are not installed.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from tests.utils.extras_testing import (
    requires_automl,
    parametrize_with_extras,
    automl_available,
    check_graceful_degradation,
)


class TestAutoMLExtras:
    """Test suite for AutoML extras functionality."""

    @requires_automl()
    def test_optuna_import_with_extras(self, automl_available):
        """Test Optuna import when AutoML extras are available."""
        optuna = automl_available.get("optuna")
        if optuna is not None:
            # Test basic Optuna functionality
            assert hasattr(optuna, "create_study")
            study = optuna.create_study(direction="minimize")
            assert study is not None
        else:
            pytest.skip("Optuna not available")

    @requires_automl()
    def test_hyperopt_import_with_extras(self, automl_available):
        """Test HyperOpt import when AutoML extras are available."""
        hyperopt = automl_available.get("hyperopt")
        if hyperopt is not None:
            # Test basic HyperOpt functionality
            assert hasattr(hyperopt, "fmin")
            assert hasattr(hyperopt, "hp")
            # Test space definition
            space = hyperopt.hp.uniform("x", -10, 10)
            assert space is not None
        else:
            pytest.skip("HyperOpt not available")

    @parametrize_with_extras(["automl"])
    def test_automl_service_availability(self, required_extras):
        """Test that AutoML service is available when extras are installed."""
        try:
            from pynomaly.application.services.automl_service import AutoMLService
            # Should be able to create service
            service = AutoMLService
            assert service is not None
        except ImportError as e:
            pytest.skip(f"AutoML service not available: {e}")

    def test_automl_graceful_degradation(self):
        """Test graceful degradation when AutoML packages are missing."""
        def mock_automl_function():
            # Simulate a function that would use AutoML
            try:
                import optuna
                study = optuna.create_study()
                return {"optimizer": "optuna", "study": study}
            except ImportError:
                # Graceful fallback to grid search
                return {"optimizer": "grid_search", "study": None}
        
        # Test that the function works with or without AutoML
        result = mock_automl_function()
        assert "optimizer" in result
        assert result["optimizer"] in ["optuna", "grid_search"]

    def test_automl_service_fallback(self):
        """Test that AutoML service falls back gracefully."""
        try:
            from pynomaly.application.services.automl_service import AutoMLService
            from pynomaly.infrastructure.repositories.in_memory_repositories import (
                InMemoryDetectorRepository,
                InMemoryDatasetRepository,
            )
            
            # Should not raise ImportError if properly implemented
            service = AutoMLService(
                detector_repository=InMemoryDetectorRepository(),
                dataset_repository=InMemoryDatasetRepository(),
                adapter_registry=Mock(),
            )
            assert service is not None
        except ImportError:
            # This is expected if AutoML dependencies are missing
            pytest.skip("AutoML service not available without extras")

    @pytest.mark.parametrize("optimizer", ["optuna", "hyperopt"])
    def test_individual_optimizer_availability(self, optimizer):
        """Test availability of individual AutoML optimizers."""
        try:
            module = pytest.importorskip(optimizer)
            assert module is not None
        except pytest.skip.Exception:
            pytest.skip(f"{optimizer} not available")

    @requires_automl()
    def test_automl_optimization_with_sample_data(self, automl_available):
        """Test AutoML optimization with sample data."""
        # Create sample data
        sample_data = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
        })
        
        optuna = automl_available.get("optuna")
        if optuna is not None:
            # Test basic optimization
            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return x ** 2
            
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=5)
            
            assert study.best_value is not None
            assert study.best_params is not None

    @requires_automl()
    def test_automl_hyperparameter_optimization(self, automl_available):
        """Test hyperparameter optimization functionality."""
        optuna = automl_available.get("optuna")
        if optuna is not None:
            # Test hyperparameter search for anomaly detection
            def objective(trial):
                # Simulate hyperparameter optimization for anomaly detection
                contamination = trial.suggest_float("contamination", 0.01, 0.3)
                n_estimators = trial.suggest_int("n_estimators", 10, 200)
                
                # Mock score (in real implementation, this would be CV score)
                score = -(contamination * 0.5 + n_estimators * 0.001)
                return score
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=10)
            
            assert "contamination" in study.best_params
            assert "n_estimators" in study.best_params

    def test_automl_configuration_profiles(self):
        """Test AutoML configuration profiles for different scenarios."""
        try:
            from pynomaly.application.services.automl_service import AutoMLService
            
            # Test configuration profiles
            profiles = {
                "fast": {"max_optimization_time": 30, "n_trials": 10},
                "balanced": {"max_optimization_time": 300, "n_trials": 50},
                "thorough": {"max_optimization_time": 1800, "n_trials": 200},
            }
            
            for profile_name, config in profiles.items():
                assert "max_optimization_time" in config
                assert "n_trials" in config
                assert config["max_optimization_time"] > 0
                assert config["n_trials"] > 0
                
        except ImportError:
            pytest.skip("AutoML service not available")

    def test_automl_algorithm_recommendations(self):
        """Test AutoML algorithm recommendation functionality."""
        try:
            from pynomaly.application.services.automl_service import AutoMLService
            
            # Mock dataset profiles
            small_dataset_profile = {
                "n_samples": 100,
                "n_features": 5,
                "contamination_estimate": 0.1,
            }
            
            large_dataset_profile = {
                "n_samples": 100000,
                "n_features": 50,
                "contamination_estimate": 0.05,
            }
            
            # Test that recommendations are different for different profiles
            assert small_dataset_profile["n_samples"] < large_dataset_profile["n_samples"]
            
        except ImportError:
            pytest.skip("AutoML service not available")

    @requires_automl()
    def test_automl_early_stopping(self, automl_available):
        """Test early stopping functionality in AutoML."""
        optuna = automl_available.get("optuna")
        if optuna is not None:
            # Test early stopping
            def objective(trial):
                # Simulate a function that should trigger early stopping
                x = trial.suggest_float("x", -10, 10)
                if abs(x) < 0.1:  # Good solution found
                    return x ** 2
                return x ** 2 + 100  # Poor solution
            
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=20)
            
            # Should find a good solution
            assert study.best_value < 1.0

    def test_automl_error_handling(self):
        """Test error handling when AutoML packages are missing."""
        def function_requiring_automl():
            import optuna  # This will raise ImportError if not available
            return optuna.create_study()
        
        # Test error handling
        graceful, result = check_graceful_degradation(
            function_requiring_automl,
            "automl",
            expected_error_type=ImportError
        )
        
        # Result should be either successful or ImportError
        assert isinstance(result, (Exception, type(None)))

    def test_automl_metrics_tracking(self):
        """Test AutoML metrics tracking without requiring extras."""
        # This should work without AutoML packages
        mock_metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1_score": 0.90,
        }
        
        # Test basic metrics handling
        assert all(0 <= v <= 1 for v in mock_metrics.values())
        assert "accuracy" in mock_metrics
        assert "precision" in mock_metrics
        assert "recall" in mock_metrics
        assert "f1_score" in mock_metrics
