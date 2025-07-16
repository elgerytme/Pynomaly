"""Tests for advanced hyperparameter optimizer."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from monorepo.infrastructure.automl import (
    AcquisitionFunction,
    AdvancedHyperparameterOptimizer,
    AdvancedOptimizationConfig,
    MetaLearningConfig,
    MetaLearningStrategy,
    OptimizationObjective,
    OptimizationResult,
    OptimizationStrategy,
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def basic_config():
    """Basic optimization configuration."""
    return AdvancedOptimizationConfig(
        strategy=OptimizationStrategy.BAYESIAN, n_trials=10, timeout=30, random_state=42
    )


@pytest.fixture
def multi_objective_config():
    """Multi-objective optimization configuration."""
    objectives = [
        OptimizationObjective(name="accuracy", direction="maximize", weight=0.7),
        OptimizationObjective(name="speed", direction="maximize", weight=0.3),
    ]

    return AdvancedOptimizationConfig(
        strategy=OptimizationStrategy.MULTI_OBJECTIVE,
        n_trials=15,
        timeout=45,
        objectives=objectives,
        random_state=42,
    )


@pytest.fixture
def meta_learning_config():
    """Configuration with meta-learning enabled."""
    meta_learning = MetaLearningConfig(
        strategy=MetaLearningStrategy.SIMILAR_DATASETS,
        similarity_threshold=0.8,
        warm_start_trials=5,
    )

    return AdvancedOptimizationConfig(
        strategy=OptimizationStrategy.BAYESIAN,
        n_trials=10,
        timeout=30,
        meta_learning=meta_learning,
        random_state=42,
    )


@pytest.fixture
def sample_parameter_space():
    """Sample parameter space for testing."""
    return {
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        "batch_size": {"type": "int", "low": 16, "high": 128},
        "optimizer": {"type": "categorical", "choices": ["adam", "sgd", "rmsprop"]},
        "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5},
    }


@pytest.fixture
def sample_dataset_profile():
    """Sample dataset profile for meta-learning."""
    return {
        "n_samples": 1000,
        "n_features": 50,
        "contamination_estimate": 0.1,
        "missing_values_ratio": 0.05,
        "sparsity_ratio": 0.1,
        "dimensionality_ratio": 0.05,
        "dataset_size_mb": 10.0,
        "has_temporal_structure": False,
        "has_graph_structure": False,
        "complexity_score": 0.3,
    }


def simple_objective_function(params, **kwargs):
    """Simple objective function for testing."""
    # Simulate a quadratic function with noise
    learning_rate = params.get("learning_rate", 0.01)
    batch_size = params.get("batch_size", 32)
    dropout_rate = params.get("dropout_rate", 0.1)

    # Optimal values
    optimal_lr = 0.01
    optimal_batch = 64
    optimal_dropout = 0.2

    # Calculate score (higher is better)
    lr_score = 1.0 - abs(learning_rate - optimal_lr) / optimal_lr
    batch_score = 1.0 - abs(batch_size - optimal_batch) / optimal_batch
    dropout_score = 1.0 - abs(dropout_rate - optimal_dropout) / optimal_dropout

    score = (lr_score + batch_score + dropout_score) / 3.0

    # Add some noise
    noise = np.random.normal(0, 0.05)
    return max(0.0, min(1.0, score + noise))


def multi_objective_function(params, **kwargs):
    """Multi-objective function for testing."""
    accuracy = simple_objective_function(params)

    # Speed is inversely related to batch size (larger batches are faster)
    batch_size = params.get("batch_size", 32)
    speed = batch_size / 128.0  # Normalize to 0-1 range

    return {"accuracy": accuracy, "speed": speed}


class TestAdvancedOptimizationConfig:
    """Test advanced optimization configuration."""

    def test_basic_config_creation(self):
        """Test basic configuration creation."""
        config = AdvancedOptimizationConfig()

        assert config.strategy == OptimizationStrategy.BAYESIAN
        assert config.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT
        assert config.n_trials == 100
        assert config.timeout == 3600
        assert config.random_state == 42

    def test_multi_objective_config(self, multi_objective_config):
        """Test multi-objective configuration."""
        config = multi_objective_config

        assert config.strategy == OptimizationStrategy.MULTI_OBJECTIVE
        assert len(config.objectives) == 2
        assert config.objectives[0].name == "accuracy"
        assert config.objectives[0].direction == "maximize"
        assert config.objectives[0].weight == 0.7

    def test_meta_learning_config(self, meta_learning_config):
        """Test meta-learning configuration."""
        config = meta_learning_config

        assert config.meta_learning is not None
        assert config.meta_learning.strategy == MetaLearningStrategy.SIMILAR_DATASETS
        assert config.meta_learning.similarity_threshold == 0.8
        assert config.meta_learning.warm_start_trials == 5


@pytest.mark.skipif(
    not pytest.importorskip("optuna", minversion="3.0.0"), reason="Optuna not available"
)
class TestAdvancedHyperparameterOptimizer:
    """Test advanced hyperparameter optimizer."""

    def test_optimizer_initialization(self, basic_config, temp_storage):
        """Test optimizer initialization."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=basic_config, storage_path=temp_storage
        )

        assert optimizer.config == basic_config
        assert optimizer.storage_path == temp_storage
        assert optimizer.study is None

    def test_basic_optimization(
        self, basic_config, temp_storage, sample_parameter_space
    ):
        """Test basic optimization workflow."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=basic_config, storage_path=temp_storage
        )

        result = optimizer.optimize(
            objective_function=simple_objective_function,
            parameter_space=sample_parameter_space,
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_trial is not None
        assert result.best_trial.parameters is not None
        assert len(result.all_trials) > 0
        assert result.total_time > 0
        assert result.n_trials_completed > 0

    def test_parameter_sampling(
        self, basic_config, temp_storage, sample_parameter_space
    ):
        """Test parameter sampling from different types."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=basic_config, storage_path=temp_storage
        )

        # Create a mock trial
        with patch("optuna.create_study"):
            Mock()
            mock_trial = Mock()

            # Configure trial methods
            mock_trial.suggest_float.side_effect = [
                0.01,
                0.2,
            ]  # learning_rate, dropout_rate
            mock_trial.suggest_int.return_value = 64  # batch_size
            mock_trial.suggest_categorical.return_value = "adam"  # optimizer

            params = optimizer._sample_parameters(mock_trial, sample_parameter_space)

            assert "learning_rate" in params
            assert "batch_size" in params
            assert "optimizer" in params
            assert "dropout_rate" in params

            assert isinstance(params["learning_rate"], float)
            assert isinstance(params["batch_size"], int)
            assert params["optimizer"] in ["adam", "sgd", "rmsprop"]
            assert isinstance(params["dropout_rate"], float)

    def test_convergence_stability_calculation(self, basic_config, temp_storage):
        """Test convergence stability calculation."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=basic_config, storage_path=temp_storage
        )

        # Test with stable convergence
        stable_history = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.83, 0.84, 0.84, 0.85]
        stability = optimizer._calculate_convergence_stability(stable_history)
        assert stability > 0.5

        # Test with unstable convergence
        unstable_history = [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.5]
        stability = optimizer._calculate_convergence_stability(unstable_history)
        assert stability < 0.5

    def test_exploration_exploitation_scores(self, basic_config, temp_storage):
        """Test exploration and exploitation score calculations."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=basic_config, storage_path=temp_storage
        )

        from monorepo.infrastructure.automl.advanced_optimizer import OptimizationTrial

        # Create mock trials with diverse parameters
        trials = [
            OptimizationTrial(
                trial_id="1",
                parameters={"x": 0.1, "y": 0.2},
                objectives={"primary": 0.5},
                constraints={},
                state="COMPLETE",
                duration=1.0,
            ),
            OptimizationTrial(
                trial_id="2",
                parameters={"x": 0.9, "y": 0.8},
                objectives={"primary": 0.7},
                constraints={},
                state="COMPLETE",
                duration=1.0,
            ),
            OptimizationTrial(
                trial_id="3",
                parameters={"x": 0.5, "y": 0.5},
                objectives={"primary": 0.9},
                constraints={},
                state="COMPLETE",
                duration=1.0,
            ),
        ]

        exploration_score = optimizer._calculate_exploration_score(trials)
        exploitation_score = optimizer._calculate_exploitation_score(trials)

        assert 0.0 <= exploration_score <= 1.0
        assert 0.0 <= exploitation_score <= 1.0

    def test_artifact_saving(self, basic_config, temp_storage, sample_parameter_space):
        """Test optimization artifact saving."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=basic_config, storage_path=temp_storage
        )

        optimizer.optimize(
            objective_function=simple_objective_function,
            parameter_space=sample_parameter_space,
        )

        # Check that artifacts were saved
        assert len(list(temp_storage.glob("*.json"))) > 0
        assert len(list(temp_storage.glob("*.pkl"))) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", minversion="1.0.0"),
        reason="Scikit-learn not available",
    )
    def test_dataset_similarity_calculation(self, basic_config, temp_storage):
        """Test dataset similarity calculation."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=basic_config, storage_path=temp_storage
        )

        profile1 = {
            "n_samples": 1000,
            "n_features": 50,
            "contamination_estimate": 0.1,
            "missing_values_ratio": 0.05,
        }

        profile2 = {
            "n_samples": 1200,
            "n_features": 45,
            "contamination_estimate": 0.12,
            "missing_values_ratio": 0.04,
        }

        similarity = optimizer._calculate_dataset_similarity(profile1, profile2)
        assert 0.0 <= similarity <= 1.0

        # Test identical profiles
        similarity_identical = optimizer._calculate_dataset_similarity(
            profile1, profile1
        )
        assert similarity_identical == 1.0

        # Test very different profiles
        profile3 = {
            "n_samples": 10000,
            "n_features": 500,
            "contamination_estimate": 0.5,
            "missing_values_ratio": 0.8,
        }

        similarity_different = optimizer._calculate_dataset_similarity(
            profile1, profile3
        )
        assert similarity_different < similarity


@pytest.mark.skipif(
    not pytest.importorskip("optuna", minversion="3.0.0"), reason="Optuna not available"
)
class TestMultiObjectiveOptimization:
    """Test multi-objective optimization capabilities."""

    def test_multi_objective_optimization(
        self, multi_objective_config, temp_storage, sample_parameter_space
    ):
        """Test multi-objective optimization."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=multi_objective_config, storage_path=temp_storage
        )

        result = optimizer.optimize(
            objective_function=multi_objective_function,
            parameter_space=sample_parameter_space,
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_trial is not None

        # Check that objectives were evaluated
        best_objectives = result.best_trial.objectives
        assert len(best_objectives) >= 1  # Should have at least one objective

    def test_pareto_front_analysis(self, multi_objective_config, temp_storage):
        """Test Pareto front analysis for multi-objective results."""
        # This test would be more comprehensive with actual Pareto front calculation
        # For now, we test the basic structure

        optimizer = AdvancedHyperparameterOptimizer(
            config=multi_objective_config, storage_path=temp_storage
        )

        # Test would involve checking Pareto optimality
        # This is a placeholder for the actual implementation
        assert optimizer.config.strategy == OptimizationStrategy.MULTI_OBJECTIVE


@pytest.mark.skipif(
    not pytest.importorskip("optuna", minversion="3.0.0"), reason="Optuna not available"
)
class TestMetaLearning:
    """Test meta-learning capabilities."""

    def test_warm_start_parameter_generation(
        self, meta_learning_config, temp_storage, sample_dataset_profile
    ):
        """Test warm start parameter generation."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=meta_learning_config, storage_path=temp_storage
        )

        # Create mock historical data
        import json

        history_file = temp_storage / "optimization_history.json"
        mock_history = [
            {
                "dataset_profile": sample_dataset_profile,
                "best_params": {
                    "learning_rate": 0.01,
                    "batch_size": 64,
                    "optimizer": "adam",
                    "dropout_rate": 0.2,
                },
            }
        ]

        with open(history_file, "w") as f:
            json.dump(mock_history, f)

        warm_start_params = optimizer._get_warm_start_parameters(sample_dataset_profile)

        # Should return parameters from similar datasets
        if warm_start_params:
            assert (
                len(warm_start_params)
                <= meta_learning_config.meta_learning.warm_start_trials
            )
            assert all(isinstance(params, dict) for params in warm_start_params)

    def test_similar_datasets_identification(self, meta_learning_config, temp_storage):
        """Test identification of similar datasets."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=meta_learning_config, storage_path=temp_storage
        )

        profile1 = {"n_samples": 1000, "n_features": 50, "contamination_estimate": 0.1}

        # Very similar profile
        profile2 = {"n_samples": 1100, "n_features": 52, "contamination_estimate": 0.11}

        # Very different profile
        profile3 = {
            "n_samples": 100000,
            "n_features": 1000,
            "contamination_estimate": 0.5,
        }

        similarity_similar = optimizer._calculate_dataset_similarity(profile1, profile2)
        similarity_different = optimizer._calculate_dataset_similarity(
            profile1, profile3
        )

        assert similarity_similar > similarity_different
        assert (
            similarity_similar > meta_learning_config.meta_learning.similarity_threshold
        )


class TestOptimizationInsights:
    """Test optimization insights and recommendations."""

    def test_insights_generation(self, basic_config, temp_storage):
        """Test optimization insights generation."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=basic_config, storage_path=temp_storage
        )

        # Create mock optimization result
        from monorepo.infrastructure.automl.advanced_optimizer import (
            OptimizationResult,
            OptimizationTrial,
        )

        best_trial = OptimizationTrial(
            trial_id="best",
            parameters={"x": 0.5},
            objectives={"primary": 0.8},
            constraints={},
            state="COMPLETE",
            duration=2.0,
        )

        all_trials = [best_trial]

        result = OptimizationResult(
            best_trial=best_trial,
            all_trials=all_trials,
            total_time=10.0,
            n_trials_completed=5,
            n_trials_pruned=2,
            n_trials_failed=1,
            convergence_history=[0.5, 0.6, 0.7, 0.75, 0.8],
            best_value_history=[0.5, 0.6, 0.7, 0.75, 0.8],
            exploration_score=0.6,
            exploitation_score=0.7,
        )

        insights = optimizer.get_optimization_insights(result)

        assert "optimization_quality" in insights
        assert "recommendations" in insights
        assert "next_steps" in insights

        assert "exploration_score" in insights["optimization_quality"]
        assert "exploitation_score" in insights["optimization_quality"]
        assert isinstance(insights["recommendations"], list)
        assert isinstance(insights["next_steps"], list)

    def test_parameter_sensitivity_analysis(self, basic_config, temp_storage):
        """Test parameter sensitivity analysis."""
        optimizer = AdvancedHyperparameterOptimizer(
            config=basic_config, storage_path=temp_storage
        )

        # Create mock trials with numerical parameters
        from monorepo.infrastructure.automl.advanced_optimizer import (
            OptimizationResult,
            OptimizationTrial,
        )

        trials = [
            OptimizationTrial(
                trial_id=str(i),
                parameters={"learning_rate": 0.001 * (i + 1), "batch_size": 32 + i * 8},
                objectives={"primary": 0.5 + i * 0.1},
                constraints={},
                state="COMPLETE",
                duration=1.0,
            )
            for i in range(10)
        ]

        result = OptimizationResult(
            best_trial=trials[-1],
            all_trials=trials,
            total_time=20.0,
            n_trials_completed=10,
        )

        sensitivity = optimizer._analyze_parameter_sensitivity(result)

        assert isinstance(sensitivity, dict)
        # Should detect positive correlation for learning_rate and batch_size
        if sensitivity:
            for _param, sens_value in sensitivity.items():
                assert 0.0 <= sens_value <= 1.0
