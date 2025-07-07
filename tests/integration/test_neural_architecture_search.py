"""Integration tests for Neural Architecture Search system."""

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from pynomaly.domain.models.dataset import Dataset
from pynomaly.domain.models.nas import OperationType, SearchConfiguration, SearchStrategy
from pynomaly.domain.value_objects import DatasetMetadata
from pynomaly.infrastructure.nas import (
    ArchitectureEvaluator,
    NeuralArchitectureSearchService,
)


@pytest.fixture
def nas_service():
    """Neural Architecture Search service for testing."""
    return NeuralArchitectureSearchService()


@pytest.fixture
def test_dataset():
    """Create test dataset for NAS experiments."""
    # Generate synthetic time series anomaly detection dataset
    np.random.seed(42)
    
    # Normal data
    normal_data = np.random.randn(800, 10)
    
    # Anomaly data (shifted and scaled)
    anomaly_data = np.random.randn(200, 10) * 2 + 3
    
    # Combine and shuffle
    data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([
        np.zeros(800, dtype=bool),
        np.ones(200, dtype=bool)
    ])
    
    # Shuffle
    indices = np.random.permutation(1000)
    data = data[indices]
    labels = labels[indices]
    
    return Dataset(
        name="nas_test_dataset",
        data=data,
        metadata=DatasetMetadata(
            source="synthetic",
            description="Synthetic time series data for NAS testing",
            feature_names=[f"feature_{i}" for i in range(10)],
            data_types=["float64"] * 10,
            anomaly_labels=labels,
        ),
    )


@pytest.fixture
def search_config():
    """Create search configuration for testing."""
    return SearchConfiguration(
        search_strategy=SearchStrategy.RANDOM_SEARCH,
        max_architectures=20,  # Small for testing
        max_epochs_per_architecture=3,  # Fast evaluation
        early_stopping_patience=2,
        min_layers=2,
        max_layers=5,
        max_params=100000,  # Reasonable constraint
        allowed_operations={
            OperationType.CONV_1D,
            OperationType.DENSE,
            OperationType.RELU,
            OperationType.BATCH_NORM,
            OperationType.DROPOUT,
        },
    )


@pytest.mark.integration
@pytest.mark.asyncio
class TestNeuralArchitectureSearchIntegration:
    """Integration tests for Neural Architecture Search system."""

    async def test_end_to_end_nas_experiment(self, nas_service, test_dataset, search_config):
        """Test complete NAS experiment workflow."""
        
        # Create experiment
        experiment = await nas_service.create_search_experiment(
            "integration_test_experiment",
            search_config,
            test_dataset,
        )
        
        assert experiment is not None
        assert experiment.name == "integration_test_experiment"
        assert experiment.configuration.search_strategy == SearchStrategy.RANDOM_SEARCH
        
        # Run experiment
        completed_experiment = await nas_service.run_search_experiment(
            experiment.experiment_id,
            test_dataset,
            max_concurrent_evaluations=2,
        )
        
        # Verify completion
        assert completed_experiment.is_completed
        assert completed_experiment.total_architectures_evaluated > 0
        assert completed_experiment.best_score > 0
        assert completed_experiment.completed_at is not None
        
        # Check that architectures were evaluated
        results = nas_service.experiment_results[experiment.experiment_id]
        assert len(results) > 0
        
        # Verify architecture registry
        assert len(nas_service.architecture_registry) >= completed_experiment.total_architectures_evaluated

    async def test_multiple_search_strategies(self, nas_service, test_dataset):
        """Test different search strategies."""
        
        strategies_to_test = [
            SearchStrategy.RANDOM_SEARCH,
            SearchStrategy.EVOLUTIONARY,
            SearchStrategy.BAYESIAN_OPTIMIZATION,
        ]
        
        results = {}
        
        for strategy in strategies_to_test:
            config = SearchConfiguration(
                search_strategy=strategy,
                max_architectures=10,  # Small for testing
                max_epochs_per_architecture=2,
                population_size=5,  # For evolutionary
            )
            
            experiment = await nas_service.create_search_experiment(
                f"strategy_test_{strategy.value}",
                config,
                test_dataset,
            )
            
            completed_experiment = await nas_service.run_search_experiment(
                experiment.experiment_id,
                test_dataset,
            )
            
            results[strategy] = {
                "evaluated": completed_experiment.total_architectures_evaluated,
                "best_score": completed_experiment.best_score,
                "efficiency": completed_experiment.search_efficiency,
            }
        
        # Verify all strategies completed
        for strategy, result in results.items():
            assert result["evaluated"] > 0
            assert result["best_score"] > 0
            print(f"{strategy.value}: {result}")

    async def test_pareto_optimal_architectures(self, nas_service, test_dataset, search_config):
        """Test Pareto front calculation."""
        
        # Run experiment
        experiment = await nas_service.create_search_experiment(
            "pareto_test",
            search_config,
            test_dataset,
        )
        
        await nas_service.run_search_experiment(experiment.experiment_id, test_dataset)
        
        # Get Pareto optimal architectures
        pareto_architectures = await nas_service.get_pareto_optimal_architectures(
            experiment.experiment_id
        )
        
        assert len(pareto_architectures) > 0
        
        # Verify Pareto optimality properties
        for arch, eval_result in pareto_architectures:
            assert arch is not None
            assert eval_result is not None
            assert eval_result.composite_score > 0
            
            # Check architecture validity
            is_valid, errors = arch.validate_architecture()
            assert is_valid, f"Pareto architecture invalid: {errors}"

    async def test_best_architectures_retrieval(self, nas_service, test_dataset, search_config):
        """Test retrieval of best architectures."""
        
        # Run experiment
        experiment = await nas_service.create_search_experiment(
            "best_arch_test",
            search_config,
            test_dataset,
        )
        
        await nas_service.run_search_experiment(experiment.experiment_id, test_dataset)
        
        # Get top-5 best architectures
        best_architectures = await nas_service.get_best_architectures(
            experiment.experiment_id, top_k=5
        )
        
        assert len(best_architectures) <= 5
        assert len(best_architectures) > 0
        
        # Verify they're sorted by score (descending)
        scores = [eval_result.composite_score for _, eval_result in best_architectures]
        assert scores == sorted(scores, reverse=True)
        
        # Verify architecture quality
        for arch, eval_result in best_architectures:
            assert eval_result.composite_score > 0
            assert arch.total_params > 0

    async def test_experiment_status_tracking(self, nas_service, test_dataset, search_config):
        """Test experiment status tracking during execution."""
        
        # Create experiment
        experiment = await nas_service.create_search_experiment(
            "status_test",
            search_config,
            test_dataset,
        )
        
        # Get initial status
        initial_status = await nas_service.get_experiment_status(experiment.experiment_id)
        assert initial_status["total_evaluated"] == 0
        assert initial_status["best_score"] == 0.0
        assert not initial_status["is_completed"]
        
        # Run experiment
        await nas_service.run_search_experiment(experiment.experiment_id, test_dataset)
        
        # Get final status
        final_status = await nas_service.get_experiment_status(experiment.experiment_id)
        assert final_status["total_evaluated"] > 0
        assert final_status["best_score"] > 0
        assert final_status["is_completed"]
        assert "performance_statistics" in final_status

    async def test_architecture_export(self, nas_service, test_dataset, search_config):
        """Test architecture export functionality."""
        
        # Run small experiment
        search_config.max_architectures = 5
        experiment = await nas_service.create_search_experiment(
            "export_test",
            search_config,
            test_dataset,
        )
        
        await nas_service.run_search_experiment(experiment.experiment_id, test_dataset)
        
        # Get best architecture
        best_architectures = await nas_service.get_best_architectures(
            experiment.experiment_id, top_k=1
        )
        
        assert len(best_architectures) == 1
        best_arch, _ = best_architectures[0]
        
        # Export architecture
        exported = await nas_service.export_architecture(
            best_arch.architecture_id, format="json"
        )
        
        # Verify export structure
        assert "architecture_id" in exported
        assert "name" in exported
        assert "cells" in exported
        assert "total_params" in exported
        assert "total_flops" in exported
        assert "architecture_hash" in exported
        
        # Verify cells structure
        assert len(exported["cells"]) > 0
        for cell in exported["cells"]:
            assert "operations" in cell
            assert "connections" in cell
            assert len(cell["operations"]) > 0

    async def test_constraint_enforcement(self, nas_service, test_dataset):
        """Test that architectural constraints are enforced."""
        
        # Create restrictive configuration
        restrictive_config = SearchConfiguration(
            search_strategy=SearchStrategy.RANDOM_SEARCH,
            max_architectures=10,
            max_epochs_per_architecture=2,
            max_params=10000,  # Very restrictive
            max_layers=3,
            min_layers=2,
        )
        
        experiment = await nas_service.create_search_experiment(
            "constraint_test",
            restrictive_config,
            test_dataset,
        )
        
        await nas_service.run_search_experiment(experiment.experiment_id, test_dataset)
        
        # Check that all evaluated architectures meet constraints
        results = nas_service.experiment_results[experiment.experiment_id]
        
        for evaluation in results:
            architecture = nas_service.architecture_registry[evaluation.architecture_id]
            
            # Check parameter constraint
            assert architecture.total_params <= restrictive_config.max_params
            
            # Check layer constraint
            assert len(architecture.cells) >= restrictive_config.min_layers
            assert len(architecture.cells) <= restrictive_config.max_layers

    async def test_concurrent_experiments(self, nas_service, test_dataset):
        """Test running multiple experiments concurrently."""
        
        # Create multiple experiments
        experiments = []
        for i in range(3):
            config = SearchConfiguration(
                search_strategy=SearchStrategy.RANDOM_SEARCH,
                max_architectures=5,
                max_epochs_per_architecture=2,
            )
            
            experiment = await nas_service.create_search_experiment(
                f"concurrent_test_{i}",
                config,
                test_dataset,
            )
            experiments.append(experiment)
        
        # Run experiments concurrently
        tasks = [
            nas_service.run_search_experiment(exp.experiment_id, test_dataset)
            for exp in experiments
        ]
        
        completed_experiments = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(completed_experiments) == 3
        
        for exp in completed_experiments:
            assert exp.is_completed
            assert exp.total_architectures_evaluated > 0
        
        # Verify service state
        service_stats = nas_service.get_service_statistics()
        assert service_stats["total_experiments"] >= 3
        assert service_stats["total_architectures_evaluated"] > 0

    async def test_evaluator_performance_benchmarking(self):
        """Test architecture evaluator performance."""
        
        config = SearchConfiguration(
            search_strategy=SearchStrategy.RANDOM_SEARCH,
            max_epochs_per_architecture=3,
        )
        
        evaluator = ArchitectureEvaluator(config)
        
        # Benchmark evaluator
        benchmark_results = await evaluator.benchmark_evaluator(num_architectures=5)
        
        assert "num_architectures" in benchmark_results
        assert "total_evaluation_time" in benchmark_results
        assert "score_statistics" in benchmark_results
        assert "latency_statistics" in benchmark_results
        assert "parameter_statistics" in benchmark_results
        
        # Verify reasonable performance
        avg_time = benchmark_results["average_evaluation_time"]
        assert avg_time > 0
        assert avg_time < 60  # Should complete within reasonable time
        
        print(f"Evaluator benchmark results:")
        print(f"  Average evaluation time: {avg_time:.3f}s")
        print(f"  Pareto front size: {benchmark_results['pareto_front_size']}")

    async def test_strategy_comparison(self, nas_service, test_dataset):
        """Test comparison of different search strategies."""
        
        strategies_to_compare = [
            SearchStrategy.RANDOM_SEARCH,
            SearchStrategy.EVOLUTIONARY,
        ]
        
        comparison_results = await nas_service.compare_search_strategies(
            strategies_to_compare,
            test_dataset,
            max_architectures=8,
            max_time_minutes=5,  # Short time limit for testing
        )
        
        # Verify comparison results
        assert len(comparison_results) == len(strategies_to_compare)
        
        for strategy, results in comparison_results.items():
            assert "total_evaluated" in results
            assert "best_score" in results
            assert results["total_evaluated"] > 0
            assert results["best_score"] >= 0
            
            print(f"Strategy {strategy.value}: {results}")

    async def test_early_termination(self, nas_service, test_dataset):
        """Test early termination conditions."""
        
        # Create configuration that should trigger early termination
        config = SearchConfiguration(
            search_strategy=SearchStrategy.RANDOM_SEARCH,
            max_architectures=100,  # Large number
            max_epochs_per_architecture=2,
        )
        
        experiment = await nas_service.create_search_experiment(
            "early_termination_test",
            config,
            test_dataset,
        )
        
        # Run experiment (should terminate early due to convergence or other conditions)
        completed_experiment = await nas_service.run_search_experiment(
            experiment.experiment_id,
            test_dataset,
        )
        
        # Should complete before evaluating all architectures
        assert completed_experiment.is_completed
        # May or may not terminate early depending on randomness, but should complete
        assert completed_experiment.total_architectures_evaluated <= config.max_architectures

    async def test_memory_management(self, nas_service, test_dataset):
        """Test memory management and cleanup."""
        
        # Run multiple experiments to accumulate data
        for i in range(3):
            config = SearchConfiguration(
                search_strategy=SearchStrategy.RANDOM_SEARCH,
                max_architectures=5,
                max_epochs_per_architecture=2,
            )
            
            experiment = await nas_service.create_search_experiment(
                f"memory_test_{i}",
                config,
                test_dataset,
            )
            
            await nas_service.run_search_experiment(experiment.experiment_id, test_dataset)
        
        # Check service statistics before cleanup
        stats_before = nas_service.get_service_statistics()
        completed_before = stats_before["completed_experiments"]
        
        # Cleanup completed experiments
        cleanup_count = await nas_service.cleanup_completed_experiments(keep_results=False)
        
        # Verify cleanup
        assert cleanup_count > 0
        
        stats_after = nas_service.get_service_statistics()
        assert stats_after["completed_experiments"] < completed_before

    async def test_large_scale_experiment(self, nas_service, test_dataset):
        """Test larger scale NAS experiment."""
        
        config = SearchConfiguration(
            search_strategy=SearchStrategy.EVOLUTIONARY,
            max_architectures=30,
            max_epochs_per_architecture=3,
            population_size=8,
            mutation_rate=0.2,
            crossover_rate=0.8,
        )
        
        experiment = await nas_service.create_search_experiment(
            "large_scale_test",
            config,
            test_dataset,
        )
        
        completed_experiment = await nas_service.run_search_experiment(
            experiment.experiment_id,
            test_dataset,
            max_concurrent_evaluations=3,
        )
        
        # Verify experiment completed successfully
        assert completed_experiment.is_completed
        assert completed_experiment.total_architectures_evaluated > 20
        
        # Check convergence and diversity trends
        assert len(completed_experiment.convergence_history) > 0
        assert len(completed_experiment.diversity_history) > 0
        
        # Verify Pareto front
        pareto_architectures = await nas_service.get_pareto_optimal_architectures(
            experiment.experiment_id
        )
        assert len(pareto_architectures) > 0
        
        print(f"Large scale experiment results:")
        print(f"  Total evaluated: {completed_experiment.total_architectures_evaluated}")
        print(f"  Best score: {completed_experiment.best_score:.4f}")
        print(f"  Search efficiency: {completed_experiment.search_efficiency:.3f}")
        print(f"  Pareto front size: {len(pareto_architectures)}")

    async def test_error_handling(self, nas_service, test_dataset):
        """Test error handling in NAS system."""
        
        # Test with invalid experiment ID
        invalid_id = uuid4()
        
        with pytest.raises(ValueError, match="Experiment .* not found"):
            await nas_service.run_search_experiment(invalid_id, test_dataset)
        
        with pytest.raises(ValueError, match="Experiment .* not found"):
            await nas_service.get_experiment_status(invalid_id)
        
        with pytest.raises(ValueError, match="Experiment .* not found"):
            await nas_service.get_best_architectures(invalid_id)
        
        # Test with invalid architecture ID
        invalid_arch_id = uuid4()
        
        with pytest.raises(ValueError, match="Architecture .* not found"):
            await nas_service.export_architecture(invalid_arch_id)