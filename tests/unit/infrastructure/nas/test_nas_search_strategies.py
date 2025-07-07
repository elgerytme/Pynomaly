"""Unit tests for Neural Architecture Search strategies."""

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from pynomaly.domain.models.nas import (
    ArchitectureEvaluation,
    NeuralArchitecture,
    OperationType,
    SearchConfiguration,
    SearchExperiment,
    SearchStrategy,
)
from pynomaly.domain.value_objects import ModelMetrics
from pynomaly.infrastructure.nas.search_strategies import (
    BayesianOptimizationStrategy,
    EvolutionarySearchStrategy,
    NASStrategyFactory,
    RandomSearchStrategy,
)


@pytest.fixture
def search_config():
    """Create search configuration for testing."""
    return SearchConfiguration(
        search_strategy=SearchStrategy.RANDOM_SEARCH,
        max_architectures=100,
        max_epochs_per_architecture=10,
        min_layers=2,
        max_layers=8,
        allowed_operations={
            OperationType.CONV_1D,
            OperationType.DENSE,
            OperationType.RELU,
            OperationType.BATCH_NORM,
            OperationType.DROPOUT,
        },
    )


@pytest.fixture
def search_experiment(search_config):
    """Create search experiment for testing."""
    return SearchExperiment(
        experiment_id=uuid4(),
        name="test_experiment",
        configuration=search_config,
        target_dataset_id=uuid4(),
    )


@pytest.fixture
def sample_evaluation():
    """Create sample architecture evaluation."""
    return ArchitectureEvaluation(
        evaluation_id=uuid4(),
        architecture_id=uuid4(),
        metrics=ModelMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
        ),
        training_time_seconds=120.0,
        inference_time_ms=15.0,
        memory_usage_mb=256.0,
        total_params=50000,
        total_flops=1000000,
        model_size_mb=2.0,
        composite_score=0.82,
    )


class TestRandomSearchStrategy:
    """Test random search strategy."""

    def test_strategy_creation(self, search_config):
        """Test random search strategy creation."""
        strategy = RandomSearchStrategy(search_config)
        
        assert strategy.name == "RandomSearch"
        assert strategy.config == search_config
        assert len(strategy.evaluated_architectures) == 0

    async def test_generate_architecture(self, search_config, search_experiment):
        """Test architecture generation."""
        strategy = RandomSearchStrategy(search_config)
        
        architecture = await strategy.generate_architecture(search_experiment)
        
        assert architecture is not None
        assert architecture.architecture_id is not None
        assert len(architecture.cells) >= search_config.min_layers
        assert len(architecture.cells) <= search_config.max_layers
        
        # Check that all operations are from allowed set
        for cell in architecture.cells:
            for op in cell.operations:
                assert op.operation_type in search_config.allowed_operations

    async def test_update_search_state(self, search_config, search_experiment, sample_evaluation):
        """Test search state update."""
        strategy = RandomSearchStrategy(search_config)
        
        # Generate architecture
        architecture = await strategy.generate_architecture(search_experiment)
        
        # Update state
        await strategy.update_search_state(architecture, sample_evaluation, search_experiment)
        
        # Check state was updated
        assert len(strategy.evaluated_architectures) == 1
        assert len(strategy.architecture_history) == 1
        assert architecture.architecture_hash in strategy.evaluated_architectures

    async def test_multiple_architectures_unique(self, search_config, search_experiment):
        """Test that multiple generated architectures are different."""
        strategy = RandomSearchStrategy(search_config)
        
        architectures = []
        for _ in range(5):
            arch = await strategy.generate_architecture(search_experiment)
            architectures.append(arch)
        
        # Check uniqueness by hash
        hashes = [arch.architecture_hash for arch in architectures]
        assert len(set(hashes)) >= 3  # At least some should be different


class TestEvolutionarySearchStrategy:
    """Test evolutionary search strategy."""

    def test_strategy_creation(self, search_config):
        """Test evolutionary strategy creation."""
        search_config.search_strategy = SearchStrategy.EVOLUTIONARY
        strategy = EvolutionarySearchStrategy(search_config)
        
        assert strategy.name == "EvolutionarySearch"
        assert strategy.population_size == search_config.population_size
        assert strategy.mutation_rate == search_config.mutation_rate
        assert strategy.crossover_rate == search_config.crossover_rate
        assert len(strategy.population) == 0

    async def test_initial_population_generation(self, search_config, search_experiment):
        """Test initial population generation."""
        search_config.search_strategy = SearchStrategy.EVOLUTIONARY
        strategy = EvolutionarySearchStrategy(search_config)
        
        # Generate initial architectures
        architectures = []
        for _ in range(3):
            arch = await strategy.generate_architecture(search_experiment)
            architectures.append(arch)
        
        assert len(architectures) == 3
        for arch in architectures:
            assert arch is not None
            assert len(arch.cells) >= search_config.min_layers

    async def test_evolutionary_operations(self, search_config, search_experiment):
        """Test evolutionary operations (crossover and mutation)."""
        search_config.search_strategy = SearchStrategy.EVOLUTIONARY
        search_config.population_size = 5
        strategy = EvolutionarySearchStrategy(search_config)
        
        # Build initial population
        for i in range(5):
            arch = await strategy.generate_architecture(search_experiment)
            eval_result = ArchitectureEvaluation(
                evaluation_id=uuid4(),
                architecture_id=arch.architecture_id,
                metrics=ModelMetrics(accuracy=0.7 + i*0.05),
                training_time_seconds=100.0,
                inference_time_ms=10.0,
                memory_usage_mb=100.0,
                total_params=10000,
                total_flops=100000,
                model_size_mb=1.0,
                composite_score=0.7 + i*0.05,
            )
            
            await strategy.update_search_state(arch, eval_result, search_experiment)
        
        # Generate offspring
        offspring = await strategy.generate_architecture(search_experiment)
        
        assert offspring is not None
        assert offspring.name.startswith("evolved_arch")

    async def test_population_management(self, search_config, search_experiment):
        """Test population size management."""
        search_config.search_strategy = SearchStrategy.EVOLUTIONARY
        search_config.population_size = 3
        strategy = EvolutionarySearchStrategy(search_config)
        
        # Add more than population size
        for i in range(5):
            arch = await strategy.generate_architecture(search_experiment)
            eval_result = ArchitectureEvaluation(
                evaluation_id=uuid4(),
                architecture_id=arch.architecture_id,
                metrics=ModelMetrics(accuracy=0.5 + i*0.1),
                training_time_seconds=100.0,
                inference_time_ms=10.0,
                memory_usage_mb=100.0,
                total_params=10000,
                total_flops=100000,
                model_size_mb=1.0,
                composite_score=0.5 + i*0.1,
            )
            
            await strategy.update_search_state(arch, eval_result, search_experiment)
        
        # Population should be limited to configured size
        assert len(strategy.population) <= search_config.population_size


class TestBayesianOptimizationStrategy:
    """Test Bayesian optimization strategy."""

    def test_strategy_creation(self, search_config):
        """Test Bayesian optimization strategy creation."""
        search_config.search_strategy = SearchStrategy.BAYESIAN_OPTIMIZATION
        strategy = BayesianOptimizationStrategy(search_config)
        
        assert strategy.name == "BayesianOptimization"
        assert len(strategy.observed_architectures) == 0
        assert len(strategy.observed_scores) == 0

    async def test_initial_exploration(self, search_config, search_experiment):
        """Test initial random exploration phase."""
        search_config.search_strategy = SearchStrategy.BAYESIAN_OPTIMIZATION
        strategy = BayesianOptimizationStrategy(search_config)
        
        # Should generate random architectures initially
        for _ in range(3):
            arch = await strategy.generate_architecture(search_experiment)
            assert arch is not None
            assert arch.name.startswith("bo_random_arch")

    async def test_bayesian_optimization_phase(self, search_config, search_experiment):
        """Test Bayesian optimization phase after initial exploration."""
        search_config.search_strategy = SearchStrategy.BAYESIAN_OPTIMIZATION
        strategy = BayesianOptimizationStrategy(search_config)
        
        # Add initial observations
        for i in range(6):  # More than the 5 needed for BO phase
            arch = await strategy.generate_architecture(search_experiment)
            eval_result = ArchitectureEvaluation(
                evaluation_id=uuid4(),
                architecture_id=arch.architecture_id,
                metrics=ModelMetrics(accuracy=0.6 + i*0.05),
                training_time_seconds=100.0,
                inference_time_ms=10.0,
                memory_usage_mb=100.0,
                total_params=10000,
                total_flops=100000,
                model_size_mb=1.0,
                composite_score=0.6 + i*0.05,
            )
            
            await strategy.update_search_state(arch, eval_result, search_experiment)
        
        # Should now use BO
        bo_arch = await strategy.generate_architecture(search_experiment)
        assert bo_arch is not None
        
        # Should have observations stored
        assert len(strategy.observed_architectures) == 6
        assert len(strategy.observed_scores) == 6

    def test_architecture_encoding_decoding(self, search_config):
        """Test architecture encoding and decoding."""
        search_config.search_strategy = SearchStrategy.BAYESIAN_OPTIMIZATION
        strategy = BayesianOptimizationStrategy(search_config)
        
        # Create test architecture
        from pynomaly.domain.models.nas import ArchitectureCell, OperationSpec
        
        cell = ArchitectureCell(
            cell_id=uuid4(),
            name="test_cell",
            operations=[
                OperationSpec(OperationType.CONV_1D, {"filters": 64}),
                OperationSpec(OperationType.RELU),
                OperationSpec(OperationType.DENSE, {"units": 128}),
            ],
        )
        
        architecture = NeuralArchitecture(
            architecture_id=uuid4(),
            name="test_arch",
            cells=[cell],
            input_shape=(100, 10),
            output_shape=(1,),
        )
        
        # Test encoding
        encoding = strategy._encode_architecture(architecture)
        assert len(encoding) == strategy.encoding_dim
        assert encoding[0] == 1  # Number of cells
        
        # Test decoding
        decoded_arch = strategy._decode_architecture(encoding)
        assert decoded_arch is not None
        assert len(decoded_arch.cells) >= search_config.min_layers


class TestNASStrategyFactory:
    """Test NAS strategy factory."""

    def test_create_random_search(self, search_config):
        """Test random search strategy creation."""
        search_config.search_strategy = SearchStrategy.RANDOM_SEARCH
        strategy = NASStrategyFactory.create_strategy(search_config)
        
        assert isinstance(strategy, RandomSearchStrategy)

    def test_create_evolutionary_search(self, search_config):
        """Test evolutionary search strategy creation."""
        search_config.search_strategy = SearchStrategy.EVOLUTIONARY
        strategy = NASStrategyFactory.create_strategy(search_config)
        
        assert isinstance(strategy, EvolutionarySearchStrategy)

    def test_create_bayesian_optimization(self, search_config):
        """Test Bayesian optimization strategy creation."""
        search_config.search_strategy = SearchStrategy.BAYESIAN_OPTIMIZATION
        strategy = NASStrategyFactory.create_strategy(search_config)
        
        assert isinstance(strategy, BayesianOptimizationStrategy)

    def test_unsupported_strategy(self, search_config):
        """Test error handling for unsupported strategies."""
        search_config.search_strategy = SearchStrategy.REINFORCEMENT_LEARNING  # Not implemented
        
        with pytest.raises(ValueError, match="Unsupported search strategy"):
            NASStrategyFactory.create_strategy(search_config)

    def test_get_available_strategies(self):
        """Test getting available strategies."""
        strategies = NASStrategyFactory.get_available_strategies()
        
        assert SearchStrategy.RANDOM_SEARCH in strategies
        assert SearchStrategy.EVOLUTIONARY in strategies
        assert SearchStrategy.BAYESIAN_OPTIMIZATION in strategies
        assert len(strategies) >= 3


class TestSearchConfigurationValidation:
    """Test search configuration validation."""

    def test_valid_configuration(self, search_config):
        """Test valid configuration."""
        assert search_config.min_layers < search_config.max_layers
        assert search_config.max_architectures > 0
        assert 0 <= search_config.mutation_rate <= 1
        assert 0 <= search_config.crossover_rate <= 1

    def test_constraint_validation(self, search_config):
        """Test architecture constraint validation."""
        from pynomaly.domain.models.nas import ArchitectureCell, OperationSpec
        
        # Create architecture that violates constraints
        large_cell = ArchitectureCell(
            cell_id=uuid4(),
            name="large_cell",
            operations=[
                OperationSpec(OperationType.CONV_1D, {"filters": 1024}),
                OperationSpec(OperationType.DENSE, {"units": 2048}),
            ] * 10,  # Very large architecture
        )
        
        large_architecture = NeuralArchitecture(
            architecture_id=uuid4(),
            name="large_arch",
            cells=[large_cell] * 10,
            input_shape=(100, 10),
            output_shape=(1,),
        )
        
        # Set parameter constraint
        search_config.max_params = 10000
        
        # Should fail validation
        assert not search_config.validate_constraints(large_architecture)
        
        # Remove constraint
        search_config.max_params = None
        
        # Should pass validation
        assert search_config.validate_constraints(large_architecture)


class TestAsyncOperations:
    """Test asynchronous operations in search strategies."""

    async def test_concurrent_architecture_generation(self, search_config, search_experiment):
        """Test concurrent architecture generation."""
        strategy = RandomSearchStrategy(search_config)
        
        # Generate multiple architectures concurrently
        tasks = [
            strategy.generate_architecture(search_experiment)
            for _ in range(5)
        ]
        
        architectures = await asyncio.gather(*tasks)
        
        assert len(architectures) == 5
        for arch in architectures:
            assert arch is not None

    async def test_concurrent_state_updates(self, search_config, search_experiment):
        """Test concurrent search state updates."""
        strategy = RandomSearchStrategy(search_config)
        
        # Generate architectures and evaluations
        architectures = []
        evaluations = []
        
        for i in range(3):
            arch = await strategy.generate_architecture(search_experiment)
            eval_result = ArchitectureEvaluation(
                evaluation_id=uuid4(),
                architecture_id=arch.architecture_id,
                metrics=ModelMetrics(accuracy=0.7 + i*0.1),
                training_time_seconds=100.0,
                inference_time_ms=10.0,
                memory_usage_mb=100.0,
                total_params=10000,
                total_flops=100000,
                model_size_mb=1.0,
                composite_score=0.7 + i*0.1,
            )
            
            architectures.append(arch)
            evaluations.append(eval_result)
        
        # Update state concurrently
        tasks = [
            strategy.update_search_state(arch, eval_result, search_experiment)
            for arch, eval_result in zip(architectures, evaluations)
        ]
        
        await asyncio.gather(*tasks)
        
        # Check all updates were processed
        assert len(strategy.evaluated_architectures) == 3
        assert len(strategy.architecture_history) == 3