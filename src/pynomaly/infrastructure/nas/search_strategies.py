"""Neural Architecture Search strategies for automated architecture optimization."""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.models.nas import (
    ArchitectureCell,
    ArchitectureEvaluation,
    NeuralArchitecture,
    OperationSpec,
    OperationType,
    SearchConfiguration,
    SearchExperiment,
    SearchStrategy,
)


class NASSearchStrategy(ABC):
    """Abstract base class for Neural Architecture Search strategies."""

    def __init__(self, name: str, config: SearchConfiguration):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Search state
        self.evaluated_architectures: Dict[str, ArchitectureEvaluation] = {}
        self.architecture_history: List[NeuralArchitecture] = []
        
    @abstractmethod
    async def generate_architecture(self, experiment: SearchExperiment) -> NeuralArchitecture:
        """Generate a new architecture candidate."""
        pass
    
    @abstractmethod
    async def update_search_state(
        self, 
        architecture: NeuralArchitecture, 
        evaluation: ArchitectureEvaluation,
        experiment: SearchExperiment,
    ) -> None:
        """Update search strategy state based on evaluation results."""
        pass
    
    def _generate_random_operation(self) -> OperationSpec:
        """Generate a random operation within constraints."""
        op_type = random.choice(list(self.config.allowed_operations))
        
        if op_type == OperationType.CONV_1D:
            return OperationSpec(
                operation_type=op_type,
                parameters={
                    "filters": random.choice([16, 32, 64, 128, 256]),
                    "kernel_size": random.choice([3, 5, 7]),
                    "stride": random.choice([1, 2]),
                    "padding": "same",
                    "activation": random.choice(["relu", "leaky_relu", "swish"]),
                }
            )
        elif op_type == OperationType.DENSE:
            return OperationSpec(
                operation_type=op_type,
                parameters={
                    "units": random.choice([64, 128, 256, 512]),
                    "activation": random.choice(["relu", "leaky_relu", "swish", "tanh"]),
                }
            )
        elif op_type == OperationType.DROPOUT:
            return OperationSpec(
                operation_type=op_type,
                parameters={"rate": random.uniform(0.1, 0.5)}
            )
        elif op_type == OperationType.BATCH_NORM:
            return OperationSpec(
                operation_type=op_type,
                parameters={
                    "momentum": random.uniform(0.9, 0.99),
                    "epsilon": random.choice([1e-3, 1e-4, 1e-5]),
                }
            )
        else:
            return OperationSpec(operation_type=op_type)
    
    def _generate_random_cell(self, cell_name: str) -> ArchitectureCell:
        """Generate a random architecture cell."""
        num_operations = random.randint(2, 6)
        operations = [self._generate_random_operation() for _ in range(num_operations)]
        
        # Create simple sequential connections
        connections = {i: [i-1] for i in range(1, num_operations)}
        
        return ArchitectureCell(
            cell_id=uuid4(),
            name=cell_name,
            operations=operations,
            connections=connections,
            input_nodes=[0],
            output_nodes=[num_operations - 1],
        )
    
    def _validate_and_fix_architecture(self, architecture: NeuralArchitecture) -> NeuralArchitecture:
        """Validate and fix architecture to meet constraints."""
        # Check parameter constraint
        if self.config.max_params and architecture.total_params > self.config.max_params:
            # Reduce architecture complexity
            for cell in architecture.cells:
                for op in cell.operations:
                    if op.operation_type == OperationType.CONV_1D:
                        current_filters = op.parameters.get("filters", 64)
                        op.parameters["filters"] = max(16, current_filters // 2)
                    elif op.operation_type == OperationType.DENSE:
                        current_units = op.parameters.get("units", 128)
                        op.parameters["units"] = max(32, current_units // 2)
        
        # Ensure minimum layer constraint
        while len(architecture.cells) < self.config.min_layers:
            cell = self._generate_random_cell(f"cell_{len(architecture.cells)}")
            architecture.cells.append(cell)
        
        # Ensure maximum layer constraint
        if len(architecture.cells) > self.config.max_layers:
            architecture.cells = architecture.cells[:self.config.max_layers]
        
        return architecture


class RandomSearchStrategy(NASSearchStrategy):
    """Random search strategy for NAS."""

    def __init__(self, config: SearchConfiguration):
        super().__init__("RandomSearch", config)
    
    async def generate_architecture(self, experiment: SearchExperiment) -> NeuralArchitecture:
        """Generate a completely random architecture."""
        num_layers = random.randint(self.config.min_layers, self.config.max_layers)
        
        cells = []
        for i in range(num_layers):
            cell = self._generate_random_cell(f"cell_{i}")
            cells.append(cell)
        
        # Create simple sequential global connections
        global_connections = {i: [i+1] for i in range(num_layers - 1)}
        
        architecture = NeuralArchitecture(
            architecture_id=uuid4(),
            name=f"random_arch_{experiment.total_architectures_evaluated}",
            cells=cells,
            global_connections=global_connections,
            input_shape=(100, 10),  # Default time series shape
            output_shape=(1,),  # Anomaly score
        )
        
        return self._validate_and_fix_architecture(architecture)
    
    async def update_search_state(
        self, 
        architecture: NeuralArchitecture, 
        evaluation: ArchitectureEvaluation,
        experiment: SearchExperiment,
    ) -> None:
        """Random search doesn't maintain state."""
        # Store evaluation for reference
        self.evaluated_architectures[architecture.architecture_hash] = evaluation
        self.architecture_history.append(architecture)


class EvolutionarySearchStrategy(NASSearchStrategy):
    """Evolutionary search strategy using genetic algorithms."""

    def __init__(self, config: SearchConfiguration):
        super().__init__("EvolutionarySearch", config)
        
        # Evolutionary state
        self.population: List[NeuralArchitecture] = []
        self.population_fitness: List[float] = []
        self.generation = 0
        
        # Initialize population
        self.population_size = config.population_size
        self.mutation_rate = config.mutation_rate
        self.crossover_rate = config.crossover_rate
    
    async def generate_architecture(self, experiment: SearchExperiment) -> NeuralArchitecture:
        """Generate architecture using evolutionary operations."""
        
        # Initialize population if empty
        if not self.population:
            return await self._initialize_population()
        
        # Perform evolutionary operations
        if random.random() < self.crossover_rate and len(self.population) >= 2:
            # Crossover
            parent1, parent2 = self._select_parents()
            offspring = await self._crossover(parent1, parent2)
        else:
            # Mutation
            parent = self._select_parent()
            offspring = await self._mutate(parent)
        
        offspring.name = f"evolved_arch_gen{self.generation}_{experiment.total_architectures_evaluated}"
        
        return self._validate_and_fix_architecture(offspring)
    
    async def update_search_state(
        self, 
        architecture: NeuralArchitecture, 
        evaluation: ArchitectureEvaluation,
        experiment: SearchExperiment,
    ) -> None:
        """Update population based on evaluation results."""
        
        # Add to population
        self.population.append(architecture)
        self.population_fitness.append(evaluation.composite_score)
        
        # Store evaluation
        self.evaluated_architectures[architecture.architecture_hash] = evaluation
        self.architecture_history.append(architecture)
        
        # Maintain population size
        if len(self.population) > self.population_size:
            # Remove worst performer
            worst_idx = np.argmin(self.population_fitness)
            self.population.pop(worst_idx)
            self.population_fitness.pop(worst_idx)
        
        # Update generation counter
        if len(self.population) >= self.population_size:
            self.generation += 1
    
    async def _initialize_population(self) -> NeuralArchitecture:
        """Initialize random population."""
        num_layers = random.randint(self.config.min_layers, self.config.max_layers)
        
        cells = []
        for i in range(num_layers):
            cell = self._generate_random_cell(f"init_cell_{i}")
            cells.append(cell)
        
        global_connections = {i: [i+1] for i in range(num_layers - 1)}
        
        return NeuralArchitecture(
            architecture_id=uuid4(),
            name=f"init_arch_{len(self.population)}",
            cells=cells,
            global_connections=global_connections,
            input_shape=(100, 10),
            output_shape=(1,),
        )
    
    def _select_parents(self) -> Tuple[NeuralArchitecture, NeuralArchitecture]:
        """Select two parents using tournament selection."""
        def tournament_select() -> NeuralArchitecture:
            tournament_size = min(3, len(self.population))
            candidates = random.sample(
                list(zip(self.population, self.population_fitness)), 
                tournament_size
            )
            return max(candidates, key=lambda x: x[1])[0]
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        
        return parent1, parent2
    
    def _select_parent(self) -> NeuralArchitecture:
        """Select single parent using tournament selection."""
        tournament_size = min(3, len(self.population))
        candidates = random.sample(
            list(zip(self.population, self.population_fitness)), 
            tournament_size
        )
        return max(candidates, key=lambda x: x[1])[0]
    
    async def _crossover(
        self, 
        parent1: NeuralArchitecture, 
        parent2: NeuralArchitecture
    ) -> NeuralArchitecture:
        """Perform crossover between two parent architectures."""
        
        # Simple crossover: take cells from both parents
        max_cells = min(len(parent1.cells), len(parent2.cells))
        crossover_point = random.randint(1, max(1, max_cells - 1))
        
        # Combine cells
        new_cells = (
            parent1.cells[:crossover_point] + 
            parent2.cells[crossover_point:max_cells]
        )
        
        # Ensure minimum layers
        while len(new_cells) < self.config.min_layers:
            new_cells.append(self._generate_random_cell(f"cross_cell_{len(new_cells)}"))
        
        # Create new global connections
        global_connections = {i: [i+1] for i in range(len(new_cells) - 1)}
        
        return NeuralArchitecture(
            architecture_id=uuid4(),
            name="crossover_offspring",
            cells=new_cells,
            global_connections=global_connections,
            input_shape=parent1.input_shape,
            output_shape=parent1.output_shape,
        )
    
    async def _mutate(self, parent: NeuralArchitecture) -> NeuralArchitecture:
        """Perform mutation on parent architecture."""
        
        # Deep copy parent
        new_cells = []
        for cell in parent.cells:
            new_operations = []
            for op in cell.operations:
                new_op = OperationSpec(
                    operation_type=op.operation_type,
                    parameters=op.parameters.copy(),
                    input_shape=op.input_shape,
                    output_shape=op.output_shape,
                )
                new_operations.append(new_op)
            
            new_cell = ArchitectureCell(
                cell_id=uuid4(),
                name=f"mutated_{cell.name}",
                operations=new_operations,
                connections=cell.connections.copy(),
                input_nodes=cell.input_nodes.copy(),
                output_nodes=cell.output_nodes.copy(),
            )
            new_cells.append(new_cell)
        
        # Apply mutations
        for cell in new_cells:
            if random.random() < self.mutation_rate:
                # Mutate cell operations
                for op in cell.operations:
                    if random.random() < 0.3:  # 30% chance to mutate each operation
                        self._mutate_operation(op)
            
            if random.random() < self.mutation_rate * 0.5:
                # Add new operation
                if len(cell.operations) < 8:  # Max operations per cell
                    new_op = self._generate_random_operation()
                    cell.operations.append(new_op)
                    # Update connections
                    op_idx = len(cell.operations) - 1
                    cell.connections[op_idx] = [op_idx - 1]
                    cell.output_nodes = [op_idx]
            
            if random.random() < self.mutation_rate * 0.2:
                # Remove operation (but keep minimum)
                if len(cell.operations) > 2:
                    remove_idx = random.randint(1, len(cell.operations) - 2)
                    cell.operations.pop(remove_idx)
                    # Update connections
                    new_connections = {}
                    for node, inputs in cell.connections.items():
                        if node > remove_idx:
                            new_connections[node - 1] = [
                                inp - 1 if inp > remove_idx else inp 
                                for inp in inputs if inp != remove_idx
                            ]
                        elif node < remove_idx:
                            new_connections[node] = [
                                inp for inp in inputs if inp != remove_idx
                            ]
                    cell.connections = new_connections
                    cell.output_nodes = [len(cell.operations) - 1]
        
        # Structure mutations
        if random.random() < self.mutation_rate:
            # Add new cell
            if len(new_cells) < self.config.max_layers:
                new_cell = self._generate_random_cell(f"mutated_cell_{len(new_cells)}")
                new_cells.append(new_cell)
        
        if random.random() < self.mutation_rate * 0.5:
            # Remove cell (but keep minimum)
            if len(new_cells) > self.config.min_layers:
                remove_idx = random.randint(1, len(new_cells) - 2)
                new_cells.pop(remove_idx)
        
        # Create new global connections
        global_connections = {i: [i+1] for i in range(len(new_cells) - 1)}
        
        return NeuralArchitecture(
            architecture_id=uuid4(),
            name="mutated_offspring",
            cells=new_cells,
            global_connections=global_connections,
            input_shape=parent.input_shape,
            output_shape=parent.output_shape,
        )
    
    def _mutate_operation(self, operation: OperationSpec) -> None:
        """Mutate parameters of a single operation."""
        
        if operation.operation_type == OperationType.CONV_1D:
            if "filters" in operation.parameters:
                current = operation.parameters["filters"]
                operation.parameters["filters"] = random.choice([
                    max(16, current // 2),
                    current,
                    min(512, current * 2)
                ])
            
            if "kernel_size" in operation.parameters:
                operation.parameters["kernel_size"] = random.choice([3, 5, 7])
        
        elif operation.operation_type == OperationType.DENSE:
            if "units" in operation.parameters:
                current = operation.parameters["units"]
                operation.parameters["units"] = random.choice([
                    max(32, current // 2),
                    current,
                    min(1024, current * 2)
                ])
        
        elif operation.operation_type == OperationType.DROPOUT:
            operation.parameters["rate"] = random.uniform(0.1, 0.5)


class BayesianOptimizationStrategy(NASSearchStrategy):
    """Bayesian optimization strategy for NAS."""

    def __init__(self, config: SearchConfiguration):
        super().__init__("BayesianOptimization", config)
        
        # BO state
        self.observed_architectures: List[np.ndarray] = []
        self.observed_scores: List[float] = []
        self.acquisition_function = "expected_improvement"
        
        # Simple encoding for architectures
        self.encoding_dim = 50  # Fixed encoding dimension
    
    async def generate_architecture(self, experiment: SearchExperiment) -> NeuralArchitecture:
        """Generate architecture using Bayesian optimization."""
        
        if len(self.observed_scores) < 5:
            # Initial random exploration
            return await self._generate_random_architecture()
        else:
            # Use Bayesian optimization
            return await self._generate_bo_architecture()
    
    async def update_search_state(
        self, 
        architecture: NeuralArchitecture, 
        evaluation: ArchitectureEvaluation,
        experiment: SearchExperiment,
    ) -> None:
        """Update Bayesian optimization model."""
        
        # Encode architecture
        encoding = self._encode_architecture(architecture)
        
        # Store observation
        self.observed_architectures.append(encoding)
        self.observed_scores.append(evaluation.composite_score)
        
        # Store evaluation
        self.evaluated_architectures[architecture.architecture_hash] = evaluation
        self.architecture_history.append(architecture)
    
    async def _generate_random_architecture(self) -> NeuralArchitecture:
        """Generate random architecture for initial exploration."""
        num_layers = random.randint(self.config.min_layers, self.config.max_layers)
        
        cells = []
        for i in range(num_layers):
            cell = self._generate_random_cell(f"bo_cell_{i}")
            cells.append(cell)
        
        global_connections = {i: [i+1] for i in range(num_layers - 1)}
        
        return NeuralArchitecture(
            architecture_id=uuid4(),
            name=f"bo_random_arch_{len(self.observed_scores)}",
            cells=cells,
            global_connections=global_connections,
            input_shape=(100, 10),
            output_shape=(1,),
        )
    
    async def _generate_bo_architecture(self) -> NeuralArchitecture:
        """Generate architecture using BO acquisition function."""
        
        # Simple implementation: optimize acquisition function
        best_encoding = None
        best_acquisition = float('-inf')
        
        # Sample candidate encodings
        for _ in range(100):
            candidate_encoding = np.random.randn(self.encoding_dim)
            acquisition_value = self._evaluate_acquisition_function(candidate_encoding)
            
            if acquisition_value > best_acquisition:
                best_acquisition = acquisition_value
                best_encoding = candidate_encoding
        
        # Decode best encoding to architecture
        return self._decode_architecture(best_encoding)
    
    def _encode_architecture(self, architecture: NeuralArchitecture) -> np.ndarray:
        """Encode architecture as fixed-size vector."""
        
        encoding = np.zeros(self.encoding_dim)
        
        # Encode basic statistics
        encoding[0] = len(architecture.cells)
        encoding[1] = architecture.total_params / 1e6  # Normalize
        encoding[2] = architecture.total_flops / 1e9  # Normalize
        
        # Encode operation counts
        op_counts = {}
        for cell in architecture.cells:
            for op in cell.operations:
                op_type = op.operation_type.value
                op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        # Map operation types to encoding indices
        op_mapping = {op.value: i for i, op in enumerate(OperationType)}
        
        for op_type, count in op_counts.items():
            if op_type in op_mapping:
                idx = 3 + op_mapping[op_type]
                if idx < self.encoding_dim:
                    encoding[idx] = count
        
        return encoding
    
    def _decode_architecture(self, encoding: np.ndarray) -> NeuralArchitecture:
        """Decode vector to architecture (simplified)."""
        
        # Extract basic structure
        num_layers = max(self.config.min_layers, 
                        min(self.config.max_layers, int(abs(encoding[0]) * 10)))
        
        cells = []
        for i in range(num_layers):
            # Generate cell based on encoding
            cell = self._generate_random_cell(f"bo_decoded_cell_{i}")
            
            # Modify based on encoding values
            for j, op in enumerate(cell.operations):
                if j + 10 < len(encoding):  # Use later encoding values
                    if op.operation_type == OperationType.CONV_1D:
                        filters = max(16, min(512, int(abs(encoding[j + 10]) * 100)))
                        op.parameters["filters"] = filters
                    elif op.operation_type == OperationType.DENSE:
                        units = max(32, min(1024, int(abs(encoding[j + 10]) * 200)))
                        op.parameters["units"] = units
            
            cells.append(cell)
        
        global_connections = {i: [i+1] for i in range(len(cells) - 1)}
        
        return NeuralArchitecture(
            architecture_id=uuid4(),
            name=f"bo_decoded_arch_{len(self.observed_scores)}",
            cells=cells,
            global_connections=global_connections,
            input_shape=(100, 10),
            output_shape=(1,),
        )
    
    def _evaluate_acquisition_function(self, encoding: np.ndarray) -> float:
        """Evaluate acquisition function (simplified Expected Improvement)."""
        
        if not self.observed_scores:
            return 0.0
        
        # Simple distance-based acquisition
        distances = []
        for obs_encoding in self.observed_architectures:
            dist = np.linalg.norm(encoding - obs_encoding)
            distances.append(dist)
        
        # Encourage exploration of distant points
        min_distance = min(distances) if distances else 1.0
        
        # Simple expected improvement approximation
        best_score = max(self.observed_scores)
        exploration_bonus = min_distance * 0.1
        
        return best_score + exploration_bonus


class NASStrategyFactory:
    """Factory for creating NAS search strategies."""
    
    @staticmethod
    def create_strategy(config: SearchConfiguration) -> NASSearchStrategy:
        """Create search strategy based on configuration."""
        
        if config.search_strategy == SearchStrategy.RANDOM_SEARCH:
            return RandomSearchStrategy(config)
        elif config.search_strategy == SearchStrategy.EVOLUTIONARY:
            return EvolutionarySearchStrategy(config)
        elif config.search_strategy == SearchStrategy.BAYESIAN_OPTIMIZATION:
            return BayesianOptimizationStrategy(config)
        else:
            raise ValueError(f"Unsupported search strategy: {config.search_strategy}")
    
    @staticmethod
    def get_available_strategies() -> List[SearchStrategy]:
        """Get list of available search strategies."""
        return [
            SearchStrategy.RANDOM_SEARCH,
            SearchStrategy.EVOLUTIONARY,
            SearchStrategy.BAYESIAN_OPTIMIZATION,
        ]