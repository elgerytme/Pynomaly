"""Neural Architecture Search domain models for automated architecture optimization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.models.base import DomainModel
from pynomaly.domain.value_objects import ModelMetrics


class SearchStrategy(Enum):
    """Neural architecture search strategies."""
    
    RANDOM_SEARCH = "random_search"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRADIENT_BASED = "gradient_based"
    PROGRESSIVE = "progressive"
    DIFFERENTIABLE = "differentiable"


class OperationType(Enum):
    """Types of neural network operations."""
    
    # Convolution operations
    CONV_1D = "conv_1d"
    CONV_2D = "conv_2d"
    DEPTHWISE_CONV = "depthwise_conv"
    SEPARABLE_CONV = "separable_conv"
    DILATED_CONV = "dilated_conv"
    
    # Pooling operations
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"
    GLOBAL_AVG_POOL = "global_avg_pool"
    ADAPTIVE_POOL = "adaptive_pool"
    
    # Dense/Linear operations
    DENSE = "dense"
    LINEAR = "linear"
    
    # Attention operations
    SELF_ATTENTION = "self_attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    CROSS_ATTENTION = "cross_attention"
    
    # Normalization operations
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    GROUP_NORM = "group_norm"
    INSTANCE_NORM = "instance_norm"
    
    # Activation operations
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    
    # Regularization operations
    DROPOUT = "dropout"
    SPATIAL_DROPOUT = "spatial_dropout"
    STOCHASTIC_DEPTH = "stochastic_depth"
    
    # Skip connections
    SKIP_CONNECTION = "skip_connection"
    RESIDUAL_BLOCK = "residual_block"
    DENSE_BLOCK = "dense_block"
    
    # Special operations
    IDENTITY = "identity"
    ZERO = "zero"
    CONCAT = "concat"
    ADD = "add"
    MULTIPLY = "multiply"


class SearchSpace(Enum):
    """Architecture search space types."""
    
    MICRO_SEARCH = "micro_search"  # Search for cell/block structures
    MACRO_SEARCH = "macro_search"  # Search for overall architecture
    MIXED_SEARCH = "mixed_search"  # Combined micro and macro search
    CUSTOM_SEARCH = "custom_search"  # User-defined search space


@dataclass
class OperationSpec:
    """Specification for a neural network operation."""
    
    operation_type: OperationType
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    trainable_params: int = 0
    flops: int = 0  # Floating point operations
    memory_usage: int = 0  # Memory usage in bytes
    
    def __post_init__(self):
        # Set default parameters based on operation type
        if self.operation_type == OperationType.CONV_1D:
            self.parameters.setdefault("filters", 64)
            self.parameters.setdefault("kernel_size", 3)
            self.parameters.setdefault("stride", 1)
            self.parameters.setdefault("padding", "same")
        elif self.operation_type == OperationType.DENSE:
            self.parameters.setdefault("units", 128)
            self.parameters.setdefault("activation", "relu")
        elif self.operation_type == OperationType.DROPOUT:
            self.parameters.setdefault("rate", 0.2)
        elif self.operation_type == OperationType.BATCH_NORM:
            self.parameters.setdefault("momentum", 0.99)
            self.parameters.setdefault("epsilon", 1e-3)
    
    def calculate_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate FLOPs for this operation given input shape."""
        if self.operation_type == OperationType.CONV_1D:
            # Simplified FLOP calculation for 1D convolution
            filters = self.parameters.get("filters", 64)
            kernel_size = self.parameters.get("kernel_size", 3)
            if len(input_shape) >= 2:
                input_channels = input_shape[-1]
                sequence_length = input_shape[-2]
                return filters * kernel_size * input_channels * sequence_length
        elif self.operation_type == OperationType.DENSE:
            units = self.parameters.get("units", 128)
            if len(input_shape) >= 1:
                input_size = input_shape[-1]
                return units * input_size
        
        return 0
    
    def calculate_params(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate trainable parameters for this operation."""
        if self.operation_type == OperationType.CONV_1D:
            filters = self.parameters.get("filters", 64)
            kernel_size = self.parameters.get("kernel_size", 3)
            if len(input_shape) >= 2:
                input_channels = input_shape[-1]
                # Weight parameters + bias parameters
                return (kernel_size * input_channels * filters) + filters
        elif self.operation_type == OperationType.DENSE:
            units = self.parameters.get("units", 128)
            if len(input_shape) >= 1:
                input_size = input_shape[-1]
                return (input_size * units) + units
        elif self.operation_type == OperationType.BATCH_NORM:
            if len(input_shape) >= 1:
                channels = input_shape[-1]
                return channels * 4  # gamma, beta, running_mean, running_var
        
        return 0


@dataclass
class ArchitectureCell:
    """A cell/block in the neural architecture."""
    
    cell_id: UUID
    name: str
    operations: List[OperationSpec]
    connections: Dict[int, List[int]] = field(default_factory=dict)  # node -> [input_nodes]
    input_nodes: List[int] = field(default_factory=list)
    output_nodes: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.input_nodes:
            self.input_nodes = [0]  # Default input node
        if not self.output_nodes and self.operations:
            self.output_nodes = [len(self.operations) - 1]  # Default output node
    
    def get_total_params(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate total trainable parameters in cell."""
        total = 0
        current_shape = input_shape
        
        for op in self.operations:
            total += op.calculate_params(current_shape)
            # Update shape for next operation (simplified)
            if op.output_shape:
                current_shape = op.output_shape
        
        return total
    
    def get_total_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate total FLOPs in cell."""
        total = 0
        current_shape = input_shape
        
        for op in self.operations:
            total += op.calculate_flops(current_shape)
            if op.output_shape:
                current_shape = op.output_shape
        
        return total
    
    def validate_connections(self) -> bool:
        """Validate that cell connections form a valid DAG."""
        # Check for cycles and ensure all nodes are connected
        num_nodes = len(self.operations)
        
        # Simple validation: each node should have at most one path to output
        visited = set()
        
        def dfs(node: int) -> bool:
            if node in visited:
                return False  # Cycle detected
            visited.add(node)
            
            for next_node in self.connections.get(node, []):
                if not dfs(next_node):
                    return False
            
            return True
        
        return all(dfs(node) for node in self.input_nodes)


@dataclass
class NeuralArchitecture(DomainModel):
    """Complete neural network architecture."""
    
    architecture_id: UUID
    name: str
    cells: List[ArchitectureCell]
    global_connections: Dict[int, List[int]] = field(default_factory=dict)
    input_shape: Tuple[int, ...] = (100, 10)  # Default time series shape
    output_shape: Tuple[int, ...] = (1,)  # Default anomaly score
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def total_params(self) -> int:
        """Calculate total trainable parameters."""
        total = 0
        current_shape = self.input_shape
        
        for cell in self.cells:
            total += cell.get_total_params(current_shape)
            # Simplified shape propagation
            if cell.operations and cell.operations[-1].output_shape:
                current_shape = cell.operations[-1].output_shape
        
        return total
    
    @property
    def total_flops(self) -> int:
        """Calculate total FLOPs."""
        total = 0
        current_shape = self.input_shape
        
        for cell in self.cells:
            total += cell.get_total_flops(current_shape)
            if cell.operations and cell.operations[-1].output_shape:
                current_shape = cell.operations[-1].output_shape
        
        return total
    
    @property
    def architecture_hash(self) -> str:
        """Generate hash for architecture comparison."""
        # Create deterministic representation
        arch_dict = {
            "cells": [
                {
                    "operations": [
                        {
                            "type": op.operation_type.value,
                            "params": sorted(op.parameters.items())
                        }
                        for op in cell.operations
                    ],
                    "connections": sorted(cell.connections.items())
                }
                for cell in self.cells
            ],
            "global_connections": sorted(self.global_connections.items()),
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }
        
        arch_str = json.dumps(arch_dict, sort_keys=True)
        return hashlib.sha256(arch_str.encode()).hexdigest()[:16]
    
    def validate_architecture(self) -> Tuple[bool, List[str]]:
        """Validate architecture consistency."""
        errors = []
        
        # Check that all cells are valid
        for i, cell in enumerate(self.cells):
            if not cell.validate_connections():
                errors.append(f"Cell {i} has invalid connections")
        
        # Check global connections
        num_cells = len(self.cells)
        for cell_idx, connections in self.global_connections.items():
            if cell_idx >= num_cells:
                errors.append(f"Global connection references invalid cell {cell_idx}")
            
            for target_cell in connections:
                if target_cell >= num_cells:
                    errors.append(f"Global connection targets invalid cell {target_cell}")
        
        # Check shape compatibility (simplified)
        if not self.input_shape or not self.output_shape:
            errors.append("Input or output shape not specified")
        
        return len(errors) == 0, errors


@dataclass
class SearchConfiguration:
    """Configuration for neural architecture search."""
    
    search_strategy: SearchStrategy
    search_space: SearchSpace
    max_architectures: int = 1000
    max_epochs_per_architecture: int = 10
    early_stopping_patience: int = 3
    population_size: int = 50  # For evolutionary strategies
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    
    # Constraints
    max_params: Optional[int] = None
    max_flops: Optional[int] = None
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    
    # Objective weights
    accuracy_weight: float = 0.7
    efficiency_weight: float = 0.2
    latency_weight: float = 0.1
    
    # Search space parameters
    min_layers: int = 2
    max_layers: int = 20
    min_filters: int = 16
    max_filters: int = 512
    allowed_operations: Set[OperationType] = field(default_factory=lambda: {
        OperationType.CONV_1D,
        OperationType.DENSE,
        OperationType.RELU,
        OperationType.BATCH_NORM,
        OperationType.DROPOUT,
        OperationType.SKIP_CONNECTION,
    })
    
    def validate_constraints(self, architecture: NeuralArchitecture) -> bool:
        """Check if architecture satisfies constraints."""
        if self.max_params and architecture.total_params > self.max_params:
            return False
        
        if self.max_flops and architecture.total_flops > self.max_flops:
            return False
        
        if len(architecture.cells) < self.min_layers or len(architecture.cells) > self.max_layers:
            return False
        
        return True


@dataclass
class ArchitectureEvaluation:
    """Evaluation results for a neural architecture."""
    
    evaluation_id: UUID
    architecture_id: UUID
    metrics: ModelMetrics
    training_time_seconds: float
    inference_time_ms: float
    memory_usage_mb: float
    convergence_epoch: Optional[int] = None
    stability_score: float = 0.0
    robustness_score: float = 0.0
    
    # Architecture characteristics
    total_params: int = 0
    total_flops: int = 0
    model_size_mb: float = 0.0
    
    # Multi-objective score
    composite_score: float = 0.0
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_composite_score(
        self,
        accuracy_weight: float = 0.7,
        efficiency_weight: float = 0.2,
        latency_weight: float = 0.1,
    ) -> float:
        """Calculate weighted composite score."""
        # Normalize metrics to [0, 1]
        accuracy_score = self.metrics.accuracy if self.metrics.accuracy else 0.0
        
        # Efficiency score (inverse of parameters and FLOPs)
        efficiency_score = 1.0 / (1.0 + np.log10(max(self.total_params, 1)))
        
        # Latency score (inverse of inference time)
        latency_score = 1.0 / (1.0 + self.inference_time_ms / 1000.0)
        
        self.composite_score = (
            accuracy_weight * accuracy_score +
            efficiency_weight * efficiency_score +
            latency_weight * latency_score
        )
        
        return self.composite_score


@dataclass
class SearchExperiment(DomainModel):
    """Neural architecture search experiment."""
    
    experiment_id: UUID
    name: str
    configuration: SearchConfiguration
    target_dataset_id: UUID
    
    # Search state
    current_generation: int = 0
    total_architectures_evaluated: int = 0
    best_architecture_id: Optional[UUID] = None
    best_score: float = 0.0
    
    # Search history
    evaluated_architectures: List[UUID] = field(default_factory=list)
    architecture_genealogy: Dict[UUID, List[UUID]] = field(default_factory=dict)  # parent -> children
    pareto_front: List[UUID] = field(default_factory=list)
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Results
    convergence_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    
    @property
    def is_completed(self) -> bool:
        """Check if search experiment is completed."""
        return (
            self.completed_at is not None or
            self.total_architectures_evaluated >= self.configuration.max_architectures
        )
    
    @property
    def search_efficiency(self) -> float:
        """Calculate search efficiency metric."""
        if self.total_architectures_evaluated == 0:
            return 0.0
        
        # Ratio of architectures that improved best score
        improvements = sum(
            1 for score in self.convergence_history 
            if score > (self.best_score * 0.95)  # Within 5% of best
        )
        
        return improvements / self.total_architectures_evaluated
    
    def update_pareto_front(self, evaluations: List[ArchitectureEvaluation]) -> None:
        """Update Pareto front with new evaluations."""
        # Simple Pareto dominance check
        def dominates(eval1: ArchitectureEvaluation, eval2: ArchitectureEvaluation) -> bool:
            better_accuracy = eval1.metrics.accuracy >= eval2.metrics.accuracy
            better_efficiency = eval1.total_params <= eval2.total_params
            better_latency = eval1.inference_time_ms <= eval2.inference_time_ms
            
            return (better_accuracy and better_efficiency and better_latency and
                    (eval1.metrics.accuracy > eval2.metrics.accuracy or
                     eval1.total_params < eval2.total_params or
                     eval1.inference_time_ms < eval2.inference_time_ms))
        
        # Update Pareto front
        new_front = []
        
        for eval_candidate in evaluations:
            is_dominated = False
            
            for eval_front in evaluations:
                if eval_front.evaluation_id != eval_candidate.evaluation_id:
                    if dominates(eval_front, eval_candidate):
                        is_dominated = True
                        break
            
            if not is_dominated:
                new_front.append(eval_candidate.architecture_id)
        
        self.pareto_front = new_front
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        return {
            "experiment_id": str(self.experiment_id),
            "name": self.name,
            "strategy": self.configuration.search_strategy.value,
            "search_space": self.configuration.search_space.value,
            "current_generation": self.current_generation,
            "total_evaluated": self.total_architectures_evaluated,
            "best_score": self.best_score,
            "is_completed": self.is_completed,
            "search_efficiency": self.search_efficiency,
            "pareto_front_size": len(self.pareto_front),
            "convergence_trend": self.convergence_history[-10:] if self.convergence_history else [],
            "diversity_trend": self.diversity_history[-10:] if self.diversity_history else [],
            "elapsed_time": (
                (self.completed_at or datetime.utcnow()) - self.started_at
            ).total_seconds(),
        }