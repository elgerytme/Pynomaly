"""
Neural Architecture Search (NAS) Engine

Advanced neural architecture search system with progressive search,
differentiable architecture search (DARTS), and evolutionary methods.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np
import copy
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import structlog

from .advanced_automl_engine import AutoMLTask, OptimizationObjective


class NASMethod(Enum):
    """Neural architecture search methods."""
    RANDOM_SEARCH = "random_search"
    EVOLUTIONARY = "evolutionary"
    DARTS = "darts"
    ENAS = "enas"
    PROGRESSIVE = "progressive"
    HYPERBAND = "hyperband"
    BOHB = "bohb"


class LayerType(Enum):
    """Neural network layer types."""
    DENSE = "dense"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    RESIDUAL = "residual"
    SKIP_CONNECTION = "skip_connection"


class ActivationFunction(Enum):
    """Activation functions."""
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


@dataclass
class LayerSpec:
    """Layer specification for architecture search."""
    layer_type: LayerType
    parameters: Dict[str, Any] = field(default_factory=dict)
    activation: Optional[ActivationFunction] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_type": self.layer_type.value,
            "parameters": self.parameters,
            "activation": self.activation.value if self.activation else None
        }


@dataclass
class Architecture:
    """Neural network architecture representation."""
    architecture_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    layers: List[LayerSpec] = field(default_factory=list)
    
    # Architecture metadata
    input_shape: Tuple[int, ...] = field(default_factory=tuple)
    output_shape: Tuple[int, ...] = field(default_factory=tuple)
    
    # Optimization configuration
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    
    # Training configuration
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Performance metrics
    validation_accuracy: float = 0.0
    validation_loss: float = float('inf')
    training_time_seconds: float = 0.0
    parameters_count: int = 0
    flops_count: int = 0
    memory_usage_mb: float = 0.0
    
    # Search metadata
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "layers": [layer.to_dict() for layer in self.layers],
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_accuracy": self.validation_accuracy,
            "validation_loss": self.validation_loss,
            "training_time_seconds": self.training_time_seconds,
            "parameters_count": self.parameters_count,
            "flops_count": self.flops_count,
            "memory_usage_mb": self.memory_usage_mb,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutations": self.mutations
        }


@dataclass
class NASConfig:
    """Configuration for neural architecture search."""
    method: NASMethod = NASMethod.EVOLUTIONARY
    task_type: AutoMLTask = AutoMLTask.BINARY_CLASSIFICATION
    optimization_objective: OptimizationObjective = OptimizationObjective.ACCURACY
    
    # Search space constraints
    min_layers: int = 2
    max_layers: int = 10
    min_units: int = 16
    max_units: int = 512
    
    # Search parameters
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5
    
    # Training parameters
    max_epochs: int = 50
    early_stopping_patience: int = 5
    validation_split: float = 0.2
    
    # Resource constraints
    max_training_time_minutes: int = 60
    max_memory_gb: float = 4.0
    max_parameters: int = 1000000
    
    # Advanced parameters
    enable_progressive_search: bool = True
    enable_weight_sharing: bool = True
    enable_early_termination: bool = True
    
    # Multi-objective optimization
    pareto_fronts: List[str] = field(default_factory=lambda: ["accuracy", "efficiency"])


class NeuralArchitectureSearchEngine:
    """Advanced neural architecture search engine."""
    
    def __init__(self, config: NASConfig = None):
        self.config = config or NASConfig()
        self.logger = structlog.get_logger(__name__)
        
        # Search state
        self.population: List[Architecture] = []
        self.search_history: List[Architecture] = []
        self.pareto_front: List[Architecture] = []
        
        # Search components
        self.architecture_generator = ArchitectureGenerator(self.config)
        self.architecture_evaluator = ArchitectureEvaluator(self.config)
        self.evolutionary_optimizer = EvolutionaryOptimizer(self.config)
        self.progressive_searcher = ProgressiveSearcher(self.config)
        
        # Performance tracking
        self.search_metrics = {
            "best_accuracy": 0.0,
            "best_architecture_id": None,
            "total_architectures_evaluated": 0,
            "search_time_seconds": 0.0,
            "convergence_generation": None
        }
        
        # Execution resources
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    async def search_architecture(self,
                                X_train: torch.Tensor,
                                y_train: torch.Tensor,
                                X_val: torch.Tensor = None,
                                y_val: torch.Tensor = None) -> Architecture:
        """Perform neural architecture search."""
        
        search_start_time = datetime.utcnow()
        
        self.logger.info(
            "Starting neural architecture search",
            method=self.config.method.value,
            task_type=self.config.task_type.value,
            population_size=self.config.population_size,
            generations=self.config.generations
        )
        
        # Prepare data
        if X_val is None or y_val is None:
            X_val, y_val = self._create_validation_split(X_train, y_train)
        
        # Initialize search based on method
        if self.config.method == NASMethod.EVOLUTIONARY:
            best_architecture = await self._evolutionary_search(X_train, y_train, X_val, y_val)
        elif self.config.method == NASMethod.DARTS:
            best_architecture = await self._darts_search(X_train, y_train, X_val, y_val)
        elif self.config.method == NASMethod.PROGRESSIVE:
            best_architecture = await self._progressive_search(X_train, y_train, X_val, y_val)
        elif self.config.method == NASMethod.HYPERBAND:
            best_architecture = await self._hyperband_search(X_train, y_train, X_val, y_val)
        else:
            best_architecture = await self._random_search(X_train, y_train, X_val, y_val)
        
        # Update search metrics
        search_time = (datetime.utcnow() - search_start_time).total_seconds()
        self.search_metrics.update({
            "search_time_seconds": search_time,
            "total_architectures_evaluated": len(self.search_history),
            "best_accuracy": best_architecture.validation_accuracy,
            "best_architecture_id": best_architecture.architecture_id
        })
        
        self.logger.info(
            "Neural architecture search completed",
            best_accuracy=best_architecture.validation_accuracy,
            search_time_seconds=search_time,
            architectures_evaluated=len(self.search_history)
        )
        
        return best_architecture
    
    async def _evolutionary_search(self,
                                 X_train: torch.Tensor,
                                 y_train: torch.Tensor,
                                 X_val: torch.Tensor,
                                 y_val: torch.Tensor) -> Architecture:
        """Perform evolutionary neural architecture search."""
        
        # Initialize population
        self.population = await self._initialize_population()
        
        # Evaluate initial population
        await self._evaluate_population(X_train, y_train, X_val, y_val)
        
        best_architecture = max(self.population, key=lambda arch: arch.validation_accuracy)
        
        # Evolution loop
        for generation in range(self.config.generations):
            self.logger.info(f"Evolution generation {generation + 1}/{self.config.generations}")
            
            # Selection, crossover, and mutation
            new_population = await self.evolutionary_optimizer.evolve_population(
                self.population, generation
            )
            
            # Evaluate new architectures
            new_architectures = [arch for arch in new_population if arch not in self.population]
            if new_architectures:
                await self._evaluate_architectures(new_architectures, X_train, y_train, X_val, y_val)
            
            self.population = new_population
            
            # Update best architecture
            generation_best = max(self.population, key=lambda arch: arch.validation_accuracy)
            if generation_best.validation_accuracy > best_architecture.validation_accuracy:
                best_architecture = generation_best
                self.logger.info(
                    f"New best architecture found",
                    generation=generation + 1,
                    accuracy=best_architecture.validation_accuracy
                )
            
            # Check for convergence
            if self._check_convergence(generation):
                self.search_metrics["convergence_generation"] = generation + 1
                break
        
        return best_architecture
    
    async def _darts_search(self,
                          X_train: torch.Tensor,
                          y_train: torch.Tensor,
                          X_val: torch.Tensor,
                          y_val: torch.Tensor) -> Architecture:
        """Perform DARTS (Differentiable Architecture Search)."""
        
        # Initialize supernet
        supernet = self._create_supernet()
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.config.batch_size
        )
        
        # Architecture parameters (alpha)
        arch_params = self._initialize_architecture_parameters(supernet)
        
        # Optimizers
        model_optimizer = optim.Adam(supernet.parameters(), lr=0.001)
        arch_optimizer = optim.Adam(arch_params, lr=0.003)
        
        # DARTS training loop
        for epoch in range(self.config.max_epochs):
            # Training phase
            supernet.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Update architecture parameters
                arch_optimizer.zero_grad()
                arch_loss = self._compute_architecture_loss(supernet, arch_params, data, target)
                arch_loss.backward()
                arch_optimizer.step()
                
                # Update model parameters
                model_optimizer.zero_grad()
                model_loss = self._compute_model_loss(supernet, arch_params, data, target)
                model_loss.backward()
                model_optimizer.step()
            
            # Validation
            val_accuracy = await self._evaluate_supernet(supernet, arch_params, val_loader)
            
            self.logger.info(f"DARTS epoch {epoch + 1}: validation accuracy = {val_accuracy:.4f}")
        
        # Derive final architecture
        final_architecture = self._derive_architecture_from_supernet(supernet, arch_params)
        
        # Evaluate final architecture
        final_architecture = await self._evaluate_single_architecture(
            final_architecture, X_train, y_train, X_val, y_val
        )
        
        return final_architecture
    
    async def _progressive_search(self,
                                X_train: torch.Tensor,
                                y_train: torch.Tensor,
                                X_val: torch.Tensor,
                                y_val: torch.Tensor) -> Architecture:
        """Perform progressive neural architecture search."""
        
        return await self.progressive_searcher.search(X_train, y_train, X_val, y_val)
    
    async def _hyperband_search(self,
                              X_train: torch.Tensor,
                              y_train: torch.Tensor,
                              X_val: torch.Tensor,
                              y_val: torch.Tensor) -> Architecture:
        """Perform Hyperband-based architecture search."""
        
        # Hyperband parameters
        max_iter = 81  # Maximum iterations
        eta = 3  # Reduction factor
        
        # Generate configurations
        configs = await self._generate_random_architectures(max_iter)
        
        best_architecture = None
        best_score = 0.0
        
        # Hyperband successive halving
        for iteration in range(int(np.log(max_iter) / np.log(eta)) + 1):
            n_configs = len(configs)
            n_iterations = max_iter // (eta ** iteration)
            
            self.logger.info(
                f"Hyperband iteration {iteration + 1}: "
                f"{n_configs} configs, {n_iterations} iterations each"
            )
            
            # Train and evaluate configurations
            results = []
            for arch in configs:
                # Train for n_iterations epochs
                temp_arch = copy.deepcopy(arch)
                temp_arch.epochs = n_iterations
                
                evaluated_arch = await self._evaluate_single_architecture(
                    temp_arch, X_train, y_train, X_val, y_val
                )
                results.append((evaluated_arch.validation_accuracy, evaluated_arch))
            
            # Select top configurations
            results.sort(key=lambda x: x[0], reverse=True)
            n_survivors = max(1, n_configs // eta)
            configs = [arch for _, arch in results[:n_survivors]]
            
            # Update best architecture
            if results[0][0] > best_score:
                best_score = results[0][0]
                best_architecture = results[0][1]
        
        return best_architecture
    
    async def _random_search(self,
                           X_train: torch.Tensor,
                           y_train: torch.Tensor,
                           X_val: torch.Tensor,
                           y_val: torch.Tensor) -> Architecture:
        """Perform random architecture search."""
        
        best_architecture = None
        best_score = 0.0
        
        n_trials = self.config.population_size * self.config.generations
        
        for trial in range(n_trials):
            self.logger.info(f"Random search trial {trial + 1}/{n_trials}")
            
            # Generate random architecture
            architecture = await self.architecture_generator.generate_random_architecture()
            
            # Evaluate architecture
            evaluated_arch = await self._evaluate_single_architecture(
                architecture, X_train, y_train, X_val, y_val
            )
            
            # Update best
            if evaluated_arch.validation_accuracy > best_score:
                best_score = evaluated_arch.validation_accuracy
                best_architecture = evaluated_arch
                
                self.logger.info(
                    f"New best architecture found",
                    trial=trial + 1,
                    accuracy=best_score
                )
        
        return best_architecture
    
    async def _initialize_population(self) -> List[Architecture]:
        """Initialize population for evolutionary search."""
        
        population = []
        
        for _ in range(self.config.population_size):
            architecture = await self.architecture_generator.generate_random_architecture()
            population.append(architecture)
        
        return population
    
    async def _evaluate_population(self,
                                 X_train: torch.Tensor,
                                 y_train: torch.Tensor,
                                 X_val: torch.Tensor,
                                 y_val: torch.Tensor) -> None:
        """Evaluate entire population."""
        
        await self._evaluate_architectures(self.population, X_train, y_train, X_val, y_val)
    
    async def _evaluate_architectures(self,
                                    architectures: List[Architecture],
                                    X_train: torch.Tensor,
                                    y_train: torch.Tensor,
                                    X_val: torch.Tensor,
                                    y_val: torch.Tensor) -> None:
        """Evaluate multiple architectures."""
        
        tasks = []
        for architecture in architectures:
            task = asyncio.create_task(
                self._evaluate_single_architecture(architecture, X_train, y_train, X_val, y_val)
            )
            tasks.append(task)
        
        evaluated_architectures = await asyncio.gather(*tasks)
        
        # Update architectures in place
        for i, evaluated_arch in enumerate(evaluated_architectures):
            architectures[i] = evaluated_arch
        
        # Add to search history
        self.search_history.extend(evaluated_architectures)
    
    async def _evaluate_single_architecture(self,
                                          architecture: Architecture,
                                          X_train: torch.Tensor,
                                          y_train: torch.Tensor,
                                          X_val: torch.Tensor,
                                          y_val: torch.Tensor) -> Architecture:
        """Evaluate a single architecture."""
        
        return await self.architecture_evaluator.evaluate(
            architecture, X_train, y_train, X_val, y_val
        )
    
    def _create_validation_split(self,
                               X_train: torch.Tensor,
                               y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create validation split from training data."""
        
        n_samples = X_train.shape[0]
        n_val = int(n_samples * self.config.validation_split)
        
        indices = torch.randperm(n_samples)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        return X_val, y_val
    
    def _check_convergence(self, generation: int) -> bool:
        """Check if search has converged."""
        
        if generation < 10:  # Minimum generations
            return False
        
        # Check if best score hasn't improved in last 10 generations
        recent_best_scores = []
        for arch in self.search_history[-self.config.population_size * 10:]:
            recent_best_scores.append(arch.validation_accuracy)
        
        if len(recent_best_scores) < 10:
            return False
        
        # Check for improvement
        first_half = np.mean(recent_best_scores[:len(recent_best_scores)//2])
        second_half = np.mean(recent_best_scores[len(recent_best_scores)//2:])
        
        improvement = (second_half - first_half) / first_half if first_half > 0 else 0
        
        return improvement < 0.01  # Less than 1% improvement
    
    async def get_search_results(self) -> Dict[str, Any]:
        """Get comprehensive search results."""
        
        # Find Pareto front
        pareto_front = self._compute_pareto_front()
        
        results = {
            "search_metrics": self.search_metrics,
            "best_architecture": max(self.search_history, key=lambda arch: arch.validation_accuracy).to_dict() if self.search_history else None,
            "pareto_front": [arch.to_dict() for arch in pareto_front],
            "search_history": [arch.to_dict() for arch in self.search_history],
            "population_diversity": self._calculate_population_diversity(),
            "convergence_analysis": self._analyze_convergence()
        }
        
        return results
    
    def _compute_pareto_front(self) -> List[Architecture]:
        """Compute Pareto front of architectures."""
        
        if not self.search_history:
            return []
        
        pareto_front = []
        
        for arch1 in self.search_history:
            is_pareto_optimal = True
            
            for arch2 in self.search_history:
                if arch1 == arch2:
                    continue
                
                # Check if arch2 dominates arch1
                dominates = True
                
                # Accuracy (maximize)
                if arch2.validation_accuracy <= arch1.validation_accuracy:
                    dominates = False
                
                # Efficiency (minimize parameters, maximize accuracy)
                efficiency1 = arch1.validation_accuracy / (arch1.parameters_count / 1000000)
                efficiency2 = arch2.validation_accuracy / (arch2.parameters_count / 1000000)
                
                if efficiency2 <= efficiency1:
                    dominates = False
                
                if dominates:
                    is_pareto_optimal = False
                    break
            
            if is_pareto_optimal:
                pareto_front.append(arch1)
        
        return pareto_front
    
    def _calculate_population_diversity(self) -> float:
        """Calculate diversity of current population."""
        
        if len(self.population) < 2:
            return 0.0
        
        # Calculate architectural diversity based on layer configurations
        diversity_scores = []
        
        for i, arch1 in enumerate(self.population):
            for j, arch2 in enumerate(self.population[i+1:], i+1):
                similarity = self._calculate_architecture_similarity(arch1, arch2)
                diversity_scores.append(1.0 - similarity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_architecture_similarity(self, arch1: Architecture, arch2: Architecture) -> float:
        """Calculate similarity between two architectures."""
        
        if len(arch1.layers) != len(arch2.layers):
            return 0.0
        
        similarities = []
        
        for layer1, layer2 in zip(arch1.layers, arch2.layers):
            layer_similarity = 1.0 if layer1.layer_type == layer2.layer_type else 0.0
            similarities.append(layer_similarity)
        
        return np.mean(similarities)
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze search convergence patterns."""
        
        if not self.search_history:
            return {}
        
        # Group by generation if available
        generation_best = {}
        for arch in self.search_history:
            gen = arch.generation
            if gen not in generation_best or arch.validation_accuracy > generation_best[gen]:
                generation_best[gen] = arch.validation_accuracy
        
        convergence_data = {
            "generation_best_scores": generation_best,
            "improvement_rate": self._calculate_improvement_rate(),
            "diversity_over_time": self._calculate_diversity_over_time()
        }
        
        return convergence_data
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate rate of improvement over generations."""
        
        if len(self.search_history) < 2:
            return 0.0
        
        sorted_history = sorted(self.search_history, key=lambda arch: arch.generation)
        
        improvements = []
        best_so_far = 0.0
        
        for arch in sorted_history:
            if arch.validation_accuracy > best_so_far:
                improvement = arch.validation_accuracy - best_so_far
                improvements.append(improvement)
                best_so_far = arch.validation_accuracy
        
        return np.mean(improvements) if improvements else 0.0
    
    def _calculate_diversity_over_time(self) -> List[float]:
        """Calculate population diversity over generations."""
        
        diversity_timeline = []
        
        # Group architectures by generation
        generations = {}
        for arch in self.search_history:
            gen = arch.generation
            if gen not in generations:
                generations[gen] = []
            generations[gen].append(arch)
        
        # Calculate diversity for each generation
        for gen in sorted(generations.keys()):
            gen_population = generations[gen]
            if len(gen_population) < 2:
                diversity_timeline.append(0.0)
                continue
            
            diversity_scores = []
            for i, arch1 in enumerate(gen_population):
                for j, arch2 in enumerate(gen_population[i+1:], i+1):
                    similarity = self._calculate_architecture_similarity(arch1, arch2)
                    diversity_scores.append(1.0 - similarity)
            
            diversity_timeline.append(np.mean(diversity_scores) if diversity_scores else 0.0)
        
        return diversity_timeline


class ArchitectureGenerator:
    """Generates neural network architectures."""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    async def generate_random_architecture(self) -> Architecture:
        """Generate a random neural network architecture."""
        
        architecture = Architecture()
        
        # Determine number of layers
        n_layers = random.randint(self.config.min_layers, self.config.max_layers)
        
        # Generate layers
        for i in range(n_layers):
            layer_type = self._sample_layer_type(i, n_layers)
            layer_spec = self._generate_layer_spec(layer_type, i, n_layers)
            architecture.layers.append(layer_spec)
        
        # Set architecture parameters
        architecture.optimizer = random.choice(["adam", "sgd", "rmsprop"])
        architecture.learning_rate = 10 ** random.uniform(-4, -1)  # 0.0001 to 0.1
        architecture.weight_decay = 10 ** random.uniform(-6, -2)  # 0.000001 to 0.01
        architecture.batch_size = random.choice([16, 32, 64, 128])
        
        return architecture
    
    def _sample_layer_type(self, layer_index: int, total_layers: int) -> LayerType:
        """Sample layer type based on position in network."""
        
        # First layer: usually dense or convolutional
        if layer_index == 0:
            return random.choice([LayerType.DENSE, LayerType.CONV1D])
        
        # Last layer: typically dense for final prediction
        elif layer_index == total_layers - 1:
            return LayerType.DENSE
        
        # Middle layers: variety of options
        else:
            layer_types = [LayerType.DENSE, LayerType.DROPOUT, LayerType.BATCH_NORM]
            
            # Add recurrent layers for sequence tasks
            if self.config.task_type == AutoMLTask.TIME_SERIES_FORECASTING:
                layer_types.extend([LayerType.LSTM, LayerType.GRU])
            
            return random.choice(layer_types)
    
    def _generate_layer_spec(self, layer_type: LayerType, layer_index: int, total_layers: int) -> LayerSpec:
        """Generate layer specification."""
        
        layer_spec = LayerSpec(layer_type=layer_type)
        
        if layer_type == LayerType.DENSE:
            # Dense layer parameters
            if layer_index == total_layers - 1:
                # Output layer
                units = self._get_output_units()
            else:
                units = random.randint(self.config.min_units, self.config.max_units)
            
            layer_spec.parameters = {"units": units}
            layer_spec.activation = self._sample_activation(layer_index, total_layers)
        
        elif layer_type == LayerType.CONV1D:
            layer_spec.parameters = {
                "filters": random.choice([32, 64, 128, 256]),
                "kernel_size": random.choice([3, 5, 7]),
                "stride": random.choice([1, 2]),
                "padding": "same"
            }
            layer_spec.activation = ActivationFunction.RELU
        
        elif layer_type == LayerType.LSTM:
            layer_spec.parameters = {
                "units": random.choice([50, 100, 200]),
                "return_sequences": layer_index < total_layers - 2,
                "dropout": random.uniform(0.0, 0.3)
            }
        
        elif layer_type == LayerType.GRU:
            layer_spec.parameters = {
                "units": random.choice([50, 100, 200]),
                "return_sequences": layer_index < total_layers - 2,
                "dropout": random.uniform(0.0, 0.3)
            }
        
        elif layer_type == LayerType.DROPOUT:
            layer_spec.parameters = {
                "rate": random.uniform(0.1, 0.5)
            }
        
        elif layer_type == LayerType.BATCH_NORM:
            layer_spec.parameters = {}
        
        return layer_spec
    
    def _get_output_units(self) -> int:
        """Get number of output units based on task type."""
        
        if self.config.task_type == AutoMLTask.BINARY_CLASSIFICATION:
            return 1
        elif self.config.task_type == AutoMLTask.MULTICLASS_CLASSIFICATION:
            return 10  # Default, should be configured based on actual data
        elif self.config.task_type == AutoMLTask.REGRESSION:
            return 1
        else:
            return 1
    
    def _sample_activation(self, layer_index: int, total_layers: int) -> ActivationFunction:
        """Sample activation function."""
        
        # Output layer activation
        if layer_index == total_layers - 1:
            if self.config.task_type == AutoMLTask.BINARY_CLASSIFICATION:
                return ActivationFunction.SIGMOID
            elif self.config.task_type == AutoMLTask.MULTICLASS_CLASSIFICATION:
                return ActivationFunction.SOFTMAX
            else:
                return None  # Linear for regression
        
        # Hidden layer activations
        return random.choice([
            ActivationFunction.RELU,
            ActivationFunction.LEAKY_RELU,
            ActivationFunction.ELU,
            ActivationFunction.SWISH,
            ActivationFunction.GELU
        ])


class ArchitectureEvaluator:
    """Evaluates neural network architectures."""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    async def evaluate(self,
                      architecture: Architecture,
                      X_train: torch.Tensor,
                      y_train: torch.Tensor,
                      X_val: torch.Tensor,
                      y_val: torch.Tensor) -> Architecture:
        """Evaluate architecture performance."""
        
        start_time = datetime.utcnow()
        
        try:
            # Build PyTorch model from architecture
            model = self._build_model(architecture, X_train.shape[1:])
            model = model.to(self.device)
            
            # Calculate model statistics
            architecture.parameters_count = sum(p.numel() for p in model.parameters())
            architecture.memory_usage_mb = self._estimate_memory_usage(model, X_train.shape)
            
            # Check resource constraints
            if architecture.parameters_count > self.config.max_parameters:
                architecture.validation_accuracy = 0.0
                architecture.validation_loss = float('inf')
                return architecture
            
            # Train and evaluate model
            val_accuracy, val_loss = await self._train_and_evaluate(
                model, architecture, X_train, y_train, X_val, y_val
            )
            
            architecture.validation_accuracy = val_accuracy
            architecture.validation_loss = val_loss
            architecture.training_time_seconds = (datetime.utcnow() - start_time).total_seconds()
            
        except Exception as e:
            self.logger.warning(f"Architecture evaluation failed: {str(e)}")
            architecture.validation_accuracy = 0.0
            architecture.validation_loss = float('inf')
            architecture.training_time_seconds = (datetime.utcnow() - start_time).total_seconds()
        
        return architecture
    
    def _build_model(self, architecture: Architecture, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build PyTorch model from architecture specification."""
        
        layers = []
        current_dim = input_shape[0] if len(input_shape) == 1 else np.prod(input_shape)
        
        for i, layer_spec in enumerate(architecture.layers):
            if layer_spec.layer_type == LayerType.DENSE:
                units = layer_spec.parameters["units"]
                layers.append(nn.Linear(current_dim, units))
                current_dim = units
                
                # Add activation
                if layer_spec.activation:
                    layers.append(self._get_activation_layer(layer_spec.activation))
            
            elif layer_spec.layer_type == LayerType.DROPOUT:
                rate = layer_spec.parameters["rate"]
                layers.append(nn.Dropout(rate))
            
            elif layer_spec.layer_type == LayerType.BATCH_NORM:
                layers.append(nn.BatchNorm1d(current_dim))
        
        return nn.Sequential(*layers)
    
    def _get_activation_layer(self, activation: ActivationFunction) -> nn.Module:
        """Get PyTorch activation layer."""
        
        activation_map = {
            ActivationFunction.RELU: nn.ReLU(),
            ActivationFunction.LEAKY_RELU: nn.LeakyReLU(),
            ActivationFunction.ELU: nn.ELU(),
            ActivationFunction.TANH: nn.Tanh(),
            ActivationFunction.SIGMOID: nn.Sigmoid(),
            ActivationFunction.SOFTMAX: nn.Softmax(dim=-1)
        }
        
        return activation_map.get(activation, nn.ReLU())
    
    async def _train_and_evaluate(self,
                                model: nn.Module,
                                architecture: Architecture,
                                X_train: torch.Tensor,
                                y_train: torch.Tensor,
                                X_val: torch.Tensor,
                                y_val: torch.Tensor) -> Tuple[float, float]:
        """Train and evaluate model."""
        
        # Move data to device
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=architecture.batch_size, shuffle=True)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=architecture.learning_rate, weight_decay=architecture.weight_decay)
        
        if self.config.task_type == AutoMLTask.REGRESSION:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_accuracy = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(min(architecture.epochs, self.config.max_epochs)):
            # Training
            model.train()
            for batch_data, batch_target in train_loader:
                optimizer.zero_grad()
                output = model(batch_data)
                
                if self.config.task_type == AutoMLTask.BINARY_CLASSIFICATION:
                    output = output.squeeze()
                    loss = criterion(output, batch_target.float())
                else:
                    loss = criterion(output, batch_target.long())
                
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                
                if self.config.task_type == AutoMLTask.REGRESSION:
                    val_loss = criterion(val_output.squeeze(), y_val.float()).item()
                    # For regression, use R² as accuracy metric
                    val_accuracy = self._calculate_r2_score(y_val, val_output.squeeze())
                
                elif self.config.task_type == AutoMLTask.BINARY_CLASSIFICATION:
                    val_output = val_output.squeeze()
                    val_loss = criterion(val_output, y_val.float()).item()
                    val_predictions = (torch.sigmoid(val_output) > 0.5).float()
                    val_accuracy = (val_predictions == y_val).float().mean().item()
                
                else:  # Multiclass classification
                    val_loss = criterion(val_output, y_val.long()).item()
                    val_predictions = torch.argmax(val_output, dim=1)
                    val_accuracy = (val_predictions == y_val).float().mean().item()
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    break
        
        return best_val_accuracy, best_val_loss
    
    def _calculate_r2_score(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Calculate R² score for regression."""
        
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()
    
    def _estimate_memory_usage(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Estimate memory usage of model in MB."""
        
        # Parameter memory
        param_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2)  # 4 bytes per float32
        
        # Activation memory (rough estimate)
        batch_size = 32  # Default batch size for estimation
        activation_memory = np.prod(input_shape) * batch_size * 4 / (1024 ** 2)
        
        return param_memory + activation_memory


class EvolutionaryOptimizer:
    """Evolutionary optimization for neural architectures."""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    async def evolve_population(self, population: List[Architecture], generation: int) -> List[Architecture]:
        """Evolve population using genetic algorithms."""
        
        # Selection
        selected_parents = self._selection(population)
        
        # Crossover and mutation
        new_population = []
        
        # Keep best individuals (elitism)
        elite_size = max(1, int(0.1 * self.config.population_size))
        elite = sorted(population, key=lambda arch: arch.validation_accuracy, reverse=True)[:elite_size]
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            
            if random.random() < self.config.crossover_rate:
                child = self._crossover(parent1, parent2, generation)
            else:
                child = copy.deepcopy(random.choice([parent1, parent2]))
                child.generation = generation
            
            if random.random() < self.config.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population[:self.config.population_size]
    
    def _selection(self, population: List[Architecture]) -> List[Architecture]:
        """Tournament selection."""
        
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda arch: arch.validation_accuracy)
            selected.append(winner)
        
        return selected
    
    def _crossover(self, parent1: Architecture, parent2: Architecture, generation: int) -> Architecture:
        """Single-point crossover for architectures."""
        
        child = Architecture()
        child.generation = generation
        child.parent_ids = [parent1.architecture_id, parent2.architecture_id]
        
        # Crossover layers
        min_layers = min(len(parent1.layers), len(parent2.layers))
        if min_layers > 0:
            crossover_point = random.randint(0, min_layers - 1)
            
            child.layers = (
                parent1.layers[:crossover_point] + 
                parent2.layers[crossover_point:min_layers]
            )
        
        # Crossover hyperparameters
        child.optimizer = random.choice([parent1.optimizer, parent2.optimizer])
        child.learning_rate = random.choice([parent1.learning_rate, parent2.learning_rate])
        child.weight_decay = random.choice([parent1.weight_decay, parent2.weight_decay])
        child.batch_size = random.choice([parent1.batch_size, parent2.batch_size])
        
        child.mutations.append("crossover")
        
        return child
    
    def _mutate(self, architecture: Architecture) -> Architecture:
        """Mutate architecture."""
        
        mutation_types = ["add_layer", "remove_layer", "modify_layer", "modify_hyperparams"]
        mutation_type = random.choice(mutation_types)
        
        if mutation_type == "add_layer" and len(architecture.layers) < self.config.max_layers:
            # Add random layer
            from .neural_architecture_search import ArchitectureGenerator
            generator = ArchitectureGenerator(self.config)
            
            new_layer = generator._generate_layer_spec(
                LayerType.DENSE, 
                len(architecture.layers), 
                len(architecture.layers) + 1
            )
            
            insert_position = random.randint(0, len(architecture.layers))
            architecture.layers.insert(insert_position, new_layer)
            architecture.mutations.append("add_layer")
        
        elif mutation_type == "remove_layer" and len(architecture.layers) > self.config.min_layers:
            # Remove random layer (except output layer)
            if len(architecture.layers) > 1:
                remove_position = random.randint(0, len(architecture.layers) - 2)
                architecture.layers.pop(remove_position)
                architecture.mutations.append("remove_layer")
        
        elif mutation_type == "modify_layer" and architecture.layers:
            # Modify random layer
            layer_index = random.randint(0, len(architecture.layers) - 1)
            layer = architecture.layers[layer_index]
            
            if layer.layer_type == LayerType.DENSE and "units" in layer.parameters:
                layer.parameters["units"] = random.randint(self.config.min_units, self.config.max_units)
            
            elif layer.layer_type == LayerType.DROPOUT and "rate" in layer.parameters:
                layer.parameters["rate"] = random.uniform(0.1, 0.5)
            
            architecture.mutations.append("modify_layer")
        
        elif mutation_type == "modify_hyperparams":
            # Modify hyperparameters
            if random.random() < 0.5:
                architecture.learning_rate *= random.uniform(0.5, 2.0)
                architecture.learning_rate = max(1e-5, min(1e-1, architecture.learning_rate))
            
            if random.random() < 0.5:
                architecture.weight_decay *= random.uniform(0.1, 10.0)
                architecture.weight_decay = max(1e-6, min(1e-1, architecture.weight_decay))
            
            architecture.mutations.append("modify_hyperparams")
        
        return architecture


class ProgressiveSearcher:
    """Progressive neural architecture search."""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    async def search(self,
                   X_train: torch.Tensor,
                   y_train: torch.Tensor,
                   X_val: torch.Tensor,
                   y_val: torch.Tensor) -> Architecture:
        """Perform progressive architecture search."""
        
        # Start with simple architectures and progressively make them more complex
        best_architecture = None
        best_score = 0.0
        
        # Progressive complexity levels
        complexity_levels = [
            {"max_layers": 2, "max_units": 64},
            {"max_layers": 4, "max_units": 128},
            {"max_layers": 6, "max_units": 256},
            {"max_layers": 8, "max_units": 512}
        ]
        
        for level_idx, level_config in enumerate(complexity_levels):
            self.logger.info(f"Progressive search level {level_idx + 1}/{len(complexity_levels)}")
            
            # Update configuration for this level
            temp_config = copy.deepcopy(self.config)
            temp_config.max_layers = level_config["max_layers"]
            temp_config.max_units = level_config["max_units"]
            
            # Search at this complexity level
            level_best = await self._search_at_complexity_level(
                temp_config, X_train, y_train, X_val, y_val
            )
            
            if level_best and level_best.validation_accuracy > best_score:
                best_score = level_best.validation_accuracy
                best_architecture = level_best
                
                self.logger.info(
                    f"New best architecture at level {level_idx + 1}",
                    accuracy=best_score
                )
        
        return best_architecture
    
    async def _search_at_complexity_level(self,
                                        config: NASConfig,
                                        X_train: torch.Tensor,
                                        y_train: torch.Tensor,
                                        X_val: torch.Tensor,
                                        y_val: torch.Tensor) -> Architecture:
        """Search at specific complexity level."""
        
        generator = ArchitectureGenerator(config)
        evaluator = ArchitectureEvaluator(config)
        
        best_architecture = None
        best_score = 0.0
        
        # Generate and evaluate architectures at this complexity level
        n_trials = config.population_size
        
        for trial in range(n_trials):
            # Generate architecture
            architecture = await generator.generate_random_architecture()
            
            # Evaluate architecture
            evaluated_arch = await evaluator.evaluate(
                architecture, X_train, y_train, X_val, y_val
            )
            
            # Update best
            if evaluated_arch.validation_accuracy > best_score:
                best_score = evaluated_arch.validation_accuracy
                best_architecture = evaluated_arch
        
        return best_architecture