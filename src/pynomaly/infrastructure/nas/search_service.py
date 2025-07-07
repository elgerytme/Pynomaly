"""Neural Architecture Search service for automated architecture optimization."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.models.dataset import Dataset
from pynomaly.domain.models.nas import (
    ArchitectureEvaluation,
    NeuralArchitecture,
    SearchConfiguration,
    SearchExperiment,
    SearchStrategy,
)
from pynomaly.infrastructure.nas.evaluator import ArchitectureEvaluator
from pynomaly.infrastructure.nas.search_strategies import (
    NASSearchStrategy,
    NASStrategyFactory,
)


class NeuralArchitectureSearchService:
    """Main service for Neural Architecture Search operations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Active experiments
        self.active_experiments: Dict[UUID, SearchExperiment] = {}
        self.experiment_strategies: Dict[UUID, NASSearchStrategy] = {}
        self.experiment_evaluators: Dict[UUID, ArchitectureEvaluator] = {}
        
        # Results storage
        self.experiment_results: Dict[UUID, List[ArchitectureEvaluation]] = {}
        self.architecture_registry: Dict[UUID, NeuralArchitecture] = {}
        
        # Performance tracking
        self.total_architectures_evaluated = 0
        self.total_search_time = 0.0

    async def create_search_experiment(
        self,
        name: str,
        config: SearchConfiguration,
        target_dataset: Dataset,
    ) -> SearchExperiment:
        """Create new neural architecture search experiment."""
        
        experiment = SearchExperiment(
            experiment_id=uuid4(),
            name=name,
            configuration=config,
            target_dataset_id=uuid4(),  # In practice, this would be the dataset ID
        )
        
        # Create strategy and evaluator
        strategy = NASStrategyFactory.create_strategy(config)
        evaluator = ArchitectureEvaluator(config)
        
        # Store experiment components
        self.active_experiments[experiment.experiment_id] = experiment
        self.experiment_strategies[experiment.experiment_id] = strategy
        self.experiment_evaluators[experiment.experiment_id] = evaluator
        self.experiment_results[experiment.experiment_id] = []
        
        self.logger.info(
            f"Created NAS experiment '{name}' with strategy {config.search_strategy.value}"
        )
        
        return experiment

    async def run_search_experiment(
        self,
        experiment_id: UUID,
        dataset: Dataset,
        max_concurrent_evaluations: int = 3,
        checkpoint_interval: int = 10,
    ) -> SearchExperiment:
        """Run neural architecture search experiment."""
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        strategy = self.experiment_strategies[experiment_id]
        evaluator = self.experiment_evaluators[experiment_id]
        
        self.logger.info(f"Starting NAS experiment '{experiment.name}'")
        start_time = datetime.utcnow()
        
        try:
            # Create semaphore for concurrent evaluations
            evaluation_semaphore = asyncio.Semaphore(max_concurrent_evaluations)
            
            # Search loop
            while not experiment.is_completed:
                # Generate new architecture
                architecture = await strategy.generate_architecture(experiment)
                
                # Check constraints
                if not experiment.configuration.validate_constraints(architecture):
                    self.logger.debug(
                        f"Architecture {architecture.architecture_hash} violates constraints, skipping"
                    )
                    continue
                
                # Store architecture
                self.architecture_registry[architecture.architecture_id] = architecture
                
                # Evaluate architecture
                evaluation = await self._evaluate_architecture_with_semaphore(
                    evaluation_semaphore, evaluator, architecture, dataset
                )
                
                # Update experiment state
                await self._update_experiment_state(
                    experiment, strategy, architecture, evaluation
                )
                
                # Checkpoint progress
                if experiment.total_architectures_evaluated % checkpoint_interval == 0:
                    await self._checkpoint_experiment(experiment)
                
                # Check early termination conditions
                if await self._should_terminate_early(experiment):
                    self.logger.info(f"Early termination triggered for experiment {experiment.name}")
                    break
            
            # Complete experiment
            experiment.completed_at = datetime.utcnow()
            experiment_duration = (experiment.completed_at - start_time).total_seconds()
            self.total_search_time += experiment_duration
            
            # Generate final results
            await self._finalize_experiment_results(experiment)
            
            self.logger.info(
                f"Completed NAS experiment '{experiment.name}' in {experiment_duration:.2f}s: "
                f"evaluated {experiment.total_architectures_evaluated} architectures, "
                f"best score: {experiment.best_score:.4f}"
            )
            
            return experiment
            
        except Exception as e:
            self.logger.error(f"Error running NAS experiment '{experiment.name}': {e}")
            experiment.completed_at = datetime.utcnow()
            raise

    async def _evaluate_architecture_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        evaluator: ArchitectureEvaluator,
        architecture: NeuralArchitecture,
        dataset: Dataset,
    ) -> ArchitectureEvaluation:
        """Evaluate architecture with concurrency control."""
        
        async with semaphore:
            return await evaluator.evaluate_architecture(architecture, dataset)

    async def _update_experiment_state(
        self,
        experiment: SearchExperiment,
        strategy: NASSearchStrategy,
        architecture: NeuralArchitecture,
        evaluation: ArchitectureEvaluation,
    ) -> None:
        """Update experiment state with new evaluation."""
        
        # Update strategy
        await strategy.update_search_state(architecture, evaluation, experiment)
        
        # Update experiment
        experiment.total_architectures_evaluated += 1
        experiment.evaluated_architectures.append(architecture.architecture_id)
        
        # Update best architecture
        if evaluation.composite_score > experiment.best_score:
            experiment.best_score = evaluation.composite_score
            experiment.best_architecture_id = architecture.architecture_id
        
        # Store evaluation
        self.experiment_results[experiment.experiment_id].append(evaluation)
        self.total_architectures_evaluated += 1
        
        # Update convergence history
        experiment.convergence_history.append(evaluation.composite_score)
        
        # Update diversity (architecture uniqueness)
        unique_hashes = set(
            self.architecture_registry[arch_id].architecture_hash
            for arch_id in experiment.evaluated_architectures
        )
        diversity = len(unique_hashes) / len(experiment.evaluated_architectures)
        experiment.diversity_history.append(diversity)

    async def _checkpoint_experiment(self, experiment: SearchExperiment) -> None:
        """Save experiment checkpoint."""
        
        self.logger.debug(
            f"Checkpoint: {experiment.name} - "
            f"Evaluated: {experiment.total_architectures_evaluated}, "
            f"Best: {experiment.best_score:.4f}, "
            f"Diversity: {experiment.diversity_history[-1]:.3f}"
        )
        
        # Update estimated completion time
        if experiment.total_architectures_evaluated > 10:
            elapsed_time = (datetime.utcnow() - experiment.started_at).total_seconds()
            avg_time_per_arch = elapsed_time / experiment.total_architectures_evaluated
            remaining_archs = (
                experiment.configuration.max_architectures - 
                experiment.total_architectures_evaluated
            )
            estimated_remaining = remaining_archs * avg_time_per_arch
            experiment.estimated_completion = datetime.utcnow() + timedelta(
                seconds=estimated_remaining
            )

    async def _should_terminate_early(self, experiment: SearchExperiment) -> bool:
        """Check if experiment should terminate early."""
        
        # Convergence check
        if len(experiment.convergence_history) >= 20:
            recent_scores = experiment.convergence_history[-20:]
            score_improvement = max(recent_scores) - min(recent_scores)
            
            if score_improvement < 0.01:  # Less than 1% improvement
                self.logger.info("Early termination: convergence detected")
                return True
        
        # Diversity check (premature convergence)
        if len(experiment.diversity_history) >= 10:
            recent_diversity = experiment.diversity_history[-5:]
            avg_diversity = np.mean(recent_diversity)
            
            if avg_diversity < 0.1:  # Very low diversity
                self.logger.info("Early termination: low diversity detected")
                return True
        
        # Time limit check
        elapsed_time = (datetime.utcnow() - experiment.started_at).total_seconds()
        max_time = 24 * 3600  # 24 hours max
        
        if elapsed_time > max_time:
            self.logger.info("Early termination: time limit reached")
            return True
        
        return False

    async def _finalize_experiment_results(self, experiment: SearchExperiment) -> None:
        """Finalize experiment results and compute final metrics."""
        
        evaluations = self.experiment_results[experiment.experiment_id]
        evaluator = self.experiment_evaluators[experiment.experiment_id]
        
        if not evaluations:
            return
        
        # Update Pareto front
        experiment.update_pareto_front(evaluations)
        
        # Calculate final statistics
        scores = [e.composite_score for e in evaluations]
        
        final_stats = {
            "total_evaluations": len(evaluations),
            "best_score": max(scores) if scores else 0.0,
            "average_score": np.mean(scores) if scores else 0.0,
            "score_std": np.std(scores) if scores else 0.0,
            "search_efficiency": experiment.search_efficiency,
            "pareto_front_size": len(experiment.pareto_front),
            "convergence_achieved": self._check_convergence(experiment),
        }
        
        # Store in experiment metadata
        experiment.metadata["final_statistics"] = final_stats
        
        self.logger.info(f"Experiment results: {final_stats}")

    def _check_convergence(self, experiment: SearchExperiment) -> bool:
        """Check if experiment converged."""
        
        if len(experiment.convergence_history) < 10:
            return False
        
        # Check if best score stabilized
        recent_best = []
        current_best = 0.0
        
        for score in experiment.convergence_history:
            if score > current_best:
                current_best = score
            recent_best.append(current_best)
        
        # Check if best score changed in last 20% of evaluations
        split_point = int(len(recent_best) * 0.8)
        early_best = recent_best[split_point]
        final_best = recent_best[-1]
        
        improvement = (final_best - early_best) / max(early_best, 1e-8)
        
        return improvement < 0.02  # Less than 2% improvement

    async def get_experiment_status(self, experiment_id: UUID) -> Dict[str, Any]:
        """Get current status of search experiment."""
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        evaluations = self.experiment_results[experiment_id]
        
        # Calculate runtime statistics
        elapsed_time = (
            (experiment.completed_at or datetime.utcnow()) - experiment.started_at
        ).total_seconds()
        
        status = {
            "experiment_id": str(experiment.experiment_id),
            "name": experiment.name,
            "strategy": experiment.configuration.search_strategy.value,
            "is_completed": experiment.is_completed,
            "elapsed_time_seconds": elapsed_time,
            "total_evaluated": experiment.total_architectures_evaluated,
            "best_score": experiment.best_score,
            "best_architecture_id": str(experiment.best_architecture_id) if experiment.best_architecture_id else None,
            "search_efficiency": experiment.search_efficiency,
            "pareto_front_size": len(experiment.pareto_front),
            "estimated_completion": experiment.estimated_completion.isoformat() if experiment.estimated_completion else None,
        }
        
        # Add recent performance trends
        if experiment.convergence_history:
            status["convergence_trend"] = experiment.convergence_history[-10:]
        if experiment.diversity_history:
            status["diversity_trend"] = experiment.diversity_history[-10:]
        
        # Add evaluation statistics
        if evaluations:
            scores = [e.composite_score for e in evaluations]
            latencies = [e.inference_time_ms for e in evaluations]
            param_counts = [e.total_params for e in evaluations]
            
            status["performance_statistics"] = {
                "score_mean": np.mean(scores),
                "score_std": np.std(scores),
                "latency_mean": np.mean(latencies),
                "latency_std": np.std(latencies),
                "params_mean": np.mean(param_counts),
                "params_std": np.std(param_counts),
            }
        
        return status

    async def get_best_architectures(
        self, 
        experiment_id: UUID, 
        top_k: int = 5
    ) -> List[Tuple[NeuralArchitecture, ArchitectureEvaluation]]:
        """Get top-k best architectures from experiment."""
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        evaluations = self.experiment_results[experiment_id]
        
        # Sort by composite score
        sorted_evaluations = sorted(
            evaluations, 
            key=lambda e: e.composite_score, 
            reverse=True
        )
        
        # Get top-k with architectures
        results = []
        for evaluation in sorted_evaluations[:top_k]:
            architecture = self.architecture_registry[evaluation.architecture_id]
            results.append((architecture, evaluation))
        
        return results

    async def get_pareto_optimal_architectures(
        self, experiment_id: UUID
    ) -> List[Tuple[NeuralArchitecture, ArchitectureEvaluation]]:
        """Get Pareto-optimal architectures from experiment."""
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        evaluator = self.experiment_evaluators[experiment_id]
        evaluations = self.experiment_results[experiment_id]
        
        # Calculate Pareto front
        pareto_evaluations = evaluator.get_pareto_front(evaluations)
        
        # Get architectures
        results = []
        for evaluation in pareto_evaluations:
            architecture = self.architecture_registry[evaluation.architecture_id]
            results.append((architecture, evaluation))
        
        return results

    async def compare_search_strategies(
        self,
        strategies: List[SearchStrategy],
        dataset: Dataset,
        max_architectures: int = 50,
        max_time_minutes: int = 60,
    ) -> Dict[SearchStrategy, Dict[str, Any]]:
        """Compare different search strategies on same dataset."""
        
        results = {}
        
        for strategy in strategies:
            self.logger.info(f"Benchmarking strategy: {strategy.value}")
            
            # Create configuration
            config = SearchConfiguration(
                search_strategy=strategy,
                max_architectures=max_architectures,
                max_epochs_per_architecture=5,  # Fast evaluation
            )
            
            # Create and run experiment
            experiment = await self.create_search_experiment(
                f"benchmark_{strategy.value}",
                config,
                dataset,
            )
            
            try:
                # Run with time limit
                completed_experiment = await asyncio.wait_for(
                    self.run_search_experiment(experiment.experiment_id, dataset),
                    timeout=max_time_minutes * 60,
                )
                
                # Collect results
                evaluations = self.experiment_results[experiment.experiment_id]
                scores = [e.composite_score for e in evaluations] if evaluations else []
                
                results[strategy] = {
                    "total_evaluated": completed_experiment.total_architectures_evaluated,
                    "best_score": completed_experiment.best_score,
                    "average_score": np.mean(scores) if scores else 0.0,
                    "search_efficiency": completed_experiment.search_efficiency,
                    "convergence_achieved": self._check_convergence(completed_experiment),
                    "elapsed_time": (
                        completed_experiment.completed_at - completed_experiment.started_at
                    ).total_seconds() if completed_experiment.completed_at else max_time_minutes * 60,
                }
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Strategy {strategy.value} timed out")
                results[strategy] = {
                    "total_evaluated": experiment.total_architectures_evaluated,
                    "best_score": experiment.best_score,
                    "timed_out": True,
                }
        
        return results

    async def export_architecture(
        self, 
        architecture_id: UUID, 
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export architecture definition."""
        
        if architecture_id not in self.architecture_registry:
            raise ValueError(f"Architecture {architecture_id} not found")
        
        architecture = self.architecture_registry[architecture_id]
        
        if format == "json":
            return {
                "architecture_id": str(architecture.architecture_id),
                "name": architecture.name,
                "input_shape": architecture.input_shape,
                "output_shape": architecture.output_shape,
                "total_params": architecture.total_params,
                "total_flops": architecture.total_flops,
                "architecture_hash": architecture.architecture_hash,
                "cells": [
                    {
                        "cell_id": str(cell.cell_id),
                        "name": cell.name,
                        "operations": [
                            {
                                "operation_type": op.operation_type.value,
                                "parameters": op.parameters,
                                "input_shape": op.input_shape,
                                "output_shape": op.output_shape,
                            }
                            for op in cell.operations
                        ],
                        "connections": cell.connections,
                        "input_nodes": cell.input_nodes,
                        "output_nodes": cell.output_nodes,
                    }
                    for cell in architecture.cells
                ],
                "global_connections": architecture.global_connections,
                "metadata": architecture.metadata,
                "created_at": architecture.created_at.isoformat(),
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        
        active_count = len(self.active_experiments)
        completed_count = sum(
            1 for exp in self.active_experiments.values() if exp.is_completed
        )
        
        return {
            "total_experiments": len(self.active_experiments),
            "active_experiments": active_count - completed_count,
            "completed_experiments": completed_count,
            "total_architectures_evaluated": self.total_architectures_evaluated,
            "total_search_time_hours": self.total_search_time / 3600,
            "architectures_in_registry": len(self.architecture_registry),
            "available_strategies": [s.value for s in NASStrategyFactory.get_available_strategies()],
        }

    async def cleanup_completed_experiments(self, keep_results: bool = True) -> int:
        """Clean up completed experiments to free memory."""
        
        cleanup_count = 0
        
        completed_experiment_ids = [
            exp_id for exp_id, exp in self.active_experiments.items()
            if exp.is_completed
        ]
        
        for exp_id in completed_experiment_ids:
            # Remove from active tracking
            del self.active_experiments[exp_id]
            del self.experiment_strategies[exp_id]
            del self.experiment_evaluators[exp_id]
            
            if not keep_results:
                del self.experiment_results[exp_id]
            
            cleanup_count += 1
        
        self.logger.info(f"Cleaned up {cleanup_count} completed experiments")
        
        return cleanup_count