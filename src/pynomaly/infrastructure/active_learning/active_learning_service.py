"""Active learning service for intelligent sample selection and iterative model improvement."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.models.active_learning import (
    ActiveLearningConfig,
    ActiveLearningSession,
    LabeledSample,
    LabelType,
    LearningIteration,
    QueryResult,
    QueryStrategy,
    SamplePool,
    SampleStatus,
)
from pynomaly.domain.value_objects import PerformanceMetrics
from pynomaly.infrastructure.active_learning.query_strategies import QueryStrategyFactory

# Type alias for backward compatibility
ModelMetrics = PerformanceMetrics


class ActiveLearningService:
    """Main service for active learning operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Session management
        self.sessions: Dict[UUID, ActiveLearningSession] = {}
        self.sample_pools: Dict[UUID, SamplePool] = {}
        
        # Performance tracking
        self.service_stats: Dict[str, Any] = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_queries": 0,
            "total_samples_labeled": 0,
            "average_improvement_per_iteration": 0.0,
        }
        
        self.logger.info("Active learning service initialized")
    
    async def create_session(
        self,
        name: str,
        config: ActiveLearningConfig,
        initial_samples: Optional[Dict[str, np.ndarray]] = None,
        initial_labels: Optional[List[LabeledSample]] = None,
    ) -> ActiveLearningSession:
        """Create new active learning session."""
        
        session_id = uuid4()
        
        session = ActiveLearningSession(
            session_id=session_id,
            name=name,
            config=config,
            initial_labeled_samples=len(initial_labels) if initial_labels else 0,
        )
        
        # Add initial labeled samples
        if initial_labels:
            session.labeled_samples.extend(initial_labels)
        
        # Create sample pool if provided
        if initial_samples:
            pool_id = uuid4()
            sample_pool = SamplePool(
                pool_id=pool_id,
                name=f"{name}_pool",
                samples=initial_samples,
            )
            
            self.sample_pools[pool_id] = sample_pool
            session.total_samples = len(initial_samples)
            session.unlabeled_pool = set(initial_samples.keys())
            
            # Remove labeled samples from unlabeled pool
            if initial_labels:
                labeled_ids = {sample.sample_id for sample in initial_labels}
                session.unlabeled_pool -= labeled_ids
        
        # Store session
        self.sessions[session_id] = session
        
        # Update service stats
        self.service_stats["total_sessions"] += 1
        self.service_stats["active_sessions"] += 1
        
        self.logger.info(
            f"Created active learning session '{name}' with "
            f"{session.total_samples} total samples, "
            f"{len(session.labeled_samples)} initially labeled"
        )
        
        return session
    
    async def add_sample_pool(
        self,
        session_id: UUID,
        samples: Dict[str, np.ndarray],
        pool_name: Optional[str] = None,
    ) -> SamplePool:
        """Add sample pool to existing session."""
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        pool_id = uuid4()
        pool_name = pool_name or f"{session.name}_pool_{len(self.sample_pools)}"
        
        sample_pool = SamplePool(
            pool_id=pool_id,
            name=pool_name,
            samples=samples,
        )
        
        self.sample_pools[pool_id] = sample_pool
        
        # Update session
        session.total_samples += len(samples)
        session.unlabeled_pool.update(samples.keys())
        
        self.logger.info(f"Added {len(samples)} samples to session '{session.name}'")
        
        return sample_pool
    
    async def start_session(self, session_id: UUID) -> None:
        """Start active learning session."""
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.started_at = datetime.utcnow()
        session.is_active = True
        
        self.logger.info(f"Started active learning session '{session.name}'")
    
    async def run_query_iteration(
        self,
        session_id: UUID,
        model_predictions: Dict[str, Any],
        performance_evaluator: Optional[Any] = None,
        committee_predictions: Optional[Dict[str, List[Any]]] = None,
        **kwargs
    ) -> LearningIteration:
        """Run single active learning query iteration."""
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        if not session.is_active:
            raise ValueError(f"Session '{session.name}' is not active")
        
        # Check stopping criteria
        should_stop, stop_reason = session.should_stop()
        if should_stop:
            await self.end_session(session_id, stop_reason)
            raise ValueError(f"Session should stop: {stop_reason}")
        
        self.logger.info(f"Running query iteration {session.current_iteration} for session '{session.name}'")
        
        iteration_start = datetime.utcnow()
        
        # Get sample pool
        sample_pool = await self._get_session_sample_pool(session)
        
        # Create query strategy
        strategy = QueryStrategyFactory.create_strategy(session.config)
        
        # Prepare labeled samples for strategy
        labeled_samples_dict = {}
        for labeled_sample in session.labeled_samples:
            labeled_samples_dict[labeled_sample.sample_id] = {
                "data": labeled_sample.data,
                "label": labeled_sample.label,
                "confidence": labeled_sample.confidence,
            }
        
        # Record pre-iteration performance
        pre_performance = None
        if performance_evaluator:
            pre_performance = await self._evaluate_performance(
                performance_evaluator, session, "pre_iteration"
            )
        
        # Execute query strategy
        query_result = await strategy.select_samples(
            sample_pool=sample_pool,
            model_predictions=model_predictions,
            labeled_samples=labeled_samples_dict,
            committee_predictions=committee_predictions,
            **kwargs
        )
        
        # Create iteration record
        iteration = LearningIteration(
            iteration_id=uuid4(),
            iteration_number=session.current_iteration,
            query_result=query_result,
            new_labels=[],  # Will be filled when labels are provided
            pre_iteration_performance=pre_performance,
            start_time=iteration_start,
        )
        
        # Update session
        session.add_iteration(iteration)
        
        # Update service stats
        self.service_stats["total_queries"] += 1
        
        self.logger.info(
            f"Query iteration {iteration.iteration_number} completed. "
            f"Selected {len(query_result.selected_samples)} samples for labeling."
        )
        
        return iteration
    
    async def add_labels(
        self,
        session_id: UUID,
        iteration_id: UUID,
        new_labels: List[LabeledSample],
        performance_evaluator: Optional[Any] = None,
    ) -> LearningIteration:
        """Add labels from human annotators and complete iteration."""
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # Find iteration
        iteration = None
        for iter_record in session.iterations:
            if iter_record.iteration_id == iteration_id:
                iteration = iter_record
                break
        
        if not iteration:
            raise ValueError(f"Iteration {iteration_id} not found")
        
        self.logger.info(f"Adding {len(new_labels)} labels to iteration {iteration.iteration_number}")
        
        # Validate and add labels
        validated_labels = []
        total_cost = 0.0
        
        for label in new_labels:
            # Quality control
            if session.config.enable_quality_control:
                if (label.label_quality is not None and 
                    label.label_quality < session.config.min_label_quality):
                    self.logger.warning(f"Label for sample {label.sample_id} rejected due to low quality")
                    continue
            
            validated_labels.append(label)
            total_cost += label.labeling_cost
            
            # Remove from unlabeled pool
            session.unlabeled_pool.discard(label.sample_id)
        
        # Update iteration
        iteration.new_labels = validated_labels
        iteration.labeling_cost = total_cost
        iteration.end_time = datetime.utcnow()
        iteration.total_labeled_samples = len(session.labeled_samples) + len(validated_labels)
        
        # Add to session labeled samples
        session.labeled_samples.extend(validated_labels)
        
        # Evaluate post-iteration performance
        if performance_evaluator:
            post_performance = await self._evaluate_performance(
                performance_evaluator, session, "post_iteration"
            )
            iteration.post_iteration_performance = post_performance
            
            # Calculate improvement
            if iteration.pre_iteration_performance and post_performance:
                pre_f1 = iteration.pre_iteration_performance.f1_score
                post_f1 = post_performance.f1_score
                iteration.performance_improvement = post_f1 - pre_f1
        
        # Update service stats
        self.service_stats["total_samples_labeled"] += len(validated_labels)
        self._update_average_improvement(iteration.performance_improvement)
        
        self.logger.info(
            f"Added {len(validated_labels)} labels. "
            f"Performance improvement: {iteration.performance_improvement:.4f}"
        )
        
        return iteration
    
    async def run_full_session(
        self,
        session_id: UUID,
        model_trainer: Any,
        performance_evaluator: Any,
        label_oracle: Any,  # Function or service that provides labels
        **kwargs
    ) -> ActiveLearningSession:
        """Run complete active learning session until stopping criteria."""
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        self.logger.info(f"Starting full active learning session '{session.name}'")
        
        await self.start_session(session_id)
        
        try:
            while session.is_active:
                # Check stopping criteria
                should_stop, stop_reason = session.should_stop()
                if should_stop:
                    await self.end_session(session_id, stop_reason)
                    break
                
                # Train current model
                current_model = await model_trainer.train(session.labeled_samples)
                
                # Get predictions on unlabeled pool
                sample_pool = await self._get_session_sample_pool(session)
                model_predictions = await self._get_model_predictions(
                    current_model, sample_pool, list(session.unlabeled_pool)
                )
                
                # Run query iteration
                iteration = await self.run_query_iteration(
                    session_id, model_predictions, performance_evaluator, **kwargs
                )
                
                # Get labels from oracle
                new_labels = await self._get_labels_from_oracle(
                    label_oracle, iteration.query_result.selected_samples, sample_pool
                )
                
                # Add labels and complete iteration
                await self.add_labels(
                    session_id, iteration.iteration_id, new_labels, performance_evaluator
                )
                
                # Optional: adaptive budget adjustment
                if session.config.adaptive_budget:
                    await self._adjust_budget(session, iteration)
        
        except Exception as e:
            self.logger.error(f"Error in active learning session: {e}")
            await self.end_session(session_id, f"Error: {e}")
            raise
        
        self.logger.info(f"Active learning session '{session.name}' completed")
        
        return session
    
    async def end_session(self, session_id: UUID, reason: str = "Manual stop") -> None:
        """End active learning session."""
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.is_active = False
        session.ended_at = datetime.utcnow()
        
        # Update service stats
        self.service_stats["active_sessions"] -= 1
        
        self.logger.info(f"Ended session '{session.name}': {reason}")
    
    async def get_session_summary(self, session_id: UUID) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        summary = session.get_session_summary()
        
        # Add additional service-level information
        summary["service_context"] = {
            "total_sessions": self.service_stats["total_sessions"],
            "session_rank": self._get_session_rank(session),
        }
        
        return summary
    
    async def compare_strategies(
        self,
        strategies: List[QueryStrategy],
        initial_samples: Dict[str, np.ndarray],
        initial_labels: List[LabeledSample],
        model_trainer: Any,
        performance_evaluator: Any,
        label_oracle: Any,
        max_iterations: int = 10,
        **kwargs
    ) -> Dict[QueryStrategy, Dict[str, Any]]:
        """Compare multiple query strategies."""
        
        self.logger.info(f"Comparing {len(strategies)} query strategies")
        
        results = {}
        
        for strategy in strategies:
            self.logger.info(f"Testing strategy: {strategy.value}")
            
            # Create config for this strategy
            config = ActiveLearningConfig(
                query_strategy=strategy,
                max_iterations=max_iterations,
                **kwargs
            )
            
            # Create session
            session = await self.create_session(
                name=f"comparison_{strategy.value}",
                config=config,
                initial_samples=initial_samples.copy(),
                initial_labels=[sample for sample in initial_labels],  # Deep copy
            )
            
            try:
                # Run session
                completed_session = await self.run_full_session(
                    session.session_id,
                    model_trainer,
                    performance_evaluator,
                    label_oracle,
                )
                
                # Collect results
                results[strategy] = {
                    "final_performance": completed_session.current_performance.to_dict() if completed_session.current_performance else None,
                    "total_cost": completed_session.total_budget_used,
                    "iterations": completed_session.current_iteration,
                    "samples_labeled": len(completed_session.labeled_samples),
                    "efficiency": self._calculate_efficiency(completed_session),
                }
                
            except Exception as e:
                self.logger.error(f"Strategy {strategy.value} failed: {e}")
                results[strategy] = {"error": str(e)}
        
        return results
    
    async def _get_session_sample_pool(self, session: ActiveLearningSession) -> SamplePool:
        """Get sample pool for session."""
        # For simplicity, return the first pool associated with this session
        # In practice, this would be more sophisticated
        for pool in self.sample_pools.values():
            if session.name in pool.name:
                return pool
        
        raise ValueError(f"No sample pool found for session {session.name}")
    
    async def _evaluate_performance(
        self, 
        evaluator: Any, 
        session: ActiveLearningSession, 
        phase: str
    ) -> ModelMetrics:
        """Evaluate model performance."""
        # This is a placeholder - actual implementation would depend on the evaluator interface
        try:
            if hasattr(evaluator, 'evaluate'):
                metrics = await evaluator.evaluate(session.labeled_samples)
                return metrics
            else:
                # Fallback: create dummy metrics
                return ModelMetrics(
                    accuracy=0.8,
                    precision=0.8,
                    recall=0.8,
                    f1_score=0.8,
                    training_time=1.0,
                    inference_time=1.0,
                    model_size=1024,
                )
        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
            return ModelMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                training_time=0.0,
                inference_time=0.0,
                model_size=0,
            )
    
    async def _get_model_predictions(
        self, 
        model: Any, 
        sample_pool: SamplePool, 
        sample_ids: List[str]
    ) -> Dict[str, Any]:
        """Get model predictions for specified samples."""
        predictions = {}
        
        for sample_id in sample_ids:
            if sample_id in sample_pool.samples:
                sample_data = sample_pool.samples[sample_id]
                
                # Placeholder prediction - would use actual model
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(sample_data.reshape(1, -1))[0]
                        predictions[sample_id] = {
                            "prediction": pred,
                            "confidence": abs(pred - 0.5) * 2,  # Distance from decision boundary
                        }
                    else:
                        # Fallback: random prediction
                        pred = np.random.random()
                        predictions[sample_id] = {
                            "prediction": pred,
                            "confidence": np.random.random(),
                        }
                except Exception as e:
                    self.logger.error(f"Prediction failed for sample {sample_id}: {e}")
                    predictions[sample_id] = {
                        "prediction": 0.5,
                        "confidence": 0.0,
                    }
        
        return predictions
    
    async def _get_labels_from_oracle(
        self, 
        oracle: Any, 
        sample_ids: List[str], 
        sample_pool: SamplePool
    ) -> List[LabeledSample]:
        """Get labels from labeling oracle."""
        labels = []
        
        for sample_id in sample_ids:
            if sample_id in sample_pool.samples:
                sample_data = sample_pool.samples[sample_id]
                
                try:
                    if hasattr(oracle, 'get_label'):
                        label_result = await oracle.get_label(sample_id, sample_data)
                        label = LabeledSample(
                            sample_id=sample_id,
                            data=sample_data,
                            label=label_result.get("label", 0),
                            label_type=LabelType.BINARY,
                            confidence=label_result.get("confidence", 1.0),
                            labeling_cost=label_result.get("cost", 1.0),
                            labeling_time=datetime.utcnow(),
                        )
                    else:
                        # Fallback: random label
                        label = LabeledSample(
                            sample_id=sample_id,
                            data=sample_data,
                            label=np.random.choice([0, 1]),
                            label_type=LabelType.BINARY,
                            confidence=0.8,
                            labeling_cost=1.0,
                            labeling_time=datetime.utcnow(),
                        )
                    
                    labels.append(label)
                    
                except Exception as e:
                    self.logger.error(f"Labeling failed for sample {sample_id}: {e}")
        
        return labels
    
    async def _adjust_budget(self, session: ActiveLearningSession, iteration: LearningIteration) -> None:
        """Adjust budget allocation based on iteration performance."""
        if not session.config.adaptive_budget:
            return
        
        # Simple adaptive strategy: increase budget if improvement is high
        if iteration.performance_improvement > 0.05:  # Good improvement
            # Allow slightly larger batch in next iteration
            session.config.batch_size = min(
                session.config.max_samples_per_iteration,
                int(session.config.batch_size * 1.2)
            )
        elif iteration.performance_improvement < 0.01:  # Poor improvement
            # Reduce batch size
            session.config.batch_size = max(
                session.config.min_samples_per_iteration,
                int(session.config.batch_size * 0.8)
            )
    
    def _update_average_improvement(self, improvement: float) -> None:
        """Update average improvement per iteration."""
        current_avg = self.service_stats["average_improvement_per_iteration"]
        total_queries = self.service_stats["total_queries"]
        
        if total_queries > 0:
            new_avg = ((current_avg * (total_queries - 1)) + improvement) / total_queries
            self.service_stats["average_improvement_per_iteration"] = new_avg
    
    def _get_session_rank(self, session: ActiveLearningSession) -> int:
        """Get session rank based on performance."""
        # Simple ranking by final F1 score
        sessions_with_performance = [
            s for s in self.sessions.values() 
            if s.current_performance is not None
        ]
        
        if not sessions_with_performance or session.current_performance is None:
            return 0
        
        sessions_with_performance.sort(
            key=lambda s: s.current_performance.f1_score, reverse=True
        )
        
        for i, s in enumerate(sessions_with_performance):
            if s.session_id == session.session_id:
                return i + 1
        
        return 0
    
    def _calculate_efficiency(self, session: ActiveLearningSession) -> float:
        """Calculate overall efficiency of the session."""
        if session.total_budget_used == 0 or not session.current_performance:
            return 0.0
        
        # Efficiency = performance gain per unit cost
        if session.initial_performance:
            performance_gain = (
                session.current_performance.f1_score - 
                session.initial_performance.f1_score
            )
        else:
            performance_gain = session.current_performance.f1_score
        
        return performance_gain / session.total_budget_used
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        active_sessions = [s for s in self.sessions.values() if s.is_active]
        completed_sessions = [s for s in self.sessions.values() if not s.is_active]
        
        return {
            "service_stats": self.service_stats,
            "session_stats": {
                "total_sessions": len(self.sessions),
                "active_sessions": len(active_sessions),
                "completed_sessions": len(completed_sessions),
                "average_iterations_per_session": (
                    np.mean([s.current_iteration for s in completed_sessions])
                    if completed_sessions else 0
                ),
            },
            "pool_stats": {
                "total_pools": len(self.sample_pools),
                "total_samples": sum(pool.get_pool_size() for pool in self.sample_pools.values()),
            },
            "efficiency_metrics": {
                "average_improvement_per_iteration": self.service_stats["average_improvement_per_iteration"],
                "average_cost_per_sample": (
                    sum(s.total_budget_used for s in self.sessions.values()) /
                    max(1, self.service_stats["total_samples_labeled"])
                ),
            },
        }
    
    async def cleanup_sessions(self, keep_active: bool = True) -> int:
        """Clean up completed sessions to free memory."""
        cleanup_count = 0
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if not keep_active or not session.is_active:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            cleanup_count += 1
        
        self.logger.info(f"Cleaned up {cleanup_count} sessions")
        
        return cleanup_count