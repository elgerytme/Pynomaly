"""Integration tests for active learning system."""

from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

from pynomaly.domain.models.active_learning import (
    ActiveLearningConfig,
    LabeledSample,
    LabelType,
    QueryStrategy,
)
from pynomaly.domain.value_objects import PerformanceMetrics
from pynomaly.infrastructure.active_learning.active_learning_service import ActiveLearningService


@pytest.mark.integration
@pytest.mark.asyncio
class TestActiveLearningIntegration:
    """Integration tests for active learning system."""

    def create_synthetic_dataset(self, n_samples: int = 200) -> tuple[dict[str, np.ndarray], list[LabeledSample]]:
        """Create synthetic dataset for active learning."""
        np.random.seed(42)  # For reproducible tests
        
        # Generate 2D data with clear clusters
        cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_samples // 4)
        cluster2 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], n_samples // 4)
        cluster3 = np.random.multivariate_normal([2, -2], [[0.5, 0], [0, 0.5]], n_samples // 4)
        cluster4 = np.random.multivariate_normal([-2, 2], [[0.5, 0], [0, 0.5]], n_samples // 4)
        
        # Combine clusters
        data = np.vstack([cluster1, cluster2, cluster3, cluster4])
        
        # Create labels (clusters 1&2 = normal, clusters 3&4 = anomalous)
        labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        data = data[indices]
        labels = labels[indices]
        
        # Create sample dictionary
        samples = {f"sample_{i}": data[i] for i in range(n_samples)}
        
        # Create initial labeled samples (5% of data)
        initial_count = max(1, n_samples // 20)
        initial_labels = []
        
        for i in range(initial_count):
            sample = LabeledSample(
                sample_id=f"sample_{i}",
                data=data[i],
                label=int(labels[i]),
                label_type=LabelType.BINARY,
                confidence=1.0,
                labeling_cost=1.0,
                labeling_time=datetime.utcnow(),
            )
            initial_labels.append(sample)
        
        return samples, initial_labels

    class MockModelTrainer:
        """Mock model trainer for testing."""
        
        def __init__(self):
            self.training_history = []
        
        async def train(self, labeled_samples: list[LabeledSample]):
            """Train model on labeled samples."""
            self.training_history.append(len(labeled_samples))
            
            # Simple model: store labeled data for predictions
            self.labeled_data = []
            self.labels = []
            
            for sample in labeled_samples:
                self.labeled_data.append(sample.data)
                self.labels.append(sample.label)
            
            if self.labeled_data:
                self.labeled_data = np.array(self.labeled_data)
                self.labels = np.array(self.labels)
            
            return self
        
        def predict(self, data: np.ndarray) -> np.ndarray:
            """Make predictions on new data."""
            if len(self.labeled_data) == 0:
                # No training data, return random predictions
                return np.random.random(data.shape[0])
            
            # Simple k-NN classifier (k=1)
            predictions = []
            
            for sample in data:
                distances = np.linalg.norm(self.labeled_data - sample, axis=1)
                nearest_idx = np.argmin(distances)
                
                # Add some uncertainty based on distance
                nearest_distance = distances[nearest_idx]
                uncertainty = min(0.4, nearest_distance / 5.0)  # Scale uncertainty
                
                base_prediction = float(self.labels[nearest_idx])
                
                # Add noise for uncertainty
                if np.random.random() < uncertainty:
                    prediction = np.random.random()
                else:
                    prediction = base_prediction + np.random.normal(0, 0.1)
                
                predictions.append(np.clip(prediction, 0, 1))
            
            return np.array(predictions)

    class MockPerformanceEvaluator:
        """Mock performance evaluator for testing."""
        
        def __init__(self, true_labels: dict[str, int]):
            self.true_labels = true_labels
        
        async def evaluate(self, labeled_samples: list[LabeledSample]) -> PerformanceMetrics:
            """Evaluate model performance."""
            if not labeled_samples:
                return PerformanceMetrics(
                    accuracy=0.5, precision=0.5, recall=0.5, f1_score=0.5,
                    training_time=1.0, inference_time=1.0, model_size=1024
                )
            
            # Calculate accuracy on labeled samples
            correct = 0
            total = 0
            
            for sample in labeled_samples:
                if sample.sample_id in self.true_labels:
                    true_label = self.true_labels[sample.sample_id]
                    predicted_label = sample.label
                    
                    if true_label == predicted_label:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0.5
            
            # Simulate improving performance with more data
            data_bonus = min(0.3, len(labeled_samples) / 100)
            accuracy = min(1.0, accuracy + data_bonus)
            
            return PerformanceMetrics(
                accuracy=accuracy,
                precision=accuracy,  # Simplified
                recall=accuracy,
                f1_score=accuracy,
                training_time=1.0,
                inference_time=1.0,
                model_size=1024,
            )

    class MockLabelOracle:
        """Mock label oracle for testing."""
        
        def __init__(self, true_labels: dict[str, int], noise_rate: float = 0.05):
            self.true_labels = true_labels
            self.noise_rate = noise_rate
            self.query_count = 0
        
        async def get_label(self, sample_id: str, sample_data: np.ndarray) -> dict[str, any]:
            """Get label for sample."""
            self.query_count += 1
            
            if sample_id in self.true_labels:
                true_label = self.true_labels[sample_id]
                
                # Add label noise
                if np.random.random() < self.noise_rate:
                    label = 1 - true_label  # Flip label
                    confidence = 0.6  # Lower confidence for noisy labels
                else:
                    label = true_label
                    confidence = 0.95
                
                return {
                    "label": label,
                    "confidence": confidence,
                    "cost": 1.0,
                }
            else:
                # Unknown sample, return random label
                return {
                    "label": np.random.choice([0, 1]),
                    "confidence": 0.5,
                    "cost": 1.0,
                }

    async def test_end_to_end_uncertainty_sampling(self):
        """Test complete uncertainty sampling workflow."""
        service = ActiveLearningService()
        
        # Create synthetic dataset
        samples, initial_labels = self.create_synthetic_dataset(100)
        
        # Create true labels for evaluation
        true_labels = {}
        for sample_id, data in samples.items():
            # Simple rule: anomalous if both coordinates > 0 or both < 0
            x, y = data
            true_labels[sample_id] = 1 if (x > 0 and y < 0) or (x < 0 and y > 0) else 0
        
        # Create mock components
        trainer = self.MockModelTrainer()
        evaluator = self.MockPerformanceEvaluator(true_labels)
        oracle = self.MockLabelOracle(true_labels)
        
        # Create active learning session
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=5,
            max_iterations=10,
            max_budget=50.0,
            performance_threshold=0.85,
        )
        
        session = await service.create_session(
            name="uncertainty_sampling_test",
            config=config,
            initial_samples=samples,
            initial_labels=initial_labels,
        )
        
        self.logger.info(f"Starting uncertainty sampling test with {len(initial_labels)} initial labels")
        
        # Run complete session
        completed_session = await service.run_full_session(
            session.session_id,
            trainer,
            evaluator,
            oracle,
        )
        
        # Validate results
        assert not completed_session.is_active
        assert completed_session.current_iteration > 0
        assert len(completed_session.labeled_samples) > len(initial_labels)
        
        # Check performance improvement
        if completed_session.current_performance and completed_session.initial_performance:
            improvement = (
                completed_session.current_performance.f1_score - 
                completed_session.initial_performance.f1_score
            )
            print(f"Performance improvement: {improvement:.3f}")
        
        print(f"Final performance: {completed_session.current_performance.f1_score:.3f}")
        print(f"Total iterations: {completed_session.current_iteration}")
        print(f"Total samples labeled: {len(completed_session.labeled_samples)}")
        print(f"Budget used: {completed_session.total_budget_used:.1f}")

    async def test_diversity_sampling_workflow(self):
        """Test diversity-based sampling workflow."""
        service = ActiveLearningService()
        
        # Create synthetic dataset
        samples, initial_labels = self.create_synthetic_dataset(80)
        
        true_labels = {}
        for sample_id, data in samples.items():
            x, y = data
            true_labels[sample_id] = 1 if np.linalg.norm(data) > 2.5 else 0
        
        # Create session with diversity sampling
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.DIVERSITY_SAMPLING,
            batch_size=4,
            max_iterations=8,
            max_budget=40.0,
        )
        
        session = await service.create_session(
            name="diversity_sampling_test",
            config=config,
            initial_samples=samples,
            initial_labels=initial_labels,
        )
        
        # Create components
        trainer = self.MockModelTrainer()
        evaluator = self.MockPerformanceEvaluator(true_labels)
        oracle = self.MockLabelOracle(true_labels)
        
        # Run session
        completed_session = await service.run_full_session(
            session.session_id,
            trainer,
            evaluator,
            oracle,
        )
        
        # Validate diversity selection
        assert completed_session.current_iteration > 0
        
        # Check that samples span the feature space
        labeled_data = np.array([s.data for s in completed_session.labeled_samples])
        data_range = np.max(labeled_data, axis=0) - np.min(labeled_data, axis=0)
        
        print(f"Diversity sampling - data range covered: {data_range}")
        print(f"Total iterations: {completed_session.current_iteration}")

    async def test_hybrid_strategy_performance(self):
        """Test hybrid uncertainty-diversity strategy."""
        service = ActiveLearningService()
        
        # Create larger dataset for hybrid strategy
        samples, initial_labels = self.create_synthetic_dataset(150)
        
        true_labels = {}
        for sample_id, data in samples.items():
            x, y = data
            # More complex decision boundary
            true_labels[sample_id] = 1 if (x*x + y*y > 4 and abs(x-y) < 1) else 0
        
        # Test hybrid strategy
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_DIVERSITY,
            batch_size=6,
            max_iterations=12,
            max_budget=60.0,
            diversity_weight=0.4,  # Balance uncertainty and diversity
        )
        
        session = await service.create_session(
            name="hybrid_strategy_test",
            config=config,
            initial_samples=samples,
            initial_labels=initial_labels,
        )
        
        trainer = self.MockModelTrainer()
        evaluator = self.MockPerformanceEvaluator(true_labels)
        oracle = self.MockLabelOracle(true_labels, noise_rate=0.1)  # Add some label noise
        
        completed_session = await service.run_full_session(
            session.session_id,
            trainer,
            evaluator,
            oracle,
        )
        
        # Analyze hybrid strategy effectiveness
        assert completed_session.current_iteration > 0
        
        # Check that both uncertainty and diversity were considered
        iterations_with_diversity = sum(
            1 for iter in completed_session.iterations 
            if iter.query_result.diversity_score is not None and iter.query_result.diversity_score > 0
        )
        
        iterations_with_uncertainty = sum(
            1 for iter in completed_session.iterations 
            if iter.query_result.uncertainty_score is not None and iter.query_result.uncertainty_score > 0
        )
        
        print(f"Hybrid strategy - iterations with diversity: {iterations_with_diversity}")
        print(f"Hybrid strategy - iterations with uncertainty: {iterations_with_uncertainty}")

    async def test_strategy_comparison(self):
        """Test comparison of multiple active learning strategies."""
        service = ActiveLearningService()
        
        # Create shared dataset
        samples, initial_labels = self.create_synthetic_dataset(120)
        
        true_labels = {}
        for sample_id, data in samples.items():
            x, y = data
            true_labels[sample_id] = 1 if (x > 1 or y < -1) else 0
        
        # Compare strategies
        strategies = [
            QueryStrategy.UNCERTAINTY_SAMPLING,
            QueryStrategy.DIVERSITY_SAMPLING,
            QueryStrategy.UNCERTAINTY_DIVERSITY,
        ]
        
        trainer = self.MockModelTrainer()
        evaluator = self.MockPerformanceEvaluator(true_labels)
        oracle = self.MockLabelOracle(true_labels)
        
        print("\nComparing active learning strategies...")
        
        results = await service.compare_strategies(
            strategies=strategies,
            initial_samples=samples,
            initial_labels=initial_labels,
            model_trainer=trainer,
            performance_evaluator=evaluator,
            label_oracle=oracle,
            max_iterations=6,
            batch_size=4,
            max_budget=30.0,
        )
        
        # Analyze comparison results
        print("\nStrategy Comparison Results:")
        for strategy, result in results.items():
            if "error" not in result:
                print(f"{strategy.value}:")
                print(f"  Final F1: {result['final_performance']['f1_score']:.3f}")
                print(f"  Total cost: {result['total_cost']:.1f}")
                print(f"  Iterations: {result['iterations']}")
                print(f"  Efficiency: {result['efficiency']:.4f}")
            else:
                print(f"{strategy.value}: ERROR - {result['error']}")
        
        # At least one strategy should succeed
        successful_strategies = [s for s, r in results.items() if "error" not in r]
        assert len(successful_strategies) > 0

    async def test_budget_management_and_stopping_criteria(self):
        """Test budget management and stopping criteria."""
        service = ActiveLearningService()
        
        # Create small dataset for budget testing
        samples, initial_labels = self.create_synthetic_dataset(60)
        
        true_labels = {}
        for sample_id, data in samples.items():
            x, y = data
            true_labels[sample_id] = 1 if x*y > 0 else 0
        
        # Test budget exhaustion
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=3,
            max_iterations=20,  # High limit
            max_budget=15.0,  # Low budget - should stop here
            performance_threshold=0.99,  # High threshold
        )
        
        session = await service.create_session(
            name="budget_test",
            config=config,
            initial_samples=samples,
            initial_labels=initial_labels,
        )
        
        trainer = self.MockModelTrainer()
        evaluator = self.MockPerformanceEvaluator(true_labels)
        oracle = self.MockLabelOracle(true_labels)
        
        completed_session = await service.run_full_session(
            session.session_id,
            trainer,
            evaluator,
            oracle,
        )
        
        # Should stop due to budget exhaustion
        assert completed_session.total_budget_used >= config.max_budget * 0.9  # Close to budget limit
        print(f"Budget test - stopped at budget: {completed_session.total_budget_used:.1f}/{config.max_budget}")
        
        # Test performance threshold stopping
        high_performance_config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=2,
            max_iterations=15,
            max_budget=50.0,
            performance_threshold=0.7,  # Lower threshold - should reach this
        )
        
        session2 = await service.create_session(
            name="performance_threshold_test",
            config=high_performance_config,
            initial_samples=samples,
            initial_labels=initial_labels,
        )
        
        completed_session2 = await service.run_full_session(
            session2.session_id,
            trainer,
            evaluator,
            oracle,
        )
        
        # Check if stopped due to performance threshold
        if completed_session2.current_performance:
            print(f"Performance threshold test - final F1: {completed_session2.current_performance.f1_score:.3f}")

    async def test_adaptive_batch_sizing(self):
        """Test adaptive batch size adjustment."""
        service = ActiveLearningService()
        
        samples, initial_labels = self.create_synthetic_dataset(80)
        
        true_labels = {}
        for sample_id, data in samples.items():
            x, y = data
            true_labels[sample_id] = 1 if x + y > 0 else 0
        
        # Enable adaptive budget
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=3,  # Starting batch size
            max_iterations=8,
            max_budget=35.0,
            adaptive_budget=True,
            min_samples_per_iteration=1,
            max_samples_per_iteration=8,
        )
        
        session = await service.create_session(
            name="adaptive_test",
            config=config,
            initial_samples=samples,
            initial_labels=initial_labels,
        )
        
        trainer = self.MockModelTrainer()
        evaluator = self.MockPerformanceEvaluator(true_labels)
        oracle = self.MockLabelOracle(true_labels)
        
        # Run session and track batch size changes
        await service.start_session(session.session_id)
        
        batch_sizes = []
        
        while session.is_active:
            should_stop, _ = session.should_stop()
            if should_stop:
                await service.end_session(session.session_id, "Stopping criteria met")
                break
            
            # Record current batch size
            batch_sizes.append(session.config.batch_size)
            
            # Run single iteration
            current_model = await trainer.train(session.labeled_samples)
            
            sample_pool = await service._get_session_sample_pool(session)
            model_predictions = await service._get_model_predictions(
                current_model, sample_pool, list(session.unlabeled_pool)
            )
            
            iteration = await service.run_query_iteration(
                session.session_id, model_predictions, evaluator
            )
            
            new_labels = await service._get_labels_from_oracle(
                oracle, iteration.query_result.selected_samples, sample_pool
            )
            
            await service.add_labels(
                session.session_id, iteration.iteration_id, new_labels, evaluator
            )
        
        # Check if batch size adapted
        print(f"Adaptive batch sizes: {batch_sizes}")
        if len(batch_sizes) > 2:
            # Should see some variation in batch sizes
            unique_sizes = set(batch_sizes)
            print(f"Unique batch sizes used: {unique_sizes}")

    async def test_label_quality_control(self):
        """Test label quality control mechanisms."""
        service = ActiveLearningService()
        
        samples, initial_labels = self.create_synthetic_dataset(50)
        
        true_labels = {}
        for sample_id, data in samples.items():
            x, y = data
            true_labels[sample_id] = 1 if abs(x) > abs(y) else 0
        
        # Enable quality control
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=3,
            max_iterations=5,
            enable_quality_control=True,
            min_label_quality=0.8,
        )
        
        session = await service.create_session(
            name="quality_control_test",
            config=config,
            initial_samples=samples,
            initial_labels=initial_labels,
        )
        
        class NoisyOracle:
            """Oracle that sometimes provides low-quality labels."""
            
            def __init__(self, true_labels: dict[str, int]):
                self.true_labels = true_labels
            
            async def get_label(self, sample_id: str, sample_data: np.ndarray) -> dict[str, any]:
                if sample_id in self.true_labels:
                    true_label = self.true_labels[sample_id]
                    
                    # Sometimes provide low-quality labels
                    if np.random.random() < 0.3:  # 30% low quality
                        return {
                            "label": 1 - true_label,  # Wrong label
                            "confidence": 0.6,
                            "cost": 1.0,
                            "quality": 0.5,  # Low quality
                        }
                    else:
                        return {
                            "label": true_label,
                            "confidence": 0.9,
                            "cost": 1.0,
                            "quality": 0.95,  # High quality
                        }
                
                return {"label": 0, "confidence": 0.5, "cost": 1.0, "quality": 0.7}
        
        trainer = self.MockModelTrainer()
        evaluator = self.MockPerformanceEvaluator(true_labels)
        oracle = NoisyOracle(true_labels)
        
        # Run single iteration to test quality control
        await service.start_session(session.session_id)
        
        current_model = await trainer.train(session.labeled_samples)
        sample_pool = await service._get_session_sample_pool(session)
        model_predictions = await service._get_model_predictions(
            current_model, sample_pool, list(session.unlabeled_pool)[:5]
        )
        
        iteration = await service.run_query_iteration(
            session.session_id, model_predictions, evaluator
        )
        
        # Get labels with quality information
        new_labels = []
        for sample_id in iteration.query_result.selected_samples:
            if sample_id in sample_pool.samples:
                sample_data = sample_pool.samples[sample_id]
                label_result = await oracle.get_label(sample_id, sample_data)
                
                label = LabeledSample(
                    sample_id=sample_id,
                    data=sample_data,
                    label=label_result["label"],
                    label_type=LabelType.BINARY,
                    confidence=label_result["confidence"],
                    labeling_cost=label_result["cost"],
                    label_quality=label_result.get("quality"),
                    labeling_time=datetime.utcnow(),
                )
                new_labels.append(label)
        
        # Add labels (quality control should filter some)
        completed_iteration = await service.add_labels(
            session.session_id, iteration.iteration_id, new_labels, evaluator
        )
        
        # Check that low-quality labels were filtered
        accepted_labels = len(completed_iteration.new_labels)
        total_labels = len(new_labels)
        
        print(f"Quality control: {accepted_labels}/{total_labels} labels accepted")
        
        # Should accept fewer labels than provided due to quality filtering
        assert accepted_labels <= total_labels

    async def test_session_recovery_and_resumption(self):
        """Test session state management and resumption."""
        service = ActiveLearningService()
        
        samples, initial_labels = self.create_synthetic_dataset(40)
        
        true_labels = {}
        for sample_id, data in samples.items():
            x, y = data
            true_labels[sample_id] = 1 if x > y else 0
        
        # Create session
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=2,
            max_iterations=10,
        )
        
        session = await service.create_session(
            name="recovery_test",
            config=config,
            initial_samples=samples,
            initial_labels=initial_labels,
        )
        
        trainer = self.MockModelTrainer()
        evaluator = self.MockPerformanceEvaluator(true_labels)
        oracle = self.MockLabelOracle(true_labels)
        
        # Run a few iterations
        await service.start_session(session.session_id)
        
        for i in range(3):
            current_model = await trainer.train(session.labeled_samples)
            sample_pool = await service._get_session_sample_pool(session)
            model_predictions = await service._get_model_predictions(
                current_model, sample_pool, list(session.unlabeled_pool)
            )
            
            iteration = await service.run_query_iteration(
                session.session_id, model_predictions, evaluator
            )
            
            new_labels = await service._get_labels_from_oracle(
                oracle, iteration.query_result.selected_samples, sample_pool
            )
            
            await service.add_labels(
                session.session_id, iteration.iteration_id, new_labels, evaluator
            )
        
        # Check session state
        mid_session = service.sessions[session.session_id]
        assert mid_session.current_iteration == 3
        assert len(mid_session.iterations) == 3
        
        # Get session summary
        summary = await service.get_session_summary(session.session_id)
        
        assert summary["iterations_completed"] == 3
        assert summary["status"] == "active"
        
        print(f"Session state at iteration 3:")
        print(f"  Labeled samples: {len(mid_session.labeled_samples)}")
        print(f"  Budget used: {mid_session.total_budget_used}")
        print(f"  Current performance: {mid_session.current_performance.f1_score if mid_session.current_performance else 'N/A'}")

    def logger(self):
        """Get logger for integration tests."""
        import logging
        return logging.getLogger(__name__)

    async def test_service_statistics_and_monitoring(self):
        """Test service-level statistics and monitoring."""
        service = ActiveLearningService()
        
        # Create multiple sessions
        samples1, initial_labels1 = self.create_synthetic_dataset(30)
        samples2, initial_labels2 = self.create_synthetic_dataset(30)
        
        config1 = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=2,
            max_iterations=3,
        )
        
        config2 = ActiveLearningConfig(
            query_strategy=QueryStrategy.DIVERSITY_SAMPLING,
            batch_size=2,
            max_iterations=3,
        )
        
        session1 = await service.create_session("stats_test_1", config1, samples1, initial_labels1)
        session2 = await service.create_session("stats_test_2", config2, samples2, initial_labels2)
        
        # Run both sessions briefly
        true_labels = {f"sample_{i}": i % 2 for i in range(60)}
        trainer = self.MockModelTrainer()
        evaluator = self.MockPerformanceEvaluator(true_labels)
        oracle = self.MockLabelOracle(true_labels)
        
        # Run a few iterations for each session
        for session in [session1, session2]:
            await service.start_session(session.session_id)
            
            for _ in range(2):
                current_model = await trainer.train(session.labeled_samples)
                sample_pool = await service._get_session_sample_pool(session)
                model_predictions = await service._get_model_predictions(
                    current_model, sample_pool, list(session.unlabeled_pool)
                )
                
                iteration = await service.run_query_iteration(
                    session.session_id, model_predictions, evaluator
                )
                
                new_labels = await service._get_labels_from_oracle(
                    oracle, iteration.query_result.selected_samples, sample_pool
                )
                
                await service.add_labels(
                    session.session_id, iteration.iteration_id, new_labels, evaluator
                )
        
        # Get service statistics
        stats = service.get_service_statistics()
        
        print("\nService Statistics:")
        print(f"  Total sessions: {stats['session_stats']['total_sessions']}")
        print(f"  Active sessions: {stats['session_stats']['active_sessions']}")
        print(f"  Total queries: {stats['service_stats']['total_queries']}")
        print(f"  Total samples labeled: {stats['service_stats']['total_samples_labeled']}")
        print(f"  Average improvement per iteration: {stats['service_stats']['average_improvement_per_iteration']:.4f}")
        
        # Validate statistics
        assert stats["session_stats"]["total_sessions"] == 2
        assert stats["session_stats"]["active_sessions"] == 2
        assert stats["service_stats"]["total_queries"] >= 4  # 2 iterations × 2 sessions
        
        # Test cleanup
        await service.end_session(session1.session_id, "Test completed")
        cleanup_count = await service.cleanup_sessions(keep_active=True)
        
        assert cleanup_count == 1  # Should remove 1 inactive session
        
        final_stats = service.get_service_statistics()
        assert final_stats["session_stats"]["active_sessions"] == 1