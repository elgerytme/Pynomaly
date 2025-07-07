"""Unit tests for active learning components."""

from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

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
from pynomaly.infrastructure.active_learning.active_learning_service import ActiveLearningService
from pynomaly.infrastructure.active_learning.query_strategies import (
    QueryStrategyFactory,
    UncertaintySamplingStrategy,
    DiversitySamplingStrategy,
    HybridQueryStrategy,
)


class TestLabeledSample:
    """Test labeled sample representation."""

    def test_labeled_sample_creation(self):
        """Test creating labeled sample."""
        sample = LabeledSample(
            sample_id="test_sample",
            data=np.array([1, 2, 3]),
            label=1,
            label_type=LabelType.BINARY,
            confidence=0.9,
            labeling_cost=2.0,
        )
        
        assert sample.sample_id == "test_sample"
        assert np.array_equal(sample.data, np.array([1, 2, 3]))
        assert sample.label == 1
        assert sample.label_type == LabelType.BINARY
        assert sample.confidence == 0.9
        assert sample.labeling_cost == 2.0

    def test_labeled_sample_validation(self):
        """Test labeled sample validation."""
        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            LabeledSample(
                sample_id="test",
                data=np.array([1, 2]),
                label=0,
                label_type=LabelType.BINARY,
                confidence=1.5,
            )
        
        # Invalid labeling cost
        with pytest.raises(ValueError, match="Labeling cost must be non-negative"):
            LabeledSample(
                sample_id="test",
                data=np.array([1, 2]),
                label=0,
                label_type=LabelType.BINARY,
                labeling_cost=-1.0,
            )


class TestQueryResult:
    """Test query result representation."""

    def test_query_result_creation(self):
        """Test creating query result."""
        result = QueryResult(
            query_id=uuid4(),
            strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            selected_samples=["sample1", "sample2"],
            selection_scores={"sample1": 0.8, "sample2": 0.6},
            query_time=datetime.utcnow(),
            expected_cost=5.0,
            labeled_ratio=0.2,
        )
        
        assert result.strategy == QueryStrategy.UNCERTAINTY_SAMPLING
        assert len(result.selected_samples) == 2
        assert result.expected_cost == 5.0
        assert result.labeled_ratio == 0.2

    def test_query_result_validation(self):
        """Test query result validation."""
        # Invalid expected cost
        with pytest.raises(ValueError, match="Expected cost must be non-negative"):
            QueryResult(
                query_id=uuid4(),
                strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
                selected_samples=["sample1"],
                selection_scores={"sample1": 0.8},
                query_time=datetime.utcnow(),
                expected_cost=-1.0,
            )
        
        # Invalid labeled ratio
        with pytest.raises(ValueError, match="Labeled ratio must be between 0 and 1"):
            QueryResult(
                query_id=uuid4(),
                strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
                selected_samples=["sample1"],
                selection_scores={"sample1": 0.8},
                query_time=datetime.utcnow(),
                labeled_ratio=1.5,
            )


class TestActiveLearningConfig:
    """Test active learning configuration."""

    def test_config_creation(self):
        """Test creating active learning config."""
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=20,
            max_iterations=50,
            max_budget=500.0,
            performance_threshold=0.9,
        )
        
        assert config.query_strategy == QueryStrategy.UNCERTAINTY_SAMPLING
        assert config.batch_size == 20
        assert config.max_iterations == 50
        assert config.max_budget == 500.0
        assert config.performance_threshold == 0.9

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            ActiveLearningConfig(
                query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
                batch_size=0,
            )
        
        # Invalid max iterations
        with pytest.raises(ValueError, match="Max iterations must be positive"):
            ActiveLearningConfig(
                query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
                max_iterations=0,
            )
        
        # Invalid performance threshold
        with pytest.raises(ValueError, match="Performance threshold must be between 0 and 1"):
            ActiveLearningConfig(
                query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
                performance_threshold=1.5,
            )


class TestSamplePool:
    """Test sample pool management."""

    def create_test_pool(self) -> SamplePool:
        """Create test sample pool."""
        samples = {
            "sample1": np.array([1.0, 2.0, 3.0]),
            "sample2": np.array([2.0, 3.0, 4.0]),
            "sample3": np.array([3.0, 4.0, 5.0]),
        }
        
        return SamplePool(
            pool_id=uuid4(),
            name="test_pool",
            samples=samples,
        )

    def test_sample_pool_creation(self):
        """Test creating sample pool."""
        pool = self.create_test_pool()
        
        assert pool.name == "test_pool"
        assert pool.get_pool_size() == 3
        assert "sample1" in pool.samples

    def test_sample_pool_validation(self):
        """Test sample pool validation."""
        # Empty pool
        with pytest.raises(ValueError, match="Sample pool cannot be empty"):
            SamplePool(
                pool_id=uuid4(),
                name="empty_pool",
                samples={},
            )
        
        # Inconsistent shapes
        with pytest.raises(ValueError, match="All samples must have the same shape"):
            SamplePool(
                pool_id=uuid4(),
                name="inconsistent_pool",
                samples={
                    "sample1": np.array([1, 2, 3]),
                    "sample2": np.array([1, 2]),  # Different shape
                },
            )

    def test_sample_operations(self):
        """Test sample pool operations."""
        pool = self.create_test_pool()
        
        # Add sample
        pool.add_sample("sample4", np.array([4.0, 5.0, 6.0]))
        assert pool.get_pool_size() == 4
        
        # Get sample data
        data = pool.get_sample_data("sample1")
        assert np.array_equal(data, np.array([1.0, 2.0, 3.0]))
        
        # Remove sample
        assert pool.remove_sample("sample1")
        assert pool.get_pool_size() == 3
        assert not pool.remove_sample("nonexistent")
        
        # Get random samples
        random_samples = pool.get_random_samples(2)
        assert len(random_samples) == 2
        assert all(sid in pool.samples for sid in random_samples)

    def test_data_matrix_operations(self):
        """Test data matrix operations."""
        pool = self.create_test_pool()
        
        # Get full data matrix
        data_matrix = pool.get_data_matrix()
        assert data_matrix.shape == (3, 3)
        
        # Get subset data matrix
        subset_matrix = pool.get_data_matrix(["sample1", "sample3"])
        assert subset_matrix.shape == (2, 3)

    def test_diversity_metrics(self):
        """Test diversity metrics computation."""
        pool = self.create_test_pool()
        
        metrics = pool.compute_diversity_metrics()
        
        assert "mean_distance" in metrics
        assert "diversity_index" in metrics
        assert isinstance(metrics["mean_distance"], float)


class TestLearningIteration:
    """Test learning iteration tracking."""

    def create_test_iteration(self) -> LearningIteration:
        """Create test learning iteration."""
        query_result = QueryResult(
            query_id=uuid4(),
            strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            selected_samples=["sample1", "sample2"],
            selection_scores={"sample1": 0.8, "sample2": 0.6},
            query_time=datetime.utcnow(),
        )
        
        labeled_sample = LabeledSample(
            sample_id="sample1",
            data=np.array([1, 2, 3]),
            label=1,
            label_type=LabelType.BINARY,
            labeling_cost=2.0,
        )
        
        return LearningIteration(
            iteration_id=uuid4(),
            iteration_number=1,
            query_result=query_result,
            new_labels=[labeled_sample],
            labeling_cost=2.0,
        )

    def test_learning_iteration_creation(self):
        """Test creating learning iteration."""
        iteration = self.create_test_iteration()
        
        assert iteration.iteration_number == 1
        assert len(iteration.new_labels) == 1
        assert iteration.labeling_cost == 2.0

    def test_iteration_validation(self):
        """Test iteration validation."""
        query_result = QueryResult(
            query_id=uuid4(),
            strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            selected_samples=["sample1"],
            selection_scores={"sample1": 0.8},
            query_time=datetime.utcnow(),
        )
        
        # Invalid iteration number
        with pytest.raises(ValueError, match="Iteration number must be non-negative"):
            LearningIteration(
                iteration_id=uuid4(),
                iteration_number=-1,
                query_result=query_result,
                new_labels=[],
            )

    def test_iteration_methods(self):
        """Test iteration methods."""
        iteration = self.create_test_iteration()
        
        # Set end time
        iteration.end_time = datetime.utcnow()
        
        # Get duration
        duration = iteration.get_duration()
        assert isinstance(duration, float)
        assert duration >= 0
        
        # Get cost efficiency
        iteration.performance_improvement = 0.1
        efficiency = iteration.get_cost_efficiency()
        assert efficiency == 0.1 / 2.0  # improvement / cost


class TestActiveLearningSession:
    """Test active learning session management."""

    def create_test_session(self) -> ActiveLearningSession:
        """Create test active learning session."""
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=10,
            max_budget=100.0,
        )
        
        return ActiveLearningSession(
            session_id=uuid4(),
            name="test_session",
            config=config,
            total_samples=100,
            initial_labeled_samples=10,
        )

    def test_session_creation(self):
        """Test creating active learning session."""
        session = self.create_test_session()
        
        assert session.name == "test_session"
        assert session.total_samples == 100
        assert session.initial_labeled_samples == 10
        assert session.current_iteration == 0
        assert session.is_active

    def test_session_validation(self):
        """Test session validation."""
        config = ActiveLearningConfig(query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING)
        
        # Invalid current iteration
        with pytest.raises(ValueError, match="Current iteration must be non-negative"):
            ActiveLearningSession(
                session_id=uuid4(),
                name="test",
                config=config,
                current_iteration=-1,
            )

    def test_iteration_management(self):
        """Test iteration management."""
        session = self.create_test_session()
        
        # Create iteration
        query_result = QueryResult(
            query_id=uuid4(),
            strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            selected_samples=["sample1"],
            selection_scores={"sample1": 0.8},
            query_time=datetime.utcnow(),
        )
        
        iteration = LearningIteration(
            iteration_id=uuid4(),
            iteration_number=0,
            query_result=query_result,
            new_labels=[],
            labeling_cost=5.0,
        )
        
        # Add iteration
        session.add_iteration(iteration)
        
        assert session.current_iteration == 1
        assert session.total_budget_used == 5.0
        assert len(session.iterations) == 1

    def test_session_metrics(self):
        """Test session metrics."""
        session = self.create_test_session()
        
        # Add some labeled samples
        for i in range(5):
            sample = LabeledSample(
                sample_id=f"sample_{i}",
                data=np.array([i, i+1, i+2]),
                label=i % 2,
                label_type=LabelType.BINARY,
            )
            session.labeled_samples.append(sample)
        
        # Test metrics
        assert session.get_labeled_count() == 5
        assert session.get_labeling_ratio() == 5 / 100
        
        # Add budget usage
        session.total_budget_used = 50.0
        assert session.get_budget_utilization() == 0.5

    def test_stopping_criteria(self):
        """Test stopping criteria evaluation."""
        session = self.create_test_session()
        
        # Should not stop initially
        should_stop, reason = session.should_stop()
        assert not should_stop
        
        # Exhaust budget
        session.total_budget_used = 100.0
        should_stop, reason = session.should_stop()
        assert should_stop
        assert "Budget exhausted" in reason
        
        # Reset and test max iterations
        session.total_budget_used = 0.0
        session.current_iteration = 100
        should_stop, reason = session.should_stop()
        assert should_stop
        assert "Max iterations reached" in reason

    def test_session_summary(self):
        """Test session summary generation."""
        session = self.create_test_session()
        
        summary = session.get_session_summary()
        
        assert "session_id" in summary
        assert "name" in summary
        assert "strategy" in summary
        assert "status" in summary
        assert "efficiency_metrics" in summary


@pytest.mark.asyncio
class TestQueryStrategies:
    """Test query strategy implementations."""

    def create_test_pool(self) -> SamplePool:
        """Create test sample pool."""
        samples = {}
        for i in range(20):
            samples[f"sample_{i}"] = np.random.randn(5)
        
        return SamplePool(
            pool_id=uuid4(),
            name="test_pool",
            samples=samples,
        )

    def create_test_predictions(self, sample_pool: SamplePool) -> Dict[str, Any]:
        """Create test model predictions."""
        predictions = {}
        for sample_id in sample_pool.samples.keys():
            confidence = np.random.random()
            predictions[sample_id] = {
                "prediction": 1 if confidence > 0.5 else 0,
                "confidence": confidence,
                "probabilities": [1-confidence, confidence],
            }
        return predictions

    async def test_uncertainty_sampling_strategy(self):
        """Test uncertainty sampling strategy."""
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=5,
        )
        
        strategy = UncertaintySamplingStrategy(config)
        sample_pool = self.create_test_pool()
        predictions = self.create_test_predictions(sample_pool)
        
        result = await strategy.select_samples(sample_pool, predictions)
        
        assert isinstance(result, QueryResult)
        assert result.strategy == QueryStrategy.UNCERTAINTY_SAMPLING
        assert len(result.selected_samples) <= 5
        assert result.uncertainty_score is not None

    async def test_diversity_sampling_strategy(self):
        """Test diversity sampling strategy."""
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.DIVERSITY_SAMPLING,
            batch_size=5,
        )
        
        strategy = DiversitySamplingStrategy(config)
        sample_pool = self.create_test_pool()
        predictions = self.create_test_predictions(sample_pool)
        
        result = await strategy.select_samples(sample_pool, predictions)
        
        assert isinstance(result, QueryResult)
        assert len(result.selected_samples) <= 5
        assert result.diversity_score is not None

    async def test_cluster_based_strategy(self):
        """Test cluster-based strategy."""
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.CLUSTER_BASED,
            batch_size=5,
        )
        
        strategy = DiversitySamplingStrategy(config)
        sample_pool = self.create_test_pool()
        predictions = self.create_test_predictions(sample_pool)
        
        result = await strategy.select_samples(sample_pool, predictions)
        
        assert isinstance(result, QueryResult)
        assert len(result.selected_samples) <= 5

    async def test_hybrid_strategy(self):
        """Test hybrid uncertainty-diversity strategy."""
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_DIVERSITY,
            batch_size=5,
            diversity_weight=0.5,
        )
        
        strategy = HybridQueryStrategy(config)
        sample_pool = self.create_test_pool()
        predictions = self.create_test_predictions(sample_pool)
        
        result = await strategy.select_samples(sample_pool, predictions)
        
        assert isinstance(result, QueryResult)
        assert result.strategy == QueryStrategy.UNCERTAINTY_DIVERSITY
        assert len(result.selected_samples) <= 5
        assert result.uncertainty_score is not None
        assert result.diversity_score is not None

    def test_query_strategy_factory(self):
        """Test query strategy factory."""
        uncertainty_config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING
        )
        uncertainty_strategy = QueryStrategyFactory.create_strategy(uncertainty_config)
        assert isinstance(uncertainty_strategy, UncertaintySamplingStrategy)
        
        diversity_config = ActiveLearningConfig(
            query_strategy=QueryStrategy.DIVERSITY_SAMPLING
        )
        diversity_strategy = QueryStrategyFactory.create_strategy(diversity_config)
        assert isinstance(diversity_strategy, DiversitySamplingStrategy)
        
        hybrid_config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_DIVERSITY
        )
        hybrid_strategy = QueryStrategyFactory.create_strategy(hybrid_config)
        assert isinstance(hybrid_strategy, HybridQueryStrategy)


@pytest.mark.asyncio
class TestActiveLearningService:
    """Test active learning service."""

    async def test_service_creation(self):
        """Test creating active learning service."""
        service = ActiveLearningService()
        
        assert len(service.sessions) == 0
        assert len(service.sample_pools) == 0
        assert service.service_stats["total_sessions"] == 0

    async def test_session_creation(self):
        """Test creating session through service."""
        service = ActiveLearningService()
        
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=5,
        )
        
        initial_samples = {
            f"sample_{i}": np.random.randn(3) 
            for i in range(10)
        }
        
        initial_labels = [
            LabeledSample(
                sample_id="sample_0",
                data=initial_samples["sample_0"],
                label=1,
                label_type=LabelType.BINARY,
            )
        ]
        
        session = await service.create_session(
            name="test_session",
            config=config,
            initial_samples=initial_samples,
            initial_labels=initial_labels,
        )
        
        assert session.name == "test_session"
        assert session.session_id in service.sessions
        assert session.total_samples == 10
        assert len(session.labeled_samples) == 1
        assert len(session.unlabeled_pool) == 9

    async def test_sample_pool_management(self):
        """Test sample pool management."""
        service = ActiveLearningService()
        
        config = ActiveLearningConfig(query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING)
        session = await service.create_session("test_session", config)
        
        # Add sample pool
        samples = {f"sample_{i}": np.random.randn(3) for i in range(5)}
        pool = await service.add_sample_pool(session.session_id, samples)
        
        assert pool.pool_id in service.sample_pools
        assert len(service.sample_pools) == 1

    async def test_session_lifecycle(self):
        """Test session start and end."""
        service = ActiveLearningService()
        
        config = ActiveLearningConfig(query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING)
        session = await service.create_session("test_session", config)
        
        # Start session
        await service.start_session(session.session_id)
        updated_session = service.sessions[session.session_id]
        assert updated_session.is_active
        assert updated_session.started_at is not None
        
        # End session
        await service.end_session(session.session_id, "Test completed")
        final_session = service.sessions[session.session_id]
        assert not final_session.is_active
        assert final_session.ended_at is not None

    async def test_query_iteration(self):
        """Test running query iteration."""
        service = ActiveLearningService()
        
        # Create session with sample pool
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=3,
        )
        
        initial_samples = {f"sample_{i}": np.random.randn(3) for i in range(10)}
        session = await service.create_session(
            "test_session", config, initial_samples=initial_samples
        )
        
        await service.start_session(session.session_id)
        
        # Create mock predictions
        predictions = {}
        for sample_id in initial_samples.keys():
            predictions[sample_id] = {
                "prediction": np.random.random(),
                "confidence": np.random.random(),
            }
        
        # Run query iteration
        iteration = await service.run_query_iteration(
            session.session_id, predictions
        )
        
        assert isinstance(iteration, LearningIteration)
        assert len(iteration.query_result.selected_samples) <= 3
        assert iteration.iteration_number == 0

    async def test_label_addition(self):
        """Test adding labels to iteration."""
        service = ActiveLearningService()
        
        # Create session and run query
        config = ActiveLearningConfig(
            query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
            batch_size=2,
        )
        
        initial_samples = {f"sample_{i}": np.random.randn(3) for i in range(5)}
        session = await service.create_session(
            "test_session", config, initial_samples=initial_samples
        )
        
        await service.start_session(session.session_id)
        
        predictions = {
            sample_id: {"prediction": 0.5, "confidence": 0.5}
            for sample_id in initial_samples.keys()
        }
        
        iteration = await service.run_query_iteration(session.session_id, predictions)
        
        # Create labels for selected samples
        new_labels = []
        for sample_id in iteration.query_result.selected_samples:
            label = LabeledSample(
                sample_id=sample_id,
                data=initial_samples[sample_id],
                label=np.random.choice([0, 1]),
                label_type=LabelType.BINARY,
                labeling_cost=1.0,
            )
            new_labels.append(label)
        
        # Add labels
        completed_iteration = await service.add_labels(
            session.session_id, iteration.iteration_id, new_labels
        )
        
        assert len(completed_iteration.new_labels) == len(new_labels)
        assert completed_iteration.labeling_cost == len(new_labels)
        
        # Check session state
        updated_session = service.sessions[session.session_id]
        assert len(updated_session.labeled_samples) == len(new_labels)

    async def test_session_summary(self):
        """Test session summary generation."""
        service = ActiveLearningService()
        
        config = ActiveLearningConfig(query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING)
        session = await service.create_session("test_session", config)
        
        summary = await service.get_session_summary(session.session_id)
        
        assert "session_id" in summary
        assert "service_context" in summary
        assert summary["name"] == "test_session"

    async def test_service_statistics(self):
        """Test service statistics."""
        service = ActiveLearningService()
        
        # Create a few sessions
        config = ActiveLearningConfig(query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING)
        for i in range(3):
            await service.create_session(f"session_{i}", config)
        
        stats = service.get_service_statistics()
        
        assert "service_stats" in stats
        assert "session_stats" in stats
        assert "pool_stats" in stats
        assert stats["session_stats"]["total_sessions"] == 3

    async def test_session_cleanup(self):
        """Test session cleanup."""
        service = ActiveLearningService()
        
        # Create sessions
        config = ActiveLearningConfig(query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING)
        active_session = await service.create_session("active_session", config)
        inactive_session = await service.create_session("inactive_session", config)
        
        # End one session
        await service.end_session(inactive_session.session_id)
        
        assert len(service.sessions) == 2
        
        # Cleanup inactive sessions
        cleanup_count = await service.cleanup_sessions(keep_active=True)
        
        assert cleanup_count == 1
        assert len(service.sessions) == 1
        assert active_session.session_id in service.sessions

    async def test_invalid_operations(self):
        """Test invalid operations."""
        service = ActiveLearningService()
        
        invalid_id = uuid4()
        
        # Query with non-existent session
        with pytest.raises(ValueError, match="Session .* not found"):
            await service.run_query_iteration(invalid_id, {})
        
        # End non-existent session
        with pytest.raises(ValueError, match="Session .* not found"):
            await service.end_session(invalid_id)
        
        # Add pool to non-existent session
        with pytest.raises(ValueError, match="Session .* not found"):
            await service.add_sample_pool(invalid_id, {})

    async def test_strategy_comparison(self):
        """Test strategy comparison functionality."""
        service = ActiveLearningService()
        
        # Create test data
        initial_samples = {f"sample_{i}": np.random.randn(3) for i in range(20)}
        initial_labels = [
            LabeledSample(
                sample_id="sample_0",
                data=initial_samples["sample_0"],
                label=1,
                label_type=LabelType.BINARY,
            )
        ]
        
        # Mock trainer and evaluator
        class MockTrainer:
            async def train(self, labeled_samples):
                return self  # Return self as trained model
            
            def predict(self, data):
                return np.random.random(data.shape[0])
        
        class MockEvaluator:
            async def evaluate(self, labeled_samples):
                from pynomaly.domain.value_objects import PerformanceMetrics
                return PerformanceMetrics(
                    accuracy=0.8,
                    precision=0.8,
                    recall=0.8,
                    f1_score=0.8,
                    training_time=1.0,
                    inference_time=1.0,
                    model_size=1024,
                )
        
        class MockOracle:
            async def get_label(self, sample_id, sample_data):
                return {
                    "label": np.random.choice([0, 1]),
                    "confidence": 0.9,
                    "cost": 1.0,
                }
        
        # Compare strategies (limit to 2 iterations for speed)
        strategies = [QueryStrategy.UNCERTAINTY_SAMPLING, QueryStrategy.DIVERSITY_SAMPLING]
        
        results = await service.compare_strategies(
            strategies=strategies,
            initial_samples=initial_samples,
            initial_labels=initial_labels,
            model_trainer=MockTrainer(),
            performance_evaluator=MockEvaluator(),
            label_oracle=MockOracle(),
            max_iterations=2,
            batch_size=2,
        )
        
        assert len(results) <= len(strategies)
        
        for strategy, result in results.items():
            if "error" not in result:
                assert "final_performance" in result
                assert "total_cost" in result
                assert "iterations" in result