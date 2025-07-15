"""Integration tests for AutoML services."""

import asyncio
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

from pynomaly.domain.entities import Dataset
from pynomaly.application.services.automl_service import AutoMLService
from pynomaly.application.services.automl_pipeline_orchestrator import (
    AutoMLPipelineOrchestrator, 
    AutoMLPipelineConfiguration,
    PipelineStage,
    PipelineStatus
)


class TestAutoMLIntegration:
    """Test AutoML service integration."""
    
    @pytest.fixture
    def sample_dataset(self) -> Dataset:
        """Create a sample dataset for testing."""
        # Generate synthetic anomaly detection data
        np.random.seed(42)
        
        # Normal data
        normal_data = np.random.multivariate_normal(
            mean=[0, 0], 
            cov=[[1, 0.5], [0.5, 1]], 
            size=800
        )
        
        # Anomalous data
        anomaly_data = np.random.multivariate_normal(
            mean=[3, 3], 
            cov=[[0.5, 0], [0, 0.5]], 
            size=200
        )
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(800), np.ones(200)])  # 0 = normal, 1 = anomaly
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
        df['target'] = y
        
        return Dataset(
            name="test_anomaly_dataset",
            data=df,
            target_column="target",
            metadata={
                "contamination_rate": 0.2,
                "feature_count": 2,
                "sample_count": 1000
            }
        )
    
    @pytest.fixture
    def automl_service(self) -> AutoMLService:
        """Create AutoML service instance."""
        return AutoMLService()
    
    @pytest.fixture
    def pipeline_config(self) -> AutoMLPipelineConfiguration:
        """Create pipeline configuration."""
        return AutoMLPipelineConfiguration(
            contamination_rate=0.2,
            stages_to_run=[
                PipelineStage.DATA_PROFILING,
                PipelineStage.FEATURE_ENGINEERING,
                PipelineStage.ALGORITHM_RECOMMENDATION,
                PipelineStage.HYPERPARAMETER_OPTIMIZATION,
                PipelineStage.MODEL_EVALUATION
            ],
            max_execution_time=300,  # 5 minutes
            optimization_config={
                "max_algorithms": 2,
                "n_trials": 10,
                "timeout": 60
            },
            evaluation_config={
                "cv_folds": 3,
                "test_size": 0.2
            }
        )
    
    @pytest.mark.asyncio
    async def test_automl_data_profiling(self, automl_service, sample_dataset):
        """Test AutoML data profiling functionality."""
        profile = await automl_service.profile_dataset(sample_dataset)
        
        # Verify profile structure
        assert isinstance(profile, dict)
        assert "features_count" in profile
        assert "samples_count" in profile
        assert "data_types" in profile
        assert "missing_values_ratio" in profile
        
        # Verify profile values
        assert profile["features_count"] == 3  # 2 features + target
        assert profile["samples_count"] == 1000
        assert profile["missing_values_ratio"] == 0.0  # No missing values
    
    @pytest.mark.asyncio
    async def test_automl_algorithm_recommendation(self, automl_service, sample_dataset):
        """Test AutoML algorithm recommendation."""
        # First profile the dataset
        profile = await automl_service.profile_dataset(sample_dataset)
        
        # Get algorithm recommendations
        recommendations = await automl_service.recommend_algorithms(sample_dataset, profile)
        
        # Verify recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert "name" in rec
            assert "score" in rec
            assert "reason" in rec
            assert isinstance(rec["score"], (int, float))
            assert 0 <= rec["score"] <= 1
    
    @pytest.mark.asyncio
    async def test_automl_pipeline_orchestration(self, sample_dataset, pipeline_config):
        """Test complete AutoML pipeline orchestration."""
        # Mock the individual services since we're testing orchestration
        automl_service = Mock()
        automl_service.profile_dataset = AsyncMock(return_value={
            "features_count": 3,
            "samples_count": 1000,
            "missing_values_ratio": 0.0,
            "data_types": {"feature_1": "float64", "feature_2": "float64", "target": "int64"}
        })
        automl_service.recommend_algorithms = AsyncMock(return_value=[
            {"name": "IsolationForest", "score": 0.85, "reason": "Good for anomaly detection"},
            {"name": "LocalOutlierFactor", "score": 0.78, "reason": "Effective for local anomalies"}
        ])
        
        hyperparameter_service = Mock()
        hyperparameter_service.optimize_hyperparameters = AsyncMock(return_value={
            "best_params": {"contamination": 0.2, "n_estimators": 100},
            "best_value": 0.82,
            "study_info": {"n_trials": 10}
        })
        
        evaluation_service = Mock()
        evaluation_service.evaluate_model = AsyncMock(return_value={
            "metrics": {"f1_score": 0.81, "precision": 0.83, "recall": 0.79},
            "cv_scores": [0.80, 0.82, 0.81],
            "significance_test": {"p_value": 0.01, "significant": True}
        })
        
        feature_service = Mock()
        feature_service.engineer_features = AsyncMock(return_value=sample_dataset.data)
        
        # Create orchestrator
        orchestrator = AutoMLPipelineOrchestrator(
            automl_service=automl_service,
            hyperparameter_service=hyperparameter_service,
            evaluation_service=evaluation_service,
            feature_service=feature_service
        )
        
        # Create and execute pipeline
        pipeline = await orchestrator.create_pipeline(
            name="test_pipeline",
            dataset=sample_dataset,
            configuration=pipeline_config
        )
        
        # Execute pipeline
        completed_pipeline = await orchestrator.execute_pipeline(pipeline, sample_dataset)
        
        # Verify pipeline completion
        assert completed_pipeline.status == PipelineStatus.COMPLETED
        assert completed_pipeline.start_time is not None
        assert completed_pipeline.end_time is not None
        assert completed_pipeline.total_duration is not None
        assert len(completed_pipeline.stage_results) > 0
        
        # Verify stage execution
        executed_stages = [result.stage for result in completed_pipeline.stage_results]
        assert PipelineStage.DATA_PROFILING in executed_stages
        assert PipelineStage.ALGORITHM_RECOMMENDATION in executed_stages
        
        # Verify final results
        assert len(completed_pipeline.recommended_models) > 0
        assert completed_pipeline.best_model is not None
        assert "execution_time" in completed_pipeline.performance_metrics
    
    @pytest.mark.asyncio
    async def test_automl_pipeline_error_handling(self, sample_dataset, pipeline_config):
        """Test AutoML pipeline error handling and recovery."""
        # Mock services with failures
        automl_service = Mock()
        automl_service.profile_dataset = AsyncMock(side_effect=Exception("Data profiling failed"))
        
        hyperparameter_service = Mock()
        evaluation_service = Mock()
        feature_service = Mock()
        
        orchestrator = AutoMLPipelineOrchestrator(
            automl_service=automl_service,
            hyperparameter_service=hyperparameter_service,
            evaluation_service=evaluation_service,
            feature_service=feature_service
        )
        
        pipeline = await orchestrator.create_pipeline(
            name="test_error_pipeline",
            dataset=sample_dataset,
            configuration=pipeline_config
        )
        
        # Execute pipeline and expect failure
        with pytest.raises(Exception):
            await orchestrator.execute_pipeline(pipeline, sample_dataset)
        
        # Verify pipeline failed status
        assert pipeline.status == PipelineStatus.FAILED
        assert pipeline.error_message is not None
    
    @pytest.mark.asyncio
    async def test_automl_pipeline_cancellation(self, sample_dataset, pipeline_config):
        """Test AutoML pipeline cancellation."""
        # Mock services with delays
        automl_service = Mock()
        automl_service.profile_dataset = AsyncMock(return_value={"features_count": 3})
        
        async def slow_algorithm_recommendation(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate slow operation
            return [{"name": "IsolationForest", "score": 0.85}]
        
        automl_service.recommend_algorithms = slow_algorithm_recommendation
        
        hyperparameter_service = Mock()
        evaluation_service = Mock()
        feature_service = Mock()
        feature_service.engineer_features = AsyncMock(return_value=sample_dataset.data)
        
        orchestrator = AutoMLPipelineOrchestrator(
            automl_service=automl_service,
            hyperparameter_service=hyperparameter_service,
            evaluation_service=evaluation_service,
            feature_service=feature_service
        )
        
        pipeline = await orchestrator.create_pipeline(
            name="test_cancel_pipeline",
            dataset=sample_dataset,
            configuration=pipeline_config
        )
        
        # Start pipeline execution
        execution_task = asyncio.create_task(
            orchestrator.execute_pipeline(pipeline, sample_dataset)
        )
        
        # Wait a bit and then cancel
        await asyncio.sleep(0.1)
        cancelled = await orchestrator.cancel_pipeline(pipeline.id)
        
        assert cancelled is True
        assert pipeline.status == PipelineStatus.CANCELLED
        
        # Cancel the execution task
        execution_task.cancel()
        try:
            await execution_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_automl_pipeline_timeout(self, sample_dataset):
        """Test AutoML pipeline timeout handling."""
        # Create config with very short timeout
        short_timeout_config = AutoMLPipelineConfiguration(
            max_execution_time=1,  # 1 second timeout
            stages_to_run=[PipelineStage.DATA_PROFILING]
        )
        
        # Mock service with delay
        automl_service = Mock()
        
        async def slow_profiling(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return {"features_count": 3}
        
        automl_service.profile_dataset = slow_profiling
        
        hyperparameter_service = Mock()
        evaluation_service = Mock()
        feature_service = Mock()
        
        orchestrator = AutoMLPipelineOrchestrator(
            automl_service=automl_service,
            hyperparameter_service=hyperparameter_service,
            evaluation_service=evaluation_service,
            feature_service=feature_service
        )
        
        pipeline = await orchestrator.create_pipeline(
            name="test_timeout_pipeline",
            dataset=sample_dataset,
            configuration=short_timeout_config
        )
        
        # Execute pipeline - should timeout
        completed_pipeline = await orchestrator.execute_pipeline(pipeline, sample_dataset)
        
        # Note: Timeout handling depends on implementation details
        # This test verifies the timeout mechanism exists
        assert completed_pipeline.status in [PipelineStatus.FAILED, PipelineStatus.CANCELLED]


class TestAutoMLConfiguration:
    """Test AutoML configuration handling."""
    
    def test_automl_pipeline_configuration_defaults(self):
        """Test AutoML pipeline configuration default values."""
        config = AutoMLPipelineConfiguration()
        
        # Verify defaults
        assert config.target_column is None
        assert config.contamination_rate is None
        assert config.enable_parallel_execution is True
        assert config.retry_failed_stages is True
        assert config.max_retries == 3
        assert config.checkpoint_frequency == 300
        assert isinstance(config.stages_to_run, list)
        assert len(config.stages_to_run) > 0
    
    def test_automl_pipeline_configuration_validation(self):
        """Test AutoML pipeline configuration validation."""
        # Test valid configuration
        config = AutoMLPipelineConfiguration(
            contamination_rate=0.2,
            max_execution_time=600,
            max_retries=5
        )
        
        assert config.contamination_rate == 0.2
        assert config.max_execution_time == 600
        assert config.max_retries == 5
        
        # Test configuration serialization
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "contamination_rate" in config_dict
        assert "max_execution_time" in config_dict


class TestAutoMLServiceIntegration:
    """Test AutoML service integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_automl_with_missing_data(self):
        """Test AutoML handling of datasets with missing values."""
        # Create dataset with missing values
        np.random.seed(42)
        data = np.random.randn(100, 3)
        df = pd.DataFrame(data, columns=['feature_1', 'feature_2', 'feature_3'])
        
        # Introduce missing values
        df.loc[10:20, 'feature_1'] = np.nan
        df.loc[30:35, 'feature_2'] = np.nan
        
        dataset = Dataset(
            name="test_missing_data",
            data=df,
            metadata={"has_missing_values": True}
        )
        
        automl_service = AutoMLService()
        profile = await automl_service.profile_dataset(dataset)
        
        # Verify missing values are detected
        assert profile["missing_values_ratio"] > 0
        assert "missing_values" in profile
    
    @pytest.mark.asyncio 
    async def test_automl_with_categorical_data(self):
        """Test AutoML handling of categorical features."""
        # Create dataset with categorical features
        np.random.seed(42)
        data = {
            'numeric_feature': np.random.randn(100),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
            'binary_feature': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(data)
        
        dataset = Dataset(
            name="test_categorical_data",
            data=df,
            metadata={"has_categorical": True}
        )
        
        automl_service = AutoMLService()
        profile = await automl_service.profile_dataset(dataset)
        
        # Verify categorical features are detected
        assert "categorical_features" in profile
        assert len(profile["categorical_features"]) > 0
    
    @pytest.mark.asyncio
    async def test_automl_performance_benchmarking(self, sample_dataset):
        """Test AutoML performance benchmarking."""
        automl_service = AutoMLService()
        
        # Profile dataset and measure time
        import time
        start_time = time.time()
        profile = await automl_service.profile_dataset(sample_dataset)
        profiling_time = time.time() - start_time
        
        # Verify profiling completes in reasonable time
        assert profiling_time < 10.0  # Should complete within 10 seconds
        
        # Get algorithm recommendations and measure time
        start_time = time.time()
        recommendations = await automl_service.recommend_algorithms(sample_dataset, profile)
        recommendation_time = time.time() - start_time
        
        # Verify recommendations complete in reasonable time
        assert recommendation_time < 5.0  # Should complete within 5 seconds
        assert len(recommendations) > 0
    
    @pytest.fixture
    def sample_dataset(self) -> Dataset:
        """Create a sample dataset for testing."""
        np.random.seed(42)
        normal_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 800)
        anomaly_data = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 200)
        
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(800), np.ones(200)])
        
        df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
        df['target'] = y
        
        return Dataset(
            name="test_anomaly_dataset",
            data=df,
            target_column="target",
            metadata={"contamination_rate": 0.2}
        )