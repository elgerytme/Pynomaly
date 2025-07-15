"""Comprehensive tests for ML Pipeline service implementation."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
from uuid import uuid4

from ..infrastructure.services.ml_pipeline_service_impl import MLPipelineServiceImpl
from ..domain.entities.machine_learning_pipeline import MachineLearningPipeline, PipelineStatus, StepType
from ..domain.value_objects.ml_model_metrics import ModelMetrics, TaskType, MetricType


@pytest.fixture
def ml_pipeline_service():
    """Create ML pipeline service instance."""
    return MLPipelineServiceImpl()


@pytest.fixture
def sample_training_data():
    """Create sample training dataset."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate binary target with some relationship to features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data["target"] = y
    
    return data


@pytest.fixture
def sample_validation_data():
    """Create sample validation dataset."""
    np.random.seed(123)
    n_samples = 200
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data["target"] = y
    
    return data


@pytest.fixture
def sample_pipeline_config():
    """Create sample pipeline configuration."""
    return {
        "name": "test_ml_pipeline",
        "pipeline_type": "training",
        "description": "Test ML pipeline for unit testing",
        "steps_config": [
            {
                "name": "data_validation",
                "type": "data_validation",
                "configuration": {"validate_schema": True}
            },
            {
                "name": "preprocessing",
                "type": "data_preprocessing",
                "configuration": {"normalize": True}
            },
            {
                "name": "feature_engineering",
                "type": "feature_engineering",
                "configuration": {"create_interactions": True}
            },
            {
                "name": "model_training",
                "type": "model_training",
                "configuration": {
                    "algorithm": "random_forest",
                    "hyperparameters": {"n_estimators": 100, "max_depth": 10}
                }
            },
            {
                "name": "model_evaluation",
                "type": "model_evaluation",
                "configuration": {"metrics": ["accuracy", "f1_score"]}
            }
        ]
    }


class TestMLPipelineServiceCreation:
    """Test ML pipeline creation and validation."""
    
    @pytest.mark.asyncio
    async def test_create_pipeline_success(self, ml_pipeline_service, sample_pipeline_config):
        """Test successful pipeline creation."""
        pipeline = await ml_pipeline_service.create_pipeline(
            name=sample_pipeline_config["name"],
            pipeline_type=sample_pipeline_config["pipeline_type"],
            steps_config=sample_pipeline_config["steps_config"],
            created_by="test_user",
            description=sample_pipeline_config["description"]
        )
        
        assert isinstance(pipeline, MachineLearningPipeline)
        assert pipeline.name == sample_pipeline_config["name"]
        assert pipeline.pipeline_type == sample_pipeline_config["pipeline_type"]
        assert pipeline.status == PipelineStatus.VALID
        assert len(pipeline.steps) == len(sample_pipeline_config["steps_config"])
        assert pipeline.created_by == "test_user"
    
    @pytest.mark.asyncio
    async def test_create_pipeline_with_invalid_steps(self, ml_pipeline_service):
        """Test pipeline creation with invalid step configuration."""
        invalid_steps = [
            {
                "name": "invalid_step",
                "type": "nonexistent_type",
                "configuration": {}
            }
        ]
        
        with pytest.raises(ValueError):
            await ml_pipeline_service.create_pipeline(
                name="invalid_pipeline",
                pipeline_type="training",
                steps_config=invalid_steps,
                created_by="test_user"
            )
    
    @pytest.mark.asyncio
    async def test_validate_pipeline_configuration(self, ml_pipeline_service, sample_pipeline_config):
        """Test pipeline configuration validation."""
        pipeline = await ml_pipeline_service.create_pipeline(
            name=sample_pipeline_config["name"],
            pipeline_type=sample_pipeline_config["pipeline_type"],
            steps_config=sample_pipeline_config["steps_config"],
            created_by="test_user"
        )
        
        validation_result = await ml_pipeline_service.validate_pipeline_configuration(pipeline)
        
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0
        assert "step_count" in validation_result["checks_performed"]
        assert "step_dependencies" in validation_result["checks_performed"]


class TestMLPipelineExecution:
    """Test ML pipeline execution."""
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_success(self, ml_pipeline_service, sample_pipeline_config, sample_training_data):
        """Test successful pipeline execution."""
        # Create pipeline
        pipeline = await ml_pipeline_service.create_pipeline(
            name=sample_pipeline_config["name"],
            pipeline_type=sample_pipeline_config["pipeline_type"],
            steps_config=sample_pipeline_config["steps_config"],
            created_by="test_user"
        )
        
        # Execute pipeline
        execution_id = await ml_pipeline_service.execute_pipeline(
            pipeline_id=pipeline.id,
            input_data=sample_training_data,
            user_id=uuid4()
        )
        
        assert execution_id is not None
        assert isinstance(execution_id, str)
        
        # Allow some time for async execution
        import asyncio
        await asyncio.sleep(0.5)
        
        # Check execution status
        status = await ml_pipeline_service.get_execution_status(pipeline.id, execution_id)
        
        assert status["execution_id"] == execution_id
        assert status["status"] in ["running", "completed"]
        assert status["progress"] >= 0.0
        assert "started_at" in status
    
    @pytest.mark.asyncio
    async def test_get_execution_status_nonexistent(self, ml_pipeline_service):
        """Test getting status for nonexistent execution."""
        with pytest.raises(ValueError, match="Execution .* not found"):
            await ml_pipeline_service.get_execution_status(uuid4(), "nonexistent_execution_id")
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_with_config(self, ml_pipeline_service, sample_pipeline_config, sample_training_data):
        """Test pipeline execution with custom execution config."""
        pipeline = await ml_pipeline_service.create_pipeline(
            name=sample_pipeline_config["name"],
            pipeline_type=sample_pipeline_config["pipeline_type"],
            steps_config=sample_pipeline_config["steps_config"],
            created_by="test_user"
        )
        
        execution_config = {
            "model": {
                "type": "logistic_regression",
                "hyperparameters": {"C": 1.0, "random_state": 42}
            }
        }
        
        execution_id = await ml_pipeline_service.execute_pipeline(
            pipeline_id=pipeline.id,
            input_data=sample_training_data,
            execution_config=execution_config
        )
        
        assert execution_id is not None


class TestModelTraining:
    """Test model training functionality."""
    
    @pytest.mark.asyncio
    async def test_train_model_random_forest(self, ml_pipeline_service, sample_training_data, sample_validation_data):
        """Test training a random forest model."""
        model_config = {
            "algorithm": "random_forest",
            "hyperparameters": {
                "n_estimators": 10,  # Small for testing
                "max_depth": 5,
                "random_state": 42
            }
        }
        
        result = await ml_pipeline_service.train_model(
            pipeline_id=uuid4(),
            model_config=model_config,
            training_data=sample_training_data,
            validation_data=sample_validation_data
        )
        
        assert "model_id" in result
        assert "training_time_seconds" in result
        assert "train_metrics" in result
        assert "validation_metrics" in result
        assert result["model_type"] == "random_forest"
        assert result["training_samples"] > 0
        assert result["validation_samples"] > 0
        
        # Check metrics
        train_metrics = result["train_metrics"]
        val_metrics = result["validation_metrics"]
        
        assert "accuracy" in train_metrics
        assert "precision" in train_metrics
        assert "recall" in train_metrics
        assert "f1_score" in train_metrics
        
        assert all(0 <= metric <= 1 for metric in train_metrics.values())
        assert all(0 <= metric <= 1 for metric in val_metrics.values())
    
    @pytest.mark.asyncio
    async def test_train_model_logistic_regression(self, ml_pipeline_service, sample_training_data):
        """Test training a logistic regression model."""
        model_config = {
            "algorithm": "logistic_regression",
            "hyperparameters": {
                "C": 1.0,
                "random_state": 42,
                "max_iter": 100
            }
        }
        
        result = await ml_pipeline_service.train_model(
            pipeline_id=uuid4(),
            model_config=model_config,
            training_data=sample_training_data
        )
        
        assert result["model_type"] == "logistic_regression"
        assert "model_id" in result
        assert "validation_metrics" in result
    
    @pytest.mark.asyncio
    async def test_train_model_invalid_data(self, ml_pipeline_service):
        """Test training with invalid data format."""
        model_config = {
            "algorithm": "random_forest",
            "hyperparameters": {}
        }
        
        with pytest.raises(ValueError, match="Training data must be a pandas DataFrame"):
            await ml_pipeline_service.train_model(
                pipeline_id=uuid4(),
                model_config=model_config,
                training_data="invalid_data"
            )


class TestModelEvaluation:
    """Test model evaluation functionality."""
    
    @pytest.mark.asyncio
    async def test_evaluate_model_success(self, ml_pipeline_service, sample_training_data, sample_validation_data):
        """Test successful model evaluation."""
        # First train a model
        model_config = {
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 10, "random_state": 42}
        }
        
        training_result = await ml_pipeline_service.train_model(
            pipeline_id=uuid4(),
            model_config=model_config,
            training_data=sample_training_data,
            validation_data=sample_validation_data
        )
        
        model_id = training_result["model_id"]
        
        # Now evaluate the model
        evaluation_result = await ml_pipeline_service.evaluate_model(
            pipeline_id=uuid4(),
            model_id=uuid4(model_id),  # Convert string to UUID
            test_data=sample_validation_data
        )
        
        assert "evaluation_metrics" in evaluation_result
        assert "test_samples" in evaluation_result
        assert "evaluation_timestamp" in evaluation_result
        assert evaluation_result["test_samples"] > 0
        
        metrics = evaluation_result["evaluation_metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
    
    @pytest.mark.asyncio
    async def test_evaluate_nonexistent_model(self, ml_pipeline_service, sample_validation_data):
        """Test evaluation of nonexistent model."""
        with pytest.raises(ValueError, match="Model .* not found in registry"):
            await ml_pipeline_service.evaluate_model(
                pipeline_id=uuid4(),
                model_id=uuid4(),
                test_data=sample_validation_data
            )


class TestHyperparameterOptimization:
    """Test hyperparameter optimization."""
    
    @pytest.mark.asyncio
    async def test_optimize_hyperparameters_optuna(self, ml_pipeline_service, sample_training_data, sample_validation_data):
        """Test hyperparameter optimization with Optuna."""
        model_config = {
            "algorithm": "random_forest"
        }
        
        optimization_config = {
            "n_trials": 5,  # Small number for testing
            "search_space": {
                "n_estimators": {"low": 10, "high": 50},
                "max_depth": {"low": 3, "high": 10}
            }
        }
        
        result = await ml_pipeline_service.optimize_hyperparameters(
            pipeline_id=uuid4(),
            model_config=model_config,
            training_data=sample_training_data,
            validation_data=sample_validation_data,
            optimization_config=optimization_config
        )
        
        assert "optimization_id" in result
        assert "best_parameters" in result
        assert "best_score" in result
        assert "trials_completed" in result
        assert "duration_seconds" in result
        
        # Check that best parameters are reasonable
        best_params = result["best_parameters"]
        if "n_estimators" in best_params:
            assert isinstance(best_params["n_estimators"], int)
            assert 10 <= best_params["n_estimators"] <= 50
        
        if "max_depth" in best_params:
            assert isinstance(best_params["max_depth"], int)
            assert 3 <= best_params["max_depth"] <= 10
    
    @pytest.mark.asyncio
    @patch('packages.data_science.infrastructure.services.ml_pipeline_service_impl.OPTUNA_AVAILABLE', False)
    async def test_optimize_hyperparameters_fallback(self, ml_pipeline_service, sample_training_data, sample_validation_data):
        """Test hyperparameter optimization fallback when Optuna unavailable."""
        model_config = {
            "algorithm": "random_forest"
        }
        
        optimization_config = {
            "n_trials": 5
        }
        
        result = await ml_pipeline_service.optimize_hyperparameters(
            pipeline_id=uuid4(),
            model_config=model_config,
            training_data=sample_training_data,
            validation_data=sample_validation_data,
            optimization_config=optimization_config
        )
        
        assert "optimization_method" in result
        assert result["optimization_method"] == "grid_search"
        assert "best_parameters" in result
        assert "best_score" in result


class TestPipelineStepExecution:
    """Test individual pipeline step execution."""
    
    @pytest.mark.asyncio
    async def test_data_validation_step(self, ml_pipeline_service, sample_training_data):
        """Test data validation step."""
        service = ml_pipeline_service
        
        result = await service._validate_data_step(sample_training_data)
        
        assert result["status"] == "success"
        assert result["rows"] == len(sample_training_data)
        assert result["columns"] == len(sample_training_data.columns)
        assert "missing_values" in result
        assert "data_types" in result
    
    @pytest.mark.asyncio
    async def test_preprocessing_step(self, ml_pipeline_service, sample_training_data):
        """Test data preprocessing step."""
        service = ml_pipeline_service
        
        result = await service._preprocess_data_step(sample_training_data, {})
        
        assert result["status"] == "success"
        assert "processed_rows" in result
        assert "numeric_columns" in result
        assert "categorical_columns" in result
        assert "preprocessing_applied" in result
    
    @pytest.mark.asyncio
    async def test_feature_engineering_step(self, ml_pipeline_service, sample_training_data):
        """Test feature engineering step."""
        service = ml_pipeline_service
        
        result = await service._feature_engineering_step(sample_training_data, {})
        
        assert result["status"] == "success"
        assert "original_features" in result
        assert "engineered_features" in result
        assert "total_features" in result
        assert "feature_types" in result
    
    @pytest.mark.asyncio
    async def test_model_training_step(self, ml_pipeline_service, sample_training_data):
        """Test model training step."""
        service = ml_pipeline_service
        
        config = {
            "model": {
                "type": "random_forest",
                "hyperparameters": {"n_estimators": 10, "random_state": 42}
            }
        }
        
        result = await service._model_training_step(sample_training_data, config)
        
        assert result["status"] == "success"
        assert "model_id" in result
        assert "model_type" in result
        assert "training_time_seconds" in result
        assert "performance_metrics" in result
        assert "training_samples" in result
    
    @pytest.mark.asyncio
    async def test_model_evaluation_step(self, ml_pipeline_service, sample_training_data):
        """Test model evaluation step."""
        service = ml_pipeline_service
        
        result = await service._model_evaluation_step(sample_training_data, {})
        
        assert result["status"] == "success"
        assert "evaluation_metrics" in result
        assert "test_samples" in result
        assert "evaluation_timestamp" in result


class TestPipelineControl:
    """Test pipeline execution control."""
    
    @pytest.mark.asyncio
    async def test_pause_resume_pipeline(self, ml_pipeline_service, sample_pipeline_config, sample_training_data):
        """Test pausing and resuming pipeline execution."""
        pipeline = await ml_pipeline_service.create_pipeline(
            name=sample_pipeline_config["name"],
            pipeline_type=sample_pipeline_config["pipeline_type"],
            steps_config=sample_pipeline_config["steps_config"],
            created_by="test_user"
        )
        
        execution_id = await ml_pipeline_service.execute_pipeline(
            pipeline_id=pipeline.id,
            input_data=sample_training_data
        )
        
        # Pause pipeline
        await ml_pipeline_service.pause_pipeline(pipeline.id, execution_id)
        
        # Check status
        status = await ml_pipeline_service.get_execution_status(pipeline.id, execution_id)
        assert status["status"] == "paused"
        
        # Resume pipeline
        await ml_pipeline_service.resume_pipeline(pipeline.id, execution_id)
        
        status = await ml_pipeline_service.get_execution_status(pipeline.id, execution_id)
        assert status["status"] == "running"
    
    @pytest.mark.asyncio
    async def test_stop_pipeline(self, ml_pipeline_service, sample_pipeline_config, sample_training_data):
        """Test stopping pipeline execution."""
        pipeline = await ml_pipeline_service.create_pipeline(
            name=sample_pipeline_config["name"],
            pipeline_type=sample_pipeline_config["pipeline_type"],
            steps_config=sample_pipeline_config["steps_config"],
            created_by="test_user"
        )
        
        execution_id = await ml_pipeline_service.execute_pipeline(
            pipeline_id=pipeline.id,
            input_data=sample_training_data
        )
        
        # Stop pipeline
        await ml_pipeline_service.stop_pipeline(pipeline.id, execution_id, reason="Test stop")
        
        # Check status
        status = await ml_pipeline_service.get_execution_status(pipeline.id, execution_id)
        assert status["status"] == "stopped"
    
    @pytest.mark.asyncio
    async def test_retry_failed_step(self, ml_pipeline_service, sample_pipeline_config, sample_training_data):
        """Test retrying a failed pipeline step."""
        pipeline = await ml_pipeline_service.create_pipeline(
            name=sample_pipeline_config["name"],
            pipeline_type=sample_pipeline_config["pipeline_type"],
            steps_config=sample_pipeline_config["steps_config"],
            created_by="test_user"
        )
        
        execution_id = await ml_pipeline_service.execute_pipeline(
            pipeline_id=pipeline.id,
            input_data=sample_training_data
        )
        
        # Retry a step
        await ml_pipeline_service.retry_failed_step(
            pipeline.id, execution_id, "model_training"
        )
        
        # Check that retry was logged
        status = await ml_pipeline_service.get_execution_status(pipeline.id, execution_id)
        assert len(status["logs"]) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_create_pipeline_empty_steps(self, ml_pipeline_service):
        """Test creating pipeline with empty steps."""
        with pytest.raises(ValueError):
            await ml_pipeline_service.create_pipeline(
                name="empty_pipeline",
                pipeline_type="training",
                steps_config=[],
                created_by="test_user"
            )
    
    @pytest.mark.asyncio
    async def test_train_model_without_sklearn(self, ml_pipeline_service, sample_training_data):
        """Test training model when sklearn unavailable."""
        with patch('packages.data_science.infrastructure.services.ml_pipeline_service_impl.SKLEARN_AVAILABLE', False):
            with pytest.raises(ValueError, match="scikit-learn is required"):
                await ml_pipeline_service.train_model(
                    pipeline_id=uuid4(),
                    model_config={"algorithm": "random_forest"},
                    training_data=sample_training_data
                )
    
    @pytest.mark.asyncio
    async def test_execute_step_unknown_type(self, ml_pipeline_service, sample_training_data):
        """Test executing unknown step type."""
        service = ml_pipeline_service
        
        unknown_step = {
            "name": "unknown_step",
            "type": "unknown_type"
        }
        
        result = await service._execute_step(unknown_step, sample_training_data, {})
        
        assert result["status"] == "skipped"
        assert "Unknown step type" in result["message"]


class TestServiceIntegration:
    """Test service integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_workflow(self, ml_pipeline_service, sample_pipeline_config, 
                                              sample_training_data, sample_validation_data):
        """Test complete end-to-end pipeline workflow."""
        # 1. Create pipeline
        pipeline = await ml_pipeline_service.create_pipeline(
            name=sample_pipeline_config["name"],
            pipeline_type=sample_pipeline_config["pipeline_type"],
            steps_config=sample_pipeline_config["steps_config"],
            created_by="test_user"
        )
        
        assert pipeline.status == PipelineStatus.VALID
        
        # 2. Execute pipeline
        execution_id = await ml_pipeline_service.execute_pipeline(
            pipeline_id=pipeline.id,
            input_data=sample_training_data
        )
        
        assert execution_id is not None
        
        # 3. Train model separately
        model_result = await ml_pipeline_service.train_model(
            pipeline_id=pipeline.id,
            model_config={
                "algorithm": "random_forest",
                "hyperparameters": {"n_estimators": 10, "random_state": 42}
            },
            training_data=sample_training_data,
            validation_data=sample_validation_data
        )
        
        model_id = model_result["model_id"]
        
        # 4. Evaluate model
        evaluation_result = await ml_pipeline_service.evaluate_model(
            pipeline_id=pipeline.id,
            model_id=uuid4(model_id),
            test_data=sample_validation_data
        )
        
        assert "evaluation_metrics" in evaluation_result
        
        # 5. Check final execution status
        final_status = await ml_pipeline_service.get_execution_status(pipeline.id, execution_id)
        assert final_status["execution_id"] == execution_id
    
    @pytest.mark.asyncio
    async def test_pipeline_validation_workflow(self, ml_pipeline_service):
        """Test pipeline validation workflow."""
        # Create pipeline with complex dependencies
        complex_steps = [
            {
                "name": "step1",
                "type": "data_loading",
                "configuration": {}
            },
            {
                "name": "step2",
                "type": "data_preprocessing",
                "configuration": {},
                "dependencies": ["step1"]
            },
            {
                "name": "step3",
                "type": "model_training",
                "configuration": {"algorithm": "random_forest"},
                "dependencies": ["step2"]
            }
        ]
        
        pipeline = await ml_pipeline_service.create_pipeline(
            name="complex_pipeline",
            pipeline_type="training",
            steps_config=complex_steps,
            created_by="test_user"
        )
        
        # Validate pipeline
        validation_result = await ml_pipeline_service.validate_pipeline_configuration(pipeline)
        
        assert validation_result["valid"] is True
        assert "step_dependencies" in validation_result["checks_performed"]
        assert len(validation_result["errors"]) == 0