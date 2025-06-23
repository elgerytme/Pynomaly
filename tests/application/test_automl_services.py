"""Comprehensive tests for AutoML services."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, List, Any
import uuid

from tests.conftest_dependencies import requires_dependency, requires_dependencies

from pynomaly.application.services.automl_service import AutoMLService
from pynomaly.application.dto.automl_dto import (
    AutoMLRequestDTO,
    AutoMLResponseDTO,
    HyperparameterOptimizationRequestDTO,
    HyperparameterOptimizationResponseDTO,
    AlgorithmRecommendationRequestDTO,
    AlgorithmRecommendationResponseDTO,
    ExperimentTrackingRequestDTO,
    ExperimentTrackingResponseDTO
)
from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.exceptions import ProcessingError, ValidationError


@requires_dependency('optuna')
class TestAutoMLService:
    """Test AutoML service functionality."""
    
    @pytest.fixture
    def automl_service(self):
        """Create an AutoML service instance."""
        return AutoMLService(
            experiment_tracker=Mock(),
            hyperparameter_optimizer=Mock(),
            algorithm_registry=Mock(),
            model_evaluator=Mock()
        )
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return Dataset(
            id=str(uuid.uuid4()),
            name="test_dataset",
            data=Mock(),
            features=["feature1", "feature2", "feature3"],
            metadata={
                "rows": 1000,
                "columns": 3,
                "anomaly_rate": 0.05,
                "data_type": "tabular"
            }
        )
    
    @pytest.mark.asyncio
    async def test_algorithm_recommendation(self, automl_service, sample_dataset):
        """Test algorithm recommendation functionality."""
        request = AlgorithmRecommendationRequestDTO(
            dataset_characteristics={
                "n_samples": 1000,
                "n_features": 3,
                "anomaly_rate": 0.05,
                "data_type": "tabular",
                "has_temporal": False
            },
            performance_requirements={
                "max_training_time": 300,
                "max_inference_time": 0.1,
                "memory_limit": "1GB"
            },
            user_preferences={
                "interpretability": "high",
                "accuracy_priority": 0.8,
                "speed_priority": 0.6
            }
        )
        
        # Mock algorithm registry
        automl_service.algorithm_registry.get_suitable_algorithms.return_value = [
            {
                "name": "IsolationForest",
                "score": 0.85,
                "reasons": ["Good for tabular data", "Fast training"],
                "framework": "sklearn"
            },
            {
                "name": "LocalOutlierFactor", 
                "score": 0.78,
                "reasons": ["Local anomaly detection", "No training needed"],
                "framework": "sklearn"
            }
        ]
        
        response = await automl_service.recommend_algorithms(request)
        
        assert isinstance(response, AlgorithmRecommendationResponseDTO)
        assert len(response.recommendations) == 2
        assert response.recommendations[0].algorithm_name == "IsolationForest"
        assert response.recommendations[0].confidence_score == 0.85
        assert "Good for tabular data" in response.recommendations[0].reasoning
    
    @pytest.mark.asyncio
    async def test_hyperparameter_optimization(self, automl_service):
        """Test hyperparameter optimization functionality."""
        request = HyperparameterOptimizationRequestDTO(
            algorithm_name="IsolationForest",
            dataset_id=str(uuid.uuid4()),
            optimization_config={
                "n_trials": 50,
                "timeout": 300,
                "optimization_metric": "f1_score",
                "cross_validation_folds": 5
            },
            search_space={
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "contamination": {"type": "float", "low": 0.01, "high": 0.3},
                "max_features": {"type": "categorical", "choices": [0.5, 0.8, 1.0]}
            }
        )
        
        # Mock hyperparameter optimizer
        automl_service.hyperparameter_optimizer.optimize.return_value = {
            "best_params": {
                "n_estimators": 150,
                "contamination": 0.05,
                "max_features": 0.8
            },
            "best_score": 0.87,
            "optimization_history": [
                {"trial": 1, "params": {"n_estimators": 100}, "score": 0.82},
                {"trial": 2, "params": {"n_estimators": 150}, "score": 0.87}
            ],
            "total_trials": 50,
            "optimization_time": 245.5
        }
        
        response = await automl_service.optimize_hyperparameters(request)
        
        assert isinstance(response, HyperparameterOptimizationResponseDTO)
        assert response.best_parameters["n_estimators"] == 150
        assert response.best_score == 0.87
        assert response.optimization_metadata["total_trials"] == 50
        assert len(response.optimization_history) == 2
    
    @pytest.mark.asyncio
    async def test_automated_model_selection(self, automl_service, sample_dataset):
        """Test automated model selection with cross-validation."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            optimization_metric="f1_score",
            time_budget=600,
            algorithms_to_try=["IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
            cross_validation_folds=3,
            test_size=0.2
        )
        
        # Mock model evaluator
        automl_service.model_evaluator.evaluate_algorithms.return_value = [
            {
                "algorithm": "IsolationForest",
                "cv_scores": [0.85, 0.87, 0.84],
                "mean_score": 0.853,
                "std_score": 0.012,
                "training_time": 12.5,
                "inference_time": 0.05,
                "best_params": {"n_estimators": 150, "contamination": 0.05}
            },
            {
                "algorithm": "LocalOutlierFactor",
                "cv_scores": [0.78, 0.82, 0.79],
                "mean_score": 0.797,
                "std_score": 0.017,
                "training_time": 0.1,
                "inference_time": 0.02,
                "best_params": {"n_neighbors": 20}
            }
        ]
        
        response = await automl_service.run_automl(request)
        
        assert isinstance(response, AutoMLResponseDTO)
        assert response.best_model.algorithm_name == "IsolationForest"
        assert response.best_model.performance_metrics["mean_cv_score"] == 0.853
        assert len(response.model_comparison) == 2
        assert response.optimization_summary["total_time"] <= 600
    
    @pytest.mark.asyncio
    async def test_experiment_tracking(self, automl_service):
        """Test experiment tracking functionality."""
        request = ExperimentTrackingRequestDTO(
            experiment_name="anomaly_detection_automl",
            experiment_description="AutoML experiment for anomaly detection",
            dataset_info={
                "name": "test_dataset",
                "size": 1000,
                "features": 3
            },
            algorithms_tested=["IsolationForest", "LocalOutlierFactor"],
            optimization_config={
                "metric": "f1_score",
                "cv_folds": 5,
                "time_budget": 300
            }
        )
        
        # Mock experiment tracker
        experiment_id = str(uuid.uuid4())
        automl_service.experiment_tracker.create_experiment.return_value = experiment_id
        automl_service.experiment_tracker.get_experiment_results.return_value = {
            "experiment_id": experiment_id,
            "status": "completed",
            "best_algorithm": "IsolationForest",
            "best_score": 0.87,
            "total_trials": 45,
            "execution_time": 280.5,
            "models_trained": 12
        }
        
        response = await automl_service.track_experiment(request)
        
        assert isinstance(response, ExperimentTrackingResponseDTO)
        assert response.experiment_id == experiment_id
        assert response.status == "completed"
        assert response.results["best_algorithm"] == "IsolationForest"
        assert response.results["best_score"] == 0.87
    
    @pytest.mark.asyncio
    async def test_data_profiling(self, automl_service, sample_dataset):
        """Test data profiling for AutoML guidance."""
        # Mock data profiling
        with patch.object(automl_service, '_profile_dataset') as mock_profile:
            mock_profile.return_value = {
                "statistics": {
                    "n_samples": 1000,
                    "n_features": 3,
                    "missing_values": 0.02,
                    "numerical_features": 3,
                    "categorical_features": 0
                },
                "data_quality": {
                    "completeness": 0.98,
                    "consistency": 0.95,
                    "validity": 0.99,
                    "anomaly_detection_suitability": "high"
                },
                "recommendations": [
                    "Dataset is suitable for tree-based algorithms",
                    "Consider feature scaling for distance-based methods",
                    "Small dataset - ensemble methods recommended"
                ]
            }
            
            profile = await automl_service.profile_dataset(sample_dataset.id)
            
            assert profile["statistics"]["n_samples"] == 1000
            assert profile["data_quality"]["anomaly_detection_suitability"] == "high"
            assert len(profile["recommendations"]) == 3
    
    @pytest.mark.asyncio
    async def test_model_ensemble_creation(self, automl_service):
        """Test automated ensemble model creation."""
        base_models = [
            {
                "name": "IsolationForest",
                "performance": 0.85,
                "diversity_score": 0.7,
                "weight": 0.4
            },
            {
                "name": "LocalOutlierFactor", 
                "performance": 0.78,
                "diversity_score": 0.8,
                "weight": 0.35
            },
            {
                "name": "OneClassSVM",
                "performance": 0.82,
                "diversity_score": 0.6,
                "weight": 0.25
            }
        ]
        
        with patch.object(automl_service, '_create_ensemble') as mock_ensemble:
            mock_ensemble.return_value = {
                "ensemble_id": str(uuid.uuid4()),
                "base_models": base_models,
                "ensemble_method": "weighted_voting",
                "expected_performance": 0.89,
                "diversity_score": 0.72
            }
            
            ensemble = await automl_service.create_ensemble(base_models)
            
            assert ensemble["ensemble_method"] == "weighted_voting"
            assert ensemble["expected_performance"] == 0.89
            assert len(ensemble["base_models"]) == 3
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, automl_service):
        """Test performance optimization recommendations."""
        model_config = {
            "algorithm": "IsolationForest",
            "parameters": {"n_estimators": 100, "contamination": 0.1},
            "performance_metrics": {
                "training_time": 45.2,
                "inference_time": 0.12,
                "memory_usage": "512MB",
                "accuracy": 0.78
            }
        }
        
        with patch.object(automl_service, '_optimize_performance') as mock_optimize:
            mock_optimize.return_value = {
                "optimizations": [
                    {
                        "type": "parameter_tuning",
                        "suggestion": "Reduce n_estimators to 80",
                        "expected_speedup": 1.25,
                        "accuracy_impact": -0.02
                    },
                    {
                        "type": "feature_selection",
                        "suggestion": "Remove low-importance features",
                        "expected_speedup": 1.15,
                        "accuracy_impact": 0.01
                    }
                ],
                "overall_speedup": 1.44,
                "accuracy_change": -0.01
            }
            
            optimizations = await automl_service.optimize_performance(model_config)
            
            assert len(optimizations["optimizations"]) == 2
            assert optimizations["overall_speedup"] == 1.44
            assert optimizations["accuracy_change"] == -0.01
    
    @pytest.mark.asyncio
    async def test_automl_pipeline_validation(self, automl_service):
        """Test AutoML pipeline validation and error handling."""
        # Test invalid request
        invalid_request = AutoMLRequestDTO(
            dataset_id="invalid_id",
            task_type="invalid_task",
            optimization_metric="invalid_metric",
            time_budget=-100
        )
        
        with pytest.raises(ValidationError):
            await automl_service.run_automl(invalid_request)
        
        # Test missing dataset
        automl_service.experiment_tracker.get_dataset.return_value = None
        
        valid_request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            optimization_metric="f1_score",
            time_budget=300
        )
        
        with pytest.raises(ProcessingError, match="Dataset not found"):
            await automl_service.run_automl(valid_request)
    
    @pytest.mark.asyncio
    async def test_automl_progress_tracking(self, automl_service):
        """Test AutoML progress tracking and callbacks."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection", 
            optimization_metric="f1_score",
            time_budget=300
        )
        
        progress_updates = []
        
        def progress_callback(progress_info):
            progress_updates.append(progress_info)
        
        automl_service.add_progress_callback(progress_callback)
        
        # Mock progress updates
        with patch.object(automl_service, '_run_automl_with_progress') as mock_run:
            mock_run.side_effect = lambda req, callback: [
                callback({"stage": "data_profiling", "progress": 0.1}),
                callback({"stage": "algorithm_selection", "progress": 0.3}),
                callback({"stage": "hyperparameter_optimization", "progress": 0.7}),
                callback({"stage": "model_evaluation", "progress": 1.0})
            ]
            
            await automl_service.run_automl_with_tracking(request)
            
            assert len(progress_updates) == 4
            assert progress_updates[0]["stage"] == "data_profiling"
            assert progress_updates[-1]["progress"] == 1.0


class TestAutoMLIntegration:
    """Integration tests for AutoML services."""
    
    @pytest.fixture
    def automl_system(self):
        """Create a complete AutoML system setup."""
        return {
            'automl_service': AutoMLService(
                experiment_tracker=Mock(),
                hyperparameter_optimizer=Mock(),
                algorithm_registry=Mock(),
                model_evaluator=Mock()
            ),
            'dataset': Dataset(
                id=str(uuid.uuid4()),
                name="integration_test_dataset",
                data=Mock(),
                features=["f1", "f2", "f3", "f4"],
                metadata={"rows": 5000, "columns": 4}
            )
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_automl_workflow(self, automl_system):
        """Test complete end-to-end AutoML workflow."""
        automl_service = automl_system['automl_service']
        dataset = automl_system['dataset']
        
        # 1. Data profiling
        profile = await automl_service.profile_dataset(dataset.id)
        assert profile is not None
        
        # 2. Algorithm recommendation
        rec_request = AlgorithmRecommendationRequestDTO(
            dataset_characteristics=profile["statistics"],
            performance_requirements={"max_training_time": 300},
            user_preferences={"accuracy_priority": 0.8}
        )
        recommendations = await automl_service.recommend_algorithms(rec_request)
        assert len(recommendations.recommendations) > 0
        
        # 3. Hyperparameter optimization
        hp_request = HyperparameterOptimizationRequestDTO(
            algorithm_name=recommendations.recommendations[0].algorithm_name,
            dataset_id=dataset.id,
            optimization_config={"n_trials": 20}
        )
        hp_result = await automl_service.optimize_hyperparameters(hp_request)
        assert hp_result.best_score > 0
        
        # 4. Full AutoML run
        automl_request = AutoMLRequestDTO(
            dataset_id=dataset.id,
            task_type="anomaly_detection",
            optimization_metric="f1_score",
            time_budget=300
        )
        automl_result = await automl_service.run_automl(automl_request)
        assert automl_result.best_model is not None
    
    @pytest.mark.asyncio
    async def test_automl_with_constraints(self, automl_system):
        """Test AutoML with various constraints."""
        automl_service = automl_system['automl_service']
        dataset = automl_system['dataset']
        
        # Test with time constraints
        constrained_request = AutoMLRequestDTO(
            dataset_id=dataset.id,
            task_type="anomaly_detection",
            optimization_metric="f1_score",
            time_budget=60,  # Very limited time
            max_models=3,
            constraints={
                "max_memory": "256MB",
                "interpretability": "high",
                "training_time_limit": 30
            }
        )
        
        result = await automl_service.run_automl(constrained_request)
        assert result.optimization_summary["total_time"] <= 60
        assert len(result.model_comparison) <= 3
    
    @pytest.mark.asyncio
    async def test_automl_error_recovery(self, automl_system):
        """Test AutoML error handling and recovery."""
        automl_service = automl_system['automl_service']
        
        # Simulate partial failures
        automl_service.model_evaluator.evaluate_algorithms.side_effect = [
            ProcessingError("Algorithm 1 failed"),
            [{
                "algorithm": "IsolationForest",
                "cv_scores": [0.85, 0.87, 0.84],
                "mean_score": 0.853
            }]
        ]
        
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            optimization_metric="f1_score",
            time_budget=300,
            algorithms_to_try=["FailingAlgorithm", "IsolationForest"]
        )
        
        # Should recover and return results for successful algorithms
        result = await automl_service.run_automl(request)
        assert result.best_model is not None
        assert len(result.failed_algorithms) == 1
    
    @pytest.mark.asyncio
    async def test_multi_objective_optimization(self, automl_service):
        """Test multi-objective optimization functionality."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            optimization_metric=["f1_score", "precision", "recall"],
            time_budget=300,
            multi_objective_config={
                "method": "pareto_optimization",
                "weights": {"f1_score": 0.5, "precision": 0.3, "recall": 0.2}
            }
        )
        
        # Mock multi-objective results
        automl_service.model_evaluator.multi_objective_optimize.return_value = {
            "pareto_front": [
                {"algorithm": "IsolationForest", "f1_score": 0.85, "precision": 0.80, "recall": 0.90},
                {"algorithm": "LocalOutlierFactor", "f1_score": 0.82, "precision": 0.88, "recall": 0.77}
            ],
            "best_compromise": {"algorithm": "IsolationForest", "weighted_score": 0.835}
        }
        
        response = await automl_service.run_multi_objective_automl(request)
        assert len(response.pareto_solutions) == 2
        assert response.best_compromise_model.algorithm_name == "IsolationForest"
    
    @pytest.mark.asyncio
    async def test_transfer_learning_recommendations(self, automl_service):
        """Test transfer learning from similar datasets."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            transfer_learning={
                "enable": True,
                "source_experiments": ["exp1", "exp2"],
                "similarity_threshold": 0.7
            }
        )
        
        # Mock transfer learning
        automl_service.experiment_tracker.find_similar_experiments.return_value = [
            {
                "experiment_id": "exp1",
                "similarity": 0.85,
                "best_algorithm": "IsolationForest",
                "best_params": {"n_estimators": 150}
            }
        ]
        
        response = await automl_service.run_automl_with_transfer(request)
        assert "transfer_learning" in response.optimization_summary
        assert response.transfer_learning_info["similarity"] > 0.7
    
    @pytest.mark.asyncio
    async def test_incremental_learning_setup(self, automl_service):
        """Test incremental learning configuration."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            incremental_learning={
                "enable": True,
                "update_frequency": "daily",
                "drift_detection": True
            }
        )
        
        response = await automl_service.setup_incremental_learning(request)
        assert response.incremental_config["update_frequency"] == "daily"
        assert response.incremental_config["drift_detection"] is True
    
    @pytest.mark.asyncio
    async def test_automl_resource_optimization(self, automl_service):
        """Test AutoML resource optimization and constraints."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            resource_constraints={
                "max_memory": "1GB",
                "max_cpu_cores": 4,
                "max_gpu_memory": "512MB"
            }
        )
        
        response = await automl_service.optimize_for_resources(request)
        assert response.resource_optimized_models is not None
        assert response.resource_usage["memory"] <= "1GB"
    
    @pytest.mark.asyncio
    async def test_automl_feature_engineering(self, automl_service):
        """Test automated feature engineering."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            feature_engineering={
                "enable": True,
                "methods": ["polynomial", "interaction", "statistical"],
                "max_features": 100
            }
        )
        
        # Mock feature engineering
        automl_service.feature_engineer.generate_features.return_value = {
            "original_features": 10,
            "engineered_features": 25,
            "selected_features": 15,
            "feature_importance": {"f1": 0.3, "f2_squared": 0.25}
        }
        
        response = await automl_service.run_automl_with_feature_engineering(request)
        assert response.feature_engineering_summary["engineered_features"] == 25
    
    @pytest.mark.asyncio
    async def test_automl_model_interpretability(self, automl_service):
        """Test AutoML with interpretability constraints."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            interpretability_requirements={
                "level": "high",
                "explanation_methods": ["shap", "lime"],
                "global_explanations": True
            }
        )
        
        response = await automl_service.run_interpretable_automl(request)
        assert response.interpretability_score > 0.8
        assert "shap" in response.available_explanations
    
    @pytest.mark.asyncio
    async def test_automl_streaming_adaptation(self, automl_service):
        """Test AutoML adaptation for streaming data."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="streaming_anomaly_detection",
            streaming_config={
                "window_size": 1000,
                "update_threshold": 0.1,
                "adaptation_method": "online_learning"
            }
        )
        
        response = await automl_service.setup_streaming_automl(request)
        assert response.streaming_model_config["window_size"] == 1000
        assert response.adaptation_strategy == "online_learning"
    
    @pytest.mark.asyncio
    async def test_automl_distributed_execution(self, automl_service):
        """Test distributed AutoML execution."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            distributed_config={
                "enable": True,
                "n_workers": 4,
                "parallel_trials": 8
            }
        )
        
        response = await automl_service.run_distributed_automl(request)
        assert response.distributed_execution_info["n_workers"] == 4
        assert response.optimization_summary["parallel_trials"] == 8
    
    @pytest.mark.asyncio
    async def test_automl_cost_optimization(self, automl_service):
        """Test cost-aware AutoML optimization."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            cost_constraints={
                "max_training_cost": 100.0,
                "max_inference_cost_per_sample": 0.01,
                "cost_metric": "computational_units"
            }
        )
        
        response = await automl_service.run_cost_aware_automl(request)
        assert response.cost_analysis["total_training_cost"] <= 100.0
        assert response.cost_analysis["inference_cost_per_sample"] <= 0.01
    
    @pytest.mark.asyncio
    async def test_automl_fairness_optimization(self, automl_service):
        """Test fairness-aware AutoML optimization."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            fairness_constraints={
                "protected_attributes": ["age_group", "location"],
                "fairness_metrics": ["demographic_parity", "equalized_odds"],
                "fairness_threshold": 0.1
            }
        )
        
        response = await automl_service.run_fair_automl(request)
        assert response.fairness_analysis["demographic_parity"] <= 0.1
        assert "age_group" in response.fairness_analysis["protected_attributes"]
    
    @pytest.mark.asyncio
    async def test_automl_uncertainty_quantification(self, automl_service):
        """Test AutoML with uncertainty quantification."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            uncertainty_quantification={
                "enable": True,
                "method": "bayesian",
                "confidence_intervals": True
            }
        )
        
        response = await automl_service.run_automl_with_uncertainty(request)
        assert response.uncertainty_metrics["prediction_intervals"] is not None
        assert response.uncertainty_metrics["epistemic_uncertainty"] > 0
    
    @pytest.mark.asyncio
    async def test_automl_active_learning(self, automl_service):
        """Test AutoML with active learning strategies."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            active_learning={
                "enable": True,
                "query_strategy": "uncertainty_sampling",
                "budget": 100
            }
        )
        
        response = await automl_service.run_automl_with_active_learning(request)
        assert response.active_learning_info["queries_made"] <= 100
        assert response.active_learning_info["improvement_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_automl_robust_optimization(self, automl_service):
        """Test robust AutoML optimization against adversarial examples."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            robustness_config={
                "adversarial_training": True,
                "noise_tolerance": 0.1,
                "robustness_metrics": ["attack_success_rate"]
            }
        )
        
        response = await automl_service.run_robust_automl(request)
        assert response.robustness_analysis["attack_success_rate"] < 0.1
        assert response.robustness_analysis["noise_tolerance"] == 0.1
    
    @pytest.mark.asyncio
    async def test_automl_continual_learning(self, automl_service):
        """Test continual learning setup for evolving data."""
        request = AutoMLRequestDTO(
            dataset_id=str(uuid.uuid4()),
            task_type="anomaly_detection",
            continual_learning={
                "enable": True,
                "forgetting_prevention": "elastic_weight_consolidation",
                "task_boundary_detection": True
            }
        )
        
        response = await automl_service.setup_continual_learning(request)
        assert response.continual_learning_config["forgetting_prevention"] == "elastic_weight_consolidation"
        assert response.continual_learning_config["task_boundary_detection"] is True