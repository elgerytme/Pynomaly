"""Comprehensive service layer integration tests.

This module contains integration tests for the service layer components,
testing the interaction between application services, domain services,
and infrastructure adapters.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import uuid

from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate, PerformanceMetrics
from pynomaly.domain.services.threshold_calculator import ThresholdCalculator
from pynomaly.domain.services.ensemble_aggregator import EnsembleAggregator
from pynomaly.domain.services.anomaly_scorer import AnomalyScorer
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestDetectionServiceIntegration:
    """Test detection service integration with domain and infrastructure layers."""
    
    @pytest.fixture
    def sample_detection_data(self):
        """Create sample data for detection service testing."""
        np.random.seed(42)
        
        # Generate normal samples
        normal_samples = np.random.multivariate_normal(
            mean=[0, 0, 0],
            cov=[[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]],
            size=900
        )
        
        # Generate anomalous samples
        anomaly_samples = np.random.multivariate_normal(
            mean=[3, 3, 3],
            cov=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
            size=100
        )
        
        # Combine samples
        data = np.vstack([normal_samples, anomaly_samples])
        labels = np.hstack([np.zeros(900), np.ones(100)])
        
        df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
        df['true_label'] = labels
        
        return Dataset(name="Detection Service Test Data", data=df)
    
    @pytest.fixture
    def mock_detection_service(self):
        """Create mock detection service for testing."""
        class MockDetectionService:
            def __init__(self):
                self.adapters = {}
                self.threshold_calculator = ThresholdCalculator()
                self.anomaly_scorer = AnomalyScorer()
            
            def register_adapter(self, name: str, adapter):
                """Register an adapter with the service."""
                self.adapters[name] = adapter
            
            def train_detector(self, adapter_name: str, dataset: Dataset, parameters: Dict[str, Any]):
                """Train a detector using specified adapter."""
                if adapter_name not in self.adapters:
                    raise ValueError(f"Adapter {adapter_name} not registered")
                
                adapter = self.adapters[adapter_name]
                
                # Update adapter parameters
                for key, value in parameters.items():
                    if hasattr(adapter, key):
                        setattr(adapter, key, value)
                
                # Train the adapter
                adapter.fit(dataset)
                
                return {
                    "adapter_name": adapter_name,
                    "dataset_id": str(dataset.id),
                    "trained_at": datetime.utcnow(),
                    "is_fitted": adapter.is_fitted
                }
            
            def detect_anomalies(self, adapter_name: str, dataset: Dataset, threshold_config: Optional[Dict] = None):
                """Detect anomalies using specified adapter."""
                if adapter_name not in self.adapters:
                    raise ValueError(f"Adapter {adapter_name} not registered")
                
                adapter = self.adapters[adapter_name]
                
                if not adapter.is_fitted:
                    raise ValueError(f"Adapter {adapter_name} is not trained")
                
                # Get anomaly scores
                scores = adapter.score(dataset)
                
                # Calculate threshold
                if threshold_config:
                    method = threshold_config.get("method", "contamination")
                    if method == "contamination":
                        contamination = threshold_config.get("contamination", 0.1)
                        threshold = self.threshold_calculator.calculate_contamination_threshold(
                            scores=scores,
                            contamination_rate=ContaminationRate(contamination)
                        )
                    elif method == "percentile":
                        percentile = threshold_config.get("percentile", 90.0)
                        threshold = self.threshold_calculator.calculate_percentile_threshold(
                            scores=scores,
                            percentile=percentile
                        )
                    else:
                        threshold = 0.5  # Default threshold
                else:
                    threshold = 0.5
                
                # Generate labels based on threshold
                labels = np.array([1 if score.value > threshold else 0 for score in scores])
                
                # Create detection result
                result = DetectionResult(
                    detector_id=adapter.id,
                    dataset_id=dataset.id,
                    scores=scores,
                    labels=labels,
                    threshold=threshold
                )
                
                return result
            
            def evaluate_performance(self, result: DetectionResult, true_labels: np.ndarray):
                """Evaluate detection performance."""
                predicted_labels = result.labels
                
                # Calculate confusion matrix
                tp = np.sum((predicted_labels == 1) & (true_labels == 1))
                fp = np.sum((predicted_labels == 1) & (true_labels == 0))
                tn = np.sum((predicted_labels == 0) & (true_labels == 0))
                fn = np.sum((predicted_labels == 0) & (true_labels == 1))
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
                
                return PerformanceMetrics(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score
                )
        
        return MockDetectionService()
    
    def test_detection_service_workflow_integration(self, mock_detection_service, sample_detection_data):
        """Test complete detection service workflow integration."""
        try:
            # Register adapters
            isolation_forest_adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 50,
                    "random_state": 42
                }
            )
            
            mock_detection_service.register_adapter("isolation_forest", isolation_forest_adapter)
            
            # Test training
            training_result = mock_detection_service.train_detector(
                adapter_name="isolation_forest",
                dataset=sample_detection_data,
                parameters={"contamination": 0.1}
            )
            
            # Verify training result
            assert training_result["adapter_name"] == "isolation_forest"
            assert training_result["is_fitted"] is True
            assert "trained_at" in training_result
            
            # Test detection
            detection_result = mock_detection_service.detect_anomalies(
                adapter_name="isolation_forest",
                dataset=sample_detection_data,
                threshold_config={
                    "method": "contamination",
                    "contamination": 0.1
                }
            )
            
            # Verify detection result
            assert isinstance(detection_result, DetectionResult)
            assert len(detection_result.scores) == sample_detection_data.n_samples
            assert len(detection_result.labels) == sample_detection_data.n_samples
            
            # Test performance evaluation
            true_labels = sample_detection_data.data['true_label'].values
            performance = mock_detection_service.evaluate_performance(
                result=detection_result,
                true_labels=true_labels
            )
            
            # Verify performance metrics
            assert isinstance(performance, PerformanceMetrics)
            assert 0.0 <= performance.accuracy <= 1.0
            assert 0.0 <= performance.precision <= 1.0
            assert 0.0 <= performance.recall <= 1.0
            assert 0.0 <= performance.f1_score <= 1.0
            
            # Performance should be reasonable for this dataset
            assert performance.accuracy > 0.5
            assert performance.f1_score > 0.3
            
        except ImportError:
            pytest.skip("scikit-learn not available")
    
    def test_detection_service_multiple_adapters_integration(self, mock_detection_service, sample_detection_data):
        """Test detection service with multiple adapters."""
        try:
            # Register multiple adapters
            adapters_config = [
                ("isolation_forest", "IsolationForest", {
                    "contamination": 0.1,
                    "n_estimators": 30,
                    "random_state": 42
                }),
                ("local_outlier_factor", "LocalOutlierFactor", {
                    "contamination": 0.1,
                    "n_neighbors": 20,
                    "novelty": True
                })
            ]
            
            # Register and train all adapters
            training_results = {}
            
            for adapter_name, algorithm_name, parameters in adapters_config:
                try:
                    adapter = SklearnAdapter(
                        algorithm_name=algorithm_name,
                        parameters=parameters
                    )
                    
                    mock_detection_service.register_adapter(adapter_name, adapter)
                    
                    training_result = mock_detection_service.train_detector(
                        adapter_name=adapter_name,
                        dataset=sample_detection_data,
                        parameters=parameters
                    )
                    
                    training_results[adapter_name] = training_result
                    
                except Exception:
                    continue
            
            # Verify at least one adapter was trained
            assert len(training_results) > 0
            
            # Test detection with all trained adapters
            detection_results = {}
            
            for adapter_name in training_results.keys():
                detection_result = mock_detection_service.detect_anomalies(
                    adapter_name=adapter_name,
                    dataset=sample_detection_data,
                    threshold_config={
                        "method": "contamination",
                        "contamination": 0.1
                    }
                )
                
                detection_results[adapter_name] = detection_result
            
            # Verify all detections completed
            assert len(detection_results) == len(training_results)
            
            # Compare performance across adapters
            true_labels = sample_detection_data.data['true_label'].values
            performance_comparison = {}
            
            for adapter_name, detection_result in detection_results.items():
                performance = mock_detection_service.evaluate_performance(
                    result=detection_result,
                    true_labels=true_labels
                )
                
                performance_comparison[adapter_name] = {
                    "accuracy": performance.accuracy,
                    "precision": performance.precision,
                    "recall": performance.recall,
                    "f1_score": performance.f1_score,
                    "contamination_rate": np.mean(detection_result.labels)
                }
            
            # Verify performance metrics are reasonable
            for adapter_name, metrics in performance_comparison.items():
                assert 0.0 <= metrics["accuracy"] <= 1.0
                assert 0.0 <= metrics["precision"] <= 1.0
                assert 0.0 <= metrics["recall"] <= 1.0
                assert 0.0 <= metrics["f1_score"] <= 1.0
                assert 0.0 <= metrics["contamination_rate"] <= 0.5
            
        except ImportError:
            pytest.skip("scikit-learn not available")
    
    def test_detection_service_error_handling_integration(self, mock_detection_service):
        """Test detection service error handling integration."""
        # Test unregistered adapter
        with pytest.raises(ValueError, match="not registered"):
            mock_detection_service.train_detector(
                adapter_name="nonexistent",
                dataset=Mock(),
                parameters={}
            )
        
        # Test detection with untrained adapter
        try:
            untrained_adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={"contamination": 0.1}
            )
            
            mock_detection_service.register_adapter("untrained", untrained_adapter)
            
            # Create minimal dataset for testing
            test_data = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'feature2': [1, 2, 3, 4, 5]
            })
            test_dataset = Dataset(name="Test", data=test_data)
            
            with pytest.raises(ValueError, match="not trained"):
                mock_detection_service.detect_anomalies(
                    adapter_name="untrained",
                    dataset=test_dataset
                )
                
        except ImportError:
            pytest.skip("scikit-learn not available")


class TestEnsembleServiceIntegration:
    """Test ensemble service integration."""
    
    @pytest.fixture
    def ensemble_test_data(self):
        """Create test data for ensemble service."""
        np.random.seed(123)
        
        # Generate data with clear patterns
        normal_data = np.random.multivariate_normal(
            mean=[0, 0],
            cov=[[1, 0.3], [0.3, 1]],
            size=800
        )
        
        anomaly_data = np.random.multivariate_normal(
            mean=[4, 4],
            cov=[[0.5, 0], [0, 0.5]],
            size=200
        )
        
        data = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([np.zeros(800), np.ones(200)])
        
        df = pd.DataFrame(data, columns=['x', 'y'])
        df['true_label'] = labels
        
        return Dataset(name="Ensemble Test Data", data=df)
    
    @pytest.fixture
    def mock_ensemble_service(self):
        """Create mock ensemble service."""
        class MockEnsembleService:
            def __init__(self):
                self.adapters = {}
                self.ensemble_aggregator = EnsembleAggregator()
            
            def register_detector(self, name: str, adapter):
                """Register a detector for ensemble."""
                self.adapters[name] = adapter
            
            def train_ensemble(self, dataset: Dataset, detector_configs: List[Dict]):
                """Train ensemble of detectors."""
                training_results = {}
                
                for config in detector_configs:
                    name = config["name"]
                    algorithm = config["algorithm"]
                    parameters = config["parameters"]
                    
                    try:
                        adapter = SklearnAdapter(
                            algorithm_name=algorithm,
                            parameters=parameters
                        )
                        
                        adapter.fit(dataset)
                        self.register_detector(name, adapter)
                        
                        training_results[name] = {
                            "success": True,
                            "algorithm": algorithm,
                            "is_fitted": adapter.is_fitted
                        }
                        
                    except Exception as e:
                        training_results[name] = {
                            "success": False,
                            "error": str(e)
                        }
                
                return training_results
            
            def ensemble_predict(self, dataset: Dataset, aggregation_method: str = "voting"):
                """Make ensemble predictions."""
                if not self.adapters:
                    raise ValueError("No detectors available for ensemble")
                
                # Get predictions from all detectors
                detector_predictions = {}
                
                for name, adapter in self.adapters.items():
                    if adapter.is_fitted:
                        scores = adapter.score(dataset)
                        detector_predictions[name] = scores
                
                if not detector_predictions:
                    raise ValueError("No trained detectors available")
                
                # Aggregate predictions
                if aggregation_method == "voting":
                    result = self.ensemble_aggregator.aggregate_voting(
                        predictions=detector_predictions,
                        threshold=0.5,
                        voting_strategy="majority"
                    )
                elif aggregation_method == "average":
                    weights = {name: 1.0 / len(detector_predictions) for name in detector_predictions}
                    result = self.ensemble_aggregator.aggregate_weighted_average(
                        predictions=detector_predictions,
                        weights=weights
                    )
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation_method}")
                
                return result
            
            def evaluate_ensemble(self, dataset: Dataset, true_labels: np.ndarray, aggregation_method: str = "voting"):
                """Evaluate ensemble performance."""
                ensemble_result = self.ensemble_predict(dataset, aggregation_method)
                
                ensemble_labels = ensemble_result.get("ensemble_labels", [])
                
                if len(ensemble_labels) == 0:
                    return None
                
                # Calculate performance metrics
                tp = np.sum((ensemble_labels == 1) & (true_labels == 1))
                fp = np.sum((ensemble_labels == 1) & (true_labels == 0))
                tn = np.sum((ensemble_labels == 0) & (true_labels == 0))
                fn = np.sum((ensemble_labels == 0) & (true_labels == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0.0
                
                return {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "ensemble_result": ensemble_result
                }
        
        return MockEnsembleService()
    
    def test_ensemble_service_training_integration(self, mock_ensemble_service, ensemble_test_data):
        """Test ensemble service training integration."""
        try:
            # Define ensemble configuration
            detector_configs = [
                {
                    "name": "if_conservative",
                    "algorithm": "IsolationForest",
                    "parameters": {
                        "contamination": 0.1,
                        "n_estimators": 30,
                        "random_state": 42
                    }
                },
                {
                    "name": "if_aggressive",
                    "algorithm": "IsolationForest", 
                    "parameters": {
                        "contamination": 0.25,
                        "n_estimators": 30,
                        "random_state": 123
                    }
                },
                {
                    "name": "lof_detector",
                    "algorithm": "LocalOutlierFactor",
                    "parameters": {
                        "contamination": 0.2,
                        "n_neighbors": 20,
                        "novelty": True
                    }
                }
            ]
            
            # Train ensemble
            training_results = mock_ensemble_service.train_ensemble(
                dataset=ensemble_test_data,
                detector_configs=detector_configs
            )
            
            # Verify training results
            assert len(training_results) == len(detector_configs)
            
            successful_trainings = [name for name, result in training_results.items() if result["success"]]
            assert len(successful_trainings) > 0, "No detectors trained successfully"
            
            # Verify trained detectors
            for name in successful_trainings:
                assert training_results[name]["is_fitted"] is True
            
        except ImportError:
            pytest.skip("scikit-learn not available")
    
    def test_ensemble_service_prediction_integration(self, mock_ensemble_service, ensemble_test_data):
        """Test ensemble service prediction integration."""
        try:
            # Train a simple ensemble
            detector_configs = [
                {
                    "name": "detector_1",
                    "algorithm": "IsolationForest",
                    "parameters": {
                        "contamination": 0.15,
                        "n_estimators": 20,
                        "random_state": 42
                    }
                },
                {
                    "name": "detector_2",
                    "algorithm": "IsolationForest",
                    "parameters": {
                        "contamination": 0.25,
                        "n_estimators": 20,
                        "random_state": 123
                    }
                }
            ]
            
            training_results = mock_ensemble_service.train_ensemble(
                dataset=ensemble_test_data,
                detector_configs=detector_configs
            )
            
            successful_trainings = [name for name, result in training_results.items() if result["success"]]
            
            if len(successful_trainings) < 2:
                pytest.skip("Need at least 2 successful trainings for ensemble")
            
            # Test different aggregation methods
            aggregation_methods = ["voting", "average"]
            
            for method in aggregation_methods:
                ensemble_result = mock_ensemble_service.ensemble_predict(
                    dataset=ensemble_test_data,
                    aggregation_method=method
                )
                
                # Verify ensemble result structure
                assert isinstance(ensemble_result, dict)
                
                if method == "voting":
                    assert "ensemble_labels" in ensemble_result
                    assert "voting_details" in ensemble_result
                    
                    labels = ensemble_result["ensemble_labels"]
                    assert len(labels) == ensemble_test_data.n_samples
                    assert all(label in [0, 1] for label in labels)
                
                elif method == "average":
                    assert "ensemble_scores" in ensemble_result
                    
                    scores = ensemble_result["ensemble_scores"]
                    assert len(scores) == ensemble_test_data.n_samples
                    assert all(0.0 <= score.value <= 1.0 for score in scores)
            
        except ImportError:
            pytest.skip("scikit-learn not available")
    
    def test_ensemble_service_evaluation_integration(self, mock_ensemble_service, ensemble_test_data):
        """Test ensemble service evaluation integration."""
        try:
            # Train ensemble
            detector_configs = [
                {
                    "name": "detector_a",
                    "algorithm": "IsolationForest",
                    "parameters": {
                        "contamination": 0.2,
                        "n_estimators": 25,
                        "random_state": 42
                    }
                }
            ]
            
            training_results = mock_ensemble_service.train_ensemble(
                dataset=ensemble_test_data,
                detector_configs=detector_configs
            )
            
            successful_trainings = [name for name, result in training_results.items() if result["success"]]
            
            if len(successful_trainings) == 0:
                pytest.skip("No successful trainings for evaluation")
            
            # Evaluate ensemble
            true_labels = ensemble_test_data.data['true_label'].values
            
            evaluation_result = mock_ensemble_service.evaluate_ensemble(
                dataset=ensemble_test_data,
                true_labels=true_labels,
                aggregation_method="voting"
            )
            
            # Verify evaluation result
            assert evaluation_result is not None
            assert "accuracy" in evaluation_result
            assert "precision" in evaluation_result
            assert "recall" in evaluation_result
            assert "f1_score" in evaluation_result
            
            # Verify metrics are valid
            metrics = ["accuracy", "precision", "recall", "f1_score"]
            for metric in metrics:
                assert 0.0 <= evaluation_result[metric] <= 1.0
            
            # Ensemble should perform reasonably on this dataset
            assert evaluation_result["accuracy"] > 0.5
            
        except ImportError:
            pytest.skip("scikit-learn not available")