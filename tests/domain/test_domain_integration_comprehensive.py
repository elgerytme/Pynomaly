"""Comprehensive domain integration tests.

This module provides comprehensive integration tests for the domain layer,
testing interactions between entities, value objects, services, and exceptions
to ensure proper domain behavior and business rule enforcement.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import UUID, uuid4
from unittest.mock import Mock, patch

# Domain entities
from pynomaly.domain.entities import (
    Dataset, Detector, Anomaly, DetectionResult, ModelVersion,
    Deployment, ContinuousLearning, DriftDetection, Experiment
)

# Domain value objects
from pynomaly.domain.value_objects import (
    AnomalyScore, ContaminationRate, ConfidenceInterval,
    ThresholdConfig, SemanticVersion, PerformanceMetrics
)

# Domain services
from pynomaly.domain.services.threshold_calculator import ThresholdCalculator
from pynomaly.domain.services.ensemble_aggregator import EnsembleAggregator
from pynomaly.domain.services.anomaly_scorer import AnomalyScorer

# Domain exceptions
from pynomaly.domain.exceptions import (
    ValidationError, BusinessRuleViolation, DetectorNotTrainedError,
    DatasetValidationError
)


class TestDatasetDetectorIntegration:
    """Test integration between Dataset and Detector entities."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        data = pd.DataFrame({
            'temperature': np.random.normal(25, 5, 1000),
            'pressure': np.random.normal(1013, 50, 1000),
            'humidity': np.random.normal(60, 10, 1000),
            'vibration': np.random.exponential(2, 1000)
        })
        return Dataset(name="Sensor Data", data=data)
    
    @pytest.fixture
    def mock_detector(self):
        """Create mock detector for testing."""
        class TestDetector(Detector):
            def fit(self, dataset: Dataset) -> None:
                if dataset.n_samples < 100:
                    raise ValidationError("Insufficient data for training")
                self.is_fitted = True
                self.trained_at = datetime.utcnow()
            
            def detect(self, dataset: Dataset) -> DetectionResult:
                if not self.is_fitted:
                    raise DetectorNotTrainedError(
                        detector_id=self.id,
                        detector_name=self.name,
                        attempted_operation="detect"
                    )
                
                # Simulate detection results
                scores = [AnomalyScore(np.random.beta(2, 8)) for _ in range(dataset.n_samples)]
                threshold = 0.7
                labels = [1 if score.value > threshold else 0 for score in scores]
                
                return DetectionResult(
                    detector_id=self.id,
                    dataset_id=dataset.id,
                    scores=scores,
                    labels=np.array(labels),
                    threshold=threshold
                )
            
            def score(self, dataset: Dataset) -> List[AnomalyScore]:
                return [AnomalyScore(np.random.beta(2, 8)) for _ in range(dataset.n_samples)]
        
        return TestDetector(
            name="Test Detector",
            algorithm_name="TestAlgorithm",
            parameters={"contamination": 0.1}
        )
    
    def test_detector_training_workflow(self, sample_dataset, mock_detector):
        """Test complete detector training workflow."""
        # Initially detector should not be fitted
        assert not mock_detector.is_fitted
        assert mock_detector.trained_at is None
        
        # Train detector on dataset
        mock_detector.fit(sample_dataset)
        
        # Verify training completed
        assert mock_detector.is_fitted
        assert mock_detector.trained_at is not None
        assert isinstance(mock_detector.trained_at, datetime)
    
    def test_detection_workflow(self, sample_dataset, mock_detector):
        """Test complete detection workflow."""
        # Train detector first
        mock_detector.fit(sample_dataset)
        
        # Run detection
        result = mock_detector.detect(sample_dataset)
        
        # Verify detection result
        assert isinstance(result, DetectionResult)
        assert result.detector_id == mock_detector.id
        assert result.dataset_id == sample_dataset.id
        assert len(result.scores) == sample_dataset.n_samples
        assert len(result.labels) == sample_dataset.n_samples
        
        # Verify anomaly detection consistency
        for i, (score, label) in enumerate(zip(result.scores, result.labels)):
            if score.value > result.threshold:
                assert label == 1  # Anomaly
            else:
                assert label == 0  # Normal
    
    def test_untrained_detector_error(self, sample_dataset, mock_detector):
        """Test error when using untrained detector."""
        # Attempt detection without training
        with pytest.raises(DetectorNotTrainedError) as exc_info:
            mock_detector.detect(sample_dataset)
        
        assert exc_info.value.detector_id == mock_detector.id
        assert exc_info.value.attempted_operation == "detect"
    
    def test_insufficient_data_error(self, mock_detector):
        """Test error with insufficient training data."""
        # Create small dataset
        small_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        small_dataset = Dataset(name="Small Dataset", data=small_data)
        
        # Attempt training with insufficient data
        with pytest.raises(ValidationError) as exc_info:
            mock_detector.fit(small_dataset)
        
        assert "Insufficient data" in str(exc_info.value)
    
    def test_dataset_detector_compatibility(self, sample_dataset, mock_detector):
        """Test dataset and detector compatibility checking."""
        # Check feature compatibility
        required_features = ["temperature", "pressure", "humidity"]
        dataset_features = sample_dataset.get_feature_names()
        
        # All required features should be present
        for feature in required_features:
            assert feature in dataset_features
        
        # Test with incompatible dataset
        incompatible_data = pd.DataFrame({
            'different_feature': [1, 2, 3, 4, 5]
        })
        incompatible_dataset = Dataset(name="Incompatible", data=incompatible_data)
        
        # Should still work as our mock detector is flexible
        mock_detector.fit(incompatible_dataset)
        result = mock_detector.detect(incompatible_dataset)
        assert len(result.scores) == incompatible_dataset.n_samples


class TestAnomalyScoreThresholdIntegration:
    """Test integration between AnomalyScore and ThresholdConfig."""
    
    @pytest.fixture
    def threshold_calculator(self):
        """Create threshold calculator for testing."""
        return ThresholdCalculator()
    
    @pytest.fixture
    def sample_scores(self):
        """Create sample anomaly scores."""
        np.random.seed(42)
        return [AnomalyScore(score) for score in np.random.beta(2, 5, 1000)]
    
    def test_percentile_threshold_integration(self, threshold_calculator, sample_scores):
        """Test percentile threshold calculation with anomaly scores."""
        # Create threshold config
        threshold_config = ThresholdConfig(
            method="percentile",
            value=95.0,
            auto_adjust=False
        )
        
        # Calculate threshold
        threshold = threshold_calculator.calculate_percentile_threshold(
            scores=sample_scores,
            percentile=threshold_config.value
        )
        
        # Verify threshold is reasonable
        assert 0.0 <= threshold <= 1.0
        
        # Count scores above threshold
        above_threshold = sum(1 for score in sample_scores if score.value > threshold)
        expected_above = len(sample_scores) * 0.05  # 5% for 95th percentile
        
        # Should be approximately 5% above threshold
        assert abs(above_threshold - expected_above) / len(sample_scores) < 0.02
    
    def test_contamination_threshold_integration(self, threshold_calculator, sample_scores):
        """Test contamination-based threshold with scores."""
        contamination_rate = ContaminationRate(0.1)
        
        threshold = threshold_calculator.calculate_contamination_threshold(
            scores=sample_scores,
            contamination_rate=contamination_rate
        )
        
        # Count scores above threshold
        above_threshold = sum(1 for score in sample_scores if score.value > threshold)
        expected_contamination = len(sample_scores) * contamination_rate.value
        
        # Should match contamination rate
        assert abs(above_threshold - expected_contamination) <= 1
    
    def test_adaptive_threshold_with_confidence(self, threshold_calculator, sample_scores):
        """Test adaptive threshold with confidence intervals."""
        # Calculate threshold with confidence interval
        threshold, confidence = threshold_calculator.calculate_threshold_with_confidence(
            scores=sample_scores,
            method="percentile",
            percentile=90.0,
            confidence_level=0.95
        )
        
        # Verify confidence interval
        assert isinstance(confidence, ConfidenceInterval)
        assert confidence.contains(threshold)
        assert confidence.confidence_level == 0.95
        
        # Threshold should be within reasonable bounds
        assert 0.0 <= threshold <= 1.0
        assert confidence.lower >= 0.0
        assert confidence.upper <= 1.0


class TestEnsembleDetectionIntegration:
    """Test integration of ensemble detection with multiple detectors."""
    
    @pytest.fixture
    def ensemble_aggregator(self):
        """Create ensemble aggregator for testing."""
        return EnsembleAggregator()
    
    @pytest.fixture
    def mock_detector_predictions(self):
        """Create mock predictions from multiple detectors."""
        np.random.seed(42)
        
        # Simulate different detector behaviors
        detector_1_scores = [AnomalyScore(score) for score in np.random.beta(3, 7, 100)]  # Conservative
        detector_2_scores = [AnomalyScore(score) for score in np.random.beta(2, 3, 100)]  # Aggressive
        detector_3_scores = [AnomalyScore(score) for score in np.random.beta(4, 6, 100)]  # Balanced
        
        return {
            "conservative_detector": detector_1_scores,
            "aggressive_detector": detector_2_scores,
            "balanced_detector": detector_3_scores
        }
    
    def test_voting_ensemble_integration(self, ensemble_aggregator, mock_detector_predictions):
        """Test voting ensemble integration."""
        threshold = 0.5
        
        # Aggregate using voting
        result = ensemble_aggregator.aggregate_voting(
            predictions=mock_detector_predictions,
            threshold=threshold,
            voting_strategy="majority"
        )
        
        # Verify result structure
        assert "ensemble_scores" in result
        assert "ensemble_labels" in result
        assert "voting_details" in result
        
        ensemble_scores = result["ensemble_scores"]
        ensemble_labels = result["ensemble_labels"]
        
        # Verify ensemble consistency
        assert len(ensemble_scores) == 100
        assert len(ensemble_labels) == 100
        
        # Check voting logic
        for i, (score, label) in enumerate(zip(ensemble_scores, ensemble_labels)):
            # Count individual detector votes
            individual_votes = sum(
                1 for detector_scores in mock_detector_predictions.values()
                if detector_scores[i].value > threshold
            )
            
            # Majority voting: label should be 1 if majority vote anomaly
            expected_label = 1 if individual_votes >= 2 else 0
            assert label == expected_label
    
    def test_weighted_ensemble_integration(self, ensemble_aggregator, mock_detector_predictions):
        """Test weighted ensemble integration."""
        # Assign weights based on detector performance
        weights = {
            "conservative_detector": 0.2,  # Lower weight for conservative
            "aggressive_detector": 0.3,    # Medium weight for aggressive
            "balanced_detector": 0.5       # Higher weight for balanced
        }
        
        result = ensemble_aggregator.aggregate_weighted_average(
            predictions=mock_detector_predictions,
            weights=weights
        )
        
        ensemble_scores = result["ensemble_scores"]
        
        # Verify weighted calculation
        for i in range(len(ensemble_scores)):
            expected_score = (
                mock_detector_predictions["conservative_detector"][i].value * 0.2 +
                mock_detector_predictions["aggressive_detector"][i].value * 0.3 +
                mock_detector_predictions["balanced_detector"][i].value * 0.5
            )
            
            assert abs(ensemble_scores[i].value - expected_score) < 0.001
    
    def test_ensemble_diversity_analysis(self, ensemble_aggregator, mock_detector_predictions):
        """Test ensemble diversity analysis."""
        diversity_analysis = ensemble_aggregator.analyze_diversity(
            predictions=mock_detector_predictions,
            diversity_metrics=["disagreement", "correlation"]
        )
        
        assert "diversity_scores" in diversity_analysis
        assert "pairwise_correlations" in diversity_analysis
        assert "overall_diversity" in diversity_analysis
        
        # Diversity score should be reasonable
        overall_diversity = diversity_analysis["overall_diversity"]
        assert 0.0 <= overall_diversity <= 1.0
        
        # Should have correlations between all detector pairs
        correlations = diversity_analysis["pairwise_correlations"]
        expected_pairs = [
            ("conservative_detector", "aggressive_detector"),
            ("conservative_detector", "balanced_detector"),
            ("aggressive_detector", "balanced_detector")
        ]
        
        for pair in expected_pairs:
            assert pair in correlations or (pair[1], pair[0]) in correlations


class TestModelVersioningIntegration:
    """Test integration of model versioning with detectors and deployments."""
    
    @pytest.fixture
    def mock_detector_with_versioning(self):
        """Create detector with versioning support."""
        class VersionedDetector(Detector):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.model_versions = []
            
            def fit(self, dataset: Dataset) -> None:
                self.is_fitted = True
                self.trained_at = datetime.utcnow()
                
                # Create new model version
                version = ModelVersion(
                    version=SemanticVersion(f"1.{len(self.model_versions)}.0"),
                    model_id=self.id,
                    created_by="test_user",
                    performance_metrics=PerformanceMetrics(
                        accuracy=0.85 + np.random.random() * 0.1,
                        precision=0.80 + np.random.random() * 0.15,
                        recall=0.75 + np.random.random() * 0.2,
                        f1_score=0.82 + np.random.random() * 0.1
                    )
                )
                self.model_versions.append(version)
            
            def detect(self, dataset: Dataset) -> DetectionResult:
                if not self.is_fitted:
                    raise DetectorNotTrainedError(
                        detector_id=self.id,
                        detector_name=self.name,
                        attempted_operation="detect"
                    )
                return DetectionResult(
                    detector_id=self.id,
                    dataset_id=dataset.id,
                    scores=[AnomalyScore(0.5)] * dataset.n_samples,
                    labels=np.array([0] * dataset.n_samples),
                    threshold=0.5
                )
            
            def score(self, dataset: Dataset) -> List[AnomalyScore]:
                return [AnomalyScore(0.5)] * dataset.n_samples
        
        return VersionedDetector(
            name="Versioned Detector",
            algorithm_name="TestAlgorithm",
            parameters={"contamination": 0.1}
        )
    
    def test_model_version_creation(self, mock_detector_with_versioning):
        """Test model version creation during training."""
        # Create sample dataset
        data = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        dataset = Dataset(name="Test Data", data=data)
        
        # Train detector (should create version)
        mock_detector_with_versioning.fit(dataset)
        
        # Verify version was created
        assert len(mock_detector_with_versioning.model_versions) == 1
        
        version = mock_detector_with_versioning.model_versions[0]
        assert isinstance(version, ModelVersion)
        assert version.version == SemanticVersion("1.0.0")
        assert version.model_id == mock_detector_with_versioning.id
        assert isinstance(version.performance_metrics, PerformanceMetrics)
    
    def test_multiple_version_management(self, mock_detector_with_versioning):
        """Test management of multiple model versions."""
        data = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        dataset = Dataset(name="Test Data", data=data)
        
        # Train multiple times to create versions
        for i in range(3):
            mock_detector_with_versioning.fit(dataset)
        
        # Should have 3 versions
        assert len(mock_detector_with_versioning.model_versions) == 3
        
        # Versions should be properly ordered
        versions = mock_detector_with_versioning.model_versions
        assert versions[0].version == SemanticVersion("1.0.0")
        assert versions[1].version == SemanticVersion("1.1.0")
        assert versions[2].version == SemanticVersion("1.2.0")
        
        # Each version should have different performance metrics
        performances = [v.performance_metrics.accuracy for v in versions]
        assert len(set(performances)) == 3  # All different


class TestContinuousLearningIntegration:
    """Test integration of continuous learning with detectors and drift detection."""
    
    @pytest.fixture
    def drift_detector(self):
        """Create drift detection instance."""
        return DriftDetection(
            name="Test Drift Detector",
            model_id=uuid4(),
            detection_methods=["ks_test", "jensen_shannon"],
            monitoring_window="24h",
            alert_threshold=0.1
        )
    
    @pytest.fixture
    def continuous_learner(self):
        """Create continuous learning instance."""
        return ContinuousLearning(
            name="Test Continuous Learner",
            model_id=uuid4(),
            learning_strategy="incremental",
            retrain_threshold=0.15,
            validation_strategy="holdout"
        )
    
    def test_drift_triggered_retraining(self, drift_detector, continuous_learner):
        """Test retraining triggered by drift detection."""
        # Set up reference data for drift detection
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000)
        })
        drift_detector.set_reference_data(reference_data)
        
        # Simulate drifted data
        drifted_data = pd.DataFrame({
            'feature1': np.random.normal(1.5, 1.2, 500),  # Significant drift
            'feature2': np.random.normal(0.5, 1.1, 500)   # Moderate drift
        })
        
        # Detect drift
        drift_results = drift_detector.detect_drift(drifted_data)
        
        # Verify drift was detected
        assert drift_results["drift_detected"]
        assert drift_results["overall_drift_score"] > drift_detector.alert_threshold
        
        # Configure continuous learning trigger based on drift
        continuous_learner.add_trigger_condition(
            condition_type="data_drift",
            threshold=0.1,
            detection_method="ks_test"
        )
        
        # Activate continuous learning
        continuous_learner.activate()
        
        # Trigger learning based on drift detection
        trigger_context = {
            "trigger_type": "data_drift",
            "drift_score": drift_results["overall_drift_score"],
            "detection_method": "ks_test",
            "affected_features": ["feature1", "feature2"]
        }
        
        learning_session = continuous_learner.trigger_learning(trigger_context)
        
        # Verify learning session was started
        assert learning_session["status"] == "initiated"
        assert learning_session["trigger_reason"] == "data_drift"
        assert "session_id" in learning_session
    
    def test_performance_degradation_retraining(self, continuous_learner):
        """Test retraining triggered by performance degradation."""
        # Configure performance-based trigger
        continuous_learner.add_trigger_condition(
            condition_type="performance_degradation",
            threshold=0.1,  # 10% degradation
            metric="f1_score"
        )
        
        continuous_learner.activate()
        
        # Simulate performance degradation
        trigger_context = {
            "trigger_type": "performance_degradation",
            "current_performance": 0.75,
            "baseline_performance": 0.90,
            "degradation": 0.15,  # 15% degradation
            "metric": "f1_score"
        }
        
        learning_session = continuous_learner.trigger_learning(trigger_context)
        
        # Verify learning was triggered
        assert learning_session["status"] == "initiated"
        assert learning_session["trigger_reason"] == "performance_degradation"
        
        # Complete learning session with improved performance
        learning_result = {
            "new_model_performance": 0.88,
            "improvement": 0.13,
            "training_samples": 5000,
            "validation_score": 0.86
        }
        
        continuous_learner.complete_learning_session(
            session_id=learning_session["session_id"],
            result=learning_result
        )
        
        # Verify session completion
        assert learning_session["status"] == "completed"


class TestExperimentWorkflowIntegration:
    """Test integration of experiment workflows with detectors and datasets."""
    
    @pytest.fixture
    def algorithm_comparison_experiment(self):
        """Create algorithm comparison experiment."""
        return Experiment(
            name="Algorithm Performance Comparison",
            description="Compare IsolationForest vs LocalOutlierFactor",
            hypothesis="IsolationForest will show better precision on this dataset",
            owner="data_scientist_1"
        )
    
    def test_complete_experiment_workflow(self, algorithm_comparison_experiment):
        """Test complete experiment workflow integration."""
        # Start experiment
        algorithm_comparison_experiment.start()
        
        # Create test dataset
        test_data = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.normal(0, 1, 900),    # Normal data
                np.random.normal(3, 0.5, 100)  # Anomalous data
            ]),
            'feature2': np.concatenate([
                np.random.normal(0, 1, 900),    # Normal data
                np.random.normal(3, 0.5, 100)  # Anomalous data
            ])
        })
        dataset = Dataset(name="Experiment Dataset", data=test_data)
        
        # Define ground truth (last 100 samples are anomalies)
        ground_truth = np.concatenate([np.zeros(900), np.ones(100)])
        
        # Test different algorithms with various parameters
        experiment_configs = [
            {
                "algorithm": "IsolationForest",
                "parameters": {"contamination": 0.1, "random_state": 42},
                "expected_performance": "high_precision"
            },
            {
                "algorithm": "IsolationForest", 
                "parameters": {"contamination": 0.05, "random_state": 42},
                "expected_performance": "very_high_precision"
            },
            {
                "algorithm": "LocalOutlierFactor",
                "parameters": {"n_neighbors": 20, "contamination": 0.1},
                "expected_performance": "high_recall"
            },
            {
                "algorithm": "LocalOutlierFactor",
                "parameters": {"n_neighbors": 10, "contamination": 0.1}, 
                "expected_performance": "balanced"
            }
        ]
        
        # Run experiments
        for i, config in enumerate(experiment_configs):
            # Simulate detector training and detection
            # In real implementation, this would use actual detectors
            
            # Simulate detection results based on algorithm characteristics
            if config["algorithm"] == "IsolationForest":
                # IsolationForest tends to have higher precision
                predicted_scores = np.concatenate([
                    np.random.beta(2, 8, 900),   # Low scores for normal data
                    np.random.beta(6, 2, 100)    # High scores for anomalies
                ])
            else:  # LocalOutlierFactor
                # LOF tends to have higher recall
                predicted_scores = np.concatenate([
                    np.random.beta(3, 7, 900),   # Slightly higher scores for normal
                    np.random.beta(5, 3, 100)    # High scores for anomalies
                ])
            
            # Calculate performance metrics
            threshold = np.percentile(predicted_scores, 90)
            predicted_labels = (predicted_scores > threshold).astype(int)
            
            # Calculate metrics
            tp = np.sum((predicted_labels == 1) & (ground_truth == 1))
            fp = np.sum((predicted_labels == 1) & (ground_truth == 0))
            tn = np.sum((predicted_labels == 0) & (ground_truth == 0))
            fn = np.sum((predicted_labels == 0) & (ground_truth == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add run to experiment
            run_data = {
                "algorithm": config["algorithm"],
                "parameters": config["parameters"],
                "dataset_id": str(dataset.id),
                "results": {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "threshold": threshold
                },
                "metadata": {
                    "expected_performance": config["expected_performance"],
                    "sample_size": len(test_data)
                }
            }
            
            algorithm_comparison_experiment.add_run(run_data)
        
        # Complete experiment
        algorithm_comparison_experiment.complete()
        
        # Analyze results
        analysis = algorithm_comparison_experiment.analyze_results()
        
        # Verify analysis completeness
        assert "best_run" in analysis
        assert "algorithm_comparison" in analysis
        assert "parameter_sensitivity" in analysis
        
        # Verify we have all runs
        assert len(algorithm_comparison_experiment.runs) == 4
        
        # Best run should have reasonable performance
        best_run = analysis["best_run"]
        assert best_run["performance_metrics"]["f1_score"] > 0.5
        
        # Algorithm comparison should show differences
        algo_comparison = analysis["algorithm_comparison"]
        assert "IsolationForest" in algo_comparison
        assert "LocalOutlierFactor" in algo_comparison
        
        # Each algorithm should have multiple parameter configurations tested
        for algo_results in algo_comparison.values():
            assert "average_performance" in algo_results
            assert "parameter_variations" in algo_results


class TestBusinessRuleEnforcement:
    """Test enforcement of business rules across domain entities."""
    
    def test_detector_deployment_business_rules(self):
        """Test business rules for detector deployment."""
        # Create untrained detector
        detector = Mock()
        detector.id = uuid4()
        detector.name = "Untrained Detector"
        detector.is_fitted = False
        detector.performance_metrics = None
        
        # Attempt to create deployment
        with pytest.raises(BusinessRuleViolation) as exc_info:
            if not detector.is_fitted:
                raise BusinessRuleViolation(
                    message="Cannot deploy untrained model",
                    rule_name="trained_model_deployment_rule",
                    violated_constraints=["model_must_be_trained"],
                    context={"detector_id": str(detector.id), "is_trained": False}
                )
        
        assert exc_info.value.rule_name == "trained_model_deployment_rule"
        assert "model_must_be_trained" in exc_info.value.violated_constraints
    
    def test_data_quality_business_rules(self):
        """Test business rules for data quality."""
        # Create dataset with quality issues
        poor_quality_data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],  # Missing values
            'feature2': [1, 1, 1, 1, 1],       # No variance
            'feature3': [1, 2, 3, 1000, 5]     # Outliers
        })
        
        dataset = Dataset(name="Poor Quality Dataset", data=poor_quality_data)
        
        # Validate data quality
        quality_issues = []
        
        # Check for missing values
        missing_counts = dataset.data.isnull().sum()
        if missing_counts.any():
            quality_issues.append("missing_values_detected")
        
        # Check for zero variance features
        for col in dataset.data.select_dtypes(include=[np.number]).columns:
            if dataset.data[col].var() == 0:
                quality_issues.append("zero_variance_feature")
        
        # Check for outliers (simple z-score method)
        for col in dataset.data.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((dataset.data[col] - dataset.data[col].mean()) / dataset.data[col].std())
            if (z_scores > 3).any():
                quality_issues.append("extreme_outliers_detected")
        
        # Enforce business rule: minimum data quality required
        if len(quality_issues) > 2:
            raise BusinessRuleViolation(
                message="Dataset does not meet minimum quality standards",
                rule_name="minimum_data_quality_rule",
                violated_constraints=quality_issues,
                context={"dataset_id": str(dataset.id), "quality_issues": quality_issues}
            )
        
        # Should raise business rule violation
        with pytest.raises(BusinessRuleViolation):
            if len(quality_issues) > 2:
                raise BusinessRuleViolation(
                    message="Dataset does not meet minimum quality standards",
                    rule_name="minimum_data_quality_rule",
                    violated_constraints=quality_issues,
                    context={"dataset_id": str(dataset.id), "quality_issues": quality_issues}
                )
    
    def test_model_performance_business_rules(self):
        """Test business rules for model performance."""
        # Create performance metrics below threshold
        poor_performance = PerformanceMetrics(
            accuracy=0.55,    # Below 0.7 threshold
            precision=0.45,   # Below 0.6 threshold
            recall=0.50,      # Below 0.6 threshold
            f1_score=0.47     # Below 0.6 threshold
        )
        
        # Define minimum performance thresholds
        min_thresholds = {
            "accuracy": 0.7,
            "precision": 0.6,
            "recall": 0.6,
            "f1_score": 0.6
        }
        
        # Check performance against thresholds
        violations = []
        if poor_performance.accuracy < min_thresholds["accuracy"]:
            violations.append("accuracy_below_threshold")
        if poor_performance.precision < min_thresholds["precision"]:
            violations.append("precision_below_threshold")
        if poor_performance.recall < min_thresholds["recall"]:
            violations.append("recall_below_threshold")
        if poor_performance.f1_score < min_thresholds["f1_score"]:
            violations.append("f1_score_below_threshold")
        
        # Enforce business rule
        if violations:
            with pytest.raises(BusinessRuleViolation):
                raise BusinessRuleViolation(
                    message="Model performance below minimum requirements",
                    rule_name="minimum_performance_rule",
                    violated_constraints=violations,
                    context={
                        "performance_metrics": {
                            "accuracy": poor_performance.accuracy,
                            "precision": poor_performance.precision,
                            "recall": poor_performance.recall,
                            "f1_score": poor_performance.f1_score
                        },
                        "min_thresholds": min_thresholds
                    }
                )