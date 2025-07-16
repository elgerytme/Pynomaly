"""Comprehensive tests for Use Case Request/Response DTOs - Phase 1 Coverage Enhancement."""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from pynomaly.application.use_cases.detect_anomalies import (
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
)
from pynomaly.application.use_cases.evaluate_model import (
    EvaluateModelRequest,
    EvaluateModelResponse,
)
from pynomaly.application.use_cases.train_detector import (
    TrainDetectorRequest,
    TrainDetectorResponse,
)
from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import (
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
)


class TestDetectAnomaliesRequest:
    """Comprehensive tests for DetectAnomaliesRequest."""

    def test_detect_anomalies_request_creation(self):
        """Test creating a DetectAnomaliesRequest."""
        detector_id = uuid4()
        features = np.random.random((100, 5))
        dataset = Dataset(name="test_dataset", features=features)

        request = DetectAnomaliesRequest(
            detector_id=detector_id,
            dataset=dataset,
            validate_features=True,
            save_results=False,
        )

        assert request.detector_id == detector_id
        assert request.dataset == dataset
        assert request.validate_features is True
        assert request.save_results is False

    def test_detect_anomalies_request_defaults(self):
        """Test DetectAnomaliesRequest default values."""
        detector_id = uuid4()
        features = np.random.random((50, 3))
        dataset = Dataset(name="default_test", features=features)

        request = DetectAnomaliesRequest(detector_id=detector_id, dataset=dataset)

        assert request.validate_features is True  # Default
        assert request.save_results is True  # Default

    def test_detect_anomalies_request_with_targets(self):
        """Test DetectAnomaliesRequest with dataset containing targets."""
        detector_id = uuid4()
        features = np.random.random((100, 4))
        targets = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        dataset = Dataset(name="labeled_dataset", features=features, targets=targets)

        request = DetectAnomaliesRequest(
            detector_id=detector_id,
            dataset=dataset,
            validate_features=False,
            save_results=True,
        )

        assert request.dataset.targets is not None
        assert len(request.dataset.targets) == 100
        assert request.validate_features is False


class TestDetectAnomaliesResponse:
    """Comprehensive tests for DetectAnomaliesResponse."""

    def test_detect_anomalies_response_creation(self):
        """Test creating a comprehensive DetectAnomaliesResponse."""
        # Create domain objects
        detector = Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1),
        )

        features = np.random.RandomState(42).normal(0, 1, (200, 6))
        dataset = Dataset(name="response_test_dataset", features=features)

        scores = np.random.RandomState(42).beta(2, 8, 200)
        anomalies = [
            Anomaly(
                score=AnomalyScore(0.95),
                index=i,
                features=features[i],
                confidence=ConfidenceInterval(0.92, 0.98, 0.95),
            )
            for i in [10, 25, 87]
        ]

        result = DetectionResult(
            detector=detector,
            dataset=dataset,
            anomalies=anomalies,
            scores=scores,
            metadata={"processing_time": 2.34, "algorithm_version": "1.2"},
        )

        quality_report = {
            "data_quality_score": 0.87,
            "feature_stability": 0.92,
            "outlier_ratio": 0.15,
            "missing_values": 0,
        }

        warnings = [
            "High variance detected in feature 2",
            "Low data quality score: consider preprocessing",
        ]

        response = DetectAnomaliesResponse(
            result=result, quality_report=quality_report, warnings=warnings
        )

        assert response.result == result
        assert response.quality_report["data_quality_score"] == 0.87
        assert response.quality_report["feature_stability"] == 0.92
        assert len(response.warnings) == 2
        assert "High variance detected" in response.warnings[0]

    def test_detect_anomalies_response_minimal(self):
        """Test DetectAnomaliesResponse with minimal data."""
        detector = Detector(
            name="minimal_detector",
            algorithm="lof",
            contamination=ContaminationRate(0.05),
        )

        features = np.random.random((10, 2))
        dataset = Dataset(name="minimal_dataset", features=features)

        result = DetectionResult(
            detector=detector,
            dataset=dataset,
            anomalies=[],
            scores=np.random.random(10),
        )

        response = DetectAnomaliesResponse(result=result)

        assert response.result == result
        assert response.quality_report is None
        assert response.warnings is None

    def test_detect_anomalies_response_with_complex_quality_report(self):
        """Test response with comprehensive quality report."""
        detector = Detector(
            name="test", algorithm="test", contamination=ContaminationRate(0.1)
        )
        features = np.random.random((50, 5))
        dataset = Dataset(name="test", features=features)
        result = DetectionResult(
            detector=detector,
            dataset=dataset,
            anomalies=[],
            scores=np.random.random(50),
        )

        complex_quality_report = {
            "overall_score": 0.73,
            "feature_analysis": {
                "feature_0": {"quality": 0.85, "outlier_ratio": 0.02},
                "feature_1": {"quality": 0.91, "outlier_ratio": 0.01},
                "feature_2": {"quality": 0.45, "outlier_ratio": 0.15},
            },
            "correlation_matrix": {
                "max_correlation": 0.67,
                "highly_correlated_pairs": [("feature_0", "feature_1")],
            },
            "distribution_analysis": {
                "normality_tests": {
                    "feature_0": {"p_value": 0.023, "is_normal": False},
                    "feature_1": {"p_value": 0.456, "is_normal": True},
                }
            },
            "recommendations": [
                "Consider log transformation for feature_2",
                "Remove or combine highly correlated features",
            ],
        }

        response = DetectAnomaliesResponse(
            result=result, quality_report=complex_quality_report
        )

        assert response.quality_report["overall_score"] == 0.73
        assert (
            response.quality_report["feature_analysis"]["feature_2"]["quality"] == 0.45
        )
        assert len(response.quality_report["recommendations"]) == 2


class TestTrainDetectorRequest:
    """Comprehensive tests for TrainDetectorRequest."""

    def test_train_detector_request_comprehensive(self):
        """Test creating a comprehensive TrainDetectorRequest."""
        detector_id = uuid4()
        features = np.random.RandomState(42).normal(0, 1, (1000, 8))
        targets = np.random.RandomState(42).choice([0, 1], size=1000, p=[0.95, 0.05])
        training_data = Dataset(name="training_set", features=features, targets=targets)

        hyperparameter_grid = {
            "n_estimators": [100, 200, 300],
            "max_samples": [256, 512, "auto"],
            "contamination": [0.05, 0.1, 0.15],
        }

        request = TrainDetectorRequest(
            detector_id=detector_id,
            training_data=training_data,
            validation_split=0.2,
            hyperparameter_grid=hyperparameter_grid,
            cv_folds=5,
            scoring_metric="f1_score",
            save_model=True,
            early_stopping=True,
            max_training_time=300,  # 5 minutes
        )

        assert request.detector_id == detector_id
        assert request.training_data == training_data
        assert request.validation_split == 0.2
        assert request.hyperparameter_grid["n_estimators"] == [100, 200, 300]
        assert request.cv_folds == 5
        assert request.scoring_metric == "f1_score"
        assert request.save_model is True
        assert request.early_stopping is True
        assert request.max_training_time == 300

    def test_train_detector_request_defaults(self):
        """Test TrainDetectorRequest default values."""
        detector_id = uuid4()
        features = np.random.random((100, 3))
        training_data = Dataset(name="simple_training", features=features)

        request = TrainDetectorRequest(
            detector_id=detector_id, training_data=training_data
        )

        # Test that defaults are sensible
        assert request.validation_split is None or request.validation_split == 0.0
        assert request.hyperparameter_grid is None or request.hyperparameter_grid == {}
        assert request.cv_folds is None or request.cv_folds == 3
        assert request.scoring_metric is None or request.scoring_metric == "roc_auc"
        assert request.save_model is True
        assert request.early_stopping is False
        assert request.max_training_time is None


class TestTrainDetectorResponse:
    """Comprehensive tests for TrainDetectorResponse."""

    def test_train_detector_response_comprehensive(self):
        """Test creating a comprehensive TrainDetectorResponse."""
        # Create trained detector
        trained_detector = Detector(
            name="trained_fraud_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.08),
            hyperparameters={"n_estimators": 200, "max_samples": 512},
            metadata={"training_completed": True, "version": "2.0"},
        )
        trained_detector._is_fitted = True

        training_metrics = {
            "training_score": 0.92,
            "validation_score": 0.87,
            "best_parameters": {"n_estimators": 200, "max_samples": 512},
            "training_time": 45.7,
            "convergence_epoch": 150,
            "cross_validation_scores": [0.85, 0.89, 0.86, 0.88, 0.84],
            "feature_importance": {
                "feature_0": 0.23,
                "feature_1": 0.18,
                "feature_2": 0.15,
                "feature_3": 0.12,
                "feature_4": 0.10,
            },
        }

        model_path = "/models/fraud_detector_v2.pkl"

        training_warnings = [
            "Convergence achieved after 150 epochs",
            "Feature 7 has low importance (0.02)",
        ]

        response = TrainDetectorResponse(
            trained_detector=trained_detector,
            training_metrics=training_metrics,
            model_path=model_path,
            training_warnings=training_warnings,
        )

        assert response.trained_detector == trained_detector
        assert response.training_metrics["training_score"] == 0.92
        assert response.training_metrics["best_parameters"]["n_estimators"] == 200
        assert response.model_path == "/models/fraud_detector_v2.pkl"
        assert len(response.training_warnings) == 2
        assert "Convergence achieved" in response.training_warnings[0]

    def test_train_detector_response_minimal(self):
        """Test TrainDetectorResponse with minimal information."""
        detector = Detector(
            name="simple_detector",
            algorithm="lof",
            contamination=ContaminationRate(0.1),
        )

        response = TrainDetectorResponse(trained_detector=detector)

        assert response.trained_detector == detector
        assert response.training_metrics is None
        assert response.model_path is None
        assert response.training_warnings is None


class TestEvaluateModelRequest:
    """Comprehensive tests for EvaluateModelRequest."""

    def test_evaluate_model_request_comprehensive(self):
        """Test creating a comprehensive EvaluateModelRequest."""
        detector_id = uuid4()
        features = np.random.RandomState(42).normal(0, 1, (500, 6))
        targets = np.random.RandomState(42).choice([0, 1], size=500, p=[0.92, 0.08])
        test_data = Dataset(name="test_set", features=features, targets=targets)

        evaluation_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "average_precision",
            "matthews_corrcoef",
        ]

        request = EvaluateModelRequest(
            detector_id=detector_id,
            test_data=test_data,
            cross_validation=True,
            cv_folds=10,
            metrics=evaluation_metrics,
            return_predictions=True,
            return_probabilities=True,
            bootstrap_samples=1000,
            confidence_level=0.95,
        )

        assert request.detector_id == detector_id
        assert request.test_data == test_data
        assert request.cross_validation is True
        assert request.cv_folds == 10
        assert len(request.metrics) == 7
        assert "roc_auc" in request.metrics
        assert request.return_predictions is True
        assert request.return_probabilities is True
        assert request.bootstrap_samples == 1000
        assert request.confidence_level == 0.95


class TestEvaluateModelResponse:
    """Comprehensive tests for EvaluateModelResponse."""

    def test_evaluate_model_response_comprehensive(self):
        """Test creating a comprehensive EvaluateModelResponse."""
        evaluation_metrics = {
            "accuracy": 0.924,
            "precision": 0.876,
            "recall": 0.823,
            "f1_score": 0.849,
            "roc_auc": 0.945,
            "average_precision": 0.887,
            "matthews_corrcoef": 0.734,
            "confusion_matrix": [[460, 8], [7, 25]],
            "classification_report": {
                "class_0": {"precision": 0.98, "recall": 0.98, "f1-score": 0.98},
                "class_1": {"precision": 0.76, "recall": 0.78, "f1-score": 0.77},
            },
        }

        cross_validation_scores = {
            "cv_accuracy": [0.92, 0.91, 0.93, 0.90, 0.94, 0.89, 0.92, 0.93, 0.91, 0.90],
            "cv_f1": [0.85, 0.84, 0.86, 0.83, 0.87, 0.82, 0.85, 0.86, 0.84, 0.83],
            "cv_mean": {"accuracy": 0.915, "f1": 0.845},
            "cv_std": {"accuracy": 0.015, "f1": 0.016},
        }

        predictions = np.array([0] * 468 + [1] * 32)
        prediction_probabilities = np.random.beta(2, 8, 500)  # Skewed towards 0

        feature_importance = {
            "feature_0": 0.28,
            "feature_1": 0.22,
            "feature_2": 0.18,
            "feature_3": 0.15,
            "feature_4": 0.10,
            "feature_5": 0.07,
        }

        confidence_intervals = {
            "accuracy": (0.905, 0.943),
            "precision": (0.847, 0.905),
            "recall": (0.789, 0.857),
            "f1_score": (0.820, 0.878),
        }

        response = EvaluateModelResponse(
            evaluation_metrics=evaluation_metrics,
            cross_validation_scores=cross_validation_scores,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            feature_importance=feature_importance,
            confidence_intervals=confidence_intervals,
        )

        assert response.evaluation_metrics["accuracy"] == 0.924
        assert response.evaluation_metrics["roc_auc"] == 0.945
        assert response.cross_validation_scores["cv_mean"]["accuracy"] == 0.915
        assert len(response.predictions) == 500
        assert len(response.prediction_probabilities) == 500
        assert response.feature_importance["feature_0"] == 0.28
        assert response.confidence_intervals["accuracy"] == (0.905, 0.943)

    def test_evaluate_model_response_without_ground_truth(self):
        """Test evaluation response when no ground truth is available."""
        # Metrics available without ground truth
        unsupervised_metrics = {
            "anomaly_score_stats": {
                "mean": 0.15,
                "std": 0.23,
                "min": 0.01,
                "max": 0.98,
                "percentiles": {"25th": 0.05, "50th": 0.12, "75th": 0.21, "95th": 0.67},
            },
            "contamination_estimate": 0.08,
            "cluster_analysis": {"silhouette_score": 0.34, "calinski_harabasz": 156.7},
        }

        predictions = np.array([0] * 460 + [1] * 40)
        prediction_scores = np.random.beta(2, 8, 500)

        response = EvaluateModelResponse(
            evaluation_metrics=unsupervised_metrics,
            predictions=predictions,
            prediction_probabilities=prediction_scores,
        )

        assert response.evaluation_metrics["contamination_estimate"] == 0.08
        assert response.evaluation_metrics["anomaly_score_stats"]["mean"] == 0.15
        assert response.cross_validation_scores is None
        assert response.confidence_intervals is None


class TestUseCaseIntegration:
    """Integration tests for use case request/response flow."""

    def test_detection_request_response_flow(self):
        """Test complete detection request-response flow."""
        # Create request
        detector_id = uuid4()
        features = np.random.random((100, 4))
        dataset = Dataset(name="integration_test", features=features)

        request = DetectAnomaliesRequest(
            detector_id=detector_id,
            dataset=dataset,
            validate_features=True,
            save_results=False,
        )

        # Simulate processing and create response
        detector = Detector(
            name="integration_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1),
        )

        result = DetectionResult(
            detector=detector,
            dataset=dataset,
            anomalies=[],
            scores=np.random.random(100),
        )

        response = DetectAnomaliesResponse(
            result=result, warnings=["Integration test completed"]
        )

        # Verify flow
        assert request.detector_id == detector_id
        assert request.dataset.name == "integration_test"
        assert response.result.dataset.name == "integration_test"
        assert len(response.warnings) == 1

    def test_training_request_response_flow(self):
        """Test complete training request-response flow."""
        detector_id = uuid4()
        features = np.random.random((200, 5))
        training_data = Dataset(name="training_integration", features=features)

        request = TrainDetectorRequest(
            detector_id=detector_id, training_data=training_data, validation_split=0.2
        )

        # Simulate training and create response
        trained_detector = Detector(
            name="trained_integration_detector",
            algorithm="lof",
            contamination=ContaminationRate(0.1),
        )

        response = TrainDetectorResponse(
            trained_detector=trained_detector,
            training_metrics={"training_score": 0.89},
            training_warnings=["Training completed successfully"],
        )

        # Verify flow
        assert request.detector_id == detector_id
        assert request.training_data.name == "training_integration"
        assert response.trained_detector.name == "trained_integration_detector"
        assert response.training_metrics["training_score"] == 0.89
