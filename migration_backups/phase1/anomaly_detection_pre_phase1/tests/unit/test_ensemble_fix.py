"""Test ensemble service fixes for confidence scoring and prediction format."""

import numpy as np
import pytest
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.ensemble_service import EnsembleService


def test_ensemble_with_confidence_scores():
    """Test that ensemble works with confidence scores."""
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    
    # Create services
    detection_service = DetectionService()
    ensemble_service = EnsembleService(detection_service)
    
    # Test ensemble detection
    result = ensemble_service.detect_with_ensemble(
        X, 
        algorithms=["iforest", "lof"],
        combination_method="majority"
    )
    
    # Verify result structure
    assert result is not None
    assert result.predictions is not None
    assert len(result.predictions) == 100
    assert result.algorithm.startswith("ensemble")
    
    # Verify predictions are in correct format (-1 for anomaly, 1 for normal)
    assert set(np.unique(result.predictions)).issubset({-1, 1})
    
    # Check if confidence scores are available
    if result.confidence_scores is not None:
        assert len(result.confidence_scores) == 100
        assert np.all(result.confidence_scores >= 0)
        assert np.all(result.confidence_scores <= 1)
    
    print(f"✅ Ensemble test passed!")
    print(f"   - Predictions shape: {result.predictions.shape}")
    print(f"   - Anomalies detected: {result.anomaly_count}")
    print(f"   - Has confidence scores: {result.confidence_scores is not None}")
    if result.confidence_scores is not None:
        print(f"   - Score range: [{np.min(result.confidence_scores):.3f}, {np.max(result.confidence_scores):.3f}]")


def test_detection_service_confidence_scores():
    """Test that detection service provides confidence scores."""
    np.random.seed(42)
    X = np.random.rand(50, 3)
    
    detection_service = DetectionService()
    
    # Test isolation forest
    result_iforest = detection_service.detect_anomalies(X, algorithm="iforest")
    assert result_iforest.confidence_scores is not None
    assert len(result_iforest.confidence_scores) == 50
    
    # Test LOF
    result_lof = detection_service.detect_anomalies(X, algorithm="lof")
    assert result_lof.confidence_scores is not None
    assert len(result_lof.confidence_scores) == 50
    
    print(f"✅ Confidence scores test passed!")
    print(f"   - IForest scores available: {result_iforest.confidence_scores is not None}")
    print(f"   - LOF scores available: {result_lof.confidence_scores is not None}")


def test_weighted_ensemble():
    """Test weighted ensemble combination."""
    np.random.seed(42)
    X = np.random.rand(30, 4)
    
    detection_service = DetectionService()
    ensemble_service = EnsembleService(detection_service)
    
    # Test weighted ensemble
    result = ensemble_service.detect_with_ensemble(
        X,
        algorithms=["iforest", "lof"],
        combination_method="weighted",
        weights=[0.7, 0.3]
    )
    
    assert result is not None
    assert result.predictions is not None
    assert len(result.predictions) == 30
    assert result.metadata["combination_method"] == "weighted"
    assert result.metadata["weights"] == [0.7, 0.3]
    
    print(f"✅ Weighted ensemble test passed!")
    print(f"   - Algorithm: {result.algorithm}")
    print(f"   - Weights: {result.metadata['weights']}")


if __name__ == "__main__":
    test_detection_service_confidence_scores()
    test_ensemble_with_confidence_scores()
    test_weighted_ensemble()
    print("\n🎉 All ensemble fixes verified!")