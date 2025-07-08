#!/usr/bin/env python3
"""Simple test for AnomalyClassificationService without DI container."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.application.services.anomaly_classification_service import AnomalyClassificationService
from pynomaly.domain.services.anomaly_classifiers import (
    DefaultSeverityClassifier,
    DefaultTypeClassifier,
    BatchProcessingSeverityClassifier,
    DashboardTypeClassifier
)

def test_basic_classification():
    """Test basic classification functionality."""
    print("Testing basic classification...")
    
    # Create service with default classifiers
    service = AnomalyClassificationService()
    
    # Create test anomaly
    score = AnomalyScore(value=0.85)
    anomaly = Anomaly(score=score, data_point={'feature1': 1.0}, detector_name="test_detector")
    
    # Classify
    service.classify(anomaly)
    
    # Check results
    assert 'severity' in anomaly.metadata
    assert 'type' in anomaly.metadata
    assert anomaly.metadata['severity'] == 'high'
    assert anomaly.metadata['type'] == 'point'
    
    print("✓ Basic classification test passed")

def test_dependency_injection():
    """Test dependency injection functionality."""
    print("Testing dependency injection...")
    
    # Create custom classifier with different thresholds
    custom_severity_classifier = DefaultSeverityClassifier({
        'critical': 0.8,
        'high': 0.6,
        'medium': 0.4,
        'low': 0.0
    })
    
    service = AnomalyClassificationService()
    service.set_severity_classifier(custom_severity_classifier)
    
    # Test that 0.75 is now classified as critical
    score = AnomalyScore(value=0.75)
    anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")
    
    service.classify(anomaly)
    
    severity = anomaly.metadata['severity']
    assert severity == 'critical', f"Expected 'critical', got '{severity}'"
    
    print("✓ Dependency injection test passed")

def test_batch_processing():
    """Test batch processing classifiers."""
    print("Testing batch processing classifiers...")
    
    service = AnomalyClassificationService()
    service.use_batch_processing_classifiers()
    
    # Process multiple anomalies
    anomalies = []
    for i in range(3):
        score = AnomalyScore(value=0.85)
        anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")
        service.classify(anomaly)
        anomalies.append(anomaly)
    
    # All should be classified
    for anomaly in anomalies:
        assert 'severity' in anomaly.metadata
        assert 'type' in anomaly.metadata
    
    # Clear cache
    service.clear_classifier_cache()
    
    print("✓ Batch processing test passed")

def test_dashboard_classifiers():
    """Test dashboard-friendly classifiers."""
    print("Testing dashboard classifiers...")
    
    service = AnomalyClassificationService()
    service.use_dashboard_classifiers()
    
    score = AnomalyScore(value=0.85)
    anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")
    
    service.classify(anomaly)
    
    # Dashboard classifiers should provide friendly names
    anomaly_type = anomaly.metadata['type']
    assert 'Anomaly' in anomaly_type, f"Expected dashboard-friendly type, got '{anomaly_type}'"
    
    print("✓ Dashboard classifiers test passed")

def test_severity_levels():
    """Test different severity levels."""
    print("Testing severity levels...")
    
    service = AnomalyClassificationService()
    
    test_cases = [
        (0.95, 'critical'),
        (0.75, 'high'),
        (0.55, 'medium'),
        (0.25, 'low')
    ]
    
    for score_value, expected_severity in test_cases:
        score = AnomalyScore(value=score_value)
        anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")
        service.classify(anomaly)
        
        actual_severity = anomaly.metadata['severity']
        assert actual_severity == expected_severity, f"Score {score_value}: expected '{expected_severity}', got '{actual_severity}'"
    
    print("✓ Severity levels test passed")

def test_type_classification():
    """Test type classification heuristics."""
    print("Testing type classification...")
    
    service = AnomalyClassificationService()
    
    # Test point anomaly (few features)
    score = AnomalyScore(value=0.85)
    anomaly = Anomaly(score=score, data_point={'feature1': 1.0}, detector_name="test_detector")
    service.classify(anomaly)
    assert anomaly.metadata['type'] == 'point'
    
    # Test collective anomaly (many features)
    score = AnomalyScore(value=0.85)
    anomaly = Anomaly(
        score=score, 
        data_point={'f1': 1.0, 'f2': 2.0, 'f3': 3.0, 'f4': 4.0, 'f5': 5.0}, 
        detector_name="test_detector"
    )
    service.classify(anomaly)
    assert anomaly.metadata['type'] == 'collective'
    
    # Test contextual anomaly (with temporal context)
    score = AnomalyScore(value=0.85)
    anomaly = Anomaly(score=score, data_point={'feature1': 1.0}, detector_name="test_detector")
    anomaly.add_metadata('temporal_context', True)
    service.classify(anomaly)
    assert anomaly.metadata['type'] == 'contextual'
    
    print("✓ Type classification test passed")

if __name__ == "__main__":
    print("Running AnomalyClassificationService tests...\n")
    
    try:
        test_basic_classification()
        test_dependency_injection()
        test_batch_processing()
        test_dashboard_classifiers()
        test_severity_levels()
        test_type_classification()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
