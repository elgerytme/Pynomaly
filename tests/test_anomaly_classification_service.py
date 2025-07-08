"""Test the AnomalyClassificationService."""

import unittest
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.infrastructure.config.container import create_container


class TestAnomalyClassificationService(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test container and services."""
        self.container = create_container(testing=True)
        self.classification_service = self.container.anomaly_classification_service

    def test_classify_anomaly(self):
        """Test the classification of an anomaly."""
        score = AnomalyScore(value=0.85)
        anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")

        self.classification_service().classify(anomaly)

        self.assertIn('severity', anomaly.metadata)
        self.assertIn('type', anomaly.metadata)

        severity = anomaly.metadata['severity']
        anomaly_type = anomaly.metadata['type']

        self.assertEqual(severity, 'high')
        self.assertEqual(anomaly_type, 'point')

    def test_use_batch_processing_classifiers(self):
        """Test switching to batch processing classifiers."""
        self.classification_service().use_batch_processing_classifiers()
        score = AnomalyScore(value=0.85)
        anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")

        self.classification_service().classify(anomaly)

        self.assertIn('severity', anomaly.metadata)
        self.assertIn('type', anomaly.metadata)

    def test_use_dashboard_classifiers(self):
        """Test switching to dashboard classifiers."""
        self.classification_service().use_dashboard_classifiers()
        score = AnomalyScore(value=0.85)
        anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")

        self.classification_service().classify(anomaly)

        self.assertIn('severity', anomaly.metadata)
        self.assertIn('type', anomaly.metadata)
        
        # Dashboard classifiers should provide friendly names
        anomaly_type = anomaly.metadata['type']
        self.assertIn('Anomaly', anomaly_type)  # Should be dashboard-friendly

    def test_dependency_injection(self):
        """Test that DI works correctly with custom classifiers."""
        from pynomaly.domain.services.anomaly_classifiers import (
            DefaultSeverityClassifier,
            DefaultTypeClassifier
        )
        
        # Test that we can inject custom classifiers
        custom_severity_classifier = DefaultSeverityClassifier({
            'critical': 0.8,
            'high': 0.6,
            'medium': 0.4,
            'low': 0.0
        })
        
        service = self.classification_service()
        service.set_severity_classifier(custom_severity_classifier)
        
        score = AnomalyScore(value=0.75)
        anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")
        
        service.classify(anomaly)
        
        severity = anomaly.metadata['severity']
        self.assertEqual(severity, 'critical')  # Should use custom thresholds

    def test_severity_classification_levels(self):
        """Test different severity classification levels."""
        service = self.classification_service()
        
        # Test critical severity
        score = AnomalyScore(value=0.95)
        anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")
        service.classify(anomaly)
        self.assertEqual(anomaly.metadata['severity'], 'critical')
        
        # Test high severity
        score = AnomalyScore(value=0.75)
        anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")
        service.classify(anomaly)
        self.assertEqual(anomaly.metadata['severity'], 'high')
        
        # Test medium severity
        score = AnomalyScore(value=0.55)
        anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")
        service.classify(anomaly)
        self.assertEqual(anomaly.metadata['severity'], 'medium')
        
        # Test low severity
        score = AnomalyScore(value=0.25)
        anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")
        service.classify(anomaly)
        self.assertEqual(anomaly.metadata['severity'], 'low')

    def test_type_classification_heuristics(self):
        """Test type classification heuristics."""
        service = self.classification_service()
        
        # Test point anomaly (few features)
        score = AnomalyScore(value=0.85)
        anomaly = Anomaly(score=score, data_point={'feature1': 1.0}, detector_name="test_detector")
        service.classify(anomaly)
        self.assertEqual(anomaly.metadata['type'], 'point')
        
        # Test collective anomaly (many features)
        score = AnomalyScore(value=0.85)
        anomaly = Anomaly(
            score=score, 
            data_point={'f1': 1.0, 'f2': 2.0, 'f3': 3.0, 'f4': 4.0, 'f5': 5.0}, 
            detector_name="test_detector"
        )
        service.classify(anomaly)
        self.assertEqual(anomaly.metadata['type'], 'collective')
        
        # Test contextual anomaly (with temporal context)
        score = AnomalyScore(value=0.85)
        anomaly = Anomaly(score=score, data_point={'feature1': 1.0}, detector_name="test_detector")
        anomaly.add_metadata('temporal_context', True)
        service.classify(anomaly)
        self.assertEqual(anomaly.metadata['type'], 'contextual')

    def test_clear_cache(self):
        """Test cache clearing for batch processing classifiers."""
        service = self.classification_service()
        service.use_batch_processing_classifiers()
        
        # Process some anomalies to populate cache
        for i in range(3):
            score = AnomalyScore(value=0.85)
            anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")
            service.classify(anomaly)
        
        # Clear cache should not raise an error
        service.clear_classifier_cache()
        
        # Should still work after cache clear
        score = AnomalyScore(value=0.85)
        anomaly = Anomaly(score=score, data_point={}, detector_name="test_detector")
        service.classify(anomaly)
        
        self.assertIn('severity', anomaly.metadata)
        self.assertIn('type', anomaly.metadata)


if __name__ == "__main__":
    unittest.main()
