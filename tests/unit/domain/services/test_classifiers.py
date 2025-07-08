import unittest
from datetime import datetime, timedelta
from pynomaly.domain.services import (
    ThresholdSeverityClassifier,
    StatisticalSeverityClassifier,
    RuleBasedTypeClassifier
)
from pynomaly.domain.value_objects import AnomalyScore


class TestThresholdSeverityClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = ThresholdSeverityClassifier()

    def test_classify_single(self):
        self.assertEqual(self.classifier.classify_single(0.2), "low")
        self.assertEqual(self.classifier.classify_single(0.5), "medium")
        self.assertEqual(self.classifier.classify_single(0.8), "high")
        self.assertEqual(self.classifier.classify_single(0.95), "critical")

    def test_update_threshold(self):
        self.classifier.update_threshold("low", 0.0, 0.2)
        self.assertEqual(self.classifier.classify_single(0.15), "low")


class TestStatisticalSeverityClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = StatisticalSeverityClassifier()

    def test_classify_batch(self):
        scores = [0.5, 1.5, 2.5, 3.5]
        self.assertEqual(
            self.classifier.classify_batch(scores), ["low", "medium", "high", "critical"]
        )


class TestRuleBasedTypeClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = RuleBasedTypeClassifier()

    def test_classify_single(self):
        context_spike = {"previous_scores": [0.1, 0.1, 0.1, 0.1, 0.1]}
        self.assertEqual(self.classifier.classify_single(0.9, context=context_spike), "spike")

        context_drift = {"previous_scores": [0.1, 0.2, 0.3, 0.4, 0.5]}
        self.assertEqual(self.classifier.classify_single(0.6, context=context_drift), "drift")

        context_cyclic = {
            "historical_anomaly_times": [
                datetime.now() - timedelta(days=i) for i in range(1, 8)
            ]
        }
        self.assertEqual(self.classifier.classify_single(0.5, context=context_cyclic), "seasonal")

        context_outlier = {"nearby_scores": [0.1, 0.1, 0.1]}
        self.assertEqual(self.classifier.classify_single(0.9, context=context_outlier), "outlier")


if __name__ == "__main__":
    unittest.main()

