"""Service for anomaly classification using severity and type classifiers."""

from __future__ import annotations

from typing import Optional

from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.services.anomaly_classifiers import (
    SeverityClassifier,
    TypeClassifier,
    DefaultSeverityClassifier,
    DefaultTypeClassifier,
    MLSeverityClassifier,
    BatchProcessingSeverityClassifier,
    DashboardTypeClassifier
)


class AnomalyClassificationService:
    """Service to run classifiers on anomalies and update their attributes."""
    
    def __init__(
        self,
        severity_classifier: Optional[SeverityClassifier] = None,
        type_classifier: Optional[TypeClassifier] = None
    ):
        """Initialize the service with the provided classifiers.
        
        Args:
            severity_classifier: Classifier for anomaly severity
            type_classifier: Classifier for anomaly type
        """
        self.severity_classifier = severity_classifier or DefaultSeverityClassifier()
        self.type_classifier = type_classifier or DefaultTypeClassifier()
    
    def classify(self, anomaly: Anomaly) -> None:
        """Classify anomaly using severity and type classifiers.
        
        Args:
            anomaly: The anomaly to classify
        
        The function updates the anomaly's metadata with the classification results.
        """
        severity = self.severity_classifier.classify_severity(anomaly)
        anomaly.add_metadata('severity', severity)
        
        anomaly_type = self.type_classifier.classify_type(anomaly)
        anomaly.add_metadata('type', anomaly_type)

    def set_severity_classifier(self, classifier: SeverityClassifier) -> None:
        """Set a custom severity classifier.
        
        Args:
            classifier: The severity classifier to inject
        """
        self.severity_classifier = classifier
    
    def set_type_classifier(self, classifier: TypeClassifier) -> None:
        """Set a custom type classifier.
        
        Args:
            classifier: The type classifier to inject
        """
        self.type_classifier = classifier

    def use_batch_processing_classifiers(self) -> None:
        """Switch to classifiers optimized for batch processing."""
        self.severity_classifier = BatchProcessingSeverityClassifier(self.severity_classifier)
        self.type_classifier = DashboardTypeClassifier(self.type_classifier)

    def use_dashboard_classifiers(self) -> None:
        """Switch to classifiers optimized for dashboard display."""
        self.severity_classifier = BatchProcessingSeverityClassifier(self.severity_classifier)
        self.type_classifier = DashboardTypeClassifier(self.type_classifier)
        
    def clear_classifier_cache(self) -> None:
        """Clear cache for batch processing classifiers if in use."""
        if isinstance(self.severity_classifier, BatchProcessingSeverityClassifier):
            self.severity_classifier.clear_cache()

