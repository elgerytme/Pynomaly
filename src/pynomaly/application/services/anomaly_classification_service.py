"""Service for anomaly classification using severity and type classifiers."""

from __future__ import annotations

from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.services.anomaly_classifiers import (
    BatchProcessingSeverityClassifier,
    DashboardTypeClassifier,
    DefaultSeverityClassifier,
    DefaultTypeClassifier,
    SeverityClassifier,
    TypeClassifier,
)


class AnomalyClassificationService:
    """Service to run classifiers on anomalies and update their attributes.
    
    The AnomalyClassificationService implements a two-dimensional anomaly classification
    taxonomy that categorizes detected anomalies by severity and type. This provides
    standardized, actionable insights for anomaly triage and response.
    
    Severity Classification:
        - Critical (≥0.9): Immediate attention required, high business impact
        - High (≥0.7): Significant deviation, requires prompt investigation  
        - Medium (≥0.5): Moderate anomaly, should be reviewed
        - Low (≥0.0): Minor deviation, informational
    
    Type Classification:
        - Point Anomaly: Individual data points that deviate from normal patterns
        - Collective Anomaly: Groups of data points forming anomalous patterns
        - Contextual Anomaly: Points anomalous in specific contexts/time periods
    
    The service uses a protocol-based design enabling easy extensibility with
    custom classifiers for domain-specific requirements.
    
    Examples:
        Basic usage with default classifiers:
        >>> service = AnomalyClassificationService()
        >>> service.classify(anomaly)
        >>> severity = anomaly.metadata.get('severity')
        >>> type_category = anomaly.metadata.get('type')
        
        Custom classifier injection:
        >>> custom_classifier = CustomSeverityClassifier()
        >>> service = AnomalyClassificationService(
        ...     severity_classifier=custom_classifier
        ... )
        
        Batch processing optimization:
        >>> service.use_batch_processing_classifiers()
        >>> for anomaly in large_batch:
        ...     service.classify(anomaly)
        >>> service.clear_classifier_cache()
    
    Attributes:
        severity_classifier: The classifier used for severity assessment
        type_classifier: The classifier used for type categorization
    """

    def __init__(
        self,
        severity_classifier: SeverityClassifier | None = None,
        type_classifier: TypeClassifier | None = None,
    ):
        """Initialize the service with the provided classifiers.

        Args:
            severity_classifier: Classifier for anomaly severity assessment.
                If None, uses DefaultSeverityClassifier with standard thresholds.
            type_classifier: Classifier for anomaly type categorization.
                If None, uses DefaultTypeClassifier with heuristic-based detection.
                
        Note:
            The service automatically initializes with sensible defaults if no
            classifiers are provided, making it immediately usable for most
            common anomaly detection scenarios.
        """
        self.severity_classifier = severity_classifier or DefaultSeverityClassifier()
        self.type_classifier = type_classifier or DefaultTypeClassifier()

    def classify(self, anomaly: Anomaly) -> None:
        """Classify anomaly using severity and type classifiers.

        This method applies both severity and type classification to the provided
        anomaly, updating its metadata with the results. The classification is
        performed in-place, modifying the anomaly object directly.

        Args:
            anomaly: The anomaly entity to classify. Must have a valid score
                and data_point attribute.

        Raises:
            AttributeError: If the anomaly lacks required attributes (score, data_point)
            ValueError: If the anomaly score is not in a valid format
            
        Side Effects:
            Updates anomaly.metadata with:
            - 'severity': String indicating severity level ('critical', 'high', 'medium', 'low')
            - 'type': String indicating anomaly type ('point', 'collective', 'contextual')
            
        Examples:
            Basic classification:
            >>> service = AnomalyClassificationService()
            >>> service.classify(anomaly)
            >>> print(f"Severity: {anomaly.metadata['severity']}")
            >>> print(f"Type: {anomaly.metadata['type']}")
            
            Classification with custom thresholds:
            >>> custom_classifier = DefaultSeverityClassifier({
            ...     'critical': 0.95, 'high': 0.8, 'medium': 0.6, 'low': 0.0
            ... })
            >>> service = AnomalyClassificationService(severity_classifier=custom_classifier)
            >>> service.classify(anomaly)
        """
        severity = self.severity_classifier.classify_severity(anomaly)
        anomaly.add_metadata("severity", severity)

        anomaly_type = self.type_classifier.classify_type(anomaly)
        anomaly.add_metadata("type", anomaly_type)

    def set_severity_classifier(self, classifier: SeverityClassifier) -> None:
        """Set a custom severity classifier.

        This method allows for the injection of a custom severity classifier to override
        the default behavior. This is essential for implementing domain-specific
        classification logic or integrating machine learning models.

        Args:
            classifier: The severity classifier to inject. Must implement the
                SeverityClassifier Protocol.

        Raises:
            TypeError: If the classifier does not fulfill the required Protocol.

        Examples:
            Using a custom ML classifier:
            >>> ml_classifier = CustomMLSeverityClassifier(model)
            >>> service.set_severity_classifier(ml_classifier)
            
            Reverting to default classification:
            >>> service.set_severity_classifier(DefaultSeverityClassifier())
        """
        self.severity_classifier = classifier

    def set_type_classifier(self, classifier: TypeClassifier) -> None:
        """Set a custom type classifier.

        This method allows for the injection of a custom type classifier to tailor the
        classification process to specific anomaly characteristics, useful for handling
        complex or domain-specific patterns.

        Args:
            classifier: The type classifier to inject. Must implement the TypeClassifier
                Protocol.

        Raises:
            TypeError: If the classifier does not fulfill the required Protocol.

        Examples:
            Setting a domain-specific type classifier:
            >>> custom_type_classifier = IoTTypeClassifier()
            >>> service.set_type_classifier(custom_type_classifier)
            
            Using a dashboard-friendly classifier:
            >>> service.set_type_classifier(DashboardTypeClassifier())
        """
        self.type_classifier = classifier

    def use_batch_processing_classifiers(self) -> None:
        """Switch to classifiers optimized for batch processing.
        
        This method wraps the current classifiers with batch-optimized versions that
        provide caching and performance improvements for high-throughput scenarios.
        The BatchProcessingSeverityClassifier caches results for similar anomalies,
        reducing redundant computation in bulk processing workflows.
        
        Side Effects:
            - Wraps current severity_classifier with BatchProcessingSeverityClassifier
            - Wraps current type_classifier with DashboardTypeClassifier
            
        Examples:
            Enable batch processing before processing large datasets:
            >>> service = AnomalyClassificationService()
            >>> service.use_batch_processing_classifiers()
            >>> 
            >>> # Process large batch efficiently
            >>> for anomaly in large_anomaly_batch:
            ...     service.classify(anomaly)
            >>> 
            >>> # Clear cache to free memory
            >>> service.clear_classifier_cache()
        """
        self.severity_classifier = BatchProcessingSeverityClassifier(
            self.severity_classifier
        )
        self.type_classifier = DashboardTypeClassifier(self.type_classifier)

    def use_dashboard_classifiers(self) -> None:
        """Switch to classifiers optimized for dashboard display.
        
        This method configures the service to use classifiers that produce
        human-readable, dashboard-friendly output. The DashboardTypeClassifier
        converts technical type names to more user-friendly labels.
        
        Side Effects:
            - Wraps current severity_classifier with BatchProcessingSeverityClassifier
            - Wraps current type_classifier with DashboardTypeClassifier
            
        Examples:
            Enable dashboard-friendly output:
            >>> service = AnomalyClassificationService()
            >>> service.use_dashboard_classifiers()
            >>> service.classify(anomaly)
            >>> 
            >>> # Returns user-friendly type names:
            >>> # 'Point Anomaly' instead of 'point'
            >>> # 'Pattern Anomaly' instead of 'collective'
            >>> # 'Context Anomaly' instead of 'contextual'
            >>> type_display = anomaly.metadata['type']
        """
        self.severity_classifier = BatchProcessingSeverityClassifier(
            self.severity_classifier
        )
        self.type_classifier = DashboardTypeClassifier(self.type_classifier)

    def clear_classifier_cache(self) -> None:
        """Clear cache for batch processing classifiers if in use.
        
        This method clears the internal cache of BatchProcessingSeverityClassifier
        if it is currently in use. This is important for memory management during
        long-running batch processing operations.
        
        Side Effects:
            - Clears the batch cache if BatchProcessingSeverityClassifier is active
            - No effect if batch processing is not enabled
            
        Examples:
            Clear cache after batch processing:
            >>> service = AnomalyClassificationService()
            >>> service.use_batch_processing_classifiers()
            >>> 
            >>> # Process large batch
            >>> for anomaly in anomaly_batch:
            ...     service.classify(anomaly)
            >>> 
            >>> # Clear cache to free memory
            >>> service.clear_classifier_cache()
        """
        if isinstance(self.severity_classifier, BatchProcessingSeverityClassifier):
            self.severity_classifier.clear_cache()
