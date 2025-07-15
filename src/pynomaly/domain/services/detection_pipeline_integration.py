"""Detection pipeline integration with advanced classification."""

from __future__ import annotations

import logging
from typing import Any

from pynomaly.domain.entities.anomaly_event import AnomalyEvent, AnomalyEventData, EventSeverity, EventType
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.services.advanced_classification_service import AdvancedClassificationService
from pynomaly.domain.value_objects.anomaly_classification import AdvancedAnomalyClassification

logger = logging.getLogger(__name__)


class DetectionPipelineIntegration:
    """Integration layer for detection pipeline with advanced classification.
    
    This service integrates the advanced classification system with the existing
    detection pipeline, creating enriched anomaly events and providing seamless
    integration with the event processing system.
    """

    def __init__(self, classification_service: AdvancedClassificationService):
        """Initialize the detection pipeline integration.
        
        Args:
            classification_service: Advanced classification service instance
        """
        self.classification_service = classification_service

    def process_detection_result(
        self,
        anomaly_score: float,
        detector: Detector,
        raw_data: dict[str, Any],
        feature_data: dict[str, Any] | None = None,
        context_data: dict[str, Any] | None = None,
    ) -> tuple[AdvancedAnomalyClassification, AnomalyEvent]:
        """Process a detection result through advanced classification.
        
        Args:
            anomaly_score: Raw anomaly score from detector
            detector: Detector that produced the score
            raw_data: Raw input data that was analyzed
            feature_data: Feature-level analysis data
            context_data: Additional context information
            
        Returns:
            Tuple of (advanced classification, anomaly event)
        """
        # Perform advanced classification
        classification = self.classification_service.classify_anomaly(
            anomaly_score=anomaly_score,
            detector=detector,
            feature_data=feature_data,
            context_data=context_data,
        )
        
        # Create enriched anomaly event
        event = self._create_anomaly_event(
            classification=classification,
            detector=detector,
            raw_data=raw_data,
            context_data=context_data or {},
        )
        
        return classification, event

    def process_batch_detection_results(
        self,
        anomaly_scores: list[float],
        detector: Detector,
        raw_data_batch: list[dict[str, Any]],
        feature_data_batch: list[dict[str, Any]] | None = None,
        context_data_batch: list[dict[str, Any]] | None = None,
    ) -> tuple[list[AdvancedAnomalyClassification], list[AnomalyEvent]]:
        """Process a batch of detection results through advanced classification.
        
        Args:
            anomaly_scores: List of anomaly scores
            detector: Detector that produced the scores
            raw_data_batch: List of raw input data
            feature_data_batch: List of feature-level analysis data
            context_data_batch: List of additional context information
            
        Returns:
            Tuple of (classifications list, events list)
        """
        if not anomaly_scores:
            return [], []
        
        # Perform batch classification
        classifications = self.classification_service.classify_batch(
            anomaly_scores=anomaly_scores,
            detector=detector,
            feature_data_batch=feature_data_batch,
            context_data_batch=context_data_batch,
        )
        
        # Create batch of enriched events
        events = []
        for i, classification in enumerate(classifications):
            raw_data = raw_data_batch[i] if i < len(raw_data_batch) else {}
            context_data = (
                context_data_batch[i] if context_data_batch and i < len(context_data_batch) 
                else {}
            )
            
            event = self._create_anomaly_event(
                classification=classification,
                detector=detector,
                raw_data=raw_data,
                context_data=context_data,
            )
            events.append(event)
        
        return classifications, events

    def _create_anomaly_event(
        self,
        classification: AdvancedAnomalyClassification,
        detector: Detector,
        raw_data: dict[str, Any],
        context_data: dict[str, Any],
    ) -> AnomalyEvent:
        """Create an enriched anomaly event from classification result."""
        # Determine event type based on classification
        event_type = self._determine_event_type(classification)
        
        # Map severity classification to event severity
        event_severity = self._map_to_event_severity(classification.severity_classification)
        
        # Create anomaly event data
        anomaly_data = AnomalyEventData(
            anomaly_score=classification.get_confidence_score(),
            confidence=classification.get_confidence_score(),
            feature_contributions=classification.basic_classification.feature_contributions,
            predicted_class=classification.get_primary_class(),
            detection_method=detector.algorithm_name,
            model_version=detector.metadata.get("model_version"),
            explanation=self._generate_explanation(classification),
        )
        
        # Create comprehensive title and description
        title = self._generate_event_title(classification, detector)
        description = self._generate_event_description(classification, detector)
        
        # Create the event
        event = AnomalyEvent(
            event_type=event_type,
            severity=event_severity,
            title=title,
            description=description,
            raw_data=raw_data,
            event_time=context_data.get("timestamp", ""),
            detector_id=detector.id,
            anomaly_data=anomaly_data,
            business_context=context_data.get("business_context", {}),
            technical_context=self._create_technical_context(classification, detector),
        )
        
        # Add classification-specific tags
        self._add_classification_tags(event, classification)
        
        # Add classification metadata
        event.metadata.update({
            "advanced_classification": classification.get_full_classification_summary(),
            "classification_service_version": "1.0.0",
        })
        
        return event

    def _determine_event_type(self, classification: AdvancedAnomalyClassification) -> EventType:
        """Determine event type based on classification."""
        if classification.get_primary_class() == "normal":
            return EventType.CUSTOM
        
        if classification.severity_classification == "critical":
            return EventType.ANOMALY_ESCALATED
        else:
            return EventType.ANOMALY_DETECTED

    def _map_to_event_severity(self, severity_classification: str) -> EventSeverity:
        """Map classification severity to event severity."""
        severity_mapping = {
            "low": EventSeverity.LOW,
            "medium": EventSeverity.MEDIUM,
            "high": EventSeverity.HIGH,
            "critical": EventSeverity.CRITICAL,
        }
        return severity_mapping.get(severity_classification, EventSeverity.MEDIUM)

    def _generate_explanation(self, classification: AdvancedAnomalyClassification) -> str:
        """Generate human-readable explanation of the classification."""
        explanation_parts = []
        
        # Basic classification explanation
        confidence_level = classification.basic_classification.confidence_level.value
        explanation_parts.append(
            f"Classified as {classification.get_primary_class()} with {confidence_level} confidence"
        )
        
        # Severity explanation
        explanation_parts.append(f"Severity level: {classification.severity_classification}")
        
        # Hierarchical explanation
        if classification.is_hierarchical():
            hierarchy_path = classification.hierarchical_classification.get_full_path()
            explanation_parts.append(f"Hierarchical classification: {hierarchy_path}")
        
        # Multi-class explanation
        if classification.is_multi_class() and classification.multi_class_classification.has_ambiguous_classification():
            explanation_parts.append("Multiple possible classifications detected (ambiguous)")
        
        # Feature contributions
        if classification.basic_classification.feature_contributions:
            top_features = sorted(
                classification.basic_classification.feature_contributions.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            feature_names = [f[0] for f in top_features]
            explanation_parts.append(f"Top contributing features: {', '.join(feature_names)}")
        
        return ". ".join(explanation_parts)

    def _generate_event_title(
        self, classification: AdvancedAnomalyClassification, detector: Detector
    ) -> str:
        """Generate event title based on classification."""
        severity = classification.severity_classification.upper()
        detector_name = detector.name
        
        if classification.is_hierarchical():
            category = classification.hierarchical_classification.primary_category
            return f"{severity} {category} anomaly detected by {detector_name}"
        else:
            return f"{severity} anomaly detected by {detector_name}"

    def _generate_event_description(
        self, classification: AdvancedAnomalyClassification, detector: Detector
    ) -> str:
        """Generate detailed event description."""
        description_parts = []
        
        # Basic information
        description_parts.append(
            f"Anomaly detected using {detector.algorithm_name} algorithm "
            f"with confidence score {classification.get_confidence_score():.3f}"
        )
        
        # Classification details
        if classification.is_hierarchical():
            hierarchy_info = classification.hierarchical_classification.get_full_path()
            description_parts.append(f"Classification hierarchy: {hierarchy_info}")
        
        # Context information
        if classification.has_temporal_context():
            description_parts.append("Temporal context analysis included")
        
        if classification.has_spatial_context():
            description_parts.append("Spatial context analysis included")
        
        # Escalation information
        if classification.requires_escalation():
            description_parts.append("This anomaly requires immediate attention and escalation")
        
        return ". ".join(description_parts)

    def _create_technical_context(
        self, classification: AdvancedAnomalyClassification, detector: Detector
    ) -> dict[str, Any]:
        """Create technical context for the event."""
        technical_context = {
            "detector_info": detector.get_info(),
            "classification_method": classification.basic_classification.classification_method.value,
            "confidence_level": classification.basic_classification.confidence_level.value,
            "requires_escalation": classification.requires_escalation(),
        }
        
        if classification.is_hierarchical():
            technical_context["hierarchy_depth"] = classification.hierarchical_classification.get_hierarchy_depth()
        
        if classification.is_multi_class():
            technical_context["alternative_classifications"] = len(
                classification.multi_class_classification.alternative_results
            )
            technical_context["has_ambiguous_classification"] = classification.multi_class_classification.has_ambiguous_classification()
        
        return technical_context

    def _add_classification_tags(
        self, event: AnomalyEvent, classification: AdvancedAnomalyClassification
    ) -> None:
        """Add classification-specific tags to the event."""
        # Basic tags
        event.add_tag(f"severity:{classification.severity_classification}")
        event.add_tag(f"confidence:{classification.basic_classification.confidence_level.value}")
        event.add_tag(f"method:{classification.basic_classification.classification_method.value}")
        
        # Hierarchical tags
        if classification.is_hierarchical():
            event.add_tag("hierarchical_classification")
            event.add_tag(f"primary_category:{classification.hierarchical_classification.primary_category}")
            if classification.hierarchical_classification.sub_type:
                event.add_tag(f"subtype:{classification.hierarchical_classification.sub_type.value}")
        
        # Multi-class tags
        if classification.is_multi_class():
            event.add_tag("multiclass_classification")
            if classification.multi_class_classification.has_ambiguous_classification():
                event.add_tag("ambiguous_classification")
        
        # Context tags
        if classification.has_temporal_context():
            event.add_tag("temporal_context")
        
        if classification.has_spatial_context():
            event.add_tag("spatial_context")
        
        # Escalation tag
        if classification.requires_escalation():
            event.add_tag("requires_escalation")

    def create_classification_summary_event(
        self, classifications: list[AdvancedAnomalyClassification], detector: Detector
    ) -> AnomalyEvent:
        """Create a summary event for a batch of classifications."""
        summary = self.classification_service.get_classification_summary(classifications)
        
        event = AnomalyEvent(
            event_type=EventType.BATCH_COMPLETED,
            severity=EventSeverity.INFO,
            title=f"Classification batch completed by {detector.name}",
            description=f"Processed {summary['total_classifications']} classifications "
                       f"with {summary['anomaly_count']} anomalies detected",
            raw_data={"classification_summary": summary},
            event_time="",
            detector_id=detector.id,
            technical_context={"batch_summary": summary},
        )
        
        # Add summary tags
        event.add_tag("batch_summary")
        event.add_tag(f"total_classifications:{summary['total_classifications']}")
        event.add_tag(f"anomaly_count:{summary['anomaly_count']}")
        
        return event