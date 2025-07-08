"""Factory for creating severity and type classifiers."""

from __future__ import annotations

from typing import Dict, Any, Optional, Union
import logging

from .threshold_severity_classifier import ThresholdSeverityClassifier
from .statistical_severity_classifier import StatisticalSeverityClassifier
from .rule_based_type_classifier import RuleBasedTypeClassifier

logger = logging.getLogger(__name__)


class ClassifierFactory:
    """Factory for creating and managing anomaly classifiers.
    
    This factory provides a centralized way to create classifiers with
    sensible defaults while allowing customization when needed.
    """
    
    @staticmethod
    def create_severity_classifier(
        classifier_type: str = "threshold",
        **kwargs
    ) -> Union[ThresholdSeverityClassifier, StatisticalSeverityClassifier]:
        """Create a severity classifier.
        
        Args:
            classifier_type: Type of classifier ("threshold" or "statistical")
            **kwargs: Additional arguments for the classifier
            
        Returns:
            Configured severity classifier
        """
        if classifier_type == "threshold":
            return ThresholdSeverityClassifier(**kwargs)
        elif classifier_type == "statistical":
            return StatisticalSeverityClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown severity classifier type: {classifier_type}")
    
    @staticmethod
    def create_type_classifier(
        classifier_type: str = "rule_based",
        **kwargs
    ) -> RuleBasedTypeClassifier:
        """Create a type classifier.
        
        Args:
            classifier_type: Type of classifier ("rule_based")
            **kwargs: Additional arguments for the classifier
            
        Returns:
            Configured type classifier
        """
        if classifier_type == "rule_based":
            return RuleBasedTypeClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown type classifier type: {classifier_type}")
    
    @staticmethod
    def create_default_classifiers() -> Dict[str, Any]:
        """Create default classifiers for maintaining current behavior.
        
        Returns:
            Dictionary containing default classifiers
        """
        return {
            "severity": ClassifierFactory.create_severity_classifier("threshold"),
            "type": ClassifierFactory.create_type_classifier("rule_based")
        }
    
    @staticmethod
    def create_custom_classifiers(
        severity_config: Optional[Dict[str, Any]] = None,
        type_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create custom classifiers with specified configurations.
        
        Args:
            severity_config: Configuration for severity classifier
            type_config: Configuration for type classifier
            
        Returns:
            Dictionary containing configured classifiers
        """
        classifiers = {}
        
        if severity_config:
            severity_type = severity_config.pop("type", "threshold")
            classifiers["severity"] = ClassifierFactory.create_severity_classifier(
                severity_type, **severity_config
            )
        else:
            classifiers["severity"] = ClassifierFactory.create_severity_classifier("threshold")
            
        if type_config:
            type_type = type_config.pop("type", "rule_based")
            classifiers["type"] = ClassifierFactory.create_type_classifier(
                type_type, **type_config
            )
        else:
            classifiers["type"] = ClassifierFactory.create_type_classifier("rule_based")
            
        return classifiers


def get_default_severity_classifier() -> ThresholdSeverityClassifier:
    """Get the default severity classifier to maintain current behavior.
    
    Returns:
        ThresholdSeverityClassifier instance with default configuration
    """
    return ThresholdSeverityClassifier()


def get_default_type_classifier() -> RuleBasedTypeClassifier:
    """Get the default type classifier.
    
    Returns:
        RuleBasedTypeClassifier instance with default configuration
    """
    return RuleBasedTypeClassifier()
