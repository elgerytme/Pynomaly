"""Categorical anomaly detection with confidence levels."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.value_objects.anomaly_category import AnomalyCategory
from pynomaly.domain.value_objects.anomaly_classification import (
    ClassificationMethod,
    ClassificationResult,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


class CategoricalAnomalyDetector:
    """Categorical anomaly detector with confidence-based classification.
    
    This detector specializes in identifying anomalies in categorical data
    and provides confidence levels for each classification decision.
    """

    def __init__(
        self,
        categories: list[str] | None = None,
        confidence_threshold: float = 0.5,
        frequency_threshold: float = 0.01,
        rare_category_threshold: float = 0.05,
    ):
        """Initialize categorical anomaly detector.
        
        Args:
            categories: Known categories (if None, will be learned from data)
            confidence_threshold: Minimum confidence for positive classification
            frequency_threshold: Minimum frequency for category to be considered normal
            rare_category_threshold: Threshold for rare category detection
        """
        self.categories = set(categories) if categories else set()
        self.confidence_threshold = confidence_threshold
        self.frequency_threshold = frequency_threshold
        self.rare_category_threshold = rare_category_threshold
        self.category_frequencies: dict[str, float] = {}
        self.total_observations = 0
        self.is_fitted = False

    def fit(self, categorical_data: list[str]) -> None:
        """Fit the detector to categorical data.
        
        Args:
            categorical_data: List of categorical values
        """
        if not categorical_data:
            raise ValueError("Cannot fit on empty data")
        
        # Count category frequencies
        category_counts = {}
        for category in categorical_data:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Update known categories
        self.categories.update(category_counts.keys())
        
        # Calculate frequencies
        self.total_observations = len(categorical_data)
        self.category_frequencies = {
            category: count / self.total_observations
            for category, count in category_counts.items()
        }
        
        self.is_fitted = True
        logger.info(f"Fitted categorical detector on {len(self.categories)} categories")

    def detect_single(self, category: str) -> ClassificationResult:
        """Detect anomaly in a single categorical value.
        
        Args:
            category: Categorical value to analyze
            
        Returns:
            Classification result with confidence
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        # Check if category is unknown
        if category not in self.categories:
            return self._create_unknown_category_result(category)
        
        # Check if category is rare
        frequency = self.category_frequencies.get(category, 0.0)
        if frequency < self.frequency_threshold:
            return self._create_rare_category_result(category, frequency)
        
        # Normal category
        return self._create_normal_category_result(category, frequency)

    def detect_batch(self, categorical_data: list[str]) -> list[ClassificationResult]:
        """Detect anomalies in batch of categorical values.
        
        Args:
            categorical_data: List of categorical values
            
        Returns:
            List of classification results
        """
        if not categorical_data:
            return []
        
        return [self.detect_single(category) for category in categorical_data]

    def _create_unknown_category_result(self, category: str) -> ClassificationResult:
        """Create result for unknown category."""
        confidence_score = 0.9  # High confidence for unknown categories
        
        return ClassificationResult(
            predicted_class="anomaly",
            confidence_score=confidence_score,
            confidence_level=ConfidenceLevel.VERY_HIGH,
            probability_distribution={
                "anomaly": confidence_score,
                "normal": 1.0 - confidence_score,
            },
            classification_method=ClassificationMethod.UNSUPERVISED,
            metadata={
                "anomaly_type": "unknown_category",
                "category": category,
                "known_categories": len(self.categories),
            },
        )

    def _create_rare_category_result(self, category: str, frequency: float) -> ClassificationResult:
        """Create result for rare category."""
        # Confidence inversely related to frequency
        confidence_score = max(0.5, 1.0 - (frequency / self.rare_category_threshold))
        
        return ClassificationResult.from_confidence_score(
            predicted_class="anomaly",
            confidence_score=confidence_score,
        ).replace(
            probability_distribution={
                "anomaly": confidence_score,
                "normal": 1.0 - confidence_score,
            },
            classification_method=ClassificationMethod.UNSUPERVISED,
            metadata={
                "anomaly_type": "rare_category",
                "category": category,
                "frequency": frequency,
                "frequency_threshold": self.frequency_threshold,
            },
        )

    def _create_normal_category_result(self, category: str, frequency: float) -> ClassificationResult:
        """Create result for normal category."""
        confidence_score = min(0.4, frequency)  # Low confidence for normal categories
        
        return ClassificationResult.from_confidence_score(
            predicted_class="normal",
            confidence_score=confidence_score,
        ).replace(
            probability_distribution={
                "anomaly": confidence_score,
                "normal": 1.0 - confidence_score,
            },
            classification_method=ClassificationMethod.UNSUPERVISED,
            metadata={
                "anomaly_type": "normal_category",
                "category": category,
                "frequency": frequency,
            },
        )

    def update_with_feedback(self, category: str, is_anomaly: bool) -> None:
        """Update detector with feedback.
        
        Args:
            category: Category that was classified
            is_anomaly: True if category is actually an anomaly
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before updating")
        
        # Add to known categories if not anomaly
        if not is_anomaly:
            self.categories.add(category)
            # Update frequency (simple incremental update)
            if category in self.category_frequencies:
                self.category_frequencies[category] += 1.0 / self.total_observations
            else:
                self.category_frequencies[category] = 1.0 / self.total_observations

    def get_category_statistics(self) -> dict[str, Any]:
        """Get statistics about learned categories."""
        if not self.is_fitted:
            return {}
        
        sorted_categories = sorted(
            self.category_frequencies.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        rare_categories = [
            cat for cat, freq in self.category_frequencies.items()
            if freq < self.rare_category_threshold
        ]
        
        return {
            "total_categories": len(self.categories),
            "total_observations": self.total_observations,
            "most_common_categories": sorted_categories[:10],
            "rare_categories": rare_categories,
            "rare_category_count": len(rare_categories),
            "frequency_threshold": self.frequency_threshold,
            "rare_category_threshold": self.rare_category_threshold,
        }

    def get_detector_info(self) -> dict[str, Any]:
        """Get detector information."""
        return {
            "detector_type": "categorical_anomaly_detector",
            "is_fitted": self.is_fitted,
            "confidence_threshold": self.confidence_threshold,
            "frequency_threshold": self.frequency_threshold,
            "rare_category_threshold": self.rare_category_threshold,
            "categories": list(self.categories),
            "category_statistics": self.get_category_statistics(),
        }


class AdvancedCategoricalAnomalyDetector:
    """Advanced categorical anomaly detector with multiple detection strategies."""

    def __init__(
        self,
        detection_strategies: list[str] | None = None,
        ensemble_method: str = "voting",
        confidence_aggregation: str = "mean",
    ):
        """Initialize advanced categorical detector.
        
        Args:
            detection_strategies: List of strategies to use
            ensemble_method: Method for combining results
            confidence_aggregation: Method for aggregating confidence scores
        """
        self.detection_strategies = detection_strategies or [
            "frequency_based",
            "pattern_based",
            "sequence_based",
        ]
        self.ensemble_method = ensemble_method
        self.confidence_aggregation = confidence_aggregation
        self.detectors: dict[str, CategoricalAnomalyDetector] = {}
        self.is_fitted = False

    def fit(self, categorical_data: list[str], sequence_data: list[list[str]] | None = None) -> None:
        """Fit the advanced detector.
        
        Args:
            categorical_data: List of categorical values
            sequence_data: List of categorical sequences (for sequence-based detection)
        """
        if not categorical_data:
            raise ValueError("Cannot fit on empty data")
        
        # Initialize detectors for each strategy
        for strategy in self.detection_strategies:
            if strategy == "frequency_based":
                self.detectors[strategy] = CategoricalAnomalyDetector()
                self.detectors[strategy].fit(categorical_data)
            
            elif strategy == "pattern_based":
                self.detectors[strategy] = self._create_pattern_detector(categorical_data)
            
            elif strategy == "sequence_based" and sequence_data:
                self.detectors[strategy] = self._create_sequence_detector(sequence_data)
        
        self.is_fitted = True
        logger.info(f"Fitted advanced categorical detector with {len(self.detectors)} strategies")

    def detect_single(self, category: str, context: dict[str, Any] | None = None) -> ClassificationResult:
        """Detect anomaly using ensemble of strategies.
        
        Args:
            category: Categorical value to analyze
            context: Additional context for detection
            
        Returns:
            Ensemble classification result
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        context = context or {}
        results = []
        
        # Get results from each detector
        for strategy, detector in self.detectors.items():
            if strategy == "frequency_based":
                result = detector.detect_single(category)
            elif strategy == "pattern_based":
                result = self._detect_pattern_anomaly(category, context)
            elif strategy == "sequence_based":
                result = self._detect_sequence_anomaly(category, context)
            else:
                continue
            
            results.append(result)
        
        # Ensemble results
        return self._ensemble_results(results, category)

    def _create_pattern_detector(self, categorical_data: list[str]) -> CategoricalAnomalyDetector:
        """Create pattern-based detector."""
        # Simple pattern detection based on character patterns
        patterns = set()
        for category in categorical_data:
            patterns.add(self._extract_pattern(category))
        
        detector = CategoricalAnomalyDetector(categories=list(patterns))
        detector.fit(list(patterns))
        return detector

    def _create_sequence_detector(self, sequence_data: list[list[str]]) -> CategoricalAnomalyDetector:
        """Create sequence-based detector."""
        # Extract transitions from sequences
        transitions = []
        for sequence in sequence_data:
            for i in range(len(sequence) - 1):
                transition = f"{sequence[i]}->{sequence[i+1]}"
                transitions.append(transition)
        
        detector = CategoricalAnomalyDetector()
        detector.fit(transitions)
        return detector

    def _extract_pattern(self, category: str) -> str:
        """Extract pattern from category."""
        # Simple pattern extraction (can be enhanced)
        pattern = ""
        for char in category:
            if char.isalpha():
                pattern += "A"
            elif char.isdigit():
                pattern += "D"
            else:
                pattern += "S"
        return pattern

    def _detect_pattern_anomaly(self, category: str, context: dict[str, Any]) -> ClassificationResult:
        """Detect pattern-based anomaly."""
        pattern = self._extract_pattern(category)
        return self.detectors["pattern_based"].detect_single(pattern)

    def _detect_sequence_anomaly(self, category: str, context: dict[str, Any]) -> ClassificationResult:
        """Detect sequence-based anomaly."""
        previous_category = context.get("previous_category")
        if not previous_category:
            return ClassificationResult.from_confidence_score("normal", 0.1)
        
        transition = f"{previous_category}->{category}"
        return self.detectors["sequence_based"].detect_single(transition)

    def _ensemble_results(self, results: list[ClassificationResult], category: str) -> ClassificationResult:
        """Ensemble classification results."""
        if not results:
            return ClassificationResult.from_confidence_score("normal", 0.1)
        
        # Count predictions
        anomaly_count = sum(1 for r in results if r.predicted_class == "anomaly")
        normal_count = len(results) - anomaly_count
        
        # Aggregate confidence scores
        if self.confidence_aggregation == "mean":
            avg_confidence = sum(r.confidence_score for r in results) / len(results)
        elif self.confidence_aggregation == "max":
            avg_confidence = max(r.confidence_score for r in results)
        else:  # min
            avg_confidence = min(r.confidence_score for r in results)
        
        # Determine ensemble prediction
        if self.ensemble_method == "voting":
            predicted_class = "anomaly" if anomaly_count > normal_count else "normal"
        else:  # confidence_based
            predicted_class = "anomaly" if avg_confidence > 0.5 else "normal"
        
        # Create ensemble result
        return ClassificationResult.from_confidence_score(
            predicted_class, avg_confidence
        ).replace(
            classification_method=ClassificationMethod.ENSEMBLE,
            metadata={
                "ensemble_method": self.ensemble_method,
                "strategy_results": len(results),
                "anomaly_votes": anomaly_count,
                "normal_votes": normal_count,
                "category": category,
            },
        )

    def get_detector_info(self) -> dict[str, Any]:
        """Get advanced detector information."""
        return {
            "detector_type": "advanced_categorical_anomaly_detector",
            "is_fitted": self.is_fitted,
            "detection_strategies": self.detection_strategies,
            "ensemble_method": self.ensemble_method,
            "confidence_aggregation": self.confidence_aggregation,
            "active_detectors": list(self.detectors.keys()),
        }