"""Rule-based type classifier using pattern heuristics and temporal features."""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import numpy as np

from pynomaly.domain.value_objects import AnomalyScore

logger = logging.getLogger(__name__)


class RuleBasedTypeClassifier:
    """Rule-based type classifier using pattern heuristics and temporal features.
    
    This classifier uses heuristic rules to classify anomalies into types
    based on patterns in the data, temporal features, and contextual information.
    """

    DEFAULT_RULES = {
        "spike": {
            "score_threshold": 0.8,
            "duration_threshold": 1,  # minutes
            "pattern": "sudden_increase"
        },
        "drift": {
            "score_threshold": 0.6,
            "duration_threshold": 60,  # minutes
            "pattern": "gradual_increase"
        },
        "seasonal": {
            "score_threshold": 0.5,
            "duration_threshold": 1440,  # minutes (24 hours)
            "pattern": "cyclic"
        },
        "outlier": {
            "score_threshold": 0.7,
            "duration_threshold": 5,  # minutes
            "pattern": "isolated"
        }
    }

    def __init__(
        self,
        rules: Optional[Dict[str, Dict[str, Any]]] = None,
        default_type: str = "outlier"
    ):
        """Initialize the rule-based type classifier.
        
        Args:
            rules: Custom rules for type classification
            default_type: Default type for anomalies that don't match any rule
        """
        self.rules = rules or self.DEFAULT_RULES.copy()
        self.default_type = default_type
        
    def classify_single(
        self,
        score: Union[float, AnomalyScore],
        timestamp: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Classify a single anomaly score into a type.
        
        Args:
            score: Anomaly score to classify
            timestamp: Timestamp of the anomaly
            context: Additional context information
            
        Returns:
            Anomaly type as string
        """
        if isinstance(score, AnomalyScore):
            score_value = score.value
        else:
            score_value = float(score)
            
        context = context or {}
        
        # Apply rules in order of priority
        for anomaly_type, rule in self.rules.items():
            if self._matches_rule(score_value, timestamp, context, rule):
                return anomaly_type
                
        return self.default_type
    
    def _matches_rule(
        self,
        score_value: float,
        timestamp: Optional[datetime],
        context: Dict[str, Any],
        rule: Dict[str, Any]
    ) -> bool:
        """Check if an anomaly matches a specific rule.
        
        Args:
            score_value: Anomaly score value
            timestamp: Timestamp of the anomaly
            context: Additional context information
            rule: Rule configuration
            
        Returns:
            True if the anomaly matches the rule
        """
        # Check score threshold
        if score_value < rule.get("score_threshold", 0.0):
            return False
            
        # Check pattern if available
        pattern = rule.get("pattern")
        if pattern and not self._matches_pattern(score_value, timestamp, context, pattern):
            return False
            
        # Check duration if available
        duration_threshold = rule.get("duration_threshold")
        if duration_threshold and not self._matches_duration(timestamp, context, duration_threshold):
            return False
            
        return True
    
    def _matches_pattern(
        self,
        score_value: float,
        timestamp: Optional[datetime],
        context: Dict[str, Any],
        pattern: str
    ) -> bool:
        """Check if an anomaly matches a specific pattern.
        
        Args:
            score_value: Anomaly score value
            timestamp: Timestamp of the anomaly
            context: Additional context information
            pattern: Pattern to match
            
        Returns:
            True if the anomaly matches the pattern
        """
        if pattern == "sudden_increase":
            return self._is_sudden_increase(score_value, context)
        elif pattern == "gradual_increase":
            return self._is_gradual_increase(score_value, context)
        elif pattern == "cyclic":
            return self._is_cyclic(timestamp, context)
        elif pattern == "isolated":
            return self._is_isolated(score_value, context)
        else:
            logger.warning(f"Unknown pattern: {pattern}")
            return False
    
    def _is_sudden_increase(self, score_value: float, context: Dict[str, Any]) -> bool:
        """Check if the anomaly represents a sudden increase."""
        previous_scores = context.get("previous_scores", [])
        if not previous_scores:
            return False
            
        # Check if current score is significantly higher than recent scores
        recent_avg = np.mean(previous_scores[-5:]) if len(previous_scores) >= 5 else np.mean(previous_scores)
        return score_value > recent_avg * 1.5
    
    def _is_gradual_increase(self, score_value: float, context: Dict[str, Any]) -> bool:
        """Check if the anomaly represents a gradual increase."""
        previous_scores = context.get("previous_scores", [])
        if len(previous_scores) < 3:
            return False
            
        # Check if there's a consistent upward trend
        trend = np.polyfit(range(len(previous_scores)), previous_scores, 1)[0]
        return trend > 0 and score_value > previous_scores[-1]
    
    def _is_cyclic(self, timestamp: Optional[datetime], context: Dict[str, Any]) -> bool:
        """Check if the anomaly follows a cyclic pattern."""
        if not timestamp:
            return False
            
        # Check if anomaly occurs at similar time of day or day of week
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        historical_times = context.get("historical_anomaly_times", [])
        if not historical_times:
            return False
            
        # Check for similar times in history
        similar_times = 0
        for hist_time in historical_times:
            if isinstance(hist_time, datetime):
                if (abs(hist_time.hour - hour) <= 1 or 
                    hist_time.weekday() == day_of_week):
                    similar_times += 1
                    
        return similar_times >= 2
    
    def _is_isolated(self, score_value: float, context: Dict[str, Any]) -> bool:
        """Check if the anomaly is isolated (no nearby anomalies)."""
        nearby_scores = context.get("nearby_scores", [])
        if not nearby_scores:
            return True  # No nearby context suggests isolation
            
        # Check if nearby scores are significantly lower
        avg_nearby = np.mean(nearby_scores)
        return score_value > avg_nearby * 2
    
    def _matches_duration(
        self,
        timestamp: Optional[datetime],
        context: Dict[str, Any],
        duration_threshold: int
    ) -> bool:
        """Check if the anomaly duration matches the threshold.
        
        Args:
            timestamp: Timestamp of the anomaly
            context: Additional context information
            duration_threshold: Duration threshold in minutes
            
        Returns:
            True if duration matches
        """
        if not timestamp:
            return True  # No timestamp info, assume it matches
            
        anomaly_start = context.get("anomaly_start_time")
        if not anomaly_start:
            return True  # No start time info, assume it matches
            
        if isinstance(anomaly_start, datetime):
            duration_minutes = (timestamp - anomaly_start).total_seconds() / 60
            return duration_minutes <= duration_threshold
            
        return True
    
    def classify_batch(
        self,
        scores: List[Union[float, AnomalyScore]],
        timestamps: Optional[List[datetime]] = None,
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Classify a batch of anomaly scores into types.
        
        Args:
            scores: List of anomaly scores to classify
            timestamps: List of timestamps for each score
            contexts: List of context dictionaries for each score
            
        Returns:
            List of anomaly types
        """
        if not scores:
            return []
            
        timestamps = timestamps or [None] * len(scores)
        contexts = contexts or [{}] * len(scores)
        
        return [
            self.classify_single(score, timestamp, context)
            for score, timestamp, context in zip(scores, timestamps, contexts)
        ]
    
    def get_type_stats(
        self,
        scores: List[Union[float, AnomalyScore]],
        timestamps: Optional[List[datetime]] = None,
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get statistics about type distribution.
        
        Args:
            scores: List of anomaly scores
            timestamps: List of timestamps for each score
            contexts: List of context dictionaries for each score
            
        Returns:
            Dictionary with type statistics
        """
        if not scores:
            return {}
            
        classifications = self.classify_batch(scores, timestamps, contexts)
        total_count = len(classifications)
        
        stats = {}
        all_types = set(self.rules.keys()) | {self.default_type}
        
        for anomaly_type in all_types:
            count = classifications.count(anomaly_type)
            stats[anomaly_type] = {
                "count": count,
                "percentage": (count / total_count) * 100 if total_count > 0 else 0.0
            }
            
        return stats
    
    def add_rule(self, anomaly_type: str, rule: Dict[str, Any]) -> None:
        """Add a new classification rule.
        
        Args:
            anomaly_type: Type of anomaly for this rule
            rule: Rule configuration
        """
        self.rules[anomaly_type] = rule.copy()
    
    def remove_rule(self, anomaly_type: str) -> None:
        """Remove a classification rule.
        
        Args:
            anomaly_type: Type of anomaly to remove
        """
        if anomaly_type in self.rules:
            del self.rules[anomaly_type]
    
    def get_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get a copy of the current rules.
        
        Returns:
            Copy of the rules dictionary
        """
        return {
            anomaly_type: rule.copy()
            for anomaly_type, rule in self.rules.items()
        }
    
    def reset_to_default(self) -> None:
        """Reset rules to default configuration."""
        self.rules = self.DEFAULT_RULES.copy()
    
    def __repr__(self) -> str:
        """String representation of the classifier."""
        return f"RuleBasedTypeClassifier(rules={len(self.rules)}, default={self.default_type})"
