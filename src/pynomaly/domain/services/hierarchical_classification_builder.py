"""Hierarchical classification builder for anomaly taxonomies."""

from __future__ import annotations

import logging
from typing import Any

from pynomaly.domain.value_objects.anomaly_classification import (
    AnomalySubType,
    ClassificationResult,
    HierarchicalClassification,
)
from pynomaly.domain.value_objects.anomaly_category import AnomalyCategory

logger = logging.getLogger(__name__)


class HierarchicalClassificationBuilder:
    """Builder for creating hierarchical anomaly classifications.
    
    This builder provides a systematic way to construct hierarchical
    classifications with parent-child relationships, supporting various
    anomaly taxonomies and classification schemes.
    """

    def __init__(self):
        """Initialize the hierarchical classification builder."""
        self.taxonomy_rules = self._load_default_taxonomy_rules()
        self.classification_cache: dict[str, HierarchicalClassification] = {}

    def _load_default_taxonomy_rules(self) -> dict[str, dict[str, Any]]:
        """Load default taxonomy rules for hierarchical classification."""
        return {
            "statistical": {
                "secondary_categories": {
                    "univariate": ["z_score", "iqr", "grubbs"],
                    "multivariate": ["mahalanobis", "pca", "correlation"],
                    "distribution": ["gaussian", "non_parametric", "empirical"],
                },
                "subtypes": {
                    "z_score": [AnomalySubType.OUTLIER, AnomalySubType.EXTREME_VALUE],
                    "mahalanobis": [AnomalySubType.OUTLIER, AnomalySubType.NOVELTY],
                    "correlation": [AnomalySubType.CONDITIONAL, AnomalySubType.PATTERN],
                },
            },
            "clustering": {
                "secondary_categories": {
                    "density_based": ["dbscan", "optics", "lof"],
                    "centroid_based": ["kmeans", "gaussian_mixture"],
                    "hierarchical": ["agglomerative", "divisive"],
                },
                "subtypes": {
                    "dbscan": [AnomalySubType.CLUSTER, AnomalySubType.NEIGHBORHOOD],
                    "lof": [AnomalySubType.NEIGHBORHOOD, AnomalySubType.LOCALIZED],
                    "kmeans": [AnomalySubType.CLUSTER, AnomalySubType.REGIONAL],
                },
            },
            "distance": {
                "secondary_categories": {
                    "nearest_neighbor": ["knn", "isolation_forest"],
                    "metric_based": ["euclidean", "manhattan", "cosine"],
                    "kernel_based": ["rbf", "polynomial", "sigmoid"],
                },
                "subtypes": {
                    "knn": [AnomalySubType.NEIGHBORHOOD, AnomalySubType.LOCALIZED],
                    "isolation_forest": [AnomalySubType.OUTLIER, AnomalySubType.NOVELTY],
                    "euclidean": [AnomalySubType.OUTLIER, AnomalySubType.EXTREME_VALUE],
                },
            },
            "neural": {
                "secondary_categories": {
                    "autoencoder": ["vanilla", "variational", "adversarial"],
                    "generative": ["gan", "vae", "normalizing_flow"],
                    "deep_learning": ["lstm", "cnn", "transformer"],
                },
                "subtypes": {
                    "autoencoder": [AnomalySubType.NOVELTY, AnomalySubType.PATTERN],
                    "gan": [AnomalySubType.NOVELTY, AnomalySubType.TEMPORAL],
                    "lstm": [AnomalySubType.SEQUENCE, AnomalySubType.TEMPORAL],
                },
            },
            "ensemble": {
                "secondary_categories": {
                    "voting": ["hard_voting", "soft_voting", "weighted_voting"],
                    "bagging": ["random_forest", "extra_trees"],
                    "boosting": ["adaboost", "gradient_boosting", "xgboost"],
                },
                "subtypes": {
                    "voting": [AnomalySubType.OUTLIER, AnomalySubType.NOVELTY],
                    "random_forest": [AnomalySubType.OUTLIER, AnomalySubType.CLUSTER],
                    "gradient_boosting": [AnomalySubType.PATTERN, AnomalySubType.CONDITIONAL],
                },
            },
        }

    def build_hierarchical_classification(
        self,
        primary_category: str,
        algorithm_name: str,
        confidence_score: float,
        feature_characteristics: dict[str, Any] | None = None,
        context_data: dict[str, Any] | None = None,
    ) -> HierarchicalClassification:
        """Build hierarchical classification.
        
        Args:
            primary_category: Primary classification category
            algorithm_name: Algorithm used for detection
            confidence_score: Confidence score for classification
            feature_characteristics: Characteristics of the features
            context_data: Additional context for classification
            
        Returns:
            Hierarchical classification with parent-child relationships
        """
        feature_characteristics = feature_characteristics or {}
        context_data = context_data or {}
        
        # Check cache first
        cache_key = f"{primary_category}_{algorithm_name}_{confidence_score:.2f}"
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        # Determine secondary category
        secondary_category = self._determine_secondary_category(
            primary_category, algorithm_name, feature_characteristics
        )
        
        # Determine tertiary category
        tertiary_category = self._determine_tertiary_category(
            primary_category, secondary_category, algorithm_name, feature_characteristics
        )
        
        # Determine anomaly subtype
        sub_type = self._determine_anomaly_subtype(
            primary_category, secondary_category, algorithm_name, 
            confidence_score, feature_characteristics, context_data
        )
        
        # Calculate confidence scores for each level
        confidence_scores = self._calculate_hierarchical_confidence_scores(
            primary_category, secondary_category, tertiary_category, 
            sub_type, confidence_score
        )
        
        # Create hierarchical classification
        classification = HierarchicalClassification(
            primary_category=primary_category,
            secondary_category=secondary_category,
            tertiary_category=tertiary_category,
            sub_type=sub_type,
            confidence_scores=confidence_scores,
        )
        
        # Cache the result
        self.classification_cache[cache_key] = classification
        
        return classification

    def _determine_secondary_category(
        self,
        primary_category: str,
        algorithm_name: str,
        feature_characteristics: dict[str, Any],
    ) -> str | None:
        """Determine secondary category based on algorithm and features."""
        if primary_category not in self.taxonomy_rules:
            return None
        
        secondary_categories = self.taxonomy_rules[primary_category]["secondary_categories"]
        algorithm_lower = algorithm_name.lower()
        
        # Direct algorithm mapping
        for category, algorithms in secondary_categories.items():
            if any(alg in algorithm_lower for alg in algorithms):
                return category
        
        # Feature-based inference
        if feature_characteristics:
            num_features = feature_characteristics.get("num_features", 0)
            
            if primary_category == "statistical":
                return "univariate" if num_features == 1 else "multivariate"
            elif primary_category == "clustering":
                return "density_based" if "density" in feature_characteristics else "centroid_based"
            elif primary_category == "distance":
                return "nearest_neighbor" if "neighbors" in feature_characteristics else "metric_based"
        
        # Default to first category if no match
        return list(secondary_categories.keys())[0]

    def _determine_tertiary_category(
        self,
        primary_category: str,
        secondary_category: str | None,
        algorithm_name: str,
        feature_characteristics: dict[str, Any],
    ) -> str | None:
        """Determine tertiary category for more specific classification."""
        if not secondary_category or primary_category not in self.taxonomy_rules:
            return None
        
        secondary_categories = self.taxonomy_rules[primary_category]["secondary_categories"]
        
        if secondary_category in secondary_categories:
            algorithms = secondary_categories[secondary_category]
            algorithm_lower = algorithm_name.lower()
            
            # Find specific algorithm match
            for algorithm in algorithms:
                if algorithm in algorithm_lower:
                    return algorithm
        
        return None

    def _determine_anomaly_subtype(
        self,
        primary_category: str,
        secondary_category: str | None,
        algorithm_name: str,
        confidence_score: float,
        feature_characteristics: dict[str, Any],
        context_data: dict[str, Any],
    ) -> AnomalySubType | None:
        """Determine specific anomaly subtype."""
        if primary_category not in self.taxonomy_rules:
            return None
        
        subtypes_mapping = self.taxonomy_rules[primary_category]["subtypes"]
        algorithm_lower = algorithm_name.lower()
        
        # Algorithm-based subtype determination
        for algorithm, subtypes in subtypes_mapping.items():
            if algorithm in algorithm_lower:
                return self._select_best_subtype(
                    subtypes, confidence_score, feature_characteristics, context_data
                )
        
        # Context-based subtype determination
        if context_data:
            if context_data.get("has_temporal_context"):
                return AnomalySubType.TEMPORAL
            elif context_data.get("has_spatial_context"):
                return AnomalySubType.SPATIAL
            elif context_data.get("has_sequence_data"):
                return AnomalySubType.SEQUENCE
        
        # Confidence-based subtype determination
        if confidence_score >= 0.9:
            return AnomalySubType.EXTREME_VALUE
        elif confidence_score >= 0.7:
            return AnomalySubType.OUTLIER
        elif confidence_score >= 0.5:
            return AnomalySubType.NOVELTY
        else:
            return AnomalySubType.CONDITIONAL

    def _select_best_subtype(
        self,
        subtypes: list[AnomalySubType],
        confidence_score: float,
        feature_characteristics: dict[str, Any],
        context_data: dict[str, Any],
    ) -> AnomalySubType:
        """Select the best subtype from available options."""
        if not subtypes:
            return AnomalySubType.OUTLIER
        
        # Score each subtype based on characteristics
        subtype_scores = {}
        
        for subtype in subtypes:
            score = 0.0
            
            # Confidence-based scoring
            if subtype == AnomalySubType.EXTREME_VALUE and confidence_score >= 0.9:
                score += 0.4
            elif subtype == AnomalySubType.OUTLIER and confidence_score >= 0.7:
                score += 0.3
            elif subtype == AnomalySubType.NOVELTY and confidence_score >= 0.5:
                score += 0.2
            
            # Feature-based scoring
            if feature_characteristics:
                if subtype == AnomalySubType.CLUSTER and "cluster_info" in feature_characteristics:
                    score += 0.3
                elif subtype == AnomalySubType.NEIGHBORHOOD and "neighbors" in feature_characteristics:
                    score += 0.3
                elif subtype == AnomalySubType.PATTERN and "patterns" in feature_characteristics:
                    score += 0.3
            
            # Context-based scoring
            if context_data:
                if subtype == AnomalySubType.TEMPORAL and context_data.get("has_temporal_context"):
                    score += 0.3
                elif subtype == AnomalySubType.SPATIAL and context_data.get("has_spatial_context"):
                    score += 0.3
                elif subtype == AnomalySubType.SEQUENCE and context_data.get("has_sequence_data"):
                    score += 0.3
            
            subtype_scores[subtype] = score
        
        # Return subtype with highest score
        return max(subtype_scores, key=subtype_scores.get)

    def _calculate_hierarchical_confidence_scores(
        self,
        primary_category: str,
        secondary_category: str | None,
        tertiary_category: str | None,
        sub_type: AnomalySubType | None,
        base_confidence: float,
    ) -> dict[str, float]:
        """Calculate confidence scores for each hierarchical level."""
        confidence_scores = {
            "primary": base_confidence,
        }
        
        # Secondary level confidence (slightly lower)
        if secondary_category:
            confidence_scores["secondary"] = base_confidence * 0.9
        
        # Tertiary level confidence (lower still)
        if tertiary_category:
            confidence_scores["tertiary"] = base_confidence * 0.8
        
        # Subtype confidence (lowest)
        if sub_type:
            confidence_scores["subtype"] = base_confidence * 0.7
        
        return confidence_scores

    def add_custom_taxonomy_rule(
        self,
        primary_category: str,
        secondary_categories: dict[str, list[str]],
        subtypes: dict[str, list[AnomalySubType]],
    ) -> None:
        """Add custom taxonomy rule.
        
        Args:
            primary_category: Primary category name
            secondary_categories: Secondary categories mapping
            subtypes: Subtypes mapping
        """
        self.taxonomy_rules[primary_category] = {
            "secondary_categories": secondary_categories,
            "subtypes": subtypes,
        }
        
        # Clear cache to force rebuilding
        self.classification_cache.clear()
        
        logger.info(f"Added custom taxonomy rule for {primary_category}")

    def get_taxonomy_structure(self) -> dict[str, Any]:
        """Get the complete taxonomy structure."""
        return {
            "primary_categories": list(self.taxonomy_rules.keys()),
            "taxonomy_rules": self.taxonomy_rules,
            "cache_size": len(self.classification_cache),
        }

    def validate_hierarchy_path(self, hierarchy_path: list[str]) -> bool:
        """Validate if a hierarchy path is valid according to taxonomy rules.
        
        Args:
            hierarchy_path: List of hierarchical categories
            
        Returns:
            True if path is valid, False otherwise
        """
        if not hierarchy_path:
            return False
        
        primary_category = hierarchy_path[0]
        
        if primary_category not in self.taxonomy_rules:
            return False
        
        if len(hierarchy_path) == 1:
            return True
        
        # Check secondary category
        secondary_category = hierarchy_path[1]
        secondary_categories = self.taxonomy_rules[primary_category]["secondary_categories"]
        
        if secondary_category not in secondary_categories:
            return False
        
        if len(hierarchy_path) == 2:
            return True
        
        # Check tertiary category
        tertiary_category = hierarchy_path[2]
        valid_tertiary = secondary_categories[secondary_category]
        
        if tertiary_category not in valid_tertiary:
            return False
        
        return True

    def get_possible_child_categories(self, parent_path: list[str]) -> list[str]:
        """Get possible child categories for a given parent path.
        
        Args:
            parent_path: Parent hierarchy path
            
        Returns:
            List of possible child categories
        """
        if not parent_path:
            return list(self.taxonomy_rules.keys())
        
        primary_category = parent_path[0]
        
        if primary_category not in self.taxonomy_rules:
            return []
        
        if len(parent_path) == 1:
            # Return secondary categories
            return list(self.taxonomy_rules[primary_category]["secondary_categories"].keys())
        
        elif len(parent_path) == 2:
            # Return tertiary categories
            secondary_category = parent_path[1]
            secondary_categories = self.taxonomy_rules[primary_category]["secondary_categories"]
            
            if secondary_category in secondary_categories:
                return secondary_categories[secondary_category]
        
        return []

    def clear_cache(self) -> None:
        """Clear the classification cache."""
        self.classification_cache.clear()
        logger.info("Classification cache cleared")

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.classification_cache),
            "cached_keys": list(self.classification_cache.keys()),
        }