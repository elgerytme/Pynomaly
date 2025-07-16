"""Feature Importance value object for ML feature analysis."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_value_object import BaseValueObject


class FeatureImportance(BaseValueObject):
    """Value object representing feature importance scores and rankings.
    
    This immutable value object encapsulates feature importance measurements
    from various machine learning algorithms and statistical methods.
    
    Attributes:
        feature_scores: Dictionary mapping feature names to importance scores
        method: Method used to calculate importance (e.g., 'random_forest', 'permutation')
        normalized: Whether scores are normalized (0-1 range)
        ranking: Ordered list of features by importance (most to least important)
        top_features: List of top N most important features
        threshold: Minimum importance threshold for feature selection
        selected_features: Features above the threshold
        statistical_significance: P-values or confidence measures if available
        stability_scores: Cross-validation stability of importance scores
        correlation_matrix: Correlation between features for redundancy analysis
        mutual_information: Mutual information scores if calculated
        univariate_scores: Univariate statistical test scores
        recursive_elimination: Results from recursive feature elimination
        l1_regularization: L1 penalty coefficients if available
        shap_values: SHAP (SHapley Additive exPlanations) values if calculated
        permutation_importance: Permutation-based importance scores
        model_specific_importance: Algorithm-specific importance measures
        feature_groups: Grouping of related features
        interaction_effects: Feature interaction importance
        temporal_stability: How stable importance is over time
    """
    
    # Core importance data
    feature_scores: dict[str, float] = Field(..., min_items=1)
    method: str = Field(..., min_length=1)
    normalized: bool = Field(default=True)
    
    # Derived rankings and selections
    ranking: list[str] = Field(default_factory=list)
    top_features: list[str] = Field(default_factory=list)
    threshold: Optional[float] = Field(None, ge=0, le=1)
    selected_features: list[str] = Field(default_factory=list)
    
    # Statistical measures
    statistical_significance: dict[str, float] = Field(default_factory=dict)
    stability_scores: dict[str, float] = Field(default_factory=dict)
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    
    # Alternative importance methods
    mutual_information: dict[str, float] = Field(default_factory=dict)
    univariate_scores: dict[str, float] = Field(default_factory=dict)
    recursive_elimination: dict[str, Any] = Field(default_factory=dict)
    l1_regularization: dict[str, float] = Field(default_factory=dict)
    
    # Advanced importance measures
    shap_values: dict[str, dict[str, float]] = Field(default_factory=dict)
    permutation_importance: dict[str, float] = Field(default_factory=dict)
    model_specific_importance: dict[str, Any] = Field(default_factory=dict)
    
    # Feature organization
    feature_groups: dict[str, list[str]] = Field(default_factory=dict)
    interaction_effects: dict[str, float] = Field(default_factory=dict)
    temporal_stability: dict[str, float] = Field(default_factory=dict)
    
    @validator('feature_scores')
    def validate_feature_scores(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate feature importance scores."""
        if not v:
            raise ValueError("Feature scores cannot be empty")
        
        for feature_name, score in v.items():
            if not feature_name.strip():
                raise ValueError("Feature names cannot be empty")
            
            if not isinstance(score, (int, float)):
                raise ValueError(f"Score for feature '{feature_name}' must be numeric")
            
            if score < 0:
                raise ValueError(f"Importance score for feature '{feature_name}' cannot be negative")
        
        return v
    
    @validator('ranking')
    def validate_ranking(cls, v: list[str], values: dict[str, Any]) -> list[str]:
        """Validate feature ranking."""
        feature_scores = values.get('feature_scores', {})
        
        if not v and feature_scores:
            # Auto-generate ranking if not provided
            return sorted(feature_scores.keys(), key=lambda x: feature_scores[x], reverse=True)
        
        # Check that ranking contains all features
        if set(v) != set(feature_scores.keys()):
            missing = set(feature_scores.keys()) - set(v)
            extra = set(v) - set(feature_scores.keys())
            
            if missing:
                raise ValueError(f"Ranking missing features: {missing}")
            if extra:
                raise ValueError(f"Ranking contains unknown features: {extra}")
        
        return v
    
    @validator('top_features')
    def validate_top_features(cls, v: list[str], values: dict[str, Any]) -> list[str]:
        """Validate top features list."""
        ranking = values.get('ranking', [])
        
        if not v and ranking:
            # Auto-generate top 10 features if not provided
            return ranking[:10]
        
        # Ensure top features are in the ranking
        for feature in v:
            if feature not in ranking:
                raise ValueError(f"Top feature '{feature}' not found in ranking")
        
        return v
    
    @validator('selected_features')
    def validate_selected_features(cls, v: list[str], values: dict[str, Any]) -> list[str]:
        """Validate selected features based on threshold."""
        if not v:
            threshold = values.get('threshold')
            feature_scores = values.get('feature_scores', {})
            
            if threshold is not None and feature_scores:
                # Auto-generate selected features based on threshold
                return [
                    feature for feature, score in feature_scores.items()
                    if score >= threshold
                ]
        
        return v
    
    def get_importance_score(self, feature_name: str) -> Optional[float]:
        """Get importance score for a specific feature."""
        return self.feature_scores.get(feature_name)
    
    def get_feature_rank(self, feature_name: str) -> Optional[int]:
        """Get rank of a specific feature (1-based)."""
        try:
            return self.ranking.index(feature_name) + 1
        except ValueError:
            return None
    
    def get_top_n_features(self, n: int) -> list[str]:
        """Get top N most important features."""
        return self.ranking[:n]
    
    def get_bottom_n_features(self, n: int) -> list[str]:
        """Get bottom N least important features."""
        return self.ranking[-n:] if n > 0 else []
    
    def filter_by_threshold(self, threshold: float) -> list[str]:
        """Get features above a certain importance threshold."""
        return [
            feature for feature, score in self.feature_scores.items()
            if score >= threshold
        ]
    
    def filter_by_percentile(self, percentile: float) -> list[str]:
        """Get features above a certain percentile of importance."""
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        
        import numpy as np
        scores = list(self.feature_scores.values())
        threshold = np.percentile(scores, percentile)
        
        return self.filter_by_threshold(threshold)
    
    def get_feature_group_importance(self, group_name: str) -> Optional[float]:
        """Get aggregate importance for a feature group."""
        if group_name not in self.feature_groups:
            return None
        
        group_features = self.feature_groups[group_name]
        group_scores = [
            self.feature_scores.get(feature, 0) 
            for feature in group_features
        ]
        
        return sum(group_scores) if group_scores else 0
    
    def get_redundant_features(self, correlation_threshold: float = 0.8) -> list[tuple[str, str]]:
        """Get pairs of highly correlated (potentially redundant) features."""
        redundant_pairs = []
        
        for feature1, correlations in self.correlation_matrix.items():
            for feature2, correlation in correlations.items():
                if (feature1 != feature2 and 
                    abs(correlation) >= correlation_threshold and
                    (feature2, feature1) not in redundant_pairs):
                    redundant_pairs.append((feature1, feature2))
        
        return redundant_pairs
    
    def get_stable_features(self, stability_threshold: float = 0.7) -> list[str]:
        """Get features with stable importance across cross-validation."""
        return [
            feature for feature, stability in self.stability_scores.items()
            if stability >= stability_threshold
        ]
    
    def compare_methods(self, other: FeatureImportance) -> dict[str, Any]:
        """Compare feature importance from different methods."""
        if not isinstance(other, FeatureImportance):
            raise ValueError("Can only compare with another FeatureImportance")
        
        common_features = set(self.feature_scores.keys()) & set(other.feature_scores.keys())
        
        if not common_features:
            return {"error": "No common features to compare"}
        
        # Calculate rank correlation
        self_ranks = {feature: self.get_feature_rank(feature) for feature in common_features}
        other_ranks = {feature: other.get_feature_rank(feature) for feature in common_features}
        
        # Calculate score correlation
        import numpy as np
        self_scores = [self.feature_scores[f] for f in common_features]
        other_scores = [other.feature_scores[f] for f in common_features]
        
        score_correlation = np.corrcoef(self_scores, other_scores)[0, 1]
        
        # Find features with largest disagreements
        score_differences = {
            feature: abs(self.feature_scores[feature] - other.feature_scores[feature])
            for feature in common_features
        }
        
        most_disagreed = sorted(score_differences.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "methods_compared": [self.method, other.method],
            "common_features": len(common_features),
            "score_correlation": score_correlation,
            "most_disagreed_features": most_disagreed,
            "top_features_overlap": len(set(self.top_features) & set(other.top_features))
        }
    
    def get_importance_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of feature importance."""
        summary = {
            "method": self.method,
            "total_features": len(self.feature_scores),
            "normalized": self.normalized,
            "top_5_features": self.get_top_n_features(5),
            "selected_features_count": len(self.selected_features),
            "threshold": self.threshold,
        }
        
        # Add statistics about scores
        import numpy as np
        scores = list(self.feature_scores.values())
        
        summary["score_statistics"] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "median": np.median(scores)
        }
        
        # Add information about stability if available
        if self.stability_scores:
            stability_values = list(self.stability_scores.values())
            summary["stability_statistics"] = {
                "mean_stability": np.mean(stability_values),
                "stable_features": len(self.get_stable_features())
            }
        
        # Add redundancy information
        if self.correlation_matrix:
            redundant = self.get_redundant_features()
            summary["redundancy_analysis"] = {
                "redundant_pairs": len(redundant),
                "max_correlation": max(
                    abs(corr) for corrs in self.correlation_matrix.values() 
                    for corr in corrs.values()
                ) if self.correlation_matrix else 0
            }
        
        return summary
    
    @classmethod
    def from_sklearn_importance(cls, feature_names: list[str], 
                              importance_scores: Any, method: str,
                              **kwargs: Any) -> FeatureImportance:
        """Create FeatureImportance from scikit-learn importance scores."""
        import numpy as np
        
        if not isinstance(importance_scores, np.ndarray):
            importance_scores = np.array(importance_scores)
        
        if len(feature_names) != len(importance_scores):
            raise ValueError("Feature names and scores must have same length")
        
        # Normalize scores to 0-1 range
        max_score = np.max(importance_scores)
        if max_score > 0:
            normalized_scores = importance_scores / max_score
        else:
            normalized_scores = importance_scores
        
        feature_scores = dict(zip(feature_names, normalized_scores))
        
        return cls(
            feature_scores=feature_scores,
            method=method,
            normalized=True,
            **kwargs
        )
    
    @classmethod
    def from_permutation_importance(cls, feature_names: list[str],
                                  importance_scores: Any, 
                                  importances_std: Optional[Any] = None,
                                  **kwargs: Any) -> FeatureImportance:
        """Create FeatureImportance from permutation importance results."""
        import numpy as np
        
        if not isinstance(importance_scores, np.ndarray):
            importance_scores = np.array(importance_scores)
        
        # Normalize scores
        max_score = np.max(importance_scores)
        if max_score > 0:
            normalized_scores = importance_scores / max_score
        else:
            normalized_scores = importance_scores
        
        feature_scores = dict(zip(feature_names, normalized_scores))
        
        # Add stability scores if standard deviations provided
        stability_scores = {}
        if importances_std is not None:
            if not isinstance(importances_std, np.ndarray):
                importances_std = np.array(importances_std)
            
            # Stability = 1 - (std / mean) for each feature
            for i, feature in enumerate(feature_names):
                mean_score = importance_scores[i]
                std_score = importances_std[i]
                
                if mean_score > 0:
                    cv = std_score / mean_score  # coefficient of variation
                    stability = max(0, 1 - cv)  # Higher stability = lower CV
                else:
                    stability = 0
                
                stability_scores[feature] = stability
        
        return cls(
            feature_scores=feature_scores,
            method="permutation_importance",
            normalized=True,
            stability_scores=stability_scores,
            **kwargs
        )