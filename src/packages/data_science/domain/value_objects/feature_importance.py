"""Feature Importance value object for ML feature analysis."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_value_object import BaseValueObject


class ImportanceMethod(str, Enum):
    """Methods for calculating feature importance."""
    
    PERMUTATION = "permutation"
    GINI = "gini"
    SHAP = "shap"
    LIME = "lime"
    COEFFICIENT = "coefficient"
    CHI_SQUARED = "chi_squared"
    MUTUAL_INFO = "mutual_info"
    VARIANCE = "variance"
    CORRELATION = "correlation"
    RECURSIVE_ELIMINATION = "recursive_elimination"
    BORUTA = "boruta"
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


class ImportanceDirection(str, Enum):
    """Direction of feature importance (higher or lower is better)."""
    
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"
    BIDIRECTIONAL = "bidirectional"


class FeatureImportance(BaseValueObject):
    """Value object representing feature importance scores and rankings.
    
    This immutable value object encapsulates feature importance measurements
    from various ML algorithms and statistical methods.
    
    Attributes:
        features: List of feature names in order
        importances: Importance scores corresponding to features
        method: Method used to calculate importance
        direction: Whether higher or lower scores are better
        
        # Statistical properties
        normalized: Whether scores are normalized (0-1)
        absolute_values: Whether importance represents absolute values
        confidence_intervals: Confidence intervals for importance scores
        p_values: Statistical significance p-values for each feature
        
        # Ranking and selection
        rankings: Rank of each feature (1 = most important)
        percentiles: Percentile rank of each feature
        top_k_features: Names of top K most important features
        selected_features: Features selected by importance threshold
        
        # Stability metrics
        stability_score: Stability of importance across different runs
        reproducibility_score: Reproducibility across different datasets
        variance_across_folds: Variance of importance in cross-validation
        
        # Metadata
        model_type: Type of model used for importance calculation
        dataset_size: Size of dataset used for calculation
        feature_count: Number of features analyzed
        calculation_time: Time taken to calculate importance
        
        # Quality metrics
        importance_sum: Sum of all importance scores
        importance_mean: Mean importance score
        importance_std: Standard deviation of importance
        entropy: Information entropy of importance distribution
        
        # Feature selection criteria
        selection_threshold: Threshold used for feature selection
        cumulative_importance: Cumulative importance up to each feature
        importance_ratio: Ratio of max to min importance
        effective_features: Number of features above selection threshold
        
        # Business context
        business_interpretability: How interpretable features are for business
        computational_cost: Computational cost of using selected features
        feature_availability: Availability of features in production
    """
    
    features: list[str] = Field(..., min_items=1)
    importances: list[float] = Field(..., min_items=1)
    method: ImportanceMethod
    direction: ImportanceDirection = ImportanceDirection.HIGHER_BETTER
    
    # Statistical properties
    normalized: bool = Field(default=False)
    absolute_values: bool = Field(default=True)
    confidence_intervals: Optional[list[tuple[float, float]]] = None
    p_values: Optional[list[float]] = None
    
    # Rankings
    rankings: Optional[list[int]] = None
    percentiles: Optional[list[float]] = None
    top_k_features: Optional[list[str]] = None
    selected_features: Optional[list[str]] = None
    
    # Stability
    stability_score: Optional[float] = Field(None, ge=0, le=1)
    reproducibility_score: Optional[float] = Field(None, ge=0, le=1)
    variance_across_folds: Optional[list[float]] = None
    
    # Metadata
    model_type: Optional[str] = None
    dataset_size: Optional[int] = Field(None, gt=0)
    feature_count: Optional[int] = Field(None, gt=0)
    calculation_time: Optional[float] = Field(None, gt=0)
    
    # Quality metrics
    importance_sum: Optional[float] = None
    importance_mean: Optional[float] = None
    importance_std: Optional[float] = Field(None, ge=0)
    entropy: Optional[float] = Field(None, ge=0)
    
    # Selection criteria
    selection_threshold: Optional[float] = None
    cumulative_importance: Optional[list[float]] = None
    importance_ratio: Optional[float] = Field(None, gt=0)
    effective_features: Optional[int] = Field(None, ge=0)
    
    # Business metrics
    business_interpretability: Optional[float] = Field(None, ge=0, le=1)
    computational_cost: Optional[float] = Field(None, ge=0)
    feature_availability: Optional[float] = Field(None, ge=0, le=1)
    
    @validator('importances')
    def validate_importances_length(cls, v: list[float], values: dict[str, Any]) -> list[float]:
        """Validate importances match features length."""
        features = values.get('features', [])
        if len(v) != len(features):
            raise ValueError("Importances length must match features length")
        return v
    
    @validator('rankings')
    def validate_rankings(cls, v: Optional[list[int]], values: dict[str, Any]) -> Optional[list[int]]:
        """Validate rankings are consistent."""
        if v is not None:
            features = values.get('features', [])
            if len(v) != len(features):
                raise ValueError("Rankings length must match features length")
            
            # Check rankings are valid (1 to n)
            expected_ranks = set(range(1, len(features) + 1))
            actual_ranks = set(v)
            if actual_ranks != expected_ranks:
                raise ValueError("Rankings must be unique integers from 1 to number of features")
        return v
    
    @validator('percentiles')
    def validate_percentiles(cls, v: Optional[list[float]], values: dict[str, Any]) -> Optional[list[float]]:
        """Validate percentiles are in valid range."""
        if v is not None:
            features = values.get('features', [])
            if len(v) != len(features):
                raise ValueError("Percentiles length must match features length")
            
            for percentile in v:
                if not 0 <= percentile <= 100:
                    raise ValueError("Percentiles must be between 0 and 100")
        return v
    
    @validator('p_values')
    def validate_p_values(cls, v: Optional[list[float]], values: dict[str, Any]) -> Optional[list[float]]:
        """Validate p-values are in valid range."""
        if v is not None:
            features = values.get('features', [])
            if len(v) != len(features):
                raise ValueError("P-values length must match features length")
            
            for p_val in v:
                if not 0 <= p_val <= 1:
                    raise ValueError("P-values must be between 0 and 1")
        return v
    
    @validator('confidence_intervals')
    def validate_confidence_intervals(cls, v: Optional[list[tuple[float, float]]], 
                                    values: dict[str, Any]) -> Optional[list[tuple[float, float]]]:
        """Validate confidence intervals."""
        if v is not None:
            features = values.get('features', [])
            if len(v) != len(features):
                raise ValueError("Confidence intervals length must match features length")
            
            for lower, upper in v:
                if lower > upper:
                    raise ValueError("Confidence interval lower bound must be <= upper bound")
        return v
    
    @validator('cumulative_importance')
    def validate_cumulative_importance(cls, v: Optional[list[float]], 
                                     values: dict[str, Any]) -> Optional[list[float]]:
        """Validate cumulative importance is monotonic."""
        if v is not None:
            features = values.get('features', [])
            if len(v) != len(features):
                raise ValueError("Cumulative importance length must match features length")
            
            # Check monotonic (non-decreasing)
            for i in range(1, len(v)):
                if v[i] < v[i-1]:
                    raise ValueError("Cumulative importance must be non-decreasing")
        return v
    
    def get_top_features(self, k: int) -> list[str]:
        """Get top K most important features."""
        if k <= 0 or k > len(self.features):
            k = len(self.features)
        
        # Sort by importance (considering direction)
        importance_pairs = list(zip(self.features, self.importances))
        
        if self.direction == ImportanceDirection.HIGHER_BETTER:
            sorted_pairs = sorted(importance_pairs, key=lambda x: x[1], reverse=True)
        else:
            sorted_pairs = sorted(importance_pairs, key=lambda x: x[1])
        
        return [feature for feature, _ in sorted_pairs[:k]]
    
    def get_features_above_threshold(self, threshold: float) -> list[str]:
        """Get features above importance threshold."""
        selected = []
        for feature, importance in zip(self.features, self.importances):
            if self.direction == ImportanceDirection.HIGHER_BETTER:
                if importance >= threshold:
                    selected.append(feature)
            else:
                if importance <= threshold:
                    selected.append(feature)
        return selected
    
    def get_features_by_percentile(self, percentile: float) -> list[str]:
        """Get features above given percentile of importance."""
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        
        import numpy as np
        threshold_value = np.percentile(self.importances, percentile)
        return self.get_features_above_threshold(threshold_value)
    
    def get_feature_ranking(self, feature_name: str) -> Optional[int]:
        """Get ranking of specific feature."""
        if feature_name not in self.features:
            return None
        
        if self.rankings:
            idx = self.features.index(feature_name)
            return self.rankings[idx]
        
        # Calculate ranking on the fly
        importance_pairs = list(zip(self.features, self.importances))
        if self.direction == ImportanceDirection.HIGHER_BETTER:
            sorted_pairs = sorted(importance_pairs, key=lambda x: x[1], reverse=True)
        else:
            sorted_pairs = sorted(importance_pairs, key=lambda x: x[1])
        
        for rank, (feature, _) in enumerate(sorted_pairs, 1):
            if feature == feature_name:
                return rank
        
        return None
    
    def get_importance_statistics(self) -> dict[str, Any]:
        """Get statistical summary of importance scores."""
        import numpy as np
        
        importances_array = np.array(self.importances)
        
        stats = {
            "count": len(self.importances),
            "mean": float(np.mean(importances_array)),
            "std": float(np.std(importances_array)),
            "min": float(np.min(importances_array)),
            "max": float(np.max(importances_array)),
            "median": float(np.median(importances_array)),
            "q25": float(np.percentile(importances_array, 25)),
            "q75": float(np.percentile(importances_array, 75)),
            "range": float(np.max(importances_array) - np.min(importances_array)),
            "sum": float(np.sum(importances_array))
        }
        
        # Add entropy if not calculated
        if self.entropy is None:
            # Normalize for entropy calculation
            if self.normalized:
                probs = importances_array / np.sum(importances_array) if np.sum(importances_array) > 0 else importances_array
            else:
                # Use absolute values and normalize
                abs_importances = np.abs(importances_array)
                probs = abs_importances / np.sum(abs_importances) if np.sum(abs_importances) > 0 else abs_importances
            
            # Calculate entropy
            probs = probs[probs > 0]  # Remove zeros to avoid log(0)
            entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
            stats["entropy"] = float(entropy)
        else:
            stats["entropy"] = self.entropy
        
        return stats
    
    def get_feature_summary(self, feature_name: str) -> Optional[dict[str, Any]]:
        """Get comprehensive summary for specific feature."""
        if feature_name not in self.features:
            return None
        
        idx = self.features.index(feature_name)
        
        summary = {
            "feature_name": feature_name,
            "importance": self.importances[idx],
            "ranking": self.get_feature_ranking(feature_name),
            "method": self.method.value,
            "direction": self.direction.value
        }
        
        if self.percentiles:
            summary["percentile"] = self.percentiles[idx]
        
        if self.p_values:
            summary["p_value"] = self.p_values[idx]
            summary["significant"] = self.p_values[idx] < 0.05
        
        if self.confidence_intervals:
            lower, upper = self.confidence_intervals[idx]
            summary["confidence_interval"] = {"lower": lower, "upper": upper}
            summary["confidence_width"] = upper - lower
        
        if self.variance_across_folds:
            summary["cv_variance"] = self.variance_across_folds[idx]
        
        return summary
    
    def compare_with(self, other: FeatureImportance) -> dict[str, Any]:
        """Compare feature importance with another importance object."""
        if not isinstance(other, FeatureImportance):
            raise ValueError("Can only compare with another FeatureImportance")
        
        # Find common features
        common_features = set(self.features) & set(other.features)
        
        if not common_features:
            return {
                "common_features": 0,
                "correlation": None,
                "rank_correlation": None,
                "agreement_rate": 0.0
            }
        
        # Get importance values for common features
        self_importances = []
        other_importances = []
        
        for feature in common_features:
            self_idx = self.features.index(feature)
            other_idx = other.features.index(feature)
            self_importances.append(self.importances[self_idx])
            other_importances.append(other.importances[other_idx])
        
        import numpy as np
        from scipy.stats import pearsonr, spearmanr
        
        comparison = {
            "common_features": len(common_features),
            "method_self": self.method.value,
            "method_other": other.method.value
        }
        
        if len(self_importances) > 1:
            # Pearson correlation
            try:
                corr, p_val = pearsonr(self_importances, other_importances)
                comparison["correlation"] = float(corr)
                comparison["correlation_p_value"] = float(p_val)
            except:
                comparison["correlation"] = None
            
            # Spearman rank correlation
            try:
                rank_corr, rank_p_val = spearmanr(self_importances, other_importances)
                comparison["rank_correlation"] = float(rank_corr)
                comparison["rank_correlation_p_value"] = float(rank_p_val)
            except:
                comparison["rank_correlation"] = None
        
        # Top-k agreement
        k_values = [5, 10, min(20, len(common_features))]
        for k in k_values:
            if k <= len(common_features):
                self_top_k = set(self.get_top_features(k)) & common_features
                other_top_k = set(other.get_top_features(k)) & common_features
                agreement = len(self_top_k & other_top_k) / k
                comparison[f"top_{k}_agreement"] = agreement
        
        return comparison
    
    def get_feature_selection_summary(self) -> dict[str, Any]:
        """Get summary of feature selection based on importance."""
        total_features = len(self.features)
        
        summary = {
            "total_features": total_features,
            "method": self.method.value,
            "selection_approach": self.direction.value
        }
        
        # Add statistics
        stats = self.get_importance_statistics()
        summary.update(stats)
        
        # Feature selection by percentiles
        percentile_thresholds = [50, 75, 90, 95]
        for percentile in percentile_thresholds:
            selected = self.get_features_by_percentile(percentile)
            summary[f"features_above_{percentile}th_percentile"] = len(selected)
        
        # Feature selection by standard deviations
        if stats["std"] > 0:
            mean_val = stats["mean"]
            std_val = stats["std"]
            
            for std_multiplier in [1, 2]:
                if self.direction == ImportanceDirection.HIGHER_BETTER:
                    threshold = mean_val + std_multiplier * std_val
                else:
                    threshold = mean_val - std_multiplier * std_val
                
                selected = self.get_features_above_threshold(threshold)
                summary[f"features_above_{std_multiplier}_std"] = len(selected)
        
        # Effective features (if threshold is set)
        if self.selection_threshold is not None:
            selected = self.get_features_above_threshold(self.selection_threshold)
            summary["selected_features_count"] = len(selected)
            summary["selection_rate"] = len(selected) / total_features
        
        return summary
    
    @classmethod
    def from_sklearn_importance(cls, feature_names: list[str], importances: Any,
                              method: ImportanceMethod = ImportanceMethod.RANDOM_FOREST,
                              **kwargs: Any) -> FeatureImportance:
        """Create FeatureImportance from sklearn feature importance."""
        import numpy as np
        
        if len(feature_names) != len(importances):
            raise ValueError("Feature names and importances must have same length")
        
        importances_list = [float(imp) for imp in importances]
        
        # Calculate rankings
        importance_pairs = list(zip(feature_names, importances_list))
        sorted_pairs = sorted(importance_pairs, key=lambda x: x[1], reverse=True)
        rankings = [0] * len(feature_names)
        
        for rank, (feature, _) in enumerate(sorted_pairs, 1):
            idx = feature_names.index(feature)
            rankings[idx] = rank
        
        # Calculate statistics
        importances_array = np.array(importances_list)
        
        return cls(
            features=feature_names,
            importances=importances_list,
            method=method,
            rankings=rankings,
            importance_sum=float(np.sum(importances_array)),
            importance_mean=float(np.mean(importances_array)),
            importance_std=float(np.std(importances_array)),
            feature_count=len(feature_names),
            normalized=False,
            **kwargs
        )