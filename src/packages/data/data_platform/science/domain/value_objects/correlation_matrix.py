"""Correlation Matrix value object for feature relationship analysis."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_value_object import BaseValueObject


class CorrelationMatrix(BaseValueObject):
    """Value object representing correlation relationships between features.
    
    This immutable value object encapsulates correlation analysis results
    including various correlation methods, significance testing, and
    relationship insights.
    
    Attributes:
        correlation_data: Matrix of correlation coefficients
        method: Correlation method used (pearson, spearman, kendall)
        feature_names: Names of features in the matrix
        sample_size: Number of observations used for correlation
        p_values: Statistical significance p-values for correlations
        confidence_intervals: Confidence intervals for correlation coefficients
        significant_correlations: Correlations that are statistically significant
        strong_correlations: Correlations above a strength threshold
        correlation_clusters: Groups of highly correlated features
        partial_correlations: Partial correlation coefficients
        distance_matrix: Distance matrix derived from correlations
        hierarchical_clustering: Hierarchical clustering of features
        network_analysis: Network analysis of correlation relationships
        outlier_correlations: Unusual or unexpected correlation patterns
        temporal_stability: How stable correlations are over time
        missing_data_impact: Impact of missing data on correlations
        non_linear_associations: Non-linear relationship measures
        mutual_information_matrix: Mutual information between features
        maximal_information_coefficient: MIC values for non-linear relationships
    """
    
    # Core correlation data
    correlation_data: dict[str, dict[str, float]] = Field(..., min_items=1)
    method: str = Field(..., min_length=1)
    feature_names: list[str] = Field(..., min_items=2)
    sample_size: int = Field(..., gt=0)
    
    # Statistical significance
    p_values: dict[str, dict[str, float]] = Field(default_factory=dict)
    confidence_intervals: dict[str, dict[str, tuple[float, float]]] = Field(default_factory=dict)
    significant_correlations: list[tuple[str, str, float]] = Field(default_factory=list)
    strong_correlations: list[tuple[str, str, float]] = Field(default_factory=list)
    
    # Pattern analysis
    correlation_clusters: dict[str, list[str]] = Field(default_factory=dict)
    partial_correlations: dict[str, dict[str, float]] = Field(default_factory=dict)
    distance_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    hierarchical_clustering: dict[str, Any] = Field(default_factory=dict)
    
    # Advanced analysis
    network_analysis: dict[str, Any] = Field(default_factory=dict)
    outlier_correlations: list[dict[str, Any]] = Field(default_factory=list)
    temporal_stability: dict[str, float] = Field(default_factory=dict)
    missing_data_impact: dict[str, float] = Field(default_factory=dict)
    
    # Non-linear relationships
    non_linear_associations: dict[str, dict[str, float]] = Field(default_factory=dict)
    mutual_information_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    maximal_information_coefficient: dict[str, dict[str, float]] = Field(default_factory=dict)
    
    @validator('correlation_data')
    def validate_correlation_data(cls, v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """Validate correlation matrix data."""
        if not v:
            raise ValueError("Correlation data cannot be empty")
        
        features = list(v.keys())
        
        # Check matrix is square
        for feature in features:
            if len(v[feature]) != len(features):
                raise ValueError("Correlation matrix must be square")
            
            # Check diagonal is 1 (or very close)
            if feature in v[feature]:
                diagonal_value = v[feature][feature]
                if abs(diagonal_value - 1.0) > 1e-6:
                    raise ValueError(f"Diagonal correlation for '{feature}' should be 1.0")
            
            # Check correlation bounds
            for other_feature, correlation in v[feature].items():
                if not -1.0 <= correlation <= 1.0:
                    raise ValueError(
                        f"Correlation between '{feature}' and '{other_feature}' "
                        f"must be between -1 and 1, got {correlation}"
                    )
        
        # Check symmetry
        for feature1 in features:
            for feature2 in features:
                if feature2 in v[feature1] and feature1 in v.get(feature2, {}):
                    corr1 = v[feature1][feature2]
                    corr2 = v[feature2][feature1]
                    if abs(corr1 - corr2) > 1e-6:
                        raise ValueError(
                            f"Correlation matrix must be symmetric: "
                            f"{feature1}-{feature2}: {corr1} != {corr2}"
                        )
        
        return v
    
    @validator('feature_names')
    def validate_feature_names(cls, v: list[str], values: dict[str, Any]) -> list[str]:
        """Validate feature names match correlation data."""
        correlation_data = values.get('correlation_data', {})
        
        if set(v) != set(correlation_data.keys()):
            raise ValueError("Feature names must match correlation data keys")
        
        return v
    
    @validator('method')
    def validate_method(cls, v: str) -> str:
        """Validate correlation method."""
        valid_methods = {
            'pearson', 'spearman', 'kendall', 'partial', 'polychoric',
            'tetrachoric', 'point_biserial', 'rank_biserial'
        }
        
        if v.lower() not in valid_methods:
            # Allow custom methods but validate it's not empty
            if not v.strip():
                raise ValueError("Correlation method cannot be empty")
        
        return v
    
    def get_correlation(self, feature1: str, feature2: str) -> Optional[float]:
        """Get correlation coefficient between two features."""
        if feature1 in self.correlation_data:
            return self.correlation_data[feature1].get(feature2)
        return None
    
    def get_p_value(self, feature1: str, feature2: str) -> Optional[float]:
        """Get p-value for correlation between two features."""
        if feature1 in self.p_values:
            return self.p_values[feature1].get(feature2)
        return None
    
    def is_significant(self, feature1: str, feature2: str, alpha: float = 0.05) -> bool:
        """Check if correlation is statistically significant."""
        p_value = self.get_p_value(feature1, feature2)
        return p_value is not None and p_value < alpha
    
    def get_strong_correlations(self, threshold: float = 0.7, 
                              exclude_diagonal: bool = True) -> list[tuple[str, str, float]]:
        """Get correlations above a strength threshold."""
        strong_corrs = []
        
        for feature1, correlations in self.correlation_data.items():
            for feature2, correlation in correlations.items():
                if exclude_diagonal and feature1 == feature2:
                    continue
                
                if abs(correlation) >= threshold:
                    # Avoid duplicates by ensuring feature1 < feature2
                    if feature1 < feature2:
                        strong_corrs.append((feature1, feature2, correlation))
        
        return sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)
    
    def get_correlations_for_feature(self, feature_name: str, 
                                   sort_by_strength: bool = True) -> dict[str, float]:
        """Get all correlations for a specific feature."""
        if feature_name not in self.correlation_data:
            return {}
        
        correlations = self.correlation_data[feature_name].copy()
        
        if sort_by_strength:
            # Sort by absolute correlation strength
            sorted_items = sorted(correlations.items(), 
                                key=lambda x: abs(x[1]), reverse=True)
            correlations = dict(sorted_items)
        
        return correlations
    
    def get_most_correlated_features(self, feature_name: str, n: int = 5,
                                   exclude_self: bool = True) -> list[tuple[str, float]]:
        """Get the N most correlated features for a given feature."""
        correlations = self.get_correlations_for_feature(feature_name, sort_by_strength=True)
        
        if exclude_self and feature_name in correlations:
            del correlations[feature_name]
        
        return list(correlations.items())[:n]
    
    def get_correlation_clusters(self, threshold: float = 0.8) -> dict[str, list[str]]:
        """Get clusters of highly correlated features."""
        if self.correlation_clusters:
            return self.correlation_clusters
        
        # Simple clustering based on correlation threshold
        clusters = {}
        assigned_features = set()
        cluster_id = 0
        
        for feature1 in self.feature_names:
            if feature1 in assigned_features:
                continue
            
            cluster_features = [feature1]
            assigned_features.add(feature1)
            
            for feature2 in self.feature_names:
                if feature2 not in assigned_features:
                    correlation = self.get_correlation(feature1, feature2)
                    if correlation is not None and abs(correlation) >= threshold:
                        cluster_features.append(feature2)
                        assigned_features.add(feature2)
            
            if len(cluster_features) > 1:
                clusters[f"cluster_{cluster_id}"] = cluster_features
                cluster_id += 1
        
        return clusters
    
    def get_redundant_features(self, threshold: float = 0.95) -> list[tuple[str, str, float]]:
        """Get pairs of features that are highly correlated (potentially redundant)."""
        redundant_pairs = []
        
        for feature1 in self.feature_names:
            for feature2 in self.feature_names:
                if feature1 < feature2:  # Avoid duplicates
                    correlation = self.get_correlation(feature1, feature2)
                    if correlation is not None and abs(correlation) >= threshold:
                        redundant_pairs.append((feature1, feature2, correlation))
        
        return sorted(redundant_pairs, key=lambda x: abs(x[2]), reverse=True)
    
    def get_independence_candidates(self, threshold: float = 0.1) -> list[str]:
        """Get features with low correlations to others (potentially independent)."""
        independent_features = []
        
        for feature in self.feature_names:
            correlations = self.get_correlations_for_feature(feature)
            
            # Calculate mean absolute correlation with other features
            other_correlations = [
                abs(corr) for other_feature, corr in correlations.items()
                if other_feature != feature
            ]
            
            if other_correlations:
                mean_correlation = sum(other_correlations) / len(other_correlations)
                if mean_correlation <= threshold:
                    independent_features.append(feature)
        
        return independent_features
    
    def calculate_vif_indicators(self) -> dict[str, float]:
        """Calculate Variance Inflation Factor indicators from correlation matrix."""
        import numpy as np
        
        # Convert correlation matrix to numpy array
        n_features = len(self.feature_names)
        corr_array = np.zeros((n_features, n_features))
        
        for i, feature1 in enumerate(self.feature_names):
            for j, feature2 in enumerate(self.feature_names):
                corr_array[i, j] = self.get_correlation(feature1, feature2) or 0
        
        try:
            # VIF = diagonal elements of inverse correlation matrix
            inv_corr = np.linalg.inv(corr_array)
            vif_values = {}
            
            for i, feature in enumerate(self.feature_names):
                vif_values[feature] = inv_corr[i, i]
            
            return vif_values
        except np.linalg.LinAlgError:
            # Singular matrix, return empty dict
            return {}
    
    def get_correlation_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of correlation analysis."""
        correlations_list = []
        
        for feature1, correlations in self.correlation_data.items():
            for feature2, correlation in correlations.items():
                if feature1 != feature2:  # Exclude diagonal
                    correlations_list.append(abs(correlation))
        
        import numpy as np
        
        summary = {
            "method": self.method,
            "n_features": len(self.feature_names),
            "sample_size": self.sample_size,
            "correlation_statistics": {
                "mean_abs_correlation": np.mean(correlations_list) if correlations_list else 0,
                "max_correlation": np.max(correlations_list) if correlations_list else 0,
                "min_correlation": np.min(correlations_list) if correlations_list else 0,
                "std_correlation": np.std(correlations_list) if correlations_list else 0,
            },
            "strong_correlations_count": len(self.get_strong_correlations()),
            "redundant_pairs_count": len(self.get_redundant_features()),
            "independent_features_count": len(self.get_independence_candidates()),
        }
        
        # Add significance information if available
        if self.p_values:
            significant_count = 0
            total_tests = 0
            
            for feature1, p_vals in self.p_values.items():
                for feature2, p_val in p_vals.items():
                    if feature1 != feature2:
                        total_tests += 1
                        if p_val < 0.05:
                            significant_count += 1
            
            summary["significance_analysis"] = {
                "significant_correlations": significant_count,
                "total_tests": total_tests,
                "significance_rate": significant_count / total_tests if total_tests > 0 else 0
            }
        
        # Add clustering information if available
        clusters = self.get_correlation_clusters()
        if clusters:
            summary["clustering_analysis"] = {
                "n_clusters": len(clusters),
                "largest_cluster_size": max(len(features) for features in clusters.values()),
                "clustered_features": sum(len(features) for features in clusters.values())
            }
        
        return summary
    
    @classmethod
    def from_pandas_corr(cls, correlation_df: Any, method: str = "pearson",
                        p_values_df: Optional[Any] = None, **kwargs: Any) -> CorrelationMatrix:
        """Create CorrelationMatrix from pandas correlation DataFrame."""
        
        # Convert DataFrame to dictionary format
        correlation_data = {}
        feature_names = list(correlation_df.index)
        
        for feature1 in feature_names:
            correlation_data[feature1] = {}
            for feature2 in feature_names:
                correlation_data[feature1][feature2] = float(correlation_df.loc[feature1, feature2])
        
        # Convert p-values if provided
        p_values = {}
        if p_values_df is not None:
            for feature1 in feature_names:
                p_values[feature1] = {}
                for feature2 in feature_names:
                    p_values[feature1][feature2] = float(p_values_df.loc[feature1, feature2])
        
        return cls(
            correlation_data=correlation_data,
            method=method,
            feature_names=feature_names,
            sample_size=kwargs.get('sample_size', len(correlation_df)),
            p_values=p_values,
            **kwargs
        )
    
    @classmethod
    def from_numpy_array(cls, correlation_array: Any, feature_names: list[str],
                        method: str = "pearson", **kwargs: Any) -> CorrelationMatrix:
        """Create CorrelationMatrix from numpy correlation array."""
        import numpy as np
        
        if not isinstance(correlation_array, np.ndarray):
            correlation_array = np.array(correlation_array)
        
        if correlation_array.shape[0] != correlation_array.shape[1]:
            raise ValueError("Correlation array must be square")
        
        if len(feature_names) != correlation_array.shape[0]:
            raise ValueError("Feature names length must match array dimensions")
        
        # Convert array to dictionary format
        correlation_data = {}
        for i, feature1 in enumerate(feature_names):
            correlation_data[feature1] = {}
            for j, feature2 in enumerate(feature_names):
                correlation_data[feature1][feature2] = float(correlation_array[i, j])
        
        return cls(
            correlation_data=correlation_data,
            method=method,
            feature_names=feature_names,
            sample_size=kwargs.get('sample_size', 100),
            **kwargs
        )