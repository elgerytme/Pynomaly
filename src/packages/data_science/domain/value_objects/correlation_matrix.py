"""Correlation Matrix value object for feature correlation analysis."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_value_object import BaseValueObject


class CorrelationType(str, Enum):
    """Types of correlation calculations."""
    
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    PARTIAL = "partial"
    DISTANCE = "distance"
    MUTUAL_INFORMATION = "mutual_information"
    MAXIMAL_INFORMATION = "maximal_information"
    POINT_BISERIAL = "point_biserial"
    POLYCHORIC = "polychoric"
    TETRACHORIC = "tetrachoric"


class CorrelationStrength(str, Enum):
    """Classification of correlation strength."""
    
    VERY_WEAK = "very_weak"      # 0.0 - 0.2
    WEAK = "weak"                # 0.2 - 0.4
    MODERATE = "moderate"        # 0.4 - 0.6
    STRONG = "strong"            # 0.6 - 0.8
    VERY_STRONG = "very_strong"  # 0.8 - 1.0
    PERFECT = "perfect"          # 1.0


class CorrelationMatrix(BaseValueObject):
    """Value object representing correlation analysis between features.
    
    This immutable value object encapsulates correlation matrices,
    significance tests, and feature relationship analysis.
    
    Attributes:
        features: List of feature names
        correlation_matrix: NxN correlation matrix
        correlation_type: Type of correlation calculation used
        
        # Statistical significance
        p_value_matrix: P-values for correlation significance
        confidence_intervals: Confidence intervals for correlations
        significance_level: Alpha level for significance testing
        significant_correlations: Boolean matrix of significant correlations
        
        # Correlation strength analysis
        strong_correlations: Pairs with strong correlation (>threshold)
        weak_correlations: Pairs with weak correlation (<threshold)
        correlation_strengths: Classification of correlation strengths
        average_correlation: Average absolute correlation
        
        # Multicollinearity detection
        vif_scores: Variance Inflation Factor scores
        condition_index: Condition index for multicollinearity
        eigenvalues: Eigenvalues of correlation matrix
        determinant: Determinant of correlation matrix
        
        # Network analysis
        correlation_network: Graph representation of correlations
        clusters: Feature clusters based on correlation
        centrality_scores: Network centrality measures
        community_structure: Community detection results
        
        # Pattern detection
        hierarchical_clustering: Dendrogram of feature relationships
        factor_loadings: Factor analysis loadings
        principal_components: PCA component loadings
        correlation_heatmap_data: Data for visualization
        
        # Time series specific (if applicable)
        lagged_correlations: Correlations at different lags
        cross_correlations: Cross-correlation functions
        lead_lag_relationships: Leading/lagging feature pairs
        
        # Partial correlations
        partial_correlation_matrix: Partial correlations controlling for others
        conditional_independence: Conditional independence relationships
        spurious_correlations: Potentially spurious correlation pairs
        
        # Robustness analysis
        bootstrap_confidence: Bootstrap confidence intervals
        correlation_stability: Stability across different samples
        outlier_sensitivity: Sensitivity to outliers
        sample_size_effect: Effect of sample size on correlations
        
        # Business interpretation
        feature_importance_ranking: Ranking based on correlation patterns
        redundant_features: Features with high redundancy
        uncorrelated_features: Features with low correlations
        correlation_summary: High-level correlation insights
        
        # Quality metrics
        matrix_quality_score: Overall quality of correlation matrix
        missing_data_impact: Impact of missing data on correlations
        heteroscedasticity_test: Test for non-constant variance
        normality_assumptions: Validity of normality assumptions
        
        # Advanced analysis
        distance_matrix: Distance matrix derived from correlations
        similarity_matrix: Similarity matrix
        dissimilarity_matrix: Dissimilarity matrix
        correlation_entropy: Information entropy of correlation patterns
    """
    
    features: list[str] = Field(..., min_items=2)
    correlation_matrix: list[list[float]] = Field(..., min_items=2)
    correlation_type: CorrelationType = CorrelationType.PEARSON
    
    # Statistical significance
    p_value_matrix: Optional[list[list[float]]] = None
    confidence_intervals: Optional[list[list[tuple[float, float]]]] = None
    significance_level: float = Field(default=0.05, gt=0, lt=1)
    significant_correlations: Optional[list[list[bool]]] = None
    
    # Correlation analysis
    strong_correlations: Optional[list[tuple[str, str, float]]] = None
    weak_correlations: Optional[list[tuple[str, str, float]]] = None
    correlation_strengths: Optional[list[list[CorrelationStrength]]] = None
    average_correlation: Optional[float] = Field(None, ge=0, le=1)
    
    # Multicollinearity
    vif_scores: Optional[dict[str, float]] = None
    condition_index: Optional[float] = Field(None, ge=1)
    eigenvalues: Optional[list[float]] = None
    determinant: Optional[float] = Field(None, ge=0, le=1)
    
    # Network analysis
    correlation_network: Optional[dict[str, Any]] = None
    clusters: Optional[list[list[str]]] = None
    centrality_scores: Optional[dict[str, float]] = None
    community_structure: Optional[dict[str, Any]] = None
    
    # Pattern detection
    hierarchical_clustering: Optional[dict[str, Any]] = None
    factor_loadings: Optional[list[list[float]]] = None
    principal_components: Optional[list[list[float]]] = None
    correlation_heatmap_data: Optional[dict[str, Any]] = None
    
    # Time series
    lagged_correlations: Optional[dict[str, list[float]]] = None
    cross_correlations: Optional[dict[str, list[float]]] = None
    lead_lag_relationships: Optional[list[tuple[str, str, int]]] = None
    
    # Partial correlations
    partial_correlation_matrix: Optional[list[list[float]]] = None
    conditional_independence: Optional[list[tuple[str, str, list[str]]]] = None
    spurious_correlations: Optional[list[tuple[str, str, float]]] = None
    
    # Robustness
    bootstrap_confidence: Optional[dict[str, tuple[float, float]]] = None
    correlation_stability: Optional[float] = Field(None, ge=0, le=1)
    outlier_sensitivity: Optional[dict[str, float]] = None
    sample_size_effect: Optional[dict[str, float]] = None
    
    # Business metrics
    feature_importance_ranking: Optional[list[str]] = None
    redundant_features: Optional[list[list[str]]] = None
    uncorrelated_features: Optional[list[str]] = None
    correlation_summary: Optional[dict[str, Any]] = None
    
    # Quality metrics
    matrix_quality_score: Optional[float] = Field(None, ge=0, le=1)
    missing_data_impact: Optional[dict[str, float]] = None
    heteroscedasticity_test: Optional[dict[str, float]] = None
    normality_assumptions: Optional[dict[str, bool]] = None
    
    # Advanced analysis
    distance_matrix: Optional[list[list[float]]] = None
    similarity_matrix: Optional[list[list[float]]] = None
    dissimilarity_matrix: Optional[list[list[float]]] = None
    correlation_entropy: Optional[float] = Field(None, ge=0)
    
    @validator('correlation_matrix')
    def validate_correlation_matrix(cls, v: list[list[float]], values: dict[str, Any]) -> list[list[float]]:
        """Validate correlation matrix properties."""
        features = values.get('features', [])
        n_features = len(features)
        
        if len(v) != n_features:
            raise ValueError(f"Correlation matrix must have {n_features} rows")
        
        for i, row in enumerate(v):
            if len(row) != n_features:
                raise ValueError(f"Correlation matrix row {i} must have {n_features} columns")
            
            # Check diagonal is 1 (or close to 1)
            if abs(row[i] - 1.0) > 1e-10:
                raise ValueError(f"Correlation matrix diagonal element [{i},{i}] must be 1.0")
            
            # Check values are in [-1, 1]
            for j, val in enumerate(row):
                if not -1 <= val <= 1:
                    raise ValueError(f"Correlation value [{i},{j}] = {val} must be between -1 and 1")
                
                # Check symmetry
                if i < len(v) and j < len(v[i]) and i != j:
                    if j < len(v) and i < len(v[j]):
                        if abs(val - v[j][i]) > 1e-10:
                            raise ValueError(f"Correlation matrix must be symmetric: [{i},{j}] != [{j},{i}]")
        
        return v
    
    @validator('p_value_matrix')
    def validate_p_value_matrix(cls, v: Optional[list[list[float]]], 
                               values: dict[str, Any]) -> Optional[list[list[float]]]:
        """Validate p-value matrix dimensions and values."""
        if v is not None:
            features = values.get('features', [])
            n_features = len(features)
            
            if len(v) != n_features:
                raise ValueError(f"P-value matrix must have {n_features} rows")
            
            for i, row in enumerate(v):
                if len(row) != n_features:
                    raise ValueError(f"P-value matrix row {i} must have {n_features} columns")
                
                for j, p_val in enumerate(row):
                    if not 0 <= p_val <= 1:
                        raise ValueError(f"P-value [{i},{j}] = {p_val} must be between 0 and 1")
        
        return v
    
    @validator('eigenvalues')
    def validate_eigenvalues(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        """Validate eigenvalues are non-negative."""
        if v is not None:
            for i, eigenval in enumerate(v):
                if eigenval < -1e-10:  # Allow small numerical errors
                    raise ValueError(f"Eigenvalue {i} = {eigenval} must be non-negative")
        return v
    
    def get_correlation(self, feature1: str, feature2: str) -> Optional[float]:
        """Get correlation between two features."""
        try:
            idx1 = self.features.index(feature1)
            idx2 = self.features.index(feature2)
            return self.correlation_matrix[idx1][idx2]
        except (ValueError, IndexError):
            return None
    
    def get_p_value(self, feature1: str, feature2: str) -> Optional[float]:
        """Get p-value for correlation between two features."""
        if self.p_value_matrix is None:
            return None
        
        try:
            idx1 = self.features.index(feature1)
            idx2 = self.features.index(feature2)
            return self.p_value_matrix[idx1][idx2]
        except (ValueError, IndexError):
            return None
    
    def is_significant(self, feature1: str, feature2: str, 
                      alpha: Optional[float] = None) -> Optional[bool]:
        """Check if correlation between features is statistically significant."""
        if alpha is None:
            alpha = self.significance_level
        
        p_value = self.get_p_value(feature1, feature2)
        if p_value is None:
            return None
        
        return p_value < alpha
    
    def get_correlation_strength(self, feature1: str, feature2: str) -> Optional[CorrelationStrength]:
        """Get correlation strength classification between two features."""
        correlation = self.get_correlation(feature1, feature2)
        if correlation is None:
            return None
        
        abs_corr = abs(correlation)
        
        if abs_corr >= 1.0:
            return CorrelationStrength.PERFECT
        elif abs_corr >= 0.8:
            return CorrelationStrength.VERY_STRONG
        elif abs_corr >= 0.6:
            return CorrelationStrength.STRONG
        elif abs_corr >= 0.4:
            return CorrelationStrength.MODERATE
        elif abs_corr >= 0.2:
            return CorrelationStrength.WEAK
        else:
            return CorrelationStrength.VERY_WEAK
    
    def get_highly_correlated_pairs(self, threshold: float = 0.7, 
                                  include_self: bool = False) -> list[tuple[str, str, float]]:
        """Get pairs of features with correlation above threshold."""
        pairs = []
        
        for i, feature1 in enumerate(self.features):
            start_j = 0 if include_self else i + 1
            for j in range(start_j, len(self.features)):
                feature2 = self.features[j]
                correlation = self.correlation_matrix[i][j]
                
                if abs(correlation) >= threshold:
                    pairs.append((feature1, feature2, correlation))
        
        # Sort by absolute correlation descending
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs
    
    def get_feature_correlations(self, feature: str, 
                               exclude_self: bool = True) -> dict[str, float]:
        """Get correlations of one feature with all others."""
        if feature not in self.features:
            raise ValueError(f"Feature '{feature}' not found")
        
        feature_idx = self.features.index(feature)
        correlations = {}
        
        for i, other_feature in enumerate(self.features):
            if exclude_self and other_feature == feature:
                continue
            
            correlation = self.correlation_matrix[feature_idx][i]
            correlations[other_feature] = correlation
        
        return correlations
    
    def get_most_correlated_features(self, feature: str, 
                                   n_top: int = 5) -> list[tuple[str, float]]:
        """Get features most correlated with given feature."""
        correlations = self.get_feature_correlations(feature, exclude_self=True)
        
        # Sort by absolute correlation descending
        sorted_correlations = sorted(correlations.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
        
        return sorted_correlations[:n_top]
    
    def detect_multicollinearity(self, vif_threshold: float = 10.0) -> dict[str, Any]:
        """Detect multicollinearity issues."""
        multicollinearity_info = {
            "has_multicollinearity": False,
            "high_vif_features": [],
            "condition_index_warning": False,
            "determinant_warning": False,
            "problematic_pairs": []
        }
        
        # VIF analysis
        if self.vif_scores:
            high_vif = [(feature, vif) for feature, vif in self.vif_scores.items() 
                       if vif > vif_threshold]
            if high_vif:
                multicollinearity_info["has_multicollinearity"] = True
                multicollinearity_info["high_vif_features"] = high_vif
        
        # Condition index
        if self.condition_index and self.condition_index > 30:
            multicollinearity_info["has_multicollinearity"] = True
            multicollinearity_info["condition_index_warning"] = True
        
        # Determinant
        if self.determinant and self.determinant < 0.1:
            multicollinearity_info["has_multicollinearity"] = True
            multicollinearity_info["determinant_warning"] = True
        
        # High correlation pairs
        high_corr_pairs = self.get_highly_correlated_pairs(threshold=0.9)
        if high_corr_pairs:
            multicollinearity_info["has_multicollinearity"] = True
            multicollinearity_info["problematic_pairs"] = high_corr_pairs
        
        return multicollinearity_info
    
    def get_correlation_summary(self) -> dict[str, Any]:
        """Get comprehensive correlation summary."""
        import numpy as np
        
        # Convert to numpy for easier calculations
        corr_array = np.array(self.correlation_matrix)
        
        # Extract upper triangle (excluding diagonal)
        n = len(self.features)
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(corr_array[i, j])
        
        upper_triangle = np.array(upper_triangle)
        
        summary = {
            "n_features": len(self.features),
            "correlation_type": self.correlation_type.value,
            "total_pairs": len(upper_triangle),
            "statistics": {
                "mean_correlation": float(np.mean(np.abs(upper_triangle))),
                "median_correlation": float(np.median(np.abs(upper_triangle))),
                "std_correlation": float(np.std(upper_triangle)),
                "min_correlation": float(np.min(upper_triangle)),
                "max_correlation": float(np.max(upper_triangle)),
                "range_correlation": float(np.max(upper_triangle) - np.min(upper_triangle))
            }
        }
        
        # Correlation strength distribution
        strength_counts = {strength.value: 0 for strength in CorrelationStrength}
        for corr in upper_triangle:
            abs_corr = abs(corr)
            if abs_corr >= 1.0:
                strength_counts[CorrelationStrength.PERFECT.value] += 1
            elif abs_corr >= 0.8:
                strength_counts[CorrelationStrength.VERY_STRONG.value] += 1
            elif abs_corr >= 0.6:
                strength_counts[CorrelationStrength.STRONG.value] += 1
            elif abs_corr >= 0.4:
                strength_counts[CorrelationStrength.MODERATE.value] += 1
            elif abs_corr >= 0.2:
                strength_counts[CorrelationStrength.WEAK.value] += 1
            else:
                strength_counts[CorrelationStrength.VERY_WEAK.value] += 1
        
        summary["strength_distribution"] = strength_counts
        
        # Positive vs negative correlations
        positive_corr = upper_triangle[upper_triangle > 0]
        negative_corr = upper_triangle[upper_triangle < 0]
        
        summary["correlation_direction"] = {
            "positive_count": len(positive_corr),
            "negative_count": len(negative_corr),
            "zero_count": len(upper_triangle) - len(positive_corr) - len(negative_corr),
            "positive_percentage": (len(positive_corr) / len(upper_triangle)) * 100,
            "negative_percentage": (len(negative_corr) / len(upper_triangle)) * 100
        }
        
        # Quality assessment
        if self.matrix_quality_score:
            summary["quality_score"] = self.matrix_quality_score
        
        # Multicollinearity check
        summary["multicollinearity"] = self.detect_multicollinearity()
        
        return summary
    
    def get_feature_network_centrality(self) -> dict[str, float]:
        """Calculate network centrality based on correlation strengths."""
        centrality = {}
        
        for feature in self.features:
            correlations = self.get_feature_correlations(feature, exclude_self=True)
            
            # Calculate weighted degree centrality
            total_abs_correlation = sum(abs(corr) for corr in correlations.values())
            centrality[feature] = total_abs_correlation / len(correlations)
        
        return centrality
    
    def identify_correlation_clusters(self, threshold: float = 0.5) -> list[list[str]]:
        """Identify clusters of highly correlated features."""
        import numpy as np
        
        # Create adjacency matrix
        n = len(self.features)
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j and abs(self.correlation_matrix[i][j]) >= threshold:
                    adj_matrix[i][j] = 1
        
        # Simple connected components algorithm
        visited = [False] * n
        clusters = []
        
        def dfs(node: int, cluster: list[int]) -> None:
            visited[node] = True
            cluster.append(node)
            for neighbor in range(n):
                if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                    dfs(neighbor, cluster)
        
        for i in range(n):
            if not visited[i]:
                cluster = []
                dfs(i, cluster)
                if len(cluster) > 1:  # Only include clusters with multiple features
                    clusters.append([self.features[idx] for idx in cluster])
        
        return clusters
    
    def calculate_correlation_distance(self) -> list[list[float]]:
        """Calculate distance matrix from correlation matrix."""
        n = len(self.features)
        distance_matrix = []
        
        for i in range(n):
            row = []
            for j in range(n):
                # Distance = 1 - |correlation|
                correlation = self.correlation_matrix[i][j]
                distance = 1 - abs(correlation)
                row.append(distance)
            distance_matrix.append(row)
        
        return distance_matrix
    
    @classmethod
    def from_pandas_corr(cls, correlation_df: Any, correlation_type: CorrelationType = CorrelationType.PEARSON,
                        p_values_df: Optional[Any] = None, **kwargs: Any) -> CorrelationMatrix:
        """Create CorrelationMatrix from pandas correlation DataFrame."""
        import numpy as np
        
        # Extract features and matrix
        features = list(correlation_df.columns)
        correlation_matrix = correlation_df.values.tolist()
        
        # Handle p-values if provided
        p_value_matrix = None
        if p_values_df is not None:
            p_value_matrix = p_values_df.values.tolist()
        
        # Calculate basic statistics
        corr_array = np.array(correlation_matrix)
        upper_triangle = []
        n = len(features)
        
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(corr_array[i, j])
        
        upper_triangle = np.array(upper_triangle)
        average_correlation = float(np.mean(np.abs(upper_triangle)))
        
        # Calculate eigenvalues for multicollinearity analysis
        try:
            eigenvals = np.linalg.eigvals(corr_array)
            eigenvalues = [float(e) for e in eigenvals if e >= 0]
            determinant = float(np.linalg.det(corr_array))
            condition_index = float(np.sqrt(np.max(eigenvals) / np.min(eigenvals))) if np.min(eigenvals) > 1e-10 else None
        except:
            eigenvalues = None
            determinant = None
            condition_index = None
        
        return cls(
            features=features,
            correlation_matrix=correlation_matrix,
            correlation_type=correlation_type,
            p_value_matrix=p_value_matrix,
            average_correlation=average_correlation,
            eigenvalues=eigenvalues,
            determinant=determinant,
            condition_index=condition_index,
            **kwargs
        )
    
    @classmethod
    def from_numpy_array(cls, correlation_array: Any, feature_names: list[str],
                        correlation_type: CorrelationType = CorrelationType.PEARSON,
                        **kwargs: Any) -> CorrelationMatrix:
        """Create CorrelationMatrix from numpy correlation array."""
        import numpy as np
        
        if correlation_array.shape[0] != len(feature_names):
            raise ValueError("Correlation array dimensions must match number of features")
        
        correlation_matrix = correlation_array.tolist()
        
        # Calculate average correlation
        upper_triangle = []
        n = len(feature_names)
        
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(correlation_array[i, j])
        
        average_correlation = float(np.mean(np.abs(upper_triangle)))
        
        return cls(
            features=feature_names,
            correlation_matrix=correlation_matrix,
            correlation_type=correlation_type,
            average_correlation=average_correlation,
            **kwargs
        )