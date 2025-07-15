"""Statistical Profile entity for advanced statistical analysis."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_entity import BaseEntity


class ProfileType(str, Enum):
    """Types of statistical profiles."""
    
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"
    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"
    DIAGNOSTIC = "diagnostic"
    TEMPORAL = "temporal"
    MULTIVARIATE = "multivariate"
    BAYESIAN = "bayesian"


class ProfileScope(str, Enum):
    """Scope of statistical analysis."""
    
    DATASET = "dataset"
    FEATURE = "feature"
    SUBSET = "subset"
    COMPARISON = "comparison"
    LONGITUDINAL = "longitudinal"
    CROSS_SECTIONAL = "cross_sectional"


class StatisticalProfile(BaseEntity):
    """Entity representing comprehensive statistical analysis results.
    
    This entity captures detailed statistical insights and analysis results
    for datasets, features, or specific data subsets.
    
    Attributes:
        name: Human-readable name for the profile
        profile_type: Type of statistical analysis performed
        scope: Scope of the analysis (dataset, feature, etc.)
        dataset_id: Reference to the analyzed dataset
        feature_names: List of features included in the analysis
        sample_size: Number of observations analyzed
        analysis_parameters: Configuration used for analysis
        descriptive_statistics: Basic descriptive statistics
        distribution_analysis: Distribution fitting and testing results
        correlation_analysis: Correlation matrices and relationships
        hypothesis_tests: Statistical hypothesis testing results
        outlier_analysis: Outlier detection and analysis
        trend_analysis: Temporal trends and patterns
        quality_assessment: Data quality evaluation results
        recommendations: Generated insights and recommendations
        statistical_significance: Significance levels and p-values
        confidence_intervals: Confidence intervals for estimates
        effect_sizes: Effect size measurements
        assumptions_validation: Statistical assumptions checking
        methodology_notes: Description of methods used
        limitations: Known limitations and caveats
        generated_at: When the profile was generated
        expires_at: When the profile becomes stale
        computation_time_seconds: Time taken to compute profile
        data_snapshot_hash: Hash of data used for reproducibility
        analysis_version: Version of analysis methodology
        quality_score: Overall quality score (0-1)
        completeness_percentage: Percentage of complete analysis
        validation_results: Cross-validation results if applicable
    """
    
    name: str = Field(..., min_length=1, max_length=255)
    profile_type: ProfileType
    scope: ProfileScope
    dataset_id: str = Field(..., min_length=1)
    feature_names: list[str] = Field(default_factory=list)
    sample_size: int = Field(..., gt=0)
    
    # Analysis configuration
    analysis_parameters: dict[str, Any] = Field(default_factory=dict)
    methodology_notes: Optional[str] = Field(None, max_length=2000)
    analysis_version: str = Field(default="1.0.0")
    
    # Statistical results
    descriptive_statistics: dict[str, Any] = Field(default_factory=dict)
    distribution_analysis: dict[str, Any] = Field(default_factory=dict)
    correlation_analysis: dict[str, Any] = Field(default_factory=dict)
    hypothesis_tests: dict[str, Any] = Field(default_factory=dict)
    outlier_analysis: dict[str, Any] = Field(default_factory=dict)
    trend_analysis: dict[str, Any] = Field(default_factory=dict)
    
    # Quality and significance
    quality_assessment: dict[str, Any] = Field(default_factory=dict)
    statistical_significance: dict[str, float] = Field(default_factory=dict)
    confidence_intervals: dict[str, dict[str, float]] = Field(default_factory=dict)
    effect_sizes: dict[str, float] = Field(default_factory=dict)
    assumptions_validation: dict[str, bool] = Field(default_factory=dict)
    
    # Insights and recommendations
    recommendations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    computation_time_seconds: Optional[float] = Field(None, ge=0)
    data_snapshot_hash: Optional[str] = None
    
    # Quality metrics
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    completeness_percentage: float = Field(default=0.0, ge=0, le=100)
    validation_results: dict[str, Any] = Field(default_factory=dict)
    
    @validator('feature_names')
    def validate_feature_names(cls, v: list[str]) -> list[str]:
        """Validate feature names."""
        if len(v) != len(set(v)):
            raise ValueError("Feature names must be unique")
            
        return [name.strip() for name in v if name.strip()]
    
    @validator('recommendations')
    def validate_recommendations(cls, v: list[str]) -> list[str]:
        """Validate recommendations."""
        return [rec.strip() for rec in v if rec.strip()]
    
    @validator('limitations')
    def validate_limitations(cls, v: list[str]) -> list[str]:
        """Validate limitations."""
        return [lim.strip() for lim in v if lim.strip()]
    
    def add_descriptive_statistic(self, name: str, value: Any) -> None:
        """Add a descriptive statistic."""
        self.descriptive_statistics[name] = value
        self._update_completeness()
        self.mark_as_updated()
    
    def add_hypothesis_test(self, test_name: str, statistic: float, 
                          p_value: float, result: dict[str, Any]) -> None:
        """Add hypothesis test results."""
        self.hypothesis_tests[test_name] = {
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
            **result
        }
        
        self.statistical_significance[test_name] = p_value
        self._update_completeness()
        self.mark_as_updated()
    
    def add_correlation(self, feature1: str, feature2: str, 
                       correlation: float, p_value: Optional[float] = None) -> None:
        """Add correlation between features."""
        if "correlations" not in self.correlation_analysis:
            self.correlation_analysis["correlations"] = {}
            
        key = f"{feature1}_{feature2}"
        self.correlation_analysis["correlations"][key] = {
            "correlation": correlation,
            "p_value": p_value,
            "significant": p_value < 0.05 if p_value else None
        }
        
        self._update_completeness()
        self.mark_as_updated()
    
    def add_distribution_fit(self, feature: str, distribution: str, 
                           parameters: dict[str, float], goodness_of_fit: float) -> None:
        """Add distribution fitting results."""
        if "distributions" not in self.distribution_analysis:
            self.distribution_analysis["distributions"] = {}
            
        self.distribution_analysis["distributions"][feature] = {
            "best_fit": distribution,
            "parameters": parameters,
            "goodness_of_fit": goodness_of_fit,
            "fitted_at": datetime.utcnow().isoformat()
        }
        
        self._update_completeness()
        self.mark_as_updated()
    
    def add_outlier_detection(self, method: str, outlier_indices: list[int], 
                            outlier_scores: list[float]) -> None:
        """Add outlier detection results."""
        self.outlier_analysis[method] = {
            "outlier_count": len(outlier_indices),
            "outlier_percentage": len(outlier_indices) / self.sample_size * 100,
            "outlier_indices": outlier_indices,
            "outlier_scores": outlier_scores,
            "detected_at": datetime.utcnow().isoformat()
        }
        
        self._update_completeness()
        self.mark_as_updated()
    
    def add_trend_analysis(self, feature: str, trend_direction: str, 
                          trend_strength: float, seasonality: bool = False) -> None:
        """Add trend analysis results."""
        self.trend_analysis[feature] = {
            "direction": trend_direction,
            "strength": trend_strength,
            "has_seasonality": seasonality,
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
        self._update_completeness()
        self.mark_as_updated()
    
    def add_confidence_interval(self, parameter: str, lower: float, 
                              upper: float, confidence_level: float = 0.95) -> None:
        """Add confidence interval for a parameter."""
        self.confidence_intervals[parameter] = {
            "lower": lower,
            "upper": upper,
            "confidence_level": confidence_level,
            "width": upper - lower
        }
        
        self._update_completeness()
        self.mark_as_updated()
    
    def add_effect_size(self, test_name: str, effect_size: float, 
                       interpretation: Optional[str] = None) -> None:
        """Add effect size measurement."""
        self.effect_sizes[test_name] = effect_size
        
        if interpretation:
            self.metadata[f"{test_name}_effect_interpretation"] = interpretation
            
        self._update_completeness()
        self.mark_as_updated()
    
    def validate_assumption(self, assumption: str, is_valid: bool, 
                          test_details: Optional[dict[str, Any]] = None) -> None:
        """Record statistical assumption validation."""
        self.assumptions_validation[assumption] = is_valid
        
        if test_details:
            self.metadata[f"{assumption}_validation"] = test_details
            
        self._update_completeness()
        self.mark_as_updated()
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add an analysis recommendation."""
        if recommendation.strip() and recommendation not in self.recommendations:
            self.recommendations.append(recommendation.strip())
            self.mark_as_updated()
    
    def add_limitation(self, limitation: str) -> None:
        """Add an analysis limitation."""
        if limitation.strip() and limitation not in self.limitations:
            self.limitations.append(limitation.strip())
            self.mark_as_updated()
    
    def calculate_quality_score(self) -> float:
        """Calculate overall profile quality score."""
        scores = []
        
        # Completeness score
        scores.append(self.completeness_percentage / 100)
        
        # Data size adequacy (larger samples = higher quality)
        size_score = min(1.0, self.sample_size / 1000)  # Max score at 1000+ samples
        scores.append(size_score)
        
        # Statistical significance (higher = better)
        if self.statistical_significance:
            sig_scores = [1 - p for p in self.statistical_significance.values()]
            scores.append(sum(sig_scores) / len(sig_scores))
        
        # Assumption validation (more valid assumptions = higher quality)
        if self.assumptions_validation:
            valid_ratio = sum(self.assumptions_validation.values()) / len(self.assumptions_validation)
            scores.append(valid_ratio)
        
        self.quality_score = sum(scores) / len(scores) if scores else 0.0
        return self.quality_score
    
    def _update_completeness(self) -> None:
        """Update completeness percentage based on available analyses."""
        total_components = 8  # Number of analysis components
        completed = 0
        
        if self.descriptive_statistics:
            completed += 1
        if self.distribution_analysis:
            completed += 1
        if self.correlation_analysis:
            completed += 1
        if self.hypothesis_tests:
            completed += 1
        if self.outlier_analysis:
            completed += 1
        if self.trend_analysis:
            completed += 1
        if self.quality_assessment:
            completed += 1
        if self.assumptions_validation:
            completed += 1
            
        self.completeness_percentage = (completed / total_components) * 100
    
    def is_stale(self) -> bool:
        """Check if profile has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_statistically_significant(self, test_name: str, alpha: float = 0.05) -> bool:
        """Check if a test result is statistically significant."""
        p_value = self.statistical_significance.get(test_name)
        return p_value is not None and p_value < alpha
    
    def get_significant_correlations(self, threshold: float = 0.05) -> dict[str, dict[str, Any]]:
        """Get statistically significant correlations."""
        if "correlations" not in self.correlation_analysis:
            return {}
            
        significant = {}
        for key, result in self.correlation_analysis["correlations"].items():
            if result.get("p_value", 1.0) < threshold:
                significant[key] = result
                
        return significant
    
    def get_profile_summary(self) -> dict[str, Any]:
        """Get a summary of the statistical profile."""
        return {
            "profile_id": str(self.id),
            "name": self.name,
            "profile_type": self.profile_type.value,
            "scope": self.scope.value,
            "sample_size": self.sample_size,
            "feature_count": len(self.feature_names),
            "completeness": self.completeness_percentage,
            "quality_score": self.calculate_quality_score(),
            "significant_tests": len([p for p in self.statistical_significance.values() if p < 0.05]),
            "total_tests": len(self.statistical_significance),
            "has_outliers": bool(self.outlier_analysis),
            "has_trends": bool(self.trend_analysis),
            "recommendations_count": len(self.recommendations),
            "limitations_count": len(self.limitations),
            "generated_at": self.generated_at.isoformat(),
            "is_stale": self.is_stale(),
        }
    
    def validate_invariants(self) -> None:
        """Validate domain invariants."""
        super().validate_invariants()
        
        # Business rule: Feature scope must have feature names
        if self.scope == ProfileScope.FEATURE and not self.feature_names:
            raise ValueError("Feature scope profiles must specify feature names")
        
        # Business rule: Sample size must be adequate for statistical analysis
        if self.sample_size < 3:
            raise ValueError("Sample size too small for statistical analysis")
        
        # Business rule: Quality score must be calculated if profile is complete
        if self.completeness_percentage == 100 and self.quality_score is None:
            self.calculate_quality_score()
        
        # Business rule: Expired profiles should not be used for new analysis
        if self.is_stale() and self.scope != ProfileScope.DATASET:
            raise ValueError("Stale profiles should be regenerated")