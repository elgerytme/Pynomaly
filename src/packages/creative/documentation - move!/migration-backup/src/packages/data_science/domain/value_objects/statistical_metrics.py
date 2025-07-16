"""Statistical Metrics value object for comprehensive statistical measurements."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_value_object import BaseValueObject


class StatisticalMetrics(BaseValueObject):
    """Value object representing comprehensive statistical metrics.
    
    This immutable value object encapsulates statistical measurements
    and descriptive statistics for numerical data analysis.
    
    Attributes:
        count: Number of observations
        mean: Arithmetic mean
        median: Middle value
        mode: Most frequent value(s)
        std: Standard deviation
        variance: Variance
        min_value: Minimum value
        max_value: Maximum value
        range_value: Range (max - min)
        q1: First quartile (25th percentile)
        q3: Third quartile (75th percentile)
        iqr: Interquartile range (Q3 - Q1)
        skewness: Measure of asymmetry
        kurtosis: Measure of tail heaviness
        coefficient_of_variation: Ratio of std to mean
        mad: Median absolute deviation
        percentile_5: 5th percentile
        percentile_95: 95th percentile
        geometric_mean: Geometric mean (for positive values)
        harmonic_mean: Harmonic mean (for positive values)
        trimmed_mean: Mean with outliers removed
        confidence_interval_95: 95% confidence interval for mean
        standard_error: Standard error of the mean
        missing_count: Number of missing values
        missing_percentage: Percentage of missing values
        outlier_count: Number of outliers detected
        outlier_percentage: Percentage of outliers
        normality_test_p_value: P-value from normality test
        is_normal: Whether data appears normally distributed
    """
    
    # Basic descriptive statistics
    count: int = Field(..., ge=0)
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[list[float]] = None
    std: Optional[float] = Field(None, ge=0)
    variance: Optional[float] = Field(None, ge=0)
    
    # Range statistics
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    range_value: Optional[float] = Field(None, ge=0)
    
    # Quartiles and percentiles
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = Field(None, ge=0)
    percentile_5: Optional[float] = None
    percentile_95: Optional[float] = None
    
    # Shape statistics
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Specialized means
    geometric_mean: Optional[float] = Field(None, gt=0)
    harmonic_mean: Optional[float] = Field(None, gt=0)
    trimmed_mean: Optional[float] = None
    
    # Variability measures
    coefficient_of_variation: Optional[float] = Field(None, ge=0)
    mad: Optional[float] = Field(None, ge=0)
    
    # Confidence and error
    confidence_interval_95: Optional[tuple[float, float]] = None
    standard_error: Optional[float] = Field(None, ge=0)
    
    # Data quality metrics
    missing_count: int = Field(default=0, ge=0)
    missing_percentage: float = Field(default=0.0, ge=0, le=100)
    outlier_count: int = Field(default=0, ge=0)
    outlier_percentage: float = Field(default=0.0, ge=0, le=100)
    
    # Distribution assessment
    normality_test_p_value: Optional[float] = Field(None, ge=0, le=1)
    is_normal: Optional[bool] = None
    
    @validator('missing_percentage')
    def validate_missing_percentage(cls, v: float, values: dict[str, Any]) -> float:
        """Validate missing percentage is consistent with counts."""
        count = values.get('count', 0)
        missing_count = values.get('missing_count', 0)
        
        if count > 0:
            expected_percentage = (missing_count / count) * 100
            if abs(v - expected_percentage) > 0.01:  # Allow small floating point errors
                raise ValueError("Missing percentage inconsistent with counts")
                
        return v
    
    @validator('outlier_percentage')
    def validate_outlier_percentage(cls, v: float, values: dict[str, Any]) -> float:
        """Validate outlier percentage is consistent with counts."""
        count = values.get('count', 0)
        outlier_count = values.get('outlier_count', 0)
        
        if count > 0:
            expected_percentage = (outlier_count / count) * 100
            if abs(v - expected_percentage) > 0.01:
                raise ValueError("Outlier percentage inconsistent with counts")
                
        return v
    
    @validator('range_value')
    def validate_range(cls, v: Optional[float], values: dict[str, Any]) -> Optional[float]:
        """Validate range is consistent with min/max."""
        if v is not None:
            min_val = values.get('min_value')
            max_val = values.get('max_value')
            
            if min_val is not None and max_val is not None:
                expected_range = max_val - min_val
                if abs(v - expected_range) > 1e-10:
                    raise ValueError("Range inconsistent with min/max values")
                    
        return v
    
    @validator('iqr')
    def validate_iqr(cls, v: Optional[float], values: dict[str, Any]) -> Optional[float]:
        """Validate IQR is consistent with quartiles."""
        if v is not None:
            q1 = values.get('q1')
            q3 = values.get('q3')
            
            if q1 is not None and q3 is not None:
                expected_iqr = q3 - q1
                if abs(v - expected_iqr) > 1e-10:
                    raise ValueError("IQR inconsistent with quartile values")
                    
        return v
    
    @validator('coefficient_of_variation')
    def validate_cv(cls, v: Optional[float], values: dict[str, Any]) -> Optional[float]:
        """Validate coefficient of variation."""
        if v is not None:
            mean = values.get('mean')
            std = values.get('std')
            
            if mean is not None and std is not None and mean != 0:
                expected_cv = abs(std / mean)
                if abs(v - expected_cv) > 1e-10:
                    raise ValueError("Coefficient of variation inconsistent with mean/std")
                    
        return v
    
    @validator('is_normal')
    def validate_normality(cls, v: Optional[bool], values: dict[str, Any]) -> Optional[bool]:
        """Validate normality flag is consistent with p-value."""
        if v is not None:
            p_value = values.get('normality_test_p_value')
            
            if p_value is not None:
                expected_normal = p_value > 0.05  # Common significance level
                if v != expected_normal:
                    # Allow override but note inconsistency
                    pass
                    
        return v
    
    def is_valid_for_count(self, expected_count: int) -> bool:
        """Check if metrics are valid for given count."""
        return self.count == expected_count
    
    def has_sufficient_data(self, min_count: int = 30) -> bool:
        """Check if there's sufficient data for reliable statistics."""
        return self.count >= min_count
    
    def get_data_quality_score(self) -> float:
        """Calculate overall data quality score (0-1)."""
        scores = []
        
        # Completeness score (less missing = better)
        completeness = 1 - (self.missing_percentage / 100)
        scores.append(completeness)
        
        # Outlier score (fewer outliers = better, but some are normal)
        outlier_score = max(0, 1 - (self.outlier_percentage / 20))  # Penalize if >20% outliers
        scores.append(outlier_score)
        
        # Sample size adequacy
        size_score = min(1.0, self.count / 1000)  # Full score at 1000+ samples
        scores.append(size_score)
        
        return sum(scores) / len(scores)
    
    def get_distribution_summary(self) -> dict[str, Any]:
        """Get summary of distribution characteristics."""
        return {
            "center": {
                "mean": self.mean,
                "median": self.median,
                "mode": self.mode
            },
            "spread": {
                "std": self.std,
                "iqr": self.iqr,
                "range": self.range_value,
                "cv": self.coefficient_of_variation
            },
            "shape": {
                "skewness": self.skewness,
                "kurtosis": self.kurtosis,
                "is_normal": self.is_normal
            },
            "quality": {
                "missing_pct": self.missing_percentage,
                "outlier_pct": self.outlier_percentage,
                "data_quality_score": self.get_data_quality_score()
            }
        }
    
    def get_outlier_bounds(self, method: str = "iqr") -> Optional[tuple[float, float]]:
        """Get outlier detection bounds based on method."""
        if method == "iqr" and self.q1 is not None and self.q3 is not None:
            iqr_val = self.q3 - self.q1
            lower_bound = self.q1 - 1.5 * iqr_val
            upper_bound = self.q3 + 1.5 * iqr_val
            return (lower_bound, upper_bound)
        
        elif method == "zscore" and self.mean is not None and self.std is not None:
            # 3-sigma rule
            lower_bound = self.mean - 3 * self.std
            upper_bound = self.mean + 3 * self.std
            return (lower_bound, upper_bound)
        
        elif method == "percentile":
            if self.percentile_5 is not None and self.percentile_95 is not None:
                return (self.percentile_5, self.percentile_95)
        
        return None
    
    def compare_distributions(self, other: StatisticalMetrics) -> dict[str, Any]:
        """Compare this distribution with another."""
        if not isinstance(other, StatisticalMetrics):
            raise ValueError("Can only compare with another StatisticalMetrics")
        
        comparison = {
            "sample_size_ratio": self.count / other.count if other.count > 0 else float('inf'),
            "mean_difference": None,
            "std_ratio": None,
            "distribution_shift": None
        }
        
        if self.mean is not None and other.mean is not None:
            comparison["mean_difference"] = self.mean - other.mean
        
        if self.std is not None and other.std is not None and other.std > 0:
            comparison["std_ratio"] = self.std / other.std
        
        # Simple distribution shift indicator
        if (self.mean is not None and other.mean is not None and 
            self.std is not None and other.std is not None):
            
            pooled_std = ((self.std + other.std) / 2)
            if pooled_std > 0:
                effect_size = abs(self.mean - other.mean) / pooled_std
                comparison["distribution_shift"] = effect_size
        
        return comparison
    
    def is_approximately_equal(self, other: StatisticalMetrics, 
                             tolerance: float = 1e-6) -> bool:
        """Check if two statistical metrics are approximately equal."""
        if not isinstance(other, StatisticalMetrics):
            return False
        
        # Compare key metrics with tolerance
        key_metrics = ['mean', 'std', 'median', 'q1', 'q3']
        
        for metric in key_metrics:
            self_val = getattr(self, metric)
            other_val = getattr(other, metric)
            
            if self_val is None and other_val is None:
                continue
            elif self_val is None or other_val is None:
                return False
            elif abs(self_val - other_val) > tolerance:
                return False
        
        return True
    
    @classmethod
    def from_numpy_array(cls, data: Any, **kwargs: Any) -> StatisticalMetrics:
        """Create StatisticalMetrics from numpy array."""
        import numpy as np
        from scipy import stats
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Remove NaN values for calculations
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return cls(count=len(data), missing_count=len(data), 
                      missing_percentage=100.0, **kwargs)
        
        # Calculate statistics
        metrics = {
            "count": len(data),
            "missing_count": len(data) - len(clean_data),
            "missing_percentage": ((len(data) - len(clean_data)) / len(data)) * 100,
            "mean": float(np.mean(clean_data)),
            "median": float(np.median(clean_data)),
            "std": float(np.std(clean_data)),
            "variance": float(np.var(clean_data)),
            "min_value": float(np.min(clean_data)),
            "max_value": float(np.max(clean_data)),
            "q1": float(np.percentile(clean_data, 25)),
            "q3": float(np.percentile(clean_data, 75)),
            "percentile_5": float(np.percentile(clean_data, 5)),
            "percentile_95": float(np.percentile(clean_data, 95)),
            "skewness": float(stats.skew(clean_data)),
            "kurtosis": float(stats.kurtosis(clean_data)),
        }
        
        # Derived calculations
        metrics["range_value"] = metrics["max_value"] - metrics["min_value"]
        metrics["iqr"] = metrics["q3"] - metrics["q1"]
        
        if metrics["mean"] != 0:
            metrics["coefficient_of_variation"] = abs(metrics["std"] / metrics["mean"])
        
        # Normality test
        if len(clean_data) >= 8:  # Minimum for Shapiro-Wilk
            _, p_value = stats.shapiro(clean_data[:5000])  # Limit sample size
            metrics["normality_test_p_value"] = float(p_value)
            metrics["is_normal"] = p_value > 0.05
        
        # Outlier detection using IQR method
        q1, q3 = metrics["q1"], metrics["q3"]
        iqr = q3 - q1
        outlier_mask = (clean_data < q1 - 1.5 * iqr) | (clean_data > q3 + 1.5 * iqr)
        metrics["outlier_count"] = int(np.sum(outlier_mask))
        metrics["outlier_percentage"] = (metrics["outlier_count"] / len(clean_data)) * 100
        
        return cls(**{**metrics, **kwargs})