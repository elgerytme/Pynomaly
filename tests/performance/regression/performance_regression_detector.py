"""Performance regression detection framework."""

import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from scipy import stats


class RegressionSeverity(str, Enum):
    """Severity levels for performance regression."""
    
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    
    name: str
    value: float
    unit: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class PerformanceBaseline:
    """Performance baseline data structure."""
    
    metric_name: str
    mean: float
    std_dev: float
    median: float
    p95: float
    p99: float
    sample_count: int
    unit: str
    created_at: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceBaseline':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


@dataclass
class RegressionResult:
    """Result of regression detection."""
    
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    severity: RegressionSeverity
    is_regression: bool
    confidence_score: float
    statistical_significance: bool
    p_value: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['severity'] = self.severity.value
        return data


class PerformanceRegressionDetector:
    """Detects performance regressions by comparing current metrics with baselines."""
    
    def __init__(
        self,
        baseline_path: Optional[Path] = None,
        history_path: Optional[Path] = None,
        regression_thresholds: Optional[Dict[RegressionSeverity, float]] = None
    ):
        """Initialize the regression detector.
        
        Args:
            baseline_path: Path to baseline storage directory
            history_path: Path to performance history storage
            regression_thresholds: Custom regression thresholds
        """
        self.baseline_path = baseline_path or Path("tests/performance/data/baselines")
        self.history_path = history_path or Path("tests/performance/data/history")
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        self.baseline_path.mkdir(parents=True, exist_ok=True)
        self.history_path.mkdir(parents=True, exist_ok=True)
        
        # Default regression thresholds (percentage increase)
        self.regression_thresholds = regression_thresholds or {
            RegressionSeverity.MINOR: 10.0,      # 10% increase
            RegressionSeverity.MODERATE: 25.0,   # 25% increase
            RegressionSeverity.SEVERE: 50.0,     # 50% increase
            RegressionSeverity.CRITICAL: 100.0   # 100% increase
        }
        
        # Statistical significance threshold
        self.significance_threshold = 0.05
        
        # Minimum sample size for statistical tests
        self.min_sample_size = 10
    
    def create_baseline(
        self,
        metric_name: str,
        values: List[float],
        unit: str = "ms",
        force_update: bool = False
    ) -> PerformanceBaseline:
        """Create or update a performance baseline.
        
        Args:
            metric_name: Name of the metric
            values: List of performance values
            unit: Unit of measurement
            force_update: Force update existing baseline
            
        Returns:
            Created or updated baseline
        """
        if not values:
            raise ValueError("Values list cannot be empty")
        
        # Check if baseline exists
        baseline_file = self.baseline_path / f"{metric_name}_baseline.json"
        
        if baseline_file.exists() and not force_update:
            self.logger.info(f"Baseline for {metric_name} already exists. Use force_update=True to overwrite.")
            return self.load_baseline(metric_name)
        
        # Calculate baseline statistics
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        median = statistics.median(values)
        p95 = np.percentile(values, 95)
        p99 = np.percentile(values, 99)
        
        # Create baseline
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            mean=mean,
            std_dev=std_dev,
            median=median,
            p95=p95,
            p99=p99,
            sample_count=len(values),
            unit=unit,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Save baseline
        self.save_baseline(baseline)
        
        self.logger.info(f"Created baseline for {metric_name}: mean={mean:.2f}{unit}, std={std_dev:.2f}")
        return baseline
    
    def load_baseline(self, metric_name: str) -> Optional[PerformanceBaseline]:
        """Load a performance baseline.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Loaded baseline or None if not found
        """
        baseline_file = self.baseline_path / f"{metric_name}_baseline.json"
        
        if not baseline_file.exists():
            self.logger.warning(f"No baseline found for {metric_name}")
            return None
        
        try:
            with open(baseline_file, 'r') as f:
                data = json.load(f)
                return PerformanceBaseline.from_dict(data)
        except Exception as e:
            self.logger.error(f"Error loading baseline for {metric_name}: {e}")
            return None
    
    def save_baseline(self, baseline: PerformanceBaseline) -> None:
        """Save a performance baseline.
        
        Args:
            baseline: Baseline to save
        """
        baseline_file = self.baseline_path / f"{baseline.metric_name}_baseline.json"
        
        try:
            with open(baseline_file, 'w') as f:
                json.dump(baseline.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving baseline for {baseline.metric_name}: {e}")
            raise
    
    def detect_regression(
        self,
        metric_name: str,
        current_value: float,
        current_values: Optional[List[float]] = None
    ) -> RegressionResult:
        """Detect if a performance regression has occurred.
        
        Args:
            metric_name: Name of the metric
            current_value: Current performance value
            current_values: List of current values for statistical testing
            
        Returns:
            Regression detection result
        """
        # Load baseline
        baseline = self.load_baseline(metric_name)
        
        if baseline is None:
            self.logger.warning(f"No baseline found for {metric_name}, cannot detect regression")
            return RegressionResult(
                metric_name=metric_name,
                current_value=current_value,
                baseline_value=0.0,
                change_percent=0.0,
                severity=RegressionSeverity.NONE,
                is_regression=False,
                confidence_score=0.0,
                statistical_significance=False,
                details={"error": "No baseline available"}
            )
        
        # Calculate change percentage
        change_percent = ((current_value - baseline.mean) / baseline.mean) * 100
        
        # Determine severity
        severity = self._determine_severity(change_percent)
        
        # Check for regression (positive change is usually bad for performance metrics)
        is_regression = change_percent > self.regression_thresholds[RegressionSeverity.MINOR]
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            current_value, baseline, change_percent
        )
        
        # Statistical significance testing
        statistical_significance = False
        p_value = None
        
        if current_values and len(current_values) >= self.min_sample_size:
            # Create baseline distribution for comparison
            baseline_values = np.random.normal(
                baseline.mean, baseline.std_dev, baseline.sample_count
            )
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(current_values, baseline_values)
            statistical_significance = p_value < self.significance_threshold
        
        result = RegressionResult(
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline.mean,
            change_percent=change_percent,
            severity=severity,
            is_regression=is_regression,
            confidence_score=confidence_score,
            statistical_significance=statistical_significance,
            p_value=p_value,
            details={
                "baseline_std_dev": baseline.std_dev,
                "baseline_sample_count": baseline.sample_count,
                "z_score": (current_value - baseline.mean) / baseline.std_dev if baseline.std_dev > 0 else 0,
                "baseline_created": baseline.created_at.isoformat(),
                "baseline_updated": baseline.last_updated.isoformat()
            }
        )
        
        # Log result
        if is_regression:
            self.logger.warning(
                f"Regression detected for {metric_name}: "
                f"{current_value:.2f}{baseline.unit} vs baseline {baseline.mean:.2f}{baseline.unit} "
                f"({change_percent:+.1f}%, severity: {severity.value})"
            )
        else:
            self.logger.info(
                f"No regression for {metric_name}: "
                f"{current_value:.2f}{baseline.unit} vs baseline {baseline.mean:.2f}{baseline.unit} "
                f"({change_percent:+.1f}%)"
            )
        
        return result
    
    def _determine_severity(self, change_percent: float) -> RegressionSeverity:
        """Determine regression severity based on change percentage.
        
        Args:
            change_percent: Percentage change from baseline
            
        Returns:
            Regression severity level
        """
        abs_change = abs(change_percent)
        
        if abs_change >= self.regression_thresholds[RegressionSeverity.CRITICAL]:
            return RegressionSeverity.CRITICAL
        elif abs_change >= self.regression_thresholds[RegressionSeverity.SEVERE]:
            return RegressionSeverity.SEVERE
        elif abs_change >= self.regression_thresholds[RegressionSeverity.MODERATE]:
            return RegressionSeverity.MODERATE
        elif abs_change >= self.regression_thresholds[RegressionSeverity.MINOR]:
            return RegressionSeverity.MINOR
        else:
            return RegressionSeverity.NONE
    
    def _calculate_confidence_score(
        self,
        current_value: float,
        baseline: PerformanceBaseline,
        change_percent: float
    ) -> float:
        """Calculate confidence score for regression detection.
        
        Args:
            current_value: Current performance value
            baseline: Performance baseline
            change_percent: Change percentage
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence on standard deviations from baseline
        if baseline.std_dev == 0:
            return 1.0 if change_percent != 0 else 0.0
        
        z_score = abs(current_value - baseline.mean) / baseline.std_dev
        
        # Convert z-score to confidence (higher z-score = higher confidence)
        confidence = min(z_score / 3.0, 1.0)  # 3-sigma gives max confidence
        
        # Adjust confidence based on sample size
        sample_factor = min(baseline.sample_count / 30.0, 1.0)  # 30 samples gives full confidence
        confidence *= sample_factor
        
        return confidence
    
    def batch_detect_regression(
        self,
        metrics: List[Tuple[str, float]],
        current_values: Optional[Dict[str, List[float]]] = None
    ) -> List[RegressionResult]:
        """Detect regressions for multiple metrics.
        
        Args:
            metrics: List of (metric_name, current_value) tuples
            current_values: Dictionary mapping metric names to lists of current values
            
        Returns:
            List of regression detection results
        """
        results = []
        current_values = current_values or {}
        
        for metric_name, current_value in metrics:
            metric_values = current_values.get(metric_name)
            result = self.detect_regression(metric_name, current_value, metric_values)
            results.append(result)
        
        return results
    
    def save_performance_history(
        self,
        metrics: List[PerformanceMetric],
        test_run_id: Optional[str] = None
    ) -> None:
        """Save performance metrics to history.
        
        Args:
            metrics: List of performance metrics
            test_run_id: Optional test run identifier
        """
        timestamp = datetime.now()
        history_file = self.history_path / f"performance_history_{timestamp.strftime('%Y%m%d')}.json"
        
        # Load existing history or create new
        history_data = []
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading history file: {e}")
                history_data = []
        
        # Add new metrics
        run_data = {
            "timestamp": timestamp.isoformat(),
            "test_run_id": test_run_id or f"run_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            "metrics": [metric.to_dict() for metric in metrics]
        }
        
        history_data.append(run_data)
        
        # Save updated history
        try:
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving history file: {e}")
            raise
    
    def get_performance_trend(
        self,
        metric_name: str,
        days: int = 30
    ) -> List[PerformanceMetric]:
        """Get performance trend for a metric over the specified number of days.
        
        Args:
            metric_name: Name of the metric
            days: Number of days to look back
            
        Returns:
            List of performance metrics for the trend
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        trend_data = []
        
        # Search through history files
        for history_file in self.history_path.glob("performance_history_*.json"):
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                for run_data in history_data:
                    run_timestamp = datetime.fromisoformat(run_data["timestamp"])
                    
                    if run_timestamp >= cutoff_date:
                        for metric_data in run_data["metrics"]:
                            if metric_data["name"] == metric_name:
                                metric = PerformanceMetric.from_dict(metric_data)
                                trend_data.append(metric)
                                
            except Exception as e:
                self.logger.error(f"Error reading history file {history_file}: {e}")
                continue
        
        # Sort by timestamp
        trend_data.sort(key=lambda x: x.timestamp)
        
        return trend_data
    
    def list_available_baselines(self) -> List[str]:
        """List all available performance baselines.
        
        Returns:
            List of metric names with available baselines
        """
        baselines = []
        
        for baseline_file in self.baseline_path.glob("*_baseline.json"):
            metric_name = baseline_file.stem.replace("_baseline", "")
            baselines.append(metric_name)
        
        return sorted(baselines)
    
    def update_baseline_from_history(
        self,
        metric_name: str,
        days: int = 7,
        min_samples: int = 20
    ) -> Optional[PerformanceBaseline]:
        """Update baseline from recent performance history.
        
        Args:
            metric_name: Name of the metric
            days: Number of days to look back
            min_samples: Minimum number of samples required
            
        Returns:
            Updated baseline or None if insufficient data
        """
        # Get recent performance data
        trend_data = self.get_performance_trend(metric_name, days)
        
        if len(trend_data) < min_samples:
            self.logger.warning(
                f"Insufficient data to update baseline for {metric_name}: "
                f"{len(trend_data)} samples, need {min_samples}"
            )
            return None
        
        # Extract values and unit
        values = [metric.value for metric in trend_data]
        unit = trend_data[0].unit if trend_data else "ms"
        
        # Create updated baseline
        return self.create_baseline(metric_name, values, unit, force_update=True)