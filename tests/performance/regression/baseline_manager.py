"""Performance baseline management utilities."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .performance_regression_detector import (
    PerformanceBaseline, 
    PerformanceMetric,
    PerformanceRegressionDetector
)


@dataclass
class BaselineUpdateResult:
    """Result of baseline update operation."""
    
    metric_name: str
    success: bool
    old_baseline: Optional[PerformanceBaseline]
    new_baseline: Optional[PerformanceBaseline]
    sample_count: int
    improvement_percent: Optional[float] = None
    error: Optional[str] = None


class BaselineManager:
    """Manages performance baselines with advanced features."""
    
    def __init__(
        self,
        baseline_path: Optional[Path] = None,
        history_path: Optional[Path] = None,
        auto_update_enabled: bool = True,
        auto_update_threshold: int = 100  # samples
    ):
        """Initialize baseline manager.
        
        Args:
            baseline_path: Path to baseline storage
            history_path: Path to performance history
            auto_update_enabled: Whether to enable automatic baseline updates
            auto_update_threshold: Minimum samples for auto-update
        """
        self.baseline_path = baseline_path or Path("tests/performance/data/baselines")
        self.history_path = history_path or Path("tests/performance/data/history")
        self.auto_update_enabled = auto_update_enabled
        self.auto_update_threshold = auto_update_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize regression detector
        self.detector = PerformanceRegressionDetector(
            baseline_path=self.baseline_path,
            history_path=self.history_path
        )
        
        # Ensure directories exist
        self.baseline_path.mkdir(parents=True, exist_ok=True)
        self.history_path.mkdir(parents=True, exist_ok=True)
    
    def create_initial_baselines(
        self,
        metrics_data: Dict[str, List[float]],
        units: Optional[Dict[str, str]] = None
    ) -> Dict[str, PerformanceBaseline]:
        """Create initial baselines for multiple metrics.
        
        Args:
            metrics_data: Dictionary mapping metric names to value lists
            units: Optional dictionary mapping metric names to units
            
        Returns:
            Dictionary mapping metric names to created baselines
        """
        units = units or {}
        baselines = {}
        
        for metric_name, values in metrics_data.items():
            if not values:
                self.logger.warning(f"Skipping {metric_name}: no values provided")
                continue
            
            try:
                unit = units.get(metric_name, "ms")
                baseline = self.detector.create_baseline(
                    metric_name=metric_name,
                    values=values,
                    unit=unit,
                    force_update=True
                )
                baselines[metric_name] = baseline
                self.logger.info(f"Created baseline for {metric_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to create baseline for {metric_name}: {e}")
                continue
        
        return baselines
    
    def update_baseline_from_recent_data(
        self,
        metric_name: str,
        days: int = 7,
        min_samples: int = 20,
        improvement_threshold: float = 5.0
    ) -> BaselineUpdateResult:
        """Update baseline from recent performance data.
        
        Args:
            metric_name: Name of the metric
            days: Number of days to look back
            min_samples: Minimum number of samples required
            improvement_threshold: Minimum improvement percentage to update
            
        Returns:
            Result of baseline update operation
        """
        try:
            # Get current baseline
            old_baseline = self.detector.load_baseline(metric_name)
            
            # Get recent performance data
            trend_data = self.detector.get_performance_trend(metric_name, days)
            
            if len(trend_data) < min_samples:
                return BaselineUpdateResult(
                    metric_name=metric_name,
                    success=False,
                    old_baseline=old_baseline,
                    new_baseline=None,
                    sample_count=len(trend_data),
                    error=f"Insufficient data: {len(trend_data)} samples, need {min_samples}"
                )
            
            # Extract values
            values = [metric.value for metric in trend_data]
            unit = trend_data[0].unit if trend_data else "ms"
            
            # Calculate potential new baseline statistics
            new_mean = sum(values) / len(values)
            
            # Check if update is warranted
            if old_baseline:
                improvement_percent = ((old_baseline.mean - new_mean) / old_baseline.mean) * 100
                
                # Only update if there's significant improvement or if explicitly requested
                if improvement_percent < improvement_threshold:
                    return BaselineUpdateResult(
                        metric_name=metric_name,
                        success=False,
                        old_baseline=old_baseline,
                        new_baseline=None,
                        sample_count=len(values),
                        improvement_percent=improvement_percent,
                        error=f"Insufficient improvement: {improvement_percent:.1f}%, need {improvement_threshold}%"
                    )
            else:
                improvement_percent = None
            
            # Create new baseline
            new_baseline = self.detector.create_baseline(
                metric_name=metric_name,
                values=values,
                unit=unit,
                force_update=True
            )
            
            return BaselineUpdateResult(
                metric_name=metric_name,
                success=True,
                old_baseline=old_baseline,
                new_baseline=new_baseline,
                sample_count=len(values),
                improvement_percent=improvement_percent
            )
            
        except Exception as e:
            self.logger.error(f"Error updating baseline for {metric_name}: {e}")
            return BaselineUpdateResult(
                metric_name=metric_name,
                success=False,
                old_baseline=None,
                new_baseline=None,
                sample_count=0,
                error=str(e)
            )
    
    def batch_update_baselines(
        self,
        metric_names: Optional[List[str]] = None,
        days: int = 7,
        min_samples: int = 20,
        improvement_threshold: float = 5.0
    ) -> List[BaselineUpdateResult]:
        """Update multiple baselines from recent data.
        
        Args:
            metric_names: List of metric names to update (None for all)
            days: Number of days to look back
            min_samples: Minimum number of samples required
            improvement_threshold: Minimum improvement percentage to update
            
        Returns:
            List of baseline update results
        """
        if metric_names is None:
            metric_names = self.detector.list_available_baselines()
        
        results = []
        
        for metric_name in metric_names:
            result = self.update_baseline_from_recent_data(
                metric_name=metric_name,
                days=days,
                min_samples=min_samples,
                improvement_threshold=improvement_threshold
            )
            results.append(result)
        
        # Log summary
        successful_updates = sum(1 for r in results if r.success)
        self.logger.info(f"Updated {successful_updates}/{len(results)} baselines")
        
        return results
    
    def backup_baselines(self, backup_path: Optional[Path] = None) -> Path:
        """Create a backup of all baselines.
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            Path to the created backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.baseline_path / f"baseline_backup_{timestamp}.json"
        
        # Collect all baselines
        baselines_data = {}
        
        for baseline_file in self.baseline_path.glob("*_baseline.json"):
            metric_name = baseline_file.stem.replace("_baseline", "")
            
            try:
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    baselines_data[metric_name] = baseline_data
            except Exception as e:
                self.logger.error(f"Error reading baseline {baseline_file}: {e}")
                continue
        
        # Save backup
        backup_data = {
            "backup_created": datetime.now().isoformat(),
            "baselines": baselines_data
        }
        
        try:
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            self.logger.info(f"Created baseline backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            raise
    
    def restore_baselines(self, backup_path: Path) -> Dict[str, bool]:
        """Restore baselines from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Dictionary mapping metric names to restore success status
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading backup file: {e}")
            raise
        
        baselines_data = backup_data.get("baselines", {})
        restore_results = {}
        
        for metric_name, baseline_data in baselines_data.items():
            try:
                baseline_file = self.baseline_path / f"{metric_name}_baseline.json"
                
                with open(baseline_file, 'w') as f:
                    json.dump(baseline_data, f, indent=2)
                
                restore_results[metric_name] = True
                self.logger.info(f"Restored baseline for {metric_name}")
                
            except Exception as e:
                self.logger.error(f"Error restoring baseline for {metric_name}: {e}")
                restore_results[metric_name] = False
        
        successful_restores = sum(1 for success in restore_results.values() if success)
        self.logger.info(f"Restored {successful_restores}/{len(restore_results)} baselines")
        
        return restore_results
    
    def get_baseline_health_report(self) -> Dict[str, Dict[str, Any]]:
        """Generate a health report for all baselines.
        
        Returns:
            Dictionary with health information for each baseline
        """
        health_report = {}
        baseline_names = self.detector.list_available_baselines()
        
        for metric_name in baseline_names:
            baseline = self.detector.load_baseline(metric_name)
            
            if baseline is None:
                health_report[metric_name] = {
                    "status": "ERROR",
                    "error": "Failed to load baseline"
                }
                continue
            
            # Calculate baseline age
            age_days = (datetime.now() - baseline.last_updated).days
            
            # Get recent data points
            recent_data = self.detector.get_performance_trend(metric_name, days=7)
            
            # Determine health status
            if age_days > 30:
                status = "STALE"
            elif baseline.sample_count < 10:
                status = "INSUFFICIENT_DATA"
            elif len(recent_data) < 5:
                status = "LOW_ACTIVITY"
            else:
                status = "HEALTHY"
            
            health_report[metric_name] = {
                "status": status,
                "age_days": age_days,
                "sample_count": baseline.sample_count,
                "recent_data_points": len(recent_data),
                "last_updated": baseline.last_updated.isoformat(),
                "baseline_value": baseline.mean,
                "unit": baseline.unit,
                "std_dev": baseline.std_dev
            }
        
        return health_report
    
    def cleanup_old_baselines(self, days: int = 90) -> List[str]:
        """Remove old baseline files.
        
        Args:
            days: Remove baselines older than this many days
            
        Returns:
            List of removed baseline names
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        removed_baselines = []
        
        for baseline_file in self.baseline_path.glob("*_baseline.json"):
            try:
                # Get file modification time
                mod_time = datetime.fromtimestamp(baseline_file.stat().st_mtime)
                
                if mod_time < cutoff_date:
                    metric_name = baseline_file.stem.replace("_baseline", "")
                    baseline_file.unlink()
                    removed_baselines.append(metric_name)
                    self.logger.info(f"Removed old baseline: {metric_name}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {baseline_file}: {e}")
                continue
        
        return removed_baselines
    
    def export_baselines_summary(self, output_file: Optional[Path] = None) -> Path:
        """Export a summary of all baselines to a file.
        
        Args:
            output_file: Path for the output file
            
        Returns:
            Path to the created summary file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.baseline_path / f"baselines_summary_{timestamp}.json"
        
        summary_data = {
            "export_timestamp": datetime.now().isoformat(),
            "baselines": []
        }
        
        baseline_names = self.detector.list_available_baselines()
        
        for metric_name in baseline_names:
            baseline = self.detector.load_baseline(metric_name)
            
            if baseline:
                summary_data["baselines"].append({
                    "metric_name": metric_name,
                    "mean": baseline.mean,
                    "std_dev": baseline.std_dev,
                    "median": baseline.median,
                    "p95": baseline.p95,
                    "p99": baseline.p99,
                    "sample_count": baseline.sample_count,
                    "unit": baseline.unit,
                    "age_days": (datetime.now() - baseline.last_updated).days
                })
        
        try:
            with open(output_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            self.logger.info(f"Exported baselines summary to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error exporting summary: {e}")
            raise
    
    def validate_baseline_integrity(self) -> Dict[str, List[str]]:
        """Validate the integrity of all baseline files.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": [],
            "invalid": [],
            "missing": []
        }
        
        # Check all baseline files
        for baseline_file in self.baseline_path.glob("*_baseline.json"):
            metric_name = baseline_file.stem.replace("_baseline", "")
            
            try:
                baseline = self.detector.load_baseline(metric_name)
                
                if baseline is None:
                    validation_results["invalid"].append(metric_name)
                elif baseline.sample_count <= 0 or baseline.mean <= 0:
                    validation_results["invalid"].append(metric_name)
                else:
                    validation_results["valid"].append(metric_name)
                    
            except Exception as e:
                self.logger.error(f"Validation error for {metric_name}: {e}")
                validation_results["invalid"].append(metric_name)
        
        # Check for missing baselines based on recent history
        recent_metrics = set()
        for history_file in self.history_path.glob("performance_history_*.json"):
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                for run_data in history_data:
                    for metric_data in run_data["metrics"]:
                        recent_metrics.add(metric_data["name"])
                        
            except Exception as e:
                self.logger.error(f"Error reading history file {history_file}: {e}")
                continue
        
        # Find metrics with history but no baseline
        existing_baselines = set(validation_results["valid"] + validation_results["invalid"])
        validation_results["missing"] = list(recent_metrics - existing_baselines)
        
        return validation_results