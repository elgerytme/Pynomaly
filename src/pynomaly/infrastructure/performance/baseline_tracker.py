"""
Performance Baseline Tracking System.

Manages historical performance data, trend analysis, and adaptive baseline updates
for comprehensive performance regression detection in CI/CD pipelines.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Configuration for baseline tracking."""
    
    min_samples: int = 20
    outlier_threshold: float = 3.0
    trend_window_days: int = 30
    baseline_update_threshold: float = 0.15  # 15% change triggers update
    confidence_level: float = 0.95
    
    
@dataclass
class TrendAnalysis:
    """Results of trend analysis."""
    
    metric_name: str
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_strength: float  # 0-1 scale
    slope: float
    confidence: float
    data_points: int
    analysis_period_days: int


class PerformanceDatabase:
    """SQLite database for storing performance history."""
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    test_run_id TEXT,
                    environment TEXT,
                    tags TEXT,
                    context TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT UNIQUE NOT NULL,
                    mean REAL NOT NULL,
                    std REAL NOT NULL,
                    p50 REAL NOT NULL,
                    p95 REAL NOT NULL,
                    p99 REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    established_at DATETIME NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    environment TEXT,
                    version INTEGER DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baseline_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    baseline_data TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    reason TEXT
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON performance_metrics(metric_name, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_baselines_name ON baselines(metric_name)")
            
            conn.commit()
    
    def store_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Store performance metrics in the database."""
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (metric_name, value, unit, timestamp, test_run_id, environment, tags, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.get('name'),
                    metric.get('value'),
                    metric.get('unit'),
                    metric.get('timestamp'),
                    metric.get('test_run_id'),
                    json.dumps(metric.get('environment', {})),
                    json.dumps(metric.get('tags', [])),
                    json.dumps(metric.get('context', {}))
                ))
            conn.commit()
    
    def get_metric_history(self, metric_name: str, days: int = 30) -> List[Tuple[datetime, float]]:
        """Get historical values for a metric."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, value 
                FROM performance_metrics 
                WHERE metric_name = ? AND timestamp >= ?
                ORDER BY timestamp
            """, (metric_name, cutoff_date.isoformat()))
            
            return [(datetime.fromisoformat(row[0]), row[1]) for row in cursor.fetchall()]
    
    def store_baseline(self, baseline_data: Dict[str, Any]) -> None:
        """Store or update baseline data."""
        with sqlite3.connect(self.db_path) as conn:
            # Check if baseline exists
            cursor = conn.execute("SELECT version FROM baselines WHERE metric_name = ?", 
                                (baseline_data['metric_name'],))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing baseline
                new_version = existing[0] + 1
                
                # Store old baseline in history
                old_baseline = conn.execute("""
                    SELECT * FROM baselines WHERE metric_name = ?
                """, (baseline_data['metric_name'],)).fetchone()
                
                if old_baseline:
                    conn.execute("""
                        INSERT INTO baseline_history (metric_name, baseline_data, version, reason)
                        VALUES (?, ?, ?, ?)
                    """, (baseline_data['metric_name'], json.dumps(dict(old_baseline)), 
                          existing[0], "automatic_update"))
                
                # Update current baseline
                conn.execute("""
                    UPDATE baselines SET 
                    mean = ?, std = ?, p50 = ?, p95 = ?, p99 = ?,
                    sample_size = ?, updated_at = CURRENT_TIMESTAMP, version = ?
                    WHERE metric_name = ?
                """, (
                    baseline_data['mean'], baseline_data['std'], baseline_data['p50'],
                    baseline_data['p95'], baseline_data['p99'], baseline_data['sample_size'],
                    new_version, baseline_data['metric_name']
                ))
            else:
                # Insert new baseline
                conn.execute("""
                    INSERT INTO baselines 
                    (metric_name, mean, std, p50, p95, p99, sample_size, established_at, environment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    baseline_data['metric_name'], baseline_data['mean'], baseline_data['std'],
                    baseline_data['p50'], baseline_data['p95'], baseline_data['p99'],
                    baseline_data['sample_size'], baseline_data['established_at'],
                    json.dumps(baseline_data.get('environment', {}))
                ))
            
            conn.commit()
    
    def get_baseline(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get current baseline for a metric."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metric_name, mean, std, p50, p95, p99, sample_size, 
                       established_at, environment, version
                FROM baselines WHERE metric_name = ?
            """, (metric_name,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'metric_name': row[0],
                    'mean': row[1],
                    'std': row[2],
                    'p50': row[3],
                    'p95': row[4],
                    'p99': row[5],
                    'sample_size': row[6],
                    'established_at': row[7],
                    'environment': json.loads(row[8]) if row[8] else {},
                    'version': row[9]
                }
        return None


class TrendAnalyzer:
    """Analyzes performance trends over time."""
    
    def __init__(self, db: PerformanceDatabase, config: BaselineConfig):
        self.db = db
        self.config = config
    
    def analyze_trend(self, metric_name: str, days: int = None) -> TrendAnalysis:
        """Analyze trend for a specific metric."""
        if days is None:
            days = self.config.trend_window_days
            
        history = self.db.get_metric_history(metric_name, days)
        
        if len(history) < 5:
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction='insufficient_data',
                trend_strength=0.0,
                slope=0.0,
                confidence=0.0,
                data_points=len(history),
                analysis_period_days=days
            )
        
        # Convert to time series data
        timestamps = [ts.timestamp() for ts, _ in history]
        values = [value for _, value in history]
        
        # Normalize timestamps to days from start
        start_time = min(timestamps)
        x = [(ts - start_time) / 86400 for ts in timestamps]  # Convert to days
        y = np.array(values)
        
        # Remove outliers for more stable trend analysis
        y_clean = self._remove_outliers(y)
        
        # Perform linear regression
        if len(y_clean) < 3:
            y_clean = y  # Use original data if too few points after outlier removal
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_clean)
        
        # Determine trend direction and strength
        trend_direction = self._classify_trend(slope, y_clean)
        trend_strength = min(abs(r_value), 1.0)
        confidence = max(0.0, 1.0 - p_value)
        
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            confidence=confidence,
            data_points=len(history),
            analysis_period_days=days
        )
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using IQR method."""
        if len(data) < 4:
            return data
            
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    def _classify_trend(self, slope: float, values: np.ndarray) -> str:
        """Classify trend direction based on slope and relative magnitude."""
        if len(values) == 0:
            return 'stable'
            
        mean_value = np.mean(values)
        if mean_value == 0:
            return 'stable'
            
        # Calculate relative slope (slope as percentage of mean value)
        relative_slope = abs(slope) / mean_value * 100
        
        if relative_slope < 1.0:  # Less than 1% change per day
            return 'stable'
        elif slope > 0:
            return 'degrading'  # Assuming higher values are worse for performance
        else:
            return 'improving'


class AdaptiveBaselineTracker:
    """Tracks and adaptively updates performance baselines."""
    
    def __init__(self, db_path: str = "performance_tracking.db", 
                 config: BaselineConfig = None):
        self.config = config or BaselineConfig()
        self.db = PerformanceDatabase(Path(db_path))
        self.trend_analyzer = TrendAnalyzer(self.db, self.config)
    
    def record_metrics(self, metrics: List[Dict[str, Any]], 
                      test_run_id: str = None) -> None:
        """Record new performance metrics."""
        # Add test run ID if provided
        for metric in metrics:
            if test_run_id:
                metric['test_run_id'] = test_run_id
        
        self.db.store_metrics(metrics)
        
        # Check if baselines need updating
        for metric in metrics:
            self._check_baseline_update(metric['name'])
    
    def establish_baseline(self, metric_name: str, force: bool = False) -> bool:
        """Establish or update baseline for a metric."""
        history = self.db.get_metric_history(metric_name, days=90)  # 3 months of data
        
        if len(history) < self.config.min_samples:
            logger.warning(f"Insufficient data for baseline: {metric_name} "
                          f"({len(history)} < {self.config.min_samples})")
            return False
        
        # Extract values and remove outliers
        values = np.array([value for _, value in history])
        clean_values = self._remove_outliers(values)
        
        if len(clean_values) < self.config.min_samples:
            logger.warning(f"Too few clean samples for baseline: {metric_name}")
            return False
        
        # Calculate baseline statistics
        baseline_data = {
            'metric_name': metric_name,
            'mean': float(np.mean(clean_values)),
            'std': float(np.std(clean_values)),
            'p50': float(np.percentile(clean_values, 50)),
            'p95': float(np.percentile(clean_values, 95)),
            'p99': float(np.percentile(clean_values, 99)),
            'sample_size': len(clean_values),
            'established_at': datetime.now().isoformat(),
            'environment': {
                'data_period_days': 90,
                'outliers_removed': len(values) - len(clean_values)
            }
        }
        
        self.db.store_baseline(baseline_data)
        logger.info(f"Established baseline for {metric_name}: "
                   f"mean={baseline_data['mean']:.2f}, "
                   f"std={baseline_data['std']:.2f}")
        return True
    
    def _check_baseline_update(self, metric_name: str) -> None:
        """Check if baseline needs updating based on recent trends."""
        current_baseline = self.db.get_baseline(metric_name)
        if not current_baseline:
            # Try to establish initial baseline
            self.establish_baseline(metric_name)
            return
        
        # Analyze recent trend
        trend = self.trend_analyzer.analyze_trend(metric_name, days=14)  # 2 weeks
        
        # Check if trend indicates significant change
        if (trend.confidence > 0.8 and 
            trend.trend_strength > 0.6 and
            trend.trend_direction in ['improving', 'degrading']):
            
            # Get recent values to check drift
            recent_history = self.db.get_metric_history(metric_name, days=7)
            if len(recent_history) >= 5:
                recent_values = [value for _, value in recent_history]
                recent_mean = np.mean(recent_values)
                
                # Calculate drift from current baseline
                drift = abs(recent_mean - current_baseline['mean']) / current_baseline['std']
                
                if drift > self.config.baseline_update_threshold:
                    logger.info(f"Significant drift detected for {metric_name}. "
                               f"Updating baseline (drift: {drift:.2f} std)")
                    self.establish_baseline(metric_name, force=True)
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using modified Z-score method."""
        if len(data) < 4:
            return data
        
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return data
        
        modified_z_scores = 0.6745 * (data - median) / mad
        return data[np.abs(modified_z_scores) < self.config.outlier_threshold]
    
    def get_baseline_status(self, metric_name: str) -> Dict[str, Any]:
        """Get comprehensive status of a metric's baseline."""
        baseline = self.db.get_baseline(metric_name)
        trend = self.trend_analyzer.analyze_trend(metric_name)
        recent_history = self.db.get_metric_history(metric_name, days=7)
        
        status = {
            'metric_name': metric_name,
            'has_baseline': baseline is not None,
            'baseline_age_days': None,
            'recent_data_points': len(recent_history),
            'trend_analysis': asdict(trend),
            'health_score': 0.0
        }
        
        if baseline:
            established_at = datetime.fromisoformat(baseline['established_at'])
            status['baseline_age_days'] = (datetime.now() - established_at).days
            
            # Calculate health score
            health_score = self._calculate_health_score(baseline, trend, recent_history)
            status['health_score'] = health_score
            status['baseline_summary'] = {
                'mean': baseline['mean'],
                'std': baseline['std'],
                'sample_size': baseline['sample_size'],
                'version': baseline['version']
            }
        
        return status
    
    def _calculate_health_score(self, baseline: Dict[str, Any], 
                               trend: TrendAnalysis, 
                               recent_history: List[Tuple[datetime, float]]) -> float:
        """Calculate a health score (0-1) for the baseline quality."""
        score = 1.0
        
        # Penalize old baselines
        age_days = (datetime.now() - datetime.fromisoformat(baseline['established_at'])).days
        if age_days > 60:
            score *= 0.8
        elif age_days > 120:
            score *= 0.6
        
        # Penalize insufficient recent data
        if len(recent_history) < 5:
            score *= 0.7
        elif len(recent_history) < 10:
            score *= 0.9
        
        # Penalize unstable trends
        if trend.trend_direction == 'degrading' and trend.confidence > 0.7:
            score *= 0.5
        elif trend.trend_strength > 0.8 and trend.confidence > 0.8:
            score *= 0.8  # High volatility
        
        # Penalize small sample size
        if baseline['sample_size'] < 50:
            score *= 0.8
        elif baseline['sample_size'] < 20:
            score *= 0.6
        
        return max(0.0, min(1.0, score))
    
    def get_all_baselines_status(self) -> Dict[str, Any]:
        """Get status summary for all metrics with baselines."""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT metric_name FROM baselines")
            metric_names = [row[0] for row in cursor.fetchall()]
        
        statuses = {}
        health_scores = []
        
        for metric_name in metric_names:
            status = self.get_baseline_status(metric_name)
            statuses[metric_name] = status
            health_scores.append(status['health_score'])
        
        return {
            'total_metrics': len(metric_names),
            'average_health_score': np.mean(health_scores) if health_scores else 0.0,
            'healthy_baselines': len([s for s in health_scores if s >= 0.8]),
            'degraded_baselines': len([s for s in health_scores if s < 0.6]),
            'metrics': statuses
        }


# Example usage and CLI integration
def create_baseline_tracker() -> AdaptiveBaselineTracker:
    """Create a baseline tracker with default configuration."""
    config = BaselineConfig(
        min_samples=20,
        outlier_threshold=3.0,
        trend_window_days=30,
        baseline_update_threshold=0.15,
        confidence_level=0.95
    )
    
    return AdaptiveBaselineTracker(
        db_path="performance_baselines/tracking.db",
        config=config
    )


if __name__ == "__main__":
    import argparse
    
    def main():
        """CLI interface for baseline tracking."""
        parser = argparse.ArgumentParser(description="Performance Baseline Tracker")
        parser.add_argument("--metric", help="Metric name to analyze")
        parser.add_argument("--establish", action="store_true", 
                          help="Establish baseline for metric")
        parser.add_argument("--status", action="store_true",
                          help="Show baseline status")
        parser.add_argument("--all", action="store_true",
                          help="Show all baselines status")
        
        args = parser.parse_args()
        
        tracker = create_baseline_tracker()
        
        if args.establish and args.metric:
            success = tracker.establish_baseline(args.metric)
            print(f"Baseline establishment for {args.metric}: {'Success' if success else 'Failed'}")
        
        elif args.status and args.metric:
            status = tracker.get_baseline_status(args.metric)
            print(json.dumps(status, indent=2, default=str))
        
        elif args.all:
            status = tracker.get_all_baselines_status()
            print(json.dumps(status, indent=2, default=str))
        
        else:
            parser.print_help()
    
    main()