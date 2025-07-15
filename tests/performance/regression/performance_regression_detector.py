#!/usr/bin/env python3
"""
Performance Regression Detection for Multi-Environment Testing - Issue #214

This module provides tools for detecting performance regressions across
different test runs and environments.
"""

import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    unit: str
    timestamp: str
    environment: str
    category: str


@dataclass
class RegressionResult:
    """Performance regression analysis result."""
    test_name: str
    baseline_value: float
    current_value: float
    change_percent: float
    is_regression: bool
    severity: str  # "minor", "moderate", "major", "critical"
    threshold_exceeded: bool


class PerformanceRegressionDetector:
    """Detects performance regressions in test results."""
    
    def __init__(self, threshold: float = 0.15, baseline_file: Optional[Path] = None):
        """
        Initialize regression detector.
        
        Args:
            threshold: Regression threshold percentage (0.15 = 15%)
            baseline_file: Path to baseline performance data
        """
        self.threshold = threshold
        self.baseline_file = baseline_file or Path("tests/performance/baselines/performance_baselines.json")
        self.baselines = self._load_baselines()
    
    def _load_baselines(self) -> Dict[str, float]:
        """Load performance baselines from file."""
        if not self.baseline_file.exists():
            return {}
        
        try:
            with open(self.baseline_file, 'r') as f:
                data = json.load(f)
                return data.get("baselines", {})
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _save_baselines(self, baselines: Dict[str, float]) -> None:
        """Save performance baselines to file."""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        
        baseline_data = {
            "meta": {
                "updated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "description": "Performance baselines for regression detection",
            },
            "baselines": baselines,
        }
        
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
    
    def parse_benchmark_results(self, results_file: Path) -> List[PerformanceMetric]:
        """Parse benchmark results from pytest-benchmark JSON output."""
        metrics = []
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            benchmarks = data.get("benchmarks", [])
            
            for benchmark in benchmarks:
                # Extract test name and performance data
                test_name = benchmark.get("name", "unknown")
                stats = benchmark.get("stats", {})
                
                # Get various performance metrics
                mean_time = stats.get("mean", 0)
                min_time = stats.get("min", 0)
                max_time = stats.get("max", 0)
                stddev = stats.get("stddev", 0)
                
                # Create metrics for different aspects
                metrics.extend([
                    PerformanceMetric(
                        name=f"{test_name}_mean",
                        value=mean_time,
                        unit="seconds",
                        timestamp=datetime.now().isoformat(),
                        environment="current",
                        category="execution_time"
                    ),
                    PerformanceMetric(
                        name=f"{test_name}_min",
                        value=min_time,
                        unit="seconds", 
                        timestamp=datetime.now().isoformat(),
                        environment="current",
                        category="execution_time"
                    ),
                    PerformanceMetric(
                        name=f"{test_name}_max",
                        value=max_time,
                        unit="seconds",
                        timestamp=datetime.now().isoformat(),
                        environment="current",
                        category="execution_time"
                    ),
                ])
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error parsing benchmark results: {e}")
        
        return metrics
    
    def detect_regressions(self, metrics: List[PerformanceMetric]) -> List[RegressionResult]:
        """Detect performance regressions against baselines."""
        regressions = []
        
        for metric in metrics:
            baseline_value = self.baselines.get(metric.name)
            
            if baseline_value is None:
                # No baseline available, save current value as baseline
                self.baselines[metric.name] = metric.value
                continue
            
            # Calculate percentage change
            change_percent = ((metric.value - baseline_value) / baseline_value) * 100
            
            # Determine if this is a regression (performance got worse)
            is_regression = change_percent > (self.threshold * 100)
            
            # Determine severity
            severity = self._calculate_severity(change_percent)
            
            # Check if threshold is exceeded
            threshold_exceeded = abs(change_percent) > (self.threshold * 100)
            
            regression_result = RegressionResult(
                test_name=metric.name,
                baseline_value=baseline_value,
                current_value=metric.value,
                change_percent=change_percent,
                is_regression=is_regression,
                severity=severity,
                threshold_exceeded=threshold_exceeded
            )
            
            regressions.append(regression_result)
        
        return regressions
    
    def _calculate_severity(self, change_percent: float) -> str:
        """Calculate regression severity based on percentage change."""
        abs_change = abs(change_percent)
        
        if abs_change < 5:
            return "minor"
        elif abs_change < 15:
            return "moderate"
        elif abs_change < 30:
            return "major"
        else:
            return "critical"
    
    def generate_regression_report(self, regressions: List[RegressionResult], output_file: Path) -> Dict[str, Any]:
        """Generate comprehensive regression analysis report."""
        # Calculate summary statistics
        total_tests = len(regressions)
        regressions_found = len([r for r in regressions if r.is_regression])
        improvements = len([r for r in regressions if r.change_percent < 0])
        
        # Group by severity
        severity_counts = {}
        for severity in ["minor", "moderate", "major", "critical"]:
            severity_counts[severity] = len([
                r for r in regressions 
                if r.is_regression and r.severity == severity
            ])
        
        # Find worst regressions
        worst_regressions = sorted([
            r for r in regressions if r.is_regression
        ], key=lambda x: x.change_percent, reverse=True)[:5]
        
        # Find best improvements
        best_improvements = sorted([
            r for r in regressions if r.change_percent < 0
        ], key=lambda x: x.change_percent)[:5]
        
        # Create report
        report = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "regression_threshold": self.threshold,
                "total_tests_analyzed": total_tests,
            },
            "summary": {
                "total_tests": total_tests,
                "regressions_found": regressions_found,
                "improvements_found": improvements,
                "regression_rate": (regressions_found / total_tests * 100) if total_tests > 0 else 0,
                "severity_breakdown": severity_counts,
            },
            "regressions": [
                {
                    "test_name": r.test_name,
                    "baseline_value": r.baseline_value,
                    "current_value": r.current_value,
                    "change_percent": round(r.change_percent, 2),
                    "severity": r.severity,
                    "threshold_exceeded": r.threshold_exceeded,
                }
                for r in worst_regressions
            ],
            "improvements": [
                {
                    "test_name": r.test_name,
                    "baseline_value": r.baseline_value,
                    "current_value": r.current_value,
                    "change_percent": round(r.change_percent, 2),
                }
                for r in best_improvements
            ],
            "all_results": [
                {
                    "test_name": r.test_name,
                    "baseline_value": r.baseline_value,
                    "current_value": r.current_value,
                    "change_percent": round(r.change_percent, 2),
                    "is_regression": r.is_regression,
                    "severity": r.severity,
                }
                for r in regressions
            ]
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def update_baselines(self, metrics: List[PerformanceMetric], force: bool = False) -> None:
        """Update baseline values with current metrics."""
        updated_count = 0
        
        for metric in metrics:
            if force or metric.name not in self.baselines:
                self.baselines[metric.name] = metric.value
                updated_count += 1
        
        if updated_count > 0:
            self._save_baselines(self.baselines)
            print(f"Updated {updated_count} baseline values")


def main():
    """Main entry point for performance regression detection."""
    parser = argparse.ArgumentParser(
        description="Performance Regression Detector for Multi-Environment Testing"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input benchmark results file (JSON)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("regression-analysis.json"),
        help="Output regression analysis file (default: regression-analysis.json)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Regression threshold percentage (default: 0.15 = 15%)"
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        help="Custom baseline file path"
    )
    parser.add_argument(
        "--update-baselines",
        action="store_true",
        help="Update baseline values with current results"
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Force update all baseline values"
    )
    parser.add_argument(
        "--suite",
        help="Test suite name for categorization"
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PerformanceRegressionDetector(
        threshold=args.threshold,
        baseline_file=args.baseline_file
    )
    
    try:
        # Parse benchmark results
        print(f"Parsing benchmark results from: {args.input}")
        metrics = detector.parse_benchmark_results(args.input)
        print(f"Found {len(metrics)} performance metrics")
        
        if not metrics:
            print("No performance metrics found in input file")
            return
        
        # Detect regressions
        print("Analyzing performance regressions...")
        regressions = detector.detect_regressions(metrics)
        
        # Generate report
        print(f"Generating regression report: {args.output}")
        report = detector.generate_regression_report(regressions, args.output)
        
        # Update baselines if requested
        if args.update_baselines:
            print("Updating performance baselines...")
            detector.update_baselines(metrics, force=args.force_update)
        
        # Print summary
        summary = report["summary"]
        print("\n" + "="*60)
        print("PERFORMANCE REGRESSION ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Tests Analyzed: {summary['total_tests']}")
        print(f"Regressions Found: {summary['regressions_found']}")
        print(f"Improvements Found: {summary['improvements_found']}")
        print(f"Regression Rate: {summary['regression_rate']:.1f}%")
        
        if summary["regressions_found"] > 0:
            print("\nSeverity Breakdown:")
            for severity, count in summary["severity_breakdown"].items():
                if count > 0:
                    print(f"  {severity.capitalize()}: {count}")
        
        # Exit with appropriate code
        if summary["regressions_found"] > 0:
            critical_regressions = summary["severity_breakdown"].get("critical", 0)
            major_regressions = summary["severity_breakdown"].get("major", 0)
            
            if critical_regressions > 0:
                print(f"\n❌ CRITICAL: {critical_regressions} critical regressions detected!")
                exit(2)
            elif major_regressions > 0:
                print(f"\n⚠️ WARNING: {major_regressions} major regressions detected!")
                exit(1)
            else:
                print(f"\n⚠️ {summary['regressions_found']} minor/moderate regressions detected")
                exit(0)
        else:
            print("\n✅ No performance regressions detected!")
            exit(0)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()