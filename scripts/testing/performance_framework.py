#!/usr/bin/env python3
"""
Performance Testing Framework for Pynomaly

Implements comprehensive performance testing framework for all packages to ensure
performance regressions are detected early and performance baselines are maintained.

Issue: #820 - Implement Performance Testing Framework
"""

import json
import time
import sys
import os
import subprocess
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from memory_profiler import profile
from pympler import tracker
import numpy as np


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: float
    package: str
    test_name: str
    baseline: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class PerformanceResult:
    """Performance test result"""
    test_name: str
    package: str
    metrics: List[PerformanceMetric]
    passed: bool
    error: Optional[str] = None
    duration: float = 0.0


class PerformanceBaseline:
    """Manages performance baselines for packages"""
    
    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline_file = Path(baseline_file)
        self.baselines = self._load_baselines()
    
    def _load_baselines(self) -> Dict[str, Any]:
        """Load performance baselines from file"""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_baselines(self):
        """Save performance baselines to file"""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)
    
    def get_baseline(self, package: str, test_name: str) -> Optional[float]:
        """Get baseline for a specific test"""
        return self.baselines.get(package, {}).get(test_name)
    
    def set_baseline(self, package: str, test_name: str, value: float):
        """Set baseline for a specific test"""
        if package not in self.baselines:
            self.baselines[package] = {}
        self.baselines[package][test_name] = value
    
    def update_baseline(self, metric: PerformanceMetric):
        """Update baseline with new metric"""
        self.set_baseline(metric.package, metric.test_name, metric.value)


class PerformanceMonitor:
    """Monitors system performance during tests"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.memory_tracker = tracker.SummaryTracker()
        self.start_time: Optional[float] = None
        self.metrics: List[PerformanceMetric] = []
    
    def start_monitoring(self, package: str, test_name: str):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.package = package
        self.test_name = test_name
        self.memory_tracker.clear()
        
        # Record initial metrics
        self._record_metric("cpu_percent", self.process.cpu_percent())
        self._record_metric("memory_mb", self.process.memory_info().rss / 1024 / 1024)
    
    def stop_monitoring(self) -> List[PerformanceMetric]:
        """Stop monitoring and return metrics"""
        if self.start_time is None:
            return []
        
        duration = time.time() - self.start_time
        
        # Record final metrics
        self._record_metric("duration_seconds", duration)
        self._record_metric("cpu_percent_final", self.process.cpu_percent())
        self._record_metric("memory_mb_final", self.process.memory_info().rss / 1024 / 1024)
        
        return self.metrics
    
    def _record_metric(self, name: str, value: float, unit: str = ""):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            package=self.package,
            test_name=self.test_name
        )
        self.metrics.append(metric)


class PerformanceTester:
    """Main performance testing class"""
    
    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline = PerformanceBaseline(baseline_file)
        self.monitor = PerformanceMonitor()
        self.results: List[PerformanceResult] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run_performance_test(self, 
                           package: str, 
                           test_func: Callable,
                           test_name: str,
                           iterations: int = 5,
                           threshold_multiplier: float = 1.5) -> PerformanceResult:
        """Run a performance test with multiple iterations"""
        
        self.logger.info(f"Running performance test: {package}.{test_name}")
        
        times = []
        all_metrics = []
        
        try:
            for i in range(iterations):
                self.monitor.start_monitoring(package, test_name)
                
                start_time = time.time()
                test_func()
                duration = time.time() - start_time
                
                times.append(duration)
                metrics = self.monitor.stop_monitoring()
                all_metrics.extend(metrics)
            
            # Calculate statistics
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            # Check against baseline
            baseline = self.baseline.get_baseline(package, test_name)
            passed = True
            
            if baseline is not None:
                threshold = baseline * threshold_multiplier
                if avg_time > threshold:
                    passed = False
                    self.logger.warning(f"Performance regression detected: {avg_time:.3f}s > {threshold:.3f}s")
            else:
                # Set new baseline
                self.baseline.set_baseline(package, test_name, avg_time)
                self.logger.info(f"Set new baseline: {avg_time:.3f}s")
            
            result = PerformanceResult(
                test_name=test_name,
                package=package,
                metrics=all_metrics,
                passed=passed,
                duration=avg_time
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Performance test failed: {str(e)}")
            result = PerformanceResult(
                test_name=test_name,
                package=package,
                metrics=[],
                passed=False,
                error=str(e)
            )
            self.results.append(result)
            return result
    
    def generate_report(self, output_file: str = "performance_report.html"):
        """Generate performance test report"""
        
        # Create performance dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Test Results', fontsize=16)
        
        # Test results summary
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = len(self.results) - passed_tests
        
        axes[0, 0].pie([passed_tests, failed_tests], 
                      labels=['Passed', 'Failed'], 
                      autopct='%1.1f%%',
                      colors=['green', 'red'])
        axes[0, 0].set_title('Test Results Summary')
        
        # Duration distribution
        durations = [r.duration for r in self.results if r.duration > 0]
        if durations:
            axes[0, 1].hist(durations, bins=10, alpha=0.7, color='blue')
            axes[0, 1].set_title('Test Duration Distribution')
            axes[0, 1].set_xlabel('Duration (seconds)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Performance by package
        package_data = {}
        for result in self.results:
            if result.package not in package_data:
                package_data[result.package] = []
            package_data[result.package].append(result.duration)
        
        if package_data:
            packages = list(package_data.keys())
            avg_durations = [statistics.mean(package_data[pkg]) for pkg in packages]
            
            axes[1, 0].bar(packages, avg_durations, color='lightblue')
            axes[1, 0].set_title('Average Duration by Package')
            axes[1, 0].set_xlabel('Package')
            axes[1, 0].set_ylabel('Duration (seconds)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Timeline of tests
        test_times = [r.duration for r in self.results]
        if test_times:
            axes[1, 1].plot(range(len(test_times)), test_times, 'o-', color='purple')
            axes[1, 1].set_title('Test Performance Timeline')
            axes[1, 1].set_xlabel('Test Number')
            axes[1, 1].set_ylabel('Duration (seconds)')
        
        plt.tight_layout()
        plt.savefig(output_file.replace('.html', '.png'))
        plt.close()
        
        # Generate HTML report
        html_content = self._generate_html_report()
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Performance report generated: {output_file}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML performance report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .metric { margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }
                .passed { border-left-color: #28a745; }
                .failed { border-left-color: #dc3545; }
                .summary { display: flex; justify-content: space-around; margin: 20px 0; }
                .summary-item { text-align: center; padding: 20px; background: #f8f9fa; border-radius: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .chart { text-align: center; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Test Report</h1>
                <p>Generated: {timestamp}</p>
            </div>
            
            <div class="summary">
                <div class="summary-item">
                    <h3>{total_tests}</h3>
                    <p>Total Tests</p>
                </div>
                <div class="summary-item">
                    <h3>{passed_tests}</h3>
                    <p>Passed</p>
                </div>
                <div class="summary-item">
                    <h3>{failed_tests}</h3>
                    <p>Failed</p>
                </div>
                <div class="summary-item">
                    <h3>{avg_duration:.3f}s</h3>
                    <p>Average Duration</p>
                </div>
            </div>
            
            <div class="chart">
                <img src="performance_report.png" alt="Performance Charts" style="max-width: 100%;">
            </div>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Package</th>
                    <th>Test</th>
                    <th>Duration</th>
                    <th>Status</th>
                    <th>Error</th>
                </tr>
                {test_rows}
            </table>
            
            <h2>Performance Metrics</h2>
            {metrics_html}
        </body>
        </html>
        """
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        avg_duration = statistics.mean([r.duration for r in self.results if r.duration > 0]) if self.results else 0
        
        # Generate test result rows
        test_rows = ""
        for result in self.results:
            status = "PASSED" if result.passed else "FAILED"
            status_class = "passed" if result.passed else "failed"
            error = result.error or ""
            
            test_rows += f"""
            <tr class="{status_class}">
                <td>{result.package}</td>
                <td>{result.test_name}</td>
                <td>{result.duration:.3f}s</td>
                <td>{status}</td>
                <td>{error}</td>
            </tr>
            """
        
        # Generate metrics HTML
        metrics_html = ""
        for result in self.results:
            for metric in result.metrics:
                metrics_html += f"""
                <div class="metric">
                    <strong>{metric.name}</strong>: {metric.value:.3f} {metric.unit}
                    <br><small>{metric.package}.{metric.test_name}</small>
                </div>
                """
        
        return html.format(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            avg_duration=avg_duration,
            test_rows=test_rows,
            metrics_html=metrics_html
        )
    
    def run_ci_performance_tests(self):
        """Run performance tests in CI/CD environment"""
        
        # Example test functions for critical algorithms
        def test_anomaly_detection_performance():
            """Test anomaly detection algorithm performance"""
            import numpy as np
            from sklearn.ensemble import IsolationForest
            
            # Generate test data
            data = np.random.normal(0, 1, (1000, 10))
            
            # Run anomaly detection
            detector = IsolationForest(contamination=0.1)
            detector.fit(data)
            predictions = detector.predict(data)
            
            return predictions
        
        def test_ml_model_performance():
            """Test ML model performance"""
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            
            # Generate test data
            X = np.random.random((1000, 20))
            y = np.random.randint(0, 2, 1000)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)
            predictions = model.predict(X)
            
            return predictions
        
        def test_data_processing_performance():
            """Test data processing performance"""
            import pandas as pd
            import numpy as np
            
            # Generate test data
            df = pd.DataFrame({
                'feature1': np.random.random(10000),
                'feature2': np.random.random(10000),
                'feature3': np.random.random(10000),
                'target': np.random.randint(0, 2, 10000)
            })
            
            # Process data
            processed = df.groupby('target').agg({
                'feature1': ['mean', 'std'],
                'feature2': ['mean', 'std'],
                'feature3': ['mean', 'std']
            })
            
            return processed
        
        # Run performance tests
        test_cases = [
            ("data.anomaly_detection", test_anomaly_detection_performance, "anomaly_detection_test"),
            ("ai.mlops", test_ml_model_performance, "ml_model_test"),
            ("formal_sciences.mathematics", test_data_processing_performance, "data_processing_test"),
        ]
        
        for package, test_func, test_name in test_cases:
            try:
                self.run_performance_test(package, test_func, test_name)
            except Exception as e:
                self.logger.error(f"Failed to run test {test_name}: {str(e)}")
        
        # Generate report
        self.generate_report()
        
        # Save baselines
        self.baseline.save_baselines()
        
        # Check if any tests failed
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            self.logger.error(f"Performance tests failed: {len(failed_tests)} tests")
            return False
        
        self.logger.info("All performance tests passed")
        return True


def main():
    """Main entry point for performance testing"""
    if len(sys.argv) > 1 and sys.argv[1] == "ci":
        # Run CI performance tests
        tester = PerformanceTester()
        success = tester.run_ci_performance_tests()
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        print("Performance Testing Framework")
        print("Usage: python performance_framework.py [ci]")
        print("  ci: Run CI performance tests")


if __name__ == "__main__":
    main()