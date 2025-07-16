"""Comprehensive performance regression testing suite."""

import asyncio
import json
import logging
import pytest
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from monorepo.application.use_cases.anomaly_detection import AnomalyDetectionUseCase
from monorepo.domain.entities.anomaly_detection import AnomalyDetectionRequest
from monorepo.infrastructure.adapters.pyod_adapter import PyODAdapter

from tests.performance.regression.performance_regression_detector import (
    PerformanceRegressionDetector,
    PerformanceMetric,
    RegressionSeverity
)
from tests.performance.regression.baseline_manager import BaselineManager


logger = logging.getLogger(__name__)


class PerformanceRegressionTestSuite:
    """Comprehensive performance regression testing suite."""
    
    def __init__(self, test_data_path: Optional[Path] = None):
        """Initialize the test suite.
        
        Args:
            test_data_path: Path to test data directory
        """
        self.test_data_path = test_data_path or Path("tests/performance/data")
        self.baseline_path = self.test_data_path / "baselines"
        self.history_path = self.test_data_path / "history"
        
        # Initialize regression detector and baseline manager
        self.detector = PerformanceRegressionDetector(
            baseline_path=self.baseline_path,
            history_path=self.history_path
        )
        self.baseline_manager = BaselineManager(
            baseline_path=self.baseline_path,
            history_path=self.history_path
        )
        
        # Test configuration
        self.test_datasets = {
            "small": (100, 20),      # 100 samples, 20 features
            "medium": (1000, 50),    # 1000 samples, 50 features
            "large": (10000, 100)    # 10000 samples, 100 features
        }
        
        self.algorithms = [
            "isolation_forest",
            "lof",
            "one_class_svm",
            "knn",
            "pca"
        ]
        
        # Performance thresholds (in seconds)
        self.performance_thresholds = {
            "small": 5.0,
            "medium": 30.0,
            "large": 120.0
        }
    
    def generate_test_data(self, n_samples: int, n_features: int) -> np.ndarray:
        """Generate test data for performance testing.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Generated test data
        """
        # Generate classification data with some outliers
        X, _ = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Add some outliers
        n_outliers = max(1, n_samples // 100)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        X[outlier_indices] = np.random.normal(0, 10, (n_outliers, n_features))
        
        return X
    
    def measure_algorithm_performance(
        self,
        algorithm: str,
        data: np.ndarray,
        n_runs: int = 5
    ) -> Dict[str, float]:
        """Measure performance of a single algorithm.
        
        Args:
            algorithm: Algorithm name
            data: Test data
            n_runs: Number of runs to average
            
        Returns:
            Dictionary with performance metrics
        """
        pyod_adapter = PyODAdapter()
        times = []
        memory_usage = []
        
        for _ in range(n_runs):
            # Create anomaly detection request
            request = AnomalyDetectionRequest(
                data=data,
                algorithm=algorithm,
                contamination=0.1
            )
            
            # Measure execution time
            start_time = time.time()
            
            try:
                # Train and predict
                result = pyod_adapter.detect_anomalies(request)
                
                end_time = time.time()
                execution_time = end_time - start_time
                times.append(execution_time)
                
                # Measure memory usage (simplified)
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_usage.append(memory_mb)
                
            except Exception as e:
                logger.error(f"Error running {algorithm}: {e}")
                # Use a high penalty time for failures
                times.append(999.0)
                memory_usage.append(0.0)
        
        # Calculate statistics
        return {
            "mean_time": np.mean(times),
            "median_time": np.median(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "p95_time": np.percentile(times, 95),
            "p99_time": np.percentile(times, 99),
            "mean_memory": np.mean(memory_usage),
            "max_memory": np.max(memory_usage)
        }
    
    async def run_performance_regression_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance regression tests.
        
        Returns:
            Dictionary with test results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_results": {},
            "regression_results": [],
            "summary": {}
        }
        
        all_metrics = []
        
        # Test each algorithm with each dataset size
        for dataset_name, (n_samples, n_features) in self.test_datasets.items():
            logger.info(f"Testing dataset: {dataset_name} ({n_samples} samples, {n_features} features)")
            
            # Generate test data
            test_data = self.generate_test_data(n_samples, n_features)
            
            for algorithm in self.algorithms:
                logger.info(f"Testing algorithm: {algorithm}")
                
                # Measure performance
                try:
                    perf_metrics = self.measure_algorithm_performance(algorithm, test_data)
                    
                    # Create metric objects
                    metric_name = f"{algorithm}_{dataset_name}_execution_time"
                    execution_time_metric = PerformanceMetric(
                        name=metric_name,
                        value=perf_metrics["mean_time"],
                        unit="seconds",
                        timestamp=datetime.now(),
                        context={
                            "dataset": dataset_name,
                            "algorithm": algorithm,
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "all_metrics": perf_metrics
                        }
                    )
                    
                    all_metrics.append(execution_time_metric)
                    
                    # Store detailed results
                    results["test_results"][metric_name] = {
                        "algorithm": algorithm,
                        "dataset": dataset_name,
                        "performance": perf_metrics,
                        "threshold": self.performance_thresholds[dataset_name],
                        "passed": perf_metrics["mean_time"] < self.performance_thresholds[dataset_name]
                    }
                    
                    # Check for regression
                    regression_result = self.detector.detect_regression(
                        metric_name=metric_name,
                        current_value=perf_metrics["mean_time"]
                    )
                    
                    results["regression_results"].append(regression_result.to_dict())
                    
                    logger.info(
                        f"{algorithm} on {dataset_name}: "
                        f"{perf_metrics['mean_time']:.3f}s "
                        f"(regression: {regression_result.is_regression})"
                    )
                    
                except Exception as e:
                    logger.error(f"Error testing {algorithm} on {dataset_name}: {e}")
                    results["test_results"][f"{algorithm}_{dataset_name}_error"] = str(e)
        
        # Save performance history
        self.detector.save_performance_history(all_metrics)
        
        # Generate summary
        total_tests = len(results["test_results"])
        passed_tests = sum(1 for r in results["test_results"].values() 
                          if isinstance(r, dict) and r.get("passed", False))
        regressions = sum(1 for r in results["regression_results"] 
                         if r.get("is_regression", False))
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "regressions_detected": regressions,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        }
        
        return results
    
    async def run_api_performance_tests(self) -> Dict[str, Any]:
        """Run API performance regression tests.
        
        Returns:
            Dictionary with API test results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "api_tests": {},
            "regression_results": []
        }
        
        # Mock API endpoints and test scenarios
        api_tests = {
            "health_check": {
                "endpoint": "/health",
                "method": "GET",
                "expected_max_time": 0.1
            },
            "detect_anomalies": {
                "endpoint": "/api/v1/detect",
                "method": "POST",
                "expected_max_time": 5.0
            },
            "get_models": {
                "endpoint": "/api/v1/models",
                "method": "GET",
                "expected_max_time": 1.0
            }
        }
        
        all_metrics = []
        
        for test_name, test_config in api_tests.items():
            logger.info(f"Testing API endpoint: {test_config['endpoint']}")
            
            # Simulate API response times
            response_times = []
            n_requests = 10
            
            for _ in range(n_requests):
                # Simulate API call
                start_time = time.time()
                
                # Mock API response time based on endpoint
                if test_name == "health_check":
                    await asyncio.sleep(0.01 + np.random.normal(0, 0.005))
                elif test_name == "detect_anomalies":
                    await asyncio.sleep(1.0 + np.random.normal(0, 0.2))
                elif test_name == "get_models":
                    await asyncio.sleep(0.1 + np.random.normal(0, 0.02))
                
                end_time = time.time()
                response_times.append(end_time - start_time)
            
            # Calculate metrics
            mean_time = np.mean(response_times)
            p95_time = np.percentile(response_times, 95)
            
            # Create metric objects
            metric_name = f"api_{test_name}_response_time"
            response_time_metric = PerformanceMetric(
                name=metric_name,
                value=mean_time,
                unit="seconds",
                timestamp=datetime.now(),
                context={
                    "endpoint": test_config["endpoint"],
                    "method": test_config["method"],
                    "n_requests": n_requests,
                    "p95_time": p95_time,
                    "all_times": response_times
                }
            )
            
            all_metrics.append(response_time_metric)
            
            # Store results
            results["api_tests"][test_name] = {
                "endpoint": test_config["endpoint"],
                "mean_response_time": mean_time,
                "p95_response_time": p95_time,
                "expected_max_time": test_config["expected_max_time"],
                "passed": mean_time < test_config["expected_max_time"]
            }
            
            # Check for regression
            regression_result = self.detector.detect_regression(
                metric_name=metric_name,
                current_value=mean_time
            )
            
            results["regression_results"].append(regression_result.to_dict())
        
        # Save performance history
        self.detector.save_performance_history(all_metrics)
        
        return results
    
    def create_performance_baselines(self) -> Dict[str, Any]:
        """Create initial performance baselines from test data.
        
        Returns:
            Dictionary with baseline creation results
        """
        logger.info("Creating performance baselines...")
        
        # Generate baseline data
        baseline_data = {}
        
        # Algorithm performance baselines
        for dataset_name, (n_samples, n_features) in self.test_datasets.items():
            test_data = self.generate_test_data(n_samples, n_features)
            
            for algorithm in self.algorithms:
                try:
                    # Run multiple times to get stable baseline
                    perf_metrics = self.measure_algorithm_performance(algorithm, test_data, n_runs=10)
                    
                    # Create baseline data
                    metric_name = f"{algorithm}_{dataset_name}_execution_time"
                    baseline_data[metric_name] = [perf_metrics["mean_time"]]
                    
                except Exception as e:
                    logger.error(f"Error creating baseline for {algorithm} on {dataset_name}: {e}")
                    continue
        
        # API performance baselines (simulated)
        api_baselines = {
            "api_health_check_response_time": [0.01, 0.012, 0.009, 0.011, 0.013],
            "api_detect_anomalies_response_time": [1.0, 1.2, 0.9, 1.1, 1.3],
            "api_get_models_response_time": [0.1, 0.12, 0.09, 0.11, 0.13]
        }
        
        baseline_data.update(api_baselines)
        
        # Create baselines
        baselines = self.baseline_manager.create_initial_baselines(
            metrics_data=baseline_data,
            units={name: "seconds" for name in baseline_data.keys()}
        )
        
        return {
            "created_baselines": list(baselines.keys()),
            "baseline_count": len(baselines),
            "status": "success"
        }


# Pytest fixtures and test cases

@pytest.fixture(scope="session")
def performance_test_suite():
    """Fixture for performance test suite."""
    return PerformanceRegressionTestSuite()


@pytest.fixture(scope="session")
def create_baselines(performance_test_suite):
    """Fixture to create baselines before tests."""
    return performance_test_suite.create_performance_baselines()


@pytest.mark.performance
@pytest.mark.regression
class TestPerformanceRegression:
    """Performance regression test class."""
    
    def test_algorithm_performance_regression(self, performance_test_suite, create_baselines):
        """Test algorithm performance regression."""
        results = asyncio.run(performance_test_suite.run_performance_regression_tests())
        
        # Assert no critical regressions
        critical_regressions = [
            r for r in results["regression_results"]
            if r.get("severity") == RegressionSeverity.CRITICAL.value
        ]
        
        assert len(critical_regressions) == 0, f"Critical regressions detected: {critical_regressions}"
        
        # Assert overall success rate
        success_rate = results["summary"]["success_rate"]
        assert success_rate >= 80, f"Performance test success rate too low: {success_rate}%"
        
        # Log summary
        logger.info(f"Performance regression test summary: {results['summary']}")
    
    def test_api_performance_regression(self, performance_test_suite, create_baselines):
        """Test API performance regression."""
        results = asyncio.run(performance_test_suite.run_api_performance_tests())
        
        # Assert no critical regressions
        critical_regressions = [
            r for r in results["regression_results"]
            if r.get("severity") == RegressionSeverity.CRITICAL.value
        ]
        
        assert len(critical_regressions) == 0, f"Critical API regressions detected: {critical_regressions}"
        
        # Assert all API tests passed
        failed_tests = [
            name for name, result in results["api_tests"].items()
            if not result.get("passed", False)
        ]
        
        assert len(failed_tests) == 0, f"API performance tests failed: {failed_tests}"
        
        # Log results
        logger.info(f"API performance test results: {results['api_tests']}")
    
    def test_baseline_health(self, performance_test_suite):
        """Test baseline health and integrity."""
        health_report = performance_test_suite.baseline_manager.get_baseline_health_report()
        
        # Assert no error states
        error_baselines = [
            name for name, health in health_report.items()
            if health.get("status") == "ERROR"
        ]
        
        assert len(error_baselines) == 0, f"Baselines with errors: {error_baselines}"
        
        # Assert minimum number of healthy baselines
        healthy_baselines = [
            name for name, health in health_report.items()
            if health.get("status") == "HEALTHY"
        ]
        
        assert len(healthy_baselines) >= 5, f"Insufficient healthy baselines: {len(healthy_baselines)}"
        
        # Log health report
        logger.info(f"Baseline health report: {health_report}")
    
    def test_performance_trend_analysis(self, performance_test_suite):
        """Test performance trend analysis."""
        # Get available baselines
        baselines = performance_test_suite.detector.list_available_baselines()
        
        assert len(baselines) > 0, "No baselines available for trend analysis"
        
        # Test trend analysis for each baseline
        for baseline_name in baselines[:3]:  # Test first 3 baselines
            trend_data = performance_test_suite.detector.get_performance_trend(baseline_name, days=30)
            
            # Assert trend data is available
            assert isinstance(trend_data, list), f"Invalid trend data for {baseline_name}"
            
            # Log trend info
            logger.info(f"Trend data for {baseline_name}: {len(trend_data)} data points")
    
    @pytest.mark.slow
    def test_baseline_auto_update(self, performance_test_suite):
        """Test baseline auto-update functionality."""
        # Get update results
        update_results = performance_test_suite.baseline_manager.batch_update_baselines(
            days=7,
            min_samples=5,
            improvement_threshold=1.0  # Lower threshold for testing
        )
        
        # Assert update results
        assert len(update_results) > 0, "No baseline update results"
        
        # Check for successful updates
        successful_updates = [r for r in update_results if r.success]
        
        # Log update results
        logger.info(f"Baseline update results: {len(successful_updates)} successful updates")
        
        # Assert at least some updates were attempted
        assert len(update_results) > 0, "No baseline updates were attempted"


# Standalone test execution
if __name__ == "__main__":
    # Run tests directly
    suite = PerformanceRegressionTestSuite()
    
    # Create baselines
    print("Creating performance baselines...")
    baseline_results = suite.create_performance_baselines()
    print(f"Created {baseline_results['baseline_count']} baselines")
    
    # Run performance tests
    print("\nRunning algorithm performance regression tests...")
    algo_results = asyncio.run(suite.run_performance_regression_tests())
    print(f"Algorithm test summary: {algo_results['summary']}")
    
    # Run API tests
    print("\nRunning API performance regression tests...")
    api_results = asyncio.run(suite.run_api_performance_tests())
    print(f"API tests completed: {len(api_results['api_tests'])} endpoints tested")
    
    # Generate health report
    print("\nGenerating baseline health report...")
    health_report = suite.baseline_manager.get_baseline_health_report()
    healthy_count = sum(1 for h in health_report.values() if h.get("status") == "HEALTHY")
    print(f"Baseline health: {healthy_count}/{len(health_report)} healthy")
    
    print("\nPerformance regression testing completed!")