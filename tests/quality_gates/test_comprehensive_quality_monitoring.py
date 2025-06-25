"""Comprehensive quality monitoring framework for continuous test quality assurance."""

import pytest
import psutil
import time
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import subprocess
import sys
import os
from collections import defaultdict
import warnings


@dataclass
class TestMetrics:
    """Comprehensive test execution metrics."""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime
    error_message: Optional[str] = None
    warnings_count: int = 0
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class QualityMetrics:
    """Overall quality metrics for test suite."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_duration: float
    average_duration: float
    success_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    flaky_tests: List[str]
    slow_tests: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class QualityDatabase:
    """SQLite database for storing quality metrics."""
    
    def __init__(self, db_path: str = "tests/quality_gates/quality_metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration REAL NOT NULL,
                    memory_usage_mb REAL NOT NULL,
                    cpu_usage_percent REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    error_message TEXT,
                    warnings_count INTEGER DEFAULT 0,
                    retry_count INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_tests INTEGER NOT NULL,
                    passed_tests INTEGER NOT NULL,
                    failed_tests INTEGER NOT NULL,
                    skipped_tests INTEGER NOT NULL,
                    error_tests INTEGER NOT NULL,
                    total_duration REAL NOT NULL,
                    average_duration REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    memory_usage_mb REAL NOT NULL,
                    cpu_usage_percent REAL NOT NULL,
                    flaky_tests TEXT,
                    slow_tests TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_test_name ON test_metrics(test_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON test_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON test_metrics(status)")
    
    def store_test_metric(self, metric: TestMetrics):
        """Store individual test metric."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO test_metrics 
                (test_name, status, duration, memory_usage_mb, cpu_usage_percent, 
                 timestamp, error_message, warnings_count, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.test_name, metric.status, metric.duration,
                metric.memory_usage_mb, metric.cpu_usage_percent,
                metric.timestamp.isoformat(), metric.error_message,
                metric.warnings_count, metric.retry_count
            ))
    
    def store_quality_metric(self, metric: QualityMetrics):
        """Store overall quality metric."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO quality_metrics 
                (total_tests, passed_tests, failed_tests, skipped_tests, error_tests,
                 total_duration, average_duration, success_rate, memory_usage_mb,
                 cpu_usage_percent, flaky_tests, slow_tests, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.total_tests, metric.passed_tests, metric.failed_tests,
                metric.skipped_tests, metric.error_tests, metric.total_duration,
                metric.average_duration, metric.success_rate, metric.memory_usage_mb,
                metric.cpu_usage_percent, json.dumps(metric.flaky_tests),
                json.dumps(metric.slow_tests), metric.timestamp.isoformat()
            ))
    
    def get_test_history(self, test_name: str, days: int = 7) -> List[TestMetrics]:
        """Get historical data for a specific test."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT test_name, status, duration, memory_usage_mb, cpu_usage_percent,
                       timestamp, error_message, warnings_count, retry_count
                FROM test_metrics 
                WHERE test_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (test_name, cutoff_date.isoformat()))
            
            results = []
            for row in cursor.fetchall():
                metric = TestMetrics(
                    test_name=row[0], status=row[1], duration=row[2],
                    memory_usage_mb=row[3], cpu_usage_percent=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    error_message=row[6], warnings_count=row[7], retry_count=row[8]
                )
                results.append(metric)
            
            return results
    
    def get_flaky_tests(self, days: int = 7, min_runs: int = 5) -> Dict[str, float]:
        """Identify flaky tests based on failure rate."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT test_name, 
                       COUNT(*) as total_runs,
                       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures
                FROM test_metrics 
                WHERE timestamp >= ?
                GROUP BY test_name
                HAVING total_runs >= ?
            """, (cutoff_date.isoformat(), min_runs))
            
            flaky_tests = {}
            for row in cursor.fetchall():
                test_name, total_runs, failures = row
                failure_rate = failures / total_runs
                if 0 < failure_rate < 1.0:  # Neither always passing nor always failing
                    flaky_tests[test_name] = failure_rate
            
            return flaky_tests


class PerformanceMonitor:
    """Real-time performance monitoring for test execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_thread = None
        self._metrics = []
    
    def start_monitoring(self):
        """Start background performance monitoring."""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated metrics."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        if not self._metrics:
            return {'memory_mb': 0.0, 'cpu_percent': 0.0}
        
        memory_values = [m['memory_mb'] for m in self._metrics]
        cpu_values = [m['cpu_percent'] for m in self._metrics]
        
        return {
            'memory_mb': max(memory_values),  # Peak memory
            'cpu_percent': sum(cpu_values) / len(cpu_values)  # Average CPU
        }
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                self._metrics.append({
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent,
                    'timestamp': datetime.now()
                })
                
                time.sleep(0.1)  # Monitor every 100ms
                
            except Exception:
                break  # Exit on any error


class QualityGate:
    """Quality gate definitions and validation."""
    
    def __init__(self):
        self.gates = {
            'success_rate': {'threshold': 0.95, 'operator': '>='},
            'average_duration': {'threshold': 2.0, 'operator': '<='},
            'flaky_test_rate': {'threshold': 0.05, 'operator': '<='},
            'memory_usage_mb': {'threshold': 500.0, 'operator': '<='},
            'slow_test_rate': {'threshold': 0.10, 'operator': '<='}
        }
    
    def validate(self, metrics: QualityMetrics) -> Dict[str, bool]:
        """Validate metrics against quality gates."""
        results = {}
        
        # Success rate gate
        results['success_rate'] = metrics.success_rate >= self.gates['success_rate']['threshold']
        
        # Average duration gate
        results['average_duration'] = metrics.average_duration <= self.gates['average_duration']['threshold']
        
        # Flaky test rate gate
        flaky_rate = len(metrics.flaky_tests) / max(metrics.total_tests, 1)
        results['flaky_test_rate'] = flaky_rate <= self.gates['flaky_test_rate']['threshold']
        
        # Memory usage gate
        results['memory_usage_mb'] = metrics.memory_usage_mb <= self.gates['memory_usage_mb']['threshold']
        
        # Slow test rate gate
        slow_rate = len(metrics.slow_tests) / max(metrics.total_tests, 1)
        results['slow_test_rate'] = slow_rate <= self.gates['slow_test_rate']['threshold']
        
        return results
    
    def get_gate_summary(self, validation_results: Dict[str, bool]) -> Dict[str, Any]:
        """Get summary of quality gate results."""
        passed_gates = sum(1 for result in validation_results.values() if result)
        total_gates = len(validation_results)
        
        return {
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'failed_gates': total_gates - passed_gates,
            'success_rate': passed_gates / total_gates,
            'all_gates_passed': all(validation_results.values()),
            'individual_results': validation_results
        }


class QualityMonitor:
    """Main quality monitoring coordinator."""
    
    def __init__(self):
        self.db = QualityDatabase()
        self.performance_monitor = PerformanceMonitor()
        self.quality_gate = QualityGate()
        self.warning_counts = defaultdict(int)
        self.test_metrics = []
    
    @contextmanager
    def monitor_test(self, test_name: str):
        """Context manager for monitoring individual test execution."""
        start_time = time.time()
        self.performance_monitor.start_monitoring()
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            try:
                yield
                status = "passed"
                error_message = None
            except pytest.skip.Exception:
                status = "skipped"
                error_message = None
            except Exception as e:
                status = "failed"
                error_message = str(e)
            finally:
                # Calculate metrics
                duration = time.time() - start_time
                perf_metrics = self.performance_monitor.stop_monitoring()
                
                # Create test metric
                metric = TestMetrics(
                    test_name=test_name,
                    status=status,
                    duration=duration,
                    memory_usage_mb=perf_metrics['memory_mb'],
                    cpu_usage_percent=perf_metrics['cpu_percent'],
                    timestamp=datetime.now(),
                    error_message=error_message,
                    warnings_count=len(warning_list),
                    retry_count=0  # Would be set by retry decorators
                )
                
                # Store metric
                self.db.store_test_metric(metric)
                self.test_metrics.append(metric)
    
    def generate_quality_report(self) -> QualityMetrics:
        """Generate comprehensive quality report."""
        if not self.test_metrics:
            return self._empty_quality_metrics()
        
        # Calculate basic statistics
        total_tests = len(self.test_metrics)
        passed_tests = sum(1 for m in self.test_metrics if m.status == "passed")
        failed_tests = sum(1 for m in self.test_metrics if m.status == "failed")
        skipped_tests = sum(1 for m in self.test_metrics if m.status == "skipped")
        error_tests = sum(1 for m in self.test_metrics if m.status == "error")
        
        total_duration = sum(m.duration for m in self.test_metrics)
        average_duration = total_duration / total_tests
        success_rate = passed_tests / total_tests
        
        # Performance metrics
        max_memory = max(m.memory_usage_mb for m in self.test_metrics)
        avg_cpu = sum(m.cpu_usage_percent for m in self.test_metrics) / total_tests
        
        # Identify problematic tests
        flaky_tests = list(self.db.get_flaky_tests().keys())
        slow_tests = [m.test_name for m in self.test_metrics if m.duration > 10.0]
        
        quality_metrics = QualityMetrics(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            total_duration=total_duration,
            average_duration=average_duration,
            success_rate=success_rate,
            memory_usage_mb=max_memory,
            cpu_usage_percent=avg_cpu,
            flaky_tests=flaky_tests,
            slow_tests=slow_tests,
            timestamp=datetime.now()
        )
        
        # Store quality metrics
        self.db.store_quality_metric(quality_metrics)
        
        return quality_metrics
    
    def _empty_quality_metrics(self) -> QualityMetrics:
        """Return empty quality metrics."""
        return QualityMetrics(
            total_tests=0, passed_tests=0, failed_tests=0,
            skipped_tests=0, error_tests=0, total_duration=0.0,
            average_duration=0.0, success_rate=0.0, memory_usage_mb=0.0,
            cpu_usage_percent=0.0, flaky_tests=[], slow_tests=[],
            timestamp=datetime.now()
        )


class TestComprehensiveQualityMonitoring:
    """Test cases for the quality monitoring framework."""
    
    @pytest.fixture
    def quality_monitor(self):
        """Create quality monitor instance."""
        return QualityMonitor()
    
    @pytest.fixture
    def quality_gate(self):
        """Create quality gate instance."""
        return QualityGate()
    
    def test_quality_monitoring_basic_functionality(self, quality_monitor):
        """Test basic quality monitoring functionality."""
        
        # Simulate some test executions
        test_results = [
            ("test_sample_1", "passed", 0.5),
            ("test_sample_2", "passed", 1.0),
            ("test_sample_3", "failed", 2.0),
            ("test_sample_4", "passed", 0.8)
        ]
        
        for test_name, status, duration in test_results:
            metric = TestMetrics(
                test_name=test_name,
                status=status,
                duration=duration,
                memory_usage_mb=50.0,
                cpu_usage_percent=20.0,
                timestamp=datetime.now()
            )
            quality_monitor.test_metrics.append(metric)
        
        # Generate quality report
        report = quality_monitor.generate_quality_report()
        
        # Verify basic statistics
        assert report.total_tests == 4
        assert report.passed_tests == 3
        assert report.failed_tests == 1
        assert report.success_rate == 0.75
        assert report.average_duration == 1.075  # (0.5 + 1.0 + 2.0 + 0.8) / 4
    
    def test_quality_gate_validation(self, quality_gate):
        """Test quality gate validation logic."""
        
        # Good quality metrics (should pass all gates)
        good_metrics = QualityMetrics(
            total_tests=100, passed_tests=98, failed_tests=2,
            skipped_tests=0, error_tests=0, total_duration=150.0,
            average_duration=1.5, success_rate=0.98, memory_usage_mb=200.0,
            cpu_usage_percent=30.0, flaky_tests=["test_flaky_1"],
            slow_tests=["test_slow_1", "test_slow_2"], timestamp=datetime.now()
        )
        
        validation_results = quality_gate.validate(good_metrics)
        gate_summary = quality_gate.get_gate_summary(validation_results)
        
        assert gate_summary['all_gates_passed']
        assert gate_summary['success_rate'] == 1.0
        
        # Poor quality metrics (should fail some gates)
        poor_metrics = QualityMetrics(
            total_tests=100, passed_tests=85, failed_tests=15,
            skipped_tests=0, error_tests=0, total_duration=500.0,
            average_duration=5.0, success_rate=0.85, memory_usage_mb=800.0,
            cpu_usage_percent=80.0, flaky_tests=["test_flaky_" + str(i) for i in range(10)],
            slow_tests=["test_slow_" + str(i) for i in range(20)], timestamp=datetime.now()
        )
        
        validation_results = quality_gate.validate(poor_metrics)
        gate_summary = quality_gate.get_gate_summary(validation_results)
        
        assert not gate_summary['all_gates_passed']
        assert gate_summary['failed_gates'] > 0
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        monitor = PerformanceMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate some work
        time.sleep(0.5)
        
        # Stop monitoring and get metrics
        metrics = monitor.stop_monitoring()
        
        assert 'memory_mb' in metrics
        assert 'cpu_percent' in metrics
        assert metrics['memory_mb'] > 0
        assert metrics['cpu_percent'] >= 0
    
    def test_database_operations(self):
        """Test database storage and retrieval operations."""
        db = QualityDatabase("tests/quality_gates/test_quality_metrics.db")
        
        # Store test metric
        test_metric = TestMetrics(
            test_name="test_database_ops",
            status="passed",
            duration=1.0,
            memory_usage_mb=100.0,
            cpu_usage_percent=25.0,
            timestamp=datetime.now(),
            warnings_count=2,
            retry_count=1
        )
        
        db.store_test_metric(test_metric)
        
        # Retrieve test history
        history = db.get_test_history("test_database_ops", days=1)
        
        assert len(history) >= 1
        retrieved_metric = history[0]
        assert retrieved_metric.test_name == "test_database_ops"
        assert retrieved_metric.status == "passed"
        assert retrieved_metric.duration == 1.0
    
    def test_flaky_test_detection(self):
        """Test flaky test detection algorithm."""
        db = QualityDatabase("tests/quality_gates/test_flaky_detection.db")
        
        # Simulate flaky test data
        test_name = "test_flaky_behavior"
        for i in range(10):
            status = "passed" if i % 3 != 0 else "failed"  # Fails ~33% of the time
            metric = TestMetrics(
                test_name=test_name,
                status=status,
                duration=1.0,
                memory_usage_mb=50.0,
                cpu_usage_percent=20.0,
                timestamp=datetime.now() - timedelta(hours=i)
            )
            db.store_test_metric(metric)
        
        # Detect flaky tests
        flaky_tests = db.get_flaky_tests(days=1, min_runs=5)
        
        assert test_name in flaky_tests
        assert 0 < flaky_tests[test_name] < 1.0  # Should have some failure rate
    
    def test_quality_trend_analysis(self, quality_monitor):
        """Test quality trend analysis over time."""
        
        # Simulate quality metrics over multiple days
        metrics_history = []
        for day in range(7):
            # Gradually improving quality
            success_rate = 0.85 + (day * 0.02)  # From 85% to 97%
            avg_duration = 3.0 - (day * 0.2)    # From 3.0s to 1.8s
            
            metric = QualityMetrics(
                total_tests=100, passed_tests=int(100 * success_rate),
                failed_tests=int(100 * (1 - success_rate)), skipped_tests=0,
                error_tests=0, total_duration=avg_duration * 100,
                average_duration=avg_duration, success_rate=success_rate,
                memory_usage_mb=200.0, cpu_usage_percent=30.0,
                flaky_tests=[], slow_tests=[],
                timestamp=datetime.now() - timedelta(days=6-day)
            )
            metrics_history.append(metric)
        
        # Analyze trends
        success_rates = [m.success_rate for m in metrics_history]
        avg_durations = [m.average_duration for m in metrics_history]
        
        # Verify improving trends
        assert success_rates[-1] > success_rates[0], "Success rate should improve"
        assert avg_durations[-1] < avg_durations[0], "Average duration should decrease"
    
    def test_comprehensive_quality_assessment(self, quality_monitor, quality_gate):
        """Test comprehensive quality assessment workflow."""
        
        # Simulate comprehensive test suite execution
        test_scenarios = [
            # Normal tests
            *[("test_normal_" + str(i), "passed", 1.0) for i in range(80)],
            # Some failures
            *[("test_failure_" + str(i), "failed", 2.0) for i in range(5)],
            # Slow tests
            *[("test_slow_" + str(i), "passed", 15.0) for i in range(3)],
            # Skipped tests
            *[("test_skip_" + str(i), "skipped", 0.1) for i in range(12)]
        ]
        
        for test_name, status, duration in test_scenarios:
            metric = TestMetrics(
                test_name=test_name,
                status=status,
                duration=duration,
                memory_usage_mb=100.0,
                cpu_usage_percent=25.0,
                timestamp=datetime.now(),
                warnings_count=0 if status == "passed" else 2
            )
            quality_monitor.test_metrics.append(metric)
        
        # Generate quality report
        report = quality_monitor.generate_quality_report()
        
        # Validate quality gates
        validation_results = quality_gate.validate(report)
        gate_summary = quality_gate.get_gate_summary(validation_results)
        
        # Comprehensive assertions
        assert report.total_tests == 100
        assert report.success_rate == 0.80  # 80/100 passed
        assert len(report.slow_tests) == 3
        
        # Quality gate assessment
        assert 'success_rate' in validation_results
        assert 'average_duration' in validation_results
        
        # Generate final assessment
        assessment = {
            'overall_quality': 'good' if gate_summary['success_rate'] >= 0.8 else 'needs_improvement',
            'key_metrics': {
                'success_rate': report.success_rate,
                'average_duration': report.average_duration,
                'total_tests': report.total_tests
            },
            'recommendations': []
        }
        
        if report.success_rate < 0.95:
            assessment['recommendations'].append('Improve test reliability')
        if report.average_duration > 2.0:
            assessment['recommendations'].append('Optimize test execution speed')
        if len(report.slow_tests) > 5:
            assessment['recommendations'].append('Address slow test performance')
        
        # Final quality check
        assert assessment['overall_quality'] in ['good', 'needs_improvement']
        assert isinstance(assessment['recommendations'], list)
    
    def test_quality_monitoring_integration(self, quality_monitor):
        """Test integration of all quality monitoring components."""
        
        # Simulate real test execution with monitoring
        with quality_monitor.monitor_test("test_integration_sample"):
            # Simulate test work
            data = [i ** 2 for i in range(1000)]
            result = sum(data)
            assert result > 0
        
        # Verify metric was captured
        assert len(quality_monitor.test_metrics) == 1
        metric = quality_monitor.test_metrics[0]
        
        assert metric.test_name == "test_integration_sample"
        assert metric.status == "passed"
        assert metric.duration > 0
        assert metric.memory_usage_mb >= 0
        assert metric.cpu_usage_percent >= 0
    
    def test_automated_quality_reporting(self, quality_monitor):
        """Test automated quality report generation."""
        
        # Add sample metrics
        for i in range(50):
            status = "passed" if i < 47 else "failed"
            metric = TestMetrics(
                test_name=f"test_auto_report_{i}",
                status=status,
                duration=1.0 + (i * 0.1),
                memory_usage_mb=50.0 + i,
                cpu_usage_percent=20.0,
                timestamp=datetime.now()
            )
            quality_monitor.test_metrics.append(metric)
        
        # Generate comprehensive report
        report = quality_monitor.generate_quality_report()
        
        # Convert to dict for JSON serialization
        report_dict = report.to_dict()
        
        # Save report
        report_path = Path("tests/quality_gates/quality_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Verify report structure
        assert 'total_tests' in report_dict
        assert 'success_rate' in report_dict
        assert 'timestamp' in report_dict
        
        print(f"Quality report saved to: {report_path}")
        print(f"Success rate: {report.success_rate:.2%}")
        print(f"Average duration: {report.average_duration:.2f}s")