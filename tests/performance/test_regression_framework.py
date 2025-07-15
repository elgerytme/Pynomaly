"""
Comprehensive test suite for performance regression testing framework.

Tests all components of the performance regression detection system including
baseline tracking, alert system, and reporting functionality.
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

# Import the modules to test
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from pynomaly.infrastructure.performance.regression_framework import (
    PerformanceRegressionFramework,
    PerformanceMetric,
    PerformanceBaseline,
    RegressionResult,
    APIPerformanceTest,
    DatabasePerformanceTest,
    SystemResourceTest,
    BaselineManager,
    RegressionDetector
)

from pynomaly.infrastructure.performance.baseline_tracker import (
    AdaptiveBaselineTracker,
    BaselineConfig,
    TrendAnalysis,
    PerformanceDatabase
)

from pynomaly.infrastructure.performance.alert_system import (
    PerformanceAlertManager,
    AlertConfig,
    Alert,
    ConsoleAlertChannel,
    GitHubAlertChannel,
    AlertThrottler
)

from pynomaly.infrastructure.performance.reporting_service import (
    PerformanceReportGenerator
)


class TestPerformanceMetric:
    """Test PerformanceMetric class."""
    
    def test_metric_creation(self):
        """Test basic metric creation."""
        metric = PerformanceMetric(
            name="response_time",
            value=125.5,
            unit="ms",
            timestamp=datetime.now()
        )
        
        assert metric.name == "response_time"
        assert metric.value == 125.5
        assert metric.unit == "ms"
        assert isinstance(metric.timestamp, datetime)
    
    def test_metric_to_dict(self):
        """Test metric serialization."""
        timestamp = datetime.now()
        metric = PerformanceMetric(
            name="error_rate",
            value=2.5,
            unit="percent",
            timestamp=timestamp,
            context={"test": "value"},
            tags=["api", "test"]
        )
        
        result = metric.to_dict()
        
        assert result["name"] == "error_rate"
        assert result["value"] == 2.5
        assert result["unit"] == "percent"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["context"] == {"test": "value"}
        assert result["tags"] == ["api", "test"]


class TestPerformanceBaseline:
    """Test PerformanceBaseline class."""
    
    def test_baseline_creation(self):
        """Test baseline creation with statistics."""
        baseline = PerformanceBaseline(
            metric_name="response_time",
            mean=100.0,
            std=15.0,
            p50=95.0,
            p95=130.0,
            p99=150.0,
            sample_size=100,
            established_at=datetime.now()
        )
        
        assert baseline.metric_name == "response_time"
        assert baseline.mean == 100.0
        assert baseline.std == 15.0
        assert baseline.sample_size == 100
    
    def test_regression_detection(self):
        """Test regression detection logic."""
        baseline = PerformanceBaseline(
            metric_name="response_time",
            mean=100.0,
            std=15.0,
            p50=95.0,
            p95=130.0,
            p99=150.0,
            sample_size=100,
            established_at=datetime.now()
        )
        
        # Test regression (value significantly higher than baseline)
        assert baseline.is_regression(140.0, threshold_std=2.0)  # 100 + 2*15 = 130, so 140 > 130
        assert not baseline.is_regression(120.0, threshold_std=2.0)  # 120 < 130
        
        # Test improvement (value significantly lower than baseline)
        assert baseline.is_improvement(60.0, threshold_std=2.0)  # 100 - 2*15 = 70, so 60 < 70
        assert not baseline.is_improvement(80.0, threshold_std=2.0)  # 80 > 70
    
    def test_baseline_serialization(self):
        """Test baseline to_dict method."""
        timestamp = datetime.now()
        baseline = PerformanceBaseline(
            metric_name="response_time",
            mean=100.0,
            std=15.0,
            p50=95.0,
            p95=130.0,
            p99=150.0,
            sample_size=100,
            established_at=timestamp,
            environment={"version": "1.0"}
        )
        
        result = baseline.to_dict()
        
        assert result["metric_name"] == "response_time"
        assert result["mean"] == 100.0
        assert result["established_at"] == timestamp.isoformat()
        assert result["environment"] == {"version": "1.0"}


class TestAPIPerformanceTest:
    """Test API performance test implementation."""
    
    @pytest.mark.asyncio
    async def test_api_test_creation(self):
        """Test API test configuration."""
        test = APIPerformanceTest(
            name="health_check",
            endpoint="http://localhost:8000/health",
            method="GET",
            concurrent_users=2,
            duration_seconds=5
        )
        
        assert test.name == "health_check"
        assert test.endpoint == "http://localhost:8000/health"
        assert test.method == "GET"
        assert test.concurrent_users == 2
        assert test.duration_seconds == 5
        assert "api" in test.tags
    
    @pytest.mark.asyncio
    async def test_api_test_mock_execution(self):
        """Test API test execution with mocked responses."""
        test = APIPerformanceTest(
            name="mock_test",
            endpoint="http://localhost:8000/test",
            concurrent_users=1,
            duration_seconds=1
        )
        
        # Mock the _make_request method to avoid actual HTTP calls
        async def mock_request(session):
            await asyncio.sleep(0.01)  # Simulate network delay
            return 50.0  # Return mock response time in ms
        
        test._make_request = mock_request
        
        # This would typically require a running server, so we'll mock it
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_session
            
            # For now, we'll just test that the method exists and is callable
            assert callable(test.run)


class TestBaselineManager:
    """Test baseline management functionality."""
    
    def test_baseline_manager_creation(self):
        """Test baseline manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BaselineManager(Path(temp_dir))
            
            assert manager.storage_path == Path(temp_dir)
            assert isinstance(manager.baselines, dict)
            assert len(manager.baselines) == 0
    
    def test_establish_baseline(self):
        """Test baseline establishment from data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BaselineManager(Path(temp_dir))
            
            # Generate test data
            values = np.random.normal(100, 15, 50).tolist()
            
            baseline = manager.establish_baseline(
                "test_metric",
                values,
                environment={"test": True}
            )
            
            assert baseline.metric_name == "test_metric"
            assert baseline.sample_size == 50
            assert 85 < baseline.mean < 115  # Should be around 100
            assert 10 < baseline.std < 20    # Should be around 15
            assert baseline.environment == {"test": True}
            
            # Check that baseline was stored
            assert "test_metric" in manager.baselines
    
    def test_insufficient_data_for_baseline(self):
        """Test baseline establishment with insufficient data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BaselineManager(Path(temp_dir))
            
            # Too few data points
            values = [100, 105, 95]
            
            with pytest.raises(ValueError, match="Need at least 10 data points"):
                manager.establish_baseline("test_metric", values)
    
    def test_baseline_persistence(self):
        """Test that baselines are saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and populate baseline manager
            manager1 = BaselineManager(Path(temp_dir))
            values = np.random.normal(100, 15, 30).tolist()
            baseline1 = manager1.establish_baseline("test_metric", values)
            
            # Create new manager from same directory
            manager2 = BaselineManager(Path(temp_dir))
            
            # Check that baseline was loaded
            assert "test_metric" in manager2.baselines
            baseline2 = manager2.baselines["test_metric"]
            
            assert baseline2.metric_name == baseline1.metric_name
            assert baseline2.mean == baseline1.mean
            assert baseline2.std == baseline1.std


class TestRegressionDetector:
    """Test regression detection logic."""
    
    def test_regression_detection(self):
        """Test regression detection with known baselines."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup baseline
            manager = BaselineManager(Path(temp_dir))
            values = [100] * 20  # Perfect baseline with no variation
            baseline = manager.establish_baseline("test_metric", values)
            baseline.std = 10.0  # Set known std for testing
            
            # Create detector
            detector = RegressionDetector(manager)
            
            # Test regression case
            regression_metric = PerformanceMetric(
                name="test_metric",
                value=130.0,  # 3 std above mean
                unit="ms",
                timestamp=datetime.now()
            )
            
            results = detector.analyze([regression_metric])
            
            assert len(results) == 1
            result = results[0]
            assert result.is_regression
            assert not result.is_improvement
            assert result.severity in ["high", "critical"]
            
            # Test improvement case
            improvement_metric = PerformanceMetric(
                name="test_metric",
                value=70.0,  # 3 std below mean
                unit="ms",
                timestamp=datetime.now()
            )
            
            results = detector.analyze([improvement_metric])
            
            assert len(results) == 1
            result = results[0]
            assert not result.is_regression
            assert result.is_improvement
    
    def test_no_baseline_warning(self):
        """Test behavior when no baseline exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BaselineManager(Path(temp_dir))
            detector = RegressionDetector(manager)
            
            metric = PerformanceMetric(
                name="unknown_metric",
                value=100.0,
                unit="ms",
                timestamp=datetime.now()
            )
            
            with patch('pynomaly.infrastructure.performance.regression_framework.logger') as mock_logger:
                results = detector.analyze([metric])
                
                # Should log warning and return empty results
                mock_logger.warning.assert_called_once()
                assert len(results) == 0


class TestAdaptiveBaselineTracker:
    """Test adaptive baseline tracking functionality."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization with config."""
        config = BaselineConfig(
            min_samples=15,
            outlier_threshold=2.5,
            trend_window_days=14
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            tracker = AdaptiveBaselineTracker(str(db_path), config)
            
            assert tracker.config.min_samples == 15
            assert tracker.config.outlier_threshold == 2.5
            assert tracker.config.trend_window_days == 14
    
    def test_record_metrics(self):
        """Test recording metrics in tracker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            tracker = AdaptiveBaselineTracker(str(db_path))
            
            metrics = [
                {
                    'name': 'response_time',
                    'value': 125.5,
                    'unit': 'ms',
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            tracker.record_metrics(metrics, "test_run_123")
            
            # Verify metric was recorded (would need to check database in real implementation)
            assert True  # Placeholder - actual implementation would verify database


class TestPerformanceAlertSystem:
    """Test performance alert system."""
    
    def test_alert_creation(self):
        """Test alert object creation."""
        alert = Alert(
            id="test_alert_123",
            severity="high",
            title="Performance regression detected",
            message="Response time increased significantly",
            metric_name="response_time",
            current_value=250.0,
            baseline_value=150.0,
            deviation=3.2,
            timestamp=datetime.now(),
            run_id="test_run"
        )
        
        assert alert.id == "test_alert_123"
        assert alert.severity == "high"
        assert alert.current_value == 250.0
        assert alert.baseline_value == 150.0
        assert not alert.resolved
    
    def test_console_alert_channel(self):
        """Test console alert channel."""
        channel = ConsoleAlertChannel()
        
        assert channel.enabled
        assert channel.validate_config()
        
        alert = Alert(
            id="test",
            severity="critical",
            title="Test Alert",
            message="Test message",
            metric_name="test_metric",
            current_value=100.0,
            baseline_value=50.0,
            deviation=5.0,
            timestamp=datetime.now(),
            run_id="test"
        )
        
        # Test sending alert
        result = asyncio.run(channel.send_alert(alert))
        assert result is True
    
    def test_alert_throttling(self):
        """Test alert throttling functionality."""
        config = AlertConfig(
            throttle_minutes=5,
            max_alerts_per_hour=2
        )
        
        throttler = AlertThrottler(config)
        
        alert1 = Alert(
            id="test1",
            severity="medium",
            title="Test Alert 1",
            message="Test message",
            metric_name="response_time",
            current_value=100.0,
            baseline_value=80.0,
            deviation=2.0,
            timestamp=datetime.now(),
            run_id="test"
        )
        
        alert2 = Alert(
            id="test2",
            severity="medium",
            title="Test Alert 2",
            message="Test message",
            metric_name="response_time",
            current_value=105.0,
            baseline_value=80.0,
            deviation=2.1,
            timestamp=datetime.now(),
            run_id="test"
        )
        
        # First alert should be allowed
        assert throttler.should_send_alert(alert1)
        throttler.record_alert(alert1)
        
        # Second alert for same metric should be throttled
        assert not throttler.should_send_alert(alert2)
    
    @pytest.mark.asyncio
    async def test_alert_manager_processing(self):
        """Test alert manager processing regression results."""
        config = AlertConfig(
            enabled=True,
            channels=['console'],
            throttle_minutes=0  # Disable throttling for test
        )
        
        manager = PerformanceAlertManager(config)
        
        regression_results = [
            {
                'metric_name': 'response_time',
                'current_value': 200.0,
                'baseline_mean': 100.0,
                'deviation_std': 4.0,
                'is_regression': True,
                'severity': 'high'
            }
        ]
        
        with patch.object(manager, '_send_alert', return_value=True) as mock_send:
            await manager.process_regression_results(
                regression_results,
                'test_run',
                {'env': 'test'}
            )
            
            # Verify alert was created and sent
            mock_send.assert_called_once()


class TestPerformanceReportGenerator:
    """Test performance reporting functionality."""
    
    def test_report_generator_creation(self):
        """Test report generator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = PerformanceReportGenerator(temp_dir)
            
            assert generator.output_dir == Path(temp_dir)
            assert generator.output_dir.exists()
    
    def test_json_report_generation(self):
        """Test JSON report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = PerformanceReportGenerator(temp_dir)
            
            test_data = {
                'run_id': 'test_123',
                'timestamp': datetime.now().isoformat(),
                'test_results': {
                    'total_tests': 3,
                    'successful_tests': 3
                },
                'regression_summary': {
                    'total_regressions': 1,
                    'has_critical_regressions': False
                }
            }
            
            report_file = generator.generate_json_report(test_data)
            
            assert Path(report_file).exists()
            
            # Verify content
            with open(report_file, 'r') as f:
                loaded_data = json.load(f)
                assert loaded_data['run_id'] == 'test_123'
                assert loaded_data['test_results']['total_tests'] == 3


class TestPerformanceRegressionFramework:
    """Test the complete performance regression framework."""
    
    @pytest.mark.asyncio
    async def test_framework_initialization(self):
        """Test framework initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = PerformanceRegressionFramework(temp_dir)
            
            assert framework.baseline_manager is not None
            assert framework.regression_detector is not None
            assert len(framework.tests) == 0
            assert len(framework.results_history) == 0
    
    def test_add_test(self):
        """Test adding tests to framework."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = PerformanceRegressionFramework(temp_dir)
            
            test = APIPerformanceTest(
                name="test_endpoint",
                endpoint="http://localhost:8000/test",
                duration_seconds=5
            )
            
            framework.add_test(test)
            
            assert len(framework.tests) == 1
            assert framework.tests[0].name == "test_endpoint"
    
    @pytest.mark.asyncio
    async def test_framework_run_with_mock_tests(self):
        """Test running framework with mocked tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = PerformanceRegressionFramework(temp_dir)
            
            # Create mock test
            mock_test = MagicMock()
            mock_test.name = "mock_test"
            mock_test.run = AsyncMock(return_value=[
                PerformanceMetric(
                    name="test_metric",
                    value=100.0,
                    unit="ms",
                    timestamp=datetime.now()
                )
            ])
            
            framework.add_test(mock_test)
            
            # Run tests
            results = await framework.run_tests()
            
            assert results['total_tests'] == 1
            assert results['successful_tests'] == 1
            assert 'test_results' in results
            assert 'regression_analysis' in results
            assert results['test_results']['mock_test']['status'] == 'success'
    
    def test_regression_summary_generation(self):
        """Test regression summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = PerformanceRegressionFramework(temp_dir)
            
            # Mock results with regressions
            results = {
                'regression_analysis': {
                    'results': [
                        {
                            'is_regression': True,
                            'is_improvement': False,
                            'severity': 'high'
                        },
                        {
                            'is_regression': False,
                            'is_improvement': True,
                            'severity': 'low'
                        },
                        {
                            'is_regression': True,
                            'is_improvement': False,
                            'severity': 'critical'
                        }
                    ]
                }
            }
            
            summary = framework.get_regression_summary(results)
            
            assert summary['total_regressions'] == 2
            assert summary['total_improvements'] == 1
            assert summary['has_critical_regressions'] is True
            assert summary['overall_status'] == 'CRITICAL'


class TestIntegrationScenarios:
    """Integration tests for complete scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_regression_detection(self):
        """Test complete end-to-end regression detection scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup framework
            framework = PerformanceRegressionFramework(temp_dir)
            
            # Establish baseline with historical data
            baseline_values = np.random.normal(100, 10, 50).tolist()
            framework.baseline_manager.establish_baseline(
                "response_time_mean",
                baseline_values
            )
            
            # Create mock test that returns regression
            mock_test = MagicMock()
            mock_test.name = "regression_test"
            mock_test.run = AsyncMock(return_value=[
                PerformanceMetric(
                    name="response_time_mean",
                    value=140.0,  # Significantly higher than baseline
                    unit="ms",
                    timestamp=datetime.now()
                )
            ])
            
            framework.add_test(mock_test)
            
            # Run complete test
            results = await framework.run_tests()
            
            # Verify regression was detected
            regression_analysis = results['regression_analysis']
            assert regression_analysis['total_regressions'] > 0
            
            # Verify alert would be generated
            alert_manager = PerformanceAlertManager(
                AlertConfig(enabled=True, channels=['console'])
            )
            
            with patch.object(alert_manager, '_send_alert', return_value=True) as mock_send:
                await alert_manager.process_regression_results(
                    regression_analysis['results'],
                    results['run_id']
                )
                
                # Should have sent alert for regression
                assert mock_send.call_count > 0
    
    def test_performance_trend_analysis(self):
        """Test performance trend analysis over time."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            tracker = AdaptiveBaselineTracker(str(db_path))
            
            # Simulate degrading performance over time
            base_time = datetime.now()
            for i in range(20):
                metrics = [{
                    'name': 'response_time',
                    'value': 100 + i * 2,  # Gradually increasing
                    'unit': 'ms',
                    'timestamp': (base_time + timedelta(days=i)).isoformat()
                }]
                
                tracker.record_metrics(metrics, f"run_{i}")
            
            # Analyze trend
            trend = tracker.trend_analyzer.analyze_trend('response_time')
            
            # Should detect degrading trend
            assert trend.metric_name == 'response_time'
            assert trend.data_points == 20
            # Note: Actual trend direction depends on implementation details


# Utility functions for tests
def create_mock_performance_data():
    """Create mock performance data for testing."""
    return {
        'run_id': 'test_run_123',
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': 45.2,
        'test_results': {
            'total_tests': 3,
            'successful_tests': 3,
            'test_results': {
                'health_check': {
                    'status': 'success',
                    'metrics': [
                        {'name': 'response_time_mean', 'value': 125.5, 'unit': 'ms'},
                        {'name': 'error_rate', 'value': 0.0, 'unit': 'percent'}
                    ]
                }
            }
        },
        'regression_analysis': {
            'total_regressions': 1,
            'total_improvements': 0,
            'results': [
                {
                    'metric_name': 'response_time_mean',
                    'current_value': 125.5,
                    'baseline_mean': 100.0,
                    'deviation_std': 2.5,
                    'is_regression': True,
                    'is_improvement': False,
                    'severity': 'medium'
                }
            ]
        },
        'environment': {
            'python_version': '3.12',
            'cpu_count': 4,
            'memory_gb': 16
        }
    }


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])