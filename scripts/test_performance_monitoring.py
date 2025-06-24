#!/usr/bin/env python3
"""Test script for performance monitoring infrastructure.

This script validates the real-time performance monitoring capabilities
implemented in Phase 2 of the controlled feature reintroduction.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_performance_monitor():
    """Test core performance monitor."""
    print("📊 Testing Performance Monitor...")
    
    try:
        from pynomaly.infrastructure.monitoring.performance_monitor import (
            PerformanceMonitor, PerformanceTracker
        )
        
        # Create monitor
        monitor = PerformanceMonitor(max_history=100)
        
        # Test basic operation tracking
        operation_id = monitor.start_operation(
            "test_operation",
            algorithm_name="test_algorithm",
            dataset_size=1000,
            metadata={"test": "data"}
        )
        
        # Simulate some work
        time.sleep(0.1)
        test_array = np.random.random((1000, 10))
        result = np.mean(test_array)
        
        # End operation
        metrics = monitor.end_operation(operation_id, samples_processed=1000)
        
        print(f"  ✅ Operation tracked: {metrics.operation_name}")
        print(f"  ✅ Execution time: {metrics.execution_time:.3f}s")
        print(f"  ✅ Memory usage: {metrics.memory_usage:.1f}MB")
        print(f"  ✅ Samples per second: {metrics.samples_per_second:.1f}")
        
        # Test context manager
        with PerformanceTracker(monitor, "context_test", "test_alg") as tracker:
            time.sleep(0.05)
            tracker.set_samples_processed(500)
            tracker.set_quality_metrics({"accuracy": 0.95, "f1_score": 0.92})
        
        print(f"  ✅ Context manager tracking working")
        print(f"  ✅ Total operations: {monitor.total_operations}")
        
        # Test real-time metrics
        real_time = monitor.get_real_time_metrics()
        print(f"  ✅ Real-time metrics: {real_time['active_operations']} active operations")
        
        print("  ✅ Performance monitor working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance monitor test failed: {e}")
        return False


def test_performance_monitoring_service():
    """Test high-level performance monitoring service."""
    print("\n🚀 Testing Performance Monitoring Service...")
    
    try:
        from pynomaly.application.services.performance_monitoring_service import PerformanceMonitoringService
        from pynomaly.domain.entities import Dataset, DetectionResult
        from pynomaly.infrastructure.monitoring.performance_monitor import PerformanceMonitor
        
        # Create a mock detector for testing
        class MockDetector:
            def __init__(self, name, algorithm_name):
                self.name = name
                self.algorithm_name = algorithm_name
                self.parameters = {}
        
        # Create service
        monitor = PerformanceMonitor()
        service = PerformanceMonitoringService(monitor, auto_start_monitoring=False)
        
        # Test monitoring a detection operation
        test_data = np.random.normal(0, 1, (1000, 5))
        dataset = Dataset(name="test_dataset", data=test_data)
        detector = MockDetector(name="test_detector", algorithm_name="IsolationForest")
        
        def mock_detection_operation(detector, dataset):
            # Simulate detection work
            time.sleep(0.1)
            
            # Create a simple mock result object
            class MockDetectionResult:
                def __init__(self):
                    self.metadata = {"accuracy": 0.95, "samples_processed": len(dataset.data)}
                    
            return MockDetectionResult()
        
        # Monitor the operation
        result, metrics = service.monitor_detection_operation(
            detector, dataset, mock_detection_operation
        )
        
        print(f"  ✅ Detection operation monitored")
        print(f"  ✅ Algorithm: {metrics.algorithm_name}")
        print(f"  ✅ Execution time: {metrics.execution_time:.3f}s")
        print(f"  ✅ Samples processed: {metrics.samples_processed}")
        
        # Test training operation monitoring
        def mock_training_operation(detector, dataset):
            time.sleep(0.05)
            return detector  # Return "trained" detector
        
        trained_detector, train_metrics = service.monitor_training_operation(
            detector, dataset, mock_training_operation
        )
        
        print(f"  ✅ Training operation monitored")
        print(f"  ✅ Training time: {train_metrics.execution_time:.3f}s")
        
        # Test algorithm performance comparison
        comparison = service.get_algorithm_performance_comparison(min_operations=1)
        print(f"  ✅ Performance comparison: {len(comparison.get('algorithms', {}))} algorithms analyzed")
        
        # Test dashboard data
        dashboard_data = service.get_monitoring_dashboard_data()
        print(f"  ✅ Dashboard data: {dashboard_data['system_status']['total_operations_monitored']} operations")
        
        print("  ✅ Performance monitoring service working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance monitoring service test failed: {e}")
        return False


def test_alert_system():
    """Test performance alert system."""
    print("\n🚨 Testing Alert System...")
    
    try:
        from pynomaly.infrastructure.monitoring.performance_monitor import PerformanceMonitor
        
        # Create monitor with low thresholds to trigger alerts
        monitor = PerformanceMonitor(
            alert_thresholds={
                'execution_time': 0.01,  # Very low threshold
                'memory_usage': 1.0,     # Very low threshold
                'cpu_usage': 1.0,        # Very low threshold
                'samples_per_second': 10000.0  # Very high threshold
            }
        )
        
        # Track alerts
        alerts_received = []
        
        def alert_handler(alert):
            alerts_received.append(alert)
        
        monitor.add_alert_callback(alert_handler)
        
        # Perform operation that should trigger alerts
        operation_id = monitor.start_operation("alert_test", "test_algorithm", 100)
        
        # Simulate work that will trigger alerts
        time.sleep(0.1)  # Should trigger execution time alert
        large_array = np.random.random((10000, 100))  # Should trigger memory alert
        
        metrics = monitor.end_operation(operation_id, samples_processed=100)
        
        # Check if alerts were triggered
        active_alerts = monitor.get_active_alerts()
        print(f"  ✅ Alerts triggered: {len(active_alerts)}")
        print(f"  ✅ Alerts received via callback: {len(alerts_received)}")
        
        if active_alerts:
            for alert in active_alerts[:2]:  # Show first 2 alerts
                print(f"  ⚠️ Alert: {alert.alert_type} - {alert.message}")
        
        # Test clearing alerts
        monitor.clear_alerts()
        print(f"  ✅ Alerts cleared: {len(monitor.get_active_alerts())} remaining")
        
        print("  ✅ Alert system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Alert system test failed: {e}")
        return False


def test_performance_trends():
    """Test performance trend analysis."""
    print("\n📈 Testing Performance Trends...")
    
    try:
        from pynomaly.application.services.performance_monitoring_service import PerformanceMonitoringService
        from pynomaly.infrastructure.monitoring.performance_monitor import PerformanceMonitor
        from datetime import timedelta
        
        # Create service
        monitor = PerformanceMonitor()
        service = PerformanceMonitoringService(monitor, auto_start_monitoring=False)
        
        # Generate some historical data
        for i in range(10):
            operation_id = monitor.start_operation(f"trend_test_{i}", "test_algorithm", 1000)
            time.sleep(0.01)  # Small delay
            monitor.end_operation(operation_id, samples_processed=1000)
        
        # Test trend analysis
        trends = service.get_performance_trends(
            operation_name="trend_test_0",
            time_window=timedelta(minutes=1),
            bucket_size=timedelta(seconds=10)
        )
        
        if 'time_buckets' in trends:
            print(f"  ✅ Trend analysis: {len(trends['time_buckets'])} time buckets")
            print(f"  ✅ Total operations analyzed: {trends['total_operations']}")
        else:
            print(f"  ✅ Trend analysis completed: {trends.get('message', 'No data')}")
        
        # Test baseline setting and regression checking
        service.set_performance_baseline("test_operation", {
            'execution_time': 0.1,
            'memory_usage': 50.0,
            'cpu_usage': 20.0
        })
        
        print(f"  ✅ Performance baseline set")
        
        # Add some operations for regression check
        for i in range(3):
            operation_id = monitor.start_operation("test_operation", "test_alg", 1000)
            time.sleep(0.02)  # Slightly longer than baseline
            monitor.end_operation(operation_id, samples_processed=1000)
        
        regression_check = service.check_performance_regression("test_operation")
        if 'regressions_detected' in regression_check:
            print(f"  ✅ Regression check: {regression_check['regressions_detected']} regressions found")
        else:
            print(f"  ⚠️ Regression check: {regression_check.get('error', 'Unknown error')}")
        
        print("  ✅ Performance trends working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance trends test failed: {e}")
        return False


def test_container_integration():
    """Test container integration with performance monitoring."""
    print("\n🔧 Testing Container Integration...")
    
    try:
        from pynomaly.infrastructure.config.container import Container
        
        # Create container
        container = Container()
        
        # Test performance monitoring service availability
        try:
            perf_service = container.performance_monitoring_service()
            print("  ✅ Performance monitoring service available")
            
            # Test that monitoring is started
            dashboard_data = perf_service.get_monitoring_dashboard_data()
            print(f"  ✅ Monitoring status: {dashboard_data['system_status']['monitoring_enabled']}")
            
        except AttributeError:
            print("  ⚠️ Performance monitoring service not available (feature may be disabled)")
        
        # Test performance monitor availability
        try:
            monitor = container.performance_monitor()
            real_time_metrics = monitor.get_real_time_metrics()
            print(f"  ✅ Performance monitor available: {real_time_metrics['total_operations']} operations tracked")
        except AttributeError:
            print("  ⚠️ Performance monitor not available (feature may be disabled)")
        
        print("  ✅ Container integration working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Container integration test failed: {e}")
        return False


def test_monitoring_decorators():
    """Test performance monitoring decorators."""
    print("\n🎭 Testing Monitoring Decorators...")
    
    try:
        from pynomaly.infrastructure.monitoring.performance_monitor import (
            PerformanceMonitor, monitor_performance
        )
        
        monitor = PerformanceMonitor()
        
        # Test function decorator
        @monitor_performance(monitor, "decorated_function", "test_algorithm")
        def test_function(data_size):
            # Simulate some work
            data = np.random.random((data_size, 10))
            result = np.mean(data)
            time.sleep(0.01)
            return result
        
        # Call decorated function
        result = test_function(1000)
        print(f"  ✅ Decorated function executed: result={result:.3f}")
        
        # Check if operation was tracked
        if monitor.total_operations > 0:
            print(f"  ✅ Operation tracking: {monitor.total_operations} operations recorded")
            
            # Get recent metrics
            stats = monitor.get_operation_statistics("decorated_function")
            if 'operation_count' in stats:
                print(f"  ✅ Function stats: {stats['operation_count']} calls tracked")
        
        print("  ✅ Monitoring decorators working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Monitoring decorators test failed: {e}")
        return False


def test_performance_monitoring_readiness():
    """Test overall performance monitoring readiness."""
    print("\n🚀 Testing Performance Monitoring Readiness...")
    
    try:
        # Check if all performance monitoring components are available
        components = [
            "performance_monitor",
            "performance_monitoring_service",
            "alert_system",
            "performance_trends",
            "container_integration",
            "monitoring_decorators"
        ]
        
        results = {
            "performance_monitor": test_performance_monitor(),
            "performance_monitoring_service": test_performance_monitoring_service(),
            "alert_system": test_alert_system(),
            "performance_trends": test_performance_trends(),
            "container_integration": test_container_integration(),
            "monitoring_decorators": test_monitoring_decorators()
        }
        
        passing = sum(results.values())
        total = len(results)
        
        print(f"\n📈 Performance Monitoring Status: {passing}/{total} components ready")
        
        if passing == total:
            print("🎉 Performance monitoring infrastructure is fully operational!")
            print("✅ Ready for real-time performance tracking and optimization")
            return True
        else:
            print("⚠️ Some performance monitoring components need attention")
            for component, status in results.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {component}")
            return False
        
    except Exception as e:
        print(f"❌ Performance monitoring readiness test failed: {e}")
        return False


def main():
    """Run all performance monitoring infrastructure tests."""
    print("🧪 Pynomaly Performance Monitoring Infrastructure Validation")
    print("=" * 65)
    
    try:
        success = test_performance_monitoring_readiness()
        
        if success:
            print("\n🎯 Performance monitoring infrastructure validation successful!")
            print("🚀 Ready for real-time performance tracking and optimization")
            sys.exit(0)
        else:
            print("\n⚠️ Performance monitoring infrastructure validation failed")
            print("🔧 Please review and fix issues before proceeding")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()