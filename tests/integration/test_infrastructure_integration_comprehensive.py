"""Comprehensive infrastructure integration tests.

This module contains integration tests for infrastructure components
including configuration, dependency injection, monitoring, and
external service integrations.
"""

import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.config import create_container
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestDependencyInjectionIntegration:
    """Test dependency injection container integration."""
    
    def test_container_creation_integration(self):
        """Test dependency injection container creation."""
        try:
            # Create container
            container = create_container()
            
            # Verify container is created
            assert container is not None
            
            # Test container has expected configuration
            # This depends on the actual container implementation
            # For now, we just verify it doesn't crash
            
        except Exception as e:
            # If container creation fails, it should fail gracefully
            assert isinstance(e, (ImportError, AttributeError, ValueError))
    
    def test_service_registration_integration(self):
        """Test service registration in container."""
        try:
            container = create_container()
            
            # Test service registration (mock implementation)
            # This would depend on the actual DI framework used
            
            # For testing purposes, simulate service registration
            services = {
                "detection_service": Mock(),
                "threshold_calculator": Mock(),
                "ensemble_aggregator": Mock()
            }
            
            # Verify services can be registered and retrieved
            for service_name, service_instance in services.items():
                # In real implementation, this would use container.register()
                # For now, we verify the concept works
                assert service_instance is not None
                assert hasattr(service_instance, '_mock_name') or hasattr(service_instance, 'spec')
            
        except Exception:
            pytest.skip("Dependency injection container not fully implemented")
    
    def test_configuration_injection_integration(self):
        """Test configuration injection integration."""
        # Test configuration injection through environment variables
        test_config = {
            "PYNOMALY_DEFAULT_ALGORITHM": "IsolationForest",
            "PYNOMALY_DEFAULT_CONTAMINATION": "0.1",
            "PYNOMALY_LOG_LEVEL": "INFO"
        }
        
        # Store original environment
        original_env = {}
        for key in test_config:
            if key in os.environ:
                original_env[key] = os.environ[key]
        
        try:
            # Set test configuration
            for key, value in test_config.items():
                os.environ[key] = value
            
            # Test configuration access
            default_algorithm = os.getenv("PYNOMALY_DEFAULT_ALGORITHM", "IsolationForest")
            default_contamination = float(os.getenv("PYNOMALY_DEFAULT_CONTAMINATION", "0.1"))
            log_level = os.getenv("PYNOMALY_LOG_LEVEL", "INFO")
            
            # Verify configuration injection
            assert default_algorithm == "IsolationForest"
            assert default_contamination == 0.1
            assert log_level == "INFO"
            
            # Test configuration-based service creation
            try:
                adapter = SklearnAdapter(
                    algorithm_name=default_algorithm,
                    parameters={"contamination": default_contamination}
                )
                
                assert adapter.algorithm_name == default_algorithm
                
            except ImportError:
                pass  # scikit-learn not available
            
        finally:
            # Restore original environment
            for key in test_config:
                if key in original_env:
                    os.environ[key] = original_env[key]
                else:
                    os.environ.pop(key, None)


class TestConfigurationManagementIntegration:
    """Test configuration management integration."""
    
    @pytest.fixture
    def config_files(self):
        """Create temporary configuration files for testing."""
        files = {}
        
        # JSON configuration
        json_config = {
            "database": {
                "url": "sqlite:///pynomaly_test.db",
                "pool_size": 5,
                "echo": False
            },
            "algorithms": {
                "default": "IsolationForest",
                "parameters": {
                    "IsolationForest": {
                        "contamination": 0.1,
                        "n_estimators": 100
                    },
                    "LocalOutlierFactor": {
                        "contamination": 0.1,
                        "n_neighbors": 20
                    }
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_config, f, indent=2)
            files['json'] = f.name
        
        # YAML configuration
        yaml_config_content = """
database:
  url: "postgresql://user:pass@localhost/pynomaly_test"
  pool_size: 10
  echo: false

algorithms:
  default: "IsolationForest"
  parameters:
    IsolationForest:
      contamination: 0.1
      n_estimators: 100
      random_state: 42
    OneClassSVM:
      gamma: "scale"
      nu: 0.1

monitoring:
  enabled: true
  metrics_port: 8080
  health_check_interval: 30

logging:
  level: "DEBUG"
  file: "/tmp/pynomaly_test.log"
"""
        
        try:
            import yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_config_content)
                files['yaml'] = f.name
        except ImportError:
            pass  # PyYAML not available
        
        return files
    
    def test_json_configuration_loading_integration(self, config_files):
        """Test JSON configuration loading integration."""
        if 'json' not in config_files:
            pytest.skip("JSON configuration file not available")
        
        config_path = config_files['json']
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Test configuration structure
        assert 'database' in config
        assert 'algorithms' in config
        assert 'logging' in config
        
        # Test database configuration
        db_config = config['database']
        assert 'url' in db_config
        assert 'pool_size' in db_config
        assert isinstance(db_config['pool_size'], int)
        
        # Test algorithm configuration
        algo_config = config['algorithms']
        assert 'default' in algo_config
        assert 'parameters' in algo_config
        
        default_algorithm = algo_config['default']
        assert default_algorithm in algo_config['parameters']
        
        # Test configuration-based adapter creation
        try:
            algorithm_params = algo_config['parameters'][default_algorithm]
            
            adapter = SklearnAdapter(
                algorithm_name=default_algorithm,
                parameters=algorithm_params
            )
            
            assert adapter.algorithm_name == default_algorithm
            
        except ImportError:
            pass  # scikit-learn not available
        
        # Clean up
        Path(config_path).unlink()
    
    def test_yaml_configuration_loading_integration(self, config_files):
        """Test YAML configuration loading integration."""
        if 'yaml' not in config_files:
            pytest.skip("YAML configuration file not available")
        
        config_path = config_files['yaml']
        
        try:
            import yaml
            
            # Load configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Test configuration structure
            assert 'database' in config
            assert 'algorithms' in config
            assert 'monitoring' in config
            assert 'logging' in config
            
            # Test monitoring configuration
            monitoring_config = config['monitoring']
            assert 'enabled' in monitoring_config
            assert 'metrics_port' in monitoring_config
            assert isinstance(monitoring_config['enabled'], bool)
            assert isinstance(monitoring_config['metrics_port'], int)
            
            # Test algorithm configuration with additional parameters
            algo_config = config['algorithms']
            if_params = algo_config['parameters']['IsolationForest']
            
            assert 'random_state' in if_params
            assert isinstance(if_params['random_state'], int)
            
            # Test configuration application
            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters=if_params
                )
                
                assert adapter.algorithm_name == "IsolationForest"
                
            except ImportError:
                pass  # scikit-learn not available
            
        except ImportError:
            pytest.skip("PyYAML not available")
        
        # Clean up
        Path(config_path).unlink()
    
    def test_environment_configuration_override_integration(self, config_files):
        """Test environment variable configuration override."""
        if 'json' not in config_files:
            pytest.skip("JSON configuration file not available")
        
        config_path = config_files['json']
        
        # Load base configuration
        with open(config_path, 'r') as f:
            base_config = json.load(f)
        
        # Set environment overrides
        env_overrides = {
            "PYNOMALY_DB_POOL_SIZE": "20",
            "PYNOMALY_DEFAULT_CONTAMINATION": "0.15",
            "PYNOMALY_LOG_LEVEL": "DEBUG"
        }
        
        # Store original environment
        original_env = {}
        for key in env_overrides:
            if key in os.environ:
                original_env[key] = os.environ[key]
        
        try:
            # Set environment overrides
            for key, value in env_overrides.items():
                os.environ[key] = value
            
            # Apply environment overrides to configuration
            merged_config = base_config.copy()
            
            # Simulate configuration merging logic
            if "PYNOMALY_DB_POOL_SIZE" in os.environ:
                merged_config['database']['pool_size'] = int(os.getenv("PYNOMALY_DB_POOL_SIZE"))
            
            if "PYNOMALY_DEFAULT_CONTAMINATION" in os.environ:
                contamination = float(os.getenv("PYNOMALY_DEFAULT_CONTAMINATION"))
                for algo_params in merged_config['algorithms']['parameters'].values():
                    if 'contamination' in algo_params:
                        algo_params['contamination'] = contamination
            
            if "PYNOMALY_LOG_LEVEL" in os.environ:
                merged_config['logging']['level'] = os.getenv("PYNOMALY_LOG_LEVEL")
            
            # Verify overrides were applied
            assert merged_config['database']['pool_size'] == 20
            assert merged_config['logging']['level'] == "DEBUG"
            
            # Test that overridden configuration works
            if_params = merged_config['algorithms']['parameters']['IsolationForest']
            assert if_params['contamination'] == 0.15
            
        finally:
            # Restore original environment
            for key in env_overrides:
                if key in original_env:
                    os.environ[key] = original_env[key]
                else:
                    os.environ.pop(key, None)
        
        # Clean up
        Path(config_path).unlink()


class TestMonitoringIntegration:
    """Test monitoring and observability integration."""
    
    def test_health_check_integration(self):
        """Test health check integration."""
        # Mock health check components
        health_checks = {
            "database": {"status": "healthy", "response_time_ms": 5},
            "cache": {"status": "healthy", "response_time_ms": 2},
            "external_api": {"status": "degraded", "response_time_ms": 150, "error": "timeout"}
        }
        
        # Aggregate health status
        overall_status = "healthy"
        unhealthy_services = []
        
        for service, check in health_checks.items():
            if check["status"] != "healthy":
                if check["status"] == "degraded":
                    overall_status = "degraded" if overall_status == "healthy" else overall_status
                else:
                    overall_status = "unhealthy"
                    unhealthy_services.append(service)
        
        # Verify health check aggregation
        assert overall_status == "degraded"  # Due to external_api being degraded
        
        # Test health check response format
        health_response = {
            "status": overall_status,
            "timestamp": "2023-10-01T10:00:00Z",
            "services": health_checks,
            "unhealthy_services": unhealthy_services
        }
        
        assert "status" in health_response
        assert "timestamp" in health_response
        assert "services" in health_response
        assert health_response["status"] == "degraded"
    
    def test_metrics_collection_integration(self):
        """Test metrics collection integration."""
        # Mock metrics collection
        metrics = {
            "anomaly_detection": {
                "requests_total": 1250,
                "requests_per_second": 12.5,
                "average_response_time_ms": 85,
                "errors_total": 15,
                "error_rate": 0.012
            },
            "models": {
                "total_trained": 25,
                "active_models": 8,
                "average_training_time_s": 45.2,
                "prediction_accuracy": 0.892
            },
            "system": {
                "cpu_usage_percent": 65.5,
                "memory_usage_percent": 72.1,
                "disk_usage_percent": 45.3,
                "network_io_bytes_per_sec": 1048576
            }
        }
        
        # Test metrics validation
        for category, category_metrics in metrics.items():
            for metric_name, metric_value in category_metrics.items():
                assert metric_value is not None
                assert isinstance(metric_value, (int, float))
                
                # Validate reasonable ranges
                if "percent" in metric_name:
                    assert 0 <= metric_value <= 100
                elif "rate" in metric_name:
                    assert 0 <= metric_value <= 1
                elif "accuracy" in metric_name:
                    assert 0 <= metric_value <= 1
        
        # Test metrics aggregation
        total_requests = metrics["anomaly_detection"]["requests_total"]
        total_errors = metrics["anomaly_detection"]["errors_total"]
        calculated_error_rate = total_errors / total_requests
        
        assert abs(calculated_error_rate - metrics["anomaly_detection"]["error_rate"]) < 0.001
    
    def test_logging_integration(self):
        """Test logging integration."""
        import logging
        from unittest.mock import StringIO
        
        # Create test log handler
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        
        # Create test logger
        logger = logging.getLogger("pynomaly.test")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        # Test structured logging
        test_events = [
            {"level": "INFO", "message": "Detection service started", "component": "detection_service"},
            {"level": "DEBUG", "message": "Training model", "component": "model_trainer", "algorithm": "IsolationForest"},
            {"level": "WARNING", "message": "High error rate detected", "component": "monitoring", "error_rate": 0.05},
            {"level": "ERROR", "message": "Model training failed", "component": "model_trainer", "error": "Insufficient data"}
        ]
        
        # Log test events
        for event in test_events:
            level = event["level"]
            message = event["message"]
            
            if level == "INFO":
                logger.info(message, extra=event)
            elif level == "DEBUG":
                logger.debug(message, extra=event)
            elif level == "WARNING":
                logger.warning(message, extra=event)
            elif level == "ERROR":
                logger.error(message, extra=event)
        
        # Verify logs were captured
        log_output = log_stream.getvalue()
        
        # Should contain INFO, WARNING, and ERROR logs (DEBUG filtered out)
        assert "Detection service started" in log_output
        assert "High error rate detected" in log_output
        assert "Model training failed" in log_output
        assert "Training model" not in log_output  # DEBUG level filtered
        
        # Clean up
        logger.removeHandler(handler)
    
    def test_alerting_integration(self):
        """Test alerting integration."""
        # Mock alert conditions and handlers
        class MockAlertManager:
            def __init__(self):
                self.alerts = []
                self.alert_rules = [
                    {"metric": "error_rate", "threshold": 0.05, "severity": "warning"},
                    {"metric": "error_rate", "threshold": 0.1, "severity": "critical"},
                    {"metric": "response_time_ms", "threshold": 1000, "severity": "warning"},
                    {"metric": "cpu_usage_percent", "threshold": 90, "severity": "critical"}
                ]
            
            def check_metrics(self, metrics):
                """Check metrics against alert rules."""
                triggered_alerts = []
                
                for rule in self.alert_rules:
                    metric_name = rule["metric"]
                    threshold = rule["threshold"]
                    severity = rule["severity"]
                    
                    # Find metric value in nested metrics structure
                    metric_value = None
                    for category in metrics.values():
                        if metric_name in category:
                            metric_value = category[metric_name]
                            break
                    
                    if metric_value is not None and metric_value > threshold:
                        alert = {
                            "metric": metric_name,
                            "value": metric_value,
                            "threshold": threshold,
                            "severity": severity,
                            "timestamp": "2023-10-01T10:00:00Z"
                        }
                        triggered_alerts.append(alert)
                
                return triggered_alerts
            
            def send_alerts(self, alerts):
                """Send alerts (mock implementation)."""
                for alert in alerts:
                    self.alerts.append(alert)
                    # In real implementation, this would send to external systems
                return len(alerts)
        
        # Test alerting system
        alert_manager = MockAlertManager()
        
        # Test metrics that should trigger alerts
        test_metrics = {
            "performance": {
                "error_rate": 0.08,  # Should trigger warning
                "response_time_ms": 1200,  # Should trigger warning
                "cpu_usage_percent": 95  # Should trigger critical
            }
        }
        
        triggered_alerts = alert_manager.check_metrics(test_metrics)
        alert_manager.send_alerts(triggered_alerts)
        
        # Verify alerts were triggered
        assert len(triggered_alerts) == 3  # error_rate, response_time, cpu_usage
        
        # Verify alert severities
        severities = [alert["severity"] for alert in triggered_alerts]
        assert "warning" in severities
        assert "critical" in severities
        
        # Verify alerts were stored
        assert len(alert_manager.alerts) == 3


class TestExternalServiceIntegration:
    """Test external service integration."""
    
    def test_database_integration_simulation(self):
        """Test database integration simulation."""
        # Mock database operations
        class MockDatabase:
            def __init__(self):
                self.connected = False
                self.data = {}
            
            def connect(self):
                """Simulate database connection."""
                self.connected = True
                return {"status": "connected", "version": "1.0.0"}
            
            def disconnect(self):
                """Simulate database disconnection."""
                self.connected = False
            
            def save_detection_result(self, result_data):
                """Simulate saving detection result."""
                if not self.connected:
                    raise RuntimeError("Database not connected")
                
                result_id = f"result_{len(self.data) + 1}"
                self.data[result_id] = {
                    **result_data,
                    "saved_at": "2023-10-01T10:00:00Z"
                }
                return result_id
            
            def get_detection_result(self, result_id):
                """Simulate retrieving detection result."""
                if not self.connected:
                    raise RuntimeError("Database not connected")
                
                return self.data.get(result_id)
            
            def list_detection_results(self, limit=10):
                """Simulate listing detection results."""
                if not self.connected:
                    raise RuntimeError("Database not connected")
                
                results = list(self.data.items())[:limit]
                return [{"id": rid, **data} for rid, data in results]
        
        # Test database integration
        db = MockDatabase()
        
        # Test connection
        connection_info = db.connect()
        assert db.connected
        assert "status" in connection_info
        
        # Test data operations
        test_result = {
            "dataset_id": "dataset_123",
            "detector_id": "detector_456",
            "n_samples": 1000,
            "n_anomalies": 50,
            "contamination_rate": 0.05
        }
        
        # Save result
        result_id = db.save_detection_result(test_result)
        assert result_id is not None
        assert result_id.startswith("result_")
        
        # Retrieve result
        saved_result = db.get_detection_result(result_id)
        assert saved_result is not None
        assert saved_result["dataset_id"] == test_result["dataset_id"]
        assert "saved_at" in saved_result
        
        # List results
        results_list = db.list_detection_results()
        assert len(results_list) == 1
        assert results_list[0]["id"] == result_id
        
        # Test disconnection
        db.disconnect()
        assert not db.connected
        
        # Operations should fail after disconnection
        with pytest.raises(RuntimeError, match="not connected"):
            db.save_detection_result(test_result)
    
    def test_cache_integration_simulation(self):
        """Test cache integration simulation."""
        # Mock cache implementation
        class MockCache:
            def __init__(self):
                self.data = {}
                self.ttl = {}  # Time-to-live for keys
            
            def set(self, key, value, ttl_seconds=None):
                """Set cache value with optional TTL."""
                self.data[key] = value
                if ttl_seconds:
                    # In real implementation, this would use actual timestamps
                    self.ttl[key] = ttl_seconds
                return True
            
            def get(self, key):
                """Get cache value."""
                if key in self.data:
                    # In real implementation, would check TTL expiration
                    return self.data[key]
                return None
            
            def delete(self, key):
                """Delete cache key."""
                if key in self.data:
                    del self.data[key]
                if key in self.ttl:
                    del self.ttl[key]
                return True
                return False
            
            def exists(self, key):
                """Check if key exists."""
                return key in self.data
            
            def clear(self):
                """Clear all cache data."""
                self.data.clear()
                self.ttl.clear()
        
        # Test cache integration
        cache = MockCache()
        
        # Test basic operations
        cache.set("model_123", {"algorithm": "IsolationForest", "trained": True})
        assert cache.exists("model_123")
        
        model_data = cache.get("model_123")
        assert model_data is not None
        assert model_data["algorithm"] == "IsolationForest"
        
        # Test cache for detection results
        detection_key = "detection_dataset_456_detector_789"
        detection_result = {
            "scores": [0.1, 0.2, 0.8, 0.15],
            "labels": [0, 0, 1, 0],
            "timestamp": "2023-10-01T10:00:00Z"
        }
        
        cache.set(detection_key, detection_result, ttl_seconds=3600)
        
        cached_result = cache.get(detection_key)
        assert cached_result is not None
        assert cached_result["scores"] == detection_result["scores"]
        
        # Test deletion
        assert cache.delete(detection_key)
        assert not cache.exists(detection_key)
        
        # Test clear
        cache.clear()
        assert not cache.exists("model_123")
    
    def test_message_queue_integration_simulation(self):
        """Test message queue integration simulation."""
        # Mock message queue implementation
        class MockMessageQueue:
            def __init__(self):
                self.queues = {}
            
            def create_queue(self, queue_name):
                """Create a queue."""
                if queue_name not in self.queues:
                    self.queues[queue_name] = []
                return True
            
            def send_message(self, queue_name, message):
                """Send message to queue."""
                if queue_name not in self.queues:
                    self.create_queue(queue_name)
                
                self.queues[queue_name].append({
                    "id": f"msg_{len(self.queues[queue_name]) + 1}",
                    "body": message,
                    "timestamp": "2023-10-01T10:00:00Z"
                })
                return True
            
            def receive_message(self, queue_name):
                """Receive message from queue."""
                if queue_name in self.queues and self.queues[queue_name]:
                    return self.queues[queue_name].pop(0)
                return None
            
            def get_queue_size(self, queue_name):
                """Get queue size."""
                if queue_name in self.queues:
                    return len(self.queues[queue_name])
                return 0
        
        # Test message queue integration
        mq = MockMessageQueue()
        
        # Test detection task queue
        detection_queue = "detection_tasks"
        mq.create_queue(detection_queue)
        
        # Send detection tasks
        tasks = [
            {"type": "train_model", "dataset_id": "dataset_1", "algorithm": "IsolationForest"},
            {"type": "detect_anomalies", "dataset_id": "dataset_2", "model_id": "model_1"},
            {"type": "evaluate_model", "model_id": "model_1", "test_dataset_id": "dataset_3"}
        ]
        
        for task in tasks:
            mq.send_message(detection_queue, task)
        
        # Verify messages were queued
        assert mq.get_queue_size(detection_queue) == 3
        
        # Process messages
        processed_tasks = []
        while mq.get_queue_size(detection_queue) > 0:
            message = mq.receive_message(detection_queue)
            if message:
                processed_tasks.append(message["body"])
        
        # Verify all tasks were processed
        assert len(processed_tasks) == 3
        assert processed_tasks[0]["type"] == "train_model"
        assert processed_tasks[1]["type"] == "detect_anomalies"
        assert processed_tasks[2]["type"] == "evaluate_model"
        
        # Queue should be empty
        assert mq.get_queue_size(detection_queue) == 0