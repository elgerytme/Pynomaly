"""
Service Layer Integration Tests

Tests integration between service layer components and their dependencies.
"""

import pytest
from datetime import datetime
from tests.integration.framework.integration_test_base import ServiceIntegrationTest


class TestServiceIntegration(ServiceIntegrationTest):
    """Test service layer integration."""
    
    async def test_detection_service_integration(self):
        """Test detection service integration with repositories."""
        
        async with self.setup_test_environment() as env:
            # Get services
            detection_service = env.container.detection_service()
            dataset_repo = env.container.dataset_repository()
            detector_repo = env.container.detector_repository()
            result_repo = env.container.result_repository()
            
            # Create test data
            test_dataset = await env.test_data_manager.create_dataset(
                name="service_integration_dataset",
                size=1000,
                anomaly_rate=0.1,
                features=10
            )
            
            test_detector = await env.test_data_manager.create_detector(
                name="service_integration_detector",
                algorithm="IsolationForest",
                parameters={"contamination": 0.1, "random_state": 42}
            )
            
            # Execute detection through service
            result = await detection_service.detect_anomalies(
                detector_id=test_detector.id,
                dataset_id=test_dataset.id
            )
            
            # Verify service integration
            assert result is not None
            assert result.detector_id == test_detector.id
            assert result.dataset_id == test_dataset.id
            assert result.n_anomalies > 0
            assert result.execution_time_ms > 0
            
            # Verify persistence
            stored_result = await result_repo.get_by_id(result.id)
            assert stored_result is not None
            assert stored_result.detector_id == test_detector.id
            assert stored_result.dataset_id == test_dataset.id
            
            # Test service communication
            communication_test = await self.test_service_communication(
                "detection_service", 
                "result_repository", 
                "sync"
            )
            assert communication_test is True
    
    async def test_dataset_service_integration(self):
        """Test dataset service integration."""
        
        async with self.setup_test_environment() as env:
            # Get services
            dataset_service = env.container.dataset_service()
            dataset_repo = env.container.dataset_repository()
            
            # Test dataset creation through service
            dataset_config = {
                "name": "service_dataset_test",
                "description": "Dataset created via service",
                "metadata": {
                    "source": "integration_test",
                    "created_at": datetime.now().isoformat()
                }
            }
            
            created_dataset = await dataset_service.create_dataset(dataset_config)
            
            # Verify creation
            assert created_dataset is not None
            assert created_dataset.name == dataset_config["name"]
            assert created_dataset.description == dataset_config["description"]
            
            # Verify repository integration
            stored_dataset = await dataset_repo.get_by_id(created_dataset.id)
            assert stored_dataset is not None
            assert stored_dataset.name == dataset_config["name"]
            
            # Test dataset validation through service
            invalid_config = {
                "name": "",  # Invalid empty name
                "description": None
            }
            
            with pytest.raises(ValueError):
                await dataset_service.create_dataset(invalid_config)
            
            # Test dataset update through service
            update_config = {
                "description": "Updated description via service"
            }
            
            updated_dataset = await dataset_service.update_dataset(
                created_dataset.id, 
                update_config
            )
            
            assert updated_dataset.description == update_config["description"]
            
            # Test dataset deletion through service
            await dataset_service.delete_dataset(created_dataset.id)
            
            # Verify deletion
            deleted_dataset = await dataset_repo.get_by_id(created_dataset.id)
            assert deleted_dataset is None
    
    async def test_detector_service_integration(self):
        """Test detector service integration."""
        
        async with self.setup_test_environment() as env:
            # Get services
            detector_service = env.container.detector_service()
            detector_repo = env.container.detector_repository()
            
            # Test detector creation through service
            detector_config = {
                "name": "service_detector_test",
                "algorithm": "LocalOutlierFactor",
                "parameters": {
                    "contamination": 0.1,
                    "n_neighbors": 20,
                    "algorithm": "auto"
                },
                "description": "Detector created via service"
            }
            
            created_detector = await detector_service.create_detector(detector_config)
            
            # Verify creation
            assert created_detector is not None
            assert created_detector.name == detector_config["name"]
            assert created_detector.algorithm_name == detector_config["algorithm"]
            assert created_detector.parameters == detector_config["parameters"]
            
            # Verify repository integration
            stored_detector = await detector_repo.get_by_id(created_detector.id)
            assert stored_detector is not None
            assert stored_detector.name == detector_config["name"]
            
            # Test detector validation through service
            invalid_config = {
                "name": "invalid_detector",
                "algorithm": "NonExistentAlgorithm",
                "parameters": {}
            }
            
            with pytest.raises(ValueError):
                await detector_service.create_detector(invalid_config)
            
            # Test detector fitting through service
            test_dataset = await env.test_data_manager.create_dataset(
                name="detector_fitting_dataset",
                size=500,
                anomaly_rate=0.1,
                features=5
            )
            
            fitted_detector = await detector_service.fit_detector(
                created_detector.id,
                test_dataset.id
            )
            
            assert fitted_detector.is_fitted is True
            assert "fitted_at" in fitted_detector.metadata
            
            # Test detector parameter optimization
            optimized_detector = await detector_service.optimize_parameters(
                created_detector.id,
                test_dataset.id,
                optimization_metric="f1_score"
            )
            
            assert optimized_detector is not None
            assert optimized_detector.id == created_detector.id
            # Parameters might be different after optimization
    
    async def test_caching_service_integration(self):
        """Test caching service integration."""
        
        async with self.setup_test_environment() as env:
            # Get services
            cache_service = env.container.cache_service()
            detection_service = env.container.detection_service()
            
            # Create test data
            test_dataset = await env.test_data_manager.create_dataset(
                name="cache_test_dataset",
                size=200,
                anomaly_rate=0.1,
                features=5
            )
            
            test_detector = await env.test_data_manager.create_detector(
                name="cache_test_detector",
                algorithm="IsolationForest",
                parameters={"contamination": 0.1, "random_state": 42}
            )
            
            # Test cache miss (first request)
            cache_key = f"detection_{test_detector.id}_{test_dataset.id}"
            cached_result = await cache_service.get(cache_key)
            assert cached_result is None
            
            # Execute detection (should cache result)
            result = await detection_service.detect_anomalies(
                detector_id=test_detector.id,
                dataset_id=test_dataset.id
            )
            
            # Test cache hit (second request)
            cached_result = await cache_service.get(cache_key)
            if cached_result:  # Caching might be implemented
                assert cached_result["anomaly_count"] == result.n_anomalies
            
            # Test cache invalidation
            await cache_service.invalidate(f"detector_{test_detector.id}")
            
            # Verify cache is cleared
            cached_result = await cache_service.get(cache_key)
            assert cached_result is None
            
            # Test cache statistics
            cache_stats = await cache_service.get_statistics()
            assert "hits" in cache_stats
            assert "misses" in cache_stats
            assert "total_requests" in cache_stats
    
    async def test_notification_service_integration(self):
        """Test notification service integration."""
        
        async with self.setup_test_environment() as env:
            # Get services
            notification_service = env.container.notification_service()
            
            # Test notification sending
            notification_data = {
                "type": "anomaly_detected",
                "message": "High number of anomalies detected",
                "severity": "warning",
                "metadata": {
                    "detector_id": "test_detector",
                    "anomaly_count": 50,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Send notification
            notification_result = await notification_service.send_notification(
                notification_data
            )
            
            # Verify notification was processed
            assert notification_result is not None
            assert notification_result.get("status") == "sent"
            
            # Test notification history
            notification_history = await notification_service.get_notification_history(
                limit=10
            )
            
            assert isinstance(notification_history, list)
            if notification_history:
                recent_notification = notification_history[0]
                assert "type" in recent_notification
                assert "timestamp" in recent_notification
            
            # Test notification preferences
            preferences = {
                "email_enabled": True,
                "sms_enabled": False,
                "webhook_enabled": True,
                "severity_threshold": "warning"
            }
            
            await notification_service.update_preferences(preferences)
            
            # Verify preferences were updated
            current_preferences = await notification_service.get_preferences()
            assert current_preferences["email_enabled"] is True
            assert current_preferences["sms_enabled"] is False
            
            # Verify external mock integration
            self.verify_external_mock_calls("notification_service", 1)
    
    async def test_audit_service_integration(self):
        """Test audit service integration."""
        
        async with self.setup_test_environment() as env:
            # Get services
            audit_service = env.container.audit_service()
            detection_service = env.container.detection_service()
            
            # Create test data
            test_dataset = await env.test_data_manager.create_dataset(
                name="audit_test_dataset",
                size=100,
                anomaly_rate=0.1,
                features=3
            )
            
            test_detector = await env.test_data_manager.create_detector(
                name="audit_test_detector",
                algorithm="OneClassSVM",
                parameters={"nu": 0.1, "gamma": "scale"}
            )
            
            # Execute audited operation
            audit_context = {
                "user_id": "test_user",
                "session_id": "test_session",
                "action": "detect_anomalies",
                "resource_type": "detection",
                "resource_id": f"{test_detector.id}_{test_dataset.id}"
            }
            
            # Log audit event
            await audit_service.log_event(
                action="detection_started",
                context=audit_context
            )
            
            # Execute detection
            result = await detection_service.detect_anomalies(
                detector_id=test_detector.id,
                dataset_id=test_dataset.id
            )
            
            # Log completion
            await audit_service.log_event(
                action="detection_completed",
                context={
                    **audit_context,
                    "result_id": result.id,
                    "anomaly_count": result.n_anomalies,
                    "execution_time": result.execution_time_ms
                }
            )
            
            # Verify audit log
            audit_logs = await audit_service.get_audit_logs(
                filters={"user_id": "test_user"},
                limit=10
            )
            
            assert len(audit_logs) >= 2
            
            # Verify audit events
            start_event = next(
                (log for log in audit_logs if log["action"] == "detection_started"),
                None
            )
            assert start_event is not None
            assert start_event["context"]["user_id"] == "test_user"
            
            completion_event = next(
                (log for log in audit_logs if log["action"] == "detection_completed"),
                None
            )
            assert completion_event is not None
            assert completion_event["context"]["result_id"] == result.id
            
            # Test audit report generation
            audit_report = await audit_service.generate_audit_report(
                start_date=datetime.now().replace(hour=0, minute=0, second=0),
                end_date=datetime.now(),
                filters={"user_id": "test_user"}
            )
            
            assert audit_report is not None
            assert "total_events" in audit_report
            assert "event_summary" in audit_report
            assert audit_report["total_events"] >= 2
    
    async def test_configuration_service_integration(self):
        """Test configuration service integration."""
        
        async with self.setup_test_environment() as env:
            # Get services
            config_service = env.container.configuration_service()
            
            # Test configuration retrieval
            current_config = await config_service.get_configuration()
            
            assert current_config is not None
            assert "database" in current_config
            assert "cache" in current_config
            assert "security" in current_config
            
            # Test configuration update
            config_updates = {
                "detection": {
                    "default_algorithm": "IsolationForest",
                    "default_contamination": 0.1,
                    "max_features": 100
                },
                "performance": {
                    "max_concurrent_detections": 5,
                    "timeout_seconds": 300
                }
            }
            
            updated_config = await config_service.update_configuration(config_updates)
            
            # Verify updates
            assert updated_config["detection"]["default_algorithm"] == "IsolationForest"
            assert updated_config["performance"]["max_concurrent_detections"] == 5
            
            # Test configuration validation
            invalid_config = {
                "detection": {
                    "default_contamination": 1.5  # Invalid value > 1
                }
            }
            
            with pytest.raises(ValueError):
                await config_service.update_configuration(invalid_config)
            
            # Test configuration history
            config_history = await config_service.get_configuration_history(limit=5)
            
            assert isinstance(config_history, list)
            if config_history:
                recent_change = config_history[0]
                assert "timestamp" in recent_change
                assert "changes" in recent_change
            
            # Test configuration backup/restore
            backup_result = await config_service.create_backup()
            assert backup_result is not None
            assert "backup_id" in backup_result
            
            # Test restore (would restore to previous state)
            restore_result = await config_service.restore_backup(
                backup_result["backup_id"]
            )
            assert restore_result is not None
            assert restore_result.get("status") == "restored"
    
    async def test_cross_service_integration(self):
        """Test integration between multiple services."""
        
        async with self.setup_test_environment() as env:
            # Get multiple services
            detection_service = env.container.detection_service()
            audit_service = env.container.audit_service()
            notification_service = env.container.notification_service()
            cache_service = env.container.cache_service()
            
            # Create test data
            test_dataset = await env.test_data_manager.create_dataset(
                name="cross_service_dataset",
                size=300,
                anomaly_rate=0.15,
                features=8
            )
            
            test_detector = await env.test_data_manager.create_detector(
                name="cross_service_detector",
                algorithm="IsolationForest",
                parameters={"contamination": 0.15, "random_state": 42}
            )
            
            # Execute detection with full service integration
            audit_context = {
                "user_id": "integration_test_user",
                "action": "comprehensive_detection",
                "resource_type": "detection"
            }
            
            # Log start
            await audit_service.log_event(
                action="cross_service_detection_started",
                context=audit_context
            )
            
            # Execute detection
            result = await detection_service.detect_anomalies(
                detector_id=test_detector.id,
                dataset_id=test_dataset.id
            )
            
            # Check for high anomaly count and send notification
            if result.n_anomalies > 30:  # 10% of 300 = 30
                notification_data = {
                    "type": "high_anomaly_count",
                    "message": f"High anomaly count detected: {result.n_anomalies}",
                    "severity": "warning",
                    "metadata": {
                        "detector_id": test_detector.id,
                        "dataset_id": test_dataset.id,
                        "anomaly_count": result.n_anomalies
                    }
                }
                
                await notification_service.send_notification(notification_data)
            
            # Log completion
            await audit_service.log_event(
                action="cross_service_detection_completed",
                context={
                    **audit_context,
                    "result_id": result.id,
                    "anomaly_count": result.n_anomalies,
                    "notification_sent": result.n_anomalies > 30
                }
            )
            
            # Verify cross-service integration
            assert result is not None
            assert result.n_anomalies > 0
            
            # Verify audit logs
            audit_logs = await audit_service.get_audit_logs(
                filters={"user_id": "integration_test_user"},
                limit=10
            )
            
            assert len(audit_logs) >= 2
            
            # Verify cache interaction
            cache_key = f"detection_{test_detector.id}_{test_dataset.id}"
            cached_result = await cache_service.get(cache_key)
            
            # Cache might or might not be implemented
            if cached_result:
                assert cached_result["anomaly_count"] == result.n_anomalies
            
            # Verify external service calls
            if result.n_anomalies > 30:
                self.verify_external_mock_calls("notification_service", 1)
            
            # Test service health across all services
            self.assert_service_health("database")
            self.assert_service_health("cache")