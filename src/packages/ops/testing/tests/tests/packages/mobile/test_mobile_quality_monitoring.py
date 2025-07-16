"""Tests for Mobile Quality Monitoring Services."""

import pytest
import asyncio
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from src.packages.mobile.quality_monitoring.mobile_quality_service import (
    MobileQualityMonitoringService, MobileAlert, MobileIncident, MobileQualityDashboard,
    AlertSeverity, AlertType
)
from src.packages.mobile.notifications.push_notification_service import (
    PushNotificationService, PushNotification, PushSubscription, 
    NotificationPlatform, NotificationPriority
)
from src.packages.mobile.storage.offline_storage_service import (
    OfflineStorageService, OfflineData, SyncOperation
)


class TestMobileQualityMonitoringService:
    """Test suite for Mobile Quality Monitoring Service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = MobileQualityMonitoringService()
        self.test_user_id = "test_user_123"
        self.test_dataset_id = "test_dataset_456"
    
    @pytest.mark.asyncio
    async def test_create_mobile_dashboard(self):
        """Test creating mobile dashboard."""
        favorite_datasets = ["dataset1", "dataset2", "dataset3"]
        
        dashboard = await self.service.create_mobile_dashboard(
            user_id=self.test_user_id,
            favorite_datasets=favorite_datasets
        )
        
        assert isinstance(dashboard, MobileQualityDashboard)
        assert dashboard.user_id == self.test_user_id
        assert dashboard.favorite_datasets == favorite_datasets
        assert dashboard.dashboard_id in self.service.mobile_dashboards
        
        # Check that dashboard data is populated
        assert dashboard.overall_quality_score > 0
        assert dashboard.total_datasets > 0
        assert len(dashboard.quality_metrics) > 0
        assert len(dashboard.trending_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_get_mobile_dashboard(self):
        """Test getting mobile dashboard."""
        # Create dashboard first
        dashboard = await self.service.create_mobile_dashboard(
            user_id=self.test_user_id
        )
        
        # Get dashboard
        retrieved_dashboard = await self.service.get_mobile_dashboard(
            dashboard.dashboard_id
        )
        
        assert retrieved_dashboard is not None
        assert retrieved_dashboard.dashboard_id == dashboard.dashboard_id
        assert retrieved_dashboard.user_id == self.test_user_id
    
    @pytest.mark.asyncio
    async def test_refresh_mobile_dashboard(self):
        """Test refreshing mobile dashboard."""
        # Create dashboard first
        dashboard = await self.service.create_mobile_dashboard(
            user_id=self.test_user_id
        )
        
        original_sync_time = dashboard.last_sync_time
        
        # Wait a bit to ensure time difference
        await asyncio.sleep(0.1)
        
        # Refresh dashboard
        refreshed_dashboard = await self.service.refresh_mobile_dashboard(
            dashboard.dashboard_id
        )
        
        assert refreshed_dashboard is not None
        assert refreshed_dashboard.last_sync_time > original_sync_time
    
    @pytest.mark.asyncio
    async def test_create_mobile_alert(self):
        """Test creating mobile alert."""
        alert = await self.service.create_mobile_alert(
            dataset_id=self.test_dataset_id,
            metric_name="completeness",
            alert_type=AlertType.THRESHOLD_BREACH,
            severity=AlertSeverity.HIGH,
            title="Data Completeness Alert",
            message="Completeness has fallen below threshold",
            metadata={"threshold": 0.9, "current_value": 0.85}
        )
        
        assert isinstance(alert, MobileAlert)
        assert alert.dataset_id == self.test_dataset_id
        assert alert.metric_name == "completeness"
        assert alert.alert_type == AlertType.THRESHOLD_BREACH
        assert alert.severity == AlertSeverity.HIGH
        assert alert.title == "Data Completeness Alert"
        assert alert.requires_immediate_attention is True  # High severity
        assert len(alert.mobile_actions) > 0
        assert alert.alert_id in self.service.mobile_alerts
    
    @pytest.mark.asyncio
    async def test_create_mobile_incident(self):
        """Test creating mobile incident."""
        affected_datasets = ["dataset1", "dataset2"]
        affected_metrics = ["completeness", "accuracy"]
        
        incident = await self.service.create_mobile_incident(
            title="Data Quality Degradation",
            description="Multiple datasets showing quality issues",
            severity=AlertSeverity.CRITICAL,
            affected_datasets=affected_datasets,
            affected_metrics=affected_metrics
        )
        
        assert isinstance(incident, MobileIncident)
        assert incident.title == "Data Quality Degradation"
        assert incident.severity == AlertSeverity.CRITICAL
        assert incident.affected_datasets == affected_datasets
        assert incident.affected_metrics == affected_metrics
        assert len(incident.mobile_workflow_steps) > 0
        assert incident.next_action is not None
        assert incident.quality_impact_score > 0
        assert incident.incident_id in self.service.mobile_incidents
    
    @pytest.mark.asyncio
    async def test_get_mobile_alerts(self):
        """Test getting mobile alerts."""
        # Create some alerts
        await self.service.create_mobile_alert(
            dataset_id=self.test_dataset_id,
            metric_name="completeness",
            alert_type=AlertType.THRESHOLD_BREACH,
            severity=AlertSeverity.HIGH,
            title="Alert 1",
            message="Test alert 1"
        )
        
        await self.service.create_mobile_alert(
            dataset_id=self.test_dataset_id,
            metric_name="accuracy",
            alert_type=AlertType.ANOMALY_DETECTED,
            severity=AlertSeverity.MEDIUM,
            title="Alert 2",
            message="Test alert 2"
        )
        
        # Get all alerts
        alerts = await self.service.get_mobile_alerts(
            user_id=self.test_user_id,
            limit=10
        )
        
        assert len(alerts) == 2
        assert all(isinstance(alert, dict) for alert in alerts)
        assert all('alert_id' in alert for alert in alerts)
        assert all('title' in alert for alert in alerts)
        assert all('severity' in alert for alert in alerts)
        
        # Test severity filtering
        high_alerts = await self.service.get_mobile_alerts(
            user_id=self.test_user_id,
            severity_filter=AlertSeverity.HIGH
        )
        
        assert len(high_alerts) == 1
        assert high_alerts[0]['severity'] == AlertSeverity.HIGH.value
    
    @pytest.mark.asyncio
    async def test_get_mobile_incidents(self):
        """Test getting mobile incidents."""
        # Create some incidents
        await self.service.create_mobile_incident(
            title="Incident 1",
            description="Test incident 1",
            severity=AlertSeverity.HIGH,
            affected_datasets=["dataset1"]
        )
        
        await self.service.create_mobile_incident(
            title="Incident 2",
            description="Test incident 2",
            severity=AlertSeverity.MEDIUM,
            affected_datasets=["dataset2"]
        )
        
        # Get all incidents
        incidents = await self.service.get_mobile_incidents(
            user_id=self.test_user_id,
            limit=10
        )
        
        assert len(incidents) == 2
        assert all(isinstance(incident, dict) for incident in incidents)
        assert all('incident_id' in incident for incident in incidents)
        assert all('title' in incident for incident in incidents)
        assert all('severity' in incident for incident in incidents)
    
    @pytest.mark.asyncio
    async def test_resolve_mobile_alert(self):
        """Test resolving mobile alert."""
        # Create alert
        alert = await self.service.create_mobile_alert(
            dataset_id=self.test_dataset_id,
            metric_name="completeness",
            alert_type=AlertType.THRESHOLD_BREACH,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message"
        )
        
        # Resolve alert
        success = await self.service.resolve_mobile_alert(
            alert_id=alert.alert_id,
            resolution_notes="Fixed by updating data source"
        )
        
        assert success is True
        assert alert.resolved_at is not None
        assert alert.metadata.get('resolution_notes') == "Fixed by updating data source"
    
    @pytest.mark.asyncio
    async def test_update_incident_status(self):
        """Test updating incident status."""
        # Create incident
        incident = await self.service.create_mobile_incident(
            title="Test Incident",
            description="Test description",
            severity=AlertSeverity.HIGH,
            affected_datasets=["dataset1"]
        )
        
        # Update incident status
        success = await self.service.update_incident_status(
            incident_id=incident.incident_id,
            status="in_progress",
            completed_step=incident.mobile_workflow_steps[0] if incident.mobile_workflow_steps else None,
            notes="Started investigating"
        )
        
        assert success is True
        assert incident.status == "in_progress"
        if incident.mobile_workflow_steps:
            assert incident.mobile_workflow_steps[0] in incident.completed_steps
    
    @pytest.mark.asyncio
    async def test_get_quality_metrics_for_mobile(self):
        """Test getting quality metrics for mobile."""
        metrics = await self.service.get_quality_metrics_for_mobile(
            dataset_id=self.test_dataset_id
        )
        
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        
        for metric in metrics:
            assert 'metric_name' in metric
            assert 'current_value' in metric
            assert 'threshold' in metric
            assert 'trend' in metric
            assert 'status' in metric
            assert 'mobile_chart_data' in metric
    
    @pytest.mark.asyncio
    async def test_get_mobile_summary(self):
        """Test getting mobile summary."""
        summary = await self.service.get_mobile_summary(
            user_id=self.test_user_id
        )
        
        assert isinstance(summary, dict)
        assert 'user_id' in summary
        assert 'overall_status' in summary
        assert 'quality_score' in summary
        assert 'alerts' in summary
        assert 'incidents' in summary
        assert 'trending_up' in summary
        assert 'trending_down' in summary
        assert 'requires_attention' in summary
        assert 'recent_activity' in summary
    
    @pytest.mark.asyncio
    async def test_sync_offline_data(self):
        """Test syncing offline data."""
        # Create dashboard first
        await self.service.create_mobile_dashboard(
            user_id=self.test_user_id,
            favorite_datasets=["dataset1", "dataset2"]
        )
        
        # Sync offline data
        offline_data = await self.service.sync_offline_data(
            user_id=self.test_user_id
        )
        
        assert isinstance(offline_data, dict)
        assert 'sync_timestamp' in offline_data
        assert 'dashboard' in offline_data
        assert 'recent_alerts' in offline_data
        assert 'active_incidents' in offline_data
        assert 'quality_metrics' in offline_data
        assert 'offline_actions' in offline_data
        
        # Check that quality metrics are included for favorite datasets
        assert 'dataset1' in offline_data['quality_metrics']
        assert 'dataset2' in offline_data['quality_metrics']
    
    def test_mobile_alert_to_mobile_dict(self):
        """Test MobileAlert to_mobile_dict method."""
        alert = MobileAlert(
            alert_type=AlertType.THRESHOLD_BREACH,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            dataset_id=self.test_dataset_id,
            metric_name="completeness",
            requires_immediate_attention=True,
            mobile_actions=["action1", "action2"]
        )
        
        mobile_dict = alert.to_mobile_dict()
        
        assert isinstance(mobile_dict, dict)
        assert mobile_dict['alert_id'] == alert.alert_id
        assert mobile_dict['type'] == AlertType.THRESHOLD_BREACH.value
        assert mobile_dict['severity'] == AlertSeverity.HIGH.value
        assert mobile_dict['title'] == "Test Alert"
        assert mobile_dict['message'] == "Test message"
        assert mobile_dict['dataset_id'] == self.test_dataset_id
        assert mobile_dict['metric_name'] == "completeness"
        assert mobile_dict['requires_attention'] is True
        assert mobile_dict['mobile_actions'] == ["action1", "action2"]
        assert 'created_at' in mobile_dict
        assert 'age_minutes' in mobile_dict
    
    def test_mobile_incident_to_mobile_dict(self):
        """Test MobileIncident to_mobile_dict method."""
        incident = MobileIncident(
            title="Test Incident",
            description="Test description",
            severity=AlertSeverity.CRITICAL,
            affected_datasets=["dataset1", "dataset2"],
            mobile_workflow_steps=["step1", "step2", "step3"],
            completed_steps=["step1"]
        )
        
        mobile_dict = incident.to_mobile_dict()
        
        assert isinstance(mobile_dict, dict)
        assert mobile_dict['incident_id'] == incident.incident_id
        assert mobile_dict['title'] == "Test Incident"
        assert mobile_dict['description'] == "Test description"
        assert mobile_dict['severity'] == AlertSeverity.CRITICAL.value
        assert mobile_dict['affected_datasets'] == ["dataset1", "dataset2"]
        assert mobile_dict['mobile_workflow_steps'] == ["step1", "step2", "step3"]
        assert mobile_dict['completed_steps'] == ["step1"]
        assert mobile_dict['progress_percentage'] == 100 / 3  # 1 of 3 steps completed
        assert 'created_at' in mobile_dict
        assert 'age_hours' in mobile_dict
    
    def test_mobile_dashboard_to_mobile_dict(self):
        """Test MobileQualityDashboard to_mobile_dict method."""
        dashboard = MobileQualityDashboard(
            user_id=self.test_user_id,
            overall_quality_score=0.85,
            total_datasets=10,
            healthy_datasets=8,
            degraded_datasets=2,
            critical_datasets=0,
            active_alerts=5,
            critical_alerts=1,
            favorite_datasets=["dataset1", "dataset2"]
        )
        
        mobile_dict = dashboard.to_mobile_dict()
        
        assert isinstance(mobile_dict, dict)
        assert mobile_dict['user_id'] == self.test_user_id
        assert mobile_dict['overall_quality_score'] == 0.85
        assert mobile_dict['dataset_summary']['total'] == 10
        assert mobile_dict['dataset_summary']['healthy'] == 8
        assert mobile_dict['dataset_summary']['degraded'] == 2
        assert mobile_dict['dataset_summary']['critical'] == 0
        assert mobile_dict['dataset_summary']['health_percentage'] == 80.0
        assert mobile_dict['alert_summary']['active'] == 5
        assert mobile_dict['alert_summary']['critical'] == 1
        assert mobile_dict['favorite_datasets'] == ["dataset1", "dataset2"]


class TestPushNotificationService:
    """Test suite for Push Notification Service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = PushNotificationService()
        self.test_user_id = "test_user_123"
    
    @pytest.mark.asyncio
    async def test_subscribe_to_push_notifications(self):
        """Test subscribing to push notifications."""
        subscription = PushSubscription(
            user_id=self.test_user_id,
            platform=NotificationPlatform.FCM,
            token="test_fcm_token",
            enabled=True
        )
        
        success = await self.service.subscribe(subscription)
        
        assert success is True
        assert self.test_user_id in self.service.subscriptions
        assert len(self.service.subscriptions[self.test_user_id]) == 1
        assert self.service.subscriptions[self.test_user_id][0].platform == NotificationPlatform.FCM
    
    @pytest.mark.asyncio
    async def test_unsubscribe_from_push_notifications(self):
        """Test unsubscribing from push notifications."""
        # Subscribe first
        subscription = PushSubscription(
            user_id=self.test_user_id,
            platform=NotificationPlatform.FCM,
            token="test_fcm_token"
        )
        await self.service.subscribe(subscription)
        
        # Unsubscribe
        success = await self.service.unsubscribe(
            user_id=self.test_user_id,
            platform=NotificationPlatform.FCM
        )
        
        assert success is True
        assert len(self.service.subscriptions[self.test_user_id]) == 0
    
    @pytest.mark.asyncio
    async def test_send_notification_no_subscription(self):
        """Test sending notification with no subscription."""
        notification = PushNotification(
            notification_id="test_notification_123",
            user_id=self.test_user_id,
            title="Test Notification",
            body="Test message",
            priority=NotificationPriority.HIGH
        )
        
        result = await self.service.send_notification(notification)
        
        assert result["success"] is False
        assert "No subscriptions found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_send_bulk_notifications(self):
        """Test sending bulk notifications."""
        notifications = [
            PushNotification(
                notification_id=f"test_notification_{i}",
                user_id=self.test_user_id,
                title=f"Test Notification {i}",
                body=f"Test message {i}"
            )
            for i in range(3)
        ]
        
        results = await self.service.send_bulk_notifications(notifications)
        
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
        assert all('notification_id' in result for result in results)
        assert all('success' in result for result in results)
    
    @pytest.mark.asyncio
    async def test_get_subscription_info(self):
        """Test getting subscription info."""
        # Subscribe to multiple platforms
        fcm_subscription = PushSubscription(
            user_id=self.test_user_id,
            platform=NotificationPlatform.FCM,
            token="test_fcm_token"
        )
        
        web_subscription = PushSubscription(
            user_id=self.test_user_id,
            platform=NotificationPlatform.WEB_PUSH,
            token="test_web_token"
        )
        
        await self.service.subscribe(fcm_subscription)
        await self.service.subscribe(web_subscription)
        
        # Get subscription info
        info = await self.service.get_subscription_info(self.test_user_id)
        
        assert info['user_id'] == self.test_user_id
        assert info['subscription_count'] == 2
        assert NotificationPlatform.FCM.value in info['platforms']
        assert NotificationPlatform.WEB_PUSH.value in info['platforms']
        assert info['enabled_count'] == 2
    
    @pytest.mark.asyncio
    async def test_get_notification_stats(self):
        """Test getting notification statistics."""
        # Add some notifications to history
        notification1 = PushNotification(
            notification_id="test1",
            user_id=self.test_user_id,
            title="Test 1",
            body="Test message 1",
            priority=NotificationPriority.HIGH
        )
        notification1.sent_at = datetime.utcnow()
        
        notification2 = PushNotification(
            notification_id="test2",
            user_id=self.test_user_id,
            title="Test 2",
            body="Test message 2",
            priority=NotificationPriority.NORMAL
        )
        notification2.sent_at = datetime.utcnow()
        
        self.service.notification_history["test1"] = notification1
        self.service.notification_history["test2"] = notification2
        
        # Get stats
        stats = await self.service.get_notification_stats(self.test_user_id)
        
        assert stats['user_id'] == self.test_user_id
        assert stats['total_notifications'] == 2
        assert stats['sent_notifications'] == 2
        assert stats['by_priority']['high'] == 1
        assert stats['by_priority']['normal'] == 1
    
    def test_push_notification_to_fcm_payload(self):
        """Test converting PushNotification to FCM payload."""
        notification = PushNotification(
            notification_id="test_123",
            user_id=self.test_user_id,
            title="Test Notification",
            body="Test message",
            priority=NotificationPriority.HIGH,
            icon="https://example.com/icon.png",
            click_action="https://example.com/action",
            data={"key1": "value1", "key2": "value2"}
        )
        
        payload = notification.to_fcm_payload()
        
        assert payload['notification']['title'] == "Test Notification"
        assert payload['notification']['body'] == "Test message"
        assert payload['notification']['icon'] == "https://example.com/icon.png"
        assert payload['notification']['click_action'] == "https://example.com/action"
        assert payload['data'] == {"key1": "value1", "key2": "value2"}
        assert payload['android']['priority'] == "high"
        assert payload['apns']['headers']['apns-priority'] == "10"
    
    def test_push_notification_to_web_push_payload(self):
        """Test converting PushNotification to Web Push payload."""
        notification = PushNotification(
            notification_id="test_123",
            user_id=self.test_user_id,
            title="Test Notification",
            body="Test message",
            priority=NotificationPriority.HIGH,
            icon="https://example.com/icon.png",
            badge="https://example.com/badge.png",
            collapse_key="test_collapse",
            click_action="https://example.com/action",
            data={"key1": "value1", "key2": "value2"}
        )
        
        payload = notification.to_web_push_payload()
        
        assert payload['notification']['title'] == "Test Notification"
        assert payload['notification']['body'] == "Test message"
        assert payload['notification']['icon'] == "https://example.com/icon.png"
        assert payload['notification']['badge'] == "https://example.com/badge.png"
        assert payload['notification']['tag'] == "test_collapse"
        assert payload['notification']['requireInteraction'] is True
        assert payload['data'] == {"key1": "value1", "key2": "value2"}
        assert len(payload['notification']['actions']) == 1


class TestOfflineStorageService:
    """Test suite for Offline Storage Service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.service = OfflineStorageService(
            storage_path=self.temp_dir,
            config={'auto_cleanup_enabled': False}  # Disable for tests
        )
        self.test_user_id = "test_user_123"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_save_and_get_data(self):
        """Test saving and retrieving data."""
        test_data = OfflineData(
            data_id="test_data_123",
            user_id=self.test_user_id,
            data_type="dashboard",
            data={"key1": "value1", "key2": "value2"},
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        # Save data
        success = await self.service.save_data(test_data)
        assert success is True
        
        # Get data
        retrieved_data = await self.service.get_data("test_data_123")
        
        assert retrieved_data is not None
        assert retrieved_data.data_id == "test_data_123"
        assert retrieved_data.user_id == self.test_user_id
        assert retrieved_data.data_type == "dashboard"
        assert retrieved_data.data == {"key1": "value1", "key2": "value2"}
    
    @pytest.mark.asyncio
    async def test_get_user_data(self):
        """Test getting all data for a user."""
        # Save multiple data items
        data1 = OfflineData(
            data_id="data1",
            user_id=self.test_user_id,
            data_type="dashboard",
            data={"dashboard": "data1"}
        )
        
        data2 = OfflineData(
            data_id="data2",
            user_id=self.test_user_id,
            data_type="alerts",
            data={"alerts": "data2"}
        )
        
        await self.service.save_data(data1)
        await self.service.save_data(data2)
        
        # Get all user data
        user_data = await self.service.get_user_data(self.test_user_id)
        
        assert len(user_data) == 2
        assert any(data.data_id == "data1" for data in user_data)
        assert any(data.data_id == "data2" for data in user_data)
        
        # Get filtered user data
        dashboard_data = await self.service.get_user_data(
            self.test_user_id, 
            data_type="dashboard"
        )
        
        assert len(dashboard_data) == 1
        assert dashboard_data[0].data_id == "data1"
    
    @pytest.mark.asyncio
    async def test_delete_data(self):
        """Test deleting data."""
        test_data = OfflineData(
            data_id="test_data_123",
            user_id=self.test_user_id,
            data_type="dashboard",
            data={"test": "data"}
        )
        
        # Save data
        await self.service.save_data(test_data)
        
        # Verify data exists
        retrieved_data = await self.service.get_data("test_data_123")
        assert retrieved_data is not None
        
        # Delete data
        success = await self.service.delete_data("test_data_123")
        assert success is True
        
        # Verify data is deleted
        retrieved_data = await self.service.get_data("test_data_123")
        assert retrieved_data is None
    
    @pytest.mark.asyncio
    async def test_save_dashboard(self):
        """Test saving dashboard data."""
        dashboard_data = {
            "dashboard_id": "dashboard_123",
            "user_id": self.test_user_id,
            "overall_quality_score": 0.85,
            "total_datasets": 10
        }
        
        success = await self.service.save_dashboard(dashboard_data)
        assert success is True
        
        # Verify data was saved
        retrieved_data = await self.service.get_data("dashboard_dashboard_123")
        assert retrieved_data is not None
        assert retrieved_data.data_type == "dashboard"
        assert retrieved_data.data == dashboard_data
    
    @pytest.mark.asyncio
    async def test_save_alerts(self):
        """Test saving alerts data."""
        alerts = [
            {"alert_id": "alert1", "title": "Alert 1"},
            {"alert_id": "alert2", "title": "Alert 2"}
        ]
        
        success = await self.service.save_alerts(self.test_user_id, alerts)
        assert success is True
        
        # Verify data was saved
        retrieved_data = await self.service.get_data(f"alerts_{self.test_user_id}")
        assert retrieved_data is not None
        assert retrieved_data.data_type == "alerts"
        assert retrieved_data.data["alerts"] == alerts
    
    @pytest.mark.asyncio
    async def test_save_and_get_offline_data(self):
        """Test saving and getting complete offline data package."""
        offline_package = {
            "sync_timestamp": datetime.utcnow().isoformat(),
            "dashboard": {"dashboard_id": "test"},
            "alerts": [{"alert_id": "alert1"}],
            "incidents": [{"incident_id": "incident1"}]
        }
        
        success = await self.service.save_offline_data(
            self.test_user_id, 
            offline_package
        )
        assert success is True
        
        # Get offline data
        retrieved_package = await self.service.get_offline_data(self.test_user_id)
        assert retrieved_package is not None
        assert retrieved_package == offline_package
    
    @pytest.mark.asyncio
    async def test_sync_operations(self):
        """Test sync operations."""
        operation = SyncOperation(
            operation_id="sync_op_123",
            user_id=self.test_user_id,
            operation_type="upload",
            data_type="dashboard",
            data_id="dashboard_123"
        )
        
        # Create sync operation
        success = await self.service.create_sync_operation(operation)
        assert success is True
        
        # Update sync operation
        success = await self.service.update_sync_operation(
            operation_id="sync_op_123",
            status="completed",
            progress=100.0,
            result={"success": True}
        )
        assert success is True
        
        # Get sync operations
        operations = await self.service.get_sync_operations(self.test_user_id)
        assert len(operations) == 1
        assert operations[0].operation_id == "sync_op_123"
        assert operations[0].status == "completed"
        assert operations[0].progress_percentage == 100.0
    
    @pytest.mark.asyncio
    async def test_get_storage_stats(self):
        """Test getting storage statistics."""
        # Save some data
        data1 = OfflineData(
            data_id="data1",
            user_id=self.test_user_id,
            data_type="dashboard",
            data={"test": "data1"}
        )
        
        data2 = OfflineData(
            data_id="data2",
            user_id=self.test_user_id,
            data_type="alerts",
            data={"test": "data2"}
        )
        
        await self.service.save_data(data1)
        await self.service.save_data(data2)
        
        # Get stats
        stats = await self.service.get_storage_stats(self.test_user_id)
        
        assert stats['total_items'] == 2
        assert stats['total_size_bytes'] > 0
        assert stats['by_type']['dashboard']['count'] == 1
        assert stats['by_type']['alerts']['count'] == 1
        assert stats['storage_usage_percentage'] >= 0
    
    @pytest.mark.asyncio
    async def test_expired_data_handling(self):
        """Test handling of expired data."""
        # Save expired data
        expired_data = OfflineData(
            data_id="expired_data",
            user_id=self.test_user_id,
            data_type="dashboard",
            data={"test": "expired"},
            expires_at=datetime.utcnow() - timedelta(hours=1)  # Already expired
        )
        
        await self.service.save_data(expired_data)
        
        # Try to get expired data
        retrieved_data = await self.service.get_data("expired_data")
        assert retrieved_data is None  # Should be None because it's expired
    
    @pytest.mark.asyncio
    async def test_export_import_user_data(self):
        """Test exporting and importing user data."""
        # Save some data
        data1 = OfflineData(
            data_id="data1",
            user_id=self.test_user_id,
            data_type="dashboard",
            data={"test": "data1"}
        )
        
        await self.service.save_data(data1)
        
        # Export user data
        export_data = await self.service.export_user_data(self.test_user_id)
        
        assert export_data['user_id'] == self.test_user_id
        assert len(export_data['data_items']) == 1
        assert export_data['data_items'][0]['data_id'] == "data1"
        
        # Clear user data
        await self.service.clear_user_data(self.test_user_id)
        
        # Verify data is cleared
        user_data = await self.service.get_user_data(self.test_user_id)
        assert len(user_data) == 0
        
        # Import user data
        success = await self.service.import_user_data(self.test_user_id, export_data)
        assert success is True
        
        # Verify data is imported
        user_data = await self.service.get_user_data(self.test_user_id)
        assert len(user_data) == 1
        assert user_data[0].data_id == "data1"
    
    def test_offline_data_to_dict(self):
        """Test OfflineData to_dict method."""
        data = OfflineData(
            data_id="test_123",
            user_id=self.test_user_id,
            data_type="dashboard",
            data={"key": "value"},
            version=2,
            size_bytes=100,
            compressed=True,
            priority=3
        )
        
        data_dict = data.to_dict()
        
        assert data_dict['data_id'] == "test_123"
        assert data_dict['user_id'] == self.test_user_id
        assert data_dict['data_type'] == "dashboard"
        assert data_dict['data'] == {"key": "value"}
        assert data_dict['version'] == 2
        assert data_dict['size_bytes'] == 100
        assert data_dict['compressed'] is True
        assert data_dict['priority'] == 3
        assert 'created_at' in data_dict
        assert 'updated_at' in data_dict
    
    def test_offline_data_is_expired(self):
        """Test OfflineData is_expired method."""
        # Not expired data
        data1 = OfflineData(
            data_id="data1",
            user_id=self.test_user_id,
            data_type="dashboard",
            data={"test": "data"},
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        assert data1.is_expired() is False
        
        # Expired data
        data2 = OfflineData(
            data_id="data2",
            user_id=self.test_user_id,
            data_type="dashboard",
            data={"test": "data"},
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        
        assert data2.is_expired() is True
        
        # No expiration data
        data3 = OfflineData(
            data_id="data3",
            user_id=self.test_user_id,
            data_type="dashboard",
            data={"test": "data"}
        )
        
        assert data3.is_expired() is False


@pytest.mark.asyncio
async def test_mobile_quality_integration():
    """Integration test for mobile quality monitoring components."""
    # Create services
    push_service = PushNotificationService()
    storage_service = OfflineStorageService(
        storage_path=tempfile.mkdtemp(),
        config={'auto_cleanup_enabled': False}
    )
    
    quality_service = MobileQualityMonitoringService(
        push_notification_service=push_service,
        offline_storage_service=storage_service
    )
    
    test_user_id = "integration_test_user"
    
    try:
        # Create dashboard
        dashboard = await quality_service.create_mobile_dashboard(
            user_id=test_user_id,
            favorite_datasets=["dataset1", "dataset2"]
        )
        
        assert dashboard is not None
        
        # Create alert
        alert = await quality_service.create_mobile_alert(
            dataset_id="dataset1",
            metric_name="completeness",
            alert_type=AlertType.THRESHOLD_BREACH,
            severity=AlertSeverity.HIGH,
            title="Integration Test Alert",
            message="Test alert message"
        )
        
        assert alert is not None
        
        # Create incident
        incident = await quality_service.create_mobile_incident(
            title="Integration Test Incident",
            description="Test incident description",
            severity=AlertSeverity.CRITICAL,
            affected_datasets=["dataset1", "dataset2"]
        )
        
        assert incident is not None
        
        # Sync offline data
        offline_data = await quality_service.sync_offline_data(test_user_id)
        
        assert offline_data is not None
        assert 'dashboard' in offline_data
        assert 'recent_alerts' in offline_data
        assert 'active_incidents' in offline_data
        
        # Get mobile summary
        summary = await quality_service.get_mobile_summary(test_user_id)
        
        assert summary is not None
        assert summary['user_id'] == test_user_id
        assert summary['alerts']['total'] > 0
        assert summary['incidents']['total'] > 0
        
    finally:
        # Clean up temp directory
        shutil.rmtree(storage_service.storage_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])