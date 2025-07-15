"""Tests for quality monitoring service."""

import pytest
import pandas as pd
import time
from uuid import uuid4
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from ..application.services.quality_monitoring_service import (
    QualityMonitoringService, MonitoringConfiguration, AlertSeverity,
    MonitoringStatus, MetricType, QualityAlert, QualityMetric
)
from ..domain.entities.quality_rule import (
    QualityRule, RuleType, LogicType, ValidationLogic, QualityThreshold,
    Severity, UserId, DatasetId, RuleId
)


@pytest.fixture
def monitoring_config():
    """Create monitoring configuration."""
    return MonitoringConfiguration(
        monitoring_interval_seconds=60,
        alert_thresholds={'critical': 0.8, 'warning': 0.9},
        notification_channels=['email', 'slack'],
        retention_days=7,
        max_alerts_per_hour=5,
        enable_trend_analysis=True,
        enable_anomaly_detection=True
    )


@pytest.fixture
def monitoring_service(monitoring_config):
    """Create monitoring service instance."""
    return QualityMonitoringService(monitoring_config)


@pytest.fixture
def sample_dataset():
    """Create sample dataset for monitoring."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'email': ['john@test.com', 'jane@test.com', 'invalid', 'alice@test.com', 'charlie@test.com'],
        'age': [25, 30, 35, 40, 45],
        'score': [85, 92, 78, 95, 88]
    })


@pytest.fixture
def quality_rule():
    """Create quality rule for testing."""
    return QualityRule(
        rule_id=RuleId(value=uuid4()),
        rule_name="Email Validation",
        rule_type=RuleType.VALIDITY,
        target_columns=['email'],
        validation_logic=ValidationLogic(
            logic_type=LogicType.REGEX,
            expression=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            parameters={},
            error_message_template="Invalid email format"
        ),
        thresholds=QualityThreshold(
            pass_rate_threshold=0.9,
            warning_threshold=0.8,
            critical_threshold=0.7
        ),
        severity=Severity.HIGH,
        created_by=UserId(value=uuid4()),
        is_enabled=True
    )


class TestQualityMonitoringService:
    """Test cases for QualityMonitoringService."""
    
    def test_service_initialization(self, monitoring_service, monitoring_config):
        """Test service initialization."""
        assert monitoring_service.config == monitoring_config
        assert monitoring_service.status == MonitoringStatus.STOPPED
        assert len(monitoring_service.monitored_datasets) == 0
        assert len(monitoring_service.active_rules) == 0
    
    def test_add_dataset_monitoring(self, monitoring_service, quality_rule):
        """Test adding dataset to monitoring."""
        dataset_id = uuid4()
        data_source_config = {'type': 'database', 'table': 'users'}
        
        monitoring_service.add_dataset_monitoring(
            dataset_id=dataset_id,
            rules=[quality_rule],
            data_source_config=data_source_config
        )
        
        assert dataset_id in monitoring_service.monitored_datasets
        assert quality_rule.rule_id in monitoring_service.active_rules
        
        dataset_info = monitoring_service.monitored_datasets[dataset_id]
        assert dataset_info['rules'] == [quality_rule.rule_id]
        assert dataset_info['data_source_config'] == data_source_config
        assert dataset_info['last_check'] is None
        assert dataset_info['check_count'] == 0
    
    def test_remove_dataset_monitoring(self, monitoring_service, quality_rule):
        """Test removing dataset from monitoring."""
        dataset_id = uuid4()
        
        # Add dataset first
        monitoring_service.add_dataset_monitoring(
            dataset_id=dataset_id,
            rules=[quality_rule],
            data_source_config={'type': 'file'}
        )
        
        # Verify it was added
        assert dataset_id in monitoring_service.monitored_datasets
        assert quality_rule.rule_id in monitoring_service.active_rules
        
        # Remove dataset
        monitoring_service.remove_dataset_monitoring(dataset_id)
        
        # Verify it was removed
        assert dataset_id not in monitoring_service.monitored_datasets
        assert quality_rule.rule_id not in monitoring_service.active_rules
    
    def test_start_stop_monitoring(self, monitoring_service):
        """Test starting and stopping monitoring service."""
        # Initially stopped
        assert monitoring_service.status == MonitoringStatus.STOPPED
        
        # Start monitoring
        monitoring_service.start_monitoring()
        assert monitoring_service.status == MonitoringStatus.ACTIVE
        assert monitoring_service.monitoring_thread is not None
        
        # Stop monitoring
        monitoring_service.stop_monitoring()
        assert monitoring_service.status == MonitoringStatus.STOPPED
    
    def test_pause_resume_monitoring(self, monitoring_service):
        """Test pausing and resuming monitoring."""
        # Start monitoring first
        monitoring_service.start_monitoring()
        assert monitoring_service.status == MonitoringStatus.ACTIVE
        
        # Pause monitoring
        monitoring_service.pause_monitoring()
        assert monitoring_service.status == MonitoringStatus.PAUSED
        
        # Resume monitoring
        monitoring_service.resume_monitoring()
        assert monitoring_service.status == MonitoringStatus.ACTIVE
        
        # Clean up
        monitoring_service.stop_monitoring()
    
    def test_validate_dataset_realtime(self, monitoring_service, quality_rule, sample_dataset):
        """Test real-time dataset validation."""
        dataset_id = uuid4()
        
        # Add dataset to monitoring
        monitoring_service.add_dataset_monitoring(
            dataset_id=dataset_id,
            rules=[quality_rule],
            data_source_config={'type': 'dataframe'}
        )
        
        # Validate dataset
        result = monitoring_service.validate_dataset_realtime(dataset_id, sample_dataset)
        
        assert result['success'] is True
        assert result['dataset_id'] == str(dataset_id)
        assert result['validation_results'] > 0
        assert result['metrics_generated'] > 0
        assert 'execution_time_seconds' in result
        assert 'overall_quality_score' in result
        
        # Check that metrics were added
        assert len(monitoring_service.metrics_buffer) > 0
    
    def test_validate_dataset_not_monitored(self, monitoring_service, sample_dataset):
        """Test validation of dataset that is not monitored."""
        dataset_id = uuid4()
        
        result = monitoring_service.validate_dataset_realtime(dataset_id, sample_dataset)
        
        assert result['success'] is False
        assert 'error' in result
        assert result['error'] == "Dataset not monitored"
    
    def test_get_quality_dashboard_data(self, monitoring_service, quality_rule):
        """Test dashboard data retrieval."""
        dataset_id = uuid4()
        
        # Add dataset to monitoring
        monitoring_service.add_dataset_monitoring(
            dataset_id=dataset_id,
            rules=[quality_rule],
            data_source_config={'type': 'file'}
        )
        
        # Get dashboard data
        dashboard_data = monitoring_service.get_quality_dashboard_data()
        
        assert 'monitoring_status' in dashboard_data
        assert dashboard_data['monitoring_status'] == MonitoringStatus.STOPPED.value
        assert dashboard_data['monitored_datasets'] == 1
        assert dashboard_data['total_active_rules'] == 1
        assert 'quality_summary' in dashboard_data
        assert 'alert_summary' in dashboard_data
        assert 'last_updated' in dashboard_data
    
    def test_get_quality_dashboard_data_with_dataset_filter(self, monitoring_service, quality_rule):
        """Test dashboard data retrieval with dataset filter."""
        dataset_id = uuid4()
        
        # Add dataset to monitoring
        monitoring_service.add_dataset_monitoring(
            dataset_id=dataset_id,
            rules=[quality_rule],
            data_source_config={'type': 'file'}
        )
        
        # Get dashboard data for specific dataset
        dashboard_data = monitoring_service.get_quality_dashboard_data(dataset_id)
        
        assert 'datasets_info' in dashboard_data
        assert str(dataset_id) in dashboard_data['datasets_info']
        assert 'trends' in dashboard_data
    
    def test_get_quality_report(self, monitoring_service, quality_rule, sample_dataset):
        """Test quality report generation."""
        dataset_id = uuid4()
        
        # Add dataset and validate to generate metrics
        monitoring_service.add_dataset_monitoring(
            dataset_id=dataset_id,
            rules=[quality_rule],
            data_source_config={'type': 'dataframe'}
        )
        
        # Validate to generate metrics
        monitoring_service.validate_dataset_realtime(dataset_id, sample_dataset)
        
        # Generate report
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow() + timedelta(hours=1)
        
        report = monitoring_service.get_quality_report(dataset_id, start_date, end_date)
        
        assert 'dataset_id' in report
        assert report['dataset_id'] == str(dataset_id)
        assert 'report_period' in report
        assert 'metric_statistics' in report
        assert 'alert_statistics' in report
        assert 'data_points' in report
        assert 'generated_at' in report


class TestQualityTrendAnalyzer:
    """Test cases for QualityTrendAnalyzer."""
    
    def test_trend_analyzer_initialization(self, monitoring_service):
        """Test trend analyzer initialization."""
        analyzer = monitoring_service.trend_analyzer
        assert analyzer.window_size == 24
        assert len(analyzer.metrics_buffer) == 0
    
    def test_add_metric_and_detect_trend(self, monitoring_service):
        """Test adding metrics and trend detection."""
        analyzer = monitoring_service.trend_analyzer
        dataset_id = uuid4()
        
        # Add metrics over time to simulate trend
        base_time = datetime.utcnow()
        for i in range(10):
            metric = QualityMetric(
                metric_type=MetricType.PASS_RATE,
                dataset_id=dataset_id,
                value=0.9 + (i * 0.01),  # Improving trend
                timestamp=base_time + timedelta(hours=i)
            )
            analyzer.add_metric(metric)
        
        # Detect trend
        trend = analyzer.detect_trend(dataset_id, MetricType.PASS_RATE)
        
        assert 'trend' in trend
        assert trend['trend'] in ['improving', 'declining', 'stable']
        assert 'confidence' in trend
        assert 0 <= trend['confidence'] <= 1
        assert 'data_points' in trend
        assert trend['data_points'] == 10
    
    def test_detect_trend_insufficient_data(self, monitoring_service):
        """Test trend detection with insufficient data."""
        analyzer = monitoring_service.trend_analyzer
        dataset_id = uuid4()
        
        # Add only one metric
        metric = QualityMetric(
            metric_type=MetricType.PASS_RATE,
            dataset_id=dataset_id,
            value=0.95
        )
        analyzer.add_metric(metric)
        
        # Try to detect trend
        trend = analyzer.detect_trend(dataset_id, MetricType.PASS_RATE)
        
        assert trend['trend'] == 'insufficient_data'
        assert trend['confidence'] == 0.0
    
    def test_detect_anomaly(self, monitoring_service):
        """Test anomaly detection."""
        analyzer = monitoring_service.trend_analyzer
        dataset_id = uuid4()
        
        # Add normal metrics
        base_time = datetime.utcnow()
        for i in range(10):
            metric = QualityMetric(
                metric_type=MetricType.PASS_RATE,
                dataset_id=dataset_id,
                value=0.95,  # Consistent value
                timestamp=base_time + timedelta(hours=i)
            )
            analyzer.add_metric(metric)
        
        # Test normal value (should not be anomaly)
        anomaly_result = analyzer.detect_anomaly(dataset_id, MetricType.PASS_RATE, 0.94)
        assert anomaly_result['is_anomaly'] is False
        
        # Test anomalous value
        anomaly_result = analyzer.detect_anomaly(dataset_id, MetricType.PASS_RATE, 0.5)
        assert anomaly_result['is_anomaly'] is True
        assert anomaly_result['confidence'] > 0
        assert 'z_score' in anomaly_result
    
    def test_detect_anomaly_insufficient_data(self, monitoring_service):
        """Test anomaly detection with insufficient data."""
        analyzer = monitoring_service.trend_analyzer
        dataset_id = uuid4()
        
        # Add only a few metrics
        for i in range(3):
            metric = QualityMetric(
                metric_type=MetricType.PASS_RATE,
                dataset_id=dataset_id,
                value=0.95
            )
            analyzer.add_metric(metric)
        
        # Try to detect anomaly
        anomaly_result = analyzer.detect_anomaly(dataset_id, MetricType.PASS_RATE, 0.5)
        
        assert anomaly_result['is_anomaly'] is False
        assert anomaly_result['confidence'] == 0.0
        assert anomaly_result['reason'] == 'insufficient_data'


class TestAlertManager:
    """Test cases for AlertManager."""
    
    def test_alert_manager_initialization(self, monitoring_service):
        """Test alert manager initialization."""
        alert_manager = monitoring_service.alert_manager
        assert len(alert_manager.active_alerts) == 0
        assert len(alert_manager.alert_history) == 0
        assert len(alert_manager.notification_handlers) == 0
    
    def test_create_alert(self, monitoring_service):
        """Test alert creation."""
        alert_manager = monitoring_service.alert_manager
        rule_id = uuid4()
        dataset_id = uuid4()
        
        alert = alert_manager.create_alert(
            rule_id=rule_id,
            dataset_id=dataset_id,
            severity=AlertSeverity.WARNING,
            alert_type='quality_threshold',
            message='Quality below threshold',
            current_value=0.85,
            threshold_value=0.9
        )
        
        assert alert is not None
        assert alert.rule_id == rule_id
        assert alert.dataset_id == dataset_id
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == 'Quality below threshold'
        assert alert.current_value == 0.85
        assert alert.threshold_value == 0.9
        assert not alert.acknowledged
        
        # Check that alert was stored
        assert alert.alert_id in alert_manager.active_alerts
        assert len(alert_manager.alert_history) == 1
    
    def test_acknowledge_alert(self, monitoring_service):
        """Test alert acknowledgment."""
        alert_manager = monitoring_service.alert_manager
        
        # Create alert
        alert = alert_manager.create_alert(
            rule_id=uuid4(),
            dataset_id=uuid4(),
            severity=AlertSeverity.CRITICAL,
            alert_type='quality_failure',
            message='Critical quality issue',
            current_value=0.5,
            threshold_value=0.8
        )
        
        # Acknowledge alert
        success = alert_manager.acknowledge_alert(alert.alert_id, 'test_user')
        
        assert success is True
        assert alert.acknowledged is True
        assert alert.acknowledged_by == 'test_user'
        assert alert.acknowledged_at is not None
    
    def test_acknowledge_nonexistent_alert(self, monitoring_service):
        """Test acknowledging non-existent alert."""
        alert_manager = monitoring_service.alert_manager
        
        success = alert_manager.acknowledge_alert(uuid4(), 'test_user')
        assert success is False
    
    def test_close_alert(self, monitoring_service):
        """Test closing an alert."""
        alert_manager = monitoring_service.alert_manager
        
        # Create alert
        alert = alert_manager.create_alert(
            rule_id=uuid4(),
            dataset_id=uuid4(),
            severity=AlertSeverity.INFO,
            alert_type='info_alert',
            message='Information alert',
            current_value=0.95,
            threshold_value=0.9
        )
        
        # Close alert
        success = alert_manager.close_alert(alert.alert_id)
        
        assert success is True
        assert alert.alert_id not in alert_manager.active_alerts
    
    def test_get_active_alerts(self, monitoring_service):
        """Test getting active alerts."""
        alert_manager = monitoring_service.alert_manager
        
        # Create alerts with different severities
        alert1 = alert_manager.create_alert(
            rule_id=uuid4(),
            dataset_id=uuid4(),
            severity=AlertSeverity.WARNING,
            alert_type='warning_alert',
            message='Warning alert',
            current_value=0.85,
            threshold_value=0.9
        )
        
        alert2 = alert_manager.create_alert(
            rule_id=uuid4(),
            dataset_id=uuid4(),
            severity=AlertSeverity.CRITICAL,
            alert_type='critical_alert',
            message='Critical alert',
            current_value=0.7,
            threshold_value=0.9
        )
        
        # Get all active alerts
        all_alerts = alert_manager.get_active_alerts()
        assert len(all_alerts) == 2
        
        # Get alerts by severity
        warning_alerts = alert_manager.get_active_alerts(AlertSeverity.WARNING)
        assert len(warning_alerts) == 1
        assert warning_alerts[0].severity == AlertSeverity.WARNING
        
        critical_alerts = alert_manager.get_active_alerts(AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL
    
    def test_alert_rate_limiting(self, monitoring_service):
        """Test alert rate limiting."""
        alert_manager = monitoring_service.alert_manager
        rule_id = uuid4()
        dataset_id = uuid4()
        
        # Create more alerts than the rate limit allows
        alerts_created = 0
        for i in range(15):  # Trying to create more than max_alerts_per_hour (10)
            alert = alert_manager.create_alert(
                rule_id=rule_id,
                dataset_id=dataset_id,
                severity=AlertSeverity.INFO,
                alert_type='rate_limit_test',
                message=f'Alert {i}',
                current_value=0.8,
                threshold_value=0.9
            )
            if alert is not None:
                alerts_created += 1
        
        # Should be limited to max_alerts_per_hour
        assert alerts_created <= alert_manager.config.max_alerts_per_hour
    
    def test_get_alert_summary(self, monitoring_service):
        """Test alert summary generation."""
        alert_manager = monitoring_service.alert_manager
        
        # Create alerts with different severities
        alert_manager.create_alert(
            rule_id=uuid4(),
            dataset_id=uuid4(),
            severity=AlertSeverity.WARNING,
            alert_type='test',
            message='Warning',
            current_value=0.85,
            threshold_value=0.9
        )
        
        alert_manager.create_alert(
            rule_id=uuid4(),
            dataset_id=uuid4(),
            severity=AlertSeverity.CRITICAL,
            alert_type='test',
            message='Critical',
            current_value=0.7,
            threshold_value=0.9
        )
        
        summary = alert_manager.get_alert_summary()
        
        assert 'total_active_alerts' in summary
        assert summary['total_active_alerts'] == 2
        assert 'by_severity' in summary
        assert summary['by_severity']['warning'] == 1
        assert summary['by_severity']['critical'] == 1
        assert 'acknowledged' in summary
        assert 'unacknowledged' in summary
        assert summary['unacknowledged'] == 2  # None acknowledged yet
    
    def test_notification_handler_registration(self, monitoring_service):
        """Test notification handler registration."""
        alert_manager = monitoring_service.alert_manager
        
        # Mock notification handler
        mock_handler = MagicMock()
        
        # Register handler
        alert_manager.register_notification_handler('email', mock_handler)
        
        assert 'email' in alert_manager.notification_handlers
        assert alert_manager.notification_handlers['email'] == mock_handler
    
    @patch('logging.getLogger')
    def test_notification_sending(self, mock_logger, monitoring_service):
        """Test notification sending when alert is created."""
        alert_manager = monitoring_service.alert_manager
        
        # Mock notification handler
        mock_handler = MagicMock()
        alert_manager.register_notification_handler('email', mock_handler)
        
        # Update config to include email channel
        alert_manager.config.notification_channels = ['email']
        
        # Create alert (should trigger notification)
        alert = alert_manager.create_alert(
            rule_id=uuid4(),
            dataset_id=uuid4(),
            severity=AlertSeverity.EMERGENCY,
            alert_type='emergency_alert',
            message='Emergency situation',
            current_value=0.1,
            threshold_value=0.9
        )
        
        # Verify notification was sent
        mock_handler.assert_called_once_with(alert)