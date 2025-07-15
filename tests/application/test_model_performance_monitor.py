"""Tests for model performance monitor service."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
    ModelTask
)
from pynomaly.application.services.model_performance_monitor import ModelPerformanceMonitor
from pynomaly.application.services.performance_alert_service import PerformanceAlertService
from pynomaly.domain.services.performance_degradation_service import PerformanceDegradationService
from pynomaly.domain.value_objects.performance_degradation_metrics import (
    DegradationReport,
    DegradationSeverity,
    DegradationType,
    MetricThreshold,
    PerformanceBaseline,
    PerformanceDegradation,
)
from pynomaly.infrastructure.repositories.performance_degradation_repository import (
    PerformanceDegradationRepository,
)


class TestModelPerformanceMonitor:
    """Test suite for ModelPerformanceMonitor."""
    
    @pytest.fixture
    def mock_repositories_and_services(self):
        """Create mock repositories and services."""
        model_repo = MagicMock()
        performance_repo = MagicMock()
        degradation_service = MagicMock()
        alert_service = MagicMock()
        
        # Configure async methods
        model_repo.get_by_id = AsyncMock()
        performance_repo.store_performance_metrics = AsyncMock()
        performance_repo.get_latest_performance_metrics = AsyncMock()
        performance_repo.get_threshold_config = AsyncMock()
        performance_repo.store_degradation = AsyncMock()
        performance_repo.store_report = AsyncMock()
        performance_repo.get_baseline = AsyncMock()
        performance_repo.store_baseline = AsyncMock()
        performance_repo.update_baseline = AsyncMock()
        performance_repo.get_all_baselines = AsyncMock()
        performance_repo.get_model_health_summary = AsyncMock()
        performance_repo.get_recent_degradations = AsyncMock()
        performance_repo.get_system_wide_health = AsyncMock()
        performance_repo.cleanup_old_data = AsyncMock()
        performance_repo.get_model_performance_history = AsyncMock()
        
        degradation_service.detect_degradation = AsyncMock()
        degradation_service.monitor_continuous_degradation = AsyncMock()
        degradation_service.generate_degradation_report = AsyncMock()
        
        alert_service.process_degradations = AsyncMock()
        alert_service.get_active_alerts = AsyncMock()
        
        return model_repo, performance_repo, degradation_service, alert_service
    
    @pytest.fixture
    def monitor_config(self):
        """Create monitoring configuration."""
        return {
            'monitoring_interval_minutes': 15,
            'baseline_update_frequency_days': 7,
            'history_retention_days': 90,
            'auto_alert': True,
            'auto_baseline_update': True,
            'degradation_detection': {
                'lookback_days': 30,
                'min_samples': 10,
                'continuous_monitoring_hours': 24
            }
        }
    
    @pytest.fixture
    def performance_monitor(self, mock_repositories_and_services, monitor_config):
        """Create performance monitor with mocked dependencies."""
        model_repo, performance_repo, degradation_service, alert_service = mock_repositories_and_services
        return ModelPerformanceMonitor(
            model_repo, performance_repo, degradation_service, alert_service, monitor_config
        )
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample performance metrics."""
        return ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            roc_auc=0.92
        )
    
    @pytest.fixture
    def sample_degradation(self):
        """Create sample degradation."""
        return PerformanceDegradation(
            degradation_type=DegradationType.ACCURACY_DROP,
            severity=DegradationSeverity.HIGH,
            metric_name="accuracy",
            current_value=0.75,
            baseline_value=0.85,
            degradation_amount=-0.10,
            degradation_percentage=-11.76,
            threshold_violated="warning",
            confidence_level=0.85,
            detection_method="baseline_comparison",
            samples_used=25
        )
    
    def test_monitor_initialization(self, performance_monitor):
        """Test monitor initialization."""
        assert performance_monitor.config['monitoring_interval_minutes'] == 15
        assert performance_monitor.config['auto_alert'] is True
        assert performance_monitor._monitoring_active is False
        assert len(performance_monitor._monitored_models) == 0
    
    def test_default_config(self, mock_repositories_and_services):
        """Test default configuration creation."""
        model_repo, performance_repo, degradation_service, alert_service = mock_repositories_and_services
        monitor = ModelPerformanceMonitor(
            model_repo, performance_repo, degradation_service, alert_service
        )
        
        config = monitor.config
        assert 'monitoring_interval_minutes' in config
        assert 'baseline_update_frequency_days' in config
        assert 'auto_alert' in config
        assert 'degradation_detection' in config
    
    @pytest.mark.asyncio
    async def test_add_model_monitoring_success(self, performance_monitor):
        """Test successfully adding a model to monitoring."""
        model_id = uuid4()
        
        # Mock successful model lookup
        performance_monitor.model_repository.get_by_id.return_value = MagicMock()
        
        custom_thresholds = {
            "accuracy": MetricThreshold(
                metric_name="accuracy",
                warning_threshold=5.0,
                critical_threshold=10.0,
                threshold_type="percentage",
                direction="decrease"
            )
        }
        
        result = await performance_monitor.add_model_monitoring(
            model_id=model_id,
            custom_thresholds=custom_thresholds
        )
        
        assert result is True
        assert model_id in performance_monitor._monitored_models
        
        # Verify repository calls
        performance_monitor.model_repository.get_by_id.assert_called_once_with(model_id)
        performance_monitor.performance_repository.store_threshold_config.assert_called_once_with(
            model_id, custom_thresholds
        )
    
    @pytest.mark.asyncio
    async def test_add_model_monitoring_model_not_found(self, performance_monitor):
        """Test adding monitoring for non-existent model."""
        model_id = uuid4()
        
        # Mock model not found
        performance_monitor.model_repository.get_by_id.return_value = None
        
        result = await performance_monitor.add_model_monitoring(model_id)
        
        assert result is False
        assert model_id not in performance_monitor._monitored_models
    
    @pytest.mark.asyncio
    async def test_remove_model_monitoring(self, performance_monitor):
        """Test removing a model from monitoring."""
        model_id = uuid4()
        
        # Add model first
        performance_monitor._monitored_models[model_id] = {
            'config': {},
            'added_at': datetime.utcnow()
        }
        
        result = await performance_monitor.remove_model_monitoring(model_id)
        
        assert result is True
        assert model_id not in performance_monitor._monitored_models
    
    @pytest.mark.asyncio
    async def test_record_performance_metrics(self, performance_monitor, sample_metrics):
        """Test recording performance metrics."""
        model_id = uuid4()
        
        # Add model to monitoring
        performance_monitor._monitored_models[model_id] = {
            'config': {},
            'added_at': datetime.utcnow(),
            'last_check': None
        }
        
        # Mock degradation detection
        mock_report = MagicMock()
        performance_monitor.check_model_degradation = AsyncMock(return_value=mock_report)
        
        result = await performance_monitor.record_performance_metrics(
            model_id=model_id,
            metrics=sample_metrics,
            auto_check_degradation=True
        )
        
        assert result == mock_report
        
        # Verify repository call
        performance_monitor.performance_repository.store_performance_metrics.assert_called_once()
        
        # Verify last check was updated
        assert performance_monitor._monitored_models[model_id]['last_check'] is not None
    
    @pytest.mark.asyncio
    async def test_check_model_degradation_with_metrics(self, performance_monitor, sample_metrics, sample_degradation):
        """Test checking model degradation with provided metrics."""
        model_id = uuid4()
        
        # Mock degradation detection
        performance_monitor.degradation_service.detect_degradation.return_value = [sample_degradation]
        performance_monitor.degradation_service.monitor_continuous_degradation.return_value = []
        
        # Mock report generation
        mock_report = DegradationReport(
            report_id="report_123",
            model_id=str(model_id),
            time_period_start=datetime.utcnow() - timedelta(days=30),
            time_period_end=datetime.utcnow(),
            degradations=[sample_degradation],
            overall_health_score=0.7
        )
        performance_monitor.degradation_service.generate_degradation_report.return_value = mock_report
        
        # Mock alert processing
        performance_monitor.alert_service.process_degradations.return_value = []
        
        result = await performance_monitor.check_model_degradation(
            model_id=model_id,
            current_metrics=sample_metrics
        )
        
        assert result == mock_report
        
        # Verify degradation detection calls
        performance_monitor.degradation_service.detect_degradation.assert_called_once()
        performance_monitor.degradation_service.monitor_continuous_degradation.assert_called_once()
        
        # Verify degradation storage
        performance_monitor.performance_repository.store_degradation.assert_called_once_with(
            model_id, sample_degradation
        )
        
        # Verify report storage
        performance_monitor.performance_repository.store_report.assert_called_once_with(mock_report)
    
    @pytest.mark.asyncio
    async def test_check_model_degradation_no_metrics(self, performance_monitor, sample_metrics):
        """Test checking degradation when no current metrics provided."""
        model_id = uuid4()
        
        # Mock latest metrics retrieval
        performance_monitor.performance_repository.get_latest_performance_metrics.return_value = sample_metrics
        
        # Mock no degradations found
        performance_monitor.degradation_service.detect_degradation.return_value = []
        performance_monitor.degradation_service.monitor_continuous_degradation.return_value = []
        
        result = await performance_monitor.check_model_degradation(model_id)
        
        # Should return None when no degradations
        assert result is None
        
        # Verify latest metrics was called
        performance_monitor.performance_repository.get_latest_performance_metrics.assert_called_once_with(model_id)
    
    @pytest.mark.asyncio
    async def test_update_model_baselines(self, performance_monitor):
        """Test updating model baselines."""
        model_id = uuid4()
        
        # Mock performance history
        performance_history = [
            ModelPerformanceMetrics(
                task_type=ModelTask.BINARY_CLASSIFICATION,
                sample_size=1000,
                accuracy=0.85 + i * 0.01,
                precision=0.82
            ) for i in range(20)
        ]
        
        performance_monitor.performance_repository.get_model_performance_history.return_value = performance_history
        performance_monitor.performance_repository.get_baseline.return_value = None  # No existing baseline
        
        results = await performance_monitor.update_model_baselines(
            model_id=model_id,
            force_update=True
        )
        
        # Should update baselines for metrics with enough data
        assert 'accuracy' in results
        assert 'precision' in results
        assert results['accuracy'] is True  # Should be updated
        assert results['precision'] is True  # Should be updated
        
        # Verify baseline storage calls
        assert performance_monitor.performance_repository.store_baseline.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_get_model_health_status(self, performance_monitor):
        """Test getting model health status."""
        model_id = uuid4()
        
        # Add model to monitoring
        performance_monitor._monitored_models[model_id] = {
            'config': {},
            'added_at': datetime.utcnow() - timedelta(hours=1),
            'last_check': datetime.utcnow() - timedelta(minutes=5)
        }
        
        # Mock repository responses
        performance_monitor.performance_repository.get_model_health_summary.return_value = {
            'model_id': str(model_id),
            'health_score': 0.85,
            'recent_degradations_count': 2,
            'baselines_count': 5
        }
        
        performance_monitor.performance_repository.get_recent_degradations.return_value = []
        performance_monitor.alert_service.get_active_alerts.return_value = []
        performance_monitor.performance_repository.get_latest_performance_metrics.return_value = MagicMock()
        performance_monitor.performance_repository.get_all_baselines.return_value = {}
        
        health_status = await performance_monitor.get_model_health_status(model_id)
        
        assert health_status['model_id'] == str(model_id)
        assert health_status['monitoring_active'] is True
        assert health_status['health_score'] == 0.85
        assert 'monitoring_since' in health_status
        assert 'last_check' in health_status
        assert 'status_summary' in health_status
    
    @pytest.mark.asyncio
    async def test_get_monitoring_dashboard(self, performance_monitor):
        """Test getting monitoring dashboard."""
        model_id_1 = uuid4()
        model_id_2 = uuid4()
        
        # Add models to monitoring
        performance_monitor._monitored_models[model_id_1] = {
            'config': {},
            'added_at': datetime.utcnow() - timedelta(hours=2)
        }
        performance_monitor._monitored_models[model_id_2] = {
            'config': {},
            'added_at': datetime.utcnow() - timedelta(hours=1)
        }
        
        # Mock system health
        performance_monitor.performance_repository.get_system_wide_health.return_value = {
            'total_models_monitored': 2,
            'system_health_score': 0.9,
            'recent_degradations_24h': 1
        }
        
        # Mock model health status
        async def mock_get_health_status(model_id):
            return {
                'model_id': str(model_id),
                'health_score': 0.8,
                'recent_degradations': 0,
                'active_alerts': 0,
                'status_summary': 'healthy'
            }
        
        performance_monitor.get_model_health_status = AsyncMock(side_effect=mock_get_health_status)
        performance_monitor.alert_service.get_active_alerts.return_value = []
        
        dashboard = await performance_monitor.get_monitoring_dashboard()
        
        assert dashboard['monitored_models_count'] == 2
        assert len(dashboard['monitored_models']) == 2
        assert 'system_health' in dashboard
        assert 'total_active_alerts' in dashboard
        assert 'monitoring_status' in dashboard
        assert dashboard['monitoring_status']['active'] is False  # Not started
    
    @pytest.mark.asyncio
    async def test_start_continuous_monitoring(self, performance_monitor):
        """Test starting continuous monitoring."""
        assert performance_monitor._monitoring_active is False
        
        await performance_monitor.start_continuous_monitoring()
        
        assert performance_monitor._monitoring_active is True
    
    @pytest.mark.asyncio
    async def test_stop_continuous_monitoring(self, performance_monitor):
        """Test stopping continuous monitoring."""
        performance_monitor._monitoring_active = True
        
        await performance_monitor.stop_continuous_monitoring()
        
        assert performance_monitor._monitoring_active is False
    
    def test_create_baseline_from_values(self, performance_monitor):
        """Test creating baseline from values."""
        metric_name = "accuracy"
        values = [0.85, 0.87, 0.84, 0.86, 0.88, 0.85, 0.89, 0.84, 0.87, 0.86]
        
        baseline = performance_monitor._create_baseline_from_values(metric_name, values)
        
        assert baseline.metric_name == metric_name
        assert baseline.sample_count == len(values)
        assert 0.84 <= baseline.baseline_value <= 0.89
        assert baseline.standard_deviation > 0
        assert baseline.min_value is not None
        assert baseline.max_value is not None
        assert baseline.median_value is not None
    
    def test_get_status_summary_healthy(self, performance_monitor):
        """Test status summary for healthy model."""
        health_summary = {'health_score': 0.95}
        recent_degradations = []
        active_alerts = []
        
        status = performance_monitor._get_status_summary(
            health_summary, recent_degradations, active_alerts
        )
        
        assert status == 'healthy'
    
    def test_get_status_summary_critical(self, performance_monitor):
        """Test status summary for critical model."""
        health_summary = {'health_score': 0.5}
        recent_degradations = [
            MagicMock(severity=DegradationSeverity.CRITICAL)
        ]
        active_alerts = []
        
        status = performance_monitor._get_status_summary(
            health_summary, recent_degradations, active_alerts
        )
        
        assert status == 'critical'
    
    def test_get_status_summary_degraded(self, performance_monitor):
        """Test status summary for degraded model."""
        health_summary = {'health_score': 0.8}
        recent_degradations = [
            MagicMock(severity=DegradationSeverity.HIGH)
        ]
        active_alerts = []
        
        status = performance_monitor._get_status_summary(
            health_summary, recent_degradations, active_alerts
        )
        
        assert status == 'degraded'
    
    def test_get_status_summary_warning(self, performance_monitor):
        """Test status summary for model with warnings."""
        health_summary = {'health_score': 0.85}
        recent_degradations = []
        active_alerts = [MagicMock()]  # Has active alerts
        
        status = performance_monitor._get_status_summary(
            health_summary, recent_degradations, active_alerts
        )
        
        assert status == 'warning'
    
    def test_count_alerts_by_severity(self, performance_monitor):
        """Test counting alerts by severity."""
        alerts = [
            MagicMock(alert_level=DegradationSeverity.CRITICAL),
            MagicMock(alert_level=DegradationSeverity.HIGH),
            MagicMock(alert_level=DegradationSeverity.HIGH),
            MagicMock(alert_level=DegradationSeverity.LOW)
        ]
        
        counts = performance_monitor._count_alerts_by_severity(alerts)
        
        assert counts['critical'] == 1
        assert counts['high'] == 2
        assert counts['medium'] == 0
        assert counts['low'] == 1
    
    @pytest.mark.asyncio
    async def test_handle_degradation_alerts(self, performance_monitor, sample_degradation):
        """Test handling degradation alerts."""
        model_id = uuid4()
        degradations = [sample_degradation]
        
        # Add model to monitoring
        performance_monitor._monitored_models[model_id] = {
            'config': {},
            'added_at': datetime.utcnow(),
            'alert_count': 0
        }
        
        # Mock alert processing
        mock_alerts = [MagicMock()]
        performance_monitor.alert_service.process_degradations.return_value = mock_alerts
        
        await performance_monitor._handle_degradation_alerts(model_id, degradations)
        
        # Verify alert service was called
        performance_monitor.alert_service.process_degradations.assert_called_once_with(
            model_id=model_id,
            degradations=degradations,
            send_alerts=True
        )
        
        # Verify alert count was updated
        assert performance_monitor._monitored_models[model_id]['alert_count'] == 1
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, performance_monitor):
        """Test cleanup of old data."""
        # Mock cleanup summary
        cleanup_summary = {
            'degradations_removed': 10,
            'alerts_removed': 5,
            'reports_removed': 2
        }
        performance_monitor.performance_repository.cleanup_old_data.return_value = cleanup_summary
        
        await performance_monitor._cleanup_old_data()
        
        # Verify cleanup was called
        performance_monitor.performance_repository.cleanup_old_data.assert_called_once_with(90)  # retention_days