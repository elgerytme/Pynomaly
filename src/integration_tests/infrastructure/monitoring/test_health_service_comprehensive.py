"""Comprehensive tests for health service monitoring infrastructure."""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from pynomaly.infrastructure.monitoring.health_service import (
    HealthCheck,
    HealthService,
    HealthStatus,
    SystemMetrics,
)


class TestHealthStatus:
    """Test health status enumeration."""

    def test_health_status_values(self):
        """Test health status enumeration values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheck:
    """Test health check data structure."""

    def test_health_check_creation(self):
        """Test health check creation."""
        check = HealthCheck(
            name="Database Connection",
            status=HealthStatus.HEALTHY,
            message="Database is responsive",
            duration_ms=125.5,
            details={"response_time": 125.5, "connection_pool": "healthy"},
        )

        assert check.name == "Database Connection"
        assert check.status == HealthStatus.HEALTHY
        assert check.message == "Database is responsive"
        assert check.duration_ms == 125.5
        assert check.details["response_time"] == 125.5
        assert check.details["connection_pool"] == "healthy"
        assert isinstance(check.timestamp, datetime)

    def test_health_check_defaults(self):
        """Test health check with default values."""
        check = HealthCheck(
            name="Simple Check",
            status=HealthStatus.HEALTHY,
            message="OK",
            duration_ms=50.0,
        )

        assert check.details == {}
        assert isinstance(check.timestamp, datetime)


class TestSystemMetrics:
    """Test system metrics data structure."""

    def test_system_metrics_creation(self):
        """Test system metrics creation."""
        metrics = SystemMetrics(
            cpu_percent=25.5,
            memory_percent=65.2,
            disk_percent=45.8,
            memory_available_mb=2048.0,
            disk_available_gb=500.0,
            load_average=[0.5, 0.8, 1.2],
            network_io={"bytes_sent": 1024, "bytes_recv": 2048},
            process_count=150,
            uptime_seconds=3600.0,
        )

        assert metrics.cpu_percent == 25.5
        assert metrics.memory_percent == 65.2
        assert metrics.disk_percent == 45.8
        assert metrics.memory_available_mb == 2048.0
        assert metrics.disk_available_gb == 500.0
        assert metrics.load_average == [0.5, 0.8, 1.2]
        assert metrics.network_io == {"bytes_sent": 1024, "bytes_recv": 2048}
        assert metrics.process_count == 150
        assert metrics.uptime_seconds == 3600.0


class TestHealthService:
    """Test health service functionality."""

    def test_health_service_initialization(self):
        """Test health service initialization."""
        service = HealthService(max_history=50)

        assert service.max_history == 50
        assert service._check_history == []
        assert isinstance(service._start_time, float)

    def test_health_service_default_initialization(self):
        """Test health service with default parameters."""
        service = HealthService()

        assert service.max_history == 100
        assert service._check_history == []

    @patch("pynomaly.infrastructure.monitoring.health_service.psutil")
    def test_check_system_resources_healthy(self, mock_psutil):
        """Test system resources check with healthy status."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=50.0,
            available=2048 * 1024 * 1024,
            total=4096 * 1024 * 1024,
        )
        mock_psutil.disk_usage.return_value = Mock(
            used=1024 * 1024 * 1024,
            total=2048 * 1024 * 1024,
            free=1024 * 1024 * 1024,
        )

        service = HealthService()

        checks = asyncio.run(service._check_system_resources())

        assert "cpu" in checks
        assert "memory" in checks
        assert "disk" in checks

        # Check CPU
        cpu_check = checks["cpu"]
        assert cpu_check.status == HealthStatus.HEALTHY
        assert "CPU usage: 30.0%" in cpu_check.message
        assert cpu_check.details["cpu_percent"] == 30.0

        # Check memory
        memory_check = checks["memory"]
        assert memory_check.status == HealthStatus.HEALTHY
        assert "Memory usage: 50.0%" in memory_check.message
        assert memory_check.details["memory_percent"] == 50.0

        # Check disk
        disk_check = checks["disk"]
        assert disk_check.status == HealthStatus.HEALTHY
        assert "Disk usage: 50.0%" in disk_check.message
        assert disk_check.details["disk_percent"] == 50.0

    @patch("pynomaly.infrastructure.monitoring.health_service.psutil")
    def test_check_system_resources_degraded(self, mock_psutil):
        """Test system resources check with degraded status."""
        # Mock psutil responses for degraded state
        mock_psutil.cpu_percent.return_value = 80.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=85.0,
            available=512 * 1024 * 1024,
            total=4096 * 1024 * 1024,
        )
        mock_psutil.disk_usage.return_value = Mock(
            used=1800 * 1024 * 1024,
            total=2048 * 1024 * 1024,
            free=248 * 1024 * 1024,
        )

        service = HealthService()

        checks = asyncio.run(service._check_system_resources())

        # Check CPU
        cpu_check = checks["cpu"]
        assert cpu_check.status == HealthStatus.DEGRADED

        # Check memory
        memory_check = checks["memory"]
        assert memory_check.status == HealthStatus.DEGRADED

        # Check disk
        disk_check = checks["disk"]
        assert disk_check.status == HealthStatus.DEGRADED

    @patch("pynomaly.infrastructure.monitoring.health_service.psutil")
    def test_check_system_resources_unhealthy(self, mock_psutil):
        """Test system resources check with unhealthy status."""
        # Mock psutil responses for unhealthy state
        mock_psutil.cpu_percent.return_value = 95.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=98.0,
            available=64 * 1024 * 1024,
            total=4096 * 1024 * 1024,
        )
        mock_psutil.disk_usage.return_value = Mock(
            used=1996 * 1024 * 1024,
            total=2048 * 1024 * 1024,
            free=52 * 1024 * 1024,
        )

        service = HealthService()

        checks = asyncio.run(service._check_system_resources())

        # Check CPU
        cpu_check = checks["cpu"]
        assert cpu_check.status == HealthStatus.UNHEALTHY

        # Check memory
        memory_check = checks["memory"]
        assert memory_check.status == HealthStatus.UNHEALTHY

        # Check disk
        disk_check = checks["disk"]
        assert disk_check.status == HealthStatus.UNHEALTHY

    @patch("pynomaly.infrastructure.monitoring.health_service.psutil")
    def test_check_system_resources_exception(self, mock_psutil):
        """Test system resources check with exception."""
        # Mock psutil to raise exception
        mock_psutil.cpu_percent.side_effect = Exception("CPU check failed")

        service = HealthService()

        checks = asyncio.run(service._check_system_resources())

        assert "system" in checks
        system_check = checks["system"]
        assert system_check.status == HealthStatus.UNHEALTHY
        assert "Failed to check system resources" in system_check.message
        assert system_check.details["error"] == "CPU check failed"

    def test_check_database_success(self):
        """Test successful database health check."""
        # Mock SQLAlchemy engine
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = (1,)
        mock_connection.execute.return_value = mock_result
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        mock_engine.connect.return_value = mock_connection

        service = HealthService()

        checks = asyncio.run(service._check_database(mock_engine))

        assert "database" in checks
        db_check = checks["database"]
        assert db_check.status == HealthStatus.HEALTHY
        assert "Database responsive" in db_check.message
        assert "response_time_ms" in db_check.details

    def test_check_database_slow_response(self):
        """Test database health check with slow response."""
        # Mock SQLAlchemy engine with slow response
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = (1,)
        mock_connection.execute.return_value = mock_result
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        mock_engine.connect.return_value = mock_connection

        service = HealthService()

        # Simulate slow response
        with patch("time.time", side_effect=[0, 0.6]):  # 600ms response
            checks = asyncio.run(service._check_database(mock_engine))

        assert "database" in checks
        db_check = checks["database"]
        assert db_check.status == HealthStatus.UNHEALTHY
        assert "Database responsive" in db_check.message

    def test_check_database_connection_failure(self):
        """Test database health check with connection failure."""
        # Mock SQLAlchemy engine to raise exception
        mock_engine = Mock()
        mock_engine.connect.side_effect = Exception("Connection failed")

        service = HealthService()

        checks = asyncio.run(service._check_database(mock_engine))

        assert "database" in checks
        db_check = checks["database"]
        assert db_check.status == HealthStatus.UNHEALTHY
        assert "Database connection failed" in db_check.message
        assert db_check.details["error"] == "Connection failed"

    def test_check_database_unexpected_result(self):
        """Test database health check with unexpected result."""
        # Mock SQLAlchemy engine with unexpected result
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = (42,)  # Unexpected result
        mock_connection.execute.return_value = mock_result
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        mock_engine.connect.return_value = mock_connection

        service = HealthService()

        checks = asyncio.run(service._check_database(mock_engine))

        assert "database" in checks
        db_check = checks["database"]
        assert db_check.status == HealthStatus.UNHEALTHY
        assert "unexpected result" in db_check.message

    def test_check_redis_success(self):
        """Test successful Redis health check."""
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.info.return_value = {
            "connected_clients": 5,
            "used_memory": 1024 * 1024,
            "keyspace_hits": 1000,
            "keyspace_misses": 100,
        }

        service = HealthService()

        checks = asyncio.run(service._check_redis(mock_redis))

        assert "redis" in checks
        redis_check = checks["redis"]
        assert redis_check.status == HealthStatus.HEALTHY
        assert "Redis responsive" in redis_check.message
        assert redis_check.details["connected_clients"] == 5
        assert redis_check.details["used_memory_mb"] == 1.0

    def test_check_redis_slow_response(self):
        """Test Redis health check with slow response."""
        # Mock Redis client with slow response
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.info.return_value = {}

        service = HealthService()

        # Simulate slow response
        with patch("time.time", side_effect=[0, 0.06]):  # 60ms response
            checks = asyncio.run(service._check_redis(mock_redis))

        assert "redis" in checks
        redis_check = checks["redis"]
        assert redis_check.status == HealthStatus.UNHEALTHY

    def test_check_redis_connection_failure(self):
        """Test Redis health check with connection failure."""
        # Mock Redis client to raise exception
        mock_redis = Mock()
        mock_redis.ping.side_effect = Exception("Connection refused")

        service = HealthService()

        checks = asyncio.run(service._check_redis(mock_redis))

        assert "redis" in checks
        redis_check = checks["redis"]
        assert redis_check.status == HealthStatus.UNHEALTHY
        assert "Redis connection failed" in redis_check.message
        assert redis_check.details["error"] == "Connection refused"

    def test_check_redis_ping_failure(self):
        """Test Redis health check with ping failure."""
        # Mock Redis client with ping failure
        mock_redis = Mock()
        mock_redis.ping.return_value = False

        service = HealthService()

        checks = asyncio.run(service._check_redis(mock_redis))

        assert "redis" in checks
        redis_check = checks["redis"]
        assert redis_check.status == HealthStatus.UNHEALTHY
        assert "Redis ping failed" in redis_check.message

    @patch("pynomaly.infrastructure.monitoring.health_service.psutil")
    def test_check_application_health(self, mock_psutil):
        """Test application health check."""
        # Mock psutil Process
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 256 * 1024 * 1024  # 256MB
        mock_memory_info.vms = 512 * 1024 * 1024  # 512MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.num_threads.return_value = 10
        mock_psutil.Process.return_value = mock_process

        service = HealthService()

        checks = asyncio.run(service._check_application_health())

        assert "uptime" in checks
        assert "application_memory" in checks

        # Check uptime
        uptime_check = checks["uptime"]
        assert uptime_check.status == HealthStatus.HEALTHY
        assert "Running for" in uptime_check.message
        assert uptime_check.details["uptime_seconds"] >= 0

        # Check application memory
        memory_check = checks["application_memory"]
        assert memory_check.status == HealthStatus.HEALTHY
        assert "Process using 256.0MB" in memory_check.message
        assert memory_check.details["rss_mb"] == 256.0
        assert memory_check.details["num_threads"] == 10

    @patch("pynomaly.infrastructure.monitoring.health_service.psutil")
    def test_check_application_health_high_memory(self, mock_psutil):
        """Test application health check with high memory usage."""
        # Mock psutil Process with high memory usage
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1536 * 1024 * 1024  # 1.5GB
        mock_memory_info.vms = 2048 * 1024 * 1024  # 2GB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.num_threads.return_value = 50
        mock_psutil.Process.return_value = mock_process

        service = HealthService()

        checks = asyncio.run(service._check_application_health())

        # Check application memory
        memory_check = checks["application_memory"]
        assert memory_check.status == HealthStatus.UNHEALTHY
        assert "Process using 1536.0MB" in memory_check.message

    @patch("pynomaly.infrastructure.monitoring.health_service.psutil")
    def test_check_application_health_exception(self, mock_psutil):
        """Test application health check with exception."""
        # Mock psutil to raise exception
        mock_psutil.Process.side_effect = Exception("Process error")

        service = HealthService()

        checks = asyncio.run(service._check_application_health())

        assert "application" in checks
        app_check = checks["application"]
        assert app_check.status == HealthStatus.UNHEALTHY
        assert "Failed to check application health" in app_check.message
        assert app_check.details["error"] == "Process error"

    def test_run_custom_checks_success(self):
        """Test running custom health checks successfully."""
        service = HealthService()

        def custom_check_1():
            return {"status": "healthy", "message": "Custom check 1 passed"}

        async def custom_check_2():
            return {
                "status": "degraded",
                "message": "Custom check 2 warning",
                "details": {"metric": 75},
            }

        custom_checks = {
            "custom_1": custom_check_1,
            "custom_2": custom_check_2,
        }

        checks = asyncio.run(service._run_custom_checks(custom_checks))

        assert "custom_1" in checks
        assert "custom_2" in checks

        # Check custom_1
        check1 = checks["custom_1"]
        assert check1.status == HealthStatus.HEALTHY
        assert check1.message == "Custom check 1 passed"

        # Check custom_2
        check2 = checks["custom_2"]
        assert check2.status == HealthStatus.DEGRADED
        assert check2.message == "Custom check 2 warning"
        assert check2.details["metric"] == 75

    def test_run_custom_checks_boolean_result(self):
        """Test running custom health checks with boolean results."""
        service = HealthService()

        def success_check():
            return True

        def failure_check():
            return False

        custom_checks = {
            "success": success_check,
            "failure": failure_check,
        }

        checks = asyncio.run(service._run_custom_checks(custom_checks))

        assert "success" in checks
        assert "failure" in checks

        # Check success
        success_check = checks["success"]
        assert success_check.status == HealthStatus.HEALTHY
        assert "returned: True" in success_check.message

        # Check failure
        failure_check = checks["failure"]
        assert failure_check.status == HealthStatus.UNHEALTHY
        assert "returned: False" in failure_check.message

    def test_run_custom_checks_exception(self):
        """Test running custom health checks with exceptions."""
        service = HealthService()

        def failing_check():
            raise Exception("Custom check failed")

        custom_checks = {
            "failing": failing_check,
        }

        checks = asyncio.run(service._run_custom_checks(custom_checks))

        assert "failing" in checks
        failing_check = checks["failing"]
        assert failing_check.status == HealthStatus.UNHEALTHY
        assert "Custom check failed" in failing_check.message
        assert failing_check.details["error"] == "Custom check failed"

    def test_store_check_results(self):
        """Test storing health check results."""
        service = HealthService(max_history=3)

        # Store first result
        checks1 = {"test": HealthCheck("test", HealthStatus.HEALTHY, "OK", 10.0)}
        service._store_check_results(checks1)

        assert len(service._check_history) == 1
        assert service._check_history[0] == checks1

        # Store second result
        checks2 = {"test": HealthCheck("test", HealthStatus.DEGRADED, "Warning", 20.0)}
        service._store_check_results(checks2)

        assert len(service._check_history) == 2
        assert service._check_history[1] == checks2

        # Store third result
        checks3 = {"test": HealthCheck("test", HealthStatus.UNHEALTHY, "Error", 30.0)}
        service._store_check_results(checks3)

        assert len(service._check_history) == 3
        assert service._check_history[2] == checks3

        # Store fourth result - should remove first
        checks4 = {"test": HealthCheck("test", HealthStatus.HEALTHY, "OK", 40.0)}
        service._store_check_results(checks4)

        assert len(service._check_history) == 3
        assert service._check_history[0] == checks2
        assert service._check_history[1] == checks3
        assert service._check_history[2] == checks4

    def test_get_overall_status_healthy(self):
        """Test getting overall status when all checks are healthy."""
        service = HealthService()

        checks = {
            "cpu": HealthCheck("CPU", HealthStatus.HEALTHY, "OK", 10.0),
            "memory": HealthCheck("Memory", HealthStatus.HEALTHY, "OK", 15.0),
            "disk": HealthCheck("Disk", HealthStatus.HEALTHY, "OK", 20.0),
        }

        overall_status = service.get_overall_status(checks)
        assert overall_status == HealthStatus.HEALTHY

    def test_get_overall_status_degraded(self):
        """Test getting overall status when some checks are degraded."""
        service = HealthService()

        checks = {
            "cpu": HealthCheck("CPU", HealthStatus.HEALTHY, "OK", 10.0),
            "memory": HealthCheck("Memory", HealthStatus.DEGRADED, "Warning", 15.0),
            "disk": HealthCheck("Disk", HealthStatus.HEALTHY, "OK", 20.0),
        }

        overall_status = service.get_overall_status(checks)
        assert overall_status == HealthStatus.DEGRADED

    def test_get_overall_status_unhealthy(self):
        """Test getting overall status when some checks are unhealthy."""
        service = HealthService()

        checks = {
            "cpu": HealthCheck("CPU", HealthStatus.HEALTHY, "OK", 10.0),
            "memory": HealthCheck("Memory", HealthStatus.DEGRADED, "Warning", 15.0),
            "disk": HealthCheck("Disk", HealthStatus.UNHEALTHY, "Error", 20.0),
        }

        overall_status = service.get_overall_status(checks)
        assert overall_status == HealthStatus.UNHEALTHY

    def test_get_overall_status_unknown(self):
        """Test getting overall status when no checks are provided."""
        service = HealthService()

        overall_status = service.get_overall_status({})
        assert overall_status == HealthStatus.UNKNOWN

    @patch("pynomaly.infrastructure.monitoring.health_service.psutil")
    def test_get_system_metrics(self, mock_psutil):
        """Test getting system metrics."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=60.0,
            available=1536 * 1024 * 1024,
        )
        mock_psutil.disk_usage.return_value = Mock(
            used=1024 * 1024 * 1024,
            total=2048 * 1024 * 1024,
            free=1024 * 1024 * 1024,
        )
        mock_psutil.getloadavg.return_value = (0.5, 0.8, 1.2)
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1024,
            bytes_recv=2048,
            packets_sent=10,
            packets_recv=20,
        )
        mock_psutil.pids.return_value = list(range(100))

        service = HealthService()

        metrics = service.get_system_metrics()

        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 45.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_percent == 50.0
        assert metrics.memory_available_mb == 1536.0
        assert metrics.disk_available_gb == 1.0
        assert metrics.load_average == [0.5, 0.8, 1.2]
        assert metrics.network_io["bytes_sent"] == 1024
        assert metrics.network_io["bytes_recv"] == 2048
        assert metrics.process_count == 100
        assert metrics.uptime_seconds >= 0

    @patch("pynomaly.infrastructure.monitoring.health_service.psutil")
    def test_get_system_metrics_windows(self, mock_psutil):
        """Test getting system metrics on Windows (no load average)."""
        # Mock psutil responses for Windows
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=60.0,
            available=1536 * 1024 * 1024,
        )
        mock_psutil.disk_usage.return_value = Mock(
            used=1024 * 1024 * 1024,
            total=2048 * 1024 * 1024,
            free=1024 * 1024 * 1024,
        )
        mock_psutil.getloadavg.side_effect = AttributeError(
            "No load average on Windows"
        )
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1024,
            bytes_recv=2048,
            packets_sent=10,
            packets_recv=20,
        )
        mock_psutil.pids.return_value = list(range(100))

        service = HealthService()

        metrics = service.get_system_metrics()

        assert metrics.load_average == [0.0, 0.0, 0.0]

    @patch("pynomaly.infrastructure.monitoring.health_service.psutil")
    def test_get_system_metrics_network_error(self, mock_psutil):
        """Test getting system metrics with network error."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=60.0,
            available=1536 * 1024 * 1024,
        )
        mock_psutil.disk_usage.return_value = Mock(
            used=1024 * 1024 * 1024,
            total=2048 * 1024 * 1024,
            free=1024 * 1024 * 1024,
        )
        mock_psutil.getloadavg.return_value = (0.5, 0.8, 1.2)
        mock_psutil.net_io_counters.side_effect = Exception("Network error")
        mock_psutil.pids.return_value = list(range(100))

        service = HealthService()

        metrics = service.get_system_metrics()

        assert metrics.network_io == {"bytes_sent": 0, "bytes_recv": 0}

    @patch("pynomaly.infrastructure.monitoring.health_service.psutil")
    def test_get_system_metrics_exception(self, mock_psutil):
        """Test getting system metrics with exception."""
        # Mock psutil to raise exception
        mock_psutil.cpu_percent.side_effect = Exception("System error")

        service = HealthService()

        with pytest.raises(Exception, match="Failed to get system metrics"):
            service.get_system_metrics()

    def test_get_health_history(self):
        """Test getting health history."""
        service = HealthService()

        # Create some historical checks
        now = datetime.utcnow()

        # Recent check (within 24 hours)
        recent_check = HealthCheck("test", HealthStatus.HEALTHY, "OK", 10.0)
        recent_check.timestamp = now - timedelta(hours=1)

        # Old check (outside 24 hours)
        old_check = HealthCheck("test", HealthStatus.HEALTHY, "OK", 10.0)
        old_check.timestamp = now - timedelta(hours=25)

        service._check_history = [
            {"recent": recent_check},
            {"old": old_check},
        ]

        # Get history for last 24 hours
        history = service.get_health_history(hours=24)

        assert len(history) == 1
        assert "recent" in history[0]

    def test_get_health_summary(self):
        """Test getting health summary."""
        service = HealthService()

        # Create some checks
        checks = {
            "cpu": HealthCheck("CPU", HealthStatus.HEALTHY, "OK", 10.0),
            "memory": HealthCheck("Memory", HealthStatus.DEGRADED, "Warning", 15.0),
            "disk": HealthCheck("Disk", HealthStatus.UNHEALTHY, "Error", 20.0),
        }

        service._check_history = [checks]

        with patch.object(service, "get_system_metrics") as mock_metrics:
            mock_metrics.return_value = SystemMetrics(
                cpu_percent=45.0,
                memory_percent=60.0,
                disk_percent=50.0,
                memory_available_mb=1536.0,
                disk_available_gb=1.0,
                load_average=[0.5, 0.8, 1.2],
                network_io={"bytes_sent": 1024, "bytes_recv": 2048},
                process_count=100,
                uptime_seconds=3600.0,
            )

            summary = service.get_health_summary()

        assert summary["overall_status"] == "unhealthy"
        assert summary["total_checks"] == 3
        assert summary["healthy_checks"] == 1
        assert summary["degraded_checks"] == 1
        assert summary["unhealthy_checks"] == 1
        assert summary["uptime_hours"] == 1.0
        assert summary["cpu_percent"] == 45.0
        assert summary["memory_percent"] == 60.0
        assert summary["disk_percent"] == 50.0

    def test_get_health_summary_no_checks(self):
        """Test getting health summary with no checks."""
        service = HealthService()

        summary = service.get_health_summary()

        assert summary["status"] == "unknown"
        assert summary["message"] == "No health checks performed"

    def test_perform_comprehensive_health_check(self):
        """Test performing comprehensive health check."""
        service = HealthService()

        # Mock all the components
        with (
            patch.object(service, "_check_system_resources") as mock_system,
            patch.object(service, "_check_database") as mock_database,
            patch.object(service, "_check_redis") as mock_redis,
            patch.object(service, "_check_application_health") as mock_app,
            patch.object(service, "_run_custom_checks") as mock_custom,
        ):
            # Setup mock returns
            mock_system.return_value = {
                "cpu": HealthCheck("CPU", HealthStatus.HEALTHY, "OK", 10.0)
            }
            mock_database.return_value = {
                "db": HealthCheck("DB", HealthStatus.HEALTHY, "OK", 20.0)
            }
            mock_redis.return_value = {
                "redis": HealthCheck("Redis", HealthStatus.HEALTHY, "OK", 5.0)
            }
            mock_app.return_value = {
                "app": HealthCheck("App", HealthStatus.HEALTHY, "OK", 15.0)
            }
            mock_custom.return_value = {
                "custom": HealthCheck("Custom", HealthStatus.HEALTHY, "OK", 8.0)
            }

            # Create mock components
            mock_engine = Mock()
            mock_redis_client = Mock()
            mock_custom_checks = {"custom_check": lambda: True}

            # Run comprehensive check
            checks = asyncio.run(
                service.perform_comprehensive_health_check(
                    database_engine=mock_engine,
                    redis_client=mock_redis_client,
                    custom_checks=mock_custom_checks,
                )
            )

            # Verify all checks were called
            mock_system.assert_called_once()
            mock_database.assert_called_once_with(mock_engine)
            mock_redis.assert_called_once_with(mock_redis_client)
            mock_app.assert_called_once()
            mock_custom.assert_called_once_with(mock_custom_checks)

            # Verify results
            assert "cpu" in checks
            assert "db" in checks
            assert "redis" in checks
            assert "app" in checks
            assert "custom" in checks

            # Verify history was stored
            assert len(service._check_history) == 1
            assert service._check_history[0] == checks

    def test_perform_comprehensive_health_check_minimal(self):
        """Test performing comprehensive health check with minimal components."""
        service = HealthService()

        # Mock only system and application checks
        with (
            patch.object(service, "_check_system_resources") as mock_system,
            patch.object(service, "_check_application_health") as mock_app,
        ):
            # Setup mock returns
            mock_system.return_value = {
                "cpu": HealthCheck("CPU", HealthStatus.HEALTHY, "OK", 10.0)
            }
            mock_app.return_value = {
                "app": HealthCheck("App", HealthStatus.HEALTHY, "OK", 15.0)
            }

            # Run comprehensive check with no optional components
            checks = asyncio.run(service.perform_comprehensive_health_check())

            # Verify only required checks were called
            mock_system.assert_called_once()
            mock_app.assert_called_once()

            # Verify results
            assert "cpu" in checks
            assert "app" in checks
            assert len(checks) == 2

    def test_concurrent_health_checks(self):
        """Test concurrent health checks don't interfere."""
        service = HealthService()

        async def run_concurrent_checks():
            # Run multiple health checks concurrently
            tasks = []
            for i in range(5):
                task = asyncio.create_task(service.perform_comprehensive_health_check())
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            return results

        # Mock system resources to avoid actual system calls
        with (
            patch.object(service, "_check_system_resources") as mock_system,
            patch.object(service, "_check_application_health") as mock_app,
        ):
            mock_system.return_value = {
                "cpu": HealthCheck("CPU", HealthStatus.HEALTHY, "OK", 10.0)
            }
            mock_app.return_value = {
                "app": HealthCheck("App", HealthStatus.HEALTHY, "OK", 15.0)
            }

            results = asyncio.run(run_concurrent_checks())

            # Verify all checks completed successfully
            assert len(results) == 5
            for result in results:
                assert "cpu" in result
                assert "app" in result

            # Verify history contains all checks
            assert len(service._check_history) == 5

    def test_health_service_performance(self):
        """Test health service performance under load."""
        service = HealthService()

        # Mock system resources to avoid actual system calls
        with (
            patch.object(service, "_check_system_resources") as mock_system,
            patch.object(service, "_check_application_health") as mock_app,
        ):
            mock_system.return_value = {
                "cpu": HealthCheck("CPU", HealthStatus.HEALTHY, "OK", 10.0)
            }
            mock_app.return_value = {
                "app": HealthCheck("App", HealthStatus.HEALTHY, "OK", 15.0)
            }

            # Measure performance
            start_time = time.time()

            # Run multiple health checks
            for _ in range(10):
                checks = asyncio.run(service.perform_comprehensive_health_check())
                assert len(checks) == 2

            end_time = time.time()
            duration = end_time - start_time

            # Should complete within reasonable time
            assert duration < 1.0  # Less than 1 second for 10 checks

            # Verify history management
            assert len(service._check_history) == 10
