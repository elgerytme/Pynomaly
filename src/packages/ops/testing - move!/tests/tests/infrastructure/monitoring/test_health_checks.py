"""Tests for health check system."""

from unittest.mock import Mock, patch

import pytest

from monorepo.infrastructure.monitoring.health_checks import (
    ComponentType,
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    SystemHealth,
    check_detector_service,
    check_filesystem_health,
    check_memory_health,
    check_model_repository,
    check_streaming_service,
    check_system_resources,
    get_health_checker,
    liveness_probe,
    readiness_probe,
    register_health_check,
)


class TestHealthCheckResult:
    """Test cases for HealthCheckResult."""

    def test_health_check_result_creation(self):
        """Test creating health check result."""
        result = HealthCheckResult(
            component="test_component",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.HEALTHY,
            message="Component is healthy",
            details={"connection_pool_size": 10},
            response_time_ms=150.5,
        )

        assert result.component == "test_component"
        assert result.component_type == ComponentType.DATABASE
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Component is healthy"
        assert result.details["connection_pool_size"] == 10
        assert result.response_time_ms == 150.5
        assert result.is_healthy() is True

    def test_health_check_result_unhealthy(self):
        """Test unhealthy health check result."""
        result = HealthCheckResult(
            component="failing_service",
            component_type=ComponentType.EXTERNAL_SERVICE,
            status=HealthStatus.UNHEALTHY,
            message="Service is down",
        )

        assert result.is_healthy() is False
        assert result.status == HealthStatus.UNHEALTHY

    def test_health_check_result_to_dict(self):
        """Test converting health check result to dictionary."""
        result = HealthCheckResult(
            component="test_db",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.DEGRADED,
            message="High latency detected",
            details={"latency_ms": 500},
        )

        result_dict = result.to_dict()

        assert result_dict["component"] == "test_db"
        assert result_dict["component_type"] == "database"
        assert result_dict["status"] == "degraded"
        assert result_dict["message"] == "High latency detected"
        assert result_dict["details"]["latency_ms"] == 500
        assert "timestamp" in result_dict


class TestSystemHealth:
    """Test cases for SystemHealth."""

    def test_system_health_creation(self):
        """Test creating system health summary."""
        check1 = HealthCheckResult(
            component="db",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.HEALTHY,
            message="Database OK",
        )

        check2 = HealthCheckResult(
            component="cache",
            component_type=ComponentType.CACHE,
            status=HealthStatus.DEGRADED,
            message="Cache slow",
        )

        system_health = SystemHealth(
            status=HealthStatus.DEGRADED,
            message="System partially degraded",
            checks=[check1, check2],
            version="1.0.0",
            uptime_seconds=3600.0,
        )

        assert system_health.status == HealthStatus.DEGRADED
        assert system_health.message == "System partially degraded"
        assert len(system_health.checks) == 2
        assert system_health.version == "1.0.0"
        assert system_health.uptime_seconds == 3600.0

    def test_system_health_to_dict(self):
        """Test converting system health to dictionary."""
        check1 = HealthCheckResult(
            component="test",
            component_type=ComponentType.CPU,
            status=HealthStatus.HEALTHY,
            message="OK",
        )

        system_health = SystemHealth(
            status=HealthStatus.HEALTHY,
            message="All systems operational",
            checks=[check1],
        )

        health_dict = system_health.to_dict()

        assert health_dict["status"] == "healthy"
        assert health_dict["message"] == "All systems operational"
        assert len(health_dict["checks"]) == 1
        assert health_dict["summary"]["total_checks"] == 1
        assert health_dict["summary"]["healthy"] == 1
        assert health_dict["summary"]["degraded"] == 0
        assert health_dict["summary"]["unhealthy"] == 0


class TestHealthChecker:
    """Test cases for HealthChecker."""

    def test_health_checker_creation(self):
        """Test creating health checker."""
        checker = HealthChecker()
        assert checker._check_functions == {}
        assert checker._last_check_results == {}
        assert checker._start_time > 0

    def test_register_check(self):
        """Test registering health check function."""
        checker = HealthChecker()

        async def dummy_check():
            return HealthCheckResult(
                component="test",
                component_type=ComponentType.EXTERNAL_SERVICE,
                status=HealthStatus.HEALTHY,
                message="Test OK",
            )

        checker.register_check("test_check", dummy_check)
        assert "test_check" in checker._check_functions
        assert checker._check_functions["test_check"] == dummy_check

    @pytest.mark.asyncio
    async def test_check_component_success(self):
        """Test successful component health check."""
        checker = HealthChecker()

        async def success_check():
            return HealthCheckResult(
                component="test_service",
                component_type=ComponentType.EXTERNAL_SERVICE,
                status=HealthStatus.HEALTHY,
                message="Service is healthy",
            )

        checker.register_check("success_test", success_check)

        result = await checker.check_component("success_test")

        assert result.component == "test_service"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Service is healthy"
        assert result.response_time_ms is not None
        assert result.response_time_ms >= 0

        # Should be cached
        assert "success_test" in checker._last_check_results

    @pytest.mark.asyncio
    async def test_check_component_failure(self):
        """Test failed component health check."""
        checker = HealthChecker()

        async def failing_check():
            raise Exception("Service connection failed")

        checker.register_check("failing_test", failing_check)

        result = await checker.check_component("failing_test")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Health check failed" in result.message
        assert "Service connection failed" in result.message
        assert result.response_time_ms is not None

    @pytest.mark.asyncio
    async def test_check_component_not_found(self):
        """Test checking non-existent component."""
        checker = HealthChecker()

        result = await checker.check_component("nonexistent")

        assert result.status == HealthStatus.UNKNOWN
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_check_all_components(self):
        """Test checking all registered components."""
        checker = HealthChecker()

        async def check1():
            return HealthCheckResult(
                component="service1",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.HEALTHY,
                message="OK",
            )

        async def check2():
            return HealthCheckResult(
                component="service2",
                component_type=ComponentType.CACHE,
                status=HealthStatus.DEGRADED,
                message="Slow",
            )

        checker.register_check("check1", check1)
        checker.register_check("check2", check2)

        results = await checker.check_all_components()

        assert len(results) == 2
        assert any(
            r.component == "service1" and r.status == HealthStatus.HEALTHY
            for r in results
        )
        assert any(
            r.component == "service2" and r.status == HealthStatus.DEGRADED
            for r in results
        )

    @pytest.mark.asyncio
    async def test_check_all_components_empty(self):
        """Test checking all components when none are registered."""
        checker = HealthChecker()

        results = await checker.check_all_components()

        assert results == []

    @pytest.mark.asyncio
    async def test_get_system_health_healthy(self):
        """Test getting system health when all components are healthy."""
        checker = HealthChecker()

        async def healthy_check():
            return HealthCheckResult(
                component="service",
                component_type=ComponentType.EXTERNAL_SERVICE,
                status=HealthStatus.HEALTHY,
                message="OK",
            )

        checker.register_check("test", healthy_check)

        system_health = await checker.get_system_health("1.0.0")

        assert system_health.status == HealthStatus.HEALTHY
        assert "All systems operational" in system_health.message
        assert system_health.version == "1.0.0"
        assert system_health.uptime_seconds > 0
        assert len(system_health.checks) == 1

    @pytest.mark.asyncio
    async def test_get_system_health_degraded(self):
        """Test getting system health when some components are degraded."""
        checker = HealthChecker()

        async def healthy_check():
            return HealthCheckResult(
                component="service1",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.HEALTHY,
                message="OK",
            )

        async def degraded_check():
            return HealthCheckResult(
                component="service2",
                component_type=ComponentType.CACHE,
                status=HealthStatus.DEGRADED,
                message="Slow",
            )

        checker.register_check("healthy", healthy_check)
        checker.register_check("degraded", degraded_check)

        system_health = await checker.get_system_health()

        assert system_health.status == HealthStatus.DEGRADED
        assert "degraded" in system_health.message
        assert len(system_health.checks) == 2

    @pytest.mark.asyncio
    async def test_get_system_health_unhealthy(self):
        """Test getting system health when components are unhealthy."""
        checker = HealthChecker()

        async def unhealthy_check():
            return HealthCheckResult(
                component="failed_service",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                message="Connection failed",
            )

        checker.register_check("unhealthy", unhealthy_check)

        system_health = await checker.get_system_health()

        assert system_health.status == HealthStatus.UNHEALTHY
        assert "unhealthy" in system_health.message

    @pytest.mark.asyncio
    async def test_get_system_health_no_checks(self):
        """Test getting system health when no checks are configured."""
        checker = HealthChecker()

        system_health = await checker.get_system_health()

        assert system_health.status == HealthStatus.UNKNOWN
        assert "No health checks configured" in system_health.message
        assert len(system_health.checks) == 0

    def test_get_cached_results(self):
        """Test getting cached health check results."""
        checker = HealthChecker()

        # Add some cached results
        result = HealthCheckResult(
            component="test",
            component_type=ComponentType.CACHE,
            status=HealthStatus.HEALTHY,
            message="Cached result",
        )
        checker._last_check_results["test"] = result

        cached = checker.get_cached_results()

        assert "test" in cached
        assert cached["test"].message == "Cached result"

        # Should be a copy, not the original
        cached["test"] = None
        assert checker._last_check_results["test"] is not None


class TestPredefinedHealthChecks:
    """Test cases for predefined health check functions."""

    @pytest.mark.asyncio
    async def test_check_system_resources(self):
        """Test system resources health check."""
        with (
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.cpu_percent") as mock_cpu,
            patch("psutil.disk_usage") as mock_disk,
        ):
            # Mock normal resource usage
            mock_memory.return_value = Mock(
                total=8 * 1024**3,
                available=4 * 1024**3,
                percent=50.0,
                used=4 * 1024**3,
            )
            mock_cpu.return_value = 25.0
            mock_disk.return_value = Mock(
                total=100 * 1024**3, used=30 * 1024**3, free=70 * 1024**3
            )

            result = await check_system_resources()

            assert result.component == "system_resources"
            assert result.component_type == ComponentType.CPU
            assert result.status == HealthStatus.HEALTHY
            assert "within normal limits" in result.message
            assert "memory" in result.details
            assert "cpu" in result.details
            assert "disk" in result.details

    @pytest.mark.asyncio
    async def test_check_system_resources_high_usage(self):
        """Test system resources health check with high usage."""
        with (
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.cpu_percent") as mock_cpu,
            patch("psutil.disk_usage") as mock_disk,
        ):
            # Mock high resource usage
            mock_memory.return_value = Mock(
                total=8 * 1024**3,
                available=1 * 1024**3,
                percent=95.0,
                used=7 * 1024**3,
            )
            mock_cpu.return_value = 98.0
            mock_disk.return_value = Mock(
                total=100 * 1024**3, used=97 * 1024**3, free=3 * 1024**3
            )

            result = await check_system_resources()

            assert result.status == HealthStatus.UNHEALTHY
            assert "Critical resource usage" in result.message

    @pytest.mark.asyncio
    async def test_check_memory_health(self):
        """Test memory health check."""
        with (
            patch("psutil.virtual_memory") as mock_virtual,
            patch("psutil.swap_memory") as mock_swap,
        ):
            mock_virtual.return_value = Mock(
                total=8 * 1024**3,
                available=4 * 1024**3,
                used=4 * 1024**3,
                percent=50.0,
            )
            mock_swap.return_value = Mock(
                total=2 * 1024**3, used=1 * 1024**3, percent=25.0
            )

            result = await check_memory_health()

            assert result.component == "memory"
            assert result.component_type == ComponentType.MEMORY
            assert result.status == HealthStatus.HEALTHY
            assert "normal" in result.message
            assert "virtual_memory" in result.details
            assert "swap_memory" in result.details

    @pytest.mark.asyncio
    async def test_check_filesystem_health(self):
        """Test filesystem health check."""
        with (
            patch("psutil.disk_usage") as mock_disk,
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("builtins.open") as mock_open,
        ):
            mock_disk.return_value = Mock(
                total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
            )

            # Mock successful file operations
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test_file"
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "health_check_test"
            )

            with patch("os.path.exists", return_value=True), patch("os.unlink"):
                result = await check_filesystem_health()

                assert result.component == "filesystem"
                assert result.component_type == ComponentType.FILESYSTEM
                assert result.status == HealthStatus.HEALTHY
                assert "healthy" in result.message
                assert result.details["write_test"] is True

    @pytest.mark.asyncio
    async def test_check_model_repository(self):
        """Test model repository health check."""
        result = await check_model_repository()

        assert result.component == "model_repository"
        assert result.component_type == ComponentType.MODEL_REPOSITORY
        assert result.status == HealthStatus.HEALTHY
        assert "operational" in result.message
        assert "models_loaded" in result.details

    @pytest.mark.asyncio
    async def test_check_detector_service(self):
        """Test detector service health check."""
        result = await check_detector_service()

        assert result.component == "detector_service"
        assert result.component_type == ComponentType.DETECTOR_SERVICE
        assert result.status == HealthStatus.HEALTHY
        assert "operational" in result.message
        assert "active_detectors" in result.details
        assert "algorithms_available" in result.details

    @pytest.mark.asyncio
    async def test_check_streaming_service(self):
        """Test streaming service health check."""
        result = await check_streaming_service()

        assert result.component == "streaming_service"
        assert result.component_type == ComponentType.STREAMING_SERVICE
        # Status could be healthy or degraded depending on mock values
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert "streaming service" in result.message.lower()
        assert "active_streams" in result.details


class TestGlobalHealthChecker:
    """Test cases for global health checker functions."""

    def test_get_health_checker_singleton(self):
        """Test that global health checker is a singleton."""
        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is checker2
        assert isinstance(checker1, HealthChecker)

        # Should have default checks registered
        assert len(checker1._check_functions) > 0
        assert "system_resources" in checker1._check_functions
        assert "memory" in checker1._check_functions

    def test_register_health_check_global(self):
        """Test registering health check globally."""

        async def custom_check():
            return HealthCheckResult(
                component="custom_service",
                component_type=ComponentType.EXTERNAL_SERVICE,
                status=HealthStatus.HEALTHY,
                message="Custom check OK",
            )

        register_health_check("custom_test", custom_check)

        checker = get_health_checker()
        assert "custom_test" in checker._check_functions


class TestKubernetesProbes:
    """Test cases for Kubernetes probe endpoints."""

    @pytest.mark.asyncio
    async def test_liveness_probe(self):
        """Test liveness probe."""
        result = await liveness_probe()

        assert result.status == "alive"
        assert result.timestamp is not None
        assert result.details is not None
        assert "uptime_seconds" in result.details

    @pytest.mark.asyncio
    async def test_readiness_probe_ready(self):
        """Test readiness probe when system is ready."""
        # Mock healthy critical components
        checker = get_health_checker()

        async def healthy_memory_check():
            return HealthCheckResult(
                component="memory",
                component_type=ComponentType.MEMORY,
                status=HealthStatus.HEALTHY,
                message="OK",
            )

        async def healthy_filesystem_check():
            return HealthCheckResult(
                component="filesystem",
                component_type=ComponentType.FILESYSTEM,
                status=HealthStatus.HEALTHY,
                message="OK",
            )

        async def healthy_model_repo_check():
            return HealthCheckResult(
                component="model_repository",
                component_type=ComponentType.MODEL_REPOSITORY,
                status=HealthStatus.HEALTHY,
                message="OK",
            )

        # Replace checks temporarily
        checker.register_check("memory", healthy_memory_check)
        checker.register_check("filesystem", healthy_filesystem_check)
        checker.register_check("model_repository", healthy_model_repo_check)

        result = await readiness_probe()

        assert result.status == "ready"
        assert result.details["memory"] == "healthy"
        assert result.details["filesystem"] == "healthy"
        assert result.details["model_repository"] == "healthy"

    @pytest.mark.asyncio
    async def test_readiness_probe_not_ready(self):
        """Test readiness probe when system is not ready."""
        checker = get_health_checker()

        async def unhealthy_memory_check():
            return HealthCheckResult(
                component="memory",
                component_type=ComponentType.MEMORY,
                status=HealthStatus.UNHEALTHY,
                message="Memory critical",
            )

        # Replace memory check temporarily
        checker.register_check("memory", unhealthy_memory_check)

        result = await readiness_probe()

        assert result.status == "not_ready"
        assert result.details["memory"] == "unhealthy"
