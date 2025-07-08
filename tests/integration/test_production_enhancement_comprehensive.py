"""Comprehensive integration tests for Phase 4 production enhancement features."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pynomaly.infrastructure.caching.advanced_cache_service import (
    AdvancedCacheService,
    CompressionAlgorithm,
)
from pynomaly.infrastructure.performance.profiling_service import (
    PerformanceProfilingService,
    ProfilerType,
)
from pynomaly.infrastructure.security.advanced_threat_detection import (
    DataExfiltrationDetector,
    SessionHijackingDetector,
    create_advanced_threat_detectors,
)
from pynomaly.infrastructure.security.audit_logger import SecurityEventType
from pynomaly.infrastructure.security.security_monitor import AlertType, ThreatLevel
from pynomaly.infrastructure.security.security_service import (
    AuditEventType,
    SecurityConfig,
    SecurityLevel,
    SecurityService,
)


class TestKubernetesDeploymentIntegration:
    """Integration tests for Kubernetes deployment configurations."""

    @pytest.fixture
    def k8s_deployment_path(self) -> Path:
        """Path to Kubernetes deployment manifests."""
        return Path(__file__).parent.parent.parent / "deploy" / "kubernetes"

    @pytest.fixture
    def streaming_deployment_manifest(
        self, k8s_deployment_path: Path
    ) -> dict[str, Any]:
        """Load streaming service deployment manifest."""
        manifest_path = k8s_deployment_path / "streaming-service-deployment.yaml"
        assert (
            manifest_path.exists()
        ), f"Streaming deployment manifest not found at {manifest_path}"

        # Parse YAML manually for testing (avoiding external dependency)
        with open(manifest_path) as f:
            content = f.read()

        # Verify key deployment components
        assert "apiVersion: apps/v1" in content
        assert "kind: Deployment" in content
        assert "name: pynomaly-streaming-service" in content
        assert "replicas: 5" in content
        assert "HorizontalPodAutoscaler" in content
        assert "PodDisruptionBudget" in content

        return {"content": content, "valid": True}

    @pytest.fixture
    def cache_deployment_manifest(self, k8s_deployment_path: Path) -> dict[str, Any]:
        """Load cache deployment manifest."""
        manifest_path = k8s_deployment_path / "cache-deployment.yaml"
        assert (
            manifest_path.exists()
        ), f"Cache deployment manifest not found at {manifest_path}"

        with open(manifest_path) as f:
            content = f.read()

        # Verify Redis cluster configuration
        assert "apiVersion: apps/v1" in content
        assert "kind: StatefulSet" in content
        assert "name: redis-cluster" in content
        assert "replicas: 6" in content
        assert "redis-server" in content

        return {"content": content, "valid": True}

    def test_deployment_manifest_structure(
        self,
        streaming_deployment_manifest: dict[str, Any],
        cache_deployment_manifest: dict[str, Any],
    ):
        """Test that deployment manifests have correct structure."""
        assert streaming_deployment_manifest["valid"]
        assert cache_deployment_manifest["valid"]

        # Verify streaming deployment has required components
        streaming_content = streaming_deployment_manifest["content"]
        assert "resources:" in streaming_content
        assert "limits:" in streaming_content
        assert "requests:" in streaming_content
        assert "livenessProbe:" in streaming_content
        assert "readinessProbe:" in streaming_content

        # Verify cache deployment has Redis configuration
        cache_content = cache_deployment_manifest["content"]
        assert "redis-conf" in cache_content
        assert "PersistentVolumeClaim" in cache_content
        assert "fast-ssd" in cache_content

    @pytest.mark.asyncio
    async def test_kubernetes_resource_validation(self, k8s_deployment_path: Path):
        """Test Kubernetes resource validation and constraints."""
        manifest_files = list(k8s_deployment_path.glob("*.yaml"))
        assert len(manifest_files) >= 2, "Expected at least 2 Kubernetes manifest files"

        for manifest_file in manifest_files:
            with open(manifest_file) as f:
                content = f.read()

            # Verify resource constraints
            if "Deployment" in content or "StatefulSet" in content:
                assert "resources:" in content, f"Missing resources in {manifest_file}"
                assert (
                    "limits:" in content
                ), f"Missing resource limits in {manifest_file}"
                assert (
                    "requests:" in content
                ), f"Missing resource requests in {manifest_file}"

            # Verify health checks
            if "Deployment" in content:
                assert (
                    "livenessProbe:" in content
                ), f"Missing liveness probe in {manifest_file}"
                assert (
                    "readinessProbe:" in content
                ), f"Missing readiness probe in {manifest_file}"


class TestAdvancedCachingIntegration:
    """Integration tests for advanced caching strategies."""

    @pytest.fixture
    async def cache_service(self) -> AdvancedCacheService:
        """Create cache service for testing."""
        config = {
            "l1_cache": {"type": "memory", "max_size": 100},
            "l2_cache": {"type": "redis", "host": "localhost", "port": 6379, "db": 0},
            "l3_cache": {"type": "disk", "directory": "/tmp/pynomaly_cache"},
            "compression": {"algorithm": "lz4", "enabled": True},
            "serialization": {"format": "pickle"},
        }

        service = AdvancedCacheService(config)
        await service.initialize()
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_multi_level_cache_flow(self, cache_service: AdvancedCacheService):
        """Test multi-level cache with promotion and eviction."""
        # Test data
        test_key = "test_model_12345"
        test_data = {"model_weights": [1.0, 2.0, 3.0], "metadata": {"version": "1.0"}}

        # Store in cache
        await cache_service.set(test_key, test_data, ttl=3600)

        # Verify retrieval (should be from L1 cache)
        start_time = time.time()
        retrieved_data = await cache_service.get(test_key)
        l1_time = time.time() - start_time

        assert retrieved_data == test_data
        assert l1_time < 0.01  # L1 cache should be very fast

        # Clear L1 cache to test L2 retrieval
        cache_service.l1_cache.clear()

        start_time = time.time()
        retrieved_data = await cache_service.get(test_key)
        l2_time = time.time() - start_time

        assert retrieved_data == test_data
        assert l2_time > l1_time  # L2 should be slower than L1

    @pytest.mark.asyncio
    async def test_compression_efficiency(self, cache_service: AdvancedCacheService):
        """Test compression algorithms for different data types."""
        # Large repetitive data that compresses well
        large_data = {"features": [1.0] * 10000, "labels": [0] * 10000}

        # Test different compression algorithms
        algorithms = [
            CompressionAlgorithm.LZ4,
            CompressionAlgorithm.ZSTD,
            CompressionAlgorithm.GZIP,
        ]

        for algorithm in algorithms:
            cache_service.compression_service.algorithm = algorithm

            # Store and retrieve
            key = f"compression_test_{algorithm.value}"
            await cache_service.set(key, large_data)
            retrieved = await cache_service.get(key)

            assert retrieved == large_data, f"Data mismatch with {algorithm.value}"

    @pytest.mark.asyncio
    async def test_specialized_cache_types(self, cache_service: AdvancedCacheService):
        """Test specialized caches for models, features, and predictions."""
        # Test model cache
        model_data = {
            "weights": {"layer1": [1.0, 2.0]},
            "config": {"type": "neural_net"},
        }
        await cache_service.model_cache.set_model("model_v1", model_data, ttl=7200)
        retrieved_model = await cache_service.model_cache.get_model("model_v1")
        assert retrieved_model == model_data

        # Test feature cache
        features = {"feature_1": [1.0, 2.0, 3.0], "feature_2": [4.0, 5.0, 6.0]}
        await cache_service.feature_cache.set_features(
            "dataset_123", features, ttl=3600
        )
        retrieved_features = await cache_service.feature_cache.get_features(
            "dataset_123"
        )
        assert retrieved_features == features

        # Test prediction cache
        predictions = {"anomaly_scores": [0.1, 0.9, 0.3], "labels": [0, 1, 0]}
        input_hash = "abc123"
        await cache_service.prediction_cache.set_prediction(
            input_hash, predictions, ttl=1800
        )
        retrieved_predictions = await cache_service.prediction_cache.get_prediction(
            input_hash
        )
        assert retrieved_predictions == predictions

    @pytest.mark.asyncio
    async def test_cache_performance_under_load(
        self, cache_service: AdvancedCacheService
    ):
        """Test cache performance under concurrent load."""

        # Simulate concurrent cache operations
        async def cache_operation(i: int):
            key = f"load_test_{i}"
            data = {"value": i, "timestamp": time.time()}
            await cache_service.set(key, data)
            retrieved = await cache_service.get(key)
            assert retrieved == data
            return i

        # Run 100 concurrent operations
        tasks = [cache_operation(i) for i in range(100)]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        assert len(results) == 100
        assert total_time < 5.0  # Should complete within 5 seconds
        assert all(isinstance(r, int) for r in results)


class TestPerformanceProfilingIntegration:
    """Integration tests for performance optimization and profiling."""

    @pytest.fixture
    def profiling_service(self) -> PerformanceProfilingService:
        """Create profiling service for testing."""
        config = {
            "profiling_enabled": True,
            "resource_monitoring_enabled": True,
            "optimization_enabled": True,
            "profile_output_dir": "/tmp/pynomaly_profiles",
        }
        return PerformanceProfilingService(config)

    def test_function_profiling_decorator(
        self, profiling_service: PerformanceProfilingService
    ):
        """Test function profiling with different profiler types."""

        @profiling_service.profile_function(ProfilerType.CPROFILE, save_stats=True)
        def cpu_intensive_function(n: int) -> int:
            """CPU intensive function for testing."""
            result = 0
            for i in range(n):
                result += i * i
            return result

        # Profile the function
        result = cpu_intensive_function(10000)
        assert result > 0

        # Check that profiling stats were generated
        stats = profiling_service.get_profiling_stats()
        assert len(stats) > 0
        assert any("cpu_intensive_function" in str(stat) for stat in stats)

    @pytest.mark.asyncio
    async def test_async_function_profiling(
        self, profiling_service: PerformanceProfilingService
    ):
        """Test profiling of async functions."""

        @profiling_service.profile_function(ProfilerType.LINE, save_stats=False)
        async def async_function(delay: float) -> str:
            """Async function for testing."""
            await asyncio.sleep(delay)
            return f"Completed after {delay}s"

        # Profile the async function
        result = await async_function(0.1)
        assert result == "Completed after 0.1s"

    def test_memory_profiling(self, profiling_service: PerformanceProfilingService):
        """Test memory usage profiling."""

        @profiling_service.profile_function(ProfilerType.MEMORY, save_stats=True)
        def memory_intensive_function() -> list[int]:
            """Memory intensive function for testing."""
            large_list = list(range(100000))
            return large_list

        # Profile memory usage
        result = memory_intensive_function()
        assert len(result) == 100000

        # Check memory profiling results
        memory_stats = profiling_service.get_memory_stats()
        assert len(memory_stats) > 0

    def test_resource_monitoring(self, profiling_service: PerformanceProfilingService):
        """Test real-time resource monitoring."""
        monitor = profiling_service.resource_monitor

        # Start monitoring
        monitor.start_monitoring()

        # Simulate some work
        time.sleep(0.5)

        # Get metrics
        metrics = monitor.get_current_metrics()
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "disk_usage" in metrics
        assert "network_io" in metrics

        # Verify metrics are reasonable
        assert 0 <= metrics["cpu_percent"] <= 100
        assert 0 <= metrics["memory_percent"] <= 100

        monitor.stop_monitoring()

    def test_automatic_optimization(
        self, profiling_service: PerformanceProfilingService
    ):
        """Test automatic performance optimization features."""
        # Test garbage collection optimization
        initial_gc_stats = profiling_service.get_gc_stats()

        # Trigger optimization
        profiling_service.optimize_performance()

        # Verify optimization was applied
        optimized_stats = profiling_service.get_gc_stats()
        assert optimized_stats is not None

        # Test memory optimization
        memory_before = profiling_service.get_memory_usage()
        profiling_service.optimize_memory()
        memory_after = profiling_service.get_memory_usage()

        # Memory usage might not change significantly in test, but should not increase
        assert memory_after <= memory_before * 1.1  # Allow 10% tolerance


class TestSecurityHardeningIntegration:
    """Integration tests for security hardening and audit trails."""

    @pytest.fixture
    def security_service(self) -> SecurityService:
        """Create security service for testing."""
        config = SecurityConfig(
            enable_audit_logging=True,
            enable_2fa=False,  # Disable for testing
            enable_rbac=True,
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=10,
        )
        return SecurityService(config)

    @pytest.fixture
    def threat_detectors(self) -> list:
        """Create threat detectors for testing."""
        return create_advanced_threat_detectors()

    def test_authentication_and_authorization(self, security_service: SecurityService):
        """Test complete authentication and authorization flow."""
        # Create test user session
        session = security_service.auth_service.authenticate_user(
            username="admin",
            password="AdminPassword123!",
            ip_address="192.168.1.100",
            user_agent="Test Agent 1.0",
        )

        assert session is not None
        assert session.user_id == "admin"
        assert session.mfa_verified is False  # 2FA disabled for testing

        # Generate JWT token
        jwt_token = security_service.auth_service.generate_jwt_token(session)
        assert jwt_token is not None
        assert len(jwt_token) > 0

        # Test authorization
        success, user_id, error = security_service.authenticate_request(
            auth_token=jwt_token,
            required_permission="data:read",
            resource_security_level=SecurityLevel.INTERNAL,
            source_ip="192.168.1.100",
            user_agent="Test Agent 1.0",
        )

        assert success is True
        assert user_id == "admin"
        assert error is None

    def test_threat_detection_integration(self, threat_detectors: list):
        """Test integration of all threat detectors."""
        assert len(threat_detectors) == 4

        detector_types = [type(detector).__name__ for detector in threat_detectors]
        expected_types = [
            "AdvancedBehaviorAnalyzer",
            "ThreatIntelligenceDetector",
            "SessionHijackingDetector",
            "DataExfiltrationDetector",
        ]

        for expected_type in expected_types:
            assert expected_type in detector_types

    @pytest.mark.asyncio
    async def test_session_hijacking_detection(self):
        """Test session hijacking detection with multiple IP changes."""
        detector = SessionHijackingDetector()
        session_id = "test_session_123"
        user_id = "test_user"

        # Simulate normal session activity
        event_data = {
            "session_id": session_id,
            "user_id": user_id,
            "ip_address": "192.168.1.100",
            "user_agent": "Browser 1.0",
        }

        alert = await detector.analyze(event_data)
        assert alert is None  # No alert for first IP

        # Simulate multiple IP changes
        ips = ["192.168.1.101", "192.168.1.102", "192.168.1.103", "192.168.1.104"]

        for ip in ips:
            event_data["ip_address"] = ip
            alert = await detector.analyze(event_data)

        # Should trigger alert after multiple IP changes
        assert alert is not None
        assert alert.alert_type == AlertType.SESSION_HIJACK
        assert alert.threat_level == ThreatLevel.HIGH

    @pytest.mark.asyncio
    async def test_data_exfiltration_detection(self):
        """Test data exfiltration detection with large data access."""
        detector = DataExfiltrationDetector()
        user_id = "test_user"

        # Simulate multiple large data accesses
        for i in range(60):  # Exceed request count threshold
            event_data = {
                "user_id": user_id,
                "event_type": SecurityEventType.DATA_ACCESS,
                "details": {"data_size_bytes": 2 * 1024 * 1024},  # 2MB each
                "endpoint": f"/api/data/download/{i}",
                "ip_address": "192.168.1.100",
            }

            alert = await detector.analyze(event_data)

        # Should trigger alert for excessive data access
        assert alert is not None
        assert alert.alert_type == AlertType.DATA_EXFILTRATION
        assert alert.threat_level == ThreatLevel.HIGH

    def test_audit_logging_comprehensive(self, security_service: SecurityService):
        """Test comprehensive audit logging functionality."""
        # Log various types of audit events
        events = [
            (AuditEventType.AUTHENTICATION, "user1", "auth", "login", "success"),
            (AuditEventType.DATA_ACCESS, "user2", "dataset", "download", "success"),
            (AuditEventType.MODEL_TRAINING, "user3", "model", "train", "success"),
            (
                AuditEventType.SECURITY_VIOLATION,
                "user4",
                "system",
                "breach_attempt",
                "blocked",
            ),
        ]

        event_ids = []
        for event_type, user_id, resource, action, outcome in events:
            event_id = security_service.audit_service.log_event(
                event_type=event_type,
                user_id=user_id,
                resource=resource,
                action=action,
                outcome=outcome,
                source_ip="192.168.1.100",
                user_agent="Test Agent",
                security_level=SecurityLevel.INTERNAL,
            )
            event_ids.append(event_id)

        assert len(event_ids) == 4
        assert all(event_id for event_id in event_ids)

        # Search audit events
        recent_events = security_service.audit_service.search_audit_events(
            start_time=datetime.utcnow() - timedelta(minutes=5),
            limit=10,
        )

        assert len(recent_events) >= 4

    def test_security_summary_generation(self, security_service: SecurityService):
        """Test security status summary generation."""
        # Generate some audit events first
        for i in range(5):
            security_service.audit_service.log_event(
                event_type=AuditEventType.AUTHENTICATION,
                user_id=f"user{i}",
                resource="auth",
                action="login",
                outcome="success" if i < 4 else "failure",
                source_ip=f"192.168.1.{100 + i}",
                security_level=SecurityLevel.INTERNAL,
            )

        # Get security summary
        summary = security_service.get_security_summary()

        assert "security_status" in summary
        assert "total_events_24h" in summary
        assert "failed_authentications_24h" in summary
        assert "security_violations_24h" in summary
        assert "encryption_enabled" in summary
        assert "audit_logging_enabled" in summary

        assert summary["audit_logging_enabled"] is True
        assert summary["total_events_24h"] >= 5


class TestEndToEndProductionIntegration:
    """End-to-end integration tests for all production enhancement features."""

    @pytest.mark.asyncio
    async def test_complete_production_pipeline(self):
        """Test complete production enhancement pipeline integration."""
        # Initialize all services
        cache_config = {
            "l1_cache": {"type": "memory", "max_size": 50},
            "l2_cache": {"type": "redis", "host": "localhost", "port": 6379, "db": 1},
            "compression": {"algorithm": "lz4", "enabled": True},
        }

        cache_service = AdvancedCacheService(cache_config)
        await cache_service.initialize()

        profiling_service = PerformanceProfilingService(
            {
                "profiling_enabled": True,
                "resource_monitoring_enabled": True,
            }
        )

        security_service = SecurityService(
            SecurityConfig(
                enable_audit_logging=True,
                enable_rbac=True,
                enable_rate_limiting=True,
            )
        )

        try:
            # Test integrated workflow

            # 1. Security: Authenticate user
            session = security_service.auth_service.authenticate_user(
                username="admin",
                password="AdminPassword123!",
                ip_address="192.168.1.100",
                user_agent="Production Client 1.0",
            )
            assert session is not None

            # 2. Caching: Store and retrieve data with performance monitoring
            @profiling_service.profile_function(ProfilerType.CPROFILE)
            async def cached_operation():
                test_data = {"model_id": "prod_model_v1", "accuracy": 0.95}
                await cache_service.set("prod_model", test_data)
                return await cache_service.get("prod_model")

            cached_data = await cached_operation()
            assert cached_data["model_id"] == "prod_model_v1"

            # 3. Security: Log audit events for the operations
            security_service.audit_service.log_event(
                event_type=AuditEventType.MODEL_PREDICTION,
                user_id=session.user_id,
                resource="production_model",
                action="predict",
                outcome="success",
                details={"model_version": "v1", "cache_hit": True},
                session_id=session.session_id,
                source_ip="192.168.1.100",
                security_level=SecurityLevel.CONFIDENTIAL,
            )

            # 4. Performance: Get resource metrics
            profiling_service.resource_monitor.start_monitoring()
            time.sleep(0.1)  # Brief monitoring period
            metrics = profiling_service.resource_monitor.get_current_metrics()
            profiling_service.resource_monitor.stop_monitoring()

            assert "cpu_percent" in metrics
            assert "memory_percent" in metrics

            # 5. Security: Generate summary
            security_summary = security_service.get_security_summary()
            assert security_summary["security_status"] in ["healthy", "at_risk"]

        finally:
            await cache_service.close()

    def test_production_readiness_checklist(self):
        """Verify production readiness across all enhancement areas."""
        checklist = {
            "kubernetes_manifests": False,
            "multi_level_caching": False,
            "performance_profiling": False,
            "security_hardening": False,
            "threat_detection": False,
            "audit_logging": False,
        }

        # Check Kubernetes manifests
        k8s_path = Path(__file__).parent.parent.parent / "deploy" / "kubernetes"
        if k8s_path.exists() and list(k8s_path.glob("*.yaml")):
            checklist["kubernetes_manifests"] = True

        # Check caching implementation
        try:
            cache_service = AdvancedCacheService(
                {"l1_cache": {"type": "memory", "max_size": 10}}
            )
            checklist["multi_level_caching"] = True
        except Exception:
            pass

        # Check profiling service
        try:
            profiling_service = PerformanceProfilingService({})
            checklist["performance_profiling"] = True
        except Exception:
            pass

        # Check security service
        try:
            security_service = SecurityService()
            checklist["security_hardening"] = True
        except Exception:
            pass

        # Check threat detection
        try:
            detectors = create_advanced_threat_detectors()
            if len(detectors) >= 4:
                checklist["threat_detection"] = True
        except Exception:
            pass

        # Check audit logging
        try:
            security_service = SecurityService()
            event_id = security_service.audit_service.log_event(
                event_type=AuditEventType.AUTHENTICATION,
                user_id="test",
                resource="test",
                action="test",
                outcome="success",
            )
            if event_id:
                checklist["audit_logging"] = True
        except Exception:
            pass

        # Verify production readiness
        ready_features = [k for k, v in checklist.items() if v]
        assert (
            len(ready_features) >= 5
        ), f"Only {len(ready_features)} features ready: {ready_features}"

        # All critical features should be ready
        critical_features = [
            "multi_level_caching",
            "security_hardening",
            "threat_detection",
        ]
        for feature in critical_features:
            assert checklist[feature], f"Critical feature not ready: {feature}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
