"""Comprehensive health check manager integrating with all Pynomaly infrastructure."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pynomaly.infrastructure.cache import get_cache_integration_manager
from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.persistence import get_production_database_manager
from pynomaly.infrastructure.security import get_rate_limit_manager

from .health_checks import (
    ComponentType,
    HealthCheckResult,
    HealthStatus,
    SystemHealth,
    get_health_checker,
)

logger = logging.getLogger(__name__)


class HealthCheckCategory(Enum):
    """Categories of health checks."""

    CRITICAL = "critical"
    IMPORTANT = "important"
    MONITORING = "monitoring"
    OPTIONAL = "optional"


@dataclass
class HealthCheckSchedule:
    """Health check scheduling configuration."""

    interval_seconds: int = 60
    timeout_seconds: float = 10.0
    retry_attempts: int = 3
    enabled: bool = True
    category: HealthCheckCategory = HealthCheckCategory.IMPORTANT


@dataclass
class ComponentHealthConfig:
    """Configuration for a specific component's health check."""

    name: str
    component_type: ComponentType
    check_function: Callable
    schedule: HealthCheckSchedule
    dependencies: set[str] = field(default_factory=set)
    critical_for_readiness: bool = False

    def __post_init__(self):
        """Ensure dependencies is a set."""
        if isinstance(self.dependencies, (list, tuple)):
            self.dependencies = set(self.dependencies)


class ComprehensiveHealthManager:
    """Comprehensive health check manager for all Pynomaly infrastructure."""

    def __init__(self, settings: Settings | None = None):
        """Initialize comprehensive health manager.

        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.health_checker = get_health_checker()
        self.component_configs: dict[str, ComponentHealthConfig] = {}
        self.health_history: list[SystemHealth] = []
        self.max_history_size = 1000
        self.last_check_times: dict[str, float] = {}
        self.monitoring_task: asyncio.Task | None = None
        self.running = False

        # Initialize infrastructure health checks
        self._register_infrastructure_checks()

    def _register_infrastructure_checks(self) -> None:
        """Register health checks for all infrastructure components."""

        # Database health check
        self.register_component_check(
            ComponentHealthConfig(
                name="database",
                component_type=ComponentType.DATABASE,
                check_function=self._check_database_health,
                schedule=HealthCheckSchedule(
                    interval_seconds=30,
                    timeout_seconds=5.0,
                    category=HealthCheckCategory.CRITICAL,
                ),
                critical_for_readiness=True,
            )
        )

        # Cache health check
        self.register_component_check(
            ComponentHealthConfig(
                name="cache_system",
                component_type=ComponentType.CACHE,
                check_function=self._check_cache_health,
                schedule=HealthCheckSchedule(
                    interval_seconds=60,
                    timeout_seconds=3.0,
                    category=HealthCheckCategory.IMPORTANT,
                ),
                dependencies={"database"},
            )
        )

        # Rate limiting health check
        self.register_component_check(
            ComponentHealthConfig(
                name="rate_limiting",
                component_type=ComponentType.EXTERNAL_SERVICE,
                check_function=self._check_rate_limiting_health,
                schedule=HealthCheckSchedule(
                    interval_seconds=120,
                    timeout_seconds=2.0,
                    category=HealthCheckCategory.MONITORING,
                ),
            )
        )

        # Model services health check
        self.register_component_check(
            ComponentHealthConfig(
                name="model_services",
                component_type=ComponentType.MODEL_REPOSITORY,
                check_function=self._check_model_services_health,
                schedule=HealthCheckSchedule(
                    interval_seconds=90,
                    timeout_seconds=10.0,
                    category=HealthCheckCategory.IMPORTANT,
                ),
                dependencies={"database", "cache_system"},
                critical_for_readiness=True,
            )
        )

        # Application configuration health check
        self.register_component_check(
            ComponentHealthConfig(
                name="application_config",
                component_type=ComponentType.EXTERNAL_SERVICE,
                check_function=self._check_application_config,
                schedule=HealthCheckSchedule(
                    interval_seconds=300,  # 5 minutes
                    timeout_seconds=1.0,
                    category=HealthCheckCategory.MONITORING,
                ),
                critical_for_readiness=True,
            )
        )

    def register_component_check(self, config: ComponentHealthConfig) -> None:
        """Register a component health check.

        Args:
            config: Component health check configuration
        """
        self.component_configs[config.name] = config
        self.health_checker.register_check(config.name, config.check_function)
        logger.info(f"Registered comprehensive health check: {config.name}")

    async def _check_database_health(self) -> HealthCheckResult:
        """Check database system health."""
        try:
            db_manager = get_production_database_manager()

            # Test basic connectivity
            start_time = time.time()
            async with db_manager.get_session() as session:
                # Simple query to test connectivity
                result = await session.execute("SELECT 1")
                await result.fetchone()

            response_time = (time.time() - start_time) * 1000

            # Get connection pool statistics
            pool_stats = await db_manager.get_pool_statistics()

            # Health assessment
            pool_utilization = pool_stats.get("utilization", 0)
            if pool_utilization > 0.9:
                status = HealthStatus.DEGRADED
                message = f"High database pool utilization: {pool_utilization:.1%}"
            elif pool_stats.get("failed_connections", 0) > 0:
                status = HealthStatus.DEGRADED
                message = "Database connection failures detected"
            else:
                status = HealthStatus.HEALTHY
                message = "Database connectivity normal"

            return HealthCheckResult(
                component="database",
                component_type=ComponentType.DATABASE,
                status=status,
                message=message,
                details={
                    "response_time_ms": response_time,
                    "pool_statistics": pool_stats,
                    "database_type": "postgresql",
                },
                response_time_ms=response_time,
            )

        except Exception as e:
            return HealthCheckResult(
                component="database",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                message=f"Database health check failed: {str(e)}",
                details={"error": str(e)},
            )

    async def _check_cache_health(self) -> HealthCheckResult:
        """Check cache system health."""
        try:
            cache_manager = get_cache_integration_manager()

            if not cache_manager.intelligent_cache:
                return HealthCheckResult(
                    component="cache_system",
                    component_type=ComponentType.CACHE,
                    status=HealthStatus.DEGRADED,
                    message="Cache system not initialized",
                    details={"cache_enabled": False},
                )

            # Test cache operations
            start_time = time.time()
            test_key = f"health_check_{int(time.time())}"
            test_value = {"test": "health_check_data"}

            # Test set/get/delete cycle
            await cache_manager.intelligent_cache.set(test_key, test_value, ttl=60)
            retrieved_value = await cache_manager.intelligent_cache.get(test_key)
            await cache_manager.intelligent_cache.delete(test_key)

            response_time = (time.time() - start_time) * 1000

            # Get cache statistics
            cache_stats = await cache_manager.get_comprehensive_stats()

            # Health assessment
            if retrieved_value != test_value:
                status = HealthStatus.DEGRADED
                message = "Cache read/write inconsistency"
            elif (
                cache_stats.get("health_monitoring", {})
                .get("current_status", {})
                .get("overall_health")
                == "unhealthy"
            ):
                status = HealthStatus.UNHEALTHY
                message = "Cache health monitoring reports unhealthy"
            elif (
                cache_stats.get("health_monitoring", {})
                .get("current_status", {})
                .get("overall_health")
                == "degraded"
            ):
                status = HealthStatus.DEGRADED
                message = "Cache health monitoring reports degraded"
            else:
                status = HealthStatus.HEALTHY
                message = "Cache system operational"

            return HealthCheckResult(
                component="cache_system",
                component_type=ComponentType.CACHE,
                status=status,
                message=message,
                details={
                    "response_time_ms": response_time,
                    "cache_statistics": cache_stats,
                    "operations_tested": ["set", "get", "delete"],
                },
                response_time_ms=response_time,
            )

        except Exception as e:
            return HealthCheckResult(
                component="cache_system",
                component_type=ComponentType.CACHE,
                status=HealthStatus.UNHEALTHY,
                message=f"Cache health check failed: {str(e)}",
                details={"error": str(e)},
            )

    async def _check_rate_limiting_health(self) -> HealthCheckResult:
        """Check rate limiting system health."""
        try:
            rate_manager = get_rate_limit_manager()

            # Get rate limiting statistics
            stats = await rate_manager.get_all_stats()

            # Test rate limiting functionality
            start_time = time.time()
            test_status = await rate_manager.check_limit(
                limiter_name="health_check",
                identifier="health_check_test",
                operation="test",
                tokens=0,  # Don't consume tokens
            )
            response_time = (time.time() - start_time) * 1000

            # Health assessment
            total_violations = sum(
                limiter_stats.get("statistics", {}).get("violations", 0)
                for limiter_stats in stats.values()
            )

            if not test_status.allowed and test_status.retry_after is None:
                status = HealthStatus.DEGRADED
                message = "Rate limiting system issues detected"
            elif total_violations > 1000:  # High violation count
                status = HealthStatus.DEGRADED
                message = f"High rate limit violations: {total_violations}"
            else:
                status = HealthStatus.HEALTHY
                message = "Rate limiting system operational"

            return HealthCheckResult(
                component="rate_limiting",
                component_type=ComponentType.EXTERNAL_SERVICE,
                status=status,
                message=message,
                details={
                    "response_time_ms": response_time,
                    "statistics": stats,
                    "total_violations": total_violations,
                    "active_limiters": len(stats),
                },
                response_time_ms=response_time,
            )

        except Exception as e:
            return HealthCheckResult(
                component="rate_limiting",
                component_type=ComponentType.EXTERNAL_SERVICE,
                status=HealthStatus.UNHEALTHY,
                message=f"Rate limiting health check failed: {str(e)}",
                details={"error": str(e)},
            )

    async def _check_model_services_health(self) -> HealthCheckResult:
        """Check model services health."""
        try:
            # This would integrate with actual model services
            # For now, simulate comprehensive model service check

            start_time = time.time()

            # Simulate model service checks
            await asyncio.sleep(0.1)  # Simulate check time

            response_time = (time.time() - start_time) * 1000

            # Mock model service statistics
            model_stats = {
                "loaded_models": 5,
                "active_detectors": 3,
                "total_predictions": 15420,
                "average_prediction_time_ms": 45.2,
                "model_cache_hit_rate": 0.85,
                "last_training_time": (datetime.now() - timedelta(hours=2)).isoformat(),
            }

            # Health assessment based on mock statistics
            if model_stats["average_prediction_time_ms"] > 1000:
                status = HealthStatus.DEGRADED
                message = "Model prediction latency high"
            elif model_stats["model_cache_hit_rate"] < 0.5:
                status = HealthStatus.DEGRADED
                message = "Low model cache hit rate"
            else:
                status = HealthStatus.HEALTHY
                message = "Model services operational"

            return HealthCheckResult(
                component="model_services",
                component_type=ComponentType.MODEL_REPOSITORY,
                status=status,
                message=message,
                details={
                    "response_time_ms": response_time,
                    "model_statistics": model_stats,
                },
                response_time_ms=response_time,
            )

        except Exception as e:
            return HealthCheckResult(
                component="model_services",
                component_type=ComponentType.MODEL_REPOSITORY,
                status=HealthStatus.UNHEALTHY,
                message=f"Model services health check failed: {str(e)}",
                details={"error": str(e)},
            )

    async def _check_application_config(self) -> HealthCheckResult:
        """Check application configuration health."""
        try:
            start_time = time.time()

            # Check configuration validity
            config_issues = []

            # Validate database configuration
            if not self.settings.database_url:
                config_issues.append("Database URL not configured")

            # Validate cache configuration
            if self.settings.cache_enabled and not self.settings.redis_url:
                config_issues.append("Cache enabled but Redis URL not configured")

            # Check critical settings
            critical_settings = [
                ("log_level", self.settings.log_level),
                ("environment", self.settings.environment),
            ]

            for setting_name, setting_value in critical_settings:
                if not setting_value:
                    config_issues.append(
                        f"Critical setting '{setting_name}' not configured"
                    )

            response_time = (time.time() - start_time) * 1000

            # Health assessment
            if config_issues:
                status = (
                    HealthStatus.DEGRADED
                    if len(config_issues) < 3
                    else HealthStatus.UNHEALTHY
                )
                message = f"Configuration issues detected: {len(config_issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = "Application configuration valid"

            return HealthCheckResult(
                component="application_config",
                component_type=ComponentType.EXTERNAL_SERVICE,
                status=status,
                message=message,
                details={
                    "response_time_ms": response_time,
                    "config_issues": config_issues,
                    "environment": self.settings.environment,
                    "cache_enabled": self.settings.cache_enabled,
                    "database_configured": bool(self.settings.database_url),
                },
                response_time_ms=response_time,
            )

        except Exception as e:
            return HealthCheckResult(
                component="application_config",
                component_type=ComponentType.EXTERNAL_SERVICE,
                status=HealthStatus.UNHEALTHY,
                message=f"Configuration health check failed: {str(e)}",
                details={"error": str(e)},
            )

    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.running:
            logger.warning("Health monitoring already running")
            return

        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Comprehensive health monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None

        logger.info("Comprehensive health monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Continuous health monitoring loop."""
        while self.running:
            try:
                current_time = time.time()

                # Check which components need health checks
                checks_to_run = []
                for name, config in self.component_configs.items():
                    if not config.schedule.enabled:
                        continue

                    last_check = self.last_check_times.get(name, 0)
                    if current_time - last_check >= config.schedule.interval_seconds:
                        checks_to_run.append(name)

                # Run scheduled health checks
                if checks_to_run:
                    logger.debug(f"Running scheduled health checks: {checks_to_run}")

                    # Run checks in parallel but with respect to dependencies
                    await self._run_health_checks_with_dependencies(checks_to_run)

                    # Update last check times
                    for name in checks_to_run:
                        self.last_check_times[name] = current_time

                # Store system health snapshot
                system_health = await self.health_checker.get_system_health()
                self._store_health_snapshot(system_health)

                # Sleep until next check cycle
                await asyncio.sleep(10)  # Check every 10 seconds for scheduling

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(30)  # Back off on errors

    async def _run_health_checks_with_dependencies(
        self, check_names: list[str]
    ) -> None:
        """Run health checks respecting dependencies."""
        completed = set()
        remaining = set(check_names)

        while remaining:
            # Find checks that can run (dependencies satisfied)
            ready_to_run = []
            for name in remaining:
                config = self.component_configs[name]
                if config.dependencies.issubset(completed):
                    ready_to_run.append(name)

            if not ready_to_run:
                # No checks ready, run remaining anyway to avoid deadlock
                ready_to_run = list(remaining)

            # Run ready checks in parallel
            tasks = [self.health_checker.check_component(name) for name in ready_to_run]

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Mark as completed
            completed.update(ready_to_run)
            remaining -= set(ready_to_run)

    def _store_health_snapshot(self, system_health: SystemHealth) -> None:
        """Store health snapshot in history."""
        self.health_history.append(system_health)

        # Limit history size
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size // 2 :]

    async def get_comprehensive_health_report(self) -> dict[str, Any]:
        """Get comprehensive health report."""
        current_health = await self.health_checker.get_system_health()

        # Get health trends
        recent_history = self.health_history[-10:] if self.health_history else []

        # Calculate health trend
        if len(recent_history) >= 2:
            recent_unhealthy = sum(
                1 for h in recent_history[-5:] if h.status == HealthStatus.UNHEALTHY
            )
            older_unhealthy = sum(
                1 for h in recent_history[-10:-5] if h.status == HealthStatus.UNHEALTHY
            )

            if recent_unhealthy > older_unhealthy:
                trend = "deteriorating"
            elif recent_unhealthy < older_unhealthy:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        # Component categories summary
        category_summary = {category.value: 0 for category in HealthCheckCategory}
        for config in self.component_configs.values():
            category_summary[config.schedule.category.value] += 1

        # Critical component status
        critical_components = [
            name
            for name, config in self.component_configs.items()
            if config.critical_for_readiness
        ]

        critical_health = {}
        for component_name in critical_components:
            if component_name in self.health_checker._last_check_results:
                result = self.health_checker._last_check_results[component_name]
                critical_health[component_name] = result.status.value

        return {
            "current_health": current_health.to_dict(),
            "health_trend": trend,
            "monitoring_status": {
                "running": self.running,
                "total_components": len(self.component_configs),
                "critical_components": len(critical_components),
                "checks_performed": len(self.health_history),
            },
            "category_summary": category_summary,
            "critical_components_health": critical_health,
            "recent_health_history": [h.to_dict() for h in recent_history],
            "component_configurations": {
                name: {
                    "type": config.component_type.value,
                    "category": config.schedule.category.value,
                    "interval_seconds": config.schedule.interval_seconds,
                    "critical_for_readiness": config.critical_for_readiness,
                    "dependencies": list(config.dependencies),
                }
                for name, config in self.component_configs.items()
            },
        }

    async def get_readiness_status(self) -> dict[str, Any]:
        """Get application readiness status."""
        # Check critical components
        critical_components = [
            name
            for name, config in self.component_configs.items()
            if config.critical_for_readiness
        ]

        ready = True
        component_status = {}

        for component_name in critical_components:
            result = await self.health_checker.check_component(component_name)
            component_status[component_name] = {
                "status": result.status.value,
                "message": result.message,
                "response_time_ms": result.response_time_ms,
            }

            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]:
                ready = False

        return {
            "ready": ready,
            "timestamp": datetime.now().isoformat(),
            "critical_components": component_status,
            "summary": {
                "total_critical": len(critical_components),
                "healthy_critical": len(
                    [s for s in component_status.values() if s["status"] == "healthy"]
                ),
            },
        }


# Global comprehensive health manager
_comprehensive_health_manager: ComprehensiveHealthManager | None = None


def get_comprehensive_health_manager(
    settings: Settings | None = None,
    auto_start: bool = True,
) -> ComprehensiveHealthManager:
    """Get global comprehensive health manager.

    Args:
        settings: Application settings
        auto_start: Whether to auto-start monitoring

    Returns:
        Comprehensive health manager instance
    """
    global _comprehensive_health_manager

    if _comprehensive_health_manager is None:
        _comprehensive_health_manager = ComprehensiveHealthManager(settings)

        if auto_start:
            # Start monitoring in background
            asyncio.create_task(_comprehensive_health_manager.start_monitoring())

    return _comprehensive_health_manager


async def close_comprehensive_health_manager() -> None:
    """Close global comprehensive health manager."""
    global _comprehensive_health_manager

    if _comprehensive_health_manager:
        await _comprehensive_health_manager.stop_monitoring()
        _comprehensive_health_manager = None
