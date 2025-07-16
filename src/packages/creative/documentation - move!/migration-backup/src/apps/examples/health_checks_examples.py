"""Health checks examples and usage patterns for Pynomaly."""

import asyncio

from pynomaly.infrastructure.monitoring import (
    ComponentHealthConfig,
    ComponentType,
    HealthCheckCategory,
    HealthCheckResult,
    HealthCheckSchedule,
    HealthCheckStatus,
    get_comprehensive_health_manager,
    get_health_checker,
    liveness_probe,
    readiness_probe,
    register_health_check,
)


# Example 1: Custom health check function
async def check_custom_service() -> HealthCheckResult:
    """Example custom health check function."""
    try:
        # Simulate custom service check
        await asyncio.sleep(0.1)  # Simulate check time

        # Mock service status
        service_healthy = True  # In real implementation, check actual service

        if service_healthy:
            status = HealthCheckStatus.HEALTHY
            message = "Custom service operational"
        else:
            status = HealthCheckStatus.UNHEALTHY
            message = "Custom service unavailable"

        return HealthCheckResult(
            component="custom_service",
            component_type=ComponentType.EXTERNAL_SERVICE,
            status=status,
            message=message,
            details={
                "service_version": "1.2.3",
                "last_check": "2024-01-01T12:00:00Z",
                "endpoint": "https://api.example.com/health",
            },
        )

    except Exception as e:
        return HealthCheckResult(
            component="custom_service",
            component_type=ComponentType.EXTERNAL_SERVICE,
            status=HealthCheckStatus.UNHEALTHY,
            message=f"Custom service check failed: {str(e)}",
            details={"error": str(e)},
        )


# Example 2: Database-specific health check
async def check_database_performance() -> HealthCheckResult:
    """Example database performance health check."""
    try:
        # Simulate database performance check
        import time

        start_time = time.time()

        # Simulate database query
        await asyncio.sleep(0.05)  # Simulate query time

        query_time = (time.time() - start_time) * 1000

        # Mock database metrics
        db_metrics = {
            "active_connections": 15,
            "max_connections": 100,
            "query_time_ms": query_time,
            "slow_queries": 2,
            "connection_pool_utilization": 0.15,
        }

        # Health assessment
        if query_time > 1000:  # Slow queries
            status = HealthCheckStatus.DEGRADED
            message = f"Database queries slow: {query_time:.1f}ms"
        elif db_metrics["connection_pool_utilization"] > 0.8:
            status = HealthCheckStatus.DEGRADED
            message = "High database connection pool utilization"
        else:
            status = HealthCheckStatus.HEALTHY
            message = "Database performance normal"

        return HealthCheckResult(
            component="database_performance",
            component_type=ComponentType.DATABASE,
            status=status,
            message=message,
            details=db_metrics,
            response_time_ms=query_time,
        )

    except Exception as e:
        return HealthCheckResult(
            component="database_performance",
            component_type=ComponentType.DATABASE,
            status=HealthCheckStatus.UNHEALTHY,
            message=f"Database performance check failed: {str(e)}",
            details={"error": str(e)},
        )


# Example 3: External API health check
async def check_external_api() -> HealthCheckResult:
    """Example external API health check."""
    try:
        import time

        start_time = time.time()

        # Simulate API call
        await asyncio.sleep(0.2)  # Simulate API response time

        response_time = (time.time() - start_time) * 1000

        # Mock API response
        api_status_code = 200  # In real implementation, make actual HTTP request

        if api_status_code == 200:
            status = HealthCheckStatus.HEALTHY
            message = "External API responding normally"
        elif 500 <= api_status_code < 600:
            status = HealthCheckStatus.UNHEALTHY
            message = f"External API server error: {api_status_code}"
        else:
            status = HealthCheckStatus.DEGRADED
            message = f"External API returned status: {api_status_code}"

        return HealthCheckResult(
            component="external_api",
            component_type=ComponentType.EXTERNAL_SERVICE,
            status=status,
            message=message,
            details={
                "api_endpoint": "https://api.external-service.com/health",
                "status_code": api_status_code,
                "response_time_ms": response_time,
                "api_version": "v2.1",
            },
            response_time_ms=response_time,
        )

    except Exception as e:
        return HealthCheckResult(
            component="external_api",
            component_type=ComponentType.EXTERNAL_SERVICE,
            status=HealthCheckStatus.UNHEALTHY,
            message=f"External API check failed: {str(e)}",
            details={"error": str(e)},
        )


# Example usage functions
async def basic_health_checks_example():
    """Example of basic health checks usage."""
    print("=== Basic Health Checks Example ===\n")

    # Get the health checker
    health_checker = get_health_checker()

    # Register custom health checks
    register_health_check("custom_service", check_custom_service)
    register_health_check("database_performance", check_database_performance)
    register_health_check("external_api", check_external_api)

    # Run individual health checks
    print("1. Individual Health Checks:")
    for check_name in ["custom_service", "database_performance", "external_api"]:
        result = await health_checker.check_component(check_name)
        print(f"   {check_name}: {result.status.value} - {result.message}")
        if result.response_time_ms:
            print(f"     Response time: {result.response_time_ms:.1f}ms")

    print()

    # Run all health checks
    print("2. All Health Checks:")
    system_health = await health_checker.get_system_health(version="1.0.0")
    print(f"   Overall Status: {system_health.status.value}")
    print(f"   Message: {system_health.message}")
    print(f"   Total Checks: {len(system_health.checks)}")
    print(f"   Uptime: {system_health.uptime_seconds:.1f} seconds")

    print()

    # Show detailed results
    print("3. Detailed Health Check Results:")
    for check in system_health.checks:
        print(f"   {check.component} ({check.component_type.value}):")
        print(f"     Status: {check.status.value}")
        print(f"     Message: {check.message}")
        if check.details:
            print(f"     Details: {list(check.details.keys())}")

    print("\n=== Basic Health Checks Complete ===\n")


async def comprehensive_health_manager_example():
    """Example of comprehensive health manager usage."""
    print("=== Comprehensive Health Manager Example ===\n")

    # Get comprehensive health manager
    health_manager = get_comprehensive_health_manager(auto_start=False)

    # Register additional custom health check
    health_manager.register_component_check(
        ComponentHealthConfig(
            name="custom_monitoring_service",
            component_type=ComponentType.EXTERNAL_SERVICE,
            check_function=check_external_api,
            schedule=HealthCheckSchedule(
                interval_seconds=30,
                timeout_seconds=5.0,
                category=HealthCheckCategory.MONITORING,
            ),
            critical_for_readiness=False,
        )
    )

    # Get comprehensive health report
    print("1. Comprehensive Health Report:")
    health_report = await health_manager.get_comprehensive_health_report()

    print(f"   Current Status: {health_report['current_health']['status']}")
    print(f"   Health Trend: {health_report['health_trend']}")
    print(
        f"   Total Components: {health_report['monitoring_status']['total_components']}"
    )
    print(
        f"   Critical Components: {health_report['monitoring_status']['critical_components']}"
    )

    print("\n   Component Categories:")
    for category, count in health_report["category_summary"].items():
        print(f"     {category}: {count} components")

    print()

    # Get readiness status
    print("2. Application Readiness Status:")
    readiness = await health_manager.get_readiness_status()
    print(f"   Ready: {readiness['ready']}")
    print("   Critical Components Status:")
    for component, status in readiness["critical_components"].items():
        print(f"     {component}: {status['status']} ({status['message']})")

    print()

    # Show monitoring configuration
    print("3. Component Monitoring Configuration:")
    for name, config in health_report["component_configurations"].items():
        print(f"   {name}:")
        print(f"     Type: {config['type']}")
        print(f"     Category: {config['category']}")
        print(f"     Check Interval: {config['interval_seconds']}s")
        print(f"     Critical for Readiness: {config['critical_for_readiness']}")
        if config["dependencies"]:
            print(f"     Dependencies: {config['dependencies']}")

    print("\n=== Comprehensive Health Manager Complete ===\n")


async def kubernetes_probes_example():
    """Example of Kubernetes liveness and readiness probes."""
    print("=== Kubernetes Probes Example ===\n")

    # Test liveness probe
    print("1. Liveness Probe:")
    liveness_result = await liveness_probe()
    print(f"   Status: {liveness_result.status}")
    print(f"   Timestamp: {liveness_result.timestamp}")
    if liveness_result.details:
        print(
            f"   Uptime: {liveness_result.details.get('uptime_seconds', 'N/A')} seconds"
        )

    print()

    # Test readiness probe
    print("2. Readiness Probe:")
    readiness_result = await readiness_probe()
    print(f"   Status: {readiness_result.status}")
    print(f"   Timestamp: {readiness_result.timestamp}")
    if readiness_result.details:
        print("   Critical Components:")
        for component, status in readiness_result.details.items():
            if component not in ["timestamp"]:
                print(f"     {component}: {status}")

    print("\n=== Kubernetes Probes Complete ===\n")


async def monitoring_loop_simulation():
    """Simulate continuous health monitoring."""
    print("=== Monitoring Loop Simulation ===\n")

    # Get comprehensive health manager and start monitoring
    health_manager = get_comprehensive_health_manager(auto_start=True)

    print("Starting health monitoring simulation for 30 seconds...")

    # Monitor for a short period
    for i in range(6):  # 6 iterations, 5 seconds each
        await asyncio.sleep(5)

        # Get current health status
        system_health = await health_manager.health_checker.get_system_health()

        print(
            f"   Check #{i+1}: {system_health.status.value} "
            f"({len([c for c in system_health.checks if c.status.value == 'healthy'])}/"
            f"{len(system_health.checks)} healthy)"
        )

    # Stop monitoring
    await health_manager.stop_monitoring()

    print("Monitoring simulation complete.")
    print("\n=== Monitoring Loop Simulation Complete ===\n")


# Web framework integration example
def create_fastapi_health_endpoints():
    """Example of integrating health checks with FastAPI."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse

        app = FastAPI()

        @app.get("/health")
        async def health_endpoint():
            """Basic health endpoint."""
            health_checker = get_health_checker()
            system_health = await health_checker.get_system_health()

            return JSONResponse(
                content=system_health.to_dict(),
                status_code=200 if system_health.status.value == "healthy" else 503,
            )

        @app.get("/health/live")
        async def liveness_endpoint():
            """Kubernetes liveness probe endpoint."""
            probe_result = await liveness_probe()
            return JSONResponse(content=probe_result.dict())

        @app.get("/health/ready")
        async def readiness_endpoint():
            """Kubernetes readiness probe endpoint."""
            probe_result = await readiness_probe()
            status_code = 200 if probe_result.status == "ready" else 503
            return JSONResponse(content=probe_result.dict(), status_code=status_code)

        @app.get("/health/comprehensive")
        async def comprehensive_health_endpoint():
            """Comprehensive health report endpoint."""
            health_manager = get_comprehensive_health_manager()
            health_report = await health_manager.get_comprehensive_health_report()
            return JSONResponse(content=health_report)

        return app

    except ImportError:
        print("FastAPI not available - skipping web framework example")
        return None


# Main example function
async def main():
    """Main example function demonstrating health checks."""
    print("=== Pynomaly Health Checks Examples ===\n")

    # Run basic health checks example
    await basic_health_checks_example()

    # Run comprehensive health manager example
    await comprehensive_health_manager_example()

    # Run Kubernetes probes example
    await kubernetes_probes_example()

    # Run monitoring simulation (shorter version for example)
    print("=== Short Monitoring Simulation ===\n")
    health_manager = get_comprehensive_health_manager(auto_start=True)

    # Quick monitoring test
    await asyncio.sleep(2)
    system_health = await health_manager.health_checker.get_system_health()
    print(f"Monitoring Test: {system_health.status.value} - {system_health.message}")

    await health_manager.stop_monitoring()
    print("\n=== Examples Complete ===")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())

    # Create FastAPI app example
    fastapi_app = create_fastapi_health_endpoints()
    if fastapi_app:
        print("\nFastAPI app with health endpoints created successfully!")
        print("Available endpoints:")
        print("  GET /health - Basic health status")
        print("  GET /health/live - Liveness probe")
        print("  GET /health/ready - Readiness probe")
        print("  GET /health/comprehensive - Comprehensive health report")
