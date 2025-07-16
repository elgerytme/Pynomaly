"""Production readiness examples and usage patterns for Pynomaly."""

import asyncio

from pynomaly.infrastructure.production import (
    Environment,
    ProductionConfig,
    ShutdownPhase,
    ShutdownTask,
    StartupPhase,
    StartupTask,
    get_deployment_validator,
    get_production_config,
    get_shutdown_manager,
    get_startup_manager,
    validate_deployment_readiness,
    validate_production_config,
)


# Example 1: Production Configuration
def production_config_example():
    """Example of production configuration setup."""
    print("=== Production Configuration Example ===\n")

    # Load configuration for different environments
    for env in [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]:
        print(f"1. {env.value.title()} Configuration:")

        config = ProductionConfig.for_environment(env)

        print(f"   Environment: {config.environment.name.value}")
        print(f"   Debug Mode: {config.environment.debug_mode}")
        print(f"   Log Level: {config.environment.log_level}")
        print(f"   Security Level: {config.security.level.value}")
        print(f"   Worker Processes: {config.performance.worker_processes}")
        print(f"   Replicas: {config.deployment.replicas}")
        print(
            f"   Health Check Interval: {config.monitoring.health_check_interval_seconds}s"
        )
        print()

    # Validate production configuration
    print("2. Production Configuration Validation:")
    prod_config = ProductionConfig.for_environment(Environment.PRODUCTION)
    validation_result = validate_production_config(prod_config, raise_on_issues=False)

    print(f"   Valid: {validation_result['valid']}")
    print(f"   Issues: {validation_result['issues']}")
    print(f"   Warnings: {validation_result['warnings']}")
    print(f"   Security Level: {validation_result['security_level']}")
    print()

    # Save configuration to file
    print("3. Configuration Export:")
    prod_config.save_to_file("production_config.json")
    print("   Configuration saved to production_config.json")

    print("\n=== Production Configuration Complete ===\n")


# Example 2: Application Startup
async def application_startup_example():
    """Example of production application startup."""
    print("=== Application Startup Example ===\n")

    # Get startup manager
    startup_manager = get_startup_manager()

    # Register custom startup task
    async def custom_initialization():
        """Custom initialization task."""
        print("   Running custom initialization...")
        await asyncio.sleep(0.1)  # Simulate work
        print("   Custom initialization completed")

    custom_task = StartupTask(
        name="custom_initialization",
        phase=StartupPhase.SERVICES,
        task_function=custom_initialization,
        dependencies={"initialize_services"},
        timeout_seconds=10.0,
    )

    startup_manager.startup.register_task(custom_task)

    # Perform startup
    print("1. Starting Application Startup:")
    success = await startup_manager.start()

    print(f"   Startup Success: {success}")
    print(f"   Startup Complete: {startup_manager.startup_complete}")
    print(f"   Application Ready: {startup_manager.startup.is_ready}")
    print(f"   Startup Time: {startup_manager.startup.startup_time:.2f}s")
    print()

    # Get startup summary
    print("2. Startup Summary:")
    summary = startup_manager.startup.get_startup_summary()
    print(f"   Total Tasks: {summary['total_tasks']}")
    print(f"   Successful Tasks: {summary['successful_tasks']}")
    print(f"   Failed Tasks: {summary['failed_tasks']}")

    if summary["failed_task_names"]:
        print(f"   Failed Task Names: {summary['failed_task_names']}")

    print("\n   Task Results:")
    for task_result in summary["task_results"]:
        status = "✓" if task_result["success"] else "✗"
        print(
            f"     {status} {task_result['name']} ({task_result['phase']}) - {task_result['duration_seconds']:.2f}s"
        )

    print("\n=== Application Startup Complete ===\n")


# Example 3: Graceful Shutdown
async def graceful_shutdown_example():
    """Example of graceful shutdown."""
    print("=== Graceful Shutdown Example ===\n")

    # Get shutdown manager
    shutdown_manager = get_shutdown_manager()

    # Register custom shutdown hook
    async def custom_cleanup():
        """Custom cleanup task."""
        print("   Running custom cleanup...")
        await asyncio.sleep(0.1)  # Simulate cleanup work
        print("   Custom cleanup completed")

    shutdown_manager.register_shutdown_hook(custom_cleanup)

    # Register custom shutdown task
    async def save_application_state():
        """Save application state during shutdown."""
        print("   Saving application state...")
        await asyncio.sleep(0.1)  # Simulate state saving
        print("   Application state saved")

    shutdown_task = ShutdownTask(
        name="save_application_state",
        phase=ShutdownPhase.STOP_SERVICES,
        task_function=save_application_state,
        dependencies={"stop_background_tasks"},
        timeout_seconds=10.0,
    )

    shutdown_manager.shutdown_manager.register_task(shutdown_task)

    # Perform shutdown
    print("1. Starting Graceful Shutdown:")
    success = await shutdown_manager.shutdown()

    print(f"   Shutdown Success: {success}")
    print(
        f"   Shutdown Complete: {shutdown_manager.shutdown_manager.shutdown_complete}"
    )
    print(f"   Shutdown Time: {shutdown_manager.shutdown_manager.shutdown_time:.2f}s")
    print()

    # Get shutdown summary
    print("2. Shutdown Summary:")
    summary = shutdown_manager.shutdown_manager.get_shutdown_summary()
    print(f"   Total Tasks: {summary['total_tasks']}")
    print(f"   Successful Tasks: {summary['successful_tasks']}")
    print(f"   Failed Tasks: {summary['failed_tasks']}")

    print("\n   Task Results:")
    for task_result in summary["task_results"]:
        status = "✓" if task_result["success"] else "✗"
        print(
            f"     {status} {task_result['name']} ({task_result['phase']}) - {task_result['duration_seconds']:.2f}s"
        )

    print("\n=== Graceful Shutdown Complete ===\n")


# Example 4: Deployment Validation
async def deployment_validation_example():
    """Example of deployment validation."""
    print("=== Deployment Validation Example ===\n")

    # Get deployment validator
    validator = get_deployment_validator()

    # Run deployment validation for different environments
    for env in [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]:
        print(f"1. {env.value.title()} Deployment Validation:")

        validation_result = await validator.deployment_validator.validate_deployment(
            target_environment=env
        )

        print(f"   Deployment Ready: {validation_result['deployment_ready']}")
        print(f"   Total Checks: {validation_result['total_checks']}")
        print(f"   Passed Checks: {validation_result['passed_checks']}")
        print(f"   Failed Checks: {validation_result['failed_checks']}")
        print(f"   Execution Time: {validation_result['execution_time_seconds']:.2f}s")

        # Show severity counts
        severity_counts = validation_result["severity_counts"]
        print(
            f"   Severity Counts: INFO={severity_counts['info']}, "
            f"WARNING={severity_counts['warning']}, ERROR={severity_counts['error']}, "
            f"CRITICAL={severity_counts['critical']}"
        )

        # Show failed checks
        if validation_result["failed_check_names"]:
            print(
                f"   Failed Checks: {', '.join(validation_result['failed_check_names'])}"
            )

        print()

    # Run comprehensive production readiness validation
    print("2. Production Readiness Validation:")
    production_validation = await validate_deployment_readiness(
        target_environment=Environment.PRODUCTION
    )

    print(f"   Deployment Ready: {production_validation['deployment_ready']}")
    print(
        f"   Deployment Recommendation: {production_validation['deployment_recommendation']}"
    )

    analysis = production_validation["production_analysis"]
    print(f"   Readiness Level: {analysis['readiness_level']}")
    print(f"   Readiness Score: {analysis['readiness_score']:.1f}%")
    print(f"   Critical Issues: {analysis['critical_issues_count']}")

    print("\n   Production Recommendations:")
    for rec in analysis["production_recommendations"][:5]:  # Show first 5
        print(f"     • {rec}")

    print()

    # Show detailed validation results
    print("3. Detailed Validation Results:")
    for result in production_validation["results"][:10]:  # Show first 10 results
        status = "✓" if result["passed"] else "✗"
        severity = result["severity"].upper()
        print(
            f"     {status} {result['check_name']} ({severity}) - {result['message']}"
        )

    print("\n=== Deployment Validation Complete ===\n")


# Example 5: Production Application Lifecycle
async def production_lifecycle_example():
    """Example of complete production application lifecycle."""
    print("=== Production Application Lifecycle Example ===\n")

    # 1. Configuration
    print("1. Loading Production Configuration:")
    config = get_production_config(Environment.PRODUCTION)
    print(f"   Environment: {config.environment.name.value}")
    print(f"   Security Level: {config.security.level.value}")
    print(f"   Worker Processes: {config.performance.worker_processes}")
    print()

    # 2. Deployment Validation
    print("2. Validating Deployment Readiness:")
    validation = await validate_deployment_readiness(Environment.PRODUCTION)
    print(f"   Deployment Ready: {validation['deployment_ready']}")
    print(f"   Recommendation: {validation['deployment_recommendation']}")
    print()

    # 3. Application Startup
    print("3. Starting Application:")
    startup_manager = get_startup_manager()
    startup_success = await startup_manager.start()
    print(f"   Startup Success: {startup_success}")
    print(f"   Application Ready: {startup_manager.startup.is_ready}")
    print()

    # 4. Simulate application runtime
    print("4. Simulating Application Runtime:")
    print("   Application is running...")
    await asyncio.sleep(1.0)  # Simulate runtime
    print("   Received shutdown signal...")
    print()

    # 5. Graceful Shutdown
    print("5. Performing Graceful Shutdown:")
    shutdown_manager = get_shutdown_manager()
    shutdown_success = await shutdown_manager.shutdown()
    print(f"   Shutdown Success: {shutdown_success}")
    print(
        f"   Shutdown Complete: {shutdown_manager.shutdown_manager.shutdown_complete}"
    )
    print()

    print("=== Production Application Lifecycle Complete ===\n")


# Example 6: Configuration Management
def configuration_management_example():
    """Example of configuration management for different environments."""
    print("=== Configuration Management Example ===\n")

    # Create configurations for different environments
    configs = {}
    for env in Environment:
        configs[env.value] = ProductionConfig.for_environment(env)

    # Compare configurations
    print("1. Environment Configuration Comparison:")
    print("   Setting                    | Development | Staging     | Production")
    print("   ---------------------------|-------------|-------------|------------")
    print(
        f"   Debug Mode                 | {str(configs['development'].environment.debug_mode):11} | {str(configs['staging'].environment.debug_mode):11} | {str(configs['production'].environment.debug_mode):10}"
    )
    print(
        f"   Log Level                  | {configs['development'].environment.log_level:11} | {configs['staging'].environment.log_level:11} | {configs['production'].environment.log_level:10}"
    )
    print(
        f"   Security Level             | {configs['development'].security.level.value:11} | {configs['staging'].security.level.value:11} | {configs['production'].security.level.value:10}"
    )
    print(
        f"   Worker Processes           | {str(configs['development'].performance.worker_processes):11} | {str(configs['staging'].performance.worker_processes):11} | {str(configs['production'].performance.worker_processes):10}"
    )
    print(
        f"   Max DB Connections         | {str(configs['development'].performance.max_database_connections):11} | {str(configs['staging'].performance.max_database_connections):11} | {str(configs['production'].performance.max_database_connections):10}"
    )
    print(
        f"   Health Check Interval      | {str(configs['development'].monitoring.health_check_interval_seconds):11} | {str(configs['staging'].monitoring.health_check_interval_seconds):11} | {str(configs['production'].monitoring.health_check_interval_seconds):10}"
    )
    print(
        f"   Replicas                   | {str(configs['development'].deployment.replicas):11} | {str(configs['staging'].deployment.replicas):11} | {str(configs['production'].deployment.replicas):10}"
    )
    print(
        f"   Enable Autoscaling         | {str(configs['development'].deployment.enable_autoscaling):11} | {str(configs['staging'].deployment.enable_autoscaling):11} | {str(configs['production'].deployment.enable_autoscaling):10}"
    )
    print()

    # Export configurations
    print("2. Exporting Configurations:")
    for env_name, config in configs.items():
        filename = f"config_{env_name}.json"
        config.save_to_file(filename)
        print(f"   {env_name.title()} configuration saved to {filename}")

    print("\n=== Configuration Management Complete ===\n")


# FastAPI integration example
def create_production_fastapi_app():
    """Example of integrating production features with FastAPI."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI(title="Pynomaly Production API")

        # Add production endpoints
        @app.get("/production/config")
        async def get_production_config_endpoint():
            """Get production configuration."""
            config = get_production_config()
            return JSONResponse(content=config.to_dict())

        @app.get("/production/startup")
        async def get_startup_status():
            """Get startup status."""
            startup_manager = get_startup_manager()
            return JSONResponse(
                content={
                    "startup_complete": startup_manager.startup_complete,
                    "ready": startup_manager.startup.is_ready,
                    "startup_time": startup_manager.startup.startup_time,
                }
            )

        @app.post("/production/validate")
        async def validate_deployment():
            """Validate deployment readiness."""
            validation = await validate_deployment_readiness(Environment.PRODUCTION)
            return JSONResponse(content=validation)

        @app.post("/production/shutdown")
        async def initiate_shutdown():
            """Initiate graceful shutdown."""
            shutdown_manager = get_shutdown_manager()
            # In a real app, you'd trigger shutdown differently
            return JSONResponse(content={"message": "Shutdown initiated"})

        # Add lifespan management
        @app.on_event("startup")
        async def startup_event():
            """Handle application startup."""
            startup_manager = get_startup_manager()
            await startup_manager.start()

        @app.on_event("shutdown")
        async def shutdown_event():
            """Handle application shutdown."""
            shutdown_manager = get_shutdown_manager()
            await shutdown_manager.shutdown()

        return app

    except ImportError:
        print("FastAPI not available - skipping web framework example")
        return None


# Main example function
async def main():
    """Main example function demonstrating production readiness."""
    print("=== Pynomaly Production Readiness Examples ===\n")

    # Run configuration example
    production_config_example()

    # Run startup example
    await application_startup_example()

    # Run validation example
    await deployment_validation_example()

    # Run shutdown example
    await graceful_shutdown_example()

    # Run lifecycle example
    await production_lifecycle_example()

    # Run configuration management example
    configuration_management_example()

    print("=== Production Readiness Examples Complete ===")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())

    # Create FastAPI app example
    fastapi_app = create_production_fastapi_app()
    if fastapi_app:
        print("\nFastAPI production app created successfully!")
        print("Available endpoints:")
        print("  GET /production/config - Get production configuration")
        print("  GET /production/startup - Get startup status")
        print("  POST /production/validate - Validate deployment")
        print("  POST /production/shutdown - Initiate graceful shutdown")
