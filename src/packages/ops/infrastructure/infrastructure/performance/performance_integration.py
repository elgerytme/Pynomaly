"""
Performance Optimization Integration Module.

This module integrates performance optimization features into existing Pynomaly services,
providing seamless performance enhancements without breaking existing functionality.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from pynomaly.infrastructure.monitoring.performance_monitor import (
    PerformanceMonitor,
    create_performance_monitor,
)
from pynomaly.infrastructure.performance.advanced_benchmarking_service import (
    AdvancedBenchmarkConfig,
    AdvancedPerformanceBenchmarkingService,
)
from pynomaly.infrastructure.performance.optimization_engine import (
    PerformanceOptimizationEngine,
    create_optimization_engine,
)
from pynomaly.infrastructure.performance.performance_reporting import (
    PerformanceReporter,
    create_performance_reporter,
)

# Configure logging
logger = logging.getLogger(__name__)


class PerformanceIntegrationManager:
    """
    Manages integration of performance optimization features into existing services.

    This class provides a unified interface for enabling performance enhancements
    across the Pynomaly system without requiring major code changes.
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        enable_optimization: bool = True,
        enable_monitoring: bool = True,
        enable_reporting: bool = True,
    ):
        """Initialize performance integration manager."""
        self.storage_path = storage_path or Path("performance_integration")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Feature flags
        self.enable_optimization = enable_optimization
        self.enable_monitoring = enable_monitoring
        self.enable_reporting = enable_reporting

        # Core components
        self.optimization_engine: PerformanceOptimizationEngine | None = None
        self.performance_monitor: PerformanceMonitor | None = None
        self.performance_reporter: PerformanceReporter | None = None
        self.benchmarking_service: AdvancedPerformanceBenchmarkingService | None = None

        # Initialize components
        self._initialize_components()

        # Enhanced service registry
        self.enhanced_services: dict[str, Any] = {}

        logger.info("Performance integration manager initialized")

    def _initialize_components(self):
        """Initialize performance components based on feature flags."""
        if self.enable_optimization:
            self.optimization_engine = create_optimization_engine(
                cache_size_mb=512, storage_path=self.storage_path / "optimization"
            )
            logger.info("Optimization engine initialized")

        if self.enable_monitoring:
            self.performance_monitor = create_performance_monitor(
                storage_path=self.storage_path / "monitoring", enable_alerts=True
            )
            logger.info("Performance monitor initialized")

        if self.enable_reporting:
            self.performance_reporter = create_performance_reporter(
                storage_path=self.storage_path / "reports"
            )
            logger.info("Performance reporter initialized")

        # Always initialize benchmarking service
        self.benchmarking_service = AdvancedPerformanceBenchmarkingService(
            storage_path=self.storage_path / "benchmarks"
        )
        logger.info("Benchmarking service initialized")

    async def start_performance_systems(self):
        """Start all performance monitoring and optimization systems."""
        if self.performance_monitor:
            await self.performance_monitor.start_monitoring()

        logger.info("Performance systems started")

    async def stop_performance_systems(self):
        """Stop all performance monitoring and optimization systems."""
        if self.performance_monitor:
            await self.performance_monitor.stop_monitoring()

        if self.optimization_engine:
            self.optimization_engine.cleanup()

        logger.info("Performance systems stopped")

    def enhance_service(self, service_class: type, service_name: str = None) -> type:
        """
        Enhance a service class with performance optimization features.

        This method returns a new class that wraps the original service with
        performance optimizations while maintaining the same interface.
        """
        if not service_name:
            service_name = service_class.__name__

        # Create enhanced service class
        class EnhancedService(service_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._performance_manager = self
                self._original_service_name = service_name

                # Add performance monitoring to async methods
                if self._performance_manager.performance_monitor:
                    self._add_monitoring_to_methods()

                # Add optimization decorators
                if self._performance_manager.optimization_engine:
                    self._add_optimization_to_methods()

            def _add_monitoring_to_methods(self):
                """Add performance monitoring to service methods."""
                monitor = self._performance_manager.performance_monitor

                # Get all async methods
                for attr_name in dir(self):
                    attr = getattr(self, attr_name)
                    if (
                        asyncio.iscoroutinefunction(attr)
                        and not attr_name.startswith("_")
                        and attr_name not in ["start_monitoring", "stop_monitoring"]
                    ):
                        # Wrap with monitoring
                        original_method = attr

                        async def monitored_method(*args, **kwargs):
                            operation_name = (
                                f"{self._original_service_name}.{attr_name}"
                            )

                            with monitor.monitor_operation(operation_name):
                                return await original_method(*args, **kwargs)

                        setattr(self, attr_name, monitored_method)

            def _add_optimization_to_methods(self):
                """Add optimization decorators to service methods."""
                engine = self._performance_manager.optimization_engine

                # Apply caching to read-heavy methods
                cache_candidates = ["get", "find", "search", "list", "retrieve", "load"]

                for attr_name in dir(self):
                    attr = getattr(self, attr_name)
                    if (
                        callable(attr)
                        and not attr_name.startswith("_")
                        and any(
                            candidate in attr_name.lower()
                            for candidate in cache_candidates
                        )
                    ):
                        # Apply caching
                        cached_method = engine.cached()(attr)
                        setattr(self, attr_name, cached_method)

                # Apply memory optimization to data processing methods
                memory_candidates = [
                    "process",
                    "transform",
                    "analyze",
                    "detect",
                    "train",
                ]

                for attr_name in dir(self):
                    attr = getattr(self, attr_name)
                    if (
                        callable(attr)
                        and not attr_name.startswith("_")
                        and any(
                            candidate in attr_name.lower()
                            for candidate in memory_candidates
                        )
                    ):
                        # Apply memory optimization
                        optimized_method = engine.memory_optimized()(attr)
                        setattr(self, attr_name, optimized_method)

        # Store enhanced service
        self.enhanced_services[service_name] = EnhancedService

        logger.info(f"Enhanced service class created: {service_name}")
        return EnhancedService

    def get_enhanced_service(self, service_name: str) -> type | None:
        """Get an enhanced service class by name."""
        return self.enhanced_services.get(service_name)

    def optimize_detection_service(self, detection_service):
        """Optimize detection service with performance enhancements."""
        if not self.optimization_engine:
            return detection_service

        # Add caching to detection methods
        if hasattr(detection_service, "detect_anomalies"):
            detection_service.detect_anomalies = self.optimization_engine.cached()(
                detection_service.detect_anomalies
            )

        if hasattr(detection_service, "batch_detect"):
            detection_service.batch_detect = self.optimization_engine.batched(
                "detection_batch"
            )(detection_service.batch_detect)

        # Add memory optimization
        if hasattr(detection_service, "process_dataset"):
            detection_service.process_dataset = (
                self.optimization_engine.memory_optimized()(
                    detection_service.process_dataset
                )
            )

        # Add parallel processing for bulk operations
        if hasattr(detection_service, "process_multiple"):
            detection_service.process_multiple = self.optimization_engine.parallel()(
                detection_service.process_multiple
            )

        logger.info("Detection service optimized")
        return detection_service

    def optimize_training_service(self, training_service):
        """Optimize training service with performance enhancements."""
        if not self.optimization_engine:
            return training_service

        # Add caching to model loading
        if hasattr(training_service, "load_model"):
            training_service.load_model = self.optimization_engine.cached()(
                training_service.load_model
            )

        # Add memory optimization to training
        if hasattr(training_service, "train_model"):
            training_service.train_model = self.optimization_engine.memory_optimized()(
                training_service.train_model
            )

        # Add parallel processing for hyperparameter optimization
        if hasattr(training_service, "optimize_hyperparameters"):
            training_service.optimize_hyperparameters = (
                self.optimization_engine.parallel(use_processes=True)(
                    training_service.optimize_hyperparameters
                )
            )

        logger.info("Training service optimized")
        return training_service

    def optimize_data_service(self, data_service):
        """Optimize data service with performance enhancements."""
        if not self.optimization_engine:
            return data_service

        # Add caching to data loading
        if hasattr(data_service, "load_data"):
            data_service.load_data = self.optimization_engine.cached()(
                data_service.load_data
            )

        # Add memory optimization to data processing
        if hasattr(data_service, "preprocess_data"):
            data_service.preprocess_data = self.optimization_engine.memory_optimized()(
                data_service.preprocess_data
            )

        # Add parallel processing for data transformation
        if hasattr(data_service, "transform_batch"):
            data_service.transform_batch = self.optimization_engine.parallel()(
                data_service.transform_batch
            )

        logger.info("Data service optimized")
        return data_service

    async def benchmark_service(
        self,
        service_instance,
        service_name: str,
        test_configurations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Benchmark a service instance with various configurations."""
        if not self.benchmarking_service:
            raise ValueError("Benchmarking service not initialized")

        # Default test configurations
        if test_configurations is None:
            test_configurations = [
                {"dataset_size": 1000, "features": 10},
                {"dataset_size": 5000, "features": 20},
                {"dataset_size": 10000, "features": 50},
            ]

        # Create benchmark configuration
        config = AdvancedBenchmarkConfig(
            benchmark_name=f"{service_name}_benchmark",
            description=f"Performance benchmark for {service_name}",
            dataset_sizes=[cfg["dataset_size"] for cfg in test_configurations],
            feature_dimensions=[cfg["features"] for cfg in test_configurations],
            iterations=3,
            enable_memory_profiling=True,
            enable_cpu_profiling=True,
        )

        # Create benchmark suite
        suite_id = await self.benchmarking_service.create_benchmark_suite(
            suite_name=f"{service_name}_performance_test",
            description=f"Automated performance test for {service_name}",
            config=config,
        )

        # Define test functions for the service
        test_functions = {}

        # Detection service tests
        if hasattr(service_instance, "detect_anomalies"):

            async def detection_test(dataset):
                return await service_instance.detect_anomalies(dataset)

            test_functions["detect_anomalies"] = detection_test

        # Training service tests
        if hasattr(service_instance, "train_model"):

            async def training_test(dataset):
                return await service_instance.train_model(dataset)

            test_functions["train_model"] = training_test

        # Data service tests
        if hasattr(service_instance, "process_data"):

            async def processing_test(dataset):
                return await service_instance.process_data(dataset)

            test_functions["process_data"] = processing_test

        # Run comprehensive benchmark
        algorithms = list(test_functions.keys())
        results = await self.benchmarking_service.run_comprehensive_benchmark(
            suite_id=suite_id,
            algorithms=algorithms,
            custom_test_functions=test_functions,
        )

        return {
            "service_name": service_name,
            "benchmark_suite_id": suite_id,
            "results": results,
            "summary": {
                "total_tests": len(results.individual_results),
                "overall_performance_score": results.overall_performance_score,
                "performance_grade": results.performance_grade,
                "recommendations": results.recommendations,
            },
        }

    async def generate_performance_report(
        self,
        service_name: str = None,
        time_period: str = "24h",
        export_format: str = "html",
    ) -> Path:
        """Generate performance report for a service or the entire system."""
        if not self.performance_reporter or not self.performance_monitor:
            raise ValueError("Performance reporting components not initialized")

        # Get performance data
        performance_data = self.performance_monitor.performance_history

        # Filter by service if specified
        if service_name:
            performance_data = [
                entry
                for entry in performance_data
                if entry.get("execution_metrics", {})
                .get("operation_name", "")
                .startswith(service_name)
            ]

        # Generate comprehensive report
        report = self.performance_reporter.generate_comprehensive_report(
            performance_data=performance_data,
            time_period=time_period,
            include_visualizations=True,
        )

        # Export report
        report_path = self.performance_reporter.export_report(
            report=report, format=export_format, include_charts=True
        )

        logger.info(f"Performance report generated: {report_path}")
        return report_path

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        stats = {}

        if self.optimization_engine:
            stats["optimization"] = self.optimization_engine.get_optimization_stats()

        if self.performance_monitor:
            stats["monitoring"] = self.performance_monitor.get_performance_summary()

        if self.benchmarking_service:
            stats["benchmarking"] = {
                "active_benchmarks": len(self.benchmarking_service.active_benchmarks),
                "total_results": len(self.benchmarking_service.performance_history),
            }

        return stats

    def get_performance_alerts(self, severity: str = None) -> list[dict[str, Any]]:
        """Get performance alerts."""
        if not self.performance_monitor:
            return []

        alerts = self.performance_monitor.get_active_alerts(severity=severity)

        return [
            {
                "alert_id": str(alert.alert_id),
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity,
                "category": alert.category,
                "title": alert.title,
                "description": alert.description,
                "component": alert.component,
                "suggested_actions": alert.suggested_actions,
            }
            for alert in alerts
        ]

    async def auto_optimize_service(
        self, service_instance, service_name: str
    ) -> dict[str, Any]:
        """Automatically optimize a service based on its type and usage patterns."""
        optimization_results = {
            "service_name": service_name,
            "optimizations_applied": [],
            "performance_impact": {},
        }

        # Benchmark current performance
        logger.info(f"Benchmarking {service_name} before optimization...")
        before_benchmark = await self.benchmark_service(service_instance, service_name)

        # Apply optimizations based on service type
        if "detection" in service_name.lower():
            service_instance = self.optimize_detection_service(service_instance)
            optimization_results["optimizations_applied"].append(
                "detection_optimizations"
            )

        elif "training" in service_name.lower():
            service_instance = self.optimize_training_service(service_instance)
            optimization_results["optimizations_applied"].append(
                "training_optimizations"
            )

        elif "data" in service_name.lower():
            service_instance = self.optimize_data_service(service_instance)
            optimization_results["optimizations_applied"].append("data_optimizations")

        else:
            # Generic optimizations
            if hasattr(service_instance, "__class__"):
                enhanced_class = self.enhance_service(
                    service_instance.__class__, service_name
                )
                # Note: In practice, you'd need to replace the instance
                optimization_results["optimizations_applied"].append(
                    "generic_optimizations"
                )

        # Benchmark after optimization
        logger.info(f"Benchmarking {service_name} after optimization...")
        after_benchmark = await self.benchmark_service(service_instance, service_name)

        # Calculate performance impact
        before_score = before_benchmark["summary"]["overall_performance_score"]
        after_score = after_benchmark["summary"]["overall_performance_score"]

        optimization_results["performance_impact"] = {
            "before_score": before_score,
            "after_score": after_score,
            "improvement": after_score - before_score,
            "improvement_percentage": ((after_score - before_score) / before_score)
            * 100
            if before_score > 0
            else 0,
        }

        logger.info(f"Auto-optimization completed for {service_name}")
        logger.info(
            f"Performance improvement: {optimization_results['performance_impact']['improvement_percentage']:.1f}%"
        )

        return optimization_results

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all performance systems."""
        health_status = {
            "overall_status": "healthy",
            "components": {},
            "recommendations": [],
        }

        # Check optimization engine
        if self.optimization_engine:
            opt_stats = self.optimization_engine.get_optimization_stats()
            cache_hit_ratio = opt_stats.get("cache_stats", {}).get("hit_ratio", 0)

            health_status["components"]["optimization_engine"] = {
                "status": "healthy" if cache_hit_ratio > 0.5 else "degraded",
                "cache_hit_ratio": cache_hit_ratio,
                "cache_size_mb": opt_stats.get("cache_stats", {}).get(
                    "total_size_mb", 0
                ),
            }

            if cache_hit_ratio < 0.5:
                health_status["recommendations"].append(
                    "Consider tuning cache configuration"
                )

        # Check performance monitor
        if self.performance_monitor:
            active_alerts = self.performance_monitor.get_active_alerts()
            critical_alerts = [a for a in active_alerts if a.severity == "critical"]

            health_status["components"]["performance_monitor"] = {
                "status": "critical" if critical_alerts else "healthy",
                "active_alerts": len(active_alerts),
                "critical_alerts": len(critical_alerts),
                "monitoring_active": self.performance_monitor.monitoring_active,
            }

            if critical_alerts:
                health_status["overall_status"] = "critical"
                health_status["recommendations"].append(
                    "Address critical performance alerts immediately"
                )

        # Check benchmarking service
        if self.benchmarking_service:
            health_status["components"]["benchmarking_service"] = {
                "status": "healthy",
                "active_benchmarks": len(self.benchmarking_service.active_benchmarks),
                "total_history": len(self.benchmarking_service.performance_history),
            }

        return health_status

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.optimization_engine:
            self.optimization_engine.cleanup()


# Global instance for easy access
_performance_integration_manager: PerformanceIntegrationManager | None = None


def get_performance_integration_manager(
    storage_path: Path | None = None, **kwargs
) -> PerformanceIntegrationManager:
    """Get global performance integration manager instance."""
    global _performance_integration_manager

    if _performance_integration_manager is None:
        _performance_integration_manager = PerformanceIntegrationManager(
            storage_path=storage_path, **kwargs
        )

    return _performance_integration_manager


def performance_enhanced(service_name: str = None):
    """
    Decorator to automatically enhance a service class with performance optimizations.

    Usage:
        @performance_enhanced("my_service")
        class MyService:
            async def process_data(self, data):
                # This method will be automatically optimized
                return processed_data
    """

    def decorator(cls):
        manager = get_performance_integration_manager()
        enhanced_cls = manager.enhance_service(cls, service_name or cls.__name__)
        return enhanced_cls

    return decorator


# Context manager for performance monitoring
class performance_context:
    """Context manager for performance monitoring of code blocks."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.manager = get_performance_integration_manager()

    def __enter__(self):
        if self.manager.performance_monitor:
            return self.manager.performance_monitor.monitor_operation(
                self.operation_name
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Utility functions for easy integration
async def optimize_service_instance(service_instance, service_name: str) -> Any:
    """Optimize a service instance with performance enhancements."""
    manager = get_performance_integration_manager()
    return await manager.auto_optimize_service(service_instance, service_name)


async def benchmark_service_performance(
    service_instance,
    service_name: str,
    configurations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Benchmark service performance."""
    manager = get_performance_integration_manager()
    return await manager.benchmark_service(
        service_instance, service_name, configurations
    )


async def generate_system_performance_report(
    time_period: str = "24h", export_format: str = "html"
) -> Path:
    """Generate system-wide performance report."""
    manager = get_performance_integration_manager()
    return await manager.generate_performance_report(
        time_period=time_period, export_format=export_format
    )


def get_performance_stats() -> dict[str, Any]:
    """Get current performance statistics."""
    manager = get_performance_integration_manager()
    return manager.get_optimization_stats()


async def start_performance_monitoring():
    """Start performance monitoring systems."""
    manager = get_performance_integration_manager()
    await manager.start_performance_systems()


async def stop_performance_monitoring():
    """Stop performance monitoring systems."""
    manager = get_performance_integration_manager()
    await manager.stop_performance_systems()


# Example usage and integration patterns
if __name__ == "__main__":
    # Example of how to use the performance integration
    async def example_usage():
        # Initialize performance integration
        manager = PerformanceIntegrationManager(
            storage_path=Path("example_performance"),
            enable_optimization=True,
            enable_monitoring=True,
            enable_reporting=True,
        )

        # Start performance systems
        await manager.start_performance_systems()

        try:
            # Example service class
            class ExampleDetectionService:
                async def detect_anomalies(self, data):
                    # Simulate processing
                    await asyncio.sleep(0.1)
                    return {"anomalies": [1, 2, 3]}

                async def process_data(self, data):
                    # Simulate data processing
                    await asyncio.sleep(0.05)
                    return data

            # Create service instance
            service = ExampleDetectionService()

            # Optimize the service
            optimization_results = await manager.auto_optimize_service(
                service, "example_detection_service"
            )

            print(f"Optimization results: {optimization_results}")

            # Generate performance report
            report_path = await manager.generate_performance_report(
                service_name="example_detection_service", time_period="1h"
            )

            print(f"Performance report generated: {report_path}")

            # Get performance stats
            stats = manager.get_optimization_stats()
            print(f"Performance stats: {stats}")

        finally:
            # Stop performance systems
            await manager.stop_performance_systems()

    # Run example
    asyncio.run(example_usage())
