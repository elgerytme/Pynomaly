"""
Unified API service for connecting all data science packages.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field

from integration.domain.entities.integration_config import IntegrationConfig
from integration.domain.value_objects.performance_metrics import PerformanceMetrics
from interfaces.shared.error_handling import handle_exceptions


logger = logging.getLogger(__name__)


@dataclass
class PackageClient:
    """Client for communicating with a specific package."""
    name: str
    instance: Any
    health_check: Callable[[], bool]
    operations: Dict[str, Callable] = field(default_factory=dict)
    last_health_check: Optional[datetime] = None
    is_healthy: bool = True
    circuit_breaker_failures: int = 0


class UnifiedApiService:
    """Service for providing unified API access to all data science packages."""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize the unified API service."""
        self.config = config
        self.clients: Dict[str, PackageClient] = {}
        self.connection_pools: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}
        self.metrics_collector = None
        
    async def initialize(self) -> None:
        """Initialize all package clients and connections."""
        logger.info("Initializing unified API service...")
        
        # Initialize package clients in dependency order
        load_order = self.config.get_package_load_order()
        
        for package_name in load_order:
            package_config = self.config.get_package_config(package_name)
            if package_config and package_config.enabled:
                await self._initialize_package_client(package_name, package_config)
        
        # Initialize connection pools
        await self._initialize_connection_pools()
        
        # Start health checks
        asyncio.create_task(self._run_health_checks())
        
        logger.info("Unified API service initialized successfully")
    
    async def _initialize_package_client(self, package_name: str, config: Any) -> None:
        """Initialize a client for a specific package."""
        try:
            # Import package dynamically
            if package_name == "data_profiling":
                from packages.data_profiling.application.services.schema_analysis_service import SchemaAnalysisService
                from packages.data_profiling.application.services.statistical_profiling_service import StatisticalProfilingService
                from packages.data_profiling.application.services.pattern_discovery_service import PatternDiscoveryService
                
                instance = {
                    "schema_analysis": SchemaAnalysisService(),
                    "statistical_profiling": StatisticalProfilingService(),
                    "pattern_discovery": PatternDiscoveryService()
                }
                
                health_check = lambda: True  # Simple health check
                
                operations = {
                    "analyze_schema": instance["schema_analysis"].analyze_schema,
                    "profile_dataset": instance["statistical_profiling"].profile_dataset,
                    "discover_patterns": instance["pattern_discovery"].discover_patterns
                }
                
            elif package_name == "data_quality":
                from packages.data_quality.application.services.intelligent_rule_discovery_service import IntelligentRuleDiscoveryService
                from packages.data_quality.application.services.ml_quality_detection_service import MLQualityDetectionService
                from packages.data_quality.application.services.quality_lineage_service import QualityLineageService
                
                instance = {
                    "rule_discovery": IntelligentRuleDiscoveryService(),
                    "ml_quality_detection": MLQualityDetectionService(),
                    "quality_lineage": QualityLineageService()
                }
                
                health_check = lambda: True
                
                operations = {
                    "discover_rules": instance["rule_discovery"].discover_rules,
                    "detect_quality_issues": instance["ml_quality_detection"].detect_quality_issues,
                    "track_lineage": instance["quality_lineage"].track_lineage
                }
                
            elif package_name == "data_science":
                from packages.data_science.domain.services.statistical_analysis_service import StatisticalAnalysisService
                from packages.data_science.domain.services.feature_engineering_service import FeatureEngineeringService
                from packages.data_science.domain.services.model_validation_service import ModelValidationService
                
                instance = {
                    "statistical_analysis": StatisticalAnalysisService(),
                    "feature_engineering": FeatureEngineeringService(),
                    "model_validation": ModelValidationService()
                }
                
                health_check = lambda: True
                
                operations = {
                    "analyze_statistics": instance["statistical_analysis"].analyze_statistics,
                    "engineer_features": instance["feature_engineering"].engineer_features,
                    "validate_model": instance["model_validation"].validate_model
                }
                
            else:
                logger.warning(f"Unknown package: {package_name}")
                return
            
            client = PackageClient(
                name=package_name,
                instance=instance,
                health_check=health_check,
                operations=operations
            )
            
            self.clients[package_name] = client
            logger.info(f"Initialized client for package: {package_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize package {package_name}: {str(e)}")
            raise
    
    async def _initialize_connection_pools(self) -> None:
        """Initialize connection pools for database and external services."""
        # Initialize database connection pool
        pool_config = self.config.performance
        
        # Create connection pools based on configuration
        # This is a simplified example - in practice, you'd use proper connection pooling
        self.connection_pools = {
            "database": {
                "max_connections": pool_config.connection_pool_size,
                "timeout": pool_config.query_timeout_seconds
            },
            "cache": {
                "max_size": pool_config.cache_size_mb,
                "ttl": 3600
            }
        }
        
        logger.info("Connection pools initialized")
    
    async def _run_health_checks(self) -> None:
        """Run periodic health checks for all package clients."""
        interval = self.config.monitoring.health_check_interval
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                for client in self.clients.values():
                    try:
                        is_healthy = client.health_check()
                        client.is_healthy = is_healthy
                        client.last_health_check = datetime.utcnow()
                        
                        if is_healthy:
                            client.circuit_breaker_failures = 0
                        else:
                            client.circuit_breaker_failures += 1
                            
                            # Circuit breaker logic
                            threshold = self.config.packages[client.name].circuit_breaker_threshold
                            if client.circuit_breaker_failures >= threshold:
                                logger.warning(f"Circuit breaker opened for package: {client.name}")
                                
                    except Exception as e:
                        logger.error(f"Health check failed for {client.name}: {str(e)}")
                        client.is_healthy = False
                        client.circuit_breaker_failures += 1
                        
            except Exception as e:
                logger.error(f"Health check loop error: {str(e)}")
    
    @handle_exceptions
    async def execute_operation(self, package_name: str, operation_name: str, 
                              **kwargs) -> Any:
        """Execute an operation on a specific package."""
        client = self.clients.get(package_name)
        if not client:
            raise ValueError(f"Package not found: {package_name}")
        
        if not client.is_healthy:
            raise RuntimeError(f"Package {package_name} is not healthy")
        
        operation = client.operations.get(operation_name)
        if not operation:
            raise ValueError(f"Operation {operation_name} not found in package {package_name}")
        
        try:
            # Execute operation with timeout
            timeout = self.config.packages[package_name].timeout_seconds
            result = await asyncio.wait_for(
                self._execute_with_retry(operation, **kwargs),
                timeout=timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Operation {operation_name} timed out for package {package_name}")
            raise
        except Exception as e:
            logger.error(f"Operation {operation_name} failed for package {package_name}: {str(e)}")
            raise
    
    async def _execute_with_retry(self, operation: Callable, **kwargs) -> Any:
        """Execute operation with retry logic."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(**kwargs)
                else:
                    return operation(**kwargs)
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                logger.warning(f"Operation failed on attempt {attempt + 1}: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    @handle_exceptions
    async def execute_workflow(self, workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a workflow across multiple packages."""
        results = {}
        
        for step in workflow_steps:
            package_name = step.get("package")
            operation_name = step.get("operation")
            params = step.get("params", {})
            
            # Use results from previous steps as input
            if "input_from" in step:
                input_step = step["input_from"]
                if input_step in results:
                    params.update(results[input_step])
            
            result = await self.execute_operation(package_name, operation_name, **params)
            results[step.get("name", f"{package_name}_{operation_name}")] = result
        
        return results
    
    @handle_exceptions
    async def get_package_status(self, package_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status information for packages."""
        if package_name:
            client = self.clients.get(package_name)
            if not client:
                raise ValueError(f"Package not found: {package_name}")
            
            return {
                "name": client.name,
                "healthy": client.is_healthy,
                "last_health_check": client.last_health_check,
                "circuit_breaker_failures": client.circuit_breaker_failures,
                "operations": list(client.operations.keys())
            }
        
        # Return status for all packages
        return {
            name: {
                "healthy": client.is_healthy,
                "last_health_check": client.last_health_check,
                "circuit_breaker_failures": client.circuit_breaker_failures,
                "operations": list(client.operations.keys())
            }
            for name, client in self.clients.items()
        }
    
    @handle_exceptions
    async def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics for all packages."""
        # This is a simplified example - in practice, you'd collect real metrics
        from integration.domain.value_objects.performance_metrics import (
            PerformanceMetrics, SystemMetrics, ApplicationMetrics, PackageMetrics
        )
        
        system_metrics = SystemMetrics(
            cpu_usage_percent=45.0,
            memory_usage_percent=60.0,
            disk_usage_percent=70.0,
            network_io_bytes_per_second=1024000,
            disk_io_bytes_per_second=512000,
            active_connections=100
        )
        
        app_metrics = ApplicationMetrics(
            request_count=1000,
            error_count=10,
            response_time_ms=250.0,
            throughput_requests_per_second=50.0,
            active_users=25,
            memory_usage_mb=512.0,
            cache_hit_rate=0.85
        )
        
        package_metrics = {}
        for name, client in self.clients.items():
            package_metrics[name] = PackageMetrics(
                package_name=name,
                operation_count=100,
                average_execution_time_ms=200.0,
                max_execution_time_ms=1000.0,
                min_execution_time_ms=50.0,
                success_count=95,
                failure_count=5,
                memory_usage_mb=256.0,
                cpu_usage_percent=30.0
            )
        
        return PerformanceMetrics(
            system=system_metrics,
            application=app_metrics,
            packages=package_metrics
        )
    
    async def shutdown(self) -> None:
        """Shutdown the unified API service."""
        logger.info("Shutting down unified API service...")
        
        # Close all connections
        for pool in self.connection_pools.values():
            # Close connection pools
            pass
        
        # Clear cache
        self.cache.clear()
        
        # Clear clients
        self.clients.clear()
        
        logger.info("Unified API service shutdown complete")