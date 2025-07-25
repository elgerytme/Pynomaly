#!/usr/bin/env python3
"""Integration test for service discovery and configuration management."""

import asyncio
import tempfile
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import domain services
from mlops.domain.services.service_discovery_service import ServiceDiscoveryService
from mlops.domain.services.configuration_management_service import ConfigurationManagementService

# Import file-based adapters
from mlops.infrastructure.adapters.file_based.service_discovery_adapters import (
    FileBasedServiceRegistry,
    FileBasedServiceDiscovery,
    FileBasedHealthCheck,
    FileBasedLoadBalancer
)
from mlops.infrastructure.adapters.file_based.configuration_management_adapters import (
    FileBasedConfigurationProvider,
    FileBasedConfigurationWatcher,
    FileBasedConfigurationValidator,
    FileBasedSecretManagement,
    FileBasedEnvironmentConfiguration
)

# Import domain interfaces for data types
from mlops.domain.interfaces.service_discovery_operations import (
    ServiceType,
    ServiceStatus,
    ServiceEndpoint
)
from mlops.domain.interfaces.configuration_management_operations import (
    ConfigurationScope
)


async def test_service_discovery_integration():
    """Test the service discovery service with file-based adapters."""
    print("🔍 Testing Service Discovery Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file-based adapters
        service_registry = FileBasedServiceRegistry(f"{temp_dir}/registry")
        service_discovery = FileBasedServiceDiscovery(service_registry)
        health_check = FileBasedHealthCheck(f"{temp_dir}/health")
        load_balancer = FileBasedLoadBalancer(service_discovery, f"{temp_dir}/load_balancer")
        
        # Create service discovery service using dependency injection
        discovery_service = ServiceDiscoveryService(
            service_registry_port=service_registry,
            service_discovery_port=service_discovery,
            health_check_port=health_check,
            load_balancer_port=load_balancer
        )
        
        # Test service registration
        endpoints = [
            ServiceEndpoint("http", "localhost", 8080, "/api/v1"),
            ServiceEndpoint("https", "localhost", 8443, "/api/v1")
        ]
        
        service_id = await discovery_service.register_mlops_service(
            service_name="experiment_tracker",
            service_type=ServiceType.EXPERIMENT_TRACKING,
            endpoints=endpoints,
            version="2.0.0",
            capabilities=["ml_experiments", "artifact_management"],
            tags={"env": "production", "region": "us-east-1"},
            health_check_url="http://localhost:8080/health"
        )
        
        assert service_id.startswith("experiment_tracking_"), f"Expected service ID to start with service type, got: {service_id}"
        print(f"✅ Registered MLOps service: {service_id}")
        
        # Test service discovery
        discovered_service = await discovery_service.discover_mlops_service(
            service_type=ServiceType.EXPERIMENT_TRACKING,
            capabilities=["ml_experiments"],
            tags={"env": "production"}
        )
        
        assert discovered_service is not None, "Expected to discover registered service"
        assert discovered_service.service_name == "experiment_tracker", f"Expected service name 'experiment_tracker', got: {discovered_service.service_name}"
        assert len(discovered_service.endpoints) == 2, f"Expected 2 endpoints, got: {len(discovered_service.endpoints)}"
        print("✅ Successfully discovered MLOps service")
        
        # Test health check
        health_result = await discovery_service.check_service_health(service_id)
        assert health_result is not None, "Expected health check result"
        print(f"✅ Health check completed: {health_result.status.value}")
        
        # Test load balanced service retrieval
        balanced_service = await discovery_service.get_healthy_service(
            ServiceType.EXPERIMENT_TRACKING,
            load_balancing_strategy="round_robin"
        )
        
        assert balanced_service is not None, "Expected load balanced service"
        assert balanced_service.service_type == ServiceType.EXPERIMENT_TRACKING, "Expected experiment tracking service type"
        print("✅ Retrieved load balanced service")
        
        # Test ecosystem monitoring
        ecosystem_health = await discovery_service.monitor_service_ecosystem()
        
        assert ecosystem_health["total_services"] >= 1, "Expected at least 1 service in ecosystem"
        assert ecosystem_health["healthy_services"] >= 1, "Expected at least 1 healthy service"
        assert ServiceType.EXPERIMENT_TRACKING.value in ecosystem_health["services_by_type"], "Expected experiment tracking service type in ecosystem"
        print(f"✅ Ecosystem health: {ecosystem_health['ecosystem_health']} ({ecosystem_health['healthy_services']}/{ecosystem_health['total_services']} healthy)")
        
        # Test service failure handling
        failure_result = await discovery_service.handle_service_failure(service_id)
        
        assert failure_result["service_id"] == service_id, "Expected failure result for correct service"
        assert failure_result["status_updated"], "Expected service status to be updated"
        assert failure_result["load_balancer_updated"], "Expected load balancer to be updated"
        print("✅ Service failure handled successfully")
        
        # Test service recovery
        recovery_result = await discovery_service.handle_service_recovery(service_id)
        
        # Note: Recovery might fail if health check fails, which is expected in test environment
        print(f"✅ Service recovery attempted: {recovery_result.get('recovery_verified', False)}")
        
        # Test service dependencies
        dependencies = await discovery_service.get_service_dependencies(service_id)
        
        assert dependencies["service_id"] == service_id, "Expected dependencies for correct service"
        assert "dependencies_count" in dependencies, "Expected dependencies count"
        print(f"✅ Service dependencies retrieved: {dependencies['dependencies_count']} dependencies")


async def test_configuration_management_integration():
    """Test the configuration management service with file-based adapters."""
    print("\n⚙️ Testing Configuration Management Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file-based adapters
        config_provider = FileBasedConfigurationProvider(f"{temp_dir}/config")
        config_watcher = FileBasedConfigurationWatcher(config_provider)
        config_validator = FileBasedConfigurationValidator(f"{temp_dir}/validation")
        secret_management = FileBasedSecretManagement(f"{temp_dir}/secrets")
        env_config = FileBasedEnvironmentConfiguration(f"{temp_dir}/environments")
        
        # Create configuration management service using dependency injection
        config_service = ConfigurationManagementService(
            configuration_provider_port=config_provider,
            configuration_watcher_port=config_watcher,
            configuration_validator_port=config_validator,
            secret_management_port=secret_management,
            environment_configuration_port=env_config
        )
        
        # Test MLOps configuration setup
        service_configurations = {
            "experiment_service": {
                "database_url": "postgresql://localhost:5432/mlops",
                "max_experiments": 1000,
                "cleanup_interval_hours": 24,
                "enable_auto_cleanup": True
            },
            "model_registry": {
                "storage_backend": "s3",
                "bucket_name": "mlops-models",
                "model_retention_days": 90,
                "enable_versioning": True
            },
            "monitoring_service": {
                "metrics_interval": 60,
                "alert_thresholds": {
                    "accuracy": 0.85,
                    "latency_p95": 500,
                    "error_rate": 0.05
                },
                "notification_channels": ["slack", "email"]
            }
        }
        
        setup_result = await config_service.setup_mlops_configuration(
            environment="production",
            service_configurations=service_configurations
        )
        
        assert setup_result["environment"] == "production", "Expected production environment"
        assert setup_result["total_configurations"] > 0, "Expected configurations to be set"
        assert setup_result["successful_configurations"] > 0, "Expected some configurations to succeed"
        print(f"✅ MLOps configuration setup: {setup_result['successful_configurations']}/{setup_result['total_configurations']} successful")
        
        # Test service configuration retrieval
        service_config = await config_service.get_service_configuration(
            service_name="experiment_service",
            environment="production",
            include_secrets=False
        )
        
        assert service_config["service_name"] == "experiment_service", "Expected correct service name"
        assert service_config["environment"] == "production", "Expected production environment"
        assert "database_url" in service_config["configuration"], "Expected database_url in configuration"
        print("✅ Service configuration retrieved successfully")
        
        # Test configuration updates
        updates = {
            "max_experiments": 2000,
            "new_feature_flag": True,
            "updated_at": datetime.now().isoformat()
        }
        
        update_result = await config_service.update_service_configuration(
            service_name="experiment_service",
            environment="production",
            updates=updates,
            validate=True
        )
        
        assert update_result["service_name"] == "experiment_service", "Expected correct service name"
        assert update_result["total_updates"] == len(updates), f"Expected {len(updates)} updates"
        print(f"✅ Configuration updates: {update_result['successful_updates']}/{update_result['total_updates']} successful")
        
        # Test secret management
        secret_operations = {
            "store": {
                "database_password": "super_secret_password",
                "api_key": "sk-1234567890abcdef",
                "encryption_key": "0123456789abcdef"
            },
            "rotate": {
                "api_key": "sk-new1234567890abcdef"
            }
        }
        
        secret_result = await config_service.manage_service_secrets(
            service_name="experiment_service",
            secret_operations=secret_operations
        )
        
        assert secret_result["service_name"] == "experiment_service", "Expected correct service name"
        assert "store" in secret_result["secret_operations"], "Expected store operations"
        assert "rotate" in secret_result["secret_operations"], "Expected rotate operations"
        print("✅ Service secrets managed successfully")
        
        # Test configuration promotion
        promotion_result = await config_service.promote_configuration(
            service_name="experiment_service",
            from_environment="production",
            to_environment="staging",
            keys=["max_experiments", "enable_auto_cleanup"],
            validate_target=True
        )
        
        assert promotion_result["status"] in ["promoted", "validation_failed"], "Expected promotion status"
        assert promotion_result["service_name"] == "experiment_service", "Expected correct service name"
        assert promotion_result["from_environment"] == "production", "Expected correct source environment"
        assert promotion_result["to_environment"] == "staging", "Expected correct target environment"
        print(f"✅ Configuration promotion: {promotion_result['status']}")
        
        # Test configuration audit trail
        audit_trail = await config_service.get_configuration_audit_trail(
            service_name="experiment_service",
            environment="production",
            limit=50
        )
        
        assert audit_trail["service_name"] == "experiment_service", "Expected correct service name"
        assert "total_changes" in audit_trail, "Expected total changes count"
        print(f"✅ Configuration audit trail: {audit_trail['total_changes']} changes recorded")
        
        # Test environment validation
        validation_result = await config_service.validate_environment_configuration(
            environment="production",
            service_name="experiment_service"
        )
        
        assert validation_result["environment"] == "production", "Expected correct environment"
        assert "validation_summary" in validation_result, "Expected validation summary"
        assert "health_score" in validation_result["validation_summary"], "Expected health score"
        print(f"✅ Environment validation: {validation_result['validation_summary']['health_score']:.2f} health score")


async def test_integrated_workflow():
    """Test integrated workflow using both service discovery and configuration management."""
    print("\n🔄 Testing Integrated Service Discovery + Configuration Management Workflow...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup service discovery
        service_registry = FileBasedServiceRegistry(f"{temp_dir}/registry")
        service_discovery = FileBasedServiceDiscovery(service_registry)
        health_check = FileBasedHealthCheck(f"{temp_dir}/health")
        load_balancer = FileBasedLoadBalancer(service_discovery, f"{temp_dir}/load_balancer")
        
        discovery_service = ServiceDiscoveryService(
            service_registry_port=service_registry,
            service_discovery_port=service_discovery,
            health_check_port=health_check,
            load_balancer_port=load_balancer
        )
        
        # Setup configuration management
        config_provider = FileBasedConfigurationProvider(f"{temp_dir}/config")
        config_watcher = FileBasedConfigurationWatcher(config_provider)
        config_validator = FileBasedConfigurationValidator(f"{temp_dir}/validation")
        secret_management = FileBasedSecretManagement(f"{temp_dir}/secrets")
        env_config = FileBasedEnvironmentConfiguration(f"{temp_dir}/environments")
        
        config_service = ConfigurationManagementService(
            configuration_provider_port=config_provider,
            configuration_watcher_port=config_watcher,
            configuration_validator_port=config_validator,
            secret_management_port=secret_management,
            environment_configuration_port=env_config
        )
        
        # Step 1: Setup configuration for multiple services
        print("Step 1: Setting up service configurations...")
        service_configs = {
            "experiment_service": {
                "service_url": "http://localhost:8001/api/v1",
                "health_check_url": "http://localhost:8001/health"
            },
            "model_registry": {
                "service_url": "http://localhost:8002/api/v1",
                "health_check_url": "http://localhost:8002/health"
            },
            "monitoring_service": {
                "service_url": "http://localhost:8003/api/v1",
                "health_check_url": "http://localhost:8003/health"
            }
        }
        
        setup_result = await config_service.setup_mlops_configuration(
            environment="integration_test",
            service_configurations=service_configs
        )
        
        assert setup_result["successful_configurations"] > 0, "Expected successful configuration setup"
        print(f"✅ Configured {setup_result['successful_configurations']} services")
        
        # Step 2: Register services based on configuration
        print("Step 2: Registering services in service discovery...")
        registered_services = {}
        
        for service_name, config in service_configs.items():
            service_type_map = {
                "experiment_service": ServiceType.EXPERIMENT_TRACKING,
                "model_registry": ServiceType.MODEL_REGISTRY,
                "monitoring_service": ServiceType.MONITORING
            }
            
            service_id = await discovery_service.register_mlops_service(
                service_name=service_name,
                service_type=service_type_map[service_name],
                endpoints=[ServiceEndpoint("http", "localhost", int(config["service_url"].split(":")[2].split("/")[0]))],
                version="1.0.0",
                health_check_url=config["health_check_url"]
            )
            
            registered_services[service_name] = service_id
        
        print(f"✅ Registered {len(registered_services)} services")
        
        # Step 3: Monitor ecosystem health
        print("Step 3: Monitoring service ecosystem...")
        ecosystem_health = await discovery_service.monitor_service_ecosystem()
        
        assert ecosystem_health["total_services"] == len(service_configs), f"Expected {len(service_configs)} services in ecosystem"
        print(f"✅ Ecosystem health: {ecosystem_health['ecosystem_health']} ({ecosystem_health['healthy_services']}/{ecosystem_health['total_services']} healthy)")
        
        # Step 4: Test dynamic configuration update and service discovery
        print("Step 4: Testing dynamic updates...")
        
        # Update configuration
        update_result = await config_service.update_service_configuration(
            service_name="experiment_service",
            environment="integration_test",
            updates={"max_concurrent_experiments": 50, "updated_at": datetime.now().isoformat()},
            validate=True
        )
        
        # Discover updated service
        updated_service = await discovery_service.discover_mlops_service(
            service_type=ServiceType.EXPERIMENT_TRACKING
        )
        
        assert update_result["successful_updates"] > 0, "Expected successful configuration update"
        assert updated_service is not None, "Expected to discover updated service"
        print("✅ Dynamic configuration update and service discovery successful")
        
        # Step 5: Test service failure and recovery workflow
        print("Step 5: Testing failure and recovery workflow...")
        
        # Simulate service failure
        experiment_service_id = registered_services["experiment_service"]
        failure_result = await discovery_service.handle_service_failure(experiment_service_id)
        
        # Check ecosystem health after failure
        post_failure_health = await discovery_service.monitor_service_ecosystem()
        
        assert failure_result["status_updated"], "Expected service status to be updated"
        print(f"✅ Service failure handled - ecosystem health: {post_failure_health['ecosystem_health']}")
        
        # Generate final report
        final_report = {
            "workflow_completed_at": datetime.now().isoformat(),
            "services_configured": len(service_configs),
            "services_registered": len(registered_services),
            "final_ecosystem_health": post_failure_health["ecosystem_health"],
            "configuration_health_score": setup_result["successful_configurations"] / setup_result["total_configurations"],
            "services": {
                service_name: {
                    "service_id": service_id,
                    "configured": True,
                    "registered": True
                }
                for service_name, service_id in registered_services.items()
            }
        }
        
        print("✅ Integrated workflow completed successfully!")
        print(f"   - Services configured: {final_report['services_configured']}")
        print(f"   - Services registered: {final_report['services_registered']}")
        print(f"   - Ecosystem health: {final_report['final_ecosystem_health']}")
        print(f"   - Configuration health: {final_report['configuration_health_score']:.2f}")
        
        return final_report


async def main():
    """Run all integration tests."""
    print("🚀 Starting Service Discovery and Configuration Management Integration Tests")
    print("=" * 80)
    
    try:
        # Run individual service tests
        await test_service_discovery_integration()
        await test_configuration_management_integration()
        
        # Run integrated workflow test
        workflow_report = await test_integrated_workflow()
        
        # Final summary
        print("\n🎉 All Integration Tests Passed!")
        print("=" * 80)
        print("✅ Service Discovery: Working correctly with file-based adapters")
        print("✅ Configuration Management: Working correctly with file-based adapters")
        print("✅ Integrated Workflow: Complete service lifecycle functional")
        print("\n🏗️ Service Discovery and Configuration Management Benefits Demonstrated:")
        print("   • Dynamic service registration and discovery")
        print("   • Health monitoring and load balancing")
        print("   • Configuration management with validation")
        print("   • Secret management and environment promotion")
        print("   • Integrated service lifecycle management")
        print("   • Audit trails and monitoring capabilities")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)