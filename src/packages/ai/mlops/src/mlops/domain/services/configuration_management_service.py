"""Configuration management domain service with dependency injection."""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from mlops.domain.interfaces.configuration_management_operations import (
    ConfigurationProviderPort,
    ConfigurationWatcherPort,
    ConfigurationValidatorPort,
    SecretManagementPort,
    EnvironmentConfigurationPort,
    ConfigurationValue,
    ConfigurationChange,
    ConfigurationQuery,
    ConfigurationValidation,
    ConfigurationScope,
    ConfigurationSource
)


class ConfigurationManagementService:
    """Domain service for configuration management operations using dependency injection."""
    
    def __init__(
        self,
        configuration_provider_port: ConfigurationProviderPort,
        configuration_watcher_port: ConfigurationWatcherPort,
        configuration_validator_port: ConfigurationValidatorPort,
        secret_management_port: SecretManagementPort,
        environment_configuration_port: EnvironmentConfigurationPort
    ):
        self.configuration_provider_port = configuration_provider_port
        self.configuration_watcher_port = configuration_watcher_port
        self.configuration_validator_port = configuration_validator_port
        self.secret_management_port = secret_management_port
        self.environment_configuration_port = environment_configuration_port
    
    async def setup_mlops_configuration(
        self,
        environment: str,
        service_configurations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Set up MLOps configuration for an environment.
        
        Args:
            environment: Environment name (dev, staging, prod)
            service_configurations: Configuration for each service
            
        Returns:
            Setup result with validation details
        """
        setup_results = {}
        validation_results = []
        
        for service_name, config in service_configurations.items():
            # Validate configuration
            for key, value in config.items():
                validation = await self.configuration_validator_port.validate_configuration(
                    f"{service_name}.{key}", value
                )
                validation_results.append(validation)
                
                if not validation.is_valid:
                    setup_results[f"{service_name}.{key}"] = {
                        "status": "validation_failed",
                        "error": validation.error_message
                    }
                    continue
            
            # Set environment configuration
            success = await self.environment_configuration_port.set_environment_config(
                environment, config, service_name
            )
            
            setup_results[service_name] = {
                "status": "configured" if success else "failed",
                "config_keys": list(config.keys())
            }
        
        return {
            "environment": environment,
            "setup_results": setup_results,
            "validation_results": [
                {
                    "key": v.key,
                    "is_valid": v.is_valid,
                    "error_message": v.error_message,
                    "warnings": v.warnings
                }
                for v in validation_results
            ],
            "total_configurations": sum(len(config) for config in service_configurations.values()),
            "successful_configurations": sum(
                1 for result in setup_results.values() 
                if isinstance(result, dict) and result.get("status") == "configured"
            ),
            "setup_completed_at": datetime.now().isoformat()
        }
    
    async def get_service_configuration(
        self,
        service_name: str,
        environment: str,
        include_secrets: bool = False
    ) -> Dict[str, Any]:
        """Get configuration for a service in an environment.
        
        Args:
            service_name: Name of the service
            environment: Environment name
            include_secrets: Whether to include secret values
            
        Returns:
            Service configuration
        """
        # Get environment-specific configuration
        env_config = await self.environment_configuration_port.get_environment_config(
            environment, service_name
        )
        
        # Get global configuration
        global_config = {}
        for key in env_config.keys():
            global_key = f"{service_name}.{key}"
            global_value = await self.configuration_provider_port.get_configuration(
                global_key, ConfigurationScope.GLOBAL
            )
            if global_value is not None:
                global_config[key] = global_value
        
        # Merge configurations (environment overrides global)
        merged_config = {**global_config, **env_config}
        
        # Include secrets if requested
        if include_secrets:
            secret_keys = await self.secret_management_port.list_secret_keys(
                ConfigurationScope.SERVICE
            )
            
            service_secret_keys = [
                key for key in secret_keys 
                if key.startswith(f"{service_name}.")
            ]
            
            for secret_key in service_secret_keys:
                config_key = secret_key.replace(f"{service_name}.", "")
                secret_value = await self.secret_management_port.get_secret(
                    secret_key, ConfigurationScope.SERVICE
                )
                if secret_value:
                    merged_config[f"secret_{config_key}"] = secret_value
        
        return {
            "service_name": service_name,
            "environment": environment,
            "configuration": merged_config,
            "configuration_sources": {
                "global_keys": list(global_config.keys()),
                "environment_keys": list(env_config.keys()),
                "secret_keys": list(secret_keys) if include_secrets else []
            },
            "retrieved_at": datetime.now().isoformat()
        }
    
    async def update_service_configuration(
        self,
        service_name: str,
        environment: str,
        updates: Dict[str, Any],
        validate: bool = True
    ) -> Dict[str, Any]:
        """Update service configuration.
        
        Args:
            service_name: Name of the service
            environment: Environment name
            updates: Configuration updates
            validate: Whether to validate before updating
            
        Returns:
            Update result
        """
        validation_results = []
        update_results = {}
        
        if validate:
            # Validate all updates
            for key, value in updates.items():
                validation = await self.configuration_validator_port.validate_configuration(
                    f"{service_name}.{key}", value
                )
                validation_results.append(validation)
                
                if not validation.is_valid:
                    update_results[key] = {
                        "status": "validation_failed",
                        "error": validation.error_message
                    }
                    continue
        
        # Apply valid updates
        valid_updates = {
            key: value for key, value in updates.items()
            if key not in update_results or update_results[key].get("status") != "validation_failed"
        }
        
        if valid_updates:
            success = await self.environment_configuration_port.set_environment_config(
                environment, valid_updates, service_name
            )
            
            for key in valid_updates:
                update_results[key] = {
                    "status": "updated" if success else "failed"
                }
        
        return {
            "service_name": service_name,
            "environment": environment,
            "update_results": update_results,
            "validation_results": [
                {
                    "key": v.key,
                    "is_valid": v.is_valid,
                    "error_message": v.error_message
                }
                for v in validation_results
            ] if validate else [],
            "total_updates": len(updates),
            "successful_updates": sum(
                1 for result in update_results.values()
                if result.get("status") == "updated"
            ),
            "updated_at": datetime.now().isoformat()
        }
    
    async def manage_service_secrets(
        self,
        service_name: str,
        secret_operations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Manage secrets for a service.
        
        Args:
            service_name: Name of the service
            secret_operations: Dictionary of operations (store, rotate, delete)
            
        Returns:
            Secret management results
        """
        results = {}
        
        for operation, secrets in secret_operations.items():
            operation_results = {}
            
            if operation == "store":
                for secret_key, secret_value in secrets.items():
                    full_key = f"{service_name}.{secret_key}"
                    success = await self.secret_management_port.store_secret(
                        full_key, secret_value, ConfigurationScope.SERVICE
                    )
                    operation_results[secret_key] = "stored" if success else "failed"
            
            elif operation == "rotate":
                for secret_key, new_value in secrets.items():
                    full_key = f"{service_name}.{secret_key}"
                    success = await self.secret_management_port.rotate_secret(
                        full_key, new_value, ConfigurationScope.SERVICE
                    )
                    operation_results[secret_key] = "rotated" if success else "failed"
            
            elif operation == "delete":
                for secret_key in secrets.keys():
                    full_key = f"{service_name}.{secret_key}"
                    success = await self.secret_management_port.delete_secret(
                        full_key, ConfigurationScope.SERVICE
                    )
                    operation_results[secret_key] = "deleted" if success else "failed"
            
            results[operation] = operation_results
        
        return {
            "service_name": service_name,
            "secret_operations": results,
            "managed_at": datetime.now().isoformat()
        }
    
    async def watch_configuration_changes(
        self,
        service_name: str,
        environment: str,
        callback: Callable[[List[ConfigurationChange]], None]
    ) -> str:
        """Watch for configuration changes for a service.
        
        Args:
            service_name: Name of the service
            environment: Environment name
            callback: Function to call when configuration changes
            
        Returns:
            Watch ID for cancelling the watch
        """
        # Watch service-specific configuration prefix
        prefix = f"{service_name}."
        
        return await self.configuration_watcher_port.watch_configuration_prefix(
            prefix, callback, ConfigurationScope.GLOBAL
        )
    
    async def promote_configuration(
        self,
        service_name: str,
        from_environment: str,
        to_environment: str,
        keys: Optional[List[str]] = None,
        validate_target: bool = True
    ) -> Dict[str, Any]:
        """Promote configuration from one environment to another.
        
        Args:
            service_name: Name of the service
            from_environment: Source environment
            to_environment: Target environment
            keys: Specific keys to promote (None = all)
            validate_target: Whether to validate in target environment
            
        Returns:
            Promotion result
        """
        # Get source configuration
        source_config = await self.environment_configuration_port.get_environment_config(
            from_environment, service_name
        )
        
        if not source_config:
            return {
                "status": "failed",
                "error": f"No configuration found for {service_name} in {from_environment}"
            }
        
        # Filter keys if specified
        config_to_promote = source_config
        if keys:
            config_to_promote = {k: v for k, v in source_config.items() if k in keys}
        
        validation_results = []
        
        # Validate in target environment if requested
        if validate_target:
            for key, value in config_to_promote.items():
                validation = await self.configuration_validator_port.validate_configuration(
                    f"{service_name}.{key}", value
                )
                validation_results.append(validation)
                
                if not validation.is_valid:
                    return {
                        "status": "validation_failed",
                        "service_name": service_name,
                        "from_environment": from_environment,
                        "to_environment": to_environment,
                        "validation_error": validation.error_message,
                        "failed_key": key
                    }
        
        # Perform promotion
        success = await self.environment_configuration_port.promote_configuration(
            from_environment, to_environment, service_name, keys
        )
        
        return {
            "status": "promoted" if success else "failed",
            "service_name": service_name,
            "from_environment": from_environment,
            "to_environment": to_environment,
            "promoted_keys": list(config_to_promote.keys()),
            "validation_results": [
                {
                    "key": v.key,
                    "is_valid": v.is_valid,
                    "error_message": v.error_message
                }
                for v in validation_results
            ],
            "promoted_at": datetime.now().isoformat()
        }
    
    async def get_configuration_audit_trail(
        self,
        service_name: str,
        environment: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get configuration audit trail for a service.
        
        Args:
            service_name: Name of the service
            environment: Optional environment filter
            limit: Maximum number of changes
            
        Returns:
            Configuration audit trail
        """
        # Get configuration history for service keys
        service_prefix = f"{service_name}."
        query = ConfigurationQuery(key_pattern=f"{service_prefix}.*")
        configurations = await self.configuration_provider_port.list_configurations(query)
        
        audit_trail = []
        
        for config in configurations:
            history = await self.configuration_provider_port.get_configuration_history(
                config.key, config.scope, limit
            )
            
            for change in history:
                audit_trail.append({
                    "key": change.key.replace(service_prefix, ""),
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "scope": change.scope.value,
                    "source": change.source.value,
                    "changed_at": change.changed_at.isoformat(),
                    "changed_by": change.changed_by,
                    "reason": change.reason
                })
        
        # Sort by timestamp
        audit_trail.sort(key=lambda x: x["changed_at"], reverse=True)
        
        return {
            "service_name": service_name,
            "environment_filter": environment,
            "total_changes": len(audit_trail),
            "audit_trail": audit_trail[:limit],
            "generated_at": datetime.now().isoformat()
        }
    
    async def validate_environment_configuration(
        self,
        environment: str,
        service_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate all configuration in an environment.
        
        Args:
            environment: Environment name
            service_name: Optional service name filter
            
        Returns:
            Validation results
        """
        env_config = await self.environment_configuration_port.get_environment_config(
            environment, service_name
        )
        
        validation_results = []
        services_validated = []
        
        if service_name:
            # Validate single service
            for key, value in env_config.items():
                validation = await self.configuration_validator_port.validate_configuration(
                    f"{service_name}.{key}", value
                )
                validation_results.append(validation)
            services_validated = [service_name]
        else:
            # Validate all services in environment
            services_config = env_config.get("services", {})
            for svc_name, svc_config in services_config.items():
                for key, value in svc_config.items():
                    validation = await self.configuration_validator_port.validate_configuration(
                        f"{svc_name}.{key}", value
                    )
                    validation_results.append(validation)
                services_validated.append(svc_name)
        
        valid_count = sum(1 for v in validation_results if v.is_valid)
        invalid_count = len(validation_results) - valid_count
        
        return {
            "environment": environment,
            "services_validated": list(set(services_validated)),
            "total_configurations": len(validation_results),
            "valid_configurations": valid_count,
            "invalid_configurations": invalid_count,
            "validation_results": [
                {
                    "key": v.key,
                    "is_valid": v.is_valid,
                    "error_message": v.error_message,
                    "warnings": v.warnings
                }
                for v in validation_results
            ],
            "validation_summary": {
                "passed": invalid_count == 0,
                "health_score": valid_count / len(validation_results) if validation_results else 1.0
            },
            "validated_at": datetime.now().isoformat()
        }