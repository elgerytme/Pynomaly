"""File-based implementations for configuration management operations."""

import json
import asyncio
import hashlib
import time
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

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


class FileBasedConfigurationProvider(ConfigurationProviderPort):
    """File-based configuration provider implementation."""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_config_files()
    
    def _initialize_config_files(self):
        """Initialize configuration files if they don't exist."""
        for scope in ConfigurationScope:
            scope_file = self.config_dir / f"{scope.value}.json"
            if not scope_file.exists():
                scope_file.write_text(json.dumps({}))
    
    def _get_config_file(self, scope: ConfigurationScope) -> Path:
        """Get configuration file path for scope."""
        return self.config_dir / f"{scope.value}.json"
    
    def _get_history_file(self, scope: ConfigurationScope) -> Path:
        """Get configuration history file path for scope."""
        return self.config_dir / f"{scope.value}_history.json"
    
    async def _load_config(self, scope: ConfigurationScope) -> Dict[str, Any]:
        """Load configuration for scope."""
        try:
            config_file = self._get_config_file(scope)
            return json.loads(config_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    async def _save_config(self, scope: ConfigurationScope, config: Dict[str, Any]):
        """Save configuration for scope."""
        config_file = self._get_config_file(scope)
        config_file.write_text(json.dumps(config, indent=2, default=str))
    
    async def _log_change(
        self, 
        key: str, 
        old_value: Any, 
        new_value: Any, 
        scope: ConfigurationScope
    ):
        """Log configuration change to history."""
        try:
            history_file = self._get_history_file(scope)
            history = []
            
            if history_file.exists():
                history = json.loads(history_file.read_text())
            
            change = {
                "key": key,
                "old_value": old_value,
                "new_value": new_value,
                "scope": scope.value,
                "source": ConfigurationSource.FILE.value,
                "changed_at": datetime.now().isoformat(),
                "changed_by": "system"
            }
            
            history.append(change)
            
            # Keep only last 1000 changes
            if len(history) > 1000:
                history = history[-1000:]
            
            history_file.write_text(json.dumps(history, indent=2))
        except Exception:
            pass  # Don't fail configuration operations due to history logging
    
    async def get_configuration(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        default: Any = None
    ) -> Any:
        """Get a configuration value."""
        try:
            config = await self._load_config(scope)
            return config.get(key, default)
        except Exception:
            return default
    
    async def set_configuration(
        self, 
        key: str, 
        value: Any,
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        tags: Optional[Dict[str, str]] = None,
        encrypt: bool = False
    ) -> bool:
        """Set a configuration value."""
        try:
            config = await self._load_config(scope)
            old_value = config.get(key)
            
            if encrypt:
                # Simple encryption (in production, use proper encryption)
                value_str = json.dumps(value) if not isinstance(value, str) else value
                encrypted_value = hashlib.sha256(value_str.encode()).hexdigest()
                config[key] = {"_encrypted": True, "value": encrypted_value}
            else:
                config[key] = value
            
            await self._save_config(scope, config)
            await self._log_change(key, old_value, value, scope)
            
            return True
        except Exception:
            return False
    
    async def delete_configuration(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> bool:
        """Delete a configuration value."""
        try:
            config = await self._load_config(scope)
            if key in config:
                old_value = config[key]
                del config[key]
                await self._save_config(scope, config)
                await self._log_change(key, old_value, None, scope)
            return True
        except Exception:
            return False
    
    async def list_configurations(
        self, 
        query: ConfigurationQuery
    ) -> List[ConfigurationValue]:
        """List configuration values matching query."""
        try:
            results = []
            scopes_to_check = [query.scope] if query.scope else list(ConfigurationScope)
            
            for scope in scopes_to_check:
                config = await self._load_config(scope)
                
                for key, value in config.items():
                    # Apply key pattern filter
                    if query.key_pattern and not re.match(query.key_pattern, key):
                        continue
                    
                    # Skip encrypted values if not requested
                    if isinstance(value, dict) and value.get("_encrypted") and not query.include_encrypted:
                        continue
                    
                    config_value = ConfigurationValue(
                        key=key,
                        value=value,
                        scope=scope,
                        source=ConfigurationSource.FILE,
                        version=1,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        encrypted=isinstance(value, dict) and value.get("_encrypted", False)
                    )
                    
                    results.append(config_value)
            
            return results
        except Exception:
            return []
    
    async def get_configuration_history(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        limit: int = 50
    ) -> List[ConfigurationChange]:
        """Get configuration change history."""
        try:
            history_file = self._get_history_file(scope)
            if not history_file.exists():
                return []
            
            history = json.loads(history_file.read_text())
            
            # Filter by key and limit
            key_history = [h for h in history if h["key"] == key][-limit:]
            
            changes = []
            for change_dict in key_history:
                change = ConfigurationChange(
                    key=change_dict["key"],
                    old_value=change_dict["old_value"],
                    new_value=change_dict["new_value"],
                    scope=ConfigurationScope(change_dict["scope"]),
                    source=ConfigurationSource(change_dict["source"]),
                    changed_at=datetime.fromisoformat(change_dict["changed_at"]),
                    changed_by=change_dict.get("changed_by"),
                    reason=change_dict.get("reason")
                )
                changes.append(change)
            
            return changes
        except Exception:
            return []


class FileBasedConfigurationWatcher(ConfigurationWatcherPort):
    """File-based configuration watcher implementation."""
    
    def __init__(self, config_provider: FileBasedConfigurationProvider):
        self.config_provider = config_provider
        self._watches: Dict[str, Dict[str, Any]] = {}
        self._watch_tasks: Dict[str, asyncio.Task] = {}
        self._file_checksums: Dict[str, str] = {}
    
    def _get_file_checksum(self, file_path: Path) -> str:
        """Get file checksum for change detection."""
        try:
            if file_path.exists():
                content = file_path.read_text()
                return hashlib.md5(content.encode()).hexdigest()
            return ""
        except Exception:
            return ""
    
    async def watch_configuration(
        self, 
        key: str, 
        callback: Callable[[ConfigurationChange], None],
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> str:
        """Watch for configuration changes."""
        watch_id = f"config_watch_{key}_{scope.value}_{int(time.time())}"
        
        self._watches[watch_id] = {
            "key": key,
            "scope": scope,
            "callback": callback,
            "last_value": await self.config_provider.get_configuration(key, scope)
        }
        
        # Start watch task
        task = asyncio.create_task(self._watch_loop(watch_id))
        self._watch_tasks[watch_id] = task
        
        return watch_id
    
    async def watch_configuration_prefix(
        self, 
        prefix: str, 
        callback: Callable[[List[ConfigurationChange]], None],
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> str:
        """Watch for configuration changes with key prefix."""
        watch_id = f"prefix_watch_{prefix}_{scope.value}_{int(time.time())}"
        
        # Get initial values
        query = ConfigurationQuery(key_pattern=f"{prefix}.*", scope=scope)
        initial_configs = await self.config_provider.list_configurations(query)
        initial_values = {config.key: config.value for config in initial_configs}
        
        self._watches[watch_id] = {
            "prefix": prefix,
            "scope": scope,
            "callback": callback,
            "last_values": initial_values
        }
        
        # Start watch task
        task = asyncio.create_task(self._prefix_watch_loop(watch_id))
        self._watch_tasks[watch_id] = task
        
        return watch_id
    
    async def _watch_loop(self, watch_id: str):
        """Watch loop for single configuration changes."""
        watch_info = self._watches[watch_id]
        key = watch_info["key"]
        scope = watch_info["scope"]
        callback = watch_info["callback"]
        
        while watch_id in self._watches:
            try:
                current_value = await self.config_provider.get_configuration(key, scope)
                last_value = watch_info["last_value"]
                
                if current_value != last_value:
                    change = ConfigurationChange(
                        key=key,
                        old_value=last_value,
                        new_value=current_value,
                        scope=scope,
                        source=ConfigurationSource.FILE,
                        changed_at=datetime.now()
                    )
                    
                    watch_info["last_value"] = current_value
                    callback(change)
                
                await asyncio.sleep(2)  # Check every 2 seconds
            except Exception:
                await asyncio.sleep(2)
    
    async def _prefix_watch_loop(self, watch_id: str):
        """Watch loop for prefix configuration changes."""
        watch_info = self._watches[watch_id]
        prefix = watch_info["prefix"]
        scope = watch_info["scope"]
        callback = watch_info["callback"]
        
        while watch_id in self._watches:
            try:
                query = ConfigurationQuery(key_pattern=f"{prefix}.*", scope=scope)
                current_configs = await self.config_provider.list_configurations(query)
                current_values = {config.key: config.value for config in current_configs}
                last_values = watch_info["last_values"]
                
                changes = []
                
                # Check for changes and new keys
                for key, value in current_values.items():
                    if key not in last_values or last_values[key] != value:
                        change = ConfigurationChange(
                            key=key,
                            old_value=last_values.get(key),
                            new_value=value,
                            scope=scope,
                            source=ConfigurationSource.FILE,
                            changed_at=datetime.now()
                        )
                        changes.append(change)
                
                # Check for deleted keys
                for key in last_values:
                    if key not in current_values:
                        change = ConfigurationChange(
                            key=key,
                            old_value=last_values[key],
                            new_value=None,
                            scope=scope,
                            source=ConfigurationSource.FILE,
                            changed_at=datetime.now()
                        )
                        changes.append(change)
                
                if changes:
                    watch_info["last_values"] = current_values
                    callback(changes)
                
                await asyncio.sleep(2)  # Check every 2 seconds
            except Exception:
                await asyncio.sleep(2)
    
    async def cancel_watch(self, watch_id: str) -> bool:
        """Cancel a configuration watch."""
        if watch_id in self._watches:
            del self._watches[watch_id]
        
        if watch_id in self._watch_tasks:
            task = self._watch_tasks[watch_id]
            task.cancel()
            del self._watch_tasks[watch_id]
            return True
        
        return False


class FileBasedConfigurationValidator(ConfigurationValidatorPort):
    """File-based configuration validator implementation."""
    
    def __init__(self, validator_dir: str):
        self.validator_dir = Path(validator_dir)
        self.validator_dir.mkdir(parents=True, exist_ok=True)
        self.schemas_file = self.validator_dir / "schemas.json"
        self._initialize_schemas()
    
    def _initialize_schemas(self):
        """Initialize schemas file if it doesn't exist."""
        if not self.schemas_file.exists():
            self.schemas_file.write_text(json.dumps({}))
    
    async def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load validation schemas."""
        try:
            return json.loads(self.schemas_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    async def _save_schemas(self, schemas: Dict[str, Dict[str, Any]]):
        """Save validation schemas."""
        self.schemas_file.write_text(json.dumps(schemas, indent=2))
    
    async def validate_configuration(
        self, 
        key: str, 
        value: Any, 
        schema: Optional[str] = None
    ) -> ConfigurationValidation:
        """Validate a configuration value."""
        try:
            if not schema:
                # Try to find schema for key
                schemas = await self._load_schemas()
                schema_info = schemas.get(key)
                if not schema_info:
                    # No schema found, assume valid
                    return ConfigurationValidation(
                        key=key,
                        is_valid=True,
                        validated_at=datetime.now()
                    )
                schema = schema_info["schema"]
                schema_type = schema_info.get("schema_type", "json_schema")
            else:
                schema_type = "json_schema"
            
            if schema_type == "json_schema" and HAS_JSONSCHEMA:
                try:
                    schema_dict = json.loads(schema) if isinstance(schema, str) else schema
                    jsonschema.validate(value, schema_dict)
                    return ConfigurationValidation(
                        key=key,
                        is_valid=True,
                        validated_at=datetime.now()
                    )
                except jsonschema.ValidationError as e:
                    return ConfigurationValidation(
                        key=key,
                        is_valid=False,
                        error_message=str(e),
                        validated_at=datetime.now()
                    )
            elif schema_type == "json_schema" and not HAS_JSONSCHEMA:
                # No jsonschema available, assume valid
                return ConfigurationValidation(
                    key=key,
                    is_valid=True,
                    warnings=["jsonschema library not available, skipping validation"],
                    validated_at=datetime.now()
                )
            
            elif schema_type == "regex":
                if isinstance(value, str) and re.match(schema, value):
                    return ConfigurationValidation(
                        key=key,
                        is_valid=True,
                        validated_at=datetime.now()
                    )
                else:
                    return ConfigurationValidation(
                        key=key,
                        is_valid=False,
                        error_message=f"Value does not match regex pattern: {schema}",
                        validated_at=datetime.now()
                    )
            
            # Unknown schema type, assume valid
            return ConfigurationValidation(
                key=key,
                is_valid=True,
                warnings=[f"Unknown schema type: {schema_type}"],
                validated_at=datetime.now()
            )
            
        except Exception as e:
            return ConfigurationValidation(
                key=key,
                is_valid=False,
                error_message=f"Validation error: {str(e)}",
                validated_at=datetime.now()
            )
    
    async def validate_all_configurations(
        self, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> List[ConfigurationValidation]:
        """Validate all configurations in a scope."""
        try:
            # This would require access to the configuration provider
            # For simplicity, return empty list - in practice, inject the provider
            return []
        except Exception:
            return []
    
    async def register_validation_schema(
        self, 
        key: str, 
        schema: str, 
        schema_type: str = "json_schema"
    ) -> bool:
        """Register a validation schema for a configuration key."""
        try:
            schemas = await self._load_schemas()
            schemas[key] = {
                "schema": schema,
                "schema_type": schema_type,
                "registered_at": datetime.now().isoformat()
            }
            await self._save_schemas(schemas)
            return True
        except Exception:
            return False


class FileBasedSecretManagement(SecretManagementPort):
    """File-based secret management implementation."""
    
    def __init__(self, secrets_dir: str):
        self.secrets_dir = Path(secrets_dir)
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        # In production, use proper encryption key management
        self.encryption_key = hashlib.sha256(b"mlops_secret_key").digest()[:16]
    
    def _encrypt_value(self, value: str) -> str:
        """Simple encryption (use proper encryption in production)."""
        return hashlib.sha256((value + "salt").encode()).hexdigest()
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Simple decryption placeholder (not reversible in this implementation)."""
        # In production, use proper reversible encryption
        return "[ENCRYPTED_VALUE]"
    
    def _get_secrets_file(self, scope: ConfigurationScope) -> Path:
        """Get secrets file path for scope."""
        return self.secrets_dir / f"{scope.value}_secrets.json"
    
    async def _load_secrets(self, scope: ConfigurationScope) -> Dict[str, Any]:
        """Load secrets for scope."""
        try:
            secrets_file = self._get_secrets_file(scope)
            if secrets_file.exists():
                return json.loads(secrets_file.read_text())
            return {}
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    async def _save_secrets(self, scope: ConfigurationScope, secrets: Dict[str, Any]):
        """Save secrets for scope."""
        secrets_file = self._get_secrets_file(scope)
        secrets_file.write_text(json.dumps(secrets, indent=2))
        # Set restrictive permissions
        secrets_file.chmod(0o600)
    
    async def store_secret(
        self, 
        key: str, 
        value: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a secret value."""
        try:
            secrets = await self._load_secrets(scope)
            encrypted_value = self._encrypt_value(value)
            
            secrets[key] = {
                "value": encrypted_value,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            await self._save_secrets(scope, secrets)
            return True
        except Exception:
            return False
    
    async def get_secret(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> Optional[str]:
        """Get a secret value."""
        try:
            secrets = await self._load_secrets(scope)
            secret_info = secrets.get(key)
            if secret_info:
                # In production, properly decrypt the value
                return self._decrypt_value(secret_info["value"])
            return None
        except Exception:
            return None
    
    async def delete_secret(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> bool:
        """Delete a secret value."""
        try:
            secrets = await self._load_secrets(scope)
            if key in secrets:
                del secrets[key]
                await self._save_secrets(scope, secrets)
            return True
        except Exception:
            return False
    
    async def list_secret_keys(
        self, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> List[str]:
        """List secret keys (not values)."""
        try:
            secrets = await self._load_secrets(scope)
            return list(secrets.keys())
        except Exception:
            return []
    
    async def rotate_secret(
        self, 
        key: str, 
        new_value: str,
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> bool:
        """Rotate a secret value."""
        try:
            secrets = await self._load_secrets(scope)
            if key in secrets:
                encrypted_value = self._encrypt_value(new_value)
                secrets[key]["value"] = encrypted_value
                secrets[key]["updated_at"] = datetime.now().isoformat()
                secrets[key]["rotated_at"] = datetime.now().isoformat()
                await self._save_secrets(scope, secrets)
                return True
            return False
        except Exception:
            return False


class FileBasedEnvironmentConfiguration(EnvironmentConfigurationPort):
    """File-based environment configuration implementation."""
    
    def __init__(self, env_dir: str):
        self.env_dir = Path(env_dir)
        self.env_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_environments()
    
    def _initialize_environments(self):
        """Initialize environment files."""
        environments = ["development", "staging", "production"]
        for env in environments:
            env_file = self.env_dir / f"{env}.json"
            if not env_file.exists():
                env_file.write_text(json.dumps({}))
    
    def _get_env_file(self, environment: str) -> Path:
        """Get environment configuration file."""
        return self.env_dir / f"{environment}.json"
    
    async def _load_env_config(self, environment: str) -> Dict[str, Any]:
        """Load environment configuration."""
        try:
            env_file = self._get_env_file(environment)
            if env_file.exists():
                return json.loads(env_file.read_text())
            return {}
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    async def _save_env_config(self, environment: str, config: Dict[str, Any]):
        """Save environment configuration."""
        env_file = self._get_env_file(environment)
        env_file.write_text(json.dumps(config, indent=2, default=str))
    
    async def get_environment_config(
        self, 
        environment: str, 
        service_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get configuration for an environment."""
        try:
            config = await self._load_env_config(environment)
            if service_name:
                return config.get("services", {}).get(service_name, {})
            return config
        except Exception:
            return {}
    
    async def set_environment_config(
        self, 
        environment: str, 
        config: Dict[str, Any],
        service_name: Optional[str] = None
    ) -> bool:
        """Set configuration for an environment."""
        try:
            env_config = await self._load_env_config(environment)
            
            if service_name:
                if "services" not in env_config:
                    env_config["services"] = {}
                env_config["services"][service_name] = config
            else:
                env_config.update(config)
            
            env_config["updated_at"] = datetime.now().isoformat()
            await self._save_env_config(environment, env_config)
            return True
        except Exception:
            return False
    
    async def promote_configuration(
        self, 
        from_environment: str, 
        to_environment: str,
        service_name: Optional[str] = None,
        keys: Optional[List[str]] = None
    ) -> bool:
        """Promote configuration from one environment to another."""
        try:
            source_config = await self.get_environment_config(from_environment, service_name)
            
            if keys:
                # Promote only specific keys
                config_to_promote = {k: v for k, v in source_config.items() if k in keys}
            else:
                # Promote all configuration
                config_to_promote = source_config
            
            # Merge with target environment
            target_config = await self.get_environment_config(to_environment, service_name)
            target_config.update(config_to_promote)
            
            return await self.set_environment_config(to_environment, target_config, service_name)
        except Exception:
            return False
    
    async def list_environments(self) -> List[str]:
        """List available environments."""
        try:
            environments = []
            for env_file in self.env_dir.glob("*.json"):
                env_name = env_file.stem
                environments.append(env_name)
            return sorted(environments)
        except Exception:
            return []