"""Enhanced comprehensive tests for infrastructure configuration - Phase 2 Coverage Improvements."""

from __future__ import annotations

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
import asyncio
from contextlib import contextmanager

from pynomaly.infrastructure.config import Settings, DependencyContainer
from pynomaly.infrastructure.config.container import Container
from pynomaly.domain.exceptions import ConfigurationError


@pytest.fixture
def production_config():
    """Create production-like configuration."""
    return {
        "environment": "production",
        "database": {
            "url": "postgresql://user:pass@db:5432/pynomaly_prod",
            "pool_size": 20,
            "max_overflow": 30,
            "echo": False,
            "ssl_mode": "require"
        },
        "redis": {
            "cluster_nodes": [
                {"host": "redis-1", "port": 6379},
                {"host": "redis-2", "port": 6379},
                {"host": "redis-3", "port": 6379}
            ],
            "password": "${REDIS_PASSWORD}",
            "ssl": True,
            "decode_responses": True
        },
        "security": {
            "jwt_secret": "${JWT_SECRET}",
            "jwt_algorithm": "HS256",
            "jwt_expiration": 3600,
            "password_min_length": 12,
            "require_mfa": True,
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 60,
                "burst_limit": 10
            }
        },
        "monitoring": {
            "enabled": True,
            "metrics_endpoint": "http://prometheus:9090/metrics",
            "health_check_interval": 30,
            "alerting": {
                "slack_webhook": "${SLACK_WEBHOOK}",
                "email_smtp": {
                    "host": "smtp.company.com",
                    "port": 587,
                    "username": "${SMTP_USER}",
                    "password": "${SMTP_PASSWORD}"
                }
            }
        },
        "algorithms": {
            "default_contamination": 0.05,
            "enable_gpu": True,
            "gpu_memory_limit": 0.8,
            "model_cache_size": 100,
            "enable_automl": True,
            "automl_timeout": 3600
        }
    }


@pytest.fixture
def configuration_secrets():
    """Create test secrets for configuration."""
    return {
        "REDIS_PASSWORD": "super_secret_redis_password",
        "JWT_SECRET": "jwt_super_secret_key_for_production",
        "SLACK_WEBHOOK": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
        "SMTP_USER": "alerts@company.com",
        "SMTP_PASSWORD": "smtp_password_123"
    }


class TestSettingsAdvanced:
    """Advanced tests for Settings configuration management."""
    
    def test_environment_specific_configuration_loading(self, production_config):
        """Test loading environment-specific configurations."""
        # Create different environment configs
        environments = {
            "development": {**production_config, "environment": "development", "database": {"url": "sqlite:///dev.db"}},
            "testing": {**production_config, "environment": "testing", "database": {"url": "sqlite:///:memory:"}},
            "staging": {**production_config, "environment": "staging"},
            "production": production_config
        }
        
        for env_name, config in environments.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{env_name}.yaml', delete=False) as f:
                yaml.dump(config, f)
                config_path = f.name
            
            try:
                # Test environment-specific loading
                settings = Settings.load_from_file(config_path, environment=env_name)
                
                assert settings.environment == env_name
                
                if env_name == "development":
                    assert "sqlite" in settings.database["url"]
                    assert settings.database.get("ssl_mode") != "require"
                elif env_name == "production":
                    assert "postgresql" in settings.database["url"]
                    assert settings.database["ssl_mode"] == "require"
                
            finally:
                Path(config_path).unlink()
    
    def test_secret_interpolation_and_resolution(self, production_config, configuration_secrets):
        """Test secret interpolation from environment variables."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(production_config, f)
            config_path = f.name
        
        try:
            # Set environment variables
            original_env = dict(os.environ)
            os.environ.update(configuration_secrets)
            
            try:
                settings = Settings.load_from_file(config_path, resolve_secrets=True)
                
                # Verify secrets were resolved
                assert settings.redis["password"] == configuration_secrets["REDIS_PASSWORD"]
                assert settings.security["jwt_secret"] == configuration_secrets["JWT_SECRET"]
                assert settings.monitoring["alerting"]["slack_webhook"] == configuration_secrets["SLACK_WEBHOOK"]
                
                # Verify nested secret resolution
                smtp_config = settings.monitoring["alerting"]["email_smtp"]
                assert smtp_config["username"] == configuration_secrets["SMTP_USER"]
                assert smtp_config["password"] == configuration_secrets["SMTP_PASSWORD"]
                
            finally:
                # Restore original environment
                os.environ.clear()
                os.environ.update(original_env)
                
        finally:
            Path(config_path).unlink()
    
    def test_configuration_validation_comprehensive(self):
        """Test comprehensive configuration validation."""
        # Test valid configuration
        valid_config = {
            "database": {
                "url": "postgresql://user:pass@localhost/db",
                "pool_size": 10
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
        
        settings = Settings(**valid_config)
        assert settings.validate() is True
        
        # Test invalid configurations
        invalid_configs = [
            # Invalid database URL
            {
                "database": {"url": "invalid_url"},
                "api": {"host": "0.0.0.0", "port": 8000}
            },
            # Invalid port number
            {
                "database": {"url": "sqlite:///test.db"},
                "api": {"host": "0.0.0.0", "port": -1}
            },
            # Missing required fields
            {
                "database": {},  # Missing URL
                "api": {"host": "0.0.0.0", "port": 8000}
            },
            # Invalid data types
            {
                "database": {"url": "sqlite:///test.db", "pool_size": "invalid"},
                "api": {"host": "0.0.0.0", "port": 8000}
            }
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, ConfigurationError)):
                settings = Settings(**invalid_config)
                settings.validate()
    
    def test_configuration_schema_validation(self):
        """Test configuration schema validation against predefined schemas."""
        # Define configuration schema
        schema = {
            "type": "object",
            "properties": {
                "database": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "pattern": r"^(sqlite|postgresql|mysql)://"},
                        "pool_size": {"type": "integer", "minimum": 1, "maximum": 100}
                    },
                    "required": ["url"]
                },
                "api": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535}
                    },
                    "required": ["host", "port"]
                }
            },
            "required": ["database", "api"]
        }
        
        # Test valid configuration against schema
        valid_config = {
            "database": {"url": "postgresql://user:pass@localhost/db", "pool_size": 10},
            "api": {"host": "0.0.0.0", "port": 8000}
        }
        
        settings = Settings(**valid_config)
        assert settings.validate_schema(schema) is True
        
        # Test invalid configuration against schema
        invalid_config = {
            "database": {"url": "invalid://url", "pool_size": 150},  # Invalid URL pattern, pool size too large
            "api": {"host": "0.0.0.0", "port": 70000}  # Port too large
        }
        
        with pytest.raises(ConfigurationError):
            settings = Settings(**invalid_config)
            settings.validate_schema(schema)
    
    def test_configuration_hot_reloading(self, production_config):
        """Test hot reloading of configuration changes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(production_config, f)
            config_path = f.name
        
        try:
            # Initial configuration load
            settings = Settings.load_from_file(config_path, enable_hot_reload=True)
            original_pool_size = settings.database["pool_size"]
            
            # Modify configuration file
            modified_config = production_config.copy()
            modified_config["database"]["pool_size"] = 50
            
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)
            
            # Trigger reload (in real implementation, this would be automatic)
            settings.reload()
            
            # Verify configuration was reloaded
            assert settings.database["pool_size"] == 50
            assert settings.database["pool_size"] != original_pool_size
            
        finally:
            Path(config_path).unlink()
    
    def test_configuration_inheritance_and_overrides(self):
        """Test configuration inheritance and override mechanisms."""
        # Base configuration
        base_config = {
            "database": {"url": "sqlite:///base.db", "pool_size": 5, "echo": False},
            "api": {"host": "localhost", "port": 8000, "debug": False},
            "algorithms": {"default_contamination": 0.1}
        }
        
        # Environment-specific overrides
        dev_overrides = {
            "database": {"echo": True, "pool_size": 2},
            "api": {"debug": True},
            "algorithms": {"enable_gpu": False}
        }
        
        prod_overrides = {
            "database": {"url": "postgresql://prod/db", "pool_size": 20},
            "api": {"host": "0.0.0.0"},
            "algorithms": {"enable_gpu": True, "model_cache_size": 100}
        }
        
        # Test development configuration
        dev_settings = Settings.merge_configs(base_config, dev_overrides)
        assert dev_settings.database["echo"] is True
        assert dev_settings.database["pool_size"] == 2
        assert dev_settings.database["url"] == "sqlite:///base.db"  # Not overridden
        assert dev_settings.api["debug"] is True
        
        # Test production configuration
        prod_settings = Settings.merge_configs(base_config, prod_overrides)
        assert prod_settings.database["url"] == "postgresql://prod/db"
        assert prod_settings.database["pool_size"] == 20
        assert prod_settings.database["echo"] is False  # From base, not overridden
        assert prod_settings.algorithms["enable_gpu"] is True
    
    def test_configuration_encryption_and_security(self, configuration_secrets):
        """Test configuration encryption and security features."""
        # Configuration with sensitive data
        sensitive_config = {
            "database": {
                "url": "postgresql://user:${DB_PASSWORD}@localhost/db"
            },
            "redis": {
                "password": "${REDIS_PASSWORD}"
            },
            "security": {
                "jwt_secret": "${JWT_SECRET}",
                "api_key": "${API_KEY}"
            }
        }
        
        # Set environment variables
        test_secrets = {
            **configuration_secrets,
            "DB_PASSWORD": "database_secret_123",
            "API_KEY": "api_key_secret_456"
        }
        
        original_env = dict(os.environ)
        os.environ.update(test_secrets)
        
        try:
            # Test encrypted storage
            settings = Settings(**sensitive_config)
            settings.resolve_secrets()
            
            # Encrypt sensitive configuration
            encrypted_config = settings.encrypt_sensitive_data()
            
            # Verify sensitive data is encrypted
            assert "${DB_PASSWORD}" not in str(encrypted_config)
            assert test_secrets["DB_PASSWORD"] not in str(encrypted_config)
            
            # Test decryption
            decrypted_settings = Settings.decrypt_sensitive_data(encrypted_config)
            assert decrypted_settings.database["url"].endswith("@localhost/db")
            assert decrypted_settings.redis["password"] == test_secrets["REDIS_PASSWORD"]
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)


class TestDependencyContainerAdvanced:
    """Advanced tests for DependencyContainer functionality."""
    
    def test_service_lifecycle_management(self):
        """Test comprehensive service lifecycle management."""
        container = DependencyContainer()
        
        # Mock services with different lifecycles
        class SingletonService:
            def __init__(self):
                self.initialized_at = "singleton_init"
        
        class TransientService:
            def __init__(self):
                self.initialized_at = f"transient_init_{id(self)}"
        
        class ScopedService:
            def __init__(self):
                self.initialized_at = f"scoped_init_{id(self)}"
        
        # Register services with different lifecycles
        container.register(SingletonService, lifecycle="singleton")
        container.register(TransientService, lifecycle="transient")
        container.register(ScopedService, lifecycle="scoped")
        
        # Test singleton behavior
        singleton1 = container.resolve(SingletonService)
        singleton2 = container.resolve(SingletonService)
        assert singleton1 is singleton2
        assert singleton1.initialized_at == singleton2.initialized_at
        
        # Test transient behavior
        transient1 = container.resolve(TransientService)
        transient2 = container.resolve(TransientService)
        assert transient1 is not transient2
        assert transient1.initialized_at != transient2.initialized_at
        
        # Test scoped behavior (within same scope)
        with container.create_scope() as scope:
            scoped1 = scope.resolve(ScopedService)
            scoped2 = scope.resolve(ScopedService)
            assert scoped1 is scoped2
        
        # Test scoped behavior (different scopes)
        with container.create_scope() as scope1:
            scoped1 = scope1.resolve(ScopedService)
        
        with container.create_scope() as scope2:
            scoped2 = scope2.resolve(ScopedService)
        
        assert scoped1 is not scoped2
    
    def test_conditional_service_registration(self):
        """Test conditional service registration based on configuration."""
        container = DependencyContainer()
        
        # Mock services for different conditions
        class ProductionService:
            def get_environment(self):
                return "production"
        
        class DevelopmentService:
            def get_environment(self):
                return "development"
        
        class RedisCache:
            def __init__(self):
                self.cache_type = "redis"
        
        class InMemoryCache:
            def __init__(self):
                self.cache_type = "memory"
        
        # Test production environment registration
        production_config = {"environment": "production", "cache": {"type": "redis"}}
        container.register_conditional(
            ProductionService,
            condition=lambda config: config.get("environment") == "production",
            config=production_config
        )
        container.register_conditional(
            RedisCache,
            condition=lambda config: config.get("cache", {}).get("type") == "redis",
            config=production_config
        )
        
        # Verify production services are registered
        prod_service = container.resolve(ProductionService)
        assert prod_service.get_environment() == "production"
        
        redis_cache = container.resolve(RedisCache)
        assert redis_cache.cache_type == "redis"
        
        # Test development environment registration
        development_config = {"environment": "development", "cache": {"type": "memory"}}
        container_dev = DependencyContainer()
        container_dev.register_conditional(
            DevelopmentService,
            condition=lambda config: config.get("environment") == "development",
            config=development_config
        )
        container_dev.register_conditional(
            InMemoryCache,
            condition=lambda config: config.get("cache", {}).get("type") == "memory",
            config=development_config
        )
        
        dev_service = container_dev.resolve(DevelopmentService)
        assert dev_service.get_environment() == "development"
        
        memory_cache = container_dev.resolve(InMemoryCache)
        assert memory_cache.cache_type == "memory"
    
    def test_complex_dependency_injection_scenarios(self):
        """Test complex dependency injection scenarios."""
        container = DependencyContainer()
        
        # Define services with complex dependencies
        class DatabaseService:
            def __init__(self, connection_string: str):
                self.connection_string = connection_string
        
        class CacheService:
            def __init__(self, host: str, port: int):
                self.host = host
                self.port = port
        
        class LoggingService:
            def __init__(self, level: str = "INFO"):
                self.level = level
        
        class BusinessService:
            def __init__(self, db: DatabaseService, cache: CacheService, logger: LoggingService):
                self.db = db
                self.cache = cache
                self.logger = logger
        
        class ApplicationService:
            def __init__(self, business: BusinessService, config: dict):
                self.business = business
                self.config = config
        
        # Register services with complex dependency graphs
        container.register(
            DatabaseService,
            factory=lambda: DatabaseService("postgresql://localhost/db")
        )
        container.register(
            CacheService,
            factory=lambda: CacheService("localhost", 6379)
        )
        container.register(LoggingService)
        container.register(BusinessService)  # Auto-inject dependencies
        
        app_config = {"app_name": "pynomaly", "version": "1.0.0"}
        container.register(
            ApplicationService,
            factory=lambda business: ApplicationService(business, app_config)
        )
        
        # Resolve complex service graph
        app_service = container.resolve(ApplicationService)
        
        # Verify dependency injection worked correctly
        assert app_service.business is not None
        assert app_service.business.db.connection_string == "postgresql://localhost/db"
        assert app_service.business.cache.host == "localhost"
        assert app_service.business.cache.port == 6379
        assert app_service.business.logger.level == "INFO"
        assert app_service.config["app_name"] == "pynomaly"
    
    def test_circular_dependency_detection(self):
        """Test detection and handling of circular dependencies."""
        container = DependencyContainer()
        
        # Define services with circular dependencies
        class ServiceA:
            def __init__(self, service_b):
                self.service_b = service_b
        
        class ServiceB:
            def __init__(self, service_c):
                self.service_c = service_c
        
        class ServiceC:
            def __init__(self, service_a):
                self.service_a = service_a
        
        # Register services with circular dependencies
        container.register(ServiceA)
        container.register(ServiceB)
        container.register(ServiceC)
        
        # Should detect circular dependency and raise error
        with pytest.raises(ConfigurationError):
            container.resolve(ServiceA)
    
    def test_container_performance_and_caching(self):
        """Test container performance and service resolution caching."""
        container = DependencyContainer()
        
        # Create a service that's expensive to initialize
        class ExpensiveService:
            def __init__(self):
                # Simulate expensive initialization
                import time
                time.sleep(0.01)  # 10ms
                self.initialized = True
        
        container.register(ExpensiveService, lifecycle="singleton")
        
        # Measure first resolution (should be slow)
        import time
        start_time = time.time()
        service1 = container.resolve(ExpensiveService)
        first_resolution_time = time.time() - start_time
        
        # Measure second resolution (should be fast due to caching)
        start_time = time.time()
        service2 = container.resolve(ExpensiveService)
        second_resolution_time = time.time() - start_time
        
        # Verify caching worked
        assert service1 is service2
        assert service1.initialized is True
        assert second_resolution_time < first_resolution_time / 2  # At least 2x faster
    
    def test_container_configuration_integration(self, production_config):
        """Test integration between container and configuration management."""
        container = DependencyContainer()
        
        # Create configuration-aware services
        class DatabaseService:
            def __init__(self, config: dict):
                self.url = config.get("database", {}).get("url")
                self.pool_size = config.get("database", {}).get("pool_size", 10)
        
        class ApiService:
            def __init__(self, config: dict):
                self.host = config.get("api", {}).get("host", "localhost")
                self.port = config.get("api", {}).get("port", 8000)
        
        # Register services with configuration injection
        container.register_with_config(DatabaseService, production_config)
        container.register_with_config(ApiService, production_config)
        
        # Resolve services and verify configuration was injected
        db_service = container.resolve(DatabaseService)
        api_service = container.resolve(ApiService)
        
        assert db_service.url == production_config["database"]["url"]
        assert db_service.pool_size == production_config["database"]["pool_size"]
        assert api_service.host == production_config.get("api", {}).get("host", "localhost")
        assert api_service.port == production_config.get("api", {}).get("port", 8000)


class TestConfigurationIntegration:
    """Integration tests for configuration components."""
    
    def test_end_to_end_configuration_flow(self, production_config, configuration_secrets):
        """Test complete end-to-end configuration flow."""
        # Step 1: Create configuration file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(production_config, f)
            config_path = f.name
        
        try:
            # Step 2: Set up environment variables
            original_env = dict(os.environ)
            os.environ.update(configuration_secrets)
            
            try:
                # Step 3: Load and validate configuration
                settings = Settings.load_from_file(config_path, resolve_secrets=True)
                assert settings.validate() is True
                
                # Step 4: Initialize dependency container with configuration
                container = DependencyContainer(config=settings.to_dict())
                
                # Step 5: Register application services
                class DatabaseConnection:
                    def __init__(self, url: str, pool_size: int):
                        self.url = url
                        self.pool_size = pool_size
                        self.connected = True
                
                class RedisConnection:
                    def __init__(self, host: str, port: int, password: str):
                        self.host = host
                        self.port = port
                        self.password = password
                        self.connected = True
                
                class AnomalyDetectionService:
                    def __init__(self, db: DatabaseConnection, cache: RedisConnection):
                        self.db = db
                        self.cache = cache
                        self.ready = True
                
                # Register services with configuration
                container.register_factory(
                    DatabaseConnection,
                    lambda: DatabaseConnection(
                        settings.database["url"],
                        settings.database["pool_size"]
                    )
                )
                
                container.register_factory(
                    RedisConnection,
                    lambda: RedisConnection(
                        settings.redis["cluster_nodes"][0]["host"],
                        settings.redis["cluster_nodes"][0]["port"],
                        settings.redis["password"]
                    )
                )
                
                container.register(AnomalyDetectionService)
                
                # Step 6: Resolve application service
                app_service = container.resolve(AnomalyDetectionService)
                
                # Step 7: Verify end-to-end configuration flow
                assert app_service.ready is True
                assert app_service.db.connected is True
                assert app_service.cache.connected is True
                assert app_service.db.url == production_config["database"]["url"]
                assert app_service.cache.password == configuration_secrets["REDIS_PASSWORD"]
                
            finally:
                # Restore original environment
                os.environ.clear()
                os.environ.update(original_env)
                
        finally:
            Path(config_path).unlink()
    
    @pytest.mark.asyncio
    async def test_async_configuration_loading(self, production_config):
        """Test asynchronous configuration loading and service initialization."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(production_config, f)
            config_path = f.name
        
        try:
            # Test async configuration loading
            settings = await Settings.load_from_file_async(config_path)
            assert settings.database["url"] == production_config["database"]["url"]
            
            # Test async service initialization
            class AsyncDatabaseService:
                def __init__(self, config: dict):
                    self.config = config
                    self.initialized = False
                
                async def initialize(self):
                    # Simulate async initialization
                    await asyncio.sleep(0.01)
                    self.initialized = True
                    return self
            
            container = DependencyContainer()
            
            # Register async service
            async def create_async_db_service():
                service = AsyncDatabaseService(settings.to_dict())
                await service.initialize()
                return service
            
            container.register_async_factory(AsyncDatabaseService, create_async_db_service)
            
            # Resolve async service
            db_service = await container.resolve_async(AsyncDatabaseService)
            assert db_service.initialized is True
            assert db_service.config["database"]["url"] == production_config["database"]["url"]
            
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])