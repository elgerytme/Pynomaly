"""Integration tests for DI container with enhanced settings."""

import os
import pytest
from unittest.mock import patch, Mock
from pathlib import Path

from pynomaly.infrastructure.config.container import Container, create_container
from pynomaly.infrastructure.config.settings import Settings, SecretsBackend


class TestContainerIntegration:
    """Test DI container integration with enhanced settings."""

    def test_container_creation_with_secrets(self):
        """Test container creation with secrets management."""
        with patch.dict(os.environ, {
            "PYNOMALY_SECRET_KEY": "test-secret-key",
            "PYNOMALY_DATABASE_URL": "sqlite:///test.db"
        }):
            container = create_container(testing=False)
            
            # Verify container is properly configured
            assert container is not None
            assert hasattr(container, 'config')
            
            # Test settings integration
            settings = container.config()
            assert settings.get_secret_key() == "test-secret-key"
            assert settings.get_database_url() == "sqlite:///test.db"

    def test_container_with_multiple_secret_backends(self):
        """Test container with multiple secret backends."""
        with patch.dict(os.environ, {
            "PYNOMALY_SECRETS_BACKENDS": '["env", "aws_ssm"]',
            "PYNOMALY_AWS_REGION": "us-west-2"
        }):
            container = create_container(testing=False)
            settings = container.config()
            
            # Should have multiple backends configured
            assert SecretsBackend.ENV in settings.secrets_backends
            assert settings.aws_region == "us-west-2"

    def test_container_repository_creation_with_secrets(self):
        """Test repository creation with secrets-based database URL."""
        with patch.dict(os.environ, {
            "PYNOMALY_DATABASE_URL": "postgresql://user:secret@localhost/test_db",
            "PYNOMALY_USE_DATABASE_REPOSITORIES": "true"
        }):
            container = create_container(testing=False)
            
            # Test that repositories can be created
            detector_repo = container.detector_repository()
            dataset_repo = container.dataset_repository()
            result_repo = container.result_repository()
            
            assert detector_repo is not None
            assert dataset_repo is not None
            assert result_repo is not None

    @patch('boto3.client')
    def test_container_with_aws_secrets(self, mock_boto3_client):
        """Test container configuration with AWS secrets."""
        # Mock AWS client
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            'SecretString': '{"SECRET_KEY": "aws-secret-key", "DATABASE_URL": "postgresql://aws-db"}'
        }
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, {
            "PYNOMALY_SECRETS_BACKENDS": '["aws_secrets_manager", "env"]',
            "PYNOMALY_AWS_REGION": "us-east-1"
        }):
            container = create_container(testing=False)
            settings = container.config()
            
            # Verify AWS region is set
            assert settings.aws_region == "us-east-1"
            assert SecretsBackend.AWS_SECRETS_MANAGER in settings.secrets_backends

    def test_container_service_injection_with_secrets(self):
        """Test that services are properly injected with secrets-aware configuration."""
        with patch.dict(os.environ, {
            "PYNOMALY_SECRET_KEY": "injection-test-key",
            "PYNOMALY_API_RATE_LIMIT": "200"
        }):
            container = create_container(testing=False)
            
            # Test core services
            detection_service = container.detection_service()
            ensemble_service = container.ensemble_service()
            
            assert detection_service is not None
            assert ensemble_service is not None
            
            # Verify settings are injected correctly
            settings = container.config()
            assert settings.get_secret_key() == "injection-test-key"
            assert settings.api_rate_limit == 200

    def test_container_optional_services_with_redis_secrets(self):
        """Test optional services with Redis URL from secrets."""
        with patch.dict(os.environ, {
            "PYNOMALY_REDIS_URL": "redis://secret-redis-host:6379/1"
        }):
            container = create_container(testing=False)
            settings = container.config()
            
            # Test Redis URL retrieval
            redis_url = settings.get_redis_url()
            assert redis_url == "redis://secret-redis-host:6379/1"

    def test_container_environment_specific_configuration(self):
        """Test container with environment-specific configuration."""
        # Test development environment
        with patch.dict(os.environ, {
            "PYNOMALY_ENV": "development",
            "PYNOMALY_APP__DEBUG": "true"
        }):
            container = create_container(testing=False)
            settings = container.config()
            
            assert settings.app.environment == "development"
            assert settings.app.debug is True
            assert not settings.is_production

    def test_container_production_configuration(self):
        """Test container with production configuration."""
        with patch.dict(os.environ, {
            "PYNOMALY_ENV": "production",
            "PYNOMALY_APP__DEBUG": "false",
            "PYNOMALY_AUTH_ENABLED": "true",
            "PYNOMALY_SECRETS_BACKENDS": '["aws_secrets_manager", "env"]'
        }):
            container = create_container(testing=False)
            settings = container.config()
            
            assert settings.app.environment == "production"
            assert settings.app.debug is False
            assert settings.auth_enabled is True

    def test_test_container_override(self):
        """Test that test container properly overrides settings."""
        test_container = create_container(testing=True)
        settings = test_container.config()
        
        # Test container should have test-specific settings
        assert "test" in str(settings.storage_path).lower()
        assert settings.app.debug is True
        assert settings.auth_enabled is False

    def test_container_wiring_with_secrets(self):
        """Test that container wiring works with secrets-aware settings."""
        with patch.dict(os.environ, {
            "PYNOMALY_SECRET_KEY": "wiring-test-key"
        }):
            container = create_container(testing=False)
            
            # Verify container is properly wired
            # This should not raise any exceptions
            container.wire(modules=["pynomaly.presentation.api"])
            
            # Test that we can access settings after wiring
            settings = container.config()
            assert settings.get_secret_key() == "wiring-test-key"

    def test_container_graceful_degradation(self):
        """Test that container gracefully handles missing optional dependencies."""
        # Test with no special environment setup
        container = create_container(testing=False)
        
        # Should still create basic services even if optional ones fail
        detection_service = container.detection_service()
        assert detection_service is not None
        
        # Settings should work with default values
        settings = container.config()
        assert settings is not None


class TestContainerPerformance:
    """Test container performance and caching."""

    def test_singleton_behavior_with_secrets(self):
        """Test that singleton services maintain secrets across calls."""
        with patch.dict(os.environ, {
            "PYNOMALY_SECRET_KEY": "singleton-test-key"
        }):
            container = create_container(testing=False)
            
            # Get settings multiple times
            settings1 = container.config()
            settings2 = container.config()
            
            # Should be the same instance (singleton)
            assert settings1 is settings2
            assert settings1.get_secret_key() == "singleton-test-key"

    def test_container_initialization_time(self):
        """Test that container initializes quickly even with secrets."""
        import time
        
        start_time = time.time()
        container = create_container(testing=False)
        end_time = time.time()
        
        # Should initialize quickly (less than 1 second)
        initialization_time = end_time - start_time
        assert initialization_time < 1.0
        
        # Should be able to access services
        assert container.config() is not None


class TestContainerErrorHandling:
    """Test container error handling."""

    def test_container_with_invalid_secrets_backend(self):
        """Test container behavior with invalid secrets backend."""
        with patch.dict(os.environ, {
            "PYNOMALY_SECRETS_BACKENDS": '["invalid_backend"]'
        }):
            # Should not raise exception, should fall back gracefully
            container = create_container(testing=False)
            settings = container.config()
            
            # Should have at least environment backend
            assert SecretsBackend.ENV in settings.secrets_backends

    def test_container_with_aws_credentials_missing(self):
        """Test container when AWS credentials are missing."""
        # Clear AWS environment variables
        aws_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
        with patch.dict(os.environ, {var: '' for var in aws_vars}, clear=True):
            with patch.dict(os.environ, {
                "PYNOMALY_SECRETS_BACKENDS": '["aws_secrets_manager", "env"]'
            }):
                # Should not raise exception
                container = create_container(testing=False)
                settings = container.config()
                
                # Should fall back to environment variables
                assert settings is not None

    def test_container_with_corrupted_database_url(self):
        """Test container with corrupted database URL."""
        with patch.dict(os.environ, {
            "PYNOMALY_DATABASE_URL": "invalid://corrupted-url",
            "PYNOMALY_USE_DATABASE_REPOSITORIES": "true"
        }):
            # Should not raise exception during container creation
            container = create_container(testing=False)
            
            # Should fall back to file repositories
            detector_repo = container.detector_repository()
            assert detector_repo is not None


class TestContainerSecurityFeatures:
    """Test container security features."""

    def test_container_secret_masking(self):
        """Test that secrets are properly masked in logs/output."""
        with patch.dict(os.environ, {
            "PYNOMALY_SECRET_KEY": "super-secret-key-123"
        }):
            container = create_container(testing=False)
            settings = container.config()
            
            # Direct access should return SecretStr
            assert hasattr(settings.secret_key, 'get_secret_value')
            
            # String representation should be masked
            assert "super-secret-key-123" not in str(settings.secret_key)

    def test_container_environment_isolation(self):
        """Test that different environments are properly isolated."""
        # Test development environment
        with patch.dict(os.environ, {
            "PYNOMALY_ENV": "development",
            "PYNOMALY_SECRET_KEY": "dev-secret"
        }):
            dev_container = create_container(testing=False)
            dev_settings = dev_container.config()
            
            assert dev_settings.app.environment == "development"
            assert dev_settings.get_secret_key() == "dev-secret"
        
        # Test production environment
        with patch.dict(os.environ, {
            "PYNOMALY_ENV": "production",
            "PYNOMALY_SECRET_KEY": "prod-secret"
        }):
            prod_container = create_container(testing=False)
            prod_settings = prod_container.config()
            
            assert prod_settings.app.environment == "production"
            assert prod_settings.get_secret_key() == "prod-secret"


if __name__ == "__main__":
    pytest.main([__file__])
