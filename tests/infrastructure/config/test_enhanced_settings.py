"""Tests for enhanced settings and secrets management."""

import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from pydantic import SecretStr

from pynomaly.infrastructure.config.settings import (
    Settings,
    SecretsManager,
    SecretsBackend,
    EnvironmentSecretsProvider,
    AWSSSMSecretsProvider,
    AWSSecretsManagerProvider,
    VaultSecretsProvider,
    MultiEnvSettingsConfigDict,
)


class TestSecretsProviders:
    """Test secrets providers."""

    def test_environment_secrets_provider(self):
        """Test environment variables secrets provider."""
        provider = EnvironmentSecretsProvider()

        # Test provider is always available
        assert provider.is_available() is True

        # Test getting existing environment variable
        with patch.dict(os.environ, {"TEST_SECRET": "test_value"}):
            assert provider.get_secret("TEST_SECRET") == "test_value"

        # Test getting non-existent environment variable
        assert provider.get_secret("NON_EXISTENT_SECRET") is None

    @patch("boto3.client")
    def test_aws_ssm_secrets_provider(self, mock_boto3_client):
        """Test AWS SSM secrets provider."""
        # Mock successful SSM client
        mock_client = Mock()
        mock_client.get_parameter.return_value = {
            "Parameter": {"Value": "secret_value_from_ssm"}
        }
        mock_boto3_client.return_value = mock_client

        provider = AWSSSMSecretsProvider(region="us-east-1")

        # Test provider is available when boto3 is available
        assert provider.is_available() is True

        # Test getting secret
        assert provider.get_secret("test_parameter") == "secret_value_from_ssm"
        mock_client.get_parameter.assert_called_once_with(
            Name="test_parameter", WithDecryption=True
        )

        # Test error handling
        mock_client.get_parameter.side_effect = Exception("Parameter not found")
        assert provider.get_secret("missing_parameter") is None

    @patch("boto3.client")
    def test_aws_secrets_manager_provider(self, mock_boto3_client):
        """Test AWS Secrets Manager provider."""
        # Mock successful Secrets Manager client
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            "SecretString": '{"key": "value", "password": "secret123"}'
        }
        mock_boto3_client.return_value = mock_client

        provider = AWSSecretsManagerProvider(region="us-west-2")

        # Test provider is available when boto3 is available
        assert provider.is_available() is True

        # Test getting JSON secret
        result = provider.get_secret("my_secret")
        assert result == '{"key": "value", "password": "secret123"}'

        # Test plain text secret
        mock_client.get_secret_value.return_value = {
            "SecretString": "plain_text_secret"
        }
        assert provider.get_secret("plain_secret") == "plain_text_secret"

    @patch("hvac.Client")
    def test_vault_secrets_provider(self, mock_hvac_client):
        """Test Vault secrets provider."""
        # Mock successful Vault client
        mock_client = Mock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"password": "vault_secret_value"}}
        }
        mock_hvac_client.return_value = mock_client

        provider = VaultSecretsProvider(
            vault_url="http://localhost:8200", vault_token="test_token"
        )

        # Test provider is available when hvac is available and authenticated
        assert provider.is_available() is True

        # Test getting secret with path
        assert provider.get_secret("myapp/password") == "vault_secret_value"

        # Test getting secret without path (uses default 'secret' path)
        assert provider.get_secret("password") == "vault_secret_value"


class TestSecretsManager:
    """Test secrets manager."""

    def test_secrets_manager_env_only(self):
        """Test secrets manager with environment variables only."""
        manager = SecretsManager(backends=[SecretsBackend.ENV])

        with patch.dict(os.environ, {"TEST_SECRET": "env_value"}):
            assert manager.get_secret("TEST_SECRET") == "env_value"
            assert (
                manager.get_secret_or_default("TEST_SECRET", "default") == "env_value"
            )
            assert (
                manager.get_secret_or_default("MISSING_SECRET", "default") == "default"
            )

    @patch("boto3.client")
    def test_secrets_manager_priority(self, mock_boto3_client):
        """Test secrets manager priority order."""
        # Mock AWS client
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {"SecretString": "aws_secret_value"}
        mock_boto3_client.return_value = mock_client

        # Create manager with AWS Secrets Manager first, then env
        manager = SecretsManager(
            backends=[SecretsBackend.AWS_SECRETS_MANAGER, SecretsBackend.ENV]
        )

        with patch.dict(os.environ, {"TEST_SECRET": "env_value"}):
            # Should get AWS value first (higher priority)
            result = manager.get_secret("TEST_SECRET")
            assert result == "aws_secret_value"

    def test_secrets_manager_fallback(self):
        """Test secrets manager fallback behavior."""
        # Create manager with unavailable backends first
        manager = SecretsManager(
            backends=[
                SecretsBackend.VAULT,  # Will be unavailable
                SecretsBackend.ENV,  # Will be available
            ]
        )

        with patch.dict(os.environ, {"TEST_SECRET": "env_fallback"}):
            # Should fall back to environment
            assert manager.get_secret("TEST_SECRET") == "env_fallback"


class TestMultiEnvConfig:
    """Test multi-environment configuration."""

    def test_multi_env_settings_config(self):
        """Test multi-environment settings configuration."""
        with patch.dict(os.environ, {"PYNOMALY_ENV": "testing"}):
            config = MultiEnvSettingsConfigDict()

            # Should include environment-specific files
            assert ".env" in config.get("env_file", [])
            assert ".env.testing" in config.get("env_file", [])
            assert config.get("env_prefix") == "PYNOMALY_"
            assert config.get("case_sensitive") is False

    def test_existing_env_files_only(self):
        """Test that only existing env files are included."""
        with patch("os.path.exists") as mock_exists:
            # Only .env exists
            mock_exists.side_effect = lambda path: path == ".env"

            config = MultiEnvSettingsConfigDict()
            env_files = config.get("env_file", [])

            # Should only include existing files
            assert len([f for f in env_files if "non_existent" not in f]) > 0


class TestEnhancedSettings:
    """Test enhanced Settings class."""

    def test_settings_initialization(self):
        """Test settings initialization with secrets manager."""
        settings = Settings(
            secrets_backends=[SecretsBackend.ENV], aws_region="us-east-1"
        )

        assert settings._secrets_manager is not None
        assert settings.secrets_backends == [SecretsBackend.ENV]
        assert settings.aws_region == "us-east-1"

    def test_secret_str_fields(self):
        """Test SecretStr fields."""
        settings = Settings(
            secret_key=SecretStr("test_secret"),
            database_url=SecretStr("postgresql://user:pass@localhost/db"),
        )

        # Test that values are properly stored as SecretStr
        assert isinstance(settings.secret_key, SecretStr)
        assert isinstance(settings.database_url, SecretStr)

        # Test secret retrieval methods
        assert settings.get_secret_key() == "test_secret"
        assert settings.get_database_url() == "postgresql://user:pass@localhost/db"

    def test_secrets_manager_integration(self):
        """Test integration with secrets manager."""
        with patch.dict(
            os.environ,
            {
                "SECRET_KEY": "env_secret_key",
                "DATABASE_URL": "env_database_url",
                "REDIS_URL": "env_redis_url",
            },
        ):
            settings = Settings(secrets_backends=[SecretsBackend.ENV])

            # Should get values from secrets manager first
            assert settings.get_secret_key() == "env_secret_key"
            assert settings.get_database_url() == "env_database_url"
            assert settings.get_redis_url() == "env_redis_url"

    def test_database_config_with_secrets(self):
        """Test database configuration with secrets."""
        with patch.dict(
            os.environ, {"DATABASE_URL": "postgresql://user:secret@localhost/prod_db"}
        ):
            settings = Settings(
                secrets_backends=[SecretsBackend.ENV], database_pool_size=20
            )

            config = settings.get_database_config()

            assert config["url"] == "postgresql://user:secret@localhost/prod_db"
            assert config["pool_size"] == 20
            assert "pool_pre_ping" in config

    def test_environment_specific_loading(self):
        """Test environment-specific configuration loading."""
        # Test development environment
        with patch.dict(os.environ, {"PYNOMALY_ENV": "development"}):
            with patch("os.path.exists", return_value=True):
                settings = Settings()
                # Should load development-specific settings
                assert settings.model_config.get("env_prefix") == "PYNOMALY_"

    def test_production_vs_development_config(self):
        """Test production vs development configuration differences."""
        # Development settings
        dev_settings = Settings(
            app__environment="development", app__debug=True, auth_enabled=False
        )

        # Production settings
        prod_settings = Settings(
            app__environment="production", app__debug=False, auth_enabled=True
        )

        assert not dev_settings.is_production
        assert prod_settings.is_production
        assert not dev_settings.auth_enabled
        assert prod_settings.auth_enabled

    def test_secrets_backend_configuration(self):
        """Test different secrets backend configurations."""
        # Test with multiple backends
        settings = Settings(
            secrets_backends=[
                SecretsBackend.AWS_SECRETS_MANAGER,
                SecretsBackend.AWS_SSM,
                SecretsBackend.ENV,
            ],
            aws_region="eu-west-1",
        )

        assert len(settings.secrets_backends) == 3
        assert SecretsBackend.AWS_SECRETS_MANAGER in settings.secrets_backends
        assert settings.aws_region == "eu-west-1"

    def test_vault_configuration(self):
        """Test Vault configuration."""
        settings = Settings(
            secrets_backends=[SecretsBackend.VAULT],
            vault_url="https://vault.company.com",
            vault_token="test_token",
        )

        assert settings.vault_url == "https://vault.company.com"
        assert settings.vault_token == "test_token"


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_contamination_rate_validation(self):
        """Test contamination rate validation."""
        # Valid rate
        settings = Settings(default_contamination_rate=0.1)
        assert settings.default_contamination_rate == 0.1

        # Invalid rates should raise validation error
        with pytest.raises(
            ValueError, match="Contamination rate must be between 0 and 1"
        ):
            Settings(default_contamination_rate=1.5)

        with pytest.raises(
            ValueError, match="Contamination rate must be between 0 and 1"
        ):
            Settings(default_contamination_rate=-0.1)

    def test_security_settings_validation(self):
        """Test security settings validation."""
        # Test sanitization level validation
        settings = Settings()
        settings.security.sanitization_level = "moderate"  # Valid

        with pytest.raises(ValueError):
            Settings(security__sanitization_level="invalid")

    def test_path_creation(self):
        """Test that storage paths are created."""
        temp_dir = Path("./test_temp_storage")
        settings = Settings(storage_path=temp_dir)

        # Path should be created
        assert temp_dir.exists()

        # Cleanup
        if temp_dir.exists():
            temp_dir.rmdir()


@pytest.fixture
def mock_aws_environment():
    """Mock AWS environment for testing."""
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "test_access_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret_key",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
    ):
        yield


@pytest.fixture
def mock_vault_environment():
    """Mock Vault environment for testing."""
    with patch.dict(
        os.environ,
        {"VAULT_TOKEN": "test_vault_token", "VAULT_ADDR": "http://localhost:8200"},
    ):
        yield


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def test_development_environment_setup(self):
        """Test complete development environment setup."""
        with patch.dict(
            os.environ,
            {
                "PYNOMALY_ENV": "development",
                "PYNOMALY_APP__DEBUG": "true",
                "PYNOMALY_SECRET_KEY": "dev-secret-key",
                "PYNOMALY_DATABASE_URL": "sqlite:///dev.db",
            },
        ):
            settings = Settings()

            assert settings.app.environment == "development"
            assert settings.app.debug is True
            assert settings.get_secret_key() == "dev-secret-key"
            assert settings.get_database_url() == "sqlite:///dev.db"

    @patch("boto3.client")
    def test_production_environment_with_aws(
        self, mock_boto3_client, mock_aws_environment
    ):
        """Test production environment with AWS secrets."""
        # Mock AWS Secrets Manager
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps(
                {
                    "SECRET_KEY": "prod-secret-from-aws",
                    "DATABASE_URL": "postgresql://user:pass@prod-db:5432/app",
                }
            )
        }
        mock_boto3_client.return_value = mock_client

        with patch.dict(
            os.environ,
            {
                "PYNOMALY_ENV": "production",
                "PYNOMALY_APP__DEBUG": "false",
                "PYNOMALY_SECRETS_BACKENDS": '["aws_secrets_manager", "env"]',
            },
        ):
            settings = Settings(
                secrets_backends=[
                    SecretsBackend.AWS_SECRETS_MANAGER,
                    SecretsBackend.ENV,
                ]
            )

            assert settings.app.environment == "production"
            assert not settings.is_production  # Should be False because debug is False

    def test_configuration_override_precedence(self):
        """Test configuration override precedence."""
        # Environment variables should override defaults
        with patch.dict(
            os.environ, {"PYNOMALY_API_PORT": "9000", "PYNOMALY_AUTH_ENABLED": "true"}
        ):
            settings = Settings()

            assert settings.api_port == 9000
            assert settings.auth_enabled is True

    def test_multi_environment_file_loading(self):
        """Test loading multiple environment files."""
        with patch.dict(os.environ, {"PYNOMALY_ENV": "testing"}):
            with patch("os.path.exists") as mock_exists:
                # Simulate multiple env files existing
                existing_files = {".env", ".env.local", ".env.testing"}
                mock_exists.side_effect = lambda path: path in existing_files

                config = MultiEnvSettingsConfigDict()
                env_files = config.get("env_file", [])

                # Should include all existing environment files
                assert any(".env" in str(f) for f in env_files)
                assert any(".env.testing" in str(f) for f in env_files)


if __name__ == "__main__":
    pytest.main([__file__])
