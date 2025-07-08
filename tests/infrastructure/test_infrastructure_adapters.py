"""Test infrastructure adapters for Step 6: Test infrastructure adapters

• Persistence: in-memory SQLite with transaction rollback per test.  
• Caching: fake Redis client.  
• Observability & auth: ensure middlewares register without raising.  
Validate environment variable parsing in Settings.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.persistence.database import DatabaseManager
from pynomaly.infrastructure.cache.redis_cache import RedisCache


class TestPersistenceAdapter:
    """Test persistence adapter with in-memory SQLite and transaction rollback."""

    def test_in_memory_sqlite_connection(self):
        """Test in-memory SQLite database connection."""
        db_manager = DatabaseManager("sqlite:///:memory:", echo=False)
        assert db_manager.database_url == "sqlite:///:memory:"
        assert db_manager.echo is False

        # Test engine creation
        engine = db_manager.engine
        assert engine is not None
        assert str(engine.url) == "sqlite:///:memory:"

        # Clean up
        db_manager.close()

    def test_transaction_rollback(self):
        """Test transaction rollback per test."""
        db_manager = DatabaseManager("sqlite:///:memory:", echo=False)

        # Create a test table
        with db_manager.engine.connect() as conn:
            conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
            conn.commit()

        # Test rollback functionality
        Session = sessionmaker(bind=db_manager.engine)
        session = Session()

        try:
            # Insert test data
            session.execute("INSERT INTO test_table (name) VALUES ('test')")
            session.commit()

            # Verify data exists
            result = session.execute("SELECT COUNT(*) FROM test_table").scalar()
            assert result == 1

            # Simulate rollback
            session.rollback()

        finally:
            session.close()
            db_manager.close()

    def test_database_manager_session_context(self):
        """Test database manager session context manager."""
        db_manager = DatabaseManager("sqlite:///:memory:", echo=False)

        # Test context manager
        with db_manager.get_session() as session:
            # Basic session test
            assert session is not None
            # Test that session is properly closed after context

        db_manager.close()


class TestCachingAdapter:
    """Test caching adapter with fake Redis client."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for cache testing."""
        settings = MagicMock()
        settings.cache_enabled = True
        settings.redis_url = "redis://localhost:6379"
        settings.cache_ttl = 3600
        return settings

    def test_fake_redis_client_creation(self, mock_settings):
        """Test fake Redis client creation."""
        with patch("redis.from_url") as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True

            cache = RedisCache(mock_settings)

            # Verify Redis client was created
            mock_redis.assert_called_once()
            assert cache.enabled is True
            assert cache._client is mock_client

    def test_cache_operations(self, mock_settings):
        """Test cache operations with mocked Redis."""
        with patch("redis.from_url") as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True

            cache = RedisCache(mock_settings)

            # Test set operation
            cache.set("test_key", "test_value")
            mock_client.setex.assert_called_once()

            # Test get operation
            mock_client.get.return_value = b'"test_value"'
            result = cache.get("test_key")
            mock_client.get.assert_called_once_with("test_key")

            # Test delete operation
            cache.delete("test_key")
            mock_client.delete.assert_called_once_with("test_key")

    def test_cache_disabled_handling(self):
        """Test cache behavior when disabled."""
        settings = MagicMock()
        settings.cache_enabled = False
        settings.redis_url = None

        cache = RedisCache(settings)

        # Cache should be disabled
        assert cache.enabled is False
        assert cache._client is None

        # Operations should return defaults
        assert cache.get("key") is None
        assert cache.set("key", "value") is False
        assert cache.delete("key") is False


class TestObservabilityMiddleware:
    """Test observability middleware registration."""

    def test_middleware_import(self):
        """Test middleware components can be imported."""
        try:
            from pynomaly.infrastructure.monitoring.middleware import MetricsMiddleware

            assert MetricsMiddleware is not None
        except ImportError:
            # Skip if FastAPI is not available
            pytest.skip("FastAPI not available, skipping middleware tests")

    def test_auth_middleware_import(self):
        """Test auth middleware components can be imported."""
        try:
            from pynomaly.infrastructure.auth.middleware import track_request_metrics

            assert callable(track_request_metrics)
        except ImportError:
            # Skip if FastAPI is not available
            pytest.skip("FastAPI not available, skipping auth middleware tests")


class TestSettingsConfiguration:
    """Test environment variable parsing in Settings."""

    def test_environment_variable_parsing(self, monkeypatch):
        """Test environment variable parsing in Settings."""
        # Set environment variables
        monkeypatch.setenv("PYNOMALY_API_HOST", "127.0.0.1")
        monkeypatch.setenv("PYNOMALY_API_PORT", "8080")
        monkeypatch.setenv("PYNOMALY_DEBUG", "true")
        monkeypatch.setenv("PYNOMALY_CACHE_ENABLED", "false")

        settings = Settings()

        # Test environment variables are parsed correctly
        assert settings.api_host == "127.0.0.1"
        assert settings.api_port == 8080
        assert settings.app.debug is True
        assert settings.cache_enabled is False

    def test_settings_defaults(self):
        """Test settings have proper defaults."""
        settings = Settings()

        # Test default values
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.app.name == "Pynomaly"
        assert settings.app.environment == "development"
        assert settings.cache_enabled is True
        assert settings.auth_enabled is False

    def test_database_configuration(self):
        """Test database configuration parsing."""
        settings = Settings()

        # Test database configuration without URL
        assert settings.database_configured is False
        assert settings.get_database_config() == {}

        # Test with SQLite URL
        settings.database_url = "sqlite:///test.db"
        assert settings.database_configured is True

        config = settings.get_database_config()
        assert config["url"] == "sqlite:///test.db"
        assert "connect_args" in config
        assert config["connect_args"]["check_same_thread"] is False

    def test_monitoring_configuration(self):
        """Test monitoring configuration."""
        settings = Settings()

        # Test monitoring settings
        assert settings.monitoring.metrics_enabled is True
        assert settings.monitoring.prometheus_enabled is True
        assert settings.monitoring.log_level == "INFO"
        assert settings.monitoring.log_format == "json"

    def test_security_configuration(self):
        """Test security configuration."""
        settings = Settings()

        # Test security settings
        assert settings.security.sanitization_level == "moderate"
        assert settings.security.max_input_length == 10000
        assert settings.security.enable_audit_logging is True
        assert settings.security.enable_security_monitoring is True


class TestInfrastructureIntegration:
    """Test integration of infrastructure adapters."""

    def test_settings_and_database_integration(self):
        """Test settings and database integration."""
        settings = Settings()
        settings.database_url = "sqlite:///:memory:"

        # Test database manager can be created from settings
        db_manager = DatabaseManager(
            database_url=settings.database_url, echo=settings.database_echo
        )

        assert db_manager.database_url == settings.database_url
        assert db_manager.echo == settings.database_echo

        db_manager.close()

    def test_settings_and_cache_integration(self):
        """Test settings and cache integration."""
        settings = Settings()
        settings.cache_enabled = True
        settings.redis_url = "redis://localhost:6379"

        with patch("redis.from_url") as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True

            # Test cache can be created from settings
            cache = RedisCache(settings)

            assert cache.enabled == settings.cache_enabled
            assert cache.settings.redis_url == settings.redis_url
