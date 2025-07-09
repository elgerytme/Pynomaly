"""Tests for error recovery module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from pynomaly.shared.error_handling.recovery import (
    RecoveryHandler,
    CacheRecoveryHandler,
    DatabaseRecoveryHandler,
    APIRecoveryHandler,
    FileRecoveryHandler,
    FallbackRecoveryHandler,
    RecoveryManager,
    RecoveryConfig,
    RecoveryResult,
    RecoveryStrategy,
    RecoveryContext,
    RecoveryMetrics,
    recovery_handler,
    fallback_on_error,
    circuit_breaker_recovery,
)
from pynomaly.shared.exceptions import (
    CacheError,
    DatabaseError,
    APIError,
    FileError,
    RecoveryError,
)


class TestRecoveryConfig:
    """Test RecoveryConfig class."""

    def test_recovery_config_initialization(self):
        """Test RecoveryConfig initialization."""
        config = RecoveryConfig(
            max_retries=3,
            retry_delay=1.0,
            timeout=30.0,
            fallback_enabled=True,
            circuit_breaker_enabled=True,
            recovery_strategies=[RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]
        )
        
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.timeout == 30.0
        assert config.fallback_enabled is True
        assert config.circuit_breaker_enabled is True
        assert RecoveryStrategy.RETRY in config.recovery_strategies
        assert RecoveryStrategy.FALLBACK in config.recovery_strategies

    def test_recovery_config_defaults(self):
        """Test RecoveryConfig default values."""
        config = RecoveryConfig()
        
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.timeout == 30.0
        assert config.fallback_enabled is True
        assert config.circuit_breaker_enabled is False
        assert config.recovery_strategies == [RecoveryStrategy.RETRY]

    def test_recovery_config_validation(self):
        """Test RecoveryConfig validation."""
        # Test invalid max_retries
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            RecoveryConfig(max_retries=-1)
        
        # Test invalid retry_delay
        with pytest.raises(ValueError, match="retry_delay must be positive"):
            RecoveryConfig(retry_delay=0)
        
        # Test invalid timeout
        with pytest.raises(ValueError, match="timeout must be positive"):
            RecoveryConfig(timeout=0)

    def test_recovery_config_to_dict(self):
        """Test RecoveryConfig to_dict method."""
        config = RecoveryConfig(
            max_retries=3,
            retry_delay=1.0,
            timeout=30.0,
            fallback_enabled=True
        )
        
        result = config.to_dict()
        
        assert result["max_retries"] == 3
        assert result["retry_delay"] == 1.0
        assert result["timeout"] == 30.0
        assert result["fallback_enabled"] is True

    def test_recovery_config_from_dict(self):
        """Test RecoveryConfig from_dict method."""
        data = {
            "max_retries": 5,
            "retry_delay": 2.0,
            "timeout": 60.0,
            "fallback_enabled": False
        }
        
        config = RecoveryConfig.from_dict(data)
        
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.timeout == 60.0
        assert config.fallback_enabled is False


class TestRecoveryContext:
    """Test RecoveryContext class."""

    def test_recovery_context_initialization(self):
        """Test RecoveryContext initialization."""
        context = RecoveryContext(
            operation="test_operation",
            error=Exception("test error"),
            attempt=1,
            total_attempts=3,
            start_time=datetime.utcnow(),
            metadata={"key": "value"}
        )
        
        assert context.operation == "test_operation"
        assert str(context.error) == "test error"
        assert context.attempt == 1
        assert context.total_attempts == 3
        assert context.start_time is not None
        assert context.metadata == {"key": "value"}

    def test_recovery_context_elapsed_time(self):
        """Test elapsed time calculation."""
        start_time = datetime.utcnow() - timedelta(seconds=5)
        context = RecoveryContext(
            operation="test_operation",
            error=Exception("test error"),
            start_time=start_time
        )
        
        elapsed = context.elapsed_time()
        assert elapsed >= timedelta(seconds=4)
        assert elapsed <= timedelta(seconds=6)

    def test_recovery_context_is_timeout(self):
        """Test timeout check."""
        start_time = datetime.utcnow() - timedelta(seconds=35)
        context = RecoveryContext(
            operation="test_operation",
            error=Exception("test error"),
            start_time=start_time
        )
        
        # Should timeout with 30 second limit
        assert context.is_timeout(30.0) is True
        
        # Should not timeout with 60 second limit
        assert context.is_timeout(60.0) is False

    def test_recovery_context_add_metadata(self):
        """Test adding metadata."""
        context = RecoveryContext(
            operation="test_operation",
            error=Exception("test error")
        )
        
        context.add_metadata("key1", "value1")
        context.add_metadata("key2", "value2")
        
        assert context.metadata["key1"] == "value1"
        assert context.metadata["key2"] == "value2"

    def test_recovery_context_to_dict(self):
        """Test RecoveryContext to_dict method."""
        context = RecoveryContext(
            operation="test_operation",
            error=Exception("test error"),
            attempt=1,
            total_attempts=3
        )
        
        result = context.to_dict()
        
        assert result["operation"] == "test_operation"
        assert result["error"] == "test error"
        assert result["attempt"] == 1
        assert result["total_attempts"] == 3
        assert "start_time" in result
        assert "metadata" in result


class TestRecoveryResult:
    """Test RecoveryResult class."""

    def test_recovery_result_initialization(self):
        """Test RecoveryResult initialization."""
        result = RecoveryResult(
            success=True,
            result="test_result",
            strategy_used=RecoveryStrategy.RETRY,
            attempts=2,
            recovery_time=timedelta(seconds=5),
            error=None
        )
        
        assert result.success is True
        assert result.result == "test_result"
        assert result.strategy_used == RecoveryStrategy.RETRY
        assert result.attempts == 2
        assert result.recovery_time == timedelta(seconds=5)
        assert result.error is None

    def test_recovery_result_failed(self):
        """Test failed RecoveryResult."""
        error = Exception("recovery failed")
        result = RecoveryResult(
            success=False,
            result=None,
            strategy_used=RecoveryStrategy.FALLBACK,
            attempts=3,
            recovery_time=timedelta(seconds=10),
            error=error
        )
        
        assert result.success is False
        assert result.result is None
        assert result.strategy_used == RecoveryStrategy.FALLBACK
        assert result.attempts == 3
        assert result.recovery_time == timedelta(seconds=10)
        assert result.error == error

    def test_recovery_result_to_dict(self):
        """Test RecoveryResult to_dict method."""
        result = RecoveryResult(
            success=True,
            result="test_result",
            strategy_used=RecoveryStrategy.RETRY,
            attempts=2,
            recovery_time=timedelta(seconds=5)
        )
        
        dict_result = result.to_dict()
        
        assert dict_result["success"] is True
        assert dict_result["result"] == "test_result"
        assert dict_result["strategy_used"] == "RETRY"
        assert dict_result["attempts"] == 2
        assert dict_result["recovery_time"] == 5.0

    def test_recovery_result_from_dict(self):
        """Test RecoveryResult from_dict method."""
        data = {
            "success": True,
            "result": "test_result",
            "strategy_used": "RETRY",
            "attempts": 2,
            "recovery_time": 5.0
        }
        
        result = RecoveryResult.from_dict(data)
        
        assert result.success is True
        assert result.result == "test_result"
        assert result.strategy_used == RecoveryStrategy.RETRY
        assert result.attempts == 2
        assert result.recovery_time == timedelta(seconds=5)


class TestRecoveryMetrics:
    """Test RecoveryMetrics class."""

    def test_recovery_metrics_initialization(self):
        """Test RecoveryMetrics initialization."""
        metrics = RecoveryMetrics()
        
        assert metrics.total_recoveries == 0
        assert metrics.successful_recoveries == 0
        assert metrics.failed_recoveries == 0
        assert metrics.total_attempts == 0
        assert metrics.average_recovery_time == 0.0
        assert metrics.strategy_usage == {}

    def test_recovery_metrics_record_recovery(self):
        """Test recording recovery."""
        metrics = RecoveryMetrics()
        
        result = RecoveryResult(
            success=True,
            result="test_result",
            strategy_used=RecoveryStrategy.RETRY,
            attempts=2,
            recovery_time=timedelta(seconds=5)
        )
        
        metrics.record_recovery(result)
        
        assert metrics.total_recoveries == 1
        assert metrics.successful_recoveries == 1
        assert metrics.failed_recoveries == 0
        assert metrics.total_attempts == 2
        assert metrics.average_recovery_time == 5.0
        assert metrics.strategy_usage[RecoveryStrategy.RETRY] == 1

    def test_recovery_metrics_record_failed_recovery(self):
        """Test recording failed recovery."""
        metrics = RecoveryMetrics()
        
        result = RecoveryResult(
            success=False,
            result=None,
            strategy_used=RecoveryStrategy.FALLBACK,
            attempts=3,
            recovery_time=timedelta(seconds=10),
            error=Exception("failed")
        )
        
        metrics.record_recovery(result)
        
        assert metrics.total_recoveries == 1
        assert metrics.successful_recoveries == 0
        assert metrics.failed_recoveries == 1
        assert metrics.total_attempts == 3
        assert metrics.average_recovery_time == 10.0
        assert metrics.strategy_usage[RecoveryStrategy.FALLBACK] == 1

    def test_recovery_metrics_get_success_rate(self):
        """Test success rate calculation."""
        metrics = RecoveryMetrics()
        
        # No recoveries
        assert metrics.get_success_rate() == 0.0
        
        # Add successful recovery
        metrics.successful_recoveries = 8
        metrics.total_recoveries = 10
        assert metrics.get_success_rate() == 0.8

    def test_recovery_metrics_get_average_attempts(self):
        """Test average attempts calculation."""
        metrics = RecoveryMetrics()
        
        # No recoveries
        assert metrics.get_average_attempts() == 0.0
        
        # Add attempts
        metrics.total_attempts = 20
        metrics.total_recoveries = 10
        assert metrics.get_average_attempts() == 2.0

    def test_recovery_metrics_reset(self):
        """Test resetting metrics."""
        metrics = RecoveryMetrics()
        
        # Set some values
        metrics.total_recoveries = 10
        metrics.successful_recoveries = 8
        metrics.total_attempts = 20
        metrics.strategy_usage[RecoveryStrategy.RETRY] = 5
        
        metrics.reset()
        
        assert metrics.total_recoveries == 0
        assert metrics.successful_recoveries == 0
        assert metrics.total_attempts == 0
        assert metrics.strategy_usage == {}

    def test_recovery_metrics_to_dict(self):
        """Test RecoveryMetrics to_dict method."""
        metrics = RecoveryMetrics()
        metrics.total_recoveries = 10
        metrics.successful_recoveries = 8
        metrics.strategy_usage[RecoveryStrategy.RETRY] = 5
        
        result = metrics.to_dict()
        
        assert result["total_recoveries"] == 10
        assert result["successful_recoveries"] == 8
        assert result["success_rate"] == 0.8
        assert result["strategy_usage"]["RETRY"] == 5


class TestRecoveryHandler:
    """Test RecoveryHandler base class."""

    def test_recovery_handler_initialization(self):
        """Test RecoveryHandler initialization."""
        config = RecoveryConfig(max_retries=3)
        handler = RecoveryHandler(config)
        
        assert handler.config == config
        assert handler.metrics is not None

    @pytest.mark.asyncio
    async def test_recovery_handler_abstract_methods(self):
        """Test abstract methods raise NotImplementedError."""
        config = RecoveryConfig()
        handler = RecoveryHandler(config)
        
        context = RecoveryContext(
            operation="test_operation",
            error=Exception("test error")
        )
        
        with pytest.raises(NotImplementedError):
            await handler.can_handle(context)
        
        with pytest.raises(NotImplementedError):
            await handler.handle_recovery(context)

    @pytest.mark.asyncio
    async def test_recovery_handler_execute_recovery(self):
        """Test execute_recovery method."""
        config = RecoveryConfig(max_retries=3)
        handler = RecoveryHandler(config)
        
        # Mock the abstract methods
        handler.can_handle = AsyncMock(return_value=True)
        handler.handle_recovery = AsyncMock(return_value="recovered_result")
        
        context = RecoveryContext(
            operation="test_operation",
            error=Exception("test error")
        )
        
        result = await handler.execute_recovery(context)
        
        assert result.success is True
        assert result.result == "recovered_result"
        assert result.attempts == 1
        assert handler.can_handle.called
        assert handler.handle_recovery.called

    @pytest.mark.asyncio
    async def test_recovery_handler_execute_recovery_cannot_handle(self):
        """Test execute_recovery when handler cannot handle error."""
        config = RecoveryConfig()
        handler = RecoveryHandler(config)
        
        # Mock cannot handle
        handler.can_handle = AsyncMock(return_value=False)
        
        context = RecoveryContext(
            operation="test_operation",
            error=Exception("test error")
        )
        
        result = await handler.execute_recovery(context)
        
        assert result.success is False
        assert result.error is not None
        assert handler.can_handle.called

    @pytest.mark.asyncio
    async def test_recovery_handler_retry_logic(self):
        """Test retry logic in execute_recovery."""
        config = RecoveryConfig(max_retries=3, retry_delay=0.1)
        handler = RecoveryHandler(config)
        
        # Mock methods
        handler.can_handle = AsyncMock(return_value=True)
        handler.handle_recovery = AsyncMock(side_effect=[
            Exception("retry 1"),
            Exception("retry 2"),
            "success"
        ])
        
        context = RecoveryContext(
            operation="test_operation",
            error=Exception("test error")
        )
        
        result = await handler.execute_recovery(context)
        
        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 3
        assert handler.handle_recovery.call_count == 3

    @pytest.mark.asyncio
    async def test_recovery_handler_timeout(self):
        """Test timeout in execute_recovery."""
        config = RecoveryConfig(timeout=0.1)
        handler = RecoveryHandler(config)
        
        # Mock methods
        handler.can_handle = AsyncMock(return_value=True)
        handler.handle_recovery = AsyncMock(side_effect=lambda ctx: asyncio.sleep(0.2))
        
        context = RecoveryContext(
            operation="test_operation",
            error=Exception("test error")
        )
        
        result = await handler.execute_recovery(context)
        
        assert result.success is False
        assert "timeout" in str(result.error).lower()


class TestCacheRecoveryHandler:
    """Test CacheRecoveryHandler class."""

    def test_cache_recovery_handler_initialization(self):
        """Test CacheRecoveryHandler initialization."""
        config = RecoveryConfig()
        handler = CacheRecoveryHandler(config)
        
        assert handler.config == config
        assert handler.fallback_cache is None

    @pytest.mark.asyncio
    async def test_cache_recovery_handler_can_handle(self):
        """Test can_handle method."""
        config = RecoveryConfig()
        handler = CacheRecoveryHandler(config)
        
        # Should handle cache errors
        cache_context = RecoveryContext(
            operation="cache_get",
            error=CacheError("cache error")
        )
        assert await handler.can_handle(cache_context) is True
        
        # Should not handle other errors
        other_context = RecoveryContext(
            operation="db_query",
            error=DatabaseError("db error")
        )
        assert await handler.can_handle(other_context) is False

    @pytest.mark.asyncio
    async def test_cache_recovery_handler_handle_recovery(self):
        """Test handle_recovery method."""
        config = RecoveryConfig()
        handler = CacheRecoveryHandler(config)
        
        # Mock fallback cache
        mock_fallback = AsyncMock()
        mock_fallback.get.return_value = "fallback_value"
        handler.fallback_cache = mock_fallback
        
        context = RecoveryContext(
            operation="cache_get",
            error=CacheError("cache error"),
            metadata={"key": "test_key"}
        )
        
        result = await handler.handle_recovery(context)
        
        assert result == "fallback_value"
        mock_fallback.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_cache_recovery_handler_no_fallback(self):
        """Test handle_recovery without fallback cache."""
        config = RecoveryConfig()
        handler = CacheRecoveryHandler(config)
        
        context = RecoveryContext(
            operation="cache_get",
            error=CacheError("cache error")
        )
        
        with pytest.raises(RecoveryError):
            await handler.handle_recovery(context)

    @pytest.mark.asyncio
    async def test_cache_recovery_handler_cache_set_operation(self):
        """Test cache set operation recovery."""
        config = RecoveryConfig()
        handler = CacheRecoveryHandler(config)
        
        # Mock fallback cache
        mock_fallback = AsyncMock()
        mock_fallback.set.return_value = True
        handler.fallback_cache = mock_fallback
        
        context = RecoveryContext(
            operation="cache_set",
            error=CacheError("cache error"),
            metadata={"key": "test_key", "value": "test_value", "ttl": 3600}
        )
        
        result = await handler.handle_recovery(context)
        
        assert result is True
        mock_fallback.set.assert_called_once_with("test_key", "test_value", ttl=3600)

    @pytest.mark.asyncio
    async def test_cache_recovery_handler_cache_delete_operation(self):
        """Test cache delete operation recovery."""
        config = RecoveryConfig()
        handler = CacheRecoveryHandler(config)
        
        # Mock fallback cache
        mock_fallback = AsyncMock()
        mock_fallback.delete.return_value = True
        handler.fallback_cache = mock_fallback
        
        context = RecoveryContext(
            operation="cache_delete",
            error=CacheError("cache error"),
            metadata={"key": "test_key"}
        )
        
        result = await handler.handle_recovery(context)
        
        assert result is True
        mock_fallback.delete.assert_called_once_with("test_key")


class TestDatabaseRecoveryHandler:
    """Test DatabaseRecoveryHandler class."""

    def test_database_recovery_handler_initialization(self):
        """Test DatabaseRecoveryHandler initialization."""
        config = RecoveryConfig()
        handler = DatabaseRecoveryHandler(config)
        
        assert handler.config == config
        assert handler.fallback_db is None

    @pytest.mark.asyncio
    async def test_database_recovery_handler_can_handle(self):
        """Test can_handle method."""
        config = RecoveryConfig()
        handler = DatabaseRecoveryHandler(config)
        
        # Should handle database errors
        db_context = RecoveryContext(
            operation="db_query",
            error=DatabaseError("db error")
        )
        assert await handler.can_handle(db_context) is True
        
        # Should not handle other errors
        other_context = RecoveryContext(
            operation="cache_get",
            error=CacheError("cache error")
        )
        assert await handler.can_handle(other_context) is False

    @pytest.mark.asyncio
    async def test_database_recovery_handler_handle_recovery(self):
        """Test handle_recovery method."""
        config = RecoveryConfig()
        handler = DatabaseRecoveryHandler(config)
        
        # Mock fallback database
        mock_fallback = AsyncMock()
        mock_fallback.execute.return_value = "query_result"
        handler.fallback_db = mock_fallback
        
        context = RecoveryContext(
            operation="db_query",
            error=DatabaseError("db error"),
            metadata={"query": "SELECT * FROM users", "params": {"id": 1}}
        )
        
        result = await handler.handle_recovery(context)
        
        assert result == "query_result"
        mock_fallback.execute.assert_called_once_with("SELECT * FROM users", {"id": 1})

    @pytest.mark.asyncio
    async def test_database_recovery_handler_no_fallback(self):
        """Test handle_recovery without fallback database."""
        config = RecoveryConfig()
        handler = DatabaseRecoveryHandler(config)
        
        context = RecoveryContext(
            operation="db_query",
            error=DatabaseError("db error")
        )
        
        with pytest.raises(RecoveryError):
            await handler.handle_recovery(context)

    @pytest.mark.asyncio
    async def test_database_recovery_handler_connection_recovery(self):
        """Test database connection recovery."""
        config = RecoveryConfig()
        handler = DatabaseRecoveryHandler(config)
        
        # Mock connection recovery
        mock_db = AsyncMock()
        mock_db.reconnect.return_value = True
        handler.fallback_db = mock_db
        
        context = RecoveryContext(
            operation="db_connect",
            error=DatabaseError("connection error")
        )
        
        result = await handler.handle_recovery(context)
        
        assert result is True
        mock_db.reconnect.assert_called_once()


class TestAPIRecoveryHandler:
    """Test APIRecoveryHandler class."""

    def test_api_recovery_handler_initialization(self):
        """Test APIRecoveryHandler initialization."""
        config = RecoveryConfig()
        handler = APIRecoveryHandler(config)
        
        assert handler.config == config
        assert handler.fallback_endpoints == []

    @pytest.mark.asyncio
    async def test_api_recovery_handler_can_handle(self):
        """Test can_handle method."""
        config = RecoveryConfig()
        handler = APIRecoveryHandler(config)
        
        # Should handle API errors
        api_context = RecoveryContext(
            operation="api_call",
            error=APIError("api error")
        )
        assert await handler.can_handle(api_context) is True
        
        # Should not handle other errors
        other_context = RecoveryContext(
            operation="cache_get",
            error=CacheError("cache error")
        )
        assert await handler.can_handle(other_context) is False

    @pytest.mark.asyncio
    async def test_api_recovery_handler_handle_recovery(self):
        """Test handle_recovery method."""
        config = RecoveryConfig()
        handler = APIRecoveryHandler(config)
        
        # Add fallback endpoint
        handler.fallback_endpoints = ["https://fallback.api.com"]
        
        context = RecoveryContext(
            operation="api_call",
            error=APIError("api error"),
            metadata={
                "url": "https://primary.api.com/endpoint",
                "method": "GET",
                "headers": {"Authorization": "Bearer token"}
            }
        )
        
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"result": "success"}
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await handler.handle_recovery(context)
            
            assert result == {"result": "success"}
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_recovery_handler_no_fallback(self):
        """Test handle_recovery without fallback endpoints."""
        config = RecoveryConfig()
        handler = APIRecoveryHandler(config)
        
        context = RecoveryContext(
            operation="api_call",
            error=APIError("api error")
        )
        
        with pytest.raises(RecoveryError):
            await handler.handle_recovery(context)

    @pytest.mark.asyncio
    async def test_api_recovery_handler_multiple_fallbacks(self):
        """Test handle_recovery with multiple fallback endpoints."""
        config = RecoveryConfig()
        handler = APIRecoveryHandler(config)
        
        # Add multiple fallback endpoints
        handler.fallback_endpoints = [
            "https://fallback1.api.com",
            "https://fallback2.api.com"
        ]
        
        context = RecoveryContext(
            operation="api_call",
            error=APIError("api error"),
            metadata={
                "url": "https://primary.api.com/endpoint",
                "method": "GET"
            }
        )
        
        with patch("aiohttp.ClientSession.get") as mock_get:
            # First fallback fails, second succeeds
            mock_get.side_effect = [
                Exception("fallback1 failed"),
                AsyncMock(status=200, json=AsyncMock(return_value={"result": "success"}))
            ]
            
            result = await handler.handle_recovery(context)
            
            assert result == {"result": "success"}
            assert mock_get.call_count == 2


class TestFileRecoveryHandler:
    """Test FileRecoveryHandler class."""

    def test_file_recovery_handler_initialization(self):
        """Test FileRecoveryHandler initialization."""
        config = RecoveryConfig()
        handler = FileRecoveryHandler(config)
        
        assert handler.config == config
        assert handler.backup_locations == []

    @pytest.mark.asyncio
    async def test_file_recovery_handler_can_handle(self):
        """Test can_handle method."""
        config = RecoveryConfig()
        handler = FileRecoveryHandler(config)
        
        # Should handle file errors
        file_context = RecoveryContext(
            operation="file_read",
            error=FileError("file error")
        )
        assert await handler.can_handle(file_context) is True
        
        # Should not handle other errors
        other_context = RecoveryContext(
            operation="cache_get",
            error=CacheError("cache error")
        )
        assert await handler.can_handle(other_context) is False

    @pytest.mark.asyncio
    async def test_file_recovery_handler_handle_recovery(self):
        """Test handle_recovery method."""
        config = RecoveryConfig()
        handler = FileRecoveryHandler(config)
        
        # Add backup location
        handler.backup_locations = ["/backup/path"]
        
        context = RecoveryContext(
            operation="file_read",
            error=FileError("file error"),
            metadata={"file_path": "/original/path/file.txt"}
        )
        
        with patch("builtins.open", MagicMock(return_value=MagicMock(read=MagicMock(return_value="file_content")))):
            result = await handler.handle_recovery(context)
            
            assert result == "file_content"

    @pytest.mark.asyncio
    async def test_file_recovery_handler_no_backup(self):
        """Test handle_recovery without backup locations."""
        config = RecoveryConfig()
        handler = FileRecoveryHandler(config)
        
        context = RecoveryContext(
            operation="file_read",
            error=FileError("file error")
        )
        
        with pytest.raises(RecoveryError):
            await handler.handle_recovery(context)

    @pytest.mark.asyncio
    async def test_file_recovery_handler_file_write_operation(self):
        """Test file write operation recovery."""
        config = RecoveryConfig()
        handler = FileRecoveryHandler(config)
        
        # Add backup location
        handler.backup_locations = ["/backup/path"]
        
        context = RecoveryContext(
            operation="file_write",
            error=FileError("file error"),
            metadata={
                "file_path": "/original/path/file.txt",
                "content": "test_content"
            }
        )
        
        with patch("builtins.open", MagicMock()) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result = await handler.handle_recovery(context)
            
            assert result is True
            mock_file.write.assert_called_once_with("test_content")


class TestFallbackRecoveryHandler:
    """Test FallbackRecoveryHandler class."""

    def test_fallback_recovery_handler_initialization(self):
        """Test FallbackRecoveryHandler initialization."""
        config = RecoveryConfig()
        fallback_func = lambda: "fallback_result"
        handler = FallbackRecoveryHandler(config, fallback_func)
        
        assert handler.config == config
        assert handler.fallback_func == fallback_func

    @pytest.mark.asyncio
    async def test_fallback_recovery_handler_can_handle(self):
        """Test can_handle method."""
        config = RecoveryConfig()
        handler = FallbackRecoveryHandler(config, lambda: "fallback")
        
        # Should handle any error
        context = RecoveryContext(
            operation="any_operation",
            error=Exception("any error")
        )
        assert await handler.can_handle(context) is True

    @pytest.mark.asyncio
    async def test_fallback_recovery_handler_handle_recovery(self):
        """Test handle_recovery method."""
        config = RecoveryConfig()
        handler = FallbackRecoveryHandler(config, lambda: "fallback_result")
        
        context = RecoveryContext(
            operation="any_operation",
            error=Exception("any error")
        )
        
        result = await handler.handle_recovery(context)
        
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_fallback_recovery_handler_async_fallback(self):
        """Test handle_recovery with async fallback function."""
        config = RecoveryConfig()
        
        async def async_fallback():
            return "async_fallback_result"
        
        handler = FallbackRecoveryHandler(config, async_fallback)
        
        context = RecoveryContext(
            operation="any_operation",
            error=Exception("any error")
        )
        
        result = await handler.handle_recovery(context)
        
        assert result == "async_fallback_result"

    @pytest.mark.asyncio
    async def test_fallback_recovery_handler_fallback_with_context(self):
        """Test handle_recovery with fallback function that takes context."""
        config = RecoveryConfig()
        
        def fallback_with_context(context):
            return f"fallback_for_{context.operation}"
        
        handler = FallbackRecoveryHandler(config, fallback_with_context)
        
        context = RecoveryContext(
            operation="test_operation",
            error=Exception("test error")
        )
        
        result = await handler.handle_recovery(context)
        
        assert result == "fallback_for_test_operation"


class TestRecoveryManager:
    """Test RecoveryManager class."""

    def test_recovery_manager_initialization(self):
        """Test RecoveryManager initialization."""
        config = RecoveryConfig()
        manager = RecoveryManager(config)
        
        assert manager.config == config
        assert manager.handlers == []
        assert manager.metrics is not None

    def test_recovery_manager_add_handler(self):
        """Test adding recovery handler."""
        config = RecoveryConfig()
        manager = RecoveryManager(config)
        
        handler = CacheRecoveryHandler(config)
        manager.add_handler(handler)
        
        assert len(manager.handlers) == 1
        assert manager.handlers[0] == handler

    def test_recovery_manager_remove_handler(self):
        """Test removing recovery handler."""
        config = RecoveryConfig()
        manager = RecoveryManager(config)
        
        handler = CacheRecoveryHandler(config)
        manager.add_handler(handler)
        
        removed = manager.remove_handler(handler)
        
        assert removed is True
        assert len(manager.handlers) == 0

    def test_recovery_manager_remove_nonexistent_handler(self):
        """Test removing non-existent handler."""
        config = RecoveryConfig()
        manager = RecoveryManager(config)
        
        handler = CacheRecoveryHandler(config)
        removed = manager.remove_handler(handler)
        
        assert removed is False

    @pytest.mark.asyncio
    async def test_recovery_manager_recover(self):
        """Test recovery process."""
        config = RecoveryConfig()
        manager = RecoveryManager(config)
        
        # Add handler
        handler = CacheRecoveryHandler(config)
        handler.can_handle = AsyncMock(return_value=True)
        handler.handle_recovery = AsyncMock(return_value="recovered_result")
        manager.add_handler(handler)
        
        context = RecoveryContext(
            operation="cache_get",
            error=CacheError("cache error")
        )
        
        result = await manager.recover(context)
        
        assert result.success is True
        assert result.result == "recovered_result"
        assert result.strategy_used == RecoveryStrategy.RETRY

    @pytest.mark.asyncio
    async def test_recovery_manager_no_suitable_handler(self):
        """Test recovery when no handler can handle the error."""
        config = RecoveryConfig()
        manager = RecoveryManager(config)
        
        # Add handler that cannot handle the error
        handler = CacheRecoveryHandler(config)
        handler.can_handle = AsyncMock(return_value=False)
        manager.add_handler(handler)
        
        context = RecoveryContext(
            operation="db_query",
            error=DatabaseError("db error")
        )
        
        result = await manager.recover(context)
        
        assert result.success is False
        assert "No suitable recovery handler found" in str(result.error)

    @pytest.mark.asyncio
    async def test_recovery_manager_multiple_handlers(self):
        """Test recovery with multiple handlers."""
        config = RecoveryConfig()
        manager = RecoveryManager(config)
        
        # Add cache handler
        cache_handler = CacheRecoveryHandler(config)
        cache_handler.can_handle = AsyncMock(return_value=False)
        manager.add_handler(cache_handler)
        
        # Add database handler
        db_handler = DatabaseRecoveryHandler(config)
        db_handler.can_handle = AsyncMock(return_value=True)
        db_handler.handle_recovery = AsyncMock(return_value="db_recovered")
        manager.add_handler(db_handler)
        
        context = RecoveryContext(
            operation="db_query",
            error=DatabaseError("db error")
        )
        
        result = await manager.recover(context)
        
        assert result.success is True
        assert result.result == "db_recovered"
        assert cache_handler.can_handle.called
        assert db_handler.can_handle.called

    def test_recovery_manager_get_metrics(self):
        """Test getting recovery metrics."""
        config = RecoveryConfig()
        manager = RecoveryManager(config)
        
        # Set some metrics
        manager.metrics.total_recoveries = 10
        manager.metrics.successful_recoveries = 8
        
        metrics = manager.get_metrics()
        
        assert metrics.total_recoveries == 10
        assert metrics.successful_recoveries == 8

    def test_recovery_manager_reset_metrics(self):
        """Test resetting recovery metrics."""
        config = RecoveryConfig()
        manager = RecoveryManager(config)
        
        # Set some metrics
        manager.metrics.total_recoveries = 10
        manager.metrics.successful_recoveries = 8
        
        manager.reset_metrics()
        
        assert manager.metrics.total_recoveries == 0
        assert manager.metrics.successful_recoveries == 0


class TestRecoveryDecorators:
    """Test recovery decorator functions."""

    @pytest.mark.asyncio
    async def test_recovery_handler_decorator(self):
        """Test recovery_handler decorator."""
        config = RecoveryConfig(max_retries=2)
        
        @recovery_handler(config)
        async def test_func():
            # Simulate failure then success
            if not hasattr(test_func, 'called'):
                test_func.called = True
                raise Exception("First call fails")
            return "success"
        
        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_fallback_on_error_decorator(self):
        """Test fallback_on_error decorator."""
        @fallback_on_error(lambda: "fallback_result")
        async def test_func():
            raise Exception("Function fails")
        
        result = await test_func()
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_decorator(self):
        """Test circuit_breaker_recovery decorator."""
        @circuit_breaker_recovery(failure_threshold=2, recovery_timeout=0.1)
        async def test_func():
            if not hasattr(test_func, 'calls'):
                test_func.calls = 0
            test_func.calls += 1
            
            if test_func.calls <= 2:
                raise Exception("Circuit breaker test")
            return "success"
        
        # First two calls should fail and open circuit
        with pytest.raises(Exception):
            await test_func()
        with pytest.raises(Exception):
            await test_func()
        
        # Third call should be blocked by circuit breaker
        with pytest.raises(Exception):
            await test_func()
        
        # After recovery timeout, circuit should be half-open
        await asyncio.sleep(0.2)
        result = await test_func()
        assert result == "success"