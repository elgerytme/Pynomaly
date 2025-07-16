"""Connection pooling implementation for database, cache, and HTTP connections."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
import redis.asyncio as redis
import structlog
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = structlog.get_logger(__name__)


class PoolType(Enum):
    """Pool type enumeration."""

    DATABASE = "database"
    REDIS = "redis"
    HTTP = "http"


@dataclass
class PoolConfiguration:
    """Pool configuration settings."""

    # Core pool settings
    min_size: int = 5
    max_size: int = 20
    timeout: float = 30.0
    max_overflow: int = 10

    # Connection lifecycle
    recycle_time: int = 3600  # 1 hour
    pre_ping: bool = True
    pool_reset_on_return: str = "commit"

    # Advanced settings
    pool_timeout: float = 30.0
    max_idle_time: int = 300  # 5 minutes
    health_check_interval: int = 60  # 1 minute

    # Retry and backoff
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0

    # Monitoring
    enable_metrics: bool = True
    log_slow_queries: bool = True
    slow_query_threshold: float = 1.0


@dataclass
class PoolStats:
    """Pool statistics."""

    pool_type: PoolType
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    overflow_connections: int = 0

    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0

    # Connection metrics
    connections_created: int = 0
    connections_closed: int = 0
    connections_recycled: int = 0
    connection_errors: int = 0

    # Timing statistics
    created_at: float = field(default_factory=time.time)
    last_reset: float = field(default_factory=time.time)


class DatabasePool:
    """Database connection pool manager."""

    def __init__(
        self, database_url: str, config: PoolConfiguration, async_mode: bool = False
    ):
        """Initialize database pool.

        Args:
            database_url: Database connection URL
            config: Pool configuration
            async_mode: Whether to use async engine
        """
        self.database_url = database_url
        self.config = config
        self.async_mode = async_mode
        self.stats = PoolStats(pool_type=PoolType.DATABASE)

        # Create connection engines
        if async_mode:
            self._async_engine = self._create_async_engine()
            self._sync_engine = None
        else:
            self._sync_engine = self._create_sync_engine()
            self._async_engine = None

        # Health monitoring
        self._health_check_task: asyncio.Task | None = None
        self._start_health_monitoring()

    def _create_sync_engine(self) -> Engine:
        """Create synchronous SQLAlchemy engine with pooling."""
        return create_engine(
            self.database_url,
            poolclass=pool.QueuePool,
            pool_size=self.config.min_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.recycle_time,
            pool_pre_ping=self.config.pre_ping,
            pool_reset_on_return=self.config.pool_reset_on_return,
            echo=False,  # Set to True for SQL debugging
        )

    def _create_async_engine(self) -> AsyncEngine:
        """Create asynchronous SQLAlchemy engine with pooling."""
        return create_async_engine(
            self.database_url,
            poolclass=pool.AsyncAdaptedQueuePool,
            pool_size=self.config.min_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.recycle_time,
            pool_pre_ping=self.config.pre_ping,
            pool_reset_on_return=self.config.pool_reset_on_return,
            echo=False,
        )

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Get sync database connection from pool."""
        if not self._sync_engine:
            raise RuntimeError("Sync engine not available")

        start_time = time.time()
        connection = None

        try:
            self.stats.total_requests += 1
            connection = self._sync_engine.connect()
            self.stats.active_connections += 1

            yield connection

            self.stats.successful_requests += 1
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)

        except Exception as e:
            self.stats.failed_requests += 1
            logger.error("Database connection error", error=str(e))
            raise
        finally:
            if connection:
                connection.close()
                self.stats.active_connections = max(
                    0, self.stats.active_connections - 1
                )

    @asynccontextmanager
    async def get_async_connection(self) -> AsyncGenerator[Any, None]:
        """Get async database connection from pool."""
        if not self._async_engine:
            raise RuntimeError("Async engine not available")

        start_time = time.time()
        connection = None

        try:
            self.stats.total_requests += 1
            connection = await self._async_engine.connect()
            self.stats.active_connections += 1

            yield connection

            self.stats.successful_requests += 1
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)

        except Exception as e:
            self.stats.failed_requests += 1
            logger.error("Async database connection error", error=str(e))
            raise
        finally:
            if connection:
                await connection.close()
                self.stats.active_connections = max(
                    0, self.stats.active_connections - 1
                )

    def _start_health_monitoring(self) -> None:
        """Start health check monitoring."""
        if self.async_mode and self.config.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error", error=str(e))

    async def _perform_health_check(self) -> None:
        """Perform database health check."""
        try:
            if self._async_engine:
                async with self.get_async_connection() as conn:
                    await conn.execute("SELECT 1")
            elif self._sync_engine:
                with self.get_connection() as conn:
                    conn.execute("SELECT 1")

            logger.debug("Database health check passed")
        except Exception as e:
            self.stats.connection_errors += 1
            logger.error("Database health check failed", error=str(e))

    def _update_avg_response_time(self, response_time: float) -> None:
        """Update average response time."""
        if self.stats.successful_requests > 1:
            self.stats.avg_response_time = (
                self.stats.avg_response_time * (self.stats.successful_requests - 1)
                + response_time
            ) / self.stats.successful_requests
        else:
            self.stats.avg_response_time = response_time

    def get_pool_info(self) -> dict[str, Any]:
        """Get current pool information."""
        if self._sync_engine:
            pool_obj = self._sync_engine.pool
        elif self._async_engine:
            pool_obj = self._async_engine.pool
        else:
            return {}

        return {
            "size": pool_obj.size(),
            "checked_in": pool_obj.checkedin(),
            "checked_out": pool_obj.checkedout(),
            "overflow": pool_obj.overflow(),
            "invalid": pool_obj.invalid(),
        }

    async def close(self) -> None:
        """Close database pool."""
        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            if self._async_engine:
                await self._async_engine.dispose()
            elif self._sync_engine:
                self._sync_engine.dispose()

            logger.info("Database pool closed")
        except Exception as e:
            logger.error("Error closing database pool", error=str(e))


class RedisPool:
    """Redis connection pool manager."""

    def __init__(self, redis_url: str, config: PoolConfiguration):
        """Initialize Redis pool.

        Args:
            redis_url: Redis connection URL
            config: Pool configuration
        """
        self.redis_url = redis_url
        self.config = config
        self.stats = PoolStats(pool_type=PoolType.REDIS)

        # Create connection pool
        self._pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=config.max_size,
            socket_timeout=config.timeout,
            socket_connect_timeout=config.timeout,
            health_check_interval=config.health_check_interval,
            retry_on_timeout=True,
        )

        self._client = redis.Redis(connection_pool=self._pool)

        # Health monitoring
        self._health_check_task: asyncio.Task | None = None
        self._start_health_monitoring()

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[redis.Redis, None]:
        """Get Redis connection from pool."""
        start_time = time.time()

        try:
            self.stats.total_requests += 1
            self.stats.active_connections += 1

            yield self._client

            self.stats.successful_requests += 1
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)

        except Exception as e:
            self.stats.failed_requests += 1
            logger.error("Redis connection error", error=str(e))
            raise
        finally:
            self.stats.active_connections = max(0, self.stats.active_connections - 1)

    def _start_health_monitoring(self) -> None:
        """Start health check monitoring."""
        if self.config.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Redis health check error", error=str(e))

    async def _perform_health_check(self) -> None:
        """Perform Redis health check."""
        try:
            async with self.get_connection() as client:
                await client.ping()
            logger.debug("Redis health check passed")
        except Exception as e:
            self.stats.connection_errors += 1
            logger.error("Redis health check failed", error=str(e))

    def _update_avg_response_time(self, response_time: float) -> None:
        """Update average response time."""
        if self.stats.successful_requests > 1:
            self.stats.avg_response_time = (
                self.stats.avg_response_time * (self.stats.successful_requests - 1)
                + response_time
            ) / self.stats.successful_requests
        else:
            self.stats.avg_response_time = response_time

    def get_pool_info(self) -> dict[str, Any]:
        """Get current pool information."""
        return {
            "created_connections": self._pool.created_connections,
            "available_connections": len(self._pool._available_connections),
            "in_use_connections": len(self._pool._in_use_connections),
            "pool_size": self.config.max_size,
        }

    async def close(self) -> None:
        """Close Redis pool."""
        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            await self._client.close()
            logger.info("Redis pool closed")
        except Exception as e:
            logger.error("Error closing Redis pool", error=str(e))


class HTTPConnectionPool:
    """HTTP connection pool manager."""

    def __init__(self, config: PoolConfiguration):
        """Initialize HTTP pool.

        Args:
            config: Pool configuration
        """
        self.config = config
        self.stats = PoolStats(pool_type=PoolType.HTTP)

        # Create HTTP client with connection pooling
        limits = httpx.Limits(
            max_keepalive_connections=config.max_size,
            max_connections=config.max_size + config.max_overflow,
            keepalive_expiry=config.max_idle_time,
        )

        timeout = httpx.Timeout(
            connect=config.timeout,
            read=config.timeout,
            write=config.timeout,
            pool=config.pool_timeout,
        )

        self._client = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            http2=True,  # Enable HTTP/2 for better performance
        )

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Get HTTP client from pool."""
        start_time = time.time()

        try:
            self.stats.total_requests += 1
            self.stats.active_connections += 1

            yield self._client

            self.stats.successful_requests += 1
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)

        except Exception as e:
            self.stats.failed_requests += 1
            logger.error("HTTP client error", error=str(e))
            raise
        finally:
            self.stats.active_connections = max(0, self.stats.active_connections - 1)

    def _update_avg_response_time(self, response_time: float) -> None:
        """Update average response time."""
        if self.stats.successful_requests > 1:
            self.stats.avg_response_time = (
                self.stats.avg_response_time * (self.stats.successful_requests - 1)
                + response_time
            ) / self.stats.successful_requests
        else:
            self.stats.avg_response_time = response_time

    def get_pool_info(self) -> dict[str, Any]:
        """Get current pool information."""
        return {
            "is_closed": self._client.is_closed,
            "limits": {
                "max_keepalive_connections": self._client._limits.max_keepalive_connections,
                "max_connections": self._client._limits.max_connections,
                "keepalive_expiry": self._client._limits.keepalive_expiry,
            },
        }

    async def close(self) -> None:
        """Close HTTP pool."""
        try:
            await self._client.aclose()
            logger.info("HTTP pool closed")
        except Exception as e:
            logger.error("Error closing HTTP pool", error=str(e))


class ConnectionPoolManager:
    """Central manager for all connection pools."""

    def __init__(self):
        """Initialize connection pool manager."""
        self._pools: dict[str, DatabasePool | RedisPool | HTTPConnectionPool] = {}
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

    def create_database_pool(
        self,
        name: str,
        database_url: str,
        config: PoolConfiguration | None = None,
        async_mode: bool = False,
    ) -> DatabasePool:
        """Create database connection pool.

        Args:
            name: Pool identifier
            database_url: Database connection URL
            config: Pool configuration
            async_mode: Whether to use async engine

        Returns:
            Database pool instance
        """
        config = config or PoolConfiguration()

        with self._lock:
            if name in self._pools:
                raise ValueError(f"Pool '{name}' already exists")

            pool = DatabasePool(database_url, config, async_mode)
            self._pools[name] = pool

            logger.info(
                "Created database pool",
                name=name,
                async_mode=async_mode,
                min_size=config.min_size,
                max_size=config.max_size,
            )

            return pool

    def create_redis_pool(
        self, name: str, redis_url: str, config: PoolConfiguration | None = None
    ) -> RedisPool:
        """Create Redis connection pool.

        Args:
            name: Pool identifier
            redis_url: Redis connection URL
            config: Pool configuration

        Returns:
            Redis pool instance
        """
        config = config or PoolConfiguration()

        with self._lock:
            if name in self._pools:
                raise ValueError(f"Pool '{name}' already exists")

            pool = RedisPool(redis_url, config)
            self._pools[name] = pool

            logger.info(
                "Created Redis pool",
                name=name,
                max_size=config.max_size,
                timeout=config.timeout,
            )

            return pool

    def create_http_pool(
        self, name: str, config: PoolConfiguration | None = None
    ) -> HTTPConnectionPool:
        """Create HTTP connection pool.

        Args:
            name: Pool identifier
            config: Pool configuration

        Returns:
            HTTP pool instance
        """
        config = config or PoolConfiguration()

        with self._lock:
            if name in self._pools:
                raise ValueError(f"Pool '{name}' already exists")

            pool = HTTPConnectionPool(config)
            self._pools[name] = pool

            logger.info(
                "Created HTTP pool",
                name=name,
                max_size=config.max_size,
                timeout=config.timeout,
            )

            return pool

    def get_pool(self, name: str) -> DatabasePool | RedisPool | HTTPConnectionPool:
        """Get pool by name.

        Args:
            name: Pool identifier

        Returns:
            Pool instance

        Raises:
            KeyError: If pool doesn't exist
        """
        with self._lock:
            if name not in self._pools:
                raise KeyError(f"Pool '{name}' not found")
            return self._pools[name]

    def list_pools(self) -> list[str]:
        """List all pool names."""
        with self._lock:
            return list(self._pools.keys())

    def get_all_stats(self) -> dict[str, PoolStats]:
        """Get statistics for all pools."""
        with self._lock:
            return {name: pool.stats for name, pool in self._pools.items()}

    def get_pool_info(self, name: str) -> dict[str, Any]:
        """Get detailed pool information.

        Args:
            name: Pool identifier

        Returns:
            Pool information dictionary
        """
        pool = self.get_pool(name)

        info = {
            "name": name,
            "type": pool.stats.pool_type.value,
            "stats": pool.stats,
            "pool_info": pool.get_pool_info(),
            "configuration": pool.config if hasattr(pool, "config") else None,
        }

        return info

    async def close_pool(self, name: str) -> None:
        """Close specific pool.

        Args:
            name: Pool identifier
        """
        with self._lock:
            if name not in self._pools:
                return

            pool = self._pools.pop(name)

        await pool.close()
        logger.info("Closed pool", name=name)

    async def close_all_pools(self) -> None:
        """Close all pools."""
        pools_to_close = []

        with self._lock:
            pools_to_close = list(self._pools.values())
            self._pools.clear()

        # Close all pools concurrently
        if pools_to_close:
            await asyncio.gather(
                *[pool.close() for pool in pools_to_close], return_exceptions=True
            )

        logger.info("Closed all connection pools", count=len(pools_to_close))

    def reset_stats(self, name: str | None = None) -> None:
        """Reset pool statistics.

        Args:
            name: Pool name (None for all pools)
        """
        with self._lock:
            if name:
                if name in self._pools:
                    pool = self._pools[name]
                    pool.stats.last_reset = time.time()
                    # Reset counters
                    pool.stats.total_requests = 0
                    pool.stats.successful_requests = 0
                    pool.stats.failed_requests = 0
                    pool.stats.connections_created = 0
                    pool.stats.connections_closed = 0
                    pool.stats.connections_recycled = 0
                    pool.stats.connection_errors = 0
                    pool.stats.avg_response_time = 0.0
            else:
                for pool in self._pools.values():
                    pool.stats.last_reset = time.time()
                    pool.stats.total_requests = 0
                    pool.stats.successful_requests = 0
                    pool.stats.failed_requests = 0
                    pool.stats.connections_created = 0
                    pool.stats.connections_closed = 0
                    pool.stats.connections_recycled = 0
                    pool.stats.connection_errors = 0
                    pool.stats.avg_response_time = 0.0


# Global connection pool manager
_connection_pool_manager: ConnectionPoolManager | None = None


def get_connection_pool_manager() -> ConnectionPoolManager:
    """Get global connection pool manager instance."""
    global _connection_pool_manager
    if _connection_pool_manager is None:
        _connection_pool_manager = ConnectionPoolManager()
    return _connection_pool_manager
