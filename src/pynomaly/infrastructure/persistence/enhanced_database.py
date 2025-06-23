"""Enhanced database manager with connection pooling and query optimization."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from pynomaly.infrastructure.performance import (
    ConnectionPoolManager,
    QueryOptimizer,
    PoolConfiguration,
    get_connection_pool_manager
)

logger = structlog.get_logger(__name__)


class EnhancedDatabaseManager:
    """Enhanced database manager with connection pooling and query optimization."""
    
    def __init__(
        self,
        database_url: str,
        pool_config: Optional[PoolConfiguration] = None,
        enable_query_optimization: bool = True
    ):
        """Initialize enhanced database manager.
        
        Args:
            database_url: Database connection URL
            pool_config: Connection pool configuration
            enable_query_optimization: Whether to enable query optimization
        """
        self.database_url = database_url
        self.pool_config = pool_config or PoolConfiguration()
        self.enable_query_optimization = enable_query_optimization
        
        # Connection pool manager
        self.pool_manager = get_connection_pool_manager()
        
        # Create database pool
        self._database_pool = None
        self._query_optimizer = None
        self._session_factory = None
        
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database with connection pooling."""
        try:
            # Create database connection pool
            self._database_pool = self.pool_manager.create_database_pool(
                name="main_database",
                database_url=self.database_url,
                config=self.pool_config,
                async_mode=True
            )
            
            # Create async session factory
            async_engine = self._database_pool._async_engine
            self._session_factory = sessionmaker(
                bind=async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Initialize query optimizer if enabled
            if self.enable_query_optimization:
                self._query_optimizer = QueryOptimizer(
                    engine=async_engine,
                    cache_size=1000,
                    cache_ttl=3600
                )
            
            logger.info(
                "Database initialized with connection pooling",
                pool_min_size=self.pool_config.min_size,
                pool_max_size=self.pool_config.max_size,
                query_optimization_enabled=self.enable_query_optimization
            )
            
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session from pool.
        
        Yields:
            AsyncSession: Database session
        """
        if not self._session_factory:
            raise RuntimeError("Database not initialized")
        
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    @asynccontextmanager
    async def get_optimized_session(self) -> AsyncGenerator[tuple[AsyncSession, Optional[QueryOptimizer]], None]:
        """Get database session with query optimizer.
        
        Yields:
            Tuple of (AsyncSession, QueryOptimizer): Database session and optimizer
        """
        async with self.get_session() as session:
            yield session, self._query_optimizer
    
    async def execute_optimized_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        cache_ttl: Optional[float] = None
    ) -> Any:
        """Execute query with optimization.
        
        Args:
            query: SQL query
            params: Query parameters
            use_cache: Whether to use cache
            cache_ttl: Cache TTL override
            
        Returns:
            Query result
        """
        if not self._query_optimizer:
            raise RuntimeError("Query optimizer not enabled")
        
        return await self._query_optimizer.execute_with_optimization(
            query=query,
            params=params,
            use_cache=use_cache,
            cache_ttl=cache_ttl
        )
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Perform database optimization.
        
        Returns:
            Optimization results
        """
        if not self._query_optimizer:
            return {"error": "Query optimizer not enabled"}
        
        return await self._query_optimizer.optimize_database()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        if not self._database_pool:
            return {}
        
        return {
            "pool_stats": self._database_pool.stats,
            "pool_info": self._database_pool.get_pool_info(),
        }
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics.
        
        Returns:
            Query statistics
        """
        if not self._query_optimizer:
            return {}
        
        return self._query_optimizer.performance_tracker.get_performance_summary()
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report.
        
        Returns:
            Optimization report
        """
        if not self._query_optimizer:
            return {"error": "Query optimizer not enabled"}
        
        return await self._query_optimizer.get_optimization_report()
    
    async def clear_query_cache(self) -> None:
        """Clear query cache."""
        if self._query_optimizer:
            await self._query_optimizer.clear_cache()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check.
        
        Returns:
            Health check results
        """
        health_status = {
            "database_available": False,
            "pool_healthy": False,
            "query_optimizer_available": self._query_optimizer is not None,
            "error": None
        }
        
        try:
            # Test database connection
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                await result.fetchone()
                health_status["database_available"] = True
            
            # Check pool health
            if self._database_pool:
                pool_info = self._database_pool.get_pool_info()
                # Consider pool healthy if it has some available connections
                # and not too many errors
                error_rate = (
                    self._database_pool.stats.failed_requests / 
                    max(1, self._database_pool.stats.total_requests)
                )
                health_status["pool_healthy"] = error_rate < 0.1  # Less than 10% error rate
                health_status["pool_info"] = pool_info
            
        except Exception as e:
            health_status["error"] = str(e)
            logger.error("Database health check failed", error=str(e))
        
        return health_status
    
    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        try:
            # Close query optimizer
            if self._query_optimizer:
                await self._query_optimizer.close()
            
            # Close database pool
            if self._database_pool:
                await self.pool_manager.close_pool("main_database")
            
            logger.info("Database manager closed")
            
        except Exception as e:
            logger.error("Error closing database manager", error=str(e))


class OptimizedRepository:
    """Base repository with query optimization support."""
    
    def __init__(self, db_manager: EnhancedDatabaseManager):
        """Initialize optimized repository.
        
        Args:
            db_manager: Enhanced database manager
        """
        self.db_manager = db_manager
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def execute_cached_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl: Optional[float] = None
    ) -> Any:
        """Execute query with caching.
        
        Args:
            query: SQL query
            params: Query parameters
            cache_ttl: Cache TTL in seconds
            
        Returns:
            Query result
        """
        try:
            return await self.db_manager.execute_optimized_query(
                query=query,
                params=params,
                use_cache=True,
                cache_ttl=cache_ttl
            )
        except Exception as e:
            self.logger.error(
                "Cached query execution failed",
                query=query[:100],
                error=str(e)
            )
            raise
    
    async def execute_uncached_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute query without caching.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Query result
        """
        try:
            return await self.db_manager.execute_optimized_query(
                query=query,
                params=params,
                use_cache=False
            )
        except Exception as e:
            self.logger.error(
                "Uncached query execution failed",
                query=query[:100],
                error=str(e)
            )
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session.
        
        Yields:
            AsyncSession: Database session
        """
        async with self.db_manager.get_session() as session:
            yield session
    
    @asynccontextmanager
    async def get_optimized_session(self) -> AsyncGenerator[tuple[AsyncSession, Optional[QueryOptimizer]], None]:
        """Get database session with optimizer.
        
        Yields:
            Tuple of (AsyncSession, QueryOptimizer): Session and optimizer
        """
        async with self.db_manager.get_optimized_session() as (session, optimizer):
            yield session, optimizer


class DatabaseHealthMonitor:
    """Monitor database health and performance."""
    
    def __init__(self, db_manager: EnhancedDatabaseManager, check_interval: float = 60.0):
        """Initialize database health monitor.
        
        Args:
            db_manager: Enhanced database manager
            check_interval: Health check interval in seconds
        """
        self.db_manager = db_manager
        self.check_interval = check_interval
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_history: list[Dict[str, Any]] = []
        self._max_history = 100
    
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Database health monitoring started", interval=self.check_interval)
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Database health monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                health_status = await self.db_manager.health_check()
                
                # Add timestamp
                health_status["timestamp"] = __import__("time").time()
                
                # Store in history
                self._health_history.append(health_status)
                if len(self._health_history) > self._max_history:
                    self._health_history.pop(0)
                
                # Log issues
                if not health_status["database_available"]:
                    logger.error("Database health check failed", status=health_status)
                elif not health_status["pool_healthy"]:
                    logger.warning("Database pool unhealthy", status=health_status)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
    
    def get_health_history(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """Get health check history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of health check results
        """
        if limit is None:
            return self._health_history.copy()
        return self._health_history[-limit:].copy()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary.
        
        Returns:
            Health summary
        """
        if not self._health_history:
            return {"status": "no_data"}
        
        recent_checks = self._health_history[-10:]  # Last 10 checks
        
        database_availability = sum(
            1 for check in recent_checks if check["database_available"]
        ) / len(recent_checks)
        
        pool_health = sum(
            1 for check in recent_checks if check["pool_healthy"]
        ) / len(recent_checks)
        
        return {
            "total_checks": len(self._health_history),
            "recent_database_availability": database_availability,
            "recent_pool_health": pool_health,
            "last_check": self._health_history[-1] if self._health_history else None,
            "status": "healthy" if database_availability > 0.9 and pool_health > 0.8 else "unhealthy"
        }