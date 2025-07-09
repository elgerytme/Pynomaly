"""Comprehensive cache integration layer for Pynomaly application."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
from contextlib import asynccontextmanager

from pynomaly.infrastructure.config.settings import Settings
from pynomaly.shared.error_handling import (
    PynamolyError,
    InfrastructureError,
    ErrorCodes,
    create_infrastructure_error,
)

from .redis_cache import RedisCache, init_cache, get_cache
from .intelligent_cache import IntelligentCacheManager, get_intelligent_cache_manager
from .decorators import (
    set_cache_manager,
    get_cache_manager,
    get_cache_invalidator,
    cached,
    cache_result,
    cache_expensive,
    cache_model_prediction,
    cache_database_query,
)

logger = logging.getLogger(__name__)


@dataclass
class CacheConfiguration:
    """Cache configuration settings."""
    enabled: bool = True
    redis_enabled: bool = True
    redis_url: Optional[str] = None
    default_ttl: int = 3600
    max_memory_size: int = 100 * 1024 * 1024  # 100MB
    compression_threshold: int = 1024
    prefetch_enabled: bool = True
    adaptive_ttl: bool = True
    health_check_interval: int = 60
    
    @classmethod
    def from_settings(cls, settings: Settings) -> CacheConfiguration:
        """Create configuration from settings."""
        return cls(
            enabled=settings.cache_enabled,
            redis_enabled=settings.cache_enabled and settings.redis_url is not None,
            redis_url=settings.redis_url,
            default_ttl=settings.cache_ttl,
        )
    
    @classmethod
    def from_environment(cls) -> CacheConfiguration:
        """Create configuration from environment variables."""
        return cls(
            enabled=os.getenv("PYNOMALY_CACHE_ENABLED", "true").lower() == "true",
            redis_enabled=os.getenv("PYNOMALY_REDIS_ENABLED", "true").lower() == "true",
            redis_url=os.getenv("PYNOMALY_REDIS_URL"),
            default_ttl=int(os.getenv("PYNOMALY_CACHE_TTL", "3600")),
            max_memory_size=int(os.getenv("PYNOMALY_CACHE_MEMORY_SIZE", str(100 * 1024 * 1024))),
            compression_threshold=int(os.getenv("PYNOMALY_CACHE_COMPRESSION_THRESHOLD", "1024")),
            prefetch_enabled=os.getenv("PYNOMALY_CACHE_PREFETCH", "true").lower() == "true",
            adaptive_ttl=os.getenv("PYNOMALY_CACHE_ADAPTIVE_TTL", "true").lower() == "true",
            health_check_interval=int(os.getenv("PYNOMALY_CACHE_HEALTH_CHECK_INTERVAL", "60")),
        )


class CacheHealthMonitor:
    """Health monitoring for cache systems."""
    
    def __init__(
        self,
        cache_manager: IntelligentCacheManager,
        check_interval: int = 60,
    ):
        """Initialize cache health monitor.
        
        Args:
            cache_manager: Cache manager to monitor
            check_interval: Health check interval in seconds
        """
        self.cache_manager = cache_manager
        self.check_interval = check_interval
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_history: List[Dict[str, Any]] = []
        self.max_history = 100
    
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Cache health monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            logger.info("Cache health monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                health_status = await self.check_health()
                
                # Store in history
                self.health_history.append(health_status)
                if len(self.health_history) > self.max_history:
                    self.health_history.pop(0)
                
                # Log warnings for poor health
                if health_status["overall_health"] == "degraded":
                    logger.warning("Cache health degraded", extra=health_status)
                elif health_status["overall_health"] == "unhealthy":
                    logger.error("Cache health unhealthy", extra=health_status)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache health monitoring error: {e}")
    
    async def check_health(self) -> Dict[str, Any]:
        """Check cache health status."""
        try:
            stats = await self.cache_manager.get_stats()
            
            # Determine health status
            cache_stats = stats["cache_stats"]
            memory_stats = stats["memory_cache"]
            
            # Health indicators
            hit_rate = cache_stats["hit_rate"]
            memory_utilization = memory_stats["utilization"]
            avg_access_time = cache_stats["avg_access_time"]
            
            # Overall health determination
            health_score = 0
            issues = []
            
            # Check hit rate
            if hit_rate >= 0.8:
                health_score += 3
            elif hit_rate >= 0.6:
                health_score += 2
            elif hit_rate >= 0.4:
                health_score += 1
            else:
                issues.append(f"Low hit rate: {hit_rate:.2%}")
            
            # Check memory utilization
            if memory_utilization <= 0.8:
                health_score += 2
            elif memory_utilization <= 0.9:
                health_score += 1
            else:
                issues.append(f"High memory utilization: {memory_utilization:.2%}")
            
            # Check access time
            if avg_access_time <= 0.01:  # 10ms
                health_score += 2
            elif avg_access_time <= 0.05:  # 50ms
                health_score += 1
            else:
                issues.append(f"Slow access time: {avg_access_time:.3f}s")
            
            # Check Redis connection
            redis_healthy = True
            try:
                redis_healthy = self.cache_manager.redis_cache.exists("health_check")
            except Exception:
                redis_healthy = False
                issues.append("Redis connection failed")
            
            if redis_healthy:
                health_score += 1
            
            # Determine overall health
            if health_score >= 7:
                overall_health = "healthy"
            elif health_score >= 5:
                overall_health = "degraded"
            else:
                overall_health = "unhealthy"
            
            return {
                "overall_health": overall_health,
                "health_score": health_score,
                "max_score": 8,
                "issues": issues,
                "metrics": {
                    "hit_rate": hit_rate,
                    "memory_utilization": memory_utilization,
                    "avg_access_time": avg_access_time,
                    "redis_healthy": redis_healthy,
                },
                "timestamp": asyncio.get_event_loop().time(),
            }
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                "overall_health": "unhealthy",
                "health_score": 0,
                "max_score": 8,
                "issues": [f"Health check failed: {str(e)}"],
                "metrics": {},
                "timestamp": asyncio.get_event_loop().time(),
            }
    
    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get health history."""
        return self.health_history[-limit:]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        if not self.health_history:
            return {"status": "no_data"}
        
        recent_checks = self.health_history[-10:]
        
        healthy_count = sum(1 for check in recent_checks if check["overall_health"] == "healthy")
        degraded_count = sum(1 for check in recent_checks if check["overall_health"] == "degraded")
        unhealthy_count = sum(1 for check in recent_checks if check["overall_health"] == "unhealthy")
        
        return {
            "total_checks": len(self.health_history),
            "recent_health_distribution": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
            },
            "current_status": self.health_history[-1]["overall_health"],
            "stability": healthy_count / len(recent_checks) if recent_checks else 0,
        }


class CacheIntegrationManager:
    """Manages cache integration across the application."""
    
    def __init__(self, config: CacheConfiguration):
        """Initialize cache integration manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.redis_cache: Optional[RedisCache] = None
        self.intelligent_cache: Optional[IntelligentCacheManager] = None
        self.health_monitor: Optional[CacheHealthMonitor] = None
        
        # Initialize cache systems
        self._initialize_caches()
    
    def _initialize_caches(self) -> None:
        """Initialize cache systems."""
        try:
            if not self.config.enabled:
                logger.info("Caching disabled by configuration")
                return
            
            # Initialize Redis cache
            if self.config.redis_enabled and self.config.redis_url:
                settings = Settings()
                settings.cache_enabled = True
                settings.redis_url = self.config.redis_url
                settings.cache_ttl = self.config.default_ttl
                
                self.redis_cache = init_cache(settings)
                logger.info("Redis cache initialized")
            else:
                logger.warning("Redis cache disabled or not configured")
                return
            
            # Initialize intelligent cache manager
            if self.redis_cache:
                self.intelligent_cache = get_intelligent_cache_manager(
                    redis_cache=self.redis_cache,
                    max_memory_size=self.config.max_memory_size,
                    compression_threshold=self.config.compression_threshold,
                    prefetch_enabled=self.config.prefetch_enabled,
                    adaptive_ttl=self.config.adaptive_ttl,
                )
                
                # Set global cache manager for decorators
                set_cache_manager(self.intelligent_cache)
                logger.info("Intelligent cache manager initialized")
            
            # Initialize health monitoring
            if self.intelligent_cache:
                self.health_monitor = CacheHealthMonitor(
                    cache_manager=self.intelligent_cache,
                    check_interval=self.config.health_check_interval,
                )
                self.health_monitor.start_monitoring()
                logger.info("Cache health monitoring initialized")
                
        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_CACHE_UNAVAILABLE,
                message=f"Failed to initialize cache system: {str(e)}",
                cause=e,
            )
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "configuration": {
                "enabled": self.config.enabled,
                "redis_enabled": self.config.redis_enabled,
                "default_ttl": self.config.default_ttl,
                "max_memory_size": self.config.max_memory_size,
                "compression_threshold": self.config.compression_threshold,
                "prefetch_enabled": self.config.prefetch_enabled,
                "adaptive_ttl": self.config.adaptive_ttl,
            },
            "redis_cache": {
                "available": self.redis_cache is not None,
                "enabled": self.redis_cache.enabled if self.redis_cache else False,
            },
            "intelligent_cache": {
                "available": self.intelligent_cache is not None,
            },
            "health_monitoring": {
                "available": self.health_monitor is not None,
            },
        }
        
        # Add intelligent cache stats
        if self.intelligent_cache:
            cache_stats = await self.intelligent_cache.get_stats()
            stats["intelligent_cache"]["stats"] = cache_stats
        
        # Add health monitoring stats
        if self.health_monitor:
            health_summary = self.health_monitor.get_health_summary()
            stats["health_monitoring"]["summary"] = health_summary
            
            # Current health status
            current_health = await self.health_monitor.check_health()
            stats["health_monitoring"]["current_status"] = current_health
        
        return stats
    
    async def perform_maintenance(self) -> Dict[str, Any]:
        """Perform cache maintenance operations."""
        maintenance_results = {
            "timestamp": asyncio.get_event_loop().time(),
            "operations": [],
            "errors": [],
        }
        
        if not self.intelligent_cache:
            maintenance_results["errors"].append("No intelligent cache available")
            return maintenance_results
        
        try:
            # Get current stats
            stats = await self.intelligent_cache.get_stats()
            maintenance_results["operations"].append("Retrieved cache statistics")
            
            # Optimize cache if needed
            cache_stats = stats["cache_stats"]
            if cache_stats["hit_rate"] < 0.5:
                # Low hit rate - might need optimization
                maintenance_results["operations"].append("Detected low hit rate")
                
                # Could trigger cache warming here
                # await self.warm_critical_cache()
                
            # Check memory utilization
            memory_stats = stats["memory_cache"]
            if memory_stats["utilization"] > 0.9:
                maintenance_results["operations"].append("High memory utilization detected")
                
                # Could trigger cache cleanup here
                # await self.cleanup_cache()
            
            maintenance_results["operations"].append("Cache maintenance completed")
            
        except Exception as e:
            error_msg = f"Cache maintenance error: {str(e)}"
            maintenance_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        return maintenance_results
    
    async def warm_critical_cache(self) -> int:
        """Warm cache with critical data."""
        if not self.intelligent_cache:
            return 0
        
        # Define critical cache keys and loaders
        critical_keys = [
            # Add your critical cache warming logic here
            # ("detector_configs", lambda: load_detector_configs()),
            # ("algorithm_metadata", lambda: load_algorithm_metadata()),
        ]
        
        return await self.intelligent_cache.warm_cache(critical_keys)
    
    async def cleanup_cache(self) -> int:
        """Clean up cache entries."""
        if not self.intelligent_cache:
            return 0
        
        # Define cleanup patterns
        cleanup_patterns = [
            "temp:*",
            "expired:*",
            "debug:*",
        ]
        
        total_deleted = 0
        for pattern in cleanup_patterns:
            deleted = await self.intelligent_cache.delete_pattern(pattern)
            total_deleted += deleted
        
        return total_deleted
    
    async def emergency_cache_reset(self) -> Dict[str, Any]:
        """Emergency cache reset."""
        reset_results = {
            "timestamp": asyncio.get_event_loop().time(),
            "actions": [],
            "status": "success",
        }
        
        try:
            # Clear Redis cache
            if self.redis_cache:
                self.redis_cache.clear()
                reset_results["actions"].append("Redis cache cleared")
            
            # Reinitialize intelligent cache
            if self.intelligent_cache:
                await self.intelligent_cache.close()
                
                # Reinitialize
                self.intelligent_cache = get_intelligent_cache_manager(
                    redis_cache=self.redis_cache,
                    max_memory_size=self.config.max_memory_size,
                    compression_threshold=self.config.compression_threshold,
                    prefetch_enabled=self.config.prefetch_enabled,
                    adaptive_ttl=self.config.adaptive_ttl,
                )
                
                set_cache_manager(self.intelligent_cache)
                reset_results["actions"].append("Intelligent cache reinitialized")
            
            logger.info("Emergency cache reset completed")
            
        except Exception as e:
            reset_results["status"] = "failed"
            reset_results["error"] = str(e)
            logger.error(f"Emergency cache reset failed: {e}")
        
        return reset_results
    
    async def close(self) -> None:
        """Close cache integration manager."""
        try:
            # Stop health monitoring
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
            
            # Close intelligent cache
            if self.intelligent_cache:
                await self.intelligent_cache.close()
            
            # Close Redis cache
            if self.redis_cache:
                self.redis_cache.close()
            
            logger.info("Cache integration manager closed")
            
        except Exception as e:
            logger.error(f"Error closing cache integration manager: {e}")


# Global cache integration manager
_cache_integration_manager: Optional[CacheIntegrationManager] = None


def get_cache_integration_manager(
    config: Optional[CacheConfiguration] = None,
) -> CacheIntegrationManager:
    """Get or create global cache integration manager."""
    global _cache_integration_manager
    
    if _cache_integration_manager is None:
        if config is None:
            config = CacheConfiguration.from_environment()
        
        _cache_integration_manager = CacheIntegrationManager(config)
    
    return _cache_integration_manager


async def close_cache_integration_manager() -> None:
    """Close global cache integration manager."""
    global _cache_integration_manager
    
    if _cache_integration_manager:
        await _cache_integration_manager.close()
        _cache_integration_manager = None


@asynccontextmanager
async def cache_system_context(config: Optional[CacheConfiguration] = None):
    """Context manager for cache system lifecycle."""
    manager = get_cache_integration_manager(config)
    
    try:
        yield manager
    finally:
        await close_cache_integration_manager()


# Convenience functions
async def get_cache_health() -> Dict[str, Any]:
    """Get cache health status."""
    manager = get_cache_integration_manager()
    if manager.health_monitor:
        return await manager.health_monitor.check_health()
    return {"status": "no_monitoring"}


async def get_cache_statistics() -> Dict[str, Any]:
    """Get cache statistics."""
    manager = get_cache_integration_manager()
    return await manager.get_comprehensive_stats()


async def perform_cache_maintenance() -> Dict[str, Any]:
    """Perform cache maintenance."""
    manager = get_cache_integration_manager()
    return await manager.perform_maintenance()


async def warm_cache_with_critical_data() -> int:
    """Warm cache with critical data."""
    manager = get_cache_integration_manager()
    return await manager.warm_critical_cache()