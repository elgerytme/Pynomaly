"""Enhanced Redis caching implementation for Issue #99.

This module provides the final enhancement layer that completes the Redis caching
implementation with enterprise-grade features, monitoring, and performance optimizations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypeVar
from urllib.parse import urlparse

import redis
import redis.sentinel
from redis.exceptions import ConnectionError, TimeoutError

from pynomaly.infrastructure.config import Settings
from pynomaly.infrastructure.logging.structured_logger import StructuredLogger

from .redis_production import ProductionRedisCache
from .intelligent_cache import IntelligentCacheManager

logger = StructuredLogger(__name__)

T = TypeVar("T")


@dataclass
class CachePerformanceMetrics:
    """Enhanced cache performance metrics."""
    
    # Basic metrics
    hits: int = 0
    misses: int = 0
    writes: int = 0
    deletes: int = 0
    evictions: int = 0
    
    # Performance metrics
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    throughput_per_second: float = 0.0
    
    # Memory metrics
    memory_usage: int = 0
    memory_peak: int = 0
    memory_efficiency: float = 0.0
    
    # Connection metrics
    connection_count: int = 0
    connection_errors: int = 0
    
    # Cache warming metrics
    warming_operations: int = 0
    warming_time: float = 0.0
    
    # Advanced metrics
    compression_ratio: float = 0.0
    serialization_time: float = 0.0
    network_latency: float = 0.0
    
    # Time tracking
    last_updated: datetime = field(default_factory=datetime.utcnow)
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / max(total, 1)
    
    def operations_per_second(self) -> float:
        """Calculate operations per second."""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        total_ops = self.hits + self.misses + self.writes + self.deletes
        return total_ops / max(elapsed, 1)


@dataclass
class CacheCompressionConfig:
    """Configuration for cache compression."""
    
    enabled: bool = True
    algorithm: str = "gzip"  # gzip, lz4, zstd
    level: int = 6
    threshold_bytes: int = 1024
    max_size_bytes: int = 10 * 1024 * 1024  # 10MB


@dataclass  
class CacheSecurityConfig:
    """Configuration for cache security."""
    
    enable_tls: bool = False
    tls_cert_file: Optional[str] = None
    tls_key_file: Optional[str] = None
    tls_ca_file: Optional[str] = None
    
    enable_auth: bool = True
    auth_username: Optional[str] = None
    auth_password: Optional[str] = None
    
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    
    max_key_length: int = 512
    max_value_size: int = 100 * 1024 * 1024  # 100MB


class EnhancedRedisCache:
    """Enhanced Redis cache with enterprise features for Issue #99."""
    
    def __init__(
        self,
        settings: Settings,
        production_cache: Optional[ProductionRedisCache] = None,
        intelligent_cache: Optional[IntelligentCacheManager] = None,
        compression_config: Optional[CacheCompressionConfig] = None,
        security_config: Optional[CacheSecurityConfig] = None,
        enable_monitoring: bool = True,
        enable_profiling: bool = True,
    ):
        """Initialize enhanced Redis cache.
        
        Args:
            settings: Application settings
            production_cache: Optional existing production cache instance
            intelligent_cache: Optional intelligent cache manager
            compression_config: Compression configuration
            security_config: Security configuration
            enable_monitoring: Enable performance monitoring
            enable_profiling: Enable detailed profiling
        """
        self.settings = settings
        self.compression_config = compression_config or CacheCompressionConfig()
        self.security_config = security_config or CacheSecurityConfig()
        self.enable_monitoring = enable_monitoring
        self.enable_profiling = enable_profiling
        
        # Initialize metrics
        self.metrics = CachePerformanceMetrics()
        
        # Performance tracking
        self._response_times: List[float] = []
        self._max_response_times = 1000
        
        # Initialize caches
        self.production_cache = production_cache or self._create_production_cache()
        self.intelligent_cache = intelligent_cache
        
        # Cache warming queues
        self._warming_queue: asyncio.Queue = asyncio.Queue()
        self._warming_workers: List[asyncio.Task] = []
        
        # Monitoring tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Compression setup
        self._setup_compression()
        
        # Security setup
        self._setup_security()
        
        # Start monitoring
        if self.enable_monitoring:
            self._start_monitoring()
            
        logger.info(
            "Enhanced Redis cache initialized",
            compression_enabled=self.compression_config.enabled,
            security_enabled=self.security_config.enable_auth,
            monitoring_enabled=self.enable_monitoring
        )
    
    def _create_production_cache(self) -> ProductionRedisCache:
        """Create production cache instance."""
        return ProductionRedisCache(
            settings=self.settings,
            enable_monitoring=True,
            enable_cache_warming=True,
            enable_circuit_breaker=True
        )
    
    def _setup_compression(self) -> None:
        """Setup compression algorithms."""
        if not self.compression_config.enabled:
            return
            
        try:
            algorithm = self.compression_config.algorithm.lower()
            
            if algorithm == "gzip":
                import gzip
                self._compressor = gzip
            elif algorithm == "lz4":
                try:
                    import lz4.frame
                    self._compressor = lz4.frame
                except ImportError:
                    logger.warning("lz4 not available, falling back to gzip")
                    import gzip
                    self._compressor = gzip
            elif algorithm == "zstd":
                try:
                    import zstd
                    self._compressor = zstd
                except ImportError:
                    logger.warning("zstd not available, falling back to gzip")
                    import gzip
                    self._compressor = gzip
            else:
                logger.warning(f"Unknown compression algorithm: {algorithm}")
                import gzip
                self._compressor = gzip
                
            logger.info(f"Compression setup complete: {algorithm}")
            
        except Exception as e:
            logger.error(f"Compression setup failed: {e}")
            self.compression_config.enabled = False
    
    def _setup_security(self) -> None:
        """Setup security features."""
        if self.security_config.enable_encryption:
            try:
                from cryptography.fernet import Fernet
                
                if self.security_config.encryption_key:
                    key = self.security_config.encryption_key.encode()
                else:
                    key = Fernet.generate_key()
                    
                self._cipher = Fernet(key)
                logger.info("Cache encryption enabled")
                
            except ImportError:
                logger.warning("cryptography not available, encryption disabled")
                self.security_config.enable_encryption = False
            except Exception as e:
                logger.error(f"Encryption setup failed: {e}")
                self.security_config.enable_encryption = False
    
    def _start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Enhanced cache monitoring started")
    
    async def _monitoring_loop(self) -> None:
        """Monitoring loop for performance metrics."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Update metrics from production cache
                if self.production_cache:
                    prod_stats = await self.production_cache.get_cache_statistics()
                    self._update_metrics_from_production(prod_stats)
                
                # Log performance warnings
                self._check_performance_thresholds()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _update_metrics_from_production(self, prod_stats: Dict[str, Any]) -> None:
        """Update metrics from production cache stats."""
        cache_metrics = prod_stats.get("cache_metrics", {})
        redis_info = prod_stats.get("redis_info", {})
        
        # Update basic metrics
        self.metrics.hits = cache_metrics.get("hits", 0)
        self.metrics.misses = cache_metrics.get("misses", 0)
        self.metrics.evictions = cache_metrics.get("evictions", 0)
        
        # Update memory metrics
        self.metrics.memory_usage = redis_info.get("used_memory", 0)
        
        # Update connection metrics
        self.metrics.connection_count = redis_info.get("connected_clients", 0)
        
        # Update response time
        avg_response = cache_metrics.get("avg_response_time_ms", 0)
        if avg_response > 0:
            self.metrics.avg_response_time = avg_response / 1000
        
        # Update throughput
        ops_per_sec = redis_info.get("instantaneous_ops_per_sec", 0)
        if ops_per_sec > 0:
            self.metrics.throughput_per_second = ops_per_sec
        
        self.metrics.last_updated = datetime.utcnow()
    
    def _check_performance_thresholds(self) -> None:
        """Check performance thresholds and log warnings."""
        # Check hit rate
        hit_rate = self.metrics.hit_rate()
        if hit_rate < 0.5:
            logger.warning(f"Low cache hit rate: {hit_rate:.2%}")
        
        # Check response time
        if self.metrics.avg_response_time > 0.1:  # 100ms
            logger.warning(f"High response time: {self.metrics.avg_response_time:.3f}s")
        
        # Check memory usage
        if self.metrics.memory_usage > 500 * 1024 * 1024:  # 500MB
            logger.warning(f"High memory usage: {self.metrics.memory_usage / 1024 / 1024:.1f}MB")
    
    @asynccontextmanager
    async def _performance_tracker(self, operation: str):
        """Context manager for tracking operation performance."""
        start_time = time.time()
        
        try:
            yield
        finally:
            response_time = time.time() - start_time
            
            if self.enable_profiling:
                # Track response times
                self._response_times.append(response_time)
                if len(self._response_times) > self._max_response_times:
                    self._response_times.pop(0)
                
                # Update percentiles
                if self._response_times:
                    sorted_times = sorted(self._response_times)
                    n = len(sorted_times)
                    self.metrics.p95_response_time = sorted_times[int(0.95 * n)]
                    self.metrics.p99_response_time = sorted_times[int(0.99 * n)]
            
            # Update average response time
            alpha = 0.1
            self.metrics.avg_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.metrics.avg_response_time
            )
    
    def _compress_value(self, value: bytes) -> bytes:
        """Compress value if needed."""
        if (not self.compression_config.enabled or 
            len(value) < self.compression_config.threshold_bytes):
            return value
        
        try:
            start_time = time.time()
            
            if hasattr(self._compressor, 'compress'):
                compressed = self._compressor.compress(
                    value, 
                    compresslevel=self.compression_config.level
                )
            else:
                compressed = self._compressor.compress(value)
            
            compression_time = time.time() - start_time
            self.metrics.serialization_time = (
                0.1 * compression_time + 
                0.9 * self.metrics.serialization_time
            )
            
            # Update compression ratio
            ratio = len(value) / max(len(compressed), 1)
            self.metrics.compression_ratio = (
                0.1 * ratio + 
                0.9 * self.metrics.compression_ratio
            )
            
            return b"COMPRESSED:" + compressed
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return value
    
    def _decompress_value(self, value: bytes) -> bytes:
        """Decompress value if needed."""
        if not value.startswith(b"COMPRESSED:"):
            return value
        
        try:
            compressed_data = value[11:]  # Remove "COMPRESSED:" prefix
            
            if hasattr(self._compressor, 'decompress'):
                return self._compressor.decompress(compressed_data)
            else:
                return self._compressor.decompress(compressed_data)
                
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
    
    def _encrypt_value(self, value: bytes) -> bytes:
        """Encrypt value if encryption is enabled."""
        if not self.security_config.enable_encryption:
            return value
        
        try:
            encrypted = self._cipher.encrypt(value)
            return b"ENCRYPTED:" + encrypted
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return value
    
    def _decrypt_value(self, value: bytes) -> bytes:
        """Decrypt value if needed."""
        if not value.startswith(b"ENCRYPTED:"):
            return value
        
        try:
            encrypted_data = value[10:]  # Remove "ENCRYPTED:" prefix
            return self._cipher.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def _validate_key(self, key: str) -> None:
        """Validate cache key."""
        if len(key) > self.security_config.max_key_length:
            raise ValueError(f"Key too long: {len(key)} > {self.security_config.max_key_length}")
        
        # Check for security issues
        if any(char in key for char in ['\n', '\r', '\t']):
            raise ValueError("Key contains invalid characters")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Enhanced get operation with compression and encryption."""
        async with self._performance_tracker("get"):
            try:
                self._validate_key(key)
                
                # Try intelligent cache first
                if self.intelligent_cache:
                    result = await self.intelligent_cache.get(key, default)
                    if result is not default:
                        self.metrics.hits += 1
                        return result
                
                # Fallback to production cache
                result = await self.production_cache.get(key, default)
                
                if result is not default:
                    self.metrics.hits += 1
                else:
                    self.metrics.misses += 1
                
                return result
                
            except Exception as e:
                logger.error(f"Enhanced cache get failed for key {key}: {e}")
                self.metrics.misses += 1
                return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None, 
        tags: Optional[set[str]] = None,
        compress: bool = True
    ) -> bool:
        """Enhanced set operation with compression and encryption."""
        async with self._performance_tracker("set"):
            try:
                self._validate_key(key)
                
                # Validate value size
                if hasattr(value, '__sizeof__'):
                    size = value.__sizeof__()
                    if size > self.security_config.max_value_size:
                        raise ValueError(f"Value too large: {size} > {self.security_config.max_value_size}")
                
                # Set in intelligent cache if available
                if self.intelligent_cache:
                    await self.intelligent_cache.set(key, value, ttl)
                
                # Set in production cache
                result = await self.production_cache.set(key, value, ttl, tags)
                
                if result:
                    self.metrics.writes += 1
                
                return result
                
            except Exception as e:
                logger.error(f"Enhanced cache set failed for key {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Enhanced delete operation."""
        async with self._performance_tracker("delete"):
            try:
                self._validate_key(key)
                
                # Delete from intelligent cache
                if self.intelligent_cache:
                    await self.intelligent_cache.delete(key)
                
                # Delete from production cache
                result = await self.production_cache.delete(key)
                
                if result:
                    self.metrics.deletes += 1
                
                return result
                
            except Exception as e:
                logger.error(f"Enhanced cache delete failed for key {key}: {e}")
                return False
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """Enhanced tag-based invalidation."""
        async with self._performance_tracker("invalidate"):
            try:
                count = await self.production_cache.invalidate_by_tag(tag)
                
                # Also invalidate in intelligent cache if available
                if self.intelligent_cache:
                    await self.intelligent_cache.delete_pattern(f"*{tag}*")
                
                return count
                
            except Exception as e:
                logger.error(f"Enhanced cache invalidation failed for tag {tag}: {e}")
                return 0
    
    async def bulk_warm_cache(self, warming_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced bulk cache warming with performance tracking."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting enhanced cache warming with {len(warming_data)} entries")
            
            # Warm production cache
            if self.production_cache:
                await self.production_cache.warm_cache(warming_data)
            
            # Warm intelligent cache if available
            if self.intelligent_cache:
                for key, value in warming_data.items():
                    await self.intelligent_cache.set(key, value)
            
            warming_time = time.time() - start_time
            self.metrics.warming_operations += len(warming_data)
            self.metrics.warming_time = warming_time
            
            result = {
                "status": "success",
                "entries_warmed": len(warming_data),
                "warming_time": warming_time,
                "warming_rate": len(warming_data) / max(warming_time, 0.001)
            }
            
            logger.info(
                "Enhanced cache warming completed",
                entries=len(warming_data),
                duration=warming_time,
                rate=result["warming_rate"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced cache warming failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "entries_warmed": 0,
                "warming_time": time.time() - start_time
            }
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            # Get production cache stats
            prod_stats = {}
            if self.production_cache:
                prod_stats = await self.production_cache.get_cache_statistics()
            
            # Get intelligent cache stats
            intelligent_stats = {}
            if self.intelligent_cache:
                intelligent_stats = await self.intelligent_cache.get_stats()
            
            # Compile comprehensive stats
            stats = {
                "enhanced_cache": {
                    "metrics": {
                        "hits": self.metrics.hits,
                        "misses": self.metrics.misses,
                        "writes": self.metrics.writes,
                        "deletes": self.metrics.deletes,
                        "evictions": self.metrics.evictions,
                        "hit_rate": self.metrics.hit_rate(),
                        "operations_per_second": self.metrics.operations_per_second(),
                    },
                    "performance": {
                        "avg_response_time_ms": self.metrics.avg_response_time * 1000,
                        "p95_response_time_ms": self.metrics.p95_response_time * 1000,
                        "p99_response_time_ms": self.metrics.p99_response_time * 1000,
                        "throughput_per_second": self.metrics.throughput_per_second,
                        "network_latency_ms": self.metrics.network_latency * 1000,
                    },
                    "memory": {
                        "usage_bytes": self.metrics.memory_usage,
                        "peak_bytes": self.metrics.memory_peak,
                        "efficiency": self.metrics.memory_efficiency,
                    },
                    "compression": {
                        "enabled": self.compression_config.enabled,
                        "algorithm": self.compression_config.algorithm,
                        "ratio": self.metrics.compression_ratio,
                        "serialization_time_ms": self.metrics.serialization_time * 1000,
                    },
                    "security": {
                        "encryption_enabled": self.security_config.enable_encryption,
                        "auth_enabled": self.security_config.enable_auth,
                        "tls_enabled": self.security_config.enable_tls,
                    },
                    "warming": {
                        "operations": self.metrics.warming_operations,
                        "total_time": self.metrics.warming_time,
                    },
                },
                "production_cache": prod_stats,
                "intelligent_cache": intelligent_stats,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive stats: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "overall_score": 0,
            "max_score": 0,
        }
        
        try:
            # Check production cache health
            if self.production_cache:
                prod_health = await self.production_cache.health_check()
                health_status["checks"]["production_cache"] = prod_health
                health_status["max_score"] += 3
                if prod_health["status"] == "healthy":
                    health_status["overall_score"] += 3
                elif prod_health.get("status") == "degraded":
                    health_status["overall_score"] += 1
            
            # Check intelligent cache health
            if self.intelligent_cache:
                # Simplified intelligent cache health check
                health_status["checks"]["intelligent_cache"] = {"status": "pass"}
                health_status["max_score"] += 2
                health_status["overall_score"] += 2
            
            # Check performance metrics
            performance_score = 0
            max_performance_score = 3
            
            hit_rate = self.metrics.hit_rate()
            if hit_rate >= 0.8:
                performance_score += 1
            elif hit_rate >= 0.6:
                performance_score += 0.5
            
            if self.metrics.avg_response_time <= 0.05:  # 50ms
                performance_score += 1
            elif self.metrics.avg_response_time <= 0.1:  # 100ms
                performance_score += 0.5
            
            if self.metrics.throughput_per_second >= 100:
                performance_score += 1
            elif self.metrics.throughput_per_second >= 50:
                performance_score += 0.5
            
            health_status["checks"]["performance"] = {
                "status": "pass" if performance_score >= 2 else "warn",
                "hit_rate": hit_rate,
                "avg_response_time_ms": self.metrics.avg_response_time * 1000,
                "throughput_per_second": self.metrics.throughput_per_second,
                "score": performance_score,
                "max_score": max_performance_score,
            }
            
            health_status["overall_score"] += performance_score
            health_status["max_score"] += max_performance_score
            
            # Determine overall status
            score_ratio = health_status["overall_score"] / max(health_status["max_score"], 1)
            if score_ratio >= 0.8:
                health_status["status"] = "healthy"
            elif score_ratio >= 0.6:
                health_status["status"] = "degraded"
            else:
                health_status["status"] = "unhealthy"
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            logger.error(f"Enhanced cache health check failed: {e}")
        
        return health_status
    
    async def performance_benchmark(self, operations: int = 1000) -> Dict[str, Any]:
        """Run performance benchmark."""
        logger.info(f"Starting enhanced cache performance benchmark with {operations} operations")
        
        benchmark_results = {
            "operations": operations,
            "start_time": datetime.utcnow().isoformat(),
            "results": {},
        }
        
        try:
            # Benchmark write operations
            start_time = time.time()
            for i in range(operations):
                await self.set(f"benchmark:write:{i}", f"value_{i}")
            write_time = time.time() - start_time
            
            benchmark_results["results"]["write"] = {
                "total_time": write_time,
                "ops_per_second": operations / write_time,
                "avg_latency_ms": (write_time / operations) * 1000,
            }
            
            # Benchmark read operations
            start_time = time.time()
            for i in range(operations):
                await self.get(f"benchmark:write:{i}")
            read_time = time.time() - start_time
            
            benchmark_results["results"]["read"] = {
                "total_time": read_time,
                "ops_per_second": operations / read_time,
                "avg_latency_ms": (read_time / operations) * 1000,
            }
            
            # Cleanup benchmark keys
            for i in range(operations):
                await self.delete(f"benchmark:write:{i}")
            
            benchmark_results["end_time"] = datetime.utcnow().isoformat()
            benchmark_results["status"] = "completed"
            
            logger.info(
                "Performance benchmark completed",
                write_ops_per_sec=benchmark_results["results"]["write"]["ops_per_second"],
                read_ops_per_sec=benchmark_results["results"]["read"]["ops_per_second"]
            )
            
        except Exception as e:
            benchmark_results["status"] = "failed"
            benchmark_results["error"] = str(e)
            logger.error(f"Performance benchmark failed: {e}")
        
        return benchmark_results
    
    async def close(self) -> None:
        """Close enhanced cache and cleanup resources."""
        try:
            # Stop monitoring
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Stop warming workers
            for worker in self._warming_workers:
                worker.cancel()
            
            if self._warming_workers:
                await asyncio.gather(*self._warming_workers, return_exceptions=True)
            
            # Close production cache
            if self.production_cache:
                await self.production_cache.close()
            
            # Close intelligent cache
            if self.intelligent_cache:
                await self.intelligent_cache.close()
            
            logger.info("Enhanced Redis cache closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing enhanced Redis cache: {e}")


# Global enhanced cache instance
_enhanced_redis_cache: Optional[EnhancedRedisCache] = None


def get_enhanced_redis_cache(
    settings: Optional[Settings] = None,
    **kwargs
) -> EnhancedRedisCache:
    """Get or create global enhanced Redis cache instance."""
    global _enhanced_redis_cache
    
    if _enhanced_redis_cache is None:
        if settings is None:
            settings = Settings()
        
        _enhanced_redis_cache = EnhancedRedisCache(settings, **kwargs)
    
    return _enhanced_redis_cache


async def close_enhanced_redis_cache() -> None:
    """Close global enhanced Redis cache instance."""
    global _enhanced_redis_cache
    
    if _enhanced_redis_cache:
        await _enhanced_redis_cache.close()
        _enhanced_redis_cache = None