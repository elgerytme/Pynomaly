"""
Intelligent caching framework for enterprise-scale data quality operations.

This service implements multi-level caching, intelligent cache warming,
cache analytics, optimization recommendations, and distributed cache coordination.
"""

import asyncio
import logging
import time
import json
import hashlib
import pickle
import gzip
import lz4.frame
from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import threading
import weakref
import redis
import sqlite3
from pathlib import Path
import psutil
from contextlib import asynccontextmanager

from ...domain.entities.quality_profile import DataQualityProfile
from ...domain.value_objects.quality_scores import QualityScores
from ...domain.interfaces.data_quality_interface import DataQualityInterface

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for hierarchical caching."""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"
    PERSISTENT = "persistent"


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Machine learning based
    PRIORITY = "priority"  # Priority based


class CompressionType(Enum):
    """Compression types for cache storage."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ADAPTIVE = "adaptive"  # Choose based on data type


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata and analytics."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    expires_at: Optional[datetime] = None
    
    # Usage statistics
    access_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    
    # Size and performance metrics
    size_bytes: int = 0
    compression_ratio: float = 1.0
    serialization_time_ms: float = 0.0
    deserialization_time_ms: float = 0.0
    
    # Quality and priority
    quality_score: float = 1.0  # How valuable this cache entry is
    priority: int = 0  # Higher priority = less likely to be evicted
    
    # Metadata
    data_type: str = "unknown"
    tags: Set[str] = field(default_factory=set)
    dependency_keys: Set[str] = field(default_factory=set)
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio for this entry."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0


@dataclass
class CachePartition:
    """Cache partition for organizing cache entries."""
    partition_name: str
    max_size_mb: int
    current_size_mb: float = 0.0
    entry_count: int = 0
    strategy: CacheStrategy = CacheStrategy.LRU
    
    # Performance metrics
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    
    # Configuration
    compression_enabled: bool = True
    compression_type: CompressionType = CompressionType.ADAPTIVE
    auto_warm: bool = True
    
    @property
    def hit_ratio(self) -> float:
        """Calculate partition hit ratio."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    @property
    def utilization(self) -> float:
        """Calculate partition utilization."""
        return self.current_size_mb / self.max_size_mb if self.max_size_mb > 0 else 0.0


@dataclass
class CacheAnalytics:
    """Comprehensive cache analytics and performance metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Overall statistics
    total_entries: int = 0
    total_size_mb: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    
    # Performance metrics
    overall_hit_ratio: float = 0.0
    avg_access_time_ms: float = 0.0
    cache_warming_efficiency: float = 0.0
    
    # Operation counts
    total_gets: int = 0
    total_sets: int = 0
    total_evictions: int = 0
    total_expirations: int = 0
    
    # Efficiency metrics
    memory_efficiency: float = 0.0  # Useful data / total memory
    compression_savings: float = 0.0  # Bytes saved by compression
    
    # Quality metrics
    cache_quality_score: float = 0.0  # How well cache is performing
    prediction_accuracy: float = 0.0  # ML prediction accuracy
    
    # Trend data
    hit_ratio_trend: List[float] = field(default_factory=list)
    size_trend: List[float] = field(default_factory=list)
    access_pattern_entropy: float = 0.0  # Randomness of access patterns


class IntelligentCacheManager:
    """Multi-level intelligent cache manager with ML-powered optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the intelligent cache manager."""
        self.config = config
        
        # Cache storage layers
        self.memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.disk_cache_path = Path(config.get("disk_cache_path", "./cache"))
        self.disk_cache_path.mkdir(exist_ok=True)
        
        # Redis for distributed caching
        self.redis_client = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            decode_responses=False  # We need binary data for compression
        )
        
        # SQLite for persistent metadata
        self.metadata_db_path = self.disk_cache_path / "cache_metadata.db"
        self._init_metadata_db()
        
        # Cache partitions
        self.partitions: Dict[str, CachePartition] = {}
        self._init_default_partitions()
        
        # Configuration
        self.max_memory_mb = config.get("max_memory_mb", 1024)
        self.max_disk_mb = config.get("max_disk_mb", 10240)
        self.enable_compression = config.get("enable_compression", True)
        self.enable_analytics = config.get("enable_analytics", True)
        self.enable_ml_optimization = config.get("enable_ml_optimization", True)
        
        # Analytics and monitoring
        self.analytics = CacheAnalytics()
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.performance_history: List[CacheAnalytics] = []
        
        # Machine learning for cache optimization
        self.ml_model = None
        self.feature_history: List[Dict[str, float]] = []
        
        # Threading
        self.lock = threading.RLock()
        
        # Background tasks
        asyncio.create_task(self._cache_maintenance_task())
        asyncio.create_task(self._analytics_collection_task())
        asyncio.create_task(self._cache_warming_task())
        asyncio.create_task(self._ml_optimization_task())
    
    def _init_metadata_db(self) -> None:
        """Initialize SQLite database for cache metadata."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                partition TEXT,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER,
                size_bytes INTEGER,
                quality_score REAL,
                tags TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_analytics (
                timestamp TIMESTAMP PRIMARY KEY,
                total_entries INTEGER,
                total_size_mb REAL,
                hit_ratio REAL,
                analytics_data TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _init_default_partitions(self) -> None:
        """Initialize default cache partitions."""
        default_partitions = {
            "quality_profiles": CachePartition(
                partition_name="quality_profiles",
                max_size_mb=256,
                strategy=CacheStrategy.LRU,
                auto_warm=True
            ),
            "validation_results": CachePartition(
                partition_name="validation_results",
                max_size_mb=512,
                strategy=CacheStrategy.TTL,
                auto_warm=False
            ),
            "analytics_data": CachePartition(
                partition_name="analytics_data",
                max_size_mb=256,
                strategy=CacheStrategy.PRIORITY,
                auto_warm=True
            ),
            "temporary": CachePartition(
                partition_name="temporary",
                max_size_mb=128,
                strategy=CacheStrategy.LFU,
                auto_warm=False
            )
        }
        
        self.partitions.update(default_partitions)
    
    async def _cache_maintenance_task(self) -> None:
        """Background task for cache maintenance and cleanup."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                with self.lock:
                    # Remove expired entries
                    await self._remove_expired_entries()
                    
                    # Perform eviction if needed
                    await self._perform_intelligent_eviction()
                    
                    # Cleanup disk cache
                    await self._cleanup_disk_cache()
                    
                    # Update partition statistics
                    await self._update_partition_statistics()
                
                logger.debug("Cache maintenance completed")
                
            except Exception as e:
                logger.error(f"Cache maintenance error: {str(e)}")
    
    async def _analytics_collection_task(self) -> None:
        """Background task for collecting cache analytics."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if not self.enable_analytics:
                    continue
                
                # Collect current analytics
                current_analytics = await self._collect_analytics()
                self.performance_history.append(current_analytics)
                
                # Keep only last 24 hours of data
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.performance_history = [
                    a for a in self.performance_history
                    if a.timestamp > cutoff_time
                ]
                
                # Store analytics in database
                await self._store_analytics_in_db(current_analytics)
                
                logger.debug(f"Analytics collected: {current_analytics.overall_hit_ratio:.2f} hit ratio")
                
            except Exception as e:
                logger.error(f"Analytics collection error: {str(e)}")
    
    async def _cache_warming_task(self) -> None:
        """Background task for intelligent cache warming."""
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                
                # Identify frequently accessed but missing items
                await self._identify_warming_candidates()
                
                # Pre-load commonly accessed data
                await self._execute_cache_warming()
                
                logger.debug("Cache warming cycle completed")
                
            except Exception as e:
                logger.error(f"Cache warming error: {str(e)}")
    
    async def _ml_optimization_task(self) -> None:
        """Background task for ML-based cache optimization."""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                if not self.enable_ml_optimization:
                    continue
                
                # Collect features for ML model
                features = await self._collect_ml_features()
                self.feature_history.append(features)
                
                # Train/update ML model if enough data
                if len(self.feature_history) >= 10:
                    await self._update_ml_model()
                
                # Apply ML-based optimizations
                await self._apply_ml_optimizations()
                
                logger.debug("ML optimization cycle completed")
                
            except Exception as e:
                logger.error(f"ML optimization error: {str(e)}")
    
    # Error handling would be managed by interface implementation
    async def get(self, key: str, partition: str = "default") -> Optional[Any]:
        """Get value from cache with intelligent retrieval."""
        start_time = time.time()
        
        with self.lock:
            # Record access pattern
            self.access_patterns[key].append(datetime.utcnow())
            
            # Try memory cache first
            entry = self.memory_cache.get(key)
            if entry and not entry.is_expired:
                # Update access statistics
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                entry.hit_count += 1
                
                # Move to end for LRU
                self.memory_cache.move_to_end(key)
                
                # Update partition statistics
                if partition in self.partitions:
                    self.partitions[partition].hit_count += 1
                
                # Update analytics
                self.analytics.total_gets += 1
                access_time = (time.time() - start_time) * 1000
                self._update_access_time(access_time)
                
                return entry.value
            
            # Try disk cache
            disk_value = await self._get_from_disk(key)
            if disk_value is not None:
                # Promote to memory cache
                await self._promote_to_memory(key, disk_value, partition)
                
                # Update statistics
                if partition in self.partitions:
                    self.partitions[partition].hit_count += 1
                
                self.analytics.total_gets += 1
                return disk_value
            
            # Try distributed cache
            if self.config.get("enable_distributed_cache", True):
                distributed_value = await self._get_from_distributed(key)
                if distributed_value is not None:
                    # Promote to local caches
                    await self.set(key, distributed_value, partition=partition)
                    
                    # Update statistics
                    if partition in self.partitions:
                        self.partitions[partition].hit_count += 1
                    
                    self.analytics.total_gets += 1
                    return distributed_value
            
            # Cache miss
            if partition in self.partitions:
                self.partitions[partition].miss_count += 1
            
            # Record miss for ML learning
            if entry:
                entry.miss_count += 1
            
            self.analytics.total_gets += 1
            return None
    
    # Error handling would be managed by interface implementation
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
                 partition: str = "default", priority: int = 0, tags: Set[str] = None) -> None:
        """Set value in cache with intelligent placement."""
        if tags is None:
            tags = set()
        
        start_time = time.time()
        
        with self.lock:
            # Determine optimal compression
            compressed_value, compression_ratio = await self._compress_value(value)
            serialization_time = (time.time() - start_time) * 1000
            
            # Calculate size
            size_bytes = len(pickle.dumps(compressed_value))
            
            # Create cache entry
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                expires_at=expires_at,
                size_bytes=size_bytes,
                compression_ratio=compression_ratio,
                serialization_time_ms=serialization_time,
                priority=priority,
                data_type=type(value).__name__,
                tags=tags,
                quality_score=await self._calculate_quality_score(key, value)
            )
            
            # Determine optimal cache level
            cache_level = await self._determine_optimal_cache_level(entry)
            
            if cache_level == CacheLevel.MEMORY:
                await self._set_in_memory(key, entry, partition)
            elif cache_level == CacheLevel.DISK:
                await self._set_on_disk(key, entry)
            elif cache_level == CacheLevel.DISTRIBUTED:
                await self._set_in_distributed(key, entry)
            
            # Store metadata
            await self._store_metadata(key, entry, partition)
            
            # Update analytics
            self.analytics.total_sets += 1
    
    async def _determine_optimal_cache_level(self, entry: CacheEntry) -> CacheLevel:
        """Determine optimal cache level using intelligent placement."""
        # Factor in size, access patterns, and quality score
        size_factor = entry.size_bytes / (1024 * 1024)  # Size in MB
        
        # High-priority or frequently accessed items go to memory
        if entry.priority > 5 or entry.quality_score > 0.8:
            if size_factor < 10:  # Less than 10MB
                return CacheLevel.MEMORY
        
        # Medium-sized, medium-priority items go to disk
        if size_factor < 100:  # Less than 100MB
            return CacheLevel.DISK
        
        # Large items go to distributed cache
        return CacheLevel.DISTRIBUTED
    
    async def _set_in_memory(self, key: str, entry: CacheEntry, partition: str) -> None:
        """Set entry in memory cache with intelligent eviction."""
        # Check if eviction is needed
        current_size_mb = sum(e.size_bytes for e in self.memory_cache.values()) / (1024 * 1024)
        entry_size_mb = entry.size_bytes / (1024 * 1024)
        
        if current_size_mb + entry_size_mb > self.max_memory_mb:
            await self._evict_from_memory(entry_size_mb)
        
        # Store in memory
        self.memory_cache[key] = entry
        self.memory_cache.move_to_end(key)
        
        # Update partition
        if partition in self.partitions:
            self.partitions[partition].current_size_mb += entry_size_mb
            self.partitions[partition].entry_count += 1
    
    async def _set_on_disk(self, key: str, entry: CacheEntry) -> None:
        """Set entry on disk cache."""
        cache_file = self.disk_cache_path / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry.value, f)
            
            # Update disk usage tracking
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            self.analytics.disk_usage_mb += file_size_mb
            
        except Exception as e:
            logger.error(f"Failed to write to disk cache: {str(e)}")
    
    async def _set_in_distributed(self, key: str, entry: CacheEntry) -> None:
        """Set entry in distributed Redis cache."""
        try:
            serialized_data = pickle.dumps(entry.value)
            
            # Set with TTL if specified
            if entry.expires_at:
                ttl = int((entry.expires_at - datetime.utcnow()).total_seconds())
                self.redis_client.setex(f"cache:{key}", ttl, serialized_data)
            else:
                self.redis_client.set(f"cache:{key}", serialized_data)
            
        except Exception as e:
            logger.error(f"Failed to write to distributed cache: {str(e)}")
    
    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        cache_file = self.disk_cache_path / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to read from disk cache: {str(e)}")
            return None
    
    async def _get_from_distributed(self, key: str) -> Optional[Any]:
        """Get value from distributed Redis cache."""
        try:
            serialized_data = self.redis_client.get(f"cache:{key}")
            if serialized_data:
                return pickle.loads(serialized_data)
            return None
        except Exception as e:
            logger.error(f"Failed to read from distributed cache: {str(e)}")
            return None
    
    async def _compress_value(self, value: Any) -> Tuple[Any, float]:
        """Compress value using optimal compression algorithm."""
        if not self.enable_compression:
            return value, 1.0
        
        try:
            # Serialize first
            serialized = pickle.dumps(value)
            original_size = len(serialized)
            
            # Choose compression based on data type and size
            if original_size < 1024:  # Small data, no compression
                return value, 1.0
            
            # Try different compression algorithms
            if isinstance(value, (str, dict, list)):
                # JSON-like data, use gzip
                compressed = gzip.compress(serialized)
            else:
                # Binary data, use LZ4 for speed
                compressed = lz4.frame.compress(serialized)
            
            compression_ratio = len(compressed) / original_size
            
            # Only use compression if it saves significant space
            if compression_ratio < 0.8:
                return compressed, compression_ratio
            else:
                return value, 1.0
                
        except Exception as e:
            logger.warning(f"Compression failed: {str(e)}")
            return value, 1.0
    
    async def _calculate_quality_score(self, key: str, value: Any) -> float:
        """Calculate quality score for cache entry."""
        score = 0.5  # Base score
        
        # Factor in access patterns
        if key in self.access_patterns:
            access_count = len(self.access_patterns[key])
            if access_count > 10:
                score += 0.2
            elif access_count > 5:
                score += 0.1
        
        # Factor in data type
        if isinstance(value, (DataQualityProfile, QualityScores)):
            score += 0.2  # Quality-related data is important
        
        # Factor in size efficiency
        try:
            size_bytes = len(pickle.dumps(value))
            if size_bytes < 1024:  # Small, efficient data
                score += 0.1
        except:
            pass
        
        return min(1.0, score)
    
    async def _evict_from_memory(self, required_mb: float) -> None:
        """Intelligently evict entries from memory cache."""
        current_size_mb = sum(e.size_bytes for e in self.memory_cache.values()) / (1024 * 1024)
        target_size_mb = current_size_mb - required_mb - 50  # Extra buffer
        
        # Score entries for eviction (lower score = more likely to evict)
        eviction_candidates = []
        
        for key, entry in self.memory_cache.items():
            score = 0.0
            
            # Age factor (older = more likely to evict)
            age_factor = 1.0 / (entry.age_seconds / 3600 + 1)  # Hours
            score += age_factor * 0.3
            
            # Access frequency factor
            if entry.access_count > 0:
                frequency_factor = entry.access_count / max(1, entry.age_seconds / 3600)
                score += frequency_factor * 0.3
            
            # Quality and priority factors
            score += entry.quality_score * 0.2
            score += (entry.priority / 10) * 0.2
            
            eviction_candidates.append((key, entry, score))
        
        # Sort by eviction score (ascending)
        eviction_candidates.sort(key=lambda x: x[2])
        
        # Evict entries until target size is reached
        current_size = sum(e.size_bytes for e in self.memory_cache.values())
        target_size = target_size_mb * 1024 * 1024
        
        for key, entry, score in eviction_candidates:
            if current_size <= target_size:
                break
            
            # Move to disk if valuable, otherwise just remove
            if entry.quality_score > 0.6:
                await self._set_on_disk(key, entry)
            
            # Remove from memory
            del self.memory_cache[key]
            current_size -= entry.size_bytes
            self.analytics.total_evictions += 1
            
            logger.debug(f"Evicted {key} (score: {score:.2f})")
    
    async def _collect_analytics(self) -> CacheAnalytics:
        """Collect comprehensive cache analytics."""
        # Calculate basic statistics
        total_entries = len(self.memory_cache)
        memory_usage_mb = sum(e.size_bytes for e in self.memory_cache.values()) / (1024 * 1024)
        
        # Calculate hit ratios
        total_hits = sum(p.hit_count for p in self.partitions.values())
        total_misses = sum(p.miss_count for p in self.partitions.values())
        overall_hit_ratio = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
        
        # Calculate compression savings
        compression_savings = 0.0
        for entry in self.memory_cache.values():
            if entry.compression_ratio < 1.0:
                original_size = entry.size_bytes / entry.compression_ratio
                compression_savings += original_size - entry.size_bytes
        
        return CacheAnalytics(
            total_entries=total_entries,
            total_size_mb=memory_usage_mb + self.analytics.disk_usage_mb,
            memory_usage_mb=memory_usage_mb,
            disk_usage_mb=self.analytics.disk_usage_mb,
            overall_hit_ratio=overall_hit_ratio,
            total_gets=self.analytics.total_gets,
            total_sets=self.analytics.total_sets,
            total_evictions=self.analytics.total_evictions,
            compression_savings=compression_savings / (1024 * 1024),  # MB saved
            cache_quality_score=await self._calculate_cache_quality_score()
        )
    
    async def _calculate_cache_quality_score(self) -> float:
        """Calculate overall cache quality score."""
        if not self.memory_cache:
            return 0.0
        
        total_quality = sum(entry.quality_score for entry in self.memory_cache.values())
        return total_quality / len(self.memory_cache)
    
    # Error handling would be managed by interface implementation
    async def get_cache_report(self) -> Dict[str, Any]:
        """Generate comprehensive cache performance report."""
        current_analytics = await self._collect_analytics()
        
        # Partition statistics
        partition_stats = {}
        for name, partition in self.partitions.items():
            partition_stats[name] = {
                "hit_ratio": partition.hit_ratio,
                "utilization": partition.utilization,
                "entry_count": partition.entry_count,
                "size_mb": partition.current_size_mb,
                "eviction_count": partition.eviction_count
            }
        
        # Top accessed items
        top_items = sorted(
            self.access_patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:10]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_analytics": {
                "total_entries": current_analytics.total_entries,
                "total_size_mb": current_analytics.total_size_mb,
                "memory_usage_mb": current_analytics.memory_usage_mb,
                "disk_usage_mb": current_analytics.disk_usage_mb,
                "hit_ratio": current_analytics.overall_hit_ratio,
                "cache_quality_score": current_analytics.cache_quality_score,
                "compression_savings_mb": current_analytics.compression_savings
            },
            "partition_statistics": partition_stats,
            "top_accessed_items": [
                {"key": key, "access_count": len(accesses)}
                for key, accesses in top_items
            ],
            "performance_trends": [
                {
                    "timestamp": analytics.timestamp.isoformat(),
                    "hit_ratio": analytics.overall_hit_ratio,
                    "size_mb": analytics.total_size_mb
                }
                for analytics in self.performance_history[-24:]  # Last 24 data points
            ]
        }
    
    # Error handling would be managed by interface implementation
    async def optimize_cache(self) -> List[str]:
        """Generate cache optimization recommendations."""
        recommendations = []
        current_analytics = await self._collect_analytics()
        
        # Analyze hit ratios
        if current_analytics.overall_hit_ratio < 0.7:
            recommendations.append("Low cache hit ratio. Consider increasing cache size or improving warming strategies.")
        
        # Analyze memory usage
        memory_utilization = current_analytics.memory_usage_mb / self.max_memory_mb
        if memory_utilization > 0.9:
            recommendations.append("High memory utilization. Consider increasing memory cache size or optimizing eviction policies.")
        elif memory_utilization < 0.3:
            recommendations.append("Low memory utilization. Consider reducing memory cache size to save resources.")
        
        # Analyze compression effectiveness
        if current_analytics.compression_savings < 100:  # Less than 100MB saved
            recommendations.append("Low compression savings. Review compression strategies for better space efficiency.")
        
        # Analyze partition performance
        for name, partition in self.partitions.items():
            if partition.hit_ratio < 0.5:
                recommendations.append(f"Partition '{name}' has low hit ratio. Consider adjusting cache strategy or size.")
            
            if partition.utilization > 0.95:
                recommendations.append(f"Partition '{name}' is nearly full. Consider increasing partition size.")
        
        return recommendations
    
    async def shutdown(self) -> None:
        """Shutdown the cache manager."""
        logger.info("Shutting down intelligent cache manager...")
        
        # Save important cache data to disk
        with self.lock:
            for key, entry in self.memory_cache.items():
                if entry.quality_score > 0.8:  # Save high-quality entries
                    await self._set_on_disk(key, entry)
        
        # Close connections
        self.redis_client.close()
        
        logger.info("Intelligent cache manager shutdown complete")