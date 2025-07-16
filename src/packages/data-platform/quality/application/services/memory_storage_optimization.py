"""
Memory and storage optimization service for enterprise-scale data quality operations.

This service implements intelligent memory management, storage optimization,
data compression, lifecycle management, and resource efficiency optimization.
"""

import asyncio
import logging
import time
import os
import gc
import mmap
import pickle
import gzip
import lz4.frame
import zstandard as zstd
from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from pathlib import Path
import psutil
import numpy as np
import pandas as pd
from contextlib import contextmanager

from ...domain.entities.quality_profile import DataQualityProfile
from ...domain.value_objects.quality_scores import QualityScores
from interfaces.data_quality_interface import DataQualityInterface

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Compression algorithms for storage optimization."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    ADAPTIVE = "adaptive"


class StorageTier(Enum):
    """Storage tiers for data lifecycle management."""
    HOT = "hot"        # High-speed SSD, frequently accessed
    WARM = "warm"      # Standard SSD, occasionally accessed
    COLD = "cold"      # HDD or cloud storage, rarely accessed
    ARCHIVE = "archive" # Compressed, long-term storage


class MemoryPoolType(Enum):
    """Types of memory pools for different data types."""
    SMALL_OBJECTS = "small_objects"    # < 1KB
    MEDIUM_OBJECTS = "medium_objects"  # 1KB - 1MB
    LARGE_OBJECTS = "large_objects"    # 1MB - 100MB
    HUGE_OBJECTS = "huge_objects"      # > 100MB


@dataclass
class MemoryBlock:
    """Memory block for efficient memory management."""
    block_id: str
    size_bytes: int
    pool_type: MemoryPoolType
    allocated: bool = False
    
    # Usage tracking
    allocated_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    # Content metadata
    data_type: str = "unknown"
    compression_ratio: float = 1.0
    
    @property
    def age_seconds(self) -> float:
        """Get age of memory block in seconds."""
        if not self.allocated_at:
            return 0.0
        return (datetime.utcnow() - self.allocated_at).total_seconds()
    
    @property
    def is_stale(self) -> bool:
        """Check if memory block is stale (not accessed recently)."""
        if not self.last_accessed:
            return False
        return (datetime.utcnow() - self.last_accessed).total_seconds() > 3600  # 1 hour


@dataclass
class StorageMetrics:
    """Storage performance and utilization metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Disk usage
    total_space_gb: float = 0.0
    used_space_gb: float = 0.0
    available_space_gb: float = 0.0
    utilization_percent: float = 0.0
    
    # Storage tier distribution
    hot_storage_gb: float = 0.0
    warm_storage_gb: float = 0.0
    cold_storage_gb: float = 0.0
    archive_storage_gb: float = 0.0
    
    # Compression efficiency
    compressed_data_gb: float = 0.0
    uncompressed_data_gb: float = 0.0
    compression_ratio: float = 1.0
    
    # I/O performance
    read_iops: float = 0.0
    write_iops: float = 0.0
    read_throughput_mbps: float = 0.0
    write_throughput_mbps: float = 0.0
    
    # Lifecycle management
    objects_migrated_to_cold: int = 0
    objects_archived: int = 0
    storage_cost_savings_percent: float = 0.0


@dataclass
class MemoryMetrics:
    """Memory performance and utilization metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # System memory
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    utilization_percent: float = 0.0
    
    # Process memory
    process_memory_gb: float = 0.0
    heap_size_gb: float = 0.0
    
    # Memory pools
    small_objects_pool_mb: float = 0.0
    medium_objects_pool_mb: float = 0.0
    large_objects_pool_mb: float = 0.0
    huge_objects_pool_mb: float = 0.0
    
    # Garbage collection
    gc_collections: int = 0
    gc_time_ms: float = 0.0
    memory_freed_mb: float = 0.0
    
    # Efficiency metrics
    memory_fragmentation_percent: float = 0.0
    allocation_efficiency: float = 1.0
    cache_hit_ratio: float = 0.0


class MemoryPool:
    """Efficient memory pool for different object sizes."""
    
    def __init__(self, pool_type: MemoryPoolType, initial_size_mb: int = 10):
        """Initialize memory pool."""
        self.pool_type = pool_type
        self.blocks: Dict[str, MemoryBlock] = {}
        self.free_blocks: deque = deque()
        self.allocated_blocks: Dict[str, MemoryBlock] = {}
        
        # Pool configuration
        self.max_size_mb = initial_size_mb * 10  # 10x initial size
        self.current_size_mb = 0.0
        self.block_size_bytes = self._get_default_block_size()
        
        # Performance metrics
        self.allocation_count = 0
        self.deallocation_count = 0
        self.fragmentation_ratio = 0.0
        
        # Initialize pool
        self._initialize_pool(initial_size_mb)
    
    def _get_default_block_size(self) -> int:
        """Get default block size based on pool type."""
        size_map = {
            MemoryPoolType.SMALL_OBJECTS: 1024,        # 1KB
            MemoryPoolType.MEDIUM_OBJECTS: 1024 * 1024, # 1MB
            MemoryPoolType.LARGE_OBJECTS: 10 * 1024 * 1024,  # 10MB
            MemoryPoolType.HUGE_OBJECTS: 100 * 1024 * 1024   # 100MB
        }
        return size_map.get(self.pool_type, 1024)
    
    def _initialize_pool(self, size_mb: int) -> None:
        """Initialize memory pool with initial blocks."""
        total_bytes = size_mb * 1024 * 1024
        block_count = total_bytes // self.block_size_bytes
        
        for i in range(block_count):
            block_id = f"{self.pool_type.value}_{i}"
            block = MemoryBlock(
                block_id=block_id,
                size_bytes=self.block_size_bytes,
                pool_type=self.pool_type
            )
            self.blocks[block_id] = block
            self.free_blocks.append(block)
        
        self.current_size_mb = size_mb
        logger.info(f"Initialized {self.pool_type.value} pool with {block_count} blocks")
    
    def allocate(self, size_bytes: int) -> Optional[MemoryBlock]:
        """Allocate memory block from pool."""
        if not self.free_blocks:
            if not self._expand_pool():
                return None
        
        # Find suitable block
        block = self.free_blocks.popleft()
        block.allocated = True
        block.allocated_at = datetime.utcnow()
        block.last_accessed = datetime.utcnow()
        
        self.allocated_blocks[block.block_id] = block
        self.allocation_count += 1
        
        return block
    
    def deallocate(self, block: MemoryBlock) -> None:
        """Return memory block to pool."""
        if block.block_id not in self.allocated_blocks:
            return
        
        block.allocated = False
        block.allocated_at = None
        block.access_count = 0
        
        del self.allocated_blocks[block.block_id]
        self.free_blocks.append(block)
        self.deallocation_count += 1
    
    def _expand_pool(self) -> bool:
        """Expand memory pool if possible."""
        if self.current_size_mb >= self.max_size_mb:
            return False
        
        # Add 50% more blocks
        expansion_mb = min(self.current_size_mb * 0.5, self.max_size_mb - self.current_size_mb)
        expansion_bytes = expansion_mb * 1024 * 1024
        new_block_count = int(expansion_bytes // self.block_size_bytes)
        
        current_block_count = len(self.blocks)
        for i in range(new_block_count):
            block_id = f"{self.pool_type.value}_{current_block_count + i}"
            block = MemoryBlock(
                block_id=block_id,
                size_bytes=self.block_size_bytes,
                pool_type=self.pool_type
            )
            self.blocks[block_id] = block
            self.free_blocks.append(block)
        
        self.current_size_mb += expansion_mb
        logger.info(f"Expanded {self.pool_type.value} pool by {expansion_mb:.2f}MB")
        return True
    
    def cleanup_stale_blocks(self) -> int:
        """Clean up stale allocated blocks."""
        stale_blocks = [
            block for block in self.allocated_blocks.values()
            if block.is_stale
        ]
        
        for block in stale_blocks:
            self.deallocate(block)
        
        return len(stale_blocks)


class CompressionEngine:
    """Intelligent compression engine for storage optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize compression engine."""
        self.config = config
        self.algorithms = {
            CompressionAlgorithm.GZIP: self._compress_gzip,
            CompressionAlgorithm.LZ4: self._compress_lz4,
            CompressionAlgorithm.ZSTD: self._compress_zstd
        }
        
        # Performance tracking
        self.compression_stats: Dict[CompressionAlgorithm, Dict[str, float]] = defaultdict(
            lambda: {"total_time": 0.0, "total_bytes": 0, "compressed_bytes": 0, "count": 0}
        )
    
    def compress(self, data: bytes, algorithm: CompressionAlgorithm = CompressionAlgorithm.ADAPTIVE) -> Tuple[bytes, float, CompressionAlgorithm]:
        """Compress data using specified or optimal algorithm."""
        if algorithm == CompressionAlgorithm.ADAPTIVE:
            algorithm = self._choose_optimal_algorithm(data)
        
        if algorithm == CompressionAlgorithm.NONE:
            return data, 1.0, algorithm
        
        start_time = time.time()
        compressed_data = self.algorithms[algorithm](data)
        compression_time = time.time() - start_time
        
        compression_ratio = len(compressed_data) / len(data)
        
        # Update statistics
        stats = self.compression_stats[algorithm]
        stats["total_time"] += compression_time
        stats["total_bytes"] += len(data)
        stats["compressed_bytes"] += len(compressed_data)
        stats["count"] += 1
        
        return compressed_data, compression_ratio, algorithm
    
    def decompress(self, data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Decompress data using specified algorithm."""
        if algorithm == CompressionAlgorithm.NONE:
            return data
        elif algorithm == CompressionAlgorithm.GZIP:
            return gzip.decompress(data)
        elif algorithm == CompressionAlgorithm.LZ4:
            return lz4.frame.decompress(data)
        elif algorithm == CompressionAlgorithm.ZSTD:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")
    
    def _choose_optimal_algorithm(self, data: bytes) -> CompressionAlgorithm:
        """Choose optimal compression algorithm based on data characteristics."""
        data_size = len(data)
        
        # For small data, compression overhead might not be worth it
        if data_size < 1024:  # 1KB
            return CompressionAlgorithm.NONE
        
        # Analyze data patterns
        if self._is_highly_compressible(data):
            if data_size > 10 * 1024 * 1024:  # 10MB
                return CompressionAlgorithm.ZSTD  # Best compression ratio
            else:
                return CompressionAlgorithm.GZIP  # Good balance
        else:
            return CompressionAlgorithm.LZ4  # Fast compression for binary data
    
    def _is_highly_compressible(self, data: bytes) -> bool:
        """Check if data is highly compressible (text, JSON, etc.)."""
        # Simple heuristic: check for text patterns
        try:
            text = data.decode('utf-8')
            # Text data usually has high redundancy
            return len(set(text)) / len(text) < 0.3  # Low unique character ratio
        except UnicodeDecodeError:
            # Binary data - check for patterns
            return len(set(data)) / len(data) < 0.5
    
    def _compress_gzip(self, data: bytes) -> bytes:
        """Compress using gzip."""
        return gzip.compress(data, compresslevel=6)
    
    def _compress_lz4(self, data: bytes) -> bytes:
        """Compress using LZ4."""
        return lz4.frame.compress(data)
    
    def _compress_zstd(self, data: bytes) -> bytes:
        """Compress using Zstandard."""
        cctx = zstd.ZstdCompressor(level=3)
        return cctx.compress(data)


class StorageLifecycleManager:
    """Manages data lifecycle across storage tiers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize storage lifecycle manager."""
        self.config = config
        self.storage_paths = {
            StorageTier.HOT: Path(config.get("hot_storage_path", "./storage/hot")),
            StorageTier.WARM: Path(config.get("warm_storage_path", "./storage/warm")),
            StorageTier.COLD: Path(config.get("cold_storage_path", "./storage/cold")),
            StorageTier.ARCHIVE: Path(config.get("archive_storage_path", "./storage/archive"))
        }
        
        # Create storage directories
        for path in self.storage_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Lifecycle policies
        self.hot_to_warm_days = config.get("hot_to_warm_days", 7)
        self.warm_to_cold_days = config.get("warm_to_cold_days", 30)
        self.cold_to_archive_days = config.get("cold_to_archive_days", 90)
        
        # File tracking
        self.file_metadata: Dict[str, Dict[str, Any]] = {}
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
    
    def store_data(self, data_id: str, data: bytes, tier: StorageTier = StorageTier.HOT,
                  metadata: Dict[str, Any] = None) -> str:
        """Store data in specified storage tier."""
        if metadata is None:
            metadata = {}
        
        file_path = self.storage_paths[tier] / f"{data_id}.dat"
        
        # Compress data if not in hot tier
        if tier != StorageTier.HOT:
            compression_engine = CompressionEngine(self.config)
            compressed_data, compression_ratio, algorithm = compression_engine.compress(data)
            data = compressed_data
            metadata["compression_algorithm"] = algorithm.value
            metadata["compression_ratio"] = compression_ratio
        
        # Write data
        with open(file_path, 'wb') as f:
            f.write(data)
        
        # Store metadata
        self.file_metadata[data_id] = {
            "tier": tier.value,
            "file_path": str(file_path),
            "size_bytes": len(data),
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "access_count": 0,
            **metadata
        }
        
        logger.debug(f"Stored data {data_id} in {tier.value} tier")
        return str(file_path)
    
    def retrieve_data(self, data_id: str) -> Optional[bytes]:
        """Retrieve data from storage."""
        if data_id not in self.file_metadata:
            return None
        
        metadata = self.file_metadata[data_id]
        file_path = Path(metadata["file_path"])
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        # Read data
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Decompress if needed
        if "compression_algorithm" in metadata:
            compression_engine = CompressionEngine(self.config)
            algorithm = CompressionAlgorithm(metadata["compression_algorithm"])
            data = compression_engine.decompress(data, algorithm)
        
        # Update access patterns
        metadata["last_accessed"] = datetime.utcnow()
        metadata["access_count"] += 1
        self.access_patterns[data_id].append(datetime.utcnow())
        
        return data
    
    def migrate_data(self, data_id: str, target_tier: StorageTier) -> bool:
        """Migrate data to different storage tier."""
        if data_id not in self.file_metadata:
            return False
        
        metadata = self.file_metadata[data_id]
        current_tier = StorageTier(metadata["tier"])
        
        if current_tier == target_tier:
            return True  # Already in target tier
        
        # Retrieve data
        data = self.retrieve_data(data_id)
        if data is None:
            return False
        
        # Remove from current tier
        current_path = Path(metadata["file_path"])
        if current_path.exists():
            current_path.unlink()
        
        # Store in target tier
        self.store_data(data_id, data, target_tier, metadata)
        
        logger.info(f"Migrated data {data_id} from {current_tier.value} to {target_tier.value}")
        return True
    
    def run_lifecycle_policies(self) -> Dict[str, int]:
        """Run lifecycle policies to migrate data between tiers."""
        migrations = {"hot_to_warm": 0, "warm_to_cold": 0, "cold_to_archive": 0}
        current_time = datetime.utcnow()
        
        for data_id, metadata in self.file_metadata.items():
            tier = StorageTier(metadata["tier"])
            created_at = metadata["created_at"]
            last_accessed = metadata["last_accessed"]
            age_days = (current_time - created_at).days
            idle_days = (current_time - last_accessed).days
            
            # Hot to Warm migration
            if tier == StorageTier.HOT and idle_days >= self.hot_to_warm_days:
                if self.migrate_data(data_id, StorageTier.WARM):
                    migrations["hot_to_warm"] += 1
            
            # Warm to Cold migration
            elif tier == StorageTier.WARM and idle_days >= self.warm_to_cold_days:
                if self.migrate_data(data_id, StorageTier.COLD):
                    migrations["warm_to_cold"] += 1
            
            # Cold to Archive migration
            elif tier == StorageTier.COLD and age_days >= self.cold_to_archive_days:
                if self.migrate_data(data_id, StorageTier.ARCHIVE):
                    migrations["cold_to_archive"] += 1
        
        return migrations


class MemoryStorageOptimizationService:
    """Comprehensive memory and storage optimization service."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the memory and storage optimization service."""
        self.config = config
        
        # Memory management
        self.memory_pools: Dict[MemoryPoolType, MemoryPool] = {}
        self._init_memory_pools()
        
        # Compression engine
        self.compression_engine = CompressionEngine(config)
        
        # Storage lifecycle manager
        self.lifecycle_manager = StorageLifecycleManager(config)
        
        # Metrics tracking
        self.memory_metrics = MemoryMetrics()
        self.storage_metrics = StorageMetrics()
        self.metrics_history: List[Tuple[MemoryMetrics, StorageMetrics]] = []
        
        # Configuration
        self.gc_threshold_mb = config.get("gc_threshold_mb", 1024)  # 1GB
        self.memory_warning_threshold = config.get("memory_warning_threshold", 0.85)  # 85%
        self.enable_memory_mapping = config.get("enable_memory_mapping", True)
        
        # Background tasks
        asyncio.create_task(self._memory_monitoring_task())
        asyncio.create_task(self._storage_monitoring_task())
        asyncio.create_task(self._garbage_collection_task())
        asyncio.create_task(self._lifecycle_management_task())
    
    def _init_memory_pools(self) -> None:
        """Initialize memory pools for different object sizes."""
        pool_configs = {
            MemoryPoolType.SMALL_OBJECTS: 50,   # 50MB
            MemoryPoolType.MEDIUM_OBJECTS: 200, # 200MB
            MemoryPoolType.LARGE_OBJECTS: 500,  # 500MB
            MemoryPoolType.HUGE_OBJECTS: 1000   # 1GB
        }
        
        for pool_type, initial_size_mb in pool_configs.items():
            self.memory_pools[pool_type] = MemoryPool(pool_type, initial_size_mb)
        
        logger.info("Initialized memory pools for optimized allocation")
    
    async def _memory_monitoring_task(self) -> None:
        """Background task for memory monitoring."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Collect memory metrics
                self.memory_metrics = await self._collect_memory_metrics()
                
                # Check for memory pressure
                if self.memory_metrics.utilization_percent > self.memory_warning_threshold * 100:
                    await self._handle_memory_pressure()
                
                # Cleanup stale memory blocks
                await self._cleanup_stale_memory()
                
                logger.debug(f"Memory monitoring: {self.memory_metrics.utilization_percent:.1f}% utilization")
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {str(e)}")
    
    async def _storage_monitoring_task(self) -> None:
        """Background task for storage monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Collect storage metrics
                self.storage_metrics = await self._collect_storage_metrics()
                
                # Record metrics history
                self.metrics_history.append((self.memory_metrics, self.storage_metrics))
                
                # Keep only last 24 hours
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics_history = [
                    (mem, stor) for mem, stor in self.metrics_history
                    if mem.timestamp > cutoff_time
                ]
                
                logger.debug(f"Storage monitoring: {self.storage_metrics.utilization_percent:.1f}% utilization")
                
            except Exception as e:
                logger.error(f"Storage monitoring error: {str(e)}")
    
    async def _garbage_collection_task(self) -> None:
        """Background task for intelligent garbage collection."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Check if garbage collection is needed
                if self.memory_metrics.used_memory_gb * 1024 > self.gc_threshold_mb:
                    await self._perform_intelligent_gc()
                
                logger.debug("Garbage collection cycle completed")
                
            except Exception as e:
                logger.error(f"Garbage collection error: {str(e)}")
    
    async def _lifecycle_management_task(self) -> None:
        """Background task for storage lifecycle management."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Run lifecycle policies
                migrations = self.lifecycle_manager.run_lifecycle_policies()
                
                if sum(migrations.values()) > 0:
                    logger.info(f"Lifecycle migrations: {migrations}")
                
            except Exception as e:
                logger.error(f"Lifecycle management error: {str(e)}")
    
    async def _collect_memory_metrics(self) -> MemoryMetrics:
        """Collect comprehensive memory metrics."""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Memory pool statistics
        pool_stats = {}
        for pool_type, pool in self.memory_pools.items():
            pool_stats[pool_type.value] = pool.current_size_mb
        
        return MemoryMetrics(
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            used_memory_gb=(memory.total - memory.available) / (1024**3),
            utilization_percent=memory.percent,
            process_memory_gb=process_memory.rss / (1024**3),
            small_objects_pool_mb=pool_stats.get("small_objects", 0),
            medium_objects_pool_mb=pool_stats.get("medium_objects", 0),
            large_objects_pool_mb=pool_stats.get("large_objects", 0),
            huge_objects_pool_mb=pool_stats.get("huge_objects", 0)
        )
    
    async def _collect_storage_metrics(self) -> StorageMetrics:
        """Collect comprehensive storage metrics."""
        # Disk usage
        disk_usage = psutil.disk_usage('/')
        
        # Storage tier distribution
        tier_sizes = {}
        for tier, path in self.lifecycle_manager.storage_paths.items():
            if path.exists():
                tier_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                tier_sizes[tier.value] = tier_size / (1024**3)  # GB
        
        # Compression statistics
        compression_stats = self.compression_engine.compression_stats
        total_compressed = sum(stats["compressed_bytes"] for stats in compression_stats.values())
        total_uncompressed = sum(stats["total_bytes"] for stats in compression_stats.values())
        compression_ratio = total_compressed / total_uncompressed if total_uncompressed > 0 else 1.0
        
        return StorageMetrics(
            total_space_gb=disk_usage.total / (1024**3),
            used_space_gb=disk_usage.used / (1024**3),
            available_space_gb=disk_usage.free / (1024**3),
            utilization_percent=(disk_usage.used / disk_usage.total) * 100,
            hot_storage_gb=tier_sizes.get("hot", 0),
            warm_storage_gb=tier_sizes.get("warm", 0),
            cold_storage_gb=tier_sizes.get("cold", 0),
            archive_storage_gb=tier_sizes.get("archive", 0),
            compressed_data_gb=total_compressed / (1024**3),
            uncompressed_data_gb=total_uncompressed / (1024**3),
            compression_ratio=compression_ratio
        )
    
    async def _handle_memory_pressure(self) -> None:
        """Handle memory pressure situations."""
        logger.warning(f"Memory pressure detected: {self.memory_metrics.utilization_percent:.1f}% utilization")
        
        # Force garbage collection
        await self._perform_intelligent_gc()
        
        # Clean up memory pools
        for pool in self.memory_pools.values():
            stale_count = pool.cleanup_stale_blocks()
            if stale_count > 0:
                logger.info(f"Cleaned up {stale_count} stale blocks from {pool.pool_type.value} pool")
        
        # Suggest optimizations
        suggestions = await self._generate_memory_optimization_suggestions()
        if suggestions:
            logger.info(f"Memory optimization suggestions: {suggestions}")
    
    async def _cleanup_stale_memory(self) -> None:
        """Clean up stale memory allocations."""
        total_cleaned = 0
        for pool in self.memory_pools.values():
            cleaned = pool.cleanup_stale_blocks()
            total_cleaned += cleaned
        
        if total_cleaned > 0:
            logger.debug(f"Cleaned up {total_cleaned} stale memory blocks")
    
    async def _perform_intelligent_gc(self) -> None:
        """Perform intelligent garbage collection."""
        start_time = time.time()
        
        # Get memory before GC
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory after GC
        memory_after = process.memory_info().rss
        memory_freed = (memory_before - memory_after) / (1024 * 1024)  # MB
        gc_time = (time.time() - start_time) * 1000  # ms
        
        # Update metrics
        self.memory_metrics.gc_collections += 1
        self.memory_metrics.gc_time_ms += gc_time
        self.memory_metrics.memory_freed_mb += memory_freed
        
        logger.info(f"Garbage collection freed {memory_freed:.2f}MB in {gc_time:.2f}ms, collected {collected} objects")
    
    # Error handling would be managed by interface implementation
    async def allocate_optimized_memory(self, size_bytes: int, data_type: str = "unknown") -> Optional[str]:
        """Allocate memory using optimized pools."""
        # Determine appropriate pool
        pool_type = self._determine_pool_type(size_bytes)
        pool = self.memory_pools[pool_type]
        
        # Allocate block
        block = pool.allocate(size_bytes)
        if not block:
            logger.warning(f"Failed to allocate {size_bytes} bytes from {pool_type.value} pool")
            return None
        
        block.data_type = data_type
        return block.block_id
    
    def _determine_pool_type(self, size_bytes: int) -> MemoryPoolType:
        """Determine appropriate memory pool type based on size."""
        if size_bytes < 1024:  # 1KB
            return MemoryPoolType.SMALL_OBJECTS
        elif size_bytes < 1024 * 1024:  # 1MB
            return MemoryPoolType.MEDIUM_OBJECTS
        elif size_bytes < 100 * 1024 * 1024:  # 100MB
            return MemoryPoolType.LARGE_OBJECTS
        else:
            return MemoryPoolType.HUGE_OBJECTS
    
    # Error handling would be managed by interface implementation
    async def store_optimized_data(self, data_id: str, data: bytes, 
                                  access_pattern: str = "random") -> str:
        """Store data using optimal storage strategy."""
        # Determine optimal storage tier based on access pattern
        tier = self._determine_optimal_tier(access_pattern, len(data))
        
        # Store data
        file_path = self.lifecycle_manager.store_data(
            data_id, data, tier, {"access_pattern": access_pattern}
        )
        
        logger.debug(f"Stored {len(data)} bytes in {tier.value} tier")
        return file_path
    
    def _determine_optimal_tier(self, access_pattern: str, size_bytes: int) -> StorageTier:
        """Determine optimal storage tier based on access pattern and size."""
        if access_pattern == "frequent" or size_bytes < 1024 * 1024:  # 1MB
            return StorageTier.HOT
        elif access_pattern == "occasional":
            return StorageTier.WARM
        elif access_pattern == "rare":
            return StorageTier.COLD
        else:
            return StorageTier.HOT  # Default to hot for unknown patterns
    
    # Error handling would be managed by interface implementation
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        # Memory pool statistics
        pool_stats = {}
        for pool_type, pool in self.memory_pools.items():
            pool_stats[pool_type.value] = {
                "current_size_mb": pool.current_size_mb,
                "allocated_blocks": len(pool.allocated_blocks),
                "free_blocks": len(pool.free_blocks),
                "allocation_count": pool.allocation_count,
                "deallocation_count": pool.deallocation_count,
                "fragmentation_ratio": pool.fragmentation_ratio
            }
        
        # Compression statistics
        compression_stats = {}
        for algorithm, stats in self.compression_engine.compression_stats.items():
            if stats["count"] > 0:
                compression_stats[algorithm.value] = {
                    "count": stats["count"],
                    "avg_compression_ratio": stats["compressed_bytes"] / stats["total_bytes"],
                    "avg_time_ms": (stats["total_time"] * 1000) / stats["count"],
                    "total_savings_mb": (stats["total_bytes"] - stats["compressed_bytes"]) / (1024 * 1024)
                }
        
        # Storage tier distribution
        storage_distribution = {
            "hot": self.storage_metrics.hot_storage_gb,
            "warm": self.storage_metrics.warm_storage_gb,
            "cold": self.storage_metrics.cold_storage_gb,
            "archive": self.storage_metrics.archive_storage_gb
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_optimization": {
                "current_utilization_percent": self.memory_metrics.utilization_percent,
                "process_memory_gb": self.memory_metrics.process_memory_gb,
                "pool_statistics": pool_stats,
                "gc_statistics": {
                    "collections": self.memory_metrics.gc_collections,
                    "total_time_ms": self.memory_metrics.gc_time_ms,
                    "memory_freed_mb": self.memory_metrics.memory_freed_mb
                }
            },
            "storage_optimization": {
                "disk_utilization_percent": self.storage_metrics.utilization_percent,
                "storage_distribution_gb": storage_distribution,
                "compression_statistics": compression_stats,
                "compression_ratio": self.storage_metrics.compression_ratio
            },
            "optimization_suggestions": await self._generate_optimization_suggestions()
        }
    
    async def _generate_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # Memory suggestions
        if self.memory_metrics.utilization_percent > 80:
            suggestions.append("High memory utilization. Consider increasing memory limits or optimizing data structures.")
        
        if self.memory_metrics.memory_fragmentation_percent > 30:
            suggestions.append("High memory fragmentation. Consider running garbage collection more frequently.")
        
        # Storage suggestions
        if self.storage_metrics.utilization_percent > 85:
            suggestions.append("High disk utilization. Consider migrating old data to archive storage.")
        
        if self.storage_metrics.compression_ratio > 0.8:
            suggestions.append("Low compression efficiency. Review compression algorithms for different data types.")
        
        # Pool suggestions
        for pool_type, pool in self.memory_pools.items():
            if len(pool.free_blocks) == 0:
                suggestions.append(f"Memory pool {pool_type.value} is exhausted. Consider increasing pool size.")
        
        return suggestions
    
    async def _generate_memory_optimization_suggestions(self) -> List[str]:
        """Generate memory-specific optimization suggestions."""
        suggestions = []
        
        if self.memory_metrics.utilization_percent > 90:
            suggestions.append("Critical memory usage. Force garbage collection and consider scaling resources.")
        elif self.memory_metrics.utilization_percent > 80:
            suggestions.append("High memory usage. Monitor for memory leaks and optimize data structures.")
        
        return suggestions
    
    async def shutdown(self) -> None:
        """Shutdown the optimization service."""
        logger.info("Shutting down memory and storage optimization service...")
        
        # Final garbage collection
        await self._perform_intelligent_gc()
        
        # Clear memory pools
        for pool in self.memory_pools.values():
            pool.blocks.clear()
            pool.allocated_blocks.clear()
            pool.free_blocks.clear()
        
        logger.info("Memory and storage optimization service shutdown complete")