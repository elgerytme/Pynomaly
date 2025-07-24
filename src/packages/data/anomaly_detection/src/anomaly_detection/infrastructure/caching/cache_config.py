"""Cache configuration and integration for anomaly detection services."""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum

from .advanced_cache_strategies import (
    MemoryCacheStore, 
    RedisCacheStore, 
    DiskCacheStore, 
    HybridCacheStore,
    AdvancedCacheManager,
    CacheStrategy,
    CacheLayer
)


class CacheProfile(str, Enum):
    """Predefined cache profiles for different use cases."""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    PRODUCTION = "production"
    HIGH_PERFORMANCE = "high_performance"
    LOW_MEMORY = "low_memory"


@dataclass
class CacheConfiguration:
    """Cache configuration settings."""
    
    # Memory cache settings
    memory_cache_enabled: bool = True
    memory_cache_max_size: int = 1000
    memory_cache_strategy: CacheStrategy = CacheStrategy.LRU
    
    # Redis cache settings
    redis_cache_enabled: bool = False
    redis_url: str = "redis://localhost:6379"
    redis_prefix: str = "anomaly_detection"
    
    # Disk cache settings
    disk_cache_enabled: bool = False
    disk_cache_dir: str = "/tmp/anomaly_detection_cache"
    disk_cache_max_size_mb: int = 1000
    
    # Global cache settings
    default_ttl_seconds: int = 3600
    enable_cache_stats: bool = True
    
    # Domain-specific cache TTLs
    model_cache_ttl: int = 7200  # 2 hours
    detection_result_ttl: int = 1800  # 30 minutes
    data_preprocessing_ttl: int = 3600  # 1 hour
    metrics_cache_ttl: int = 300  # 5 minutes


class CacheConfigurationFactory:
    """Factory for creating cache configurations based on profiles."""
    
    @staticmethod
    def create_config(profile: CacheProfile) -> CacheConfiguration:
        """Create cache configuration for the specified profile."""
        
        if profile == CacheProfile.DEVELOPMENT:
            return CacheConfiguration(
                memory_cache_enabled=True,
                memory_cache_max_size=500,
                redis_cache_enabled=False,
                disk_cache_enabled=False,
                default_ttl_seconds=1800,
                enable_cache_stats=True
            )
        
        elif profile == CacheProfile.TESTING:
            return CacheConfiguration(
                memory_cache_enabled=True,
                memory_cache_max_size=100,
                redis_cache_enabled=False,
                disk_cache_enabled=False,
                default_ttl_seconds=300,
                enable_cache_stats=True,
                model_cache_ttl=600,
                detection_result_ttl=300,
                data_preprocessing_ttl=600
            )
        
        elif profile == CacheProfile.PRODUCTION:
            return CacheConfiguration(
                memory_cache_enabled=True,
                memory_cache_max_size=2000,
                memory_cache_strategy=CacheStrategy.ADAPTIVE,
                redis_cache_enabled=True,
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
                disk_cache_enabled=True,
                disk_cache_max_size_mb=5000,
                default_ttl_seconds=3600,
                model_cache_ttl=14400,  # 4 hours
                detection_result_ttl=7200,  # 2 hours
                data_preprocessing_ttl=10800  # 3 hours
            )
        
        elif profile == CacheProfile.HIGH_PERFORMANCE:
            return CacheConfiguration(
                memory_cache_enabled=True,
                memory_cache_max_size=5000,
                memory_cache_strategy=CacheStrategy.LFU,
                redis_cache_enabled=True,
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
                disk_cache_enabled=True,
                disk_cache_max_size_mb=10000,
                default_ttl_seconds=7200,
                model_cache_ttl=28800,  # 8 hours
                detection_result_ttl=14400,  # 4 hours
                data_preprocessing_ttl=21600  # 6 hours
            )
        
        elif profile == CacheProfile.LOW_MEMORY:
            return CacheConfiguration(
                memory_cache_enabled=True,
                memory_cache_max_size=200,
                memory_cache_strategy=CacheStrategy.LRU,
                redis_cache_enabled=False,
                disk_cache_enabled=True,
                disk_cache_max_size_mb=1000,
                default_ttl_seconds=1800,
                model_cache_ttl=3600,
                detection_result_ttl=900,
                data_preprocessing_ttl=1800
            )
        
        else:
            # Default to development profile
            return CacheConfigurationFactory.create_config(CacheProfile.DEVELOPMENT)
    
    @staticmethod
    def create_from_environment() -> CacheConfiguration:
        """Create cache configuration from environment variables."""
        profile_name = os.getenv("CACHE_PROFILE", "development").lower()
        
        try:
            profile = CacheProfile(profile_name)
        except ValueError:
            profile = CacheProfile.DEVELOPMENT
        
        config = CacheConfigurationFactory.create_config(profile)
        
        # Override with environment variables if present
        if os.getenv("MEMORY_CACHE_MAX_SIZE"):
            config.memory_cache_max_size = int(os.getenv("MEMORY_CACHE_MAX_SIZE"))
        
        if os.getenv("REDIS_CACHE_ENABLED"):
            config.redis_cache_enabled = os.getenv("REDIS_CACHE_ENABLED").lower() == "true"
        
        if os.getenv("DISK_CACHE_ENABLED"):
            config.disk_cache_enabled = os.getenv("DISK_CACHE_ENABLED").lower() == "true"
        
        if os.getenv("DISK_CACHE_MAX_SIZE_MB"):
            config.disk_cache_max_size_mb = int(os.getenv("DISK_CACHE_MAX_SIZE_MB"))
        
        if os.getenv("DEFAULT_TTL_SECONDS"):
            config.default_ttl_seconds = int(os.getenv("DEFAULT_TTL_SECONDS"))
        
        return config


class CacheManagerFactory:
    """Factory for creating configured cache managers."""
    
    @staticmethod
    def create_cache_manager(config: CacheConfiguration) -> AdvancedCacheManager:
        """Create cache manager from configuration."""
        
        # Create cache stores based on configuration
        memory_cache = None
        redis_cache = None
        disk_cache = None
        
        if config.memory_cache_enabled:
            memory_cache = MemoryCacheStore(
                max_size=config.memory_cache_max_size,
                strategy=config.memory_cache_strategy
            )
        
        if config.redis_cache_enabled:
            redis_cache = RedisCacheStore(
                redis_url=config.redis_url,
                prefix=config.redis_prefix
            )
        
        if config.disk_cache_enabled:
            disk_cache = DiskCacheStore(
                cache_dir=config.disk_cache_dir,
                max_size_mb=config.disk_cache_max_size_mb
            )
        
        # Create appropriate cache store
        if sum([config.memory_cache_enabled, config.redis_cache_enabled, config.disk_cache_enabled]) > 1:
            # Multi-layer hybrid cache
            cache_store = HybridCacheStore(
                memory_cache=memory_cache,
                redis_cache=redis_cache,
                disk_cache=disk_cache
            )
        elif config.memory_cache_enabled:
            cache_store = memory_cache
        elif config.redis_cache_enabled:
            cache_store = redis_cache
        elif config.disk_cache_enabled:
            cache_store = disk_cache
        else:
            # Fallback to memory cache
            cache_store = MemoryCacheStore(max_size=100)
        
        return AdvancedCacheManager(
            cache_store=cache_store,
            enable_stats=config.enable_cache_stats
        )


class DomainCacheManagers:
    """Domain-specific cache managers with appropriate configurations."""
    
    def __init__(self, base_config: CacheConfiguration):
        self.base_config = base_config
        
        # Create specialized cache managers for different domains
        self.model_cache_manager = self._create_model_cache_manager()
        self.detection_cache_manager = self._create_detection_cache_manager()
        self.data_cache_manager = self._create_data_cache_manager()
        self.metrics_cache_manager = self._create_metrics_cache_manager()
    
    def _create_model_cache_manager(self) -> AdvancedCacheManager:
        """Create cache manager optimized for ML models."""
        # Models benefit from disk caching due to size
        config = CacheConfiguration(
            memory_cache_enabled=True,
            memory_cache_max_size=50,  # Fewer models in memory
            memory_cache_strategy=CacheStrategy.LFU,  # Keep frequently used models
            redis_cache_enabled=self.base_config.redis_cache_enabled,
            redis_url=self.base_config.redis_url,
            redis_prefix=f"{self.base_config.redis_prefix}:models",
            disk_cache_enabled=True,  # Always enable disk for models
            disk_cache_dir=f"{self.base_config.disk_cache_dir}/models",
            disk_cache_max_size_mb=self.base_config.disk_cache_max_size_mb // 2,
            default_ttl_seconds=self.base_config.model_cache_ttl,
            enable_cache_stats=self.base_config.enable_cache_stats
        )
        
        return CacheManagerFactory.create_cache_manager(config)
    
    def _create_detection_cache_manager(self) -> AdvancedCacheManager:
        """Create cache manager optimized for detection results."""
        config = CacheConfiguration(
            memory_cache_enabled=True,
            memory_cache_max_size=self.base_config.memory_cache_max_size,
            memory_cache_strategy=CacheStrategy.LRU,  # Recent results more likely to be reused
            redis_cache_enabled=self.base_config.redis_cache_enabled,
            redis_url=self.base_config.redis_url,
            redis_prefix=f"{self.base_config.redis_prefix}:detection",
            disk_cache_enabled=self.base_config.disk_cache_enabled,
            disk_cache_dir=f"{self.base_config.disk_cache_dir}/detection",
            disk_cache_max_size_mb=self.base_config.disk_cache_max_size_mb // 4,
            default_ttl_seconds=self.base_config.detection_result_ttl,
            enable_cache_stats=self.base_config.enable_cache_stats
        )
        
        return CacheManagerFactory.create_cache_manager(config)
    
    def _create_data_cache_manager(self) -> AdvancedCacheManager:
        """Create cache manager optimized for data preprocessing."""
        config = CacheConfiguration(
            memory_cache_enabled=True,
            memory_cache_max_size=self.base_config.memory_cache_max_size // 2,
            memory_cache_strategy=CacheStrategy.LRU,
            redis_cache_enabled=self.base_config.redis_cache_enabled,
            redis_url=self.base_config.redis_url,
            redis_prefix=f"{self.base_config.redis_prefix}:data",
            disk_cache_enabled=self.base_config.disk_cache_enabled,
            disk_cache_dir=f"{self.base_config.disk_cache_dir}/data",
            disk_cache_max_size_mb=self.base_config.disk_cache_max_size_mb // 4,
            default_ttl_seconds=self.base_config.data_preprocessing_ttl,
            enable_cache_stats=self.base_config.enable_cache_stats
        )
        
        return CacheManagerFactory.create_cache_manager(config)
    
    def _create_metrics_cache_manager(self) -> AdvancedCacheManager:
        """Create cache manager optimized for metrics and monitoring data."""
        config = CacheConfiguration(
            memory_cache_enabled=True,
            memory_cache_max_size=500,  # Metrics are small but numerous
            memory_cache_strategy=CacheStrategy.TTL,  # Time-based eviction for metrics
            redis_cache_enabled=self.base_config.redis_cache_enabled,
            redis_url=self.base_config.redis_url,
            redis_prefix=f"{self.base_config.redis_prefix}:metrics",
            disk_cache_enabled=False,  # Metrics don't need disk persistence
            default_ttl_seconds=self.base_config.metrics_cache_ttl,
            enable_cache_stats=self.base_config.enable_cache_stats
        )
        
        return CacheManagerFactory.create_cache_manager(config)
    
    async def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all domain cache managers."""
        stats = {}
        
        managers = {
            'model': self.model_cache_manager,
            'detection': self.detection_cache_manager,
            'data': self.data_cache_manager,
            'metrics': self.metrics_cache_manager
        }
        
        for domain, manager in managers.items():
            domain_stats = manager.get_stats()
            if domain_stats:
                stats[f"{domain}_cache"] = domain_stats
        
        return stats
    
    async def clear_all_caches(self) -> Dict[str, bool]:
        """Clear all domain caches."""
        results = {}
        
        managers = {
            'model': self.model_cache_manager,
            'detection': self.detection_cache_manager,
            'data': self.data_cache_manager,
            'metrics': self.metrics_cache_manager
        }
        
        for domain, manager in managers.items():
            try:
                results[domain] = await manager.clear()
            except Exception as e:
                results[domain] = False
                print(f"Failed to clear {domain} cache: {e}")
        
        return results


# Global cache configuration and managers
_global_cache_config: Optional[CacheConfiguration] = None
_global_domain_managers: Optional[DomainCacheManagers] = None


def get_cache_config() -> CacheConfiguration:
    """Get global cache configuration."""
    global _global_cache_config
    if _global_cache_config is None:
        _global_cache_config = CacheConfigurationFactory.create_from_environment()
    return _global_cache_config


def get_domain_cache_managers() -> DomainCacheManagers:
    """Get global domain cache managers."""
    global _global_domain_managers
    if _global_domain_managers is None:
        config = get_cache_config()
        _global_domain_managers = DomainCacheManagers(config)
    return _global_domain_managers


def initialize_cache_system(profile: Optional[CacheProfile] = None) -> DomainCacheManagers:
    """Initialize the cache system with specified profile."""
    global _global_cache_config, _global_domain_managers
    
    if profile:
        _global_cache_config = CacheConfigurationFactory.create_config(profile)
    else:
        _global_cache_config = CacheConfigurationFactory.create_from_environment()
    
    _global_domain_managers = DomainCacheManagers(_global_cache_config)
    
    return _global_domain_managers


# Utility functions for common cache operations
async def cache_model_prediction(model_id: str, data_hash: str, prediction_result: Any) -> bool:
    """Cache model prediction result."""
    managers = get_domain_cache_managers()
    cache_key = f"prediction:{model_id}:{data_hash}"
    return await managers.detection_cache_manager.set(cache_key, prediction_result)


async def get_cached_model_prediction(model_id: str, data_hash: str) -> Optional[Any]:
    """Get cached model prediction result."""
    managers = get_domain_cache_managers()
    cache_key = f"prediction:{model_id}:{data_hash}"
    return await managers.detection_cache_manager.get(cache_key)


async def cache_preprocessed_data(data_hash: str, preprocessed_data: Any) -> bool:
    """Cache preprocessed data."""
    managers = get_domain_cache_managers()
    cache_key = f"preprocessed:{data_hash}"
    return await managers.data_cache_manager.set(cache_key, preprocessed_data)


async def get_cached_preprocessed_data(data_hash: str) -> Optional[Any]:
    """Get cached preprocessed data."""
    managers = get_domain_cache_managers()
    cache_key = f"preprocessed:{data_hash}"
    return await managers.data_cache_manager.get(cache_key)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def demo():
        print("🚀 Cache Configuration Demo")
        print("=" * 40)
        
        # Test different profiles
        profiles = [CacheProfile.DEVELOPMENT, CacheProfile.PRODUCTION, CacheProfile.HIGH_PERFORMANCE]
        
        for profile in profiles:
            print(f"\n📋 {profile.value.upper()} Profile:")
            config = CacheConfigurationFactory.create_config(profile)
            
            print(f"   Memory Cache: {config.memory_cache_enabled} (max: {config.memory_cache_max_size})")
            print(f"   Redis Cache: {config.redis_cache_enabled}")
            print(f"   Disk Cache: {config.disk_cache_enabled} (max: {config.disk_cache_max_size_mb}MB)")
            print(f"   Default TTL: {config.default_ttl_seconds}s")
        
        # Test domain-specific managers
        print(f"\n🏗️ Domain Cache Managers:")
        managers = initialize_cache_system(CacheProfile.PRODUCTION)
        
        # Test caching operations
        await cache_model_prediction("model_123", "data_abc", {"anomalies": [1, 5, 10]})
        cached_result = await get_cached_model_prediction("model_123", "data_abc")
        print(f"   Cached prediction: {cached_result}")
        
        # Test statistics
        stats = await managers.get_combined_stats()
        print(f"\n📊 Cache Statistics:")
        for domain, domain_stats in stats.items():
            print(f"   {domain}: {domain_stats}")
        
        print("\n✅ Cache configuration demo completed!")
    
    asyncio.run(demo())