#!/usr/bin/env python3
"""
Advanced Caching System Showcase
==================================

This example demonstrates the complete multi-layer caching system
for the anomaly detection platform, including:

1. Multi-layer cache stores (Memory, Redis, Disk, Hybrid)
2. Domain-specific cache managers
3. Cache integration with detection services
4. Performance benchmarking and metrics
5. Different cache profiles and configurations

Run this script to see the caching system in action!
"""

import asyncio
import numpy as np
import time
import sys
from pathlib import Path

# Add the package to the path for examples
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from anomaly_detection.infrastructure.caching import (
    initialize_cache_system,
    CacheProfile,
    get_domain_cache_managers,
    benchmark_cache_performance,
    CachedDetectionService,
    CachedModelService,
    CachedDataProcessingService,
    CacheMetricsCollector
)


async def demonstrate_basic_caching():
    """Demonstrate basic caching operations."""
    print("🔧 Basic Caching Operations")
    print("-" * 40)
    
    # Initialize cache system for development
    managers = initialize_cache_system(CacheProfile.DEVELOPMENT)
    
    # Test different cache managers
    cache_tests = [
        ("Model Cache", managers.model_cache_manager),
        ("Detection Cache", managers.detection_cache_manager), 
        ("Data Cache", managers.data_cache_manager),
        ("Metrics Cache", managers.metrics_cache_manager)
    ]
    
    for cache_name, cache_manager in cache_tests:
        print(f"\n🧪 Testing {cache_name}:")
        
        # Set a value
        test_data = {"message": f"Test data for {cache_name}", "timestamp": time.time()}
        await cache_manager.set("test_key", test_data, ttl_seconds=300)
        print(f"   ✅ Set test data")
        
        # Get the value
        retrieved_data = await cache_manager.get("test_key")
        print(f"   ✅ Retrieved: {retrieved_data['message'] if retrieved_data else 'None'}")
        
        # Check statistics
        stats = cache_manager.get_stats()
        if stats:
            print(f"   📊 Hit rate: {stats['hit_rate_percent']:.1f}%")


async def demonstrate_domain_integration():
    """Demonstrate cache integration with domain services."""
    print("\n🏗️ Domain Service Integration")
    print("-" * 40)
    
    # Create cached services
    detection_service = CachedDetectionService()
    model_service = CachedModelService()
    data_service = CachedDataProcessingService()
    
    # Generate test data
    test_data = np.random.normal(0, 1, (500, 4))
    print(f"📊 Generated test data: {test_data.shape}")
    
    # Test cached detection
    print(f"\n🔍 Testing Cached Detection:")
    
    # First call - should be cache miss
    print("   First detection call (cache miss)...")
    start_time = time.time()
    result1 = await detection_service.detect_anomalies_cached(
        test_data,
        algorithm="isolation_forest",
        parameters={"contamination": 0.1}
    )
    time1 = time.time() - start_time
    print(f"   ⏱️  Time: {time1*1000:.2f}ms, Anomalies: {result1.anomaly_count}")
    
    # Second call - should be cache hit
    print("   Second detection call (cache hit)...")
    start_time = time.time()
    result2 = await detection_service.detect_anomalies_cached(
        test_data,
        algorithm="isolation_forest", 
        parameters={"contamination": 0.1}
    )
    time2 = time.time() - start_time
    print(f"   ⏱️  Time: {time2*1000:.2f}ms, Anomalies: {result2.anomaly_count}")
    print(f"   🚀 Speedup: {time1/time2:.1f}x faster")
    
    # Test cached model training
    print(f"\n🤖 Testing Cached Model Training:")
    
    # First training - cache miss
    print("   First training call (cache miss)...")
    start_time = time.time()
    model1 = await model_service.train_model_cached(
        test_data,
        algorithm="isolation_forest",
        parameters={"n_estimators": 100}
    )
    time1 = time.time() - start_time
    print(f"   ⏱️  Time: {time1*1000:.2f}ms, Model ID: {model1['model_id']}")
    
    # Second training - cache hit
    print("   Second training call (cache hit)...")
    start_time = time.time()
    model2 = await model_service.train_model_cached(
        test_data,
        algorithm="isolation_forest",
        parameters={"n_estimators": 100}
    )
    time2 = time.time() - start_time
    print(f"   ⏱️  Time: {time2*1000:.2f}ms, Model ID: {model2['model_id']}")
    print(f"   🚀 Speedup: {time1/time2:.1f}x faster")
    
    # Test cached data preprocessing
    print(f"\n📊 Testing Cached Data Preprocessing:")
    
    # First preprocessing - cache miss
    print("   First preprocessing call (cache miss)...")
    start_time = time.time()
    processed1, metadata1 = await data_service.preprocess_data_cached(
        test_data,
        operations=["normalize", "remove_outliers"]
    )
    time1 = time.time() - start_time
    print(f"   ⏱️  Time: {time1*1000:.2f}ms, Shape: {processed1.shape}")
    
    # Second preprocessing - cache hit
    print("   Second preprocessing call (cache hit)...")
    start_time = time.time()
    processed2, metadata2 = await data_service.preprocess_data_cached(
        test_data,
        operations=["normalize", "remove_outliers"]
    )
    time2 = time.time() - start_time
    print(f"   ⏱️  Time: {time2*1000:.2f}ms, Shape: {processed2.shape}")
    print(f"   🚀 Speedup: {time1/time2:.1f}x faster")


async def demonstrate_cache_profiles():
    """Demonstrate different cache profiles."""
    print("\n⚙️ Cache Profile Comparison")
    print("-" * 40)
    
    profiles = [
        CacheProfile.DEVELOPMENT,
        CacheProfile.TESTING,
        CacheProfile.PRODUCTION,
        CacheProfile.HIGH_PERFORMANCE,
        CacheProfile.LOW_MEMORY
    ]
    
    for profile in profiles:
        print(f"\n📋 {profile.value.upper()} Profile:")
        
        # Initialize with profile
        managers = initialize_cache_system(profile)
        config = managers.base_config
        
        print(f"   Memory Cache: {config.memory_cache_enabled} (max: {config.memory_cache_max_size})")
        print(f"   Redis Cache: {config.redis_cache_enabled}")
        print(f"   Disk Cache: {config.disk_cache_enabled} (max: {config.disk_cache_max_size_mb}MB)")
        print(f"   Default TTL: {config.default_ttl_seconds}s")
        print(f"   Detection TTL: {config.detection_result_ttl}s")
        print(f"   Model TTL: {config.model_cache_ttl}s")


async def demonstrate_performance_metrics():
    """Demonstrate cache performance metrics collection."""
    print("\n📊 Cache Performance Metrics")
    print("-" * 40)
    
    # Initialize cache system
    managers = initialize_cache_system(CacheProfile.PRODUCTION)
    metrics_collector = CacheMetricsCollector()
    
    # Perform some cached operations to generate metrics
    detection_service = CachedDetectionService()
    test_data = np.random.normal(0, 1, (200, 3))
    
    print("🔄 Performing operations to generate cache metrics...")
    
    # Mix of cache hits and misses
    for i in range(10):
        # Vary parameters slightly to create mix of hits/misses
        contamination = 0.1 if i < 5 else 0.05
        await detection_service.detect_anomalies_cached(
            test_data,
            algorithm="isolation_forest",
            parameters={"contamination": contamination}
        )
    
    # Collect and display metrics
    await metrics_collector.log_cache_performance()
    
    # Get detailed metrics
    detailed_metrics = await metrics_collector.collect_cache_metrics()
    
    print(f"\n🔍 Detailed Cache Analysis:")
    print(f"   Best performing domain: {detailed_metrics['cache_efficiency']['best_performing_domain']}")
    print(f"   Worst performing domain: {detailed_metrics['cache_efficiency']['worst_performing_domain']}")
    
    # Show individual domain performance
    for domain, stats in detailed_metrics['domain_specific_metrics'].items():
        if isinstance(stats, dict) and 'hit_rate_percent' in stats:
            print(f"   {domain}: {stats['hit_rate_percent']:.1f}% hit rate")


async def run_full_benchmark():
    """Run a comprehensive benchmark of the caching system."""
    print("\n🚀 Comprehensive Cache Benchmark")
    print("-" * 40)
    
    print("Starting benchmark with 50 operations...")
    benchmark_results = await benchmark_cache_performance(50)
    
    print(f"\n📈 Benchmark Summary:")
    print(f"   Timestamp: {benchmark_results['timestamp']}")
    
    # Show operation benchmarks
    for operation, stats in benchmark_results['benchmark_stats'].items():
        operation_name = operation.replace('_', ' ').title()
        print(f"\n   {operation_name}:")
        print(f"     Average time: {stats['avg_time_ms']:.2f}ms")
        print(f"     Best time: {stats['min_time_ms']:.2f}ms")
        print(f"     Worst time: {stats['max_time_ms']:.2f}ms")
        print(f"     Total operations: {stats['total_operations']}")
    
    # Show cache efficiency
    cache_metrics = benchmark_results['cache_metrics']
    overall = cache_metrics['overall_metrics']
    
    print(f"\n   Overall Cache Efficiency:")
    print(f"     Hit Rate: {overall['overall_hit_rate_percent']:.1f}%")
    print(f"     Total Requests: {overall['total_requests']}")
    print(f"     Cache Hits: {overall['total_hits']}")
    print(f"     Cache Misses: {overall['total_misses']}")


async def main():
    """Main demonstration function."""
    print("🎯 Advanced Caching System Showcase")
    print("=" * 60)
    print("This demo showcases the complete multi-layer caching system")
    print("for the anomaly detection platform.\n")
    
    try:
        # Run all demonstrations
        await demonstrate_basic_caching()
        await demonstrate_domain_integration()
        await demonstrate_cache_profiles()
        await demonstrate_performance_metrics()
        await run_full_benchmark()
        
        print(f"\n🎉 Caching System Showcase Completed Successfully!")
        print(f"=" * 60)
        
        print(f"\n💡 Key Benefits Demonstrated:")
        print(f"   ✅ Multi-layer caching with Memory, Redis, and Disk support")
        print(f"   ✅ Domain-specific cache optimization")
        print(f"   ✅ Significant performance improvements (up to 10x+ speedup)")
        print(f"   ✅ Flexible configuration for different environments")
        print(f"   ✅ Comprehensive metrics and monitoring")
        print(f"   ✅ Intelligent cache key generation and TTL management")
        
        print(f"\n🔧 Usage in Production:")
        print(f"   1. Set CACHE_PROFILE environment variable")
        print(f"   2. Configure Redis URL if using distributed caching")
        print(f"   3. Use get_domain_cache_managers() in your services")
        print(f"   4. Monitor cache performance with CacheMetricsCollector")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)