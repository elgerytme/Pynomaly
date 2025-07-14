"""Tests for intelligent cache module."""

from datetime import datetime, timedelta

import pytest

from pynomaly.infrastructure.cache.intelligent_cache import (
    AccessPattern,
    CacheAnalytics,
    CacheEntry,
    CacheLayer,
    CacheMetrics,
    CacheOptimizer,
    CacheStrategy,
    IntelligentCacheManager,
    adaptive_cache,
    cache_strategy,
    predictive_cache,
)


class TestAccessPattern:
    """Test AccessPattern class."""

    def test_access_pattern_initialization(self):
        """Test AccessPattern initialization."""
        pattern = AccessPattern(
            key="test_key",
            frequency=10,
            last_access=datetime.utcnow(),
            access_times=[datetime.utcnow()],
            predictive_score=0.8
        )

        assert pattern.key == "test_key"
        assert pattern.frequency == 10
        assert pattern.last_access is not None
        assert len(pattern.access_times) == 1
        assert pattern.predictive_score == 0.8

    def test_access_pattern_record_access(self):
        """Test recording access in AccessPattern."""
        pattern = AccessPattern(key="test_key")
        initial_frequency = pattern.frequency

        pattern.record_access()

        assert pattern.frequency == initial_frequency + 1
        assert pattern.last_access is not None
        assert len(pattern.access_times) == 1

    def test_access_pattern_get_access_frequency(self):
        """Test getting access frequency over time window."""
        pattern = AccessPattern(key="test_key")
        now = datetime.utcnow()

        # Add access times
        pattern.access_times = [
            now - timedelta(minutes=5),
            now - timedelta(minutes=3),
            now - timedelta(minutes=1),
        ]

        # Get frequency in last 10 minutes
        freq = pattern.get_access_frequency(timedelta(minutes=10))
        assert freq == 3

        # Get frequency in last 2 minutes
        freq = pattern.get_access_frequency(timedelta(minutes=2))
        assert freq == 1

    def test_access_pattern_calculate_predictive_score(self):
        """Test calculating predictive score."""
        pattern = AccessPattern(key="test_key")
        now = datetime.utcnow()

        # Add regular access pattern
        pattern.access_times = [
            now - timedelta(minutes=10),
            now - timedelta(minutes=5),
            now,
        ]

        score = pattern.calculate_predictive_score()
        assert 0 <= score <= 1

    def test_access_pattern_is_hot(self):
        """Test hot key detection."""
        pattern = AccessPattern(key="test_key")

        # Not hot by default
        assert pattern.is_hot() is False

        # Make it hot
        pattern.frequency = 100
        pattern.last_access = datetime.utcnow()
        assert pattern.is_hot() is True

    def test_access_pattern_is_cold(self):
        """Test cold key detection."""
        pattern = AccessPattern(key="test_key")

        # Not cold by default
        assert pattern.is_cold() is False

        # Make it cold
        pattern.last_access = datetime.utcnow() - timedelta(hours=2)
        assert pattern.is_cold() is True

    def test_access_pattern_to_dict(self):
        """Test AccessPattern to_dict method."""
        pattern = AccessPattern(
            key="test_key",
            frequency=10,
            predictive_score=0.8
        )

        result = pattern.to_dict()

        assert result["key"] == "test_key"
        assert result["frequency"] == 10
        assert result["predictive_score"] == 0.8
        assert "last_access" in result
        assert "access_times" in result


class TestCacheEntry:
    """Test CacheEntry class."""

    def test_cache_entry_initialization(self):
        """Test CacheEntry initialization."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=3600,
            layer=CacheLayer.L1,
            access_count=5,
            last_access=datetime.utcnow()
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl == 3600
        assert entry.layer == CacheLayer.L1
        assert entry.access_count == 5
        assert entry.last_access is not None

    def test_cache_entry_is_expired(self):
        """Test cache entry expiration check."""
        # Not expired
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=3600,
            created_at=datetime.utcnow()
        )
        assert entry.is_expired() is False

        # Expired
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=3600,
            created_at=datetime.utcnow() - timedelta(seconds=3700)
        )
        assert entry.is_expired() is True

    def test_cache_entry_time_to_expiry(self):
        """Test time to expiry calculation."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=3600,
            created_at=datetime.utcnow()
        )

        tte = entry.time_to_expiry()
        assert tte > timedelta(seconds=3500)
        assert tte <= timedelta(seconds=3600)

    def test_cache_entry_record_access(self):
        """Test recording access in CacheEntry."""
        entry = CacheEntry(key="test_key", value="test_value")
        initial_count = entry.access_count

        entry.record_access()

        assert entry.access_count == initial_count + 1
        assert entry.last_access is not None

    def test_cache_entry_should_promote(self):
        """Test cache entry promotion logic."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            layer=CacheLayer.L2,
            access_count=10,
            last_access=datetime.utcnow()
        )

        # Should promote hot, frequently accessed entries
        assert entry.should_promote() is True

        # Should not promote cold entries
        entry.last_access = datetime.utcnow() - timedelta(hours=2)
        assert entry.should_promote() is False

    def test_cache_entry_should_demote(self):
        """Test cache entry demotion logic."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            layer=CacheLayer.L1,
            access_count=1,
            last_access=datetime.utcnow() - timedelta(hours=1)
        )

        # Should demote cold, infrequently accessed entries
        assert entry.should_demote() is True

        # Should not demote hot entries
        entry.last_access = datetime.utcnow()
        entry.access_count = 100
        assert entry.should_demote() is False

    def test_cache_entry_to_dict(self):
        """Test CacheEntry to_dict method."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=3600,
            layer=CacheLayer.L1,
            access_count=5
        )

        result = entry.to_dict()

        assert result["key"] == "test_key"
        assert result["value"] == "test_value"
        assert result["ttl"] == 3600
        assert result["layer"] == "L1"
        assert result["access_count"] == 5
        assert "created_at" in result
        assert "last_access" in result


class TestCacheMetrics:
    """Test CacheMetrics class."""

    def test_cache_metrics_initialization(self):
        """Test CacheMetrics initialization."""
        metrics = CacheMetrics()

        assert metrics.l1_hits == 0
        assert metrics.l1_misses == 0
        assert metrics.l2_hits == 0
        assert metrics.l2_misses == 0
        assert metrics.promotions == 0
        assert metrics.demotions == 0
        assert metrics.evictions == 0
        assert metrics.prefetches == 0

    def test_cache_metrics_record_hit(self):
        """Test recording cache hit."""
        metrics = CacheMetrics()

        metrics.record_hit(CacheLayer.L1)
        assert metrics.l1_hits == 1
        assert metrics.l1_misses == 0

        metrics.record_hit(CacheLayer.L2)
        assert metrics.l2_hits == 1
        assert metrics.l2_misses == 0

    def test_cache_metrics_record_miss(self):
        """Test recording cache miss."""
        metrics = CacheMetrics()

        metrics.record_miss(CacheLayer.L1)
        assert metrics.l1_hits == 0
        assert metrics.l1_misses == 1

        metrics.record_miss(CacheLayer.L2)
        assert metrics.l2_hits == 0
        assert metrics.l2_misses == 1

    def test_cache_metrics_record_promotion(self):
        """Test recording cache promotion."""
        metrics = CacheMetrics()

        metrics.record_promotion()
        assert metrics.promotions == 1

    def test_cache_metrics_record_demotion(self):
        """Test recording cache demotion."""
        metrics = CacheMetrics()

        metrics.record_demotion()
        assert metrics.demotions == 1

    def test_cache_metrics_record_eviction(self):
        """Test recording cache eviction."""
        metrics = CacheMetrics()

        metrics.record_eviction()
        assert metrics.evictions == 1

    def test_cache_metrics_record_prefetch(self):
        """Test recording cache prefetch."""
        metrics = CacheMetrics()

        metrics.record_prefetch()
        assert metrics.prefetches == 1

    def test_cache_metrics_get_l1_hit_rate(self):
        """Test L1 hit rate calculation."""
        metrics = CacheMetrics()

        # No requests
        assert metrics.get_l1_hit_rate() == 0.0

        # With hits and misses
        metrics.l1_hits = 8
        metrics.l1_misses = 2
        assert metrics.get_l1_hit_rate() == 0.8

    def test_cache_metrics_get_l2_hit_rate(self):
        """Test L2 hit rate calculation."""
        metrics = CacheMetrics()

        # No requests
        assert metrics.get_l2_hit_rate() == 0.0

        # With hits and misses
        metrics.l2_hits = 6
        metrics.l2_misses = 4
        assert metrics.get_l2_hit_rate() == 0.6

    def test_cache_metrics_get_overall_hit_rate(self):
        """Test overall hit rate calculation."""
        metrics = CacheMetrics()

        # No requests
        assert metrics.get_overall_hit_rate() == 0.0

        # With hits and misses
        metrics.l1_hits = 8
        metrics.l1_misses = 2
        metrics.l2_hits = 6
        metrics.l2_misses = 4
        assert metrics.get_overall_hit_rate() == 0.7

    def test_cache_metrics_reset(self):
        """Test resetting metrics."""
        metrics = CacheMetrics()

        # Set some values
        metrics.l1_hits = 10
        metrics.l1_misses = 5
        metrics.promotions = 3

        metrics.reset()

        assert metrics.l1_hits == 0
        assert metrics.l1_misses == 0
        assert metrics.promotions == 0

    def test_cache_metrics_to_dict(self):
        """Test CacheMetrics to_dict method."""
        metrics = CacheMetrics()
        metrics.l1_hits = 10
        metrics.l1_misses = 5
        metrics.l2_hits = 6
        metrics.l2_misses = 4

        result = metrics.to_dict()

        assert result["l1_hits"] == 10
        assert result["l1_misses"] == 5
        assert result["l2_hits"] == 6
        assert result["l2_misses"] == 4
        assert result["l1_hit_rate"] == 0.667
        assert result["l2_hit_rate"] == 0.6
        assert result["overall_hit_rate"] == 0.64


class TestCacheStrategy:
    """Test CacheStrategy class."""

    def test_cache_strategy_initialization(self):
        """Test CacheStrategy initialization."""
        strategy = CacheStrategy(
            name="test_strategy",
            l1_size=1000,
            l2_size=10000,
            l1_ttl=3600,
            l2_ttl=7200,
            promotion_threshold=5,
            demotion_threshold=2,
            eviction_policy="lru"
        )

        assert strategy.name == "test_strategy"
        assert strategy.l1_size == 1000
        assert strategy.l2_size == 10000
        assert strategy.l1_ttl == 3600
        assert strategy.l2_ttl == 7200
        assert strategy.promotion_threshold == 5
        assert strategy.demotion_threshold == 2
        assert strategy.eviction_policy == "lru"

    def test_cache_strategy_should_promote(self):
        """Test cache strategy promotion logic."""
        strategy = CacheStrategy(promotion_threshold=5)

        entry = CacheEntry(
            key="test_key",
            value="test_value",
            access_count=6,
            last_access=datetime.utcnow()
        )

        assert strategy.should_promote(entry) is True

        entry.access_count = 3
        assert strategy.should_promote(entry) is False

    def test_cache_strategy_should_demote(self):
        """Test cache strategy demotion logic."""
        strategy = CacheStrategy(demotion_threshold=2)

        entry = CacheEntry(
            key="test_key",
            value="test_value",
            access_count=1,
            last_access=datetime.utcnow() - timedelta(hours=1)
        )

        assert strategy.should_demote(entry) is True

        entry.access_count = 5
        assert strategy.should_demote(entry) is False

    def test_cache_strategy_should_evict(self):
        """Test cache strategy eviction logic."""
        strategy = CacheStrategy(eviction_policy="lru")

        # Old entry should be evicted
        old_entry = CacheEntry(
            key="old_key",
            value="old_value",
            last_access=datetime.utcnow() - timedelta(hours=2)
        )

        # New entry should not be evicted
        new_entry = CacheEntry(
            key="new_key",
            value="new_value",
            last_access=datetime.utcnow()
        )

        entries = [old_entry, new_entry]

        assert strategy.should_evict(old_entry, entries) is True
        assert strategy.should_evict(new_entry, entries) is False

    def test_cache_strategy_get_ttl(self):
        """Test getting TTL for cache layer."""
        strategy = CacheStrategy(l1_ttl=3600, l2_ttl=7200)

        assert strategy.get_ttl(CacheLayer.L1) == 3600
        assert strategy.get_ttl(CacheLayer.L2) == 7200

    def test_cache_strategy_get_size_limit(self):
        """Test getting size limit for cache layer."""
        strategy = CacheStrategy(l1_size=1000, l2_size=10000)

        assert strategy.get_size_limit(CacheLayer.L1) == 1000
        assert strategy.get_size_limit(CacheLayer.L2) == 10000

    def test_cache_strategy_to_dict(self):
        """Test CacheStrategy to_dict method."""
        strategy = CacheStrategy(
            name="test_strategy",
            l1_size=1000,
            l2_size=10000,
            promotion_threshold=5
        )

        result = strategy.to_dict()

        assert result["name"] == "test_strategy"
        assert result["l1_size"] == 1000
        assert result["l2_size"] == 10000
        assert result["promotion_threshold"] == 5

    def test_cache_strategy_from_dict(self):
        """Test CacheStrategy from_dict method."""
        data = {
            "name": "test_strategy",
            "l1_size": 1000,
            "l2_size": 10000,
            "promotion_threshold": 5
        }

        strategy = CacheStrategy.from_dict(data)

        assert strategy.name == "test_strategy"
        assert strategy.l1_size == 1000
        assert strategy.l2_size == 10000
        assert strategy.promotion_threshold == 5


class TestIntelligentCacheManager:
    """Test IntelligentCacheManager class."""

    def test_intelligent_cache_manager_initialization(self):
        """Test IntelligentCacheManager initialization."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        assert manager.strategy == strategy
        assert manager.l1_cache == {}
        assert manager.l2_cache == {}
        assert manager.access_patterns == {}
        assert manager.metrics is not None
        assert manager.is_running is False

    @pytest.mark.asyncio
    async def test_intelligent_cache_get_l1_hit(self):
        """Test cache get with L1 hit."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        # Add entry to L1 cache
        entry = CacheEntry(key="test_key", value="test_value", layer=CacheLayer.L1)
        manager.l1_cache["test_key"] = entry

        result = await manager.get("test_key")

        assert result == "test_value"
        assert manager.metrics.l1_hits == 1
        assert manager.metrics.l1_misses == 0

    @pytest.mark.asyncio
    async def test_intelligent_cache_get_l2_hit(self):
        """Test cache get with L2 hit."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        # Add entry to L2 cache
        entry = CacheEntry(key="test_key", value="test_value", layer=CacheLayer.L2)
        manager.l2_cache["test_key"] = entry

        result = await manager.get("test_key")

        assert result == "test_value"
        assert manager.metrics.l2_hits == 1
        assert manager.metrics.l2_misses == 0

    @pytest.mark.asyncio
    async def test_intelligent_cache_get_miss(self):
        """Test cache get with miss."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        result = await manager.get("nonexistent_key")

        assert result is None
        assert manager.metrics.l1_misses == 1
        assert manager.metrics.l2_misses == 1

    @pytest.mark.asyncio
    async def test_intelligent_cache_set_l1(self):
        """Test cache set to L1."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        await manager.set("test_key", "test_value", ttl=3600)

        assert "test_key" in manager.l1_cache
        assert manager.l1_cache["test_key"].value == "test_value"
        assert manager.l1_cache["test_key"].layer == CacheLayer.L1

    @pytest.mark.asyncio
    async def test_intelligent_cache_set_l2(self):
        """Test cache set to L2."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        await manager.set("test_key", "test_value", ttl=7200, layer=CacheLayer.L2)

        assert "test_key" in manager.l2_cache
        assert manager.l2_cache["test_key"].value == "test_value"
        assert manager.l2_cache["test_key"].layer == CacheLayer.L2

    @pytest.mark.asyncio
    async def test_intelligent_cache_delete(self):
        """Test cache delete."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        # Add entries to both caches
        await manager.set("test_key", "test_value")
        await manager.set("test_key", "test_value", layer=CacheLayer.L2)

        result = await manager.delete("test_key")

        assert result is True
        assert "test_key" not in manager.l1_cache
        assert "test_key" not in manager.l2_cache

    @pytest.mark.asyncio
    async def test_intelligent_cache_clear(self):
        """Test cache clear."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        # Add entries to both caches
        await manager.set("key1", "value1")
        await manager.set("key2", "value2", layer=CacheLayer.L2)

        await manager.clear()

        assert len(manager.l1_cache) == 0
        assert len(manager.l2_cache) == 0

    @pytest.mark.asyncio
    async def test_intelligent_cache_promote_entry(self):
        """Test entry promotion from L2 to L1."""
        strategy = CacheStrategy(name="test_strategy", promotion_threshold=5)
        manager = IntelligentCacheManager(strategy)

        # Add entry to L2 with high access count
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            layer=CacheLayer.L2,
            access_count=10
        )
        manager.l2_cache["test_key"] = entry

        await manager.promote_entry("test_key")

        assert "test_key" in manager.l1_cache
        assert "test_key" not in manager.l2_cache
        assert manager.l1_cache["test_key"].layer == CacheLayer.L1
        assert manager.metrics.promotions == 1

    @pytest.mark.asyncio
    async def test_intelligent_cache_demote_entry(self):
        """Test entry demotion from L1 to L2."""
        strategy = CacheStrategy(name="test_strategy", demotion_threshold=2)
        manager = IntelligentCacheManager(strategy)

        # Add entry to L1 with low access count
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            layer=CacheLayer.L1,
            access_count=1
        )
        manager.l1_cache["test_key"] = entry

        await manager.demote_entry("test_key")

        assert "test_key" in manager.l2_cache
        assert "test_key" not in manager.l1_cache
        assert manager.l2_cache["test_key"].layer == CacheLayer.L2
        assert manager.metrics.demotions == 1

    @pytest.mark.asyncio
    async def test_intelligent_cache_evict_lru(self):
        """Test LRU eviction."""
        strategy = CacheStrategy(name="test_strategy", l1_size=2, eviction_policy="lru")
        manager = IntelligentCacheManager(strategy)

        # Fill L1 cache
        await manager.set("key1", "value1")
        await manager.set("key2", "value2")

        # Access key1 to make it more recently used
        await manager.get("key1")

        # Add third entry (should evict key2)
        await manager.set("key3", "value3")

        assert "key1" in manager.l1_cache
        assert "key2" not in manager.l1_cache
        assert "key3" in manager.l1_cache
        assert manager.metrics.evictions == 1

    @pytest.mark.asyncio
    async def test_intelligent_cache_optimize(self):
        """Test cache optimization."""
        strategy = CacheStrategy(name="test_strategy", promotion_threshold=3)
        manager = IntelligentCacheManager(strategy)

        # Add entries with different access patterns
        await manager.set("hot_key", "hot_value", layer=CacheLayer.L2)
        await manager.set("cold_key", "cold_value", layer=CacheLayer.L1)

        # Make hot_key frequently accessed
        manager.l2_cache["hot_key"].access_count = 10
        manager.l2_cache["hot_key"].last_access = datetime.utcnow()

        # Make cold_key infrequently accessed
        manager.l1_cache["cold_key"].access_count = 1
        manager.l1_cache["cold_key"].last_access = datetime.utcnow() - timedelta(hours=1)

        await manager.optimize()

        # Hot key should be promoted to L1
        assert "hot_key" in manager.l1_cache

        # Cold key should be demoted to L2
        assert "cold_key" in manager.l2_cache

    @pytest.mark.asyncio
    async def test_intelligent_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        # Add expired entry
        expired_entry = CacheEntry(
            key="expired_key",
            value="expired_value",
            ttl=3600,
            created_at=datetime.utcnow() - timedelta(seconds=3700)
        )
        manager.l1_cache["expired_key"] = expired_entry

        # Add valid entry
        valid_entry = CacheEntry(
            key="valid_key",
            value="valid_value",
            ttl=3600,
            created_at=datetime.utcnow()
        )
        manager.l1_cache["valid_key"] = valid_entry

        await manager.cleanup_expired()

        assert "expired_key" not in manager.l1_cache
        assert "valid_key" in manager.l1_cache

    def test_intelligent_cache_get_metrics(self):
        """Test getting cache metrics."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        # Set some metrics
        manager.metrics.l1_hits = 10
        manager.metrics.l1_misses = 5
        manager.metrics.promotions = 3

        metrics = manager.get_metrics()

        assert metrics.l1_hits == 10
        assert metrics.l1_misses == 5
        assert metrics.promotions == 3

    def test_intelligent_cache_reset_metrics(self):
        """Test resetting cache metrics."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        # Set some metrics
        manager.metrics.l1_hits = 10
        manager.metrics.l1_misses = 5

        manager.reset_metrics()

        assert manager.metrics.l1_hits == 0
        assert manager.metrics.l1_misses == 0

    @pytest.mark.asyncio
    async def test_intelligent_cache_get_cache_info(self):
        """Test getting cache information."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        # Add some entries
        await manager.set("key1", "value1")
        await manager.set("key2", "value2", layer=CacheLayer.L2)

        info = await manager.get_cache_info()

        assert info["l1_size"] == 1
        assert info["l2_size"] == 1
        assert info["total_size"] == 2
        assert "l1_hit_rate" in info
        assert "l2_hit_rate" in info

    @pytest.mark.asyncio
    async def test_intelligent_cache_get_access_patterns(self):
        """Test getting access patterns."""
        strategy = CacheStrategy(name="test_strategy")
        manager = IntelligentCacheManager(strategy)

        # Add access pattern
        pattern = AccessPattern(key="test_key", frequency=10)
        manager.access_patterns["test_key"] = pattern

        patterns = await manager.get_access_patterns()

        assert len(patterns) == 1
        assert patterns[0].key == "test_key"
        assert patterns[0].frequency == 10


class TestCacheOptimizer:
    """Test CacheOptimizer class."""

    def test_cache_optimizer_initialization(self):
        """Test CacheOptimizer initialization."""
        optimizer = CacheOptimizer()

        assert optimizer.optimization_history == []
        assert optimizer.performance_baseline is None

    def test_cache_optimizer_analyze_performance(self):
        """Test performance analysis."""
        optimizer = CacheOptimizer()

        metrics = CacheMetrics()
        metrics.l1_hits = 80
        metrics.l1_misses = 20
        metrics.l2_hits = 15
        metrics.l2_misses = 5

        analysis = optimizer.analyze_performance(metrics)

        assert "l1_hit_rate" in analysis
        assert "l2_hit_rate" in analysis
        assert "recommendations" in analysis
        assert analysis["l1_hit_rate"] == 0.8
        assert analysis["l2_hit_rate"] == 0.75

    def test_cache_optimizer_generate_recommendations(self):
        """Test generating optimization recommendations."""
        optimizer = CacheOptimizer()

        metrics = CacheMetrics()
        metrics.l1_hits = 50
        metrics.l1_misses = 50
        metrics.l2_hits = 30
        metrics.l2_misses = 20

        recommendations = optimizer.generate_recommendations(metrics)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check for common recommendations
        rec_text = " ".join(recommendations)
        assert "hit rate" in rec_text.lower()

    def test_cache_optimizer_optimize_strategy(self):
        """Test strategy optimization."""
        optimizer = CacheOptimizer()

        strategy = CacheStrategy(
            name="test_strategy",
            l1_size=1000,
            l2_size=5000,
            promotion_threshold=5
        )

        metrics = CacheMetrics()
        metrics.l1_hits = 50
        metrics.l1_misses = 50
        metrics.promotions = 100  # High promotion rate

        optimized_strategy = optimizer.optimize_strategy(strategy, metrics)

        assert optimized_strategy.name == strategy.name
        # Should adjust promotion threshold due to high promotion rate
        assert optimized_strategy.promotion_threshold != strategy.promotion_threshold

    def test_cache_optimizer_record_optimization(self):
        """Test recording optimization."""
        optimizer = CacheOptimizer()

        old_strategy = CacheStrategy(name="old", l1_size=1000)
        new_strategy = CacheStrategy(name="new", l1_size=2000)

        optimizer.record_optimization(old_strategy, new_strategy, "Test optimization")

        assert len(optimizer.optimization_history) == 1
        history_entry = optimizer.optimization_history[0]
        assert history_entry["old_strategy"] == old_strategy
        assert history_entry["new_strategy"] == new_strategy
        assert history_entry["reason"] == "Test optimization"

    def test_cache_optimizer_get_optimization_history(self):
        """Test getting optimization history."""
        optimizer = CacheOptimizer()

        strategy1 = CacheStrategy(name="strategy1")
        strategy2 = CacheStrategy(name="strategy2")

        optimizer.record_optimization(strategy1, strategy2, "Test 1")
        optimizer.record_optimization(strategy2, strategy1, "Test 2")

        history = optimizer.get_optimization_history()

        assert len(history) == 2
        assert history[0]["reason"] == "Test 1"
        assert history[1]["reason"] == "Test 2"


class TestCacheAnalytics:
    """Test CacheAnalytics class."""

    def test_cache_analytics_initialization(self):
        """Test CacheAnalytics initialization."""
        analytics = CacheAnalytics()

        assert analytics.metrics_history == []
        assert analytics.performance_reports == []

    def test_cache_analytics_record_metrics(self):
        """Test recording metrics."""
        analytics = CacheAnalytics()

        metrics = CacheMetrics()
        metrics.l1_hits = 10
        metrics.l1_misses = 5

        analytics.record_metrics(metrics)

        assert len(analytics.metrics_history) == 1
        assert analytics.metrics_history[0]["metrics"] == metrics
        assert "timestamp" in analytics.metrics_history[0]

    def test_cache_analytics_generate_performance_report(self):
        """Test generating performance report."""
        analytics = CacheAnalytics()

        # Add metrics history
        for i in range(5):
            metrics = CacheMetrics()
            metrics.l1_hits = 10 + i
            metrics.l1_misses = 5 - i
            analytics.record_metrics(metrics)

        report = analytics.generate_performance_report()

        assert "period" in report
        assert "average_hit_rate" in report
        assert "trends" in report
        assert "recommendations" in report

    def test_cache_analytics_get_trend_analysis(self):
        """Test trend analysis."""
        analytics = CacheAnalytics()

        # Add metrics with improving trend
        for i in range(10):
            metrics = CacheMetrics()
            metrics.l1_hits = 10 + i  # Increasing hits
            metrics.l1_misses = 10 - i  # Decreasing misses
            analytics.record_metrics(metrics)

        trends = analytics.get_trend_analysis()

        assert "hit_rate_trend" in trends
        assert "performance_trend" in trends
        assert trends["hit_rate_trend"] == "improving"  # Should detect improvement

    def test_cache_analytics_get_usage_patterns(self):
        """Test usage pattern analysis."""
        analytics = CacheAnalytics()

        # Add metrics with different patterns
        for i in range(24):  # 24 hours
            metrics = CacheMetrics()
            metrics.l1_hits = 50 + (i % 8) * 10  # Peak every 8 hours
            metrics.l1_misses = 10
            analytics.record_metrics(metrics)

        patterns = analytics.get_usage_patterns()

        assert "peak_hours" in patterns
        assert "low_usage_hours" in patterns
        assert "average_requests_per_hour" in patterns

    def test_cache_analytics_export_report(self):
        """Test exporting report."""
        analytics = CacheAnalytics()

        # Add some data
        metrics = CacheMetrics()
        metrics.l1_hits = 10
        metrics.l1_misses = 5
        analytics.record_metrics(metrics)

        # Export to dict
        report = analytics.export_report("dict")

        assert isinstance(report, dict)
        assert "metrics_history" in report
        assert "performance_reports" in report


class TestCacheDecorators:
    """Test cache decorator functions."""

    @pytest.mark.asyncio
    async def test_cache_strategy_decorator(self):
        """Test cache_strategy decorator."""
        strategy = CacheStrategy(name="test_strategy")

        @cache_strategy(strategy)
        async def test_func(x, y):
            return x + y

        result = await test_func(1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_adaptive_cache_decorator(self):
        """Test adaptive_cache decorator."""
        @adaptive_cache(ttl=3600)
        async def test_func(x, y):
            return x * y

        result = await test_func(2, 3)
        assert result == 6

    @pytest.mark.asyncio
    async def test_predictive_cache_decorator(self):
        """Test predictive_cache decorator."""
        @predictive_cache(prefetch_threshold=0.8)
        async def test_func(x, y):
            return x ** y

        result = await test_func(2, 3)
        assert result == 8
