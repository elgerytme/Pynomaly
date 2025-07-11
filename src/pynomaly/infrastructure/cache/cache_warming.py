"""Cache warming strategies and utilities for Issue #99 enhanced Redis implementation.

This module provides intelligent cache warming capabilities to proactively
load frequently accessed data and improve application performance.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class WarmingStrategy:
    """Configuration for a specific cache warming strategy."""
    
    name: str
    priority: int = 1  # 1 = highest, 10 = lowest
    enabled: bool = True
    schedule: Optional[str] = None  # cron expression
    warm_on_startup: bool = True
    warm_on_demand: bool = True
    batch_size: int = 100
    delay_between_batches: float = 0.1
    ttl_seconds: Optional[int] = 3600
    tags: Set[str] = field(default_factory=set)
    data_generator: Optional[Callable] = None


@dataclass
class WarmingMetrics:
    """Metrics for cache warming operations."""
    
    strategy_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    items_warmed: int = 0
    items_failed: int = 0
    total_time_seconds: float = 0.0
    avg_item_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        total = self.items_warmed + self.items_failed
        return (self.items_warmed / total * 100) if total > 0 else 0.0
    
    @property
    def items_per_second(self) -> float:
        """Calculate items warmed per second."""
        return self.items_warmed / self.total_time_seconds if self.total_time_seconds > 0 else 0.0


class CacheWarmingService:
    """Service for managing cache warming strategies and execution."""
    
    def __init__(self, cache_backend, enable_background_warming: bool = True):
        """Initialize cache warming service.
        
        Args:
            cache_backend: Redis cache backend instance
            enable_background_warming: Enable automatic background warming
        """
        self.cache = cache_backend
        self.enable_background_warming = enable_background_warming
        self.strategies: Dict[str, WarmingStrategy] = {}
        self.metrics: Dict[str, List[WarmingMetrics]] = {}
        self.warming_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Default strategies
        self._register_default_strategies()
        
        logger.info(f"CacheWarmingService initialized - Background warming: {enable_background_warming}")
    
    def _register_default_strategies(self) -> None:
        """Register default cache warming strategies."""
        
        # Critical application data (highest priority)
        self.register_strategy(WarmingStrategy(
            name="critical_app_data",
            priority=1,
            warm_on_startup=True,
            batch_size=50,
            delay_between_batches=0.05,
            ttl_seconds=7200,  # 2 hours
            tags={"critical", "app_config"},
            data_generator=self._generate_critical_data
        ))
        
        # Detector models and configurations
        self.register_strategy(WarmingStrategy(
            name="detector_models",
            priority=2,
            warm_on_startup=True,
            batch_size=20,
            delay_between_batches=0.1,
            ttl_seconds=3600,  # 1 hour
            tags={"models", "detectors"},
            data_generator=self._generate_detector_data
        ))
        
        # Frequently accessed datasets
        self.register_strategy(WarmingStrategy(
            name="popular_datasets",
            priority=3,
            warm_on_startup=False,
            batch_size=10,
            delay_between_batches=0.2,
            ttl_seconds=1800,  # 30 minutes
            tags={"datasets", "popular"},
            data_generator=self._generate_dataset_data
        ))
        
        # API response caches
        self.register_strategy(WarmingStrategy(
            name="api_responses",
            priority=4,
            warm_on_startup=False,
            batch_size=100,
            delay_between_batches=0.05,
            ttl_seconds=600,  # 10 minutes
            tags={"api", "responses"},
            data_generator=self._generate_api_cache_data
        ))
        
        # User session data
        self.register_strategy(WarmingStrategy(
            name="user_sessions",
            priority=5,
            warm_on_startup=False,
            batch_size=50,
            delay_between_batches=0.1,
            ttl_seconds=1800,  # 30 minutes
            tags={"users", "sessions"},
            data_generator=self._generate_session_data
        ))
    
    def register_strategy(self, strategy: WarmingStrategy) -> None:
        """Register a new warming strategy."""
        self.strategies[strategy.name] = strategy
        if strategy.name not in self.metrics:
            self.metrics[strategy.name] = []
        logger.info(f"Registered warming strategy: {strategy.name} (priority {strategy.priority})")
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a warming strategy."""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            logger.info(f"Removed warming strategy: {strategy_name}")
            return True
        return False
    
    async def warm_on_startup(self) -> Dict[str, WarmingMetrics]:
        """Execute startup warming for all enabled strategies."""
        logger.info("Starting cache warming on application startup")
        
        startup_strategies = [
            strategy for strategy in self.strategies.values()
            if strategy.enabled and strategy.warm_on_startup
        ]
        
        # Sort by priority (lowest number = highest priority)
        startup_strategies.sort(key=lambda s: s.priority)
        
        results = {}
        for strategy in startup_strategies:
            try:
                metrics = await self._execute_strategy(strategy)
                results[strategy.name] = metrics
                logger.info(
                    f"Startup warming completed for {strategy.name}: "
                    f"{metrics.items_warmed} items in {metrics.total_time_seconds:.2f}s"
                )
            except Exception as e:
                logger.error(f"Startup warming failed for {strategy.name}: {e}")
                results[strategy.name] = WarmingMetrics(
                    strategy_name=strategy.name,
                    started_at=datetime.utcnow(),
                    items_failed=1,
                    errors=[str(e)]
                )
        
        logger.info(f"Startup cache warming completed for {len(results)} strategies")
        return results
    
    async def warm_strategy(self, strategy_name: str) -> Optional[WarmingMetrics]:
        """Manually execute a specific warming strategy."""
        if strategy_name not in self.strategies:
            logger.error(f"Unknown warming strategy: {strategy_name}")
            return None
        
        strategy = self.strategies[strategy_name]
        if not strategy.enabled:
            logger.warning(f"Strategy {strategy_name} is disabled")
            return None
        
        logger.info(f"Manually executing warming strategy: {strategy_name}")
        return await self._execute_strategy(strategy)
    
    async def warm_all_strategies(self) -> Dict[str, WarmingMetrics]:
        """Execute all enabled warming strategies."""
        logger.info("Executing all enabled warming strategies")
        
        enabled_strategies = [
            strategy for strategy in self.strategies.values()
            if strategy.enabled
        ]
        
        # Sort by priority
        enabled_strategies.sort(key=lambda s: s.priority)
        
        results = {}
        for strategy in enabled_strategies:
            try:
                metrics = await self._execute_strategy(strategy)
                results[strategy.name] = metrics
            except Exception as e:
                logger.error(f"Warming failed for {strategy.name}: {e}")
                results[strategy.name] = WarmingMetrics(
                    strategy_name=strategy.name,
                    started_at=datetime.utcnow(),
                    items_failed=1,
                    errors=[str(e)]
                )
        
        return results
    
    async def _execute_strategy(self, strategy: WarmingStrategy) -> WarmingMetrics:
        """Execute a single warming strategy."""
        metrics = WarmingMetrics(
            strategy_name=strategy.name,
            started_at=datetime.utcnow()
        )
        
        try:
            # Generate warming data
            if strategy.data_generator:
                warming_data = await strategy.data_generator()
            else:
                warming_data = {}
            
            if not warming_data:
                logger.warning(f"No warming data generated for strategy {strategy.name}")
                metrics.completed_at = datetime.utcnow()
                return metrics
            
            # Execute warming in batches
            keys = list(warming_data.keys())
            total_items = len(keys)
            
            logger.info(f"Warming {total_items} items for strategy {strategy.name}")
            
            for i in range(0, total_items, strategy.batch_size):
                batch_keys = keys[i:i + strategy.batch_size]
                batch_start = time.time()
                
                # Warm batch items
                for key in batch_keys:
                    try:
                        await self.cache.set(
                            key=key,
                            value=warming_data[key],
                            ttl=strategy.ttl_seconds,
                            tags=strategy.tags
                        )
                        metrics.items_warmed += 1
                    except Exception as e:
                        metrics.items_failed += 1
                        metrics.errors.append(f"Failed to warm {key}: {e}")
                        logger.warning(f"Failed to warm cache key {key}: {e}")
                
                batch_time = time.time() - batch_start
                
                # Log batch progress
                if i + strategy.batch_size < total_items:
                    progress = ((i + strategy.batch_size) / total_items) * 100
                    logger.debug(
                        f"Strategy {strategy.name} progress: {progress:.1f}% "
                        f"({i + strategy.batch_size}/{total_items})"
                    )
                
                # Delay between batches to prevent overwhelming Redis
                if strategy.delay_between_batches > 0 and i + strategy.batch_size < total_items:
                    await asyncio.sleep(strategy.delay_between_batches)
            
            # Calculate final metrics
            metrics.completed_at = datetime.utcnow()
            metrics.total_time_seconds = (
                metrics.completed_at - metrics.started_at
            ).total_seconds()
            
            if metrics.items_warmed > 0:
                metrics.avg_item_time_ms = (
                    metrics.total_time_seconds / metrics.items_warmed * 1000
                )
            
            # Store metrics
            self.metrics[strategy.name].append(metrics)
            
            # Keep only last 10 metrics per strategy
            if len(self.metrics[strategy.name]) > 10:
                self.metrics[strategy.name] = self.metrics[strategy.name][-10:]
            
            logger.info(
                f"Warming strategy {strategy.name} completed: "
                f"{metrics.items_warmed} items warmed, "
                f"{metrics.items_failed} failed, "
                f"{metrics.success_rate:.1f}% success rate, "
                f"{metrics.total_time_seconds:.2f}s total time"
            )
            
        except Exception as e:
            metrics.completed_at = datetime.utcnow()
            metrics.items_failed += 1
            metrics.errors.append(f"Strategy execution failed: {e}")
            logger.error(f"Warming strategy {strategy.name} failed: {e}")
            raise
        
        return metrics
    
    async def start_background_warming(self) -> None:
        """Start background warming tasks."""
        if not self.enable_background_warming:
            logger.info("Background warming is disabled")
            return
        
        logger.info("Starting background cache warming service")
        
        # Start periodic warming task
        warming_task = asyncio.create_task(self._background_warming_loop())
        self.warming_tasks.add(warming_task)
        
        # Cleanup finished tasks
        warming_task.add_done_callback(self.warming_tasks.discard)
    
    async def _background_warming_loop(self) -> None:
        """Background loop for periodic cache warming."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for next warming cycle (default: every 30 minutes)
                await asyncio.wait_for(
                    self._shutdown_event.wait(), 
                    timeout=1800  # 30 minutes
                )
                break  # Shutdown event was set
            except asyncio.TimeoutError:
                # Timeout reached, execute warming cycle
                pass
            
            try:
                logger.info("Executing background cache warming cycle")
                
                # Warm strategies that support background warming
                background_strategies = [
                    strategy for strategy in self.strategies.values()
                    if strategy.enabled and hasattr(strategy, 'warm_in_background')
                    and getattr(strategy, 'warm_in_background', False)
                ]
                
                for strategy in background_strategies:
                    await self._execute_strategy(strategy)
                
                logger.info("Background warming cycle completed")
                
            except Exception as e:
                logger.error(f"Background warming cycle failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown warming service and cleanup tasks."""
        logger.info("Shutting down cache warming service")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel all warming tasks
        for task in self.warming_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.warming_tasks:
            await asyncio.gather(*self.warming_tasks, return_exceptions=True)
        
        logger.info("Cache warming service shutdown complete")
    
    def get_strategy_metrics(self, strategy_name: str) -> List[WarmingMetrics]:
        """Get metrics for a specific strategy."""
        return self.metrics.get(strategy_name, [])
    
    def get_all_metrics(self) -> Dict[str, List[WarmingMetrics]]:
        """Get all warming metrics."""
        return self.metrics.copy()
    
    def get_strategy_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all warming strategies."""
        summary = {}
        
        for name, strategy in self.strategies.items():
            recent_metrics = self.metrics.get(name, [])
            last_execution = recent_metrics[-1] if recent_metrics else None
            
            summary[name] = {
                "enabled": strategy.enabled,
                "priority": strategy.priority,
                "warm_on_startup": strategy.warm_on_startup,
                "batch_size": strategy.batch_size,
                "ttl_seconds": strategy.ttl_seconds,
                "tags": list(strategy.tags),
                "executions_count": len(recent_metrics),
                "last_execution": {
                    "started_at": last_execution.started_at.isoformat() if last_execution else None,
                    "completed_at": last_execution.completed_at.isoformat() if last_execution and last_execution.completed_at else None,
                    "items_warmed": last_execution.items_warmed if last_execution else 0,
                    "items_failed": last_execution.items_failed if last_execution else 0,
                    "success_rate": last_execution.success_rate if last_execution else 0.0,
                    "total_time_seconds": last_execution.total_time_seconds if last_execution else 0.0,
                } if last_execution else None
            }
        
        return summary
    
    # Data generation methods for default strategies
    async def _generate_critical_data(self) -> Dict[str, Any]:
        """Generate critical application data for warming."""
        return {
            "app:config:settings": {"cache_enabled": True, "log_level": "INFO"},
            "app:config:features": {"automl": True, "explanations": True},
            "app:health:status": {"status": "healthy", "timestamp": datetime.utcnow().isoformat()},
            "app:version": {"version": "1.0.0", "build": "latest"},
        }
    
    async def _generate_detector_data(self) -> Dict[str, Any]:
        """Generate detector model data for warming."""
        return {
            "detector:isolation_forest:default": {"contamination": 0.1, "random_state": 42},
            "detector:local_outlier_factor:default": {"n_neighbors": 20, "contamination": 0.1},
            "detector:one_class_svm:default": {"nu": 0.1, "kernel": "rbf"},
            "detector:elliptic_envelope:default": {"contamination": 0.1, "random_state": 42},
        }
    
    async def _generate_dataset_data(self) -> Dict[str, Any]:
        """Generate popular dataset metadata for warming."""
        return {
            "dataset:sample:metadata": {"name": "sample", "features": 3, "samples": 100},
            "dataset:popular:list": ["sample", "test", "demo"],
            "dataset:stats:summary": {"total_datasets": 10, "avg_size": 1000},
        }
    
    async def _generate_api_cache_data(self) -> Dict[str, Any]:
        """Generate API response cache data for warming."""
        return {
            "api:algorithms:list": ["IsolationForest", "LOF", "OneClassSVM"],
            "api:status:health": {"status": "ok", "checks": {"cache": "pass"}},
            "api:config:openapi": {"version": "3.0.0", "title": "Pynomaly API"},
        }
    
    async def _generate_session_data(self) -> Dict[str, Any]:
        """Generate user session data for warming."""
        return {
            "session:default:preferences": {"theme": "light", "notifications": True},
            "session:config:defaults": {"timeout": 3600, "remember_me": False},
        }