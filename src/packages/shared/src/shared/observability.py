"""
Observability framework for monitoring package interactions.

This module provides comprehensive monitoring, metrics collection, and
observability features for the package interaction framework including:
- Event bus monitoring
- Dependency injection metrics
- Cross-domain communication tracking
- Performance monitoring
- Health checks
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json

from interfaces.events import DomainEvent, EventPriority
from interfaces.patterns import HealthCheck, MetricsCollector


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""
    timestamp: datetime
    event_bus_metrics: Dict[str, Any]
    di_container_metrics: Dict[str, Any]
    system_metrics: Dict[str, Any]
    active_handlers: int
    queue_depths: Dict[str, int]


class EventBusMonitor:
    """Monitor for event bus operations and performance."""
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.metrics = {
            'events_published_total': 0,
            'events_processed_total': 0,
            'events_failed_total': 0,
            'handler_execution_times': deque(maxlen=1000),
            'queue_depths_history': deque(maxlen=100),
            'error_rate_history': deque(maxlen=100),
            'throughput_history': deque(maxlen=100),
        }
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.last_metrics_update = datetime.utcnow()
        self.event_type_stats = defaultdict(lambda: {
            'count': 0,
            'avg_processing_time': 0.0,
            'error_count': 0
        })
        
        # Health thresholds
        self.health_thresholds = {
            'max_queue_depth': 1000,
            'max_error_rate': 0.05,  # 5%
            'max_avg_processing_time': 1000,  # 1 second
        }
    
    def record_event_published(self, event: DomainEvent):
        """Record when an event is published."""
        self.metrics['events_published_total'] += 1
        
        # Update event type stats
        event_type = type(event).__name__
        self.event_type_stats[event_type]['count'] += 1
        
        logger.debug(f"Event published: {event_type} (ID: {event.event_id})")
    
    def record_event_processed(self, event: DomainEvent, processing_time_ms: float):
        """Record when an event is successfully processed."""
        self.metrics['events_processed_total'] += 1
        self.metrics['handler_execution_times'].append(processing_time_ms)
        
        # Update event type stats
        event_type = type(event).__name__
        stats = self.event_type_stats[event_type]
        
        # Calculate moving average
        current_avg = stats['avg_processing_time']
        count = stats['count']
        stats['avg_processing_time'] = ((current_avg * (count - 1)) + processing_time_ms) / count
        
        logger.debug(f"Event processed: {event_type} in {processing_time_ms:.2f}ms")
    
    def record_event_failed(self, event: DomainEvent, error: Exception):
        """Record when event processing fails."""
        self.metrics['events_failed_total'] += 1
        
        # Update event type stats
        event_type = type(event).__name__
        self.event_type_stats[event_type]['error_count'] += 1
        
        logger.error(f"Event processing failed: {event_type} - {error}")
    
    def record_queue_depths(self, queue_depths: Dict[str, int]):
        """Record current queue depths."""
        total_depth = sum(queue_depths.values())
        self.metrics['queue_depths_history'].append({
            'timestamp': datetime.utcnow(),
            'total_depth': total_depth,
            'by_priority': queue_depths.copy()
        })
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the event bus."""
        current_time = datetime.utcnow()
        uptime_seconds = (current_time - self.start_time).total_seconds()
        
        # Calculate error rate
        total_events = self.metrics['events_published_total']
        failed_events = self.metrics['events_failed_total']
        error_rate = failed_events / max(total_events, 1)
        
        # Calculate average processing time
        processing_times = list(self.metrics['handler_execution_times'])
        avg_processing_time = sum(processing_times) / max(len(processing_times), 1)
        
        # Get current queue depth
        latest_queue_data = (
            list(self.metrics['queue_depths_history'])[-1] 
            if self.metrics['queue_depths_history'] else None
        )
        current_queue_depth = latest_queue_data['total_depth'] if latest_queue_data else 0
        
        # Determine health status
        health_issues = []
        
        if current_queue_depth > self.health_thresholds['max_queue_depth']:
            health_issues.append(f"High queue depth: {current_queue_depth}")
        
        if error_rate > self.health_thresholds['max_error_rate']:
            health_issues.append(f"High error rate: {error_rate:.2%}")
        
        if avg_processing_time > self.health_thresholds['max_avg_processing_time']:
            health_issues.append(f"Slow processing: {avg_processing_time:.2f}ms")
        
        status = "healthy" if not health_issues else "degraded"
        
        return {
            'status': status,
            'uptime_seconds': uptime_seconds,
            'events_published': total_events,
            'events_processed': self.metrics['events_processed_total'],
            'events_failed': failed_events,
            'error_rate': error_rate,
            'avg_processing_time_ms': avg_processing_time,
            'current_queue_depth': current_queue_depth,
            'health_issues': health_issues,
            'event_type_stats': dict(self.event_type_stats),
            'timestamp': current_time
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        processing_times = list(self.metrics['handler_execution_times'])
        
        if processing_times:
            processing_times.sort()
            p50 = processing_times[len(processing_times) // 2]
            p95 = processing_times[int(len(processing_times) * 0.95)]
            p99 = processing_times[int(len(processing_times) * 0.99)]
        else:
            p50 = p95 = p99 = 0
        
        return {
            'processing_time_p50': p50,
            'processing_time_p95': p95,
            'processing_time_p99': p99,
            'throughput_events_per_second': self._calculate_throughput(),
            'queue_depths_history': list(self.metrics['queue_depths_history']),
            'event_distribution': self._get_event_distribution()
        }
    
    def _calculate_throughput(self) -> float:
        """Calculate events per second throughput."""
        current_time = datetime.utcnow()
        time_window = timedelta(seconds=60)  # 1 minute window
        
        # Count events in the last minute (simplified)
        # In a real implementation, would track timestamps
        uptime_seconds = (current_time - self.start_time).total_seconds()
        if uptime_seconds > 0:
            return self.metrics['events_processed_total'] / uptime_seconds
        return 0.0
    
    def _get_event_distribution(self) -> Dict[str, int]:
        """Get distribution of events by type."""
        return {
            event_type: stats['count']
            for event_type, stats in self.event_type_stats.items()
        }


class DIContainerMonitor:
    """Monitor for dependency injection container operations."""
    
    def __init__(self):
        self.metrics = {
            'resolutions_total': 0,
            'resolution_times': deque(maxlen=1000),
            'cache_hits': 0,
            'cache_misses': 0,
            'circular_dependency_errors': 0,
            'registration_count': 0,
        }
        
        self.service_stats = defaultdict(lambda: {
            'resolution_count': 0,
            'avg_resolution_time': 0.0,
            'lifecycle': 'unknown'
        })
        
        self.start_time = datetime.utcnow()
    
    def record_resolution(self, service_type: type, resolution_time_ms: float, from_cache: bool = False):
        """Record a service resolution."""
        self.metrics['resolutions_total'] += 1
        self.metrics['resolution_times'].append(resolution_time_ms)
        
        if from_cache:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
        
        # Update service stats
        service_name = service_type.__name__
        stats = self.service_stats[service_name]
        stats['resolution_count'] += 1
        
        # Calculate moving average
        current_avg = stats['avg_resolution_time']
        count = stats['resolution_count']
        stats['avg_resolution_time'] = ((current_avg * (count - 1)) + resolution_time_ms) / count
    
    def record_registration(self, service_type: type, lifecycle: str):
        """Record a service registration."""
        self.metrics['registration_count'] += 1
        service_name = service_type.__name__
        self.service_stats[service_name]['lifecycle'] = lifecycle
    
    def record_circular_dependency_error(self):
        """Record a circular dependency error."""
        self.metrics['circular_dependency_errors'] += 1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the DI container."""
        resolution_times = list(self.metrics['resolution_times'])
        avg_resolution_time = sum(resolution_times) / max(len(resolution_times), 1)
        
        cache_hit_rate = (
            self.metrics['cache_hits'] / 
            max(self.metrics['cache_hits'] + self.metrics['cache_misses'], 1)
        )
        
        health_issues = []
        if avg_resolution_time > 10:  # 10ms threshold
            health_issues.append(f"Slow resolution: {avg_resolution_time:.2f}ms")
        
        if cache_hit_rate < 0.8:  # 80% threshold
            health_issues.append(f"Low cache hit rate: {cache_hit_rate:.2%}")
        
        if self.metrics['circular_dependency_errors'] > 0:
            health_issues.append(f"Circular dependencies: {self.metrics['circular_dependency_errors']}")
        
        status = "healthy" if not health_issues else "degraded"
        
        return {
            'status': status,
            'resolutions_total': self.metrics['resolutions_total'],
            'avg_resolution_time_ms': avg_resolution_time,
            'cache_hit_rate': cache_hit_rate,
            'registered_services': self.metrics['registration_count'],
            'circular_dependency_errors': self.metrics['circular_dependency_errors'],
            'health_issues': health_issues,
            'service_stats': dict(self.service_stats),
            'timestamp': datetime.utcnow()
        }


class InteractionFrameworkMetrics(MetricsCollector):
    """Comprehensive metrics collector for the interaction framework."""
    
    def __init__(self):
        self.event_bus_monitor = EventBusMonitor()
        self.di_container_monitor = DIContainerMonitor()
        self.custom_metrics = {}
        self.start_time = datetime.utcnow()
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        if name not in self.custom_metrics:
            self.custom_metrics[name] = 0
        self.custom_metrics[name] += value
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        self.custom_metrics[name] = value
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        hist_key = f"{name}_histogram"
        if hist_key not in self.custom_metrics:
            self.custom_metrics[hist_key] = deque(maxlen=1000)
        self.custom_metrics[hist_key].append(value)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            'framework': {
                'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
                'version': '1.0.0',
                'timestamp': datetime.utcnow()
            },
            'event_bus': self.event_bus_monitor.get_health_status(),
            'event_bus_performance': self.event_bus_monitor.get_performance_metrics(),
            'di_container': self.di_container_monitor.get_health_status(),
            'custom_metrics': dict(self.custom_metrics)
        }
    
    def create_performance_snapshot(self) -> PerformanceSnapshot:
        """Create a performance snapshot."""
        event_bus_metrics = self.event_bus_monitor.get_health_status()
        di_metrics = self.di_container_monitor.get_health_status()
        
        return PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            event_bus_metrics=event_bus_metrics,
            di_container_metrics=di_metrics,
            system_metrics=self.custom_metrics.copy(),
            active_handlers=len(self.event_bus_monitor.event_type_stats),
            queue_depths={}  # Would be populated from actual event bus
        )


class InteractionFrameworkHealthCheck(HealthCheck):
    """Health check for the interaction framework."""
    
    def __init__(self, metrics_collector: InteractionFrameworkMetrics):
        self.metrics_collector = metrics_collector
    
    async def check_health(self) -> Dict[str, Any]:
        """Check overall health of the interaction framework."""
        event_bus_health = self.metrics_collector.event_bus_monitor.get_health_status()
        di_health = self.metrics_collector.di_container_monitor.get_health_status()
        
        # Determine overall status
        component_statuses = [event_bus_health['status'], di_health['status']]
        overall_status = "healthy" if all(s == "healthy" for s in component_statuses) else "degraded"
        
        return {
            'status': overall_status,
            'timestamp': datetime.utcnow(),
            'components': {
                'event_bus': {
                    'status': event_bus_health['status'],
                    'issues': event_bus_health['health_issues']
                },
                'di_container': {
                    'status': di_health['status'],
                    'issues': di_health['health_issues']
                }
            },
            'summary': {
                'total_events_processed': event_bus_health['events_processed'],
                'total_services_resolved': di_health['resolutions_total'],
                'uptime_seconds': (datetime.utcnow() - self.metrics_collector.start_time).total_seconds()
            }
        }
    
    def get_component_name(self) -> str:
        """Get the component name."""
        return "interaction_framework"
    
    async def check_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Check health of framework dependencies."""
        return {
            'event_bus': self.metrics_collector.event_bus_monitor.get_health_status(),
            'di_container': self.metrics_collector.di_container_monitor.get_health_status()
        }


class ObservabilityDashboard:
    """Dashboard for visualizing interaction framework metrics."""
    
    def __init__(self, metrics_collector: InteractionFrameworkMetrics):
        self.metrics_collector = metrics_collector
        self.health_check = InteractionFrameworkHealthCheck(metrics_collector)
    
    async def generate_status_report(self) -> str:
        """Generate a text-based status report."""
        health = await self.health_check.check_health()
        metrics = await self.metrics_collector.get_metrics()
        
        report = []
        report.append("🔍 Interaction Framework Status Report")
        report.append("=" * 50)
        report.append(f"Overall Status: {'✅' if health['status'] == 'healthy' else '⚠️'} {health['status'].upper()}")
        report.append(f"Uptime: {health['summary']['uptime_seconds']:.0f} seconds")
        report.append("")
        
        # Event Bus Status
        report.append("📡 Event Bus")
        report.append("-" * 20)
        eb_metrics = metrics['event_bus']
        report.append(f"Status: {eb_metrics['status']}")
        report.append(f"Events Published: {eb_metrics['events_published']}")
        report.append(f"Events Processed: {eb_metrics['events_processed']}")
        report.append(f"Error Rate: {eb_metrics['error_rate']:.2%}")
        report.append(f"Avg Processing Time: {eb_metrics['avg_processing_time_ms']:.2f}ms")
        
        if eb_metrics['health_issues']:
            report.append("Issues:")
            for issue in eb_metrics['health_issues']:
                report.append(f"  - {issue}")
        report.append("")
        
        # DI Container Status
        report.append("🔧 Dependency Injection")
        report.append("-" * 25)
        di_metrics = metrics['di_container']
        report.append(f"Status: {di_metrics['status']}")
        report.append(f"Total Resolutions: {di_metrics['resolutions_total']}")
        report.append(f"Cache Hit Rate: {di_metrics['cache_hit_rate']:.2%}")
        report.append(f"Avg Resolution Time: {di_metrics['avg_resolution_time_ms']:.2f}ms")
        report.append(f"Registered Services: {di_metrics['registered_services']}")
        
        if di_metrics['health_issues']:
            report.append("Issues:")
            for issue in di_metrics['health_issues']:
                report.append(f"  - {issue}")
        report.append("")
        
        # Performance Summary
        perf_metrics = metrics['event_bus_performance']
        report.append("📊 Performance Summary")
        report.append("-" * 22)
        report.append(f"P50 Processing Time: {perf_metrics['processing_time_p50']:.2f}ms")
        report.append(f"P95 Processing Time: {perf_metrics['processing_time_p95']:.2f}ms")
        report.append(f"P99 Processing Time: {perf_metrics['processing_time_p99']:.2f}ms")
        report.append(f"Throughput: {perf_metrics['throughput_events_per_second']:.2f} events/sec")
        
        return "\n".join(report)
    
    async def generate_json_metrics(self) -> str:
        """Generate JSON metrics for external monitoring systems."""
        health = await self.health_check.check_health()
        metrics = await self.metrics_collector.get_metrics()
        
        combined_metrics = {
            'health': health,
            'metrics': metrics,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return json.dumps(combined_metrics, indent=2, default=str)


# Global instances for easy access
_global_metrics_collector = None
_global_health_check = None
_global_dashboard = None


def get_metrics_collector() -> InteractionFrameworkMetrics:
    """Get the global metrics collector."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = InteractionFrameworkMetrics()
    return _global_metrics_collector


def get_health_check() -> InteractionFrameworkHealthCheck:
    """Get the global health check."""
    global _global_health_check
    if _global_health_check is None:
        _global_health_check = InteractionFrameworkHealthCheck(get_metrics_collector())
    return _global_health_check


def get_dashboard() -> ObservabilityDashboard:
    """Get the global observability dashboard."""
    global _global_dashboard
    if _global_dashboard is None:
        _global_dashboard = ObservabilityDashboard(get_metrics_collector())
    return _global_dashboard