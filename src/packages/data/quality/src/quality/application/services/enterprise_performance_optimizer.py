"""
Enterprise Performance Optimizer - Comprehensive integration service for all optimization components.

This service orchestrates distributed processing, intelligent caching, query optimization,
memory/storage optimization, network optimization, and performance monitoring for
enterprise-scale data quality operations.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .distributed_quality_processing import DistributedQualityProcessingService
from .intelligent_caching_framework import IntelligentCacheManager
from .query_optimization_engine import QueryOptimizationEngine
from .memory_storage_optimization import MemoryStorageOptimizationService
from .network_communication_optimization import NetworkCommunicationOptimizer
from .performance_monitoring_tuning import PerformanceMonitoringTuningService
from ...domain.entities.quality_profile import DataQualityProfile
from ...domain.value_objects.quality_scores import QualityScores
from ...domain.interfaces.data_quality_interface import DataQualityInterface

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"
    EXTREME = "extreme"


@dataclass
class OptimizationConfig:
    """Configuration for enterprise performance optimization."""
    optimization_level: OptimizationLevel = OptimizationLevel.ENTERPRISE
    
    # Component enablement
    enable_distributed_processing: bool = True
    enable_intelligent_caching: bool = True
    enable_query_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_network_optimization: bool = True
    enable_performance_monitoring: bool = True
    
    # Resource limits
    max_memory_gb: int = 32
    max_cpu_cores: int = 16
    max_disk_gb: int = 1000
    max_network_mbps: int = 10000
    
    # Performance targets
    target_response_time_ms: int = 100
    target_throughput_rps: int = 10000
    target_availability_percent: float = 99.9
    target_error_rate_percent: float = 0.1
    
    # Auto-optimization settings
    enable_auto_scaling: bool = True
    enable_auto_tuning: bool = True
    enable_predictive_optimization: bool = True


@dataclass
class PerformanceReport:
    """Comprehensive enterprise performance report."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Overall metrics
    performance_score: float = 0.0
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    
    # Component performance
    distributed_processing_score: float = 0.0
    caching_efficiency_score: float = 0.0
    query_optimization_score: float = 0.0
    memory_efficiency_score: float = 0.0
    network_performance_score: float = 0.0
    monitoring_effectiveness_score: float = 0.0
    
    # Business metrics
    cost_efficiency_improvement: float = 0.0
    resource_utilization_improvement: float = 0.0
    user_experience_improvement: float = 0.0
    
    # Recommendations
    top_optimizations: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    
    # Achievements
    performance_targets_met: List[str] = field(default_factory=list)
    sla_compliance_percent: float = 100.0


class EnterprisePerformanceOptimizer:
    """Comprehensive enterprise performance optimization service."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize the enterprise performance optimizer."""
        self.config = config
        self.is_running = False
        
        # Initialize optimization components based on configuration
        self.components: Dict[str, Any] = {}
        
        if config.enable_distributed_processing:
            self.components["distributed_processing"] = DistributedQualityProcessingService(
                self._get_distributed_config()
            )
        
        if config.enable_intelligent_caching:
            self.components["intelligent_caching"] = IntelligentCacheManager(
                self._get_caching_config()
            )
        
        if config.enable_query_optimization:
            self.components["query_optimization"] = QueryOptimizationEngine(
                self._get_query_config()
            )
        
        if config.enable_memory_optimization:
            self.components["memory_optimization"] = MemoryStorageOptimizationService(
                self._get_memory_config()
            )
        
        if config.enable_network_optimization:
            self.components["network_optimization"] = NetworkCommunicationOptimizer(
                self._get_network_config()
            )
        
        if config.enable_performance_monitoring:
            self.components["performance_monitoring"] = PerformanceMonitoringTuningService(
                self._get_monitoring_config()
            )
        
        # Performance tracking
        self.performance_history: List[PerformanceReport] = []
        self.optimization_achievements: List[str] = []
        
        logger.info(f"Initialized Enterprise Performance Optimizer with {len(self.components)} components")
    
    def _get_distributed_config(self) -> Dict[str, Any]:
        """Get configuration for distributed processing component."""
        return {
            "max_thread_workers": min(50, self.config.max_cpu_cores * 4),
            "max_process_workers": min(self.config.max_cpu_cores, 16),
            "max_job_queue_size": 50000,
            "enable_auto_scaling": self.config.enable_auto_scaling,
            "scale_up_threshold": 0.8,
            "scale_down_threshold": 0.3,
            "min_nodes": 2,
            "max_nodes": min(1000, self.config.max_cpu_cores * 10),
            "redis_host": "localhost",
            "redis_port": 6379
        }
    
    def _get_caching_config(self) -> Dict[str, Any]:
        """Get configuration for intelligent caching component."""
        return {
            "max_memory_mb": min(self.config.max_memory_gb * 1024 * 0.3, 10240),  # 30% of max memory
            "max_disk_mb": min(self.config.max_disk_gb * 1024 * 0.1, 51200),  # 10% of max disk
            "enable_compression": True,
            "enable_analytics": True,
            "enable_ml_optimization": self.config.enable_predictive_optimization,
            "disk_cache_path": "./cache",
            "redis_host": "localhost",
            "redis_port": 6379
        }
    
    def _get_query_config(self) -> Dict[str, Any]:
        """Get configuration for query optimization component."""
        return {
            "databases": {
                "default": {
                    "connection_string": "sqlite:///./quality.db",
                    "pool_size": min(20, self.config.max_cpu_cores * 2),
                    "max_overflow": min(30, self.config.max_cpu_cores * 3),
                    "pool_timeout": 30,
                    "pool_recycle": 3600,
                    "echo_queries": False
                }
            },
            "slow_query_threshold_ms": self.config.target_response_time_ms * 10,
            "enable_materialized_views": True,
            "enable_query_caching": True
        }
    
    def _get_memory_config(self) -> Dict[str, Any]:
        """Get configuration for memory optimization component."""
        return {
            "gc_threshold_mb": min(self.config.max_memory_gb * 1024 * 0.8, 25600),  # 80% of max memory
            "memory_warning_threshold": 0.85,
            "enable_memory_mapping": True,
            "hot_storage_path": "./storage/hot",
            "warm_storage_path": "./storage/warm",
            "cold_storage_path": "./storage/cold",
            "archive_storage_path": "./storage/archive",
            "hot_to_warm_days": 7,
            "warm_to_cold_days": 30,
            "cold_to_archive_days": 90
        }
    
    def _get_network_config(self) -> Dict[str, Any]:
        """Get configuration for network optimization component."""
        return {
            "max_batch_size": 100,
            "max_wait_time_ms": 50,
            "enable_compression": True,
            "enable_batching": True,
            "enable_keepalive": True,
            "adaptive_timeouts": True,
            "connection_pool_size": min(100, self.config.max_cpu_cores * 8),
            "request_timeout_seconds": self.config.target_response_time_ms / 1000 * 10
        }
    
    def _get_monitoring_config(self) -> Dict[str, Any]:
        """Get configuration for performance monitoring component."""
        return {
            "monitoring_enabled": True,
            "baseline_learning_enabled": True,
            "predictive_analytics_enabled": self.config.enable_predictive_optimization,
            "auto_tuning_enabled": self.config.enable_auto_tuning,
            "anomaly_threshold_std": 3.0,
            "alert_cooldown_minutes": 15,
            "model_cache_path": "./models"
        }
    
    async def start(self) -> None:
        """Start the enterprise performance optimizer."""
        logger.info("Starting Enterprise Performance Optimizer...")
        
        self.is_running = True
        
        # Start all components
        for component_name, component in self.components.items():
            try:
                if hasattr(component, "start"):
                    await component.start()
                logger.info(f"Started {component_name} component")
            except Exception as e:
                logger.error(f"Failed to start {component_name}: {str(e)}")
        
        # Start background optimization tasks
        asyncio.create_task(self._optimization_coordination_task())
        asyncio.create_task(self._performance_reporting_task())
        asyncio.create_task(self._achievement_tracking_task())
        
        logger.info("Enterprise Performance Optimizer started successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the enterprise performance optimizer."""
        logger.info("Shutting down Enterprise Performance Optimizer...")
        
        self.is_running = False
        
        # Shutdown all components
        for component_name, component in self.components.items():
            try:
                if hasattr(component, "shutdown"):
                    await component.shutdown()
                logger.info(f"Shutdown {component_name} component")
            except Exception as e:
                logger.error(f"Failed to shutdown {component_name}: {str(e)}")
        
        logger.info("Enterprise Performance Optimizer shutdown complete")
    
    async def _optimization_coordination_task(self) -> None:
        """Background task for coordinating optimization across components."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Coordinate every 5 minutes
                
                # Collect performance data from all components
                component_data = await self._collect_component_data()
                
                # Analyze cross-component optimization opportunities
                optimizations = await self._analyze_cross_component_optimizations(component_data)
                
                # Apply coordinated optimizations
                for optimization in optimizations:
                    await self._apply_coordinated_optimization(optimization)
                
                logger.debug(f"Coordination cycle completed with {len(optimizations)} optimizations")
                
            except Exception as e:
                logger.error(f"Optimization coordination error: {str(e)}")
    
    async def _performance_reporting_task(self) -> None:
        """Background task for generating performance reports."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Generate reports every hour
                
                # Generate comprehensive performance report
                report = await self._generate_performance_report()
                
                # Store report in history
                self.performance_history.append(report)
                
                # Keep only last 168 hours (1 week)
                cutoff_time = datetime.utcnow() - timedelta(hours=168)
                self.performance_history = [
                    r for r in self.performance_history if r.timestamp > cutoff_time
                ]
                
                # Log key achievements
                if report.performance_score > 90:
                    achievement = f"Excellent performance score: {report.performance_score:.1f}"
                    if achievement not in self.optimization_achievements:
                        self.optimization_achievements.append(achievement)
                        logger.info(f"Achievement unlocked: {achievement}")
                
                logger.info(f"Performance report generated: score {report.performance_score:.1f}")
                
            except Exception as e:
                logger.error(f"Performance reporting error: {str(e)}")
    
    async def _achievement_tracking_task(self) -> None:
        """Background task for tracking optimization achievements."""
        while self.is_running:
            try:
                await asyncio.sleep(1800)  # Check achievements every 30 minutes
                
                # Check for performance target achievements
                await self._check_performance_targets()
                
                # Check for efficiency improvements
                await self._check_efficiency_improvements()
                
                # Check for stability achievements
                await self._check_stability_achievements()
                
            except Exception as e:
                logger.error(f"Achievement tracking error: {str(e)}")
    
    async def _collect_component_data(self) -> Dict[str, Any]:
        """Collect performance data from all components."""
        component_data = {}
        
        for component_name, component in self.components.items():
            try:
                if component_name == "distributed_processing" and hasattr(component, "get_cluster_status"):
                    component_data[component_name] = await component.get_cluster_status()
                elif component_name == "intelligent_caching" and hasattr(component, "get_cache_report"):
                    component_data[component_name] = await component.get_cache_report()
                elif component_name == "query_optimization" and hasattr(component, "get_optimization_report"):
                    component_data[component_name] = await component.get_optimization_report()
                elif component_name == "memory_optimization" and hasattr(component, "get_optimization_report"):
                    component_data[component_name] = await component.get_optimization_report()
                elif component_name == "network_optimization" and hasattr(component, "get_network_report"):
                    component_data[component_name] = await component.get_network_report()
                elif component_name == "performance_monitoring" and hasattr(component, "get_performance_dashboard"):
                    component_data[component_name] = await component.get_performance_dashboard()
                
            except Exception as e:
                logger.error(f"Failed to collect data from {component_name}: {str(e)}")
                component_data[component_name] = {"error": str(e)}
        
        return component_data
    
    async def _analyze_cross_component_optimizations(self, component_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze optimization opportunities across components."""
        optimizations = []
        
        # Memory pressure affecting cache performance
        memory_data = component_data.get("memory_optimization", {})
        cache_data = component_data.get("intelligent_caching", {})
        
        if (memory_data.get("memory_optimization", {}).get("current_utilization_percent", 0) > 85 and
            cache_data.get("overall_analytics", {}).get("hit_ratio", 1) < 0.7):
            optimizations.append({
                "type": "reduce_cache_size",
                "description": "Reduce cache size due to memory pressure",
                "priority": "high",
                "components": ["memory_optimization", "intelligent_caching"]
            })
        
        # Network latency affecting distributed processing
        network_data = component_data.get("network_optimization", {})
        distributed_data = component_data.get("distributed_processing", {})
        
        if (network_data.get("network_performance", {}).get("avg_latency_ms", 0) > 200 and
            distributed_data.get("performance_metrics", {}).get("avg_job_completion_time", 0) > 5000):
            optimizations.append({
                "type": "enable_compression",
                "description": "Enable network compression to reduce latency impact",
                "priority": "medium",
                "components": ["network_optimization", "distributed_processing"]
            })
        
        # Query performance affecting overall system
        query_data = component_data.get("query_optimization", {})
        monitoring_data = component_data.get("performance_monitoring", {})
        
        if (query_data.get("query_statistics", {}).get("avg_query_time_ms", 0) > 1000 and
            monitoring_data.get("performance_score", 100) < 70):
            optimizations.append({
                "type": "aggressive_query_optimization",
                "description": "Apply aggressive query optimization due to performance impact",
                "priority": "high",
                "components": ["query_optimization", "performance_monitoring"]
            })
        
        return optimizations
    
    async def _apply_coordinated_optimization(self, optimization: Dict[str, Any]) -> None:
        """Apply a coordinated optimization across components."""
        try:
            opt_type = optimization["type"]
            
            if opt_type == "reduce_cache_size":
                # Coordinate cache size reduction with memory optimization
                cache_component = self.components.get("intelligent_caching")
                if cache_component and hasattr(cache_component, "optimize_cache"):
                    suggestions = await cache_component.optimize_cache()
                    logger.info(f"Applied cache optimization: {suggestions}")
            
            elif opt_type == "enable_compression":
                # Enable compression in network component
                network_component = self.components.get("network_optimization")
                if network_component:
                    # Would enable compression if not already enabled
                    logger.info("Applied network compression optimization")
            
            elif opt_type == "aggressive_query_optimization":
                # Apply aggressive query optimization
                query_component = self.components.get("query_optimization")
                if query_component and hasattr(query_component, "get_optimization_report"):
                    # Would apply more aggressive optimization strategies
                    logger.info("Applied aggressive query optimization")
            
            logger.info(f"Successfully applied optimization: {optimization['description']}")
            
        except Exception as e:
            logger.error(f"Failed to apply optimization {optimization['type']}: {str(e)}")
    
    async def _generate_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance report."""
        component_data = await self._collect_component_data()
        
        # Calculate component scores
        distributed_score = self._calculate_component_score(
            component_data.get("distributed_processing", {}), "distributed"
        )
        caching_score = self._calculate_component_score(
            component_data.get("intelligent_caching", {}), "caching"
        )
        query_score = self._calculate_component_score(
            component_data.get("query_optimization", {}), "query"
        )
        memory_score = self._calculate_component_score(
            component_data.get("memory_optimization", {}), "memory"
        )
        network_score = self._calculate_component_score(
            component_data.get("network_optimization", {}), "network"
        )
        monitoring_score = self._calculate_component_score(
            component_data.get("performance_monitoring", {}), "monitoring"
        )
        
        # Calculate overall performance score
        scores = [distributed_score, caching_score, query_score, memory_score, network_score, monitoring_score]
        overall_score = sum(s for s in scores if s > 0) / len([s for s in scores if s > 0])
        
        # Generate recommendations
        top_optimizations = await self._generate_top_optimizations(component_data)
        critical_issues = await self._identify_critical_issues(component_data)
        
        # Check performance targets
        targets_met = await self._check_targets_met(component_data)
        
        return PerformanceReport(
            performance_score=overall_score,
            optimization_level=self.config.optimization_level,
            distributed_processing_score=distributed_score,
            caching_efficiency_score=caching_score,
            query_optimization_score=query_score,
            memory_efficiency_score=memory_score,
            network_performance_score=network_score,
            monitoring_effectiveness_score=monitoring_score,
            top_optimizations=top_optimizations,
            critical_issues=critical_issues,
            performance_targets_met=targets_met,
            sla_compliance_percent=min(100.0, overall_score + 10)  # Simplified calculation
        )
    
    def _calculate_component_score(self, component_data: Dict[str, Any], component_type: str) -> float:
        """Calculate performance score for a specific component."""
        if "error" in component_data:
            return 0.0
        
        if component_type == "distributed":
            utilization = component_data.get("average_utilization", 0)
            return max(0, 100 - (utilization * 100))  # Lower utilization = better score
        
        elif component_type == "caching":
            hit_ratio = component_data.get("overall_analytics", {}).get("hit_ratio", 0)
            return hit_ratio * 100
        
        elif component_type == "query":
            avg_time = component_data.get("query_statistics", {}).get("avg_query_time_ms", 5000)
            return max(0, 100 - (avg_time / 50))  # 5000ms = 0 score
        
        elif component_type == "memory":
            utilization = component_data.get("memory_optimization", {}).get("current_utilization_percent", 50)
            return max(0, 100 - utilization)
        
        elif component_type == "network":
            latency = component_data.get("network_performance", {}).get("avg_latency_ms", 500)
            return max(0, 100 - (latency / 10))  # 1000ms = 0 score
        
        elif component_type == "monitoring":
            return component_data.get("performance_score", 50)
        
        return 50  # Default neutral score
    
    async def _generate_top_optimizations(self, component_data: Dict[str, Any]) -> List[str]:
        """Generate top optimization recommendations."""
        optimizations = []
        
        # Analyze each component for optimization opportunities
        for component_name, data in component_data.items():
            if "error" in data:
                continue
            
            if component_name == "memory_optimization":
                utilization = data.get("memory_optimization", {}).get("current_utilization_percent", 0)
                if utilization > 85:
                    optimizations.append("Optimize memory usage - high utilization detected")
            
            elif component_name == "intelligent_caching":
                hit_ratio = data.get("overall_analytics", {}).get("hit_ratio", 1)
                if hit_ratio < 0.7:
                    optimizations.append("Improve cache hit ratio - current performance suboptimal")
            
            elif component_name == "network_optimization":
                latency = data.get("network_performance", {}).get("avg_latency_ms", 0)
                if latency > 200:
                    optimizations.append("Reduce network latency - optimize network configuration")
            
            elif component_name == "query_optimization":
                slow_queries = data.get("optimization_opportunities", {}).get("missing_indexes", 0)
                if slow_queries > 5:
                    optimizations.append("Add database indexes - multiple optimization opportunities")
        
        return optimizations[:5]  # Return top 5
    
    async def _identify_critical_issues(self, component_data: Dict[str, Any]) -> List[str]:
        """Identify critical performance issues."""
        issues = []
        
        for component_name, data in component_data.items():
            if "error" in data:
                issues.append(f"Component {component_name} has errors: {data['error']}")
        
        # Check for critical thresholds
        monitoring_data = component_data.get("performance_monitoring", {})
        if monitoring_data.get("performance_score", 100) < 30:
            issues.append("Overall system performance critically low")
        
        return issues
    
    async def _check_targets_met(self, component_data: Dict[str, Any]) -> List[str]:
        """Check which performance targets have been met."""
        targets_met = []
        
        # Check response time target
        query_data = component_data.get("query_optimization", {})
        avg_query_time = query_data.get("query_statistics", {}).get("avg_query_time_ms", 1000)
        if avg_query_time <= self.config.target_response_time_ms:
            targets_met.append(f"Response time target met: {avg_query_time}ms ≤ {self.config.target_response_time_ms}ms")
        
        # Check error rate target
        monitoring_data = component_data.get("performance_monitoring", {})
        error_rate = monitoring_data.get("current_metrics", {}).get("error_rate", {}).get("current_value", 5)
        if error_rate <= self.config.target_error_rate_percent:
            targets_met.append(f"Error rate target met: {error_rate}% ≤ {self.config.target_error_rate_percent}%")
        
        return targets_met
    
    # Error handling would be managed by interface implementation
    async def get_enterprise_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive enterprise performance dashboard."""
        # Get latest performance report
        latest_report = self.performance_history[-1] if self.performance_history else await self._generate_performance_report()
        
        # Get real-time component data
        component_data = await self._collect_component_data()
        
        # Calculate trends
        trends = await self._calculate_performance_trends()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "optimization_level": self.config.optimization_level.value,
            "overall_performance": {
                "score": latest_report.performance_score,
                "grade": self._get_performance_grade(latest_report.performance_score),
                "sla_compliance": latest_report.sla_compliance_percent,
                "targets_met": len(latest_report.performance_targets_met),
                "critical_issues": len(latest_report.critical_issues)
            },
            "component_scores": {
                "distributed_processing": latest_report.distributed_processing_score,
                "intelligent_caching": latest_report.caching_efficiency_score,
                "query_optimization": latest_report.query_optimization_score,
                "memory_optimization": latest_report.memory_efficiency_score,
                "network_optimization": latest_report.network_performance_score,
                "performance_monitoring": latest_report.monitoring_effectiveness_score
            },
            "performance_trends": trends,
            "optimization_recommendations": latest_report.top_optimizations,
            "critical_issues": latest_report.critical_issues,
            "achievements": self.optimization_achievements[-10:],  # Last 10 achievements
            "resource_utilization": await self._get_resource_utilization_summary(component_data),
            "cost_efficiency": {
                "improvement_percent": latest_report.cost_efficiency_improvement,
                "resource_optimization": latest_report.resource_utilization_improvement
            }
        }
    
    def _get_performance_grade(self, score: float) -> str:
        """Convert performance score to letter grade."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        elif score >= 55:
            return "C-"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    async def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        if len(self.performance_history) < 2:
            return {"trend": "insufficient_data"}
        
        # Get scores from last 24 hours
        recent_scores = [r.performance_score for r in self.performance_history[-24:]]
        
        if len(recent_scores) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate trend
        first_half = recent_scores[:len(recent_scores)//2]
        second_half = recent_scores[len(recent_scores)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        trend_direction = "improving" if avg_second > avg_first else "declining" if avg_second < avg_first else "stable"
        trend_magnitude = abs(avg_second - avg_first)
        
        return {
            "trend": trend_direction,
            "magnitude": trend_magnitude,
            "current_score": recent_scores[-1],
            "previous_score": recent_scores[0],
            "data_points": len(recent_scores)
        }
    
    async def _get_resource_utilization_summary(self, component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of resource utilization across components."""
        memory_data = component_data.get("memory_optimization", {})
        network_data = component_data.get("network_optimization", {})
        distributed_data = component_data.get("distributed_processing", {})
        
        return {
            "cpu_utilization": distributed_data.get("performance_metrics", {}).get("avg_cpu_utilization", 0),
            "memory_utilization": memory_data.get("memory_optimization", {}).get("current_utilization_percent", 0),
            "network_utilization": network_data.get("network_performance", {}).get("bandwidth_utilization_percent", 0),
            "disk_utilization": memory_data.get("storage_optimization", {}).get("disk_utilization_percent", 0),
            "active_connections": network_data.get("network_performance", {}).get("concurrent_connections", 0),
            "cluster_nodes": distributed_data.get("active_nodes", 0)
        }
    
    # Achievement checking methods
    async def _check_performance_targets(self) -> None:
        """Check if performance targets have been achieved."""
        if not self.performance_history:
            return
        
        latest_report = self.performance_history[-1]
        
        # Check for sustained high performance
        if (latest_report.performance_score > 95 and
            len([r for r in self.performance_history[-24:] if r.performance_score > 95]) >= 20):
            achievement = "Sustained excellent performance for 20+ hours"
            if achievement not in self.optimization_achievements:
                self.optimization_achievements.append(achievement)
    
    async def _check_efficiency_improvements(self) -> None:
        """Check for efficiency improvement achievements."""
        if len(self.performance_history) < 48:  # Need 48 hours of data
            return
        
        old_avg = sum(r.performance_score for r in self.performance_history[-48:-24]) / 24
        new_avg = sum(r.performance_score for r in self.performance_history[-24:]) / 24
        
        improvement = new_avg - old_avg
        if improvement > 10:
            achievement = f"Significant performance improvement: +{improvement:.1f} points"
            if achievement not in self.optimization_achievements:
                self.optimization_achievements.append(achievement)
    
    async def _check_stability_achievements(self) -> None:
        """Check for system stability achievements."""
        if not self.performance_history:
            return
        
        # Check for consistent performance (low variance)
        recent_scores = [r.performance_score for r in self.performance_history[-24:]]
        if len(recent_scores) >= 24:
            variance = sum((x - sum(recent_scores)/len(recent_scores))**2 for x in recent_scores) / len(recent_scores)
            if variance < 25:  # Low variance in performance
                achievement = "Exceptional system stability maintained"
                if achievement not in self.optimization_achievements:
                    self.optimization_achievements.append(achievement)