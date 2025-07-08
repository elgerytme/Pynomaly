"""Resource optimization system for dynamic allocation and cost optimization."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from uuid import UUID, uuid4

import numpy as np

logger = logging.getLogger(__name__)

class ResourceType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"
    POWER = "power"

class OptimizationObjective(str, Enum):
    COST = "cost"
    PERFORMANCE = "performance"
    ENERGY_EFFICIENCY = "energy_efficiency"
    CARBON_FOOTPRINT = "carbon_footprint"
    BALANCED = "balanced"

class ScalingPolicy(str, Enum):
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    PROACTIVE = "proactive"
    INTELLIGENT = "intelligent"

class ResourceProvider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"

@dataclass
class ResourceSpec:
    """Specification for a resource requirement."""
    resource_type: ResourceType
    amount: float
    unit: str
    priority: int = 1

    # Time constraints
    required_duration_hours: float = 1.0
    max_startup_time_minutes: float = 5.0

    # Cost constraints
    max_cost_per_hour: float = float('inf')
    budget_limit: float = float('inf')

    # Performance constraints
    min_performance_score: float = 0.7
    max_latency_ms: float = 1000.0

    # Optimization preferences
    prefer_spot_instances: bool = False
    allow_preemption: bool = False
    carbon_aware: bool = False

@dataclass
class ResourceInstance:
    """Represents an allocated resource instance."""
    instance_id: str
    provider: ResourceProvider
    resource_type: ResourceType
    capacity: float
    allocated: float

    # Cost information
    cost_per_hour: float
    spot_price: Optional[float] = None
    reserved_instance: bool = False

    # Performance characteristics
    performance_score: float = 1.0
    latency_ms: float = 100.0
    throughput_ops_per_sec: float = 1000.0

    # Energy and carbon information
    power_consumption_watts: float = 100.0
    carbon_intensity_g_co2_per_kwh: float = 400.0

    # Status information
    status: str = "available"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    utilization: float = 0.0

    def get_available_capacity(self) -> float:
        """Get available capacity."""
        return self.capacity - self.allocated

    def get_utilization_percent(self) -> float:
        """Get utilization percentage."""
        return (self.allocated / self.capacity) * 100 if self.capacity > 0 else 0

    def get_cost_per_unit_hour(self) -> float:
        """Get cost per unit per hour."""
        return self.cost_per_hour / self.capacity if self.capacity > 0 else 0

    def calculate_carbon_footprint(self, usage_hours: float) -> float:
        """Calculate carbon footprint for usage period."""
        energy_kwh = (self.power_consumption_watts * usage_hours) / 1000
        return energy_kwh * self.carbon_intensity_g_co2_per_kwh

@dataclass
class OptimizationResult:
    """Result of resource optimization."""
    optimization_id: UUID
    objective: OptimizationObjective
    timestamp: datetime

    # Resource allocation
    allocated_instances: List[ResourceInstance]
    total_cost: float
    total_performance_score: float
    total_carbon_footprint_g: float

    # Optimization metrics
    cost_savings_percent: float = 0.0
    performance_improvement_percent: float = 0.0
    carbon_reduction_percent: float = 0.0

    # Implementation details
    scaling_actions: List[Dict[str, Any]] = field(default_factory=list)
    migration_required: bool = False
    estimated_migration_time_minutes: float = 0.0

    # Confidence and risk
    confidence_score: float = 0.8
    risk_assessment: str = "low"

    def get_total_capacity(self, resource_type: ResourceType) -> float:
        """Get total capacity for a resource type."""
        return sum(
            instance.capacity for instance in self.allocated_instances
            if instance.resource_type == resource_type
        )

class DynamicResourceAllocator:
    """Dynamically allocates and deallocates resources based on demand."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaling_policy = ScalingPolicy(config.get("scaling_policy", "intelligent"))

        # Resource pools
        self.available_instances: Dict[ResourceProvider, List[ResourceInstance]] = {}
        self.allocated_instances: Dict[str, ResourceInstance] = {}

        # Demand forecasting
        self.demand_predictor = DemandPredictor(config.get("demand_prediction", {}))
        self.load_monitor = LoadMonitor(config.get("load_monitoring", {}))

        # Optimization components
        self.cost_optimizer = CostOptimizer(config.get("cost_optimization", {}))
        self.performance_optimizer = PerformanceOptimizer(config.get("performance_optimization", {}))

        # Statistics
        self.allocation_history: List[Dict[str, Any]] = []
        self.optimization_metrics = {
            "total_allocations": 0,
            "successful_allocations": 0,
            "cost_savings": 0.0,
            "performance_improvements": 0.0,
        }

    async def initialize(self) -> bool:
        """Initialize the dynamic resource allocator."""
        try:
            logger.info("Initializing dynamic resource allocator")

            # Initialize resource pools
            await self._initialize_resource_pools()

            # Start monitoring and prediction
            await self.load_monitor.start_monitoring()
            await self.demand_predictor.start_prediction()

            # Start allocation loop
            asyncio.create_task(self._allocation_loop())

            logger.info("Dynamic resource allocator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize dynamic resource allocator: {e}")
            return False

    async def _initialize_resource_pools(self) -> None:
        """Initialize available resource pools."""
        # Simulate available resources from different providers
        providers_config = {
            ResourceProvider.AWS: {"cpu_instances": 100, "gpu_instances": 20, "cost_multiplier": 1.0},
            ResourceProvider.AZURE: {"cpu_instances": 80, "gpu_instances": 15, "cost_multiplier": 0.95},
            ResourceProvider.GCP: {"cpu_instances": 90, "gpu_instances": 18, "cost_multiplier": 0.98},
            ResourceProvider.ON_PREMISE: {"cpu_instances": 50, "gpu_instances": 10, "cost_multiplier": 0.7},
        }

        for provider, config in providers_config.items():
            self.available_instances[provider] = []

            # Create CPU instances
            for i in range(config["cpu_instances"]):
                instance = ResourceInstance(
                    instance_id=f"{provider.value}-cpu-{i}",
                    provider=provider,
                    resource_type=ResourceType.CPU,
                    capacity=np.random.uniform(4, 32),  # 4-32 cores
                    allocated=0.0,
                    cost_per_hour=np.random.uniform(0.1, 2.0) * config["cost_multiplier"],
                    performance_score=np.random.uniform(0.7, 1.0),
                    power_consumption_watts=np.random.uniform(100, 400),
                    carbon_intensity_g_co2_per_kwh=np.random.uniform(200, 600),
                )
                self.available_instances[provider].append(instance)

            # Create GPU instances
            for i in range(config["gpu_instances"]):
                instance = ResourceInstance(
                    instance_id=f"{provider.value}-gpu-{i}",
                    provider=provider,
                    resource_type=ResourceType.GPU,
                    capacity=np.random.uniform(1, 8),  # 1-8 GPUs
                    allocated=0.0,
                    cost_per_hour=np.random.uniform(2.0, 10.0) * config["cost_multiplier"],
                    performance_score=np.random.uniform(0.8, 1.0),
                    power_consumption_watts=np.random.uniform(250, 400),
                    carbon_intensity_g_co2_per_kwh=np.random.uniform(200, 600),
                )
                self.available_instances[provider].append(instance)

    async def allocate_resources(self, requirements: List[ResourceSpec], objective: OptimizationObjective) -> OptimizationResult:
        """Allocate resources based on requirements and optimization objective."""
        try:
            start_time = datetime.utcnow()

            # Find optimal allocation
            if objective == OptimizationObjective.COST:
                allocation = await self.cost_optimizer.optimize_for_cost(requirements, self.available_instances)
            elif objective == OptimizationObjective.PERFORMANCE:
                allocation = await self.performance_optimizer.optimize_for_performance(requirements, self.available_instances)
            elif objective == OptimizationObjective.ENERGY_EFFICIENCY:
                allocation = await self._optimize_for_energy_efficiency(requirements)
            elif objective == OptimizationObjective.CARBON_FOOTPRINT:
                allocation = await self._optimize_for_carbon_footprint(requirements)
            else:  # BALANCED
                allocation = await self._optimize_balanced(requirements)

            # Create optimization result
            result = OptimizationResult(
                optimization_id=uuid4(),
                objective=objective,
                timestamp=start_time,
                allocated_instances=allocation["instances"],
                total_cost=allocation["total_cost"],
                total_performance_score=allocation["total_performance"],
                total_carbon_footprint_g=allocation["total_carbon"],
                confidence_score=allocation.get("confidence", 0.8),
            )

            # Update allocation tracking
            for instance in result.allocated_instances:
                self.allocated_instances[instance.instance_id] = instance

            # Update metrics
            self.optimization_metrics["total_allocations"] += 1
            self.optimization_metrics["successful_allocations"] += 1

            # Log allocation event
            self.allocation_history.append({
                "timestamp": start_time,
                "objective": objective.value,
                "requirements": len(requirements),
                "allocated_instances": len(result.allocated_instances),
                "total_cost": result.total_cost,
            })

            return result

        except Exception as e:
            logger.error(f"Resource allocation failed: {e}")
            self.optimization_metrics["total_allocations"] += 1
            raise

    async def _optimize_for_energy_efficiency(self, requirements: List[ResourceSpec]) -> Dict[str, Any]:
        """Optimize allocation for energy efficiency."""
        best_allocation = {"instances": [], "total_cost": 0, "total_performance": 0, "total_carbon": 0}
        best_efficiency = 0

        # Try different combinations of instances
        for provider in self.available_instances:
            allocation = await self._try_provider_allocation(requirements, provider)

            if allocation:
                # Calculate energy efficiency (performance per watt)
                total_power = sum(inst.power_consumption_watts for inst in allocation["instances"])
                efficiency = allocation["total_performance"] / max(total_power, 1)

                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_allocation = allocation

        return best_allocation

    async def _optimize_for_carbon_footprint(self, requirements: List[ResourceSpec]) -> Dict[str, Any]:
        """Optimize allocation for minimal carbon footprint."""
        best_allocation = {"instances": [], "total_cost": 0, "total_performance": 0, "total_carbon": float('inf')}

        # Try different combinations of instances
        for provider in self.available_instances:
            allocation = await self._try_provider_allocation(requirements, provider)

            if allocation and allocation["total_carbon"] < best_allocation["total_carbon"]:
                best_allocation = allocation

        return best_allocation

    async def _optimize_balanced(self, requirements: List[ResourceSpec]) -> Dict[str, Any]:
        """Optimize allocation with balanced objectives."""
        best_allocation = {"instances": [], "total_cost": 0, "total_performance": 0, "total_carbon": 0}
        best_score = -1

        # Try different combinations of instances
        for provider in self.available_instances:
            allocation = await self._try_provider_allocation(requirements, provider)

            if allocation:
                # Calculate balanced score (normalize and weight different factors)
                cost_score = 1.0 / (1.0 + allocation["total_cost"] / 100)  # Lower cost is better
                performance_score = allocation["total_performance"]  # Higher performance is better
                carbon_score = 1.0 / (1.0 + allocation["total_carbon"] / 1000)  # Lower carbon is better

                # Weighted combination
                balanced_score = (
                    cost_score * 0.4 +
                    performance_score * 0.4 +
                    carbon_score * 0.2
                )

                if balanced_score > best_score:
                    best_score = balanced_score
                    best_allocation = allocation

        return best_allocation

    async def _try_provider_allocation(self, requirements: List[ResourceSpec], provider: ResourceProvider) -> Optional[Dict[str, Any]]:
        """Try to allocate resources from a specific provider."""
        try:
            available = self.available_instances[provider]
            allocated_instances = []
            total_cost = 0
            total_performance = 0
            total_carbon = 0

            for req in requirements:
                # Find best matching instance
                suitable_instances = [
                    inst for inst in available
                    if (inst.resource_type == req.resource_type and
                        inst.get_available_capacity() >= req.amount and
                        inst.cost_per_hour <= req.max_cost_per_hour and
                        inst.performance_score >= req.min_performance_score)
                ]

                if not suitable_instances:
                    return None  # Cannot satisfy requirement

                # Select best instance (lowest cost per unit)
                best_instance = min(suitable_instances, key=lambda x: x.get_cost_per_unit_hour())

                # Allocate capacity
                best_instance.allocated += req.amount
                allocated_instances.append(best_instance)

                # Calculate costs and metrics
                instance_cost = best_instance.cost_per_hour * req.required_duration_hours
                total_cost += instance_cost
                total_performance += best_instance.performance_score * req.amount
                total_carbon += best_instance.calculate_carbon_footprint(req.required_duration_hours)

            return {
                "instances": allocated_instances,
                "total_cost": total_cost,
                "total_performance": total_performance / len(requirements),  # Average performance
                "total_carbon": total_carbon,
                "confidence": 0.8,
            }

        except Exception as e:
            logger.error(f"Provider allocation failed for {provider}: {e}")
            return None

    async def _allocation_loop(self) -> None:
        """Main allocation loop for dynamic scaling."""
        while True:
            try:
                # Check for scaling opportunities
                current_load = await self.load_monitor.get_current_load()
                predicted_demand = await self.demand_predictor.predict_demand(horizon_minutes=30)

                # Determine if scaling is needed
                scaling_decision = await self._evaluate_scaling_decision(current_load, predicted_demand)

                if scaling_decision["action"] != "none":
                    await self._execute_scaling_action(scaling_decision)

                # Sleep until next evaluation
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Allocation loop error: {e}")
                await asyncio.sleep(60)

    async def _evaluate_scaling_decision(self, current_load: Dict[str, float], predicted_demand: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate whether scaling action is needed."""
        if self.scaling_policy == ScalingPolicy.REACTIVE:
            # React to current load
            if current_load.get("cpu_utilization", 0) > 80:
                return {"action": "scale_up", "resource_type": "cpu", "factor": 1.5}
            elif current_load.get("cpu_utilization", 0) < 20:
                return {"action": "scale_down", "resource_type": "cpu", "factor": 0.7}

        elif self.scaling_policy == ScalingPolicy.PREDICTIVE:
            # React to predicted demand
            if predicted_demand.get("cpu_demand", 0) > current_load.get("cpu_capacity", 0) * 0.8:
                return {"action": "scale_up", "resource_type": "cpu", "factor": 1.3}

        elif self.scaling_policy == ScalingPolicy.INTELLIGENT:
            # Use ML-based decision making
            decision = await self._intelligent_scaling_decision(current_load, predicted_demand)
            return decision

        return {"action": "none"}

    async def _intelligent_scaling_decision(self, current_load: Dict[str, float], predicted_demand: Dict[str, float]) -> Dict[str, Any]:
        """Make intelligent scaling decision using ML."""
        # Simplified intelligent decision making
        # In practice, this would use sophisticated ML models

        load_trend = predicted_demand.get("cpu_demand", 0) - current_load.get("cpu_utilization", 0)
        cost_sensitivity = self.config.get("cost_sensitivity", 0.5)
        performance_sensitivity = self.config.get("performance_sensitivity", 0.5)

        if load_trend > 20 and performance_sensitivity > 0.7:
            return {"action": "scale_up", "resource_type": "cpu", "factor": 1.4}
        elif load_trend < -30 and cost_sensitivity > 0.7:
            return {"action": "scale_down", "resource_type": "cpu", "factor": 0.8}

        return {"action": "none"}

    async def _execute_scaling_action(self, scaling_decision: Dict[str, Any]) -> None:
        """Execute a scaling action."""
        try:
            action = scaling_decision["action"]
            resource_type = scaling_decision.get("resource_type", "cpu")
            factor = scaling_decision.get("factor", 1.0)

            logger.info(f"Executing scaling action: {action} for {resource_type} with factor {factor}")

            if action == "scale_up":
                await self._scale_up_resources(resource_type, factor)
            elif action == "scale_down":
                await self._scale_down_resources(resource_type, factor)

        except Exception as e:
            logger.error(f"Failed to execute scaling action: {e}")

    async def _scale_up_resources(self, resource_type: str, factor: float) -> None:
        """Scale up resources."""
        # Implementation would add new instances or increase capacity
        logger.info(f"Scaling up {resource_type} by factor {factor}")

    async def _scale_down_resources(self, resource_type: str, factor: float) -> None:
        """Scale down resources."""
        # Implementation would remove instances or decrease capacity
        logger.info(f"Scaling down {resource_type} by factor {factor}")

    async def get_allocation_status(self) -> Dict[str, Any]:
        """Get current allocation status."""
        total_instances = sum(len(instances) for instances in self.available_instances.values())
        allocated_instances_count = len(self.allocated_instances)

        total_cost = sum(inst.cost_per_hour for inst in self.allocated_instances.values())
        total_capacity = {}
        total_allocated = {}

        for instance in self.allocated_instances.values():
            resource_type = instance.resource_type.value
            total_capacity[resource_type] = total_capacity.get(resource_type, 0) + instance.capacity
            total_allocated[resource_type] = total_allocated.get(resource_type, 0) + instance.allocated

        return {
            "total_available_instances": total_instances,
            "allocated_instances": allocated_instances_count,
            "utilization_percent": (allocated_instances_count / max(total_instances, 1)) * 100,
            "total_cost_per_hour": total_cost,
            "capacity_by_type": total_capacity,
            "allocated_by_type": total_allocated,
            "optimization_metrics": self.optimization_metrics,
            "recent_allocations": self.allocation_history[-10:],
        }

class DemandPredictor:
    """Predicts future resource demand."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prediction_window_minutes = config.get("prediction_window", 60)
        self.historical_data: List[Dict[str, Any]] = []
        self.prediction_active = False

    async def start_prediction(self) -> None:
        """Start demand prediction."""
        self.prediction_active = True
        asyncio.create_task(self._prediction_loop())

    async def _prediction_loop(self) -> None:
        """Main prediction loop."""
        while self.prediction_active:
            try:
                # Collect current metrics
                current_metrics = await self._collect_current_metrics()
                self.historical_data.append(current_metrics)

                # Keep only recent history
                if len(self.historical_data) > 1440:  # 24 hours of minute data
                    self.historical_data = self.historical_data[-1440:]

                await asyncio.sleep(60)  # Collect every minute

            except Exception as e:
                logger.error(f"Demand prediction loop error: {e}")
                await asyncio.sleep(60)

    async def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        return {
            "timestamp": datetime.utcnow(),
            "cpu_utilization": np.random.uniform(20, 80),
            "memory_utilization": np.random.uniform(30, 70),
            "gpu_utilization": np.random.uniform(10, 90),
            "network_utilization": np.random.uniform(5, 50),
            "request_rate": np.random.uniform(100, 1000),
        }

    async def predict_demand(self, horizon_minutes: int = 30) -> Dict[str, float]:
        """Predict resource demand for the given horizon."""
        if len(self.historical_data) < 10:
            # Not enough data for prediction
            return {"cpu_demand": 50, "memory_demand": 50, "gpu_demand": 50}

        # Simple linear trend prediction
        recent_data = self.historical_data[-10:]

        # Calculate trends
        cpu_trend = np.polyfit(
            range(len(recent_data)),
            [d["cpu_utilization"] for d in recent_data],
            1
        )[0]

        # Project trend forward
        current_cpu = recent_data[-1]["cpu_utilization"]
        predicted_cpu = current_cpu + (cpu_trend * horizon_minutes)

        # Similar for other resources
        predicted_memory = recent_data[-1]["memory_utilization"] + np.random.uniform(-5, 5)
        predicted_gpu = recent_data[-1]["gpu_utilization"] + np.random.uniform(-10, 10)

        return {
            "cpu_demand": max(0, min(100, predicted_cpu)),
            "memory_demand": max(0, min(100, predicted_memory)),
            "gpu_demand": max(0, min(100, predicted_gpu)),
        }

class LoadMonitor:
    """Monitors current system load."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_active = False
        self.current_load = {}

    async def start_monitoring(self) -> None:
        """Start load monitoring."""
        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.current_load = await self._collect_load_metrics()
                await asyncio.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                logger.error(f"Load monitoring loop error: {e}")
                await asyncio.sleep(5)

    async def _collect_load_metrics(self) -> Dict[str, float]:
        """Collect current load metrics."""
        return {
            "cpu_utilization": np.random.uniform(20, 80),
            "cpu_capacity": 100.0,
            "memory_utilization": np.random.uniform(30, 70),
            "memory_capacity": 100.0,
            "gpu_utilization": np.random.uniform(10, 90),
            "gpu_capacity": 100.0,
            "active_connections": np.random.randint(50, 500),
            "request_rate": np.random.uniform(100, 1000),
        }

    async def get_current_load(self) -> Dict[str, float]:
        """Get current system load."""
        return self.current_load

class CostOptimizer:
    """Optimizes resource allocation for cost efficiency."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spot_instance_preference = config.get("spot_instance_preference", 0.8)
        self.reserved_instance_utilization = config.get("reserved_instance_utilization", 0.9)

    async def optimize_for_cost(self, requirements: List[ResourceSpec], available_instances: Dict[ResourceProvider, List[ResourceInstance]]) -> Dict[str, Any]:
        """Optimize allocation for minimum cost."""
        best_allocation = {"instances": [], "total_cost": float('inf'), "total_performance": 0, "total_carbon": 0}

        # Try different provider combinations
        for provider in available_instances:
            allocation = await self._try_cost_optimized_allocation(requirements, provider, available_instances[provider])

            if allocation and allocation["total_cost"] < best_allocation["total_cost"]:
                best_allocation = allocation

        return best_allocation

    async def _try_cost_optimized_allocation(self, requirements: List[ResourceSpec], provider: ResourceProvider, instances: List[ResourceInstance]) -> Optional[Dict[str, Any]]:
        """Try cost-optimized allocation for a provider."""
        try:
            # Sort instances by cost per unit (ascending)
            sorted_instances = sorted(instances, key=lambda x: x.get_cost_per_unit_hour())

            allocated_instances = []
            total_cost = 0
            total_performance = 0
            total_carbon = 0

            for req in requirements:
                # Find cheapest suitable instance
                suitable_instances = [
                    inst for inst in sorted_instances
                    if (inst.resource_type == req.resource_type and
                        inst.get_available_capacity() >= req.amount and
                        inst.cost_per_hour <= req.max_cost_per_hour)
                ]

                if not suitable_instances:
                    return None

                # Select cheapest instance
                cheapest_instance = suitable_instances[0]

                # Consider spot instances for additional savings
                if req.prefer_spot_instances and cheapest_instance.spot_price:
                    actual_cost = cheapest_instance.spot_price
                else:
                    actual_cost = cheapest_instance.cost_per_hour

                # Allocate capacity
                cheapest_instance.allocated += req.amount
                allocated_instances.append(cheapest_instance)

                # Calculate totals
                instance_cost = actual_cost * req.required_duration_hours
                total_cost += instance_cost
                total_performance += cheapest_instance.performance_score * req.amount
                total_carbon += cheapest_instance.calculate_carbon_footprint(req.required_duration_hours)

            return {
                "instances": allocated_instances,
                "total_cost": total_cost,
                "total_performance": total_performance / len(requirements),
                "total_carbon": total_carbon,
                "confidence": 0.9,
            }

        except Exception as e:
            logger.error(f"Cost optimization failed for {provider}: {e}")
            return None

class PerformanceOptimizer:
    """Optimizes resource allocation for maximum performance."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_weight = config.get("performance_weight", 1.0)
        self.latency_weight = config.get("latency_weight", 0.5)

    async def optimize_for_performance(self, requirements: List[ResourceSpec], available_instances: Dict[ResourceProvider, List[ResourceInstance]]) -> Dict[str, Any]:
        """Optimize allocation for maximum performance."""
        best_allocation = {"instances": [], "total_cost": 0, "total_performance": -1, "total_carbon": 0}

        # Try different provider combinations
        for provider in available_instances:
            allocation = await self._try_performance_optimized_allocation(requirements, provider, available_instances[provider])

            if allocation and allocation["total_performance"] > best_allocation["total_performance"]:
                best_allocation = allocation

        return best_allocation

    async def _try_performance_optimized_allocation(self, requirements: List[ResourceSpec], provider: ResourceProvider, instances: List[ResourceInstance]) -> Optional[Dict[str, Any]]:
        """Try performance-optimized allocation for a provider."""
        try:
            # Sort instances by performance score (descending)
            sorted_instances = sorted(instances, key=lambda x: x.performance_score, reverse=True)

            allocated_instances = []
            total_cost = 0
            total_performance = 0
            total_carbon = 0

            for req in requirements:
                # Find highest performance suitable instance
                suitable_instances = [
                    inst for inst in sorted_instances
                    if (inst.resource_type == req.resource_type and
                        inst.get_available_capacity() >= req.amount and
                        inst.performance_score >= req.min_performance_score and
                        inst.latency_ms <= req.max_latency_ms)
                ]

                if not suitable_instances:
                    return None

                # Select highest performance instance
                best_instance = suitable_instances[0]

                # Allocate capacity
                best_instance.allocated += req.amount
                allocated_instances.append(best_instance)

                # Calculate totals
                instance_cost = best_instance.cost_per_hour * req.required_duration_hours
                total_cost += instance_cost
                total_performance += best_instance.performance_score * req.amount
                total_carbon += best_instance.calculate_carbon_footprint(req.required_duration_hours)

            return {
                "instances": allocated_instances,
                "total_cost": total_cost,
                "total_performance": total_performance / len(requirements),
                "total_carbon": total_carbon,
                "confidence": 0.8,
            }

        except Exception as e:
            logger.error(f"Performance optimization failed for {provider}: {e}")
            return None

class ResourceOptimizationOrchestrator:
    """Main orchestrator for resource optimization."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.allocator = DynamicResourceAllocator(config.get("allocator", {}))
        self.carbon_monitor = CarbonFootprintMonitor(config.get("carbon_monitoring", {}))
        self.cost_tracker = CostTracker(config.get("cost_tracking", {}))

    async def initialize(self) -> bool:
        """Initialize resource optimization system."""
        try:
            logger.info("Initializing resource optimization system")

            # Initialize components
            await self.allocator.initialize()
            await self.carbon_monitor.start_monitoring()
            await self.cost_tracker.start_tracking()

            logger.info("Resource optimization system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize resource optimization system: {e}")
            return False

    async def optimize_resources(self, requirements: List[ResourceSpec], objective: OptimizationObjective) -> OptimizationResult:
        """Optimize resource allocation."""
        try:
            # Get optimization result
            result = await self.allocator.allocate_resources(requirements, objective)

            # Track costs and carbon footprint
            await self.cost_tracker.track_allocation(result)
            await self.carbon_monitor.track_allocation(result)

            return result

        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            raise

    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        allocation_status = await self.allocator.get_allocation_status()
        carbon_metrics = await self.carbon_monitor.get_carbon_metrics()
        cost_metrics = await self.cost_tracker.get_cost_metrics()

        return {
            "allocation": allocation_status,
            "carbon": carbon_metrics,
            "cost": cost_metrics,
            "overall_efficiency": {
                "cost_efficiency": cost_metrics.get("cost_per_unit", 0),
                "carbon_efficiency": carbon_metrics.get("carbon_per_unit", 0),
                "performance_efficiency": allocation_status.get("utilization_percent", 0),
            },
        }

class CarbonFootprintMonitor:
    """Monitors and tracks carbon footprint."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.carbon_tracking = {}
        self.monitoring_active = False

    async def start_monitoring(self) -> None:
        """Start carbon footprint monitoring."""
        self.monitoring_active = True

    async def track_allocation(self, allocation_result: OptimizationResult) -> None:
        """Track carbon footprint for an allocation."""
        self.carbon_tracking[str(allocation_result.optimization_id)] = allocation_result.total_carbon_footprint_g

    async def get_carbon_metrics(self) -> Dict[str, Any]:
        """Get carbon footprint metrics."""
        total_carbon = sum(self.carbon_tracking.values())
        return {
            "total_carbon_g": total_carbon,
            "carbon_per_unit": total_carbon / max(len(self.carbon_tracking), 1),
            "tracked_allocations": len(self.carbon_tracking),
        }

class CostTracker:
    """Tracks and analyzes costs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cost_tracking = {}
        self.tracking_active = False

    async def start_tracking(self) -> None:
        """Start cost tracking."""
        self.tracking_active = True

    async def track_allocation(self, allocation_result: OptimizationResult) -> None:
        """Track costs for an allocation."""
        self.cost_tracking[str(allocation_result.optimization_id)] = allocation_result.total_cost

    async def get_cost_metrics(self) -> Dict[str, Any]:
        """Get cost metrics."""
        total_cost = sum(self.cost_tracking.values())
        return {
            "total_cost": total_cost,
            "cost_per_unit": total_cost / max(len(self.cost_tracking), 1),
            "tracked_allocations": len(self.cost_tracking),
        }

# Example usage and testing
async def create_sample_resource_requirements() -> List[ResourceSpec]:
    """Create sample resource requirements for testing."""
    return [
        ResourceSpec(
            resource_type=ResourceType.CPU,
            amount=8.0,
            unit="cores",
            required_duration_hours=4.0,
            max_cost_per_hour=5.0,
            min_performance_score=0.8,
        ),
        ResourceSpec(
            resource_type=ResourceType.GPU,
            amount=2.0,
            unit="gpus",
            required_duration_hours=4.0,
            max_cost_per_hour=20.0,
            min_performance_score=0.9,
            prefer_spot_instances=True,
        ),
        ResourceSpec(
            resource_type=ResourceType.MEMORY,
            amount=32.0,
            unit="gb",
            required_duration_hours=4.0,
            max_cost_per_hour=2.0,
            carbon_aware=True,
        ),
    ]
