"""Resource quota service for managing tenant resource allocation and usage."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pynomaly.domain.models.multitenancy import (
    ResourceQuota,
    ResourceType,
    Tenant,
    TenantContext,
    TenantEvent,
)


class ResourceQuotaService:
    """Service for managing and enforcing tenant resource quotas."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Resource tracking
        self.tenant_quotas: Dict[UUID, Dict[ResourceType, ResourceQuota]] = {}
        self.resource_usage_history: Dict[UUID, List[Dict[str, Any]]] = {}

        # Real-time usage tracking
        self.current_usage: Dict[UUID, Dict[ResourceType, float]] = {}
        self.usage_events: List[Dict[str, Any]] = []

        # Quota enforcement settings
        self.enforcement_enabled = True
        self.warning_threshold = 0.8  # 80% usage warning
        self.soft_limit_threshold = 0.9  # 90% soft limit

        # Background tasks
        self.monitoring_tasks: Set[asyncio.Task] = set()
        self.is_running = False

        # Rate limiting for burst usage
        self.burst_allowances: Dict[UUID, Dict[ResourceType, float]] = {}

        self.logger.info("Resource quota service initialized")

    async def start_monitoring(self) -> None:
        """Start background resource monitoring."""

        if self.is_running:
            return

        self.is_running = True

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._quota_enforcement_loop()),
            asyncio.create_task(self._usage_monitoring_loop()),
            asyncio.create_task(self._quota_reset_loop()),
            asyncio.create_task(self._burst_management_loop()),
            asyncio.create_task(self._usage_analytics_loop()),
            asyncio.create_task(self._cleanup_loop()),
        ]

        self.monitoring_tasks.update(tasks)

        self.logger.info("Started resource quota monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""

        self.is_running = False

        for task in self.monitoring_tasks:
            task.cancel()

        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()

        self.logger.info("Stopped resource quota monitoring")

    async def initialize_tenant_quotas(self, tenant: Tenant) -> None:
        """Initialize resource quotas for a new tenant."""

        tenant_id = tenant.tenant_id
        self.tenant_quotas[tenant_id] = tenant.resource_quotas.copy()
        self.current_usage[tenant_id] = {
            resource_type: 0.0 for resource_type in ResourceType
        }
        self.resource_usage_history[tenant_id] = []
        self.burst_allowances[tenant_id] = {}

        self.logger.info(f"Initialized quotas for tenant {tenant.name}")

    async def allocate_resource(
        self,
        tenant_context: TenantContext,
        resource_type: ResourceType,
        amount: float,
        duration_seconds: Optional[int] = None,
    ) -> bool:
        """Allocate resources for a tenant operation."""

        tenant_id = tenant_context.tenant.tenant_id

        # Check if tenant has quota for this resource
        if tenant_id not in self.tenant_quotas:
            await self.initialize_tenant_quotas(tenant_context.tenant)

        quota = self.tenant_quotas[tenant_id].get(resource_type)
        if not quota:
            self.logger.warning(f"No quota defined for {resource_type.value} for tenant {tenant_id}")
            return False

        # Check if allocation would exceed quota
        current_usage = self.current_usage[tenant_id].get(resource_type, 0.0)

        if not self._can_allocate_resource(quota, current_usage, amount):
            await self._handle_quota_exceeded(tenant_context, resource_type, amount)
            return False

        # Allocate the resource
        success = quota.consume(amount)
        if success:
            self.current_usage[tenant_id][resource_type] = quota.used_amount

            # Track in context for cleanup
            tenant_context.consumed_resources[resource_type] = (
                tenant_context.consumed_resources.get(resource_type, 0.0) + amount
            )

            # Log usage event
            await self._log_usage_event(
                tenant_id,
                resource_type,
                "allocate",
                amount,
                quota.used_amount,
                duration_seconds
            )

            # Check for warnings
            await self._check_usage_thresholds(tenant_context, resource_type, quota)

            self.logger.debug(f"Allocated {amount} {resource_type.value} to tenant {tenant_id}")

        return success

    async def release_resource(
        self,
        tenant_id: UUID,
        resource_type: ResourceType,
        amount: float,
    ) -> None:
        """Release allocated resources."""

        if tenant_id not in self.tenant_quotas:
            return

        quota = self.tenant_quotas[tenant_id].get(resource_type)
        if quota:
            quota.release(amount)
            self.current_usage[tenant_id][resource_type] = quota.used_amount

            # Log usage event
            await self._log_usage_event(
                tenant_id,
                resource_type,
                "release",
                amount,
                quota.used_amount,
            )

            self.logger.debug(f"Released {amount} {resource_type.value} from tenant {tenant_id}")

    async def update_tenant_quota(
        self,
        tenant_id: UUID,
        resource_type: ResourceType,
        new_quota: ResourceQuota,
    ) -> bool:
        """Update quota for a specific resource type."""

        if tenant_id not in self.tenant_quotas:
            self.tenant_quotas[tenant_id] = {}

        # Preserve current usage
        old_quota = self.tenant_quotas[tenant_id].get(resource_type)
        if old_quota:
            new_quota.used_amount = min(old_quota.used_amount, new_quota.allocated_amount)

        self.tenant_quotas[tenant_id][resource_type] = new_quota
        self.current_usage[tenant_id][resource_type] = new_quota.used_amount

        await self._log_usage_event(
            tenant_id,
            resource_type,
            "quota_updated",
            new_quota.allocated_amount,
            new_quota.used_amount,
        )

        self.logger.info(f"Updated {resource_type.value} quota for tenant {tenant_id}")
        return True

    async def get_tenant_usage_summary(self, tenant_id: UUID) -> Dict[str, Any]:
        """Get comprehensive usage summary for tenant."""

        if tenant_id not in self.tenant_quotas:
            return {}

        quotas = self.tenant_quotas[tenant_id]
        usage_summary = {}

        for resource_type, quota in quotas.items():
            utilization = quota.get_utilization_percentage()

            usage_summary[resource_type.value] = {
                "allocated": quota.allocated_amount,
                "used": quota.used_amount,
                "available": quota.allocated_amount - quota.used_amount,
                "utilization_percentage": utilization,
                "unit": quota.unit,
                "soft_limit": quota.soft_limit,
                "hard_limit": quota.hard_limit,
                "burst_limit": quota.burst_limit,
                "is_over_soft_limit": quota.is_soft_limit_exceeded(),
                "is_over_hard_limit": quota.is_hard_limit_exceeded(),
                "peak_usage": quota.peak_usage,
                "average_usage": quota.average_usage,
                "last_reset": quota.last_reset.isoformat(),
            }

        return {
            "tenant_id": str(tenant_id),
            "resources": usage_summary,
            "overall_utilization": self._calculate_overall_utilization(quotas),
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_resource_usage_history(
        self,
        tenant_id: UUID,
        resource_type: Optional[ResourceType] = None,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get historical resource usage data."""

        if tenant_id not in self.resource_usage_history:
            return []

        history = self.resource_usage_history[tenant_id]
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Filter by time and resource type
        filtered_history = []
        for event in history:
            event_time = datetime.fromisoformat(event["timestamp"])
            if event_time >= cutoff_time:
                if resource_type is None or event["resource_type"] == resource_type.value:
                    filtered_history.append(event)

        return sorted(filtered_history, key=lambda x: x["timestamp"])

    async def predict_resource_exhaustion(
        self,
        tenant_id: UUID,
        resource_type: ResourceType,
    ) -> Optional[datetime]:
        """Predict when a resource will be exhausted based on usage trends."""

        # Get recent usage history
        history = await self.get_resource_usage_history(tenant_id, resource_type, hours=24)

        if len(history) < 2:
            return None

        # Calculate usage trend
        allocate_events = [e for e in history if e["action"] == "allocate"]
        if len(allocate_events) < 2:
            return None

        # Simple linear regression for usage trend
        times = []
        usage_levels = []

        for event in allocate_events[-10:]:  # Use last 10 events
            timestamp = datetime.fromisoformat(event["timestamp"])
            times.append(timestamp.timestamp())
            usage_levels.append(event["current_usage"])

        if len(times) < 2:
            return None

        # Calculate slope (usage rate)
        x_mean = sum(times) / len(times)
        y_mean = sum(usage_levels) / len(usage_levels)

        numerator = sum((times[i] - x_mean) * (usage_levels[i] - y_mean) for i in range(len(times)))
        denominator = sum((times[i] - x_mean) ** 2 for i in range(len(times)))

        if denominator == 0:
            return None

        slope = numerator / denominator

        if slope <= 0:  # No growth or declining usage
            return None

        # Get current quota
        quota = self.tenant_quotas[tenant_id].get(resource_type)
        if not quota:
            return None

        # Calculate time to exhaustion
        current_usage = quota.used_amount
        available = quota.allocated_amount - current_usage

        if available <= 0:
            return datetime.utcnow()  # Already exhausted

        seconds_to_exhaustion = available / slope
        exhaustion_time = datetime.utcnow() + timedelta(seconds=seconds_to_exhaustion)

        return exhaustion_time

    async def suggest_quota_adjustments(self, tenant_id: UUID) -> Dict[str, Any]:
        """Suggest quota adjustments based on usage patterns."""

        if tenant_id not in self.tenant_quotas:
            return {}

        suggestions = {}
        quotas = self.tenant_quotas[tenant_id]

        for resource_type, quota in quotas.items():
            utilization = quota.get_utilization_percentage()

            suggestion = {
                "current_quota": quota.allocated_amount,
                "current_usage": quota.used_amount,
                "utilization": utilization,
                "suggestion": "no_change",
                "recommended_quota": quota.allocated_amount,
                "reason": "",
            }

            # Analyze usage patterns
            if utilization > 95:
                # Very high utilization - suggest increase
                suggested_increase = quota.allocated_amount * 0.5  # 50% increase
                suggestion.update({
                    "suggestion": "increase",
                    "recommended_quota": quota.allocated_amount + suggested_increase,
                    "reason": "High utilization indicates need for more resources",
                })

            elif utilization < 10:
                # Very low utilization - suggest decrease
                suggested_decrease = quota.allocated_amount * 0.3  # 30% decrease
                suggestion.update({
                    "suggestion": "decrease",
                    "recommended_quota": max(quota.used_amount * 1.2, quota.allocated_amount - suggested_decrease),
                    "reason": "Low utilization indicates over-allocation",
                })

            elif quota.is_soft_limit_exceeded():
                suggestion.update({
                    "suggestion": "increase",
                    "recommended_quota": quota.allocated_amount * 1.2,
                    "reason": "Soft limit exceeded frequently",
                })

            # Check for predicted exhaustion
            exhaustion_time = await self.predict_resource_exhaustion(tenant_id, resource_type)
            if exhaustion_time and exhaustion_time < datetime.utcnow() + timedelta(days=7):
                suggestion.update({
                    "suggestion": "increase",
                    "recommended_quota": quota.allocated_amount * 1.5,
                    "reason": f"Resource exhaustion predicted by {exhaustion_time.isoformat()}",
                    "urgency": "high",
                })

            suggestions[resource_type.value] = suggestion

        return {
            "tenant_id": str(tenant_id),
            "suggestions": suggestions,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _can_allocate_resource(
        self,
        quota: ResourceQuota,
        current_usage: float,
        requested_amount: float,
    ) -> bool:
        """Check if resource can be allocated without exceeding limits."""

        new_usage = current_usage + requested_amount

        # Check hard limit first
        if quota.hard_limit and new_usage > quota.hard_limit:
            return False

        # Check allocated amount
        if new_usage > quota.allocated_amount:
            # Check if burst is allowed
            if quota.burst_limit and new_usage <= quota.burst_limit:
                return True
            return False

        return True

    async def _handle_quota_exceeded(
        self,
        tenant_context: TenantContext,
        resource_type: ResourceType,
        requested_amount: float,
    ) -> None:
        """Handle quota exceeded scenario."""

        tenant_id = tenant_context.tenant.tenant_id
        quota = self.tenant_quotas[tenant_id][resource_type]

        await self._log_usage_event(
            tenant_id,
            resource_type,
            "quota_exceeded",
            requested_amount,
            quota.used_amount,
        )

        self.logger.warning(
            f"Quota exceeded for tenant {tenant_id}: "
            f"requested {requested_amount} {resource_type.value}, "
            f"but only {quota.allocated_amount - quota.used_amount} available"
        )

    async def _check_usage_thresholds(
        self,
        tenant_context: TenantContext,
        resource_type: ResourceType,
        quota: ResourceQuota,
    ) -> None:
        """Check usage thresholds and generate warnings."""

        utilization = quota.get_utilization_percentage()

        if utilization >= self.warning_threshold * 100:
            await self._log_usage_event(
                tenant_context.tenant.tenant_id,
                resource_type,
                "warning_threshold_exceeded",
                0,
                quota.used_amount,
            )

        if quota.is_soft_limit_exceeded():
            await self._log_usage_event(
                tenant_context.tenant.tenant_id,
                resource_type,
                "soft_limit_exceeded",
                0,
                quota.used_amount,
            )

    async def _log_usage_event(
        self,
        tenant_id: UUID,
        resource_type: ResourceType,
        action: str,
        amount: float,
        current_usage: float,
        duration_seconds: Optional[int] = None,
    ) -> None:
        """Log resource usage event."""

        event = {
            "tenant_id": str(tenant_id),
            "resource_type": resource_type.value,
            "action": action,
            "amount": amount,
            "current_usage": current_usage,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.usage_events.append(event)

        # Add to tenant history
        if tenant_id not in self.resource_usage_history:
            self.resource_usage_history[tenant_id] = []

        self.resource_usage_history[tenant_id].append(event)

        # Limit history size
        if len(self.resource_usage_history[tenant_id]) > 10000:
            self.resource_usage_history[tenant_id] = self.resource_usage_history[tenant_id][-5000:]

    def _calculate_overall_utilization(
        self,
        quotas: Dict[ResourceType, ResourceQuota],
    ) -> float:
        """Calculate overall utilization across all resources."""

        if not quotas:
            return 0.0

        total_utilization = sum(quota.get_utilization_percentage() for quota in quotas.values())
        return total_utilization / len(quotas)

    # Background monitoring tasks

    async def _quota_enforcement_loop(self) -> None:
        """Background task for quota enforcement."""

        while self.is_running:
            try:
                if self.enforcement_enabled:
                    for tenant_id, quotas in self.tenant_quotas.items():
                        for resource_type, quota in quotas.items():
                            if quota.is_hard_limit_exceeded():
                                await self._enforce_hard_limit(tenant_id, resource_type, quota)

            except Exception as e:
                self.logger.error(f"Quota enforcement error: {e}")

            await asyncio.sleep(60)  # Check every minute

    async def _enforce_hard_limit(
        self,
        tenant_id: UUID,
        resource_type: ResourceType,
        quota: ResourceQuota,
    ) -> None:
        """Enforce hard limit for resource."""

        # In production, this would take corrective actions like:
        # - Throttling requests
        # - Suspending tenant operations
        # - Scaling down resources

        self.logger.critical(
            f"Hard limit exceeded for tenant {tenant_id} "
            f"on {resource_type.value}: {quota.used_amount}/{quota.hard_limit}"
        )

    async def _usage_monitoring_loop(self) -> None:
        """Background task for monitoring usage patterns."""

        while self.is_running:
            try:
                for tenant_id, quotas in self.tenant_quotas.items():
                    for resource_type, quota in quotas.items():
                        # Update average usage
                        if quota.data_points:
                            recent_usage = [
                                p.value for p in quota.data_points[-100:]  # Last 100 data points
                                if isinstance(p.value, (int, float))
                            ]
                            if recent_usage:
                                quota.average_usage = sum(recent_usage) / len(recent_usage)

            except Exception as e:
                self.logger.error(f"Usage monitoring error: {e}")

            await asyncio.sleep(300)  # Check every 5 minutes

    async def _quota_reset_loop(self) -> None:
        """Background task for resetting time-based quotas."""

        while self.is_running:
            try:
                now = datetime.utcnow()

                for tenant_id, quotas in self.tenant_quotas.items():
                    for resource_type, quota in quotas.items():
                        if quota.reset_schedule and quota.time_window:
                            # Check if reset is due
                            if self._should_reset_quota(quota, now):
                                quota.reset_usage()
                                self.current_usage[tenant_id][resource_type] = 0.0

                                await self._log_usage_event(
                                    tenant_id,
                                    resource_type,
                                    "quota_reset",
                                    0,
                                    0,
                                )

            except Exception as e:
                self.logger.error(f"Quota reset error: {e}")

            await asyncio.sleep(3600)  # Check every hour

    def _should_reset_quota(self, quota: ResourceQuota, current_time: datetime) -> bool:
        """Check if quota should be reset based on schedule."""

        if not quota.reset_schedule or not quota.last_reset:
            return False

        time_since_reset = current_time - quota.last_reset

        if quota.reset_schedule == "daily":
            return time_since_reset >= timedelta(days=1)
        elif quota.reset_schedule == "weekly":
            return time_since_reset >= timedelta(weeks=1)
        elif quota.reset_schedule == "monthly":
            return time_since_reset >= timedelta(days=30)  # Simplified

        return False

    async def _burst_management_loop(self) -> None:
        """Background task for managing burst allowances."""

        while self.is_running:
            try:
                # Reset burst allowances periodically
                for tenant_id in self.burst_allowances:
                    self.burst_allowances[tenant_id] = {}

            except Exception as e:
                self.logger.error(f"Burst management error: {e}")

            await asyncio.sleep(3600)  # Reset every hour

    async def _usage_analytics_loop(self) -> None:
        """Background task for usage analytics and optimization."""

        while self.is_running:
            try:
                # Analyze usage patterns and suggest optimizations
                for tenant_id in self.tenant_quotas:
                    suggestions = await self.suggest_quota_adjustments(tenant_id)

                    # Log significant suggestions
                    for resource_type, suggestion in suggestions.get("suggestions", {}).items():
                        if suggestion["suggestion"] != "no_change":
                            self.logger.info(
                                f"Quota suggestion for tenant {tenant_id} "
                                f"{resource_type}: {suggestion['suggestion']} "
                                f"to {suggestion['recommended_quota']}"
                            )

            except Exception as e:
                self.logger.error(f"Usage analytics error: {e}")

            await asyncio.sleep(3600)  # Analyze every hour

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up old usage data."""

        while self.is_running:
            try:
                # Clean up old usage events
                cutoff_time = datetime.utcnow() - timedelta(days=30)

                self.usage_events = [
                    event for event in self.usage_events
                    if datetime.fromisoformat(event["timestamp"]) > cutoff_time
                ]

                # Clean up tenant usage history
                for tenant_id, history in self.resource_usage_history.items():
                    self.resource_usage_history[tenant_id] = [
                        event for event in history
                        if datetime.fromisoformat(event["timestamp"]) > cutoff_time
                    ]

            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

            await asyncio.sleep(86400)  # Clean up daily

    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service metrics and statistics."""

        total_tenants = len(self.tenant_quotas)
        total_resources = sum(len(quotas) for quotas in self.tenant_quotas.values())

        # Calculate utilization statistics
        all_utilizations = []
        for quotas in self.tenant_quotas.values():
            for quota in quotas.values():
                all_utilizations.append(quota.get_utilization_percentage())

        avg_utilization = sum(all_utilizations) / len(all_utilizations) if all_utilizations else 0

        # Count quota violations
        soft_limit_violations = 0
        hard_limit_violations = 0

        for quotas in self.tenant_quotas.values():
            for quota in quotas.values():
                if quota.is_soft_limit_exceeded():
                    soft_limit_violations += 1
                if quota.is_hard_limit_exceeded():
                    hard_limit_violations += 1

        return {
            "total_tenants": total_tenants,
            "total_resources": total_resources,
            "average_utilization": avg_utilization,
            "soft_limit_violations": soft_limit_violations,
            "hard_limit_violations": hard_limit_violations,
            "enforcement_enabled": self.enforcement_enabled,
            "warning_threshold": self.warning_threshold,
            "total_usage_events": len(self.usage_events),
            "monitoring_active": self.is_running,
        }
