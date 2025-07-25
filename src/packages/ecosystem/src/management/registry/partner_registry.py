"""
Partner registry for managing ecosystem partnerships.

This module provides centralized management and registry of all
platform partnerships and integrations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from uuid import UUID, uuid4

import structlog

from ...core.interfaces import (
    IntegrationInterface, PartnerInterface, PartnerTier,
    PartnerCapability, PartnerContract, PartnerMetrics,
    ConnectionHealth, IntegrationStatus
)

logger = structlog.get_logger(__name__)


class PartnerRegistryError(Exception):
    """Partner registry specific exceptions."""
    pass


class PartnerRegistry:
    """
    Centralized registry for managing ecosystem partnerships.
    
    This registry maintains the catalog of all partnerships,
    monitors their health, and provides management capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize partner registry."""
        self.config = config or {}
        self.id = uuid4()
        self.logger = logger.bind(registry_id=str(self.id))
        
        # Partner storage
        self._partners: Dict[str, PartnerInterface] = {}
        self._integrations: Dict[str, IntegrationInterface] = {}
        self._contracts: Dict[str, PartnerContract] = {}
        
        # Monitoring
        self._health_checks: Dict[str, datetime] = {}
        self._health_callbacks: List[Callable[[str, ConnectionHealth], None]] = []
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval = self.config.get("monitoring_interval_minutes", 5)
        
        self.logger.info("Partner registry initialized")
    
    # Partner registration and management
    
    async def register_partner(
        self,
        name: str,
        partner: PartnerInterface,
        integration: IntegrationInterface,
        tier: PartnerTier = PartnerTier.PROFESSIONAL,
        capabilities: Optional[Set[PartnerCapability]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register new partner.
        
        Args:
            name: Partner name
            partner: Partner interface implementation
            integration: Integration interface implementation
            tier: Partnership tier
            capabilities: Partner capabilities
            metadata: Additional metadata
            
        Returns:
            str: Partner ID
        """
        try:
            # Validate inputs
            if name in self._partners:
                raise PartnerRegistryError(f"Partner '{name}' already registered")
            
            # Store partner and integration
            self._partners[name] = partner
            self._integrations[name] = integration
            
            # Store contract
            contract = partner.contract
            contract.tier = tier
            if capabilities:
                contract.capabilities = capabilities
            
            self._contracts[name] = contract
            
            # Initialize health monitoring
            self._health_checks[name] = datetime.utcnow()
            
            # Log registration
            self.logger.info(
                "Partner registered",
                partner_name=name,
                tier=tier.value,
                contract_id=contract.contract_id,
                capabilities=len(capabilities) if capabilities else 0
            )
            
            return name
            
        except Exception as e:
            self.logger.error(
                "Failed to register partner",
                partner_name=name,
                error=str(e)
            )
            raise PartnerRegistryError(f"Registration failed: {str(e)}")
    
    async def unregister_partner(self, partner_name: str) -> bool:
        """
        Unregister partner.
        
        Args:
            partner_name: Partner name
            
        Returns:
            bool: True if unregistration successful, False otherwise
        """
        try:
            if partner_name not in self._partners:
                self.logger.warning(
                    "Partner not found for unregistration",
                    partner_name=partner_name
                )
                return False
            
            # Disconnect integration
            integration = self._integrations.get(partner_name)
            if integration:
                await integration.disconnect()
            
            # Remove from registry
            self._partners.pop(partner_name, None)
            self._integrations.pop(partner_name, None)
            self._contracts.pop(partner_name, None)
            self._health_checks.pop(partner_name, None)
            
            self.logger.info("Partner unregistered", partner_name=partner_name)
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to unregister partner",
                partner_name=partner_name,
                error=str(e)
            )
            return False
    
    async def get_partner(self, partner_name: str) -> Optional[PartnerInterface]:
        """Get partner by name."""
        return self._partners.get(partner_name)
    
    async def get_integration(self, partner_name: str) -> Optional[IntegrationInterface]:
        """Get integration by partner name."""
        return self._integrations.get(partner_name)
    
    async def get_contract(self, partner_name: str) -> Optional[PartnerContract]:
        """Get contract by partner name."""
        return self._contracts.get(partner_name)
    
    async def list_partners(
        self,
        tier: Optional[PartnerTier] = None,
        capability: Optional[PartnerCapability] = None,
        active_only: bool = True
    ) -> List[str]:
        """
        List registered partners with optional filters.
        
        Args:
            tier: Filter by partnership tier
            capability: Filter by capability
            active_only: Only return active partners
            
        Returns:
            List[str]: List of partner names
        """
        partners = []
        
        for name, contract in self._contracts.items():
            # Filter by tier
            if tier and contract.tier != tier:
                continue
            
            # Filter by capability
            if capability and capability not in contract.capabilities:
                continue
            
            # Filter by active status
            if active_only and not contract.is_active:
                continue
            
            partners.append(name)
        
        return partners
    
    # Health monitoring
    
    async def check_health(self, partner_name: str) -> Optional[ConnectionHealth]:
        """
        Check health of specific partner.
        
        Args:
            partner_name: Partner name
            
        Returns:
            ConnectionHealth: Health status, None if partner not found
        """
        integration = self._integrations.get(partner_name)
        if not integration:
            return None
        
        try:
            health = await integration.test_connection()
            self._health_checks[partner_name] = datetime.utcnow()
            
            # Notify health callbacks
            for callback in self._health_callbacks:
                try:
                    callback(partner_name, health)
                except Exception as e:
                    self.logger.error(
                        "Health callback failed",
                        partner_name=partner_name,
                        error=str(e)
                    )
            
            return health
            
        except Exception as e:
            self.logger.error(
                "Health check failed",
                partner_name=partner_name,
                error=str(e)
            )
            return ConnectionHealth.UNHEALTHY
    
    async def check_all_health(self) -> Dict[str, ConnectionHealth]:
        """
        Check health of all registered partners.
        
        Returns:
            Dict[str, ConnectionHealth]: Health status for each partner
        """
        health_results = {}
        
        # Run health checks concurrently
        tasks = []
        for partner_name in self._partners.keys():
            task = asyncio.create_task(self.check_health(partner_name))
            tasks.append((partner_name, task))
        
        # Collect results
        for partner_name, task in tasks:
            try:
                health = await task
                health_results[partner_name] = health
            except Exception as e:
                self.logger.error(
                    "Health check task failed",
                    partner_name=partner_name,
                    error=str(e)
                )
                health_results[partner_name] = ConnectionHealth.UNHEALTHY
        
        return health_results
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """
        Get health summary for all partners.
        
        Returns:
            Dict[str, Any]: Health summary statistics
        """
        health_results = await self.check_all_health()
        
        summary = {
            "total_partners": len(health_results),
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "unknown": 0,
            "health_percentage": 0.0,
            "last_check": datetime.utcnow().isoformat()
        }
        
        for health in health_results.values():
            if health == ConnectionHealth.HEALTHY:
                summary["healthy"] += 1
            elif health == ConnectionHealth.DEGRADED:
                summary["degraded"] += 1
            elif health == ConnectionHealth.UNHEALTHY:
                summary["unhealthy"] += 1
            else:
                summary["unknown"] += 1
        
        if summary["total_partners"] > 0:
            summary["health_percentage"] = (
                summary["healthy"] / summary["total_partners"]
            ) * 100.0
        
        return summary
    
    # Capability management
    
    async def find_partners_by_capability(
        self, 
        capability: PartnerCapability
    ) -> List[str]:
        """
        Find partners that provide specific capability.
        
        Args:
            capability: Required capability
            
        Returns:
            List[str]: List of partner names with the capability
        """
        partners = []
        
        for name, contract in self._contracts.items():
            if capability in contract.capabilities:
                partners.append(name)
        
        return partners
    
    async def get_available_capabilities(self) -> Set[PartnerCapability]:
        """
        Get all capabilities available across partners.
        
        Returns:
            Set[PartnerCapability]: Set of available capabilities
        """
        capabilities = set()
        
        for contract in self._contracts.values():
            capabilities.update(contract.capabilities)
        
        return capabilities
    
    async def get_capability_coverage(self) -> Dict[PartnerCapability, List[str]]:
        """
        Get capability coverage mapping.
        
        Returns:
            Dict[PartnerCapability, List[str]]: Mapping of capabilities to partners
        """
        coverage = {}
        
        for capability in PartnerCapability:
            partners = await self.find_partners_by_capability(capability)
            if partners:
                coverage[capability] = partners
        
        return coverage
    
    # Contract management
    
    async def get_expiring_contracts(
        self,
        days_threshold: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get contracts expiring within threshold.
        
        Args:
            days_threshold: Days before expiry to consider as expiring
            
        Returns:
            List[Dict[str, Any]]: List of expiring contracts
        """
        expiring = []
        
        for name, contract in self._contracts.items():
            days_until_expiry = contract.days_until_expiry
            
            if days_until_expiry is not None and days_until_expiry <= days_threshold:
                expiring.append({
                    "partner_name": name,
                    "contract_id": contract.contract_id,
                    "days_until_expiry": days_until_expiry,
                    "end_date": contract.end_date.isoformat() if contract.end_date else None,
                    "tier": contract.tier.value
                })
        
        return expiring
    
    async def update_contract(
        self,
        partner_name: str,
        contract: PartnerContract
    ) -> bool:
        """
        Update partner contract.
        
        Args:
            partner_name: Partner name
            contract: New contract
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if partner_name not in self._partners:
                return False
            
            partner = self._partners[partner_name]
            success = await partner.update_contract(contract)
            
            if success:
                self._contracts[partner_name] = contract
                self.logger.info(
                    "Contract updated",
                    partner_name=partner_name,
                    contract_id=contract.contract_id
                )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Failed to update contract",
                partner_name=partner_name,
                error=str(e)
            )
            return False
    
    # Metrics and analytics
    
    async def collect_partner_metrics(
        self,
        partner_name: str
    ) -> Optional[PartnerMetrics]:
        """
        Collect metrics for specific partner.
        
        Args:
            partner_name: Partner name
            
        Returns:
            PartnerMetrics: Partner metrics, None if not found
        """
        partner = self._partners.get(partner_name)
        if not partner:
            return None
        
        try:
            return await partner.collect_metrics()
        except Exception as e:
            self.logger.error(
                "Failed to collect partner metrics",
                partner_name=partner_name,
                error=str(e)
            )
            return None
    
    async def collect_all_metrics(self) -> Dict[str, PartnerMetrics]:
        """
        Collect metrics for all partners.
        
        Returns:
            Dict[str, PartnerMetrics]: Metrics for each partner
        """
        metrics = {}
        
        tasks = []
        for partner_name in self._partners.keys():
            task = asyncio.create_task(self.collect_partner_metrics(partner_name))
            tasks.append((partner_name, task))
        
        for partner_name, task in tasks:
            try:
                partner_metrics = await task
                if partner_metrics:
                    metrics[partner_name] = partner_metrics
            except Exception as e:
                self.logger.error(
                    "Metrics collection task failed",
                    partner_name=partner_name,
                    error=str(e)
                )
        
        return metrics
    
    async def generate_usage_report(
        self,
        start_date: datetime,
        end_date: datetime,
        partner_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate usage report for specified period.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            partner_names: Specific partners to include (all if None)
            
        Returns:
            Dict[str, Any]: Usage report
        """
        report = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_partners": 0,
                "active_partners": 0,
                "total_api_calls": 0,
                "total_data_volume_gb": 0.0,
                "total_cost_usd": 0.0
            },
            "partners": {}
        }
        
        target_partners = partner_names if partner_names else list(self._partners.keys())
        
        for partner_name in target_partners:
            partner = self._partners.get(partner_name)
            if not partner:
                continue
            
            try:
                # Generate partner-specific report
                partner_report = await partner.generate_usage_report(start_date, end_date)
                report["partners"][partner_name] = partner_report
                
                # Update summary
                report["summary"]["total_partners"] += 1
                if partner_report.get("active", False):
                    report["summary"]["active_partners"] += 1
                
                report["summary"]["total_api_calls"] += partner_report.get("api_calls", 0)
                report["summary"]["total_data_volume_gb"] += partner_report.get("data_volume_gb", 0.0)
                report["summary"]["total_cost_usd"] += partner_report.get("cost_usd", 0.0)
                
            except Exception as e:
                self.logger.error(
                    "Failed to generate partner usage report",
                    partner_name=partner_name,
                    error=str(e)
                )
        
        return report
    
    # Background monitoring
    
    async def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            return
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info(
            "Background monitoring started",
            interval_minutes=self._monitoring_interval
        )
    
    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            
            self.logger.info("Background monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                # Perform health checks
                await self.check_all_health()
                
                # Check for expiring contracts
                expiring = await self.get_expiring_contracts()
                if expiring:
                    self.logger.warning(
                        "Contracts expiring soon",
                        count=len(expiring),
                        contracts=[c["partner_name"] for c in expiring]
                    )
                
                # Wait for next check
                await asyncio.sleep(self._monitoring_interval * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    # Event handling
    
    def add_health_callback(
        self,
        callback: Callable[[str, ConnectionHealth], None]
    ) -> None:
        """Add health change callback."""
        self._health_callbacks.append(callback)
    
    def remove_health_callback(
        self,
        callback: Callable[[str, ConnectionHealth], None]
    ) -> None:
        """Remove health change callback."""
        if callback in self._health_callbacks:
            self._health_callbacks.remove(callback)
    
    # Context manager support
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_monitoring()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PartnerRegistry(partners={len(self._partners)}, "
            f"monitoring={'active' if self._monitoring_task else 'inactive'})"
        )