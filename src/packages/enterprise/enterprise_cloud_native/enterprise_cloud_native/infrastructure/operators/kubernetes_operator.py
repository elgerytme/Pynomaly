"""
Kubernetes Operator Base Class

Provides a base class for implementing Kubernetes operators with
standardized reconciliation patterns and resource management.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from structlog import get_logger
import kubernetes.client
from kubernetes.client.rest import ApiException

from ...domain.entities.kubernetes_resource import OperatorResource, OperatorState
from .operator_framework import ReconcilerState

logger = get_logger(__name__)


class KubernetesOperator(ABC):
    """
    Base class for Kubernetes operators.
    
    Provides common functionality for operator implementation including
    resource management, reconciliation loops, and error handling.
    """
    
    def __init__(
        self,
        name: str,
        crd_group: str,
        crd_version: str,
        crd_kind: str,
        namespace: Optional[str] = None,
        reconcile_interval: int = 300
    ):
        self.name = name
        self.crd_group = crd_group
        self.crd_version = crd_version
        self.crd_kind = crd_kind
        self.namespace = namespace
        self.reconcile_interval = reconcile_interval
        
        self.state = ReconcilerState()
        self.running = False
        self.reconcile_task: Optional[asyncio.Task] = None
        
        # Kubernetes clients
        self.api_client = kubernetes.client.ApiClient()
        self.core_v1 = kubernetes.client.CoreV1Api()
        self.apps_v1 = kubernetes.client.AppsV1Api()
        self.custom_objects_api = kubernetes.client.CustomObjectsApi()
        
        logger.info("KubernetesOperator initialized", 
                   name=name, crd=f"{crd_kind}.{crd_group}")
    
    async def start(self) -> None:
        """Start the operator."""
        if self.running:
            logger.warning("Operator already running", name=self.name)
            return
        
        logger.info("Starting operator", name=self.name)
        self.running = True
        
        # Start reconciliation loop
        self.reconcile_task = asyncio.create_task(self._reconciliation_loop())
        
        # Call operator-specific startup
        await self.on_startup()
        
        logger.info("Operator started", name=self.name)
    
    async def stop(self) -> None:
        """Stop the operator."""
        if not self.running:
            logger.warning("Operator not running", name=self.name)
            return
        
        logger.info("Stopping operator", name=self.name)
        self.running = False
        
        # Cancel reconciliation loop
        if self.reconcile_task:
            self.reconcile_task.cancel()
            try:
                await self.reconcile_task
            except asyncio.CancelledError:
                pass
        
        # Call operator-specific cleanup
        await self.on_shutdown()
        
        logger.info("Operator stopped", name=self.name)
    
    async def _reconciliation_loop(self) -> None:
        """Main reconciliation loop."""
        logger.info("Starting reconciliation loop", name=self.name)
        
        while self.running:
            try:
                await self._reconcile_all_resources()
                self.state.record_reconciliation(success=True)
                
            except Exception as e:
                logger.error("Reconciliation loop error", name=self.name, error=str(e))
                self.state.record_reconciliation(success=False, error=str(e))
            
            # Wait for next reconciliation
            try:
                await asyncio.sleep(self.reconcile_interval)
            except asyncio.CancelledError:
                break
        
        logger.info("Reconciliation loop stopped", name=self.name)
    
    async def _reconcile_all_resources(self) -> None:
        """Reconcile all managed resources."""
        logger.debug("Starting resource reconciliation", name=self.name)
        
        try:
            # Get all custom resources of this type
            if self.namespace:
                resources = await self._list_namespaced_custom_resources()
            else:
                resources = await self._list_cluster_custom_resources()
            
            # Reconcile each resource
            for resource_data in resources.get('items', []):
                try:
                    await self._reconcile_resource(resource_data)
                except Exception as e:
                    logger.error("Failed to reconcile resource", 
                               name=resource_data.get('metadata', {}).get('name'),
                               error=str(e))
            
        except Exception as e:
            logger.error("Failed to list custom resources", name=self.name, error=str(e))
            raise
    
    async def _reconcile_resource(self, resource_data: Dict[str, Any]) -> None:
        """Reconcile a single resource."""
        metadata = resource_data.get('metadata', {})
        name = metadata.get('name')
        namespace = metadata.get('namespace')
        
        logger.debug("Reconciling resource", operator=self.name, resource=name)
        
        try:
            # Get or create operator resource
            operator_resource = self.state.get_resource(namespace or 'default', name)
            
            if not operator_resource:
                # Create new operator resource
                operator_resource = OperatorResource(
                    crd_name=f"{self.crd_kind.lower()}s.{self.crd_group}",
                    crd_version=self.crd_version,
                    crd_group=self.crd_group,
                    crd_kind=self.crd_kind,
                    name=name,
                    namespace=namespace or 'default',
                    tenant_id=UUID(metadata.get('labels', {}).get('pynomaly.io/tenant-id', str(uuid4()))),
                    operator_name=self.name,
                    spec=resource_data.get('spec', {}),
                    status=resource_data.get('status', {})
                )
                
                self.state.update_resource(operator_resource)
            
            # Update resource with latest data
            operator_resource.spec = resource_data.get('spec', {})
            operator_resource.update_status(resource_data.get('status', {}))
            
            # Call operator-specific reconciliation
            await self.reconcile(operator_resource, resource_data)
            
            # Mark as successfully reconciled
            operator_resource.record_reconciliation_success()
            self.state.update_resource(operator_resource)
            
        except Exception as e:
            if operator_resource:
                operator_resource.record_reconciliation_error(str(e))
                self.state.update_resource(operator_resource)
            raise
    
    async def _list_namespaced_custom_resources(self) -> Dict[str, Any]:
        """List namespaced custom resources."""
        try:
            return self.custom_objects_api.list_namespaced_custom_object(
                group=self.crd_group,
                version=self.crd_version,
                namespace=self.namespace,
                plural=f"{self.crd_kind.lower()}s"
            )
        except ApiException as e:
            logger.error("Failed to list namespaced custom resources", 
                        error=str(e), status=e.status)
            raise
    
    async def _list_cluster_custom_resources(self) -> Dict[str, Any]:
        """List cluster-scoped custom resources."""
        try:
            return self.custom_objects_api.list_cluster_custom_object(
                group=self.crd_group,
                version=self.crd_version,
                plural=f"{self.crd_kind.lower()}s"
            )
        except ApiException as e:
            logger.error("Failed to list cluster custom resources", 
                        error=str(e), status=e.status)
            raise
    
    # Handler methods (called by Kopf)
    
    async def handle_create(
        self,
        spec: Dict[str, Any],
        name: str,
        namespace: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle custom resource creation."""
        logger.info("Handling resource creation", operator=self.name, name=name)
        
        try:
            # Create operator resource
            operator_resource = OperatorResource(
                crd_name=f"{self.crd_kind.lower()}s.{self.crd_group}",
                crd_version=self.crd_version,
                crd_group=self.crd_group,
                crd_kind=self.crd_kind,
                name=name,
                namespace=namespace,
                tenant_id=UUID(kwargs.get('labels', {}).get('pynomaly.io/tenant-id', str(uuid4()))),
                operator_name=self.name,
                spec=spec
            )
            
            # Call operator-specific creation logic
            result = await self.create_resource(operator_resource, spec, **kwargs)
            
            # Update state
            operator_resource.record_reconciliation_success()
            self.state.update_resource(operator_resource)
            
            return result or {"status": {"phase": "Created"}}
            
        except Exception as e:
            logger.error("Resource creation failed", 
                        operator=self.name, name=name, error=str(e))
            raise
    
    async def handle_update(
        self,
        spec: Dict[str, Any],
        old: Dict[str, Any],
        new: Dict[str, Any],
        diff: List[Any],
        name: str,
        namespace: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Handle custom resource updates."""
        logger.info("Handling resource update", operator=self.name, name=name)
        
        try:
            # Get existing operator resource
            operator_resource = self.state.get_resource(namespace, name)
            
            if not operator_resource:
                logger.warning("Resource not found in state", name=name)
                return None
            
            # Update spec
            operator_resource.spec = spec
            
            # Call operator-specific update logic
            result = await self.update_resource(operator_resource, old, new, diff, **kwargs)
            
            # Update state
            operator_resource.record_reconciliation_success()
            self.state.update_resource(operator_resource)
            
            return result
            
        except Exception as e:
            logger.error("Resource update failed", 
                        operator=self.name, name=name, error=str(e))
            if operator_resource:
                operator_resource.record_reconciliation_error(str(e))
                self.state.update_resource(operator_resource)
            raise
    
    async def handle_delete(
        self,
        spec: Dict[str, Any],
        name: str,
        namespace: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Handle custom resource deletion."""
        logger.info("Handling resource deletion", operator=self.name, name=name)
        
        try:
            # Get existing operator resource
            operator_resource = self.state.get_resource(namespace, name)
            
            if operator_resource:
                # Call operator-specific deletion logic
                result = await self.delete_resource(operator_resource, spec, **kwargs)
                
                # Remove from state
                self.state.remove_resource(namespace, name)
                
                return result
            else:
                logger.warning("Resource not found in state during deletion", name=name)
                return None
            
        except Exception as e:
            logger.error("Resource deletion failed", 
                        operator=self.name, name=name, error=str(e))
            raise
    
    async def handle_status_check(
        self,
        spec: Dict[str, Any],
        name: str,
        namespace: str,
        **kwargs
    ) -> None:
        """Handle periodic status checks."""
        logger.debug("Handling status check", operator=self.name, name=name)
        
        try:
            # Get operator resource
            operator_resource = self.state.get_resource(namespace, name)
            
            if operator_resource:
                # Call operator-specific status check
                await self.check_resource_status(operator_resource, spec, **kwargs)
            
        except Exception as e:
            logger.error("Status check failed", 
                        operator=self.name, name=name, error=str(e))
    
    async def get_status(self) -> Dict[str, Any]:
        """Get operator status."""
        return {
            "name": self.name,
            "running": self.running,
            "crd": f"{self.crd_kind}.{self.crd_group}",
            "namespace": self.namespace,
            "reconcile_interval": self.reconcile_interval,
            "statistics": self.state.get_statistics()
        }
    
    # Abstract methods to be implemented by concrete operators
    
    @abstractmethod
    async def reconcile(
        self,
        resource: OperatorResource,
        resource_data: Dict[str, Any]
    ) -> None:
        """
        Reconcile a resource to desired state.
        
        This method should contain the main operator logic.
        """
        pass
    
    @abstractmethod
    async def create_resource(
        self,
        resource: OperatorResource,
        spec: Dict[str, Any],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Handle resource creation."""
        pass
    
    @abstractmethod
    async def update_resource(
        self,
        resource: OperatorResource,
        old: Dict[str, Any],
        new: Dict[str, Any],
        diff: List[Any],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Handle resource updates."""
        pass
    
    @abstractmethod
    async def delete_resource(
        self,
        resource: OperatorResource,
        spec: Dict[str, Any],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Handle resource deletion."""
        pass
    
    async def check_resource_status(
        self,
        resource: OperatorResource,
        spec: Dict[str, Any],
        **kwargs
    ) -> None:
        """Check resource status (optional override)."""
        pass
    
    async def on_startup(self) -> None:
        """Called when operator starts (optional override)."""
        pass
    
    async def on_shutdown(self) -> None:
        """Called when operator stops (optional override)."""
        pass