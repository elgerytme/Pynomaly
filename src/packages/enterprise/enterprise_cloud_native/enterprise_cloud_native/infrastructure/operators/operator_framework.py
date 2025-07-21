"""
Kubernetes Operator Framework Integration

Provides integration with Kopf and other operator frameworks for managing
custom resources and reconciliation loops in Kubernetes.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from uuid import UUID

from structlog import get_logger
import kopf
import kubernetes.client
from kubernetes import config

from ...domain.entities.kubernetes_resource import OperatorResource, OperatorState

logger = get_logger(__name__)


class OperatorFramework:
    """
    Kubernetes operator framework integration.
    
    Provides a high-level interface for creating and managing
    Kubernetes operators with custom resource definitions.
    """
    
    def __init__(
        self,
        namespace: Optional[str] = None,
        cluster_scoped: bool = False,
        peering_name: Optional[str] = None
    ):
        self.namespace = namespace
        self.cluster_scoped = cluster_scoped
        self.peering_name = peering_name
        self.operators: Dict[str, 'KubernetesOperator'] = {}
        self.reconcilers: Dict[str, Callable] = {}
        self.k8s_client = None
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes configuration")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logger.info("Loaded local Kubernetes configuration")
            except config.ConfigException:
                logger.error("Failed to load Kubernetes configuration")
        
        self.k8s_client = kubernetes.client.ApiClient()
        
        logger.info("OperatorFramework initialized", 
                   namespace=namespace, cluster_scoped=cluster_scoped)
    
    def register_operator(
        self,
        name: str,
        crd_group: str,
        crd_version: str,
        crd_kind: str,
        operator_instance: 'KubernetesOperator'
    ) -> None:
        """Register a Kubernetes operator."""
        logger.info("Registering operator", name=name, crd=f"{crd_kind}.{crd_group}")
        
        self.operators[name] = operator_instance
        
        # Register reconciler handlers with Kopf
        self._register_kopf_handlers(name, crd_group, crd_version, crd_kind, operator_instance)
        
        logger.info("Operator registered successfully", name=name)
    
    def _register_kopf_handlers(
        self,
        name: str,
        crd_group: str,
        crd_version: str,
        crd_kind: str,
        operator: 'KubernetesOperator'
    ) -> None:
        """Register Kopf handlers for operator."""
        
        @kopf.on.create(crd_group, crd_version, crd_kind)
        async def create_handler(spec, name, namespace, logger, **kwargs):
            """Handle custom resource creation."""
            try:
                logger.info("Creating custom resource", name=name, namespace=namespace)
                result = await operator.handle_create(spec, name, namespace, **kwargs)
                logger.info("Custom resource created successfully", name=name)
                return result
            except Exception as e:
                logger.error("Failed to create custom resource", 
                           name=name, error=str(e))
                raise kopf.PermanentError(f"Creation failed: {str(e)}")
        
        @kopf.on.update(crd_group, crd_version, crd_kind)
        async def update_handler(spec, old, new, diff, name, namespace, logger, **kwargs):
            """Handle custom resource updates."""
            try:
                logger.info("Updating custom resource", name=name, namespace=namespace)
                result = await operator.handle_update(spec, old, new, diff, name, namespace, **kwargs)
                logger.info("Custom resource updated successfully", name=name)
                return result
            except Exception as e:
                logger.error("Failed to update custom resource", 
                           name=name, error=str(e))
                raise kopf.TemporaryError(f"Update failed: {str(e)}", delay=60)
        
        @kopf.on.delete(crd_group, crd_version, crd_kind)
        async def delete_handler(spec, name, namespace, logger, **kwargs):
            """Handle custom resource deletion."""
            try:
                logger.info("Deleting custom resource", name=name, namespace=namespace)
                result = await operator.handle_delete(spec, name, namespace, **kwargs)
                logger.info("Custom resource deleted successfully", name=name)
                return result
            except Exception as e:
                logger.error("Failed to delete custom resource", 
                           name=name, error=str(e))
                raise kopf.TemporaryError(f"Deletion failed: {str(e)}", delay=30)
        
        @kopf.timer(crd_group, crd_version, crd_kind, interval=300)
        async def status_handler(spec, name, namespace, logger, **kwargs):
            """Handle periodic status updates."""
            try:
                logger.debug("Checking resource status", name=name, namespace=namespace)
                await operator.handle_status_check(spec, name, namespace, **kwargs)
            except Exception as e:
                logger.error("Status check failed", name=name, error=str(e))
        
        # Store handlers for reference
        self.reconcilers[name] = {
            'create': create_handler,
            'update': update_handler,
            'delete': delete_handler,
            'status': status_handler
        }
    
    async def start_operators(self) -> None:
        """Start all registered operators."""
        logger.info("Starting Kubernetes operators", count=len(self.operators))
        
        # Configure Kopf
        kopf_config = kopf.configure(
            verbose=True,
            log_level='INFO'
        )
        
        if self.peering_name:
            kopf_config = kopf_config.peering.name(self.peering_name)
        
        if self.namespace:
            kopf_config = kopf_config.watching.namespace(self.namespace)
        
        # Start operators
        for name, operator in self.operators.items():
            logger.info("Starting operator", name=name)
            await operator.start()
        
        logger.info("All operators started successfully")
    
    async def stop_operators(self) -> None:
        """Stop all operators."""
        logger.info("Stopping Kubernetes operators", count=len(self.operators))
        
        for name, operator in self.operators.items():
            logger.info("Stopping operator", name=name)
            await operator.stop()
        
        logger.info("All operators stopped")
    
    async def get_operator_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get operator status."""
        if name not in self.operators:
            return None
        
        operator = self.operators[name]
        return await operator.get_status()
    
    async def get_all_operators_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all operators."""
        status = {}
        
        for name, operator in self.operators.items():
            try:
                status[name] = await operator.get_status()
            except Exception as e:
                logger.error("Failed to get operator status", name=name, error=str(e))
                status[name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return status
    
    def get_operator(self, name: str) -> Optional['KubernetesOperator']:
        """Get operator instance by name."""
        return self.operators.get(name)
    
    def list_operators(self) -> List[str]:
        """List registered operator names."""
        return list(self.operators.keys())


class ReconcilerState:
    """State management for reconciliation loops."""
    
    def __init__(self):
        self.resources: Dict[str, OperatorResource] = {}
        self.reconciliation_count = 0
        self.last_reconciliation = None
        self.error_count = 0
        self.last_error = None
    
    def update_resource(self, resource: OperatorResource) -> None:
        """Update resource state."""
        key = f"{resource.namespace}/{resource.name}"
        self.resources[key] = resource
    
    def remove_resource(self, namespace: str, name: str) -> None:
        """Remove resource from state."""
        key = f"{namespace}/{name}"
        self.resources.pop(key, None)
    
    def get_resource(self, namespace: str, name: str) -> Optional[OperatorResource]:
        """Get resource by key."""
        key = f"{namespace}/{name}"
        return self.resources.get(key)
    
    def record_reconciliation(self, success: bool, error: Optional[str] = None) -> None:
        """Record reconciliation attempt."""
        self.reconciliation_count += 1
        self.last_reconciliation = datetime.utcnow()
        
        if success:
            self.error_count = 0
            self.last_error = None
        else:
            self.error_count += 1
            self.last_error = error
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reconciliation statistics."""
        return {
            "resources": len(self.resources),
            "reconciliation_count": self.reconciliation_count,
            "last_reconciliation": self.last_reconciliation.isoformat() if self.last_reconciliation else None,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "healthy_resources": sum(1 for r in self.resources.values() if r.is_ready()),
            "error_resources": sum(1 for r in self.resources.values() if r.operator_state == OperatorState.ERROR)
        }