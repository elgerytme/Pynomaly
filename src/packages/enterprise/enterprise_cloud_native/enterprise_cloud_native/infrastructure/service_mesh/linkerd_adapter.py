"""
Linkerd Service Mesh Adapter

Provides integration with Linkerd service mesh for lightweight
service-to-service communication and observability.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID

from structlog import get_logger
import kubernetes.client

from ...domain.entities.service_mesh import (
    ServiceMeshConfiguration, ServiceMeshService, TrafficPolicy, 
    SecurityPolicy, ServiceMeshGateway, ServiceMeshType
)

logger = get_logger(__name__)


class LinkerdAdapter:
    """
    Linkerd service mesh adapter.
    
    Provides integration with Linkerd for traffic management,
    security policies, and observability.
    """
    
    def __init__(self, namespace: str = "linkerd"):
        self.namespace = namespace
        self.k8s_client = kubernetes.client.ApiClient()
        self.custom_objects_api = kubernetes.client.CustomObjectsApi()
        self.core_v1 = kubernetes.client.CoreV1Api()
        self.apps_v1 = kubernetes.client.AppsV1Api()
        
        # Linkerd API groups and versions
        self.policy_group = "policy.linkerd.io"
        self.policy_version = "v1beta1"
        
        logger.info("LinkerdAdapter initialized", namespace=namespace)
    
    async def install_linkerd(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> Dict[str, Any]:
        """Install Linkerd service mesh."""
        logger.info("Installing Linkerd", config_id=mesh_config.id)
        
        try:
            # Create Linkerd namespace if it doesn't exist
            await self._ensure_namespace(self.namespace)
            
            # Apply Linkerd control plane configuration
            await self._apply_linkerd_control_plane(mesh_config)
            
            # Configure data plane
            await self._configure_data_plane(mesh_config)
            
            # Set up default policies
            await self._setup_default_policies(mesh_config)
            
            return {
                "status": "installed",
                "components": ["linkerd-controller", "linkerd-proxy", "linkerd-viz"],
                "namespace": self.namespace
            }
            
        except Exception as e:
            logger.error("Failed to install Linkerd", error=str(e))
            raise
    
    async def create_server_policy(
        self,
        security_policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Create Linkerd Server policy."""
        logger.info("Creating Server policy", policy_id=security_policy.id)
        
        try:
            # Build Server policy spec
            server_spec = await self._build_server_policy_spec(security_policy)
            
            server_policy = {
                "apiVersion": f"{self.policy_group}/{self.policy_version}",
                "kind": "Server",
                "metadata": {
                    "name": f"{security_policy.name}-server",
                    "namespace": security_policy.target_services[0].split('/')[0] if '/' in security_policy.target_services[0] else "default",
                    "labels": {
                        "pynomaly.io/managed": "true",
                        "pynomaly.io/policy-id": str(security_policy.id)
                    }
                },
                "spec": server_spec
            }
            
            # Apply Server policy
            result = await self._apply_custom_resource(
                self.policy_group,
                self.policy_version,
                "servers",
                server_policy
            )
            
            logger.info("Server policy created", name=server_policy["metadata"]["name"])
            return result
            
        except Exception as e:
            logger.error("Failed to create Server policy", error=str(e))
            raise
    
    async def create_server_authorization(
        self,
        security_policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Create Linkerd ServerAuthorization policy."""
        logger.info("Creating ServerAuthorization", policy_id=security_policy.id)
        
        try:
            # Build ServerAuthorization spec
            auth_spec = await self._build_server_authorization_spec(security_policy)
            
            server_authorization = {
                "apiVersion": f"{self.policy_group}/{self.policy_version}",
                "kind": "ServerAuthorization",
                "metadata": {
                    "name": f"{security_policy.name}-authz",
                    "namespace": security_policy.target_services[0].split('/')[0] if '/' in security_policy.target_services[0] else "default",
                    "labels": {
                        "pynomaly.io/managed": "true",
                        "pynomaly.io/policy-id": str(security_policy.id)
                    }
                },
                "spec": auth_spec
            }
            
            # Apply ServerAuthorization
            result = await self._apply_custom_resource(
                self.policy_group,
                self.policy_version,
                "serverauthorizations",
                server_authorization
            )
            
            logger.info("ServerAuthorization created", name=server_authorization["metadata"]["name"])
            return result
            
        except Exception as e:
            logger.error("Failed to create ServerAuthorization", error=str(e))
            raise
    
    async def create_traffic_split(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Create Linkerd TrafficSplit for canary deployments."""
        logger.info("Creating TrafficSplit", policy_id=traffic_policy.id)
        
        try:
            # Build TrafficSplit spec
            split_spec = await self._build_traffic_split_spec(traffic_policy)
            
            traffic_split = {
                "apiVersion": "split.smi-spec.io/v1alpha1",
                "kind": "TrafficSplit",
                "metadata": {
                    "name": f"{traffic_policy.name}-split",
                    "namespace": traffic_policy.target_services[0].split('/')[0] if '/' in traffic_policy.target_services[0] else "default",
                    "labels": {
                        "pynomaly.io/managed": "true",
                        "pynomaly.io/policy-id": str(traffic_policy.id)
                    }
                },
                "spec": split_spec
            }
            
            # Apply TrafficSplit
            result = await self._apply_custom_resource(
                "split.smi-spec.io",
                "v1alpha1",
                "trafficsplits",
                traffic_split
            )
            
            logger.info("TrafficSplit created", name=traffic_split["metadata"]["name"])
            return result
            
        except Exception as e:
            logger.error("Failed to create TrafficSplit", error=str(e))
            raise
    
    async def create_http_route(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Create Linkerd HTTPRoute for advanced routing."""
        logger.info("Creating HTTPRoute", policy_id=traffic_policy.id)
        
        try:
            # Build HTTPRoute spec
            route_spec = await self._build_http_route_spec(traffic_policy)
            
            http_route = {
                "apiVersion": f"{self.policy_group}/{self.policy_version}",
                "kind": "HTTPRoute",
                "metadata": {
                    "name": f"{traffic_policy.name}-route",
                    "namespace": traffic_policy.target_services[0].split('/')[0] if '/' in traffic_policy.target_services[0] else "default",
                    "labels": {
                        "pynomaly.io/managed": "true",
                        "pynomaly.io/policy-id": str(traffic_policy.id)
                    }
                },
                "spec": route_spec
            }
            
            # Apply HTTPRoute
            result = await self._apply_custom_resource(
                self.policy_group,
                self.policy_version,
                "httproutes",
                http_route
            )
            
            logger.info("HTTPRoute created", name=http_route["metadata"]["name"])
            return result
            
        except Exception as e:
            logger.error("Failed to create HTTPRoute", error=str(e))
            raise
    
    async def get_mesh_status(self) -> Dict[str, Any]:
        """Get Linkerd mesh status."""
        try:
            # Check control plane status
            controller_status = await self._get_deployment_status("linkerd-controller")
            
            # Check proxy injector status
            injector_status = await self._get_deployment_status("linkerd-proxy-injector")
            
            # Get proxy status
            proxy_status = await self._get_proxy_status()
            
            # Check viz extension
            viz_status = await self._get_deployment_status("web", namespace="linkerd-viz")
            
            return {
                "control_plane": {
                    "controller": controller_status,
                    "proxy_injector": injector_status
                },
                "viz": viz_status,
                "proxies": proxy_status,
                "mesh_type": "linkerd"
            }
            
        except Exception as e:
            logger.error("Failed to get mesh status", error=str(e))
            raise
    
    async def inject_proxy(self, namespace: str) -> Dict[str, Any]:
        """Enable proxy injection for a namespace."""
        logger.info("Enabling proxy injection", namespace=namespace)
        
        try:
            # Add linkerd.io/inject: enabled annotation to namespace
            body = {
                "metadata": {
                    "annotations": {
                        "linkerd.io/inject": "enabled"
                    }
                }
            }
            
            result = self.core_v1.patch_namespace(
                name=namespace,
                body=body
            )
            
            logger.info("Proxy injection enabled", namespace=namespace)
            return {"status": "enabled", "namespace": namespace}
            
        except Exception as e:
            logger.error("Failed to enable proxy injection", namespace=namespace, error=str(e))
            raise
    
    # Private helper methods
    
    async def _ensure_namespace(self, namespace: str) -> None:
        """Ensure namespace exists."""
        try:
            self.core_v1.read_namespace(namespace)
            logger.debug("Namespace exists", namespace=namespace)
        except kubernetes.client.rest.ApiException as e:
            if e.status == 404:
                # Create namespace
                ns = kubernetes.client.V1Namespace(
                    metadata=kubernetes.client.V1ObjectMeta(
                        name=namespace,
                        annotations={
                            "linkerd.io/inject": "enabled" if namespace != self.namespace else None
                        }
                    )
                )
                self.core_v1.create_namespace(ns)
                logger.info("Namespace created", namespace=namespace)
            else:
                raise
    
    async def _apply_linkerd_control_plane(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> None:
        """Apply Linkerd control plane configuration."""
        logger.info("Applying Linkerd control plane configuration")
        # This would typically involve applying Linkerd control plane manifests
        pass
    
    async def _configure_data_plane(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> None:
        """Configure Linkerd data plane settings."""
        logger.info("Configuring Linkerd data plane")
        
        # Configure proxy injection
        if mesh_config.sidecar_injection_enabled:
            await self._enable_proxy_injection(mesh_config)
    
    async def _enable_proxy_injection(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> None:
        """Enable automatic proxy injection."""
        # This would configure proxy injection settings
        pass
    
    async def _setup_default_policies(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> None:
        """Set up default Linkerd policies."""
        logger.info("Setting up default Linkerd policies")
        
        if mesh_config.mtls_enabled:
            await self._create_default_mtls_policy(mesh_config)
    
    async def _create_default_mtls_policy(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> None:
        """Create default mTLS policy."""
        # Linkerd automatically provides mTLS for meshed services
        pass
    
    async def _build_server_policy_spec(
        self,
        security_policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Build Server policy specification."""
        spec = {
            "podSelector": {
                "matchLabels": security_policy.match_labels or {}
            },
            "port": "http"  # Default port name
        }
        
        # Add specific port configuration if available
        if security_policy.authentication_config:
            port_config = security_policy.authentication_config.get("port")
            if port_config:
                spec["port"] = port_config
        
        return spec
    
    async def _build_server_authorization_spec(
        self,
        security_policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Build ServerAuthorization specification."""
        spec = {
            "server": {
                "name": f"{security_policy.name}-server"
            },
            "requiredRoutes": []
        }
        
        # Add authorization rules
        for rule in security_policy.authorization_rules:
            route = {
                "pathRegex": rule.get("path", "/.*"),
                "method": rule.get("method", "GET")
            }
            spec["requiredRoutes"].append(route)
        
        # Add client selection
        if security_policy.match_labels:
            spec["client"] = {
                "podSelector": {
                    "matchLabels": security_policy.match_labels
                }
            }
        
        return spec
    
    async def _build_traffic_split_spec(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Build TrafficSplit specification."""
        split_config = traffic_policy.traffic_split_config or {}
        
        spec = {
            "service": traffic_policy.target_services[0],
            "backends": [
                {
                    "service": f"{traffic_policy.target_services[0]}-stable",
                    "weight": split_config.get("stable_weight", 90)
                },
                {
                    "service": f"{traffic_policy.target_services[0]}-canary", 
                    "weight": split_config.get("canary_weight", 10)
                }
            ]
        }
        
        return spec
    
    async def _build_http_route_spec(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Build HTTPRoute specification."""
        spec = {
            "parentRefs": [
                {
                    "name": f"{traffic_policy.name}-server",
                    "kind": "Server",
                    "group": self.policy_group
                }
            ],
            "rules": []
        }
        
        # Add retry configuration
        retry_config = traffic_policy.get_retry_config()
        if retry_config:
            rule = {
                "backendRefs": [
                    {
                        "name": traffic_policy.target_services[0],
                        "port": 80
                    }
                ],
                "timeouts": {
                    "request": retry_config.get("perTryTimeout", "30s")
                }
            }
            spec["rules"].append(rule)
        
        return spec
    
    async def _apply_custom_resource(
        self,
        group: str,
        version: str,
        plural: str,
        resource: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply custom resource to Kubernetes."""
        namespace = resource["metadata"].get("namespace")
        
        try:
            if namespace:
                result = self.custom_objects_api.create_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    body=resource
                )
            else:
                result = self.custom_objects_api.create_cluster_custom_object(
                    group=group,
                    version=version,
                    plural=plural,
                    body=resource
                )
            
            return result
            
        except kubernetes.client.rest.ApiException as e:
            if e.status == 409:
                # Resource already exists, try to update
                logger.info("Resource exists, updating", name=resource["metadata"]["name"])
                return await self._update_custom_resource(group, version, plural, resource)
            else:
                raise
    
    async def _update_custom_resource(
        self,
        group: str,
        version: str,
        plural: str,
        resource: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update existing custom resource."""
        namespace = resource["metadata"].get("namespace")
        name = resource["metadata"]["name"]
        
        if namespace:
            return self.custom_objects_api.replace_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=name,
                body=resource
            )
        else:
            return self.custom_objects_api.replace_cluster_custom_object(
                group=group,
                version=version,
                plural=plural,
                name=name,
                body=resource
            )
    
    async def _get_deployment_status(self, deployment_name: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get deployment status."""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace or self.namespace
            )
            
            return {
                "name": deployment_name,
                "ready": deployment.status.ready_replicas or 0,
                "replicas": deployment.status.replicas or 0,
                "available": deployment.status.available_replicas or 0,
                "status": "Ready" if deployment.status.ready_replicas == deployment.status.replicas else "NotReady"
            }
            
        except Exception as e:
            logger.error("Failed to get deployment status", deployment=deployment_name, error=str(e))
            return {"name": deployment_name, "status": "Unknown", "error": str(e)}
    
    async def _get_proxy_status(self) -> Dict[str, Any]:
        """Get proxy status from all sidecars."""
        # This would query Linkerd control plane for proxy status
        return {
            "total_proxies": 0,
            "healthy_proxies": 0,
            "version_info": {}
        }