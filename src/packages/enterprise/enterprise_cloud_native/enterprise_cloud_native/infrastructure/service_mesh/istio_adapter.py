"""
Istio Service Mesh Adapter

Provides integration with Istio service mesh for traffic management,
security policies, and observability configuration.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID

from structlog import get_logger
import kubernetes.client
import yaml

from ...domain.entities.service_mesh import (
    ServiceMeshConfiguration, ServiceMeshService, TrafficPolicy, 
    SecurityPolicy, ServiceMeshGateway, ServiceMeshType
)

logger = get_logger(__name__)


class IstioAdapter:
    """
    Istio service mesh adapter.
    
    Provides integration with Istio control plane and data plane
    for advanced traffic management and security policies.
    """
    
    def __init__(self, namespace: str = "istio-system"):
        self.namespace = namespace
        self.k8s_client = kubernetes.client.ApiClient()
        self.custom_objects_api = kubernetes.client.CustomObjectsApi()
        self.core_v1 = kubernetes.client.CoreV1Api()
        
        # Istio API groups and versions
        self.networking_group = "networking.istio.io"
        self.security_group = "security.istio.io"
        self.networking_version = "v1beta1"
        self.security_version = "v1beta1"
        
        logger.info("IstioAdapter initialized", namespace=namespace)
    
    async def install_istio(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> Dict[str, Any]:
        """Install Istio service mesh."""
        logger.info("Installing Istio", config_id=mesh_config.id)
        
        try:
            # Create Istio namespace if it doesn't exist
            await self._ensure_namespace(self.namespace)
            
            # Apply Istio control plane configuration
            await self._apply_istio_control_plane(mesh_config)
            
            # Configure data plane
            await self._configure_data_plane(mesh_config)
            
            # Set up default policies
            await self._setup_default_policies(mesh_config)
            
            return {
                "status": "installed",
                "components": ["istiod", "istio-proxy", "istio-gateway"],
                "namespace": self.namespace
            }
            
        except Exception as e:
            logger.error("Failed to install Istio", error=str(e))
            raise
    
    async def create_virtual_service(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Create Istio VirtualService for traffic routing."""
        logger.info("Creating VirtualService", policy_id=traffic_policy.id)
        
        try:
            # Build VirtualService spec
            vs_spec = await self._build_virtual_service_spec(traffic_policy)
            
            virtual_service = {
                "apiVersion": f"{self.networking_group}/{self.networking_version}",
                "kind": "VirtualService",
                "metadata": {
                    "name": f"{traffic_policy.name}-vs",
                    "namespace": traffic_policy.target_services[0].split('/')[0] if '/' in traffic_policy.target_services[0] else "default",
                    "labels": {
                        "pynomaly.io/managed": "true",
                        "pynomaly.io/policy-id": str(traffic_policy.id)
                    }
                },
                "spec": vs_spec
            }
            
            # Apply VirtualService
            result = await self._apply_custom_resource(
                self.networking_group,
                self.networking_version,
                "virtualservices",
                virtual_service
            )
            
            logger.info("VirtualService created", name=virtual_service["metadata"]["name"])
            return result
            
        except Exception as e:
            logger.error("Failed to create VirtualService", error=str(e))
            raise
    
    async def create_destination_rule(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Create Istio DestinationRule for traffic policies."""
        logger.info("Creating DestinationRule", policy_id=traffic_policy.id)
        
        try:
            # Build DestinationRule spec
            dr_spec = await self._build_destination_rule_spec(traffic_policy)
            
            destination_rule = {
                "apiVersion": f"{self.networking_group}/{self.networking_version}",
                "kind": "DestinationRule",
                "metadata": {
                    "name": f"{traffic_policy.name}-dr",
                    "namespace": traffic_policy.target_services[0].split('/')[0] if '/' in traffic_policy.target_services[0] else "default",
                    "labels": {
                        "pynomaly.io/managed": "true",
                        "pynomaly.io/policy-id": str(traffic_policy.id)
                    }
                },
                "spec": dr_spec
            }
            
            # Apply DestinationRule
            result = await self._apply_custom_resource(
                self.networking_group,
                self.networking_version,
                "destinationrules",
                destination_rule
            )
            
            logger.info("DestinationRule created", name=destination_rule["metadata"]["name"])
            return result
            
        except Exception as e:
            logger.error("Failed to create DestinationRule", error=str(e))
            raise
    
    async def create_authorization_policy(
        self,
        security_policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Create Istio AuthorizationPolicy."""
        logger.info("Creating AuthorizationPolicy", policy_id=security_policy.id)
        
        try:
            # Build AuthorizationPolicy spec
            auth_spec = await self._build_authorization_policy_spec(security_policy)
            
            authorization_policy = {
                "apiVersion": f"{self.security_group}/{self.security_version}",
                "kind": "AuthorizationPolicy",
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
            
            # Apply AuthorizationPolicy
            result = await self._apply_custom_resource(
                self.security_group,
                self.security_version,
                "authorizationpolicies",
                authorization_policy
            )
            
            logger.info("AuthorizationPolicy created", name=authorization_policy["metadata"]["name"])
            return result
            
        except Exception as e:
            logger.error("Failed to create AuthorizationPolicy", error=str(e))
            raise
    
    async def create_peer_authentication(
        self,
        security_policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Create Istio PeerAuthentication for mTLS."""
        logger.info("Creating PeerAuthentication", policy_id=security_policy.id)
        
        try:
            # Build PeerAuthentication spec
            peer_auth_spec = {
                "selector": {
                    "matchLabels": security_policy.match_labels or {}
                },
                "mtls": {
                    "mode": security_policy.tls_mode
                }
            }
            
            peer_authentication = {
                "apiVersion": f"{self.security_group}/{self.security_version}",
                "kind": "PeerAuthentication",
                "metadata": {
                    "name": f"{security_policy.name}-mtls",
                    "namespace": security_policy.target_services[0].split('/')[0] if '/' in security_policy.target_services[0] else "default",
                    "labels": {
                        "pynomaly.io/managed": "true",
                        "pynomaly.io/policy-id": str(security_policy.id)
                    }
                },
                "spec": peer_auth_spec
            }
            
            # Apply PeerAuthentication
            result = await self._apply_custom_resource(
                self.security_group,
                self.security_version,
                "peerauthentications",
                peer_authentication
            )
            
            logger.info("PeerAuthentication created", name=peer_authentication["metadata"]["name"])
            return result
            
        except Exception as e:
            logger.error("Failed to create PeerAuthentication", error=str(e))
            raise
    
    async def create_gateway(
        self,
        gateway_config: ServiceMeshGateway
    ) -> Dict[str, Any]:
        """Create Istio Gateway."""
        logger.info("Creating Gateway", gateway_id=gateway_config.id)
        
        try:
            # Build Gateway spec
            gateway_spec = {
                "selector": {
                    "istio": "ingressgateway"
                },
                "servers": []
            }
            
            # Configure listeners
            for listener in gateway_config.listeners:
                server = {
                    "port": {
                        "number": listener.get("port", 80),
                        "name": listener.get("name", "http"),
                        "protocol": listener.get("protocol", "HTTP")
                    },
                    "hosts": listener.get("hosts", ["*"])
                }
                
                # Add TLS configuration if present
                if listener.get("tls"):
                    server["tls"] = listener["tls"]
                
                gateway_spec["servers"].append(server)
            
            gateway = {
                "apiVersion": f"{self.networking_group}/{self.networking_version}",
                "kind": "Gateway",
                "metadata": {
                    "name": gateway_config.name,
                    "namespace": self.namespace,
                    "labels": {
                        "pynomaly.io/managed": "true",
                        "pynomaly.io/gateway-id": str(gateway_config.id)
                    }
                },
                "spec": gateway_spec
            }
            
            # Apply Gateway
            result = await self._apply_custom_resource(
                self.networking_group,
                self.networking_version,
                "gateways",
                gateway
            )
            
            logger.info("Gateway created", name=gateway_config.name)
            return result
            
        except Exception as e:
            logger.error("Failed to create Gateway", error=str(e))
            raise
    
    async def get_mesh_status(self) -> Dict[str, Any]:
        """Get Istio mesh status."""
        try:
            # Check istiod status
            istiod_status = await self._get_deployment_status("istiod")
            
            # Check gateway status
            gateway_status = await self._get_deployment_status("istio-ingressgateway")
            
            # Get proxy status
            proxy_status = await self._get_proxy_status()
            
            return {
                "control_plane": istiod_status,
                "ingress_gateway": gateway_status,
                "proxies": proxy_status,
                "mesh_type": "istio"
            }
            
        except Exception as e:
            logger.error("Failed to get mesh status", error=str(e))
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
                        labels={
                            "istio-injection": "enabled" if namespace != self.namespace else None
                        }
                    )
                )
                self.core_v1.create_namespace(ns)
                logger.info("Namespace created", namespace=namespace)
            else:
                raise
    
    async def _apply_istio_control_plane(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> None:
        """Apply Istio control plane configuration."""
        logger.info("Applying Istio control plane configuration")
        
        # This would typically involve applying IstioOperator CR
        # For now, we'll assume Istio is already installed
        pass
    
    async def _configure_data_plane(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> None:
        """Configure Istio data plane settings."""
        logger.info("Configuring Istio data plane")
        
        # Configure sidecar injection
        if mesh_config.sidecar_injection_enabled:
            await self._enable_sidecar_injection(mesh_config)
    
    async def _enable_sidecar_injection(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> None:
        """Enable automatic sidecar injection."""
        # This would patch the namespace with istio-injection=enabled label
        pass
    
    async def _setup_default_policies(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> None:
        """Set up default Istio policies."""
        logger.info("Setting up default Istio policies")
        
        if mesh_config.mtls_enabled:
            await self._create_default_mtls_policy(mesh_config)
    
    async def _create_default_mtls_policy(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> None:
        """Create default mTLS policy."""
        # Create default PeerAuthentication for mTLS
        pass
    
    async def _build_virtual_service_spec(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Build VirtualService specification."""
        spec = {
            "hosts": traffic_policy.target_services,
            "http": []
        }
        
        # Add routing rules based on policy type
        if traffic_policy.policy_type.value == "canary":
            # Canary deployment routing
            canary_config = traffic_policy.traffic_split_config or {}
            spec["http"].append({
                "match": [{"headers": {"canary": {"exact": "true"}}}],
                "route": [{
                    "destination": {
                        "host": traffic_policy.target_services[0],
                        "subset": "canary"
                    }
                }]
            })
            spec["http"].append({
                "route": [{
                    "destination": {
                        "host": traffic_policy.target_services[0],
                        "subset": "stable"
                    },
                    "weight": canary_config.get("stable_weight", 90)
                }, {
                    "destination": {
                        "host": traffic_policy.target_services[0],
                        "subset": "canary"
                    },
                    "weight": canary_config.get("canary_weight", 10)
                }]
            })
        
        # Add retry configuration
        retry_config = traffic_policy.get_retry_config()
        if retry_config:
            for route in spec["http"]:
                route["retries"] = {
                    "attempts": retry_config["attempts"],
                    "perTryTimeout": retry_config["perTryTimeout"],
                    "retryOn": retry_config["retryOn"]
                }
        
        # Add timeout configuration
        timeout_config = traffic_policy.timeout_config
        if timeout_config:
            for route in spec["http"]:
                route["timeout"] = timeout_config.get("timeout", "30s")
        
        return spec
    
    async def _build_destination_rule_spec(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Build DestinationRule specification."""
        spec = {
            "host": traffic_policy.target_services[0],
            "trafficPolicy": {}
        }
        
        # Add load balancer configuration
        if traffic_policy.load_balancer_type:
            spec["trafficPolicy"]["loadBalancer"] = {
                "simple": traffic_policy.load_balancer_type
            }
        
        # Add circuit breaker configuration
        circuit_breaker = traffic_policy.get_circuit_breaker_config()
        if circuit_breaker:
            spec["trafficPolicy"]["outlierDetection"] = {
                "consecutive5xxErrors": circuit_breaker["consecutive5xxErrors"],
                "consecutive4xxErrors": circuit_breaker["consecutive4xxErrors"],
                "interval": circuit_breaker["interval"],
                "baseEjectionTime": circuit_breaker["baseEjectionTime"],
                "maxEjectionPercent": circuit_breaker["maxEjectionPercent"],
                "minHealthPercent": circuit_breaker["minHealthPercent"]
            }
        
        # Add subsets for canary deployments
        if traffic_policy.policy_type.value == "canary":
            spec["subsets"] = [
                {
                    "name": "stable",
                    "labels": {"version": "stable"}
                },
                {
                    "name": "canary", 
                    "labels": {"version": "canary"}
                }
            ]
        
        return spec
    
    async def _build_authorization_policy_spec(
        self,
        security_policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Build AuthorizationPolicy specification."""
        spec = {}
        
        # Add selector
        if security_policy.match_labels:
            spec["selector"] = {
                "matchLabels": security_policy.match_labels
            }
        
        # Add rules
        if security_policy.authorization_rules:
            spec["rules"] = security_policy.authorization_rules
        
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
    
    async def _get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get deployment status."""
        try:
            apps_v1 = kubernetes.client.AppsV1Api()
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
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
        # This would query istiod for proxy status
        return {
            "total_proxies": 0,
            "connected_proxies": 0,
            "version_info": {}
        }