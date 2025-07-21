"""
Envoy Proxy Adapter

Provides integration with standalone Envoy proxy for advanced
load balancing, traffic management, and observability.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
import json

from structlog import get_logger
import kubernetes.client

from ...domain.entities.service_mesh import (
    ServiceMeshConfiguration, ServiceMeshService, TrafficPolicy,
    SecurityPolicy, ServiceMeshGateway, ServiceMeshType
)

logger = get_logger(__name__)


class EnvoyAdapter:
    """
    Envoy proxy adapter.
    
    Provides integration with standalone Envoy proxy deployments
    for advanced traffic management and load balancing.
    """
    
    def __init__(self, namespace: str = "envoy-system"):
        self.namespace = namespace
        self.k8s_client = kubernetes.client.ApiClient()
        self.core_v1 = kubernetes.client.CoreV1Api()
        self.apps_v1 = kubernetes.client.AppsV1Api()
        
        logger.info("EnvoyAdapter initialized", namespace=namespace)
    
    async def deploy_envoy_proxy(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> Dict[str, Any]:
        """Deploy Envoy proxy."""
        logger.info("Deploying Envoy proxy", config_id=mesh_config.id)
        
        try:
            # Create namespace if it doesn't exist
            await self._ensure_namespace(self.namespace)
            
            # Create Envoy configuration
            config_map = await self._create_envoy_config(mesh_config)
            
            # Deploy Envoy proxy
            deployment = await self._create_envoy_deployment(mesh_config, config_map["metadata"]["name"])
            
            # Create service
            service = await self._create_envoy_service(mesh_config)
            
            return {
                "status": "deployed",
                "components": {
                    "configmap": config_map["metadata"]["name"],
                    "deployment": deployment["metadata"]["name"],
                    "service": service["metadata"]["name"]
                },
                "namespace": self.namespace
            }
            
        except Exception as e:
            logger.error("Failed to deploy Envoy proxy", error=str(e))
            raise
    
    async def update_envoy_config(
        self,
        mesh_config: ServiceMeshConfiguration,
        traffic_policies: List[TrafficPolicy],
        security_policies: List[SecurityPolicy]
    ) -> Dict[str, Any]:
        """Update Envoy configuration with new policies."""
        logger.info("Updating Envoy configuration", config_id=mesh_config.id)
        
        try:
            # Generate new Envoy configuration
            envoy_config = await self._generate_envoy_config(
                mesh_config, traffic_policies, security_policies
            )
            
            # Update ConfigMap
            config_map = await self._update_envoy_config_map(mesh_config.name, envoy_config)
            
            # Restart Envoy deployment to pick up new config
            await self._restart_envoy_deployment(mesh_config.name)
            
            return {
                "status": "updated",
                "config_version": config_map["metadata"]["resourceVersion"]
            }
            
        except Exception as e:
            logger.error("Failed to update Envoy configuration", error=str(e))
            raise
    
    async def create_load_balancer_config(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Create Envoy load balancer configuration."""
        logger.info("Creating load balancer config", policy_id=traffic_policy.id)
        
        try:
            # Build cluster configuration
            cluster_config = await self._build_cluster_config(traffic_policy)
            
            # Build route configuration
            route_config = await self._build_route_config(traffic_policy)
            
            # Build listener configuration
            listener_config = await self._build_listener_config(traffic_policy)
            
            return {
                "clusters": [cluster_config],
                "routes": [route_config],
                "listeners": [listener_config]
            }
            
        except Exception as e:
            logger.error("Failed to create load balancer config", error=str(e))
            raise
    
    async def get_proxy_stats(self, proxy_name: str) -> Dict[str, Any]:
        """Get Envoy proxy statistics."""
        logger.debug("Getting proxy stats", proxy=proxy_name)
        
        try:
            # Get proxy pod
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app=envoy-proxy,instance={proxy_name}"
            )
            
            if not pods.items:
                return {"error": "Proxy pod not found"}
            
            pod = pods.items[0]
            
            # Get stats from admin interface (port 9901 by default)
            stats = await self._get_envoy_admin_stats(pod.metadata.name)
            
            return {
                "proxy": proxy_name,
                "pod": pod.metadata.name,
                "stats": stats,
                "status": pod.status.phase
            }
            
        except Exception as e:
            logger.error("Failed to get proxy stats", proxy=proxy_name, error=str(e))
            raise
    
    async def get_mesh_status(self) -> Dict[str, Any]:
        """Get Envoy mesh status."""
        try:
            # Get all Envoy deployments
            deployments = self.apps_v1.list_namespaced_deployment(
                namespace=self.namespace,
                label_selector="app=envoy-proxy"
            )
            
            proxy_status = []
            for deployment in deployments.items:
                status = {
                    "name": deployment.metadata.name,
                    "ready": deployment.status.ready_replicas or 0,
                    "replicas": deployment.status.replicas or 0,
                    "status": "Ready" if deployment.status.ready_replicas == deployment.status.replicas else "NotReady"
                }
                proxy_status.append(status)
            
            return {
                "proxies": proxy_status,
                "total_proxies": len(proxy_status),
                "healthy_proxies": sum(1 for p in proxy_status if p["status"] == "Ready"),
                "mesh_type": "envoy"
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
                    metadata=kubernetes.client.V1ObjectMeta(name=namespace)
                )
                self.core_v1.create_namespace(ns)
                logger.info("Namespace created", namespace=namespace)
            else:
                raise
    
    async def _create_envoy_config(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> Dict[str, Any]:
        """Create Envoy configuration ConfigMap."""
        logger.info("Creating Envoy configuration")
        
        # Generate basic Envoy configuration
        envoy_config = await self._generate_basic_envoy_config(mesh_config)
        
        config_map = kubernetes.client.V1ConfigMap(
            metadata=kubernetes.client.V1ObjectMeta(
                name=f"{mesh_config.name}-envoy-config",
                namespace=self.namespace,
                labels={
                    "app": "envoy-proxy",
                    "pynomaly.io/managed": "true"
                }
            ),
            data={
                "envoy.yaml": json.dumps(envoy_config, indent=2)
            }
        )
        
        result = self.core_v1.create_namespaced_config_map(
            namespace=self.namespace,
            body=config_map
        )
        
        return result.to_dict()
    
    async def _create_envoy_deployment(
        self,
        mesh_config: ServiceMeshConfiguration,
        config_map_name: str
    ) -> Dict[str, Any]:
        """Create Envoy proxy deployment."""
        logger.info("Creating Envoy deployment")
        
        # Build deployment spec
        deployment = kubernetes.client.V1Deployment(
            metadata=kubernetes.client.V1ObjectMeta(
                name=f"{mesh_config.name}-envoy-proxy",
                namespace=self.namespace,
                labels={
                    "app": "envoy-proxy",
                    "instance": mesh_config.name,
                    "pynomaly.io/managed": "true"
                }
            ),
            spec=kubernetes.client.V1DeploymentSpec(
                replicas=mesh_config.replicas.get("proxy", 2),
                selector=kubernetes.client.V1LabelSelector(
                    match_labels={
                        "app": "envoy-proxy",
                        "instance": mesh_config.name
                    }
                ),
                template=kubernetes.client.V1PodTemplateSpec(
                    metadata=kubernetes.client.V1ObjectMeta(
                        labels={
                            "app": "envoy-proxy",
                            "instance": mesh_config.name
                        }
                    ),
                    spec=kubernetes.client.V1PodSpec(
                        containers=[
                            kubernetes.client.V1Container(
                                name="envoy",
                                image="envoyproxy/envoy:v1.28-latest",
                                ports=[
                                    kubernetes.client.V1ContainerPort(container_port=8080, name="http"),
                                    kubernetes.client.V1ContainerPort(container_port=8443, name="https"),
                                    kubernetes.client.V1ContainerPort(container_port=9901, name="admin")
                                ],
                                volume_mounts=[
                                    kubernetes.client.V1VolumeMount(
                                        name="envoy-config",
                                        mount_path="/etc/envoy"
                                    )
                                ],
                                command=[
                                    "envoy",
                                    "-c", "/etc/envoy/envoy.yaml",
                                    "--service-cluster", mesh_config.name,
                                    "--service-node", "envoy"
                                ],
                                resources=kubernetes.client.V1ResourceRequirements(
                                    requests={
                                        "cpu": mesh_config.resource_requirements.get("proxy", {}).get("requests", {}).get("cpu", "100m"),
                                        "memory": mesh_config.resource_requirements.get("proxy", {}).get("requests", {}).get("memory", "128Mi")
                                    },
                                    limits={
                                        "cpu": mesh_config.resource_requirements.get("proxy", {}).get("limits", {}).get("cpu", "200m"),
                                        "memory": mesh_config.resource_requirements.get("proxy", {}).get("limits", {}).get("memory", "256Mi")
                                    }
                                )
                            )
                        ],
                        volumes=[
                            kubernetes.client.V1Volume(
                                name="envoy-config",
                                config_map=kubernetes.client.V1ConfigMapVolumeSource(
                                    name=config_map_name
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        result = self.apps_v1.create_namespaced_deployment(
            namespace=self.namespace,
            body=deployment
        )
        
        return result.to_dict()
    
    async def _create_envoy_service(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> Dict[str, Any]:
        """Create Envoy service."""
        logger.info("Creating Envoy service")
        
        service = kubernetes.client.V1Service(
            metadata=kubernetes.client.V1ObjectMeta(
                name=f"{mesh_config.name}-envoy-proxy",
                namespace=self.namespace,
                labels={
                    "app": "envoy-proxy",
                    "instance": mesh_config.name
                }
            ),
            spec=kubernetes.client.V1ServiceSpec(
                selector={
                    "app": "envoy-proxy",
                    "instance": mesh_config.name
                },
                ports=[
                    kubernetes.client.V1ServicePort(
                        name="http",
                        port=8080,
                        target_port=8080
                    ),
                    kubernetes.client.V1ServicePort(
                        name="https",
                        port=8443,
                        target_port=8443
                    ),
                    kubernetes.client.V1ServicePort(
                        name="admin",
                        port=9901,
                        target_port=9901
                    )
                ],
                type="LoadBalancer" if mesh_config.ingress_gateway_enabled else "ClusterIP"
            )
        )
        
        result = self.core_v1.create_namespaced_service(
            namespace=self.namespace,
            body=service
        )
        
        return result.to_dict()
    
    async def _generate_basic_envoy_config(
        self,
        mesh_config: ServiceMeshConfiguration
    ) -> Dict[str, Any]:
        """Generate basic Envoy configuration."""
        config = {
            "admin": {
                "address": {
                    "socket_address": {
                        "address": "0.0.0.0",
                        "port_value": 9901
                    }
                }
            },
            "static_resources": {
                "listeners": [
                    {
                        "name": "http_listener",
                        "address": {
                            "socket_address": {
                                "address": "0.0.0.0",
                                "port_value": 8080
                            }
                        },
                        "filter_chains": [
                            {
                                "filters": [
                                    {
                                        "name": "envoy.filters.network.http_connection_manager",
                                        "typed_config": {
                                            "@type": "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager",
                                            "stat_prefix": "ingress_http",
                                            "codec_type": "AUTO",
                                            "route_config": {
                                                "name": "local_route",
                                                "virtual_hosts": [
                                                    {
                                                        "name": "backend",
                                                        "domains": ["*"],
                                                        "routes": [
                                                            {
                                                                "match": {
                                                                    "prefix": "/"
                                                                },
                                                                "route": {
                                                                    "cluster": "service_backend"
                                                                }
                                                            }
                                                        ]
                                                    }
                                                ]
                                            },
                                            "http_filters": [
                                                {
                                                    "name": "envoy.filters.http.router"
                                                }
                                            ]
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ],
                "clusters": [
                    {
                        "name": "service_backend",
                        "type": "STRICT_DNS",
                        "lb_policy": "ROUND_ROBIN",
                        "load_assignment": {
                            "cluster_name": "service_backend",
                            "endpoints": [
                                {
                                    "lb_endpoints": [
                                        {
                                            "endpoint": {
                                                "address": {
                                                    "socket_address": {
                                                        "address": "127.0.0.1",
                                                        "port_value": 8000
                                                    }
                                                }
                                            }
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        return config
    
    async def _generate_envoy_config(
        self,
        mesh_config: ServiceMeshConfiguration,
        traffic_policies: List[TrafficPolicy],
        security_policies: List[SecurityPolicy]
    ) -> Dict[str, Any]:
        """Generate comprehensive Envoy configuration."""
        # Start with basic config
        config = await self._generate_basic_envoy_config(mesh_config)
        
        # Add traffic policies
        for policy in traffic_policies:
            await self._add_traffic_policy_to_config(config, policy)
        
        # Add security policies  
        for policy in security_policies:
            await self._add_security_policy_to_config(config, policy)
        
        return config
    
    async def _add_traffic_policy_to_config(
        self,
        config: Dict[str, Any],
        policy: TrafficPolicy
    ) -> None:
        """Add traffic policy configuration to Envoy config."""
        # Add circuit breaker configuration
        if policy.circuit_breaker_config:
            cb_config = policy.get_circuit_breaker_config()
            # Add circuit breaker to cluster configuration
            pass
        
        # Add retry policy
        if policy.retry_config:
            retry_config = policy.get_retry_config()
            # Add retry policy to route configuration
            pass
    
    async def _add_security_policy_to_config(
        self,
        config: Dict[str, Any],
        policy: SecurityPolicy
    ) -> None:
        """Add security policy configuration to Envoy config."""
        # Add authentication filters
        if policy.authentication_config:
            # Add auth filters to HTTP connection manager
            pass
        
        # Add TLS configuration
        if policy.requires_mtls():
            # Add TLS configuration to listeners
            pass
    
    async def _build_cluster_config(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Build Envoy cluster configuration."""
        cluster = {
            "name": f"{traffic_policy.name}_cluster",
            "type": "STRICT_DNS",
            "lb_policy": traffic_policy.load_balancer_type or "ROUND_ROBIN",
            "load_assignment": {
                "cluster_name": f"{traffic_policy.name}_cluster",
                "endpoints": []
            }
        }
        
        # Add circuit breaker
        if traffic_policy.circuit_breaker_config:
            cb_config = traffic_policy.get_circuit_breaker_config()
            cluster["circuit_breakers"] = {
                "thresholds": [
                    {
                        "priority": "DEFAULT",
                        "max_connections": cb_config.get("maxConnections", 1024),
                        "max_pending_requests": cb_config.get("maxPendingRequests", 1024),
                        "max_requests": cb_config.get("maxRequests", 1024),
                        "max_retries": cb_config.get("maxRetries", 3)
                    }
                ]
            }
        
        return cluster
    
    async def _build_route_config(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Build Envoy route configuration."""
        route = {
            "name": f"{traffic_policy.name}_route",
            "virtual_hosts": [
                {
                    "name": f"{traffic_policy.name}_vhost",
                    "domains": ["*"],
                    "routes": [
                        {
                            "match": {"prefix": "/"},
                            "route": {
                                "cluster": f"{traffic_policy.name}_cluster"
                            }
                        }
                    ]
                }
            ]
        }
        
        # Add retry policy
        if traffic_policy.retry_config:
            retry_config = traffic_policy.get_retry_config()
            for vhost in route["virtual_hosts"]:
                for route_rule in vhost["routes"]:
                    route_rule["route"]["retry_policy"] = {
                        "retry_on": retry_config.get("retryOn", "5xx"),
                        "num_retries": retry_config.get("attempts", 3),
                        "per_try_timeout": retry_config.get("perTryTimeout", "2s")
                    }
        
        return route
    
    async def _build_listener_config(
        self,
        traffic_policy: TrafficPolicy
    ) -> Dict[str, Any]:
        """Build Envoy listener configuration."""
        listener = {
            "name": f"{traffic_policy.name}_listener",
            "address": {
                "socket_address": {
                    "address": "0.0.0.0",
                    "port_value": 8080
                }
            },
            "filter_chains": [
                {
                    "filters": [
                        {
                            "name": "envoy.filters.network.http_connection_manager",
                            "typed_config": {
                                "@type": "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager",
                                "stat_prefix": f"{traffic_policy.name}_http",
                                "codec_type": "AUTO",
                                "route_config": {
                                    "name": f"{traffic_policy.name}_route"
                                },
                                "http_filters": [
                                    {"name": "envoy.filters.http.router"}
                                ]
                            }
                        }
                    ]
                }
            ]
        }
        
        return listener
    
    async def _update_envoy_config_map(
        self,
        instance_name: str,
        envoy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update Envoy ConfigMap with new configuration."""
        config_map_name = f"{instance_name}-envoy-config"
        
        config_map = kubernetes.client.V1ConfigMap(
            metadata=kubernetes.client.V1ObjectMeta(
                name=config_map_name,
                namespace=self.namespace
            ),
            data={
                "envoy.yaml": json.dumps(envoy_config, indent=2)
            }
        )
        
        result = self.core_v1.replace_namespaced_config_map(
            name=config_map_name,
            namespace=self.namespace,
            body=config_map
        )
        
        return result.to_dict()
    
    async def _restart_envoy_deployment(self, instance_name: str) -> None:
        """Restart Envoy deployment to pick up configuration changes."""
        deployment_name = f"{instance_name}-envoy-proxy"
        
        # Patch deployment with restart annotation
        patch_body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": str(kubernetes.client.api_client.datetime.utcnow())
                        }
                    }
                }
            }
        }
        
        self.apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=self.namespace,
            body=patch_body
        )
        
        logger.info("Envoy deployment restarted", deployment=deployment_name)
    
    async def _get_envoy_admin_stats(self, pod_name: str) -> Dict[str, Any]:
        """Get stats from Envoy admin interface."""
        # This would make an HTTP request to the pod's admin interface
        # For now, return mock stats
        return {
            "listeners": 1,
            "clusters": 1,
            "routes": 1,
            "connections": 10,
            "requests": 1000
        }