"""
Kubernetes Auto-scaling Adapter

Provides integration with Kubernetes HPA, VPA, and Cluster Autoscaler
for comprehensive auto-scaling capabilities.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
import asyncio

from structlog import get_logger
import kubernetes.client
from kubernetes.client.rest import ApiException

from ...domain.entities.autoscaling import (
    HorizontalPodAutoscaler, VerticalPodAutoscaler, ClusterAutoscaler,
    PredictiveScalingPolicy, ScalingEvent
)

logger = get_logger(__name__)


class KubernetesAutoscalingAdapter:
    """
    Kubernetes auto-scaling adapter.
    
    Provides integration with native Kubernetes auto-scaling components
    including HPA, VPA, and Cluster Autoscaler.
    """
    
    def __init__(self):
        self.k8s_client = kubernetes.client.ApiClient()
        self.autoscaling_v2 = kubernetes.client.AutoscalingV2Api()
        self.custom_objects_api = kubernetes.client.CustomObjectsApi()
        self.apps_v1 = kubernetes.client.AppsV1Api()
        self.core_v1 = kubernetes.client.CoreV1Api()
        
        # VPA API details
        self.vpa_group = "autoscaling.k8s.io"
        self.vpa_version = "v1"
        
        logger.info("KubernetesAutoscalingAdapter initialized")
    
    async def create_hpa(
        self,
        hpa: HorizontalPodAutoscaler
    ) -> Dict[str, Any]:
        """Create Kubernetes HorizontalPodAutoscaler."""
        logger.info("Creating HPA", name=hpa.name, namespace=hpa.namespace)
        
        try:
            # Build HPA manifest
            hpa_manifest = await self._build_hpa_manifest(hpa)
            
            # Create HPA
            result = self.autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=hpa.namespace,
                body=hpa_manifest
            )
            
            logger.info("HPA created successfully", name=hpa.name)
            return result.to_dict()
            
        except ApiException as e:
            logger.error("Failed to create HPA", name=hpa.name, error=str(e))
            raise
    
    async def update_hpa(
        self,
        hpa: HorizontalPodAutoscaler
    ) -> Dict[str, Any]:
        """Update Kubernetes HorizontalPodAutoscaler."""
        logger.info("Updating HPA", name=hpa.name, namespace=hpa.namespace)
        
        try:
            # Build updated HPA manifest
            hpa_manifest = await self._build_hpa_manifest(hpa)
            
            # Update HPA
            result = self.autoscaling_v2.replace_namespaced_horizontal_pod_autoscaler(
                name=hpa.name,
                namespace=hpa.namespace,
                body=hpa_manifest
            )
            
            logger.info("HPA updated successfully", name=hpa.name)
            return result.to_dict()
            
        except ApiException as e:
            logger.error("Failed to update HPA", name=hpa.name, error=str(e))
            raise
    
    async def delete_hpa(
        self,
        name: str,
        namespace: str
    ) -> bool:
        """Delete Kubernetes HorizontalPodAutoscaler."""
        logger.info("Deleting HPA", name=name, namespace=namespace)
        
        try:
            self.autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(
                name=name,
                namespace=namespace
            )
            
            logger.info("HPA deleted successfully", name=name)
            return True
            
        except ApiException as e:
            if e.status == 404:
                logger.info("HPA not found", name=name)
                return True
            logger.error("Failed to delete HPA", name=name, error=str(e))
            return False
    
    async def get_hpa_status(
        self,
        name: str,
        namespace: str
    ) -> Optional[Dict[str, Any]]:
        """Get HPA status from Kubernetes."""
        try:
            hpa = self.autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                name=name,
                namespace=namespace
            )
            
            return {
                "current_replicas": hpa.status.current_replicas,
                "desired_replicas": hpa.status.desired_replicas,
                "current_metrics": hpa.status.current_metrics or [],
                "conditions": [c.to_dict() for c in hpa.status.conditions or []],
                "last_scale_time": hpa.status.last_scale_time.isoformat() if hpa.status.last_scale_time else None
            }
            
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error("Failed to get HPA status", name=name, error=str(e))
            raise
    
    async def create_vpa(
        self,
        vpa: VerticalPodAutoscaler
    ) -> Dict[str, Any]:
        """Create Kubernetes VerticalPodAutoscaler."""
        logger.info("Creating VPA", name=vpa.name, namespace=vpa.namespace)
        
        try:
            # Build VPA manifest
            vpa_manifest = await self._build_vpa_manifest(vpa)
            
            # Create VPA using custom objects API
            result = self.custom_objects_api.create_namespaced_custom_object(
                group=self.vpa_group,
                version=self.vpa_version,
                namespace=vpa.namespace,
                plural="verticalpodautoscalers",
                body=vpa_manifest
            )
            
            logger.info("VPA created successfully", name=vpa.name)
            return result
            
        except ApiException as e:
            logger.error("Failed to create VPA", name=vpa.name, error=str(e))
            raise
    
    async def update_vpa(
        self,
        vpa: VerticalPodAutoscaler
    ) -> Dict[str, Any]:
        """Update Kubernetes VerticalPodAutoscaler."""
        logger.info("Updating VPA", name=vpa.name, namespace=vpa.namespace)
        
        try:
            # Build updated VPA manifest
            vpa_manifest = await self._build_vpa_manifest(vpa)
            
            # Update VPA
            result = self.custom_objects_api.replace_namespaced_custom_object(
                group=self.vpa_group,
                version=self.vpa_version,
                namespace=vpa.namespace,
                plural="verticalpodautoscalers",
                name=vpa.name,
                body=vpa_manifest
            )
            
            logger.info("VPA updated successfully", name=vpa.name)
            return result
            
        except ApiException as e:
            logger.error("Failed to update VPA", name=vpa.name, error=str(e))
            raise
    
    async def get_vpa_status(
        self,
        name: str,
        namespace: str
    ) -> Optional[Dict[str, Any]]:
        """Get VPA status from Kubernetes."""
        try:
            vpa = self.custom_objects_api.get_namespaced_custom_object(
                group=self.vpa_group,
                version=self.vpa_version,
                namespace=namespace,
                plural="verticalpodautoscalers",
                name=name
            )
            
            return vpa.get("status", {})
            
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error("Failed to get VPA status", name=name, error=str(e))
            raise
    
    async def scale_deployment(
        self,
        deployment_name: str,
        namespace: str,
        replicas: int
    ) -> bool:
        """Scale a deployment to specified replica count."""
        logger.info("Scaling deployment", deployment=deployment_name, replicas=replicas)
        
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update replica count
            deployment.spec.replicas = replicas
            
            # Apply update
            self.apps_v1.replace_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info("Deployment scaled successfully", deployment=deployment_name, replicas=replicas)
            return True
            
        except ApiException as e:
            logger.error("Failed to scale deployment", deployment=deployment_name, error=str(e))
            return False
    
    async def get_deployment_metrics(
        self,
        deployment_name: str,
        namespace: str
    ) -> Dict[str, Any]:
        """Get deployment resource metrics."""
        try:
            # Get deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Get pods for deployment
            label_selector = ",".join([f"{k}={v}" for k, v in deployment.spec.selector.match_labels.items()])
            pods = self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector
            )
            
            # Calculate aggregate metrics
            total_cpu = 0
            total_memory = 0
            running_pods = 0
            
            for pod in pods.items:
                if pod.status.phase == "Running":
                    running_pods += 1
                    # This would integrate with metrics server to get actual usage
                    # For now, return mock values
                    total_cpu += 50  # Mock CPU utilization
                    total_memory += 60  # Mock memory utilization
            
            return {
                "deployment": deployment_name,
                "replicas": {
                    "desired": deployment.spec.replicas,
                    "current": deployment.status.replicas or 0,
                    "ready": deployment.status.ready_replicas or 0,
                    "running_pods": running_pods
                },
                "metrics": {
                    "cpu_utilization": total_cpu / max(running_pods, 1),
                    "memory_utilization": total_memory / max(running_pods, 1)
                }
            }
            
        except ApiException as e:
            logger.error("Failed to get deployment metrics", deployment=deployment_name, error=str(e))
            raise
    
    async def list_hpas(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List HPAs in namespace or cluster."""
        try:
            if namespace:
                hpas = self.autoscaling_v2.list_namespaced_horizontal_pod_autoscaler(
                    namespace=namespace,
                    label_selector=label_selector
                )
            else:
                hpas = self.autoscaling_v2.list_horizontal_pod_autoscaler_for_all_namespaces(
                    label_selector=label_selector
                )
            
            return [hpa.to_dict() for hpa in hpas.items]
            
        except ApiException as e:
            logger.error("Failed to list HPAs", error=str(e))
            raise
    
    async def list_vpas(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List VPAs in namespace or cluster."""
        try:
            if namespace:
                vpas = self.custom_objects_api.list_namespaced_custom_object(
                    group=self.vpa_group,
                    version=self.vpa_version,
                    namespace=namespace,
                    plural="verticalpodautoscalers",
                    label_selector=label_selector
                )
            else:
                vpas = self.custom_objects_api.list_cluster_custom_object(
                    group=self.vpa_group,
                    version=self.vpa_version,
                    plural="verticalpodautoscalers",
                    label_selector=label_selector
                )
            
            return vpas.get("items", [])
            
        except ApiException as e:
            logger.error("Failed to list VPAs", error=str(e))
            raise
    
    # Private helper methods
    
    async def _build_hpa_manifest(
        self,
        hpa: HorizontalPodAutoscaler
    ) -> kubernetes.client.V2HorizontalPodAutoscaler:
        """Build Kubernetes HPA manifest."""
        
        # Build metrics
        metrics = []
        for metric in hpa.metrics:
            if metric.get("type") == "Resource":
                k8s_metric = kubernetes.client.V2MetricSpec(
                    type="Resource",
                    resource=kubernetes.client.V2ResourceMetricSource(
                        name=metric["resource"]["name"],
                        target=kubernetes.client.V2MetricTarget(
                            type="Utilization",
                            average_utilization=metric["resource"]["target"]["averageUtilization"]
                        )
                    )
                )
                metrics.append(k8s_metric)
            elif metric.get("type") == "Pods":
                k8s_metric = kubernetes.client.V2MetricSpec(
                    type="Pods",
                    pods=kubernetes.client.V2PodsMetricSource(
                        metric=kubernetes.client.V2MetricIdentifier(
                            name=metric["pods"]["metric"]["name"]
                        ),
                        target=kubernetes.client.V2MetricTarget(
                            type="AverageValue",
                            average_value=metric["pods"]["target"]["averageValue"]
                        )
                    )
                )
                metrics.append(k8s_metric)
        
        # Build behavior
        behavior = None
        if hpa.scale_up_behavior or hpa.scale_down_behavior:
            behavior = kubernetes.client.V2HorizontalPodAutoscalerBehavior()
            
            if hpa.scale_up_behavior:
                behavior.scale_up = kubernetes.client.V2HPAScalingRules(
                    stabilization_window_seconds=hpa.scale_up_behavior.get("stabilizationWindowSeconds", 0),
                    policies=[
                        kubernetes.client.V2HPAScalingPolicy(
                            type=policy.get("type", "Percent"),
                            value=policy.get("value", 100),
                            period_seconds=policy.get("periodSeconds", 15)
                        ) for policy in hpa.scale_up_behavior.get("policies", [])
                    ]
                )
            
            if hpa.scale_down_behavior:
                behavior.scale_down = kubernetes.client.V2HPAScalingRules(
                    stabilization_window_seconds=hpa.scale_down_behavior.get("stabilizationWindowSeconds", 300),
                    policies=[
                        kubernetes.client.V2HPAScalingPolicy(
                            type=policy.get("type", "Percent"),
                            value=policy.get("value", 100),
                            period_seconds=policy.get("periodSeconds", 15)
                        ) for policy in hpa.scale_down_behavior.get("policies", [])
                    ]
                )
        
        # Build HPA manifest
        hpa_manifest = kubernetes.client.V2HorizontalPodAutoscaler(
            metadata=kubernetes.client.V1ObjectMeta(
                name=hpa.name,
                namespace=hpa.namespace,
                labels=dict(hpa.labels),
                annotations=dict(hpa.annotations)
            ),
            spec=kubernetes.client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=kubernetes.client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind=hpa.target_resource_type,
                    name=hpa.target_resource_name
                ),
                min_replicas=hpa.min_replicas,
                max_replicas=hpa.max_replicas,
                metrics=metrics,
                behavior=behavior
            )
        )
        
        return hpa_manifest
    
    async def _build_vpa_manifest(
        self,
        vpa: VerticalPodAutoscaler
    ) -> Dict[str, Any]:
        """Build Kubernetes VPA manifest."""
        
        manifest = {
            "apiVersion": f"{self.vpa_group}/{self.vpa_version}",
            "kind": "VerticalPodAutoscaler",
            "metadata": {
                "name": vpa.name,
                "namespace": vpa.namespace,
                "labels": dict(vpa.labels),
                "annotations": dict(vpa.annotations)
            },
            "spec": {
                "targetRef": {
                    "apiVersion": "apps/v1",
                    "kind": vpa.target_resource_type,
                    "name": vpa.target_resource_name
                },
                "updatePolicy": {
                    "updateMode": vpa.update_mode
                }
            }
        }
        
        # Add resource policy if present
        if vpa.resource_policy:
            manifest["spec"]["resourcePolicy"] = vpa.resource_policy
        
        # Add container policies
        if vpa.container_policies:
            manifest["spec"]["resourcePolicy"] = manifest["spec"].get("resourcePolicy", {})
            manifest["spec"]["resourcePolicy"]["containerPolicies"] = vpa.container_policies
        
        return manifest