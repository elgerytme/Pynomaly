"""
Custom Resource Management

Provides utilities for managing Kubernetes Custom Resources and
Custom Resource Definitions for operators.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID

from structlog import get_logger
import kubernetes.client
from kubernetes.client.rest import ApiException
import yaml

from ...domain.entities.kubernetes_resource import OperatorResource

logger = get_logger(__name__)


class CustomResourceManager:
    """
    Manager for Kubernetes Custom Resources.
    
    Provides high-level operations for creating, managing, and
    interacting with Custom Resource Definitions and instances.
    """
    
    def __init__(self):
        self.api_client = kubernetes.client.ApiClient()
        self.custom_objects_api = kubernetes.client.CustomObjectsApi()
        self.extensions_v1_api = kubernetes.client.ApiextensionsV1Api()
        
        logger.info("CustomResourceManager initialized")
    
    async def create_crd(
        self,
        name: str,
        group: str,
        version: str,
        kind: str,
        plural: str,
        schema: Dict[str, Any],
        scope: str = "Namespaced"
    ) -> Dict[str, Any]:
        """Create a Custom Resource Definition."""
        logger.info("Creating CRD", name=name, group=group, kind=kind)
        
        try:
            # Build CRD manifest
            crd_manifest = {
                "apiVersion": "apiextensions.k8s.io/v1",
                "kind": "CustomResourceDefinition",
                "metadata": {
                    "name": f"{plural}.{group}",
                    "labels": {
                        "app.kubernetes.io/managed-by": "pynomaly-enterprise",
                        "pynomaly.io/component": "cloud-native"
                    }
                },
                "spec": {
                    "group": group,
                    "versions": [
                        {
                            "name": version,
                            "served": True,
                            "storage": True,
                            "schema": {
                                "openAPIV3Schema": schema
                            },
                            "subresources": {
                                "status": {},
                                "scale": {
                                    "specReplicasPath": ".spec.replicas",
                                    "statusReplicasPath": ".status.replicas",
                                    "labelSelectorPath": ".status.labelSelector"
                                }
                            }
                        }
                    ],
                    "scope": scope,
                    "names": {
                        "plural": plural,
                        "singular": plural.rstrip('s'),
                        "kind": kind,
                        "categories": ["pynomaly", "enterprise"]
                    },
                    "conversion": {
                        "strategy": "None"
                    }
                }
            }
            
            # Create CRD
            crd = kubernetes.client.V1CustomResourceDefinition(**crd_manifest)
            result = self.extensions_v1_api.create_custom_resource_definition(crd)
            
            logger.info("CRD created successfully", name=name)
            return result.to_dict()
            
        except ApiException as e:
            if e.status == 409:
                logger.info("CRD already exists", name=name)
                return await self.get_crd(f"{plural}.{group}")
            else:
                logger.error("Failed to create CRD", name=name, error=str(e))
                raise
    
    async def get_crd(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a Custom Resource Definition."""
        try:
            result = self.extensions_v1_api.read_custom_resource_definition(name)
            return result.to_dict()
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error("Failed to get CRD", name=name, error=str(e))
            raise
    
    async def delete_crd(self, name: str) -> bool:
        """Delete a Custom Resource Definition."""
        logger.info("Deleting CRD", name=name)
        
        try:
            self.extensions_v1_api.delete_custom_resource_definition(name)
            logger.info("CRD deleted successfully", name=name)
            return True
        except ApiException as e:
            if e.status == 404:
                logger.info("CRD not found", name=name)
                return True
            logger.error("Failed to delete CRD", name=name, error=str(e))
            return False
    
    async def create_custom_resource(
        self,
        group: str,
        version: str,
        plural: str,
        namespace: Optional[str],
        body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a custom resource instance."""
        logger.info("Creating custom resource", 
                   group=group, version=version, plural=plural,
                   name=body.get('metadata', {}).get('name'))
        
        try:
            if namespace:
                result = self.custom_objects_api.create_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    body=body
                )
            else:
                result = self.custom_objects_api.create_cluster_custom_object(
                    group=group,
                    version=version,
                    plural=plural,
                    body=body
                )
            
            logger.info("Custom resource created successfully")
            return result
            
        except ApiException as e:
            logger.error("Failed to create custom resource", error=str(e))
            raise
    
    async def get_custom_resource(
        self,
        group: str,
        version: str,
        plural: str,
        name: str,
        namespace: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a custom resource instance."""
        try:
            if namespace:
                result = self.custom_objects_api.get_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    name=name
                )
            else:
                result = self.custom_objects_api.get_cluster_custom_object(
                    group=group,
                    version=version,
                    plural=plural,
                    name=name
                )
            
            return result
            
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error("Failed to get custom resource", 
                        name=name, error=str(e))
            raise
    
    async def update_custom_resource(
        self,
        group: str,
        version: str,
        plural: str,
        name: str,
        body: Dict[str, Any],
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update a custom resource instance."""
        logger.info("Updating custom resource", name=name)
        
        try:
            if namespace:
                result = self.custom_objects_api.replace_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    name=name,
                    body=body
                )
            else:
                result = self.custom_objects_api.replace_cluster_custom_object(
                    group=group,
                    version=version,
                    plural=plural,
                    name=name,
                    body=body
                )
            
            logger.info("Custom resource updated successfully", name=name)
            return result
            
        except ApiException as e:
            logger.error("Failed to update custom resource", 
                        name=name, error=str(e))
            raise
    
    async def patch_custom_resource_status(
        self,
        group: str,
        version: str,
        plural: str,
        name: str,
        status: Dict[str, Any],
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update custom resource status."""
        logger.debug("Updating custom resource status", name=name)
        
        try:
            body = {"status": status}
            
            if namespace:
                result = self.custom_objects_api.patch_namespaced_custom_object_status(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    name=name,
                    body=body
                )
            else:
                result = self.custom_objects_api.patch_cluster_custom_object_status(
                    group=group,
                    version=version,
                    plural=plural,
                    name=name,
                    body=body
                )
            
            return result
            
        except ApiException as e:
            logger.error("Failed to update custom resource status", 
                        name=name, error=str(e))
            raise
    
    async def delete_custom_resource(
        self,
        group: str,
        version: str,
        plural: str,
        name: str,
        namespace: Optional[str] = None
    ) -> bool:
        """Delete a custom resource instance."""
        logger.info("Deleting custom resource", name=name)
        
        try:
            if namespace:
                self.custom_objects_api.delete_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    name=name
                )
            else:
                self.custom_objects_api.delete_cluster_custom_object(
                    group=group,
                    version=version,
                    plural=plural,
                    name=name
                )
            
            logger.info("Custom resource deleted successfully", name=name)
            return True
            
        except ApiException as e:
            if e.status == 404:
                logger.info("Custom resource not found", name=name)
                return True
            logger.error("Failed to delete custom resource", 
                        name=name, error=str(e))
            return False
    
    async def list_custom_resources(
        self,
        group: str,
        version: str,
        plural: str,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List custom resource instances."""
        logger.debug("Listing custom resources", group=group, plural=plural)
        
        try:
            if namespace:
                result = self.custom_objects_api.list_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    label_selector=label_selector
                )
            else:
                result = self.custom_objects_api.list_cluster_custom_object(
                    group=group,
                    version=version,
                    plural=plural,
                    label_selector=label_selector
                )
            
            return result.get('items', [])
            
        except ApiException as e:
            logger.error("Failed to list custom resources", error=str(e))
            raise
    
    def get_default_anomaly_detection_schema(self) -> Dict[str, Any]:
        """Get default schema for anomaly detection custom resource."""
        return {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "object",
                    "properties": {
                        "tenantId": {
                            "type": "string",
                            "description": "Tenant identifier"
                        },
                        "anomalyDetectionConfig": {
                            "type": "object",
                            "properties": {
                                "algorithm": {
                                    "type": "string",
                                    "enum": ["isolation_forest", "one_class_svm", "ensemble"],
                                    "default": "isolation_forest"
                                },
                                "threshold": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "default": 0.5
                                },
                                "windowSize": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "default": 100
                                }
                            },
                            "required": ["algorithm"]
                        },
                        "dataSource": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["kafka", "kinesis", "pubsub", "webhook"]
                                },
                                "config": {
                                    "type": "object"
                                }
                            },
                            "required": ["type", "config"]
                        },
                        "scaling": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": true
                                },
                                "minReplicas": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "default": 1
                                },
                                "maxReplicas": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "default": 10
                                }
                            }
                        }
                    },
                    "required": ["tenantId", "anomalyDetectionConfig", "dataSource"]
                },
                "status": {
                    "type": "object",
                    "properties": {
                        "phase": {
                            "type": "string",
                            "enum": ["Pending", "Running", "Failed", "Succeeded"]
                        },
                        "message": {
                            "type": "string"
                        },
                        "lastProcessedTimestamp": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "anomaliesDetected": {
                            "type": "integer",
                            "minimum": 0
                        },
                        "accuracy": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
                    }
                }
            },
            "required": ["spec"]
        }