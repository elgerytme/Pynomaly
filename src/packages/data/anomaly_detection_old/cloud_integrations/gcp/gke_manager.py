"""
Google Kubernetes Engine Manager for Pynomaly Detection
========================================================

Manages GKE clusters for scalable anomaly detection deployment.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from google.cloud import container_v1
    from google.cloud import resource_manager
    from google.api_core import exceptions
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class GKEClusterConfig:
    """GKE cluster configuration."""
    cluster_name: str
    project_id: str
    zone: str
    region: str = None
    kubernetes_version: str = "1.27"
    node_pool_name: str = "pynomaly-nodes"
    machine_type: str = "e2-standard-4"
    disk_size_gb: int = 100
    min_nodes: int = 3
    max_nodes: int = 20
    initial_node_count: int = 5
    enable_autoscaling: bool = True
    enable_autorepair: bool = True
    enable_autoupgrade: bool = True
    preemptible: bool = False
    
    def __post_init__(self):
        if self.region is None:
            # Extract region from zone (e.g., us-central1-a -> us-central1)
            self.region = '-'.join(self.zone.split('-')[:-1])

class GKEManager:
    """Google Kubernetes Engine management for Pynomaly Detection."""
    
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        """Initialize GKE manager.
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to service account credentials (optional)
        """
        if not GCP_AVAILABLE:
            raise ImportError("Google Cloud SDK is required for GKE integration")
        
        self.project_id = project_id
        self.credentials_path = credentials_path
        
        # Initialize GCP clients
        if credentials_path:
            import os
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        self.container_client = container_v1.ClusterManagerClient()
        
        logger.info(f"GKE Manager initialized for project: {project_id}")
    
    def create_cluster(self, config: GKEClusterConfig) -> Dict[str, Any]:
        """Create GKE cluster with Pynomaly Detection optimizations.
        
        Args:
            config: GKE cluster configuration
            
        Returns:
            Cluster creation response
        """
        try:
            # Define cluster configuration
            cluster_config = {
                "name": config.cluster_name,
                "description": "Pynomaly Detection cluster",
                "initial_node_count": config.initial_node_count,
                "node_config": {
                    "machine_type": config.machine_type,
                    "disk_size_gb": config.disk_size_gb,
                    "oauth_scopes": [
                        "https://www.googleapis.com/auth/cloud-platform"
                    ],
                    "labels": {
                        "application": "pynomaly-detection",
                        "environment": "production"
                    },
                    "preemptible": config.preemptible
                },
                "master_auth": {
                    "client_certificate_config": {
                        "issue_client_certificate": False
                    }
                },
                "legacy_abac": {
                    "enabled": False
                },
                "network_policy": {
                    "enabled": True,
                    "provider": "CALICO"
                },
                "ip_allocation_policy": {
                    "use_ip_aliases": True
                },
                "master_authorized_networks_config": {
                    "enabled": True,
                    "cidr_blocks": [
                        {
                            "cidr_block": "0.0.0.0/0",
                            "display_name": "All networks"
                        }
                    ]
                },
                "addons_config": {
                    "http_load_balancing": {
                        "disabled": False
                    },
                    "horizontal_pod_autoscaling": {
                        "disabled": False
                    },
                    "network_policy_config": {
                        "disabled": False
                    }
                },
                "enable_shielded_nodes": True,
                "resource_labels": {
                    "application": "pynomaly-detection",
                    "managed-by": "pynomaly-gke-manager"
                }
            }
            
            # Set location (zone or region)
            if config.region:
                parent = f"projects/{config.project_id}/locations/{config.region}"
            else:
                parent = f"projects/{config.project_id}/locations/{config.zone}"
            
            # Create cluster
            operation = self.container_client.create_cluster(
                parent=parent,
                cluster=cluster_config
            )
            
            logger.info(f"GKE cluster creation initiated: {config.cluster_name}")
            
            # Wait for cluster creation to complete
            self._wait_for_operation(operation, config.project_id, config.zone)
            
            # Create node pool if autoscaling is enabled
            if config.enable_autoscaling:
                self._create_node_pool(config)
            
            return {
                "cluster_name": config.cluster_name,
                "operation_name": operation.name,
                "status": "created"
            }
            
        except exceptions.GoogleAPIError as e:
            logger.error(f"Failed to create GKE cluster: {e}")
            raise
    
    def deploy_pynomaly(self, cluster_name: str, project_id: str, 
                       zone: str, namespace: str = "pynomaly-production") -> bool:
        """Deploy Pynomaly Detection to GKE cluster.
        
        Args:
            cluster_name: GKE cluster name
            project_id: GCP project ID
            zone: GCP zone
            namespace: Kubernetes namespace
            
        Returns:
            True if deployment successful
        """
        try:
            # Get cluster credentials
            self._get_cluster_credentials(cluster_name, project_id, zone)
            
            # Apply Kubernetes manifests
            kubectl_commands = [
                f"kubectl create namespace {namespace} --dry-run=client -o yaml | kubectl apply -f -",
                f"kubectl apply -k k8s/overlays/production/ -n {namespace}",
                f"kubectl rollout status deployment/pynomaly-detection -n {namespace}"
            ]
            
            for cmd in kubectl_commands:
                result = self._run_kubectl_command(cmd)
                if not result:
                    logger.error(f"Failed to execute: {cmd}")
                    return False
            
            logger.info(f"Pynomaly Detection deployed to GKE cluster: {cluster_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy Pynomaly to GKE: {e}")
            return False
    
    def scale_cluster(self, cluster_name: str, project_id: str, zone: str,
                     node_pool_name: str, node_count: int) -> bool:
        """Scale GKE cluster node pool.
        
        Args:
            cluster_name: GKE cluster name
            project_id: GCP project ID
            zone: GCP zone
            node_pool_name: Node pool name
            node_count: Target node count
            
        Returns:
            True if scaling successful
        """
        try:
            parent = f"projects/{project_id}/locations/{zone}"
            name = f"{parent}/clusters/{cluster_name}/nodePools/{node_pool_name}"
            
            operation = self.container_client.set_node_pool_size(
                name=name,
                node_count=node_count
            )
            
            self._wait_for_operation(operation, project_id, zone)
            
            logger.info(f"GKE node pool scaled: {node_pool_name} to {node_count} nodes")
            return True
            
        except exceptions.GoogleAPIError as e:
            logger.error(f"Failed to scale GKE cluster: {e}")
            return False
    
    def get_cluster_status(self, cluster_name: str, project_id: str, zone: str) -> Dict[str, Any]:
        """Get GKE cluster status and information.
        
        Args:
            cluster_name: GKE cluster name
            project_id: GCP project ID
            zone: GCP zone
            
        Returns:
            Cluster status information
        """
        try:
            name = f"projects/{project_id}/locations/{zone}/clusters/{cluster_name}"
            
            cluster = self.container_client.get_cluster(name=name)
            
            # Get node pools
            node_pools = []
            for node_pool in cluster.node_pools:
                node_pools.append({
                    "name": node_pool.name,
                    "status": node_pool.status.name,
                    "initial_node_count": node_pool.initial_node_count,
                    "machine_type": node_pool.config.machine_type,
                    "disk_size_gb": node_pool.config.disk_size_gb,
                    "autoscaling": {
                        "enabled": node_pool.autoscaling.enabled,
                        "min_node_count": node_pool.autoscaling.min_node_count,
                        "max_node_count": node_pool.autoscaling.max_node_count
                    } if node_pool.autoscaling else None
                })
            
            return {
                "cluster_name": cluster.name,
                "status": cluster.status.name,
                "location": cluster.location,
                "kubernetes_version": cluster.current_master_version,
                "node_version": cluster.current_node_version,
                "endpoint": cluster.endpoint,
                "node_pools": node_pools,
                "network": cluster.network,
                "subnetwork": cluster.subnetwork,
                "services_ipv4_cidr": cluster.services_ipv4_cidr,
                "cluster_ipv4_cidr": cluster.cluster_ipv4_cidr,
                "create_time": cluster.create_time,
                "resource_labels": dict(cluster.resource_labels)
            }
            
        except exceptions.GoogleAPIError as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {}
    
    def delete_cluster(self, cluster_name: str, project_id: str, zone: str) -> bool:
        """Delete GKE cluster.
        
        Args:
            cluster_name: GKE cluster name
            project_id: GCP project ID
            zone: GCP zone
            
        Returns:
            True if deletion successful
        """
        try:
            name = f"projects/{project_id}/locations/{zone}/clusters/{cluster_name}"
            
            operation = self.container_client.delete_cluster(name=name)
            
            self._wait_for_operation(operation, project_id, zone)
            
            logger.info(f"GKE cluster deleted: {cluster_name}")
            return True
            
        except exceptions.GoogleAPIError as e:
            logger.error(f"Failed to delete GKE cluster: {e}")
            return False
    
    def _create_node_pool(self, config: GKEClusterConfig):
        """Create node pool with autoscaling."""
        try:
            node_pool_config = {
                "name": config.node_pool_name,
                "config": {
                    "machine_type": config.machine_type,
                    "disk_size_gb": config.disk_size_gb,
                    "oauth_scopes": [
                        "https://www.googleapis.com/auth/cloud-platform"
                    ],
                    "labels": {
                        "application": "pynomaly-detection",
                        "pool": config.node_pool_name
                    },
                    "preemptible": config.preemptible
                },
                "initial_node_count": config.initial_node_count,
                "autoscaling": {
                    "enabled": config.enable_autoscaling,
                    "min_node_count": config.min_nodes,
                    "max_node_count": config.max_nodes
                },
                "management": {
                    "auto_repair": config.enable_autorepair,
                    "auto_upgrade": config.enable_autoupgrade
                }
            }
            
            parent = f"projects/{config.project_id}/locations/{config.zone}/clusters/{config.cluster_name}"
            
            operation = self.container_client.create_node_pool(
                parent=parent,
                node_pool=node_pool_config
            )
            
            self._wait_for_operation(operation, config.project_id, config.zone)
            
            logger.info(f"Node pool created: {config.node_pool_name}")
            
        except exceptions.GoogleAPIError as e:
            logger.error(f"Failed to create node pool: {e}")
            raise
    
    def _wait_for_operation(self, operation, project_id: str, zone: str, timeout: int = 1800):
        """Wait for GKE operation to complete."""
        operation_name = operation.name
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                name = f"projects/{project_id}/locations/{zone}/operations/{operation_name}"
                current_operation = self.container_client.get_operation(name=name)
                
                if current_operation.status == container_v1.Operation.Status.DONE:
                    logger.info(f"GKE operation completed: {operation_name}")
                    return True
                elif current_operation.status == container_v1.Operation.Status.ABORTING:
                    logger.error(f"GKE operation aborted: {operation_name}")
                    return False
                
                time.sleep(30)
                
            except exceptions.GoogleAPIError as e:
                logger.error(f"Error checking operation status: {e}")
                return False
        
        logger.error(f"Timeout waiting for operation: {operation_name}")
        return False
    
    def _get_cluster_credentials(self, cluster_name: str, project_id: str, zone: str):
        """Get GKE cluster credentials."""
        import subprocess
        
        cmd = [
            'gcloud', 'container', 'clusters', 'get-credentials',
            cluster_name,
            '--zone', zone,
            '--project', project_id
        ]
        
        subprocess.run(cmd, check=True)
    
    def _run_kubectl_command(self, command: str) -> bool:
        """Run kubectl command."""
        import subprocess
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"kubectl command failed: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"kubectl command timed out: {command}")
            return False
        except Exception as e:
            logger.error(f"Error running kubectl command: {e}")
            return False