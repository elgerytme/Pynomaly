"""
AWS EKS Manager for Pynomaly Detection
=====================================

Manages Elastic Kubernetes Service clusters for scalable anomaly detection.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EKSClusterConfig:
    """EKS cluster configuration."""
    cluster_name: str
    region: str
    kubernetes_version: str = "1.27"
    node_group_name: str = "pynomaly-nodes"
    instance_types: List[str] = None
    min_size: int = 3
    max_size: int = 20
    desired_size: int = 5
    disk_size: int = 100
    subnet_ids: List[str] = None
    security_group_ids: List[str] = None
    
    def __post_init__(self):
        if self.instance_types is None:
            self.instance_types = ["c5.2xlarge", "c5.4xlarge"]
        if self.subnet_ids is None:
            self.subnet_ids = []
        if self.security_group_ids is None:
            self.security_group_ids = []

class EKSManager:
    """AWS EKS management for Pynomaly Detection."""
    
    def __init__(self, region: str = "us-east-1", profile_name: Optional[str] = None):
        """Initialize EKS manager.
        
        Args:
            region: AWS region
            profile_name: AWS profile name (optional)
        """
        if not AWS_AVAILABLE:
            raise ImportError("AWS SDK (boto3) is required for EKS integration")
        
        self.region = region
        self.profile_name = profile_name
        
        # Initialize AWS clients
        session = boto3.Session(profile_name=profile_name)
        self.eks_client = session.client('eks', region_name=region)
        self.ec2_client = session.client('ec2', region_name=region)
        self.iam_client = session.client('iam', region_name=region)
        
        logger.info(f"EKS Manager initialized for region: {region}")
    
    def create_cluster(self, config: EKSClusterConfig) -> Dict[str, Any]:
        """Create EKS cluster with Pynomaly Detection optimizations.
        
        Args:
            config: EKS cluster configuration
            
        Returns:
            Cluster creation response
        """
        try:
            # Create cluster service role if not exists
            service_role_arn = self._ensure_cluster_service_role()
            
            # Create cluster
            cluster_config = {
                'name': config.cluster_name,
                'version': config.kubernetes_version,
                'roleArn': service_role_arn,
                'resourcesVpcConfig': {
                    'subnetIds': config.subnet_ids,
                    'securityGroupIds': config.security_group_ids,
                    'endpointConfigPrivate': True,
                    'endpointConfigPublic': True,
                    'publicAccessCidrs': ['0.0.0.0/0']
                },
                'logging': {
                    'enable': [
                        {'types': ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler']}
                    ]
                },
                'tags': {
                    'Application': 'Pynomaly-Detection',
                    'Environment': 'Production',
                    'ManagedBy': 'Pynomaly-EKS-Manager'
                }
            }
            
            response = self.eks_client.create_cluster(**cluster_config)
            
            logger.info(f"EKS cluster creation initiated: {config.cluster_name}")
            
            # Wait for cluster to be active
            self._wait_for_cluster_active(config.cluster_name)
            
            # Create node group
            self._create_node_group(config, service_role_arn)
            
            return response
            
        except ClientError as e:
            logger.error(f"Failed to create EKS cluster: {e}")
            raise
    
    def deploy_pynomaly(self, cluster_name: str, namespace: str = "pynomaly-production") -> bool:
        """Deploy Pynomaly Detection to EKS cluster.
        
        Args:
            cluster_name: EKS cluster name
            namespace: Kubernetes namespace
            
        Returns:
            True if deployment successful
        """
        try:
            # Update kubeconfig
            self._update_kubeconfig(cluster_name)
            
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
            
            logger.info(f"Pynomaly Detection deployed to EKS cluster: {cluster_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy Pynomaly to EKS: {e}")
            return False
    
    def scale_cluster(self, cluster_name: str, node_group_name: str, 
                     desired_size: int, min_size: int = None, max_size: int = None) -> bool:
        """Scale EKS cluster node group.
        
        Args:
            cluster_name: EKS cluster name
            node_group_name: Node group name
            desired_size: Desired number of nodes
            min_size: Minimum number of nodes
            max_size: Maximum number of nodes
            
        Returns:
            True if scaling successful
        """
        try:
            scaling_config = {'desiredSize': desired_size}
            
            if min_size is not None:
                scaling_config['minSize'] = min_size
            if max_size is not None:
                scaling_config['maxSize'] = max_size
            
            response = self.eks_client.update_nodegroup_config(
                clusterName=cluster_name,
                nodegroupName=node_group_name,
                scalingConfig=scaling_config
            )
            
            logger.info(f"EKS node group scaling initiated: {node_group_name}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to scale EKS cluster: {e}")
            return False
    
    def get_cluster_status(self, cluster_name: str) -> Dict[str, Any]:
        """Get EKS cluster status and metrics.
        
        Args:
            cluster_name: EKS cluster name
            
        Returns:
            Cluster status information
        """
        try:
            cluster_response = self.eks_client.describe_cluster(name=cluster_name)
            cluster = cluster_response['cluster']
            
            # Get node groups
            node_groups_response = self.eks_client.list_nodegroups(clusterName=cluster_name)
            node_groups = []
            
            for ng_name in node_groups_response['nodegroups']:
                ng_response = self.eks_client.describe_nodegroup(
                    clusterName=cluster_name,
                    nodegroupName=ng_name
                )
                node_groups.append(ng_response['nodegroup'])
            
            return {
                'cluster_name': cluster['name'],
                'status': cluster['status'],
                'version': cluster['version'],
                'endpoint': cluster['endpoint'],
                'platform_version': cluster['platformVersion'],
                'node_groups': node_groups,
                'vpc_config': cluster['resourcesVpcConfig'],
                'logging': cluster.get('logging', {}),
                'created_at': cluster['createdAt'].isoformat(),
                'tags': cluster.get('tags', {})
            }
            
        except ClientError as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {}
    
    def delete_cluster(self, cluster_name: str) -> bool:
        """Delete EKS cluster and associated resources.
        
        Args:
            cluster_name: EKS cluster name
            
        Returns:
            True if deletion successful
        """
        try:
            # Delete node groups first
            node_groups_response = self.eks_client.list_nodegroups(clusterName=cluster_name)
            
            for ng_name in node_groups_response['nodegroups']:
                self.eks_client.delete_nodegroup(
                    clusterName=cluster_name,
                    nodegroupName=ng_name
                )
                logger.info(f"Deleting node group: {ng_name}")
            
            # Wait for node groups to be deleted
            self._wait_for_node_groups_deleted(cluster_name)
            
            # Delete cluster
            self.eks_client.delete_cluster(name=cluster_name)
            logger.info(f"EKS cluster deletion initiated: {cluster_name}")
            
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete EKS cluster: {e}")
            return False
    
    def _ensure_cluster_service_role(self) -> str:
        """Ensure cluster service role exists."""
        role_name = "Pynomaly-EKS-ServiceRole"
        
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            return response['Role']['Arn']
        except ClientError:
            # Create role if it doesn't exist
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "eks.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="EKS service role for Pynomaly Detection"
            )
            
            # Attach required policies
            policies = [
                "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy",
                "arn:aws:iam::aws:policy/AmazonEKSServicePolicy"
            ]
            
            for policy_arn in policies:
                try:
                    self.iam_client.attach_role_policy(
                        RoleName=role_name,
                        PolicyArn=policy_arn
                    )
                except ClientError:
                    pass  # Policy might already be attached
            
            return response['Role']['Arn']
    
    def _create_node_group(self, config: EKSClusterConfig, service_role_arn: str):
        """Create EKS node group."""
        node_role_arn = self._ensure_node_group_role()
        
        node_group_config = {
            'clusterName': config.cluster_name,
            'nodegroupName': config.node_group_name,
            'scalingConfig': {
                'minSize': config.min_size,
                'maxSize': config.max_size,
                'desiredSize': config.desired_size
            },
            'diskSize': config.disk_size,
            'subnets': config.subnet_ids,
            'instanceTypes': config.instance_types,
            'amiType': 'AL2_x86_64',
            'nodeRole': node_role_arn,
            'tags': {
                'Application': 'Pynomaly-Detection',
                'NodeGroup': config.node_group_name
            }
        }
        
        self.eks_client.create_nodegroup(**node_group_config)
        logger.info(f"Node group creation initiated: {config.node_group_name}")
    
    def _ensure_node_group_role(self) -> str:
        """Ensure node group role exists."""
        role_name = "Pynomaly-EKS-NodeRole"
        
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            return response['Role']['Arn']
        except ClientError:
            # Create role
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "ec2.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="EKS node group role for Pynomaly Detection"
            )
            
            # Attach required policies
            policies = [
                "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
                "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
                "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
            ]
            
            for policy_arn in policies:
                try:
                    self.iam_client.attach_role_policy(
                        RoleName=role_name,
                        PolicyArn=policy_arn
                    )
                except ClientError:
                    pass
            
            return response['Role']['Arn']
    
    def _wait_for_cluster_active(self, cluster_name: str, timeout: int = 1800):
        """Wait for cluster to become active."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.eks_client.describe_cluster(name=cluster_name)
                status = response['cluster']['status']
                
                if status == 'ACTIVE':
                    logger.info(f"EKS cluster is active: {cluster_name}")
                    return True
                elif status == 'FAILED':
                    logger.error(f"EKS cluster creation failed: {cluster_name}")
                    return False
                
                time.sleep(30)
                
            except ClientError as e:
                logger.error(f"Error checking cluster status: {e}")
                return False
        
        logger.error(f"Timeout waiting for cluster to become active: {cluster_name}")
        return False
    
    def _wait_for_node_groups_deleted(self, cluster_name: str, timeout: int = 1800):
        """Wait for all node groups to be deleted."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.eks_client.list_nodegroups(clusterName=cluster_name)
                if not response['nodegroups']:
                    logger.info(f"All node groups deleted for cluster: {cluster_name}")
                    return True
                
                time.sleep(30)
                
            except ClientError:
                return True  # Cluster might already be deleted
        
        logger.error(f"Timeout waiting for node groups to be deleted: {cluster_name}")
        return False
    
    def _update_kubeconfig(self, cluster_name: str):
        """Update kubeconfig for EKS cluster."""
        import subprocess
        
        cmd = [
            'aws', 'eks', 'update-kubeconfig',
            '--region', self.region,
            '--name', cluster_name
        ]
        
        if self.profile_name:
            cmd.extend(['--profile', self.profile_name])
        
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