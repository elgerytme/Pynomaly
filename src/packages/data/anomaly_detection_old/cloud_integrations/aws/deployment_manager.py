"""
AWS Deployment Manager for Pynomaly Detection
==============================================

Comprehensive deployment management across AWS services.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .eks_manager import EKSManager, EKSClusterConfig
from .s3_storage import S3StorageManager, S3StorageConfig
from .cloudwatch_monitor import CloudWatchMonitor, CloudWatchConfig

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class AWSDeploymentConfig:
    """AWS deployment configuration."""
    deployment_name: str
    region: str = "us-east-1"
    environment: str = "production"
    
    # EKS Configuration
    cluster_name: str = None
    kubernetes_version: str = "1.27"
    node_instance_types: List[str] = None
    min_nodes: int = 3
    max_nodes: int = 20
    desired_nodes: int = 5
    
    # S3 Configuration
    bucket_name: str = None
    enable_versioning: bool = True
    enable_encryption: bool = True
    
    # CloudWatch Configuration
    namespace: str = "Pynomaly/Detection"
    log_retention_days: int = 30
    enable_detailed_monitoring: bool = True
    
    # Networking
    vpc_id: Optional[str] = None
    subnet_ids: List[str] = None
    security_group_ids: List[str] = None
    
    # Tags
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.cluster_name is None:
            self.cluster_name = f"pynomaly-{self.deployment_name}-{self.environment}"
        
        if self.bucket_name is None:
            self.bucket_name = f"pynomaly-{self.deployment_name}-{self.environment}"
        
        if self.node_instance_types is None:
            self.node_instance_types = ["c5.2xlarge", "c5.4xlarge"]
        
        if self.subnet_ids is None:
            self.subnet_ids = []
        
        if self.security_group_ids is None:
            self.security_group_ids = []
        
        if self.tags is None:
            self.tags = {
                "Application": "Pynomaly-Detection",
                "Environment": self.environment,
                "DeploymentName": self.deployment_name
            }

class AWSDeploymentManager:
    """Comprehensive AWS deployment manager for Pynomaly Detection."""
    
    def __init__(self, config: AWSDeploymentConfig, profile_name: Optional[str] = None):
        """Initialize AWS deployment manager.
        
        Args:
            config: AWS deployment configuration
            profile_name: AWS profile name (optional)
        """
        if not AWS_AVAILABLE:
            raise ImportError("AWS SDK (boto3) is required for AWS integration")
        
        self.config = config
        self.profile_name = profile_name
        
        # Initialize AWS session
        self.session = boto3.Session(profile_name=profile_name)
        
        # Initialize service managers
        self.eks_manager = None
        self.s3_manager = None
        self.cloudwatch_monitor = None
        
        logger.info(f"AWS Deployment Manager initialized for: {config.deployment_name}")
    
    def deploy_full_stack(self) -> Dict[str, Any]:
        """Deploy complete Pynomaly Detection stack to AWS.
        
        Returns:
            Deployment result with all component details
        """
        try:
            logger.info(f"Starting full stack deployment: {self.config.deployment_name}")
            
            deployment_result = {
                'deployment_name': self.config.deployment_name,
                'region': self.config.region,
                'environment': self.config.environment,
                'started_at': datetime.now().isoformat(),
                'components': {},
                'status': 'in_progress'
            }
            
            # Step 1: Setup networking (if not provided)
            if not self.config.vpc_id or not self.config.subnet_ids:
                logger.info("Setting up networking infrastructure...")
                networking_result = self._setup_networking()
                deployment_result['components']['networking'] = networking_result
            
            # Step 2: Setup S3 storage
            logger.info("Setting up S3 storage...")
            s3_result = self._setup_s3()
            deployment_result['components']['s3'] = s3_result
            
            # Step 3: Setup CloudWatch monitoring
            logger.info("Setting up CloudWatch monitoring...")
            cloudwatch_result = self._setup_cloudwatch()
            deployment_result['components']['cloudwatch'] = cloudwatch_result
            
            # Step 4: Setup EKS cluster
            logger.info("Setting up EKS cluster...")
            eks_result = self._setup_eks()
            deployment_result['components']['eks'] = eks_result
            
            # Step 5: Deploy Pynomaly Detection
            logger.info("Deploying Pynomaly Detection application...")
            app_result = self._deploy_application()
            deployment_result['components']['application'] = app_result
            
            # Step 6: Setup monitoring and alerting
            logger.info("Setting up monitoring and alerting...")
            monitoring_result = self._setup_monitoring()
            deployment_result['components']['monitoring'] = monitoring_result
            
            # Step 7: Validate deployment
            logger.info("Validating deployment...")
            validation_result = self._validate_deployment()
            deployment_result['components']['validation'] = validation_result
            
            deployment_result['status'] = 'completed'
            deployment_result['completed_at'] = datetime.now().isoformat()
            
            logger.info(f"Full stack deployment completed: {self.config.deployment_name}")
            return deployment_result
            
        except Exception as e:
            logger.error(f"Full stack deployment failed: {e}")
            deployment_result['status'] = 'failed'
            deployment_result['error'] = str(e)
            deployment_result['failed_at'] = datetime.now().isoformat()
            return deployment_result
    
    def update_deployment(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing deployment.
        
        Args:
            updates: Updates to apply
            
        Returns:
            Update result
        """
        try:
            logger.info(f"Updating deployment: {self.config.deployment_name}")
            
            update_result = {
                'deployment_name': self.config.deployment_name,
                'updates': updates,
                'started_at': datetime.now().isoformat(),
                'status': 'in_progress'
            }
            
            # Update EKS cluster
            if 'eks' in updates:
                logger.info("Updating EKS cluster...")
                eks_updates = updates['eks']
                
                if 'scaling' in eks_updates:
                    scaling = eks_updates['scaling']
                    self.eks_manager.scale_cluster(
                        cluster_name=self.config.cluster_name,
                        node_group_name=f"{self.config.cluster_name}-nodes",
                        desired_size=scaling.get('desired_size', self.config.desired_nodes),
                        min_size=scaling.get('min_size', self.config.min_nodes),
                        max_size=scaling.get('max_size', self.config.max_nodes)
                    )
                
                if 'application_update' in eks_updates:
                    self._update_application(eks_updates['application_update'])
            
            # Update monitoring
            if 'monitoring' in updates:
                logger.info("Updating monitoring configuration...")
                self._update_monitoring(updates['monitoring'])
            
            update_result['status'] = 'completed'
            update_result['completed_at'] = datetime.now().isoformat()
            
            logger.info(f"Deployment update completed: {self.config.deployment_name}")
            return update_result
            
        except Exception as e:
            logger.error(f"Deployment update failed: {e}")
            update_result['status'] = 'failed'
            update_result['error'] = str(e)
            return update_result
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get deployment status and health.
        
        Returns:
            Deployment status
        """
        try:
            status = {
                'deployment_name': self.config.deployment_name,
                'region': self.config.region,
                'environment': self.config.environment,
                'checked_at': datetime.now().isoformat(),
                'components': {}
            }
            
            # EKS cluster status
            if self.eks_manager:
                eks_status = self.eks_manager.get_cluster_status(self.config.cluster_name)
                status['components']['eks'] = eks_status
            
            # S3 storage status
            if self.s3_manager:
                s3_metrics = self.s3_manager.get_storage_metrics()
                status['components']['s3'] = s3_metrics
            
            # CloudWatch monitoring status
            if self.cloudwatch_monitor:
                monitoring_summary = self.cloudwatch_monitor.get_monitoring_summary()
                status['components']['cloudwatch'] = monitoring_summary
            
            # Overall health
            status['health'] = self._assess_health(status['components'])
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {'error': str(e)}
    
    def destroy_deployment(self) -> Dict[str, Any]:
        """Destroy complete deployment.
        
        Returns:
            Destruction result
        """
        try:
            logger.info(f"Destroying deployment: {self.config.deployment_name}")
            
            destruction_result = {
                'deployment_name': self.config.deployment_name,
                'started_at': datetime.now().isoformat(),
                'status': 'in_progress',
                'destroyed_components': []
            }
            
            # Destroy EKS cluster
            if self.eks_manager:
                logger.info("Destroying EKS cluster...")
                eks_destroyed = self.eks_manager.delete_cluster(self.config.cluster_name)
                if eks_destroyed:
                    destruction_result['destroyed_components'].append('eks')
            
            # Clean up S3 (optional - may want to keep data)
            logger.info("S3 data preserved (manual cleanup required)")
            
            # Clean up CloudWatch resources
            logger.info("CloudWatch resources preserved (manual cleanup required)")
            
            destruction_result['status'] = 'completed'
            destruction_result['completed_at'] = datetime.now().isoformat()
            
            logger.info(f"Deployment destruction completed: {self.config.deployment_name}")
            return destruction_result
            
        except Exception as e:
            logger.error(f"Deployment destruction failed: {e}")
            destruction_result['status'] = 'failed'
            destruction_result['error'] = str(e)
            return destruction_result
    
    def _setup_networking(self) -> Dict[str, Any]:
        """Setup networking infrastructure."""
        try:
            ec2_client = self.session.client('ec2', region_name=self.config.region)
            
            # Create VPC if not provided
            if not self.config.vpc_id:
                vpc_response = ec2_client.create_vpc(
                    CidrBlock='10.0.0.0/16',
                    TagSpecifications=[
                        {
                            'ResourceType': 'vpc',
                            'Tags': [
                                {'Key': 'Name', 'Value': f'{self.config.deployment_name}-vpc'},
                                *[{'Key': k, 'Value': v} for k, v in self.config.tags.items()]
                            ]
                        }
                    ]
                )
                self.config.vpc_id = vpc_response['Vpc']['VpcId']
            
            # Create subnets if not provided
            if not self.config.subnet_ids:
                # Get availability zones
                azs_response = ec2_client.describe_availability_zones()
                azs = [az['ZoneName'] for az in azs_response['AvailabilityZones'][:3]]
                
                for i, az in enumerate(azs):
                    subnet_response = ec2_client.create_subnet(
                        VpcId=self.config.vpc_id,
                        CidrBlock=f'10.0.{i+1}.0/24',
                        AvailabilityZone=az,
                        TagSpecifications=[
                            {
                                'ResourceType': 'subnet',
                                'Tags': [
                                    {'Key': 'Name', 'Value': f'{self.config.deployment_name}-subnet-{i+1}'},
                                    *[{'Key': k, 'Value': v} for k, v in self.config.tags.items()]
                                ]
                            }
                        ]
                    )
                    self.config.subnet_ids.append(subnet_response['Subnet']['SubnetId'])
            
            return {
                'vpc_id': self.config.vpc_id,
                'subnet_ids': self.config.subnet_ids,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Networking setup failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _setup_s3(self) -> Dict[str, Any]:
        """Setup S3 storage."""
        try:
            s3_config = S3StorageConfig(
                bucket_name=self.config.bucket_name,
                region=self.config.region,
                versioning=self.config.enable_versioning,
                encryption="AES256" if self.config.enable_encryption else None
            )
            
            self.s3_manager = S3StorageManager(s3_config, self.profile_name)
            
            return {
                'bucket_name': self.config.bucket_name,
                'region': self.config.region,
                'versioning': self.config.enable_versioning,
                'encryption': self.config.enable_encryption,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"S3 setup failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _setup_cloudwatch(self) -> Dict[str, Any]:
        """Setup CloudWatch monitoring."""
        try:
            cloudwatch_config = CloudWatchConfig(
                region=self.config.region,
                namespace=self.config.namespace,
                log_group_name=f"/aws/pynomaly/{self.config.deployment_name}",
                retention_days=self.config.log_retention_days,
                enable_detailed_monitoring=self.config.enable_detailed_monitoring,
                dashboard_name=f"Pynomaly-{self.config.deployment_name}"
            )
            
            self.cloudwatch_monitor = CloudWatchMonitor(cloudwatch_config, self.profile_name)
            
            return {
                'namespace': self.config.namespace,
                'log_group': f"/aws/pynomaly/{self.config.deployment_name}",
                'dashboard': f"Pynomaly-{self.config.deployment_name}",
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"CloudWatch setup failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _setup_eks(self) -> Dict[str, Any]:
        """Setup EKS cluster."""
        try:
            self.eks_manager = EKSManager(self.config.region, self.profile_name)
            
            eks_config = EKSClusterConfig(
                cluster_name=self.config.cluster_name,
                region=self.config.region,
                kubernetes_version=self.config.kubernetes_version,
                node_group_name=f"{self.config.cluster_name}-nodes",
                instance_types=self.config.node_instance_types,
                min_size=self.config.min_nodes,
                max_size=self.config.max_nodes,
                desired_size=self.config.desired_nodes,
                subnet_ids=self.config.subnet_ids,
                security_group_ids=self.config.security_group_ids
            )
            
            cluster_response = self.eks_manager.create_cluster(eks_config)
            
            return {
                'cluster_name': self.config.cluster_name,
                'cluster_arn': cluster_response['cluster']['arn'],
                'endpoint': cluster_response['cluster']['endpoint'],
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"EKS setup failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _deploy_application(self) -> Dict[str, Any]:
        """Deploy Pynomaly Detection application."""
        try:
            namespace = f"pynomaly-{self.config.environment}"
            
            deployed = self.eks_manager.deploy_pynomaly(
                cluster_name=self.config.cluster_name,
                namespace=namespace
            )
            
            if deployed:
                return {
                    'namespace': namespace,
                    'cluster': self.config.cluster_name,
                    'status': 'completed'
                }
            else:
                return {'status': 'failed', 'error': 'Application deployment failed'}
                
        except Exception as e:
            logger.error(f"Application deployment failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _setup_monitoring(self) -> Dict[str, Any]:
        """Setup monitoring and alerting."""
        try:
            # Setup CloudWatch alarms
            alarms_setup = self.cloudwatch_monitor.setup_detection_alarms()
            
            # Create dashboard
            dashboard_created = self.cloudwatch_monitor.create_detection_dashboard()
            
            return {
                'alarms_setup': alarms_setup,
                'dashboard_created': dashboard_created,
                'dashboard_url': self.cloudwatch_monitor.get_dashboard_url(),
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _validate_deployment(self) -> Dict[str, Any]:
        """Validate deployment."""
        try:
            validation_results = {}
            
            # Validate EKS cluster
            cluster_status = self.eks_manager.get_cluster_status(self.config.cluster_name)
            validation_results['eks_cluster'] = cluster_status.get('status') == 'ACTIVE'
            
            # Validate S3 bucket
            s3_metrics = self.s3_manager.get_storage_metrics()
            validation_results['s3_bucket'] = bool(s3_metrics.get('bucket_name'))
            
            # Validate CloudWatch
            monitoring_summary = self.cloudwatch_monitor.get_monitoring_summary()
            validation_results['cloudwatch'] = bool(monitoring_summary)
            
            all_valid = all(validation_results.values())
            
            return {
                'overall_valid': all_valid,
                'component_validation': validation_results,
                'status': 'completed' if all_valid else 'partially_failed'
            }
            
        except Exception as e:
            logger.error(f"Deployment validation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _update_application(self, update_config: Dict[str, Any]):
        """Update application deployment."""
        # Implementation for application updates
        pass
    
    def _update_monitoring(self, monitoring_config: Dict[str, Any]):
        """Update monitoring configuration."""
        # Implementation for monitoring updates
        pass
    
    def _assess_health(self, components: Dict[str, Any]) -> str:
        """Assess overall deployment health."""
        try:
            health_scores = []
            
            # EKS health
            if 'eks' in components:
                eks_health = components['eks'].get('status') == 'ACTIVE'
                health_scores.append(eks_health)
            
            # S3 health
            if 's3' in components:
                s3_health = bool(components['s3'].get('bucket_name'))
                health_scores.append(s3_health)
            
            # CloudWatch health
            if 'cloudwatch' in components:
                cloudwatch_health = bool(components['cloudwatch'])
                health_scores.append(cloudwatch_health)
            
            if not health_scores:
                return 'unknown'
            
            healthy_ratio = sum(health_scores) / len(health_scores)
            
            if healthy_ratio >= 0.9:
                return 'healthy'
            elif healthy_ratio >= 0.7:
                return 'degraded'
            else:
                return 'unhealthy'
                
        except Exception as e:
            logger.error(f"Health assessment failed: {e}")
            return 'unknown'