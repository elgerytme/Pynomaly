"""
Cross-Cloud Deployment Framework

Advanced deployment orchestration across multiple cloud providers with
intelligent workload placement, automated failover, and cost optimization.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import subprocess
import yaml
from pathlib import Path
import hashlib
import base64

import boto3
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from google.cloud import container_v1
import kubernetes
from kubernetes import client, config as k8s_config
import docker
from prometheus_client import Counter, Histogram, Gauge
import structlog

from ..monitoring.advanced_observability_platform import AdvancedObservabilityPlatform


class DeploymentTarget(Enum):
    """Deployment target environments."""
    AWS_EKS = "aws_eks"
    AWS_FARGATE = "aws_fargate"
    AWS_EC2 = "aws_ec2"
    AZURE_AKS = "azure_aks"
    AZURE_CONTAINER_INSTANCES = "azure_container_instances"
    AZURE_VMS = "azure_vms"
    GCP_GKE = "gcp_gke"
    GCP_CLOUD_RUN = "gcp_cloud_run"
    GCP_COMPUTE_ENGINE = "gcp_compute_engine"
    EDGE_KUBERNETES = "edge_kubernetes"
    ON_PREMISES = "on_premises"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING_UPDATE = "rolling_update"
    RECREATE = "recreate"
    A_B_TEST = "a_b_test"
    SHADOW = "shadow"
    MULTI_CLOUD_ACTIVE = "multi_cloud_active"
    MULTI_CLOUD_STANDBY = "multi_cloud_standby"


class TrafficSplitStrategy(Enum):
    """Traffic splitting strategies."""
    PERCENTAGE = "percentage"
    HEADER_BASED = "header_based"
    GEOGRAPHIC = "geographic"
    DEVICE_TYPE = "device_type"
    USER_SEGMENT = "user_segment"
    FEATURE_FLAG = "feature_flag"


class FailoverMode(Enum):
    """Failover modes."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    HYBRID = "hybrid"


@dataclass
class CloudCredentials:
    """Cloud provider credentials."""
    provider: str
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: Optional[str] = None
    subscription_id: Optional[str] = None
    tenant_id: Optional[str] = None
    project_id: Optional[str] = None
    service_account_path: Optional[str] = None
    cluster_name: Optional[str] = None
    additional_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentSpec:
    """Deployment specification."""
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"
    
    # Application configuration
    container_image: str = ""
    container_tag: str = "latest"
    replicas: int = 1
    
    # Resource requirements
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    storage_size: str = "1Gi"
    
    # Networking
    ports: List[Dict[str, Any]] = field(default_factory=list)
    ingress_enabled: bool = True
    load_balancer_type: str = "application"
    
    # Environment configuration
    environment_variables: Dict[str, str] = field(default_factory=dict)
    config_maps: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)
    
    # Deployment settings
    targets: List[DeploymentTarget] = field(default_factory=list)
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    traffic_split: Dict[str, float] = field(default_factory=dict)
    
    # Health checks
    health_check_path: str = "/health"
    readiness_probe_path: str = "/ready"
    liveness_probe_path: str = "/health"
    
    # Scaling configuration
    hpa_enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    
    # Monitoring and observability
    monitoring_enabled: bool = True
    logging_enabled: bool = True
    tracing_enabled: bool = True
    
    # Security
    security_context: Dict[str, Any] = field(default_factory=dict)
    network_policies: List[str] = field(default_factory=list)
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentStatus:
    """Deployment status tracking."""
    deployment_id: str
    target: DeploymentTarget
    status: str = "pending"  # pending, deploying, deployed, failed, terminating
    
    # Progress tracking
    progress_percentage: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    
    # Resource information
    cluster_name: Optional[str] = None
    namespace: str = "default"
    service_urls: List[str] = field(default_factory=list)
    
    # Health and metrics
    healthy_replicas: int = 0
    total_replicas: int = 0
    last_health_check: Optional[datetime] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TrafficSplit:
    """Traffic splitting configuration."""
    strategy: TrafficSplitStrategy
    splits: Dict[str, float]  # target -> percentage
    conditions: Dict[str, Any] = field(default_factory=dict)
    sticky_sessions: bool = False
    session_timeout: int = 3600


@dataclass
class FailoverConfig:
    """Failover configuration."""
    mode: FailoverMode = FailoverMode.AUTOMATIC
    primary_target: DeploymentTarget
    secondary_targets: List[DeploymentTarget] = field(default_factory=list)
    
    # Health check configuration
    health_check_interval: int = 30
    failure_threshold: int = 3
    success_threshold: int = 2
    
    # Failover timing
    failover_timeout: int = 300
    recovery_timeout: int = 600
    
    # Data synchronization
    data_sync_enabled: bool = True
    sync_strategies: List[str] = field(default_factory=list)


class CrossCloudDeploymentOrchestrator:
    """Advanced cross-cloud deployment orchestration system."""
    
    def __init__(self, monitoring_platform: AdvancedObservabilityPlatform = None):
        self.logger = structlog.get_logger(__name__)
        self.monitoring = monitoring_platform
        
        # Cloud provider clients
        self.aws_clients: Dict[str, Any] = {}
        self.azure_clients: Dict[str, Any] = {}
        self.gcp_clients: Dict[str, Any] = {}
        self.k8s_clients: Dict[str, client.ApiClient] = {}
        
        # Deployment state
        self.deployments: Dict[str, DeploymentSpec] = {}
        self.deployment_statuses: Dict[str, List[DeploymentStatus]] = {}
        self.traffic_splits: Dict[str, TrafficSplit] = {}
        self.failover_configs: Dict[str, FailoverConfig] = {}
        
        # Orchestration engines
        self.kubernetes_deployer = KubernetesDeployer()
        self.docker_deployer = DockerDeployer()
        self.traffic_manager = TrafficManager()
        self.failover_manager = FailoverManager()
        
        # Background tasks
        self.orchestration_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Metrics
        self._init_metrics()
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        self.deployment_counter = Counter(
            'cross_cloud_deployments_total',
            'Total cross-cloud deployments',
            ['target', 'strategy', 'status']
        )
        
        self.deployment_duration = Histogram(
            'cross_cloud_deployment_duration_seconds',
            'Deployment duration',
            ['target', 'strategy']
        )
        
        self.active_deployments = Gauge(
            'cross_cloud_active_deployments',
            'Active deployments by target',
            ['target']
        )
        
        self.traffic_split_ratio = Gauge(
            'cross_cloud_traffic_split_ratio',
            'Traffic split ratios',
            ['deployment_id', 'target']
        )
    
    async def initialize_cloud_clients(self, credentials: List[CloudCredentials]) -> None:
        """Initialize cloud provider clients."""
        
        for cred in credentials:
            try:
                if cred.provider == "aws":
                    await self._init_aws_clients(cred)
                elif cred.provider == "azure":
                    await self._init_azure_clients(cred)
                elif cred.provider == "gcp":
                    await self._init_gcp_clients(cred)
                elif cred.provider == "kubernetes":
                    await self._init_k8s_clients(cred)
                
                self.logger.info(f"Initialized {cred.provider} client")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {cred.provider} client: {e}")
    
    async def _init_aws_clients(self, cred: CloudCredentials) -> None:
        """Initialize AWS clients."""
        session = boto3.Session(
            aws_access_key_id=cred.access_key,
            aws_secret_access_key=cred.secret_key,
            region_name=cred.region
        )
        
        self.aws_clients[cred.region] = {
            'eks': session.client('eks'),
            'ecs': session.client('ecs'),
            'ec2': session.client('ec2'),
            'elbv2': session.client('elbv2'),
            'route53': session.client('route53'),
            'cloudformation': session.client('cloudformation')
        }
    
    async def _init_azure_clients(self, cred: CloudCredentials) -> None:
        """Initialize Azure clients."""
        credential = DefaultAzureCredential()
        
        self.azure_clients[cred.subscription_id] = {
            'container_instances': ContainerInstanceManagementClient(
                credential, cred.subscription_id
            ),
            'credential': credential,
            'subscription_id': cred.subscription_id
        }
    
    async def _init_gcp_clients(self, cred: CloudCredentials) -> None:
        """Initialize GCP clients."""
        self.gcp_clients[cred.project_id] = {
            'container': container_v1.ClusterManagerClient(),
            'project_id': cred.project_id
        }
    
    async def _init_k8s_clients(self, cred: CloudCredentials) -> None:
        """Initialize Kubernetes clients."""
        try:
            if cred.cluster_name:
                # Load specific cluster config
                k8s_config.load_kube_config(context=cred.cluster_name)
            else:
                # Try in-cluster config first, then local config
                try:
                    k8s_config.load_incluster_config()
                except k8s_config.ConfigException:
                    k8s_config.load_kube_config()
            
            self.k8s_clients[cred.cluster_name or "default"] = client.ApiClient()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {e}")
    
    async def deploy(self, deployment_spec: DeploymentSpec) -> str:
        """Deploy application across specified targets."""
        
        deployment_id = deployment_spec.deployment_id
        self.deployments[deployment_id] = deployment_spec
        self.deployment_statuses[deployment_id] = []
        
        self.logger.info(
            f"Starting cross-cloud deployment {deployment_id}",
            targets=[t.value for t in deployment_spec.targets],
            strategy=deployment_spec.strategy.value
        )
        
        # Create deployment status for each target
        for target in deployment_spec.targets:
            status = DeploymentStatus(
                deployment_id=deployment_id,
                target=target,
                total_replicas=deployment_spec.replicas,
                started_at=datetime.utcnow()
            )
            self.deployment_statuses[deployment_id].append(status)
        
        # Start deployment tasks for each target
        deployment_tasks = []
        for target in deployment_spec.targets:
            task = asyncio.create_task(
                self._deploy_to_target(deployment_spec, target)
            )
            deployment_tasks.append(task)
        
        # Wait for all deployments to complete or handle strategy-specific logic
        if deployment_spec.strategy == DeploymentStrategy.BLUE_GREEN:
            await self._handle_blue_green_deployment(deployment_spec, deployment_tasks)
        elif deployment_spec.strategy == DeploymentStrategy.CANARY:
            await self._handle_canary_deployment(deployment_spec, deployment_tasks)
        elif deployment_spec.strategy == DeploymentStrategy.MULTI_CLOUD_ACTIVE:
            await self._handle_multi_cloud_active_deployment(deployment_spec, deployment_tasks)
        else:
            # Wait for all deployments
            await asyncio.gather(*deployment_tasks, return_exceptions=True)
        
        # Configure traffic splitting if specified
        if deployment_spec.traffic_split:
            await self._configure_traffic_split(deployment_spec)
        
        self.logger.info(f"Cross-cloud deployment {deployment_id} completed")
        return deployment_id
    
    async def _deploy_to_target(self, spec: DeploymentSpec, target: DeploymentTarget) -> None:
        """Deploy to a specific target environment."""
        
        status = self._get_deployment_status(spec.deployment_id, target)
        
        try:
            status.status = "deploying"
            status.current_step = f"Deploying to {target.value}"
            
            start_time = datetime.utcnow()
            
            if target in [DeploymentTarget.AWS_EKS, DeploymentTarget.AZURE_AKS, 
                         DeploymentTarget.GCP_GKE, DeploymentTarget.EDGE_KUBERNETES]:
                await self.kubernetes_deployer.deploy(spec, target, status)
            
            elif target in [DeploymentTarget.AWS_FARGATE, DeploymentTarget.AWS_EC2]:
                await self._deploy_to_aws_compute(spec, target, status)
            
            elif target in [DeploymentTarget.AZURE_CONTAINER_INSTANCES, DeploymentTarget.AZURE_VMS]:
                await self._deploy_to_azure_compute(spec, target, status)
            
            elif target in [DeploymentTarget.GCP_CLOUD_RUN, DeploymentTarget.GCP_COMPUTE_ENGINE]:
                await self._deploy_to_gcp_compute(spec, target, status)
            
            else:
                raise ValueError(f"Unsupported deployment target: {target}")
            
            # Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.deployment_duration.labels(
                target=target.value,
                strategy=spec.strategy.value
            ).observe(duration)
            
            self.deployment_counter.labels(
                target=target.value,
                strategy=spec.strategy.value,
                status="success"
            ).inc()
            
            status.status = "deployed"
            status.completed_at = datetime.utcnow()
            status.progress_percentage = 100.0
            
            self.logger.info(f"Successfully deployed to {target.value}")
            
        except Exception as e:
            status.status = "failed"
            status.error_message = str(e)
            status.completed_at = datetime.utcnow()
            
            self.deployment_counter.labels(
                target=target.value,
                strategy=spec.strategy.value,
                status="failed"
            ).inc()
            
            self.logger.error(f"Failed to deploy to {target.value}: {e}")
            
            # Retry logic
            if status.retry_count < status.max_retries:
                status.retry_count += 1
                self.logger.info(f"Retrying deployment to {target.value} (attempt {status.retry_count})")
                await asyncio.sleep(30)  # Wait before retry
                await self._deploy_to_target(spec, target)
    
    def _get_deployment_status(self, deployment_id: str, target: DeploymentTarget) -> DeploymentStatus:
        """Get deployment status for specific target."""
        for status in self.deployment_statuses[deployment_id]:
            if status.target == target:
                return status
        raise ValueError(f"Status not found for deployment {deployment_id} target {target}")
    
    async def _deploy_to_aws_compute(self, spec: DeploymentSpec, target: DeploymentTarget, status: DeploymentStatus) -> None:
        """Deploy to AWS compute services."""
        
        if target == DeploymentTarget.AWS_FARGATE:
            # Deploy to ECS Fargate
            await self._deploy_to_ecs_fargate(spec, status)
        elif target == DeploymentTarget.AWS_EC2:
            # Deploy to EC2 instances
            await self._deploy_to_ec2(spec, status)
    
    async def _deploy_to_ecs_fargate(self, spec: DeploymentSpec, status: DeploymentStatus) -> None:
        """Deploy to AWS ECS Fargate."""
        
        ecs_client = self.aws_clients[list(self.aws_clients.keys())[0]]['ecs']
        
        # Create task definition
        task_definition = {
            'family': spec.name,
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': '256',
            'memory': '512',
            'containerDefinitions': [
                {
                    'name': spec.name,
                    'image': f"{spec.container_image}:{spec.container_tag}",
                    'portMappings': spec.ports,
                    'environment': [
                        {'name': k, 'value': v} for k, v in spec.environment_variables.items()
                    ],
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': f'/ecs/{spec.name}',
                            'awslogs-region': 'us-east-1',
                            'awslogs-stream-prefix': 'ecs'
                        }
                    }
                }
            ]
        }
        
        # Register task definition
        response = ecs_client.register_task_definition(**task_definition)
        task_def_arn = response['taskDefinition']['taskDefinitionArn']
        
        # Create or update service
        try:
            ecs_client.create_service(
                cluster='default',
                serviceName=spec.name,
                taskDefinition=task_def_arn,
                desiredCount=spec.replicas,
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': ['subnet-12345'],  # Configure actual subnets
                        'assignPublicIp': 'ENABLED'
                    }
                }
            )
        except ecs_client.exceptions.ServiceAlreadyExistsException:
            ecs_client.update_service(
                cluster='default',
                service=spec.name,
                taskDefinition=task_def_arn,
                desiredCount=spec.replicas
            )
        
        status.cluster_name = "default"
        status.namespace = "ecs"
    
    async def _deploy_to_azure_compute(self, spec: DeploymentSpec, target: DeploymentTarget, status: DeploymentStatus) -> None:
        """Deploy to Azure compute services."""
        
        if target == DeploymentTarget.AZURE_CONTAINER_INSTANCES:
            await self._deploy_to_azure_container_instances(spec, status)
        elif target == DeploymentTarget.AZURE_VMS:
            await self._deploy_to_azure_vms(spec, status)
    
    async def _deploy_to_azure_container_instances(self, spec: DeploymentSpec, status: DeploymentStatus) -> None:
        """Deploy to Azure Container Instances."""
        
        aci_client = self.azure_clients[list(self.azure_clients.keys())[0]]['container_instances']
        
        container_group = {
            'location': 'eastus',
            'containers': [
                {
                    'name': spec.name,
                    'image': f"{spec.container_image}:{spec.container_tag}",
                    'resources': {
                        'requests': {
                            'cpu': 1.0,
                            'memory_in_gb': 1.0
                        }
                    },
                    'ports': [{'port': port.get('containerPort', 80)} for port in spec.ports],
                    'environment_variables': [
                        {'name': k, 'value': v} for k, v in spec.environment_variables.items()
                    ]
                }
            ],
            'os_type': 'Linux',
            'ip_address': {
                'type': 'Public',
                'ports': [{'port': port.get('containerPort', 80), 'protocol': 'TCP'} for port in spec.ports]
            }
        }
        
        # Create container group
        operation = aci_client.container_groups.begin_create_or_update(
            resource_group_name='default-rg',
            container_group_name=spec.name,
            container_group=container_group
        )
        
        # Wait for deployment
        result = operation.result()
        
        status.cluster_name = "aci"
        status.service_urls = [f"http://{result.ip_address.ip}"]
    
    async def _deploy_to_gcp_compute(self, spec: DeploymentSpec, target: DeploymentTarget, status: DeploymentStatus) -> None:
        """Deploy to GCP compute services."""
        
        if target == DeploymentTarget.GCP_CLOUD_RUN:
            await self._deploy_to_cloud_run(spec, status)
        elif target == DeploymentTarget.GCP_COMPUTE_ENGINE:
            await self._deploy_to_compute_engine(spec, status)
    
    async def _deploy_to_cloud_run(self, spec: DeploymentSpec, status: DeploymentStatus) -> None:
        """Deploy to Google Cloud Run."""
        
        # Use gcloud CLI for Cloud Run deployment
        gcloud_command = [
            'gcloud', 'run', 'deploy', spec.name,
            '--image', f"{spec.container_image}:{spec.container_tag}",
            '--platform', 'managed',
            '--region', 'us-central1',
            '--allow-unauthenticated',
            '--format', 'json'
        ]
        
        # Add environment variables
        for key, value in spec.environment_variables.items():
            gcloud_command.extend(['--set-env-vars', f"{key}={value}"])
        
        # Execute deployment
        result = subprocess.run(gcloud_command, capture_output=True, text=True)
        
        if result.returncode == 0:
            deployment_info = json.loads(result.stdout)
            status.service_urls = [deployment_info.get('status', {}).get('url', '')]
        else:
            raise Exception(f"Cloud Run deployment failed: {result.stderr}")
    
    async def _handle_blue_green_deployment(self, spec: DeploymentSpec, deployment_tasks: List[asyncio.Task]) -> None:
        """Handle blue-green deployment strategy."""
        
        self.logger.info("Executing blue-green deployment strategy")
        
        # Deploy to all targets (green environment)
        await asyncio.gather(*deployment_tasks, return_exceptions=True)
        
        # Wait for health checks to pass
        await self._wait_for_health_checks(spec.deployment_id)
        
        # Switch traffic to green environment
        await self._switch_traffic(spec.deployment_id, "green")
        
        # Optional: cleanup old blue environment
        # await self._cleanup_old_environment(spec.deployment_id, "blue")
    
    async def _handle_canary_deployment(self, spec: DeploymentSpec, deployment_tasks: List[asyncio.Task]) -> None:
        """Handle canary deployment strategy."""
        
        self.logger.info("Executing canary deployment strategy")
        
        # Deploy canary version to subset of targets
        canary_targets = spec.targets[:1]  # Deploy to first target as canary
        canary_tasks = deployment_tasks[:1]
        
        await asyncio.gather(*canary_tasks, return_exceptions=True)
        
        # Configure traffic split (e.g., 10% to canary)
        traffic_split = TrafficSplit(
            strategy=TrafficSplitStrategy.PERCENTAGE,
            splits={"canary": 0.1, "stable": 0.9}
        )
        self.traffic_splits[spec.deployment_id] = traffic_split
        
        # Monitor canary metrics
        await self._monitor_canary_metrics(spec.deployment_id)
        
        # If canary is successful, deploy to remaining targets
        if await self._is_canary_successful(spec.deployment_id):
            remaining_tasks = deployment_tasks[1:]
            await asyncio.gather(*remaining_tasks, return_exceptions=True)
            
            # Gradually shift traffic to new version
            await self._gradual_traffic_shift(spec.deployment_id)
        else:
            # Rollback canary
            await self._rollback_canary(spec.deployment_id)
    
    async def _handle_multi_cloud_active_deployment(self, spec: DeploymentSpec, deployment_tasks: List[asyncio.Task]) -> None:
        """Handle multi-cloud active deployment."""
        
        self.logger.info("Executing multi-cloud active deployment strategy")
        
        # Deploy to all targets simultaneously
        await asyncio.gather(*deployment_tasks, return_exceptions=True)
        
        # Configure traffic splitting across clouds
        equal_split = 1.0 / len(spec.targets)
        traffic_split = TrafficSplit(
            strategy=TrafficSplitStrategy.GEOGRAPHIC,
            splits={target.value: equal_split for target in spec.targets}
        )
        self.traffic_splits[spec.deployment_id] = traffic_split
        
        # Configure failover between clouds
        failover_config = FailoverConfig(
            mode=FailoverMode.AUTOMATIC,
            primary_target=spec.targets[0],
            secondary_targets=spec.targets[1:]
        )
        self.failover_configs[spec.deployment_id] = failover_config
        
        # Start monitoring for failover
        await self._start_failover_monitoring(spec.deployment_id)
    
    async def _configure_traffic_split(self, spec: DeploymentSpec) -> None:
        """Configure traffic splitting for deployment."""
        
        traffic_split = TrafficSplit(
            strategy=TrafficSplitStrategy.PERCENTAGE,
            splits=spec.traffic_split
        )
        
        self.traffic_splits[spec.deployment_id] = traffic_split
        await self.traffic_manager.configure_split(spec.deployment_id, traffic_split)
        
        # Update metrics
        for target, ratio in spec.traffic_split.items():
            self.traffic_split_ratio.labels(
                deployment_id=spec.deployment_id,
                target=target
            ).set(ratio)
    
    async def _wait_for_health_checks(self, deployment_id: str) -> None:
        """Wait for all deployments to pass health checks."""
        
        timeout = 600  # 10 minutes
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            all_healthy = True
            
            for status in self.deployment_statuses[deployment_id]:
                if not await self._check_deployment_health(status):
                    all_healthy = False
                    break
            
            if all_healthy:
                self.logger.info(f"All deployments healthy for {deployment_id}")
                return
            
            await asyncio.sleep(30)
        
        raise Exception(f"Health check timeout for deployment {deployment_id}")
    
    async def _check_deployment_health(self, status: DeploymentStatus) -> bool:
        """Check health of a specific deployment."""
        
        try:
            if status.target in [DeploymentTarget.AWS_EKS, DeploymentTarget.AZURE_AKS, 
                               DeploymentTarget.GCP_GKE, DeploymentTarget.EDGE_KUBERNETES]:
                return await self.kubernetes_deployer.check_health(status)
            else:
                # For other targets, implement specific health checks
                return status.status == "deployed"
                
        except Exception as e:
            self.logger.error(f"Health check failed for {status.deployment_id}: {e}")
            return False
    
    async def _monitor_canary_metrics(self, deployment_id: str) -> None:
        """Monitor canary deployment metrics."""
        
        # Monitor for 10 minutes
        monitoring_duration = 600
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < monitoring_duration:
            # Check error rates, latency, and other metrics
            metrics = await self._get_canary_metrics(deployment_id)
            
            if metrics.get('error_rate', 0) > 0.05:  # 5% error rate threshold
                self.logger.warning(f"High error rate detected in canary {deployment_id}")
                return
            
            if metrics.get('latency_p99', 0) > 1000:  # 1 second latency threshold
                self.logger.warning(f"High latency detected in canary {deployment_id}")
                return
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _get_canary_metrics(self, deployment_id: str) -> Dict[str, float]:
        """Get metrics for canary deployment."""
        
        # Placeholder for metrics collection
        # In production, integrate with monitoring platform
        return {
            'error_rate': 0.01,
            'latency_p99': 250,
            'throughput': 100
        }
    
    async def _is_canary_successful(self, deployment_id: str) -> bool:
        """Determine if canary deployment is successful."""
        
        metrics = await self._get_canary_metrics(deployment_id)
        
        # Define success criteria
        return (
            metrics.get('error_rate', 1.0) < 0.05 and
            metrics.get('latency_p99', 2000) < 1000
        )
    
    async def _gradual_traffic_shift(self, deployment_id: str) -> None:
        """Gradually shift traffic to new version."""
        
        traffic_percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
        
        for percentage in traffic_percentages:
            # Update traffic split
            traffic_split = TrafficSplit(
                strategy=TrafficSplitStrategy.PERCENTAGE,
                splits={"new": percentage, "old": 1.0 - percentage}
            )
            
            await self.traffic_manager.configure_split(deployment_id, traffic_split)
            
            # Wait and monitor
            await asyncio.sleep(300)  # 5 minutes
            
            if not await self._is_canary_successful(deployment_id):
                await self._rollback_canary(deployment_id)
                return
        
        self.logger.info(f"Traffic shift completed for {deployment_id}")
    
    async def _rollback_canary(self, deployment_id: str) -> None:
        """Rollback canary deployment."""
        
        self.logger.warning(f"Rolling back canary deployment {deployment_id}")
        
        # Revert traffic to stable version
        traffic_split = TrafficSplit(
            strategy=TrafficSplitStrategy.PERCENTAGE,
            splits={"canary": 0.0, "stable": 1.0}
        )
        
        await self.traffic_manager.configure_split(deployment_id, traffic_split)
        
        # Terminate canary instances
        for status in self.deployment_statuses[deployment_id]:
            if "canary" in status.cluster_name or "canary" in status.namespace:
                await self._terminate_deployment(status)
    
    async def _start_failover_monitoring(self, deployment_id: str) -> None:
        """Start monitoring for automatic failover."""
        
        task = asyncio.create_task(
            self.failover_manager.monitor_and_failover(
                deployment_id, 
                self.failover_configs[deployment_id]
            )
        )
        self.orchestration_tasks.append(task)
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get comprehensive deployment status."""
        
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        spec = self.deployments[deployment_id]
        statuses = self.deployment_statuses.get(deployment_id, [])
        
        # Calculate overall progress
        total_progress = sum(status.progress_percentage for status in statuses)
        overall_progress = total_progress / len(statuses) if statuses else 0
        
        # Count statuses
        status_counts = {}
        for status in statuses:
            status_counts[status.status] = status_counts.get(status.status, 0) + 1
        
        return {
            'deployment_id': deployment_id,
            'name': spec.name,
            'version': spec.version,
            'strategy': spec.strategy.value,
            'overall_progress': overall_progress,
            'status_counts': status_counts,
            'targets': [
                {
                    'target': status.target.value,
                    'status': status.status,
                    'progress': status.progress_percentage,
                    'healthy_replicas': status.healthy_replicas,
                    'total_replicas': status.total_replicas,
                    'service_urls': status.service_urls,
                    'error_message': status.error_message
                }
                for status in statuses
            ],
            'traffic_split': asdict(self.traffic_splits.get(deployment_id)) if deployment_id in self.traffic_splits else None,
            'failover_config': asdict(self.failover_configs.get(deployment_id)) if deployment_id in self.failover_configs else None
        }
    
    async def scale_deployment(self, deployment_id: str, target: DeploymentTarget, replicas: int) -> None:
        """Scale a specific deployment target."""
        
        status = self._get_deployment_status(deployment_id, target)
        
        if target in [DeploymentTarget.AWS_EKS, DeploymentTarget.AZURE_AKS, 
                     DeploymentTarget.GCP_GKE, DeploymentTarget.EDGE_KUBERNETES]:
            await self.kubernetes_deployer.scale(status, replicas)
        else:
            # Implement scaling for other compute targets
            await self._scale_compute_target(status, replicas)
        
        status.total_replicas = replicas
        self.logger.info(f"Scaled {deployment_id} on {target.value} to {replicas} replicas")
    
    async def update_traffic_split(self, deployment_id: str, new_splits: Dict[str, float]) -> None:
        """Update traffic splitting configuration."""
        
        if deployment_id not in self.traffic_splits:
            self.traffic_splits[deployment_id] = TrafficSplit(
                strategy=TrafficSplitStrategy.PERCENTAGE,
                splits=new_splits
            )
        else:
            self.traffic_splits[deployment_id].splits = new_splits
        
        await self.traffic_manager.configure_split(
            deployment_id, 
            self.traffic_splits[deployment_id]
        )
        
        # Update metrics
        for target, ratio in new_splits.items():
            self.traffic_split_ratio.labels(
                deployment_id=deployment_id,
                target=target
            ).set(ratio)
        
        self.logger.info(f"Updated traffic split for {deployment_id}: {new_splits}")
    
    async def trigger_failover(self, deployment_id: str, from_target: DeploymentTarget, to_target: DeploymentTarget) -> None:
        """Manually trigger failover between targets."""
        
        self.logger.info(f"Triggering failover for {deployment_id}: {from_target.value} -> {to_target.value}")
        
        await self.failover_manager.execute_failover(
            deployment_id,
            from_target,
            to_target,
            self.deployment_statuses[deployment_id]
        )
    
    async def rollback_deployment(self, deployment_id: str, to_version: str = None) -> None:
        """Rollback deployment to previous version."""
        
        self.logger.info(f"Rolling back deployment {deployment_id} to version {to_version or 'previous'}")
        
        for status in self.deployment_statuses[deployment_id]:
            await self._rollback_target(status, to_version)
    
    async def _rollback_target(self, status: DeploymentStatus, to_version: str = None) -> None:
        """Rollback specific target to previous version."""
        
        if status.target in [DeploymentTarget.AWS_EKS, DeploymentTarget.AZURE_AKS, 
                           DeploymentTarget.GCP_GKE, DeploymentTarget.EDGE_KUBERNETES]:
            await self.kubernetes_deployer.rollback(status, to_version)
        else:
            # Implement rollback for other targets
            pass
    
    async def terminate_deployment(self, deployment_id: str) -> None:
        """Terminate entire deployment across all targets."""
        
        self.logger.info(f"Terminating deployment {deployment_id}")
        
        for status in self.deployment_statuses[deployment_id]:
            await self._terminate_deployment(status)
        
        # Clean up state
        if deployment_id in self.deployments:
            del self.deployments[deployment_id]
        if deployment_id in self.deployment_statuses:
            del self.deployment_statuses[deployment_id]
        if deployment_id in self.traffic_splits:
            del self.traffic_splits[deployment_id]
        if deployment_id in self.failover_configs:
            del self.failover_configs[deployment_id]
    
    async def _terminate_deployment(self, status: DeploymentStatus) -> None:
        """Terminate deployment on specific target."""
        
        try:
            if status.target in [DeploymentTarget.AWS_EKS, DeploymentTarget.AZURE_AKS, 
                               DeploymentTarget.GCP_GKE, DeploymentTarget.EDGE_KUBERNETES]:
                await self.kubernetes_deployer.terminate(status)
            else:
                # Implement termination for other targets
                pass
            
            status.status = "terminated"
            self.logger.info(f"Terminated deployment on {status.target.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to terminate deployment on {status.target.value}: {e}")


class KubernetesDeployer:
    """Kubernetes-specific deployment operations."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    async def deploy(self, spec: DeploymentSpec, target: DeploymentTarget, status: DeploymentStatus) -> None:
        """Deploy to Kubernetes cluster."""
        
        # Create namespace if it doesn't exist
        await self._ensure_namespace(spec.name)
        
        # Deploy application
        await self._deploy_application(spec, status)
        
        # Create service
        await self._create_service(spec, status)
        
        # Create ingress if enabled
        if spec.ingress_enabled:
            await self._create_ingress(spec, status)
        
        # Configure HPA if enabled
        if spec.hpa_enabled:
            await self._create_hpa(spec, status)
    
    async def _ensure_namespace(self, namespace: str) -> None:
        """Ensure namespace exists."""
        
        v1 = client.CoreV1Api()
        
        try:
            v1.read_namespace(namespace)
        except client.exceptions.ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace_obj = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
                v1.create_namespace(namespace_obj)
    
    async def _deploy_application(self, spec: DeploymentSpec, status: DeploymentStatus) -> None:
        """Deploy application to Kubernetes."""
        
        apps_v1 = client.AppsV1Api()
        
        # Create deployment manifest
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=spec.name,
                namespace=spec.name,
                labels=spec.labels
            ),
            spec=client.V1DeploymentSpec(
                replicas=spec.replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": spec.name}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": spec.name}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name=spec.name,
                                image=f"{spec.container_image}:{spec.container_tag}",
                                ports=[
                                    client.V1ContainerPort(container_port=port.get('containerPort', 80))
                                    for port in spec.ports
                                ],
                                env=[
                                    client.V1EnvVar(name=k, value=v)
                                    for k, v in spec.environment_variables.items()
                                ],
                                resources=client.V1ResourceRequirements(
                                    requests={
                                        "cpu": spec.cpu_request,
                                        "memory": spec.memory_request
                                    },
                                    limits={
                                        "cpu": spec.cpu_limit,
                                        "memory": spec.memory_limit
                                    }
                                ),
                                liveness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path=spec.liveness_probe_path,
                                        port=spec.ports[0].get('containerPort', 80) if spec.ports else 80
                                    ),
                                    initial_delay_seconds=30,
                                    period_seconds=10
                                ),
                                readiness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path=spec.readiness_probe_path,
                                        port=spec.ports[0].get('containerPort', 80) if spec.ports else 80
                                    ),
                                    initial_delay_seconds=5,
                                    period_seconds=5
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        # Create or update deployment
        try:
            apps_v1.create_namespaced_deployment(
                namespace=spec.name,
                body=deployment
            )
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                apps_v1.patch_namespaced_deployment(
                    name=spec.name,
                    namespace=spec.name,
                    body=deployment
                )
        
        status.namespace = spec.name
    
    async def _create_service(self, spec: DeploymentSpec, status: DeploymentStatus) -> None:
        """Create Kubernetes service."""
        
        v1 = client.CoreV1Api()
        
        service = client.V1Service(
            metadata=client.V1ObjectMeta(
                name=spec.name,
                namespace=spec.name
            ),
            spec=client.V1ServiceSpec(
                selector={"app": spec.name},
                ports=[
                    client.V1ServicePort(
                        port=port.get('port', 80),
                        target_port=port.get('containerPort', 80),
                        name=port.get('name', f"port-{i}")
                    )
                    for i, port in enumerate(spec.ports)
                ],
                type="LoadBalancer" if spec.load_balancer_type == "network" else "ClusterIP"
            )
        )
        
        try:
            v1.create_namespaced_service(
                namespace=spec.name,
                body=service
            )
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                v1.patch_namespaced_service(
                    name=spec.name,
                    namespace=spec.name,
                    body=service
                )
    
    async def _create_ingress(self, spec: DeploymentSpec, status: DeploymentStatus) -> None:
        """Create Kubernetes ingress."""
        
        networking_v1 = client.NetworkingV1Api()
        
        ingress = client.V1Ingress(
            metadata=client.V1ObjectMeta(
                name=spec.name,
                namespace=spec.name,
                annotations={
                    "kubernetes.io/ingress.class": "nginx",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                }
            ),
            spec=client.V1IngressSpec(
                rules=[
                    client.V1IngressRule(
                        host=f"{spec.name}.example.com",  # Configure actual domain
                        http=client.V1HTTPIngressRuleValue(
                            paths=[
                                client.V1HTTPIngressPath(
                                    path="/",
                                    path_type="Prefix",
                                    backend=client.V1IngressBackend(
                                        service=client.V1IngressServiceBackend(
                                            name=spec.name,
                                            port=client.V1ServiceBackendPort(
                                                number=spec.ports[0].get('port', 80) if spec.ports else 80
                                            )
                                        )
                                    )
                                )
                            ]
                        )
                    )
                ]
            )
        )
        
        try:
            networking_v1.create_namespaced_ingress(
                namespace=spec.name,
                body=ingress
            )
            status.service_urls.append(f"https://{spec.name}.example.com")
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                networking_v1.patch_namespaced_ingress(
                    name=spec.name,
                    namespace=spec.name,
                    body=ingress
                )
    
    async def _create_hpa(self, spec: DeploymentSpec, status: DeploymentStatus) -> None:
        """Create Horizontal Pod Autoscaler."""
        
        autoscaling_v2 = client.AutoscalingV2Api()
        
        hpa = client.V2HorizontalPodAutoscaler(
            metadata=client.V1ObjectMeta(
                name=spec.name,
                namespace=spec.name
            ),
            spec=client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=spec.name
                ),
                min_replicas=spec.min_replicas,
                max_replicas=spec.max_replicas,
                metrics=[
                    client.V2MetricSpec(
                        type="Resource",
                        resource=client.V2ResourceMetricSource(
                            name="cpu",
                            target=client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=spec.target_cpu_utilization
                            )
                        )
                    )
                ]
            )
        )
        
        try:
            autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=spec.name,
                body=hpa
            )
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(
                    name=spec.name,
                    namespace=spec.name,
                    body=hpa
                )
    
    async def check_health(self, status: DeploymentStatus) -> bool:
        """Check health of Kubernetes deployment."""
        
        try:
            apps_v1 = client.AppsV1Api()
            
            deployment = apps_v1.read_namespaced_deployment(
                name=status.deployment_id,
                namespace=status.namespace
            )
            
            if deployment.status:
                status.healthy_replicas = deployment.status.ready_replicas or 0
                status.total_replicas = deployment.status.replicas or 0
                
                return status.healthy_replicas == status.total_replicas
            
            return False
            
        except Exception as e:
            return False
    
    async def scale(self, status: DeploymentStatus, replicas: int) -> None:
        """Scale Kubernetes deployment."""
        
        apps_v1 = client.AppsV1Api()
        
        # Update deployment replicas
        deployment = apps_v1.read_namespaced_deployment(
            name=status.deployment_id,
            namespace=status.namespace
        )
        
        deployment.spec.replicas = replicas
        
        apps_v1.patch_namespaced_deployment(
            name=status.deployment_id,
            namespace=status.namespace,
            body=deployment
        )
    
    async def rollback(self, status: DeploymentStatus, to_version: str = None) -> None:
        """Rollback Kubernetes deployment."""
        
        apps_v1 = client.AppsV1Api()
        
        # Rollback to previous revision
        # This is a simplified implementation
        deployment = apps_v1.read_namespaced_deployment(
            name=status.deployment_id,
            namespace=status.namespace
        )
        
        # In production, implement proper rollback logic with revision history
        # For now, just restart the deployment
        deployment.spec.template.metadata.annotations = {
            "kubectl.kubernetes.io/restartedAt": datetime.utcnow().isoformat()
        }
        
        apps_v1.patch_namespaced_deployment(
            name=status.deployment_id,
            namespace=status.namespace,
            body=deployment
        )
    
    async def terminate(self, status: DeploymentStatus) -> None:
        """Terminate Kubernetes deployment."""
        
        apps_v1 = client.AppsV1Api()
        v1 = client.CoreV1Api()
        
        # Delete deployment
        try:
            apps_v1.delete_namespaced_deployment(
                name=status.deployment_id,
                namespace=status.namespace
            )
        except client.exceptions.ApiException:
            pass
        
        # Delete service
        try:
            v1.delete_namespaced_service(
                name=status.deployment_id,
                namespace=status.namespace
            )
        except client.exceptions.ApiException:
            pass


class DockerDeployer:
    """Docker-specific deployment operations."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.docker_client = docker.from_env()
    
    async def deploy(self, spec: DeploymentSpec, target: DeploymentTarget, status: DeploymentStatus) -> None:
        """Deploy using Docker."""
        
        # Pull image
        self.docker_client.images.pull(f"{spec.container_image}:{spec.container_tag}")
        
        # Run container
        container = self.docker_client.containers.run(
            f"{spec.container_image}:{spec.container_tag}",
            name=spec.name,
            environment=spec.environment_variables,
            ports={f"{port.get('containerPort', 80)}/tcp": port.get('port', 80) for port in spec.ports},
            detach=True,
            restart_policy={"Name": "always"}
        )
        
        status.cluster_name = "docker"
        status.namespace = container.id
        status.service_urls = [f"http://localhost:{spec.ports[0].get('port', 80)}"] if spec.ports else []


class TrafficManager:
    """Manages traffic splitting and routing."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    async def configure_split(self, deployment_id: str, traffic_split: TrafficSplit) -> None:
        """Configure traffic splitting."""
        
        self.logger.info(f"Configuring traffic split for {deployment_id}: {traffic_split.splits}")
        
        # In production, this would configure:
        # - Load balancer weights
        # - Service mesh routing rules
        # - DNS-based routing
        # - CDN configurations
        
        # For now, log the configuration
        for target, percentage in traffic_split.splits.items():
            self.logger.info(f"Routing {percentage*100:.1f}% traffic to {target}")


class FailoverManager:
    """Manages automatic failover between deployment targets."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    async def monitor_and_failover(self, deployment_id: str, config: FailoverConfig) -> None:
        """Monitor deployment health and trigger failover if needed."""
        
        failure_count = 0
        
        while True:
            try:
                # Check primary target health
                primary_healthy = await self._check_target_health(
                    deployment_id, 
                    config.primary_target
                )
                
                if primary_healthy:
                    failure_count = 0
                else:
                    failure_count += 1
                    self.logger.warning(
                        f"Health check failed for primary target {config.primary_target.value} "
                        f"(failure {failure_count}/{config.failure_threshold})"
                    )
                    
                    if failure_count >= config.failure_threshold:
                        # Trigger failover
                        await self._trigger_automatic_failover(deployment_id, config)
                        failure_count = 0
                
                await asyncio.sleep(config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in failover monitoring: {e}")
                await asyncio.sleep(config.health_check_interval)
    
    async def _check_target_health(self, deployment_id: str, target: DeploymentTarget) -> bool:
        """Check health of specific deployment target."""
        
        # In production, implement actual health checks:
        # - HTTP health endpoints
        # - Kubernetes pod status
        # - Cloud provider health checks
        # - Custom application metrics
        
        return True  # Placeholder
    
    async def _trigger_automatic_failover(self, deployment_id: str, config: FailoverConfig) -> None:
        """Trigger automatic failover to secondary target."""
        
        if not config.secondary_targets:
            self.logger.error(f"No secondary targets available for failover of {deployment_id}")
            return
        
        secondary_target = config.secondary_targets[0]
        
        self.logger.critical(
            f"Triggering automatic failover for {deployment_id}: "
            f"{config.primary_target.value} -> {secondary_target.value}"
        )
        
        await self.execute_failover(deployment_id, config.primary_target, secondary_target, [])
    
    async def execute_failover(self, 
                             deployment_id: str, 
                             from_target: DeploymentTarget, 
                             to_target: DeploymentTarget,
                             deployment_statuses: List[DeploymentStatus]) -> None:
        """Execute failover between targets."""
        
        self.logger.info(f"Executing failover: {from_target.value} -> {to_target.value}")
        
        # 1. Redirect traffic to healthy target
        # 2. Scale up healthy target if needed
        # 3. Drain traffic from failed target
        # 4. Update monitoring and alerting
        
        # In production, implement actual failover logic
        self.logger.info(f"Failover completed: {deployment_id}")


# Example usage and configuration
def create_sample_deployment_spec() -> DeploymentSpec:
    """Create a sample deployment specification."""
    
    return DeploymentSpec(
        name="ml-inference-service",
        version="1.2.0",
        container_image="myregistry/ml-inference",
        container_tag="v1.2.0",
        replicas=3,
        cpu_request="200m",
        cpu_limit="1000m",
        memory_request="256Mi",
        memory_limit="1Gi",
        ports=[
            {"containerPort": 8080, "port": 80, "name": "http"},
            {"containerPort": 9090, "port": 9090, "name": "metrics"}
        ],
        environment_variables={
            "MODEL_PATH": "/models/inference",
            "LOG_LEVEL": "INFO",
            "METRICS_ENABLED": "true"
        },
        targets=[
            DeploymentTarget.AWS_EKS,
            DeploymentTarget.AZURE_AKS,
            DeploymentTarget.GCP_GKE
        ],
        strategy=DeploymentStrategy.MULTI_CLOUD_ACTIVE,
        traffic_split={
            "aws": 0.4,
            "azure": 0.3,
            "gcp": 0.3
        },
        hpa_enabled=True,
        min_replicas=2,
        max_replicas=20,
        target_cpu_utilization=70,
        monitoring_enabled=True,
        labels={
            "app": "ml-inference",
            "tier": "production",
            "version": "1.2.0"
        }
    )


async def example_cross_cloud_deployment():
    """Example of cross-cloud deployment orchestration."""
    
    # Initialize orchestrator
    orchestrator = CrossCloudDeploymentOrchestrator()
    
    # Configure cloud credentials
    credentials = [
        CloudCredentials(
            provider="aws",
            access_key="your-aws-access-key",
            secret_key="your-aws-secret-key",
            region="us-east-1"
        ),
        CloudCredentials(
            provider="azure",
            subscription_id="your-azure-subscription-id",
            tenant_id="your-azure-tenant-id"
        ),
        CloudCredentials(
            provider="gcp",
            project_id="your-gcp-project-id",
            service_account_path="path/to/service-account.json"
        ),
        CloudCredentials(
            provider="kubernetes",
            cluster_name="production-cluster"
        )
    ]
    
    await orchestrator.initialize_cloud_clients(credentials)
    
    # Create deployment specification
    deployment_spec = create_sample_deployment_spec()
    
    # Deploy across clouds
    deployment_id = await orchestrator.deploy(deployment_spec)
    
    # Monitor deployment status
    while True:
        status = await orchestrator.get_deployment_status(deployment_id)
        print(f"Deployment progress: {status['overall_progress']:.1f}%")
        
        if status['overall_progress'] >= 100:
            break
        
        await asyncio.sleep(30)
    
    print("Cross-cloud deployment completed successfully!")
    
    return orchestrator, deployment_id