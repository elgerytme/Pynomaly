#!/usr/bin/env python3
"""
Streaming Orchestration System
Coordinates and manages all real-time streaming components with comprehensive monitoring.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import yaml

import redis.asyncio as redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
from kubernetes import client, config as k8s_config
import aiohttp


# Metrics
ORCHESTRATOR_OPERATIONS = Counter('orchestration_operations_total', 'Total orchestration operations', ['operation', 'status'])
COMPONENT_HEALTH = Gauge('streaming_component_health', 'Component health status', ['component', 'instance'])
ORCHESTRATION_TIME = Histogram('orchestration_processing_seconds', 'Time spent on orchestration', ['operation'])
ACTIVE_COMPONENTS = Gauge('streaming_active_components', 'Number of active streaming components', ['component_type'])
THROUGHPUT_MONITORING = Gauge('streaming_total_throughput', 'Total streaming throughput', ['pipeline'])


class ComponentType(Enum):
    """Streaming component types."""
    INGESTION = "ingestion"
    PROCESSOR = "processor"
    ANALYTICS = "analytics"
    STORAGE = "storage"
    ALERTING = "alerting"


class ComponentStatus(Enum):
    """Component status."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    ERROR = "error"


class OrchestrationAction(Enum):
    """Orchestration actions."""
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HEAL = "heal"
    MIGRATE = "migrate"


@dataclass
class ComponentConfig:
    """Streaming component configuration."""
    name: str
    component_type: ComponentType
    image: str
    replicas: int = 1
    resources: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    health_check_path: str = "/health"
    health_check_port: int = 8080
    scaling_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Streaming pipeline configuration."""
    name: str
    description: str
    components: List[str]  # Component names in order
    data_flow: Dict[str, List[str]]  # source -> [destinations]
    sla_config: Dict[str, Any] = field(default_factory=dict)
    alerting_config: Dict[str, Any] = field(default_factory=dict)
    auto_scaling: bool = True
    error_handling: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentInstance:
    """Running component instance."""
    id: str
    component_name: str
    pod_name: str
    namespace: str
    status: ComponentStatus
    health_score: float = 1.0
    last_health_check: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    restart_count: int = 0


class StreamingOrchestrator:
    """Comprehensive streaming orchestration system."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/3", namespace: str = "streaming"):
        self.redis_url = redis_url
        self.namespace = namespace
        
        # Initialize components
        self.redis_client = None
        self.k8s_apps_v1 = None
        self.k8s_core_v1 = None
        
        # Configuration
        self.component_configs: Dict[str, ComponentConfig] = {}
        self.pipeline_configs: Dict[str, PipelineConfig] = {}
        self.component_instances: Dict[str, ComponentInstance] = {}
        
        # Orchestration state
        self.is_running = False
        self.orchestration_tasks: List[asyncio.Task] = []
        self.health_checkers: Dict[str, asyncio.Task] = {}
        self.scaling_decisions: Dict[str, Dict[str, Any]] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'component_unhealthy': [],
            'component_started': [],
            'component_stopped': [],
            'pipeline_degraded': [],
            'scaling_triggered': []
        }
        
        # Monitoring
        self.logger = logging.getLogger("streaming_orchestrator")
        self.metrics_cache: Dict[str, Any] = {}
        self.alert_conditions: Dict[str, Dict[str, Any]] = {}
        
        # Auto-healing configuration
        self.auto_healing_enabled = True
        self.healing_cooldown_seconds = 300
        self.last_healing_actions: Dict[str, datetime] = {}
    
    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        try:
            self.logger.info("Initializing streaming orchestrator...")
            
            # Initialize Redis
            self.redis_client = redis.Redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize Kubernetes client
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()
            
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            
            # Load existing configurations
            await self._load_configurations()
            
            # Initialize monitoring
            await self._initialize_monitoring()
            
            self.logger.info("Streaming orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def _load_configurations(self) -> None:
        """Load configurations from Redis."""
        try:
            # Load component configurations
            component_keys = await self.redis_client.keys("streaming:component:*")
            for key in component_keys:
                config_data = await self.redis_client.get(key)
                if config_data:
                    config_dict = json.loads(config_data)
                    config = ComponentConfig(**config_dict)
                    self.component_configs[config.name] = config
            
            # Load pipeline configurations
            pipeline_keys = await self.redis_client.keys("streaming:pipeline:*")
            for key in pipeline_keys:
                config_data = await self.redis_client.get(key)
                if config_data:
                    config_dict = json.loads(config_data)
                    config = PipelineConfig(**config_dict)
                    self.pipeline_configs[config.name] = config
            
            self.logger.info(f"Loaded {len(self.component_configs)} components and {len(self.pipeline_configs)} pipelines")
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
    
    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring and alerting."""
        try:
            # Set up default alert conditions
            self.alert_conditions = {
                'component_health_low': {
                    'threshold': 0.7,
                    'evaluation_window': 300,
                    'severity': 'warning'
                },
                'component_error_rate_high': {
                    'threshold': 0.1,
                    'evaluation_window': 300,
                    'severity': 'critical'
                },
                'pipeline_throughput_low': {
                    'threshold': 0.5,
                    'evaluation_window': 600,
                    'severity': 'warning'
                }
            }
            
            self.logger.info("Monitoring initialized with default alert conditions")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
    
    def register_component(self, config: ComponentConfig) -> None:
        """Register a streaming component."""
        self.component_configs[config.name] = config
        
        # Store in Redis
        asyncio.create_task(self._store_component_config(config))
        
        self.logger.info(f"Registered component: {config.name} ({config.component_type.value})")
    
    def register_pipeline(self, config: PipelineConfig) -> None:
        """Register a streaming pipeline."""
        self.pipeline_configs[config.name] = config
        
        # Store in Redis
        asyncio.create_task(self._store_pipeline_config(config))
        
        self.logger.info(f"Registered pipeline: {config.name}")
    
    async def _store_component_config(self, config: ComponentConfig) -> None:
        """Store component configuration in Redis."""
        try:
            key = f"streaming:component:{config.name}"
            value = json.dumps(asdict(config), default=str)
            await self.redis_client.set(key, value)
        except Exception as e:
            self.logger.error(f"Failed to store component config: {e}")
    
    async def _store_pipeline_config(self, config: PipelineConfig) -> None:
        """Store pipeline configuration in Redis."""
        try:
            key = f"streaming:pipeline:{config.name}"
            value = json.dumps(asdict(config), default=str)
            await self.redis_client.set(key, value)
        except Exception as e:
            self.logger.error(f"Failed to store pipeline config: {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register an event handler."""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            self.logger.info(f"Registered event handler for: {event_type}")
    
    async def start(self) -> None:
        """Start the orchestrator."""
        if self.is_running:
            self.logger.warning("Orchestrator is already running")
            return
        
        self.is_running = True
        self.logger.info("Starting streaming orchestrator...")
        
        # Start orchestration tasks
        self.orchestration_tasks = [
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._scaling_decision_loop()),
            asyncio.create_task(self._auto_healing_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._alert_evaluation_loop()),
            asyncio.create_task(self._resource_optimization_loop())
        ]
        
        # Discover existing components
        await self._discover_existing_components()
        
        self.logger.info(f"Started {len(self.orchestration_tasks)} orchestration tasks")
    
    async def _discover_existing_components(self) -> None:
        """Discover existing streaming components in Kubernetes."""
        try:
            # List deployments in the streaming namespace
            deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace=self.namespace)
            
            for deployment in deployments.items:
                # Check if this is a streaming component
                labels = deployment.metadata.labels or {}
                if labels.get('app.kubernetes.io/component') == 'streaming':
                    component_name = labels.get('app.kubernetes.io/name', deployment.metadata.name)
                    
                    # Get pods for this deployment
                    pods = self.k8s_core_v1.list_namespaced_pod(
                        namespace=self.namespace,
                        label_selector=f"app.kubernetes.io/name={component_name}"
                    )
                    
                    # Create component instances
                    for pod in pods.items:
                        instance = ComponentInstance(
                            id=str(uuid.uuid4()),
                            component_name=component_name,
                            pod_name=pod.metadata.name,
                            namespace=pod.metadata.namespace,
                            status=self._get_pod_status(pod),
                            last_health_check=datetime.utcnow()
                        )
                        
                        self.component_instances[instance.id] = instance
                        
                        # Start health checking for this instance
                        self.health_checkers[instance.id] = asyncio.create_task(
                            self._health_check_loop(instance.id)
                        )
            
            self.logger.info(f"Discovered {len(self.component_instances)} existing component instances")
            
        except Exception as e:
            self.logger.error(f"Failed to discover existing components: {e}")
    
    def _get_pod_status(self, pod) -> ComponentStatus:
        """Get component status from pod."""
        phase = pod.status.phase
        
        if phase == 'Running':
            # Check container statuses
            container_statuses = pod.status.container_statuses or []
            if all(status.ready for status in container_statuses):
                return ComponentStatus.HEALTHY
            else:
                return ComponentStatus.DEGRADED
        elif phase == 'Pending':
            return ComponentStatus.STARTING
        else:
            return ComponentStatus.UNHEALTHY
    
    async def deploy_component(self, component_name: str, replicas: Optional[int] = None) -> str:
        """Deploy a streaming component."""
        try:
            if component_name not in self.component_configs:
                raise ValueError(f"Component {component_name} not registered")
            
            config = self.component_configs[component_name]
            deployment_replicas = replicas or config.replicas
            
            # Create Kubernetes deployment
            deployment_id = await self._create_k8s_deployment(config, deployment_replicas)
            
            # Wait for deployment to be ready
            await self._wait_for_deployment(component_name, timeout_seconds=300)
            
            # Start health checking
            await self._start_health_checking(component_name)
            
            ORCHESTRATOR_OPERATIONS.labels(operation="deploy", status="success").inc()
            self.logger.info(f"Successfully deployed component: {component_name}")
            
            # Trigger event handlers
            await self._trigger_event_handlers('component_started', {
                'component_name': component_name,
                'deployment_id': deployment_id,
                'replicas': deployment_replicas
            })
            
            return deployment_id
            
        except Exception as e:
            ORCHESTRATOR_OPERATIONS.labels(operation="deploy", status="error").inc()
            self.logger.error(f"Failed to deploy component {component_name}: {e}")
            raise
    
    async def _create_k8s_deployment(self, config: ComponentConfig, replicas: int) -> str:
        """Create Kubernetes deployment for component."""
        try:
            # Create deployment manifest
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(
                    name=config.name,
                    namespace=self.namespace,
                    labels={
                        'app.kubernetes.io/name': config.name,
                        'app.kubernetes.io/component': 'streaming',
                        'app.kubernetes.io/part-of': 'streaming-platform'
                    }
                ),
                spec=client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=client.V1LabelSelector(
                        match_labels={'app.kubernetes.io/name': config.name}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={'app.kubernetes.io/name': config.name}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=config.name,
                                    image=config.image,
                                    env=[
                                        client.V1EnvVar(name=k, value=v)
                                        for k, v in config.environment.items()
                                    ],
                                    resources=client.V1ResourceRequirements(
                                        requests=config.resources,
                                        limits=config.resources
                                    ),
                                    liveness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path=config.health_check_path,
                                            port=config.health_check_port
                                        ),
                                        initial_delay_seconds=30,
                                        period_seconds=10
                                    ),
                                    readiness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path=config.health_check_path,
                                            port=config.health_check_port
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
            
            # Create deployment
            result = self.k8s_apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            
            return result.metadata.name
            
        except Exception as e:
            self.logger.error(f"Failed to create Kubernetes deployment: {e}")
            raise
    
    async def _wait_for_deployment(self, component_name: str, timeout_seconds: int = 300) -> bool:
        """Wait for deployment to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=component_name,
                    namespace=self.namespace
                )
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas >= deployment.spec.replicas):
                    return True
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error checking deployment status: {e}")
                await asyncio.sleep(10)
        
        self.logger.error(f"Deployment {component_name} did not become ready within {timeout_seconds} seconds")
        return False
    
    async def _start_health_checking(self, component_name: str) -> None:
        """Start health checking for component instances."""
        try:
            # Get pods for this component
            pods = self.k8s_core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app.kubernetes.io/name={component_name}"
            )
            
            for pod in pods.items:
                instance = ComponentInstance(
                    id=str(uuid.uuid4()),
                    component_name=component_name,
                    pod_name=pod.metadata.name,
                    namespace=pod.metadata.namespace,
                    status=self._get_pod_status(pod),
                    last_health_check=datetime.utcnow()
                )
                
                self.component_instances[instance.id] = instance
                
                # Start health check loop
                self.health_checkers[instance.id] = asyncio.create_task(
                    self._health_check_loop(instance.id)
                )
            
        except Exception as e:
            self.logger.error(f"Failed to start health checking: {e}")
    
    async def deploy_pipeline(self, pipeline_name: str) -> Dict[str, str]:
        """Deploy a complete streaming pipeline."""
        try:
            if pipeline_name not in self.pipeline_configs:
                raise ValueError(f"Pipeline {pipeline_name} not registered")
            
            pipeline_config = self.pipeline_configs[pipeline_name]
            deployment_results = {}
            
            # Deploy components in dependency order
            deployed_components = set()
            
            while len(deployed_components) < len(pipeline_config.components):
                for component_name in pipeline_config.components:
                    if component_name in deployed_components:
                        continue
                    
                    # Check if dependencies are met
                    if component_name in self.component_configs:
                        component_config = self.component_configs[component_name]
                        dependencies_met = all(
                            dep in deployed_components 
                            for dep in component_config.dependencies
                        )
                        
                        if dependencies_met:
                            deployment_id = await self.deploy_component(component_name)
                            deployment_results[component_name] = deployment_id
                            deployed_components.add(component_name)
                            
                            # Wait a bit before deploying next component
                            await asyncio.sleep(10)
            
            # Start pipeline monitoring
            await self._start_pipeline_monitoring(pipeline_name)
            
            ORCHESTRATOR_OPERATIONS.labels(operation="deploy_pipeline", status="success").inc()
            self.logger.info(f"Successfully deployed pipeline: {pipeline_name}")
            
            return deployment_results
            
        except Exception as e:
            ORCHESTRATOR_OPERATIONS.labels(operation="deploy_pipeline", status="error").inc()
            self.logger.error(f"Failed to deploy pipeline {pipeline_name}: {e}")
            raise
    
    async def _start_pipeline_monitoring(self, pipeline_name: str) -> None:
        """Start monitoring for a pipeline."""
        try:
            # Create pipeline monitoring task
            task = asyncio.create_task(self._pipeline_monitoring_loop(pipeline_name))
            self.orchestration_tasks.append(task)
            
            self.logger.info(f"Started monitoring for pipeline: {pipeline_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline monitoring: {e}")
    
    async def _health_monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        self.logger.info("Started health monitoring loop")
        
        while self.is_running:
            try:
                await self._check_component_health()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_component_health(self) -> None:
        """Check health of all components."""
        for instance_id, instance in self.component_instances.items():
            try:
                health_score = await self._get_component_health_score(instance)
                instance.health_score = health_score
                instance.last_health_check = datetime.utcnow()
                
                # Update status based on health score
                if health_score >= 0.8:
                    instance.status = ComponentStatus.HEALTHY
                elif health_score >= 0.5:
                    instance.status = ComponentStatus.DEGRADED
                else:
                    instance.status = ComponentStatus.UNHEALTHY
                
                # Update metrics
                COMPONENT_HEALTH.labels(
                    component=instance.component_name,
                    instance=instance.pod_name
                ).set(health_score)
                
                # Check for alerts
                if health_score < 0.7:
                    await self._trigger_event_handlers('component_unhealthy', {
                        'instance_id': instance_id,
                        'component_name': instance.component_name,
                        'health_score': health_score,
                        'pod_name': instance.pod_name
                    })
                
            except Exception as e:
                self.logger.error(f"Error checking health for instance {instance_id}: {e}")
                instance.error_count += 1
    
    async def _get_component_health_score(self, instance: ComponentInstance) -> float:
        """Get health score for a component instance."""
        try:
            # Get pod metrics from Kubernetes
            pod = self.k8s_core_v1.read_namespaced_pod(
                name=instance.pod_name,
                namespace=instance.namespace
            )
            
            health_score = 1.0
            
            # Check pod phase
            if pod.status.phase != 'Running':
                health_score *= 0.3
            
            # Check container statuses
            container_statuses = pod.status.container_statuses or []
            ready_containers = sum(1 for status in container_statuses if status.ready)
            total_containers = len(container_statuses)
            
            if total_containers > 0:
                container_health = ready_containers / total_containers
                health_score *= container_health
            
            # Check restart count
            restart_count = sum(status.restart_count for status in container_statuses)
            if restart_count > instance.restart_count:
                instance.restart_count = restart_count
                health_score *= max(0.1, 1.0 - (restart_count * 0.1))
            
            # Try to get application-specific health metrics
            try:
                health_score *= await self._get_application_health_score(instance)
            except Exception:
                pass  # Application health not available
            
            return max(0.0, min(1.0, health_score))
            
        except Exception as e:
            self.logger.error(f"Error getting health score: {e}")
            return 0.0
    
    async def _get_application_health_score(self, instance: ComponentInstance) -> float:
        """Get application-specific health score."""
        try:
            # Get component configuration
            if instance.component_name not in self.component_configs:
                return 1.0
            
            config = self.component_configs[instance.component_name]
            
            # Try to get health from HTTP endpoint
            pod_ip = await self._get_pod_ip(instance)
            if not pod_ip:
                return 1.0
            
            health_url = f"http://{pod_ip}:{config.health_check_port}{config.health_check_path}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Parse health response
                        if isinstance(health_data, dict):
                            return health_data.get('health_score', 1.0)
                        
                        return 1.0
                    else:
                        return 0.5
            
        except Exception:
            return 1.0  # Default to healthy if we can't check
    
    async def _get_pod_ip(self, instance: ComponentInstance) -> Optional[str]:
        """Get pod IP address."""
        try:
            pod = self.k8s_core_v1.read_namespaced_pod(
                name=instance.pod_name,
                namespace=instance.namespace
            )
            return pod.status.pod_ip
        except Exception:
            return None
    
    async def _health_check_loop(self, instance_id: str) -> None:
        """Health check loop for specific instance."""
        while self.is_running and instance_id in self.component_instances:
            try:
                instance = self.component_instances[instance_id]
                health_score = await self._get_component_health_score(instance)
                
                # Update instance
                instance.health_score = health_score
                instance.last_health_check = datetime.utcnow()
                
                # Update metrics cache
                await self._update_instance_metrics(instance)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in health check loop for {instance_id}: {e}")
                await asyncio.sleep(60)
    
    async def _update_instance_metrics(self, instance: ComponentInstance) -> None:
        """Update metrics for instance."""
        try:
            metrics_data = {
                'instance_id': instance.id,
                'component_name': instance.component_name,
                'pod_name': instance.pod_name,
                'status': instance.status.value,
                'health_score': instance.health_score,
                'error_count': instance.error_count,
                'restart_count': instance.restart_count,
                'last_health_check': instance.last_health_check.isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Store in Redis
            key = f"streaming:metrics:instance:{instance.id}"
            await self.redis_client.setex(key, 300, json.dumps(metrics_data))
            
        except Exception as e:
            self.logger.error(f"Failed to update instance metrics: {e}")
    
    async def _scaling_decision_loop(self) -> None:
        """Auto-scaling decision loop."""
        self.logger.info("Started scaling decision loop")
        
        while self.is_running:
            try:
                await self._evaluate_scaling_decisions()
                await asyncio.sleep(60)  # Evaluate every minute
                
            except Exception as e:
                self.logger.error(f"Error in scaling decision loop: {e}")
                await asyncio.sleep(120)
    
    async def _evaluate_scaling_decisions(self) -> None:
        """Evaluate scaling decisions for components."""
        for component_name, config in self.component_configs.items():
            if not config.scaling_config.get('enabled', False):
                continue
            
            try:
                scaling_decision = await self._make_scaling_decision(component_name, config)
                
                if scaling_decision['action'] != 'none':
                    await self._execute_scaling_action(component_name, scaling_decision)
                
            except Exception as e:
                self.logger.error(f"Error evaluating scaling for {component_name}: {e}")
    
    async def _make_scaling_decision(self, component_name: str, config: ComponentConfig) -> Dict[str, Any]:
        """Make scaling decision for component."""
        try:
            # Get current instances
            instances = [i for i in self.component_instances.values() if i.component_name == component_name]
            current_replicas = len(instances)
            
            # Get metrics
            avg_health = sum(i.health_score for i in instances) / len(instances) if instances else 0
            
            scaling_config = config.scaling_config
            min_replicas = scaling_config.get('min_replicas', 1)
            max_replicas = scaling_config.get('max_replicas', 10)
            target_health = scaling_config.get('target_health', 0.8)
            
            # Decision logic
            if avg_health < target_health and current_replicas < max_replicas:
                return {
                    'action': 'scale_up',
                    'current_replicas': current_replicas,
                    'target_replicas': min(current_replicas + 1, max_replicas),
                    'reason': f'Low health score: {avg_health:.2f}'
                }
            
            elif avg_health > 0.95 and current_replicas > min_replicas:
                return {
                    'action': 'scale_down',
                    'current_replicas': current_replicas,
                    'target_replicas': max(current_replicas - 1, min_replicas),
                    'reason': f'High health score: {avg_health:.2f}'
                }
            
            return {'action': 'none', 'current_replicas': current_replicas}
            
        except Exception as e:
            self.logger.error(f"Error making scaling decision: {e}")
            return {'action': 'none'}
    
    async def _execute_scaling_action(self, component_name: str, decision: Dict[str, Any]) -> None:
        """Execute scaling action."""
        try:
            target_replicas = decision['target_replicas']
            
            # Update deployment
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=component_name,
                namespace=self.namespace
            )
            
            deployment.spec.replicas = target_replicas
            
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=component_name,
                namespace=self.namespace,
                body=deployment
            )
            
            # Update metrics
            ORCHESTRATOR_OPERATIONS.labels(operation="scale", status="success").inc()
            
            # Trigger event handlers
            await self._trigger_event_handlers('scaling_triggered', {
                'component_name': component_name,
                'action': decision['action'],
                'from_replicas': decision['current_replicas'],
                'to_replicas': target_replicas,
                'reason': decision['reason']
            })
            
            self.logger.info(f"Scaled {component_name} from {decision['current_replicas']} to {target_replicas} replicas")
            
        except Exception as e:
            ORCHESTRATOR_OPERATIONS.labels(operation="scale", status="error").inc()
            self.logger.error(f"Failed to execute scaling action: {e}")
    
    async def _auto_healing_loop(self) -> None:
        """Auto-healing loop."""
        self.logger.info("Started auto-healing loop")
        
        while self.is_running:
            try:
                if self.auto_healing_enabled:
                    await self._perform_auto_healing()
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                self.logger.error(f"Error in auto-healing loop: {e}")
                await asyncio.sleep(300)
    
    async def _perform_auto_healing(self) -> None:
        """Perform auto-healing actions."""
        for instance_id, instance in list(self.component_instances.items()):
            try:
                # Check if healing is needed
                if instance.status == ComponentStatus.UNHEALTHY:
                    # Check cooldown
                    last_healing = self.last_healing_actions.get(instance.component_name)
                    if last_healing and (datetime.utcnow() - last_healing).total_seconds() < self.healing_cooldown_seconds:
                        continue
                    
                    # Attempt healing
                    await self._heal_component_instance(instance)
                    self.last_healing_actions[instance.component_name] = datetime.utcnow()
                
            except Exception as e:
                self.logger.error(f"Error healing instance {instance_id}: {e}")
    
    async def _heal_component_instance(self, instance: ComponentInstance) -> None:
        """Heal a component instance."""
        try:
            self.logger.info(f"Attempting to heal instance: {instance.pod_name}")
            
            # Delete the pod to trigger restart
            self.k8s_core_v1.delete_namespaced_pod(
                name=instance.pod_name,
                namespace=instance.namespace
            )
            
            # Remove from tracking (will be rediscovered)
            if instance.id in self.component_instances:
                del self.component_instances[instance.id]
            
            # Cancel health checker
            if instance.id in self.health_checkers:
                self.health_checkers[instance.id].cancel()
                del self.health_checkers[instance.id]
            
            ORCHESTRATOR_OPERATIONS.labels(operation="heal", status="success").inc()
            self.logger.info(f"Initiated healing for pod: {instance.pod_name}")
            
        except Exception as e:
            ORCHESTRATOR_OPERATIONS.labels(operation="heal", status="error").inc()
            self.logger.error(f"Failed to heal instance: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Collect and aggregate metrics."""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(120)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-wide metrics."""
        try:
            # Update component counts
            component_counts = {}
            for instance in self.component_instances.values():
                component_type = self.component_configs.get(instance.component_name, {}).component_type
                if component_type:
                    component_counts[component_type.value] = component_counts.get(component_type.value, 0) + 1
            
            for component_type, count in component_counts.items():
                ACTIVE_COMPONENTS.labels(component_type=component_type).set(count)
            
            # Store aggregated metrics
            metrics_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'total_components': len(self.component_instances),
                'component_counts': component_counts,
                'healthy_components': len([i for i in self.component_instances.values() if i.status == ComponentStatus.HEALTHY]),
                'unhealthy_components': len([i for i in self.component_instances.values() if i.status == ComponentStatus.UNHEALTHY]),
                'average_health_score': sum(i.health_score for i in self.component_instances.values()) / len(self.component_instances) if self.component_instances else 0
            }
            
            await self.redis_client.setex(
                "streaming:system_metrics",
                300,
                json.dumps(metrics_data)
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _alert_evaluation_loop(self) -> None:
        """Evaluate alert conditions."""
        while self.is_running:
            try:
                await self._evaluate_alerts()
                await asyncio.sleep(60)  # Evaluate every minute
                
            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(120)
    
    async def _evaluate_alerts(self) -> None:
        """Evaluate all alert conditions."""
        for alert_name, condition in self.alert_conditions.items():
            try:
                if await self._evaluate_alert_condition(alert_name, condition):
                    await self._trigger_alert(alert_name, condition)
                
            except Exception as e:
                self.logger.error(f"Error evaluating alert {alert_name}: {e}")
    
    async def _evaluate_alert_condition(self, alert_name: str, condition: Dict[str, Any]) -> bool:
        """Evaluate a single alert condition."""
        try:
            threshold = condition['threshold']
            
            if alert_name == 'component_health_low':
                unhealthy_count = len([i for i in self.component_instances.values() if i.health_score < threshold])
                return unhealthy_count > 0
            
            elif alert_name == 'component_error_rate_high':
                total_instances = len(self.component_instances)
                if total_instances == 0:
                    return False
                
                error_instances = len([i for i in self.component_instances.values() if i.error_count > 0])
                error_rate = error_instances / total_instances
                return error_rate > threshold
            
            # Add more alert conditions as needed
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating alert condition: {e}")
            return False
    
    async def _trigger_alert(self, alert_name: str, condition: Dict[str, Any]) -> None:
        """Trigger an alert."""
        try:
            alert_data = {
                'alert_name': alert_name,
                'severity': condition['severity'],
                'threshold': condition['threshold'],
                'triggered_at': datetime.utcnow().isoformat(),
                'system_state': {
                    'total_components': len(self.component_instances),
                    'healthy_components': len([i for i in self.component_instances.values() if i.status == ComponentStatus.HEALTHY]),
                    'unhealthy_components': len([i for i in self.component_instances.values() if i.status == ComponentStatus.UNHEALTHY])
                }
            }
            
            # Store alert
            key = f"streaming:alert:{alert_name}:{datetime.utcnow().isoformat()}"
            await self.redis_client.setex(key, 3600, json.dumps(alert_data))
            
            self.logger.warning(f"Alert triggered: {alert_name} ({condition['severity']})")
            
        except Exception as e:
            self.logger.error(f"Failed to trigger alert: {e}")
    
    async def _resource_optimization_loop(self) -> None:
        """Resource optimization loop."""
        while self.is_running:
            try:
                await self._optimize_resources()
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in resource optimization loop: {e}")
                await asyncio.sleep(600)
    
    async def _optimize_resources(self) -> None:
        """Optimize resource allocation."""
        try:
            # Placeholder for resource optimization logic
            # This could include:
            # - Adjusting resource requests/limits based on usage
            # - Optimizing pod placement
            # - Load balancing across nodes
            
            self.logger.debug("Resource optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error optimizing resources: {e}")
    
    async def _pipeline_monitoring_loop(self, pipeline_name: str) -> None:
        """Monitor specific pipeline."""
        while self.is_running:
            try:
                pipeline_health = await self._get_pipeline_health(pipeline_name)
                
                # Store pipeline metrics
                metrics_data = {
                    'pipeline_name': pipeline_name,
                    'health_score': pipeline_health,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                key = f"streaming:pipeline_metrics:{pipeline_name}"
                await self.redis_client.setex(key, 300, json.dumps(metrics_data))
                
                # Check for pipeline degradation
                if pipeline_health < 0.7:
                    await self._trigger_event_handlers('pipeline_degraded', {
                        'pipeline_name': pipeline_name,
                        'health_score': pipeline_health
                    })
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error monitoring pipeline {pipeline_name}: {e}")
                await asyncio.sleep(120)
    
    async def _get_pipeline_health(self, pipeline_name: str) -> float:
        """Get health score for a pipeline."""
        try:
            if pipeline_name not in self.pipeline_configs:
                return 0.0
            
            pipeline_config = self.pipeline_configs[pipeline_name]
            component_healths = []
            
            for component_name in pipeline_config.components:
                instances = [i for i in self.component_instances.values() if i.component_name == component_name]
                if instances:
                    avg_health = sum(i.health_score for i in instances) / len(instances)
                    component_healths.append(avg_health)
            
            return sum(component_healths) / len(component_healths) if component_healths else 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline health: {e}")
            return 0.0
    
    async def _trigger_event_handlers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trigger registered event handlers."""
        try:
            handlers = self.event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    await asyncio.get_event_loop().run_in_executor(None, handler, event_data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")
        except Exception as e:
            self.logger.error(f"Error triggering event handlers: {e}")
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        self.logger.info("Stopping streaming orchestrator...")
        self.is_running = False
        
        # Cancel all tasks
        all_tasks = self.orchestration_tasks + list(self.health_checkers.values())
        for task in all_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Close connections
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("Streaming orchestrator stopped")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            'is_running': self.is_running,
            'component_configs': len(self.component_configs),
            'pipeline_configs': len(self.pipeline_configs),
            'active_instances': len(self.component_instances),
            'healthy_instances': len([i for i in self.component_instances.values() if i.status == ComponentStatus.HEALTHY]),
            'unhealthy_instances': len([i for i in self.component_instances.values() if i.status == ComponentStatus.UNHEALTHY]),
            'auto_healing_enabled': self.auto_healing_enabled,
            'orchestration_tasks': len(self.orchestration_tasks),
            'health_checkers': len(self.health_checkers)
        }


# Example usage
async def create_streaming_orchestrator():
    """Create and configure streaming orchestrator."""
    orchestrator = StreamingOrchestrator()
    await orchestrator.initialize()
    
    # Register example components
    ingestion_config = ComponentConfig(
        name="high-throughput-ingestion",
        component_type=ComponentType.INGESTION,
        image="streaming/ingestion:latest",
        replicas=3,
        resources={"cpu": "500m", "memory": "1Gi"},
        scaling_config={
            "enabled": True,
            "min_replicas": 1,
            "max_replicas": 10,
            "target_health": 0.8
        }
    )
    orchestrator.register_component(ingestion_config)
    
    # Register example pipeline
    pipeline_config = PipelineConfig(
        name="real-time-analytics",
        description="Real-time data analytics pipeline",
        components=["high-throughput-ingestion", "stream-processor", "analytics-engine"],
        data_flow={"ingestion": ["processor"], "processor": ["analytics"]},
        auto_scaling=True
    )
    orchestrator.register_pipeline(pipeline_config)
    
    return orchestrator