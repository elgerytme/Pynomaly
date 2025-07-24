"""
Enterprise Scalability Package

Comprehensive distributed computing and streaming processing
for enterprise-scale anomaly detection and data processing.
"""

# Domain entities
from .domain.entities.compute_cluster import (
    ComputeCluster, ComputeNode,
    ClusterType, ClusterStatus, NodeStatus, ScalingPolicy
)

from .domain.entities.stream_processor import (
    StreamProcessor, StreamSource, StreamSink, ProcessingWindow,
    StreamType, ProcessorStatus, ProcessingMode, WindowType
)

from .domain.entities.distributed_task import (
    DistributedTask, TaskBatch, ResourceRequirements, TaskResult,
    TaskStatus, TaskPriority, TaskType
)

# Application services
from .application.services.scalability_service import ScalabilityService

# DTOs
from .application.dto.scalability_dto import (
    # Cluster DTOs
    ClusterCreateRequest, ClusterScaleRequest, ClusterResponse,
    NodeResponse, ClusterMetricsResponse,
    
    # Stream Processing DTOs
    StreamSourceRequest, StreamSinkRequest,
    StreamProcessorRequest, StreamProcessorResponse,
    
    # Task DTOs
    TaskSubmitRequest, TaskBatchSubmitRequest,
    TaskResponse, TaskBatchResponse, TaskResultResponse,
    
    # General DTOs
    ScalabilityOverviewResponse, ResourceUtilizationResponse,
    PerformanceMetricsResponse, ScalingRecommendationResponse,
    HealthCheckResponse
)

__version__ = "0.1.0"
__author__ = "anomaly_detection Enterprise Team"
__email__ = "enterprise@anomaly_detection.org"

__all__ = [
    # Domain entities
    "ComputeCluster", "ComputeNode",
    "ClusterType", "ClusterStatus", "NodeStatus", "ScalingPolicy",
    
    "StreamProcessor", "StreamSource", "StreamSink", "ProcessingWindow",
    "StreamType", "ProcessorStatus", "ProcessingMode", "WindowType",
    
    "DistributedTask", "TaskBatch", "ResourceRequirements", "TaskResult",
    "TaskStatus", "TaskPriority", "TaskType",
    
    # Application services
    "ScalabilityService",
    
    # DTOs
    "ClusterCreateRequest", "ClusterScaleRequest", "ClusterResponse",
    "NodeResponse", "ClusterMetricsResponse",
    
    "StreamSourceRequest", "StreamSinkRequest",
    "StreamProcessorRequest", "StreamProcessorResponse",
    
    "TaskSubmitRequest", "TaskBatchSubmitRequest",
    "TaskResponse", "TaskBatchResponse", "TaskResultResponse",
    
    "ScalabilityOverviewResponse", "ResourceUtilizationResponse",
    "PerformanceMetricsResponse", "ScalingRecommendationResponse",
    "HealthCheckResponse",
]

# Package metadata
PACKAGE_INFO = {
    "name": "anomaly_detection-enterprise-scalability",
    "version": __version__,
    "description": "Enterprise scalability with distributed computing and streaming for anomaly_detection",
    "author": __author__,
    "email": __email__,
    "features": [
        "Distributed compute cluster management (Dask, Ray, Kubernetes)",
        "Auto-scaling based on workload and resource utilization",
        "Real-time stream processing with Kafka, Kinesis, PubSub",
        "Distributed task scheduling and execution",
        "High-performance anomaly detection at scale",
        "GPU acceleration support for ML workloads",
        "Enterprise monitoring and observability",
        "Cloud-native deployment and orchestration"
    ],
    "supported_frameworks": [
        "Dask - Distributed computing for Python",
        "Ray - Distributed AI/ML framework",
        "Apache Kafka - Stream processing monorepo",
        "Apache Beam - Unified batch/stream processing",
        "Kubernetes - Container orchestration",
        "Apache Spark - Big data processing (integration)",
        "CUDA/CuPy - GPU acceleration",
        "Rapids cuDF - GPU DataFrames"
    ],
    "deployment_targets": [
        "Kubernetes clusters",
        "AWS EKS, ECS, Lambda",
        "Azure AKS, Container Instances",
        "Google GKE, Cloud Run",
        "On-premises infrastructure",
        "Hybrid cloud environments"
    ],
    "integrations": [
        "Enterprise monitoring systems (Prometheus, Grafana, Datadog)",
        "Message queues and streams (Kafka, Kinesis, PubSub, RabbitMQ)",
        "Object storage (S3, GCS, Azure Blob)",
        "Data warehouses (Snowflake, BigQuery, Redshift)",
        "ML monorepos (MLflow, Kubeflow, SageMaker)",
        "CI/CD pipelines (Jenkins, GitLab, GitHub Actions)"
    ]
}

# Configuration defaults
DEFAULT_CONFIG = {
    "compute": {
        "default_cluster_type": "dask",
        "default_node_type": "standard",
        "auto_scaling_enabled": True,
        "max_cluster_nodes": 100,
        "node_heartbeat_interval": 30,
        "resource_check_interval": 60
    },
    "streaming": {
        "default_parallelism": 1,
        "max_parallelism": 50,
        "checkpoint_interval_ms": 60000,
        "auto_scaling_enabled": True,
        "metrics_collection_interval": 10
    },
    "tasks": {
        "default_priority": "normal",
        "default_timeout_seconds": 3600,
        "max_retries": 3,
        "retry_backoff_seconds": 60,
        "task_cleanup_days": 30
    },
    "monitoring": {
        "metrics_retention_days": 90,
        "health_check_interval": 30,
        "performance_sampling_rate": 1.0,
        "alert_thresholds": {
            "cpu_utilization": 85.0,
            "memory_utilization": 90.0,
            "error_rate": 5.0
        }
    }
}