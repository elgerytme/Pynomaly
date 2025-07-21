# Enterprise Cloud-Native Package

The Enterprise Cloud-Native package provides comprehensive Kubernetes operators, service mesh integration, and advanced auto-scaling capabilities for enterprise-grade cloud-native applications.

## Features

### 🚀 Kubernetes Operators
- **Custom Resource Management**: Define and manage custom Kubernetes resources with automated lifecycle management
- **Operator Framework Integration**: Built-in integration with Kopf and other operator frameworks
- **Reconciliation Loops**: Automated resource reconciliation with error handling and retry logic
- **Multi-tenant Support**: Tenant-aware resource management and isolation

### 🕸️ Service Mesh Integration
- **Istio Support**: Full integration with Istio service mesh for traffic management and security
- **Linkerd Support**: Lightweight service mesh integration for microservices communication
- **Envoy Proxy**: Standalone Envoy proxy configuration for advanced load balancing
- **Traffic Policies**: Circuit breakers, retries, timeouts, and canary deployments
- **Security Policies**: mTLS, authorization, authentication, and RBAC
- **Observability**: Distributed tracing, metrics collection, and access logging

### ⚡ Advanced Auto-scaling
- **Horizontal Pod Autoscaling (HPA)**: Native Kubernetes HPA with custom metrics support
- **Vertical Pod Autoscaling (VPA)**: Automatic resource request and limit optimization
- **Cluster Autoscaling**: Node-level scaling based on resource requirements
- **Predictive Scaling**: ML-based predictive scaling using historical patterns
- **Multi-dimensional Scaling**: Combined horizontal, vertical, and cluster scaling
- **Custom Metrics**: Integration with Prometheus, custom metrics APIs, and external data sources

### 🤖 Machine Learning Integration
- **Predictive Models**: Linear regression, Random Forest, and ensemble models
- **Feature Engineering**: Automated feature extraction from metrics and workload patterns
- **Model Training**: Continuous learning from historical data and feedback loops
- **Anomaly Detection**: Integration with anomaly detection for scaling decisions

## Installation

```bash
# Install with basic dependencies
pip install pynomaly-enterprise-cloud-native

# Install with all features
pip install pynomaly-enterprise-cloud-native[all]

# Install specific integrations
pip install pynomaly-enterprise-cloud-native[operators,service-mesh,autoscaling]
```

## Quick Start

### Basic Cloud-Native Service Setup

```python
from enterprise_cloud_native import CloudNativeService
from enterprise_cloud_native.domain.entities.kubernetes_resource import ResourceType
from enterprise_cloud_native.domain.entities.service_mesh import ServiceMeshType

# Initialize the service
cloud_native_service = CloudNativeService(
    kubernetes_repository=k8s_repo,
    service_mesh_repository=mesh_repo,
    autoscaling_repository=autoscaling_repo,
    kubernetes_client=k8s_client,
    service_mesh_client=mesh_client,
    metrics_client=metrics_client,
    operator_framework=operator_framework
)

# Create a Kubernetes deployment
deployment = await cloud_native_service.create_kubernetes_resource(
    tenant_id=tenant_id,
    resource_type=ResourceType.DEPLOYMENT,
    name="my-app",
    namespace="production",
    spec={
        "replicas": 3,
        "selector": {"matchLabels": {"app": "my-app"}},
        "template": {
            "metadata": {"labels": {"app": "my-app"}},
            "spec": {
                "containers": [{
                    "name": "app",
                    "image": "my-app:v1.0.0",
                    "ports": [{"containerPort": 8080}]
                }]
            }
        }
    }
)
```

### Service Mesh Configuration

```python
# Install Istio service mesh
mesh_config = await cloud_native_service.install_service_mesh(
    tenant_id=tenant_id,
    mesh_type=ServiceMeshType.ISTIO,
    name="production-mesh",
    configuration={
        "telemetry": {"enabled": True},
        "security": {"mtls_mode": "STRICT"}
    }
)

# Create traffic policy for canary deployment
traffic_policy = await cloud_native_service.create_traffic_policy(
    service_mesh_id=mesh_config.id,
    name="my-app-canary",
    policy_type=TrafficPolicyType.CANARY,
    target_services=["my-app"],
    policy_config={
        "canary_weight": 10,
        "stable_weight": 90
    }
)
```

### Auto-scaling Setup

```python
# Create Horizontal Pod Autoscaler
hpa = await cloud_native_service.create_horizontal_pod_autoscaler(
    tenant_id=tenant_id,
    name="my-app-hpa",
    namespace="production",
    target_resource="my-app",
    min_replicas=2,
    max_replicas=10,
    metrics=[
        {
            "type": "Resource",
            "resource": {
                "name": "cpu",
                "target": {"averageUtilization": 70}
            }
        }
    ]
)

# Create predictive scaling policy
predictive_policy = await cloud_native_service.create_predictive_scaling_policy(
    tenant_id=tenant_id,
    name="my-app-predictive",
    target_resource="my-app",
    prediction_model_type="ensemble",
    prediction_horizon_minutes=60
)
```

### Kubernetes Operator Development

```python
from enterprise_cloud_native.infrastructure.operators import KubernetesOperator
from enterprise_cloud_native.domain.entities.kubernetes_resource import OperatorResource

class AnomalyDetectionOperator(KubernetesOperator):
    def __init__(self):
        super().__init__(
            name="anomaly-detection-operator",
            crd_group="pynomaly.io",
            crd_version="v1",
            crd_kind="AnomalyDetector"
        )
    
    async def reconcile(self, resource: OperatorResource, resource_data: Dict[str, Any]) -> None:
        """Reconcile anomaly detection resource."""
        spec = resource_data["spec"]
        
        # Create anomaly detection deployment
        await self._create_anomaly_detector_deployment(resource, spec)
        
        # Configure data sources
        await self._setup_data_sources(resource, spec["dataSource"])
        
        # Apply scaling configuration
        if spec.get("scaling", {}).get("enabled"):
            await self._configure_auto_scaling(resource, spec["scaling"])
    
    async def create_resource(self, resource: OperatorResource, spec: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Handle resource creation."""
        # Implementation for creating anomaly detection resources
        return {"status": {"phase": "Created"}}
```

## Configuration

### Environment Variables

```bash
# Kubernetes configuration
KUBECONFIG=/path/to/kubeconfig
KUBERNETES_NAMESPACE=default

# Service mesh
ISTIO_NAMESPACE=istio-system
LINKERD_NAMESPACE=linkerd

# Metrics collection
PROMETHEUS_URL=http://prometheus:9090
METRICS_RETENTION_DAYS=30

# Auto-scaling
SCALING_CHECK_INTERVAL=60
PREDICTIVE_SCALING_ENABLED=true
MODEL_RETRAIN_INTERVAL=24
```

### YAML Configuration

```yaml
# cloud-native-config.yaml
cloudNative:
  kubernetes:
    namespace: production
    resourceQuotas:
      enabled: true
      cpu: "4000m"
      memory: "8Gi"
  
  serviceMesh:
    enabled: true
    type: istio
    mtls:
      mode: STRICT
    observability:
      tracing: true
      metrics: true
      accessLogs: true
  
  autoScaling:
    hpa:
      enabled: true
      defaultCpuTarget: 70
      defaultMemoryTarget: 80
    
    vpa:
      enabled: true
      updateMode: Auto
    
    predictiveScaling:
      enabled: true
      modelType: ensemble
      trainingInterval: 24h
      predictionHorizon: 1h
```

## Advanced Features

### Custom Resource Definitions

The package includes several built-in CRDs for advanced functionality:

```yaml
# AnomalyDetector CRD
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: anomalydetectors.pynomaly.io
spec:
  group: pynomaly.io
  versions:
  - name: v1
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              algorithm:
                type: string
                enum: ["isolation_forest", "one_class_svm", "ensemble"]
              threshold:
                type: number
                minimum: 0
                maximum: 1
              dataSource:
                type: object
                properties:
                  type:
                    type: string
                    enum: ["kafka", "kinesis", "pubsub"]
                  config:
                    type: object
```

### Monitoring and Observability

```python
# Get comprehensive cloud-native overview
overview = await cloud_native_service.get_tenant_cloud_native_overview(tenant_id)

print(f"Kubernetes Resources: {overview['kubernetes']['total']}")
print(f"Service Mesh Status: {overview['service_mesh']['healthy']}")
print(f"Active HPAs: {overview['autoscaling']['hpas']['active']}")
```

### Multi-Cloud Support

```python
# AWS EKS integration
from enterprise_cloud_native.infrastructure.cloud import AWSAdapter

aws_adapter = AWSAdapter(region="us-west-2")
await aws_adapter.configure_cluster_autoscaler(cluster_name="production")

# Google GKE integration
from enterprise_cloud_native.infrastructure.cloud import GCPAdapter

gcp_adapter = GCPAdapter(project_id="my-project")
await gcp_adapter.enable_workload_identity(namespace="production")
```

## Architecture

The Enterprise Cloud-Native package follows Clean Architecture principles:

```
enterprise_cloud_native/
├── domain/
│   ├── entities/           # Core business entities
│   └── repositories/       # Repository interfaces
├── application/
│   └── services/          # Application services and use cases
├── infrastructure/
│   ├── operators/         # Kubernetes operators framework
│   ├── service_mesh/      # Service mesh adapters
│   ├── autoscaling/       # Auto-scaling implementations
│   └── cloud/            # Cloud provider integrations
└── presentation/
    ├── api/              # REST API endpoints
    └── cli/              # Command-line interface
```

## Performance and Scaling

### Benchmarks

- **Operator Reconciliation**: < 100ms for simple resources, < 1s for complex deployments
- **Service Mesh Policy Application**: < 500ms for traffic policies, < 2s for security policies  
- **Auto-scaling Decision Time**: < 5s for reactive scaling, < 30s for predictive scaling
- **Resource Throughput**: 1000+ resources/minute for bulk operations

### Optimization Features

- **Concurrent Processing**: Parallel resource operations and batch processing
- **Caching**: Intelligent caching of Kubernetes API responses and metrics
- **Connection Pooling**: Efficient connection management for external services
- **Memory Management**: Optimized memory usage for large-scale deployments

## Security

### Security Features

- **RBAC Integration**: Native Kubernetes RBAC support with tenant isolation
- **Secret Management**: Secure handling of credentials and certificates
- **Network Policies**: Automated network policy generation for service mesh
- **Security Scanning**: Container and configuration security validation
- **Audit Logging**: Comprehensive audit trails for all operations

### Best Practices

```python
# Secure operator deployment
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pynomaly-operator
  namespace: pynomaly-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: pynomaly-operator
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
```

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests
pytest -m integration            # Integration tests
pytest -m kubernetes             # Kubernetes integration tests
pytest -m operators              # Operator tests
pytest -m service_mesh           # Service mesh tests
pytest -m autoscaling            # Auto-scaling tests

# Run performance tests
pytest -m performance --benchmark-only
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/pynomaly
cd pynomaly/src/packages/enterprise/enterprise_cloud_native

# Install development dependencies
pip install -e ".[dev,test,all]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## License

This project is licensed under the MIT License - see the [LICENSE](../../../../LICENSE) file for details.

## Support

- 📧 Email: enterprise@pynomaly.org
- 💬 Discord: [Pynomaly Community](https://discord.gg/pynomaly)
- 📖 Documentation: [https://docs.pynomaly.org/enterprise/cloud-native](https://docs.pynomaly.org/enterprise/cloud-native)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/pynomaly/issues)