# 🚀 Real-Time Data Streaming & Processing Guide

## Overview

The Real-Time Data Streaming & Processing system provides enterprise-grade capabilities for high-throughput data ingestion, stream processing, real-time analytics, and orchestration. This comprehensive system handles millions of messages per second with advanced monitoring, auto-scaling, and fault tolerance.

## Architecture Components

### 1. High-Throughput Ingestion Engine (`high_throughput_ingestion.py`)
Scalable data ingestion with multiple sources, validation, and routing:

- **Multi-Source Support**: HTTP API, Kafka, Database, File System, Webhooks
- **Validation Engine**: Schema validation, transformation rules, data quality checks
- **Deduplication**: Redis-based deduplication with configurable TTL
- **Batching & Routing**: Intelligent batching and rule-based routing
- **Performance**: 100,000+ messages/second throughput capability

### 2. Kafka Stream Processor (`kafka_stream_processor.py`)
Advanced stream processing with comprehensive monitoring:

- **Message Processing**: Configurable processors with retry and error handling
- **Circuit Breaker**: Automatic failure protection and recovery
- **Dead Letter Queue**: Failed message handling and analysis
- **Metrics & Monitoring**: Prometheus metrics and Redis health tracking
- **Exactly-Once Processing**: Ensures message processing guarantees

### 3. Real-Time Analytics Engine (`real_time_analytics.py`)
Advanced analytics with windowing and pattern detection:

- **Windowing**: Tumbling, sliding, session, and count-based windows
- **Aggregations**: Sum, count, average, percentiles, distinct counts
- **Pattern Detection**: Threshold breaches, trends, correlations, sequences
- **Anomaly Detection**: Statistical anomaly detection with z-score analysis
- **Real-Time Alerts**: Immediate notifications for detected patterns

### 4. Streaming Orchestrator (`streaming_orchestrator.py`)
Comprehensive orchestration and management:

- **Component Management**: Deploy, scale, and monitor streaming components
- **Pipeline Orchestration**: End-to-end pipeline deployment and monitoring
- **Auto-Healing**: Automatic failure detection and recovery
- **Auto-Scaling**: Dynamic scaling based on health and performance metrics
- **Kubernetes Integration**: Native Kubernetes deployment and management

## Quick Start

### 1. Initialize the Streaming System

```python
# Initialize Ingestion Engine
from real_time_processor.infrastructure.ingestion.high_throughput_ingestion import DataIngestionEngine, IngestionConfig

config = IngestionConfig(
    name="production_ingestion",
    max_workers=16,
    max_queue_size=100000,
    batch_size=1000,
    enable_validation=True,
    enable_deduplication=True
)

kafka_config = {
    "bootstrap_servers": ["kafka-1:9092", "kafka-2:9092", "kafka-3:9092"],
    "security_protocol": "SASL_SSL",
    "sasl_mechanism": "PLAIN",
    "sasl_plain_username": "producer",
    "sasl_plain_password": "secure_password"
}

ingestion_engine = DataIngestionEngine(config, kafka_config)
await ingestion_engine.initialize()
await ingestion_engine.start()

# Initialize Stream Processor
from real_time_processor.infrastructure.streaming.kafka_stream_processor import KafkaStreamProcessor, ProcessorConfig

processor_config = ProcessorConfig(
    name="analytics_processor",
    topics=["user_events", "system_metrics", "application_logs"],
    consumer_group="analytics_group",
    batch_size=500,
    max_retries=3,
    dead_letter_topic="failed_messages"
)

stream_processor = KafkaStreamProcessor(processor_config, kafka_config)
await stream_processor.initialize()
await stream_processor.start_processing()
```

### 2. Set Up Real-Time Analytics

```python
from real_time_processor.infrastructure.processing.real_time_analytics import RealTimeAnalyticsEngine, WindowConfig, WindowType

analytics_engine = RealTimeAnalyticsEngine(kafka_config)
await analytics_engine.initialize()

# Configure 5-minute tumbling window
window_config = WindowConfig(
    name="5min_metrics",
    window_type=WindowType.TUMBLING,
    size_seconds=300
)
analytics_engine.register_window(window_config)

# Add aggregations
from real_time_processor.infrastructure.processing.real_time_analytics import AggregationConfig, AggregationType

analytics_engine.register_aggregation(
    "5min_metrics",
    AggregationConfig("avg_response_time", AggregationType.AVERAGE, "response_time")
)

analytics_engine.register_aggregation(
    "5min_metrics", 
    AggregationConfig("error_count", AggregationType.COUNT, "error_indicator")
)

await analytics_engine.start()
```

### 3. Deploy with Orchestration

```python
from streaming.orchestration.streaming_orchestrator import StreamingOrchestrator, ComponentConfig, ComponentType

orchestrator = StreamingOrchestrator()
await orchestrator.initialize()

# Register components
ingestion_component = ComponentConfig(
    name="data-ingestion",
    component_type=ComponentType.INGESTION,
    image="streaming/ingestion:v1.2.0",
    replicas=3,
    resources={"cpu": "1", "memory": "2Gi"},
    scaling_config={
        "enabled": True,
        "min_replicas": 2,
        "max_replicas": 10,
        "target_health": 0.8
    }
)

orchestrator.register_component(ingestion_component)
await orchestrator.start()

# Deploy component
deployment_id = await orchestrator.deploy_component("data-ingestion")
```

## Configuration

### Environment Variables

```bash
# Kafka Configuration
export KAFKA_BOOTSTRAP_SERVERS="kafka-1:9092,kafka-2:9092,kafka-3:9092"
export KAFKA_SECURITY_PROTOCOL="SASL_SSL"
export KAFKA_SASL_MECHANISM="PLAIN"
export KAFKA_SASL_USERNAME="streaming_user"
export KAFKA_SASL_PASSWORD="secure_password"

# Redis Configuration
export REDIS_URL="redis://redis-cluster:6379/0"
export REDIS_PASSWORD="redis_password"

# Database Configuration
export POSTGRES_URL="postgresql://streaming:password@postgres:5432/streaming_db"

# Monitoring Configuration
export PROMETHEUS_URL="http://prometheus:9090"
export GRAFANA_URL="http://grafana:3000"

# Kubernetes Configuration
export KUBERNETES_NAMESPACE="streaming-system"
export KUBERNETES_CONFIG_PATH="/etc/kubernetes/config"

# Alerting Configuration
export SLACK_WEBHOOK_URL="https://hooks.slack.com/your-webhook"
export EMAIL_NOTIFICATIONS="streaming-team@company.com"
```

### Kubernetes Deployment

```yaml
# streaming-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: streaming-system
  labels:
    name: streaming-system

---
# ingestion-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-ingestion
  namespace: streaming-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: data-ingestion
  template:
    metadata:
      labels:
        app: data-ingestion
        component: ingestion
    spec:
      containers:
      - name: ingestion
        image: streaming/ingestion:v1.2.0
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka-1:9092,kafka-2:9092,kafka-3:9092"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: streaming-secrets
              key: redis-url
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8000
          name: metrics
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
# orchestrator-deployment.yaml
apiVersion: apps/v1
kind: Deployment  
metadata:
  name: streaming-orchestrator
  namespace: streaming-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: streaming-orchestrator
  template:
    metadata:
      labels:
        app: streaming-orchestrator
    spec:
      serviceAccountName: streaming-orchestrator
      containers:
      - name: orchestrator
        image: streaming/orchestrator:v1.2.0
        resources:
          requests:
            cpu: "200m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "2Gi"
        env:
        - name: KUBERNETES_NAMESPACE
          value: "streaming-system"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: streaming-secrets
              key: redis-url

---
# Service Account for Orchestrator
apiVersion: v1
kind: ServiceAccount
metadata:
  name: streaming-orchestrator
  namespace: streaming-system

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: streaming-orchestrator
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch", "delete"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: streaming-orchestrator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: streaming-orchestrator
subjects:
- kind: ServiceAccount
  name: streaming-orchestrator
  namespace: streaming-system
```

## Data Flow Architecture

### 1. Ingestion Flow
```
External Sources → Validation → Deduplication → Routing → Kafka Topics
     ↓               ↓            ↓             ↓          ↓
  - APIs         - Schema      - Redis       - Rules   - Partitioned
  - Files        - Rules       - TTL         - Logic   - Replicated  
  - Databases    - Transform   - Hash        - Dynamic - Durable
  - Webhooks     - Enrich      - Check       - Config  - Ordered
```

### 2. Processing Flow
```
Kafka Topics → Stream Processors → Analytics Engine → Results Storage
     ↓              ↓                    ↓                 ↓
  - Multiple     - Parallel          - Windowing        - Database
  - Partitions   - Processing        - Aggregation      - Cache
  - Replicated   - Error Handling    - Pattern Detection - Notifications
  - Ordered      - Dead Letter       - Anomaly Detection - Dashboards
```

### 3. Orchestration Flow
```
Configuration → Deployment → Monitoring → Auto-Scaling → Healing
     ↓             ↓           ↓            ↓             ↓
  - Components  - Kubernetes - Health     - Metrics     - Detection
  - Pipelines   - Pods       - Status     - Thresholds  - Recovery
  - Scaling     - Services   - Metrics    - Decisions   - Restart
  - Rules       - Secrets    - Alerts     - Actions     - Replace
```

## Performance Optimization

### 1. Ingestion Optimization
```python
# High-throughput configuration
config = IngestionConfig(
    name="high_perf_ingestion",
    max_workers=32,  # Increase for CPU-intensive workloads
    max_queue_size=500000,  # Large queue for burst handling
    batch_size=5000,  # Larger batches for better throughput
    flush_interval_seconds=1,  # Faster flushing for low latency
    enable_compression=True,  # Reduce network overhead
    buffer_size=50000  # Large buffer for batch optimization
)

# Producer optimization
kafka_config = {
    "bootstrap_servers": ["kafka-1:9092", "kafka-2:9092", "kafka-3:9092"],
    "batch_size": 65536,  # Large batch size
    "linger_ms": 5,  # Small linger for responsiveness
    "compression_type": "lz4",  # Fast compression
    "acks": "1",  # Balance between speed and durability
    "retries": 3,
    "max_in_flight_requests_per_connection": 5
}
```

### 2. Processing Optimization
```python
# Stream processor optimization
processor_config = ProcessorConfig(
    name="high_perf_processor",
    topics=["high_volume_topic"],
    consumer_group="performance_group",
    batch_size=1000,  # Process in batches
    max_poll_interval_ms=600000,  # Longer poll interval
    session_timeout_ms=45000,  # Longer session timeout
    processing_timeout_seconds=60,  # Adequate processing time
    buffer_size=100000  # Large buffer for batch processing
)

# Consumer optimization
consumer_config = {
    "fetch_min_bytes": 50000,  # Fetch larger batches
    "fetch_max_wait_ms": 100,  # Balance latency and throughput
    "max_partition_fetch_bytes": 10485760,  # 10MB max fetch
    "receive_buffer_bytes": 262144,  # 256KB buffer
    "send_buffer_bytes": 131072   # 128KB buffer
}
```

### 3. Analytics Optimization
```python
# Analytics engine optimization
analytics_config = {
    "window_buffer_size": 100000,  # Large window buffers
    "aggregation_workers": 8,  # Parallel aggregation
    "pattern_detection_interval": 30,  # Less frequent pattern checks
    "metrics_collection_interval": 60,  # Optimize metrics collection
    "cleanup_interval": 3600  # Hourly cleanup
}

# Memory optimization for windows
window_config = WindowConfig(
    name="optimized_window",
    window_type=WindowType.TUMBLING,
    size_seconds=60,  # Smaller windows for memory efficiency
    enable_late_data=False,  # Disable if not needed
    late_data_threshold_seconds=30  # Shorter threshold
)
```

## Monitoring and Alerting

### Key Metrics

#### Ingestion Metrics
- `ingestion_messages_total` - Total messages processed
- `ingestion_processing_seconds` - Processing time histogram
- `ingestion_queue_size` - Queue size by source
- `ingestion_throughput_per_second` - Messages per second
- `ingestion_validation_errors_total` - Validation errors

#### Processing Metrics
- `stream_messages_processed_total` - Processed messages by status
- `stream_processing_seconds` - Processing time by topic
- `stream_consumer_lag` - Consumer lag by partition
- `stream_processing_errors_total` - Processing errors

#### Analytics Metrics
- `analytics_events_processed_total` - Analytics events processed
- `analytics_active_windows` - Number of active windows
- `analytics_patterns_detected_total` - Patterns detected
- `analytics_anomalies_detected_total` - Anomalies detected

#### Orchestration Metrics
- `orchestration_operations_total` - Orchestration operations
- `streaming_component_health` - Component health scores
- `streaming_active_components` - Active components by type
- `streaming_total_throughput` - Total system throughput

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Real-Time Streaming System",
    "panels": [
      {
        "title": "Ingestion Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ingestion_messages_total[5m])",
            "legendFormat": "{{source}} - {{status}}"
          }
        ]
      },
      {
        "title": "Processing Latency",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(stream_processing_seconds_bucket[5m])",
            "legendFormat": "{{topic}}"
          }
        ]
      },
      {
        "title": "Component Health",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(streaming_component_health)",
            "legendFormat": "Average Health"
          }
        ]
      },
      {
        "title": "Error Rates",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ingestion_validation_errors_total[5m])",
            "legendFormat": "Validation Errors"
          },
          {
            "expr": "rate(stream_processing_errors_total[5m])",
            "legendFormat": "Processing Errors"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# prometheus-alerts.yaml
groups:
- name: streaming-system
  rules:
  - alert: HighIngestionLatency
    expr: histogram_quantile(0.95, rate(ingestion_processing_seconds_bucket[5m])) > 1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High ingestion latency detected"
      description: "95th percentile ingestion latency is {{ $value }}s"

  - alert: ProcessingErrors
    expr: rate(stream_processing_errors_total[5m]) > 0.01
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "High processing error rate"
      description: "Processing error rate is {{ $value }} errors/sec"

  - alert: ComponentUnhealthy
    expr: streaming_component_health < 0.7
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "Component health degraded"
      description: "Component {{ $labels.component }} health is {{ $value }}"

  - alert: ConsumerLag
    expr: stream_consumer_lag > 10000
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High consumer lag"
      description: "Consumer lag for {{ $labels.topic }} is {{ $value }} messages"
```

## Scaling Guidelines

### Horizontal Scaling

#### Ingestion Scaling
```python
# Scale based on queue size and throughput
def calculate_ingestion_replicas(queue_size, throughput, target_queue_size=10000):
    if queue_size > target_queue_size:
        scale_factor = queue_size / target_queue_size
        return min(20, max(2, int(scale_factor)))
    return 2

# Auto-scaling configuration
scaling_config = {
    "enabled": True,
    "min_replicas": 2,
    "max_replicas": 20,
    "target_metrics": {
        "queue_size": 10000,
        "cpu_utilization": 0.7,
        "memory_utilization": 0.8
    },
    "scale_up_cooldown": 300,
    "scale_down_cooldown": 600
}
```

#### Processing Scaling
```python
# Scale based on consumer lag and processing time
def calculate_processor_replicas(consumer_lag, processing_time, target_lag=5000):
    if consumer_lag > target_lag or processing_time > 1.0:
        scale_factor = max(consumer_lag / target_lag, processing_time)
        return min(15, max(1, int(scale_factor)))
    return 1

# Kafka partition scaling
def calculate_partitions(throughput_mbps, target_partition_throughput=10):
    return max(1, int(throughput_mbps / target_partition_throughput))
```

### Vertical Scaling

#### Resource Allocation
```yaml
# Resource templates for different workloads
resources:
  small:
    requests:
      cpu: "250m"
      memory: "512Mi"
    limits:
      cpu: "500m"
      memory: "1Gi"
  
  medium:
    requests:
      cpu: "500m"
      memory: "1Gi"
    limits:
      cpu: "1"
      memory: "2Gi"
  
  large:
    requests:
      cpu: "1"
      memory: "2Gi"
    limits:
      cpu: "2"
      memory: "4Gi"
  
  xlarge:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
```

## Troubleshooting

### Common Issues

#### 1. High Ingestion Latency
```bash
# Check queue sizes
kubectl exec -it deployment/data-ingestion -- curl localhost:8080/metrics | grep queue_size

# Check worker utilization
kubectl top pods -l app=data-ingestion

# Scale up if needed
kubectl scale deployment data-ingestion --replicas=6
```

#### 2. Processing Errors
```bash
# Check error logs
kubectl logs -l app=stream-processor --tail=100 | grep ERROR

# Check dead letter queue
kafka-console-consumer --bootstrap-server kafka:9092 --topic failed_messages

# Restart processing if needed
kubectl rollout restart deployment/stream-processor
```

#### 3. Consumer Lag
```bash
# Check consumer group status
kafka-consumer-groups --bootstrap-server kafka:9092 --group analytics_group --describe

# Check partition distribution
kubectl get pods -l app=stream-processor -o wide

# Increase partitions if needed
kafka-topics --bootstrap-server kafka:9092 --topic user_events --alter --partitions 12
```

#### 4. Component Health Issues
```bash
# Check orchestrator status
kubectl exec -it deployment/streaming-orchestrator -- curl localhost:8080/status

# Check component health
kubectl get pods -l component=streaming -o wide

# Force healing
kubectl exec -it deployment/streaming-orchestrator -- curl -X POST localhost:8080/heal/component-name
```

### Debugging Commands

```bash
# View real-time metrics
watch kubectl get pods -l component=streaming

# Check resource usage
kubectl top pods -l component=streaming

# View orchestrator logs
kubectl logs -f deployment/streaming-orchestrator

# Check Redis connectivity
kubectl exec -it deployment/data-ingestion -- redis-cli -h redis ping

# Test Kafka connectivity
kubectl exec -it deployment/stream-processor -- kafka-console-producer --bootstrap-server kafka:9092 --topic test
```

## Security Considerations

### 1. Network Security
- Use TLS for all Kafka connections
- Implement VPC/network policies for pod communication
- Enable mutual TLS for internal service communication

### 2. Authentication & Authorization
- Use SASL/SCRAM for Kafka authentication
- Implement RBAC for Kubernetes resources
- Use service accounts with minimal permissions

### 3. Data Protection
- Encrypt data in transit and at rest
- Implement data masking for sensitive fields
- Regular security audits and vulnerability scanning

### 4. Secrets Management
```yaml
# Use Kubernetes secrets for sensitive data
apiVersion: v1
kind: Secret
metadata:
  name: streaming-secrets
  namespace: streaming-system
type: Opaque
data:
  kafka-username: <base64-encoded>
  kafka-password: <base64-encoded>
  redis-url: <base64-encoded>
```

## Best Practices

### 1. Development
- Use schema registry for message schemas
- Implement comprehensive testing for stream processors
- Use feature flags for gradual rollouts
- Maintain backward compatibility

### 2. Operations
- Monitor all components continuously
- Implement automated alerts and responses
- Regular capacity planning and performance testing
- Document all operational procedures

### 3. Data Management
- Implement proper data retention policies
- Use compression for large messages
- Consider data partitioning strategies
- Regular backup and disaster recovery testing

## Support and Resources

- **Documentation**: `streaming/docs/`
- **API Documentation**: `http://localhost:8080/docs`
- **Monitoring Dashboard**: `http://grafana.streaming.local`
- **Support Channel**: `#streaming-system`
- **Issue Tracker**: GitHub Issues with `streaming` label

For detailed implementation examples and advanced configuration, see the `examples/` directory and component-specific documentation.