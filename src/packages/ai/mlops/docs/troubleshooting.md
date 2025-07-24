# MLOps Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using the MLOps package.

## Common Issues and Solutions

### Experiment Tracking Issues

#### Issue: MLflow Connection Errors

**Symptoms:**
- `ConnectionError: Could not connect to MLflow tracking server`
- `HTTPError: 404 Client Error`
- Experiments not appearing in MLflow UI

**Solutions:**

1. **Check MLflow server status:**
```bash
# Check if MLflow server is running
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Start MLflow server if not running
mlflow server --host 0.0.0.0 --port 5000
```

2. **Verify connection configuration:**
```python
from mlops.experiments import ExperimentTracker

# Check current tracking URI
import mlflow
print(f"Current tracking URI: {mlflow.get_tracking_uri()}")

# Set correct URI
tracker = ExperimentTracker(tracking_uri="http://localhost:5000")
```

3. **Network connectivity issues:**
```bash
# Test network connectivity
ping localhost
telnet localhost 5000

# Check firewall rules
sudo ufw status
sudo iptables -L
```

#### Issue: Experiment Logging Slow

**Symptoms:**
- Long delays when logging metrics
- Timeouts during experiment operations
- High memory usage during logging

**Solutions:**

1. **Optimize logging frequency:**
```python
# Batch metrics logging
metrics_batch = {}
for step in range(100):
    metrics_batch[f"step_{step}"] = compute_metric(step)

# Log batch at once
experiment.log_metrics(metrics_batch)
```

2. **Use async logging:**
```python
import asyncio
from mlops.experiments import AsyncExperimentTracker

async def log_metrics_async():
    tracker = AsyncExperimentTracker()
    await tracker.log_metrics_async(metrics)
```

3. **Configure MLflow backend:**
```python
# Use database backend instead of file store
mlflow.set_tracking_uri("postgresql://user:pass@localhost/mlflow")
```

### Model Registry Issues

#### Issue: Model Registration Failures

**Symptoms:**
- `ModelRegistryError: Failed to register model`
- `PermissionError: Access denied`
- Model artifacts not saved properly

**Solutions:**

1. **Check storage permissions:**
```bash
# Check artifact store permissions
ls -la /path/to/mlflow/artifacts
chmod 755 /path/to/mlflow/artifacts

# For S3 backend
aws s3 ls s3://mlflow-artifacts/
aws iam get-user
```

2. **Verify model serialization:**
```python
import pickle
import joblib

# Test model serialization
try:
    serialized = pickle.dumps(model)
    deserialized = pickle.loads(serialized)
    print("Model serialization successful")
except Exception as e:
    print(f"Serialization error: {e}")
```

3. **Check disk space:**
```bash
df -h
du -sh /path/to/mlflow/artifacts
```

#### Issue: Model Version Conflicts

**Symptoms:**
- `ModelVersionConflictError: Version already exists`
- Unable to promote model stages
- Model metadata inconsistencies

**Solutions:**

1. **Use proper versioning:**
```python
from mlops.models import ModelRegistry

registry = ModelRegistry()

# Check existing versions
versions = registry.list_model_versions("model_name")
print(f"Existing versions: {[v.version for v in versions]}")

# Register with auto-increment
new_version = registry.register_model(
    name="model_name",
    model=model,
    auto_increment=True
)
```

2. **Clean up orphaned versions:**
```python
# Remove unused model versions
registry.delete_model_version("model_name", "1.0.0")
```

### Deployment Issues

#### Issue: Kubernetes Deployment Failures

**Symptoms:**
- `DeploymentError: Failed to deploy to Kubernetes`
- Pods in `CrashLoopBackOff` state
- Resource allocation errors

**Solutions:**

1. **Check cluster resources:**
```bash
kubectl get nodes
kubectl describe nodes
kubectl top nodes

# Check available resources
kubectl describe pod <pod-name>
kubectl get events --sort-by=.metadata.creationTimestamp
```

2. **Verify Docker image:**
```bash
# Test Docker image locally
docker run -p 8080:8080 your-model-image:latest

# Check image layers
docker history your-model-image:latest
```

3. **Review deployment configuration:**
```python
from mlops.models.deployment import KubernetesDeployment

deployment = KubernetesDeployment(
    model_version="model_v1",
    replicas=1,  # Start with 1 replica
    resources={
        "requests": {"cpu": "100m", "memory": "256Mi"},
        "limits": {"cpu": "500m", "memory": "1Gi"}
    }
)
```

#### Issue: Model Serving Errors

**Symptoms:**
- `HTTP 500 Internal Server Error`
- High latency responses
- Memory leaks in serving containers

**Solutions:**

1. **Check model loading:**
```python
# Test model loading locally
from mlops.models import ModelRegistry

registry = ModelRegistry()
model = registry.get_model("model_name", "latest")

# Test prediction
try:
    prediction = model.predict(test_data)
    print("Model prediction successful")
except Exception as e:
    print(f"Prediction error: {e}")
```

2. **Monitor resource usage:**
```bash
# Check container resources
kubectl top pods
kubectl logs <pod-name>

# Check memory usage
kubectl exec <pod-name> -- free -h
kubectl exec <pod-name> -- df -h
```

3. **Optimize model serving:**
```python
# Use model batching
from mlops.models.serving import BatchedModelServing

serving = BatchedModelServing(
    model=model,
    batch_size=32,
    timeout=1.0
)
```

### Monitoring Issues

#### Issue: Drift Detection False Positives

**Symptoms:**
- Frequent drift alerts
- Drift detected on stable data
- Inconsistent drift scores

**Solutions:**

1. **Adjust detection thresholds:**
```python
from mlops.monitoring import DriftDetector

detector = DriftDetector(
    reference_data=reference_data,
    threshold=0.2,  # Increase threshold
    window_size="2h",  # Increase window size
    min_samples=1000  # Require minimum samples
)
```

2. **Use multiple drift detection methods:**
```python
detector = DriftDetector(
    methods=["ks_test", "psi", "js_divergence"],
    consensus_threshold=0.6  # Require 60% consensus
)
```

3. **Update reference data:**
```python
# Periodically update reference data
detector.update_reference_data(new_reference_data)
```

#### Issue: Performance Monitoring Delays

**Symptoms:**
- Delayed metric updates
- Missing performance data
- Dashboard not updating

**Solutions:**

1. **Check monitoring pipeline:**
```python
from mlops.monitoring import PerformanceMonitor

monitor = PerformanceMonitor(
    model_version="model_v1",
    buffer_size=100,  # Reduce buffer size
    flush_interval=30  # Flush every 30 seconds
)
```

2. **Verify data collection:**
```bash
# Check monitoring logs
kubectl logs -f monitoring-deployment
tail -f /var/log/mlops/monitoring.log
```

### Pipeline Issues

#### Issue: Pipeline Execution Failures

**Symptoms:**
- `PipelineExecutionError: Pipeline failed`
- Tasks stuck in pending state
- Resource exhaustion during pipeline runs

**Solutions:**

1. **Check pipeline configuration:**
```python
from mlops.pipelines import TrainingPipeline

pipeline = TrainingPipeline(
    name="training_pipeline",
    config={
        "timeout": 3600,  # 1 hour timeout
        "retry_count": 3,
        "resource_limits": {
            "cpu": "2",
            "memory": "4Gi"
        }
    }
)
```

2. **Monitor pipeline execution:**
```bash
# Check pipeline status
kubectl get pods -l job-name=training-pipeline
kubectl logs -f job/training-pipeline
```

3. **Optimize pipeline steps:**
```python
# Use parallel execution
pipeline.add_step("data_preprocessing", parallel=True)
pipeline.add_step("feature_extraction", parallel=True)
pipeline.add_step("model_training", depends_on=["data_preprocessing", "feature_extraction"])
```

### Data Issues

#### Issue: Feature Store Access Errors

**Symptoms:**
- `FeatureStoreError: Cannot access feature store`
- Missing feature values
- Stale feature data

**Solutions:**

1. **Check feature store connection:**
```python
from mlops.features import FeatureStore

store = FeatureStore(
    backend="feast",
    offline_store="s3://features/offline",
    online_store="redis://localhost:6379"
)

# Test connection
try:
    store.health_check()
    print("Feature store connection successful")
except Exception as e:
    print(f"Connection error: {e}")
```

2. **Verify feature definitions:**
```python
# List available features
features = store.list_features()
print(f"Available features: {features}")

# Check feature freshness
freshness = store.get_feature_freshness("feature_name")
print(f"Feature freshness: {freshness}")
```

## Debug Mode

### Enable Debug Logging

```python
import logging
from mlops.config import enable_debug_mode

# Enable debug mode
enable_debug_mode(
    experiment_tracking=True,
    model_serving=True,
    monitoring=True,
    pipeline_execution=True
)

# Set logging level
logging.basicConfig(level=logging.DEBUG)
```

### Debug-Specific Tools

```python
from mlops.debug import DebugTools

debug_tools = DebugTools()

# Trace experiment operations
with debug_tools.trace_experiment():
    experiment.log_metrics(metrics)

# Monitor resource usage
debug_tools.monitor_resources()

# Validate model performance
debug_tools.validate_model(model, test_data)
```

## Performance Optimization

### Memory Optimization

```python
# Use memory-efficient data structures
import numpy as np
from mlops.utils import MemoryOptimizer

optimizer = MemoryOptimizer()

# Optimize data loading
data = optimizer.load_data_efficiently(data_path)

# Monitor memory usage
optimizer.monitor_memory_usage()
```

### CPU Optimization

```python
# Use parallel processing
from concurrent.futures import ThreadPoolExecutor
import asyncio

async def parallel_training():
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(train_model, data_chunk)
            for data_chunk in data_chunks
        ]
        results = await asyncio.gather(*futures)
```

### I/O Optimization

```python
# Use async I/O operations
import aiofiles
import asyncio

async def async_data_loading():
    async with aiofiles.open('data.csv', 'r') as f:
        data = await f.read()
    return data
```

## Monitoring and Alerting

### Health Checks

```python
from mlops.monitoring import HealthChecker

health_checker = HealthChecker()

# Check component health
health_status = health_checker.check_all_components()
print(f"Health status: {health_status}")

# Set up automated health checks
health_checker.schedule_health_checks(interval=300)  # Every 5 minutes
```

### Custom Alerts

```python
from mlops.monitoring import AlertManager

alert_manager = AlertManager()

# Set up custom alerts
alert_manager.add_alert(
    name="model_latency",
    condition="latency > 1000ms",
    action="send_notification",
    channels=["slack", "email"]
)
```

## Troubleshooting Checklist

### Before Reporting Issues

1. **Check system requirements:**
   - Python version (3.11+)
   - Dependencies versions
   - System resources

2. **Verify configuration:**
   - Environment variables
   - Configuration files
   - Network settings

3. **Test connectivity:**
   - MLflow server
   - Database connections
   - Kubernetes cluster

4. **Review logs:**
   - Application logs
   - System logs
   - Container logs

5. **Check resources:**
   - CPU utilization
   - Memory usage
   - Disk space
   - Network bandwidth

### Collecting Debug Information

```python
from mlops.debug import collect_debug_info

# Collect comprehensive debug information
debug_info = collect_debug_info()

# Save to file
with open("debug_info.json", "w") as f:
    json.dump(debug_info, f, indent=2)
```

### Creating Support Tickets

When creating support tickets, include:

1. **Environment information:**
   - Python version
   - Package versions
   - Operating system
   - Hardware specifications

2. **Error details:**
   - Full error messages
   - Stack traces
   - Reproduction steps

3. **Configuration:**
   - Configuration files
   - Environment variables
   - Deployment manifests

4. **Logs:**
   - Application logs
   - System logs
   - Debug information

## Getting Help

### Community Resources

- **Documentation**: [MLOps Documentation](../docs/)
- **Examples**: [Code Examples](../examples/)
- **GitHub Issues**: [Report Issues](https://github.com/your-org/repo/issues)
- **Discussions**: [Community Discussions](https://github.com/your-org/repo/discussions)

### Support Channels

- **Email**: support@yourcompany.com
- **Slack**: #mlops-support
- **Office Hours**: Weekly office hours for live support

### Contributing

If you've solved a problem not covered in this guide:

1. Document the solution
2. Create a pull request
3. Add to this troubleshooting guide
4. Share with the community

## Preventive Measures

### Best Practices

1. **Regular health checks**
2. **Automated testing**
3. **Resource monitoring**
4. **Configuration validation**
5. **Regular backups**

### Monitoring Setup

```python
from mlops.monitoring import ProactiveMonitoring

monitor = ProactiveMonitoring()

# Set up proactive monitoring
monitor.enable_resource_monitoring()
monitor.enable_performance_monitoring()
monitor.enable_error_monitoring()
monitor.enable_security_monitoring()
```

This troubleshooting guide should help you resolve most common issues. For complex problems, don't hesitate to reach out to the community or support team.