# Performance Degradation Detection Guide

## Overview

Pynomaly's Performance Degradation Detection system provides comprehensive monitoring and alerting capabilities for machine learning models in production. This feature automatically detects when model performance begins to degrade and can trigger automated responses including retraining workflows.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Usage Examples](#usage-examples)
6. [Alert Handling](#alert-handling)
7. [Integration with Automated Retraining](#integration-with-automated-retraining)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Architecture Overview

The Performance Degradation Detection system is built on a layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │        PerformanceMonitoringService                 │    │
│  │  - Baseline Management                              │    │
│  │  - Regression Detection                             │    │
│  │  - Trend Analysis                                   │    │
│  │  - Alert Coordination                               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           PerformanceMonitor                        │    │
│  │  - Real-time Metrics Collection                     │    │
│  │  - Threshold-based Alerting                         │    │
│  │  - Historical Data Storage                          │    │
│  │  - Export Capabilities                              │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. PerformanceMonitor

The core infrastructure component that handles:
- Real-time metrics collection (CPU, memory, execution time, throughput)
- Threshold-based alerting
- Historical data storage with configurable retention
- Export capabilities (JSON, CSV)
- Alert callback system

### 2. PerformanceMonitoringService

The application service that provides:
- High-level performance monitoring operations
- Baseline management and comparison
- Regression detection algorithms
- Performance trend analysis
- Integration with automated retraining

### 3. PerformanceMetrics

Data container for performance measurements:
- **Timing metrics**: execution_time, setup_time, cleanup_time
- **Resource metrics**: cpu_usage, memory_usage, memory_peak
- **Throughput metrics**: samples_processed, samples_per_second
- **Quality metrics**: accuracy, precision, recall, f1_score
- **Metadata**: timestamps, operation names, algorithm names

### 4. PerformanceAlert

Alert container for performance issues:
- **Alert types**: threshold_exceeded, anomaly_detected, degradation
- **Severity levels**: low, medium, high, critical
- **Alert metadata**: timestamps, threshold values, current values

## Configuration

### Basic Configuration

```python
from pynomaly.infrastructure.monitoring.performance_monitor import PerformanceMonitor
from pynomaly.application.services.performance_monitoring_service import PerformanceMonitoringService

# Configure performance monitor
performance_monitor = PerformanceMonitor(
    max_history=1000,                    # Maximum metrics to store
    alert_thresholds={
        "execution_time": 30.0,          # seconds
        "memory_usage": 1000.0,          # MB
        "cpu_usage": 80.0,               # percentage
        "samples_per_second": 100.0      # minimum throughput
    },
    monitoring_interval=1.0              # seconds
)

# Create monitoring service
monitoring_service = PerformanceMonitoringService(
    performance_monitor=performance_monitor,
    auto_start_monitoring=True
)
```

### Advanced Configuration

```python
# Custom alert thresholds per operation
custom_thresholds = {
    "isolation_forest": {
        "execution_time": 10.0,
        "memory_usage": 500.0,
        "cpu_usage": 60.0
    },
    "one_class_svm": {
        "execution_time": 20.0,
        "memory_usage": 800.0,
        "cpu_usage": 70.0
    }
}

# Configure with custom settings
performance_monitor = PerformanceMonitor(
    max_history=5000,
    alert_thresholds=custom_thresholds.get("isolation_forest", {}),
    monitoring_interval=0.5
)
```

## API Reference

### PerformanceMonitoringService

#### Core Methods

```python
# Start/Stop monitoring
monitoring_service.start_monitoring()
monitoring_service.stop_monitoring()

# Monitor operations
result, metrics = monitoring_service.monitor_detection_operation(
    detector=detector,
    dataset=dataset,
    operation_func=detection_function
)

result, metrics = monitoring_service.monitor_training_operation(
    detector=detector,
    dataset=dataset,
    training_func=training_function
)
```

#### Baseline Management

```python
# Set performance baseline
monitoring_service.set_performance_baseline(
    operation_name="isolation_forest_detection",
    baseline_metrics={
        "execution_time": 2.5,
        "memory_usage": 150.0,
        "cpu_usage": 45.0
    }
)

# Check for regression
regression_result = monitoring_service.check_performance_regression(
    operation_name="isolation_forest_detection",
    recent_window=timedelta(hours=1)
)
```

#### Analysis Methods

```python
# Get performance trends
trends = monitoring_service.get_performance_trends(
    operation_name="isolation_forest_detection",
    time_window=timedelta(days=7),
    bucket_size=timedelta(hours=1)
)

# Compare algorithm performance
comparison = monitoring_service.get_algorithm_performance_comparison(
    time_window=timedelta(hours=24),
    min_operations=3
)

# Get dashboard data
dashboard_data = monitoring_service.get_monitoring_dashboard_data()
```

### PerformanceMonitor

#### Low-level Operations

```python
# Start/end operation tracking
operation_id = performance_monitor.start_operation(
    operation_name="detection",
    algorithm_name="IsolationForest",
    dataset_size=1000
)

metrics = performance_monitor.end_operation(
    operation_id=operation_id,
    samples_processed=1000,
    quality_metrics={"accuracy": 0.95}
)
```

#### Alert Management

```python
# Add alert callback
def alert_handler(alert):
    print(f"Alert: {alert.message}")

performance_monitor.add_alert_callback(alert_handler)

# Get active alerts
active_alerts = performance_monitor.get_active_alerts()

# Clear alerts
performance_monitor.clear_alerts(alert_type="threshold_exceeded")
```

## Usage Examples

### Basic Performance Monitoring

```python
import pandas as pd
import numpy as np
from pynomaly.application.services.performance_monitoring_service import PerformanceMonitoringService
from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

# Initialize monitoring
monitoring_service = PerformanceMonitoringService()

# Create data and detector
data = pd.DataFrame(np.random.normal(0, 1, (1000, 5)))
dataset = Dataset(name="Production Data", data=data)
detector = SklearnAdapter(algorithm_name="IsolationForest", name="Detector")

# Monitor detection operation
def run_detection(detector, dataset):
    detector.fit(dataset)
    return detector.detect(dataset)

result, metrics = monitoring_service.monitor_detection_operation(
    detector=detector,
    dataset=dataset,
    operation_func=run_detection
)

print(f"Execution time: {metrics.execution_time:.2f}s")
print(f"Memory usage: {metrics.memory_usage:.1f}MB")
```

### Performance Regression Detection

```python
from datetime import timedelta

# Set baseline
monitoring_service.set_performance_baseline(
    operation_name="detection_IsolationForest",
    baseline_metrics={
        "execution_time": 2.0,
        "memory_usage": 100.0,
        "cpu_usage": 40.0
    }
)

# Run multiple operations to build history
for i in range(10):
    result, metrics = monitoring_service.monitor_detection_operation(
        detector=detector,
        dataset=dataset,
        operation_func=run_detection
    )
    time.sleep(1)  # Simulate real-world timing

# Check for regression
regression_result = monitoring_service.check_performance_regression(
    operation_name="detection_IsolationForest",
    recent_window=timedelta(minutes=5)
)

if regression_result.get("regressions_detected", 0) > 0:
    print("⚠️ Performance regression detected!")
    for metric, regression in regression_result["regressions"].items():
        print(f"  {metric}: {regression['degradation_percent']:.1f}% degradation")
```

### Performance Trend Analysis

```python
# Analyze trends over time
trends = monitoring_service.get_performance_trends(
    operation_name="detection_IsolationForest",
    time_window=timedelta(hours=24),
    bucket_size=timedelta(hours=1)
)

print(f"Execution time trend: {trends['trends']['execution_time']}")
print(f"Memory usage trend: {trends['trends']['memory_usage']}")

# Compare algorithms
comparison = monitoring_service.get_algorithm_performance_comparison(
    time_window=timedelta(hours=24)
)

if comparison.get("rankings"):
    print("Algorithm rankings:")
    for criterion, ranking in comparison["rankings"].items():
        print(f"  {criterion}: {' > '.join(ranking)}")
```

## Alert Handling

### Custom Alert Handlers

```python
def handle_performance_alert(alert):
    """Custom alert handler for performance issues"""
    print(f"🚨 Performance Alert: {alert.severity.upper()}")
    print(f"   Metric: {alert.metric_name}")
    print(f"   Current: {alert.current_value}")
    print(f"   Threshold: {alert.threshold_value}")
    
    # Automated response based on severity
    if alert.severity == "critical":
        # Trigger immediate action
        trigger_emergency_response(alert)
    elif alert.severity == "high":
        # Schedule maintenance
        schedule_maintenance(alert)
    elif alert.severity == "medium":
        # Increase monitoring
        increase_monitoring_frequency(alert)

# Add handler to monitoring service
monitoring_service.add_alert_handler(handle_performance_alert)
```

### Alert Escalation

```python
def escalation_handler(alert):
    """Escalate alerts based on severity and frequency"""
    if alert.severity == "critical":
        # Immediate escalation
        send_sms_alert(alert)
        create_incident_ticket(alert)
    elif alert.severity == "high":
        # Email notification
        send_email_alert(alert)
    
    # Log all alerts
    log_alert_to_database(alert)

monitoring_service.add_alert_handler(escalation_handler)
```

## Integration with Automated Retraining

### Retraining Decision Logic

```python
from pynomaly.application.services.auto_retraining_service import AutoRetrainingService

# Initialize auto-retraining service
auto_retraining = AutoRetrainingService()

def retraining_decision_handler(alert):
    """Make retraining decisions based on performance alerts"""
    if alert.metric_name in ["accuracy", "f1_score"] and alert.severity in ["high", "critical"]:
        # Assess degradation
        degradation_percent = calculate_degradation_percent(alert)
        
        # Make retraining decision
        if degradation_percent > 25:
            print("🔄 Triggering automated retraining")
            
            # Create retraining plan
            plan = auto_retraining.create_retraining_plan(
                model_id=get_model_id_from_alert(alert),
                trigger="performance_degradation"
            )
            
            # Execute retraining
            result = auto_retraining.execute_retraining_plan(plan)
            
            if result.success:
                print("✅ Retraining completed successfully")
                # Update baseline with new performance
                update_performance_baseline(alert.operation_name, result)
            else:
                print("❌ Retraining failed")
                escalate_retraining_failure(alert, result)

# Add retraining handler
monitoring_service.add_alert_handler(retraining_decision_handler)
```

### Continuous Learning Integration

```python
def continuous_learning_handler(alert):
    """Integrate with continuous learning system"""
    if alert.alert_type == "degradation":
        # Update learning parameters
        update_learning_rate(alert.operation_name, alert.current_value)
        
        # Trigger incremental learning
        trigger_incremental_learning(
            model_id=get_model_id_from_alert(alert),
            performance_metrics=get_recent_metrics(alert.operation_name)
        )

monitoring_service.add_alert_handler(continuous_learning_handler)
```

## Best Practices

### 1. Threshold Configuration

```python
# Set realistic thresholds based on historical data
thresholds = {
    "execution_time": calculate_p95_execution_time() * 1.5,
    "memory_usage": calculate_avg_memory_usage() * 2.0,
    "cpu_usage": 80.0,  # Fixed threshold
    "samples_per_second": calculate_min_acceptable_throughput()
}
```

### 2. Baseline Management

```python
# Regularly update baselines
def update_baselines_weekly():
    """Update baselines weekly based on recent performance"""
    for operation_name in get_active_operations():
        recent_metrics = get_recent_performance_metrics(
            operation_name, 
            time_window=timedelta(days=7)
        )
        
        if len(recent_metrics) >= 50:  # Minimum sample size
            new_baseline = calculate_baseline_from_metrics(recent_metrics)
            monitoring_service.set_performance_baseline(
                operation_name=operation_name,
                baseline_metrics=new_baseline
            )
```

### 3. Alert Fatigue Prevention

```python
# Implement alert suppression
def smart_alert_handler(alert):
    """Prevent alert fatigue with intelligent suppression"""
    
    # Check if similar alert was recently sent
    if is_duplicate_alert(alert, time_window=timedelta(minutes=30)):
        return  # Suppress duplicate
    
    # Check if alert is part of expected degradation pattern
    if is_expected_degradation(alert):
        return  # Suppress expected degradation
    
    # Process alert normally
    process_alert(alert)

monitoring_service.add_alert_handler(smart_alert_handler)
```

### 4. Performance Optimization

```python
# Configure monitoring for high-volume environments
high_volume_config = PerformanceMonitor(
    max_history=10000,           # Larger history for better trends
    monitoring_interval=0.1,     # More frequent monitoring
    alert_thresholds={
        "execution_time": 1.0,   # Stricter thresholds
        "memory_usage": 200.0,
        "cpu_usage": 60.0
    }
)
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```python
   # Reduce history size
   performance_monitor = PerformanceMonitor(max_history=500)
   
   # Increase monitoring interval
   performance_monitor = PerformanceMonitor(monitoring_interval=5.0)
   ```

2. **False Positive Alerts**
   ```python
   # Adjust thresholds
   performance_monitor.alert_thresholds["execution_time"] = 10.0
   
   # Implement alert suppression
   def suppress_false_positives(alert):
       if alert.current_value < alert.threshold_value * 1.1:
           return  # Suppress if only slightly over threshold
       process_alert(alert)
   ```

3. **Missing Metrics**
   ```python
   # Check if monitoring is enabled
   if not monitoring_service.monitoring_enabled:
       monitoring_service.start_monitoring()
   
   # Verify operation tracking
   metrics = monitoring_service.monitor.get_operation_statistics()
   print(f"Total operations: {metrics.get('operation_count', 0)}")
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Export metrics for analysis
json_export = monitoring_service.monitor.export_metrics(
    format_type="json",
    time_window=timedelta(hours=1)
)

# Analyze exported data
for metric in json_export["metrics"]:
    print(f"Operation: {metric['operation_name']}")
    print(f"Execution time: {metric['execution_time']}")
    print(f"Memory usage: {metric['memory_usage']}")
```

## CLI Commands

```bash
# Monitor performance in real-time
pynomaly perf monitor

# Run performance benchmarks
pynomaly perf benchmark --suite comprehensive

# Generate performance report
pynomaly perf report --format html --output report.html

# Check for performance regression
pynomaly perf check-regression --operation isolation_forest

# Set performance baseline
pynomaly perf set-baseline --operation detector --execution-time 2.5

# Export metrics
pynomaly perf export --format json --output metrics.json
```

## Conclusion

The Performance Degradation Detection system provides a comprehensive solution for monitoring machine learning model performance in production environments. By combining real-time monitoring, intelligent alerting, and automated response capabilities, it enables proactive maintenance of ML systems and helps prevent performance degradation from impacting production workloads.

For more advanced usage and integration examples, see the [sample notebook](../sample_notebook.ipynb) included in this documentation.
