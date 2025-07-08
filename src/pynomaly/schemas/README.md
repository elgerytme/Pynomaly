# Pynomaly Schemas Package

This package provides unified data contracts and metric schemas for the Pynomaly anomaly detection platform. It includes Pydantic models for real-time metrics, KPIs, system health monitoring, financial impact assessment, and ROI calculations.

## Overview

The schemas package is designed to provide:
- **Backward compatibility** with previous versions
- **Serializable schemas** for API responses and message queues
- **Data validation** for integrity and consistency
- **Optimized performance** for real-time processing
- **Version management** for schema evolution

## Package Structure

```
src/pynomaly/schemas/
├── __init__.py              # Main package exports
├── README.md                # This file
├── analytics/               # Analytics schemas
│   ├── __init__.py
│   ├── base.py             # Base metric frames
│   ├── anomaly_kpis.py     # Anomaly KPI schemas
│   ├── system_health.py    # System health schemas
│   ├── financial_impact.py # Financial impact schemas
│   └── roi.py              # ROI analysis schemas
├── validation.py           # Validation utilities
└── versioning.py           # Version management
```

## Core Schemas

### Base Schemas

#### `MetricFrame`
Base schema for all metric frames with common fields:
- `metric_id`: Unique identifier
- `name`: Human-readable name
- `value`: Metric value
- `timestamp`: Collection timestamp
- `metadata`: Optional metadata

#### `RealTimeMetricFrame`
Enhanced metric frame with real-time properties:
- All `MetricFrame` fields
- `delay`: Collection delay in seconds

### Anomaly KPIs

#### `AnomalyKPIFrame`
Main anomaly detection KPI frame containing:
- **Detection metrics**: Accuracy, precision, recall, F1-score, ROC/PR AUC
- **Classification metrics**: Confusion matrix, severity distribution
- **Time series metrics**: Latency, processing time, drift detection
- **Operational metrics**: Throughput, resource usage, alerts
- **Quality metrics**: Confidence and data quality scores

#### `AnomalyDetectionMetrics`
Core performance metrics:
```python
AnomalyDetectionMetrics(
    accuracy=0.95,
    precision=0.92,
    recall=0.88,
    f1_score=0.9,
    false_positive_rate=0.05,
    false_negative_rate=0.12,
    roc_auc=0.94,
    pr_auc=0.91
)
```

#### `AnomalyClassificationMetrics`
Classification quality metrics:
```python
AnomalyClassificationMetrics(
    true_positives=150,
    false_positives=8,
    true_negatives=1842,
    false_negatives=20,
    anomalies_detected=158,
    anomalies_confirmed=150,
    anomalies_dismissed=8,
    severity_distribution={
        AnomalySeverity.CRITICAL: 5,
        AnomalySeverity.HIGH: 25,
        AnomalySeverity.MEDIUM: 70,
        AnomalySeverity.LOW: 50
    }
)
```

### System Health

#### `SystemHealthFrame`
Comprehensive system health monitoring:
- **Resource metrics**: CPU, memory, disk, network usage
- **Performance metrics**: Response times, throughput, error rates
- **Status metrics**: Service health, alerts, maintenance mode
- **Health scores**: Overall, availability, reliability scores
- **Trend indicators**: Resource usage trends

#### `SystemResourceMetrics`
Resource utilization metrics:
```python
SystemResourceMetrics(
    cpu_usage_percent=45.2,
    cpu_load_average=1.8,
    cpu_cores=8,
    memory_usage_percent=68.5,
    memory_used_mb=5500.0,
    memory_total_mb=8192.0,
    disk_usage_percent=72.3,
    disk_used_gb=362.0,
    disk_total_gb=500.0,
    # ... networking and process metrics
)
```

### Financial Impact

#### `FinancialImpactFrame`
Financial impact assessment:
- **Cost metrics**: Total costs, cost per unit, budget tracking
- **Savings metrics**: Achieved savings, savings rate
- **Revenue metrics**: Revenue generation, growth rates
- **ROI calculation**: Investment returns and profitability

#### `CostMetrics`
Cost-related metrics:
```python
CostMetrics(
    total_cost=15000.0,
    cost_per_unit=1.25,
    budget=20000.0
)
```

### ROI Analysis

#### `ROIFrame`
Return on investment analysis:
- **Cost-benefit analysis**: Benefits vs costs comparison
- **Investment metrics**: Initial investment, return rates
- **Profitability assessment**: ROI calculations and viability

#### `CostBenefitAnalysis`
Cost-benefit analysis:
```python
CostBenefitAnalysis(
    total_benefits=75000.0,
    total_costs=25000.0,
    internal_rate_of_return=0.18
)
```

## Validation and Compatibility

### Backward Compatibility

The package includes comprehensive backward compatibility validation:

```python
from pynomaly.schemas.validation import (
    BackwardCompatibilityValidator,
    validate_schema_compatibility,
    ensure_backward_compatibility
)

# Validate schema compatibility
validator = BackwardCompatibilityValidator()
is_compatible = validate_schema_compatibility(old_schema, new_schema)

# Test with historical data
ensure_backward_compatibility(AnomalyKPIFrame, historical_data)
```

### Schema Versioning

Version management with semantic versioning:

```python
from pynomaly.schemas.versioning import (
    SchemaVersion,
    is_compatible_version,
    compare_versions
)

# Create version objects
version = SchemaVersion("1.2.3")
print(f"Major: {version.MAJOR}, Minor: {version.MINOR}, Patch: {version.PATCH}")

# Check compatibility
is_compatible = is_compatible_version("1.2.3", "1.5.0")  # True
is_compatible = is_compatible_version("1.2.3", "2.0.0")  # False
```

## Usage Examples

### Basic Metric Creation

```python
from datetime import datetime
from pynomaly.schemas.analytics.base import MetricFrame

metric = MetricFrame(
    metric_id="cpu_usage_001",
    name="CPU Usage",
    value=75.5,
    timestamp=datetime.utcnow()
)
```

### Anomaly KPI Monitoring

```python
from pynomaly.schemas.analytics.anomaly_kpis import (
    AnomalyKPIFrame,
    AnomalyDetectionMetrics,
    AnomalyClassificationMetrics
)

# Create detection metrics
detection_metrics = AnomalyDetectionMetrics(
    accuracy=0.95,
    precision=0.92,
    recall=0.88,
    f1_score=0.9,
    false_positive_rate=0.05,
    false_negative_rate=0.12
)

# Create classification metrics
classification_metrics = AnomalyClassificationMetrics(
    true_positives=150,
    false_positives=8,
    true_negatives=1842,
    false_negatives=20,
    anomalies_detected=158,
    anomalies_confirmed=150,
    anomalies_dismissed=8
)

# Create KPI frame
kpi_frame = AnomalyKPIFrame(
    metric_id="anomaly_kpi_001",
    name="Production Anomaly Detection",
    value=95.0,
    timestamp=datetime.utcnow(),
    detection_metrics=detection_metrics,
    classification_metrics=classification_metrics,
    model_name="IsolationForest",
    model_version="1.2.3",
    dataset_id="production_001",
    throughput=1250.0,
    cpu_usage=65.5,
    memory_usage=2048.0,
    active_alerts=3,
    critical_alerts=1,
    confidence_score=0.92,
    data_quality_score=0.88
)

# Check system health
print(f"System is healthy: {kpi_frame.is_healthy()}")
print(f"Anomaly rate: {kpi_frame.get_anomaly_rate():.3f}")
```

### System Health Monitoring

```python
from pynomaly.schemas.analytics.system_health import (
    SystemHealthFrame,
    SystemResourceMetrics,
    SystemPerformanceMetrics,
    SystemStatusMetrics,
    SystemStatus
)

# Create resource metrics
resource_metrics = SystemResourceMetrics(
    cpu_usage_percent=45.2,
    cpu_load_average=1.8,
    cpu_cores=8,
    memory_usage_percent=68.5,
    memory_used_mb=5500.0,
    memory_total_mb=8192.0,
    disk_usage_percent=72.3,
    disk_used_gb=362.0,
    disk_total_gb=500.0,
    # ... other metrics
)

# Create performance metrics
performance_metrics = SystemPerformanceMetrics(
    avg_response_time_ms=125.5,
    p95_response_time_ms=280.0,
    p99_response_time_ms=450.0,
    requests_per_second=850.0,
    error_rate=0.025,
    # ... other metrics
)

# Create status metrics
status_metrics = SystemStatusMetrics(
    system_status=SystemStatus.HEALTHY,
    uptime_seconds=2678400.0,
    services_total=15,
    services_healthy=14,
    active_alerts=2,
    critical_alerts=0,
    # ... other metrics
)

# Create health frame
health_frame = SystemHealthFrame(
    metric_id="system_health_001",
    name="Production System Health",
    value=0.92,
    timestamp=datetime.utcnow(),
    resource_metrics=resource_metrics,
    performance_metrics=performance_metrics,
    status_metrics=status_metrics,
    hostname="prod-server-01",
    environment="production",
    overall_health_score=0.92,
    availability_score=0.995,
    reliability_score=0.98,
    capacity_utilization=0.68
)

# Check system health
print(f"System is healthy: {health_frame.is_healthy()}")
print(f"Needs attention: {health_frame.needs_attention()}")
```

### Financial Impact Analysis

```python
from pynomaly.schemas.analytics.financial_impact import (
    FinancialImpactFrame,
    CostMetrics,
    SavingsMetrics,
    RevenueMetrics,
    ROICalculation
)

# Create financial metrics
cost_metrics = CostMetrics(
    total_cost=15000.0,
    cost_per_unit=1.25,
    budget=20000.0
)

savings_metrics = SavingsMetrics(
    total_savings=5000.0,
    savings_rate=0.25
)

revenue_metrics = RevenueMetrics(
    total_revenue=50000.0,
    revenue_per_unit=4.15,
    revenue_growth_rate=0.15
)

roi_calculation = ROICalculation(
    investment=15000.0,
    returns=55000.0
)

# Create financial impact frame
impact_frame = FinancialImpactFrame(
    metric_id="financial_impact_001",
    name="Q4 Financial Impact",
    value=0.75,
    timestamp=datetime.utcnow(),
    cost_metrics=cost_metrics,
    savings_metrics=savings_metrics,
    revenue_metrics=revenue_metrics,
    roi_calculation=roi_calculation
)

# Analyze profitability
print(f"ROI: {impact_frame.roi_calculation.roi:.2%}")
print(f"Is profitable: {impact_frame.roi_calculation.is_profitable()}")
print(f"Total benefits: ${impact_frame.total_benefits:,.2f}")
```

## API Integration

All schemas are designed for seamless API integration:

```python
from fastapi import FastAPI
from pynomaly.schemas.analytics import AnomalyKPIFrame

app = FastAPI()

@app.post("/metrics/anomaly-kpis")
async def create_anomaly_kpi(kpi_frame: AnomalyKPIFrame):
    # Validate and process the KPI frame
    if kpi_frame.is_healthy():
        return {"status": "healthy", "data": kpi_frame.dict()}
    else:
        return {"status": "unhealthy", "data": kpi_frame.dict()}

@app.get("/metrics/anomaly-kpis/{metric_id}")
async def get_anomaly_kpi(metric_id: str) -> AnomalyKPIFrame:
    # Retrieve and return KPI frame
    pass
```

## Testing

The package includes comprehensive testing utilities:

```bash
# Run direct tests
python test_schemas_direct.py

# Run validation tests
python -c "
from pynomaly.schemas.validation import ensure_backward_compatibility
from pynomaly.schemas.analytics import AnomalyKPIFrame
ensure_backward_compatibility(AnomalyKPIFrame, historical_data)
"
```

## Best Practices

1. **Use semantic versioning** for schema changes
2. **Validate backward compatibility** before releasing new versions
3. **Include comprehensive metadata** for traceability
4. **Monitor schema performance** in production
5. **Test with real data** during development
6. **Document breaking changes** clearly

## Version History

- **v1.0.0**: Initial release with core schemas
  - Base metric frames
  - Anomaly KPI schemas
  - System health monitoring
  - Financial impact assessment
  - ROI analysis
  - Validation utilities
  - Version management

## License

This package is part of the Pynomaly project and follows the same MIT license.
