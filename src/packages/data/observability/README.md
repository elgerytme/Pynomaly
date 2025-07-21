# Data Observability Package

The Data Observability Package provides comprehensive monitoring, tracking, and quality assurance capabilities for data assets within the Monorepo ecosystem. This package implements four core components that work together to provide complete visibility into your data infrastructure.

## Overview

Data observability is crucial for maintaining high-quality data pipelines and ensuring reliable analytics. This package provides:

- **Data Lineage Tracking**: Understand data flow and dependencies
- **Pipeline Health Monitoring**: Real-time monitoring of pipeline execution
- **Data Catalog Management**: Centralized metadata and asset discovery
- **Predictive Quality Monitoring**: Proactive quality issue detection

## Architecture

The package follows Domain-Driven Design principles with clear separation of concerns:

```
src/packages/data_observability/
├── domain/
│   └── entities/           # Core domain models
├── application/
│   ├── services/          # Application services
│   └── facades/           # Unified interfaces
├── infrastructure/
│   └── di/                # Dependency injection
└── README.md
```

## Core Components

### 1. Data Lineage Tracking System

**Purpose**: Track data transformations and dependencies across your data ecosystem.

**Key Features**:
- Track data transformation relationships
- Perform impact analysis for changes
- Visualize data flow graphs
- Find paths between data assets

**Usage Example**:
```python
from anomaly_detection.packages.data_observability import DataObservabilityFacade

# Track a data transformation
facade.track_data_transformation(
    source_id=source_asset_id,
    target_id=target_asset_id,
    transformation_type="aggregation",
    transformation_details={
        "aggregation_type": "sum",
        "group_by": ["customer_id", "date"]
    }
)

# Analyze impact of changes
impact = facade.analyze_impact(asset_id, "downstream")
print(f"Changes will affect {len(impact['affected_nodes'])} downstream assets")
```

### 2. Pipeline Health Monitoring

**Purpose**: Monitor the health and performance of data pipelines in real-time.

**Key Features**:
- Real-time health score calculation
- Automated alerting for issues
- Historical health tracking
- Performance metrics monitoring

**Usage Example**:
```python
# Monitor pipeline health
health = facade.monitor_pipeline_health(
    pipeline_id=pipeline_id,
    metrics={
        "execution_time_ms": 45000,
        "memory_usage_mb": 512,
        "rows_processed": 1000000,
        "error_rate": 0.001
    }
)

# Get health summary
summary = facade.get_pipeline_health_summary(pipeline_id)
print(f"Pipeline health score: {summary['health_score']}")
```

### 3. Data Catalog with Smart Discovery

**Purpose**: Centralized catalog for data asset metadata with intelligent discovery capabilities.

**Key Features**:
- Asset registration and metadata management
- Intelligent search and discovery
- Usage tracking and analytics
- Business glossary integration
- Auto-classification and tagging

**Usage Example**:
```python
# Register a data asset
asset = facade.register_data_asset(
    name="customer_transactions",
    asset_type=DataAssetType.TABLE,
    location="s3://data-lake/customer_transactions/",
    data_format=DataFormat.PARQUET,
    description="Customer transaction data from payment systems",
    owner="data-team@company.com",
    domain="finance"
)

# Discover similar assets
similar = facade.get_asset_recommendations(asset.id)
```

### 4. Predictive Data Quality Service

**Purpose**: Proactive monitoring and prediction of data quality issues.

**Key Features**:
- Multiple prediction models (linear regression, exponential smoothing, seasonal decomposition)
- Quality trend analysis
- Anomaly detection
- Forecasting capabilities
- Automated alerting

**Usage Example**:
```python
# Add quality metrics
facade.add_quality_metric(
    asset_id=asset_id,
    metric_type="completeness",
    value=0.95,
    timestamp=datetime.utcnow()
)

# Predict quality issues
prediction = facade.predict_quality_issues(
    asset_id=asset_id,
    prediction_type=PredictionType.QUALITY_DEGRADATION,
    target_time=datetime.utcnow() + timedelta(hours=24)
)

# Forecast quality metrics
forecast = facade.forecast_quality_metrics(
    asset_id=asset_id,
    metric_type="completeness",
    horizon_hours=48
)
```

## Getting Started

### Installation

The Data Observability Package is included as part of the anomaly detection framework. Ensure you have the main framework installed:

```bash
pip install anomaly_detection
```

### Basic Setup

```python
from anomaly_detection.packages.data_observability.infrastructure.di.container import DataObservabilityContainer

# Initialize the container
container = DataObservabilityContainer()

# Get the facade
facade = container.observability_facade()
```

### Configuration

The package can be configured through environment variables or configuration files:

```python
# Configure through container
container.config.from_dict({
    "quality_prediction": {
        "default_model": "exponential_smoothing",
        "prediction_horizon_hours": 24,
        "confidence_threshold": 0.7
    },
    "pipeline_health": {
        "alert_threshold": 0.8,
        "metric_retention_days": 30
    }
})
```

## Advanced Features

### Comprehensive Asset View

Get a complete view of an asset across all observability dimensions:

```python
asset_view = facade.get_comprehensive_asset_view(asset_id)
print(f"Asset has {asset_view['lineage']['total_connected_assets']} connected assets")
print(f"Quality score: {asset_view['quality']['quality_score']}")
print(f"Active alerts: {asset_view['quality']['active_alerts']}")
```

### Data Health Dashboard

Get organization-wide data health metrics:

```python
dashboard = facade.get_data_health_dashboard()
print(f"Total assets: {dashboard['catalog']['total_assets']}")
print(f"Healthy pipelines: {dashboard['pipeline_health']['healthy_pipelines']}")
print(f"Active quality alerts: {dashboard['quality_predictions']['total_active_alerts']}")
```

### Data Issue Investigation

Investigate data issues across all observability dimensions:

```python
investigation = facade.investigate_data_issue(
    asset_id=asset_id,
    issue_type="quality_degradation",
    severity="high"
)

print(f"Investigation found {len(investigation['findings'])} issues")
for finding in investigation['findings']:
    print(f"- {finding['category']}: {finding['finding']}")
```

## API Reference

### Core Facades

- `DataObservabilityFacade`: Main entry point for all observability operations

### Services

- `DataLineageService`: Lineage tracking and impact analysis
- `PipelineHealthService`: Pipeline health monitoring
- `DataCatalogService`: Data catalog management
- `PredictiveQualityService`: Quality prediction and forecasting

### Domain Entities

- `DataLineage`, `LineageNode`, `LineageEdge`: Lineage modeling
- `PipelineHealth`, `PipelineMetric`, `PipelineAlert`: Health monitoring
- `DataCatalogEntry`, `DataSchema`, `DataUsage`: Catalog management
- `QualityPrediction`, `QualityForecast`, `QualityTrend`: Quality monitoring

## Best Practices

### Data Lineage
- Track transformations at appropriate granularity
- Use meaningful transformation types and descriptions
- Regularly analyze impact before making changes
- Maintain up-to-date lineage graphs

### Pipeline Health
- Monitor key metrics consistently
- Set appropriate alert thresholds
- Investigate health degradations promptly
- Track metrics over time for trend analysis

### Data Catalog
- Provide comprehensive metadata
- Use consistent naming conventions
- Regularly review and update asset information
- Leverage auto-classification features

### Quality Prediction
- Feed quality metrics regularly
- Validate predictions against actual outcomes
- Adjust prediction models based on accuracy
- Act on quality alerts proactively

## Integration with anomaly_detection

The Data Observability Package integrates seamlessly with other anomaly_detection components:

- **Anomaly Detection**: Quality predictions can trigger anomaly detection workflows
- **Processing Pipelines**: Pipeline health monitoring supports model training pipelines
- **Data Processing**: Lineage tracking captures data transformation steps
- **Monitoring**: Health metrics integrate with overall system monitoring

## Performance Considerations

- **Scalability**: Services are designed to handle large numbers of assets and metrics
- **Storage**: Historical data is automatically pruned based on retention policies
- **Indexing**: Search operations are optimized with appropriate indexing
- **Caching**: Frequently accessed data is cached for performance

## Troubleshooting

### Common Issues

1. **Memory Usage**: Large lineage graphs may consume significant memory
   - Solution: Limit graph depth and implement pagination

2. **Performance**: Complex similarity calculations may be slow
   - Solution: Implement caching and optimize similarity algorithms

3. **Prediction Accuracy**: Quality predictions may have low accuracy initially
   - Solution: Provide more training data and validate predictions regularly

### Monitoring

Enable logging to monitor package performance:

```python
import logging
logging.getLogger('anomaly_detection.packages.data_observability').setLevel(logging.INFO)
```

## Contributing

To contribute to the Data Observability Package:

1. Follow the existing architecture patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure compatibility with existing integrations

## License

This package is part of the anomaly detection framework and follows the same licensing terms.