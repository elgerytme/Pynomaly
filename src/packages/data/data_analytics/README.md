# Data Analytics Package

A comprehensive data analytics and business intelligence package for generating insights from complex datasets.

## Overview

This package provides advanced data analytics capabilities including statistical analysis, business intelligence, predictive analytics, and automated reporting systems.

## Features

- **Statistical Analysis**: Advanced statistical analysis, hypothesis testing, and modeling
- **Business Intelligence**: Comprehensive BI reporting, KPI tracking, and dashboard creation
- **Predictive Analytics**: Machine learning-based forecasting and trend analysis
- **Time Series Analysis**: Specialized time series analysis and forecasting
- **Custom Metrics**: Flexible custom metrics calculation and aggregation
- **Real-time Analytics**: Streaming analytics for real-time insights
- **Data Visualization**: Interactive charts, graphs, and dashboard components
- **Report Generation**: Automated report generation and distribution

## Architecture

This package follows clean architecture principles with clear domain boundaries:

```
data_analytics/
├── domain/                 # Core analytics business logic
│   ├── entities/          # Analytics entities (Reports, Metrics, KPIs)
│   ├── services/          # Analytics domain services
│   └── value_objects/     # Analytics value objects
├── application/           # Use cases and orchestration  
│   ├── services/          # Application services
│   ├── use_cases/         # Analytics use cases
│   └── dto/               # Data transfer objects
├── infrastructure/        # External integrations and persistence
│   ├── repositories/      # Data storage implementations
│   ├── adapters/          # External service adapters
│   └── config/            # Configuration management
└── presentation/          # APIs and interfaces
    ├── api/               # REST API endpoints
    ├── cli/               # Command-line interface
    └── dashboards/        # Dashboard components
```

## Quick Start

```python
from src.packages.data.data_analytics.application.services import AnalyticsService
from src.packages.data.data_analytics.domain.entities import AnalyticsQuery

# Initialize analytics service
analytics = AnalyticsService()

# Create analytics query
query = AnalyticsQuery(
    dataset="sales_data",
    metrics=["revenue", "growth_rate", "conversion_rate"],
    dimensions=["region", "product_category"],
    filters={"date_range": "last_30_days"}
)

# Generate analytics report
report = analytics.generate_report(query)
print(f"Total revenue: {report.metrics['revenue']}")

# Create real-time dashboard
dashboard = analytics.create_dashboard(
    name="Sales Performance",
    widgets=["revenue_trend", "top_products", "regional_performance"]
)
```

## Installation

```bash
# Install from package directory
cd src/packages/data/data_analytics
pip install -e .

# Install with optional dependencies
pip install -e ".[visualization,ml]"
```

## Testing

```bash
# Run package tests
pytest tests/

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/performance/    # Performance tests
```

## Use Cases

- **Business Intelligence**: KPI tracking, performance monitoring, and executive reporting
- **Sales Analytics**: Revenue analysis, conversion tracking, and sales forecasting
- **Customer Analytics**: Customer behavior analysis, segmentation, and lifetime value
- **Operational Analytics**: Process optimization, efficiency metrics, and operational insights
- **Financial Analytics**: Financial reporting, budgeting, and variance analysis

## Integration

This package integrates with other data domain packages:

```python
# With data quality for validated analytics
from src.packages.data.data_quality.application.services import DataQualityService

quality_service = DataQualityService()
quality_report = quality_service.validate_dataset(dataset)

if quality_report.is_valid:
    analytics_report = analytics.generate_report(query)

# With data visualization for enhanced reporting
from src.packages.data.data_visualization.application.services import VisualizationService

viz_service = VisualizationService()
dashboard = viz_service.create_dashboard(analytics_report)
```

## Configuration

```yaml
# analytics_config.yaml
analytics:
  computation:
    engine: "pandas"  # pandas, dask, spark
    parallel_processing: true
    memory_limit: "8GB"
  
  reporting:
    output_format: "html"  # html, pdf, excel
    auto_refresh: true
    refresh_interval: "1h"
  
  caching:
    enabled: true
    ttl: 3600
    backend: "redis"
  
  visualization:
    default_theme: "corporate"
    interactive: true
    export_formats: ["png", "svg", "pdf"]
```

## Documentation

See the [docs/](docs/) directory for:
- [Analytics Guide](docs/analytics_guide.md)
- [API Reference](docs/api_reference.md)
- [Visualization Cookbook](docs/visualization_cookbook.md)
- [Performance Tuning](docs/performance_tuning.md)

## License

MIT License