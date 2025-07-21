# Data Quality Package

A comprehensive data quality assessment, validation, and monitoring framework for ensuring data reliability and trustworthiness.

## Overview

This package provides advanced data quality capabilities including automated profiling, validation rule engines, quality scoring, and continuous monitoring to maintain high-quality data across the platform.

## Features

- **Quality Profiling**: Automated data profiling with statistical analysis and pattern detection
- **Validation Engine**: Flexible, rule-based validation framework with custom rule support
- **Quality Scoring**: Comprehensive data quality scoring with configurable weightings
- **Anomaly Detection**: Statistical anomaly detection for data quality issues
- **Monitoring & Alerting**: Continuous quality monitoring with real-time alerts
- **Issue Remediation**: Automated data cleaning and quality improvement suggestions
- **Compliance Checking**: Data compliance validation against regulatory standards
- **Quality Lineage**: Track quality metrics across data transformation pipelines

## Architecture

This package follows clean architecture principles with clear domain boundaries:

```
data_quality/
├── domain/                 # Core data quality business logic
│   ├── entities/          # Quality entities (Report, Rule, Score, Issue)
│   ├── services/          # Quality assessment and validation services
│   └── value_objects/     # Quality metrics and thresholds
├── application/           # Use cases and orchestration  
│   ├── services/          # Application services
│   ├── use_cases/         # Quality assessment workflows
│   └── dto/               # Data transfer objects
├── infrastructure/        # External integrations and persistence
│   ├── repositories/      # Quality data storage
│   ├── adapters/          # External validation services
│   └── monitoring/        # Quality monitoring infrastructure
└── presentation/          # APIs and interfaces
    ├── api/               # REST API endpoints
    ├── cli/               # Command-line interface
    └── dashboards/        # Quality dashboards
```

## Quick Start

```python
from src.packages.data.data_quality.application.services import DataQualityService
from src.packages.data.data_quality.domain.entities import ValidationRule

# Initialize data quality service
quality_service = DataQualityService()

# Assess data quality
quality_report = quality_service.assess_data_quality(
    dataset=dataset,
    rules=[
        ValidationRule.not_null("customer_id"),
        ValidationRule.unique("email"),
        ValidationRule.range("age", min_value=0, max_value=120),
        ValidationRule.pattern("phone", r"^\+?[\d\s\-\(\)]+$")
    ]
)

# Check results
print(f"Overall quality score: {quality_report.overall_score}")
print(f"Issues found: {len(quality_report.issues)}")

# Generate quality profile
profile = quality_service.profile_dataset(dataset)
print(f"Dataset statistics: {profile.statistics}")

# Set up continuous monitoring
monitor = quality_service.create_monitor(
    dataset_id="customer_data",
    schedule="hourly",
    alert_threshold=0.8
)
```

## Core Components

### Quality Assessment
```python
from src.packages.data.data_quality.domain.services import QualityAssessor

assessor = QualityAssessor()

# Comprehensive quality assessment
assessment = assessor.assess(
    data=dataset,
    dimensions=["completeness", "accuracy", "consistency", "validity", "uniqueness"]
)

# Custom quality rules
custom_rules = [
    ValidationRule.custom("revenue_positive", lambda x: x["revenue"] > 0),
    ValidationRule.cross_field("start_end_dates", lambda row: row["end_date"] > row["start_date"])
]
```

### Data Profiling
```python
from src.packages.data.data_quality.domain.services import DataProfiler

profiler = DataProfiler()

# Generate comprehensive profile
profile = profiler.profile(dataset)

# Access profile statistics
print(f"Row count: {profile.row_count}")
print(f"Column count: {profile.column_count}")
print(f"Missing values: {profile.missing_percentage}")
print(f"Data types: {profile.column_types}")
print(f"Unique values: {profile.uniqueness_stats}")
```

### Quality Monitoring
```python
from src.packages.data.data_quality.application.services import QualityMonitor

# Set up monitoring
monitor = QualityMonitor(
    dataset_id="sales_data",
    quality_rules=validation_rules,
    monitoring_schedule="*/15 * * * *",  # Every 15 minutes
    alert_channels=["email", "slack"]
)

# Start monitoring
await monitor.start()

# Check monitoring status
status = monitor.get_status()
print(f"Monitoring active: {status.is_active}")
print(f"Last check: {status.last_check_time}")
```

## Installation

```bash
# Install from package directory
cd src/packages/data/data_quality
pip install -e .

# Install with monitoring dependencies
pip install -e ".[monitoring,alerts]"
```

## Testing

```bash
# Run package tests
pytest tests/

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/quality/        # Quality validation tests
```

## Use Cases

- **Data Ingestion Validation**: Validate incoming data quality before processing
- **Pipeline Quality Gates**: Quality checkpoints in data transformation pipelines
- **Compliance Monitoring**: Ensure data meets regulatory and business requirements
- **Data Quality Dashboards**: Real-time visibility into data quality metrics
- **Anomaly Detection**: Detect unusual patterns that may indicate quality issues
- **Data Certification**: Certify datasets for downstream consumption

## Integration

This package integrates seamlessly with other data domain packages:

```python
# With data engineering for pipeline quality gates
from src.packages.data.data_engineering.application.services import PipelineService

pipeline = PipelineService()
pipeline.add_quality_gate(
    stage="after_transformation",
    quality_rules=validation_rules,
    fail_on_threshold=0.85
)

# With data observability for quality lineage
from src.packages.data.observability.application.services import LineageService

lineage = LineageService()
lineage.track_quality_metrics(quality_report, dataset_id="customer_data")
```

## Configuration

```yaml
# quality_config.yaml
data_quality:
  profiling:
    sample_size: 10000
    profile_depth: "comprehensive"  # basic, standard, comprehensive
    compute_correlations: true
  
  validation:
    fail_fast: false
    parallel_validation: true
    max_workers: 4
  
  monitoring:
    default_schedule: "0 */6 * * *"  # Every 6 hours
    retention_days: 90
    alert_thresholds:
      warning: 0.9
      critical: 0.8
  
  scoring:
    weights:
      completeness: 0.25
      accuracy: 0.25
      consistency: 0.20
      validity: 0.15
      uniqueness: 0.15
```

## Quality Dimensions

The package supports comprehensive quality assessment across multiple dimensions:

- **Completeness**: Measure of missing or null values
- **Accuracy**: Correctness of data values
- **Consistency**: Consistency across related data elements
- **Validity**: Conformance to defined formats and constraints
- **Uniqueness**: Identification of duplicate records
- **Timeliness**: Freshness and currency of data
- **Relevance**: Appropriateness for intended use

## Performance

Optimized for large-scale data quality assessment:

- **Sampling Strategies**: Intelligent sampling for large datasets
- **Parallel Processing**: Multi-threaded validation execution
- **Incremental Assessment**: Quality assessment on data changes only
- **Caching**: Rule result caching for repeated assessments
- **Memory Optimization**: Efficient memory usage for large datasets

## Documentation

See the [docs/](docs/) directory for:
- [Data Quality Guide](docs/quality_guide.md)
- [Validation Rules Reference](docs/validation_rules.md)
- [Monitoring Setup](docs/monitoring_setup.md)
- [API Reference](docs/api_reference.md)

## License

MIT License