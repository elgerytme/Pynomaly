# Data Quality Package User Guide

## Overview

The Data Quality package provides comprehensive data validation, quality monitoring, and automated data cleansing capabilities. It helps ensure your data meets quality standards before anomaly detection analysis.

## Quick Start

### Installation

```bash
pip install pynomaly[standard]
```

### Basic Usage

```python
from pynomaly.packages.data_quality import DataQualityEngine
import pandas as pd

# Load your dataset
data = pd.read_csv('your_data.csv')

# Create quality engine
quality_engine = DataQualityEngine()

# Run quality assessment
quality_report = quality_engine.assess_quality(data)

# View results
print(f"Overall quality score: {quality_report.overall_score}")
print(f"Quality issues found: {quality_report.issues_count}")
```

## Core Features

### 1. Data Validation

#### Rule-Based Validation

```python
from pynomaly.packages.data_quality import ValidationRule, ValidationEngine

# Define validation rules
rules = [
    ValidationRule.not_null('user_id'),
    ValidationRule.range_check('age', min_val=0, max_val=120),
    ValidationRule.format_check('email', pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$'),
    ValidationRule.uniqueness_check('user_id')
]

# Create validation engine
validator = ValidationEngine(rules=rules)

# Validate data
validation_result = validator.validate(data)

# Check results
if validation_result.is_valid:
    print("Data passed all validation rules")
else:
    print(f"Validation failed: {validation_result.failed_rules}")
```

#### Custom Validation Rules

```python
from pynomaly.packages.data_quality import CustomValidationRule

class BusinessDateRule(CustomValidationRule):
    def validate(self, data):
        # Custom business logic
        invalid_dates = data[data['date'] > pd.Timestamp.now()]
        return ValidationResult(
            is_valid=len(invalid_dates) == 0,
            error_count=len(invalid_dates),
            error_details=invalid_dates.index.tolist()
        )

# Use custom rule
validator.add_rule(BusinessDateRule())
```

### 2. Quality Monitoring

#### Real-time Quality Monitoring

```python
from pynomaly.packages.data_quality import QualityMonitor

# Create quality monitor
monitor = QualityMonitor(
    check_interval='5min',
    alert_threshold=0.8,
    enable_alerts=True
)

# Start monitoring
monitor.start_monitoring(data_source)

# Get real-time quality metrics
metrics = monitor.get_current_metrics()
print(f"Current quality score: {metrics.quality_score}")
```

#### Quality Governance Framework

```python
from pynomaly.packages.data_quality import QualityGovernance

# Define quality standards
governance = QualityGovernance()
governance.set_quality_standard('completeness', min_threshold=0.95)
governance.set_quality_standard('accuracy', min_threshold=0.90)
governance.set_quality_standard('consistency', min_threshold=0.85)

# Apply governance
compliance_report = governance.assess_compliance(data)
```

### 3. Data Cleansing

#### Automated Data Cleaning

```python
from pynomaly.packages.data_quality import DataCleansingEngine

# Create cleansing engine
cleanser = DataCleansingEngine()

# Configure cleaning operations
cleanser.configure_cleaning(
    handle_missing_values='auto',
    normalize_formats=True,
    remove_duplicates=True,
    fix_data_types=True
)

# Clean the data
cleaned_data = cleanser.clean_data(data)

# Get cleaning report
cleaning_report = cleanser.get_cleaning_report()
print(f"Rows cleaned: {cleaning_report.rows_affected}")
print(f"Operations performed: {cleaning_report.operations}")
```

#### Advanced Cleansing Options

```python
# Custom cleansing configuration
cleansing_config = {
    'missing_values': {
        'strategy': 'advanced_imputation',
        'numerical_method': 'knn',
        'categorical_method': 'mode',
        'k_neighbors': 5
    },
    'outliers': {
        'detection_method': 'isolation_forest',
        'treatment': 'cap',  # or 'remove', 'flag'
        'contamination': 0.1
    },
    'duplicates': {
        'strategy': 'keep_first',
        'similarity_threshold': 0.95
    }
}

cleanser = DataCleansingEngine(config=cleansing_config)
```

## Advanced Features

### 1. Self-Healing Data Pipelines

```python
from pynomaly.packages.data_quality import SelfHealingPipeline

# Create self-healing pipeline
pipeline = SelfHealingPipeline(
    auto_fix_threshold=0.8,
    escalation_threshold=0.6,
    learning_enabled=True
)

# Process data through pipeline
result = pipeline.process(data)

if result.auto_fixed:
    print(f"Pipeline auto-fixed {result.fixes_applied} issues")
    cleaned_data = result.data
else:
    print("Manual intervention required")
    issues = result.unresolved_issues
```

### 2. Data Lineage and Impact Tracking

```python
from pynomaly.packages.data_quality import DataLineageTracker

# Track data lineage
lineage_tracker = DataLineageTracker()

# Register data sources
lineage_tracker.register_source('raw_data', source_type='csv')
lineage_tracker.register_transformation('cleaning', inputs=['raw_data'])
lineage_tracker.register_output('clean_data', source='cleaning')

# Track quality impact
impact_analysis = lineage_tracker.analyze_quality_impact(
    source='raw_data',
    quality_degradation=0.15
)

print(f"Downstream systems affected: {impact_analysis.affected_systems}")
```

### 3. Executive Reporting and Analytics

```python
from pynomaly.packages.data_quality import ExecutiveReporting

# Generate executive reports
reporter = ExecutiveReporting()

# Create quality dashboard
dashboard = reporter.create_quality_dashboard(
    data_sources=['sales', 'customers', 'products'],
    time_period='last_30_days'
)

# Generate KPIs
quality_kpis = reporter.generate_quality_kpis()
print(f"Overall data quality: {quality_kpis.overall_score}")
print(f"Trend: {quality_kpis.trend_direction}")
```

## Quality Metrics

### Core Quality Dimensions

```python
# Completeness - measure of missing data
completeness = quality_engine.measure_completeness(data)
print(f"Data completeness: {completeness.score * 100}%")

# Accuracy - measure of correct data
accuracy = quality_engine.measure_accuracy(data, reference_data)
print(f"Data accuracy: {accuracy.score * 100}%")

# Consistency - measure of data uniformity
consistency = quality_engine.measure_consistency(data)
print(f"Data consistency: {consistency.score * 100}%")

# Validity - measure of data format compliance
validity = quality_engine.measure_validity(data)
print(f"Data validity: {validity.score * 100}%")

# Uniqueness - measure of duplicate data
uniqueness = quality_engine.measure_uniqueness(data)
print(f"Data uniqueness: {uniqueness.score * 100}%")
```

### Custom Quality Metrics

```python
from pynomaly.packages.data_quality import CustomQualityMetric

class BusinessRelevanceMetric(CustomQualityMetric):
    def calculate(self, data):
        # Custom business logic for relevance
        relevant_records = self.assess_business_relevance(data)
        return relevant_records / len(data)

# Register custom metric
quality_engine.register_metric('business_relevance', BusinessRelevanceMetric())
```

## Integration Patterns

### Integration with Data Profiling

```python
from pynomaly.packages.data_profiling import DataProfiler
from pynomaly.packages.data_quality import DataQualityEngine

# Profile data first
profiler = DataProfiler()
profile = profiler.profile_dataset(data)

# Use profile insights for quality assessment
quality_engine = DataQualityEngine()
quality_engine.configure_from_profile(profile)

# Enhanced quality assessment
quality_report = quality_engine.assess_quality(data)
```

### Integration with Anomaly Detection

```python
from pynomaly import AnomalyDetector

# Ensure data quality before detection
quality_score = quality_engine.assess_quality(data).overall_score

if quality_score >= 0.8:
    # Proceed with anomaly detection
    detector = AnomalyDetector()
    results = detector.detect(cleaned_data)
else:
    print("Data quality too low for reliable anomaly detection")
    # Implement data improvement workflow
```

## Configuration Management

### Quality Policies

```python
from pynomaly.packages.data_quality import QualityPolicy

# Define quality policy
policy = QualityPolicy()
policy.set_minimum_completeness(0.95)
policy.set_maximum_error_rate(0.05)
policy.set_required_formats({
    'email': r'^[\w\.-]+@[\w\.-]+\.\w+$',
    'phone': r'^\+?1?\d{9,15}$'
})

# Apply policy
compliance = quality_engine.check_policy_compliance(data, policy)
```

### Environment-Specific Configuration

```python
# Development environment
dev_config = QualityConfig(
    strict_validation=False,
    auto_fix_enabled=True,
    alert_level='warning'
)

# Production environment
prod_config = QualityConfig(
    strict_validation=True,
    auto_fix_enabled=False,
    alert_level='error',
    audit_logging=True
)
```

## Troubleshooting

### Common Quality Issues

1. **High Missing Value Rate**
   ```python
   # Analyze missing value patterns
   missing_analysis = quality_engine.analyze_missing_patterns(data)
   print(f"Missing patterns: {missing_analysis.patterns}")
   
   # Implement targeted imputation
   imputer = quality_engine.create_targeted_imputer(missing_analysis)
   ```

2. **Data Type Inconsistencies**
   ```python
   # Detect type inconsistencies
   type_issues = quality_engine.detect_type_issues(data)
   
   # Auto-fix type issues
   fixed_data = quality_engine.fix_type_issues(data, type_issues)
   ```

3. **Duplicate Records**
   ```python
   # Advanced duplicate detection
   duplicates = quality_engine.find_fuzzy_duplicates(
       data, 
       similarity_threshold=0.85
   )
   
   # Resolve duplicates with business rules
   resolved_data = quality_engine.resolve_duplicates(
       data, 
       duplicates, 
       strategy='most_complete'
   )
   ```

## Performance Optimization

### Large Dataset Processing

```python
# Configure for large datasets
quality_engine.configure_processing(
    chunk_size=50000,
    parallel_processing=True,
    memory_efficient=True
)

# Use sampling for initial assessment
sample_quality = quality_engine.assess_quality_sample(
    data, 
    sample_size=10000,
    confidence_level=0.95
)
```

### Caching and Persistence

```python
# Enable result caching
quality_engine.enable_caching(
    cache_location='/tmp/quality_cache',
    cache_ttl='1hour'
)

# Persist quality rules
quality_engine.save_rules('quality_rules.json')
```

## API Reference

### Core Classes

- `DataQualityEngine`: Main quality assessment engine
- `ValidationEngine`: Data validation framework
- `DataCleansingEngine`: Automated data cleaning
- `QualityMonitor`: Real-time quality monitoring
- `QualityGovernance`: Governance framework

### Key Methods

- `assess_quality(data)`: Comprehensive quality assessment
- `validate(data, rules)`: Rule-based validation
- `clean_data(data)`: Automated data cleaning
- `monitor_quality(data_stream)`: Real-time monitoring

## Complete Example

```python
import pandas as pd
from pynomaly.packages.data_quality import (
    DataQualityEngine, 
    ValidationRule, 
    DataCleansingEngine
)

# Load data
data = pd.read_csv('customer_data.csv')

# Create quality engine
quality_engine = DataQualityEngine()

# Define validation rules
rules = [
    ValidationRule.not_null('customer_id'),
    ValidationRule.range_check('age', min_val=0, max_val=120),
    ValidationRule.format_check('email', pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$'),
    ValidationRule.uniqueness_check('customer_id')
]

# Validate data
validation_result = quality_engine.validate(data, rules)

if not validation_result.is_valid:
    print(f"Validation issues found: {validation_result.failed_rules}")
    
    # Clean the data
    cleanser = DataCleansingEngine()
    cleaned_data = cleanser.clean_data(data)
    
    # Re-validate
    validation_result = quality_engine.validate(cleaned_data, rules)

# Assess overall quality
quality_report = quality_engine.assess_quality(cleaned_data)

print("=== Data Quality Report ===")
print(f"Overall quality score: {quality_report.overall_score:.2f}")
print(f"Completeness: {quality_report.completeness:.2f}")
print(f"Accuracy: {quality_report.accuracy:.2f}")
print(f"Consistency: {quality_report.consistency:.2f}")

# Use quality data for anomaly detection
if quality_report.overall_score >= 0.8:
    print("Data quality sufficient for anomaly detection")
    # Proceed with anomaly detection...
else:
    print("Data quality needs improvement")
```

For more detailed examples and advanced usage, see the [API Reference documentation](../api-reference/).