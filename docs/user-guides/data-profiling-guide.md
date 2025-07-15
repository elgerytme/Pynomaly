# Data Profiling Package User Guide

## Overview

The Data Profiling package provides comprehensive data analysis and profiling capabilities, enabling users to understand their data quality, structure, and characteristics before applying anomaly detection algorithms.

## Quick Start

### Installation

The data profiling package is included with Pynomaly by default:

```bash
pip install pynomaly[standard]
```

### Basic Usage

```python
from pynomaly.packages.data_profiling import DataProfiler
import pandas as pd

# Load your dataset
data = pd.read_csv('your_data.csv')

# Create a profiler instance
profiler = DataProfiler()

# Generate a comprehensive profile
profile = profiler.profile_dataset(data)

# Access profile results
print(f"Dataset shape: {profile.shape}")
print(f"Missing values: {profile.missing_values}")
print(f"Data types: {profile.data_types}")
```

## Features

### 1. Statistical Profiling

Generate comprehensive statistical summaries of your data:

```python
# Statistical analysis
stats_profile = profiler.statistical_analysis(data)

# Access key statistics
print(f"Mean values: {stats_profile.means}")
print(f"Standard deviations: {stats_profile.std_devs}")
print(f"Correlation matrix: {stats_profile.correlation_matrix}")
```

### 2. Schema Analysis

Analyze data structure and types:

```python
# Schema analysis
schema_profile = profiler.analyze_schema(data)

# Check data types and structure
print(f"Column types: {schema_profile.column_types}")
print(f"Nullable columns: {schema_profile.nullable_columns}")
print(f"Unique constraints: {schema_profile.unique_constraints}")
```

### 3. Pattern Discovery

Discover patterns and anomalies in your data:

```python
# Pattern discovery
patterns = profiler.discover_patterns(data)

# Access discovered patterns
print(f"Frequent patterns: {patterns.frequent_patterns}")
print(f"Anomalous patterns: {patterns.anomalous_patterns}")
print(f"Seasonal patterns: {patterns.seasonal_patterns}")
```

### 4. Data Quality Assessment

Assess data quality metrics:

```python
# Quality assessment
quality_report = profiler.assess_quality(data)

# Check quality metrics
print(f"Completeness score: {quality_report.completeness}")
print(f"Consistency score: {quality_report.consistency}")
print(f"Validity score: {quality_report.validity}")
```

## Advanced Features

### Real-time Streaming Profiling

Profile data in real-time streams:

```python
from pynomaly.packages.data_profiling import StreamingProfiler

# Create streaming profiler
streaming_profiler = StreamingProfiler(
    window_size=1000,
    update_frequency='1min'
)

# Profile streaming data
async def profile_stream():
    async for batch in data_stream:
        profile = await streaming_profiler.profile_batch(batch)
        print(f"Latest profile: {profile}")
```

### Custom Profiling Rules

Define custom profiling rules:

```python
from pynomaly.packages.data_profiling import CustomProfileRule

# Define a custom rule
class DateFormatRule(CustomProfileRule):
    def evaluate(self, column_data):
        # Custom logic for date format validation
        return self.validate_date_format(column_data)

# Apply custom rule
profiler.add_custom_rule(DateFormatRule())
profile = profiler.profile_dataset(data)
```

### Profiling Configuration

Configure profiling behavior:

```python
from pynomaly.packages.data_profiling import ProfilingConfig

# Create configuration
config = ProfilingConfig(
    enable_statistical_profiling=True,
    enable_pattern_discovery=True,
    enable_schema_analysis=True,
    sample_size=10000,
    confidence_level=0.95
)

# Use configuration
profiler = DataProfiler(config=config)
```

## Best Practices

### 1. Data Sampling for Large Datasets

For large datasets, use sampling to improve performance:

```python
# Sample large datasets
sampled_data = profiler.smart_sample(
    data, 
    sample_size=50000, 
    strategy='stratified'
)

# Profile the sample
profile = profiler.profile_dataset(sampled_data)
```

### 2. Incremental Profiling

For evolving datasets, use incremental profiling:

```python
# Initial profile
initial_profile = profiler.profile_dataset(initial_data)

# Update profile with new data
updated_profile = profiler.update_profile(
    initial_profile, 
    new_data
)
```

### 3. Profile Comparison

Compare profiles across different time periods:

```python
# Compare profiles
comparison = profiler.compare_profiles(
    profile_v1, 
    profile_v2
)

print(f"Schema changes: {comparison.schema_changes}")
print(f"Distribution shifts: {comparison.distribution_shifts}")
```

## Output Formats

### JSON Export

```python
# Export to JSON
profile_json = profile.to_json()
with open('profile_report.json', 'w') as f:
    f.write(profile_json)
```

### HTML Report

```python
# Generate HTML report
html_report = profiler.generate_html_report(profile)
with open('profile_report.html', 'w') as f:
    f.write(html_report)
```

### DataFrame Summary

```python
# Convert to DataFrame for analysis
profile_df = profile.to_dataframe()
print(profile_df.head())
```

## Integration with Anomaly Detection

Use profiling results to improve anomaly detection:

```python
from pynomaly import AnomalyDetector

# Use profile for feature selection
detector = AnomalyDetector(
    algorithm='IsolationForest',
    feature_selection_strategy=profile.recommended_features
)

# Configure based on data characteristics
if profile.has_categorical_features:
    detector.enable_categorical_encoding()

if profile.has_temporal_features:
    detector.enable_temporal_features()
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Datasets**
   ```python
   # Use chunked processing
   profiler.enable_chunked_processing(chunk_size=10000)
   ```

2. **Slow Performance**
   ```python
   # Enable parallel processing
   profiler.enable_parallel_processing(n_jobs=4)
   ```

3. **Missing Value Handling**
   ```python
   # Configure missing value strategy
   profiler.set_missing_value_strategy('ignore')  # or 'impute', 'flag'
   ```

### Performance Optimization

```python
# Optimize for speed
config = ProfilingConfig(
    enable_heavy_computations=False,
    use_approximations=True,
    cache_results=True
)
```

## API Reference

### Core Classes

- `DataProfiler`: Main profiling class
- `ProfileResult`: Container for profiling results
- `ProfilingConfig`: Configuration options
- `StreamingProfiler`: Real-time profiling

### Key Methods

- `profile_dataset(data)`: Generate comprehensive profile
- `statistical_analysis(data)`: Statistical profiling
- `analyze_schema(data)`: Schema analysis
- `discover_patterns(data)`: Pattern discovery
- `assess_quality(data)`: Quality assessment

## Examples

### Complete Workflow Example

```python
import pandas as pd
from pynomaly.packages.data_profiling import DataProfiler, ProfilingConfig

# Load data
data = pd.read_csv('sales_data.csv')

# Configure profiler
config = ProfilingConfig(
    enable_statistical_profiling=True,
    enable_pattern_discovery=True,
    enable_schema_analysis=True,
    confidence_level=0.95
)

# Create profiler
profiler = DataProfiler(config=config)

# Generate profile
profile = profiler.profile_dataset(data)

# Analyze results
print("=== Data Profile Summary ===")
print(f"Dataset shape: {profile.shape}")
print(f"Data quality score: {profile.overall_quality_score}")
print(f"Recommended preprocessing: {profile.preprocessing_recommendations}")

# Generate reports
html_report = profiler.generate_html_report(profile)
with open('data_profile_report.html', 'w') as f:
    f.write(html_report)

print("Profile analysis complete!")
```

For more examples and advanced usage, see the [API Reference documentation](../api-reference/).