# Data Preprocessing CLI Reference

This guide covers Pynomaly's comprehensive data preprocessing capabilities available through the command-line interface. The preprocessing commands bridge the gap between raw data and effective anomaly detection by providing production-ready data cleaning, transformation, and pipeline management.

## Overview

The preprocessing CLI is organized into three main command groups:

- **`pynomaly data clean`** - Data cleaning operations (missing values, outliers, duplicates)
- **`pynomaly data transform`** - Feature transformations (scaling, encoding, engineering)
- **`pynomaly data pipeline`** - Pipeline management (create, apply, save, load)

## Quick Start

```bash
# Analyze data quality and get preprocessing recommendations
pynomaly dataset quality <dataset_id>

# Clean data with recommended strategies
pynomaly data clean <dataset_id> --missing fill_median --outliers clip --duplicates

# Transform features for better anomaly detection
pynomaly data transform <dataset_id> --scaling standard --encoding onehot

# Create and apply preprocessing pipelines
pynomaly data pipeline create --name my_pipeline
pynomaly data pipeline apply --name my_pipeline --dataset <dataset_id>
```

## Data Cleaning Commands

### Basic Cleaning

```bash
# Preview cleaning operations (dry run)
pynomaly data clean <dataset_id> --dry-run

# Handle missing values
pynomaly data clean <dataset_id> --missing drop_rows
pynomaly data clean <dataset_id> --missing fill_median
pynomaly data clean <dataset_id> --missing fill_mean
pynomaly data clean <dataset_id> --missing knn_impute

# Handle outliers
pynomaly data clean <dataset_id> --outliers remove
pynomaly data clean <dataset_id> --outliers clip --outlier-threshold 3.0
pynomaly data clean <dataset_id> --outliers winsorize

# Remove duplicates
pynomaly data clean <dataset_id> --duplicates
```

### Advanced Cleaning

```bash
# Handle zero and infinite values
pynomaly data clean <dataset_id> --zeros remove --infinite clip

# Comprehensive cleaning
pynomaly data clean <dataset_id> \
  --missing fill_median \
  --outliers clip \
  --duplicates \
  --zeros remove \
  --infinite remove

# Save cleaned data as new dataset
pynomaly data clean <dataset_id> \
  --missing fill_median \
  --save-as cleaned_dataset
```

### Missing Value Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `drop_rows` | Remove rows with missing values | Small amounts of missing data |
| `drop_columns` | Remove columns with missing values | Columns with high missing rates |
| `fill_mean` | Fill with column mean | Numeric data, normal distribution |
| `fill_median` | Fill with column median | Numeric data, skewed distribution |
| `fill_mode` | Fill with most frequent value | Categorical data |
| `fill_constant` | Fill with specified constant | Domain-specific defaults |
| `fill_forward` | Forward fill (time series) | Sequential/time series data |
| `fill_backward` | Backward fill (time series) | Sequential/time series data |
| `interpolate` | Linear interpolation | Time series data |
| `knn_impute` | K-nearest neighbors imputation | Complex missing patterns |

### Outlier Handling Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `remove` | Remove outlier rows | Clean datasets needed |
| `clip` | Clip to threshold boundaries | Preserve data volume |
| `transform_log` | Log transformation | Right-skewed distributions |
| `transform_sqrt` | Square root transformation | Moderate skewness |
| `winsorize` | Replace with percentile values | Robust statistics needed |

## Data Transformation Commands

### Feature Scaling

```bash
# Standard scaling (z-score normalization)
pynomaly data transform <dataset_id> --scaling standard

# Min-max scaling (0-1 range)
pynomaly data transform <dataset_id> --scaling minmax

# Robust scaling (median and IQR)
pynomaly data transform <dataset_id> --scaling robust

# Quantile transformation
pynomaly data transform <dataset_id> --scaling quantile

# Power transformation (Yeo-Johnson)
pynomaly data transform <dataset_id> --scaling power
```

### Categorical Encoding

```bash
# One-hot encoding
pynomaly data transform <dataset_id> --encoding onehot

# Label encoding
pynomaly data transform <dataset_id> --encoding label

# Ordinal encoding
pynomaly data transform <dataset_id> --encoding ordinal

# Target encoding (requires target column)
pynomaly data transform <dataset_id> --encoding target

# Binary encoding
pynomaly data transform <dataset_id> --encoding binary

# Frequency encoding
pynomaly data transform <dataset_id> --encoding frequency
```

### Feature Engineering

```bash
# Feature selection
pynomaly data transform <dataset_id> --feature-selection variance_threshold
pynomaly data transform <dataset_id> --feature-selection univariate
pynomaly data transform <dataset_id> --feature-selection correlation_threshold

# Polynomial features
pynomaly data transform <dataset_id> --polynomial 2

# Data type optimization
pynomaly data transform <dataset_id> --optimize-dtypes

# Column name normalization
pynomaly data transform <dataset_id> --normalize-names

# Exclude specific columns
pynomaly data transform <dataset_id> \
  --scaling standard \
  --exclude "id,timestamp,target"
```

### Comprehensive Transformation

```bash
# Complete preprocessing pipeline
pynomaly data transform <dataset_id> \
  --scaling standard \
  --encoding onehot \
  --feature-selection variance_threshold \
  --normalize-names \
  --optimize-dtypes \
  --save-as preprocessed_dataset
```

## Pipeline Management

### Creating Pipelines

```bash
# Interactive pipeline creation
pynomaly data pipeline create --name my_pipeline

# Create from configuration file
pynomaly data pipeline create --name my_pipeline --config config.json
```

### Pipeline Configuration Format

```json
{
  "name": "comprehensive_preprocessing",
  "steps": [
    {
      "name": "handle_missing",
      "operation": "handle_missing_values",
      "parameters": {"strategy": "fill_median"},
      "enabled": true,
      "description": "Fill missing values with median"
    },
    {
      "name": "remove_outliers",
      "operation": "handle_outliers",
      "parameters": {"strategy": "clip", "threshold": 3.0},
      "enabled": true,
      "description": "Clip outliers beyond 3 standard deviations"
    },
    {
      "name": "scale_features",
      "operation": "scale_features",
      "parameters": {"strategy": "standard"},
      "enabled": true,
      "description": "Apply standard scaling"
    }
  ]
}
```

### Managing Pipelines

```bash
# List all pipelines
pynomaly data pipeline list

# Show pipeline details
pynomaly data pipeline show --name my_pipeline

# Apply pipeline to dataset
pynomaly data pipeline apply --name my_pipeline --dataset <dataset_id>

# Preview pipeline application
pynomaly data pipeline apply --name my_pipeline --dataset <dataset_id> --dry-run

# Save pipeline to file
pynomaly data pipeline save --name my_pipeline --output my_pipeline.json

# Load pipeline from file
pynomaly data pipeline load --config my_pipeline.json --name loaded_pipeline

# Delete pipeline
pynomaly data pipeline delete --name my_pipeline
```

## Integration with Data Quality Analysis

The preprocessing commands integrate seamlessly with Pynomaly's data quality analysis:

```bash
# Analyze data quality and get preprocessing recommendations
pynomaly dataset quality <dataset_id>
```

This command now provides specific preprocessing command suggestions:

```
Preprocessing Commands:
  Suggested commands to improve data quality:
    pynomaly data clean abc12345 --missing fill_median --infinite remove
    pynomaly data transform abc12345 --scaling standard --encoding onehot
    pynomaly data pipeline create --name dataset_name_pipeline

  To preview changes without applying them, add --dry-run to any command.
  To save cleaned data as a new dataset, add --save-as new_name.
```

## Best Practices

### Data Cleaning

1. **Always use dry run first**: Preview changes with `--dry-run` before applying
2. **Save intermediate results**: Use `--save-as` to preserve original data
3. **Handle missing values contextually**: Choose strategies based on data type and domain
4. **Consider outlier impact**: Understand whether outliers are errors or important signals
5. **Document decisions**: Use pipeline descriptions to document preprocessing choices

### Feature Transformation

1. **Scale before anomaly detection**: Most algorithms benefit from feature scaling
2. **Encode categoricals appropriately**: Choose encoding based on cardinality and relationships
3. **Remove low-variance features**: Constant features don't contribute to anomaly detection
4. **Optimize data types**: Reduce memory usage with appropriate dtypes
5. **Test transformation impact**: Compare anomaly detection performance before/after

### Pipeline Management

1. **Create reusable pipelines**: Build pipelines for common data types or domains
2. **Version control configurations**: Save pipeline configs to version control
3. **Test on sample data**: Validate pipelines on representative samples first
4. **Monitor pipeline performance**: Track data quality improvements from preprocessing
5. **Document business logic**: Include descriptions explaining preprocessing decisions

## Common Workflows

### Financial Data Preprocessing

```bash
# Financial transaction data typical workflow
pynomaly data clean <dataset_id> \
  --missing fill_forward \
  --outliers winsorize \
  --duplicates

pynomaly data transform <dataset_id> \
  --scaling robust \
  --encoding frequency \
  --feature-selection correlation_threshold
```

### IoT Sensor Data Preprocessing

```bash
# IoT sensor data typical workflow
pynomaly data clean <dataset_id> \
  --missing interpolate \
  --infinite remove \
  --zeros keep

pynomaly data transform <dataset_id> \
  --scaling minmax \
  --normalize-names \
  --optimize-dtypes
```

### E-commerce Data Preprocessing

```bash
# E-commerce transaction data workflow
pynomaly data clean <dataset_id> \
  --missing fill_mode \
  --outliers clip \
  --duplicates

pynomaly data transform <dataset_id> \
  --scaling standard \
  --encoding onehot \
  --polynomial 2
```

## Performance Considerations

### Memory Management

- Use `--optimize-dtypes` to reduce memory usage
- Process large datasets in chunks for memory efficiency
- Monitor memory usage during polynomial feature generation

### Processing Speed

- Dry run operations are fast and help plan processing time
- Feature selection reduces dimensionality and processing time
- Consider sampling large datasets for pipeline development

### Quality vs. Speed Trade-offs

- KNN imputation is accurate but slow for large datasets
- Polynomial features create many features quickly
- Standard scaling is faster than quantile transformation

## Error Handling

### Common Errors and Solutions

```bash
# Dataset not found
Error: Dataset with ID 'abc123' not found
Solution: Use `pynomaly dataset list` to find correct ID

# Invalid strategy
Error: Invalid missing value strategy: 'invalid_strategy'
Solution: Use --help to see available strategies

# Insufficient data for operation
Error: Not enough data for KNN imputation
Solution: Use simpler imputation strategy or increase dataset size

# Memory issues with large datasets
Error: MemoryError during transformation
Solution: Use data type optimization or process in chunks
```

### Debugging Tips

1. Use `--dry-run` to preview operations without applying changes
2. Start with small datasets to test pipeline configurations
3. Check data types and shapes before and after transformations
4. Monitor memory usage during large dataset processing
5. Validate results with sample data inspection

## Advanced Usage

### Custom Preprocessing Scripts

Combine CLI commands in scripts for automated workflows:

```bash
#!/bin/bash
# Automated preprocessing workflow

DATASET_ID=$1
PIPELINE_NAME="${DATASET_ID}_pipeline"

# Analyze quality
pynomaly dataset quality $DATASET_ID

# Apply cleaning based on quality analysis
pynomaly data clean $DATASET_ID \
  --missing fill_median \
  --outliers clip \
  --duplicates \
  --save-as "${DATASET_ID}_cleaned"

# Transform features
pynomaly data transform "${DATASET_ID}_cleaned" \
  --scaling standard \
  --encoding onehot \
  --save-as "${DATASET_ID}_preprocessed"

echo "Preprocessing complete: ${DATASET_ID}_preprocessed"
```

### Integration with CI/CD

```yaml
# Example GitHub Actions workflow
- name: Preprocess data
  run: |
    pynomaly data clean ${{ env.DATASET_ID }} \
      --missing fill_median \
      --outliers clip \
      --dry-run
    
    pynomaly data pipeline apply \
      --name production_pipeline \
      --dataset ${{ env.DATASET_ID }}
```

## Conclusion

Pynomaly's preprocessing CLI provides a comprehensive, production-ready solution for preparing data for anomaly detection. The combination of flexible cleaning operations, powerful transformations, and reusable pipelines enables efficient and reproducible data preprocessing workflows.

Key benefits:

- **Comprehensive**: Covers all common data quality issues
- **Flexible**: Multiple strategies for each operation
- **Safe**: Dry-run mode and save-as options protect original data
- **Reusable**: Pipeline management for consistent preprocessing
- **Integrated**: Seamless connection with anomaly detection workflow
- **Production-ready**: Error handling, performance optimization, and automation support

For more examples and advanced usage patterns, see the [preprocessing examples](../examples/preprocessing_cli_examples.py) and [anomaly detection workflow guide](workflow.md).