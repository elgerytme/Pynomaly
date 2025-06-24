# Autonomous Mode User Guide

The Pynomaly Autonomous Mode is an intelligent anomaly detection system that automatically analyzes your data, selects optimal algorithms, and provides actionable insights without requiring deep machine learning expertise.

## What is Autonomous Mode?

Autonomous Mode is Pynomaly's flagship feature that provides:

- **🤖 Intelligent Algorithm Selection**: Automatically chooses the best anomaly detection algorithms based on your data characteristics
- **📊 Smart Data Profiling**: Analyzes your data to understand patterns, distributions, and complexity
- **⚙️ Automatic Configuration**: Sets optimal parameters and contamination rates without manual tuning
- **📈 Performance Optimization**: Ranks algorithms by expected performance and confidence
- **💾 Seamless Integration**: Works with multiple data formats and provides various output options

## Quick Start

### Basic Usage

```bash
# Quick anomaly detection on your data
pynomaly auto quick mydata.csv

# Comprehensive analysis with results export
pynomaly auto detect mydata.csv --output results.csv

# Profile your data first to understand characteristics
pynomaly auto profile mydata.csv --verbose
```

### Supported Data Formats

Autonomous Mode automatically detects and processes:

- **CSV files** (`.csv`)
- **Parquet files** (`.parquet`)
- **JSON files** (`.json`)
- **Excel files** (`.xlsx`, `.xls`)
- **Arrow files** (`.arrow`)

## How It Works

### 1. Data Loading & Validation
```bash
pynomaly auto detect data.csv
```
- Automatically detects file format
- Validates data integrity
- Handles missing values appropriately
- Performs basic data cleaning

### 2. Intelligent Data Profiling
The system analyzes:
- **Sample size** and feature count
- **Data types** (numeric, categorical, temporal)
- **Distribution patterns** and statistical properties
- **Correlation structure** between features
- **Missing value patterns**
- **Data complexity score**

### 3. Algorithm Recommendation
Based on profiling results, the system:
- Recommends 1-5 best algorithms
- Estimates performance for each algorithm
- Provides confidence scores and reasoning
- Suggests optimal contamination rates

### 4. Autonomous Detection
- Runs selected algorithms automatically
- Compares results across algorithms
- Ranks by performance and reliability
- Generates comprehensive reports

## Command Reference

### Quick Detection
```bash
# Fastest analysis with default settings
pynomaly auto quick data.csv

# Options:
--output FILE        # Save results to file
--verbose           # Show detailed progress
--threshold FLOAT   # Set confidence threshold (0.0-1.0)
```

### Comprehensive Detection
```bash
# Full autonomous analysis
pynomaly auto detect data.csv

# Advanced options:
--max-algorithms N      # Max algorithms to try (1-10, default: 3)
--auto-tune            # Enable hyperparameter tuning
--confidence-threshold FLOAT  # Minimum confidence (0.0-1.0)
--contamination FLOAT  # Override contamination rate
--output FORMAT        # Output format: csv, json, parquet
--save-models         # Save trained models
--export-plots        # Generate visualization plots
```

### Data Profiling
```bash
# Analyze data characteristics only
pynomaly auto profile data.csv

# Options:
--verbose              # Detailed profiling report
--output FILE         # Save profile to file
--include-correlations # Include correlation analysis
--sample-size N       # Limit sample size for large datasets
```

### Batch Processing
```bash
# Process multiple files
pynomaly auto batch-detect directory/ --pattern "*.csv"

# Options:
--pattern GLOB        # File pattern to match
--output-dir DIR     # Output directory
--parallel N         # Number of parallel processes
--recursive          # Search subdirectories
```

## Understanding Results

### Quick Detection Output
```
Dataset: mydata.csv
Samples: 10,000 | Features: 15 | Complexity: 0.245
Recommended Algorithm: IsolationForest (Confidence: 87%)
Anomalies Found: 127 (1.3%)
Processing Time: 2.3s
```

### Comprehensive Detection Report
```json
{
  "dataset_info": {
    "name": "mydata.csv",
    "samples": 10000,
    "features": 15,
    "complexity_score": 0.245
  },
  "algorithm_recommendations": [
    {
      "algorithm": "IsolationForest",
      "confidence": 0.87,
      "reasoning": "Excellent for high-dimensional data with mixed types",
      "expected_performance": 0.82
    }
  ],
  "detection_results": {
    "anomalies_found": 127,
    "anomaly_rate": 0.013,
    "threshold": 0.15,
    "execution_time_ms": 2300
  }
}
```

## Best Practices

### Data Preparation
1. **Clean Data**: Remove obvious errors and duplicates
2. **Feature Selection**: Include relevant features only
3. **Sample Size**: Aim for 1000+ samples for reliable results
4. **Data Quality**: Ensure consistent formatting and types

### Configuration Tips
1. **Start Simple**: Use `quick` mode first to understand your data
2. **Iterate**: Use profiling results to refine your approach
3. **Validate**: Compare results with domain knowledge
4. **Document**: Save configurations that work well

### Performance Optimization
1. **Large Datasets**: Use sampling for initial analysis
2. **Batch Processing**: Process multiple files efficiently
3. **Resource Management**: Monitor memory usage for large datasets
4. **Parallel Processing**: Enable for batch operations

## Common Use Cases

### Fraud Detection
```bash
# Financial transaction data
pynomaly auto detect transactions.csv \
  --max-algorithms 5 \
  --confidence-threshold 0.9 \
  --output fraud_results.json
```

### Quality Control
```bash
# Manufacturing sensor data
pynomaly auto detect sensor_data.csv \
  --auto-tune \
  --export-plots \
  --output quality_analysis.csv
```

### Network Security
```bash
# Network traffic logs
pynomaly auto detect network_logs.parquet \
  --contamination 0.001 \
  --save-models \
  --output security_alerts.json
```

### Business Intelligence
```bash
# Customer behavior data
pynomaly auto profile customer_data.csv --verbose
pynomaly auto detect customer_data.csv \
  --max-algorithms 3 \
  --output customer_insights.csv
```

## Troubleshooting

### Common Issues

**"No suitable algorithms found"**
- Check data quality and format
- Ensure sufficient sample size (>100 rows)
- Verify feature types are appropriate

**"Low confidence results"**
- Try different contamination rates
- Increase max-algorithms parameter
- Enable auto-tuning for better performance

**"Memory errors with large datasets"**
- Use sampling: `--sample-size 10000`
- Process in batches
- Consider data preprocessing

**"Inconsistent results"**
- Set random seed for reproducibility
- Validate data preprocessing steps
- Check for data drift over time

### Getting Help

```bash
# Show detailed help
pynomaly auto --help
pynomaly auto detect --help

# Enable verbose logging
pynomaly auto detect data.csv --verbose --log-level DEBUG

# Test with sample data
pynomaly auto quick examples/sample_data.csv
```

## Advanced Features

### Custom Contamination Rates
```bash
# Override automatic contamination estimation
pynomaly auto detect data.csv --contamination 0.05
```

### Algorithm Filtering
```bash
# Specify preferred algorithm types
pynomaly auto detect data.csv --prefer-algorithms "isolation,lof"
```

### Export Options
```bash
# Multiple output formats
pynomaly auto detect data.csv \
  --output results.json \
  --export-plots plots/ \
  --save-models models/
```

### Integration with Workflows
```bash
# Pipeline integration
pynomaly auto detect data.csv --output results.json
python process_results.py results.json
```

## API Integration

For programmatic access, use the Python SDK:

```python
from pynomaly import AutonomousDetector

# Initialize detector
detector = AutonomousDetector()

# Quick detection
results = detector.quick_detect('data.csv')

# Comprehensive analysis
results = detector.detect(
    'data.csv',
    max_algorithms=5,
    auto_tune=True,
    confidence_threshold=0.8
)

# Data profiling
profile = detector.profile_data('data.csv')
```

## Next Steps

1. **Try Different Modes**: Experiment with quick, detect, and profile commands
2. **Explore Advanced Options**: Use auto-tuning and custom parameters
3. **Integrate into Workflows**: Automate with batch processing
4. **Monitor Performance**: Track results over time
5. **Scale Up**: Process larger datasets and multiple files

For more advanced usage, see the [Developer Guide](../development/autonomous-mode-developer-guide.md) and [API Reference](../api/autonomous-mode-api.md).