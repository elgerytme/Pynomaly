# Pynomaly Usage Guide

This guide demonstrates how to use Pynomaly for anomaly detection with practical examples.

## Quick Start

### 1. Check System Requirements
```bash
python3 pynomaly_cli.py perf-stats
```

### 2. View Available Commands
```bash
python3 pynomaly_cli.py help
```

### 3. List Available Algorithms
```bash
python3 pynomaly_cli.py detector-list
```

## Basic Usage Examples

### Example 1: Simple Anomaly Detection

Using the provided simple anomalies dataset:

```bash
# First, examine the dataset
python3 pynomaly_cli.py dataset-info examples/datasets/simple_anomalies.csv

# Validate data quality
python3 pynomaly_cli.py validate examples/datasets/simple_anomalies.csv

# Run anomaly detection with default settings
python3 pynomaly_cli.py detect examples/datasets/simple_anomalies.csv

# Run with specific algorithm and contamination rate
python3 pynomaly_cli.py detect examples/datasets/simple_anomalies.csv IsolationForest 0.15
```

Expected output: Should detect anomalies at indices corresponding to extreme values (15.0, 25.0, 35.0), (20.5, 30.2, 40.8), and (-5.0, -8.0, -12.0).

### Example 2: Time Series Anomaly Detection

Using the time series dataset:

```bash
# Examine the time series data
python3 pynomaly_cli.py dataset-info examples/datasets/time_series_anomalies.csv

# Run detection with LocalOutlierFactor (good for time series)
python3 pynomaly_cli.py detect examples/datasets/time_series_anomalies.csv LocalOutlierFactor 0.1

# Compare different algorithms
python3 pynomaly_cli.py benchmark examples/datasets/time_series_anomalies.csv
```

Expected output: Should detect anomalies in the extreme readings at timestamps 05:00:00, 11:00:00, and 18:00:00.

### Example 3: Credit Card Fraud Detection

Using the credit card sample dataset:

```bash
# Check data quality first
python3 pynomaly_cli.py validate examples/datasets/credit_card_sample.csv

# Use OneClassSVM for fraud detection
python3 pynomaly_cli.py detect examples/datasets/credit_card_sample.csv OneClassSVM 0.1

# Compare performance across algorithms
python3 pynomaly_cli.py benchmark examples/datasets/credit_card_sample.csv
```

Expected output: Should detect the unusual transactions like the $9,999.99 and $50,000.00 transactions.

## Advanced Usage

### Algorithm Selection Guide

- **IsolationForest**: Best for datasets with clear outliers, works well with high-dimensional data
- **LocalOutlierFactor**: Excellent for local anomalies and time series data
- **OneClassSVM**: Good for complex decision boundaries and fraud detection
- **EllipticEnvelope**: Assumes Gaussian distribution, good for clean numerical data

### Contamination Rate Guidelines

- **0.01-0.05**: Very clean datasets with rare anomalies
- **0.05-0.10**: Typical real-world datasets
- **0.10-0.20**: Noisy datasets or exploratory analysis
- **0.20+**: Highly contaminated datasets (use with caution)

### Performance Optimization

```bash
# Check system performance before large datasets
python3 pynomaly_cli.py perf-stats

# For large datasets, start with IsolationForest (fastest)
python3 pynomaly_cli.py detect large_dataset.csv IsolationForest 0.05

# Use benchmark to compare algorithms on your specific data
python3 pynomaly_cli.py benchmark your_dataset.csv
```

## Data Preparation Tips

### 1. Data Quality Validation
Always validate your data first:
```bash
python3 pynomaly_cli.py validate your_data.csv
```

Common issues and solutions:
- **Missing values**: Remove rows or use imputation
- **Categorical features**: Convert to numeric or exclude
- **Constant features**: Will be automatically flagged for removal
- **Highly skewed data**: Consider log transformation

### 2. Supported Data Formats
- CSV files (most common)
- JSON files
- Must contain numeric features for analysis

### 3. Dataset Size Recommendations
Based on your system specs (check with `perf-stats`):
- **<2GB RAM**: Datasets up to 1,000 samples
- **2-4GB RAM**: Datasets up to 10,000 samples  
- **4-8GB RAM**: Datasets up to 100,000 samples
- **>8GB RAM**: Large datasets (>100,000 samples)

## API Server Usage

Start the API server for web-based access:

```bash
python3 pynomaly_cli.py server-start
```

Then visit:
- **API Documentation**: http://127.0.0.1:8000/docs
- **Main API**: http://127.0.0.1:8000/
- **Health Check**: http://127.0.0.1:8000/api/health

## Troubleshooting

### Common Issues

1. **"File not found"**
   ```bash
   # Check file path and existence
   ls -la your_file.csv
   python3 pynomaly_cli.py dataset-info your_file.csv
   ```

2. **"Insufficient samples"**
   ```bash
   # Ensure dataset has at least 10 rows
   wc -l your_file.csv
   ```

3. **"No numeric features found"**
   ```bash
   # Check data types in your dataset
   python3 pynomaly_cli.py dataset-info your_file.csv
   ```

4. **Poor performance**
   ```bash
   # Check system resources
   python3 pynomaly_cli.py perf-stats
   # Try a smaller dataset or simpler algorithm
   ```

### Error Messages

Pynomaly provides detailed error messages with suggestions:

- **ValidationError**: Data quality issues with specific recommendations
- **ResourceNotFoundError**: File access problems with path details
- **ConfigurationError**: Algorithm or parameter issues with valid options

## Best Practices

### 1. Workflow Recommendation
1. Check system performance (`perf-stats`)
2. Examine your data (`dataset-info`)
3. Validate data quality (`validate`)
4. Start with default detection (`detect file.csv`)
5. Compare algorithms (`benchmark file.csv`)
6. Fine-tune parameters based on results

### 2. Algorithm Selection
- Start with IsolationForest for general anomaly detection
- Use LocalOutlierFactor for time series or local anomalies
- Try OneClassSVM for complex patterns or fraud detection
- Use EllipticEnvelope for normally distributed data

### 3. Parameter Tuning
- Start with contamination rate of 0.1 (10%)
- Adjust based on domain knowledge and results
- Use validation results to guide parameter selection

### 4. Performance Monitoring
- Use `perf-stats` to check system resources
- Use `benchmark` to compare algorithm performance
- Monitor execution times for large datasets

## Example Workflows

### Workflow 1: New Dataset Analysis
```bash
# Step 1: Examine the data
python3 pynomaly_cli.py dataset-info new_data.csv

# Step 2: Validate quality
python3 pynomaly_cli.py validate new_data.csv

# Step 3: Quick detection test
python3 pynomaly_cli.py detect new_data.csv

# Step 4: Compare algorithms
python3 pynomaly_cli.py benchmark new_data.csv

# Step 5: Fine-tune best algorithm
python3 pynomaly_cli.py detect new_data.csv IsolationForest 0.05
```

### Workflow 2: Production Monitoring
```bash
# Check system health
python3 pynomaly_cli.py perf-stats

# Process new data batch
python3 pynomaly_cli.py detect daily_data.csv IsolationForest 0.1

# Monitor API health
curl http://127.0.0.1:8000/api/health
```

### Workflow 3: Research and Experimentation
```bash
# Detailed data analysis
python3 pynomaly_cli.py validate research_data.csv

# Comprehensive algorithm comparison
python3 pynomaly_cli.py benchmark research_data.csv

# Test different contamination rates
python3 pynomaly_cli.py detect research_data.csv IsolationForest 0.05
python3 pynomaly_cli.py detect research_data.csv IsolationForest 0.10
python3 pynomaly_cli.py detect research_data.csv IsolationForest 0.15
```

## Next Steps

- Explore the API endpoints at `/docs` when running the server
- Check out additional datasets in the `examples/datasets/` directory
- Review the system architecture documentation
- Consider integrating Pynomaly into your data pipeline

## Support

For issues or questions:
- Check the troubleshooting section above
- Validate your data format and content
- Ensure system requirements are met
- Review error messages for specific guidance