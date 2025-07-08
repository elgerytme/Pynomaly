# Autonomous Anomaly Detection Mode

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🟢 [Basic Usage](README.md) > 🤖 Autonomous Mode

---


Pynomaly's autonomous mode provides a fully automated anomaly detection pipeline that requires minimal configuration. Simply point it at your data, and it will automatically:

- **Auto-detect data format** (CSV, JSON, Excel, Parquet, etc.)
- **Profile your dataset** to understand its characteristics
- **Recommend optimal algorithms** based on data properties
- **Auto-tune hyperparameters** for best performance
- **Run detection** with multiple algorithms
- **Provide comprehensive insights** and export results

## Quick Start

### Simple Detection
```bash
# Just provide your data file - everything else is automatic
pynomaly auto detect data.csv

# Export results
pynomaly auto detect data.csv --output results.csv

# Quick mode for fast results
pynomaly auto quick data.parquet
```

### Data Profiling
```bash
# Understand your data characteristics
pynomaly auto profile data.json --verbose

# Profile with custom sample size
pynomaly auto profile large_data.csv --max-samples 5000
```

### Advanced Configuration
```bash
# Comprehensive detection with tuning
pynomaly auto detect data.csv \
  --max-algorithms 5 \
  --auto-tune \
  --confidence 0.8 \
  --output results.xlsx \
  --format excel \
  --verbose

# Production mode with saving
pynomaly auto detect data.csv \
  --save \
  --max-algorithms 3 \
  --confidence 0.9
```

## How It Works

### 1. Automatic Data Detection

The system automatically detects and loads various data formats:

```python
# Supports multiple formats out of the box
formats = {
    "CSV/TSV": ["csv", "tsv", "txt"],
    "JSON": ["json", "jsonl"],
    "Excel": ["xlsx", "xls", "xlsm"],
    "Parquet": ["parquet", "pq"],
    "Arrow": ["arrow", "feather"]
}
```

**Format Detection Logic:**
- Extension-based detection (e.g., `.csv`, `.json`)
- Content-based detection for ambiguous files
- Automatic delimiter detection for CSV files
- Encoding detection for text files
- Sheet detection for Excel files

### 2. Intelligent Data Profiling

The profiling system analyzes your data to understand:

```python
@dataclass
class DataProfile:
    n_samples: int              # Number of rows
    n_features: int             # Number of columns
    numeric_features: int       # Count of numeric columns
    categorical_features: int   # Count of categorical columns
    temporal_features: int      # Count of date/time columns
    missing_values_ratio: float # Proportion of missing data
    correlation_score: float    # Average correlation between features
    sparsity_ratio: float       # Proportion of zero values
    outlier_ratio_estimate: float # Estimated outlier rate
    seasonality_detected: bool  # Time series seasonality
    trend_detected: bool        # Time series trend
    complexity_score: float     # Overall data complexity (0-1)
    recommended_contamination: float # Suggested contamination rate
```

### 3. Smart Algorithm Selection

Based on data characteristics, the system recommends algorithms:

| Data Characteristics | Recommended Algorithms | Reasoning |
|---------------------|----------------------|-----------|
| **General Purpose** | IsolationForest | Handles mixed data types, scalable |
| **Mostly Numeric** | LocalOutlierFactor | Excellent for density-based anomalies |
| **Complex/Large** | AutoEncoder | Deep learning for complex patterns |
| **High Correlation** | EllipticEnvelope | Good for Gaussian-distributed data |
| **Small Datasets** | OneClassSVM | Handles complex decision boundaries |

### 4. Automatic Hyperparameter Tuning

The system optimizes key parameters:

```python
# Example: IsolationForest tuning
tuning_params = {
    "n_estimators": [50, 100, 200],
    "max_features": [0.5, 0.7, 1.0],
    "contamination": profile.recommended_contamination
}
```

### 5. Comprehensive Results

The autonomous mode provides detailed insights:

```json
{
  "autonomous_detection_results": {
    "success": true,
    "best_algorithm": "IsolationForest",
    "data_profile": {
      "samples": 10000,
      "features": 25,
      "complexity_score": 0.67
    },
    "algorithm_recommendations": [
      {
        "algorithm": "IsolationForest",
        "confidence": 0.85,
        "reasoning": "Handles mixed data types well"
      }
    ],
    "best_result": {
      "algorithm": "IsolationForest",
      "anomalies": 127,
      "anomaly_rate": "1.27%",
      "confidence": "High"
    }
  }
}
```

## Configuration Options

### AutonomousConfig Parameters

```python
@dataclass
class AutonomousConfig:
    max_samples_analysis: int = 10000      # Max samples for profiling
    confidence_threshold: float = 0.8      # Min algorithm confidence
    max_algorithms: int = 5                # Max algorithms to try
    auto_tune_hyperparams: bool = True     # Enable auto-tuning
    save_results: bool = True              # Save to database
    export_results: bool = False           # Export to file
    export_format: str = "csv"             # Export format
    verbose: bool = False                  # Verbose output
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max-algorithms` | Maximum algorithms to try | 5 |
| `--confidence` | Minimum confidence threshold | 0.8 |
| `--auto-tune/--no-tune` | Enable hyperparameter tuning | True |
| `--save/--no-save` | Save results to database | True |
| `--output` | Export results to file | None |
| `--format` | Export format (csv, excel, parquet) | csv |
| `--verbose` | Detailed output | False |
| `--max-samples` | Max samples for analysis | 10000 |

## Python API Usage

### Basic Usage

```python
import asyncio
from pynomaly.application.services.autonomous_service import (
    AutonomousDetectionService, AutonomousConfig
)

# Setup
config = AutonomousConfig(verbose=True)
service = AutonomousDetectionService(...)

# Run autonomous detection
results = await service.detect_autonomous("data.csv", config)

# Access results
best_algorithm = results["autonomous_detection_results"]["best_algorithm"]
anomalies = results["autonomous_detection_results"]["best_result"]["anomalies"]
```

### Advanced Usage

```python
# Custom configuration
config = AutonomousConfig(
    max_algorithms=3,
    confidence_threshold=0.9,
    auto_tune_hyperparams=True,
    export_results=True,
    export_format="parquet"
)

# Direct DataFrame input
import pandas as pd
df = pd.read_csv("data.csv")
results = await service.detect_autonomous(df, config)

# Profile-only mode
dataset = await service._auto_load_data("data.csv", config)
profile = await service._profile_data(dataset, config)
recommendations = await service._recommend_algorithms(profile, config)
```

## Supported Data Types

### Tabular Data
- **CSV/TSV**: Automatic delimiter detection
- **Excel**: Multi-sheet support, automatic cleaning
- **Parquet**: High-performance columnar format
- **JSON**: Automatic nested structure flattening

### Data Characteristics
- **Mixed Types**: Numeric, categorical, temporal
- **Missing Values**: Automatic handling and analysis
- **High Dimensionality**: Efficient algorithms for many features
- **Large Datasets**: Sampling and batch processing
- **Time Series**: Trend and seasonality detection

## Best Practices

### 1. Data Preparation
```bash
# Clean data is better, but not required
# The system handles common issues automatically:
# - Missing values
# - Mixed data types
# - Encoding issues
# - Inconsistent formats
```

### 2. Performance Optimization
```bash
# For large datasets, limit analysis samples
pynomaly auto detect large_data.csv --max-samples 5000

# Use quick mode for initial exploration
pynomaly auto quick data.csv

# Disable tuning for speed
pynomaly auto detect data.csv --no-tune
```

### 3. Quality Control
```bash
# Use higher confidence threshold for critical applications
pynomaly auto detect data.csv --confidence 0.9

# Enable verbose mode for debugging
pynomaly auto detect data.csv --verbose

# Profile data first to understand characteristics
pynomaly auto profile data.csv
```

## Example Workflows

### 1. Quick Exploration
```bash
# Rapid anomaly detection
pynomaly auto quick data.csv
# → Get immediate results with minimal configuration
```

### 2. Production Pipeline
```bash
# Comprehensive detection with export
pynomaly auto detect data.csv \
  --auto-tune \
  --save \
  --output production_results.xlsx \
  --format excel \
  --confidence 0.85
```

### 3. Data Investigation
```bash
# Understand your data first
pynomaly auto profile data.csv --verbose

# Then run targeted detection
pynomaly auto detect data.csv \
  --max-algorithms 3 \
  --confidence 0.8
```

### 4. Batch Processing
```bash
# Process multiple files
for file in *.csv; do
    pynomaly auto quick "$file" --output "results_${file%.csv}.csv"
done
```

## Integration Examples

### Jupyter Notebook
```python
# In a notebook cell
import asyncio
from pynomaly.application.services.autonomous_service import *

# Quick detection
config = AutonomousConfig(verbose=True)
results = await autonomous_service.detect_autonomous("data.csv", config)

# Display results
print(f"Best algorithm: {results['autonomous_detection_results']['best_algorithm']}")
print(f"Anomalies found: {results['autonomous_detection_results']['best_result']['summary']['total_anomalies']}")
```

### Python Script
```python
#!/usr/bin/env python3
import sys
import asyncio
from autonomous_service import run_autonomous_detection

async def main():
    if len(sys.argv) != 2:
        print("Usage: script.py <data_file>")
        return

    results = await run_autonomous_detection(sys.argv[1])
    print(f"Detection complete. Found {results['anomalies']} anomalies.")

if __name__ == "__main__":
    asyncio.run(main())
```

### Docker Integration
```dockerfile
FROM python:3.11
RUN pip install pynomaly
COPY data.csv /data/
CMD ["pynomaly", "auto", "detect", "/data/data.csv", "--output", "/results/anomalies.csv"]
```

## Troubleshooting

### Common Issues

1. **File Format Not Detected**
   ```bash
   # Specify format explicitly if auto-detection fails
   # Currently handled automatically, but you can use specific loaders
   ```

2. **Memory Issues with Large Files**
   ```bash
   # Reduce sample size for profiling
   pynomaly auto detect large_file.csv --max-samples 1000
   ```

3. **No Algorithms Recommended**
   ```bash
   # Lower confidence threshold
   pynomaly auto detect data.csv --confidence 0.5
   ```

4. **Poor Detection Quality**
   ```bash
   # Enable tuning and try more algorithms
   pynomaly auto detect data.csv --auto-tune --max-algorithms 7
   ```

### Getting Help

```bash
# CLI help
pynomaly auto --help
pynomaly auto detect --help
pynomaly auto profile --help

# Verbose output for debugging
pynomaly auto detect data.csv --verbose
```

## Performance Considerations

| Dataset Size | Recommendation | Sample Size | Algorithms |
|-------------|----------------|-------------|------------|
| < 10MB | Full processing | All data | 5+ algorithms |
| 10MB - 100MB | Standard processing | 10K samples | 3-5 algorithms |
| 100MB - 1GB | Optimized processing | 5K samples | 2-3 algorithms |
| > 1GB | Sampled processing | 2K samples | 1-2 algorithms |

The autonomous mode automatically adjusts these parameters based on data characteristics and available resources.

## Next Steps

- See [Algorithm Guide](../../reference/algorithms/README.md) for detailed algorithm information
- Check [Performance Tuning](../advanced-features/performance-tuning.md) for optimization tips
- Read [Production Deployment](../../deployment/production-deployment.md) for enterprise usage
- Try the [Examples](../../examples/README.md) for hands-on learning

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
