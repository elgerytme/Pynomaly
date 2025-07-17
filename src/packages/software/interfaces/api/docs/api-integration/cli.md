# CLI Command Reference

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🔌 [API Integration](README.md) > ⌨️ CLI

---


The Software CLI provides a comprehensive command-line interface for all anomaly processing operations, from data management to processor training and deployment.

## Overview

The CLI is built with Click and provides intuitive commands organized into logical groups:

- **Detectors**: Create, train, and manage anomaly detectors
- **Datasets**: Upload, analyze, and manage datasets
- **Processing**: Run anomaly processing on data
- **Server**: Start the API server and web interface
- **Experiments**: Manage and run experiments
- **Export**: Export models and results

## Installation and Setup

```bash
# Install Software CLI
pip install software

# Verify installation
software --version

# Get help
software --help
```

## Global Options

All commands support these global options:

```bash
--config PATH     Configuration file path (default: ~/.software/config.yml)
--log-level TEXT  Logging level (DEBUG, INFO, WARNING, ERROR)
--output FORMAT   Output format: json, yaml, table (default: table)
--quiet          Suppress non-essential output
--verbose        Enable verbose output
--help           Show help message
```

## Commands

### software detectors

Manage anomaly detectors.

#### detectors list

List all available detectors.

```bash
software detectors list [OPTIONS]

Options:
  --algorithm TEXT    Filter by algorithm name
  --trained          Show only trained detectors
  --limit INTEGER    Maximum number of results (default: 50)
  --format TEXT      Output format: table, json, yaml
```

**Examples:**
```bash
# List all detectors
software detectors list

# List only trained detectors
software detectors list --trained

# List IsolationForest detectors in JSON format
software detectors list --algorithm IsolationForest --format json
```

#### detectors create

Create a new anomaly detector.

```bash
software detectors create [OPTIONS] NAME ALGORITHM

Arguments:
  NAME        Detector name
  ALGORITHM   Algorithm name (IsolationForest, LOF, OCSVM, etc.)

Options:
  --description TEXT       Detector description
  --contamination FLOAT    Expected contamination rate (0.0-0.5, default: 0.1)
  --parameter KEY=VALUE    Algorithm-specific parameters
  --save-config PATH       Save configuration to file
```

**Examples:**
```bash
# Create basic IsolationForest detector
software detectors create "Fraud Detector" IsolationForest

# Create detector with custom parameters
software detectors create "Advanced Fraud" IsolationForest \
  --contamination 0.05 \
  --parameter n_estimators=200 \
  --parameter random_state=42

# Create LOF detector with description
software detectors create "Local Outlier Detector" LOF \
  --description "Detects local anomalies in customer behavior" \
  --parameter n_neighbors=20
```

#### detectors show

Show detailed information about a detector.

```bash
software detectors show [OPTIONS] DETECTOR_ID

Options:
  --include-measurements    Show performance measurements
  --include-config     Show full configuration
```

**Examples:**
```bash
# Show basic detector info
software detectors show detector_123

# Show with measurements and configuration
software detectors show detector_123 --include-measurements --include-config
```

#### detectors train

Train a detector with a data_collection.

```bash
software detectors train [OPTIONS] DETECTOR_ID DATASET_ID

Options:
  --validation-split FLOAT    Validation split ratio (default: 0.2)
  --cross-validation          Use cross-validation
  --save-processor PATH          Save trained processor to file
  --force                    Force retaining if already trained
```

**Examples:**
```bash
# Train detector with default settings
software detectors train detector_123 data_collection_456

# Train with custom validation split
software detectors train detector_123 data_collection_456 --validation-split 0.3

# Train with cross-validation and save processor
software detectors train detector_123 data_collection_456 \
  --cross-validation \
  --save-processor /path/to/processor.pkl
```

#### detectors delete

Delete a detector.

```bash
software detectors delete [OPTIONS] DETECTOR_ID

Options:
  --force    Skip confirmation prompt
```

#### detectors algorithms

List available algorithms and their parameters.

```bash
software detectors algorithms [OPTIONS]

Options:
  --category TEXT    Filter by algorithm category
  --detailed         Show parameter details
```

**Examples:**
```bash
# List all algorithms
software detectors algorithms

# Show detailed parameter information
software detectors algorithms --detailed

# Filter by category
software detectors algorithms --category tree_based
```

### software datasets

Manage datasets for anomaly processing.

#### datasets list

List all datasets.

```bash
software datasets list [OPTIONS]

Options:
  --format TEXT      Filter by data format (csv, parquet, json)
  --limit INTEGER    Maximum number of results
  --sort TEXT        Sort by: name, created_at, size
```

#### datasets upload

Upload a data_collection from file.

```bash
software datasets upload [OPTIONS] FILE_PATH

Options:
  --name TEXT           DataCollection name (default: filename)
  --description TEXT    DataCollection description
  --format TEXT         Force format (csv, parquet, json, excel)
  --separator TEXT      CSV separator (default: ',')
  --encoding TEXT       File encoding (default: utf-8)
  --has-header         CSV has header row
  --target-column TEXT  Target column name (for labeled data)
  --sample-size INT     Upload only a sample (for large files)
```

**Examples:**
```bash
# Upload CSV file
software datasets upload transactions.csv \
  --name "Credit Card Transactions" \
  --description "Historical transaction data for fraud processing"

# Upload with custom settings
software datasets upload data.csv \
  --separator ";" \
  --encoding "latin1" \
  --target-column "is_fraud"

# Upload sample of large file
software datasets upload large_data_collection.csv \
  --sample-size 10000 \
  --name "Sample DataCollection"
```

#### datasets create

Create a data_collection from JSON data.

```bash
software datasets create [OPTIONS] NAME

Options:
  --data TEXT           JSON data string
  --file PATH          Read JSON data from file
  --description TEXT    DataCollection description
```

**Examples:**
```bash
# Create from JSON string
software datasets create "Test Data" \
  --data '[{"feature1": 1.0, "feature2": 2.0}, {"feature1": 3.0, "feature2": 4.0}]'

# Create from JSON file
software datasets create "API Data" \
  --file api_data.json \
  --description "Data from API endpoint"
```

#### datasets show

Show data_collection information and statistics.

```bash
software datasets show [OPTIONS] DATASET_ID

Options:
  --sample-size INT    Number of sample rows to show (default: 10)
  --statistics        Show detailed statistics
  --missing-values    Show missing value analysis
```

#### datasets validate

Validate data_collection quality and detect issues.

```bash
software datasets validate [OPTIONS] DATASET_ID

Options:
  --check-duplicates     Check for duplicate rows
  --check-outliers       Check for statistical outliers
  --check-missing        Check missing value patterns
  --report-file PATH     Save validation report to file
```

**Examples:**
```bash
# Basic validation
software datasets validate data_collection_123

# Comprehensive validation with report
software datasets validate data_collection_123 \
  --check-duplicates \
  --check-outliers \
  --check-missing \
  --report-file validation_report.json
```

#### datasets sample

Extract a sample from a data_collection.

```bash
software datasets sample [OPTIONS] DATASET_ID

Options:
  --size INTEGER       Sample size (default: 100)
  --method TEXT        Sampling method: random, stratified, systematic
  --output PATH        Save sample to file
  --seed INTEGER       Random seed for reproducibility
```

#### datasets delete

Delete a data_collection.

```bash
software datasets delete [OPTIONS] DATASET_ID

Options:
  --force    Skip confirmation prompt
```

### software detect

Run anomaly processing operations.

#### detect run

Run processing on data.

```bash
software detect run [OPTIONS] DETECTOR_ID

Options:
  --data TEXT          JSON data string
  --file PATH          Read data from file
  --data_collection ID         Use existing data_collection
  --output PATH        Save results to file
  --threshold FLOAT    Anomaly threshold (0.0-1.0)
  --explain           Include explanations for anomalies
  --batch-size INT     Batch size for large datasets
```

**Examples:**
```bash
# Detect on JSON data
software detect run detector_123 \
  --data '[{"amount": 100.0, "merchant": "grocery"}]'

# Detect on file
software detect run detector_123 \
  --file new_transactions.csv \
  --output results.json \
  --explain

# Detect on existing data_collection
software detect run detector_123 \
  --data_collection data_collection_456 \
  --threshold 0.8 \
  --batch-size 1000
```

#### detect batch

Run batch processing on large datasets.

```bash
software detect batch [OPTIONS] DETECTOR_ID DATASET_ID

Options:
  --output-format TEXT    Output format: json, csv, parquet
  --output-path PATH      Output file path
  --chunk-size INTEGER    Processing chunk size
  --parallel             Use parallel processing
  --anomalies-only       Output only anomalies
```

#### detect stream

Run real-time processing on streaming data.

```bash
software detect stream [OPTIONS] DETECTOR_ID

Options:
  --input-format TEXT     Input format: json, csv
  --buffer-size INTEGER   Buffer size for batching
  --output-file PATH      Log results to file
  --webhook-url URL       Send results to webhook
  --kafka-topic TEXT      Kafka topic for input data
```

**Examples:**
```bash
# Stream processing from stdin
echo '{"amount": 5000}' | software detect stream detector_123

# Stream with webhook notifications
software detect stream detector_123 \
  --webhook-url https://api.example.com/alerts \
  --buffer-size 100
```

### software server

Manage the API server and web interface.

#### server start

Start the API server.

```bash
software server start [OPTIONS]

Options:
  --host TEXT        Host address (default: 127.0.0.1)
  --port INTEGER     Port number (default: 8000)
  --workers INTEGER  Number of worker processes
  --reload          Enable auto-reload for development
  --access-log      Enable access logging
  --ssl-cert PATH   SSL certificate file
  --ssl-key PATH    SSL private key file
```

**Examples:**
```bash
# Start development server
software server start --reload

# Start production server
software server start \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4

# Start with SSL
software server start \
  --ssl-cert cert.pem \
  --ssl-key key.pem \
  --port 8443
```

#### server status

Check server status.

```bash
software server status [OPTIONS]

Options:
  --url TEXT    Server URL (default: http://localhost:8000)
  --timeout INT Timeout in seconds
```

### software experiments

Manage experiments and processor comparisons.

#### experiments create

Create a new experiment.

```bash
software experiments create [OPTIONS] NAME DATASET_ID

Options:
  --description TEXT          Experiment description
  --algorithm TEXT            Algorithm to test (can be used multiple times)
  --parameter KEY=VALUE       Parameters for algorithms
  --metric TEXT              Evaluation measurements
  --cross-validation INTEGER  Number of CV folds
  --test-split FLOAT         Test set split ratio
```

**Examples:**
```bash
# Compare multiple algorithms
software experiments create "Algorithm Comparison" data_collection_123 \
  --algorithm IsolationForest \
  --algorithm LOF \
  --algorithm OCSVM \
  --metric precision \
  --metric recall \
  --metric f1_score

# Custom parameters experiment
software experiments create "Parameter Tuning" data_collection_123 \
  --algorithm IsolationForest \
  --parameter contamination=0.05 \
  --parameter contamination=0.1 \
  --parameter contamination=0.15
```

#### experiments list

List experiments.

```bash
software experiments list [OPTIONS]

Options:
  --status TEXT     Filter by status: running, completed, failed
  --limit INTEGER   Maximum number of results
```

#### experiments show

Show experiment results.

```bash
software experiments show [OPTIONS] EXPERIMENT_ID

Options:
  --detailed        Show detailed results
  --export PATH     Export results to file
```

#### experiments run

Run an experiment.

```bash
software experiments run [OPTIONS] EXPERIMENT_ID

Options:
  --async          Run asynchronously
  --timeout INT    Timeout in seconds
```

### software export

Export models, results, and configurations.

#### export processor

Export a trained processor.

```bash
software export processor [OPTIONS] DETECTOR_ID OUTPUT_PATH

Options:
  --format TEXT     Export format: pickle, joblib, onnx
  --include-config  Include detector configuration
  --compress       Compress output file
```

**Examples:**
```bash
# Export processor as pickle
software export processor detector_123 fraud_processor.pkl

# Export with configuration
software export processor detector_123 processor.pkl \
  --include-config \
  --compress
```

#### export results

Export processing results.

```bash
software export results [OPTIONS] DETECTOR_ID OUTPUT_PATH

Options:
  --format TEXT        Export format: csv, json, parquet
  --start-date TEXT    Filter from date (YYYY-MM-DD)
  --end-date TEXT      Filter to date (YYYY-MM-DD)
  --anomalies-only    Export only anomalies
```

#### export config

Export detector or data_collection configuration.

```bash
software export config [OPTIONS] RESOURCE_ID OUTPUT_PATH

Options:
  --type TEXT    Resource type: detector, data_collection, experiment
  --format TEXT  Config format: yaml, json
```

## Configuration

### Configuration File

Create a configuration file at `~/.software/config.yml`:

```yaml
# API Configuration
api:
  base_url: "http://localhost:8000"
  timeout: 30
  api_key: "your-api-key"

# Default Settings
defaults:
  output_format: "table"
  log_level: "INFO"
  contamination: 0.1

# Database Configuration
database:
  url: "postgresql://user:pass@localhost/software"

# Cache Configuration
cache:
  enabled: true
  ttl: 300
  backend: "memory"  # or "redis"
```

### Environment Variables

```bash
# API Configuration
export PYNOMALY_API_URL="http://localhost:8000"
export PYNOMALY_API_KEY="your-api-key"

# Database
export PYNOMALY_DATABASE_URL="postgresql://user:pass@localhost/software"

# Logging
export PYNOMALY_LOG_LEVEL="INFO"
export PYNOMALY_LOG_FILE="/var/log/software.log"

# Cache
export PYNOMALY_CACHE_BACKEND="redis"
export PYNOMALY_REDIS_URL="redis://localhost:6379"
```

## Common Workflows

### 1. Quick Start Workflow

```bash
# 1. Upload data_collection
software datasets upload transactions.csv --name "Transactions"

# 2. Create detector
software detectors create "Fraud Detector" IsolationForest

# 3. Train detector
software detectors train detector_123 data_collection_456

# 4. Run processing
software detect run detector_123 --file new_data.csv --output results.json
```

### 2. Experiment Workflow

```bash
# 1. Create experiment
software experiments create "Algorithm Comparison" data_collection_123 \
  --algorithm IsolationForest \
  --algorithm LOF \
  --algorithm OCSVM

# 2. Run experiment
software experiments run experiment_789

# 3. View results
software experiments show experiment_789 --detailed

# 4. Export best processor
software export processor best_detector_id fraud_processor.pkl
```

### 3. Production Deployment Workflow

```bash
# 1. Start server in production mode
software server start \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4

# 2. Check server status
software server status

# 3. Set up monitoring (separate terminal)
software detect stream detector_123 \
  --webhook-url https://monitoring.example.com/alerts
```

## Error Handling

The CLI provides detailed error messages and exit codes:

- **0**: Success
- **1**: General error
- **2**: Invalid arguments
- **3**: Configuration error
- **4**: Network/API error
- **5**: Authentication error

### Common Error Messages

```bash
# Invalid detector ID
Error: Detector 'invalid_id' not found
Exit code: 4

# Missing required argument
Error: Missing argument 'DETECTOR_ID'
Exit code: 2

# Configuration file not found
Error: Configuration file '~/.software/config.yml' not found
Exit code: 3

# API server unreachable
Error: Could not connect to API server at http://localhost:8000
Exit code: 4
```

## Debugging

### Enable Debug Logging

```bash
# Set debug level
software --log-level DEBUG detectors list

# Or via environment variable
export PYNOMALY_LOG_LEVEL=DEBUG
software detectors list
```

### Verbose Output

```bash
# Enable verbose output
software --verbose detectors create "Test" IsolationForest

# Quiet mode (minimal output)
software --quiet detectors list
```

### Check Configuration

```bash
# Show current configuration
software config show

# Validate configuration
software config validate
```

## Integration Examples

### Shell Scripts

```bash
#!/bin/bash
# Automated processing pipeline

DATASET_ID=$(software datasets upload data.csv --format json | jq -r '.id')
DETECTOR_ID=$(software detectors create "Auto Detector" IsolationForest --format json | jq -r '.id')

software detectors train $DETECTOR_ID $DATASET_ID
software detect run $DETECTOR_ID --data_collection $DATASET_ID --output results.json

echo "Processing completed. Results saved to results.json"
```

### CI/CD Integration

```yaml
# .github/workflows/anomaly-processing.yml
name: Anomaly Processing Pipeline

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

jobs:
  detect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install Software
        run: pip install software

      - name: Run Processing
        env:
          PYNOMALY_API_KEY: ${{ secrets.PYNOMALY_API_KEY }}
        run: |
          software detect run $DETECTOR_ID \
            --file latest_data.csv \
            --output results.json \
            --anomalies-only

      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: processing-results
          path: results.json
```

This CLI reference provides comprehensive documentation for all Software command-line operations, making it easy for users to integrate anomaly processing into their workflows and automation pipelines.

---

## 🔗 **Related Documentation**

### **Development**
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](../contributing/README.md)** - Local development environment
- **[Architecture Overview](../architecture/overview.md)** - System design
- **[Implementation Guide](../contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards

### **API Integration**
- **[REST API](../api-integration/rest-api.md)** - HTTP API reference
- **[Python SDK](../api-integration/python-sdk.md)** - Python client library
- **[CLI Reference](../api-integration/cli.md)** - Command-line interface
- **[Authentication](../api-integration/authentication.md)** - Security and auth

### **User Documentation**
- **[User Guides](../../user-guides/README.md)** - Feature usage guides
- **[Getting Started](../../getting-started/README.md)** - Installation and setup
- **[Examples](../../examples/README.md)** - Real-world use cases

### **Deployment**
- **[Production Deployment](../../deployment/README.md)** - Production deployment
- **[Security Setup](../../deployment/SECURITY.md)** - Security configuration
- **[Monitoring](../../user-guides/basic-usage/monitoring.md)** - System observability

---

## 🆘 **Getting Help**

- **[Development Troubleshooting](../contributing/troubleshooting/)** - Development issues
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - Contribution process
