# CLI Command Reference

The Pynomaly CLI provides a comprehensive command-line interface for all anomaly detection operations, from data management to model training and deployment.

## Overview

The CLI is built with Click and provides intuitive commands organized into logical groups:

- **Detectors**: Create, train, and manage anomaly detectors
- **Datasets**: Upload, analyze, and manage datasets
- **Detection**: Run anomaly detection on data
- **Server**: Start the API server and web interface
- **Experiments**: Manage and run experiments
- **Export**: Export models and results

## Installation and Setup

```bash
# Install Pynomaly CLI
pip install pynomaly

# Verify installation
pynomaly --version

# Get help
pynomaly --help
```

## Global Options

All commands support these global options:

```bash
--config PATH     Configuration file path (default: ~/.pynomaly/config.yml)
--log-level TEXT  Logging level (DEBUG, INFO, WARNING, ERROR)
--output FORMAT   Output format: json, yaml, table (default: table)
--quiet          Suppress non-essential output
--verbose        Enable verbose output
--help           Show help message
```

## Commands

### pynomaly detectors

Manage anomaly detectors.

#### detectors list

List all available detectors.

```bash
pynomaly detectors list [OPTIONS]

Options:
  --algorithm TEXT    Filter by algorithm name
  --trained          Show only trained detectors
  --limit INTEGER    Maximum number of results (default: 50)
  --format TEXT      Output format: table, json, yaml
```

**Examples:**
```bash
# List all detectors
pynomaly detectors list

# List only trained detectors
pynomaly detectors list --trained

# List IsolationForest detectors in JSON format
pynomaly detectors list --algorithm IsolationForest --format json
```

#### detectors create

Create a new anomaly detector.

```bash
pynomaly detectors create [OPTIONS] NAME ALGORITHM

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
pynomaly detectors create "Fraud Detector" IsolationForest

# Create detector with custom parameters
pynomaly detectors create "Advanced Fraud" IsolationForest \
  --contamination 0.05 \
  --parameter n_estimators=200 \
  --parameter random_state=42

# Create LOF detector with description
pynomaly detectors create "Local Outlier Detector" LOF \
  --description "Detects local anomalies in customer behavior" \
  --parameter n_neighbors=20
```

#### detectors show

Show detailed information about a detector.

```bash
pynomaly detectors show [OPTIONS] DETECTOR_ID

Options:
  --include-metrics    Show performance metrics
  --include-config     Show full configuration
```

**Examples:**
```bash
# Show basic detector info
pynomaly detectors show detector_123

# Show with metrics and configuration
pynomaly detectors show detector_123 --include-metrics --include-config
```

#### detectors train

Train a detector with a dataset.

```bash
pynomaly detectors train [OPTIONS] DETECTOR_ID DATASET_ID

Options:
  --validation-split FLOAT    Validation split ratio (default: 0.2)
  --cross-validation          Use cross-validation
  --save-model PATH          Save trained model to file
  --force                    Force retaining if already trained
```

**Examples:**
```bash
# Train detector with default settings
pynomaly detectors train detector_123 dataset_456

# Train with custom validation split
pynomaly detectors train detector_123 dataset_456 --validation-split 0.3

# Train with cross-validation and save model
pynomaly detectors train detector_123 dataset_456 \
  --cross-validation \
  --save-model /path/to/model.pkl
```

#### detectors delete

Delete a detector.

```bash
pynomaly detectors delete [OPTIONS] DETECTOR_ID

Options:
  --force    Skip confirmation prompt
```

#### detectors algorithms

List available algorithms and their parameters.

```bash
pynomaly detectors algorithms [OPTIONS]

Options:
  --category TEXT    Filter by algorithm category
  --detailed         Show parameter details
```

**Examples:**
```bash
# List all algorithms
pynomaly detectors algorithms

# Show detailed parameter information
pynomaly detectors algorithms --detailed

# Filter by category
pynomaly detectors algorithms --category tree_based
```

### pynomaly datasets

Manage datasets for anomaly detection.

#### datasets list

List all datasets.

```bash
pynomaly datasets list [OPTIONS]

Options:
  --format TEXT      Filter by data format (csv, parquet, json)
  --limit INTEGER    Maximum number of results
  --sort TEXT        Sort by: name, created_at, size
```

#### datasets upload

Upload a dataset from file.

```bash
pynomaly datasets upload [OPTIONS] FILE_PATH

Options:
  --name TEXT           Dataset name (default: filename)
  --description TEXT    Dataset description
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
pynomaly datasets upload transactions.csv \
  --name "Credit Card Transactions" \
  --description "Historical transaction data for fraud detection"

# Upload with custom settings
pynomaly datasets upload data.csv \
  --separator ";" \
  --encoding "latin1" \
  --target-column "is_fraud"

# Upload sample of large file
pynomaly datasets upload large_dataset.csv \
  --sample-size 10000 \
  --name "Sample Dataset"
```

#### datasets create

Create a dataset from JSON data.

```bash
pynomaly datasets create [OPTIONS] NAME

Options:
  --data TEXT           JSON data string
  --file PATH          Read JSON data from file
  --description TEXT    Dataset description
```

**Examples:**
```bash
# Create from JSON string
pynomaly datasets create "Test Data" \
  --data '[{"feature1": 1.0, "feature2": 2.0}, {"feature1": 3.0, "feature2": 4.0}]'

# Create from JSON file
pynomaly datasets create "API Data" \
  --file api_data.json \
  --description "Data from API endpoint"
```

#### datasets show

Show dataset information and statistics.

```bash
pynomaly datasets show [OPTIONS] DATASET_ID

Options:
  --sample-size INT    Number of sample rows to show (default: 10)
  --statistics        Show detailed statistics
  --missing-values    Show missing value analysis
```

#### datasets validate

Validate dataset quality and detect issues.

```bash
pynomaly datasets validate [OPTIONS] DATASET_ID

Options:
  --check-duplicates     Check for duplicate rows
  --check-outliers       Check for statistical outliers
  --check-missing        Check missing value patterns
  --report-file PATH     Save validation report to file
```

**Examples:**
```bash
# Basic validation
pynomaly datasets validate dataset_123

# Comprehensive validation with report
pynomaly datasets validate dataset_123 \
  --check-duplicates \
  --check-outliers \
  --check-missing \
  --report-file validation_report.json
```

#### datasets sample

Extract a sample from a dataset.

```bash
pynomaly datasets sample [OPTIONS] DATASET_ID

Options:
  --size INTEGER       Sample size (default: 100)
  --method TEXT        Sampling method: random, stratified, systematic
  --output PATH        Save sample to file
  --seed INTEGER       Random seed for reproducibility
```

#### datasets delete

Delete a dataset.

```bash
pynomaly datasets delete [OPTIONS] DATASET_ID

Options:
  --force    Skip confirmation prompt
```

### pynomaly detect

Run anomaly detection operations.

#### detect run

Run detection on data.

```bash
pynomaly detect run [OPTIONS] DETECTOR_ID

Options:
  --data TEXT          JSON data string
  --file PATH          Read data from file
  --dataset ID         Use existing dataset
  --output PATH        Save results to file
  --threshold FLOAT    Anomaly threshold (0.0-1.0)
  --explain           Include explanations for anomalies
  --batch-size INT     Batch size for large datasets
```

**Examples:**
```bash
# Detect on JSON data
pynomaly detect run detector_123 \
  --data '[{"amount": 100.0, "merchant": "grocery"}]'

# Detect on file
pynomaly detect run detector_123 \
  --file new_transactions.csv \
  --output results.json \
  --explain

# Detect on existing dataset
pynomaly detect run detector_123 \
  --dataset dataset_456 \
  --threshold 0.8 \
  --batch-size 1000
```

#### detect batch

Run batch detection on large datasets.

```bash
pynomaly detect batch [OPTIONS] DETECTOR_ID DATASET_ID

Options:
  --output-format TEXT    Output format: json, csv, parquet
  --output-path PATH      Output file path
  --chunk-size INTEGER    Processing chunk size
  --parallel             Use parallel processing
  --anomalies-only       Output only anomalies
```

#### detect stream

Run real-time detection on streaming data.

```bash
pynomaly detect stream [OPTIONS] DETECTOR_ID

Options:
  --input-format TEXT     Input format: json, csv
  --buffer-size INTEGER   Buffer size for batching
  --output-file PATH      Log results to file
  --webhook-url URL       Send results to webhook
  --kafka-topic TEXT      Kafka topic for input data
```

**Examples:**
```bash
# Stream detection from stdin
echo '{"amount": 5000}' | pynomaly detect stream detector_123

# Stream with webhook notifications
pynomaly detect stream detector_123 \
  --webhook-url https://api.example.com/alerts \
  --buffer-size 100
```

### pynomaly server

Manage the API server and web interface.

#### server start

Start the API server.

```bash
pynomaly server start [OPTIONS]

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
pynomaly server start --reload

# Start production server
pynomaly server start \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4

# Start with SSL
pynomaly server start \
  --ssl-cert cert.pem \
  --ssl-key key.pem \
  --port 8443
```

#### server status

Check server status.

```bash
pynomaly server status [OPTIONS]

Options:
  --url TEXT    Server URL (default: http://localhost:8000)
  --timeout INT Timeout in seconds
```

### pynomaly experiments

Manage experiments and model comparisons.

#### experiments create

Create a new experiment.

```bash
pynomaly experiments create [OPTIONS] NAME DATASET_ID

Options:
  --description TEXT          Experiment description
  --algorithm TEXT            Algorithm to test (can be used multiple times)
  --parameter KEY=VALUE       Parameters for algorithms
  --metric TEXT              Evaluation metrics
  --cross-validation INTEGER  Number of CV folds
  --test-split FLOAT         Test set split ratio
```

**Examples:**
```bash
# Compare multiple algorithms
pynomaly experiments create "Algorithm Comparison" dataset_123 \
  --algorithm IsolationForest \
  --algorithm LOF \
  --algorithm OCSVM \
  --metric precision \
  --metric recall \
  --metric f1_score

# Custom parameters experiment
pynomaly experiments create "Parameter Tuning" dataset_123 \
  --algorithm IsolationForest \
  --parameter contamination=0.05 \
  --parameter contamination=0.1 \
  --parameter contamination=0.15
```

#### experiments list

List experiments.

```bash
pynomaly experiments list [OPTIONS]

Options:
  --status TEXT     Filter by status: running, completed, failed
  --limit INTEGER   Maximum number of results
```

#### experiments show

Show experiment results.

```bash
pynomaly experiments show [OPTIONS] EXPERIMENT_ID

Options:
  --detailed        Show detailed results
  --export PATH     Export results to file
```

#### experiments run

Run an experiment.

```bash
pynomaly experiments run [OPTIONS] EXPERIMENT_ID

Options:
  --async          Run asynchronously
  --timeout INT    Timeout in seconds
```

### pynomaly export

Export models, results, and configurations.

#### export model

Export a trained model.

```bash
pynomaly export model [OPTIONS] DETECTOR_ID OUTPUT_PATH

Options:
  --format TEXT     Export format: pickle, joblib, onnx
  --include-config  Include detector configuration
  --compress       Compress output file
```

**Examples:**
```bash
# Export model as pickle
pynomaly export model detector_123 fraud_model.pkl

# Export with configuration
pynomaly export model detector_123 model.pkl \
  --include-config \
  --compress
```

#### export results

Export detection results.

```bash
pynomaly export results [OPTIONS] DETECTOR_ID OUTPUT_PATH

Options:
  --format TEXT        Export format: csv, json, parquet
  --start-date TEXT    Filter from date (YYYY-MM-DD)
  --end-date TEXT      Filter to date (YYYY-MM-DD)
  --anomalies-only    Export only anomalies
```

#### export config

Export detector or dataset configuration.

```bash
pynomaly export config [OPTIONS] RESOURCE_ID OUTPUT_PATH

Options:
  --type TEXT    Resource type: detector, dataset, experiment
  --format TEXT  Config format: yaml, json
```

## Configuration

### Configuration File

Create a configuration file at `~/.pynomaly/config.yml`:

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
  url: "postgresql://user:pass@localhost/pynomaly"

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
export PYNOMALY_DATABASE_URL="postgresql://user:pass@localhost/pynomaly"

# Logging
export PYNOMALY_LOG_LEVEL="INFO"
export PYNOMALY_LOG_FILE="/var/log/pynomaly.log"

# Cache
export PYNOMALY_CACHE_BACKEND="redis"
export PYNOMALY_REDIS_URL="redis://localhost:6379"
```

## Common Workflows

### 1. Quick Start Workflow

```bash
# 1. Upload dataset
pynomaly datasets upload transactions.csv --name "Transactions"

# 2. Create detector
pynomaly detectors create "Fraud Detector" IsolationForest

# 3. Train detector
pynomaly detectors train detector_123 dataset_456

# 4. Run detection
pynomaly detect run detector_123 --file new_data.csv --output results.json
```

### 2. Experiment Workflow

```bash
# 1. Create experiment
pynomaly experiments create "Algorithm Comparison" dataset_123 \
  --algorithm IsolationForest \
  --algorithm LOF \
  --algorithm OCSVM

# 2. Run experiment
pynomaly experiments run experiment_789

# 3. View results
pynomaly experiments show experiment_789 --detailed

# 4. Export best model
pynomaly export model best_detector_id fraud_model.pkl
```

### 3. Production Deployment Workflow

```bash
# 1. Start server in production mode
pynomaly server start \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4

# 2. Check server status
pynomaly server status

# 3. Set up monitoring (separate terminal)
pynomaly detect stream detector_123 \
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
Error: Configuration file '~/.pynomaly/config.yml' not found
Exit code: 3

# API server unreachable
Error: Could not connect to API server at http://localhost:8000
Exit code: 4
```

## Debugging

### Enable Debug Logging

```bash
# Set debug level
pynomaly --log-level DEBUG detectors list

# Or via environment variable
export PYNOMALY_LOG_LEVEL=DEBUG
pynomaly detectors list
```

### Verbose Output

```bash
# Enable verbose output
pynomaly --verbose detectors create "Test" IsolationForest

# Quiet mode (minimal output)
pynomaly --quiet detectors list
```

### Check Configuration

```bash
# Show current configuration
pynomaly config show

# Validate configuration
pynomaly config validate
```

## Integration Examples

### Shell Scripts

```bash
#!/bin/bash
# Automated detection pipeline

DATASET_ID=$(pynomaly datasets upload data.csv --format json | jq -r '.id')
DETECTOR_ID=$(pynomaly detectors create "Auto Detector" IsolationForest --format json | jq -r '.id')

pynomaly detectors train $DETECTOR_ID $DATASET_ID
pynomaly detect run $DETECTOR_ID --dataset $DATASET_ID --output results.json

echo "Detection completed. Results saved to results.json"
```

### CI/CD Integration

```yaml
# .github/workflows/anomaly-detection.yml
name: Anomaly Detection Pipeline

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
      
      - name: Install Pynomaly
        run: pip install pynomaly
      
      - name: Run Detection
        env:
          PYNOMALY_API_KEY: ${{ secrets.PYNOMALY_API_KEY }}
        run: |
          pynomaly detect run $DETECTOR_ID \
            --file latest_data.csv \
            --output results.json \
            --anomalies-only
      
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: detection-results
          path: results.json
```

This CLI reference provides comprehensive documentation for all Pynomaly command-line operations, making it easy for users to integrate anomaly detection into their workflows and automation pipelines.