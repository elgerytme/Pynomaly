# CLI Command Reference

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Cli

---


## Overview

The Pynomaly CLI provides a comprehensive command-line interface for anomaly detection operations. This reference covers all available commands with detailed examples, options, and usage patterns for both interactive and automated workflows.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Global Options](#global-options)
3. [Core Commands](#core-commands)
4. [Detector Management](#detector-management)
5. [Dataset Operations](#dataset-operations)
6. [Detection Workflows](#detection-workflows)
7. [Server Management](#server-management)
8. [Performance Commands](#performance-commands)
9. [Configuration Management](#configuration-management)
10. [Advanced Usage](#advanced-usage)
11. [Scripting and Automation](#scripting-and-automation)

## Installation and Setup

### Installing the CLI

```bash
# Install Pynomaly with CLI support
pip install pynomaly[cli]

# Or install from source
git clone https://github.com/your-org/pynomaly.git
cd pynomaly
poetry install --extras cli

# Verify installation
pynomaly --version
```

### Initial Configuration

```bash
# Initialize configuration
pynomaly config init

# Set default configuration
pynomaly config set database_url "postgresql://user:pass@localhost/pynomaly"
pynomaly config set log_level "INFO"
pynomaly config set cache_enabled true

# View current configuration
pynomaly config show
```

## Global Options

All Pynomaly commands support these global options:

```bash
pynomaly [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]

Global Options:
  --config PATH          Configuration file path [default: ~/.pynomaly/config.toml]
  --log-level LEVEL      Logging level [default: INFO]
  --output FORMAT        Output format: json, table, csv [default: table]
  --quiet                Suppress non-essential output
  --verbose              Enable verbose output
  --help                 Show help and exit
  --version              Show version and exit
```

## Core Commands

### Version and Help

```bash
# Show version information
pynomaly --version
pynomaly version

# Get help for any command
pynomaly --help
pynomaly COMMAND --help

# Show system status
pynomaly status
```

**Example Output:**
```
Pynomaly CLI v1.0.0
Python: 3.11.5
Platform: Linux-5.15.0-generic
Available algorithms: 79
Database: Connected (PostgreSQL 14.2)
Cache: Connected (Redis 7.0)
```

### Quick Start

```bash
# Interactive quick start wizard
pynomaly quickstart

# Quick start with specific dataset
pynomaly quickstart --dataset data.csv --algorithm IsolationForest

# Automated quick start (non-interactive)
pynomaly quickstart --auto --dataset data.csv
```

## Detector Management

### Creating Detectors

```bash
# Create a basic detector
pynomaly detector create --name "Fraud Detection" --algorithm IsolationForest

# Create with specific hyperparameters
pynomaly detector create \
  --name "Network Anomaly Detector" \
  --algorithm IsolationForest \
  --contamination 0.1 \
  --n-estimators 200 \
  --max-samples auto

# Create with configuration file
pynomaly detector create --config detector_config.json

# Create multiple detectors from template
pynomaly detector create --template ensemble_template.yaml
```

**Configuration File Example (detector_config.json):**
```json
{
  "name": "Advanced Fraud Detector",
  "algorithm_name": "IsolationForest",
  "contamination_rate": 0.05,
  "hyperparameters": {
    "n_estimators": 500,
    "max_samples": 0.8,
    "contamination": 0.05,
    "random_state": 42
  },
  "description": "High-sensitivity fraud detection model"
}
```

### Listing and Viewing Detectors

```bash
# List all detectors
pynomaly detector list

# List with filtering
pynomaly detector list --algorithm IsolationForest --fitted-only
pynomaly detector list --created-after 2024-01-01

# Show detailed detector information
pynomaly detector show DETECTOR_ID

# Show detector in different formats
pynomaly detector show DETECTOR_ID --output json
pynomaly detector show DETECTOR_ID --output yaml

# List available algorithms
pynomaly detector algorithms
pynomaly detector algorithms --framework pytorch --category ensemble
```

**Example Output:**
```
┌──────────────────────────────────────┬─────────────────────┬─────────────────┬──────────┬─────────────┐
│ ID                                   │ Name                │ Algorithm       │ Fitted   │ Created     │
├──────────────────────────────────────┼─────────────────────┼─────────────────┼──────────┼─────────────┤
│ 123e4567-e89b-12d3-a456-426614174000 │ Fraud Detection     │ IsolationForest │ Yes      │ 2024-01-15  │
│ 456e7890-e89b-12d3-a456-426614174001 │ Network Monitor     │ LOF             │ No       │ 2024-01-16  │
│ 789e1234-e89b-12d3-a456-426614174002 │ Log Anomaly         │ AutoEncoder     │ Yes      │ 2024-01-17  │
└──────────────────────────────────────┴─────────────────────┴─────────────────┴──────────┴─────────────┘
```

### Updating and Deleting Detectors

```bash
# Update detector configuration
pynomaly detector update DETECTOR_ID --name "Updated Name"
pynomaly detector update DETECTOR_ID --contamination 0.15

# Update hyperparameters
pynomaly detector update DETECTOR_ID --hyperparameters '{"n_estimators": 300}'

# Delete detector
pynomaly detector delete DETECTOR_ID

# Delete with confirmation bypass
pynomaly detector delete DETECTOR_ID --force

# Bulk operations
pynomaly detector delete --all --algorithm IsolationForest --confirm
```

## Dataset Operations

### Loading Datasets

```bash
# Load CSV dataset
pynomaly dataset load data.csv --name "Transaction Data"

# Load with specific options
pynomaly dataset load data.csv \
  --name "Sales Data" \
  --description "Monthly sales transactions" \
  --target-column is_anomaly \
  --separator ";" \
  --skip-rows 1

# Load Parquet dataset
pynomaly dataset load data.parquet --name "Time Series Data"

# Load from URL
pynomaly dataset load https://example.com/data.csv --name "Remote Data"

# Load multiple files
pynomaly dataset load "data/*.csv" --name-pattern "Data {filename}"

# Load with preprocessing
pynomaly dataset load data.csv \
  --name "Preprocessed Data" \
  --normalize \
  --remove-nulls \
  --feature-selection auto
```

### Dataset Information and Sampling

```bash
# List datasets
pynomaly dataset list
pynomaly dataset list --format csv --has-target

# Show dataset details
pynomaly dataset show DATASET_ID
pynomaly dataset show DATASET_ID --statistics

# Sample dataset
pynomaly dataset sample DATASET_ID --size 10
pynomaly dataset sample DATASET_ID --size 0.1 --percentage

# Export dataset sample
pynomaly dataset sample DATASET_ID --size 1000 --output sample.csv
```

**Example Output:**
```
Dataset: Transaction Data (ID: 456e7890-e89b-12d3-a456-426614174000)

Basic Information:
  Name: Transaction Data
  Format: CSV
  Size: 2.5 MB
  Samples: 10,000
  Features: 15
  Target Column: is_fraud
  Created: 2024-01-15 10:30:00

Feature Information:
┌────────────────┬─────────┬──────────┬─────────────┬─────────────┐
│ Feature        │ Type    │ Missing  │ Min         │ Max         │
├────────────────┼─────────┼──────────┼─────────────┼─────────────┤
│ amount         │ float64 │ 0        │ 0.01        │ 9999.99     │
│ merchant_cat   │ object  │ 0        │ -           │ -           │
│ hour_of_day    │ int64   │ 0        │ 0           │ 23          │
└────────────────┴─────────┴──────────┴─────────────┴─────────────┘
```

### Dataset Export and Management

```bash
# Export dataset
pynomaly dataset export DATASET_ID --output exported_data.csv
pynomaly dataset export DATASET_ID --output data.parquet --format parquet

# Export with filtering
pynomaly dataset export DATASET_ID \
  --output filtered_data.csv \
  --filter "amount > 100" \
  --sample-size 1000

# Delete dataset
pynomaly dataset delete DATASET_ID

# Dataset statistics
pynomaly dataset stats DATASET_ID
pynomaly dataset stats DATASET_ID --export stats.json
```

## Detection Workflows

### Training Models

```bash
# Basic training
pynomaly train --detector DETECTOR_ID --dataset DATASET_ID

# Training with validation split
pynomaly train \
  --detector DETECTOR_ID \
  --dataset DATASET_ID \
  --validation-split 0.2 \
  --save-model

# Cross-validation training
pynomaly train \
  --detector DETECTOR_ID \
  --dataset DATASET_ID \
  --cross-validation \
  --cv-folds 5

# Training with custom parameters
pynomaly train \
  --detector DETECTOR_ID \
  --dataset DATASET_ID \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001

# Hyperparameter tuning
pynomaly train \
  --detector DETECTOR_ID \
  --dataset DATASET_ID \
  --tune-hyperparameters \
  --trials 50 \
  --metric f1_score
```

**Example Output:**
```
Training Detector: Fraud Detection (IsolationForest)
Dataset: Transaction Data (10,000 samples, 15 features)

[██████████████████████████████] 100% Training completed

Results:
  Training Time: 2.3 seconds
  Validation Metrics:
    Precision: 0.924
    Recall: 0.887
    F1 Score: 0.905
    AUC Score: 0.943
  
  Model saved to: models/detector_123e4567_20240115.pkl
  Training log: logs/training_20240115_103045.log
```

### Running Predictions

```bash
# Predict on dataset
pynomaly predict --detector DETECTOR_ID --dataset DATASET_ID

# Predict with threshold
pynomaly predict \
  --detector DETECTOR_ID \
  --dataset DATASET_ID \
  --threshold 0.7 \
  --output predictions.csv

# Predict on single data point
pynomaly predict \
  --detector DETECTOR_ID \
  --data '{"amount": 1500, "hour": 3}' \
  --explain

# Batch prediction
pynomaly predict \
  --detector DETECTOR_ID \
  --input batch_data.csv \
  --output results.csv \
  --batch-size 1000

# Real-time prediction
pynomaly predict \
  --detector DETECTOR_ID \
  --stream \
  --input-format json \
  --output-format json
```

### Batch Processing

```bash
# Batch process multiple files
pynomaly batch \
  --detector DETECTOR_ID \
  --input-pattern "data/batch_*.csv" \
  --output-dir results/ \
  --parallel-jobs 4

# Batch with preprocessing
pynomaly batch \
  --detector DETECTOR_ID \
  --input-pattern "data/*.csv" \
  --output-dir results/ \
  --preprocess \
  --normalize \
  --feature-selection

# Scheduled batch processing
pynomaly batch \
  --detector DETECTOR_ID \
  --input-dir /data/incoming \
  --output-dir /data/processed \
  --watch \
  --interval 300  # Process every 5 minutes
```

### Results Analysis

```bash
# View prediction results
pynomaly results list
pynomaly results show RESULT_ID

# Results with filtering
pynomaly results list --detector DETECTOR_ID --anomalies-only
pynomaly results list --date-from 2024-01-01 --date-to 2024-01-31

# Export results
pynomaly results export RESULT_ID --output results.csv
pynomaly results export --detector DETECTOR_ID --output all_results.json

# Results statistics
pynomaly results stats RESULT_ID
pynomaly results summary --detector DETECTOR_ID
```

## Server Management

### Starting and Stopping Server

```bash
# Start API server
pynomaly server start

# Start with custom configuration
pynomaly server start \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4 \
  --reload

# Start in development mode
pynomaly server start --dev

# Start with SSL
pynomaly server start \
  --ssl-cert /path/to/cert.pem \
  --ssl-key /path/to/key.pem

# Start as daemon
pynomaly server start --daemon --pid-file pynomaly.pid
```

### Server Monitoring

```bash
# Server status
pynomaly server status

# Server logs
pynomaly server logs
pynomaly server logs --tail 100 --follow

# Stop server
pynomaly server stop
pynomaly server stop --pid-file pynomaly.pid

# Restart server
pynomaly server restart
```

**Example Output:**
```
Server Status: Running
PID: 12345
Host: 0.0.0.0:8000
Workers: 4
Uptime: 2 days, 14 hours
Requests Served: 45,234
Average Response Time: 125ms
Memory Usage: 512MB
```

## Performance Commands

### Performance Monitoring

```bash
# Check performance metrics
pynomaly perf metrics

# Monitor connection pools
pynomaly perf pools
pynomaly perf pools --detailed

# Query performance analysis
pynomaly perf queries
pynomaly perf queries --slow-only --min-time 1000

# Cache performance
pynomaly perf cache
pynomaly perf cache --hit-rate --memory-usage

# System resource monitoring
pynomaly perf system
pynomaly perf system --cpu --memory --disk
```

### Performance Optimization

```bash
# Optimize database
pynomaly perf optimize --database
pynomaly perf optimize --indexes --vacuum

# Cache optimization
pynomaly perf optimize --cache --clear-expired

# General optimization recommendations
pynomaly perf recommendations
pynomaly perf recommendations --export recommendations.json

# Performance reporting
pynomaly perf report --output performance_report.html
pynomaly perf report --period 7d --format pdf
```

### Monitoring and Alerting

```bash
# Start performance monitoring
pynomaly perf monitor --threshold-cpu 80 --threshold-memory 85

# Real-time monitoring dashboard
pynomaly perf dashboard

# Performance alerts
pynomaly perf alerts list
pynomaly perf alerts configure --email admin@example.com
```

## Configuration Management

### Configuration Commands

```bash
# Initialize new configuration
pynomaly config init

# Show current configuration
pynomaly config show
pynomaly config show --section database
pynomaly config show --format json

# Set configuration values
pynomaly config set database.host localhost
pynomaly config set cache.enabled true
pynomaly config set logging.level DEBUG

# Unset configuration
pynomaly config unset cache.redis_url

# Validate configuration
pynomaly config validate
pynomaly config validate --fix-issues

# Export/Import configuration
pynomaly config export --output config_backup.toml
pynomaly config import --file config_backup.toml
```

### Environment-Specific Configuration

```bash
# Set environment
pynomaly config env set production

# Environment-specific settings
pynomaly config set --env production database.pool_size 50
pynomaly config set --env development log_level DEBUG

# Switch environments
pynomaly config env use development
pynomaly config env use production

# List environments
pynomaly config env list
```

**Configuration File Example (~/.pynomaly/config.toml):**
```toml
[database]
url = "postgresql://user:pass@localhost/pynomaly"
pool_size = 20
pool_timeout = 30

[cache]
enabled = true
redis_url = "redis://localhost:6379/0"
ttl = 3600

[logging]
level = "INFO"
format = "json"
file = "/var/log/pynomaly.log"

[api]
host = "0.0.0.0"
port = 8000
workers = 4

[performance]
max_memory_percent = 80
connection_timeout = 30
```

## Advanced Usage

### Ensemble Detection

```bash
# Create ensemble detector
pynomaly ensemble create \
  --name "Multi-Algorithm Ensemble" \
  --detectors DETECTOR_ID1,DETECTOR_ID2,DETECTOR_ID3 \
  --voting-method majority

# Ensemble with weights
pynomaly ensemble create \
  --name "Weighted Ensemble" \
  --detectors DETECTOR_ID1,DETECTOR_ID2 \
  --weights 0.7,0.3 \
  --voting-method weighted

# Run ensemble detection
pynomaly ensemble predict \
  --ensemble ENSEMBLE_ID \
  --dataset DATASET_ID \
  --output ensemble_results.csv
```

### AutoML Operations

```bash
# Automated algorithm selection
pynomaly automl select \
  --dataset DATASET_ID \
  --metric f1_score \
  --trials 100 \
  --timeout 3600

# Hyperparameter optimization
pynomaly automl tune \
  --detector DETECTOR_ID \
  --dataset DATASET_ID \
  --metric auc_score \
  --trials 50

# Auto-feature selection
pynomaly automl features \
  --dataset DATASET_ID \
  --method recursive_elimination \
  --target-features 10
```

### Model Explainability

```bash
# Explain predictions
pynomaly explain \
  --detector DETECTOR_ID \
  --data '{"amount": 1500, "hour": 3}' \
  --method shap

# Batch explanation
pynomaly explain \
  --detector DETECTOR_ID \
  --dataset DATASET_ID \
  --method lime \
  --output explanations.json

# Feature importance
pynomaly explain importance \
  --detector DETECTOR_ID \
  --output feature_importance.csv

# Generate explanation report
pynomaly explain report \
  --detector DETECTOR_ID \
  --output explanation_report.html
```

### Data Preprocessing

```bash
# Preprocessing pipeline
pynomaly preprocess \
  --input data.csv \
  --output processed_data.csv \
  --normalize \
  --remove-outliers \
  --feature-selection auto

# Custom preprocessing
pynomaly preprocess \
  --input data.csv \
  --output processed_data.csv \
  --config preprocessing_config.yaml

# Preprocessing statistics
pynomaly preprocess stats \
  --input data.csv \
  --output preprocessing_stats.json
```

## Scripting and Automation

### Bash Scripting Examples

```bash
#!/bin/bash

# Complete anomaly detection pipeline
DATASET_ID=$(pynomaly dataset load data.csv --name "Daily Data" --output json | jq -r '.id')
DETECTOR_ID=$(pynomaly detector create --name "Daily Detector" --algorithm IsolationForest --output json | jq -r '.id')

# Train the model
pynomaly train --detector $DETECTOR_ID --dataset $DATASET_ID --validation-split 0.2

# Run predictions
pynomaly predict --detector $DETECTOR_ID --dataset $DATASET_ID --output predictions.csv

# Generate report
pynomaly results stats --detector $DETECTOR_ID --output daily_report.json

echo "Pipeline completed successfully"
```

### Automated Monitoring Script

```bash
#!/bin/bash

# Continuous monitoring script
while true; do
    # Check server health
    if ! pynomaly server status --quiet; then
        echo "Server is down, restarting..."
        pynomaly server restart
    fi
    
    # Check performance metrics
    CPU_USAGE=$(pynomaly perf system --cpu --output json | jq '.cpu_usage')
    if (( $(echo "$CPU_USAGE > 90" | bc -l) )); then
        echo "High CPU usage detected: $CPU_USAGE%"
        # Send alert or take action
    fi
    
    # Wait 5 minutes
    sleep 300
done
```

### Python Integration

```python
#!/usr/bin/env python3
import subprocess
import json
import pandas as pd

def run_pynomaly_command(command):
    """Execute Pynomaly CLI command and return JSON result."""
    cmd = f"pynomaly {command} --output json"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Command failed: {result.stderr}")
    
    return json.loads(result.stdout)

# Example: Automated batch processing
def process_daily_data(data_file, detector_id):
    # Load dataset
    dataset = run_pynomaly_command(f"dataset load {data_file} --name 'Daily Batch'")
    dataset_id = dataset['id']
    
    # Run predictions
    results = run_pynomaly_command(f"predict --detector {detector_id} --dataset {dataset_id}")
    
    # Export results
    subprocess.run(f"pynomaly results export {results['id']} --output daily_results.csv", shell=True)
    
    # Load and analyze results
    df = pd.read_csv("daily_results.csv")
    anomaly_count = df['is_anomaly'].sum()
    
    print(f"Processed {len(df)} records, found {anomaly_count} anomalies")
    
    return df

# Run daily processing
if __name__ == "__main__":
    process_daily_data("today_data.csv", "your-detector-id")
```

### CI/CD Integration

```yaml
# .github/workflows/anomaly-detection.yml
name: Daily Anomaly Detection

on:
  schedule:
    - cron: '0 2 * * *'  # Run daily at 2 AM

jobs:
  anomaly-detection:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install Pynomaly
      run: pip install pynomaly[cli]
    
    - name: Download data
      run: wget ${{ secrets.DATA_URL }} -O daily_data.csv
    
    - name: Run anomaly detection
      run: |
        pynomaly dataset load daily_data.csv --name "Daily Data"
        pynomaly predict --detector ${{ secrets.DETECTOR_ID }} --dataset daily_data.csv --output results.csv
    
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: anomaly-results
        path: results.csv
    
    - name: Send notification
      if: failure()
      run: echo "Anomaly detection pipeline failed" | mail -s "Pipeline Alert" admin@example.com
```

## Troubleshooting

### Common Issues

```bash
# Debug connection issues
pynomaly config validate --verbose
pynomaly server status --debug

# Clear cache and reset
pynomaly cache clear
pynomaly config reset --confirm

# Verbose logging for debugging
pynomaly --log-level DEBUG command

# Memory and performance issues
pynomaly perf system --memory --detailed
pynomaly perf recommendations
```

### Getting Help

```bash
# Command-specific help
pynomaly COMMAND --help

# List all available commands
pynomaly --help

# Show examples for a command
pynomaly COMMAND --examples

# Check CLI version and dependencies
pynomaly version --dependencies
```

## Best Practices

### 1. Configuration Management
- Use environment-specific configurations
- Store sensitive data in environment variables
- Validate configuration before deployment

### 2. Performance Optimization
- Monitor resource usage regularly
- Use batch processing for large datasets
- Implement proper error handling

### 3. Automation
- Create reusable scripts for common workflows
- Use CI/CD pipelines for production deployments
- Implement monitoring and alerting

### 4. Security
- Use secure connection strings
- Implement proper authentication
- Regular security audits

This comprehensive CLI reference provides complete coverage of all Pynomaly command-line operations with practical examples and best practices for both interactive and automated usage.
