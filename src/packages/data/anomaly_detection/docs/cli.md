# CLI User Guide

This comprehensive guide covers all command-line interface features of the Anomaly Detection package, including commands, options, workflows, and automation patterns.

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Basic Commands](#basic-commands)
4. [Detection Commands](#detection-commands)
5. [Ensemble Commands](#ensemble-commands)
6. [Streaming Commands](#streaming-commands)
7. [Model Management](#model-management)
8. [Data Commands](#data-commands)
9. [Worker Commands](#worker-commands)
10. [Configuration Commands](#configuration-commands)
11. [Utility Commands](#utility-commands)
12. [Advanced Workflows](#advanced-workflows)
13. [Automation & Scripting](#automation--scripting)
14. [Best Practices](#best-practices)

## Overview

The Anomaly Detection CLI provides a complete command-line interface for:

- **Detection Operations**: Run anomaly detection on various data sources
- **Model Management**: Train, save, load, and manage detection models
- **Streaming Processing**: Real-time anomaly detection from streams
- **Ensemble Methods**: Combine multiple algorithms for better accuracy
- **Data Processing**: Load, validate, and preprocess data
- **System Management**: Start services, manage workers, monitor health

### Command Structure

All commands follow the pattern:
```bash
anomaly-detection [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

### Global Options

```bash
--config, -c PATH       Configuration file path
--verbose, -v           Verbose output (can be repeated: -vv, -vvv)
--quiet, -q            Suppress output
--log-level LEVEL      Set logging level (DEBUG, INFO, WARN, ERROR)
--help, -h             Show help message
--version              Show version information
```

## Installation & Setup

### CLI Installation

```bash
# Install with CLI support
pip install anomaly-detection[cli]

# Verify installation
anomaly-detection --version
anomaly-detection --help
```

### Shell Completion

```bash
# Bash completion
eval "$(_ANOMALY_DETECTION_COMPLETE=bash_source anomaly-detection)"

# Zsh completion
eval "$(_ANOMALY_DETECTION_COMPLETE=zsh_source anomaly-detection)"

# Fish completion
eval (env _ANOMALY_DETECTION_COMPLETE=fish_source anomaly-detection)
```

### Configuration Setup

```bash
# Create default configuration
anomaly-detection config init

# Edit configuration
anomaly-detection config edit

# Validate configuration
anomaly-detection config validate

# Show current configuration
anomaly-detection config show
```

## Basic Commands

### Help and Information

```bash
# General help
anomaly-detection --help

# Command-specific help
anomaly-detection detect --help
anomaly-detection ensemble --help

# List all available commands
anomaly-detection --help | grep "Commands:"

# Show version and build info
anomaly-detection --version
anomaly-detection version --detailed
```

### Health Checks

```bash
# Check system health
anomaly-detection health

# Detailed health check
anomaly-detection health --detailed

# Check specific components
anomaly-detection health --check database
anomaly-detection health --check redis
anomaly-detection health --check algorithms
```

### Algorithm Information

```bash
# List all available algorithms
anomaly-detection algorithms list

# Get algorithm details
anomaly-detection algorithms info --name isolation_forest
anomaly-detection algorithms info --name local_outlier_factor

# List algorithms by category
anomaly-detection algorithms list --category pyod
anomaly-detection algorithms list --category sklearn
anomaly-detection algorithms list --category deep_learning

# Show algorithm parameters
anomaly-detection algorithms params --name isolation_forest
```

## Detection Commands

### Basic Detection

```bash
# Detect anomalies in CSV file
anomaly-detection detect run \
    --input data.csv \
    --algorithm isolation_forest \
    --output results.json

# With custom parameters
anomaly-detection detect run \
    --input data.csv \
    --algorithm local_outlier_factor \
    --contamination 0.05 \
    --param n_neighbors=20 \
    --param metric=euclidean \
    --output results.json
```

### Advanced Detection Options

```bash
# Multiple input files
anomaly-detection detect run \
    --input data1.csv data2.csv data3.csv \
    --algorithm isolation_forest \
    --merge-strategy concat \
    --output combined_results.json

# Specify feature columns
anomaly-detection detect run \
    --input data.csv \
    --features temperature humidity pressure \
    --algorithm isolation_forest \
    --output results.json

# Custom contamination and thresholds
anomaly-detection detect run \
    --input data.csv \
    --algorithm isolation_forest \
    --contamination 0.1 \
    --threshold-method percentile \
    --threshold-value 95 \
    --output results.json

# Preprocessing options
anomaly-detection detect run \
    --input data.csv \
    --algorithm local_outlier_factor \
    --preprocess \
    --scaling standard \
    --handle-missing mean \
    --remove-duplicates \
    --output results.json
```

### Batch Detection

```bash
# Process multiple files in batch
anomaly-detection detect batch \
    --input-dir data/ \
    --pattern "*.csv" \
    --algorithm isolation_forest \
    --output-dir results/ \
    --parallel 4

# Batch with custom naming
anomaly-detection detect batch \
    --input-dir sensors/ \
    --pattern "sensor_*.csv" \
    --algorithm local_outlier_factor \
    --output-template "anomalies_{basename}.json" \
    --summary results_summary.json
```

### Detection with Different Data Formats

```bash
# JSON input
anomaly-detection detect run \
    --input data.json \
    --data-key features \
    --algorithm isolation_forest \
    --output results.json

# Parquet input
anomaly-detection detect run \
    --input data.parquet \
    --algorithm isolation_forest \
    --output results.json

# Database query
anomaly-detection detect run \
    --input "postgresql://user:pass@localhost/db" \
    --query "SELECT * FROM sensor_data WHERE date >= '2024-01-01'" \
    --algorithm isolation_forest \
    --output results.json

# API endpoint
anomaly-detection detect run \
    --input https://api.example.com/data \
    --headers "Authorization: Bearer token123" \
    --algorithm isolation_forest \
    --output results.json
```

### Output Formats

```bash
# JSON output (default)
anomaly-detection detect run \
    --input data.csv \
    --algorithm isolation_forest \
    --output results.json

# CSV output
anomaly-detection detect run \
    --input data.csv \
    --algorithm isolation_forest \
    --output results.csv \
    --format csv

# Detailed output with explanations
anomaly-detection detect run \
    --input data.csv \
    --algorithm isolation_forest \
    --output results.json \
    --include-explanations \
    --include-feature-importance

# Summary statistics
anomaly-detection detect run \
    --input data.csv \
    --algorithm isolation_forest \
    --output results.json \
    --summary summary.txt
```

## Ensemble Commands

### Basic Ensemble Detection

```bash
# Simple ensemble with majority voting
anomaly-detection ensemble run \
    --input data.csv \
    --algorithms isolation_forest local_outlier_factor one_class_svm \
    --method majority \
    --output ensemble_results.json

# Weighted ensemble
anomaly-detection ensemble run \
    --input data.csv \
    --algorithms isolation_forest local_outlier_factor \
    --method weighted \
    --weights 0.6 0.4 \
    --output ensemble_results.json
```

### Advanced Ensemble Methods

```bash
# Stacking ensemble
anomaly-detection ensemble run \
    --input train_data.csv \
    --test-input test_data.csv \
    --algorithms isolation_forest local_outlier_factor copod \
    --method stacking \
    --meta-algorithm logistic_regression \
    --cv-folds 5 \
    --output stacking_results.json

# Average ensemble with custom parameters
anomaly-detection ensemble run \
    --input data.csv \
    --config ensemble_config.yaml \
    --method average \
    --output ensemble_results.json
```

### Ensemble Configuration File

```yaml
# ensemble_config.yaml
ensemble:
  algorithms:
    - name: isolation_forest
      weight: 0.4
      params:
        n_estimators: 100
        contamination: 0.1
    - name: local_outlier_factor
      weight: 0.3
      params:
        n_neighbors: 20
        contamination: 0.1
    - name: one_class_svm
      weight: 0.3
      params:
        kernel: rbf
        nu: 0.1
  method: weighted
  require_unanimous: false
```

```bash
# Use configuration file
anomaly-detection ensemble run \
    --input data.csv \
    --config ensemble_config.yaml \
    --output results.json
```

### Ensemble Evaluation

```bash
# Compare ensemble methods
anomaly-detection ensemble compare \
    --input data.csv \
    --labels labels.csv \
    --algorithms isolation_forest local_outlier_factor copod \
    --methods majority average weighted stacking \
    --output comparison.json

# Cross-validation evaluation
anomaly-detection ensemble evaluate \
    --input data.csv \
    --labels labels.csv \
    --algorithms isolation_forest local_outlier_factor \
    --method majority \
    --cv-folds 5 \
    --metrics precision recall f1 auc \
    --output evaluation.json
```

## Streaming Commands

### Stream Processing

```bash
# Process Kafka stream
anomaly-detection stream process \
    --source kafka://localhost:9092/sensor-data \
    --algorithm isolation_forest \
    --window-size 1000 \
    --output-topic anomaly-alerts \
    --consumer-group anomaly-detectors

# Process Redis stream
anomaly-detection stream process \
    --source redis://localhost:6379/sensor-stream \
    --algorithm local_outlier_factor \
    --window-size 500 \
    --output-stream alerts

# Process file stream (simulated)
anomaly-detection stream simulate \
    --input data.csv \
    --algorithm isolation_forest \
    --batch-size 10 \
    --delay 1.0 \
    --output stream_results.jsonl
```

### Stream Monitoring

```bash
# Monitor stream with real-time display
anomaly-detection stream monitor \
    --source kafka://localhost:9092/sensor-data \
    --algorithm isolation_forest \
    --window-size 1000 \
    --dashboard \
    --alert-webhook https://alerts.example.com/webhook

# Monitor with concept drift detection
anomaly-detection stream monitor \
    --source kafka://localhost:9092/sensor-data \
    --algorithm ensemble \
    --algorithms isolation_forest local_outlier_factor \
    --drift-detection \
    --auto-retrain \
    --output-dir monitoring_logs/
```

### Stream Configuration

```bash
# Generate stream configuration
anomaly-detection stream config init \
    --source-type kafka \
    --algorithm isolation_forest \
    --output stream_config.yaml

# Validate stream configuration
anomaly-detection stream config validate \
    --config stream_config.yaml

# Test stream connection
anomaly-detection stream test \
    --source kafka://localhost:9092/test-topic \
    --samples 100
```

## Model Management

### Training Models

```bash
# Train and save model
anomaly-detection train \
    --input training_data.csv \
    --algorithm isolation_forest \
    --save-model models/detector_v1.pkl \
    --model-name "Production Detector v1"

# Train with validation
anomaly-detection train \
    --input training_data.csv \
    --validation-input validation_data.csv \
    --algorithm local_outlier_factor \
    --save-model models/lof_detector.pkl \
    --validation-metrics precision recall f1
```

### Using Saved Models

```bash
# Predict with saved model
anomaly-detection predict \
    --input new_data.csv \
    --model models/detector_v1.pkl \
    --output predictions.json

# Batch prediction
anomaly-detection predict batch \
    --input-dir new_data/ \
    --model models/detector_v1.pkl \
    --output-dir predictions/ \
    --parallel 4
```

### Model Registry

```bash
# List saved models
anomaly-detection models list

# Show model details
anomaly-detection models info --model-id detector_v1

# Compare models
anomaly-detection models compare \
    --models models/detector_v1.pkl models/detector_v2.pkl \
    --test-data test_data.csv \
    --metrics accuracy precision recall

# Export model metadata
anomaly-detection models export \
    --model models/detector_v1.pkl \
    --output model_metadata.json

# Validate model
anomaly-detection models validate \
    --model models/detector_v1.pkl \
    --test-data validation_data.csv
```

### Model Versioning

```bash
# Create model version
anomaly-detection models version create \
    --model models/detector.pkl \
    --version 2.0 \
    --description "Updated with new training data"

# List model versions
anomaly-detection models version list --model-name detector

# Rollback to previous version
anomaly-detection models version rollback \
    --model-name detector \
    --version 1.0

# Tag model version
anomaly-detection models version tag \
    --model-name detector \
    --version 2.0 \
    --tag production
```

## Data Commands

### Data Validation

```bash
# Validate data file
anomaly-detection data validate \
    --input data.csv \
    --schema schema.json \
    --output validation_report.json

# Check data quality
anomaly-detection data quality \
    --input data.csv \
    --output quality_report.json \
    --checks missing_values duplicates outliers

# Data profiling
anomaly-detection data profile \
    --input data.csv \
    --output profile.html \
    --include-distributions \
    --include-correlations
```

### Data Preprocessing

```bash
# Preprocess data
anomaly-detection data preprocess \
    --input raw_data.csv \
    --output processed_data.csv \
    --scaling standard \
    --handle-missing mean \
    --remove-duplicates \
    --remove-outliers

# Feature engineering
anomaly-detection data features \
    --input data.csv \
    --output features.csv \
    --polynomial-degree 2 \
    --interaction-features \
    --statistical-features mean std skew kurtosis
```

### Data Conversion

```bash
# Convert between formats
anomaly-detection data convert \
    --input data.csv \
    --output data.parquet \
    --format parquet

# Extract from database
anomaly-detection data extract \
    --source "postgresql://user:pass@localhost/db" \
    --query "SELECT * FROM sensor_data" \
    --output extracted_data.csv

# Sample data
anomaly-detection data sample \
    --input large_data.csv \
    --output sample.csv \
    --method random \
    --size 10000
```

## Worker Commands

### Starting Workers

```bash
# Start background worker
anomaly-detection worker start \
    --queue anomaly-jobs \
    --concurrency 4 \
    --log-level INFO

# Start worker with specific algorithms
anomaly-detection worker start \
    --queue anomaly-jobs \
    --algorithms isolation_forest local_outlier_factor \
    --max-memory 2G \
    --timeout 600

# Start worker cluster
anomaly-detection worker cluster start \
    --workers 8 \
    --queue anomaly-jobs \
    --redis-url redis://localhost:6379
```

### Managing Workers

```bash
# List active workers
anomaly-detection worker list

# Show worker status
anomaly-detection worker status --worker-id worker-001

# Stop worker gracefully
anomaly-detection worker stop --worker-id worker-001

# Stop all workers
anomaly-detection worker stop --all

# Restart worker
anomaly-detection worker restart --worker-id worker-001
```

### Job Management

```bash
# Submit detection job
anomaly-detection jobs submit \
    --input data.csv \
    --algorithm isolation_forest \
    --priority high \
    --callback-url https://api.example.com/results

# List jobs
anomaly-detection jobs list --status pending
anomaly-detection jobs list --user current --limit 10

# Get job status
anomaly-detection jobs status --job-id job-12345

# Cancel job
anomaly-detection jobs cancel --job-id job-12345

# Get job results
anomaly-detection jobs results --job-id job-12345 --output results.json
```

## Configuration Commands

### Configuration Management

```bash
# Initialize configuration
anomaly-detection config init

# Show current configuration
anomaly-detection config show

# Edit configuration
anomaly-detection config edit

# Validate configuration
anomaly-detection config validate

# Set specific values
anomaly-detection config set algorithms.default isolation_forest
anomaly-detection config set server.port 8001

# Get specific values
anomaly-detection config get algorithms.default
anomaly-detection config get server
```

### Environment Management

```bash
# List environments
anomaly-detection config env list

# Switch environment
anomaly-detection config env use production

# Create new environment
anomaly-detection config env create staging \
    --copy-from development

# Export environment configuration
anomaly-detection config env export production \
    --output production_config.yaml
```

### Secrets Management

```bash
# Set secrets
anomaly-detection config secrets set DATABASE_PASSWORD
anomaly-detection config secrets set API_KEY --from-file api_key.txt

# List secrets (names only)
anomaly-detection config secrets list

# Remove secret
anomaly-detection config secrets remove OLD_API_KEY
```

## Utility Commands

### Benchmarking

```bash
# Benchmark algorithms
anomaly-detection benchmark run \
    --input data.csv \
    --algorithms isolation_forest local_outlier_factor one_class_svm \
    --metrics accuracy precision recall f1 processing_time \
    --output benchmark_results.json

# Performance profiling
anomaly-detection profile \
    --input data.csv \
    --algorithm isolation_forest \
    --output profile_report.html \
    --include-memory \
    --include-cpu
```

### Testing

```bash
# Run system tests
anomaly-detection test system

# Test specific algorithm
anomaly-detection test algorithm --name isolation_forest

# Test with custom data
anomaly-detection test algorithm \
    --name local_outlier_factor \
    --test-data test_data.csv \
    --expected-results expected.json

# Load testing
anomaly-detection test load \
    --input data.csv \
    --algorithm isolation_forest \
    --concurrent-requests 100 \
    --duration 60
```

### Debugging

```bash
# Debug detection process
anomaly-detection debug detect \
    --input data.csv \
    --algorithm isolation_forest \
    --verbose \
    --save-intermediate debug_output/

# Analyze model behavior
anomaly-detection debug model \
    --model models/detector.pkl \
    --input data.csv \
    --explain \
    --visualize debug_plots/

# Debug configuration
anomaly-detection debug config \
    --check-all \
    --fix-issues
```

## Advanced Workflows

### Automated Pipeline

```bash
#!/bin/bash
# automated_detection_pipeline.sh

# Set configuration
export ANOMALY_DETECTION_CONFIG="production.yaml"

# Validate data
anomaly-detection data validate \
    --input daily_data.csv \
    --schema data_schema.json

# Preprocess data
anomaly-detection data preprocess \
    --input daily_data.csv \
    --output processed_data.csv \
    --config preprocessing.yaml

# Run ensemble detection
anomaly-detection ensemble run \
    --input processed_data.csv \
    --config ensemble_config.yaml \
    --output anomaly_results.json

# Generate alerts for high-confidence anomalies
anomaly-detection alerts generate \
    --input anomaly_results.json \
    --confidence-threshold 0.8 \
    --output alerts.json \
    --webhook https://alerts.company.com/webhook

# Update model if needed
if [ -f retrain_trigger.flag ]; then
    anomaly-detection train \
        --input processed_data.csv \
        --algorithm isolation_forest \
        --save-model models/detector_$(date +%Y%m%d).pkl \
        --replace-production
    rm retrain_trigger.flag
fi
```

### Continuous Monitoring

```bash
#!/bin/bash
# continuous_monitoring.sh

while true; do
    # Process latest data
    anomaly-detection stream process \
        --source kafka://localhost:9092/sensor-data \
        --algorithm isolation_forest \
        --window-size 1000 \
        --output-topic anomaly-alerts \
        --duration 3600  # 1 hour
    
    # Check for concept drift
    if anomaly-detection stream drift-check \
        --source kafka://localhost:9092/sensor-data \
        --reference-model models/current_model.pkl \
        --threshold 0.1; then
        
        echo "Concept drift detected, retraining model..."
        
        # Retrain model
        anomaly-detection train \
            --input-stream kafka://localhost:9092/sensor-data \
            --samples 10000 \
            --algorithm isolation_forest \
            --save-model models/model_$(date +%Y%m%d_%H%M).pkl \
            --replace-current
    fi
    
    # Health check
    anomaly-detection health --detailed || {
        echo "Health check failed, restarting services..."
        anomaly-detection worker restart --all
    }
    
    sleep 300  # 5 minutes
done
```

### A/B Testing Workflow

```bash
#!/bin/bash
# ab_testing_workflow.sh

# Deploy model A
anomaly-detection models deploy \
    --model models/model_a.pkl \
    --name model_a \
    --traffic-split 50

# Deploy model B
anomaly-detection models deploy \
    --model models/model_b.pkl \
    --name model_b \
    --traffic-split 50

# Monitor performance
anomaly-detection models monitor \
    --models model_a model_b \
    --metrics precision recall f1 latency \
    --duration 86400  # 24 hours

# Analyze results
anomaly-detection models compare \
    --models model_a model_b \
    --metrics-file monitoring_results.json \
    --significance-test \
    --output ab_test_results.json

# Promote winner
WINNER=$(jq -r '.winner' ab_test_results.json)
anomaly-detection models promote \
    --model $WINNER \
    --to production \
    --traffic-split 100
```

## Automation & Scripting

### Cron Jobs

```bash
# Daily anomaly detection report
0 2 * * * /usr/local/bin/anomaly-detection detect batch \
    --input-dir /data/daily/ \
    --algorithm isolation_forest \
    --output-dir /reports/anomalies/ \
    --email-report admin@company.com

# Weekly model retraining
0 3 * * 0 /usr/local/bin/anomaly-detection train \
    --input /data/weekly/training_data.csv \
    --algorithm isolation_forest \
    --save-model /models/weekly_model.pkl \
    --replace-production \
    --notify-webhook https://hooks.company.com/model-updated

# Monthly performance benchmark
0 4 1 * * /usr/local/bin/anomaly-detection benchmark run \
    --input /data/benchmark/test_data.csv \
    --algorithms isolation_forest local_outlier_factor \
    --output /reports/benchmarks/$(date +%Y%m)_benchmark.json
```

### GitHub Actions Workflow

```yaml
# .github/workflows/anomaly-detection.yml
name: Anomaly Detection Pipeline

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  detect-anomalies:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install anomaly-detection[cli]
    
    - name: Download data
      run: |
        aws s3 cp s3://data-bucket/daily/ data/ --recursive
    
    - name: Run anomaly detection
      run: |
        anomaly-detection detect batch \
          --input-dir data/ \
          --algorithm isolation_forest \
          --output-dir results/ \
          --parallel 4
    
    - name: Upload results
      run: |
        aws s3 cp results/ s3://results-bucket/$(date +%Y%m%d)/ --recursive
    
    - name: Send notifications
      if: failure()
      run: |
        curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
          -H 'Content-type: application/json' \
          --data '{"text":"Anomaly detection pipeline failed"}'
```

### Docker Compose Automation

```yaml
# docker-compose.yml
version: '3.8'

services:
  anomaly-detector:
    image: anomaly-detection:latest
    command: >
      anomaly-detection stream process
      --source kafka://kafka:9092/sensor-data
      --algorithm isolation_forest
      --output-topic anomaly-alerts
    environment:
      - ANOMALY_DETECTION_CONFIG=/config/production.yaml
    volumes:
      - ./config:/config
      - ./models:/models
    depends_on:
      - kafka
      - redis
    restart: unless-stopped
  
  model-trainer:
    image: anomaly-detection:latest
    command: >
      sh -c "
      while true; do
        anomaly-detection train
        --input-stream kafka://kafka:9092/training-data
        --algorithm isolation_forest
        --save-model /models/model_\$(date +%Y%m%d).pkl
        --schedule daily
        sleep 86400
      done
      "
    volumes:
      - ./models:/models
    depends_on:
      - kafka
    restart: unless-stopped
  
  health-monitor:
    image: anomaly-detection:latest
    command: >
      anomaly-detection monitor
      --services anomaly-detector model-trainer
      --alert-webhook http://alertmanager:9093/api/v1/alerts
    restart: unless-stopped
```

### Kubernetes CronJobs

```yaml
# k8s-cronjobs.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-anomaly-detection
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: anomaly-detector
            image: anomaly-detection:latest
            command:
            - anomaly-detection
            - detect
            - batch
            - --input-dir
            - /data
            - --algorithm
            - isolation_forest
            - --output-dir
            - /results
            env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access-key-id
            volumeMounts:
            - name: data-volume
              mountPath: /data
            - name: results-volume
              mountPath: /results
          volumes:
          - name: data-volume
            persistentVolumeClaim:
              claimName: data-pvc
          - name: results-volume
            persistentVolumeClaim:
              claimName: results-pvc
          restartPolicy: OnFailure
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: weekly-model-training
spec:
  schedule: "0 3 * * 0"  # Weekly on Sunday at 3 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: model-trainer
            image: anomaly-detection:latest
            command:
            - anomaly-detection
            - train
            - --input
            - /data/training_data.csv
            - --algorithm
            - isolation_forest
            - --save-model
            - /models/weekly_model.pkl
            - --replace-production
            volumeMounts:
            - name: data-volume
              mountPath: /data
            - name: models-volume
              mountPath: /models
          restartPolicy: OnFailure
```

## Best Practices

### 1. Configuration Management

```bash
# Use environment-specific configurations
anomaly-detection --config configs/development.yaml detect run --input data.csv
anomaly-detection --config configs/production.yaml detect run --input data.csv

# Validate configuration before deployment
anomaly-detection config validate --config configs/production.yaml
```

### 2. Error Handling

```bash
#!/bin/bash
# robust_detection_script.sh

set -e  # Exit on error

# Function to handle errors
handle_error() {
    echo "Error occurred in detection pipeline"
    # Send alert
    curl -X POST "$ALERT_WEBHOOK" -d '{"error": "Detection pipeline failed"}'
    exit 1
}

trap handle_error ERR

# Run detection with retry logic
for i in {1..3}; do
    if anomaly-detection detect run \
        --input data.csv \
        --algorithm isolation_forest \
        --output results.json \
        --timeout 300; then
        break
    else
        echo "Attempt $i failed, retrying..."
        sleep 10
    fi
done
```

### 3. Resource Management

```bash
# Limit memory usage
anomaly-detection detect run \
    --input large_data.csv \
    --algorithm isolation_forest \
    --max-memory 2G \
    --batch-size 1000

# Parallel processing with resource limits
anomaly-detection detect batch \
    --input-dir data/ \
    --algorithm isolation_forest \
    --parallel 4 \
    --max-memory-per-job 1G
```

### 4. Monitoring and Logging

```bash
# Enable detailed logging
anomaly-detection --verbose detect run \
    --input data.csv \
    --algorithm isolation_forest \
    --log-file detection.log \
    --log-level DEBUG

# Monitor resource usage
anomaly-detection detect run \
    --input data.csv \
    --algorithm isolation_forest \
    --monitor-resources \
    --metrics-output metrics.json
```

### 5. Data Security

```bash
# Use environment variables for sensitive data
export DATABASE_URL="postgresql://user:$DB_PASSWORD@localhost/db"

anomaly-detection detect run \
    --input "$DATABASE_URL" \
    --query "SELECT * FROM sensitive_data" \
    --algorithm isolation_forest \
    --encrypt-output \
    --output encrypted_results.json.enc

# Secure model storage
anomaly-detection train \
    --input training_data.csv \
    --algorithm isolation_forest \
    --save-model s3://secure-bucket/models/detector.pkl \
    --encrypt \
    --kms-key-id arn:aws:kms:region:account:key/key-id
```

### 6. Performance Optimization

```bash
# Optimize for speed
anomaly-detection detect run \
    --input data.csv \
    --algorithm hbos \
    --fast-mode \
    --cache-models \
    --output results.json

# Optimize for accuracy
anomaly-detection ensemble run \
    --input data.csv \
    --algorithms isolation_forest local_outlier_factor copod \
    --method stacking \
    --cv-folds 10 \
    --output results.json
```

This comprehensive CLI guide provides everything needed to effectively use the Anomaly Detection package from the command line, including basic usage, advanced workflows, and production automation patterns.