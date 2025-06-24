# Pynomaly Autonomous Mode Implementation Guide

## Quick Start with Enhanced Features

### 1. Basic Autonomous Detection

```bash
# Simple autonomous detection
pynomaly auto detect data.csv

# With preprocessing and quality assessment
pynomaly auto detect data.csv --preprocess --quality-threshold 0.8

# Save results with specific format
pynomaly auto detect data.csv --output results.xlsx --format excel
```

### 2. Enhanced All-Classifier Testing

```bash
# Test ALL compatible classifiers
pynomaly auto detect-all data.csv --confidence 0.6 --ensemble

# Extended testing with analysis
pynomaly auto detect-all data.csv --max-time 3600 --verbose --output comprehensive_results.json
```

### 3. Family-Based Ensemble Detection

```bash
# Test specific algorithm families
pynomaly auto detect-by-family data.csv --family statistical distance_based isolation_based

# Full hierarchical ensemble approach
pynomaly auto detect-by-family data.csv \
  --family statistical distance_based neural_networks \
  --family-ensemble \
  --meta-ensemble \
  --output family_results.csv
```

### 4. Algorithm Choice Explanations

```bash
# Get detailed explanations for algorithm choices
pynomaly auto explain-choices data.csv --alternatives --save

# Analyze specific dataset characteristics
pynomaly auto explain-choices data.csv --max-algorithms 10 --save-explanation
```

### 5. Results Analysis

```bash
# Comprehensive analysis of detection results
pynomaly auto analyze-results results.csv --type comprehensive --interactive

# Statistical analysis only
pynomaly auto analyze-results results.csv --type statistical --output analysis_report.json
```

## API Usage Examples

### 1. Autonomous Detection via API

```python
import requests
import json

# Upload file for autonomous detection
with open('data.csv', 'rb') as f:
    files = {'file': f}
    data = {
        'max_algorithms': 10,
        'confidence_threshold': 0.7,
        'auto_tune': True,
        'enable_preprocessing': True
    }
    
    response = requests.post(
        'http://localhost:8000/api/autonomous/detect',
        files=files,
        data=data
    )
    
    results = response.json()
    print(f"Detection successful: {results['success']}")
    print(f"Best algorithm: {results['results']['autonomous_detection_results']['best_algorithm']}")
```

### 2. AutoML Optimization

```python
# AutoML optimization for existing dataset
automl_request = {
    "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
    "objective": "AUC",
    "max_algorithms": 8,
    "optimization_time": 1800,
    "enable_ensemble": True
}

response = requests.post(
    'http://localhost:8000/api/autonomous/automl/optimize',
    json=automl_request
)

result = response.json()
print(f"Best algorithm: {result['automl_result']['best_algorithm']}")
print(f"Best score: {result['automl_result']['best_score']}")
print(f"Optimization time: {result['automl_result']['optimization_time']}s")
```

### 3. Family-Based Ensemble Creation

```python
# Create hierarchical family-based ensemble
family_request = {
    "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
    "families": ["statistical", "distance_based", "isolation_based"],
    "enable_family_ensembles": True,
    "enable_meta_ensemble": True,
    "optimization_time": 2400
}

response = requests.post(
    'http://localhost:8000/api/autonomous/ensemble/create-by-family',
    json=family_request
)

result = response.json()
for family, data in result['family_results'].items():
    print(f"{family}: {data['best_algorithm']} (score: {data['best_score']:.3f})")
```

### 4. Algorithm Choice Explanations

```python
# Get explanations for algorithm choices
with open('data.csv', 'rb') as f:
    files = {'data_file': f}
    data = {
        'max_algorithms': 5,
        'include_alternatives': True,
        'include_data_analysis': True
    }
    
    response = requests.post(
        'http://localhost:8000/api/autonomous/explain/choices',
        files=files,
        data=data
    )
    
    explanations = response.json()['explanations']
    
    # Display top recommendation
    top_rec = explanations['algorithm_recommendations'][0]
    print(f"Top Recommendation: {top_rec['algorithm']}")
    print(f"Confidence: {top_rec['confidence']:.1%}")
    print(f"Reasoning: {top_rec['reasoning']}")
```

## Python Script Integration

### 1. Direct Service Usage

```python
from pynomaly.application.services.autonomous_service import (
    AutonomousDetectionService, 
    AutonomousConfig
)
from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
from pynomaly.presentation.cli.container import get_cli_container

# Setup
container = get_cli_container()
data_loaders = {"csv": CSVLoader()}

service = AutonomousDetectionService(
    detector_repository=container.detector_repository(),
    result_repository=container.result_repository(),
    data_loaders=data_loaders
)

# Configure for comprehensive testing
config = AutonomousConfig(
    max_algorithms=15,
    confidence_threshold=0.6,
    auto_tune_hyperparams=True,
    enable_preprocessing=True,
    verbose=True
)

# Run autonomous detection
import asyncio
results = asyncio.run(service.detect_autonomous("data.csv", config))

# Access results
auto_results = results["autonomous_detection_results"]
best_algorithm = auto_results["best_algorithm"]
detection_results = auto_results["detection_results"]

print(f"Best performing algorithm: {best_algorithm}")
for algo, result in detection_results.items():
    print(f"{algo}: {result['anomalies_found']} anomalies ({result['anomaly_rate']:.1%})")
```

### 2. AutoML Service Integration

```python
from pynomaly.application.services.automl_service import AutoMLService, OptimizationObjective

# Create AutoML service
automl_service = AutoMLService(
    detector_repository=container.detector_repository(),
    dataset_repository=container.dataset_repository(),
    adapter_registry=container.adapter_registry(),
    max_optimization_time=3600,
    n_trials=200
)

# Profile dataset
dataset_id = "your-dataset-id"
profile = asyncio.run(automl_service.profile_dataset(dataset_id))

print(f"Dataset complexity: {profile.complexity_score:.2f}")
print(f"Recommended contamination: {profile.contamination_estimate:.1%}")

# Get algorithm recommendations
recommendations = automl_service.recommend_algorithms(profile, max_algorithms=8)

for rec in recommendations:
    print(f"{rec}: Confidence {recommendations[rec]:.1%}")

# Run full AutoML optimization
automl_result = asyncio.run(automl_service.auto_select_and_optimize(
    dataset_id=dataset_id,
    objective=OptimizationObjective.AUC,
    max_algorithms=5,
    enable_ensemble=True
))

print(f"Best algorithm: {automl_result.best_algorithm}")
print(f"Best parameters: {automl_result.best_params}")
print(f"Optimization completed in {automl_result.optimization_time:.1f}s")
```

## Advanced Configuration Examples

### 1. Custom Algorithm Selection Strategy

```python
# Custom configuration for large datasets
large_dataset_config = AutonomousConfig(
    max_samples_analysis=50000,  # Analyze more samples
    confidence_threshold=0.75,   # Higher confidence requirement
    max_algorithms=8,            # Test more algorithms
    auto_tune_hyperparams=True,
    max_preprocessing_time=600,  # Allow more preprocessing time
    preprocessing_strategy="aggressive"  # More thorough preprocessing
)

# Custom configuration for quick analysis
quick_config = AutonomousConfig(
    max_samples_analysis=5000,   # Smaller sample for speed
    confidence_threshold=0.6,    # Lower confidence for more options
    max_algorithms=3,            # Test fewer algorithms
    auto_tune_hyperparams=False, # Skip tuning for speed
    preprocessing_strategy="minimal"
)
```

### 2. Family-Specific Optimization

```python
# Statistical methods focus
statistical_families = ["statistical"]
statistical_config = AutonomousConfig(
    max_algorithms=5,
    confidence_threshold=0.7
)

# Neural network focus for complex data
neural_families = ["neural_networks"]
neural_config = AutonomousConfig(
    max_algorithms=3,
    confidence_threshold=0.8,
    auto_tune_hyperparams=True
)

# Multi-family approach
all_families = ["statistical", "distance_based", "isolation_based", "neural_networks"]
comprehensive_config = AutonomousConfig(
    max_algorithms=12,
    confidence_threshold=0.6,
    auto_tune_hyperparams=True
)
```

## Performance Monitoring

### 1. Algorithm Performance Tracking

```python
def analyze_algorithm_performance(results):
    """Analyze performance across algorithms."""
    detection_results = results["autonomous_detection_results"]["detection_results"]
    
    performance_metrics = {}
    for algo, result in detection_results.items():
        performance_metrics[algo] = {
            'execution_time': result['execution_time_ms'],
            'anomaly_rate': result['anomaly_rate'],
            'anomalies_found': result['anomalies_found'],
            'efficiency_score': result['anomalies_found'] / (result['execution_time_ms'] / 1000)
        }
    
    # Sort by efficiency
    sorted_algos = sorted(
        performance_metrics.items(), 
        key=lambda x: x[1]['efficiency_score'], 
        reverse=True
    )
    
    print("Algorithm Performance Ranking:")
    for i, (algo, metrics) in enumerate(sorted_algos, 1):
        print(f"{i}. {algo}: {metrics['anomalies_found']} anomalies in {metrics['execution_time']}ms")
    
    return performance_metrics

# Usage
results = asyncio.run(service.detect_autonomous("data.csv", config))
performance = analyze_algorithm_performance(results)
```

### 2. Ensemble Effectiveness Analysis

```python
def compare_ensemble_vs_individual(ensemble_result, individual_results):
    """Compare ensemble performance to individual algorithms."""
    
    ensemble_anomalies = ensemble_result['n_anomalies']
    ensemble_rate = ensemble_result['anomaly_rate']
    
    individual_rates = [r['anomaly_rate'] for r in individual_results.values()]
    avg_individual_rate = sum(individual_rates) / len(individual_rates)
    
    print(f"Ensemble anomaly rate: {ensemble_rate:.1%}")
    print(f"Average individual rate: {avg_individual_rate:.1%}")
    print(f"Ensemble improvement: {((ensemble_rate - avg_individual_rate) / avg_individual_rate * 100):+.1f}%")
    
    return {
        'ensemble_rate': ensemble_rate,
        'individual_avg': avg_individual_rate,
        'improvement': (ensemble_rate - avg_individual_rate) / avg_individual_rate
    }
```

## Troubleshooting Common Issues

### 1. Algorithm Selection Issues

```python
# Debug algorithm selection
def debug_algorithm_selection(profile, recommendations):
    """Debug why certain algorithms were/weren't selected."""
    
    print(f"Dataset characteristics:")
    print(f"  Samples: {profile.n_samples:,}")
    print(f"  Features: {profile.n_features}")
    print(f"  Complexity: {profile.complexity_score:.2f}")
    print(f"  Missing data: {profile.missing_values_ratio:.1%}")
    
    print(f"\nAlgorithm recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec.algorithm} (confidence: {rec.confidence:.1%})")
        print(f"     Reasoning: {rec.reasoning}")
        
    # Check for common issues
    if profile.n_samples < 1000:
        print("\n⚠️  Small dataset - consider neural networks may be excluded")
    
    if profile.missing_values_ratio > 0.2:
        print("\n⚠️  High missing data - preprocessing recommended")
    
    if profile.complexity_score > 0.8:
        print("\n⚠️  Complex dataset - consider neural networks or ensembles")
```

### 2. Performance Optimization

```python
# Optimize for speed
speed_config = AutonomousConfig(
    max_samples_analysis=5000,      # Smaller sample
    max_algorithms=3,               # Fewer algorithms
    auto_tune_hyperparams=False,    # Skip tuning
    enable_preprocessing=False      # Skip preprocessing
)

# Optimize for accuracy
accuracy_config = AutonomousConfig(
    max_samples_analysis=25000,     # Larger sample
    max_algorithms=10,              # More algorithms
    auto_tune_hyperparams=True,     # Enable tuning
    preprocessing_strategy="aggressive"  # Thorough preprocessing
)
```

## Integration with Existing Workflows

### 1. Batch Processing

```python
import glob
from pathlib import Path

def batch_autonomous_detection(data_directory, output_directory):
    """Process multiple files with autonomous detection."""
    
    data_files = glob.glob(f"{data_directory}/*.csv")
    results_summary = {}
    
    for file_path in data_files:
        file_name = Path(file_path).stem
        print(f"Processing {file_name}...")
        
        try:
            results = asyncio.run(service.detect_autonomous(file_path, config))
            
            # Extract key metrics
            auto_results = results["autonomous_detection_results"]
            results_summary[file_name] = {
                'success': auto_results.get('success', False),
                'best_algorithm': auto_results.get('best_algorithm'),
                'total_anomalies': auto_results.get('best_result', {}).get('summary', {}).get('total_anomalies', 0),
                'processing_time': auto_results.get('processing_time', 0)
            }
            
            # Save individual results
            output_file = f"{output_directory}/{file_name}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            results_summary[file_name] = {'success': False, 'error': str(e)}
    
    return results_summary
```

### 2. Continuous Monitoring

```python
import time
from datetime import datetime

def continuous_monitoring(data_source, check_interval=3600):
    """Continuously monitor data source for anomalies."""
    
    while True:
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] Running anomaly detection...")
        
        try:
            results = asyncio.run(service.detect_autonomous(data_source, config))
            
            auto_results = results["autonomous_detection_results"]
            if auto_results.get('success'):
                best_result = auto_results.get('best_result', {})
                anomaly_count = best_result.get('summary', {}).get('total_anomalies', 0)
                
                if anomaly_count > 0:
                    print(f"⚠️  {anomaly_count} anomalies detected!")
                    # Implement alerting logic here
                else:
                    print("✅ No anomalies detected")
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
        
        time.sleep(check_interval)
```

This implementation guide provides practical examples for using all the enhanced autonomous features across different interfaces and use cases. The examples demonstrate both basic usage and advanced configurations for production deployments.