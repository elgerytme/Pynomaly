# Data Science Package User Guide

## Overview

The Data Science package provides comprehensive statistical analysis, machine learning pipeline framework, and advanced analytics capabilities. It serves as the analytical engine for data science workflows and anomaly detection.

## Quick Start

### Installation

```bash
pip install pynomaly[ml-all]
```

### Basic Usage

```python
from pynomaly.packages.data_science import DataScienceEngine
import pandas as pd

# Load your dataset
data = pd.read_csv('your_data.csv')

# Create data science engine
ds_engine = DataScienceEngine()

# Perform statistical analysis
stats_result = ds_engine.statistical_analysis(data)

# View results
print(f"Dataset summary: {stats_result.summary}")
print(f"Correlation matrix: {stats_result.correlation_matrix}")
```

## Core Features

### 1. Statistical Analysis Engine

#### Descriptive Statistics

```python
from pynomaly.packages.data_science import StatisticalAnalysisService

# Create statistical analyzer
stats_analyzer = StatisticalAnalysisService()

# Generate comprehensive statistics
stats = stats_analyzer.analyze(data)

# Access key statistics
print(f"Mean values: {stats.means}")
print(f"Standard deviations: {stats.standard_deviations}")
print(f"Skewness: {stats.skewness}")
print(f"Kurtosis: {stats.kurtosis}")
```

#### Advanced Statistical Tests

```python
# Normality tests
normality_results = stats_analyzer.test_normality(data)
print(f"Shapiro-Wilk test: {normality_results.shapiro_wilk}")
print(f"Kolmogorov-Smirnov test: {normality_results.ks_test}")

# Correlation analysis
correlation_analysis = stats_analyzer.correlation_analysis(data)
print(f"Pearson correlations: {correlation_analysis.pearson}")
print(f"Spearman correlations: {correlation_analysis.spearman}")

# Hypothesis testing
hypothesis_results = stats_analyzer.hypothesis_testing(
    data['group_a'], 
    data['group_b'],
    test_type='t_test'
)
```

#### Time Series Analysis

```python
from pynomaly.packages.data_science import TimeSeriesAnalyzer

# Create time series analyzer
ts_analyzer = TimeSeriesAnalyzer()

# Decompose time series
decomposition = ts_analyzer.decompose(time_series_data)
print(f"Trend: {decomposition.trend}")
print(f"Seasonality: {decomposition.seasonal}")
print(f"Residuals: {decomposition.residuals}")

# Detect seasonality
seasonality = ts_analyzer.detect_seasonality(time_series_data)
print(f"Seasonal periods: {seasonality.periods}")
print(f"Seasonal strength: {seasonality.strength}")
```

### 2. Machine Learning Pipeline Framework

#### Pipeline Creation

```python
from pynomaly.packages.data_science import MLPipelineFramework

# Create ML pipeline
pipeline_framework = MLPipelineFramework()

# Define pipeline stages
pipeline = pipeline_framework.create_pipeline([
    'data_preprocessing',
    'feature_engineering',
    'model_training',
    'model_evaluation'
])

# Configure pipeline
pipeline.configure(
    preprocessing_config={
        'handle_missing': 'impute',
        'normalize': True,
        'encode_categorical': True
    },
    feature_config={
        'feature_selection': 'auto',
        'dimensionality_reduction': 'pca',
        'n_components': 0.95
    },
    model_config={
        'algorithm': 'random_forest',
        'hyperparameter_tuning': True,
        'cross_validation': 5
    }
)

# Execute pipeline
pipeline_result = pipeline.execute(data, target_column='target')
```

#### Advanced Pipeline Features

```python
# Custom pipeline stages
from pynomaly.packages.data_science import CustomPipelineStage

class DataAugmentationStage(CustomPipelineStage):
    def execute(self, data, config):
        # Custom data augmentation logic
        augmented_data = self.augment_data(data)
        return augmented_data

# Add custom stage
pipeline.add_stage('data_augmentation', DataAugmentationStage())

# Parallel pipeline execution
pipeline.enable_parallel_execution(n_jobs=4)

# Pipeline caching for reproducibility
pipeline.enable_caching(cache_dir='./pipeline_cache')
```

### 3. Feature Engineering Framework

#### Automated Feature Engineering

```python
from pynomaly.packages.data_science import FeatureEngineeringService

# Create feature engineering service
feature_engineer = FeatureEngineeringService()

# Automated feature generation
enhanced_features = feature_engineer.auto_feature_engineering(
    data,
    target='target_column',
    max_features=50,
    feature_types=['polynomial', 'interaction', 'aggregate']
)

# Feature importance analysis
feature_importance = feature_engineer.analyze_feature_importance(
    enhanced_features, 
    target
)
print(f"Top features: {feature_importance.top_features}")
```

#### Custom Feature Engineering

```python
# Define custom feature transformations
feature_config = {
    'numerical_features': {
        'transformations': ['log', 'sqrt', 'polynomial'],
        'polynomial_degree': 2,
        'interaction_terms': True
    },
    'categorical_features': {
        'encoding': 'target_encoding',
        'handle_rare_categories': True,
        'min_frequency': 0.01
    },
    'temporal_features': {
        'extract_components': True,
        'rolling_statistics': [7, 30, 90],
        'lag_features': [1, 7, 30]
    }
}

# Apply feature engineering
engineered_data = feature_engineer.transform_features(data, feature_config)
```

### 4. Data Visualization and Reporting

#### Statistical Visualizations

```python
from pynomaly.packages.data_science import DataVisualizationService

# Create visualization service
viz_service = DataVisualizationService()

# Generate statistical plots
viz_service.create_distribution_plots(data)
viz_service.create_correlation_heatmap(data)
viz_service.create_box_plots(data)

# Advanced visualizations
viz_service.create_pca_visualization(data)
viz_service.create_feature_importance_plot(feature_importance)
viz_service.create_model_performance_dashboard(model_results)
```

#### Interactive Dashboards

```python
# Create interactive dashboard
dashboard = viz_service.create_interactive_dashboard(
    data=data,
    include_widgets=['filter', 'selector', 'slider'],
    auto_refresh=True
)

# Export dashboard
dashboard.export('data_science_dashboard.html')
```

## Advanced Features

### 1. AutoML Integration

```python
from pynomaly.packages.data_science import AutoMLService

# Create AutoML service
automl = AutoMLService()

# Automated model selection and tuning
automl_result = automl.auto_model_selection(
    data=data,
    target='target_column',
    problem_type='classification',  # or 'regression'
    time_budget=3600,  # 1 hour
    ensemble_size=5
)

# Get best model
best_model = automl_result.best_model
print(f"Best algorithm: {best_model.algorithm}")
print(f"Best score: {best_model.score}")
print(f"Best parameters: {best_model.parameters}")
```

### 2. Model Validation and Testing

```python
from pynomaly.packages.data_science import ModelValidationService

# Create validation service
validator = ModelValidationService()

# Comprehensive model validation
validation_results = validator.comprehensive_validation(
    model=trained_model,
    data=test_data,
    target=target,
    validation_types=['cross_validation', 'bootstrap', 'holdout']
)

# Model fairness assessment
fairness_results = validator.assess_model_fairness(
    model=trained_model,
    data=test_data,
    protected_attributes=['gender', 'race']
)

# Model stability testing
stability_results = validator.test_model_stability(
    model=trained_model,
    data=test_data,
    perturbation_types=['noise', 'missing_values', 'outliers']
)
```

### 3. Experiment Tracking

```python
from pynomaly.packages.data_science import ExperimentTracker

# Initialize experiment tracking
tracker = ExperimentTracker(
    experiment_name='anomaly_detection_optimization',
    tracking_backend='mlflow'  # or 'wandb', 'tensorboard'
)

# Track experiment
with tracker.start_run():
    # Log parameters
    tracker.log_params({
        'algorithm': 'isolation_forest',
        'contamination': 0.1,
        'n_estimators': 100
    })
    
    # Train model
    model = train_model(data)
    
    # Log metrics
    tracker.log_metrics({
        'precision': 0.85,
        'recall': 0.82,
        'f1_score': 0.83
    })
    
    # Log artifacts
    tracker.log_artifact(model, 'trained_model.pkl')
```

## Integration Patterns

### Integration with Data Quality

```python
from pynomaly.packages.data_quality import DataQualityEngine

# Ensure data quality before analysis
quality_engine = DataQualityEngine()
quality_report = quality_engine.assess_quality(data)

if quality_report.overall_score >= 0.8:
    # Proceed with data science analysis
    ds_engine = DataScienceEngine()
    analysis_result = ds_engine.comprehensive_analysis(data)
else:
    print("Data quality insufficient for reliable analysis")
```

### Integration with Anomaly Detection

```python
from pynomaly import AnomalyDetector

# Use data science insights for anomaly detection
feature_importance = ds_engine.analyze_feature_importance(data)
recommended_features = feature_importance.top_features[:10]

# Configure detector with insights
detector = AnomalyDetector(
    algorithm='IsolationForest',
    features=recommended_features,
    contamination=ds_engine.estimate_contamination(data)
)

# Enhanced detection with statistical insights
results = detector.detect(data)
```

## Performance Optimization

### Distributed Computing

```python
# Enable distributed processing
ds_engine.configure_distributed_processing(
    framework='dask',  # or 'spark', 'ray'
    cluster_config={
        'n_workers': 4,
        'memory_per_worker': '4GB'
    }
)

# Distributed statistical analysis
distributed_stats = ds_engine.distributed_analysis(large_dataset)
```

### Memory Optimization

```python
# Configure for large datasets
ds_engine.configure_memory_optimization(
    chunk_size=10000,
    use_sparse_matrices=True,
    memory_mapping=True
)

# Incremental analysis for streaming data
incremental_stats = ds_engine.incremental_analysis(
    data_stream,
    update_frequency='1min'
)
```

## Advanced Analytics

### Anomaly Detection in Statistical Context

```python
# Statistical anomaly detection
statistical_anomalies = ds_engine.detect_statistical_anomalies(
    data,
    methods=['z_score', 'iqr', 'isolation_forest'],
    confidence_level=0.95
)

# Multivariate anomaly detection
multivariate_anomalies = ds_engine.detect_multivariate_anomalies(
    data,
    method='mahalanobis_distance'
)
```

### Causal Analysis

```python
from pynomaly.packages.data_science import CausalAnalysisService

# Causal analysis
causal_analyzer = CausalAnalysisService()

# Discover causal relationships
causal_graph = causal_analyzer.discover_causal_structure(data)

# Estimate causal effects
causal_effects = causal_analyzer.estimate_causal_effects(
    data,
    treatment='treatment_variable',
    outcome='outcome_variable',
    confounders=['confounder1', 'confounder2']
)
```

## Configuration and Customization

### Engine Configuration

```python
# Configure data science engine
config = {
    'statistical_analysis': {
        'confidence_level': 0.95,
        'robust_statistics': True,
        'parallel_processing': True
    },
    'machine_learning': {
        'auto_feature_selection': True,
        'hyperparameter_tuning': True,
        'cross_validation_folds': 5
    },
    'visualization': {
        'interactive_plots': True,
        'high_resolution': True,
        'theme': 'professional'
    }
}

ds_engine = DataScienceEngine(config=config)
```

### Custom Analytics

```python
from pynomaly.packages.data_science import CustomAnalyzer

class BusinessMetricsAnalyzer(CustomAnalyzer):
    def analyze(self, data):
        # Custom business analytics
        return self.calculate_business_metrics(data)

# Register custom analyzer
ds_engine.register_analyzer('business_metrics', BusinessMetricsAnalyzer())
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Datasets**
   ```python
   # Use chunked processing
   ds_engine.enable_chunked_processing(chunk_size=5000)
   ```

2. **Convergence Issues in ML Algorithms**
   ```python
   # Adjust convergence parameters
   ml_config = {
       'max_iterations': 1000,
       'tolerance': 1e-6,
       'early_stopping': True
   }
   ```

3. **Feature Engineering Performance**
   ```python
   # Optimize feature engineering
   feature_engineer.configure_optimization(
       parallel_processing=True,
       feature_caching=True,
       batch_processing=True
   )
   ```

## API Reference

### Core Classes

- `DataScienceEngine`: Main analytics engine
- `StatisticalAnalysisService`: Statistical analysis
- `MLPipelineFramework`: ML pipeline management
- `FeatureEngineeringService`: Feature engineering
- `ModelValidationService`: Model validation

### Key Methods

- `statistical_analysis(data)`: Comprehensive statistical analysis
- `create_ml_pipeline()`: Create ML pipeline
- `auto_feature_engineering()`: Automated feature engineering
- `model_validation()`: Model validation and testing

## Complete Example

```python
import pandas as pd
from pynomaly.packages.data_science import (
    DataScienceEngine,
    StatisticalAnalysisService,
    FeatureEngineeringService,
    MLPipelineFramework
)

# Load data
data = pd.read_csv('sensor_data.csv')

# Create data science engine
ds_engine = DataScienceEngine()

# Statistical analysis
stats_service = StatisticalAnalysisService()
stats_result = stats_service.comprehensive_analysis(data)

print("=== Statistical Analysis ===")
print(f"Dataset shape: {stats_result.shape}")
print(f"Missing values: {stats_result.missing_values}")
print(f"Data distribution: {stats_result.distribution_summary}")

# Feature engineering
feature_engineer = FeatureEngineeringService()
engineered_data = feature_engineer.auto_feature_engineering(
    data,
    target='anomaly_label',
    max_features=30
)

# Create ML pipeline
pipeline_framework = MLPipelineFramework()
pipeline = pipeline_framework.create_anomaly_detection_pipeline()

# Configure pipeline
pipeline.configure({
    'preprocessing': {
        'normalize': True,
        'handle_missing': 'impute'
    },
    'feature_selection': {
        'method': 'mutual_info',
        'k_features': 20
    },
    'model': {
        'algorithm': 'isolation_forest',
        'contamination': 'auto'
    }
})

# Execute pipeline
pipeline_result = pipeline.execute(engineered_data)

print("=== Pipeline Results ===")
print(f"Model performance: {pipeline_result.performance_metrics}")
print(f"Feature importance: {pipeline_result.feature_importance}")
print(f"Anomalies detected: {pipeline_result.anomaly_count}")

# Generate insights report
insights_report = ds_engine.generate_insights_report(
    data=data,
    pipeline_result=pipeline_result,
    stats_result=stats_result
)

print("=== Key Insights ===")
print(insights_report.summary)
```

For more detailed examples and advanced usage, see the [API Reference documentation](../api-reference/).