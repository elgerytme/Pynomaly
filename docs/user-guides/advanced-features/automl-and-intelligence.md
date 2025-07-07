# AutoML and Intelligent Features Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔶 [Advanced Features](README.md) > 🧠 AutoML

---


## Overview

Pynomaly's AutoML capabilities enable automatic algorithm selection, hyperparameter optimization, and intelligent anomaly detection configuration. This guide covers autonomous detection, dataset profiling, intelligent threshold calculation, and advanced optimization techniques.

## 🤖 Autonomous Anomaly Detection

### Quick Start

```python
from pynomaly import autonomous_detect

# Fully automatic anomaly detection
result = autonomous_detect(
    data_path="data.csv",
    output_path="results.json"
)

print(f"Detected {result.anomalies_found} anomalies")
print(f"Best algorithm: {result.best_algorithm}")
print(f"Confidence: {result.confidence:.2%}")
```

### Advanced Configuration

```python
from pynomaly import AutonomousDetector

detector = AutonomousDetector(
    # Algorithm selection strategy
    algorithm_selection="comprehensive",  # fast, balanced, comprehensive
    
    # Evaluation strategy
    evaluation_method="cross_validation",
    cv_folds=5,
    
    # Optimization settings
    optimization_budget=100,  # Number of trials
    optimization_timeout=3600,  # 1 hour timeout
    
    # Resource constraints
    max_memory_gb=8,
    use_gpu=True,
    
    # Output preferences
    explain_results=True,
    generate_visualizations=True
)

# Profile dataset and recommend approach
profile = detector.profile_dataset("data.csv")
print(f"Dataset characteristics: {profile.summary}")
print(f"Recommended algorithms: {profile.recommendations}")

# Run autonomous detection
result = detector.detect("data.csv")
```

## 📊 Dataset Profiling

### Automatic Data Analysis

```python
from pynomaly.application.services import DatasetProfiler

profiler = DatasetProfiler()
profile = profiler.analyze("dataset.csv")

# Dataset characteristics
print(f"Shape: {profile.shape}")
print(f"Data types: {profile.dtypes}")
print(f"Missing values: {profile.missing_percent:.1%}")
print(f"Numeric features: {len(profile.numeric_features)}")
print(f"Categorical features: {len(profile.categorical_features)}")

# Statistical summary
print(f"Skewness: {profile.skewness}")
print(f"Correlation strength: {profile.correlation_strength}")
print(f"Outlier percentage: {profile.outlier_percentage:.1%}")

# Recommendations
print(f"Suggested contamination rate: {profile.suggested_contamination}")
print(f"Recommended algorithms: {profile.algorithm_recommendations}")
print(f"Preprocessing needed: {profile.preprocessing_recommendations}")
```

### Advanced Profiling Features

```python
# Deep statistical analysis
deep_profile = profiler.deep_analyze(
    data="dataset.csv",
    include_distributions=True,
    include_correlations=True,
    include_outlier_analysis=True,
    include_time_series_analysis=True  # if temporal data
)

# Data quality assessment
quality = deep_profile.quality_assessment
print(f"Data quality score: {quality.overall_score:.2f}")
print(f"Completeness: {quality.completeness:.2%}")
print(f"Consistency: {quality.consistency:.2%}")
print(f"Accuracy: {quality.accuracy:.2%}")

# Feature importance estimation
feature_importance = deep_profile.feature_analysis
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.3f}")
```

## 🎯 Intelligent Algorithm Selection

### Automatic Selection

```python
from pynomaly.application.services import AlgorithmSelector

selector = AlgorithmSelector()

# Automatic selection based on data characteristics
recommendations = selector.recommend_algorithms(
    dataset_profile=profile,
    performance_priority="balanced",  # speed, accuracy, balanced
    resource_constraints={
        "max_training_time": 300,  # 5 minutes
        "max_memory_mb": 4096,     # 4GB
        "require_interpretability": False
    }
)

print("Algorithm recommendations:")
for algo in recommendations:
    print(f"- {algo.name}: {algo.score:.3f} (reason: {algo.reason})")
```

### Custom Selection Criteria

```python
# Define custom selection criteria
custom_selector = AlgorithmSelector(
    criteria={
        "accuracy_weight": 0.4,
        "speed_weight": 0.3,
        "interpretability_weight": 0.2,
        "scalability_weight": 0.1
    },
    constraints={
        "max_complexity": "medium",
        "required_features": ["feature_importance", "probabilistic_scores"],
        "excluded_algorithms": ["neural_networks"]  # if interpretability needed
    }
)

recommendations = custom_selector.recommend(dataset_profile=profile)
```

## ⚙️ Hyperparameter Optimization

### Optuna-Based Optimization

```python
from pynomaly.application.services import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(
    optimization_framework="optuna",
    n_trials=100,
    timeout=3600,  # 1 hour
    direction="maximize",  # maximize F1 score
    metric="f1_score"
)

# Optimize single algorithm
best_params = optimizer.optimize_algorithm(
    algorithm="IsolationForest",
    dataset="train.csv",
    validation_dataset="val.csv",
    search_space={
        "n_estimators": {"type": "int", "low": 50, "high": 300},
        "contamination": {"type": "float", "low": 0.01, "high": 0.3},
        "max_samples": {"type": "categorical", "choices": ["auto", 0.5, 0.8, 1.0]}
    }
)

print(f"Best parameters: {best_params}")
print(f"Best score: {optimizer.best_score:.4f}")
```

### Multi-Algorithm Optimization

```python
# Optimize multiple algorithms simultaneously
multi_optimizer = HyperparameterOptimizer(
    multi_algorithm=True,
    algorithms=["IsolationForest", "LOF", "AutoEncoder"],
    ensemble_strategy="voting"  # voting, stacking, blending
)

best_ensemble = multi_optimizer.optimize_ensemble(
    dataset="train.csv",
    validation_dataset="val.csv",
    trials_per_algorithm=50
)

print(f"Best ensemble composition: {best_ensemble.algorithms}")
print(f"Best ensemble weights: {best_ensemble.weights}")
print(f"Ensemble score: {best_ensemble.score:.4f}")
```

## 🧮 Intelligent Threshold Calculation

### Automatic Threshold Selection

```python
from pynomaly.domain.services import ThresholdCalculator

calculator = ThresholdCalculator()

# Calculate optimal threshold using multiple methods
threshold_analysis = calculator.analyze_thresholds(
    anomaly_scores=scores,
    methods=["percentile", "statistical", "elbow", "roc_optimal"],
    true_labels=y_true  # if available for validation
)

print(f"Recommended threshold: {threshold_analysis.best_threshold}")
print(f"Method used: {threshold_analysis.best_method}")
print(f"Expected precision: {threshold_analysis.expected_precision:.3f}")
print(f"Expected recall: {threshold_analysis.expected_recall:.3f}")
```

### Adaptive Thresholding

```python
# Adaptive threshold for streaming data
adaptive_calculator = ThresholdCalculator(
    mode="adaptive",
    window_size=1000,
    adaptation_rate=0.1,
    stability_threshold=0.05
)

# Update threshold as new data arrives
for batch in data_stream:
    scores = detector.predict(batch)
    threshold = adaptive_calculator.update(scores)
    anomalies = scores > threshold
    
    # Handle concept drift
    if adaptive_calculator.drift_detected:
        print("Concept drift detected - retraining recommended")
```

## 📈 Performance Evaluation and Monitoring

### Comprehensive Evaluation

```python
from pynomaly.application.services import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate model performance
evaluation = evaluator.evaluate_comprehensive(
    detector=detector,
    test_data="test.csv",
    true_labels="labels.csv",  # if available
    metrics=["precision", "recall", "f1", "auc_roc", "auc_pr"]
)

print(f"Performance metrics:")
for metric, value in evaluation.metrics.items():
    print(f"  {metric}: {value:.4f}")

# Statistical significance testing
significance = evaluator.statistical_significance_test(
    detector_a=detector_1,
    detector_b=detector_2,
    test_data="test.csv",
    test_type="mcnemar"
)

print(f"Statistical significance: p-value = {significance.p_value:.4f}")
```

### Automated Model Monitoring

```python
from pynomaly.application.services import ModelMonitor

monitor = ModelMonitor(
    performance_threshold=0.8,
    drift_detection_method="ks_test",
    monitoring_window=1000,
    alert_mechanisms=["email", "webhook"]
)

# Monitor model in production
monitor.start_monitoring(
    detector=detector,
    data_stream=production_stream,
    alert_config={
        "email": "admin@company.com",
        "webhook_url": "https://alerts.company.com/webhook"
    }
)

# Check monitoring status
status = monitor.get_status()
print(f"Model health: {status.health_score:.2f}")
print(f"Performance trend: {status.performance_trend}")
print(f"Drift detected: {status.drift_detected}")
```

## 🔧 Custom AutoML Pipelines

### Building Custom Pipelines

```python
from pynomaly.application.services import AutoMLPipeline

# Create custom AutoML pipeline
pipeline = AutoMLPipeline(
    stages=[
        "data_validation",
        "preprocessing",
        "feature_engineering",
        "algorithm_selection",
        "hyperparameter_optimization",
        "ensemble_construction",
        "model_validation",
        "explanation_generation"
    ]
)

# Configure each stage
pipeline.configure_stage("preprocessing", {
    "handle_missing": "auto",
    "scale_features": "auto",
    "encode_categoricals": "auto"
})

pipeline.configure_stage("algorithm_selection", {
    "strategy": "comprehensive",
    "max_algorithms": 5,
    "time_budget": 1800  # 30 minutes
})

pipeline.configure_stage("ensemble_construction", {
    "method": "stacking",
    "meta_learner": "logistic_regression",
    "cross_validation": 5
})

# Run the complete pipeline
result = pipeline.run("dataset.csv")
print(f"Final model accuracy: {result.accuracy:.4f}")
print(f"Pipeline execution time: {result.execution_time:.1f} seconds")
```

### Pipeline Templates

```python
# Quick templates for common scenarios
from pynomaly.application.services import PipelineTemplates

# Fraud detection template
fraud_pipeline = PipelineTemplates.fraud_detection(
    data_path="transactions.csv",
    target_precision=0.95,
    max_training_time=3600
)

# IoT sensor monitoring template
iot_pipeline = PipelineTemplates.iot_monitoring(
    data_path="sensor_data.csv",
    time_column="timestamp",
    sensor_columns=["temp", "pressure", "vibration"],
    real_time_mode=True
)

# Manufacturing quality control template
quality_pipeline = PipelineTemplates.quality_control(
    data_path="production_data.csv",
    quality_metrics=["dimension_1", "dimension_2", "surface_finish"],
    inspection_rate=0.1  # 10% inspection rate
)
```

## 🎛️ Configuration and Customization

### Global AutoML Settings

```python
from pynomaly import configure_automl

# Set global AutoML preferences
configure_automl({
    "default_optimization_budget": 200,
    "default_timeout": 7200,  # 2 hours
    "default_cv_folds": 5,
    "enable_gpu": True,
    "memory_limit_gb": 16,
    "cache_optimizations": True,
    "parallel_trials": 4,
    "log_level": "INFO"
})
```

### Custom Optimization Objectives

```python
from pynomaly.application.services import CustomObjective

# Define custom optimization objective
class BusinessImpactObjective(CustomObjective):
    def __init__(self, cost_matrix):
        self.cost_matrix = cost_matrix
    
    def evaluate(self, y_true, y_pred):
        # Calculate business impact score
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        cost = (
            fp * self.cost_matrix["false_positive"] +
            fn * self.cost_matrix["false_negative"]
        )
        
        return -cost  # Minimize cost (maximize negative cost)

# Use custom objective
business_objective = BusinessImpactObjective({
    "false_positive": 100,   # $100 cost per false positive
    "false_negative": 1000   # $1000 cost per false negative
})

optimizer = HyperparameterOptimizer(
    objective=business_objective,
    direction="maximize"
)
```

## 📊 Visualization and Interpretation

### AutoML Results Visualization

```python
from pynomaly.application.services import AutoMLVisualizer

visualizer = AutoMLVisualizer()

# Create comprehensive AutoML report
report = visualizer.create_automl_report(
    optimization_history=optimizer.trials,
    best_model=best_model,
    dataset_profile=profile,
    output_format="html"
)

# Generate specific visualizations
visualizer.plot_optimization_history()
visualizer.plot_algorithm_comparison()
visualizer.plot_hyperparameter_importance()
visualizer.plot_performance_tradeoffs()

# Export interactive dashboard
dashboard = visualizer.create_interactive_dashboard(
    results=automl_results,
    include_explanation=True,
    include_recommendations=True
)
dashboard.save("automl_dashboard.html")
```

## 🚀 Best Practices

### Data Preparation for AutoML

```python
# Ensure data quality before AutoML
from pynomaly.infrastructure.preprocessing import DataQualityChecker

quality_checker = DataQualityChecker()
quality_report = quality_checker.check("dataset.csv")

if quality_report.critical_issues:
    print("Critical data quality issues found:")
    for issue in quality_report.critical_issues:
        print(f"- {issue}")
    
    # Apply automatic fixes
    cleaned_data = quality_checker.auto_fix("dataset.csv")
    print("Data automatically cleaned")
```

### Resource Management

```python
# Monitor resource usage during AutoML
from pynomaly.infrastructure.monitoring import ResourceMonitor

with ResourceMonitor() as monitor:
    result = autonomous_detect("large_dataset.csv")
    
print(f"Peak memory usage: {monitor.peak_memory_mb} MB")
print(f"Total CPU time: {monitor.cpu_time_seconds} seconds")
print(f"GPU utilization: {monitor.gpu_utilization:.1%}")
```

### Reproducibility

```python
# Ensure reproducible AutoML results
from pynomaly import set_random_seed

set_random_seed(42)  # Set global random seed

automl_config = {
    "random_state": 42,
    "deterministic_mode": True,
    "save_intermediate_results": True,
    "checkpoint_frequency": 10  # Save every 10 trials
}

result = autonomous_detect(
    "dataset.csv",
    config=automl_config,
    save_path="automl_checkpoint.pkl"
)

# Resume from checkpoint if needed
resumed_result = autonomous_detect(
    "dataset.csv",
    resume_from="automl_checkpoint.pkl"
)
```

## 🔗 Integration with Other Systems

### MLOps Integration

```python
# MLflow integration for experiment tracking
from pynomaly.infrastructure.mlops import MLflowIntegration

mlflow_integration = MLflowIntegration(
    tracking_uri="http://mlflow-server:5000",
    experiment_name="anomaly_detection_automl"
)

with mlflow_integration.run():
    result = autonomous_detect("dataset.csv")
    
    # Automatically log metrics, parameters, and artifacts
    mlflow_integration.log_automl_results(result)
```

### Continuous Learning

```python
# Set up continuous learning pipeline
from pynomaly.application.services import ContinuousLearner

learner = ContinuousLearner(
    model=best_model,
    retraining_schedule="weekly",
    performance_threshold=0.85,
    data_drift_threshold=0.1
)

# Monitor and retrain automatically
learner.start_monitoring(
    data_stream=production_stream,
    feedback_stream=label_stream
)
```

## 📚 Advanced Topics

- [Custom Algorithm Development](../development/custom-algorithms.md)
- [Ensemble Methods](../guides/ensemble-methods.md)
- [Streaming AutoML](../guides/streaming-automl.md)
- [Distributed AutoML](../guides/distributed-automl.md)
- [AutoML API Reference](../api/automl-reference.md)

## 🤝 Contributing

See our [Contributing Guide](../development/contributing.md) for information on extending AutoML capabilities.
