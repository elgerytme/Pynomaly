# 🤖 AutoML Service Guide

## Overview

The Pynomaly AutoML service provides comprehensive automated machine learning capabilities specifically designed for anomaly detection use cases. This guide covers the complete AutoML pipeline, from data profiling to model deployment.

## Table of Contents

1. [AutoML Components](#automl-components)
2. [Pipeline Orchestration](#pipeline-orchestration)
3. [Feature Engineering](#feature-engineering)
4. [Hyperparameter Optimization](#hyperparameter-optimization)
5. [Model Evaluation](#model-evaluation)
6. [Result Tracking](#result-tracking)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)

## AutoML Components

### Core Services

The AutoML system consists of several interconnected services:

```
AutoML Pipeline Orchestrator
├── AutoML Service (Core)
├── Feature Engineering Service
├── Hyperparameter Optimization Service
├── Model Evaluation & Selection Service
└── Result Tracking Service
```

### Service Responsibilities

- **AutoML Service**: Data profiling, algorithm recommendation
- **Pipeline Orchestrator**: Workflow coordination, state management
- **Feature Engineering**: Automated feature generation and selection
- **Hyperparameter Optimization**: Bayesian optimization with Optuna
- **Model Evaluation**: Cross-validation, statistical significance testing
- **Result Tracking**: Experiment management, model comparison

## Pipeline Orchestration

### Pipeline Stages

The AutoML pipeline executes in the following stages:

1. **Data Profiling**: Analyze dataset characteristics
2. **Feature Engineering**: Generate and select features
3. **Algorithm Recommendation**: Select appropriate algorithms
4. **Hyperparameter Optimization**: Optimize model parameters
5. **Model Evaluation**: Evaluate and compare models
6. **Ensemble Creation**: Create ensemble models
7. **Validation**: Validate final model performance
8. **Deployment Preparation**: Prepare artifacts for deployment

### Basic Usage

```python
from pynomaly.application.services import (
    AutoMLPipelineOrchestrator,
    AutoMLPipelineConfiguration,
    AutoMLService,
    HyperparameterOptimizationService,
    ModelEvaluationSelectionService,
    FeatureEngineeringService
)
from pynomaly.domain.entities import Dataset

# Initialize services
automl_service = AutoMLService()
hyperparameter_service = HyperparameterOptimizationService()
evaluation_service = ModelEvaluationSelectionService()
feature_service = FeatureEngineeringService()

# Create orchestrator
orchestrator = AutoMLPipelineOrchestrator(
    automl_service=automl_service,
    hyperparameter_service=hyperparameter_service,
    evaluation_service=evaluation_service,
    feature_service=feature_service
)

# Configure pipeline
config = AutoMLPipelineConfiguration(
    contamination_rate=0.1,
    max_execution_time=3600,  # 1 hour
    optimization_config={
        "n_trials": 100,
        "timeout": 300
    },
    evaluation_config={
        "cv_folds": 5,
        "test_size": 0.2
    }
)

# Create and execute pipeline
pipeline = await orchestrator.create_pipeline(
    name="anomaly_detection_experiment",
    dataset=your_dataset,
    configuration=config
)

result = await orchestrator.execute_pipeline(pipeline, your_dataset)

# Access results
best_model = result.best_model
performance_metrics = result.performance_metrics
recommendations = result.recommended_models
```

### Advanced Configuration

```python
# Advanced pipeline configuration
advanced_config = AutoMLPipelineConfiguration(
    # Data configuration
    contamination_rate=0.05,
    target_column="is_anomaly",
    
    # Pipeline execution
    max_execution_time=7200,  # 2 hours
    enable_parallel_execution=True,
    retry_failed_stages=True,
    max_retries=3,
    
    # Stage-specific configurations
    feature_engineering_config={
        "enable_polynomial_features": True,
        "enable_interaction_features": True,
        "max_feature_combinations": 50,
        "feature_selection_method": "mutual_info"
    },
    
    optimization_config={
        "n_trials": 200,
        "timeout": 600,
        "pruner": "hyperband",
        "sampler": "tpe"
    },
    
    evaluation_config={
        "cv_folds": 10,
        "test_size": 0.3,
        "scoring_metrics": ["f1", "precision", "recall", "roc_auc"],
        "statistical_tests": True
    },
    
    # Output configuration
    save_intermediate_results=True,
    output_directory="./automl_results"
)
```

## Feature Engineering

### Automated Feature Generation

The feature engineering service automatically generates features:

```python
feature_service = FeatureEngineeringService()

# Basic feature engineering
engineered_data = await feature_service.engineer_features(
    X=your_dataframe,
    y=target_series
)

# Advanced feature engineering with configuration
engineered_data = await feature_service.engineer_features(
    X=your_dataframe,
    y=target_series,
    enable_polynomial_features=True,
    polynomial_degree=2,
    enable_interaction_features=True,
    max_interactions=20,
    enable_statistical_features=True,
    enable_temporal_features=True,  # For time series data
    feature_selection_method="mutual_info",
    max_features=100
)
```

### Feature Engineering Capabilities

- **Statistical Features**: Mean, std, skewness, kurtosis
- **Polynomial Features**: Higher-order feature combinations
- **Interaction Features**: Cross-feature interactions
- **Temporal Features**: Time-based features for time series
- **Missing Value Handling**: Intelligent imputation strategies
- **Feature Selection**: Variance, correlation, and information-based selection

## Hyperparameter Optimization

### Bayesian Optimization with Optuna

```python
hyperopt_service = HyperparameterOptimizationService()

# Basic hyperparameter optimization
result = await hyperopt_service.optimize_hyperparameters(
    algorithm_name="IsolationForest",
    data=your_data,
    n_trials=100,
    timeout=300
)

# Advanced optimization with custom search spaces
custom_search_space = {
    "n_estimators": ("int", 50, 500),
    "contamination": ("float", 0.01, 0.3),
    "max_features": ("categorical", [1.0, "sqrt", "log2"]),
    "bootstrap": ("categorical", [True, False])
}

result = await hyperopt_service.optimize_hyperparameters(
    algorithm_name="IsolationForest",
    data=your_data,
    search_space=custom_search_space,
    n_trials=200,
    timeout=600,
    pruner="hyperband",
    sampler="tpe",
    n_jobs=-1
)

# Access optimization results
best_params = result["best_params"]
best_score = result["best_value"]
optimization_history = result["trials"]
```

### Multi-Objective Optimization

```python
# Optimize for multiple objectives
result = await hyperopt_service.multi_objective_optimize(
    algorithm_name="IsolationForest",
    data=your_data,
    objectives=["f1_score", "training_time"],
    directions=["maximize", "minimize"],
    n_trials=150
)

# Get Pareto frontier
pareto_solutions = result["pareto_frontier"]
```

## Model Evaluation

### Comprehensive Evaluation

```python
evaluation_service = ModelEvaluationSelectionService()

# Evaluate a single model
result = await evaluation_service.evaluate_model(
    algorithm_name="IsolationForest",
    hyperparameters={"contamination": 0.1, "n_estimators": 100},
    data=your_data,
    cv_folds=5,
    test_size=0.2,
    scoring_metrics=["f1", "precision", "recall", "roc_auc"]
)

# Access evaluation results
metrics = result["metrics"]
cv_scores = result["cv_scores"]
statistical_significance = result["significance_test"]
```

### Model Comparison

```python
# Compare multiple models
models_to_compare = [
    {"name": "IsolationForest", "params": {"contamination": 0.1}},
    {"name": "LocalOutlierFactor", "params": {"contamination": 0.1}},
    {"name": "OneClassSVM", "params": {"nu": 0.1}}
]

comparison = await evaluation_service.compare_models(
    models=models_to_compare,
    data=your_data,
    cv_folds=5,
    statistical_tests=True
)

# Get ranked models
ranked_models = comparison["ranked_models"]
statistical_tests = comparison["statistical_significance"]
```

## Result Tracking

### Experiment Management

```python
from pynomaly.application.services import AutoMLResultTrackingService

tracking_service = AutoMLResultTrackingService()

# Create experiment
experiment = await tracking_service.create_experiment(
    experiment_name="anomaly_detection_v1",
    dataset_name="sensor_data",
    dataset_id=dataset.id,
    algorithm_name="IsolationForest",
    hyperparameters={"contamination": 0.1, "n_estimators": 100},
    tags=["production", "sensors", "v1"]
)

# Start experiment
await tracking_service.start_experiment(experiment.id)

# Complete experiment with results
metrics = ModelMetrics(
    f1_score=0.85,
    precision=0.87,
    recall=0.83,
    roc_auc=0.91,
    training_time=45.2,
    cv_scores=[0.84, 0.86, 0.85, 0.84, 0.87]
)

await tracking_service.complete_experiment(
    experiment.id,
    metrics=metrics,
    model_path="./models/isolation_forest_v1.pkl",
    feature_importance={"feature_1": 0.45, "feature_2": 0.32, "feature_3": 0.23}
)
```

### Experiment Comparison and Leaderboard

```python
# Compare experiments
comparison = await tracking_service.compare_experiments(
    experiment_ids=[exp1.id, exp2.id, exp3.id],
    primary_metric="f1_score",
    secondary_metrics=["precision", "recall"]
)

# Get leaderboard
leaderboard = await tracking_service.get_leaderboard(
    dataset_name="sensor_data",
    metric="f1_score",
    limit=10
)

# Get insights
insights = await tracking_service.get_experiment_insights(
    dataset_name="sensor_data",
    days=30
)
```

## Usage Examples

### Complete End-to-End Example

```python
import asyncio
import pandas as pd
import numpy as np
from pynomaly.application.services import *
from pynomaly.domain.entities import Dataset

async def run_automl_experiment():
    # 1. Prepare data
    data = pd.read_csv("anomaly_data.csv")
    dataset = Dataset(
        name="production_anomalies",
        data=data,
        target_column="is_anomaly",
        metadata={"source": "production_sensors"}
    )
    
    # 2. Initialize services
    automl_service = AutoMLService()
    hyperparameter_service = HyperparameterOptimizationService()
    evaluation_service = ModelEvaluationSelectionService()
    feature_service = FeatureEngineeringService()
    tracking_service = AutoMLResultTrackingService()
    
    # 3. Create orchestrator
    orchestrator = AutoMLPipelineOrchestrator(
        automl_service=automl_service,
        hyperparameter_service=hyperparameter_service,
        evaluation_service=evaluation_service,
        feature_service=feature_service
    )
    
    # 4. Configure pipeline
    config = AutoMLPipelineConfiguration(
        contamination_rate=0.05,
        max_execution_time=3600,
        optimization_config={"n_trials": 100, "timeout": 300},
        evaluation_config={"cv_folds": 5, "test_size": 0.2}
    )
    
    # 5. Create and execute pipeline
    pipeline = await orchestrator.create_pipeline(
        name="production_anomaly_detection",
        dataset=dataset,
        configuration=config
    )
    
    # 6. Track experiment
    experiment = await tracking_service.create_experiment(
        experiment_name="production_experiment_1",
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        algorithm_name="AutoML",
        pipeline_id=pipeline.id
    )
    
    await tracking_service.start_experiment(experiment.id)
    
    try:
        # 7. Execute pipeline
        result = await orchestrator.execute_pipeline(pipeline, dataset)
        
        # 8. Complete experiment tracking
        await tracking_service.complete_experiment(
            experiment.id,
            metrics=ModelMetrics(
                f1_score=result.performance_metrics.get("best_model_score", 0),
                training_time=result.performance_metrics.get("execution_time", 0)
            ),
            model_path=f"./models/{pipeline.id}_best_model.pkl"
        )
        
        # 9. Access results
        print(f"Best model: {result.best_model}")
        print(f"Performance: {result.performance_metrics}")
        print(f"Recommended models: {len(result.recommended_models)}")
        
        return result
        
    except Exception as e:
        await tracking_service.fail_experiment(experiment.id, str(e))
        raise

# Run the experiment
result = asyncio.run(run_automl_experiment())
```

### Batch Processing Multiple Datasets

```python
async def batch_automl_experiments(datasets):
    orchestrator = AutoMLPipelineOrchestrator(...)
    tracking_service = AutoMLResultTrackingService()
    
    results = []
    
    for dataset in datasets:
        config = AutoMLPipelineConfiguration(
            contamination_rate=dataset.metadata.get("contamination_rate", 0.1),
            max_execution_time=1800  # 30 minutes per dataset
        )
        
        pipeline = await orchestrator.create_pipeline(
            name=f"batch_{dataset.name}",
            dataset=dataset,
            configuration=config
        )
        
        result = await orchestrator.execute_pipeline(pipeline, dataset)
        results.append(result)
    
    # Compare all experiments
    experiment_ids = [r.best_model["experiment_id"] for r in results if r.best_model]
    comparison = await tracking_service.compare_experiments(experiment_ids)
    
    return results, comparison
```

## Best Practices

### Dataset Preparation

1. **Data Quality**: Ensure clean, well-formatted data
2. **Feature Names**: Use descriptive feature names
3. **Missing Values**: Handle missing values appropriately
4. **Data Types**: Ensure correct data types for features
5. **Contamination Rate**: Provide accurate contamination estimates

### Pipeline Configuration

1. **Time Budgets**: Set realistic execution time limits
2. **Resource Allocation**: Configure parallel execution appropriately
3. **Checkpointing**: Enable checkpointing for long-running experiments
4. **Validation Strategy**: Choose appropriate cross-validation strategies
5. **Metric Selection**: Select relevant evaluation metrics

### Hyperparameter Optimization

1. **Search Space**: Define reasonable search spaces
2. **Trial Budget**: Balance trial count with time constraints
3. **Pruning**: Use pruning for efficiency
4. **Multiple Runs**: Run multiple optimization sessions for robustness
5. **Seed Setting**: Set random seeds for reproducibility

### Model Selection

1. **Multiple Metrics**: Evaluate models on multiple metrics
2. **Statistical Testing**: Use statistical significance testing
3. **Cross-Validation**: Use appropriate CV strategies
4. **Ensemble Methods**: Consider ensemble approaches
5. **Production Constraints**: Factor in deployment constraints

### Experiment Tracking

1. **Descriptive Names**: Use clear experiment and model names
2. **Comprehensive Metadata**: Track all relevant metadata
3. **Version Control**: Version datasets and code
4. **Reproducibility**: Ensure experiments are reproducible
5. **Documentation**: Document experiment purposes and findings

### Performance Optimization

1. **Parallel Processing**: Leverage parallel execution
2. **Early Stopping**: Use early stopping for efficiency
3. **Resource Monitoring**: Monitor CPU and memory usage
4. **Caching**: Cache intermediate results when possible
5. **Profiling**: Profile code for bottlenecks

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch sizes or use data streaming
2. **Timeout Errors**: Increase time budgets or reduce search spaces
3. **Convergence Issues**: Adjust optimization parameters
4. **Validation Errors**: Check data quality and preprocessing
5. **Performance Issues**: Profile and optimize bottlenecks

### Debugging

1. **Enable Logging**: Use structured logging for debugging
2. **Intermediate Results**: Save intermediate results for analysis
3. **Error Handling**: Implement comprehensive error handling
4. **Monitoring**: Monitor resource usage and performance
5. **Testing**: Test with smaller datasets first

---

This guide provides comprehensive coverage of the Pynomaly AutoML system. For specific use cases or advanced configurations, refer to the API documentation or contact the development team.