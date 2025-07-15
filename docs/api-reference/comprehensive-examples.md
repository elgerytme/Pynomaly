# Comprehensive API Examples

## Overview

This document provides comprehensive examples demonstrating the full capabilities of Pynomaly's API across all major components and use cases.

## Table of Contents

1. [Basic Anomaly Detection](#basic-anomaly-detection)
2. [Data Science Package Examples](#data-science-package)
3. [Data Quality Package Examples](#data-quality-package)
4. [Data Profiling Package Examples](#data-profiling-package)
5. [Data Transformation Package Examples](#data-transformation-package)
6. [Advanced Integration Examples](#advanced-integration)
7. [Production Deployment Examples](#production-deployment)

## Basic Anomaly Detection

### Simple Detection Workflow

```python
import pandas as pd
import numpy as np
from pynomaly import AnomalyDetector

# Generate sample data
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 5))
anomalous_data = np.random.normal(3, 0.5, (50, 5))
data = np.vstack([normal_data, anomalous_data])

df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])

# Basic detection
detector = AnomalyDetector(
    algorithm='IsolationForest',
    contamination=0.05,
    random_state=42
)

# Fit and detect
detector.fit(df)
results = detector.detect(df)

print(f"Anomalies detected: {results.n_anomalies}")
print(f"Anomaly rate: {results.anomaly_rate:.2%}")
print(f"Detection threshold: {results.threshold:.3f}")
```

### Advanced Ensemble Detection

```python
from pynomaly import AnomalyDetector

# Configure ensemble detector
ensemble_detector = AnomalyDetector(
    algorithm='AutoEnsemble',
    algorithms=[
        'IsolationForest',
        'LocalOutlierFactor',
        'OneClassSVM',
        'EllipticEnvelope'
    ],
    ensemble_method='weighted_voting',
    contamination='auto'
)

# Configure individual algorithms
ensemble_detector.configure_algorithms({
    'IsolationForest': {
        'n_estimators': 200,
        'max_samples': 'auto',
        'max_features': 1.0
    },
    'LocalOutlierFactor': {
        'n_neighbors': 20,
        'metric': 'minkowski',
        'p': 2
    },
    'OneClassSVM': {
        'kernel': 'rbf',
        'gamma': 'scale',
        'nu': 0.05
    },
    'EllipticEnvelope': {
        'support_fraction': None,
        'contamination': 0.05
    }
})

# Fit ensemble
ensemble_detector.fit(df)
ensemble_results = ensemble_detector.detect(df)

# Get detailed ensemble information
ensemble_info = ensemble_detector.get_ensemble_info()
print(f"Algorithm weights: {ensemble_info.weights}")
print(f"Individual algorithm scores: {ensemble_info.algorithm_scores}")
```

### Custom Algorithm Implementation

```python
from pynomaly.algorithms import BaseAnomalyAlgorithm
from sklearn.cluster import DBSCAN
import numpy as np

class DBSCANAnomalyDetector(BaseAnomalyAlgorithm):
    """Custom DBSCAN-based anomaly detector."""
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.clusterer = None
        
    def fit(self, X, y=None):
        """Fit the DBSCAN model."""
        self.clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.clusterer.fit(X)
        return self
    
    def predict(self, X):
        """Predict anomalies (noise points in DBSCAN)."""
        if self.clusterer is None:
            raise ValueError("Model must be fitted before prediction")
        
        labels = self.clusterer.fit_predict(X)
        # In DBSCAN, -1 indicates noise (anomalies)
        anomaly_labels = (labels == -1).astype(int)
        # Convert to standard format: 1 = normal, -1 = anomaly
        return np.where(anomaly_labels == 1, -1, 1)
    
    def decision_function(self, X):
        """Calculate anomaly scores."""
        labels = self.clusterer.fit_predict(X)
        # Simple score: distance to nearest cluster center
        scores = np.zeros(len(X))
        
        for cluster_id in set(labels):
            if cluster_id != -1:  # Not noise
                cluster_points = X[labels == cluster_id]
                cluster_center = cluster_points.mean(axis=0)
                
                # Calculate distances
                for i, point in enumerate(X):
                    if labels[i] == cluster_id:
                        scores[i] = np.linalg.norm(point - cluster_center)
                    elif labels[i] == -1:  # Noise point
                        scores[i] = np.inf
        
        return scores

# Register and use custom algorithm
detector = AnomalyDetector(algorithm=DBSCANAnomalyDetector(eps=0.3, min_samples=10))
detector.fit(df)
custom_results = detector.detect(df)
```

## Data Science Package

### Statistical Analysis Engine

```python
from pynomaly.packages.data_science import StatisticalAnalysisService
import pandas as pd
import numpy as np

# Create sample time series data
dates = pd.date_range('2023-01-01', periods=1000, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.cumsum(np.random.normal(0, 1, 1000)) + np.sin(np.arange(1000) * 2 * np.pi / 365),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'feature1': np.random.normal(10, 2, 1000),
    'feature2': np.random.exponential(2, 1000)
})

# Initialize statistical analysis service
stats_service = StatisticalAnalysisService()

# Comprehensive statistical analysis
analysis_result = stats_service.comprehensive_analysis(ts_data.select_dtypes(include=[np.number]))

print("=== Descriptive Statistics ===")
print(f"Mean values:\n{analysis_result.descriptive_stats.means}")
print(f"Standard deviations:\n{analysis_result.descriptive_stats.std_devs}")
print(f"Skewness:\n{analysis_result.descriptive_stats.skewness}")
print(f"Kurtosis:\n{analysis_result.descriptive_stats.kurtosis}")

# Correlation analysis
correlation_result = stats_service.correlation_analysis(ts_data.select_dtypes(include=[np.number]))
print(f"\n=== Correlation Analysis ===")
print(f"Pearson correlations:\n{correlation_result.pearson}")
print(f"Spearman correlations:\n{correlation_result.spearman}")

# Distribution analysis
distribution_result = stats_service.analyze_distributions(ts_data.select_dtypes(include=[np.number]))
print(f"\n=== Distribution Analysis ===")
for column, dist_info in distribution_result.items():
    print(f"{column}: Best fit = {dist_info.best_distribution}, p-value = {dist_info.p_value:.4f}")

# Time series specific analysis
ts_analysis = stats_service.time_series_analysis(ts_data['value'])
print(f"\n=== Time Series Analysis ===")
print(f"Trend: {ts_analysis.trend}")
print(f"Seasonality detected: {ts_analysis.has_seasonality}")
print(f"Seasonal period: {ts_analysis.seasonal_period}")
print(f"Stationarity test p-value: {ts_analysis.stationarity_test.p_value:.4f}")
```

### Machine Learning Pipeline Framework

```python
from pynomaly.packages.data_science import MLPipelineFramework
from sklearn.datasets import make_classification

# Generate classification dataset
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
data['target'] = y

# Create ML pipeline framework
pipeline_framework = MLPipelineFramework()

# Create anomaly detection pipeline
anomaly_pipeline = pipeline_framework.create_anomaly_detection_pipeline()

# Configure pipeline stages
pipeline_config = {
    'data_validation': {
        'check_missing_values': True,
        'check_data_types': True,
        'check_outliers': True
    },
    'preprocessing': {
        'handle_missing_values': {
            'strategy': 'knn_imputation',
            'n_neighbors': 5
        },
        'feature_scaling': {
            'method': 'robust_scaler',
            'quantile_range': (25.0, 75.0)
        },
        'outlier_treatment': {
            'method': 'isolation_forest',
            'contamination': 0.1,
            'action': 'flag'  # Don't remove, just flag
        }
    },
    'feature_engineering': {
        'polynomial_features': {
            'degree': 2,
            'interaction_only': True,
            'include_bias': False
        },
        'feature_selection': {
            'method': 'mutual_info',
            'k_features': 15
        }
    },
    'model_training': {
        'algorithm': 'isolation_forest',
        'hyperparameter_tuning': {
            'method': 'bayesian_optimization',
            'n_trials': 50,
            'cv_folds': 5
        }
    },
    'model_evaluation': {
        'metrics': ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score'],
        'cross_validation': True,
        'stability_testing': True
    }
}

# Execute pipeline
anomaly_pipeline.configure(pipeline_config)
pipeline_result = anomaly_pipeline.execute(data)

print("=== Pipeline Execution Results ===")
print(f"Pipeline status: {pipeline_result.status}")
print(f"Execution time: {pipeline_result.execution_time:.2f} seconds")
print(f"Anomalies detected: {pipeline_result.n_anomalies}")
print(f"Model performance metrics: {pipeline_result.performance_metrics}")
print(f"Feature importance: {pipeline_result.feature_importance}")
```

### AutoML Integration

```python
from pynomaly.packages.data_science import AutoMLService

# Create AutoML service
automl_service = AutoMLService()

# Configure AutoML for anomaly detection
automl_config = {
    'problem_type': 'anomaly_detection',
    'time_budget': 3600,  # 1 hour
    'algorithms_to_try': [
        'isolation_forest',
        'local_outlier_factor',
        'one_class_svm',
        'elliptic_envelope',
        'auto_encoder'
    ],
    'ensemble_methods': ['voting', 'stacking', 'blending'],
    'hyperparameter_optimization': {
        'method': 'optuna',
        'n_trials': 100,
        'pruning': True
    },
    'feature_engineering': {
        'auto_feature_generation': True,
        'feature_selection': True,
        'dimensionality_reduction': True
    },
    'validation_strategy': {
        'method': 'time_series_split' if 'date' in data.columns else 'k_fold',
        'n_splits': 5,
        'shuffle': True
    }
}

# Run AutoML
automl_result = automl_service.auto_optimize(data, automl_config)

print("=== AutoML Results ===")
print(f"Best algorithm: {automl_result.best_model.algorithm}")
print(f"Best parameters: {automl_result.best_model.parameters}")
print(f"Cross-validation score: {automl_result.best_score:.4f}")
print(f"Model selection reasoning: {automl_result.selection_reasoning}")

# Get model leaderboard
leaderboard = automl_result.get_leaderboard()
print(f"\n=== Model Leaderboard ===")
print(leaderboard.head(10))

# Use best model for predictions
best_detector = automl_result.best_model
automl_predictions = best_detector.detect(data)
```

## Data Quality Package

### Comprehensive Quality Assessment

```python
from pynomaly.packages.data_quality import DataQualityEngine, QualityRule

# Create data with various quality issues
np.random.seed(42)
problematic_data = pd.DataFrame({
    'id': range(1000),
    'age': np.random.normal(35, 10, 1000),
    'income': np.random.exponential(50000, 1000),
    'email': ['user@example.com'] * 950 + ['invalid_email'] * 50,
    'category': np.random.choice(['A', 'B', 'C', 'D'], 1000, p=[0.4, 0.3, 0.2, 0.1])
})

# Introduce quality issues
problematic_data.loc[np.random.choice(1000, 50, replace=False), 'age'] = np.nan
problematic_data.loc[np.random.choice(1000, 30, replace=False), 'income'] = -999  # Invalid values
problematic_data.loc[np.random.choice(1000, 20, replace=False), 'age'] = 150  # Outliers

# Initialize quality engine
quality_engine = DataQualityEngine()

# Define custom quality rules
quality_rules = [
    QualityRule.completeness_rule('age', min_completeness=0.95),
    QualityRule.range_rule('age', min_value=0, max_value=120),
    QualityRule.range_rule('income', min_value=0, max_value=1000000),
    QualityRule.format_rule('email', pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$'),
    QualityRule.uniqueness_rule('id', min_uniqueness=1.0),
    QualityRule.consistency_rule(['age', 'income'], 
                               condition='income should increase with age generally')
]

# Assess quality
quality_assessment = quality_engine.assess_quality(
    data=problematic_data,
    rules=quality_rules
)

print("=== Data Quality Assessment ===")
print(f"Overall quality score: {quality_assessment.overall_score:.2f}")
print(f"Completeness score: {quality_assessment.dimensions.completeness:.2f}")
print(f"Accuracy score: {quality_assessment.dimensions.accuracy:.2f}")
print(f"Consistency score: {quality_assessment.dimensions.consistency:.2f}")
print(f"Validity score: {quality_assessment.dimensions.validity:.2f}")

# Detailed quality issues
print(f"\n=== Quality Issues ===")
for issue in quality_assessment.issues:
    print(f"- {issue.severity}: {issue.description} (affects {issue.affected_rows} rows)")

# Quality improvement suggestions
improvement_suggestions = quality_engine.suggest_improvements(quality_assessment)
print(f"\n=== Improvement Suggestions ===")
for suggestion in improvement_suggestions:
    print(f"- {suggestion.action}: {suggestion.description}")
    print(f"  Expected improvement: {suggestion.expected_impact:.2f}")
```

### Automated Data Cleansing

```python
from pynomaly.packages.data_quality import DataCleansingEngine

# Create cleansing engine
cleansing_engine = DataCleansingEngine()

# Configure advanced cleansing
cleansing_config = {
    'missing_values': {
        'detection_method': 'comprehensive',  # Detect various missing patterns
        'imputation_strategy': 'adaptive',    # Choose best method per column
        'numerical_imputation': {
            'method': 'knn',
            'k_neighbors': 5,
            'weights': 'distance'
        },
        'categorical_imputation': {
            'method': 'most_frequent',
            'add_missing_indicator': True
        }
    },
    'outliers': {
        'detection_methods': ['z_score', 'iqr', 'isolation_forest'],
        'ensemble_method': 'voting',
        'treatment_strategy': 'cap_and_flag',
        'contamination': 0.05
    },
    'duplicates': {
        'detection_method': 'fuzzy_matching',
        'similarity_threshold': 0.85,
        'resolution_strategy': 'keep_most_complete'
    },
    'data_types': {
        'auto_inference': True,
        'datetime_formats': ['%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S'],
        'categorical_threshold': 0.05  # If < 5% unique values, treat as categorical
    },
    'format_standardization': {
        'phone_numbers': True,
        'email_addresses': True,
        'postal_codes': True,
        'names': True
    }
}

# Perform cleansing
cleansing_result = cleansing_engine.clean_data(
    data=problematic_data,
    config=cleansing_config
)

print("=== Data Cleansing Results ===")
print(f"Original data shape: {problematic_data.shape}")
print(f"Cleaned data shape: {cleansing_result.cleaned_data.shape}")
print(f"Rows modified: {cleansing_result.statistics.rows_modified}")
print(f"Missing values imputed: {cleansing_result.statistics.missing_values_imputed}")
print(f"Outliers treated: {cleansing_result.statistics.outliers_treated}")
print(f"Duplicates removed: {cleansing_result.statistics.duplicates_removed}")

# Get cleaning report
cleaning_report = cleansing_result.get_detailed_report()
print(f"\n=== Detailed Cleaning Report ===")
for operation in cleaning_report.operations:
    print(f"- {operation.type}: {operation.description}")
    print(f"  Affected columns: {operation.affected_columns}")
    print(f"  Success rate: {operation.success_rate:.2%}")

# Quality comparison
post_cleaning_quality = quality_engine.assess_quality(
    data=cleansing_result.cleaned_data,
    rules=quality_rules
)

print(f"\n=== Quality Improvement ===")
print(f"Before cleaning: {quality_assessment.overall_score:.2f}")
print(f"After cleaning: {post_cleaning_quality.overall_score:.2f}")
print(f"Improvement: {post_cleaning_quality.overall_score - quality_assessment.overall_score:.2f}")
```

### Real-time Quality Monitoring

```python
from pynomaly.packages.data_quality import QualityMonitor
import asyncio

# Create quality monitor
quality_monitor = QualityMonitor(
    reference_data=cleansing_result.cleaned_data,
    monitoring_rules=quality_rules
)

# Configure monitoring
monitoring_config = {
    'check_interval': 60,  # Check every minute
    'alert_thresholds': {
        'quality_degradation': 0.1,  # Alert if quality drops by 10%
        'anomaly_rate_change': 0.05,  # Alert if anomaly rate changes by 5%
        'data_drift': 0.05            # Alert if data drift exceeds 5%
    },
    'notification_channels': {
        'email': ['admin@company.com'],
        'slack': {'webhook_url': 'your_slack_webhook'},
        'dashboard': {'update_frequency': 30}
    },
    'data_profiling': {
        'enabled': True,
        'profile_interval': 300,  # Profile every 5 minutes
        'store_profiles': True
    }
}

quality_monitor.configure(monitoring_config)

# Simulate streaming data monitoring
async def simulate_data_stream():
    """Simulate incoming data stream for monitoring."""
    
    for i in range(100):  # Simulate 100 batches
        # Generate new batch (gradually degrading quality)
        batch_size = np.random.randint(50, 200)
        degradation_factor = 1 + (i / 100) * 0.3  # Gradual quality degradation
        
        new_batch = pd.DataFrame({
            'id': range(i * batch_size, (i + 1) * batch_size),
            'age': np.random.normal(35, 10 * degradation_factor, batch_size),
            'income': np.random.exponential(50000, batch_size),
            'email': ['user@example.com'] * int(batch_size * 0.9) + 
                    ['invalid_email'] * int(batch_size * 0.1),
            'category': np.random.choice(['A', 'B', 'C', 'D'], batch_size)
        })
        
        # Introduce increasing quality issues
        missing_rate = min(0.2, i / 500)  # Up to 20% missing
        outlier_rate = min(0.1, i / 1000)  # Up to 10% outliers
        
        # Add missing values
        missing_indices = np.random.choice(
            batch_size, 
            int(batch_size * missing_rate), 
            replace=False
        )
        new_batch.loc[missing_indices, 'age'] = np.nan
        
        # Add outliers
        outlier_indices = np.random.choice(
            batch_size, 
            int(batch_size * outlier_rate), 
            replace=False
        )
        new_batch.loc[outlier_indices, 'age'] = np.random.uniform(200, 300, len(outlier_indices))
        
        # Monitor batch
        monitoring_result = quality_monitor.monitor_batch(new_batch)
        
        if monitoring_result.alerts:
            print(f"Batch {i}: Quality alerts triggered!")
            for alert in monitoring_result.alerts:
                print(f"  - {alert.severity}: {alert.message}")
        
        # Simulate processing delay
        await asyncio.sleep(0.1)

# Run monitoring simulation
print("Starting quality monitoring simulation...")
# asyncio.run(simulate_data_stream())  # Uncomment to run simulation
```

## Data Profiling Package

### Comprehensive Data Profiling

```python
from pynomaly.packages.data_profiling import DataProfiler

# Create comprehensive dataset for profiling
np.random.seed(42)
profiling_data = pd.DataFrame({
    'transaction_id': [f'TXN_{i:06d}' for i in range(10000)],
    'timestamp': pd.date_range('2023-01-01', periods=10000, freq='5min'),
    'amount': np.random.lognormal(3, 1, 10000),
    'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'other'], 
                                        10000, p=[0.3, 0.15, 0.25, 0.2, 0.1]),
    'customer_age': np.random.normal(35, 12, 10000),
    'customer_income': np.random.normal(75000, 25000, 10000),
    'is_weekend': np.random.choice([True, False], 10000, p=[0.3, 0.7]),
    'location_risk_score': np.random.beta(2, 5, 10000),
    'previous_transactions_30d': np.random.poisson(15, 10000)
})

# Add some data quality issues for realistic profiling
profiling_data.loc[np.random.choice(10000, 100), 'amount'] = np.nan
profiling_data.loc[profiling_data['amount'] < 0, 'amount'] = np.abs(profiling_data.loc[profiling_data['amount'] < 0, 'amount'])

# Initialize profiler
profiler = DataProfiler()

# Comprehensive profiling
profile_config = {
    'statistical_profiling': {
        'enabled': True,
        'confidence_level': 0.95,
        'correlation_threshold': 0.1
    },
    'pattern_discovery': {
        'enabled': True,
        'temporal_patterns': True,
        'categorical_patterns': True,
        'numerical_patterns': True
    },
    'quality_assessment': {
        'enabled': True,
        'completeness_threshold': 0.95,
        'uniqueness_threshold': 0.8
    },
    'anomaly_hints': {
        'enabled': True,
        'contamination_estimation': True,
        'algorithm_recommendations': True
    }
}

# Generate comprehensive profile
comprehensive_profile = profiler.profile_dataset(
    data=profiling_data,
    config=profile_config
)

print("=== Data Profile Summary ===")
print(f"Dataset shape: {comprehensive_profile.shape}")
print(f"Data types: {comprehensive_profile.data_types}")
print(f"Missing value percentages: {comprehensive_profile.missing_percentages}")
print(f"Estimated data quality score: {comprehensive_profile.quality_score:.2f}")

# Statistical insights
print(f"\n=== Statistical Insights ===")
print(f"Numerical features summary:")
for feature, stats in comprehensive_profile.numerical_stats.items():
    print(f"  {feature}: mean={stats.mean:.2f}, std={stats.std:.2f}, skew={stats.skewness:.2f}")

print(f"\nCategorical features summary:")
for feature, stats in comprehensive_profile.categorical_stats.items():
    print(f"  {feature}: {stats.unique_count} unique values, entropy={stats.entropy:.2f}")

# Pattern discovery results
print(f"\n=== Pattern Discovery ===")
if comprehensive_profile.patterns.temporal_patterns:
    print("Temporal patterns found:")
    for pattern in comprehensive_profile.patterns.temporal_patterns:
        print(f"  - {pattern.description} (strength: {pattern.strength:.2f})")

if comprehensive_profile.patterns.categorical_patterns:
    print("Categorical patterns found:")
    for pattern in comprehensive_profile.patterns.categorical_patterns:
        print(f"  - {pattern.description} (frequency: {pattern.frequency:.2f})")

# Anomaly detection recommendations
print(f"\n=== Anomaly Detection Recommendations ===")
recommendations = comprehensive_profile.anomaly_recommendations
print(f"Recommended algorithms: {recommendations.algorithms}")
print(f"Estimated contamination rate: {recommendations.contamination_estimate:.4f}")
print(f"Key features for detection: {recommendations.important_features}")
print(f"Preprocessing suggestions: {recommendations.preprocessing_suggestions}")

# Generate HTML report
html_report = profiler.generate_html_report(comprehensive_profile)
# with open('data_profile_report.html', 'w') as f:
#     f.write(html_report)
print("HTML report generated successfully!")
```

### Advanced Pattern Discovery

```python
from pynomaly.packages.data_profiling import PatternDiscoveryService

# Initialize pattern discovery service
pattern_service = PatternDiscoveryService()

# Advanced pattern discovery configuration
pattern_config = {
    'temporal_analysis': {
        'seasonality_detection': True,
        'trend_analysis': True,
        'cyclical_patterns': True,
        'anomalous_time_periods': True
    },
    'association_rules': {
        'min_support': 0.01,
        'min_confidence': 0.5,
        'max_antecedents': 3
    },
    'sequence_patterns': {
        'min_pattern_length': 2,
        'max_pattern_length': 5,
        'min_frequency': 0.001
    },
    'correlation_analysis': {
        'methods': ['pearson', 'spearman', 'mutual_info'],
        'threshold': 0.1,
        'network_analysis': True
    }
}

# Discover patterns
pattern_results = pattern_service.discover_patterns(
    data=profiling_data,
    config=pattern_config
)

print("=== Advanced Pattern Discovery Results ===")

# Temporal patterns
if pattern_results.temporal_patterns:
    print("Temporal patterns:")
    for pattern in pattern_results.temporal_patterns:
        print(f"  - {pattern.type}: {pattern.description}")
        print(f"    Period: {pattern.period}, Strength: {pattern.strength:.3f}")

# Association rules
if pattern_results.association_rules:
    print("\nAssociation rules:")
    for rule in pattern_results.association_rules[:10]:  # Top 10 rules
        print(f"  - {rule.antecedent} => {rule.consequent}")
        print(f"    Support: {rule.support:.3f}, Confidence: {rule.confidence:.3f}")

# Correlation network
correlation_network = pattern_results.correlation_network
print(f"\nCorrelation network:")
print(f"  Number of nodes: {correlation_network.n_nodes}")
print(f"  Number of edges: {correlation_network.n_edges}")
print(f"  Strongly connected components: {correlation_network.n_components}")

# Anomalous patterns
if pattern_results.anomalous_patterns:
    print("\nAnomalous patterns detected:")
    for anomaly_pattern in pattern_results.anomalous_patterns:
        print(f"  - Type: {anomaly_pattern.type}")
        print(f"    Description: {anomaly_pattern.description}")
        print(f"    Anomaly score: {anomaly_pattern.anomaly_score:.3f}")
```

## Data Transformation Package

### Comprehensive Data Pipeline

```python
from pynomaly.packages.data_transformation import (
    DataTransformationEngine,
    TransformationPipeline,
    NumericalFeatureProcessor,
    CategoricalFeatureProcessor,
    TemporalFeatureProcessor
)

# Create transformation engine
transformation_engine = DataTransformationEngine()

# Define comprehensive transformation pipeline
pipeline_config = {
    'data_validation': {
        'check_schema': True,
        'check_data_types': True,
        'check_missing_values': True,
        'validation_rules': [
            {'column': 'amount', 'rule': 'positive'},
            {'column': 'customer_age', 'rule': 'range', 'min': 18, 'max': 100},
            {'column': 'timestamp', 'rule': 'datetime_format'}
        ]
    },
    'missing_value_handling': {
        'strategy': 'adaptive',
        'numerical_strategy': 'knn_imputation',
        'categorical_strategy': 'mode_with_indicator',
        'temporal_strategy': 'interpolation'
    },
    'outlier_treatment': {
        'detection_methods': ['isolation_forest', 'z_score', 'iqr'],
        'ensemble_voting': True,
        'treatment_strategy': 'cap_and_flag',
        'contamination': 0.05
    },
    'feature_engineering': {
        'numerical_transformations': {
            'log_transform': ['amount'],
            'power_transform': ['customer_income'],
            'polynomial_features': {
                'features': ['customer_age', 'previous_transactions_30d'],
                'degree': 2,
                'interaction_only': True
            },
            'binning': {
                'amount': {'bins': 10, 'strategy': 'quantile'},
                'location_risk_score': {'bins': 5, 'strategy': 'uniform'}
            }
        },
        'categorical_transformations': {
            'encoding_strategy': 'adaptive',
            'high_cardinality_method': 'target_encoding',
            'low_cardinality_method': 'one_hot',
            'rare_category_threshold': 0.01
        },
        'temporal_transformations': {
            'extract_components': ['hour', 'day_of_week', 'month', 'quarter'],
            'cyclical_encoding': True,
            'lag_features': [1, 7, 30],
            'rolling_statistics': {
                'windows': [7, 30],
                'statistics': ['mean', 'std', 'min', 'max']
            }
        }
    },
    'feature_scaling': {
        'method': 'robust_scaler',
        'feature_range': (-1, 1),
        'quantile_range': (25.0, 75.0)
    },
    'feature_selection': {
        'method': 'mutual_info_regression',
        'k_features': 20,
        'remove_correlated': True,
        'correlation_threshold': 0.95
    },
    'dimensionality_reduction': {
        'method': 'pca',
        'n_components': 0.95,  # Retain 95% of variance
        'whiten': True
    }
}

# Create and configure pipeline
transformation_pipeline = TransformationPipeline()
transformation_pipeline.configure(pipeline_config)

# Execute transformation pipeline
transformation_result = transformation_pipeline.execute(profiling_data)

print("=== Transformation Pipeline Results ===")
print(f"Original shape: {profiling_data.shape}")
print(f"Transformed shape: {transformation_result.transformed_data.shape}")
print(f"Features created: {transformation_result.n_features_created}")
print(f"Features removed: {transformation_result.n_features_removed}")
print(f"Processing time: {transformation_result.processing_time:.2f} seconds")

# Detailed transformation report
transformation_report = transformation_result.get_detailed_report()
print(f"\n=== Transformation Report ===")
for stage_name, stage_result in transformation_report.stage_results.items():
    print(f"{stage_name}:")
    print(f"  Status: {stage_result.status}")
    print(f"  Duration: {stage_result.duration:.2f}s")
    print(f"  Changes: {stage_result.changes_summary}")

# Feature importance after transformation
feature_importance = transformation_result.feature_importance
print(f"\n=== Feature Importance (Top 10) ===")
for feature, importance in feature_importance.head(10).items():
    print(f"  {feature}: {importance:.4f}")
```

### Custom Transformation Functions

```python
from pynomaly.packages.data_transformation import CustomTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class BusinessRuleTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer implementing business logic."""
    
    def __init__(self, business_rules=None):
        self.business_rules = business_rules or {}
        
    def fit(self, X, y=None):
        # Store fitting parameters if needed
        self.feature_medians_ = X.select_dtypes(include=[np.number]).median()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # Business rule 1: Flag high-risk transactions
        if 'amount' in X_transformed.columns and 'customer_income' in X_transformed.columns:
            X_transformed['high_amount_flag'] = (
                X_transformed['amount'] > X_transformed['customer_income'] * 0.1
            ).astype(int)
        
        # Business rule 2: Create customer segment based on transaction behavior
        if 'previous_transactions_30d' in X_transformed.columns:
            conditions = [
                (X_transformed['previous_transactions_30d'] <= 5),
                (X_transformed['previous_transactions_30d'] <= 15),
                (X_transformed['previous_transactions_30d'] <= 30),
                (X_transformed['previous_transactions_30d'] > 30)
            ]
            choices = ['low_activity', 'medium_activity', 'high_activity', 'very_high_activity']
            X_transformed['customer_segment'] = np.select(conditions, choices, default='unknown')
        
        # Business rule 3: Risk score based on multiple factors
        risk_factors = []
        if 'location_risk_score' in X_transformed.columns:
            risk_factors.append(X_transformed['location_risk_score'])
        if 'is_weekend' in X_transformed.columns:
            risk_factors.append(X_transformed['is_weekend'].astype(float) * 0.3)
        
        if risk_factors:
            X_transformed['composite_risk_score'] = np.mean(risk_factors, axis=0)
        
        return X_transformed

# Integrate custom transformer into pipeline
custom_transformer = BusinessRuleTransformer()

# Add to transformation pipeline
transformation_pipeline.add_custom_stage(
    'business_rules', 
    custom_transformer,
    position='after_feature_engineering'
)

# Re-execute pipeline with custom transformer
custom_transformation_result = transformation_pipeline.execute(profiling_data)

print("=== Custom Transformation Results ===")
print(f"New features created by business rules:")
new_features = set(custom_transformation_result.transformed_data.columns) - set(profiling_data.columns)
for feature in new_features:
    print(f"  - {feature}")
```

## Advanced Integration

### End-to-End Anomaly Detection Workflow

```python
from pynomaly.packages.data_profiling import DataProfiler
from pynomaly.packages.data_quality import DataQualityEngine
from pynomaly.packages.data_transformation import DataTransformationEngine
from pynomaly.packages.data_science import AutoMLService
from pynomaly import AnomalyDetector

class ComprehensiveAnomalyDetectionWorkflow:
    """End-to-end anomaly detection workflow integrating all packages."""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.profiler = DataProfiler()
        self.quality_engine = DataQualityEngine()
        self.transformer = DataTransformationEngine()
        self.automl = AutoMLService()
        self.detector = None
        self.workflow_history = []
    
    def _default_config(self):
        return {
            'quality_threshold': 0.8,
            'profiling_enabled': True,
            'auto_feature_engineering': True,
            'automl_enabled': True,
            'ensemble_detection': True,
            'explain_results': True
        }
    
    def execute_workflow(self, data, target_contamination=None):
        """Execute complete anomaly detection workflow."""
        
        workflow_results = {}
        
        # Step 1: Data Profiling
        if self.config['profiling_enabled']:
            print("Step 1: Data Profiling...")
            profile = self.profiler.profile_dataset(data)
            workflow_results['profile'] = profile
            
            # Use profile insights for configuration
            if target_contamination is None:
                target_contamination = profile.anomaly_recommendations.contamination_estimate
        
        # Step 2: Data Quality Assessment
        print("Step 2: Data Quality Assessment...")
        quality_report = self.quality_engine.assess_quality(data)
        workflow_results['quality_report'] = quality_report
        
        if quality_report.overall_score < self.config['quality_threshold']:
            print(f"Data quality score ({quality_report.overall_score:.2f}) below threshold. Applying improvements...")
            improved_data = self.quality_engine.improve_quality(data)
            workflow_results['quality_improvements'] = True
            data = improved_data
        
        # Step 3: Data Transformation
        print("Step 3: Data Transformation...")
        if self.config['profiling_enabled']:
            # Configure transformation based on profile
            transformation_config = self._create_transformation_config(profile)
        else:
            transformation_config = self._default_transformation_config()
        
        transformation_result = self.transformer.comprehensive_transform(
            data, transformation_config
        )
        workflow_results['transformation_result'] = transformation_result
        transformed_data = transformation_result.transformed_data
        
        # Step 4: Algorithm Selection and Training
        print("Step 4: Algorithm Selection and Training...")
        if self.config['automl_enabled']:
            # Use AutoML for optimal algorithm selection
            automl_config = {
                'contamination': target_contamination,
                'time_budget': 1800,  # 30 minutes
                'data_characteristics': profile.characteristics if self.config['profiling_enabled'] else None
            }
            
            automl_result = self.automl.auto_optimize(transformed_data, automl_config)
            self.detector = automl_result.best_model
            workflow_results['automl_result'] = automl_result
        else:
            # Use ensemble approach
            self.detector = AnomalyDetector(
                algorithm='AutoEnsemble',
                contamination=target_contamination,
                ensemble_method='weighted_voting'
            )
            self.detector.fit(transformed_data)
        
        # Step 5: Anomaly Detection
        print("Step 5: Anomaly Detection...")
        detection_results = self.detector.detect(transformed_data)
        workflow_results['detection_results'] = detection_results
        
        # Step 6: Results Explanation (if requested)
        if self.config['explain_results']:
            print("Step 6: Results Explanation...")
            explanations = self._explain_results(
                transformed_data, detection_results, transformation_result
            )
            workflow_results['explanations'] = explanations
        
        # Step 7: Workflow Summary
        print("Step 7: Generating Workflow Summary...")
        summary = self._generate_workflow_summary(workflow_results)
        workflow_results['summary'] = summary
        
        self.workflow_history.append(workflow_results)
        return workflow_results
    
    def _create_transformation_config(self, profile):
        """Create transformation config based on data profile."""
        config = {
            'missing_value_handling': {
                'strategy': 'knn_imputation' if profile.missing_rate < 0.1 else 'drop_high_missing'
            },
            'feature_engineering': {
                'auto_feature_generation': self.config['auto_feature_engineering'],
                'polynomial_degree': 2 if profile.n_features < 50 else 1
            },
            'scaling': {
                'method': 'robust_scaler' if profile.has_outliers else 'standard_scaler'
            },
            'dimensionality_reduction': {
                'apply': profile.n_features > 100,
                'method': 'pca',
                'n_components': 0.95
            }
        }
        return config
    
    def _default_transformation_config(self):
        """Default transformation configuration."""
        return {
            'missing_value_handling': {'strategy': 'median_imputation'},
            'feature_engineering': {'auto_feature_generation': True},
            'scaling': {'method': 'robust_scaler'},
            'dimensionality_reduction': {'apply': False}
        }
    
    def _explain_results(self, data, results, transformation_result):
        """Generate explanations for anomaly detection results."""
        from pynomaly.explainability import AnomalyExplainer
        
        explainer = AnomalyExplainer(self.detector)
        explanations = {}
        
        # Explain top anomalies
        top_anomalies = results.anomaly_indices[:10]  # Top 10 anomalies
        
        for idx in top_anomalies:
            explanation = explainer.explain_anomaly(
                data=data,
                anomaly_index=idx,
                feature_names=transformation_result.feature_names
            )
            explanations[idx] = explanation
        
        return explanations
    
    def _generate_workflow_summary(self, workflow_results):
        """Generate comprehensive workflow summary."""
        summary = {
            'workflow_status': 'completed',
            'execution_time': sum([
                workflow_results.get('transformation_result', {}).get('processing_time', 0),
                workflow_results.get('automl_result', {}).get('training_time', 0)
            ]),
            'data_quality_score': workflow_results['quality_report'].overall_score,
            'anomalies_detected': workflow_results['detection_results'].n_anomalies,
            'detection_confidence': workflow_results['detection_results'].confidence,
            'key_insights': []
        }
        
        # Add key insights
        if 'profile' in workflow_results:
            profile = workflow_results['profile']
            summary['key_insights'].extend([
                f"Dataset has {profile.n_features} features and {profile.n_samples} samples",
                f"Estimated contamination rate: {profile.anomaly_recommendations.contamination_estimate:.2%}",
                f"Recommended algorithms: {profile.anomaly_recommendations.algorithms}"
            ])
        
        if 'automl_result' in workflow_results:
            automl_result = workflow_results['automl_result']
            summary['key_insights'].append(
                f"Best algorithm: {automl_result.best_model.algorithm} with score {automl_result.best_score:.3f}"
            )
        
        return summary

# Example usage of comprehensive workflow
workflow = ComprehensiveAnomalyDetectionWorkflow(config={
    'quality_threshold': 0.8,
    'profiling_enabled': True,
    'auto_feature_engineering': True,
    'automl_enabled': True,
    'explain_results': True
})

# Execute workflow
print("Executing Comprehensive Anomaly Detection Workflow...")
workflow_results = workflow.execute_workflow(profiling_data)

print("=== Workflow Summary ===")
summary = workflow_results['summary']
print(f"Workflow Status: {summary['workflow_status']}")
print(f"Execution Time: {summary['execution_time']:.2f} seconds")
print(f"Data Quality Score: {summary['data_quality_score']:.2f}")
print(f"Anomalies Detected: {summary['anomalies_detected']}")
print(f"Detection Confidence: {summary['detection_confidence']:.3f}")

print(f"\nKey Insights:")
for insight in summary['key_insights']:
    print(f"  - {insight}")

# Access detailed results
if 'explanations' in workflow_results:
    print(f"\n=== Top Anomaly Explanations ===")
    for idx, explanation in list(workflow_results['explanations'].items())[:3]:
        print(f"Anomaly {idx}:")
        print(f"  Score: {explanation.anomaly_score:.3f}")
        print(f"  Top contributing features:")
        for feature, contribution in explanation.feature_contributions.items():
            if abs(contribution) > 0.1:  # Only show significant contributions
                print(f"    {feature}: {contribution:.3f}")
```

This comprehensive examples document demonstrates the full capabilities of Pynomaly's API across all packages and use cases. Each example includes detailed configuration options, error handling, and integration patterns that users can adapt for their specific needs.