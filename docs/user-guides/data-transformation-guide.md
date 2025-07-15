# Data Transformation Package User Guide

## Overview

The Data Transformation package provides comprehensive data preprocessing, transformation pipelines, and feature processing capabilities. It serves as the data preparation layer for anomaly detection and machine learning workflows.

## Quick Start

### Installation

```bash
pip install pynomaly[standard]
```

### Basic Usage

```python
from pynomaly.packages.data_transformation import DataTransformationEngine
import pandas as pd

# Load your dataset
data = pd.read_csv('your_data.csv')

# Create transformation engine
transformer = DataTransformationEngine()

# Apply basic transformations
transformed_data = transformer.transform(data)

# View transformation results
print(f"Original shape: {data.shape}")
print(f"Transformed shape: {transformed_data.shape}")
```

## Core Features

### 1. Data Preprocessing Pipeline

#### Basic Preprocessing

```python
from pynomaly.packages.data_transformation import DataPreprocessor

# Create preprocessor
preprocessor = DataPreprocessor()

# Configure preprocessing steps
preprocessing_config = {
    'handle_missing_values': {
        'strategy': 'impute',
        'numerical_method': 'median',
        'categorical_method': 'mode'
    },
    'normalize_data': {
        'method': 'standard_scaler',
        'feature_range': (-1, 1)
    },
    'encode_categorical': {
        'method': 'one_hot',
        'handle_unknown': 'ignore',
        'drop_first': True
    },
    'handle_outliers': {
        'method': 'iqr',
        'factor': 1.5,
        'action': 'cap'
    }
}

# Apply preprocessing
preprocessed_data = preprocessor.preprocess(data, preprocessing_config)
```

#### Advanced Preprocessing

```python
# Custom preprocessing pipeline
from pynomaly.packages.data_transformation import PreprocessingPipeline

pipeline = PreprocessingPipeline()

# Add custom preprocessing steps
pipeline.add_step('data_validation', DataValidationStep())
pipeline.add_step('missing_value_imputation', AdvancedImputationStep())
pipeline.add_step('feature_scaling', RobustScalingStep())
pipeline.add_step('categorical_encoding', TargetEncodingStep())
pipeline.add_step('outlier_treatment', IsolationForestOutlierStep())

# Execute pipeline
result = pipeline.execute(data)
processed_data = result.transformed_data
transformation_report = result.report
```

### 2. Feature Processing Framework

#### Numerical Feature Processing

```python
from pynomaly.packages.data_transformation import NumericalFeatureProcessor

# Create numerical processor
num_processor = NumericalFeatureProcessor()

# Apply transformations
numerical_transformations = {
    'log_transform': ['sales_amount', 'revenue'],
    'power_transform': ['skewed_feature1', 'skewed_feature2'],
    'polynomial_features': {
        'features': ['feature1', 'feature2'],
        'degree': 2,
        'interaction_only': False
    },
    'binning': {
        'age': {'bins': 5, 'strategy': 'quantile'},
        'income': {'bins': [0, 25000, 50000, 100000, float('inf')]}
    }
}

# Transform numerical features
transformed_numerical = num_processor.transform(data, numerical_transformations)
```

#### Categorical Feature Processing

```python
from pynomaly.packages.data_transformation import CategoricalFeatureProcessor

# Create categorical processor
cat_processor = CategoricalFeatureProcessor()

# Configure categorical transformations
categorical_config = {
    'encoding_strategies': {
        'high_cardinality': 'target_encoding',
        'low_cardinality': 'one_hot',
        'ordinal': 'ordinal_encoding'
    },
    'target_encoding_config': {
        'smoothing': 1.0,
        'min_samples_leaf': 10,
        'noise_level': 0.01
    },
    'rare_category_handling': {
        'threshold': 0.01,
        'replace_with': 'other'
    }
}

# Transform categorical features
transformed_categorical = cat_processor.transform(data, categorical_config)
```

#### Temporal Feature Processing

```python
from pynomaly.packages.data_transformation import TemporalFeatureProcessor

# Create temporal processor
temporal_processor = TemporalFeatureProcessor()

# Extract temporal features
temporal_config = {
    'datetime_features': ['timestamp', 'created_date'],
    'extract_components': ['year', 'month', 'day', 'hour', 'dayofweek'],
    'create_cyclical_features': True,
    'lag_features': [1, 7, 30],
    'rolling_statistics': {
        'windows': [7, 30, 90],
        'functions': ['mean', 'std', 'min', 'max']
    }
}

# Transform temporal features
transformed_temporal = temporal_processor.transform(data, temporal_config)
```

### 3. Transformation Pipeline Framework

#### Pipeline Creation and Configuration

```python
from pynomaly.packages.data_transformation import TransformationPipeline

# Create transformation pipeline
pipeline = TransformationPipeline()

# Define pipeline stages
pipeline_config = {
    'stages': [
        {
            'name': 'data_validation',
            'processor': 'DataValidationProcessor',
            'config': {'strict_mode': True}
        },
        {
            'name': 'missing_value_handling',
            'processor': 'MissingValueProcessor',
            'config': {
                'strategy': 'advanced_imputation',
                'imputation_method': 'knn'
            }
        },
        {
            'name': 'feature_engineering',
            'processor': 'FeatureEngineeringProcessor',
            'config': {
                'create_interactions': True,
                'polynomial_degree': 2
            }
        },
        {
            'name': 'feature_scaling',
            'processor': 'ScalingProcessor',
            'config': {
                'method': 'robust_scaler',
                'quantile_range': (25.0, 75.0)
            }
        }
    ],
    'parallel_execution': True,
    'error_handling': 'continue_on_error',
    'logging_level': 'INFO'
}

# Configure pipeline
pipeline.configure(pipeline_config)

# Execute pipeline
pipeline_result = pipeline.execute(data)
```

#### Dynamic Pipeline Construction

```python
# Build pipeline dynamically based on data characteristics
from pynomaly.packages.data_transformation import PipelineBuilder

builder = PipelineBuilder()

# Analyze data and recommend pipeline
data_analysis = builder.analyze_data(data)
recommended_pipeline = builder.recommend_pipeline(data_analysis)

print(f"Recommended stages: {recommended_pipeline.stages}")
print(f"Estimated processing time: {recommended_pipeline.estimated_time}")

# Build and execute recommended pipeline
pipeline = builder.build_pipeline(recommended_pipeline)
result = pipeline.execute(data)
```

## Advanced Features

### 1. Custom Transformation Functions

#### Creating Custom Transformers

```python
from pynomaly.packages.data_transformation import BaseTransformer

class BusinessRuleTransformer(BaseTransformer):
    def __init__(self, business_rules):
        self.business_rules = business_rules
    
    def fit(self, X, y=None):
        # Fit transformation parameters
        return self
    
    def transform(self, X):
        # Apply business rules
        transformed_X = X.copy()
        for rule in self.business_rules:
            transformed_X = rule.apply(transformed_X)
        return transformed_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Use custom transformer
business_rules = [
    BusinessRule('age_validation', lambda x: x['age'].clip(0, 120)),
    BusinessRule('income_normalization', lambda x: x['income'] / x['cost_of_living'])
]

custom_transformer = BusinessRuleTransformer(business_rules)
transformed_data = custom_transformer.fit_transform(data)
```

#### Transformation Templates

```python
from pynomaly.packages.data_transformation import TransformationTemplate

# Pre-built transformation templates
templates = {
    'anomaly_detection_preprocessing': TransformationTemplate([
        'missing_value_imputation',
        'outlier_detection_and_treatment',
        'feature_scaling',
        'dimensionality_reduction'
    ]),
    'time_series_preprocessing': TransformationTemplate([
        'temporal_feature_extraction',
        'lag_feature_creation',
        'seasonal_decomposition',
        'trend_removal'
    ]),
    'categorical_heavy_preprocessing': TransformationTemplate([
        'rare_category_handling',
        'target_encoding',
        'feature_hashing',
        'category_embedding'
    ])
}

# Apply template
template = templates['anomaly_detection_preprocessing']
preprocessed_data = template.apply(data)
```

### 2. Data Quality Integration

```python
from pynomaly.packages.data_transformation import QualityAwareTransformer

# Create quality-aware transformer
qa_transformer = QualityAwareTransformer()

# Configure quality checks
quality_config = {
    'data_quality_gates': {
        'min_completeness': 0.95,
        'max_outlier_ratio': 0.1,
        'min_uniqueness': 0.8
    },
    'quality_actions': {
        'on_quality_failure': 'warn_and_continue',  # or 'stop', 'auto_fix'
        'quality_improvement': True
    }
}

# Transform with quality monitoring
transformation_result = qa_transformer.transform_with_quality_check(
    data, 
    quality_config
)

if transformation_result.quality_passed:
    transformed_data = transformation_result.data
else:
    print(f"Quality issues: {transformation_result.quality_issues}")
```

### 3. Streaming Data Transformation

```python
from pynomaly.packages.data_transformation import StreamingTransformer

# Create streaming transformer
streaming_transformer = StreamingTransformer(
    buffer_size=1000,
    processing_interval='10s'
)

# Configure for streaming
streaming_config = {
    'real_time_scaling': True,
    'adaptive_imputation': True,
    'concept_drift_detection': True,
    'model_update_frequency': '1h'
}

# Process streaming data
async def process_stream():
    async for batch in data_stream:
        transformed_batch = await streaming_transformer.transform_batch(
            batch, 
            streaming_config
        )
        # Process transformed batch
        await downstream_processor.process(transformed_batch)
```

## Performance Optimization

### Large Dataset Processing

```python
# Configure for large datasets
transformer.configure_large_dataset_processing(
    chunk_size=50000,
    parallel_processing=True,
    memory_efficient=True,
    use_sparse_matrices=True
)

# Distributed processing
transformer.enable_distributed_processing(
    framework='dask',
    cluster_config={
        'n_workers': 8,
        'memory_per_worker': '4GB'
    }
)

# Process large dataset
large_transformed_data = transformer.transform_large_dataset(large_data)
```

### Caching and Persistence

```python
# Enable transformation caching
transformer.enable_caching(
    cache_backend='redis',
    cache_ttl=3600,
    cache_key_strategy='content_hash'
)

# Persist transformation pipeline
transformer.save_pipeline('transformation_pipeline.pkl')

# Load and reuse pipeline
loaded_transformer = DataTransformationEngine.load_pipeline(
    'transformation_pipeline.pkl'
)
```

## Integration Patterns

### Integration with Data Profiling

```python
from pynomaly.packages.data_profiling import DataProfiler

# Profile data first
profiler = DataProfiler()
profile = profiler.profile_dataset(data)

# Configure transformation based on profile
transformer.configure_from_profile(profile)

# Apply profile-optimized transformations
optimized_transformed_data = transformer.transform(data)
```

### Integration with Data Quality

```python
from pynomaly.packages.data_quality import DataQualityEngine

# Ensure quality before transformation
quality_engine = DataQualityEngine()
quality_report = quality_engine.assess_quality(data)

if quality_report.overall_score >= 0.8:
    # Proceed with transformation
    transformed_data = transformer.transform(data)
    
    # Validate transformation quality
    post_transform_quality = quality_engine.assess_quality(transformed_data)
    
    if post_transform_quality.overall_score < quality_report.overall_score:
        print("Warning: Transformation may have degraded data quality")
```

## Configuration Management

### Transformation Profiles

```python
# Define transformation profiles for different use cases
profiles = {
    'anomaly_detection': {
        'missing_value_strategy': 'advanced_imputation',
        'scaling_method': 'robust_scaler',
        'outlier_treatment': 'isolation_forest',
        'feature_selection': 'mutual_info'
    },
    'time_series_analysis': {
        'temporal_features': True,
        'lag_features': [1, 7, 30],
        'seasonal_decomposition': True,
        'trend_removal': True
    },
    'high_dimensional': {
        'dimensionality_reduction': 'pca',
        'feature_selection': 'variance_threshold',
        'sparse_matrices': True
    }
}

# Apply profile
transformer.apply_profile('anomaly_detection')
```

### Environment-Specific Configuration

```python
# Development environment
dev_config = TransformationConfig(
    verbose_logging=True,
    error_tolerance=0.1,
    performance_monitoring=True
)

# Production environment
prod_config = TransformationConfig(
    verbose_logging=False,
    error_tolerance=0.0,
    performance_monitoring=False,
    optimization_level='aggressive'
)
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Datasets**
   ```python
   # Use chunked processing
   transformer.enable_chunked_processing(chunk_size=10000)
   
   # Use sparse matrices for high-dimensional data
   transformer.enable_sparse_processing()
   ```

2. **Performance Issues**
   ```python
   # Enable parallel processing
   transformer.enable_parallel_processing(n_jobs=4)
   
   # Use approximate algorithms for large datasets
   transformer.enable_approximate_algorithms()
   ```

3. **Data Type Issues**
   ```python
   # Automatic data type optimization
   transformer.enable_dtype_optimization()
   
   # Handle mixed data types
   transformer.configure_mixed_type_handling('coerce')
   ```

## API Reference

### Core Classes

- `DataTransformationEngine`: Main transformation engine
- `TransformationPipeline`: Pipeline framework
- `DataPreprocessor`: Data preprocessing utilities
- `FeatureProcessor`: Feature processing framework

### Key Methods

- `transform(data)`: Apply transformations
- `fit_transform(data)`: Fit and transform
- `create_pipeline()`: Create transformation pipeline
- `configure_processing()`: Configure processing options

## Complete Example

```python
import pandas as pd
from pynomaly.packages.data_transformation import (
    DataTransformationEngine,
    TransformationPipeline,
    NumericalFeatureProcessor,
    CategoricalFeatureProcessor
)

# Load data
data = pd.read_csv('customer_behavior.csv')

# Create transformation engine
transformer = DataTransformationEngine()

# Configure comprehensive transformation
transformation_config = {
    'preprocessing': {
        'handle_missing_values': {
            'numerical_strategy': 'knn_imputation',
            'categorical_strategy': 'mode_imputation'
        },
        'outlier_treatment': {
            'method': 'isolation_forest',
            'contamination': 0.1
        }
    },
    'numerical_features': {
        'scaling': 'robust_scaler',
        'transformations': ['log', 'sqrt'],
        'polynomial_degree': 2
    },
    'categorical_features': {
        'encoding': 'target_encoding',
        'rare_category_threshold': 0.01
    },
    'feature_engineering': {
        'create_interactions': True,
        'dimensionality_reduction': {
            'method': 'pca',
            'n_components': 0.95
        }
    }
}

# Apply transformations
transformation_result = transformer.comprehensive_transform(
    data, 
    transformation_config
)

print("=== Transformation Results ===")
print(f"Original shape: {data.shape}")
print(f"Transformed shape: {transformation_result.data.shape}")
print(f"Features created: {transformation_result.new_features}")
print(f"Features removed: {transformation_result.removed_features}")
print(f"Transformation time: {transformation_result.processing_time:.2f}s")

# Access transformed data
transformed_data = transformation_result.data

# Get transformation report
transformation_report = transformation_result.report
print(f"Missing values handled: {transformation_report.missing_values_handled}")
print(f"Outliers treated: {transformation_report.outliers_treated}")
print(f"Categorical features encoded: {transformation_report.categorical_encoded}")

# Save transformation pipeline for reuse
transformer.save_pipeline('customer_transformation_pipeline.pkl')

print("Transformation pipeline saved for future use!")
```

For more detailed examples and advanced usage, see the [API Reference documentation](../api-reference/).