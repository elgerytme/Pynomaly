# Data Processing and Dataset Management Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🟢 [Basic Usage](README.md) > 📊 Datasets

---


This comprehensive guide covers data processing, dataset management, and optimization strategies in Pynomaly. Learn how to leverage high-performance data loaders, handle large datasets, and optimize processing for different use cases.

## Table of Contents

1. [Data Loader Overview](#data-loader-overview)
2. [Dataset Formats and Loaders](#dataset-formats-and-loaders)
3. [High-Performance Processing](#high-performance-processing)
4. [Data Validation and Quality](#data-validation-and-quality)
5. [Streaming and Large Datasets](#streaming-and-large-datasets)
6. [Performance Optimization](#performance-optimization)
7. [Production Patterns](#production-patterns)

## Data Loader Overview

Pynomaly provides multiple data loaders optimized for different scenarios:

- **Pandas Loader**: Standard DataFrame operations, excellent for small-medium datasets
- **Polars Loader**: High-performance alternative with lazy evaluation and multi-threading
- **Arrow Loader**: Columnar processing with native compute functions
- **Spark Loader**: Distributed processing for big data scenarios
- **Auto Loader**: Intelligent selection based on file size and format

### Quick Start Example

```python
from pynomaly.infrastructure.data_loaders import load_auto
import asyncio

async def quick_data_loading():
    # Auto-select optimal loader based on file characteristics
    dataset = await load_auto("data/transactions.csv")
    print(f"Loaded {dataset.n_samples} samples with {dataset.n_features} features")
    print(f"Dataset size: {dataset.memory_usage_mb:.2f} MB")

# Run the example
asyncio.run(quick_data_loading())
```

## Dataset Formats and Loaders

### CSV Data Processing

#### Standard CSV Loading (Pandas)
```python
from pynomaly.infrastructure.data_loaders import CSVLoader

# Basic CSV loading
loader = CSVLoader()
dataset = await loader.load_async("data/sensor_readings.csv")

# Advanced CSV options
dataset = await loader.load_async(
    "data/large_dataset.csv",
    encoding="utf-8",
    delimiter=",",
    chunk_size=10000,  # Process in chunks
    sample_rows=None,  # Load all rows
    target_column="is_anomaly"
)
    
    print(f"Loaded {dataset.sample_count} samples")
    print(f"Features: {dataset.feature_count}")
    print(f"Data quality: {dataset.quality_score:.2f}")
    
    return dataset

# Usage
dataset = asyncio.run(load_single_dataset())
```

### High-Performance Loading

```python
from pynomaly.infrastructure.loaders import PolarsLoader, PyArrowLoader

class HighPerformanceDataManager:
    """Optimized data loading for large datasets."""
    
    def __init__(self):
        self.polars_loader = PolarsLoader()
        self.arrow_loader = PyArrowLoader()
    
    async def load_large_dataset(self, file_path, chunk_size=50000):
        """Load large dataset with chunking."""
        
        if file_path.endswith('.parquet'):
            # Use PyArrow for Parquet files
            return await self.arrow_loader.load_with_chunks(
                file_path, 
                chunk_size=chunk_size
            )
        
        elif file_path.endswith('.csv'):
            # Use Polars for CSV files
            return await self.polars_loader.load_streaming(
                file_path,
                lazy=True,  # Lazy evaluation
                streaming=True
            )
        
        else:
            raise ValueError(f"Unsupported format: {file_path}")
    
    async def load_multiple_formats(self, file_paths):
        """Load multiple files of different formats."""
        
        datasets = []
        
        for path in file_paths:
            if path.endswith('.parquet'):
                df = await self.arrow_loader.load(path)
            elif path.endswith('.csv'):
                df = await self.polars_loader.load(path)
            else:
                # Fallback to pandas
                import pandas as pd
                df = pd.read_csv(path)
            
            datasets.append(df)
        
        return datasets

# Example usage
manager = HighPerformanceDataManager()
datasets = await manager.load_multiple_formats([
    'data1.parquet',
    'data2.csv', 
    'data3.csv'
])
```

## Multi-Dataset Workflows

### Dataset Combination Strategies

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class DatasetMetadata:
    """Metadata for dataset tracking."""
    name: str
    source: str
    created_at: str
    sample_count: int
    feature_count: int
    anomaly_rate: float = None

class MultiDatasetProcessor:
    """Process and combine multiple datasets."""
    
    def __init__(self):
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, DatasetMetadata] = {}
    
    async def add_dataset(self, name: str, data: pd.DataFrame, source: str = "unknown"):
        """Add dataset to the collection."""
        
        self.datasets[name] = data
        self.metadata[name] = DatasetMetadata(
            name=name,
            source=source,
            created_at=pd.Timestamp.now().isoformat(),
            sample_count=len(data),
            feature_count=len(data.columns)
        )
        
        print(f"Added dataset '{name}': {len(data)} samples, {len(data.columns)} features")
    
    def combine_datasets(self, names: List[str], strategy: str = "concat") -> pd.DataFrame:
        """Combine multiple datasets with different strategies."""
        
        if strategy == "concat":
            # Vertical concatenation (stack datasets)
            datasets_to_combine = [self.datasets[name] for name in names]
            combined = pd.concat(datasets_to_combine, ignore_index=True)
            
            # Add source column to track origin
            source_labels = []
            for name in names:
                source_labels.extend([name] * len(self.datasets[name]))
            combined['_source_dataset'] = source_labels
            
            return combined
        
        elif strategy == "join":
            # Horizontal join (merge on common columns)
            result = self.datasets[names[0]].copy()
            
            for name in names[1:]:
                # Find common columns for joining
                common_cols = set(result.columns) & set(self.datasets[name].columns)
                if common_cols:
                    result = result.merge(
                        self.datasets[name], 
                        on=list(common_cols), 
                        how='outer',
                        suffixes=('', f'_{name}')
                    )
                else:
                    print(f"Warning: No common columns found for {name}")
            
            return result
        
        elif strategy == "sample":
            # Sample equal amounts from each dataset
            min_samples = min(len(self.datasets[name]) for name in names)
            sampled_datasets = []
            
            for name in names:
                sampled = self.datasets[name].sample(min_samples, random_state=42)
                sampled['_source_dataset'] = name
                sampled_datasets.append(sampled)
            
            return pd.concat(sampled_datasets, ignore_index=True)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def align_features(self, names: List[str]) -> Dict[str, pd.DataFrame]:
        """Align features across datasets."""
        
        # Find common features
        all_features = [set(self.datasets[name].columns) for name in names]
        common_features = set.intersection(*all_features)
        
        print(f"Common features: {len(common_features)}")
        print(f"Features: {sorted(common_features)}")
        
        # Align datasets to common features
        aligned_datasets = {}
        for name in names:
            aligned_datasets[name] = self.datasets[name][list(common_features)]
        
        return aligned_datasets
    
    def compare_datasets(self, names: List[str]) -> pd.DataFrame:
        """Compare statistics across datasets."""
        
        comparison = []
        
        for name in names:
            data = self.datasets[name]
            numeric_data = data.select_dtypes(include=[np.number])
            
            stats = {
                'Dataset': name,
                'Samples': len(data),
                'Features': len(data.columns),
                'Numeric_Features': len(numeric_data.columns),
                'Missing_Values': data.isnull().sum().sum(),
                'Mean_Value': numeric_data.mean().mean() if len(numeric_data.columns) > 0 else 0,
                'Std_Value': numeric_data.std().mean() if len(numeric_data.columns) > 0 else 0
            }
            
            comparison.append(stats)
        
        return pd.DataFrame(comparison)

# Example usage
processor = MultiDatasetProcessor()

# Add multiple datasets
await processor.add_dataset("fraud_data", fraud_df, "fraud_detection_system")
await processor.add_dataset("normal_data", normal_df, "transaction_logs")
await processor.add_dataset("test_data", test_df, "validation_set")

# Compare datasets
comparison = processor.compare_datasets(["fraud_data", "normal_data", "test_data"])
print(comparison)

# Combine datasets
combined = processor.combine_datasets(
    ["fraud_data", "normal_data"], 
    strategy="sample"
)
```

### Cross-Dataset Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

class CrossDatasetValidator:
    """Validate models across different datasets."""
    
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        self.datasets = datasets
        self.results = {}
    
    async def cross_dataset_validation(self, detector_configs: List[Dict]):
        """Test detectors across different datasets."""
        
        results = {}
        
        for config in detector_configs:
            algorithm = config['algorithm']
            results[algorithm] = {}
            
            # Test each dataset
            for dataset_name, data in self.datasets.items():
                print(f"Testing {algorithm} on {dataset_name}...")
                
                # Create and train detector
                detector = await self._create_detector(config)
                
                # Split data for validation
                train_size = int(len(data) * 0.7)
                train_data = data[:train_size]
                test_data = data[train_size:]
                
                # Train on this dataset
                detector.fit(train_data.select_dtypes(include=[np.number]))
                
                # Test on all datasets (including this one)
                dataset_results = {}
                
                for test_name, test_dataset in self.datasets.items():
                    test_numeric = test_dataset.select_dtypes(include=[np.number])
                    predictions = detector.predict(test_numeric)
                    scores = detector.decision_function(test_numeric)
                    
                    anomaly_rate = (predictions == -1).mean()
                    avg_score = scores.mean()
                    
                    dataset_results[test_name] = {
                        'anomaly_rate': anomaly_rate,
                        'avg_score': avg_score,
                        'samples': len(test_dataset)
                    }
                
                results[algorithm][dataset_name] = dataset_results
        
        return results
    
    async def _create_detector(self, config):
        """Create detector from configuration."""
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        
        if config['algorithm'] == 'IsolationForest':
            return IsolationForest(**config.get('parameters', {}))
        elif config['algorithm'] == 'LOF':
            return LocalOutlierFactor(**config.get('parameters', {}))
        else:
            raise ValueError(f"Unknown algorithm: {config['algorithm']}")
    
    def generate_cross_dataset_report(self, results: Dict) -> str:
        """Generate comprehensive cross-dataset report."""
        
        report = ["# Cross-Dataset Validation Report\n"]
        
        for algorithm, dataset_results in results.items():
            report.append(f"## {algorithm}\n")
            
            # Create summary table
            summary_data = []
            for train_dataset, test_results in dataset_results.items():
                for test_dataset, metrics in test_results.items():
                    summary_data.append({
                        'Trained_On': train_dataset,
                        'Tested_On': test_dataset,
                        'Anomaly_Rate': f"{metrics['anomaly_rate']:.3f}",
                        'Avg_Score': f"{metrics['avg_score']:.3f}",
                        'Samples': metrics['samples']
                    })
            
            summary_df = pd.DataFrame(summary_data)
            report.append(summary_df.to_markdown(index=False))
            report.append("\n")
        
        return "\n".join(report)

# Usage example
validator = CrossDatasetValidator({
    'financial': financial_df,
    'healthcare': healthcare_df,
    'manufacturing': manufacturing_df
})

detector_configs = [
    {'algorithm': 'IsolationForest', 'parameters': {'contamination': 0.1}},
    {'algorithm': 'LOF', 'parameters': {'contamination': 0.1, 'n_neighbors': 20}}
]

results = await validator.cross_dataset_validation(detector_configs)
report = validator.generate_cross_dataset_report(results)
print(report)
```

## Data Validation and Quality

### Comprehensive Data Validation

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    dataset_name: str
    total_samples: int
    total_features: int
    missing_values: int
    duplicate_rows: int
    outlier_percentage: float
    data_types: Dict[str, int]
    quality_score: float
    recommendations: List[str]

class DataQualityValidator:
    """Comprehensive data quality validation."""
    
    def __init__(self):
        self.reports = {}
    
    def validate_dataset(self, data: pd.DataFrame, name: str = "Unknown") -> DataQualityReport:
        """Perform comprehensive data quality validation."""
        
        # Basic statistics
        total_samples = len(data)
        total_features = len(data.columns)
        missing_values = data.isnull().sum().sum()
        duplicate_rows = data.duplicated().sum()
        
        # Data types analysis
        data_types = {
            'numeric': len(data.select_dtypes(include=[np.number]).columns),
            'categorical': len(data.select_dtypes(include=['object']).columns),
            'datetime': len(data.select_dtypes(include=['datetime64']).columns)
        }
        
        # Outlier detection
        numeric_data = data.select_dtypes(include=[np.number])
        outlier_percentage = self._detect_outliers(numeric_data)
        
        # Quality score calculation
        quality_score = self._calculate_quality_score(
            total_samples, missing_values, duplicate_rows, outlier_percentage
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing_values, duplicate_rows, outlier_percentage, data_types
        )
        
        report = DataQualityReport(
            dataset_name=name,
            total_samples=total_samples,
            total_features=total_features,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            outlier_percentage=outlier_percentage,
            data_types=data_types,
            quality_score=quality_score,
            recommendations=recommendations
        )
        
        self.reports[name] = report
        return report
    
    def _detect_outliers(self, numeric_data: pd.DataFrame) -> float:
        """Detect outliers using IQR method."""
        if numeric_data.empty:
            return 0.0
        
        outlier_count = 0
        total_values = 0
        
        for column in numeric_data.columns:
            Q1 = numeric_data[column].quantile(0.25)
            Q3 = numeric_data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_data[
                (numeric_data[column] < lower_bound) | 
                (numeric_data[column] > upper_bound)
            ][column]
            
            outlier_count += len(outliers)
            total_values += len(numeric_data[column].dropna())
        
        return (outlier_count / total_values * 100) if total_values > 0 else 0.0
    
    def _calculate_quality_score(self, samples: int, missing: int, duplicates: int, outliers: float) -> float:
        """Calculate overall data quality score (0-100)."""
        
        # Penalize missing values
        missing_penalty = (missing / (samples * 10)) * 20  # Assuming 10 features average
        
        # Penalize duplicates
        duplicate_penalty = (duplicates / samples) * 30
        
        # Penalize excessive outliers
        outlier_penalty = max(0, (outliers - 5) / 100) * 25  # 5% outliers considered normal
        
        # Base score
        score = 100 - missing_penalty - duplicate_penalty - outlier_penalty
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, missing: int, duplicates: int, outliers: float, data_types: Dict) -> List[str]:
        """Generate data improvement recommendations."""
        
        recommendations = []
        
        if missing > 0:
            recommendations.append(f"Handle {missing} missing values using imputation or removal")
        
        if duplicates > 0:
            recommendations.append(f"Remove {duplicates} duplicate rows")
        
        if outliers > 10:
            recommendations.append(f"Review {outliers:.1f}% outliers - consider capping or removal")
        
        if data_types['categorical'] > data_types['numeric']:
            recommendations.append("Consider encoding categorical variables for better performance")
        
        if data_types['numeric'] == 0:
            recommendations.append("No numeric features found - ensure proper data types")
        
        return recommendations
    
    def compare_datasets(self, names: List[str]) -> pd.DataFrame:
        """Compare quality across multiple datasets."""
        
        comparison_data = []
        
        for name in names:
            if name in self.reports:
                report = self.reports[name]
                comparison_data.append({
                    'Dataset': name,
                    'Samples': report.total_samples,
                    'Features': report.total_features,
                    'Missing_Values': report.missing_values,
                    'Duplicates': report.duplicate_rows,
                    'Outliers_%': f"{report.outlier_percentage:.1f}",
                    'Quality_Score': f"{report.quality_score:.1f}"
                })
        
        return pd.DataFrame(comparison_data)
    
    def visualize_quality_comparison(self, names: List[str]):
        """Create quality comparison visualization."""
        
        scores = []
        dataset_names = []
        
        for name in names:
            if name in self.reports:
                scores.append(self.reports[name].quality_score)
                dataset_names.append(name)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(dataset_names, scores, color=['green' if s >= 80 else 'orange' if s >= 60 else 'red' for s in scores])
        
        plt.title('Data Quality Comparison Across Datasets')
        plt.ylabel('Quality Score (0-100)')
        plt.ylim(0, 100)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{score:.1f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Usage example
validator = DataQualityValidator()

# Validate multiple datasets
datasets = {
    'training_data': training_df,
    'validation_data': validation_df,
    'production_data': production_df
}

for name, data in datasets.items():
    report = validator.validate_dataset(data, name)
    print(f"\n{name} Quality Report:")
    print(f"Quality Score: {report.quality_score:.1f}/100")
    print(f"Recommendations: {report.recommendations}")

# Compare all datasets
comparison = validator.compare_datasets(list(datasets.keys()))
print("\nDataset Comparison:")
print(comparison)

# Visualize comparison
validator.visualize_quality_comparison(list(datasets.keys()))
```

## Data Preprocessing Pipeline

### Advanced Preprocessing

```python
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

class AdvancedPreprocessor:
    """Advanced data preprocessing pipeline."""
    
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.dimensionality_reducer = None
        self.preprocessing_steps = []
    
    def build_pipeline(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Build and apply preprocessing pipeline."""
        
        processed_data = data.copy()
        self.preprocessing_steps = []
        
        # Step 1: Handle missing values
        if config.get('handle_missing', True):
            processed_data = self._handle_missing_values(
                processed_data, 
                strategy=config.get('missing_strategy', 'mean')
            )
            self.preprocessing_steps.append('missing_values_handled')
        
        # Step 2: Remove constant features
        if config.get('remove_constant', True):
            processed_data = self._remove_constant_features(processed_data)
            self.preprocessing_steps.append('constant_features_removed')
        
        # Step 3: Handle outliers
        if config.get('handle_outliers', False):
            processed_data = self._handle_outliers(
                processed_data,
                method=config.get('outlier_method', 'iqr')
            )
            self.preprocessing_steps.append('outliers_handled')
        
        # Step 4: Scale features
        if config.get('scale_features', True):
            processed_data = self._scale_features(
                processed_data,
                method=config.get('scaling_method', 'standard')
            )
            self.preprocessing_steps.append('features_scaled')
        
        # Step 5: Feature selection
        if config.get('select_features', False):
            processed_data = self._select_features(
                processed_data,
                n_features=config.get('n_features', 50)
            )
            self.preprocessing_steps.append('features_selected')
        
        # Step 6: Dimensionality reduction
        if config.get('reduce_dimensions', False):
            processed_data = self._reduce_dimensions(
                processed_data,
                n_components=config.get('n_components', 0.95)
            )
            self.preprocessing_steps.append('dimensions_reduced')
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values with different strategies."""
        
        if strategy == 'mean':
            return data.fillna(data.mean())
        elif strategy == 'median':
            return data.fillna(data.median())
        elif strategy == 'mode':
            return data.fillna(data.mode().iloc[0])
        elif strategy == 'drop':
            return data.dropna()
        elif strategy == 'forward_fill':
            return data.fillna(method='ffill')
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")
    
    def _remove_constant_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove features with zero variance."""
        
        selector = VarianceThreshold(threshold=0)
        selected_data = selector.fit_transform(data.select_dtypes(include=[np.number]))
        
        selected_columns = data.select_dtypes(include=[np.number]).columns[selector.get_support()]
        result = pd.DataFrame(selected_data, columns=selected_columns, index=data.index)
        
        # Add back non-numeric columns
        non_numeric = data.select_dtypes(exclude=[np.number])
        if not non_numeric.empty:
            result = pd.concat([result, non_numeric], axis=1)
        
        print(f"Removed {len(data.columns) - len(result.columns)} constant features")
        return result
    
    def _handle_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Handle outliers using various methods."""
        
        numeric_data = data.select_dtypes(include=[np.number])
        processed_numeric = numeric_data.copy()
        
        if method == 'iqr':
            # Interquartile Range method
            for column in numeric_data.columns:
                Q1 = numeric_data[column].quantile(0.25)
                Q3 = numeric_data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                processed_numeric[column] = processed_numeric[column].clip(
                    lower=lower_bound, upper=upper_bound
                )
        
        elif method == 'zscore':
            # Z-score method
            from scipy import stats
            z_scores = np.abs(stats.zscore(numeric_data))
            processed_numeric = processed_numeric[(z_scores < 3).all(axis=1)]
        
        elif method == 'isolation_forest':
            # Use Isolation Forest to detect and remove outliers
            from sklearn.ensemble import IsolationForest
            
            clf = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = clf.fit_predict(numeric_data)
            processed_numeric = processed_numeric[outlier_labels == 1]
        
        # Combine with non-numeric data
        non_numeric = data.select_dtypes(exclude=[np.number])
        if not non_numeric.empty:
            result = pd.concat([processed_numeric, non_numeric.loc[processed_numeric.index]], axis=1)
        else:
            result = processed_numeric
        
        print(f"Outlier handling reduced dataset from {len(data)} to {len(result)} samples")
        return result
    
    def _scale_features(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale numeric features."""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        scaled_data = self.scaler.fit_transform(numeric_data)
        scaled_df = pd.DataFrame(
            scaled_data, 
            columns=numeric_data.columns, 
            index=data.index
        )
        
        # Combine with non-numeric data
        non_numeric = data.select_dtypes(exclude=[np.number])
        if not non_numeric.empty:
            result = pd.concat([scaled_df, non_numeric], axis=1)
        else:
            result = scaled_df
        
        return result
    
    def _select_features(self, data: pd.DataFrame, n_features: int = 50) -> pd.DataFrame:
        """Select top K features based on variance."""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) <= n_features:
            return data
        
        # Use variance-based feature selection for unsupervised learning
        variances = numeric_data.var()
        top_features = variances.nlargest(n_features).index
        
        selected_numeric = numeric_data[top_features]
        
        # Combine with non-numeric data
        non_numeric = data.select_dtypes(exclude=[np.number])
        if not non_numeric.empty:
            result = pd.concat([selected_numeric, non_numeric], axis=1)
        else:
            result = selected_numeric
        
        print(f"Selected {len(top_features)} features out of {len(numeric_data.columns)}")
        return result
    
    def _reduce_dimensions(self, data: pd.DataFrame, n_components: float = 0.95) -> pd.DataFrame:
        """Reduce dimensionality using PCA."""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        self.dimensionality_reducer = PCA(n_components=n_components)
        reduced_data = self.dimensionality_reducer.fit_transform(numeric_data)
        
        # Create column names for PCA components
        component_names = [f'PC{i+1}' for i in range(reduced_data.shape[1])]
        reduced_df = pd.DataFrame(
            reduced_data,
            columns=component_names,
            index=data.index
        )
        
        print(f"Reduced {len(numeric_data.columns)} features to {reduced_data.shape[1]} components")
        print(f"Explained variance ratio: {self.dimensionality_reducer.explained_variance_ratio_.sum():.3f}")
        
        return reduced_df
    
    def get_preprocessing_summary(self) -> Dict:
        """Get summary of preprocessing steps applied."""
        
        summary = {
            'steps_applied': self.preprocessing_steps,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'feature_selector_type': type(self.feature_selector).__name__ if self.feature_selector else None,
            'dimensionality_reducer_type': type(self.dimensionality_reducer).__name__ if self.dimensionality_reducer else None
        }
        
        if self.dimensionality_reducer:
            summary['explained_variance'] = self.dimensionality_reducer.explained_variance_ratio_.sum()
        
        return summary

# Usage example
preprocessor = AdvancedPreprocessor()

# Define preprocessing configuration
preprocessing_config = {
    'handle_missing': True,
    'missing_strategy': 'mean',
    'remove_constant': True,
    'handle_outliers': True,
    'outlier_method': 'iqr',
    'scale_features': True,
    'scaling_method': 'standard',
    'select_features': True,
    'n_features': 20,
    'reduce_dimensions': False,
    'n_components': 0.95
}

# Apply preprocessing to multiple datasets
processed_datasets = {}
for name, data in datasets.items():
    processed_data = preprocessor.build_pipeline(data, preprocessing_config)
    processed_datasets[name] = processed_data
    
    print(f"\n{name} preprocessing summary:")
    summary = preprocessor.get_preprocessing_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
```

#### High-Performance CSV (Polars)
```python
from pynomaly.infrastructure.data_loaders import PolarsLoader

# Fast CSV loading with lazy evaluation
loader = PolarsLoader(lazy=True, streaming=True)
dataset = await loader.load_async("data/large_transactions.csv")

# Performance comparison
performance = await loader.compare_performance("data/benchmark.csv")
print(f"Polars speedup: {performance['speedup_factor']:.2f}x")
print(f"Memory reduction: {performance['memory_reduction_percent']:.1f}%")
```

### Parquet Processing

#### Standard Parquet (Pandas)
```python
from pynomaly.infrastructure.data_loaders import ParquetLoader

loader = ParquetLoader()
dataset = await loader.load_async(
    "data/timeseries.parquet",
    columns=["timestamp", "value", "sensor_id"],  # Select specific columns
    filters=[("sensor_id", "in", ["sensor_001", "sensor_002"])],
    use_pandas_metadata=True
)
```

#### Optimized Parquet (Arrow)
```python
from pynomaly.infrastructure.data_loaders import ArrowLoader

loader = ArrowLoader(use_threads=True, streaming=True)
dataset = await loader.load_async("data/large_dataset.parquet")

# Apply transformations during loading
dataset = await loader.load_with_transforms(
    "data/raw_data.parquet",
    transforms={
        "normalize": ["feature1", "feature2"],
        "zscore": ["feature3", "feature4"],
        "log_transform": ["feature5"]
    }
)
```

### Big Data Processing (Spark)

```python
from pynomaly.infrastructure.data_loaders import SparkLoader

# Configure Spark for your cluster
spark_config = {
    "spark.executor.memory": "4g",
    "spark.executor.cores": "2",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
}

loader = SparkLoader(
    app_name="AnomalyDetection",
    master="local[*]",  # or "spark://cluster:7077"
    config=spark_config
)

# Load distributed dataset
dataset = await loader.load_async("hdfs://cluster/data/massive_dataset.parquet")

# Distributed anomaly detection
detector = SparkAnomalyDetector("IsolationForest")
results = await detector.detect_distributed(dataset, contamination=0.1)
```

## High-Performance Processing

### Memory-Efficient Loading Strategies

#### Chunked Processing for Large Files
```python
from pynomaly.infrastructure.data_loaders import CSVLoader
from pynomaly.application.use_cases import DetectAnomalies

async def process_large_file_in_chunks():
    loader = CSVLoader()
    detector = DetectAnomalies()
    
    all_results = []
    
    # Process file in manageable chunks
    async for chunk_dataset in loader.load_chunked(
        "data/massive_file.csv", 
        chunk_size=50000
    ):
        # Detect anomalies in chunk
        results = await detector.execute(
            dataset=chunk_dataset,
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.05}
        )
        all_results.append(results)
    
    # Combine results
    combined_results = await combine_detection_results(all_results)
    return combined_results
```

### Performance Optimization Patterns

#### Smart Loader Selection
```python
from pynomaly.infrastructure.data_loaders import auto_select_loader

async def optimized_loading(file_path: str):
    # Automatically choose best loader based on file characteristics
    loader_info = await auto_select_loader(file_path)
    
    print(f"Selected loader: {loader_info['loader_type']}")
    print(f"Reason: {loader_info['reason']}")
    print(f"Expected speedup: {loader_info['expected_speedup']:.2f}x")
    
    # Load with optimal configuration
    dataset = await loader_info['loader'].load_async(file_path)
    return dataset
```

## Data Validation and Quality

### Schema Validation
```python
from pynomaly.domain.entities import Dataset
import pandas as pd

async def validate_dataset_schema(dataset: Dataset):
    """Comprehensive dataset validation."""
    
    validation_results = {
        'schema_valid': True,
        'issues': [],
        'recommendations': []
    }
    
    # Check for required columns
    required_columns = ['timestamp', 'value']
    missing_columns = [col for col in required_columns if col not in dataset.data.columns]
    if missing_columns:
        validation_results['issues'].append(f"Missing columns: {missing_columns}")
        validation_results['schema_valid'] = False
    
    # Check data types
    if 'timestamp' in dataset.data.columns:
        if not pd.api.types.is_datetime64_any_dtype(dataset.data['timestamp']):
            validation_results['issues'].append("Timestamp column is not datetime type")
            validation_results['recommendations'].append(
                "Convert timestamp: pd.to_datetime(df['timestamp'])"
            )
    
    return validation_results
```

## Streaming and Large Datasets

### Real-Time Data Processing
```python
import asyncio
from pynomaly.infrastructure.data_loaders import PolarsLoader
from pynomaly.application.use_cases import DetectAnomalies

class RealTimeAnomalyProcessor:
    def __init__(self):
        self.loader = PolarsLoader(streaming=True)
        self.detector = DetectAnomalies()
        self.buffer = []
        self.buffer_size = 1000
    
    async def process_streaming_data(self, data_stream):
        """Process continuous data stream for anomalies."""
        
        async for data_point in data_stream:
            self.buffer.append(data_point)
            
            # Process when buffer is full
            if len(self.buffer) >= self.buffer_size:
                await self.process_buffer()
                self.buffer = []
    
    async def process_buffer(self):
        """Process accumulated data points."""
        if not self.buffer:
            return
        
        # Convert buffer to dataset
        df = pd.DataFrame(self.buffer)
        dataset = Dataset(name="streaming_batch", data=df)
        
        # Detect anomalies
        results = await self.detector.execute(
            dataset=dataset,
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.05}
        )
        
        # Handle results (alert, store, etc.)
        await self.handle_anomaly_results(results)
```

## Performance Optimization

### Memory Management
```python
import psutil
import gc

class MemoryOptimizedLoader:
    def __init__(self, memory_limit_gb=4):
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
    
    async def load_with_memory_management(self, file_path: str):
        """Load data with active memory monitoring."""
        
        # Check available memory
        available_memory = psutil.virtual_memory().available
        
        if available_memory < self.memory_limit_bytes:
            # Use streaming approach
            return await self.load_streaming(file_path)
        else:
            # Standard loading
            return await self.load_standard(file_path)
```

## Production Patterns

### Error Handling and Resilience
```python
from pynomaly.infrastructure.resilience import ml_resilient

class ProductionDataLoader:
    def __init__(self):
        self.retry_config = {
            'max_attempts': 3,
            'base_delay': 1.0,
            'max_delay': 10.0
        }
    
    @ml_resilient(timeout_seconds=300, max_attempts=2)
    async def load_with_resilience(self, file_path: str):
        """Production-ready data loading with comprehensive error handling."""
        
        try:
            # Validate file exists and is readable
            await self.validate_file_access(file_path)
            
            # Choose optimal loader
            loader = await self.select_optimal_loader(file_path)
            
            # Load with monitoring
            dataset = await self.load_with_monitoring(loader, file_path)
            
            # Validate loaded data
            await self.validate_loaded_data(dataset)
            
            return dataset
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except PermissionError:
            raise PermissionError(f"No permission to read file: {file_path}")
        except Exception as e:
            # Log error with context
            logger.error(f"Data loading failed for {file_path}: {str(e)}")
            raise
```

## Best Practices Summary

### 1. Loader Selection Guidelines
- **Small files (<100MB)**: Use Pandas loader for simplicity
- **Medium files (100MB-1GB)**: Use Polars loader for performance
- **Large files (1GB-10GB)**: Use Arrow loader with streaming
- **Massive files (>10GB)**: Use Spark loader for distributed processing

### 2. Memory Management
- Always monitor memory usage during processing
- Use streaming for files larger than available RAM
- Implement chunked processing for very large datasets
- Clear intermediate variables and force garbage collection

### 3. Performance Optimization
- Cache frequently accessed datasets
- Use lazy evaluation when possible
- Parallelize independent operations
- Choose appropriate data types to minimize memory usage

### 4. Production Readiness
- Implement comprehensive error handling
- Add monitoring and observability
- Use circuit breakers for external dependencies
- Validate data quality at load time

### 5. Data Quality
- Always validate schema before processing
- Check for missing values and outliers
- Monitor data drift over time
- Implement data lineage tracking

This guide provides the foundation for robust, scalable data processing in production anomaly detection systems. Combine these patterns with domain-specific requirements for optimal results.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
