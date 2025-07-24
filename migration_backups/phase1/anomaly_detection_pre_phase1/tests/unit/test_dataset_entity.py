"""Unit tests for Dataset entity."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from anomaly_detection.domain.entities.dataset import Dataset


class TestDatasetEntity:
    """Test suite for Dataset entity."""
    
    def test_dataset_creation_numpy(self):
        """Test dataset creation from numpy array."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        dataset = Dataset(data=data)
        
        assert np.array_equal(dataset.data, data)
        assert dataset.shape == (3, 3)
        assert dataset.n_samples == 3
        assert dataset.n_features == 3
        assert dataset.feature_names is None
        assert dataset.metadata == {}
    
    def test_dataset_creation_pandas(self):
        """Test dataset creation from pandas DataFrame."""
        df = pd.DataFrame({
            'temperature': [20.5, 21.0, 19.8],
            'pressure': [1013, 1015, 1012],
            'humidity': [45, 50, 48]
        })
        
        dataset = Dataset(data=df)
        
        assert isinstance(dataset.data, np.ndarray)
        assert dataset.shape == (3, 3)
        assert dataset.feature_names == ['temperature', 'pressure', 'humidity']
    
    def test_dataset_with_feature_names(self):
        """Test dataset with custom feature names."""
        data = np.random.randn(10, 4)
        feature_names = ['f1', 'f2', 'f3', 'f4']
        
        dataset = Dataset(data=data, feature_names=feature_names)
        
        assert dataset.feature_names == feature_names
        assert len(dataset.feature_names) == dataset.n_features
    
    def test_dataset_with_metadata(self):
        """Test dataset with metadata."""
        data = np.random.randn(100, 5)
        metadata = {
            'source': 'sensor_network',
            'timestamp': datetime.now(),
            'sampling_rate': 100,
            'units': ['celsius', 'bar', 'percent', 'm/s', 'hz']
        }
        
        dataset = Dataset(data=data, metadata=metadata)
        
        assert dataset.metadata == metadata
        assert dataset.metadata['source'] == 'sensor_network'
        assert dataset.metadata['sampling_rate'] == 100
    
    def test_dataset_validation_empty(self):
        """Test validation of empty dataset."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            Dataset(data=np.array([]))
        
        with pytest.raises(ValueError, match="Data cannot be empty"):
            Dataset(data=np.array([[]]))
    
    def test_dataset_validation_1d(self):
        """Test handling of 1D data."""
        data_1d = np.array([1, 2, 3, 4, 5])
        
        # Should reshape to 2D
        dataset = Dataset(data=data_1d)
        
        assert dataset.shape == (5, 1)
        assert dataset.n_samples == 5
        assert dataset.n_features == 1
    
    def test_dataset_validation_feature_names_mismatch(self):
        """Test validation of feature names length."""
        data = np.random.randn(10, 3)
        wrong_names = ['f1', 'f2']  # Only 2 names for 3 features
        
        with pytest.raises(ValueError, match="Number of feature names"):
            Dataset(data=data, feature_names=wrong_names)
    
    def test_dataset_validation_non_numeric(self):
        """Test handling of non-numeric data."""
        # Mixed types DataFrame
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'text': ['a', 'b', 'c'],
            'bool': [True, False, True]
        })
        
        # Should extract only numeric columns
        dataset = Dataset(data=df)
        
        assert dataset.shape == (3, 1)  # Only numeric column
        assert dataset.feature_names == ['numeric']
    
    def test_dataset_validation_all_non_numeric(self):
        """Test error when no numeric features."""
        df = pd.DataFrame({
            'text1': ['a', 'b', 'c'],
            'text2': ['x', 'y', 'z']
        })
        
        with pytest.raises(ValueError, match="No numeric features found"):
            Dataset(data=df)
    
    def test_dataset_validate_method(self):
        """Test explicit validation method."""
        data = np.array([[1, 2], [3, np.nan], [5, 6]])
        dataset = Dataset(data=data)
        
        # Validation should detect NaN
        issues = dataset.validate()
        
        assert 'missing_values' in issues
        assert issues['missing_values']['count'] == 1
        assert issues['missing_values']['indices'] == [(1, 1)]
    
    def test_dataset_validate_infinite_values(self):
        """Test validation of infinite values."""
        data = np.array([[1, 2], [3, np.inf], [5, -np.inf]])
        dataset = Dataset(data=data)
        
        issues = dataset.validate()
        
        assert 'infinite_values' in issues
        assert issues['infinite_values']['count'] == 2
    
    def test_dataset_normalize(self):
        """Test data normalization."""
        data = np.array([[1, 100], [2, 200], [3, 300]])
        dataset = Dataset(data=data)
        
        # Standard normalization
        normalized = dataset.normalize(method='standard')
        
        assert normalized.shape == dataset.shape
        assert np.abs(np.mean(normalized.data, axis=0)).max() < 1e-10
        assert np.abs(np.std(normalized.data, axis=0) - 1).max() < 1e-10
    
    def test_dataset_normalize_minmax(self):
        """Test min-max normalization."""
        data = np.array([[1, 10], [2, 20], [3, 30]])
        dataset = Dataset(data=data)
        
        normalized = dataset.normalize(method='minmax')
        
        assert normalized.data.min() >= 0
        assert normalized.data.max() <= 1
        assert np.allclose(normalized.data.min(axis=0), 0)
        assert np.allclose(normalized.data.max(axis=0), 1)
    
    def test_dataset_normalize_robust(self):
        """Test robust normalization."""
        data = np.array([[1, 10], [2, 20], [3, 30], [100, 1000]])  # Last row is outlier
        dataset = Dataset(data=data)
        
        normalized = dataset.normalize(method='robust')
        
        # Check that outliers don't dominate
        assert normalized.shape == dataset.shape
        # Robust scaling should handle outliers better
        assert np.abs(normalized.data[-1]).max() < 10  # Outlier should be scaled down
    
    def test_dataset_split(self):
        """Test train-test split."""
        data = np.random.randn(100, 5)
        dataset = Dataset(data=data)
        
        train, test = dataset.split(test_ratio=0.3, random_state=42)
        
        assert isinstance(train, Dataset)
        assert isinstance(test, Dataset)
        assert train.n_samples == 70
        assert test.n_samples == 30
        assert train.n_features == test.n_features == 5
        
        # Check no overlap
        train_set = set(map(tuple, train.data))
        test_set = set(map(tuple, test.data))
        assert len(train_set & test_set) == 0
    
    def test_dataset_split_stratified(self):
        """Test stratified split with labels."""
        data = np.random.randn(100, 3)
        labels = np.array([0] * 80 + [1] * 20)  # Imbalanced
        dataset = Dataset(data=data, labels=labels)
        
        train, test = dataset.split(test_ratio=0.2, stratify=True, random_state=42)
        
        # Check label distribution is preserved
        train_ratio = np.mean(train.labels == 1)
        test_ratio = np.mean(test.labels == 1)
        
        assert np.abs(train_ratio - 0.2) < 0.05
        assert np.abs(test_ratio - 0.2) < 0.05
    
    def test_dataset_get_feature(self):
        """Test getting specific feature."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = Dataset(data=data, feature_names=['a', 'b', 'c'])
        
        # Get by index
        feature_0 = dataset.get_feature(0)
        assert np.array_equal(feature_0, [1, 4, 7])
        
        # Get by name
        feature_b = dataset.get_feature('b')
        assert np.array_equal(feature_b, [2, 5, 8])
    
    def test_dataset_get_sample(self):
        """Test getting specific sample."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = Dataset(data=data)
        
        sample_1 = dataset.get_sample(1)
        assert np.array_equal(sample_1, [4, 5, 6])
        
        # Test negative indexing
        last_sample = dataset.get_sample(-1)
        assert np.array_equal(last_sample, [7, 8, 9])
    
    def test_dataset_add_feature(self):
        """Test adding new feature."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        dataset = Dataset(data=data, feature_names=['f1', 'f2'])
        
        new_feature = np.array([10, 20, 30])
        updated = dataset.add_feature(new_feature, name='f3')
        
        assert updated.shape == (3, 3)
        assert updated.feature_names == ['f1', 'f2', 'f3']
        assert np.array_equal(updated.get_feature('f3'), new_feature)
    
    def test_dataset_remove_feature(self):
        """Test removing feature."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = Dataset(data=data, feature_names=['a', 'b', 'c'])
        
        # Remove by index
        reduced = dataset.remove_feature(1)
        assert reduced.shape == (3, 2)
        assert reduced.feature_names == ['a', 'c']
        
        # Remove by name
        reduced2 = dataset.remove_feature('b')
        assert reduced2.shape == (3, 2)
        assert reduced2.feature_names == ['a', 'c']
    
    def test_dataset_subset(self):
        """Test creating subset of samples."""
        data = np.random.randn(100, 5)
        dataset = Dataset(data=data)
        
        # Get first 10 samples
        subset = dataset.subset(indices=range(10))
        assert subset.shape == (10, 5)
        
        # Get specific samples
        subset2 = dataset.subset(indices=[0, 5, 10, 15, 20])
        assert subset2.shape == (5, 5)
    
    def test_dataset_describe(self):
        """Test statistical description."""
        data = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
        dataset = Dataset(data=data, feature_names=['x', 'y'])
        
        stats = dataset.describe()
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'percentiles' in stats
        
        assert stats['mean']['x'] == 3.0
        assert stats['mean']['y'] == 30.0
        assert stats['min']['x'] == 1.0
        assert stats['max']['y'] == 50.0
    
    def test_dataset_correlation_matrix(self):
        """Test correlation matrix calculation."""
        # Create correlated data
        size = 100
        x = np.random.randn(size)
        y = 2 * x + np.random.randn(size) * 0.1  # Strong positive correlation
        z = -x + np.random.randn(size) * 0.1     # Strong negative correlation
        
        data = np.column_stack([x, y, z])
        dataset = Dataset(data=data, feature_names=['x', 'y', 'z'])
        
        corr_matrix = dataset.correlation_matrix()
        
        assert corr_matrix.shape == (3, 3)
        assert np.abs(corr_matrix[0, 1] - 1) < 0.2  # Strong positive
        assert np.abs(corr_matrix[0, 2] + 1) < 0.2  # Strong negative
        assert np.allclose(np.diag(corr_matrix), 1)  # Diagonal should be 1
    
    def test_dataset_to_pandas(self):
        """Test conversion to pandas DataFrame."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        dataset = Dataset(data=data, feature_names=['a', 'b', 'c'])
        
        df = dataset.to_pandas()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ['a', 'b', 'c']
        assert df.iloc[0, 0] == 1
        assert df.iloc[1, 2] == 6
    
    def test_dataset_save_load(self, tmp_path):
        """Test saving and loading dataset."""
        data = np.random.randn(50, 4)
        metadata = {'source': 'test', 'date': '2024-01-23'}
        dataset = Dataset(
            data=data,
            feature_names=['f1', 'f2', 'f3', 'f4'],
            metadata=metadata
        )
        
        # Save
        filepath = tmp_path / 'dataset.pkl'
        dataset.save(str(filepath))
        
        # Load
        loaded = Dataset.load(str(filepath))
        
        assert np.array_equal(loaded.data, data)
        assert loaded.feature_names == dataset.feature_names
        assert loaded.metadata == metadata
    
    def test_dataset_equality(self):
        """Test dataset equality comparison."""
        data = np.array([[1, 2], [3, 4]])
        
        dataset1 = Dataset(data=data, feature_names=['a', 'b'])
        dataset2 = Dataset(data=data, feature_names=['a', 'b'])
        dataset3 = Dataset(data=data * 2, feature_names=['a', 'b'])
        
        assert dataset1 == dataset2
        assert dataset1 != dataset3
        assert dataset1 != "not a dataset"
    
    def test_dataset_repr(self):
        """Test string representation."""
        data = np.random.randn(100, 5)
        dataset = Dataset(data=data)
        
        repr_str = repr(dataset)
        assert 'Dataset' in repr_str
        assert '100' in repr_str
        assert '5' in repr_str
    
    def test_dataset_copy(self):
        """Test dataset copying."""
        data = np.array([[1, 2], [3, 4]])
        dataset = Dataset(data=data, feature_names=['a', 'b'])
        
        # Shallow copy should share data
        shallow = dataset.copy(deep=False)
        assert shallow is not dataset
        assert shallow.data is dataset.data
        
        # Deep copy should not share data
        deep = dataset.copy(deep=True)
        assert deep is not dataset
        assert deep.data is not dataset.data
        assert np.array_equal(deep.data, dataset.data)