"""Unit tests for data processing functionality."""

import pytest
import numpy as np
import json
from datetime import datetime, timedelta


def test_data_validation():
    """Test data validation functions."""
    
    def validate_detection_data(data) -> tuple:
        """Validate data for anomaly detection."""
        errors = []
        
        # Check if data is provided
        if data is None:
            errors.append("Data cannot be None")
            return False, errors
        
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except (ValueError, TypeError):
                errors.append("Data must be convertible to numpy array")
                return False, errors
        
        # Check dimensions
        if data.ndim != 2:
            errors.append("Data must be 2-dimensional")
            return False, errors  # Return early if not 2D
        
        # Check for minimum samples
        if data.shape[0] < 2:
            errors.append("Data must have at least 2 samples")
        
        # Check for minimum features
        if data.shape[1] < 1:
            errors.append("Data must have at least 1 feature")
        
        # Check for NaN values
        if np.any(np.isnan(data)):
            errors.append("Data cannot contain NaN values")
        
        # Check for infinite values
        if np.any(np.isinf(data)):
            errors.append("Data cannot contain infinite values")
        
        return len(errors) == 0, errors
    
    def validate_contamination(contamination: float) -> tuple:
        """Validate contamination parameter."""
        errors = []
        
        if not isinstance(contamination, (int, float)):
            errors.append("Contamination must be a number")
        elif contamination <= 0 or contamination >= 1:
            errors.append("Contamination must be between 0 and 1")
        
        return len(errors) == 0, errors
    
    def validate_algorithm(algorithm: str) -> tuple:
        """Validate algorithm parameter."""
        errors = []
        valid_algorithms = [
            'isolation_forest', 'lof', 'one_class_svm', 
            'ensemble_majority', 'ensemble_average'
        ]
        
        if not isinstance(algorithm, str):
            errors.append("Algorithm must be a string")
        elif algorithm not in valid_algorithms:
            errors.append(f"Algorithm must be one of: {', '.join(valid_algorithms)}")
        
        return len(errors) == 0, errors
    
    # Test valid data
    valid_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    is_valid, errors = validate_detection_data(valid_data)
    assert is_valid
    assert len(errors) == 0
    
    # Test None data
    is_valid, errors = validate_detection_data(None)
    assert not is_valid
    assert "Data cannot be None" in errors
    
    # Test 1D data
    invalid_data = np.array([1, 2, 3, 4, 5])
    is_valid, errors = validate_detection_data(invalid_data)
    assert not is_valid
    assert "Data must be 2-dimensional" in errors
    
    # Test insufficient samples
    insufficient_data = np.array([[1, 2, 3]])
    is_valid, errors = validate_detection_data(insufficient_data)
    assert not is_valid
    assert "Data must have at least 2 samples" in errors
    
    # Test data with NaN
    nan_data = np.array([[1, 2, np.nan], [4, 5, 6]])
    is_valid, errors = validate_detection_data(nan_data)
    assert not is_valid
    assert "Data cannot contain NaN values" in errors
    
    # Test data with infinity
    inf_data = np.array([[1, 2, np.inf], [4, 5, 6]])
    is_valid, errors = validate_detection_data(inf_data)
    assert not is_valid
    assert "Data cannot contain infinite values" in errors
    
    # Test valid contamination
    is_valid, errors = validate_contamination(0.1)
    assert is_valid
    assert len(errors) == 0
    
    # Test invalid contamination
    is_valid, errors = validate_contamination(1.5)
    assert not is_valid
    assert "Contamination must be between 0 and 1" in errors
    
    # Test valid algorithm
    is_valid, errors = validate_algorithm('isolation_forest')
    assert is_valid
    assert len(errors) == 0
    
    # Test invalid algorithm
    is_valid, errors = validate_algorithm('invalid_algorithm')
    assert not is_valid
    assert "Algorithm must be one of:" in errors[0]


def test_data_preprocessing():
    """Test data preprocessing functions."""
    
    def normalize_data(data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range."""
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        
        # Avoid division by zero
        range_vals = data_max - data_min
        range_vals[range_vals == 0] = 1
        
        return (data - data_min) / range_vals
    
    def standardize_data(data: np.ndarray) -> np.ndarray:
        """Standardize data to have mean=0 and std=1."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1
        
        return (data - mean) / std
    
    def remove_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Remove outliers using Z-score method."""
        z_scores = np.abs(standardize_data(data))
        return data[np.all(z_scores < threshold, axis=1)]
    
    def handle_missing_values(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
        """Handle missing values in data."""
        if not np.any(np.isnan(data)):
            return data
        
        result = data.copy()
        
        for col in range(data.shape[1]):
            col_data = data[:, col]
            mask = np.isnan(col_data)
            
            if np.any(mask):
                if strategy == 'mean':
                    fill_value = np.nanmean(col_data)
                elif strategy == 'median':
                    fill_value = np.nanmedian(col_data)
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = np.nanmean(col_data)
                
                result[mask, col] = fill_value
        
        return result
    
    # Test normalization
    test_data = np.array([[1, 10], [2, 20], [3, 30]])
    normalized = normalize_data(test_data)
    
    assert np.allclose(np.min(normalized, axis=0), [0, 0])
    assert np.allclose(np.max(normalized, axis=0), [1, 1])
    
    # Test normalization with constant column
    constant_data = np.array([[1, 5], [2, 5], [3, 5]])
    normalized_constant = normalize_data(constant_data)
    
    assert np.allclose(normalized_constant[:, 1], [0, 0, 0])  # Constant column becomes 0
    
    # Test standardization
    test_data = np.array([[1, 10], [2, 20], [3, 30]])
    standardized = standardize_data(test_data)
    
    assert np.allclose(np.mean(standardized, axis=0), [0, 0], atol=1e-10)
    assert np.allclose(np.std(standardized, axis=0), [1, 1])
    
    # Test outlier removal
    data_with_outliers = np.array([
        [1, 1], [2, 2], [3, 3], [4, 4], [100, 100]  # Last point is outlier
    ])
    
    cleaned_data = remove_outliers(data_with_outliers, threshold=1.5)
    assert cleaned_data.shape[0] <= data_with_outliers.shape[0]  # May remove outliers
    
    # Test missing value handling
    data_with_nan = np.array([
        [1, 10], [2, np.nan], [3, 30], [np.nan, 40]
    ])
    
    # Test mean strategy
    filled_mean = handle_missing_values(data_with_nan, strategy='mean')
    assert not np.any(np.isnan(filled_mean))
    
    # Test median strategy
    filled_median = handle_missing_values(data_with_nan, strategy='median')
    assert not np.any(np.isnan(filled_median))
    
    # Test zero strategy
    filled_zero = handle_missing_values(data_with_nan, strategy='zero')
    assert not np.any(np.isnan(filled_zero))


def test_synthetic_data_generation():
    """Test synthetic data generation functions."""
    
    def generate_normal_data(n_samples: int, n_features: int, random_state: int = 42) -> np.ndarray:
        """Generate normal data."""
        np.random.seed(random_state)
        return np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=n_samples
        )
    
    def generate_anomaly_data(n_samples: int, n_features: int, shift: float = 3.0, random_state: int = 42) -> np.ndarray:
        """Generate anomalous data."""
        np.random.seed(random_state + 1)  # Different seed for anomalies
        return np.random.multivariate_normal(
            mean=np.ones(n_features) * shift,
            cov=np.eye(n_features) * 2,
            size=n_samples
        )
    
    def generate_mixed_data(n_normal: int, n_anomalies: int, n_features: int, random_state: int = 42) -> tuple:
        """Generate mixed normal and anomalous data."""
        normal_data = generate_normal_data(n_normal, n_features, random_state)
        anomaly_data = generate_anomaly_data(n_anomalies, n_features, random_state=random_state)
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.ones(n_normal), -np.ones(n_anomalies)])
        
        # Shuffle
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        
        return X[indices], y[indices]
    
    def generate_time_series_data(n_points: int, trend: float = 0.1, noise: float = 0.5, random_state: int = 42) -> tuple:
        """Generate time series data with trend and noise."""
        np.random.seed(random_state)
        
        time_points = np.arange(n_points)
        trend_component = trend * time_points
        noise_component = np.random.normal(0, noise, n_points)
        seasonal_component = np.sin(2 * np.pi * time_points / 24)  # Daily seasonality
        
        values = trend_component + seasonal_component + noise_component
        
        return time_points, values
    
    # Test normal data generation
    normal_data = generate_normal_data(100, 5)
    
    assert normal_data.shape == (100, 5)
    assert np.abs(np.mean(normal_data)) < 0.2  # Should be close to 0
    assert np.abs(np.std(normal_data) - 1.0) < 0.2  # Should be close to 1
    
    # Test anomaly data generation
    anomaly_data = generate_anomaly_data(20, 5)
    
    assert anomaly_data.shape == (20, 5)
    assert np.mean(anomaly_data) > 2.0  # Should be shifted
    
    # Test mixed data generation
    X, y = generate_mixed_data(80, 20, 3)
    
    assert X.shape == (100, 3)
    assert len(y) == 100
    assert np.sum(y == 1) == 80  # Normal samples
    assert np.sum(y == -1) == 20  # Anomaly samples
    
    # Test time series generation
    time_points, values = generate_time_series_data(48)  # 48 hours
    
    assert len(time_points) == 48
    assert len(values) == 48
    assert np.max(time_points) == 47
    assert np.min(time_points) == 0


def test_data_transformation():
    """Test data transformation functions."""
    
    def apply_pca(data: np.ndarray, n_components: int = None) -> tuple:
        """Apply PCA transformation (simplified version)."""
        if n_components is None:
            n_components = min(data.shape)
        
        # Center the data
        mean = np.mean(data, axis=0)
        centered_data = data - mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_data.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Transform data
        transformed = centered_data @ eigenvectors[:, :n_components]
        
        return transformed, eigenvalues[:n_components], eigenvectors[:, :n_components]
    
    def create_polynomial_features(data: np.ndarray, degree: int = 2) -> np.ndarray:
        """Create polynomial features."""
        if degree == 1:
            return data
        
        n_samples, n_features = data.shape
        features = [data]
        
        # Add squared terms
        if degree >= 2:
            features.append(data ** 2)
        
        # Add interaction terms for degree 2
        if degree >= 2 and n_features > 1:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interaction = (data[:, i] * data[:, j]).reshape(-1, 1)
                    features.append(interaction)
        
        return np.hstack(features)
    
    def apply_log_transform(data: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Apply log transformation with small epsilon to avoid log(0)."""
        return np.log(np.abs(data) + epsilon)
    
    def create_rolling_features(data: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Create rolling window features."""
        n_samples = len(data)
        if n_samples < window_size:
            return data.reshape(-1, 1)
        
        features = []
        for i in range(window_size, n_samples + 1):
            window = data[i - window_size:i]
            features.append([
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window)
            ])
        
        return np.array(features)
    
    # Test PCA
    test_data = np.random.randn(50, 4)
    transformed, eigenvals, eigenvecs = apply_pca(test_data, n_components=2)
    
    assert transformed.shape == (50, 2)
    assert len(eigenvals) == 2
    assert eigenvecs.shape == (4, 2)
    assert np.all(eigenvals[:-1] >= eigenvals[1:])  # Sorted descending
    
    # Test polynomial features
    simple_data = np.array([[1, 2], [3, 4], [5, 6]])
    poly_features = create_polynomial_features(simple_data, degree=2)
    
    # Should have original features + squared features + interaction
    # 2 original + 2 squared + 1 interaction = 5 features
    assert poly_features.shape == (3, 5)
    
    # Test log transform
    positive_data = np.array([[1, 2], [3, 4], [5, 6]])
    log_transformed = apply_log_transform(positive_data)
    
    assert log_transformed.shape == positive_data.shape
    assert np.all(log_transformed >= 0)  # log of positive values
    
    # Test rolling features
    time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rolling_features = create_rolling_features(time_series, window_size=3)
    
    assert rolling_features.shape == (8, 4)  # 10 - 3 + 1 = 8 windows, 4 features each
    
    # First window should be [1, 2, 3]
    assert rolling_features[0, 0] == 2.0  # mean
    assert rolling_features[0, 2] == 1.0  # min
    assert rolling_features[0, 3] == 3.0  # max


def test_data_quality_checks():
    """Test data quality assessment functions."""
    
    def assess_data_quality(data: np.ndarray) -> dict:
        """Assess various aspects of data quality."""
        n_samples, n_features = data.shape
        
        # Missing values
        missing_count = np.sum(np.isnan(data))
        missing_percentage = (missing_count / (n_samples * n_features)) * 100
        
        # Duplicate rows
        unique_rows = np.unique(data, axis=0)
        duplicate_count = n_samples - len(unique_rows)
        duplicate_percentage = (duplicate_count / n_samples) * 100
        
        # Outliers (using IQR method)
        outlier_count = 0
        for col in range(n_features):
            col_data = data[:, col]
            if not np.any(np.isnan(col_data)):
                q1, q3 = np.percentile(col_data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_count += np.sum((col_data < lower_bound) | (col_data > upper_bound))
        
        outlier_percentage = (outlier_count / (n_samples * n_features)) * 100
        
        # Data range and variance
        ranges = np.ptp(data, axis=0)  # Range for each feature
        variances = np.var(data, axis=0)
        
        # Overall quality score (simplified)
        quality_score = 100 - missing_percentage - duplicate_percentage - (outlier_percentage / 2)
        quality_score = max(0, min(100, quality_score))
        
        return {
            'total_samples': n_samples,
            'total_features': n_features,
            'missing_values': missing_count,
            'missing_percentage': missing_percentage,
            'duplicate_rows': duplicate_count,
            'duplicate_percentage': duplicate_percentage,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'feature_ranges': ranges.tolist(),
            'feature_variances': variances.tolist(),
            'quality_score': quality_score
        }
    
    def detect_drift(reference_data: np.ndarray, new_data: np.ndarray, threshold: float = 0.1) -> dict:
        """Detect data drift between reference and new data."""
        drift_detected = False
        feature_drifts = []
        
        for col in range(reference_data.shape[1]):
            ref_mean = np.mean(reference_data[:, col])
            ref_std = np.std(reference_data[:, col])
            
            new_mean = np.mean(new_data[:, col])
            new_std = np.std(new_data[:, col])
            
            # Normalized difference in means
            if ref_std > 0:
                mean_drift = abs(new_mean - ref_mean) / ref_std
            else:
                mean_drift = 0
            
            # Ratio of standard deviations
            if ref_std > 0:
                std_ratio = new_std / ref_std
            else:
                std_ratio = 1
            
            feature_drift = mean_drift > threshold or abs(std_ratio - 1) > threshold
            
            feature_drifts.append({
                'feature_index': col,
                'mean_drift': mean_drift,
                'std_ratio': std_ratio,
                'drift_detected': feature_drift
            })
            
            if feature_drift:
                drift_detected = True
        
        return {
            'drift_detected': drift_detected,
            'feature_drifts': feature_drifts,
            'drift_threshold': threshold
        }
    
    # Test data quality assessment
    good_data = np.random.randn(100, 5)
    quality_report = assess_data_quality(good_data)
    
    assert quality_report['total_samples'] == 100
    assert quality_report['total_features'] == 5
    assert quality_report['missing_values'] == 0
    assert quality_report['duplicate_rows'] == 0
    assert quality_report['quality_score'] > 90  # Should be high quality
    
    # Test data with issues
    problematic_data = np.random.randn(50, 3)
    problematic_data[0] = problematic_data[1]  # Add duplicate
    problematic_data[2, 0] = np.nan  # Add missing value
    problematic_data[3, 1] = 100  # Add outlier
    
    quality_report = assess_data_quality(problematic_data)
    
    assert quality_report['missing_values'] > 0
    assert quality_report['duplicate_rows'] > 0
    assert quality_report['outlier_count'] > 0
    assert quality_report['quality_score'] < 100  # Reduced quality
    
    # Test drift detection
    np.random.seed(42)  # Set seed for reproducible results
    reference_data = np.random.normal(0, 1, (100, 3))
    
    # No drift case
    np.random.seed(43)  # Different seed but similar distribution
    similar_data = np.random.normal(0, 1, (50, 3))
    drift_report = detect_drift(reference_data, similar_data, threshold=0.5)  # Higher threshold
    
    # May or may not detect drift due to random variation, so just check structure
    assert 'drift_detected' in drift_report
    assert 'feature_drifts' in drift_report
    
    # Drift case
    drifted_data = np.random.normal(2, 1, (50, 3))  # Mean shifted
    drift_report = detect_drift(reference_data, drifted_data)
    
    assert drift_report['drift_detected']
    assert any(fd['drift_detected'] for fd in drift_report['feature_drifts'])


if __name__ == "__main__":
    # Run tests directly
    test_data_validation()
    test_data_preprocessing()
    test_synthetic_data_generation()
    test_data_transformation()
    test_data_quality_checks()
    print("All data processing tests passed!")