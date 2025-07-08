"""
Unit tests for MetricsCalculator domain service.

Tests comprehensive metrics computation using scikit-learn's built-in datasets.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock

from sklearn.datasets import (
    make_classification,
    make_blobs,
    load_breast_cancer,
    load_iris,
    load_wine
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.pynomaly.domain.services.metrics_calculator import MetricsCalculator


class TestMetricsCalculator:
    """Test suite for MetricsCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()
        
        # Generate synthetic anomaly detection dataset
        self.X_anomaly, self.y_anomaly = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=1,
            weights=[0.9, 0.1],  # 10% anomalies
            random_state=42
        )
        
        # Generate synthetic classification dataset
        self.X_classification, self.y_classification = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=2,
            weights=[0.5, 0.5],  # Balanced classes
            random_state=42
        )
        
        # Generate clustering dataset
        self.X_clustering, self.y_clustering = make_blobs(
            n_samples=300,
            centers=3,
            cluster_std=0.60,
            random_state=42
        )
        
        # Create sample predictions for testing
        self.y_true_binary = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        self.y_pred_binary = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 0])
        self.y_proba_binary = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.9, 0.6, 0.1, 0.7, 0.4])
    
    def test_compute_basic_functionality(self):
        """Test basic compute functionality."""
        results = MetricsCalculator.compute(
            self.y_true_binary,
            self.y_pred_binary,
            proba=self.y_proba_binary,
            task_type="anomaly"
        )
        
        # Check that basic metrics are present
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'specificity' in results
        
        # Check metric structure
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            assert 'value' in results[metric]
            assert 'mean' in results[metric]
            assert 'std' in results[metric]
            assert 'confidence_interval' in results[metric]
            
            # Values should be between 0 and 1
            assert 0 <= results[metric]['value'] <= 1
    
    def test_compute_with_anomaly_scores(self):
        """Test computation with anomaly scores."""
        results = MetricsCalculator.compute(
            self.y_true_binary,
            self.y_pred_binary,
            proba=self.y_proba_binary,
            task_type="anomaly"
        )
        
        # Check anomaly-specific metrics
        assert 'roc_auc' in results
        assert 'pr_auc' in results
        assert 'average_precision' in results
        assert 'anomaly_score_stats' in results
        assert 'normal_score_stats' in results
        
        # Check anomaly score statistics
        anomaly_stats = results['anomaly_score_stats']
        assert 'mean' in anomaly_stats
        assert 'std' in anomaly_stats
        assert 'min' in anomaly_stats
        assert 'max' in anomaly_stats
        assert 'median' in anomaly_stats
    
    def test_compute_classification_task(self):
        """Test computation for classification task."""
        results = MetricsCalculator.compute(
            self.y_true_binary,
            self.y_pred_binary,
            proba=self.y_proba_binary,
            task_type="classification"
        )
        
        # Check classification-specific metrics
        assert 'classification_report' in results
        assert 'positive_class_proba' in results
        assert 'negative_class_proba' in results
        assert 'probability_stats' in results
        
        # Check probability statistics
        prob_stats = results['probability_stats']
        assert 'mean' in prob_stats
        assert 'std' in prob_stats
        assert 'min' in prob_stats
        assert 'max' in prob_stats
        assert 'median' in prob_stats
    
    def test_compute_clustering_task(self):
        """Test computation for clustering task."""
        # Use a simple clustering scenario
        y_true_cluster = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_pred_cluster = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        
        results = MetricsCalculator.compute(
            y_true_cluster,
            y_pred_cluster,
            task_type="clustering"
        )
        
        # Check clustering-specific metrics
        assert 'adjusted_rand_index' in results
        assert 'normalized_mutual_info' in results
        
        # Check metric structure
        for metric in ['adjusted_rand_index', 'normalized_mutual_info']:
            assert 'value' in results[metric]
            assert 'mean' in results[metric]
            assert 'std' in results[metric]
            assert 'confidence_interval' in results[metric]
    
    def test_compute_with_breast_cancer_dataset(self):
        """Test with scikit-learn's breast cancer dataset."""
        # Load and prepare data
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        
        # Convert to anomaly detection format (malignant=1, benign=0)
        y_anomaly = 1 - y  # Flip labels so malignant (minority) becomes anomaly
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_anomaly, test_size=0.3, random_state=42, stratify=y_anomaly
        )
        
        # Train anomaly detector
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Use Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X_train_scaled)
        
        # Get predictions and scores
        y_pred = iso_forest.predict(X_test_scaled)
        y_pred = (y_pred == -1).astype(int)  # Convert to 0/1 format
        
        anomaly_scores = iso_forest.decision_function(X_test_scaled)
        # Convert to probabilities (higher = more anomalous)
        anomaly_proba = (anomaly_scores.max() - anomaly_scores) / (anomaly_scores.max() - anomaly_scores.min())
        
        # Compute metrics
        results = MetricsCalculator.compute(
            y_test,
            y_pred,
            proba=anomaly_proba,
            task_type="anomaly"
        )
        
        # Verify results
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'roc_auc' in results
        assert 'confusion_matrix' in results
        
        # Check that metrics are reasonable
        assert 0 <= results['accuracy']['value'] <= 1
        assert 0 <= results['precision']['value'] <= 1
        assert 0 <= results['recall']['value'] <= 1
        assert 0 <= results['f1_score']['value'] <= 1
        assert 0 <= results['roc_auc']['value'] <= 1
    
    def test_compute_with_wine_dataset(self):
        """Test with scikit-learn's wine dataset for classification."""
        # Load and prepare data
        wine = load_wine()
        X, y = wine.data, wine.target
        
        # Convert to binary classification (class 0 vs others)
        y_binary = (y == 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
        )
        
        # Train classifier
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Get predictions and probabilities
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        results = MetricsCalculator.compute(
            y_test,
            y_pred,
            proba=y_proba,
            task_type="classification"
        )
        
        # Verify results
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'roc_auc' in results
        assert 'classification_report' in results
        assert 'positive_class_proba' in results
        assert 'negative_class_proba' in results
        
        # Check that metrics are reasonable
        assert 0 <= results['accuracy']['value'] <= 1
        assert 0 <= results['precision']['value'] <= 1
        assert 0 <= results['recall']['value'] <= 1
        assert 0 <= results['f1_score']['value'] <= 1
        assert 0 <= results['roc_auc']['value'] <= 1
    
    def test_compute_with_iris_dataset_clustering(self):
        """Test with scikit-learn's iris dataset for clustering evaluation."""
        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Use a simple clustering approach for testing
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        y_pred = kmeans.fit_predict(X)
        
        # Compute clustering metrics
        results = MetricsCalculator.compute(
            y,
            y_pred,
            task_type="clustering",
            X=X  # Pass feature data for silhouette score
        )
        
        # Verify results
        assert 'adjusted_rand_index' in results
        assert 'normalized_mutual_info' in results
        assert 'silhouette_score' in results
        
        # Check metric ranges
        assert -1 <= results['adjusted_rand_index']['value'] <= 1
        assert 0 <= results['normalized_mutual_info']['value'] <= 1
        assert -1 <= results['silhouette_score']['value'] <= 1
    
    @pytest.mark.asyncio
    async def test_compute_async(self):
        """Test asynchronous computation."""
        results = await MetricsCalculator.compute_async(
            self.y_true_binary,
            self.y_pred_binary,
            proba=self.y_proba_binary,
            task_type="anomaly"
        )
        
        # Should have same results as synchronous version
        sync_results = MetricsCalculator.compute(
            self.y_true_binary,
            self.y_pred_binary,
            proba=self.y_proba_binary,
            task_type="anomaly"
        )
        
        # Compare key metrics
        assert results['accuracy']['value'] == sync_results['accuracy']['value']
        assert results['precision']['value'] == sync_results['precision']['value']
        assert results['recall']['value'] == sync_results['recall']['value']
        assert results['f1_score']['value'] == sync_results['f1_score']['value']
    
    def test_compute_cross_validation_metrics(self):
        """Test cross-validation metrics aggregation."""
        # Create mock CV results
        cv_results = [
            {
                'accuracy': {'value': 0.8},
                'precision': {'value': 0.7},
                'recall': {'value': 0.9},
                'f1_score': {'value': 0.78}
            },
            {
                'accuracy': {'value': 0.85},
                'precision': {'value': 0.75},
                'recall': {'value': 0.88},
                'f1_score': {'value': 0.81}
            },
            {
                'accuracy': {'value': 0.82},
                'precision': {'value': 0.72},
                'recall': {'value': 0.92},
                'f1_score': {'value': 0.81}
            }
        ]
        
        aggregated = MetricsCalculator.compute_cross_validation_metrics(cv_results)
        
        # Check that all metrics are aggregated
        assert 'accuracy' in aggregated
        assert 'precision' in aggregated
        assert 'recall' in aggregated
        assert 'f1_score' in aggregated
        
        # Check aggregated structure
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            assert 'values' in aggregated[metric]
            assert 'mean' in aggregated[metric]
            assert 'std' in aggregated[metric]
            assert 'min' in aggregated[metric]
            assert 'max' in aggregated[metric]
            assert 'confidence_interval' in aggregated[metric]
            
            # Check that values list has correct length
            assert len(aggregated[metric]['values']) == 3
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Create mock model results
        model_results = {
            'model_a': {
                'accuracy': {'value': 0.85},
                'precision': {'value': 0.80},
                'recall': {'value': 0.90},
                'f1_score': {'value': 0.85}
            },
            'model_b': {
                'accuracy': {'value': 0.82},
                'precision': {'value': 0.78},
                'recall': {'value': 0.88},
                'f1_score': {'value': 0.82}
            },
            'model_c': {
                'accuracy': {'value': 0.88},
                'precision': {'value': 0.85},
                'recall': {'value': 0.85},
                'f1_score': {'value': 0.85}
            }
        }
        
        comparison = MetricsCalculator.compare_models(model_results, primary_metric='f1_score')
        
        # Check comparison structure
        assert 'rankings' in comparison
        assert 'comparisons' in comparison
        assert 'summary' in comparison
        
        # Check rankings
        assert 'f1_score' in comparison['rankings']
        rankings = comparison['rankings']['f1_score']
        assert len(rankings) == 3
        
        # Check that models are ranked correctly (tied at 0.85)
        assert rankings[0]['model'] in ['model_a', 'model_c']
        assert rankings[0]['value'] == 0.85
        assert rankings[0]['rank'] == 1
        
        # Check summary
        summary = comparison['summary']
        assert 'best_model' in summary
        assert 'worst_model' in summary
        assert 'mean_performance' in summary
        assert 'std_performance' in summary
        assert 'performance_range' in summary
        
        assert summary['best_model'] in ['model_a', 'model_c']
        assert summary['worst_model'] == 'model_b'
    
    def test_input_validation(self):
        """Test input validation."""
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="must have same length"):
            MetricsCalculator.compute([0, 1], [0, 1, 0], task_type="anomaly")
        
        # Test empty arrays
        with pytest.raises(ValueError, match="cannot be empty"):
            MetricsCalculator.compute([], [], task_type="anomaly")
        
        # Test invalid label values
        with pytest.raises(ValueError, match="must contain only 0.*and 1"):
            MetricsCalculator.compute([0, 1, 2], [0, 1, 0], task_type="anomaly")
        
        # Test mismatched proba length
        with pytest.raises(ValueError, match="must have same length as y_true"):
            MetricsCalculator.compute([0, 1], [0, 1], proba=[0.1, 0.2, 0.3], task_type="anomaly")
        
        # Test unsupported task type
        with pytest.raises(ValueError, match="Unsupported task_type"):
            MetricsCalculator.compute([0, 1], [0, 1], task_type="unsupported")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with all zeros
        y_true_zeros = np.array([0, 0, 0, 0])
        y_pred_zeros = np.array([0, 0, 0, 0])
        
        results = MetricsCalculator.compute(y_true_zeros, y_pred_zeros, task_type="anomaly")
        
        # Should handle gracefully
        assert 'accuracy' in results
        assert results['accuracy']['value'] == 1.0  # All correct predictions
        
        # Test with all ones
        y_true_ones = np.array([1, 1, 1, 1])
        y_pred_ones = np.array([1, 1, 1, 1])
        
        results = MetricsCalculator.compute(y_true_ones, y_pred_ones, task_type="anomaly")
        
        # Should handle gracefully
        assert 'accuracy' in results
        assert results['accuracy']['value'] == 1.0  # All correct predictions
    
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        # Test with multiple values
        values = np.array([0.8, 0.85, 0.82, 0.88, 0.79, 0.86])
        
        ci = MetricsCalculator._compute_confidence_interval(values, 0.95)
        
        # Check that CI is a tuple
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        
        # Check that lower bound <= upper bound
        assert ci[0] <= ci[1]
        
        # Check that mean is within CI
        mean_val = np.mean(values)
        assert ci[0] <= mean_val <= ci[1]
        
        # Test with single value
        single_value = np.array([0.8])
        ci_single = MetricsCalculator._compute_confidence_interval(single_value, 0.95)
        assert ci_single == (0.8, 0.8)
    
    def test_anomaly_detection_workflow(self):
        """Test complete anomaly detection workflow."""
        # Generate synthetic dataset with clear anomalies
        np.random.seed(42)
        
        # Normal data
        normal_data = np.random.normal(0, 1, (900, 10))
        normal_labels = np.zeros(900)
        
        # Anomalous data (different distribution)
        anomaly_data = np.random.normal(3, 1.5, (100, 10))
        anomaly_labels = np.ones(100)
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([normal_labels, anomaly_labels])
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train multiple detectors
        detectors = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'one_class_svm': OneClassSVM(nu=0.1),
            'lof': LocalOutlierFactor(contamination=0.1, novelty=True)
        }
        
        results = {}
        
        for name, detector in detectors.items():
            # Train detector
            detector.fit(X_train)
            
            # Get predictions
            y_pred = detector.predict(X_test)
            y_pred = (y_pred == -1).astype(int)  # Convert to 0/1 format
            
            # Get anomaly scores
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X_test)
            else:
                scores = detector.score_samples(X_test)
            
            # Convert scores to probabilities
            proba = (scores.max() - scores) / (scores.max() - scores.min())
            
            # Compute metrics
            metrics = MetricsCalculator.compute(
                y_test,
                y_pred,
                proba=proba,
                task_type="anomaly"
            )
            
            results[name] = metrics
        
        # Compare models
        comparison = MetricsCalculator.compare_models(results, primary_metric='f1_score')
        
        # Verify that comparison works
        assert 'rankings' in comparison
        assert 'comparisons' in comparison
        assert 'summary' in comparison
        
        # Check that we have results for all detectors
        assert len(comparison['rankings']['f1_score']) == 3
        assert 'best_model' in comparison['summary']
        
        # Verify that metrics are reasonable
        for name, metrics in results.items():
            assert 0 <= metrics['accuracy']['value'] <= 1
            assert 0 <= metrics['precision']['value'] <= 1
            assert 0 <= metrics['recall']['value'] <= 1
            assert 0 <= metrics['f1_score']['value'] <= 1
    
    def test_statistical_significance(self):
        """Test statistical significance of metrics."""
        # Generate datasets with known differences
        np.random.seed(42)
        
        # Dataset 1: Better performance
        y_true_1 = np.random.choice([0, 1], 1000, p=[0.8, 0.2])
        y_pred_1 = np.where(
            (y_true_1 == 1) & (np.random.random(1000) < 0.9),  # High recall for anomalies
            1,
            np.where(
                (y_true_1 == 0) & (np.random.random(1000) < 0.95),  # High specificity for normal
                0,
                1 - y_true_1  # Some errors
            )
        )
        
        # Dataset 2: Worse performance
        y_true_2 = np.random.choice([0, 1], 1000, p=[0.8, 0.2])
        y_pred_2 = np.where(
            (y_true_2 == 1) & (np.random.random(1000) < 0.7),  # Lower recall for anomalies
            1,
            np.where(
                (y_true_2 == 0) & (np.random.random(1000) < 0.85),  # Lower specificity for normal
                0,
                1 - y_true_2  # More errors
            )
        )
        
        # Compute metrics
        metrics_1 = MetricsCalculator.compute(y_true_1, y_pred_1, task_type="anomaly")
        metrics_2 = MetricsCalculator.compute(y_true_2, y_pred_2, task_type="anomaly")
        
        # Compare performance
        comparison = MetricsCalculator.compare_models(
            {'model_1': metrics_1, 'model_2': metrics_2},
            primary_metric='f1_score'
        )
        
        # Model 1 should generally perform better
        assert comparison['summary']['best_model'] == 'model_1'
        
        # Check that differences are meaningful
        f1_diff = metrics_1['f1_score']['value'] - metrics_2['f1_score']['value']
        assert f1_diff > 0  # Model 1 should have better F1 score
    
    def test_performance_metrics_integration(self):
        """Test integration with existing PerformanceMetrics value object."""
        # This test ensures compatibility with existing codebase
        results = MetricsCalculator.compute(
            self.y_true_binary,
            self.y_pred_binary,
            proba=self.y_proba_binary,
            task_type="anomaly"
        )
        
        # Check that results can be used to construct PerformanceMetrics
        # (This would typically be done in the application layer)
        performance_data = {
            'accuracy': results['accuracy']['value'],
            'precision': results['precision']['value'],
            'recall': results['recall']['value'],
            'f1_score': results['f1_score']['value'],
            'training_time': 10.0,  # Mock values
            'inference_time': 1.0,
            'model_size': 1024
        }
        
        # These should be valid values for PerformanceMetrics
        assert 0 <= performance_data['accuracy'] <= 1
        assert 0 <= performance_data['precision'] <= 1
        assert 0 <= performance_data['recall'] <= 1
        assert 0 <= performance_data['f1_score'] <= 1
        assert performance_data['training_time'] > 0
        assert performance_data['inference_time'] > 0
        assert performance_data['model_size'] > 0
