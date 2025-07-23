#!/usr/bin/env python3
"""
Testing and Validation Guide for Anomaly Detection Systems
=========================================================

Comprehensive guide for testing anomaly detection models, including
validation strategies, performance testing, and quality assurance.

Usage:
    python testing_and_validation_guide.py

Requirements:
    - anomaly_detection
    - scikit-learn
    - pandas
    - numpy
    - pytest (optional)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import time
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from anomaly_detection import DetectionService, EnsembleService
from anomaly_detection.domain.services.streaming_service import StreamingService


@dataclass
class ValidationResult:
    """Result of a validation test."""
    
    test_name: str
    algorithm: str
    dataset_name: str
    success: bool
    metrics: Dict[str, float]
    processing_time: float
    errors: List[str]
    warnings: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'algorithm': self.algorithm,
            'dataset_name': self.dataset_name,
            'success': self.success,
            'metrics': self.metrics,
            'processing_time': self.processing_time,
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat()
        }


class AnomalyDetectionValidator:
    """Comprehensive validator for anomaly detection systems."""
    
    def __init__(self):
        self.detection_service = DetectionService()
        self.ensemble_service = EnsembleService()
        self.results: List[ValidationResult] = []
        
        # Test datasets
        self.test_datasets = {}
        self._generate_test_datasets()
    
    def _generate_test_datasets(self):
        """Generate various test datasets for validation."""
        
        # 1. Simple Gaussian dataset
        np.random.seed(42)
        normal_data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 800)
        anomaly_data = np.random.multivariate_normal([4, 4], [[1, 0], [0, 1]], 200)
        
        self.test_datasets['gaussian_2d'] = {
            'data': np.vstack([normal_data, anomaly_data]),
            'labels': np.hstack([np.ones(800), -np.ones(200)]),
            'contamination': 0.2,
            'description': 'Simple 2D Gaussian with clear separation'
        }
        
        # 2. High-dimensional dataset
        normal_high = np.random.multivariate_normal(np.zeros(20), np.eye(20), 900)
        anomaly_high = np.random.multivariate_normal(np.ones(20) * 3, np.eye(20), 100)
        
        self.test_datasets['high_dimensional'] = {
            'data': np.vstack([normal_high, anomaly_high]),
            'labels': np.hstack([np.ones(900), -np.ones(100)]),
            'contamination': 0.1,
            'description': '20D dataset testing high-dimensional performance'
        }
        
        # 3. Imbalanced dataset
        normal_imb = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 980)
        anomaly_imb = np.random.multivariate_normal([3, -2], [[0.5, 0], [0, 0.5]], 20)
        
        self.test_datasets['imbalanced'] = {
            'data': np.vstack([normal_imb, anomaly_imb]),
            'labels': np.hstack([np.ones(980), -np.ones(20)]),
            'contamination': 0.02,
            'description': 'Highly imbalanced dataset (2% anomalies)'
        }
        
        # 4. Clustered dataset
        cluster1 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], 300)
        cluster2 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 300)
        cluster3 = np.random.multivariate_normal([-2, 2], [[0.5, 0], [0, 0.5]], 300)
        outliers = np.random.uniform(-5, 5, (100, 2))
        
        self.test_datasets['clustered'] = {
            'data': np.vstack([cluster1, cluster2, cluster3, outliers]),
            'labels': np.hstack([np.ones(900), -np.ones(100)]),
            'contamination': 0.1,
            'description': 'Multi-cluster dataset with scattered outliers'
        }
        
        # 5. Time series dataset
        t = np.linspace(0, 10, 1000)
        normal_ts = np.column_stack([
            np.sin(t) + np.random.normal(0, 0.1, 1000),
            np.cos(t) + np.random.normal(0, 0.1, 1000)
        ])
        
        # Add anomalies at specific points
        anomaly_indices = [100, 300, 500, 700, 900]
        anomaly_ts = normal_ts.copy()
        for idx in anomaly_indices:
            anomaly_ts[idx] = [5, 5]  # Spike anomalies
        
        labels_ts = np.ones(1000)
        labels_ts[anomaly_indices] = -1
        
        self.test_datasets['time_series'] = {
            'data': anomaly_ts,
            'labels': labels_ts,
            'contamination': 0.005,
            'description': 'Time series with spike anomalies'
        }
    
    def validate_algorithm_performance(
        self,
        algorithm: str,
        dataset_name: str,
        **algorithm_params
    ) -> ValidationResult:
        """
        Validate algorithm performance on a specific dataset.
        
        Args:
            algorithm: Algorithm name to test
            dataset_name: Name of test dataset
            **algorithm_params: Algorithm-specific parameters
            
        Returns:
            ValidationResult with performance metrics
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Get dataset
            if dataset_name not in self.test_datasets:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            dataset = self.test_datasets[dataset_name]
            X = dataset['data']
            y_true = dataset['labels']
            contamination = dataset['contamination']
            
            # Run detection
            result = self.detection_service.detect_anomalies(
                data=X,
                algorithm=algorithm,
                contamination=contamination,
                **algorithm_params
            )
            
            if not result.success:
                raise RuntimeError("Detection failed")
            
            # Calculate metrics
            y_pred = result.predictions
            
            # Convert to binary format for metrics
            y_true_binary = (y_true == -1).astype(int)
            y_pred_binary = (y_pred == -1).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(y_true_binary, y_pred_binary),
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'contamination_rate': np.mean(y_pred_binary),
                'expected_contamination': contamination,
                'contamination_error': abs(np.mean(y_pred_binary) - contamination),
                'total_samples': len(X),
                'detected_anomalies': np.sum(y_pred_binary),
                'true_anomalies': np.sum(y_true_binary)
            }
            
            # Add AUC if we have anomaly scores
            if hasattr(result, 'anomaly_scores') and result.anomaly_scores is not None:
                try:
                    metrics['auc_roc'] = roc_auc_score(y_true_binary, result.anomaly_scores)
                except Exception as e:
                    warnings.append(f"Could not calculate AUC: {e}")
            
            # Performance checks
            if metrics['contamination_error'] > 0.05:  # 5% tolerance
                warnings.append(f"Contamination rate error: {metrics['contamination_error']:.3f}")
            
            if metrics['f1_score'] < 0.5:
                warnings.append(f"Low F1 score: {metrics['f1_score']:.3f}")
            
            processing_time = time.time() - start_time
            
            return ValidationResult(
                test_name=f"algorithm_performance_{algorithm}_{dataset_name}",
                algorithm=algorithm,
                dataset_name=dataset_name,
                success=True,
                metrics=metrics,
                processing_time=processing_time,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            errors.append(str(e))
            processing_time = time.time() - start_time
            
            return ValidationResult(
                test_name=f"algorithm_performance_{algorithm}_{dataset_name}",
                algorithm=algorithm,
                dataset_name=dataset_name,
                success=False,
                metrics={},
                processing_time=processing_time,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now()
            )
    
    def cross_validate_algorithm(
        self,
        algorithm: str,
        dataset_name: str,
        cv_folds: int = 5,
        **algorithm_params
    ) -> ValidationResult:
        """
        Perform cross-validation on an algorithm.
        
        Args:
            algorithm: Algorithm name
            dataset_name: Name of test dataset
            cv_folds: Number of cross-validation folds
            **algorithm_params: Algorithm parameters
            
        Returns:
            ValidationResult with cross-validation metrics
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            dataset = self.test_datasets[dataset_name]
            X = dataset['data']
            y_true = dataset['labels']
            contamination = dataset['contamination']
            
            # Prepare for cross-validation
            y_true_binary = (y_true == -1).astype(int)
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            fold_results = []
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_true_binary)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_test = y_true[test_idx]
                
                # Fit and predict
                self.detection_service.fit(X_train, algorithm=algorithm, **algorithm_params)
                result = self.detection_service.predict(X_test, algorithm=algorithm)
                
                if result.success:
                    y_pred = result.predictions
                    y_test_binary = (y_test == -1).astype(int)
                    y_pred_binary = (y_pred == -1).astype(int)
                    
                    fold_metrics = {
                        'accuracy': accuracy_score(y_test_binary, y_pred_binary),
                        'precision': precision_score(y_test_binary, y_pred_binary, zero_division=0),
                        'recall': recall_score(y_test_binary, y_pred_binary, zero_division=0),
                        'f1_score': f1_score(y_test_binary, y_pred_binary, zero_division=0)
                    }
                    
                    fold_results.append(fold_metrics)
            
            # Calculate mean and std across folds
            if fold_results:
                metrics = {}
                for metric_name in fold_results[0].keys():
                    values = [fold[metric_name] for fold in fold_results]
                    metrics[f'{metric_name}_mean'] = np.mean(values)
                    metrics[f'{metric_name}_std'] = np.std(values)
                
                metrics['cv_folds'] = len(fold_results)
                metrics['successful_folds'] = len(fold_results)
                
                # Check for high variance
                for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
                    std_key = f'{metric_name}_std'
                    if std_key in metrics and metrics[std_key] > 0.1:
                        warnings.append(f"High variance in {metric_name}: {metrics[std_key]:.3f}")
            else:
                raise RuntimeError("All cross-validation folds failed")
            
            processing_time = time.time() - start_time
            
            return ValidationResult(
                test_name=f"cross_validation_{algorithm}_{dataset_name}",
                algorithm=algorithm,
                dataset_name=dataset_name,
                success=True,
                metrics=metrics,
                processing_time=processing_time,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            errors.append(str(e))
            processing_time = time.time() - start_time
            
            return ValidationResult(
                test_name=f"cross_validation_{algorithm}_{dataset_name}",
                algorithm=algorithm,
                dataset_name=dataset_name,
                success=False,
                metrics={},
                processing_time=processing_time,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now()
            )
    
    def validate_ensemble_performance(
        self,
        algorithms: List[str],
        ensemble_method: str,
        dataset_name: str
    ) -> ValidationResult:
        """
        Validate ensemble method performance.
        
        Args:
            algorithms: List of algorithms to ensemble
            ensemble_method: Ensemble combination method
            dataset_name: Name of test dataset
            
        Returns:
            ValidationResult for ensemble
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            dataset = self.test_datasets[dataset_name]
            X = dataset['data']
            y_true = dataset['labels']
            contamination = dataset['contamination']
            
            # Run ensemble detection
            result = self.ensemble_service.detect_with_ensemble(
                data=X,
                algorithms=algorithms,
                method=ensemble_method,
                contamination=contamination
            )
            
            if not result.success:
                raise RuntimeError("Ensemble detection failed")
            
            # Calculate metrics
            y_pred = result.predictions
            y_true_binary = (y_true == -1).astype(int)
            y_pred_binary = (y_pred == -1).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(y_true_binary, y_pred_binary),
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'num_algorithms': len(algorithms),
                'ensemble_method': ensemble_method,
                'contamination_rate': np.mean(y_pred_binary),
                'expected_contamination': contamination
            }
            
            processing_time = time.time() - start_time
            
            return ValidationResult(
                test_name=f"ensemble_{ensemble_method}_{dataset_name}",
                algorithm=f"ensemble_{ensemble_method}",
                dataset_name=dataset_name,
                success=True,
                metrics=metrics,
                processing_time=processing_time,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            errors.append(str(e))
            processing_time = time.time() - start_time
            
            return ValidationResult(
                test_name=f"ensemble_{ensemble_method}_{dataset_name}",
                algorithm=f"ensemble_{ensemble_method}",
                dataset_name=dataset_name,
                success=False,
                metrics={},
                processing_time=processing_time,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now()
            )
    
    def validate_streaming_performance(
        self,
        dataset_name: str,
        window_size: int = 100,
        update_frequency: int = 10
    ) -> ValidationResult:
        """
        Validate streaming detection performance.
        
        Args:
            dataset_name: Name of test dataset
            window_size: Streaming window size
            update_frequency: Model update frequency
            
        Returns:
            ValidationResult for streaming
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            dataset = self.test_datasets[dataset_name]
            X = dataset['data']
            y_true = dataset['labels']
            
            # Initialize streaming service
            streaming_service = StreamingService(
                window_size=window_size,
                update_frequency=update_frequency
            )
            
            # Process samples one by one
            predictions = []
            for i, sample in enumerate(X):
                result = streaming_service.process_sample(sample)
                predictions.append(-1 if result.is_anomaly else 1)
            
            # Calculate metrics
            y_true_binary = (y_true == -1).astype(int)
            y_pred_binary = (np.array(predictions) == -1).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(y_true_binary, y_pred_binary),
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'window_size': window_size,
                'update_frequency': update_frequency,
                'total_samples_processed': len(X),
                'streaming_contamination_rate': np.mean(y_pred_binary)
            }
            
            processing_time = time.time() - start_time
            
            return ValidationResult(
                test_name=f"streaming_{dataset_name}",
                algorithm="streaming_detection",
                dataset_name=dataset_name,
                success=True,
                metrics=metrics,
                processing_time=processing_time,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            errors.append(str(e))
            processing_time = time.time() - start_time
            
            return ValidationResult(
                test_name=f"streaming_{dataset_name}",
                algorithm="streaming_detection",
                dataset_name=dataset_name,
                success=False,
                metrics={},
                processing_time=processing_time,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now()
            )
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation across all algorithms and datasets.
        
        Returns:
            Summary of all validation results
        """
        print("🧪 Running Comprehensive Validation Suite")
        print("=" * 50)
        
        # Algorithms to test
        algorithms = ['iforest', 'lof', 'ocsvm', 'elliptic']
        
        # Test each algorithm on each dataset
        for algorithm in algorithms:
            print(f"\n🔍 Testing {algorithm}...")
            
            for dataset_name in self.test_datasets.keys():
                print(f"  Dataset: {dataset_name}")
                
                try:
                    # Basic performance test
                    result = self.validate_algorithm_performance(algorithm, dataset_name)
                    self.results.append(result)
                    
                    if result.success:
                        f1 = result.metrics.get('f1_score', 0)
                        processing_time = result.processing_time
                        print(f"    ✅ F1: {f1:.3f}, Time: {processing_time:.2f}s")
                    else:
                        print(f"    ❌ Failed: {', '.join(result.errors)}")
                    
                    # Cross-validation (only for smaller datasets)
                    if dataset_name in ['gaussian_2d', 'clustered']:
                        cv_result = self.cross_validate_algorithm(algorithm, dataset_name)
                        self.results.append(cv_result)
                        
                        if cv_result.success:
                            f1_mean = cv_result.metrics.get('f1_score_mean', 0)
                            f1_std = cv_result.metrics.get('f1_score_std', 0)
                            print(f"    📊 CV F1: {f1_mean:.3f} ± {f1_std:.3f}")
                
                except Exception as e:
                    print(f"    ❌ Exception: {e}")
        
        # Test ensemble methods
        print(f"\n🎯 Testing Ensemble Methods...")
        
        ensemble_methods = ['majority', 'average', 'maximum']
        base_algorithms = ['iforest', 'lof']
        
        for method in ensemble_methods:
            for dataset_name in ['gaussian_2d', 'clustered']:
                result = self.validate_ensemble_performance(
                    base_algorithms, method, dataset_name
                )
                self.results.append(result)
                
                if result.success:
                    f1 = result.metrics.get('f1_score', 0)
                    print(f"  {method} on {dataset_name}: F1={f1:.3f}")
        
        # Test streaming performance
        print(f"\n🌊 Testing Streaming Performance...")
        
        for dataset_name in ['time_series', 'gaussian_2d']:
            result = self.validate_streaming_performance(dataset_name)
            self.results.append(result)
            
            if result.success:
                f1 = result.metrics.get('f1_score', 0)
                print(f"  Streaming {dataset_name}: F1={f1:.3f}")
        
        # Generate summary
        return self.generate_validation_report()
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        # Overall statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        # Performance by algorithm
        algorithm_performance = {}
        for result in self.results:
            if result.success and 'f1_score' in result.metrics:
                algo = result.algorithm
                if algo not in algorithm_performance:
                    algorithm_performance[algo] = []
                algorithm_performance[algo].append(result.metrics['f1_score'])
        
        # Calculate mean performance
        for algo in algorithm_performance:
            scores = algorithm_performance[algo]
            algorithm_performance[algo] = {
                'mean_f1': np.mean(scores),
                'std_f1': np.std(scores),
                'test_count': len(scores)
            }
        
        # Dataset difficulty analysis
        dataset_performance = {}
        for result in self.results:
            if result.success and 'f1_score' in result.metrics:
                dataset = result.dataset_name
                if dataset not in dataset_performance:
                    dataset_performance[dataset] = []
                dataset_performance[dataset].append(result.metrics['f1_score'])
        
        for dataset in dataset_performance:
            scores = dataset_performance[dataset]
            dataset_performance[dataset] = {
                'mean_f1': np.mean(scores),
                'std_f1': np.std(scores),
                'difficulty': 'Easy' if np.mean(scores) > 0.8 else 'Medium' if np.mean(scores) > 0.6 else 'Hard'
            }
        
        # Processing time analysis
        processing_times = [r.processing_time for r in self.results if r.success]
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0
            },
            'algorithm_performance': algorithm_performance,
            'dataset_analysis': dataset_performance,
            'performance_metrics': {
                'mean_processing_time': np.mean(processing_times) if processing_times else 0,
                'median_processing_time': np.median(processing_times) if processing_times else 0,
                'max_processing_time': np.max(processing_times) if processing_times else 0
            },
            'validation_timestamp': datetime.now().isoformat(),
            'detailed_results': [r.to_dict() for r in self.results]
        }
        
        return report
    
    def save_validation_report(self, filename: str = "validation_report.json"):
        """Save validation report to file."""
        report = self.generate_validation_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Validation report saved to {filename}")


def demonstrate_testing_strategies():
    """Demonstrate different testing strategies for anomaly detection."""
    
    print("🧪 Testing and Validation Strategies Guide")
    print("=" * 50)
    
    print("\n📋 Testing Checklist:")
    
    checklist = [
        "✅ Algorithm Performance Testing",
        "✅ Cross-validation with Multiple Folds",
        "✅ Ensemble Method Validation",
        "✅ Streaming Performance Testing",
        "✅ Scalability Testing (Large Datasets)",
        "✅ Robustness Testing (Noisy Data)",
        "✅ Edge Case Testing (Empty/Invalid Data)",
        "✅ Performance benchmarking",
        "✅ Memory Usage Validation",
        "✅ Reproducibility Testing"
    ]
    
    for item in checklist:
        print(f"   {item}")
    
    print("\n🎯 Key Metrics to Track:")
    
    metrics = [
        "• Precision: How many detected anomalies are actually anomalies?",
        "• Recall: How many actual anomalies were detected?",
        "• F1-Score: Harmonic mean of precision and recall",
        "• AUC-ROC: Area under the ROC curve",
        "• Processing Time: Time to detect anomalies",
        "• Memory Usage: Peak memory consumption",
        "• Contamination Error: Difference between expected and actual anomaly rate",
        "• Cross-validation Stability: Variance across folds",
        "• Scalability: Performance with increasing data size"
    ]
    
    for metric in metrics:
        print(f"   {metric}")
    
    print("\n🔍 Testing Best Practices:")
    
    practices = [
        "• Use multiple diverse datasets for comprehensive validation",
        "• Implement automated testing with CI/CD pipelines",
        "• Test with realistic data volumes and characteristics",
        "• Validate performance across different contamination rates",
        "• Test algorithm stability with repeated runs",
        "• Monitor resource usage during testing",
        "• Validate streaming performance with real-time data",
        "• Test edge cases and error conditions",
        "• Compare against baseline methods",
        "• Document all test configurations and results"
    ]
    
    for practice in practices:
        print(f"   {practice}")


def main():
    """Main function to run validation guide."""
    
    demonstrate_testing_strategies()
    
    print(f"\n🚀 Ready to run comprehensive validation?")
    choice = input("Run validation suite? (y/n): ").lower().strip()
    
    if choice == 'y':
        validator = AnomalyDetectionValidator()
        
        print("\nStarting comprehensive validation...")
        print("This may take several minutes...")
        
        report = validator.run_comprehensive_validation()
        
        # Print summary
        print(f"\n📊 Validation Summary:")
        print(f"   Total tests: {report['summary']['total_tests']}")
        print(f"   Success rate: {report['summary']['success_rate']:.1%}")
        print(f"   Mean processing time: {report['performance_metrics']['mean_processing_time']:.2f}s")
        
        # Save detailed report
        validator.save_validation_report()
        
        print(f"\n✅ Validation complete!")
    
    else:
        print("Validation skipped. Use the ValidationResult class to implement your own tests!")


if __name__ == "__main__":
    main()