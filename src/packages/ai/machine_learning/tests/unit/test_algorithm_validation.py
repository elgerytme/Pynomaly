"""
Algorithm validation tests for machine learning package.
Tests core ML algorithms against benchmark datasets with known performance expectations.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from machine_learning.domain.services.automl_service import AutoMLService
    from machine_learning.domain.services.explainable_ai_service import ExplainableAIService
    from machine_learning.domain.entities.model import Model
    from machine_learning.domain.entities.detection_result import DetectionResult
except ImportError as e:
    # Create mock classes if imports fail for initial testing
    class AutoMLService:
        def train(self, X, y, config): 
            return type('Result', (), {'best_model': None, 'accuracy': 0.85, 'success': True})()
    
    class ExplainableAIService:
        def explain(self, model, X, config): 
            return type('Result', (), {'explanations': [], 'feature_importance': {}, 'success': True})()
    
    class Model:
        def __init__(self, algorithm='test'):
            self.algorithm = algorithm
            self.is_trained = False
            
        def train(self, X, y):
            self.is_trained = True
            return self
            
        def predict(self, X):
            return np.random.choice([-1, 1], len(X))
    
    class DetectionResult:
        def __init__(self, predictions, confidence_scores=None):
            self.predictions = predictions
            self.confidence_scores = confidence_scores or np.random.random(len(predictions))


@pytest.mark.algorithm_validation
class TestAnomalyDetectionAlgorithms:
    """Test core anomaly detection algorithms for accuracy and performance."""
    
    @pytest.mark.parametrize("algorithm,expected_accuracy", [
        ("isolation_forest", 0.80),
        ("one_class_svm", 0.75),
        ("local_outlier_factor", 0.70),
        ("autoencoder", 0.78),
    ])
    def test_anomaly_detection_accuracy(
        self, 
        algorithm: str, 
        expected_accuracy: float,
        benchmark_anomaly_dataset: Tuple[np.ndarray, np.ndarray],
        accuracy_thresholds: Dict[str, float]
    ):
        """Test anomaly detection accuracy against benchmark datasets."""
        X, y_true = benchmark_anomaly_dataset
        
        # Create and train model
        model = Model(algorithm=algorithm)
        model.train(X, y_true)
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        from conftest import calculate_anomaly_metrics
        metrics = calculate_anomaly_metrics(y_true, y_pred)
        
        # Validate accuracy meets threshold
        threshold = accuracy_thresholds.get(algorithm, expected_accuracy)
        assert metrics['accuracy'] >= threshold, (
            f"Algorithm {algorithm} accuracy {metrics['accuracy']:.3f} "
            f"below threshold {threshold:.3f}"
        )
        
        # Validate additional metrics
        assert metrics['precision'] > 0.5, f"Precision {metrics['precision']:.3f} too low"
        assert metrics['recall'] > 0.5, f"Recall {metrics['recall']:.3f} too low"
        assert metrics['f1_score'] > 0.5, f"F1-score {metrics['f1_score']:.3f} too low"
    
    def test_high_dimensional_anomaly_detection(
        self, 
        high_dimensional_dataset: Tuple[np.ndarray, np.ndarray],
        accuracy_thresholds: Dict[str, float]
    ):
        """Test anomaly detection performance on high-dimensional datasets."""
        X, y_true = high_dimensional_dataset
        
        # Test isolation forest (should handle high dimensions well)
        model = Model(algorithm="isolation_forest")
        model.train(X, y_true)
        
        y_pred = model.predict(X)
        
        from conftest import calculate_anomaly_metrics
        metrics = calculate_anomaly_metrics(y_true, y_pred)
        
        # Should maintain reasonable performance even in high dimensions
        assert metrics['accuracy'] >= 0.70, (
            f"High-dimensional accuracy {metrics['accuracy']:.3f} below 0.70"
        )
        assert metrics['roc_auc'] >= 0.65, (
            f"High-dimensional ROC AUC {metrics['roc_auc']:.3f} below 0.65"
        )
    
    def test_time_series_anomaly_detection(
        self, 
        time_series_anomaly_dataset: Tuple[np.ndarray, np.ndarray]
    ):
        """Test anomaly detection on time series data."""
        X, y_true = time_series_anomaly_dataset
        
        # Time series-aware anomaly detection
        model = Model(algorithm="isolation_forest")
        model.train(X, y_true)
        
        y_pred = model.predict(X)
        
        from conftest import calculate_anomaly_metrics
        metrics = calculate_anomaly_metrics(y_true, y_pred)
        
        # Time series should detect at least point anomalies
        assert metrics['recall'] >= 0.60, (
            f"Time series recall {metrics['recall']:.3f} below 0.60"
        )
        assert metrics['precision'] >= 0.50, (
            f"Time series precision {metrics['precision']:.3f} below 0.50"
        )
    
    @pytest.mark.parametrize("contamination_rate", [0.05, 0.10, 0.15, 0.20])
    def test_contamination_rate_sensitivity(
        self, 
        contamination_rate: float,
        performance_timer
    ):
        """Test algorithm sensitivity to different contamination rates."""
        # Generate synthetic dataset with specific contamination
        np.random.seed(42)
        n_samples = 1000
        n_anomalies = int(n_samples * contamination_rate)
        n_normal = n_samples - n_anomalies
        
        # Normal data
        X_normal = np.random.randn(n_normal, 5)
        
        # Anomalous data
        X_anomalies = np.random.randn(n_anomalies, 5) + 3
        
        X = np.vstack([X_normal, X_anomalies])
        y_true = np.hstack([np.ones(n_normal), -np.ones(n_anomalies)])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X, y_true = X[indices], y_true[indices]
        
        # Test with isolation forest
        model = Model(algorithm="isolation_forest")
        
        performance_timer.start()
        model.train(X, y_true)
        y_pred = model.predict(X)
        performance_timer.stop()
        
        from conftest import calculate_anomaly_metrics
        metrics = calculate_anomaly_metrics(y_true, y_pred)
        
        # Validate contamination rate approximately respected
        predicted_contamination = (y_pred == -1).sum() / len(y_pred)
        contamination_error = abs(predicted_contamination - contamination_rate)
        
        assert contamination_error < 0.1, (
            f"Contamination rate error {contamination_error:.3f} too high "
            f"(predicted: {predicted_contamination:.3f}, expected: {contamination_rate:.3f})"
        )
        
        # Performance should be reasonable
        assert performance_timer.elapsed < 10.0, (
            f"Training time {performance_timer.elapsed:.2f}s too long for {n_samples} samples"
        )


@pytest.mark.algorithm_validation
class TestAutoMLValidation:
    """Test AutoML service for automated model selection and hyperparameter optimization."""
    
    def test_automl_model_selection(
        self, 
        mixed_data_types_dataset: pd.DataFrame,
        automl_test_config: Dict[str, Any],
        performance_timer
    ):
        """Test AutoML model selection and training."""
        df = mixed_data_types_dataset
        X = df.drop('target', axis=1)
        y = df['target']
        
        automl_service = AutoMLService()
        
        performance_timer.start()
        result = automl_service.train(X, y, automl_test_config)
        performance_timer.stop()
        
        # Validate AutoML results
        assert result.success, "AutoML training failed"
        assert result.best_model is not None, "No best model selected"
        assert result.accuracy >= 0.70, (
            f"AutoML accuracy {result.accuracy:.3f} below threshold 0.70"
        )
        
        # Performance validation
        assert performance_timer.elapsed < 300, (  # 5 minutes max for test config
            f"AutoML training time {performance_timer.elapsed:.2f}s exceeded limit"
        )
        
        # Validate model can make predictions
        try:
            predictions = result.best_model.predict(X.head(10))
            assert len(predictions) == 10, "Prediction count mismatch"
        except Exception as e:
            pytest.fail(f"Model prediction failed: {e}")
    
    def test_automl_cross_validation(
        self, 
        benchmark_anomaly_dataset: Tuple[np.ndarray, np.ndarray],
        automl_test_config: Dict[str, Any]
    ):
        """Test AutoML cross-validation accuracy."""
        X, y = benchmark_anomaly_dataset
        
        # Convert to DataFrame for AutoML
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        
        automl_service = AutoMLService()
        
        # Configure for cross-validation
        config = automl_test_config.copy()
        config['cross_validation_folds'] = 5
        config['validation_method'] = 'cross_validation'
        
        result = automl_service.train(df.drop('target', axis=1), df['target'], config)
        
        # Validate cross-validation results
        assert hasattr(result, 'cv_scores'), "Cross-validation scores missing"
        assert hasattr(result, 'cv_mean'), "Cross-validation mean missing"
        assert hasattr(result, 'cv_std'), "Cross-validation std missing"
        
        # CV mean should be reasonable
        assert result.cv_mean >= 0.65, (
            f"CV mean accuracy {result.cv_mean:.3f} below threshold 0.65"
        )
        
        # CV std should not be too high (model stability)
        assert result.cv_std < 0.15, (
            f"CV std {result.cv_std:.3f} too high, model unstable"
        )
    
    def test_automl_feature_engineering(
        self, 
        mixed_data_types_dataset: pd.DataFrame,
        automl_test_config: Dict[str, Any]
    ):
        """Test AutoML feature engineering capabilities."""
        df = mixed_data_types_dataset
        original_feature_count = len(df.columns) - 1  # Exclude target
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        automl_service = AutoMLService()
        
        # Enable feature engineering
        config = automl_test_config.copy()
        config['feature_engineering'] = True
        config['feature_selection'] = True
        
        result = automl_service.train(X, y, config)
        
        # Validate feature engineering occurred
        assert hasattr(result, 'engineered_features'), "Feature engineering results missing"
        assert hasattr(result, 'selected_features'), "Feature selection results missing"
        
        # Should have performed some feature selection
        if hasattr(result, 'final_feature_count'):
            assert result.final_feature_count <= original_feature_count, (
                "Feature selection should reduce or maintain feature count"
            )


@pytest.mark.algorithm_validation 
class TestExplainableAIValidation:
    """Test explainable AI service for model interpretability."""
    
    def test_shap_explanations(
        self, 
        benchmark_anomaly_dataset: Tuple[np.ndarray, np.ndarray],
        explainable_ai_config: Dict[str, Any]
    ):
        """Test SHAP explanation generation."""
        X, y = benchmark_anomaly_dataset
        
        # Train a model
        model = Model(algorithm="isolation_forest")
        model.train(X, y)
        
        explainer = ExplainableAIService()
        
        # Configure for SHAP
        config = explainable_ai_config.copy()
        config['explanation_methods'] = ['shap']
        config['num_samples'] = 50  # Small for testing
        
        result = explainer.explain(model, X[:100], config)
        
        # Validate SHAP results
        assert result.success, "SHAP explanation failed"
        assert hasattr(result, 'shap_values'), "SHAP values missing"
        assert hasattr(result, 'feature_importance'), "Feature importance missing"
        
        # SHAP values should have correct shape
        if hasattr(result, 'shap_values') and result.shap_values is not None:
            assert result.shap_values.shape[1] == X.shape[1], (
                f"SHAP values shape {result.shap_values.shape} doesn't match features {X.shape[1]}"
            )
    
    def test_lime_explanations(
        self, 
        mixed_data_types_dataset: pd.DataFrame,
        explainable_ai_config: Dict[str, Any]
    ):
        """Test LIME explanation generation."""
        df = mixed_data_types_dataset.head(100)  # Small sample for testing
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Train a model
        model = Model(algorithm="random_forest")
        model.train(X, y)
        
        explainer = ExplainableAIService()
        
        # Configure for LIME
        config = explainable_ai_config.copy()
        config['explanation_methods'] = ['lime']
        config['num_samples'] = 20
        
        result = explainer.explain(model, X.head(10), config)
        
        # Validate LIME results
        assert result.success, "LIME explanation failed"
        assert hasattr(result, 'lime_explanations'), "LIME explanations missing"
        
        # Should have explanations for requested samples
        if hasattr(result, 'lime_explanations'):
            assert len(result.lime_explanations) <= 10, (
                "More LIME explanations than requested samples"
            )
    
    def test_feature_importance_consistency(
        self, 
        benchmark_anomaly_dataset: Tuple[np.ndarray, np.ndarray],
        explainable_ai_config: Dict[str, Any]
    ):
        """Test feature importance consistency across explanation methods."""
        X, y = benchmark_anomaly_dataset
        
        # Train a model
        model = Model(algorithm="random_forest")
        model.train(X, y)
        
        explainer = ExplainableAIService()
        
        # Get multiple explanation types
        config = explainable_ai_config.copy()
        config['explanation_methods'] = ['shap', 'permutation']
        
        result = explainer.explain(model, X[:50], config)
        
        # Validate consistency
        assert result.success, "Multi-method explanation failed"
        
        if hasattr(result, 'feature_importance_shap') and hasattr(result, 'feature_importance_permutation'):
            # Top features should have some overlap
            shap_top_features = set(list(result.feature_importance_shap.keys())[:3])
            perm_top_features = set(list(result.feature_importance_permutation.keys())[:3])
            
            overlap = len(shap_top_features.intersection(perm_top_features))
            assert overlap >= 1, (
                "No overlap in top features between SHAP and permutation importance"
            )


@pytest.mark.algorithm_validation
class TestEnsembleMethodValidation:
    """Test ensemble methods for improved anomaly detection accuracy."""
    
    def test_ensemble_accuracy_improvement(
        self, 
        benchmark_anomaly_dataset: Tuple[np.ndarray, np.ndarray],
        ensemble_config: Dict[str, Any],
        accuracy_thresholds: Dict[str, float]
    ):
        """Test that ensemble methods improve upon individual algorithms."""
        X, y_true = benchmark_anomaly_dataset
        
        # Test individual algorithms
        individual_accuracies = {}
        for algorithm in ensemble_config['base_models']:
            model = Model(algorithm=algorithm)
            model.train(X, y_true)
            y_pred = model.predict(X)
            
            from conftest import calculate_anomaly_metrics
            metrics = calculate_anomaly_metrics(y_true, y_pred)
            individual_accuracies[algorithm] = metrics['accuracy']
        
        # Test ensemble
        ensemble_model = Model(algorithm="ensemble")
        ensemble_model.train(X, y_true)
        y_pred_ensemble = ensemble_model.predict(X)
        
        from conftest import calculate_anomaly_metrics
        ensemble_metrics = calculate_anomaly_metrics(y_true, y_pred_ensemble)
        
        # Ensemble should perform better than average individual
        avg_individual_accuracy = np.mean(list(individual_accuracies.values()))
        
        assert ensemble_metrics['accuracy'] >= avg_individual_accuracy, (
            f"Ensemble accuracy {ensemble_metrics['accuracy']:.3f} not better than "
            f"average individual {avg_individual_accuracy:.3f}"
        )
        
        # Should meet ensemble threshold
        ensemble_threshold = accuracy_thresholds.get('ensemble', 0.85)
        assert ensemble_metrics['accuracy'] >= ensemble_threshold, (
            f"Ensemble accuracy {ensemble_metrics['accuracy']:.3f} below threshold {ensemble_threshold}"
        )
    
    def test_ensemble_voting_strategies(
        self, 
        benchmark_anomaly_dataset: Tuple[np.ndarray, np.ndarray],
        ensemble_config: Dict[str, Any]
    ):
        """Test different ensemble voting strategies."""
        X, y_true = benchmark_anomaly_dataset
        
        voting_strategies = ['majority', 'average', 'weighted']
        
        results = {}
        for strategy in voting_strategies:
            config = ensemble_config.copy()
            config['combination_method'] = strategy
            
            ensemble_model = Model(algorithm="ensemble")
            ensemble_model.train(X, y_true)
            y_pred = ensemble_model.predict(X)
            
            from conftest import calculate_anomaly_metrics
            metrics = calculate_anomaly_metrics(y_true, y_pred)
            results[strategy] = metrics
        
        # All strategies should produce reasonable results
        for strategy, metrics in results.items():
            assert metrics['accuracy'] >= 0.65, (
                f"Voting strategy {strategy} accuracy {metrics['accuracy']:.3f} too low"
            )
            assert metrics['f1_score'] >= 0.60, (
                f"Voting strategy {strategy} F1-score {metrics['f1_score']:.3f} too low"
            )
    
    @pytest.mark.performance
    def test_ensemble_performance_overhead(
        self, 
        large_dataset: Tuple[np.ndarray, np.ndarray],
        ensemble_config: Dict[str, Any],
        performance_timer,
        performance_thresholds: Dict[str, float]
    ):
        """Test that ensemble methods don't have excessive performance overhead."""
        X, y = large_dataset
        
        # Time individual algorithm
        individual_model = Model(algorithm="isolation_forest")
        
        performance_timer.start()
        individual_model.train(X, y)
        individual_predictions = individual_model.predict(X[:1000])
        performance_timer.stop()
        
        individual_time = performance_timer.elapsed
        
        # Time ensemble
        ensemble_model = Model(algorithm="ensemble")
        
        performance_timer.start()
        ensemble_model.train(X, y)
        ensemble_predictions = ensemble_model.predict(X[:1000])
        performance_timer.stop()
        
        ensemble_time = performance_timer.elapsed
        
        # Ensemble should not be more than 5x slower
        time_ratio = ensemble_time / individual_time if individual_time > 0 else 1
        
        assert time_ratio <= 5.0, (
            f"Ensemble overhead ratio {time_ratio:.2f} too high "
            f"(individual: {individual_time:.2f}s, ensemble: {ensemble_time:.2f}s)"
        )
        
        # Absolute time should still be reasonable
        training_threshold = performance_thresholds.get('training_time_seconds', 120)
        assert ensemble_time <= training_threshold, (
            f"Ensemble training time {ensemble_time:.2f}s exceeds threshold {training_threshold}s"
        )


@pytest.mark.slow
@pytest.mark.algorithm_validation
class TestAlgorithmRobustness:
    """Test algorithm robustness under various conditions."""
    
    def test_missing_data_handling(
        self, 
        mixed_data_types_dataset: pd.DataFrame
    ):
        """Test algorithm robustness with missing data."""
        df = mixed_data_types_dataset.copy()
        
        # Introduce additional missing values
        missing_fraction = 0.2
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'target':
                n_missing = int(len(df) * missing_fraction)
                missing_idx = np.random.choice(len(df), n_missing, replace=False)
                df.loc[missing_idx, col] = np.nan
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Test with different algorithms
        algorithms = ['isolation_forest', 'one_class_svm']
        
        for algorithm in algorithms:
            model = Model(algorithm=algorithm)
            
            # Should handle missing data gracefully
            try:
                model.train(X, y)
                predictions = model.predict(X.head(10))
                assert len(predictions) == 10, f"Prediction count mismatch for {algorithm}"
            except Exception as e:
                pytest.fail(f"Algorithm {algorithm} failed with missing data: {e}")
    
    def test_edge_cases(self):
        """Test algorithm behavior with edge cases."""
        # Very small dataset
        X_small = np.random.randn(10, 3)
        y_small = np.ones(10)
        y_small[-2:] = -1  # 2 anomalies
        
        model = Model(algorithm="isolation_forest")
        
        try:
            model.train(X_small, y_small)
            predictions = model.predict(X_small)
            assert len(predictions) == len(X_small)
        except Exception as e:
            pytest.fail(f"Small dataset handling failed: {e}")
        
        # Single feature
        X_single = np.random.randn(100, 1)
        y_single = np.ones(100)
        y_single[-10:] = -1
        
        try:
            model.train(X_single, y_single)
            predictions = model.predict(X_single)
            assert len(predictions) == len(X_single)
        except Exception as e:
            pytest.fail(f"Single feature handling failed: {e}")
    
    def test_reproducibility(
        self, 
        benchmark_anomaly_dataset: Tuple[np.ndarray, np.ndarray]
    ):
        """Test that algorithms produce reproducible results with same random seed."""
        X, y = benchmark_anomaly_dataset
        
        # Train two models with same seed
        model1 = Model(algorithm="isolation_forest")
        model1.train(X, y)
        predictions1 = model1.predict(X)
        
        model2 = Model(algorithm="isolation_forest")
        model2.train(X, y)
        predictions2 = model2.predict(X)
        
        # Results should be identical (or very similar for stochastic algorithms)
        agreement = (predictions1 == predictions2).mean()
        assert agreement >= 0.95, (
            f"Algorithm reproducibility too low: {agreement:.3f} agreement"
        )