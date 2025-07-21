"""Model Validation Service for comprehensive model evaluation and validation."""

from __future__ import annotations

from typing import Any, Optional, Union, Tuple
import logging

# TODO: Implement within data platform science domain - from packages.data_science.domain.value_objects.ml_model_metrics import MLModelMetrics
# TODO: Implement within data platform science domain - from packages.data_science.domain.entities.experiment import Experiment


logger = logging.getLogger(__name__)


class ModelValidationService:
    """Domain service for model validation and evaluation operations.
    
    This service provides comprehensive model validation capabilities
    including cross-validation, holdout validation, bootstrap validation,
    and statistical significance testing.
    """
    
    def __init__(self) -> None:
        """Initialize the model validation service."""
        self._logger = logger
    
    def perform_cross_validation(self, model: Any, X: Any, y: Any,
                                cv_folds: int = 5,
                                scoring_metrics: Optional[list[str]] = None,
                                stratified: bool = True,
                                shuffle: bool = True,
                                random_state: int = 42) -> dict[str, Any]:
        """Perform cross-validation on a model.
        
        Args:
            model: ML model to validate
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            scoring_metrics: List of scoring metrics to evaluate
            stratified: Whether to use stratified splits
            shuffle: Whether to shuffle data before splitting
            random_state: Random state for reproducibility
            
        Returns:
            Cross-validation results with detailed metrics
        """
        try:
            import numpy as np
            from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
            from sklearn.metrics import get_scorer
            
            if scoring_metrics is None:
                scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            
            # Choose cross-validation strategy
            if stratified and self._is_classification_target(y):
                cv_strategy = StratifiedKFold(
                    n_splits=cv_folds, 
                    shuffle=shuffle, 
                    random_state=random_state
                )
            else:
                cv_strategy = KFold(
                    n_splits=cv_folds, 
                    shuffle=shuffle, 
                    random_state=random_state
                )
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y,
                cv=cv_strategy,
                scoring=scoring_metrics,
                return_train_score=True,
                return_estimator=True,
                n_jobs=-1
            )
            
            # Process results
            processed_results = {
                "cv_folds": cv_folds,
                "scoring_metrics": scoring_metrics,
                "cv_strategy": str(cv_strategy),
                "fold_results": {},
                "summary_statistics": {},
                "trained_models": cv_results.get('estimator', [])
            }
            
            # Calculate statistics for each metric
            for metric in scoring_metrics:
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                processed_results["fold_results"][metric] = {
                    "test_scores": test_scores.tolist(),
                    "train_scores": train_scores.tolist(),
                    "fold_differences": (train_scores - test_scores).tolist()
                }
                
                processed_results["summary_statistics"][metric] = {
                    "test_mean": float(np.mean(test_scores)),
                    "test_std": float(np.std(test_scores)),
                    "test_min": float(np.min(test_scores)),
                    "test_max": float(np.max(test_scores)),
                    "train_mean": float(np.mean(train_scores)),
                    "train_std": float(np.std(train_scores)),
                    "overfitting_score": float(np.mean(train_scores - test_scores)),
                    "stability_score": float(1 - np.std(test_scores) / np.mean(test_scores)) if np.mean(test_scores) > 0 else 0,
                    "confidence_interval_95": self._calculate_confidence_interval(test_scores, 0.95)
                }
            
            # Calculate overall validation score
            primary_metric = scoring_metrics[0]
            processed_results["overall_score"] = processed_results["summary_statistics"][primary_metric]["test_mean"]
            processed_results["overall_stability"] = processed_results["summary_statistics"][primary_metric]["stability_score"]
            
            # Add timing information
            processed_results["fit_time"] = {
                "mean": float(np.mean(cv_results['fit_time'])),
                "std": float(np.std(cv_results['fit_time'])),
                "total": float(np.sum(cv_results['fit_time']))
            }
            
            processed_results["score_time"] = {
                "mean": float(np.mean(cv_results['score_time'])),
                "std": float(np.std(cv_results['score_time'])),
                "total": float(np.sum(cv_results['score_time']))
            }
            
            return processed_results
            
        except ImportError:
            raise ImportError("scikit-learn is required for cross-validation")
        except Exception as e:
            self._logger.error(f"Cross-validation failed: {e}")
            raise
    
    def perform_holdout_validation(self, model: Any, X: Any, y: Any,
                                 test_size: float = 0.2,
                                 validation_size: float = 0.2,
                                 stratify: bool = True,
                                 random_state: int = 42) -> dict[str, Any]:
        """Perform holdout validation with train/validation/test splits.
        
        Args:
            model: ML model to validate
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            validation_size: Proportion of remaining data for validation
            stratify: Whether to stratify splits
            random_state: Random state for reproducibility
            
        Returns:
            Holdout validation results
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, confusion_matrix
            import numpy as np
            
            # Split data into train and test
            stratify_y = y if stratify and self._is_classification_target(y) else None
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, stratify=stratify_y, random_state=random_state
            )
            
            # Split remaining data into train and validation
            if validation_size > 0:
                stratify_temp = y_temp if stratify and self._is_classification_target(y_temp) else None
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=validation_size, 
                    stratify=stratify_temp, random_state=random_state
                )
            else:
                X_train, y_train = X_temp, y_temp
                X_val, y_val = None, None
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            if X_val is not None:
                y_val_pred = model.predict(X_val)
            
            # Calculate metrics
            model_type = "classification" if self._is_classification_target(y) else "regression"
            
            train_metrics = MLModelMetrics.from_sklearn_metrics(
                y_train, y_train_pred, model_type, task_type="train_split"
            )
            
            test_metrics = MLModelMetrics.from_sklearn_metrics(
                y_test, y_test_pred, model_type, task_type="test_split"
            )
            
            results = {
                "model_type": model_type,
                "split_sizes": {
                    "train": len(X_train),
                    "test": len(X_test),
                    "validation": len(X_val) if X_val is not None else 0
                },
                "split_proportions": {
                    "train": len(X_train) / len(X),
                    "test": len(X_test) / len(X),
                    "validation": len(X_val) / len(X) if X_val is not None else 0
                },
                "metrics": {
                    "train": train_metrics,
                    "test": test_metrics
                },
                "predictions": {
                    "train": y_train_pred,
                    "test": y_test_pred
                }
            }
            
            # Add validation metrics if available
            if X_val is not None:
                val_metrics = MLModelMetrics.from_sklearn_metrics(
                    y_val, y_val_pred, model_type, task_type="validation_split"
                )
                results["metrics"]["validation"] = val_metrics
                results["predictions"]["validation"] = y_val_pred
            
            # Calculate overfitting indicators
            train_score = train_metrics.get_primary_metric()
            test_score = test_metrics.get_primary_metric()
            
            if train_score is not None and test_score is not None:
                results["overfitting_analysis"] = {
                    "train_score": train_score,
                    "test_score": test_score,
                    "overfitting_score": train_score - test_score,
                    "generalization_ratio": test_score / train_score if train_score > 0 else 0,
                    "is_overfitting": (train_score - test_score) > 0.1  # 10% threshold
                }
            
            return results
            
        except ImportError:
            raise ImportError("scikit-learn is required for holdout validation")
        except Exception as e:
            self._logger.error(f"Holdout validation failed: {e}")
            raise
    
    def perform_bootstrap_validation(self, model: Any, X: Any, y: Any,
                                   n_bootstraps: int = 100,
                                   sample_size: Optional[int] = None,
                                   random_state: int = 42) -> dict[str, Any]:
        """Perform bootstrap validation on a model.
        
        Args:
            model: ML model to validate
            X: Feature matrix
            y: Target vector
            n_bootstraps: Number of bootstrap samples
            sample_size: Size of each bootstrap sample
            random_state: Random state for reproducibility
            
        Returns:
            Bootstrap validation results
        """
        try:
            import numpy as np
            from sklearn.utils import resample
            from sklearn.base import clone
            
            if sample_size is None:
                sample_size = len(X)
            
            np.random.seed(random_state)
            
            model_type = "classification" if self._is_classification_target(y) else "regression"
            bootstrap_scores = []
            oob_scores = []
            
            for i in range(n_bootstraps):
                # Create bootstrap sample
                bootstrap_indices = np.random.choice(len(X), size=sample_size, replace=True)
                oob_indices = np.setdiff1d(np.arange(len(X)), bootstrap_indices)
                
                X_bootstrap = X[bootstrap_indices] if hasattr(X, '__getitem__') else X.iloc[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices] if hasattr(y, '__getitem__') else y.iloc[bootstrap_indices]
                
                # Train model on bootstrap sample
                model_clone = clone(model)
                model_clone.fit(X_bootstrap, y_bootstrap)
                
                # Evaluate on bootstrap sample
                y_bootstrap_pred = model_clone.predict(X_bootstrap)
                bootstrap_metrics = MLModelMetrics.from_sklearn_metrics(
                    y_bootstrap, y_bootstrap_pred, model_type, task_type="bootstrap"
                )
                bootstrap_scores.append(bootstrap_metrics.get_primary_metric())
                
                # Evaluate on out-of-bag samples if available
                if len(oob_indices) > 0:
                    X_oob = X[oob_indices] if hasattr(X, '__getitem__') else X.iloc[oob_indices]
                    y_oob = y[oob_indices] if hasattr(y, '__getitem__') else y.iloc[oob_indices]
                    
                    y_oob_pred = model_clone.predict(X_oob)
                    oob_metrics = MLModelMetrics.from_sklearn_metrics(
                        y_oob, y_oob_pred, model_type, task_type="out_of_bag"
                    )
                    oob_scores.append(oob_metrics.get_primary_metric())
            
            # Calculate statistics
            bootstrap_scores = np.array(bootstrap_scores)
            oob_scores = np.array(oob_scores) if oob_scores else np.array([])
            
            results = {
                "n_bootstraps": n_bootstraps,
                "sample_size": sample_size,
                "model_type": model_type,
                "bootstrap_scores": bootstrap_scores.tolist(),
                "bootstrap_statistics": {
                    "mean": float(np.mean(bootstrap_scores)),
                    "std": float(np.std(bootstrap_scores)),
                    "min": float(np.min(bootstrap_scores)),
                    "max": float(np.max(bootstrap_scores)),
                    "median": float(np.median(bootstrap_scores)),
                    "confidence_interval_95": self._calculate_confidence_interval(bootstrap_scores, 0.95),
                    "confidence_interval_99": self._calculate_confidence_interval(bootstrap_scores, 0.99)
                }
            }
            
            # Add OOB statistics if available
            if len(oob_scores) > 0:
                results["oob_scores"] = oob_scores.tolist()
                results["oob_statistics"] = {
                    "mean": float(np.mean(oob_scores)),
                    "std": float(np.std(oob_scores)),
                    "min": float(np.min(oob_scores)),
                    "max": float(np.max(oob_scores)),
                    "median": float(np.median(oob_scores)),
                    "confidence_interval_95": self._calculate_confidence_interval(oob_scores, 0.95)
                }
                
                # Calculate bias estimate
                results["bias_estimate"] = float(np.mean(bootstrap_scores) - np.mean(oob_scores))
            
            return results
            
        except ImportError:
            raise ImportError("scikit-learn is required for bootstrap validation")
        except Exception as e:
            self._logger.error(f"Bootstrap validation failed: {e}")
            raise
    
    def compare_models(self, models: dict[str, Any], X: Any, y: Any,
                      validation_method: str = "cross_validation",
                      significance_test: str = "paired_t_test",
                      **validation_kwargs) -> dict[str, Any]:
        """Compare multiple models using statistical significance testing.
        
        Args:
            models: Dictionary of model names and model objects
            X: Feature matrix
            y: Target vector
            validation_method: Validation method to use
            significance_test: Statistical test for comparison
            **validation_kwargs: Additional arguments for validation
            
        Returns:
            Model comparison results with significance testing
        """
        try:
            import numpy as np
            from scipy import stats
            
            # Validate models
            model_results = {}
            
            for model_name, model in models.items():
                try:
                    if validation_method == "cross_validation":
                        result = self.perform_cross_validation(model, X, y, **validation_kwargs)
                    elif validation_method == "holdout":
                        result = self.perform_holdout_validation(model, X, y, **validation_kwargs)
                    elif validation_method == "bootstrap":
                        result = self.perform_bootstrap_validation(model, X, y, **validation_kwargs)
                    else:
                        raise ValueError(f"Unsupported validation method: {validation_method}")
                    
                    model_results[model_name] = result
                    
                except Exception as e:
                    self._logger.error(f"Validation failed for model {model_name}: {e}")
                    continue
            
            if len(model_results) < 2:
                return {"error": "Need at least 2 models for comparison"}
            
            # Perform pairwise comparisons
            comparison_results = {
                "validation_method": validation_method,
                "significance_test": significance_test,
                "model_rankings": [],
                "pairwise_comparisons": [],
                "statistical_tests": []
            }
            
            # Extract scores for comparison
            model_scores = {}
            for model_name, result in model_results.items():
                if validation_method == "cross_validation":
                    primary_metric = result["scoring_metrics"][0]
                    scores = result["fold_results"][primary_metric]["test_scores"]
                elif validation_method == "bootstrap":
                    scores = result["bootstrap_scores"]
                else:
                    # For holdout, create single score
                    scores = [result["metrics"]["test"].get_primary_metric()]
                
                model_scores[model_name] = scores
            
            # Perform pairwise statistical tests
            model_names = list(model_scores.keys())
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    scores1, scores2 = model_scores[model1], model_scores[model2]
                    
                    # Perform significance test
                    if significance_test == "paired_t_test":
                        if len(scores1) == len(scores2):
                            statistic, p_value = stats.ttest_rel(scores1, scores2)
                            test_name = "Paired t-test"
                        else:
                            statistic, p_value = stats.ttest_ind(scores1, scores2)
                            test_name = "Independent t-test"
                    elif significance_test == "wilcoxon":
                        if len(scores1) == len(scores2):
                            statistic, p_value = stats.wilcoxon(scores1, scores2)
                            test_name = "Wilcoxon signed-rank test"
                        else:
                            statistic, p_value = stats.mannwhitneyu(scores1, scores2)
                            test_name = "Mann-Whitney U test"
                    else:
                        raise ValueError(f"Unsupported significance test: {significance_test}")
                    
                    # Determine winner
                    mean1, mean2 = np.mean(scores1), np.mean(scores2)
                    winner = model1 if mean1 > mean2 else model2
                    is_significant = p_value < 0.05
                    
                    comparison_results["pairwise_comparisons"].append({
                        "model1": model1,
                        "model2": model2,
                        "mean1": float(mean1),
                        "mean2": float(mean2),
                        "winner": winner,
                        "difference": float(abs(mean1 - mean2)),
                        "is_significant": is_significant,
                        "p_value": float(p_value),
                        "test_statistic": float(statistic),
                        "test_name": test_name
                    })
            
            # Create overall ranking
            mean_scores = [(name, np.mean(scores)) for name, scores in model_scores.items()]
            mean_scores.sort(key=lambda x: x[1], reverse=True)
            
            comparison_results["model_rankings"] = [
                {
                    "rank": i + 1,
                    "model": name,
                    "mean_score": float(score),
                    "std_score": float(np.std(model_scores[name])),
                    "confidence_interval": self._calculate_confidence_interval(model_scores[name], 0.95)
                }
                for i, (name, score) in enumerate(mean_scores)
            ]
            
            # Add detailed model results
            comparison_results["detailed_results"] = model_results
            
            return comparison_results
            
        except ImportError:
            raise ImportError("scipy is required for statistical significance testing")
        except Exception as e:
            self._logger.error(f"Model comparison failed: {e}")
            raise
    
    def validate_experiment(self, experiment: Experiment,
                          validation_strategy: str = "comprehensive") -> dict[str, Any]:
        """Validate a complete experiment with multiple validation strategies.
        
        Args:
            experiment: Experiment entity to validate
            validation_strategy: Validation strategy to use
            
        Returns:
            Comprehensive validation results
        """
        try:
            if not experiment.is_completed():
                raise ValueError("Experiment must be completed before validation")
            
            validation_results = {
                "experiment_id": str(experiment.experiment_id),
                "experiment_name": experiment.name,
                "validation_strategy": validation_strategy,
                "validation_timestamp": self._get_timestamp(),
                "validation_summary": {}
            }
            
            # Basic validation checks
            basic_checks = {
                "has_metrics": experiment.metrics is not None,
                "has_feature_importance": experiment.feature_importance is not None,
                "has_hyperparameters": bool(experiment.hyperparameters),
                "has_model_artifacts": bool(experiment.model_artifacts),
                "proper_duration": experiment.duration_seconds is not None and experiment.duration_seconds > 0
            }
            
            validation_results["basic_checks"] = basic_checks
            validation_results["basic_checks_passed"] = all(basic_checks.values())
            
            # Validate metrics consistency
            if experiment.metrics:
                metrics_validation = self._validate_metrics_consistency(experiment.metrics)
                validation_results["metrics_validation"] = metrics_validation
            
            # Validate feature importance consistency
            if experiment.feature_importance:
                feature_validation = self._validate_feature_importance_consistency(experiment.feature_importance)
                validation_results["feature_validation"] = feature_validation
            
            # Validate hyperparameter consistency
            if experiment.hyperparameters:
                hyperparameter_validation = self._validate_hyperparameters(experiment.hyperparameters)
                validation_results["hyperparameter_validation"] = hyperparameter_validation
            
            # Overall validation score
            validation_score = self._calculate_validation_score(validation_results)
            validation_results["validation_score"] = validation_score
            validation_results["is_valid"] = validation_score >= 0.8
            
            return validation_results
            
        except Exception as e:
            self._logger.error(f"Experiment validation failed: {e}")
            raise
    
    def _is_classification_target(self, y: Any) -> bool:
        """Check if target is classification or regression."""
        try:
            import numpy as np
            
            if hasattr(y, 'dtype'):
                # Check if it's integer or object dtype
                if y.dtype.kind in ['i', 'O', 'U', 'S']:
                    return True
                elif y.dtype.kind == 'f':
                    # Check if float values are actually discrete
                    unique_values = np.unique(y)
                    return len(unique_values) <= 20 and np.all(unique_values == unique_values.astype(int))
            
            # Fallback: check number of unique values
            unique_values = np.unique(y)
            return len(unique_values) <= 20
            
        except Exception:
            return False
    
    def _calculate_confidence_interval(self, scores: Any, confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for scores."""
        import numpy as np
        
        scores = np.array(scores)
        mean = np.mean(scores)
        std = np.std(scores)
        n = len(scores)
        
        # Use t-distribution for small samples
        if n < 30:
            from scipy.stats import t
            t_value = t.ppf((1 + confidence) / 2, n - 1)
        else:
            # Use normal distribution for large samples
            t_value = 1.96 if confidence == 0.95 else 2.576
        
        margin = t_value * (std / np.sqrt(n))
        
        return (float(mean - margin), float(mean + margin))
    
    def _validate_metrics_consistency(self, metrics: MLModelMetrics) -> dict[str, Any]:
        """Validate metrics for consistency and reasonableness."""
        validation_results = {
            "is_consistent": True,
            "issues": [],
            "warnings": []
        }
        
        # Check for reasonable metric values
        if metrics.is_classification_model():
            if metrics.accuracy is not None and not (0 <= metrics.accuracy <= 1):
                validation_results["issues"].append("Accuracy out of valid range [0,1]")
                validation_results["is_consistent"] = False
            
            if metrics.precision and any(not (0 <= p <= 1) for p in metrics.precision.values()):
                validation_results["issues"].append("Precision values out of valid range [0,1]")
                validation_results["is_consistent"] = False
        
        elif metrics.is_regression_model():
            if metrics.r2_score is not None and metrics.r2_score < -1:
                validation_results["warnings"].append("R² score is very low (< -1)")
            
            if metrics.mse is not None and metrics.mse < 0:
                validation_results["issues"].append("MSE cannot be negative")
                validation_results["is_consistent"] = False
        
        # Check for impossible combinations
        if metrics.cv_scores and metrics.cv_mean:
            import numpy as np
            calculated_mean = np.mean(metrics.cv_scores)
            if abs(calculated_mean - metrics.cv_mean) > 0.001:
                validation_results["issues"].append("CV mean doesn't match CV scores")
                validation_results["is_consistent"] = False
        
        return validation_results
    
    def _validate_feature_importance_consistency(self, feature_importance: FeatureImportance) -> dict[str, Any]:
        """Validate feature importance for consistency."""
        validation_results = {
            "is_consistent": True,
            "issues": [],
            "warnings": []
        }
        
        # Check if normalized flag matches actual values
        if feature_importance.normalized:
            max_score = max(feature_importance.feature_scores.values())
            if max_score > 1.01:  # Small tolerance
                validation_results["issues"].append("Normalized=True but max score > 1")
                validation_results["is_consistent"] = False
        
        # Check ranking consistency
        if feature_importance.ranking:
            expected_ranking = sorted(
                feature_importance.feature_scores.keys(),
                key=lambda x: feature_importance.feature_scores[x],
                reverse=True
            )
            
            if feature_importance.ranking != expected_ranking:
                validation_results["warnings"].append("Ranking doesn't match importance scores")
        
        return validation_results
    
    def _validate_hyperparameters(self, hyperparameters: dict[str, Any]) -> dict[str, Any]:
        """Validate hyperparameters for common issues."""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check for common hyperparameter issues
        if "learning_rate" in hyperparameters:
            lr = hyperparameters["learning_rate"]
            if lr <= 0 or lr > 1:
                validation_results["warnings"].append("Learning rate outside typical range (0,1]")
        
        if "n_estimators" in hyperparameters:
            n_est = hyperparameters["n_estimators"]
            if n_est <= 0:
                validation_results["issues"].append("n_estimators must be positive")
                validation_results["is_valid"] = False
        
        if "max_depth" in hyperparameters:
            max_depth = hyperparameters["max_depth"]
            if max_depth is not None and max_depth <= 0:
                validation_results["issues"].append("max_depth must be positive or None")
                validation_results["is_valid"] = False
        
        return validation_results
    
    def _calculate_validation_score(self, validation_results: dict[str, Any]) -> float:
        """Calculate overall validation score."""
        score = 0.0
        total_weight = 0.0
        
        # Basic checks (weight: 0.3)
        if validation_results.get("basic_checks_passed"):
            score += 0.3
        total_weight += 0.3
        
        # Metrics validation (weight: 0.4)
        if "metrics_validation" in validation_results:
            if validation_results["metrics_validation"].get("is_consistent"):
                score += 0.4
            total_weight += 0.4
        
        # Feature validation (weight: 0.2)
        if "feature_validation" in validation_results:
            if validation_results["feature_validation"].get("is_consistent"):
                score += 0.2
            total_weight += 0.2
        
        # Hyperparameter validation (weight: 0.1)
        if "hyperparameter_validation" in validation_results:
            if validation_results["hyperparameter_validation"].get("is_valid"):
                score += 0.1
            total_weight += 0.1
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()