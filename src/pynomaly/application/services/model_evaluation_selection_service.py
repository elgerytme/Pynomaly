"""
Model Evaluation and Selection Service for Anomaly Detection

Comprehensive model evaluation and selection system providing advanced metrics,
statistical analysis, and intelligent model selection specifically designed for
anomaly detection use cases.

This addresses Issue #143: Phase 2.2: Data Science Package - Machine Learning Pipeline Framework
Component 4: Model Evaluation and Selection System
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    log_loss,
)
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    StratifiedKFold,
    TimeSeriesSplit,
    validation_curve,
    learning_curve,
)

# Optional dependencies with graceful fallback
try:
    import scipy.stats as stats
    from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from pynomaly.domain.entities.pipeline import Pipeline, PipelineStep
from pynomaly.domain.value_objects.performance_metrics import PerformanceMetrics
from pynomaly.infrastructure.logging.structured_logger import StructuredLogger


class EvaluationStrategy(Enum):
    """Model evaluation strategies for different scenarios."""
    
    STANDARD = "standard"
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES = "time_series"
    HOLDOUT = "holdout"
    BOOTSTRAP = "bootstrap"
    MONTE_CARLO = "monte_carlo"


class ModelSelectionCriteria(Enum):
    """Model selection criteria for anomaly detection."""
    
    ROC_AUC = "roc_auc"
    PRECISION_RECALL_AUC = "precision_recall_auc"
    F1_SCORE = "f1_score"
    BALANCED_ACCURACY = "balanced_accuracy"
    MATTHEWS_CORRELATION = "matthews_correlation"
    CUSTOM_WEIGHTED = "custom_weighted"
    ENSEMBLE_VOTING = "ensemble_voting"


class AnomalyDetectionMetric(Enum):
    """Specialized metrics for anomaly detection evaluation."""
    
    # Standard Classification Metrics
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    
    # ROC and Precision-Recall Metrics
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    AVERAGE_PRECISION = "average_precision"
    
    # Anomaly-Specific Metrics
    FALSE_POSITIVE_RATE = "false_positive_rate"
    FALSE_NEGATIVE_RATE = "false_negative_rate"
    DETECTION_RATE = "detection_rate"
    CONTAMINATION_ACCURACY = "contamination_accuracy"
    
    # Advanced Metrics
    MATTHEWS_CORRELATION = "matthews_correlation"
    COHEN_KAPPA = "cohen_kappa"
    LOG_LOSS = "log_loss"
    BRIER_SCORE = "brier_score"


@dataclass
class EvaluationConfiguration:
    """Configuration for model evaluation."""
    
    strategy: EvaluationStrategy = EvaluationStrategy.CROSS_VALIDATION
    metrics: List[AnomalyDetectionMetric] = field(default_factory=lambda: [
        AnomalyDetectionMetric.ROC_AUC,
        AnomalyDetectionMetric.PR_AUC,
        AnomalyDetectionMetric.F1_SCORE,
        AnomalyDetectionMetric.PRECISION,
        AnomalyDetectionMetric.RECALL
    ])
    
    # Cross-validation parameters
    cv_folds: int = 5
    cv_shuffle: bool = True
    cv_random_state: int = 42
    
    # Time series parameters
    time_series_splits: int = 5
    time_series_test_size: Optional[int] = None
    
    # Bootstrap parameters
    n_bootstrap_samples: int = 1000
    bootstrap_sample_size: Optional[int] = None
    bootstrap_confidence_level: float = 0.95
    
    # Statistical testing
    significance_level: float = 0.05
    multiple_testing_correction: str = "bonferroni"  # "bonferroni", "fdr_bh", "none"
    
    # Performance thresholds
    min_precision: float = 0.7
    min_recall: float = 0.7
    min_f1_score: float = 0.7
    min_roc_auc: float = 0.8


@dataclass
class ModelEvaluationResult:
    """Comprehensive model evaluation results."""
    
    model_id: str
    model_name: str
    evaluation_id: str
    evaluation_timestamp: datetime
    
    # Core metrics
    metrics: Dict[str, float]
    metric_confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Cross-validation results
    cv_scores: Optional[Dict[str, List[float]]] = None
    cv_mean_scores: Optional[Dict[str, float]] = None
    cv_std_scores: Optional[Dict[str, float]] = None
    
    # Confusion matrix and classification report
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict[str, Any]] = None
    
    # ROC and Precision-Recall curves
    roc_curve_data: Optional[Dict[str, np.ndarray]] = None
    pr_curve_data: Optional[Dict[str, np.ndarray]] = None
    
    # Learning and validation curves
    learning_curve_data: Optional[Dict[str, Any]] = None
    validation_curve_data: Optional[Dict[str, Any]] = None
    
    # Performance metadata
    evaluation_time: float = 0.0
    data_characteristics: Optional[Dict[str, Any]] = None
    model_complexity: Optional[Dict[str, Any]] = None
    
    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ModelComparisonResult:
    """Results of comparing multiple models."""
    
    comparison_id: str
    comparison_timestamp: datetime
    models: List[str]
    
    # Ranking and selection
    ranking: List[Tuple[str, float]]  # (model_id, overall_score)
    selected_model: str
    selection_criteria: ModelSelectionCriteria
    selection_confidence: float
    
    # Statistical significance tests
    pairwise_comparisons: Dict[str, Dict[str, Any]]
    statistical_significance: Dict[str, bool]
    
    # Performance summary
    performance_summary: Dict[str, Dict[str, float]]
    metric_distributions: Dict[str, Dict[str, List[float]]]
    
    # Recommendations
    recommendations: List[str]
    ensemble_candidates: List[str]


class ModelEvaluationSelectionService:
    """Advanced model evaluation and selection service for anomaly detection."""
    
    def __init__(
        self,
        results_dir: str = "evaluation_results",
        cache_enabled: bool = True,
        parallel_jobs: int = -1
    ):
        self.logger = StructuredLogger("model_evaluation")
        
        # Storage configuration
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Performance configuration
        self.cache_enabled = cache_enabled
        self.parallel_jobs = parallel_jobs
        
        # Results cache
        self._evaluation_cache: Dict[str, ModelEvaluationResult] = {}
        self._comparison_cache: Dict[str, ModelComparisonResult] = {}
        
        self.logger.info("Model evaluation and selection service initialized")
    
    async def evaluate_model(
        self,
        model: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model_name: str,
        config: Optional[EvaluationConfiguration] = None
    ) -> ModelEvaluationResult:
        """Comprehensive model evaluation for anomaly detection."""
        
        if config is None:
            config = EvaluationConfiguration()
        
        model_id = f"{model_name}_{int(time.time())}"
        evaluation_id = f"eval_{model_id}"
        
        self.logger.info(f"Starting model evaluation: {model_name}")
        start_time = time.time()
        
        try:
            # Convert inputs to numpy arrays
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            y_array = y.values if isinstance(y, pd.Series) else y
            
            # Data characteristics analysis
            data_characteristics = await self._analyze_data_characteristics(X_array, y_array)
            
            # Model complexity analysis
            model_complexity = await self._analyze_model_complexity(model)
            
            # Core evaluation based on strategy
            if config.strategy == EvaluationStrategy.CROSS_VALIDATION:
                evaluation_results = await self._cross_validation_evaluation(
                    model, X_array, y_array, config
                )
            elif config.strategy == EvaluationStrategy.TIME_SERIES:
                evaluation_results = await self._time_series_evaluation(
                    model, X_array, y_array, config
                )
            elif config.strategy == EvaluationStrategy.BOOTSTRAP:
                evaluation_results = await self._bootstrap_evaluation(
                    model, X_array, y_array, config
                )
            else:
                evaluation_results = await self._standard_evaluation(
                    model, X_array, y_array, config
                )
            
            # Generate additional analysis
            roc_curve_data = await self._generate_roc_curve(model, X_array, y_array)
            pr_curve_data = await self._generate_pr_curve(model, X_array, y_array)
            learning_curve_data = await self._generate_learning_curve(model, X_array, y_array)
            
            # Performance warnings and recommendations
            warnings, recommendations = await self._generate_performance_insights(
                evaluation_results["metrics"], data_characteristics, config
            )
            
            evaluation_time = time.time() - start_time
            
            # Create comprehensive result
            result = ModelEvaluationResult(
                model_id=model_id,
                model_name=model_name,
                evaluation_id=evaluation_id,
                evaluation_timestamp=datetime.now(),
                metrics=evaluation_results["metrics"],
                metric_confidence_intervals=evaluation_results.get("confidence_intervals", {}),
                cv_scores=evaluation_results.get("cv_scores"),
                cv_mean_scores=evaluation_results.get("cv_mean_scores"),
                cv_std_scores=evaluation_results.get("cv_std_scores"),
                confusion_matrix=evaluation_results.get("confusion_matrix"),
                classification_report=evaluation_results.get("classification_report"),
                roc_curve_data=roc_curve_data,
                pr_curve_data=pr_curve_data,
                learning_curve_data=learning_curve_data,
                evaluation_time=evaluation_time,
                data_characteristics=data_characteristics,
                model_complexity=model_complexity,
                warnings=warnings,
                recommendations=recommendations
            )
            
            # Cache and save results
            if self.cache_enabled:
                self._evaluation_cache[evaluation_id] = result
            
            await self._save_evaluation_result(result)
            
            self.logger.info(
                f"Model evaluation completed: {model_name} "
                f"(ROC-AUC: {result.metrics.get('roc_auc', 0):.4f}, "
                f"Time: {evaluation_time:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed for {model_name}: {e}")
            raise
    
    async def compare_models(
        self,
        models: List[Tuple[BaseEstimator, str]],  # (model, name)
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        selection_criteria: ModelSelectionCriteria = ModelSelectionCriteria.ROC_AUC,
        config: Optional[EvaluationConfiguration] = None
    ) -> ModelComparisonResult:
        """Compare multiple models and select the best performer."""
        
        if config is None:
            config = EvaluationConfiguration()
        
        comparison_id = f"comparison_{int(time.time())}"
        self.logger.info(f"Starting model comparison with {len(models)} models")
        
        try:
            # Evaluate all models
            evaluation_results = []
            for model, name in models:
                result = await self.evaluate_model(model, X, y, name, config)
                evaluation_results.append(result)
            
            # Extract performance metrics
            performance_summary = {}
            metric_distributions = {}
            
            for result in evaluation_results:
                performance_summary[result.model_name] = result.metrics
                
                if result.cv_scores:
                    for metric, scores in result.cv_scores.items():
                        if metric not in metric_distributions:
                            metric_distributions[metric] = {}
                        metric_distributions[metric][result.model_name] = scores
            
            # Perform pairwise statistical comparisons
            pairwise_comparisons = await self._perform_pairwise_comparisons(
                evaluation_results, config.significance_level
            )
            
            # Model ranking and selection
            ranking, selected_model, selection_confidence = await self._rank_and_select_models(
                evaluation_results, selection_criteria
            )
            
            # Statistical significance assessment
            statistical_significance = await self._assess_statistical_significance(
                pairwise_comparisons, config.significance_level
            )
            
            # Generate recommendations
            recommendations = await self._generate_comparison_recommendations(
                evaluation_results, ranking, statistical_significance
            )
            
            # Identify ensemble candidates
            ensemble_candidates = await self._identify_ensemble_candidates(
                evaluation_results, ranking
            )
            
            result = ModelComparisonResult(
                comparison_id=comparison_id,
                comparison_timestamp=datetime.now(),
                models=[result.model_name for result in evaluation_results],
                ranking=ranking,
                selected_model=selected_model,
                selection_criteria=selection_criteria,
                selection_confidence=selection_confidence,
                pairwise_comparisons=pairwise_comparisons,
                statistical_significance=statistical_significance,
                performance_summary=performance_summary,
                metric_distributions=metric_distributions,
                recommendations=recommendations,
                ensemble_candidates=ensemble_candidates
            )
            
            # Cache and save results
            if self.cache_enabled:
                self._comparison_cache[comparison_id] = result
            
            await self._save_comparison_result(result)
            
            self.logger.info(
                f"Model comparison completed. Selected: {selected_model} "
                f"(confidence: {selection_confidence:.3f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            raise
    
    async def validate_model_performance(
        self,
        model: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        performance_thresholds: Optional[Dict[str, float]] = None,
        config: Optional[EvaluationConfiguration] = None
    ) -> Dict[str, Any]:
        """Validate model performance against predefined thresholds."""
        
        if config is None:
            config = EvaluationConfiguration()
        
        if performance_thresholds is None:
            performance_thresholds = {
                "min_roc_auc": config.min_roc_auc,
                "min_precision": config.min_precision,
                "min_recall": config.min_recall,
                "min_f1_score": config.min_f1_score,
            }
        
        self.logger.info("Validating model performance against thresholds")
        
        # Evaluate model
        evaluation_result = await self.evaluate_model(
            model, X, y, "validation_model", config
        )
        
        # Check performance against thresholds
        validation_results = {
            "passed": True,
            "violations": [],
            "metrics": evaluation_result.metrics,
            "thresholds": performance_thresholds,
        }
        
        for threshold_name, threshold_value in performance_thresholds.items():
            metric_name = threshold_name.replace("min_", "")
            actual_value = evaluation_result.metrics.get(metric_name, 0.0)
            
            if actual_value < threshold_value:
                validation_results["passed"] = False
                validation_results["violations"].append({
                    "metric": metric_name,
                    "actual": actual_value,
                    "threshold": threshold_value,
                    "difference": threshold_value - actual_value
                })
        
        # Generate validation summary
        if validation_results["passed"]:
            summary = "Model performance validation PASSED"
        else:
            violations = len(validation_results["violations"])
            summary = f"Model performance validation FAILED ({violations} violations)"
        
        validation_results["summary"] = summary
        
        self.logger.info(f"Performance validation completed: {summary}")
        return validation_results
    
    async def _cross_validation_evaluation(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        config: EvaluationConfiguration
    ) -> Dict[str, Any]:
        """Perform cross-validation evaluation."""
        
        cv = StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=config.cv_shuffle,
            random_state=config.cv_random_state
        )
        
        # Define scoring metrics
        scoring_metrics = {}
        for metric in config.metrics:
            if metric == AnomalyDetectionMetric.ROC_AUC:
                scoring_metrics["roc_auc"] = "roc_auc"
            elif metric == AnomalyDetectionMetric.PRECISION:
                scoring_metrics["precision"] = "precision_weighted"
            elif metric == AnomalyDetectionMetric.RECALL:
                scoring_metrics["recall"] = "recall_weighted"
            elif metric == AnomalyDetectionMetric.F1_SCORE:
                scoring_metrics["f1_score"] = "f1_weighted"
            elif metric == AnomalyDetectionMetric.ACCURACY:
                scoring_metrics["accuracy"] = "accuracy"
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring_metrics,
            n_jobs=self.parallel_jobs,
            return_train_score=True
        )
        
        # Calculate metrics and confidence intervals
        metrics = {}
        confidence_intervals = {}
        cv_scores = {}
        cv_mean_scores = {}
        cv_std_scores = {}
        
        for metric_name, scores in cv_results.items():
            if metric_name.startswith("test_"):
                metric_key = metric_name.replace("test_", "")
                cv_scores[metric_key] = scores.tolist()
                cv_mean_scores[metric_key] = np.mean(scores)
                cv_std_scores[metric_key] = np.std(scores)
                metrics[metric_key] = np.mean(scores)
                
                # Calculate confidence interval
                if len(scores) > 1:
                    confidence_interval = stats.t.interval(
                        0.95,
                        len(scores) - 1,
                        loc=np.mean(scores),
                        scale=stats.sem(scores)
                    ) if SCIPY_AVAILABLE else (np.mean(scores) - np.std(scores), np.mean(scores) + np.std(scores))
                    confidence_intervals[metric_key] = confidence_interval
        
        # Fit model on full data for additional metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Additional metrics not available in cross_validate
        try:
            if y_proba is not None:
                metrics["pr_auc"] = average_precision_score(y, y_proba)
            
            metrics["balanced_accuracy"] = balanced_accuracy_score(y, y_pred)
            metrics["matthews_correlation"] = matthews_corrcoef(y, y_pred)
            
            # Confusion matrix and classification report
            cm = confusion_matrix(y, y_pred)
            cr = classification_report(y, y_pred, output_dict=True)
            
        except Exception as e:
            self.logger.warning(f"Error calculating additional metrics: {e}")
            cm = None
            cr = None
        
        return {
            "metrics": metrics,
            "confidence_intervals": confidence_intervals,
            "cv_scores": cv_scores,
            "cv_mean_scores": cv_mean_scores,
            "cv_std_scores": cv_std_scores,
            "confusion_matrix": cm,
            "classification_report": cr
        }
    
    async def _time_series_evaluation(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        config: EvaluationConfiguration
    ) -> Dict[str, Any]:
        """Perform time series evaluation with temporal splits."""
        
        cv = TimeSeriesSplit(
            n_splits=config.time_series_splits,
            test_size=config.time_series_test_size
        )
        
        scores_by_metric = {metric.value: [] for metric in config.metrics}
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics for this fold
            for metric in config.metrics:
                try:
                    if metric == AnomalyDetectionMetric.ROC_AUC and y_proba is not None:
                        score = roc_auc_score(y_test, y_proba)
                    elif metric == AnomalyDetectionMetric.PRECISION:
                        score = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    elif metric == AnomalyDetectionMetric.RECALL:
                        score = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    elif metric == AnomalyDetectionMetric.F1_SCORE:
                        score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    elif metric == AnomalyDetectionMetric.ACCURACY:
                        score = accuracy_score(y_test, y_pred)
                    else:
                        continue
                    
                    scores_by_metric[metric.value].append(score)
                except Exception as e:
                    self.logger.warning(f"Error calculating {metric.value}: {e}")
                    scores_by_metric[metric.value].append(0.0)
        
        # Calculate final metrics and confidence intervals
        metrics = {}
        confidence_intervals = {}
        cv_scores = {}
        cv_mean_scores = {}
        cv_std_scores = {}
        
        for metric_name, scores in scores_by_metric.items():
            if scores:
                cv_scores[metric_name] = scores
                cv_mean_scores[metric_name] = np.mean(scores)
                cv_std_scores[metric_name] = np.std(scores)
                metrics[metric_name] = np.mean(scores)
                
                if len(scores) > 1 and SCIPY_AVAILABLE:
                    confidence_intervals[metric_name] = stats.t.interval(
                        0.95,
                        len(scores) - 1,
                        loc=np.mean(scores),
                        scale=stats.sem(scores)
                    )
        
        return {
            "metrics": metrics,
            "confidence_intervals": confidence_intervals,
            "cv_scores": cv_scores,
            "cv_mean_scores": cv_mean_scores,
            "cv_std_scores": cv_std_scores
        }
    
    async def _bootstrap_evaluation(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        config: EvaluationConfiguration
    ) -> Dict[str, Any]:
        """Perform bootstrap evaluation for robust confidence intervals."""
        
        n_samples = len(X)
        bootstrap_size = config.bootstrap_sample_size or n_samples
        
        scores_by_metric = {metric.value: [] for metric in config.metrics}
        
        for _ in range(config.n_bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Create out-of-bag test set
            oob_indices = np.setdiff1d(np.arange(n_samples), indices)
            if len(oob_indices) == 0:
                continue
            
            X_test = X[oob_indices]
            y_test = y[oob_indices]
            
            try:
                # Train and predict
                model.fit(X_bootstrap, y_bootstrap)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                for metric in config.metrics:
                    try:
                        if metric == AnomalyDetectionMetric.ROC_AUC and y_proba is not None:
                            score = roc_auc_score(y_test, y_proba)
                        elif metric == AnomalyDetectionMetric.PRECISION:
                            score = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        elif metric == AnomalyDetectionMetric.RECALL:
                            score = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        elif metric == AnomalyDetectionMetric.F1_SCORE:
                            score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        elif metric == AnomalyDetectionMetric.ACCURACY:
                            score = accuracy_score(y_test, y_pred)
                        else:
                            continue
                        
                        scores_by_metric[metric.value].append(score)
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        # Calculate metrics with bootstrap confidence intervals
        metrics = {}
        confidence_intervals = {}
        
        for metric_name, scores in scores_by_metric.items():
            if scores:
                metrics[metric_name] = np.mean(scores)
                
                # Bootstrap confidence interval
                alpha = 1 - config.bootstrap_confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                confidence_intervals[metric_name] = (
                    np.percentile(scores, lower_percentile),
                    np.percentile(scores, upper_percentile)
                )
        
        return {
            "metrics": metrics,
            "confidence_intervals": confidence_intervals,
            "bootstrap_scores": scores_by_metric
        }
    
    async def _standard_evaluation(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        config: EvaluationConfiguration
    ) -> Dict[str, Any]:
        """Perform standard holdout evaluation."""
        
        # Fit model and make predictions
        model.fit(X, y)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {}
        
        try:
            for metric in config.metrics:
                if metric == AnomalyDetectionMetric.ROC_AUC and y_proba is not None:
                    metrics["roc_auc"] = roc_auc_score(y, y_proba)
                elif metric == AnomalyDetectionMetric.PR_AUC and y_proba is not None:
                    metrics["pr_auc"] = average_precision_score(y, y_proba)
                elif metric == AnomalyDetectionMetric.PRECISION:
                    metrics["precision"] = precision_score(y, y_pred, average='weighted', zero_division=0)
                elif metric == AnomalyDetectionMetric.RECALL:
                    metrics["recall"] = recall_score(y, y_pred, average='weighted', zero_division=0)
                elif metric == AnomalyDetectionMetric.F1_SCORE:
                    metrics["f1_score"] = f1_score(y, y_pred, average='weighted', zero_division=0)
                elif metric == AnomalyDetectionMetric.ACCURACY:
                    metrics["accuracy"] = accuracy_score(y, y_pred)
                elif metric == AnomalyDetectionMetric.BALANCED_ACCURACY:
                    metrics["balanced_accuracy"] = balanced_accuracy_score(y, y_pred)
                elif metric == AnomalyDetectionMetric.MATTHEWS_CORRELATION:
                    metrics["matthews_correlation"] = matthews_corrcoef(y, y_pred)
            
            # Additional analysis
            cm = confusion_matrix(y, y_pred)
            cr = classification_report(y, y_pred, output_dict=True)
            
        except Exception as e:
            self.logger.error(f"Error in standard evaluation: {e}")
            cm = None
            cr = None
        
        return {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": cr
        }
    
    async def _analyze_data_characteristics(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze dataset characteristics for evaluation context."""
        
        characteristics = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "class_distribution": np.bincount(y).tolist(),
            "class_imbalance_ratio": len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else float('inf'),
            "feature_sparsity": np.mean(X == 0),
            "feature_statistics": {
                "mean_values": np.mean(X, axis=0).tolist(),
                "std_values": np.std(X, axis=0).tolist(),
                "min_values": np.min(X, axis=0).tolist(),
                "max_values": np.max(X, axis=0).tolist(),
            }
        }
        
        return characteristics
    
    async def _analyze_model_complexity(
        self,
        model: BaseEstimator
    ) -> Dict[str, Any]:
        """Analyze model complexity characteristics."""
        
        complexity = {
            "model_type": type(model).__name__,
            "model_module": type(model).__module__,
        }
        
        # Extract key parameters
        try:
            params = model.get_params()
            complexity["parameters"] = {k: str(v) for k, v in params.items()}
            complexity["n_parameters"] = len(params)
        except Exception:
            complexity["parameters"] = {}
            complexity["n_parameters"] = 0
        
        return complexity
    
    async def _generate_roc_curve(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        """Generate ROC curve data."""
        
        try:
            if hasattr(model, 'predict_proba'):
                model.fit(X, y)
                y_proba = model.predict_proba(X)[:, 1]
                fpr, tpr, thresholds = roc_curve(y, y_proba)
                
                return {
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds,
                    "auc": roc_auc_score(y, y_proba)
                }
        except Exception as e:
            self.logger.warning(f"Error generating ROC curve: {e}")
        
        return None
    
    async def _generate_pr_curve(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        """Generate Precision-Recall curve data."""
        
        try:
            if hasattr(model, 'predict_proba'):
                model.fit(X, y)
                y_proba = model.predict_proba(X)[:, 1]
                precision, recall, thresholds = precision_recall_curve(y, y_proba)
                
                return {
                    "precision": precision,
                    "recall": recall,
                    "thresholds": thresholds,
                    "auc": average_precision_score(y, y_proba)
                }
        except Exception as e:
            self.logger.warning(f"Error generating PR curve: {e}")
        
        return None
    
    async def _generate_learning_curve(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Generate learning curve data."""
        
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y,
                cv=3,
                n_jobs=self.parallel_jobs,
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            
            return {
                "train_sizes": train_sizes,
                "train_scores_mean": np.mean(train_scores, axis=1),
                "train_scores_std": np.std(train_scores, axis=1),
                "val_scores_mean": np.mean(val_scores, axis=1),
                "val_scores_std": np.std(val_scores, axis=1)
            }
        except Exception as e:
            self.logger.warning(f"Error generating learning curve: {e}")
        
        return None
    
    async def _generate_performance_insights(
        self,
        metrics: Dict[str, float],
        data_characteristics: Dict[str, Any],
        config: EvaluationConfiguration
    ) -> Tuple[List[str], List[str]]:
        """Generate performance warnings and recommendations."""
        
        warnings = []
        recommendations = []
        
        # Check against thresholds
        if metrics.get("roc_auc", 0) < config.min_roc_auc:
            warnings.append(f"ROC-AUC {metrics.get('roc_auc', 0):.3f} below threshold {config.min_roc_auc}")
            recommendations.append("Consider feature engineering or algorithm selection")
        
        if metrics.get("precision", 0) < config.min_precision:
            warnings.append(f"Precision {metrics.get('precision', 0):.3f} below threshold {config.min_precision}")
            recommendations.append("Consider adjusting decision threshold or improving specificity")
        
        if metrics.get("recall", 0) < config.min_recall:
            warnings.append(f"Recall {metrics.get('recall', 0):.3f} below threshold {config.min_recall}")
            recommendations.append("Consider improving sensitivity or addressing class imbalance")
        
        # Data-specific recommendations
        imbalance_ratio = data_characteristics.get("class_imbalance_ratio", 1)
        if imbalance_ratio > 10:
            warnings.append(f"High class imbalance (ratio: {imbalance_ratio:.1f})")
            recommendations.append("Consider resampling techniques or cost-sensitive learning")
        
        sparsity = data_characteristics.get("feature_sparsity", 0)
        if sparsity > 0.5:
            warnings.append(f"High feature sparsity ({sparsity:.2%})")
            recommendations.append("Consider sparse-aware algorithms or feature selection")
        
        return warnings, recommendations
    
    async def _perform_pairwise_comparisons(
        self,
        evaluation_results: List[ModelEvaluationResult],
        significance_level: float
    ) -> Dict[str, Dict[str, Any]]:
        """Perform statistical pairwise comparisons between models."""
        
        comparisons = {}
        
        for i, result1 in enumerate(evaluation_results):
            for j, result2 in enumerate(evaluation_results[i+1:], i+1):
                comparison_key = f"{result1.model_name}_vs_{result2.model_name}"
                
                # Compare ROC-AUC scores if available
                if (result1.cv_scores and result2.cv_scores and 
                    "roc_auc" in result1.cv_scores and "roc_auc" in result2.cv_scores):
                    
                    scores1 = result1.cv_scores["roc_auc"]
                    scores2 = result2.cv_scores["roc_auc"]
                    
                    # Perform statistical test
                    if SCIPY_AVAILABLE and len(scores1) > 1 and len(scores2) > 1:
                        try:
                            statistic, p_value = stats.ttest_rel(scores1, scores2)
                            effect_size = (np.mean(scores1) - np.mean(scores2)) / np.sqrt(
                                (np.var(scores1) + np.var(scores2)) / 2
                            )
                        except Exception:
                            statistic, p_value, effect_size = 0, 1.0, 0
                    else:
                        statistic, p_value, effect_size = 0, 1.0, 0
                    
                    comparisons[comparison_key] = {
                        "model1": result1.model_name,
                        "model2": result2.model_name,
                        "model1_mean": np.mean(scores1),
                        "model2_mean": np.mean(scores2),
                        "difference": np.mean(scores1) - np.mean(scores2),
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": p_value < significance_level,
                        "effect_size": effect_size
                    }
        
        return comparisons
    
    async def _rank_and_select_models(
        self,
        evaluation_results: List[ModelEvaluationResult],
        selection_criteria: ModelSelectionCriteria
    ) -> Tuple[List[Tuple[str, float]], str, float]:
        """Rank models and select the best performer."""
        
        model_scores = []
        
        for result in evaluation_results:
            if selection_criteria == ModelSelectionCriteria.ROC_AUC:
                score = result.metrics.get("roc_auc", 0)
            elif selection_criteria == ModelSelectionCriteria.PRECISION_RECALL_AUC:
                score = result.metrics.get("pr_auc", 0)
            elif selection_criteria == ModelSelectionCriteria.F1_SCORE:
                score = result.metrics.get("f1_score", 0)
            elif selection_criteria == ModelSelectionCriteria.BALANCED_ACCURACY:
                score = result.metrics.get("balanced_accuracy", 0)
            elif selection_criteria == ModelSelectionCriteria.MATTHEWS_CORRELATION:
                score = result.metrics.get("matthews_correlation", 0)
            elif selection_criteria == ModelSelectionCriteria.CUSTOM_WEIGHTED:
                # Weighted combination of metrics
                score = (
                    0.4 * result.metrics.get("roc_auc", 0) +
                    0.3 * result.metrics.get("f1_score", 0) +
                    0.2 * result.metrics.get("precision", 0) +
                    0.1 * result.metrics.get("recall", 0)
                )
            else:
                score = result.metrics.get("roc_auc", 0)
            
            model_scores.append((result.model_name, score))
        
        # Sort by score (descending)
        ranking = sorted(model_scores, key=lambda x: x[1], reverse=True)
        
        # Select best model
        selected_model = ranking[0][0]
        best_score = ranking[0][1]
        
        # Calculate selection confidence based on score gap
        if len(ranking) > 1:
            score_gap = best_score - ranking[1][1]
            selection_confidence = min(1.0, score_gap / 0.1)  # Normalize by 0.1 gap
        else:
            selection_confidence = 1.0
        
        return ranking, selected_model, selection_confidence
    
    async def _assess_statistical_significance(
        self,
        pairwise_comparisons: Dict[str, Dict[str, Any]],
        significance_level: float
    ) -> Dict[str, bool]:
        """Assess statistical significance of model differences."""
        
        significance_results = {}
        
        for comparison_key, comparison_data in pairwise_comparisons.items():
            significance_results[comparison_key] = comparison_data.get("significant", False)
        
        return significance_results
    
    async def _generate_comparison_recommendations(
        self,
        evaluation_results: List[ModelEvaluationResult],
        ranking: List[Tuple[str, float]],
        statistical_significance: Dict[str, bool]
    ) -> List[str]:
        """Generate recommendations based on model comparison."""
        
        recommendations = []
        
        best_model = ranking[0][0]
        recommendations.append(f"Selected model: {best_model} (score: {ranking[0][1]:.4f})")
        
        # Check for statistical significance
        significant_differences = sum(statistical_significance.values())
        if significant_differences == 0:
            recommendations.append("No statistically significant differences found between models")
            recommendations.append("Consider ensemble methods to combine model strengths")
        
        # Performance recommendations
        best_result = next(r for r in evaluation_results if r.model_name == best_model)
        if best_result.metrics.get("roc_auc", 0) < 0.8:
            recommendations.append("Consider additional feature engineering or different algorithms")
        
        if len(ranking) > 1 and ranking[0][1] - ranking[1][1] < 0.01:
            recommendations.append("Top models perform similarly - consider ensemble approach")
        
        return recommendations
    
    async def _identify_ensemble_candidates(
        self,
        evaluation_results: List[ModelEvaluationResult],
        ranking: List[Tuple[str, float]]
    ) -> List[str]:
        """Identify models suitable for ensemble methods."""
        
        # Select top performing models within certain threshold
        best_score = ranking[0][1]
        threshold = 0.05  # Within 5% of best performance
        
        ensemble_candidates = []
        for model_name, score in ranking:
            if score >= best_score - threshold:
                ensemble_candidates.append(model_name)
        
        return ensemble_candidates[:5]  # Limit to top 5 candidates
    
    async def _save_evaluation_result(self, result: ModelEvaluationResult):
        """Save evaluation result to disk."""
        
        result_path = self.results_dir / f"{result.evaluation_id}.json"
        
        # Convert result to serializable format
        result_data = {
            "model_id": result.model_id,
            "model_name": result.model_name,
            "evaluation_id": result.evaluation_id,
            "evaluation_timestamp": result.evaluation_timestamp.isoformat(),
            "metrics": result.metrics,
            "metric_confidence_intervals": {
                k: list(v) for k, v in result.metric_confidence_intervals.items()
            },
            "cv_scores": result.cv_scores,
            "cv_mean_scores": result.cv_mean_scores,
            "cv_std_scores": result.cv_std_scores,
            "confusion_matrix": result.confusion_matrix.tolist() if result.confusion_matrix is not None else None,
            "classification_report": result.classification_report,
            "evaluation_time": result.evaluation_time,
            "data_characteristics": result.data_characteristics,
            "model_complexity": result.model_complexity,
            "warnings": result.warnings,
            "recommendations": result.recommendations
        }
        
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    async def _save_comparison_result(self, result: ModelComparisonResult):
        """Save comparison result to disk."""
        
        result_path = self.results_dir / f"{result.comparison_id}.json"
        
        # Convert result to serializable format
        result_data = {
            "comparison_id": result.comparison_id,
            "comparison_timestamp": result.comparison_timestamp.isoformat(),
            "models": result.models,
            "ranking": result.ranking,
            "selected_model": result.selected_model,
            "selection_criteria": result.selection_criteria.value,
            "selection_confidence": result.selection_confidence,
            "pairwise_comparisons": result.pairwise_comparisons,
            "statistical_significance": result.statistical_significance,
            "performance_summary": result.performance_summary,
            "metric_distributions": result.metric_distributions,
            "recommendations": result.recommendations,
            "ensemble_candidates": result.ensemble_candidates
        }
        
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    async def get_evaluation_history(
        self,
        model_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get evaluation history with optional filtering."""
        
        history = []
        
        for evaluation_id, result in self._evaluation_cache.items():
            if model_name is None or result.model_name == model_name:
                history.append({
                    "evaluation_id": evaluation_id,
                    "model_name": result.model_name,
                    "timestamp": result.evaluation_timestamp.isoformat(),
                    "roc_auc": result.metrics.get("roc_auc", 0),
                    "f1_score": result.metrics.get("f1_score", 0),
                    "precision": result.metrics.get("precision", 0),
                    "recall": result.metrics.get("recall", 0)
                })
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        if limit:
            history = history[:limit]
        
        return history
    
    async def get_best_models(
        self,
        metric: str = "roc_auc",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top performing models by specified metric."""
        
        model_performances = []
        
        for result in self._evaluation_cache.values():
            if metric in result.metrics:
                model_performances.append({
                    "model_name": result.model_name,
                    "evaluation_id": result.evaluation_id,
                    "metric_value": result.metrics[metric],
                    "timestamp": result.evaluation_timestamp.isoformat(),
                    "all_metrics": result.metrics
                })
        
        # Sort by metric value (descending)
        model_performances.sort(key=lambda x: x["metric_value"], reverse=True)
        
        return model_performances[:top_k]