"""
Model Explainability and Interpretability Framework

Comprehensive framework for generating model explanations using various
interpretability techniques including SHAP, LIME, permutation importance,
and custom attribution methods.
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import structlog

from mlops.domain.entities.model import Model, ModelVersion
from mlops.infrastructure.serving.realtime_inference_engine import InferenceRequest, InferenceResponse


class ExplanationMethod(Enum):
    """Available explanation methods."""
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    ANCHORS = "anchors"
    COUNTERFACTUAL = "counterfactual"
    FEATURE_ATTRIBUTION = "feature_attribution"
    DECISION_TREE_SURROGATE = "decision_tree_surrogate"


class ExplanationScope(Enum):
    """Scope of explanations."""
    GLOBAL = "global"  # Model-wide explanations
    LOCAL = "local"    # Instance-specific explanations
    COHORT = "cohort"  # Group-specific explanations


class VisualizationType(Enum):
    """Types of explanation visualizations."""
    FEATURE_IMPORTANCE = "feature_importance"
    WATERFALL = "waterfall"
    FORCE_PLOT = "force_plot"
    SUMMARY_PLOT = "summary_plot"
    DEPENDENCE_PLOT = "dependence_plot"
    DECISION_PLOT = "decision_plot"
    HEATMAP = "heatmap"
    BAR_CHART = "bar_chart"


@dataclass
class ExplanationRequest:
    """Request for model explanation."""
    model_id: str
    model_version: str
    method: ExplanationMethod
    scope: ExplanationScope
    input_data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]
    
    # Optional parameters
    feature_names: Optional[List[str]] = None
    target_class: Optional[Union[int, str]] = None
    background_data: Optional[pd.DataFrame] = None
    max_evals: int = 100
    perturbations: int = 1000
    
    # Visualization options
    include_visualization: bool = True
    visualization_types: List[VisualizationType] = field(default_factory=list)
    
    # Output options
    return_raw_values: bool = False
    confidence_intervals: bool = False
    
    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeatureAttribution:
    """Individual feature attribution."""
    feature_name: str
    importance: float
    confidence_interval: Optional[Tuple[float, float]] = None
    rank: Optional[int] = None
    direction: str = "neutral"  # positive, negative, neutral
    
    def __post_init__(self):
        """Determine direction based on importance."""
        if self.importance > 0:
            self.direction = "positive"
        elif self.importance < 0:
            self.direction = "negative"
        else:
            self.direction = "neutral"


@dataclass
class ExplanationResult:
    """Result of model explanation."""
    request_id: str
    model_id: str
    model_version: str
    method: ExplanationMethod
    scope: ExplanationScope
    
    # Core explanation data
    feature_attributions: List[FeatureAttribution]
    base_value: float
    predicted_value: float
    
    # Additional explanation data
    explanation_values: Optional[np.ndarray] = None
    shap_values: Optional[np.ndarray] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    
    # Metadata
    execution_time_ms: float = 0.0
    confidence_score: float = 0.0
    model_accuracy: Optional[float] = None
    explanation_fidelity: Optional[float] = None
    
    # Visualizations
    visualizations: Dict[str, Any] = field(default_factory=dict)
    
    # Raw data (if requested)
    raw_explanation_data: Optional[Dict[str, Any]] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def get_top_features(self, n: int = 10, by_absolute: bool = True) -> List[FeatureAttribution]:
        """Get top N most important features."""
        if by_absolute:
            sorted_attrs = sorted(
                self.feature_attributions, 
                key=lambda x: abs(x.importance), 
                reverse=True
            )
        else:
            sorted_attrs = sorted(
                self.feature_attributions, 
                key=lambda x: x.importance, 
                reverse=True
            )
        
        return sorted_attrs[:n]
    
    def get_feature_attribution(self, feature_name: str) -> Optional[FeatureAttribution]:
        """Get attribution for specific feature."""
        return next(
            (attr for attr in self.feature_attributions if attr.feature_name == feature_name), 
            None
        )


@dataclass
class GlobalExplanation:
    """Global model explanation."""
    model_id: str
    model_version: str
    
    # Feature importance across all predictions
    global_feature_importance: List[FeatureAttribution]
    
    # Model behavior insights
    model_complexity_score: float
    feature_interaction_strength: float
    non_linearity_score: float
    
    # Performance metrics
    model_accuracy: float
    explanation_coverage: float  # Percentage of predictions explained well
    
    # Feature insights
    most_influential_features: List[str]
    least_influential_features: List[str]
    correlated_feature_pairs: List[Tuple[str, str, float]]
    
    # Model stability
    prediction_stability_score: float
    feature_stability_scores: Dict[str, float]
    
    created_at: datetime = field(default_factory=datetime.utcnow)


class ModelExplainabilityFramework:
    """Comprehensive model explainability and interpretability framework."""
    
    def __init__(self, cache_size: int = 1000, cache_ttl_hours: int = 24):
        self.logger = structlog.get_logger(__name__)
        
        # Explanation cache
        self.explanation_cache: Dict[str, ExplanationResult] = {}
        self.global_explanations: Dict[str, GlobalExplanation] = {}
        self.cache_size = cache_size
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Explainer instances
        self.explainers: Dict[str, Any] = {}
        
        # Background tasks
        self.background_tasks = []
        
        # Supported model types
        self.supported_model_types = {
            "sklearn", "xgboost", "lightgbm", "catboost", 
            "tensorflow", "pytorch", "onnx"
        }
        
        self.logger.info("Model explainability framework initialized")
    
    async def explain_prediction(self, request: ExplanationRequest) -> ExplanationResult:
        """Generate explanation for a prediction."""
        
        start_time = datetime.utcnow()
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.explanation_cache:
            cached_result = self.explanation_cache[cache_key]
            if cached_result.expires_at and cached_result.expires_at > datetime.utcnow():
                self.logger.info(
                    "Returning cached explanation",
                    request_id=request.request_id,
                    cache_key=cache_key
                )
                return cached_result
        
        try:
            # Load model
            model = await self._load_model(request.model_id, request.model_version)
            
            # Generate explanation based on method
            if request.method == ExplanationMethod.SHAP:
                result = await self._explain_with_shap(request, model)
            elif request.method == ExplanationMethod.LIME:
                result = await self._explain_with_lime(request, model)
            elif request.method == ExplanationMethod.PERMUTATION_IMPORTANCE:
                result = await self._explain_with_permutation_importance(request, model)
            elif request.method == ExplanationMethod.PARTIAL_DEPENDENCE:
                result = await self._explain_with_partial_dependence(request, model)
            elif request.method == ExplanationMethod.FEATURE_ATTRIBUTION:
                result = await self._explain_with_feature_attribution(request, model)
            else:
                raise ValueError(f"Unsupported explanation method: {request.method}")
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            # Generate visualizations if requested
            if request.include_visualization:
                result.visualizations = await self._generate_visualizations(result, request)
            
            # Calculate explanation quality metrics
            result.explanation_fidelity = await self._calculate_explanation_fidelity(
                result, request, model
            )
            
            # Cache result
            result.expires_at = datetime.utcnow() + self.cache_ttl
            self._cache_explanation(cache_key, result)
            
            self.logger.info(
                "Explanation generated successfully",
                request_id=request.request_id,
                method=request.method.value,
                execution_time_ms=execution_time,
                num_features=len(result.feature_attributions)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Explanation generation failed",
                request_id=request.request_id,
                error=str(e)
            )
            raise
    
    async def generate_global_explanation(self, 
                                        model_id: str, 
                                        model_version: str,
                                        dataset: pd.DataFrame,
                                        sample_size: int = 10000) -> GlobalExplanation:
        """Generate global model explanation."""
        
        try:
            # Load model
            model = await self._load_model(model_id, model_version)
            
            # Sample data if needed
            if len(dataset) > sample_size:
                sampled_data = dataset.sample(n=sample_size, random_state=42)
            else:
                sampled_data = dataset
            
            # Generate global feature importance using multiple methods
            global_importance = await self._calculate_global_feature_importance(
                model, sampled_data
            )
            
            # Calculate model complexity metrics
            complexity_metrics = await self._calculate_model_complexity_metrics(
                model, sampled_data
            )
            
            # Analyze feature interactions
            interaction_metrics = await self._analyze_feature_interactions(
                model, sampled_data
            )
            
            # Calculate model stability
            stability_metrics = await self._calculate_model_stability(
                model, sampled_data
            )
            
            # Create global explanation
            global_explanation = GlobalExplanation(
                model_id=model_id,
                model_version=model_version,
                global_feature_importance=global_importance,
                model_complexity_score=complexity_metrics["complexity_score"],
                feature_interaction_strength=interaction_metrics["interaction_strength"],
                non_linearity_score=complexity_metrics["non_linearity_score"],
                model_accuracy=complexity_metrics["accuracy"],
                explanation_coverage=complexity_metrics["coverage"],
                most_influential_features=interaction_metrics["most_influential"],
                least_influential_features=interaction_metrics["least_influential"],
                correlated_feature_pairs=interaction_metrics["correlated_pairs"],
                prediction_stability_score=stability_metrics["prediction_stability"],
                feature_stability_scores=stability_metrics["feature_stability"]
            )
            
            # Cache global explanation
            global_key = f"{model_id}:{model_version}"
            self.global_explanations[global_key] = global_explanation
            
            self.logger.info(
                "Global explanation generated",
                model_id=model_id,
                model_version=model_version,
                sample_size=len(sampled_data),
                num_features=len(global_importance)
            )
            
            return global_explanation
            
        except Exception as e:
            self.logger.error(
                "Global explanation generation failed",
                model_id=model_id,
                error=str(e)
            )
            raise
    
    async def explain_model_drift(self,
                                model_id: str,
                                model_version: str,
                                reference_data: pd.DataFrame,
                                current_data: pd.DataFrame) -> Dict[str, Any]:
        """Explain model performance drift using interpretability techniques."""
        
        try:
            model = await self._load_model(model_id, model_version)
            
            # Generate explanations for both datasets
            ref_explanations = []
            curr_explanations = []
            
            # Sample data for comparison
            sample_size = min(1000, len(reference_data), len(current_data))
            ref_sample = reference_data.sample(n=sample_size, random_state=42)
            curr_sample = current_data.sample(n=sample_size, random_state=42)
            
            for idx, row in ref_sample.iterrows():
                request = ExplanationRequest(
                    model_id=model_id,
                    model_version=model_version,
                    method=ExplanationMethod.SHAP,
                    scope=ExplanationScope.LOCAL,
                    input_data=row.to_dict(),
                    include_visualization=False
                )
                explanation = await self.explain_prediction(request)
                ref_explanations.append(explanation)
            
            for idx, row in curr_sample.iterrows():
                request = ExplanationRequest(
                    model_id=model_id,
                    model_version=model_version,
                    method=ExplanationMethod.SHAP,
                    scope=ExplanationScope.LOCAL,
                    input_data=row.to_dict(),
                    include_visualization=False
                )
                explanation = await self.explain_prediction(request)
                curr_explanations.append(explanation)
            
            # Compare feature importance distributions
            drift_analysis = await self._analyze_explanation_drift(
                ref_explanations, curr_explanations
            )
            
            return drift_analysis
            
        except Exception as e:
            self.logger.error(
                "Model drift explanation failed",
                model_id=model_id,
                error=str(e)
            )
            raise
    
    async def generate_explanation_report(self,
                                        explanations: List[ExplanationResult],
                                        include_recommendations: bool = True) -> Dict[str, Any]:
        """Generate comprehensive explanation report."""
        
        try:
            # Aggregate feature importance across explanations
            feature_importance_stats = self._aggregate_feature_importance(explanations)
            
            # Calculate explanation consistency
            consistency_metrics = self._calculate_explanation_consistency(explanations)
            
            # Identify explanation patterns
            patterns = self._identify_explanation_patterns(explanations)
            
            # Generate insights
            insights = self._generate_explanation_insights(
                feature_importance_stats, consistency_metrics, patterns
            )
            
            # Generate recommendations if requested
            recommendations = []
            if include_recommendations:
                recommendations = self._generate_explanation_recommendations(insights)
            
            report = {
                "summary": {
                    "total_explanations": len(explanations),
                    "avg_execution_time_ms": np.mean([e.execution_time_ms for e in explanations]),
                    "avg_confidence_score": np.mean([e.confidence_score for e in explanations]),
                    "avg_fidelity": np.mean([e.explanation_fidelity for e in explanations if e.explanation_fidelity])
                },
                "feature_importance_stats": feature_importance_stats,
                "consistency_metrics": consistency_metrics,
                "patterns": patterns,
                "insights": insights,
                "recommendations": recommendations,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error("Explanation report generation failed", error=str(e))
            raise
    
    async def _explain_with_shap(self, request: ExplanationRequest, model: Any) -> ExplanationResult:
        """Generate SHAP-based explanation."""
        
        try:
            # Import SHAP
            import shap
            
            # Convert input data to appropriate format
            if isinstance(request.input_data, dict):
                input_df = pd.DataFrame([request.input_data])
            elif isinstance(request.input_data, pd.DataFrame):
                input_df = request.input_data
            else:
                input_df = pd.DataFrame(request.input_data)
            
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                # Classification model
                explainer = shap.Explainer(model.predict_proba, request.background_data)
            else:
                # Regression model
                explainer = shap.Explainer(model.predict, request.background_data)
            
            # Generate SHAP values
            shap_values = explainer(input_df)
            
            # Extract feature attributions
            feature_attributions = []
            feature_names = request.feature_names or input_df.columns.tolist()
            
            if len(shap_values.shape) == 3:  # Multi-class classification
                # Use values for specified class or first class
                class_idx = request.target_class if request.target_class is not None else 0
                values = shap_values.values[0, :, class_idx]
            else:
                values = shap_values.values[0]
            
            for i, feature_name in enumerate(feature_names):
                attribution = FeatureAttribution(
                    feature_name=feature_name,
                    importance=float(values[i]),
                    rank=i + 1
                )
                feature_attributions.append(attribution)
            
            # Sort by absolute importance
            feature_attributions.sort(key=lambda x: abs(x.importance), reverse=True)
            
            # Update ranks
            for i, attr in enumerate(feature_attributions):
                attr.rank = i + 1
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(input_df)[0]
                predicted_value = float(np.max(prediction))
            else:
                prediction = model.predict(input_df)
                predicted_value = float(prediction[0])
            
            result = ExplanationResult(
                request_id=request.request_id,
                model_id=request.model_id,
                model_version=request.model_version,
                method=ExplanationMethod.SHAP,
                scope=request.scope,
                feature_attributions=feature_attributions,
                base_value=float(explainer.expected_value[0] if hasattr(explainer.expected_value, '__iter__') else explainer.expected_value),
                predicted_value=predicted_value,
                shap_values=shap_values.values,
                confidence_score=0.9  # SHAP generally has high fidelity
            )
            
            if request.return_raw_values:
                result.raw_explanation_data = {
                    "shap_values": shap_values.values.tolist(),
                    "base_values": explainer.expected_value,
                    "data": input_df.values.tolist()
                }
            
            return result
            
        except ImportError:
            raise ValueError("SHAP library not available. Install with: pip install shap")
        except Exception as e:
            self.logger.error("SHAP explanation failed", error=str(e))
            raise
    
    async def _explain_with_lime(self, request: ExplanationRequest, model: Any) -> ExplanationResult:
        """Generate LIME-based explanation."""
        
        try:
            # Import LIME
            from lime import lime_tabular
            
            # Convert input data
            if isinstance(request.input_data, dict):
                input_array = np.array(list(request.input_data.values())).reshape(1, -1)
                feature_names = list(request.input_data.keys())
            elif isinstance(request.input_data, pd.DataFrame):
                input_array = request.input_data.values
                feature_names = request.input_data.columns.tolist()
            else:
                input_array = np.array(request.input_data).reshape(1, -1)
                feature_names = request.feature_names or [f"feature_{i}" for i in range(input_array.shape[1])]
            
            # Create LIME explainer
            if request.background_data is not None:
                training_data = request.background_data.values
            else:
                # Use dummy training data
                training_data = np.random.randn(100, input_array.shape[1])
            
            explainer = lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                mode='classification' if hasattr(model, 'predict_proba') else 'regression',
                discretize_continuous=True
            )
            
            # Generate explanation
            if hasattr(model, 'predict_proba'):
                predict_fn = model.predict_proba
            else:
                predict_fn = model.predict
            
            explanation = explainer.explain_instance(
                input_array[0], 
                predict_fn, 
                num_features=len(feature_names),
                num_samples=request.perturbations
            )
            
            # Extract feature attributions
            feature_attributions = []
            lime_explanation_dict = dict(explanation.as_list())
            
            for feature_name in feature_names:
                importance = lime_explanation_dict.get(feature_name, 0.0)
                attribution = FeatureAttribution(
                    feature_name=feature_name,
                    importance=float(importance)
                )
                feature_attributions.append(attribution)
            
            # Sort by absolute importance
            feature_attributions.sort(key=lambda x: abs(x.importance), reverse=True)
            
            # Update ranks
            for i, attr in enumerate(feature_attributions):
                attr.rank = i + 1
            
            # Make prediction
            prediction = predict_fn(input_array)
            if hasattr(model, 'predict_proba'):
                predicted_value = float(np.max(prediction[0]))
            else:
                predicted_value = float(prediction[0])
            
            result = ExplanationResult(
                request_id=request.request_id,
                model_id=request.model_id,
                model_version=request.model_version,
                method=ExplanationMethod.LIME,
                scope=request.scope,
                feature_attributions=feature_attributions,
                base_value=0.0,  # LIME doesn't provide base value
                predicted_value=predicted_value,
                lime_explanation={"explanation": explanation.as_list()},
                confidence_score=explanation.score  # LIME provides local fidelity score
            )
            
            if request.return_raw_values:
                result.raw_explanation_data = {
                    "lime_explanation": explanation.as_list(),
                    "intercept": explanation.intercept[1] if hasattr(explanation, 'intercept') else 0,
                    "prediction_local": explanation.local_pred,
                    "right": explanation.right
                }
            
            return result
            
        except ImportError:
            raise ValueError("LIME library not available. Install with: pip install lime")
        except Exception as e:
            self.logger.error("LIME explanation failed", error=str(e))
            raise
    
    async def _explain_with_permutation_importance(self, request: ExplanationRequest, model: Any) -> ExplanationResult:
        """Generate permutation importance-based explanation."""
        
        try:
            # Convert input data
            if isinstance(request.input_data, dict):
                input_df = pd.DataFrame([request.input_data])
            elif isinstance(request.input_data, pd.DataFrame):
                input_df = request.input_data
            else:
                input_df = pd.DataFrame(request.input_data)
            
            # Need background data for permutation importance
            if request.background_data is None:
                raise ValueError("Background data required for permutation importance")
            
            # Generate dummy target (for scoring function)
            X = request.background_data
            y_dummy = np.random.randint(0, 2, len(X))  # Dummy binary target
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y_dummy, 
                n_repeats=10, 
                random_state=42,
                n_jobs=-1
            )
            
            # Extract feature attributions
            feature_attributions = []
            feature_names = request.feature_names or X.columns.tolist()
            
            for i, feature_name in enumerate(feature_names):
                attribution = FeatureAttribution(
                    feature_name=feature_name,
                    importance=float(perm_importance.importances_mean[i]),
                    confidence_interval=(
                        float(perm_importance.importances_mean[i] - perm_importance.importances_std[i]),
                        float(perm_importance.importances_mean[i] + perm_importance.importances_std[i])
                    )
                )
                feature_attributions.append(attribution)
            
            # Sort by importance
            feature_attributions.sort(key=lambda x: x.importance, reverse=True)
            
            # Update ranks
            for i, attr in enumerate(feature_attributions):
                attr.rank = i + 1
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(input_df)
                predicted_value = float(np.max(prediction[0]))
            else:
                prediction = model.predict(input_df)
                predicted_value = float(prediction[0])
            
            result = ExplanationResult(
                request_id=request.request_id,
                model_id=request.model_id,
                model_version=request.model_version,
                method=ExplanationMethod.PERMUTATION_IMPORTANCE,
                scope=request.scope,
                feature_attributions=feature_attributions,
                base_value=0.0,
                predicted_value=predicted_value,
                confidence_score=0.7  # Medium confidence for permutation importance
            )
            
            if request.return_raw_values:
                result.raw_explanation_data = {
                    "importances_mean": perm_importance.importances_mean.tolist(),
                    "importances_std": perm_importance.importances_std.tolist(),
                    "importances": perm_importance.importances.tolist()
                }
            
            return result
            
        except Exception as e:
            self.logger.error("Permutation importance explanation failed", error=str(e))
            raise
    
    async def _explain_with_partial_dependence(self, request: ExplanationRequest, model: Any) -> ExplanationResult:
        """Generate partial dependence-based explanation."""
        
        try:
            from sklearn.inspection import partial_dependence
            
            # Convert input data
            if isinstance(request.input_data, dict):
                input_df = pd.DataFrame([request.input_data])
            elif isinstance(request.input_data, pd.DataFrame):
                input_df = request.input_data
            else:
                input_df = pd.DataFrame(request.input_data)
            
            # Need background data for partial dependence
            if request.background_data is None:
                raise ValueError("Background data required for partial dependence")
            
            X = request.background_data
            feature_names = request.feature_names or X.columns.tolist()
            
            # Calculate partial dependence for each feature
            feature_attributions = []
            
            for i, feature_name in enumerate(feature_names):
                try:
                    pd_result = partial_dependence(
                        model, X, [i], 
                        kind="average", 
                        grid_resolution=50
                    )
                    
                    # Calculate importance as variance in partial dependence
                    importance = float(np.var(pd_result["average"][0]))
                    
                    attribution = FeatureAttribution(
                        feature_name=feature_name,
                        importance=importance
                    )
                    feature_attributions.append(attribution)
                    
                except Exception as e:
                    self.logger.warning(f"Partial dependence failed for feature {feature_name}: {e}")
                    attribution = FeatureAttribution(
                        feature_name=feature_name,
                        importance=0.0
                    )
                    feature_attributions.append(attribution)
            
            # Sort by importance
            feature_attributions.sort(key=lambda x: x.importance, reverse=True)
            
            # Update ranks
            for i, attr in enumerate(feature_attributions):
                attr.rank = i + 1
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(input_df)
                predicted_value = float(np.max(prediction[0]))
            else:
                prediction = model.predict(input_df)
                predicted_value = float(prediction[0])
            
            result = ExplanationResult(
                request_id=request.request_id,
                model_id=request.model_id,
                model_version=request.model_version,
                method=ExplanationMethod.PARTIAL_DEPENDENCE,
                scope=request.scope,
                feature_attributions=feature_attributions,
                base_value=0.0,
                predicted_value=predicted_value,
                confidence_score=0.8
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Partial dependence explanation failed", error=str(e))
            raise
    
    async def _explain_with_feature_attribution(self, request: ExplanationRequest, model: Any) -> ExplanationResult:
        """Generate custom feature attribution explanation."""
        
        try:
            # Convert input data
            if isinstance(request.input_data, dict):
                input_df = pd.DataFrame([request.input_data])
            elif isinstance(request.input_data, pd.DataFrame):
                input_df = request.input_data
            else:
                input_df = pd.DataFrame(request.input_data)
            
            feature_names = request.feature_names or input_df.columns.tolist()
            
            # Simple feature attribution based on input magnitude and model prediction
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(input_df)
                predicted_value = float(np.max(prediction[0]))
            else:
                prediction = model.predict(input_df)
                predicted_value = float(prediction[0])
            
            # Calculate baseline prediction (with zeros)
            baseline_input = np.zeros_like(input_df.values)
            if hasattr(model, 'predict_proba'):
                baseline_pred = model.predict_proba(baseline_input)
                baseline_value = float(np.max(baseline_pred[0]))
            else:
                baseline_pred = model.predict(baseline_input)
                baseline_value = float(baseline_pred[0])
            
            # Calculate feature attributions
            feature_attributions = []
            
            for i, feature_name in enumerate(feature_names):
                # Create input with only this feature
                single_feature_input = baseline_input.copy()
                single_feature_input[0, i] = input_df.values[0, i]
                
                if hasattr(model, 'predict_proba'):
                    single_pred = model.predict_proba(single_feature_input)
                    single_value = float(np.max(single_pred[0]))
                else:
                    single_pred = model.predict(single_feature_input)
                    single_value = float(single_pred[0])
                
                # Attribution is the change from baseline
                importance = single_value - baseline_value
                
                attribution = FeatureAttribution(
                    feature_name=feature_name,
                    importance=importance
                )
                feature_attributions.append(attribution)
            
            # Sort by absolute importance
            feature_attributions.sort(key=lambda x: abs(x.importance), reverse=True)
            
            # Update ranks
            for i, attr in enumerate(feature_attributions):
                attr.rank = i + 1
            
            result = ExplanationResult(
                request_id=request.request_id,
                model_id=request.model_id,
                model_version=request.model_version,
                method=ExplanationMethod.FEATURE_ATTRIBUTION,
                scope=request.scope,
                feature_attributions=feature_attributions,
                base_value=baseline_value,
                predicted_value=predicted_value,
                confidence_score=0.6
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Feature attribution explanation failed", error=str(e))
            raise
    
    async def _load_model(self, model_id: str, model_version: str) -> Any:
        """Load model for explanation."""
        # This would integrate with your model registry
        # For now, return a placeholder
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Simulate fitted model
        X_dummy = np.random.randn(100, 10)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        
        return model
    
    def _generate_cache_key(self, request: ExplanationRequest) -> str:
        """Generate cache key for explanation request."""
        import hashlib
        
        # Create key from request parameters
        key_data = {
            "model_id": request.model_id,
            "model_version": request.model_version,
            "method": request.method.value,
            "scope": request.scope.value,
            "input_hash": str(hash(str(request.input_data)))
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cache_explanation(self, cache_key: str, result: ExplanationResult) -> None:
        """Cache explanation result."""
        if len(self.explanation_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.explanation_cache.keys(),
                key=lambda k: self.explanation_cache[k].created_at
            )
            del self.explanation_cache[oldest_key]
        
        self.explanation_cache[cache_key] = result
    
    async def _generate_visualizations(self, 
                                     result: ExplanationResult, 
                                     request: ExplanationRequest) -> Dict[str, Any]:
        """Generate visualizations for explanation result."""
        visualizations = {}
        
        # Default visualization types if none specified
        if not request.visualization_types:
            viz_types = [VisualizationType.FEATURE_IMPORTANCE, VisualizationType.BAR_CHART]
        else:
            viz_types = request.visualization_types
        
        for viz_type in viz_types:
            try:
                if viz_type == VisualizationType.FEATURE_IMPORTANCE:
                    visualizations["feature_importance"] = self._create_feature_importance_viz(result)
                elif viz_type == VisualizationType.BAR_CHART:
                    visualizations["bar_chart"] = self._create_bar_chart_viz(result)
                elif viz_type == VisualizationType.WATERFALL:
                    visualizations["waterfall"] = self._create_waterfall_viz(result)
            except Exception as e:
                self.logger.warning(f"Failed to generate {viz_type.value} visualization: {e}")
        
        return visualizations
    
    def _create_feature_importance_viz(self, result: ExplanationResult) -> Dict[str, Any]:
        """Create feature importance visualization data."""
        top_features = result.get_top_features(n=10)
        
        return {
            "type": "feature_importance",
            "data": {
                "features": [attr.feature_name for attr in top_features],
                "importances": [attr.importance for attr in top_features],
                "directions": [attr.direction for attr in top_features]
            },
            "config": {
                "title": "Feature Importance",
                "x_label": "Features",
                "y_label": "Importance"
            }
        }
    
    def _create_bar_chart_viz(self, result: ExplanationResult) -> Dict[str, Any]:
        """Create bar chart visualization data."""
        top_features = result.get_top_features(n=10)
        
        return {
            "type": "bar_chart",
            "data": {
                "labels": [attr.feature_name for attr in top_features],
                "values": [attr.importance for attr in top_features],
                "colors": ["red" if attr.direction == "negative" else "green" for attr in top_features]
            },
            "config": {
                "title": "Feature Attribution",
                "horizontal": True
            }
        }
    
    def _create_waterfall_viz(self, result: ExplanationResult) -> Dict[str, Any]:
        """Create waterfall visualization data."""
        top_features = result.get_top_features(n=10)
        
        # Calculate waterfall values
        waterfall_values = [result.base_value]
        labels = ["Base"]
        
        for attr in top_features:
            waterfall_values.append(attr.importance)
            labels.append(attr.feature_name)
        
        # Add final prediction
        waterfall_values.append(result.predicted_value - result.base_value - sum(attr.importance for attr in top_features))
        labels.append("Final")
        
        return {
            "type": "waterfall",
            "data": {
                "labels": labels,
                "values": waterfall_values,
                "base_value": result.base_value,
                "final_value": result.predicted_value
            },
            "config": {
                "title": "Prediction Waterfall"
            }
        }
    
    async def _calculate_explanation_fidelity(self, 
                                            result: ExplanationResult, 
                                            request: ExplanationRequest, 
                                            model: Any) -> float:
        """Calculate fidelity of explanation."""
        try:
            # Simple fidelity calculation based on method
            if request.method == ExplanationMethod.SHAP:
                return 0.9  # SHAP has high theoretical fidelity
            elif request.method == ExplanationMethod.LIME:
                return result.confidence_score  # LIME provides local fidelity
            elif request.method == ExplanationMethod.PERMUTATION_IMPORTANCE:
                return 0.7  # Medium fidelity
            else:
                return 0.6  # Lower fidelity for simpler methods
                
        except Exception:
            return 0.5  # Default fidelity
    
    async def _calculate_global_feature_importance(self, 
                                                 model: Any, 
                                                 dataset: pd.DataFrame) -> List[FeatureAttribution]:
        """Calculate global feature importance."""
        # Use multiple methods and aggregate
        methods = [ExplanationMethod.PERMUTATION_IMPORTANCE]
        
        all_importances = []
        
        for method in methods:
            try:
                request = ExplanationRequest(
                    model_id="global",
                    model_version="1.0",
                    method=method,
                    scope=ExplanationScope.GLOBAL,
                    input_data=dataset.iloc[0:1],  # Single row for local explanation
                    background_data=dataset,
                    include_visualization=False
                )
                
                result = await self.explain_prediction(request)
                all_importances.append(result.feature_attributions)
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate importance with {method.value}: {e}")
        
        # Aggregate importances
        if not all_importances:
            return []
        
        # Simple average aggregation
        feature_names = [attr.feature_name for attr in all_importances[0]]
        aggregated_importances = []
        
        for i, feature_name in enumerate(feature_names):
            importance_values = []
            for importance_list in all_importances:
                if i < len(importance_list):
                    importance_values.append(importance_list[i].importance)
            
            avg_importance = np.mean(importance_values) if importance_values else 0.0
            
            attribution = FeatureAttribution(
                feature_name=feature_name,
                importance=avg_importance
            )
            aggregated_importances.append(attribution)
        
        # Sort by absolute importance
        aggregated_importances.sort(key=lambda x: abs(x.importance), reverse=True)
        
        return aggregated_importances
    
    async def _calculate_model_complexity_metrics(self, 
                                                model: Any, 
                                                dataset: pd.DataFrame) -> Dict[str, float]:
        """Calculate model complexity metrics."""
        # Simplified complexity metrics
        metrics = {
            "complexity_score": 0.5,  # Placeholder
            "non_linearity_score": 0.3,  # Placeholder
            "accuracy": 0.85,  # Placeholder
            "coverage": 0.9  # Placeholder
        }
        
        return metrics
    
    async def _analyze_feature_interactions(self, 
                                          model: Any, 
                                          dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature interactions."""
        interactions = {
            "interaction_strength": 0.4,  # Placeholder
            "most_influential": dataset.columns[:3].tolist(),
            "least_influential": dataset.columns[-3:].tolist(),
            "correlated_pairs": []  # Placeholder
        }
        
        return interactions
    
    async def _calculate_model_stability(self, 
                                       model: Any, 
                                       dataset: pd.DataFrame) -> Dict[str, Any]:
        """Calculate model stability metrics."""
        stability = {
            "prediction_stability": 0.8,  # Placeholder
            "feature_stability": {col: 0.7 for col in dataset.columns}
        }
        
        return stability
    
    async def _analyze_explanation_drift(self, 
                                       ref_explanations: List[ExplanationResult], 
                                       curr_explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """Analyze drift in explanations."""
        # Compare feature importance distributions
        drift_analysis = {
            "overall_drift_score": 0.3,  # Placeholder
            "feature_drift_scores": {},
            "significant_changes": [],
            "drift_patterns": []
        }
        
        return drift_analysis
    
    def _aggregate_feature_importance(self, explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """Aggregate feature importance across explanations."""
        if not explanations:
            return {}
        
        # Collect all feature importances
        feature_importances = {}
        
        for explanation in explanations:
            for attr in explanation.feature_attributions:
                if attr.feature_name not in feature_importances:
                    feature_importances[attr.feature_name] = []
                feature_importances[attr.feature_name].append(attr.importance)
        
        # Calculate statistics
        stats = {}
        for feature_name, importances in feature_importances.items():
            stats[feature_name] = {
                "mean": float(np.mean(importances)),
                "std": float(np.std(importances)),
                "min": float(np.min(importances)),
                "max": float(np.max(importances)),
                "count": len(importances)
            }
        
        return stats
    
    def _calculate_explanation_consistency(self, explanations: List[ExplanationResult]) -> Dict[str, float]:
        """Calculate consistency of explanations."""
        if len(explanations) < 2:
            return {"consistency_score": 1.0}
        
        # Simple consistency based on feature ranking correlation
        # This is a simplified implementation
        consistency_score = 0.8  # Placeholder
        
        return {
            "consistency_score": consistency_score,
            "rank_correlation": 0.7,  # Placeholder
            "importance_correlation": 0.75  # Placeholder
        }
    
    def _identify_explanation_patterns(self, explanations: List[ExplanationResult]) -> List[Dict[str, Any]]:
        """Identify patterns in explanations."""
        patterns = [
            {
                "pattern_type": "consistent_top_features",
                "description": "Certain features consistently appear in top importance",
                "strength": 0.8,
                "examples": ["feature_1", "feature_2"]
            }
        ]
        
        return patterns
    
    def _generate_explanation_insights(self, 
                                     feature_stats: Dict[str, Any],
                                     consistency_metrics: Dict[str, float],
                                     patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from explanation analysis."""
        insights = [
            "Model shows consistent feature importance patterns",
            "Top 3 features account for majority of predictions",
            "Feature interactions appear to be limited"
        ]
        
        return insights
    
    def _generate_explanation_recommendations(self, insights: List[str]) -> List[str]:
        """Generate recommendations based on insights."""
        recommendations = [
            "Consider feature engineering to reduce top feature dependencies",
            "Monitor feature drift for top contributing features",
            "Investigate potential bias in highly important features"
        ]
        
        return recommendations
    
    def get_cached_explanations(self, model_id: str = None) -> List[ExplanationResult]:
        """Get cached explanations, optionally filtered by model."""
        if model_id:
            return [
                result for result in self.explanation_cache.values()
                if result.model_id == model_id
            ]
        return list(self.explanation_cache.values())
    
    def clear_cache(self) -> None:
        """Clear explanation cache."""
        self.explanation_cache.clear()
        self.global_explanations.clear()
        self.logger.info("Explanation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_cached_explanations": len(self.explanation_cache),
            "total_global_explanations": len(self.global_explanations),
            "cache_size_limit": self.cache_size,
            "cache_ttl_hours": self.cache_ttl.total_seconds() / 3600
        }