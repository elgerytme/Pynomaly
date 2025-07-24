"""
Model Explainability and Interpretability Framework

Comprehensive framework for explaining ML model predictions using various
techniques including SHAP, LIME, feature importance, and custom explainers.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text, plot_tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import structlog

# Optional imports for advanced explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from mlops.domain.entities.model import Model


class ExplanationMethod(Enum):
    """Available explanation methods."""
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    FEATURE_IMPORTANCE = "feature_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    ICE_PLOTS = "ice_plots"
    ANCHOR = "anchor"
    COUNTERFACTUAL = "counterfactual"


class ExplanationScope(Enum):
    """Scope of explanation."""
    GLOBAL = "global"  # Model-level explanations
    LOCAL = "local"    # Instance-level explanations
    COHORT = "cohort"  # Group-level explanations


@dataclass
class ExplanationConfig:
    """Configuration for model explanations."""
    method: ExplanationMethod
    scope: ExplanationScope
    feature_names: List[str]
    target_features: Optional[List[str]] = None
    sample_size: int = 1000
    background_samples: int = 100
    confidence_level: float = 0.95
    
    # Method-specific parameters
    shap_explainer_type: str = "auto"  # "tree", "linear", "kernel", "auto"
    lime_mode: str = "tabular"  # "tabular", "text", "image"
    lime_num_features: int = 10
    
    # Advanced options
    enable_interactions: bool = False
    enable_clustering: bool = False
    generate_visualizations: bool = True
    export_format: str = "json"  # "json", "html", "pdf"


@dataclass
class ExplanationResult:
    """Result of model explanation."""
    explanation_id: str
    model_id: str
    method: ExplanationMethod
    scope: ExplanationScope
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Explanation data
    feature_importance: Dict[str, float] = field(default_factory=dict)
    shap_values: Optional[np.ndarray] = None
    lime_explanation: Optional[Dict] = None
    
    # Instance-specific (for local explanations)
    instance_id: Optional[str] = None
    prediction: Optional[float] = None
    actual_value: Optional[float] = None
    
    # Visualization data
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    execution_time_ms: float = 0.0
    confidence_score: float = 0.0
    explanation_quality: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "explanation_id": self.explanation_id,
            "model_id": self.model_id,
            "method": self.method.value,
            "scope": self.scope.value,
            "timestamp": self.timestamp.isoformat(),
            "feature_importance": self.feature_importance,
            "instance_id": self.instance_id,
            "prediction": self.prediction,
            "actual_value": self.actual_value,
            "visualization_data": self.visualization_data,
            "execution_time_ms": self.execution_time_ms,
            "confidence_score": self.confidence_score,
            "explanation_quality": self.explanation_quality
        }
        
        # Handle numpy arrays
        if self.shap_values is not None:
            result["shap_values"] = self.shap_values.tolist()
        
        return result


class BaseExplainer(ABC):
    """Base class for model explainers."""
    
    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    @abstractmethod
    async def explain_global(self, 
                           model: Any,
                           X: pd.DataFrame, 
                           y: Optional[pd.Series] = None) -> ExplanationResult:
        """Generate global explanations for the model."""
        pass
    
    @abstractmethod
    async def explain_local(self, 
                          model: Any,
                          X: pd.DataFrame,
                          instance_idx: int) -> ExplanationResult:
        """Generate local explanations for a specific instance."""
        pass
    
    def _create_base_result(self, model_id: str, scope: ExplanationScope) -> ExplanationResult:
        """Create a base explanation result."""
        return ExplanationResult(
            explanation_id=str(uuid.uuid4()),
            model_id=model_id,
            method=self.config.method,
            scope=scope
        )


class ShapExplainer(BaseExplainer):
    """SHAP-based model explainer."""
    
    def __init__(self, config: ExplanationConfig):
        super().__init__(config)
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        self.explainer = None
        self.background_data = None
    
    async def explain_global(self, 
                           model: Any,
                           X: pd.DataFrame, 
                           y: Optional[pd.Series] = None) -> ExplanationResult:
        """Generate global SHAP explanations."""
        
        start_time = datetime.utcnow()
        result = self._create_base_result(getattr(model, 'id', 'unknown'), ExplanationScope.GLOBAL)
        
        try:
            # Initialize explainer if needed
            if self.explainer is None:
                self.explainer = self._create_shap_explainer(model, X)
            
            # Sample data for global explanation
            sample_X = X.sample(min(self.config.sample_size, len(X)))
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(sample_X)
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first output for simplicity
            
            result.shap_values = shap_values
            
            # Calculate feature importance as mean absolute SHAP values
            feature_importance = {}
            for i, feature_name in enumerate(self.config.feature_names):
                if i < shap_values.shape[1]:
                    importance = np.mean(np.abs(shap_values[:, i]))
                    feature_importance[feature_name] = float(importance)
            
            result.feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            # Generate visualization data
            result.visualization_data = {
                "summary_plot_data": self._create_summary_plot_data(shap_values, sample_X),
                "feature_importance_plot": result.feature_importance,
                "dependence_plots": self._create_dependence_plot_data(
                    shap_values, sample_X, list(result.feature_importance.keys())[:5]
                )
            }
            
            result.confidence_score = self._calculate_explanation_confidence(shap_values)
            result.explanation_quality = self._assess_explanation_quality(result.confidence_score)
            
        except Exception as e:
            self.logger.error(f"SHAP global explanation failed: {str(e)}")
            result.explanation_quality = "failed"
        
        result.execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return result
    
    async def explain_local(self, 
                          model: Any,
                          X: pd.DataFrame,
                          instance_idx: int) -> ExplanationResult:
        """Generate local SHAP explanations for a specific instance."""
        
        start_time = datetime.utcnow()
        result = self._create_base_result(getattr(model, 'id', 'unknown'), ExplanationScope.LOCAL)
        result.instance_id = str(instance_idx)
        
        try:
            # Initialize explainer if needed
            if self.explainer is None:
                self.explainer = self._create_shap_explainer(model, X)
            
            # Get instance
            instance = X.iloc[instance_idx:instance_idx+1]
            
            # Calculate SHAP values for the instance
            shap_values = self.explainer.shap_values(instance)
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            result.shap_values = shap_values
            
            # Get prediction
            if hasattr(model, 'predict'):
                result.prediction = float(model.predict(instance)[0])
            
            # Calculate feature contributions
            feature_contributions = {}
            for i, feature_name in enumerate(self.config.feature_names):
                if i < shap_values.shape[1]:
                    contribution = float(shap_values[0, i])
                    feature_contributions[feature_name] = contribution
            
            result.feature_importance = dict(sorted(
                feature_contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            # Generate visualization data
            result.visualization_data = {
                "waterfall_data": self._create_waterfall_data(
                    shap_values[0], instance, result.prediction
                ),
                "force_plot_data": {
                    "base_value": float(self.explainer.expected_value),
                    "shap_values": shap_values[0].tolist(),
                    "feature_values": instance.iloc[0].to_dict(),
                    "prediction": result.prediction
                }
            }
            
            result.confidence_score = self._calculate_explanation_confidence(shap_values)
            result.explanation_quality = self._assess_explanation_quality(result.confidence_score)
            
        except Exception as e:
            self.logger.error(f"SHAP local explanation failed: {str(e)}")
            result.explanation_quality = "failed"
        
        result.execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return result
    
    def _create_shap_explainer(self, model: Any, X: pd.DataFrame) -> Any:
        """Create appropriate SHAP explainer based on model type."""
        
        # Select background data
        background_size = min(self.config.background_samples, len(X))
        self.background_data = X.sample(background_size)
        
        model_type = type(model).__name__.lower()
        
        if self.config.shap_explainer_type == "auto":
            # Auto-detect explainer type
            if any(tree_type in model_type for tree_type in ['forest', 'tree', 'xgb', 'lgb']):
                return shap.TreeExplainer(model)
            elif any(linear_type in model_type for linear_type in ['linear', 'logistic']):
                return shap.LinearExplainer(model, self.background_data)
            else:
                return shap.KernelExplainer(model.predict, self.background_data)
        
        elif self.config.shap_explainer_type == "tree":
            return shap.TreeExplainer(model)
        elif self.config.shap_explainer_type == "linear":
            return shap.LinearExplainer(model, self.background_data)
        elif self.config.shap_explainer_type == "kernel":
            return shap.KernelExplainer(model.predict, self.background_data)
        else:
            return shap.Explainer(model, self.background_data)
    
    def _create_summary_plot_data(self, shap_values: np.ndarray, X: pd.DataFrame) -> Dict[str, Any]:
        """Create data for SHAP summary plot."""
        return {
            "feature_names": self.config.feature_names[:shap_values.shape[1]],
            "shap_values": shap_values.tolist(),
            "feature_values": X.iloc[:shap_values.shape[0]].values.tolist()
        }
    
    def _create_dependence_plot_data(self, 
                                   shap_values: np.ndarray, 
                                   X: pd.DataFrame,
                                   top_features: List[str]) -> Dict[str, Any]:
        """Create data for SHAP dependence plots."""
        dependence_data = {}
        
        for feature in top_features:
            if feature in X.columns:
                feature_idx = list(X.columns).index(feature)
                if feature_idx < shap_values.shape[1]:
                    dependence_data[feature] = {
                        "feature_values": X[feature].tolist(),
                        "shap_values": shap_values[:, feature_idx].tolist()
                    }
        
        return dependence_data
    
    def _create_waterfall_data(self, 
                             shap_values: np.ndarray,
                             instance: pd.DataFrame,
                             prediction: float) -> Dict[str, Any]:
        """Create data for waterfall plot."""
        return {
            "base_value": float(self.explainer.expected_value),
            "contributions": [
                {
                    "feature": feature,
                    "value": float(instance[feature].iloc[0]),
                    "contribution": float(shap_values[i])
                }
                for i, feature in enumerate(self.config.feature_names[:len(shap_values)])
            ],
            "final_prediction": prediction
        }
    
    def _calculate_explanation_confidence(self, shap_values: np.ndarray) -> float:
        """Calculate confidence in the explanation."""
        if shap_values.size == 0:
            return 0.0
        
        # Use variance in SHAP values as a proxy for explanation stability
        variance = np.var(shap_values)
        max_variance = np.var(np.abs(shap_values))
        
        if max_variance == 0:
            return 1.0
        
        # Higher variance means less stable explanation
        stability = 1.0 - min(1.0, variance / max_variance)
        return float(stability)
    
    def _assess_explanation_quality(self, confidence_score: float) -> str:
        """Assess the quality of the explanation."""
        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        elif confidence_score >= 0.4:
            return "low"
        else:
            return "very_low"


class LimeExplainer(BaseExplainer):
    """LIME-based model explainer."""
    
    def __init__(self, config: ExplanationConfig):
        super().__init__(config)
        if not LIME_AVAILABLE:
            raise ImportError("LIME not available. Install with: pip install lime")
        
        self.explainer = None
    
    async def explain_global(self, 
                           model: Any,
                           X: pd.DataFrame, 
                           y: Optional[pd.Series] = None) -> ExplanationResult:
        """Generate global explanations using LIME (aggregated local explanations)."""
        
        start_time = datetime.utcnow()
        result = self._create_base_result(getattr(model, 'id', 'unknown'), ExplanationScope.GLOBAL)
        
        try:
            # Initialize LIME explainer
            if self.explainer is None:
                self.explainer = lime.lime_tabular.LimeTabularExplainer(
                    X.values,
                    feature_names=self.config.feature_names,
                    class_names=['class_0', 'class_1'] if hasattr(model, 'predict_proba') else None,
                    mode='classification' if hasattr(model, 'predict_proba') else 'regression'
                )
            
            # Sample instances for global explanation
            sample_size = min(self.config.sample_size, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            
            # Collect explanations for sampled instances
            all_explanations = []
            feature_importance_sum = {}
            
            for idx in sample_indices:
                instance = X.iloc[idx].values
                
                if hasattr(model, 'predict_proba'):
                    explanation = self.explainer.explain_instance(
                        instance, model.predict_proba, num_features=self.config.lime_num_features
                    )
                else:
                    explanation = self.explainer.explain_instance(
                        instance, model.predict, num_features=self.config.lime_num_features
                    )
                
                all_explanations.append(explanation)
                
                # Aggregate feature importance
                for feature_idx, importance in explanation.as_list():
                    feature_name = self.config.feature_names[feature_idx] if isinstance(feature_idx, int) else feature_idx
                    if feature_name not in feature_importance_sum:
                        feature_importance_sum[feature_name] = 0
                    feature_importance_sum[feature_name] += abs(importance)
            
            # Average feature importance
            for feature in feature_importance_sum:
                feature_importance_sum[feature] /= len(all_explanations)
            
            result.feature_importance = dict(sorted(
                feature_importance_sum.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            # Generate visualization data
            result.visualization_data = {
                "global_importance_plot": result.feature_importance,
                "sample_explanations": [
                    {
                        "instance_id": int(sample_indices[i]),
                        "explanation": exp.as_list()[:5],  # Top 5 features
                        "score": float(exp.score)
                    }
                    for i, exp in enumerate(all_explanations[:10])  # Show first 10
                ]
            }
            
            result.confidence_score = np.mean([exp.score for exp in all_explanations])
            result.explanation_quality = self._assess_explanation_quality(result.confidence_score)
            
        except Exception as e:
            self.logger.error(f"LIME global explanation failed: {str(e)}")
            result.explanation_quality = "failed"
        
        result.execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return result
    
    async def explain_local(self, 
                          model: Any,
                          X: pd.DataFrame,
                          instance_idx: int) -> ExplanationResult:
        """Generate local LIME explanation for a specific instance."""
        
        start_time = datetime.utcnow()
        result = self._create_base_result(getattr(model, 'id', 'unknown'), ExplanationScope.LOCAL)
        result.instance_id = str(instance_idx)
        
        try:
            # Initialize LIME explainer
            if self.explainer is None:
                self.explainer = lime.lime_tabular.LimeTabularExplainer(
                    X.values,
                    feature_names=self.config.feature_names,
                    class_names=['class_0', 'class_1'] if hasattr(model, 'predict_proba') else None,
                    mode='classification' if hasattr(model, 'predict_proba') else 'regression'
                )
            
            # Get instance
            instance = X.iloc[instance_idx].values
            
            # Generate explanation
            if hasattr(model, 'predict_proba'):
                explanation = self.explainer.explain_instance(
                    instance, model.predict_proba, num_features=self.config.lime_num_features
                )
                result.prediction = float(model.predict_proba([instance])[0][1])  # Probability of positive class
            else:
                explanation = self.explainer.explain_instance(
                    instance, model.predict, num_features=self.config.lime_num_features
                )
                result.prediction = float(model.predict([instance])[0])
            
            # Extract feature importance
            feature_importance = {}
            for feature_idx, importance in explanation.as_list():
                feature_name = self.config.feature_names[feature_idx] if isinstance(feature_idx, int) else feature_idx
                feature_importance[feature_name] = float(importance)
            
            result.feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            result.lime_explanation = {
                "explanation_as_list": explanation.as_list(),
                "local_exp": explanation.local_exp,
                "score": float(explanation.score)
            }
            
            # Generate visualization data
            result.visualization_data = {
                "lime_plot_data": {
                    "feature_contributions": result.feature_importance,
                    "instance_values": {
                        feature: float(X.iloc[instance_idx][feature])
                        for feature in self.config.feature_names
                        if feature in X.columns
                    },
                    "prediction": result.prediction
                }
            }
            
            result.confidence_score = float(explanation.score)
            result.explanation_quality = self._assess_explanation_quality(result.confidence_score)
            
        except Exception as e:
            self.logger.error(f"LIME local explanation failed: {str(e)}")
            result.explanation_quality = "failed"
        
        result.execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return result
    
    def _assess_explanation_quality(self, confidence_score: float) -> str:
        """Assess the quality of the explanation."""
        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        elif confidence_score >= 0.4:
            return "low"
        else:
            return "very_low"


class PermutationImportanceExplainer(BaseExplainer):
    """Permutation importance-based explainer."""
    
    async def explain_global(self, 
                           model: Any,
                           X: pd.DataFrame, 
                           y: Optional[pd.Series] = None) -> ExplanationResult:
        """Generate global explanations using permutation importance."""
        
        start_time = datetime.utcnow()
        result = self._create_base_result(getattr(model, 'id', 'unknown'), ExplanationScope.GLOBAL)
        
        try:
            if y is None:
                raise ValueError("Target values (y) required for permutation importance")
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature_name in enumerate(self.config.feature_names):
                if i < len(perm_importance.importances_mean):
                    feature_importance[feature_name] = float(perm_importance.importances_mean[i])
            
            result.feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            # Generate visualization data
            result.visualization_data = {
                "importance_plot": result.feature_importance,
                "importance_std": {
                    feature: float(perm_importance.importances_std[i])
                    for i, feature in enumerate(self.config.feature_names)
                    if i < len(perm_importance.importances_std)
                }
            }
            
            # Calculate confidence based on consistency of importance values
            importance_values = list(result.feature_importance.values())
            if importance_values:
                cv = np.std(importance_values) / np.mean(importance_values) if np.mean(importance_values) > 0 else 1
                result.confidence_score = float(max(0, 1 - cv))
            
            result.explanation_quality = self._assess_explanation_quality(result.confidence_score)
            
        except Exception as e:
            self.logger.error(f"Permutation importance explanation failed: {str(e)}")
            result.explanation_quality = "failed"
        
        result.execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return result
    
    async def explain_local(self, 
                          model: Any,
                          X: pd.DataFrame,
                          instance_idx: int) -> ExplanationResult:
        """Permutation importance doesn't provide local explanations."""
        result = self._create_base_result(getattr(model, 'id', 'unknown'), ExplanationScope.LOCAL)
        result.explanation_quality = "not_applicable"
        return result
    
    def _assess_explanation_quality(self, confidence_score: float) -> str:
        """Assess the quality of the explanation."""
        if confidence_score >= 0.7:
            return "high"
        elif confidence_score >= 0.5:
            return "medium"
        else:
            return "low"


class ModelExplainabilityFramework:
    """Comprehensive framework for model explainability and interpretability."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        
        # Available explainers
        self.explainers = {
            ExplanationMethod.SHAP: ShapExplainer,
            ExplanationMethod.LIME: LimeExplainer,
            ExplanationMethod.PERMUTATION_IMPORTANCE: PermutationImportanceExplainer,
        }
        
        # Explanation storage
        self.explanations: Dict[str, ExplanationResult] = {}
        
        # Model registry for explanations
        self.model_explanations: Dict[str, List[str]] = {}
        
        self.logger.info("Model explainability framework initialized")
    
    async def explain_model(self,
                          model: Any,
                          X: pd.DataFrame,
                          y: Optional[pd.Series] = None,
                          config: ExplanationConfig = None,
                          model_id: str = None) -> ExplanationResult:
        """Generate model explanations."""
        
        if config is None:
            config = ExplanationConfig(
                method=ExplanationMethod.SHAP,
                scope=ExplanationScope.GLOBAL,
                feature_names=list(X.columns)
            )
        
        model_id = model_id or getattr(model, 'id', f'model_{uuid.uuid4()}')
        
        # Get appropriate explainer
        if config.method not in self.explainers:
            raise ValueError(f"Explainer {config.method} not available")
        
        explainer_class = self.explainers[config.method]
        explainer = explainer_class(config)
        
        # Generate explanation
        if config.scope == ExplanationScope.GLOBAL:
            result = await explainer.explain_global(model, X, y)
        else:
            raise ValueError("Use explain_instance for local explanations")
        
        # Store explanation
        self.explanations[result.explanation_id] = result
        
        if model_id not in self.model_explanations:
            self.model_explanations[model_id] = []
        self.model_explanations[model_id].append(result.explanation_id)
        
        self.logger.info(
            "Model explanation generated",
            model_id=model_id,
            method=config.method.value,
            scope=config.scope.value,
            execution_time_ms=result.execution_time_ms,
            quality=result.explanation_quality
        )
        
        return result
    
    async def explain_instance(self,
                             model: Any,
                             X: pd.DataFrame,
                             instance_idx: int,
                             config: ExplanationConfig = None,
                             model_id: str = None) -> ExplanationResult:
        """Generate instance-level explanations."""
        
        if config is None:
            config = ExplanationConfig(
                method=ExplanationMethod.SHAP,
                scope=ExplanationScope.LOCAL,
                feature_names=list(X.columns)
            )
        
        config.scope = ExplanationScope.LOCAL  # Ensure local scope
        model_id = model_id or getattr(model, 'id', f'model_{uuid.uuid4()}')
        
        # Get appropriate explainer
        if config.method not in self.explainers:
            raise ValueError(f"Explainer {config.method} not available")
        
        explainer_class = self.explainers[config.method]
        explainer = explainer_class(config)
        
        # Generate explanation
        result = await explainer.explain_local(model, X, instance_idx)
        
        # Store explanation
        self.explanations[result.explanation_id] = result
        
        if model_id not in self.model_explanations:
            self.model_explanations[model_id] = []
        self.model_explanations[model_id].append(result.explanation_id)
        
        self.logger.info(
            "Instance explanation generated",
            model_id=model_id,
            instance_idx=instance_idx,
            method=config.method.value,
            execution_time_ms=result.execution_time_ms,
            quality=result.explanation_quality
        )
        
        return result
    
    async def explain_cohort(self,
                           model: Any,
                           X: pd.DataFrame,
                           cohort_filter: Callable[[pd.DataFrame], pd.DataFrame],
                           config: ExplanationConfig = None,
                           model_id: str = None) -> ExplanationResult:
        """Generate explanations for a specific cohort/segment."""
        
        # Filter data for cohort
        cohort_X = cohort_filter(X)
        
        if len(cohort_X) == 0:
            raise ValueError("Cohort filter resulted in empty dataset")
        
        if config is None:
            config = ExplanationConfig(
                method=ExplanationMethod.SHAP,
                scope=ExplanationScope.COHORT,
                feature_names=list(X.columns)
            )
        
        config.scope = ExplanationScope.COHORT
        model_id = model_id or getattr(model, 'id', f'model_{uuid.uuid4()}')
        
        # Generate explanation on cohort data
        result = await self.explain_model(model, cohort_X, None, config, model_id)
        result.scope = ExplanationScope.COHORT
        
        self.logger.info(
            "Cohort explanation generated",
            model_id=model_id,
            cohort_size=len(cohort_X),
            method=config.method.value
        )
        
        return result
    
    async def compare_explanations(self,
                                 explanation_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple explanations."""
        
        explanations = []
        for exp_id in explanation_ids:
            if exp_id in self.explanations:
                explanations.append(self.explanations[exp_id])
        
        if len(explanations) < 2:
            raise ValueError("Need at least 2 explanations to compare")
        
        # Compare feature importance across explanations
        all_features = set()
        for exp in explanations:
            all_features.update(exp.feature_importance.keys())
        
        comparison = {
            "explanations": [exp.explanation_id for exp in explanations],
            "methods": [exp.method.value for exp in explanations],
            "scopes": [exp.scope.value for exp in explanations],
            "feature_importance_comparison": {},
            "agreement_score": 0.0,
            "consistency_metrics": {}
        }
        
        # Calculate feature importance agreement
        feature_rankings = {}
        for feature in all_features:
            rankings = []
            for exp in explanations:
                importance = exp.feature_importance.get(feature, 0)
                # Rank relative to other features in this explanation
                sorted_features = sorted(exp.feature_importance.items(), key=lambda x: x[1], reverse=True)
                rank = next((i for i, (f, _) in enumerate(sorted_features) if f == feature), len(sorted_features))
                rankings.append(rank)
            feature_rankings[feature] = rankings
        
        # Calculate Spearman correlation for agreement
        if len(all_features) > 1:
            from scipy.stats import spearmanr
            all_rankings = list(feature_rankings.values())
            if len(all_rankings) >= 2:
                correlations = []
                for i in range(len(explanations)):
                    for j in range(i + 1, len(explanations)):
                        ranks_i = [feature_rankings[f][i] for f in all_features]
                        ranks_j = [feature_rankings[f][j] for f in all_features]
                        corr, _ = spearmanr(ranks_i, ranks_j)
                        correlations.append(corr if not np.isnan(corr) else 0)
                
                comparison["agreement_score"] = float(np.mean(correlations))
        
        comparison["feature_importance_comparison"] = feature_rankings
        
        return comparison
    
    async def generate_explanation_report(self,
                                        explanation_id: str,
                                        format: str = "json") -> str:
        """Generate a comprehensive explanation report."""
        
        if explanation_id not in self.explanations:
            raise ValueError(f"Explanation {explanation_id} not found")
        
        explanation = self.explanations[explanation_id]
        
        if format == "json":
            report = explanation.to_dict()
            return json.dumps(report, indent=2)
        
        elif format == "html":
            return self._generate_html_report(explanation)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_html_report(self, explanation: ExplanationResult) -> str:
        """Generate HTML report for explanation."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Explanation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .feature-importance {{ margin: 10px 0; }}
                .feature {{ display: flex; justify-content: space-between; margin: 5px 0; }}
                .quality-{explanation.explanation_quality} {{ color: {'green' if explanation.explanation_quality == 'high' else 'orange' if explanation.explanation_quality == 'medium' else 'red'}; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Explanation Report</h1>
                <p><strong>Explanation ID:</strong> {explanation.explanation_id}</p>
                <p><strong>Model ID:</strong> {explanation.model_id}</p>
                <p><strong>Method:</strong> {explanation.method.value}</p>
                <p><strong>Scope:</strong> {explanation.scope.value}</p>
                <p><strong>Generated:</strong> {explanation.timestamp.isoformat()}</p>
                <p><strong>Execution Time:</strong> {explanation.execution_time_ms:.2f} ms</p>
                <p><strong>Quality:</strong> <span class="quality-{explanation.explanation_quality}">{explanation.explanation_quality}</span></p>
            </div>
            
            <div class="section">
                <h2>Feature Importance</h2>
                <div class="feature-importance">
                    {''.join([f'<div class="feature"><span>{feature}</span><span>{importance:.4f}</span></div>' for feature, importance in list(explanation.feature_importance.items())[:10]])}
                </div>
            </div>
            
            {'<div class="section"><h2>Instance Details</h2><p><strong>Instance ID:</strong> ' + str(explanation.instance_id) + '</p><p><strong>Prediction:</strong> ' + str(explanation.prediction) + '</p></div>' if explanation.instance_id else ''}
            
            <div class="section">
                <h2>Confidence and Quality</h2>
                <p><strong>Confidence Score:</strong> {explanation.confidence_score:.4f}</p>
                <p><strong>Explanation Quality:</strong> {explanation.explanation_quality}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def get_explanation(self, explanation_id: str) -> Optional[ExplanationResult]:
        """Get explanation by ID."""
        return self.explanations.get(explanation_id)
    
    def get_model_explanations(self, model_id: str) -> List[ExplanationResult]:
        """Get all explanations for a model."""
        if model_id not in self.model_explanations:
            return []
        
        return [
            self.explanations[exp_id] 
            for exp_id in self.model_explanations[model_id]
            if exp_id in self.explanations
        ]
    
    def get_available_methods(self) -> List[str]:
        """Get list of available explanation methods."""
        available = []
        
        if SHAP_AVAILABLE:
            available.append(ExplanationMethod.SHAP.value)
        if LIME_AVAILABLE:
            available.append(ExplanationMethod.LIME.value)
        
        available.append(ExplanationMethod.PERMUTATION_IMPORTANCE.value)
        
        return available
    
    async def cleanup_old_explanations(self, days: int = 30) -> int:
        """Clean up explanations older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        to_remove = []
        for exp_id, explanation in self.explanations.items():
            if explanation.timestamp < cutoff_date:
                to_remove.append(exp_id)
        
        # Remove old explanations
        for exp_id in to_remove:
            del self.explanations[exp_id]
            
            # Clean up model references
            for model_id, exp_list in self.model_explanations.items():
                if exp_id in exp_list:
                    exp_list.remove(exp_id)
        
        self.logger.info(f"Cleaned up {len(to_remove)} old explanations")
        return len(to_remove)