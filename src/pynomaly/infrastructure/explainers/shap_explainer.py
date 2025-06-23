"""SHAP-based explainer implementation."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import uuid

import numpy as np

from pynomaly.domain.services.explainability_service import (
    ExplainerProtocol,
    LocalExplanation,
    GlobalExplanation, 
    CohortExplanation,
    FeatureContribution,
    ExplanationMethod
)

logger = logging.getLogger(__name__)

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


class SHAPExplainer(ExplainerProtocol):
    """SHAP-based explainer for anomaly detection models."""
    
    def __init__(
        self,
        explainer_type: str = "auto",
        background_data: Optional[np.ndarray] = None,
        n_background_samples: int = 100,
        **kwargs
    ):
        """Initialize SHAP explainer.
        
        Args:
            explainer_type: Type of SHAP explainer ('auto', 'tree', 'linear', 'kernel', 'permutation')
            background_data: Background data for explainer initialization
            n_background_samples: Number of background samples to use
            **kwargs: Additional arguments for SHAP explainer
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAPExplainer. Install with: pip install shap")
        
        self.explainer_type = explainer_type
        self.background_data = background_data
        self.n_background_samples = n_background_samples
        self.explainer_kwargs = kwargs
        self._explainer = None
        self._model = None
    
    def _get_explainer(self, model: Any, data: Optional[np.ndarray] = None) -> shap.Explainer:
        """Get or create SHAP explainer for the model."""
        if self._explainer is None or self._model != model:
            self._model = model
            
            # Determine background data
            background = self._get_background_data(data)
            
            # Create explainer based on type
            if self.explainer_type == "auto":
                try:
                    # Try to auto-detect explainer type
                    self._explainer = shap.Explainer(model, background, **self.explainer_kwargs)
                except Exception as e:
                    logger.warning(f"Auto explainer failed: {e}. Falling back to Permutation explainer.")
                    self._explainer = shap.explainers.Permutation(
                        model.predict if hasattr(model, 'predict') else model,
                        background,
                        **self.explainer_kwargs
                    )
            elif self.explainer_type == "tree":
                self._explainer = shap.TreeExplainer(model, background, **self.explainer_kwargs)
            elif self.explainer_type == "linear":
                self._explainer = shap.LinearExplainer(model, background, **self.explainer_kwargs)
            elif self.explainer_type == "kernel":
                self._explainer = shap.KernelExplainer(
                    model.predict if hasattr(model, 'predict') else model,
                    background,
                    **self.explainer_kwargs
                )
            elif self.explainer_type == "permutation":
                self._explainer = shap.explainers.Permutation(
                    model.predict if hasattr(model, 'predict') else model,
                    background,
                    **self.explainer_kwargs
                )
            else:
                raise ValueError(f"Unknown explainer type: {self.explainer_type}")
        
        return self._explainer
    
    def _get_background_data(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """Get background data for explainer."""
        if self.background_data is not None:
            return self.background_data
        elif data is not None:
            # Sample background data from provided data
            n_samples = min(self.n_background_samples, len(data))
            indices = np.random.choice(len(data), n_samples, replace=False)
            return data[indices]
        else:
            raise ValueError("Background data is required for SHAP explainer")
    
    def explain_local(
        self,
        instance: np.ndarray,
        model: Any,
        feature_names: List[str],
        **kwargs
    ) -> LocalExplanation:
        """Generate local explanation using SHAP."""
        try:
            # Get explainer
            explainer = self._get_explainer(model, None)
            
            # Ensure instance is 2D
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)
            
            # Calculate SHAP values
            shap_values = explainer(instance)
            
            # Handle different SHAP value formats
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0]  # Get first instance
                base_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0
            else:
                values = shap_values[0]  # Get first instance
                base_value = 0
            
            # Get model prediction
            if hasattr(model, 'decision_function'):
                anomaly_score = model.decision_function(instance)[0]
            elif hasattr(model, 'predict_proba'):
                anomaly_score = model.predict_proba(instance)[0, 1]  # Anomaly probability
            else:
                anomaly_score = model.predict(instance)[0]
            
            # Create feature contributions
            feature_contributions = []
            for i, (feature_name, contribution) in enumerate(zip(feature_names, values)):
                feature_contributions.append(FeatureContribution(
                    feature_name=feature_name,
                    value=float(instance[0, i]),
                    contribution=float(contribution),
                    importance=abs(float(contribution)),
                    rank=i + 1,
                    description=f"SHAP contribution for {feature_name}"
                ))
            
            # Sort by importance
            feature_contributions.sort(key=lambda x: x.importance, reverse=True)
            for i, contrib in enumerate(feature_contributions):
                contrib.rank = i + 1
            
            # Determine prediction
            threshold = kwargs.get('threshold', 0.5)
            prediction = "anomaly" if anomaly_score > threshold else "normal"
            confidence = float(abs(anomaly_score - threshold))
            
            return LocalExplanation(
                instance_id=str(uuid.uuid4()),
                anomaly_score=float(anomaly_score),
                prediction=prediction,
                confidence=confidence,
                feature_contributions=feature_contributions,
                explanation_method=ExplanationMethod.SHAP,
                model_name=model.__class__.__name__,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"SHAP local explanation failed: {e}")
            raise
    
    def explain_global(
        self,
        data: np.ndarray,
        model: Any,
        feature_names: List[str],
        **kwargs
    ) -> GlobalExplanation:
        """Generate global explanation using SHAP."""
        try:
            # Get explainer
            explainer = self._get_explainer(model, data)
            
            # Calculate SHAP values for all data
            max_samples = kwargs.get('max_samples', 1000)
            if len(data) > max_samples:
                indices = np.random.choice(len(data), max_samples, replace=False)
                sample_data = data[indices]
            else:
                sample_data = data
            
            shap_values = explainer(sample_data)
            
            # Handle different SHAP value formats
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values
            
            # Calculate feature importances (mean absolute SHAP values)
            feature_importances = {}
            mean_abs_shap = np.mean(np.abs(values), axis=0)
            
            for i, feature_name in enumerate(feature_names):
                feature_importances[feature_name] = float(mean_abs_shap[i])
            
            # Get top features
            sorted_features = sorted(
                feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_features = [f[0] for f in sorted_features[:10]]
            
            # Calculate model performance if possible
            model_performance = {}
            try:
                if hasattr(model, 'score'):
                    model_performance['score'] = float(model.score(data))
            except:
                pass
            
            # Create summary
            top_3_features = [f[0] for f in sorted_features[:3]]
            summary = f"Model is most sensitive to: {', '.join(top_3_features)}"
            
            return GlobalExplanation(
                model_name=model.__class__.__name__,
                feature_importances=feature_importances,
                top_features=top_features,
                explanation_method=ExplanationMethod.SHAP,
                model_performance=model_performance,
                timestamp=datetime.now().isoformat(),
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"SHAP global explanation failed: {e}")
            raise
    
    def explain_cohort(
        self,
        instances: np.ndarray,
        model: Any,
        feature_names: List[str],
        cohort_id: str,
        **kwargs
    ) -> CohortExplanation:
        """Generate cohort explanation using SHAP."""
        try:
            # Get explainer
            explainer = self._get_explainer(model, None)
            
            # Calculate SHAP values for cohort
            shap_values = explainer(instances)
            
            # Handle different SHAP value formats
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values
            
            # Calculate average contributions for cohort
            mean_contributions = np.mean(values, axis=0)
            mean_abs_contributions = np.mean(np.abs(values), axis=0)
            
            # Create common feature contributions
            common_features = []
            for i, feature_name in enumerate(feature_names):
                common_features.append(FeatureContribution(
                    feature_name=feature_name,
                    value=float(np.mean(instances[:, i])),
                    contribution=float(mean_contributions[i]),
                    importance=float(mean_abs_contributions[i]),
                    rank=i + 1,
                    description=f"Average SHAP contribution for {feature_name} in cohort"
                ))
            
            # Sort by importance
            common_features.sort(key=lambda x: x.importance, reverse=True)
            for i, contrib in enumerate(common_features):
                contrib.rank = i + 1
            
            # Create cohort description
            top_features = [f.feature_name for f in common_features[:3]]
            cohort_description = f"Cohort characterized by high importance of: {', '.join(top_features)}"
            
            return CohortExplanation(
                cohort_id=cohort_id,
                cohort_description=cohort_description,
                instance_count=len(instances),
                common_features=common_features,
                explanation_method=ExplanationMethod.SHAP,
                model_name=model.__class__.__name__,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"SHAP cohort explanation failed: {e}")
            raise