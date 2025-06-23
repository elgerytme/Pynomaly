"""LIME-based explainer implementation."""

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

# Optional LIME import
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")


class LIMEExplainer(ExplainerProtocol):
    """LIME-based explainer for anomaly detection models."""
    
    def __init__(
        self,
        training_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        mode: str = "regression",
        discretize_continuous: bool = True,
        **kwargs
    ):
        """Initialize LIME explainer.
        
        Args:
            training_data: Training data for LIME initialization
            feature_names: Names of features
            categorical_features: Indices of categorical features
            mode: LIME mode ('regression' or 'classification')
            discretize_continuous: Whether to discretize continuous features
            **kwargs: Additional arguments for LIME explainer
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for LIMEExplainer. Install with: pip install lime")
        
        self.training_data = training_data
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.mode = mode
        self.discretize_continuous = discretize_continuous
        self.explainer_kwargs = kwargs
        self._explainer = None
        self._model = None
    
    def _get_explainer(
        self,
        training_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> lime.lime_tabular.LimeTabularExplainer:
        """Get or create LIME explainer."""
        # Use provided data or fallback to initialization data
        data = training_data if training_data is not None else self.training_data
        names = feature_names if feature_names is not None else self.feature_names
        
        if data is None:
            raise ValueError("Training data is required for LIME explainer")
        
        if self._explainer is None:
            # Create feature names if not provided
            if names is None:
                names = [f"feature_{i}" for i in range(data.shape[1])]
            
            self._explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=data,
                feature_names=names,
                categorical_features=self.categorical_features,
                mode=self.mode,
                discretize_continuous=self.discretize_continuous,
                **self.explainer_kwargs
            )
        
        return self._explainer
    
    def _create_prediction_function(self, model: Any) -> callable:
        """Create prediction function for LIME."""
        def predict_fn(instances):
            # Handle different model types
            if hasattr(model, 'decision_function'):
                # Anomaly detection models (sklearn-style)
                scores = model.decision_function(instances)
                if self.mode == "regression":
                    return scores
                else:
                    # Convert to probabilities for classification mode
                    # Sigmoid transformation for anomaly scores
                    probs = 1 / (1 + np.exp(-scores))
                    return np.column_stack([1 - probs, probs])
            elif hasattr(model, 'predict_proba'):
                # Classification models
                return model.predict_proba(instances)
            elif hasattr(model, 'predict'):
                # Generic prediction
                predictions = model.predict(instances)
                if self.mode == "regression":
                    return predictions
                else:
                    # Convert to probabilities
                    probs = predictions.astype(float)
                    return np.column_stack([1 - probs, probs])
            else:
                # Assume callable model
                return model(instances)
        
        return predict_fn
    
    def explain_local(
        self,
        instance: np.ndarray,
        model: Any,
        feature_names: List[str],
        **kwargs
    ) -> LocalExplanation:
        """Generate local explanation using LIME."""
        try:
            # Get explainer
            explainer = self._get_explainer(
                training_data=kwargs.get('training_data'),
                feature_names=feature_names
            )
            
            # Ensure instance is 1D
            if instance.ndim > 1:
                instance = instance.flatten()
            
            # Create prediction function
            predict_fn = self._create_prediction_function(model)
            
            # Generate explanation
            num_features = kwargs.get('num_features', len(feature_names))
            explanation = explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features,
                **{k: v for k, v in kwargs.items() if k not in ['training_data', 'num_features']}
            )
            
            # Get model prediction
            instance_2d = instance.reshape(1, -1)
            if hasattr(model, 'decision_function'):
                anomaly_score = model.decision_function(instance_2d)[0]
            elif hasattr(model, 'predict_proba'):
                anomaly_score = model.predict_proba(instance_2d)[0, 1]  # Anomaly probability
            else:
                anomaly_score = model.predict(instance_2d)[0]
            
            # Extract feature contributions from LIME explanation
            feature_contributions = []
            explanation_map = dict(explanation.as_list())
            
            for i, feature_name in enumerate(feature_names):
                contribution = explanation_map.get(feature_name, 0.0)
                feature_contributions.append(FeatureContribution(
                    feature_name=feature_name,
                    value=float(instance[i]),
                    contribution=float(contribution),
                    importance=abs(float(contribution)),
                    rank=i + 1,
                    description=f"LIME contribution for {feature_name}"
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
                explanation_method=ExplanationMethod.LIME,
                model_name=model.__class__.__name__,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"LIME local explanation failed: {e}")
            raise
    
    def explain_global(
        self,
        data: np.ndarray,
        model: Any,
        feature_names: List[str],
        **kwargs
    ) -> GlobalExplanation:
        """Generate global explanation using LIME."""
        try:
            # For global explanation, generate local explanations for multiple instances
            max_samples = kwargs.get('max_samples', 100)
            sample_size = min(max_samples, len(data))
            
            # Sample data for global explanation
            indices = np.random.choice(len(data), sample_size, replace=False)
            sample_data = data[indices]
            
            # Generate local explanations for sampled instances
            local_explanations = []
            for instance in sample_data:
                try:
                    local_exp = self.explain_local(
                        instance=instance,
                        model=model,
                        feature_names=feature_names,
                        training_data=data,
                        **kwargs
                    )
                    local_explanations.append(local_exp)
                except Exception as e:
                    logger.warning(f"Failed to generate local explanation: {e}")
                    continue
            
            if not local_explanations:
                raise ValueError("Failed to generate any local explanations for global analysis")
            
            # Aggregate feature importances
            feature_importances = {}
            for feature_name in feature_names:
                importances = []
                for exp in local_explanations:
                    for contrib in exp.feature_contributions:
                        if contrib.feature_name == feature_name:
                            importances.append(contrib.importance)
                            break
                
                if importances:
                    feature_importances[feature_name] = float(np.mean(importances))
                else:
                    feature_importances[feature_name] = 0.0
            
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
            summary = f"Based on {len(local_explanations)} local explanations, " \
                     f"model is most sensitive to: {', '.join(top_3_features)}"
            
            return GlobalExplanation(
                model_name=model.__class__.__name__,
                feature_importances=feature_importances,
                top_features=top_features,
                explanation_method=ExplanationMethod.LIME,
                model_performance=model_performance,
                timestamp=datetime.now().isoformat(),
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"LIME global explanation failed: {e}")
            raise
    
    def explain_cohort(
        self,
        instances: np.ndarray,
        model: Any,
        feature_names: List[str],
        cohort_id: str,
        **kwargs
    ) -> CohortExplanation:
        """Generate cohort explanation using LIME."""
        try:
            # Generate local explanations for all instances in cohort
            local_explanations = []
            for instance in instances:
                try:
                    local_exp = self.explain_local(
                        instance=instance,
                        model=model,
                        feature_names=feature_names,
                        training_data=instances,
                        **kwargs
                    )
                    local_explanations.append(local_exp)
                except Exception as e:
                    logger.warning(f"Failed to generate local explanation: {e}")
                    continue
            
            if not local_explanations:
                raise ValueError("Failed to generate any local explanations for cohort")
            
            # Calculate average contributions for cohort
            common_features = []
            for feature_name in feature_names:
                contributions = []
                importances = []
                values = []
                
                for exp in local_explanations:
                    for contrib in exp.feature_contributions:
                        if contrib.feature_name == feature_name:
                            contributions.append(contrib.contribution)
                            importances.append(contrib.importance)
                            values.append(contrib.value)
                            break
                
                if contributions:
                    common_features.append(FeatureContribution(
                        feature_name=feature_name,
                        value=float(np.mean(values)),
                        contribution=float(np.mean(contributions)),
                        importance=float(np.mean(importances)),
                        rank=0,  # Will be set after sorting
                        description=f"Average LIME contribution for {feature_name} in cohort"
                    ))
            
            # Sort by importance
            common_features.sort(key=lambda x: x.importance, reverse=True)
            for i, contrib in enumerate(common_features):
                contrib.rank = i + 1
            
            # Create cohort description
            top_features = [f.feature_name for f in common_features[:3]]
            cohort_description = f"Cohort of {len(instances)} instances characterized by " \
                               f"high importance of: {', '.join(top_features)}"
            
            return CohortExplanation(
                cohort_id=cohort_id,
                cohort_description=cohort_description,
                instance_count=len(instances),
                common_features=common_features,
                explanation_method=ExplanationMethod.LIME,
                model_name=model.__class__.__name__,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"LIME cohort explanation failed: {e}")
            raise