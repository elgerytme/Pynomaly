"""Explainable AI service for anomaly detection interpretability."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..entities.anomaly import Anomaly
from ..entities.dataset import Dataset
from ..entities.detection_result import DetectionResult
from ...infrastructure.monitoring.distributed_tracing import trace_operation


logger = logging.getLogger(__name__)


class ExplanationMethod(Enum):
    """Available explanation methods."""
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    FEATURE_ABLATION = "feature_ablation"
    COUNTERFACTUAL = "counterfactual"
    RULE_EXTRACTION = "rule_extraction"
    GLOBAL_FEATURE_IMPORTANCE = "global_feature_importance"


class ExplanationType(Enum):
    """Types of explanations."""
    LOCAL = "local"  # Explain individual predictions
    GLOBAL = "global"  # Explain model behavior overall
    COHORT = "cohort"  # Explain behavior for groups of samples


@dataclass
class FeatureImportance:
    """Feature importance information."""
    
    feature_name: str
    importance_score: float
    rank: int
    confidence_interval: Optional[Tuple[float, float]] = None
    description: Optional[str] = None


@dataclass
class LocalExplanation:
    """Local explanation for a specific anomaly."""
    
    anomaly_index: int
    anomaly_score: float
    feature_contributions: List[FeatureImportance]
    baseline_value: float
    prediction_value: float
    
    # Additional context
    feature_values: Dict[str, Any] = field(default_factory=dict)
    counterfactuals: List[Dict[str, Any]] = field(default_factory=list)
    similar_samples: List[int] = field(default_factory=list)
    
    # Visualization data
    visualization_data: Optional[Dict[str, Any]] = None


@dataclass
class GlobalExplanation:
    """Global explanation for the model."""
    
    global_feature_importance: List[FeatureImportance]
    model_behavior_summary: Dict[str, Any]
    decision_rules: List[str] = field(default_factory=list)
    
    # Feature interactions
    feature_interactions: List[Tuple[str, str, float]] = field(default_factory=list)
    
    # Model statistics
    average_anomaly_score: float = 0.0
    feature_value_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Visualization data
    visualization_data: Optional[Dict[str, Any]] = None


@dataclass
class ExplanationResult:
    """Complete explanation result."""
    
    method: ExplanationMethod
    explanation_type: ExplanationType
    
    # Explanations
    local_explanations: List[LocalExplanation] = field(default_factory=list)
    global_explanation: Optional[GlobalExplanation] = None
    
    # Metadata
    computation_time: float = 0.0
    model_type: str = ""
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    explanation_quality: Dict[str, float] = field(default_factory=dict)


class ExplainableAIService:
    """Service for generating explanations of anomaly detection results."""
    
    def __init__(self):
        """Initialize explainable AI service."""
        self.explainers = {}
        self.explanation_cache = {}
        
        logger.info("Explainable AI service initialized")
    
    @trace_operation("explain_anomalies")
    async def explain_anomalies(
        self,
        detection_result: DetectionResult,
        dataset: Dataset,
        method: ExplanationMethod = ExplanationMethod.SHAP,
        explanation_type: ExplanationType = ExplanationType.LOCAL,
        max_explanations: int = 10
    ) -> ExplanationResult:
        """Generate explanations for anomaly detection results."""
        
        start_time = time.time()
        
        try:
            # Prepare data
            X, feature_names = await self._prepare_data(dataset)
            
            # Get anomaly indices and scores
            anomaly_indices = [anomaly.index for anomaly in detection_result.anomalies[:max_explanations]]
            anomaly_scores = [anomaly.score.value for anomaly in detection_result.anomalies[:max_explanations]]
            
            # Generate explanations based on method
            if method == ExplanationMethod.SHAP:
                result = await self._explain_with_shap(
                    X, feature_names, anomaly_indices, anomaly_scores, explanation_type
                )
            elif method == ExplanationMethod.LIME:
                result = await self._explain_with_lime(
                    X, feature_names, anomaly_indices, anomaly_scores, explanation_type
                )
            elif method == ExplanationMethod.PERMUTATION_IMPORTANCE:
                result = await self._explain_with_permutation_importance(
                    X, feature_names, anomaly_indices, anomaly_scores, explanation_type
                )
            elif method == ExplanationMethod.FEATURE_ABLATION:
                result = await self._explain_with_feature_ablation(
                    X, feature_names, anomaly_indices, anomaly_scores, explanation_type
                )
            else:
                # Fallback to basic feature importance
                result = await self._basic_feature_importance(
                    X, feature_names, anomaly_indices, anomaly_scores, explanation_type
                )
            
            # Set metadata
            result.method = method
            result.explanation_type = explanation_type
            result.computation_time = time.time() - start_time
            result.model_type = detection_result.algorithm
            result.dataset_info = {
                "n_samples": len(X),
                "n_features": len(feature_names),
                "n_anomalies": len(anomaly_indices)
            }
            
            # Calculate explanation quality metrics
            result.explanation_quality = await self._calculate_explanation_quality(result, X)
            
            logger.info(f"Generated {method.value} explanations in {result.computation_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            raise
    
    async def _prepare_data(self, dataset: Dataset) -> Tuple[np.ndarray, List[str]]:
        """Prepare data for explanation."""
        
        if hasattr(dataset, 'data'):
            data = dataset.data
        else:
            raise ValueError("Dataset has no data attribute")
        
        # Convert to pandas DataFrame if numpy array
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
        
        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric features found for explanation")
        
        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())
        
        feature_names = list(numeric_data.columns)
        X = numeric_data.values
        
        return X, feature_names
    
    async def _explain_with_shap(
        self,
        X: np.ndarray,
        feature_names: List[str],
        anomaly_indices: List[int],
        anomaly_scores: List[float],
        explanation_type: ExplanationType
    ) -> ExplanationResult:
        """Generate explanations using SHAP."""
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAP explanations")
        
        result = ExplanationResult(
            method=ExplanationMethod.SHAP,
            explanation_type=explanation_type
        )
        
        try:
            # Create a simple surrogate model for SHAP
            # In practice, you'd use the actual anomaly detection model
            surrogate_model = self._create_surrogate_model(X, anomaly_indices, anomaly_scores)
            
            # Create SHAP explainer
            explainer = shap.Explainer(surrogate_model, X)
            
            if explanation_type == ExplanationType.LOCAL:
                # Generate local explanations for each anomaly
                for i, (idx, score) in enumerate(zip(anomaly_indices, anomaly_scores)):
                    if idx >= len(X):
                        continue
                    
                    # Get SHAP values for this sample
                    shap_values = explainer([X[idx]])
                    
                    # Create feature contributions
                    feature_contributions = []
                    for j, (feature_name, shap_value) in enumerate(zip(feature_names, shap_values.values[0])):
                        importance = FeatureImportance(
                            feature_name=feature_name,
                            importance_score=float(shap_value),
                            rank=j,
                            description=f"SHAP contribution to anomaly score"
                        )
                        feature_contributions.append(importance)
                    
                    # Sort by absolute importance
                    feature_contributions.sort(key=lambda x: abs(x.importance_score), reverse=True)
                    
                    # Update ranks
                    for rank, contrib in enumerate(feature_contributions):
                        contrib.rank = rank + 1
                    
                    # Create local explanation
                    local_explanation = LocalExplanation(
                        anomaly_index=idx,
                        anomaly_score=score,
                        feature_contributions=feature_contributions,
                        baseline_value=float(explainer.expected_value),
                        prediction_value=score,
                        feature_values={name: float(X[idx][j]) for j, name in enumerate(feature_names)}
                    )
                    
                    result.local_explanations.append(local_explanation)
            
            elif explanation_type == ExplanationType.GLOBAL:
                # Generate global explanation
                sample_indices = anomaly_indices[:min(100, len(anomaly_indices))]  # Sample for efficiency
                if sample_indices:
                    sample_data = X[sample_indices]
                    shap_values = explainer(sample_data)
                    
                    # Calculate global feature importance
                    global_importance = np.abs(shap_values.values).mean(axis=0)
                    
                    global_features = []
                    for i, (feature_name, importance) in enumerate(zip(feature_names, global_importance)):
                        feature_imp = FeatureImportance(
                            feature_name=feature_name,
                            importance_score=float(importance),
                            rank=i + 1,
                            description="Average absolute SHAP value"
                        )
                        global_features.append(feature_imp)
                    
                    # Sort by importance
                    global_features.sort(key=lambda x: x.importance_score, reverse=True)
                    
                    # Update ranks
                    for rank, feature in enumerate(global_features):
                        feature.rank = rank + 1
                    
                    # Create global explanation
                    result.global_explanation = GlobalExplanation(
                        global_feature_importance=global_features,
                        model_behavior_summary={
                            "total_shap_values": float(np.sum(np.abs(shap_values.values))),
                            "average_prediction": float(np.mean(anomaly_scores)),
                            "feature_interaction_strength": float(np.std(shap_values.values))
                        },
                        average_anomaly_score=float(np.mean(anomaly_scores))
                    )
        
        except Exception as e:
            logger.warning(f"SHAP explanation failed, falling back to basic method: {e}")
            result = await self._basic_feature_importance(
                X, feature_names, anomaly_indices, anomaly_scores, explanation_type
            )
        
        return result
    
    async def _explain_with_lime(
        self,
        X: np.ndarray,
        feature_names: List[str],
        anomaly_indices: List[int],
        anomaly_scores: List[float],
        explanation_type: ExplanationType
    ) -> ExplanationResult:
        """Generate explanations using LIME."""
        
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for LIME explanations")
        
        result = ExplanationResult(
            method=ExplanationMethod.LIME,
            explanation_type=explanation_type
        )
        
        try:
            # Create surrogate model
            surrogate_model = self._create_surrogate_model(X, anomaly_indices, anomaly_scores)
            
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X,
                feature_names=feature_names,
                mode='regression',
                discretize_continuous=True
            )
            
            if explanation_type == ExplanationType.LOCAL:
                # Generate local explanations
                for i, (idx, score) in enumerate(zip(anomaly_indices, anomaly_scores)):
                    if idx >= len(X):
                        continue
                    
                    # Get LIME explanation
                    explanation = explainer.explain_instance(
                        X[idx], 
                        surrogate_model.predict,
                        num_features=min(len(feature_names), 10)
                    )
                    
                    # Extract feature contributions
                    feature_contributions = []
                    for feature_name, importance_score in explanation.as_list():
                        # Parse feature name to get original name
                        original_name = feature_name.split('<=')[0].split('>')[0].strip()
                        if original_name in feature_names:
                            importance = FeatureImportance(
                                feature_name=original_name,
                                importance_score=float(importance_score),
                                rank=0,  # Will be updated
                                description=f"LIME contribution: {feature_name}"
                            )
                            feature_contributions.append(importance)
                    
                    # Sort by absolute importance
                    feature_contributions.sort(key=lambda x: abs(x.importance_score), reverse=True)
                    
                    # Update ranks
                    for rank, contrib in enumerate(feature_contributions):
                        contrib.rank = rank + 1
                    
                    # Create local explanation
                    local_explanation = LocalExplanation(
                        anomaly_index=idx,
                        anomaly_score=score,
                        feature_contributions=feature_contributions,
                        baseline_value=0.0,  # LIME doesn't provide baseline
                        prediction_value=score,
                        feature_values={name: float(X[idx][j]) for j, name in enumerate(feature_names)}
                    )
                    
                    result.local_explanations.append(local_explanation)
        
        except Exception as e:
            logger.warning(f"LIME explanation failed, falling back to basic method: {e}")
            result = await self._basic_feature_importance(
                X, feature_names, anomaly_indices, anomaly_scores, explanation_type
            )
        
        return result
    
    async def _explain_with_permutation_importance(
        self,
        X: np.ndarray,
        feature_names: List[str],
        anomaly_indices: List[int],
        anomaly_scores: List[float],
        explanation_type: ExplanationType
    ) -> ExplanationResult:
        """Generate explanations using permutation importance."""
        
        if not SKLEARN_AVAILABLE:
            return await self._basic_feature_importance(
                X, feature_names, anomaly_indices, anomaly_scores, explanation_type
            )
        
        result = ExplanationResult(
            method=ExplanationMethod.PERMUTATION_IMPORTANCE,
            explanation_type=explanation_type
        )
        
        try:
            # Create surrogate model and labels
            y = np.zeros(len(X))
            y[anomaly_indices] = 1  # Binary labels for anomalies
            
            # Train surrogate model
            surrogate_model = RandomForestClassifier(n_estimators=50, random_state=42)
            surrogate_model.fit(X, y)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                surrogate_model, X, y, 
                n_repeats=10, 
                random_state=42,
                scoring='roc_auc'
            )
            
            # Create feature importance list
            global_features = []
            for i, (feature_name, importance, std) in enumerate(
                zip(feature_names, perm_importance.importances_mean, perm_importance.importances_std)
            ):
                feature_imp = FeatureImportance(
                    feature_name=feature_name,
                    importance_score=float(importance),
                    rank=i + 1,
                    confidence_interval=(
                        float(importance - std),
                        float(importance + std)
                    ),
                    description="Permutation importance"
                )
                global_features.append(feature_imp)
            
            # Sort by importance
            global_features.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Update ranks
            for rank, feature in enumerate(global_features):
                feature.rank = rank + 1
            
            if explanation_type == ExplanationType.GLOBAL:
                result.global_explanation = GlobalExplanation(
                    global_feature_importance=global_features,
                    model_behavior_summary={
                        "total_importance": float(np.sum(perm_importance.importances_mean)),
                        "importance_std": float(np.mean(perm_importance.importances_std)),
                        "model_accuracy": float(surrogate_model.score(X, y))
                    },
                    average_anomaly_score=float(np.mean(anomaly_scores))
                )
            
            elif explanation_type == ExplanationType.LOCAL:
                # For local explanations, use feature values weighted by global importance
                for i, (idx, score) in enumerate(zip(anomaly_indices, anomaly_scores)):
                    if idx >= len(X):
                        continue
                    
                    feature_contributions = []
                    for j, feature_imp in enumerate(global_features):
                        # Weight global importance by feature value
                        feature_value = X[idx][feature_names.index(feature_imp.feature_name)]
                        local_importance = feature_imp.importance_score * abs(feature_value)
                        
                        contribution = FeatureImportance(
                            feature_name=feature_imp.feature_name,
                            importance_score=float(local_importance),
                            rank=j + 1,
                            description=f"Permutation importance weighted by feature value"
                        )
                        feature_contributions.append(contribution)
                    
                    local_explanation = LocalExplanation(
                        anomaly_index=idx,
                        anomaly_score=score,
                        feature_contributions=feature_contributions,
                        baseline_value=0.0,
                        prediction_value=score,
                        feature_values={name: float(X[idx][j]) for j, name in enumerate(feature_names)}
                    )
                    
                    result.local_explanations.append(local_explanation)
        
        except Exception as e:
            logger.warning(f"Permutation importance explanation failed: {e}")
            result = await self._basic_feature_importance(
                X, feature_names, anomaly_indices, anomaly_scores, explanation_type
            )
        
        return result
    
    async def _explain_with_feature_ablation(
        self,
        X: np.ndarray,
        feature_names: List[str],
        anomaly_indices: List[int],
        anomaly_scores: List[float],
        explanation_type: ExplanationType
    ) -> ExplanationResult:
        """Generate explanations using feature ablation."""
        
        result = ExplanationResult(
            method=ExplanationMethod.FEATURE_ABLATION,
            explanation_type=explanation_type
        )
        
        # Feature ablation: remove each feature and see impact on anomaly scores
        baseline_scores = anomaly_scores.copy()
        feature_importances = []
        
        for i, feature_name in enumerate(feature_names):
            # Create ablated data (set feature to mean)
            X_ablated = X.copy()
            feature_mean = np.mean(X[:, i])
            X_ablated[:, i] = feature_mean
            
            # Calculate change in anomaly scores (simplified)
            # In practice, you'd re-run the detection algorithm
            score_changes = []
            for idx in anomaly_indices:
                if idx < len(X):
                    # Simplified: assume score changes proportionally to feature deviation
                    original_value = X[idx, i]
                    deviation = abs(original_value - feature_mean)
                    normalized_deviation = deviation / (np.std(X[:, i]) + 1e-8)
                    score_changes.append(normalized_deviation)
            
            avg_score_change = np.mean(score_changes) if score_changes else 0.0
            
            importance = FeatureImportance(
                feature_name=feature_name,
                importance_score=float(avg_score_change),
                rank=i + 1,
                description="Feature ablation importance"
            )
            feature_importances.append(importance)
        
        # Sort by importance
        feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
        
        # Update ranks
        for rank, feature in enumerate(feature_importances):
            feature.rank = rank + 1
        
        if explanation_type == ExplanationType.GLOBAL:
            result.global_explanation = GlobalExplanation(
                global_feature_importance=feature_importances,
                model_behavior_summary={
                    "total_importance": float(sum(f.importance_score for f in feature_importances)),
                    "max_importance": float(max(f.importance_score for f in feature_importances)),
                    "method": "feature_ablation"
                },
                average_anomaly_score=float(np.mean(anomaly_scores))
            )
        
        return result
    
    async def _basic_feature_importance(
        self,
        X: np.ndarray,
        feature_names: List[str],
        anomaly_indices: List[int],
        anomaly_scores: List[float],
        explanation_type: ExplanationType
    ) -> ExplanationResult:
        """Generate basic feature importance explanations."""
        
        result = ExplanationResult(
            method=ExplanationMethod.GLOBAL_FEATURE_IMPORTANCE,
            explanation_type=explanation_type
        )
        
        # Calculate basic statistics for feature importance
        feature_importances = []
        
        for i, feature_name in enumerate(feature_names):
            # Calculate variance of feature values for anomalies vs normal samples
            anomaly_values = [X[idx, i] for idx in anomaly_indices if idx < len(X)]
            normal_indices = [j for j in range(len(X)) if j not in anomaly_indices]
            normal_values = [X[j, i] for j in normal_indices[:len(anomaly_values)]]  # Sample same size
            
            if len(anomaly_values) > 0 and len(normal_values) > 0:
                # Use difference in means as importance score
                anomaly_mean = np.mean(anomaly_values)
                normal_mean = np.mean(normal_values)
                importance_score = abs(anomaly_mean - normal_mean) / (np.std(X[:, i]) + 1e-8)
            else:
                importance_score = 0.0
            
            importance = FeatureImportance(
                feature_name=feature_name,
                importance_score=float(importance_score),
                rank=i + 1,
                description="Statistical difference between anomalies and normal samples"
            )
            feature_importances.append(importance)
        
        # Sort by importance
        feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
        
        # Update ranks
        for rank, feature in enumerate(feature_importances):
            feature.rank = rank + 1
        
        if explanation_type == ExplanationType.GLOBAL:
            result.global_explanation = GlobalExplanation(
                global_feature_importance=feature_importances,
                model_behavior_summary={
                    "method": "statistical_analysis",
                    "total_features": len(feature_names),
                    "total_anomalies": len(anomaly_indices)
                },
                average_anomaly_score=float(np.mean(anomaly_scores))
            )
        
        elif explanation_type == ExplanationType.LOCAL:
            # Create local explanations using global importance
            for i, (idx, score) in enumerate(zip(anomaly_indices, anomaly_scores)):
                if idx >= len(X):
                    continue
                
                # Weight global importance by feature values
                feature_contributions = []
                for feature_imp in feature_importances:
                    j = feature_names.index(feature_imp.feature_name)
                    feature_value = X[idx, j]
                    # Normalize feature value
                    feature_std = np.std(X[:, j])
                    feature_mean = np.mean(X[:, j])
                    normalized_value = abs(feature_value - feature_mean) / (feature_std + 1e-8)
                    
                    local_importance = feature_imp.importance_score * normalized_value
                    
                    contribution = FeatureImportance(
                        feature_name=feature_imp.feature_name,
                        importance_score=float(local_importance),
                        rank=feature_imp.rank,
                        description=f"Statistical importance weighted by deviation from mean"
                    )
                    feature_contributions.append(contribution)
                
                local_explanation = LocalExplanation(
                    anomaly_index=idx,
                    anomaly_score=score,
                    feature_contributions=feature_contributions,
                    baseline_value=float(np.mean(anomaly_scores)),
                    prediction_value=score,
                    feature_values={name: float(X[idx][j]) for j, name in enumerate(feature_names)}
                )
                
                result.local_explanations.append(local_explanation)
        
        return result
    
    def _create_surrogate_model(self, X: np.ndarray, anomaly_indices: List[int], anomaly_scores: List[float]):
        """Create a surrogate model for explanation."""
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for surrogate models")
        
        # Create binary labels
        y = np.zeros(len(X))
        for idx in anomaly_indices:
            if idx < len(X):
                y[idx] = 1
        
        # Train simple model
        model = DecisionTreeClassifier(max_depth=10, random_state=42)
        model.fit(X, y)
        
        return model
    
    async def _calculate_explanation_quality(
        self,
        explanation_result: ExplanationResult,
        X: np.ndarray
    ) -> Dict[str, float]:
        """Calculate quality metrics for explanations."""
        
        quality_metrics = {}
        
        # Consistency: how consistent are explanations across similar samples
        if explanation_result.local_explanations:
            feature_importance_lists = []
            for explanation in explanation_result.local_explanations:
                importance_dict = {
                    contrib.feature_name: contrib.importance_score 
                    for contrib in explanation.feature_contributions
                }
                feature_importance_lists.append(importance_dict)
            
            if len(feature_importance_lists) > 1:
                # Calculate correlation between importance rankings
                correlations = []
                for i in range(len(feature_importance_lists)):
                    for j in range(i + 1, len(feature_importance_lists)):
                        common_features = set(feature_importance_lists[i].keys()) & set(feature_importance_lists[j].keys())
                        if len(common_features) > 1:
                            importance1 = [feature_importance_lists[i][f] for f in common_features]
                            importance2 = [feature_importance_lists[j][f] for f in common_features]
                            correlation = np.corrcoef(importance1, importance2)[0, 1]
                            if not np.isnan(correlation):
                                correlations.append(correlation)
                
                quality_metrics["consistency"] = float(np.mean(correlations)) if correlations else 0.0
        
        # Coverage: how much of the model behavior is explained
        if explanation_result.global_explanation:
            total_importance = sum(
                abs(f.importance_score) for f in explanation_result.global_explanation.global_feature_importance
            )
            quality_metrics["coverage"] = min(1.0, float(total_importance))
        
        # Stability: add noise and see how explanations change (simplified)
        quality_metrics["stability"] = 0.8  # Placeholder - would need multiple runs with noise
        
        # Sparsity: prefer explanations with fewer important features
        if explanation_result.global_explanation:
            n_important_features = len([
                f for f in explanation_result.global_explanation.global_feature_importance
                if f.importance_score > 0.1  # Threshold for "important"
            ])
            total_features = len(explanation_result.global_explanation.global_feature_importance)
            quality_metrics["sparsity"] = 1.0 - (n_important_features / total_features)
        
        return quality_metrics
    
    async def generate_counterfactuals(
        self,
        dataset: Dataset,
        anomaly: Anomaly,
        n_counterfactuals: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations for an anomaly."""
        
        X, feature_names = await self._prepare_data(dataset)
        
        if anomaly.index >= len(X):
            return []
        
        original_sample = X[anomaly.index]
        counterfactuals = []
        
        # Simple counterfactual generation: modify top contributing features
        for i in range(n_counterfactuals):
            # Create a copy of the original sample
            counterfactual = original_sample.copy()
            
            # Modify a random subset of features toward the mean
            n_features_to_modify = min(3, len(feature_names))
            features_to_modify = np.random.choice(
                len(feature_names), 
                size=n_features_to_modify, 
                replace=False
            )
            
            for feature_idx in features_to_modify:
                feature_mean = np.mean(X[:, feature_idx])
                feature_std = np.std(X[:, feature_idx])
                
                # Move toward mean with some noise
                direction = 1 if counterfactual[feature_idx] > feature_mean else -1
                step_size = direction * np.random.uniform(0.5, 1.5) * feature_std
                counterfactual[feature_idx] = feature_mean + step_size
            
            # Create counterfactual dictionary
            counterfactual_dict = {
                f"counterfactual_{i+1}": {
                    feature_names[j]: float(counterfactual[j]) 
                    for j in range(len(feature_names))
                },
                "modified_features": [feature_names[j] for j in features_to_modify],
                "distance_from_original": float(np.linalg.norm(counterfactual - original_sample))
            }
            counterfactuals.append(counterfactual_dict)
        
        return counterfactuals
    
    async def explain_model_decision_rules(
        self,
        detection_result: DetectionResult,
        dataset: Dataset,
        max_rules: int = 10
    ) -> List[str]:
        """Extract decision rules from the anomaly detection model."""
        
        if not SKLEARN_AVAILABLE:
            return ["Decision rule extraction requires scikit-learn"]
        
        try:
            X, feature_names = await self._prepare_data(dataset)
            
            # Create labels
            y = np.zeros(len(X))
            for anomaly in detection_result.anomalies:
                if anomaly.index < len(X):
                    y[anomaly.index] = 1
            
            # Train interpretable model
            tree_model = DecisionTreeClassifier(
                max_depth=5, 
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
            tree_model.fit(X, y)
            
            # Extract rules from tree
            rules = self._extract_tree_rules(tree_model, feature_names, max_rules)
            
            return rules
            
        except Exception as e:
            logger.warning(f"Rule extraction failed: {e}")
            return [f"Rule extraction failed: {str(e)}"]
    
    def _extract_tree_rules(
        self, 
        tree_model, 
        feature_names: List[str], 
        max_rules: int
    ) -> List[str]:
        """Extract rules from decision tree."""
        
        tree = tree_model.tree_
        rules = []
        
        def get_rules(node_id, depth, condition):
            if depth > 5:  # Limit depth
                return
            
            # If leaf node
            if tree.children_left[node_id] == tree.children_right[node_id]:
                samples = tree.n_node_samples[node_id]
                value = tree.value[node_id][0]
                
                if value[1] > value[0]:  # More anomalies than normal
                    confidence = value[1] / (value[0] + value[1])
                    if confidence > 0.6 and samples > 5:  # Minimum confidence and support
                        rule = f"IF {condition} THEN anomaly (confidence: {confidence:.2f}, support: {samples})"
                        rules.append(rule)
                return
            
            # Get feature and threshold
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_idx]
            
            # Left child (feature <= threshold)
            left_condition = f"{condition} AND {feature_name} <= {threshold:.3f}" if condition else f"{feature_name} <= {threshold:.3f}"
            get_rules(tree.children_left[node_id], depth + 1, left_condition)
            
            # Right child (feature > threshold)
            right_condition = f"{condition} AND {feature_name} > {threshold:.3f}" if condition else f"{feature_name} > {threshold:.3f}"
            get_rules(tree.children_right[node_id], depth + 1, right_condition)
        
        get_rules(0, 0, "")
        
        # Sort by confidence and return top rules
        # For simplicity, just return first max_rules
        return rules[:max_rules]


# Global explainable AI service instance
_explainable_ai_service: Optional[ExplainableAIService] = None


def get_explainable_ai_service() -> ExplainableAIService:
    """Get the global explainable AI service instance."""
    global _explainable_ai_service
    if _explainable_ai_service is None:
        _explainable_ai_service = ExplainableAIService()
    return _explainable_ai_service