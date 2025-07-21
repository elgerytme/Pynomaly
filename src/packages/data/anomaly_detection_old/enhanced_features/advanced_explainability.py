"""Advanced explainability features for anomaly detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import warnings

from simplified_services.core_detection_service import DetectionResult


@dataclass
class ExplanationResult:
    """Result of anomaly explanation analysis."""
    sample_index: int
    is_anomaly: bool
    anomaly_score: float
    feature_importance: Dict[str, float]
    top_contributing_features: List[Tuple[str, float]]
    explanation_text: str
    confidence: float
    local_context: Dict[str, Any]


@dataclass 
class GlobalExplanation:
    """Global model explanation results."""
    algorithm: str
    feature_importance_global: Dict[str, float]
    anomaly_patterns: List[Dict[str, Any]]
    decision_boundaries: Optional[Dict[str, Any]]
    model_interpretability_score: float
    summary: str


class AdvancedExplainability:
    """Advanced explainability system for anomaly detection.
    
    This system provides comprehensive explanations for anomaly detection results:
    - Feature importance analysis using multiple methods
    - Local explanations for individual predictions
    - Global model interpretability
    - Counterfactual explanations
    - Pattern-based explanations
    - Natural language explanations
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        explanation_methods: Optional[List[str]] = None
    ):
        """Initialize advanced explainability system.
        
        Args:
            feature_names: Names of features for interpretation
            explanation_methods: Methods to use for explanations
        """
        self.feature_names = feature_names
        self.explanation_methods = explanation_methods or [
            "permutation", "gradient", "shap_approx", "local_outlier"
        ]
        
        # Cache for computations
        self._feature_stats_cache: Dict[str, Any] = {}
        self._explanation_cache: Dict[str, ExplanationResult] = {}
    
    def explain_prediction(
        self,
        sample: npt.NDArray[np.floating],
        sample_index: int,
        detection_result: DetectionResult,
        training_data: Optional[npt.NDArray[np.floating]] = None,
        model_info: Optional[Dict[str, Any]] = None
    ) -> ExplanationResult:
        """Explain a single anomaly prediction.
        
        Args:
            sample: The sample to explain
            sample_index: Index of the sample
            detection_result: Detection result from anomaly detector
            training_data: Training data for context
            model_info: Model information for interpretation
            
        Returns:
            Detailed explanation of the prediction
        """
        is_anomaly = detection_result.predictions[sample_index] == 1
        anomaly_score = detection_result.scores[sample_index] if detection_result.scores is not None else 0.0
        
        # Calculate feature importance using multiple methods
        feature_importance = self._calculate_feature_importance(
            sample, training_data, detection_result, model_info
        )
        
        # Get top contributing features
        top_features = self._get_top_contributing_features(feature_importance, top_k=5)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            sample, is_anomaly, anomaly_score, top_features, training_data
        )
        
        # Calculate confidence in explanation
        confidence = self._calculate_explanation_confidence(
            feature_importance, anomaly_score, training_data
        )
        
        # Get local context
        local_context = self._get_local_context(sample, training_data)
        
        return ExplanationResult(
            sample_index=sample_index,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            feature_importance=feature_importance,
            top_contributing_features=top_features,
            explanation_text=explanation_text,
            confidence=confidence,
            local_context=local_context
        )
    
    def explain_global_model(
        self,
        training_data: npt.NDArray[np.floating],
        detection_result: DetectionResult,
        model_info: Optional[Dict[str, Any]] = None
    ) -> GlobalExplanation:
        """Explain the global behavior of the anomaly detection model.
        
        Args:
            training_data: Training data
            detection_result: Detection results
            model_info: Model information
            
        Returns:
            Global model explanation
        """
        # Calculate global feature importance
        global_importance = self._calculate_global_feature_importance(
            training_data, detection_result, model_info
        )
        
        # Identify anomaly patterns
        anomaly_patterns = self._identify_anomaly_patterns(
            training_data, detection_result
        )
        
        # Estimate decision boundaries (simplified)
        decision_boundaries = self._estimate_decision_boundaries(
            training_data, detection_result
        )
        
        # Calculate model interpretability score
        interpretability_score = self._calculate_interpretability_score(
            detection_result, global_importance
        )
        
        # Generate summary
        summary = self._generate_global_summary(
            detection_result, global_importance, anomaly_patterns
        )
        
        return GlobalExplanation(
            algorithm=detection_result.algorithm,
            feature_importance_global=global_importance,
            anomaly_patterns=anomaly_patterns,
            decision_boundaries=decision_boundaries,
            model_interpretability_score=interpretability_score,
            summary=summary
        )
    
    def generate_counterfactuals(
        self,
        sample: npt.NDArray[np.floating],
        training_data: npt.NDArray[np.floating],
        detection_result: DetectionResult,
        n_counterfactuals: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations.
        
        Args:
            sample: Sample to generate counterfactuals for
            training_data: Training data for reference
            detection_result: Detection result
            n_counterfactuals: Number of counterfactuals to generate
            
        Returns:
            List of counterfactual explanations
        """
        counterfactuals = []
        
        if len(training_data) == 0:
            return counterfactuals
        
        # Find normal samples (non-anomalies) in training data
        normal_samples = training_data[detection_result.predictions == 0]
        
        if len(normal_samples) == 0:
            return counterfactuals
        
        # Generate counterfactuals by modifying sample towards normal samples
        for i in range(min(n_counterfactuals, len(normal_samples))):
            target_normal = normal_samples[i]
            
            # Calculate minimal changes needed
            differences = target_normal - sample
            
            # Apply gradual changes
            for alpha in [0.3, 0.5, 0.7]:
                counterfactual = sample + alpha * differences
                
                # Calculate feature changes
                feature_changes = {}
                for j, (original, modified) in enumerate(zip(sample, counterfactual)):
                    if abs(modified - original) > 1e-6:
                        feature_name = self._get_feature_name(j)
                        feature_changes[feature_name] = {
                            'original': float(original),
                            'modified': float(modified),
                            'change': float(modified - original)
                        }
                
                if feature_changes:  # Only add if there are meaningful changes
                    counterfactuals.append({
                        'counterfactual_sample': counterfactual.tolist(),
                        'feature_changes': feature_changes,
                        'alpha': alpha,
                        'distance_to_original': float(np.linalg.norm(differences * alpha)),
                        'explanation': f"If you change {len(feature_changes)} features by {alpha*100:.0f}% towards normal patterns, the sample would likely be classified as normal."
                    })
                
                if len(counterfactuals) >= n_counterfactuals:
                    break
            
            if len(counterfactuals) >= n_counterfactuals:
                break
        
        return counterfactuals[:n_counterfactuals]
    
    def explain_anomaly_clusters(
        self,
        training_data: npt.NDArray[np.floating],
        detection_result: DetectionResult,
        n_clusters: int = 3
    ) -> List[Dict[str, Any]]:
        """Explain different types/clusters of anomalies.
        
        Args:
            training_data: Training data
            detection_result: Detection results
            n_clusters: Number of anomaly clusters to identify
            
        Returns:
            List of anomaly cluster explanations
        """
        # Get anomalous samples
        anomaly_indices = np.where(detection_result.predictions == 1)[0]
        
        if len(anomaly_indices) == 0:
            return []
        
        anomalous_samples = training_data[anomaly_indices]
        
        # Simple clustering using k-means approach (without sklearn dependency)
        clusters = self._simple_clustering(anomalous_samples, n_clusters)
        
        cluster_explanations = []
        
        for cluster_id, cluster_samples in clusters.items():
            if len(cluster_samples) == 0:
                continue
            
            # Calculate cluster characteristics
            cluster_center = np.mean(cluster_samples, axis=0)
            cluster_std = np.std(cluster_samples, axis=0)
            
            # Find distinguishing features
            distinguishing_features = self._find_distinguishing_features(
                cluster_center, training_data
            )
            
            # Generate cluster explanation
            explanation = self._generate_cluster_explanation(
                cluster_id, len(cluster_samples), distinguishing_features
            )
            
            cluster_explanations.append({
                'cluster_id': cluster_id,
                'size': len(cluster_samples),
                'center': cluster_center.tolist(),
                'std': cluster_std.tolist(),
                'distinguishing_features': distinguishing_features,
                'explanation': explanation,
                'sample_indices': [int(idx) for idx in anomaly_indices[cluster_samples]]
            })
        
        return cluster_explanations
    
    def _calculate_feature_importance(
        self,
        sample: npt.NDArray[np.floating],
        training_data: Optional[npt.NDArray[np.floating]],
        detection_result: DetectionResult,
        model_info: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate feature importance for a sample using multiple methods."""
        importance_scores = {}
        
        # Method 1: Permutation importance (simplified)
        if "permutation" in self.explanation_methods and training_data is not None:
            perm_importance = self._permutation_importance(sample, training_data)
            for i, score in enumerate(perm_importance):
                feature_name = self._get_feature_name(i)
                importance_scores[f"{feature_name}_permutation"] = score
        
        # Method 2: Gradient-based importance (approximation)
        if "gradient" in self.explanation_methods:
            grad_importance = self._gradient_importance(sample, training_data)
            for i, score in enumerate(grad_importance):
                feature_name = self._get_feature_name(i)
                importance_scores[f"{feature_name}_gradient"] = score
        
        # Method 3: SHAP approximation
        if "shap_approx" in self.explanation_methods and training_data is not None:
            shap_importance = self._shap_approximation(sample, training_data)
            for i, score in enumerate(shap_importance):
                feature_name = self._get_feature_name(i)
                importance_scores[f"{feature_name}_shap"] = score
        
        # Method 4: Local outlier factor contribution
        if "local_outlier" in self.explanation_methods and training_data is not None:
            lof_importance = self._local_outlier_importance(sample, training_data)
            for i, score in enumerate(lof_importance):
                feature_name = self._get_feature_name(i)
                importance_scores[f"{feature_name}_lof"] = score
        
        # Aggregate importance scores
        aggregated_importance = {}
        feature_count = len(sample)
        
        for i in range(feature_count):
            feature_name = self._get_feature_name(i)
            scores = [
                importance_scores.get(f"{feature_name}_{method}", 0.0)
                for method in ["permutation", "gradient", "shap", "lof"]
            ]
            aggregated_importance[feature_name] = np.mean(scores)
        
        return aggregated_importance
    
    def _calculate_global_feature_importance(
        self,
        training_data: npt.NDArray[np.floating],
        detection_result: DetectionResult,
        model_info: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate global feature importance across all samples."""
        feature_count = training_data.shape[1]
        global_importance = {}
        
        # Calculate variance-based importance
        feature_variances = np.var(training_data, axis=0)
        max_variance = np.max(feature_variances)
        
        for i in range(feature_count):
            feature_name = self._get_feature_name(i)
            
            # Normalize variance
            variance_importance = feature_variances[i] / max_variance if max_variance > 0 else 0.0
            
            # Calculate anomaly correlation (simplified)
            feature_values = training_data[:, i]
            anomaly_correlation = abs(np.corrcoef(feature_values, detection_result.predictions)[0, 1])
            if np.isnan(anomaly_correlation):
                anomaly_correlation = 0.0
            
            # Combine importance measures
            global_importance[feature_name] = (variance_importance + anomaly_correlation) / 2
        
        return global_importance
    
    def _permutation_importance(
        self,
        sample: npt.NDArray[np.floating],
        training_data: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Calculate permutation-based feature importance."""
        importance = np.zeros(len(sample))
        
        # Calculate baseline score (distance to nearest neighbors)
        baseline_score = self._calculate_outlier_score(sample, training_data)
        
        for i in range(len(sample)):
            # Create permuted sample
            permuted_sample = sample.copy()
            
            # Replace feature with random value from training data
            if len(training_data) > 0:
                random_value = np.random.choice(training_data[:, i])
                permuted_sample[i] = random_value
            
            # Calculate new score
            permuted_score = self._calculate_outlier_score(permuted_sample, training_data)
            
            # Importance is the change in score
            importance[i] = abs(baseline_score - permuted_score)
        
        # Normalize
        if np.max(importance) > 0:
            importance = importance / np.max(importance)
        
        return importance
    
    def _gradient_importance(
        self,
        sample: npt.NDArray[np.floating],
        training_data: Optional[npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Calculate gradient-based feature importance (approximation)."""
        importance = np.zeros(len(sample))
        
        if training_data is None or len(training_data) == 0:
            return importance
        
        # Calculate gradients using finite differences
        epsilon = 1e-6
        baseline_score = self._calculate_outlier_score(sample, training_data)
        
        for i in range(len(sample)):
            # Perturb feature slightly
            perturbed_sample = sample.copy()
            perturbed_sample[i] += epsilon
            
            perturbed_score = self._calculate_outlier_score(perturbed_sample, training_data)
            
            # Gradient approximation
            gradient = (perturbed_score - baseline_score) / epsilon
            importance[i] = abs(gradient)
        
        # Normalize
        if np.max(importance) > 0:
            importance = importance / np.max(importance)
        
        return importance
    
    def _shap_approximation(
        self,
        sample: npt.NDArray[np.floating],
        training_data: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Calculate SHAP-like importance (simplified approximation)."""
        importance = np.zeros(len(sample))
        
        # Use training data mean as baseline
        baseline = np.mean(training_data, axis=0)
        
        # Calculate contribution of each feature
        for i in range(len(sample)):
            # Create sample with feature i set to baseline
            modified_sample = sample.copy()
            modified_sample[i] = baseline[i]
            
            # Calculate scores
            original_score = self._calculate_outlier_score(sample, training_data)
            modified_score = self._calculate_outlier_score(modified_sample, training_data)
            
            # SHAP-like contribution
            importance[i] = abs(original_score - modified_score)
        
        # Normalize
        if np.max(importance) > 0:
            importance = importance / np.max(importance)
        
        return importance
    
    def _local_outlier_importance(
        self,
        sample: npt.NDArray[np.floating],
        training_data: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Calculate feature importance based on local outlier factor."""
        importance = np.zeros(len(sample))
        
        if len(training_data) < 5:
            return importance
        
        # Find k nearest neighbors
        k = min(5, len(training_data))
        distances = np.linalg.norm(training_data - sample, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        nearest_neighbors = training_data[nearest_indices]
        
        # Calculate feature-wise deviations from neighbors
        neighbor_mean = np.mean(nearest_neighbors, axis=0)
        neighbor_std = np.std(nearest_neighbors, axis=0)
        
        # Calculate normalized deviations
        for i in range(len(sample)):
            if neighbor_std[i] > 1e-8:
                deviation = abs(sample[i] - neighbor_mean[i]) / neighbor_std[i]
                importance[i] = deviation
        
        # Normalize
        if np.max(importance) > 0:
            importance = importance / np.max(importance)
        
        return importance
    
    def _calculate_outlier_score(
        self,
        sample: npt.NDArray[np.floating],
        training_data: npt.NDArray[np.floating]
    ) -> float:
        """Calculate simple outlier score for a sample."""
        if len(training_data) == 0:
            return 0.0
        
        # Use average distance to k nearest neighbors
        k = min(5, len(training_data))
        distances = np.linalg.norm(training_data - sample, axis=1)
        k_nearest_distances = np.sort(distances)[:k]
        
        return np.mean(k_nearest_distances)
    
    def _get_top_contributing_features(
        self,
        feature_importance: Dict[str, float],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top contributing features sorted by importance."""
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:top_k]
    
    def _generate_explanation_text(
        self,
        sample: npt.NDArray[np.floating],
        is_anomaly: bool,
        anomaly_score: float,
        top_features: List[Tuple[str, float]],
        training_data: Optional[npt.NDArray[np.floating]]
    ) -> str:
        """Generate natural language explanation."""
        if is_anomaly:
            explanation = f"This sample is classified as an anomaly (score: {anomaly_score:.3f}). "
        else:
            explanation = f"This sample is classified as normal (score: {anomaly_score:.3f}). "
        
        if top_features:
            explanation += "The most contributing features are: "
            feature_descriptions = []
            
            for feature_name, importance in top_features[:3]:
                feature_descriptions.append(f"{feature_name} (importance: {importance:.3f})")
            
            explanation += ", ".join(feature_descriptions) + ". "
        
        if training_data is not None and is_anomaly:
            # Add context about how unusual the sample is
            distances = np.linalg.norm(training_data - sample, axis=1)
            min_distance = np.min(distances)
            avg_distance = np.mean(distances)
            
            if min_distance > avg_distance:
                explanation += f"This sample is unusually far from normal patterns (min distance: {min_distance:.3f}, avg: {avg_distance:.3f})."
        
        return explanation
    
    def _calculate_explanation_confidence(
        self,
        feature_importance: Dict[str, float],
        anomaly_score: float,
        training_data: Optional[npt.NDArray[np.floating]]
    ) -> float:
        """Calculate confidence in the explanation."""
        # Base confidence on feature importance distribution
        importance_values = list(feature_importance.values())
        
        if not importance_values:
            return 0.5
        
        # Higher confidence if there are clearly dominant features
        max_importance = max(importance_values)
        avg_importance = np.mean(importance_values)
        
        importance_confidence = max_importance / (avg_importance + 1e-8) if avg_importance > 0 else 1.0
        importance_confidence = min(importance_confidence / 3.0, 1.0)  # Normalize
        
        # Higher confidence for extreme anomaly scores
        score_confidence = min(abs(anomaly_score) / 2.0, 1.0)
        
        # Combine confidences
        overall_confidence = (importance_confidence + score_confidence) / 2
        
        return min(max(overall_confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95
    
    def _get_local_context(
        self,
        sample: npt.NDArray[np.floating],
        training_data: Optional[npt.NDArray[np.floating]]
    ) -> Dict[str, Any]:
        """Get local context information for the sample."""
        context = {}
        
        if training_data is not None and len(training_data) > 0:
            # Find nearest neighbors
            distances = np.linalg.norm(training_data - sample, axis=1)
            nearest_idx = np.argmin(distances)
            
            context.update({
                'nearest_neighbor_distance': float(distances[nearest_idx]),
                'average_distance_to_training': float(np.mean(distances)),
                'percentile_rank': float(np.percentile(distances, 95))
            })
        
        return context
    
    def _get_feature_name(self, feature_index: int) -> str:
        """Get feature name for a given index."""
        if self.feature_names and feature_index < len(self.feature_names):
            return self.feature_names[feature_index]
        return f"feature_{feature_index}"
    
    def _identify_anomaly_patterns(
        self,
        training_data: npt.NDArray[np.floating],
        detection_result: DetectionResult
    ) -> List[Dict[str, Any]]:
        """Identify common patterns in anomalous samples."""
        anomaly_indices = np.where(detection_result.predictions == 1)[0]
        
        if len(anomaly_indices) == 0:
            return []
        
        anomalous_samples = training_data[anomaly_indices]
        
        patterns = []
        
        # Pattern 1: Extreme values
        for i in range(training_data.shape[1]):
            feature_name = self._get_feature_name(i)
            feature_values = anomalous_samples[:, i]
            
            # Check for extreme high values
            high_threshold = np.percentile(training_data[:, i], 95)
            high_anomalies = np.sum(feature_values > high_threshold)
            
            if high_anomalies > len(anomalous_samples) * 0.3:  # 30% threshold
                patterns.append({
                    'type': 'extreme_high',
                    'feature': feature_name,
                    'description': f'{high_anomalies} anomalies have unusually high {feature_name} values',
                    'threshold': float(high_threshold),
                    'affected_samples': int(high_anomalies)
                })
            
            # Check for extreme low values
            low_threshold = np.percentile(training_data[:, i], 5)
            low_anomalies = np.sum(feature_values < low_threshold)
            
            if low_anomalies > len(anomalous_samples) * 0.3:
                patterns.append({
                    'type': 'extreme_low',
                    'feature': feature_name,
                    'description': f'{low_anomalies} anomalies have unusually low {feature_name} values',
                    'threshold': float(low_threshold),
                    'affected_samples': int(low_anomalies)
                })
        
        return patterns
    
    def _estimate_decision_boundaries(
        self,
        training_data: npt.NDArray[np.floating],
        detection_result: DetectionResult
    ) -> Optional[Dict[str, Any]]:
        """Estimate decision boundaries (simplified)."""
        if training_data.shape[1] > 10:  # Too high dimensional
            return None
        
        # Simple boundary estimation based on anomaly distribution
        anomaly_indices = np.where(detection_result.predictions == 1)[0]
        normal_indices = np.where(detection_result.predictions == 0)[0]
        
        if len(anomaly_indices) == 0 or len(normal_indices) == 0:
            return None
        
        anomalous_samples = training_data[anomaly_indices]
        normal_samples = training_data[normal_indices]
        
        boundaries = {}
        
        for i in range(training_data.shape[1]):
            feature_name = self._get_feature_name(i)
            
            normal_values = normal_samples[:, i]
            anomaly_values = anomalous_samples[:, i]
            
            boundaries[feature_name] = {
                'normal_range': [float(np.min(normal_values)), float(np.max(normal_values))],
                'anomaly_range': [float(np.min(anomaly_values)), float(np.max(anomaly_values))],
                'separation_quality': float(abs(np.mean(normal_values) - np.mean(anomaly_values)) / 
                                           (np.std(normal_values) + np.std(anomaly_values) + 1e-8))
            }
        
        return boundaries
    
    def _calculate_interpretability_score(
        self,
        detection_result: DetectionResult,
        global_importance: Dict[str, float]
    ) -> float:
        """Calculate overall model interpretability score."""
        # Factor 1: Feature importance distribution
        importance_values = list(global_importance.values())
        if not importance_values:
            importance_factor = 0.0
        else:
            # Higher score if few features dominate
            max_importance = max(importance_values)
            avg_importance = np.mean(importance_values)
            importance_factor = max_importance / (avg_importance + 1e-8)
            importance_factor = min(importance_factor / 5.0, 1.0)  # Normalize
        
        # Factor 2: Anomaly ratio (easier to interpret with clear separation)
        anomaly_ratio = detection_result.n_anomalies / detection_result.n_samples
        ratio_factor = 1.0 - abs(anomaly_ratio - 0.1)  # Penalize extreme ratios
        ratio_factor = max(ratio_factor, 0.0)
        
        # Factor 3: Algorithm interpretability
        algorithm_scores = {
            'iforest': 0.7,
            'lof': 0.8,
            'svm': 0.5,
            'statistical': 0.9,
            'timeseries': 0.8
        }
        algorithm_factor = algorithm_scores.get(detection_result.algorithm.lower(), 0.6)
        
        # Combine factors
        overall_score = (importance_factor + ratio_factor + algorithm_factor) / 3
        
        return min(max(overall_score, 0.0), 1.0)
    
    def _generate_global_summary(
        self,
        detection_result: DetectionResult,
        global_importance: Dict[str, float],
        anomaly_patterns: List[Dict[str, Any]]
    ) -> str:
        """Generate global model summary."""
        summary = f"Model: {detection_result.algorithm}\n"
        summary += f"Detected {detection_result.n_anomalies} anomalies out of {detection_result.n_samples} samples "
        summary += f"({detection_result.n_anomalies/detection_result.n_samples*100:.1f}%).\n\n"
        
        # Top features
        top_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_features:
            summary += "Most important features for anomaly detection:\n"
            for feature, importance in top_features:
                summary += f"  • {feature}: {importance:.3f}\n"
            summary += "\n"
        
        # Patterns
        if anomaly_patterns:
            summary += f"Identified {len(anomaly_patterns)} anomaly patterns:\n"
            for pattern in anomaly_patterns[:3]:  # Top 3 patterns
                summary += f"  • {pattern['description']}\n"
        
        return summary
    
    def _simple_clustering(
        self,
        data: npt.NDArray[np.floating],
        n_clusters: int
    ) -> Dict[int, List[int]]:
        """Simple clustering implementation without sklearn."""
        if len(data) <= n_clusters:
            return {i: [i] for i in range(len(data))}
        
        # Initialize centroids randomly
        centroids = data[np.random.choice(len(data), n_clusters, replace=False)]
        
        # Simple k-means iterations
        for _ in range(10):  # Max 10 iterations
            # Assign points to clusters
            clusters = {i: [] for i in range(n_clusters)}
            
            for point_idx, point in enumerate(data):
                distances = [np.linalg.norm(point - centroid) for centroid in centroids]
                closest_cluster = np.argmin(distances)
                clusters[closest_cluster].append(point_idx)
            
            # Update centroids
            new_centroids = []
            for cluster_id in range(n_clusters):
                if clusters[cluster_id]:
                    cluster_points = data[clusters[cluster_id]]
                    new_centroids.append(np.mean(cluster_points, axis=0))
                else:
                    new_centroids.append(centroids[cluster_id])  # Keep old centroid
            
            centroids = np.array(new_centroids)
        
        return clusters
    
    def _find_distinguishing_features(
        self,
        cluster_center: npt.NDArray[np.floating],
        training_data: npt.NDArray[np.floating]
    ) -> Dict[str, float]:
        """Find features that distinguish this cluster."""
        distinguishing = {}
        
        overall_mean = np.mean(training_data, axis=0)
        overall_std = np.std(training_data, axis=0)
        
        # Ensure cluster_center is 1D array
        if isinstance(cluster_center, np.ndarray) and cluster_center.ndim > 1:
            cluster_center = cluster_center.flatten()
        
        # Handle case where cluster_center might be a scalar
        if np.isscalar(cluster_center):
            cluster_center = np.array([cluster_center])
        
        for i in range(min(len(cluster_center), len(overall_mean), len(overall_std))):
            cluster_val = cluster_center[i]
            overall_val = overall_mean[i]
            std_val = overall_std[i]
            
            if std_val > 1e-8:  # Avoid division by zero
                z_score = abs(cluster_val - overall_val) / std_val
                if z_score > 1.5:  # Significant difference
                    feature_name = self._get_feature_name(i)
                    distinguishing[feature_name] = float(z_score)
        
        return distinguishing
    
    def _generate_cluster_explanation(
        self,
        cluster_id: int,
        cluster_size: int,
        distinguishing_features: Dict[str, float]
    ) -> str:
        """Generate explanation for anomaly cluster."""
        explanation = f"Cluster {cluster_id} contains {cluster_size} anomalous samples. "
        
        if distinguishing_features:
            top_features = sorted(distinguishing_features.items(), key=lambda x: x[1], reverse=True)[:2]
            feature_descriptions = [f"{name} (z-score: {score:.2f})" for name, score in top_features]
            explanation += f"Distinguished by unusual values in: {', '.join(feature_descriptions)}."
        else:
            explanation += "No clearly distinguishing features identified."
        
        return explanation