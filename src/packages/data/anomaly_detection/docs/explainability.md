# Explainability and Interpretability Guide

This guide covers model explainability and interpretability features in the Anomaly Detection package, helping you understand why models make specific predictions and build trust in your anomaly detection systems.

## Table of Contents

1. [Overview](#overview)
2. [Feature Importance Analysis](#feature-importance-analysis)
3. [SHAP Integration](#shap-integration)
4. [Local Explanations](#local-explanations)
5. [Global Explanations](#global-explanations)
6. [Algorithm-Specific Explanations](#algorithm-specific-explanations)
7. [Visualization Techniques](#visualization-techniques)
8. [Interactive Exploration](#interactive-exploration)
9. [Explanation APIs](#explanation-apis)
10. [Custom Explainers](#custom-explainers)
11. [Production Integration](#production-integration)
12. [Best Practices](#best-practices)

## Overview

Model explainability is crucial for understanding anomaly detection decisions, especially in critical applications where transparency and trust are essential.

### Types of Explanations

#### Global Explanations
- **Feature Importance**: Which features are most important overall
- **Model Behavior**: How the model generally behaves across the feature space
- **Decision Boundaries**: Where the model draws lines between normal and anomalous

#### Local Explanations
- **Instance-Level**: Why a specific data point was classified as anomalous
- **Feature Contributions**: How each feature contributed to the anomaly score
- **Counterfactuals**: What changes would make the instance normal

### Supported Algorithms

Different algorithms provide varying levels of explainability:

```python
from anomaly_detection.core.domain.services.explainability import ExplainabilityService
from anomaly_detection.core.application.services.explanation_service import ExplanationService

# Algorithms with native explainability support
EXPLAINABLE_ALGORITHMS = {
    'isolation_forest': ['feature_importance', 'path_length', 'isolation_paths'],
    'local_outlier_factor': ['local_reachability', 'neighbor_analysis'],
    'one_class_svm': ['support_vectors', 'decision_function'],
    'elliptic_envelope': ['mahalanobis_distance', 'covariance_analysis'],
    'copod': ['tail_probabilities', 'empirical_cdf'],
    'hbos': ['histogram_analysis', 'bin_contributions']
}
```

## Feature Importance Analysis

### Global Feature Importance

```python
from anomaly_detection.core.domain.services.explainability import GlobalExplainer
import pandas as pd
import numpy as np

class GlobalFeatureImportance:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.explainer = GlobalExplainer()
    
    def permutation_importance(self, n_repeats=10):
        """Calculate feature importance using permutation method."""
        baseline_scores = self.model.decision_function(self.data)
        baseline_anomalies = np.sum(baseline_scores < 0)
        
        importances = {}
        
        for feature_idx in range(self.data.shape[1]):
            scores_list = []
            
            for _ in range(n_repeats):
                # Create permuted data
                data_permuted = self.data.copy()
                np.random.shuffle(data_permuted[:, feature_idx])
                
                # Calculate scores with permuted feature
                permuted_scores = self.model.decision_function(data_permuted)
                permuted_anomalies = np.sum(permuted_scores < 0)
                
                # Calculate change in anomaly detection
                score_change = abs(baseline_anomalies - permuted_anomalies) / len(self.data)
                scores_list.append(score_change)
            
            importances[f'feature_{feature_idx}'] = {
                'importance': np.mean(scores_list),
                'std': np.std(scores_list)
            }
        
        return importances
    
    def ablation_importance(self):
        """Calculate feature importance using ablation method."""
        baseline_scores = self.model.decision_function(self.data)
        baseline_performance = np.mean(np.abs(baseline_scores))
        
        importances = {}
        
        for feature_idx in range(self.data.shape[1]):
            # Remove feature (set to mean)
            data_ablated = self.data.copy()
            data_ablated[:, feature_idx] = np.mean(data_ablated[:, feature_idx])
            
            # Calculate performance without feature
            ablated_scores = self.model.decision_function(data_ablated)
            ablated_performance = np.mean(np.abs(ablated_scores))
            
            # Importance is performance drop
            importance = abs(baseline_performance - ablated_performance)
            importances[f'feature_{feature_idx}'] = importance
        
        return importances

# Example usage
data = pd.read_csv('sensor_data.csv')
X = data[['temperature', 'humidity', 'pressure', 'vibration']].values

from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

# Calculate global feature importance
explainer = GlobalFeatureImportance(model, X)
perm_importance = explainer.permutation_importance()
ablation_importance = explainer.ablation_importance()

print("Permutation Importance:")
for feature, metrics in perm_importance.items():
    print(f"{feature}: {metrics['importance']:.4f} ± {metrics['std']:.4f}")
```

### Algorithm-Specific Feature Importance

```python
class AlgorithmSpecificImportance:
    @staticmethod
    def isolation_forest_importance(model, feature_names):
        """Extract feature importance from Isolation Forest."""
        # Average path length contribution per feature
        importances = {}
        
        for tree in model.estimators_:
            for node in range(tree.tree_.node_count):
                if tree.tree_.feature[node] != -2:  # Not a leaf
                    feature_idx = tree.tree_.feature[node]
                    feature_name = feature_names[feature_idx]
                    
                    if feature_name not in importances:
                        importances[feature_name] = 0
                    
                    # Weight by samples in node
                    samples = tree.tree_.n_node_samples[node]
                    importances[feature_name] += samples
        
        # Normalize
        total = sum(importances.values())
        return {k: v/total for k, v in importances.items()}
    
    @staticmethod
    def lof_importance(model, X, feature_names):
        """Extract feature importance from LOF using neighbor analysis."""
        from sklearn.neighbors import NearestNeighbors
        
        # Analyze which features contribute most to nearest neighbor distances
        nn = NearestNeighbors(n_neighbors=model.n_neighbors, metric='euclidean')
        nn.fit(X)
        
        distances, indices = nn.kneighbors(X)
        
        importances = {}
        for i, feature_name in enumerate(feature_names):
            # Calculate feature-specific distances
            feature_distances = []
            for j in range(len(X)):
                neighbors = indices[j]
                feature_diffs = [abs(X[j, i] - X[neighbor, i]) for neighbor in neighbors]
                feature_distances.extend(feature_diffs)
            
            importances[feature_name] = np.mean(feature_distances)
        
        # Normalize
        total = sum(importances.values())
        return {k: v/total for k, v in importances.items()}

# Example usage
feature_names = ['temperature', 'humidity', 'pressure', 'vibration']

# For Isolation Forest
if hasattr(model, 'estimators_'):
    if_importance = AlgorithmSpecificImportance.isolation_forest_importance(
        model, feature_names
    )
    print("Isolation Forest Feature Importance:")
    for feature, importance in sorted(if_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
```

## SHAP Integration

### SHAP Explainer Setup

```python
import shap
import pandas as pd
import numpy as np
from anomaly_detection.core.domain.services.explainability import SHAPExplainer

class AnomalyDetectionSHAP:
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Set up appropriate SHAP explainer based on model type."""
        model_name = type(self.model).__name__.lower()
        
        if 'isolationforest' in model_name:
            # Use TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        elif 'svm' in model_name:
            # Use KernelExplainer for SVM
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self.explainer = shap.KernelExplainer(
                self.model.decision_function, background
            )
        else:
            # Use KernelExplainer as fallback
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self.explainer = shap.KernelExplainer(
                self.model.decision_function, background
            )
    
    def explain_instance(self, instance):
        """Generate SHAP explanation for a single instance."""
        if isinstance(instance, pd.Series):
            instance = instance.values.reshape(1, -1)
        elif len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(instance)
        
        return {
            'shap_values': shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            'base_value': self.explainer.expected_value,
            'feature_values': instance[0],
            'feature_names': self.feature_names
        }
    
    def explain_batch(self, instances, max_evals=1000):
        """Generate SHAP explanations for multiple instances."""
        if isinstance(instances, pd.DataFrame):
            instances = instances.values
        
        # Limit evaluations for performance
        if len(instances) > max_evals:
            indices = np.random.choice(len(instances), max_evals, replace=False)
            instances = instances[indices]
        
        shap_values = self.explainer.shap_values(instances)
        
        return {
            'shap_values': shap_values,
            'base_value': self.explainer.expected_value,
            'feature_values': instances,
            'feature_names': self.feature_names
        }
    
    def summary_plot(self, instances, max_display=20):
        """Create SHAP summary plot."""
        explanations = self.explain_batch(instances)
        
        shap.summary_plot(
            explanations['shap_values'],
            explanations['feature_values'],
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
    
    def waterfall_plot(self, instance, show_data=True):
        """Create SHAP waterfall plot for single instance."""
        explanation = self.explain_instance(instance)
        
        shap.waterfall_plot(
            shap.Explanation(
                values=explanation['shap_values'],
                base_values=explanation['base_value'],
                data=explanation['feature_values'] if show_data else None,
                feature_names=self.feature_names
            ),
            show=False
        )

# Example usage
data = pd.read_csv('sensor_data.csv')
feature_names = ['temperature', 'humidity', 'pressure', 'vibration']
X = data[feature_names].values

# Train model
from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

# Create SHAP explainer
shap_explainer = AnomalyDetectionSHAP(model, X, feature_names)

# Explain a specific anomalous instance
anomaly_scores = model.decision_function(X)
anomaly_idx = np.argmin(anomaly_scores)  # Most anomalous
anomalous_instance = X[anomaly_idx]

explanation = shap_explainer.explain_instance(anomalous_instance)
print(f"Base value (expected): {explanation['base_value']:.4f}")
print(f"Prediction: {model.decision_function(anomalous_instance.reshape(1, -1))[0]:.4f}")
print("\nFeature contributions:")
for i, (feature, shap_val, actual_val) in enumerate(zip(
    feature_names, explanation['shap_values'], explanation['feature_values']
)):
    print(f"{feature}: {actual_val:.2f} -> SHAP: {shap_val:.4f}")
```

### Advanced SHAP Analysis

```python
class AdvancedSHAPAnalysis:
    def __init__(self, shap_explainer):
        self.shap_explainer = shap_explainer
    
    def feature_interaction_analysis(self, instances, feature_pairs):
        """Analyze feature interactions using SHAP interaction values."""
        if hasattr(self.shap_explainer.explainer, 'shap_interaction_values'):
            interaction_values = self.shap_explainer.explainer.shap_interaction_values(instances)
            
            interactions = {}
            for i, j in feature_pairs:
                feature_i = self.shap_explainer.feature_names[i]
                feature_j = self.shap_explainer.feature_names[j]
                
                # Average interaction effect
                interaction_effect = np.mean(interaction_values[:, i, j])
                interactions[f"{feature_i}_x_{feature_j}"] = interaction_effect
            
            return interactions
        else:
            return "Interaction values not supported for this explainer type"
    
    def dependence_analysis(self, instances, feature_idx, interaction_feature=None):
        """Analyze how predictions depend on a specific feature."""
        explanations = self.shap_explainer.explain_batch(instances)
        
        feature_values = explanations['feature_values'][:, feature_idx]
        shap_values = explanations['shap_values'][:, feature_idx]
        
        # Sort by feature values for better visualization
        sorted_indices = np.argsort(feature_values)
        
        dependence_data = {
            'feature_values': feature_values[sorted_indices],
            'shap_values': shap_values[sorted_indices],
            'feature_name': self.shap_explainer.feature_names[feature_idx]
        }
        
        if interaction_feature is not None:
            dependence_data['interaction_values'] = feature_values[sorted_indices, interaction_feature]
            dependence_data['interaction_name'] = self.shap_explainer.feature_names[interaction_feature]
        
        return dependence_data
    
    def anomaly_explanation_ranking(self, instances, top_k=10):
        """Rank instances by explanation complexity/unusualness."""
        explanations = self.shap_explainer.explain_batch(instances)
        
        # Calculate explanation metrics
        explanation_scores = []
        for i in range(len(instances)):
            shap_vals = explanations['shap_values'][i]
            
            # Metrics for ranking
            total_impact = np.sum(np.abs(shap_vals))
            max_impact = np.max(np.abs(shap_vals))
            num_important_features = np.sum(np.abs(shap_vals) > 0.1)
            entropy = -np.sum(np.abs(shap_vals) * np.log(np.abs(shap_vals) + 1e-10))
            
            explanation_scores.append({
                'index': i,
                'total_impact': total_impact,
                'max_impact': max_impact,
                'num_important_features': num_important_features,
                'entropy': entropy,
                'anomaly_score': self.shap_explainer.model.decision_function(instances[i:i+1])[0]
            })
        
        # Sort by total impact (most complex explanations first)
        explanation_scores.sort(key=lambda x: x['total_impact'], reverse=True)
        
        return explanation_scores[:top_k]

# Example usage
advanced_analysis = AdvancedSHAPAnalysis(shap_explainer)

# Analyze feature interactions
feature_pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]  # temperature-humidity, etc.
interactions = advanced_analysis.feature_interaction_analysis(X[:100], feature_pairs)
print("Feature Interactions:")
for pair, interaction in interactions.items():
    print(f"{pair}: {interaction:.4f}")

# Find most complex anomaly explanations
complex_anomalies = advanced_analysis.anomaly_explanation_ranking(X, top_k=5)
print("\nMost Complex Anomaly Explanations:")
for i, item in enumerate(complex_anomalies, 1):
    print(f"{i}. Index {item['index']}: Total Impact={item['total_impact']:.4f}, "
          f"Features={item['num_important_features']}, Score={item['anomaly_score']:.4f}")
```

## Local Explanations

### Instance-Level Explanations

```python
class LocalExplainer:
    def __init__(self, model, X_reference, feature_names=None):
        self.model = model
        self.X_reference = X_reference
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_reference.shape[1])]
    
    def explain_anomaly(self, instance, method='lime', n_samples=1000):
        """Explain why an instance is considered anomalous."""
        if isinstance(instance, pd.Series):
            instance = instance.values
        elif len(instance.shape) > 1:
            instance = instance.flatten()
        
        if method == 'lime':
            return self._lime_explanation(instance, n_samples)
        elif method == 'anchor':
            return self._anchor_explanation(instance)
        elif method == 'counterfactual':
            return self._counterfactual_explanation(instance)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
    
    def _lime_explanation(self, instance, n_samples):
        """Generate LIME explanation."""
        from sklearn.linear_model import LinearRegression
        
        # Generate perturbed samples around the instance
        perturbed_samples = self._generate_perturbations(instance, n_samples)
        
        # Get model predictions for perturbed samples
        predictions = self.model.decision_function(perturbed_samples)
        
        # Calculate distances (weights) from original instance
        distances = np.sqrt(np.sum((perturbed_samples - instance) ** 2, axis=1))
        weights = np.exp(-distances / np.mean(distances))  # Exponential kernel
        
        # Fit local linear model
        local_model = LinearRegression()
        local_model.fit(perturbed_samples, predictions, sample_weight=weights)
        
        # Extract feature importance from local model
        importance = local_model.coef_
        
        return {
            'method': 'lime',
            'feature_importance': dict(zip(self.feature_names, importance)),
            'local_prediction': local_model.predict(instance.reshape(1, -1))[0],
            'actual_prediction': self.model.decision_function(instance.reshape(1, -1))[0],
            'r2_score': local_model.score(perturbed_samples, predictions, sample_weight=weights)
        }
    
    def _generate_perturbations(self, instance, n_samples):
        """Generate perturbations around an instance."""
        # Calculate feature statistics from reference data
        feature_means = np.mean(self.X_reference, axis=0)
        feature_stds = np.std(self.X_reference, axis=0)
        
        # Generate perturbations using normal distribution
        perturbations = np.random.normal(0, feature_stds * 0.1, (n_samples, len(instance)))
        perturbed_samples = instance + perturbations
        
        return perturbed_samples
    
    def _counterfactual_explanation(self, instance, max_iterations=1000):
        """Find counterfactual explanation (what changes would make it normal)."""
        from scipy.optimize import minimize
        
        original_score = self.model.decision_function(instance.reshape(1, -1))[0]
        
        def objective(x):
            # We want to maximize the anomaly score (make it less anomalous)
            score = self.model.decision_function(x.reshape(1, -1))[0]
            # Add penalty for large changes
            distance_penalty = np.sum((x - instance) ** 2)
            return -score + 0.1 * distance_penalty
        
        # Optimize to find counterfactual
        result = minimize(
            objective,
            instance,
            method='L-BFGS-B',
            options={'maxiter': max_iterations}
        )
        
        if result.success:
            counterfactual = result.x
            cf_score = self.model.decision_function(counterfactual.reshape(1, -1))[0]
            
            # Calculate feature changes
            changes = counterfactual - instance
            relative_changes = changes / (np.std(self.X_reference, axis=0) + 1e-10)
            
            return {
                'method': 'counterfactual',
                'original_instance': instance,
                'counterfactual_instance': counterfactual,
                'original_score': original_score,
                'counterfactual_score': cf_score,
                'changes': dict(zip(self.feature_names, changes)),
                'relative_changes': dict(zip(self.feature_names, relative_changes)),
                'success': True
            }
        else:
            return {
                'method': 'counterfactual',
                'success': False,
                'message': 'Could not find counterfactual explanation'
            }

# Example usage
local_explainer = LocalExplainer(model, X, feature_names)

# Find most anomalous instance
anomaly_scores = model.decision_function(X)
most_anomalous_idx = np.argmin(anomaly_scores)
anomalous_instance = X[most_anomalous_idx]

# Generate different types of explanations
lime_explanation = local_explainer.explain_anomaly(anomalous_instance, method='lime')
print("LIME Explanation:")
print(f"Local R² score: {lime_explanation['r2_score']:.4f}")
print("Feature importance:")
for feature, importance in sorted(lime_explanation['feature_importance'].items(), 
                                 key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feature}: {importance:.4f}")

print("\n" + "="*50 + "\n")

# Counterfactual explanation
cf_explanation = local_explainer.explain_anomaly(anomalous_instance, method='counterfactual')
if cf_explanation['success']:
    print("Counterfactual Explanation:")
    print(f"Original score: {cf_explanation['original_score']:.4f}")
    print(f"Counterfactual score: {cf_explanation['counterfactual_score']:.4f}")
    print("Required changes:")
    for feature, change in cf_explanation['changes'].items():
        relative_change = cf_explanation['relative_changes'][feature]
        print(f"  {feature}: {change:+.4f} ({relative_change:+.2f} std)")
```

## Global Explanations

### Model Behavior Analysis

```python
class GlobalExplainer:
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
    
    def decision_boundary_analysis(self, feature_pairs, resolution=50):
        """Analyze decision boundaries between feature pairs."""
        boundaries = {}
        
        for i, j in feature_pairs:
            feature_i_name = self.feature_names[i]
            feature_j_name = self.feature_names[j]
            
            # Create mesh grid
            feature_i_min, feature_i_max = self.X_train[:, i].min(), self.X_train[:, i].max()
            feature_j_min, feature_j_max = self.X_train[:, j].min(), self.X_train[:, j].max()
            
            # Expand range slightly
            i_range = feature_i_max - feature_i_min
            j_range = feature_j_max - feature_j_min
            feature_i_min -= 0.1 * i_range
            feature_i_max += 0.1 * i_range
            feature_j_min -= 0.1 * j_range
            feature_j_max += 0.1 * j_range
            
            xx, yy = np.meshgrid(
                np.linspace(feature_i_min, feature_i_max, resolution),
                np.linspace(feature_j_min, feature_j_max, resolution)
            )
            
            # Create prediction grid
            grid_points = np.zeros((resolution * resolution, self.X_train.shape[1]))
            
            # Set other features to their mean values
            for k in range(self.X_train.shape[1]):
                if k not in [i, j]:
                    grid_points[:, k] = np.mean(self.X_train[:, k])
            
            # Set the two features of interest
            grid_points[:, i] = xx.ravel()
            grid_points[:, j] = yy.ravel()
            
            # Get predictions
            Z = self.model.decision_function(grid_points)
            Z = Z.reshape(xx.shape)
            
            boundaries[f"{feature_i_name}_vs_{feature_j_name}"] = {
                'feature_i': feature_i_name,
                'feature_j': feature_j_name,
                'xx': xx,
                'yy': yy,
                'Z': Z,
                'feature_i_values': self.X_train[:, i],
                'feature_j_values': self.X_train[:, j]
            }
        
        return boundaries
    
    def sensitivity_analysis(self, feature_ranges=None, n_steps=20):
        """Analyze model sensitivity to feature changes."""
        if feature_ranges is None:
            # Use training data ranges
            feature_ranges = {
                name: (self.X_train[:, i].min(), self.X_train[:, i].max())
                for i, name in enumerate(self.feature_names)
            }
        
        sensitivity_results = {}
        
        # Use median instance as baseline
        baseline_instance = np.median(self.X_train, axis=0)
        baseline_score = self.model.decision_function(baseline_instance.reshape(1, -1))[0]
        
        for feature_idx, feature_name in enumerate(self.feature_names):
            min_val, max_val = feature_ranges[feature_name]
            
            # Create range of values for this feature
            values = np.linspace(min_val, max_val, n_steps)
            scores = []
            
            for value in values:
                # Create instance with modified feature
                modified_instance = baseline_instance.copy()
                modified_instance[feature_idx] = value
                
                # Get prediction
                score = self.model.decision_function(modified_instance.reshape(1, -1))[0]
                scores.append(score)
            
            # Calculate sensitivity metrics
            score_range = max(scores) - min(scores)
            score_std = np.std(scores)
            
            # Find optimal value (highest score = least anomalous)
            optimal_idx = np.argmax(scores)
            optimal_value = values[optimal_idx]
            optimal_score = scores[optimal_idx]
            
            sensitivity_results[feature_name] = {
                'values': values,
                'scores': scores,
                'baseline_value': baseline_instance[feature_idx],
                'baseline_score': baseline_score,
                'score_range': score_range,
                'score_std': score_std,
                'optimal_value': optimal_value,
                'optimal_score': optimal_score,
                'sensitivity': score_range / (max_val - min_val)  # Score change per unit change
            }
        
        return sensitivity_results
    
    def model_coverage_analysis(self, n_samples=10000):
        """Analyze what regions of feature space the model considers normal vs anomalous."""
        # Generate random samples across feature space
        feature_mins = np.min(self.X_train, axis=0)
        feature_maxs = np.max(self.X_train, axis=0)
        
        # Expand ranges
        ranges = feature_maxs - feature_mins
        expanded_mins = feature_mins - 0.2 * ranges
        expanded_maxs = feature_maxs + 0.2 * ranges
        
        # Generate uniform random samples
        random_samples = np.random.uniform(
            expanded_mins, expanded_maxs, (n_samples, self.X_train.shape[1])
        )
        
        # Get predictions
        scores = self.model.decision_function(random_samples)
        anomaly_predictions = scores < 0
        
        # Analyze coverage
        total_samples = len(random_samples)
        anomalous_samples = np.sum(anomaly_predictions)
        normal_samples = total_samples - anomalous_samples
        
        # Analyze by feature ranges
        feature_analysis = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_values = random_samples[:, i]
            
            # Divide into quantiles
            quantiles = np.percentile(feature_values, [0, 25, 50, 75, 100])
            
            quantile_analysis = {}
            for q in range(4):
                mask = (feature_values >= quantiles[q]) & (feature_values <= quantiles[q+1])
                quantile_anomalies = np.sum(anomaly_predictions[mask])
                quantile_total = np.sum(mask)
                
                quantile_analysis[f'Q{q+1}'] = {
                    'range': (quantiles[q], quantiles[q+1]),
                    'total_samples': quantile_total,
                    'anomalous_samples': quantile_anomalies,
                    'anomaly_rate': quantile_anomalies / quantile_total if quantile_total > 0 else 0
                }
            
            feature_analysis[feature_name] = quantile_analysis
        
        return {
            'total_samples': total_samples,
            'normal_samples': normal_samples,
            'anomalous_samples': anomalous_samples,
            'global_anomaly_rate': anomalous_samples / total_samples,
            'feature_analysis': feature_analysis
        }

# Example usage
global_explainer = GlobalExplainer(model, X, feature_names)

# Analyze decision boundaries
feature_pairs = [(0, 1), (0, 2), (1, 2)]  # temperature-humidity, temperature-pressure, etc.
boundaries = global_explainer.decision_boundary_analysis(feature_pairs)

print("Decision Boundary Analysis:")
for boundary_name, boundary_data in boundaries.items():
    print(f"\n{boundary_name}:")
    print(f"  Score range: {boundary_data['Z'].min():.4f} to {boundary_data['Z'].max():.4f}")
    anomaly_threshold = 0  # Isolation Forest threshold
    anomalous_region = np.sum(boundary_data['Z'] < anomaly_threshold) / boundary_data['Z'].size
    print(f"  Anomalous region: {anomalous_region:.2%}")

# Sensitivity analysis
sensitivity = global_explainer.sensitivity_analysis()
print("\nSensitivity Analysis:")
for feature, data in sensitivity.items():
    print(f"\n{feature}:")
    print(f"  Sensitivity: {data['sensitivity']:.4f}")
    print(f"  Score range: {data['score_range']:.4f}")
    print(f"  Optimal value: {data['optimal_value']:.4f} (score: {data['optimal_score']:.4f})")

# Model coverage analysis
coverage = global_explainer.model_coverage_analysis()
print(f"\nModel Coverage Analysis:")
print(f"Global anomaly rate: {coverage['global_anomaly_rate']:.2%}")
print("\nPer-feature anomaly rates by quantile:")
for feature, analysis in coverage['feature_analysis'].items():
    print(f"\n{feature}:")
    for quantile, data in analysis.items():
        print(f"  {quantile}: {data['anomaly_rate']:.2%} "
              f"(range: {data['range'][0]:.2f} - {data['range'][1]:.2f})")
```

## Visualization Techniques

### Interactive Explanation Dashboard

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class ExplanationVisualizer:
    def __init__(self, model, X, feature_names, predictions=None):
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.predictions = predictions or model.decision_function(X)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def feature_importance_plot(self, importance_dict, title="Feature Importance"):
        """Create feature importance visualization."""
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        # Sort by importance
        sorted_pairs = sorted(zip(features, importances), key=lambda x: abs(x[1]), reverse=True)
        features, importances = zip(*sorted_pairs)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if imp < 0 else 'blue' for imp in importances]
        bars = ax.barh(features, importances, color=colors, alpha=0.7)
        
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, importances):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{imp:.3f}', ha='left' if width >= 0 else 'right', va='center')
        
        plt.tight_layout()
        return fig
    
    def shap_summary_plot(self, shap_values, max_display=20):
        """Create SHAP summary plot."""
        import shap
        
        # Ensure we have the right shape
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 0]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values[:min(len(shap_values), 1000)],  # Limit for performance
            self.X[:min(len(self.X), 1000)],
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        return plt.gcf()
    
    def anomaly_score_distribution(self):
        """Plot distribution of anomaly scores."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(self.predictions, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', label='Decision Boundary')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Anomaly Scores')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(self.predictions, vert=True)
        ax2.axhline(y=0, color='red', linestyle='--', label='Decision Boundary')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title('Anomaly Score Box Plot')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def feature_vs_anomaly_scatter(self, feature_idx, highlight_anomalies=True):
        """Scatter plot of feature value vs anomaly score."""
        feature_name = self.feature_names[feature_idx]
        feature_values = self.X[:, feature_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if highlight_anomalies:
            # Color points by anomaly/normal
            colors = ['red' if score < 0 else 'blue' for score in self.predictions]
            labels = ['Anomaly' if score < 0 else 'Normal' for score in self.predictions]
            
            # Create scatter plot with different colors
            for color, label in [('red', 'Anomaly'), ('blue', 'Normal')]:
                mask = [c == color for c in colors]
                ax.scatter(feature_values[mask], self.predictions[mask], 
                          c=color, label=label, alpha=0.6)
        else:
            ax.scatter(feature_values, self.predictions, alpha=0.6)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Decision Boundary')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Anomaly Score')
        ax.set_title(f'{feature_name} vs Anomaly Score')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def interactive_feature_exploration(self):
        """Create interactive Plotly dashboard for feature exploration."""
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Correlation', 'Anomaly Score Distribution', 
                           'Feature vs Anomaly Score', 'Feature Importance'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Feature correlation heatmap (simplified for 2 features)
        if len(self.feature_names) >= 2:
            corr_matrix = np.corrcoef(self.X[:, :min(len(self.feature_names), 5)].T)
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    x=self.feature_names[:min(len(self.feature_names), 5)],
                    y=self.feature_names[:min(len(self.feature_names), 5)],
                    colorscale='RdBu',
                    zmid=0
                ),
                row=1, col=1
            )
        
        # Anomaly score distribution
        fig.add_trace(
            go.Histogram(x=self.predictions, nbinsx=50, name='Anomaly Scores'),
            row=1, col=2
        )
        
        # Feature vs Anomaly Score (for first feature)
        colors = ['red' if score < 0 else 'blue' for score in self.predictions]
        fig.add_trace(
            go.Scatter(
                x=self.X[:, 0],
                y=self.predictions,
                mode='markers',
                marker=dict(color=colors, opacity=0.6),
                name=f'{self.feature_names[0]} vs Score',
                text=[f'{self.feature_names[0]}: {val:.2f}<br>Score: {score:.3f}' 
                      for val, score in zip(self.X[:, 0], self.predictions)],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Feature importance (placeholder - would need actual importance values)
        importance_values = np.random.random(len(self.feature_names))  # Placeholder
        fig.add_trace(
            go.Bar(
                x=self.feature_names,
                y=importance_values,
                name='Feature Importance'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Anomaly Detection Explanation Dashboard",
            showlegend=False
        )
        
        # Add decision boundary line to anomaly score plots
        fig.add_hline(y=0, line_dash="dash", line_color="red", 
                     annotation_text="Decision Boundary", row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        return fig
    
    def create_explanation_report(self, instance_idx, local_explanation, 
                                 global_explanation=None, output_file=None):
        """Create comprehensive explanation report for a specific instance."""
        instance = self.X[instance_idx]
        score = self.predictions[instance_idx]
        
        # Create multi-page figure
        fig = plt.figure(figsize=(16, 20))
        
        # Page 1: Instance Overview
        gs1 = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, top=0.95, bottom=0.7)
        
        # Instance details
        ax1 = fig.add_subplot(gs1[0, :])
        ax1.axis('off')
        ax1.text(0.5, 0.5, f'Instance {instance_idx} Analysis\n'
                           f'Anomaly Score: {score:.4f}\n'
                           f'Classification: {"Anomaly" if score < 0 else "Normal"}',
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Feature values
        ax2 = fig.add_subplot(gs1[1, 0])
        feature_vals = instance
        bars = ax2.bar(range(len(self.feature_names)), feature_vals, alpha=0.7)
        ax2.set_xticks(range(len(self.feature_names)))
        ax2.set_xticklabels(self.feature_names, rotation=45)
        ax2.set_title('Feature Values')
        ax2.set_ylabel('Value')
        
        # Feature importance from local explanation
        ax3 = fig.add_subplot(gs1[1, 1])
        if 'feature_importance' in local_explanation:
            importance = local_explanation['feature_importance']
            features = list(importance.keys())
            importances = list(importance.values())
            colors = ['red' if imp < 0 else 'blue' for imp in importances]
            ax3.barh(features, importances, color=colors, alpha=0.7)
            ax3.set_title('Local Feature Importance')
            ax3.set_xlabel('Importance')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Comparison with normal instances
        ax4 = fig.add_subplot(gs1[2, :])
        normal_mask = self.predictions >= 0
        if np.any(normal_mask):
            normal_instances = self.X[normal_mask]
            normal_means = np.mean(normal_instances, axis=0)
            normal_stds = np.std(normal_instances, axis=0)
            
            x_pos = np.arange(len(self.feature_names))
            width = 0.35
            
            ax4.bar(x_pos - width/2, normal_means, width, label='Normal Mean', 
                   alpha=0.7, yerr=normal_stds, capsize=5)
            ax4.bar(x_pos + width/2, instance, width, label='This Instance', alpha=0.7)
            
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(self.feature_names, rotation=45)
            ax4.set_title('Comparison with Normal Instances')
            ax4.set_ylabel('Value')
            ax4.legend()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return fig

# Example usage
visualizer = ExplanationVisualizer(model, X, feature_names, model.decision_function(X))

# Create various visualizations
fig1 = visualizer.anomaly_score_distribution()
plt.show()

fig2 = visualizer.feature_vs_anomaly_scatter(0)  # First feature
plt.show()

# Create interactive dashboard
interactive_fig = visualizer.interactive_feature_exploration()
interactive_fig.show()

# Generate comprehensive report for most anomalous instance
most_anomalous_idx = np.argmin(model.decision_function(X))
local_exp = local_explainer.explain_anomaly(X[most_anomalous_idx], method='lime')
report_fig = visualizer.create_explanation_report(most_anomalous_idx, local_exp)
plt.show()
```

## Production Integration

### Real-time Explanation Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import json

app = FastAPI(title="Anomaly Detection Explanation Service")

class ExplanationRequest(BaseModel):
    instance: List[float]
    feature_names: Optional[List[str]] = None
    explanation_type: str = "lime"  # lime, shap, counterfactual
    include_global: bool = False

class ExplanationResponse(BaseModel):
    instance_id: str
    anomaly_score: float
    is_anomaly: bool
    explanation_type: str
    feature_importance: Dict[str, float]
    confidence: float
    global_context: Optional[Dict] = None

class ExplanationService:
    def __init__(self, model, reference_data, feature_names):
        self.model = model
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.local_explainer = LocalExplainer(model, reference_data, feature_names)
        self.global_explainer = GlobalExplainer(model, reference_data, feature_names)
        
        # Cache for global explanations
        self._global_cache = {}
    
    def explain_instance(self, instance, explanation_type='lime', include_global=False):
        """Generate explanation for a single instance."""
        instance = np.array(instance).reshape(1, -1)
        
        # Get prediction
        anomaly_score = self.model.decision_function(instance)[0]
        is_anomaly = anomaly_score < 0
        
        # Generate local explanation
        local_exp = self.local_explainer.explain_anomaly(
            instance.flatten(), method=explanation_type
        )
        
        response = {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'explanation_type': explanation_type,
            'feature_importance': local_exp.get('feature_importance', {}),
            'confidence': float(abs(anomaly_score))  # Use absolute score as confidence
        }
        
        # Add global context if requested
        if include_global:
            if 'global_sensitivity' not in self._global_cache:
                self._global_cache['global_sensitivity'] = self.global_explainer.sensitivity_analysis()
            
            response['global_context'] = {
                'feature_sensitivity': {
                    feature: data['sensitivity'] 
                    for feature, data in self._global_cache['global_sensitivity'].items()
                }
            }
        
        return response

# Initialize service
explanation_service = None

@app.on_event("startup")
async def startup_event():
    global explanation_service
    # Initialize with your trained model and data
    # explanation_service = ExplanationService(model, X, feature_names)
    pass

@app.post("/explain", response_model=ExplanationResponse)
async def explain_anomaly(request: ExplanationRequest):
    """Generate explanation for an anomaly detection result."""
    try:
        if explanation_service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Generate unique instance ID
        import hashlib
        instance_str = str(request.instance)
        instance_id = hashlib.md5(instance_str.encode()).hexdigest()[:8]
        
        # Get explanation
        explanation = explanation_service.explain_instance(
            request.instance,
            explanation_type=request.explanation_type,
            include_global=request.include_global
        )
        
        return ExplanationResponse(
            instance_id=instance_id,
            **explanation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "explanation-service"}

@app.get("/algorithms")
async def get_supported_algorithms():
    """Get list of supported explanation algorithms."""
    return {
        "explanation_types": ["lime", "shap", "counterfactual"],
        "model_types": ["isolation_forest", "local_outlier_factor", "one_class_svm"]
    }
```

### Batch Explanation Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

class BatchExplanationProcessor:
    def __init__(self, model, reference_data, feature_names, max_workers=4):
        self.model = model
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize explainers
        self.local_explainer = LocalExplainer(model, reference_data, feature_names)
        self.global_explainer = GlobalExplainer(model, reference_data, feature_names)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, instances, explanation_types=['lime'], 
                     output_file=None, progress_callback=None):
        """Process batch of instances for explanation."""
        
        async def process_batch_async():
            tasks = []
            
            for i, instance in enumerate(instances):
                for exp_type in explanation_types:
                    task = asyncio.create_task(
                        self._process_single_instance(i, instance, exp_type)
                    )
                    tasks.append(task)
            
            results = []
            completed = 0
            total = len(tasks)
            
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total)
                
                if completed % 10 == 0:
                    self.logger.info(f"Processed {completed}/{total} explanations")
            
            return results
        
        # Run async batch processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(process_batch_async())
        finally:
            loop.close()
        
        # Organize results
        organized_results = self._organize_results(results)
        
        # Save to file if requested
        if output_file:
            self._save_results(organized_results, output_file)
        
        return organized_results
    
    async def _process_single_instance(self, instance_id, instance, explanation_type):
        """Process single instance asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run explanation in thread pool to avoid blocking
        result = await loop.run_in_executor(
            self.executor,
            self._explain_instance_sync,
            instance_id, instance, explanation_type
        )
        
        return result
    
    def _explain_instance_sync(self, instance_id, instance, explanation_type):
        """Synchronous explanation processing."""
        try:
            # Get anomaly score
            instance_array = np.array(instance).reshape(1, -1)
            anomaly_score = self.model.decision_function(instance_array)[0]
            
            # Generate explanation
            explanation = self.local_explainer.explain_anomaly(
                instance, method=explanation_type
            )
            
            return {
                'instance_id': instance_id,
                'instance': instance,
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(anomaly_score < 0),
                'explanation_type': explanation_type,
                'explanation': explanation,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            return {
                'instance_id': instance_id,
                'error': str(e),
                'explanation_type': explanation_type,
                'timestamp': pd.Timestamp.now().isoformat()
            }
    
    def _organize_results(self, results):
        """Organize results by instance and explanation type."""
        organized = {}
        
        for result in results:
            instance_id = result['instance_id']
            exp_type = result['explanation_type']
            
            if instance_id not in organized:
                organized[instance_id] = {
                    'instance': result.get('instance'),
                    'anomaly_score': result.get('anomaly_score'),
                    'is_anomaly': result.get('is_anomaly'),
                    'explanations': {}
                }
            
            if 'error' in result:
                organized[instance_id]['explanations'][exp_type] = {
                    'error': result['error']
                }
            else:
                organized[instance_id]['explanations'][exp_type] = result['explanation']
        
        return organized
    
    def _save_results(self, results, output_file):
        """Save results to file."""
        if output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif output_file.endswith('.csv'):
            # Flatten for CSV
            flattened = []
            for instance_id, data in results.items():
                for exp_type, explanation in data['explanations'].items():
                    row = {
                        'instance_id': instance_id,
                        'anomaly_score': data['anomaly_score'],
                        'is_anomaly': data['is_anomaly'],
                        'explanation_type': exp_type
                    }
                    
                    if 'error' not in explanation:
                        if 'feature_importance' in explanation:
                            for feature, importance in explanation['feature_importance'].items():
                                row[f'importance_{feature}'] = importance
                    
                    flattened.append(row)
            
            pd.DataFrame(flattened).to_csv(output_file, index=False)

# Example usage
batch_processor = BatchExplanationProcessor(model, X, feature_names, max_workers=4)

# Process batch of anomalous instances
anomaly_indices = np.where(model.decision_function(X) < 0)[0][:50]  # First 50 anomalies
anomalous_instances = X[anomaly_indices]

def progress_callback(completed, total):
    print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

results = batch_processor.process_batch(
    anomalous_instances,
    explanation_types=['lime', 'counterfactual'],
    output_file='batch_explanations.json',
    progress_callback=progress_callback
)

print(f"Processed {len(results)} instances")
print(f"Example result keys: {list(results.keys())[:3]}")
```

## Best Practices

### 1. Choosing Explanation Methods

```python
def choose_explanation_method(model_type, instance_count, feature_count, time_budget):
    """Recommend explanation method based on constraints."""
    
    recommendations = {
        'method': None,
        'reason': '',
        'alternatives': []
    }
    
    # For tree-based models, prefer SHAP TreeExplainer
    if 'isolation' in model_type.lower() or 'forest' in model_type.lower():
        if time_budget > 10:  # seconds
            recommendations['method'] = 'shap'
            recommendations['reason'] = 'Tree-based model with sufficient time budget'
            recommendations['alternatives'] = ['lime', 'permutation_importance']
        else:
            recommendations['method'] = 'permutation_importance'
            recommendations['reason'] = 'Tree-based model with tight time budget'
            recommendations['alternatives'] = ['lime']
    
    # For high-dimensional data, use feature selection first
    elif feature_count > 100:
        recommendations['method'] = 'lime'
        recommendations['reason'] = 'High-dimensional data - LIME handles feature selection'
        recommendations['alternatives'] = ['permutation_importance']
    
    # For batch processing, prefer faster methods
    elif instance_count > 1000:
        recommendations['method'] = 'permutation_importance'
        recommendations['reason'] = 'Large batch - faster global method preferred'
        recommendations['alternatives'] = ['lime']
    
    # Default to LIME for general cases
    else:
        recommendations['method'] = 'lime'
        recommendations['reason'] = 'General case - LIME provides good local explanations'
        recommendations['alternatives'] = ['shap', 'counterfactual']
    
    return recommendations
```

### 2. Explanation Validation

```python
class ExplanationValidator:
    def __init__(self, model, reference_data):
        self.model = model
        self.reference_data = reference_data
    
    def validate_lime_explanation(self, instance, explanation, tolerance=0.1):
        """Validate LIME explanation quality."""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'metrics': {}
        }
        
        # Check R² score
        r2_score = explanation.get('r2_score', 0)
        validation_results['metrics']['r2_score'] = r2_score
        
        if r2_score < 0.5:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Low R² score ({r2_score:.3f}) - local model may not be reliable"
            )
        
        # Check prediction consistency
        local_pred = explanation.get('local_prediction', 0)
        actual_pred = explanation.get('actual_prediction', 0)
        pred_diff = abs(local_pred - actual_pred)
        validation_results['metrics']['prediction_difference'] = pred_diff
        
        if pred_diff > tolerance:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Large prediction difference ({pred_diff:.3f}) between local and actual model"
            )
        
        return validation_results
    
    def validate_feature_importance_stability(self, instance, method='lime', n_runs=5):
        """Check stability of feature importance across multiple runs."""
        importances_list = []
        
        for _ in range(n_runs):
            if method == 'lime':
                explainer = LocalExplainer(self.model, self.reference_data)
                explanation = explainer.explain_anomaly(instance, method='lime')
                importances = explanation['feature_importance']
            else:
                # Add other methods as needed
                continue
            
            importances_list.append(list(importances.values()))
        
        # Calculate stability metrics
        importances_array = np.array(importances_list)
        mean_importance = np.mean(importances_array, axis=0)
        std_importance = np.std(importances_array, axis=0)
        cv_importance = std_importance / (np.abs(mean_importance) + 1e-10)
        
        return {
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'coefficient_of_variation': cv_importance,
            'max_cv': np.max(cv_importance),
            'is_stable': np.max(cv_importance) < 0.5  # Threshold for stability
        }
```

### 3. Performance Optimization

```python
class OptimizedExplainer:
    def __init__(self, model, reference_data, feature_names):
        self.model = model
        self.reference_data = reference_data
        self.feature_names = feature_names
        
        # Pre-compute expensive operations
        self._precompute_statistics()
        
        # Caching
        from functools import lru_cache
        self._cached_predictions = lru_cache(maxsize=1000)(self._predict_single)
    
    def _precompute_statistics(self):
        """Pre-compute statistics for faster explanations."""
        self.feature_means = np.mean(self.reference_data, axis=0)
        self.feature_stds = np.std(self.reference_data, axis=0)
        self.feature_ranges = np.ptp(self.reference_data, axis=0)  # peak-to-peak
        
        # Pre-compute reference predictions for faster baselines
        self.reference_predictions = self.model.decision_function(self.reference_data)
        self.baseline_score = np.median(self.reference_predictions)
    
    def _predict_single(self, instance_tuple):
        """Cached prediction function."""
        instance = np.array(instance_tuple).reshape(1, -1)
        return self.model.decision_function(instance)[0]
    
    def fast_feature_importance(self, instance, method='perturbation', n_samples=100):
        """Fast feature importance calculation."""
        if method == 'perturbation':
            return self._fast_perturbation_importance(instance, n_samples)
        elif method == 'gradient':
            return self._gradient_based_importance(instance)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _fast_perturbation_importance(self, instance, n_samples):
        """Fast perturbation-based importance."""
        baseline_score = self._predict_single(tuple(instance))
        importances = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Generate perturbations for this feature only
            perturbations = np.random.normal(
                instance[i], self.feature_stds[i] * 0.1, n_samples
            )
            
            scores = []
            for pert_val in perturbations:
                pert_instance = instance.copy()
                pert_instance[i] = pert_val
                score = self._predict_single(tuple(pert_instance))
                scores.append(score)
            
            # Importance is correlation between feature change and score change
            feature_changes = perturbations - instance[i]
            score_changes = np.array(scores) - baseline_score
            
            correlation = np.corrcoef(feature_changes, score_changes)[0, 1]
            importances[feature_name] = correlation if not np.isnan(correlation) else 0
        
        return importances
    
    def _gradient_based_importance(self, instance):
        """Approximate gradient-based importance for non-differentiable models."""
        importances = {}
        baseline_score = self._predict_single(tuple(instance))
        
        epsilon = 1e-6
        
        for i, feature_name in enumerate(self.feature_names):
            # Forward difference
            instance_plus = instance.copy()
            instance_plus[i] += epsilon
            score_plus = self._predict_single(tuple(instance_plus))
            
            # Backward difference
            instance_minus = instance.copy()
            instance_minus[i] -= epsilon
            score_minus = self._predict_single(tuple(instance_minus))
            
            # Central difference approximation
            gradient = (score_plus - score_minus) / (2 * epsilon)
            importances[feature_name] = gradient
        
        return importances

# Example usage
optimized_explainer = OptimizedExplainer(model, X, feature_names)

# Fast explanation for anomalous instance
anomalous_idx = np.argmin(model.decision_function(X))
anomalous_instance = X[anomalous_idx]

# Time comparison
import time

start_time = time.time()
fast_importance = optimized_explainer.fast_feature_importance(
    anomalous_instance, method='perturbation', n_samples=50
)
fast_time = time.time() - start_time

print(f"Fast explanation time: {fast_time:.3f} seconds")
print("Fast importance:")
for feature, importance in sorted(fast_importance.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feature}: {importance:.4f}")
```

This comprehensive explainability guide provides the tools and techniques needed to understand and trust anomaly detection models in production environments. The combination of local and global explanations, interactive visualizations, and production-ready APIs ensures that model decisions are transparent and actionable.