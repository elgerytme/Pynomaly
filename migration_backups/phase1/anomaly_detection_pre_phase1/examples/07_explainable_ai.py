#!/usr/bin/env python3
"""
Explainable AI Examples for Anomaly Detection Package

This example demonstrates model interpretability and explainability techniques including:
- SHAP (SHapley Additive exPlanations) integration
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis
- Counterfactual explanations
- Anomaly reasoning and justification
- Interactive visualizations for explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

# SHAP for model explanations
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP available for global and local explanations")
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

# LIME for local explanations
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
    print("LIME available for local explanations")
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")

# Additional visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")

# ML libraries
try:
    from sklearn.datasets import make_classification, make_blobs
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")

# Import anomaly detection components
try:
    from anomaly_detection import DetectionService, EnsembleService
    from anomaly_detection.domain.entities.detection_result import DetectionResult
except ImportError:
    print("Please install the anomaly_detection package first:")
    print("pip install -e .")
    exit(1)


class ExplainableAnomalyDetector:
    """Explainable anomaly detection with interpretability features."""
    
    def __init__(self, algorithm: str = 'iforest'):
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        # Initialize SHAP explainer
        self.shap_explainer = None
        self.lime_explainer = None
        
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit the anomaly detection model."""
        
        # Store feature names
        if feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit model
        if self.algorithm == 'iforest':
            self.model = IsolationForest(contamination=0.1, random_state=42)
        elif self.algorithm == 'ocsvm':
            self.model = OneClassSVM(gamma='scale')
        elif self.algorithm == 'lof':
            self.model = LocalOutlierFactor(contamination=0.1, novelty=True)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        # Initialize explainers
        self._initialize_explainers(X_scaled)
        
        return self
    
    def _initialize_explainers(self, X_scaled: np.ndarray):
        """Initialize SHAP and LIME explainers."""
        
        # SHAP explainer
        if SHAP_AVAILABLE:
            try:
                if self.algorithm == 'iforest':
                    # Use TreeExplainer for tree-based models
                    self.shap_explainer = shap.TreeExplainer(self.model)
                else:
                    # Use KernelExplainer for other models
                    def model_predict(X):
                        return self.model.decision_function(X)
                    
                    # Use a subset for background
                    background = shap.sample(X_scaled, min(100, len(X_scaled)))
                    self.shap_explainer = shap.KernelExplainer(model_predict, background)
                    
            except Exception as e:
                print(f"Warning: Could not initialize SHAP explainer: {e}")
                self.shap_explainer = None
        
        # LIME explainer
        if LIME_AVAILABLE:
            try:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_scaled,
                    feature_names=self.feature_names,
                    class_names=['Normal', 'Anomaly'],
                    mode='classification',
                    discretize_continuous=True
                )
            except Exception as e:
                print(f"Warning: Could not initialize LIME explainer: {e}")
                self.lime_explainer = None
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies and return scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if self.algorithm == 'lof':
            predictions = self.model.predict(X_scaled)
            scores = self.model.decision_function(X_scaled)
        else:
            predictions = self.model.predict(X_scaled)
            scores = self.model.decision_function(X_scaled)
        
        return predictions, scores
    
    def explain_global(self, X: np.ndarray, max_display: int = 10) -> Dict[str, Any]:
        """Generate global explanations using SHAP."""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return self._fallback_global_explanation(X)
        
        try:
            X_scaled = self.scaler.transform(X)
            
            # Calculate SHAP values
            if self.algorithm == 'iforest':
                shap_values = self.shap_explainer.shap_values(X_scaled)
            else:
                shap_values = self.shap_explainer.shap_values(X_scaled, nsamples=100)
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(0)
            
            # Sort by importance
            importance_indices = np.argsort(feature_importance)[::-1]
            
            explanation = {
                'method': 'SHAP',
                'feature_importance': {
                    self.feature_names[i]: float(feature_importance[i])
                    for i in importance_indices[:max_display]
                },
                'shap_values': shap_values,
                'feature_names': self.feature_names,
                'mean_abs_shap': feature_importance
            }
            
            return explanation
            
        except Exception as e:
            print(f"Warning: SHAP explanation failed: {e}")
            return self._fallback_global_explanation(X)
    
    def explain_local(self, X: np.ndarray, instance_idx: int = 0, 
                     method: str = 'both') -> Dict[str, Any]:
        """Generate local explanations for a specific instance."""
        
        if instance_idx >= len(X):
            raise ValueError(f"Instance index {instance_idx} out of range")
        
        instance = X[instance_idx:instance_idx+1]
        X_scaled = self.scaler.transform(instance)
        
        explanations = {
            'instance_index': instance_idx,
            'instance_values': X[instance_idx],
            'prediction': None,
            'score': None
        }
        
        # Get prediction for the instance
        pred, score = self.predict(instance)
        explanations['prediction'] = pred[0]
        explanations['score'] = score[0]
        
        # SHAP explanation
        if (method in ['shap', 'both'] and SHAP_AVAILABLE and 
            self.shap_explainer is not None):
            try:
                if self.algorithm == 'iforest':
                    shap_values = self.shap_explainer.shap_values(X_scaled)
                else:
                    shap_values = self.shap_explainer.shap_values(X_scaled, nsamples=100)
                
                explanations['shap'] = {
                    'values': shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                    'expected_value': self.shap_explainer.expected_value,
                    'feature_contributions': {
                        self.feature_names[i]: float(shap_values[0][i])
                        for i in range(len(self.feature_names))
                    }
                }
            except Exception as e:
                print(f"Warning: SHAP local explanation failed: {e}")
                explanations['shap'] = None
        
        # LIME explanation
        if (method in ['lime', 'both'] and LIME_AVAILABLE and 
            self.lime_explainer is not None):
            try:
                def predict_fn(X):
                    # LIME expects probabilities
                    _, scores = self.predict(self.scaler.inverse_transform(X))
                    # Convert scores to probabilities (simple transformation)
                    probs = np.column_stack([1 - (scores + 1) / 2, (scores + 1) / 2])
                    return probs
                
                lime_explanation = self.lime_explainer.explain_instance(
                    X_scaled[0], predict_fn, num_features=len(self.feature_names)
                )
                
                explanations['lime'] = {
                    'explanation': lime_explanation,
                    'feature_contributions': dict(lime_explanation.as_list()),
                    'local_pred': lime_explanation.predict_proba
                }
            except Exception as e:
                print(f"Warning: LIME local explanation failed: {e}")
                explanations['lime'] = None
        
        return explanations
    
    def _fallback_global_explanation(self, X: np.ndarray) -> Dict[str, Any]:
        """Fallback global explanation when SHAP is not available."""
        
        # Use simple statistical approach
        X_scaled = self.scaler.transform(X)
        predictions, scores = self.predict(X)
        
        # Find anomalies
        anomaly_mask = predictions == -1
        normal_mask = predictions == 1
        
        if np.sum(anomaly_mask) == 0:
            return {'method': 'Statistical', 'error': 'No anomalies detected'}
        
        # Calculate feature statistics for normal vs anomaly
        normal_data = X[normal_mask]
        anomaly_data = X[anomaly_mask]
        
        feature_importance = {}
        
        for i, feature_name in enumerate(self.feature_names):
            if len(normal_data) > 0 and len(anomaly_data) > 0:
                # Calculate difference in means (normalized by std)
                normal_mean = np.mean(normal_data[:, i])
                anomaly_mean = np.mean(anomaly_data[:, i])
                combined_std = np.std(X[:, i])
                
                if combined_std > 0:
                    importance = abs(anomaly_mean - normal_mean) / combined_std
                else:
                    importance = 0
                
                feature_importance[feature_name] = importance
        
        return {
            'method': 'Statistical',
            'feature_importance': feature_importance,
            'normal_stats': {
                self.feature_names[i]: {
                    'mean': np.mean(normal_data[:, i]),
                    'std': np.std(normal_data[:, i])
                } for i in range(len(self.feature_names))
            } if len(normal_data) > 0 else {},
            'anomaly_stats': {
                self.feature_names[i]: {
                    'mean': np.mean(anomaly_data[:, i]),
                    'std': np.std(anomaly_data[:, i])
                } for i in range(len(self.feature_names))
            } if len(anomaly_data) > 0 else {}
        }
    
    def generate_counterfactual(self, X: np.ndarray, instance_idx: int,
                              n_iterations: int = 1000, learning_rate: float = 0.01) -> Dict[str, Any]:
        """Generate counterfactual explanation for an anomaly."""
        
        if instance_idx >= len(X):
            raise ValueError(f"Instance index {instance_idx} out of range")
        
        original_instance = X[instance_idx].copy()
        original_pred, original_score = self.predict(original_instance.reshape(1, -1))
        
        if original_pred[0] != -1:
            return {
                'error': 'Instance is not an anomaly',
                'original_prediction': original_pred[0],
                'original_score': original_score[0]
            }
        
        # Initialize counterfactual with original instance
        counterfactual = original_instance.copy()
        
        # Gradient-free optimization (simple random search)
        best_counterfactual = counterfactual.copy()
        best_score = original_score[0]
        
        for iteration in range(n_iterations):
            # Add small random perturbation
            noise = np.random.normal(0, 0.1, size=counterfactual.shape)
            candidate = counterfactual + noise * learning_rate
            
            # Predict
            pred, score = self.predict(candidate.reshape(1, -1))
            
            # Check if we found a normal instance
            if pred[0] == 1:
                best_counterfactual = candidate.copy()
                best_score = score[0]
                break
            
            # Update if score improved (closer to normal)
            if score[0] > best_score:
                best_counterfactual = candidate.copy()
                best_score = score[0]
                counterfactual = candidate.copy()
        
        # Calculate changes
        changes = best_counterfactual - original_instance
        
        return {
            'original_instance': original_instance,
            'counterfactual_instance': best_counterfactual,
            'changes': changes,
            'feature_changes': {
                self.feature_names[i]: {
                    'original': float(original_instance[i]),
                    'counterfactual': float(best_counterfactual[i]),
                    'change': float(changes[i])
                } for i in range(len(self.feature_names))
            },
            'original_score': float(original_score[0]),
            'counterfactual_score': float(best_score),
            'success': best_score > original_score[0]
        }


class ExplanationVisualizer:
    """Visualization utilities for explainable AI."""
    
    def __init__(self):
        self.colors = {
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#708090'
        }
    
    def plot_global_feature_importance(self, explanation: Dict[str, Any], 
                                     title: str = "Global Feature Importance"):
        """Plot global feature importance."""
        
        if 'feature_importance' not in explanation:
            print("No feature importance found in explanation")
            return
        
        importance_dict = explanation['feature_importance']
        features = list(importance_dict.keys())
        importance = list(importance_dict.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        features = [features[i] for i in sorted_indices]
        importance = [importance[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(features)), importance, 
                       color=self.colors['positive'], alpha=0.7)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance)):
            plt.text(val + max(importance) * 0.01, i, f'{val:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_shap_summary(self, explanation: Dict[str, Any]):
        """Plot SHAP summary plot."""
        
        if not SHAP_AVAILABLE or 'shap_values' not in explanation:
            print("SHAP values not available")
            return
        
        try:
            shap.summary_plot(
                explanation['shap_values'],
                feature_names=explanation['feature_names'],
                show=False
            )
            plt.title("SHAP Summary Plot")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not create SHAP summary plot: {e}")
    
    def plot_local_explanation(self, explanation: Dict[str, Any],
                             title: str = "Local Explanation"):
        """Plot local explanation for a single instance."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # SHAP explanation
        if explanation.get('shap') is not None:
            shap_contrib = explanation['shap']['feature_contributions']
            features = list(shap_contrib.keys())
            values = list(shap_contrib.values())
            
            colors = [self.colors['positive'] if v >= 0 else self.colors['negative'] 
                     for v in values]
            
            axes[0].barh(range(len(features)), values, color=colors, alpha=0.7)
            axes[0].set_yticks(range(len(features)))
            axes[0].set_yticklabels(features)
            axes[0].set_xlabel('SHAP Value')
            axes[0].set_title('SHAP Local Explanation')
            axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[0].grid(True, axis='x', alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'SHAP explanation\nnot available', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('SHAP Local Explanation')
        
        # LIME explanation
        if explanation.get('lime') is not None:
            lime_contrib = explanation['lime']['feature_contributions']
            features = list(lime_contrib.keys())
            values = list(lime_contrib.values())
            
            colors = [self.colors['positive'] if v >= 0 else self.colors['negative'] 
                     for v in values]
            
            axes[1].barh(range(len(features)), values, color=colors, alpha=0.7)
            axes[1].set_yticks(range(len(features)))
            axes[1].set_yticklabels(features)
            axes[1].set_xlabel('LIME Value')
            axes[1].set_title('LIME Local Explanation')
            axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[1].grid(True, axis='x', alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'LIME explanation\nnot available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('LIME Local Explanation')
        
        plt.suptitle(f"{title}\nPrediction: {explanation['prediction']}, "
                    f"Score: {explanation['score']:.3f}")
        plt.tight_layout()
        plt.show()
    
    def plot_counterfactual(self, counterfactual: Dict[str, Any]):
        """Plot counterfactual explanation."""
        
        if 'error' in counterfactual:
            print(f"Counterfactual error: {counterfactual['error']}")
            return
        
        changes = counterfactual['feature_changes']
        features = list(changes.keys())
        
        original_values = [changes[f]['original'] for f in features]
        counterfactual_values = [changes[f]['counterfactual'] for f in features]
        change_values = [changes[f]['change'] for f in features]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original vs Counterfactual values
        x_pos = np.arange(len(features))
        width = 0.35
        
        axes[0].bar(x_pos - width/2, original_values, width, 
                   label='Original', alpha=0.7, color=self.colors['negative'])
        axes[0].bar(x_pos + width/2, counterfactual_values, width,
                   label='Counterfactual', alpha=0.7, color=self.colors['positive'])
        
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Values')
        axes[0].set_title('Original vs Counterfactual Values')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(features, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Changes
        colors = [self.colors['positive'] if v >= 0 else self.colors['negative'] 
                 for v in change_values]
        
        axes[1].bar(x_pos, change_values, color=colors, alpha=0.7)
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Change')
        axes[1].set_title('Required Changes')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(features, rotation=45, ha='right')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Counterfactual Explanation\n"
                    f"Score: {counterfactual['original_score']:.3f} → "
                    f"{counterfactual['counterfactual_score']:.3f}")
        plt.tight_layout()
        plt.show()
    
    def create_interactive_explanation(self, explanation: Dict[str, Any]):
        """Create interactive explanation using Plotly."""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for interactive explanations")
            return
        
        if 'feature_importance' not in explanation:
            print("No feature importance found")
            return
        
        importance_dict = explanation['feature_importance']
        features = list(importance_dict.keys())
        importance = list(importance_dict.values())
        
        # Create interactive bar plot
        fig = go.Figure(data=[
            go.Bar(
                y=features,
                x=importance,
                orientation='h',
                marker_color=['darkgreen' if x >= 0 else 'darkred' for x in importance],
                text=[f'{x:.3f}' for x in importance],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Interactive Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(features) * 30),
            showlegend=False
        )
        
        fig.show()


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 10, 
                          anomaly_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate synthetic data with interpretable anomalies."""
    
    np.random.seed(42)
    
    # Generate normal data
    n_normal = int(n_samples * (1 - anomaly_fraction))
    n_anomalies = n_samples - n_normal
    
    # Create feature names with meaningful names
    feature_names = [
        'temperature', 'pressure', 'flow_rate', 'vibration', 'power_consumption',
        'efficiency', 'rpm', 'torque', 'current', 'voltage'
    ][:n_features]
    
    if len(feature_names) < n_features:
        feature_names.extend([f'feature_{i}' for i in range(len(feature_names), n_features)])
    
    # Normal data - correlated features
    correlation_matrix = np.eye(n_features)
    for i in range(n_features - 1):
        correlation_matrix[i, i+1] = 0.3
        correlation_matrix[i+1, i] = 0.3
    
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=correlation_matrix,
        size=n_normal
    )
    
    # Anomalous data - different patterns
    anomalous_data = []
    
    # Type 1: Global outliers (25% of anomalies)
    n_global = max(1, n_anomalies // 4)
    global_outliers = np.random.multivariate_normal(
        mean=np.full(n_features, 4),  # Far from normal
        cov=np.eye(n_features),
        size=n_global
    )
    anomalous_data.append(global_outliers)
    
    # Type 2: Feature-specific anomalies (50% of anomalies)
    n_feature_specific = max(1, n_anomalies // 2)
    feature_anomalies = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=correlation_matrix,
        size=n_feature_specific
    )
    # Make specific features anomalous
    feature_anomalies[:, 0] += 5  # temperature anomaly
    feature_anomalies[:, 3] += 4  # vibration anomaly
    anomalous_data.append(feature_anomalies)
    
    # Type 3: Contextual anomalies (remaining anomalies)
    n_contextual = n_anomalies - n_global - n_feature_specific
    if n_contextual > 0:
        contextual_anomalies = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=correlation_matrix * 3,  # Higher variance
            size=n_contextual
        )
        anomalous_data.append(contextual_anomalies)
    
    # Combine all data
    X = np.vstack([normal_data] + anomalous_data)
    y = np.hstack([
        np.ones(n_normal),
        np.full(n_anomalies, -1)
    ])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y, feature_names


def example_1_shap_explanations():
    """Example 1: SHAP-based global and local explanations."""
    print("\n" + "="*60)
    print("Example 1: SHAP-Based Explanations")
    print("="*60)
    
    # Generate synthetic data
    X, y_true, feature_names = generate_synthetic_data(n_samples=800, n_features=8)
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True anomaly rate: {np.mean(y_true == -1):.2%}")
    
    # Initialize explainable detector
    detector = ExplainableAnomalyDetector(algorithm='iforest')
    detector.fit(X, feature_names)
    
    # Make predictions
    predictions, scores = detector.predict(X)
    
    print(f"Detected anomaly rate: {np.mean(predictions == -1):.2%}")
    
    # Global explanation
    print("\n1.1 Global Feature Importance Analysis")
    print("-" * 40)
    
    global_explanation = detector.explain_global(X)
    
    print(f"Explanation method: {global_explanation['method']}")
    if 'feature_importance' in global_explanation:
        print("\nTop 5 most important features:")
        importance_items = list(global_explanation['feature_importance'].items())[:5]
        for feature, importance in importance_items:
            print(f"  {feature}: {importance:.4f}")
    
    # Visualize global importance
    visualizer = ExplanationVisualizer()
    visualizer.plot_global_feature_importance(
        global_explanation, 
        "Global Feature Importance (SHAP)"
    )
    
    # SHAP summary plot
    if global_explanation['method'] == 'SHAP':
        visualizer.plot_shap_summary(global_explanation)
    
    # Local explanations for specific instances
    print("\n1.2 Local Explanations")
    print("-" * 25)
    
    # Find some anomalies to explain
    anomaly_indices = np.where(predictions == -1)[0][:3]
    
    for i, idx in enumerate(anomaly_indices):
        print(f"\nExplaining anomaly {i+1} (index {idx}):")
        
        local_explanation = detector.explain_local(X, idx, method='both')
        
        print(f"  Prediction: {local_explanation['prediction']}")
        print(f"  Score: {local_explanation['score']:.3f}")
        
        if local_explanation.get('shap') is not None:
            shap_contrib = local_explanation['shap']['feature_contributions']
            top_features = sorted(shap_contrib.items(), 
                                key=lambda x: abs(x[1]), reverse=True)[:3]
            print("  Top SHAP contributions:")
            for feature, contribution in top_features:
                print(f"    {feature}: {contribution:+.3f}")
        
        # Visualize local explanation
        visualizer.plot_local_explanation(
            local_explanation, 
            f"Local Explanation - Anomaly {i+1}"
        )


def example_2_lime_explanations():
    """Example 2: LIME-based local explanations."""
    print("\n" + "="*60)
    print("Example 2: LIME-Based Local Explanations")
    print("="*60)
    
    if not LIME_AVAILABLE:
        print("LIME not available. Skipping LIME examples.")
        return
    
    # Generate synthetic data
    X, y_true, feature_names = generate_synthetic_data(n_samples=500, n_features=6)
    
    # Initialize explainable detector with OCSVM (works well with LIME)
    detector = ExplainableAnomalyDetector(algorithm='ocsvm')
    detector.fit(X, feature_names)
    
    predictions, scores = detector.predict(X)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Detected {np.sum(predictions == -1)} anomalies")
    
    # Find interesting instances to explain
    anomaly_indices = np.where(predictions == -1)[0]
    normal_indices = np.where(predictions == 1)[0]
    
    instances_to_explain = [
        ('Anomaly', anomaly_indices[0] if len(anomaly_indices) > 0 else 0),
        ('Normal', normal_indices[0] if len(normal_indices) > 0 else 1),
        ('Anomaly', anomaly_indices[1] if len(anomaly_indices) > 1 else 0)
    ]
    
    print("\n2.1 LIME Local Explanations")
    print("-" * 30)
    
    for instance_type, idx in instances_to_explain:
        print(f"\nExplaining {instance_type} instance (index {idx}):")
        
        local_explanation = detector.explain_local(X, idx, method='lime')
        
        print(f"  Prediction: {local_explanation['prediction']}")
        print(f"  Score: {local_explanation['score']:.3f}")
        
        if local_explanation.get('lime') is not None:
            lime_contrib = local_explanation['lime']['feature_contributions']
            
            print("  LIME feature contributions:")
            for feature, contribution in lime_contrib.items():
                print(f"    {feature}: {contribution:+.3f}")
            
            # Visualize
            visualizer = ExplanationVisualizer()
            visualizer.plot_local_explanation(
                local_explanation,
                f"LIME Explanation - {instance_type} (Index {idx})"
            )
        else:
            print("  LIME explanation failed")


def example_3_counterfactual_explanations():
    """Example 3: Counterfactual explanations."""
    print("\n" + "="*60)
    print("Example 3: Counterfactual Explanations")
    print("="*60)
    
    # Generate synthetic data with clear decision boundaries
    X, y_true, feature_names = generate_synthetic_data(n_samples=400, n_features=5)
    
    # Initialize detector
    detector = ExplainableAnomalyDetector(algorithm='iforest')
    detector.fit(X, feature_names)
    
    predictions, scores = detector.predict(X)
    
    # Find anomalies to generate counterfactuals for
    anomaly_indices = np.where(predictions == -1)[0][:3]
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Found {len(anomaly_indices)} anomalies to explain")
    
    visualizer = ExplanationVisualizer()
    
    print("\n3.1 Generating Counterfactual Explanations")
    print("-" * 45)
    
    for i, idx in enumerate(anomaly_indices):
        print(f"\nCounterfactual for anomaly {i+1} (index {idx}):")
        
        # Generate counterfactual
        counterfactual = detector.generate_counterfactual(
            X, idx, n_iterations=1000, learning_rate=0.01
        )
        
        if 'error' in counterfactual:
            print(f"  Error: {counterfactual['error']}")
            continue
        
        print(f"  Original score: {counterfactual['original_score']:.3f}")
        print(f"  Counterfactual score: {counterfactual['counterfactual_score']:.3f}")
        print(f"  Success: {counterfactual['success']}")
        
        # Show top changes required
        changes = counterfactual['feature_changes']
        sorted_changes = sorted(changes.items(), 
                              key=lambda x: abs(x[1]['change']), reverse=True)
        
        print("  Top 3 required changes:")
        for feature, change_info in sorted_changes[:3]:
            print(f"    {feature}: {change_info['original']:.3f} → "
                  f"{change_info['counterfactual']:.3f} "
                  f"(Δ {change_info['change']:+.3f})")
        
        # Visualize counterfactual
        visualizer.plot_counterfactual(counterfactual)


def example_4_ensemble_explanations():
    """Example 4: Explaining ensemble model decisions."""
    print("\n" + "="*60)
    print("Example 4: Ensemble Model Explanations")
    print("="*60)
    
    # Generate synthetic data
    X, y_true, feature_names = generate_synthetic_data(n_samples=600, n_features=8)
    
    # Compare explanations from different algorithms
    algorithms = ['iforest', 'ocsvm', 'lof']
    explanations = {}
    predictions_dict = {}
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print("\n4.1 Individual Algorithm Explanations")
    print("-" * 40)
    
    for algorithm in algorithms:
        print(f"\nAnalyzing {algorithm.upper()}...")
        
        detector = ExplainableAnomalyDetector(algorithm=algorithm)
        detector.fit(X, feature_names)
        
        predictions, scores = detector.predict(X)
        predictions_dict[algorithm] = predictions
        
        # Global explanation
        global_explanation = detector.explain_global(X)
        explanations[algorithm] = global_explanation
        
        print(f"  Detected {np.sum(predictions == -1)} anomalies")
        print(f"  Method: {global_explanation['method']}")
        
        if 'feature_importance' in global_explanation:
            top_features = list(global_explanation['feature_importance'].items())[:3]
            print("  Top 3 features:")
            for feature, importance in top_features:
                print(f"    {feature}: {importance:.4f}")
    
    # Compare feature importance across algorithms
    print("\n4.2 Cross-Algorithm Feature Importance Comparison")
    print("-" * 50)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    feature_comparison = {}
    for feature in feature_names:
        feature_comparison[feature] = []
        for algorithm in algorithms:
            if 'feature_importance' in explanations[algorithm]:
                importance = explanations[algorithm]['feature_importance'].get(feature, 0)
            else:
                importance = 0
            feature_comparison[feature].append(importance)
    
    # Plot grouped bar chart
    x = np.arange(len(feature_names))
    width = 0.25
    
    for i, algorithm in enumerate(algorithms):
        importances = [feature_comparison[feature][i] for feature in feature_names]
        ax.bar(x + i * width, importances, width, label=algorithm.upper(), alpha=0.8)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance Score')
    ax.set_title('Feature Importance Comparison Across Algorithms')
    ax.set_xticks(x + width)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Agreement analysis
    print("\n4.3 Algorithm Agreement Analysis")
    print("-" * 35)
    
    # Calculate agreement between algorithms
    agreement_matrix = np.zeros((len(algorithms), len(algorithms)))
    
    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms):
            pred1 = predictions_dict[alg1]
            pred2 = predictions_dict[alg2]
            agreement = np.mean(pred1 == pred2)
            agreement_matrix[i, j] = agreement
    
    # Plot agreement heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(agreement_matrix, 
                xticklabels=[alg.upper() for alg in algorithms],
                yticklabels=[alg.upper() for alg in algorithms],
                annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Algorithm Agreement Matrix')
    plt.ylabel('Algorithm')
    plt.xlabel('Algorithm')
    plt.tight_layout()
    plt.show()
    
    # Find consensus anomalies
    consensus_anomalies = np.ones(len(X), dtype=bool)
    for algorithm in algorithms:
        consensus_anomalies &= (predictions_dict[algorithm] == -1)
    
    print(f"Consensus anomalies (all algorithms agree): {np.sum(consensus_anomalies)}")
    print(f"Consensus rate: {np.mean(consensus_anomalies):.2%}")


def example_5_interactive_explanations():
    """Example 5: Interactive explanations and dashboards."""
    print("\n" + "="*60)
    print("Example 5: Interactive Explanations")
    print("="*60)
    
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping interactive examples.")
        return
    
    # Generate synthetic data
    X, y_true, feature_names = generate_synthetic_data(n_samples=500, n_features=6)
    
    # Initialize detector
    detector = ExplainableAnomalyDetector(algorithm='iforest')
    detector.fit(X, feature_names)
    
    predictions, scores = detector.predict(X)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Detected {np.sum(predictions == -1)} anomalies")
    
    # Global explanation
    global_explanation = detector.explain_global(X)
    
    # Create interactive visualizations
    visualizer = ExplanationVisualizer()
    
    print("\n5.1 Interactive Feature Importance")
    print("-" * 35)
    visualizer.create_interactive_explanation(global_explanation)
    
    # Interactive anomaly scatter plot
    print("\n5.2 Interactive Anomaly Detection Results")
    print("-" * 45)
    
    # Create scatter plot with first two features
    if X.shape[1] >= 2:
        fig = go.Figure()
        
        # Normal points
        normal_mask = predictions == 1
        fig.add_trace(go.Scatter(
            x=X[normal_mask, 0],
            y=X[normal_mask, 1],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=8, opacity=0.6),
            hovertemplate='<b>Normal</b><br>' +
                         f'{feature_names[0]}: %{{x:.3f}}<br>' +
                         f'{feature_names[1]}: %{{y:.3f}}<br>' +
                         'Score: %{customdata:.3f}<extra></extra>',
            customdata=scores[normal_mask]
        ))
        
        # Anomaly points
        anomaly_mask = predictions == -1
        fig.add_trace(go.Scatter(
            x=X[anomaly_mask, 0],
            y=X[anomaly_mask, 1],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=12, symbol='x', opacity=0.8),
            hovertemplate='<b>Anomaly</b><br>' +
                         f'{feature_names[0]}: %{{x:.3f}}<br>' +
                         f'{feature_names[1]}: %{{y:.3f}}<br>' +
                         'Score: %{customdata:.3f}<extra></extra>',
            customdata=scores[anomaly_mask]
        ))
        
        fig.update_layout(
            title="Interactive Anomaly Detection Results",
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1],
            hovermode='closest'
        )
        
        fig.show()
    
    # Feature distribution comparison
    print("\n5.3 Interactive Feature Distributions")
    print("-" * 40)
    
    # Create subplot for each feature
    n_cols = min(3, len(feature_names))
    n_rows = (len(feature_names) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=feature_names,
        vertical_spacing=0.08
    )
    
    for i, feature_name in enumerate(feature_names):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Normal distribution
        fig.add_trace(go.Histogram(
            x=X[normal_mask, i],
            name=f'Normal - {feature_name}',
            opacity=0.7,
            nbinsx=30,
            legendgroup='normal',
            showlegend=(i == 0)
        ), row=row, col=col)
        
        # Anomaly distribution
        fig.add_trace(go.Histogram(
            x=X[anomaly_mask, i],
            name=f'Anomaly - {feature_name}',
            opacity=0.7,
            nbinsx=30,
            legendgroup='anomaly',
            showlegend=(i == 0)
        ), row=row, col=col)
    
    fig.update_layout(
        title="Feature Distributions: Normal vs Anomaly",
        height=300 * n_rows,
        barmode='overlay'
    )
    
    fig.show()


def main():
    """Run all explainable AI examples."""
    print("\n" + "="*60)
    print("EXPLAINABLE AI FOR ANOMALY DETECTION")
    print("="*60)
    
    examples = [
        ("SHAP-Based Explanations", example_1_shap_explanations),
        ("LIME-Based Local Explanations", example_2_lime_explanations),
        ("Counterfactual Explanations", example_3_counterfactual_explanations),
        ("Ensemble Model Explanations", example_4_ensemble_explanations),
        ("Interactive Explanations", example_5_interactive_explanations)
    ]
    
    while True:
        print("\nAvailable Examples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        print("0. Exit")
        
        try:
            choice = int(input("\nSelect an example (0-5): "))
            if choice == 0:
                print("Exiting...")
                break
            elif 1 <= choice <= len(examples):
                examples[choice-1][1]()
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error running example: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()