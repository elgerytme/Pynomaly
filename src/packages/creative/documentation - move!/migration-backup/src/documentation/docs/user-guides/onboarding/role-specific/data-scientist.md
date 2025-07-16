# Data Scientist Onboarding Guide

🍞 **Breadcrumb:** 🏠 [Home](../../../index.md) > 📚 [User Guides](../../README.md) > 🚀 [Onboarding](../README.md) > 🎯 [Role-Specific](README.md) > 📊 Data Scientist

---

Welcome, Data Scientist! This guide will help you quickly master Pynomaly's anomaly detection capabilities, from algorithm selection to model evaluation and advanced statistical analysis.

## 🎯 Learning Objectives

By the end of this guide, you'll be able to:

- **Select** the optimal anomaly detection algorithm for your use case
- **Engineer** features and preprocess data for maximum detection performance  
- **Evaluate** model performance using statistical metrics and validation techniques
- **Tune** hyperparameters and optimize detection accuracy
- **Interpret** results and provide statistical insights to stakeholders

## 📊 Quick Start (5 minutes)

### Installation for Data Scientists

```bash
# Install with science and visualization dependencies
pip install "pynomaly[science,viz]"

# Verify installation
python -c "import pynomaly; print(f'Pynomaly version: {pynomaly.__version__}')"
```

### First Detection

```python
import pandas as pd
import numpy as np
from pynomaly import IsolationForest, LocalOutlierFactor
from pynomaly.datasets import load_sample_data
import matplotlib.pyplot as plt

# Load sample data
data = load_sample_data('financial_transactions')
print(f"Dataset shape: {data.shape}")
print(data.head())

# Quick anomaly detection
detector = IsolationForest(contamination=0.1)
anomalies = detector.fit_predict(data)

# Visualize results
detector.plot_anomalies(data, anomalies)
plt.title('Financial Transaction Anomalies')
plt.show()

print(f"Found {sum(anomalies == -1)} anomalies out of {len(anomalies)} samples")
```

## 🧠 Algorithm Selection Framework

### Decision Tree for Algorithm Choice

```mermaid
graph TD
    A[Start: What type of data?] --> B{Dimensionality}
    B -->|High (>50 features)| C[Isolation Forest]
    B -->|Medium (10-50 features)| D{Local patterns important?}
    B -->|Low (<10 features)| E{Distribution known?}
    
    D -->|Yes| F[Local Outlier Factor]
    D -->|No| G[One-Class SVM]
    
    E -->|Yes| H[Statistical Methods]
    E -->|No| I[Isolation Forest]
    
    C --> J[Ensemble Methods]
    F --> K[Density-based Methods]
    G --> L[Support Vector Approaches]
    H --> M[Parametric Methods]
    I --> N[Tree-based Methods]
```

### Algorithm Comparison Matrix

| Algorithm | Best For | Pros | Cons | Contamination |
|-----------|----------|------|------|---------------|
| **Isolation Forest** | High-dimensional, mixed data | Fast, handles mixed types | Less interpretable | 0.05-0.2 |
| **Local Outlier Factor** | Local anomalies, clusters | Good for local patterns | Sensitive to parameters | 0.05-0.15 |
| **One-Class SVM** | Non-linear boundaries | Robust, kernel flexibility | Slow on large data | 0.05-0.1 |
| **Elliptic Envelope** | Gaussian data | Fast, interpretable | Assumes normality | 0.05-0.15 |
| **DBSCAN** | Density-based clusters | Finds clusters + outliers | Parameter sensitive | Auto-determined |

### Hands-On Algorithm Comparison

```python
from pynomaly import (
    IsolationForest, 
    LocalOutlierFactor, 
    OneClassSVM,
    EllipticEnvelope
)
from pynomaly.evaluation import compare_algorithms
from pynomaly.datasets import load_sample_data

# Load data
data = load_sample_data('multivariate_normal', n_samples=1000, contamination=0.1)
X, y_true = data['features'], data['labels']

# Define algorithms to compare
algorithms = {
    'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
    'LOF': LocalOutlierFactor(contamination=0.1),
    'One-Class SVM': OneClassSVM(nu=0.1),
    'Elliptic Envelope': EllipticEnvelope(contamination=0.1)
}

# Compare algorithms
results = compare_algorithms(algorithms, X, y_true)
print(results.summary())

# Visualize comparison
results.plot_comparison()
```

## 📈 Feature Engineering for Anomaly Detection

### Time Series Features

```python
import pandas as pd
from pynomaly.preprocessing import TimeSeriesFeatureExtractor

def create_time_features(df, timestamp_col='timestamp', value_col='value'):
    """Extract comprehensive time series features."""
    
    feature_extractor = TimeSeriesFeatureExtractor()
    
    # Statistical features
    features = feature_extractor.extract_statistical_features(
        df[value_col], 
        window_size=24  # 24-hour window
    )
    
    # Temporal features
    temporal_features = feature_extractor.extract_temporal_features(
        df[timestamp_col]
    )
    
    # Lag features
    lag_features = feature_extractor.extract_lag_features(
        df[value_col],
        lags=[1, 3, 7, 24]  # 1h, 3h, 1d, 1w
    )
    
    # Seasonal decomposition
    seasonal_features = feature_extractor.extract_seasonal_features(
        df[value_col],
        period=24  # Daily seasonality
    )
    
    return pd.concat([features, temporal_features, lag_features, seasonal_features], axis=1)

# Example usage
time_data = load_sample_data('time_series')
engineered_features = create_time_features(time_data)
print(f"Original features: {time_data.shape[1]}")
print(f"Engineered features: {engineered_features.shape[1]}")
```

### Multivariate Feature Engineering

```python
from pynomaly.preprocessing import MultivariateFeatureExtractor
from sklearn.preprocessing import StandardScaler, RobustScaler

def engineer_multivariate_features(df):
    """Create advanced multivariate features for anomaly detection."""
    
    extractor = MultivariateFeatureExtractor()
    
    # Correlation-based features
    correlation_features = extractor.extract_correlation_features(df)
    
    # Principal component features
    pca_features = extractor.extract_pca_features(df, n_components=0.95)
    
    # Interaction features
    interaction_features = extractor.extract_interaction_features(
        df, 
        degree=2, 
        interaction_only=True
    )
    
    # Distance-based features
    distance_features = extractor.extract_distance_features(df)
    
    # Combine all features
    all_features = pd.concat([
        correlation_features,
        pca_features,
        interaction_features,
        distance_features
    ], axis=1)
    
    # Scale features
    scaler = RobustScaler()  # Robust to outliers
    scaled_features = pd.DataFrame(
        scaler.fit_transform(all_features),
        columns=all_features.columns,
        index=all_features.index
    )
    
    return scaled_features, scaler

# Example usage
multivariate_data = load_sample_data('multivariate_mixed')
features, scaler = engineer_multivariate_features(multivariate_data)
```

## 🔬 Model Evaluation and Validation

### Comprehensive Evaluation Framework

```python
from pynomaly.evaluation import AnomalyEvaluator
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

class DataScienceEvaluator:
    """Comprehensive evaluation suite for data scientists."""
    
    def __init__(self):
        self.evaluator = AnomalyEvaluator()
        
    def evaluate_with_confidence_intervals(self, detector, X, y_true, n_bootstrap=100):
        """Evaluate with statistical confidence intervals."""
        
        scores = []
        for i in range(n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            y_boot = y_true[indices]
            
            # Fit and predict
            y_pred = detector.fit_predict(X_boot)
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(y_boot, y_pred)
            scores.append(metrics)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for metric in scores[0].keys():
            values = [score[metric] for score in scores]
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            mean_value = np.mean(values)
            
            confidence_intervals[metric] = {
                'mean': mean_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': np.std(values)
            }
            
        return confidence_intervals
    
    def time_series_validation(self, detector, X, y_true, n_splits=5):
        """Time series cross-validation."""
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_true[train_idx], y_true[test_idx]
            
            # Fit on training data
            detector.fit(X_train)
            
            # Predict on test data
            y_pred = detector.predict(X_test)
            
            # Evaluate
            metrics = self.evaluator.calculate_metrics(y_test, y_pred)
            scores.append(metrics)
        
        return scores
    
    def stability_analysis(self, detector, X, y_true, n_runs=20):
        """Analyze model stability across runs."""
        
        stability_scores = []
        
        for run in range(n_runs):
            # Add small amount of noise
            X_noisy = X + np.random.normal(0, 0.01, X.shape)
            
            # Fit and predict
            y_pred = detector.fit_predict(X_noisy)
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(y_true, y_pred)
            stability_scores.append(metrics)
        
        # Calculate stability metrics
        stability_analysis = {}
        for metric in stability_scores[0].keys():
            values = [score[metric] for score in stability_scores]
            stability_analysis[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'coefficient_of_variation': np.std(values) / np.mean(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return stability_analysis

# Example usage
evaluator = DataScienceEvaluator()

# Load data and detector
data = load_sample_data('financial_fraud')
X, y_true = data['features'], data['labels']
detector = IsolationForest(contamination=0.1, random_state=42)

# Comprehensive evaluation
print("=== Confidence Intervals ===")
ci_results = evaluator.evaluate_with_confidence_intervals(detector, X, y_true)
for metric, stats in ci_results.items():
    print(f"{metric}: {stats['mean']:.3f} [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")

print("\n=== Time Series Validation ===")
ts_results = evaluator.time_series_validation(detector, X, y_true)
for i, metrics in enumerate(ts_results):
    print(f"Fold {i+1}: F1={metrics['f1_score']:.3f}, Precision={metrics['precision']:.3f}")

print("\n=== Stability Analysis ===")
stability_results = evaluator.stability_analysis(detector, X, y_true)
for metric, stats in stability_results.items():
    print(f"{metric}: CV={stats['coefficient_of_variation']:.3f}")
```

### Statistical Significance Testing

```python
from scipy import stats
from pynomaly.evaluation import statistical_tests

def compare_algorithms_statistically(algorithms, X, y_true, n_runs=30):
    """Compare algorithms with statistical significance testing."""
    
    results = {}
    
    for name, detector in algorithms.items():
        scores = []
        for run in range(n_runs):
            # Set different random seed for each run
            if hasattr(detector, 'random_state'):
                detector.random_state = run
            
            y_pred = detector.fit_predict(X)
            f1 = f1_score(y_true, y_pred)
            scores.append(f1)
        
        results[name] = scores
    
    # Perform pairwise statistical tests
    algorithm_names = list(algorithms.keys())
    comparison_matrix = pd.DataFrame(
        index=algorithm_names, 
        columns=algorithm_names
    )
    
    for i, alg1 in enumerate(algorithm_names):
        for j, alg2 in enumerate(algorithm_names):
            if i != j:
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(results[alg1], results[alg2])
                comparison_matrix.loc[alg1, alg2] = p_value
            else:
                comparison_matrix.loc[alg1, alg2] = 1.0
    
    return results, comparison_matrix

# Example usage
algorithms = {
    'Isolation Forest': IsolationForest(contamination=0.1),
    'LOF': LocalOutlierFactor(contamination=0.1),
    'One-Class SVM': OneClassSVM(nu=0.1)
}

score_results, p_value_matrix = compare_algorithms_statistically(algorithms, X, y_true)

print("P-value matrix (lower is more significant):")
print(p_value_matrix.round(4))
```

## 🎯 Hyperparameter Optimization

### Bayesian Optimization for Anomaly Detection

```python
from pynomaly.optimization import BayesianHyperparameterOptimizer
from sklearn.model_selection import cross_val_score
import optuna

class AnomalyDetectionOptimizer:
    """Advanced hyperparameter optimization for anomaly detection."""
    
    def __init__(self, detector_class, X, y_true):
        self.detector_class = detector_class
        self.X = X
        self.y_true = y_true
        
    def optimize_isolation_forest(self, n_trials=100):
        """Optimize Isolation Forest hyperparameters."""
        
        def objective(trial):
            # Define hyperparameter space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_samples': trial.suggest_float('max_samples', 0.1, 1.0),
                'contamination': trial.suggest_float('contamination', 0.05, 0.3),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'random_state': 42
            }
            
            # Create detector
            detector = IsolationForest(**params)
            
            # Cross-validation score
            scores = []
            for train_idx, val_idx in self._get_cv_splits():
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y_true[train_idx], self.y_true[val_idx]
                
                # Fit and predict
                detector.fit(X_train)
                y_pred = detector.predict(X_val)
                
                # Calculate F1 score
                f1 = f1_score(y_val, y_pred)
                scores.append(f1)
            
            return np.mean(scores)
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params, study.best_value
    
    def optimize_lof(self, n_trials=100):
        """Optimize Local Outlier Factor hyperparameters."""
        
        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 5, 50),
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree']),
                'leaf_size': trial.suggest_int('leaf_size', 10, 50),
                'contamination': trial.suggest_float('contamination', 0.05, 0.3),
                'p': trial.suggest_int('p', 1, 2)
            }
            
            detector = LocalOutlierFactor(**params)
            
            # Cross-validation
            scores = []
            for train_idx, val_idx in self._get_cv_splits():
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y_true[train_idx], self.y_true[val_idx]
                
                # LOF requires fit_predict
                y_pred_train = detector.fit_predict(X_train)
                
                # For validation, we need to use decision_function
                detector.fit(X_train)
                decision_scores = detector.decision_function(X_val)
                threshold = np.percentile(decision_scores, params['contamination'] * 100)
                y_pred = (decision_scores < threshold).astype(int) * 2 - 1
                
                f1 = f1_score(y_val, y_pred)
                scores.append(f1)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params, study.best_value
    
    def _get_cv_splits(self, n_splits=5):
        """Get cross-validation splits."""
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return cv.split(self.X, self.y_true)

# Example usage
optimizer = AnomalyDetectionOptimizer(IsolationForest, X, y_true)

print("Optimizing Isolation Forest...")
best_params_if, best_score_if = optimizer.optimize_isolation_forest(n_trials=50)
print(f"Best IF params: {best_params_if}")
print(f"Best IF score: {best_score_if:.4f}")

print("\nOptimizing Local Outlier Factor...")
best_params_lof, best_score_lof = optimizer.optimize_lof(n_trials=50)
print(f"Best LOF params: {best_params_lof}")
print(f"Best LOF score: {best_score_lof:.4f}")
```

## 📊 Advanced Visualization and Interpretation

### Comprehensive Anomaly Analysis Dashboard

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

class AnomalyVisualizationSuite:
    """Advanced visualization tools for anomaly analysis."""
    
    def __init__(self, X, y_true, y_pred, feature_names=None):
        self.X = X
        self.y_true = y_true
        self.y_pred = y_pred
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
    def create_comprehensive_dashboard(self):
        """Create interactive dashboard with multiple views."""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Anomaly Distribution', 'Feature Importance',
                'Principal Component Analysis', 'Confusion Matrix',
                'Anomaly Scores Distribution', 'Time Series View'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "histogram"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Anomaly Distribution (PCA)
        self._add_pca_plot(fig, row=1, col=1)
        
        # 2. Feature Importance
        self._add_feature_importance(fig, row=1, col=2)
        
        # 3. PCA explained variance
        self._add_pca_variance(fig, row=2, col=1)
        
        # 4. Confusion Matrix
        self._add_confusion_matrix(fig, row=2, col=2)
        
        # 5. Anomaly Scores
        self._add_score_distribution(fig, row=3, col=1)
        
        # 6. Time series (if applicable)
        self._add_time_series(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Comprehensive Anomaly Detection Analysis",
            showlegend=True
        )
        
        return fig
    
    def _add_pca_plot(self, fig, row, col):
        """Add PCA visualization."""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        
        # Normal points
        normal_mask = self.y_pred == 1
        fig.add_trace(
            go.Scatter(
                x=X_pca[normal_mask, 0],
                y=X_pca[normal_mask, 1],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=4, opacity=0.6)
            ),
            row=row, col=col
        )
        
        # Anomalous points
        anomaly_mask = self.y_pred == -1
        fig.add_trace(
            go.Scatter(
                x=X_pca[anomaly_mask, 0],
                y=X_pca[anomaly_mask, 1],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=6, opacity=0.8)
            ),
            row=row, col=col
        )
    
    def _add_feature_importance(self, fig, row, col):
        """Add feature importance analysis."""
        from sklearn.ensemble import IsolationForest
        
        # Calculate feature importance using permutation
        detector = IsolationForest(contamination=0.1, random_state=42)
        detector.fit(self.X)
        
        importances = []
        base_score = detector.score_samples(self.X).mean()
        
        for i in range(self.X.shape[1]):
            X_permuted = self.X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            permuted_score = detector.score_samples(X_permuted).mean()
            importance = abs(base_score - permuted_score)
            importances.append(importance)
        
        fig.add_trace(
            go.Bar(
                x=self.feature_names,
                y=importances,
                name='Feature Importance'
            ),
            row=row, col=col
        )

# Example usage
viz = AnomalyVisualizationSuite(X, y_true, y_pred, feature_names=X.columns.tolist())
dashboard = viz.create_comprehensive_dashboard()
dashboard.show()
```

## 🔍 Interpretability and Explainability

### SHAP Integration for Anomaly Explanation

```python
import shap
from pynomaly.explainability import AnomalyExplainer

class AnomalyInterpreter:
    """Provide interpretable explanations for anomaly predictions."""
    
    def __init__(self, detector, X_train):
        self.detector = detector
        self.X_train = X_train
        self.explainer = AnomalyExplainer(detector)
        
    def explain_anomalies(self, X_test, anomaly_indices=None):
        """Explain why specific instances are flagged as anomalies."""
        
        if anomaly_indices is None:
            predictions = self.detector.predict(X_test)
            anomaly_indices = np.where(predictions == -1)[0]
        
        explanations = {}
        
        for idx in anomaly_indices:
            instance = X_test.iloc[idx:idx+1] if hasattr(X_test, 'iloc') else X_test[idx:idx+1]
            
            # SHAP explanation
            shap_explanation = self._get_shap_explanation(instance)
            
            # Local outlier factor explanation
            lof_explanation = self._get_lof_explanation(instance)
            
            # Feature contribution analysis
            feature_contributions = self._analyze_feature_contributions(instance)
            
            explanations[idx] = {
                'shap_values': shap_explanation,
                'lof_analysis': lof_explanation,
                'feature_contributions': feature_contributions,
                'anomaly_score': self.detector.score_samples(instance)[0]
            }
        
        return explanations
    
    def _get_shap_explanation(self, instance):
        """Get SHAP explanation for the instance."""
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            self._predict_proba_wrapper,
            shap.sample(self.X_train, 100)
        )
        
        # Get SHAP values
        shap_values = explainer.shap_values(instance)
        
        return {
            'shap_values': shap_values,
            'expected_value': explainer.expected_value,
            'feature_names': self.X_train.columns.tolist() if hasattr(self.X_train, 'columns') else None
        }
    
    def _predict_proba_wrapper(self, X):
        """Wrapper to convert anomaly scores to probabilities."""
        scores = self.detector.score_samples(X)
        # Convert to probabilities (higher score = more normal)
        probs = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probs, probs])
    
    def _get_lof_explanation(self, instance):
        """Analyze local neighborhood characteristics."""
        
        # Find nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=20)
        nn.fit(self.X_train)
        distances, indices = nn.kneighbors(instance)
        
        neighbors = self.X_train.iloc[indices[0]] if hasattr(self.X_train, 'iloc') else self.X_train[indices[0]]
        
        # Compare instance to neighbors
        comparison = {
            'mean_distance_to_neighbors': distances[0].mean(),
            'std_distance_to_neighbors': distances[0].std(),
            'feature_deviations': {}
        }
        
        for i, feature in enumerate(neighbors.columns if hasattr(neighbors, 'columns') else range(neighbors.shape[1])):
            instance_value = instance[0, i] if hasattr(instance, 'iloc') else instance[0][i]
            neighbor_values = neighbors.iloc[:, i] if hasattr(neighbors, 'iloc') else neighbors[:, i]
            
            comparison['feature_deviations'][feature] = {
                'instance_value': instance_value,
                'neighbor_mean': neighbor_values.mean(),
                'neighbor_std': neighbor_values.std(),
                'z_score': (instance_value - neighbor_values.mean()) / (neighbor_values.std() + 1e-8)
            }
        
        return comparison
    
    def create_explanation_report(self, explanations, output_file=None):
        """Create comprehensive explanation report."""
        
        report = []
        report.append("# Anomaly Detection Explanation Report\\n")
        
        for idx, explanation in explanations.items():
            report.append(f"## Instance {idx}")
            report.append(f"**Anomaly Score:** {explanation['anomaly_score']:.4f}\\n")
            
            # Feature contributions
            feature_contrib = explanation['feature_contributions']
            report.append("### Feature Contributions")
            
            for feature, contrib in sorted(feature_contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                sign = "+" if contrib > 0 else ""
                report.append(f"- **{feature}:** {sign}{contrib:.4f}")
            
            # Neighborhood analysis
            lof_analysis = explanation['lof_analysis']
            report.append("\\n### Local Neighborhood Analysis")
            report.append(f"- **Mean distance to neighbors:** {lof_analysis['mean_distance_to_neighbors']:.4f}")
            
            # Top deviating features
            deviations = lof_analysis['feature_deviations']
            top_deviations = sorted(deviations.items(), key=lambda x: abs(x[1]['z_score']), reverse=True)[:3]
            
            report.append("- **Most deviating features:**")
            for feature, dev in top_deviations:
                report.append(f"  - {feature}: Z-score = {dev['z_score']:.2f}")
            
            report.append("\\n---\\n")
        
        report_text = "\\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text

# Example usage
interpreter = AnomalyInterpreter(detector, X_train)
explanations = interpreter.explain_anomalies(X_test)

# Generate explanation report
report = interpreter.create_explanation_report(explanations, 'anomaly_explanations.md')
print(report)
```

## 🎓 Advanced Topics and Research

### Ensemble Methods for Robust Detection

```python
from pynomaly.ensemble import AnomalyEnsemble
from sklearn.metrics import roc_auc_score

class AdvancedEnsembleMethods:
    """Advanced ensemble techniques for anomaly detection."""
    
    def __init__(self):
        self.ensembles = {}
        
    def create_diverse_ensemble(self, X, contamination=0.1):
        """Create ensemble with diverse base detectors."""
        
        base_detectors = {
            'isolation_forest': IsolationForest(
                contamination=contamination,
                n_estimators=100,
                random_state=42
            ),
            'lof': LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=20
            ),
            'one_class_svm': OneClassSVM(
                nu=contamination,
                kernel='rbf'
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=contamination,
                support_fraction=None
            )
        }
        
        # Fit all detectors
        predictions = {}
        scores = {}
        
        for name, detector in base_detectors.items():
            if name == 'lof':
                pred = detector.fit_predict(X)
                scores[name] = detector.negative_outlier_factor_
            else:
                detector.fit(X)
                pred = detector.predict(X)
                scores[name] = detector.score_samples(X)
            
            predictions[name] = pred
        
        return predictions, scores, base_detectors
    
    def weighted_ensemble(self, predictions, scores, weights=None):
        """Create weighted ensemble based on individual performance."""
        
        if weights is None:
            # Calculate weights based on silhouette score
            from sklearn.metrics import silhouette_score
            weights = {}
            
            for name, pred in predictions.items():
                if len(np.unique(pred)) > 1:  # Must have both classes
                    # Use absolute scores for silhouette calculation
                    abs_scores = np.abs(scores[name])
                    weight = silhouette_score(abs_scores.reshape(-1, 1), pred)
                    weights[name] = max(0, weight)  # Ensure non-negative
                else:
                    weights[name] = 0.1  # Minimal weight for poor detectors
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Weighted ensemble prediction
        weighted_scores = np.zeros(len(next(iter(scores.values()))))
        
        for name, score in scores.items():
            # Normalize scores to [0, 1]
            normalized_score = (score - score.min()) / (score.max() - score.min() + 1e-8)
            weighted_scores += weights[name] * normalized_score
        
        # Convert to predictions
        threshold = np.percentile(weighted_scores, (1 - contamination) * 100)
        ensemble_predictions = (weighted_scores < threshold).astype(int) * 2 - 1
        
        return ensemble_predictions, weighted_scores, weights
    
    def stacking_ensemble(self, X, y_true, base_detectors):
        """Create stacking ensemble with meta-learner."""
        
        from sklearn.model_selection import cross_val_predict
        from sklearn.linear_model import LogisticRegression
        
        # Generate meta-features using cross-validation
        meta_features = []
        
        for name, detector in base_detectors.items():
            if name == 'lof':
                # Special handling for LOF
                cv_scores = cross_val_predict(
                    detector, X, cv=5, method='score_samples'
                )
            else:
                cv_scores = cross_val_predict(
                    detector, X, cv=5, method='score_samples'
                )
            
            meta_features.append(cv_scores)
        
        meta_features = np.column_stack(meta_features)
        
        # Train meta-learner
        meta_learner = LogisticRegression()
        meta_learner.fit(meta_features, y_true)
        
        return meta_learner, meta_features

# Example usage
ensemble_methods = AdvancedEnsembleMethods()

# Create diverse ensemble
predictions, scores, base_detectors = ensemble_methods.create_diverse_ensemble(X)

# Weighted ensemble
ensemble_pred, ensemble_scores, weights = ensemble_methods.weighted_ensemble(predictions, scores)

print("Ensemble weights:")
for name, weight in weights.items():
    print(f"  {name}: {weight:.3f}")

# Evaluate ensemble
ensemble_f1 = f1_score(y_true, ensemble_pred)
print(f"\\nEnsemble F1 Score: {ensemble_f1:.4f}")

# Compare with individual detectors
print("\\nIndividual detector performance:")
for name, pred in predictions.items():
    individual_f1 = f1_score(y_true, pred)
    print(f"  {name}: {individual_f1:.4f}")
```

## 📋 Practice Exercises

### Exercise 1: Financial Fraud Detection (30 minutes)

```python
# Load financial transaction data
fraud_data = load_sample_data('financial_fraud')
X, y_true = fraud_data['features'], fraud_data['labels']

# Tasks:
# 1. Explore the data and identify key characteristics
# 2. Engineer relevant features for fraud detection
# 3. Compare at least 3 different algorithms
# 4. Optimize hyperparameters for the best algorithm
# 5. Provide statistical confidence intervals
# 6. Create visualizations and explanations

# Your solution here...
```

### Exercise 2: Time Series Anomaly Detection (45 minutes)

```python
# Load time series data
ts_data = load_sample_data('server_metrics')

# Tasks:
# 1. Extract time-based features
# 2. Handle seasonality and trends
# 3. Implement online/streaming detection
# 4. Evaluate using time-aware cross-validation
# 5. Create interpretable explanations

# Your solution here...
```

### Exercise 3: Custom Algorithm Implementation (60 minutes)

```python
# Tasks:
# 1. Implement a novel anomaly detection algorithm
# 2. Compare with existing methods
# 3. Provide theoretical justification
# 4. Test on multiple datasets
# 5. Document your approach

class CustomAnomalyDetector:
    """Your custom anomaly detection algorithm."""
    
    def __init__(self, **params):
        # Your implementation here
        pass
    
    def fit(self, X):
        # Your implementation here
        pass
    
    def predict(self, X):
        # Your implementation here
        pass
    
    def score_samples(self, X):
        # Your implementation here
        pass

# Your solution here...
```

## 🚀 Next Steps

### Advanced Learning Paths

1. **Research Track**: Dive into cutting-edge research papers and implementations
2. **Production Track**: Focus on deployment, monitoring, and MLOps
3. **Domain Expert Track**: Specialize in specific industries (finance, healthcare, etc.)

### Recommended Resources

- **Papers**: [Anomaly Detection Research Papers](https://github.com/hoya012/awesome-anomaly-detection)
- **Datasets**: [Outlier Detection DataSets (ODDS)](http://odds.cs.stonybrook.edu/)
- **Competitions**: [Kaggle Anomaly Detection Competitions](https://www.kaggle.com/competitions?search=anomaly)

### Community Engagement

- Join the **Data Science Slack**: `#data-science-anomaly`
- Attend weekly **Office Hours**: Wednesdays 2-3 PM PST
- Contribute to **Open Source**: Submit algorithms and improvements

---

**Congratulations!** You've completed the Data Scientist onboarding guide. You're now equipped with advanced anomaly detection skills and ready to tackle real-world challenges.

**Ready for more?** Continue with:
- 🚀 **[ML Engineer Guide](ml-engineer.md)** - Learn production deployment
- 📊 **[Advanced Analytics](../../advanced-features/README.md)** - Explore cutting-edge techniques
- 🏆 **[Certification Program](../certification.md)** - Earn your Data Science certification