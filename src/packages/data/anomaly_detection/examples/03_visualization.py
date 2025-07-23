#!/usr/bin/env python3
"""
Visualization Examples for Anomaly Detection Package

This example demonstrates various visualization techniques for anomaly detection results,
including 2D/3D plots, heatmaps, time series, and interactive dashboards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from typing import Dict, Any, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Install plotly for interactive visualizations: pip install plotly")

# Import anomaly detection components
try:
    from anomaly_detection import DetectionService, EnsembleService
except ImportError:
    print("Please install the anomaly_detection package first:")
    print("pip install -e .")
    exit(1)


class AnomalyVisualizer:
    """Utility class for visualizing anomaly detection results."""
    
    def __init__(self, style: str = 'seaborn'):
        """Initialize visualizer with matplotlib style."""
        plt.style.use(style)
        self.colors = {
            'normal': '#3498db',
            'anomaly': '#e74c3c',
            'uncertain': '#f39c12'
        }
    
    def plot_2d_scatter(self, 
                       X: np.ndarray, 
                       predictions: np.ndarray,
                       scores: Optional[np.ndarray] = None,
                       title: str = "Anomaly Detection Results",
                       feature_names: Optional[List[str]] = None,
                       show_scores: bool = True,
                       figsize: Tuple[int, int] = (10, 8)):
        """
        Create 2D scatter plot with anomalies highlighted.
        
        Args:
            X: Feature matrix (n_samples, 2)
            predictions: Binary predictions (1 or -1)
            scores: Anomaly scores
            title: Plot title
            feature_names: Names of features
            show_scores: Whether to show score contours
            figsize: Figure size
        """
        if X.shape[1] != 2:
            raise ValueError("Input must have exactly 2 features for 2D plotting")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Separate normal and anomaly points
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        # Plot points
        ax.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                  c=self.colors['normal'], label='Normal', 
                  alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        ax.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                  c=self.colors['anomaly'], label='Anomaly', 
                  alpha=0.8, s=100, marker='x', linewidths=2)
        
        # Add score contours if available
        if show_scores and scores is not None:
            self._add_score_contours(ax, X, scores)
        
        # Labels and title
        xlabel = feature_names[0] if feature_names else 'Feature 1'
        ylabel = feature_names[1] if feature_names else 'Feature 2'
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    def plot_3d_scatter(self,
                       X: np.ndarray,
                       predictions: np.ndarray,
                       title: str = "3D Anomaly Detection",
                       feature_names: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (12, 9)):
        """
        Create 3D scatter plot for 3-dimensional data.
        
        Args:
            X: Feature matrix (n_samples, 3)
            predictions: Binary predictions
            title: Plot title
            feature_names: Names of features
            figsize: Figure size
        """
        if X.shape[1] != 3:
            raise ValueError("Input must have exactly 3 features for 3D plotting")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Separate normal and anomaly points
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        # Plot points
        ax.scatter(X[normal_mask, 0], X[normal_mask, 1], X[normal_mask, 2],
                  c=self.colors['normal'], label='Normal',
                  alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        
        ax.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], X[anomaly_mask, 2],
                  c=self.colors['anomaly'], label='Anomaly',
                  alpha=0.8, s=60, marker='^')
        
        # Labels
        xlabel = feature_names[0] if feature_names else 'Feature 1'
        ylabel = feature_names[1] if feature_names else 'Feature 2'
        zlabel = feature_names[2] if feature_names else 'Feature 3'
        
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_zlabel(zlabel, fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(loc='best')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_feature_importance(self,
                               feature_importance: Dict[str, float],
                               title: str = "Feature Importance for Anomaly Detection",
                               figsize: Tuple[int, int] = (10, 6)):
        """
        Plot feature importance as a horizontal bar chart.
        
        Args:
            feature_importance: Dictionary of feature names to importance scores
            title: Plot title
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1])
        features = [f[0] for f in sorted_features]
        importance = [f[1] for f in sorted_features]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, color=self.colors['normal'], alpha=0.8)
        
        # Color top features differently
        top_n = min(3, len(bars))
        for i in range(1, top_n + 1):
            bars[-i].set_color(self.colors['anomaly'])
            bars[-i].set_alpha(0.9)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (feat, imp) in enumerate(zip(features, importance)):
            ax.text(imp + max(importance) * 0.01, i, f'{imp:.3f}', 
                   va='center', fontsize=9)
        
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_anomaly_scores_distribution(self,
                                       scores: np.ndarray,
                                       predictions: np.ndarray,
                                       threshold: Optional[float] = None,
                                       title: str = "Anomaly Score Distribution",
                                       figsize: Tuple[int, int] = (10, 6)):
        """
        Plot distribution of anomaly scores.
        
        Args:
            scores: Anomaly scores
            predictions: Binary predictions
            threshold: Decision threshold
            title: Plot title
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Main histogram
        normal_scores = scores[predictions == 1]
        anomaly_scores = scores[predictions == -1]
        
        # Plot distributions
        ax1.hist(normal_scores, bins=50, alpha=0.7, label='Normal', 
                color=self.colors['normal'], density=True, edgecolor='white')
        ax1.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', 
                color=self.colors['anomaly'], density=True, edgecolor='white')
        
        # Add threshold line if provided
        if threshold is not None:
            ax1.axvline(threshold, color='black', linestyle='--', linewidth=2,
                       label=f'Threshold: {threshold:.3f}')
        
        ax1.set_xlabel('Anomaly Score', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot below
        data_for_box = [normal_scores, anomaly_scores]
        box_plot = ax2.boxplot(data_for_box, vert=False, patch_artist=True,
                              labels=['Normal', 'Anomaly'])
        
        # Color the boxes
        colors = [self.colors['normal'], self.colors['anomaly']]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Anomaly Score', fontsize=12)
        ax2.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_time_series_anomalies(self,
                                  timestamps: pd.DatetimeIndex,
                                  values: np.ndarray,
                                  predictions: np.ndarray,
                                  title: str = "Time Series Anomaly Detection",
                                  figsize: Tuple[int, int] = (14, 6)):
        """
        Plot time series with anomalies highlighted.
        
        Args:
            timestamps: Time index
            values: Time series values
            predictions: Binary predictions
            title: Plot title
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot normal points
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        # Main time series line
        ax.plot(timestamps, values, color='gray', alpha=0.5, linewidth=1)
        
        # Highlight normal and anomaly points
        ax.scatter(timestamps[normal_mask], values[normal_mask],
                  c=self.colors['normal'], s=20, alpha=0.6, label='Normal')
        ax.scatter(timestamps[anomaly_mask], values[anomaly_mask],
                  c=self.colors['anomaly'], s=100, marker='x', 
                  linewidths=2, label='Anomaly')
        
        # Add shaded regions for anomaly periods
        if np.any(anomaly_mask):
            self._add_anomaly_regions(ax, timestamps, anomaly_mask)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self,
                                X: np.ndarray,
                                predictions: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                title: str = "Feature Correlation Heatmap",
                                figsize: Tuple[int, int] = (10, 8)):
        """
        Plot correlation heatmap with anomaly statistics.
        
        Args:
            X: Feature matrix
            predictions: Binary predictions
            feature_names: Names of features
            title: Plot title
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Prepare feature names
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
        
        # Calculate correlations for normal data
        normal_data = X[predictions == 1]
        corr_normal = pd.DataFrame(normal_data, columns=feature_names).corr()
        
        # Plot normal correlation
        sns.heatmap(corr_normal, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('Normal Data Correlations', fontsize=12)
        
        # Calculate correlations for anomaly data if enough samples
        anomaly_data = X[predictions == -1]
        if len(anomaly_data) > 3:
            corr_anomaly = pd.DataFrame(anomaly_data, columns=feature_names).corr()
            sns.heatmap(corr_anomaly, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, ax=ax2, cbar_kws={'shrink': 0.8})
            ax2.set_title('Anomaly Data Correlations', fontsize=12)
        else:
            ax2.text(0.5, 0.5, 'Not enough anomaly samples', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Anomaly Data Correlations', fontsize=12)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_decision_boundary(self,
                             X: np.ndarray,
                             model_predict_func,
                             predictions: np.ndarray,
                             title: str = "Decision Boundary",
                             resolution: int = 100,
                             figsize: Tuple[int, int] = (10, 8)):
        """
        Plot decision boundary for 2D data.
        
        Args:
            X: Feature matrix (n_samples, 2)
            model_predict_func: Function that predicts for new data
            predictions: Binary predictions for X
            title: Plot title
            resolution: Grid resolution
            figsize: Figure size
        """
        if X.shape[1] != 2:
            raise ValueError("Decision boundary plotting requires 2D data")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Predict on mesh grid
        Z = model_predict_func(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1],
                   colors=[self.colors['anomaly'], 'white', self.colors['normal']])
        
        # Plot data points
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        ax.scatter(X[normal_mask, 0], X[normal_mask, 1],
                  c=self.colors['normal'], label='Normal',
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1],
                  c=self.colors['anomaly'], label='Anomaly',
                  alpha=0.8, s=100, marker='x', linewidths=2)
        
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_plot(self,
                              X: np.ndarray,
                              predictions: np.ndarray,
                              scores: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              title: str = "Interactive Anomaly Detection"):
        """
        Create interactive plot using Plotly.
        
        Args:
            X: Feature matrix
            predictions: Binary predictions
            scores: Anomaly scores
            feature_names: Names of features
            title: Plot title
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return None
        
        # Prepare data
        df = pd.DataFrame(X, columns=feature_names or [f'Feature {i+1}' for i in range(X.shape[1])])
        df['Anomaly'] = predictions == -1
        df['Score'] = scores
        df['Type'] = df['Anomaly'].map({True: 'Anomaly', False: 'Normal'})
        
        # Create scatter plot matrix
        fig = px.scatter_matrix(
            df,
            dimensions=df.columns[:-3],  # Exclude added columns
            color='Type',
            color_discrete_map={'Normal': self.colors['normal'], 
                              'Anomaly': self.colors['anomaly']},
            hover_data=['Score'],
            title=title,
            height=800
        )
        
        fig.update_traces(diagonal_visible=False, showupperhalf=False,
                         marker=dict(size=5))
        
        return fig
    
    def _add_score_contours(self, ax, X: np.ndarray, scores: np.ndarray):
        """Add anomaly score contours to 2D plot."""
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        
        # Interpolate scores on grid
        from scipy.interpolate import griddata
        zz = griddata((X[:, 0], X[:, 1]), scores, (xx, yy), method='linear')
        
        # Plot contours
        contour = ax.contour(xx, yy, zz, levels=10, colors='black', 
                            alpha=0.4, linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8)
    
    def _add_anomaly_regions(self, ax, timestamps, anomaly_mask):
        """Add shaded regions for consecutive anomalies."""
        # Find consecutive anomaly regions
        anomaly_indices = np.where(anomaly_mask)[0]
        if len(anomaly_indices) == 0:
            return
        
        regions = []
        start = anomaly_indices[0]
        
        for i in range(1, len(anomaly_indices)):
            if anomaly_indices[i] != anomaly_indices[i-1] + 1:
                regions.append((start, anomaly_indices[i-1]))
                start = anomaly_indices[i]
        regions.append((start, anomaly_indices[-1]))
        
        # Add shaded regions
        for start, end in regions:
            ax.axvspan(timestamps[start], timestamps[end], 
                      alpha=0.2, color=self.colors['anomaly'])


def generate_example_data(data_type: str = 'blob', n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Generate example data for visualization."""
    np.random.seed(42)
    
    if data_type == 'blob':
        # Blob data with outliers
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=n_samples, centers=3, n_features=2, 
                         center_box=(-10, 10), random_state=42)
        
        # Add anomalies
        n_anomalies = int(n_samples * 0.1)
        anomalies = np.random.uniform(-15, 15, (n_anomalies, 2))
        X = np.vstack([X[:-n_anomalies], anomalies])
        
    elif data_type == 'moon':
        # Moon-shaped data
        from sklearn.datasets import make_moons
        X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
        X = X * 5  # Scale up
        
        # Add anomalies in the middle
        n_anomalies = int(n_samples * 0.1)
        anomalies = np.random.normal(2.5, 0.5, (n_anomalies, 2))
        X = np.vstack([X[:-n_anomalies], anomalies])
        
    elif data_type == 'time_series':
        # Time series with anomalies
        t = np.linspace(0, 4 * np.pi, n_samples)
        signal = np.sin(t) + 0.5 * np.sin(3 * t) + np.random.normal(0, 0.1, n_samples)
        
        # Add anomalies
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        signal[anomaly_indices] += np.random.normal(0, 2, len(anomaly_indices))
        
        X = np.column_stack([t, signal])
        
    elif data_type == 'high_dim':
        # High-dimensional data
        X = np.random.randn(n_samples, 10)
        
        # Add anomalies
        n_anomalies = int(n_samples * 0.1)
        X[-n_anomalies:] += np.random.normal(3, 1, (n_anomalies, 10))
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], indices


def example_1_basic_visualizations():
    """Example 1: Basic 2D and 3D visualizations."""
    print("\n" + "="*60)
    print("Example 1: Basic Visualizations")
    print("="*60)
    
    visualizer = AnomalyVisualizer()
    service = DetectionService()
    
    # Generate 2D data
    X_2d, _ = generate_example_data('blob', n_samples=300)
    
    # Detect anomalies
    result = service.detect_anomalies(X_2d, algorithm='iforest', contamination=0.1)
    
    # 2D scatter plot
    fig1 = visualizer.plot_2d_scatter(
        X_2d, result.predictions, result.anomaly_scores,
        title="2D Anomaly Detection - Isolation Forest",
        feature_names=['X', 'Y']
    )
    plt.show()
    
    # Generate 3D data
    X_3d = np.random.randn(300, 3)
    X_3d[-30:] += np.array([3, 3, 3])  # Add anomalies
    
    result_3d = service.detect_anomalies(X_3d, algorithm='lof', contamination=0.1)
    
    # 3D scatter plot
    fig2 = visualizer.plot_3d_scatter(
        X_3d, result_3d.predictions,
        title="3D Anomaly Detection - Local Outlier Factor",
        feature_names=['X', 'Y', 'Z']
    )
    plt.show()


def example_2_score_analysis():
    """Example 2: Anomaly score analysis and distributions."""
    print("\n" + "="*60)
    print("Example 2: Anomaly Score Analysis")
    print("="*60)
    
    visualizer = AnomalyVisualizer()
    service = DetectionService()
    
    # Generate data
    X, _ = generate_example_data('moon', n_samples=500)
    
    # Detect anomalies
    result = service.detect_anomalies(X, algorithm='iforest', contamination=0.15)
    
    # Score distribution
    fig = visualizer.plot_anomaly_scores_distribution(
        result.anomaly_scores,
        result.predictions,
        threshold=np.percentile(result.anomaly_scores, 85),
        title="Anomaly Score Distribution Analysis"
    )
    plt.show()
    
    # Feature importance (simulated for demonstration)
    feature_importance = {
        'Feature 1': 0.65,
        'Feature 2': 0.35,
        'Interaction 1-2': 0.15,
        'Feature 1 Squared': 0.25,
        'Feature 2 Squared': 0.10
    }
    
    fig2 = visualizer.plot_feature_importance(
        feature_importance,
        title="Feature Importance for Anomaly Detection"
    )
    plt.show()


def example_3_time_series_visualization():
    """Example 3: Time series anomaly visualization."""
    print("\n" + "="*60)
    print("Example 3: Time Series Anomaly Visualization")
    print("="*60)
    
    visualizer = AnomalyVisualizer()
    service = DetectionService()
    
    # Generate time series data
    dates = pd.date_range('2024-01-01', periods=500, freq='h')
    
    # Create synthetic time series with patterns
    t = np.arange(len(dates))
    trend = 0.01 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 24)  # Daily pattern
    noise = np.random.normal(0, 0.5, len(dates))
    values = trend + seasonal + noise
    
    # Add anomalies
    anomaly_indices = [100, 101, 102, 250, 251, 400, 401, 402, 403]
    for idx in anomaly_indices:
        values[idx] += np.random.choice([-3, 3]) * np.random.uniform(1, 2)
    
    # Prepare data for anomaly detection
    # Use rolling features
    window = 24
    features = pd.DataFrame({
        'value': values,
        'rolling_mean': pd.Series(values).rolling(window).mean(),
        'rolling_std': pd.Series(values).rolling(window).std(),
        'diff': pd.Series(values).diff()
    }).dropna()
    
    X = features.values
    
    # Detect anomalies
    result = service.detect_anomalies(X, algorithm='iforest', contamination=0.05)
    
    # Adjust for dropped rows
    adjusted_dates = dates[window:]
    adjusted_values = values[window:]
    
    # Time series plot
    fig = visualizer.plot_time_series_anomalies(
        adjusted_dates,
        adjusted_values,
        result.predictions,
        title="Time Series Anomaly Detection"
    )
    plt.show()


def example_4_correlation_analysis():
    """Example 4: Correlation and feature relationship analysis."""
    print("\n" + "="*60)
    print("Example 4: Correlation Analysis")
    print("="*60)
    
    visualizer = AnomalyVisualizer()
    service = DetectionService()
    
    # Generate correlated features
    n_samples = 500
    mean = [0, 0, 0, 0]
    cov = [[1, 0.8, 0.2, 0],
           [0.8, 1, 0.3, 0.1],
           [0.2, 0.3, 1, 0.7],
           [0, 0.1, 0.7, 1]]
    
    X_normal = np.random.multivariate_normal(mean, cov, n_samples - 50)
    
    # Generate anomalies with different correlation structure
    cov_anomaly = [[1, -0.5, 0, 0],
                   [-0.5, 1, 0, 0],
                   [0, 0, 1, -0.5],
                   [0, 0, -0.5, 1]]
    X_anomaly = np.random.multivariate_normal([2, 2, -2, -2], cov_anomaly, 50)
    
    X = np.vstack([X_normal, X_anomaly])
    
    # Detect anomalies
    result = service.detect_anomalies(X, algorithm='iforest', contamination=0.1)
    
    # Correlation heatmap
    fig = visualizer.plot_correlation_heatmap(
        X,
        result.predictions,
        feature_names=['Sensor A', 'Sensor B', 'Sensor C', 'Sensor D'],
        title="Feature Correlation Analysis"
    )
    plt.show()


def example_5_decision_boundaries():
    """Example 5: Decision boundary visualization."""
    print("\n" + "="*60)
    print("Example 5: Decision Boundary Visualization")
    print("="*60)
    
    visualizer = AnomalyVisualizer()
    service = DetectionService()
    
    # Generate 2D data
    X, _ = generate_example_data('moon', n_samples=300)
    
    # Train model
    service.fit(X, algorithm='ocsvm')
    result = service.predict(X, algorithm='ocsvm')
    
    # Create prediction function for decision boundary
    def predict_func(X_new):
        result = service.predict(X_new, algorithm='ocsvm')
        return result.predictions
    
    # Plot decision boundary
    fig = visualizer.plot_decision_boundary(
        X,
        predict_func,
        result.predictions,
        title="One-Class SVM Decision Boundary"
    )
    plt.show()


def example_6_interactive_visualization():
    """Example 6: Interactive visualizations with Plotly."""
    print("\n" + "="*60)
    print("Example 6: Interactive Visualizations")
    print("="*60)
    
    if not PLOTLY_AVAILABLE:
        print("Plotly not installed. Skipping interactive examples.")
        print("Install with: pip install plotly")
        return
    
    visualizer = AnomalyVisualizer()
    service = DetectionService()
    
    # Generate multi-dimensional data
    n_samples = 500
    X = np.random.randn(n_samples, 4)
    
    # Add anomalies
    n_anomalies = 50
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    X[anomaly_indices] += np.random.normal(2, 0.5, (n_anomalies, 4))
    
    # Detect anomalies
    result = service.detect_anomalies(X, algorithm='iforest', contamination=0.1)
    
    # Create interactive plot
    fig = visualizer.create_interactive_plot(
        X,
        result.predictions,
        result.anomaly_scores,
        feature_names=['Temperature', 'Pressure', 'Flow Rate', 'Vibration'],
        title="Interactive Multi-Dimensional Anomaly Analysis"
    )
    
    if fig:
        fig.show()
    
    # Create interactive 3D plot
    fig_3d = go.Figure(data=[
        go.Scatter3d(
            x=X[result.predictions == 1, 0],
            y=X[result.predictions == 1, 1],
            z=X[result.predictions == 1, 2],
            mode='markers',
            name='Normal',
            marker=dict(
                size=5,
                color='blue',
                opacity=0.6
            )
        ),
        go.Scatter3d(
            x=X[result.predictions == -1, 0],
            y=X[result.predictions == -1, 1],
            z=X[result.predictions == -1, 2],
            mode='markers',
            name='Anomaly',
            marker=dict(
                size=8,
                color='red',
                symbol='x',
                opacity=0.9
            )
        )
    ])
    
    fig_3d.update_layout(
        title="Interactive 3D Anomaly Visualization",
        scene=dict(
            xaxis_title="Temperature",
            yaxis_title="Pressure",
            zaxis_title="Flow Rate"
        ),
        height=700
    )
    
    fig_3d.show()


def example_7_dashboard_style():
    """Example 7: Dashboard-style visualization."""
    print("\n" + "="*60)
    print("Example 7: Dashboard-Style Visualization")
    print("="*60)
    
    visualizer = AnomalyVisualizer()
    service = DetectionService()
    
    # Generate data
    X, _ = generate_example_data('blob', n_samples=1000)
    
    # Run multiple algorithms
    algorithms = ['iforest', 'lof', 'ocsvm']
    results = {}
    
    for algo in algorithms:
        try:
            results[algo] = service.detect_anomalies(X, algorithm=algo, contamination=0.1)
        except:
            print(f"Skipping {algo}")
    
    # Create dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    if 'iforest' in results:
        result = results['iforest']
        normal_mask = result.predictions == 1
        anomaly_mask = result.predictions == -1
        
        ax_main.scatter(X[normal_mask, 0], X[normal_mask, 1],
                       c='blue', alpha=0.6, s=30)
        ax_main.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1],
                       c='red', alpha=0.8, s=60, marker='x')
        ax_main.set_title('Anomaly Detection Results', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
    
    # Algorithm comparison
    ax_compare = fig.add_subplot(gs[0, 2])
    algo_names = list(results.keys())
    anomaly_counts = [results[algo].anomaly_count for algo in algo_names]
    
    bars = ax_compare.bar(algo_names, anomaly_counts, color=['#3498db', '#e74c3c', '#f39c12'])
    ax_compare.set_title('Algorithm Comparison', fontsize=12)
    ax_compare.set_ylabel('Anomalies Detected')
    
    # Add value labels on bars
    for bar, count in zip(bars, anomaly_counts):
        height = bar.get_height()
        ax_compare.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}', ha='center', va='bottom')
    
    # Score distribution
    ax_dist = fig.add_subplot(gs[1, 2])
    if 'iforest' in results:
        scores = results['iforest'].anomaly_scores
        ax_dist.hist(scores, bins=30, color='purple', alpha=0.7, edgecolor='white')
        ax_dist.set_title('Score Distribution', fontsize=12)
        ax_dist.set_xlabel('Anomaly Score')
        ax_dist.set_ylabel('Frequency')
    
    # Summary statistics
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    # Create summary text
    summary_text = "Summary Statistics\n" + "="*50 + "\n"
    summary_text += f"Total Samples: {len(X)}\n"
    
    for algo, result in results.items():
        summary_text += f"\n{algo.upper()}:\n"
        summary_text += f"  - Anomalies: {result.anomaly_count} ({result.anomaly_rate:.1%})\n"
        summary_text += f"  - Score Range: [{np.min(result.anomaly_scores):.3f}, {np.max(result.anomaly_scores):.3f}]\n"
    
    ax_stats.text(0.1, 0.9, summary_text, transform=ax_stats.transAxes,
                 fontsize=10, fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Anomaly Detection Dashboard', fontsize=16, fontweight='bold')
    plt.show()


def main():
    """Run all visualization examples."""
    print("\n" + "="*60)
    print("ANOMALY DETECTION - VISUALIZATION EXAMPLES")
    print("="*60)
    
    examples = [
        ("Basic 2D/3D Visualizations", example_1_basic_visualizations),
        ("Score Analysis & Distributions", example_2_score_analysis),
        ("Time Series Visualization", example_3_time_series_visualization),
        ("Correlation Analysis", example_4_correlation_analysis),
        ("Decision Boundaries", example_5_decision_boundaries),
        ("Interactive Visualizations", example_6_interactive_visualization),
        ("Dashboard-Style Display", example_7_dashboard_style)
    ]
    
    while True:
        print("\nAvailable Examples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        print("0. Exit")
        
        try:
            choice = int(input("\nSelect an example (0-7): "))
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