"""Advanced visualization service for anomaly detection with interactive charts and plots."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from pynomaly_detection.domain.entities.dataset import Dataset
from pynomaly_detection.domain.entities.detection_result import DetectionResult
from pynomaly_detection.domain.entities.detector import Detector
from pynomaly_detection.domain.exceptions import ValidationError

# Optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class VisualizationType(Enum):
    """Types of visualizations available."""

    SCATTER_2D = "scatter_2d"
    SCATTER_3D = "scatter_3d"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    HEATMAP = "heatmap"
    TIME_SERIES = "time_series"
    CORRELATION_MATRIX = "correlation_matrix"
    FEATURE_IMPORTANCE = "feature_importance"
    ROC_CURVE = "roc_curve"
    PRECISION_RECALL = "precision_recall"
    CONFUSION_MATRIX = "confusion_matrix"
    ANOMALY_SCORE_DISTRIBUTION = "anomaly_score_distribution"
    TSNE_EMBEDDING = "tsne_embedding"
    PCA_PROJECTION = "pca_projection"
    ENSEMBLE_COMPARISON = "ensemble_comparison"
    DRIFT_DETECTION = "drift_detection"
    EXPLAINABILITY_PLOT = "explainability_plot"
    DASHBOARD = "dashboard"


class PlotStyle(Enum):
    """Plot styling options."""

    DEFAULT = "default"
    DARK = "dark"
    MINIMAL = "minimal"
    COLORFUL = "colorful"
    PROFESSIONAL = "professional"
    SCIENTIFIC = "scientific"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""

    # Output settings
    output_format: str = "html"  # html, png, svg, pdf
    output_dir: str = "visualizations"
    save_plots: bool = False
    show_plots: bool = True

    # Styling
    plot_style: PlotStyle = PlotStyle.DEFAULT
    color_palette: str = "viridis"
    figure_size: tuple[int, int] = (12, 8)
    dpi: int = 300

    # Interactive features
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_hover: bool = True
    enable_selection: bool = True

    # Plot-specific settings
    max_points_scatter: int = 10000
    max_features_heatmap: int = 50
    bin_count_histogram: int = 50
    alpha_transparency: float = 0.7

    # Anomaly highlighting
    anomaly_color: str = "red"
    normal_color: str = "blue"
    highlight_anomalies: bool = True

    # Performance settings
    use_sampling: bool = True
    sample_size: int = 5000
    enable_caching: bool = True


@dataclass
class PlotData:
    """Container for plot data and metadata."""

    x_data: np.ndarray | None = None
    y_data: np.ndarray | None = None
    z_data: np.ndarray | None = None
    colors: np.ndarray | None = None
    sizes: np.ndarray | None = None
    labels: list[str] | None = None
    hover_text: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationResult:
    """Result of visualization generation."""

    plot_id: UUID
    plot_type: VisualizationType
    title: str
    html_content: str | None = None
    json_data: dict[str, Any] | None = None
    file_path: str | None = None
    thumbnail_path: str | None = None
    interactive: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class VisualizationService:
    """Advanced visualization service for anomaly detection."""

    def __init__(self, config: VisualizationConfig | None = None):
        """Initialize visualization service.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)

        # Visualization cache
        self._plot_cache: dict[str, VisualizationResult] = {}

        # Check library availability
        if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
            self.logger.warning("No visualization libraries available")

        # Set default style
        self._setup_styling()

    def _setup_styling(self) -> None:
        """Setup default styling for plots."""

        if MATPLOTLIB_AVAILABLE:
            if self.config.plot_style == PlotStyle.DARK:
                plt.style.use("dark_background")
            elif self.config.plot_style == PlotStyle.MINIMAL:
                plt.style.use("seaborn-v0_8-whitegrid")
            elif self.config.plot_style == PlotStyle.SCIENTIFIC:
                plt.style.use("seaborn-v0_8-paper")

        if PLOTLY_AVAILABLE:
            if self.config.plot_style == PlotStyle.DARK:
                pio.templates.default = "plotly_dark"
            elif self.config.plot_style == PlotStyle.MINIMAL:
                pio.templates.default = "simple_white"
            else:
                pio.templates.default = "plotly"

    async def create_anomaly_scatter_plot(
        self,
        data: np.ndarray,
        anomaly_scores: np.ndarray,
        labels: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        title: str = "Anomaly Detection Results",
        use_3d: bool = False,
    ) -> VisualizationResult:
        """Create scatter plot showing anomalies.

        Args:
            data: Input data matrix
            anomaly_scores: Anomaly scores for each point
            labels: True labels (if available)
            feature_names: Names of features
            title: Plot title
            use_3d: Whether to create 3D plot

        Returns:
            Visualization result
        """
        plot_id = uuid4()

        try:
            # Reduce dimensionality if needed
            plot_data = await self._prepare_scatter_data(
                data, anomaly_scores, labels, use_3d
            )

            if PLOTLY_AVAILABLE:
                fig = await self._create_plotly_scatter(
                    plot_data, title, use_3d, anomaly_scores
                )
                html_content = fig.to_html(include_plotlyjs=True)
                json_data = fig.to_dict()
            else:
                # Fallback to matplotlib
                fig = await self._create_matplotlib_scatter(
                    plot_data, title, use_3d, anomaly_scores
                )
                html_content = None
                json_data = None

            return VisualizationResult(
                plot_id=plot_id,
                plot_type=(
                    VisualizationType.SCATTER_3D
                    if use_3d
                    else VisualizationType.SCATTER_2D
                ),
                title=title,
                html_content=html_content,
                json_data=json_data,
                interactive=PLOTLY_AVAILABLE,
                metadata={
                    "n_points": len(data),
                    "n_features": data.shape[1] if len(data.shape) > 1 else 1,
                    "n_anomalies": int(
                        np.sum(anomaly_scores > np.median(anomaly_scores))
                    ),
                },
            )

        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {e}")
            raise ValidationError(f"Scatter plot creation failed: {e}")

    async def create_time_series_plot(
        self,
        timestamps: list[datetime],
        values: np.ndarray,
        anomaly_indices: list[int] | None = None,
        anomaly_scores: np.ndarray | None = None,
        title: str = "Time Series Anomaly Detection",
    ) -> VisualizationResult:
        """Create time series plot with anomaly highlighting.

        Args:
            timestamps: Time points
            values: Time series values
            anomaly_indices: Indices of anomalous points
            anomaly_scores: Anomaly scores
            title: Plot title

        Returns:
            Visualization result
        """
        plot_id = uuid4()

        try:
            if PLOTLY_AVAILABLE:
                fig = go.Figure()

                # Main time series
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=values,
                        mode="lines+markers",
                        name="Time Series",
                        line=dict(color="blue", width=2),
                        marker=dict(size=4),
                    )
                )

                # Highlight anomalies
                if anomaly_indices:
                    anomaly_times = [
                        timestamps[i] for i in anomaly_indices if i < len(timestamps)
                    ]
                    anomaly_vals = [
                        values[i] for i in anomaly_indices if i < len(values)
                    ]

                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_times,
                            y=anomaly_vals,
                            mode="markers",
                            name="Anomalies",
                            marker=dict(
                                color="red",
                                size=8,
                                symbol="x",
                            ),
                        )
                    )

                # Add anomaly score subplot if available
                if anomaly_scores is not None:
                    fig_subplots = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=["Time Series", "Anomaly Scores"],
                        vertical_spacing=0.1,
                    )

                    # Add main plot
                    fig_subplots.add_trace(
                        go.Scatter(x=timestamps, y=values, name="Time Series"),
                        row=1,
                        col=1,
                    )

                    # Add anomaly scores
                    fig_subplots.add_trace(
                        go.Scatter(
                            x=timestamps, y=anomaly_scores, name="Anomaly Scores"
                        ),
                        row=2,
                        col=1,
                    )

                    fig = fig_subplots

                fig.update_layout(
                    title=title,
                    xaxis_title="Time",
                    yaxis_title="Value",
                    hovermode="x unified",
                    template=pio.templates.default,
                )

                html_content = fig.to_html(include_plotlyjs=True)
                json_data = fig.to_dict()

            else:
                html_content = None
                json_data = None

            return VisualizationResult(
                plot_id=plot_id,
                plot_type=VisualizationType.TIME_SERIES,
                title=title,
                html_content=html_content,
                json_data=json_data,
                interactive=PLOTLY_AVAILABLE,
                metadata={
                    "time_range": {
                        "start": timestamps[0].isoformat() if timestamps else None,
                        "end": timestamps[-1].isoformat() if timestamps else None,
                    },
                    "n_points": len(values),
                    "n_anomalies": len(anomaly_indices) if anomaly_indices else 0,
                },
            )

        except Exception as e:
            self.logger.error(f"Error creating time series plot: {e}")
            raise ValidationError(f"Time series plot creation failed: {e}")

    async def create_feature_importance_plot(
        self,
        feature_names: list[str],
        importances: list[float],
        title: str = "Feature Importance",
        top_k: int = 20,
    ) -> VisualizationResult:
        """Create feature importance bar plot.

        Args:
            feature_names: Names of features
            importances: Importance scores
            title: Plot title
            top_k: Number of top features to show

        Returns:
            Visualization result
        """
        plot_id = uuid4()

        try:
            # Sort by importance and take top k
            sorted_indices = np.argsort(importances)[::-1][:top_k]
            top_features = [feature_names[i] for i in sorted_indices]
            top_importances = [importances[i] for i in sorted_indices]

            if PLOTLY_AVAILABLE:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=top_importances,
                            y=top_features,
                            orientation="h",
                            marker=dict(
                                color=top_importances,
                                colorscale="viridis",
                                showscale=True,
                            ),
                        )
                    ]
                )

                fig.update_layout(
                    title=title,
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    template=pio.templates.default,
                    height=max(400, len(top_features) * 25),
                )

                html_content = fig.to_html(include_plotlyjs=True)
                json_data = fig.to_dict()

            else:
                html_content = None
                json_data = None

            return VisualizationResult(
                plot_id=plot_id,
                plot_type=VisualizationType.FEATURE_IMPORTANCE,
                title=title,
                html_content=html_content,
                json_data=json_data,
                interactive=PLOTLY_AVAILABLE,
                metadata={
                    "n_features": len(feature_names),
                    "top_k": top_k,
                    "max_importance": max(importances),
                    "min_importance": min(importances),
                },
            )

        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {e}")
            raise ValidationError(f"Feature importance plot creation failed: {e}")

    async def create_correlation_heatmap(
        self,
        data: np.ndarray,
        feature_names: list[str] | None = None,
        title: str = "Feature Correlation Heatmap",
    ) -> VisualizationResult:
        """Create correlation heatmap.

        Args:
            data: Input data matrix
            feature_names: Names of features
            title: Plot title

        Returns:
            Visualization result
        """
        plot_id = uuid4()

        try:
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(data.T)

            # Limit number of features for readability
            max_features = self.config.max_features_heatmap
            if data.shape[1] > max_features:
                # Sample features
                selected_indices = np.random.choice(
                    data.shape[1], max_features, replace=False
                )
                correlation_matrix = correlation_matrix[
                    np.ix_(selected_indices, selected_indices)
                ]

                if feature_names:
                    feature_names = [feature_names[i] for i in selected_indices]

            if not feature_names:
                feature_names = [
                    f"Feature_{i}" for i in range(correlation_matrix.shape[0])
                ]

            if PLOTLY_AVAILABLE:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=correlation_matrix,
                        x=feature_names,
                        y=feature_names,
                        colorscale="RdBu",
                        zmid=0,
                        text=np.round(correlation_matrix, 2),
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        hoverongaps=False,
                    )
                )

                fig.update_layout(
                    title=title,
                    template=pio.templates.default,
                    height=max(400, len(feature_names) * 20),
                    width=max(400, len(feature_names) * 20),
                )

                html_content = fig.to_html(include_plotlyjs=True)
                json_data = fig.to_dict()

            else:
                html_content = None
                json_data = None

            return VisualizationResult(
                plot_id=plot_id,
                plot_type=VisualizationType.HEATMAP,
                title=title,
                html_content=html_content,
                json_data=json_data,
                interactive=PLOTLY_AVAILABLE,
                metadata={
                    "n_features": len(feature_names),
                    "max_correlation": float(
                        np.max(correlation_matrix[correlation_matrix < 1])
                    ),
                    "min_correlation": float(np.min(correlation_matrix)),
                },
            )

        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {e}")
            raise ValidationError(f"Correlation heatmap creation failed: {e}")

    async def create_anomaly_score_distribution(
        self,
        scores: np.ndarray,
        threshold: float | None = None,
        labels: np.ndarray | None = None,
        title: str = "Anomaly Score Distribution",
    ) -> VisualizationResult:
        """Create histogram of anomaly scores.

        Args:
            scores: Anomaly scores
            threshold: Anomaly threshold
            labels: True labels (if available)
            title: Plot title

        Returns:
            Visualization result
        """
        plot_id = uuid4()

        try:
            if PLOTLY_AVAILABLE:
                fig = go.Figure()

                # Main histogram
                fig.add_trace(
                    go.Histogram(
                        x=scores,
                        nbinsx=self.config.bin_count_histogram,
                        name="All Scores",
                        opacity=0.7,
                    )
                )

                # Separate normal and anomaly if labels available
                if labels is not None:
                    normal_scores = scores[labels == 0]
                    anomaly_scores = scores[labels == 1]

                    fig = go.Figure()

                    fig.add_trace(
                        go.Histogram(
                            x=normal_scores,
                            nbinsx=self.config.bin_count_histogram,
                            name="Normal",
                            opacity=0.7,
                            marker_color="blue",
                        )
                    )

                    fig.add_trace(
                        go.Histogram(
                            x=anomaly_scores,
                            nbinsx=self.config.bin_count_histogram,
                            name="Anomaly",
                            opacity=0.7,
                            marker_color="red",
                        )
                    )

                # Add threshold line
                if threshold is not None:
                    fig.add_vline(
                        x=threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Threshold: {threshold:.3f}",
                    )

                fig.update_layout(
                    title=title,
                    xaxis_title="Anomaly Score",
                    yaxis_title="Frequency",
                    barmode="overlay",
                    template=pio.templates.default,
                )

                html_content = fig.to_html(include_plotlyjs=True)
                json_data = fig.to_dict()

            else:
                html_content = None
                json_data = None

            return VisualizationResult(
                plot_id=plot_id,
                plot_type=VisualizationType.ANOMALY_SCORE_DISTRIBUTION,
                title=title,
                html_content=html_content,
                json_data=json_data,
                interactive=PLOTLY_AVAILABLE,
                metadata={
                    "n_scores": len(scores),
                    "score_range": {
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                    },
                    "threshold": threshold,
                },
            )

        except Exception as e:
            self.logger.error(f"Error creating score distribution plot: {e}")
            raise ValidationError(f"Score distribution plot creation failed: {e}")

    async def create_ensemble_comparison_plot(
        self,
        ensemble_results: dict[str, np.ndarray],
        ground_truth: np.ndarray | None = None,
        title: str = "Ensemble Method Comparison",
    ) -> VisualizationResult:
        """Create comparison plot for ensemble methods.

        Args:
            ensemble_results: Dictionary of method_name -> scores
            ground_truth: True labels for evaluation
            title: Plot title

        Returns:
            Visualization result
        """
        plot_id = uuid4()

        try:
            if PLOTLY_AVAILABLE:
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=[
                        "Score Distributions",
                        "Method Correlations",
                        "Performance Comparison",
                        "Score Scatter Matrix",
                    ],
                    specs=[
                        [{"secondary_y": False}, {"secondary_y": False}],
                        [{"secondary_y": False}, {"secondary_y": False}],
                    ],
                )

                # 1. Score distributions
                for method_name, scores in ensemble_results.items():
                    fig.add_trace(
                        go.Histogram(
                            x=scores,
                            name=method_name,
                            opacity=0.7,
                            nbinsx=30,
                        ),
                        row=1,
                        col=1,
                    )

                # 2. Method correlations
                if len(ensemble_results) > 1:
                    methods = list(ensemble_results.keys())
                    scores_matrix = np.array(
                        [ensemble_results[method] for method in methods]
                    )
                    correlation_matrix = np.corrcoef(scores_matrix)

                    fig.add_trace(
                        go.Heatmap(
                            z=correlation_matrix,
                            x=methods,
                            y=methods,
                            colorscale="RdBu",
                            zmid=0,
                        ),
                        row=1,
                        col=2,
                    )

                # 3. Performance comparison (if ground truth available)
                if ground_truth is not None:
                    method_names = []
                    auc_scores = []

                    for method_name, scores in ensemble_results.items():
                        try:
                            from sklearn.metrics import roc_auc_score

                            auc = roc_auc_score(ground_truth, scores)
                            method_names.append(method_name)
                            auc_scores.append(auc)
                        except:
                            # Fallback metric
                            auc_scores.append(0.5)

                    fig.add_trace(
                        go.Bar(x=method_names, y=auc_scores, name="AUC Score"),
                        row=2,
                        col=1,
                    )

                # 4. First two methods scatter plot
                if len(ensemble_results) >= 2:
                    methods = list(ensemble_results.keys())
                    scores1 = ensemble_results[methods[0]]
                    scores2 = ensemble_results[methods[1]]

                    colors = ground_truth if ground_truth is not None else None

                    fig.add_trace(
                        go.Scatter(
                            x=scores1,
                            y=scores2,
                            mode="markers",
                            marker=dict(
                                color=colors,
                                colorscale="viridis" if colors is not None else None,
                                showscale=colors is not None,
                            ),
                            name=f"{methods[0]} vs {methods[1]}",
                        ),
                        row=2,
                        col=2,
                    )

                fig.update_layout(
                    title=title,
                    template=pio.templates.default,
                    height=800,
                    showlegend=False,
                )

                html_content = fig.to_html(include_plotlyjs=True)
                json_data = fig.to_dict()

            else:
                html_content = None
                json_data = None

            return VisualizationResult(
                plot_id=plot_id,
                plot_type=VisualizationType.ENSEMBLE_COMPARISON,
                title=title,
                html_content=html_content,
                json_data=json_data,
                interactive=PLOTLY_AVAILABLE,
                metadata={
                    "n_methods": len(ensemble_results),
                    "methods": list(ensemble_results.keys()),
                    "has_ground_truth": ground_truth is not None,
                },
            )

        except Exception as e:
            self.logger.error(f"Error creating ensemble comparison plot: {e}")
            raise ValidationError(f"Ensemble comparison plot creation failed: {e}")

    async def create_dashboard(
        self,
        detection_results: list[DetectionResult],
        datasets: list[Dataset],
        detectors: list[Detector],
        title: str = "Anomaly Detection Dashboard",
    ) -> VisualizationResult:
        """Create comprehensive dashboard.

        Args:
            detection_results: List of detection results
            datasets: List of datasets
            detectors: List of detectors
            title: Dashboard title

        Returns:
            Visualization result
        """
        plot_id = uuid4()

        try:
            if not PLOTLY_AVAILABLE:
                raise ValidationError("Plotly required for dashboard creation")

            # Create dashboard with multiple subplots
            fig = make_subplots(
                rows=3,
                cols=3,
                subplot_titles=[
                    "Detection Overview",
                    "Score Distribution",
                    "Time Series",
                    "Detector Performance",
                    "Dataset Summary",
                    "Alert Trends",
                    "Resource Usage",
                    "Model Health",
                    "Recent Activity",
                ],
                specs=[
                    [{"type": "indicator"}, {"type": "histogram"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "pie"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "indicator"}, {"type": "table"}],
                ],
            )

            # 1. Detection Overview (KPIs)
            total_detections = len(detection_results)
            total_anomalies = sum(1 for r in detection_results if r.is_anomaly)
            anomaly_rate = (
                total_anomalies / total_detections if total_detections > 0 else 0
            )

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=anomaly_rate * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Anomaly Rate (%)"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 100], "color": "gray"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                ),
                row=1,
                col=1,
            )

            # 2. Score Distribution
            all_scores = [
                max(r.scores, key=lambda s: s.value).value
                for r in detection_results
                if r.scores
            ]
            if all_scores:
                fig.add_trace(
                    go.Histogram(x=all_scores, nbinsx=30, name="Score Distribution"),
                    row=1,
                    col=2,
                )

            # 3. Time Series of Detections
            timestamps = [r.timestamp for r in detection_results]
            anomaly_counts = []

            if timestamps:
                # Group by hour
                hourly_counts = {}
                for r in detection_results:
                    hour = r.timestamp.replace(minute=0, second=0, microsecond=0)
                    if hour not in hourly_counts:
                        hourly_counts[hour] = 0
                    if r.is_anomaly:
                        hourly_counts[hour] += 1

                sorted_hours = sorted(hourly_counts.keys())
                hourly_anomalies = [hourly_counts[h] for h in sorted_hours]

                fig.add_trace(
                    go.Scatter(
                        x=sorted_hours,
                        y=hourly_anomalies,
                        mode="lines+markers",
                        name="Anomalies per Hour",
                    ),
                    row=1,
                    col=3,
                )

            # 4. Detector Performance
            detector_stats = {}
            for detector in detectors:
                detector_results = [
                    r for r in detection_results if r.detector_id == detector.id
                ]
                if detector_results:
                    avg_score = np.mean(
                        [
                            max(r.scores, key=lambda s: s.value).value
                            for r in detector_results
                            if r.scores
                        ]
                    )
                    detector_stats[detector.name] = avg_score

            if detector_stats:
                fig.add_trace(
                    go.Bar(
                        x=list(detector_stats.keys()),
                        y=list(detector_stats.values()),
                        name="Avg Score by Detector",
                    ),
                    row=2,
                    col=1,
                )

            # 5. Dataset Summary
            dataset_sizes = [
                len(d.data) if hasattr(d, "data") and d.data is not None else 0
                for d in datasets
            ]
            dataset_names = [f"Dataset {i + 1}" for i in range(len(datasets))]

            if dataset_sizes:
                fig.add_trace(
                    go.Pie(
                        labels=dataset_names, values=dataset_sizes, name="Dataset Sizes"
                    ),
                    row=2,
                    col=2,
                )

            # 6. Alert Trends (last 24 hours)
            recent_results = [
                r
                for r in detection_results
                if r.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]

            if recent_results:
                recent_timestamps = [r.timestamp for r in recent_results]
                recent_anomalies = [1 if r.is_anomaly else 0 for r in recent_results]

                fig.add_trace(
                    go.Scatter(
                        x=recent_timestamps,
                        y=np.cumsum(recent_anomalies),
                        mode="lines",
                        name="Cumulative Anomalies (24h)",
                    ),
                    row=2,
                    col=3,
                )

            # 7. Resource Usage (placeholder)
            cpu_usage = np.random.uniform(20, 80, 24)  # Simulated
            hours = list(range(24))

            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=cpu_usage,
                    mode="lines+markers",
                    name="CPU Usage (%)",
                    line=dict(color="orange"),
                ),
                row=3,
                col=1,
            )

            # 8. Model Health Indicator
            health_score = 85  # Simulated

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=health_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Model Health Score"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {
                            "color": (
                                "green"
                                if health_score > 80
                                else "orange"
                                if health_score > 60
                                else "red"
                            )
                        },
                        "steps": [
                            {"range": [0, 60], "color": "lightgray"},
                            {"range": [60, 100], "color": "gray"},
                        ],
                    },
                ),
                row=3,
                col=2,
            )

            # 9. Recent Activity Table
            recent_activity = []
            for r in detection_results[-10:]:  # Last 10 results
                recent_activity.append(
                    [
                        r.timestamp.strftime("%H:%M:%S"),
                        "Anomaly" if r.is_anomaly else "Normal",
                        (
                            f"{max(r.scores, key=lambda s: s.value).value:.3f}"
                            if r.scores
                            else "N/A"
                        ),
                    ]
                )

            if recent_activity:
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=["Time", "Type", "Score"],
                            fill_color="paleturquoise",
                            align="left",
                        ),
                        cells=dict(
                            values=list(zip(*recent_activity, strict=False)),
                            fill_color="lavender",
                            align="left",
                        ),
                    ),
                    row=3,
                    col=3,
                )

            fig.update_layout(
                title=title,
                template=pio.templates.default,
                height=1200,
                showlegend=False,
            )

            html_content = fig.to_html(include_plotlyjs=True)
            json_data = fig.to_dict()

            return VisualizationResult(
                plot_id=plot_id,
                plot_type=VisualizationType.DASHBOARD,
                title=title,
                html_content=html_content,
                json_data=json_data,
                interactive=True,
                metadata={
                    "total_detections": total_detections,
                    "total_anomalies": total_anomalies,
                    "anomaly_rate": anomaly_rate,
                    "n_detectors": len(detectors),
                    "n_datasets": len(datasets),
                },
            )

        except Exception as e:
            self.logger.error(f"Error creating dashboard: {e}")
            raise ValidationError(f"Dashboard creation failed: {e}")

    async def _prepare_scatter_data(
        self,
        data: np.ndarray,
        anomaly_scores: np.ndarray,
        labels: np.ndarray | None = None,
        use_3d: bool = False,
    ) -> PlotData:
        """Prepare data for scatter plot."""

        # Sample data if too large
        if len(data) > self.config.max_points_scatter and self.config.use_sampling:
            indices = np.random.choice(
                len(data), self.config.max_points_scatter, replace=False
            )
            data = data[indices]
            anomaly_scores = anomaly_scores[indices]
            if labels is not None:
                labels = labels[indices]

        # Reduce dimensionality for visualization
        if data.shape[1] > (3 if use_3d else 2):
            if SKLEARN_AVAILABLE:
                if use_3d:
                    reducer = PCA(n_components=3)
                else:
                    reducer = PCA(n_components=2)

                data_reduced = reducer.fit_transform(
                    StandardScaler().fit_transform(data)
                )
            else:
                # Fallback: take first 2/3 dimensions
                data_reduced = data[:, : (3 if use_3d else 2)]
        else:
            data_reduced = data

        # Prepare plot data
        plot_data = PlotData()
        plot_data.x_data = data_reduced[:, 0]
        plot_data.y_data = data_reduced[:, 1]

        if use_3d and data_reduced.shape[1] >= 3:
            plot_data.z_data = data_reduced[:, 2]

        # Color by anomaly score or labels
        if labels is not None:
            plot_data.colors = labels
        else:
            plot_data.colors = anomaly_scores

        # Size by anomaly score
        plot_data.sizes = anomaly_scores * 10 + 5

        # Hover text
        plot_data.hover_text = [
            f"Score: {score:.3f}<br>Label: {label if labels is not None else 'Unknown'}"
            for score, label in zip(
                anomaly_scores,
                labels if labels is not None else [None] * len(anomaly_scores),
                strict=False,
            )
        ]

        return plot_data

    async def _create_plotly_scatter(
        self,
        plot_data: PlotData,
        title: str,
        use_3d: bool,
        anomaly_scores: np.ndarray,
    ):
        """Create Plotly scatter plot."""

        if use_3d and plot_data.z_data is not None:
            fig = go.Figure(
                data=go.Scatter3d(
                    x=plot_data.x_data,
                    y=plot_data.y_data,
                    z=plot_data.z_data,
                    mode="markers",
                    marker=dict(
                        size=plot_data.sizes,
                        color=plot_data.colors,
                        colorscale="viridis",
                        opacity=self.config.alpha_transparency,
                        showscale=True,
                        colorbar=dict(title="Anomaly Score"),
                    ),
                    text=plot_data.hover_text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    zaxis_title="PC3",
                ),
                template=pio.templates.default,
            )
        else:
            fig = go.Figure(
                data=go.Scatter(
                    x=plot_data.x_data,
                    y=plot_data.y_data,
                    mode="markers",
                    marker=dict(
                        size=plot_data.sizes,
                        color=plot_data.colors,
                        colorscale="viridis",
                        opacity=self.config.alpha_transparency,
                        showscale=True,
                        colorbar=dict(title="Anomaly Score"),
                    ),
                    text=plot_data.hover_text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title="PC1",
                yaxis_title="PC2",
                template=pio.templates.default,
            )

        return fig

    async def _create_matplotlib_scatter(
        self,
        plot_data: PlotData,
        title: str,
        use_3d: bool,
        anomaly_scores: np.ndarray,
    ):
        """Create matplotlib scatter plot (fallback)."""

        if not MATPLOTLIB_AVAILABLE:
            return None

        if use_3d:
            fig = plt.figure(figsize=self.config.figure_size)
            ax = fig.add_subplot(111, projection="3d")

            scatter = ax.scatter(
                plot_data.x_data,
                plot_data.y_data,
                plot_data.z_data,
                c=plot_data.colors,
                s=plot_data.sizes,
                alpha=self.config.alpha_transparency,
                cmap="viridis",
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)

            scatter = ax.scatter(
                plot_data.x_data,
                plot_data.y_data,
                c=plot_data.colors,
                s=plot_data.sizes,
                alpha=self.config.alpha_transparency,
                cmap="viridis",
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

        plt.title(title)
        plt.colorbar(scatter, label="Anomaly Score")

        return fig

    async def save_visualization(
        self,
        result: VisualizationResult,
        filename: str | None = None,
    ) -> str:
        """Save visualization to file.

        Args:
            result: Visualization result to save
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        import os

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Generate filename
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{result.plot_type.value}_{timestamp}.{self.config.output_format}"
            )

        file_path = os.path.join(self.config.output_dir, filename)

        # Save based on format
        if self.config.output_format == "html" and result.html_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result.html_content)
        elif self.config.output_format == "json" and result.json_data:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result.json_data, f, indent=2)

        self.logger.info(f"Visualization saved to: {file_path}")

        # Update result with file path
        result.file_path = file_path

        return file_path

    async def get_cached_visualization(
        self, cache_key: str
    ) -> VisualizationResult | None:
        """Get cached visualization result.

        Args:
            cache_key: Cache key for the visualization

        Returns:
            Cached visualization result if available
        """
        return self._plot_cache.get(cache_key)

    async def cache_visualization(
        self,
        cache_key: str,
        result: VisualizationResult,
    ) -> None:
        """Cache visualization result.

        Args:
            cache_key: Cache key for the visualization
            result: Visualization result to cache
        """
        if self.config.enable_caching:
            self._plot_cache[cache_key] = result
