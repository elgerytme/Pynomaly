"""
Multi-modal anomaly detection fusion adapter.

This module implements multi-modal data fusion capabilities for anomaly detection,
combining time-series, tabular, graph, and text data modalities into unified
anomaly detection pipelines with cross-modal analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.infrastructure.adapters.pygod_adapter import PyGODAdapter
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.infrastructure.adapters.text_adapter import (
    TextAnomalyDetector,
    TextDetectionConfig,
)
from pynomaly.infrastructure.adapters.time_series_adapter import TimeSeriesAdapter
from pynomaly.shared.protocols import DetectorProtocol

logger = logging.getLogger(__name__)


class MultiModalConfig(BaseModel):
    """Configuration for multi-modal anomaly detection."""

    # Data modality weights
    time_series_weight: float = Field(default=0.25, description="Weight for time series modality")
    tabular_weight: float = Field(default=0.25, description="Weight for tabular modality")
    graph_weight: float = Field(default=0.25, description="Weight for graph modality")
    text_weight: float = Field(default=0.25, description="Weight for text modality")

    # Fusion strategy
    fusion_method: str = Field(default="weighted_average", description="Fusion method for combining scores")
    cross_modal_analysis: bool = Field(default=True, description="Enable cross-modal correlation analysis")

    # Thresholds
    consensus_threshold: float = Field(default=0.6, description="Consensus threshold for multi-modal detection")
    cross_modal_threshold: float = Field(default=0.7, description="Threshold for cross-modal anomalies")

    # Algorithm configurations for each modality
    time_series_config: dict[str, Any] = Field(default_factory=dict)
    tabular_config: dict[str, Any] = Field(default_factory=dict)
    graph_config: dict[str, Any] = Field(default_factory=dict)
    text_config: dict[str, Any] = Field(default_factory=dict)


class ModalityDetector:
    """Base class for modality-specific detectors."""

    def __init__(self, detector: DetectorProtocol, weight: float):
        self.detector = detector
        self.weight = weight
        self.is_fitted = False
        self.last_scores = None

    def fit(self, data: Any) -> None:
        """Fit the modality detector."""
        if hasattr(self.detector, 'fit'):
            self.detector.fit(data)
        self.is_fitted = True

    def predict(self, data: Any) -> tuple[list[float], list[bool]]:
        """Predict anomalies for this modality."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        if hasattr(self.detector, 'score'):
            scores = self.detector.score(data)
        else:
            # Fallback for detectors without score method
            scores = [0.5] * len(data) if hasattr(data, '__len__') else [0.5]

        if hasattr(self.detector, 'predict'):
            predictions = self.detector.predict(data)
            if isinstance(predictions, np.ndarray):
                # Convert sklearn format (-1/1) to boolean
                labels = [pred == -1 if pred in [-1, 1] else bool(pred) for pred in predictions]
            else:
                labels = [bool(pred) for pred in predictions]
        else:
            # Fallback: use threshold on scores
            threshold = np.percentile(scores, 90)
            labels = [score > threshold for score in scores]

        self.last_scores = scores
        return scores, labels


class MultiModalFusionAdapter(DetectorProtocol):
    """Multi-modal anomaly detection with data fusion capabilities."""

    def __init__(self, config: MultiModalConfig | None = None):
        self.config = config or MultiModalConfig()
        self.modality_detectors: dict[str, ModalityDetector] = {}
        self.is_fitted = False
        self.data_types = set()

        # Initialize detectors based on configuration
        self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        """Initialize detectors for each modality."""
        # Time series detector
        if self.config.time_series_weight > 0:
            from pynomaly.domain.entities import Detector
            from pynomaly.domain.value_objects import ContaminationRate

            ts_detector = Detector(
                name="TimeSeriesModality",
                algorithm_name="StatisticalTS",
                contamination_rate=ContaminationRate(0.1),
                parameters=self.config.time_series_config
            )
            ts_adapter = TimeSeriesAdapter(ts_detector)
            self.modality_detectors["time_series"] = ModalityDetector(
                ts_adapter, self.config.time_series_weight
            )

        # Graph detector
        if self.config.graph_weight > 0:
            graph_adapter = PyGODAdapter(
                algorithm_name="DOMINANT",
                contamination_rate=ContaminationRate(0.1),
                **self.config.graph_config
            )
            self.modality_detectors["graph"] = ModalityDetector(
                graph_adapter, self.config.graph_weight
            )

        # Text detector
        if self.config.text_weight > 0:
            text_config = TextDetectionConfig(**self.config.text_config)
            text_adapter = TextAnomalyDetector(text_config)
            self.modality_detectors["text"] = ModalityDetector(
                text_adapter, self.config.text_weight
            )

        # Tabular detector
        if self.config.tabular_weight > 0:
            tabular_adapter = SklearnAdapter(
                algorithm_name="IsolationForest",
                contamination_rate=ContaminationRate(0.1),
                **self.config.tabular_config
            )
            self.modality_detectors["tabular"] = ModalityDetector(
                tabular_adapter, self.config.tabular_weight
            )

    def _identify_data_types(self, dataset: Dataset) -> list[str]:
        """Identify which data modalities are present in the dataset."""
        modalities = []
        df = dataset.data

        # Check for time series data
        if self._has_time_series_structure(df):
            modalities.append("time_series")

        # Check for graph data
        if self._has_graph_structure(df, dataset):
            modalities.append("graph")

        # Check for text data
        if self._has_text_data(df):
            modalities.append("text")

        # Always consider tabular if we have numeric data
        if self._has_tabular_data(df):
            modalities.append("tabular")

        return modalities

    def _has_time_series_structure(self, df: pd.DataFrame) -> bool:
        """Check if data has time series structure."""
        # Look for datetime columns or sequential numeric data
        has_datetime = any(df[col].dtype.name.startswith('datetime') for col in df.columns)
        has_time_column = any(col.lower() in ['time', 'timestamp', 'date'] for col in df.columns)
        has_sequential_index = df.index.is_monotonic_increasing

        return has_datetime or has_time_column or has_sequential_index

    def _has_graph_structure(self, df: pd.DataFrame, dataset: Dataset) -> bool:
        """Check if data has graph structure."""
        # Check for edge list format
        has_edge_columns = ("source" in df.columns and "target" in df.columns) or "edge_index" in df.columns

        # Check for adjacency matrix in metadata
        has_adjacency = "adjacency_matrix" in dataset.metadata if dataset.metadata else False

        return has_edge_columns or has_adjacency

    def _has_text_data(self, df: pd.DataFrame) -> bool:
        """Check if data contains text columns."""
        text_columns = df.select_dtypes(include=['object', 'string']).columns

        # Check if any text column contains substantial text (more than just categories)
        for col in text_columns:
            sample_values = df[col].dropna().head(10)
            if any(isinstance(val, str) and len(val.split()) > 3 for val in sample_values):
                return True

        return False

    def _has_tabular_data(self, df: pd.DataFrame) -> bool:
        """Check if data has tabular numeric features."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        return len(numeric_columns) > 0

    def fit(self, dataset: Dataset) -> None:
        """Fit multi-modal detector on dataset."""
        logger.info("Fitting multi-modal anomaly detector")

        # Identify data modalities present
        self.data_types = set(self._identify_data_types(dataset))
        logger.info(f"Detected data modalities: {self.data_types}")

        # Fit relevant modality detectors
        for modality in self.data_types:
            if modality in self.modality_detectors:
                logger.info(f"Fitting {modality} detector")
                try:
                    if modality == "text":
                        # Extract text data
                        text_data = self._extract_text_data(dataset)
                        self.modality_detectors[modality].fit(text_data)
                    else:
                        self.modality_detectors[modality].fit(dataset)
                except Exception as e:
                    logger.warning(f"Failed to fit {modality} detector: {e}")
                    # Remove failed detector from active set
                    self.data_types.discard(modality)

        self.is_fitted = True
        logger.info("Multi-modal detector fitted successfully")

    def predict(self, dataset: Dataset) -> DetectionResult:
        """Predict anomalies using multi-modal fusion."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        # Get predictions from each modality
        modality_results = {}
        n_samples = len(dataset.data)

        for modality in self.data_types:
            if modality in self.modality_detectors:
                try:
                    if modality == "text":
                        text_data = self._extract_text_data(dataset)
                        scores, labels = self.modality_detectors[modality].predict(text_data)
                    else:
                        scores, labels = self.modality_detectors[modality].predict(dataset)

                    modality_results[modality] = {
                        "scores": scores,
                        "labels": labels,
                        "weight": self.modality_detectors[modality].weight
                    }
                except Exception as e:
                    logger.warning(f"Failed to predict with {modality} detector: {e}")

        # Fuse results
        fused_scores, fused_labels = self._fuse_predictions(modality_results, n_samples)

        # Cross-modal analysis
        cross_modal_anomalies = []
        if self.config.cross_modal_analysis:
            cross_modal_anomalies = self._detect_cross_modal_anomalies(modality_results)

        # Create anomaly scores
        anomaly_scores = [
            AnomalyScore(value=float(score), method="multimodal_fusion")
            for score in fused_scores
        ]

        # Create detection result
        anomaly_indices = [i for i, is_anomaly in enumerate(fused_labels) if is_anomaly]

        return DetectionResult(
            detector_id="MultiModalFusion",
            dataset_id=dataset.id,
            scores=anomaly_scores,
            labels=fused_labels,
            metadata={
                "fusion_method": self.config.fusion_method,
                "active_modalities": list(self.data_types),
                "modality_weights": {
                    mod: self.modality_detectors[mod].weight
                    for mod in self.data_types if mod in self.modality_detectors
                },
                "cross_modal_anomalies": cross_modal_anomalies,
                "n_anomalies": len(anomaly_indices),
                "contamination_rate": len(anomaly_indices) / n_samples if n_samples > 0 else 0
            }
        )

    def _extract_text_data(self, dataset: Dataset) -> list[str]:
        """Extract text data from dataset."""
        df = dataset.data
        text_columns = df.select_dtypes(include=['object', 'string']).columns

        # Find the main text column
        text_column = None
        for col in text_columns:
            sample_values = df[col].dropna().head(10)
            if any(isinstance(val, str) and len(val.split()) > 3 for val in sample_values):
                text_column = col
                break

        if text_column is None:
            raise ValueError("No substantial text data found")

        return df[text_column].fillna("").tolist()

    def _fuse_predictions(self, modality_results: dict[str, dict], n_samples: int) -> tuple[list[float], list[bool]]:
        """Fuse predictions from multiple modalities."""
        if not modality_results:
            return [0.0] * n_samples, [False] * n_samples

        if self.config.fusion_method == "weighted_average":
            return self._weighted_average_fusion(modality_results, n_samples)
        elif self.config.fusion_method == "consensus":
            return self._consensus_fusion(modality_results, n_samples)
        elif self.config.fusion_method == "max_score":
            return self._max_score_fusion(modality_results, n_samples)
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")

    def _weighted_average_fusion(self, modality_results: dict[str, dict], n_samples: int) -> tuple[list[float], list[bool]]:
        """Fuse using weighted average of scores."""
        fused_scores = np.zeros(n_samples)
        total_weight = 0

        for modality, results in modality_results.items():
            scores = np.array(results["scores"])
            weight = results["weight"]

            # Ensure scores array has correct length
            if len(scores) != n_samples:
                scores = np.resize(scores, n_samples)

            fused_scores += scores * weight
            total_weight += weight

        if total_weight > 0:
            fused_scores /= total_weight

        # Convert to labels using threshold
        threshold = np.percentile(fused_scores, (1 - 0.1) * 100)  # 10% contamination
        fused_labels = fused_scores > threshold

        return fused_scores.tolist(), fused_labels.tolist()

    def _consensus_fusion(self, modality_results: dict[str, dict], n_samples: int) -> tuple[list[float], list[bool]]:
        """Fuse using consensus voting."""
        vote_matrix = np.zeros((n_samples, len(modality_results)))
        score_matrix = np.zeros((n_samples, len(modality_results)))

        for i, (modality, results) in enumerate(modality_results.items()):
            labels = np.array(results["labels"])
            scores = np.array(results["scores"])

            # Ensure arrays have correct length
            if len(labels) != n_samples:
                labels = np.resize(labels, n_samples)
            if len(scores) != n_samples:
                scores = np.resize(scores, n_samples)

            vote_matrix[:, i] = labels.astype(int)
            score_matrix[:, i] = scores

        # Calculate consensus
        consensus_votes = np.mean(vote_matrix, axis=1)
        consensus_scores = np.mean(score_matrix, axis=1)

        # Apply consensus threshold
        fused_labels = consensus_votes >= self.config.consensus_threshold

        return consensus_scores.tolist(), fused_labels.tolist()

    def _max_score_fusion(self, modality_results: dict[str, dict], n_samples: int) -> tuple[list[float], list[bool]]:
        """Fuse using maximum score across modalities."""
        max_scores = np.zeros(n_samples)
        any_anomaly = np.zeros(n_samples, dtype=bool)

        for modality, results in modality_results.items():
            scores = np.array(results["scores"])
            labels = np.array(results["labels"])

            # Ensure arrays have correct length
            if len(scores) != n_samples:
                scores = np.resize(scores, n_samples)
            if len(labels) != n_samples:
                labels = np.resize(labels, n_samples)

            max_scores = np.maximum(max_scores, scores)
            any_anomaly = np.logical_or(any_anomaly, labels)

        return max_scores.tolist(), any_anomaly.tolist()

    def _detect_cross_modal_anomalies(self, modality_results: dict[str, dict]) -> list[int]:
        """Detect anomalies that are consistent across multiple modalities."""
        if len(modality_results) < 2:
            return []

        # Find samples that are anomalous in multiple modalities
        modality_labels = []
        for results in modality_results.values():
            labels = np.array(results["labels"])
            modality_labels.append(labels)

        # Stack labels and count anomalies per sample
        if modality_labels:
            label_matrix = np.column_stack(modality_labels)
            anomaly_counts = np.sum(label_matrix, axis=1)

            # Cross-modal anomalies are those detected by multiple modalities
            cross_modal_threshold = max(2, len(modality_results) * self.config.cross_modal_threshold)
            cross_modal_indices = np.where(anomaly_counts >= cross_modal_threshold)[0]

            return cross_modal_indices.tolist()

        return []

    def detect(self, data: Any, **kwargs) -> DetectionResult:
        """Main detection method following DetectorProtocol."""
        if isinstance(data, Dataset):
            dataset = data
        else:
            # Convert to Dataset if needed
            if isinstance(data, pd.DataFrame):
                from pynomaly.domain.entities import Dataset
                dataset = Dataset(
                    data=data,
                    name="MultiModalData",
                    description="Multi-modal dataset for anomaly detection"
                )
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

        if not self.is_fitted:
            self.fit(dataset)

        return self.predict(dataset)


# Factory function for easy instantiation
def create_multimodal_detector(
    time_series_weight: float = 0.25,
    tabular_weight: float = 0.25,
    graph_weight: float = 0.25,
    text_weight: float = 0.25,
    fusion_method: str = "weighted_average",
    **kwargs
) -> MultiModalFusionAdapter:
    """Create a multi-modal anomaly detector with specified configuration."""
    config = MultiModalConfig(
        time_series_weight=time_series_weight,
        tabular_weight=tabular_weight,
        graph_weight=graph_weight,
        text_weight=text_weight,
        fusion_method=fusion_method,
        **kwargs
    )
    return MultiModalFusionAdapter(config)
