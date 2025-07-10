"""PyGOD adapter for graph anomaly detection algorithms.

This module integrates algorithms from the PyGOD (Python Graph Outlier Detection) library.
PyGOD provides state-of-the-art algorithms for detecting anomalies in graph-structured data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.exceptions import AdapterError, AlgorithmNotFoundError
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate

logger = logging.getLogger(__name__)


class PyGODAdapter:
    """Adapter for PyGOD graph anomaly detection algorithms.

    This adapter implements DetectorProtocol and maintains clean architecture
    by keeping infrastructure concerns separate from domain logic.
    """

    # Lazy imports to avoid import errors if PyGOD not installed
    _algorithm_map: dict[str, type] | None = None

    @classmethod
    def _get_algorithm_map(cls) -> dict[str, type]:
        """Lazily import and return PyGOD algorithm mapping."""
        if cls._algorithm_map is None:
            try:
                # Import PyGOD algorithms
                from pygod.models import (
                    ANOMALOUS,
                    ANOMALYDAE,
                    CONAD,
                    DOMINANT,
                    GAAN,
                    GADNR,
                    GCNAE,
                    GUIDE,
                    MLPAE,
                    RADAR,
                    SCAN,
                )

                cls._algorithm_map = {
                    # Deep learning methods
                    "DOMINANT": DOMINANT,  # Deep Anomaly Detection on Attributed Networks
                    "ANOMALOUS": ANOMALOUS,  # Graph anomaly detection with graph autoencoders
                    "GCNAE": GCNAE,  # Graph Convolutional Network Autoencoder
                    "MLPAE": MLPAE,  # Multi-Layer Perceptron Autoencoder
                    "ANOMALYDAE": ANOMALYDAE,  # Anomaly Detection with Deep Autoencoders
                    "GAAN": GAAN,  # Generative Adversarial Attributed Network
                    "GUIDE": GUIDE,  # Graph Neural Network for Anomaly Detection
                    "CONAD": CONAD,  # Contrastive Learning for Anomaly Detection
                    "GADNR": GADNR,  # Graph Anomaly Detection via Neighborhood Reconstruction
                    # Statistical methods
                    "SCAN": SCAN,  # Structural Clustering Algorithm for Networks
                    "RADAR": RADAR,  # Residual Analysis for Anomaly Detection
                }
            except ImportError as e:
                logger.error(f"Failed to import PyGOD: {e}")
                cls._algorithm_map = {}

        return cls._algorithm_map

    def __init__(
        self,
        algorithm_name: str,
        name: str | None = None,
        contamination_rate: ContaminationRate | None = None,
        **kwargs: Any,
    ):
        """Initialize PyGOD adapter with detector configuration.

        Args:
            algorithm_name: Name of the PyGOD algorithm
            name: Optional custom name for the detector
            contamination_rate: Expected contamination rate
            **kwargs: Algorithm-specific parameters
        """
        # Infrastructure state (no domain entity composition)
        self._name = name or f"PyGOD_{algorithm_name}"
        self._algorithm_name = algorithm_name
        self._contamination_rate = contamination_rate or ContaminationRate(0.1)
        self._parameters = kwargs
        self._is_fitted = False

        self._model = None
        self._init_algorithm()

    # DetectorProtocol properties
    @property
    def name(self) -> str:
        """Get the name of the detector."""
        return self._name

    @property
    def contamination_rate(self) -> ContaminationRate:
        """Get the contamination rate."""
        return self._contamination_rate

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._is_fitted

    @property
    def parameters(self) -> dict[str, Any]:
        """Get the current parameters of the detector."""
        return self._parameters

    @property
    def algorithm_name(self) -> str:
        """Get the algorithm name."""
        return self._algorithm_name

    def _init_algorithm(self) -> None:
        """Initialize the PyGOD algorithm instance."""
        algorithm_map = self._get_algorithm_map()

        if self._algorithm_name not in algorithm_map:
            available = ", ".join(algorithm_map.keys())
            raise AlgorithmNotFoundError(
                f"Algorithm '{self._algorithm_name}' not found in PyGOD. "
                f"Available algorithms: {available}"
            )

        try:
            algorithm_class = algorithm_map[self._algorithm_name]

            # Configure parameters
            params = self._parameters.copy()

            # Handle common parameter mappings
            if "contamination" in params:
                params["contamination"] = float(params["contamination"])
            else:
                params["contamination"] = 0.1  # Default

            # Algorithm-specific parameter handling
            if self.algorithm in ["DOMINANT", "GCNAE", "ANOMALOUS"]:
                # Deep learning models
                params.setdefault("hidden_dim", 64)
                params.setdefault("num_layers", 2)
                params.setdefault("epoch", 100)
                params.setdefault("dropout", 0.3)

                # GPU support
                if "gpu" in params:
                    params["gpu"] = int(params["gpu"]) if params["gpu"] else -1
                else:
                    params["gpu"] = -1  # CPU by default

            elif self.algorithm == "SCAN":
                # SCAN specific parameters
                params.setdefault("eps", 0.5)
                params.setdefault("mu", 2)

            elif self.algorithm == "RADAR":
                # RADAR specific parameters
                params.setdefault("gamma", 1.0)
                params.setdefault("k", 5)

            # Initialize the model
            self._model = algorithm_class(**params)

        except Exception as e:
            raise AdapterError(
                f"Failed to initialize PyGOD algorithm '{self.algorithm}': {e}"
            )

    def fit(self, dataset: Dataset) -> None:
        """Train the anomaly detector on the graph dataset.

        Args:
            dataset: Training dataset with graph structure
        """
        if self._model is None:
            raise AdapterError("Model not initialized")

        try:
            # Convert dataset to graph format
            graph_data = self._prepare_graph_data(dataset)

            # Fit the model
            if hasattr(self._model, "fit"):
                self._model.fit(graph_data)
            else:
                # Some models might use different training methods
                raise AdapterError(
                    f"Model {self.algorithm} does not support fit method"
                )

            self.is_fitted = True
            logger.info(f"Successfully trained PyGOD {self.algorithm}")

        except Exception as e:
            raise AdapterError(f"Failed to train PyGOD model: {e}")

    def predict(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies in the graph dataset.

        Args:
            dataset: Graph dataset to analyze

        Returns:
            Detection results with anomaly scores and labels
        """
        if not self.is_fitted:
            raise AdapterError("Model must be fitted before prediction")

        try:
            # Convert dataset to graph format
            graph_data = self._prepare_graph_data(dataset)

            # Get predictions
            if hasattr(self._model, "predict"):
                # Returns binary labels (0: normal, 1: anomaly)
                labels = self._model.predict(graph_data)

                # Get anomaly scores if available
                if hasattr(self._model, "decision_function"):
                    scores = self._model.decision_function(graph_data)
                else:
                    # Use labels as scores if no score function
                    scores = labels.astype(float)
            else:
                raise AdapterError(
                    f"Model {self.algorithm} does not support predict method"
                )

            # Normalize scores to [0, 1] range
            min_score = np.min(scores)
            max_score = np.max(scores)

            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)

            # Create anomaly scores with confidence
            anomaly_scores = [
                AnomalyScore(
                    value=float(score),
                    confidence=self._calculate_confidence(score, label),
                )
                for score, label in zip(normalized_scores, labels, strict=False)
            ]

            return DetectionResult(
                detector_id=self.name,
                dataset_id=dataset.id,
                scores=anomaly_scores,
                labels=labels.tolist(),
                metadata={
                    "algorithm": self.algorithm,
                    "n_anomalies": int(np.sum(labels)),
                    "contamination_rate": float(np.sum(labels) / len(labels)),
                    "is_graph": True,
                    "n_nodes": (
                        graph_data.x.shape[0]
                        if hasattr(graph_data, "x")
                        else len(dataset.data)
                    ),
                    "n_edges": (
                        graph_data.edge_index.shape[1]
                        if hasattr(graph_data, "edge_index")
                        else 0
                    ),
                },
            )

        except Exception as e:
            raise AdapterError(f"Failed to predict with PyGOD model: {e}")

    def _prepare_graph_data(self, dataset: Dataset):
        """Prepare data in PyGOD/PyTorch Geometric format.

        Args:
            dataset: Input dataset

        Returns:
            PyTorch Geometric Data object
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise AdapterError(
                "PyTorch Geometric is required for PyGOD. Install with: pip install torch torch-geometric"
            )

        df = dataset.data

        # Check if dataset has graph structure columns
        if "edge_index" in df.columns or (
            "source" in df.columns and "target" in df.columns
        ):
            # Edge list format
            if "edge_index" in df.columns:
                edge_index = torch.tensor(df["edge_index"].values, dtype=torch.long)
            else:
                # Convert source/target to edge index
                edge_index = torch.tensor(
                    np.array([df["source"].values, df["target"].values]),
                    dtype=torch.long,
                )

            # Node features (exclude graph structure columns)
            feature_cols = [
                col
                for col in df.columns
                if col not in ["edge_index", "source", "target", "node_id"]
            ]

            if feature_cols:
                # Use existing features
                x = torch.tensor(df[feature_cols].values, dtype=torch.float)
            else:
                # Generate dummy features if none exist
                n_nodes = max(edge_index.max().item() + 1, len(df))
                x = torch.ones((n_nodes, 1), dtype=torch.float)

        elif "adjacency_matrix" in dataset.metadata:
            # Adjacency matrix format
            adj_matrix = dataset.metadata["adjacency_matrix"]
            edge_index = self._adj_to_edge_index(adj_matrix)

            # Node features
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if dataset.target_column and dataset.target_column in feature_cols:
                feature_cols.remove(dataset.target_column)

            if feature_cols:
                x = torch.tensor(df[feature_cols].values, dtype=torch.float)
            else:
                x = torch.ones((len(df), 1), dtype=torch.float)

        else:
            # Try to infer graph structure from data
            # This is a fallback - ideally graph structure should be provided
            logger.warning(
                "No explicit graph structure found. Creating k-NN graph from features."
            )

            # Use numeric features to create k-NN graph
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if dataset.target_column and dataset.target_column in feature_cols:
                feature_cols.remove(dataset.target_column)

            if not feature_cols:
                raise AdapterError("No numeric features found to construct graph")

            x = torch.tensor(df[feature_cols].values, dtype=torch.float)

            # Create k-NN graph
            from sklearn.neighbors import kneighbors_graph

            k = min(10, len(df) - 1)  # Number of neighbors
            adj_matrix = kneighbors_graph(x.numpy(), n_neighbors=k, mode="connectivity")
            edge_index = self._adj_to_edge_index(adj_matrix)

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index)

        # Add labels if available
        if dataset.target_column and dataset.target_column in df.columns:
            y = torch.tensor(df[dataset.target_column].values, dtype=torch.long)
            data.y = y

        return data

    def _adj_to_edge_index(self, adj_matrix):
        """Convert adjacency matrix to edge index format.

        Args:
            adj_matrix: Adjacency matrix (scipy sparse or numpy array)

        Returns:
            Edge index tensor
        """
        import torch

        if hasattr(adj_matrix, "tocoo"):
            # Scipy sparse matrix
            adj_coo = adj_matrix.tocoo()
            edge_index = torch.tensor(
                np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long
            )
        else:
            # Numpy array
            rows, cols = np.where(adj_matrix > 0)
            edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)

        return edge_index

    def _calculate_confidence(self, score: float, label: int) -> float:
        """Calculate confidence score for anomaly.

        Args:
            score: Normalized anomaly score
            label: Binary label (0 or 1)

        Returns:
            Confidence value between 0 and 1
        """
        if label == 0:
            # Normal - confidence inversely related to score
            return 1.0 - score * 0.5
        else:
            # Anomaly - confidence directly related to score
            return 0.5 + score * 0.5

    @classmethod
    def get_supported_algorithms(cls) -> list[str]:
        """Get list of supported PyGOD algorithms.

        Returns:
            List of algorithm names
        """
        return list(cls._get_algorithm_map().keys())

    @classmethod
    def get_algorithm_info(cls, algorithm: str) -> dict[str, Any]:
        """Get information about a specific algorithm.

        Args:
            algorithm: Algorithm name

        Returns:
            Algorithm metadata and parameters
        """
        algorithm_map = cls._get_algorithm_map()

        if algorithm not in algorithm_map:
            raise AlgorithmNotFoundError(f"Algorithm '{algorithm}' not found")

        # Algorithm-specific information
        info = {
            "DOMINANT": {
                "name": "DOMINANT",
                "type": "Deep Learning",
                "description": "Deep Anomaly Detection on Attributed Networks using GCN autoencoder",
                "parameters": {
                    "hidden_dim": {
                        "type": "int",
                        "default": 64,
                        "description": "Hidden layer dimension",
                    },
                    "num_layers": {
                        "type": "int",
                        "default": 2,
                        "description": "Number of GCN layers",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Expected anomaly rate",
                    },
                    "epoch": {
                        "type": "int",
                        "default": 100,
                        "description": "Training epochs",
                    },
                    "dropout": {
                        "type": "float",
                        "default": 0.3,
                        "description": "Dropout rate",
                    },
                },
                "suitable_for": [
                    "attributed_graphs",
                    "social_networks",
                    "citation_networks",
                ],
                "pros": [
                    "Handles node attributes",
                    "Scalable",
                    "State-of-the-art performance",
                ],
                "cons": ["Requires GPU for large graphs", "Memory intensive"],
            },
            "GCNAE": {
                "name": "Graph Convolutional Network Autoencoder",
                "type": "Deep Learning",
                "description": "GCN-based autoencoder for graph anomaly detection",
                "parameters": {
                    "hidden_dim": {
                        "type": "int",
                        "default": 32,
                        "description": "Hidden dimension",
                    },
                    "num_layers": {
                        "type": "int",
                        "default": 2,
                        "description": "Number of layers",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Anomaly rate",
                    },
                },
                "suitable_for": ["homogeneous_graphs", "node_anomaly_detection"],
                "pros": ["Simple architecture", "Effective for node anomalies"],
                "cons": ["Limited to homogeneous graphs"],
            },
            "SCAN": {
                "name": "Structural Clustering Algorithm for Networks",
                "type": "Statistical",
                "description": "Density-based structural clustering for anomaly detection",
                "parameters": {
                    "eps": {
                        "type": "float",
                        "default": 0.5,
                        "description": "Neighborhood threshold",
                    },
                    "mu": {
                        "type": "int",
                        "default": 2,
                        "description": "Minimum cluster size",
                    },
                },
                "suitable_for": ["community_detection", "structural_anomalies"],
                "pros": [
                    "Parameter intuitive",
                    "Fast computation",
                    "No training required",
                ],
                "cons": [
                    "Sensitive to parameters",
                    "Less effective for attributed graphs",
                ],
            },
            "GAAN": {
                "name": "Generative Adversarial Attributed Network",
                "type": "Deep Learning",
                "description": "GAN-based approach for graph anomaly detection",
                "parameters": {
                    "hidden_dim": {
                        "type": "int",
                        "default": 64,
                        "description": "Hidden dimension",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Anomaly rate",
                    },
                    "epoch": {
                        "type": "int",
                        "default": 100,
                        "description": "Training epochs",
                    },
                },
                "suitable_for": ["complex_graphs", "high_dimensional_attributes"],
                "pros": ["Captures complex patterns", "Robust to noise"],
                "cons": ["Training instability", "Computationally expensive"],
            },
        }

        return info.get(
            algorithm,
            {
                "name": algorithm,
                "type": "Graph-based",
                "description": f"PyGOD implementation of {algorithm}",
                "parameters": {"contamination": {"type": "float", "default": 0.1}},
                "suitable_for": ["graph_data"],
                "pros": ["Graph structure aware"],
                "cons": ["Requires graph format"],
            },
        )
