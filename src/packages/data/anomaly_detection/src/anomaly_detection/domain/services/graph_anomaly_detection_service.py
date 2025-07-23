"""Graph anomaly detection service using PyGOD algorithms."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt

from ..entities.detection_result import DetectionResult
from ...infrastructure.logging import get_logger, log_decorator, timing_decorator
from ...infrastructure.logging.error_handler import ErrorHandler, InputValidationError, AlgorithmError
from ...infrastructure.monitoring import MetricsCollector, get_metrics_collector

logger = get_logger(__name__)
error_handler = ErrorHandler(logger._logger)
metrics_collector = get_metrics_collector()


class GraphAnomalyDetectionService:
    """Service for graph-based anomaly detection using PyGOD algorithms.
    
    Supports various graph neural network approaches for detecting anomalous nodes
    and edges in graph structures.
    """
    
    def __init__(self):
        """Initialize graph anomaly detection service."""
        self._fitted_models: Dict[str, Any] = {}
        self._graph_cache: Dict[str, Any] = {}
    
    @log_decorator(operation="graph_anomaly_detection", log_args=True, log_duration=True)
    def detect_anomalies(
        self,
        graph_data: Union[Dict[str, Any], npt.NDArray[np.floating], Tuple],
        algorithm: str = "gcn",
        contamination: float = 0.1,
        **kwargs: Any
    ) -> DetectionResult:
        """Detect anomalies in graph data.
        
        Args:
            graph_data: Graph data in various formats:
                - Dict with 'edge_index', 'x' (node features)
                - Adjacency matrix as numpy array
                - Tuple of (edge_index, node_features)
            algorithm: Algorithm to use ('gcn', 'gaan', 'anomalydae', 'radar', 'dominant')
            contamination: Expected proportion of anomalies
            **kwargs: Additional algorithm parameters
            
        Returns:
            DetectionResult with graph-specific metadata
        """
        try:
            # Parse and validate graph data
            edge_index, node_features, num_nodes = self._parse_graph_data(graph_data)
            
            # Validate inputs
            self._validate_inputs(edge_index, node_features, algorithm, contamination)
            
            # Select and run algorithm
            if algorithm == "gcn":
                predictions, scores = self._gcn_detect(edge_index, node_features, contamination, **kwargs)
            elif algorithm == "gaan":
                predictions, scores = self._gaan_detect(edge_index, node_features, contamination, **kwargs)
            elif algorithm == "anomalydae":
                predictions, scores = self._anomalydae_detect(edge_index, node_features, contamination, **kwargs)
            elif algorithm == "radar":
                predictions, scores = self._radar_detect(edge_index, node_features, contamination, **kwargs)
            elif algorithm == "dominant":
                predictions, scores = self._dominant_detect(edge_index, node_features, contamination, **kwargs)
            elif algorithm == "simple_graph":
                predictions, scores = self._simple_graph_detect(edge_index, node_features, contamination, **kwargs)
            else:
                raise AlgorithmError(
                    f"Unknown graph algorithm: {algorithm}",
                    details={
                        "requested_algorithm": algorithm,
                        "available_algorithms": self.list_available_algorithms()
                    }
                )
            
            # Create result with graph metadata
            result = DetectionResult(
                predictions=predictions,
                confidence_scores=scores,
                algorithm=algorithm,
                metadata={
                    "contamination": contamination,
                    "algorithm_params": kwargs,
                    "is_graph": True,
                    "num_nodes": num_nodes,
                    "num_edges": edge_index.shape[1] if edge_index is not None else 0,
                    "node_features_dim": node_features.shape[1] if node_features is not None else 0,
                    "graph_stats": self._calculate_graph_stats(edge_index, num_nodes)
                }
            )
            
            # Log detection statistics
            logger.info("Graph anomaly detection completed successfully", 
                       algorithm=algorithm,
                       total_nodes=result.total_samples,
                       anomalies_detected=result.anomaly_count,
                       anomaly_rate=result.anomaly_rate,
                       num_edges=result.metadata["num_edges"])
            
            return result
            
        except Exception as e:
            return error_handler.handle_error(
                error=e,
                context={
                    "algorithm": algorithm,
                    "contamination": contamination,
                    "graph_type": type(graph_data).__name__
                },
                operation="graph_anomaly_detection",
                reraise=True
            )
    
    @timing_decorator(operation="gcn_detection")
    def _gcn_detect(
        self,
        edge_index: npt.NDArray[np.integer],
        node_features: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Graph Convolutional Network based anomaly detection."""
        try:
            # Try to import PyGOD
            try:
                from pygod.detector import GCNAE
            except ImportError:
                logger.warning("PyGOD not available, falling back to simple graph method")
                return self._simple_graph_detect(edge_index, node_features, contamination, **kwargs)
            
            # Parameters
            hidden_dim = kwargs.get('hidden_dim', 64)
            num_layers = kwargs.get('num_layers', 2)  
            dropout = kwargs.get('dropout', 0.3)
            lr = kwargs.get('lr', 0.004)
            epoch = kwargs.get('epoch', 100)
            
            logger.debug("Running GCN anomaly detection", 
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        num_nodes=node_features.shape[0])
            
            # Convert to PyTorch format if needed
            graph_data = self._prepare_torch_geometric_data(edge_index, node_features)
            
            # Initialize and fit GCN detector
            detector = GCNAE(
                hid_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
                epoch=epoch,
                contamination=contamination,
                verbose=0
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detector.fit(graph_data)
            
            # Get predictions and scores
            predictions = detector.predict(graph_data)
            scores = detector.decision_function(graph_data)
            
            # Convert to sklearn format (-1 for anomaly, 1 for normal)
            predictions = np.where(predictions == 1, -1, 1).astype(np.integer)
            
            # Normalize scores
            scores = self._normalize_scores(scores)
            
            logger.debug("GCN detection completed", 
                        anomalies_found=np.sum(predictions == -1))
            
            return predictions, scores
            
        except Exception as e:
            raise AlgorithmError(
                f"GCN detection failed: {str(e)}",
                details={"contamination": contamination, "num_nodes": node_features.shape[0]},
                original_error=e
            )
    
    @timing_decorator(operation="gaan_detection")
    def _gaan_detect(
        self,
        edge_index: npt.NDArray[np.integer],
        node_features: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Graph Attention Anomaly Network detection."""
        try:
            try:
                from pygod.detector import GAAN
            except ImportError:
                logger.warning("PyGOD not available, falling back to simple graph method")
                return self._simple_graph_detect(edge_index, node_features, contamination, **kwargs)
            
            # Parameters
            hidden_dim = kwargs.get('hidden_dim', 32)
            num_layers = kwargs.get('num_layers', 2)
            dropout = kwargs.get('dropout', 0.3)
            lr = kwargs.get('lr', 0.005)
            epoch = kwargs.get('epoch', 100)
            
            logger.debug("Running GAAN detection", 
                        hidden_dim=hidden_dim,
                        num_layers=num_layers)
            
            # Prepare data
            graph_data = self._prepare_torch_geometric_data(edge_index, node_features)
            
            # Initialize detector
            detector = GAAN(
                hid_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
                epoch=epoch,
                contamination=contamination,
                verbose=0
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detector.fit(graph_data)
            
            predictions = detector.predict(graph_data)
            scores = detector.decision_function(graph_data)
            
            predictions = np.where(predictions == 1, -1, 1).astype(np.integer)
            scores = self._normalize_scores(scores)
            
            logger.debug("GAAN detection completed", 
                        anomalies_found=np.sum(predictions == -1))
            
            return predictions, scores
            
        except Exception as e:
            raise AlgorithmError(
                f"GAAN detection failed: {str(e)}",
                details={"contamination": contamination, "num_nodes": node_features.shape[0]},
                original_error=e
            )
    
    @timing_decorator(operation="anomalydae_detection")
    def _anomalydae_detect(
        self,
        edge_index: npt.NDArray[np.integer],
        node_features: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """AnomalyDAE detection for graph data."""
        try:
            try:
                from pygod.detector import AnomalyDAE
            except ImportError:
                logger.warning("PyGOD not available, falling back to simple graph method")
                return self._simple_graph_detect(edge_index, node_features, contamination, **kwargs)
            
            # Parameters
            hidden_dim = kwargs.get('hidden_dim', 64)
            num_layers = kwargs.get('num_layers', 4)
            dropout = kwargs.get('dropout', 0.3)
            lr = kwargs.get('lr', 0.005)
            epoch = kwargs.get('epoch', 100)
            
            logger.debug("Running AnomalyDAE detection")
            
            # Prepare data
            graph_data = self._prepare_torch_geometric_data(edge_index, node_features)
            
            # Initialize detector
            detector = AnomalyDAE(
                hid_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
                epoch=epoch,
                contamination=contamination,
                verbose=0
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detector.fit(graph_data)
            
            predictions = detector.predict(graph_data)
            scores = detector.decision_function(graph_data)
            
            predictions = np.where(predictions == 1, -1, 1).astype(np.integer)
            scores = self._normalize_scores(scores)
            
            logger.debug("AnomalyDAE detection completed", 
                        anomalies_found=np.sum(predictions == -1))
            
            return predictions, scores
            
        except Exception as e:
            raise AlgorithmError(
                f"AnomalyDAE detection failed: {str(e)}",
                details={"contamination": contamination, "num_nodes": node_features.shape[0]},
                original_error=e
            )
    
    @timing_decorator(operation="radar_detection")
    def _radar_detect(
        self,
        edge_index: npt.NDArray[np.integer],
        node_features: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """RADAR detection for graph anomalies."""
        try:
            try:
                from pygod.detector import RADAR
            except ImportError:
                logger.warning("PyGOD not available, falling back to simple graph method")
                return self._simple_graph_detect(edge_index, node_features, contamination, **kwargs)
            
            # Parameters
            gamma = kwargs.get('gamma', 1.0)
            lr = kwargs.get('lr', 0.01)
            epoch = kwargs.get('epoch', 200)
            
            logger.debug("Running RADAR detection")
            
            # Prepare data
            graph_data = self._prepare_torch_geometric_data(edge_index, node_features)
            
            # Initialize detector
            detector = RADAR(
                gamma=gamma,
                lr=lr,
                epoch=epoch,
                contamination=contamination,
                verbose=0
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detector.fit(graph_data)
            
            predictions = detector.predict(graph_data)
            scores = detector.decision_function(graph_data)
            
            predictions = np.where(predictions == 1, -1, 1).astype(np.integer)
            scores = self._normalize_scores(scores)
            
            logger.debug("RADAR detection completed", 
                        anomalies_found=np.sum(predictions == -1))
            
            return predictions, scores
            
        except Exception as e:
            raise AlgorithmError(
                f"RADAR detection failed: {str(e)}",
                details={"contamination": contamination, "num_nodes": node_features.shape[0]},
                original_error=e
            )
    
    @timing_decorator(operation="dominant_detection")
    def _dominant_detect(
        self,
        edge_index: npt.NDArray[np.integer],
        node_features: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """DOMINANT detection for graph anomalies."""
        try:
            try:
                from pygod.detector import DOMINANT
            except ImportError:
                logger.warning("PyGOD not available, falling back to simple graph method")
                return self._simple_graph_detect(edge_index, node_features, contamination, **kwargs)
            
            # Parameters
            hidden_dim = kwargs.get('hidden_dim', 64)
            num_layers = kwargs.get('num_layers', 4)
            dropout = kwargs.get('dropout', 0.3)
            lr = kwargs.get('lr', 0.005)
            epoch = kwargs.get('epoch', 100)
            
            logger.debug("Running DOMINANT detection")
            
            # Prepare data
            graph_data = self._prepare_torch_geometric_data(edge_index, node_features)
            
            # Initialize detector
            detector = DOMINANT(
                hid_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
                epoch=epoch,
                contamination=contamination,
                verbose=0
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detector.fit(graph_data)
            
            predictions = detector.predict(graph_data)
            scores = detector.decision_function(graph_data)
            
            predictions = np.where(predictions == 1, -1, 1).astype(np.integer)
            scores = self._normalize_scores(scores)
            
            logger.debug("DOMINANT detection completed", 
                        anomalies_found=np.sum(predictions == -1))
            
            return predictions, scores
            
        except Exception as e:
            raise AlgorithmError(
                f"DOMINANT detection failed: {str(e)}",
                details={"contamination": contamination, "num_nodes": node_features.shape[0]},
                original_error=e
            )
    
    @timing_decorator(operation="simple_graph_detection")
    def _simple_graph_detect(
        self,
        edge_index: npt.NDArray[np.integer],
        node_features: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Simple graph-based anomaly detection using graph statistics."""
        try:
            logger.debug("Running simple graph-based detection")
            
            num_nodes = node_features.shape[0]
            
            # Calculate node-level graph statistics
            node_scores = np.zeros(num_nodes)
            
            # 1. Degree centrality anomalies
            degrees = self._calculate_node_degrees(edge_index, num_nodes)
            degree_scores = np.abs(degrees - np.mean(degrees)) / (np.std(degrees) + 1e-8)
            
            # 2. Feature-based anomalies using simple statistical methods
            feature_scores = np.zeros(num_nodes)
            if node_features.shape[1] > 0:
                # Z-score based on feature means
                feature_means = np.mean(node_features, axis=1)
                global_feature_mean = np.mean(feature_means)
                global_feature_std = np.std(feature_means)
                if global_feature_std > 0:
                    feature_scores = np.abs(feature_means - global_feature_mean) / global_feature_std
            
            # 3. Local clustering coefficient anomalies
            clustering_scores = self._calculate_clustering_anomalies(edge_index, num_nodes)
            
            # Combine scores
            node_scores = (0.4 * degree_scores + 0.4 * feature_scores + 0.2 * clustering_scores)
            
            # Determine threshold and predictions
            threshold = np.percentile(node_scores, (1 - contamination) * 100)
            predictions = np.where(node_scores > threshold, -1, 1).astype(np.integer)
            
            # Normalize confidence scores
            confidence_scores = self._normalize_scores(node_scores)
            
            logger.debug("Simple graph detection completed", 
                        anomalies_found=np.sum(predictions == -1),
                        avg_degree=np.mean(degrees),
                        threshold=threshold)
            
            return predictions, confidence_scores
            
        except Exception as e:
            raise AlgorithmError(
                f"Simple graph detection failed: {str(e)}",
                details={"contamination": contamination, "num_nodes": num_nodes},
                original_error=e
            )
    
    def _parse_graph_data(
        self,
        graph_data: Union[Dict[str, Any], npt.NDArray[np.floating], Tuple]
    ) -> Tuple[Optional[npt.NDArray[np.integer]], npt.NDArray[np.floating], int]:
        """Parse graph data from various input formats."""
        if isinstance(graph_data, dict):
            # PyTorch Geometric style dict
            edge_index = graph_data.get('edge_index')
            node_features = graph_data.get('x', graph_data.get('node_features'))
            
            if edge_index is not None:
                edge_index = np.array(edge_index, dtype=np.int32)
            if node_features is not None:
                node_features = np.array(node_features, dtype=np.float64)
                
        elif isinstance(graph_data, tuple) and len(graph_data) == 2:
            # (edge_index, node_features) tuple
            edge_index, node_features = graph_data
            edge_index = np.array(edge_index, dtype=np.int32) if edge_index is not None else None
            node_features = np.array(node_features, dtype=np.float64)
            
        elif isinstance(graph_data, np.ndarray):
            # Adjacency matrix
            adj_matrix = graph_data
            edge_index = self._adjacency_to_edge_index(adj_matrix)
            # Create simple node features (node degrees or identity)
            num_nodes = adj_matrix.shape[0]
            degrees = np.sum(adj_matrix, axis=1, keepdims=True)
            node_features = degrees.astype(np.float64)
            
        else:
            raise InputValidationError(
                f"Unsupported graph data format: {type(graph_data)}. "
                f"Expected dict, tuple, or numpy array."
            )
        
        # Validate and determine number of nodes
        if node_features is None:
            raise InputValidationError("Node features are required")
        
        num_nodes = node_features.shape[0]
        
        # Validate edge_index if provided
        if edge_index is not None:
            if edge_index.shape[0] != 2:
                raise InputValidationError(
                    f"Edge index must have shape (2, num_edges), got {edge_index.shape}"
                )
            if np.max(edge_index) >= num_nodes:
                raise InputValidationError(
                    f"Edge index contains node indices >= num_nodes ({num_nodes})"
                )
        
        return edge_index, node_features, num_nodes
    
    def _validate_inputs(
        self,
        edge_index: Optional[npt.NDArray[np.integer]],
        node_features: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float
    ) -> None:
        """Validate inputs for graph anomaly detection."""
        if node_features.size == 0:
            raise InputValidationError("Node features cannot be empty")
        
        if len(node_features.shape) != 2:
            raise InputValidationError(
                f"Node features must be 2D, got shape {node_features.shape}"
            )
        
        if node_features.shape[0] < 3:
            raise InputValidationError(
                f"Need at least 3 nodes, got {node_features.shape[0]}"
            )
        
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise InputValidationError("Algorithm name must be a non-empty string")
        
        if not (0.001 <= contamination <= 0.5):
            raise InputValidationError(
                f"Contamination must be between 0.001 and 0.5, got {contamination}"
            )
        
        # Check for non-finite values
        if not np.isfinite(node_features).all():
            raise InputValidationError("Node features contain non-finite values (NaN or inf)")
        
        if edge_index is not None and not np.isfinite(edge_index).all():
            raise InputValidationError("Edge index contains non-finite values")
    
    def _prepare_torch_geometric_data(
        self,
        edge_index: npt.NDArray[np.integer],
        node_features: npt.NDArray[np.floating]
    ) -> Any:
        """Prepare data for PyTorch Geometric format."""
        try:
            import torch
            from torch_geometric.data import Data
            
            # Convert to torch tensors
            x = torch.FloatTensor(node_features)
            
            if edge_index is not None:
                edge_index_tensor = torch.LongTensor(edge_index)
            else:
                # Create empty edge index
                edge_index_tensor = torch.LongTensor(2, 0)
            
            # Create Data object
            data = Data(x=x, edge_index=edge_index_tensor)
            return data
            
        except ImportError:
            raise AlgorithmError(
                "PyTorch and PyTorch Geometric required for advanced graph algorithms",
                details={"missing_dependencies": ["torch", "torch_geometric"]}
            )
    
    def _adjacency_to_edge_index(self, adj_matrix: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Convert adjacency matrix to edge index format."""
        # Find non-zero entries (edges)
        edges = np.nonzero(adj_matrix)
        edge_index = np.vstack(edges).astype(np.int32)
        return edge_index
    
    def _calculate_node_degrees(self, edge_index: Optional[npt.NDArray[np.integer]], num_nodes: int) -> npt.NDArray[np.floating]:
        """Calculate node degrees from edge index."""
        if edge_index is None:
            return np.zeros(num_nodes, dtype=np.float64)
        
        degrees = np.zeros(num_nodes, dtype=np.float64)
        unique, counts = np.unique(edge_index.flatten(), return_counts=True)
        degrees[unique] = counts
        return degrees
    
    def _calculate_clustering_anomalies(
        self,
        edge_index: Optional[npt.NDArray[np.integer]],
        num_nodes: int
    ) -> npt.NDArray[np.floating]:
        """Calculate clustering coefficient based anomaly scores."""
        if edge_index is None:
            return np.zeros(num_nodes, dtype=np.float64)
        
        # Simple clustering coefficient approximation
        degrees = self._calculate_node_degrees(edge_index, num_nodes)
        
        # Nodes with very high or very low degrees relative to the graph
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)
        
        if std_degree > 0:
            clustering_scores = np.abs(degrees - mean_degree) / std_degree
        else:
            clustering_scores = np.zeros(num_nodes, dtype=np.float64)
        
        return clustering_scores
    
    def _calculate_graph_stats(self, edge_index: Optional[npt.NDArray[np.integer]], num_nodes: int) -> Dict[str, Any]:
        """Calculate basic graph statistics."""
        stats = {"num_nodes": num_nodes}
        
        if edge_index is not None:
            num_edges = edge_index.shape[1]
            degrees = self._calculate_node_degrees(edge_index, num_nodes)
            
            stats.update({
                "num_edges": num_edges,
                "avg_degree": float(np.mean(degrees)),
                "max_degree": int(np.max(degrees)),
                "min_degree": int(np.min(degrees)),
                "density": float(2 * num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0
            })
        else:
            stats.update({
                "num_edges": 0,
                "avg_degree": 0.0,
                "max_degree": 0,
                "min_degree": 0,
                "density": 0.0
            })
        
        return stats
    
    def _normalize_scores(self, scores: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Normalize anomaly scores to [0, 1] range."""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > min_score:
            normalized = (scores - min_score) / (max_score - min_score)
        else:
            normalized = np.zeros_like(scores)
        
        return normalized.astype(np.float64)
    
    def list_available_algorithms(self) -> List[str]:
        """List all available graph algorithms."""
        return ["gcn", "gaan", "anomalydae", "radar", "dominant", "simple_graph"]
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """Get information about a graph algorithm."""
        info = {"name": algorithm, "type": "graph"}
        
        if algorithm == "gcn":
            info.update({
                "description": "Graph Convolutional Network Autoencoder",
                "requires": ["pygod", "torch", "torch_geometric"],
                "supports_node_features": True,
                "supports_edge_features": False
            })
        elif algorithm == "gaan":
            info.update({
                "description": "Graph Attention Anomaly Network",
                "requires": ["pygod", "torch", "torch_geometric"],
                "supports_node_features": True,
                "supports_edge_features": False
            })
        elif algorithm == "anomalydae":
            info.update({
                "description": "Deep Autoencoder for Graph Anomaly Detection",
                "requires": ["pygod", "torch", "torch_geometric"],
                "supports_node_features": True,
                "supports_edge_features": False
            })
        elif algorithm == "radar":
            info.update({
                "description": "Residual Analysis for Anomaly Detection in Attributed Networks",
                "requires": ["pygod", "torch", "torch_geometric"],
                "supports_node_features": True,
                "supports_edge_features": False
            })
        elif algorithm == "dominant":
            info.update({
                "description": "Deep Anomaly Detection on Attributed Networks",
                "requires": ["pygod", "torch", "torch_geometric"],
                "supports_node_features": True,
                "supports_edge_features": False
            })
        elif algorithm == "simple_graph":
            info.update({
                "description": "Simple statistical graph anomaly detection",
                "requires": [],
                "supports_node_features": True,
                "supports_edge_features": False
            })
        
        return info