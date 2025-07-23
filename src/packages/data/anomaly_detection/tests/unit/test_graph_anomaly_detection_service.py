#!/usr/bin/env python3
"""Tests for graph anomaly detection service."""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

from anomaly_detection.domain.services.graph_anomaly_detection_service import GraphAnomalyDetectionService
from anomaly_detection.domain.entities.detection_result import DetectionResult


class TestGraphAnomalyDetectionService:
    """Test graph anomaly detection service functionality."""
    
    @pytest.fixture
    def graph_service(self):
        """Create graph anomaly detection service."""
        return GraphAnomalyDetectionService()
    
    @pytest.fixture
    def sample_graph_dict(self):
        """Generate sample graph data as dictionary."""
        # Create a small graph with 10 nodes
        np.random.seed(42)
        
        # Edge index for a simple graph (COO format)
        edge_index = np.array([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 8, 7, 9, 8]
        ], dtype=np.int32)
        
        # Node features (10 nodes, 3 features each)
        node_features = np.random.randn(10, 3).astype(np.float64)
        
        # Add some anomalous nodes
        node_features[7] += [3, -3, 2]  # Anomalous node
        node_features[9] += [-2, 4, -1]  # Another anomalous node
        
        return {
            'edge_index': edge_index,
            'x': node_features
        }
    
    @pytest.fixture
    def sample_adjacency_matrix(self):
        """Generate sample adjacency matrix."""
        # 8x8 adjacency matrix
        adj = np.array([
            [0, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0]
        ], dtype=np.float64)
        return adj
    
    @pytest.fixture
    def sample_edge_tuple(self):
        """Generate sample graph as edge index tuple."""
        edge_index = np.array([
            [0, 1, 2, 2, 3, 4, 4, 5],
            [1, 2, 1, 3, 4, 3, 5, 4]
        ], dtype=np.int32)
        
        node_features = np.random.randn(6, 2).astype(np.float64)
        node_features[0] += [2, -2]  # Anomalous node
        
        return (edge_index, node_features)
    
    def test_simple_graph_detection_dict_input(self, graph_service, sample_graph_dict):
        """Test simple graph detection with dictionary input."""
        result = graph_service.detect_anomalies(
            graph_data=sample_graph_dict,
            algorithm="simple_graph",
            contamination=0.2
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.algorithm == "simple_graph"
        assert result.total_samples == 10
        assert result.anomaly_count > 0
        assert result.anomaly_count <= 3  # Should detect some anomalies
        assert len(result.predictions) == 10
        assert len(result.confidence_scores) == 10
        assert result.metadata["is_graph"] is True
        assert result.metadata["num_nodes"] == 10
        assert result.metadata["num_edges"] == 16
        assert result.metadata["node_features_dim"] == 3
        
        print(f"Dict input: {result.anomaly_count} anomalies detected")
    
    def test_simple_graph_detection_adjacency_matrix(self, graph_service, sample_adjacency_matrix):
        """Test simple graph detection with adjacency matrix."""
        result = graph_service.detect_anomalies(
            graph_data=sample_adjacency_matrix,
            algorithm="simple_graph",
            contamination=0.25
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.algorithm == "simple_graph"
        assert result.total_samples == 8
        assert result.anomaly_count >= 0
        assert len(result.predictions) == 8
        assert result.metadata["num_nodes"] == 8
        assert result.metadata["node_features_dim"] == 1  # Degree features
        
        print(f"Adjacency matrix: {result.anomaly_count} anomalies detected")
    
    def test_simple_graph_detection_tuple_input(self, graph_service, sample_edge_tuple):
        """Test simple graph detection with tuple input."""
        result = graph_service.detect_anomalies(
            graph_data=sample_edge_tuple,
            algorithm="simple_graph",
            contamination=0.15
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.algorithm == "simple_graph"
        assert result.total_samples == 6
        assert len(result.predictions) == 6
        assert result.metadata["num_nodes"] == 6
        assert result.metadata["node_features_dim"] == 2
        
        print(f"Tuple input: {result.anomaly_count} anomalies detected")
    
    def test_gcn_detection_fallback(self, graph_service, sample_graph_dict):
        """Test GCN detection (should fallback to simple_graph if PyGOD not available)."""
        result = graph_service.detect_anomalies(
            graph_data=sample_graph_dict,
            algorithm="gcn",
            contamination=0.2,
            hidden_dim=32,
            num_layers=2
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        # Algorithm might be gcn or simple_graph (fallback)
        assert result.algorithm in ["gcn", "simple_graph"]
        assert result.total_samples == 10
        assert len(result.predictions) == 10
        
        print(f"GCN detection: {result.anomaly_count} anomalies detected (algorithm: {result.algorithm})")
    
    def test_gaan_detection_fallback(self, graph_service, sample_graph_dict):
        """Test GAAN detection (should fallback to simple_graph if PyGOD not available)."""
        result = graph_service.detect_anomalies(
            graph_data=sample_graph_dict,
            algorithm="gaan",
            contamination=0.2,
            hidden_dim=16,
            epoch=50
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.algorithm in ["gaan", "simple_graph"]
        assert result.total_samples == 10
        assert len(result.predictions) == 10
        
        print(f"GAAN detection: {result.anomaly_count} anomalies detected (algorithm: {result.algorithm})")
    
    def test_anomalydae_detection_fallback(self, graph_service, sample_graph_dict):
        """Test AnomalyDAE detection."""
        result = graph_service.detect_anomalies(
            graph_data=sample_graph_dict,
            algorithm="anomalydae",
            contamination=0.2,
            hidden_dim=64,
            num_layers=3
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.algorithm in ["anomalydae", "simple_graph"]
        assert result.total_samples == 10
        
        print(f"AnomalyDAE detection: {result.anomaly_count} anomalies detected (algorithm: {result.algorithm})")
    
    def test_radar_detection_fallback(self, graph_service, sample_graph_dict):
        """Test RADAR detection."""
        result = graph_service.detect_anomalies(
            graph_data=sample_graph_dict,
            algorithm="radar",
            contamination=0.2,
            gamma=1.5,
            epoch=100
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.algorithm in ["radar", "simple_graph"]
        assert result.total_samples == 10
        
        print(f"RADAR detection: {result.anomaly_count} anomalies detected (algorithm: {result.algorithm})")
    
    def test_dominant_detection_fallback(self, graph_service, sample_graph_dict):
        """Test DOMINANT detection."""
        result = graph_service.detect_anomalies(
            graph_data=sample_graph_dict,
            algorithm="dominant",
            contamination=0.2,
            hidden_dim=32
        )
        
        assert isinstance(result, DetectionResult)
        assert result.success
        assert result.algorithm in ["dominant", "simple_graph"]
        assert result.total_samples == 10
        
        print(f"DOMINANT detection: {result.anomaly_count} anomalies detected (algorithm: {result.algorithm})")
    
    def test_algorithm_info(self, graph_service):
        """Test algorithm information retrieval."""
        algorithms = graph_service.list_available_algorithms()
        assert "gcn" in algorithms
        assert "gaan" in algorithms
        assert "anomalydae" in algorithms
        assert "radar" in algorithms
        assert "dominant" in algorithms
        assert "simple_graph" in algorithms
        
        # Test individual algorithm info
        gcn_info = graph_service.get_algorithm_info("gcn")
        assert gcn_info["type"] == "graph"
        assert "pygod" in gcn_info["requires"]
        assert gcn_info["supports_node_features"] is True
        
        simple_info = graph_service.get_algorithm_info("simple_graph")
        assert simple_info["type"] == "graph"
        assert simple_info["requires"] == []
    
    def test_graph_statistics_calculation(self, graph_service, sample_graph_dict):
        """Test graph statistics calculation."""
        result = graph_service.detect_anomalies(
            graph_data=sample_graph_dict,
            algorithm="simple_graph",
            contamination=0.2
        )
        
        stats = result.metadata["graph_stats"]
        assert stats["num_nodes"] == 10
        assert stats["num_edges"] == 16
        assert stats["avg_degree"] > 0
        assert stats["max_degree"] >= stats["min_degree"]
        assert 0 <= stats["density"] <= 1
        
        print(f"Graph stats: {stats}")
    
    def test_error_handling(self, graph_service):
        """Test error handling for invalid inputs."""
        # Empty node features
        with pytest.raises(Exception):
            graph_service.detect_anomalies(
                graph_data={"edge_index": np.array([[0, 1], [1, 0]]), "x": np.array([])},
                algorithm="simple_graph"
            )
        
        # Too few nodes
        with pytest.raises(Exception):
            graph_service.detect_anomalies(
                graph_data={"edge_index": np.array([[0, 1], [1, 0]]), "x": np.array([[1], [2]])},
                algorithm="simple_graph"
            )
        
        # Invalid algorithm
        with pytest.raises(Exception):
            graph_service.detect_anomalies(
                graph_data={"edge_index": np.array([[0, 1, 2], [1, 2, 0]]), "x": np.random.randn(3, 2)},
                algorithm="invalid_algorithm"
            )
        
        # Invalid contamination
        with pytest.raises(Exception):
            graph_service.detect_anomalies(
                graph_data={"edge_index": np.array([[0, 1, 2], [1, 2, 0]]), "x": np.random.randn(3, 2)},
                algorithm="simple_graph",
                contamination=1.5
            )
        
        # Invalid edge index shape
        with pytest.raises(Exception):
            graph_service.detect_anomalies(
                graph_data={"edge_index": np.array([0, 1, 2]), "x": np.random.randn(3, 2)},
                algorithm="simple_graph"
            )
        
        # Node index out of bounds
        with pytest.raises(Exception):
            graph_service.detect_anomalies(
                graph_data={"edge_index": np.array([[0, 5], [1, 2]]), "x": np.random.randn(3, 2)},
                algorithm="simple_graph"
            )
    
    def test_data_format_parsing(self, graph_service):
        """Test different graph data format parsing."""
        # Test with missing edge_index (nodes only)
        node_only_data = {"x": np.random.randn(5, 2)}
        result = graph_service.detect_anomalies(
            graph_data=node_only_data,
            algorithm="simple_graph",
            contamination=0.2
        )
        assert result.success
        assert result.metadata["num_edges"] == 0
        
        # Test tuple with None edge_index
        tuple_data = (None, np.random.randn(4, 3))
        result = graph_service.detect_anomalies(
            graph_data=tuple_data,
            algorithm="simple_graph",
            contamination=0.25
        )
        assert result.success
        assert result.metadata["num_edges"] == 0
    
    def test_confidence_scores_normalization(self, graph_service, sample_graph_dict):
        """Test confidence scores are properly normalized."""
        result = graph_service.detect_anomalies(
            graph_data=sample_graph_dict,
            algorithm="simple_graph",
            contamination=0.2
        )
        
        # Check confidence scores are in [0, 1] range
        assert np.all(result.confidence_scores >= 0)
        assert np.all(result.confidence_scores <= 1)
        
        # Check that higher scores correspond to anomalies
        anomaly_indices = np.where(result.predictions == -1)[0]
        normal_indices = np.where(result.predictions == 1)[0]
        
        if len(anomaly_indices) > 0 and len(normal_indices) > 0:
            avg_anomaly_confidence = np.mean(result.confidence_scores[anomaly_indices])
            avg_normal_confidence = np.mean(result.confidence_scores[normal_indices])
            # Anomalies should generally have higher confidence scores
            assert avg_anomaly_confidence >= avg_normal_confidence or abs(avg_anomaly_confidence - avg_normal_confidence) < 0.1


if __name__ == "__main__":
    print("Graph Anomaly Detection Service Test")
    print("=" * 40)
    
    # Quick smoke test
    service = GraphAnomalyDetectionService()
    
    # Generate test graph
    np.random.seed(42)
    test_graph = {
        'edge_index': np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int32),
        'x': np.random.randn(3, 2).astype(np.float64)
    }
    test_graph['x'][2] += [3, -3]  # Make node 2 anomalous
    
    try:
        result = service.detect_anomalies(
            graph_data=test_graph,
            algorithm="simple_graph",
            contamination=0.3
        )
        
        print(f"✓ Quick test passed: {result.anomaly_count} anomalies detected")
        print(f"  Algorithm: {result.algorithm}")
        print(f"  Graph stats: {result.metadata['graph_stats']}")
        print("Ready to run comprehensive graph tests")
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        print("Graph tests may not run properly")