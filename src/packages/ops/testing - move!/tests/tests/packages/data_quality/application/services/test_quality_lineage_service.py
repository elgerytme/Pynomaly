"""Tests for Quality Lineage Service."""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.packages.data_quality.application.services.quality_lineage_service import (
    QualityLineageService, LineageServiceConfig, ImpactPropagationRule
)
from src.packages.data_quality.domain.entities.quality_lineage import (
    QualityLineageGraph, LineageNode, LineageEdge, QualityImpactAnalysis,
    LineageNodeId, LineageEdgeId, LineageNodeType, ImpactType, ImpactSeverity
)
from src.packages.data_quality.domain.entities.quality_anomaly import QualityAnomaly
from src.packages.data_quality.domain.entities.quality_profile import DataQualityProfile


class TestQualityLineageService:
    """Test suite for Quality Lineage Service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = LineageServiceConfig(
            max_propagation_depth=5,
            min_impact_threshold=0.1,
            enable_caching=True,
            enable_parallel_processing=False,  # Disable for tests
            enable_persistent_cache=False  # Disable for tests
        )
        self.service = QualityLineageService(self.config)
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service.config == self.config
        assert isinstance(self.service._lineage_graphs, dict)
        assert isinstance(self.service._impact_cache, dict)
        assert isinstance(self.service._propagation_rules, list)
        assert len(self.service._propagation_rules) > 0
    
    def test_service_initialization_with_defaults(self):
        """Test service initialization with default config."""
        service = QualityLineageService()
        assert service.config is not None
        assert service.config.max_propagation_depth == 10
        assert service.config.enable_caching is True
    
    def test_create_lineage_graph(self):
        """Test creating a lineage graph."""
        graph = self.service.create_lineage_graph(
            graph_name="Test Graph",
            description="Test description"
        )
        
        assert isinstance(graph, QualityLineageGraph)
        assert graph.name == "Test Graph"
        assert graph.description == "Test description"
        assert graph.graph_id in self.service._lineage_graphs
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
    
    def test_add_dataset_node(self):
        """Test adding a dataset node."""
        graph = self.service.create_lineage_graph()
        
        # Create mock quality profile
        quality_profile = Mock(spec=DataQualityProfile)
        quality_profile.overall_quality_score = 0.8
        quality_profile.row_count = 1000
        quality_profile.column_count = 10
        
        node = self.service.add_dataset_node(
            graph=graph,
            dataset_id="test_dataset",
            dataset_name="Test Dataset",
            quality_profile=quality_profile,
            custom_attr="test_value"
        )
        
        assert isinstance(node, LineageNode)
        assert node.node_type == LineageNodeType.DATASET
        assert node.name == "Test Dataset"
        assert node.quality_score == 0.8
        assert node.attributes['dataset_id'] == "test_dataset"
        assert node.attributes['custom_attr'] == "test_value"
        assert node.attributes['row_count'] == 1000
        assert node.attributes['column_count'] == 10
        assert len(graph.nodes) == 1
    
    def test_add_column_node(self):
        """Test adding a column node."""
        graph = self.service.create_lineage_graph()
        dataset_node = self.service.add_dataset_node(
            graph, "test_dataset", "Test Dataset"
        )
        
        column_node = self.service.add_column_node(
            graph=graph,
            dataset_node=dataset_node,
            column_name="test_column",
            column_type="STRING",
            quality_score=0.9
        )
        
        assert isinstance(column_node, LineageNode)
        assert column_node.node_type == LineageNodeType.COLUMN
        assert column_node.name == "test_column"
        assert column_node.quality_score == 0.9
        assert column_node.attributes['column_name'] == "test_column"
        assert column_node.attributes['column_type'] == "STRING"
        assert column_node.attributes['parent_dataset'] == str(dataset_node.node_id)
        
        # Check that relationship edge was created
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        
        # Verify the edge
        edge = list(graph.edges.values())[0]
        assert edge.source_node_id == dataset_node.node_id
        assert edge.target_node_id == column_node.node_id
        assert edge.relationship_type == "contains"
    
    def test_add_transformation_node(self):
        """Test adding a transformation node."""
        graph = self.service.create_lineage_graph()
        
        # Create input and output nodes
        input_node = self.service.add_dataset_node(graph, "input", "Input Dataset")
        output_node = self.service.add_dataset_node(graph, "output", "Output Dataset")
        
        transformation_node = self.service.add_transformation_node(
            graph=graph,
            transformation_name="Data Cleaning",
            transformation_type="cleaning",
            input_nodes=[input_node],
            output_nodes=[output_node]
        )
        
        assert isinstance(transformation_node, LineageNode)
        assert transformation_node.node_type == LineageNodeType.TRANSFORMATION
        assert transformation_node.name == "Data Cleaning"
        assert transformation_node.attributes['transformation_type'] == "cleaning"
        assert transformation_node.attributes['input_count'] == 1
        assert transformation_node.attributes['output_count'] == 1
        
        # Check edges were created (2 input edges + 2 output edges)
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
    
    def test_add_downstream_system_node(self):
        """Test adding a downstream system node."""
        graph = self.service.create_lineage_graph()
        source_dataset = self.service.add_dataset_node(graph, "source", "Source Dataset")
        
        system_node = self.service.add_downstream_system_node(
            graph=graph,
            system_name="ML Model",
            system_type="ml_model",
            source_datasets=[source_dataset]
        )
        
        assert isinstance(system_node, LineageNode)
        assert system_node.node_type == LineageNodeType.ML_MODEL
        assert system_node.name == "ML Model"
        assert system_node.attributes['system_type'] == "ml_model"
        assert system_node.attributes['source_count'] == 1
        
        # Check dependency edge was created
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        
        edge = list(graph.edges.values())[0]
        assert edge.source_node_id == source_dataset.node_id
        assert edge.target_node_id == system_node.node_id
        assert edge.relationship_type == "consumed_by"
        assert edge.impact_type == ImpactType.AMPLIFIED  # ML models have amplified impact
    
    def test_analyze_quality_impact_downstream(self):
        """Test downstream quality impact analysis."""
        graph = self._create_test_graph()
        root_node = list(graph.nodes.values())[0]
        
        analysis = self.service.analyze_quality_impact(
            graph=graph,
            root_node_id=root_node.node_id,
            impact_scope="downstream"
        )
        
        assert isinstance(analysis, QualityImpactAnalysis)
        assert analysis.root_node_id == root_node.node_id
        assert analysis.impact_scope == "downstream"
        assert len(analysis.impacted_nodes) > 0
        assert analysis.total_nodes_analyzed > 0
        assert analysis.analysis_duration_seconds > 0
    
    def test_analyze_quality_impact_upstream(self):
        """Test upstream quality impact analysis."""
        graph = self._create_test_graph()
        # Get a node that has upstream dependencies
        target_node = list(graph.nodes.values())[-1]
        
        analysis = self.service.analyze_quality_impact(
            graph=graph,
            root_node_id=target_node.node_id,
            impact_scope="upstream"
        )
        
        assert isinstance(analysis, QualityImpactAnalysis)
        assert analysis.root_node_id == target_node.node_id
        assert analysis.impact_scope == "upstream"
        assert analysis.total_nodes_analyzed > 0
    
    def test_analyze_quality_impact_with_caching(self):
        """Test impact analysis with caching."""
        graph = self._create_test_graph()
        root_node = list(graph.nodes.values())[0]
        
        # First analysis
        analysis1 = self.service.analyze_quality_impact(
            graph=graph,
            root_node_id=root_node.node_id,
            impact_scope="downstream"
        )
        
        # Second analysis should be cached
        analysis2 = self.service.analyze_quality_impact(
            graph=graph,
            root_node_id=root_node.node_id,
            impact_scope="downstream"
        )
        
        assert analysis1.analysis_id == analysis2.analysis_id
        assert analysis1.analysis_date == analysis2.analysis_date
    
    def test_analyze_quality_impact_large_graph(self):
        """Test impact analysis for large graphs."""
        # Create a large graph that exceeds max_nodes_per_analysis
        large_config = LineageServiceConfig(
            max_nodes_per_analysis=5,  # Very small for testing
            enable_caching=False,
            enable_parallel_processing=False,
            enable_persistent_cache=False
        )
        service = QualityLineageService(large_config)
        
        graph = self._create_test_graph(num_nodes=10)  # Exceeds the limit
        root_node = list(graph.nodes.values())[0]
        
        analysis = service.analyze_quality_impact(
            graph=graph,
            root_node_id=root_node.node_id,
            impact_scope="downstream"
        )
        
        # Should still complete successfully using optimized algorithm
        assert isinstance(analysis, QualityImpactAnalysis)
        assert analysis.total_nodes_analyzed > 0
    
    def test_perform_root_cause_analysis(self):
        """Test root cause analysis."""
        graph = self._create_test_graph()
        
        # Get a node to analyze
        issue_node = list(graph.nodes.values())[-1]  # Last node in chain
        
        analysis = self.service.perform_root_cause_analysis(
            graph=graph,
            quality_issue_node_id=issue_node.node_id,
            analysis_depth=3
        )
        
        assert 'issue_node' in analysis
        assert 'root_causes' in analysis
        assert 'contributing_factors' in analysis
        assert 'analysis_summary' in analysis
        assert 'recommendations' in analysis
        assert 'next_steps' in analysis
        
        assert analysis['issue_node']['node_id'] == str(issue_node.node_id)
        assert analysis['analysis_summary']['analysis_depth'] == 3
        assert isinstance(analysis['recommendations'], list)
        assert isinstance(analysis['next_steps'], list)
    
    def test_find_quality_dependencies(self):
        """Test finding quality dependencies."""
        graph = self._create_test_graph()
        middle_node = list(graph.nodes.values())[1]  # Middle node should have both
        
        dependencies = self.service.find_quality_dependencies(
            graph=graph,
            node_id=middle_node.node_id,
            dependency_type="all"
        )
        
        assert 'upstream' in dependencies
        assert 'downstream' in dependencies
        assert 'critical_dependencies' in dependencies
        assert 'weak_dependencies' in dependencies
        
        assert isinstance(dependencies['upstream'], list)
        assert isinstance(dependencies['downstream'], list)
    
    def test_get_quality_lineage_summary(self):
        """Test getting quality lineage summary."""
        graph = self._create_test_graph()
        
        summary = self.service.get_quality_lineage_summary(graph)
        
        assert 'graph_statistics' in summary
        assert 'quality_metrics' in summary
        assert 'critical_paths' in summary
        assert 'quality_bottlenecks' in summary
        assert 'risk_assessment' in summary
        assert 'recommendations' in summary
        
        # Check quality metrics
        quality_metrics = summary['quality_metrics']
        assert 'average_quality_score' in quality_metrics
        assert 'total_quality_issues' in quality_metrics
        assert 'quality_coverage' in quality_metrics
        
        # Check risk assessment
        risk_assessment = summary['risk_assessment']
        assert 'overall_risk' in risk_assessment
        assert 'risk_score' in risk_assessment
        assert 'risk_factors' in risk_assessment
    
    def test_benchmark_performance(self):
        """Test performance benchmarking."""
        # Use small scenarios for testing
        test_scenarios = [
            {
                'name': 'Small Test',
                'type': 'impact_analysis',
                'nodes': 5,
                'edges': 5,
                'depth': 3
            },
            {
                'name': 'Lineage Test',
                'type': 'lineage_discovery',
                'tables': 10,
                'columns_per_table': 3,
                'relationships': 5
            }
        ]
        
        results = self.service.benchmark_performance(test_scenarios)
        
        assert 'benchmark_timestamp' in results
        assert 'performance_targets' in results
        assert 'test_results' in results
        assert 'overall_performance' in results
        assert 'performance_summary' in results
        
        assert len(results['test_results']) == 2
        
        for test_result in results['test_results']:
            assert 'scenario_name' in test_result
            assert 'scenario_type' in test_result
            assert 'duration_seconds' in test_result
            assert 'target_met' in test_result
    
    def test_export_lineage_graph(self):
        """Test exporting lineage graph."""
        graph = self._create_test_graph()
        
        export_data = self.service.export_lineage_graph(
            graph=graph,
            format="json",
            include_analysis=True
        )
        
        assert 'graph_id' in export_data
        assert 'name' in export_data
        assert 'nodes' in export_data
        assert 'edges' in export_data
        assert 'statistics' in export_data
        assert 'lineage_summary' in export_data
        
        # Check that nodes and edges are properly serialized
        assert len(export_data['nodes']) == len(graph.nodes)
        assert len(export_data['edges']) == len(graph.edges)
    
    def test_get_lineage_graph(self):
        """Test getting lineage graph by ID."""
        graph = self.service.create_lineage_graph("Test Graph")
        
        retrieved_graph = self.service.get_lineage_graph(graph.graph_id)
        assert retrieved_graph is not None
        assert retrieved_graph.graph_id == graph.graph_id
        assert retrieved_graph.name == "Test Graph"
        
        # Test with non-existent ID
        non_existent = self.service.get_lineage_graph("non-existent")
        assert non_existent is None
    
    def test_list_lineage_graphs(self):
        """Test listing lineage graphs."""
        # Create some graphs
        graph1 = self.service.create_lineage_graph("Graph 1")
        graph2 = self.service.create_lineage_graph("Graph 2")
        
        graph_list = self.service.list_lineage_graphs()
        
        assert len(graph_list) == 2
        assert all('graph_id' in g for g in graph_list)
        assert all('name' in g for g in graph_list)
        assert all('node_count' in g for g in graph_list)
        assert all('edge_count' in g for g in graph_list)
        
        graph_names = [g['name'] for g in graph_list]
        assert "Graph 1" in graph_names
        assert "Graph 2" in graph_names
    
    def test_propagation_rules_initialization(self):
        """Test that propagation rules are properly initialized."""
        rules = self.service._propagation_rules
        
        assert len(rules) > 0
        assert all(isinstance(rule, ImpactPropagationRule) for rule in rules)
        
        # Check some specific rules
        rule_types = [(rule.source_node_type, rule.target_node_type) for rule in rules]
        assert (LineageNodeType.DATA_SOURCE, LineageNodeType.DATASET) in rule_types
        assert (LineageNodeType.DATASET, LineageNodeType.COLUMN) in rule_types
        assert (LineageNodeType.DATASET, LineageNodeType.ML_MODEL) in rule_types
    
    def test_calculate_impact_severity(self):
        """Test impact severity calculation."""
        graph = self._create_test_graph()
        node = list(graph.nodes.values())[0]
        
        # Test different impact scores
        critical_severity = self.service._calculate_impact_severity(0.9, node)
        assert critical_severity == ImpactSeverity.CRITICAL
        
        high_severity = self.service._calculate_impact_severity(0.7, node)
        assert high_severity == ImpactSeverity.HIGH
        
        medium_severity = self.service._calculate_impact_severity(0.5, node)
        assert medium_severity == ImpactSeverity.MEDIUM
        
        low_severity = self.service._calculate_impact_severity(0.3, node)
        assert low_severity == ImpactSeverity.LOW
        
        minimal_severity = self.service._calculate_impact_severity(0.1, node)
        assert minimal_severity == ImpactSeverity.MINIMAL
    
    def test_calculate_root_cause_score(self):
        """Test root cause score calculation."""
        graph = self._create_test_graph()
        candidate_node = list(graph.nodes.values())[0]
        issue_node = list(graph.nodes.values())[-1]
        
        # Test with data source node (should have high score)
        candidate_node.node_type = LineageNodeType.DATA_SOURCE
        candidate_node.quality_score = 0.3  # Low quality
        
        score = self.service._calculate_root_cause_score(candidate_node, issue_node, 1)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be relatively high for data source with low quality
    
    def test_memory_optimization_config(self):
        """Test memory optimization configuration."""
        config = LineageServiceConfig(
            enable_memory_optimization=True,
            lazy_loading=True,
            node_attribute_compression=True
        )
        
        service = QualityLineageService(config)
        assert service.config.enable_memory_optimization is True
        assert service.config.lazy_loading is True
        assert service.config.node_attribute_compression is True
    
    def test_error_handling_invalid_node(self):
        """Test error handling for invalid node operations."""
        graph = self.service.create_lineage_graph()
        invalid_node_id = LineageNodeId()
        
        with pytest.raises(ValueError, match="Root node not found"):
            self.service.analyze_quality_impact(
                graph=graph,
                root_node_id=invalid_node_id,
                impact_scope="downstream"
            )
        
        with pytest.raises(ValueError, match="Quality issue node not found"):
            self.service.perform_root_cause_analysis(
                graph=graph,
                quality_issue_node_id=invalid_node_id
            )
    
    def _create_test_graph(self, num_nodes: int = 5) -> QualityLineageGraph:
        """Create a test graph with connected nodes."""
        graph = self.service.create_lineage_graph("Test Graph")
        
        # Create a chain of connected nodes
        nodes = []
        for i in range(num_nodes):
            node_type = LineageNodeType.DATASET if i % 2 == 0 else LineageNodeType.COLUMN
            node = self.service.add_dataset_node(
                graph=graph,
                dataset_id=f"test_dataset_{i}",
                dataset_name=f"Test Dataset {i}"
            )
            node.node_type = node_type
            node.quality_score = 0.8 if i % 2 == 0 else 0.6
            nodes.append(node)
        
        # Connect nodes in a chain
        for i in range(len(nodes) - 1):
            edge = LineageEdge(
                edge_id=LineageEdgeId(),
                source_node_id=nodes[i].node_id,
                target_node_id=nodes[i + 1].node_id,
                relationship_type="feeds",
                quality_impact_factor=0.8,
                impact_type=ImpactType.DIRECT
            )
            graph.add_edge(edge)
        
        return graph


class TestLineageServiceConfig:
    """Test suite for LineageServiceConfig."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = LineageServiceConfig()
        
        assert config.max_propagation_depth == 10
        assert config.min_impact_threshold == 0.1
        assert config.enable_impact_scoring is True
        assert config.max_nodes_per_analysis == 1000000
        assert config.enable_caching is True
        assert config.cache_ttl_hours == 24
        assert config.enable_parallel_processing is True
        assert config.max_workers == 4
        assert config.batch_size == 1000
    
    def test_config_customization(self):
        """Test configuration customization."""
        config = LineageServiceConfig(
            max_propagation_depth=15,
            min_impact_threshold=0.05,
            enable_caching=False,
            max_workers=8,
            batch_size=500
        )
        
        assert config.max_propagation_depth == 15
        assert config.min_impact_threshold == 0.05
        assert config.enable_caching is False
        assert config.max_workers == 8
        assert config.batch_size == 500


class TestImpactPropagationRule:
    """Test suite for ImpactPropagationRule."""
    
    def test_rule_creation(self):
        """Test creating impact propagation rules."""
        rule = ImpactPropagationRule(
            source_node_type=LineageNodeType.DATA_SOURCE,
            target_node_type=LineageNodeType.DATASET,
            relationship_type="feeds",
            impact_factor=0.9,
            impact_type=ImpactType.DIRECT,
            conditions={"min_quality": 0.5}
        )
        
        assert rule.source_node_type == LineageNodeType.DATA_SOURCE
        assert rule.target_node_type == LineageNodeType.DATASET
        assert rule.relationship_type == "feeds"
        assert rule.impact_factor == 0.9
        assert rule.impact_type == ImpactType.DIRECT
        assert rule.conditions == {"min_quality": 0.5}