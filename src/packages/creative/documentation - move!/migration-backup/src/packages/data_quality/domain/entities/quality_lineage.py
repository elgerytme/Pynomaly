"""Quality lineage and impact analysis entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from uuid import uuid4


class LineageNodeType(Enum):
    """Types of nodes in quality lineage."""
    DATA_SOURCE = "data_source"
    DATASET = "dataset"
    COLUMN = "column"
    TRANSFORMATION = "transformation"
    QUALITY_RULE = "quality_rule"
    QUALITY_ISSUE = "quality_issue"
    DOWNSTREAM_SYSTEM = "downstream_system"
    REPORT = "report"
    DASHBOARD = "dashboard"
    ML_MODEL = "ml_model"
    API_ENDPOINT = "api_endpoint"


class ImpactType(Enum):
    """Types of quality impact."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CASCADING = "cascading"
    AMPLIFIED = "amplified"
    MITIGATED = "mitigated"


class ImpactSeverity(Enum):
    """Severity levels for quality impact."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class LineageNodeId:
    """Unique identifier for lineage node."""
    value: str = field(default_factory=lambda: str(uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class LineageEdgeId:
    """Unique identifier for lineage edge."""
    value: str = field(default_factory=lambda: str(uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass
class LineageNode:
    """Node in the quality lineage graph."""
    
    node_id: LineageNodeId
    node_type: LineageNodeType
    name: str
    description: str
    
    # Node attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Quality information
    quality_score: Optional[float] = None
    last_quality_check: Optional[datetime] = None
    quality_issues: List[str] = field(default_factory=list)
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Location and ownership
    system_name: Optional[str] = None
    owner: Optional[str] = None
    location: Optional[str] = None
    
    def update_quality_score(self, score: float) -> None:
        """Update quality score."""
        self.quality_score = score
        self.last_quality_check = datetime.now()
        self.updated_at = datetime.now()
    
    def add_quality_issue(self, issue_id: str) -> None:
        """Add quality issue."""
        if issue_id not in self.quality_issues:
            self.quality_issues.append(issue_id)
            self.updated_at = datetime.now()
    
    def remove_quality_issue(self, issue_id: str) -> None:
        """Remove quality issue."""
        if issue_id in self.quality_issues:
            self.quality_issues.remove(issue_id)
            self.updated_at = datetime.now()
    
    def add_tag(self, tag: str) -> None:
        """Add tag."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def update_attribute(self, key: str, value: Any) -> None:
        """Update attribute."""
        self.attributes[key] = value
        self.updated_at = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get node summary."""
        return {
            'node_id': str(self.node_id),
            'node_type': self.node_type.value,
            'name': self.name,
            'description': self.description,
            'quality_score': self.quality_score,
            'quality_issues_count': len(self.quality_issues),
            'system_name': self.system_name,
            'owner': self.owner,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': self.tags
        }


@dataclass
class LineageEdge:
    """Edge in the quality lineage graph representing relationships."""
    
    edge_id: LineageEdgeId
    source_node_id: LineageNodeId
    target_node_id: LineageNodeId
    relationship_type: str
    
    # Edge attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Quality propagation
    quality_impact_factor: float = 1.0
    impact_type: ImpactType = ImpactType.DIRECT
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    last_traversed: Optional[datetime] = None
    
    # Relationship strength
    strength: float = 1.0
    confidence: float = 1.0
    
    def update_traversal(self) -> None:
        """Update last traversal time."""
        self.last_traversed = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get edge summary."""
        return {
            'edge_id': str(self.edge_id),
            'source_node_id': str(self.source_node_id),
            'target_node_id': str(self.target_node_id),
            'relationship_type': self.relationship_type,
            'quality_impact_factor': self.quality_impact_factor,
            'impact_type': self.impact_type.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'last_traversed': self.last_traversed.isoformat() if self.last_traversed else None
        }


@dataclass
class QualityImpactAnalysis:
    """Analysis of quality impact propagation."""
    
    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    root_node_id: LineageNodeId = None
    impact_scope: str = "downstream"  # upstream, downstream, both
    
    # Impact results
    impacted_nodes: List[Dict[str, Any]] = field(default_factory=list)
    impact_summary: Dict[str, Any] = field(default_factory=dict)
    propagation_paths: List[List[LineageNodeId]] = field(default_factory=list)
    
    # Analysis metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    analysis_duration_seconds: float = 0.0
    total_nodes_analyzed: int = 0
    
    def add_impacted_node(self, 
                         node_id: LineageNodeId,
                         node_name: str,
                         impact_severity: ImpactSeverity,
                         impact_type: ImpactType,
                         impact_score: float,
                         propagation_path: List[LineageNodeId]) -> None:
        """Add impacted node to analysis."""
        self.impacted_nodes.append({
            'node_id': str(node_id),
            'node_name': node_name,
            'impact_severity': impact_severity.value,
            'impact_type': impact_type.value,
            'impact_score': impact_score,
            'propagation_path': [str(node_id) for node_id in propagation_path],
            'path_length': len(propagation_path)
        })
    
    def calculate_summary(self) -> None:
        """Calculate impact summary."""
        if not self.impacted_nodes:
            self.impact_summary = {
                'total_impacted_nodes': 0,
                'severity_breakdown': {},
                'impact_type_breakdown': {},
                'avg_impact_score': 0.0,
                'max_propagation_depth': 0
            }
            return
        
        # Calculate summary statistics
        severity_counts = {}
        impact_type_counts = {}
        
        for node in self.impacted_nodes:
            severity = node['impact_severity']
            impact_type = node['impact_type']
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            impact_type_counts[impact_type] = impact_type_counts.get(impact_type, 0) + 1
        
        avg_impact_score = sum(node['impact_score'] for node in self.impacted_nodes) / len(self.impacted_nodes)
        max_depth = max(node['path_length'] for node in self.impacted_nodes) if self.impacted_nodes else 0
        
        self.impact_summary = {
            'total_impacted_nodes': len(self.impacted_nodes),
            'severity_breakdown': severity_counts,
            'impact_type_breakdown': impact_type_counts,
            'avg_impact_score': round(avg_impact_score, 3),
            'max_propagation_depth': max_depth,
            'critical_nodes': [
                node for node in self.impacted_nodes
                if node['impact_severity'] == ImpactSeverity.CRITICAL.value
            ]
        }
    
    def get_critical_impacts(self) -> List[Dict[str, Any]]:
        """Get critical impact nodes."""
        return [
            node for node in self.impacted_nodes
            if node['impact_severity'] in [ImpactSeverity.CRITICAL.value, ImpactSeverity.HIGH.value]
        ]
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """Get comprehensive analysis report."""
        return {
            'analysis_id': self.analysis_id,
            'root_node_id': str(self.root_node_id) if self.root_node_id else None,
            'impact_scope': self.impact_scope,
            'analysis_date': self.analysis_date.isoformat(),
            'analysis_duration_seconds': self.analysis_duration_seconds,
            'total_nodes_analyzed': self.total_nodes_analyzed,
            'impact_summary': self.impact_summary,
            'critical_impacts': self.get_critical_impacts(),
            'propagation_paths': [
                [str(node_id) for node_id in path]
                for path in self.propagation_paths
            ],
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on impact analysis."""
        recommendations = []
        
        if self.impact_summary.get('total_impacted_nodes', 0) > 10:
            recommendations.append("High impact scope detected - consider targeted quality improvements")
        
        critical_count = len(self.get_critical_impacts())
        if critical_count > 0:
            recommendations.append(f"{critical_count} critical impacts require immediate attention")
        
        max_depth = self.impact_summary.get('max_propagation_depth', 0)
        if max_depth > 3:
            recommendations.append("Deep propagation paths detected - consider impact mitigation strategies")
        
        return recommendations


@dataclass
class QualityLineageGraph:
    """Quality lineage graph containing nodes and edges."""
    
    graph_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "Quality Lineage Graph"
    description: str = "Graph representing quality relationships and dependencies"
    
    # Graph components
    nodes: Dict[str, LineageNode] = field(default_factory=dict)
    edges: Dict[str, LineageEdge] = field(default_factory=dict)
    
    # Graph metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    # Indexes for efficient traversal
    _outgoing_edges: Dict[str, List[str]] = field(default_factory=dict)
    _incoming_edges: Dict[str, List[str]] = field(default_factory=dict)
    _node_type_index: Dict[LineageNodeType, List[str]] = field(default_factory=dict)
    
    def add_node(self, node: LineageNode) -> None:
        """Add node to graph."""
        node_id_str = str(node.node_id)
        self.nodes[node_id_str] = node
        
        # Update indexes
        if node.node_type not in self._node_type_index:
            self._node_type_index[node.node_type] = []
        self._node_type_index[node.node_type].append(node_id_str)
        
        if node_id_str not in self._outgoing_edges:
            self._outgoing_edges[node_id_str] = []
        if node_id_str not in self._incoming_edges:
            self._incoming_edges[node_id_str] = []
        
        self.updated_at = datetime.now()
    
    def add_edge(self, edge: LineageEdge) -> None:
        """Add edge to graph."""
        edge_id_str = str(edge.edge_id)
        source_id_str = str(edge.source_node_id)
        target_id_str = str(edge.target_node_id)
        
        # Validate nodes exist
        if source_id_str not in self.nodes:
            raise ValueError(f"Source node {source_id_str} not found in graph")
        if target_id_str not in self.nodes:
            raise ValueError(f"Target node {target_id_str} not found in graph")
        
        self.edges[edge_id_str] = edge
        
        # Update indexes
        self._outgoing_edges[source_id_str].append(edge_id_str)
        self._incoming_edges[target_id_str].append(edge_id_str)
        
        self.updated_at = datetime.now()
    
    def get_node(self, node_id: LineageNodeId) -> Optional[LineageNode]:
        """Get node by ID."""
        return self.nodes.get(str(node_id))
    
    def get_edge(self, edge_id: LineageEdgeId) -> Optional[LineageEdge]:
        """Get edge by ID."""
        return self.edges.get(str(edge_id))
    
    def get_nodes_by_type(self, node_type: LineageNodeType) -> List[LineageNode]:
        """Get nodes by type."""
        node_ids = self._node_type_index.get(node_type, [])
        return [self.nodes[node_id] for node_id in node_ids]
    
    def get_downstream_nodes(self, node_id: LineageNodeId) -> List[LineageNode]:
        """Get downstream nodes."""
        node_id_str = str(node_id)
        downstream_nodes = []
        
        for edge_id in self._outgoing_edges.get(node_id_str, []):
            edge = self.edges[edge_id]
            downstream_node = self.nodes[str(edge.target_node_id)]
            downstream_nodes.append(downstream_node)
        
        return downstream_nodes
    
    def get_upstream_nodes(self, node_id: LineageNodeId) -> List[LineageNode]:
        """Get upstream nodes."""
        node_id_str = str(node_id)
        upstream_nodes = []
        
        for edge_id in self._incoming_edges.get(node_id_str, []):
            edge = self.edges[edge_id]
            upstream_node = self.nodes[str(edge.source_node_id)]
            upstream_nodes.append(upstream_node)
        
        return upstream_nodes
    
    def get_outgoing_edges(self, node_id: LineageNodeId) -> List[LineageEdge]:
        """Get outgoing edges from node."""
        node_id_str = str(node_id)
        return [self.edges[edge_id] for edge_id in self._outgoing_edges.get(node_id_str, [])]
    
    def get_incoming_edges(self, node_id: LineageNodeId) -> List[LineageEdge]:
        """Get incoming edges to node."""
        node_id_str = str(node_id)
        return [self.edges[edge_id] for edge_id in self._incoming_edges.get(node_id_str, [])]
    
    def find_path(self, source_id: LineageNodeId, target_id: LineageNodeId) -> Optional[List[LineageNodeId]]:
        """Find path between two nodes using BFS."""
        if str(source_id) not in self.nodes or str(target_id) not in self.nodes:
            return None
        
        if source_id == target_id:
            return [source_id]
        
        # BFS to find shortest path
        queue = [(source_id, [source_id])]
        visited = {str(source_id)}
        
        while queue:
            current_node, path = queue.pop(0)
            
            # Get downstream nodes
            for downstream_node in self.get_downstream_nodes(current_node):
                downstream_id = downstream_node.node_id
                downstream_id_str = str(downstream_id)
                
                if downstream_id == target_id:
                    return path + [downstream_id]
                
                if downstream_id_str not in visited:
                    visited.add(downstream_id_str)
                    queue.append((downstream_id, path + [downstream_id]))
        
        return None
    
    def find_all_paths(self, 
                      source_id: LineageNodeId, 
                      target_id: LineageNodeId,
                      max_depth: int = 10) -> List[List[LineageNodeId]]:
        """Find all paths between two nodes."""
        paths = []
        
        def dfs(current_id: LineageNodeId, path: List[LineageNodeId], depth: int):
            if depth > max_depth:
                return
            
            if current_id == target_id:
                paths.append(path.copy())
                return
            
            for downstream_node in self.get_downstream_nodes(current_id):
                downstream_id = downstream_node.node_id
                if downstream_id not in path:  # Avoid cycles
                    path.append(downstream_id)
                    dfs(downstream_id, path, depth + 1)
                    path.pop()
        
        dfs(source_id, [source_id], 0)
        return paths
    
    def get_connected_component(self, node_id: LineageNodeId) -> Set[LineageNodeId]:
        """Get all nodes in the same connected component."""
        component = set()
        stack = [node_id]
        
        while stack:
            current_id = stack.pop()
            current_id_str = str(current_id)
            
            if current_id_str in component:
                continue
            
            component.add(current_id_str)
            
            # Add upstream and downstream nodes
            for node in self.get_upstream_nodes(current_id) + self.get_downstream_nodes(current_id):
                if str(node.node_id) not in component:
                    stack.append(node.node_id)
        
        return {LineageNodeId(node_id) for node_id in component}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'graph_id': self.graph_id,
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_type_distribution': {
                node_type.value: len(node_ids)
                for node_type, node_ids in self._node_type_index.items()
            },
            'avg_node_degree': sum(len(edges) for edges in self._outgoing_edges.values()) / len(self.nodes) if self.nodes else 0,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version
        }
    
    def export_graph(self, format: str = 'json') -> Dict[str, Any]:
        """Export graph in specified format."""
        if format == 'json':
            return {
                'graph_id': self.graph_id,
                'name': self.name,
                'description': self.description,
                'nodes': [node.get_summary() for node in self.nodes.values()],
                'edges': [edge.get_summary() for edge in self.edges.values()],
                'statistics': self.get_graph_statistics()
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def remove_node(self, node_id: LineageNodeId) -> None:
        """Remove node and associated edges."""
        node_id_str = str(node_id)
        
        if node_id_str not in self.nodes:
            return
        
        # Remove associated edges
        edges_to_remove = []
        
        # Remove outgoing edges
        for edge_id in self._outgoing_edges.get(node_id_str, []):
            edges_to_remove.append(edge_id)
        
        # Remove incoming edges
        for edge_id in self._incoming_edges.get(node_id_str, []):
            edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            self._remove_edge_by_id(edge_id)
        
        # Remove node
        node = self.nodes[node_id_str]
        del self.nodes[node_id_str]
        
        # Update indexes
        self._node_type_index[node.node_type].remove(node_id_str)
        del self._outgoing_edges[node_id_str]
        del self._incoming_edges[node_id_str]
        
        self.updated_at = datetime.now()
    
    def _remove_edge_by_id(self, edge_id: str) -> None:
        """Remove edge by ID."""
        if edge_id not in self.edges:
            return
        
        edge = self.edges[edge_id]
        source_id_str = str(edge.source_node_id)
        target_id_str = str(edge.target_node_id)
        
        # Remove from indexes
        self._outgoing_edges[source_id_str].remove(edge_id)
        self._incoming_edges[target_id_str].remove(edge_id)
        
        # Remove edge
        del self.edges[edge_id]