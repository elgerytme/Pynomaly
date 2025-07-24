"""
Data Lineage Domain Entities

Defines the domain model for data lineage tracking, including nodes, edges,
and the complete lineage graph structure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class LineageNodeType(str):
    """Types of nodes in the data lineage graph."""
    
    SOURCE = "source"
    TRANSFORM = "transform"
    SINK = "sink"
    MODEL = "model"
    FEATURE = "feature"
    DATASET = "dataset"
    PIPELINE = "pipeline"
    SERVICE = "service"


class LineageRelationType(str, Enum):
    """Types of relationships between lineage nodes."""
    
    DERIVES_FROM = "derives_from"
    TRANSFORMS_TO = "transforms_to"
    DEPENDS_ON = "depends_on"
    GENERATES = "generates"
    CONSUMES = "consumes"
    VALIDATES = "validates"
    ENRICHES = "enriches"


@dataclass(frozen=True)
class LineageMetadata:
    """Metadata associated with lineage nodes and edges."""
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the metadata."""
        self.tags.add(tag)
    
    def set_property(self, key: str, value: Any) -> None:
        """Set a property in the metadata."""
        self.properties[key] = value
        self.updated_at = datetime.utcnow()


class LineageNode(BaseModel):
    """Represents a node in the data lineage graph."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Human-readable name of the node")
    type: LineageNodeType = Field(..., description="Type of the lineage node")
    namespace: str = Field(..., description="Namespace/domain of the node")
    description: Optional[str] = None
    asset_id: Optional[UUID] = None # Link to DataCatalogEntry
    metadata: LineageMetadata = Field(default_factory=LineageMetadata)
    
    # Data-specific attributes
    # schema_: Optional[Dict[str, Any]] = Field(None, alias="schema") # Removed, part of DataCatalogEntry
    # location: Optional[str] = None # Removed, part of DataCatalogEntry
    size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    
    # Quality attributes
    quality_score: Optional[float] = None
    last_validated: Optional[datetime] = None
    validation_status: Optional[str] = None
    
    # Operational attributes
    last_updated: Optional[datetime] = None
    update_frequency: Optional[str] = None
    owner: Optional[str] = None
    criticality: Optional[str] = None
    
    model_config = ConfigDict(
        use_enum_values=True,
        populate_by_name=True
    )
    
    def update_quality_score(self, score: float) -> None:
        """Update the quality score for this node."""
        self.quality_score = score
        self.last_validated = datetime.utcnow()
        self.metadata.updated_at = datetime.utcnow()
    
    def set_validation_status(self, status: str) -> None:
        """Set the validation status for this node."""
        self.validation_status = status
        self.last_validated = datetime.utcnow()
        self.metadata.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to this node."""
        self.metadata.add_tag(tag)
    
    def set_property(self, key: str, value: Any) -> None:
        """Set a property on this node."""
        self.metadata.set_property(key, value)


class LineageEdge(BaseModel):
    """Represents an edge (relationship) in the data lineage graph."""
    
    id: UUID = Field(default_factory=uuid4)
    source_node_id: UUID = Field(..., description="ID of the source node")
    target_node_id: UUID = Field(..., description="ID of the target node")
    relationship_type: LineageRelationType = Field(..., description="Type of relationship")
    metadata: LineageMetadata = Field(default_factory=LineageMetadata)
    
    # Transform-specific attributes
    transform_logic: Optional[str] = None
    transform_type: Optional[str] = None
    column_mapping: Optional[Dict[str, str]] = None
    
    # Quality attributes
    data_quality_impact: Optional[float] = None
    error_rate: Optional[float] = None
    
    # Operational attributes
    pipeline_id: Optional[UUID] = None
    execution_time: Optional[float] = None
    last_executed: Optional[datetime] = None
    
    model_config = ConfigDict(
        use_enum_values=True
    )
    
    def set_transform_logic(self, logic: str, transform_type: str = None) -> None:
        """Set the transformation logic for this edge."""
        self.transform_logic = logic
        if transform_type:
            self.transform_type = transform_type
        self.metadata.updated_at = datetime.utcnow()
    
    def set_column_mapping(self, mapping: Dict[str, str]) -> None:
        """Set the column mapping for this transformation."""
        self.column_mapping = mapping
        self.metadata.updated_at = datetime.utcnow()
    
    def update_execution_stats(self, execution_time: float, error_rate: float = None) -> None:
        """Update execution statistics for this edge."""
        self.execution_time = execution_time
        self.last_executed = datetime.utcnow()
        if error_rate is not None:
            self.error_rate = error_rate
        self.metadata.updated_at = datetime.utcnow()


class DataLineage(BaseModel):
    """Represents a complete data lineage graph."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Name of the lineage graph")
    description: Optional[str] = None
    metadata: LineageMetadata = Field(default_factory=LineageMetadata)
    
    nodes: Dict[UUID, LineageNode] = Field(default_factory=dict)
    edges: Dict[UUID, LineageEdge] = Field(default_factory=dict)
    
    # Graph-level attributes
    version: str = "1.0"
    namespace: str = "default"
    
    model_config = ConfigDict(
        use_enum_values=True
    )
    
    def add_node(self, node: LineageNode) -> None:
        """Add a node to the lineage graph."""
        self.nodes[node.id] = node
        self.metadata.updated_at = datetime.utcnow()
    
    def add_edge(self, edge: LineageEdge) -> None:
        """Add an edge to the lineage graph."""
        if edge.source_node_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_node_id} not found in lineage")
        if edge.target_node_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_node_id} not found in lineage")
        
        self.edges[edge.id] = edge
        self.metadata.updated_at = datetime.utcnow()
    
    def get_node(self, node_id: UUID) -> Optional[LineageNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: UUID) -> Optional[LineageEdge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
    
    def get_upstream_nodes(self, node_id: UUID) -> List[LineageNode]:
        """Get all upstream nodes for a given node."""
        upstream_nodes = []
        for edge in self.edges.values():
            if edge.target_node_id == node_id:
                source_node = self.nodes.get(edge.source_node_id)
                if source_node:
                    upstream_nodes.append(source_node)
        return upstream_nodes
    
    def get_downstream_nodes(self, node_id: UUID) -> List[LineageNode]:
        """Get all downstream nodes for a given node."""
        downstream_nodes = []
        for edge in self.edges.values():
            if edge.source_node_id == node_id:
                target_node = self.nodes.get(edge.target_node_id)
                if target_node:
                    downstream_nodes.append(target_node)
        return downstream_nodes
    
    def get_path(self, source_node_id: UUID, target_node_id: UUID) -> List[LineageNode]:
        """Get the path between two nodes using BFS."""
        if source_node_id == target_node_id:
            return [self.nodes[source_node_id]]
        
        visited = set()
        queue = [(source_node_id, [source_node_id])]
        
        while queue:
            current_node_id, path = queue.pop(0)
            
            if current_node_id in visited:
                continue
            
            visited.add(current_node_id)
            
            # Get downstream nodes
            for edge in self.edges.values():
                if edge.source_node_id == current_node_id:
                    next_node_id = edge.target_node_id
                    
                    if next_node_id == target_node_id:
                        # Found the target
                        path_nodes = []
                        for node_id in path + [next_node_id]:
                            if node_id in self.nodes:
                                path_nodes.append(self.nodes[node_id])
                        return path_nodes
                    
                    if next_node_id not in visited:
                        queue.append((next_node_id, path + [next_node_id]))
        
        return []  # No path found
    
    def get_impact_analysis(self, node_id: UUID) -> Dict[str, List[LineageNode]]:
        """Get impact analysis for a node (upstream and downstream)."""
        return {
            "upstream": self.get_upstream_nodes(node_id),
            "downstream": self.get_downstream_nodes(node_id)
        }
    
    def get_nodes_by_type(self, node_type: LineageNodeType) -> List[LineageNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.type == node_type]
    
    def get_critical_path(self) -> List[LineageNode]:
        """Get the critical path through the lineage (nodes with highest impact)."""
        # Calculate impact score for each node based on downstream dependencies
        impact_scores = {}
        
        for node_id, node in self.nodes.items():
            downstream_count = len(self.get_downstream_nodes(node_id))
            upstream_count = len(self.get_upstream_nodes(node_id))
            
            # Impact score considers both downstream dependencies and quality score
            quality_factor = node.quality_score or 1.0
            impact_scores[node_id] = (downstream_count + upstream_count) * quality_factor
        
        # Sort nodes by impact score
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: impact_scores.get(n.id, 0),
            reverse=True
        )
        
        return sorted_nodes
    
    def validate_lineage(self) -> List[str]:
        """Validate the lineage graph for consistency."""
        errors = []
        
        # Check for orphaned edges
        for edge in self.edges.values():
            if edge.source_node_id not in self.nodes:
                errors.append(f"Edge {edge.id} references missing source node {edge.source_node_id}")
            if edge.target_node_id not in self.nodes:
                errors.append(f"Edge {edge.id} references missing target node {edge.target_node_id}")
        
        # Check for circular dependencies
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: UUID) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for edge in self.edges.values():
                if edge.source_node_id == node_id:
                    next_node_id = edge.target_node_id
                    if next_node_id not in visited:
                        if has_cycle(next_node_id):
                            return True
                    elif next_node_id in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    errors.append(f"Circular dependency detected involving node {node_id}")
                    break
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert lineage to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "created_by": self.metadata.created_by,
                "tags": list(self.metadata.tags),
                "properties": self.metadata.properties,
                "version": self.metadata.version
            },
            "nodes": {
                str(node_id): node.dict() 
                for node_id, node in self.nodes.items()
            },
            "edges": {
                str(edge_id): edge.dict()
                for edge_id, edge in self.edges.items()
            },
            "version": self.version,
            "namespace": self.namespace
        }