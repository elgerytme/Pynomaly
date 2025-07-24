"""Repository interface for data lineage."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.data_lineage import DataLineage, LineageNode, LineageEdge


class DataLineageRepository(ABC):
    """Abstract repository for data lineage."""
    
    @abstractmethod
    async def save_lineage(self, lineage: DataLineage) -> DataLineage:
        """Save a data lineage graph."""
        pass
    
    @abstractmethod
    async def get_lineage_by_id(self, lineage_id: UUID) -> Optional[DataLineage]:
        """Get lineage graph by ID."""
        pass
    
    @abstractmethod
    async def get_lineage_by_name(self, name: str, namespace: str = "default") -> Optional[DataLineage]:
        """Get lineage graph by name and namespace."""
        pass
    
    @abstractmethod
    async def list_lineages(self, namespace: str = None) -> List[DataLineage]:
        """List all lineages, optionally filtered by namespace."""
        pass
    
    @abstractmethod
    async def add_node(self, lineage_id: UUID, node: LineageNode) -> None:
        """Add a node to a lineage."""
        pass
    
    @abstractmethod
    async def add_edge(self, lineage_id: UUID, edge: LineageEdge) -> None:
        """Add an edge to a lineage."""
        pass
    
    @abstractmethod
    async def get_node_by_id(self, node_id: UUID) -> Optional[LineageNode]:
        """Get a node by ID."""
        pass
    
    @abstractmethod
    async def get_edge_by_id(self, edge_id: UUID) -> Optional[LineageEdge]:
        """Get an edge by ID."""
        pass
    
    @abstractmethod
    async def find_nodes_by_name(self, name: str) -> List[LineageNode]:
        """Find all nodes with a given name across all lineages."""
        pass
    
    @abstractmethod
    async def find_nodes_by_type(self, node_type: str) -> List[LineageNode]:
        """Find all nodes of a given type across all lineages."""
        pass

    @abstractmethod
    async def get_nodes_by_asset_id(self, asset_id: UUID) -> List[LineageNode]:
        """Get all lineage nodes associated with a given asset ID."""
        pass
    
    @abstractmethod
    async def update_node(self, node: LineageNode) -> LineageNode:
        """Update an existing lineage node."""
        pass
    
    @abstractmethod
    async def update_edge(self, edge: LineageEdge) -> LineageEdge:
        """Update an existing lineage edge."""
        pass
    
    @abstractmethod
    async def delete_lineage(self, lineage_id: UUID) -> bool:
        """Delete a lineage graph by ID."""
        pass
    
    @abstractmethod
    async def delete_node(self, node_id: UUID) -> bool:
        """Delete a node by ID."""
        pass
    
    @abstractmethod
    async def delete_edge(self, edge_id: UUID) -> bool:
        """Delete an edge by ID."""
        pass
