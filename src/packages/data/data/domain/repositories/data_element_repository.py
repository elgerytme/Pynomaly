"""Data element repository interface."""

from abc import abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID
from packages.core.domain.abstractions.repository_interface import RepositoryInterface
from ..entities.data_element import DataElement, ElementType, ElementStatus
from ..value_objects.data_type import PrimitiveDataType


class DataElementRepository(RepositoryInterface[DataElement]):
    """Repository interface for data elements."""
    
    @abstractmethod
    async def find_by_dataset_id(self, dataset_id: UUID) -> List[DataElement]:
        """Find elements by parent dataset ID."""
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str, dataset_id: Optional[UUID] = None) -> List[DataElement]:
        """Find elements by name, optionally within a specific dataset."""
        pass
    
    @abstractmethod
    async def find_by_type(self, element_type: ElementType) -> List[DataElement]:
        """Find elements by type."""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: ElementStatus) -> List[DataElement]:
        """Find elements by status."""
        pass
    
    @abstractmethod
    async def find_by_data_type(self, primitive_type: PrimitiveDataType) -> List[DataElement]:
        """Find elements by primitive data type."""
        pass
    
    @abstractmethod
    async def find_primary_keys(self, dataset_id: Optional[UUID] = None) -> List[DataElement]:
        """Find primary key elements, optionally within a specific dataset."""
        pass
    
    @abstractmethod
    async def find_foreign_keys(self, dataset_id: Optional[UUID] = None) -> List[DataElement]:
        """Find foreign key elements, optionally within a specific dataset."""
        pass
    
    @abstractmethod
    async def find_calculated_elements(self, dataset_id: Optional[UUID] = None) -> List[DataElement]:
        """Find calculated/derived elements, optionally within a specific dataset."""
        pass
    
    @abstractmethod
    async def find_active_elements(self, dataset_id: Optional[UUID] = None) -> List[DataElement]:
        """Find active elements, optionally within a specific dataset."""
        pass
    
    @abstractmethod
    async def find_deprecated_elements(self) -> List[DataElement]:
        """Find deprecated elements."""
        pass
    
    @abstractmethod
    async def find_by_owner(self, owner: str) -> List[DataElement]:
        """Find elements by owner."""
        pass
    
    @abstractmethod
    async def find_by_steward(self, steward: str) -> List[DataElement]:
        """Find elements by data steward."""
        pass
    
    @abstractmethod
    async def find_required_elements(self, dataset_id: UUID) -> List[DataElement]:
        """Find required elements in a dataset."""
        pass
    
    @abstractmethod
    async def find_optional_elements(self, dataset_id: UUID) -> List[DataElement]:
        """Find optional elements in a dataset."""
        pass
    
    @abstractmethod
    async def find_unique_elements(self, dataset_id: Optional[UUID] = None) -> List[DataElement]:
        """Find elements with uniqueness constraints."""
        pass
    
    @abstractmethod
    async def find_indexed_elements(self, dataset_id: Optional[UUID] = None) -> List[DataElement]:
        """Find indexed elements."""
        pass
    
    @abstractmethod
    async def find_with_allowed_values(self, dataset_id: Optional[UUID] = None) -> List[DataElement]:
        """Find elements with enumerated allowed values."""
        pass
    
    @abstractmethod
    async def find_high_cardinality_elements(self, threshold_ratio: float = 0.8) -> List[DataElement]:
        """Find elements with high cardinality."""
        pass
    
    @abstractmethod
    async def find_elements_requiring_masking(self) -> List[DataElement]:
        """Find elements requiring data masking."""
        pass
    
    @abstractmethod
    async def find_by_classification_level(self, sensitivity_level: str) -> List[DataElement]:
        """Find elements by data classification level."""
        pass
    
    @abstractmethod
    async def find_by_position_range(self, dataset_id: UUID, start_pos: int, end_pos: int) -> List[DataElement]:
        """Find elements within position range in a dataset."""
        pass
    
    @abstractmethod
    async def find_upstream_dependencies(self, element_id: UUID) -> List[DataElement]:
        """Find upstream element dependencies."""
        pass
    
    @abstractmethod
    async def find_downstream_consumers(self, element_id: UUID) -> List[DataElement]:
        """Find downstream element consumers."""
        pass
    
    @abstractmethod
    async def find_by_business_rule(self, rule_pattern: str) -> List[DataElement]:
        """Find elements with specific business rule patterns."""
        pass
    
    @abstractmethod
    async def find_by_validation_rule(self, rule_pattern: str) -> List[DataElement]:
        """Find elements with specific validation rule patterns."""
        pass
    
    @abstractmethod
    async def find_by_tags(self, tags: List[str]) -> List[DataElement]:
        """Find elements with specific tags."""
        pass
    
    @abstractmethod
    async def get_element_statistics(self, element_id: UUID) -> Dict[str, Any]:
        """Get comprehensive statistics for an element."""
        pass
    
    @abstractmethod
    async def get_validation_history(self, element_id: UUID, limit: int = 10) -> List[Dict[str, Any]]:
        """Get validation history for an element."""
        pass
    
    @abstractmethod
    async def search_elements(self, query: str, dataset_id: Optional[UUID] = None) -> List[DataElement]:
        """Search elements by name, description, or business definition."""
        pass