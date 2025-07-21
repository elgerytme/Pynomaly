"""Data asset repository interface."""

from abc import abstractmethod
from typing import List, Optional
from uuid import UUID
from packages.core.domain.abstractions.repository_interface import RepositoryInterface
from ..entities.data_asset import DataAsset, AssetType, AssetStatus


class DataAssetRepository(RepositoryInterface[DataAsset]):
    """Repository interface for data assets."""
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[DataAsset]:
        """Find asset by name."""
        pass
    
    @abstractmethod
    async def find_by_type(self, asset_type: AssetType) -> List[DataAsset]:
        """Find assets by type."""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: AssetStatus) -> List[DataAsset]:
        """Find assets by status."""
        pass
    
    @abstractmethod
    async def find_by_owner(self, owner: str) -> List[DataAsset]:
        """Find assets by owner."""
        pass
    
    @abstractmethod
    async def find_by_steward(self, steward: str) -> List[DataAsset]:
        """Find assets by data steward."""
        pass
    
    @abstractmethod
    async def find_by_domain(self, domain: str) -> List[DataAsset]:
        """Find assets by business domain."""
        pass
    
    @abstractmethod
    async def find_by_subject_area(self, subject_area: str) -> List[DataAsset]:
        """Find assets by subject area."""
        pass
    
    @abstractmethod
    async def find_active_assets(self) -> List[DataAsset]:
        """Find all active assets."""
        pass
    
    @abstractmethod
    async def find_deprecated_assets(self) -> List[DataAsset]:
        """Find all deprecated assets."""
        pass
    
    @abstractmethod
    async def find_by_quality_threshold(self, min_score: float) -> List[DataAsset]:
        """Find assets meeting minimum quality score."""
        pass
    
    @abstractmethod
    async def find_high_impact_assets(self, min_impact_score: float) -> List[DataAsset]:
        """Find assets with high business impact."""
        pass
    
    @abstractmethod
    async def find_frequently_used(self, min_usage: int) -> List[DataAsset]:
        """Find frequently accessed assets."""
        pass
    
    @abstractmethod
    async def find_by_source_system(self, source_system: str) -> List[DataAsset]:
        """Find assets by source system."""
        pass
    
    @abstractmethod
    async def find_upstream_dependencies(self, asset_id: UUID) -> List[DataAsset]:
        """Find upstream asset dependencies."""
        pass
    
    @abstractmethod
    async def find_downstream_consumers(self, asset_id: UUID) -> List[DataAsset]:
        """Find downstream asset consumers."""
        pass
    
    @abstractmethod
    async def find_related_assets(self, asset_id: UUID) -> List[DataAsset]:
        """Find related assets."""
        pass
    
    @abstractmethod
    async def find_by_classification_level(self, sensitivity_level: str) -> List[DataAsset]:
        """Find assets by data classification level."""
        pass
    
    @abstractmethod
    async def find_with_compliance_requirements(self, compliance_tags: List[str]) -> List[DataAsset]:
        """Find assets with specific compliance requirements."""
        pass
    
    @abstractmethod
    async def find_by_tags(self, tags: List[str]) -> List[DataAsset]:
        """Find assets with specific tags."""
        pass
    
    @abstractmethod
    async def search_assets(self, query: str) -> List[DataAsset]:
        """Search assets by name, description, or tags."""
        pass