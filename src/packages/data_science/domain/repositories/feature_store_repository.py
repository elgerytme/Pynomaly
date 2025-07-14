"""Feature Store repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from ..entities.feature_store import FeatureStore, FeatureStoreId, UserId


class IFeatureStoreRepository(ABC):
    """Repository interface for feature store persistence."""
    
    @abstractmethod
    async def save(self, feature_store: FeatureStore) -> None:
        """Save a feature store."""
        pass
    
    @abstractmethod
    async def get_by_id(self, feature_store_id: FeatureStoreId) -> Optional[FeatureStore]:
        """Get feature store by ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[FeatureStore]:
        """Get feature store by name."""
        pass
    
    @abstractmethod
    async def get_by_owner_id(self, owner_id: UserId) -> List[FeatureStore]:
        """Get all feature stores owned by a user."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str) -> List[FeatureStore]:
        """Get feature stores by status."""
        pass
    
    @abstractmethod
    async def get_active_stores(self) -> List[FeatureStore]:
        """Get all active feature stores."""
        pass
    
    @abstractmethod
    async def search_stores(self, query: str, tags: Optional[List[str]] = None) -> List[FeatureStore]:
        """Search feature stores by query and tags."""
        pass
    
    @abstractmethod
    async def get_stores_by_feature_count(self, min_features: int, max_features: Optional[int] = None) -> List[FeatureStore]:
        """Get stores by feature count range."""
        pass
    
    @abstractmethod
    async def get_stores_with_feature(self, feature_name: str) -> List[FeatureStore]:
        """Get stores containing a specific feature."""
        pass
    
    @abstractmethod
    async def get_feature_lineage(self, feature_store_id: FeatureStoreId, feature_name: str) -> Dict[str, Any]:
        """Get lineage information for a specific feature."""
        pass
    
    @abstractmethod
    async def get_feature_statistics(self, feature_store_id: FeatureStoreId, feature_name: str) -> Dict[str, Any]:
        """Get statistics for a specific feature."""
        pass
    
    @abstractmethod
    async def get_feature_usage_metrics(self, feature_store_id: FeatureStoreId) -> Dict[str, Any]:
        """Get usage metrics for all features in store."""
        pass
    
    @abstractmethod
    async def add_feature_group(self, feature_store_id: FeatureStoreId, feature_group: Dict[str, Any]) -> None:
        """Add a feature group to the store."""
        pass
    
    @abstractmethod
    async def remove_feature_group(self, feature_store_id: FeatureStoreId, group_name: str) -> None:
        """Remove a feature group from the store."""
        pass
    
    @abstractmethod
    async def update_feature_metadata(self, feature_store_id: FeatureStoreId, 
                                    feature_name: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific feature."""
        pass
    
    @abstractmethod
    async def validate_feature_quality(self, feature_store_id: FeatureStoreId, 
                                     feature_name: str) -> Dict[str, Any]:
        """Validate quality of a specific feature."""
        pass
    
    @abstractmethod
    async def get_feature_schema(self, feature_store_id: FeatureStoreId) -> Dict[str, Any]:
        """Get schema information for the feature store."""
        pass
    
    @abstractmethod
    async def update_access_permissions(self, feature_store_id: FeatureStoreId, 
                                      permissions: Dict[str, Any]) -> None:
        """Update access permissions for the feature store."""
        pass
    
    @abstractmethod
    async def get_access_logs(self, feature_store_id: FeatureStoreId, 
                            since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get access logs for the feature store."""
        pass
    
    @abstractmethod
    async def create_feature_snapshot(self, feature_store_id: FeatureStoreId, 
                                    snapshot_name: str) -> str:
        """Create a snapshot of the feature store."""
        pass
    
    @abstractmethod
    async def restore_from_snapshot(self, feature_store_id: FeatureStoreId, 
                                  snapshot_id: str) -> None:
        """Restore feature store from snapshot."""
        pass
    
    @abstractmethod
    async def get_snapshots(self, feature_store_id: FeatureStoreId) -> List[Dict[str, Any]]:
        """Get all snapshots for a feature store."""
        pass
    
    @abstractmethod
    async def sync_with_source(self, feature_store_id: FeatureStoreId) -> Dict[str, Any]:
        """Sync feature store with its data source."""
        pass
    
    @abstractmethod
    async def get_data_drift_metrics(self, feature_store_id: FeatureStoreId, 
                                   baseline_date: datetime) -> Dict[str, Any]:
        """Get data drift metrics compared to baseline."""
        pass
    
    @abstractmethod
    async def archive_store(self, feature_store_id: FeatureStoreId) -> None:
        """Archive a feature store."""
        pass
    
    @abstractmethod
    async def restore_store(self, feature_store_id: FeatureStoreId) -> None:
        """Restore an archived feature store."""
        pass
    
    @abstractmethod
    async def delete(self, feature_store_id: FeatureStoreId) -> None:
        """Delete a feature store."""
        pass
    
    @abstractmethod
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[FeatureStore]:
        """List all feature stores with pagination."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Count total number of feature stores."""
        pass
    
    @abstractmethod
    async def get_global_feature_catalog(self) -> Dict[str, Any]:
        """Get global catalog of all features across stores."""
        pass