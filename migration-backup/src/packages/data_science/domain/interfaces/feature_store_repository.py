"""Repository interface for FeatureStore entities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import UUID

from packages.core.domain.abstractions.repository_interface import RepositoryInterface
from packages.data_science.domain.entities.feature_store import (
    FeatureStore,
    FeatureType,
    FeatureStatus,
)


class FeatureStoreRepository(RepositoryInterface[FeatureStore], ABC):
    """Repository interface for feature store persistence operations."""

    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[FeatureStore]:
        """Find feature store by name.
        
        Args:
            name: Feature store name to search for
            
        Returns:
            FeatureStore if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_namespace(self, namespace: str) -> list[FeatureStore]:
        """Find feature stores by namespace.
        
        Args:
            namespace: Namespace to search for
            
        Returns:
            List of feature stores in the specified namespace
        """
        pass

    @abstractmethod
    async def find_by_owner(self, owner: str) -> list[FeatureStore]:
        """Find feature stores by owner.
        
        Args:
            owner: Owner to search for
            
        Returns:
            List of feature stores owned by the specified user
        """
        pass

    @abstractmethod
    async def find_by_collaborator(self, collaborator: str) -> list[FeatureStore]:
        """Find feature stores where user is a collaborator.
        
        Args:
            collaborator: Collaborator to search for
            
        Returns:
            List of feature stores where user is a collaborator
        """
        pass

    @abstractmethod
    async def find_by_tags(self, tags: list[str]) -> list[FeatureStore]:
        """Find feature stores by tags.
        
        Args:
            tags: List of tags to search for
            
        Returns:
            List of feature stores containing any of the specified tags
        """
        pass

    @abstractmethod
    async def find_stores_with_feature(self, feature_name: str) -> list[FeatureStore]:
        """Find feature stores containing a specific feature.
        
        Args:
            feature_name: Feature name to search for
            
        Returns:
            List of feature stores containing the feature
        """
        pass

    @abstractmethod
    async def find_features_by_type(
        self, feature_type: FeatureType
    ) -> list[tuple[FeatureStore, str]]:
        """Find features by type across all stores.
        
        Args:
            feature_type: Type of features to search for
            
        Returns:
            List of tuples (feature_store, feature_name) for matching features
        """
        pass

    @abstractmethod
    async def find_features_by_status(
        self, status: FeatureStatus
    ) -> list[tuple[FeatureStore, str]]:
        """Find features by status across all stores.
        
        Args:
            status: Status of features to search for
            
        Returns:
            List of tuples (feature_store, feature_name) for matching features
        """
        pass

    @abstractmethod
    async def search_features_by_description(
        self, search_terms: list[str]
    ) -> list[tuple[FeatureStore, str, dict[str, Any]]]:
        """Search features by description text.
        
        Args:
            search_terms: Terms to search for in feature descriptions
            
        Returns:
            List of tuples (feature_store, feature_name, feature_definition)
        """
        pass

    @abstractmethod
    async def find_feature_dependencies(
        self, feature_store_id: UUID, feature_name: str
    ) -> list[dict[str, Any]]:
        """Find dependencies for a specific feature.
        
        Args:
            feature_store_id: Feature store ID
            feature_name: Feature name
            
        Returns:
            List of feature dependencies
        """
        pass

    @abstractmethod
    async def find_dependent_features(
        self, feature_store_id: UUID, feature_name: str
    ) -> list[tuple[FeatureStore, str]]:
        """Find features that depend on the given feature.
        
        Args:
            feature_store_id: Feature store ID
            feature_name: Feature name
            
        Returns:
            List of tuples (feature_store, feature_name) for dependent features
        """
        pass

    @abstractmethod
    async def get_feature_usage_stats(
        self, feature_store_id: UUID, feature_name: str
    ) -> dict[str, Any]:
        """Get usage statistics for a specific feature.
        
        Args:
            feature_store_id: Feature store ID
            feature_name: Feature name
            
        Returns:
            Dictionary of usage statistics
        """
        pass

    @abstractmethod
    async def get_feature_quality_metrics(
        self, feature_store_id: UUID, feature_name: str
    ) -> dict[str, Any]:
        """Get quality metrics for a specific feature.
        
        Args:
            feature_store_id: Feature store ID
            feature_name: Feature name
            
        Returns:
            Dictionary of quality metrics
        """
        pass

    @abstractmethod
    async def find_similar_features(
        self, feature_store_id: UUID, feature_name: str, similarity_threshold: float = 0.8
    ) -> list[tuple[FeatureStore, str, float]]:
        """Find features similar to the given feature.
        
        Args:
            feature_store_id: Feature store ID
            feature_name: Reference feature name
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of tuples (feature_store, feature_name, similarity_score)
        """
        pass

    @abstractmethod
    async def get_feature_lineage(
        self, feature_store_id: UUID, feature_name: str
    ) -> dict[str, Any]:
        """Get complete lineage for a feature.
        
        Args:
            feature_store_id: Feature store ID
            feature_name: Feature name
            
        Returns:
            Dictionary containing feature lineage information
        """
        pass

    @abstractmethod
    async def archive_unused_features(
        self, days_unused: int = 90
    ) -> int:
        """Archive features that haven't been used for a specified period.
        
        Args:
            days_unused: Number of days without usage to consider for archiving
            
        Returns:
            Number of features archived
        """
        pass

    @abstractmethod
    async def validate_feature_groups(
        self, feature_store_id: UUID
    ) -> dict[str, list[str]]:
        """Validate all feature groups in a store.
        
        Args:
            feature_store_id: Feature store ID
            
        Returns:
            Dictionary of validation errors by group name
        """
        pass

    @abstractmethod
    async def get_feature_store_health(
        self, feature_store_id: UUID
    ) -> dict[str, Any]:
        """Get health metrics for a feature store.
        
        Args:
            feature_store_id: Feature store ID
            
        Returns:
            Dictionary of health metrics
        """
        pass

    @abstractmethod
    async def find_stores_needing_approval(self) -> list[FeatureStore]:
        """Find feature stores with pending changes requiring approval.
        
        Returns:
            List of feature stores needing approval
        """
        pass

    @abstractmethod
    async def bulk_update_feature_status(
        self, feature_store_id: UUID, feature_names: list[str], new_status: FeatureStatus
    ) -> int:
        """Bulk update status for multiple features.
        
        Args:
            feature_store_id: Feature store ID
            feature_names: List of feature names to update
            new_status: New status to set
            
        Returns:
            Number of features updated
        """
        pass

    @abstractmethod
    async def get_compliance_report(
        self, feature_store_id: UUID
    ) -> dict[str, Any]:
        """Get compliance report for a feature store.
        
        Args:
            feature_store_id: Feature store ID
            
        Returns:
            Dictionary containing compliance information
        """
        pass

    @abstractmethod
    async def backup_feature_store(
        self, feature_store_id: UUID, backup_location: str
    ) -> dict[str, Any]:
        """Create a backup of a feature store.
        
        Args:
            feature_store_id: Feature store ID
            backup_location: Location to store the backup
            
        Returns:
            Dictionary containing backup information
        """
        pass

    @abstractmethod
    async def restore_feature_store(
        self, backup_location: str, target_store_id: Optional[UUID] = None
    ) -> FeatureStore:
        """Restore a feature store from backup.
        
        Args:
            backup_location: Location of the backup
            target_store_id: Optional target store ID
            
        Returns:
            Restored feature store
        """
        pass