"""Model registry entity for centralized model management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pynomaly_detection.domain.entities.model_version import ModelStatus, ModelVersion
from pynomaly_detection.domain.value_objects.semantic_version import SemanticVersion


class AccessLevel(Enum):
    """Access levels for model registry."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class Environment(Enum):
    """Deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class AccessPolicy:
    """Access policy for model registry."""

    users: dict[str, AccessLevel] = field(default_factory=dict)
    groups: dict[str, AccessLevel] = field(default_factory=dict)
    public_read: bool = False

    def can_read(self, user: str, groups: list[str] | None = None) -> bool:
        """Check if user can read from registry."""
        if self.public_read:
            return True

        # Check user permissions
        if user in self.users:
            return True

        # Check group permissions
        if groups:
            for group in groups:
                if group in self.groups:
                    return True

        return False

    def can_write(self, user: str, groups: list[str] | None = None) -> bool:
        """Check if user can write to registry."""
        # Check user permissions
        user_level = self.users.get(user)
        if user_level and user_level in {AccessLevel.WRITE, AccessLevel.ADMIN}:
            return True

        # Check group permissions
        if groups:
            for group in groups:
                group_level = self.groups.get(group)
                if group_level and group_level in {
                    AccessLevel.WRITE,
                    AccessLevel.ADMIN,
                }:
                    return True

        return False

    def is_admin(self, user: str, groups: list[str] | None = None) -> bool:
        """Check if user has admin access."""
        # Check user permissions
        if self.users.get(user) == AccessLevel.ADMIN:
            return True

        # Check group permissions
        if groups:
            for group in groups:
                if self.groups.get(group) == AccessLevel.ADMIN:
                    return True

        return False


@dataclass
class Model:
    """Model entity in the registry.

    Represents a logical model that can have multiple versions.

    Attributes:
        id: Unique identifier for the model
        name: Human-readable name
        description: Detailed description
        algorithm: Algorithm used (e.g., "IsolationForest", "LOF")
        domain: Application domain (e.g., "finance", "security", "iot")
        tags: Semantic tags for organization
        owner: User who owns this model
        created_at: When the model was first registered
        updated_at: When the model was last updated
        versions: Dictionary of version string to ModelVersion
        active_versions: Set of currently active version IDs
        production_version_id: ID of version deployed to production
        staging_version_id: ID of version deployed to staging
        metadata: Additional metadata
        is_public: Whether model is publicly accessible
        is_archived: Whether model is archived
    """

    name: str
    description: str
    algorithm: str
    domain: str
    owner: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)
    versions: dict[str, ModelVersion] = field(default_factory=dict)
    active_versions: set[UUID] = field(default_factory=set)
    production_version_id: UUID | None = None
    staging_version_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_public: bool = False
    is_archived: bool = False

    def __post_init__(self) -> None:
        """Validate model after initialization."""
        if not self.name:
            raise ValueError("Model name cannot be empty")

        if not self.algorithm:
            raise ValueError("Algorithm cannot be empty")

        if not self.domain:
            raise ValueError("Domain cannot be empty")

        if not self.owner:
            raise ValueError("Owner cannot be empty")

    def add_version(self, version: ModelVersion) -> None:
        """Add a new version to this model.

        Args:
            version: ModelVersion to add

        Raises:
            ValueError: If version already exists or belongs to different model
        """
        if version.model_id != self.id:
            raise ValueError(f"Version belongs to different model: {version.model_id}")

        version_str = version.version_string
        if version_str in self.versions:
            raise ValueError(f"Version {version_str} already exists")

        self.versions[version_str] = version
        self.active_versions.add(version.id)
        self.updated_at = datetime.utcnow()

    def get_version(self, version: str) -> ModelVersion | None:
        """Get a specific version by version string."""
        return self.versions.get(version)

    def get_latest_version(self) -> ModelVersion | None:
        """Get the latest version (highest version number)."""
        if not self.versions:
            return None

        latest_version_str = max(
            self.versions.keys(), key=lambda v: SemanticVersion.from_string(v)
        )

        return self.versions[latest_version_str]

    def get_production_version(self) -> ModelVersion | None:
        """Get the production version."""
        if not self.production_version_id:
            return None

        for version in self.versions.values():
            if version.id == self.production_version_id:
                return version

        return None

    def get_staging_version(self) -> ModelVersion | None:
        """Get the staging version."""
        if not self.staging_version_id:
            return None

        for version in self.versions.values():
            if version.id == self.staging_version_id:
                return version

        return None

    def promote_to_staging(self, version_id: UUID) -> None:
        """Promote a version to staging.

        Args:
            version_id: ID of version to promote

        Raises:
            ValueError: If version not found
        """
        # Verify version exists
        version_found = any(v.id == version_id for v in self.versions.values())
        if not version_found:
            raise ValueError(f"Version {version_id} not found in model")

        self.staging_version_id = version_id
        self.updated_at = datetime.utcnow()

    def promote_to_production(self, version_id: UUID) -> None:
        """Promote a version to production.

        Args:
            version_id: ID of version to promote

        Raises:
            ValueError: If version not found
        """
        # Verify version exists
        version_found = any(v.id == version_id for v in self.versions.values())
        if not version_found:
            raise ValueError(f"Version {version_id} not found in model")

        self.production_version_id = version_id
        self.updated_at = datetime.utcnow()

    def archive_version(self, version_id: UUID) -> None:
        """Archive a version.

        Args:
            version_id: ID of version to archive
        """
        for version in self.versions.values():
            if version.id == version_id:
                version.update_status(ModelStatus.ARCHIVED)
                self.active_versions.discard(version_id)
                self.updated_at = datetime.utcnow()
                break

    def get_versions_by_status(self, status: ModelStatus) -> list[ModelVersion]:
        """Get all versions with specific status."""
        return [v for v in self.versions.values() if v.status == status]

    def add_tag(self, tag: str) -> None:
        """Add a tag to this model."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from this model."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()

    def has_tag(self, tag: str) -> bool:
        """Check if model has a specific tag."""
        return tag in self.tags

    def archive(self) -> None:
        """Archive this model."""
        self.is_archived = True
        self.updated_at = datetime.utcnow()

    def unarchive(self) -> None:
        """Unarchive this model."""
        self.is_archived = False
        self.updated_at = datetime.utcnow()

    def update_metadata(self, key: str, value: Any) -> None:
        """Update model metadata."""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()

    def get_summary(self) -> dict[str, Any]:
        """Get model summary information."""
        latest_version = self.get_latest_version()
        production_version = self.get_production_version()

        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "algorithm": self.algorithm,
            "domain": self.domain,
            "owner": self.owner,
            "tags": self.tags.copy(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_versions": len(self.versions),
            "active_versions": len(self.active_versions),
            "latest_version": latest_version.version_string if latest_version else None,
            "production_version": (
                production_version.version_string if production_version else None
            ),
            "is_public": self.is_public,
            "is_archived": self.is_archived,
        }

    def get_info(self) -> dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "algorithm": self.algorithm,
            "domain": self.domain,
            "owner": self.owner,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags.copy(),
            "is_public": self.is_public,
            "is_archived": self.is_archived,
            "metadata": self.metadata.copy(),
            "versions": {
                version_str: version.get_info()
                for version_str, version in self.versions.items()
            },
            "active_versions": [str(vid) for vid in self.active_versions],
            "production_version_id": (
                str(self.production_version_id) if self.production_version_id else None
            ),
            "staging_version_id": (
                str(self.staging_version_id) if self.staging_version_id else None
            ),
        }


@dataclass
class ModelRegistry:
    """Central registry for all models.

    Provides a centralized catalog for model discovery, access control,
    and lifecycle management.

    Attributes:
        id: Unique identifier for the registry
        name: Registry name
        description: Registry description
        created_at: When registry was created
        models: Dictionary of model ID to Model
        access_policy: Access control policy
        metadata: Additional registry metadata
        tags: Registry-level tags
        is_active: Whether registry is active
    """

    name: str
    description: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    models: dict[UUID, Model] = field(default_factory=dict)
    access_policy: AccessPolicy = field(default_factory=AccessPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    is_active: bool = True

    def __post_init__(self) -> None:
        """Validate registry after initialization."""
        if not self.name:
            raise ValueError("Registry name cannot be empty")

    def register_model(
        self, model: Model, user: str, groups: list[str] | None = None
    ) -> None:
        """Register a new model in the registry.

        Args:
            model: Model to register
            user: User registering the model
            groups: User's groups

        Raises:
            PermissionError: If user doesn't have write access
            ValueError: If model already exists
        """
        if not self.access_policy.can_write(user, groups):
            raise PermissionError(f"User {user} doesn't have write access")

        if model.id in self.models:
            raise ValueError(f"Model {model.id} already exists in registry")

        self.models[model.id] = model

    def get_model(
        self, model_id: UUID, user: str, groups: list[str] | None = None
    ) -> Model | None:
        """Get a model by ID.

        Args:
            model_id: Model ID
            user: User requesting the model
            groups: User's groups

        Returns:
            Model if found and accessible, None otherwise
        """
        if not self.access_policy.can_read(user, groups):
            return None

        model = self.models.get(model_id)
        if model and (model.is_public or self.access_policy.can_read(user, groups)):
            return model

        return None

    def get_model_by_name(
        self, name: str, user: str, groups: list[str] | None = None
    ) -> Model | None:
        """Get a model by name.

        Args:
            name: Model name
            user: User requesting the model
            groups: User's groups

        Returns:
            Model if found and accessible, None otherwise
        """
        for model in self.models.values():
            if model.name == name:
                return self.get_model(model.id, user, groups)

        return None

    def list_models(
        self,
        user: str,
        groups: list[str] | None = None,
        domain_filter: str | None = None,
        algorithm_filter: str | None = None,
        tag_filter: list[str] | None = None,
        owner_filter: str | None = None,
        include_archived: bool = False,
    ) -> list[Model]:
        """List models with optional filtering.

        Args:
            user: User requesting the list
            groups: User's groups
            domain_filter: Filter by domain
            algorithm_filter: Filter by algorithm
            tag_filter: Filter by tags (must have all tags)
            owner_filter: Filter by owner
            include_archived: Whether to include archived models

        Returns:
            List of accessible models matching filters
        """
        if not self.access_policy.can_read(user, groups):
            return []

        models = []

        for model in self.models.values():
            # Check accessibility
            if not (model.is_public or self.access_policy.can_read(user, groups)):
                continue

            # Apply filters
            if not include_archived and model.is_archived:
                continue

            if domain_filter and model.domain != domain_filter:
                continue

            if algorithm_filter and model.algorithm != algorithm_filter:
                continue

            if owner_filter and model.owner != owner_filter:
                continue

            if tag_filter:
                if not all(tag in model.tags for tag in tag_filter):
                    continue

            models.append(model)

        # Sort by updated_at (most recent first)
        models.sort(key=lambda m: m.updated_at, reverse=True)

        return models

    def search_models(
        self,
        query: str,
        user: str,
        groups: list[str] | None = None,
        limit: int | None = None,
    ) -> list[Model]:
        """Search models by name, description, tags, or algorithm.

        Args:
            query: Search query
            user: User performing the search
            groups: User's groups
            limit: Maximum number of results

        Returns:
            List of matching models
        """
        query_lower = query.lower()
        matching_models = []

        for model in self.models.values():
            # Check accessibility
            if not (model.is_public or self.access_policy.can_read(user, groups)):
                continue

            # Skip archived models
            if model.is_archived:
                continue

            # Search in name, description, algorithm, and tags
            searchable_text = " ".join(
                [
                    model.name.lower(),
                    model.description.lower(),
                    model.algorithm.lower(),
                    " ".join(model.tags).lower(),
                ]
            )

            if query_lower in searchable_text:
                matching_models.append(model)

        # Sort by relevance (simple scoring based on matches in name)
        def relevance_score(model: Model) -> float:
            score = 0.0
            if query_lower in model.name.lower():
                score += 2.0
            if query_lower in model.algorithm.lower():
                score += 1.5
            if any(query_lower in tag.lower() for tag in model.tags):
                score += 1.0
            if query_lower in model.description.lower():
                score += 0.5
            return score

        matching_models.sort(key=relevance_score, reverse=True)

        if limit:
            matching_models = matching_models[:limit]

        return matching_models

    def get_models_by_domain(
        self, domain: str, user: str, groups: list[str] | None = None
    ) -> list[Model]:
        """Get all models for a specific domain."""
        return self.list_models(user, groups, domain_filter=domain)

    def get_models_by_algorithm(
        self, algorithm: str, user: str, groups: list[str] | None = None
    ) -> list[Model]:
        """Get all models using a specific algorithm."""
        return self.list_models(user, groups, algorithm_filter=algorithm)

    def get_models_by_owner(
        self, owner: str, user: str, groups: list[str] | None = None
    ) -> list[Model]:
        """Get all models owned by a specific user."""
        return self.list_models(user, groups, owner_filter=owner)

    def remove_model(
        self, model_id: UUID, user: str, groups: list[str] | None = None
    ) -> bool:
        """Remove a model from the registry.

        Args:
            model_id: Model ID to remove
            user: User removing the model
            groups: User's groups

        Returns:
            True if removed, False if not found

        Raises:
            PermissionError: If user doesn't have admin access
        """
        if not self.access_policy.is_admin(user, groups):
            raise PermissionError(f"User {user} doesn't have admin access")

        if model_id in self.models:
            del self.models[model_id]
            return True

        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics."""
        total_models = len(self.models)
        active_models = sum(1 for m in self.models.values() if not m.is_archived)
        archived_models = total_models - active_models

        # Domain distribution
        domains = {}
        for model in self.models.values():
            domains[model.domain] = domains.get(model.domain, 0) + 1

        # Algorithm distribution
        algorithms = {}
        for model in self.models.values():
            algorithms[model.algorithm] = algorithms.get(model.algorithm, 0) + 1

        # Version statistics
        total_versions = sum(len(m.versions) for m in self.models.values())
        active_versions = sum(len(m.active_versions) for m in self.models.values())

        return {
            "total_models": total_models,
            "active_models": active_models,
            "archived_models": archived_models,
            "total_versions": total_versions,
            "active_versions": active_versions,
            "domains": domains,
            "algorithms": algorithms,
            "created_at": self.created_at.isoformat(),
        }

    def add_user_access(self, user: str, access_level: AccessLevel) -> None:
        """Add user access to registry."""
        self.access_policy.users[user] = access_level

    def add_group_access(self, group: str, access_level: AccessLevel) -> None:
        """Add group access to registry."""
        self.access_policy.groups[group] = access_level

    def remove_user_access(self, user: str) -> None:
        """Remove user access from registry."""
        self.access_policy.users.pop(user, None)

    def remove_group_access(self, group: str) -> None:
        """Remove group access from registry."""
        self.access_policy.groups.pop(group, None)

    def get_info(self) -> dict[str, Any]:
        """Get comprehensive registry information."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
            "tags": self.tags.copy(),
            "metadata": self.metadata.copy(),
            "statistics": self.get_statistics(),
            "access_policy": {
                "users": {
                    user: level.value
                    for user, level in self.access_policy.users.items()
                },
                "groups": {
                    group: level.value
                    for group, level in self.access_policy.groups.items()
                },
                "public_read": self.access_policy.public_read,
            },
        }
