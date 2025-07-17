"""Processor registry service for centralized processor management."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from monorepo.domain.entities.model_registry import (
    AccessLevel,
    AccessPolicy,
    Processor,
    ModelRegistry,
)
from monorepo.domain.entities.model_version import ModelStatus, ModelVersion


class ModelRegistryError(Exception):
    """Base exception for processor registry errors."""

    pass


class ModelNotFoundError(ModelRegistryError):
    """Processor not found in registry."""

    pass


class VersionNotFoundError(ModelRegistryError):
    """Processor version not found."""

    pass


class AccessDeniedError(ModelRegistryError):
    """Access denied to registry resource."""

    pass


class ModelAlreadyExistsError(ModelRegistryError):
    """Processor already exists in registry."""

    pass


@dataclass
class SearchCriteria:
    """Search criteria for processor discovery."""

    query: str | None = None
    domain: str | None = None
    algorithm: str | None = None
    tags: list[str] | None = None
    owner: str | None = None
    min_accuracy: float | None = None
    max_inference_time: float | None = None
    include_archived: bool = False
    limit: int | None = None

    def matches_model(self, model: Model) -> bool:
        """Check if a processor matches these criteria."""
        if not self.include_archived and processor.is_archived:
            return False

        if self.domain and processor.domain != self.domain:
            return False

        if self.algorithm and processor.algorithm != self.algorithm:
            return False

        if self.owner and processor.owner != self.owner:
            return False

        if self.tags:
            if not all(tag in processor.tags for tag in self.tags):
                return False

        # Performance-based filtering
        if self.min_accuracy or self.max_inference_time:
            latest_version = processor.get_latest_version()
            if latest_version:
                measurements = latest_version.performance_measurements

                if self.min_accuracy and measurements.accuracy < self.min_accuracy:
                    return False

                if (
                    self.max_inference_time
                    and measurements.inference_time > self.max_inference_time
                ):
                    return False

        return True


@dataclass
class ModelRecommendation:
    """Processor recommendation for a specific use case."""

    processor: Processor
    version: ModelVersion
    confidence: float
    reasoning: str
    expected_performance: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "processor_id": str(self.processor.id),
            "processor_name": self.processor.name,
            "version": self.version.version_string,
            "algorithm": self.processor.algorithm,
            "domain": self.processor.domain,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "expected_performance": self.expected_performance,
            "performance_measurements": self.version.performance_measurements.to_dict(),
        }


class ModelRegistryService:
    """Service for managing processor registries and providing processor discovery.

    This service provides comprehensive processor lifecycle management including:
    - Processor registration and cataloging
    - Version management across environments
    - Access control and permissions
    - Processor search and discovery
    - Performance tracking and comparison
    - Recommendation engine for processor selection
    """

    def __init__(self, storage_path: Path, default_registry_name: str = "default"):
        """Initialize processor registry service.

        Args:
            storage_path: Path for registry persistence
            default_registry_name: Name of default registry
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.registries: dict[str, ModelRegistry] = {}
        self.default_registry_name = default_registry_name

        # Load existing registries
        asyncio.create_task(self._load_registries())

    async def _load_registries(self) -> None:
        """Load registries from storage."""
        registry_files = self.storage_path.glob("*.json")

        for registry_file in registry_files:
            try:
                registry = await self._load_registry_from_file(registry_file)
                self.registries[registry.name] = registry
            except Exception:
                # Skip corrupted registry files
                continue

        # Create default registry if none exist
        if not self.registries:
            await self.create_registry(
                name=self.default_registry_name,
                description="Default processor registry",
                creator="system",
            )

    async def create_registry(
        self,
        name: str,
        description: str,
        creator: str,
        access_policy: AccessPolicy | None = None,
    ) -> ModelRegistry:
        """Create a new processor registry.

        Args:
            name: Registry name
            description: Registry description
            creator: User creating the registry
            access_policy: Access control policy

        Returns:
            Created ModelRegistry

        Raises:
            ModelRegistryError: If registry already exists
        """
        if name in self.registries:
            raise ModelRegistryError(f"Registry '{name}' already exists")

        # Create default access policy if none provided
        if access_policy is None:
            access_policy = AccessPolicy()
            access_policy.users[creator] = AccessLevel.ADMIN
            access_policy.public_read = True

        registry = ModelRegistry(
            name=name, description=description, access_policy=access_policy
        )

        self.registries[name] = registry
        await self._save_registry(registry)

        return registry

    async def get_registry(self, name: str | None = None) -> ModelRegistry:
        """Get a registry by name.

        Args:
            name: Registry name (uses default if None)

        Returns:
            ModelRegistry

        Raises:
            ModelRegistryError: If registry not found
        """
        registry_name = name or self.default_registry_name

        if registry_name not in self.registries:
            raise ModelRegistryError(f"Registry '{registry_name}' not found")

        return self.registries[registry_name]

    async def register_processor(
        self,
        name: str,
        description: str,
        algorithm: str,
        domain: str,
        owner: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> Processor:
        """Register a new processor in the registry.

        Args:
            name: Processor name
            description: Processor description
            algorithm: Algorithm used
            domain: Application domain
            owner: Processor owner
            tags: Optional tags
            metadata: Optional metadata
            registry_name: Registry to use (default if None)
            user: User registering the processor
            groups: User's groups

        Returns:
            Registered Processor

        Raises:
            ModelAlreadyExistsError: If processor name already exists
            AccessDeniedError: If user lacks permission
        """
        registry = await self.get_registry(registry_name)

        # Check if processor name already exists
        existing_processor = registry.get_processor_by_name(name, user, groups)
        if existing_processor:
            raise ModelAlreadyExistsError(f"Processor '{name}' already exists")

        processor = Processor(
            name=name,
            description=description,
            algorithm=algorithm,
            domain=domain,
            owner=owner,
            tags=tags or [],
            metadata=metadata or {},
        )

        registry.register_processor(processor, user, groups)
        await self._save_registry(registry)

        return processor

    async def add_processor_version(
        self,
        processor_id: UUID,
        version: ModelVersion,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> None:
        """Add a version to an existing processor.

        Args:
            processor_id: Processor ID
            version: ModelVersion to add
            registry_name: Registry name
            user: User adding the version
            groups: User's groups

        Raises:
            ModelNotFoundError: If processor not found
            AccessDeniedError: If user lacks permission
        """
        registry = await self.get_registry(registry_name)
        processor = registry.get_processor(processor_id, user, groups)

        if not processor:
            raise ModelNotFoundError(f"Processor {processor_id} not found")

        if not registry.access_policy.can_write(user, groups):
            raise AccessDeniedError(f"User {user} lacks write permission")

        processor.add_version(version)
        await self._save_registry(registry)

    async def get_processor(
        self,
        processor_id: UUID,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> Processor:
        """Get a processor by ID.

        Args:
            processor_id: Processor ID
            registry_name: Registry name
            user: User requesting the processor
            groups: User's groups

        Returns:
            Processor

        Raises:
            ModelNotFoundError: If processor not found
        """
        registry = await self.get_registry(registry_name)
        processor = registry.get_processor(processor_id, user, groups)

        if not processor:
            raise ModelNotFoundError(f"Processor {processor_id} not found")

        return processor

    async def get_processor_by_name(
        self,
        name: str,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> Processor:
        """Get a processor by name.

        Args:
            name: Processor name
            registry_name: Registry name
            user: User requesting the processor
            groups: User's groups

        Returns:
            Processor

        Raises:
            ModelNotFoundError: If processor not found
        """
        registry = await self.get_registry(registry_name)
        processor = registry.get_processor_by_name(name, user, groups)

        if not processor:
            raise ModelNotFoundError(f"Processor '{name}' not found")

        return processor

    async def list_processors(
        self,
        criteria: SearchCriteria | None = None,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> list[Processor]:
        """List models with optional filtering.

        Args:
            criteria: Search criteria
            registry_name: Registry name
            user: User requesting the list
            groups: User's groups

        Returns:
            List of models matching criteria
        """
        registry = await self.get_registry(registry_name)
        criteria = criteria or SearchCriteria()

        models = registry.list_processors(
            user=user,
            groups=groups,
            domain_filter=criteria.domain,
            algorithm_filter=criteria.algorithm,
            tag_filter=criteria.tags,
            owner_filter=criteria.owner,
            include_archived=criteria.include_archived,
        )

        # Apply additional criteria filtering
        filtered_processors = [processor for processor in models if criteria.matches_processor(processor)]

        # Apply limit
        if criteria.limit:
            filtered_processors = filtered_processors[: criteria.limit]

        return filtered_processors

    async def search_processors(
        self,
        query: str,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
        limit: int | None = None,
    ) -> list[Processor]:
        """Search models by query.

        Args:
            query: Search query
            registry_name: Registry name
            user: User performing search
            groups: User's groups
            limit: Maximum results

        Returns:
            List of matching models
        """
        registry = await self.get_registry(registry_name)
        return registry.search_processors(query, user, groups, limit)

    async def get_processor_version(
        self,
        processor_id: UUID,
        version: str,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> ModelVersion:
        """Get a specific processor version.

        Args:
            processor_id: Processor ID
            version: Version string
            registry_name: Registry name
            user: User requesting the version
            groups: User's groups

        Returns:
            ModelVersion

        Raises:
            ModelNotFoundError: If processor not found
            VersionNotFoundError: If version not found
        """
        processor = await self.get_processor(processor_id, registry_name, user, groups)
        processor_version = processor.get_version(version)

        if not processor_version:
            raise VersionNotFoundError(
                f"Version {version} not found for processor {processor_id}"
            )

        return processor_version

    async def promote_to_staging(
        self,
        processor_id: UUID,
        version_id: UUID,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> None:
        """Promote a processor version to staging.

        Args:
            processor_id: Processor ID
            version_id: Version ID to promote
            registry_name: Registry name
            user: User performing promotion
            groups: User's groups
        """
        registry = await self.get_registry(registry_name)
        processor = registry.get_processor(processor_id, user, groups)

        if not processor:
            raise ModelNotFoundError(f"Processor {processor_id} not found")

        if not registry.access_policy.can_write(user, groups):
            raise AccessDeniedError(f"User {user} lacks write permission")

        processor.promote_to_staging(version_id)
        await self._save_registry(registry)

    async def promote_to_production(
        self,
        processor_id: UUID,
        version_id: UUID,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> None:
        """Promote a processor version to production.

        Args:
            processor_id: Processor ID
            version_id: Version ID to promote
            registry_name: Registry name
            user: User performing promotion
            groups: User's groups
        """
        registry = await self.get_registry(registry_name)
        processor = registry.get_processor(processor_id, user, groups)

        if not processor:
            raise ModelNotFoundError(f"Processor {processor_id} not found")

        if not registry.access_policy.is_admin(user, groups):
            raise AccessDeniedError(
                f"User {user} lacks admin permission for production promotion"
            )

        processor.promote_to_production(version_id)
        await self._save_registry(registry)

    async def compare_processor_versions(
        self,
        processor_id: UUID,
        version1: str,
        version2: str,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare two versions of a processor.

        Args:
            processor_id: Processor ID
            version1: First version string
            version2: Second version string
            registry_name: Registry name
            user: User performing comparison
            groups: User's groups

        Returns:
            Comparison results
        """
        v1 = await self.get_processor_version(
            processor_id, version1, registry_name, user, groups
        )
        v2 = await self.get_processor_version(
            processor_id, version2, registry_name, user, groups
        )

        performance_diff = v1.performance_measurements.compare_with(v2.performance_measurements)

        return {
            "processor_id": str(processor_id),
            "version1": {
                "version": v1.version_string,
                "status": v1.status.value,
                "created_at": v1.created_at.isoformat(),
                "performance": v1.performance_measurements.to_dict(),
            },
            "version2": {
                "version": v2.version_string,
                "status": v2.status.value,
                "created_at": v2.created_at.isoformat(),
                "performance": v2.performance_measurements.to_dict(),
            },
            "performance_difference": performance_diff,
            "version1_is_better": v1.performance_measurements.is_better_than(
                v2.performance_measurements
            ),
            "recommendations": self._generate_version_recommendations(v1, v2),
        }

    async def recommend_processors(
        self,
        domain: str,
        use_case: str,
        performance_requirements: dict[str, float] | None = None,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
        limit: int = 5,
    ) -> list[ModelRecommendation]:
        """Recommend models for a specific use case.

        Args:
            domain: Application domain
            use_case: Specific use case description
            performance_requirements: Required performance measurements
            registry_name: Registry name
            user: User requesting recommendations
            groups: User's groups
            limit: Maximum recommendations

        Returns:
            List of processor recommendations
        """
        # Get models in the domain
        criteria = SearchCriteria(domain=domain, include_archived=False)
        models = await self.list_processors(criteria, registry_name, user, groups)

        recommendations = []

        for processor in models:
            latest_version = processor.get_latest_version()
            if not latest_version:
                continue

            # Calculate recommendation score
            score, reasoning = self._calculate_recommendation_score(
                processor, latest_version, use_case, performance_requirements
            )

            if score > 0.3:  # Minimum threshold
                recommendation = ModelRecommendation(
                    processor=processor,
                    version=latest_version,
                    confidence=score,
                    reasoning=reasoning,
                    expected_performance=latest_version.performance_measurements.to_dict(),
                )
                recommendations.append(recommendation)

        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return recommendations[:limit]

    async def get_registry_statistics(
        self, registry_name: str | None = None
    ) -> dict[str, Any]:
        """Get registry statistics.

        Args:
            registry_name: Registry name

        Returns:
            Registry statistics
        """
        registry = await self.get_registry(registry_name)
        return registry.get_statistics()

    async def archive_processor(
        self,
        processor_id: UUID,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> None:
        """Archive a processor.

        Args:
            processor_id: Processor ID
            registry_name: Registry name
            user: User archiving the processor
            groups: User's groups
        """
        registry = await self.get_registry(registry_name)
        processor = registry.get_processor(processor_id, user, groups)

        if not processor:
            raise ModelNotFoundError(f"Processor {processor_id} not found")

        if not registry.access_policy.can_write(user, groups):
            raise AccessDeniedError(f"User {user} lacks write permission")

        processor.archive()
        await self._save_registry(registry)

    async def _save_registry(self, registry: ModelRegistry) -> None:
        """Save registry to storage."""
        registry_file = self.storage_path / f"{registry.name}.json"

        with open(registry_file, "w") as f:
            json.dump(registry.get_info(), f, indent=2, default=str)

    async def _load_registry_from_file(self, registry_file: Path) -> ModelRegistry:
        """Load registry from file."""
        with open(registry_file) as f:
            data = json.load(f)

        # Reconstruct registry (simplified for example)
        # In practice, you'd need full deserialization logic
        registry = ModelRegistry(
            name=data["name"],
            description=data["description"],
            id=UUID(data["id"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

        return registry

    def _calculate_recommendation_score(
        self,
        processor: Processor,
        version: ModelVersion,
        use_case: str,
        performance_requirements: dict[str, float] | None,
    ) -> tuple[float, str]:
        """Calculate recommendation score for a processor."""
        score = 0.0
        reasons = []

        # Base score from performance
        measurements = version.performance_measurements
        score += measurements.performance_score * 0.4

        # Algorithm suitability (simplified)
        algorithm_scores = {
            "IsolationForest": 0.9,
            "LocalOutlierFactor": 0.8,
            "OneClassSVM": 0.7,
            "EllipticEnvelope": 0.6,
        }

        algo_score = algorithm_scores.get(processor.algorithm, 0.5)
        score += algo_score * 0.3
        reasons.append(f"Algorithm {processor.algorithm} suitable for anomaly processing")

        # Performance requirements check
        if performance_requirements:
            meets_requirements = True
            for metric, requirement in performance_requirements.items():
                if hasattr(measurements, metric):
                    actual_value = getattr(measurements, metric)
                    if metric in ["accuracy", "precision", "recall", "f1_score"]:
                        # Higher is better
                        if actual_value < requirement:
                            meets_requirements = False
                            break
                    elif metric in ["inference_time", "training_time"]:
                        # Lower is better
                        if actual_value > requirement:
                            meets_requirements = False
                            break

            if meets_requirements:
                score += 0.2
                reasons.append("Meets all performance requirements")
            else:
                score -= 0.1
                reasons.append("Does not meet some performance requirements")

        # Recent activity bonus
        days_since_update = (datetime.utcnow() - processor.updated_at).days
        if days_since_update < 30:
            score += 0.1
            reasons.append("Recently updated")

        reasoning = "; ".join(reasons)
        return min(score, 1.0), reasoning

    def _generate_version_recommendations(
        self, version1: ModelVersion, version2: ModelVersion
    ) -> list[str]:
        """Generate recommendations based on version comparison."""
        recommendations = []

        perf_diff = version1.performance_measurements.compare_with(
            version2.performance_measurements
        )

        if perf_diff["accuracy"] > 0.05:
            recommendations.append(
                f"Version {version1.version_string} has significantly better accuracy"
            )
        elif perf_diff["accuracy"] < -0.05:
            recommendations.append(
                f"Version {version2.version_string} has significantly better accuracy"
            )

        if perf_diff["inference_time"] < -10:  # ms improvement
            recommendations.append(
                f"Version {version1.version_string} is faster for inference"
            )
        elif perf_diff["inference_time"] > 10:
            recommendations.append(
                f"Version {version2.version_string} is faster for inference"
            )

        if version1.status == ModelStatus.DEPLOYED:
            recommendations.append(
                f"Version {version1.version_string} is currently deployed"
            )
        elif version2.status == ModelStatus.DEPLOYED:
            recommendations.append(
                f"Version {version2.version_string} is currently deployed"
            )

        if not recommendations:
            recommendations.append(
                "Both versions have similar performance characteristics"
            )

        return recommendations
