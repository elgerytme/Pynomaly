"""Model registry service for centralized model management."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pynomaly_detection.domain.entities.model_registry import (
    AccessLevel,
    AccessPolicy,
    Model,
    ModelRegistry,
)
from pynomaly_detection.domain.entities.model_version import ModelStatus, ModelVersion


class ModelRegistryError(Exception):
    """Base exception for model registry errors."""

    pass


class ModelNotFoundError(ModelRegistryError):
    """Model not found in registry."""

    pass


class VersionNotFoundError(ModelRegistryError):
    """Model version not found."""

    pass


class AccessDeniedError(ModelRegistryError):
    """Access denied to registry resource."""

    pass


class ModelAlreadyExistsError(ModelRegistryError):
    """Model already exists in registry."""

    pass


@dataclass
class SearchCriteria:
    """Search criteria for model discovery."""

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
        """Check if a model matches these criteria."""
        if not self.include_archived and model.is_archived:
            return False

        if self.domain and model.domain != self.domain:
            return False

        if self.algorithm and model.algorithm != self.algorithm:
            return False

        if self.owner and model.owner != self.owner:
            return False

        if self.tags:
            if not all(tag in model.tags for tag in self.tags):
                return False

        # Performance-based filtering
        if self.min_accuracy or self.max_inference_time:
            latest_version = model.get_latest_version()
            if latest_version:
                metrics = latest_version.performance_metrics

                if self.min_accuracy and metrics.accuracy < self.min_accuracy:
                    return False

                if (
                    self.max_inference_time
                    and metrics.inference_time > self.max_inference_time
                ):
                    return False

        return True


@dataclass
class ModelRecommendation:
    """Model recommendation for a specific use case."""

    model: Model
    version: ModelVersion
    confidence: float
    reasoning: str
    expected_performance: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": str(self.model.id),
            "model_name": self.model.name,
            "version": self.version.version_string,
            "algorithm": self.model.algorithm,
            "domain": self.model.domain,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "expected_performance": self.expected_performance,
            "performance_metrics": self.version.performance_metrics.to_dict(),
        }


class ModelRegistryService:
    """Service for managing model registries and providing model discovery.

    This service provides comprehensive model lifecycle management including:
    - Model registration and cataloging
    - Version management across environments
    - Access control and permissions
    - Model search and discovery
    - Performance tracking and comparison
    - Recommendation engine for model selection
    """

    def __init__(self, storage_path: Path, default_registry_name: str = "default"):
        """Initialize model registry service.

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
                description="Default model registry",
                creator="system",
            )

    async def create_registry(
        self,
        name: str,
        description: str,
        creator: str,
        access_policy: AccessPolicy | None = None,
    ) -> ModelRegistry:
        """Create a new model registry.

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

    async def register_model(
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
    ) -> Model:
        """Register a new model in the registry.

        Args:
            name: Model name
            description: Model description
            algorithm: Algorithm used
            domain: Application domain
            owner: Model owner
            tags: Optional tags
            metadata: Optional metadata
            registry_name: Registry to use (default if None)
            user: User registering the model
            groups: User's groups

        Returns:
            Registered Model

        Raises:
            ModelAlreadyExistsError: If model name already exists
            AccessDeniedError: If user lacks permission
        """
        registry = await self.get_registry(registry_name)

        # Check if model name already exists
        existing_model = registry.get_model_by_name(name, user, groups)
        if existing_model:
            raise ModelAlreadyExistsError(f"Model '{name}' already exists")

        model = Model(
            name=name,
            description=description,
            algorithm=algorithm,
            domain=domain,
            owner=owner,
            tags=tags or [],
            metadata=metadata or {},
        )

        registry.register_model(model, user, groups)
        await self._save_registry(registry)

        return model

    async def add_model_version(
        self,
        model_id: UUID,
        version: ModelVersion,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> None:
        """Add a version to an existing model.

        Args:
            model_id: Model ID
            version: ModelVersion to add
            registry_name: Registry name
            user: User adding the version
            groups: User's groups

        Raises:
            ModelNotFoundError: If model not found
            AccessDeniedError: If user lacks permission
        """
        registry = await self.get_registry(registry_name)
        model = registry.get_model(model_id, user, groups)

        if not model:
            raise ModelNotFoundError(f"Model {model_id} not found")

        if not registry.access_policy.can_write(user, groups):
            raise AccessDeniedError(f"User {user} lacks write permission")

        model.add_version(version)
        await self._save_registry(registry)

    async def get_model(
        self,
        model_id: UUID,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> Model:
        """Get a model by ID.

        Args:
            model_id: Model ID
            registry_name: Registry name
            user: User requesting the model
            groups: User's groups

        Returns:
            Model

        Raises:
            ModelNotFoundError: If model not found
        """
        registry = await self.get_registry(registry_name)
        model = registry.get_model(model_id, user, groups)

        if not model:
            raise ModelNotFoundError(f"Model {model_id} not found")

        return model

    async def get_model_by_name(
        self,
        name: str,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> Model:
        """Get a model by name.

        Args:
            name: Model name
            registry_name: Registry name
            user: User requesting the model
            groups: User's groups

        Returns:
            Model

        Raises:
            ModelNotFoundError: If model not found
        """
        registry = await self.get_registry(registry_name)
        model = registry.get_model_by_name(name, user, groups)

        if not model:
            raise ModelNotFoundError(f"Model '{name}' not found")

        return model

    async def list_models(
        self,
        criteria: SearchCriteria | None = None,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> list[Model]:
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

        models = registry.list_models(
            user=user,
            groups=groups,
            domain_filter=criteria.domain,
            algorithm_filter=criteria.algorithm,
            tag_filter=criteria.tags,
            owner_filter=criteria.owner,
            include_archived=criteria.include_archived,
        )

        # Apply additional criteria filtering
        filtered_models = [model for model in models if criteria.matches_model(model)]

        # Apply limit
        if criteria.limit:
            filtered_models = filtered_models[: criteria.limit]

        return filtered_models

    async def search_models(
        self,
        query: str,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
        limit: int | None = None,
    ) -> list[Model]:
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
        return registry.search_models(query, user, groups, limit)

    async def get_model_version(
        self,
        model_id: UUID,
        version: str,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> ModelVersion:
        """Get a specific model version.

        Args:
            model_id: Model ID
            version: Version string
            registry_name: Registry name
            user: User requesting the version
            groups: User's groups

        Returns:
            ModelVersion

        Raises:
            ModelNotFoundError: If model not found
            VersionNotFoundError: If version not found
        """
        model = await self.get_model(model_id, registry_name, user, groups)
        model_version = model.get_version(version)

        if not model_version:
            raise VersionNotFoundError(
                f"Version {version} not found for model {model_id}"
            )

        return model_version

    async def promote_to_staging(
        self,
        model_id: UUID,
        version_id: UUID,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> None:
        """Promote a model version to staging.

        Args:
            model_id: Model ID
            version_id: Version ID to promote
            registry_name: Registry name
            user: User performing promotion
            groups: User's groups
        """
        registry = await self.get_registry(registry_name)
        model = registry.get_model(model_id, user, groups)

        if not model:
            raise ModelNotFoundError(f"Model {model_id} not found")

        if not registry.access_policy.can_write(user, groups):
            raise AccessDeniedError(f"User {user} lacks write permission")

        model.promote_to_staging(version_id)
        await self._save_registry(registry)

    async def promote_to_production(
        self,
        model_id: UUID,
        version_id: UUID,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> None:
        """Promote a model version to production.

        Args:
            model_id: Model ID
            version_id: Version ID to promote
            registry_name: Registry name
            user: User performing promotion
            groups: User's groups
        """
        registry = await self.get_registry(registry_name)
        model = registry.get_model(model_id, user, groups)

        if not model:
            raise ModelNotFoundError(f"Model {model_id} not found")

        if not registry.access_policy.is_admin(user, groups):
            raise AccessDeniedError(
                f"User {user} lacks admin permission for production promotion"
            )

        model.promote_to_production(version_id)
        await self._save_registry(registry)

    async def compare_model_versions(
        self,
        model_id: UUID,
        version1: str,
        version2: str,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare two versions of a model.

        Args:
            model_id: Model ID
            version1: First version string
            version2: Second version string
            registry_name: Registry name
            user: User performing comparison
            groups: User's groups

        Returns:
            Comparison results
        """
        v1 = await self.get_model_version(
            model_id, version1, registry_name, user, groups
        )
        v2 = await self.get_model_version(
            model_id, version2, registry_name, user, groups
        )

        performance_diff = v1.performance_metrics.compare_with(v2.performance_metrics)

        return {
            "model_id": str(model_id),
            "version1": {
                "version": v1.version_string,
                "status": v1.status.value,
                "created_at": v1.created_at.isoformat(),
                "performance": v1.performance_metrics.to_dict(),
            },
            "version2": {
                "version": v2.version_string,
                "status": v2.status.value,
                "created_at": v2.created_at.isoformat(),
                "performance": v2.performance_metrics.to_dict(),
            },
            "performance_difference": performance_diff,
            "version1_is_better": v1.performance_metrics.is_better_than(
                v2.performance_metrics
            ),
            "recommendations": self._generate_version_recommendations(v1, v2),
        }

    async def recommend_models(
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
            performance_requirements: Required performance metrics
            registry_name: Registry name
            user: User requesting recommendations
            groups: User's groups
            limit: Maximum recommendations

        Returns:
            List of model recommendations
        """
        # Get models in the domain
        criteria = SearchCriteria(domain=domain, include_archived=False)
        models = await self.list_models(criteria, registry_name, user, groups)

        recommendations = []

        for model in models:
            latest_version = model.get_latest_version()
            if not latest_version:
                continue

            # Calculate recommendation score
            score, reasoning = self._calculate_recommendation_score(
                model, latest_version, use_case, performance_requirements
            )

            if score > 0.3:  # Minimum threshold
                recommendation = ModelRecommendation(
                    model=model,
                    version=latest_version,
                    confidence=score,
                    reasoning=reasoning,
                    expected_performance=latest_version.performance_metrics.to_dict(),
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

    async def archive_model(
        self,
        model_id: UUID,
        registry_name: str | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> None:
        """Archive a model.

        Args:
            model_id: Model ID
            registry_name: Registry name
            user: User archiving the model
            groups: User's groups
        """
        registry = await self.get_registry(registry_name)
        model = registry.get_model(model_id, user, groups)

        if not model:
            raise ModelNotFoundError(f"Model {model_id} not found")

        if not registry.access_policy.can_write(user, groups):
            raise AccessDeniedError(f"User {user} lacks write permission")

        model.archive()
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
        model: Model,
        version: ModelVersion,
        use_case: str,
        performance_requirements: dict[str, float] | None,
    ) -> tuple[float, str]:
        """Calculate recommendation score for a model."""
        score = 0.0
        reasons = []

        # Base score from performance
        metrics = version.performance_metrics
        score += metrics.performance_score * 0.4

        # Algorithm suitability (simplified)
        algorithm_scores = {
            "IsolationForest": 0.9,
            "LocalOutlierFactor": 0.8,
            "OneClassSVM": 0.7,
            "EllipticEnvelope": 0.6,
        }

        algo_score = algorithm_scores.get(model.algorithm, 0.5)
        score += algo_score * 0.3
        reasons.append(f"Algorithm {model.algorithm} suitable for anomaly detection")

        # Performance requirements check
        if performance_requirements:
            meets_requirements = True
            for metric, requirement in performance_requirements.items():
                if hasattr(metrics, metric):
                    actual_value = getattr(metrics, metric)
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
        days_since_update = (datetime.utcnow() - model.updated_at).days
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

        perf_diff = version1.performance_metrics.compare_with(
            version2.performance_metrics
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
