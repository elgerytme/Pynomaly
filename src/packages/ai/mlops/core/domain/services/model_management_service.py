"""Domain service for managing ML models and versions."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from ..entities.model import Model
from ..value_objects.model_value_objects import (
    ModelStorageInfo,
    PerformanceMetrics,
    ModelVersion as SemanticVersion,
)


class ModelRepositoryProtocol(Protocol):
    """Protocol for model repository."""
    
    async def save(self, model: Model) -> Model:
        """Save a model."""
        ...
    
    async def find_by_id(self, model_id: UUID) -> Model | None:
        """Find model by ID."""
        ...


class ModelVersionRepositoryProtocol(Protocol):
    """Protocol for model version repository."""
    
    async def save(self, version: Any) -> Any:
        """Save a model version."""
        ...


class ModelManagementService:
    """Service for managing ML models and their versions."""

    def __init__(
        self,
        model_repository: ModelRepositoryProtocol,
        model_version_repository: ModelVersionRepositoryProtocol,
    ):
        """Initialize the service.

        Args:
            model_repository: Repository for models
            model_version_repository: Repository for model versions
        """
        self.model_repository = model_repository
        self.model_version_repository = model_version_repository

    async def create_model(
        self,
        name: str,
        description: str,
        model_type: ModelType,
        algorithm_family: str,
        created_by: str,
        team: str = "",
        use_cases: list[str] | None = None,
        data_requirements: dict[str, Any] | None = None,
    ) -> Model:
        """Create a new model.

        Args:
            name: Name of the model
            description: Description of the model
            model_type: Type of model
            algorithm_family: Algorithm family
            created_by: User creating the model
            team: Team responsible for the model
            use_cases: List of use cases
            data_requirements: Data requirements for the model

        Returns:
            Created model
        """
        # Check if model name already exists
        existing_models = await self.model_repository.find_by_name(name)
        if existing_models:
            raise ValueError(f"Model with name '{name}' already exists")

        model = Model(
            name=name,
            description=description,
            model_type=model_type,
            algorithm_family=algorithm_family,
            created_by=created_by,
            team=team,
            use_cases=use_cases or [],
            data_requirements=data_requirements or {},
        )

        await self.model_repository.save(model)
        return model

    async def create_model_version(
        self,
        model_id: UUID,
        detector_id: UUID,
        version: SemanticVersion,
        performance_metrics: PerformanceMetrics,
        storage_info: ModelStorageInfo,
        created_by: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> ModelVersion:
        """Create a new version of a model.

        Args:
            model_id: ID of the parent model
            detector_id: ID of the detector this version was created from
            version: Semantic version
            performance_metrics: Performance metrics for this version
            storage_info: Storage information
            created_by: User creating the version
            description: Description of this version
            tags: Tags for the version

        Returns:
            Created model version
        """
        # Verify model exists
        model = await self.model_repository.find_by_id(model_id)
        if not model:
            raise ModelNotFoundError(model_id=model_id)

        # Check if version already exists
        existing_version = (
            await self.model_version_repository.find_by_model_and_version(
                model_id, version
            )
        )
        if existing_version:
            raise ValueError(
                f"Version {version.version_string} already exists for model {model_id}"
            )

        model_version = ModelVersion(
            model_id=model_id,
            detector_id=detector_id,
            version=version,
            performance_metrics=performance_metrics,
            storage_info=storage_info,
            created_by=created_by,
            description=description,
            tags=tags or [],
        )

        await self.model_version_repository.save(model_version)

        # Update model's latest version
        model.set_latest_version(model_version.id)
        await self.model_repository.save(model)

        return model_version

    async def promote_to_production(
        self,
        model_id: UUID,
        version_id: UUID,
        promoted_by: str,
        validation_results: dict[str, Any] | None = None,
    ) -> Model:
        """Promote a model version to production.

        Args:
            model_id: ID of the model
            version_id: ID of the version to promote
            promoted_by: User promoting the model
            validation_results: Results of pre-deployment validation

        Returns:
            Updated model
        """
        # Verify model and version exist
        model = await self.model_repository.find_by_id(model_id)
        if not model:
            raise ModelNotFoundError(model_id=model_id)

        model_version = await self.model_version_repository.find_by_id(version_id)
        if not model_version or model_version.model_id != model_id:
            raise ValueError(f"Version {version_id} not found for model {model_id}")

        # Check deployment readiness
        can_deploy, issues = model.can_deploy()
        if not can_deploy:
            raise InvalidModelStateError(
                model_id=model_id,
                operation="promote_to_production",
                reason=f"Model not ready for deployment: {'; '.join(issues)}",
            )

        # Update model version status
        model_version.update_status(ModelStatus.DEPLOYED)

        # Demote current production version if exists
        if model.current_version_id:
            current_version = await self.model_version_repository.find_by_id(
                model.current_version_id
            )
            if current_version and current_version.is_deployed:
                current_version.update_status(ModelStatus.DEPRECATED)
                current_version.update_metadata(
                    "demoted_at", datetime.utcnow().isoformat()
                )
                current_version.update_metadata("demoted_by", promoted_by)
                await self.model_version_repository.save(current_version)

        # Promote new version
        model.promote_to_production(version_id, promoted_by)

        # Store validation results
        if validation_results:
            model.update_metadata(
                "last_validation_results", validation_results, promoted_by
            )

        await self.model_repository.save(model)
        await self.model_version_repository.save(model_version)

        return model

    async def deprecate_model(
        self, model_id: UUID, deprecated_by: str, reason: str = ""
    ) -> Model:
        """Deprecate a model.

        Args:
            model_id: ID of the model to deprecate
            deprecated_by: User deprecating the model
            reason: Reason for deprecation

        Returns:
            Updated model
        """
        model = await self.model_repository.find_by_id(model_id)
        if not model:
            raise ModelNotFoundError(model_id=model_id)

        if model.is_archived:
            raise InvalidModelStateError(
                model_id=model_id,
                operation="deprecate",
                reason="Cannot deprecate archived model",
            )

        # Deprecate current production version if exists
        if model.current_version_id:
            current_version = await self.model_version_repository.find_by_id(
                model.current_version_id
            )
            if current_version and current_version.is_deployed:
                current_version.update_status(ModelStatus.DEPRECATED)
                current_version.update_metadata("deprecated_by", deprecated_by)
                current_version.update_metadata("deprecation_reason", reason)
                await self.model_version_repository.save(current_version)

        model.update_stage(ModelStage.ARCHIVED)
        model.update_metadata("deprecated_by", deprecated_by)
        if reason:
            model.update_metadata("deprecation_reason", reason)

        await self.model_repository.save(model)
        return model

    async def get_model_with_versions(self, model_id: UUID) -> dict[str, Any]:
        """Get model with all its versions.

        Args:
            model_id: ID of the model

        Returns:
            Model information with versions
        """
        model = await self.model_repository.find_by_id(model_id)
        if not model:
            raise ModelNotFoundError(model_id=model_id)

        versions = await self.model_version_repository.find_by_model_id(model_id)

        return {
            "model": model.get_info(),
            "versions": [version.get_info() for version in versions],
            "current_version": None,
            "latest_version": None,
        }

    async def compare_model_versions(
        self, version_id_1: UUID, version_id_2: UUID
    ) -> dict[str, Any]:
        """Compare two model versions.

        Args:
            version_id_1: First version to compare
            version_id_2: Second version to compare

        Returns:
            Comparison results
        """
        version1 = await self.model_version_repository.find_by_id(version_id_1)
        version2 = await self.model_version_repository.find_by_id(version_id_2)

        if not version1 or not version2:
            raise ValueError("One or both versions not found")

        comparison = version1.compare_performance(version2)

        return {
            "version1": {
                "id": str(version1.id),
                "version": version1.version_string,
                "performance": version1.get_performance_summary(),
            },
            "version2": {
                "id": str(version2.id),
                "version": version2.version_string,
                "performance": version2.get_performance_summary(),
            },
            "performance_difference": comparison,
            "recommendation": self._get_version_recommendation(comparison),
        }

    def _get_version_recommendation(self, comparison: dict[str, float]) -> str:
        """Get recommendation based on performance comparison."""
        accuracy_diff = comparison.get("accuracy", 0)
        f1_diff = comparison.get("f1_score", 0)
        training_time_diff = comparison.get("training_time", 0)

        if accuracy_diff > 0.05 and f1_diff > 0.05:
            return "Version 1 shows significantly better performance"
        elif accuracy_diff < -0.05 and f1_diff < -0.05:
            return "Version 2 shows significantly better performance"
        elif abs(accuracy_diff) < 0.01 and abs(f1_diff) < 0.01:
            if training_time_diff < 0:
                return "Similar performance, Version 1 is faster"
            else:
                return "Similar performance, Version 2 is faster"
        else:
            return "Mixed results, manual review recommended"

    async def get_production_models(self) -> list[Model]:
        """Get all models currently in production.

        Returns:
            List of production models
        """
        all_models = await self.model_repository.find_by_stage(ModelStage.PRODUCTION)
        return [model for model in all_models if model.has_current_version]

    async def get_model_performance_history(self, model_id: UUID) -> dict[str, Any]:
        """Get performance history across all versions of a model.

        Args:
            model_id: ID of the model

        Returns:
            Performance history data
        """
        model = await self.model_repository.find_by_id(model_id)
        if not model:
            raise ModelNotFoundError(model_id=model_id)

        versions = await self.model_version_repository.find_by_model_id(model_id)

        # Sort by creation date
        versions.sort(key=lambda v: v.created_at)

        history = {
            "model_id": str(model_id),
            "model_name": model.name,
            "version_count": len(versions),
            "metrics_over_time": [],
            "performance_trend": {},
        }

        metrics_data = []
        for version in versions:
            performance = version.get_performance_summary()
            performance["version"] = version.version_string
            performance["created_at"] = version.created_at.isoformat()
            performance["status"] = version.status.value
            metrics_data.append(performance)

        history["metrics_over_time"] = metrics_data

        # Calculate trends
        if len(versions) >= 2:
            first_version = versions[0].get_performance_summary()
            latest_version = versions[-1].get_performance_summary()

            for metric in ["accuracy", "precision", "recall", "f1_score"]:
                if metric in first_version and metric in latest_version:
                    change = latest_version[metric] - first_version[metric]
                    history["performance_trend"][metric] = {
                        "change": change,
                        "trend": (
                            "improving"
                            if change > 0
                            else "declining"
                            if change < 0
                            else "stable"
                        ),
                    }

        return history
