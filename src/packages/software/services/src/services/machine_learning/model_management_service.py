"""Application service for managing ML models and versions."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from monorepo.domain.entities import (
    Processor,
    ModelStage,
    ModelStatus,
    ModelType,
    ModelVersion,
)
from monorepo.domain.exceptions import InvalidModelStateError, ModelNotFoundError
from monorepo.domain.value_objects import (
    ModelStorageInfo,
    PerformanceMetrics,
    SemanticVersion,
)
from monorepo.shared.protocols import (
    ModelRepositoryProtocol,
    ModelVersionRepositoryProtocol,
)


class ModelManagementService:
    """Service for managing ML models and their versions."""

    def __init__(
        self,
        processor_repository: ModelRepositoryProtocol,
        processor_version_repository: ModelVersionRepositoryProtocol,
    ):
        """Initialize the service.

        Args:
            processor_repository: Repository for models
            processor_version_repository: Repository for processor versions
        """
        self.processor_repository = processor_repository
        self.processor_version_repository = processor_version_repository

    async def create_processor(
        self,
        name: str,
        description: str,
        processor_type: ModelType,
        algorithm_family: str,
        created_by: str,
        team: str = "",
        use_cases: list[str] | None = None,
        data_requirements: dict[str, Any] | None = None,
    ) -> Processor:
        """Create a new processor.

        Args:
            name: Name of the processor
            description: Description of the processor
            processor_type: Type of processor
            algorithm_family: Algorithm family
            created_by: User creating the processor
            team: Team responsible for the processor
            use_cases: List of use cases
            data_requirements: Data requirements for the processor

        Returns:
            Created processor
        """
        # Check if processor name already exists
        existing_processors = await self.processor_repository.find_by_name(name)
        if existing_processors:
            raise ValueError(f"Processor with name '{name}' already exists")

        processor = Processor(
            name=name,
            description=description,
            processor_type=processor_type,
            algorithm_family=algorithm_family,
            created_by=created_by,
            team=team,
            use_cases=use_cases or [],
            data_requirements=data_requirements or {},
        )

        await self.processor_repository.save(processor)
        return processor

    async def create_processor_version(
        self,
        processor_id: UUID,
        detector_id: UUID,
        version: SemanticVersion,
        performance_measurements: PerformanceMetrics,
        storage_info: ModelStorageInfo,
        created_by: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> ModelVersion:
        """Create a new version of a processor.

        Args:
            processor_id: ID of the parent processor
            detector_id: ID of the detector this version was created from
            version: Semantic version
            performance_measurements: Performance measurements for this version
            storage_info: Storage information
            created_by: User creating the version
            description: Description of this version
            tags: Tags for the version

        Returns:
            Created processor version
        """
        # Verify processor exists
        processor = await self.processor_repository.find_by_id(processor_id)
        if not processor:
            raise ModelNotFoundError(processor_id=processor_id)

        # Check if version already exists
        existing_version = (
            await self.processor_version_repository.find_by_processor_and_version(
                processor_id, version
            )
        )
        if existing_version:
            raise ValueError(
                f"Version {version.version_string} already exists for processor {processor_id}"
            )

        processor_version = ModelVersion(
            processor_id=processor_id,
            detector_id=detector_id,
            version=version,
            performance_measurements=performance_measurements,
            storage_info=storage_info,
            created_by=created_by,
            description=description,
            tags=tags or [],
        )

        await self.processor_version_repository.save(processor_version)

        # Update processor's latest version
        processor.set_latest_version(processor_version.id)
        await self.processor_repository.save(processor)

        return processor_version

    async def promote_to_production(
        self,
        processor_id: UUID,
        version_id: UUID,
        promoted_by: str,
        validation_results: dict[str, Any] | None = None,
    ) -> Processor:
        """Promote a processor version to production.

        Args:
            processor_id: ID of the processor
            version_id: ID of the version to promote
            promoted_by: User promoting the processor
            validation_results: Results of pre-deployment validation

        Returns:
            Updated processor
        """
        # Verify processor and version exist
        processor = await self.processor_repository.find_by_id(processor_id)
        if not processor:
            raise ModelNotFoundError(processor_id=processor_id)

        processor_version = await self.processor_version_repository.find_by_id(version_id)
        if not processor_version or processor_version.processor_id != processor_id:
            raise ValueError(f"Version {version_id} not found for processor {processor_id}")

        # Check deployment readiness
        can_deploy, issues = processor.can_deploy()
        if not can_deploy:
            raise InvalidModelStateError(
                processor_id=processor_id,
                operation="promote_to_production",
                reason=f"Processor not ready for deployment: {'; '.join(issues)}",
            )

        # Update processor version status
        processor_version.update_status(ModelStatus.DEPLOYED)

        # Demote current production version if exists
        if processor.current_version_id:
            current_version = await self.processor_version_repository.find_by_id(
                processor.current_version_id
            )
            if current_version and current_version.is_deployed:
                current_version.update_status(ModelStatus.DEPRECATED)
                current_version.update_metadata(
                    "demoted_at", datetime.utcnow().isoformat()
                )
                current_version.update_metadata("demoted_by", promoted_by)
                await self.processor_version_repository.save(current_version)

        # Promote new version
        processor.promote_to_production(version_id, promoted_by)

        # Store validation results
        if validation_results:
            processor.update_metadata(
                "last_validation_results", validation_results, promoted_by
            )

        await self.processor_repository.save(processor)
        await self.processor_version_repository.save(processor_version)

        return processor

    async def deprecate_processor(
        self, processor_id: UUID, deprecated_by: str, reason: str = ""
    ) -> Processor:
        """Deprecate a processor.

        Args:
            processor_id: ID of the processor to deprecate
            deprecated_by: User deprecating the processor
            reason: Reason for deprecation

        Returns:
            Updated processor
        """
        processor = await self.processor_repository.find_by_id(processor_id)
        if not processor:
            raise ModelNotFoundError(processor_id=processor_id)

        if processor.is_archived:
            raise InvalidModelStateError(
                processor_id=processor_id,
                operation="deprecate",
                reason="Cannot deprecate archived processor",
            )

        # Deprecate current production version if exists
        if processor.current_version_id:
            current_version = await self.processor_version_repository.find_by_id(
                processor.current_version_id
            )
            if current_version and current_version.is_deployed:
                current_version.update_status(ModelStatus.DEPRECATED)
                current_version.update_metadata("deprecated_by", deprecated_by)
                current_version.update_metadata("deprecation_reason", reason)
                await self.processor_version_repository.save(current_version)

        processor.update_stage(ModelStage.ARCHIVED)
        processor.update_metadata("deprecated_by", deprecated_by)
        if reason:
            processor.update_metadata("deprecation_reason", reason)

        await self.processor_repository.save(processor)
        return processor

    async def get_processor_with_versions(self, processor_id: UUID) -> dict[str, Any]:
        """Get processor with all its versions.

        Args:
            processor_id: ID of the processor

        Returns:
            Processor information with versions
        """
        processor = await self.processor_repository.find_by_id(processor_id)
        if not processor:
            raise ModelNotFoundError(processor_id=processor_id)

        versions = await self.processor_version_repository.find_by_processor_id(processor_id)

        return {
            "processor": processor.get_info(),
            "versions": [version.get_info() for version in versions],
            "current_version": None,
            "latest_version": None,
        }

    async def compare_processor_versions(
        self, version_id_1: UUID, version_id_2: UUID
    ) -> dict[str, Any]:
        """Compare two processor versions.

        Args:
            version_id_1: First version to compare
            version_id_2: Second version to compare

        Returns:
            Comparison results
        """
        version1 = await self.processor_version_repository.find_by_id(version_id_1)
        version2 = await self.processor_version_repository.find_by_id(version_id_2)

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

    async def get_production_processors(self) -> list[Processor]:
        """Get all models currently in production.

        Returns:
            List of production models
        """
        all_processors = await self.processor_repository.find_by_stage(ModelStage.PRODUCTION)
        return [processor for processor in all_processors if processor.has_current_version]

    async def get_processor_performance_history(self, processor_id: UUID) -> dict[str, Any]:
        """Get performance history across all versions of a processor.

        Args:
            processor_id: ID of the processor

        Returns:
            Performance history data
        """
        processor = await self.processor_repository.find_by_id(processor_id)
        if not processor:
            raise ModelNotFoundError(processor_id=processor_id)

        versions = await self.processor_version_repository.find_by_processor_id(processor_id)

        # Sort by creation date
        versions.sort(key=lambda v: v.created_at)

        history = {
            "processor_id": str(processor_id),
            "processor_name": processor.name,
            "version_count": len(versions),
            "measurements_over_time": [],
            "performance_trend": {},
        }

        measurements_data = []
        for version in versions:
            performance = version.get_performance_summary()
            performance["version"] = version.version_string
            performance["created_at"] = version.created_at.isoformat()
            performance["status"] = version.status.value
            measurements_data.append(performance)

        history["measurements_over_time"] = measurements_data

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
