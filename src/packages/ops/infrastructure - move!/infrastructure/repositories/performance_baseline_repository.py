"""Repository for performance baselines."""

from uuid import UUID

from monorepo.domain.entities.model_performance import ModelPerformanceBaseline


class PerformanceBaselineRepository:
    """Repository for model performance baseline data."""

    def __init__(self):
        """Initialize the baseline repository."""
        self._baselines: dict[UUID, ModelPerformanceBaseline] = {}

    async def save(self, baseline: ModelPerformanceBaseline) -> None:
        """Save a model performance baseline record."""
        self._baselines[baseline.id] = baseline

    async def get_by_id(self, baseline_id: UUID) -> ModelPerformanceBaseline | None:
        """Get a model performance baseline by ID."""
        return self._baselines.get(baseline_id)

    async def get_all(self) -> list[ModelPerformanceBaseline]:
        """Get all baseline records."""
        return list(self._baselines.values())

    async def delete(self, baseline_id: UUID) -> None:
        """Delete a baseline record."""
        if baseline_id in self._baselines:
            del self._baselines[baseline_id]
