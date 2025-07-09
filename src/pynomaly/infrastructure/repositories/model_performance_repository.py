"""Model performance repository implementation."""

from uuid import UUID

from pynomaly.domain.entities.model_performance import (
    ModelPerformanceBaseline,
    ModelPerformanceMetrics,
)


class ModelPerformanceRepository:
    """Repository for model performance data."""

    def __init__(self):
        """Initialize the repository."""
        self._performances: dict[UUID, ModelPerformanceMetrics] = {}

    async def save(self, performance: ModelPerformanceMetrics) -> None:
        """Save a model performance record."""
        self._performances[performance.id] = performance

    async def get_by_id(
        self, performance_id: UUID
    ) -> ModelPerformanceMetrics | None:
        """Get a model performance record by ID."""
        return self._performances.get(performance_id)

    async def get_by_model_id(self, model_id: str) -> list[ModelPerformanceMetrics]:
        """Get all performance records for a specific model."""
        return [p for p in self._performances.values() if p.model_id == model_id]

    async def get_all(self) -> list[ModelPerformanceMetrics]:
        """Get all performance records."""
        return list(self._performances.values())

    async def delete(self, performance_id: UUID) -> None:
        """Delete a performance record."""
        if performance_id in self._performances:
            del self._performances[performance_id]


class PerformanceBaselineRepository:
    """Repository for performance baseline data."""

    def __init__(self):
        """Initialize the repository."""
        self._baselines: dict[UUID, ModelPerformanceBaseline] = {}

    async def save(self, baseline: ModelPerformanceBaseline) -> None:
        """Save a performance baseline record."""
        self._baselines[baseline.id] = baseline

    async def get_by_id(self, baseline_id: UUID) -> ModelPerformanceBaseline | None:
        """Get a performance baseline record by ID."""
        return self._baselines.get(baseline_id)

    async def get_by_model_id(self, model_id: str) -> list[ModelPerformanceBaseline]:
        """Get all baseline records for a specific model."""
        return [b for b in self._baselines.values() if b.model_id == model_id]

    async def get_all(self) -> list[ModelPerformanceBaseline]:
        """Get all baseline records."""
        return list(self._baselines.values())

    async def delete(self, baseline_id: UUID) -> None:
        """Delete a baseline record."""
        if baseline_id in self._baselines:
            del self._baselines[baseline_id]
