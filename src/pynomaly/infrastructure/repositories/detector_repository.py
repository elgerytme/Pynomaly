"""Detector repository implementation."""

from uuid import UUID

from pynomaly.domain.entities import Detector
from pynomaly.shared.protocols import DetectorRepositoryProtocol


class DetectorRepository(DetectorRepositoryProtocol):
    """Repository for detector entities - standardized async implementation."""

    def __init__(self):
        """Initialize repository."""
        self._items: dict[UUID, Detector] = {}
        self._model_artifacts: dict[UUID, bytes] = {}
        self._name_index: dict[str, UUID] = {}

    async def save(self, entity: Detector) -> None:
        """Save detector."""
        self._items[entity.id] = entity
        # Update name index if detector has a name in metadata
        if hasattr(entity, 'name') and entity.name:
            self._name_index[entity.name] = entity.id
        elif entity.metadata and entity.metadata.get('name'):
            self._name_index[entity.metadata['name']] = entity.id

    async def find_by_id(self, entity_id: UUID) -> Detector | None:
        """Find detector by ID."""
        return self._items.get(entity_id)

    async def find_all(self) -> list[Detector]:
        """Find all detectors."""
        return list(self._items.values())

    async def delete(self, entity_id: UUID) -> bool:
        """Delete detector."""
        if entity_id in self._items:
            detector = self._items[entity_id]
            del self._items[entity_id]

            # Remove from name index
            if hasattr(detector, 'name') and detector.name in self._name_index:
                del self._name_index[detector.name]
            elif detector.metadata and detector.metadata.get('name') in self._name_index:
                del self._name_index[detector.metadata['name']]

            # Remove model artifact if exists
            if entity_id in self._model_artifacts:
                del self._model_artifacts[entity_id]

            return True
        return False

    async def exists(self, entity_id: UUID) -> bool:
        """Check if detector exists."""
        return entity_id in self._items

    async def count(self) -> int:
        """Count detectors."""
        return len(self._items)

    async def find_by_name(self, name: str) -> Detector | None:
        """Find a detector by name."""
        detector_id = self._name_index.get(name)
        if detector_id:
            return self._items.get(detector_id)
        return None

    async def find_by_algorithm(self, algorithm_name: str) -> list[Detector]:
        """Find all detectors using a specific algorithm."""
        return [
            detector
            for detector in self._items.values()
            if detector.algorithm_name == algorithm_name
        ]

    async def find_fitted(self) -> list[Detector]:
        """Find all fitted detectors."""
        return [detector for detector in self._items.values() if detector.is_fitted]

    async def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
        """Save the trained model artifact."""
        self._model_artifacts[detector_id] = artifact

    async def load_model_artifact(self, detector_id: UUID) -> bytes | None:
        """Load the trained model artifact."""
        return self._model_artifacts.get(detector_id)
