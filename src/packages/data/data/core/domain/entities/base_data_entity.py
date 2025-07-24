"""
BaseDataEntity

Base class for all data domain entities.
"""

from abstractions.base_entity import BaseEntity
from core.domain.value_objects.data_identifier import DataIdentifier

class BaseDataEntity(BaseEntity):
    """
    Base class for all data domain entities.
    """
    def __init__(self, id: DataIdentifier):
        if not isinstance(id, DataIdentifier):
            raise TypeError("id must be an instance of DataIdentifier")
        super().__init__(id.value) # Assuming BaseEntity expects a string ID
        self._id = id

    @property
    def id(self) -> DataIdentifier:
        return self._id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseDataEntity):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
