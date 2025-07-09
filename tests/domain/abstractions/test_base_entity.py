"""Tests for base entity abstraction."""

from datetime import datetime
from uuid import UUID, uuid4

import pytest

from pynomaly.domain.abstractions.base_entity import BaseEntity
from pynomaly.domain.abstractions.domain_event import DomainEvent


class TestEntity(BaseEntity):
    """Test entity for testing purposes."""

    name: str
    value: int = 0

    def __init__(self, name: str, value: int = 0, **kwargs):
        super().__init__(name=name, value=value, **kwargs)


class TestDomainEvent(DomainEvent):
    """Test domain event."""

    def __init__(self, aggregate_id: UUID, event_data: dict = None, **kwargs):
        super().__init__(
            aggregate_id=aggregate_id,
            aggregate_type="TestEntity",
            aggregate_version=1,
            event_data=event_data or {},
            **kwargs,
        )


class TestBaseEntity:
    """Test cases for BaseEntity."""

    def test_entity_creation(self):
        """Test entity creation with auto-generated ID."""
        entity = TestEntity(name="test", value=42)

        assert entity.name == "test"
        assert entity.value == 42
        assert isinstance(entity.id, UUID)
        assert entity.version == 1
        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.updated_at, datetime)
        assert entity.created_at == entity.updated_at

    def test_entity_creation_with_id(self):
        """Test entity creation with provided ID."""
        entity_id = uuid4()
        entity = TestEntity(name="test", id=entity_id)

        assert entity.id == entity_id
        assert entity.name == "test"

    def test_entity_equality(self):
        """Test entity equality based on ID."""
        entity_id = uuid4()
        entity1 = TestEntity(name="test1", id=entity_id)
        entity2 = TestEntity(name="test2", id=entity_id)
        entity3 = TestEntity(name="test1")

        assert entity1 == entity2  # Same ID, different data
        assert entity1 != entity3  # Different ID

    def test_entity_hash(self):
        """Test entity hash based on ID."""
        entity_id = uuid4()
        entity1 = TestEntity(name="test1", id=entity_id)
        entity2 = TestEntity(name="test2", id=entity_id)

        assert hash(entity1) == hash(entity2)

    def test_entity_repr(self):
        """Test entity string representation."""
        entity = TestEntity(name="test")
        repr_str = repr(entity)

        assert "TestEntity" in repr_str
        assert str(entity.id) in repr_str

    def test_domain_events(self):
        """Test domain event management."""
        entity = TestEntity(name="test")
        event = TestDomainEvent(aggregate_id=entity.id)

        # Initially no events
        assert len(entity.get_domain_events()) == 0

        # Add event
        entity.add_domain_event(event)
        events = entity.get_domain_events()
        assert len(events) == 1
        assert events[0] == event

        # Clear events
        entity.clear_domain_events()
        assert len(entity.get_domain_events()) == 0

    def test_mark_as_updated(self):
        """Test marking entity as updated."""
        entity = TestEntity(name="test")
        original_updated_at = entity.updated_at
        original_version = entity.version

        # Wait a bit to ensure timestamp difference
        import time

        time.sleep(0.01)

        entity.mark_as_updated()

        assert entity.updated_at > original_updated_at
        assert entity.version == original_version + 1

    def test_is_new(self):
        """Test checking if entity is new."""
        entity = TestEntity(name="test")

        assert entity.is_new() is True

        entity.mark_as_updated()
        assert entity.is_new() is False

    def test_clone(self):
        """Test entity cloning."""
        entity = TestEntity(name="test", value=42)
        entity.add_domain_event(TestDomainEvent(aggregate_id=entity.id))

        cloned = entity.clone()

        assert cloned.id != entity.id
        assert cloned.name == entity.name
        assert cloned.value == entity.value
        assert cloned.version == 1
        assert len(cloned.get_domain_events()) == 0

    def test_apply_changes(self):
        """Test applying changes to entity."""
        entity = TestEntity(name="test", value=42)
        original_version = entity.version

        entity.apply_changes({"name": "updated", "value": 100})

        assert entity.name == "updated"
        assert entity.value == 100
        assert entity.version == original_version + 1

    def test_validate_invariants(self):
        """Test entity invariant validation."""
        entity = TestEntity(name="test")

        # Should not raise exception
        entity.validate_invariants()

        # Test invalid version
        entity.version = 0
        with pytest.raises(ValueError, match="Entity version must be positive"):
            entity.validate_invariants()

    def test_to_dict(self):
        """Test converting entity to dictionary."""
        entity = TestEntity(name="test", value=42)
        event = TestDomainEvent(aggregate_id=entity.id)
        entity.add_domain_event(event)

        # Without events
        data = entity.to_dict()
        assert data["name"] == "test"
        assert data["value"] == 42
        assert "domain_events" not in data

        # With events
        data_with_events = entity.to_dict(include_events=True)
        assert "domain_events" in data_with_events
        assert len(data_with_events["domain_events"]) == 1

    def test_create_from_dict(self):
        """Test creating entity from dictionary."""
        data = {"name": "test", "value": 42, "id": str(uuid4()), "version": 2}

        entity = TestEntity.create_from_dict(data)

        assert entity.name == "test"
        assert entity.value == 42
        assert entity.version == 2

    def test_get_entity_type(self):
        """Test getting entity type."""
        entity = TestEntity(name="test")

        assert entity.get_entity_type() == "TestEntity"

    def test_get_identifier_field(self):
        """Test getting identifier field name."""
        assert TestEntity.get_identifier_field() == "id"

    def test_metadata(self):
        """Test entity metadata."""
        entity = TestEntity(name="test")

        assert entity.metadata == {}

        entity.metadata["custom_field"] = "value"
        assert entity.metadata["custom_field"] == "value"
