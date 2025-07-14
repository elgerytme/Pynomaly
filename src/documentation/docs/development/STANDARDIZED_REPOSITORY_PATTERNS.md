# Standardized Repository Patterns

This document outlines the standardized async repository patterns implemented in Pynomaly to ensure consistency, maintainability, and proper async/await support throughout the codebase.

## Overview

All repositories in Pynomaly now follow a standardized async pattern based on the `RepositoryProtocol` interfaces defined in `pynomaly.shared.protocols.repository_protocol`. This standardization addresses previous inconsistencies in async/sync patterns, method naming, and error handling.

## Repository Protocol Hierarchy

### Base Repository Protocol

All repositories implement the base `RepositoryProtocol[T]`:

```python
from pynomaly.shared.protocols import RepositoryProtocol

class RepositoryProtocol(Protocol[T]):
    async def save(self, entity: T) -> None
    async def find_by_id(self, entity_id: UUID) -> T | None
    async def find_all(self) -> list[T]
    async def delete(self, entity_id: UUID) -> bool
    async def exists(self, entity_id: UUID) -> bool
    async def count(self) -> int
```

### Specialized Repository Protocols

Specific entity types have specialized protocols that extend the base protocol:

- `DetectorRepositoryProtocol` - For detector entities
- `DatasetRepositoryProtocol` - For dataset entities  
- `DetectionResultRepositoryProtocol` - For detection result entities
- `ModelRepositoryProtocol` - For model entities
- `ExperimentRepositoryProtocol` - For experiment entities

## Standardized Patterns

### 1. Async-First Design

**All repository methods are async**:

```python
# ✅ Correct - Async method
async def find_by_id(self, entity_id: UUID) -> Entity | None:
    return await self._some_async_operation(entity_id)

# ❌ Incorrect - Sync method
def find_by_id(self, entity_id: UUID) -> Entity | None:
    return self._some_sync_operation(entity_id)
```

### 2. Consistent Method Naming

**Standardized method names across all repositories**:

- `find_by_id()` - Not `get_by_id()` or `get()`
- `find_all()` - Not `get_all()` or `list_all()`
- `save()` - For both create and update operations
- `delete()` - Returns `bool` indicating success
- `exists()` - Returns `bool`
- `count()` - Returns `int`

### 3. UUID-Based Entity IDs

**All entity IDs are UUIDs**:

```python
from uuid import UUID

async def find_by_id(self, entity_id: UUID) -> Entity | None:
    # Implementation
```

### 4. Consistent Return Types

**Standardized return patterns**:

- `save()` returns `None` (idempotent operation)
- `delete()` returns `bool` (True if deleted, False if not found)
- `exists()` returns `bool`
- `count()` returns `int`
- Find methods return `Entity | None` or `list[Entity]`

### 5. Error Handling

**Consistent exception patterns**:

```python
async def save(self, entity: Entity) -> None:
    try:
        # Save operation
        pass
    except DatabaseError as e:
        raise RepositoryError(f"Failed to save entity: {e}") from e
```

## Implementation Examples

### In-Memory Repository

```python
from pynomaly.shared.protocols import DetectorRepositoryProtocol

class InMemoryDetectorRepository(DetectorRepositoryProtocol):
    def __init__(self):
        self._storage: dict[UUID, Detector] = {}
        self._model_artifacts: dict[UUID, bytes] = {}

    async def save(self, entity: Detector) -> None:
        self._storage[entity.id] = entity

    async def find_by_id(self, entity_id: UUID) -> Detector | None:
        return self._storage.get(entity_id)

    async def find_all(self) -> list[Detector]:
        return list(self._storage.values())

    async def delete(self, entity_id: UUID) -> bool:
        if entity_id in self._storage:
            del self._storage[entity_id]
            return True
        return False

    async def exists(self, entity_id: UUID) -> bool:
        return entity_id in self._storage

    async def count(self) -> int:
        return len(self._storage)
```

### Database Repository

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from pynomaly.shared.protocols import DetectorRepositoryProtocol

class DatabaseDetectorRepository(DetectorRepositoryProtocol):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory

    async def save(self, detector: Detector) -> None:
        async with self.session_factory() as session:
            stmt = select(DetectorModel).filter_by(id=detector.id)
            result = await session.execute(stmt)
            existing = result.scalars().first()

            if existing:
                # Update existing
                existing.algorithm = detector.algorithm_name
                # ... update other fields
            else:
                # Insert new
                model = DetectorModel(
                    id=detector.id,
                    algorithm=detector.algorithm_name,
                    # ... other fields
                )
                session.add(model)

            await session.commit()
```

## Repository Factory

Use the `StandardizedRepositoryFactory` to create repository instances:

```python
from pynomaly.infrastructure.repositories import StandardizedRepositoryFactory

# In-memory repositories (for testing)
factory = StandardizedRepositoryFactory("memory")
detector_repo = factory.create_detector_repository()

# Database repositories (for production)
factory = StandardizedRepositoryFactory("database", session_factory)
dataset_repo = factory.create_dataset_repository()

# All repositories at once
repos = factory.create_all_repositories()
```

## Migration Guide

### From Legacy Repositories

**Old pattern (deprecated)**:
```python
# Old sync repository
class LegacyRepository:
    def save(self, entity) -> entity:  # Returns entity
        pass
    
    def get_by_id(self, entity_id: str) -> Entity:  # String ID
        pass
```

**New pattern (standardized)**:
```python
# New async repository
class StandardRepository(RepositoryProtocol[Entity]):
    async def save(self, entity: Entity) -> None:  # Returns None
        pass
    
    async def find_by_id(self, entity_id: UUID) -> Entity | None:  # UUID ID
        pass
```

### Migration Steps

1. **Update repository interface**:
   - Extend appropriate protocol from `pynomaly.shared.protocols`
   - Make all methods async
   - Use standardized method names
   - Use UUID for entity IDs

2. **Update service layer**:
   - Add `await` to all repository method calls
   - Update error handling for new exception patterns
   - Use repository factory for dependency injection

3. **Update tests**:
   - Make test methods async
   - Use standardized in-memory repositories
   - Update assertions for new return types

## Benefits

### 1. Consistency
- All repositories follow the same pattern
- Predictable method names and signatures
- Consistent error handling

### 2. Type Safety
- Full type hints with protocols
- Generic type support
- Compile-time interface validation

### 3. Async Performance
- Non-blocking I/O operations
- Better resource utilization
- Improved scalability

### 4. Testability
- Easy mocking with protocols
- Standardized in-memory implementations
- Consistent test patterns

### 5. Maintainability
- Single source of truth for repository interfaces
- Easy to add new repository implementations
- Clear migration path for legacy code

## Best Practices

### 1. Always Use Factory

```python
# ✅ Good - Use factory
from pynomaly.infrastructure.repositories import get_memory_repositories
repos = get_memory_repositories()

# ❌ Bad - Direct instantiation
repo = InMemoryDetectorRepository()
```

### 2. Proper Error Handling

```python
# ✅ Good - Specific exception handling
try:
    await detector_repo.save(detector)
except RepositoryError as e:
    logger.error(f"Failed to save detector: {e}")
    raise ServiceError("Could not save detector") from e
```

### 3. Use Type Hints

```python
# ✅ Good - Full type hints
async def get_detector(
    repo: DetectorRepositoryProtocol, 
    detector_id: UUID
) -> Detector | None:
    return await repo.find_by_id(detector_id)
```

### 4. Consistent Naming

```python
# ✅ Good - Standard names
detector_repo = factory.create_detector_repository()
datasets = await dataset_repo.find_all()

# ❌ Bad - Non-standard names
detector_repository_instance = factory.get_detector_repo()
datasets = await dataset_repo.list_datasets()
```

## Testing

### Unit Tests

```python
import pytest
from pynomaly.infrastructure.repositories import get_memory_repositories

@pytest.fixture
async def repositories():
    return get_memory_repositories()

async def test_detector_repository(repositories):
    detector_repo = repositories["detector"]
    
    # Test save
    await detector_repo.save(detector)
    
    # Test find_by_id
    found = await detector_repo.find_by_id(detector.id)
    assert found == detector
    
    # Test exists
    assert await detector_repo.exists(detector.id)
    
    # Test count
    assert await detector_repo.count() == 1
```

### Integration Tests

```python
async def test_database_repository_integration(db_session_factory):
    factory = StandardizedRepositoryFactory("database", db_session_factory)
    detector_repo = factory.create_detector_repository()
    
    # Test persistence across sessions
    await detector_repo.save(detector)
    found = await detector_repo.find_by_id(detector.id)
    assert found.algorithm_name == detector.algorithm_name
```

## Future Considerations

1. **Repository Middleware**: Add support for caching, metrics, etc.
2. **Query Builders**: Standardized query interface for complex queries
3. **Transaction Support**: Coordinated multi-repository operations
4. **Event Sourcing**: Optional event-driven repository pattern
5. **Monitoring**: Built-in performance and health monitoring

This standardization provides a solid foundation for consistent, maintainable, and scalable repository patterns throughout the Pynomaly codebase.