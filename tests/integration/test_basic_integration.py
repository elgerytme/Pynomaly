"""Basic integration tests to verify fixed issues."""

import pytest
import numpy as np
from unittest.mock import Mock

from pynomaly.domain.entities import Dataset
from pynomaly.features.advanced_analytics import TrendAnalyzer
from pynomaly.infrastructure.persistence.memory_repository import MemoryRepository


class TestBasicIntegration:
    """Test basic integration functionality."""

    def test_trend_analyzer_import(self):
        """Test that TrendAnalyzer can be imported."""
        analyzer = TrendAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze_trends')

    def test_memory_repository_import(self):
        """Test that MemoryRepository can be imported."""
        repo = MemoryRepository()
        assert repo is not None
        assert hasattr(repo, 'save')
        assert hasattr(repo, 'find_by_id')

    @pytest.mark.asyncio
    async def test_memory_repository_operations(self):
        """Test basic memory repository operations."""
        repo = MemoryRepository()
        
        # Test save and find
        entity = Mock()
        entity.id = "test_id"
        entity.name = "test_entity"
        
        await repo.save(entity)
        
        found = await repo.find_by_id("test_id")
        assert found is not None
        assert found.name == "test_entity"
        
        # Test list all
        all_entities = await repo.list_all()
        assert len(all_entities) == 1
        
        # Test count
        count = await repo.count()
        assert count == 1
        
        # Test delete
        await repo.delete("test_id")
        found = await repo.find_by_id("test_id")
        assert found is None

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        data = np.random.rand(100, 5)
        dataset = Dataset(name="test_dataset", data=data)
        
        assert dataset.name == "test_dataset"
        assert dataset.data.shape == (100, 5)

    def test_import_structure(self):
        """Test that key imports work without MRO conflicts."""
        # This would fail if there are MRO conflicts
        from pynomaly.features.advanced_analytics import TrendAnalyzer
        from pynomaly.infrastructure.persistence.memory_repository import MemoryRepository
        
        analyzer = TrendAnalyzer()
        repo = MemoryRepository()
        
        assert analyzer is not None
        assert repo is not None