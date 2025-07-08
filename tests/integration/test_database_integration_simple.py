"""Simple database integration tests to verify async repository fixes."""

from uuid import uuid4

import pytest
from pynomaly.domain.entities import Detector
from pynomaly.domain.value_objects import ContaminationRate


class TestDatabaseIntegrationSimple:
    """Simple database integration tests."""

    @pytest.mark.asyncio
    async def test_async_detector_repository_basic_operations(
        self, test_async_database_repositories
    ):
        """Test basic CRUD operations with async detector repository."""
        detector_repo = test_async_database_repositories["detector_repository"]

        # Test count
        initial_count = await detector_repo.count()
        assert initial_count == 0

        # Test save
        detector = Detector(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
        )
        await detector_repo.save(detector)

        # Test count after save
        count_after_save = await detector_repo.count()
        assert count_after_save == 1

        # Test find_by_id
        found_detector = await detector_repo.find_by_id(detector.id)
        assert found_detector is not None
        assert found_detector.name == "test_detector"
        assert found_detector.algorithm_name == "IsolationForest"

        # Test find_all
        all_detectors = await detector_repo.find_all()
        assert len(all_detectors) == 1
        assert all_detectors[0].id == detector.id

        # Test exists
        exists = await detector_repo.exists(detector.id)
        assert exists is True

        # Test find_by_name
        found_by_name = await detector_repo.find_by_name("test_detector")
        assert found_by_name is not None
        assert found_by_name.id == detector.id

        # Test delete
        deleted = await detector_repo.delete(detector.id)
        assert deleted is True

        # Test count after delete
        final_count = await detector_repo.count()
        assert final_count == 0

    @pytest.mark.asyncio
    async def test_async_dataset_repository_basic_operations(
        self, test_async_database_repositories
    ):
        """Test basic CRUD operations with async dataset repository."""
        dataset_repo = test_async_database_repositories["dataset_repository"]

        # Test initial state
        initial_count = await dataset_repo.count()
        assert initial_count == 0

        # Test save (this will be limited since we can't easily create Dataset with data in test)
        # Just verify the async methods are callable
        all_datasets = await dataset_repo.find_all()
        assert isinstance(all_datasets, list)
        assert len(all_datasets) == 0

        # Test non-existent entity
        non_existent = await dataset_repo.find_by_id(uuid4())
        assert non_existent is None

    @pytest.mark.asyncio
    async def test_async_result_repository_basic_operations(
        self, test_async_database_repositories
    ):
        """Test basic CRUD operations with async result repository."""
        result_repo = test_async_database_repositories["result_repository"]

        # Test initial state
        initial_count = await result_repo.count()
        assert initial_count == 0

        # Test find operations (without creating complex DetectionResult)
        all_results = await result_repo.find_all()
        assert isinstance(all_results, list)
        assert len(all_results) == 0

        # Test recent results
        recent_results = await result_repo.find_recent(limit=5)
        assert isinstance(recent_results, list)
        assert len(recent_results) == 0

    def test_async_repository_wrappers_are_properly_configured(
        self, test_async_database_repositories
    ):
        """Test that async repository wrappers are properly configured."""
        repos = test_async_database_repositories

        assert "detector_repository" in repos
        assert "dataset_repository" in repos
        assert "result_repository" in repos

        # Verify they have the expected async methods
        detector_repo = repos["detector_repository"]
        assert hasattr(detector_repo, "save")
        assert hasattr(detector_repo, "find_by_id")
        assert hasattr(detector_repo, "find_all")
        assert hasattr(detector_repo, "count")
        assert hasattr(detector_repo, "exists")
        assert hasattr(detector_repo, "delete")

        # Verify compatibility methods
        assert hasattr(detector_repo, "get")
        assert hasattr(detector_repo, "get_by_id")


if __name__ == "__main__":
    # Quick test to verify the test can be imported
    print("✅ Database integration test module loaded successfully")
