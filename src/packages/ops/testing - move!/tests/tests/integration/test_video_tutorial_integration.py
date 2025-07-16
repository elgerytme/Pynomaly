"""
Integration tests for video tutorial system.

Tests the complete video tutorial management system including service,
CLI integration, and web interface components.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from uuid import UUID

import pytest

from monorepo.application.services.video_tutorial_service import (
    VideoTutorial,
    VideoTutorialService,
    VideoSeries,
    UserProgress,
    VideoAnalytics,
)


class TestVideoTutorialServiceIntegration:
    """Integration tests for video tutorial service."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    async def video_service(self, temp_storage):
        """Create video tutorial service with test data."""
        service = VideoTutorialService(temp_storage)
        # Initialize with default content
        return service

    @pytest.mark.asyncio
    async def test_service_initialization_with_default_content(self, video_service):
        """Test service initializes with default video content."""
        # Check that default series were created
        series_list = await video_service.get_video_series()
        assert len(series_list) >= 3  # At least 3 default series
        
        # Verify series categories
        categories = {series.category for series in series_list}
        expected_categories = {"getting_started", "algorithms", "applications"}
        assert expected_categories.issubset(categories)
        
        # Check that tutorials were created
        quickstart_series = next(
            (s for s in series_list if s.name == "Pynomaly Quickstart"), None
        )
        assert quickstart_series is not None
        assert len(quickstart_series.videos) >= 3

    @pytest.mark.asyncio
    async def test_video_series_management(self, video_service):
        """Test video series creation and retrieval."""
        series_list = await video_service.get_video_series()
        initial_count = len(series_list)
        
        # Get a specific series
        if series_list:
            series_id = series_list[0].id
            series = await video_service.get_video_series_by_id(series_id)
            assert series is not None
            assert series.id == series_id

    @pytest.mark.asyncio
    async def test_tutorial_search_functionality(self, video_service):
        """Test tutorial search with various filters."""
        # Search for installation-related tutorials
        results = await video_service.search_tutorials("installation")
        assert len(results) > 0
        
        # Verify results contain installation-related content
        installation_tutorial = next(
            (t for t in results if "installation" in t.title.lower()), None
        )
        assert installation_tutorial is not None
        
        # Search by category
        results = await video_service.search_tutorials(
            query="algorithm", category="algorithms"
        )
        # Should only return results from algorithms category
        for result in results:
            # Find the series for this tutorial
            series_list = await video_service.get_video_series()
            tutorial_series = next(
                (s for s in series_list if any(v.id == result.id for v in s.videos)),
                None
            )
            if tutorial_series:
                assert tutorial_series.category == "algorithms"

    @pytest.mark.asyncio
    async def test_user_progress_tracking(self, video_service):
        """Test user progress tracking and analytics."""
        user_id = "test_user_123"
        
        # Get a tutorial to track progress for
        series_list = await video_service.get_video_series()
        assert len(series_list) > 0
        
        first_series = series_list[0]
        assert len(first_series.videos) > 0
        
        tutorial = first_series.videos[0]
        
        # Track video view
        await video_service.track_video_view(
            tutorial_id=tutorial.id,
            user_id=user_id,
            watch_time=120,  # 2 minutes
            device_type="desktop"
        )
        
        # Check progress was recorded
        progress_list = await video_service.get_user_progress(user_id)
        assert len(progress_list) == 1
        
        progress = progress_list[0]
        assert progress.user_id == user_id
        assert progress.tutorial_id == tutorial.id
        assert progress.watch_time_seconds == 120
        
        # Calculate expected completion percentage
        total_duration = tutorial.duration_minutes * 60
        expected_percentage = min(100.0, (120 / total_duration) * 100)
        assert abs(progress.completion_percentage - expected_percentage) < 0.1

    @pytest.mark.asyncio
    async def test_video_analytics_tracking(self, video_service):
        """Test video analytics and engagement tracking."""
        # Get a tutorial
        series_list = await video_service.get_video_series()
        tutorial = series_list[0].videos[0]
        
        # Track multiple views
        for i in range(3):
            await video_service.track_video_view(
                tutorial_id=tutorial.id,
                user_id=f"user_{i}",
                watch_time=60 + i * 30,
                device_type="mobile" if i % 2 else "desktop"
            )
        
        # Check analytics
        analytics = await video_service.get_tutorial_analytics(tutorial.id)
        assert analytics is not None
        assert analytics.total_views == 3
        assert analytics.total_watch_time == 60 + 90 + 120  # 270 seconds
        assert "mobile" in analytics.device_stats
        assert "desktop" in analytics.device_stats

    @pytest.mark.asyncio
    async def test_quiz_and_bookmark_functionality(self, video_service):
        """Test quiz submission and bookmark management."""
        user_id = "test_user_456"
        
        # Get a tutorial
        series_list = await video_service.get_video_series()
        tutorial = series_list[0].videos[0]
        
        # First track a view to create progress
        await video_service.track_video_view(
            tutorial_id=tutorial.id,
            user_id=user_id,
            watch_time=60
        )
        
        # Submit quiz score
        await video_service.submit_quiz_score(user_id, tutorial.id, 85.5)
        
        # Add bookmarks
        await video_service.add_bookmark(user_id, tutorial.id, 120)  # 2:00
        await video_service.add_bookmark(user_id, tutorial.id, 300)  # 5:00
        
        # Check progress includes quiz and bookmarks
        progress_list = await video_service.get_user_progress(user_id)
        progress = progress_list[0]
        
        assert len(progress.quiz_scores) == 1
        assert progress.quiz_scores[0] == 85.5
        assert len(progress.bookmarks) == 2
        assert 120 in progress.bookmarks
        assert 300 in progress.bookmarks

    @pytest.mark.asyncio
    async def test_tutorial_rating_system(self, video_service):
        """Test tutorial rating and feedback system."""
        tutorial_id = None
        
        # Get a tutorial
        series_list = await video_service.get_video_series()
        tutorial = series_list[0].videos[0]
        tutorial_id = tutorial.id
        
        # Multiple users rate the tutorial
        ratings = [4.0, 5.0, 3.5, 4.5, 5.0]
        for i, rating in enumerate(ratings):
            user_id = f"user_{i}"
            
            # Track view first (required for rating)
            await video_service.track_video_view(
                tutorial_id=tutorial_id,
                user_id=user_id,
                watch_time=60
            )
            
            # Submit rating
            await video_service.rate_tutorial(
                user_id=user_id,
                tutorial_id=tutorial_id,
                rating=rating,
                feedback=f"Great tutorial! Rating: {rating}"
            )
        
        # Check tutorial average rating was updated
        updated_tutorial = await video_service.get_video_tutorial(tutorial_id)
        expected_avg = sum(ratings) / len(ratings)
        assert abs(updated_tutorial.average_rating - expected_avg) < 0.1

    @pytest.mark.asyncio
    async def test_recommendation_system(self, video_service):
        """Test personalized tutorial recommendations."""
        user_id = "recommendation_user"
        
        # Complete some beginner tutorials
        series_list = await video_service.get_video_series()
        quickstart_series = next(
            (s for s in series_list if s.difficulty_level == "beginner"), None
        )
        
        if quickstart_series and quickstart_series.videos:
            for tutorial in quickstart_series.videos[:2]:  # Complete first 2
                # Track full completion
                await video_service.track_video_view(
                    tutorial_id=tutorial.id,
                    user_id=user_id,
                    watch_time=tutorial.duration_minutes * 60  # Full duration
                )
        
        # Get recommendations
        recommendations = await video_service.get_recommended_tutorials(user_id, limit=3)
        
        # Should get intermediate level recommendations since completed some tutorials
        assert len(recommendations) > 0
        # After completing tutorials, should get intermediate recommendations
        # (unless no intermediate tutorials available)

    @pytest.mark.asyncio
    async def test_certificate_generation(self, video_service):
        """Test completion certificate generation."""
        user_id = "certificate_user"
        
        # Get a series
        series_list = await video_service.get_video_series()
        target_series = series_list[0]
        
        # Complete all tutorials in the series
        for tutorial in target_series.videos:
            await video_service.track_video_view(
                tutorial_id=tutorial.id,
                user_id=user_id,
                watch_time=tutorial.duration_minutes * 60  # Full duration for completion
            )
        
        # Generate certificate
        certificate_id = await video_service.generate_certificate(user_id, target_series.id)
        
        assert certificate_id is not None
        assert f"CERT-{target_series.id}" in certificate_id
        assert user_id in certificate_id
        
        # Check that progress records show certificate earned
        progress_list = await video_service.get_user_progress_for_series(user_id, target_series.id)
        for progress in progress_list:
            assert progress.certificate_earned == certificate_id

    @pytest.mark.asyncio
    async def test_learning_analytics(self, video_service):
        """Test comprehensive learning analytics."""
        # Create activity for multiple users
        users = ["analytics_user_1", "analytics_user_2", "analytics_user_3"]
        
        series_list = await video_service.get_video_series()
        tutorial = series_list[0].videos[0]
        
        for user in users:
            await video_service.track_video_view(
                tutorial_id=tutorial.id,
                user_id=user,
                watch_time=180  # 3 minutes
            )
        
        # Get system-wide analytics
        system_analytics = await video_service.get_learning_analytics()
        
        assert "total_tutorials" in system_analytics
        assert "total_series" in system_analytics
        assert "total_users" in system_analytics
        assert system_analytics["total_users"] >= len(users)
        assert system_analytics["total_views"] >= len(users)
        
        # Get user-specific analytics
        user_analytics = await video_service.get_learning_analytics(users[0])
        
        assert "tutorials_started" in user_analytics
        assert "tutorials_completed" in user_analytics
        assert "total_watch_time_minutes" in user_analytics
        assert user_analytics["tutorials_started"] >= 1

    @pytest.mark.asyncio
    async def test_data_persistence(self, temp_storage):
        """Test that video tutorial data persists across service restarts."""
        # Create first service instance
        service1 = VideoTutorialService(temp_storage)
        
        # Get initial data
        initial_series = await service1.get_video_series()
        initial_count = len(initial_series)
        
        # Track some user progress
        user_id = "persistence_user"
        if initial_series and initial_series[0].videos:
            tutorial = initial_series[0].videos[0]
            await service1.track_video_view(
                tutorial_id=tutorial.id,
                user_id=user_id,
                watch_time=120
            )
        
        # Create second service instance (simulating restart)
        service2 = VideoTutorialService(temp_storage)
        
        # Verify data was loaded
        loaded_series = await service2.get_video_series()
        assert len(loaded_series) == initial_count
        
        # Verify user progress was loaded
        progress = await service2.get_user_progress(user_id)
        assert len(progress) >= 1
        assert progress[0].watch_time_seconds == 120

    def test_video_tutorial_data_models(self):
        """Test video tutorial data model validation."""
        from datetime import datetime
        from uuid import uuid4
        
        # Test VideoTutorial model
        tutorial = VideoTutorial(
            title="Test Tutorial",
            description="A test tutorial",
            series="Test Series",
            episode_number=1,
            duration_minutes=10,
            video_url="https://example.com/video1",
            learning_objectives=["Learn testing", "Understand models"],
            topics=["testing", "models"]
        )
        
        assert tutorial.title == "Test Tutorial"
        assert tutorial.episode_number == 1
        assert tutorial.difficulty_level == "beginner"  # default
        assert len(tutorial.learning_objectives) == 2
        
        # Test VideoSeries model
        series = VideoSeries(
            name="Test Series",
            description="A test series",
            category="testing",
            total_duration_minutes=60,
            videos=[tutorial]
        )
        
        assert series.name == "Test Series"
        assert series.category == "testing"
        assert len(series.videos) == 1
        
        # Test UserProgress model
        progress = UserProgress(
            user_id="test_user",
            tutorial_id=tutorial.id,
            series_id=series.id,
            watch_time_seconds=300,
            completion_percentage=50.0
        )
        
        assert progress.user_id == "test_user"
        assert progress.watch_time_seconds == 300
        assert progress.completion_percentage == 50.0
        
        # Test VideoAnalytics model
        analytics = VideoAnalytics(
            tutorial_id=tutorial.id,
            total_views=100,
            unique_viewers=75,
            completion_rate=85.5
        )
        
        assert analytics.tutorial_id == tutorial.id
        assert analytics.total_views == 100
        assert analytics.completion_rate == 85.5


if __name__ == "__main__":
    pytest.main([__file__])