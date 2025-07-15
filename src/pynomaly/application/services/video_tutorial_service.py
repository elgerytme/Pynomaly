"""
Video Tutorial Management Service

This service manages the video tutorial system, including content delivery,
progress tracking, and interactive features.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class VideoTutorial(BaseModel):
    """Represents a single video tutorial."""
    
    id: UUID = Field(default_factory=uuid4)
    title: str
    description: str
    series: str
    episode_number: int
    duration_minutes: int
    video_url: str
    transcript_url: Optional[str] = None
    captions_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    difficulty_level: str = Field(default="beginner")  # beginner, intermediate, advanced
    prerequisites: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    code_examples: List[str] = Field(default_factory=list)
    downloadable_resources: List[str] = Field(default_factory=list)
    quiz_questions: List[Dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_published: bool = Field(default=False)
    view_count: int = Field(default=0)
    average_rating: float = Field(default=0.0)
    completion_rate: float = Field(default=0.0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class VideoSeries(BaseModel):
    """Represents a video tutorial series."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    category: str
    total_duration_minutes: int
    difficulty_level: str = Field(default="beginner")
    prerequisites: List[str] = Field(default_factory=list)
    learning_outcomes: List[str] = Field(default_factory=list)
    videos: List[VideoTutorial] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_published: bool = Field(default=False)
    completion_certificate: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class UserProgress(BaseModel):
    """Tracks user progress through video tutorials."""
    
    user_id: str
    tutorial_id: UUID
    series_id: UUID
    watch_time_seconds: int = 0
    completion_percentage: float = 0.0
    last_watched_position: int = 0
    quiz_scores: List[float] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    bookmarks: List[int] = Field(default_factory=list)  # Timestamp bookmarks
    completed_at: Optional[datetime] = None
    certificate_earned: Optional[str] = None
    rating: Optional[float] = None
    feedback: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class VideoAnalytics(BaseModel):
    """Video analytics and engagement metrics."""
    
    tutorial_id: UUID
    total_views: int = 0
    unique_viewers: int = 0
    total_watch_time: int = 0  # in seconds
    average_watch_time: float = 0.0
    completion_rate: float = 0.0
    drop_off_points: List[int] = Field(default_factory=list)  # Timestamps where users drop off
    replay_segments: List[Tuple[int, int]] = Field(default_factory=list)  # Most replayed segments
    quiz_performance: Dict[str, float] = Field(default_factory=dict)
    device_stats: Dict[str, int] = Field(default_factory=dict)
    quality_preferences: Dict[str, int] = Field(default_factory=dict)
    loading_times: List[float] = Field(default_factory=list)
    error_events: List[Dict] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class VideoTutorialService:
    """Service for managing video tutorials and user progress."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage files
        self.series_file = self.storage_path / "video_series.json"
        self.tutorials_file = self.storage_path / "video_tutorials.json"
        self.progress_file = self.storage_path / "user_progress.json"
        self.analytics_file = self.storage_path / "video_analytics.json"
        
        # In-memory storage
        self.video_series: Dict[UUID, VideoSeries] = {}
        self.video_tutorials: Dict[UUID, VideoTutorial] = {}
        self.user_progress: Dict[str, List[UserProgress]] = {}
        self.video_analytics: Dict[UUID, VideoAnalytics] = {}
        
        # Load existing data
        self._load_data()
        
        # Initialize default content if empty
        if not self.video_series:
            self._initialize_default_content()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_data(self):
        """Load data from storage files."""
        try:
            # Load video series
            if self.series_file.exists():
                with open(self.series_file, 'r') as f:
                    series_data = json.load(f)
                    for series_id, series_info in series_data.items():
                        series_info['id'] = UUID(series_id)
                        # Convert video IDs back to UUIDs
                        for video in series_info.get('videos', []):
                            video['id'] = UUID(video['id'])
                        self.video_series[UUID(series_id)] = VideoSeries(**series_info)
            
            # Load video tutorials
            if self.tutorials_file.exists():
                with open(self.tutorials_file, 'r') as f:
                    tutorials_data = json.load(f)
                    for tutorial_id, tutorial_info in tutorials_data.items():
                        tutorial_info['id'] = UUID(tutorial_id)
                        self.video_tutorials[UUID(tutorial_id)] = VideoTutorial(**tutorial_info)
            
            # Load user progress
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    for user_id, progress_list in progress_data.items():
                        self.user_progress[user_id] = []
                        for progress_info in progress_list:
                            progress_info['tutorial_id'] = UUID(progress_info['tutorial_id'])
                            progress_info['series_id'] = UUID(progress_info['series_id'])
                            self.user_progress[user_id].append(UserProgress(**progress_info))
            
            # Load analytics
            if self.analytics_file.exists():
                with open(self.analytics_file, 'r') as f:
                    analytics_data = json.load(f)
                    for tutorial_id, analytics_info in analytics_data.items():
                        analytics_info['tutorial_id'] = UUID(tutorial_id)
                        self.video_analytics[UUID(tutorial_id)] = VideoAnalytics(**analytics_info)
        
        except Exception as e:
            self.logger.error(f"Error loading video tutorial data: {e}")
    
    def _save_data(self):
        """Save data to storage files."""
        try:
            # Save video series
            with open(self.series_file, 'w') as f:
                series_data = {}
                for series_id, series in self.video_series.items():
                    series_data[str(series_id)] = series.dict()
                json.dump(series_data, f, indent=2, default=str)
            
            # Save video tutorials
            with open(self.tutorials_file, 'w') as f:
                tutorials_data = {}
                for tutorial_id, tutorial in self.video_tutorials.items():
                    tutorials_data[str(tutorial_id)] = tutorial.dict()
                json.dump(tutorials_data, f, indent=2, default=str)
            
            # Save user progress
            with open(self.progress_file, 'w') as f:
                progress_data = {}
                for user_id, progress_list in self.user_progress.items():
                    progress_data[user_id] = [progress.dict() for progress in progress_list]
                json.dump(progress_data, f, indent=2, default=str)
            
            # Save analytics
            with open(self.analytics_file, 'w') as f:
                analytics_data = {}
                for tutorial_id, analytics in self.video_analytics.items():
                    analytics_data[str(tutorial_id)] = analytics.dict()
                json.dump(analytics_data, f, indent=2, default=str)
        
        except Exception as e:
            self.logger.error(f"Error saving video tutorial data: {e}")
    
    def _initialize_default_content(self):
        """Initialize default video tutorial content."""
        # Create default video series
        default_series = [
            {
                "name": "Pynomaly Quickstart",
                "description": "Perfect for new users getting started with anomaly detection",
                "category": "getting_started",
                "difficulty_level": "beginner",
                "videos": [
                    {
                        "title": "Installation & Setup",
                        "description": "Complete installation guide with environment setup",
                        "duration_minutes": 5,
                        "video_url": "https://tutorials.pynomaly.com/quickstart/installation",
                        "learning_objectives": [
                            "Install Pynomaly on your system",
                            "Set up development environment",
                            "Verify installation"
                        ],
                        "topics": ["installation", "setup", "environment"]
                    },
                    {
                        "title": "First Anomaly Detection",
                        "description": "Your first anomaly detection in 10 minutes",
                        "duration_minutes": 10,
                        "video_url": "https://tutorials.pynomaly.com/quickstart/first-detection",
                        "learning_objectives": [
                            "Load sample data",
                            "Run basic anomaly detection",
                            "Interpret results"
                        ],
                        "topics": ["anomaly detection", "data loading", "basic usage"]
                    },
                    {
                        "title": "Understanding Results",
                        "description": "Interpreting detection results and visualizations",
                        "duration_minutes": 8,
                        "video_url": "https://tutorials.pynomaly.com/quickstart/understanding-results",
                        "learning_objectives": [
                            "Interpret anomaly scores",
                            "Understand visualizations",
                            "Evaluate detection quality"
                        ],
                        "topics": ["results", "visualization", "interpretation"]
                    }
                ]
            },
            {
                "name": "Algorithm Masterclass",
                "description": "Deep dive into anomaly detection algorithms",
                "category": "algorithms",
                "difficulty_level": "intermediate",
                "videos": [
                    {
                        "title": "Isolation Forest Explained",
                        "description": "How Isolation Forest works with visual examples",
                        "duration_minutes": 12,
                        "video_url": "https://tutorials.pynomaly.com/algorithms/isolation-forest",
                        "learning_objectives": [
                            "Understand Isolation Forest algorithm",
                            "Learn parameter tuning",
                            "Apply to real datasets"
                        ],
                        "topics": ["isolation forest", "tree algorithms", "parameter tuning"]
                    },
                    {
                        "title": "Local Outlier Factor Deep Dive",
                        "description": "Understanding LOF with real-world examples",
                        "duration_minutes": 10,
                        "video_url": "https://tutorials.pynomaly.com/algorithms/lof",
                        "learning_objectives": [
                            "Master LOF algorithm",
                            "Understand density-based detection",
                            "Choose optimal parameters"
                        ],
                        "topics": ["LOF", "density-based", "neighborhood analysis"]
                    }
                ]
            },
            {
                "name": "Industry Applications",
                "description": "Real-world use cases and implementations",
                "category": "applications",
                "difficulty_level": "advanced",
                "videos": [
                    {
                        "title": "Financial Fraud Detection",
                        "description": "Complete fraud detection system implementation",
                        "duration_minutes": 25,
                        "video_url": "https://tutorials.pynomaly.com/applications/fraud-detection",
                        "learning_objectives": [
                            "Build fraud detection system",
                            "Handle financial data",
                            "Implement real-time detection"
                        ],
                        "topics": ["fraud detection", "financial data", "real-time processing"]
                    },
                    {
                        "title": "Industrial IoT Monitoring",
                        "description": "Monitoring industrial sensors and equipment",
                        "duration_minutes": 20,
                        "video_url": "https://tutorials.pynomaly.com/applications/iot-monitoring",
                        "learning_objectives": [
                            "Monitor IoT sensors",
                            "Detect equipment failures",
                            "Set up alerting systems"
                        ],
                        "topics": ["IoT", "sensor monitoring", "predictive maintenance"]
                    }
                ]
            }
        ]
        
        # Create video series and tutorials
        for series_data in default_series:
            series_id = uuid4()
            videos = []
            
            for i, video_data in enumerate(series_data["videos"]):
                video_id = uuid4()
                video = VideoTutorial(
                    id=video_id,
                    title=video_data["title"],
                    description=video_data["description"],
                    series=series_data["name"],
                    episode_number=i + 1,
                    duration_minutes=video_data["duration_minutes"],
                    video_url=video_data["video_url"],
                    learning_objectives=video_data["learning_objectives"],
                    topics=video_data["topics"],
                    is_published=True
                )
                videos.append(video)
                self.video_tutorials[video_id] = video
                
                # Initialize analytics
                self.video_analytics[video_id] = VideoAnalytics(tutorial_id=video_id)
            
            # Create series
            total_duration = sum(video.duration_minutes for video in videos)
            series = VideoSeries(
                id=series_id,
                name=series_data["name"],
                description=series_data["description"],
                category=series_data["category"],
                total_duration_minutes=total_duration,
                difficulty_level=series_data["difficulty_level"],
                videos=videos,
                is_published=True
            )
            
            self.video_series[series_id] = series
        
        # Save initialized data
        self._save_data()
    
    async def get_video_series(self, published_only: bool = True) -> List[VideoSeries]:
        """Get all video series."""
        series_list = list(self.video_series.values())
        if published_only:
            series_list = [s for s in series_list if s.is_published]
        return sorted(series_list, key=lambda x: x.created_at)
    
    async def get_video_series_by_id(self, series_id: UUID) -> Optional[VideoSeries]:
        """Get a specific video series by ID."""
        return self.video_series.get(series_id)
    
    async def get_video_tutorial(self, tutorial_id: UUID) -> Optional[VideoTutorial]:
        """Get a specific video tutorial by ID."""
        return self.video_tutorials.get(tutorial_id)
    
    async def get_tutorials_by_series(self, series_id: UUID) -> List[VideoTutorial]:
        """Get all tutorials in a specific series."""
        series = self.video_series.get(series_id)
        if not series:
            return []
        return sorted(series.videos, key=lambda x: x.episode_number)
    
    async def search_tutorials(self, query: str, category: Optional[str] = None, 
                             difficulty: Optional[str] = None) -> List[VideoTutorial]:
        """Search for tutorials based on query and filters."""
        results = []
        query_lower = query.lower()
        
        for tutorial in self.video_tutorials.values():
            # Skip unpublished tutorials
            if not tutorial.is_published:
                continue
            
            # Text search
            if (query_lower in tutorial.title.lower() or
                query_lower in tutorial.description.lower() or
                any(query_lower in topic.lower() for topic in tutorial.topics)):
                
                # Apply filters
                if category:
                    series = next((s for s in self.video_series.values() 
                                 if any(v.id == tutorial.id for v in s.videos)), None)
                    if not series or series.category != category:
                        continue
                
                if difficulty and tutorial.difficulty_level != difficulty:
                    continue
                
                results.append(tutorial)
        
        return sorted(results, key=lambda x: x.created_at)
    
    async def track_video_view(self, tutorial_id: UUID, user_id: str, 
                             watch_time: int, device_type: str = "desktop"):
        """Track video view and engagement."""
        # Update video analytics
        if tutorial_id in self.video_analytics:
            analytics = self.video_analytics[tutorial_id]
            analytics.total_views += 1
            analytics.total_watch_time += watch_time
            analytics.device_stats[device_type] = analytics.device_stats.get(device_type, 0) + 1
            analytics.last_updated = datetime.utcnow()
        
        # Update user progress
        if user_id not in self.user_progress:
            self.user_progress[user_id] = []
        
        # Find or create user progress for this tutorial
        progress = None
        for p in self.user_progress[user_id]:
            if p.tutorial_id == tutorial_id:
                progress = p
                break
        
        if not progress:
            # Find series ID for this tutorial
            series_id = None
            for series in self.video_series.values():
                if any(v.id == tutorial_id for v in series.videos):
                    series_id = series.id
                    break
            
            if series_id:
                progress = UserProgress(
                    user_id=user_id,
                    tutorial_id=tutorial_id,
                    series_id=series_id
                )
                self.user_progress[user_id].append(progress)
        
        if progress:
            progress.watch_time_seconds += watch_time
            progress.last_updated = datetime.utcnow()
            
            # Calculate completion percentage
            tutorial = self.video_tutorials.get(tutorial_id)
            if tutorial:
                total_duration = tutorial.duration_minutes * 60
                progress.completion_percentage = min(100.0, 
                                                   (progress.watch_time_seconds / total_duration) * 100)
                
                # Mark as completed if watched 80% or more
                if progress.completion_percentage >= 80 and not progress.completed_at:
                    progress.completed_at = datetime.utcnow()
        
        # Save updated data
        self._save_data()
    
    async def get_user_progress(self, user_id: str) -> List[UserProgress]:
        """Get user's progress across all tutorials."""
        return self.user_progress.get(user_id, [])
    
    async def get_user_progress_for_series(self, user_id: str, series_id: UUID) -> List[UserProgress]:
        """Get user's progress for a specific series."""
        user_progress = self.user_progress.get(user_id, [])
        return [p for p in user_progress if p.series_id == series_id]
    
    async def submit_quiz_score(self, user_id: str, tutorial_id: UUID, score: float):
        """Submit quiz score for a tutorial."""
        if user_id not in self.user_progress:
            return
        
        for progress in self.user_progress[user_id]:
            if progress.tutorial_id == tutorial_id:
                progress.quiz_scores.append(score)
                progress.last_updated = datetime.utcnow()
                break
        
        self._save_data()
    
    async def add_bookmark(self, user_id: str, tutorial_id: UUID, timestamp: int):
        """Add a bookmark to a tutorial."""
        if user_id not in self.user_progress:
            return
        
        for progress in self.user_progress[user_id]:
            if progress.tutorial_id == tutorial_id:
                if timestamp not in progress.bookmarks:
                    progress.bookmarks.append(timestamp)
                    progress.bookmarks.sort()
                    progress.last_updated = datetime.utcnow()
                break
        
        self._save_data()
    
    async def rate_tutorial(self, user_id: str, tutorial_id: UUID, rating: float, 
                          feedback: Optional[str] = None):
        """Rate a tutorial and provide feedback."""
        if user_id not in self.user_progress:
            return
        
        for progress in self.user_progress[user_id]:
            if progress.tutorial_id == tutorial_id:
                progress.rating = rating
                if feedback:
                    progress.feedback = feedback
                progress.last_updated = datetime.utcnow()
                break
        
        # Update tutorial average rating
        if tutorial_id in self.video_tutorials:
            tutorial = self.video_tutorials[tutorial_id]
            all_ratings = []
            for user_progress_list in self.user_progress.values():
                for progress in user_progress_list:
                    if progress.tutorial_id == tutorial_id and progress.rating:
                        all_ratings.append(progress.rating)
            
            if all_ratings:
                tutorial.average_rating = sum(all_ratings) / len(all_ratings)
        
        self._save_data()
    
    async def get_tutorial_analytics(self, tutorial_id: UUID) -> Optional[VideoAnalytics]:
        """Get analytics for a specific tutorial."""
        return self.video_analytics.get(tutorial_id)
    
    async def get_recommended_tutorials(self, user_id: str, limit: int = 5) -> List[VideoTutorial]:
        """Get recommended tutorials for a user based on their progress."""
        user_progress = self.user_progress.get(user_id, [])
        completed_tutorials = {p.tutorial_id for p in user_progress if p.completed_at}
        
        # Get user's skill level based on completed tutorials
        completed_count = len(completed_tutorials)
        if completed_count == 0:
            target_difficulty = "beginner"
        elif completed_count < 5:
            target_difficulty = "intermediate"
        else:
            target_difficulty = "advanced"
        
        # Find tutorials of appropriate difficulty that user hasn't completed
        recommendations = []
        for tutorial in self.video_tutorials.values():
            if (tutorial.is_published and 
                tutorial.id not in completed_tutorials and
                tutorial.difficulty_level == target_difficulty):
                recommendations.append(tutorial)
        
        # Sort by popularity (view count) and rating
        recommendations.sort(key=lambda x: (x.view_count, x.average_rating), reverse=True)
        
        return recommendations[:limit]
    
    async def generate_certificate(self, user_id: str, series_id: UUID) -> Optional[str]:
        """Generate a completion certificate for a series."""
        series = self.video_series.get(series_id)
        if not series:
            return None
        
        user_progress = self.user_progress.get(user_id, [])
        series_progress = [p for p in user_progress if p.series_id == series_id]
        
        # Check if all tutorials in the series are completed
        completed_tutorials = {p.tutorial_id for p in series_progress if p.completed_at}
        series_tutorial_ids = {v.id for v in series.videos}
        
        if completed_tutorials >= series_tutorial_ids:
            # Generate certificate
            certificate_id = f"CERT-{series_id}-{user_id}-{datetime.utcnow().strftime('%Y%m%d')}"
            
            # Mark certificate as earned for all tutorials in series
            for progress in series_progress:
                progress.certificate_earned = certificate_id
            
            self._save_data()
            return certificate_id
        
        return None
    
    async def get_learning_analytics(self, user_id: Optional[str] = None) -> Dict:
        """Get learning analytics for a user or system-wide."""
        if user_id:
            # User-specific analytics
            user_progress = self.user_progress.get(user_id, [])
            
            completed_tutorials = [p for p in user_progress if p.completed_at]
            total_watch_time = sum(p.watch_time_seconds for p in user_progress)
            
            return {
                "user_id": user_id,
                "tutorials_started": len(user_progress),
                "tutorials_completed": len(completed_tutorials),
                "total_watch_time_minutes": total_watch_time / 60,
                "average_completion_rate": sum(p.completion_percentage for p in user_progress) / len(user_progress) if user_progress else 0,
                "certificates_earned": len([p for p in user_progress if p.certificate_earned]),
                "favorite_topics": self._get_favorite_topics(user_progress)
            }
        else:
            # System-wide analytics
            total_tutorials = len(self.video_tutorials)
            total_series = len(self.video_series)
            total_users = len(self.user_progress)
            
            all_progress = []
            for user_progress_list in self.user_progress.values():
                all_progress.extend(user_progress_list)
            
            return {
                "total_tutorials": total_tutorials,
                "total_series": total_series,
                "total_users": total_users,
                "total_views": sum(analytics.total_views for analytics in self.video_analytics.values()),
                "total_watch_time_hours": sum(analytics.total_watch_time for analytics in self.video_analytics.values()) / 3600,
                "average_completion_rate": sum(p.completion_percentage for p in all_progress) / len(all_progress) if all_progress else 0,
                "popular_tutorials": self._get_popular_tutorials(),
                "trending_topics": self._get_trending_topics()
            }
    
    def _get_favorite_topics(self, user_progress: List[UserProgress]) -> List[str]:
        """Get user's favorite topics based on completed tutorials."""
        topic_counts = {}
        
        for progress in user_progress:
            if progress.completed_at:
                tutorial = self.video_tutorials.get(progress.tutorial_id)
                if tutorial:
                    for topic in tutorial.topics:
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:5]
    
    def _get_popular_tutorials(self) -> List[Dict]:
        """Get most popular tutorials by view count."""
        tutorials = [(t, self.video_analytics.get(t.id)) for t in self.video_tutorials.values()]
        tutorials = [(t, a) for t, a in tutorials if a and t.is_published]
        tutorials.sort(key=lambda x: x[1].total_views, reverse=True)
        
        return [
            {
                "id": str(tutorial.id),
                "title": tutorial.title,
                "views": analytics.total_views,
                "rating": tutorial.average_rating
            }
            for tutorial, analytics in tutorials[:10]
        ]
    
    def _get_trending_topics(self) -> List[str]:
        """Get trending topics based on recent activity."""
        topic_counts = {}
        
        for tutorial in self.video_tutorials.values():
            if tutorial.is_published:
                analytics = self.video_analytics.get(tutorial.id)
                if analytics:
                    for topic in tutorial.topics:
                        topic_counts[topic] = topic_counts.get(topic, 0) + analytics.total_views
        
        return sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:10]