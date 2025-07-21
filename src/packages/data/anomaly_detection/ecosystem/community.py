"""
Community Features and Collaboration Tools for Pynomaly Detection
================================================================

Comprehensive community platform providing:
- User profiles and community management
- Project sharing and collaboration features
- Discussion forums and knowledge sharing
- Code review and peer feedback systems
- Community challenges and competitions
- Mentorship and learning programs
- Resource sharing and documentation wiki
"""

import logging
import json
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User role enumeration."""
    MEMBER = "member"
    CONTRIBUTOR = "contributor"
    MAINTAINER = "maintainer"
    ADMIN = "admin"
    MODERATOR = "moderator"

class ProjectVisibility(Enum):
    """Project visibility enumeration."""
    PUBLIC = "public"
    PRIVATE = "private"
    ORGANIZATION = "organization"
    COMMUNITY = "community"

class PostType(Enum):
    """Forum post type enumeration."""
    QUESTION = "question"
    DISCUSSION = "discussion"
    ANNOUNCEMENT = "announcement"
    TUTORIAL = "tutorial"
    SHOWCASE = "showcase"
    HELP_REQUEST = "help_request"

class CollaborationStatus(Enum):
    """Collaboration status enumeration."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"

@dataclass
class UserProfile:
    """User profile information."""
    user_id: str
    username: str
    email: str
    full_name: str
    bio: str = ""
    location: str = ""
    organization: str = ""
    role: UserRole = UserRole.MEMBER
    skills: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    reputation_score: int = 0
    total_contributions: int = 0
    join_date: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    profile_image: Optional[str] = None
    social_links: Dict[str, str] = field(default_factory=dict)
    is_verified: bool = False
    is_active: bool = True

@dataclass
class Project:
    """Community project information."""
    project_id: str
    name: str
    description: str
    owner_id: str
    visibility: ProjectVisibility
    repository_url: Optional[str] = None
    documentation_url: Optional[str] = None
    demo_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    collaborators: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    star_count: int = 0
    fork_count: int = 0
    view_count: int = 0
    status: CollaborationStatus = CollaborationStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ForumPost:
    """Forum post information."""
    post_id: str
    title: str
    content: str
    author_id: str
    post_type: PostType
    category: str
    tags: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    view_count: int = 0
    upvotes: int = 0
    downvotes: int = 0
    reply_count: int = 0
    is_pinned: bool = False
    is_locked: bool = False
    is_solved: bool = False
    parent_id: Optional[str] = None  # For replies
    attachments: List[str] = field(default_factory=list)

@dataclass
class CodeReview:
    """Code review information."""
    review_id: str
    project_id: str
    title: str
    description: str
    author_id: str
    code_files: List[Dict[str, str]] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_review, approved, rejected
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    comments: List[Dict[str, Any]] = field(default_factory=list)
    overall_rating: Optional[float] = None

@dataclass
class Challenge:
    """Community challenge information."""
    challenge_id: str
    title: str
    description: str
    difficulty: str  # beginner, intermediate, advanced
    category: str
    start_date: datetime
    end_date: datetime
    prize_description: str = ""
    rules: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    submissions: List[Dict[str, Any]] = field(default_factory=list)
    organizer_id: str = ""
    is_active: bool = True

class CommunityHub:
    """Central community management system."""
    
    def __init__(self):
        """Initialize community hub."""
        # User management
        self.users: Dict[str, UserProfile] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Project management
        self.projects: Dict[str, Project] = {}
        self.project_collaborations: Dict[str, List[str]] = {}
        
        # Forum system
        self.forum_posts: Dict[str, ForumPost] = {}
        self.forum_categories: Dict[str, Dict[str, Any]] = {}
        
        # Code review system
        self.code_reviews: Dict[str, CodeReview] = {}
        
        # Challenge system
        self.challenges: Dict[str, Challenge] = {}
        
        # Community metrics
        self.community_stats = {
            'total_users': 0,
            'active_users_30d': 0,
            'total_projects': 0,
            'total_posts': 0,
            'total_reviews': 0,
            'total_challenges': 0
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Initialize default forum categories
        self._initialize_forum_categories()
        
        logger.info("Community Hub initialized")
    
    def register_user(self, user_data: Dict[str, Any]) -> Optional[UserProfile]:
        """Register new user.
        
        Args:
            user_data: User registration data
            
        Returns:
            User profile if successful, None otherwise
        """
        try:
            with self.lock:
                # Validate user data
                if not self._validate_user_data(user_data):
                    return None
                
                # Check if username/email already exists
                username = user_data['username']
                email = user_data['email']
                
                for user in self.users.values():
                    if user.username == username:
                        logger.error(f"Username already exists: {username}")
                        return None
                    if user.email == email:
                        logger.error(f"Email already exists: {email}")
                        return None
                
                # Create user profile
                user_profile = UserProfile(
                    user_id=str(uuid.uuid4()),
                    username=username,
                    email=email,
                    full_name=user_data.get('full_name', ''),
                    bio=user_data.get('bio', ''),
                    location=user_data.get('location', ''),
                    organization=user_data.get('organization', ''),
                    skills=user_data.get('skills', []),
                    interests=user_data.get('interests', [])
                )
                
                # Store user
                self.users[user_profile.user_id] = user_profile
                self.community_stats['total_users'] += 1
                
                logger.info(f"User registered: {username}")
                return user_profile
                
        except Exception as e:
            logger.error(f"User registration failed: {e}")
            return None
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile or None
        """
        return self.users.get(user_id)
    
    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile.
        
        Args:
            user_id: User identifier
            updates: Profile updates
            
        Returns:
            True if update successful
        """
        try:
            with self.lock:
                user = self.users.get(user_id)
                if not user:
                    return False
                
                # Update allowed fields
                allowed_fields = {
                    'full_name', 'bio', 'location', 'organization',
                    'skills', 'interests', 'social_links', 'profile_image'
                }
                
                for field, value in updates.items():
                    if field in allowed_fields and hasattr(user, field):
                        setattr(user, field, value)
                
                user.last_active = datetime.now()
                
                logger.info(f"User profile updated: {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"User profile update failed: {e}")
            return False
    
    def create_project(self, project_data: Dict[str, Any], owner_id: str) -> Optional[Project]:
        """Create new community project.
        
        Args:
            project_data: Project information
            owner_id: Project owner user ID
            
        Returns:
            Project instance if successful, None otherwise
        """
        try:
            with self.lock:
                # Validate owner exists
                if owner_id not in self.users:
                    logger.error(f"Project owner not found: {owner_id}")
                    return None
                
                # Create project
                project = Project(
                    project_id=str(uuid.uuid4()),
                    name=project_data['name'],
                    description=project_data.get('description', ''),
                    owner_id=owner_id,
                    visibility=ProjectVisibility(project_data.get('visibility', 'public')),
                    repository_url=project_data.get('repository_url'),
                    documentation_url=project_data.get('documentation_url'),
                    demo_url=project_data.get('demo_url'),
                    tags=project_data.get('tags', [])
                )
                
                # Store project
                self.projects[project.project_id] = project
                self.project_collaborations[project.project_id] = [owner_id]
                self.community_stats['total_projects'] += 1
                
                logger.info(f"Project created: {project.name}")
                return project
                
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            return None
    
    def add_collaborator(self, project_id: str, user_id: str, inviter_id: str) -> bool:
        """Add collaborator to project.
        
        Args:
            project_id: Project identifier
            user_id: User to add as collaborator
            inviter_id: User sending invitation
            
        Returns:
            True if collaboration added successfully
        """
        try:
            with self.lock:
                project = self.projects.get(project_id)
                if not project:
                    return False
                
                # Check permissions
                if not self._can_manage_project(project, inviter_id):
                    return False
                
                # Add collaborator
                if user_id not in project.collaborators:
                    project.collaborators.append(user_id)
                    
                    if project_id in self.project_collaborations:
                        self.project_collaborations[project_id].append(user_id)
                    
                    project.last_updated = datetime.now()
                    
                    logger.info(f"Collaborator added to project {project_id}: {user_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Add collaborator failed: {e}")
            return False
    
    def create_forum_post(self, post_data: Dict[str, Any], author_id: str) -> Optional[ForumPost]:
        """Create forum post.
        
        Args:
            post_data: Post information
            author_id: Post author user ID
            
        Returns:
            Forum post if successful, None otherwise
        """
        try:
            with self.lock:
                # Validate author exists
                if author_id not in self.users:
                    return None
                
                # Create post
                post = ForumPost(
                    post_id=str(uuid.uuid4()),
                    title=post_data['title'],
                    content=post_data['content'],
                    author_id=author_id,
                    post_type=PostType(post_data.get('post_type', 'discussion')),
                    category=post_data.get('category', 'general'),
                    tags=post_data.get('tags', []),
                    parent_id=post_data.get('parent_id')
                )
                
                # Store post
                self.forum_posts[post.post_id] = post
                self.community_stats['total_posts'] += 1
                
                # Update user contribution
                user = self.users[author_id]
                user.total_contributions += 1
                user.last_active = datetime.now()
                
                logger.info(f"Forum post created: {post.title}")
                return post
                
        except Exception as e:
            logger.error(f"Forum post creation failed: {e}")
            return None
    
    def submit_code_review(self, review_data: Dict[str, Any], author_id: str) -> Optional[CodeReview]:
        """Submit code for review.
        
        Args:
            review_data: Code review information
            author_id: Review author user ID
            
        Returns:
            Code review if successful, None otherwise
        """
        try:
            with self.lock:
                # Validate author and project
                if author_id not in self.users:
                    return None
                
                project_id = review_data.get('project_id')
                if project_id and project_id not in self.projects:
                    return None
                
                # Create code review
                review = CodeReview(
                    review_id=str(uuid.uuid4()),
                    project_id=project_id or '',
                    title=review_data['title'],
                    description=review_data.get('description', ''),
                    author_id=author_id,
                    code_files=review_data.get('code_files', []),
                    reviewers=review_data.get('reviewers', [])
                )
                
                # Store review
                self.code_reviews[review.review_id] = review
                self.community_stats['total_reviews'] += 1
                
                # Update user contribution
                user = self.users[author_id]
                user.total_contributions += 1
                user.last_active = datetime.now()
                
                logger.info(f"Code review submitted: {review.title}")
                return review
                
        except Exception as e:
            logger.error(f"Code review submission failed: {e}")
            return None
    
    def create_challenge(self, challenge_data: Dict[str, Any], organizer_id: str) -> Optional[Challenge]:
        """Create community challenge.
        
        Args:
            challenge_data: Challenge information
            organizer_id: Challenge organizer user ID
            
        Returns:
            Challenge if successful, None otherwise
        """
        try:
            with self.lock:
                # Validate organizer exists and has permissions
                if organizer_id not in self.users:
                    return None
                
                organizer = self.users[organizer_id]
                if organizer.role not in [UserRole.MAINTAINER, UserRole.ADMIN]:
                    logger.error(f"Insufficient permissions to create challenge: {organizer_id}")
                    return None
                
                # Create challenge
                challenge = Challenge(
                    challenge_id=str(uuid.uuid4()),
                    title=challenge_data['title'],
                    description=challenge_data['description'],
                    difficulty=challenge_data.get('difficulty', 'intermediate'),
                    category=challenge_data.get('category', 'general'),
                    start_date=datetime.fromisoformat(challenge_data['start_date']),
                    end_date=datetime.fromisoformat(challenge_data['end_date']),
                    prize_description=challenge_data.get('prize_description', ''),
                    rules=challenge_data.get('rules', []),
                    evaluation_criteria=challenge_data.get('evaluation_criteria', []),
                    organizer_id=organizer_id
                )
                
                # Store challenge
                self.challenges[challenge.challenge_id] = challenge
                self.community_stats['total_challenges'] += 1
                
                logger.info(f"Challenge created: {challenge.title}")
                return challenge
                
        except Exception as e:
            logger.error(f"Challenge creation failed: {e}")
            return None
    
    def search_projects(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Project]:
        """Search community projects.
        
        Args:
            query: Search query
            filters: Optional search filters
            
        Returns:
            List of matching projects
        """
        try:
            results = []
            query_lower = query.lower()
            filters = filters or {}
            
            for project in self.projects.values():
                # Visibility check
                if filters.get('visibility'):
                    if project.visibility != ProjectVisibility(filters['visibility']):
                        continue
                
                # Text search
                searchable_text = f"{project.name} {project.description} {' '.join(project.tags)}".lower()
                
                if query_lower in searchable_text:
                    results.append(project)
            
            # Sort by relevance (star count, view count)
            results.sort(key=lambda p: (p.star_count, p.view_count), reverse=True)
            
            return results[:50]  # Limit results
            
        except Exception as e:
            logger.error(f"Project search failed: {e}")
            return []
    
    def search_forum_posts(self, query: str, category: Optional[str] = None) -> List[ForumPost]:
        """Search forum posts.
        
        Args:
            query: Search query
            category: Optional category filter
            
        Returns:
            List of matching posts
        """
        try:
            results = []
            query_lower = query.lower()
            
            for post in self.forum_posts.values():
                # Category filter
                if category and post.category != category:
                    continue
                
                # Text search
                searchable_text = f"{post.title} {post.content} {' '.join(post.tags)}".lower()
                
                if query_lower in searchable_text:
                    results.append(post)
            
            # Sort by relevance (upvotes, view count, recency)
            results.sort(key=lambda p: (p.upvotes, p.view_count, p.created_date), reverse=True)
            
            return results[:50]  # Limit results
            
        except Exception as e:
            logger.error(f"Forum search failed: {e}")
            return []
    
    def get_user_activity_feed(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user activity feed.
        
        Args:
            user_id: User identifier
            limit: Maximum number of activities
            
        Returns:
            List of user activities
        """
        try:
            activities = []
            
            # Get user's projects
            user_projects = [p for p in self.projects.values() if p.owner_id == user_id]
            for project in user_projects:
                activities.append({
                    'type': 'project_created',
                    'timestamp': project.created_date,
                    'data': {
                        'project_id': project.project_id,
                        'project_name': project.name
                    }
                })
            
            # Get user's forum posts
            user_posts = [p for p in self.forum_posts.values() if p.author_id == user_id]
            for post in user_posts:
                activities.append({
                    'type': 'forum_post_created',
                    'timestamp': post.created_date,
                    'data': {
                        'post_id': post.post_id,
                        'post_title': post.title,
                        'post_type': post.post_type.value
                    }
                })
            
            # Get user's code reviews
            user_reviews = [r for r in self.code_reviews.values() if r.author_id == user_id]
            for review in user_reviews:
                activities.append({
                    'type': 'code_review_submitted',
                    'timestamp': review.created_date,
                    'data': {
                        'review_id': review.review_id,
                        'review_title': review.title
                    }
                })
            
            # Sort by timestamp and limit
            activities.sort(key=lambda a: a['timestamp'], reverse=True)
            
            return activities[:limit]
            
        except Exception as e:
            logger.error(f"Get user activity feed failed: {e}")
            return []
    
    def get_community_leaderboard(self, metric: str = 'reputation', limit: int = 10) -> List[Dict[str, Any]]:
        """Get community leaderboard.
        
        Args:
            metric: Ranking metric (reputation, contributions)
            limit: Number of top users to return
            
        Returns:
            List of top users
        """
        try:
            users = list(self.users.values())
            
            # Sort by metric
            if metric == 'reputation':
                users.sort(key=lambda u: u.reputation_score, reverse=True)
            elif metric == 'contributions':
                users.sort(key=lambda u: u.total_contributions, reverse=True)
            else:
                return []
            
            # Build leaderboard
            leaderboard = []
            for i, user in enumerate(users[:limit]):
                leaderboard.append({
                    'rank': i + 1,
                    'user_id': user.user_id,
                    'username': user.username,
                    'full_name': user.full_name,
                    'score': user.reputation_score if metric == 'reputation' else user.total_contributions,
                    'profile_image': user.profile_image
                })
            
            return leaderboard
            
        except Exception as e:
            logger.error(f"Get community leaderboard failed: {e}")
            return []
    
    def get_community_statistics(self) -> Dict[str, Any]:
        """Get community statistics.
        
        Returns:
            Community statistics dictionary
        """
        try:
            with self.lock:
                stats = self.community_stats.copy()
                
                # Calculate additional metrics
                now = datetime.now()
                thirty_days_ago = now - timedelta(days=30)
                
                # Active users in last 30 days
                active_users = sum(1 for user in self.users.values() 
                                 if user.last_active > thirty_days_ago)
                stats['active_users_30d'] = active_users
                
                # Top categories
                category_counts = {}
                for post in self.forum_posts.values():
                    category_counts[post.category] = category_counts.get(post.category, 0) + 1
                stats['top_categories'] = sorted(category_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:5]
                
                # Recent activity
                stats['recent_users'] = len([u for u in self.users.values() 
                                           if (now - u.join_date).days <= 7])
                stats['recent_projects'] = len([p for p in self.projects.values() 
                                              if (now - p.created_date).days <= 7])
                stats['recent_posts'] = len([p for p in self.forum_posts.values() 
                                           if (now - p.created_date).days <= 7])
                
                return stats
                
        except Exception as e:
            logger.error(f"Get community statistics failed: {e}")
            return self.community_stats.copy()
    
    def _validate_user_data(self, user_data: Dict[str, Any]) -> bool:
        """Validate user registration data."""
        required_fields = ['username', 'email', 'full_name']
        
        for field in required_fields:
            if not user_data.get(field):
                logger.error(f"Required field missing: {field}")
                return False
        
        # Basic email validation
        email = user_data['email']
        if '@' not in email or '.' not in email.split('@')[-1]:
            logger.error("Invalid email format")
            return False
        
        return True
    
    def _can_manage_project(self, project: Project, user_id: str) -> bool:
        """Check if user can manage project."""
        user = self.users.get(user_id)
        if not user:
            return False
        
        # Owner can always manage
        if project.owner_id == user_id:
            return True
        
        # Admin/maintainer can manage
        if user.role in [UserRole.ADMIN, UserRole.MAINTAINER]:
            return True
        
        # Collaborators can manage if project allows
        if user_id in project.collaborators:
            return True
        
        return False
    
    def _initialize_forum_categories(self):
        """Initialize default forum categories."""
        default_categories = {
            'general': {
                'name': 'General Discussion',
                'description': 'General community discussions'
            },
            'help': {
                'name': 'Help & Support',
                'description': 'Get help with anomaly detection'
            },
            'showcase': {
                'name': 'Showcase',
                'description': 'Show off your projects and achievements'
            },
            'tutorials': {
                'name': 'Tutorials & Guides',
                'description': 'Educational content and how-tos'
            },
            'announcements': {
                'name': 'Announcements',
                'description': 'Official announcements and news'
            },
            'feedback': {
                'name': 'Feedback & Suggestions',
                'description': 'Share feedback and improvement ideas'
            }
        }
        
        self.forum_categories.update(default_categories)


class CollaborationTools:
    """Advanced collaboration tools for community projects."""
    
    def __init__(self, community_hub: CommunityHub):
        """Initialize collaboration tools.
        
        Args:
            community_hub: Community hub instance
        """
        self.community_hub = community_hub
        
        # Real-time collaboration
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.collaborative_documents: Dict[str, Dict[str, Any]] = {}
        
        # Review workflows
        self.review_workflows: Dict[str, List[Dict[str, Any]]] = {}
        
        # Mentorship programs
        self.mentorship_pairs: Dict[str, Dict[str, Any]] = {}
        self.learning_paths: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Collaboration Tools initialized")
    
    def start_collaboration_session(self, project_id: str, session_data: Dict[str, Any]) -> Optional[str]:
        """Start real-time collaboration session.
        
        Args:
            project_id: Project identifier
            session_data: Session configuration
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            session_id = str(uuid.uuid4())
            
            session = {
                'session_id': session_id,
                'project_id': project_id,
                'host_id': session_data['host_id'],
                'participants': [session_data['host_id']],
                'session_type': session_data.get('type', 'code_review'),
                'started_at': datetime.now(),
                'status': 'active',
                'shared_documents': [],
                'chat_messages': [],
                'screen_sharing': False
            }
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Collaboration session started: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Start collaboration session failed: {e}")
            return None
    
    def join_collaboration_session(self, session_id: str, user_id: str) -> bool:
        """Join collaboration session.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            True if join successful
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session or session['status'] != 'active':
                return False
            
            if user_id not in session['participants']:
                session['participants'].append(user_id)
                
                # Add join message
                session['chat_messages'].append({
                    'timestamp': datetime.now(),
                    'user_id': 'system',
                    'message': f"User {user_id} joined the session",
                    'type': 'system'
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Join collaboration session failed: {e}")
            return False
    
    def create_shared_document(self, session_id: str, document_data: Dict[str, Any]) -> Optional[str]:
        """Create shared document for collaboration.
        
        Args:
            session_id: Session identifier
            document_data: Document information
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            document_id = str(uuid.uuid4())
            
            document = {
                'document_id': document_id,
                'session_id': session_id,
                'title': document_data['title'],
                'content': document_data.get('content', ''),
                'type': document_data.get('type', 'text'),
                'created_at': datetime.now(),
                'last_modified': datetime.now(),
                'editors': [],
                'version_history': []
            }
            
            self.collaborative_documents[document_id] = document
            
            # Add to session
            session = self.active_sessions.get(session_id)
            if session:
                session['shared_documents'].append(document_id)
            
            logger.info(f"Shared document created: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Create shared document failed: {e}")
            return None
    
    def setup_mentorship(self, mentor_id: str, mentee_id: str, 
                        program_data: Dict[str, Any]) -> Optional[str]:
        """Set up mentorship relationship.
        
        Args:
            mentor_id: Mentor user ID
            mentee_id: Mentee user ID
            program_data: Mentorship program data
            
        Returns:
            Mentorship ID if successful, None otherwise
        """
        try:
            # Validate users exist
            mentor = self.community_hub.get_user_profile(mentor_id)
            mentee = self.community_hub.get_user_profile(mentee_id)
            
            if not mentor or not mentee:
                return None
            
            # Check mentor eligibility
            if mentor.role not in [UserRole.MAINTAINER, UserRole.ADMIN] and mentor.reputation_score < 100:
                logger.error(f"User not eligible to be mentor: {mentor_id}")
                return None
            
            mentorship_id = str(uuid.uuid4())
            
            mentorship = {
                'mentorship_id': mentorship_id,
                'mentor_id': mentor_id,
                'mentee_id': mentee_id,
                'program_type': program_data.get('type', 'general'),
                'duration_weeks': program_data.get('duration_weeks', 12),
                'goals': program_data.get('goals', []),
                'meeting_schedule': program_data.get('meeting_schedule', 'weekly'),
                'started_at': datetime.now(),
                'status': 'active',
                'progress_milestones': [],
                'meeting_logs': []
            }
            
            self.mentorship_pairs[mentorship_id] = mentorship
            
            logger.info(f"Mentorship established: {mentorship_id}")
            return mentorship_id
            
        except Exception as e:
            logger.error(f"Setup mentorship failed: {e}")
            return None
    
    def create_learning_path(self, path_data: Dict[str, Any], creator_id: str) -> Optional[str]:
        """Create structured learning path.
        
        Args:
            path_data: Learning path information
            creator_id: Path creator user ID
            
        Returns:
            Learning path ID if successful, None otherwise
        """
        try:
            path_id = str(uuid.uuid4())
            
            learning_path = {
                'path_id': path_id,
                'title': path_data['title'],
                'description': path_data['description'],
                'creator_id': creator_id,
                'difficulty': path_data.get('difficulty', 'intermediate'),
                'estimated_duration': path_data.get('duration', '4 weeks'),
                'modules': path_data.get('modules', []),
                'prerequisites': path_data.get('prerequisites', []),
                'learning_outcomes': path_data.get('outcomes', []),
                'created_at': datetime.now(),
                'enrolled_users': [],
                'completion_rate': 0.0,
                'rating': 0.0,
                'reviews': []
            }
            
            self.learning_paths[path_id] = learning_path
            
            logger.info(f"Learning path created: {path_data['title']}")
            return path_id
            
        except Exception as e:
            logger.error(f"Create learning path failed: {e}")
            return None
    
    def get_collaboration_metrics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get collaboration metrics.
        
        Args:
            project_id: Optional project filter
            
        Returns:
            Collaboration metrics
        """
        try:
            metrics = {
                'total_sessions': 0,
                'active_sessions': 0,
                'total_participants': set(),
                'average_session_duration': 0,
                'popular_collaboration_types': {},
                'mentorship_programs': len(self.mentorship_pairs),
                'learning_paths': len(self.learning_paths)
            }
            
            sessions = list(self.active_sessions.values())
            
            if project_id:
                sessions = [s for s in sessions if s['project_id'] == project_id]
            
            metrics['total_sessions'] = len(sessions)
            metrics['active_sessions'] = len([s for s in sessions if s['status'] == 'active'])
            
            for session in sessions:
                metrics['total_participants'].update(session['participants'])
                
                session_type = session['session_type']
                metrics['popular_collaboration_types'][session_type] = \
                    metrics['popular_collaboration_types'].get(session_type, 0) + 1
            
            metrics['total_participants'] = len(metrics['total_participants'])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Get collaboration metrics failed: {e}")
            return {}