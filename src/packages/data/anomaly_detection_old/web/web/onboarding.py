"""
Interactive user onboarding system for Pynomaly.
Provides personalized learning paths and progressive disclosure.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles for personalized onboarding."""

    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    BUSINESS_ANALYST = "business_analyst"
    DEVOPS_ENGINEER = "devops_engineer"
    STUDENT = "student"
    RESEARCHER = "researcher"
    FIRST_TIME_USER = "first_time_user"


class ExperienceLevel(str, Enum):
    """Experience levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class OnboardingStage(str, Enum):
    """Onboarding stages."""

    WELCOME = "welcome"
    ROLE_SELECTION = "role_selection"
    EXPERIENCE_ASSESSMENT = "experience_assessment"
    ENVIRONMENT_SETUP = "environment_setup"
    FIRST_DETECTION = "first_detection"
    FEATURE_EXPLORATION = "feature_exploration"
    ADVANCED_CONCEPTS = "advanced_concepts"
    PRODUCTION_READINESS = "production_readiness"
    COMPLETED = "completed"


class OnboardingGoal(str, Enum):
    """User goals."""

    EVALUATE_TOOL = "evaluate_tool"
    LEARN_ANOMALY_DETECTION = "learn_anomaly_detection"
    IMPLEMENT_PRODUCTION = "implement_production"
    RESEARCH_PROJECT = "research_project"
    ACADEMIC_STUDY = "academic_study"
    PROOF_OF_CONCEPT = "proof_of_concept"


class OnboardingStep(BaseModel):
    """Individual onboarding step."""

    id: str
    title: str
    description: str
    content: str
    estimated_time: int  # minutes
    difficulty: str
    required: bool = True
    completed: bool = False
    skipped: bool = False
    completion_time: datetime | None = None
    prerequisites: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    resources: list[dict[str, str]] = Field(default_factory=list)
    interactive_elements: list[dict[str, Any]] = Field(default_factory=list)


class UserProfile(BaseModel):
    """User profile for personalized onboarding."""

    user_id: str
    role: UserRole
    experience_level: ExperienceLevel
    goals: list[OnboardingGoal]
    preferred_learning_style: str = "interactive"  # visual, hands_on, documentation
    programming_languages: list[str] = Field(default_factory=list)
    ml_frameworks: list[str] = Field(default_factory=list)
    industry: str | None = None
    use_case: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class OnboardingProgress(BaseModel):
    """User onboarding progress tracking."""

    user_id: str
    current_stage: OnboardingStage
    completed_steps: list[str] = Field(default_factory=list)
    skipped_steps: list[str] = Field(default_factory=list)
    current_step: str | None = None
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    estimated_completion: datetime | None = None
    actual_completion: datetime | None = None
    feedback: list[dict[str, Any]] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


class OnboardingService:
    """Service for managing user onboarding experience."""

    def __init__(self):
        self.steps_db = self._load_onboarding_steps()
        self.learning_paths = self._create_learning_paths()
        self.user_profiles: dict[str, UserProfile] = {}
        self.user_progress: dict[str, OnboardingProgress] = {}

    def _load_onboarding_steps(self) -> dict[str, OnboardingStep]:
        """Load onboarding steps configuration."""
        steps = {}

        # Welcome and introduction steps
        steps["welcome"] = OnboardingStep(
            id="welcome",
            title="Welcome to Pynomaly",
            description="Get introduced to the world of anomaly detection",
            content="""
            <div class="welcome-container">
                <h2>🎉 Welcome to Pynomaly!</h2>
                <p>Pynomaly is a comprehensive anomaly detection platform that helps you identify unusual patterns in your data.</p>

                <div class="value-proposition">
                    <h3>What you'll achieve:</h3>
                    <ul>
                        <li>🔍 Detect anomalies in any dataset</li>
                        <li>🚀 Deploy production-ready ML models</li>
                        <li>📊 Visualize and understand your data</li>
                        <li>🛡️ Protect your systems from outliers</li>
                    </ul>
                </div>

                <div class="next-steps">
                    <p>Let's start by understanding your background and goals.</p>
                </div>
            </div>
            """,
            estimated_time=2,
            difficulty="easy",
            required=True,
            resources=[
                {
                    "type": "video",
                    "title": "Pynomaly Overview",
                    "url": "/videos/overview",
                },
                {
                    "type": "doc",
                    "title": "What is Anomaly Detection?",
                    "url": "/docs/concepts/anomaly-detection",
                },
            ],
        )

        steps["role_selection"] = OnboardingStep(
            id="role_selection",
            title="Tell us about yourself",
            description="Help us personalize your learning experience",
            content="""
            <div class="role-selection">
                <h3>What best describes your role?</h3>
                <div class="role-cards">
                    <div class="role-card" data-role="data_scientist">
                        <h4>📊 Data Scientist</h4>
                        <p>I analyze data and build predictive models</p>
                    </div>
                    <div class="role-card" data-role="ml_engineer">
                        <h4>🔧 ML Engineer</h4>
                        <p>I deploy and maintain ML systems in production</p>
                    </div>
                    <div class="role-card" data-role="business_analyst">
                        <h4>📈 Business Analyst</h4>
                        <p>I use data to make business decisions</p>
                    </div>
                    <div class="role-card" data-role="devops_engineer">
                        <h4>⚙️ DevOps Engineer</h4>
                        <p>I manage infrastructure and deployments</p>
                    </div>
                    <div class="role-card" data-role="student">
                        <h4>🎓 Student/Researcher</h4>
                        <p>I'm learning about ML and data science</p>
                    </div>
                    <div class="role-card" data-role="first_time_user">
                        <h4>🆕 First-time User</h4>
                        <p>I'm new to anomaly detection</p>
                    </div>
                </div>
            </div>
            """,
            estimated_time=1,
            difficulty="easy",
            required=True,
            interactive_elements=[
                {
                    "type": "role_selector",
                    "options": [role.value for role in UserRole],
                    "required": True,
                }
            ],
        )

        steps["environment_setup"] = OnboardingStep(
            id="environment_setup",
            title="Set up your environment",
            description="Get Pynomaly installed and configured",
            content="""
            <div class="setup-guide">
                <h3>🛠️ Installation Options</h3>

                <div class="setup-tabs">
                    <div class="tab-content" id="pip-install">
                        <h4>Quick Install (Recommended for beginners)</h4>
                        <div class="code-block">
                            <code>pip install pynomaly</code>
                        </div>
                        <p>✅ Best for: Quick evaluation, simple use cases</p>
                    </div>

                    <div class="tab-content" id="dev-install">
                        <h4>Development Install</h4>
                        <div class="code-block">
                            <code>
                            git clone https://github.com/your-org/monorepo.git<br>
                            cd pynomaly<br>
                            pip install hatch<br>
                            hatch env create<br>
                            hatch shell
                            </code>
                        </div>
                        <p>✅ Best for: Contributors, advanced users</p>
                    </div>

                    <div class="tab-content" id="docker-install">
                        <h4>Docker Install</h4>
                        <div class="code-block">
                            <code>
                            docker pull pynomaly/pynomaly:latest<br>
                            docker run -p 8000:8000 pynomaly/pynomaly
                            </code>
                        </div>
                        <p>✅ Best for: Production deployment, isolated environments</p>
                    </div>
                </div>

                <div class="verification">
                    <h4>Verify Installation</h4>
                    <div class="code-block">
                        <code>
                        python -c "import monorepo; print('✅ Pynomaly installed successfully!')"
                        </code>
                    </div>
                </div>
            </div>
            """,
            estimated_time=10,
            difficulty="medium",
            required=True,
            interactive_elements=[
                {
                    "type": "installation_checker",
                    "checks": ["python_version", "pynomaly_import", "dependencies"],
                },
                {
                    "type": "code_executor",
                    "language": "bash",
                    "initial_code": "pip install pynomaly",
                },
            ],
        )

        steps["first_detection"] = OnboardingStep(
            id="first_detection",
            title="Your first anomaly detection",
            description="Detect anomalies in sample data in under 5 minutes",
            content="""
            <div class="first-detection">
                <h3>🎯 Let's detect some anomalies!</h3>

                <div class="sample-data">
                    <h4>Sample Dataset: E-commerce Transactions</h4>
                    <p>We'll use a sample dataset of e-commerce transactions to find unusual patterns.</p>
                </div>

                <div class="code-walkthrough">
                    <h4>Step-by-step code:</h4>
                    <div class="code-step">
                        <h5>1. Import libraries</h5>
                        <div class="code-block editable" data-step="1">
                            <code>
import pandas as pd
from pynomaly import detect_anomalies
import numpy as np
                            </code>
                        </div>
                    </div>

                    <div class="code-step">
                        <h5>2. Load sample data</h5>
                        <div class="code-block editable" data-step="2">
                            <code>
# Load sample e-commerce data
data = pd.read_csv('sample_data/ecommerce_transactions.csv')
print(f"Dataset shape: {data.shape}")
data.head()
                            </code>
                        </div>
                    </div>

                    <div class="code-step">
                        <h5>3. Detect anomalies</h5>
                        <div class="code-block editable" data-step="3">
                            <code>
# Detect anomalies using Isolation Forest
anomalies = detect_anomalies(
    data[['amount', 'frequency', 'recency']],
    contamination=0.05  # Expect 5% anomalies
)

print(f"Found {anomalies.sum()} anomalies out of {len(data)} transactions")
                            </code>
                        </div>
                    </div>

                    <div class="code-step">
                        <h5>4. Visualize results</h5>
                        <div class="code-block editable" data-step="4">
                            <code>
# View anomalous transactions
anomalous_transactions = data[anomalies]
print("Top 5 anomalous transactions:")
print(anomalous_transactions.head())
                            </code>
                        </div>
                    </div>
                </div>

                <div class="interactive-runner">
                    <button class="run-code-btn">▶️ Run Complete Example</button>
                    <div class="output-area"></div>
                </div>
            </div>
            """,
            estimated_time=15,
            difficulty="easy",
            required=True,
            interactive_elements=[
                {
                    "type": "code_playground",
                    "language": "python",
                    "sample_data": "ecommerce_transactions.csv",
                    "expected_output": "anomaly_detection_results",
                },
                {
                    "type": "visualization",
                    "chart_type": "scatter",
                    "data_source": "detection_results",
                },
            ],
        )

        # Add more steps for different roles and experience levels
        steps.update(self._create_role_specific_steps())

        return steps

    def _create_role_specific_steps(self) -> dict[str, OnboardingStep]:
        """Create role-specific onboarding steps."""
        steps = {}

        # Data Scientist specific steps
        steps["ds_algorithm_selection"] = OnboardingStep(
            id="ds_algorithm_selection",
            title="Choosing the right algorithm",
            description="Learn which algorithms work best for different types of data",
            content="""
            <div class="algorithm-guide">
                <h3>🔬 Algorithm Selection Guide</h3>

                <div class="algorithm-comparison">
                    <div class="algorithm-card">
                        <h4>Isolation Forest</h4>
                        <p><strong>Best for:</strong> High-dimensional data, mixed data types</p>
                        <p><strong>Pros:</strong> Fast, handles categorical features</p>
                        <p><strong>Cons:</strong> Less effective in very high dimensions</p>
                        <div class="use-cases">
                            <strong>Use cases:</strong> Fraud detection, network security, quality control
                        </div>
                    </div>

                    <div class="algorithm-card">
                        <h4>Local Outlier Factor (LOF)</h4>
                        <p><strong>Best for:</strong> Local density-based anomalies</p>
                        <p><strong>Pros:</strong> Finds local outliers, good for clusters</p>
                        <p><strong>Cons:</strong> Sensitive to parameter selection</p>
                        <div class="use-cases">
                            <strong>Use cases:</strong> Image analysis, sensor data, time series
                        </div>
                    </div>

                    <div class="algorithm-card">
                        <h4>One-Class SVM</h4>
                        <p><strong>Best for:</strong> Non-linear boundaries, small datasets</p>
                        <p><strong>Pros:</strong> Powerful kernel methods, theoretical foundation</p>
                        <p><strong>Cons:</strong> Slower, requires parameter tuning</p>
                        <div class="use-cases">
                            <strong>Use cases:</strong> Text analysis, bioinformatics, security
                        </div>
                    </div>
                </div>

                <div class="decision-tree">
                    <h4>🌳 Decision Tree: Which Algorithm?</h4>
                    <div class="interactive-flowchart">
                        <!-- Interactive flowchart would be implemented with JavaScript -->
                    </div>
                </div>
            </div>
            """,
            estimated_time=20,
            difficulty="intermediate",
            required=False,
            prerequisites=["first_detection"],
            resources=[
                {
                    "type": "doc",
                    "title": "Algorithm Comparison",
                    "url": "/docs/algorithms/comparison",
                },
                {
                    "type": "notebook",
                    "title": "Algorithm Benchmarks",
                    "url": "/notebooks/algorithm_comparison.ipynb",
                },
            ],
        )

        # ML Engineer specific steps
        steps["mle_production_deployment"] = OnboardingStep(
            id="mle_production_deployment",
            title="Production deployment patterns",
            description="Learn how to deploy Pynomaly in production environments",
            content="""
            <div class="deployment-guide">
                <h3>🚀 Production Deployment</h3>

                <div class="deployment-patterns">
                    <div class="pattern-card">
                        <h4>🐳 Container Deployment</h4>
                        <div class="code-block">
                            <code>
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "monorepo.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
                            </code>
                        </div>
                    </div>

                    <div class="pattern-card">
                        <h4>☸️ Kubernetes Deployment</h4>
                        <div class="code-block">
                            <code>
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pynomaly-api
  template:
    metadata:
      labels:
        app: pynomaly-api
    spec:
      containers:
      - name: pynomaly
        image: pynomaly/pynomaly:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
                            </code>
                        </div>
                    </div>
                </div>

                <div class="monitoring-setup">
                    <h4>📊 Monitoring & Observability</h4>
                    <ul>
                        <li>✅ Health checks and readiness probes</li>
                        <li>✅ Prometheus metrics integration</li>
                        <li>✅ Distributed tracing with OpenTelemetry</li>
                        <li>✅ Structured logging with correlation IDs</li>
                    </ul>
                </div>
            </div>
            """,
            estimated_time=30,
            difficulty="advanced",
            required=False,
            prerequisites=["environment_setup"],
            resources=[
                {
                    "type": "doc",
                    "title": "Production Guide",
                    "url": "/docs/deployment/production",
                },
                {
                    "type": "template",
                    "title": "K8s Templates",
                    "url": "/templates/kubernetes/",
                },
            ],
        )

        return steps

    def _create_learning_paths(self) -> dict[UserRole, list[str]]:
        """Create personalized learning paths for different user roles."""
        return {
            UserRole.DATA_SCIENTIST: [
                "welcome",
                "role_selection",
                "environment_setup",
                "first_detection",
                "ds_algorithm_selection",
                "feature_engineering",
                "model_evaluation",
                "advanced_visualization",
            ],
            UserRole.ML_ENGINEER: [
                "welcome",
                "role_selection",
                "environment_setup",
                "first_detection",
                "api_integration",
                "mle_production_deployment",
                "monitoring_setup",
                "performance_optimization",
            ],
            UserRole.BUSINESS_ANALYST: [
                "welcome",
                "role_selection",
                "environment_setup",
                "first_detection",
                "business_use_cases",
                "dashboard_setup",
                "interpretation_guide",
            ],
            UserRole.DEVOPS_ENGINEER: [
                "welcome",
                "role_selection",
                "environment_setup",
                "container_deployment",
                "monitoring_setup",
                "security_configuration",
                "scaling_strategies",
            ],
            UserRole.STUDENT: [
                "welcome",
                "role_selection",
                "environment_setup",
                "anomaly_detection_theory",
                "first_detection",
                "algorithm_deep_dive",
                "academic_resources",
            ],
            UserRole.FIRST_TIME_USER: [
                "welcome",
                "role_selection",
                "anomaly_detection_basics",
                "environment_setup",
                "first_detection",
                "common_use_cases",
            ],
        }

    def start_onboarding(self, user_id: str, request: Request) -> OnboardingProgress:
        """Start onboarding for a new user."""
        progress = OnboardingProgress(
            user_id=user_id,
            current_stage=OnboardingStage.WELCOME,
            current_step="welcome",
        )

        self.user_progress[user_id] = progress
        return progress

    def update_user_profile(
        self, user_id: str, profile_data: dict[str, Any]
    ) -> UserProfile:
        """Update user profile with onboarding responses."""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            for key, value in profile_data.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            profile.updated_at = datetime.now()
        else:
            profile = UserProfile(user_id=user_id, **profile_data)
            self.user_profiles[user_id] = profile

        # Update learning path based on new profile
        self._update_learning_path(user_id)

        return profile

    def _update_learning_path(self, user_id: str):
        """Update user's learning path based on their profile."""
        if user_id not in self.user_profiles or user_id not in self.user_progress:
            return

        profile = self.user_profiles[user_id]
        progress = self.user_progress[user_id]

        # Get personalized learning path
        base_path = self.learning_paths.get(
            profile.role, self.learning_paths[UserRole.FIRST_TIME_USER]
        )

        # Customize based on experience level
        if profile.experience_level == ExperienceLevel.BEGINNER:
            # Add more introductory steps
            pass
        elif profile.experience_level == ExperienceLevel.ADVANCED:
            # Skip basic steps, add advanced ones
            base_path = [step for step in base_path if not step.startswith("basic_")]

        # Update progress with new path
        progress.learning_path = base_path

    def get_next_step(self, user_id: str) -> OnboardingStep | None:
        """Get the next step for user's onboarding."""
        if user_id not in self.user_progress:
            return None

        progress = self.user_progress[user_id]
        profile = self.user_profiles.get(user_id)

        if not profile:
            return self.steps_db.get("welcome")

        learning_path = self.learning_paths.get(profile.role, [])

        # Find next uncompleted step
        for step_id in learning_path:
            if step_id not in progress.completed_steps:
                step = self.steps_db.get(step_id)
                if step and self._check_prerequisites(step, progress.completed_steps):
                    return step

        return None

    def _check_prerequisites(
        self, step: OnboardingStep, completed_steps: list[str]
    ) -> bool:
        """Check if step prerequisites are met."""
        return all(prereq in completed_steps for prereq in step.prerequisites)

    def complete_step(
        self, user_id: str, step_id: str, feedback: dict[str, Any] | None = None
    ) -> bool:
        """Mark a step as completed."""
        if user_id not in self.user_progress:
            return False

        progress = self.user_progress[user_id]

        if step_id not in progress.completed_steps:
            progress.completed_steps.append(step_id)
            progress.last_activity = datetime.now()

            # Add feedback if provided
            if feedback:
                feedback_entry = {
                    "step_id": step_id,
                    "feedback": feedback,
                    "timestamp": datetime.now().isoformat(),
                }
                progress.feedback.append(feedback_entry)

            # Update current step to next one
            next_step = self.get_next_step(user_id)
            progress.current_step = next_step.id if next_step else None

            # Check for achievements
            self._check_achievements(user_id)

            # Update stage if needed
            self._update_stage(user_id)

        return True

    def _check_achievements(self, user_id: str):
        """Check and award achievements."""
        progress = self.user_progress[user_id]
        profile = self.user_profiles.get(user_id)

        if not profile:
            return

        achievements = []

        # First detection achievement
        if (
            "first_detection" in progress.completed_steps
            and "first_detector" not in progress.achievements
        ):
            achievements.append("first_detector")

        # Speed achievements
        if len(progress.completed_steps) >= 5:
            if (
                datetime.now() - progress.started_at
            ).total_seconds() < 1800:  # 30 minutes
                achievements.append("speed_learner")

        # Completionist
        learning_path = self.learning_paths.get(profile.role, [])
        if len(progress.completed_steps) >= len(learning_path) * 0.8:
            achievements.append("completionist")

        progress.achievements.extend(achievements)

    def _update_stage(self, user_id: str):
        """Update user's onboarding stage."""
        progress = self.user_progress[user_id]
        profile = self.user_profiles.get(user_id)

        if not profile:
            return

        completed_count = len(progress.completed_steps)
        learning_path = self.learning_paths.get(profile.role, [])
        completion_ratio = completed_count / len(learning_path) if learning_path else 0

        # Update stage based on completion
        if completion_ratio >= 1.0:
            progress.current_stage = OnboardingStage.COMPLETED
            progress.actual_completion = datetime.now()
        elif completion_ratio >= 0.8:
            progress.current_stage = OnboardingStage.PRODUCTION_READINESS
        elif completion_ratio >= 0.6:
            progress.current_stage = OnboardingStage.ADVANCED_CONCEPTS
        elif completion_ratio >= 0.4:
            progress.current_stage = OnboardingStage.FEATURE_EXPLORATION
        elif "first_detection" in progress.completed_steps:
            progress.current_stage = OnboardingStage.FIRST_DETECTION
        elif "environment_setup" in progress.completed_steps:
            progress.current_stage = OnboardingStage.ENVIRONMENT_SETUP
        elif "role_selection" in progress.completed_steps:
            progress.current_stage = OnboardingStage.EXPERIENCE_ASSESSMENT

    def get_progress_summary(self, user_id: str) -> dict[str, Any]:
        """Get comprehensive progress summary."""
        if user_id not in self.user_progress:
            return {}

        progress = self.user_progress[user_id]
        profile = self.user_profiles.get(user_id)

        if not profile:
            return {}

        learning_path = self.learning_paths.get(profile.role, [])
        completion_ratio = (
            len(progress.completed_steps) / len(learning_path) if learning_path else 0
        )

        # Calculate estimated time remaining
        remaining_steps = [
            step_id
            for step_id in learning_path
            if step_id not in progress.completed_steps
        ]
        estimated_time_remaining = sum(
            self.steps_db[step_id].estimated_time
            for step_id in remaining_steps
            if step_id in self.steps_db
        )

        return {
            "user_id": user_id,
            "role": profile.role.value,
            "experience_level": profile.experience_level.value,
            "current_stage": progress.current_stage.value,
            "completion_ratio": completion_ratio,
            "completed_steps": len(progress.completed_steps),
            "total_steps": len(learning_path),
            "estimated_time_remaining": estimated_time_remaining,
            "achievements": progress.achievements,
            "started_at": progress.started_at.isoformat(),
            "last_activity": progress.last_activity.isoformat(),
            "next_step": self.get_next_step(user_id).title
            if self.get_next_step(user_id)
            else None,
        }

    def get_personalized_resources(self, user_id: str) -> list[dict[str, Any]]:
        """Get personalized learning resources."""
        profile = self.user_profiles.get(user_id)
        progress = self.user_progress.get(user_id)

        if not profile or not progress:
            return []

        resources = []

        # Role-specific resources
        role_resources = {
            UserRole.DATA_SCIENTIST: [
                {
                    "type": "notebook",
                    "title": "Advanced Algorithm Comparison",
                    "url": "/notebooks/advanced_algorithms.ipynb",
                },
                {
                    "type": "doc",
                    "title": "Statistical Testing",
                    "url": "/docs/statistics/testing",
                },
                {
                    "type": "video",
                    "title": "Feature Engineering",
                    "url": "/videos/feature_engineering",
                },
            ],
            UserRole.ML_ENGINEER: [
                {
                    "type": "doc",
                    "title": "MLOps Best Practices",
                    "url": "/docs/mlops/best_practices",
                },
                {
                    "type": "template",
                    "title": "CI/CD Pipeline",
                    "url": "/templates/cicd/",
                },
                {
                    "type": "video",
                    "title": "Model Monitoring",
                    "url": "/videos/monitoring",
                },
            ],
            UserRole.BUSINESS_ANALYST: [
                {
                    "type": "doc",
                    "title": "Business Value of Anomaly Detection",
                    "url": "/docs/business/value",
                },
                {
                    "type": "case_study",
                    "title": "ROI Calculator",
                    "url": "/tools/roi_calculator",
                },
                {
                    "type": "video",
                    "title": "Executive Presentation",
                    "url": "/videos/executive_summary",
                },
            ],
        }

        resources.extend(role_resources.get(profile.role, []))

        # Experience-level specific resources
        if profile.experience_level == ExperienceLevel.BEGINNER:
            resources.extend(
                [
                    {
                        "type": "tutorial",
                        "title": "ML Fundamentals",
                        "url": "/tutorials/ml_basics",
                    },
                    {
                        "type": "glossary",
                        "title": "Anomaly Detection Terms",
                        "url": "/docs/glossary",
                    },
                ]
            )
        elif profile.experience_level == ExperienceLevel.ADVANCED:
            resources.extend(
                [
                    {
                        "type": "research",
                        "title": "Latest Papers",
                        "url": "/research/papers",
                    },
                    {
                        "type": "benchmark",
                        "title": "Algorithm Benchmarks",
                        "url": "/benchmarks/algorithms",
                    },
                ]
            )

        return resources


# Global onboarding service instance
onboarding_service = OnboardingService()
