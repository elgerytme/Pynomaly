#!/usr/bin/env python3
"""
Beta User Feedback Collection System for Pynomaly

Comprehensive feedback collection, analysis, and reporting system for beta users
with real-time analytics and sentiment analysis.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    USABILITY = "usability"
    PERFORMANCE = "performance"
    SATISFACTION = "satisfaction"
    NPS = "nps"
    GENERAL = "general"


class FeedbackPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FeedbackStatus(Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class FeedbackItem:
    id: str
    user_id: str
    user_email: str
    feedback_type: FeedbackType
    title: str
    description: str
    rating: int | None  # 1-5 scale
    priority: FeedbackPriority
    status: FeedbackStatus
    tags: list[str]
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None
    response: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FeedbackSurvey:
    id: str
    title: str
    description: str
    questions: list[dict[str, Any]]
    target_users: list[str]
    created_at: datetime
    expires_at: datetime
    responses: list[dict[str, Any]] = None
    active: bool = True

    def __post_init__(self):
        if self.responses is None:
            self.responses = []


class BetaFeedbackSystem:
    """Comprehensive feedback collection and analysis system."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.feedback_db = []
        self.surveys = []
        self.feedback_analytics = {}
        self.system_id = f"feedback_sys_{int(time.time())}"

    def collect_feedback(
        self,
        user_id: str,
        user_email: str,
        feedback_type: FeedbackType,
        title: str,
        description: str,
        rating: int | None = None,
        tags: list[str] = None,
    ) -> str:
        """Collect feedback from beta users."""
        feedback_id = f"feedback_{int(time.time())}_{hash(user_email) % 1000}"

        # Determine priority based on feedback type and content
        priority = self._determine_priority(feedback_type, description, rating)

        feedback = FeedbackItem(
            id=feedback_id,
            user_id=user_id,
            user_email=user_email,
            feedback_type=feedback_type,
            title=title,
            description=description,
            rating=rating,
            priority=priority,
            status=FeedbackStatus.NEW,
            tags=tags or [],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                "user_agent": "Pynomaly Beta Client v1.0.0",
                "platform": "web",
                "session_id": f"session_{hash(user_email) % 10000}",
            },
        )

        self.feedback_db.append(feedback)

        logger.info(f"✅ Feedback collected: {feedback_id} from {user_email}")
        return feedback_id

    def _determine_priority(
        self, feedback_type: FeedbackType, description: str, rating: int | None
    ) -> FeedbackPriority:
        """Determine feedback priority based on type and content."""
        # Critical keywords
        critical_keywords = ["crash", "error", "broken", "cannot", "unable", "critical"]
        high_keywords = ["slow", "performance", "urgent", "important", "blocking"]

        description_lower = description.lower()

        # Critical priority conditions
        if feedback_type == FeedbackType.BUG_REPORT and any(
            word in description_lower for word in critical_keywords
        ):
            return FeedbackPriority.CRITICAL

        if rating is not None and rating <= 2:
            return FeedbackPriority.HIGH

        # High priority conditions
        if feedback_type == FeedbackType.BUG_REPORT:
            return FeedbackPriority.HIGH

        if any(word in description_lower for word in high_keywords):
            return FeedbackPriority.HIGH

        # Medium priority conditions
        if feedback_type in [FeedbackType.FEATURE_REQUEST, FeedbackType.PERFORMANCE]:
            return FeedbackPriority.MEDIUM

        if rating is not None and rating == 3:
            return FeedbackPriority.MEDIUM

        # Default to low priority
        return FeedbackPriority.LOW

    def create_feedback_survey(
        self,
        title: str,
        description: str,
        questions: list[dict[str, Any]],
        target_users: list[str],
        expires_in_days: int = 7,
    ) -> str:
        """Create a feedback survey for beta users."""
        survey_id = f"survey_{int(time.time())}"

        survey = FeedbackSurvey(
            id=survey_id,
            title=title,
            description=description,
            questions=questions,
            target_users=target_users,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=expires_in_days),
        )

        self.surveys.append(survey)

        logger.info(f"📋 Survey created: {survey_id} for {len(target_users)} users")
        return survey_id

    def submit_survey_response(
        self, survey_id: str, user_id: str, responses: dict[str, Any]
    ) -> bool:
        """Submit survey response from user."""
        survey = next((s for s in self.surveys if s.id == survey_id), None)

        if not survey:
            logger.error(f"Survey not found: {survey_id}")
            return False

        if not survey.active or datetime.now() > survey.expires_at:
            logger.error(f"Survey expired or inactive: {survey_id}")
            return False

        response_entry = {
            "user_id": user_id,
            "submitted_at": datetime.now().isoformat(),
            "responses": responses,
        }

        survey.responses.append(response_entry)

        logger.info(f"📝 Survey response submitted: {survey_id} by {user_id}")
        return True

    def analyze_feedback_sentiment(self) -> dict[str, Any]:
        """Analyze sentiment across all feedback."""
        sentiment_analysis = {
            "overall_sentiment": "neutral",
            "positive_feedback_count": 0,
            "negative_feedback_count": 0,
            "neutral_feedback_count": 0,
            "sentiment_by_type": {},
            "trending_topics": [],
            "satisfaction_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        }

        positive_keywords = [
            "great",
            "excellent",
            "love",
            "amazing",
            "perfect",
            "awesome",
            "helpful",
        ]
        negative_keywords = [
            "terrible",
            "awful",
            "hate",
            "horrible",
            "worst",
            "broken",
            "useless",
        ]

        for feedback in self.feedback_db:
            description_lower = feedback.description.lower()
            title_lower = feedback.title.lower()
            combined_text = f"{title_lower} {description_lower}"

            # Sentiment classification
            positive_score = sum(
                1 for word in positive_keywords if word in combined_text
            )
            negative_score = sum(
                1 for word in negative_keywords if word in combined_text
            )

            if feedback.rating:
                sentiment_analysis["satisfaction_distribution"][feedback.rating] += 1

                if feedback.rating >= 4:
                    positive_score += 2
                elif feedback.rating <= 2:
                    negative_score += 2

            if positive_score > negative_score:
                sentiment = "positive"
                sentiment_analysis["positive_feedback_count"] += 1
            elif negative_score > positive_score:
                sentiment = "negative"
                sentiment_analysis["negative_feedback_count"] += 1
            else:
                sentiment = "neutral"
                sentiment_analysis["neutral_feedback_count"] += 1

            # Sentiment by type
            feedback_type = feedback.feedback_type.value
            if feedback_type not in sentiment_analysis["sentiment_by_type"]:
                sentiment_analysis["sentiment_by_type"][feedback_type] = {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                }
            sentiment_analysis["sentiment_by_type"][feedback_type][sentiment] += 1

        # Overall sentiment calculation
        total_feedback = len(self.feedback_db)
        if total_feedback > 0:
            positive_ratio = (
                sentiment_analysis["positive_feedback_count"] / total_feedback
            )
            negative_ratio = (
                sentiment_analysis["negative_feedback_count"] / total_feedback
            )

            if positive_ratio > 0.6:
                sentiment_analysis["overall_sentiment"] = "positive"
            elif negative_ratio > 0.4:
                sentiment_analysis["overall_sentiment"] = "negative"
            else:
                sentiment_analysis["overall_sentiment"] = "neutral"

        # Trending topics (simulated)
        sentiment_analysis["trending_topics"] = [
            {"topic": "API Documentation", "mentions": 8, "sentiment": "mixed"},
            {"topic": "Dashboard Performance", "mentions": 6, "sentiment": "negative"},
            {
                "topic": "Anomaly Detection Accuracy",
                "mentions": 12,
                "sentiment": "positive",
            },
            {"topic": "Onboarding Process", "mentions": 5, "sentiment": "positive"},
            {"topic": "Export Functionality", "mentions": 4, "sentiment": "mixed"},
        ]

        return sentiment_analysis

    def generate_feedback_insights(self) -> dict[str, Any]:
        """Generate actionable insights from feedback data."""
        insights = {
            "summary": {
                "total_feedback_items": len(self.feedback_db),
                "critical_issues": len(
                    [
                        f
                        for f in self.feedback_db
                        if f.priority == FeedbackPriority.CRITICAL
                    ]
                ),
                "high_priority_items": len(
                    [f for f in self.feedback_db if f.priority == FeedbackPriority.HIGH]
                ),
                "average_rating": 0,
                "response_time_avg": "2.5 days",  # Simulated
            },
            "top_issues": [],
            "feature_requests": [],
            "user_satisfaction": {},
            "actionable_recommendations": [],
        }

        # Calculate average rating
        ratings = [f.rating for f in self.feedback_db if f.rating is not None]
        if ratings:
            insights["summary"]["average_rating"] = sum(ratings) / len(ratings)

        # Top issues by priority and frequency
        issue_counts = {}
        feature_requests = []

        for feedback in self.feedback_db:
            if feedback.feedback_type == FeedbackType.BUG_REPORT:
                issue_key = feedback.title.lower()
                if issue_key not in issue_counts:
                    issue_counts[issue_key] = {
                        "title": feedback.title,
                        "count": 0,
                        "priority": feedback.priority.value,
                        "latest_report": feedback.created_at.isoformat(),
                    }
                issue_counts[issue_key]["count"] += 1

            elif feedback.feedback_type == FeedbackType.FEATURE_REQUEST:
                feature_requests.append(
                    {
                        "title": feedback.title,
                        "description": feedback.description,
                        "user_email": feedback.user_email,
                        "created_at": feedback.created_at.isoformat(),
                        "priority": feedback.priority.value,
                    }
                )

        # Sort top issues by count and priority
        insights["top_issues"] = sorted(
            issue_counts.values(),
            key=lambda x: (x["count"], 1 if x["priority"] == "critical" else 0),
            reverse=True,
        )[:10]

        # Sort feature requests by priority
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        insights["feature_requests"] = sorted(
            feature_requests,
            key=lambda x: priority_order.get(x["priority"], 0),
            reverse=True,
        )[:10]

        # User satisfaction analysis
        satisfaction_by_user = {}
        for feedback in self.feedback_db:
            if feedback.rating is not None:
                user = feedback.user_email
                if user not in satisfaction_by_user:
                    satisfaction_by_user[user] = []
                satisfaction_by_user[user].append(feedback.rating)

        for user, ratings in satisfaction_by_user.items():
            avg_rating = sum(ratings) / len(ratings)
            insights["user_satisfaction"][user] = {
                "average_rating": round(avg_rating, 2),
                "feedback_count": len(ratings),
                "satisfaction_level": "high"
                if avg_rating >= 4
                else "medium"
                if avg_rating >= 3
                else "low",
            }

        # Generate actionable recommendations
        insights["actionable_recommendations"] = self._generate_recommendations(
            insights
        )

        return insights

    def _generate_recommendations(self, insights: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on feedback analysis."""
        recommendations = []

        # Critical issues
        if insights["summary"]["critical_issues"] > 0:
            recommendations.append(
                f"🚨 URGENT: Address {insights['summary']['critical_issues']} critical issues immediately"
            )

        # High priority items
        if insights["summary"]["high_priority_items"] > 5:
            recommendations.append(
                f"⚠️ HIGH PRIORITY: {insights['summary']['high_priority_items']} high-priority items need attention"
            )

        # Low satisfaction
        if insights["summary"]["average_rating"] < 3.5:
            recommendations.append(
                "📉 LOW SATISFACTION: Overall user satisfaction is below target - conduct user interviews"
            )

        # Top issues
        if insights["top_issues"]:
            top_issue = insights["top_issues"][0]
            recommendations.append(
                f"🔧 FOCUS AREA: '{top_issue['title']}' reported {top_issue['count']} times - prioritize fix"
            )

        # Feature requests
        if len(insights["feature_requests"]) > 10:
            recommendations.append(
                "💡 FEATURE DEVELOPMENT: High volume of feature requests - consider roadmap prioritization"
            )

        # User engagement
        low_satisfaction_users = [
            user
            for user, data in insights["user_satisfaction"].items()
            if data["satisfaction_level"] == "low"
        ]
        if low_satisfaction_users:
            recommendations.append(
                f"👥 USER OUTREACH: {len(low_satisfaction_users)} users have low satisfaction - reach out personally"
            )

        return recommendations

    def generate_feedback_report(self) -> dict[str, Any]:
        """Generate comprehensive feedback report."""
        sentiment_analysis = self.analyze_feedback_sentiment()
        insights = self.generate_feedback_insights()

        report = {
            "report_id": f"feedback_report_{int(time.time())}",
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start_date": min(f.created_at for f in self.feedback_db).isoformat()
                if self.feedback_db
                else None,
                "end_date": max(f.created_at for f in self.feedback_db).isoformat()
                if self.feedback_db
                else None,
            },
            "summary_metrics": insights["summary"],
            "sentiment_analysis": sentiment_analysis,
            "feedback_insights": insights,
            "feedback_distribution": {
                "by_type": {},
                "by_priority": {},
                "by_status": {},
            },
            "survey_analytics": {
                "total_surveys": len(self.surveys),
                "active_surveys": len([s for s in self.surveys if s.active]),
                "total_responses": sum(len(s.responses) for s in self.surveys),
                "response_rate": "68%",  # Simulated
            },
            "raw_feedback": [
                asdict(feedback) for feedback in self.feedback_db[-50:]
            ],  # Last 50 items
            "recommendations": insights["actionable_recommendations"],
        }

        # Calculate distributions
        for feedback in self.feedback_db:
            # By type
            feedback_type = feedback.feedback_type.value
            report["feedback_distribution"]["by_type"][feedback_type] = (
                report["feedback_distribution"]["by_type"].get(feedback_type, 0) + 1
            )

            # By priority
            priority = feedback.priority.value
            report["feedback_distribution"]["by_priority"][priority] = (
                report["feedback_distribution"]["by_priority"].get(priority, 0) + 1
            )

            # By status
            status = feedback.status.value
            report["feedback_distribution"]["by_status"][status] = (
                report["feedback_distribution"]["by_status"].get(status, 0) + 1
            )

        return report

    def simulate_beta_feedback(self, num_feedback_items: int = 25) -> None:
        """Simulate realistic beta feedback for demonstration."""
        logger.info(f"🎭 Simulating {num_feedback_items} beta feedback items...")

        # Sample beta users
        beta_users = [
            ("user_001", "sarah.chen@fintechcorp.com"),
            ("user_002", "james.miller@cryptobank.io"),
            ("user_003", "david.park@automaker.com"),
            ("user_004", "dr.smith@hospital.org"),
            ("user_005", "lisa.anderson@aerospace.com"),
        ]

        # Sample feedback scenarios
        feedback_scenarios = [
            {
                "type": FeedbackType.FEATURE_REQUEST,
                "title": "Advanced Visualization Options",
                "description": "Would love to see more chart types and customization options for the dashboard visualizations. Current options are limited.",
                "rating": 4,
                "tags": ["dashboard", "visualization", "enhancement"],
            },
            {
                "type": FeedbackType.BUG_REPORT,
                "title": "API Rate Limiting Issues",
                "description": "Getting rate limited errors when making API calls even though we're within documented limits. This is blocking our integration.",
                "rating": 2,
                "tags": ["api", "rate-limiting", "integration"],
            },
            {
                "type": FeedbackType.USABILITY,
                "title": "Onboarding Process Feedback",
                "description": "The onboarding was smooth overall, but could use more examples for different use cases. Great documentation though!",
                "rating": 4,
                "tags": ["onboarding", "documentation", "examples"],
            },
            {
                "type": FeedbackType.PERFORMANCE,
                "title": "Dashboard Loading Speed",
                "description": "Dashboard takes a long time to load when we have large datasets. Performance optimization would be helpful.",
                "rating": 3,
                "tags": ["performance", "dashboard", "optimization"],
            },
            {
                "type": FeedbackType.SATISFACTION,
                "title": "Overall Product Satisfaction",
                "description": "Really impressed with the anomaly detection accuracy! This is solving our exact use case perfectly.",
                "rating": 5,
                "tags": ["satisfaction", "anomaly-detection", "accuracy"],
            },
            {
                "type": FeedbackType.FEATURE_REQUEST,
                "title": "Custom Alert Thresholds",
                "description": "Need ability to set custom thresholds for alerts based on our specific business rules and requirements.",
                "rating": 4,
                "tags": ["alerts", "customization", "thresholds"],
            },
            {
                "type": FeedbackType.BUG_REPORT,
                "title": "Export Functionality Not Working",
                "description": "CSV export is failing for datasets larger than 10MB. Getting timeout errors consistently.",
                "rating": 2,
                "tags": ["export", "csv", "timeout", "bug"],
            },
            {
                "type": FeedbackType.NPS,
                "title": "Net Promoter Score Feedback",
                "description": "Would definitely recommend Pynomaly to colleagues. The ML models are very accurate for our use case.",
                "rating": 9,
                "tags": ["nps", "recommendation", "ml-accuracy"],
            },
        ]

        # Generate feedback items
        import random

        for i in range(num_feedback_items):
            user_id, user_email = random.choice(beta_users)
            scenario = random.choice(feedback_scenarios)

            # Add some variation to the feedback
            variations = [
                " Also experiencing this in our production environment.",
                " This would be a game-changer for our team.",
                " Similar to what we've seen in other tools.",
                " Would appreciate a quick resolution on this.",
                " Great work overall though!",
                "",
            ]

            description = scenario["description"] + random.choice(variations)

            self.collect_feedback(
                user_id=user_id,
                user_email=user_email,
                feedback_type=scenario["type"],
                title=scenario["title"],
                description=description,
                rating=scenario.get("rating"),
                tags=scenario.get("tags", []),
            )

            # Add some time variation
            time.sleep(0.1)

        logger.info(f"✅ Generated {len(self.feedback_db)} feedback items")

    def create_sample_surveys(self) -> None:
        """Create sample feedback surveys."""
        # User Satisfaction Survey
        satisfaction_questions = [
            {
                "id": "q1",
                "type": "rating",
                "question": "How satisfied are you with Pynomaly overall?",
                "scale": {
                    "min": 1,
                    "max": 5,
                    "labels": ["Very Dissatisfied", "Very Satisfied"],
                },
            },
            {
                "id": "q2",
                "type": "multiple_choice",
                "question": "Which feature do you find most valuable?",
                "options": [
                    "Anomaly Detection",
                    "Dashboard",
                    "API",
                    "Alerts",
                    "Export",
                ],
            },
            {
                "id": "q3",
                "type": "text",
                "question": "What would you like to see improved first?",
                "required": False,
            },
            {
                "id": "q4",
                "type": "rating",
                "question": "How likely are you to recommend Pynomaly? (NPS)",
                "scale": {"min": 0, "max": 10, "labels": ["Not Likely", "Very Likely"]},
            },
        ]

        self.create_feedback_survey(
            title="Beta User Satisfaction Survey",
            description="Help us improve Pynomaly by sharing your experience",
            questions=satisfaction_questions,
            target_users=["user_001", "user_002", "user_003", "user_004", "user_005"],
            expires_in_days=14,
        )

        # Feature Priority Survey
        feature_questions = [
            {
                "id": "f1",
                "type": "ranking",
                "question": "Rank these upcoming features by priority for your use case",
                "options": [
                    "Real-time streaming support",
                    "Advanced visualization options",
                    "Custom alert rules",
                    "API rate limit increases",
                    "Mobile dashboard",
                ],
            },
            {
                "id": "f2",
                "type": "text",
                "question": "Describe a feature that would make Pynomaly indispensable for your workflow",
                "required": True,
            },
        ]

        self.create_feedback_survey(
            title="Feature Prioritization Survey",
            description="Help us prioritize our development roadmap",
            questions=feature_questions,
            target_users=["user_001", "user_002", "user_003"],
            expires_in_days=10,
        )

        logger.info("📋 Created sample surveys")


def main():
    """Main feedback system demonstration."""
    project_root = Path(__file__).parent.parent.parent.parent
    feedback_system = BetaFeedbackSystem(project_root)

    print("🧪 Beta Feedback System Demo")
    print("=" * 50)

    # Simulate feedback collection
    feedback_system.simulate_beta_feedback(25)

    # Create sample surveys
    feedback_system.create_sample_surveys()

    # Generate and display report
    report = feedback_system.generate_feedback_report()

    print(f"\n📊 Feedback Report Generated: {report['report_id']}")
    print(
        f"⏱️  Period: {report['period']['start_date']} to {report['period']['end_date']}"
    )
    print(
        f"📈 Total Feedback Items: {report['summary_metrics']['total_feedback_items']}"
    )
    print(f"⭐ Average Rating: {report['summary_metrics']['average_rating']:.2f}/5.0")
    print(f"🚨 Critical Issues: {report['summary_metrics']['critical_issues']}")

    print("\n🎭 Sentiment Analysis:")
    sentiment = report["sentiment_analysis"]
    print(f"  Overall Sentiment: {sentiment['overall_sentiment'].title()}")
    print(f"  Positive: {sentiment['positive_feedback_count']}")
    print(f"  Negative: {sentiment['negative_feedback_count']}")
    print(f"  Neutral: {sentiment['neutral_feedback_count']}")

    print("\n📋 Top Recommendations:")
    for recommendation in report["recommendations"][:5]:
        print(f"  • {recommendation}")

    # Save report
    report_file = project_root / f"beta_feedback_report_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📄 Full report saved to: {report_file}")

    return report


if __name__ == "__main__":
    main()
