#!/usr/bin/env python3
"""
Beta User Check-in and Engagement System for Pynomaly

Automated system for conducting weekly check-ins with beta users,
tracking engagement, and providing personalized support.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EngagementLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INACTIVE = "inactive"


class CheckinType(Enum):
    AUTOMATED = "automated"
    PERSONAL = "personal"
    ONBOARDING = "onboarding"
    FEEDBACK = "feedback"
    SUPPORT = "support"


@dataclass
class BetaUserProfile:
    user_id: str
    email: str
    name: str
    company: str
    industry: str
    role: str
    registration_date: datetime
    last_active: datetime | None
    engagement_level: EngagementLevel
    usage_metrics: dict[str, Any]
    feedback_count: int = 0
    support_tickets: int = 0
    last_checkin: datetime | None = None
    checkin_frequency: str = "weekly"
    preferred_contact: str = "email"
    notes: str = ""


@dataclass
class CheckinRecord:
    checkin_id: str
    user_id: str
    checkin_type: CheckinType
    scheduled_date: datetime
    completed_date: datetime | None
    response_received: bool
    satisfaction_score: int | None
    issues_raised: list[str]
    follow_up_required: bool
    notes: str
    next_checkin: datetime | None = None


class BetaUserCheckinSystem:
    """Comprehensive beta user check-in and engagement system."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.users = []
        self.checkin_records = []
        self.system_id = f"checkin_sys_{int(time.time())}"
        self.load_beta_users()

    def load_beta_users(self):
        """Load beta users from the beta launch data."""
        try:
            # Look for the most recent beta launch report
            beta_reports = list(self.project_root.glob("beta_launch_report_*.json"))
            if beta_reports:
                latest_report = max(beta_reports, key=lambda p: p.stat().st_mtime)

                with open(latest_report) as f:
                    launch_data = json.load(f)

                # Convert launch data to user profiles
                for user_data in launch_data.get("beta_users", []):
                    user_profile = BetaUserProfile(
                        user_id=user_data.get("email", "").split("@")[0],
                        email=user_data.get("email", ""),
                        name=user_data.get("email", "")
                        .split("@")[0]
                        .replace(".", " ")
                        .title(),
                        company=user_data.get("company", "Unknown Company"),
                        industry=user_data.get("industry", "unknown"),
                        role=user_data.get("role", "User"),
                        registration_date=datetime.fromisoformat(
                            user_data.get("invited_at")
                        ),
                        last_active=datetime.fromisoformat(user_data.get("last_active"))
                        if user_data.get("last_active")
                        else None,
                        engagement_level=self._determine_engagement_level(user_data),
                        usage_metrics=user_data.get("usage_metrics", {}),
                        feedback_count=len(user_data.get("feedback_responses", [])),
                        notes=f"Beta user since {user_data.get('invited_at', 'unknown')}",
                    )
                    self.users.append(user_profile)

                logger.info(
                    f"📊 Loaded {len(self.users)} beta users for check-in system"
                )
            else:
                # Create sample users if no launch data available
                self._create_sample_users()

        except Exception as e:
            logger.error(f"Failed to load beta users: {e}")
            self._create_sample_users()

    def _determine_engagement_level(self, user_data: dict[str, Any]) -> EngagementLevel:
        """Determine user engagement level based on activity data."""
        usage_metrics = user_data.get("usage_metrics", {})
        status = user_data.get("status", "invited")
        last_active = user_data.get("last_active")

        # Check activity recency
        if last_active:
            days_since_active = (
                datetime.now() - datetime.fromisoformat(last_active)
            ).days
        else:
            days_since_active = 999  # Very inactive

        # Calculate engagement score
        engagement_score = 0

        if status == "active":
            engagement_score += 40
        elif status == "onboarded":
            engagement_score += 20
        elif status == "registered":
            engagement_score += 10

        # Usage metrics scoring
        datasets_uploaded = usage_metrics.get("datasets_uploaded", 0)
        detections_run = usage_metrics.get("detections_run", 0)
        api_calls = usage_metrics.get("api_calls_made", 0)
        time_spent = usage_metrics.get("time_spent_minutes", 0)

        if datasets_uploaded > 5:
            engagement_score += 15
        elif datasets_uploaded > 2:
            engagement_score += 10
        elif datasets_uploaded > 0:
            engagement_score += 5

        if detections_run > 50:
            engagement_score += 15
        elif detections_run > 20:
            engagement_score += 10
        elif detections_run > 5:
            engagement_score += 5

        if api_calls > 200:
            engagement_score += 10
        elif api_calls > 50:
            engagement_score += 5

        if time_spent > 400:
            engagement_score += 10
        elif time_spent > 120:
            engagement_score += 5

        # Recency penalty
        if days_since_active > 7:
            engagement_score -= 20
        elif days_since_active > 3:
            engagement_score -= 10

        # Determine level
        if engagement_score >= 70:
            return EngagementLevel.HIGH
        elif engagement_score >= 40:
            return EngagementLevel.MEDIUM
        elif engagement_score >= 20:
            return EngagementLevel.LOW
        else:
            return EngagementLevel.INACTIVE

    def _create_sample_users(self):
        """Create sample beta users for demonstration."""
        sample_users = [
            {
                "user_id": "sarah_chen",
                "email": "sarah.chen@fintechcorp.com",
                "name": "Sarah Chen",
                "company": "FinTech Corp",
                "industry": "fintech",
                "role": "Risk Manager",
                "engagement_level": EngagementLevel.HIGH,
                "usage_metrics": {
                    "datasets_uploaded": 8,
                    "detections_run": 67,
                    "api_calls_made": 234,
                    "time_spent_minutes": 520,
                },
            },
            {
                "user_id": "david_park",
                "email": "david.park@automaker.com",
                "name": "David Park",
                "company": "AutoMaker Inc",
                "industry": "manufacturing",
                "role": "Quality Engineer",
                "engagement_level": EngagementLevel.MEDIUM,
                "usage_metrics": {
                    "datasets_uploaded": 4,
                    "detections_run": 23,
                    "api_calls_made": 89,
                    "time_spent_minutes": 180,
                },
            },
            {
                "user_id": "dr_smith",
                "email": "dr.smith@hospital.org",
                "name": "Dr. Smith",
                "company": "City General Hospital",
                "industry": "healthcare",
                "role": "Chief Medical Officer",
                "engagement_level": EngagementLevel.LOW,
                "usage_metrics": {
                    "datasets_uploaded": 2,
                    "detections_run": 8,
                    "api_calls_made": 34,
                    "time_spent_minutes": 90,
                },
            },
        ]

        for user_data in sample_users:
            user_profile = BetaUserProfile(
                user_id=user_data["user_id"],
                email=user_data["email"],
                name=user_data["name"],
                company=user_data["company"],
                industry=user_data["industry"],
                role=user_data["role"],
                registration_date=datetime.now() - timedelta(days=14),
                last_active=datetime.now() - timedelta(days=1),
                engagement_level=user_data["engagement_level"],
                usage_metrics=user_data["usage_metrics"],
                notes="Sample beta user for demonstration",
            )
            self.users.append(user_profile)

        logger.info(f"📊 Created {len(self.users)} sample beta users")

    def schedule_weekly_checkins(self) -> list[CheckinRecord]:
        """Schedule weekly check-ins for all active beta users."""
        logger.info("📅 Scheduling weekly check-ins for beta users...")

        scheduled_checkins = []
        now = datetime.now()

        for user in self.users:
            # Determine check-in type based on engagement level and history
            checkin_type = self._determine_checkin_type(user)

            # Calculate next check-in date
            if user.last_checkin:
                days_since_checkin = (now - user.last_checkin).days
                if days_since_checkin < 7:
                    continue  # Skip if checked in recently
                next_checkin = user.last_checkin + timedelta(days=7)
            else:
                next_checkin = now + timedelta(days=1)  # Schedule soon for new users

            checkin_record = CheckinRecord(
                checkin_id=f"checkin_{user.user_id}_{int(time.time())}",
                user_id=user.user_id,
                checkin_type=checkin_type,
                scheduled_date=next_checkin,
                completed_date=None,
                response_received=False,
                satisfaction_score=None,
                issues_raised=[],
                follow_up_required=False,
                notes=f"Scheduled {checkin_type.value} check-in for {user.engagement_level.value} engagement user",
            )

            scheduled_checkins.append(checkin_record)
            self.checkin_records.append(checkin_record)

        logger.info(f"✅ Scheduled {len(scheduled_checkins)} check-ins")
        return scheduled_checkins

    def _determine_checkin_type(self, user: BetaUserProfile) -> CheckinType:
        """Determine appropriate check-in type for user."""
        days_since_registration = (datetime.now() - user.registration_date).days

        # New users get onboarding check-ins
        if days_since_registration < 7:
            return CheckinType.ONBOARDING

        # Inactive users get personal attention
        if user.engagement_level == EngagementLevel.INACTIVE:
            return CheckinType.PERSONAL

        # Users with low feedback get feedback-focused check-ins
        if user.feedback_count < 2 and days_since_registration > 14:
            return CheckinType.FEEDBACK

        # Users with support tickets get support check-ins
        if user.support_tickets > 0:
            return CheckinType.SUPPORT

        # Default to automated check-in
        return CheckinType.AUTOMATED

    async def conduct_automated_checkin(self, checkin_record: CheckinRecord) -> bool:
        """Conduct automated check-in via email."""
        user = next(
            (u for u in self.users if u.user_id == checkin_record.user_id), None
        )
        if not user:
            logger.error(f"User not found: {checkin_record.user_id}")
            return False

        logger.info(f"📧 Conducting automated check-in with {user.email}")

        # Generate personalized check-in email
        email_content = self._generate_checkin_email(user, checkin_record)

        # Simulate sending email (in production, would use actual email service)
        success = await self._send_checkin_email(user.email, email_content)

        if success:
            checkin_record.completed_date = datetime.now()
            user.last_checkin = datetime.now()
            checkin_record.next_checkin = datetime.now() + timedelta(days=7)

            # Simulate response (in production, would track actual responses)
            await asyncio.sleep(0.5)  # Simulate email processing time
            response_success = await self._simulate_user_response(user, checkin_record)

            return response_success

        return False

    def _generate_checkin_email(
        self, user: BetaUserProfile, checkin_record: CheckinRecord
    ) -> dict[str, str]:
        """Generate personalized check-in email content."""

        # Base email templates by check-in type
        templates = {
            CheckinType.AUTOMATED: {
                "subject": f"📊 Weekly Check-in: How's your Pynomaly experience, {user.name}?",
                "greeting": f"Hi {user.name},",
                "main_content": """
Hope you're having a great week! As part of our beta program, we'd love to check in and see how Pynomaly is working for your {industry} use cases.

**Quick Check-in Questions:**
1. How satisfied are you with Pynomaly this week? (1-5 scale)
2. Any blockers or issues we can help with?
3. What feature would be most valuable for your next project?

**Your Recent Activity:**
- Datasets processed: {datasets}
- Detections run: {detections}
- Time saved: ~{time_saved} hours

Reply to this email with your thoughts, or schedule a quick 15-minute call if you prefer to chat!
                """,
                "cta": "💬 Reply with feedback or 📞 Schedule a call",
            },
            CheckinType.ONBOARDING: {
                "subject": f"🚀 Welcome to Pynomaly Beta! How's your first week, {user.name}?",
                "greeting": f"Welcome {user.name}!",
                "main_content": """
Welcome to the Pynomaly beta program! We're excited to have {company} as part of our beta community.

**Getting Started Check:**
- Have you uploaded your first dataset?
- Any questions about the {industry} specific features?
- Need help with API integration or dashboard setup?

**Resources for {industry} Teams:**
- Industry-specific examples: [link]
- Best practices guide: [link]
- 1-on-1 onboarding call: [link]

Don't hesitate to reach out with any questions - we're here to help you succeed!
                """,
                "cta": "📚 Access resources or 🤝 Schedule onboarding call",
            },
            CheckinType.PERSONAL: {
                "subject": "👋 Personal Check-in: Let's get you back on track with Pynomaly",
                "greeting": f"Hi {user.name},",
                "main_content": """
I noticed you haven't been as active on Pynomaly lately, and I wanted to personally reach out to see how we can better support you.

**Let's troubleshoot together:**
- Are there any blockers preventing you from using Pynomaly?
- Would a personalized demo for {industry} use cases be helpful?
- Any specific features missing for your workflow?

**I'm here to help:**
As a beta user, you have direct access to our product team. Let's schedule a quick call to understand your needs better and ensure Pynomaly delivers value for {company}.
                """,
                "cta": "📞 Schedule a personal consultation",
            },
            CheckinType.FEEDBACK: {
                "subject": f"💭 Your Feedback Shapes Pynomaly's Future - Quick Survey for {user.name}",
                "greeting": f"Hi {user.name},",
                "main_content": """
Your expertise in {industry} is invaluable to us! We'd love to gather your feedback to prioritize our development roadmap.

**Quick 2-Minute Survey:**
- Overall satisfaction with Pynomaly
- Most valuable features for {industry}
- Feature requests and priorities
- Integration challenges or needs

**Your Impact:**
Previous beta feedback has led to:
✅ Improved {industry}-specific algorithms
✅ Better API documentation
✅ Enhanced dashboard performance

As a thank you, we'll send you early access to new features and a case study template for {company}.
                """,
                "cta": "📝 Complete 2-minute survey",
            },
            CheckinType.SUPPORT: {
                "subject": "🔧 Support Follow-up: Resolving your Pynomaly issues",
                "greeting": f"Hi {user.name},",
                "main_content": """
Following up on your recent support request. I want to ensure we've fully resolved your issues and that Pynomaly is working smoothly for {company}.

**Support Summary:**
- Issues raised: {support_count} tickets
- Resolution status: [tracking]
- Satisfaction with support: [pending]

**Next Steps:**
- Verify all issues are resolved
- Prevent similar issues in the future
- Improve your overall experience

Let's schedule a quick call to review everything and ensure you're getting maximum value from Pynomaly.
                """,
                "cta": "✅ Confirm resolution or 📞 Schedule follow-up",
            },
        }

        template = templates.get(
            checkin_record.checkin_type, templates[CheckinType.AUTOMATED]
        )

        # Format template with user data
        formatted_content = template["main_content"].format(
            industry=user.industry.title(),
            company=user.company,
            datasets=user.usage_metrics.get("datasets_uploaded", 0),
            detections=user.usage_metrics.get("detections_run", 0),
            time_saved=round(user.usage_metrics.get("time_spent_minutes", 0) / 60, 1),
            support_count=user.support_tickets,
        )

        return {
            "subject": template["subject"],
            "greeting": template["greeting"],
            "content": formatted_content,
            "cta": template["cta"],
            "signature": """
Best regards,
The Pynomaly Team

P.S. As a beta user, you have our direct email for any urgent questions: beta-support@monorepo.io
            """,
        }

    async def _send_checkin_email(self, email: str, content: dict[str, str]) -> bool:
        """Send check-in email (simulated)."""
        try:
            # In production, would use actual email service (SendGrid, SES, etc.)
            logger.info(f"📧 Sending check-in email to {email}")
            logger.debug(f"Subject: {content['subject']}")

            # Simulate email sending delay
            await asyncio.sleep(0.2)

            # Simulate 95% success rate
            if random.random() < 0.95:
                logger.info(f"✅ Email sent successfully to {email}")
                return True
            else:
                logger.warning(f"⚠️ Email failed to send to {email}")
                return False

        except Exception as e:
            logger.error(f"❌ Email sending error for {email}: {e}")
            return False

    async def _simulate_user_response(
        self, user: BetaUserProfile, checkin_record: CheckinRecord
    ) -> bool:
        """Simulate user response to check-in (for demo purposes)."""

        # Response rate based on engagement level
        response_rates = {
            EngagementLevel.HIGH: 0.8,
            EngagementLevel.MEDIUM: 0.6,
            EngagementLevel.LOW: 0.3,
            EngagementLevel.INACTIVE: 0.1,
        }

        response_rate = response_rates[user.engagement_level]

        if random.random() < response_rate:
            checkin_record.response_received = True

            # Generate simulated response data
            checkin_record.satisfaction_score = (
                random.randint(3, 5)
                if user.engagement_level
                in [EngagementLevel.HIGH, EngagementLevel.MEDIUM]
                else random.randint(2, 4)
            )

            # Simulate issues based on engagement level
            potential_issues = [
                "API documentation needs improvement",
                "Dashboard loading performance",
                "More visualization options needed",
                "Integration with existing tools",
                "Custom alert thresholds",
                "Export functionality enhancement",
            ]

            if user.engagement_level == EngagementLevel.LOW:
                checkin_record.issues_raised = random.sample(
                    potential_issues, random.randint(1, 2)
                )
                checkin_record.follow_up_required = True
            elif user.engagement_level == EngagementLevel.MEDIUM:
                if random.random() < 0.4:
                    checkin_record.issues_raised = random.sample(potential_issues, 1)
                    checkin_record.follow_up_required = random.random() < 0.3

            checkin_record.notes += (
                f" | Response: Satisfaction {checkin_record.satisfaction_score}/5"
            )
            if checkin_record.issues_raised:
                checkin_record.notes += (
                    f" | Issues: {', '.join(checkin_record.issues_raised)}"
                )

            logger.info(
                f"📝 Received response from {user.email}: {checkin_record.satisfaction_score}/5 satisfaction"
            )
            return True
        else:
            logger.info(f"📭 No response from {user.email}")
            return False

    async def conduct_all_scheduled_checkins(self) -> dict[str, Any]:
        """Conduct all scheduled check-ins."""
        scheduled_checkins = self.schedule_weekly_checkins()

        if not scheduled_checkins:
            logger.info("📭 No check-ins scheduled")
            return {"total": 0, "completed": 0, "response_rate": 0}

        logger.info(f"🚀 Starting {len(scheduled_checkins)} check-ins...")

        completed_count = 0
        response_count = 0

        for checkin in scheduled_checkins:
            success = await self.conduct_automated_checkin(checkin)
            if success:
                completed_count += 1
                if checkin.response_received:
                    response_count += 1

            # Small delay between check-ins
            await asyncio.sleep(0.1)

        response_rate = (
            (response_count / len(scheduled_checkins)) * 100
            if scheduled_checkins
            else 0
        )

        results = {
            "total_scheduled": len(scheduled_checkins),
            "completed": completed_count,
            "responses_received": response_count,
            "response_rate": round(response_rate, 1),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"✅ Check-ins completed: {completed_count}/{len(scheduled_checkins)} | Response rate: {response_rate:.1f}%"
        )

        return results

    def analyze_checkin_results(self) -> dict[str, Any]:
        """Analyze check-in results and user engagement."""

        if not self.checkin_records:
            return {"message": "No check-in data available"}

        recent_checkins = [
            c
            for c in self.checkin_records
            if c.completed_date and (datetime.now() - c.completed_date).days <= 7
        ]

        analysis = {
            "summary": {
                "total_checkins": len(recent_checkins),
                "response_rate": len(
                    [c for c in recent_checkins if c.response_received]
                )
                / len(recent_checkins)
                * 100
                if recent_checkins
                else 0,
                "avg_satisfaction": sum(
                    c.satisfaction_score
                    for c in recent_checkins
                    if c.satisfaction_score
                )
                / len([c for c in recent_checkins if c.satisfaction_score])
                if any(c.satisfaction_score for c in recent_checkins)
                else 0,
                "follow_ups_needed": len(
                    [c for c in recent_checkins if c.follow_up_required]
                ),
            },
            "engagement_trends": {},
            "common_issues": {},
            "user_segments": {},
            "recommendations": [],
        }

        # Engagement trends by level
        for level in EngagementLevel:
            level_users = [u for u in self.users if u.engagement_level == level]
            level_checkins = [
                c
                for c in recent_checkins
                if any(
                    u.user_id == c.user_id and u.engagement_level == level
                    for u in self.users
                )
            ]

            if level_checkins:
                analysis["engagement_trends"][level.value] = {
                    "user_count": len(level_users),
                    "checkin_count": len(level_checkins),
                    "response_rate": len(
                        [c for c in level_checkins if c.response_received]
                    )
                    / len(level_checkins)
                    * 100,
                    "avg_satisfaction": sum(
                        c.satisfaction_score
                        for c in level_checkins
                        if c.satisfaction_score
                    )
                    / len([c for c in level_checkins if c.satisfaction_score])
                    if any(c.satisfaction_score for c in level_checkins)
                    else 0,
                }

        # Common issues analysis
        all_issues = []
        for checkin in recent_checkins:
            all_issues.extend(checkin.issues_raised)

        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        analysis["common_issues"] = dict(
            sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )

        # Generate recommendations
        recommendations = []

        if analysis["summary"]["response_rate"] < 50:
            recommendations.append(
                "🔔 Low response rate - consider personalizing check-in messages"
            )

        if analysis["summary"]["avg_satisfaction"] < 3.5:
            recommendations.append(
                "⚠️ Low satisfaction scores - schedule personal calls with unsatisfied users"
            )

        if analysis["summary"]["follow_ups_needed"] > len(recent_checkins) * 0.3:
            recommendations.append(
                "📞 High follow-up rate - consider improving onboarding process"
            )

        if "API documentation needs improvement" in analysis["common_issues"]:
            recommendations.append(
                "📚 API documentation is a common issue - prioritize docs improvement"
            )

        if (
            analysis["engagement_trends"].get("inactive", {}).get("user_count", 0)
            > len(self.users) * 0.2
        ):
            recommendations.append(
                "😴 High inactive user count - implement re-engagement campaign"
            )

        analysis["recommendations"] = recommendations

        return analysis

    def generate_checkin_report(self) -> dict[str, Any]:
        """Generate comprehensive check-in report."""
        analysis = self.analyze_checkin_results()

        report = {
            "report_id": f"checkin_report_{int(time.time())}",
            "generated_at": datetime.now().isoformat(),
            "period": "Last 7 days",
            "user_overview": {
                "total_beta_users": len(self.users),
                "engagement_distribution": {
                    level.value: len(
                        [u for u in self.users if u.engagement_level == level]
                    )
                    for level in EngagementLevel
                },
            },
            "checkin_analysis": analysis,
            "user_profiles": [asdict(user) for user in self.users],
            "checkin_records": [asdict(record) for record in self.checkin_records],
            "next_steps": [
                "Continue weekly check-ins with all active users",
                "Implement personal outreach for inactive users",
                "Address common issues identified in feedback",
                "Track satisfaction trends over time",
                "Expand beta program to new user segments",
            ],
        }

        return report


async def main():
    """Main check-in system execution."""
    project_root = Path(__file__).parent.parent.parent
    checkin_system = BetaUserCheckinSystem(project_root)

    print("👥 Beta User Check-in System")
    print("=" * 50)

    # Conduct weekly check-ins
    results = await checkin_system.conduct_all_scheduled_checkins()

    print("\n📊 Check-in Results:")
    print(f"  📧 Emails sent: {results['completed']}/{results['total_scheduled']}")
    print(f"  📝 Responses received: {results['responses_received']}")
    print(f"  📈 Response rate: {results['response_rate']}%")

    # Generate analysis
    analysis = checkin_system.analyze_checkin_results()

    print("\n🎯 Engagement Analysis:")
    print(
        f"  ⭐ Average satisfaction: {analysis['summary']['avg_satisfaction']:.1f}/5.0"
    )
    print(f"  📞 Follow-ups needed: {analysis['summary']['follow_ups_needed']}")

    if analysis["common_issues"]:
        print("\n🔍 Top Issues:")
        for issue, count in list(analysis["common_issues"].items())[:3]:
            print(f"  • {issue}: {count} mentions")

    print("\n💡 Recommendations:")
    for rec in analysis["recommendations"][:3]:
        print(f"  • {rec}")

    # Save report
    report = checkin_system.generate_checkin_report()
    report_file = project_root / f"beta_checkin_report_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📄 Full report saved to: {report_file}")

    return report


if __name__ == "__main__":
    asyncio.run(main())
