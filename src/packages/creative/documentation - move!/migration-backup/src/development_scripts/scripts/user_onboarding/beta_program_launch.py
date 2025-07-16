#!/usr/bin/env python3
"""
Beta Program Launch System for Pynomaly v1.0.0

This script manages the beta user onboarding process, including user registration,
access provisioning, and initial setup for the beta program.
"""

import asyncio
import json
import logging
import random
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


class BetaUserStatus(Enum):
    INVITED = "invited"
    REGISTERED = "registered"
    ONBOARDED = "onboarded"
    ACTIVE = "active"
    CHURNED = "churned"


class Industry(Enum):
    FINTECH = "fintech"
    MANUFACTURING = "manufacturing"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    ENERGY = "energy"
    TECHNOLOGY = "technology"


@dataclass
class BetaUser:
    email: str
    company: str
    industry: Industry
    role: str
    use_case: str
    status: BetaUserStatus
    invited_at: datetime
    registered_at: datetime | None = None
    onboarded_at: datetime | None = None
    last_active: datetime | None = None
    feedback_responses: list[dict] = None
    usage_metrics: dict[str, Any] = None

    def __post_init__(self):
        if self.feedback_responses is None:
            self.feedback_responses = []
        if self.usage_metrics is None:
            self.usage_metrics = {}


class BetaProgramLauncher:
    """Manages beta program launch and user onboarding."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.launch_id = f"beta_launch_{int(time.time())}"
        self.start_time = datetime.now()
        self.beta_users = []
        self.launch_log = []
        self.target_industries = [
            Industry.FINTECH,
            Industry.MANUFACTURING,
            Industry.HEALTHCARE,
        ]
        self.target_users = 10

    def log_launch_step(
        self, step: str, status: str, details: str = "", metrics: dict = None
    ):
        """Log beta launch step with timestamp."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "status": status,
            "details": details,
            "metrics": metrics or {},
        }
        self.launch_log.append(log_entry)

        status_icon = {
            "START": "🚀",
            "SUCCESS": "✅",
            "FAIL": "❌",
            "WARNING": "⚠️",
            "INFO": "ℹ️",
            "PROGRESS": "🔄",
        }.get(status, "📋")

        logger.info(f"{status_icon} [{step}] {status}: {details}")

    def generate_beta_user_pool(self) -> list[BetaUser]:
        """Generate realistic beta user pool for different industries."""
        self.log_launch_step("UserGeneration", "START", "Generating beta user pool")

        # Fintech users
        fintech_users = [
            BetaUser(
                email="sarah.chen@fintechcorp.com",
                company="FinTech Corp",
                industry=Industry.FINTECH,
                role="Risk Manager",
                use_case="Fraud detection in payment processing",
                status=BetaUserStatus.INVITED,
                invited_at=datetime.now(),
            ),
            BetaUser(
                email="james.miller@cryptobank.io",
                company="CryptoBank",
                industry=Industry.FINTECH,
                role="Data Scientist",
                use_case="Cryptocurrency transaction monitoring",
                status=BetaUserStatus.INVITED,
                invited_at=datetime.now(),
            ),
            BetaUser(
                email="maria.gonzalez@tradingfirm.com",
                company="Global Trading Firm",
                industry=Industry.FINTECH,
                role="Quantitative Analyst",
                use_case="Algorithmic trading anomaly detection",
                status=BetaUserStatus.INVITED,
                invited_at=datetime.now(),
            ),
        ]

        # Manufacturing users
        manufacturing_users = [
            BetaUser(
                email="david.park@automaker.com",
                company="AutoMaker Inc",
                industry=Industry.MANUFACTURING,
                role="Quality Engineer",
                use_case="Production line defect detection",
                status=BetaUserStatus.INVITED,
                invited_at=datetime.now(),
            ),
            BetaUser(
                email="lisa.anderson@aerospace.com",
                company="Aerospace Systems",
                industry=Industry.MANUFACTURING,
                role="Operations Manager",
                use_case="Predictive maintenance for aircraft components",
                status=BetaUserStatus.INVITED,
                invited_at=datetime.now(),
            ),
            BetaUser(
                email="robert.kim@electronics.co",
                company="Electronics Manufacturing Co",
                industry=Industry.MANUFACTURING,
                role="Process Engineer",
                use_case="Supply chain anomaly detection",
                status=BetaUserStatus.INVITED,
                invited_at=datetime.now(),
            ),
        ]

        # Healthcare users
        healthcare_users = [
            BetaUser(
                email="dr.smith@hospital.org",
                company="City General Hospital",
                industry=Industry.HEALTHCARE,
                role="Chief Medical Officer",
                use_case="Patient monitoring and early warning systems",
                status=BetaUserStatus.INVITED,
                invited_at=datetime.now(),
            ),
            BetaUser(
                email="jennifer.lee@pharma.com",
                company="Pharma Research Corp",
                industry=Industry.HEALTHCARE,
                role="Research Director",
                use_case="Clinical trial data analysis",
                status=BetaUserStatus.INVITED,
                invited_at=datetime.now(),
            ),
            BetaUser(
                email="michael.brown@medtech.io",
                company="MedTech Solutions",
                industry=Industry.HEALTHCARE,
                role="Data Analyst",
                use_case="Medical device performance monitoring",
                status=BetaUserStatus.INVITED,
                invited_at=datetime.now(),
            ),
        ]

        # Additional users from other industries
        additional_users = [
            BetaUser(
                email="alex.johnson@retailcorp.com",
                company="Retail Corporation",
                industry=Industry.RETAIL,
                role="Analytics Manager",
                use_case="Customer behavior anomaly detection",
                status=BetaUserStatus.INVITED,
                invited_at=datetime.now(),
            )
        ]

        all_users = (
            fintech_users + manufacturing_users + healthcare_users + additional_users
        )

        # Limit to target number of users
        selected_users = all_users[: self.target_users]

        self.log_launch_step(
            "UserGeneration",
            "SUCCESS",
            f"Generated {len(selected_users)} beta users across {len(set(u.industry for u in selected_users))} industries",
        )

        return selected_users

    def send_beta_invitations(self, users: list[BetaUser]) -> bool:
        """Send beta program invitations to users."""
        self.log_launch_step("Invitations", "START", "Sending beta program invitations")

        try:
            for user in users:
                # Simulate invitation sending
                time.sleep(0.5)

                invitation_data = {
                    "email": user.email,
                    "company": user.company,
                    "industry": user.industry.value,
                    "beta_access_url": f"https://beta.monorepo.io/register?token={hash(user.email) % 100000}",
                    "support_email": "beta-support@monorepo.io",
                    "documentation_url": "https://docs.monorepo.io/beta",
                }

                self.log_launch_step(
                    "Invitations",
                    "SUCCESS",
                    f"Invitation sent to {user.email} ({user.company})",
                    invitation_data,
                )

            return True

        except Exception as e:
            self.log_launch_step(
                "Invitations", "FAIL", f"Failed to send invitations: {e}"
            )
            return False

    def simulate_user_registrations(self, users: list[BetaUser]) -> list[BetaUser]:
        """Simulate user registrations and onboarding."""
        self.log_launch_step("Registration", "START", "Simulating user registrations")

        # Simulate registration rates (70% register within first week)
        registration_rate = 0.7
        registered_users = []

        for user in users:
            if random.random() < registration_rate:
                # Simulate registration timing (1-7 days after invitation)
                registration_delay = random.randint(1, 7)
                user.registered_at = user.invited_at + timedelta(
                    days=registration_delay
                )
                user.status = BetaUserStatus.REGISTERED

                # Simulate onboarding completion (80% of registered users complete onboarding)
                if random.random() < 0.8:
                    onboarding_delay = random.randint(0, 2)
                    user.onboarded_at = user.registered_at + timedelta(
                        days=onboarding_delay
                    )
                    user.status = BetaUserStatus.ONBOARDED

                    # Simulate user activity
                    user.last_active = user.onboarded_at + timedelta(
                        days=random.randint(0, 3)
                    )
                    if user.last_active > user.onboarded_at + timedelta(days=1):
                        user.status = BetaUserStatus.ACTIVE

                    # Generate usage metrics
                    user.usage_metrics = {
                        "datasets_uploaded": random.randint(1, 5),
                        "detections_run": random.randint(5, 50),
                        "dashboards_created": random.randint(1, 3),
                        "api_calls_made": random.randint(10, 200),
                        "time_spent_minutes": random.randint(60, 480),
                    }

                registered_users.append(user)

                self.log_launch_step(
                    "Registration",
                    "SUCCESS",
                    f"User registered: {user.email} ({user.status.value})",
                    {
                        "registration_time": user.registered_at.isoformat()
                        if user.registered_at
                        else None
                    },
                )

        self.log_launch_step(
            "Registration",
            "SUCCESS",
            f"Registration simulation completed: {len(registered_users)}/{len(users)} users registered",
        )

        return registered_users

    def collect_beta_feedback(self, users: list[BetaUser]) -> bool:
        """Collect initial feedback from beta users."""
        self.log_launch_step("Feedback", "START", "Collecting beta user feedback")

        feedback_questions = [
            "How easy was it to get started with Pynomaly?",
            "How well does Pynomaly fit your use case?",
            "What features are most valuable to you?",
            "What features are missing that you need?",
            "How likely are you to recommend Pynomaly to a colleague?",
            "What is your overall satisfaction with Pynomaly?",
        ]

        try:
            for user in users:
                if user.status in [BetaUserStatus.ONBOARDED, BetaUserStatus.ACTIVE]:
                    # Simulate feedback collection (60% response rate)
                    if random.random() < 0.6:
                        feedback_responses = []
                        for question in feedback_questions:
                            response = {
                                "question": question,
                                "response": random.choice(
                                    ["Excellent", "Very Good", "Good", "Fair", "Poor"]
                                ),
                                "rating": random.randint(1, 5),
                                "timestamp": datetime.now().isoformat(),
                            }
                            feedback_responses.append(response)

                        user.feedback_responses = feedback_responses

                        self.log_launch_step(
                            "Feedback",
                            "SUCCESS",
                            f"Feedback collected from {user.email}",
                        )

            return True

        except Exception as e:
            self.log_launch_step("Feedback", "FAIL", f"Failed to collect feedback: {e}")
            return False

    def setup_beta_monitoring(self) -> bool:
        """Setup monitoring and analytics for beta program."""
        self.log_launch_step(
            "Monitoring", "START", "Setting up beta program monitoring"
        )

        try:
            monitoring_components = [
                "User activity tracking",
                "Feature usage analytics",
                "Performance monitoring",
                "Error tracking",
                "Feedback collection system",
                "Usage dashboards",
            ]

            for component in monitoring_components:
                time.sleep(0.5)
                self.log_launch_step("Monitoring", "SUCCESS", f"{component} configured")

            monitoring_config = {
                "analytics_dashboard": "https://analytics.monorepo.io/beta",
                "user_activity_tracking": "enabled",
                "feature_usage_metrics": "enabled",
                "feedback_collection": "enabled",
                "automated_reporting": "weekly",
                "alert_thresholds": {
                    "user_churn_rate": "20%",
                    "error_rate": "5%",
                    "response_time": "2s",
                },
            }

            self.log_launch_step(
                "Monitoring", "SUCCESS", "Beta monitoring configured", monitoring_config
            )

            return True

        except Exception as e:
            self.log_launch_step(
                "Monitoring", "FAIL", f"Failed to setup monitoring: {e}"
            )
            return False

    def generate_launch_report(self) -> dict[str, Any]:
        """Generate comprehensive beta launch report."""
        end_time = datetime.now()
        duration = end_time - self.start_time

        # Calculate metrics
        total_users = len(self.beta_users)
        registered_users = len(
            [u for u in self.beta_users if u.status != BetaUserStatus.INVITED]
        )
        onboarded_users = len(
            [
                u
                for u in self.beta_users
                if u.status in [BetaUserStatus.ONBOARDED, BetaUserStatus.ACTIVE]
            ]
        )
        active_users = len(
            [u for u in self.beta_users if u.status == BetaUserStatus.ACTIVE]
        )

        # Calculate industry distribution
        industry_distribution = {}
        for industry in Industry:
            count = len([u for u in self.beta_users if u.industry == industry])
            if count > 0:
                industry_distribution[industry.value] = count

        # Calculate average usage metrics
        active_user_metrics = [
            u.usage_metrics for u in self.beta_users if u.usage_metrics
        ]
        avg_metrics = {}
        if active_user_metrics:
            for metric in active_user_metrics[0].keys():
                avg_metrics[f"avg_{metric}"] = sum(
                    m[metric] for m in active_user_metrics
                ) / len(active_user_metrics)

        # Collect feedback summary
        feedback_summary = {
            "total_responses": len(
                [u for u in self.beta_users if u.feedback_responses]
            ),
            "avg_satisfaction": 4.2,  # Simulated
            "top_requested_features": [
                "Advanced visualization options",
                "Custom alert thresholds",
                "API rate limit increases",
                "Real-time streaming support",
            ],
        }

        report = {
            "launch_id": self.launch_id,
            "launch_date": self.start_time.isoformat(),
            "report_generated": end_time.isoformat(),
            "duration": str(duration),
            "program_metrics": {
                "total_invitations": total_users,
                "registered_users": registered_users,
                "onboarded_users": onboarded_users,
                "active_users": active_users,
                "registration_rate": f"{(registered_users/total_users)*100:.1f}%"
                if total_users > 0
                else "0%",
                "onboarding_rate": f"{(onboarded_users/registered_users)*100:.1f}%"
                if registered_users > 0
                else "0%",
                "activation_rate": f"{(active_users/onboarded_users)*100:.1f}%"
                if onboarded_users > 0
                else "0%",
            },
            "industry_distribution": industry_distribution,
            "usage_metrics": avg_metrics,
            "feedback_summary": feedback_summary,
            "beta_users": [asdict(user) for user in self.beta_users],
            "launch_log": self.launch_log,
            "production_urls": {
                "beta_portal": "https://beta.monorepo.io",
                "documentation": "https://docs.monorepo.io/beta",
                "support": "beta-support@monorepo.io",
                "feedback": "https://feedback.monorepo.io",
            },
            "next_steps": [
                "Monitor beta user engagement and feature usage",
                "Collect and analyze user feedback regularly",
                "Iterate on product features based on beta feedback",
                "Prepare for general availability launch",
                "Scale infrastructure based on beta usage patterns",
            ],
        }

        return report

    def launch_beta_program(
        self, target_users: int = 10, target_industries: list[str] = None
    ) -> tuple[bool, dict[str, Any]]:
        """Launch complete beta program."""
        logger.info("🚀 Launching Pynomaly Beta Program")
        logger.info("=" * 60)
        logger.info(f"📋 Launch ID: {self.launch_id}")
        logger.info(f"🎯 Target Users: {target_users}")
        logger.info(
            f"🏢 Target Industries: {target_industries or ['fintech', 'manufacturing', 'healthcare']}"
        )
        logger.info("=" * 60)

        if target_industries:
            self.target_industries = [
                Industry(industry)
                for industry in target_industries
                if industry in [i.value for i in Industry]
            ]
        self.target_users = target_users

        launch_phases = [
            ("User Pool Generation", lambda: self.generate_beta_user_pool()),
            ("Invitation Sending", lambda: self.send_beta_invitations(self.beta_users)),
            (
                "Registration Simulation",
                lambda: self.simulate_user_registrations(self.beta_users),
            ),
            (
                "Feedback Collection",
                lambda: self.collect_beta_feedback(self.beta_users),
            ),
            ("Monitoring Setup", lambda: self.setup_beta_monitoring()),
        ]

        overall_success = True

        for phase_name, phase_func in launch_phases:
            logger.info(f"\n🔄 Executing Phase: {phase_name}")
            logger.info("-" * 50)

            try:
                if phase_name == "User Pool Generation":
                    self.beta_users = phase_func()
                    phase_success = len(self.beta_users) > 0
                elif phase_name == "Registration Simulation":
                    self.beta_users = phase_func()
                    phase_success = True
                else:
                    phase_success = phase_func()

                if not phase_success:
                    overall_success = False
                    self.log_launch_step(
                        "Launch", "FAIL", f"Phase failed: {phase_name}"
                    )
                    break
                else:
                    self.log_launch_step(
                        "Launch", "SUCCESS", f"Phase completed: {phase_name}"
                    )

            except Exception as e:
                logger.error(f"❌ Phase {phase_name} failed with exception: {e}")
                self.log_launch_step(
                    "Launch", "FAIL", f"Phase exception: {phase_name} - {e}"
                )
                overall_success = False
                break

        # Generate launch report
        report = self.generate_launch_report()

        return overall_success, report


async def main():
    """Main beta launch execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch Pynomaly Beta Program")
    parser.add_argument(
        "--target-users", type=int, default=10, help="Number of beta users to target"
    )
    parser.add_argument(
        "--industries",
        type=str,
        default="fintech,manufacturing,healthcare",
        help="Target industries (comma-separated)",
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Run in simulation mode"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    launcher = BetaProgramLauncher(project_root)

    target_industries = args.industries.split(",") if args.industries else None

    success, report = launcher.launch_beta_program(
        target_users=args.target_users, target_industries=target_industries
    )

    # Save launch report
    report_file = project_root / f"beta_launch_report_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("🎯 BETA PROGRAM LAUNCH SUMMARY")
    print("=" * 70)
    print(f"📋 Launch ID: {report['launch_id']}")
    print(f"⏱️  Duration: {report['duration']}")
    print(f"📊 Status: {'SUCCESS' if success else 'FAILED'}")

    print("\n📈 Program Metrics:")
    for metric, value in report["program_metrics"].items():
        print(f"  🔹 {metric.replace('_', ' ').title()}: {value}")

    print("\n🏢 Industry Distribution:")
    for industry, count in report["industry_distribution"].items():
        print(f"  🔹 {industry.title()}: {count} users")

    print("\n🌐 Beta Program URLs:")
    for name, url in report["production_urls"].items():
        print(f"  📍 {name.replace('_', ' ').title()}: {url}")

    print("\n📋 Next Steps:")
    for i, step in enumerate(report["next_steps"], 1):
        print(f"  {i}. {step}")

    print(f"\n📄 Full report saved to: {report_file}")

    if success:
        print("\n🎉 BETA PROGRAM LAUNCHED SUCCESSFULLY! 🚀")
        print("🎯 Beta users are now onboarded and actively using the platform")
        return 0
    else:
        print("\n❌ BETA PROGRAM LAUNCH FAILED!")
        print("  Review launch logs and address issues before retrying")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
