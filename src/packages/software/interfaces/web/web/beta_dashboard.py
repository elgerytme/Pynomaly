#!/usr/bin/env python3
"""
Beta Program Management Dashboard for Pynomaly

Provides comprehensive management interface for beta program operations,
user tracking, feedback analysis, and program metrics.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
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
class BetaMetrics:
    total_users: int
    active_users: int
    churned_users: int
    registration_rate: float
    activation_rate: float
    avg_session_time: float
    feature_adoption_rate: dict[str, float]
    satisfaction_score: float
    nps_score: float


class BetaDashboard:
    """Beta program management dashboard."""

    def __init__(self):
        self.dashboard_title = "🧪 Pynomaly Beta Program Dashboard"
        self.last_updated = datetime.now()

        # Initialize session state
        if "beta_data" not in st.session_state:
            st.session_state.beta_data = self.load_beta_data()

        if "selected_filters" not in st.session_state:
            st.session_state.selected_filters = {
                "industry": "All",
                "status": "All",
                "date_range": 30,
            }

    def load_beta_data(self) -> dict[str, Any]:
        """Load beta program data (simulated for demo)."""
        # In production, this would load from database or API
        return {
            "users": [
                {
                    "id": "user_001",
                    "email": "sarah.chen@fintechcorp.com",
                    "company": "FinTech Corp",
                    "industry": "fintech",
                    "role": "Risk Manager",
                    "status": "active",
                    "invited_at": "2025-07-05T10:00:00",
                    "registered_at": "2025-07-06T14:30:00",
                    "onboarded_at": "2025-07-06T16:00:00",
                    "last_active": "2025-07-10T09:15:00",
                    "usage_metrics": {
                        "datasets_uploaded": 5,
                        "detections_run": 42,
                        "dashboards_created": 3,
                        "api_calls_made": 156,
                        "time_spent_minutes": 420,
                    },
                    "satisfaction_score": 4.5,
                    "nps_score": 9,
                },
                {
                    "id": "user_002",
                    "email": "james.miller@cryptobank.io",
                    "company": "CryptoBank",
                    "industry": "fintech",
                    "role": "Data Scientist",
                    "status": "active",
                    "invited_at": "2025-07-05T10:00:00",
                    "registered_at": "2025-07-07T11:20:00",
                    "onboarded_at": "2025-07-07T12:45:00",
                    "last_active": "2025-07-10T08:30:00",
                    "usage_metrics": {
                        "datasets_uploaded": 8,
                        "detections_run": 63,
                        "dashboards_created": 5,
                        "api_calls_made": 284,
                        "time_spent_minutes": 680,
                    },
                    "satisfaction_score": 4.8,
                    "nps_score": 10,
                },
                {
                    "id": "user_003",
                    "email": "david.park@automaker.com",
                    "company": "AutoMaker Inc",
                    "industry": "manufacturing",
                    "role": "Quality Engineer",
                    "status": "onboarded",
                    "invited_at": "2025-07-05T10:00:00",
                    "registered_at": "2025-07-08T09:15:00",
                    "onboarded_at": "2025-07-08T11:30:00",
                    "last_active": "2025-07-09T16:45:00",
                    "usage_metrics": {
                        "datasets_uploaded": 3,
                        "detections_run": 18,
                        "dashboards_created": 2,
                        "api_calls_made": 87,
                        "time_spent_minutes": 240,
                    },
                    "satisfaction_score": 4.2,
                    "nps_score": 8,
                },
                {
                    "id": "user_004",
                    "email": "dr.smith@hospital.org",
                    "company": "City General Hospital",
                    "industry": "healthcare",
                    "role": "Chief Medical Officer",
                    "status": "active",
                    "invited_at": "2025-07-05T10:00:00",
                    "registered_at": "2025-07-06T13:20:00",
                    "onboarded_at": "2025-07-06T15:10:00",
                    "last_active": "2025-07-10T07:20:00",
                    "usage_metrics": {
                        "datasets_uploaded": 4,
                        "detections_run": 28,
                        "dashboards_created": 2,
                        "api_calls_made": 124,
                        "time_spent_minutes": 360,
                    },
                    "satisfaction_score": 4.3,
                    "nps_score": 8,
                },
                {
                    "id": "user_005",
                    "email": "lisa.anderson@aerospace.com",
                    "company": "Aerospace Systems",
                    "industry": "manufacturing",
                    "role": "Operations Manager",
                    "status": "registered",
                    "invited_at": "2025-07-05T10:00:00",
                    "registered_at": "2025-07-09T10:30:00",
                    "onboarded_at": None,
                    "last_active": "2025-07-09T11:00:00",
                    "usage_metrics": {
                        "datasets_uploaded": 1,
                        "detections_run": 3,
                        "dashboards_created": 0,
                        "api_calls_made": 12,
                        "time_spent_minutes": 45,
                    },
                    "satisfaction_score": 3.8,
                    "nps_score": 6,
                },
            ],
            "program_metrics": {
                "launch_date": "2025-07-05T10:00:00",
                "total_invitations": 10,
                "registered_users": 8,
                "onboarded_users": 6,
                "active_users": 4,
                "churned_users": 0,
            },
            "feature_usage": {
                "dataset_upload": 0.85,
                "anomaly_detection": 0.92,
                "dashboard_creation": 0.78,
                "api_integration": 0.65,
                "alert_configuration": 0.45,
                "export_functionality": 0.38,
            },
            "feedback_themes": [
                {"theme": "Ease of Use", "mentions": 12, "sentiment": "positive"},
                {"theme": "Feature Completeness", "mentions": 8, "sentiment": "mixed"},
                {"theme": "Performance", "mentions": 6, "sentiment": "positive"},
                {"theme": "Documentation", "mentions": 4, "sentiment": "mixed"},
                {"theme": "API Functionality", "mentions": 3, "sentiment": "positive"},
            ],
        }

    def calculate_metrics(self, data: dict[str, Any]) -> BetaMetrics:
        """Calculate key beta program metrics."""
        users = data["users"]
        program_metrics = data["program_metrics"]

        total_users = len(users)
        active_users = len([u for u in users if u["status"] == "active"])
        churned_users = len([u for u in users if u["status"] == "churned"])

        registration_rate = (
            program_metrics["registered_users"]
            / program_metrics["total_invitations"]
            * 100
        )
        activation_rate = (
            active_users / program_metrics["registered_users"] * 100
            if program_metrics["registered_users"] > 0
            else 0
        )

        # Calculate average session time
        total_time = sum(
            u["usage_metrics"]["time_spent_minutes"]
            for u in users
            if u["usage_metrics"]
        )
        avg_session_time = total_time / len(users) if users else 0

        # Calculate satisfaction metrics
        satisfaction_scores = [
            u["satisfaction_score"] for u in users if "satisfaction_score" in u
        ]
        nps_scores = [u["nps_score"] for u in users if "nps_score" in u]

        avg_satisfaction = (
            sum(satisfaction_scores) / len(satisfaction_scores)
            if satisfaction_scores
            else 0
        )
        avg_nps = sum(nps_scores) / len(nps_scores) if nps_scores else 0

        return BetaMetrics(
            total_users=total_users,
            active_users=active_users,
            churned_users=churned_users,
            registration_rate=registration_rate,
            activation_rate=activation_rate,
            avg_session_time=avg_session_time,
            feature_adoption_rate=data["feature_usage"],
            satisfaction_score=avg_satisfaction,
            nps_score=avg_nps,
        )

    def render_header(self):
        """Render dashboard header."""
        st.set_page_config(
            page_title="Pynomaly Beta Dashboard",
            page_icon="🧪",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title(self.dashboard_title)

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Real-time beta program monitoring and analytics**")
        with col2:
            st.metric("Last Updated", self.last_updated.strftime("%H:%M:%S"))
        with col3:
            if st.button("🔄 Refresh Data"):
                st.session_state.beta_data = self.load_beta_data()
                st.rerun()

    def render_sidebar(self):
        """Render sidebar filters and controls."""
        st.sidebar.title("📊 Filters & Controls")

        # Industry filter
        industries = ["All"] + list(
            set(u["industry"] for u in st.session_state.beta_data["users"])
        )
        st.session_state.selected_filters["industry"] = st.sidebar.selectbox(
            "Industry", industries, index=0
        )

        # Status filter
        statuses = ["All"] + list(
            set(u["status"] for u in st.session_state.beta_data["users"])
        )
        st.session_state.selected_filters["status"] = st.sidebar.selectbox(
            "User Status", statuses, index=0
        )

        # Date range filter
        st.session_state.selected_filters["date_range"] = st.sidebar.slider(
            "Date Range (days)", 1, 90, 30
        )

        st.sidebar.markdown("---")

        # Quick actions
        st.sidebar.title("🚀 Quick Actions")

        if st.sidebar.button("📧 Send Feedback Survey"):
            st.sidebar.success("Feedback survey sent to active users!")

        if st.sidebar.button("📊 Generate Report"):
            st.sidebar.success("Weekly report generated!")

        if st.sidebar.button("👥 Export User Data"):
            st.sidebar.success("User data exported to CSV!")

        st.sidebar.markdown("---")

        # Program status
        st.sidebar.title("📈 Program Status")
        metrics = self.calculate_metrics(st.session_state.beta_data)

        st.sidebar.metric("Total Users", metrics.total_users)
        st.sidebar.metric(
            "Active Users",
            metrics.active_users,
            f"+{metrics.active_users - metrics.churned_users}",
        )
        st.sidebar.metric("Registration Rate", f"{metrics.registration_rate:.1f}%")

    def render_overview_metrics(self):
        """Render overview metrics cards."""
        st.subheader("📈 Program Overview")

        metrics = self.calculate_metrics(st.session_state.beta_data)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Beta Users", metrics.total_users, delta=None)

        with col2:
            st.metric(
                "Active Users",
                metrics.active_users,
                delta=f"+{metrics.active_users - metrics.churned_users}",
            )

        with col3:
            st.metric(
                "Registration Rate", f"{metrics.registration_rate:.1f}%", delta=None
            )

        with col4:
            st.metric("Activation Rate", f"{metrics.activation_rate:.1f}%", delta=None)

        with col5:
            st.metric(
                "Avg Session Time", f"{metrics.avg_session_time:.0f}min", delta=None
            )

    def render_user_funnel(self):
        """Render user funnel visualization."""
        st.subheader("👥 User Funnel Analysis")

        program_metrics = st.session_state.beta_data["program_metrics"]

        funnel_data = {
            "Stage": ["Invited", "Registered", "Onboarded", "Active"],
            "Users": [
                program_metrics["total_invitations"],
                program_metrics["registered_users"],
                program_metrics["onboarded_users"],
                program_metrics["active_users"],
            ],
            "Conversion": [100, 80, 75, 50],  # Simulated conversion rates
        }

        fig = go.Figure()

        fig.add_trace(
            go.Funnel(
                y=funnel_data["Stage"],
                x=funnel_data["Users"],
                textinfo="value+percent initial",
                marker=dict(color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]),
            )
        )

        fig.update_layout(title="Beta User Conversion Funnel", height=400)

        st.plotly_chart(fig, use_container_width=True)

    def render_usage_analytics(self):
        """Render usage analytics charts."""
        st.subheader("📊 Usage Analytics")

        col1, col2 = st.columns(2)

        with col1:
            # Feature adoption chart
            feature_data = st.session_state.beta_data["feature_usage"]

            fig_features = px.bar(
                x=list(feature_data.keys()),
                y=[v * 100 for v in feature_data.values()],
                title="Feature Adoption Rates",
                labels={"y": "Adoption Rate (%)", "x": "Features"},
            )
            fig_features.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_features, use_container_width=True)

        with col2:
            # User activity heatmap (simulated)
            import numpy as np

            activity_data = np.random.rand(7, 24) * 100  # 7 days, 24 hours

            fig_heatmap = px.imshow(
                activity_data,
                x=[f"{i:02d}:00" for i in range(24)],
                y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                title="User Activity Heatmap",
                color_continuous_scale="Blues",
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)

    def render_industry_analysis(self):
        """Render industry analysis."""
        st.subheader("🏢 Industry Analysis")

        users = st.session_state.beta_data["users"]

        col1, col2 = st.columns(2)

        with col1:
            # Industry distribution
            industry_counts = {}
            for user in users:
                industry = user["industry"]
                industry_counts[industry] = industry_counts.get(industry, 0) + 1

            fig_pie = px.pie(
                values=list(industry_counts.values()),
                names=list(industry_counts.keys()),
                title="User Distribution by Industry",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Industry usage comparison
            industry_usage = {}
            for user in users:
                industry = user["industry"]
                if industry not in industry_usage:
                    industry_usage[industry] = []
                industry_usage[industry].append(
                    user["usage_metrics"]["time_spent_minutes"]
                )

            avg_usage = {
                industry: sum(times) / len(times)
                for industry, times in industry_usage.items()
            }

            fig_bar = px.bar(
                x=list(avg_usage.keys()),
                y=list(avg_usage.values()),
                title="Average Usage Time by Industry",
                labels={"y": "Average Time (minutes)", "x": "Industry"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    def render_feedback_analysis(self):
        """Render feedback and satisfaction analysis."""
        st.subheader("💬 Feedback Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Satisfaction scores
            users = st.session_state.beta_data["users"]
            satisfaction_scores = [
                u["satisfaction_score"] for u in users if "satisfaction_score" in u
            ]

            fig_hist = px.histogram(
                satisfaction_scores,
                nbins=10,
                title="User Satisfaction Distribution",
                labels={"x": "Satisfaction Score", "y": "Number of Users"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Feedback themes
            themes = st.session_state.beta_data["feedback_themes"]

            fig_themes = px.bar(
                x=[t["theme"] for t in themes],
                y=[t["mentions"] for t in themes],
                color=[t["sentiment"] for t in themes],
                title="Feedback Themes",
                labels={"y": "Mentions", "x": "Theme"},
            )
            fig_themes.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_themes, use_container_width=True)

    def render_user_table(self):
        """Render detailed user table."""
        st.subheader("👤 Beta User Details")

        users = st.session_state.beta_data["users"]

        # Apply filters
        filtered_users = users

        if st.session_state.selected_filters["industry"] != "All":
            filtered_users = [
                u
                for u in filtered_users
                if u["industry"] == st.session_state.selected_filters["industry"]
            ]

        if st.session_state.selected_filters["status"] != "All":
            filtered_users = [
                u
                for u in filtered_users
                if u["status"] == st.session_state.selected_filters["status"]
            ]

        # Create DataFrame
        user_data = []
        for user in filtered_users:
            user_data.append(
                {
                    "Email": user["email"],
                    "Company": user["company"],
                    "Industry": user["industry"].title(),
                    "Role": user["role"],
                    "Status": user["status"].title(),
                    "Datasets": user["usage_metrics"]["datasets_uploaded"],
                    "Detections": user["usage_metrics"]["detections_run"],
                    "API Calls": user["usage_metrics"]["api_calls_made"],
                    "Time Spent (min)": user["usage_metrics"]["time_spent_minutes"],
                    "Satisfaction": f"{user['satisfaction_score']:.1f}/5.0"
                    if "satisfaction_score" in user
                    else "N/A",
                    "Last Active": user.get("last_active", "Never")[:10]
                    if user.get("last_active")
                    else "Never",
                }
            )

        if user_data:
            df = pd.DataFrame(user_data)
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.info("No users match the selected filters.")

    def render_action_items(self):
        """Render action items and recommendations."""
        st.subheader("📋 Action Items & Recommendations")

        metrics = self.calculate_metrics(st.session_state.beta_data)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 High Priority Actions")

            if metrics.activation_rate < 60:
                st.warning("**Low Activation Rate**: Focus on onboarding improvements")

            if metrics.avg_session_time < 30:
                st.warning(
                    "**Low Engagement**: Review user experience and feature discoverability"
                )

            if metrics.satisfaction_score < 4.0:
                st.error(
                    "**Low Satisfaction**: Urgent attention needed for user feedback"
                )

            st.success("**Feature Usage**: Good adoption of core features")
            st.info("**Documentation**: Consider expanding API examples")

        with col2:
            st.markdown("### 📈 Growth Opportunities")

            st.markdown("""
            - **Industry Expansion**: Consider targeting retail and energy sectors
            - **Feature Development**: Focus on API enhancements and alert configuration
            - **User Advocacy**: Leverage high NPS users for testimonials
            - **Onboarding**: Improve time-to-value for new users
            - **Community**: Create beta user forum or Slack channel
            """)

    def run(self):
        """Run the beta dashboard."""
        self.render_header()
        self.render_sidebar()

        # Main content
        self.render_overview_metrics()

        st.markdown("---")

        # Analytics sections
        col1, col2 = st.columns(2)
        with col1:
            self.render_user_funnel()
        with col2:
            self.render_industry_analysis()

        st.markdown("---")

        self.render_usage_analytics()

        st.markdown("---")

        self.render_feedback_analysis()

        st.markdown("---")

        self.render_user_table()

        st.markdown("---")

        self.render_action_items()

        # Footer
        st.markdown("---")
        st.markdown(
            f"**Dashboard Last Updated:** {self.last_updated.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        st.markdown(
            "**Pynomaly Beta Program** | For questions: beta-support@example.com"
        )


def main():
    """Main dashboard application."""
    dashboard = BetaDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
