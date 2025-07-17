"""
Reporting service for business measurements and analytics.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any

from monorepo.domain.entities.reporting import (
    Alert,
    Dashboard,
    Metric,
    MetricType,
    MetricValue,
    Report,
    ReportFilter,
    ReportSection,
    ReportStatus,
    ReportType,
    TimeGranularity,
)
from monorepo.shared.exceptions import (
    AuthorizationError,
    ReportNotFoundError,
)
from monorepo.shared.types import TenantId, UserId


class ReportingService:
    """Service for generating reports and managing business measurements."""

    def __init__(self, metrics_repository, report_repository, user_service):
        self._measurements_repo = measurements_repository
        self._report_repo = report_repository
        self._user_service = user_service

    # Report Generation
    async def generate_report(
        self,
        report_type: ReportType,
        tenant_id: TenantId,
        user_id: UserId,
        filters: ReportFilter | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> Report:
        """Generate a new business report."""
        # Validate user permissions
        user = await self._user_service.get_user_by_id(user_id)
        if not user or not user.has_role_in_tenant(
            tenant_id, ["analyst", "data_scientist", "tenant_admin"]
        ):
            raise AuthorizationError("Insufficient permissions to generate reports")

        # Set default filters
        if filters is None:
            filters = ReportFilter(
                start_date=datetime.utcnow() - timedelta(days=30),
                end_date=datetime.utcnow(),
                tenant_ids=[tenant_id],
            )

        # Create report
        report = Report(
            id=str(uuid.uuid4()),
            title=title or f"{report_type.value.replace('_', ' ').title()} Report",
            description=description or f"Generated {report_type.value} report",
            report_type=report_type,
            status=ReportStatus.PENDING,
            tenant_id=tenant_id,
            created_by=user_id,
            filters=filters,
            expires_at=datetime.utcnow() + timedelta(days=30),
        )

        # Save initial report
        await self._report_repo.create_report(report)

        # Generate report content asynchronously
        await self._generate_report_content(report)

        return report

    async def _generate_report_content(self, report: Report) -> None:
        """Generate the actual report content based on type."""
        try:
            report.status = ReportStatus.GENERATING
            await self._report_repo.update_report(report)

            if report.report_type == ReportType.DETECTION_SUMMARY:
                await self._generate_processing_summary(report)
            elif report.report_type == ReportType.BUSINESS_METRICS:
                await self._generate_business_measurements(report)
            elif report.report_type == ReportType.PERFORMANCE_ANALYSIS:
                await self._generate_performance_analysis(report)
            elif report.report_type == ReportType.USAGE_ANALYTICS:
                await self._generate_usage_analytics(report)
            elif report.report_type == ReportType.COST_ANALYSIS:
                await self._generate_cost_analysis(report)
            elif report.report_type == ReportType.COMPLIANCE_REPORT:
                await self._generate_compliance_report(report)
            elif report.report_type == ReportType.TREND_ANALYSIS:
                await self._generate_trend_analysis(report)

            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.utcnow()

        except Exception as e:
            report.status = ReportStatus.FAILED
            report.metadata["error"] = str(e)

        finally:
            await self._report_repo.update_report(report)

    async def _generate_processing_summary(self, report: Report) -> None:
        """Generate processing summary section."""
        # Fetch processing measurements
        processing_data = await self._measurements_repo.get_processing_measurements(
            tenant_id=report.tenant_id,
            start_date=report.filters.start_date,
            end_date=report.filters.end_date,
        )

        # Create measurements
        measurements = []

        # Success rate metric
        success_rate_metric = Metric(
            id="processing_success_rate",
            name="Processing Success Rate",
            description="Percentage of successful anomaly detections",
            metric_type=MetricType.PERCENTAGE,
        )
        success_rate_metric.add_value(processing_data.success_rate)
        measurements.append(success_rate_metric)

        # Total detections metric
        total_processings_metric = Metric(
            id="total_processings",
            name="Total Detections",
            description="Total number of anomaly processing runs",
            metric_type=MetricType.COUNTER,
        )
        total_processings_metric.add_value(processing_data.total_processings)
        measurements.append(total_processings_metric)

        # Anomalies found metric
        anomalies_metric = Metric(
            id="anomalies_found",
            name="Anomalies Found",
            description="Total anomalies detected",
            metric_type=MetricType.COUNTER,
        )
        anomalies_metric.add_value(processing_data.anomalies_found)
        measurements.append(anomalies_metric)

        # Processor performance measurements
        if processing_data.precision > 0:
            precision_metric = Metric(
                id="precision",
                name="Processor Precision",
                description="Precision of anomaly processing models",
                metric_type=MetricType.PERCENTAGE,
            )
            precision_metric.add_value(processing_data.precision * 100)
            measurements.append(precision_metric)

        # Create charts
        charts = [
            {
                "type": "pie",
                "title": "Processing Results",
                "data": {
                    "successful": processing_data.successful_processings,
                    "failed": processing_data.failed_processings,
                },
            },
            {
                "type": "bar",
                "title": "Processing Performance",
                "data": {
                    "precision": processing_data.precision * 100,
                    "recall": processing_data.recall * 100,
                    "f1_score": processing_data.f1_score * 100,
                },
            },
        ]

        # Generate insights
        insights = []
        if processing_data.success_rate < 90:
            insights.append(
                "Processing success rate is below recommended threshold of 90%"
            )
        if processing_data.anomaly_rate > 10:
            insights.append(
                "High anomaly rate detected - consider reviewing processing thresholds"
            )
        if processing_data.average_processing_time > 300:  # 5 minutes
            insights.append(
                "Average processing time exceeds 5 minutes - consider performance optimization"
            )

        # Create section
        section = ReportSection(
            id="processing_summary",
            title="Processing Summary",
            description="Overview of anomaly processing performance",
            measurements=measurements,
            charts=charts,
            insights=insights,
            order=1,
        )

        report.add_section(section)

    async def _generate_business_measurements(self, report: Report) -> None:
        """Generate business measurements section."""
        business_data = await self._measurements_repo.get_business_measurements(
            tenant_id=report.tenant_id,
            start_date=report.filters.start_date,
            end_date=report.filters.end_date,
        )

        measurements = []

        # Cost savings metric
        cost_savings_metric = Metric(
            id="cost_savings",
            name="Cost Savings",
            description="Estimated cost savings from anomaly processing",
            metric_type=MetricType.CURRENCY,
        )
        cost_savings_metric.add_value(business_data.cost_savings)
        measurements.append(cost_savings_metric)

        # ROI metric
        roi_metric = Metric(
            id="roi",
            name="Return on Investment",
            description="ROI from anomaly processing implementation",
            metric_type=MetricType.PERCENTAGE,
        )
        roi_metric.add_value(
            business_data.calculate_roi(10000)
        )  # Assuming $10k investment
        measurements.append(roi_metric)

        # Time to insight metric
        time_to_insight_metric = Metric(
            id="time_to_insight",
            name="Time to Insight",
            description="Average time from data ingestion to actionable insights",
            metric_type=MetricType.DURATION,
        )
        time_to_insight_metric.add_value(
            business_data.time_to_insight * 3600
        )  # Convert hours to seconds
        measurements.append(time_to_insight_metric)

        section = ReportSection(
            id="business_measurements",
            title="Business Impact",
            description="Key business measurements and ROI analysis",
            measurements=measurements,
            order=2,
        )

        report.add_section(section)

    async def _generate_usage_analytics(self, report: Report) -> None:
        """Generate usage analytics section."""
        usage_data = await self._measurements_repo.get_usage_measurements(
            tenant_id=report.tenant_id,
            start_date=report.filters.start_date,
            end_date=report.filters.end_date,
        )

        measurements = []

        # API usage metric
        api_usage_metric = Metric(
            id="api_usage",
            name="API Usage",
            description="Total API calls this month",
            metric_type=MetricType.COUNTER,
        )
        api_usage_metric.add_value(usage_data.api_calls_this_month)
        measurements.append(api_usage_metric)

        # Storage usage metric
        storage_metric = Metric(
            id="storage_usage",
            name="Storage Usage",
            description="Total storage used in GB",
            metric_type=MetricType.GAUGE,
        )
        storage_metric.add_value(usage_data.storage_used_gb)
        measurements.append(storage_metric)

        section = ReportSection(
            id="usage_analytics",
            title="Usage Analytics",
            description="System usage patterns and resource consumption",
            measurements=measurements,
            order=3,
        )

        report.add_section(section)

    # Dashboard Management
    async def create_dashboard(
        self,
        name: str,
        tenant_id: TenantId,
        user_id: UserId,
        description: str = "",
        widgets: list[dict[str, Any]] | None = None,
    ) -> Dashboard:
        """Create a new dashboard."""
        dashboard = Dashboard(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            tenant_id=tenant_id,
            created_by=user_id,
            widgets=widgets or [],
        )

        return await self._report_repo.create_dashboard(dashboard)

    async def get_dashboard(self, dashboard_id: str, user_id: UserId) -> Dashboard:
        """Get dashboard by ID with permission check."""
        dashboard = await self._report_repo.get_dashboard_by_id(dashboard_id)
        if not dashboard:
            raise ReportNotFoundError("Dashboard not found")

        # Check user permissions
        user = await self._user_service.get_user_by_id(user_id)
        if not (
            user.is_super_admin()
            or user.has_role_in_tenant(
                dashboard.tenant_id,
                ["viewer", "analyst", "data_scientist", "tenant_admin"],
            )
            or dashboard.is_public
        ):
            raise AuthorizationError("Access denied")

        # Update last accessed
        dashboard.last_accessed = datetime.utcnow()
        await self._report_repo.update_dashboard(dashboard)

        return dashboard

    async def update_dashboard_widgets(
        self, dashboard_id: str, user_id: UserId, widgets: list[dict[str, Any]]
    ) -> Dashboard:
        """Update dashboard widgets."""
        dashboard = await self.get_dashboard(dashboard_id, user_id)

        # Check edit permissions
        user = await self._user_service.get_user_by_id(user_id)
        if not (
            user.is_super_admin()
            or dashboard.created_by == user_id
            or user.has_role_in_tenant(
                dashboard.tenant_id, ["data_scientist", "tenant_admin"]
            )
        ):
            raise AuthorizationError("Insufficient permissions to edit dashboard")

        dashboard.widgets = widgets
        dashboard.updated_at = datetime.utcnow()

        return await self._report_repo.update_dashboard(dashboard)

    # Measurements Management
    async def get_real_time_measurements(
        self, tenant_id: TenantId, metric_ids: list[str]
    ) -> dict[str, Any]:
        """Get real-time measurements for dashboard."""
        measurements = {}

        for metric_id in metric_ids:
            metric_data = await self._measurements_repo.get_metric(tenant_id, metric_id)
            if metric_data:
                measurements[metric_id] = {
                    "current_value": metric_data.current_value,
                    "formatted_value": metric_data.latest_value.format_value()
                    if metric_data.latest_value
                    else "N/A",
                    "last_updated": metric_data.updated_at.isoformat(),
                    "type": metric_data.metric_type.value,
                }

        return measurements

    async def get_metric_history(
        self,
        tenant_id: TenantId,
        metric_id: str,
        start_date: datetime,
        end_date: datetime,
        granularity: TimeGranularity = TimeGranularity.HOUR,
    ) -> list[dict[str, Any]]:
        """Get metric history for charting."""
        metric = await self._measurements_repo.get_metric(tenant_id, metric_id)
        if not metric:
            return []

        values = metric.get_values_in_range(start_date, end_date)

        # Aggregate by granularity
        aggregated = self._aggregate_metric_values(values, granularity)

        return [
            {
                "timestamp": timestamp.isoformat(),
                "value": value,
                "formatted_value": MetricValue(
                    value, timestamp, metric.metric_type
                ).format_value(),
            }
            for timestamp, value in aggregated.items()
        ]

    def _aggregate_metric_values(
        self, values: list[MetricValue], granularity: TimeGranularity
    ) -> dict[datetime, float]:
        """Aggregate metric values by time granularity."""
        aggregated = {}

        for value in values:
            # Round timestamp to granularity
            if granularity == TimeGranularity.MINUTE:
                key_time = value.timestamp.replace(second=0, microsecond=0)
            elif granularity == TimeGranularity.HOUR:
                key_time = value.timestamp.replace(minute=0, second=0, microsecond=0)
            elif granularity == TimeGranularity.DAY:
                key_time = value.timestamp.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
            else:
                key_time = value.timestamp

            if key_time not in aggregated:
                aggregated[key_time] = []
            aggregated[key_time].append(float(value.value))

        # Calculate averages for each time bucket
        return {
            timestamp: sum(values) / len(values)
            for timestamp, values in aggregated.items()
        }

    # Alert Management
    async def create_alert(
        self,
        name: str,
        metric_id: str,
        tenant_id: TenantId,
        condition: str,
        threshold: float,
        notification_channels: list[str],
        description: str = "",
    ) -> Alert:
        """Create a new metric alert."""
        alert = Alert(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            metric_id=metric_id,
            tenant_id=tenant_id,
            condition=condition,
            threshold=threshold,
            notification_channels=notification_channels,
        )

        return await self._report_repo.create_alert(alert)

    async def check_alerts(self, tenant_id: TenantId) -> list[dict[str, Any]]:
        """Check all active alerts for a tenant."""
        alerts = await self._report_repo.get_active_alerts(tenant_id)
        triggered_alerts = []

        for alert in alerts:
            metric = await self._measurements_repo.get_metric(tenant_id, alert.metric_id)
            if not metric or not metric.latest_value:
                continue

            current_value = float(metric.latest_value.value)

            # Get previous value for change calculations
            previous_value = None
            if len(metric.values) > 1:
                previous_value = float(metric.values[-2].value)

            if alert.should_trigger(current_value, previous_value):
                triggered_alerts.append(
                    {
                        "alert_id": alert.id,
                        "alert_name": alert.name,
                        "metric_id": alert.metric_id,
                        "current_value": current_value,
                        "threshold": alert.threshold,
                        "condition": alert.condition,
                        "notification_channels": alert.notification_channels,
                    }
                )

                # Update alert trigger info
                alert.last_triggered = datetime.utcnow()
                alert.trigger_count += 1
                await self._report_repo.update_alert(alert)

        return triggered_alerts

    # Predefined Reports
    async def create_standard_dashboard(
        self, tenant_id: TenantId, user_id: UserId
    ) -> Dashboard:
        """Create a standard dashboard with common widgets."""
        standard_widgets = [
            {
                "type": "metric_card",
                "title": "Processing Success Rate",
                "metric_id": "processing_success_rate",
                "size": "small",
                "position": {"x": 0, "y": 0, "w": 3, "h": 2},
            },
            {
                "type": "metric_card",
                "title": "Monthly Cost Savings",
                "metric_id": "monthly_cost_savings",
                "size": "small",
                "position": {"x": 3, "y": 0, "w": 3, "h": 2},
            },
            {
                "type": "line_chart",
                "title": "Processing Trends",
                "metric_id": "total_processings",
                "time_range": "7d",
                "position": {"x": 0, "y": 2, "w": 6, "h": 4},
            },
            {
                "type": "pie_chart",
                "title": "Usage Distribution",
                "metric_id": "api_usage",
                "position": {"x": 6, "y": 0, "w": 3, "h": 4},
            },
        ]

        return await self.create_dashboard(
            name="Standard Analytics Dashboard",
            tenant_id=tenant_id,
            user_id=user_id,
            description="Default dashboard with key business measurements",
            widgets=standard_widgets,
        )
