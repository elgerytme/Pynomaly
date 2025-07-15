"""
Performance Metrics API for Real User Monitoring (RUM)
Collects and analyzes performance data from web clients
"""

import logging
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/metrics", tags=["performance"])

# In-memory storage for demo (in production, use proper database)
performance_metrics_store: list[dict[str, Any]] = []


@dataclass
class CoreWebVitals:
    """Core Web Vitals metrics"""

    lcp: float | None = None  # Largest Contentful Paint
    fid: float | None = None  # First Input Delay
    cls: float | None = None  # Cumulative Layout Shift
    fcp: float | None = None  # First Contentful Paint


@dataclass
class PerformanceMetricData:
    """Individual performance metric data point"""

    type: str
    data: dict[str, Any]
    page: str
    user_agent: str
    timestamp: int


class PerformanceMetricsPayload(BaseModel):
    """Payload for performance metrics submission"""

    metrics: list[dict[str, Any]] = Field(
        ..., description="Array of performance metrics"
    )
    session: str = Field(..., description="Session identifier")
    page_load_time: int = Field(..., description="Total page load time in milliseconds")


class PerformanceAnalysis(BaseModel):
    """Performance analysis results"""

    core_web_vitals: dict[str, Any]
    performance_score: float
    recommendations: list[str]
    trends: dict[str, Any]
    summary: dict[str, Any]


class PerformanceThresholds:
    """Performance thresholds for scoring"""

    LCP_GOOD = 2500  # ms
    LCP_POOR = 4000  # ms
    FID_GOOD = 100  # ms
    FID_POOR = 300  # ms
    CLS_GOOD = 0.1  # score
    CLS_POOR = 0.25  # score
    FCP_GOOD = 1800  # ms
    FCP_POOR = 3000  # ms


class PerformanceAnalyzer:
    """Analyzes performance metrics and generates insights"""

    @staticmethod
    def analyze_core_web_vitals(metrics: list[PerformanceMetricData]) -> CoreWebVitals:
        """Extract and analyze Core Web Vitals from metrics"""
        cwv = CoreWebVitals()

        for metric in metrics:
            if metric.type == "lcp" and "value" in metric.data:
                cwv.lcp = metric.data["value"]
            elif metric.type == "fid" and "value" in metric.data:
                cwv.fid = metric.data["value"]
            elif metric.type == "cls" and "value" in metric.data:
                cwv.cls = metric.data["value"]
            elif metric.type == "fcp" and "value" in metric.data:
                cwv.fcp = metric.data["value"]

        return cwv

    @staticmethod
    def calculate_performance_score(cwv: CoreWebVitals) -> float:
        """Calculate overall performance score from Core Web Vitals"""
        score = 100.0

        # LCP scoring
        if cwv.lcp is not None:
            if cwv.lcp > PerformanceThresholds.LCP_POOR:
                score -= 40
            elif cwv.lcp > PerformanceThresholds.LCP_GOOD:
                score -= 20

        # FID scoring
        if cwv.fid is not None:
            if cwv.fid > PerformanceThresholds.FID_POOR:
                score -= 30
            elif cwv.fid > PerformanceThresholds.FID_GOOD:
                score -= 15

        # CLS scoring
        if cwv.cls is not None:
            if cwv.cls > PerformanceThresholds.CLS_POOR:
                score -= 25
            elif cwv.cls > PerformanceThresholds.CLS_GOOD:
                score -= 12

        # FCP bonus/penalty
        if cwv.fcp is not None:
            if cwv.fcp > PerformanceThresholds.FCP_POOR:
                score -= 5
            elif cwv.fcp <= PerformanceThresholds.FCP_GOOD:
                score += 5

        return max(0.0, min(100.0, score))

    @staticmethod
    def generate_recommendations(
        cwv: CoreWebVitals, metrics: list[PerformanceMetricData]
    ) -> list[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        # LCP recommendations
        if cwv.lcp and cwv.lcp > PerformanceThresholds.LCP_GOOD:
            recommendations.append(
                "Improve Largest Contentful Paint (LCP): Optimize images, "
                "reduce server response times, or implement preloading for critical resources"
            )

        # FID recommendations
        if cwv.fid and cwv.fid > PerformanceThresholds.FID_GOOD:
            recommendations.append(
                "Reduce First Input Delay (FID): Minimize JavaScript execution time, "
                "break up long tasks, or use web workers for heavy computations"
            )

        # CLS recommendations
        if cwv.cls and cwv.cls > PerformanceThresholds.CLS_GOOD:
            recommendations.append(
                "Minimize Cumulative Layout Shift (CLS): Set explicit dimensions for images, "
                "avoid inserting content above existing content, and use transform animations"
            )

        # FCP recommendations
        if cwv.fcp and cwv.fcp > PerformanceThresholds.FCP_GOOD:
            recommendations.append(
                "Optimize First Contentful Paint (FCP): Eliminate render-blocking resources, "
                "minimize critical path length, or inline critical CSS"
            )

        # Analyze specific slow resources
        slow_resources = [m for m in metrics if m.type == "slow_resource"]
        if slow_resources:
            recommendations.append(
                f"Optimize {len(slow_resources)} slow-loading resources: "
                "Consider compression, CDN usage, or lazy loading"
            )

        # API performance recommendations
        api_metrics = [m for m in metrics if m.type == "api_response"]
        if api_metrics:
            slow_apis = [m for m in api_metrics if m.data.get("duration", 0) > 1000]
            if slow_apis:
                recommendations.append(
                    f"Optimize {len(slow_apis)} slow API endpoints: "
                    "Review backend performance, implement caching, or optimize queries"
                )

        return recommendations

    @staticmethod
    def analyze_trends(recent_hours: int = 24) -> dict[str, Any]:
        """Analyze performance trends over recent time period"""
        cutoff_time = datetime.now() - timedelta(hours=recent_hours)
        cutoff_timestamp = int(cutoff_time.timestamp() * 1000)

        recent_metrics = [
            m
            for m in performance_metrics_store
            if m.get("timestamp", 0) > cutoff_timestamp
        ]

        if not recent_metrics:
            return {"message": "No recent data available"}

        # Group metrics by type
        metric_groups = {}
        for metric in recent_metrics:
            metric_type = metric.get("type", "unknown")
            if metric_type not in metric_groups:
                metric_groups[metric_type] = []
            metric_groups[metric_type].append(metric)

        trends = {}

        # Analyze Core Web Vitals trends
        for cwv_type in ["lcp", "fid", "cls", "fcp"]:
            if cwv_type in metric_groups:
                values = [
                    m["data"].get("value")
                    for m in metric_groups[cwv_type]
                    if "value" in m.get("data", {})
                ]
                if values:
                    trends[cwv_type] = {
                        "count": len(values),
                        "avg": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                    }

        # Analyze API response trends
        if "api_response" in metric_groups:
            api_durations = [
                m["data"].get("duration")
                for m in metric_groups["api_response"]
                if "duration" in m.get("data", {})
            ]
            if api_durations:
                trends["api_response"] = {
                    "count": len(api_durations),
                    "avg_duration": statistics.mean(api_durations),
                    "slow_requests": len([d for d in api_durations if d > 1000]),
                }

        return trends


@router.post("/performance")
async def submit_performance_metrics(
    payload: PerformanceMetricsPayload, background_tasks: BackgroundTasks
) -> dict[str, str]:
    """
    Submit performance metrics from web clients
    """
    try:
        # Convert payload to internal format
        metrics = []
        for metric_data in payload.metrics:
            metric = PerformanceMetricData(
                type=metric_data.get("type", "unknown"),
                data=metric_data.get("data", {}),
                page=metric_data.get("page", "/"),
                user_agent=metric_data.get("userAgent", "unknown"),
                timestamp=metric_data.get(
                    "timestamp", int(datetime.now().timestamp() * 1000)
                ),
            )
            metrics.append(asdict(metric))

        # Store metrics (in production, use proper database)
        performance_metrics_store.extend(metrics)

        # Process metrics in background
        background_tasks.add_task(process_metrics_async, metrics, payload.session)

        logger.info(
            f"Received {len(metrics)} performance metrics for session {payload.session}"
        )

        return {"status": "success", "message": f"Processed {len(metrics)} metrics"}

    except Exception as e:
        logger.error(f"Error processing performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to process metrics")


@router.get("/performance/analysis")
async def get_performance_analysis(
    session_id: str | None = None, hours: int = 24
) -> PerformanceAnalysis:
    """
    Get performance analysis and recommendations
    """
    try:
        # Filter metrics by session if provided
        if session_id:
            session_metrics = [
                m for m in performance_metrics_store if m.get("session") == session_id
            ]
        else:
            # Get recent metrics
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            session_metrics = [
                m
                for m in performance_metrics_store
                if m.get("timestamp", 0) > cutoff_timestamp
            ]

        if not session_metrics:
            raise HTTPException(status_code=404, detail="No metrics found")

        # Convert to PerformanceMetricData objects
        metric_objects = [
            PerformanceMetricData(
                type=m.get("type", "unknown"),
                data=m.get("data", {}),
                page=m.get("page", "/"),
                user_agent=m.get("user_agent", "unknown"),
                timestamp=m.get("timestamp", 0),
            )
            for m in session_metrics
        ]

        # Analyze metrics
        analyzer = PerformanceAnalyzer()
        cwv = analyzer.analyze_core_web_vitals(metric_objects)
        score = analyzer.calculate_performance_score(cwv)
        recommendations = analyzer.generate_recommendations(cwv, metric_objects)
        trends = analyzer.analyze_trends(hours)

        # Generate summary
        summary = {
            "total_metrics": len(session_metrics),
            "pages_tracked": len({m.get("page", "/") for m in session_metrics}),
            "session_count": len(
                {m.get("session") for m in session_metrics if m.get("session")}
            ),
            "time_range_hours": hours,
        }

        return PerformanceAnalysis(
            core_web_vitals={
                "lcp": cwv.lcp,
                "fid": cwv.fid,
                "cls": cwv.cls,
                "fcp": cwv.fcp,
                "scores": {
                    "lcp": (
                        "good"
                        if cwv.lcp and cwv.lcp <= PerformanceThresholds.LCP_GOOD
                        else (
                            "needs_improvement"
                            if cwv.lcp and cwv.lcp <= PerformanceThresholds.LCP_POOR
                            else "poor"
                        )
                    ),
                    "fid": (
                        "good"
                        if cwv.fid and cwv.fid <= PerformanceThresholds.FID_GOOD
                        else (
                            "needs_improvement"
                            if cwv.fid and cwv.fid <= PerformanceThresholds.FID_POOR
                            else "poor"
                        )
                    ),
                    "cls": (
                        "good"
                        if cwv.cls and cwv.cls <= PerformanceThresholds.CLS_GOOD
                        else (
                            "needs_improvement"
                            if cwv.cls and cwv.cls <= PerformanceThresholds.CLS_POOR
                            else "poor"
                        )
                    ),
                },
            },
            performance_score=score,
            recommendations=recommendations,
            trends=trends,
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating performance analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate analysis")


@router.get("/performance/summary")
async def get_performance_summary(hours: int = 24) -> dict[str, Any]:
    """
    Get a summary of performance metrics
    """
    try:
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_timestamp = int(cutoff_time.timestamp() * 1000)

        recent_metrics = [
            m
            for m in performance_metrics_store
            if m.get("timestamp", 0) > cutoff_timestamp
        ]

        if not recent_metrics:
            return {
                "message": "No recent metrics available",
                "time_range_hours": hours,
                "total_metrics": 0,
            }

        # Group by page
        page_metrics = {}
        for metric in recent_metrics:
            page = metric.get("page", "/")
            if page not in page_metrics:
                page_metrics[page] = []
            page_metrics[page].append(metric)

        # Calculate averages per page
        page_summaries = {}
        for page, metrics in page_metrics.items():
            cwv_metrics = {}
            for metric in metrics:
                metric_type = metric.get("type")
                if metric_type in [
                    "lcp",
                    "fid",
                    "cls",
                    "fcp",
                ] and "value" in metric.get("data", {}):
                    if metric_type not in cwv_metrics:
                        cwv_metrics[metric_type] = []
                    cwv_metrics[metric_type].append(metric["data"]["value"])

            # Calculate averages
            page_avg = {}
            for metric_type, values in cwv_metrics.items():
                if values:
                    page_avg[f"avg_{metric_type}"] = statistics.mean(values)

            page_summaries[page] = {"total_metrics": len(metrics), **page_avg}

        return {
            "time_range_hours": hours,
            "total_metrics": len(recent_metrics),
            "pages_tracked": len(page_metrics),
            "unique_sessions": len(
                {m.get("session") for m in recent_metrics if m.get("session")}
            ),
            "page_summaries": page_summaries,
        }

    except Exception as e:
        logger.error(f"Error generating performance summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary")


@router.delete("/performance/clear")
async def clear_performance_metrics() -> dict[str, str]:
    """
    Clear all stored performance metrics (for testing/development)
    """
    global performance_metrics_store
    cleared_count = len(performance_metrics_store)
    performance_metrics_store.clear()

    logger.info(f"Cleared {cleared_count} performance metrics")

    return {"status": "success", "message": f"Cleared {cleared_count} metrics"}


async def process_metrics_async(metrics: list[dict[str, Any]], session_id: str):
    """
    Background task to process metrics
    """
    try:
        # In a real implementation, this would:
        # 1. Store metrics in a proper database
        # 2. Update real-time dashboards
        # 3. Trigger alerts for performance regressions
        # 4. Calculate aggregated statistics

        logger.info(f"Processing {len(metrics)} metrics for session {session_id}")

        # Example: Check for performance issues
        for metric in metrics:
            if (
                metric.get("type") == "lcp"
                and metric.get("data", {}).get("value", 0) > 4000
            ):
                logger.warning(
                    f"High LCP detected: {metric['data']['value']}ms for page {metric.get('page')}"
                )

            if (
                metric.get("type") == "api_response"
                and metric.get("data", {}).get("duration", 0) > 2000
            ):
                logger.warning(
                    f"Slow API response detected: {metric['data']['duration']}ms for {metric['data'].get('url')}"
                )

    except Exception as e:
        logger.error(f"Error in background metric processing: {e}")


# Health check endpoint
@router.get("/performance/health")
async def performance_monitoring_health() -> dict[str, Any]:
    """
    Health check for performance monitoring system
    """
    return {
        "status": "healthy",
        "metrics_stored": len(performance_metrics_store),
        "last_metric_time": (
            max([m.get("timestamp", 0) for m in performance_metrics_store])
            if performance_metrics_store
            else None
        ),
        "timestamp": int(datetime.now().timestamp() * 1000),
    }
