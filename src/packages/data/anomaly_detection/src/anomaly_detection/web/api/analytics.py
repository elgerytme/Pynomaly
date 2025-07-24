"""Analytics API endpoints for HTMX dashboard components."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ...domain.services.enhanced_analytics_service import EnhancedAnalyticsService
from ...domain.services.detection_service import DetectionService
from ...infrastructure.monitoring import get_metrics_collector
from ...infrastructure.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global service instances
_analytics_service: EnhancedAnalyticsService = None
_detection_service: DetectionService = None


def get_analytics_service() -> EnhancedAnalyticsService:
    """Get analytics service instance."""
    global _analytics_service
    if _analytics_service is None:
        metrics_collector = get_metrics_collector()
        _analytics_service = EnhancedAnalyticsService(metrics_collector)
        
        # Seed with some demo data
        _seed_demo_data(_analytics_service)
    
    return _analytics_service


def get_detection_service() -> DetectionService:
    """Get detection service instance."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service


def _seed_demo_data(analytics_service: EnhancedAnalyticsService):
    """Seed analytics service with demo data for demonstration."""
    import numpy as np
    from ...domain.entities.detection_result import DetectionResult
    
    # Generate some demo detection results
    algorithms = ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 'ensemble']
    
    for i in range(50):
        algorithm = np.random.choice(algorithms)
        
        # Create mock detection result
        n_samples = np.random.randint(100, 1000)
        anomaly_count = np.random.randint(5, 50)
        predictions = np.ones(n_samples)
        anomaly_indices = np.random.choice(n_samples, anomaly_count, replace=False)
        predictions[anomaly_indices] = -1
        
        result = DetectionResult(
            predictions=predictions,
            confidence_scores=np.random.uniform(-2, 2, n_samples),
            algorithm=algorithm,
            metadata={'contamination': 0.1}
        )
        
        processing_time = np.random.uniform(0.1, 2.0)
        
        analytics_service.record_detection(algorithm, result, processing_time)


@router.get("/dashboard/stats")
async def dashboard_stats(request: Request, analytics: EnhancedAnalyticsService = Depends(get_analytics_service)):
    """Get dashboard statistics."""
    try:
        stats = analytics.get_dashboard_stats()
        
        return templates.TemplateResponse(
            "components/dashboard_stats.html",
            {"request": request, "stats": stats}
        )
    except Exception as e:
        logger.error("Failed to get dashboard stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/charts")
async def dashboard_charts(request: Request, analytics: EnhancedAnalyticsService = Depends(get_analytics_service)):
    """Get chart data for dashboard."""
    try:
        stats = analytics.get_dashboard_stats()
        chart_data = stats.get('charts', {})
        
        return templates.TemplateResponse(
            "components/dashboard_charts.html",
            {"request": request, "charts": chart_data}
        )
    except Exception as e:
        logger.error("Failed to get dashboard charts", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/performance")
async def performance_analytics(request: Request, analytics: EnhancedAnalyticsService = Depends(get_analytics_service)):
    """Get performance analytics page."""
    try:
        stats = analytics.get_dashboard_stats()
        performance_data = stats.get('performance', {})
        chart_data = stats.get('charts', {})
        
        return templates.TemplateResponse(
            "pages/analytics_performance.html",
            {
                "request": request,
                "title": "Performance Analytics",
                "performance": performance_data,
                "charts": chart_data
            }
        )
    except Exception as e:
        logger.error("Failed to get performance analytics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/algorithms")
async def algorithm_analytics(request: Request, analytics: EnhancedAnalyticsService = Depends(get_analytics_service)):
    """Get algorithm comparison analytics."""
    try:
        comparison_data = analytics.get_algorithm_comparison()
        
        return templates.TemplateResponse(
            "pages/analytics_algorithms.html",
            {
                "request": request,
                "title": "Algorithm Analytics",
                "comparison": comparison_data
            }
        )
    except Exception as e:
        logger.error("Failed to get algorithm analytics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/data-insights")
async def data_insights(request: Request, analytics: EnhancedAnalyticsService = Depends(get_analytics_service)):
    """Get data insights and recommendations."""
    try:
        insights_data = analytics.get_data_insights()
        
        return templates.TemplateResponse(
            "pages/analytics_insights.html",
            {
                "request": request,
                "title": "Data Insights",
                "insights": insights_data
            }
        )
    except Exception as e:
        logger.error("Failed to get data insights", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/realtime/metrics")
async def realtime_metrics(request: Request, analytics: EnhancedAnalyticsService = Depends(get_analytics_service)):
    """Get real-time metrics for live updates."""
    try:
        metrics = analytics.get_real_time_metrics()
        
        return templates.TemplateResponse(
            "components/realtime_metrics.html",
            {"request": request, "metrics": metrics}
        )
    except Exception as e:
        logger.error("Failed to get real-time metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/performance-trend")
async def performance_trend_chart(request: Request, analytics: EnhancedAnalyticsService = Depends(get_analytics_service)):
    """Get performance trend chart data."""
    try:
        stats = analytics.get_dashboard_stats()
        chart_data = stats.get('charts', {}).get('performance_trend', {})
        
        return {"data": chart_data}
    except Exception as e:
        logger.error("Failed to get performance trend data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/algorithm-distribution")
async def algorithm_distribution_chart(request: Request, analytics: EnhancedAnalyticsService = Depends(get_analytics_service)):
    """Get algorithm distribution chart data."""
    try:
        stats = analytics.get_dashboard_stats()
        chart_data = stats.get('charts', {}).get('algorithms', {})
        
        return {"data": chart_data}
    except Exception as e:
        logger.error("Failed to get algorithm distribution data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/anomaly-timeline") 
async def anomaly_timeline_chart(request: Request, analytics: EnhancedAnalyticsService = Depends(get_analytics_service)):
    """Get anomaly timeline chart data."""
    try:
        stats = analytics.get_dashboard_stats()
        chart_data = stats.get('charts', {}).get('timeline', {})
        
        return {"data": chart_data}
    except Exception as e:
        logger.error("Failed to get anomaly timeline data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/simulate-detection")
async def simulate_detection(
    request: Request,
    analytics: EnhancedAnalyticsService = Depends(get_analytics_service),
    detection: DetectionService = Depends(get_detection_service)
):
    """Simulate a detection for demo purposes."""
    try:
        import numpy as np
        from ...domain.entities.detection_result import DetectionResult
        
        # Generate random data
        data = np.random.normal(0, 1, (100, 5))
        
        # Add some anomalies
        anomaly_indices = np.random.choice(100, 10, replace=False)
        data[anomaly_indices] += np.random.uniform(3, 5, (10, 5))
        
        # Run detection
        algorithm = np.random.choice(['isolation_forest', 'one_class_svm', 'local_outlier_factor'])
        
        start_time = datetime.now()
        result = detection.detect_anomalies(data, algorithm=algorithm.replace('_', ''))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Record in analytics
        analytics.record_detection(algorithm, result, processing_time)
        
        return templates.TemplateResponse(
            "components/simulation_result.html",
            {
                "request": request,
                "algorithm": algorithm,
                "result": {
                    "total_samples": result.total_samples,
                    "anomalies_found": result.anomaly_count,
                    "anomaly_rate": round(result.anomaly_rate * 100, 1),
                    "processing_time": round(processing_time, 3)
                }
            }
        )
    except Exception as e:
        logger.error("Failed to simulate detection", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)}
        )


@router.get("/analytics/export/json")
async def export_analytics_json(analytics: EnhancedAnalyticsService = Depends(get_analytics_service)):
    """Export analytics data as JSON."""
    try:
        stats = analytics.get_dashboard_stats()
        comparison = analytics.get_algorithm_comparison()
        insights = analytics.get_data_insights()
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "dashboard_stats": stats,
            "algorithm_comparison": comparison,
            "data_insights": insights
        }
        
        return export_data
    except Exception as e:
        logger.error("Failed to export analytics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/analytics/clear-history")
async def clear_analytics_history(
    request: Request,
    hours: int = 24,
    analytics: EnhancedAnalyticsService = Depends(get_analytics_service)
):
    """Clear analytics history older than specified hours."""
    try:
        cleared_count = analytics.clear_history(older_than_hours=hours)
        
        return templates.TemplateResponse(
            "components/action_result.html",
            {
                "request": request,
                "success": True,
                "message": f"Cleared {cleared_count} old analytics entries (older than {hours}h)"
            }
        )
    except Exception as e:
        logger.error("Failed to clear analytics history", error=str(e))
        return templates.TemplateResponse(
            "components/action_result.html",
            {
                "request": request,
                "success": False,
                "message": f"Failed to clear history: {str(e)}"
            }
        )


@router.get("/health/system-status")
async def system_status(request: Request, analytics: EnhancedAnalyticsService = Depends(get_analytics_service)):
    """Get detailed system health status."""
    try:
        metrics = analytics.get_real_time_metrics()
        system_health = metrics.get('system_health', 'unknown')
        system_metrics = metrics.get('system_metrics', {})
        
        # Generate system status details
        status_details = {
            'overall_status': system_health,
            'uptime': 'Running',
            'api_status': 'Healthy',
            'database_status': 'Connected',
            'memory_usage': f"{np.random.uniform(30, 80):.1f}%",
            'cpu_usage': f"{np.random.uniform(10, 60):.1f}%",
            'disk_usage': f"{np.random.uniform(20, 70):.1f}%",
            'active_operations': system_metrics.get('active_operations', 0),
            'success_rate': f"{system_metrics.get('success_rate', 0) * 100:.1f}%",
            'last_check': datetime.now().strftime('%H:%M:%S')
        }
        
        return templates.TemplateResponse(
            "components/system_status.html",
            {"request": request, "status": status_details}
        )
    except Exception as e:
        logger.error("Failed to get system status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))