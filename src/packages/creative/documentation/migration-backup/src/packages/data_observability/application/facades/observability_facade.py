"""
Data Observability Application Facade

Provides a unified interface for all data observability capabilities,
integrating lineage tracking, pipeline health monitoring, data catalog,
and predictive quality services.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

from ..services.data_lineage_service import DataLineageService
from ..services.pipeline_health_service import PipelineHealthService
from ..services.data_catalog_service import DataCatalogService
from ..services.predictive_quality_service import PredictiveQualityService

from ...domain.entities.data_lineage import DataLineage, LineageNode
from ...domain.entities.pipeline_health import PipelineHealth, PipelineAlert
from ...domain.entities.data_catalog import DataCatalogEntry, DataAssetType, DataFormat
from ...domain.entities.quality_prediction import (
    QualityPrediction, 
    QualityForecast, 
    QualityTrend,
    PredictionType
)


class DataObservabilityFacade:
    """
    Unified facade for all data observability capabilities.
    
    This facade provides a single entry point for:
    - Data lineage tracking and impact analysis
    - Pipeline health monitoring and alerting
    - Data catalog management and discovery
    - Predictive data quality monitoring
    """
    
    def __init__(
        self,
        lineage_service: DataLineageService,
        health_service: PipelineHealthService,
        catalog_service: DataCatalogService,
        quality_service: PredictiveQualityService
    ):
        self.lineage_service = lineage_service
        self.health_service = health_service
        self.catalog_service = catalog_service
        self.quality_service = quality_service
    
    # ==== Data Lineage Operations ====
    
    def track_data_transformation(
        self,
        source_id: UUID,
        target_id: UUID,
        transformation_type: str,
        transformation_details: Dict[str, Any] = None
    ) -> None:
        """Track a data transformation between assets."""
        self.lineage_service.add_transformation(
            source_id=source_id,
            target_id=target_id,
            transformation_type=transformation_type,
            transformation_details=transformation_details or {}
        )
    
    def analyze_impact(self, node_id: UUID, impact_type: str = "downstream") -> Dict[str, Any]:
        """Analyze the impact of changes to a data asset."""
        return self.lineage_service.analyze_impact(node_id, impact_type)
    
    def get_lineage_graph(self, node_id: UUID, depth: int = 3) -> DataLineage:
        """Get the complete lineage graph for a data asset."""
        return self.lineage_service.get_lineage_graph(node_id, depth)
    
    def find_data_path(self, source_id: UUID, target_id: UUID) -> Optional[List[UUID]]:
        """Find the path between two data assets."""
        return self.lineage_service.find_path(source_id, target_id)
    
    # ==== Pipeline Health Operations ====
    
    def monitor_pipeline_health(
        self,
        pipeline_id: UUID,
        metrics: Dict[str, float],
        context: Dict[str, Any] = None
    ) -> PipelineHealth:
        """Monitor pipeline health with current metrics."""
        return self.health_service.update_pipeline_health(
            pipeline_id=pipeline_id,
            metrics=metrics,
            context=context or {}
        )
    
    def get_pipeline_alerts(self, pipeline_id: UUID = None) -> List[PipelineAlert]:
        """Get active pipeline alerts."""
        return self.health_service.get_active_alerts(pipeline_id)
    
    def get_pipeline_health_summary(self, pipeline_id: UUID) -> Dict[str, Any]:
        """Get comprehensive health summary for a pipeline."""
        health = self.health_service.get_pipeline_health(pipeline_id)
        if not health:
            return {}
        
        return {
            "pipeline_id": str(pipeline_id),
            "overall_health": health.overall_health,
            "health_score": health.health_score,
            "status": health.status,
            "last_updated": health.last_updated.isoformat(),
            "metrics_summary": health.get_metrics_summary(),
            "recent_alerts": len([a for a in self.get_pipeline_alerts(pipeline_id) 
                               if a.created_at >= datetime.utcnow() - timedelta(hours=24)])
        }
    
    # ==== Data Catalog Operations ====
    
    def register_data_asset(
        self,
        name: str,
        asset_type: DataAssetType,
        location: str,
        data_format: DataFormat,
        description: str = None,
        owner: str = None,
        domain: str = None,
        **kwargs
    ) -> DataCatalogEntry:
        """Register a new data asset in the catalog."""
        return self.catalog_service.register_asset(
            name=name,
            asset_type=asset_type,
            location=location,
            data_format=data_format,
            description=description,
            owner=owner,
            domain=domain,
            **kwargs
        )
    
    def discover_data_assets(self, query: str, limit: int = 20) -> List[DataCatalogEntry]:
        """Discover data assets using intelligent search."""
        return self.catalog_service.search_assets(query=query, limit=limit)
    
    def get_asset_recommendations(self, asset_id: UUID) -> List[Tuple[DataCatalogEntry, float]]:
        """Get recommendations for similar or related assets."""
        return self.catalog_service.discover_similar_assets(asset_id)
    
    def track_asset_usage(
        self,
        asset_id: UUID,
        user_id: str,
        usage_type: str = "read",
        **kwargs
    ) -> None:
        """Track usage of a data asset."""
        self.catalog_service.record_usage(
            asset_id=asset_id,
            user_id=user_id,
            usage_type=usage_type,
            **kwargs
        )
    
    # ==== Predictive Quality Operations ====
    
    def predict_quality_issues(
        self,
        asset_id: UUID,
        prediction_type: PredictionType,
        target_time: datetime,
        **kwargs
    ) -> QualityPrediction:
        """Predict potential quality issues for an asset."""
        return self.quality_service.create_prediction(
            asset_id=asset_id,
            prediction_type=prediction_type,
            target_time=target_time,
            **kwargs
        )
    
    def forecast_quality_metrics(
        self,
        asset_id: UUID,
        metric_type: str,
        horizon_hours: int = 24,
        **kwargs
    ) -> QualityForecast:
        """Forecast quality metrics over time."""
        return self.quality_service.create_forecast(
            asset_id=asset_id,
            metric_type=metric_type,
            horizon_hours=horizon_hours,
            **kwargs
        )
    
    def analyze_quality_trends(
        self,
        asset_id: UUID,
        metric_type: str,
        days: int = 30
    ) -> QualityTrend:
        """Analyze quality trends for an asset."""
        return self.quality_service.analyze_trends(
            asset_id=asset_id,
            metric_type=metric_type,
            days=days
        )
    
    def add_quality_metric(
        self,
        asset_id: UUID,
        metric_type: str,
        value: float,
        timestamp: datetime = None
    ) -> None:
        """Add a quality metric data point."""
        self.quality_service.add_metric_point(
            asset_id=asset_id,
            metric_type=metric_type,
            value=value,
            timestamp=timestamp
        )
    
    # ==== Cross-Service Operations ====
    
    def get_comprehensive_asset_view(self, asset_id: UUID) -> Dict[str, Any]:
        """Get a comprehensive view of an asset across all observability dimensions."""
        
        # Get catalog information
        catalog_entry = self.catalog_service.get_asset(asset_id)
        if not catalog_entry:
            return {"error": f"Asset {asset_id} not found in catalog"}
        
        # Get lineage information
        lineage = self.lineage_service.get_lineage_graph(asset_id, depth=2)
        
        # Get usage analytics
        usage_analytics = self.catalog_service.get_usage_analytics(asset_id)
        
        # Get quality predictions
        quality_alerts = self.quality_service.get_active_alerts(asset_id)
        
        # Get pipeline health (if asset is associated with a pipeline)
        pipeline_health = None
        if hasattr(catalog_entry, 'pipeline_id') and catalog_entry.pipeline_id:
            pipeline_health = self.health_service.get_pipeline_health(catalog_entry.pipeline_id)
        
        return {
            "asset_info": catalog_entry.to_summary_dict(),
            "lineage": {
                "upstream_count": len(lineage.nodes) - 1 if lineage.nodes else 0,
                "downstream_count": len([e for e in lineage.edges if e.source_id == asset_id]),
                "total_connected_assets": len(lineage.nodes),
                "critical_path": lineage.find_critical_path()
            },
            "usage": usage_analytics,
            "quality": {
                "active_alerts": len(quality_alerts),
                "recent_predictions": len([a for a in quality_alerts 
                                        if a.created_at >= datetime.utcnow() - timedelta(days=7)]),
                "quality_score": catalog_entry.quality_score
            },
            "pipeline_health": {
                "health_score": pipeline_health.health_score if pipeline_health else None,
                "status": pipeline_health.status if pipeline_health else "unknown",
                "last_updated": pipeline_health.last_updated.isoformat() if pipeline_health else None
            } if pipeline_health else None
        }
    
    def get_data_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive data health dashboard."""
        
        # Catalog statistics
        catalog_stats = self.catalog_service.get_catalog_statistics()
        
        # Pipeline health overview
        all_pipelines = self.health_service.get_all_pipeline_health()
        pipeline_stats = {
            "total_pipelines": len(all_pipelines),
            "healthy_pipelines": len([p for p in all_pipelines if p.health_score >= 0.8]),
            "degraded_pipelines": len([p for p in all_pipelines if 0.6 <= p.health_score < 0.8]),
            "unhealthy_pipelines": len([p for p in all_pipelines if p.health_score < 0.6])
        }
        
        # Quality predictions overview
        all_alerts = self.quality_service.get_active_alerts()
        quality_stats = {
            "total_active_alerts": len(all_alerts),
            "critical_alerts": len([a for a in all_alerts if a.severity == "critical"]),
            "high_priority_alerts": len([a for a in all_alerts if a.severity == "high"]),
            "recent_predictions": len([a for a in all_alerts 
                                    if a.created_at >= datetime.utcnow() - timedelta(hours=24)])
        }
        
        # Lineage statistics
        lineage_stats = self.lineage_service.get_lineage_statistics()
        
        return {
            "catalog": catalog_stats,
            "pipeline_health": pipeline_stats,
            "quality_predictions": quality_stats,
            "lineage": lineage_stats,
            "dashboard_generated_at": datetime.utcnow().isoformat()
        }
    
    def investigate_data_issue(
        self,
        asset_id: UUID,
        issue_type: str,
        severity: str = "medium"
    ) -> Dict[str, Any]:
        """Investigate a data issue across all observability dimensions."""
        
        investigation = {
            "asset_id": str(asset_id),
            "issue_type": issue_type,
            "severity": severity,
            "investigation_time": datetime.utcnow().isoformat(),
            "findings": []
        }
        
        # Check lineage for upstream issues
        impact_analysis = self.lineage_service.analyze_impact(asset_id, "upstream")
        if impact_analysis.get("affected_nodes"):
            investigation["findings"].append({
                "category": "lineage",
                "finding": f"Issue may be caused by upstream dependencies: {len(impact_analysis['affected_nodes'])} assets affected",
                "details": impact_analysis
            })
        
        # Check pipeline health
        catalog_entry = self.catalog_service.get_asset(asset_id)
        if catalog_entry and hasattr(catalog_entry, 'pipeline_id'):
            pipeline_health = self.health_service.get_pipeline_health(catalog_entry.pipeline_id)
            if pipeline_health and pipeline_health.health_score < 0.7:
                investigation["findings"].append({
                    "category": "pipeline_health",
                    "finding": f"Associated pipeline shows degraded health: {pipeline_health.health_score:.2f}",
                    "details": pipeline_health.get_metrics_summary()
                })
        
        # Check quality predictions
        quality_alerts = self.quality_service.get_active_alerts(asset_id)
        if quality_alerts:
            investigation["findings"].append({
                "category": "quality_predictions",
                "finding": f"Active quality alerts detected: {len(quality_alerts)} alerts",
                "details": [{"type": a.alert_type, "severity": a.severity, "created": a.created_at.isoformat()} 
                           for a in quality_alerts]
            })
        
        # Check usage patterns
        usage_analytics = self.catalog_service.get_usage_analytics(asset_id, days=7)
        if usage_analytics.get("total_accesses", 0) == 0:
            investigation["findings"].append({
                "category": "usage",
                "finding": "No recent usage detected - asset may be stale or abandoned",
                "details": usage_analytics
            })
        
        # Generate recommendations
        recommendations = []
        if len(investigation["findings"]) == 0:
            recommendations.append("No immediate issues detected. Consider implementing proactive monitoring.")
        else:
            for finding in investigation["findings"]:
                if finding["category"] == "lineage":
                    recommendations.append("Review upstream data sources and their quality controls")
                elif finding["category"] == "pipeline_health":
                    recommendations.append("Investigate pipeline execution logs and resource utilization")
                elif finding["category"] == "quality_predictions":
                    recommendations.append("Address predicted quality issues before they impact downstream systems")
                elif finding["category"] == "usage":
                    recommendations.append("Verify if asset is still needed or should be deprecated")
        
        investigation["recommendations"] = recommendations
        investigation["investigation_summary"] = f"Found {len(investigation['findings'])} potential issues across {len(set(f['category'] for f in investigation['findings']))} categories"
        
        return investigation