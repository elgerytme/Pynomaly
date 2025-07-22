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
        lineage_id: UUID,
        source_node_id: UUID,
        target_node_id: UUID,
        transform_logic: str,
        column_mapping: Dict[str, str] = None,
        execution_time: float = None,
        error_rate: float = None
    ) -> None:
        """Track a data transformation between assets."""
        self.lineage_service.track_data_transformation(
            lineage_id=lineage_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            transform_logic=transform_logic,
            column_mapping=column_mapping,
            execution_time=execution_time,
            error_rate=error_rate
        )
    
    def analyze_impact(self, lineage_id: UUID, node_id: UUID) -> Dict[str, Any]:
        """Analyze the impact of changes to a data asset."""
        return self.lineage_service.get_impact_analysis(lineage_id, node_id)
    
    def get_lineage_graph(self, lineage_id: UUID) -> Optional[DataLineage]:
        """Get the complete lineage graph for a lineage ID."""
        return self.lineage_service.get_lineage(lineage_id)
    
    def find_data_path(self, lineage_id: UUID, source_node_id: UUID, target_node_id: UUID) -> Dict[str, Any]:
        """Find the path between two data assets."""
        return self.lineage_service.get_data_flow_path(lineage_id, source_node_id, target_node_id)
    
    # ==== Pipeline Health Operations ====
    
    def monitor_pipeline_health(
        self,
        pipeline_id: UUID,
        metrics: Dict[str, float],
        context: Dict[str, Any] = None
    ) -> Optional[PipelineHealth]:
        """Monitor pipeline health with current metrics."""
        # First ensure pipeline is registered
        health = self.health_service.get_pipeline_health(pipeline_id)
        if not health:
            # Register pipeline if not exists
            health = self.health_service.register_pipeline(pipeline_id, f"Pipeline {pipeline_id}")
        
        return health
    
    def get_pipeline_alerts(self, pipeline_id: UUID = None) -> List[PipelineAlert]:
        """Get active pipeline alerts."""
        if pipeline_id:
            health = self.health_service.get_pipeline_health(pipeline_id)
            return health.active_alerts if health else []
        else:
            # Get all alerts from all pipelines
            alerts = []
            for health in self.health_service.get_all_pipelines():
                alerts.extend(health.active_alerts)
            return alerts
    
    def get_pipeline_health_summary(self, pipeline_id: UUID) -> Dict[str, Any]:
        """Get comprehensive health summary for a pipeline."""
        health = self.health_service.get_pipeline_health(pipeline_id)
        if not health:
            return {}
        
        return {
            "pipeline_id": str(pipeline_id),
            "health_score": health.get_health_score(),
            "status": health.status,
            "last_updated": health.last_updated.isoformat(),
            "availability": health.availability_percentage,
            "error_rate": health.error_rate,
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
        
        # Get lineage information - simplified since we don't have direct asset->lineage mapping
        # This would need to be implemented based on your specific lineage tracking approach
        lineage_info = {"upstream_count": 0, "downstream_count": 0, "total_connected_assets": 0}
        
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
            "lineage": lineage_info,
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
        all_pipelines = self.health_service.get_all_pipelines()
        pipeline_stats = {
            "total_pipelines": len(all_pipelines),
            "healthy_pipelines": len([p for p in all_pipelines if p.get_health_score() >= 0.8]),
            "degraded_pipelines": len([p for p in all_pipelines if 0.6 <= p.get_health_score() < 0.8]),
            "unhealthy_pipelines": len([p for p in all_pipelines if p.get_health_score() < 0.6])
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
        
        # Lineage statistics - simplified
        all_lineages = self.lineage_service.list_lineages()
        lineage_stats = {
            "total_lineages": len(all_lineages),
            "total_nodes": sum(len(l.nodes) for l in all_lineages),
            "total_edges": sum(len(l.edges) for l in all_lineages)
        }
        
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
        
        # Check lineage for upstream issues - simplified since we need lineage_id
        # This would need to be implemented based on your specific asset->lineage mapping
        # For now, skip lineage analysis in investigation
        
        # Check pipeline health
        catalog_entry = self.catalog_service.get_asset(asset_id)
        # For pipeline health check, we'd need to establish asset->pipeline relationship
        # This is simplified for now
        
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
                if finding["category"] == "pipeline_health":
                    recommendations.append("Investigate pipeline execution logs and resource utilization")
                elif finding["category"] == "quality_predictions":
                    recommendations.append("Address predicted quality issues before they impact downstream systems")
                elif finding["category"] == "usage":
                    recommendations.append("Verify if asset is still needed or should be deprecated")
        
        investigation["recommendations"] = recommendations
        investigation["investigation_summary"] = f"Found {len(investigation['findings'])} potential issues across {len(set(f['category'] for f in investigation['findings']))} categories"
        
        return investigation