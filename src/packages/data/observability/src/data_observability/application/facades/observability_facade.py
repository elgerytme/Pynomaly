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
    
    async def track_data_transformation(
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
        try:
            await self.lineage_service.track_data_transformation(
                lineage_id=lineage_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                transform_logic=transform_logic,
                column_mapping=column_mapping,
                execution_time=execution_time,
                error_rate=error_rate
            )
            print(f"Successfully tracked data transformation for lineage {lineage_id}")
        except Exception as e:
            print(f"Error tracking data transformation for lineage {lineage_id}: {e}")
            raise # Re-raise for higher-level handling if needed
    
    async def analyze_impact(self, lineage_id: UUID, node_id: UUID) -> Dict[str, Any]:
        """Analyze the impact of changes to a data asset."""
        try:
            result = await self.lineage_service.get_impact_analysis(lineage_id, node_id)
            print(f"Successfully analyzed impact for node {node_id} in lineage {lineage_id}")
            return result
        except Exception as e:
            print(f"Error analyzing impact for node {node_id} in lineage {lineage_id}: {e}")
            raise # Re-raise for higher-level handling if needed
    
    async def get_lineage_graph(self, lineage_id: UUID) -> Optional[DataLineage]:
        """Get the complete lineage graph for a lineage ID."""
        try:
            graph = await self.lineage_service.get_lineage(lineage_id)
            if graph:
                print(f"Successfully retrieved lineage graph for ID {lineage_id}")
            else:
                print(f"Lineage graph with ID {lineage_id} not found.")
            return graph
        except Exception as e:
            print(f"Error retrieving lineage graph for ID {lineage_id}: {e}")
            raise # Re-raise for higher-level handling if needed
    
    async def find_data_path(self, lineage_id: UUID, source_node_id: UUID, target_node_id: UUID) -> Dict[str, Any]:
        """Find the path between two data assets."""
        try:
            path = await self.lineage_service.get_data_flow_path(lineage_id, source_node_id, target_node_id)
            print(f"Successfully found data path from {source_node_id} to {target_node_id} in lineage {lineage_id}")
            return path
        except Exception as e:
            print(f"Error finding data path from {source_node_id} to {target_node_id} in lineage {lineage_id}: {e}")
            raise # Re-raise for higher-level handling if needed

    # ==== Pipeline Health Operations ====

    async def monitor_pipeline_health(
        self,
        pipeline_id: UUID,
        metrics: Dict[str, float],
        context: Dict[str, Any] = None
    ) -> Optional[PipelineHealth]:
        """Monitor pipeline health with current metrics."""
        try:
            # First ensure pipeline is registered
            health = await self.health_service.get_pipeline_health(pipeline_id)
            if not health:
                # Register pipeline if not exists
                health = await self.health_service.register_pipeline(pipeline_id, f"Pipeline {pipeline_id}")
                print(f"Registered new pipeline: {pipeline_id}")
            
            # Record metrics
            for metric_name, value in metrics.items():
                metric_type = MetricType.THROUGHPUT # Default
                unit = "count" # Default

                # Infer metric type and unit based on common keywords
                lower_metric_name = metric_name.lower()
                if "error_rate" in lower_metric_name or "error" in lower_metric_name:
                    metric_type = MetricType.ERROR_RATE
                    unit = "percentage"
                elif "latency" in lower_metric_name or "duration" in lower_metric_name:
                    metric_type = MetricType.LATENCY
                    unit = "ms"
                elif "throughput" in lower_metric_name or "rate" in lower_metric_name:
                    metric_type = MetricType.THROUGHPUT
                    unit = "count/s"
                elif "resource_usage" in lower_metric_name or "cpu" in lower_metric_name or "memory" in lower_metric_name:
                    metric_type = MetricType.RESOURCE_USAGE
                    unit = "%"
                elif "availability" in lower_metric_name:
                    metric_type = MetricType.AVAILABILITY
                    unit = "%"
                elif "quality" in lower_metric_name:
                    metric_type = MetricType.DATA_QUALITY
                    unit = "score"
                elif "performance" in lower_metric_name:
                    metric_type = MetricType.PERFORMANCE
                    unit = "score"

                await self.health_service.record_metric(
                    pipeline_id=pipeline_id,
                    metric_type=metric_type,
                    name=metric_name,
                    value=value,
                    unit=unit,
                    labels=context
                )
                print(f"Recorded metric {metric_name} for pipeline {pipeline_id}")
            
            return health
        except Exception as e:
            print(f"Error monitoring pipeline health for {pipeline_id}: {e}")
            raise # Re-raise for higher-level handling if needed

    async def get_pipeline_alerts(self, pipeline_id: UUID = None) -> List[PipelineAlert]:
        """Get active pipeline alerts."""
        try:
            alerts = await self.health_service.get_active_alerts(pipeline_id)
            print(f"Retrieved {len(alerts)} active alerts for pipeline {pipeline_id or 'all pipelines'}")
            return alerts
        except Exception as e:
            print(f"Error retrieving pipeline alerts for {pipeline_id or 'all pipelines'}: {e}")
            raise # Re-raise for higher-level handling if needed

    async def get_pipeline_health_summary(self, pipeline_id: UUID) -> Dict[str, Any]:
        """Get comprehensive health summary for a pipeline."""
        try:
            health = await self.health_service.get_pipeline_health(pipeline_id)
            if not health:
                print(f"Pipeline {pipeline_id} not found for health summary.")
                return {}
            
            alerts = await self.get_pipeline_alerts(pipeline_id)
            
            summary = {
                "pipeline_id": str(pipeline_id),
                "health_score": health.get_health_score(),
                "status": health.status,
                "last_updated": health.last_updated.isoformat(),
                "availability": health.availability_percentage,
                "error_rate": health.error_rate,
                "recent_alerts": len([a for a in alerts 
                                   if a.created_at >= datetime.utcnow() - timedelta(hours=24)])
            }
            print(f"Generated health summary for pipeline {pipeline_id}")
            return summary
        except Exception as e:
            print(f"Error generating health summary for pipeline {pipeline_id}: {e}")
            raise # Re-raise for higher-level handling if needed

    # ==== Data Catalog Operations ====

    async def register_data_asset(
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
        try:
            asset = await self.catalog_service.register_asset(
                name=name,
                asset_type=asset_type,
                location=location,
                data_format=data_format,
                description=description,
                owner=owner,
                domain=domain,
                **kwargs
            )
            print(f"Successfully registered data asset: {asset.name} ({asset.id})")
            return asset
        except Exception as e:
            print(f"Error registering data asset {name}: {e}")
            raise # Re-raise for higher-level handling if needed

    async def discover_data_assets(self, query: str, limit: int = 20) -> List[DataCatalogEntry]:
        """Discover data assets using intelligent search."""
        try:
            assets = await self.catalog_service.search_assets(query=query, limit=limit)
            print(f"Discovered {len(assets)} data assets for query '{query}'")
            return assets
        except Exception as e:
            print(f"Error discovering data assets for query '{query}': {e}")
            raise # Re-raise for higher-level handling if needed

    async def get_asset_recommendations(self, asset_id: UUID) -> List[Tuple[DataCatalogEntry, float]]:
        """Get recommendations for similar or related assets."""
        try:
            recommendations = await self.catalog_service.discover_similar_assets(asset_id)
            print(f"Generated {len(recommendations)} recommendations for asset {asset_id}")
            return recommendations
        except Exception as e:
            print(f"Error getting asset recommendations for {asset_id}: {e}")
            raise # Re-raise for higher-level handling if needed

    async def track_asset_usage(
        self,
        asset_id: UUID,
        user_id: str = None,
        user_name: str = None,
        usage_type: str = "read",
        query: str = None,
        rows_accessed: int = None,
        columns_accessed: List[str] = None,
        duration_ms: int = None,
        application: str = None,
        purpose: str = None
    ) -> None:
        """Track usage of a data asset."""
        try:
            await self.catalog_service.record_usage(
                asset_id=asset_id,
                user_id=user_id,
                user_name=user_name,
                usage_type=usage_type,
                query=query,
                rows_accessed=rows_accessed,
                columns_accessed=columns_accessed,
                duration_ms=duration_ms,
                application=application,
                purpose=purpose
            )
            print(f"Successfully tracked usage for asset {asset_id} by user {user_id}")
        except Exception as e:
            print(f"Error tracking usage for asset {asset_id}: {e}")
            raise # Re-raise for higher-level handling if needed

    # ==== Predictive Quality Operations ====

    async def predict_quality_issues(
        self,
        asset_id: UUID,
        prediction_type: PredictionType,
        target_time: datetime,
        **kwargs
    ) -> QualityPrediction:
        """Predict potential quality issues for an asset."""
        try:
            prediction = await self.quality_service.create_prediction(
                asset_id=asset_id,
                prediction_type=prediction_type,
                target_time=target_time,
                **kwargs
            )
            print(f"Created quality prediction for asset {asset_id} (Type: {prediction_type.value})")
            return prediction
        except Exception as e:
            print(f"Error predicting quality issues for asset {asset_id}: {e}")
            raise # Re-raise for higher-level handling if needed

    async def forecast_quality_metrics(
        self,
        asset_id: UUID,
        metric_type: str,
        horizon_hours: int = 24,
        **kwargs
    ) -> QualityForecast:
        """Forecast quality metrics over time."""
        try:
            forecast = await self.quality_service.create_forecast(
                asset_id=asset_id,
                metric_type=metric_type,
                horizon_hours=horizon_hours,
                **kwargs
            )
            print(f"Created quality forecast for asset {asset_id} (Metric: {metric_type})")
            return forecast
        except Exception as e:
            print(f"Error forecasting quality metrics for asset {asset_id}: {e}")
            raise # Re-raise for higher-level handling if needed

    async def analyze_quality_trends(
        self,
        asset_id: UUID,
        metric_type: str,
        days: int = 30
    ) -> QualityTrend:
        """Analyze quality trends for an asset."""
        try:
            trend = await self.quality_service.analyze_trends(
                asset_id=asset_id,
                metric_type=metric_type,
                days=days
            )
            print(f"Analyzed quality trends for asset {asset_id} (Metric: {metric_type})")
            return trend
        except Exception as e:
            print(f"Error analyzing quality trends for asset {asset_id}: {e}")
            raise # Re-raise for higher-level handling if needed

    async def add_quality_metric(
        self,
        asset_id: UUID,
        metric_type: str,
        value: float,
        timestamp: datetime = None
    ) -> None:
        """Add a quality metric data point."""
        try:
            await self.quality_service.add_metric_point(
                asset_id=asset_id,
                metric_type=metric_type,
                value=value,
                timestamp=timestamp
            )
            print(f"Added quality metric point for asset {asset_id} (Metric: {metric_type}, Value: {value})")
        except Exception as e:
            print(f"Error adding quality metric point for asset {asset_id}: {e}")
            raise # Re-raise for higher-level handling if needed

    # ==== Cross-Service Operations ====

    async def get_comprehensive_asset_view(self, asset_id: UUID) -> Dict[str, Any]:
        """Get a comprehensive view of an asset across all observability dimensions."""
        try:
            # Get catalog information
            catalog_entry = await self.catalog_service.get_asset(asset_id)
            if not catalog_entry:
                print(f"Asset {asset_id} not found in catalog for comprehensive view.")
                return {"error": f"Asset {asset_id} not found in catalog"}
            
            # Get lineage information
            lineage_info = {"upstream_count": 0, "downstream_count": 0, "total_connected_assets": 0}
            asset_lineage_nodes = await self.lineage_service.get_nodes_by_asset_id(asset_id)
            if asset_lineage_nodes:
                main_node = asset_lineage_nodes[0]
                lineage_id = main_node.metadata.properties.get("lineage_id")
                if lineage_id:
                    lineage_graph = await self.lineage_service.get_lineage(lineage_id)
                    if lineage_graph:
                        lineage_info["upstream_count"] = len(lineage_graph.get_upstream_nodes(main_node.id))
                        lineage_info["downstream_count"] = len(lineage_graph.get_downstream_nodes(main_node.id))
                        lineage_info["total_connected_assets"] = len(lineage_graph.nodes) # Approximation
            
            # Get usage analytics
            usage_analytics = self.catalog_service.get_usage_analytics(asset_id)
            
            # Get quality predictions
            quality_alerts = await self.quality_service.get_active_alerts(asset_id)
            
            # Get pipeline health (if asset is associated with a pipeline)
            pipeline_health = None
            if hasattr(catalog_entry, 'pipeline_id') and catalog_entry.pipeline_id:
                pipeline_health = await self.health_service.get_pipeline_health(catalog_entry.pipeline_id)
            
            view = {
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
                    "health_score": pipeline_health.get_health_score() if pipeline_health else None,
                    "status": pipeline_health.status if pipeline_health else "unknown",
                    "last_updated": pipeline_health.last_updated.isoformat() if pipeline_health else None
                } if pipeline_health else None
            }
            print(f"Generated comprehensive asset view for {asset_id}")
            return view
        except Exception as e:
            print(f"Error getting comprehensive asset view for {asset_id}: {e}")
            raise # Re-raise for higher-level handling if needed

    async def get_data_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive data health dashboard."""
        try:
            # Catalog statistics
            catalog_stats = await self.catalog_service.get_catalog_statistics()
            
            # Pipeline health overview
            all_pipelines = await self.health_service.get_all_pipelines()
            pipeline_stats = {
                "total_pipelines": len(all_pipelines),
                "healthy_pipelines": len([p for p in all_pipelines if p.get_health_score() >= 0.8]),
                "degraded_pipelines": len([p for p in all_pipelines if 0.6 <= p.get_health_score() < 0.8]),
                "unhealthy_pipelines": len([p for p in all_pipelines if p.get_health_score() < 0.6])
            }
            
            # Quality predictions overview
            all_alerts = await self.quality_service.get_active_alerts()
            quality_stats = {
                "total_active_alerts": len(all_alerts),
                "critical_alerts": len([a for a in all_alerts if a.severity == "critical"]),
                "high_priority_alerts": len([a for a in all_alerts if a.severity == "high"]),
                "recent_predictions": len([a for a in all_alerts 
                                        if a.created_at >= datetime.utcnow() - timedelta(hours=24)])
            }
            
            # Lineage statistics - simplified
            all_lineages = await self.lineage_service.list_lineages()
            lineage_stats = {
                "total_lineages": len(all_lineages),
                "total_nodes": sum(len(l.nodes) for l in all_lineages),
                "total_edges": sum(len(l.edges) for l in all_lineages)
            }
            
            dashboard = {
                "catalog": catalog_stats,
                "pipeline_health": pipeline_stats,
                "quality_predictions": quality_stats,
                "lineage": lineage_stats,
                "dashboard_generated_at": datetime.utcnow().isoformat()
            }
            print("Generated comprehensive data health dashboard.")
            return dashboard
        except Exception as e:
            print(f"Error generating data health dashboard: {e}")
            raise # Re-raise for higher-level handling if needed

    async def investigate_data_issue(
        self,
        asset_id: UUID,
        issue_type: str,
        severity: str = "medium"
    ) -> Dict[str, Any]:
        """Investigate a data issue across all observability dimensions."""
        try:
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
            catalog_entry = await self.catalog_service.get_asset(asset_id)
            # For pipeline health check, we'd need to establish asset->pipeline relationship
            # This is simplified for now
            
            # Check quality predictions
            quality_alerts = await self.quality_service.get_active_alerts(asset_id)
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
            
            print(f"Completed investigation for asset {asset_id} with {len(investigation['findings'])} findings.")
            return investigation
        except Exception as e:
            print(f"Error investigating data issue for asset {asset_id}: {e}")
            raise # Re-raise for higher-level handling if needed