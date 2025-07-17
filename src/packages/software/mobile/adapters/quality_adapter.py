"""Adapter for quality service integration in mobile package."""

from typing import Dict, Any, List, Optional
from ...interfaces.data_quality_interface import (
    DataQualityInterface, QualityReport, QualityIssue, QualityLevel
)


class MobileQualityAdapter:
    """Adapter for mobile quality monitoring using interfaces."""
    
    def __init__(self, quality_service: DataQualityInterface):
        """Initialize with quality service interface."""
        self.quality_service = quality_service
    
    def get_mobile_quality_summary(self, dataset_id: str) -> Dict[str, Any]:
        """Get mobile-optimized quality summary."""
        report = self.quality_service.assess_data_collection_quality(data_collection_id)
        
        return {
            "overall_score": report.overall_score,
            "quality_level": report.quality_level.value,
            "issues_count": len(report.issues),
            "critical_issues": [
                issue for issue in report.issues 
                if issue.severity > 0.8
            ],
            "recommendations": report.recommendations[:3],  # Top 3 for mobile
            "last_updated": report.metadata.get("timestamp", ""),
        }
    
    def get_quality_alerts(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get quality alerts for mobile notifications."""
        report = self.quality_service.assess_data_collection_quality(data_collection_id)
        
        alerts = []
        for issue in report.issues:
            if issue.severity > 0.7:  # Only high severity issues
                alerts.append({
                    "type": issue.type,
                    "severity": issue.severity,
                    "message": issue.description,
                    "action": issue.recommendation,
                    "affected_columns": issue.affected_columns,
                })
        
        return alerts
    
    def get_quality_trends(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get quality trends for mobile dashboard."""
        trends = self.quality_service.monitor_quality_trends(data_collection_id)
        
        return [
            {
                "timestamp": trend.metadata.get("timestamp", ""),
                "score": trend.overall_score,
                "level": trend.quality_level.value,
                "issues_count": len(trend.issues),
            }
            for trend in trends[-10:]  # Last 10 for mobile
        ]