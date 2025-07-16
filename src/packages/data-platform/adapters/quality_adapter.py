"""Adapter for quality service integration in data-platform package."""

from typing import Dict, Any, List, Optional
from ...interfaces.data_quality_interface import (
    DataQualityInterface, QualityReport, QualityIssue, QualityLevel
)


class DataPlatformQualityAdapter:
    """Adapter for data platform quality orchestration using interfaces."""
    
    def __init__(self, quality_service: DataQualityInterface):
        """Initialize with quality service interface."""
        self.quality_service = quality_service
    
    def orchestrate_quality_pipeline(self, dataset_id: str) -> Dict[str, Any]:
        """Orchestrate quality pipeline for data platform."""
        report = self.quality_service.assess_dataset_quality(dataset_id)
        recommendations = self.quality_service.get_quality_recommendations(dataset_id)
        
        return {
            "pipeline_id": f"quality_pipeline_{dataset_id}",
            "quality_report": {
                "score": report.overall_score,
                "level": report.quality_level.value,
                "issues": [
                    {
                        "type": issue.type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "columns": issue.affected_columns,
                        "recommendation": issue.recommendation,
                    }
                    for issue in report.issues
                ]
            },
            "recommendations": recommendations,
            "next_steps": self._generate_next_steps(report),
            "automation_ready": report.overall_score > 0.7,
        }
    
    def monitor_quality_across_pipelines(self, dataset_ids: List[str]) -> List[Dict[str, Any]]:
        """Monitor quality across multiple data pipelines."""
        results = []
        
        for dataset_id in dataset_ids:
            report = self.quality_service.assess_dataset_quality(dataset_id)
            
            results.append({
                "dataset_id": dataset_id,
                "quality_score": report.overall_score,
                "quality_level": report.quality_level.value,
                "critical_issues": len([
                    issue for issue in report.issues
                    if issue.severity > 0.8
                ]),
                "status": "healthy" if report.overall_score > 0.8 else "needs_attention",
                "last_checked": report.metadata.get("timestamp", ""),
            })
        
        return results
    
    def get_quality_dashboard_data(self, dataset_id: str) -> Dict[str, Any]:
        """Get quality dashboard data for data platform."""
        report = self.quality_service.assess_dataset_quality(dataset_id)
        trends = self.quality_service.monitor_quality_trends(dataset_id)
        
        return {
            "current_quality": {
                "score": report.overall_score,
                "level": report.quality_level.value,
                "issues_count": len(report.issues),
            },
            "trends": [
                {
                    "timestamp": trend.metadata.get("timestamp", ""),
                    "score": trend.overall_score,
                    "level": trend.quality_level.value,
                }
                for trend in trends[-30:]  # Last 30 for dashboard
            ],
            "issue_breakdown": self._categorize_issues(report.issues),
            "recommendations": report.recommendations,
        }
    
    def _generate_next_steps(self, report: QualityReport) -> List[str]:
        """Generate next steps based on quality report."""
        next_steps = []
        
        if report.overall_score < 0.5:
            next_steps.append("Immediate data quality remediation required")
        elif report.overall_score < 0.8:
            next_steps.append("Schedule data quality improvement tasks")
        
        high_severity_issues = [
            issue for issue in report.issues
            if issue.severity > 0.8
        ]
        
        if high_severity_issues:
            next_steps.append("Address high severity issues first")
        
        next_steps.extend(report.recommendations[:3])  # Top 3 recommendations
        
        return next_steps
    
    def _categorize_issues(self, issues: List[QualityIssue]) -> Dict[str, int]:
        """Categorize issues by type."""
        categories = {}
        
        for issue in issues:
            issue_type = issue.type
            if issue_type not in categories:
                categories[issue_type] = 0
            categories[issue_type] += 1
        
        return categories