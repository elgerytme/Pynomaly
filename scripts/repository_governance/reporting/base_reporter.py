"""
Base class for repository governance reporters.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseReporter(ABC):
    """Base class for governance reporters."""
    
    def __init__(self, output_path: Optional[Path] = None):
        """Initialize the reporter."""
        self.output_path = output_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate_report(self, check_results: Dict[str, Any], fix_results: Dict[str, Any] = None) -> str:
        """Generate a report from check and fix results."""
        pass
    
    @abstractmethod
    def save_report(self, report_content: str, filename: str = None) -> bool:
        """Save the report to a file."""
        pass
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Name of the report format."""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for this report format."""
        pass
    
    def get_severity_color(self, severity: str) -> str:
        """Get color code for severity level."""
        colors = {
            "high": "#FF0000",      # Red
            "medium": "#FFA500",    # Orange
            "low": "#FFFF00",       # Yellow
            "info": "#00FF00"       # Green
        }
        return colors.get(severity, "#808080")  # Gray for unknown
    
    def get_severity_icon(self, severity: str) -> str:
        """Get icon for severity level."""
        icons = {
            "high": "🔴",
            "medium": "🟠",
            "low": "🟡",
            "info": "🟢"
        }
        return icons.get(severity, "⚪")
    
    def calculate_overall_score(self, check_results: Dict[str, Any]) -> float:
        """Calculate an overall governance score."""
        total_score = 0
        checker_count = 0
        
        for checker_name, result in check_results.items():
            if isinstance(result, dict) and "score" in result:
                total_score += result["score"]
                checker_count += 1
        
        return total_score / checker_count if checker_count > 0 else 0
    
    def get_score_grade(self, score: float) -> str:
        """Get letter grade for score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def get_score_color(self, score: float) -> str:
        """Get color for score."""
        if score >= 90:
            return "#00FF00"  # Green
        elif score >= 80:
            return "#90EE90"  # Light Green
        elif score >= 70:
            return "#FFFF00"  # Yellow
        elif score >= 60:
            return "#FFA500"  # Orange
        else:
            return "#FF0000"  # Red
    
    def format_timestamp(self, timestamp: str = None) -> str:
        """Format timestamp for display."""
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
        return timestamp
    
    def extract_summary_stats(self, check_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary statistics from check results."""
        stats = {
            "total_violations": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "info_severity": 0,
            "checkers_run": 0,
            "checkers_passed": 0,
            "checkers_failed": 0
        }
        
        for checker_name, result in check_results.items():
            if isinstance(result, dict):
                stats["checkers_run"] += 1
                
                violations = result.get("violations", [])
                if violations:
                    stats["checkers_failed"] += 1
                    stats["total_violations"] += len(violations)
                    
                    # Count by severity
                    for violation in violations:
                        severity = violation.get("severity", "info")
                        stats[f"{severity}_severity"] += 1
                else:
                    stats["checkers_passed"] += 1
        
        return stats
    
    def extract_fix_summary(self, fix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary statistics from fix results."""
        if not fix_results:
            return {
                "total_fixes_attempted": 0,
                "successful_fixes": 0,
                "failed_fixes": 0,
                "files_changed": 0,
                "fixers_run": 0
            }
        
        stats = {
            "total_fixes_attempted": 0,
            "successful_fixes": 0,
            "failed_fixes": 0,
            "files_changed": 0,
            "fixers_run": 0
        }
        
        for fixer_name, results in fix_results.items():
            if isinstance(results, list):
                stats["fixers_run"] += 1
                stats["total_fixes_attempted"] += len(results)
                
                for result in results:
                    if isinstance(result, dict):
                        if result.get("success", False):
                            stats["successful_fixes"] += 1
                        else:
                            stats["failed_fixes"] += 1
                        
                        # Count unique files changed
                        files_changed = result.get("files_changed", [])
                        stats["files_changed"] += len(files_changed)
        
        return stats
    
    def get_recommendations(self, check_results: Dict[str, Any]) -> List[str]:
        """Extract recommendations from check results."""
        all_recommendations = []
        
        for checker_name, result in check_results.items():
            if isinstance(result, dict) and "recommendations" in result:
                recommendations = result["recommendations"]
                if isinstance(recommendations, list):
                    all_recommendations.extend(recommendations)
        
        return all_recommendations
    
    def create_default_filename(self, prefix: str = "governance_report") -> str:
        """Create a default filename with timestamp."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{self.file_extension}"