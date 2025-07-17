"""
JSON reporter for repository governance.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .base_reporter import BaseReporter


class JSONReporter(BaseReporter):
    """JSON reporter for governance results."""
    
    def __init__(self, output_path: Optional[Path] = None, pretty_print: bool = True):
        """Initialize the JSON reporter."""
        super().__init__(output_path)
        self.pretty_print = pretty_print
    
    @property
    def format_name(self) -> str:
        """Name of the report format."""
        return "JSON"
    
    @property
    def file_extension(self) -> str:
        """File extension for this report format."""
        return "json"
    
    def generate_report(self, check_results: Dict[str, Any], fix_results: Dict[str, Any] = None) -> str:
        """Generate a JSON report from check and fix results."""
        report_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "report_format": self.format_name,
                "version": "1.0.0"
            },
            "summary": {
                "check_summary": self.extract_summary_stats(check_results),
                "fix_summary": self.extract_fix_summary(fix_results) if fix_results else None,
                "overall_score": self.calculate_overall_score(check_results),
                "grade": self.get_score_grade(self.calculate_overall_score(check_results))
            },
            "check_results": self._process_check_results(check_results),
            "fix_results": self._process_fix_results(fix_results) if fix_results else None,
            "recommendations": self.get_recommendations(check_results)
        }
        
        if self.pretty_print:
            return json.dumps(report_data, indent=2, ensure_ascii=False)
        else:
            return json.dumps(report_data, ensure_ascii=False)
    
    def save_report(self, report_content: str, filename: str = None) -> bool:
        """Save the report to a file."""
        try:
            if filename is None:
                filename = self.create_default_filename()
            
            if self.output_path is None:
                print(report_content)
                return True
            
            output_file = self.output_path / filename
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            output_file.write_text(report_content, encoding='utf-8')
            print(f"JSON report saved to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON report: {e}")
            return False
    
    def _process_check_results(self, check_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process check results for JSON serialization."""
        processed_results = {}
        
        for checker_name, result in check_results.items():
            if isinstance(result, dict):
                processed_results[checker_name] = {
                    "score": result.get("score", 0),
                    "violations": result.get("violations", []),
                    "total_violations": result.get("total_violations", 0),
                    "recommendations": result.get("recommendations", [])
                }
            else:
                processed_results[checker_name] = {
                    "score": 0,
                    "violations": [],
                    "total_violations": 0,
                    "recommendations": [],
                    "error": str(result)
                }
        
        return processed_results
    
    def _process_fix_results(self, fix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process fix results for JSON serialization."""
        processed_results = {}
        
        for fixer_name, results in fix_results.items():
            if isinstance(results, list):
                processed_results[fixer_name] = {
                    "results": results,
                    "summary": {
                        "total_attempts": len(results),
                        "successful": sum(1 for r in results if r.get("success", False)),
                        "failed": sum(1 for r in results if not r.get("success", False)),
                        "files_changed": sum(len(r.get("files_changed", [])) for r in results)
                    }
                }
            else:
                processed_results[fixer_name] = {
                    "results": [],
                    "summary": {
                        "total_attempts": 0,
                        "successful": 0,
                        "failed": 0,
                        "files_changed": 0
                    },
                    "error": str(results)
                }
        
        return processed_results
    
    def load_report(self, filename: str) -> Dict[str, Any]:
        """Load a JSON report from file."""
        try:
            if self.output_path is None:
                report_file = Path(filename)
            else:
                report_file = self.output_path / filename
            
            if not report_file.exists():
                raise FileNotFoundError(f"Report file not found: {report_file}")
            
            with open(report_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Failed to load JSON report: {e}")
            raise
    
    def merge_reports(self, reports: list) -> Dict[str, Any]:
        """Merge multiple JSON reports."""
        if not reports:
            return {}
        
        merged_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "report_format": self.format_name,
                "version": "1.0.0",
                "merged_from": len(reports)
            },
            "summary": {
                "check_summary": {"total_violations": 0, "checkers_run": 0},
                "fix_summary": {"total_fixes_attempted": 0, "successful_fixes": 0},
                "overall_score": 0,
                "grade": "F"
            },
            "check_results": {},
            "fix_results": {},
            "recommendations": []
        }
        
        # Merge check results
        all_checkers = set()
        for report in reports:
            if "check_results" in report:
                all_checkers.update(report["check_results"].keys())
        
        for checker in all_checkers:
            merged_violations = []
            total_score = 0
            checker_count = 0
            
            for report in reports:
                if "check_results" in report and checker in report["check_results"]:
                    checker_result = report["check_results"][checker]
                    merged_violations.extend(checker_result.get("violations", []))
                    total_score += checker_result.get("score", 0)
                    checker_count += 1
            
            merged_report["check_results"][checker] = {
                "score": total_score / checker_count if checker_count > 0 else 0,
                "violations": merged_violations,
                "total_violations": len(merged_violations),
                "recommendations": []
            }
        
        # Calculate merged summary
        merged_report["summary"]["overall_score"] = self.calculate_overall_score(merged_report["check_results"])
        merged_report["summary"]["grade"] = self.get_score_grade(merged_report["summary"]["overall_score"])
        
        return merged_report