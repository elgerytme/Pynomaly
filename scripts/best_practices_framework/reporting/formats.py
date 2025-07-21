"""
Report Format Classes
====================
Individual report format implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from ..core.base_validator import ValidationReport


class BaseReporter(ABC):
    """Base class for all report formatters"""
    
    @abstractmethod
    def generate(self, report: ValidationReport) -> Union[str, bytes, Dict[str, Any]]:
        """Generate report in specific format"""
        pass


class HTMLReporter(BaseReporter):
    """HTML format reporter"""
    
    def generate(self, report: ValidationReport) -> str:
        from .report_generator import ReportGenerator
        generator = ReportGenerator()
        return generator.generate_html_report(report)


class MarkdownReporter(BaseReporter):
    """Markdown format reporter"""
    
    def generate(self, report: ValidationReport) -> str:
        from .report_generator import ReportGenerator
        generator = ReportGenerator()
        return generator.generate_markdown_report(report)


class JSONReporter(BaseReporter):
    """JSON format reporter"""
    
    def generate(self, report: ValidationReport) -> Dict[str, Any]:
        from .report_generator import ReportGenerator
        generator = ReportGenerator()
        return generator.generate_json_report(report)


class SARIFReporter(BaseReporter):
    """SARIF format reporter"""
    
    def generate(self, report: ValidationReport) -> Dict[str, Any]:
        from .report_generator import ReportGenerator
        generator = ReportGenerator()
        return generator.generate_sarif_report(report)


class JUnitReporter(BaseReporter):
    """JUnit XML format reporter"""
    
    def generate(self, report: ValidationReport) -> str:
        from .report_generator import ReportGenerator
        generator = ReportGenerator()
        return generator.generate_junit_report(report)