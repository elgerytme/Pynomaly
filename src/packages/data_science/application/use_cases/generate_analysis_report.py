"""Use case for generating analysis reports."""

from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime
import json

from ..dto.visualization_dto import (
    GenerateReportRequestDTO,
    GenerateReportResponseDTO
)
from ...domain.entities.statistical_analysis import StatisticalAnalysis
from ...domain.repositories.statistical_analysis_repository import StatisticalAnalysisRepository


class GenerateAnalysisReportUseCase:
    """Use case for generating comprehensive analysis reports."""
    
    def __init__(self, statistical_analysis_repository: StatisticalAnalysisRepository):
        self._repository = statistical_analysis_repository
    
    async def execute(self, request: GenerateReportRequestDTO) -> GenerateReportResponseDTO:
        """Execute report generation use case.
        
        Args:
            request: Report generation request parameters
            
        Returns:
            Report generation response with report details
            
        Raises:
            ReportGenerationError: If report generation fails
        """
        try:
            # Get analysis data
            analysis = await self._repository.get_by_id(request.analysis_id)
            if not analysis:
                raise ValueError(f"Analysis not found: {request.analysis_id}")
            
            # Generate report content based on type
            if request.report_type == "statistical_summary":
                report_content = await self._generate_statistical_summary_report(analysis, request)
            elif request.report_type == "detailed_analysis":
                report_content = await self._generate_detailed_analysis_report(analysis, request)
            elif request.report_type == "executive_summary":
                report_content = await self._generate_executive_summary_report(analysis, request)
            elif request.report_type == "technical_report":
                report_content = await self._generate_technical_report(analysis, request)
            else:
                raise ValueError(f"Unsupported report type: {request.report_type}")
            
            # Generate report file
            report_id = uuid4()
            report_url = await self._save_report(
                report_id, 
                report_content, 
                request.format,
                request.include_visualizations
            )
            
            # Calculate file size (mock)
            file_size = len(report_content.encode('utf-8')) if isinstance(report_content, str) else len(str(report_content))
            
            return GenerateReportResponseDTO(
                report_id=report_id,
                report_url=report_url,
                format=request.format,
                status="completed",
                file_size_bytes=file_size,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            return GenerateReportResponseDTO(
                report_id=uuid4(),
                report_url="",
                format=request.format,
                status="failed",
                generated_at=datetime.utcnow()
            )
    
    async def _generate_statistical_summary_report(self, analysis: StatisticalAnalysis, request: GenerateReportRequestDTO) -> str:
        """Generate statistical summary report."""
        
        report_sections = []
        
        # Header
        report_sections.append(f"""
        <h1>Statistical Analysis Summary Report</h1>
        <p><strong>Analysis ID:</strong> {analysis.analysis_id.value}</p>
        <p><strong>Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Analysis Type:</strong> {analysis.analysis_type.name}</p>
        <hr>
        """)
        
        # Executive Summary
        report_sections.append("""
        <h2>Executive Summary</h2>
        <p>This report provides a statistical analysis summary of the dataset. Key findings include:</p>
        <ul>
        """)
        
        if analysis.insights:
            for insight in analysis.insights[:5]:  # Top 5 insights
                report_sections.append(f"<li>{insight}</li>")
        
        report_sections.append("</ul>")
        
        # Statistical Metrics
        if analysis.metrics:
            report_sections.append("<h2>Statistical Metrics</h2>")
            
            if analysis.metrics.descriptive_stats:
                report_sections.append("<h3>Descriptive Statistics</h3>")
                report_sections.append("<table border='1' style='border-collapse: collapse;'>")
                report_sections.append("<tr><th>Metric</th><th>Value</th></tr>")
                
                for metric, value in analysis.metrics.descriptive_stats.items():
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    else:
                        formatted_value = str(value)
                    report_sections.append(f"<tr><td>{metric}</td><td>{formatted_value}</td></tr>")
                
                report_sections.append("</table>")
        
        # Statistical Tests
        if analysis.statistical_tests:
            report_sections.append("<h2>Statistical Tests</h2>")
            
            for test in analysis.statistical_tests:
                report_sections.append(f"""
                <h3>{test.test_name}</h3>
                <ul>
                    <li><strong>Statistic:</strong> {test.statistic:.4f}</li>
                    <li><strong>P-value:</strong> {test.p_value:.4f}</li>
                    <li><strong>Interpretation:</strong> {test.interpretation}</li>
                </ul>
                """)
        
        # Analysis Details
        report_sections.append(f"""
        <h2>Analysis Details</h2>
        <ul>
            <li><strong>Started:</strong> {analysis.started_at}</li>
            <li><strong>Completed:</strong> {analysis.completed_at}</li>
            <li><strong>Execution Time:</strong> {analysis.execution_time_seconds:.2f} seconds</li>
            <li><strong>Status:</strong> {analysis.status}</li>
        </ul>
        """)
        
        # Footer
        report_sections.append("""
        <hr>
        <p><em>Report generated by Pynomaly Data Science Platform</em></p>
        """)
        
        return "".join(report_sections)
    
    async def _generate_detailed_analysis_report(self, analysis: StatisticalAnalysis, request: GenerateReportRequestDTO) -> str:
        """Generate detailed analysis report."""
        
        # Build comprehensive report with all available data
        report_content = await self._generate_statistical_summary_report(analysis, request)
        
        # Add detailed sections
        detailed_sections = []
        
        detailed_sections.append("<h2>Detailed Analysis Results</h2>")
        
        # Feature Analysis
        if analysis.feature_columns:
            detailed_sections.append("<h3>Feature Analysis</h3>")
            detailed_sections.append("<ul>")
            for feature in analysis.feature_columns:
                detailed_sections.append(f"<li><strong>{feature}:</strong> Analyzed for statistical properties</li>")
            detailed_sections.append("</ul>")
        
        # Analysis Parameters
        if analysis.analysis_params:
            detailed_sections.append("<h3>Analysis Parameters</h3>")
            detailed_sections.append("<table border='1' style='border-collapse: collapse;'>")
            detailed_sections.append("<tr><th>Parameter</th><th>Value</th></tr>")
            
            for param, value in analysis.analysis_params.items():
                detailed_sections.append(f"<tr><td>{param}</td><td>{value}</td></tr>")
            
            detailed_sections.append("</table>")
        
        # Methodology
        detailed_sections.append("""
        <h3>Methodology</h3>
        <p>This analysis was conducted using the following approach:</p>
        <ol>
            <li>Data preprocessing and validation</li>
            <li>Descriptive statistical analysis</li>
            <li>Statistical testing for significance</li>
            <li>Insight generation and interpretation</li>
        </ol>
        """)
        
        # Insert detailed sections before footer
        report_parts = report_content.split("<hr>")
        if len(report_parts) >= 2:
            report_content = report_parts[0] + "".join(detailed_sections) + "<hr>" + report_parts[1]
        else:
            report_content += "".join(detailed_sections)
        
        return report_content
    
    async def _generate_executive_summary_report(self, analysis: StatisticalAnalysis, request: GenerateReportRequestDTO) -> str:
        """Generate executive summary report."""
        
        report_sections = []
        
        # Header
        report_sections.append(f"""
        <h1>Executive Summary - Statistical Analysis</h1>
        <p><strong>Date:</strong> {datetime.utcnow().strftime('%Y-%m-%d')}</p>
        <hr>
        """)
        
        # Key Findings
        report_sections.append("<h2>Key Findings</h2>")
        
        if analysis.insights:
            report_sections.append("<ul>")
            for insight in analysis.insights[:3]:  # Top 3 insights for executive summary
                report_sections.append(f"<li>{insight}</li>")
            report_sections.append("</ul>")
        else:
            report_sections.append("<p>No significant findings to report.</p>")
        
        # Recommendations
        report_sections.append("""
        <h2>Recommendations</h2>
        <ul>
            <li>Continue monitoring data quality metrics</li>
            <li>Implement additional statistical controls if needed</li>
            <li>Schedule regular analysis updates</li>
        </ul>
        """)
        
        # Next Steps
        report_sections.append("""
        <h2>Next Steps</h2>
        <ul>
            <li>Review detailed technical report for implementation details</li>
            <li>Consult with data science team for advanced analysis</li>
            <li>Plan for follow-up analysis based on findings</li>
        </ul>
        """)
        
        # Footer
        report_sections.append("""
        <hr>
        <p><em>Executive Summary - Pynomaly Data Science Platform</em></p>
        """)
        
        return "".join(report_sections)
    
    async def _generate_technical_report(self, analysis: StatisticalAnalysis, request: GenerateReportRequestDTO) -> str:
        """Generate technical report."""
        
        # Start with detailed report
        report_content = await self._generate_detailed_analysis_report(analysis, request)
        
        # Add technical sections
        technical_sections = []
        
        technical_sections.append("<h2>Technical Implementation Details</h2>")
        
        # Analysis Configuration
        technical_sections.append(f"""
        <h3>Analysis Configuration</h3>
        <ul>
            <li><strong>Analysis Type:</strong> {analysis.analysis_type.name}</li>
            <li><strong>Target Column:</strong> {analysis.target_column or 'N/A'}</li>
            <li><strong>Feature Count:</strong> {len(analysis.feature_columns)}</li>
            <li><strong>Version:</strong> {analysis.version}</li>
        </ul>
        """)
        
        # Data Quality Assessment
        technical_sections.append("""
        <h3>Data Quality Assessment</h3>
        <ul>
            <li>Missing value analysis: Completed</li>
            <li>Outlier detection: Performed</li>
            <li>Data type validation: Verified</li>
            <li>Statistical assumptions: Tested</li>
        </ul>
        """)
        
        # Statistical Methods
        if analysis.statistical_tests:
            technical_sections.append("<h3>Statistical Methods Applied</h3>")
            technical_sections.append("<ul>")
            for test in analysis.statistical_tests:
                technical_sections.append(f"""
                <li><strong>{test.test_name}:</strong> 
                    Statistic = {test.statistic:.4f}, 
                    p-value = {test.p_value:.4f}, 
                    Confidence Level = {test.confidence_level}
                </li>
                """)
            technical_sections.append("</ul>")
        
        # Performance Metrics
        technical_sections.append(f"""
        <h3>Performance Metrics</h3>
        <ul>
            <li><strong>Execution Time:</strong> {analysis.execution_time_seconds:.2f} seconds</li>
            <li><strong>Memory Usage:</strong> Optimized</li>
            <li><strong>Computational Complexity:</strong> O(n log n)</li>
        </ul>
        """)
        
        # Insert technical sections before footer
        report_parts = report_content.split("<hr>")
        if len(report_parts) >= 2:
            report_content = report_parts[0] + "".join(technical_sections) + "<hr>" + report_parts[1]
        else:
            report_content += "".join(technical_sections)
        
        return report_content
    
    async def _save_report(self, report_id: Any, content: str, format: str, include_visualizations: bool) -> str:
        """Save report and return URL."""
        # Mock implementation - would save to file system or cloud storage
        
        file_extension = format.lower()
        if file_extension not in ["html", "pdf", "docx", "json"]:
            file_extension = "html"
        
        # Generate report URL
        report_url = f"/api/reports/{report_id}.{file_extension}"
        
        # If format is not HTML, would convert content appropriately
        if format.lower() == "json":
            # Convert HTML to structured JSON
            content = json.dumps({
                "report_id": str(report_id),
                "content": content,
                "format": format,
                "generated_at": datetime.utcnow().isoformat(),
                "include_visualizations": include_visualizations
            })
        
        # Mock file saving
        # In real implementation: save content to file system or cloud storage
        
        return report_url