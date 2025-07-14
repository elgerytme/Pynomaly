"""Validation result reporting and error handling service."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
import pandas as pd

from ...domain.services.validation_engine import ValidationResult, ValidationError, ValidationSeverity, ValidationCategory

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Report output formats."""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    MARKDOWN = "markdown"


class ReportLevel(str, Enum):
    """Report detail levels."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class ReportScope(str, Enum):
    """Report scope options."""
    ALL = "all"
    ERRORS_ONLY = "errors_only"
    WARNINGS_ONLY = "warnings_only"
    CRITICAL_ONLY = "critical_only"


@dataclass
class ReportConfiguration:
    """Configuration for validation report generation."""
    format: ReportFormat = ReportFormat.JSON
    level: ReportLevel = ReportLevel.DETAILED
    scope: ReportScope = ReportScope.ALL
    include_statistics: bool = True
    include_recommendations: bool = True
    include_error_samples: bool = True
    max_error_samples: int = 10
    group_by_rule: bool = True
    group_by_column: bool = False
    sort_by_severity: bool = True
    include_timestamps: bool = True
    include_metadata: bool = True


class ValidationReportGenerator:
    """Generate comprehensive validation reports in various formats."""
    
    def __init__(self, config: Optional[ReportConfiguration] = None):
        self.config = config or ReportConfiguration()
    
    def generate_report(
        self,
        results: List[ValidationResult],
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate validation report based on configuration."""
        
        filtered_results = self._filter_results(results)
        
        report = {
            'metadata': self._generate_metadata(results, dataset_info),
            'summary': self._generate_summary(filtered_results),
            'rule_analysis': self._generate_rule_analysis(filtered_results),
            'error_analysis': self._generate_error_analysis(filtered_results),
        }
        
        if self.config.include_statistics:
            report['statistics'] = self._generate_statistics(filtered_results)
        
        if self.config.include_recommendations:
            report['recommendations'] = self._generate_recommendations(filtered_results)
        
        if self.config.level == ReportLevel.COMPREHENSIVE:
            report['detailed_results'] = self._generate_detailed_results(filtered_results)
        
        return report
    
    def _filter_results(self, results: List[ValidationResult]) -> List[ValidationResult]:
        """Filter results based on report scope."""
        if self.config.scope == ReportScope.ALL:
            return results
        elif self.config.scope == ReportScope.ERRORS_ONLY:
            return [r for r in results if not r.passed]
        elif self.config.scope == ReportScope.WARNINGS_ONLY:
            return [r for r in results if r.severity == ValidationSeverity.WARNING]
        elif self.config.scope == ReportScope.CRITICAL_ONLY:
            return [r for r in results if r.severity == ValidationSeverity.CRITICAL]
        return results
    
    def _generate_metadata(
        self,
        results: List[ValidationResult],
        dataset_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate report metadata."""
        metadata = {
            'report_id': str(uuid4()),
            'generated_at': datetime.utcnow().isoformat(),
            'report_config': {
                'format': self.config.format.value,
                'level': self.config.level.value,
                'scope': self.config.scope.value
            },
            'validation_summary': {
                'total_rules': len(results),
                'rules_executed': len([r for r in results if r.metrics.records_processed > 0]),
                'execution_time_total_ms': sum(r.metrics.execution_time_ms for r in results)
            }
        }
        
        if dataset_info:
            metadata['dataset_info'] = dataset_info
        
        if self.config.include_timestamps:
            execution_times = [r.executed_at for r in results if r.executed_at]
            if execution_times:
                metadata['validation_period'] = {
                    'started_at': min(execution_times).isoformat(),
                    'completed_at': max(execution_times).isoformat()
                }
        
        return metadata
    
    def _generate_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate high-level summary."""
        total_rules = len(results)
        passed_rules = sum(1 for r in results if r.passed)
        failed_rules = total_rules - passed_rules
        
        total_errors = sum(len(r.errors) for r in results)
        
        severity_counts = {}
        category_counts = {}
        
        for result in results:
            for error in result.errors:
                severity = error.severity.value
                category = error.category.value
                
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Calculate overall quality score
        if total_rules > 0:
            overall_pass_rate = passed_rules / total_rules
            quality_score = max(0, min(100, overall_pass_rate * 100))
        else:
            quality_score = 0
        
        return {
            'overall_status': 'PASSED' if failed_rules == 0 else 'FAILED',
            'quality_score': round(quality_score, 2),
            'rules': {
                'total': total_rules,
                'passed': passed_rules,
                'failed': failed_rules,
                'pass_rate': round((passed_rules / total_rules) * 100, 2) if total_rules > 0 else 0
            },
            'errors': {
                'total': total_errors,
                'by_severity': severity_counts,
                'by_category': category_counts
            },
            'performance': {
                'avg_execution_time_ms': round(
                    sum(r.metrics.execution_time_ms for r in results) / len(results), 2
                ) if results else 0,
                'total_records_processed': sum(r.metrics.records_processed for r in results),
                'avg_pass_rate': round(
                    sum(r.metrics.pass_rate for r in results) / len(results), 2
                ) if results else 0
            }
        }
    
    def _generate_rule_analysis(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate rule-by-rule analysis."""
        rule_analysis = {}
        
        for result in results:
            rule_info = {
                'rule_name': result.rule_name,
                'category': result.category.value,
                'severity': result.severity.value,
                'status': 'PASSED' if result.passed else 'FAILED',
                'metrics': {
                    'total_records': result.metrics.total_records,
                    'records_processed': result.metrics.records_processed,
                    'records_passed': result.metrics.records_passed,
                    'records_failed': result.metrics.records_failed,
                    'pass_rate': round(result.metrics.pass_rate * 100, 2),
                    'execution_time_ms': round(result.metrics.execution_time_ms, 2)
                },
                'error_count': len(result.errors)
            }
            
            if result.errors and self.config.include_error_samples:
                sample_errors = result.errors[:self.config.max_error_samples]
                rule_info['error_samples'] = [
                    {
                        'error_code': error.error_code,
                        'message': error.error_message,
                        'column': error.column_name,
                        'row_index': error.row_index,
                        'invalid_value': str(error.invalid_value) if error.invalid_value is not None else None
                    }
                    for error in sample_errors
                ]
            
            rule_analysis[result.rule_id] = rule_info
        
        return rule_analysis
    
    def _generate_error_analysis(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate detailed error analysis."""
        all_errors = []
        for result in results:
            for error in result.errors:
                all_errors.append({
                    'rule_id': result.rule_id,
                    'rule_name': result.rule_name,
                    'error': error
                })
        
        # Group errors by different dimensions
        errors_by_column = {}
        errors_by_severity = {}
        errors_by_category = {}
        errors_by_code = {}
        
        for error_info in all_errors:
            error = error_info['error']
            
            # By column
            if error.column_name:
                if error.column_name not in errors_by_column:
                    errors_by_column[error.column_name] = []
                errors_by_column[error.column_name].append(error_info)
            
            # By severity
            severity = error.severity.value
            if severity not in errors_by_severity:
                errors_by_severity[severity] = []
            errors_by_severity[severity].append(error_info)
            
            # By category
            category = error.category.value
            if category not in errors_by_category:
                errors_by_category[category] = []
            errors_by_category[category].append(error_info)
            
            # By error code
            if error.error_code:
                if error.error_code not in errors_by_code:
                    errors_by_code[error.error_code] = []
                errors_by_code[error.error_code].append(error_info)
        
        # Generate column-level statistics
        column_stats = {}
        for column, column_errors in errors_by_column.items():
            column_stats[column] = {
                'total_errors': len(column_errors),
                'error_types': list(set(e['error'].error_code for e in column_errors if e['error'].error_code)),
                'affected_rules': list(set(e['rule_id'] for e in column_errors)),
                'severity_distribution': self._count_by_attribute(column_errors, lambda x: x['error'].severity.value)
            }
        
        return {
            'total_errors': len(all_errors),
            'errors_by_column': column_stats,
            'errors_by_severity': {
                k: len(v) for k, v in errors_by_severity.items()
            },
            'errors_by_category': {
                k: len(v) for k, v in errors_by_category.items()
            },
            'errors_by_code': {
                k: len(v) for k, v in errors_by_code.items()
            },
            'most_problematic_columns': self._get_top_problematic_columns(column_stats, 5),
            'most_common_error_codes': self._get_most_common_errors(errors_by_code, 5)
        }
    
    def _generate_statistics(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate statistical analysis of validation results."""
        if not results:
            return {}
        
        pass_rates = [r.metrics.pass_rate for r in results]
        execution_times = [r.metrics.execution_time_ms for r in results]
        
        statistics = {
            'pass_rate_statistics': {
                'mean': round(sum(pass_rates) / len(pass_rates), 4),
                'median': round(sorted(pass_rates)[len(pass_rates) // 2], 4),
                'min': round(min(pass_rates), 4),
                'max': round(max(pass_rates), 4),
                'std_dev': round(self._calculate_std_dev(pass_rates), 4)
            },
            'execution_time_statistics': {
                'mean_ms': round(sum(execution_times) / len(execution_times), 2),
                'median_ms': round(sorted(execution_times)[len(execution_times) // 2], 2),
                'min_ms': round(min(execution_times), 2),
                'max_ms': round(max(execution_times), 2),
                'total_ms': round(sum(execution_times), 2)
            },
            'rule_performance': {
                'fastest_rule': min(results, key=lambda r: r.metrics.execution_time_ms).rule_id,
                'slowest_rule': max(results, key=lambda r: r.metrics.execution_time_ms).rule_id,
                'best_pass_rate_rule': max(results, key=lambda r: r.metrics.pass_rate).rule_id,
                'worst_pass_rate_rule': min(results, key=lambda r: r.metrics.pass_rate).rule_id
            }
        }
        
        return statistics
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Check for high failure rates
        high_failure_rules = [r for r in results if r.metrics.pass_rate < 0.8]
        if high_failure_rules:
            recommendations.append({
                'type': 'DATA_QUALITY',
                'priority': 'HIGH',
                'title': 'Address High Failure Rate Rules',
                'description': f'{len(high_failure_rules)} rules have pass rates below 80%',
                'affected_rules': [r.rule_id for r in high_failure_rules],
                'suggested_actions': [
                    'Review rule definitions for accuracy',
                    'Investigate data sources for systematic issues',
                    'Consider adjusting rule thresholds if appropriate'
                ]
            })
        
        # Check for performance issues
        slow_rules = [r for r in results if r.metrics.execution_time_ms > 1000]  # > 1 second
        if slow_rules:
            recommendations.append({
                'type': 'PERFORMANCE',
                'priority': 'MEDIUM',
                'title': 'Optimize Slow Validation Rules',
                'description': f'{len(slow_rules)} rules took longer than 1 second to execute',
                'affected_rules': [r.rule_id for r in slow_rules],
                'suggested_actions': [
                    'Review rule logic for optimization opportunities',
                    'Consider sampling for large datasets',
                    'Implement parallel processing where applicable'
                ]
            })
        
        # Check for missing critical validations
        categories_present = set(r.category for r in results)
        critical_categories = {ValidationCategory.DATA_TYPE, ValidationCategory.COMPLETENESS}
        missing_categories = critical_categories - categories_present
        if missing_categories:
            recommendations.append({
                'type': 'COVERAGE',
                'priority': 'HIGH',
                'title': 'Add Missing Critical Validations',
                'description': f'Missing validation categories: {[c.value for c in missing_categories]}',
                'suggested_actions': [
                    'Implement data type validation rules',
                    'Add completeness checks for required fields',
                    'Review validation coverage completeness'
                ]
            })
        
        # Check for error patterns
        error_patterns = self._analyze_error_patterns(results)
        if error_patterns:
            recommendations.extend(error_patterns)
        
        return recommendations
    
    def _generate_detailed_results(self, results: List[ValidationResult]) -> List[Dict[str, Any]]:
        """Generate detailed results for comprehensive reports."""
        detailed_results = []
        
        for result in results:
            detailed_result = {
                'rule_id': result.rule_id,
                'rule_name': result.rule_name,
                'category': result.category.value,
                'severity': result.severity.value,
                'passed': result.passed,
                'executed_at': result.executed_at.isoformat() if result.executed_at else None,
                'metrics': {
                    'total_records': result.metrics.total_records,
                    'records_processed': result.metrics.records_processed,
                    'records_passed': result.metrics.records_passed,
                    'records_failed': result.metrics.records_failed,
                    'pass_rate': result.metrics.pass_rate,
                    'execution_time_ms': result.metrics.execution_time_ms,
                    'memory_usage_mb': result.metrics.memory_usage_mb,
                    'cpu_usage_percent': result.metrics.cpu_usage_percent
                },
                'errors': [
                    {
                        'error_id': str(error.error_id),
                        'row_index': error.row_index,
                        'column_name': error.column_name,
                        'invalid_value': error.invalid_value,
                        'expected_value': error.expected_value,
                        'error_message': error.error_message,
                        'error_code': error.error_code,
                        'severity': error.severity.value,
                        'category': error.category.value,
                        'context': error.context
                    }
                    for error in result.errors
                ],
                'statistics': result.statistics,
                'context': {
                    'validation_id': str(result.validation_id),
                    'dataset_name': result.context.dataset_name if result.context else None,
                    'execution_id': str(result.context.execution_id) if result.context else None,
                    'metadata': result.context.metadata if result.context else {}
                }
            }
            
            detailed_results.append(detailed_result)
        
        return detailed_results
    
    def _count_by_attribute(self, items: List[Dict], attr_func) -> Dict[str, int]:
        """Count items by attribute value."""
        counts = {}
        for item in items:
            attr_value = attr_func(item)
            counts[attr_value] = counts.get(attr_value, 0) + 1
        return counts
    
    def _get_top_problematic_columns(self, column_stats: Dict, limit: int) -> List[Dict[str, Any]]:
        """Get columns with most errors."""
        sorted_columns = sorted(
            column_stats.items(),
            key=lambda x: x[1]['total_errors'],
            reverse=True
        )
        
        return [
            {
                'column_name': col_name,
                'error_count': stats['total_errors'],
                'error_types': stats['error_types']
            }
            for col_name, stats in sorted_columns[:limit]
        ]
    
    def _get_most_common_errors(self, errors_by_code: Dict, limit: int) -> List[Dict[str, Any]]:
        """Get most common error codes."""
        sorted_errors = sorted(
            errors_by_code.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        return [
            {
                'error_code': error_code,
                'count': len(errors),
                'sample_message': errors[0]['error'].error_message if errors else None
            }
            for error_code, errors in sorted_errors[:limit]
        ]
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _analyze_error_patterns(self, results: List[ValidationResult]) -> List[Dict[str, Any]]:
        """Analyze error patterns and generate recommendations."""
        recommendations = []
        
        # Analyze error distribution by column
        column_error_counts = {}
        for result in results:
            for error in result.errors:
                if error.column_name:
                    column_error_counts[error.column_name] = column_error_counts.get(error.column_name, 0) + 1
        
        # Find columns with disproportionate errors
        if column_error_counts:
            total_errors = sum(column_error_counts.values())
            avg_errors_per_column = total_errors / len(column_error_counts)
            
            problematic_columns = [
                col for col, count in column_error_counts.items()
                if count > avg_errors_per_column * 2  # More than 2x average
            ]
            
            if problematic_columns:
                recommendations.append({
                    'type': 'DATA_PATTERN',
                    'priority': 'MEDIUM',
                    'title': 'Investigate Problematic Columns',
                    'description': f'Columns with unusually high error rates: {problematic_columns}',
                    'affected_columns': problematic_columns,
                    'suggested_actions': [
                        'Review data collection processes for these columns',
                        'Check for systematic data entry issues',
                        'Consider additional data validation at source'
                    ]
                })
        
        return recommendations


class ValidationReportExporter:
    """Export validation reports to various formats."""
    
    def __init__(self):
        self.exporters = {
            ReportFormat.JSON: self._export_json,
            ReportFormat.HTML: self._export_html,
            ReportFormat.EXCEL: self._export_excel,
            ReportFormat.CSV: self._export_csv,
            ReportFormat.MARKDOWN: self._export_markdown
        }
    
    def export_report(
        self,
        report: Dict[str, Any],
        format_type: ReportFormat,
        output_path: str
    ) -> str:
        """Export report to specified format."""
        if format_type not in self.exporters:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        return self.exporters[format_type](report, output_path)
    
    def _export_json(self, report: Dict[str, Any], output_path: str) -> str:
        """Export report as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        return output_path
    
    def _export_html(self, report: Dict[str, Any], output_path: str) -> str:
        """Export report as HTML."""
        html_content = self._generate_html_report(report)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return output_path
    
    def _export_excel(self, report: Dict[str, Any], output_path: str) -> str:
        """Export report as Excel."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([report['summary']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Rule analysis sheet
            if 'rule_analysis' in report:
                rule_data = []
                for rule_id, rule_info in report['rule_analysis'].items():
                    rule_data.append({
                        'Rule ID': rule_id,
                        'Rule Name': rule_info.get('rule_name', ''),
                        'Category': rule_info.get('category', ''),
                        'Status': rule_info.get('status', ''),
                        'Pass Rate': rule_info.get('metrics', {}).get('pass_rate', 0),
                        'Error Count': rule_info.get('error_count', 0),
                        'Execution Time (ms)': rule_info.get('metrics', {}).get('execution_time_ms', 0)
                    })
                
                rule_df = pd.DataFrame(rule_data)
                rule_df.to_excel(writer, sheet_name='Rule Analysis', index=False)
            
            # Error analysis sheet
            if 'error_analysis' in report:
                error_data = []
                for column, stats in report['error_analysis'].get('errors_by_column', {}).items():
                    error_data.append({
                        'Column': column,
                        'Total Errors': stats['total_errors'],
                        'Error Types': ', '.join(stats['error_types']),
                        'Affected Rules': ', '.join(stats['affected_rules'])
                    })
                
                if error_data:
                    error_df = pd.DataFrame(error_data)
                    error_df.to_excel(writer, sheet_name='Error Analysis', index=False)
        
        return output_path
    
    def _export_csv(self, report: Dict[str, Any], output_path: str) -> str:
        """Export report as CSV (summary data)."""
        # Create a flattened view of the report for CSV
        csv_data = []
        
        if 'rule_analysis' in report:
            for rule_id, rule_info in report['rule_analysis'].items():
                csv_data.append({
                    'Rule ID': rule_id,
                    'Rule Name': rule_info.get('rule_name', ''),
                    'Category': rule_info.get('category', ''),
                    'Status': rule_info.get('status', ''),
                    'Pass Rate': rule_info.get('metrics', {}).get('pass_rate', 0),
                    'Error Count': rule_info.get('error_count', 0),
                    'Execution Time (ms)': rule_info.get('metrics', {}).get('execution_time_ms', 0)
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        return output_path
    
    def _export_markdown(self, report: Dict[str, Any], output_path: str) -> str:
        """Export report as Markdown."""
        markdown_content = self._generate_markdown_report(report)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        return output_path
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        # Simple HTML template - in production, use proper templating
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Data Quality Validation Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Overall Status:</strong> <span class="{report['summary']['overall_status'].lower()}">{report['summary']['overall_status']}</span></p>
        <p><strong>Quality Score:</strong> {report['summary']['quality_score']}%</p>
        <p><strong>Rules Passed:</strong> {report['summary']['rules']['passed']} / {report['summary']['rules']['total']}</p>
        <p><strong>Total Errors:</strong> {report['summary']['errors']['total']}</p>
    </div>
    
    <h2>Rule Analysis</h2>
    <table>
        <tr>
            <th>Rule Name</th>
            <th>Category</th>
            <th>Status</th>
            <th>Pass Rate</th>
            <th>Error Count</th>
        </tr>
        """
        
        for rule_id, rule_info in report.get('rule_analysis', {}).items():
            status_class = 'passed' if rule_info['status'] == 'PASSED' else 'failed'
            html += f"""
        <tr>
            <td>{rule_info.get('rule_name', rule_id)}</td>
            <td>{rule_info.get('category', '')}</td>
            <td class="{status_class}">{rule_info.get('status', '')}</td>
            <td>{rule_info.get('metrics', {}).get('pass_rate', 0)}%</td>
            <td>{rule_info.get('error_count', 0)}</td>
        </tr>
            """
        
        html += """
    </table>
</body>
</html>
        """
        
        return html
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate Markdown report content."""
        markdown = f"""# Data Quality Validation Report

## Summary

- **Overall Status:** {report['summary']['overall_status']}
- **Quality Score:** {report['summary']['quality_score']}%
- **Rules Passed:** {report['summary']['rules']['passed']} / {report['summary']['rules']['total']}
- **Total Errors:** {report['summary']['errors']['total']}

## Rule Analysis

| Rule Name | Category | Status | Pass Rate | Error Count |
|-----------|----------|--------|-----------|-------------|
"""
        
        for rule_id, rule_info in report.get('rule_analysis', {}).items():
            markdown += f"| {rule_info.get('rule_name', rule_id)} | {rule_info.get('category', '')} | {rule_info.get('status', '')} | {rule_info.get('metrics', {}).get('pass_rate', 0)}% | {rule_info.get('error_count', 0)} |\n"
        
        if 'recommendations' in report and report['recommendations']:
            markdown += "\n## Recommendations\n\n"
            for i, rec in enumerate(report['recommendations'], 1):
                markdown += f"### {i}. {rec['title']} (Priority: {rec['priority']})\n\n"
                markdown += f"{rec['description']}\n\n"
                if 'suggested_actions' in rec:
                    markdown += "**Suggested Actions:**\n"
                    for action in rec['suggested_actions']:
                        markdown += f"- {action}\n"
                    markdown += "\n"
        
        return markdown


class ValidationErrorHandler:
    """Handle and categorize validation errors for reporting and analysis."""
    
    def __init__(self):
        self.error_handlers = {
            'INVALID_TYPE': self._handle_type_error,
            'INVALID_FORMAT': self._handle_format_error,
            'NULL_VALUE': self._handle_null_error,
            'DUPLICATE_VALUE': self._handle_duplicate_error,
            'BELOW_MINIMUM': self._handle_range_error,
            'ABOVE_MAXIMUM': self._handle_range_error,
            'BUSINESS_RULE_VIOLATION': self._handle_business_rule_error
        }
    
    def process_errors(self, errors: List[ValidationError]) -> Dict[str, Any]:
        """Process and categorize validation errors."""
        processed_errors = {
            'total_count': len(errors),
            'by_category': {},
            'by_severity': {},
            'by_error_code': {},
            'actionable_insights': [],
            'error_patterns': []
        }
        
        # Categorize errors
        for error in errors:
            # By category
            category = error.category.value
            if category not in processed_errors['by_category']:
                processed_errors['by_category'][category] = []
            processed_errors['by_category'][category].append(error)
            
            # By severity
            severity = error.severity.value
            if severity not in processed_errors['by_severity']:
                processed_errors['by_severity'][severity] = []
            processed_errors['by_severity'][severity].append(error)
            
            # By error code
            if error.error_code:
                if error.error_code not in processed_errors['by_error_code']:
                    processed_errors['by_error_code'][error.error_code] = []
                processed_errors['by_error_code'][error.error_code].append(error)
        
        # Generate actionable insights
        processed_errors['actionable_insights'] = self._generate_actionable_insights(errors)
        
        # Detect error patterns
        processed_errors['error_patterns'] = self._detect_error_patterns(errors)
        
        return processed_errors
    
    def _handle_type_error(self, error: ValidationError) -> Dict[str, Any]:
        """Handle data type validation errors."""
        return {
            'resolution_priority': 'HIGH',
            'suggested_action': 'Review data ingestion process and implement type conversion',
            'impact': 'Data processing failures, incorrect analytics results'
        }
    
    def _handle_format_error(self, error: ValidationError) -> Dict[str, Any]:
        """Handle format validation errors."""
        return {
            'resolution_priority': 'MEDIUM',
            'suggested_action': 'Implement data cleansing rules or update format requirements',
            'impact': 'Data integration issues, reporting inconsistencies'
        }
    
    def _handle_null_error(self, error: ValidationError) -> Dict[str, Any]:
        """Handle null value errors."""
        return {
            'resolution_priority': 'HIGH',
            'suggested_action': 'Implement required field validation at data entry',
            'impact': 'Incomplete analytics, business process disruption'
        }
    
    def _handle_duplicate_error(self, error: ValidationError) -> Dict[str, Any]:
        """Handle duplicate value errors."""
        return {
            'resolution_priority': 'MEDIUM',
            'suggested_action': 'Implement unique constraints and deduplication processes',
            'impact': 'Data redundancy, skewed analytics results'
        }
    
    def _handle_range_error(self, error: ValidationError) -> Dict[str, Any]:
        """Handle range validation errors."""
        return {
            'resolution_priority': 'MEDIUM',
            'suggested_action': 'Review business rules and implement input validation',
            'impact': 'Data quality issues, potential business rule violations'
        }
    
    def _handle_business_rule_error(self, error: ValidationError) -> Dict[str, Any]:
        """Handle business rule violations."""
        return {
            'resolution_priority': 'HIGH',
            'suggested_action': 'Review business processes and update validation rules',
            'impact': 'Business process violations, compliance issues'
        }
    
    def _generate_actionable_insights(self, errors: List[ValidationError]) -> List[Dict[str, Any]]:
        """Generate actionable insights from error patterns."""
        insights = []
        
        # Group errors by type and analyze
        error_groups = {}
        for error in errors:
            error_code = error.error_code or 'UNKNOWN'
            if error_code not in error_groups:
                error_groups[error_code] = []
            error_groups[error_code].append(error)
        
        for error_code, error_list in error_groups.items():
            if len(error_list) > 1:  # Multiple occurrences
                handler = self.error_handlers.get(error_code, self._handle_generic_error)
                error_info = handler(error_list[0])
                
                insights.append({
                    'error_type': error_code,
                    'occurrence_count': len(error_list),
                    'affected_columns': list(set(e.column_name for e in error_list if e.column_name)),
                    'resolution_priority': error_info['resolution_priority'],
                    'suggested_action': error_info['suggested_action'],
                    'business_impact': error_info['impact']
                })
        
        return insights
    
    def _detect_error_patterns(self, errors: List[ValidationError]) -> List[Dict[str, Any]]:
        """Detect patterns in validation errors."""
        patterns = []
        
        # Pattern 1: High error concentration in specific columns
        column_error_counts = {}
        for error in errors:
            if error.column_name:
                column_error_counts[error.column_name] = column_error_counts.get(error.column_name, 0) + 1
        
        if column_error_counts:
            max_errors = max(column_error_counts.values())
            problematic_columns = [col for col, count in column_error_counts.items() if count > max_errors * 0.5]
            
            if len(problematic_columns) < len(column_error_counts) * 0.2:  # Less than 20% of columns
                patterns.append({
                    'pattern_type': 'COLUMN_CONCENTRATION',
                    'description': f'High error concentration in {len(problematic_columns)} columns',
                    'affected_columns': problematic_columns,
                    'recommendation': 'Focus data quality efforts on these specific columns'
                })
        
        # Pattern 2: Systematic error types
        error_type_counts = {}
        for error in errors:
            if error.error_code:
                error_type_counts[error.error_code] = error_type_counts.get(error.error_code, 0) + 1
        
        if error_type_counts:
            dominant_error_types = [
                error_type for error_type, count in error_type_counts.items()
                if count > len(errors) * 0.3  # More than 30% of errors
            ]
            
            if dominant_error_types:
                patterns.append({
                    'pattern_type': 'SYSTEMATIC_ERRORS',
                    'description': f'Systematic errors of types: {dominant_error_types}',
                    'error_types': dominant_error_types,
                    'recommendation': 'Investigate upstream data sources for systematic issues'
                })
        
        return patterns
    
    def _handle_generic_error(self, error: ValidationError) -> Dict[str, Any]:
        """Handle generic errors."""
        return {
            'resolution_priority': 'MEDIUM',
            'suggested_action': 'Review error details and implement appropriate data quality measures',
            'impact': 'General data quality degradation'
        }