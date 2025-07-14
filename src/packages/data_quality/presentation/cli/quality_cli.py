"""Data Quality CLI interface for validation, cleansing, and monitoring operations."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4

import click
import pandas as pd
import structlog
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.panel import Panel
from rich.tree import Tree

from ...application.services.validation_engine import ValidationEngine
from ...application.services.data_cleansing_engine import DataCleansingEngine
from ...application.services.quality_monitoring_service import (
    QualityMonitoringService, MonitoringConfiguration, AlertSeverity
)
from ...domain.entities.quality_rule import (
    QualityRule, RuleType, LogicType, ValidationLogic, QualityThreshold,
    Severity, UserId, DatasetId, RuleId
)

logger = structlog.get_logger(__name__)
console = Console()

# Global instances
validation_engine = ValidationEngine()
cleansing_engine = DataCleansingEngine()
monitoring_service = QualityMonitoringService(MonitoringConfiguration())

@click.group()
@click.version_option(version="1.0.0", prog_name="Data Quality CLI")
def cli():
    """
    Data Quality CLI - Comprehensive data quality management toolkit.
    
    Provides validation, cleansing, and monitoring capabilities for ensuring
    high-quality data across your organization.
    """
    pass

@cli.group()
def rules():
    """Manage data quality validation rules."""
    pass

@rules.command()
@click.option('--name', required=True, help='Name of the quality rule')
@click.option('--type', 'rule_type', required=True, 
              type=click.Choice(['completeness', 'uniqueness', 'validity', 'consistency', 'accuracy', 'timeliness', 'custom']),
              help='Type of validation rule')
@click.option('--columns', help='Target columns (comma-separated)')
@click.option('--logic-type', required=True,
              type=click.Choice(['regex', 'range', 'list', 'sql', 'python']),
              help='Type of validation logic')
@click.option('--expression', required=True, help='Validation expression')
@click.option('--pass-threshold', type=float, default=0.95, help='Pass rate threshold (default: 0.95)')
@click.option('--warning-threshold', type=float, default=0.90, help='Warning threshold (default: 0.90)')
@click.option('--critical-threshold', type=float, default=0.80, help='Critical threshold (default: 0.80)')
@click.option('--severity', type=click.Choice(['low', 'medium', 'high', 'critical']), default='medium', help='Rule severity')
@click.option('--description', help='Rule description')
@click.option('--output', type=click.Choice(['table', 'json']), default='table', help='Output format')
def create(name, rule_type, columns, logic_type, expression, pass_threshold, warning_threshold, critical_threshold, severity, description, output):
    """Create a new data quality rule."""
    try:
        # Parse columns
        target_columns = [col.strip() for col in columns.split(',')] if columns else []
        
        # Create rule
        rule = QualityRule(
            rule_id=RuleId(value=uuid4()),
            rule_name=name,
            rule_type=RuleType(rule_type),
            target_columns=target_columns,
            validation_logic=ValidationLogic(
                logic_type=LogicType(logic_type),
                expression=expression,
                parameters={},
                error_message_template=f"Validation failed for rule '{name}': {{column}}={{value}}"
            ),
            thresholds=QualityThreshold(
                pass_rate_threshold=pass_threshold,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold
            ),
            severity=Severity(severity),
            description=description,
            created_by=UserId(value=uuid4()),
            is_enabled=True
        )
        
        if output == 'json':
            result = {
                'rule_id': str(rule.rule_id.value),
                'name': rule.rule_name,
                'type': rule.rule_type.value,
                'status': 'created'
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"✅ Quality rule '{name}' created successfully")
            console.print(f"   Rule ID: {rule.rule_id.value}")
            console.print(f"   Type: {rule.rule_type.value}")
            console.print(f"   Columns: {', '.join(target_columns) if target_columns else 'All columns'}")
        
    except Exception as e:
        console.print(f"❌ Failed to create rule: {str(e)}", style="red")
        sys.exit(1)

@cli.group()
def validate():
    """Validate datasets against quality rules."""
    pass

@validate.command()
@click.option('--file', 'file_path', required=True, type=click.Path(exists=True), help='Path to dataset file (CSV)')
@click.option('--rules', help='Rule IDs to apply (comma-separated)')
@click.option('--rule-type', type=click.Choice(['completeness', 'uniqueness', 'validity']), help='Apply rules of specific type')
@click.option('--output', type=click.Choice(['table', 'json', 'report']), default='table', help='Output format')
@click.option('--save-report', type=click.Path(), help='Save detailed report to file')
def dataset(file_path, rules, rule_type, output, save_report):
    """Validate a dataset file against quality rules."""
    try:
        # Load dataset
        console.print(f"📊 Loading dataset from {file_path}...")
        df = pd.read_csv(file_path)
        
        if df.empty:
            console.print("❌ Dataset is empty", style="red")
            sys.exit(1)
        
        console.print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Create mock rules based on type or rule IDs
        validation_rules = []
        if rules:
            rule_ids = [r.strip() for r in rules.split(',')]
            for rule_id in rule_ids:
                # Create a basic completeness rule as example
                rule = QualityRule(
                    rule_id=RuleId(value=uuid4()),
                    rule_name=f"Rule {rule_id}",
                    rule_type=RuleType.COMPLETENESS,
                    target_columns=list(df.columns),
                    validation_logic=ValidationLogic(
                        logic_type=LogicType.PYTHON,
                        expression="df.notna().all(axis=1)",
                        parameters={},
                        error_message_template="Missing values found"
                    ),
                    thresholds=QualityThreshold(
                        pass_rate_threshold=0.95,
                        warning_threshold=0.90,
                        critical_threshold=0.80
                    ),
                    severity=Severity.MEDIUM,
                    created_by=UserId(value=uuid4()),
                    is_enabled=True
                )
                validation_rules.append(rule)
        else:
            # Create default completeness rule
            rule = QualityRule(
                rule_id=RuleId(value=uuid4()),
                rule_name="Default Completeness Check",
                rule_type=RuleType(rule_type) if rule_type else RuleType.COMPLETENESS,
                target_columns=list(df.columns),
                validation_logic=ValidationLogic(
                    logic_type=LogicType.PYTHON,
                    expression="df.notna().all(axis=1)",
                    parameters={},
                    error_message_template="Missing values found"
                ),
                thresholds=QualityThreshold(
                    pass_rate_threshold=0.95,
                    warning_threshold=0.90,
                    critical_threshold=0.80
                ),
                severity=Severity.MEDIUM,
                created_by=UserId(value=uuid4()),
                is_enabled=True
            )
            validation_rules.append(rule)
        
        # Execute validation
        with Progress() as progress:
            task = progress.add_task("Validating dataset...", total=len(validation_rules))
            
            results = validation_engine.validate_dataset(
                rules=validation_rules,
                df=df,
                dataset_id=DatasetId(value=uuid4())
            )
            
            progress.update(task, completed=len(validation_rules))
        
        # Display results
        if output == 'json':
            json_results = []
            for result in results:
                json_results.append({
                    'rule_id': str(result.rule_id.value),
                    'status': result.status.value,
                    'pass_rate': result.pass_rate,
                    'records_passed': result.records_passed,
                    'records_failed': result.records_failed,
                    'errors': len(result.validation_errors)
                })
            console.print(json.dumps(json_results, indent=2))
            
        elif output == 'report':
            _display_validation_report(results, df, file_path)
            
        else:
            _display_validation_table(results)
        
        # Save report if requested
        if save_report:
            _save_validation_report(results, df, file_path, save_report)
            console.print(f"📝 Report saved to {save_report}")
        
        # Exit with appropriate code
        failed_rules = sum(1 for r in results if r.pass_rate < r.thresholds.pass_rate_threshold)
        if failed_rules > 0:
            console.print(f"\n⚠️  {failed_rules} rule(s) failed", style="yellow")
            sys.exit(1)
        else:
            console.print("\n✅ All rules passed", style="green")
        
    except Exception as e:
        console.print(f"❌ Validation failed: {str(e)}", style="red")
        sys.exit(1)

@cli.group()
def cleanse():
    """Cleanse datasets to improve data quality."""
    pass

@cleanse.command()
@click.option('--file', 'file_path', required=True, type=click.Path(exists=True), help='Path to dataset file (CSV)')
@click.option('--actions', required=True, help='Cleansing actions (comma-separated)')
@click.option('--output-file', type=click.Path(), help='Save cleaned dataset to file')
@click.option('--config-file', type=click.Path(exists=True), help='JSON configuration file')
@click.option('--report', type=click.Choice(['table', 'json']), default='table', help='Report format')
def dataset(file_path, actions, output_file, config_file, report):
    """Cleanse a dataset file."""
    try:
        # Load dataset
        console.print(f"📊 Loading dataset from {file_path}...")
        df = pd.read_csv(file_path)
        
        if df.empty:
            console.print("❌ Dataset is empty", style="red")
            sys.exit(1)
        
        console.print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Prepare cleansing configuration
        if config_file:
            with open(config_file, 'r') as f:
                cleansing_config = json.load(f)
        else:
            action_list = [a.strip() for a in actions.split(',')]
            cleansing_config = {
                'actions': action_list,
                'columns': {col: {'strategy': 'standardize'} for col in df.columns}
            }
        
        # Execute cleansing
        with Progress() as progress:
            task = progress.add_task("Cleansing dataset...", total=100)
            
            cleaned_df, cleansing_report = cleansing_engine.clean_dataset(
                df=df,
                cleansing_config=cleansing_config,
                dataset_name=Path(file_path).stem
            )
            
            progress.update(task, completed=100)
        
        # Display results
        if report == 'json':
            report_data = {
                'dataset_name': cleansing_report.dataset_name,
                'total_records': cleansing_report.total_records,
                'overall_success': cleansing_report.overall_success,
                'execution_time': cleansing_report.execution_time_seconds,
                'results': [
                    {
                        'action': result.action.value,
                        'column': result.column_name,
                        'records_affected': result.records_affected,
                        'success': result.success
                    }
                    for result in cleansing_report.cleansing_results
                ]
            }
            console.print(json.dumps(report_data, indent=2))
        else:
            _display_cleansing_report(cleansing_report)
        
        # Save cleaned dataset
        if output_file:
            cleaned_df.to_csv(output_file, index=False)
            console.print(f"💾 Cleaned dataset saved to {output_file}")
        
        console.print("✅ Cleansing completed successfully", style="green")
        
    except Exception as e:
        console.print(f"❌ Cleansing failed: {str(e)}", style="red")
        sys.exit(1)

@cli.group()
def monitor():
    """Monitor data quality in real-time."""
    pass

@monitor.command()
@click.option('--dataset-id', required=True, help='Dataset ID to monitor')
@click.option('--file', 'file_path', type=click.Path(exists=True), help='Dataset file to monitor')
@click.option('--rules', help='Rule IDs to apply (comma-separated)')
@click.option('--interval', type=int, default=300, help='Monitoring interval in seconds (default: 300)')
def start(dataset_id, file_path, rules, interval):
    """Start monitoring a dataset."""
    try:
        # Create monitoring rules
        monitoring_rules = []
        if rules:
            rule_ids = [r.strip() for r in rules.split(',')]
            for rule_id in rule_ids:
                rule = QualityRule(
                    rule_id=RuleId(value=uuid4()),
                    rule_name=f"Monitoring Rule {rule_id}",
                    rule_type=RuleType.COMPLETENESS,
                    target_columns=[],
                    validation_logic=ValidationLogic(
                        logic_type=LogicType.PYTHON,
                        expression="df.notna().all(axis=1)",
                        parameters={},
                        error_message_template="Quality issue detected"
                    ),
                    thresholds=QualityThreshold(
                        pass_rate_threshold=0.95,
                        warning_threshold=0.90,
                        critical_threshold=0.80
                    ),
                    severity=Severity.MEDIUM,
                    created_by=UserId(value=uuid4()),
                    is_enabled=True
                )
                monitoring_rules.append(rule)
        else:
            # Create default rule
            rule = QualityRule(
                rule_id=RuleId(value=uuid4()),
                rule_name="Default Monitoring Rule",
                rule_type=RuleType.COMPLETENESS,
                target_columns=[],
                validation_logic=ValidationLogic(
                    logic_type=LogicType.PYTHON,
                    expression="df.notna().all(axis=1)",
                    parameters={},
                    error_message_template="Quality issue detected"
                ),
                thresholds=QualityThreshold(
                    pass_rate_threshold=0.95,
                    warning_threshold=0.90,
                    critical_threshold=0.80
                ),
                severity=Severity.MEDIUM,
                created_by=UserId(value=uuid4()),
                is_enabled=True
            )
            monitoring_rules.append(rule)
        
        # Configure data source
        data_source_config = {}
        if file_path:
            data_source_config = {
                'type': 'file',
                'path': file_path
            }
        
        # Add dataset to monitoring
        monitoring_service.add_dataset_monitoring(
            dataset_id=UUID(dataset_id),
            rules=monitoring_rules,
            data_source_config=data_source_config
        )
        
        # Start monitoring service
        monitoring_service.start_monitoring()
        
        console.print(f"🔍 Started monitoring dataset {dataset_id}")
        console.print(f"   Rules: {len(monitoring_rules)}")
        console.print(f"   Interval: {interval} seconds")
        console.print("   Press Ctrl+C to stop monitoring")
        
        # Keep monitoring running
        try:
            while True:
                click.pause()
        except KeyboardInterrupt:
            monitoring_service.stop_monitoring()
            console.print("\n🛑 Monitoring stopped")
        
    except Exception as e:
        console.print(f"❌ Failed to start monitoring: {str(e)}", style="red")
        sys.exit(1)

@monitor.command()
def dashboard():
    """Display monitoring dashboard."""
    try:
        dashboard_data = monitoring_service.get_quality_dashboard_data()
        
        # Create dashboard display
        console.print(Panel.fit("📊 Data Quality Monitoring Dashboard", style="bold blue"))
        
        # Status section
        status_table = Table(title="System Status")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")
        
        status_table.add_row("Monitoring Status", dashboard_data['monitoring_status'])
        status_table.add_row("Monitored Datasets", str(dashboard_data['monitored_datasets']))
        status_table.add_row("Active Rules", str(dashboard_data['total_active_rules']))
        
        console.print(status_table)
        
        # Quality summary
        quality_summary = dashboard_data['quality_summary']
        quality_table = Table(title="Quality Summary")
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Value", style="yellow")
        
        quality_table.add_row("Average Quality", f"{quality_summary['average_quality']:.2%}")
        quality_table.add_row("Minimum Quality", f"{quality_summary['minimum_quality']:.2%}")
        quality_table.add_row("Maximum Quality", f"{quality_summary['maximum_quality']:.2%}")
        quality_table.add_row("Total Metrics", str(quality_summary['total_metrics']))
        
        console.print(quality_table)
        
        # Alert summary
        alert_summary = dashboard_data['alert_summary']
        alert_table = Table(title="Alert Summary")
        alert_table.add_column("Severity", style="cyan")
        alert_table.add_column("Count", style="red")
        
        for severity, count in alert_summary['by_severity'].items():
            alert_table.add_row(severity.title(), str(count))
        
        console.print(alert_table)
        
    except Exception as e:
        console.print(f"❌ Failed to display dashboard: {str(e)}", style="red")
        sys.exit(1)

@monitor.command()
@click.option('--severity', type=click.Choice(['info', 'warning', 'critical', 'emergency']), help='Filter by severity')
@click.option('--unacknowledged', is_flag=True, help='Show only unacknowledged alerts')
def alerts(severity, unacknowledged):
    """Display quality alerts."""
    try:
        severity_filter = AlertSeverity(severity) if severity else None
        alerts = monitoring_service.alert_manager.get_active_alerts(severity_filter)
        
        if unacknowledged:
            alerts = [alert for alert in alerts if not alert.acknowledged]
        
        if not alerts:
            console.print("✅ No alerts found", style="green")
            return
        
        # Display alerts table
        alert_table = Table(title=f"Quality Alerts ({len(alerts)} total)")
        alert_table.add_column("ID", style="cyan")
        alert_table.add_column("Severity", style="red")
        alert_table.add_column("Type", style="yellow")
        alert_table.add_column("Message", style="white")
        alert_table.add_column("Triggered", style="blue")
        alert_table.add_column("Ack", style="green")
        
        for alert in alerts[:20]:  # Limit to 20 alerts
            ack_status = "✅" if alert.acknowledged else "❌"
            alert_table.add_row(
                str(alert.alert_id)[:8],
                alert.severity.value.upper(),
                alert.alert_type,
                alert.message[:50] + "..." if len(alert.message) > 50 else alert.message,
                alert.triggered_at.strftime("%H:%M:%S"),
                ack_status
            )
        
        console.print(alert_table)
        
        if len(alerts) > 20:
            console.print(f"... and {len(alerts) - 20} more alerts")
        
    except Exception as e:
        console.print(f"❌ Failed to display alerts: {str(e)}", style="red")
        sys.exit(1)

# Helper functions
def _display_validation_table(results):
    """Display validation results in table format."""
    table = Table(title="Validation Results")
    table.add_column("Rule", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Pass Rate", style="yellow")
    table.add_column("Passed", style="blue")
    table.add_column("Failed", style="red")
    table.add_column("Errors", style="magenta")
    
    for result in results:
        status_style = "green" if result.status.value == "passed" else "red"
        table.add_row(
            str(result.rule_id.value)[:8],
            f"[{status_style}]{result.status.value.upper()}[/{status_style}]",
            f"{result.pass_rate:.2%}",
            str(result.records_passed),
            str(result.records_failed),
            str(len(result.validation_errors))
        )
    
    console.print(table)

def _display_validation_report(results, df, file_path):
    """Display detailed validation report."""
    console.print(Panel.fit(f"📋 Validation Report - {Path(file_path).name}", style="bold blue"))
    
    # Dataset info
    console.print(f"📊 Dataset: {len(df)} rows, {len(df.columns)} columns")
    console.print(f"📁 File: {file_path}")
    console.print(f"🔍 Rules executed: {len(results)}")
    
    # Summary
    passed = sum(1 for r in results if r.status.value == "passed")
    failed = len(results) - passed
    
    summary = Tree("📈 Summary")
    summary.add(f"✅ Passed: {passed}")
    summary.add(f"❌ Failed: {failed}")
    summary.add(f"📊 Overall pass rate: {passed/len(results):.1%}")
    
    console.print(summary)
    
    # Detailed results
    _display_validation_table(results)

def _display_cleansing_report(report):
    """Display cleansing report."""
    console.print(Panel.fit(f"🧹 Cleansing Report - {report.dataset_name}", style="bold green"))
    
    # Summary
    console.print(f"📊 Records processed: {report.total_records}")
    console.print(f"📋 Columns processed: {report.total_columns}")
    console.print(f"⏱️  Execution time: {report.execution_time_seconds:.2f}s")
    console.print(f"✅ Overall success: {report.overall_success}")
    
    # Results table
    results_table = Table(title="Cleansing Results")
    results_table.add_column("Action", style="cyan")
    results_table.add_column("Column", style="yellow")
    results_table.add_column("Records Affected", style="blue")
    results_table.add_column("Strategy", style="green")
    results_table.add_column("Success", style="red")
    
    for result in report.cleansing_results:
        success_icon = "✅" if result.success else "❌"
        results_table.add_row(
            result.action.value,
            result.column_name,
            str(result.records_affected),
            result.strategy_used.value,
            success_icon
        )
    
    console.print(results_table)

def _save_validation_report(results, df, file_path, output_path):
    """Save validation report to file."""
    report_data = {
        'file_path': str(file_path),
        'dataset_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns)
        },
        'validation_summary': {
            'total_rules': len(results),
            'passed_rules': sum(1 for r in results if r.status.value == "passed"),
            'failed_rules': sum(1 for r in results if r.status.value == "failed"),
            'overall_pass_rate': sum(r.pass_rate for r in results) / len(results) if results else 0
        },
        'results': [
            {
                'rule_id': str(result.rule_id.value),
                'status': result.status.value,
                'pass_rate': result.pass_rate,
                'records_passed': result.records_passed,
                'records_failed': result.records_failed,
                'error_count': len(result.validation_errors),
                'execution_time': result.execution_time_seconds
            }
            for result in results
        ],
        'generated_at': datetime.utcnow().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)

if __name__ == '__main__':
    cli()