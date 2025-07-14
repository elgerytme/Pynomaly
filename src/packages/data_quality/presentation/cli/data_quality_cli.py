#!/usr/bin/env python3
"""Data Quality CLI using Typer."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import pandas as pd
import structlog
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.tree import Tree

# Import domain entities and use cases
from ...application.use_cases.execute_quality_validation import ExecuteQualityValidationUseCase
from ...application.use_cases.manage_quality_rules import ManageQualityRulesUseCase
from ...domain.entities.quality_rule import (
    QualityRule, RuleId, DatasetId, UserId, RuleType, Severity,
    QualityCategory, ValidationLogic, RuleMetadata, LogicType,
    ValidationResult, QualityThreshold
)
from ...infrastructure.adapters.in_memory_quality_rule_repository import InMemoryQualityRuleRepository

logger = structlog.get_logger(__name__)
console = Console()

# Create Typer app
app = typer.Typer(
    name="data-quality",
    help="🔍 Data Quality Validation and Rule Management Tools",
    add_completion=True,
    rich_markup_mode="rich"
)

# Dependency setup
def get_repository():
    """Get repository instance."""
    return InMemoryQualityRuleRepository()

def get_validation_use_case(repository=None):
    """Get validation use case instance."""
    if repository is None:
        repository = get_repository()
    return ExecuteQualityValidationUseCase(repository)

def get_management_use_case(repository=None):
    """Get management use case instance."""
    if repository is None:
        repository = get_repository()
    return ManageQualityRulesUseCase(repository)

def get_current_user():
    """Get current user (mock implementation)."""
    return UserId()

@app.command("create-rule")
def create_rule(
    rule_name: str = typer.Argument(..., help="Name of the quality rule"),
    rule_type: str = typer.Option(..., "--type", "-t", help="Rule type: completeness, uniqueness, validity, consistency, accuracy"),
    target_columns: str = typer.Option(..., "--columns", "-c", help="Comma-separated list of target columns"),
    logic_type: str = typer.Option("regex", "--logic", "-l", help="Logic type: regex, range, sql, python"),
    expression: str = typer.Option(..., "--expression", "-e", help="Validation expression"),
    severity: str = typer.Option("medium", "--severity", "-s", help="Rule severity: low, medium, high, critical"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Rule description"),
    pass_threshold: float = typer.Option(0.95, "--pass-threshold", help="Pass rate threshold (0-1)"),
    warning_threshold: float = typer.Option(0.90, "--warning-threshold", help="Warning threshold (0-1)"),
    critical_threshold: float = typer.Option(0.80, "--critical-threshold", help="Critical threshold (0-1)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Create a new data quality rule."""
    try:
        console.print(f"[bold blue]🔍 Creating Quality Rule[/bold blue]")
        console.print(f"Rule name: {rule_name}")
        console.print(f"Rule type: {rule_type}")
        console.print(f"Target columns: {target_columns}")
        
        # Parse target columns
        target_column_list = [col.strip() for col in target_columns.split(',')]
        
        # Create validation logic
        validation_logic = ValidationLogic(
            logic_type=LogicType(logic_type),
            expression=expression,
            parameters={},
            error_message_template=f"Validation failed for rule '{rule_name}': {{value}}",
            success_criteria=None
        )
        
        # Create quality thresholds
        thresholds = QualityThreshold(
            pass_rate_threshold=pass_threshold,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold
        )
        
        # Create rule metadata
        metadata = RuleMetadata(
            description=description or f"Quality rule for {rule_type} validation",
            business_justification=f"Ensure data {rule_type} for business operations",
            data_owner=None,
            business_glossary_terms=[],
            related_regulations=[],
            documentation_url=None
        )
        
        # Create mock dataset IDs
        target_datasets = [DatasetId(value=uuid4())]
        
        # Create rule
        use_case = get_management_use_case()
        current_user = get_current_user()
        
        rule = asyncio.run(
            use_case.create_rule(
                rule_name=rule_name,
                rule_type=RuleType(rule_type),
                category=QualityCategory.DATA_VALIDATION,
                severity=Severity(severity),
                validation_logic=validation_logic,
                metadata=metadata,
                created_by=current_user,
                target_datasets=target_datasets,
                target_columns=target_column_list
            )
        )
        
        # Display created rule
        console.print(f"[green]✅ Rule created successfully[/green]")
        console.print(f"Rule ID: {rule.rule_id.value}")
        
        if verbose:
            _display_rule_details(rule)
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

@app.command("validate")
def validate_data(
    data_file: Path = typer.Argument(..., help="Path to dataset file (CSV, JSON, Parquet)"),
    rule_ids: Optional[str] = typer.Option(None, "--rules", "-r", help="Comma-separated list of rule IDs"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results (JSON)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Validate dataset against quality rules."""
    try:
        console.print(f"[bold blue]🔍 Data Quality Validation[/bold blue]")
        console.print(f"Data file: {data_file}")
        
        # Validate file exists
        if not data_file.exists():
            console.print(f"[red]❌ File not found: {data_file}[/red]")
            raise typer.Exit(1)
        
        # Load data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)
            
            if data_file.suffix.lower() == '.csv':
                data = pd.read_csv(data_file)
            elif data_file.suffix.lower() == '.json':
                data = pd.read_json(data_file)
            elif data_file.suffix.lower() == '.parquet':
                data = pd.read_parquet(data_file)
            else:
                console.print(f"[red]❌ Unsupported file format: {data_file.suffix}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description=f"Loaded {len(data)} rows, {len(data.columns)} columns")
        
        # Parse rule IDs if provided
        rule_id_list = []
        if rule_ids:
            rule_id_list = [rule_id.strip() for rule_id in rule_ids.split(',')]
            console.print(f"Using specific rules: {rule_id_list}")
        else:
            console.print("Using all active rules for dataset")
        
        # Execute validation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing validation...", total=None)
            
            # Create mock dataset ID
            dataset_id = DatasetId(value=uuid4())
            
            use_case = get_validation_use_case()
            
            if rule_id_list:
                # Execute specific rules
                results = []
                for rule_id_str in rule_id_list:
                    try:
                        rule = await use_case.get_rule_by_id(RuleId(value=UUID(rule_id_str)))
                        if rule:
                            result = await use_case.execute_rule_validation(rule, data, dataset_id)
                            results.append(result)
                        else:
                            console.print(f"[yellow]⚠️ Rule not found: {rule_id_str}[/yellow]")
                    except ValueError:
                        console.print(f"[yellow]⚠️ Invalid rule ID format: {rule_id_str}[/yellow]")
                results = asyncio.run(_execute_specific_rules_async(use_case, rule_id_list, data, dataset_id))
            else:
                # Execute all active rules for dataset (mock implementation)
                results = await use_case.execute_dataset_validation(dataset_id, data)
                results = asyncio.run(_execute_dataset_validation_async(use_case, dataset_id, data))
            
            progress.update(task, description="Validation completed")
        
        # Display results
        _display_validation_results(results, verbose)
        
        # Save results if output specified
        if output:
            results_dict = _convert_validation_results_to_dict(results)
            
            with open(output, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            
            console.print(f"[green]✅ Results saved to: {output}[/green]")
        
        console.print("[green]✅ Validation completed successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

async def _execute_specific_rules_async(use_case, rule_id_list, data, dataset_id):
    """Execute specific rules asynchronously."""
    results = []
    for rule_id_str in rule_id_list:
        try:
            rule = await use_case.get_rule_by_id(RuleId(value=UUID(rule_id_str)))
            if rule:
                result = await use_case.execute_rule_validation(rule, data, dataset_id)
                results.append(result)
            else:
                console.print(f"[yellow]⚠️ Rule not found: {rule_id_str}[/yellow]")
        except ValueError:
            console.print(f"[yellow]⚠️ Invalid rule ID format: {rule_id_str}[/yellow]")
    return results

async def _execute_dataset_validation_async(use_case, dataset_id, data):
    """Execute dataset validation asynchronously."""
    return await use_case.execute_dataset_validation(dataset_id, data)

@app.command("list-rules")
def list_rules(
    rule_type: Optional[str] = typer.Option(None, "--type", help="Filter by rule type"),
    category: Optional[str] = typer.Option(None, "--category", help="Filter by category"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """List quality rules."""
    try:
        use_case = get_management_use_case()
        
        # Get rules based on filters
        if rule_type:
            rules = asyncio.run(use_case.get_rules_by_type(RuleType(rule_type)))
        elif category:
            rules = asyncio.run(use_case.get_rules_by_category(QualityCategory(category)))
        else:
            rules = asyncio.run(use_case.repository.list_all())
        
        # Apply status filter if provided
        if status:
            from ...domain.entities.quality_rule import RuleStatus
            rules = [r for r in rules if r.status.value == status]
        
        # Limit results
        rules = rules[:limit]
        
        if not rules:
            console.print("[yellow]No rules found[/yellow]")
            return
        
        # Display table
        table = Table(title="Quality Rules")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Severity", style="red")
        table.add_column("Status", style="blue")
        table.add_column("Created", style="dim")
        
        if verbose:
            table.add_column("Columns", style="white")
            table.add_column("Logic", style="magenta")
        
        for rule in rules:
            row = [
                str(rule.rule_id.value)[:8] + "...",
                rule.rule_name,
                rule.rule_type.value,
                rule.severity.value,
                rule.status.value,
                rule.created_at.strftime("%Y-%m-%d %H:%M")
            ]
            
            if verbose:
                columns_str = ", ".join(rule.target_columns[:3]) + ("..." if len(rule.target_columns) > 3 else "")
                logic_str = f"{rule.validation_logic.logic_type.value}: {rule.validation_logic.expression[:30]}..."
                row.extend([columns_str, logic_str])
            
            table.add_row(*row)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("get-rule")
def get_rule(
    rule_id: str = typer.Argument(..., help="Rule ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """Get details of a specific quality rule."""
    try:
        use_case = get_management_use_case()
        
        # Get rule
        rule = asyncio.run(
            use_case.get_rule_by_id(RuleId(value=UUID(rule_id)))
        )
        
        if not rule:
            console.print(f"[red]❌ Rule not found: {rule_id}[/red]")
            raise typer.Exit(1)
        
        # Display rule details
        _display_rule_details(rule, verbose)
        
    except ValueError as e:
        console.print(f"[red]❌ Invalid rule ID: {rule_id}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("activate-rule")
def activate_rule(
    rule_id: str = typer.Argument(..., help="Rule ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Activate a quality rule."""
    try:
        use_case = get_management_use_case()
        current_user = get_current_user()
        
        # Activate rule
        rule = asyncio.run(
            use_case.activate_rule(
                RuleId(value=UUID(rule_id)),
                approved_by=current_user
            )
        )
        
        console.print(f"[green]✅ Rule activated: {rule_id}[/green]")
        
        if verbose:
            _display_rule_details(rule)
        
    except ValueError as e:
        console.print(f"[red]❌ Invalid rule ID: {rule_id}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("deactivate-rule")
def deactivate_rule(
    rule_id: str = typer.Argument(..., help="Rule ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Deactivate a quality rule."""
    try:
        use_case = get_management_use_case()
        
        # Deactivate rule
        rule = asyncio.run(
            use_case.deactivate_rule(RuleId(value=UUID(rule_id)))
        )
        
        console.print(f"[green]✅ Rule deactivated: {rule_id}[/green]")
        
        if verbose:
            _display_rule_details(rule)
        
    except ValueError as e:
        console.print(f"[red]❌ Invalid rule ID: {rule_id}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("violations")
def get_violations(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """Get rules with current threshold violations."""
    try:
        use_case = get_management_use_case()
        
        # Get rules with violations
        rules = asyncio.run(use_case.get_rules_with_violations())
        
        if not rules:
            console.print("[green]✅ No threshold violations found[/green]")
            return
        
        console.print(f"[red]⚠️ Found {len(rules)} rules with threshold violations[/red]")
        
        # Display violations table
        table = Table(title="Rules with Threshold Violations")
        table.add_column("Rule ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="yellow")
        table.add_column("Type", style="blue")
        table.add_column("Severity", style="red")
        table.add_column("Last Result", style="white")
        
        for rule in rules:
            # Get most recent result
            recent_result = rule.recent_results[-1] if rule.recent_results else None
            last_result = f"{recent_result.pass_rate:.2f}" if recent_result else "N/A"
            
            table.add_row(
                str(rule.rule_id.value)[:8] + "...",
                rule.rule_name,
                rule.rule_type.value,
                rule.severity.value,
                last_result
            )
        
        console.print(table)
        
        if verbose:
            for rule in rules:
                console.print(f"\n[bold]Rule: {rule.rule_name}[/bold]")
                if rule.recent_results:
                    result = rule.recent_results[-1]
                    console.print(f"  Pass Rate: {result.pass_rate:.2f}")
                    console.print(f"  Threshold: {rule.quality_thresholds.pass_rate_threshold:.2f}")
                    console.print(f"  Failed Records: {result.records_failed}")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("health")
def health_check():
    """Check the health of the data quality service."""
    try:
        console.print("[blue]🔍 Checking service health...[/blue]")
        
        # Test repository connection
        repository = get_repository()
        
        # Basic health check
        health_status = {
            "service": "data-quality",
            "status": "healthy",
            "repository": "connected",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Display health status
        table = Table(title="Service Health")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        for key, value in health_status.items():
            if key != "timestamp":
                table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)
        console.print(f"[green]✅ Service is healthy[/green]")
        console.print(f"[dim]Checked at: {health_status['timestamp']}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Service unhealthy: {str(e)}[/red]")
        raise typer.Exit(1)

def _display_rule_details(rule, verbose: bool = False):
    """Display rule details in a formatted way."""
    
    # Basic information panel
    info_content = f"""
[bold]Rule ID:[/bold] {rule.rule_id.value}
[bold]Name:[/bold] {rule.rule_name}
[bold]Type:[/bold] {rule.rule_type.value}
[bold]Category:[/bold] {rule.category.value}
[bold]Severity:[/bold] {rule.severity.value}
[bold]Status:[/bold] {rule.status.value}
[bold]Target Columns:[/bold] {', '.join(rule.target_columns)}
[bold]Created:[/bold] {rule.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    if rule.approved_by:
        info_content += f"[bold]Approved By:[/bold] {rule.approved_by.value}\n"
        info_content += f"[bold]Approved At:[/bold] {rule.approved_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    console.print(Panel(info_content.strip(), title="Rule Information", border_style="blue"))
    
    # Validation Logic
    logic_content = f"""
[bold]Logic Type:[/bold] {rule.validation_logic.logic_type.value}
[bold]Expression:[/bold] {rule.validation_logic.expression}
[bold]Error Template:[/bold] {rule.validation_logic.error_message_template}
"""
    
    if rule.validation_logic.parameters:
        logic_content += f"[bold]Parameters:[/bold] {rule.validation_logic.parameters}\n"
    
    console.print(Panel(logic_content.strip(), title="Validation Logic", border_style="green"))
    
    # Quality Thresholds
    if hasattr(rule, 'quality_thresholds') and rule.quality_thresholds:
        thresholds_content = f"""
[bold]Pass Rate Threshold:[/bold] {rule.quality_thresholds.pass_rate_threshold:.2f}
[bold]Warning Threshold:[/bold] {rule.quality_thresholds.warning_threshold:.2f}
[bold]Critical Threshold:[/bold] {rule.quality_thresholds.critical_threshold:.2f}
"""
        console.print(Panel(thresholds_content.strip(), title="Quality Thresholds", border_style="yellow"))
    
    # Recent Results
    if verbose and rule.recent_results:
        console.print("\n[bold]📊 Recent Validation Results:[/bold]")
        
        results_table = Table()
        results_table.add_column("Date", style="dim")
        results_table.add_column("Status", style="yellow")
        results_table.add_column("Pass Rate", style="green")
        results_table.add_column("Failed Records", style="red")
        results_table.add_column("Execution Time", style="blue")
        
        for result in rule.recent_results[-5:]:  # Show last 5 results
            results_table.add_row(
                result.executed_at.strftime("%Y-%m-%d %H:%M"),
                result.status.value,
                f"{result.pass_rate:.2f}",
                str(result.records_failed),
                f"{result.execution_time_seconds:.2f}s"
            )
        
        console.print(results_table)
    
    # Metadata
    if verbose and rule.rule_metadata:
        metadata_content = f"""
[bold]Description:[/bold] {rule.rule_metadata.description}
[bold]Business Justification:[/bold] {rule.rule_metadata.business_justification}
"""
        
        if rule.rule_metadata.data_owner:
            metadata_content += f"[bold]Data Owner:[/bold] {rule.rule_metadata.data_owner}\n"
        
        if rule.rule_metadata.documentation_url:
            metadata_content += f"[bold]Documentation:[/bold] {rule.rule_metadata.documentation_url}\n"
        
        console.print(Panel(metadata_content.strip(), title="Rule Metadata", border_style="magenta"))

def _display_validation_results(results, verbose: bool = False):
    """Display validation results in a formatted way."""
    
    if not results:
        console.print("[yellow]No validation results[/yellow]")
        return
    
    # Summary panel
    total_rules = len(results)
    passed_rules = sum(1 for r in results if r.pass_rate >= 0.95)  # Assuming 95% pass rate threshold
    failed_rules = total_rules - passed_rules
    
    summary_content = f"""
[bold]Total Rules Executed:[/bold] {total_rules}
[bold]Rules Passed:[/bold] {passed_rules}
[bold]Rules Failed:[/bold] {failed_rules}
[bold]Overall Success Rate:[/bold] {(passed_rules/total_rules)*100:.1f}%
"""
    
    console.print(Panel(summary_content.strip(), title="Validation Summary", border_style="blue"))
    
    # Results table
    table = Table(title="Validation Results")
    table.add_column("Rule ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="yellow")
    table.add_column("Total Records", style="blue")
    table.add_column("Failed Records", style="red")
    table.add_column("Pass Rate", style="green")
    table.add_column("Execution Time", style="dim")
    
    for result in results:
        status_style = "green" if result.pass_rate >= 0.95 else "red"
        
        table.add_row(
            str(result.rule_id.value)[:8] + "...",
            f"[{status_style}]{result.status.value}[/{status_style}]",
            str(result.total_records),
            str(result.records_failed),
            f"{result.pass_rate:.2f}",
            f"{result.execution_time_seconds:.2f}s"
        )
    
    console.print(table)
    
    # Detailed errors if verbose
    if verbose:
        for result in results:
            if result.validation_errors:
                console.print(f"\n[bold red]❌ Errors for Rule {result.rule_id.value}:[/bold red]")
                
                error_table = Table()
                error_table.add_column("Row", style="dim")
                error_table.add_column("Column", style="cyan")
                error_table.add_column("Invalid Value", style="yellow")
                error_table.add_column("Error Message", style="red")
                
                for error in result.validation_errors[:10]:  # Show first 10 errors
                    error_table.add_row(
                        error.row_identifier or "N/A",
                        error.column_name or "N/A",
                        str(error.invalid_value) if error.invalid_value is not None else "N/A",
                        error.error_message
                    )
                
                console.print(error_table)
                
                if len(result.validation_errors) > 10:
                    console.print(f"[dim]... and {len(result.validation_errors) - 10} more errors[/dim]")

def _convert_validation_results_to_dict(results) -> Dict[str, Any]:
    """Convert validation results to dictionary for JSON serialization."""
    return {
        "validation_summary": {
            "total_rules": len(results),
            "timestamp": pd.Timestamp.now().isoformat(),
            "overall_pass_rate": sum(r.pass_rate for r in results) / len(results) if results else 0
        },
        "results": [
            {
                "rule_id": str(result.rule_id.value),
                "dataset_id": str(result.dataset_id.value),
                "status": result.status.value,
                "total_records": result.total_records,
                "records_passed": result.records_passed,
                "records_failed": result.records_failed,
                "pass_rate": result.pass_rate,
                "execution_time_seconds": result.execution_time_seconds,
                "executed_at": result.executed_at.isoformat(),
                "validation_errors": [
                    {
                        "error_id": str(error.error_id),
                        "row_identifier": error.row_identifier,
                        "column_name": error.column_name,
                        "invalid_value": str(error.invalid_value) if error.invalid_value is not None else None,
                        "error_message": error.error_message,
                        "error_code": error.error_code
                    }
                    for error in result.validation_errors
                ]
            }
            for result in results
        ]
    }

if __name__ == "__main__":
    app()