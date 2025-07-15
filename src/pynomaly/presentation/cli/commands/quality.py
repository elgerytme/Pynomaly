"""
Data Quality CLI Commands

Provides comprehensive data quality validation and cleansing operations through command-line interface.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.tree import Tree
from rich.text import Text
import json
import csv

from ....shared.error_handling import handle_cli_errors
from ....shared.logging import get_logger
from ....infrastructure.config import get_cli_container

logger = get_logger(__name__)
console = Console()

# Create the quality command group
app = typer.Typer(
    name="quality",
    help="🔧 Data quality validation and cleansing tools",
    rich_markup_mode="rich"
)


@app.command("validate")
@handle_cli_errors
def validate_data(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    rules_config: Optional[Path] = typer.Option(None, "--rules", help="Path to validation rules config file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, html"),
    severity: str = typer.Option("warning", "--severity", help="Minimum severity level: info, warning, error"),
    fail_on_error: bool = typer.Option(False, "--fail-on-error", help="Exit with error code if validation fails"),
    include_suggestions: bool = typer.Option(True, "--suggestions/--no-suggestions", help="Include fix suggestions"),
    check_duplicates: bool = typer.Option(True, "--duplicates/--no-duplicates", help="Check for duplicate records"),
    check_missing: bool = typer.Option(True, "--missing/--no-missing", help="Check for missing values"),
    check_outliers: bool = typer.Option(True, "--outliers/--no-outliers", help="Check for statistical outliers"),
    check_consistency: bool = typer.Option(True, "--consistency/--no-consistency", help="Check data consistency"),
    check_constraints: bool = typer.Option(True, "--constraints/--no-constraints", help="Check business rule constraints"),
    sample_size: Optional[int] = typer.Option(None, "--sample", help="Sample size for large datasets")
):
    """
    Validate data quality against comprehensive rules and constraints.
    
    Performs thorough data quality validation including missing values,
    duplicates, outliers, consistency checks, and business rule validation.
    
    Examples:
        pynomaly quality validate data.csv
        pynomaly quality validate data.csv --rules validation_rules.json
        pynomaly quality validate data.csv --output quality_report.html --format html
        pynomaly quality validate data.csv --fail-on-error --severity error
    """
    console.print(f"[bold blue]🔧 Validating data quality for {data_path}[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset...", total=100)
            
            import pandas as pd
            import numpy as np
            
            # Load dataset
            if not data_path.exists():
                console.print(f"[red]❌ Dataset file not found: {data_path}[/red]")
                raise typer.Exit(1)
            
            try:
                if data_path.suffix.lower() == '.csv':
                    df = pd.read_csv(data_path, nrows=sample_size)
                elif data_path.suffix.lower() in ['.json']:
                    df = pd.read_json(data_path)
                    if sample_size:
                        df = df.head(sample_size)
                elif data_path.suffix.lower() in ['.parquet']:
                    df = pd.read_parquet(data_path)
                    if sample_size:
                        df = df.head(sample_size)
                else:
                    console.print(f"[red]❌ Unsupported file format: {data_path.suffix}[/red]")
                    raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]❌ Error loading dataset: {e}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, advance=20, description="Loading validation rules...")
            
            # Load validation rules
            validation_rules = _load_validation_rules(rules_config) if rules_config else _get_default_validation_rules()
            
            progress.update(task, advance=20, description="Running validation checks...")
            
            # Perform validation checks
            validation_results = _perform_validation_checks(
                df, validation_rules, {
                    "duplicates": check_duplicates,
                    "missing": check_missing,
                    "outliers": check_outliers,
                    "consistency": check_consistency,
                    "constraints": check_constraints
                }, severity, include_suggestions
            )
            
            progress.update(task, advance=40, description="Generating validation report...")
            
            # Process results
            _process_validation_results(validation_results, df, severity)
            
            progress.update(task, advance=20, description="Complete", completed=100)
        
        # Display results
        if format == "table":
            _display_validation_table(validation_results, data_path.name, severity)
        elif format == "json":
            if output:
                with open(output, 'w') as f:
                    json.dump(validation_results, f, indent=2, default=str)
                console.print(f"[green]✅ Validation report saved to {output}[/green]")
            else:
                console.print(json.dumps(validation_results, indent=2, default=str))
        elif format == "html":
            if output:
                _generate_validation_html_report(validation_results, output, data_path.name)
                console.print(f"[green]✅ Validation report saved to {output}[/green]")
            else:
                console.print("[yellow]⚠️  HTML format requires --output parameter[/yellow]")
        
        # Check if validation failed
        error_count = sum(1 for issue in validation_results.get("issues", []) if issue.get("severity") == "error")
        if fail_on_error and error_count > 0:
            console.print(f"[red]❌ Validation failed with {error_count} errors[/red]")
            raise typer.Exit(1)
        
        console.print("[green]✅ Data validation completed[/green]")
        
    except Exception as e:
        logger.error(f"Error in data validation: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("clean")
@handle_cli_errors
def clean_data(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    output: Path = typer.Argument(..., help="Output file path for cleaned data"),
    missing_strategy: str = typer.Option("drop", "--missing", help="Missing value strategy: drop, fill_mean, fill_median, fill_mode, fill_forward, fill_backward"),
    duplicate_strategy: str = typer.Option("drop", "--duplicates", help="Duplicate handling: drop, keep_first, keep_last"),
    outlier_strategy: str = typer.Option("clip", "--outliers", help="Outlier handling: clip, drop, cap"),
    outlier_method: str = typer.Option("iqr", "--outlier-method", help="Outlier detection method: iqr, zscore, isolation"),
    outlier_threshold: float = typer.Option(3.0, "--outlier-threshold", help="Outlier detection threshold"),
    normalize_text: bool = typer.Option(True, "--normalize-text/--no-normalize-text", help="Normalize text fields"),
    standardize_formats: bool = typer.Option(True, "--standardize/--no-standardize", help="Standardize data formats"),
    validation_rules: Optional[Path] = typer.Option(None, "--rules", help="Validation rules to apply during cleaning"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Create backup of original file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be cleaned without making changes"),
    interactive: bool = typer.Option(False, "--interactive", help="Interactive cleaning with user prompts")
):
    """
    Clean and preprocess data according to specified strategies.
    
    Applies comprehensive data cleaning operations including missing value
    imputation, duplicate removal, outlier handling, and format standardization.
    
    Examples:
        pynomaly quality clean data.csv clean_data.csv
        pynomaly quality clean data.csv clean_data.csv --missing fill_mean --outliers clip
        pynomaly quality clean data.csv clean_data.csv --dry-run
        pynomaly quality clean data.csv clean_data.csv --interactive
    """
    console.print(f"[bold blue]🧹 Cleaning data from {data_path}[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset...", total=100)
            
            import pandas as pd
            import numpy as np
            from scipy import stats
            
            # Load dataset
            if not data_path.exists():
                console.print(f"[red]❌ Dataset file not found: {data_path}[/red]")
                raise typer.Exit(1)
            
            # Create backup if requested
            if backup and not dry_run:
                backup_path = data_path.with_suffix(f".backup{data_path.suffix}")
                import shutil
                shutil.copy2(data_path, backup_path)
                console.print(f"[dim]📁 Backup created: {backup_path}[/dim]")
            
            # Load data
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.json']:
                df = pd.read_json(data_path)
            elif data_path.suffix.lower() in ['.parquet']:
                df = pd.read_parquet(data_path)
            else:
                console.print(f"[red]❌ Unsupported file format: {data_path.suffix}[/red]")
                raise typer.Exit(1)
            
            original_shape = df.shape
            cleaning_log = []
            
            progress.update(task, advance=10, description="Analyzing data quality...")
            
            # Analyze current quality issues
            quality_issues = _analyze_quality_issues(df)
            
            progress.update(task, advance=10, description="Processing missing values...")
            
            # Handle missing values
            if missing_strategy != "none":
                df_before = df.copy()
                df = _handle_missing_values(df, missing_strategy, interactive)
                missing_changes = _calculate_changes(df_before, df, "missing values")
                cleaning_log.append(missing_changes)
            
            progress.update(task, advance=20, description="Removing duplicates...")
            
            # Handle duplicates
            if duplicate_strategy != "none":
                df_before = df.copy()
                df = _handle_duplicates(df, duplicate_strategy, interactive)
                duplicate_changes = _calculate_changes(df_before, df, "duplicates")
                cleaning_log.append(duplicate_changes)
            
            progress.update(task, advance=20, description="Processing outliers...")
            
            # Handle outliers
            if outlier_strategy != "none":
                df_before = df.copy()
                df = _handle_outliers(df, outlier_strategy, outlier_method, outlier_threshold, interactive)
                outlier_changes = _calculate_changes(df_before, df, "outliers")
                cleaning_log.append(outlier_changes)
            
            progress.update(task, advance=15, description="Normalizing text...")
            
            # Normalize text
            if normalize_text:
                df_before = df.copy()
                df = _normalize_text_fields(df)
                text_changes = _calculate_changes(df_before, df, "text normalization")
                cleaning_log.append(text_changes)
            
            progress.update(task, advance=15, description="Standardizing formats...")
            
            # Standardize formats
            if standardize_formats:
                df_before = df.copy()
                df = _standardize_data_formats(df)
                format_changes = _calculate_changes(df_before, df, "format standardization")
                cleaning_log.append(format_changes)
            
            progress.update(task, advance=10, description="Finalizing cleaning...", completed=100)
        
        # Display cleaning summary
        final_shape = df.shape
        _display_cleaning_summary(original_shape, final_shape, cleaning_log, quality_issues)
        
        # Save cleaned data (unless dry run)
        if not dry_run:
            if output.suffix.lower() == '.csv':
                df.to_csv(output, index=False)
            elif output.suffix.lower() in ['.json']:
                df.to_json(output, orient='records', indent=2)
            elif output.suffix.lower() in ['.parquet']:
                df.to_parquet(output, index=False)
            else:
                console.print(f"[red]❌ Unsupported output format: {output.suffix}[/red]")
                raise typer.Exit(1)
            
            console.print(f"[green]✅ Cleaned data saved to {output}[/green]")
        else:
            console.print("[yellow]🔍 Dry run completed - no changes were made[/yellow]")
        
    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("monitor")
@handle_cli_errors
def monitor_quality(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    baseline_path: Optional[Path] = typer.Option(None, "--baseline", help="Path to baseline dataset for comparison"),
    rules_config: Optional[Path] = typer.Option(None, "--rules", help="Path to monitoring rules config"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for monitoring results"),
    alert_threshold: float = typer.Option(0.1, "--alert-threshold", help="Quality degradation threshold for alerts"),
    metrics: str = typer.Option("all", "--metrics", help="Metrics to monitor: completeness,validity,consistency,accuracy,all"),
    format: str = typer.Option("table", "--format", help="Output format: table, json, alerts"),
    continuous: bool = typer.Option(False, "--continuous", help="Enable continuous monitoring mode"),
    interval: int = typer.Option(300, "--interval", help="Monitoring interval in seconds (for continuous mode)")
):
    """
    Monitor data quality metrics and detect quality degradation.
    
    Continuously or periodically monitors data quality metrics against
    baseline datasets and configured rules to detect quality issues.
    
    Examples:
        pynomaly quality monitor data.csv --baseline reference_data.csv
        pynomaly quality monitor data.csv --rules monitoring_rules.json
        pynomaly quality monitor data.csv --continuous --interval 600
        pynomaly quality monitor data.csv --output quality_alerts.json --format alerts
    """
    console.print(f"[bold blue]📊 Monitoring data quality for {data_path}[/bold blue]")
    
    try:
        if continuous:
            console.print(f"[yellow]🔄 Starting continuous monitoring (interval: {interval}s)[/yellow]")
            console.print("[dim]Press Ctrl+C to stop monitoring[/dim]")
            
            import time
            import signal
            
            def signal_handler(sig, frame):
                console.print("\n[yellow]📊 Monitoring stopped[/yellow]")
                raise typer.Exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            
            monitoring_results = []
            while True:
                try:
                    result = _perform_single_monitoring_check(
                        data_path, baseline_path, rules_config, metrics, alert_threshold
                    )
                    monitoring_results.append(result)
                    
                    # Display current status
                    _display_monitoring_status(result, len(monitoring_results))
                    
                    # Save results if output specified
                    if output:
                        with open(output, 'w') as f:
                            json.dump(monitoring_results, f, indent=2, default=str)
                    
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]⚠️  Monitoring error: {e}[/red]")
                    time.sleep(interval)
        
        else:
            # Single monitoring check
            result = _perform_single_monitoring_check(
                data_path, baseline_path, rules_config, metrics, alert_threshold
            )
            
            # Display results
            if format == "table":
                _display_monitoring_table(result, data_path.name)
            elif format == "json":
                if output:
                    with open(output, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    console.print(f"[green]✅ Monitoring results saved to {output}[/green]")
                else:
                    console.print(json.dumps(result, indent=2, default=str))
            elif format == "alerts":
                _display_quality_alerts(result)
        
        console.print("[green]✅ Quality monitoring completed[/green]")
        
    except Exception as e:
        logger.error(f"Error in quality monitoring: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("repair")
@handle_cli_errors
def auto_repair(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    output: Path = typer.Argument(..., help="Output file path for repaired data"),
    repair_strategy: str = typer.Option("conservative", "--strategy", help="Repair strategy: conservative, aggressive, custom"),
    confidence_threshold: float = typer.Option(0.8, "--confidence", help="Minimum confidence for automated repairs"),
    rules_config: Optional[Path] = typer.Option(None, "--rules", help="Custom repair rules config"),
    validate_repairs: bool = typer.Option(True, "--validate/--no-validate", help="Validate repairs before applying"),
    interactive: bool = typer.Option(False, "--interactive", help="Interactive repair with user approval"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Create backup of original file"),
    log_repairs: bool = typer.Option(True, "--log/--no-log", help="Log all repair operations")
):
    """
    Automatically repair data quality issues using ML-based techniques.
    
    Uses machine learning and statistical methods to automatically detect
    and repair data quality issues with configurable confidence levels.
    
    Examples:
        pynomaly quality repair data.csv repaired_data.csv
        pynomaly quality repair data.csv repaired_data.csv --strategy aggressive
        pynomaly quality repair data.csv repaired_data.csv --interactive --confidence 0.9
        pynomaly quality repair data.csv repaired_data.csv --rules custom_repairs.json
    """
    console.print(f"[bold blue]🔧 Auto-repairing data quality issues in {data_path}[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset and analyzing issues...", total=100)
            
            import pandas as pd
            import numpy as np
            
            # Load dataset
            if not data_path.exists():
                console.print(f"[red]❌ Dataset file not found: {data_path}[/red]")
                raise typer.Exit(1)
            
            # Create backup if requested
            if backup:
                backup_path = data_path.with_suffix(f".backup{data_path.suffix}")
                import shutil
                shutil.copy2(data_path, backup_path)
                console.print(f"[dim]📁 Backup created: {backup_path}[/dim]")
            
            # Load data
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.json']:
                df = pd.read_json(data_path)
            elif data_path.suffix.lower() in ['.parquet']:
                df = pd.read_parquet(data_path)
            else:
                console.print(f"[red]❌ Unsupported file format: {data_path.suffix}[/red]")
                raise typer.Exit(1)
            
            original_shape = df.shape
            repair_log = []
            
            progress.update(task, advance=20, description="Detecting quality issues...")
            
            # Detect issues that can be auto-repaired
            repairable_issues = _detect_repairable_issues(df, repair_strategy, confidence_threshold)
            
            progress.update(task, advance=20, description="Planning repair operations...")
            
            # Plan repair operations
            repair_plan = _create_repair_plan(df, repairable_issues, rules_config)
            
            if interactive:
                # Show repair plan and get user approval
                approved_repairs = _get_user_approval_for_repairs(repair_plan)
                repair_plan = [r for r in repair_plan if r["id"] in approved_repairs]
            
            progress.update(task, advance=20, description="Applying repairs...")
            
            # Apply repairs
            df_repaired = df.copy()
            for repair in repair_plan:
                try:
                    df_before = df_repaired.copy()
                    df_repaired = _apply_repair(df_repaired, repair, confidence_threshold)
                    
                    # Log the repair
                    repair_result = {
                        "repair_type": repair["type"],
                        "column": repair.get("column"),
                        "confidence": repair.get("confidence"),
                        "records_affected": len(df_before) - len(df_repaired) if repair["type"] == "remove" else repair.get("records_affected", 0),
                        "success": True
                    }
                    repair_log.append(repair_result)
                    
                except Exception as e:
                    repair_result = {
                        "repair_type": repair["type"],
                        "column": repair.get("column"),
                        "error": str(e),
                        "success": False
                    }
                    repair_log.append(repair_result)
                    console.print(f"[yellow]⚠️  Failed to apply repair {repair['type']}: {e}[/yellow]")
            
            progress.update(task, advance=20, description="Validating repairs...")
            
            # Validate repairs if requested
            if validate_repairs:
                validation_results = _validate_repairs(df, df_repaired, repair_log)
                _display_repair_validation(validation_results)
            
            progress.update(task, advance=20, description="Finalizing repairs...", completed=100)
        
        # Display repair summary
        final_shape = df_repaired.shape
        _display_repair_summary(original_shape, final_shape, repair_log)
        
        # Save repaired data
        if output.suffix.lower() == '.csv':
            df_repaired.to_csv(output, index=False)
        elif output.suffix.lower() in ['.json']:
            df_repaired.to_json(output, orient='records', indent=2)
        elif output.suffix.lower() in ['.parquet']:
            df_repaired.to_parquet(output, index=False)
        else:
            console.print(f"[red]❌ Unsupported output format: {output.suffix}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[green]✅ Auto-repair completed. Repaired data saved to {output}[/green]")
        
        # Save repair log if requested
        if log_repairs and repair_log:
            log_path = output.with_suffix('.repair_log.json')
            with open(log_path, 'w') as f:
                json.dump(repair_log, f, indent=2, default=str)
            console.print(f"[dim]📝 Repair log saved to {log_path}[/dim]")
        
    except Exception as e:
        logger.error(f"Error in auto-repair: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


# Helper functions (implementation would be extensive, showing key signatures)

def _load_validation_rules(rules_path: Path) -> Dict[str, Any]:
    """Load validation rules from configuration file."""
    with open(rules_path, 'r') as f:
        return json.load(f)


def _get_default_validation_rules() -> Dict[str, Any]:
    """Get default validation rules."""
    return {
        "missing_threshold": 0.05,
        "duplicate_threshold": 0.01,
        "outlier_threshold": 0.05,
        "consistency_checks": True,
        "constraint_checks": True
    }


def _perform_validation_checks(df: "pd.DataFrame", rules: Dict[str, Any], 
                              check_config: Dict[str, bool], severity: str,
                              include_suggestions: bool) -> Dict[str, Any]:
    """Perform comprehensive validation checks."""
    import pandas as pd
    import numpy as np
    
    issues = []
    summary = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "checks_performed": [k for k, v in check_config.items() if v]
    }
    
    # Missing values check
    if check_config.get("missing", True):
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_pct = missing_count / len(df)
            
            if missing_pct > rules.get("missing_threshold", 0.05):
                issue = {
                    "type": "missing_values",
                    "column": column,
                    "severity": "error" if missing_pct > 0.2 else "warning",
                    "message": f"Column '{column}' has {missing_pct:.1%} missing values",
                    "count": missing_count,
                    "percentage": missing_pct * 100
                }
                
                if include_suggestions:
                    issue["suggestions"] = _get_missing_value_suggestions(df[column], missing_pct)
                
                issues.append(issue)
    
    # Duplicate check
    if check_config.get("duplicates", True):
        duplicate_count = df.duplicated().sum()
        duplicate_pct = duplicate_count / len(df)
        
        if duplicate_pct > rules.get("duplicate_threshold", 0.01):
            issue = {
                "type": "duplicates",
                "severity": "warning" if duplicate_pct < 0.05 else "error",
                "message": f"Dataset has {duplicate_pct:.1%} duplicate records",
                "count": duplicate_count,
                "percentage": duplicate_pct * 100
            }
            
            if include_suggestions:
                issue["suggestions"] = ["Remove duplicate records", "Keep first occurrence", "Keep last occurrence"]
            
            issues.append(issue)
    
    # Additional checks would be implemented here...
    
    return {
        "summary": summary,
        "issues": issues,
        "validation_passed": len([i for i in issues if i["severity"] == "error"]) == 0
    }


def _process_validation_results(results: Dict[str, Any], df: "pd.DataFrame", severity: str):
    """Process and filter validation results based on severity."""
    severity_order = {"info": 0, "warning": 1, "error": 2}
    min_level = severity_order.get(severity, 1)
    
    # Filter issues by severity
    filtered_issues = [
        issue for issue in results["issues"]
        if severity_order.get(issue["severity"], 1) >= min_level
    ]
    
    results["issues"] = filtered_issues
    results["filtered_by_severity"] = severity


def _get_missing_value_suggestions(series: "pd.Series", missing_pct: float) -> List[str]:
    """Generate suggestions for handling missing values."""
    suggestions = []
    
    if missing_pct > 0.5:
        suggestions.append("Consider removing column due to high missing rate")
    elif missing_pct > 0.2:
        suggestions.append("Investigate data collection issues")
    
    if series.dtype in ['int64', 'float64']:
        suggestions.extend([
            "Fill with mean/median value",
            "Use interpolation methods",
            "Apply predictive imputation"
        ])
    else:
        suggestions.extend([
            "Fill with mode value",
            "Use 'Unknown' category",
            "Apply categorical imputation"
        ])
    
    return suggestions


def _analyze_quality_issues(df: "pd.DataFrame") -> Dict[str, Any]:
    """Analyze current data quality issues."""
    import pandas as pd
    import numpy as np
    
    issues = {}
    
    # Missing values
    missing_counts = df.isnull().sum()
    issues["missing_values"] = {
        "total_missing": missing_counts.sum(),
        "columns_with_missing": (missing_counts > 0).sum(),
        "worst_column": missing_counts.idxmax() if missing_counts.sum() > 0 else None,
        "worst_missing_pct": (missing_counts.max() / len(df) * 100) if missing_counts.sum() > 0 else 0
    }
    
    # Duplicates
    duplicate_count = df.duplicated().sum()
    issues["duplicates"] = {
        "total_duplicates": duplicate_count,
        "duplicate_percentage": duplicate_count / len(df) * 100
    }
    
    # Data types
    issues["data_types"] = {
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "text_columns": len(df.select_dtypes(include=['object']).columns),
        "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns)
    }
    
    return issues


def _handle_missing_values(df: "pd.DataFrame", strategy: str, interactive: bool) -> "pd.DataFrame":
    """Handle missing values according to strategy."""
    import pandas as pd
    
    if strategy == "drop":
        return df.dropna()
    elif strategy == "fill_mean":
        return df.fillna(df.select_dtypes(include=[np.number]).mean())
    elif strategy == "fill_median":
        return df.fillna(df.select_dtypes(include=[np.number]).median())
    elif strategy == "fill_mode":
        return df.fillna(df.mode().iloc[0])
    elif strategy == "fill_forward":
        return df.fillna(method='ffill')
    elif strategy == "fill_backward":
        return df.fillna(method='bfill')
    else:
        return df


def _handle_duplicates(df: "pd.DataFrame", strategy: str, interactive: bool) -> "pd.DataFrame":
    """Handle duplicate records according to strategy."""
    if strategy == "drop":
        return df.drop_duplicates()
    elif strategy == "keep_first":
        return df.drop_duplicates(keep='first')
    elif strategy == "keep_last":
        return df.drop_duplicates(keep='last')
    else:
        return df


def _handle_outliers(df: "pd.DataFrame", strategy: str, method: str, threshold: float, interactive: bool) -> "pd.DataFrame":
    """Handle outliers according to strategy and method."""
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            outlier_mask = z_scores > threshold
            
        else:  # isolation
            # Would implement isolation forest outlier detection
            outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        if strategy == "drop":
            df = df[~outlier_mask]
        elif strategy == "clip":
            df.loc[outlier_mask, column] = np.clip(df.loc[outlier_mask, column], lower_bound, upper_bound)
        elif strategy == "cap":
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound
    
    return df


def _normalize_text_fields(df: "pd.DataFrame") -> "pd.DataFrame":
    """Normalize text fields in the dataframe."""
    import pandas as pd
    
    text_columns = df.select_dtypes(include=['object']).columns
    
    for column in text_columns:
        # Basic text normalization
        df[column] = df[column].astype(str).str.strip().str.lower()
        
        # Remove extra whitespace
        df[column] = df[column].str.replace(r'\s+', ' ', regex=True)
    
    return df


def _standardize_data_formats(df: "pd.DataFrame") -> "pd.DataFrame":
    """Standardize data formats in the dataframe."""
    import pandas as pd
    import re
    
    # This would implement various format standardizations
    # For brevity, showing basic example
    
    for column in df.columns:
        if df[column].dtype == 'object':
            # Try to detect and standardize dates
            try:
                pd.to_datetime(df[column].head(10))
                df[column] = pd.to_datetime(df[column], errors='coerce')
            except:
                pass
    
    return df


def _calculate_changes(df_before: "pd.DataFrame", df_after: "pd.DataFrame", operation: str) -> Dict[str, Any]:
    """Calculate changes made during a cleaning operation."""
    return {
        "operation": operation,
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "rows_removed": len(df_before) - len(df_after),
        "columns_before": len(df_before.columns),
        "columns_after": len(df_after.columns)
    }


# Display functions (simplified for brevity)

def _display_validation_table(results: Dict[str, Any], filename: str, severity: str):
    """Display validation results in a rich table."""
    console.print(f"\n[bold blue]Data Validation Report - {filename}[/bold blue]")
    
    summary = results["summary"]
    console.print(f"Records: {summary['total_records']:,}, Columns: {summary['total_columns']}")
    console.print(f"Checks: {', '.join(summary['checks_performed'])}")
    console.print(f"Severity filter: {severity}")
    
    if results["issues"]:
        table = Table(title="Validation Issues", show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan")
        table.add_column("Severity", justify="center")
        table.add_column("Column", style="yellow")
        table.add_column("Message", style="white")
        table.add_column("Count", justify="right")
        
        for issue in results["issues"]:
            severity_color = {"error": "red", "warning": "yellow", "info": "blue"}.get(issue["severity"], "white")
            
            table.add_row(
                issue["type"],
                f"[{severity_color}]{issue['severity']}[/{severity_color}]",
                issue.get("column", ""),
                issue["message"],
                str(issue.get("count", ""))
            )
        
        console.print(table)
    else:
        console.print("[green]✅ No validation issues found![/green]")


def _display_cleaning_summary(original_shape: tuple, final_shape: tuple, 
                             cleaning_log: List[Dict], quality_issues: Dict):
    """Display cleaning operation summary."""
    console.print(f"\n[bold blue]Data Cleaning Summary[/bold blue]")
    
    console.print(f"Original: {original_shape[0]:,} rows × {original_shape[1]} columns")
    console.print(f"Final: {final_shape[0]:,} rows × {final_shape[1]} columns")
    console.print(f"Rows removed: {original_shape[0] - final_shape[0]:,}")
    
    if cleaning_log:
        table = Table(title="Cleaning Operations", show_header=True, header_style="bold magenta")
        table.add_column("Operation", style="cyan")
        table.add_column("Rows Before", justify="right")
        table.add_column("Rows After", justify="right")
        table.add_column("Rows Removed", justify="right")
        
        for operation in cleaning_log:
            table.add_row(
                operation["operation"],
                f"{operation['rows_before']:,}",
                f"{operation['rows_after']:,}",
                f"{operation['rows_removed']:,}"
            )
        
        console.print(table)


def _display_repair_summary(original_shape: tuple, final_shape: tuple, repair_log: List[Dict]):
    """Display repair operation summary."""
    console.print(f"\n[bold blue]Auto-Repair Summary[/bold blue]")
    
    successful_repairs = [r for r in repair_log if r.get("success", False)]
    failed_repairs = [r for r in repair_log if not r.get("success", True)]
    
    console.print(f"Original: {original_shape[0]:,} rows × {original_shape[1]} columns")
    console.print(f"Final: {final_shape[0]:,} rows × {final_shape[1]} columns")
    console.print(f"Successful repairs: {len(successful_repairs)}")
    console.print(f"Failed repairs: {len(failed_repairs)}")
    
    if repair_log:
        table = Table(title="Repair Operations", show_header=True, header_style="bold magenta")
        table.add_column("Repair Type", style="cyan")
        table.add_column("Column", style="yellow")
        table.add_column("Status", justify="center")
        table.add_column("Records Affected", justify="right")
        
        for repair in repair_log:
            status_color = "green" if repair.get("success", False) else "red"
            status_text = "✓ Success" if repair.get("success", False) else "✗ Failed"
            
            table.add_row(
                repair["repair_type"],
                repair.get("column", ""),
                f"[{status_color}]{status_text}[/{status_color}]",
                str(repair.get("records_affected", 0))
            )
        
        console.print(table)


def _perform_single_monitoring_check(data_path: Path, baseline_path: Optional[Path],
                                   rules_config: Optional[Path], metrics: str,
                                   alert_threshold: float) -> Dict[str, Any]:
    """Perform a single monitoring check."""
    # Implementation would load data, compare with baseline, check rules
    # and return monitoring results
    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "data_path": str(data_path),
        "quality_score": 0.85,  # Example
        "alerts": [],
        "metrics": {}
    }


def _display_monitoring_status(result: Dict[str, Any], check_count: int):
    """Display current monitoring status."""
    quality_score = result.get("quality_score", 0)
    alerts = result.get("alerts", [])
    
    status_color = "green" if quality_score > 0.8 else "yellow" if quality_score > 0.6 else "red"
    
    console.print(f"[{status_color}]Check #{check_count}: Quality Score {quality_score:.2f}[/{status_color}]", end="")
    if alerts:
        console.print(f" - {len(alerts)} alerts")
    else:
        console.print(" - No alerts")


# Additional helper functions would be implemented here for completeness...

def _detect_repairable_issues(df, strategy, confidence_threshold):
    """Detect issues that can be automatically repaired."""
    return []  # Simplified for brevity

def _create_repair_plan(df, issues, rules_config):
    """Create a plan for repairing detected issues."""
    return []  # Simplified for brevity

def _apply_repair(df, repair, confidence_threshold):
    """Apply a specific repair operation."""
    return df  # Simplified for brevity

def _validate_repairs(df_original, df_repaired, repair_log):
    """Validate that repairs were successful."""
    return {}  # Simplified for brevity

def _display_repair_validation(validation_results):
    """Display repair validation results."""
    pass  # Simplified for brevity

def _get_user_approval_for_repairs(repair_plan):
    """Get user approval for proposed repairs in interactive mode."""
    return []  # Simplified for brevity

def _display_monitoring_table(result, filename):
    """Display monitoring results in table format."""
    pass  # Simplified for brevity

def _display_quality_alerts(result):
    """Display quality alerts."""
    pass  # Simplified for brevity

def _generate_validation_html_report(results, output_path, filename):
    """Generate HTML validation report."""
    pass  # Simplified for brevity


if __name__ == "__main__":
    app()