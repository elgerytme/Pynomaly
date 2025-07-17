"""Data Quality CLI Commands for validation, monitoring, and quality management."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

console = Console()

# Create the quality CLI app
quality_app = typer.Typer(
    name="quality",
    help="Data quality operations including validation, monitoring, and quality management",
    rich_markup_mode="rich"
)


@quality_app.command("validate")
def validate_dataset(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    rules_file: Optional[Path] = typer.Option(
        None, "--rules", "-r", help="Validation rules file (JSON/YAML)"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for validation results"
    ),
    validation_type: str = typer.Option(
        "comprehensive", "--type", "-t",
        help="Validation type: [comprehensive|schema|quality|business|custom]"
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Output format: [json|html|csv|yaml]"
    ),
    fail_fast: bool = typer.Option(
        False, "--fail-fast", help="Stop validation on first failure"
    ),
    severity_threshold: str = typer.Option(
        "warning", "--threshold", help="Minimum severity to report: [info|warning|error|critical]"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--sequential", help="Use parallel validation"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Validate data_collection against quality rules and constraints."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_validation.{format}"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import quality packages
            from packages.data_quality.application.services.quality_validation_service import QualityValidationService
            from packages.data_quality.application.services.rule_engine_service import RuleEngineService
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            progress.update(task, description="Loading validation rules...")
            
            # Initialize services
            validation_service = QualityValidationService()
            rule_engine = RuleEngineService()
            
            # Load rules
            if rules_file and rules_file.exists():
                with open(rules_file, 'r') as f:
                    if rules_file.suffix == '.yaml':
                        import yaml
                        rules = yaml.safe_load(f)
                    else:
                        rules = json.load(f)
            else:
                # Generate default rules based on validation type
                rules = validation_service.generate_default_rules(data_collection, validation_type)
            
            progress.update(task, description="Validating data_collection...")
            
            # Configure validation
            config = {
                "fail_fast": fail_fast,
                "severity_threshold": severity_threshold,
                "parallel_processing": parallel,
                "validation_type": validation_type
            }
            
            # Perform validation
            validation_results = validation_service.validate_data_collection(
                data_collection, rules, config
            )
            
            progress.update(task, description="Generating validation report...")
            
            # Save results
            if format == "json":
                with open(output_file, 'w') as f:
                    json.dump(validation_results, f, indent=2, default=str)
            elif format == "yaml":
                import yaml
                with open(output_file, 'w') as f:
                    yaml.dump(validation_results, f, default_flow_style=False)
            elif format == "html":
                validation_service.generate_html_report(validation_results, output_file)
            elif format == "csv":
                validation_service.export_to_csv(validation_results, output_file)
            
            progress.update(task, description="Validation complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required quality packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during validation: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    # Display results summary
    console.print("\n[green]✓ DataCollection validation completed successfully![/green]")
    console.print(f"Results saved to: {output_file}")
    
    # Display validation summary
    if validation_results:
        total_rules = validation_results.get("total_rules", 0)
        passed_rules = validation_results.get("passed_rules", 0)
        failed_rules = validation_results.get("failed_rules", 0)
        warnings = validation_results.get("warnings", 0)
        
        summary_text = f"""
[bold]Validation Summary[/bold]
• Total Rules: {total_rules}
• Passed: [green]{passed_rules}[/green]
• Failed: [red]{failed_rules}[/red]
• Warnings: [yellow]{warnings}[/yellow]
• Success Rate: {(passed_rules/total_rules*100) if total_rules > 0 else 0:.1f}%
        """
        
        console.print(Panel(summary_text, title="Validation Results", border_style="blue"))


@quality_app.command("monitor")
def monitor_quality(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    baseline_file: Optional[Path] = typer.Option(
        None, "--baseline", "-b", help="Baseline quality profile for comparison"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for monitoring results"
    ),
    thresholds_file: Optional[Path] = typer.Option(
        None, "--thresholds", "-t", help="Quality thresholds configuration file"
    ),
    alert_on_drift: bool = typer.Option(
        True, "--alert-drift/--no-alert-drift", help="Alert on quality drift"
    ),
    drift_threshold: float = typer.Option(
        0.1, "--drift-threshold", help="Threshold for detecting quality drift"
    ),
    generate_alerts: bool = typer.Option(
        True, "--alerts/--no-alerts", help="Generate quality alerts"
    ),
    continuous: bool = typer.Option(
        False, "--continuous", help="Run continuous monitoring"
    ),
    interval: int = typer.Option(
        300, "--interval", help="Monitoring interval in seconds (for continuous mode)"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Monitor data quality and detect quality drift."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_monitoring"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing quality monitoring...", total=None)
        
        try:
            # Import monitoring packages
            from packages.data_quality.application.services.quality_monitoring_service import QualityMonitoringService
            from packages.data_quality.application.services.drift_detection_service import DriftDetectionService
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            progress.update(task, description="Loading baseline and thresholds...")
            
            # Initialize services
            monitoring_service = QualityMonitoringService()
            drift_service = DriftDetectionService()
            
            # Load baseline if provided
            baseline = None
            if baseline_file and baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline = json.load(f)
            
            # Load thresholds if provided
            thresholds = None
            if thresholds_file and thresholds_file.exists():
                with open(thresholds_file, 'r') as f:
                    thresholds = json.load(f)
            
            if continuous:
                console.print(f"[blue]Starting continuous monitoring (interval: {interval}s)[/blue]")
                console.print("[yellow]Press Ctrl+C to stop[/yellow]")
                
                try:
                    import time
                    iteration = 0
                    while True:
                        iteration += 1
                        progress.update(task, description=f"Monitoring iteration {iteration}...")
                        
                        # Perform monitoring
                        monitoring_results = monitoring_service.monitor_data_collection_quality(
                            data_collection, baseline=baseline, thresholds=thresholds
                        )
                        
                        # Check for drift
                        if baseline and alert_on_drift:
                            drift_results = drift_service.detect_quality_drift(
                                data_collection, baseline, threshold=drift_threshold
                            )
                            monitoring_results["drift_processing"] = drift_results
                        
                        # Save iteration results
                        timestamp = str(int(time.time()))
                        iteration_file = output_dir / f"monitoring_{timestamp}.json"
                        with open(iteration_file, 'w') as f:
                            json.dump(monitoring_results, f, indent=2, default=str)
                        
                        # Display alerts
                        if generate_alerts and monitoring_results.get("alerts"):
                            for alert in monitoring_results["alerts"]:
                                severity = alert.get("severity", "info")
                                message = alert.get("message", "")
                                console.print(f"[red]ALERT[/red] [{severity}]: {message}")
                        
                        time.sleep(interval)
                        
                except KeyboardInterrupt:
                    console.print("\n[yellow]Monitoring stopped by user[/yellow]")
            else:
                progress.update(task, description="Performing quality monitoring...")
                
                # Single monitoring run
                monitoring_results = monitoring_service.monitor_data_collection_quality(
                    data_collection, baseline=baseline, thresholds=thresholds
                )
                
                # Check for drift
                if baseline and alert_on_drift:
                    drift_results = drift_service.detect_quality_drift(
                        data_collection, baseline, threshold=drift_threshold
                    )
                    monitoring_results["drift_processing"] = drift_results
                
                # Save results
                results_file = output_dir / "monitoring_results.json"
                with open(results_file, 'w') as f:
                    json.dump(monitoring_results, f, indent=2, default=str)
                
                # Generate monitoring report
                report_file = output_dir / "monitoring_report.html"
                monitoring_service.generate_monitoring_report(monitoring_results, report_file)
            
            progress.update(task, description="Monitoring complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during monitoring: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    if not continuous:
        console.print("\n[green]✓ Quality monitoring completed successfully![/green]")
        console.print(f"Results saved to: {output_dir}")
        
        # Display monitoring summary
        if monitoring_results:
            quality_score = monitoring_results.get("overall_quality_score", 0)
            issues_found = len(monitoring_results.get("quality_issues", []))
            alerts_generated = len(monitoring_results.get("alerts", []))
            
            console.print(f"[blue]Overall Quality Score: {quality_score:.2f}[/blue]")
            console.print(f"[yellow]Issues Found: {issues_found}[/yellow]")
            console.print(f"[red]Alerts Generated: {alerts_generated}[/red]")


@quality_app.command("cleanse")
def cleanse_dataset(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for cleansed data_collection"
    ),
    cleansing_rules: Optional[Path] = typer.Option(
        None, "--rules", "-r", help="Cleansing rules configuration file"
    ),
    cleansing_type: str = typer.Option(
        "comprehensive", "--type", "-t",
        help="Cleansing type: [comprehensive|basic|custom|ml_guided]"
    ),
    auto_fix: bool = typer.Option(
        False, "--auto-fix", help="Automatically fix detected issues"
    ),
    backup_original: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup of original data_collection"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be cleansed without making changes"
    ),
    confidence_threshold: float = typer.Option(
        0.8, "--confidence", help="Confidence threshold for automatic fixes"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Cleanse data_collection by fixing data quality issues."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_cleansed{input_file.suffix}"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import cleansing packages
            from packages.data_quality.application.services.data_cleansing_service import DataCleansingService
            from packages.data_quality.application.services.ml_cleansing_service import MLCleansingService
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            if backup_original and not dry_run:
                backup_file = input_file.parent / f"{input_file.stem}_backup{input_file.suffix}"
                data_collection.to_csv(backup_file, index=False)
                console.print(f"[blue]Backup created: {backup_file}[/blue]")
            
            progress.update(task, description="Loading cleansing rules...")
            
            # Initialize services
            if cleansing_type == "ml_guided":
                cleansing_service = MLCleansingService()
            else:
                cleansing_service = DataCleansingService()
            
            # Load cleansing rules
            rules = None
            if cleansing_rules and cleansing_rules.exists():
                with open(cleansing_rules, 'r') as f:
                    rules = json.load(f)
            
            progress.update(task, description="Analyzing data quality issues...")
            
            # Analyze issues first
            quality_issues = cleansing_service.analyze_quality_issues(data_collection)
            
            if dry_run:
                console.print("\n[yellow]DRY RUN - Issues that would be fixed:[/yellow]")
                for issue in quality_issues:
                    console.print(f"  • {issue.get('type', 'Unknown')}: {issue.get('description', '')}")
                console.print(f"\nTotal issues found: {len(quality_issues)}")
                return
            
            progress.update(task, description="Cleansing data_collection...")
            
            # Configure cleansing
            config = {
                "cleansing_type": cleansing_type,
                "auto_fix": auto_fix,
                "confidence_threshold": confidence_threshold,
                "rules": rules
            }
            
            # Perform cleansing
            cleansed_data_collection, cleansing_report = cleansing_service.cleanse_data_collection(
                data_collection, config
            )
            
            progress.update(task, description="Saving cleansed data_collection...")
            
            # Save cleansed data_collection
            cleansed_data_collection.to_csv(output_file, index=False)
            
            # Save cleansing report
            report_file = output_file.parent / f"{output_file.stem}_cleansing_report.json"
            with open(report_file, 'w') as f:
                json.dump(cleansing_report, f, indent=2, default=str)
            
            progress.update(task, description="Cleansing complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during cleansing: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ DataCollection cleansing completed successfully![/green]")
    console.print(f"Cleansed data_collection saved to: {output_file}")
    console.print(f"Cleansing report saved to: {report_file}")
    
    # Display cleansing summary
    if cleansing_report:
        issues_fixed = cleansing_report.get("issues_fixed", 0)
        issues_detected = cleansing_report.get("issues_detected", 0)
        rows_affected = cleansing_report.get("rows_affected", 0)
        
        summary_text = f"""
[bold]Cleansing Summary[/bold]
• Issues Detected: {issues_detected}
• Issues Fixed: [green]{issues_fixed}[/green]
• Rows Affected: {rows_affected:,}
• Success Rate: {(issues_fixed/issues_detected*100) if issues_detected > 0 else 0:.1f}%
        """
        
        console.print(Panel(summary_text, title="Cleansing Results", border_style="green"))


@quality_app.command("score")
def calculate_quality_score(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for quality score results"
    ),
    scoring_processor: str = typer.Option(
        "comprehensive", "--processor", "-m",
        help="Scoring processor: [comprehensive|weighted|ml_based|business_focused]"
    ),
    weights_file: Optional[Path] = typer.Option(
        None, "--weights", "-w", help="Custom weights configuration file"
    ),
    include_breakdown: bool = typer.Option(
        True, "--breakdown/--no-breakdown", help="Include detailed score breakdown"
    ),
    include_recommendations: bool = typer.Option(
        True, "--recommendations/--no-recommendations", help="Include improvement recommendations"
    ),
    benchmark_mode: bool = typer.Option(
        False, "--benchmark", help="Run in benchmark mode for performance comparison"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Calculate comprehensive data quality score."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_quality_score.json"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import scoring packages
            from packages.data_quality.application.services.quality_scoring_service import QualityScoringService
            from packages.data_quality.application.services.ml_quality_assessment_service import MLQualityAssessmentService
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            progress.update(task, description="Initializing scoring processor...")
            
            # Initialize services
            if scoring_processor == "ml_based":
                scoring_service = MLQualityAssessmentService()
            else:
                scoring_service = QualityScoringService()
            
            # Load custom weights if provided
            weights = None
            if weights_file and weights_file.exists():
                with open(weights_file, 'r') as f:
                    weights = json.load(f)
            
            progress.update(task, description="Calculating quality scores...")
            
            # Configure scoring
            config = {
                "scoring_processor": scoring_processor,
                "weights": weights,
                "include_breakdown": include_breakdown,
                "include_recommendations": include_recommendations,
                "benchmark_mode": benchmark_mode
            }
            
            # Calculate quality score
            scoring_results = scoring_service.calculate_quality_score(data_collection, config)
            
            progress.update(task, description="Generating quality report...")
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(scoring_results, f, indent=2, default=str)
            
            # Generate detailed report if requested
            if include_breakdown:
                report_file = output_file.parent / f"{output_file.stem}_detailed_report.html"
                scoring_service.generate_quality_report(scoring_results, data_collection, report_file)
            
            progress.update(task, description="Scoring complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during scoring: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ Quality scoring completed successfully![/green]")
    console.print(f"Results saved to: {output_file}")
    
    # Display quality score
    if scoring_results:
        overall_score = scoring_results.get("overall_score", 0)
        grade = scoring_results.get("quality_grade", "Unknown")
        
        # Determine color based on score
        if overall_score >= 90:
            score_color = "green"
        elif overall_score >= 75:
            score_color = "yellow"
        elif overall_score >= 60:
            score_color = "orange"
        else:
            score_color = "red"
        
        console.print(f"\n[{score_color}]Overall Quality Score: {overall_score:.1f}/100 (Grade: {grade})[/{score_color}]")
        
        # Display score breakdown if available
        if include_breakdown and "score_breakdown" in scoring_results:
            table = Table(title="Quality Score Breakdown")
            table.add_column("Dimension", style="cyan")
            table.add_column("Score", style="green")
            table.add_column("Weight", style="blue")
            table.add_column("Contribution", style="yellow")
            
            for dimension, details in scoring_results["score_breakdown"].items():
                table.add_row(
                    dimension.title(),
                    f"{details.get('score', 0):.1f}",
                    f"{details.get('weight', 0):.2f}",
                    f"{details.get('contribution', 0):.1f}"
                )
            
            console.print(table)


@quality_app.command("rules")
def manage_quality_rules(
    action: str = typer.Argument(..., help="Action: [create|edit|validate|export|import]"),
    rules_file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Rules file path"
    ),
    rule_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Rule type: [schema|business|statistical|custom]"
    ),
    template: Optional[str] = typer.Option(
        None, "--template", help="Rule template to use"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive rule creation"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Manage data quality rules."""
    
    if action not in ["create", "edit", "validate", "export", "import"]:
        console.print(f"[red]Invalid action: {action}[/red]")
        console.print("Valid actions: create, edit, validate, export, import")
        raise typer.Exit(1)
    
    try:
        # Import rule management packages
        from packages.data_quality.application.services.rule_management_service import RuleManagementService
        
        rule_service = RuleManagementService()
        
        if action == "create":
            if interactive:
                # Interactive rule creation
                console.print("[blue]Interactive Quality Rule Creation[/blue]")
                
                rule_name = Prompt.ask("Rule name")
                rule_description = Prompt.ask("Rule description")
                rule_type_input = Prompt.ask(
                    "Rule type", 
                    choices=["schema", "business", "statistical", "custom"],
                    default="business"
                )
                
                # Create rule interactively
                rule = rule_service.create_rule_interactive(
                    rule_name, rule_description, rule_type_input
                )
                
                if rules_file:
                    rule_service.save_rule(rule, rules_file)
                    console.print(f"[green]Rule saved to {rules_file}[/green]")
                else:
                    console.print(json.dumps(rule, indent=2))
            else:
                # Template-based rule creation
                if not template:
                    console.print("[red]Template required for non-interactive rule creation[/red]")
                    raise typer.Exit(1)
                
                rule = rule_service.create_rule_from_template(template, rule_type)
                
                if rules_file:
                    rule_service.save_rule(rule, rules_file)
                    console.print(f"[green]Rule created and saved to {rules_file}[/green]")
                else:
                    console.print(json.dumps(rule, indent=2))
        
        elif action == "validate":
            if not rules_file or not rules_file.exists():
                console.print("[red]Rules file required for validation[/red]")
                raise typer.Exit(1)
            
            # Validate rules file
            validation_results = rule_service.validate_rules_file(rules_file)
            
            if validation_results["valid"]:
                console.print("[green]✓ Rules file is valid[/green]")
            else:
                console.print("[red]✗ Rules file has errors:[/red]")
                for error in validation_results["errors"]:
                    console.print(f"  • {error}")
        
        elif action == "export":
            # Export rules to different format
            if not rules_file:
                console.print("[red]Rules file required for export[/red]")
                raise typer.Exit(1)
            
            export_format = Prompt.ask(
                "Export format", 
                choices=["json", "yaml", "sql", "python"],
                default="json"
            )
            
            exported_rules = rule_service.export_rules(rules_file, export_format)
            
            export_file = rules_file.parent / f"{rules_file.stem}_exported.{export_format}"
            with open(export_file, 'w') as f:
                f.write(exported_rules)
            
            console.print(f"[green]Rules exported to {export_file}[/green]")
        
        elif action == "import":
            # Import rules from external source
            source_file = Prompt.ask("Source file path")
            source_format = Prompt.ask(
                "Source format",
                choices=["json", "yaml", "csv", "sql"],
                default="json"
            )
            
            imported_rules = rule_service.import_rules(source_file, source_format)
            
            if rules_file:
                rule_service.save_rule(imported_rules, rules_file)
                console.print(f"[green]Rules imported and saved to {rules_file}[/green]")
            else:
                console.print(json.dumps(imported_rules, indent=2))
        
        elif action == "edit":
            # Edit existing rules file
            if not rules_file or not rules_file.exists():
                console.print("[red]Rules file required for editing[/red]")
                raise typer.Exit(1)
            
            console.print(f"[blue]Opening {rules_file} for editing...[/blue]")
            # This would open an editor - simplified for CLI
            console.print("Edit functionality would open your default editor")
    
    except ImportError as e:
        console.print(f"[red]Error: Required packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error managing rules: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@quality_app.command("alerts")
def manage_quality_alerts(
    action: str = typer.Argument(..., help="Action: [list|create|configure|test]"),
    alert_config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Alert configuration file"
    ),
    alert_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Alert type: [threshold|drift|anomaly|custom]"
    ),
    test_data_collection: Optional[Path] = typer.Option(
        None, "--test-data", help="Test data_collection for alert testing"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: [table|json|csv]"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Manage data quality alerts and notifications."""
    
    try:
        # Import alerting packages
        from packages.data_quality.application.services.alerting_service import AlertingService
        
        alerting_service = AlertingService()
        
        if action == "list":
            # List configured alerts
            alerts = alerting_service.list_configured_alerts()
            
            if format == "table":
                table = Table(title="Configured Quality Alerts")
                table.add_column("Alert ID", style="cyan")
                table.add_column("Type", style="green")
                table.add_column("Status", style="yellow")
                table.add_column("Last Triggered", style="blue")
                
                for alert in alerts:
                    table.add_row(
                        alert.get("id", ""),
                        alert.get("type", ""),
                        alert.get("status", ""),
                        alert.get("last_triggered", "Never")
                    )
                
                console.print(table)
            else:
                console.print(json.dumps(alerts, indent=2, default=str))
        
        elif action == "create":
            # Create new alert
            console.print("[blue]Creating new quality alert...[/blue]")
            
            alert_name = Prompt.ask("Alert name")
            alert_type_input = Prompt.ask(
                "Alert type",
                choices=["threshold", "drift", "anomaly", "custom"],
                default="threshold"
            )
            
            alert_config_dict = alerting_service.create_alert_interactive(
                alert_name, alert_type_input
            )
            
            if alert_config:
                with open(alert_config, 'w') as f:
                    json.dump(alert_config_dict, f, indent=2)
                console.print(f"[green]Alert configuration saved to {alert_config}[/green]")
            else:
                console.print(json.dumps(alert_config_dict, indent=2))
        
        elif action == "configure":
            # Configure alert settings
            if not alert_config or not alert_config.exists():
                console.print("[red]Alert configuration file required[/red]")
                raise typer.Exit(1)
            
            console.print(f"[blue]Configuring alerts from {alert_config}[/blue]")
            
            # Load and apply alert configuration
            alerting_service.configure_alerts(alert_config)
            console.print("[green]Alert configuration applied successfully[/green]")
        
        elif action == "test":
            # Test alert configuration
            if not alert_config or not alert_config.exists():
                console.print("[red]Alert configuration file required for testing[/red]")
                raise typer.Exit(1)
            
            if test_data_collection and test_data_collection.exists():
                # Test with actual data_collection
                from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
                
                adapter = DataSourceAdapter()
                data_collection = adapter.load_data_collection(str(test_data_collection))
                
                test_results = alerting_service.test_alerts(alert_config, data_collection)
            else:
                # Test with synthetic data
                test_results = alerting_service.test_alerts(alert_config)
            
            console.print("[green]Alert testing completed:[/green]")
            for result in test_results:
                status = "✓" if result["triggered"] else "✗"
                console.print(f"  {status} {result['alert_name']}: {result['message']}")
    
    except ImportError as e:
        console.print(f"[red]Error: Required packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error managing alerts: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


if __name__ == "__main__":
    quality_app()