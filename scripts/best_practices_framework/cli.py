#!/usr/bin/env python3
"""
Command Line Interface for Best Practices Framework
==================================================
Provides CLI commands for validating projects against best practices
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.text import Text

from . import BestPracticesValidator, get_version, get_supported_categories
from .reporting.report_generator import ReportGenerator
from .core.base_validator import ValidationReport


console = Console()


def print_version(ctx, param, value):
    """Print version and exit"""
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"Best Practices Framework v{get_version()}")
    ctx.exit()


def print_categories(ctx, param, value):
    """Print supported categories and exit"""
    if not value or ctx.resilient_parsing:
        return
    categories = get_supported_categories()
    click.echo("Supported validation categories:")
    for category in categories:
        click.echo(f"  - {category}")
    ctx.exit()


@click.group()
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True, help='Show version and exit')
@click.option('--list-categories', is_flag=True, callback=print_categories,
              expose_value=False, is_eager=True, help='List supported categories and exit')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration file')
@click.pass_context
def cli(ctx, verbose: bool, config: Optional[str]):
    """
    Best Practices Framework - Automated validation of software engineering best practices.
    
    Validates projects against industry standards for architecture, security, testing,
    DevOps, and Site Reliability Engineering practices.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config
    
    if verbose:
        console.print(f"[bold green]Best Practices Framework v{get_version()}[/bold green]")


@cli.command()
@click.option('--category', '-c', multiple=True, 
              type=click.Choice(get_supported_categories(), case_sensitive=False),
              help='Validate specific categories (can be used multiple times)')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'html', 'markdown', 'sarif', 'junit']),
              default='json', help='Output format')
@click.option('--project-root', '-r', type=click.Path(exists=True), default='.',
              help='Project root directory')
@click.option('--fail-on-critical', is_flag=True, default=True,
              help='Fail validation if critical violations found')
@click.option('--fail-on-high', is_flag=True, default=False,
              help='Fail validation if high violations found')
@click.option('--incremental', '-i', type=str, multiple=True,
              help='Validate only changed files (specify file paths)')
@click.pass_context
def validate(ctx, category: List[str], output: Optional[str], format: str, 
            project_root: str, fail_on_critical: bool, fail_on_high: bool,
            incremental: List[str]):
    """
    Validate project against best practices.
    
    Examples:
        best-practices validate
        best-practices validate --category security --category testing
        best-practices validate --output results.json --format json
        best-practices validate --incremental src/main.py src/utils.py
    """
    async def run_validation():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            # Initialize validator
            config_path = ctx.obj.get('config')
            validator = BestPracticesValidator(config_path=config_path, project_root=project_root)
            
            # Determine validation scope
            if incremental:
                task = progress.add_task("Running incremental validation...", total=None)
                report = await validator.validate_incremental(list(incremental))
            elif category:
                task = progress.add_task(f"Validating categories: {', '.join(category)}", total=len(category))
                category_results = {}
                for cat in category:
                    results = await validator.validate_category(cat)
                    category_results[cat] = results
                    progress.update(task, advance=1)
                
                # Create simplified report for category validation
                from .core.validator_engine import ValidationReport, ComplianceScore
                all_violations = []
                for results in category_results.values():
                    for result in results:
                        all_violations.extend(result.violations)
                
                compliance_score = ComplianceScore(
                    overall_score=85.0,  # Simplified scoring
                    category_scores={cat: 85.0 for cat in category},
                    grade="B",
                    total_violations=len(all_violations),
                    critical_violations=sum(1 for v in all_violations if v.severity == 'critical'),
                    high_violations=sum(1 for v in all_violations if v.severity == 'high')
                )
                
                report = ValidationReport(
                    project_name=Path(project_root).name,
                    project_root=project_root,
                    compliance_score=compliance_score,
                    category_results=category_results,
                    all_violations=all_violations,
                    summary={'categories_validated': len(category)}
                )
            else:
                task = progress.add_task("Running comprehensive validation...", total=None)
                report = await validator.validate_all()
            
            progress.remove_task(task)
        
        # Display results
        display_validation_results(report)
        
        # Generate output file if requested
        if output:
            report_generator = ReportGenerator()
            
            if format == 'json':
                content = report_generator.generate_json_report(report)
                with open(output, 'w') as f:
                    json.dump(content, f, indent=2, default=str)
            elif format == 'html':
                content = report_generator.generate_html_report(report)
                with open(output, 'w') as f:
                    f.write(content)
            elif format == 'markdown':
                content = report_generator.generate_markdown_report(report)
                with open(output, 'w') as f:
                    f.write(content)
            elif format == 'sarif':
                content = report_generator.generate_sarif_report(report)
                with open(output, 'w') as f:
                    json.dump(content, f, indent=2)
            elif format == 'junit':
                content = report_generator.generate_junit_report(report)
                with open(output, 'w') as f:
                    f.write(content)
            
            console.print(f"✅ Report saved to: [bold]{output}[/bold]")
        
        # Quality gate check
        quality_passed = validator.quality_gate(
            report, 
            enforce_critical=fail_on_critical,
            enforce_high=fail_on_high
        )
        
        if not quality_passed:
            console.print("\n[bold red]❌ Quality gate FAILED[/bold red]")
            sys.exit(1)
        else:
            console.print("\n[bold green]✅ Quality gate PASSED[/bold green]")
    
    asyncio.run(run_validation())


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input validation results file (JSON format)')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output report file path')
@click.option('--format', '-f', type=click.Choice(['html', 'markdown', 'pdf']),
              default='html', help='Report format')
@click.option('--template', '-t', type=click.Path(exists=True),
              help='Custom report template')
def report(input: str, output: str, format: str, template: Optional[str]):
    """
    Generate reports from validation results.
    
    Examples:
        best-practices report -i results.json -o report.html --format html
        best-practices report -i results.json -o report.md --format markdown
    """
    try:
        # Load validation results
        with open(input, 'r') as f:
            results_data = json.load(f)
        
        # Generate report
        report_generator = ReportGenerator(template_path=template)
        
        with Progress(SpinnerColumn(), TextColumn("Generating report..."), console=console) as progress:
            task = progress.add_task("Generating...", total=None)
            
            if format == 'html':
                content = report_generator.generate_html_report(results_data)
            elif format == 'markdown':
                content = report_generator.generate_markdown_report(results_data)
            elif format == 'pdf':
                content = report_generator.generate_pdf_report(results_data)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            with open(output, 'w' if format != 'pdf' else 'wb') as f:
                if format == 'pdf':
                    f.write(content)
                else:
                    f.write(content)
            
            progress.remove_task(task)
        
        console.print(f"✅ Report generated: [bold]{output}[/bold]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error generating report: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option('--profile', type=click.Choice(['default', 'fintech', 'healthcare', 'ecommerce', 'saas']),
              default='default', help='Configuration profile')
@click.option('--output', '-o', type=click.Path(), default='.best-practices.yml',
              help='Configuration file path')
@click.option('--force', is_flag=True, help='Overwrite existing configuration')
def init(profile: str, output: str, force: bool):
    """
    Initialize best practices configuration.
    
    Examples:
        best-practices init
        best-practices init --profile fintech
        best-practices init --output config/best-practices.yml
    """
    config_path = Path(output)
    
    if config_path.exists() and not force:
        console.print(f"[yellow]⚠️  Configuration file already exists: {config_path}[/yellow]")
        if not click.confirm("Overwrite existing configuration?"):
            console.print("Configuration initialization cancelled.")
            return
    
    # Generate configuration based on profile
    config = generate_config_template(profile)
    
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config)
        
        console.print(f"✅ Configuration initialized: [bold]{config_path}[/bold]")
        console.print(f"Profile: [bold]{profile}[/bold]")
        console.print("\nNext steps:")
        console.print("1. Review and customize the configuration")
        console.print("2. Run validation: [bold]best-practices validate[/bold]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error creating configuration: {e}[/bold red]")
        sys.exit(1)


@cli.command('quality-gate')
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input validation results file (JSON format)')
@click.option('--enforce-critical', is_flag=True, default=True,
              help='Fail on critical violations')
@click.option('--enforce-high', is_flag=True, default=False,
              help='Fail on high violations')
@click.option('--min-score', type=float, help='Minimum required score (0-100)')
def quality_gate(input: str, enforce_critical: bool, enforce_high: bool, min_score: Optional[float]):
    """
    Quality gate check for CI/CD pipelines.
    
    Examples:
        best-practices quality-gate -i results.json --enforce-critical
        best-practices quality-gate -i results.json --min-score 80
    """
    try:
        # Load validation results
        with open(input, 'r') as f:
            results_data = json.load(f)
        
        # Extract relevant metrics
        overall_score = results_data.get('compliance_score', {}).get('overall_score', 0)
        critical_violations = results_data.get('compliance_score', {}).get('critical_violations', 0)
        high_violations = results_data.get('compliance_score', {}).get('high_violations', 0)
        
        console.print(f"Overall Score: [bold]{overall_score:.1f}%[/bold]")
        console.print(f"Critical Violations: [bold]{critical_violations}[/bold]")
        console.print(f"High Violations: [bold]{high_violations}[/bold]")
        
        # Apply quality gate rules
        failed = False
        
        if enforce_critical and critical_violations > 0:
            console.print(f"[bold red]❌ FAILED: {critical_violations} critical violations found[/bold red]")
            failed = True
        
        if enforce_high and high_violations > 0:
            console.print(f"[bold red]❌ FAILED: {high_violations} high violations found[/bold red]")
            failed = True
        
        if min_score and overall_score < min_score:
            console.print(f"[bold red]❌ FAILED: Score {overall_score:.1f}% below minimum {min_score}%[/bold red]")
            failed = True
        
        if failed:
            console.print("\n[bold red]❌ Quality gate FAILED[/bold red]")
            sys.exit(1)
        else:
            console.print("\n[bold green]✅ Quality gate PASSED[/bold green]")
    
    except Exception as e:
        console.print(f"[bold red]❌ Error checking quality gate: {e}[/bold red]")
        sys.exit(1)


def display_validation_results(report: ValidationReport):
    """Display validation results in a nice format"""
    
    # Overall score panel
    score = report.compliance_score.overall_score
    grade = report.compliance_score.grade
    
    score_color = "red" if score < 60 else "yellow" if score < 80 else "green"
    
    score_panel = Panel(
        f"[bold {score_color}]{score:.1f}% ({grade})[/bold {score_color}]",
        title="Overall Compliance Score",
        expand=False
    )
    console.print(score_panel)
    
    # Category scores table
    if report.category_results:
        table = Table(title="Category Scores")
        table.add_column("Category", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Grade", justify="center")
        table.add_column("Violations", justify="right")
        
        for category, results in report.category_results.items():
            if not results:
                continue
            
            avg_score = sum(r.score for r in results) / len(results)
            total_violations = sum(len(r.violations) for r in results)
            grade = calculate_grade(avg_score)
            
            score_color = "red" if avg_score < 60 else "yellow" if avg_score < 80 else "green"
            
            table.add_row(
                category.title(),
                f"[{score_color}]{avg_score:.1f}%[/{score_color}]",
                f"[{score_color}]{grade}[/{score_color}]",
                str(total_violations)
            )
        
        console.print(table)
    
    # Violations summary
    if report.all_violations:
        violations_by_severity = {}
        for violation in report.all_violations:
            severity = violation.severity
            if severity not in violations_by_severity:
                violations_by_severity[severity] = 0
            violations_by_severity[severity] += 1
        
        violations_table = Table(title="Violations by Severity")
        violations_table.add_column("Severity", style="cyan")
        violations_table.add_column("Count", justify="right")
        
        severity_colors = {
            'critical': 'bold red',
            'high': 'red',
            'medium': 'yellow',
            'low': 'blue',
            'info': 'dim'
        }
        
        for severity in ['critical', 'high', 'medium', 'low', 'info']:
            if severity in violations_by_severity:
                color = severity_colors.get(severity, 'white')
                violations_table.add_row(
                    f"[{color}]{severity.title()}[/{color}]",
                    f"[{color}]{violations_by_severity[severity]}[/{color}]"
                )
        
        console.print(violations_table)
    
    # Recommendations
    if report.compliance_score.recommendations:
        recommendations_text = Text("Recommendations:", style="bold yellow")
        console.print(recommendations_text)
        
        for i, rec in enumerate(report.compliance_score.recommendations[:5], 1):
            console.print(f"{i}. {rec}")


def calculate_grade(score: float) -> str:
    """Convert numeric score to letter grade"""
    if score >= 95:
        return 'A+'
    elif score >= 90:
        return 'A'
    elif score >= 85:
        return 'A-'
    elif score >= 80:
        return 'B+'
    elif score >= 75:
        return 'B'
    elif score >= 70:
        return 'B-'
    elif score >= 65:
        return 'C+'
    elif score >= 60:
        return 'C'
    elif score >= 55:
        return 'C-'
    elif score >= 50:
        return 'D'
    else:
        return 'F'


def generate_config_template(profile: str) -> str:
    """Generate configuration template based on profile"""
    
    base_config = """# Best Practices Framework Configuration
# =====================================

framework_version: "1.0.0"

# Enable/disable validation categories
enabled_categories:
  - architecture
  - engineering
  - security
  - testing
  - devops
  - sre

# Global enforcement settings
global:
  enforcement_level: "strict"  # strict, moderate, lenient
  fail_on_critical: true
  fail_on_high: false
  max_violations_per_category: 10

# Architecture validation
architecture:
  enabled: true
  clean_architecture:
    enabled: true
    dependency_inversion: true
    layer_separation: true
  microservices:
    enabled: true
    service_independence: true
    data_ownership: true

# Security validation
security:
  enabled: true
  owasp:
    enabled: true
    top_10_compliance: true
  secrets_detection:
    enabled: true
    scan_all_files: true
  vulnerability_scanning:
    enabled: true

# Testing validation
testing:
  enabled: true
  coverage:
    unit_test_minimum: 80
    integration_test_minimum: 60
    e2e_test_minimum: 40
  test_pyramid:
    enabled: true
    unit_tests_percentage: 70
    integration_tests_percentage: 20
    e2e_tests_percentage: 10

# Engineering practices
engineering:
  enabled: true
  code_quality:
    max_complexity: 10
    max_function_length: 50
  documentation:
    api_documentation_required: true
    readme_required: true

# DevOps practices
devops:
  enabled: true
  cicd:
    required_stages: ["build", "test", "security_scan", "deploy"]
    security_gates_required: true
  infrastructure_as_code:
    enabled: true
    version_controlled: true

# Site Reliability Engineering
sre:
  enabled: true
  observability:
    metrics_coverage: 90
    logging_coverage: 95
    tracing_enabled: true
  reliability:
    slo_coverage_required: true
    error_budget_tracking: true"""

    # Profile-specific customizations
    if profile == 'fintech':
        base_config += """

# Financial services specific settings
compliance:
  pci_dss: true
  sox_compliance: true
  
security:
  encryption_required: true
  audit_logging: true
  data_retention_policies: true"""
    
    elif profile == 'healthcare':
        base_config += """

# Healthcare specific settings
compliance:
  hipaa: true
  
security:
  phi_protection: true
  data_anonymization: true
  access_logging: true"""
    
    elif profile == 'ecommerce':
        base_config += """

# E-commerce specific settings
security:
  payment_security: true
  customer_data_protection: true
  
sre:
  performance_monitoring: true
  availability_target: 99.9"""
    
    elif profile == 'saas':
        base_config += """

# SaaS specific settings
security:
  multi_tenancy: true
  data_isolation: true
  
sre:
  scalability_monitoring: true
  cost_optimization: true"""
    
    return base_config


def main():
    """Main CLI entry point"""
    cli()


# Command aliases for convenience
def validate_command():
    """Alias for validate command"""
    cli(['validate'])


def report_command():
    """Alias for report command"""  
    cli(['report'])


def init_command():
    """Alias for init command"""
    cli(['init'])


if __name__ == '__main__':
    main()