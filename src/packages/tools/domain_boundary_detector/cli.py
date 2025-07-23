#!/usr/bin/env python3
"""CLI interface for domain boundary detector."""

import click
import sys
from pathlib import Path
from typing import Optional
import os

from core.domain.services.scanner import Scanner
from core.domain.services.registry import DomainRegistry
from core.domain.services.analyzer import BoundaryAnalyzer
from core.domain.services.documentation_scanner import DocumentationScanner
from core.domain.services.integrated_boundary_detector import IntegratedBoundaryDetector
from core.application.services.reporter import (
    ConsoleReporter, JsonReporter, MarkdownReporter, ReportOptions
)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """Domain Boundary Detector - Enforce clean architecture boundaries."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config


@cli.command()
@click.option('--path', '-p', type=click.Path(exists=True), default='.', 
              help='Path to scan (default: current directory)')
@click.option('--format', '-f', type=click.Choice(['console', 'json', 'markdown']), 
              default='console', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file (for json/markdown)')
@click.option('--strict', is_flag=True, help='Exit with error code on violations')
@click.option('--show-exempted', is_flag=True, help='Show exempted violations')
@click.option('--max-violations', type=int, help='Maximum violations to show')
@click.option('--group-by', type=click.Choice(['severity', 'type', 'domain']), 
              default='severity', help='Group violations by')
@click.option('--include-docs', is_flag=True, default=True, help='Include documentation scanning')
@click.option('--include-code', is_flag=True, default=True, help='Include code scanning')
@click.option('--docs-only', is_flag=True, help='Scan only documentation files')
@click.pass_context
def scan(ctx: click.Context, path: str, format: str, output: Optional[str], 
         strict: bool, show_exempted: bool, max_violations: Optional[int], 
         group_by: str, include_docs: bool, include_code: bool, docs_only: bool) -> None:
    """Scan for domain boundary violations."""
    verbose = ctx.obj['verbose']
    config_path = ctx.obj['config'] or '.domain-boundaries.yaml'
    
    # Handle docs-only mode
    if docs_only:
        include_docs = True
        include_code = False
    
    # Find monorepo root
    monorepo_root = _find_monorepo_root(Path(path))
    if not monorepo_root:
        click.echo("Error: Could not find monorepo root (looking for src/packages)", err=True)
        sys.exit(1)
        
    if verbose:
        click.echo(f"Monorepo root: {monorepo_root}")
        click.echo(f"Scanning: {path}")
        click.echo(f"Include docs: {include_docs}, Include code: {include_code}")
    
    # Use integrated boundary detector for comprehensive scanning
    detector = IntegratedBoundaryDetector(config_path)
    
    if verbose:
        click.echo("Scanning for violations...")
    
    # Perform integrated scan
    scan_result = detector.scan_repository(
        repository_path=str(monorepo_root),
        include_code=include_code,
        include_docs=include_docs
    )
    
    if verbose:
        click.echo(f"Found {scan_result.total_violations} total violations")
        click.echo(f"  - Code violations: {scan_result.summary['code_violations']}")
        click.echo(f"  - Documentation violations: {scan_result.summary['documentation_violations']}")
    
    # Generate and output report
    if output:
        report_content = detector.generate_report(scan_result, format, include_suggestions=True)
        with open(output, 'w', encoding='utf-8') as f:
            f.write(report_content)
        click.echo(f"Report written to: {output}")
    else:
        report_content = detector.generate_report(scan_result, format, include_suggestions=True)
        click.echo(report_content)
    
    # Exit code for CI/CD
    if strict:
        exit_code = detector.check_exit_code(scan_result)
        if exit_code > 0:
            sys.exit(exit_code)


@cli.command()
@click.option('--path', '-p', type=click.Path(exists=True), default='.', 
              help='Path to scan (default: current directory)')
@click.option('--format', '-f', type=click.Choice(['console', 'json', 'markdown']), 
              default='console', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file (for json/markdown)')
@click.option('--strict', is_flag=True, help='Exit with error code on violations')
@click.pass_context
def scan_docs(ctx: click.Context, path: str, format: str, output: Optional[str], strict: bool) -> None:
    """Scan only documentation files for domain boundary violations."""
    verbose = ctx.obj['verbose']
    config_path = ctx.obj['config'] or '.domain-boundaries.yaml'
    
    if verbose:
        click.echo(f"Scanning documentation files in: {path}")
        click.echo(f"Using config: {config_path}")
    
    # Initialize documentation scanner
    doc_scanner = DocumentationScanner(config_path)
    
    # Scan for documentation violations
    if Path(path).is_file() and path.endswith(('.md', '.rst')):
        violations = doc_scanner.scan_file(path)
    else:
        violations = doc_scanner.scan_repository(path)
    
    if verbose:
        click.echo(f"Found {len(violations)} documentation violations")
    
    # Generate report
    doc_scanner.violations = violations
    report_content = doc_scanner.generate_report(format)
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(report_content)
        click.echo(f"Documentation report written to: {output}")
    else:
        click.echo(report_content)
    
    # Exit code for CI/CD
    if strict and violations:
        critical_count = sum(1 for v in violations if v.severity == 'critical')
        if critical_count > 0:
            sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='.domain-boundaries.yaml',
              help='Output file path')
@click.option('--analyze', '-a', is_flag=True, 
              help='Analyze existing code to suggest domains')
@click.pass_context
def init(ctx: click.Context, output: str, analyze: bool) -> None:
    """Initialize a domain boundary configuration file."""
    verbose = ctx.obj['verbose']
    
    if analyze:
        click.echo("Analyzing codebase for domain suggestions...")
        # TODO: Implement smart domain analysis
        
    config_template = """# Domain Boundary Configuration
# This file defines the domain boundaries for your monorepo

domains:
  ai:
    description: "Artificial Intelligence and Machine Learning"
    packages:
      - ai/mlops
      - ai/ml_platform
      - ai/neuro_symbolic
    allowed_dependencies:
      - shared
      - infrastructure
      - data

  data:
    description: "Data processing and management"
    packages:
      - data/analytics
      - data/ingestion
      - data/quality
    allowed_dependencies:
      - shared
      - infrastructure

  finance:
    description: "Financial services and billing"
    packages:
      - finance/billing
      - finance/payments
    allowed_dependencies:
      - shared
      - infrastructure

  infrastructure:
    description: "Technical infrastructure"
    packages:
      - infrastructure/logging
      - infrastructure/monitoring
    allowed_dependencies: []  # Infrastructure should not depend on domains

  shared:
    description: "Shared utilities"
    packages:
      - shared/utils
      - shared/types
    allowed_dependencies:
      - infrastructure

rules:
  - name: no_cross_domain_imports
    description: "Prevent direct imports between business domains"
    severity: critical
    exceptions: []

  - name: no_circular_dependencies
    description: "Prevent circular dependencies"
    severity: critical

  - name: no_private_access
    description: "Prevent access to private modules"
    severity: warning

options:
  ignore_tests: true
  ignore_examples: true
  strict_mode: false
"""
    
    output_path = Path(output)
    
    if output_path.exists():
        if not click.confirm(f"{output} already exists. Overwrite?"):
            return
            
    output_path.write_text(config_template)
    click.echo(f"Created configuration file: {output}")
    click.echo("Edit this file to match your domain structure.")


@cli.command()
@click.argument('from_package')
@click.argument('to_package')
@click.option('--reason', '-r', required=True, help='Reason for exception')
@click.option('--expires', '-e', help='Expiration date (YYYY-MM-DD)')
@click.option('--config', '-c', type=click.Path(exists=True), 
              default='.domain-boundaries.yaml', help='Configuration file')
@click.pass_context
def add_exception(ctx: click.Context, from_package: str, to_package: str, 
                  reason: str, expires: Optional[str], config: str) -> None:
    """Add an exception for a boundary violation."""
    # TODO: Implement exception management
    click.echo(f"Adding exception: {from_package} -> {to_package}")
    click.echo(f"Reason: {reason}")
    if expires:
        click.echo(f"Expires: {expires}")


def _find_monorepo_root(start_path: Path) -> Optional[Path]:
    """Find the monorepo root by looking for src/packages."""
    current = start_path.resolve()
    
    while current != current.parent:
        if (current / 'src' / 'packages').exists():
            return current
        current = current.parent
        
    return None


if __name__ == '__main__':
    cli()