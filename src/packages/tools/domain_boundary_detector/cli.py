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
@click.pass_context
def scan(ctx: click.Context, path: str, format: str, output: Optional[str], 
         strict: bool, show_exempted: bool, max_violations: Optional[int], 
         group_by: str) -> None:
    """Scan for domain boundary violations."""
    verbose = ctx.obj['verbose']
    config_path = ctx.obj['config']
    
    # Find monorepo root
    monorepo_root = _find_monorepo_root(Path(path))
    if not monorepo_root:
        click.echo("Error: Could not find monorepo root (looking for src/packages)", err=True)
        sys.exit(1)
        
    if verbose:
        click.echo(f"Monorepo root: {monorepo_root}")
        click.echo(f"Scanning: {path}")
        
    # Load or create registry
    registry = DomainRegistry()
    if config_path:
        if verbose:
            click.echo(f"Loading config from: {config_path}")
        registry.load_from_file(Path(config_path))
    else:
        # Use default registry
        registry = registry.get_default_registry()
        
    # Scan files
    scanner = Scanner()
    scan_path = Path(path).resolve()
    
    if verbose:
        click.echo("Scanning for imports...")
        
    if scan_path.is_file():
        scan_result = scanner.scan_file(scan_path)
    else:
        scan_result = scanner.scan_directory(scan_path)
        
    if verbose:
        click.echo(f"Found {len(scan_result.imports)} imports")
        
    # Analyze violations
    analyzer = BoundaryAnalyzer(registry, str(monorepo_root))
    analysis_result = analyzer.analyze(scan_result)
    
    # Report results
    if format == 'console':
        options = ReportOptions(
            show_exempted=show_exempted,
            max_violations=max_violations,
            group_by=group_by,
            verbose=verbose
        )
        reporter = ConsoleReporter(options)
        reporter.report(analysis_result)
    elif format == 'json':
        reporter = JsonReporter()
        json_output = reporter.report(analysis_result, Path(output) if output else None)
        if not output:
            click.echo(json_output)
    elif format == 'markdown':
        reporter = MarkdownReporter()
        md_output = reporter.report(analysis_result, Path(output) if output else None)
        if not output:
            click.echo(md_output)
            
    # Exit code for CI/CD
    if strict and analysis_result.violations:
        critical_count = sum(1 for v in analysis_result.violations 
                           if v.severity.value == 'critical' and not v.is_exempted())
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