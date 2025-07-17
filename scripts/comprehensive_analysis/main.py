"""Main entry point for comprehensive static analysis."""

import asyncio
import sys
import json
from pathlib import Path
from typing import List, Optional
import click
import logging

from .config.manager import ConfigManager, AnalysisConfig
from .orchestrator import AnalysisOrchestrator
from .reporting.console_reporter import ConsoleReporter

logger = logging.getLogger(__name__)


@click.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), 
              help="Path to configuration file")
@click.option("--profile", "-p", type=click.Choice(["strict", "balanced", "permissive"]),
              help="Analysis profile to use")
@click.option("--output-format", "-f", type=click.Choice(["console", "json", "html", "junit"]),
              help="Output format", default="console")
@click.option("--max-workers", "-j", type=int, help="Maximum number of parallel workers")
@click.option("--enable-caching/--disable-caching", default=None,
              help="Enable or disable result caching")
@click.option("--cache-dir", type=click.Path(), help="Cache directory")
@click.option("--include", multiple=True, help="Include patterns")
@click.option("--exclude", multiple=True, help="Exclude patterns")
@click.option("--enable-type-checking/--disable-type-checking", default=None,
              help="Enable or disable type checking")
@click.option("--enable-security/--disable-security", default=None,
              help="Enable or disable security scanning")
@click.option("--enable-performance/--disable-performance", default=None,
              help="Enable or disable performance analysis")
@click.option("--enable-docs/--disable-docs", default=None,
              help="Enable or disable documentation checking")
@click.option("--strict", is_flag=True, help="Enable strict mode")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--show-config", is_flag=True, help="Show effective configuration and exit")
@click.option("--output-file", "-o", type=click.Path(), help="Output file for results")
def main(paths, config, profile, output_format, max_workers, enable_caching, 
         cache_dir, include, exclude, enable_type_checking, enable_security,
         enable_performance, enable_docs, strict, quiet, verbose, show_config,
         output_file):
    """
    Comprehensive static analysis for Python projects.
    
    Provides compiler-level analysis including type checking, security scanning,
    performance analysis, and documentation validation.
    
    Examples:
        # Analyze current directory
        comprehensive-analysis .
        
        # Use strict profile
        comprehensive-analysis --profile strict src/
        
        # Generate JSON report
        comprehensive-analysis --output-format json --output-file report.json .
        
        # Analyze specific files
        comprehensive-analysis src/main.py tests/test_main.py
    """
    try:
        # Load configuration
        config_manager = ConfigManager()
        analysis_config = config_manager.load_config(Path(config) if config else None)
        
        # Apply CLI overrides
        analysis_config = _apply_cli_overrides(
            analysis_config, profile, output_format, max_workers, enable_caching,
            cache_dir, include, exclude, enable_type_checking, enable_security,
            enable_performance, enable_docs, strict, quiet, verbose
        )
        
        # Show configuration if requested
        if show_config:
            config_dict = config_manager.get_effective_config_dict(analysis_config)
            print(json.dumps(config_dict, indent=2, default=str))
            return
        
        # Set up paths
        analysis_paths = [Path(p) for p in paths] if paths else [Path.cwd()]
        
        # Run analysis
        asyncio.run(_run_analysis(analysis_config, analysis_paths, output_format, output_file))
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _apply_cli_overrides(config: AnalysisConfig, profile, output_format, max_workers,
                        enable_caching, cache_dir, include, exclude, enable_type_checking,
                        enable_security, enable_performance, enable_docs, strict,
                        quiet, verbose) -> AnalysisConfig:
    """Apply command-line overrides to configuration."""
    if profile:
        config.profile = profile
    if output_format:
        config.output_format = output_format
    if max_workers:
        config.max_workers = max_workers
    if enable_caching is not None:
        config.enable_caching = enable_caching
    if cache_dir:
        config.cache_dir = Path(cache_dir)
    if include:
        config.include_patterns = list(include)
    if exclude:
        config.exclude_patterns = list(exclude)
    if enable_type_checking is not None:
        config.enable_type_checking = enable_type_checking
    if enable_security is not None:
        config.enable_security_scanning = enable_security
    if enable_performance is not None:
        config.enable_performance_analysis = enable_performance
    if enable_docs is not None:
        config.enable_documentation_checking = enable_docs
    if strict:
        config.profile = "strict"
        # Enable strict mode for type checking
        if "mypy" not in config.tool_configs:
            config.tool_configs["mypy"] = {}
        config.tool_configs["mypy"]["strict"] = True
    if quiet:
        config.show_progress = False
    if verbose:
        config.show_context = True
    
    return config


async def _run_analysis(config: AnalysisConfig, paths: List[Path], 
                       output_format: str, output_file: Optional[Path]):
    """Run the analysis and generate output."""
    orchestrator = AnalysisOrchestrator(config)
    
    try:
        # Run analysis
        result = await orchestrator.analyze(paths)
        
        # Generate output
        if output_format == "console":
            reporter = ConsoleReporter(config)
            output = reporter.generate_report(result)
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(output)
                print(f"Console report written to {output_file}")
            else:
                print(output)
                
        elif output_format == "json":
            json_output = json.dumps(result.to_dict(), indent=2, default=str)
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(json_output)
                print(f"JSON report written to {output_file}")
            else:
                print(json_output)
                
        else:
            print(f"Output format '{output_format}' not yet implemented")
            return
        
        # Exit with appropriate code
        if not result.success:
            sys.exit(1)
        
        # Check for errors
        error_count = sum(
            len([issue for issue in tool_result.issues if issue.severity == "error"])
            for tool_result in result.results
        )
        
        if error_count > 0:
            sys.exit(1)
            
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    main()