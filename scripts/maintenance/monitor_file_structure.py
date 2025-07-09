#!/usr/bin/env python3
"""
Monitor file structure script.

This script monitors the project file structure and emits Prometheus-friendly metrics,
including violations_total and cleanup_duration_seconds.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console

# Import the existing validation logic
sys.path.insert(0, str(Path(__file__).parent.parent / "validation"))
from validate_file_organization import validate_file_organization

app = typer.Typer(
    help="Monitor file structure and emit Prometheus-friendly metrics.",
    add_completion=False
)
console = Console()


class PrometheusMetrics:
    """Container for Prometheus-style metrics."""
    
    def __init__(self) -> None:
        self.metrics: Dict[str, float] = {}
        self.labels: Dict[str, Dict[str, str]] = {}
        self.help_text: Dict[str, str] = {}
        self.metric_types: Dict[str, str] = {}
    
    def add_counter(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, help_text: str = "") -> None:
        """Add a counter metric."""
        self.metrics[name] = value
        self.labels[name] = labels or {}
        self.help_text[name] = help_text
        self.metric_types[name] = "counter"
    
    def add_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, help_text: str = "") -> None:
        """Add a gauge metric."""
        self.metrics[name] = value
        self.labels[name] = labels or {}
        self.help_text[name] = help_text
        self.metric_types[name] = "gauge"
    
    def add_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, help_text: str = "") -> None:
        """Add a histogram metric."""
        self.metrics[name] = value
        self.labels[name] = labels or {}
        self.help_text[name] = help_text
        self.metric_types[name] = "histogram"
    
    def to_prometheus_format(self) -> str:
        """Convert metrics to Prometheus exposition format."""
        output = []
        
        for metric_name, value in self.metrics.items():
            # Add help text
            if self.help_text[metric_name]:
                output.append(f"# HELP {metric_name} {self.help_text[metric_name]}")
            
            # Add type
            output.append(f"# TYPE {metric_name} {self.metric_types[metric_name]}")
            
            # Add metric with labels
            labels = self.labels[metric_name]
            if labels:
                label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
                output.append(f"{metric_name}{{{label_str}}} {value}")
            else:
                output.append(f"{metric_name} {value}")
            
            output.append("")  # Empty line between metrics
        
        return "\\n".join(output)
    
    def to_json(self) -> Dict:
        """Convert metrics to JSON format."""
        return {
            "metrics": self.metrics,
            "labels": self.labels,
            "help_text": self.help_text,
            "metric_types": self.metric_types,
            "timestamp": datetime.now().isoformat()
        }


def collect_file_structure_metrics() -> PrometheusMetrics:
    """
    Collect file structure metrics for monitoring.
    
    Returns:
        PrometheusMetrics: Container with collected metrics
    """
    metrics = PrometheusMetrics()
    
    # Measure validation duration
    start_time = time.time()
    is_valid, violations, suggestions = validate_file_organization()
    end_time = time.time()
    
    validation_duration = end_time - start_time
    
    # Add basic metrics
    metrics.add_counter(
        "file_structure_violations_total",
        len(violations),
        labels={"component": "file_organization"},
        help_text="Total number of file structure violations found"
    )
    
    metrics.add_counter(
        "file_structure_suggestions_total",
        len(suggestions),
        labels={"component": "file_organization"},
        help_text="Total number of suggestions for file structure improvements"
    )
    
    metrics.add_histogram(
        "file_structure_validation_duration_seconds",
        validation_duration,
        labels={"component": "file_organization"},
        help_text="Duration of file structure validation in seconds"
    )
    
    metrics.add_gauge(
        "file_structure_is_valid",
        1.0 if is_valid else 0.0,
        labels={"component": "file_organization"},
        help_text="Whether the file structure is valid (1) or not (0)"
    )
    
    # Add detailed metrics by violation type
    violation_types = {}
    for violation in violations:
        if "file" in violation.lower():
            violation_types["file"] = violation_types.get("file", 0) + 1
        elif "directory" in violation.lower():
            violation_types["directory"] = violation_types.get("directory", 0) + 1
        else:
            violation_types["other"] = violation_types.get("other", 0) + 1
    
    for violation_type, count in violation_types.items():
        metrics.add_counter(
            f"file_structure_violations_by_type_total",
            count,
            labels={"component": "file_organization", "type": violation_type},
            help_text=f"Total number of {violation_type} structure violations"
        )
    
    # Add project health metrics
    project_root = Path.cwd()
    
    # Count Python files
    python_files = len(list(project_root.rglob("*.py")))
    metrics.add_gauge(
        "project_python_files_total",
        python_files,
        labels={"component": "project_stats"},
        help_text="Total number of Python files in the project"
    )
    
    # Count test files
    test_files = len(list(project_root.rglob("test_*.py"))) + len(list(project_root.rglob("*_test.py")))
    metrics.add_gauge(
        "project_test_files_total",
        test_files,
        labels={"component": "project_stats"},
        help_text="Total number of test files in the project"
    )
    
    # Count directories
    directories = len([d for d in project_root.rglob("*") if d.is_dir()])
    metrics.add_gauge(
        "project_directories_total",
        directories,
        labels={"component": "project_stats"},
        help_text="Total number of directories in the project"
    )
    
    # Check for important files
    important_files = ["README.md", "pyproject.toml", "setup.py", "requirements.txt"]
    for file_name in important_files:
        file_exists = 1.0 if (project_root / file_name).exists() else 0.0
        metrics.add_gauge(
            "project_important_file_exists",
            file_exists,
            labels={"component": "project_stats", "file": file_name},
            help_text=f"Whether {file_name} exists in the project root"
        )
    
    return metrics


@app.command()
def main(
    format: str = typer.Option("prometheus", "--format", "-f", help="Output format: prometheus, json"),
    output: str = typer.Option(None, "--output", "-o", help="Output file (default: stdout)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    include_timestamp: bool = typer.Option(True, "--timestamp/--no-timestamp", help="Include timestamp in output")
) -> None:
    """
    Monitor file structure and emit Prometheus-friendly metrics.
    
    This tool analyzes the project file structure and generates metrics suitable for
    monitoring systems like Prometheus. It tracks violations, validation duration,
    and project health indicators.
    
    Examples:
        # Output metrics to stdout in Prometheus format
        python monitor_file_structure.py
        
        # Output metrics to file in JSON format
        python monitor_file_structure.py --format json --output metrics.json
        
        # Output with verbose logging
        python monitor_file_structure.py --verbose
    """
    if verbose:
        console.print("[bold blue]📊 File Structure Monitoring[/bold blue]")
        console.print("")
        console.print("Collecting file structure metrics...")
    
    # Collect metrics
    metrics = collect_file_structure_metrics()
    
    # Format output
    if format.lower() == "prometheus":
        output_text = metrics.to_prometheus_format()
    elif format.lower() == "json":
        output_data = metrics.to_json()
        output_text = json.dumps(output_data, indent=2)
    else:
        console.print(f"[red]Unknown format: {format}[/red]")
        sys.exit(1)
    
    # Add timestamp if requested
    if include_timestamp and format.lower() == "prometheus":
        timestamp_comment = f"# Generated at {datetime.now().isoformat()}"
        output_text = f"{timestamp_comment}\\n{output_text}"
    
    # Write output
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path, 'w') as f:
                f.write(output_text)
            if verbose:
                console.print(f"[green]Metrics saved to:[/green] {output_path}")
        except OSError as e:
            console.print(f"[red]Failed to save metrics:[/red] {e}")
            sys.exit(1)
    else:
        print(output_text)
    
    # Exit with appropriate code based on validation results
    if metrics.metrics.get("file_structure_is_valid", 0.0) == 0.0:
        if verbose:
            violations_count = int(metrics.metrics.get("file_structure_violations_total", 0))
            console.print(f"[red]File structure validation failed with {violations_count} violations[/red]")
        sys.exit(1)
    else:
        if verbose:
            console.print("[green]File structure validation passed[/green]")
        sys.exit(0)


if __name__ == "__main__":
    app()
