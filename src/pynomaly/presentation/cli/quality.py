"""CLI commands for quality gates and code quality validation."""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ...infrastructure.quality.quality_gates import (
    QualityGateType,
    QualityGateValidator,
    QualityLevel,
)

console = Console()


@click.group()
def quality():
    """Quality gates and code quality validation commands."""
    pass


@quality.command()
@click.argument("feature_path", type=click.Path(exists=True, path_type=Path))
@click.option("--feature-name", "-n", help="Name of the feature being validated")
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output file for report"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "html", "table"]),
    default="table",
    help="Output format",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=80.0,
    help="Minimum success rate threshold for approval",
)
@click.option(
    "--fail-on-critical",
    is_flag=True,
    default=True,
    help="Fail if any critical quality gates fail",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed gate results")
def validate(
    feature_path: Path,
    feature_name: str | None,
    output: Path | None,
    output_format: str,
    threshold: float,
    fail_on_critical: bool,
    verbose: bool,
):
    """Validate feature against quality gates.

    FEATURE_PATH: Path to the feature file or directory to validate
    """
    console.print(f"🔍 Validating feature: {feature_path}")

    if feature_name is None:
        feature_name = feature_path.stem

    # Run validation with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running quality gate validation...", total=None)

        try:
            validator = QualityGateValidator()
            report = validator.validate_feature(feature_path, feature_name)
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"❌ Validation failed: {e}", style="red")
            raise click.Abort()

    # Display results
    if output_format == "table":
        _display_table_report(report, verbose)
    elif output_format == "json":
        report_json = json.dumps(report.to_dict(), indent=2)
        if output:
            output.write_text(report_json)
            console.print(f"📄 JSON report saved to: {output}")
        else:
            console.print(report_json)
    elif output_format == "html":
        html_report = validator.generate_report_html(report)
        if output:
            output.write_text(html_report)
            console.print(f"📄 HTML report saved to: {output}")
        else:
            console.print("HTML report generated (use --output to save)")

    # Determine validation result
    success = True

    if fail_on_critical and report.critical_failures > 0:
        console.print(
            f"❌ Critical failures detected: {report.critical_failures}", style="red"
        )
        success = False

    if report.success_rate < threshold:
        console.print(
            f"❌ Success rate {report.success_rate:.1f}% below threshold {threshold}%",
            style="red",
        )
        success = False

    if success:
        if report.integration_approved:
            console.print("✅ Feature approved for integration", style="green")
        else:
            console.print(
                "⚠️ Feature has issues but meets minimum requirements", style="yellow"
            )
    else:
        console.print("❌ Feature validation failed", style="red")
        raise click.Abort()


@quality.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--pattern", "-p", default="*.py", help="File pattern to match")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for reports",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "html", "summary"]),
    default="summary",
    help="Output format",
)
@click.option("--parallel", is_flag=True, help="Run validation in parallel")
@click.option(
    "--threshold", "-t", type=float, default=80.0, help="Minimum success rate threshold"
)
def batch(
    directory: Path,
    pattern: str,
    output: Path | None,
    output_format: str,
    parallel: bool,
    threshold: float,
):
    """Batch validate multiple features in a directory.

    DIRECTORY: Directory containing features to validate
    """
    console.print(f"🔍 Batch validating features in: {directory}")
    console.print(f"📁 Pattern: {pattern}")

    # Find all matching files
    files = list(directory.rglob(pattern))
    if not files:
        console.print(f"❌ No files found matching pattern: {pattern}", style="red")
        raise click.Abort()

    console.print(f"📊 Found {len(files)} files to validate")

    # Validate each file
    reports = []
    failed_validations = []

    with Progress(console=console) as progress:
        task = progress.add_task("Validating features...", total=len(files))

        for file_path in files:
            try:
                feature_name = file_path.stem
                validator = QualityGateValidator()
                report = validator.validate_feature(file_path, feature_name)
                reports.append(report)

                if report.success_rate < threshold:
                    failed_validations.append((file_path, report))

                progress.advance(task)

            except Exception as e:
                console.print(f"❌ Failed to validate {file_path}: {e}", style="red")
                failed_validations.append((file_path, None))
                progress.advance(task)

    # Generate batch report
    _display_batch_summary(reports, failed_validations, threshold)

    # Save reports if requested
    if output:
        output.mkdir(exist_ok=True)

        for report in reports:
            if output_format == "json":
                report_file = output / f"{report.feature_name}_quality_report.json"
                report_file.write_text(json.dumps(report.to_dict(), indent=2))
            elif output_format == "html":
                validator = QualityGateValidator()
                html_content = validator.generate_report_html(report)
                report_file = output / f"{report.feature_name}_quality_report.html"
                report_file.write_text(html_content)

        console.print(f"📄 Reports saved to: {output}")

    # Exit with error code if validations failed
    if failed_validations:
        console.print(f"❌ {len(failed_validations)} validations failed", style="red")
        raise click.Abort()


@quality.command()
@click.argument("report_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "input_format",
    type=click.Choice(["json"]),
    default="json",
    help="Input report format",
)
def show(report_file: Path, input_format: str):
    """Display a previously generated quality gate report.

    REPORT_FILE: Path to the quality gate report file
    """
    try:
        if input_format == "json":
            report_data = json.loads(report_file.read_text())
            _display_report_from_dict(report_data)
        else:
            console.print(f"❌ Unsupported format: {input_format}", style="red")
            raise click.Abort()

    except Exception as e:
        console.print(f"❌ Failed to load report: {e}", style="red")
        raise click.Abort()


@quality.command()
@click.option(
    "--gate-type",
    type=click.Choice([t.value for t in QualityGateType]),
    help="Filter by gate type",
)
@click.option(
    "--quality-level",
    type=click.Choice([l.value for l in QualityLevel]),
    help="Filter by quality level",
)
def gates(gate_type: str | None, quality_level: str | None):
    """List available quality gates and their descriptions."""
    console.print("🚪 Available Quality Gates")

    # Define gate descriptions
    gate_descriptions = {
        "Cyclomatic Complexity": "Measures code complexity and maintainability",
        "Code Style": "Checks adherence to coding style guidelines",
        "Type Hints": "Validates type annotation coverage",
        "Import Quality": "Checks import organization and practices",
        "Execution Performance": "Analyzes runtime performance characteristics",
        "Memory Usage": "Evaluates memory consumption patterns",
        "Algorithmic Complexity": "Identifies inefficient algorithmic patterns",
        "Docstring Coverage": "Measures documentation coverage",
        "Documentation Quality": "Assesses documentation completeness and clarity",
        "API Documentation": "Validates public API documentation",
        "Clean Architecture": "Enforces architectural layer separation",
        "Dependency Management": "Checks dependency organization",
        "Interface Design": "Validates interface design quality",
        "Test Coverage": "Measures test coverage percentage",
        "Test Quality": "Assesses test comprehensiveness and quality",
        "Edge Cases Coverage": "Validates edge case testing",
        "Security Patterns": "Identifies security vulnerabilities",
        "Input Validation": "Checks input validation practices",
    }

    # Gate type mapping
    gate_types = {
        "Cyclomatic Complexity": QualityGateType.CODE_QUALITY,
        "Code Style": QualityGateType.CODE_QUALITY,
        "Type Hints": QualityGateType.CODE_QUALITY,
        "Import Quality": QualityGateType.CODE_QUALITY,
        "Execution Performance": QualityGateType.PERFORMANCE,
        "Memory Usage": QualityGateType.PERFORMANCE,
        "Algorithmic Complexity": QualityGateType.PERFORMANCE,
        "Docstring Coverage": QualityGateType.DOCUMENTATION,
        "Documentation Quality": QualityGateType.DOCUMENTATION,
        "API Documentation": QualityGateType.DOCUMENTATION,
        "Clean Architecture": QualityGateType.ARCHITECTURE,
        "Dependency Management": QualityGateType.ARCHITECTURE,
        "Interface Design": QualityGateType.ARCHITECTURE,
        "Test Coverage": QualityGateType.TESTING,
        "Test Quality": QualityGateType.TESTING,
        "Edge Cases Coverage": QualityGateType.TESTING,
        "Security Patterns": QualityGateType.SECURITY,
        "Input Validation": QualityGateType.SECURITY,
    }

    # Quality levels (simplified mapping)
    gate_levels = {
        "Clean Architecture": QualityLevel.CRITICAL,
        "Test Coverage": QualityLevel.CRITICAL,
        "Security Patterns": QualityLevel.HIGH,
        "Cyclomatic Complexity": QualityLevel.HIGH,
        "Type Hints": QualityLevel.HIGH,
        "Docstring Coverage": QualityLevel.HIGH,
        "Code Style": QualityLevel.MEDIUM,
        "Import Quality": QualityLevel.MEDIUM,
        "Edge Cases Coverage": QualityLevel.MEDIUM,
        "Documentation Quality": QualityLevel.MEDIUM,
        "API Documentation": QualityLevel.MEDIUM,
        "Dependency Management": QualityLevel.HIGH,
        "Interface Design": QualityLevel.HIGH,
        "Test Quality": QualityLevel.HIGH,
        "Input Validation": QualityLevel.HIGH,
        "Execution Performance": QualityLevel.HIGH,
        "Memory Usage": QualityLevel.MEDIUM,
        "Algorithmic Complexity": QualityLevel.MEDIUM,
    }

    # Create table
    table = Table(title="Quality Gates")
    table.add_column("Gate Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Level", style="yellow")
    table.add_column("Description", style="white")

    for gate_name, description in gate_descriptions.items():
        gate_type_enum = gate_types.get(gate_name, QualityGateType.CODE_QUALITY)
        gate_level_enum = gate_levels.get(gate_name, QualityLevel.MEDIUM)

        # Apply filters
        if gate_type and gate_type_enum.value != gate_type:
            continue
        if quality_level and gate_level_enum.value != quality_level:
            continue

        # Style based on level
        level_style = {
            QualityLevel.CRITICAL: "red",
            QualityLevel.HIGH: "orange3",
            QualityLevel.MEDIUM: "yellow",
            QualityLevel.LOW: "green",
        }.get(gate_level_enum, "white")

        table.add_row(
            gate_name,
            gate_type_enum.value.title(),
            f"[{level_style}]{gate_level_enum.value.title()}[/{level_style}]",
            description,
        )

    console.print(table)


def _display_table_report(report, verbose: bool = False):
    """Display quality gate report as a table."""
    # Summary panel
    summary_text = f"""
Feature: {report.feature_name}
Path: {report.feature_path}
Overall Score: {report.overall_percentage:.1f}% ({report.overall_score:.1f}/{report.max_overall_score:.1f})
Success Rate: {report.success_rate:.1f}% ({report.passed_gates}/{report.total_gates} gates passed)
Critical Failures: {report.critical_failures}
Integration Approved: {"✅ Yes" if report.integration_approved else "❌ No"}
Validation Time: {report.validation_time_seconds:.2f}s
    """.strip()

    panel_style = "green" if report.integration_approved else "red"
    console.print(Panel(summary_text, title="Quality Gate Summary", style=panel_style))

    # Gates table
    table = Table(title="Quality Gate Results")
    table.add_column("Gate", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Level", style="yellow")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right", style="green")

    if verbose:
        table.add_column("Recommendations", style="dim")

    for result in report.gate_results:
        # Status icon and style
        if result.passed:
            status = "✅ Pass"
            status_style = "green"
        else:
            status = "❌ Fail"
            status_style = (
                "red" if result.quality_level == QualityLevel.CRITICAL else "orange3"
            )

        # Level style
        level_style = {
            QualityLevel.CRITICAL: "red",
            QualityLevel.HIGH: "orange3",
            QualityLevel.MEDIUM: "yellow",
            QualityLevel.LOW: "green",
        }.get(result.quality_level, "white")

        row = [
            result.gate_name,
            result.gate_type.value.title(),
            f"[{level_style}]{result.quality_level.value.title()}[/{level_style}]",
            f"[{status_style}]{status}[/{status_style}]",
            f"{result.percentage_score:.1f}%",
        ]

        if verbose and result.recommendations:
            row.append(
                "; ".join(result.recommendations[:2])
            )  # Limit to first 2 recommendations
        elif verbose:
            row.append("-")

        table.add_row(*row)

    console.print(table)


def _display_batch_summary(reports, failed_validations, threshold: float):
    """Display batch validation summary."""
    total_features = len(reports) + len([f for f in failed_validations if f[1] is None])
    passed_features = len([r for r in reports if r.success_rate >= threshold])
    failed_features = total_features - passed_features

    summary_text = f"""
Total Features: {total_features}
Passed: {passed_features}
Failed: {failed_features}
Success Rate: {(passed_features / total_features * 100) if total_features > 0 else 0:.1f}%
Threshold: {threshold}%
    """.strip()

    panel_style = "green" if failed_features == 0 else "red"
    console.print(
        Panel(summary_text, title="Batch Validation Summary", style=panel_style)
    )

    # Show failed validations
    if failed_validations:
        console.print("\n❌ Failed Validations:")
        for file_path, report in failed_validations:
            if report:
                console.print(
                    f"  {file_path}: {report.success_rate:.1f}% (Critical: {report.critical_failures})"
                )
            else:
                console.print(f"  {file_path}: Validation error")


def _display_report_from_dict(report_data):
    """Display report from dictionary data."""
    # Convert dict back to display format
    console.print(f"📊 Quality Gate Report: {report_data['feature_name']}")
    console.print(f"📁 Path: {report_data['feature_path']}")
    console.print(f"📈 Overall Score: {report_data['overall_percentage']:.1f}%")
    console.print(f"✅ Success Rate: {report_data['success_rate']:.1f}%")
    console.print(f"🚨 Critical Failures: {report_data['critical_failures']}")
    console.print(
        f"🔒 Integration Approved: {'Yes' if report_data['integration_approved'] else 'No'}"
    )

    # Show gate results
    table = Table(title="Gate Results")
    table.add_column("Gate", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right", style="green")

    for gate_result in report_data["gate_results"]:
        status = "✅ Pass" if gate_result["passed"] else "❌ Fail"
        status_style = "green" if gate_result["passed"] else "red"

        table.add_row(
            gate_result["gate_name"],
            f"[{status_style}]{status}[/{status_style}]",
            f"{gate_result['percentage_score']:.1f}%",
        )

    console.print(table)
