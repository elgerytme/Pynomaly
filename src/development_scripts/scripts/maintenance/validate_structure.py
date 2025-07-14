#!/usr/bin/env python3
"""
Enhanced validate_structure.py for maintenance directory.

This script validates the project structure, returns non-zero exit code on errors,
and generates SARIF reports for GitHub code-scanning.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

# Import the existing validation logic
sys.path.insert(0, str(Path(__file__).parent.parent / "validation"))
from validate_file_organization import validate_file_organization

app = typer.Typer(
    help="Validate project structure and generate reports for GitHub code scanning.",
    add_completion=False,
)
console = Console()

SARIF_VERSION = "2.1.0"


def generate_sarif_report(violations: list[str]) -> dict:
    """
    Generate SARIF report from validation violations.

    Args:
        violations: List of validation violations

    Returns:
        Dict: SARIF report structure
    """
    rules = []
    results = []

    for i, violation in enumerate(violations):
        rule_id = f"file-organization-{i+1}"
        rule = {
            "id": rule_id,
            "shortDescription": {"text": "File organization violation"},
            "fullDescription": {
                "text": "File or directory violates project organization standards"
            },
            "helpUri": "https://github.com/pynomaly/pynomaly/blob/main/docs/development/FILE_ORGANIZATION_STANDARDS.md",
            "properties": {"category": "maintainability", "severity": "error"},
        }
        rules.append(rule)

        result = {
            "ruleId": rule_id,
            "level": "error",
            "message": {"text": violation},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": "."},
                        "region": {"startLine": 1, "startColumn": 1},
                    }
                }
            ],
        }
        results.append(result)

    sarif_report = {
        "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "validate_structure",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/pynomaly/pynomaly",
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }

    return sarif_report


def display_validation_results(
    is_valid: bool, violations: list[str], suggestions: list[str]
) -> None:
    """
    Display validation results in a formatted table.

    Args:
        is_valid: Whether validation passed
        violations: List of validation violations
        suggestions: List of suggested fixes
    """
    if is_valid:
        console.print("[green]✅ Project structure validation PASSED[/green]")
        return

    console.print("[red]❌ Project structure validation FAILED[/red]")
    console.print("")

    if violations:
        console.print("[red]Violations found:[/red]")
        for violation in violations:
            console.print(f"  • {violation}")
        console.print("")

    if suggestions:
        console.print("[yellow]Suggested actions:[/yellow]")
        for suggestion in suggestions:
            console.print(f"  • {suggestion}")
        console.print("")

    console.print("For more information, see:")
    console.print("  docs/development/FILE_ORGANIZATION_STANDARDS.md")


@app.command()
def main(
    output_dir: str = typer.Option(
        "reports/quality", "--output-dir", "-o", help="Output directory for reports"
    ),
    sarif: bool = typer.Option(
        True,
        "--sarif/--no-sarif",
        help="Generate SARIF report for GitHub code scanning",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """
    Validate project structure against FILE_ORGANIZATION_STANDARDS.

    This tool validates the project structure and generates reports for GitHub code scanning.
    It returns a non-zero exit code when validation fails, making it suitable for CI/CD pipelines.

    Examples:
        # Basic validation
        python validate_structure.py

        # Validate with custom output directory
        python validate_structure.py --output-dir custom/reports

        # Validate without SARIF generation
        python validate_structure.py --no-sarif
    """
    console.print("[bold blue]🔍 Project Structure Validation[/bold blue]")
    console.print("")

    if verbose:
        console.print(
            "Validating project structure against FILE_ORGANIZATION_STANDARDS..."
        )

    # Run the validation
    is_valid, violations, suggestions = validate_file_organization()

    # Display results
    display_validation_results(is_valid, violations, suggestions)

    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate reports
    timestamp = datetime.now().isoformat()

    json_report = {
        "timestamp": timestamp,
        "is_valid": is_valid,
        "violations": violations,
        "suggestions": suggestions,
        "validation_type": "structure",
        "summary": {
            "total_violations": len(violations),
            "total_suggestions": len(suggestions),
        },
    }

    # Save JSON report
    json_path = output_path / "structure_validation.json"
    try:
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2)
        if verbose:
            console.print(f"[green]JSON report saved to:[/green] {json_path}")
    except OSError as e:
        console.print(f"[red]Failed to save JSON report:[/red] {e}")
        sys.exit(1)

    # Generate and save SARIF report
    if sarif:
        sarif_report = generate_sarif_report(violations)
        sarif_path = output_path / "structure_validation.sarif"
        try:
            with open(sarif_path, "w") as f:
                json.dump(sarif_report, f, indent=2)
            if verbose:
                console.print(f"[green]SARIF report saved to:[/green] {sarif_path}")
        except OSError as e:
            console.print(f"[red]Failed to save SARIF report:[/red] {e}")
            sys.exit(1)

    # Exit with appropriate code
    if not is_valid:
        console.print(
            f"\\n[red]Validation failed with {len(violations)} violations[/red]"
        )
        sys.exit(1)
    else:
        console.print("\\n[green]Validation passed successfully[/green]")
        sys.exit(0)


if __name__ == "__main__":
    app()
