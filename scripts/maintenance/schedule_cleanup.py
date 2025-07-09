#!/usr/bin/env python3
"""
Schedule cleanup script - CLI wrapper for maintenance tasks.

This script runs both cleanup_repository.py and validate_structure.py and pushes
reports to reports/quality/ directory.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    help="CLI wrapper that runs cleanup and validation scripts, generating consolidated reports.",
    add_completion=False
)
console = Console()


class ScheduledCleanupResult:
    """Result container for scheduled cleanup operations."""
    
    def __init__(self) -> None:
        self.timestamp = datetime.now().isoformat()
        self.cleanup_success = False
        self.validation_success = False
        self.cleanup_report: Optional[Dict] = None
        self.validation_report: Optional[Dict] = None
        self.errors: List[str] = []
        self.duration: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "duration_seconds": self.duration,
            "cleanup_success": self.cleanup_success,
            "validation_success": self.validation_success,
            "cleanup_report": self.cleanup_report,
            "validation_report": self.validation_report,
            "errors": self.errors,
            "overall_success": self.cleanup_success and self.validation_success
        }


def run_cleanup_repository(dry_run: bool = False, verbose: bool = False) -> tuple[bool, Optional[Dict]]:
    """
    Run the cleanup repository script.
    
    Args:
        dry_run: Whether to run in dry-run mode
        verbose: Whether to enable verbose output
        
    Returns:
        Tuple of (success, report_data)
    """
    script_path = Path(__file__).parent / "cleanup_repository.py"
    cmd = [sys.executable, str(script_path)]
    
    if dry_run:
        cmd.append("--dry-run")
    if verbose:
        cmd.append("--verbose")
    
    cmd.extend(["--output", "reports/quality/cleanup_report.json"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Load the generated report
        report_path = Path("reports/quality/cleanup_report.json")
        if report_path.exists():
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            return True, report_data
        else:
            return True, None
            
    except subprocess.CalledProcessError as e:
        if verbose:
            console.print(f"[red]Cleanup failed:[/red] {e}")
            console.print(f"[red]stderr:[/red] {e.stderr}")
        return False, None
    except Exception as e:
        if verbose:
            console.print(f"[red]Unexpected error in cleanup:[/red] {e}")
        return False, None


def run_validate_structure(verbose: bool = False) -> tuple[bool, Optional[Dict]]:
    """
    Run the validate structure script.
    
    Args:
        verbose: Whether to enable verbose output
        
    Returns:
        Tuple of (success, report_data)
    """
    script_path = Path(__file__).parent / "validate_structure.py"
    cmd = [sys.executable, str(script_path)]
    
    if verbose:
        cmd.append("--verbose")
    
    cmd.extend(["--output-dir", "reports/quality"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Load the generated report
        report_path = Path("reports/quality/structure_validation.json")
        if report_path.exists():
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            return True, report_data
        else:
            return True, None
            
    except subprocess.CalledProcessError as e:
        # Structure validation returns non-zero exit code on validation failures
        # This is expected behavior, so we still try to load the report
        report_path = Path("reports/quality/structure_validation.json")
        if report_path.exists():
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            return False, report_data
        else:
            if verbose:
                console.print(f"[red]Structure validation failed:[/red] {e}")
                console.print(f"[red]stderr:[/red] {e.stderr}")
            return False, None
    except Exception as e:
        if verbose:
            console.print(f"[red]Unexpected error in structure validation:[/red] {e}")
        return False, None


def display_summary(result: ScheduledCleanupResult) -> None:
    """
    Display a summary of the scheduled cleanup results.
    
    Args:
        result: The scheduled cleanup result object
    """
    table = Table(title="Scheduled Cleanup Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")
    
    # Cleanup status
    cleanup_status = "✅ SUCCESS" if result.cleanup_success else "❌ FAILED"
    cleanup_details = ""
    if result.cleanup_report:
        metrics = result.cleanup_report.get("metrics", {})
        files_removed = metrics.get("files_removed", 0)
        dirs_removed = metrics.get("directories_removed", 0)
        cleanup_details = f"Files: {files_removed}, Dirs: {dirs_removed}"
    
    table.add_row("Repository Cleanup", cleanup_status, cleanup_details)
    
    # Validation status
    validation_status = "✅ SUCCESS" if result.validation_success else "❌ FAILED"
    validation_details = ""
    if result.validation_report:
        violations = len(result.validation_report.get("violations", []))
        validation_details = f"Violations: {violations}"
    
    table.add_row("Structure Validation", validation_status, validation_details)
    
    # Overall status
    overall_status = "✅ SUCCESS" if result.cleanup_success and result.validation_success else "❌ FAILED"
    overall_details = f"Duration: {result.duration:.2f}s" if result.duration else "N/A"
    
    table.add_row("Overall", overall_status, overall_details)
    
    console.print(table)
    
    if result.errors:
        console.print("\\n[red]Errors encountered:[/red]")
        for error in result.errors:
            console.print(f"  • {error}")


@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Run cleanup in dry-run mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    output: str = typer.Option("reports/quality/scheduled_cleanup_report.json", "--output", "-o", help="Output consolidated report file"),
    skip_cleanup: bool = typer.Option(False, "--skip-cleanup", help="Skip repository cleanup"),
    skip_validation: bool = typer.Option(False, "--skip-validation", help="Skip structure validation")
) -> None:
    """
    Run scheduled cleanup and validation tasks.
    
    This tool runs both the repository cleanup and structure validation scripts,
    generating consolidated reports in the reports/quality/ directory.
    
    Examples:
        # Run full cleanup and validation
        python schedule_cleanup.py
        
        # Run in dry-run mode
        python schedule_cleanup.py --dry-run
        
        # Run only cleanup
        python schedule_cleanup.py --skip-validation
        
        # Run only validation
        python schedule_cleanup.py --skip-cleanup
    """
    console.print("[bold blue]🔧 Scheduled Cleanup & Validation[/bold blue]")
    console.print("")
    
    result = ScheduledCleanupResult()
    start_time = datetime.now()
    
    # Ensure output directory exists
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        
        if not skip_cleanup:
            task = progress.add_task("Running repository cleanup...", total=None)
            cleanup_success, cleanup_report = run_cleanup_repository(dry_run=dry_run, verbose=verbose)
            result.cleanup_success = cleanup_success
            result.cleanup_report = cleanup_report
            
            if not cleanup_success:
                result.errors.append("Repository cleanup failed")
        else:
            result.cleanup_success = True  # Skip counts as success
            
        if not skip_validation:
            task = progress.add_task("Running structure validation...", total=None)
            validation_success, validation_report = run_validate_structure(verbose=verbose)
            result.validation_success = validation_success
            result.validation_report = validation_report
            
            if not validation_success:
                result.errors.append("Structure validation failed")
        else:
            result.validation_success = True  # Skip counts as success
    
    end_time = datetime.now()
    result.duration = (end_time - start_time).total_seconds()
    
    # Display summary
    display_summary(result)
    
    # Save consolidated report
    try:
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"\\n[green]Consolidated report saved to:[/green] {output_path}")
    except OSError as e:
        console.print(f"[red]Failed to save consolidated report:[/red] {e}")
        sys.exit(1)
    
    # Exit with appropriate code
    if not (result.cleanup_success and result.validation_success):
        console.print(f"\\n[red]Scheduled cleanup completed with errors[/red]")
        sys.exit(1)
    else:
        console.print("\\n[green]Scheduled cleanup completed successfully[/green]")
        sys.exit(0)


if __name__ == "__main__":
    app()
