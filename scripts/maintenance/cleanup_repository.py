#!/usr/bin/env python3
"""
Cleanup repository script.

This script performs a comprehensive cleanup of the repository with options for dry-run,
execution metrics, and JSON report output. It removes temporary files, cache directories,
and other artifacts to maintain a clean repository.
"""

import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    help="Cleanup the repository with options for dry-run and report generation.",
    add_completion=False
)
console = Console()


class CleanupMetrics:
    """Class to track cleanup execution metrics."""
    
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.files_removed: int = 0
        self.directories_removed: int = 0
        self.bytes_freed: int = 0
        self.errors: List[str] = []

    def start(self) -> None:
        """Start timing the cleanup process."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop timing the cleanup process."""
        self.end_time = time.time()

    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the cleanup process in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def add_file_removed(self, file_path: Path) -> None:
        """Record a file removal."""
        self.files_removed += 1
        if file_path.exists():
            self.bytes_freed += file_path.stat().st_size

    def add_directory_removed(self, dir_path: Path) -> None:
        """Record a directory removal."""
        self.directories_removed += 1
        if dir_path.exists():
            self.bytes_freed += sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())

    def add_error(self, error: str) -> None:
        """Record an error during cleanup."""
        self.errors.append(error)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration,
            "files_removed": self.files_removed,
            "directories_removed": self.directories_removed,
            "bytes_freed": self.bytes_freed,
            "errors": self.errors
        }


# Patterns for files and directories to clean up
CLEANUP_PATTERNS = {
    "temp_files": [
        "*.tmp", "*.temp", "*.swp", "*.swo", "*.bak", "*.backup",
        "*~", "*.pyc", "*.pyo", "*.pyd", "*.orig", "*.rej"
    ],
    "cache_dirs": [
        "__pycache__", ".pytest_cache", ".mypy_cache", ".coverage",
        ".ruff_cache", ".tox", "node_modules", ".venv", "venv",
        "env", ".env", "dist", "build", "*.egg-info"
    ],
    "log_files": [
        "*.log", "*.log.*", "*.out", "*.err", "debug.txt",
        "error.txt", "output.txt", "*.debug"
    ],
    "ide_files": [
        ".vscode", ".idea", "*.sublime-project", "*.sublime-workspace",
        "*.code-workspace", ".DS_Store", "Thumbs.db", "desktop.ini"
    ],
    "test_artifacts": [
        "htmlcov", "coverage.xml", "pytest_cache", "test-results",
        "test-reports", "*.cover", "*.coverage", "nosetests.xml"
    ]
}


def get_file_size(file_path: Path) -> int:
    """Get file size safely."""
    try:
        return file_path.stat().st_size
    except (OSError, FileNotFoundError):
        return 0


def get_directory_size(dir_path: Path) -> int:
    """Get total size of directory and its contents."""
    total_size = 0
    try:
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                total_size += get_file_size(file_path)
    except (OSError, FileNotFoundError):
        pass
    return total_size


def cleanup_repository(dry_run: bool = False, verbose: bool = False) -> Tuple[List[str], CleanupMetrics]:
    """
    Perform comprehensive cleanup operations on the repository.

    Args:
        dry_run (bool): If True, performs a dry-run without making changes.
        verbose (bool): If True, provides detailed output.

    Returns:
        Tuple[List[str], CleanupMetrics]: A list of cleaned items and metrics.
    """
    cleaned_items = []
    metrics = CleanupMetrics()
    metrics.start()
    
    project_root = Path.cwd()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        
        # Clean temporary files
        task = progress.add_task("Cleaning temporary files...", total=None)
        for pattern in CLEANUP_PATTERNS["temp_files"]:
            for file_path in project_root.rglob(pattern):
                if file_path.is_file():
                    try:
                        size = get_file_size(file_path)
                        if dry_run:
                            if verbose:
                                console.print(f"[yellow]DRY-RUN:[/yellow] Would remove {file_path}")
                        else:
                            file_path.unlink()
                            if verbose:
                                console.print(f"[green]Removed:[/green] {file_path}")
                        
                        cleaned_items.append(str(file_path))
                        metrics.add_file_removed(file_path)
                    except OSError as e:
                        error_msg = f"Failed to remove {file_path}: {e}"
                        metrics.add_error(error_msg)
                        if verbose:
                            console.print(f"[red]Error:[/red] {error_msg}")
        
        # Clean cache directories
        task = progress.add_task("Cleaning cache directories...", total=None)
        for pattern in CLEANUP_PATTERNS["cache_dirs"]:
            for dir_path in project_root.rglob(pattern):
                if dir_path.is_dir():
                    try:
                        size = get_directory_size(dir_path)
                        if dry_run:
                            if verbose:
                                console.print(f"[yellow]DRY-RUN:[/yellow] Would remove directory {dir_path}")
                        else:
                            shutil.rmtree(dir_path)
                            if verbose:
                                console.print(f"[green]Removed directory:[/green] {dir_path}")
                        
                        cleaned_items.append(str(dir_path))
                        metrics.add_directory_removed(dir_path)
                    except OSError as e:
                        error_msg = f"Failed to remove directory {dir_path}: {e}"
                        metrics.add_error(error_msg)
                        if verbose:
                            console.print(f"[red]Error:[/red] {error_msg}")
        
        # Clean log files
        task = progress.add_task("Cleaning log files...", total=None)
        for pattern in CLEANUP_PATTERNS["log_files"]:
            for file_path in project_root.rglob(pattern):
                if file_path.is_file():
                    try:
                        size = get_file_size(file_path)
                        if dry_run:
                            if verbose:
                                console.print(f"[yellow]DRY-RUN:[/yellow] Would remove {file_path}")
                        else:
                            file_path.unlink()
                            if verbose:
                                console.print(f"[green]Removed:[/green] {file_path}")
                        
                        cleaned_items.append(str(file_path))
                        metrics.add_file_removed(file_path)
                    except OSError as e:
                        error_msg = f"Failed to remove {file_path}: {e}"
                        metrics.add_error(error_msg)
                        if verbose:
                            console.print(f"[red]Error:[/red] {error_msg}")
        
        # Clean IDE-specific files (optional)
        task = progress.add_task("Cleaning IDE files...", total=None)
        for pattern in CLEANUP_PATTERNS["ide_files"]:
            for path in project_root.rglob(pattern):
                if path.exists():
                    try:
                        if path.is_file():
                            size = get_file_size(path)
                            if dry_run:
                                if verbose:
                                    console.print(f"[yellow]DRY-RUN:[/yellow] Would remove {path}")
                            else:
                                path.unlink()
                                if verbose:
                                    console.print(f"[green]Removed:[/green] {path}")
                            metrics.add_file_removed(path)
                        elif path.is_dir():
                            size = get_directory_size(path)
                            if dry_run:
                                if verbose:
                                    console.print(f"[yellow]DRY-RUN:[/yellow] Would remove directory {path}")
                            else:
                                shutil.rmtree(path)
                                if verbose:
                                    console.print(f"[green]Removed directory:[/green] {path}")
                            metrics.add_directory_removed(path)
                        
                        cleaned_items.append(str(path))
                    except OSError as e:
                        error_msg = f"Failed to remove {path}: {e}"
                        metrics.add_error(error_msg)
                        if verbose:
                            console.print(f"[red]Error:[/red] {error_msg}")
        
        # Clean test artifacts
        task = progress.add_task("Cleaning test artifacts...", total=None)
        for pattern in CLEANUP_PATTERNS["test_artifacts"]:
            for path in project_root.rglob(pattern):
                if path.exists():
                    try:
                        if path.is_file():
                            size = get_file_size(path)
                            if dry_run:
                                if verbose:
                                    console.print(f"[yellow]DRY-RUN:[/yellow] Would remove {path}")
                            else:
                                path.unlink()
                                if verbose:
                                    console.print(f"[green]Removed:[/green] {path}")
                            metrics.add_file_removed(path)
                        elif path.is_dir():
                            size = get_directory_size(path)
                            if dry_run:
                                if verbose:
                                    console.print(f"[yellow]DRY-RUN:[/yellow] Would remove directory {path}")
                            else:
                                shutil.rmtree(path)
                                if verbose:
                                    console.print(f"[green]Removed directory:[/green] {path}")
                            metrics.add_directory_removed(path)
                        
                        cleaned_items.append(str(path))
                    except OSError as e:
                        error_msg = f"Failed to remove {path}: {e}"
                        metrics.add_error(error_msg)
                        if verbose:
                            console.print(f"[red]Error:[/red] {error_msg}")
    
    metrics.stop()
    return cleaned_items, metrics


@app.command()
def main(dry_run: bool = typer.Option(False, help="Perform a dry-run without making changes."),
         output: str = typer.Option("cleanup_report.json", help="Output JSON report file.")):
    """
    Main function for cleanup.
    """
    metrics = CleanupMetrics()
    metrics.start()

    typer.echo("Starting repository cleanup...")
    cleaned_items = cleanup_repository(dry_run)

    metrics.stop()

    typer.echo("Cleanup completed.")
    typer.echo(f"Execution Time: {metrics.duration} seconds")

    # Generate JSON report
    report = {
        "dry_run": dry_run,
        "execution_time": metrics.duration,
        "cleaned_items": cleaned_items
    }

    with open(output, 'w') as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    app()
