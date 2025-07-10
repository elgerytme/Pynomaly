"""
CLI Help Formatting Utilities

Provides standardized help text formatting across all CLI commands
to ensure consistent user experience.
"""

from __future__ import annotations

import textwrap

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class HelpFormatter:
    """Standardized help text formatter for CLI commands."""

    def __init__(self):
        self.console = Console()

    @staticmethod
    def format_command_help(
        description: str,
        usage_examples: list[str] | None = None,
        notes: str | None = None,
        max_width: int = 80,
    ) -> str:
        """Format command help with consistent structure."""
        # Wrap description to consistent width
        wrapped_desc = textwrap.fill(description, width=max_width)

        # Build help text
        help_parts = [wrapped_desc]

        if usage_examples:
            help_parts.append("")
            help_parts.append("Examples:")
            for example in usage_examples:
                help_parts.append(f"  {example}")

        if notes:
            help_parts.append("")
            help_parts.append(f"Note: {notes}")

        return "\n".join(help_parts)

    @staticmethod
    def format_option_help(description: str, max_width: int = 60) -> str:
        """Format option help text."""
        return textwrap.fill(description, width=max_width)

    @staticmethod
    def create_examples_table(examples: dict[str, str]) -> Table:
        """Create a rich table for command examples."""
        table = Table(
            title="Usage Examples", show_header=True, header_style="bold cyan"
        )
        table.add_column("Command", style="green", width=40)
        table.add_column("Description", style="dim", width=40)

        for command, description in examples.items():
            table.add_row(command, description)

        return table

    @staticmethod
    def create_tips_panel(tips: list[str]) -> Panel:
        """Create a rich panel with helpful tips."""
        tips_text = "\n".join(f"• {tip}" for tip in tips)
        return Panel(tips_text, title="💡 Tips", border_style="blue", padding=(1, 2))


# Standard help text templates
STANDARD_HELP_TEXTS = {
    # Main commands
    "autonomous": {
        "help": "🤖 Autonomous anomaly detection with automatic configuration",
        "description": "Automatically configure and run anomaly detection with minimal input. "
        "Analyzes your data characteristics and selects optimal algorithms.",
        "examples": [
            "pynomaly auto detect data.csv --output results.csv",
            "pynomaly auto profile data.csv --verbose",
            "pynomaly auto quick data.csv --contamination 0.05",
        ],
    },
    "automl": {
        "help": "🧠 Advanced AutoML and hyperparameter optimization",
        "description": "Automated machine learning for anomaly detection with intelligent "
        "hyperparameter tuning and algorithm selection.",
        "examples": [
            "pynomaly automl run data.csv --algorithms IF LOF OCSVM",
            "pynomaly automl tune detector_id --metric f1",
            "pynomaly automl compare experiment_results.json",
        ],
    },
    "detector": {
        "help": "🔧 Create and manage anomaly detectors",
        "description": "Create, configure, and manage anomaly detection models. "
        "Supports multiple algorithms and hyperparameter configurations.",
        "examples": [
            "pynomaly detector create --name my_detector --algorithm IsolationForest",
            "pynomaly detector list --filter trained",
            "pynomaly detector show detector_id --verbose",
        ],
    },
    "dataset": {
        "help": "📊 Load and manage datasets",
        "description": "Load, validate, and manage datasets for anomaly detection. "
        "Supports CSV, Parquet, and JSON formats with automatic validation.",
        "examples": [
            "pynomaly dataset load data.csv --name my_data",
            "pynomaly dataset list --filter large",
            "pynomaly dataset show dataset_id --stats",
        ],
    },
    "data": {
        "help": "🔄 Data preprocessing and transformation",
        "description": "Clean, transform, and prepare data for anomaly detection. "
        "Includes missing value handling, scaling, and feature engineering.",
        "examples": [
            "pynomaly data clean dataset_id --missing drop --outliers clip",
            "pynomaly data transform dataset_id --scaling standard",
            "pynomaly data pipeline dataset_id --config preprocessing.yml",
        ],
    },
    "detect": {
        "help": "🎯 Run anomaly detection",
        "description": "Train detectors and run anomaly detection on datasets. "
        "Supports batch processing and real-time detection.",
        "examples": [
            "pynomaly detect train detector_id dataset_id",
            "pynomaly detect run detector_id dataset_id --output results.csv",
            "pynomaly detect evaluate detector_id dataset_id --cv --folds 5",
        ],
    },
    "explainability": {
        "help": "🔍 Model explainability and interpretability",
        "description": "Generate explanations for anomaly detection results. "
        "Provides feature importance, local explanations, and bias analysis.",
        "examples": [
            "pynomaly explainability global detector_id dataset_id",
            "pynomaly explainability local result_id --sample-id 123",
            "pynomaly explainability bias detector_id dataset_id --protected age",
        ],
    },
    "export": {
        "help": "📤 Export results to external platforms",
        "description": "Export detection results to business intelligence platforms. "
        "Supports Excel, PowerBI, Tableau, and custom formats.",
        "examples": [
            "pynomaly export excel results.json output.xlsx",
            "pynomaly export powerbi results.json --connection default",
            "pynomaly export csv results.json output.csv --format detailed",
        ],
    },
    "server": {
        "help": "🌐 Manage API server",
        "description": "Start and manage the Pynomaly REST API server for web applications. "
        "Provides endpoints for all CLI functionality.",
        "examples": [
            "pynomaly server start --host 0.0.0.0 --port 8000",
            "pynomaly server status",
            "pynomaly server restart --workers 4",
        ],
    },
    "validate": {
        "help": "✅ Data and model validation",
        "description": "Validate data quality, model performance, and system health. "
        "Includes comprehensive checks and GitHub integration.",
        "examples": [
            "pynomaly validate data dataset_id --checks all",
            "pynomaly validate model detector_id --metrics accuracy recall",
            "pynomaly validate system --github-report",
        ],
    },
}

# Standard option help texts
STANDARD_OPTION_HELP = {
    "verbose": "Enable detailed output with progress information and diagnostics",
    "quiet": "Suppress all output except errors and critical information",
    "output": "Specify output file path for results (supports CSV, JSON, Excel)",
    "format": "Output format: csv, json, excel, parquet",
    "force": "Force operation without confirmation prompts",
    "dry_run": "Show what would be done without making changes",
    "config": "Path to configuration file (YAML or JSON format)",
    "contamination": "Expected proportion of anomalies (0.0-1.0, default: 0.1)",
    "algorithm": "Anomaly detection algorithm: IsolationForest, LOF, OCSVM, etc.",
    "name": "Human-readable name for the resource",
    "filter": "Filter results by status, type, or other criteria",
    "limit": "Maximum number of results to return",
    "sort": "Sort results by field: name, date, score, etc.",
    "cv": "Enable cross-validation for model evaluation",
    "folds": "Number of cross-validation folds (default: 5)",
    "metrics": "Evaluation metrics: precision, recall, f1, auc_roc, auc_pr",
    "threshold": "Decision threshold for anomaly classification",
    "seed": "Random seed for reproducible results",
    "workers": "Number of parallel workers (default: auto-detect)",
    "memory": "Memory limit for processing (e.g., 4GB, 512MB)",
    "gpu": "Enable GPU acceleration if available",
    "cache": "Enable result caching for faster repeated operations",
    "backup": "Create backup before destructive operations",
    "interactive": "Use interactive mode with prompts and confirmations",
}


def get_standard_help(command: str) -> dict[str, str]:
    """Get standardized help text for a command."""
    return STANDARD_HELP_TEXTS.get(
        command,
        {
            "help": f"Manage {command} operations",
            "description": f"Perform {command}-related operations in Pynomaly.",
            "examples": [f"pynomaly {command} --help"],
        },
    )


def get_option_help(option: str) -> str:
    """Get standardized help text for an option."""
    return STANDARD_OPTION_HELP.get(option, f"Configure {option} setting")


def format_rich_help(
    title: str,
    description: str,
    examples: list[str] | None = None,
    tips: list[str] | None = None,
) -> None:
    """Display rich-formatted help with examples and tips."""
    console = Console()

    # Main description
    console.print(f"\n[bold blue]{title}[/bold blue]")
    console.print(description)

    # Examples section
    if examples:
        console.print("\n[bold green]Examples:[/bold green]")
        for example in examples:
            console.print(f"  [cyan]{example}[/cyan]")

    # Tips section
    if tips:
        console.print()
        console.print(HelpFormatter.create_tips_panel(tips))

    console.print()
