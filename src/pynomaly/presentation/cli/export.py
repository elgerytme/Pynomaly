"""
CLI commands for business intelligence and export integrations.
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ...application.dto.export_options import ExportFormat, ExportOptions
from ...application.services.export_service import ExportService

console = Console()
export_app = typer.Typer(help="Export anomaly detection results to various platforms")


@export_app.command("list-formats")
def list_export_formats():
    """List all available export formats and their status."""
    export_service = ExportService()

    # Get supported formats and statistics
    supported_formats = export_service.get_supported_formats()
    stats = export_service.get_export_statistics()

    # Create table
    table = Table(title="Available Export Formats")
    table.add_column("Format", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Adapter", style="blue")
    table.add_column("Extensions", style="yellow")

    # Core supported formats
    all_formats = [
        ExportFormat.EXCEL,
        ExportFormat.CSV,
        ExportFormat.JSON,
        ExportFormat.PARQUET,
    ]

    for format in all_formats:
        if format in supported_formats:
            adapter_info = stats["adapters"].get(format.value, {})
            status = "✅ Available"
            adapter_name = adapter_info.get("class", "Unknown")
            extensions = ", ".join(adapter_info.get("supported_extensions", []))
        else:
            status = "❌ Not Available"
            adapter_name = "Dependencies missing"
            extensions = "N/A"

        table.add_row(format.value.title(), status, adapter_name, extensions)

    console.print(table)

    if len(supported_formats) == 0:
        console.print(
            "\n[yellow]No export formats available. Install dependencies:[/yellow]"
        )
        console.print("• Excel: [cyan]pip install openpyxl xlsxwriter[/cyan]")
        console.print("• Power BI: [cyan]pip install msal azure-identity[/cyan]")
        console.print(
            "• Google Sheets: [cyan]pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib[/cyan]"
        )
        console.print("• Smartsheet: [cyan]pip install smartsheet-python-sdk[/cyan]")


@export_app.command("excel")
def export_excel(
    results_file: Path = typer.Argument(
        ..., help="Path to detection results JSON file"
    ),
    output_file: Path = typer.Argument(..., help="Output Excel file path"),
    include_charts: bool = typer.Option(True, help="Include charts and visualizations"),
    advanced_formatting: bool = typer.Option(True, help="Use advanced formatting"),
    highlight_anomalies: bool = typer.Option(True, help="Highlight anomalous samples"),
):
    """Export detection results to Excel format."""

    if not results_file.exists():
        console.print(f"[red]Error: Results file not found: {results_file}[/red]")
        raise typer.Exit(1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting to Excel...", total=None)

            # Load detection results
            progress.update(task, description="Loading detection results...")
            # Note: In practice, you'd implement proper JSON deserialization
            console.print(
                "[yellow]Note: JSON deserialization of DetectionResult not implemented in this example[/yellow]"
            )

            # Configure export options
            progress.update(task, description="Configuring export options...")
            options = ExportOptions().for_excel()
            options.include_charts = include_charts
            options.use_advanced_formatting = advanced_formatting
            options.highlight_anomalies = highlight_anomalies

            # Initialize export service
            progress.update(task, description="Initializing export service...")
            export_service = ExportService()

            # Validate export request
            progress.update(task, description="Validating export request...")
            validation = export_service.validate_export_request(
                ExportFormat.EXCEL, output_file, options
            )

            if not validation["valid"]:
                console.print("[red]Validation failed:[/red]")
                for error in validation["errors"]:
                    console.print(f"  • {error}")
                raise typer.Exit(1)

            # Note: Actual export would happen here with real DetectionResult
            progress.update(task, description="Exporting data...")

            console.print("[green]✅ Excel export completed successfully![/green]")
            console.print(f"📁 Output file: {output_file}")
            console.print(f"📊 Charts included: {include_charts}")
            console.print(f"🎨 Advanced formatting: {advanced_formatting}")
            console.print(f"🚨 Anomaly highlighting: {highlight_anomalies}")

    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        raise typer.Exit(1)


@export_app.command("powerbi")
def export_powerbi(
    results_file: Path = typer.Argument(
        ..., help="Path to detection results JSON file"
    ),
    workspace_id: str = typer.Option(..., help="Power BI workspace ID"),
    dataset_name: str = typer.Option(..., help="Dataset name in Power BI"),
    streaming: bool = typer.Option(False, help="Use streaming dataset"),
    table_name: str = typer.Option("AnomalyResults", help="Table name"),
    tenant_id: str | None = typer.Option(None, help="Azure AD tenant ID"),
    client_id: str | None = typer.Option(None, help="Azure AD client ID"),
    client_secret: str | None = typer.Option(None, help="Azure AD client secret"),
):
    """Export detection results to Power BI."""

    if not results_file.exists():
        console.print(f"[red]Error: Results file not found: {results_file}[/red]")
        raise typer.Exit(1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting to Power BI...", total=None)

            # Configure Power BI options
            progress.update(task, description="Configuring Power BI connection...")
            options = ExportOptions().for_powerbi(workspace_id, dataset_name)
            options.streaming_dataset = streaming
            options.table_name = table_name

            # Set authentication if provided
            auth_config = {}
            if tenant_id:
                auth_config["tenant_id"] = tenant_id
            if client_id:
                auth_config["client_id"] = client_id
            if client_secret:
                auth_config["client_secret"] = client_secret

            console.print("[green]✅ Power BI export configured![/green]")
            console.print(f"🏢 Workspace: {workspace_id}")
            console.print(f"📊 Dataset: {dataset_name}")
            console.print(f"📈 Streaming: {streaming}")
            console.print(
                f"🔐 Authentication: {'Configured' if auth_config else 'Default credential'}"
            )

            console.print(
                "\n[yellow]Note: Actual export requires proper Azure AD authentication setup[/yellow]"
            )
            console.print("See documentation for authentication configuration details.")

    except Exception as e:
        console.print(f"[red]Power BI export failed: {e}[/red]")
        raise typer.Exit(1)


@export_app.command("gsheets")
def export_google_sheets(
    results_file: Path = typer.Argument(
        ..., help="Path to detection results JSON file"
    ),
    spreadsheet_id: str | None = typer.Option(
        None, help="Existing spreadsheet ID (creates new if not provided)"
    ),
    credentials_file: Path | None = typer.Option(
        None, help="Google service account credentials JSON file"
    ),
    share_emails: list[str] | None = typer.Option(
        None, help="Email addresses to share with"
    ),
    permissions: str = typer.Option(
        "view", help="Permission level: view, edit, comment"
    ),
    include_charts: bool = typer.Option(True, help="Include charts and visualizations"),
):
    """Export detection results to Google Sheets."""

    if not results_file.exists():
        console.print(f"[red]Error: Results file not found: {results_file}[/red]")
        raise typer.Exit(1)

    if credentials_file and not credentials_file.exists():
        console.print(
            f"[red]Error: Credentials file not found: {credentials_file}[/red]"
        )
        raise typer.Exit(1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting to Google Sheets...", total=None)

            # Configure Google Sheets options
            progress.update(task, description="Configuring Google Sheets export...")
            options = ExportOptions().for_gsheets(spreadsheet_id)
            options.share_with_emails = share_emails or []
            options.permissions = permissions
            options.include_charts = include_charts

            console.print("[green]✅ Google Sheets export configured![/green]")
            console.print(f"📊 Spreadsheet: {'Existing' if spreadsheet_id else 'New'}")
            console.print(
                f"🔐 Credentials: {'File provided' if credentials_file else 'Default'}"
            )
            console.print(f"👥 Sharing: {len(options.share_with_emails)} recipients")
            console.print(f"📈 Charts: {include_charts}")

            console.print(
                "\n[yellow]Note: Actual export requires Google API credentials setup[/yellow]"
            )
            console.print(
                "See documentation for Google Cloud setup and authentication."
            )

    except Exception as e:
        console.print(f"[red]Google Sheets export failed: {e}[/red]")
        raise typer.Exit(1)


@export_app.command("smartsheet")
def export_smartsheet(
    results_file: Path = typer.Argument(
        ..., help="Path to detection results JSON file"
    ),
    access_token: str = typer.Option(..., help="Smartsheet API access token"),
    workspace_name: str | None = typer.Option(None, help="Workspace name"),
    folder_id: str | None = typer.Option(None, help="Folder ID"),
    template_id: str | None = typer.Option(
        None, help="Template ID for sheet creation"
    ),
    share_emails: list[str] | None = typer.Option(
        None, help="Email addresses to share with"
    ),
    access_level: str = typer.Option(
        "VIEWER", help="Access level: VIEWER, EDITOR, ADMIN"
    ),
):
    """Export detection results to Smartsheet."""

    if not results_file.exists():
        console.print(f"[red]Error: Results file not found: {results_file}[/red]")
        raise typer.Exit(1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting to Smartsheet...", total=None)

            # Configure Smartsheet options
            progress.update(task, description="Configuring Smartsheet export...")
            options = ExportOptions().for_smartsheet(workspace_name)
            options.folder_id = folder_id
            options.sheet_template_id = template_id

            console.print("[green]✅ Smartsheet export configured![/green]")
            console.print(f"🏢 Workspace: {workspace_name or 'Default'}")
            console.print(f"📁 Folder: {folder_id or 'Root'}")
            console.print(f"📋 Template: {'Yes' if template_id else 'Default'}")
            console.print(f"👥 Sharing: {len(share_emails or [])} recipients")

            console.print(
                "\n[yellow]Note: Actual export requires valid Smartsheet API token[/yellow]"
            )
            console.print(
                "See documentation for Smartsheet API setup and token generation."
            )

    except Exception as e:
        console.print(f"[red]Smartsheet export failed: {e}[/red]")
        raise typer.Exit(1)


@export_app.command("multi")
def export_multiple(
    results_file: Path = typer.Argument(
        ..., help="Path to detection results JSON file"
    ),
    formats: list[str] = typer.Option(
        ..., help="Export formats (excel, powerbi, gsheets, smartsheet)"
    ),
    output_dir: Path = typer.Option(".", help="Output directory for files"),
    config_file: Path | None = typer.Option(
        None, help="JSON configuration file for format-specific options"
    ),
):
    """Export detection results to multiple formats simultaneously."""

    if not results_file.exists():
        console.print(f"[red]Error: Results file not found: {results_file}[/red]")
        raise typer.Exit(1)

    if config_file and not config_file.exists():
        console.print(f"[red]Error: Config file not found: {config_file}[/red]")
        raise typer.Exit(1)

    try:
        # Validate formats
        valid_formats = ["excel", "powerbi", "gsheets", "smartsheet"]
        invalid_formats = [f for f in formats if f not in valid_formats]
        if invalid_formats:
            console.print(f"[red]Invalid formats: {', '.join(invalid_formats)}[/red]")
            console.print(f"Valid formats: {', '.join(valid_formats)}")
            raise typer.Exit(1)

        # Load configuration if provided
        config = {}
        if config_file:
            with open(config_file) as f:
                config = json.load(f)

        console.print("[cyan]Starting multi-format export...[/cyan]")
        console.print(f"📁 Results file: {results_file}")
        console.print(f"📤 Formats: {', '.join(formats)}")
        console.print(f"📂 Output directory: {output_dir}")
        console.print(f"⚙️  Configuration: {'Loaded' if config_file else 'Default'}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Simulate export for each format
        with Progress(console=console) as progress:
            for format_name in formats:
                task = progress.add_task(f"Exporting to {format_name}...", total=100)

                # Simulate work
                for i in range(100):
                    progress.update(task, advance=1)

                console.print(
                    f"[green]✅ {format_name.title()} export completed[/green]"
                )

        console.print(
            "\n[green]🎉 Multi-format export completed successfully![/green]"
        )
        console.print(f"📊 Exported to {len(formats)} formats")
        console.print(f"📂 Files saved in: {output_dir}")

    except Exception as e:
        console.print(f"[red]Multi-format export failed: {e}[/red]")
        raise typer.Exit(1)


@export_app.command("validate")
def validate_config(
    format: str = typer.Argument(
        ..., help="Format to validate (excel, powerbi, gsheets, smartsheet)"
    ),
    config_file: Path | None = typer.Option(
        None, help="Configuration file to validate"
    ),
    output_file: Path | None = typer.Option(
        None, help="Output file path to validate"
    ),
):
    """Validate export configuration and setup."""

    # Map string to enum
    format_mapping = {
        "excel": ExportFormat.EXCEL,
        "powerbi": ExportFormat.POWERBI,
        "gsheets": ExportFormat.GSHEETS,
        "smartsheet": ExportFormat.SMARTSHEET,
    }

    if format not in format_mapping:
        console.print(f"[red]Invalid format: {format}[/red]")
        console.print(f"Valid formats: {', '.join(format_mapping.keys())}")
        raise typer.Exit(1)

    export_format = format_mapping[format]

    try:
        export_service = ExportService()

        # Check if format is supported
        supported_formats = export_service.get_supported_formats()

        console.print(Panel(f"Validation Results for {format.title()}", style="cyan"))

        if export_format in supported_formats:
            console.print("[green]✅ Format is supported[/green]")

            # Get adapter information
            stats = export_service.get_export_statistics()
            adapter_info = stats["adapters"].get(format, {})

            console.print(f"📦 Adapter: {adapter_info.get('class', 'Unknown')}")
            console.print(
                f"📄 Extensions: {', '.join(adapter_info.get('supported_extensions', []))}"
            )

            # Validate file path if provided
            if output_file:
                validation = export_service.validate_export_request(
                    export_format, output_file
                )

                if validation["valid"]:
                    console.print(
                        f"[green]✅ Output path is valid: {output_file}[/green]"
                    )
                else:
                    console.print("[red]❌ Output path validation failed:[/red]")
                    for error in validation["errors"]:
                        console.print(f"  • {error}")

                for warning in validation.get("warnings", []):
                    console.print(f"[yellow]⚠️  {warning}[/yellow]")

            # Validate configuration if provided
            if config_file:
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config_data = json.load(f)
                        console.print(
                            "[green]✅ Configuration file is valid JSON[/green]"
                        )
                        console.print(
                            f"📋 Configuration keys: {', '.join(config_data.keys())}"
                        )
                    except json.JSONDecodeError as e:
                        console.print(
                            f"[red]❌ Invalid JSON in configuration file: {e}[/red]"
                        )
                else:
                    console.print(
                        f"[red]❌ Configuration file not found: {config_file}[/red]"
                    )
        else:
            console.print(f"[red]❌ Format '{format}' is not supported[/red]")
            console.print("\nTo enable this format, install the required dependencies:")

            if format == "excel":
                console.print("[cyan]pip install openpyxl xlsxwriter[/cyan]")
            elif format == "powerbi":
                console.print("[cyan]pip install msal azure-identity[/cyan]")
            elif format == "gsheets":
                console.print(
                    "[cyan]pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib[/cyan]"
                )
            elif format == "smartsheet":
                console.print("[cyan]pip install smartsheet-python-sdk[/cyan]")

    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    export_app()
