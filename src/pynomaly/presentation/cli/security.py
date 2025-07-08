"""CLI commands for security and compliance functionality."""

import asyncio
import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pynomaly.application.services.security_compliance_service import (
    EncryptionContext,
    SecurityComplianceService,
    SecurityConfiguration,
)
from pynomaly.domain.entities.security_compliance import (
    ComplianceFramework,
    DataClassification,
)
from pynomaly.infrastructure.config.container import Container

console = Console()


@click.group(name="security")
def security_commands():
    """Security and compliance management commands."""
    pass


@security_commands.command()
@click.option(
    "--frameworks",
    type=click.Choice(["soc2", "gdpr", "hipaa", "pci_dss", "iso27001", "nist", "ccpa"]),
    multiple=True,
    default=["soc2", "gdpr"],
    help="Compliance frameworks to assess",
)
@click.option("--output-path", help="Path to save assessment reports")
@click.option("--scope", help="Assessment scope description")
@click.option("--detailed", is_flag=True, help="Generate detailed assessment report")
def assess_compliance(
    frameworks: list[str],
    output_path: str | None,
    scope: str | None,
    detailed: bool,
):
    """Assess compliance with regulatory frameworks."""

    async def run_assessment():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize security service
            task1 = progress.add_task(
                "Initializing security compliance service...", total=None
            )
            storage_path = Path("./security")
            security_service = SecurityComplianceService(storage_path)
            progress.update(task1, completed=True)

            # Run assessments
            task2 = progress.add_task("Running compliance assessments...", total=None)

            try:
                framework_enums = [ComplianceFramework(fw) for fw in frameworks]
                reports = await security_service.generate_compliance_report(
                    framework_enums
                )

                progress.update(task2, completed=True)

                # Display results
                _display_compliance_reports(reports, detailed)

                # Save reports if requested
                if output_path:
                    await _save_compliance_reports(reports, output_path)
                    console.print(
                        f"[green]Compliance reports saved to {output_path}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error during compliance assessment: {e}[/red]")
                return

    asyncio.run(run_assessment())


@security_commands.command()
@click.option("--data-file", required=True, help="Path to data file for encryption")
@click.option("--output-file", help="Path for encrypted output")
@click.option(
    "--classification",
    type=click.Choice(
        ["public", "internal", "confidential", "restricted", "personal", "phi"]
    ),
    default="confidential",
    help="Data classification level",
)
@click.option(
    "--frameworks",
    type=click.Choice(["soc2", "gdpr", "hipaa", "pci_dss"]),
    multiple=True,
    help="Applicable compliance frameworks",
)
def encrypt_data(
    data_file: str,
    output_file: str | None,
    classification: str,
    frameworks: list[str],
):
    """Encrypt sensitive data with compliance controls."""

    async def run_encryption():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize security service
            task1 = progress.add_task("Initializing encryption service...", total=None)
            storage_path = Path("./security")
            security_service = SecurityComplianceService(storage_path)
            progress.update(task1, completed=True)

            # Load and encrypt data
            task2 = progress.add_task("Encrypting data...", total=None)

            try:
                # Read data file
                with open(data_file, "rb") as f:
                    data = f.read()

                # Create encryption context
                context = EncryptionContext(
                    data_classification=DataClassification(classification),
                    compliance_frameworks=[
                        ComplianceFramework(fw) for fw in frameworks
                    ],
                )

                # Encrypt data
                (
                    encrypted_data,
                    encryption_context,
                ) = await security_service.encrypt_data(data, context)

                progress.update(task2, completed=True)

                # Save encrypted data
                output_path = output_file or f"{data_file}.encrypted"
                with open(output_path, "wb") as f:
                    f.write(encrypted_data)

                # Save encryption metadata
                metadata_path = f"{output_path}.metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(encryption_context.get_encryption_metadata(), f, indent=2)

                console.print(
                    Panel(
                        f"[green]Data encrypted successfully[/green]\n"
                        f"Input file: {data_file}\n"
                        f"Output file: {output_path}\n"
                        f"Metadata file: {metadata_path}\n"
                        f"Classification: {classification}\n"
                        f"Key ID: {encryption_context.key_id}",
                        title="Encryption Complete",
                    )
                )

            except Exception as e:
                console.print(f"[red]Error during encryption: {e}[/red]")
                return

    asyncio.run(run_encryption())


@security_commands.command()
@click.option(
    "--subject-data", required=True, help="JSON file with data subject information"
)
@click.option(
    "--processing-purposes", required=True, help="Comma-separated processing purposes"
)
@click.option("--consent-given", is_flag=True, help="Whether consent was given")
def register_data_subject(
    subject_data: str, processing_purposes: str, consent_given: bool
):
    """Register data subject for GDPR compliance."""

    async def run_registration():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize security service
            task1 = progress.add_task(
                "Initializing GDPR compliance service...", total=None
            )
            storage_path = Path("./security")
            security_service = SecurityComplianceService(storage_path)
            progress.update(task1, completed=True)

            # Register data subject
            task2 = progress.add_task("Registering data subject...", total=None)

            try:
                # Load subject data
                with open(subject_data) as f:
                    data = json.load(f)

                # Parse processing purposes
                purposes = [p.strip() for p in processing_purposes.split(",")]

                # Register data subject
                subject_id = await security_service.register_data_subject(
                    data, purposes, consent_given
                )

                progress.update(task2, completed=True)

                console.print(
                    Panel(
                        f"[green]Data subject registered successfully[/green]\n"
                        f"Subject ID: {subject_id}\n"
                        f"Data types: {', '.join(data.keys())}\n"
                        f"Processing purposes: {', '.join(purposes)}\n"
                        f"Consent given: {consent_given}",
                        title="GDPR Registration Complete",
                    )
                )

            except Exception as e:
                console.print(f"[red]Error during data subject registration: {e}[/red]")
                return

    asyncio.run(run_registration())


@security_commands.command()
@click.option("--subject-id", required=True, help="Data subject UUID")
@click.option(
    "--request-type",
    type=click.Choice(["access", "rectification", "erasure", "portability"]),
    required=True,
    help="Type of GDPR request",
)
@click.option("--output-file", help="File to save request response")
def gdpr_request(subject_id: str, request_type: str, output_file: str | None):
    """Handle GDPR data subject rights requests."""

    async def run_gdpr_request():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize security service
            task1 = progress.add_task("Initializing GDPR service...", total=None)
            storage_path = Path("./security")
            security_service = SecurityComplianceService(storage_path)
            progress.update(task1, completed=True)

            # Process GDPR request
            task2 = progress.add_task(
                f"Processing {request_type} request...", total=None
            )

            try:
                from uuid import UUID

                subject_uuid = UUID(subject_id)

                result = await security_service.handle_data_subject_request(
                    subject_uuid, request_type
                )

                progress.update(task2, completed=True)

                # Display result
                console.print(
                    Panel(
                        f"[green]GDPR request processed successfully[/green]\n"
                        f"Request type: {request_type}\n"
                        f"Status: {result['status']}\n"
                        f"Subject ID: {subject_id}",
                        title="GDPR Request Result",
                    )
                )

                if request_type == "access" and "data" in result:
                    _display_data_subject_info(result["data"])

                # Save result if requested
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(result, f, indent=2)
                    console.print(
                        f"[green]Request result saved to {output_file}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error processing GDPR request: {e}[/red]")
                return

    asyncio.run(run_gdpr_request())


@security_commands.command()
@click.option("--access-logs", required=True, help="JSON file with access logs")
@click.option("--output-file", help="File to save breach detection results")
@click.option(
    "--alert-threshold", type=int, default=3, help="Minimum incidents for alert"
)
def detect_breach(access_logs: str, output_file: str | None, alert_threshold: int):
    """Detect potential data breaches from access logs."""

    async def run_breach_detection():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize security service
            task1 = progress.add_task("Initializing breach detection...", total=None)
            storage_path = Path("./security")
            security_service = SecurityComplianceService(storage_path)
            progress.update(task1, completed=True)

            # Analyze access logs
            task2 = progress.add_task("Analyzing access patterns...", total=None)

            try:
                # Load access logs
                with open(access_logs) as f:
                    logs = json.load(f)

                # Detect breaches
                incidents = await security_service.detect_data_breach(logs)

                progress.update(task2, completed=True)

                # Display results
                _display_security_incidents(incidents, alert_threshold)

                # Save results if requested
                if output_file:
                    incidents_data = [
                        incident.get_incident_summary() for incident in incidents
                    ]
                    with open(output_file, "w") as f:
                        json.dump(incidents_data, f, indent=2)
                    console.print(
                        f"[green]Breach detection results saved to {output_file}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error during breach detection: {e}[/red]")
                return

    asyncio.run(run_breach_detection())


@app.command()
def anonymize_data(
    data_file: Annotated[
        str,
        typer.Option("--data-file", help="Data file to anonymize")
    ],
    rules_file: Annotated[
        str,
        typer.Option("--rules-file", help="JSON file with anonymization rules")
    ],
    output_file: Annotated[
        str | None,
        typer.Option("--output-file", help="File to save anonymized data")
    ] = None,
):
    """Anonymize data for privacy compliance."""

    async def run_anonymization():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize security service
            task1 = progress.add_task("Initializing privacy protection...", total=None)
            storage_path = Path("./security")
            security_service = SecurityComplianceService(storage_path)
            progress.update(task1, completed=True)

            # Anonymize data
            task2 = progress.add_task("Anonymizing data...", total=None)

            try:
                # Load data and rules
                with open(data_file) as f:
                    data = json.load(f)

                with open(rules_file) as f:
                    rules = json.load(f)

                # Anonymize data
                anonymized_data = await security_service.anonymize_data(data, rules)

                progress.update(task2, completed=True)

                # Save anonymized data
                output_path = output_file or f"{data_file}.anonymized.json"
                with open(output_path, "w") as f:
                    json.dump(anonymized_data, f, indent=2)

                console.print(
                    Panel(
                        f"[green]Data anonymized successfully[/green]\n"
                        f"Input file: {data_file}\n"
                        f"Output file: {output_path}\n"
                        f"Fields anonymized: {len(rules)}\n"
                        f"Anonymization rules: {', '.join(rules.keys())}",
                        title="Anonymization Complete",
                    )
                )

                # Show before/after comparison
                _display_anonymization_comparison(data, anonymized_data, rules)

            except Exception as e:
                console.print(f"[red]Error during anonymization: {e}[/red]")
                return

    asyncio.run(run_anonymization())


@security_commands.command()
@click.option("--config-file", help="Security configuration file")
@click.option("--output-file", help="File to save security status")
def status(config_file: str | None, output_file: str | None):
    """Get overall security and compliance status."""

    async def run_status_check():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize security service
            task1 = progress.add_task("Checking security status...", total=None)
            storage_path = Path("./security")

            # Load configuration if provided
            config = None
            if config_file:
                with open(config_file) as f:
                    config_data = json.load(f)
                config = SecurityConfiguration(**config_data)

            security_service = SecurityComplianceService(storage_path, config)
            progress.update(task1, completed=True)

            # Get security status
            task2 = progress.add_task("Gathering security metrics...", total=None)

            try:
                status_data = await security_service.get_security_status()

                progress.update(task2, completed=True)

                # Display status
                _display_security_status(status_data)

                # Save status if requested
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(status_data, f, indent=2)
                    console.print(
                        f"[green]Security status saved to {output_file}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error getting security status: {e}[/red]")
                return

    asyncio.run(run_status_check())


@security_commands.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be cleaned without actually doing it",
)
@click.option("--force", is_flag=True, help="Force cleanup without confirmation")
def cleanup(dry_run: bool, force: bool):
    """Clean up expired data per retention policies."""

    async def run_cleanup():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize security service
            task1 = progress.add_task("Initializing cleanup service...", total=None)
            storage_path = Path("./security")
            security_service = SecurityComplianceService(storage_path)
            progress.update(task1, completed=True)

            # Run cleanup
            task2 = progress.add_task(
                (
                    "Analyzing data for cleanup..."
                    if dry_run
                    else "Cleaning up expired data..."
                ),
                total=None,
            )

            try:
                if dry_run:
                    # Simulate cleanup
                    console.print(
                        "[yellow]DRY RUN MODE - No data will be actually deleted[/yellow]"
                    )

                    # Show what would be cleaned
                    expired_count = len(
                        [
                            s
                            for s in security_service.data_subjects.values()
                            if s.should_be_deleted()
                        ]
                    )

                    console.print(
                        Panel(
                            f"[blue]Cleanup Analysis[/blue]\n"
                            f"Data subjects that would be removed: {expired_count}\n"
                            f"Audit logs that would be archived: (estimated)\n"
                            f"Encryption keys that would be rotated: (estimated)",
                            title="Dry Run Results",
                        )
                    )

                else:
                    # Confirm cleanup
                    if not force:
                        confirm = click.confirm(
                            "This will permanently delete expired data. Continue?"
                        )
                        if not confirm:
                            console.print("[yellow]Cleanup cancelled[/yellow]")
                            return

                    # Perform actual cleanup
                    cleanup_stats = await security_service.cleanup_expired_data()

                    console.print(
                        Panel(
                            f"[green]Cleanup completed successfully[/green]\n"
                            f"Data subjects removed: {cleanup_stats['data_subjects_removed']}\n"
                            f"Audit logs archived: {cleanup_stats['audit_logs_archived']}\n"
                            f"Encryption keys rotated: {cleanup_stats['encryption_keys_rotated']}",
                            title="Cleanup Results",
                        )
                    )

                progress.update(task2, completed=True)

            except Exception as e:
                console.print(f"[red]Error during cleanup: {e}[/red]")
                return

    asyncio.run(run_cleanup())


# Helper functions for display


def _display_compliance_reports(reports: dict, detailed: bool):
    """Display compliance assessment reports."""
    console.print(
        Panel(
            f"[bold blue]Compliance Assessment Results[/bold blue]\n"
            f"Frameworks assessed: {len(reports)}\n"
            f"Assessment date: {list(reports.values())[0].assessment_date.strftime('%Y-%m-%d %H:%M')}",
            title="Compliance Overview",
        )
    )

    # Summary table
    table = Table(title="Compliance Summary")
    table.add_column("Framework", style="cyan")
    table.add_column("Score", style="yellow")
    table.add_column("Grade", style="green")
    table.add_column("Status", style="blue")
    table.add_column("Violations", style="red")

    for framework_name, report in reports.items():
        status_color = "green" if report.is_compliant() else "red"
        violation_count = len(report.violations)

        table.add_row(
            framework_name.upper(),
            f"{report.get_compliance_percentage():.1f}%",
            report.get_compliance_grade(),
            f"[{status_color}]{'Compliant' if report.is_compliant() else 'Non-Compliant'}[/{status_color}]",
            str(violation_count),
        )

    console.print(table)

    # Detailed violations if requested
    if detailed:
        for framework_name, report in reports.items():
            if report.violations:
                _display_compliance_violations(framework_name, report.violations)


def _display_compliance_violations(framework: str, violations: list):
    """Display compliance violations."""
    console.print(
        Panel(
            f"[bold red]{framework.upper()} Violations[/bold red]",
            title="Compliance Issues",
        )
    )

    table = Table()
    table.add_column("Control ID", style="cyan")
    table.add_column("Severity", style="yellow")
    table.add_column("Description", style="white")
    table.add_column("Status", style="blue")

    for violation in violations:
        severity_color = (
            "red"
            if violation.severity == "critical"
            else "yellow" if violation.severity == "high" else "white"
        )

        table.add_row(
            violation.control_id,
            f"[{severity_color}]{violation.severity.upper()}[/{severity_color}]",
            (
                violation.description[:60] + "..."
                if len(violation.description) > 60
                else violation.description
            ),
            violation.status,
        )

    console.print(table)


def _display_security_incidents(incidents: list, alert_threshold: int):
    """Display security incidents."""
    critical_incidents = [i for i in incidents if i.severity.value == "critical"]
    high_incidents = [i for i in incidents if i.severity.value == "high"]

    status_color = "red" if len(incidents) >= alert_threshold else "green"

    console.print(
        Panel(
            f"[bold {status_color}]Security Incident Analysis[/bold {status_color}]\n"
            f"Total incidents detected: {len(incidents)}\n"
            f"Critical incidents: {len(critical_incidents)}\n"
            f"High severity incidents: {len(high_incidents)}\n"
            f"Alert threshold: {alert_threshold}",
            title="Breach Detection Results",
        )
    )

    if incidents:
        table = Table(title="Security Incidents")
        table.add_column("Type", style="cyan")
        table.add_column("Severity", style="yellow")
        table.add_column("Description", style="white")
        table.add_column("Affected Data", style="blue")
        table.add_column("Detection", style="green")

        for incident in incidents[:10]:  # Show top 10
            severity_color = (
                "red" if incident.severity.value == "critical" else "yellow"
            )

            table.add_row(
                incident.incident_type,
                f"[{severity_color}]{incident.severity.value.upper()}[/{severity_color}]",
                (
                    incident.description[:50] + "..."
                    if len(incident.description) > 50
                    else incident.description
                ),
                ", ".join(incident.affected_data_types),
                incident.detection_method,
            )

        console.print(table)


def _display_data_subject_info(data: dict):
    """Display data subject information."""
    console.print(
        Panel(
            f"[bold blue]Data Subject Information[/bold blue]\n"
            f"Subject ID: {data.get('subject_id', 'N/A')}\n"
            f"Data types: {', '.join(data.get('data_types', []))}\n"
            f"Processing purposes: {', '.join(data.get('processing_purposes', []))}\n"
            f"Consent given: {data.get('consent_given', False)}\n"
            f"Last access: {data.get('last_access', 'N/A')}",
            title="GDPR Data Subject Details",
        )
    )


def _display_anonymization_comparison(original: dict, anonymized: dict, rules: dict):
    """Display before/after anonymization comparison."""
    console.print(
        Panel(
            "[bold blue]Anonymization Results[/bold blue]", title="Data Transformation"
        )
    )

    table = Table()
    table.add_column("Field", style="cyan")
    table.add_column("Original", style="yellow")
    table.add_column("Anonymized", style="green")
    table.add_column("Method", style="blue")

    for field, rule in rules.items():
        if field in original:
            orig_val = (
                str(original[field])[:20] + "..."
                if len(str(original[field])) > 20
                else str(original[field])
            )
            anon_val = (
                str(anonymized.get(field, "REMOVED"))[:20] + "..."
                if len(str(anonymized.get(field, "REMOVED"))) > 20
                else str(anonymized.get(field, "REMOVED"))
            )

            table.add_row(field, orig_val, anon_val, rule)

    console.print(table)


def _display_security_status(status: dict):
    """Display security status."""
    status_color = "green" if status["status"] == "operational" else "red"

    console.print(
        Panel(
            f"[bold {status_color}]Security Status: {status['status'].upper()}[/bold {status_color}]\n"
            f"Security level: {status['security_level']}\n"
            f"Encryption enabled: {status['encryption_enabled']}\n"
            f"Audit logging enabled: {status['audit_logging_enabled']}\n"
            f"Compliance frameworks: {', '.join(status['compliance_frameworks'])}\n"
            f"Data subjects: {status['data_subjects_count']}\n"
            f"Encryption keys: {status['encryption_keys_count']}\n"
            f"Issues detected: {status['issues_count']}",
            title="Security Overview",
        )
    )

    if status["issues"]:
        console.print(
            Panel(
                "\n".join([f"• {issue}" for issue in status["issues"]]),
                title="[red]Security Issues[/red]",
            )
        )


# Helper functions for saving results


async def _save_compliance_reports(reports: dict, output_path: str):
    """Save compliance reports to file."""
    report_data = {
        "assessment_date": list(reports.values())[0].assessment_date.isoformat(),
        "frameworks": {name: report.to_dict() for name, report in reports.items()},
    }

    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)


if __name__ == "__main__":
    security_commands()
