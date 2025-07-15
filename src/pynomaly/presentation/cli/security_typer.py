"""Typer-compatible wrapper for security CLI commands."""

import typer
from pathlib import Path
from typing import List, Optional

from pynomaly.presentation.cli.security import (
    assess_compliance,
    encrypt_data,
    register_data_subject,
    gdpr_request,
    detect_breach,
    anonymize_data,
    status,
    cleanup,
)

# Create Typer app
app = typer.Typer(
    name="security",
    help="🔐 Security and compliance management commands",
    add_completion=True,
    rich_markup_mode="rich",
)


@app.command()
def assess_compliance_cmd(
    frameworks: List[str] = typer.Option(
        ["soc2", "gdpr"],
        "--frameworks",
        help="Compliance frameworks to assess",
    ),
    output_path: Optional[str] = typer.Option(
        None, "--output-path", help="Path to save assessment reports"
    ),
    scope: Optional[str] = typer.Option(None, "--scope", help="Assessment scope description"),
    detailed: bool = typer.Option(False, "--detailed", help="Generate detailed assessment report"),
):
    """Assess compliance with regulatory frameworks."""
    # Call the original Click command with converted parameters
    assess_compliance.callback(frameworks, output_path, scope, detailed)


@app.command()
def encrypt_data_cmd(
    data_file: str = typer.Option(..., "--data-file", help="Path to data file for encryption"),
    output_file: Optional[str] = typer.Option(None, "--output-file", help="Path for encrypted output"),
    classification: str = typer.Option("confidential", "--classification", help="Data classification level"),
    frameworks: List[str] = typer.Option([], "--frameworks", help="Applicable compliance frameworks"),
):
    """Encrypt sensitive data with compliance controls."""
    encrypt_data.callback(data_file, output_file, classification, frameworks)


@app.command()
def register_data_subject_cmd(
    subject_data: str = typer.Option(..., "--subject-data", help="JSON file with data subject information"),
    processing_purposes: str = typer.Option(..., "--processing-purposes", help="Comma-separated processing purposes"),
    consent_given: bool = typer.Option(False, "--consent-given", help="Whether consent was given"),
):
    """Register data subject for GDPR compliance."""
    register_data_subject.callback(subject_data, processing_purposes, consent_given)


@app.command()
def gdpr_request_cmd(
    subject_id: str = typer.Option(..., "--subject-id", help="Data subject UUID"),
    request_type: str = typer.Option(..., "--request-type", help="Type of GDPR request"),
    output_file: Optional[str] = typer.Option(None, "--output-file", help="File to save request response"),
):
    """Handle GDPR data subject rights requests."""
    gdpr_request.callback(subject_id, request_type, output_file)


@app.command()
def detect_breach_cmd(
    access_logs: str = typer.Option(..., "--access-logs", help="JSON file with access logs"),
    output_file: Optional[str] = typer.Option(None, "--output-file", help="File to save breach detection results"),
    alert_threshold: int = typer.Option(3, "--alert-threshold", help="Minimum incidents for alert"),
):
    """Detect potential data breaches from access logs."""
    detect_breach.callback(access_logs, output_file, alert_threshold)


@app.command()
def anonymize_data_cmd(
    data_file: str = typer.Option(..., "--data-file", help="Data file to anonymize"),
    rules_file: str = typer.Option(..., "--rules-file", help="JSON file with anonymization rules"),
    output_file: Optional[str] = typer.Option(None, "--output-file", help="File to save anonymized data"),
):
    """Anonymize data for privacy compliance."""
    anonymize_data.callback(data_file, rules_file, output_file)


@app.command()
def status_cmd(
    config_file: Optional[str] = typer.Option(None, "--config-file", help="Security configuration file"),
    output_file: Optional[str] = typer.Option(None, "--output-file", help="File to save security status"),
):
    """Get overall security and compliance status."""
    status.callback(config_file, output_file)


@app.command()
def cleanup_cmd(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be cleaned without actually doing it"),
    force: bool = typer.Option(False, "--force", help="Force cleanup without confirmation"),
):
    """Clean up expired data per retention policies."""
    cleanup.callback(dry_run, force)


if __name__ == "__main__":
    app()