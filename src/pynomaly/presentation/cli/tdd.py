"""TDD CLI commands for managing test-driven development workflow."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pynomaly.presentation.cli.async_utils import cli_runner

from pynomaly.infrastructure.config.tdd_config import TDDSettings, get_tdd_config
from pynomaly.infrastructure.persistence.tdd_repository import FileTDDRepository
from pynomaly.infrastructure.tdd.enforcement import TDDEnforcementEngine
from pynomaly.infrastructure.tdd.git_hooks import GitHookManager, PreCommitConfig

# Create Typer app for TDD commands
app = typer.Typer(
    name="tdd",
    help="🧪 Test-Driven Development (TDD) management and enforcement",
    rich_markup_mode="rich",
)

# Create console for rich output
console = Console()


@app.command("init")
def init_tdd(
    enable: bool = typer.Option(False, "--enable", help="Enable TDD enforcement"),
    strict: bool = typer.Option(False, "--strict", help="Enable strict TDD mode"),
    coverage_threshold: float = typer.Option(
        0.8, "--coverage", help="Minimum coverage threshold"
    ),
    project_root: str | None = typer.Option(
        None, "--root", help="Project root directory"
    ),
):
    """Initialize TDD configuration for the project."""
    try:
        project_path = Path(project_root) if project_root else Path.cwd()

        console.print(f"[blue]Initializing TDD for project: {project_path}[/blue]")

        # Initialize TDD configuration
        config_manager = get_tdd_config()

        if enable:
            config_manager.enable_tdd(strict_mode=strict)
            config_manager.update_coverage_threshold(coverage_threshold)

            console.print(
                f"✅ TDD enabled with {coverage_threshold:.1%} coverage threshold"
            )
            if strict:
                console.print(
                    "⚠️  Strict mode enabled - tests must be written before implementation"
                )
        else:
            # Just initialize configuration file
            settings = TDDSettings(
                enabled=False, strict_mode=strict, min_test_coverage=coverage_threshold
            )
            config_manager.save_settings(settings)
            console.print("📝 TDD configuration initialized (disabled)")

        # Create TDD storage directory
        tdd_storage = project_path / "tdd_storage"
        tdd_storage.mkdir(exist_ok=True)

        # Initialize repository
        FileTDDRepository(tdd_storage)

        console.print(f"📁 TDD storage initialized at: {tdd_storage}")
        console.print("\n[green]TDD initialization complete![/green]")
        console.print("\nNext steps:")
        console.print("  • Run 'pynomaly tdd status' to check current compliance")
        console.print("  • Use 'pynomaly tdd require' to create test requirements")
        console.print("  • Use 'pynomaly tdd validate' to check violations")

    except Exception as e:
        console.print(f"[red]Error initializing TDD: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("status")
def show_status(
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed compliance report"
    ),
    project_root: str | None = typer.Option(
        None, "--root", help="Project root directory"
    ),
):
    """Show TDD compliance status."""
    try:
        project_path = Path(project_root) if project_root else Path.cwd()

        # Initialize TDD engine
        config_manager = get_tdd_config()
        repository = FileTDDRepository(project_path / "tdd_storage")
        engine = TDDEnforcementEngine(config_manager, repository)

        # Get compliance report
        report_data = engine.get_compliance_report()
        report = report_data["compliance_report"]
        settings = report_data["settings"]

        # Display status
        status_color = "green" if settings["enabled"] else "yellow"
        status_text = "ENABLED" if settings["enabled"] else "DISABLED"

        console.print(
            Panel(
                f"[{status_color}]TDD Status: {status_text}[/{status_color}]",
                title="🧪 Test-Driven Development",
                expand=False,
            )
        )

        if settings["enabled"]:
            # Show compliance summary
            table = Table(title="TDD Compliance Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Requirements", str(report.total_requirements))
            table.add_row("Pending", str(report.pending_requirements))
            table.add_row("Implemented", str(report.implemented_requirements))
            table.add_row("Validated", str(report.validated_requirements))
            table.add_row("Overall Compliance", f"{report.overall_compliance:.1%}")
            table.add_row("Violations", str(len(report.violations)))
            table.add_row("Recent Violations", str(report_data["recent_violations"]))

            console.print(table)

            if detailed:
                # Show module-specific compliance
                if report.module_compliance:
                    console.print("\n[bold]Module Compliance:[/bold]")
                    module_table = Table()
                    module_table.add_column("Module", style="cyan")
                    module_table.add_column("Compliance", style="green")

                    for module, compliance in report.module_compliance.items():
                        module_table.add_row(module, f"{compliance:.1%}")

                    console.print(module_table)

                # Show coverage report
                if report.coverage_report:
                    console.print("\n[bold]Coverage Report:[/bold]")
                    coverage_table = Table()
                    coverage_table.add_column("File", style="cyan")
                    coverage_table.add_column("Coverage", style="green")

                    for file_path, coverage in report.coverage_report.items():
                        coverage_color = (
                            "green" if coverage >= settings["min_coverage"] else "red"
                        )
                        coverage_table.add_row(
                            file_path,
                            f"[{coverage_color}]{coverage:.1%}[/{coverage_color}]",
                        )

                    console.print(coverage_table)

        else:
            console.print("\n[yellow]TDD is currently disabled.[/yellow]")
            console.print("Run 'pynomaly tdd init --enable' to enable TDD enforcement.")

    except Exception as e:
        console.print(f"[red]Error getting TDD status: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("require")
def create_requirement(
    module_path: str = typer.Argument(..., help="Path to the module requiring tests"),
    function_name: str = typer.Argument(..., help="Name of the function to test"),
    description: str = typer.Option(
        ..., "--desc", help="Description of what should be tested"
    ),
    specification: str = typer.Option(
        ..., "--spec", help="Detailed test specification"
    ),
    coverage_target: float = typer.Option(
        0.8, "--coverage", help="Target coverage percentage"
    ),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags"),
    project_root: str | None = typer.Option(
        None, "--root", help="Project root directory"
    ),
):
    """Create a test requirement for a function."""
    try:
        project_path = Path(project_root) if project_root else Path.cwd()

        # Initialize TDD engine
        config_manager = get_tdd_config()
        repository = FileTDDRepository(project_path / "tdd_storage")
        engine = TDDEnforcementEngine(config_manager, repository)

        # Parse tags
        tag_set = set()
        if tags:
            tag_set = {tag.strip() for tag in tags.split(",")}

        # Create requirement
        requirement = engine.create_test_requirement(
            module_path=module_path,
            function_name=function_name,
            description=description,
            test_specification=specification,
            coverage_target=coverage_target,
            tags=tag_set,
        )

        console.print(f"✅ Created test requirement: {requirement.id}")
        console.print(f"📝 Module: {module_path}")
        console.print(f"🎯 Function: {function_name}")
        console.print(f"📊 Coverage Target: {coverage_target:.1%}")

        if tag_set:
            console.print(f"🏷️  Tags: {', '.join(tag_set)}")

        console.print("\n[green]Test requirement created successfully![/green]")
        console.print("Next steps:")
        console.print("  • Write the test according to the specification")
        console.print("  • Implement the function to make the test pass")
        console.print("  • Run 'pynomaly tdd validate' to check compliance")

    except Exception as e:
        console.print(f"[red]Error creating test requirement: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("validate")
def validate_compliance(
    fix: bool = typer.Option(False, "--fix", help="Attempt to auto-fix violations"),
    coverage: bool = typer.Option(False, "--coverage", help="Run coverage analysis"),
    project_root: str | None = typer.Option(
        None, "--root", help="Project root directory"
    ),
    file_path: str | None = typer.Option(
        None, "--file", help="Validate specific file only"
    ),
):
    """Validate TDD compliance for the project."""
    try:
        project_path = Path(project_root) if project_root else Path.cwd()

        # Initialize TDD engine
        config_manager = get_tdd_config()
        repository = FileTDDRepository(project_path / "tdd_storage")
        engine = TDDEnforcementEngine(config_manager, repository)

        if not engine.settings.enabled:
            console.print(
                "[yellow]TDD is not enabled. Run 'pynomaly tdd init --enable' first.[/yellow]"
            )
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Validate files
            if file_path:
                task = progress.add_task("Validating file...", total=1)
                violations = engine.validate_file(Path(file_path))
                progress.update(task, advance=1)
            else:
                task = progress.add_task("Validating project...", total=1)
                violations = engine.validate_project(project_path)
                progress.update(task, advance=1)

            # Run coverage analysis if requested
            coverage_data = {}
            if coverage:
                coverage_task = progress.add_task(
                    "Running coverage analysis...", total=1
                )
                coverage_data = engine.run_coverage_analysis(project_path)
                coverage_violations = engine.validate_coverage_thresholds(coverage_data)
                violations.extend(coverage_violations)
                progress.update(coverage_task, advance=1)

        # Display results
        if not violations:
            console.print("✅ [green]No TDD violations found![/green]")
            return

        console.print(f"\n⚠️  Found {len(violations)} TDD violations:")

        # Group violations by type
        violation_groups = {}
        for violation in violations:
            if violation.violation_type not in violation_groups:
                violation_groups[violation.violation_type] = []
            violation_groups[violation.violation_type].append(violation)

        for violation_type, group_violations in violation_groups.items():
            console.print(
                f"\n[bold]{violation_type.replace('_', ' ').title()}[/bold] ({len(group_violations)} violations)"
            )

            for violation in group_violations[:5]:  # Show first 5 of each type
                severity_color = {"error": "red", "warning": "yellow", "info": "blue"}[
                    violation.severity
                ]
                console.print(
                    f"  [{severity_color}]•[/{severity_color}] {violation.description}"
                )
                if violation.suggestion:
                    console.print(f"    💡 {violation.suggestion}")

            if len(group_violations) > 5:
                console.print(f"    ... and {len(group_violations) - 5} more")

        # Auto-fix if requested
        if fix:
            console.print("\n🔧 Attempting to auto-fix violations...")
            fixes = engine.auto_fix_violations(violations)

            if fixes:
                console.print(f"✅ Applied {len(fixes)} automatic fixes:")
                for fix in fixes:
                    console.print(f"  • {fix}")
            else:
                console.print("ℹ️  No auto-fixable violations found")

        # Show coverage summary
        if coverage_data:
            avg_coverage = (
                sum(coverage_data.values()) / len(coverage_data) if coverage_data else 0
            )
            coverage_color = (
                "green" if avg_coverage >= engine.settings.min_test_coverage else "red"
            )
            console.print(
                f"\n📊 Average Coverage: [{coverage_color}]{avg_coverage:.1%}[/{coverage_color}]"
            )

    except Exception as e:
        console.print(f"[red]Error validating TDD compliance: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("exempt")
def add_exemption(
    pattern: str = typer.Argument(..., help="File pattern to exempt from TDD"),
    remove: bool = typer.Option(
        False, "--remove", help="Remove exemption instead of adding"
    ),
    list_exemptions: bool = typer.Option(
        False, "--list", help="List current exemptions"
    ),
):
    """Manage TDD exemptions for file patterns."""
    try:
        config_manager = get_tdd_config()

        if list_exemptions:
            exemptions = config_manager.settings.exemption_patterns
            if not exemptions:
                console.print("No TDD exemptions configured.")
                return

            console.print("Current TDD exemptions:")
            for exemption in exemptions:
                console.print(f"  • {exemption}")
            return

        if remove:
            config_manager.remove_exemption(pattern)
            console.print(f"✅ Removed TDD exemption: {pattern}")
        else:
            config_manager.add_exemption(pattern)
            console.print(f"✅ Added TDD exemption: {pattern}")

        console.print("\nCurrent exemptions:")
        for exemption in config_manager.settings.exemption_patterns:
            console.print(f"  • {exemption}")

    except Exception as e:
        console.print(f"[red]Error managing TDD exemption: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("enable")
def enable_tdd(
    strict: bool = typer.Option(False, "--strict", help="Enable strict TDD mode"),
    coverage_threshold: float = typer.Option(
        0.8, "--coverage", help="Set coverage threshold"
    ),
):
    """Enable TDD enforcement."""
    try:
        config_manager = get_tdd_config()
        config_manager.enable_tdd(strict_mode=strict)
        config_manager.update_coverage_threshold(coverage_threshold)

        console.print("✅ [green]TDD enforcement enabled![/green]")
        console.print(f"📊 Coverage threshold: {coverage_threshold:.1%}")
        if strict:
            console.print("⚠️  Strict mode: Tests must be written before implementation")

        console.print("\nRun 'pynomaly tdd status' to check current compliance.")

    except Exception as e:
        console.print(f"[red]Error enabling TDD: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("disable")
def disable_tdd():
    """Disable TDD enforcement."""
    try:
        config_manager = get_tdd_config()
        config_manager.disable_tdd()

        console.print("⏸️  [yellow]TDD enforcement disabled.[/yellow]")
        console.print("Configuration and test requirements are preserved.")
        console.print("Run 'pynomaly tdd enable' to re-enable enforcement.")

    except Exception as e:
        console.print(f"[red]Error disabling TDD: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("requirements")
def list_requirements(
    status: str | None = typer.Option(
        None, "--status", help="Filter by status (pending, implemented, validated)"
    ),
    module: str | None = typer.Option(None, "--module", help="Filter by module path"),
    tags: str | None = typer.Option(
        None, "--tags", help="Filter by tags (comma-separated)"
    ),
    project_root: str | None = typer.Option(
        None, "--root", help="Project root directory"
    ),
):
    """List test requirements."""
    try:
        project_path = Path(project_root) if project_root else Path.cwd()

        # Initialize repository
        repository = FileTDDRepository(project_path / "tdd_storage")

        # Get requirements based on filters
        if status:
            requirements = repository.find_requirements_by_status(status)
        elif module:
            requirements = repository.find_requirements_by_module(module)
        elif tags:
            tag_set = {tag.strip() for tag in tags.split(",")}
            requirements = repository.find_requirements_by_tags(tag_set)
        else:
            requirements = repository.find_all()

        if not requirements:
            console.print("No test requirements found.")
            return

        # Display requirements
        table = Table(title=f"Test Requirements ({len(requirements)})")
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Module", style="blue")
        table.add_column("Function", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Coverage", style="magenta")
        table.add_column("Description", style="white")

        for req in requirements:
            status_color = {
                "pending": "red",
                "implemented": "yellow",
                "validated": "green",
            }.get(req.status, "white")

            table.add_row(
                req.id[:8],
                req.module_path,
                req.function_name,
                f"[{status_color}]{req.status}[/{status_color}]",
                f"{req.coverage_target:.1%}",
                (
                    req.description[:50] + "..."
                    if len(req.description) > 50
                    else req.description
                ),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing requirements: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("coverage")
def run_coverage(
    project_root: str | None = typer.Option(
        None, "--root", help="Project root directory"
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", help="Coverage threshold to check against"
    ),
):
    """Run test coverage analysis."""
    try:
        project_path = Path(project_root) if project_root else Path.cwd()

        # Initialize TDD engine
        config_manager = get_tdd_config()
        repository = FileTDDRepository(project_path / "tdd_storage")
        engine = TDDEnforcementEngine(config_manager, repository)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running coverage analysis...", total=1)
            coverage_data = engine.run_coverage_analysis(project_path)
            progress.update(task, advance=1)

        if not coverage_data:
            console.print(
                "[yellow]No coverage data available. Make sure pytest and pytest-cov are installed.[/yellow]"
            )
            return

        # Use provided threshold or default from settings
        coverage_threshold = (
            threshold if threshold is not None else engine.settings.min_test_coverage
        )

        # Display coverage report
        table = Table(title="Test Coverage Report")
        table.add_column("File", style="cyan")
        table.add_column("Coverage", style="green")
        table.add_column("Status", style="yellow")

        total_coverage = 0
        files_count = 0

        for file_path, coverage in coverage_data.items():
            coverage_color = "green" if coverage >= coverage_threshold else "red"
            status = "✅" if coverage >= coverage_threshold else "❌"

            table.add_row(
                file_path,
                f"[{coverage_color}]{coverage:.1%}[/{coverage_color}]",
                status,
            )

            total_coverage += coverage
            files_count += 1

        console.print(table)

        # Show summary
        avg_coverage = total_coverage / files_count if files_count > 0 else 0
        avg_color = "green" if avg_coverage >= coverage_threshold else "red"

        console.print(
            f"\n📊 Average Coverage: [{avg_color}]{avg_coverage:.1%}[/{avg_color}]"
        )
        console.print(f"🎯 Threshold: {coverage_threshold:.1%}")

        files_above_threshold = sum(
            1 for cov in coverage_data.values() if cov >= coverage_threshold
        )
        console.print(
            f"✅ Files above threshold: {files_above_threshold}/{files_count}"
        )

    except Exception as e:
        console.print(f"[red]Error running coverage analysis: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("config")
def show_config(
    edit: bool = typer.Option(False, "--edit", help="Edit TDD configuration"),
    reset: bool = typer.Option(False, "--reset", help="Reset to default configuration"),
):
    """Show or edit TDD configuration."""
    try:
        config_manager = get_tdd_config()

        if reset:
            config_manager.reset_to_defaults()
            console.print("✅ TDD configuration reset to defaults.")
            return

        settings = config_manager.settings

        # Display current configuration
        table = Table(title="TDD Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        config_items = [
            ("Enabled", str(settings.enabled)),
            ("Strict Mode", str(settings.strict_mode)),
            ("Auto Validation", str(settings.auto_validation)),
            ("Min Coverage", f"{settings.min_test_coverage:.1%}"),
            ("Fail Under Coverage", f"{settings.coverage_fail_under:.1%}"),
            ("Branch Coverage Required", str(settings.branch_coverage_required)),
            ("Git Hooks Enabled", str(settings.git_hooks_enabled)),
            ("Pre-commit Validation", str(settings.pre_commit_validation)),
            ("CI Validation", str(settings.ci_validation_enabled)),
            ("Fail on Violations", str(settings.fail_on_violations)),
            ("Test Naming Convention", settings.test_naming_convention),
            ("Require Test Docstrings", str(settings.require_test_docstrings)),
        ]

        for key, value in config_items:
            table.add_row(key, value)

        console.print(table)

        # Show patterns
        console.print("\n[bold]File Patterns:[/bold]")
        console.print(f"Test Files: {', '.join(settings.test_file_patterns)}")
        console.print(
            f"Implementation Files: {', '.join(settings.implementation_patterns)}"
        )
        console.print(f"Exemptions: {', '.join(settings.exemption_patterns)}")

        console.print("\n[bold]Enforced Packages:[/bold]")
        for package in settings.enforce_on_packages:
            console.print(f"  • {package}")

        if edit:
            console.print(
                "\n[yellow]Configuration editing via CLI coming soon![/yellow]"
            )
            console.print("For now, edit the tdd_config.json file directly.")

    except Exception as e:
        console.print(f"[red]Error showing TDD configuration: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("hooks")
def manage_git_hooks(
    install: bool = typer.Option(False, "--install", help="Install git hooks"),
    uninstall: bool = typer.Option(False, "--uninstall", help="Uninstall git hooks"),
    status: bool = typer.Option(False, "--status", help="Show git hooks status"),
    pre_commit_config: bool = typer.Option(
        False, "--pre-commit", help="Add to pre-commit configuration"
    ),
):
    """Manage git hooks for TDD enforcement."""
    try:
        hook_manager = GitHookManager()

        if install:
            hook_manager.install_hooks()
            console.print("✅ [green]TDD git hooks installed![/green]")

            # Also install pre-commit config if requested
            if pre_commit_config:
                pc_config = PreCommitConfig()
                pc_config.add_tdd_hook()
                console.print("✅ Added TDD hook to .pre-commit-config.yaml")

        elif uninstall:
            hook_manager.uninstall_hooks()
            console.print("🗑️  [yellow]TDD git hooks uninstalled.[/yellow]")

            if pre_commit_config:
                pc_config = PreCommitConfig()
                pc_config.remove_tdd_hook()
                console.print("🗑️  Removed TDD hook from .pre-commit-config.yaml")

        elif status:
            # Check hook status
            hooks_status = []

            for hook_name in ["pre-commit", "pre-push"]:
                is_installed = hook_manager.is_hook_installed(hook_name)
                status_icon = "✅" if is_installed else "❌"
                hooks_status.append(
                    (
                        hook_name,
                        status_icon,
                        "Installed" if is_installed else "Not installed",
                    )
                )

            table = Table(title="Git Hooks Status")
            table.add_column("Hook", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Description")

            for hook_name, icon, description in hooks_status:
                table.add_row(
                    f"{icon} {hook_name}", description, "Validates TDD compliance"
                )

            console.print(table)

            # Check pre-commit config
            pc_config = PreCommitConfig()
            if pc_config.pre_commit_config.exists():
                console.print(f"\n📄 Pre-commit config: {pc_config.pre_commit_config}")
            else:
                console.print("\n📄 No .pre-commit-config.yaml found")

        else:
            console.print("Use --install, --uninstall, or --status")
            console.print("Example: pynomaly tdd hooks --install --pre-commit")

    except Exception as e:
        console.print(f"[red]Error managing git hooks: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("report")
def generate_report(
    output_file: str | None = typer.Option(
        None, "--output", "-o", help="Output file for report"
    ),
    format: str = typer.Option(
        "json", "--format", help="Report format (json, yaml, html)"
    ),
    project_root: str | None = typer.Option(
        None, "--root", help="Project root directory"
    ),
):
    """Generate comprehensive TDD compliance report."""
    try:
        project_path = Path(project_root) if project_root else Path.cwd()

        # Initialize TDD engine
        config_manager = get_tdd_config()
        repository = FileTDDRepository(project_path / "tdd_storage")
        engine = TDDEnforcementEngine(config_manager, repository)

        # Generate comprehensive report
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating TDD report...", total=3)

            # Get compliance report
            compliance_data = engine.get_compliance_report()
            progress.update(task, advance=1)

            # Run validation
            violations = engine.validate_project(project_path)
            progress.update(task, advance=1)

            # Run coverage analysis
            coverage_data = engine.run_coverage_analysis(project_path)
            progress.update(task, advance=1)

        # Create comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(project_path),
            "tdd_settings": compliance_data["settings"],
            "compliance": compliance_data["compliance_report"].__dict__,
            "violations": [
                {
                    "type": v.violation_type,
                    "file": v.file_path,
                    "line": v.line_number,
                    "description": v.description,
                    "severity": v.severity,
                    "rule": v.rule_name,
                    "suggestion": v.suggestion,
                    "auto_fixable": v.auto_fixable,
                }
                for v in violations
            ],
            "coverage": coverage_data,
            "summary": {
                "total_violations": len(violations),
                "critical_violations": len(
                    [v for v in violations if v.severity == "error"]
                ),
                "warning_violations": len(
                    [v for v in violations if v.severity == "warning"]
                ),
                "auto_fixable_violations": len(
                    [v for v in violations if v.auto_fixable]
                ),
                "average_coverage": (
                    sum(coverage_data.values()) / len(coverage_data)
                    if coverage_data
                    else 0
                ),
            },
        }

        # Output report
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "json":
                with open(output_path, "w") as f:
                    json.dump(report, f, indent=2, default=str)
            elif format.lower() == "yaml":
                import yaml

                with open(output_path, "w") as f:
                    yaml.dump(report, f, default_flow_style=False)
            elif format.lower() == "html":
                html_content = _generate_html_report(report)
                with open(output_path, "w") as f:
                    f.write(html_content)

            console.print(f"✅ Report saved to: {output_path}")
        else:
            # Display summary to console
            console.print("\n[bold]TDD Compliance Report Summary[/bold]")
            console.print(
                f"📊 Overall Compliance: {report['compliance']['overall_compliance']:.1%}"
            )
            console.print(
                f"⚠️  Total Violations: {report['summary']['total_violations']}"
            )
            console.print(f"❌ Critical: {report['summary']['critical_violations']}")
            console.print(f"🟡 Warnings: {report['summary']['warning_violations']}")
            console.print(
                f"🔧 Auto-fixable: {report['summary']['auto_fixable_violations']}"
            )
            console.print(
                f"📈 Average Coverage: {report['summary']['average_coverage']:.1%}"
            )

    except Exception as e:
        console.print(f"[red]Error generating TDD report: {str(e)}[/red]")
        raise typer.Exit(1)


def _generate_html_report(report: dict) -> str:
    """Generate HTML report from TDD data."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TDD Compliance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .violation {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ff6b6b; background: #fff5f5; }}
        .warning {{ border-left-color: #ffa500; background: #fffacd; }}
        .coverage-item {{ margin: 5px 0; }}
        .good {{ color: #28a745; }}
        .bad {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>TDD Compliance Report</h1>
        <p>Generated: {report["timestamp"]}</p>
        <p>Project: {report["project_root"]}</p>
    </div>

    <h2>Summary</h2>
    <div class="metric">
        <strong>Overall Compliance</strong><br>
        {report["compliance"]["overall_compliance"]:.1%}
    </div>
    <div class="metric">
        <strong>Total Violations</strong><br>
        {report["summary"]["total_violations"]}
    </div>
    <div class="metric">
        <strong>Average Coverage</strong><br>
        {report["summary"]["average_coverage"]:.1%}
    </div>

    <h2>Violations</h2>
"""

    for violation in report["violations"][:20]:  # Show first 20 violations
        severity_class = "violation" if violation["severity"] == "error" else "warning"
        html += f"""
    <div class="{severity_class}">
        <strong>{violation["type"].replace("_", " ").title()}</strong><br>
        File: {violation["file"]}<br>
        {violation["description"]}<br>
        {f"<em>Suggestion: {violation['suggestion']}</em>" if violation["suggestion"] else ""}
    </div>
"""

    html += """
    <h2>Coverage Report</h2>
"""

    for file_path, coverage in list(report["coverage"].items())[
        :20
    ]:  # Show first 20 files
        coverage_class = "good" if coverage >= 0.8 else "bad"
        html += f"""
    <div class="coverage-item">
        <span class="{coverage_class}">{coverage:.1%}</span> - {file_path}
    </div>
"""

    html += """
</body>
</html>
"""

    return html


if __name__ == "__main__":
    app()
