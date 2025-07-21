#!/usr/bin/env python3
"""
anomaly_detection SDK Generator CLI

A comprehensive command-line interface for generating client SDKs for the anomaly detection API.
This tool provides easy-to-use commands for generating, managing, and publishing SDKs
across multiple programming languages.

Usage:
    python sdk_cli.py generate --languages python typescript java
    python sdk_cli.py list-languages
    python sdk_cli.py validate --language python
    python sdk_cli.py publish --language python --registry pypi
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from sdk_generator import SDKGenerator


class SDKCLIManager:
    """Advanced CLI manager for SDK operations."""

    def __init__(self):
        self.console = Console()
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = self.project_root / "config" / "sdk_generator_config.yaml"
        self.generator = SDKGenerator()

        # Load configuration
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load SDK generator configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}

    def _save_config(self):
        """Save configuration back to file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def list_languages(self) -> None:
        """List all supported languages with their status."""
        table = Table(title="Supported SDK Languages")
        table.add_column("Language", style="cyan")
        table.add_column("Generator", style="blue")
        table.add_column("Package Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Features", style="magenta")

        for lang, config in self.generator.languages.items():
            # Check if language is enabled in config
            enabled = (
                self.config.get("languages", {}).get(lang, {}).get("enabled", True)
            )
            status = "✅ Enabled" if enabled else "❌ Disabled"

            # Get features
            features = []
            lang_config = self.config.get("languages", {}).get(lang, {})
            if lang_config.get("features", {}).get("async_support"):
                features.append("Async")
            if lang_config.get("features", {}).get("retry_logic"):
                features.append("Retry")
            if lang_config.get("features", {}).get("rate_limiting"):
                features.append("Rate Limiting")

            table.add_row(
                lang.title(),
                config["generator"],
                config["package_name"],
                status,
                ", ".join(features) or "Basic",
            )

        self.console.print(table)

    def validate_environment(self) -> bool:
        """Validate that all required tools are available."""
        tools = [
            ("openapi-generator-cli", "OpenAPI Generator"),
            ("python", "Python"),
            ("node", "Node.js"),
            ("npm", "npm"),
            ("git", "Git"),
        ]

        missing_tools = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Checking environment...", total=len(tools))

            for tool, name in tools:
                try:
                    subprocess.run(
                        [tool, "--version"], capture_output=True, check=True, timeout=10
                    )
                    progress.update(task, advance=1, description=f"✅ {name} found")
                except (
                    subprocess.CalledProcessError,
                    FileNotFoundError,
                    subprocess.TimeoutExpired,
                ):
                    missing_tools.append(name)
                    progress.update(task, advance=1, description=f"❌ {name} not found")

        if missing_tools:
            self.console.print(
                f"\n[red]Missing required tools: {', '.join(missing_tools)}[/red]"
            )
            self.console.print("\n[yellow]Installation instructions:[/yellow]")
            for tool in missing_tools:
                if tool == "OpenAPI Generator":
                    self.console.print(
                        "- OpenAPI Generator: npm install -g @openapitools/openapi-generator-cli"
                    )
                elif tool == "Node.js":
                    self.console.print("- Node.js: https://nodejs.org/")
                elif tool == "npm":
                    self.console.print("- npm: Included with Node.js")
                elif tool == "Git":
                    self.console.print("- Git: https://git-scm.com/")
            return False

        self.console.print("\n[green]✅ Environment validation passed![/green]")
        return True

    def generate_sdks(
        self, languages: list[str] | None = None, validate_only: bool = False
    ) -> dict[str, bool]:
        """Generate SDKs with progress tracking."""
        if not self.validate_environment():
            return {}

        if languages is None:
            # Get enabled languages from config
            enabled_languages = []
            for lang, config in self.config.get("languages", {}).items():
                if config.get("enabled", True):
                    enabled_languages.append(lang)
            languages = enabled_languages or list(self.generator.languages.keys())

        self.console.print(
            f"\n[blue]🚀 Starting SDK generation for: {', '.join(languages)}[/blue]"
        )

        results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            # Validation phase
            task = progress.add_task("Validating OpenAPI specification...", total=1)
            try:
                spec = self.generator.load_openapi_spec()
                progress.update(
                    task, advance=1, description="✅ OpenAPI spec validated"
                )
            except Exception as e:
                progress.update(
                    task,
                    advance=1,
                    description=f"❌ OpenAPI spec validation failed: {e}",
                )
                return {}

            if validate_only:
                self.console.print(
                    "[green]✅ Validation completed successfully![/green]"
                )
                return {"validation": True}

            # Generation phase
            for lang in languages:
                if lang not in self.generator.languages:
                    self.console.print(
                        f"[yellow]⚠️  Skipping unsupported language: {lang}[/yellow]"
                    )
                    continue

                task = progress.add_task(f"Generating {lang.title()} SDK...", total=1)

                try:
                    success = self.generator.generate_sdk(lang, spec)
                    results[lang] = success

                    if success:
                        progress.update(
                            task,
                            advance=1,
                            description=f"✅ {lang.title()} SDK generated",
                        )
                        # Run post-generation steps
                        self._run_post_generation_steps(lang)
                    else:
                        progress.update(
                            task,
                            advance=1,
                            description=f"❌ {lang.title()} SDK generation failed",
                        )

                except Exception as e:
                    results[lang] = False
                    progress.update(
                        task,
                        advance=1,
                        description=f"❌ {lang.title()} SDK failed: {str(e)[:50]}",
                    )

        # Generate summary
        self._display_generation_summary(results)
        return results

    def _run_post_generation_steps(self, language: str) -> None:
        """Run post-generation steps for a language."""
        scripts = (
            self.config.get("post_generation", {}).get("scripts", {}).get(language, [])
        )

        if not scripts:
            return

        sdk_dir = self.generator.output_dir / language

        for script in scripts:
            try:
                result = subprocess.run(
                    script,
                    shell=True,
                    cwd=sdk_dir,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if result.returncode == 0:
                    self.console.print(
                        f"[green]✅ Post-generation script completed: {script}[/green]"
                    )
                else:
                    self.console.print(
                        f"[yellow]⚠️  Post-generation script warning: {script}[/yellow]"
                    )
                    self.console.print(f"[dim]{result.stderr}[/dim]")

            except subprocess.TimeoutExpired:
                self.console.print(
                    f"[red]❌ Post-generation script timed out: {script}[/red]"
                )
            except Exception as e:
                self.console.print(
                    f"[red]❌ Post-generation script failed: {script} - {e}[/red]"
                )

    def _display_generation_summary(self, results: dict[str, bool]) -> None:
        """Display a comprehensive generation summary."""
        successful = [lang for lang, success in results.items() if success]
        failed = [lang for lang, success in results.items() if not success]

        # Create summary table
        table = Table(title="SDK Generation Summary")
        table.add_column("Language", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Output Directory", style="blue")
        table.add_column("Package Name", style="yellow")

        for lang, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            output_dir = f"sdks/{lang}/"
            package_name = self.generator.languages.get(lang, {}).get(
                "package_name", "N/A"
            )

            table.add_row(lang.title(), status, output_dir, package_name)

        self.console.print("\n")
        self.console.print(table)

        # Summary statistics
        total = len(results)
        success_count = len(successful)
        failure_count = len(failed)

        if success_count == total:
            self.console.print(
                f"\n[green]🎉 All {total} SDKs generated successfully![/green]"
            )
        elif success_count > 0:
            self.console.print(
                f"\n[yellow]⚠️  {success_count}/{total} SDKs generated successfully[/yellow]"
            )
            if failed:
                self.console.print(f"[red]Failed: {', '.join(failed)}[/red]")
        else:
            self.console.print("\n[red]💥 All SDK generations failed![/red]")

        # Next steps
        if successful:
            self.console.print("\n[blue]📋 Next Steps:[/blue]")
            self.console.print("1. Review generated SDKs in the 'sdks/' directory")
            self.console.print("2. Run tests: anomaly_detection sdk test --language <language>")
            self.console.print(
                "3. Validate: anomaly_detection sdk validate --language <language>"
            )
            self.console.print("4. Publish: anomaly_detection sdk publish --language <language>")

    def test_sdk(self, language: str) -> bool:
        """Run tests for a specific SDK."""
        sdk_dir = self.generator.output_dir / language

        if not sdk_dir.exists():
            self.console.print(
                f"[red]❌ SDK for {language} not found. Generate it first.[/red]"
            )
            return False

        self.console.print(
            f"[blue]🧪 Running tests for {language.title()} SDK...[/blue]"
        )

        # Language-specific test commands
        test_commands = {
            "python": ["python", "-m", "pytest", "-v"],
            "typescript": ["npm", "test"],
            "java": ["mvn", "test"],
            "go": ["go", "test", "./..."],
            "csharp": ["dotnet", "test"],
            "php": ["composer", "test"],
            "ruby": ["bundle", "exec", "rspec"],
            "rust": ["cargo", "test"],
        }

        if language not in test_commands:
            self.console.print(
                f"[yellow]⚠️  No test command configured for {language}[/yellow]"
            )
            return False

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"Running {language} tests...", total=1)

                result = subprocess.run(
                    test_commands[language],
                    cwd=sdk_dir,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    progress.update(
                        task,
                        advance=1,
                        description=f"✅ {language.title()} tests passed",
                    )
                    self.console.print(
                        f"[green]✅ All tests passed for {language.title()} SDK![/green]"
                    )
                    return True
                else:
                    progress.update(
                        task,
                        advance=1,
                        description=f"❌ {language.title()} tests failed",
                    )
                    self.console.print(
                        f"[red]❌ Tests failed for {language.title()} SDK[/red]"
                    )
                    self.console.print(f"[dim]{result.stdout}[/dim]")
                    self.console.print(f"[dim]{result.stderr}[/dim]")
                    return False

        except subprocess.TimeoutExpired:
            self.console.print(
                f"[red]❌ Tests timed out for {language.title()} SDK[/red]"
            )
            return False
        except Exception as e:
            self.console.print(
                f"[red]❌ Error running tests for {language.title()} SDK: {e}[/red]"
            )
            return False

    def validate_sdk(self, language: str) -> bool:
        """Validate a generated SDK against quality gates."""
        sdk_dir = self.generator.output_dir / language

        if not sdk_dir.exists():
            self.console.print(
                f"[red]❌ SDK for {language} not found. Generate it first.[/red]"
            )
            return False

        self.console.print(f"[blue]🔍 Validating {language.title()} SDK...[/blue]")

        # Quality gates from config
        requirements = self.config.get("quality_gates", {}).get("requirements", [])

        validation_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            for requirement in requirements:
                task = progress.add_task(f"Checking {requirement['name']}...", total=1)

                passed = self._check_requirement(language, requirement)
                validation_results.append(
                    {
                        "name": requirement["name"],
                        "description": requirement["description"],
                        "mandatory": requirement.get("mandatory", False),
                        "passed": passed,
                    }
                )

                status = "✅" if passed else "❌"
                progress.update(
                    task, advance=1, description=f"{status} {requirement['name']}"
                )

        # Display validation results
        table = Table(title=f"{language.title()} SDK Validation Results")
        table.add_column("Requirement", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Mandatory", style="yellow")
        table.add_column("Description", style="blue")

        all_mandatory_passed = True

        for result in validation_results:
            status = "✅ Pass" if result["passed"] else "❌ Fail"
            mandatory = "Yes" if result["mandatory"] else "No"

            if result["mandatory"] and not result["passed"]:
                all_mandatory_passed = False

            table.add_row(result["name"], status, mandatory, result["description"])

        self.console.print("\n")
        self.console.print(table)

        if all_mandatory_passed:
            self.console.print(
                f"\n[green]✅ {language.title()} SDK validation passed![/green]"
            )
            return True
        else:
            self.console.print(
                f"\n[red]❌ {language.title()} SDK validation failed![/red]"
            )
            return False

    def _check_requirement(self, language: str, requirement: dict[str, Any]) -> bool:
        """Check if a specific requirement is met."""
        sdk_dir = self.generator.output_dir / language
        requirement_name = requirement["name"]

        # Basic file existence checks
        if requirement_name == "authentication_support":
            # Check for auth-related files or classes
            return self._check_files_exist(sdk_dir, ["*auth*", "*token*", "*login*"])

        elif requirement_name == "error_handling":
            # Check for exception/error handling files
            return self._check_files_exist(
                sdk_dir, ["*error*", "*exception*", "*fault*"]
            )

        elif requirement_name == "comprehensive_tests":
            # Check for test files
            return self._check_files_exist(sdk_dir, ["*test*", "*spec*"])

        elif requirement_name == "documentation":
            # Check for documentation files
            return (sdk_dir / "README.md").exists()

        elif requirement_name == "compile_check":
            # Try to compile/build the SDK
            return self._try_compile(language, sdk_dir)

        # Default to True for unknown requirements
        return True

    def _check_files_exist(self, sdk_dir: Path, patterns: list[str]) -> bool:
        """Check if files matching patterns exist."""
        for pattern in patterns:
            matches = list(sdk_dir.rglob(pattern))
            if matches:
                return True
        return False

    def _try_compile(self, language: str, sdk_dir: Path) -> bool:
        """Try to compile/build the SDK."""
        compile_commands = {
            "python": ["python", "-m", "py_compile", "**/*.py"],
            "typescript": ["npm", "run", "build"],
            "java": ["mvn", "compile"],
            "go": ["go", "build", "./..."],
            "csharp": ["dotnet", "build"],
            "rust": ["cargo", "build"],
        }

        if language not in compile_commands:
            return True  # Assume success for unsupported languages

        try:
            result = subprocess.run(
                compile_commands[language],
                cwd=sdk_dir,
                capture_output=True,
                timeout=180,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    def publish_sdk(
        self, language: str, registry: str | None = None, dry_run: bool = False
    ) -> bool:
        """Publish SDK to package registry."""
        sdk_dir = self.generator.output_dir / language

        if not sdk_dir.exists():
            self.console.print(
                f"[red]❌ SDK for {language} not found. Generate it first.[/red]"
            )
            return False

        # Validate SDK before publishing
        if not self.validate_sdk(language):
            self.console.print("[red]❌ SDK validation failed. Cannot publish.[/red]")
            return False

        # Get registry configuration
        registries = (
            self.config.get("ci_cd", {}).get("registries", {}).get(language, [])
        )

        if not registries:
            self.console.print(
                f"[yellow]⚠️  No registries configured for {language}[/yellow]"
            )
            return False

        # Use specified registry or default to first one
        if registry:
            target_registry = next(
                (r for r in registries if r["name"].lower() == registry.lower()), None
            )
        else:
            target_registry = registries[0]

        if not target_registry:
            self.console.print(
                f"[red]❌ Registry '{registry}' not found for {language}[/red]"
            )
            return False

        registry_name = target_registry["name"]

        if dry_run:
            self.console.print(
                f"[blue]🚀 [DRY RUN] Would publish {language.title()} SDK to {registry_name}[/blue]"
            )
            return True

        self.console.print(
            f"[blue]🚀 Publishing {language.title()} SDK to {registry_name}...[/blue]"
        )

        # Language-specific publish commands
        publish_commands = {
            "python": ["python", "-m", "twine", "upload", "dist/*"],
            "typescript": ["npm", "publish"],
            "java": ["mvn", "deploy"],
            "csharp": ["dotnet", "nuget", "push"],
            "php": ["composer", "publish"],
            "ruby": ["gem", "push"],
            "rust": ["cargo", "publish"],
        }

        if language not in publish_commands:
            self.console.print(
                f"[yellow]⚠️  No publish command configured for {language}[/yellow]"
            )
            return False

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"Publishing {language} SDK...", total=1)

                result = subprocess.run(
                    publish_commands[language],
                    cwd=sdk_dir,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    progress.update(
                        task,
                        advance=1,
                        description=f"✅ {language.title()} SDK published",
                    )
                    self.console.print(
                        f"[green]🎉 {language.title()} SDK published successfully to {registry_name}![/green]"
                    )
                    return True
                else:
                    progress.update(
                        task,
                        advance=1,
                        description=f"❌ {language.title()} SDK publish failed",
                    )
                    self.console.print(
                        f"[red]❌ Failed to publish {language.title()} SDK[/red]"
                    )
                    self.console.print(f"[dim]{result.stderr}[/dim]")
                    return False

        except subprocess.TimeoutExpired:
            self.console.print(
                f"[red]❌ Publish timed out for {language.title()} SDK[/red]"
            )
            return False
        except Exception as e:
            self.console.print(
                f"[red]❌ Error publishing {language.title()} SDK: {e}[/red]"
            )
            return False

    def show_sdk_info(self, language: str) -> None:
        """Show detailed information about a generated SDK."""
        sdk_dir = self.generator.output_dir / language

        if not sdk_dir.exists():
            self.console.print(f"[red]❌ SDK for {language} not found.[/red]")
            return

        # Create info tree
        tree = Tree(f"[bold blue]{language.title()} SDK Information[/bold blue]")

        # Basic info
        basic_info = tree.add("[bold green]Basic Information[/bold green]")
        lang_config = self.generator.languages.get(language, {})
        basic_info.add(f"Generator: {lang_config.get('generator', 'N/A')}")
        basic_info.add(f"Package Name: {lang_config.get('package_name', 'N/A')}")
        basic_info.add(f"Client Class: {lang_config.get('client_name', 'N/A')}")
        basic_info.add(f"Output Directory: {sdk_dir}")

        # File structure
        file_tree = tree.add("[bold yellow]File Structure[/bold yellow]")
        self._add_directory_to_tree(file_tree, sdk_dir, max_depth=2)

        # Features
        features_config = (
            self.config.get("languages", {}).get(language, {}).get("features", {})
        )
        if features_config:
            features_tree = tree.add("[bold magenta]Features[/bold magenta]")
            for feature, enabled in features_config.items():
                status = "✅" if enabled else "❌"
                features_tree.add(f"{status} {feature.replace('_', ' ').title()}")

        # Statistics
        stats = tree.add("[bold cyan]Statistics[/bold cyan]")
        file_count = len(list(sdk_dir.rglob("*")))
        code_files = (
            len(list(sdk_dir.rglob("*.py")))
            if language == "python"
            else len(list(sdk_dir.rglob("*")))
        )
        stats.add(f"Total Files: {file_count}")
        stats.add(f"Code Files: {code_files}")

        # Creation time
        if sdk_dir.exists():
            creation_time = datetime.fromtimestamp(sdk_dir.stat().st_mtime)
            stats.add(f"Last Modified: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.console.print(tree)

    def _add_directory_to_tree(
        self, tree_node, directory: Path, max_depth: int = 2, current_depth: int = 0
    ):
        """Recursively add directory structure to tree."""
        if current_depth >= max_depth:
            return

        try:
            items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name))
            for item in items[:10]:  # Limit to first 10 items
                if item.is_dir():
                    dir_node = tree_node.add(f"📁 {item.name}/")
                    if current_depth < max_depth - 1:
                        self._add_directory_to_tree(
                            dir_node, item, max_depth, current_depth + 1
                        )
                else:
                    tree_node.add(f"📄 {item.name}")

            # Show if there are more items
            total_items = len(list(directory.iterdir()))
            if total_items > 10:
                tree_node.add(f"... and {total_items - 10} more items")

        except PermissionError:
            tree_node.add("[red]Permission denied[/red]")


# CLI Commands
@click.group()
@click.version_option()
def cli():
    """anomaly_detection SDK Generator CLI - Generate client SDKs for multiple languages."""
    pass


@cli.command()
def list_languages():
    """List all supported programming languages."""
    manager = SDKCLIManager()
    manager.list_languages()


@cli.command()
@click.option("--languages", "-l", multiple=True, help="Languages to generate SDKs for")
@click.option("--all", is_flag=True, help="Generate SDKs for all supported languages")
@click.option(
    "--validate-only",
    is_flag=True,
    help="Only validate the OpenAPI spec without generating",
)
def generate(languages, all, validate_only):
    """Generate client SDKs for specified languages."""
    manager = SDKCLIManager()

    if all:
        target_languages = None
    elif languages:
        target_languages = list(languages)
    else:
        # Interactive selection
        manager.console.print(
            "[blue]No languages specified. Use --all for all languages or specify with -l[/blue]"
        )
        return

    results = manager.generate_sdks(target_languages, validate_only)

    if not any(results.values()) and not validate_only:
        sys.exit(1)


@cli.command()
@click.argument("language")
def test(language):
    """Run tests for a specific SDK."""
    manager = SDKCLIManager()

    if not manager.test_sdk(language):
        sys.exit(1)


@cli.command()
@click.argument("language")
def validate(language):
    """Validate a generated SDK against quality gates."""
    manager = SDKCLIManager()

    if not manager.validate_sdk(language):
        sys.exit(1)


@cli.command()
@click.argument("language")
@click.option("--registry", "-r", help="Target registry name")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be published without actually publishing",
)
def publish(language, registry, dry_run):
    """Publish SDK to package registry."""
    manager = SDKCLIManager()

    if not manager.publish_sdk(language, registry, dry_run):
        sys.exit(1)


@cli.command()
@click.argument("language")
def info(language):
    """Show detailed information about a generated SDK."""
    manager = SDKCLIManager()
    manager.show_sdk_info(language)


@cli.command()
def validate_environment():
    """Validate that all required tools are installed."""
    manager = SDKCLIManager()

    if not manager.validate_environment():
        sys.exit(1)


@cli.command()
@click.option(
    "--output", "-o", default="sdk_status.json", help="Output file for status report"
)
def status(output):
    """Generate comprehensive status report for all SDKs."""
    manager = SDKCLIManager()

    status_report = {
        "timestamp": datetime.now().isoformat(),
        "languages": {},
        "summary": {},
    }

    for language in manager.generator.languages.keys():
        sdk_dir = manager.generator.output_dir / language

        if sdk_dir.exists():
            # Check if tests pass
            test_passed = manager.test_sdk(language)

            # Check validation
            validation_passed = manager.validate_sdk(language)

            status_report["languages"][language] = {
                "exists": True,
                "last_modified": datetime.fromtimestamp(
                    sdk_dir.stat().st_mtime
                ).isoformat(),
                "test_passed": test_passed,
                "validation_passed": validation_passed,
                "files_count": len(list(sdk_dir.rglob("*"))),
            }
        else:
            status_report["languages"][language] = {"exists": False}

    # Generate summary
    existing_sdks = sum(
        1
        for lang_status in status_report["languages"].values()
        if lang_status.get("exists")
    )
    passing_tests = sum(
        1
        for lang_status in status_report["languages"].values()
        if lang_status.get("test_passed")
    )
    passing_validation = sum(
        1
        for lang_status in status_report["languages"].values()
        if lang_status.get("validation_passed")
    )

    status_report["summary"] = {
        "total_languages": len(manager.generator.languages),
        "existing_sdks": existing_sdks,
        "passing_tests": passing_tests,
        "passing_validation": passing_validation,
    }

    # Save report
    with open(output, "w") as f:
        json.dump(status_report, f, indent=2)

    manager.console.print(f"[green]📊 Status report saved to {output}[/green]")


if __name__ == "__main__":
    cli()
