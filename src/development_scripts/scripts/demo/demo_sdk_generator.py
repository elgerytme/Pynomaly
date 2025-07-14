#!/usr/bin/env python3
"""
SDK Generator Demo Script

This script demonstrates the capabilities of the Pynomaly SDK Generator
by generating a sample client library and showcasing its features.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from scripts.cli.sdk_cli import SDKCLIManager
from scripts.sdk_generator import SDKGenerator


class SDKGeneratorDemo:
    """Demo class for showcasing SDK Generator capabilities."""

    def __init__(self):
        self.console = Console()
        self.generator = SDKGenerator()
        self.cli_manager = SDKCLIManager()

    def print_header(self, title: str):
        """Print a styled header."""
        panel = Panel(f"[bold blue]{title}[/bold blue]", style="blue", expand=True)
        self.console.print(panel)

    def print_step(self, step: str, description: str):
        """Print a demo step."""
        self.console.print(f"\n[bold green]Step {step}:[/bold green] {description}")

    def demo_overview(self):
        """Show overview of SDK Generator capabilities."""
        self.print_header("Pynomaly SDK Generator Demo")

        self.console.print("""
[bold]Welcome to the Pynomaly SDK Generator Demo![/bold]

This demo will showcase the key capabilities of our multi-language SDK generator:

1. 🌐 Support for 8+ programming languages
2. 🚀 Automatic code generation from OpenAPI specs
3. 🔒 Built-in authentication and security
4. 🔄 Error handling and retry logic
5. 📚 Comprehensive documentation and examples
6. 🧪 Automated testing and validation
7. 📦 CI/CD integration and publishing
8. ⚙️  Extensive customization options
        """)

    def demo_supported_languages(self):
        """Demonstrate supported languages."""
        self.print_step("1", "Supported Programming Languages")

        table = Table(title="Supported Languages and Features")
        table.add_column("Language", style="cyan", width=12)
        table.add_column("Generator", style="blue", width=15)
        table.add_column("Package Manager", style="green", width=15)
        table.add_column("Key Features", style="yellow")

        language_features = {
            "python": ["Async/Await", "Type Hints", "Pydantic", "pytest"],
            "typescript": ["Type Definitions", "Node.js/Browser", "Jest", "ESLint"],
            "java": ["Maven/Gradle", "Java 11+", "JUnit 5", "OkHttp"],
            "go": ["Go Modules", "Context", "Built-in Testing", "Standard Library"],
            "csharp": [".NET 6.0+", "NuGet", "xUnit", "HttpClient"],
            "php": ["Composer", "PSR-4", "PHPUnit", "Guzzle"],
            "ruby": ["Bundler", "RSpec", "RubyGems", "Faraday"],
            "rust": ["Cargo", "Async", "Tokio", "Reqwest"],
        }

        for lang, config in self.generator.languages.items():
            features = language_features.get(lang, ["Standard Features"])
            table.add_row(
                lang.title(),
                config["generator"],
                self._get_package_manager(lang),
                ", ".join(features),
            )

        self.console.print(table)

    def _get_package_manager(self, language: str) -> str:
        """Get package manager for language."""
        managers = {
            "python": "pip/PyPI",
            "typescript": "npm",
            "java": "Maven/Gradle",
            "go": "Go modules",
            "csharp": "NuGet",
            "php": "Composer",
            "ruby": "Bundler/Gem",
            "rust": "Cargo",
        }
        return managers.get(language, "Standard")

    def demo_configuration(self):
        """Demonstrate configuration capabilities."""
        self.print_step("2", "Configuration and Customization")

        self.console.print("""
[bold]Configuration Features:[/bold]

• [green]✅ Language-specific settings[/green] - Customize each language independently
• [green]✅ Feature toggles[/green] - Enable/disable async, retry logic, rate limiting
• [green]✅ Package configuration[/green] - Custom names, versions, dependencies
• [green]✅ Template overrides[/green] - Custom code templates and examples
• [green]✅ Post-generation scripts[/green] - Automated formatting and validation
• [green]✅ Quality gates[/green] - Enforce quality standards and requirements
        """)

        # Show sample configuration
        sample_config = {
            "languages": {
                "python": {
                    "enabled": True,
                    "package": {"name": "pynomaly_client"},
                    "features": {
                        "async_support": True,
                        "retry_logic": True,
                        "rate_limiting": True,
                    },
                }
            }
        }

        self.console.print("\n[bold]Sample Configuration:[/bold]")
        self.console.print(f"[dim]{yaml.dump(sample_config, indent=2)}[/dim]")

    def demo_generation_process(self):
        """Demonstrate the generation process."""
        self.print_step("3", "SDK Generation Process")

        tree = Tree("[bold blue]Generation Workflow[/bold blue]")

        # Prerequisites
        prereq = tree.add("[yellow]1. Prerequisites Validation[/yellow]")
        prereq.add("✓ OpenAPI Generator CLI")
        prereq.add("✓ Language-specific tools")
        prereq.add("✓ OpenAPI specification")

        # Generation
        generation = tree.add("[yellow]2. Code Generation[/yellow]")
        generation.add("📋 Load OpenAPI specification")
        generation.add("🔧 Configure language-specific settings")
        generation.add("⚙️  Execute OpenAPI Generator")
        generation.add("📝 Generate client code")

        # Post-processing
        postproc = tree.add("[yellow]3. Post-Processing[/yellow]")
        postproc.add("📚 Generate custom documentation")
        postproc.add("💡 Add usage examples")
        postproc.add("🔧 Apply code formatting")
        postproc.add("⚙️  Setup CI/CD configuration")

        # Validation
        validation = tree.add("[yellow]4. Validation & Testing[/yellow]")
        validation.add("🧪 Run quality gates")
        validation.add("✅ Execute test suites")
        validation.add("🔍 Validate compilation")
        validation.add("📊 Generate reports")

        self.console.print(tree)

    def demo_generated_features(self):
        """Demonstrate features of generated SDKs."""
        self.print_step("4", "Generated SDK Features")

        features_table = Table(title="SDK Features Matrix")
        features_table.add_column("Feature", style="cyan")
        features_table.add_column("Description", style="blue")
        features_table.add_column("All Languages", style="green")
        features_table.add_column("Selected Languages", style="yellow")

        features = [
            ("Authentication", "JWT & API Key support", "✅", ""),
            ("Error Handling", "Comprehensive exception handling", "✅", ""),
            ("Rate Limiting", "Automatic backoff and retry", "✅", ""),
            ("Documentation", "API docs and examples", "✅", ""),
            (
                "Type Safety",
                "Strong typing where supported",
                "",
                "Python, TS, Java, C#, Rust",
            ),
            ("Async Support", "Asynchronous operations", "", "Python, TS, C#, Rust"),
            ("Testing", "Unit test suites", "✅", ""),
            ("CI/CD", "GitHub Actions workflows", "✅", ""),
            ("Packaging", "Ready-to-publish packages", "✅", ""),
        ]

        for feature, description, all_langs, selected_langs in features:
            features_table.add_row(feature, description, all_langs, selected_langs)

        self.console.print(features_table)

    def demo_cli_commands(self):
        """Demonstrate CLI commands."""
        self.print_step("5", "CLI Commands and Usage")

        commands_table = Table(title="CLI Commands")
        commands_table.add_column("Command", style="cyan")
        commands_table.add_column("Description", style="blue")
        commands_table.add_column("Example", style="green")

        commands = [
            ("list-languages", "Show supported languages", "sdk_cli.py list-languages"),
            (
                "generate",
                "Generate SDKs",
                "sdk_cli.py generate --languages python typescript",
            ),
            ("test", "Run SDK tests", "sdk_cli.py test python"),
            ("validate", "Validate SDK quality", "sdk_cli.py validate python"),
            (
                "publish",
                "Publish to registry",
                "sdk_cli.py publish python --registry pypi",
            ),
            ("info", "Show SDK information", "sdk_cli.py info python"),
            (
                "status",
                "Generate status report",
                "sdk_cli.py status --output report.json",
            ),
        ]

        for command, description, example in commands:
            commands_table.add_row(command, description, example)

        self.console.print(commands_table)

    def demo_code_examples(self):
        """Show code examples from generated SDKs."""
        self.print_step("6", "Generated Code Examples")

        # Python example
        python_example = """# Python SDK Example
from pynomaly_client import PynomaliClient
from pynomaly_client.exceptions import ApiException

async def main():
    # Initialize client
    client = PynomaliClient(base_url="https://api.pynomaly.com")

    try:
        # Authenticate
        token = await client.auth.login("username", "password")

        # Detect anomalies
        result = await client.detection.detect(
            data=[1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
            algorithm="isolation_forest",
            parameters={"contamination": 0.1}
        )

        print(f"Anomalies detected: {result.anomalies}")

    except ApiException as e:
        print(f"API Error: {e.status} - {e.reason}")

    finally:
        await client.close()"""

        # TypeScript example
        typescript_example = """// TypeScript SDK Example
import { PynomaliClient } from '@pynomaly/client';

const client = new PynomaliClient({
    basePath: 'https://api.pynomaly.com'
});

async function detectAnomalies() {
    try {
        // Authenticate
        const tokenResponse = await client.auth.login({
            username: 'username',
            password: 'password'
        });

        client.setAccessToken(tokenResponse.access_token);

        // Detect anomalies
        const result = await client.detection.detect({
            data: [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
            algorithm: 'isolation_forest',
            parameters: { contamination: 0.1 }
        });

        console.log('Anomalies:', result.anomalies);

    } catch (error) {
        console.error('API Error:', error);
    }
}"""

        self.console.print("\n[bold]Python SDK Example:[/bold]")
        self.console.print(f"[dim]{python_example}[/dim]")

        self.console.print("\n[bold]TypeScript SDK Example:[/bold]")
        self.console.print(f"[dim]{typescript_example}[/dim]")

    def demo_quality_assurance(self):
        """Demonstrate quality assurance features."""
        self.print_step("7", "Quality Assurance and Validation")

        qa_tree = Tree("[bold blue]Quality Assurance Pipeline[/bold blue]")

        # Pre-generation validation
        pre_gen = qa_tree.add("[yellow]Pre-Generation Validation[/yellow]")
        pre_gen.add("✓ OpenAPI specification validation")
        pre_gen.add("✓ Environment prerequisites check")
        pre_gen.add("✓ Configuration validation")

        # Generation quality
        gen_quality = qa_tree.add("[yellow]Generation Quality[/yellow]")
        gen_quality.add("✓ Code compilation/syntax validation")
        gen_quality.add("✓ Template consistency checks")
        gen_quality.add("✓ File structure validation")

        # Post-generation testing
        post_gen = qa_tree.add("[yellow]Post-Generation Testing[/yellow]")
        post_gen.add("✓ Unit test execution")
        post_gen.add("✓ Integration test validation")
        post_gen.add("✓ Code style and linting")

        # Quality gates
        quality_gates = qa_tree.add("[yellow]Quality Gates[/yellow]")
        quality_gates.add("✓ Authentication support verification")
        quality_gates.add("✓ Error handling completeness")
        quality_gates.add("✓ Documentation coverage")
        quality_gates.add("✓ Test coverage requirements")

        self.console.print(qa_tree)

    def demo_publishing_workflow(self):
        """Demonstrate publishing workflow."""
        self.print_step("8", "Publishing and Distribution")

        self.console.print("""
[bold]Publishing Workflow:[/bold]

1. [blue]Validation[/blue] - Ensure SDK passes all quality gates
2. [blue]Testing[/blue] - Execute comprehensive test suite
3. [blue]Building[/blue] - Create distribution packages
4. [blue]Publishing[/blue] - Upload to package registries

[bold]Supported Registries:[/bold]

• [green]Python[/green] → PyPI (pip install pynomaly-client)
• [green]TypeScript[/green] → npm (npm install @pynomaly/client)
• [green]Java[/green] → Maven Central (Maven/Gradle dependency)
• [green]Go[/green] → Go modules (go get github.com/pynomaly/go-client)
• [green]C#[/green] → NuGet (NuGet package)
• [green]PHP[/green] → Packagist (Composer package)
• [green]Ruby[/green] → RubyGems (gem install pynomaly_client)
• [green]Rust[/green] → crates.io (cargo add pynomaly_client)
        """)

    def demo_summary(self):
        """Show demo summary."""
        self.print_step("9", "Summary and Next Steps")

        summary_panel = Panel(
            """
[bold green]✅ Demo Complete![/bold green]

[bold]What we've covered:[/bold]
• 8+ programming languages supported
• Comprehensive feature set for each SDK
• Flexible configuration and customization
• Quality assurance and validation
• Complete publishing workflow

[bold]Getting Started:[/bold]
1. Install dependencies: [cyan]pip install -r requirements.txt[/cyan]
2. Validate environment: [cyan]python scripts/cli/sdk_cli.py validate-environment[/cyan]
3. Generate SDKs: [cyan]python scripts/cli/sdk_cli.py generate --all[/cyan]
4. Test and validate: [cyan]python scripts/cli/sdk_cli.py test python[/cyan]

[bold]Resources:[/bold]
• Documentation: [blue]docs/developer-guides/SDK_GENERATOR_GUIDE.md[/blue]
• Configuration: [blue]config/sdk_generator_config.yaml[/blue]
• CLI Help: [blue]python scripts/cli/sdk_cli.py --help[/blue]
            """,
            title="Demo Summary",
            style="green",
        )

        self.console.print(summary_panel)

    async def run_interactive_demo(self):
        """Run an interactive demo."""
        self.print_header("Interactive SDK Generator Demo")

        self.console.print("""
[bold]Choose a demo section to explore:[/bold]

1. Supported Languages Overview
2. Configuration and Customization
3. Generation Process Walkthrough
4. Generated SDK Features
5. CLI Commands and Usage
6. Code Examples
7. Quality Assurance Pipeline
8. Publishing Workflow
9. Run Complete Demo
0. Exit
        """)

        while True:
            try:
                choice = self.console.input(
                    "\n[bold blue]Enter your choice (0-9): [/bold blue]"
                )

                if choice == "0":
                    self.console.print(
                        "[green]Thanks for exploring the SDK Generator demo![/green]"
                    )
                    break
                elif choice == "1":
                    self.demo_supported_languages()
                elif choice == "2":
                    self.demo_configuration()
                elif choice == "3":
                    self.demo_generation_process()
                elif choice == "4":
                    self.demo_generated_features()
                elif choice == "5":
                    self.demo_cli_commands()
                elif choice == "6":
                    self.demo_code_examples()
                elif choice == "7":
                    self.demo_quality_assurance()
                elif choice == "8":
                    self.demo_publishing_workflow()
                elif choice == "9":
                    await self.run_complete_demo()
                else:
                    self.console.print("[red]Invalid choice. Please enter 0-9.[/red]")

                # Pause before showing menu again
                if choice != "0":
                    self.console.input("\n[dim]Press Enter to continue...[/dim]")
                    self.console.clear()

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Demo interrupted by user.[/yellow]")
                break
            except EOFError:
                break

    async def run_complete_demo(self):
        """Run the complete demo."""
        sections = [
            ("Overview", self.demo_overview),
            ("Supported Languages", self.demo_supported_languages),
            ("Configuration", self.demo_configuration),
            ("Generation Process", self.demo_generation_process),
            ("Generated Features", self.demo_generated_features),
            ("CLI Commands", self.demo_cli_commands),
            ("Code Examples", self.demo_code_examples),
            ("Quality Assurance", self.demo_quality_assurance),
            ("Publishing", self.demo_publishing_workflow),
            ("Summary", self.demo_summary),
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            for section_name, section_func in sections:
                task = progress.add_task(f"Running {section_name} demo...", total=1)

                # Clear screen and run section
                self.console.clear()
                section_func()

                progress.update(task, advance=1)

                # Brief pause between sections
                await asyncio.sleep(2)

        self.console.print("\n[bold green]🎉 Complete demo finished![/bold green]")


async def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(description="Pynomaly SDK Generator Demo")
    parser.add_argument(
        "--mode",
        choices=["interactive", "complete", "overview"],
        default="interactive",
        help="Demo mode to run",
    )

    args = parser.parse_args()

    demo = SDKGeneratorDemo()

    if args.mode == "interactive":
        await demo.run_interactive_demo()
    elif args.mode == "complete":
        await demo.run_complete_demo()
    elif args.mode == "overview":
        demo.demo_overview()
        demo.demo_supported_languages()
        demo.demo_summary()


if __name__ == "__main__":
    asyncio.run(main())
