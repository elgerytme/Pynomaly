"""Health check command for Pynomaly CLI.

This module provides health check functionality to validate the Pynomaly installation
and its dependencies.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pynomaly.presentation.cli.ux_improvements import CLIHelpers

app = typer.Typer(help="🏥 Health check and dependency validation commands")
console = Console()
cli_helpers = CLIHelpers()


@app.command()
def dependencies(
    groups: Optional[str] = typer.Option(
        None,
        "--groups",
        "-g",
        help="Comma-separated list of optional groups to check (api,cli,ml,all)",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        "-f",
        help="Attempt to automatically fix missing dependencies",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed validation output",
    ),
) -> None:
    """Validate Pynomaly dependencies and installation health."""
    
    cli_helpers.print_header("Pynomaly Health Check", "🏥")
    
    # Find and run the validation script
    script_path = Path(__file__).parent.parent.parent.parent.parent / "scripts" / "validate_dependencies.py"
    
    if not script_path.exists():
        console.print("❌ Validation script not found", style="red")
        console.print(f"Expected location: {script_path}")
        raise typer.Exit(1)
    
    # Build command arguments
    cmd = [sys.executable, str(script_path)]
    
    if groups:
        group_list = [g.strip() for g in groups.split(",")]
        cmd.extend(["--groups"] + group_list)
    
    if fix:
        cmd.append("--fix")
    
    try:
        # Run the validation script
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=False
        )
        
        if verbose or result.returncode != 0:
            if result.stdout:
                console.print(result.stdout)
            if result.stderr:
                console.print(result.stderr, style="red")
        
        if result.returncode == 0:
            console.print("✅ Health check passed!", style="green")
        else:
            console.print("❌ Health check found issues", style="red")
            if not verbose:
                console.print("💡 Run with --verbose for detailed output")
        
        raise typer.Exit(result.returncode)
        
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Health check failed: {e}", style="red")
        raise typer.Exit(1)
    except FileNotFoundError:
        console.print("❌ Python interpreter not found", style="red")
        raise typer.Exit(1)


@app.command()
def system(
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed system information",
    )
) -> None:
    """Check system compatibility and environment."""
    
    cli_helpers.print_header("System Health Check", "🖥️")
    
    # Create system info table
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="white")
    
    # Python version check
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_status = "✅ Compatible" if sys.version_info >= (3, 11) else "❌ Incompatible"
    table.add_row("Python Version", python_status, python_version)
    
    # Platform info
    import platform
    table.add_row("Platform", "ℹ️ Info", platform.platform())
    table.add_row("Architecture", "ℹ️ Info", platform.machine())
    
    # Memory check
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_status = "✅ Sufficient" if memory_gb >= 2 else "⚠️ Limited"
        table.add_row("Available Memory", memory_status, f"{memory_gb:.1f} GB")
    except ImportError:
        table.add_row("Available Memory", "❓ Unknown", "psutil not installed")
    
    # Virtual environment check
    venv_status = "✅ Active" if hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix else "⚠️ Global"
    venv_path = sys.prefix
    table.add_row("Virtual Environment", venv_status, venv_path)
    
    console.print(table)
    
    if detailed:
        console.print("\n📋 Detailed Environment:")
        console.print(f"Executable: {sys.executable}")
        console.print(f"Python Path: {sys.path[0]}")
        
        # Check for common issues
        issues = []
        if sys.version_info < (3, 11):
            issues.append("Python version too old (requires >=3.11)")
        
        try:
            import psutil
            if psutil.virtual_memory().total < 2 * 1024**3:
                issues.append("Low memory (recommended: >=2GB)")
        except ImportError:
            pass
        
        if issues:
            console.print("\n⚠️ Potential Issues:", style="yellow")
            for issue in issues:
                console.print(f"  - {issue}")
        else:
            console.print("\n✅ No system issues detected", style="green")


@app.command()
def connectivity(
    timeout: int = typer.Option(
        10,
        "--timeout",
        "-t",
        help="Connection timeout in seconds",
    )
) -> None:
    """Check network connectivity for package installation."""
    
    cli_helpers.print_header("Connectivity Health Check", "🌐")
    
    # Test connections to package repositories
    test_urls = [
        ("PyPI", "https://pypi.org/simple/"),
        ("Python.org", "https://www.python.org/"),
        ("GitHub", "https://github.com/"),
    ]
    
    table = Table(title="Connectivity Test")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Response Time", style="white")
    
    try:
        import requests
        import time
        
        for name, url in test_urls:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=timeout, allow_redirects=True)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    status = "✅ Connected"
                    timing = f"{response_time:.2f}s"
                else:
                    status = f"⚠️ Status {response.status_code}"
                    timing = f"{response_time:.2f}s"
                    
            except requests.exceptions.Timeout:
                status = "❌ Timeout"
                timing = f">{timeout}s"
            except requests.exceptions.ConnectionError:
                status = "❌ Connection Failed"
                timing = "N/A"
            except Exception as e:
                status = f"❌ Error: {type(e).__name__}"
                timing = "N/A"
            
            table.add_row(name, status, timing)
    
    except ImportError:
        table.add_row("All Services", "❓ Cannot Test", "requests not installed")
    
    console.print(table)
    
    # Provide recommendations
    failed_tests = [row for row in table.rows if "❌" in str(row[1])]
    if failed_tests:
        console.print("\n💡 Recommendations:", style="yellow")
        console.print("  - Check your internet connection")
        console.print("  - Verify firewall/proxy settings")
        console.print("  - Consider using a different network")
    else:
        console.print("\n✅ All connectivity tests passed", style="green")


@app.command()
def full(
    fix: bool = typer.Option(
        False,
        "--fix",
        "-f",
        help="Attempt to automatically fix issues",
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed output for all checks",
    ),
) -> None:
    """Run complete health check (system + dependencies + connectivity)."""
    
    cli_helpers.print_header("Complete Pynomaly Health Check", "🔬")
    
    all_passed = True
    
    try:
        # System check
        console.print("🖥️ Running system check...\n")
        system(detailed=detailed)
        console.print()
        
        # Connectivity check
        console.print("🌐 Running connectivity check...\n")
        connectivity()
        console.print()
        
        # Dependencies check
        console.print("📦 Running dependencies check...\n")
        dependencies(groups="all", fix=fix, verbose=detailed)
        
    except typer.Exit as e:
        if e.exit_code != 0:
            all_passed = False
    
    # Final summary
    console.print("\n" + "="*50)
    if all_passed:
        console.print("🎉 Complete health check PASSED!", style="green bold")
        console.print("Your Pynomaly installation is ready to use.", style="green")
    else:
        console.print("⚠️ Health check found issues", style="yellow bold")
        console.print("Please review the results above and take recommended actions.", style="yellow")
    
    raise typer.Exit(0 if all_passed else 1)


if __name__ == "__main__":
    app()