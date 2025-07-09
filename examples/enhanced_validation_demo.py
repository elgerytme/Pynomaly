#!/usr/bin/env python3
"""
Enhanced Validation Demo

This script demonstrates the enhanced validation system with:
- Rich console output with colorized violations grouped by severity
- GitHub comment generation for PR integration
- Pre-commit hook integration
- Developer-friendly UX enhancements
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for demo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pynomaly.presentation.cli.validation import (
    EnhancedValidator,
    GitHubCommentGenerator,
    RichOutputFormatter,
    ValidationViolation,
    ViolationSeverity,
)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def create_demo_project():
    """Create a demo project with various validation issues."""
    print("🏗️  Creating demo project with validation issues...")
    
    # Create temporary directory
    demo_dir = Path(tempfile.mkdtemp(prefix="pynomaly_validation_demo_"))
    print(f"📁 Demo project created at: {demo_dir}")
    
    # Create directory structure
    (demo_dir / "src").mkdir()
    (demo_dir / "tests").mkdir()
    (demo_dir / "docs").mkdir()
    
    # Create forbidden directory (will trigger HIGH severity violation)
    (demo_dir / "build").mkdir()
    
    # Create Python file with various issues
    py_file = demo_dir / "src" / "problematic.py"
    py_file.write_text("""#!/usr/bin/env python3
# TODO: This file has intentional issues for demo purposes
# FIXME: Multiple issues to demonstrate validation

import subprocess
import os

def insecure_function(user_input):
    \"\"\"Function with security issues.\"\"\"
    print(f"Debug: Processing {user_input}")  # Debug print statement
    
    # Critical security issue
    result = eval(user_input)
    
    # Another security issue
    os.system(f"echo {user_input}")
    
    return result

def poor_quality_function():
    # No docstring
    data = []
    for i in range(100):
        for j in range(100):
            for k in range(100):  # Deeply nested loops
                if i > 0:
                    if j > 0:
                        if k > 0:
                            if i + j + k > 150:  # Complex conditions
                                data.append((i, j, k))
                                print("Adding:", i, j, k)  # More debug prints
    return data

class UndocumentedClass:
    def method_without_docstring(self):
        pass
""")
    
    # Create another file with different issues
    py_file2 = demo_dir / "src" / "another_issue.py"
    py_file2.write_text("""
# Another file with issues
def function_with_exec(code):
    exec(code)  # Critical security issue
    return "executed"

# TODO: Add proper error handling
def risky_subprocess(cmd):
    subprocess.call(cmd, shell=True)  # Security issue
""")
    
    # Create a clean file for contrast
    clean_file = demo_dir / "src" / "clean.py"
    clean_file.write_text("""\"\"\"Clean Python module for demo purposes.\"\"\"

from typing import List


def clean_function(items: List[str]) -> str:
    \"\"\"A properly documented function with type hints.
    
    Args:
        items: List of strings to process
        
    Returns:
        Processed string result
    \"\"\"
    return " ".join(items)


class WellDocumentedClass:
    \"\"\"A well-documented class example.\"\"\"
    
    def __init__(self, name: str) -> None:
        \"\"\"Initialize the class.
        
        Args:
            name: The name for this instance
        \"\"\"
        self.name = name
    
    def get_name(self) -> str:
        \"\"\"Return the name.
        
        Returns:
            The instance name
        \"\"\"
        return self.name
""")
    
    # Create missing README (will trigger HIGH severity violation)
    # Intentionally not creating README.md to trigger violation
    
    return demo_dir


def demo_rich_output():
    """Demonstrate rich console output with colorized violations."""
    console = Console()
    
    console.print(Panel(
        "[bold blue]Enhanced Validation Demo[/bold blue]\n"
        "Demonstrating rich console output with colorized violations grouped by severity",
        title="🔍 Validation Demo",
        border_style="blue"
    ))
    
    # Create demo project
    demo_dir = create_demo_project()
    
    try:
        # Run validation
        console.print("\n[bold]Running enhanced validation...[/bold]")
        validator = EnhancedValidator(demo_dir)
        result = validator.validate_project()
        
        # Display results with rich formatting
        formatter = RichOutputFormatter(console)
        formatter.display_results(result)
        
        # Show violation details
        console.print("\n[bold]Violation Details:[/bold]")
        grouped = result.group_by_severity()
        
        for severity, violations in grouped.items():
            if violations:
                console.print(f"\n[bold]{severity.value.upper()} Violations ({len(violations)}):[/bold]")
                for i, violation in enumerate(violations, 1):
                    console.print(f"  {i}. {violation.message}")
                    if violation.fix_suggestion:
                        console.print(f"     💡 Fix: {violation.fix_suggestion}")
        
        return result
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(demo_dir)
        console.print(f"\n🧹 Cleaned up demo project at: {demo_dir}")


def demo_github_comment():
    """Demonstrate GitHub comment generation."""
    console = Console()
    
    console.print(Panel(
        "[bold green]GitHub Comment Generation Demo[/bold green]\n"
        "Demonstrating PR comment generation with violation summaries and fix suggestions",
        title="📝 GitHub Integration",
        border_style="green"
    ))
    
    # Create demo project
    demo_dir = create_demo_project()
    
    try:
        # Run validation
        validator = EnhancedValidator(demo_dir)
        result = validator.validate_project()
        
        # Generate GitHub comment
        comment_generator = GitHubCommentGenerator()
        comment = comment_generator.generate_comment(result)
        
        # Display the comment
        console.print("\n[bold]Generated GitHub Comment:[/bold]")
        console.print(Panel(comment, title="GitHub PR Comment", border_style="blue"))
        
        # Show environment variables needed for posting
        console.print("\n[bold]Environment Variables for GitHub Integration:[/bold]")
        env_vars = [
            ("GITHUB_TOKEN", "GitHub personal access token"),
            ("GITHUB_REPOSITORY", "Repository name (e.g., 'owner/repo')"),
            ("GITHUB_PR_NUMBER", "Pull request number"),
        ]
        
        for var, description in env_vars:
            value = os.getenv(var, "Not set")
            console.print(f"  {var}: {value} ({description})")
        
        return comment
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(demo_dir)


def demo_pre_commit_integration():
    """Demonstrate pre-commit hook integration."""
    console = Console()
    
    console.print(Panel(
        "[bold yellow]Pre-commit Integration Demo[/bold yellow]\n"
        "Demonstrating pre-commit hook integration and developer reminders",
        title="🔧 Pre-commit Hooks",
        border_style="yellow"
    ))
    
    # Check pre-commit status
    from pynomaly.presentation.cli.validation import check_pre_commit_installed, show_pre_commit_reminder
    
    console.print("\n[bold]Pre-commit Status Check:[/bold]")
    
    if check_pre_commit_installed():
        console.print("✅ Pre-commit is installed!")
        
        # Check if hooks are active
        try:
            hooks_path = Path(".git/hooks/pre-commit")
            if hooks_path.exists():
                console.print("✅ Pre-commit hooks are active!")
            else:
                console.print("⚠️  Pre-commit installed but hooks not active.")
                console.print("Run: [cyan]pre-commit install[/cyan]")
        except Exception:
            console.print("⚠️  Could not check git hooks status.")
            
    else:
        console.print("❌ Pre-commit is not installed.")
        show_pre_commit_reminder()
    
    # Show pre-commit configuration
    console.print("\n[bold]Pre-commit Configuration:[/bold]")
    console.print("The enhanced validation is integrated into .pre-commit-config.yaml:")
    
    config_example = """
- repo: local
  hooks:
    - id: enhanced-validation
      name: Enhanced Validation with Rich Output
      entry: python -m pynomaly.presentation.cli.app validate run --format rich
      language: python
      pass_filenames: false
      always_run: true
      stages: [pre-commit]
      verbose: true
"""
    
    console.print(Panel(config_example, title="Pre-commit Config", border_style="cyan"))


def demo_cli_usage():
    """Demonstrate CLI usage examples."""
    console = Console()
    
    console.print(Panel(
        "[bold magenta]CLI Usage Demo[/bold magenta]\n"
        "Demonstrating command-line interface usage and options",
        title="💻 CLI Usage",
        border_style="magenta"
    ))
    
    # Show CLI commands
    cli_examples = [
        ("Basic validation", "pynomaly validate run"),
        ("Rich output format", "pynomaly validate run --format rich"),
        ("JSON output format", "pynomaly validate run --format json"),
        ("Save report to file", "pynomaly validate run --save report.json"),
        ("GitHub comment mode", "pynomaly validate run --github-comment"),
        ("Validate specific path", "pynomaly validate run /path/to/project"),
        ("Check pre-commit status", "pynomaly validate check-pre-commit"),
        ("Install pre-commit hooks", "pynomaly validate install-hooks"),
    ]
    
    console.print("\n[bold]Available CLI Commands:[/bold]")
    for description, command in cli_examples:
        console.print(f"  [cyan]{command}[/cyan]")
        console.print(f"    {description}")
        console.print()
    
    # Show integration with existing CLI
    console.print("\n[bold]Integration with Existing CLI:[/bold]")
    console.print("The enhanced validation is integrated into the main pynomaly CLI:")
    console.print("  [cyan]pynomaly validate --help[/cyan]")
    console.print("  [cyan]pynomaly validate run --help[/cyan]")


def main():
    """Run the complete enhanced validation demo."""
    console = Console()
    
    # Main demo header
    console.print(Panel(
        Text.assemble(
            ("🚀 ", "bold blue"),
            ("Enhanced Validation System Demo", "bold white"),
            ("\n\n", ""),
            ("This demo showcases the three main developer UX enhancements:\n", ""),
            ("1. ", "bold"), ("Colorised console output (rich) grouping violations by severity", ""),
            ("\n2. ", "bold"), ("GitHub PR comments with violation summaries and fix suggestions", ""),
            ("\n3. ", "bold"), ("Pre-commit hook integration with developer reminders", ""),
        ),
        title="Pynomaly Enhanced Validation",
        border_style="bold blue",
        padding=(1, 2)
    ))
    
    try:
        # Run demos
        console.print("\n" + "="*70)
        console.print("DEMO 1: Rich Console Output")
        console.print("="*70)
        result = demo_rich_output()
        
        console.print("\n" + "="*70)
        console.print("DEMO 2: GitHub Comment Generation")
        console.print("="*70)
        comment = demo_github_comment()
        
        console.print("\n" + "="*70)
        console.print("DEMO 3: Pre-commit Integration")
        console.print("="*70)
        demo_pre_commit_integration()
        
        console.print("\n" + "="*70)
        console.print("DEMO 4: CLI Usage Examples")
        console.print("="*70)
        demo_cli_usage()
        
        # Summary
        console.print("\n" + "="*70)
        console.print("SUMMARY")
        console.print("="*70)
        
        console.print(Panel(
            Text.assemble(
                ("✅ ", "green"), ("Rich Console Output: ", "bold"), ("Colorized violations grouped by severity\n", ""),
                ("✅ ", "green"), ("GitHub Integration: ", "bold"), ("PR comments with fix suggestions\n", ""),
                ("✅ ", "green"), ("Pre-commit Hooks: ", "bold"), ("Automated validation with developer reminders\n", ""),
                ("✅ ", "green"), ("CLI Integration: ", "bold"), ("Seamless integration with existing pynomaly CLI\n", ""),
                ("\n", ""),
                ("🎯 ", "bold blue"), ("Developer Experience: ", "bold"), ("Enhanced UX with actionable feedback", ""),
            ),
            title="🎉 Demo Complete",
            border_style="green"
        ))
        
    except KeyboardInterrupt:
        console.print("\n⏹️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
