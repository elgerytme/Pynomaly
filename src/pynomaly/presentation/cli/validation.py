"""Enhanced validation CLI with rich output and GitHub integration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from pynomaly.domain.value_objects.severity_score import SeverityScore
from pynomaly.docs_validation.core.config import ValidationConfig
from pynomaly.docs_validation.core.reporter import ValidationReporter
from pynomaly.docs_validation.core.validator import DocumentationValidator

app = typer.Typer(name="validation", help="🔍 Enhanced validation with rich output and GitHub integration")

console = Console()


class ViolationSeverity(str, Enum):
    """Violation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationViolation:
    """Represents a validation violation."""
    
    def __init__(
        self,
        message: str,
        severity: ViolationSeverity,
        file_path: str,
        line_number: Optional[int] = None,
        rule_id: Optional[str] = None,
        fix_suggestion: Optional[str] = None,
    ):
        self.message = message
        self.severity = severity
        self.file_path = file_path
        self.line_number = line_number
        self.rule_id = rule_id
        self.fix_suggestion = fix_suggestion
        
    def __repr__(self):
        return f"ValidationViolation(message='{self.message}', severity={self.severity}, file={self.file_path})"


class ValidationResult:
    """Represents the result of a validation run."""
    
    def __init__(self):
        self.violations: List[ValidationViolation] = []
        self.passed = True
        self.file_count = 0
        self.duration_seconds = 0.0
        self.metrics: Dict[str, any] = {}
    
    @property
    def errors(self) -> List[str]:
        """Get error messages for compatibility."""
        return [v.message for v in self.violations if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH]]
    
    @property
    def warnings(self) -> List[str]:
        """Get warning messages for compatibility."""
        return [v.message for v in self.violations if v.severity in [ViolationSeverity.MEDIUM, ViolationSeverity.LOW]]
    
    def add_violation(self, violation: ValidationViolation):
        """Add a validation violation."""
        self.violations.append(violation)
        if violation.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH]:
            self.passed = False
    
    def group_by_severity(self) -> Dict[ViolationSeverity, List[ValidationViolation]]:
        """Group violations by severity level."""
        grouped = {}
        for violation in self.violations:
            if violation.severity not in grouped:
                grouped[violation.severity] = []
            grouped[violation.severity].append(violation)
        return grouped


class EnhancedValidator:
    """Enhanced validator with rich output and GitHub integration."""
    
    def __init__(self, root_path: Optional[Path] = None):
        self.root_path = root_path or Path.cwd()
        self.result = ValidationResult()
        
    def validate_project(self) -> ValidationResult:
        """Run comprehensive project validation."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Structure validation
            task = progress.add_task("Validating project structure...", total=1)
            self._validate_structure()
            progress.update(task, advance=1)
            
            # Code quality validation
            task = progress.add_task("Validating code quality...", total=1)
            self._validate_code_quality()
            progress.update(task, advance=1)
            
            # Documentation validation
            task = progress.add_task("Validating documentation...", total=1)
            self._validate_documentation()
            progress.update(task, advance=1)
            
            # Security validation
            task = progress.add_task("Validating security...", total=1)
            self._validate_security()
            progress.update(task, advance=1)
        
        return self.result
    
    def _validate_structure(self):
        """Validate project structure."""
        # Check for forbidden directories
        forbidden_dirs = {'build', 'dist', 'output', 'tmp', 'temp'}
        for item in self.root_path.iterdir():
            if item.is_dir() and item.name in forbidden_dirs:
                self.result.add_violation(ValidationViolation(
                    message=f"Forbidden directory found: {item.name}/",
                    severity=ViolationSeverity.HIGH,
                    file_path=str(item.relative_to(self.root_path)),
                    rule_id="STRUCT_001",
                    fix_suggestion=f"Remove or move {item.name}/ to appropriate location (e.g., artifacts/, reports/)"
                ))
        
        # Check for missing essential directories
        essential_dirs = {'src', 'tests', 'docs'}
        for dir_name in essential_dirs:
            if not (self.root_path / dir_name).exists():
                self.result.add_violation(ValidationViolation(
                    message=f"Essential directory missing: {dir_name}/",
                    severity=ViolationSeverity.MEDIUM,
                    file_path=dir_name,
                    rule_id="STRUCT_002",
                    fix_suggestion=f"Create {dir_name}/ directory with proper structure"
                ))
    
    def _validate_code_quality(self):
        """Validate code quality."""
        # Check for TODO/FIXME comments
        for py_file in self.root_path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines, 1):
                    if 'TODO' in line or 'FIXME' in line:
                        self.result.add_violation(ValidationViolation(
                            message=f"TODO/FIXME comment found: {line.strip()}",
                            severity=ViolationSeverity.LOW,
                            file_path=str(py_file.relative_to(self.root_path)),
                            line_number=i,
                            rule_id="CODE_001",
                            fix_suggestion="Address TODO/FIXME comment or create GitHub issue"
                        ))
                    
                    if 'print(' in line and 'test' not in str(py_file):
                        self.result.add_violation(ValidationViolation(
                            message=f"Debug print statement found: {line.strip()}",
                            severity=ViolationSeverity.MEDIUM,
                            file_path=str(py_file.relative_to(self.root_path)),
                            line_number=i,
                            rule_id="CODE_002",
                            fix_suggestion="Replace print() with proper logging using structlog"
                        ))
            except Exception as e:
                self.result.add_violation(ValidationViolation(
                    message=f"Error reading file: {str(e)}",
                    severity=ViolationSeverity.HIGH,
                    file_path=str(py_file.relative_to(self.root_path)),
                    rule_id="CODE_003",
                    fix_suggestion="Fix file encoding or permissions"
                ))
    
    def _validate_documentation(self):
        """Validate documentation."""
        docs_path = self.root_path / "docs"
        if not docs_path.exists():
            self.result.add_violation(ValidationViolation(
                message="Documentation directory not found",
                severity=ViolationSeverity.MEDIUM,
                file_path="docs",
                rule_id="DOCS_001",
                fix_suggestion="Create docs/ directory with README.md"
            ))
            return
        
        # Check for README.md
        readme_path = self.root_path / "README.md"
        if not readme_path.exists():
            self.result.add_violation(ValidationViolation(
                message="Root README.md not found",
                severity=ViolationSeverity.HIGH,
                file_path="README.md",
                rule_id="DOCS_002",
                fix_suggestion="Create comprehensive README.md with project description and usage"
            ))
    
    def _validate_security(self):
        """Validate security aspects."""
        # Check for potential security issues
        for py_file in self.root_path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for dangerous functions
                dangerous_patterns = [
                    ('eval(', 'Use of eval() is dangerous'),
                    ('exec(', 'Use of exec() is dangerous'),
                    ('subprocess.call(', 'Use subprocess.run() with shell=False'),
                    ('os.system(', 'Use subprocess.run() instead of os.system()'),
                ]
                
                for pattern, message in dangerous_patterns:
                    if pattern in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if pattern in line:
                                self.result.add_violation(ValidationViolation(
                                    message=f"Security issue: {message}",
                                    severity=ViolationSeverity.CRITICAL,
                                    file_path=str(py_file.relative_to(self.root_path)),
                                    line_number=i,
                                    rule_id="SEC_001",
                                    fix_suggestion=f"Replace {pattern} with secure alternative"
                                ))
            except Exception:
                pass  # Skip files that can't be read
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped in validation."""
        skip_patterns = [
            '.venv', 'node_modules', '.git', '__pycache__', '.pytest_cache',
            '.mypy_cache', '.tox', 'htmlcov', 'build', 'dist'
        ]
        
        for pattern in skip_patterns:
            if pattern in str(file_path):
                return True
        return False


class RichOutputFormatter:
    """Formats validation output with rich styling."""
    
    def __init__(self, console: Console):
        self.console = console
    
    def display_results(self, result: ValidationResult):
        """Display validation results with rich formatting."""
        if result.passed:
            self.console.print(Panel(
                "✅ All validations passed!",
                title="Validation Results",
                border_style="green"
            ))
            return
        
        # Group violations by severity
        grouped = result.group_by_severity()
        
        # Create summary table
        table = Table(title="Validation Summary")
        table.add_column("Severity", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Status", justify="center")
        
        severity_colors = {
            ViolationSeverity.CRITICAL: "red",
            ViolationSeverity.HIGH: "red",
            ViolationSeverity.MEDIUM: "yellow",
            ViolationSeverity.LOW: "blue",
            ViolationSeverity.INFO: "green"
        }
        
        for severity in ViolationSeverity:
            count = len(grouped.get(severity, []))
            if count > 0:
                color = severity_colors[severity]
                status = "❌" if severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH] else "⚠️"
                table.add_row(
                    f"[{color}]{severity.value.upper()}[/{color}]",
                    f"[{color}]{count}[/{color}]",
                    status
                )
        
        self.console.print(table)
        
        # Display violations grouped by severity
        for severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH, ViolationSeverity.MEDIUM, ViolationSeverity.LOW]:
            violations = grouped.get(severity, [])
            if not violations:
                continue
            
            color = severity_colors[severity]
            self.console.print(f"\n[{color}]{'='*50}[/{color}]")
            self.console.print(f"[{color}]{severity.value.upper()} VIOLATIONS ({len(violations)})[/{color}]")
            self.console.print(f"[{color}]{'='*50}[/{color}]")
            
            for i, violation in enumerate(violations, 1):
                self._display_violation(violation, i, color)
    
    def _display_violation(self, violation: ValidationViolation, index: int, color: str):
        """Display a single violation with rich formatting."""
        # Create violation panel
        content = []
        
        # Add file and line info
        location = violation.file_path
        if violation.line_number:
            location += f":{violation.line_number}"
        content.append(f"📁 {location}")
        
        # Add rule ID if available
        if violation.rule_id:
            content.append(f"🔍 Rule: {violation.rule_id}")
        
        # Add message
        content.append(f"💬 {violation.message}")
        
        # Add fix suggestion if available
        if violation.fix_suggestion:
            content.append(f"🔧 Fix: {violation.fix_suggestion}")
        
        panel_content = "\n".join(content)
        
        self.console.print(Panel(
            panel_content,
            title=f"Violation {index}",
            border_style=color,
            padding=(1, 2)
        ))


class GitHubCommentGenerator:
    """Generates GitHub comments for PR validation results."""
    
    def __init__(self):
        self.max_violations = 10
    
    def generate_comment(self, result: ValidationResult) -> str:
        """Generate GitHub comment from validation results."""
        if result.passed:
            return self._generate_success_comment()
        
        return self._generate_failure_comment(result)
    
    def _generate_success_comment(self) -> str:
        """Generate comment for successful validation."""
        return """## ✅ Validation Passed

All validation checks have passed successfully! 🎉

**Summary:**
- ✅ Project structure validation
- ✅ Code quality validation  
- ✅ Documentation validation
- ✅ Security validation

Ready for merge! 🚀
"""
    
    def _generate_failure_comment(self, result: ValidationResult) -> str:
        """Generate comment for failed validation."""
        grouped = result.group_by_severity()
        
        # Build comment sections
        sections = []
        
        # Header
        sections.append("## ❌ Validation Failed")
        sections.append("")
        
        # Summary table
        sections.append("### 📊 Summary")
        sections.append("| Severity | Count | Status |")
        sections.append("|----------|-------|--------|")
        
        for severity in ViolationSeverity:
            count = len(grouped.get(severity, []))
            if count > 0:
                emoji = "❌" if severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH] else "⚠️"
                sections.append(f"| {severity.value.upper()} | {count} | {emoji} |")
        
        sections.append("")
        
        # Top violations
        all_violations = []
        for severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH, ViolationSeverity.MEDIUM, ViolationSeverity.LOW]:
            all_violations.extend(grouped.get(severity, []))
        
        if all_violations:
            sections.append(f"### 🔍 Top {min(self.max_violations, len(all_violations))} Violations")
            sections.append("")
            
            for i, violation in enumerate(all_violations[:self.max_violations], 1):
                sections.append(f"#### {i}. {violation.severity.value.upper()}: {violation.message}")
                sections.append(f"**File:** `{violation.file_path}`")
                if violation.line_number:
                    sections.append(f"**Line:** {violation.line_number}")
                if violation.rule_id:
                    sections.append(f"**Rule:** {violation.rule_id}")
                if violation.fix_suggestion:
                    sections.append(f"**How to fix:** {violation.fix_suggestion}")
                sections.append("")
        
        # Pre-commit reminder
        sections.append("### 🔧 Quick Fix")
        sections.append("To avoid validation failures in the future, install pre-commit hooks:")
        sections.append("```bash")
        sections.append("pip install pre-commit")
        sections.append("pre-commit install")
        sections.append("```")
        sections.append("")
        
        # Footer
        sections.append("Please fix these issues before merging. 🙏")
        
        return "\n".join(sections)
    
    def post_to_github(self, comment: str) -> bool:
        """Post comment to GitHub PR."""
        github_token = os.getenv("GITHUB_TOKEN")
        github_repository = os.getenv("GITHUB_REPOSITORY")
        pr_number = os.getenv("GITHUB_PR_NUMBER")
        
        if not all([github_token, github_repository, pr_number]):
            console.print("⚠️ GitHub environment variables not found. Skipping PR comment.")
            return False
        
        try:
            import requests
            
            url = f"https://api.github.com/repos/{github_repository}/issues/{pr_number}/comments"
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            
            data = {"body": comment}
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 201:
                console.print("✅ GitHub comment posted successfully!")
                return True
            else:
                console.print(f"❌ Failed to post GitHub comment: {response.status_code}")
                return False
                
        except ImportError:
            console.print("⚠️ requests library not available. Install with: pip install requests")
            return False
        except Exception as e:
            console.print(f"❌ Error posting GitHub comment: {str(e)}")
            return False


def check_pre_commit_installed() -> bool:
    """Check if pre-commit is installed."""
    try:
        result = subprocess.run(
            ["pre-commit", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def show_pre_commit_reminder():
    """Show reminder to install pre-commit hooks."""
    console.print(Panel(
        "[yellow]⚠️ Pre-commit hooks not installed![/yellow]\n\n"
        "To prevent validation failures in the future, install pre-commit hooks:\n\n"
        "[cyan]pip install pre-commit[/cyan]\n"
        "[cyan]pre-commit install[/cyan]\n\n"
        "This will run validations automatically before each commit.",
        title="Pre-commit Reminder",
        border_style="yellow"
    ))


@app.command("run")
def run_validation(
    path: Optional[Path] = typer.Argument(None, help="Path to validate (default: current directory)"),
    github_comment: bool = typer.Option(False, "--github-comment", help="Post results to GitHub PR"),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop on first critical violation"),
    output_format: str = typer.Option("rich", "--format", help="Output format: rich, json, markdown"),
    save_report: Optional[Path] = typer.Option(None, "--save", help="Save report to file"),
) -> None:
    """🔍 Run comprehensive validation with rich output."""
    
    target_path = path or Path.cwd()
    
    # Initialize validator
    validator = EnhancedValidator(target_path)
    formatter = RichOutputFormatter(console)
    
    # Run validation
    with console.status("Running validation..."):
        result = validator.validate_project()
    
    # Display results
    if output_format == "rich":
        formatter.display_results(result)
    elif output_format == "json":
        report_data = {
            "passed": result.passed,
            "violations": [
                {
                    "message": v.message,
                    "severity": v.severity.value,
                    "file_path": v.file_path,
                    "line_number": v.line_number,
                    "rule_id": v.rule_id,
                    "fix_suggestion": v.fix_suggestion,
                }
                for v in result.violations
            ]
        }
        console.print(json.dumps(report_data, indent=2))
    
    # Save report if requested
    if save_report:
        report_content = json.dumps({
            "passed": result.passed,
            "violations": [
                {
                    "message": v.message,
                    "severity": v.severity.value,
                    "file_path": v.file_path,
                    "line_number": v.line_number,
                    "rule_id": v.rule_id,
                    "fix_suggestion": v.fix_suggestion,
                }
                for v in result.violations
            ]
        }, indent=2)
        
        save_report.write_text(report_content)
        console.print(f"📄 Report saved to: {save_report}")
    
    # Post to GitHub if requested and in CI
    if github_comment or os.getenv("CI"):
        comment_generator = GitHubCommentGenerator()
        comment = comment_generator.generate_comment(result)
        
        if github_comment:
            comment_generator.post_to_github(comment)
        else:
            console.print("\n[dim]GitHub comment preview:[/dim]")
            console.print(Panel(comment, title="GitHub Comment", border_style="blue"))
    
    # Show pre-commit reminder for local failures
    if not result.passed and not os.getenv("CI"):
        if not check_pre_commit_installed():
            show_pre_commit_reminder()
    
    # Exit with appropriate code
    if not result.passed:
        raise typer.Exit(1)


@app.command("check-pre-commit")
def check_pre_commit() -> None:
    """🔧 Check if pre-commit hooks are installed."""
    if check_pre_commit_installed():
        console.print("✅ Pre-commit hooks are installed!")
        
        # Check if hooks are actually installed in git
        try:
            hooks_path = Path(".git/hooks/pre-commit")
            if hooks_path.exists():
                console.print("✅ Pre-commit hooks are active in git!")
            else:
                console.print("⚠️ Pre-commit is installed but hooks are not active.")
                console.print("Run: [cyan]pre-commit install[/cyan]")
        except Exception:
            console.print("⚠️ Could not check git hooks status.")
    else:
        show_pre_commit_reminder()
        raise typer.Exit(1)


@app.command("install-hooks")
def install_hooks() -> None:
    """🔧 Install pre-commit hooks."""
    try:
        # Install pre-commit if not available
        if not check_pre_commit_installed():
            console.print("📦 Installing pre-commit...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], check=True)
        
        # Install hooks
        console.print("🔧 Installing pre-commit hooks...")
        subprocess.run(["pre-commit", "install"], check=True)
        
        console.print("✅ Pre-commit hooks installed successfully!")
        console.print("Now validation will run automatically before each commit.")
        
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Failed to install pre-commit hooks: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
