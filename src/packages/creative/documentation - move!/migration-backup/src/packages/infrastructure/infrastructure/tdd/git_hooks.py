"""Git hooks and integration for TDD enforcement."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from monorepo.infrastructure.config.tdd_config import get_tdd_config
from monorepo.infrastructure.persistence.tdd_repository import FileTDDRepository
from monorepo.infrastructure.tdd.enforcement import TDDEnforcementEngine


class GitHookManager:
    """Manager for Git hooks integration with TDD enforcement."""

    def __init__(self, repo_root: Path | None = None):
        self.repo_root = repo_root or self._find_git_root()
        self.hooks_dir = self.repo_root / ".git" / "hooks"
        self.config_manager = get_tdd_config()

    def _find_git_root(self) -> Path:
        """Find the root of the git repository."""
        current = Path.cwd()
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
        raise RuntimeError("Not in a git repository")

    def install_hooks(self) -> None:
        """Install TDD git hooks."""
        if not self.config_manager.settings.git_hooks_enabled:
            return

        hooks_to_install = []

        if self.config_manager.settings.pre_commit_validation:
            hooks_to_install.append("pre-commit")

        if self.config_manager.settings.pre_push_validation:
            hooks_to_install.append("pre-push")

        for hook_name in hooks_to_install:
            self._install_hook(hook_name)

    def _install_hook(self, hook_name: str) -> None:
        """Install a specific git hook."""
        hook_file = self.hooks_dir / hook_name

        # Create hooks directory if it doesn't exist
        self.hooks_dir.mkdir(exist_ok=True)

        # Create hook script
        hook_content = self._generate_hook_script(hook_name)

        with open(hook_file, "w") as f:
            f.write(hook_content)

        # Make executable
        hook_file.chmod(0o755)

    def _generate_hook_script(self, hook_name: str) -> str:
        """Generate the content for a git hook script."""
        python_path = sys.executable

        if hook_name == "pre-commit":
            return f"""#!/bin/bash
# TDD Pre-commit Hook - Validate TDD compliance before commit

echo "🧪 Running TDD validation..."

# Run TDD validation
{python_path} -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

try:
    from monorepo.infrastructure.tdd.git_hooks import run_pre_commit_validation
    exit_code = run_pre_commit_validation()
    sys.exit(exit_code)
except Exception as e:
    print(f'TDD validation failed: {{e}}')
    sys.exit(1)
"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "❌ TDD validation failed. Commit rejected."
    echo "Run 'pynomaly tdd validate --fix' to resolve issues."
    exit 1
fi

echo "✅ TDD validation passed."
exit 0
"""

        elif hook_name == "pre-push":
            return f"""#!/bin/bash
# TDD Pre-push Hook - Validate TDD compliance before push

echo "🧪 Running comprehensive TDD validation..."

# Run comprehensive TDD validation including coverage
{python_path} -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

try:
    from monorepo.infrastructure.tdd.git_hooks import run_pre_push_validation
    exit_code = run_pre_push_validation()
    sys.exit(exit_code)
except Exception as e:
    print(f'TDD validation failed: {{e}}')
    sys.exit(1)
"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "❌ TDD validation failed. Push rejected."
    echo "Run 'pynomaly tdd validate --coverage --fix' to resolve issues."
    exit 1
fi

echo "✅ TDD validation passed."
exit 0
"""

        return ""

    def uninstall_hooks(self) -> None:
        """Uninstall TDD git hooks."""
        hook_files = ["pre-commit", "pre-push"]

        for hook_name in hook_files:
            hook_file = self.hooks_dir / hook_name
            if hook_file.exists():
                # Check if it's our TDD hook
                try:
                    with open(hook_file) as f:
                        content = f.read()

                    if "TDD" in content and "monorepo" in content:
                        hook_file.unlink()
                        print(f"Removed TDD {hook_name} hook")

                except Exception:
                    # Skip if we can't read the file
                    pass

    def is_hook_installed(self, hook_name: str) -> bool:
        """Check if a TDD hook is installed."""
        hook_file = self.hooks_dir / hook_name

        if not hook_file.exists():
            return False

        try:
            with open(hook_file) as f:
                content = f.read()
            return "TDD" in content and "monorepo" in content
        except Exception:
            return False


def run_pre_commit_validation() -> int:
    """Run TDD validation for pre-commit hook.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Get staged files
        staged_files = get_staged_files()

        if not staged_files:
            return 0  # No files to validate

        # Initialize TDD engine
        config_manager = get_tdd_config()
        if not config_manager.settings.enabled:
            return 0  # TDD not enabled

        repo_root = Path.cwd()
        repository = FileTDDRepository(repo_root / "tdd_storage")
        engine = TDDEnforcementEngine(config_manager, repository)

        # Validate only staged Python files
        python_files = [f for f in staged_files if f.endswith(".py")]
        violations = []

        for file_path in python_files:
            file_violations = engine.validate_file(Path(file_path))
            violations.extend(file_violations)

        # Filter out non-critical violations for pre-commit
        critical_violations = [
            v
            for v in violations
            if v.severity in ["error"]
            and v.violation_type
            in ["missing_test", "implementation_before_test", "parse_error"]
        ]

        if critical_violations:
            print(f"\n❌ Found {len(critical_violations)} critical TDD violations:")
            for violation in critical_violations[:5]:
                print(f"  • {violation.description}")

            if len(critical_violations) > 5:
                print(f"  ... and {len(critical_violations) - 5} more")

            return 1

        # Show warnings but don't block commit
        warning_violations = [v for v in violations if v.severity == "warning"]
        if warning_violations:
            print(
                f"\n⚠️  Found {len(warning_violations)} TDD warnings (not blocking commit):"
            )
            for violation in warning_violations[:3]:
                print(f"  • {violation.description}")

        return 0

    except Exception as e:
        print(f"TDD pre-commit validation failed: {str(e)}")
        return 1


def run_pre_push_validation() -> int:
    """Run comprehensive TDD validation for pre-push hook.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Initialize TDD engine
        config_manager = get_tdd_config()
        if not config_manager.settings.enabled:
            return 0  # TDD not enabled

        repo_root = Path.cwd()
        repository = FileTDDRepository(repo_root / "tdd_storage")
        engine = TDDEnforcementEngine(config_manager, repository)

        # Run comprehensive validation
        violations = engine.validate_project(repo_root)

        # Run coverage analysis
        coverage_data = engine.run_coverage_analysis(repo_root)
        coverage_violations = engine.validate_coverage_thresholds(coverage_data)
        violations.extend(coverage_violations)

        # Check for critical violations
        critical_violations = [
            v
            for v in violations
            if v.severity == "error"
            or (
                v.severity == "warning"
                and v.violation_type in ["low_coverage", "missing_test"]
            )
        ]

        if critical_violations:
            print(
                f"\n❌ Found {len(critical_violations)} TDD violations blocking push:"
            )

            # Group by type
            violation_groups = {}
            for violation in critical_violations:
                if violation.violation_type not in violation_groups:
                    violation_groups[violation.violation_type] = []
                violation_groups[violation.violation_type].append(violation)

            for violation_type, group_violations in violation_groups.items():
                print(
                    f"\n{violation_type.replace('_', ' ').title()} ({len(group_violations)}):"
                )
                for violation in group_violations[:3]:
                    print(f"  • {violation.description}")
                if len(group_violations) > 3:
                    print(f"  ... and {len(group_violations) - 3} more")

            # Show coverage summary if relevant
            if coverage_data:
                avg_coverage = sum(coverage_data.values()) / len(coverage_data)
                threshold = engine.settings.min_test_coverage
                if avg_coverage < threshold:
                    print(
                        f"\n📊 Average coverage: {avg_coverage:.1%} (required: {threshold:.1%})"
                    )

            return 1

        # Show compliance summary
        compliance_report = engine.get_compliance_report()
        report = compliance_report["compliance_report"]

        print(f"✅ TDD Compliance: {report.overall_compliance:.1%}")
        if coverage_data:
            avg_coverage = sum(coverage_data.values()) / len(coverage_data)
            print(f"📊 Test Coverage: {avg_coverage:.1%}")

        return 0

    except Exception as e:
        print(f"TDD pre-push validation failed: {str(e)}")
        return 1


def get_staged_files() -> list[str]:
    """Get list of staged files from git.

    Returns:
        List of staged file paths
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    except subprocess.CalledProcessError:
        return []


def get_changed_files(base_branch: str = "main") -> list[str]:
    """Get list of files changed compared to base branch.

    Args:
        base_branch: Base branch to compare against

    Returns:
        List of changed file paths
    """
    try:
        result = subprocess.run(
            ["git", "diff", f"{base_branch}...HEAD", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    except subprocess.CalledProcessError:
        return []


class PreCommitConfig:
    """Manager for pre-commit configuration integration."""

    def __init__(self, repo_root: Path | None = None):
        self.repo_root = repo_root or Path.cwd()
        self.pre_commit_config = self.repo_root / ".pre-commit-config.yaml"

    def add_tdd_hook(self) -> None:
        """Add TDD hook to pre-commit configuration."""
        import yaml

        # TDD hook configuration
        tdd_hook = {
            "repo": "local",
            "hooks": [
                {
                    "id": "tdd-validation",
                    "name": "TDD Validation",
                    "entry": "python -m monorepo.infrastructure.tdd.git_hooks",
                    "language": "system",
                    "files": r"\.py$",
                    "stages": ["commit"],
                }
            ],
        }

        # Load existing config or create new one
        if self.pre_commit_config.exists():
            with open(self.pre_commit_config) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {"repos": []}

        # Add TDD hook if not already present
        repos = config.get("repos", [])
        tdd_repo_exists = any(
            repo.get("repo") == "local"
            and any(
                hook.get("id") == "tdd-validation" for hook in repo.get("hooks", [])
            )
            for repo in repos
        )

        if not tdd_repo_exists:
            repos.append(tdd_hook)
            config["repos"] = repos

            with open(self.pre_commit_config, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

    def remove_tdd_hook(self) -> None:
        """Remove TDD hook from pre-commit configuration."""
        import yaml

        if not self.pre_commit_config.exists():
            return

        with open(self.pre_commit_config) as f:
            config = yaml.safe_load(f) or {}

        repos = config.get("repos", [])

        # Remove TDD hooks
        updated_repos = []
        for repo in repos:
            if repo.get("repo") == "local":
                # Filter out TDD hooks
                hooks = [
                    hook
                    for hook in repo.get("hooks", [])
                    if hook.get("id") != "tdd-validation"
                ]
                if hooks:
                    repo["hooks"] = hooks
                    updated_repos.append(repo)
            else:
                updated_repos.append(repo)

        config["repos"] = updated_repos

        with open(self.pre_commit_config, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


def main():
    """Main entry point for git hook execution."""
    import sys

    if len(sys.argv) > 1:
        hook_type = sys.argv[1]

        if hook_type == "pre-commit":
            sys.exit(run_pre_commit_validation())
        elif hook_type == "pre-push":
            sys.exit(run_pre_push_validation())

    # Default to pre-commit validation
    sys.exit(run_pre_commit_validation())


if __name__ == "__main__":
    main()
