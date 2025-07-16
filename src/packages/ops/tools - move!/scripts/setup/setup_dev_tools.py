#!/usr/bin/env python3
"""
Developer Tools Setup Automation

This script automates the setup of development tools for Pynomaly,
including IDE configuration, git hooks, and development utilities.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


class DevToolsSetup:
    """Development tools setup automation."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.system = platform.system().lower()
        self.home_dir = Path.home()
        
    def _run_command(self, cmd: List[str], description: str,
                    cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command with nice output."""
        print(f"{Colors.BLUE}→{Colors.END} {description}")
        print(f"  {Colors.CYAN}${Colors.END} {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                check=check,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n')[:5]:  # Limit output
                    print(f"  {line}")
                if len(result.stdout.strip().split('\n')) > 5:
                    print(f"  ... (output truncated)")
            
            print(f"{Colors.GREEN}✓{Colors.END} {description} completed\n")
            return result
            
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}✗{Colors.END} {description} failed")
            if e.stderr:
                print(f"Error: {e.stderr}")
            if not check:
                return e
            raise
    
    def setup_vscode(self) -> None:
        """Setup VS Code configuration."""
        print(f"{Colors.BOLD}{Colors.WHITE}🔧 Setting Up VS Code{Colors.END}\n")
        
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        # Settings configuration
        settings = {
            "python.defaultInterpreterPath": "./environments/.venv/bin/python",
            "python.terminal.activateEnvironment": True,
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["tests/"],
            "python.testing.cwd": "${workspaceFolder}",
            "python.linting.enabled": True,
            "python.linting.ruffEnabled": True,
            "python.formatting.provider": "ruff",
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": True,
                "source.fixAll": True
            },
            "files.exclude": {
                "**/__pycache__": True,
                "**/.mypy_cache": True,
                "**/.ruff_cache": True,
                "**/.pytest_cache": True,
                "**/node_modules": True,
                "**/dist": True,
                "**/build": True,
                "**/.coverage": True,
                "**/htmlcov": True
            },
            "search.exclude": {
                "**/node_modules": True,
                "**/dist": True,
                "**/build": True,
                "**/.venv": True,
                "**/environments": True
            },
            "files.watcherExclude": {
                "**/.git/objects/**": True,
                "**/.git/subtree-cache/**": True,
                "**/node_modules/**": True,
                "**/environments/**": True,
                "**/dist/**": True,
                "**/build/**": True
            }
        }
        
        settings_file = vscode_dir / "settings.json"
        settings_file.write_text(json.dumps(settings, indent=2))
        print(f"{Colors.GREEN}✓{Colors.END} Created VS Code settings")
        
        # Extensions recommendations
        extensions = {
            "recommendations": [
                "ms-python.python",
                "ms-python.ruff",
                "ms-python.mypy-type-checker",
                "ms-python.debugpy",
                "ms-toolsai.jupyter",
                "charliermarsh.ruff",
                "ms-vscode.vscode-json",
                "redhat.vscode-yaml",
                "ms-vscode.vscode-markdown",
                "davidanson.vscode-markdownlint",
                "ms-vscode.vscode-github-issue-notebooks",
                "github.vscode-github-actions",
                "ms-vscode.test-adapter-converter",
                "littlefoxteam.vscode-python-test-adapter",
                "ms-vscode.vscode-docker"
            ]
        }
        
        extensions_file = vscode_dir / "extensions.json"
        extensions_file.write_text(json.dumps(extensions, indent=2))
        print(f"{Colors.GREEN}✓{Colors.END} Created VS Code extensions recommendations")
        
        # Launch configuration for debugging
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python: Current File",
                    "type": "python",
                    "request": "launch",
                    "program": "${file}",
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                },
                {
                    "name": "Python: Test File",
                    "type": "python",
                    "request": "launch",
                    "module": "pytest",
                    "args": ["${file}", "-v"],
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                },
                {
                    "name": "Pynomaly CLI",
                    "type": "python",
                    "request": "launch",
                    "module": "monorepo.presentation.cli.app",
                    "args": ["--help"],
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                },
                {
                    "name": "FastAPI Server",
                    "type": "python",
                    "request": "launch",
                    "module": "uvicorn",
                    "args": [
                        "monorepo.presentation.api.app:app",
                        "--reload",
                        "--port",
                        "8000"
                    ],
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                }
            ]
        }
        
        launch_file = vscode_dir / "launch.json"
        launch_file.write_text(json.dumps(launch_config, indent=2))
        print(f"{Colors.GREEN}✓{Colors.END} Created VS Code launch configuration")
        
        # Tasks configuration
        tasks_config = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Run Tests",
                    "type": "shell",
                    "command": "pytest",
                    "args": ["tests/", "-v"],
                    "group": "test",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    },
                    "options": {
                        "env": {
                            "PYTHONPATH": "${workspaceFolder}/src"
                        }
                    }
                },
                {
                    "label": "Run Linting",
                    "type": "shell",
                    "command": "ruff",
                    "args": ["check", "src/", "tests/"],
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                },
                {
                    "label": "Format Code",
                    "type": "shell",
                    "command": "ruff",
                    "args": ["format", "src/", "tests/"],
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                },
                {
                    "label": "Type Check",
                    "type": "shell",
                    "command": "mypy",
                    "args": ["src/pynomaly/"],
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                }
            ]
        }
        
        tasks_file = vscode_dir / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_config, indent=2))
        print(f"{Colors.GREEN}✓{Colors.END} Created VS Code tasks configuration")
    
    def setup_git_hooks(self) -> None:
        """Setup Git hooks for development."""
        print(f"{Colors.BOLD}{Colors.WHITE}🪝 Setting Up Git Hooks{Colors.END}\n")
        
        # Check if pre-commit is available
        try:
            self._run_command(["pre-commit", "--version"], "Checking pre-commit installation")
            
            # Install pre-commit hooks
            self._run_command(["pre-commit", "install"], "Installing pre-commit hooks")
            self._run_command(["pre-commit", "install", "--hook-type", "commit-msg"], 
                            "Installing commit-msg hook")
            
        except subprocess.CalledProcessError:
            print(f"{Colors.YELLOW}⚠{Colors.END} pre-commit not available, creating manual hooks")
            self._create_manual_git_hooks()
    
    def _create_manual_git_hooks(self) -> None:
        """Create manual git hooks if pre-commit is not available."""
        hooks_dir = self.project_root / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        
        # Pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
# Pynomaly pre-commit hook

set -e

echo "Running pre-commit checks..."

# Check for Python syntax errors
python -m py_compile $(find src/ tests/ -name "*.py")

# Run basic linting
if command -v ruff &> /dev/null; then
    echo "Running ruff check..."
    ruff check src/ tests/
else
    echo "Ruff not available, skipping lint check"
fi

# Run type checking
if command -v mypy &> /dev/null; then
    echo "Running mypy check..."
    mypy src/pynomaly/
else
    echo "MyPy not available, skipping type check"
fi

# Run tests on changed files
if command -v pytest &> /dev/null; then
    echo "Running relevant tests..."
    # Get changed Python files
    changed_files=$(git diff --cached --name-only --diff-filter=ACM | grep -E "\\.py$" || true)
    if [ ! -z "$changed_files" ]; then
        # Run tests for changed modules
        PYTHONPATH=src pytest tests/unit/ -x --tb=short
    fi
else
    echo "Pytest not available, skipping tests"
fi

echo "Pre-commit checks passed!"
"""
        
        pre_commit_hook.write_text(pre_commit_content)
        pre_commit_hook.chmod(0o755)
        print(f"{Colors.GREEN}✓{Colors.END} Created manual pre-commit hook")
        
        # Commit message hook
        commit_msg_hook = hooks_dir / "commit-msg"
        commit_msg_content = """#!/bin/bash
# Pynomaly commit message hook

commit_file=$1
commit_message=$(cat $commit_file)

# Check commit message format (conventional commits)
if ! echo "$commit_message" | grep -qE "^(feat|fix|docs|style|refactor|test|chore|perf|ci|build)(\\(.+\\))?: .{1,50}"; then
    echo "Error: Commit message does not follow conventional format!"
    echo "Format: type(scope): description"
    echo "Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build"
    echo "Example: feat(detector): add new isolation forest algorithm"
    exit 1
fi

echo "Commit message format OK"
"""
        
        commit_msg_hook.write_text(commit_msg_content)
        commit_msg_hook.chmod(0o755)
        print(f"{Colors.GREEN}✓{Colors.END} Created manual commit-msg hook")
    
    def setup_shell_aliases(self) -> None:
        """Setup useful shell aliases for development."""
        print(f"{Colors.BOLD}{Colors.WHITE}💻 Setting Up Shell Aliases{Colors.END}\n")
        
        aliases = {
            # Development commands
            "pyn-dev": "cd " + str(self.project_root) + " && source environments/.venv/bin/activate",
            "pyn-test": "PYTHONPATH=src pytest tests/",
            "pyn-test-unit": "PYTHONPATH=src pytest tests/unit/ -v",
            "pyn-test-integration": "PYTHONPATH=src pytest tests/integration/ -v",
            "pyn-test-coverage": "PYTHONPATH=src pytest tests/ --cov=src/pynomaly --cov-report=html",
            "pyn-lint": "ruff check src/ tests/",
            "pyn-format": "ruff format src/ tests/",
            "pyn-type": "mypy src/pynomaly/",
            "pyn-quality": "python scripts/quality_gates.py",
            "pyn-docs": "mkdocs serve",
            "pyn-api": "PYTHONPATH=src uvicorn monorepo.presentation.api.app:app --reload",
            "pyn-cli": "PYTHONPATH=src python -m monorepo.presentation.cli.app",
            "pyn-clean": "find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true",
        }
        
        # Detect shell and create appropriate alias file
        shell_config_file = None
        if os.environ.get('SHELL', '').endswith('bash'):
            shell_config_file = self.home_dir / ".bashrc"
        elif os.environ.get('SHELL', '').endswith('zsh'):
            shell_config_file = self.home_dir / ".zshrc"
        
        if shell_config_file and shell_config_file.exists():
            # Create alias section
            alias_section = "\n# Pynomaly Development Aliases\n"
            for alias, command in aliases.items():
                alias_section += f"alias {alias}='{command}'\n"
            
            # Check if aliases already exist
            current_content = shell_config_file.read_text()
            if "# Pynomaly Development Aliases" not in current_content:
                shell_config_file.write_text(current_content + alias_section)
                print(f"{Colors.GREEN}✓{Colors.END} Added aliases to {shell_config_file}")
                print(f"{Colors.YELLOW}ℹ{Colors.END} Run 'source {shell_config_file}' to activate aliases")
            else:
                print(f"{Colors.YELLOW}ℹ{Colors.END} Aliases already exist in {shell_config_file}")
        else:
            # Create standalone alias file
            alias_file = self.project_root / "scripts" / "dev_aliases.sh"
            alias_content = "#!/bin/bash\n# Pynomaly Development Aliases\n\n"
            for alias, command in aliases.items():
                alias_content += f"alias {alias}='{command}'\n"
            
            alias_content += "\necho 'Pynomaly development aliases loaded!'\n"
            alias_content += "echo 'Available commands: " + " ".join(aliases.keys()) + "'\n"
            
            alias_file.write_text(alias_content)
            alias_file.chmod(0o755)
            print(f"{Colors.GREEN}✓{Colors.END} Created alias file: {alias_file}")
            print(f"{Colors.YELLOW}ℹ{Colors.END} Run 'source {alias_file}' to load aliases")
    
    def setup_development_config(self) -> None:
        """Setup development configuration files."""
        print(f"{Colors.BOLD}{Colors.WHITE}⚙️ Setting Up Development Config{Colors.END}\n")
        
        # Create .env file for development
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_content = f"""# Pynomaly Development Environment Configuration
# Generated by setup_dev_tools.py

# Environment
PYNOMALY_ENVIRONMENT=development
PYNOMALY_LOG_LEVEL=DEBUG
PYNOMALY_DEBUG=true

# Python Path
PYTHONPATH={self.project_root}/src

# Database
PYNOMALY_DATABASE_URL=sqlite:///storage/dev.db

# API Configuration
PYNOMALY_API_HOST=localhost
PYNOMALY_API_PORT=8000
PYNOMALY_API_RELOAD=true

# Cache Configuration
PYNOMALY_CACHE_ENABLED=true
PYNOMALY_CACHE_TTL=300

# Testing Configuration
TESTING=false
PYTEST_CURRENT_TEST=""

# Performance
PYNOMALY_ENABLE_PERFORMANCE_MONITORING=true
PYNOMALY_PROFILE_ENABLED=false
"""
            env_file.write_text(env_content)
            print(f"{Colors.GREEN}✓{Colors.END} Created .env file")
        else:
            print(f"{Colors.YELLOW}ℹ{Colors.END} .env file already exists")
        
        # Create development-specific gitignore additions
        dev_gitignore = self.project_root / ".gitignore.dev"
        if not dev_gitignore.exists():
            gitignore_content = """# Development-specific ignores
# Add to main .gitignore if needed

# IDE files
.vscode/settings.json.bak
.idea/
*.swp
*.swo
*~

# Development databases
*.db
*.sqlite
*.sqlite3

# Development logs
dev.log
debug.log
*.log

# Temporary development files
tmp/
temp/
.tmp/

# Development environment
.env.local
.env.development

# Development artifacts
profiles/
benchmarks/
performance_reports/

# Development notebooks
notebooks/scratch/
*.ipynb_checkpoints

# OS specific
.DS_Store
Thumbs.db
"""
            dev_gitignore.write_text(gitignore_content)
            print(f"{Colors.GREEN}✓{Colors.END} Created development gitignore")
    
    def setup_debugging_tools(self) -> None:
        """Setup debugging and profiling tools."""
        print(f"{Colors.BOLD}{Colors.WHITE}🐛 Setting Up Debugging Tools{Colors.END}\n")
        
        # Create debugging utility script
        debug_script = self.project_root / "scripts" / "debug_utils.py"
        debug_content = '''#!/usr/bin/env python3
"""
Debugging utilities for Pynomaly development.

Usage:
    python scripts/debug_utils.py profile function_name
    python scripts/debug_utils.py memory function_name
    python scripts/debug_utils.py trace module_name
"""

import argparse
import cProfile
import pstats
import tracemalloc
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def profile_function(module_name: str, function_name: str):
    """Profile a specific function."""
    print(f"Profiling {module_name}.{function_name}")
    
    # Import the module and function
    module = __import__(module_name, fromlist=[function_name])
    func = getattr(module, function_name)
    
    # Profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Call function (you may need to modify this based on your needs)
    try:
        result = func()
        print(f"Function result: {result}")
    except Exception as e:
        print(f"Function failed: {e}")
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)


def trace_memory():
    """Start memory tracing."""
    tracemalloc.start()
    print("Memory tracing started. Run your code, then call print_memory_stats()")


def print_memory_stats():
    """Print current memory statistics."""
    if not tracemalloc.is_tracing():
        print("Memory tracing not started")
        return
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    # Top memory consumers
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("Top 10 memory consumers:")
    for stat in top_stats[:10]:
        print(f"  {stat}")


def main():
    parser = argparse.ArgumentParser(description="Debugging utilities")
    parser.add_argument("command", choices=["profile", "memory", "trace"])
    parser.add_argument("target", help="Target module or function")
    parser.add_argument("--function", help="Function name for profiling")
    
    args = parser.parse_args()
    
    if args.command == "profile":
        profile_function(args.target, args.function or "main")
    elif args.command == "memory":
        trace_memory()
    elif args.command == "trace":
        print(f"Tracing module: {args.target}")
        # Add tracing logic here


if __name__ == "__main__":
    main()
'''
        
        debug_script.write_text(debug_content)
        debug_script.chmod(0o755)
        print(f"{Colors.GREEN}✓{Colors.END} Created debugging utilities")
        
        # Create performance testing script
        perf_script = self.project_root / "scripts" / "performance_monitor.py"
        perf_content = '''#!/usr/bin/env python3
"""
Performance monitoring utilities for Pynomaly.

Usage:
    python scripts/performance_monitor.py benchmark
    python scripts/performance_monitor.py monitor
"""

import argparse
import time
import psutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def monitor_system():
    """Monitor system resources during development."""
    print("Monitoring system resources (Ctrl+C to stop)")
    print("Time\\t\\tCPU%\\tMemory%\\tDisk%")
    print("-" * 50)
    
    try:
        while True:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            current_time = time.strftime("%H:%M:%S")
            print(f"{current_time}\\t{cpu_percent:.1f}\\t{memory_percent:.1f}\\t{disk_percent:.1f}")
            
            time.sleep(5)
    except KeyboardInterrupt:
        print("\\nMonitoring stopped")


def run_benchmarks():
    """Run performance benchmarks."""
    print("Running Pynomaly performance benchmarks...")
    
    try:
        import pytest
        import subprocess
        
        # Run benchmark tests
        result = subprocess.run([
            "pytest", "tests/performance/", 
            "--benchmark-only", 
            "--benchmark-sort=mean",
            "-v"
        ], cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("Benchmarks completed successfully")
        else:
            print("Benchmarks failed or not found")
            
    except ImportError:
        print("pytest not available for running benchmarks")


def main():
    parser = argparse.ArgumentParser(description="Performance monitoring")
    parser.add_argument("command", choices=["benchmark", "monitor"])
    
    args = parser.parse_args()
    
    if args.command == "benchmark":
        run_benchmarks()
    elif args.command == "monitor":
        monitor_system()


if __name__ == "__main__":
    main()
'''
        
        perf_script.write_text(perf_content)
        perf_script.chmod(0o755)
        print(f"{Colors.GREEN}✓{Colors.END} Created performance monitoring utilities")
    
    def print_summary(self) -> None:
        """Print setup summary and next steps."""
        print(f"{Colors.BOLD}{Colors.GREEN}🎉 Developer Tools Setup Complete!{Colors.END}\n")
        
        print(f"{Colors.BOLD}📋 What was configured:{Colors.END}")
        print(f"  ✓ VS Code settings and extensions")
        print(f"  ✓ Git hooks for code quality")
        print(f"  ✓ Shell aliases for common tasks")
        print(f"  ✓ Development environment configuration")
        print(f"  ✓ Debugging and performance tools")
        print()
        
        print(f"{Colors.BOLD}🚀 Next steps:{Colors.END}")
        print(f"  1. Restart VS Code to apply settings")
        print(f"  2. Install recommended extensions when prompted")
        print(f"  3. Source your shell config or run:")
        print(f"     {Colors.CYAN}source scripts/dev_aliases.sh{Colors.END}")
        print()
        
        print(f"{Colors.BOLD}🛠️ Available commands:{Colors.END}")
        print(f"  • {Colors.CYAN}pyn-dev{Colors.END} - Navigate to project and activate venv")
        print(f"  • {Colors.CYAN}pyn-test{Colors.END} - Run all tests")
        print(f"  • {Colors.CYAN}pyn-lint{Colors.END} - Run linting")
        print(f"  • {Colors.CYAN}pyn-format{Colors.END} - Format code")
        print(f"  • {Colors.CYAN}pyn-api{Colors.END} - Start API server")
        print(f"  • {Colors.CYAN}pyn-cli{Colors.END} - Run CLI interface")
        print()
        
        print(f"{Colors.BOLD}🐛 Debugging tools:{Colors.END}")
        print(f"  • {Colors.CYAN}python scripts/debug_utils.py{Colors.END} - Profiling utilities")
        print(f"  • {Colors.CYAN}python scripts/performance_monitor.py{Colors.END} - Performance monitoring")
        print()
        
        print(f"{Colors.GREEN}Happy developing! 🚀{Colors.END}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup development tools for Pynomaly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup/setup_dev_tools.py                    # Full setup
  python scripts/setup/setup_dev_tools.py --vscode-only     # VS Code only
  python scripts/setup/setup_dev_tools.py --git-only        # Git hooks only
        """
    )
    
    parser.add_argument(
        "--vscode-only",
        action="store_true",
        help="Setup VS Code configuration only"
    )
    
    parser.add_argument(
        "--git-only",
        action="store_true",
        help="Setup Git hooks only"
    )
    
    parser.add_argument(
        "--no-aliases",
        action="store_true",
        help="Skip shell aliases setup"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    try:
        setup = DevToolsSetup(args.project_root)
        
        if args.vscode_only:
            setup.setup_vscode()
        elif args.git_only:
            setup.setup_git_hooks()
        else:
            # Full setup
            setup.setup_vscode()
            setup.setup_git_hooks()
            
            if not args.no_aliases:
                setup.setup_shell_aliases()
            
            setup.setup_development_config()
            setup.setup_debugging_tools()
        
        setup.print_summary()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠ Setup interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}❌ Setup failed: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()