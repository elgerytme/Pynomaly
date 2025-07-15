#!/usr/bin/env python3
"""Automated test environment provisioning and management."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml


class TestEnvironmentManager:
    """Manage automated test environment provisioning."""

    def __init__(self, base_dir: str = "environments"):
        """Initialize test environment manager.

        Args:
            base_dir: Base directory for test environments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.active_environments = {}

    def create_environment(
        self,
        name: str,
        python_version: str = "3.11",
        packages: Optional[List[str]] = None,
        requirements_file: Optional[str] = None,
        clean: bool = False
    ) -> Dict[str, Any]:
        """Create a new test environment.

        Args:
            name: Environment name
            python_version: Python version to use
            packages: List of packages to install
            requirements_file: Path to requirements file
            clean: Whether to clean existing environment

        Returns:
            Environment information
        """
        env_path = self.base_dir / f".{name}"

        if env_path.exists() and clean:
            self.logger.info(f"Cleaning existing environment: {name}")
            shutil.rmtree(env_path)

        if env_path.exists():
            self.logger.info(f"Using existing environment: {name}")
            return self._get_environment_info(name)

        self.logger.info(f"Creating new environment: {name}")

        # Create virtual environment
        self._create_virtual_environment(env_path, python_version)

        # Install packages
        if packages or requirements_file:
            self._install_packages(env_path, packages, requirements_file)

        # Store environment info
        env_info = {
            "name": name,
            "path": str(env_path),
            "python_version": python_version,
            "created_at": time.time(),
            "packages": packages or [],
            "requirements_file": requirements_file
        }

        self._save_environment_info(name, env_info)
        self.active_environments[name] = env_info

        return env_info

    def _create_virtual_environment(self, env_path: Path, python_version: str):
        """Create virtual environment using venv or virtualenv."""
        try:
            # Try using python -m venv first
            cmd = [f"python{python_version}", "-m", "venv", str(env_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                # Fall back to virtualenv
                cmd = ["virtualenv", "-p", f"python{python_version}", str(env_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode != 0:
                    raise RuntimeError(f"Failed to create virtual environment: {result.stderr}")

            self.logger.info(f"Virtual environment created at: {env_path}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Virtual environment creation timed out")
        except FileNotFoundError as e:
            raise RuntimeError(f"Python interpreter not found: {e}")

    def _install_packages(
        self,
        env_path: Path,
        packages: Optional[List[str]],
        requirements_file: Optional[str]
    ):
        """Install packages in the virtual environment."""
        pip_path = self._get_pip_path(env_path)

        try:
            # Upgrade pip first
            subprocess.run([
                str(pip_path), "install", "--upgrade", "pip"
            ], check=True, capture_output=True, timeout=300)

            # Install from requirements file
            if requirements_file and Path(requirements_file).exists():
                self.logger.info(f"Installing from requirements file: {requirements_file}")
                subprocess.run([
                    str(pip_path), "install", "-r", requirements_file
                ], check=True, capture_output=True, timeout=1800)  # 30 min timeout

            # Install individual packages
            if packages:
                self.logger.info(f"Installing packages: {packages}")
                subprocess.run([
                    str(pip_path), "install"
                ] + packages, check=True, capture_output=True, timeout=1800)

        except subprocess.TimeoutExpired:
            raise RuntimeError("Package installation timed out")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Package installation failed: {e}")

    def _get_pip_path(self, env_path: Path) -> Path:
        """Get pip executable path for the environment."""
        if os.name == 'nt':  # Windows
            return env_path / "Scripts" / "pip.exe"
        else:  # Unix-like
            return env_path / "bin" / "pip"

    def _get_python_path(self, env_path: Path) -> Path:
        """Get Python executable path for the environment."""
        if os.name == 'nt':  # Windows
            return env_path / "Scripts" / "python.exe"
        else:  # Unix-like
            return env_path / "bin" / "python"

    def _save_environment_info(self, name: str, info: Dict[str, Any]):
        """Save environment information to file."""
        info_file = self.base_dir / f"{name}.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

    def _get_environment_info(self, name: str) -> Dict[str, Any]:
        """Get environment information from file."""
        info_file = self.base_dir / f"{name}.json"
        if info_file.exists():
            with open(info_file) as f:
                return json.load(f)
        else:
            env_path = self.base_dir / f".{name}"
            return {
                "name": name,
                "path": str(env_path),
                "created_at": time.time(),
                "packages": []
            }

    def run_tests_in_environment(
        self,
        env_name: str,
        test_command: str,
        working_dir: Optional[str] = None,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """Run tests in a specific environment.

        Args:
            env_name: Name of the environment
            test_command: Test command to run
            working_dir: Working directory for tests
            timeout: Timeout in seconds

        Returns:
            Test execution results
        """
        env_info = self._get_environment_info(env_name)
        env_path = Path(env_info["path"])

        if not env_path.exists():
            raise ValueError(f"Environment {env_name} does not exist")

        python_path = self._get_python_path(env_path)

        # Prepare environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd() / "src")

        # Set virtual environment activation
        if os.name == 'nt':  # Windows
            env["PATH"] = f"{env_path / 'Scripts'};{env['PATH']}"
        else:  # Unix-like
            env["PATH"] = f"{env_path / 'bin'}:{env['PATH']}"
            env["VIRTUAL_ENV"] = str(env_path)

        self.logger.info(f"Running tests in environment {env_name}: {test_command}")

        start_time = time.time()

        try:
            # Replace 'python' in command with full path
            if test_command.startswith("python "):
                test_command = test_command.replace("python ", f"{python_path} ", 1)
            elif test_command == "python":
                test_command = str(python_path)

            result = subprocess.run(
                test_command.split(),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir or os.getcwd(),
                env=env
            )

            execution_time = time.time() - start_time

            return {
                "environment": env_name,
                "command": test_command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time,
                "success": result.returncode == 0,
                "timeout": False
            }

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return {
                "environment": env_name,
                "command": test_command,
                "returncode": -1,
                "stdout": "",
                "stderr": "Test execution timed out",
                "execution_time": execution_time,
                "success": False,
                "timeout": True
            }

    def create_test_matrix(self, config_file: str) -> List[Dict[str, Any]]:
        """Create test environments from configuration matrix.

        Args:
            config_file: Path to YAML configuration file

        Returns:
            List of created environments
        """
        with open(config_file) as f:
            config = yaml.safe_load(f)

        environments = []

        for env_config in config.get("environments", []):
            name = env_config["name"]

            try:
                env_info = self.create_environment(
                    name=name,
                    python_version=env_config.get("python_version", "3.11"),
                    packages=env_config.get("packages", []),
                    requirements_file=env_config.get("requirements_file"),
                    clean=env_config.get("clean", False)
                )
                environments.append(env_info)
                self.logger.info(f"Created environment: {name}")

            except Exception as e:
                self.logger.error(f"Failed to create environment {name}: {e}")
                environments.append({
                    "name": name,
                    "error": str(e),
                    "success": False
                })

        return environments

    def run_test_matrix(
        self,
        config_file: str,
        test_command: str = "python -m pytest tests/",
        parallel: bool = False
    ) -> Dict[str, Any]:
        """Run tests across multiple environments.

        Args:
            config_file: Path to test matrix configuration
            test_command: Test command to run
            parallel: Whether to run tests in parallel

        Returns:
            Test matrix results
        """
        with open(config_file) as f:
            config = yaml.safe_load(f)

        results = {
            "config_file": config_file,
            "test_command": test_command,
            "environments": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0
            }
        }

        environments = config.get("environments", [])

        if parallel:
            # Run tests in parallel (simplified implementation)
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_env = {}

                for env_config in environments:
                    env_name = env_config["name"]
                    future = executor.submit(
                        self.run_tests_in_environment,
                        env_name,
                        test_command
                    )
                    future_to_env[future] = env_name

                for future in concurrent.futures.as_completed(future_to_env):
                    env_name = future_to_env[future]
                    try:
                        test_result = future.result()
                        results["environments"].append(test_result)
                    except Exception as e:
                        results["environments"].append({
                            "environment": env_name,
                            "error": str(e),
                            "success": False
                        })
        else:
            # Run tests sequentially
            for env_config in environments:
                env_name = env_config["name"]

                try:
                    test_result = self.run_tests_in_environment(env_name, test_command)
                    results["environments"].append(test_result)
                except Exception as e:
                    results["environments"].append({
                        "environment": env_name,
                        "error": str(e),
                        "success": False
                    })

        # Calculate summary
        results["summary"]["total"] = len(results["environments"])
        for env_result in results["environments"]:
            if env_result.get("success"):
                results["summary"]["passed"] += 1
            elif env_result.get("timeout") or "error" in env_result:
                results["summary"]["errors"] += 1
            else:
                results["summary"]["failed"] += 1

        return results

    def cleanup_environment(self, name: str):
        """Clean up a test environment."""
        env_path = self.base_dir / f".{name}"
        info_file = self.base_dir / f"{name}.json"

        if env_path.exists():
            shutil.rmtree(env_path)
            self.logger.info(f"Removed environment directory: {env_path}")

        if info_file.exists():
            info_file.unlink()
            self.logger.info(f"Removed environment info: {info_file}")

        if name in self.active_environments:
            del self.active_environments[name]

    def cleanup_all_environments(self):
        """Clean up all test environments."""
        for env_dir in self.base_dir.glob(".*"):
            if env_dir.is_dir():
                shutil.rmtree(env_dir)

        for info_file in self.base_dir.glob("*.json"):
            info_file.unlink()

        self.active_environments.clear()
        self.logger.info("Cleaned up all test environments")

    def list_environments(self) -> List[Dict[str, Any]]:
        """List all available test environments."""
        environments = []

        for info_file in self.base_dir.glob("*.json"):
            try:
                with open(info_file) as f:
                    env_info = json.load(f)
                environments.append(env_info)
            except Exception as e:
                self.logger.warning(f"Could not read environment info {info_file}: {e}")

        return environments

    def generate_report(self, matrix_results: Dict[str, Any], output_file: str = "test_matrix_report.html") -> str:
        """Generate HTML report for test matrix results."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Matrix Report - Pynomaly</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background: white; padding: 15px; border: 1px solid #ddd; border-radius: 5px; flex: 1; }}
                .success {{ color: #28a745; }}
                .failure {{ color: #dc3545; }}
                .error {{ color: #fd7e14; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #f8f9fa; }}
                pre {{ background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; max-height: 200px; }}
                .collapsible {{ cursor: pointer; background: #f1f1f1; padding: 10px; border: none; width: 100%; text-align: left; }}
                .content {{ display: none; padding: 10px; background: #f9f9f9; }}
            </style>
            <script>
                function toggleContent(id) {{
                    var content = document.getElementById(id);
                    if (content.style.display === "none") {{
                        content.style.display = "block";
                    }} else {{
                        content.style.display = "none";
                    }}
                }}
            </script>
        </head>
        <body>
            <div class="header">
                <h1>Test Matrix Report</h1>
                <p>Command: {matrix_results['test_command']}</p>
                <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="summary">
                <div class="metric">
                    <h3>Total Environments</h3>
                    <div style="font-size: 24px; font-weight: bold;">{matrix_results['summary']['total']}</div>
                </div>
                <div class="metric">
                    <h3>Passed</h3>
                    <div style="font-size: 24px; font-weight: bold; color: #28a745;">{matrix_results['summary']['passed']}</div>
                </div>
                <div class="metric">
                    <h3>Failed</h3>
                    <div style="font-size: 24px; font-weight: bold; color: #dc3545;">{matrix_results['summary']['failed']}</div>
                </div>
                <div class="metric">
                    <h3>Errors</h3>
                    <div style="font-size: 24px; font-weight: bold; color: #fd7e14;">{matrix_results['summary']['errors']}</div>
                </div>
            </div>

            <h2>Environment Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Environment</th>
                        <th>Status</th>
                        <th>Execution Time</th>
                        <th>Return Code</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
        """

        for i, env_result in enumerate(matrix_results['environments']):
            env_name = env_result.get('environment', 'Unknown')

            if env_result.get('success'):
                status = '<span class="success">PASSED</span>'
            elif env_result.get('timeout'):
                status = '<span class="error">TIMEOUT</span>'
            elif 'error' in env_result:
                status = '<span class="error">ERROR</span>'
            else:
                status = '<span class="failure">FAILED</span>'

            execution_time = env_result.get('execution_time', 0)
            return_code = env_result.get('returncode', 'N/A')

            html_content += f"""
                    <tr>
                        <td>{env_name}</td>
                        <td>{status}</td>
                        <td>{execution_time:.2f}s</td>
                        <td>{return_code}</td>
                        <td>
                            <button class="collapsible" onclick="toggleContent('details{i}')">Show Details</button>
                            <div id="details{i}" class="content">
            """

            if 'stdout' in env_result and env_result['stdout']:
                html_content += f'<h4>Output:</h4><pre>{env_result["stdout"]}</pre>'

            if 'stderr' in env_result and env_result['stderr']:
                html_content += f'<h4>Errors:</h4><pre>{env_result["stderr"]}</pre>'

            if 'error' in env_result:
                html_content += f'<h4>Error:</h4><pre>{env_result["error"]}</pre>'

            html_content += """
                            </div>
                        </td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """

        with open(output_file, 'w') as f:
            f.write(html_content)

        return output_file


@click.group()
def cli():
    """Test environment management CLI."""
    logging.basicConfig(level=logging.INFO)


@cli.command()
@click.option("--name", "-n", required=True, help="Environment name")
@click.option("--python-version", "-p", default="3.11", help="Python version")
@click.option("--packages", "-pkg", multiple=True, help="Packages to install")
@click.option("--requirements", "-r", help="Requirements file")
@click.option("--clean", is_flag=True, help="Clean existing environment")
def create(name, python_version, packages, requirements, clean):
    """Create a new test environment."""
    manager = TestEnvironmentManager()

    try:
        env_info = manager.create_environment(
            name=name,
            python_version=python_version,
            packages=list(packages) if packages else None,
            requirements_file=requirements,
            clean=clean
        )

        click.echo(f"✅ Environment created: {name}")
        click.echo(f"   Path: {env_info['path']}")
        click.echo(f"   Python: {env_info['python_version']}")

    except Exception as e:
        click.echo(f"❌ Failed to create environment: {e}")


@cli.command()
@click.option("--config", "-c", required=True, help="Test matrix config file")
@click.option("--command", "-cmd", default="python -m pytest tests/", help="Test command")
@click.option("--parallel", is_flag=True, help="Run tests in parallel")
@click.option("--report", "-r", default="test_matrix_report.html", help="Report output file")
def matrix(config, command, parallel, report):
    """Run test matrix across multiple environments."""
    manager = TestEnvironmentManager()

    try:
        click.echo(f"🧪 Creating test environments from: {config}")
        environments = manager.create_test_matrix(config)

        click.echo(f"🚀 Running test matrix with command: {command}")
        results = manager.run_test_matrix(config, command, parallel)

        # Generate report
        report_path = manager.generate_report(results, report)

        # Show summary
        summary = results['summary']
        click.echo(f"\n📊 Test Matrix Results:")
        click.echo(f"   Total: {summary['total']}")
        click.echo(f"   Passed: {summary['passed']}")
        click.echo(f"   Failed: {summary['failed']}")
        click.echo(f"   Errors: {summary['errors']}")
        click.echo(f"   Report: {report_path}")

        # Exit with error if any tests failed
        if summary['failed'] > 0 or summary['errors'] > 0:
            click.echo("❌ Some tests failed")
            exit(1)
        else:
            click.echo("✅ All tests passed")

    except Exception as e:
        click.echo(f"❌ Test matrix failed: {e}")
        exit(1)


@cli.command()
def list():
    """List all test environments."""
    manager = TestEnvironmentManager()
    environments = manager.list_environments()

    if not environments:
        click.echo("No test environments found")
        return

    click.echo("Test Environments:")
    for env in environments:
        click.echo(f"  📦 {env['name']}")
        click.echo(f"     Path: {env['path']}")
        click.echo(f"     Python: {env.get('python_version', 'Unknown')}")
        click.echo(f"     Created: {time.ctime(env.get('created_at', 0))}")


@cli.command()
@click.option("--name", "-n", help="Environment name to clean (all if not specified)")
@click.option("--all", "clean_all", is_flag=True, help="Clean all environments")
def clean(name, clean_all):
    """Clean up test environments."""
    manager = TestEnvironmentManager()

    if clean_all or name is None:
        manager.cleanup_all_environments()
        click.echo("🧹 Cleaned up all test environments")
    elif name:
        manager.cleanup_environment(name)
        click.echo(f"🧹 Cleaned up environment: {name}")


if __name__ == "__main__":
    cli()
