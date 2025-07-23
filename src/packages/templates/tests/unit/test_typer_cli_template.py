"""
Comprehensive tests for Typer CLI template validation.

Tests template integrity, variable substitution, CLI command structure,
argument parsing, and script generation for the Typer CLI template.
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest
from unittest.mock import patch, MagicMock

from test_utilities.factories import TestDataFactory
from test_utilities.fixtures import async_test


class TestTyperCLITemplateValidation:
    """Test Typer CLI template structure and validation."""
    
    @pytest.fixture
    def template_vars(self) -> Dict[str, str]:
        """Template variables for testing."""
        return {
            "package_name": "test_cli",
            "description": "Test Typer CLI Application",
            "author": "Test Author <test@example.com>"
        }
    
    @pytest.fixture
    def template_root(self) -> Path:
        """Path to Typer CLI template root."""
        return Path("/mnt/c/Users/andre/monorepo/src/packages/templates/typer_cli")
    
    def test_template_directory_structure(self, template_root: Path):
        """Test that all required template files exist."""
        required_files = [
            "pyproject.toml.template",
            "README.md",
            "src/{{package_name}}/__init__.py.template",
            "src/{{package_name}}/cli.py.template",
            "src/{{package_name}}/commands/config.py.template",
            "src/{{package_name}}/commands/data.py.template",
            "src/{{package_name}}/commands/process.py.template",
            "src/{{package_name}}/core/config.py.template",
            "src/{{package_name}}/core/logging.py.template",
            "src/{{package_name}}/utils/data.py.template",
            "src/{{package_name}}/utils/process.py.template"
        ]
        
        for file_path in required_files:
            full_path = template_root / file_path
            assert full_path.exists(), f"Required template file missing: {file_path}"
    
    def test_pyproject_template_cli_structure(self, template_root: Path):
        """Test pyproject.toml template has correct CLI setup."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # Check template variables
        assert "{{package_name}}" in content
        assert "{{description}}" in content
        assert "{{author}}" in content
        
        # Check CLI script configuration
        assert "[tool.poetry.scripts]" in content
        assert "{{package_name}} = \"{{package_name}}.cli:app\"" in content
        
        # Check Typer CLI dependencies
        cli_deps = [
            "typer",
            "rich = \"^13.7.0\"",
            "click = \"^8.1.7\"",
            "pydantic = \"^2.5.0\"",
            "structlog = \"^24.1.0\"",
            "python-dotenv = \"^1.0.0\""
        ]
        
        for dep in cli_deps:
            assert dep in content, f"Missing CLI dependency: {dep}"
    
    def test_main_cli_template_structure(self, template_root: Path):
        """Test main CLI template has correct Typer setup."""
        cli_path = template_root / "src/{{package_name}}/cli.py.template"
        content = cli_path.read_text()
        
        # Check Typer imports and setup
        assert "import typer" in content
        assert "app = typer.Typer(" in content
        
        # Check Rich console setup for better CLI output
        assert "from rich" in content or "rich" in content
        
        # Check command registration
        assert "add_typer" in content or "@app" in content or "command" in content
        
        # Check main function
        assert "def main(" in content or "if __name__ == \"__main__\":" in content
    
    def test_command_templates_structure(self, template_root: Path):
        """Test command templates have correct Typer structure."""
        commands = ["config.py", "data.py", "process.py"]
        commands_dir = template_root / "src/{{package_name}}/commands"
        
        for command_file in commands:
            command_path = commands_dir / f"{command_file}.template"
            content = command_path.read_text()
            
            # Check Typer command setup
            assert "import typer" in content
            assert "from typer import" in content or "typer." in content
            
            # Check command definition patterns
            assert "def " in content  # Should have command functions
            assert "@" in content or "typer.Typer()" in content  # Should have decorators or Typer app
            
            # Check Rich usage for better output
            assert "rich" in content or "console" in content or "print" in content
    
    def test_config_command_template(self, template_root: Path):
        """Test config command template has correct structure."""
        config_path = template_root / "src/{{package_name}}/commands/config.py.template"
        content = config_path.read_text()
        
        # Should have config-related functions
        config_functions = ["show", "set", "get", "list", "init"]
        has_config_function = any(func in content.lower() for func in config_functions)
        assert has_config_function, "Config command should have config management functions"
        
        # Should handle configuration files
        assert "config" in content.lower()
        assert "typer" in content
    
    def test_data_command_template(self, template_root: Path):
        """Test data command template has correct structure.""" 
        data_path = template_root / "src/{{package_name}}/commands/data.py.template"
        content = data_path.read_text()
        
        # Should have data-related functions
        data_functions = ["load", "save", "export", "import", "process"]
        has_data_function = any(func in content.lower() for func in data_functions)
        assert has_data_function, "Data command should have data processing functions"
        
        # Should use data utilities
        assert "data" in content.lower()
        assert "typer" in content
    
    def test_process_command_template(self, template_root: Path):
        """Test process command template has correct structure."""
        process_path = template_root / "src/{{package_name}}/commands/process.py.template"
        content = process_path.read_text()
        
        # Should have process-related functions
        process_functions = ["run", "start", "stop", "status", "execute"]
        has_process_function = any(func in content.lower() for func in process_functions)
        assert has_process_function, "Process command should have process management functions"
        
        # Should handle process operations
        assert "process" in content.lower()
        assert "typer" in content
    
    def test_core_config_template(self, template_root: Path):
        """Test core configuration template structure."""
        config_path = template_root / "src/{{package_name}}/core/config.py.template"
        content = config_path.read_text()
        
        # Check Pydantic settings usage
        assert "from pydantic" in content
        assert "BaseSettings" in content or "Settings" in content
        
        # Check environment variable support
        assert "dotenv" in content or "env" in content or "getenv" in content
        
        # Should have configuration class
        assert "class" in content
        assert "Config" in content or "Settings" in content
    
    def test_logging_template_structure(self, template_root: Path):
        """Test logging configuration template."""
        logging_path = template_root / "src/{{package_name}}/core/logging.py.template"
        content = logging_path.read_text()
        
        # Check structlog usage for better CLI logging
        assert "structlog" in content
        assert "configure" in content or "get_logger" in content
        
        # Should have logging setup functions
        assert "def" in content
        assert "setup" in content.lower() or "configure" in content.lower()
    
    def test_utility_templates_structure(self, template_root: Path):
        """Test utility templates have correct structure."""
        utils = ["data.py", "process.py"]
        utils_dir = template_root / "src/{{package_name}}/utils"
        
        for util_file in utils:
            util_path = utils_dir / f"{util_file}.template"
            content = util_path.read_text()
            
            # Should have utility functions
            assert "def " in content  # Should have functions
            
            # Should have proper imports
            assert "from" in content or "import" in content
            
            # Should be relevant to the utility type
            if util_file == "data.py":
                assert "data" in content.lower()
            elif util_file == "process.py":
                assert "process" in content.lower()
    
    def test_template_variable_substitution(self, template_root: Path, template_vars: Dict[str, str]):
        """Test template variable substitution works correctly."""
        template_files = list(template_root.rglob("*.template"))
        
        for template_file in template_files:
            if template_file.is_file():
                content = template_file.read_text()
                
                # Substitute variables
                substituted = content
                for var, value in template_vars.items():
                    substituted = substituted.replace(f"{{{{{var}}}}}", value)
                
                # Verify substitution worked if template contained variables
                if "{{package_name}}" in content:
                    assert template_vars["package_name"] in substituted
                    # Count should decrease after substitution
                    original_count = content.count("{{package_name}}")
                    substituted_count = substituted.count("{{package_name}}")
                    assert substituted_count == 0, f"Template variables not fully substituted in {template_file}"
    
    def test_cli_script_generation(self, template_root: Path, template_vars: Dict[str, str]):
        """Test CLI script entry point generation."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # Substitute template variables
        for var, value in template_vars.items():
            content = content.replace(f"{{{{{var}}}}}", value)
        
        # Check script entry point
        expected_script = f"{template_vars['package_name']} = \"{template_vars['package_name']}.cli:app\""
        assert expected_script in content, "CLI script entry point not correctly generated"
    
    def test_python_syntax_validity(self, template_root: Path, template_vars: Dict[str, str]):
        """Test that substituted Python files have valid syntax."""
        python_templates = list(template_root.rglob("*.py.template"))
        
        for template_file in python_templates:
            content = template_file.read_text()
            
            # Substitute variables
            for var, value in template_vars.items():
                content = content.replace(f"{{{{{var}}}}}", value)
            
            # Try to compile the Python code
            try:
                compile(content, str(template_file), 'exec')
            except SyntaxError as e:
                pytest.fail(f"Invalid Python syntax in {template_file}: {e}")
    
    def test_import_structure_validity(self, template_root: Path, template_vars: Dict[str, str]):
        """Test that template imports are valid after substitution."""
        python_templates = list(template_root.rglob("*.py.template"))
        
        for template_file in python_templates:
            content = template_file.read_text()
            
            # Substitute variables
            for var, value in template_vars.items():
                content = content.replace(f"{{{{{var}}}}}", value)
            
            # Check import statements
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                
                # Check relative imports
                if stripped.startswith('from .') or stripped.startswith('import .'):
                    assert not stripped.endswith('.'), f"Invalid relative import in {template_file}:{line_num}"
                
                # Check package imports after substitution
                if f"from {template_vars['package_name']}" in stripped:
                    assert "." in stripped, f"Package import should use module path in {template_file}:{line_num}"


class TestTyperCLIDependencies:
    """Test Typer CLI template dependency validation."""
    
    @pytest.fixture
    def template_root(self) -> Path:
        """Path to Typer CLI template root."""
        return Path("/mnt/c/Users/andre/monorepo/src/packages/templates/typer_cli")
    
    def test_cli_dependencies_completeness(self, template_root: Path):
        """Test that all necessary CLI dependencies are included."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # Core CLI dependencies
        cli_deps = [
            "typer",  # Main CLI framework
            "rich",   # Better terminal output
            "click",  # Click integration
            "pydantic",  # Configuration validation
            "structlog",  # Structured logging
            "python-dotenv"  # Environment variables
        ]
        
        for dep in cli_deps:
            assert dep in content, f"Missing CLI dependency: {dep}"
    
    def test_typer_version_compatibility(self, template_root: Path):
        """Test Typer version compatibility with other dependencies."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # Check Typer with 'all' extras for full functionality
        assert "typer = {extras = [\"all\"], version = \"^0.9.0\"}" in content
        
        # Rich should be compatible with Typer
        if "typer = {extras = [\"all\"]" in content:
            assert "rich = " in content, "Rich should be included for better CLI output"
    
    def test_development_dependencies_for_cli(self, template_root: Path):
        """Test development dependencies suitable for CLI development."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        cli_dev_deps = [
            "pytest",      # Testing
            "pytest-cov",  # Coverage
            "pytest-mock", # Mocking for CLI tests
            "black",       # Code formatting
            "ruff",        # Linting
            "mypy"         # Type checking
        ]
        
        for dep in cli_dev_deps:
            assert dep in content, f"Missing CLI development dependency: {dep}"
    
    def test_no_unnecessary_dependencies(self, template_root: Path):
        """Test that template doesn't include unnecessary dependencies."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # These shouldn't be in a CLI template
        unnecessary_deps = [
            "fastapi",     # Web framework
            "uvicorn",     # ASGI server
            "jinja2",      # Template engine
            "sqlalchemy",  # Database ORM
            "alembic",     # Database migrations
            "asyncpg",     # Async PostgreSQL
            "redis"        # Redis client
        ]
        
        for dep in unnecessary_deps:
            assert f"{dep} = " not in content, f"Unnecessary dependency found: {dep}"


class TestTyperCLITemplateGeneration:
    """Test Typer CLI template generation and output validation."""
    
    @pytest.fixture
    def template_vars(self) -> Dict[str, str]:
        """Template variables for testing."""
        return {
            "package_name": "my_cli_tool",
            "description": "My CLI Tool Application",
            "author": "CLI Developer <cli@example.com>"
        }
    
    def test_generate_complete_cli_project(self, template_vars: Dict[str, str]):
        """Test generating a complete CLI project from template."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/typer_cli")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / template_vars["package_name"]
            output_dir.mkdir()
            
            # Process template files
            template_files = list(template_root.rglob("*.template"))
            
            for template_file in template_files:
                if template_file.is_file():
                    content = template_file.read_text()
                    
                    # Substitute variables
                    for var, value in template_vars.items():
                        content = content.replace(f"{{{{{var}}}}}", value)
                    
                    # Create output file path
                    rel_path = template_file.relative_to(template_root)
                    output_path = output_dir / str(rel_path).replace('.template', '')
                    
                    # Replace package name in paths
                    output_path_str = str(output_path).replace('{{package_name}}', template_vars["package_name"])
                    output_path = Path(output_path_str)
                    
                    # Create directories
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write processed content
                    output_path.write_text(content)
            
            # Verify generated CLI project structure
            expected_files = [
                f"pyproject.toml",
                f"src/{template_vars['package_name']}/cli.py",
                f"src/{template_vars['package_name']}/commands/config.py",
                f"src/{template_vars['package_name']}/commands/data.py",
                f"src/{template_vars['package_name']}/commands/process.py",
                f"src/{template_vars['package_name']}/core/config.py",
                f"src/{template_vars['package_name']}/core/logging.py",
                f"src/{template_vars['package_name']}/utils/data.py",
                f"src/{template_vars['package_name']}/utils/process.py"
            ]
            
            for expected_file in expected_files:
                file_path = output_dir / expected_file
                assert file_path.exists(), f"Generated CLI file missing: {expected_file}"
    
    def test_generated_cli_script_entry(self, template_vars: Dict[str, str]):
        """Test that generated CLI script has correct entry point."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/typer_cli")
        pyproject_template = template_root / "pyproject.toml.template"
        content = pyproject_template.read_text()
        
        # Substitute variables
        for var, value in template_vars.items():
            content = content.replace(f"{{{{{var}}}}}", value)
        
        # Check script configuration
        expected_script_line = f"{template_vars['package_name']} = \"{template_vars['package_name']}.cli:app\""
        assert expected_script_line in content
        
        # Check that it's in the right section
        assert "[tool.poetry.scripts]" in content
    
    def test_generated_cli_command_structure(self, template_vars: Dict[str, str]):
        """Test that generated CLI commands have correct structure."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/typer_cli")
        cli_template = template_root / "src/{{package_name}}/cli.py.template"
        content = cli_template.read_text()
        
        # Substitute variables
        for var, value in template_vars.items():
            content = content.replace(f"{{{{{var}}}}}", value)
        
        # Check main Typer app setup
        assert "app = typer.Typer(" in content
        
        # Check command imports from the correct package
        if f"from {template_vars['package_name']}" in content:
            assert f"from {template_vars['package_name']}.commands" in content
    
    def test_generated_configuration_structure(self, template_vars: Dict[str, str]):
        """Test that generated configuration is properly structured."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/typer_cli")
        config_template = template_root / "src/{{package_name}}/core/config.py.template"
        content = config_template.read_text()
        
        # Substitute variables
        for var, value in template_vars.items():
            content = content.replace(f"{{{{{var}}}}}", value)
        
        # Check settings class naming
        assert "class Settings" in content or "Config" in content
        
        # Should reference the package name appropriately
        package_upper = template_vars["package_name"].upper()
        if package_upper in content:
            # Environment variable prefixes often use uppercase package names
            assert len(package_upper) > 0
    
    def test_generated_command_imports(self, template_vars: Dict[str, str]):
        """Test that generated commands have correct import structure."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/typer_cli")
        
        commands = ["config.py", "data.py", "process.py"]
        for command_file in commands:
            command_template = template_root / f"src/{{{{package_name}}}}/commands/{command_file}.template"
            if command_template.exists():
                content = command_template.read_text()
                
                # Substitute variables
                for var, value in template_vars.items():
                    content = content.replace(f"{{{{{var}}}}}", value)
                
                # Check imports reference correct package
                if f"from {template_vars['package_name']}" in content:
                    # Should import from utils or core modules
                    assert ("utils" in content or "core" in content), f"Command {command_file} should import from utils or core"