"""
Comprehensive tests for FastAPI API template validation.

Tests template integrity, variable substitution, dependency validation,
and generated project structure for the FastAPI template.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest
from unittest.mock import patch, MagicMock

from test_utilities.factories import TestDataFactory
from test_utilities.fixtures import async_test


class TestFastAPITemplateValidation:
    """Test FastAPI template structure and validation."""
    
    @pytest.fixture
    def template_vars(self) -> Dict[str, str]:
        """Template variables for testing."""
        return {
            "package_name": "test_api",
            "description": "Test FastAPI Application", 
            "author": "Test Author <test@example.com>"
        }
    
    @pytest.fixture
    def template_root(self) -> Path:
        """Path to FastAPI template root."""
        return Path("/mnt/c/Users/andre/monorepo/src/packages/templates/fastapi_api")
    
    def test_template_directory_structure(self, template_root: Path):
        """Test that all required template files exist."""
        required_files = [
            "pyproject.toml.template",
            "Dockerfile.template",
            "README.md",
            "src/{{package_name}}/__init__.py.template",
            "src/{{package_name}}/main.py.template",
            "src/{{package_name}}/api/health.py.template",
            "src/{{package_name}}/api/v1/items.py.template",
            "src/{{package_name}}/api/v1/users.py.template",
            "src/{{package_name}}/core/config.py.template",
            "src/{{package_name}}/core/logging.py.template",
            "src/{{package_name}}/db/base.py.template",
            "src/{{package_name}}/db/session.py.template",
            "src/{{package_name}}/schemas/item.py.template",
            "src/{{package_name}}/schemas/user.py.template",
            "src/{{package_name}}/services/base.py.template",
            "src/{{package_name}}/services/item.py.template",
            "src/{{package_name}}/services/user.py.template"
        ]
        
        for file_path in required_files:
            full_path = template_root / file_path
            assert full_path.exists(), f"Required template file missing: {file_path}"
    
    def test_pyproject_template_structure(self, template_root: Path):
        """Test pyproject.toml template has correct structure."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # Check template variables
        assert "{{package_name}}" in content
        assert "{{description}}" in content
        assert "{{author}}" in content
        
        # Check required dependencies
        required_deps = [
            "fastapi = \"^0.109.0\"",
            "uvicorn",
            "pydantic = \"^2.5.0\"",
            "sqlalchemy = \"^2.0.25\"",
            "alembic = \"^1.13.1\"",
            "asyncpg = \"^0.29.0\"",
            "redis = \"^5.0.1\"",
            "structlog = \"^24.1.0\""
        ]
        
        for dep in required_deps:
            assert dep in content, f"Missing dependency: {dep}"
        
        # Check dev dependencies
        dev_deps = [
            "pytest = \"^7.4.4\"",
            "pytest-asyncio = \"^0.23.3\"",
            "pytest-cov = \"^4.1.0\"",
            "black = \"^23.12.1\"",
            "ruff = \"^0.1.11\"",
            "mypy = \"^1.8.0\""
        ]
        
        for dep in dev_deps:
            assert dep in content, f"Missing dev dependency: {dep}"
    
    def test_template_variable_substitution(self, template_root: Path, template_vars: Dict[str, str]):
        """Test template variable substitution works correctly."""
        pyproject_path = template_root / "pyproject.toml.template"
        template_content = pyproject_path.read_text()
        
        # Substitute variables
        substituted = template_content
        for var, value in template_vars.items():
            substituted = substituted.replace(f"{{{{{var}}}}}", value)
        
        # Verify substitution worked
        assert template_vars["package_name"] in substituted
        assert template_vars["description"] in substituted
        assert template_vars["author"] in substituted
        assert "{{" not in substituted, "Template variables not fully substituted"
    
    def test_main_template_structure(self, template_root: Path):
        """Test main.py template has correct FastAPI structure."""
        main_path = template_root / "src/{{package_name}}/main.py.template"
        content = main_path.read_text()
        
        # Check FastAPI imports and setup
        assert "from fastapi import FastAPI" in content
        assert "app = FastAPI(" in content
        assert "{{package_name}}" in content
        
        # Check router inclusion
        assert "from .api" in content or "include_router" in content
    
    def test_api_endpoints_structure(self, template_root: Path):
        """Test API endpoint templates have correct structure."""
        # Test health endpoint
        health_path = template_root / "src/{{package_name}}/api/health.py.template"
        health_content = health_path.read_text()
        
        assert "from fastapi import APIRouter" in health_content
        assert "router = APIRouter()" in health_content
        assert "@router.get" in health_content
        
        # Test items endpoint
        items_path = template_root / "src/{{package_name}}/api/v1/items.py.template"
        items_content = items_path.read_text()
        
        assert "from fastapi import APIRouter" in items_content
        assert "router = APIRouter()" in items_content
        assert "prefix=\"/items\"" in items_content or "/items" in items_content
        
        # Test users endpoint
        users_path = template_root / "src/{{package_name}}/api/v1/users.py.template"
        users_content = users_path.read_text()
        
        assert "from fastapi import APIRouter" in users_content
        assert "router = APIRouter()" in users_content
        assert "prefix=\"/users\"" in users_content or "/users" in users_content
    
    def test_config_template_structure(self, template_root: Path):
        """Test configuration template has correct structure."""
        config_path = template_root / "src/{{package_name}}/core/config.py.template"
        content = config_path.read_text()
        
        # Check pydantic settings usage
        assert "from pydantic_settings import BaseSettings" in content or "BaseSettings" in content
        assert "class Settings" in content
        assert "class Config:" in content or "model_config" in content
    
    def test_database_template_structure(self, template_root: Path):
        """Test database templates have correct structure."""
        # Test base database setup
        base_path = template_root / "src/{{package_name}}/db/base.py.template"
        base_content = base_path.read_text()
        
        assert "from sqlalchemy" in base_content
        assert "Base" in base_content
        
        # Test session management
        session_path = template_root / "src/{{package_name}}/db/session.py.template"
        session_content = session_path.read_text()
        
        assert "from sqlalchemy.ext.asyncio import" in session_content or "AsyncSession" in session_content
        assert "engine" in session_content
        assert "SessionLocal" in session_content or "async_session" in session_content
    
    def test_schema_templates_structure(self, template_root: Path):
        """Test Pydantic schema templates have correct structure."""
        # Test item schema
        item_path = template_root / "src/{{package_name}}/schemas/item.py.template"
        item_content = item_path.read_text()
        
        assert "from pydantic import" in item_content
        assert "BaseModel" in item_content
        assert "class" in item_content
        
        # Test user schema
        user_path = template_root / "src/{{package_name}}/schemas/user.py.template"
        user_content = user_path.read_text()
        
        assert "from pydantic import" in user_content
        assert "BaseModel" in user_content
        assert "class" in user_content
    
    def test_service_templates_structure(self, template_root: Path):
        """Test service templates have correct structure."""
        # Test base service
        base_path = template_root / "src/{{package_name}}/services/base.py.template"
        base_content = base_path.read_text()
        
        assert "class" in base_content
        assert "BaseService" in base_content or "Service" in base_content
        
        # Test item service
        item_path = template_root / "src/{{package_name}}/services/item.py.template"
        item_content = item_path.read_text()
        
        assert "class" in item_content
        assert "ItemService" in item_content or "Service" in item_content
        
        # Test user service
        user_path = template_root / "src/{{package_name}}/services/user.py.template"
        user_content = user_path.read_text()
        
        assert "class" in user_content
        assert "UserService" in user_content or "Service" in user_content
    
    def test_dockerfile_template_structure(self, template_root: Path):
        """Test Dockerfile template has correct structure."""
        dockerfile_path = template_root / "Dockerfile.template"
        content = dockerfile_path.read_text()
        
        # Check base image and Python setup
        assert "FROM python:" in content
        assert "WORKDIR" in content
        assert "COPY" in content
        assert "RUN pip install" in content or "poetry install" in content
        assert "CMD" in content or "ENTRYPOINT" in content
        
        # Check template variables
        assert "{{package_name}}" in content
    
    def test_logging_template_structure(self, template_root: Path):
        """Test logging configuration template."""
        logging_path = template_root / "src/{{package_name}}/core/logging.py.template"
        content = logging_path.read_text()
        
        # Check structlog usage
        assert "structlog" in content
        assert "configure" in content or "get_logger" in content
        assert "def" in content  # Should have logging setup functions
    
    def test_template_consistency(self, template_root: Path):
        """Test template files are consistent with each other."""
        # Check that all template files use consistent variable naming
        template_files = list(template_root.rglob("*.template"))
        
        for template_file in template_files:
            if template_file.is_file():
                content = template_file.read_text()
                
                # All should use {{package_name}} consistently
                if "package_name" in content:
                    assert "{{package_name}}" in content, f"Inconsistent variable in {template_file}"
                    # Should not have malformed variables
                    assert "{package_name}" not in content, f"Malformed variable in {template_file}"
                    assert "{ package_name }" not in content, f"Malformed variable in {template_file}"
    
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
    
    def test_import_validity(self, template_root: Path, template_vars: Dict[str, str]):
        """Test that template imports are valid after substitution."""
        python_templates = list(template_root.rglob("*.py.template"))
        
        for template_file in python_templates:
            content = template_file.read_text()
            
            # Substitute variables
            for var, value in template_vars.items():
                content = content.replace(f"{{{{{var}}}}}", value)
            
            # Check for relative imports that might be invalid
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                if line.strip().startswith('from .') or line.strip().startswith('import .'):
                    # Basic validation of relative import structure
                    assert not line.strip().endswith('.'), f"Invalid relative import in {template_file}:{line_num}"


class TestFastAPITemplateDependencies:
    """Test FastAPI template dependency validation."""
    
    @pytest.fixture
    def template_root(self) -> Path:
        """Path to FastAPI template root."""
        return Path("/mnt/c/Users/andre/monorepo/src/packages/templates/fastapi_api")
    
    def test_dependency_versions_compatibility(self, template_root: Path):
        """Test that dependency versions are compatible."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # Check for known incompatible version combinations
        # FastAPI and Pydantic should be compatible
        if "fastapi = \"^0.109.0\"" in content:
            assert "pydantic = \"^2.5.0\"" in content, "FastAPI 0.109+ requires Pydantic v2"
        
        # SQLAlchemy and asyncpg should be compatible
        if "sqlalchemy = \"^2.0.25\"" in content:
            assert "asyncpg = \"^0.29.0\"" in content, "SQLAlchemy 2.x works with asyncpg 0.29+"
    
    def test_security_dependencies(self, template_root: Path):
        """Test that security-related dependencies are included."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # Should include dependencies for common security needs
        security_related = [
            "pydantic",  # For data validation
            "structlog", # For secure logging
            "python-dotenv"  # For environment variable management
        ]
        
        for dep in security_related:
            assert dep in content, f"Missing security-related dependency: {dep}"
    
    def test_development_dependencies_completeness(self, template_root: Path):
        """Test that all necessary development dependencies are included."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        dev_tools = [
            "pytest",        # Testing
            "pytest-asyncio", # Async testing
            "pytest-cov",    # Coverage
            "black",         # Code formatting
            "ruff",          # Linting
            "mypy",          # Type checking
            "pre-commit"     # Git hooks
        ]
        
        for tool in dev_tools:
            assert tool in content, f"Missing development dependency: {tool}"


class TestFastAPITemplateGeneration:
    """Test FastAPI template generation and output validation."""
    
    @pytest.fixture
    def template_vars(self) -> Dict[str, str]:
        """Template variables for testing."""
        return {
            "package_name": "my_api",
            "description": "My FastAPI Application",
            "author": "John Doe <john@example.com>"
        }
    
    def test_generate_complete_project(self, template_vars: Dict[str, str]):
        """Test generating a complete project from template."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/fastapi_api")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / template_vars["package_name"]
            output_dir.mkdir()
            
            # Simulate template processing
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
            
            # Verify generated project structure
            expected_files = [
                f"pyproject.toml",
                f"src/{template_vars['package_name']}/main.py",
                f"src/{template_vars['package_name']}/api/health.py",
                f"src/{template_vars['package_name']}/core/config.py"
            ]
            
            for expected_file in expected_files:
                assert (output_dir / expected_file).exists(), f"Generated file missing: {expected_file}"
    
    def test_generated_project_imports(self, template_vars: Dict[str, str]):
        """Test that generated project has correct import statements."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/fastapi_api")
        
        # Test main.py imports
        main_template = template_root / "src/{{package_name}}/main.py.template"
        main_content = main_template.read_text()
        
        # Substitute variables
        for var, value in template_vars.items():
            main_content = main_content.replace(f"{{{{{var}}}}}", value)
        
        # Check that imports reference the correct package name
        if f"from {template_vars['package_name']}" in main_content:
            assert f"from {template_vars['package_name']}.api" in main_content or \
                   f"from {template_vars['package_name']}.core" in main_content
    
    def test_generated_configuration_validity(self, template_vars: Dict[str, str]):
        """Test that generated configuration is valid."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/fastapi_api")
        config_template = template_root / "src/{{package_name}}/core/config.py.template"
        content = config_template.read_text()
        
        # Substitute variables
        for var, value in template_vars.items():
            content = content.replace(f"{{{{{var}}}}}", value)
        
        # Verify configuration class naming
        assert "class Settings" in content
        assert template_vars["package_name"].upper() in content or template_vars["package_name"] in content