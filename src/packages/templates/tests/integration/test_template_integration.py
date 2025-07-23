"""
Integration tests for template system validation.

Tests template system integrity, cross-template consistency,
and end-to-end template generation workflows.
"""
import os
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pytest
from unittest.mock import patch, MagicMock

from test_utilities.factories import TestDataFactory
from test_utilities.fixtures import async_test


class TestTemplateSystemIntegration:
    """Test template system integration and consistency."""
    
    @pytest.fixture
    def templates_root(self) -> Path:
        """Path to templates root directory."""
        return Path("/mnt/c/Users/andre/monorepo/src/packages/templates")
    
    @pytest.fixture
    def template_types(self) -> List[str]:
        """List of available template types."""
        return ["fastapi_api", "htmx_tailwind_app", "typer_cli"]
    
    def test_all_templates_exist(self, templates_root: Path, template_types: List[str]):
        """Test that all expected template directories exist."""
        for template_type in template_types:
            template_dir = templates_root / template_type
            assert template_dir.exists(), f"Template directory missing: {template_type}"
            assert template_dir.is_dir(), f"Template path is not a directory: {template_type}"
    
    def test_template_consistency_across_types(self, templates_root: Path, template_types: List[str]):
        """Test consistency of template structure across different types."""
        # Check that all templates have required files
        required_files = [
            "pyproject.toml.template",
            "README.md"
        ]
        
        for template_type in template_types:
            template_dir = templates_root / template_type
            for required_file in required_files:
                file_path = template_dir / required_file
                assert file_path.exists(), f"Required file missing in {template_type}: {required_file}"
    
    def test_template_variable_consistency(self, templates_root: Path, template_types: List[str]):
        """Test that template variables are used consistently across templates."""
        common_variables = ["{{package_name}}", "{{description}}", "{{author}}"]
        
        for template_type in template_types:
            template_dir = templates_root / template_type
            pyproject_path = template_dir / "pyproject.toml.template"
            
            if pyproject_path.exists():
                content = pyproject_path.read_text()
                
                for var in common_variables:
                    assert var in content, f"Common variable {var} missing in {template_type} pyproject.toml"
    
    def test_python_version_consistency(self, templates_root: Path, template_types: List[str]):
        """Test that all templates use consistent Python version requirements."""
        expected_python_version = "^3.11"
        
        for template_type in template_types:
            template_dir = templates_root / template_type
            pyproject_path = template_dir / "pyproject.toml.template"
            
            if pyproject_path.exists():
                content = pyproject_path.read_text()
                assert f"python = \"{expected_python_version}\"" in content, \
                    f"Inconsistent Python version in {template_type}"
    
    def test_development_tools_consistency(self, templates_root: Path, template_types: List[str]):
        """Test that templates use consistent development tools."""
        common_dev_tools = ["black", "ruff", "mypy", "pytest"]
        
        for template_type in template_types:
            template_dir = templates_root / template_type
            pyproject_path = template_dir / "pyproject.toml.template"
            
            if pyproject_path.exists():
                content = pyproject_path.read_text()
                
                for tool in common_dev_tools:
                    assert tool in content, f"Common dev tool {tool} missing in {template_type}"
    
    def test_template_source_structure_consistency(self, templates_root: Path, template_types: List[str]):
        """Test that templates have consistent source structure."""
        for template_type in template_types:
            template_dir = templates_root / template_type
            src_dir = template_dir / "src" / "{{package_name}}"
            
            # All templates should have src/{{package_name}} structure
            assert src_dir.exists(), f"Source directory missing in {template_type}"
            
            # All should have __init__.py template
            init_file = src_dir / "__init__.py.template"
            assert init_file.exists(), f"__init__.py.template missing in {template_type}"


class TestTemplateGeneration:
    """Test end-to-end template generation."""
    
    @pytest.fixture
    def template_vars(self) -> Dict[str, str]:
        """Common template variables for testing."""
        return {
            "package_name": "test_generated_app",
            "description": "Test Generated Application",
            "author": "Test Generator <test@example.com>"
        }
    
    def test_generate_all_template_types(self, template_vars: Dict[str, str]):
        """Test generating projects from all template types."""
        templates_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates")
        template_types = ["fastapi_api", "htmx_tailwind_app", "typer_cli"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_base = Path(temp_dir)
            
            for template_type in template_types:
                template_dir = templates_root / template_type
                output_dir = output_base / f"{template_vars['package_name']}_{template_type}"
                output_dir.mkdir()
                
                # Generate project from template
                self._generate_project_from_template(template_dir, output_dir, template_vars)
                
                # Verify generated project
                self._verify_generated_project(output_dir, template_type, template_vars)
    
    def _generate_project_from_template(self, template_dir: Path, output_dir: Path, template_vars: Dict[str, str]):
        """Generate a project from a template directory."""
        # Process all template files
        template_files = list(template_dir.rglob("*.template"))
        
        for template_file in template_files:
            if template_file.is_file():
                # Read and substitute template content
                content = template_file.read_text()
                for var, value in template_vars.items():
                    content = content.replace(f"{{{{{var}}}}}", value)
                
                # Create output file path
                rel_path = template_file.relative_to(template_dir)
                output_path = output_dir / str(rel_path).replace('.template', '')
                
                # Replace package name in path
                output_path_str = str(output_path).replace('{{package_name}}', template_vars["package_name"])
                output_path = Path(output_path_str)
                
                # Create directories and write file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(content)
        
        # Copy non-template files
        for item in template_dir.rglob("*"):
            if item.is_file() and not item.name.endswith('.template'):
                rel_path = item.relative_to(template_dir)
                
                # Skip certain files that shouldn't be copied
                if item.name in ["README.md", ".gitignore"]:
                    continue
                
                output_path = output_dir / str(rel_path)
                output_path_str = str(output_path).replace('{{package_name}}', template_vars["package_name"])
                output_path = Path(output_path_str)
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, output_path)
    
    def _verify_generated_project(self, output_dir: Path, template_type: str, template_vars: Dict[str, str]):
        """Verify a generated project has correct structure."""
        package_name = template_vars["package_name"]
        
        # Check basic structure
        assert (output_dir / "pyproject.toml").exists(), f"pyproject.toml missing in {template_type}"
        assert (output_dir / "src" / package_name).exists(), f"Source package missing in {template_type}"
        assert (output_dir / "src" / package_name / "__init__.py").exists(), f"__init__.py missing in {template_type}"
        
        # Check pyproject.toml content
        pyproject_content = (output_dir / "pyproject.toml").read_text()
        assert f"name = \"{package_name}\"" in pyproject_content
        assert template_vars["description"] in pyproject_content
        assert template_vars["author"] in pyproject_content
        
        # Template-specific checks
        if template_type == "fastapi_api":
            assert (output_dir / "src" / package_name / "main.py").exists()
            assert (output_dir / "src" / package_name / "api").exists()
        elif template_type == "htmx_tailwind_app":
            assert (output_dir / "src" / package_name / "main.py").exists()
            assert (output_dir / "src" / package_name / "templates").exists()
            assert (output_dir / "src" / package_name / "static").exists()
        elif template_type == "typer_cli":
            assert (output_dir / "src" / package_name / "cli.py").exists()
            assert (output_dir / "src" / package_name / "commands").exists()
    
    def test_generated_projects_python_validity(self, template_vars: Dict[str, str]):
        """Test that generated projects have valid Python syntax."""
        templates_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates")
        template_types = ["fastapi_api", "htmx_tailwind_app", "typer_cli"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_base = Path(temp_dir)
            
            for template_type in template_types:
                template_dir = templates_root / template_type
                output_dir = output_base / f"{template_vars['package_name']}_{template_type}"
                output_dir.mkdir()
                
                # Generate project
                self._generate_project_from_template(template_dir, output_dir, template_vars)
                
                # Check Python syntax validity
                python_files = list(output_dir.rglob("*.py"))
                for py_file in python_files:
                    content = py_file.read_text()
                    try:
                        compile(content, str(py_file), 'exec')
                    except SyntaxError as e:
                        pytest.fail(f"Invalid Python syntax in generated {template_type}/{py_file.name}: {e}")


class TestTemplateCompatibility:
    """Test template compatibility and dependencies."""
    
    @pytest.fixture
    def templates_root(self) -> Path:
        """Path to templates root directory."""
        return Path("/mnt/c/Users/andre/monorepo/src/packages/templates")
    
    def test_dependency_version_compatibility(self, templates_root: Path):
        """Test that template dependencies are compatible across templates."""
        template_types = ["fastapi_api", "htmx_tailwind_app", "typer_cli"]
        
        # Collect all dependencies and versions
        all_dependencies = {}
        
        for template_type in template_types:
            pyproject_path = templates_root / template_type / "pyproject.toml.template"
            if pyproject_path.exists():
                content = pyproject_path.read_text()
                
                # Extract dependency versions
                lines = content.split('\n')
                for line in lines:
                    if ' = "^' in line and not line.strip().startswith('#'):
                        # Parse dependency line
                        parts = line.split(' = "^')
                        if len(parts) == 2:
                            dep_name = parts[0].strip()
                            version = parts[1].split('"')[0]
                            
                            if dep_name in all_dependencies:
                                # Check version consistency
                                existing_version = all_dependencies[dep_name]
                                if existing_version != version:
                                    # Allow minor version differences for some packages
                                    if not self._are_versions_compatible(existing_version, version):
                                        pytest.fail(f"Incompatible versions for {dep_name}: {existing_version} vs {version}")
                            else:
                                all_dependencies[dep_name] = version
    
    def _are_versions_compatible(self, version1: str, version2: str) -> bool:
        """Check if two semantic versions are compatible."""
        def parse_version(v):
            return tuple(map(int, v.split('.')[:2]))  # Major.minor
        
        try:
            v1_major, v1_minor = parse_version(version1)
            v2_major, v2_minor = parse_version(version2)
            
            # Same major version is considered compatible
            return v1_major == v2_major
        except (ValueError, TypeError):
            return False
    
    def test_no_conflicting_dependencies(self, templates_root: Path):
        """Test that templates don't have conflicting dependencies."""
        template_types = ["fastapi_api", "htmx_tailwind_app", "typer_cli"]
        
        # Known incompatible combinations
        incompatible_pairs = [
            ("django", "fastapi"),  # Different web frameworks
            ("flask", "fastapi"),   # Different web frameworks
        ]
        
        for template_type in template_types:
            pyproject_path = templates_root / template_type / "pyproject.toml.template"
            if pyproject_path.exists():
                content = pyproject_path.read_text()
                
                for dep1, dep2 in incompatible_pairs:
                    has_dep1 = dep1 in content.lower()
                    has_dep2 = dep2 in content.lower()
                    assert not (has_dep1 and has_dep2), \
                        f"Template {template_type} has conflicting dependencies: {dep1} and {dep2}"


class TestTemplateDocumentation:
    """Test template documentation and examples."""
    
    @pytest.fixture
    def templates_root(self) -> Path:
        """Path to templates root directory."""
        return Path("/mnt/c/Users/andre/monorepo/src/packages/templates")
    
    def test_all_templates_have_readme(self, templates_root: Path):
        """Test that all templates have README documentation."""
        template_types = ["fastapi_api", "htmx_tailwind_app", "typer_cli"]
        
        for template_type in template_types:
            readme_path = templates_root / template_type / "README.md"
            assert readme_path.exists(), f"README.md missing for template: {template_type}"
            
            # Check README has content
            readme_content = readme_path.read_text()
            assert len(readme_content.strip()) > 0, f"README.md is empty for template: {template_type}"
            
            # Should mention the template type
            assert template_type.replace('_', ' ').lower() in readme_content.lower() or \
                   any(word in readme_content.lower() for word in template_type.split('_')), \
                   f"README.md doesn't describe template type: {template_type}"
    
    def test_template_usage_documentation(self, templates_root: Path):
        """Test that templates have usage documentation."""
        # Check if there's a general templates README
        general_readme = templates_root / "README.md"
        if general_readme.exists():
            content = general_readme.read_text()
            
            # Should document available templates
            template_types = ["fastapi", "htmx", "typer", "cli"]
            for template_type in template_types:
                assert template_type in content.lower(), \
                    f"Template type {template_type} not documented in general README"
    
    def test_template_test_documentation(self, templates_root: Path):
        """Test that template testing is documented."""
        test_readme = templates_root / "tests" / "README.md"
        if test_readme.exists():
            content = test_readme.read_text()
            
            # Should document test structure
            test_concepts = ["unit", "integration", "validation", "template"]
            for concept in test_concepts:
                assert concept in content.lower(), \
                    f"Test concept {concept} not documented in test README"