"""
Comprehensive tests for HTMX/Tailwind template validation.

Tests template integrity, variable substitution, frontend component structure,
and generated project architecture for the HTMX/Tailwind template.
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest
from unittest.mock import patch, MagicMock

from test_utilities.factories import TestDataFactory
from test_utilities.fixtures import async_test


class TestHTMXTemplateValidation:
    """Test HTMX/Tailwind template structure and validation."""
    
    @pytest.fixture
    def template_vars(self) -> Dict[str, str]:
        """Template variables for testing."""
        return {
            "package_name": "test_htmx_app",
            "description": "Test HTMX/Tailwind Application",
            "author": "Test Author <test@example.com>"
        }
    
    @pytest.fixture
    def template_root(self) -> Path:
        """Path to HTMX template root."""
        return Path("/mnt/c/Users/andre/monorepo/src/packages/templates/htmx_tailwind_app")
    
    def test_template_directory_structure(self, template_root: Path):
        """Test that all required template files exist."""
        required_files = [
            "pyproject.toml.template",
            "Dockerfile.template", 
            "README.md",
            "src/{{package_name}}/__init__.py.template",
            "src/{{package_name}}/main.py.template",
            "src/{{package_name}}/api/htmx.py.template",
            "src/{{package_name}}/api/pages.py.template",
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
    
    def test_frontend_template_structure(self, template_root: Path):
        """Test frontend-specific template files exist."""
        frontend_files = [
            "src/{{package_name}}/static/css/style.css",
            "src/{{package_name}}/static/js/app.js",
            "src/{{package_name}}/templates/layouts/base.html",
            "src/{{package_name}}/templates/pages/home.html",
            "src/{{package_name}}/templates/pages/dashboard.html",
            "src/{{package_name}}/templates/pages/items.html",
            "src/{{package_name}}/templates/components/item_card.html",
            "src/{{package_name}}/templates/components/item_list.html",
            "src/{{package_name}}/templates/components/item_edit_form.html",
            "src/{{package_name}}/templates/components/notification.html"
        ]
        
        for file_path in frontend_files:
            full_path = template_root / file_path
            assert full_path.exists(), f"Required frontend file missing: {file_path}"
    
    def test_pyproject_template_structure(self, template_root: Path):
        """Test pyproject.toml template has correct HTMX dependencies."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # Check template variables
        assert "{{package_name}}" in content
        assert "{{description}}" in content
        assert "{{author}}" in content
        
        # Check HTMX-specific dependencies
        htmx_deps = [
            "fastapi = \"^0.109.0\"",
            "jinja2 = \"^3.1.3\"",  # Template engine for HTMX
            "python-multipart = \"^0.0.6\"",  # Form handling
            "pydantic = \"^2.5.0\"",
            "sqlalchemy = \"^2.0.25\"",
            "uvicorn"
        ]
        
        for dep in htmx_deps:
            assert dep in content, f"Missing HTMX dependency: {dep}"
    
    def test_main_template_htmx_setup(self, template_root: Path):
        """Test main.py template has correct HTMX/Jinja2 setup."""
        main_path = template_root / "src/{{package_name}}/main.py.template"
        content = main_path.read_text()
        
        # Check FastAPI imports and setup
        assert "from fastapi import FastAPI" in content
        assert "app = FastAPI(" in content
        
        # Check Jinja2 templates setup
        assert "Jinja2Templates" in content or "templates" in content
        assert "static" in content or "StaticFiles" in content
        
        # Check template and static mounting
        assert "mount" in content or "StaticFiles" in content
    
    def test_htmx_api_endpoints(self, template_root: Path):
        """Test HTMX-specific API endpoints structure."""
        htmx_api_path = template_root / "src/{{package_name}}/api/htmx.py.template"
        content = htmx_api_path.read_text()
        
        # Check HTMX-specific imports
        assert "from fastapi import APIRouter" in content
        assert "Request" in content or "HTMLResponse" in content
        
        # Check HTMX response handling
        assert "templates.TemplateResponse" in content or "TemplateResponse" in content
        assert "router = APIRouter()" in content
        
        # Check HTMX endpoint patterns
        assert "@router" in content
        assert "request: Request" in content or "Request" in content
    
    def test_pages_api_structure(self, template_root: Path):
        """Test pages API endpoints for serving HTML."""
        pages_path = template_root / "src/{{package_name}}/api/pages.py.template"
        content = pages_path.read_text()
        
        # Check page serving setup
        assert "from fastapi import APIRouter" in content
        assert "Request" in content
        assert "TemplateResponse" in content or "templates.TemplateResponse" in content
        
        # Check router setup
        assert "router = APIRouter()" in content
        assert "@router.get" in content
    
    def test_base_html_template_structure(self, template_root: Path):
        """Test base HTML template has correct structure."""
        base_path = template_root / "src/{{package_name}}/templates/layouts/base.html"
        content = base_path.read_text()
        
        # Check HTML5 structure
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "<head>" in content
        assert "<body>" in content
        
        # Check Tailwind CSS inclusion
        assert "tailwindcss" in content or "cdn.tailwindcss.com" in content
        
        # Check HTMX inclusion
        assert "htmx" in content or "unpkg.com/htmx.org" in content
        
        # Check template blocks
        assert "{% block" in content or "{{ block" in content or "{%" in content
    
    def test_component_templates_structure(self, template_root: Path):
        """Test component templates have correct HTMX structure."""
        components = [
            "item_card.html",
            "item_list.html", 
            "item_edit_form.html",
            "notification.html"
        ]
        
        component_dir = template_root / "src/{{package_name}}/templates/components"
        
        for component in components:
            component_path = component_dir / component
            content = component_path.read_text()
            
            # Check HTMX attributes usage
            htmx_attrs = ["hx-get", "hx-post", "hx-put", "hx-delete", "hx-target", "hx-swap"]
            has_htmx = any(attr in content for attr in htmx_attrs)
            
            # Item edit form should definitely have HTMX
            if component == "item_edit_form.html":
                assert has_htmx, f"Component {component} should use HTMX attributes"
    
    def test_page_templates_structure(self, template_root: Path):
        """Test page templates have correct structure."""
        pages = [
            "home.html",
            "dashboard.html",
            "items.html",
            "item_detail.html",
            "about.html",
            "404.html",
            "500.html"
        ]
        
        pages_dir = template_root / "src/{{package_name}}/templates/pages"
        
        for page in pages:
            page_path = pages_dir / page
            content = page_path.read_text()
            
            # Check template inheritance
            assert "{% extends" in content or "extends" in content
            
            # Check block usage
            assert "{% block" in content or "block" in content
    
    def test_static_css_structure(self, template_root: Path):
        """Test static CSS file structure."""
        css_path = template_root / "src/{{package_name}}/static/css/style.css"
        content = css_path.read_text()
        
        # Check for Tailwind compatibility
        # Should either have Tailwind imports or custom styles
        assert len(content.strip()) > 0, "CSS file should not be empty"
        
        # Check for common CSS patterns
        css_patterns = ["@", ".", "#", "{", "}", "/*"]
        has_css = any(pattern in content for pattern in css_patterns)
        assert has_css, "CSS file should contain valid CSS"
    
    def test_static_js_structure(self, template_root: Path):
        """Test static JavaScript file structure."""
        js_path = template_root / "src/{{package_name}}/static/js/app.js"
        content = js_path.read_text()
        
        # Should have some JavaScript content
        assert len(content.strip()) > 0, "JavaScript file should not be empty"
        
        # Check for HTMX-related JavaScript patterns
        js_patterns = ["function", "document", "htmx", "addEventListener", "//", "/*"]
        has_js = any(pattern in content for pattern in js_patterns)
        assert has_js, "JavaScript file should contain valid JavaScript"
    
    def test_template_variable_substitution(self, template_root: Path, template_vars: Dict[str, str]):
        """Test template variable substitution in all files."""
        template_files = list(template_root.rglob("*.template"))
        
        for template_file in template_files:
            if template_file.is_file():
                content = template_file.read_text()
                
                # Substitute variables
                substituted = content
                for var, value in template_vars.items():
                    substituted = substituted.replace(f"{{{{{var}}}}}", value)
                
                # Verify substitution worked
                if "{{package_name}}" in content:
                    assert template_vars["package_name"] in substituted
                    assert "{{" not in substituted.count("{{package_name}}"), f"Template variables not fully substituted in {template_file}"
    
    def test_dockerfile_htmx_setup(self, template_root: Path):
        """Test Dockerfile template has correct HTMX app setup."""
        dockerfile_path = template_root / "Dockerfile.template"
        content = dockerfile_path.read_text()
        
        # Check base Python image
        assert "FROM python:" in content
        assert "WORKDIR" in content
        assert "COPY" in content
        
        # Check static files handling
        assert "static" in content or "templates" in content or "{{package_name}}" in content
        
        # Check application startup
        assert "CMD" in content or "ENTRYPOINT" in content
        assert "uvicorn" in content or "python" in content


class TestHTMXTemplateDependencies:
    """Test HTMX template dependency validation."""
    
    @pytest.fixture
    def template_root(self) -> Path:
        """Path to HTMX template root."""
        return Path("/mnt/c/Users/andre/monorepo/src/packages/templates/htmx_tailwind_app")
    
    def test_frontend_dependencies(self, template_root: Path):
        """Test frontend-specific dependencies are included."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # Check frontend-specific dependencies
        frontend_deps = [
            "jinja2",           # Template engine
            "python-multipart", # Form handling
            "uvicorn"          # ASGI server
        ]
        
        for dep in frontend_deps:
            assert dep in content, f"Missing frontend dependency: {dep}"
    
    def test_development_dependencies_for_frontend(self, template_root: Path):
        """Test development dependencies suitable for frontend development."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # Should include file watching for development
        assert "watchfiles" in content, "Missing watchfiles for development"
        
        # Standard dev dependencies
        dev_deps = [
            "pytest",
            "pytest-asyncio", 
            "black",
            "ruff",
            "mypy"
        ]
        
        for dep in dev_deps:
            assert dep in content, f"Missing development dependency: {dep}"
    
    def test_template_engine_compatibility(self, template_root: Path):
        """Test Jinja2 template engine compatibility."""
        pyproject_path = template_root / "pyproject.toml.template"
        content = pyproject_path.read_text()
        
        # Check Jinja2 version compatibility with FastAPI
        if "jinja2 = \"^3.1.3\"" in content:
            assert "fastapi = \"^0.109.0\"" in content, "Jinja2 version should be compatible with FastAPI"


class TestHTMXTemplateGeneration:
    """Test HTMX template generation and output validation."""
    
    @pytest.fixture
    def template_vars(self) -> Dict[str, str]:
        """Template variables for testing."""
        return {
            "package_name": "my_htmx_app",
            "description": "My HTMX Application", 
            "author": "Jane Doe <jane@example.com>"
        }
    
    def test_generate_complete_htmx_project(self, template_vars: Dict[str, str]):
        """Test generating a complete HTMX project from template."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/htmx_tailwind_app")
        
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
            
            # Copy static files (non-template)
            static_files = [
                "src/{{package_name}}/static/css/style.css",
                "src/{{package_name}}/static/js/app.js"
            ]
            
            for static_file in static_files:
                src_path = template_root / static_file
                if src_path.exists():
                    dst_path_str = str(output_dir / static_file).replace('{{package_name}}', template_vars["package_name"])
                    dst_path = Path(dst_path_str)
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    dst_path.write_text(src_path.read_text())
            
            # Verify generated HTMX project structure
            expected_files = [
                f"pyproject.toml",
                f"src/{template_vars['package_name']}/main.py",
                f"src/{template_vars['package_name']}/api/htmx.py",
                f"src/{template_vars['package_name']}/api/pages.py",
                f"src/{template_vars['package_name']}/static/css/style.css",
                f"src/{template_vars['package_name']}/static/js/app.js",
                f"src/{template_vars['package_name']}/templates/layouts/base.html"
            ]
            
            for expected_file in expected_files:
                file_path = output_dir / expected_file
                assert file_path.exists(), f"Generated HTMX file missing: {expected_file}"
    
    def test_generated_template_responses(self, template_vars: Dict[str, str]):
        """Test that generated templates have correct response structure."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/htmx_tailwind_app")
        
        # Test HTMX API template
        htmx_template = template_root / "src/{{package_name}}/api/htmx.py.template"
        content = htmx_template.read_text()
        
        # Substitute variables
        for var, value in template_vars.items():
            content = content.replace(f"{{{{{var}}}}}", value)
        
        # Check that template responses are properly configured
        assert "TemplateResponse" in content
        assert "request" in content
        
        # Should reference the correct package name in imports
        if f"from {template_vars['package_name']}" in content:
            # Validate import structure
            assert "." in content  # Should have relative imports
    
    def test_generated_static_file_references(self, template_vars: Dict[str, str]):
        """Test that generated templates correctly reference static files."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/htmx_tailwind_app")
        base_template = template_root / "src/{{package_name}}/templates/layouts/base.html"
        
        if base_template.exists():
            content = base_template.read_text()
            
            # Check static file references
            if "/static/" in content:
                # Should reference CSS and JS files
                assert ".css" in content or ".js" in content
            
            # Check HTMX and Tailwind CDN references
            assert "htmx" in content or "tailwind" in content
    
    def test_generated_htmx_functionality(self, template_vars: Dict[str, str]):
        """Test that generated HTMX components have proper functionality."""
        template_root = Path("/mnt/c/Users/andre/monorepo/src/packages/templates/htmx_tailwind_app")
        
        # Test item edit form component
        form_path = template_root / "src/{{package_name}}/templates/components/item_edit_form.html"
        if form_path.exists():
            content = form_path.read_text()
            
            # Should have HTMX attributes for form submission
            htmx_form_attrs = ["hx-post", "hx-put", "hx-target", "hx-swap"]
            has_htmx_form = any(attr in content for attr in htmx_form_attrs)
            
            if "<form" in content:
                assert has_htmx_form, "Form component should use HTMX attributes"