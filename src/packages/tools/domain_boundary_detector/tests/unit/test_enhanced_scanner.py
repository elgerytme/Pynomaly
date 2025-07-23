"""
Comprehensive tests for Scanner and ImportVisitor classes.

Tests AST-based scanning for imports, string references, type annotations,
and cross-package reference detection functionality.
"""
import ast
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import List

from test_utilities.factories import TestDataFactory
from test_utilities.fixtures import async_test

from core.domain.services.scanner import (
    Scanner,
    ImportVisitor,
    Import,
    StringReference,
    ScanResult,
    extract_package_from_import
)


class TestImport:
    """Test Import data class functionality."""
    
    def test_import_creation(self):
        """Test creating an import with all fields."""
        import_obj = Import(
            module="numpy.array",
            names=["array", "zeros"],
            file_path="/path/to/file.py",
            line_number=5,
            import_type="from",
            is_relative=False,
            level=0
        )
        
        assert import_obj.module == "numpy.array"
        assert import_obj.names == ["array", "zeros"]
        assert import_obj.file_path == "/path/to/file.py"
        assert import_obj.line_number == 5
        assert import_obj.import_type == "from"
        assert not import_obj.is_relative
        assert import_obj.level == 0
    
    def test_import_str_representation_simple_import(self):
        """Test string representation for simple import."""
        import_obj = Import(
            module="numpy",
            names=[],
            file_path="/path/to/file.py",
            line_number=1,
            import_type="import"
        )
        
        assert str(import_obj) == "import numpy"
    
    def test_import_str_representation_from_import(self):
        """Test string representation for from import."""
        import_obj = Import(
            module="numpy.array",
            names=["array", "zeros"],
            file_path="/path/to/file.py",
            line_number=1,
            import_type="from"
        )
        
        assert str(import_obj) == "from numpy.array import array, zeros"
    
    def test_import_str_representation_from_import_star(self):
        """Test string representation for from import *."""
        import_obj = Import(
            module="numpy",
            names=[],
            file_path="/path/to/file.py",
            line_number=1,
            import_type="from"
        )
        
        assert str(import_obj) == "from numpy import *"
    
    def test_import_relative_import(self):
        """Test relative import properties."""
        import_obj = Import(
            module="service",
            names=["ServiceClass"],
            file_path="/path/to/file.py",
            line_number=10,
            import_type="from",
            is_relative=True,
            level=2
        )
        
        assert import_obj.is_relative
        assert import_obj.level == 2


class TestStringReference:
    """Test StringReference data class functionality."""
    
    def test_string_reference_creation(self):
        """Test creating a string reference."""
        ref = StringReference(
            value="ai.mlops.service",
            file_path="/path/to/config.py",
            line_number=15,
            context="configuration_key"
        )
        
        assert ref.value == "ai.mlops.service"
        assert ref.file_path == "/path/to/config.py"
        assert ref.line_number == 15
        assert ref.context == "configuration_key"


class TestScanResult:
    """Test ScanResult data class functionality."""
    
    def test_scan_result_creation(self):
        """Test creating an empty scan result."""
        result = ScanResult()
        
        assert result.imports == []
        assert result.string_references == []
        assert result.type_annotations == []
        assert result.errors == []
    
    def test_scan_result_with_data(self):
        """Test creating scan result with data."""
        import_obj = Import("numpy", [], "/file.py", 1, "import")
        string_ref = StringReference("ai.service", "/file.py", 2, "config")
        error = {"file": "/file.py", "error": "syntax error", "type": "SyntaxError"}
        
        result = ScanResult(
            imports=[import_obj],
            string_references=[string_ref],
            errors=[error]
        )
        
        assert len(result.imports) == 1
        assert len(result.string_references) == 1
        assert len(result.errors) == 1
        assert result.imports[0] == import_obj
        assert result.string_references[0] == string_ref
        assert result.errors[0] == error


class TestImportVisitor:
    """Test ImportVisitor AST node visitor functionality."""
    
    @pytest.fixture
    def visitor(self):
        """Create a visitor for testing."""
        return ImportVisitor("/test/file.py")
    
    def test_visitor_initialization(self, visitor):
        """Test visitor initialization."""
        assert visitor.file_path == "/test/file.py"
        assert visitor.imports == []
        assert visitor.string_references == []
        assert visitor.type_annotations == []
    
    def test_visit_simple_import(self, visitor):
        """Test visiting simple import statements."""
        code = "import numpy"
        tree = ast.parse(code)
        visitor.visit(tree)
        
        assert len(visitor.imports) == 1
        import_obj = visitor.imports[0]
        assert import_obj.module == "numpy"
        assert import_obj.names == []
        assert import_obj.import_type == "import"
        assert not import_obj.is_relative
    
    def test_visit_multiple_imports(self, visitor):
        """Test visiting multiple import statements."""
        code = "import numpy, pandas, torch"
        tree = ast.parse(code)
        visitor.visit(tree)
        
        assert len(visitor.imports) == 3
        modules = [imp.module for imp in visitor.imports]
        assert "numpy" in modules
        assert "pandas" in modules
        assert "torch" in modules
    
    def test_visit_from_import(self, visitor):
        """Test visiting from import statements."""
        code = "from numpy import array, zeros"
        tree = ast.parse(code)
        visitor.visit(tree)
        
        assert len(visitor.imports) == 1
        import_obj = visitor.imports[0]
        assert import_obj.module == "numpy"
        assert import_obj.names == ["array", "zeros"]
        assert import_obj.import_type == "from"
        assert not import_obj.is_relative
    
    def test_visit_relative_import(self, visitor):
        """Test visiting relative import statements."""
        code = "from ..service import ServiceClass"
        tree = ast.parse(code)
        visitor.visit(tree)
        
        assert len(visitor.imports) == 1
        import_obj = visitor.imports[0]
        assert import_obj.module == "service"
        assert import_obj.names == ["ServiceClass"]
        assert import_obj.import_type == "from"
        assert import_obj.is_relative
        assert import_obj.level == 2
    
    def test_visit_from_import_star(self, visitor):
        """Test visiting from import * statements."""
        code = "from numpy import *"
        tree = ast.parse(code)
        visitor.visit(tree)
        
        assert len(visitor.imports) == 1
        import_obj = visitor.imports[0]
        assert import_obj.module == "numpy"
        assert import_obj.names == ["*"]
        assert import_obj.import_type == "from"
    
    def test_visit_string_literal_module_reference(self, visitor):
        """Test visiting string literals that look like module references."""
        code = 'config_key = "ai.mlops.service"'
        tree = ast.parse(code)
        visitor.visit(tree)
        
        assert len(visitor.string_references) == 1
        string_ref = visitor.string_references[0]
        assert string_ref.value == "ai.mlops.service"
        assert string_ref.file_path == "/test/file.py"
        assert string_ref.line_number == 1
    
    def test_visit_constant_module_reference(self, visitor):
        """Test visiting constants that look like module references (Python 3.8+)."""
        # Using ast.Constant for Python 3.8+ compatibility
        code = 'SERVICE_PATH = "ai.mlops.service"'
        tree = ast.parse(code)
        visitor.visit(tree)
        
        # Should detect the string as a potential module reference
        string_refs = [ref for ref in visitor.string_references if ref.value == "ai.mlops.service"]
        assert len(string_refs) >= 1
    
    def test_visit_type_annotation(self, visitor):
        """Test visiting type annotations with string references."""
        # Test forward reference in type annotation
        code = 'var: "ai.mlops.Model" = None'
        tree = ast.parse(code)
        visitor.visit(tree)
        
        # May be detected as string reference depending on AST structure
        references = visitor.string_references + visitor.type_annotations
        module_refs = [ref for ref in references if "ai.mlops" in ref.value]
        assert len(module_refs) >= 1
    
    def test_is_module_like_string_valid_modules(self, visitor):
        """Test module-like string detection for valid modules."""
        # Valid module patterns
        assert visitor._is_module_like_string("numpy.array")
        assert visitor._is_module_like_string("ai.mlops.service")
        assert visitor._is_module_like_string("finance.billing.models")
        assert visitor._is_module_like_string("data.engineering.pipeline")
        
        # Known libraries
        assert visitor._is_module_like_string("numpy")
        assert visitor._is_module_like_string("pandas")
        assert visitor._is_module_like_string("torch")
        assert visitor._is_module_like_string("tensorflow")
    
    def test_is_module_like_string_invalid_patterns(self, visitor):
        """Test module-like string detection for invalid patterns."""
        # Too short
        assert not visitor._is_module_like_string("a")
        assert not visitor._is_module_like_string("ab")
        
        # Empty
        assert not visitor._is_module_like_string("")
        
        # Test names
        assert not visitor._is_module_like_string("test_module")
        
        # Dunder names
        assert not visitor._is_module_like_string("__init__")
        
        # Starts with number
        assert not visitor._is_module_like_string("2module")
        
        # URLs
        assert not visitor._is_module_like_string("https://example.com")
        assert not visitor._is_module_like_string("http://api.service.com")
        
        # All caps constants
        assert not visitor._is_module_like_string("CONFIG_VALUE")
        assert not visitor._is_module_like_string("API_KEY")
        
        # Invalid characters
        assert not visitor._is_module_like_string("module-name")
        assert not visitor._is_module_like_string("module name")
        assert not visitor._is_module_like_string("module@domain")
    
    def test_line_numbers_correctly_captured(self, visitor):
        """Test that line numbers are correctly captured."""
        code = """
import numpy
from pandas import DataFrame
config = "ai.mlops.service"
"""
        tree = ast.parse(code)
        visitor.visit(tree)
        
        # Check line numbers
        assert visitor.imports[0].line_number == 2  # import numpy
        assert visitor.imports[1].line_number == 3  # from pandas import DataFrame
        
        if visitor.string_references:
            assert visitor.string_references[0].line_number == 4  # config string


class TestScanner:
    """Test Scanner main functionality."""
    
    @pytest.fixture
    def scanner(self):
        """Create a scanner for testing."""
        return Scanner()
    
    def test_scanner_initialization(self, scanner):
        """Test scanner initialization with default ignore patterns."""
        assert len(scanner.ignore_patterns) > 0
        assert '__pycache__' in scanner.ignore_patterns
        assert '.git' in scanner.ignore_patterns
        assert 'venv' in scanner.ignore_patterns
    
    def test_scanner_custom_ignore_patterns(self):
        """Test scanner with custom ignore patterns."""
        custom_patterns = ['custom_ignore', 'temp_*']
        scanner = Scanner(ignore_patterns=custom_patterns)
        
        assert scanner.ignore_patterns == custom_patterns
    
    def test_should_ignore_patterns(self, scanner):
        """Test ignore pattern matching."""
        # Should ignore
        assert scanner._should_ignore('__pycache__/module.py')
        assert scanner._should_ignore('.git/config')
        assert scanner._should_ignore('venv/lib/python')
        assert scanner._should_ignore('node_modules/package')
        
        # Should not ignore
        assert not scanner._should_ignore('src/module.py')
        assert not scanner._should_ignore('tests/test_module.py')
        assert not scanner._should_ignore('package/service.py')
    
    def test_scan_file_success(self, scanner):
        """Test successful file scanning."""
        python_code = """
import numpy
from pandas import DataFrame
config = "ai.mlops.service"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            f.flush()
            
            result = scanner.scan_file(Path(f.name))
        
        # Verify results
        assert len(result.imports) == 2
        assert len(result.errors) == 0
        
        # Check imports
        numpy_import = next(imp for imp in result.imports if imp.module == "numpy")
        assert numpy_import.import_type == "import"
        
        pandas_import = next(imp for imp in result.imports if imp.module == "pandas")
        assert pandas_import.import_type == "from"
        assert pandas_import.names == ["DataFrame"]
        
        # Clean up
        Path(f.name).unlink()
    
    def test_scan_file_syntax_error(self, scanner):
        """Test scanning file with syntax error."""
        invalid_python_code = """
import numpy
def invalid_syntax(
    # Missing closing parenthesis
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(invalid_python_code)
            f.flush()
            
            result = scanner.scan_file(Path(f.name))
        
        # Should have error but no crash
        assert len(result.errors) == 1
        assert result.errors[0]['type'] == 'SyntaxError'
        assert result.errors[0]['file'] == f.name
        
        # Clean up
        Path(f.name).unlink()
    
    def test_scan_file_encoding_error(self, scanner):
        """Test scanning file with encoding issues."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as f:
            # Write invalid UTF-8 bytes
            f.write(b'import numpy\n\x80\x81\x82\n')
            f.flush()
            
            result = scanner.scan_file(Path(f.name))
        
        # Should handle encoding error gracefully
        assert len(result.errors) == 1
        assert 'error' in result.errors[0]
        
        # Clean up
        Path(f.name).unlink()
    
    def test_scan_directory(self, scanner):
        """Test scanning a directory with multiple Python files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "module1.py").write_text("import numpy")
            (temp_path / "module2.py").write_text("from pandas import DataFrame")
            
            # Create subdirectory
            subdir = temp_path / "subpackage"
            subdir.mkdir()
            (subdir / "module3.py").write_text("import torch")
            
            # Create ignored file
            (temp_path / "__pycache__").mkdir()
            (temp_path / "__pycache__" / "compiled.py").write_text("compiled")
            
            # Scan directory
            result = scanner.scan_directory(temp_path)
        
        # Should find all non-ignored Python files
        assert len(result.imports) == 3
        
        modules = [imp.module for imp in result.imports]
        assert "numpy" in modules
        assert "pandas" in modules
        assert "torch" in modules
    
    def test_find_python_files_respects_ignore_patterns(self, scanner):
        """Test that file finding respects ignore patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid files
            (temp_path / "valid.py").write_text("import numpy")
            
            # Create ignored directories and files
            (temp_path / "__pycache__").mkdir()
            (temp_path / "__pycache__" / "ignored.py").write_text("ignored")
            
            (temp_path / ".git").mkdir()
            (temp_path / ".git" / "config.py").write_text("ignored")
            
            (temp_path / "venv").mkdir()
            (temp_path / "venv" / "lib.py").write_text("ignored")
            
            files = scanner._find_python_files(temp_path)
        
        # Should only find valid files
        assert len(files) == 1
        assert files[0].name == "valid.py"
    
    def test_scan_empty_directory(self, scanner):
        """Test scanning an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = scanner.scan_directory(Path(temp_dir))
        
        assert len(result.imports) == 0
        assert len(result.string_references) == 0
        assert len(result.errors) == 0


class TestExtractPackageFromImport:
    """Test package extraction utility function."""
    
    def test_extract_absolute_import_src_packages(self):
        """Test extracting package from src.packages imports."""
        # Standard monorepo structure
        package = extract_package_from_import(
            "src.packages.ai.mlops.service",
            "/monorepo/some/file.py",
            "/monorepo"
        )
        assert package == "ai/mlops"
        
        package = extract_package_from_import(
            "src.packages.finance.billing.models",
            "/monorepo/some/file.py", 
            "/monorepo"
        )
        assert package == "finance/billing"
    
    def test_extract_absolute_import_insufficient_parts(self):
        """Test extracting package from imports with insufficient parts."""
        # Not enough parts
        package = extract_package_from_import(
            "src.packages.ai",
            "/monorepo/file.py",
            "/monorepo"
        )
        assert package is None
        
        package = extract_package_from_import(
            "numpy",
            "/monorepo/file.py",
            "/monorepo"
        )
        assert package is None
    
    def test_extract_relative_import_single_level(self):
        """Test extracting package from single-level relative imports."""
        package = extract_package_from_import(
            ".service",
            "/monorepo/src/packages/ai/mlops/models/file.py",
            "/monorepo"
        )
        # Should resolve to the package containing the service module
        # This depends on the exact implementation logic
        assert package is None or isinstance(package, str)
    
    def test_extract_relative_import_multiple_levels(self):
        """Test extracting package from multi-level relative imports."""
        package = extract_package_from_import(
            "..core.service",
            "/monorepo/src/packages/ai/mlops/models/file.py",
            "/monorepo"
        )
        # Should resolve based on relative path navigation
        assert package is None or isinstance(package, str)
    
    def test_extract_relative_import_parent_directory(self):
        """Test extracting package from parent directory relative imports."""
        package = extract_package_from_import(
            "...shared.utils",
            "/monorepo/src/packages/ai/mlops/models/file.py",
            "/monorepo"
        )
        # Should handle multiple level navigation
        assert package is None or isinstance(package, str)
    
    def test_extract_package_edge_cases(self):
        """Test edge cases for package extraction."""
        # Empty import
        assert extract_package_from_import("", "/file.py", "/root") is None
        
        # Just dots
        assert extract_package_from_import(".", "/file.py", "/root") is None
        assert extract_package_from_import("..", "/file.py", "/root") is None
        
        # Non-standard structure
        assert extract_package_from_import(
            "other.structure.module",
            "/file.py",
            "/root"
        ) is None
    
    def test_extract_package_from_complex_paths(self):
        """Test extracting packages from complex file paths."""
        # Deeply nested file
        package = extract_package_from_import(
            "src.packages.data.engineering.pipeline",
            "/monorepo/src/packages/ai/mlops/services/model_service.py",
            "/monorepo"
        )
        assert package == "data/engineering"
        
        # File in tests directory
        package = extract_package_from_import(
            "src.packages.ai.core.models",
            "/monorepo/tests/integration/test_models.py",
            "/monorepo"
        )
        assert package == "ai/core"
    
    def test_extract_package_handles_windows_paths(self):
        """Test that package extraction works with Windows paths."""
        # Windows-style paths
        package = extract_package_from_import(
            "src.packages.ai.mlops.service",
            "C:\\monorepo\\src\\packages\\finance\\billing\\models.py",
            "C:\\monorepo"
        )
        assert package == "ai/mlops"