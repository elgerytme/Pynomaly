"""Unit tests for the scanner module."""

import pytest
from pathlib import Path
import tempfile
import os

from core.domain.services.scanner import Scanner, Import, ImportVisitor


class TestScanner:
    """Test the Scanner class."""
    
    def test_scan_simple_import(self):
        """Test scanning a file with simple imports."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import os
import sys
from pathlib import Path
from typing import List, Dict
""")
            f.flush()
            
            scanner = Scanner()
            result = scanner.scan_file(Path(f.name))
            
            assert len(result.imports) == 4
            assert any(imp.module == 'os' for imp in result.imports)
            assert any(imp.module == 'pathlib' and 'Path' in imp.names for imp in result.imports)
            
        os.unlink(f.name)
        
    def test_scan_cross_domain_import(self):
        """Test scanning imports that cross domains."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
from src.packages.ai.mlops import Model
from src.packages.finance.billing import Invoice
from src.packages.shared.utils import format_date
import src.packages.data.analytics.processor
""")
            f.flush()
            
            scanner = Scanner()
            result = scanner.scan_file(Path(f.name))
            
            assert len(result.imports) == 4
            
            # Check specific imports
            finance_import = next((imp for imp in result.imports 
                                 if 'finance' in imp.module), None)
            assert finance_import is not None
            assert finance_import.module == 'src.packages.finance.billing'
            assert 'Invoice' in finance_import.names
            
        os.unlink(f.name)
        
    def test_scan_relative_imports(self):
        """Test scanning relative imports."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
from . import sibling_module
from ..parent import something
from ...grandparent.module import Class
""")
            f.flush()
            
            scanner = Scanner()
            result = scanner.scan_file(Path(f.name))
            
            assert len(result.imports) == 3
            
            # Check relative import levels
            assert any(imp.is_relative and imp.level == 1 for imp in result.imports)
            assert any(imp.is_relative and imp.level == 2 for imp in result.imports)
            assert any(imp.is_relative and imp.level == 3 for imp in result.imports)
            
        os.unlink(f.name)
        
    def test_scan_string_references(self):
        """Test detection of module-like strings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
config = {
    'service': 'finance.payment_processor',
    'module': 'ai.mlops.trainer',
    'not_a_module': 'hello world',
    'url': 'https://example.com'
}

import_string('data.analytics.engine')
""")
            f.flush()
            
            scanner = Scanner()
            result = scanner.scan_file(Path(f.name))
            
            # Should detect module-like strings
            assert len(result.string_references) >= 3
            module_strings = [ref.value for ref in result.string_references]
            assert 'finance.payment_processor' in module_strings
            assert 'ai.mlops.trainer' in module_strings
            assert 'data.analytics.engine' in module_strings
            
            # Should not detect non-module strings
            assert 'hello world' not in module_strings
            assert 'https://example.com' not in module_strings
            
        os.unlink(f.name)
        
    def test_scan_directory(self):
        """Test scanning a directory with multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / 'file1.py').write_text("""
import os
from src.packages.ai.mlops import Model
""")
            (Path(tmpdir) / 'file2.py').write_text("""
import sys
from src.packages.finance.billing import Invoice
""")
            
            # Create subdirectory
            subdir = Path(tmpdir) / 'subdir'
            subdir.mkdir()
            (subdir / 'file3.py').write_text("""
from pathlib import Path
import json
""")
            
            scanner = Scanner()
            result = scanner.scan_directory(Path(tmpdir))
            
            # Should find imports from all files
            assert len(result.imports) >= 6
            modules = [imp.module for imp in result.imports]
            assert 'os' in modules
            assert 'sys' in modules
            assert 'json' in modules
            assert 'src.packages.ai.mlops' in modules
            assert 'src.packages.finance.billing' in modules
            
    def test_ignore_patterns(self):
        """Test that ignore patterns work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files that should be ignored
            test_dir = Path(tmpdir) / 'tests'
            test_dir.mkdir()
            (test_dir / 'test_something.py').write_text("import forbidden")
            
            cache_dir = Path(tmpdir) / '__pycache__'
            cache_dir.mkdir()
            (cache_dir / 'module.pyc').write_text("import forbidden")
            
            # Create file that should be scanned
            (Path(tmpdir) / 'main.py').write_text("import allowed")
            
            scanner = Scanner()
            result = scanner.scan_directory(Path(tmpdir))
            
            # Should only find the allowed import
            assert len(result.imports) == 1
            assert result.imports[0].module == 'allowed'
            
    def test_syntax_error_handling(self):
        """Test handling of files with syntax errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import os
this is not valid python code
import sys
""")
            f.flush()
            
            scanner = Scanner()
            result = scanner.scan_file(Path(f.name))
            
            # Should have recorded an error
            assert len(result.errors) == 1
            assert 'SyntaxError' in result.errors[0]['type']
            assert f.name in result.errors[0]['file']
            
        os.unlink(f.name)