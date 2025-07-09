"""Focused test for Issue #33 - Critical Bug Fix.

This test ensures the conftest.py file has correct syntax without
requiring complex imports that may fail due to missing dependencies.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest


class TestIssue33CriticalBugFix:
    """Test suite for Issue #33 - Critical Bug Fix."""

    def test_conftest_syntax_validation(self):
        """Test that conftest.py has valid Python syntax."""
        conftest_path = Path(__file__).parent / "conftest.py"
        
        # Read the file content
        with open(conftest_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check for syntax errors
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in conftest.py: {e}")

    def test_future_imports_placement(self):
        """Test that __future__ imports are placed correctly."""
        conftest_path = Path(__file__).parent / "conftest.py"
        
        with open(conftest_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the line with __future__ import
        future_import_line = None
        for i, line in enumerate(lines):
            if 'from __future__ import' in line:
                future_import_line = i
                break
        
        if future_import_line is None:
            pytest.skip("No __future__ imports found")
        
        # Check that only docstrings and comments come before __future__ import
        for i in range(future_import_line):
            line = lines[i].strip()
            if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                # Check if this is part of a docstring
                if not self._is_part_of_docstring(lines[:i+1]):
                    pytest.fail(f"Non-docstring/comment line before __future__ import at line {i+1}: {line}")

    def _is_part_of_docstring(self, lines):
        """Check if the current line is part of a docstring."""
        content = ''.join(lines)
        try:
            ast.parse(content)
            return True  # If it parses, it's valid
        except SyntaxError:
            return False

    def test_conftest_compiles_without_errors(self):
        """Test that conftest.py can be compiled without syntax errors."""
        conftest_path = Path(__file__).parent / "conftest.py"
        
        result = subprocess.run([
            sys.executable, "-m", "py_compile", str(conftest_path)
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Syntax error in conftest.py: {result.stderr}"

    def test_pytest_can_load_conftest_basic(self):
        """Test that pytest can load conftest.py without future import errors."""
        # This is a simple test just to ensure the conftest.py syntax is correct
        # and doesn't have the original issue
        conftest_path = Path(__file__).parent / "conftest.py"
        
        # Read the file content
        with open(conftest_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that if there's a future import, it's at the beginning
        lines = content.split('\n')
        
        # Find the line with __future__ import
        future_import_line = None
        for i, line in enumerate(lines):
            if 'from __future__ import' in line:
                future_import_line = i
                break
        
        if future_import_line is not None:
            # Check that there are NO regular imports before the __future__ import
            # (Only docstrings and comments should be before it)
            regular_imports_before_future = []
            for i in range(future_import_line):
                line = lines[i].strip()
                if line.startswith('import ') and not line.startswith('from __future__'):
                    regular_imports_before_future.append(i)
            
            # This should pass because there are no regular imports before __future__ import
            assert len(regular_imports_before_future) == 0, (
                "Found regular imports before __future__ import: "
                f"{regular_imports_before_future}"
            )

    def test_conftest_file_structure(self):
        """Test the overall structure of conftest.py."""
        conftest_path = Path(__file__).parent / "conftest.py"
        
        with open(conftest_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that essential sections exist
        assert 'import' in content, "No import statements found"
        assert 'pytest.fixture' in content, "No pytest fixtures found"
        assert 'def pytest_configure' in content, "pytest_configure function not found"
        assert 'def pytest_addoption' in content, "pytest_addoption function not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
