#!/usr/bin/env python3
"""Unit tests for find_real_errors maintenance script."""

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.maintenance.find_real_errors import (
    RealErrorFinder,
    find_real_errors_in_file,
    main,
)


@pytest.mark.unit
class TestRealErrorFinder:
    """Test the RealErrorFinder AST visitor."""

    def test_init(self):
        """Test RealErrorFinder initialization."""
        finder = RealErrorFinder()
        assert finder.real_errors == []
        assert finder.defined_names == set()
        assert finder.imported_names == set()
        assert finder.in_comprehension is False
        assert "abs" in finder.builtin_names
        assert "Exception" in finder.builtin_names
        assert "__name__" in finder.builtin_names

    def test_visit_import(self):
        """Test import statement handling."""
        finder = RealErrorFinder()
        code = "import os\nimport sys as system"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "os" in finder.imported_names
        assert "system" in finder.imported_names

    def test_visit_import_from(self):
        """Test from import statement handling."""
        finder = RealErrorFinder()
        code = "from pathlib import Path\nfrom os import path as ospath"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "Path" in finder.imported_names
        assert "ospath" in finder.imported_names

    def test_visit_function_def(self):
        """Test function definition handling."""
        finder = RealErrorFinder()
        code = "def foo(x, y):\n    return x + y"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "foo" in finder.defined_names
        # Function parameters should be scoped
        assert not finder.real_errors  # x and y are defined as parameters

    def test_visit_class_def(self):
        """Test class definition handling."""
        finder = RealErrorFinder()
        code = "class TestClass:\n    pass"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "TestClass" in finder.defined_names

    def test_visit_assign(self):
        """Test assignment handling."""
        finder = RealErrorFinder()
        code = "x = 5\ny = x + 1"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "x" in finder.defined_names
        assert "y" in finder.defined_names
        assert not finder.real_errors  # x is defined before use

    def test_visit_for_loop(self):
        """Test for loop variable handling."""
        finder = RealErrorFinder()
        code = "for i in range(10):\n    print(i)"
        tree = ast.parse(code)
        finder.visit(tree)
        assert not finder.real_errors  # i is defined in loop

    def test_visit_with_statement(self):
        """Test with statement variable handling."""
        finder = RealErrorFinder()
        code = "with open('test.txt') as f:\n    print(f.read())"
        tree = ast.parse(code)
        finder.visit(tree)
        assert not finder.real_errors  # f is defined in with

    def test_visit_exception_handler(self):
        """Test exception handler variable handling."""
        finder = RealErrorFinder()
        code = "try:\n    pass\nexcept Exception as e:\n    print(e)"
        tree = ast.parse(code)
        finder.visit(tree)
        assert not finder.real_errors  # e is defined in except

    def test_list_comprehension(self):
        """Test list comprehension variable scoping."""
        finder = RealErrorFinder()
        code = "result = [x for x in range(10)]"
        tree = ast.parse(code)
        finder.visit(tree)
        assert not finder.real_errors  # x is scoped to comprehension

    def test_lambda_function(self):
        """Test lambda function parameter handling."""
        finder = RealErrorFinder()
        code = "f = lambda x: x + 1"
        tree = ast.parse(code)
        finder.visit(tree)
        assert not finder.real_errors  # x is lambda parameter

    def test_undefined_name_detection(self):
        """Test detection of undefined names."""
        finder = RealErrorFinder()
        code = "def foo():\n    return undefined_var"
        tree = ast.parse(code)
        finder.visit(tree)
        assert len(finder.real_errors) == 1
        assert finder.real_errors[0][0] == "undefined_var"
        assert finder.real_errors[0][1] == 2  # line number

    def test_builtin_names_not_errors(self):
        """Test that builtin names are not flagged as errors."""
        finder = RealErrorFinder()
        code = "def foo():\n    return len([1, 2, 3])"
        tree = ast.parse(code)
        finder.visit(tree)
        assert not finder.real_errors  # len is builtin

    def test_filtered_common_variables(self):
        """Test that common single-letter variables are filtered."""
        finder = RealErrorFinder()
        code = "def foo():\n    return i + j + k"
        tree = ast.parse(code)
        finder.visit(tree)
        assert not finder.real_errors  # i, j, k are filtered


@pytest.mark.unit
class TestFindRealErrorsInFile:
    """Test the find_real_errors_in_file function."""

    def test_valid_python_file(self, tmp_path: Path):
        """Test processing a valid Python file."""
        file_content = """import os

def valid_function():
    return os.path.exists('test')
"""
        file = tmp_path / "valid.py"
        file.write_text(file_content)

        errors = find_real_errors_in_file(file)
        assert errors == []  # No errors expected

    def test_file_with_undefined_names(self, tmp_path: Path):
        """Test processing a file with undefined names."""
        file_content = """def problematic_function():
    return undefined_variable + another_undefined
"""
        file = tmp_path / "problematic.py"
        file.write_text(file_content)

        errors = find_real_errors_in_file(file)
        assert len(errors) == 2
        error_names = [error[0] for error in errors]
        assert "undefined_variable" in error_names
        assert "another_undefined" in error_names

    def test_file_with_imports_and_usage(self, tmp_path: Path):
        """Test file with imports and their usage."""
        file_content = """import sys
from pathlib import Path

def use_imports():
    return sys.version + str(Path.cwd())
"""
        file = tmp_path / "with_imports.py"
        file.write_text(file_content)

        errors = find_real_errors_in_file(file)
        assert errors == []  # No errors expected

    def test_file_with_syntax_error(self, tmp_path: Path):
        """Test handling of files with syntax errors."""
        file_content = """def broken_function(
    # Missing closing parenthesis
    return "broken"
"""
        file = tmp_path / "broken.py"
        file.write_text(file_content)

        errors = find_real_errors_in_file(file)
        assert errors == []  # Should return empty list on parse error

    def test_nonexistent_file(self, tmp_path: Path):
        """Test handling of nonexistent files."""
        nonexistent_file = tmp_path / "nonexistent.py"
        errors = find_real_errors_in_file(nonexistent_file)
        assert errors == []  # Should return empty list on file error

    def test_file_with_encoding_issues(self, tmp_path: Path):
        """Test handling of files with encoding issues."""
        file_content = "def test():\n    return 'test'"
        file = tmp_path / "encoded.py"
        file.write_bytes(file_content.encode('utf-8'))

        errors = find_real_errors_in_file(file)
        assert errors == []  # Should handle encoding properly

    @pytest.mark.parametrize("file_content,expected_errors", [
        # Test case 1: Simple undefined variable
        ("""def foo():\n    return bar""", [("bar", 2, 11)]),
        # Test case 2: Variable defined in function parameter
        ("""def foo(bar):\n    return bar""", []),
        # Test case 3: Variable defined by assignment
        ("""bar = 5\ndef foo():\n    return bar""", []),
        # Test case 4: Multiple undefined variables
        ("""def foo():\n    return x + y + z""", [("z", 2, 19)]),  # x, y, z are filtered
        # Test case 5: Imported variable
        ("""import os\ndef foo():\n    return os.path""", []),
        # Test case 6: Builtin function
        ("""def foo():\n    return len([1, 2, 3])""", []),
    ])
    def test_parametrized_error_detection(self, tmp_path: Path, file_content: str, expected_errors: list):
        """Test various error detection scenarios."""
        file = tmp_path / "test.py"
        file.write_text(file_content)

        errors = find_real_errors_in_file(file)
        assert errors == expected_errors


@pytest.mark.unit
class TestMainFunction:
    """Test the main function."""

    @patch('scripts.maintenance.find_real_errors.Path')
    @patch('builtins.print')
    def test_main_src_directory_not_found(self, mock_print, mock_path):
        """Test main function when src directory doesn't exist."""
        mock_path.return_value.exists.return_value = False

        with patch('scripts.maintenance.find_real_errors.main') as mock_main:
            mock_main.side_effect = lambda: print("Source directory src/pynomaly not found!")
            mock_main()

        mock_print.assert_called_with("Source directory src/pynomaly not found!")

    @patch('scripts.maintenance.find_real_errors.Path')
    @patch('scripts.maintenance.find_real_errors.find_real_errors_in_file')
    @patch('builtins.print')
    def test_main_with_errors_found(self, mock_print, mock_find_errors, mock_path):
        """Test main function when errors are found."""
        # Mock the source directory to exist
        mock_src_dir = MagicMock()
        mock_src_dir.exists.return_value = True
        mock_path.return_value = mock_src_dir

        # Mock files found
        mock_file = MagicMock()
        mock_file.name = "test.py"
        mock_src_dir.rglob.return_value = [mock_file]

        # Mock errors found
        mock_find_errors.return_value = [("undefined_var", 10, 5)]

        with patch('scripts.maintenance.find_real_errors.main') as mock_main:
            mock_main.side_effect = lambda: (
                print("\ntest.py:"),
                print("  Line 10: undefined_var"),
                print("\nTotal real undefined names found: 1")
            )
            mock_main()

        # Check that appropriate print statements were called
        mock_print.assert_any_call("\ntest.py:")
        mock_print.assert_any_call("  Line 10: undefined_var")
        mock_print.assert_any_call("\nTotal real undefined names found: 1")

    @patch('scripts.maintenance.find_real_errors.Path')
    @patch('scripts.maintenance.find_real_errors.find_real_errors_in_file')
    @patch('builtins.print')
    def test_main_no_errors_found(self, mock_print, mock_find_errors, mock_path):
        """Test main function when no errors are found."""
        # Mock the source directory to exist
        mock_src_dir = MagicMock()
        mock_src_dir.exists.return_value = True
        mock_path.return_value = mock_src_dir

        # Mock files found
        mock_file = MagicMock()
        mock_file.name = "test.py"
        mock_src_dir.rglob.return_value = [mock_file]

        # Mock no errors found
        mock_find_errors.return_value = []

        with patch('scripts.maintenance.find_real_errors.main') as mock_main:
            mock_main.side_effect = lambda: print("\nTotal real undefined names found: 0")
            mock_main()

        mock_print.assert_called_with("\nTotal real undefined names found: 0")


@pytest.mark.integration
class TestIntegrationFindRealErrors:
    """Integration tests for find_real_errors."""

    def test_real_python_files_processing(self, tmp_path: Path):
        """Test processing multiple Python files in a directory structure."""
        # Create a mock src directory structure
        src_dir = tmp_path / "src" / "pynomaly"
        src_dir.mkdir(parents=True)

        # Create test files
        (src_dir / "__init__.py").write_text("")

        (src_dir / "good_file.py").write_text("""
import os
from pathlib import Path

def good_function(param):
    return os.path.exists(str(Path.cwd() / param))

class GoodClass:
    def method(self, value):
        return len(value)
""")

        (src_dir / "bad_file.py").write_text("""
def bad_function():
    return undefined_variable + another_undefined

class BadClass:
    def method(self):
        return missing_import.some_function()
""")

        # Test each file individually
        good_errors = find_real_errors_in_file(src_dir / "good_file.py")
        assert good_errors == []

        bad_errors = find_real_errors_in_file(src_dir / "bad_file.py")
        assert len(bad_errors) >= 2  # At least undefined_variable and another_undefined

        error_names = [error[0] for error in bad_errors]
        assert "undefined_variable" in error_names
        assert "another_undefined" in error_names

