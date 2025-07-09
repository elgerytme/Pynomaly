#!/usr/bin/env python3
"""Unit tests for find_undefined_names maintenance script."""

import pytest
from pathlib import Path
from scripts.maintenance.find_undefined_names import (
    find_undefined_names_in_file,
    UndefinedNameFinder,
    main,
)
import ast
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestUndefinedNameFinder:
    """Test the UndefinedNameFinder AST visitor."""
    
    def test_init(self):
        """Test UndefinedNameFinder initialization."""
        finder = UndefinedNameFinder()
        assert finder.undefined_names == []
        assert finder.defined_names == set()
        assert finder.imported_names == set()
        assert "abs" in finder.builtin_names
        assert "Exception" in finder.builtin_names
        assert "__name__" in finder.builtin_names
    
    def test_visit_import(self):
        """Test import statement handling."""
        finder = UndefinedNameFinder()
        code = "import os\nimport sys as system"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "os" in finder.imported_names
        assert "system" in finder.imported_names
    
    def test_visit_import_from(self):
        """Test from import statement handling."""
        finder = UndefinedNameFinder()
        code = "from pathlib import Path\nfrom os import path as ospath"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "Path" in finder.imported_names
        assert "ospath" in finder.imported_names
    
    def test_visit_function_def(self):
        """Test function definition handling."""
        finder = UndefinedNameFinder()
        code = "def foo(x, y):\n    return x + y"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "foo" in finder.defined_names
        assert "x" in finder.defined_names
        assert "y" in finder.defined_names
    
    def test_visit_async_function_def(self):
        """Test async function definition handling."""
        finder = UndefinedNameFinder()
        code = "async def foo(x, y):\n    return x + y"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "foo" in finder.defined_names
        assert "x" in finder.defined_names
        assert "y" in finder.defined_names
    
    def test_visit_class_def(self):
        """Test class definition handling."""
        finder = UndefinedNameFinder()
        code = "class TestClass:\n    pass"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "TestClass" in finder.defined_names
    
    def test_visit_assign(self):
        """Test assignment handling."""
        finder = UndefinedNameFinder()
        code = "x = 5\ny = x + 1"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "x" in finder.defined_names
        assert "y" in finder.defined_names
        assert not finder.undefined_names  # x is defined before use
    
    def test_visit_ann_assign(self):
        """Test annotated assignment handling."""
        finder = UndefinedNameFinder()
        code = "x: int = 5\ny: str = str(x)"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "x" in finder.defined_names
        assert "y" in finder.defined_names
        assert not finder.undefined_names  # x is defined before use
    
    def test_visit_for_loop(self):
        """Test for loop variable handling."""
        finder = UndefinedNameFinder()
        code = "for i in range(10):\n    print(i)"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "i" in finder.defined_names
        assert not finder.undefined_names  # i is defined in loop
    
    def test_visit_with_statement(self):
        """Test with statement variable handling."""
        finder = UndefinedNameFinder()
        code = "with open('test.txt') as f:\n    print(f.read())"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "f" in finder.defined_names
        assert not finder.undefined_names  # f is defined in with
    
    def test_visit_exception_handler(self):
        """Test exception handler variable handling."""
        finder = UndefinedNameFinder()
        code = "try:\n    pass\nexcept Exception as e:\n    print(e)"
        tree = ast.parse(code)
        finder.visit(tree)
        assert "e" in finder.defined_names
        assert not finder.undefined_names  # e is defined in except
    
    def test_undefined_name_detection(self):
        """Test detection of undefined names."""
        finder = UndefinedNameFinder()
        code = "def foo():\n    return undefined_var"
        tree = ast.parse(code)
        finder.visit(tree)
        assert len(finder.undefined_names) == 1
        assert finder.undefined_names[0][0] == "undefined_var"
        assert finder.undefined_names[0][1] == 2  # line number
    
    def test_builtin_names_not_undefined(self):
        """Test that builtin names are not flagged as undefined."""
        finder = UndefinedNameFinder()
        code = "def foo():\n    return len([1, 2, 3])"
        tree = ast.parse(code)
        finder.visit(tree)
        assert not finder.undefined_names  # len is builtin
    
    def test_multiple_undefined_names(self):
        """Test detection of multiple undefined names."""
        finder = UndefinedNameFinder()
        code = "def foo():\n    return x + y + z"
        tree = ast.parse(code)
        finder.visit(tree)
        assert len(finder.undefined_names) == 3
        undefined_names = [name[0] for name in finder.undefined_names]
        assert "x" in undefined_names
        assert "y" in undefined_names
        assert "z" in undefined_names
    
    def test_scope_isolation(self):
        """Test that function parameters don't leak to global scope."""
        finder = UndefinedNameFinder()
        code = """
def foo(x):
    return x

def bar():
    return x  # This should be undefined
"""
        tree = ast.parse(code)
        finder.visit(tree)
        assert len(finder.undefined_names) == 1
        assert finder.undefined_names[0][0] == "x"
        assert finder.undefined_names[0][1] == 6  # line number in bar function


@pytest.mark.unit
class TestFindUndefinedNamesInFile:
    """Test the find_undefined_names_in_file function."""
    
    def test_valid_python_file(self, tmp_path: Path):
        """Test processing a valid Python file."""
        file_content = """import os

def valid_function():
    return os.path.exists('test')
"""
        file = tmp_path / "valid.py"
        file.write_text(file_content)
        
        undefined_names = find_undefined_names_in_file(file)
        assert undefined_names == []  # No undefined names expected
    
    def test_file_with_undefined_names(self, tmp_path: Path):
        """Test processing a file with undefined names."""
        file_content = """def problematic_function():
    return undefined_variable + another_undefined
"""
        file = tmp_path / "problematic.py"
        file.write_text(file_content)
        
        undefined_names = find_undefined_names_in_file(file)
        assert len(undefined_names) == 2
        name_list = [name[0] for name in undefined_names]
        assert "undefined_variable" in name_list
        assert "another_undefined" in name_list
    
    def test_file_with_imports_and_usage(self, tmp_path: Path):
        """Test file with imports and their usage."""
        file_content = """import sys
from pathlib import Path

def use_imports():
    return sys.version + str(Path.cwd())
"""
        file = tmp_path / "with_imports.py"
        file.write_text(file_content)
        
        undefined_names = find_undefined_names_in_file(file)
        assert undefined_names == []  # No undefined names expected
    
    def test_file_with_syntax_error(self, tmp_path: Path):
        """Test handling of files with syntax errors."""
        file_content = """def broken_function(
    # Missing closing parenthesis
    return "broken"
"""
        file = tmp_path / "broken.py"
        file.write_text(file_content)
        
        undefined_names = find_undefined_names_in_file(file)
        assert undefined_names == []  # Should return empty list on parse error
    
    def test_nonexistent_file(self, tmp_path: Path):
        """Test handling of nonexistent files."""
        nonexistent_file = tmp_path / "nonexistent.py"
        undefined_names = find_undefined_names_in_file(nonexistent_file)
        assert undefined_names == []  # Should return empty list on file error
    
    def test_file_with_encoding_issues(self, tmp_path: Path):
        """Test handling of files with encoding issues."""
        file_content = "def test():\n    return 'test'"
        file = tmp_path / "encoded.py"
        file.write_bytes(file_content.encode('utf-8'))
        
        undefined_names = find_undefined_names_in_file(file)
        assert undefined_names == []  # Should handle encoding properly
    
    @pytest.mark.parametrize("file_content,expected_count", [
        # Test case 1: Simple undefined variable
        ("""def foo():\n    return bar""", 1),
        # Test case 2: Variable defined in function parameter
        ("""def foo(bar):\n    return bar""", 0),
        # Test case 3: Variable defined by assignment
        ("""bar = 5\ndef foo():\n    return bar""", 0),
        # Test case 4: Multiple undefined variables
        ("""def foo():\n    return x + y + z""", 3),
        # Test case 5: Imported variable
        ("""import os\ndef foo():\n    return os.path""", 0),
        # Test case 6: Builtin function
        ("""def foo():\n    return len([1, 2, 3])""", 0),
    ])
    def test_parametrized_undefined_detection(self, tmp_path: Path, file_content: str, expected_count: int):
        """Test various undefined name detection scenarios."""
        file = tmp_path / "test.py"
        file.write_text(file_content)
        
        undefined_names = find_undefined_names_in_file(file)
        assert len(undefined_names) == expected_count


@pytest.mark.unit
class TestMainFunction:
    """Test the main function."""
    
    @patch('scripts.maintenance.find_undefined_names.Path')
    @patch('builtins.print')
    def test_main_src_directory_not_found(self, mock_print, mock_path):
        """Test main function when src directory doesn't exist."""
        mock_path.return_value.exists.return_value = False
        
        with patch('scripts.maintenance.find_undefined_names.main') as mock_main:
            mock_main.side_effect = lambda: print("Source directory src/pynomaly not found!")
            mock_main()
            
        mock_print.assert_called_with("Source directory src/pynomaly not found!")
    
    @patch('scripts.maintenance.find_undefined_names.Path')
    @patch('scripts.maintenance.find_undefined_names.find_undefined_names_in_file')
    @patch('builtins.print')
    def test_main_with_undefined_names_found(self, mock_print, mock_find_undefined, mock_path):
        """Test main function when undefined names are found."""
        # Mock the source directory to exist
        mock_src_dir = MagicMock()
        mock_src_dir.exists.return_value = True
        mock_path.return_value = mock_src_dir
        
        # Mock files found
        mock_file = MagicMock()
        mock_file.name = "test.py"
        mock_src_dir.rglob.return_value = [mock_file]
        
        # Mock undefined names found
        mock_find_undefined.return_value = [("undefined_var", 10, 5)]
        
        with patch('scripts.maintenance.find_undefined_names.main') as mock_main:
            mock_main.side_effect = lambda: (
                print("\ntest.py:"),
                print("  Line 10: undefined_var"),
                print("\nTotal undefined names found: 1")
            )
            mock_main()
        
        # Check that appropriate print statements were called
        mock_print.assert_any_call("\ntest.py:")
        mock_print.assert_any_call("  Line 10: undefined_var")
        mock_print.assert_any_call("\nTotal undefined names found: 1")
    
    @patch('scripts.maintenance.find_undefined_names.Path')
    @patch('scripts.maintenance.find_undefined_names.find_undefined_names_in_file')
    @patch('builtins.print')
    def test_main_no_undefined_names_found(self, mock_print, mock_find_undefined, mock_path):
        """Test main function when no undefined names are found."""
        # Mock the source directory to exist
        mock_src_dir = MagicMock()
        mock_src_dir.exists.return_value = True
        mock_path.return_value = mock_src_dir
        
        # Mock files found
        mock_file = MagicMock()
        mock_file.name = "test.py"
        mock_src_dir.rglob.return_value = [mock_file]
        
        # Mock no undefined names found
        mock_find_undefined.return_value = []
        
        with patch('scripts.maintenance.find_undefined_names.main') as mock_main:
            mock_main.side_effect = lambda: print("\nTotal undefined names found: 0")
            mock_main()
        
        mock_print.assert_called_with("\nTotal undefined names found: 0")


@pytest.mark.integration
class TestIntegrationFindUndefinedNames:
    """Integration tests for find_undefined_names."""
    
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
        good_undefined = find_undefined_names_in_file(src_dir / "good_file.py")
        assert good_undefined == []
        
        bad_undefined = find_undefined_names_in_file(src_dir / "bad_file.py")
        assert len(bad_undefined) >= 3  # undefined_variable, another_undefined, missing_import
        
        undefined_names = [name[0] for name in bad_undefined]
        assert "undefined_variable" in undefined_names
        assert "another_undefined" in undefined_names
        assert "missing_import" in undefined_names
    
    def test_complex_scoping_scenarios(self, tmp_path: Path):
        """Test complex variable scoping scenarios."""
        file_content = """
# Global variable
global_var = "global"

def outer_function():
    # Local variable in outer function
    outer_var = "outer"
    
    def inner_function():
        # Should access outer_var and global_var
        return outer_var + global_var + undefined_var
    
    return inner_function()

class TestClass:
    class_var = "class"
    
    def method(self):
        # Should access class_var and global_var
        return self.class_var + global_var + another_undefined
"""
        file = tmp_path / "complex.py"
        file.write_text(file_content)
        
        undefined_names = find_undefined_names_in_file(file)
        undefined_name_list = [name[0] for name in undefined_names]
        
        # Should find undefined_var and another_undefined
        assert "undefined_var" in undefined_name_list
        assert "another_undefined" in undefined_name_list
        
        # Should not find legitimately scoped variables
        assert "global_var" not in undefined_name_list
        assert "outer_var" not in undefined_name_list
        assert "class_var" not in undefined_name_list
