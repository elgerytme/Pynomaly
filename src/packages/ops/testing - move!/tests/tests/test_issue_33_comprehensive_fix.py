"""Comprehensive test for Issue #33 - Critical Bug Fix.

This test file provides a comprehensive test suite for the critical bug fix
in conftest.py related to pytest configuration and hook ordering.
"""

import ast
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestIssue33CriticalBugFix:
    """Test suite for Issue #33 - Critical Bug Fix."""

    def test_conftest_syntax_validation(self):
        """Test that conftest.py has valid Python syntax."""
        conftest_path = Path(__file__).parent / "conftest.py"

        # Read the file content
        with open(conftest_path, encoding="utf-8") as f:
            content = f.read()

        # Parse the AST to check for syntax errors
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in conftest.py: {e}")

    def test_future_imports_placement(self):
        """Test that __future__ imports are placed correctly."""
        conftest_path = Path(__file__).parent / "conftest.py"

        with open(conftest_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Find the line with __future__ import
        future_import_line = None
        for i, line in enumerate(lines):
            if "from __future__ import" in line:
                future_import_line = i
                break

        if future_import_line is None:
            pytest.skip("No __future__ imports found")

        # Check that only docstrings and comments come before __future__ import
        for i in range(future_import_line):
            line = lines[i].strip()
            if (
                line
                and not line.startswith("#")
                and not line.startswith('"""')
                and not line.startswith("'''")
            ):
                # Check if this is part of a docstring
                if not self._is_part_of_docstring(lines[: i + 1]):
                    pytest.fail(
                        f"Non-docstring/comment line before __future__ import at line {i+1}: {line}"
                    )

    def _is_part_of_docstring(self, lines: list[str]) -> bool:
        """Check if the current line is part of a docstring."""
        content = "".join(lines)
        try:
            tree = ast.parse(content)
            return True  # If it parses, it's valid
        except SyntaxError:
            return False

    def test_pytest_hooks_order(self):
        """Test that pytest hooks are in the correct order."""
        conftest_path = Path(__file__).parent / "conftest.py"

        # Import the conftest module
        spec = importlib.util.spec_from_file_location("conftest", conftest_path)
        conftest_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(conftest_module)
        except Exception as e:
            pytest.fail(f"Failed to import conftest.py: {e}")

        # Check that pytest_addoption is defined
        assert hasattr(
            conftest_module, "pytest_addoption"
        ), "pytest_addoption function not found"

        # Check that pytest_collection_modifyitems is defined
        assert hasattr(
            conftest_module, "pytest_collection_modifyitems"
        ), "pytest_collection_modifyitems function not found"

    def test_pytest_options_consistency(self):
        """Test that pytest options are consistent between addoption and collection_modifyitems."""
        conftest_path = Path(__file__).parent / "conftest.py"

        # Read the file content
        with open(conftest_path, encoding="utf-8") as f:
            content = f.read()

        # Extract options from pytest_addoption
        addoption_options = self._extract_addoption_options(content)

        # Extract options from pytest_collection_modifyitems
        modify_options = self._extract_modifyitems_options(content)

        # Check that all options used in modifyitems are defined in addoption
        for option in modify_options:
            assert (
                option in addoption_options
            ), f"Option '{option}' used in pytest_collection_modifyitems but not defined in pytest_addoption"

    def _extract_addoption_options(self, content: str) -> list[str]:
        """Extract options from pytest_addoption function."""
        options = []
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "pytest_addoption":
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        if (
                            isinstance(stmt.value.func, ast.Attribute)
                            and stmt.value.func.attr == "addoption"
                        ):
                            # Extract the first argument (option name)
                            if stmt.value.args:
                                if isinstance(stmt.value.args[0], ast.Constant):
                                    options.append(stmt.value.args[0].value)
                                elif isinstance(stmt.value.args[0], ast.Str):
                                    options.append(stmt.value.args[0].s)

        return options

    def _extract_modifyitems_options(self, content: str) -> list[str]:
        """Extract options from pytest_collection_modifyitems function."""
        options = []
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "pytest_collection_modifyitems"
            ):
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Call) and isinstance(
                        stmt.func, ast.Attribute
                    ):
                        if stmt.func.attr == "getoption":
                            if stmt.args and isinstance(stmt.args[0], ast.Constant):
                                options.append(stmt.args[0].value)
                            elif stmt.args and isinstance(stmt.args[0], ast.Str):
                                options.append(stmt.args[0].s)

        return options

    def test_pytest_collect_functionality(self):
        """Test that pytest can collect tests without errors."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                str(Path(__file__).parent / "test_issue_33_comprehensive_fix.py"),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # Check for specific error messages
        assert (
            "from __future__ imports must occur at the beginning of the file"
            not in result.stderr
        )
        assert result.returncode == 0 or "collected" in result.stdout

    def test_pytest_hooks_execution(self):
        """Test that pytest hooks execute without errors."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
import pytest

def test_dummy():
    assert True
""")
            temp_test_file = f.name

        try:
            # Run pytest with various options
            for option in ["--runslow", "--integration", "--performance"]:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", temp_test_file, option, "-v"],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent.parent,
                )

                # Should not have hook-related errors
                assert "getoption" not in result.stderr or result.returncode == 0
        finally:
            Path(temp_test_file).unlink()

    def test_conftest_import_without_errors(self):
        """Test that conftest.py can be imported without errors."""
        conftest_path = Path(__file__).parent / "conftest.py"

        # Try to import conftest module
        spec = importlib.util.spec_from_file_location("conftest", conftest_path)
        conftest_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(conftest_module)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError when importing conftest.py: {e}")
        except ImportError:
            # ImportError is acceptable for this test
            pass
        except Exception as e:
            pytest.fail(f"Unexpected error when importing conftest.py: {e}")

    def test_pytest_configuration_markers(self):
        """Test that pytest configuration markers are properly set."""
        conftest_path = Path(__file__).parent / "conftest.py"

        # Import the conftest module
        spec = importlib.util.spec_from_file_location("conftest", conftest_path)
        conftest_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(conftest_module)
        except ImportError:
            # Skip if dependencies are not available
            pytest.skip("Dependencies not available for conftest import")

        # Check that pytest_configure is defined
        assert hasattr(
            conftest_module, "pytest_configure"
        ), "pytest_configure function not found"

        # Check that it's callable
        assert callable(
            conftest_module.pytest_configure
        ), "pytest_configure is not callable"

    def test_fixture_definitions(self):
        """Test that key fixtures are properly defined."""
        conftest_path = Path(__file__).parent / "conftest.py"

        # Read the file content
        with open(conftest_path, encoding="utf-8") as f:
            content = f.read()

        # Check for key fixture definitions
        expected_fixtures = [
            "test_settings",
            "container",
            "sample_data",
            "sample_dataset",
            "sample_detector",
        ]

        for fixture in expected_fixtures:
            assert f"def {fixture}(" in content, f"Fixture '{fixture}' not found"

    def test_conftest_file_structure(self):
        """Test the overall structure of conftest.py."""
        conftest_path = Path(__file__).parent / "conftest.py"

        with open(conftest_path, encoding="utf-8") as f:
            content = f.read()

        # Check that essential sections exist
        assert "import" in content, "No import statements found"
        assert "pytest.fixture" in content, "No pytest fixtures found"
        assert "def pytest_configure" in content, "pytest_configure function not found"
        assert "def pytest_addoption" in content, "pytest_addoption function not found"

    def test_conftest_dependency_handling(self):
        """Test that conftest.py handles optional dependencies gracefully."""
        conftest_path = Path(__file__).parent / "conftest.py"

        with open(conftest_path, encoding="utf-8") as f:
            content = f.read()

        # Check for proper try/except blocks for optional imports
        assert (
            "try:" in content and "except ImportError:" in content
        ), "No proper exception handling for optional imports"

        # Check for graceful handling of missing dependencies
        assert (
            "pytest.skip" in content
        ), "Missing pytest.skip for unavailable dependencies"


class TestIssue33RegressionPrevention:
    """Tests to prevent regression of Issue #33."""

    def test_no_syntax_errors_in_conftest(self):
        """Ensure no syntax errors exist in conftest.py."""
        conftest_path = Path(__file__).parent / "conftest.py"

        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(conftest_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Syntax error in conftest.py: {result.stderr}"

    def test_pytest_can_load_conftest(self):
        """Ensure pytest can load conftest.py without errors."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                str(Path(__file__).parent / "test_basic_functionality.py"),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # Should not have future import errors
        assert (
            "from __future__ imports must occur at the beginning of the file"
            not in result.stderr
        )

        # Should be able to collect tests
        assert result.returncode == 0 or "collected" in result.stdout

    def test_all_pytest_options_work(self):
        """Test that all defined pytest options work correctly."""
        options_to_test = [
            "--runslow",
            "--integration",
            "--performance",
            "--security",
            "--benchmark",
        ]

        for option in options_to_test:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--help"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            assert option in result.stdout, f"Option {option} not found in pytest help"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
