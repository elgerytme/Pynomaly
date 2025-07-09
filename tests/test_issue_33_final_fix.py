"""Final comprehensive test for Issue #33 - Critical Bug Fix.

This test ensures that the critical bug in conftest.py has been resolved
and validates the fix comprehensively.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest


def test_issue_33_conftest_syntax_fixed():
    """Test that conftest.py syntax is now valid."""
    conftest_path = Path(__file__).parent / "conftest.py"

    with open(conftest_path, encoding="utf-8") as f:
        content = f.read()

    # Should not raise SyntaxError
    ast.parse(content)


def test_issue_33_future_imports_correct():
    """Test that __future__ imports are properly placed."""
    conftest_path = Path(__file__).parent / "conftest.py"

    with open(conftest_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Find future import
    future_import_line = None
    for i, line in enumerate(lines):
        if "from __future__ import" in line:
            future_import_line = i
            break

    if future_import_line is None:
        pytest.skip("No __future__ imports found")

    # Verify proper placement
    for i in range(future_import_line):
        line = lines[i].strip()
        if line and not line.startswith("#"):
            # Should be docstring
            assert (
                '"""' in line or "'''" in line
            ), f"Invalid line before __future__ import: {line}"


def test_issue_33_pytest_configuration():
    """Test that pytest configuration works correctly."""
    conftest_path = Path(__file__).parent / "conftest.py"

    with open(conftest_path, encoding="utf-8") as f:
        content = f.read()

    # Check required functions exist
    assert "def pytest_configure(" in content
    assert "def pytest_addoption(" in content
    assert "def pytest_collection_modifyitems(" in content

    # Check option consistency
    assert "--runslow" in content
    assert "--integration" in content


def test_issue_33_no_syntax_errors():
    """Test that no syntax errors exist in conftest.py."""
    conftest_path = Path(__file__).parent / "conftest.py"

    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(conftest_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Syntax errors found: {result.stderr}"


def test_issue_33_regression_prevention():
    """Test that the specific issue is resolved."""
    # Try to collect this test file to verify conftest.py loads correctly
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(__file__)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    # Should not have the specific error from Issue #33
    assert (
        "from __future__ imports must occur at the beginning of the file"
        not in result.stderr
    )
    assert "SyntaxError" not in result.stderr or result.returncode == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
