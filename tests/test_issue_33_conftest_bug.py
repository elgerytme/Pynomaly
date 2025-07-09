<<<<<<< HEAD
"""Test for Issue #33 - Critical bug fix in conftest.py

This test verifies that the SyntaxError caused by `from __future__ import annotations`
not being at the beginning of the file has been fixed.
"""

import sys
from pathlib import Path

import pytest


def test_conftest_import_syntax_fixed():
    """Test that verifies the syntax error in conftest.py has been fixed.

    This test ensures that `from __future__ import annotations` is now at the
    beginning of the file, which is a Python requirement.
    """
    # Get the conftest.py path
    conftest_path = Path(__file__).parent / "conftest.py"

    # Read the file content
    with open(conftest_path) as f:
        content = f.read()

    # Check that the __future__ import is at the beginning
    lines = content.split('\n')

    # Find the line with __future__ import
    future_import_line = None
    for i, line in enumerate(lines):
        if 'from __future__ import annotations' in line:
            future_import_line = i
            break

    # Verify that __future__ import exists
    assert future_import_line is not None, "Could not find __future__ import"

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

    # Try to compile the file - this should now succeed
    try:
        compile(content, str(conftest_path), 'exec')
    except SyntaxError as e:
        pytest.fail(f"Compilation failed with SyntaxError: {e}")


def test_conftest_can_be_imported():
    """Test that conftest.py can be imported without syntax errors."""
    try:
        # Try to import conftest module
        import tests.conftest  # noqa: F401

        # If we get here, the import was successful
        assert True
    except SyntaxError as e:
        pytest.fail(f"conftest.py import failed with SyntaxError: {e}")
    except ImportError as e:
        # Import errors are fine for this test - we're testing syntax, not dependencies
        print(f"Import error (expected): {e}")
        assert True


def test_pytest_can_load_conftest():
    """Test that pytest can load conftest.py without syntax errors."""
    import subprocess

    # Run pytest --collect-only to test if conftest.py loads
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "--collect-only",
        "-q",  # quiet mode
        str(Path(__file__).parent / "test_issue_33_conftest_bug.py")
    ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

    # Check that pytest can at least load conftest.py
    # (even if other import errors occur)
    future_error_msg = "from __future__ imports must occur at the beginning of the file"
    assert (
        future_error_msg not in result.stderr
    ), f"Syntax error still present: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
=======
"""Test for Issue #33 - Critical bug fix in conftest.py

This test verifies that the SyntaxError caused by `from __future__ import annotations`
not being at the beginning of the file has been fixed.
"""

import sys
from pathlib import Path

import pytest


def test_conftest_import_syntax_fixed():
    """Test that verifies the syntax error in conftest.py has been fixed.

    This test ensures that `from __future__ import annotations` is now at the
    beginning of the file, which is a Python requirement.
    """
    # Get the conftest.py path
    conftest_path = Path(__file__).parent / "conftest.py"

    # Read the file content
    with open(conftest_path) as f:
        content = f.read()

    # Check that the __future__ import is at the beginning
    lines = content.split('\n')

    # Find the line with __future__ import
    future_import_line = None
    for i, line in enumerate(lines):
        if 'from __future__ import annotations' in line:
            future_import_line = i
            break

    # Verify that __future__ import exists
    assert future_import_line is not None, "Could not find __future__ import"

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

    # Try to compile the file - this should now succeed
    try:
        compile(content, str(conftest_path), 'exec')
    except SyntaxError as e:
        pytest.fail(f"Compilation failed with SyntaxError: {e}")


def test_conftest_can_be_imported():
    """Test that conftest.py can be imported without syntax errors."""
    try:
        # Try to import conftest module
        import tests.conftest  # noqa: F401
        # If we get here, the import was successful
        assert True
    except SyntaxError as e:
        pytest.fail(f"conftest.py import failed with SyntaxError: {e}")
    except ImportError as e:
        # Import errors are fine for this test - we're testing syntax, not dependencies
        print(f"Import error (expected): {e}")
        assert True


def test_pytest_can_load_conftest():
    """Test that pytest can load conftest.py without syntax errors."""
    import subprocess

    # Run pytest --collect-only to test if conftest.py loads
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "--collect-only",
        "-q",  # quiet mode
        str(Path(__file__).parent / "test_issue_33_conftest_bug.py")
    ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

    # Check that pytest can at least load conftest.py
    # (even if other import errors occur)
    future_error_msg = "from __future__ imports must occur at the beginning of the file"
    assert (
        future_error_msg not in result.stderr
    ), f"Syntax error still present: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
>>>>>>> origin/fix/33-critical-bug-fix
