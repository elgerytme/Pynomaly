#!/usr/bin/env python3
"""Simple test runner to check conftest.py syntax without full imports."""

import ast
import sys
from pathlib import Path


def test_conftest_syntax():
    """Test that conftest.py has valid Python syntax."""
    conftest_path = Path("tests/conftest.py")

    if not conftest_path.exists():
        print("ERROR: conftest.py not found")
        return False

    # Read the file content
    with open(conftest_path, encoding="utf-8") as f:
        content = f.read()

    # Parse the AST to check for syntax errors
    try:
        ast.parse(content)
        print("SUCCESS: conftest.py has valid syntax")
        return True
    except SyntaxError as e:
        print(f"ERROR: Syntax error in conftest.py: {e}")
        return False


def test_future_imports_placement():
    """Test that __future__ imports are placed correctly."""
    conftest_path = Path("tests/conftest.py")

    with open(conftest_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Find the line with __future__ import
    future_import_line = None
    for i, line in enumerate(lines):
        if "from __future__ import" in line:
            future_import_line = i
            break

    if future_import_line is None:
        print("INFO: No __future__ imports found")
        return True

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
            if not is_part_of_docstring(lines[: i + 1]):
                print(
                    f"ERROR: Non-docstring/comment line before __future__ import at line {i+1}: {line}"
                )
                return False

    print("SUCCESS: __future__ imports are correctly placed")
    return True


def is_part_of_docstring(lines):
    """Check if the current line is part of a docstring."""
    content = "".join(lines)
    try:
        ast.parse(content)
        return True  # If it parses, it's valid
    except SyntaxError:
        return False


if __name__ == "__main__":
    print("Testing conftest.py syntax and structure...")

    test1 = test_conftest_syntax()
    test2 = test_future_imports_placement()

    if test1 and test2:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
