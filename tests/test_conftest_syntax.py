#!/usr/bin/env python3
"""Test script to check conftest.py syntax issues."""

import ast
import sys
from pathlib import Path


def main():
    # Test the conftest syntax
    conftest_path = Path("tests/conftest.py")
    with open(conftest_path, encoding="utf-8") as f:
        content = f.read()

    try:
        ast.parse(content)
        print("✓ conftest.py syntax is valid")
    except SyntaxError as e:
        print(f"✗ Syntax error in conftest.py: {e}")
        return 1

    # Test future imports placement
    lines = content.split("\n")
    future_import_line = None
    for i, line in enumerate(lines):
        if "from __future__ import" in line:
            future_import_line = i
            break

    if future_import_line is not None:
        print(f"✓ Found __future__ import at line {future_import_line + 1}")

        # Check for problematic imports before __future__
        problems = []
        for i in range(future_import_line):
            line = lines[i].strip()
            if (
                line
                and not line.startswith("#")
                and not line.startswith('"""')
                and not line.startswith("'''")
            ):
                if line.startswith("import ") or (
                    line.startswith("from ") and not line.startswith("from __future__")
                ):
                    problems.append(f"Line {i+1}: {line}")

        if problems:
            print("✗ Found imports before __future__ import:")
            for problem in problems:
                print(f"  {problem}")
            return 1
        else:
            print("✓ __future__ imports are properly placed")
    else:
        print("✓ No __future__ imports found (acceptable)")

    # Test pytest option consistency
    print("✓ Testing pytest option consistency...")

    # Extract options from pytest_addoption
    addoption_options = []
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
                                addoption_options.append(stmt.value.args[0].value)
                            elif isinstance(stmt.value.args[0], ast.Str):
                                addoption_options.append(stmt.value.args[0].s)

    print(f"✓ Found pytest options: {addoption_options}")

    # Extract options from pytest_collection_modifyitems
    modify_options = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "pytest_collection_modifyitems"
        ):
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                    if stmt.func.attr == "getoption":
                        if stmt.args and isinstance(stmt.args[0], ast.Constant):
                            modify_options.append(stmt.args[0].value)
                        elif stmt.args and isinstance(stmt.args[0], ast.Str):
                            modify_options.append(stmt.args[0].s)

    print(f"✓ Found collection modify options: {modify_options}")

    # Check consistency
    inconsistent = []
    for option in modify_options:
        if option not in addoption_options:
            inconsistent.append(option)

    if inconsistent:
        print(f"✗ Inconsistent options found: {inconsistent}")
        return 1
    else:
        print("✓ All options are consistent")

    return 0


if __name__ == "__main__":
    sys.exit(main())
