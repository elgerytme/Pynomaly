#!/usr/bin/env python3

import ast
from pathlib import Path


class RealErrorFinder(ast.NodeVisitor):
    def __init__(self):
        self.real_errors = []
        self.defined_names = set()
        self.imported_names = set()
        self.in_comprehension = False
        self.builtin_names = {
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "bool",
            "breakpoint",
            "bytearray",
            "bytes",
            "callable",
            "chr",
            "classmethod",
            "compile",
            "complex",
            "delattr",
            "dict",
            "dir",
            "divmod",
            "enumerate",
            "eval",
            "exec",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "globals",
            "hasattr",
            "hash",
            "help",
            "hex",
            "id",
            "input",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "locals",
            "map",
            "max",
            "memoryview",
            "min",
            "next",
            "object",
            "oct",
            "open",
            "ord",
            "pow",
            "print",
            "property",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "staticmethod",
            "str",
            "sum",
            "super",
            "tuple",
            "type",
            "vars",
            "zip",
            "Exception",
            "BaseException",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "NotImplementedError",
            "ImportError",
            "KeyboardInterrupt",
            "EOFError",
            "PermissionError",
            "UnicodeDecodeError",
            "OSError",
            "MemoryError",
            "RuntimeError",
            "TimeoutError",
            "NotImplemented",
            "__name__",
            "__file__",
            "__doc__",
            "__package__",
            "__spec__",
            "True",
            "False",
            "None",
            "Ellipsis",
            "__debug__",
        }

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imported_names.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imported_names.add(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.defined_names.add(node.name)
        # Save current state
        old_defined = self.defined_names.copy()
        # Add function parameters
        for arg in node.args.args:
            self.defined_names.add(arg.arg)
        self.generic_visit(node)
        # Restore state
        self.defined_names = old_defined

    def visit_AsyncFunctionDef(self, node):
        self.defined_names.add(node.name)
        # Save current state
        old_defined = self.defined_names.copy()
        # Add function parameters
        for arg in node.args.args:
            self.defined_names.add(arg.arg)
        self.generic_visit(node)
        # Restore state
        self.defined_names = old_defined

    def visit_ClassDef(self, node):
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined_names.add(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.defined_names.add(node.target.id)
        self.generic_visit(node)

    def visit_For(self, node):
        old_defined = self.defined_names.copy()
        if isinstance(node.target, ast.Name):
            self.defined_names.add(node.target.id)
        self.generic_visit(node)
        self.defined_names = old_defined

    def visit_With(self, node):
        old_defined = self.defined_names.copy()
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                self.defined_names.add(item.optional_vars.id)
        self.generic_visit(node)
        self.defined_names = old_defined

    def visit_ExceptHandler(self, node):
        old_defined = self.defined_names.copy()
        if node.name:
            self.defined_names.add(node.name)
        self.generic_visit(node)
        self.defined_names = old_defined

    def visit_ListComp(self, node):
        # Handle list comprehensions
        old_defined = self.defined_names.copy()
        for generator in node.generators:
            if isinstance(generator.target, ast.Name):
                self.defined_names.add(generator.target.id)
        self.generic_visit(node)
        self.defined_names = old_defined

    def visit_SetComp(self, node):
        # Handle set comprehensions
        old_defined = self.defined_names.copy()
        for generator in node.generators:
            if isinstance(generator.target, ast.Name):
                self.defined_names.add(generator.target.id)
        self.generic_visit(node)
        self.defined_names = old_defined

    def visit_DictComp(self, node):
        # Handle dict comprehensions
        old_defined = self.defined_names.copy()
        for generator in node.generators:
            if isinstance(generator.target, ast.Name):
                self.defined_names.add(generator.target.id)
        self.generic_visit(node)
        self.defined_names = old_defined

    def visit_GeneratorExp(self, node):
        # Handle generator expressions
        old_defined = self.defined_names.copy()
        for generator in node.generators:
            if isinstance(generator.target, ast.Name):
                self.defined_names.add(generator.target.id)
        self.generic_visit(node)
        self.defined_names = old_defined

    def visit_Lambda(self, node):
        # Handle lambda functions
        old_defined = self.defined_names.copy()
        for arg in node.args.args:
            self.defined_names.add(arg.arg)
        self.generic_visit(node)
        self.defined_names = old_defined

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if (
                node.id not in self.defined_names
                and node.id not in self.imported_names
                and node.id not in self.builtin_names
                and node.id not in ["Any"]
            ):  # Common type hints
                # Filter out common variable names that are likely false positives
                if node.id not in [
                    "i",
                    "j",
                    "k",
                    "x",
                    "y",
                    "z",
                    "v",
                    "r",
                    "d",
                    "m",
                    "a",
                    "c",
                    "t",
                    "p",
                    "n",
                    "w",
                    "h",
                ]:
                    self.real_errors.append((node.id, node.lineno, node.col_offset))
        self.generic_visit(node)


def find_real_errors_in_file(file_path: Path) -> list:
    """Find real undefined names in a Python file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        finder = RealErrorFinder()
        finder.visit(tree)
        return finder.real_errors
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def main():
    src_dir = Path("src/anomaly_detection")

    if not src_dir.exists():
        print(f"Source directory {src_dir} not found!")
        return

    all_real_errors = []

    for py_file in src_dir.rglob("*.py"):
        real_errors = find_real_errors_in_file(py_file)
        if real_errors:
            print(f"\n{py_file}:")
            for name, line, col in real_errors:
                print(f"  Line {line}: {name}")
                all_real_errors.append((py_file, name, line, col))

    print(f"\nTotal real undefined names found: {len(all_real_errors)}")

    if all_real_errors:
        from collections import Counter

        name_counts = Counter([name for _, name, _, _ in all_real_errors])
        print("\nMost common real undefined names:")
        for name, count in name_counts.most_common(20):
            print(f"  {name}: {count} occurrences")


if __name__ == "__main__":
    main()
