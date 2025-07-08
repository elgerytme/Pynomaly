#!/usr/bin/env python3

import ast
from pathlib import Path


class UndefinedNameFinder(ast.NodeVisitor):
    def __init__(self):
        self.undefined_names = []
        self.defined_names = set()
        self.imported_names = set()
        self.builtin_names = set(
            [
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
                "__name__",
                "__file__",
                "__doc__",
                "__package__",
                "__spec__",
                "True",
                "False",
                "None",
            ]
        )

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
        # Add function parameters
        for arg in node.args.args:
            self.defined_names.add(arg.arg)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.defined_names.add(node.name)
        # Add function parameters
        for arg in node.args.args:
            self.defined_names.add(arg.arg)
        self.generic_visit(node)

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
        if isinstance(node.target, ast.Name):
            self.defined_names.add(node.target.id)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                self.defined_names.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if (
                node.id not in self.defined_names
                and node.id not in self.imported_names
                and node.id not in self.builtin_names
            ):
                self.undefined_names.append((node.id, node.lineno, node.col_offset))
        self.generic_visit(node)


def find_undefined_names_in_file(file_path: Path) -> list[tuple[str, int, int]]:
    """Find undefined names in a Python file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        finder = UndefinedNameFinder()
        finder.visit(tree)
        return finder.undefined_names
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def main():
    src_dir = Path("src/pynomaly")

    if not src_dir.exists():
        print(f"Source directory {src_dir} not found!")
        return

    all_undefined = []

    for py_file in src_dir.rglob("*.py"):
        undefined_names = find_undefined_names_in_file(py_file)
        if undefined_names:
            print(f"\n{py_file}:")
            for name, line, col in undefined_names:
                print(f"  Line {line}: {name}")
                all_undefined.append((py_file, name, line, col))

    print(f"\nTotal undefined names found: {len(all_undefined)}")

    if all_undefined:
        print("\nSummary of most common undefined names:")
        from collections import Counter

        name_counts = Counter([name for _, name, _, _ in all_undefined])
        for name, count in name_counts.most_common(10):
            print(f"  {name}: {count} occurrences")


if __name__ == "__main__":
    main()
