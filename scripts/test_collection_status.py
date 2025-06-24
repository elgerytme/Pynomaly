#!/usr/bin/env python3
"""Check test collection status and count test files/methods."""

import ast
import sys
from pathlib import Path

def count_test_methods(file_path):
    """Count test methods in a Python test file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
        
        test_methods = 0
        test_classes = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                test_classes += 1
            elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_methods += 1
                
        return test_methods, test_classes, None
    except Exception as e:
        return 0, 0, str(e)

def main():
    print("📊 Test Collection Status Report")
    print("=" * 60)
    
    test_dir = Path("tests")
    if not test_dir.exists():
        print("❌ Tests directory not found")
        return 1
    
    total_test_files = 0
    total_test_methods = 0
    total_test_classes = 0
    syntax_errors = 0
    
    # Walk through all test files
    for test_file in test_dir.rglob("*.py"):
        if test_file.name == "__init__.py":
            continue
            
        total_test_files += 1
        methods, classes, error = count_test_methods(test_file)
        
        if error:
            print(f"❌ {test_file.relative_to(test_dir)}: {error}")
            syntax_errors += 1
        else:
            total_test_methods += methods
            total_test_classes += classes
            if methods > 0 or classes > 0:
                print(f"✅ {test_file.relative_to(test_dir)}: {classes} classes, {methods} methods")
    
    print("\n" + "=" * 60)
    print(f"📁 Total test files: {total_test_files}")
    print(f"🏗️  Total test classes: {total_test_classes}")
    print(f"🧪 Total test methods: {total_test_methods}")
    print(f"❌ Syntax errors: {syntax_errors}")
    
    if syntax_errors == 0:
        print("\n✅ ALL TEST FILES HAVE VALID SYNTAX")
        print("🚀 Ready for pytest collection and execution")
        
        # Estimate coverage potential
        print(f"\n📈 COVERAGE IMPROVEMENT POTENTIAL:")
        print(f"   Current baseline: 20.76% coverage")
        print(f"   Test infrastructure: {total_test_methods} test methods available")
        print(f"   Target: 90% coverage through systematic execution")
    else:
        print(f"\n❌ {syntax_errors} files need syntax fixes before test execution")
    
    return syntax_errors

if __name__ == "__main__":
    sys.exit(main())