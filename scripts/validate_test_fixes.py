#!/usr/bin/env python3
"""Validate test infrastructure fixes without requiring dependencies."""

import ast
import sys
from pathlib import Path

def validate_syntax(file_path):
    """Check if Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def validate_imports(file_path):
    """Check if imports are structurally valid."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return True, imports
    except Exception as e:
        return False, str(e)

def main():
    print("🔍 Validating Test Infrastructure Fixes")
    print("=" * 50)
    
    # Files to validate
    test_files = [
        "tests/application/test_integration_workflows.py",
        "src/pynomaly/application/dto/detector_dto.py", 
        "src/pynomaly/application/dto/experiment_dto.py",
        "src/pynomaly/application/dto/__init__.py",
        "src/pynomaly/infrastructure/config/container.py"
    ]
    
    all_valid = True
    
    for file_path in test_files:
        full_path = Path(file_path)
        if not full_path.exists():
            print(f"❌ {file_path}: File not found")
            all_valid = False
            continue
            
        # Validate syntax
        syntax_valid, syntax_error = validate_syntax(full_path)
        if syntax_valid:
            print(f"✅ {file_path}: Syntax valid")
        else:
            print(f"❌ {file_path}: Syntax error - {syntax_error}")
            all_valid = False
            continue
            
        # Validate imports structure
        imports_valid, imports_result = validate_imports(full_path)
        if imports_valid:
            print(f"   📦 {len(imports_result)} imports found")
        else:
            print(f"❌ {file_path}: Import validation failed - {imports_result}")
            all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("🚀 Test infrastructure fixes are syntactically correct")
        print("📊 Ready for test execution once dependencies are available")
    else:
        print("❌ VALIDATION FAILED")
        print("🔧 Additional fixes needed")
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())