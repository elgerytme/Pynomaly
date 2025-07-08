#!/usr/bin/env python
"""Standalone validation test for explainable AI service import logic."""

import ast
import sys
from pathlib import Path

def validate_import_pattern(file_path: Path) -> dict:
    """Validate that a Python file has proper SHAP/LIME import patterns."""
    result = {
        'file': str(file_path),
        'shap_available_defined': False,
        'lime_available_defined': False,
        'proper_try_except': False,
        'flags_are_booleans': False,
        'graceful_fallback': False
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Look for SHAP and LIME import patterns
        shap_pattern_found = False
        lime_pattern_found = False
        
        for node in ast.walk(tree):
            # Look for try-except blocks with import statements
            if isinstance(node, ast.Try):
                for stmt in node.body:
                    if isinstance(stmt, ast.Import) or isinstance(stmt, ast.ImportFrom):
                        # Check for SHAP import
                        if hasattr(stmt, 'module') and stmt.module and 'shap' in stmt.module:
                            shap_pattern_found = True
                        elif isinstance(stmt, ast.Import):
                            for alias in stmt.names:
                                if 'shap' in alias.name:
                                    shap_pattern_found = True
                        
                        # Check for LIME import
                        if hasattr(stmt, 'module') and stmt.module and 'lime' in stmt.module:
                            lime_pattern_found = True
                        elif isinstance(stmt, ast.Import):
                            for alias in stmt.names:
                                if 'lime' in alias.name:
                                    lime_pattern_found = True
                
                # Check exception handlers
                for handler in node.handlers:
                    if isinstance(handler.type, ast.Name) and handler.type.id == 'ImportError':
                        result['proper_try_except'] = True
        
        # Check for variable assignments
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == 'SHAP_AVAILABLE':
                            result['shap_available_defined'] = True
                        elif target.id == 'LIME_AVAILABLE':
                            result['lime_available_defined'] = True
        
        # Simple content checks
        if 'SHAP_AVAILABLE = True' in content and 'SHAP_AVAILABLE = False' in content:
            result['flags_are_booleans'] = True
        if 'LIME_AVAILABLE = True' in content and 'LIME_AVAILABLE = False' in content:
            result['flags_are_booleans'] = True
        
        if 'except ImportError:' in content:
            result['graceful_fallback'] = True
            
    except Exception as e:
        result['error'] = str(e)
    
    return result

def main():
    """Main validation function."""
    print("Validating explainable AI service import patterns...")
    print("=" * 60)
    
    # Files to validate
    files_to_check = [
        Path("src/pynomaly/application/services/explainable_ai_service.py"),
        Path("src/pynomaly/domain/services/explainable_ai_service.py")
    ]
    
    all_passed = True
    
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            all_passed = False
            continue
        
        print(f"\nValidating: {file_path}")
        print("-" * 40)
        
        result = validate_import_pattern(file_path)
        
        if 'error' in result:
            print(f"✗ Error parsing file: {result['error']}")
            all_passed = False
            continue
        
        # Check individual criteria
        checks = [
            ('SHAP_AVAILABLE defined', result['shap_available_defined']),
            ('LIME_AVAILABLE defined', result['lime_available_defined']),
            ('Proper try-except blocks', result['proper_try_except']),
            ('Boolean flags used', result['flags_are_booleans']),
            ('Graceful fallback', result['graceful_fallback'])
        ]
        
        file_passed = True
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"{status} {check_name}")
            if not passed:
                file_passed = False
        
        if file_passed:
            print(f"✓ {file_path.name} passed all checks")
        else:
            print(f"✗ {file_path.name} failed some checks")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL FILES PASSED VALIDATION!")
        print("✓ SHAP_AVAILABLE and LIME_AVAILABLE flags are properly defined")
        print("✓ Graceful fallback patterns are implemented")
        return True
    else:
        print("✗ Some files failed validation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
