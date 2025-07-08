#!/usr/bin/env python3
"""
Simple script to reproduce circular import issues in Pynomaly.

This script ensures everyone sees the same circular import tracebacks
and provides consistent reproduction steps.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_circular_imports():
    """Test various import scenarios that trigger circular imports."""
    print("=" * 60)
    print("PYNOMALY CIRCULAR IMPORT REPRODUCTION SCRIPT")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    print(f"Python version: {sys.version}")
    print("=" * 60)
    
    # Test scenarios that trigger circular imports
    test_cases = [
        {
            'name': 'Web App Import',
            'module': 'pynomaly.presentation.web.app',
            'function': 'create_web_app'
        },
        {
            'name': 'API App Import', 
            'module': 'pynomaly.presentation.api.app',
            'function': 'create_app'
        },
        {
            'name': 'Domain Entities Import',
            'module': 'pynomaly.domain.entities',
            'function': None
        },
        {
            'name': 'Infrastructure Config Import',
            'module': 'pynomaly.infrastructure.config',
            'function': None
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Testing: {test_case['name']}")
        print(f"Module: {test_case['module']}")
        print("-" * 40)
        
        try:
            # Import the module
            module = __import__(test_case['module'], fromlist=[''])
            print("✅ Module imported successfully")
            
            # Test function call if specified
            if test_case['function']:
                func = getattr(module, test_case['function'])
                result = func()
                print(f"✅ Function {test_case['function']}() executed successfully")
                
            results.append({
                'test': test_case['name'],
                'status': 'SUCCESS',
                'error': None
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"💀 Traceback:")
            traceback.print_exc()
            
            results.append({
                'test': test_case['name'], 
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("REPRODUCTION SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed_count = len(results) - success_count
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    
    if failed_count > 0:
        print(f"\n❌ FAILED TESTS:")
        for result in results:
            if result['status'] == 'FAILED':
                print(f"  - {result['test']}: {result['error']}")
    
    print(f"\n🔄 CIRCULAR IMPORT DETECTION:")
    print("To see detailed circular import chains, run:")
    print("  python debug_imports.py")
    print("  python debug_circular_imports.py")
    
    return failed_count == 0

def main():
    """Main function."""
    try:
        success = test_circular_imports()
        
        if success:
            print(f"\n✅ All tests passed - no import errors detected")
            print("Note: This doesn't mean no circular imports exist.")
            print("Run the debug scripts for detailed analysis.")
            sys.exit(0)
        else:
            print(f"\n❌ Some tests failed - circular import issues detected")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
