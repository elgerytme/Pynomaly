#!/usr/bin/env python3
"""
Quick Buck2 System Test
Simple validation that our Buck2 scripts work independently.
"""

import subprocess
import sys
from pathlib import Path

def test_script_help(script_name):
    """Test that a script can show help without errors."""
    try:
        result = subprocess.run([
            "python3", f"scripts/{script_name}", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        return {
            "script": script_name,
            "status": "passed" if result.returncode == 0 else "failed",
            "has_help": "usage:" in result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }
    except Exception as e:
        return {
            "script": script_name,
            "status": "error",
            "error": str(e)
        }

def main():
    """Test all Buck2 scripts."""
    scripts = [
        "buck2_change_detector.py",
        "buck2_incremental_test.py", 
        "buck2_git_integration.py",
        "buck2_impact_analyzer.py",
        "buck2_workflow.py"
    ]
    
    print("=== Quick Buck2 System Test ===")
    
    results = []
    for script in scripts:
        result = test_script_help(script)
        results.append(result)
        
        status_symbol = {"passed": "✓", "failed": "✗", "error": "!"}[result["status"]]
        print(f"{status_symbol} {script}: {result['status']}")
        
        if result["status"] != "passed":
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Summary
    passed = sum(1 for r in results if r["status"] == "passed")
    total = len(results)
    
    print(f"\nSummary: {passed}/{total} scripts working")
    
    if passed == total:
        print("✓ All Buck2 scripts are functional!")
        return 0
    else:
        print("✗ Some Buck2 scripts have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())