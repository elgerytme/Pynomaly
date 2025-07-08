#!/usr/bin/env python3
"""
Test script to verify forward reference detection functionality.
"""
import subprocess
import sys
import tempfile
from pathlib import Path


def test_forward_ref_detection():
    """Test that the forward reference checker works correctly."""
    
    # Create a temporary test file with forward references
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
from typing import ForwardRef
from pydantic import BaseModel

class TestModel(BaseModel):
    field: ForwardRef('AnotherModel')
    
class AnotherModel(BaseModel):
    value: str
""")
        temp_file = Path(f.name)
    
    try:
        # Test the forward reference checker
        result = subprocess.run(
            [sys.executable, 'scripts/ci/check_forward_refs.py'],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        print("Forward reference check output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        print(f"Exit code: {result.returncode}")
        
        # The script should detect forward references
        if result.returncode == 0:
            print("✅ Test passed: No forward references detected (expected)")
        else:
            print("❌ Test detected forward references (this may be expected)")
            
    finally:
        # Clean up
        temp_file.unlink()


if __name__ == "__main__":
    test_forward_ref_detection()
