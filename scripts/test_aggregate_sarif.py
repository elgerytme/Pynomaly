#!/usr/bin/env python3
"""
Test script for SARIF aggregation functionality.
"""

import json
import os
import tempfile
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from aggregate_sarif import aggregate_sarif


def create_test_sarif(runs_count: int = 1) -> str:
    """Create a test SARIF file with the specified number of runs."""
    sarif_data = {
        "version": "2.1.0",
        "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
        "runs": []
    }
    
    for i in range(runs_count):
        run = {
            "tool": {
                "driver": {
                    "name": f"test-tool-{i}",
                    "version": "1.0.0"
                }
            },
            "results": [
                {
                    "ruleId": f"test-rule-{i}",
                    "level": "warning",
                    "message": {"text": f"Test finding {i}"},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {
                                    "uri": "test.py"
                                }
                            }
                        }
                    ]
                }
            ]
        }
        sarif_data["runs"].append(run)
    
    return json.dumps(sarif_data, indent=2)


def test_sarif_aggregation():
    """Test the SARIF aggregation functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test SARIF files
        test_files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f"test_{i}.sarif")
            with open(file_path, 'w') as f:
                f.write(create_test_sarif(runs_count=2))
            test_files.append(file_path)
        
        # Test aggregation
        output_file = os.path.join(temp_dir, "combined_test.sarif")
        aggregate_sarif(test_files, output_file)
        
        # Verify the output
        with open(output_file, 'r') as f:
            combined_data = json.load(f)
        
        # Check that all runs were aggregated
        assert len(combined_data['runs']) == 6  # 3 files * 2 runs each
        assert combined_data['version'] == '2.1.0'
        
        print("✅ SARIF aggregation test passed!")
        print(f"  - Combined {len(test_files)} SARIF files")
        print(f"  - Total runs in combined file: {len(combined_data['runs'])}")
        
        return True


def test_missing_files():
    """Test handling of missing files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create one valid file and one missing file
        valid_file = os.path.join(temp_dir, "valid.sarif")
        missing_file = os.path.join(temp_dir, "missing.sarif")
        
        with open(valid_file, 'w') as f:
            f.write(create_test_sarif(runs_count=1))
        
        # Test aggregation with missing file
        output_file = os.path.join(temp_dir, "combined_missing_test.sarif")
        aggregate_sarif([valid_file, missing_file], output_file)
        
        # Verify only the valid file was processed
        with open(output_file, 'r') as f:
            combined_data = json.load(f)
        
        assert len(combined_data['runs']) == 1
        print("✅ Missing file handling test passed!")
        
        return True


def test_empty_runs():
    """Test handling of SARIF files with no runs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a SARIF file with no runs
        empty_file = os.path.join(temp_dir, "empty.sarif")
        with open(empty_file, 'w') as f:
            json.dump({
                "version": "2.1.0",
                "runs": []
            }, f)
        
        # Test aggregation
        output_file = os.path.join(temp_dir, "combined_empty_test.sarif")
        aggregate_sarif([empty_file], output_file)
        
        # Verify the output
        with open(output_file, 'r') as f:
            combined_data = json.load(f)
        
        assert len(combined_data['runs']) == 0
        print("✅ Empty runs handling test passed!")
        
        return True


if __name__ == "__main__":
    print("🧪 Testing SARIF aggregation functionality...")
    
    try:
        test_sarif_aggregation()
        test_missing_files()
        test_empty_runs()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
