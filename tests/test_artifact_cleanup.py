#!/usr/bin/env python3
"""Test script to verify artifact cleanup configuration."""

from pathlib import Path
import tempfile
import os
import sys

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "analysis"))

from analyze_project_structure import (
    load_config, 
    categorize_stray_file, 
    categorize_stray_directory,
    should_delete_item,
    matches_pattern
)

def test_config_loading():
    """Test that configuration loads correctly."""
    print("Testing configuration loading...")
    
    config = load_config(".pyno-org.yaml")
    if config:
        print("✓ Configuration loaded successfully")
        print(f"  - Found {len(config.get('delete_patterns', []))} delete patterns")
        print(f"  - Found {len(config.get('allowlist', []))} allowlist items")
        print(f"  - Found {len(config.get('allowed_root_files', []))} allowed root files")
        print(f"  - Found {len(config.get('allowed_root_directories', []))} allowed root directories")
    else:
        print("✗ Configuration loading failed")
        return False
    return True

def test_pattern_matching():
    """Test pattern matching functionality."""
    print("\nTesting pattern matching...")
    
    # Test delete patterns
    patterns = ["*.log", "dist/", "build/", "*.egg-info", ".*.swp"]
    
    test_cases = [
        ("test.log", True),
        ("dist/", True),
        ("build/", True),
        ("package.egg-info", True),
        (".main.py.swp", True),
        ("normal.py", False),
        ("README.md", False),
    ]
    
    for path, expected in test_cases:
        result = matches_pattern(path, patterns)
        if result == expected:
            print(f"✓ {path} -> {result}")
        else:
            print(f"✗ {path} -> {result} (expected {expected})")
            return False
    
    return True

def test_artifact_categorization():
    """Test artifact categorization with configuration."""
    print("\nTesting artifact categorization...")
    
    config = load_config(".pyno-org.yaml")
    if not config:
        print("✗ Cannot test without configuration")
        return False
    
    # Create temporary files for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        test_files = [
            "test.log",
            "build/",
            "dist/",
            "package.egg-info", 
            ".main.py.swp",
            "normal.py",
            "README.md"
        ]
        
        for file_name in test_files:
            if file_name.endswith('/'):
                # Create directory
                (temp_path / file_name.rstrip('/')).mkdir(exist_ok=True)
            else:
                # Create file
                (temp_path / file_name).touch()
        
        # Test categorization
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_path)
            
            for file_name in test_files:
                if file_name.endswith('/'):
                    file_path = temp_path / file_name.rstrip('/')
                    category = categorize_stray_directory(file_path, config)
                else:
                    file_path = temp_path / file_name
                    category = categorize_stray_file(file_path, config)
                
                print(f"  {file_name} -> {category}")
                
                # Check if artifacts are correctly categorized for deletion
                if file_name in ["test.log", "build/", "dist/", "package.egg-info", ".main.py.swp"]:
                    if category != "artifacts_for_deletion":
                        print(f"✗ {file_name} should be categorized as artifacts_for_deletion, got {category}")
                        return False
                    print(f"✓ {file_name} correctly categorized for deletion")
        
        finally:
            os.chdir(old_cwd)
    
    return True

def test_allowlist():
    """Test allowlist functionality."""
    print("\nTesting allowlist functionality...")
    
    config = load_config(".pyno-org.yaml")
    if not config:
        print("✗ Cannot test without configuration")
        return False
    
    # Test cases where items should NOT be deleted despite matching patterns
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files that match delete patterns but are in allowlist
        test_files = [
            "docs/build/",  # Should not be deleted due to allowlist
            "examples/logs/",  # Should not be deleted due to allowlist
            "logging.yaml",  # Should not be deleted due to allowlist
        ]
        
        for file_name in test_files:
            if file_name.endswith('/'):
                # Create directory
                (temp_path / file_name.rstrip('/')).mkdir(parents=True, exist_ok=True)
            else:
                # Create file
                file_path = temp_path / file_name
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()
        
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_path)
            
            for file_name in test_files:
                if file_name.endswith('/'):
                    file_path = temp_path / file_name.rstrip('/')
                    should_delete = should_delete_item(file_path, config.get('delete_patterns', []), config.get('allowlist', []))
                else:
                    file_path = temp_path / file_name
                    should_delete = should_delete_item(file_path, config.get('delete_patterns', []), config.get('allowlist', []))
                
                print(f"  {file_name} -> should_delete: {should_delete}")
                
                if should_delete:
                    print(f"✗ {file_name} should be protected by allowlist")
                    return False
                else:
                    print(f"✓ {file_name} correctly protected by allowlist")
        
        finally:
            os.chdir(old_cwd)
    
    return True

def main():
    """Run all tests."""
    print("Running artifact cleanup configuration tests...")
    print("=" * 60)
    
    tests = [
        test_config_loading,
        test_pattern_matching,
        test_artifact_categorization,
        test_allowlist,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
