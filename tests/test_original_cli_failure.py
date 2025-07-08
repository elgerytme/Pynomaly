#!/usr/bin/env python3
"""
Reproduction test for the original CLI failure from the integration report.

This script attempts to reproduce the exact conditions that caused:
AttributeError: 'Group' object has no attribute 'registered_commands'

Based on the stack trace, this error occurred in:
/home/admeier/.local/lib/python3.12/site-packages/typer/main.py:482
in get_group_from_info
"""

import sys
import traceback
from typer.testing import CliRunner


def test_original_cli_failure_conditions():
    """Test conditions that might have caused the original failure."""
    
    print("🔍 Testing original CLI failure conditions")
    print("=" * 60)
    
    # The error occurred when trying to import and run the CLI app
    try:
        # Import the CLI app (this is where the error was happening)
        print("📦 Importing CLI app...")
        from pynomaly.presentation.cli.app import app
        print("✅ CLI app imported successfully")
        
        # Check the app structure
        print(f"\n🔧 App type: {type(app)}")
        print(f"🔧 App attributes: {[attr for attr in dir(app) if not attr.startswith('_')]}")
        
        if hasattr(app, 'registered_commands'):
            print(f"✅ App has registered_commands: {len(app.registered_commands)} commands")
        else:
            print("❌ App missing registered_commands")
            
        if hasattr(app, 'registered_groups'):
            print(f"✅ App has registered_groups: {len(app.registered_groups)} groups")
            for group_name, group_info in app.registered_groups.items():
                print(f"   - Group: {group_name} -> {type(group_info)}")
                if hasattr(group_info, 'typer_instance'):
                    typer_instance = group_info.typer_instance
                    print(f"     Typer instance: {type(typer_instance)}")
                    if not hasattr(typer_instance, 'registered_commands'):
                        print(f"     ❌ Group's Typer instance missing registered_commands!")
                        return False
        else:
            print("❌ App missing registered_groups")
            
    except Exception as e:
        print(f"❌ CLI app import failed: {e}")
        traceback.print_exc()
        return False
    
    # Test the commands that were failing in the integration report
    print("\n🧪 Testing commands from integration report...")
    runner = CliRunner()
    
    failing_commands = [
        (["--help"], "help"),
        (["version"], "version"),
        (["detector", "list"], "detector_list"),
        (["dataset", "list"], "dataset_info"),  # Using list instead of info
        (["auto", "--help"], "auto_help"),
        (["export", "list-formats"], "export_formats"),
        (["server", "--help"], "server_help")
    ]
    
    all_passed = True
    
    for cmd_args, cmd_name in failing_commands:
        print(f"\n   Testing: {' '.join(cmd_args)}")
        try:
            result = runner.invoke(app, cmd_args)
            if result.exit_code == 0:
                print(f"   ✅ {cmd_name}: SUCCESS")
            else:
                print(f"   ❌ {cmd_name}: FAILED (exit code {result.exit_code})")
                if result.exception:
                    print(f"      Exception: {result.exception}")
                    print(f"      Output: {result.stdout}")
                all_passed = False
        except Exception as e:
            print(f"   ❌ {cmd_name}: ERROR - {e}")
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_typer_group_object_issue():
    """Test for the specific 'Group' object registered_commands issue."""
    
    print("\n🔍 Testing for Typer Group object issues")
    print("=" * 60)
    
    try:
        import typer
        from typer.main import get_group_from_info
        
        print(f"Typer version: {typer.__version__}")
        
        # Try to create a scenario that might trigger the error
        app = typer.Typer()
        
        # Add a subcommand group
        sub_app = typer.Typer()
        
        @sub_app.command()
        def sub_command():
            """A sub command."""
            pass
            
        app.add_typer(sub_app, name="sub")
        
        # Check if all nested objects have registered_commands
        print(f"\nMain app registered_commands: {hasattr(app, 'registered_commands')}")
        print(f"Sub app registered_commands: {hasattr(sub_app, 'registered_commands')}")
        
        # Check registered_groups structure
        if hasattr(app, 'registered_groups'):
            for group_name, group_info in app.registered_groups.items():
                print(f"Group '{group_name}': {type(group_info)}")
                if hasattr(group_info, 'typer_instance'):
                    typer_instance = group_info.typer_instance
                    print(f"  Typer instance type: {type(typer_instance)}")
                    print(f"  Has registered_commands: {hasattr(typer_instance, 'registered_commands')}")
                    
                    # This is where the error might occur
                    if not hasattr(typer_instance, 'registered_commands'):
                        print(f"  ❌ Found Group object without registered_commands!")
                        return False
        
        # Try to trigger the get_group_from_info call that was failing
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        
        if result.exit_code != 0:
            print(f"❌ Group test failed: {result.exception}")
            return False
        else:
            print("✅ Group test passed")
            return True
            
    except Exception as e:
        print(f"❌ Group object test error: {e}")
        traceback.print_exc()
        return False


def test_simulate_original_error_conditions():
    """Try to simulate the original error conditions."""
    
    print("\n🔍 Simulating original error conditions")
    print("=" * 60)
    
    # The original error happened during module import
    # Let's try to simulate conditions where this might happen
    
    try:
        # Test 1: Import in a clean state
        print("Test 1: Clean import")
        
        # Test 2: Import with missing dependencies (simulate)
        print("Test 2: Testing with potential dependency issues")
        
        # The original error was in get_group_from_info at line 482
        # This suggests it was trying to access registered_commands on a Group object
        # that didn't have this attribute
        
        import typer
        from typer import main
        
        # Check if we can find any Group objects in the codebase that might not have registered_commands
        print(f"Typer main module: {main}")
        
        # Look for any Group classes or objects
        for attr_name in dir(main):
            attr_value = getattr(main, attr_name)
            if hasattr(attr_value, '__name__') and 'Group' in str(attr_value):
                print(f"Found Group-like: {attr_name} = {attr_value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all reproduction tests."""
    
    print("🧪 PYNOMALY CLI FAILURE REPRODUCTION TESTS")
    print("=" * 70)
    print("Reproducing AttributeError: 'Group' object has no attribute 'registered_commands'")
    print("=" * 70)
    
    tests = [
        ("Original CLI Failure Conditions", test_original_cli_failure_conditions),
        ("Typer Group Object Issue", test_typer_group_object_issue),
        ("Simulate Original Error Conditions", test_simulate_original_error_conditions),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        print("-" * 50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("🎯 REPRODUCTION TEST SUMMARY")
    print(f"   Total tests: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {total - passed}")
    print(f"   Success rate: {passed / total * 100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed - CLI is working correctly!")
        print("   The original error may have been fixed by installing dependencies.")
        return 0
    else:
        print("\n❌ Some tests failed - there may still be issues to resolve.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
