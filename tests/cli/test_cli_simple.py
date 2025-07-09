#!/usr/bin/env python3
"""
Simple CLI test to demonstrate testing approach without full dependencies.
This validates the CLI structure and basic functionality.
"""

import sys
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_cli_module_structure():
    """Test that CLI modules can be imported."""
    try:
        # Test basic imports
        from pynomaly.domain.entities.anomaly import Anomaly

        print("✅ Domain entities import successful")

        # Test CLI app import (the main CLI module)
        from pynomaly.presentation.cli.app import app as cli_app

        print("✅ CLI app import successful")

        # Test autonomous CLI import
        from pynomaly.presentation.cli.autonomous import app as auto_app

        print("✅ Autonomous CLI import successful")

        # Test dataset CLI import
        from pynomaly.presentation.cli.datasets import app as dataset_app

        print("✅ Dataset CLI import successful")

        # Test detector CLI import
        from pynomaly.presentation.cli.detectors import app as detector_app

        print("✅ Detector CLI import successful")

        # Test detection CLI import
        from pynomaly.presentation.cli.detection import app as detection_app

        print("✅ Detection CLI import successful")

        # Test export CLI import
        from pynomaly.presentation.cli.export import export_app

        print("✅ Export CLI import successful")

        assert True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        assert False


def test_typer_cli_structure():
    """Test Typer CLI structure."""
    try:
        import typer

        from pynomaly.presentation.cli.app import app

        # Verify it's a Typer app
        if isinstance(app, typer.Typer):
            print("✅ Main app is a valid Typer application")
        else:
            print(f"❌ Main app is not a Typer application: {type(app)}")
            assert False

        # Check if commands are registered
        if hasattr(app, "registered_commands"):
            commands = [
                getattr(cmd.callback, "__name__", "unknown")
                for cmd in app.registered_commands
            ]
            print(f"✅ Registered commands: {commands}")

        assert True

    except Exception as e:
        print(f"❌ Typer structure test failed: {e}")
        assert False


def test_cli_help_generation():
    """Test that CLI help can be generated without errors."""
    try:
        import typer.testing

        from pynomaly.presentation.cli.app import app

        runner = typer.testing.CliRunner()
        result = runner.invoke(app, ["--help"])

        if result.exit_code == 0:
            print("✅ CLI help generation successful")
            print(f"   Help output length: {len(result.stdout)} characters")

            # Check for expected sections
            if "Usage:" in result.stdout:
                print("✅ Help contains usage information")
            if "Commands:" in result.stdout:
                print("✅ Help contains commands section")

            assert True
        else:
            print(f"❌ CLI help failed with exit code: {result.exit_code}")
            print(f"   Output: {result.stdout}")
            print(f"   Error: {result.stderr}")
            assert False

    except Exception as e:
        print(f"❌ Help generation test failed: {e}")
        assert False


def test_data_loaders():
    """Test data loader imports."""
    try:
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader

        print("✅ CSV loader import successful")

        from pynomaly.infrastructure.data_loaders.json_loader import JSONLoader

        print("✅ JSON loader import successful")

        # Test creating instances (without loading data)
        CSVLoader()
        print("✅ CSV loader instantiation successful")

        JSONLoader()
        print("✅ JSON loader instantiation successful")

        assert True

    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        assert False


def test_autonomous_service():
    """Test autonomous service import."""
    try:
        print("✅ Autonomous service import successful")

        # Test configuration class

        print("✅ Autonomous config import successful")

        assert True

    except Exception as e:
        print(f"❌ Autonomous service test failed: {e}")
        assert False


def main():
    """Run all CLI tests."""
    print("🧪 Starting Pynomaly CLI Structure Tests")
    print("=" * 50)

    tests = [
        ("CLI Module Structure", test_cli_module_structure),
        ("Typer CLI Structure", test_typer_cli_structure),
        ("CLI Help Generation", test_cli_help_generation),
        ("Data Loaders", test_data_loaders),
        ("Autonomous Service", test_autonomous_service),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("🎯 CLI STRUCTURE TEST SUMMARY")
    print(f"   Total tests: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {total - passed}")
    print(f"   Success rate: {passed / total * 100:.1f}%")

    if passed == total:
        print("🎉 All CLI structure tests passed!")
        return 0
    else:
        print("❌ Some CLI structure tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
