#!/usr/bin/env python3
"""Test script to validate pyproject.toml setup"""

import subprocess
import sys
from pathlib import Path


def test_setup():
    print("🔍 Testing pyproject.toml setup validation...")

    # Check if we're in an externally managed environment
    externally_managed = False

    # Test basic install in dry-run mode
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=".",
        )

        if result.returncode == 0:
            print("✅ pyproject.toml is valid for pip installation")
            return True
        elif "externally-managed-environment" in result.stderr:
            externally_managed = True
            print("⚠️  Environment is externally managed (PEP 668)")
        else:
            print("❌ pip dry-run failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

    # If externally managed, try alternative validation methods
    if externally_managed:
        print(
            "🔍 Attempting alternative validation for externally managed environment..."
        )

        # Check if pyproject.toml exists and is readable
        pyproject_path = Path("pyproject.toml")
        if not pyproject_path.exists():
            print("❌ pyproject.toml not found")
            return False

        try:
            # Try to parse the pyproject.toml
            import tomllib

            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)

            # Check essential fields
            if "project" not in pyproject_data:
                print("❌ pyproject.toml missing [project] section")
                return False

            project = pyproject_data["project"]
            required_fields = ["name", "version", "description"]

            # Handle dynamic version
            if "version" not in project and "dynamic" in project:
                if "version" in project["dynamic"]:
                    print("✅ Dynamic version configuration detected")
                    required_fields.remove("version")

            missing_fields = [
                field for field in required_fields if field not in project
            ]
            if missing_fields:
                print(f"❌ pyproject.toml missing required fields: {missing_fields}")
                return False

            # Check if dependencies are properly formatted
            if "dependencies" in project:
                deps = project["dependencies"]
                if not isinstance(deps, list):
                    print("❌ dependencies must be a list")
                    return False
                print(f"✅ Found {len(deps)} core dependencies")

            # Check if optional dependencies are properly formatted
            if "optional-dependencies" in project:
                opt_deps = project["optional-dependencies"]
                if not isinstance(opt_deps, dict):
                    print("❌ optional-dependencies must be a dict")
                    return False
                print(f"✅ Found {len(opt_deps)} optional dependency groups")

            print("✅ pyproject.toml structure validation passed")
            print(
                "💡 Note: Full pip validation skipped due to externally managed environment"
            )
            return True

        except ImportError:
            print("⚠️  tomllib not available, trying alternative parsing...")
            try:
                # Fallback to basic text parsing
                content = pyproject_path.read_text()
                if "[project]" in content and 'name = "monorepo"' in content:
                    print("✅ Basic pyproject.toml structure appears valid")
                    return True
                else:
                    print("❌ pyproject.toml appears malformed")
                    return False
            except Exception as e:
                print(f"❌ Failed to read pyproject.toml: {e}")
                return False
        except Exception as e:
            print(f"❌ Failed to parse pyproject.toml: {e}")
            return False


if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
