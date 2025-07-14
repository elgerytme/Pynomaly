#!/usr/bin/env python3
"""Version management utilities for Hatch."""

import subprocess
import sys


def get_version():
    """Get current version from Hatch."""
    try:
        result = subprocess.run(
            ["hatch", "version"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting version: {e}")
        return None


def set_version(version: str):
    """Set version using Hatch."""
    try:
        subprocess.run(["hatch", "version", version], check=True)
        print(f"✅ Version set to {version}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error setting version: {e}")


def main():
    if len(sys.argv) == 1:
        version = get_version()
        if version:
            print(f"Current version: {version}")
    elif len(sys.argv) == 2:
        set_version(sys.argv[1])
    else:
        print("Usage: python scripts/hatch_version.py [new_version]")


if __name__ == "__main__":
    main()
