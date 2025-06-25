#!/usr/bin/env python3
"""Hatch environment management utilities."""

import subprocess
import sys


def list_envs():
    """List all Hatch environments."""
    try:
        result = subprocess.run(
            ["hatch", "env", "show"], capture_output=True, text=True, check=True
        )
        print("Available environments:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error listing environments: {e}")


def create_env(env_name: str):
    """Create a new Hatch environment."""
    try:
        subprocess.run(["hatch", "env", "create", env_name], check=True)
        print(f"✅ Environment '{env_name}' created")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating environment: {e}")


def remove_env(env_name: str):
    """Remove a Hatch environment."""
    try:
        subprocess.run(["hatch", "env", "remove", env_name], check=True)
        print(f"✅ Environment '{env_name}' removed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error removing environment: {e}")


def run_in_env(env_name: str, command: str):
    """Run a command in a specific environment."""
    try:
        subprocess.run(
            ["hatch", "env", "run", "-e", env_name] + command.split(), check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command in {env_name}: {e}")


def main():
    if len(sys.argv) == 1:
        list_envs()
    elif sys.argv[1] == "create" and len(sys.argv) == 3:
        create_env(sys.argv[2])
    elif sys.argv[1] == "remove" and len(sys.argv) == 3:
        remove_env(sys.argv[2])
    elif sys.argv[1] == "run" and len(sys.argv) >= 4:
        run_in_env(sys.argv[2], " ".join(sys.argv[3:]))
    else:
        print("Usage:")
        print("  python scripts/hatch_env.py                    # List environments")
        print("  python scripts/hatch_env.py create <name>      # Create environment")
        print("  python scripts/hatch_env.py remove <name>      # Remove environment")
        print("  python scripts/hatch_env.py run <env> <cmd>    # Run command in env")


if __name__ == "__main__":
    main()
