#!/usr/bin/env python3
"""
Helper script to run branch-specific docker-compose commands.

This script generates the branch-specific compose files and provides easy commands
to manage Docker services for the current branch.
"""

import os
import sys
import subprocess
from pathlib import Path


def get_current_branch():
    """Get the current git branch."""
    try:
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def generate_branch_compose():
    """Generate branch-specific compose files."""
    script_path = Path(__file__).parent / "gen-branch-compose.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def run_compose_command(compose_file, command_args):
    """Run a docker-compose command with the specified file."""
    cmd = ['docker-compose', '-f', compose_file] + command_args
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Exit code: {e.returncode}")
        return False


def main():
    """Main function to handle branch-specific docker-compose operations."""
    if len(sys.argv) < 2:
        print("Usage: python run-branch-compose.py <compose-command> [args...]")
        print("")
        print("Examples:")
        print("  python run-branch-compose.py up -d")
        print("  python run-branch-compose.py down")
        print("  python run-branch-compose.py ps")
        print("  python run-branch-compose.py logs")
        print("  python run-branch-compose.py exec pynomaly bash")
        print("")
        print("Available compose files:")
        print("  - docker-compose.branch.yml (basic)")
        print("  - docker-compose.production.branch.yml (production)")
        print("  - docker-compose.testing.branch.yml (testing)")
        print("  - docker-compose.test.branch.yml (test)")
        return 1
    
    # Get current branch
    branch = get_current_branch()
    if not branch:
        print("Error: Could not determine current git branch")
        return 1
    
    print(f"Current branch: {branch}")
    
    # Generate branch-specific compose files
    print("Generating branch-specific compose files...")
    if not generate_branch_compose():
        print("Error: Failed to generate branch-specific compose files")
        return 1
    
    # Determine which compose file to use
    compose_file = "deploy/docker/docker-compose.branch.yml"
    
    # Check if user wants a specific compose file
    if len(sys.argv) > 1 and sys.argv[1] in ['--production', '--prod']:
        compose_file = "deploy/docker/docker-compose.production.branch.yml"
        command_args = sys.argv[2:]
    elif len(sys.argv) > 1 and sys.argv[1] in ['--testing', '--test']:
        compose_file = "deploy/docker/docker-compose.testing.branch.yml"
        command_args = sys.argv[2:]
    elif len(sys.argv) > 1 and sys.argv[1] in ['--unit-test']:
        compose_file = "docker-compose.test.branch.yml"
        command_args = sys.argv[2:]
    else:
        command_args = sys.argv[1:]
    
    # Check if compose file exists
    if not Path(compose_file).exists():
        print(f"Error: Compose file {compose_file} not found")
        return 1
    
    print(f"Using compose file: {compose_file}")
    print(f"Running command: docker-compose -f {compose_file} {' '.join(command_args)}")
    
    # Run the docker-compose command
    if run_compose_command(compose_file, command_args):
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
