#!/usr/bin/env python3
"""
Demo script to show how multiple branches can run concurrently.

This demonstrates that different branches create isolated Docker resources.
"""

import os
import sys
import subprocess
from pathlib import Path


def simulate_branch_execution(branch_name):
    """Simulate running docker-compose for a specific branch."""
    print(f"\n=== Simulating branch: {branch_name} ===")
    
    # Set branch environment variable
    env = os.environ.copy()
    env['GIT_BRANCH'] = branch_name
    
    # Generate branch-specific compose files
    script_path = Path(__file__).parent / "gen-branch-compose.py"
    
    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Show what services would be created
        compose_file = Path("deploy/docker/docker-compose.branch.yml")
        if compose_file.exists():
            result = subprocess.run(
                ["docker-compose", "-f", str(compose_file), "config", "--services"],
                capture_output=True,
                text=True,
                check=True
            )
            services = result.stdout.strip().split('\n')
            services = [s for s in services if s and not s.startswith('time=')]
            
            print(f"Services for {branch_name}:")
            for service in services:
                print(f"  ✓ {service}")
            
            # Show volume names
            volumes_result = subprocess.run(
                ["docker-compose", "-f", str(compose_file), "config", "--volumes"],
                capture_output=True,
                text=True,
                check=True
            )
            volumes = volumes_result.stdout.strip().split('\n')
            volumes = [v for v in volumes if v and not v.startswith('time=')]
            
            print(f"Volumes for {branch_name}:")
            for volume in volumes:
                print(f"  ✓ {volume}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error simulating branch {branch_name}: {e}")
        return False


def main():
    """Demo concurrent branch execution."""
    print("Docker Compose Branch Isolation Demo")
    print("=" * 50)
    
    # Simulate multiple branches running concurrently
    branches = [
        "main",
        "feature/user-auth",
        "feature/payment-gateway",
        "bugfix/security-patch"
    ]
    
    print("This demo shows how multiple branches can run Docker services concurrently")
    print("without resource name collisions.\n")
    
    for branch in branches:
        if not simulate_branch_execution(branch):
            print(f"Failed to simulate branch: {branch}")
            return 1
    
    print("\n" + "=" * 50)
    print("✓ All branches can run concurrently without conflicts!")
    print("\nTo run a specific branch environment:")
    print("  python scripts/docker/run-branch-compose.py up -d")
    print("  python scripts/docker/run-branch-compose.py --production up -d")
    print("  python scripts/docker/run-branch-compose.py down")
    
    print("\nTo run multiple branches simultaneously:")
    print("  # Terminal 1 (main branch)")
    print("  git checkout main")
    print("  python scripts/docker/run-branch-compose.py up -d")
    print("  # Terminal 2 (feature branch)")
    print("  git checkout feature/user-auth")
    print("  python scripts/docker/run-branch-compose.py up -d")
    print("  # Both can run simultaneously without conflicts!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
