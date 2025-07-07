#!/usr/bin/env python3
"""
Domain Layer Mutation Testing
Tests the quality of domain layer tests through mutation testing.
"""

import subprocess


def run_domain_mutations():
    """Run mutation testing on domain layer."""
    print("🧬 Running domain layer mutation testing...")

    # Target domain entities and value objects
    target_files = [
        "src/pynomaly/domain/entities/",
        "src/pynomaly/domain/value_objects/",
        "src/pynomaly/domain/services/",
    ]

    for target in target_files:
        print(f"\n🎯 Testing mutations in {target}")

        cmd = [
            "mutmut",
            "run",
            "--paths-to-mutate",
            target,
            "--tests-dir",
            "tests/domain/",
            "--runner",
            "python -m pytest tests/domain/ -x --tb=no -q",
            "--timeout",
            "120",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            print(f"✅ Mutation testing completed for {target}")
            print(f"Mutations: {result.returncode}")

            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars

        except subprocess.TimeoutExpired:
            print(f"⏰ Mutation testing timed out for {target}")
        except Exception as e:
            print(f"❌ Error testing {target}: {e}")


if __name__ == "__main__":
    run_domain_mutations()
